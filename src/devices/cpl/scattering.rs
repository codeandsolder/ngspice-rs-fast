//! Bounded, reciprocal scattering composition of a uniform distributed line.
//!
//! The reference impedance is a positive scalar; waves are V +/- Zref I,
//! with both terminal currents directed inward. Scattering retains ideal
//! voltage/current constraints even when neither an impedance nor an
//! admittance description exists. Cascading bounded scattering matrices
//! does not form the exponentially growing transfer of an attenuating line.
use super::{
    CoupledLine, invalid,
    plan::CoupledLinePlan,
    wide::{Wide, WideComplex, WideMatrix},
};
use crate::devices::MnaUnknowns;
use crate::primitives::{Complex, Real, SpiceResult};

struct Scattering {
    reflection: WideMatrix,
    transmission: WideMatrix,
}

impl Scattering {
    /// Short-segment transfer from the even and odd series of exp(M).
    /// Z,Y have infinity norm <= 1/2; (ZY)^k/(2k)! therefore converges
    /// at least as fast as the scalar series with argument 1/4.
    /// The remainder after 16 terms is below two-word roundoff.
    fn from_material(series: &WideMatrix, shunt: &WideMatrix) -> SpiceResult<Self> {
        let order = series.n;
        let product = series.mul(shunt);
        let mut forward_v = WideMatrix::identity(order);
        let mut odd = forward_v.clone();
        let mut term = forward_v.clone();
        for k in 1..=16_u32 {
            let two_k = 2 * k;
            let denominator = Wide::from_real(Real::from(two_k * (two_k - 1)));
            term = term.mul(&product).scale(Wide::ONE / denominator);
            forward_v = forward_v.add(&term);
            odd = odd.add(&term.scale(Wide::ONE / Wide::from_real(Real::from(two_k + 1))));
            if term.magnitude() <= Real::EPSILON * Real::EPSILON {
                break;
            }
        }
        let forward_i = forward_v.transpose();
        let series_block = odd.mul(series).scale(-Wide::ONE).symmetric();
        let shunt_block = shunt.mul(&odd).scale(-Wide::ONE).symmetric();
        // [b_R, a_R] = W [a_L, b_L].
        // W22 = (A+D-B-C)/2, -W21 = (C+D-A-B)/2.
        // W22 is nonsingular for positive reference impedance and
        // a passive short segment, even with a singular series block.
        let half = Wide::from_real(0.5);
        let denominator = forward_v
            .add(&forward_i)
            .sub(&series_block.add(&shunt_block))
            .scale(half);
        let numerator = shunt_block
            .add(&forward_i)
            .sub(&forward_v.add(&series_block))
            .scale(half);
        let transmission = denominator.solve(&WideMatrix::identity(order))?;
        let reflection = transmission.mul(&numerator);
        Self::checked(reflection, transmission)
    }

    fn checked(reflection: WideMatrix, transmission: WideMatrix) -> SpiceResult<Self> {
        reflection.check_finite()?;
        transmission.check_finite()?;
        // Uniform reciprocal material is invariant under reversal, and
        // reciprocity makes both N by N blocks symmetric.
        Ok(Self {
            reflection: reflection.symmetric(),
            transmission: transmission.symmetric(),
        })
    }

    fn double(&self) -> SpiceResult<Self> {
        // Internal port waves are interchanged. Eliminate them by the
        // Redheffer star product, not an inverse of the series impedance.
        let identity = WideMatrix::identity(self.reflection.n);
        let interface = identity.sub(&self.reflection.mul(&self.reflection));
        let waves = interface.solve(&self.transmission)?;
        let transmission = self.transmission.mul(&waves);
        let reflection = self
            .reflection
            .add(&self.transmission.mul(&self.reflection.mul(&waves)));
        Self::checked(reflection, transmission)
    }

    fn entry(&self, row: usize, col: usize) -> WideComplex {
        let n = self.reflection.n;
        let matrix = if row / n == col / n {
            &self.reflection
        } else {
            &self.transmission
        };
        matrix.at(row % n, col % n)
    }
}

impl CoupledLinePlan {
    fn scattering_at(&self, hz: Real) -> SpiceResult<(Scattering, Real)> {
        let (_, scale, steps) = self.bounded_operator(hz, self.length, 0.5)?;
        let exponent = -i32::try_from(steps).map_err(|_| invalid("scattering split overflow"))?;
        let n = self.conductors();
        let omega = std::f64::consts::TAU * hz;
        let mut z = WideMatrix::zero(n);
        let mut y = WideMatrix::zero(n);
        for row in 0..n {
            for col in 0..n {
                let index = row * n + col;
                let series = |value, factor| {
                    Wide::product_ratio(&[value, factor, self.length], scale, exponent)
                };
                let shunt = |value, factor| {
                    Wide::product_ratio(&[value, factor, self.length, scale], 1., exponent)
                };
                z.set(
                    row,
                    col,
                    WideComplex::new(
                        series(self.r.data()[index], 1.),
                        series(self.l.data()[index], omega),
                    ),
                );
                y.set(
                    row,
                    col,
                    WideComplex::new(
                        shunt(self.g.data()[index], 1.),
                        shunt(self.c.data()[index], omega),
                    ),
                );
            }
        }
        let mut network = Scattering::from_material(&z, &y)?;
        for _ in 0..steps {
            network = network.double()?;
        }
        Ok((network, scale))
    }

    pub(super) fn scattering_stamps(
        &self,
        line: &CoupledLine,
        hz: Real,
        unknowns: &MnaUnknowns,
        first: usize,
    ) -> SpiceResult<Vec<(usize, usize, Complex)>> {
        let (network, scale) = self.scattering_at(hz)?;
        let ports = 2 * self.conductors();
        let mut entries = line.terminal_kcl(unknowns, first)?;
        for row in 0..ports {
            for col in 0..ports {
                let s = network.entry(row, col);
                let delta = if row == col {
                    WideComplex::ONE
                } else {
                    WideComplex::ZERO
                };
                // V - Zref I = S (V + Zref I).
                // Form I±S before narrowing: ideal shunt/series modes
                // need the small complementary coefficient even when S
                // is closer to ±1 than a single binary64 ulp.
                let v = ((delta - s) * WideComplex::real(Wide::from_real(scale.recip().min(1.))))
                    .narrow();
                let i = (-(delta + s) * WideComplex::real(Wide::from_real(scale.min(1.)))).narrow();
                let (node, ret) = line.physical_pair(col);
                for (col, value) in [
                    (unknowns.checked_node_row(node)?, v),
                    (unknowns.checked_node_row(ret)?, -v),
                    (Some(first + col), i),
                ] {
                    if let Some(col) = col
                        && value != Complex::ZERO
                    {
                        entries.push((first + row, col, value));
                    }
                }
            }
        }
        if entries.iter().any(|(_, _, value)| !value.is_finite()) {
            return Err(invalid("nonfinite passive scattering stamp"));
        }
        Ok(entries)
    }
}

// Project dissipative material into the same reciprocal wave coordinates
// used by the exact LC propagator. A static stage represents the exact
// distributed R/G exponential over one section, not a lumped pi network.
pub(super) fn modal_static_scattering(
    plan: &CoupledLinePlan,
    basis: &super::lossless::ModalBasis,
    sections: usize,
) -> SpiceResult<Vec<Real>> {
    let n = plan.conductors();
    let count = Real::from(u32::try_from(sections).map_err(|_| invalid("section count"))?);
    let project = |material: &[Real], transform: &[Real]| {
        let mut left = WideMatrix::zero(n);
        let mut left_abs = WideMatrix::zero(n);
        let mut right = WideMatrix::zero(n);
        let mut right_abs = WideMatrix::zero(n);
        for row in 0..n {
            for col in 0..n {
                let value = WideComplex::real(Wide::from_real(transform[row * n + col]));
                right.set(row, col, value);
                right_abs.set(
                    row,
                    col,
                    WideComplex::real(Wide::from_real(transform[row * n + col].abs())),
                );
                let value = Wide::product_ratio(&[material[row * n + col], plan.length], count, 0);
                left.set(row, col, WideComplex::real(value));
                left_abs.set(row, col, WideComplex::real(value.abs()));
            }
        }
        let projected = right.transpose().mul(&left).mul(&right).symmetric();
        let absolute = right_abs.transpose().mul(&left_abs).mul(&right_abs);
        (projected, absolute)
    };
    let (mut series, series_absolute) = project(plan.r.data(), &basis.current_basis);
    let (mut shunt, shunt_absolute) = project(plan.g.data(), &basis.voltage_basis);
    // Discard only inter-block numerical cancellation. Keeping principal
    // blocks intact preserves the PSD loss form; selectively zeroing
    // individual entries within a connected block could make it indefinite.
    let bound = 64. * Real::EPSILON * Real::from(u32::try_from(n).unwrap_or(u32::MAX));
    let significant = |matrix: &WideMatrix, absolute: &WideMatrix, row, col| {
        matrix.at(row, col).narrow().re.abs() > bound * absolute.at(row, col).narrow().re
    };
    let mut component: Vec<_> = (0..n).collect();
    for row in 0..n {
        for col in 0..row {
            if significant(&series, &series_absolute, row, col)
                || significant(&shunt, &shunt_absolute, row, col)
            {
                let from = component[col];
                let to = component[row];
                for label in &mut component {
                    if *label == from {
                        *label = to;
                    }
                }
            }
        }
    }
    for row in 0..n {
        for col in 0..row {
            if component[row] != component[col] {
                for matrix in [&mut series, &mut shunt] {
                    matrix.set(row, col, WideComplex::ZERO);
                    matrix.set(col, row, WideComplex::ZERO);
                }
            }
        }
    }
    series.check_finite()?;
    shunt.check_finite()?;
    let mut steps = 0;
    loop {
        if series.infinity_norm().max(shunt.infinity_norm()) <= 0.5 {
            break;
        }
        if steps >= plan.squaring_limit() {
            return Err(invalid("modal loss scattering exceeds squaring budget"));
        }
        series = series.scale(Wide::from_real(0.5));
        shunt = shunt.scale(Wide::from_real(0.5));
        steps += 1;
    }
    let mut network = Scattering::from_material(&series, &shunt)?;
    for _ in 0..steps {
        network = network.double()?;
    }
    let ports = 2 * n;
    let mut coefficients = Vec::with_capacity(ports * ports);
    for row in 0..ports {
        for col in 0..ports {
            coefficients.push(network.entry(row, col).narrow().re);
        }
    }
    Ok(coefficients)
}
