//! General multiconductor telegrapher-equation transfer plan.
//!
//! Symmetric RLGC coupling is preserved as one noncommuting, 2N-dimensional
//! system. All matrices are physical per-unit-length coefficients.
use crate::maths::{Matrix, dense_complex::DenseComplex};
use crate::primitives::{Complex, Real, SpiceError, SpiceResult};

fn invalid(message: impl std::fmt::Display) -> SpiceError {
    SpiceError::Numerical {
        context: "coupled transmission line".to_owned(),
        message: message.to_string(),
    }
}

/// Matrix-memory and conductor-count limits for the coupled-line solver.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CoupledLineLimits {
    /// Highest allowed physical conductor count, excluding the returns.
    pub conductors: usize,
    /// Cap on the total dense complex exponential workspace in bytes.
    pub workspace_bytes: usize,
    /// Cap on exponent squaring iterations.
    pub squarings: usize,
}
impl Default for CoupledLineLimits {
    fn default() -> Self {
        Self {
            conductors: 32,
            workspace_bytes: 32 * 1024 * 1024,
            squarings: 128,
        }
    }
}
impl CoupledLineLimits {
    fn check(self, conductors: usize) -> SpiceResult<()> {
        if conductors == 0 || conductors > self.conductors {
            return Err(invalid("conductor count exceeds configured bounds"));
        }
        let bytes = conductors
            .checked_mul(2)
            .and_then(|v| v.checked_mul(v))
            .and_then(|v| v.checked_mul(std::mem::size_of::<Complex>()))
            .and_then(|v| v.checked_mul(16))
            .ok_or_else(|| invalid("matrix workspace allocation overflow"))?;
        if bytes > self.workspace_bytes {
            return Err(invalid("dense matrix workspace budget exceeded"));
        }
        Ok(())
    }
}

/// Positive-length, passive per-unit-length RLGC model.
#[derive(Debug, Clone)]
pub struct CoupledLinePlan {
    /// Physical length in the units used by the four per-length matrices.
    pub length: Real,
    /// Series resistance in ohms per length.
    pub r: Matrix,
    /// Series inductance in henries per length.
    pub l: Matrix,
    /// Shunt conductance in siemens per length.
    pub g: Matrix,
    /// Shunt capacitance in farads per length.
    pub c: Matrix,
    limits: CoupledLineLimits,
}
impl CoupledLinePlan {
    /// Expand C CPL model upper-triangular coefficients in row-major upper order.
    /// # Errors
    /// Wrong coefficient count, nonfinite coefficient or resource budget.
    pub fn unpack_symmetric(
        conductors: usize,
        values: &[Real],
        limits: CoupledLineLimits,
    ) -> SpiceResult<Matrix> {
        limits.check(conductors)?;
        let expected = conductors
            .checked_add(1)
            .and_then(|next| next.checked_mul(conductors))
            .map(|value| value / 2)
            .ok_or_else(|| invalid("triangular matrix dimension overflow"))?;
        if values.len() != expected || values.iter().any(|v| !v.is_finite()) {
            return Err(invalid(format!(
                "expected {expected} finite triangular values; got {}",
                values.len()
            )));
        }
        let mut result = Matrix::zeros(conductors, conductors)?;
        let mut index = 0;
        for row in 0..conductors {
            for col in row..conductors {
                result.set(row, col, values[index])?;
                result.set(col, row, values[index])?;
                index += 1;
            }
        }
        Ok(result)
    }

    /// Validate a general symmetric R/L/G/C cable with explicit length.
    ///
    /// Loss matrices R and G must be positive semidefinite; dynamic matrices
    /// L and C may be semidefinite: purely RC, RL and conductive cable
    /// limits remain physically meaningful. Singular ideal constraints are
    /// diagnosed when a finite port admittance is requested.
    /// This permits cross-coupling, arbitrary modal rotations and zero losses.
    /// # Errors
    /// Invalid dimensions, active or singular material, allocation budget.
    pub fn new(
        length: Real,
        resistance: Matrix,
        inductance: Matrix,
        conductance: Matrix,
        capacitance: Matrix,
        limits: CoupledLineLimits,
    ) -> SpiceResult<Self> {
        let conductors = resistance.rows();
        limits.check(conductors)?;
        if !length.is_finite() || length <= 0. {
            return Err(invalid("cable length must be finite and positive"));
        }
        for (name, matrix) in [
            ("R", &resistance),
            ("L", &inductance),
            ("G", &conductance),
            ("C", &capacitance),
        ] {
            if !matrix.is_square() || matrix.rows() != conductors {
                return Err(invalid(format!(
                    "{name} must be {conductors} by {conductors}"
                )));
            }
            let max = matrix
                .data()
                .iter()
                .fold(0.0_f64, |acc, &entry| acc.max(entry.abs()));
            if !max.is_finite() {
                return Err(invalid(format!("{name} contains nonfinite coefficients")));
            }
            for row in 0..conductors {
                for col in 0..row {
                    if matrix.get(row, col) != matrix.get(col, row) {
                        return Err(invalid(format!("{name} is nonsymmetric at ({row},{col})")));
                    }
                }
            }
            let values = matrix.symmetric_eigenvalues()?;
            let smallest = values
                .first()
                .copied()
                .ok_or_else(|| invalid("empty eigenspectrum"))?;
            let dimension = f64::from(
                u32::try_from(conductors).map_err(|_| invalid("dimension conversion overflow"))?,
            );
            let uncertainty = 128. * Real::EPSILON * dimension * max;
            if smallest < -uncertainty {
                return Err(invalid(format!(
                    "{name} violates passive positive semidefiniteness (minimum eigenvalue {smallest:.7e})"
                )));
            }
        }
        Ok(Self {
            length,
            r: resistance,
            l: inductance,
            g: conductance,
            c: capacitance,
            limits,
        })
    }

    /// Parse the same upper-triangle per-length parameter arrays as C CPL.
    /// # Errors
    /// Invalid array lengths, material eigenvalues, or resource limits.
    pub fn from_packed(
        conductors: usize,
        length: Real,
        r: &[Real],
        l: &[Real],
        g: &[Real],
        c: &[Real],
        limits: CoupledLineLimits,
    ) -> SpiceResult<Self> {
        Self::new(
            length,
            Self::unpack_symmetric(conductors, r, limits)?,
            Self::unpack_symmetric(conductors, l, limits)?,
            Self::unpack_symmetric(conductors, g, limits)?,
            Self::unpack_symmetric(conductors, c, limits)?,
            limits,
        )
    }

    /// Count of physical coupled signal conductors.
    #[must_use]
    pub const fn conductors(&self) -> usize {
        self.r.rows()
    }

    fn operator(&self, hz: Real) -> SpiceResult<(DenseComplex, Real)> {
        if !hz.is_finite() || hz < 0. {
            return Err(invalid("frequency must be positive or zero"));
        }
        let omega = std::f64::consts::TAU * hz;
        if !omega.is_finite() {
            return Err(invalid("angular frequency overflow"));
        }
        let n = self.conductors();
        let mut series = Vec::new();
        let mut shunt = Vec::new();
        let (mut zmax, mut ymax) = (0.0_f64, 0.0_f64);
        series
            .try_reserve(n * n)
            .map_err(|_| invalid("series matrix allocation"))?;
        shunt
            .try_reserve(n * n)
            .map_err(|_| invalid("shunt matrix allocation"))?;
        for row in 0..n {
            let (mut zr, mut yr) = (0., 0.);
            for col in 0..n {
                let idx = row * n + col;
                let z = Complex::new(self.r.data()[idx], omega * self.l.data()[idx]);
                let y = Complex::new(self.g.data()[idx], omega * self.c.data()[idx]);
                if !z.is_finite() || !y.is_finite() {
                    return Err(invalid("nonfinite RLGC impedance or admittance"));
                }
                zr += z.magnitude();
                yr += y.magnitude();
                series.push(z);
                shunt.push(y);
            }
            zmax = zmax.max(zr);
            ymax = ymax.max(yr);
        }
        // Similarity transform state I' = Zref I, balancing Z and Y blocks.
        // Crucial for physical L~nH and C~pF, not a change to the line model.
        let impedance_scale = if zmax > 0. && ymax > 0. {
            (zmax / ymax).sqrt()
        } else {
            1.
        };
        if !impedance_scale.is_finite() || impedance_scale <= 0. {
            return Err(invalid("cannot normalize coupled wave impedances"));
        }
        let mut generator = DenseComplex::zeros(2 * n)?;
        for row in 0..n {
            for col in 0..n {
                let index = row * n + col;
                let upper = -series[index] * Complex::real(self.length / impedance_scale);
                let lower = -shunt[index] * Complex::real(self.length * impedance_scale);
                if !upper.is_finite() || !lower.is_finite() {
                    return Err(invalid("telegrapher matrix overflow"));
                }
                generator.set(row, n + col, upper)?;
                generator.set(n + row, col, lower)?;
            }
        }
        Ok((generator, impedance_scale))
    }

    /// Real DC propagator for a half spatial section; same physical
    /// telegrapher generator as the exact DC/AC 2N-port.
    pub(super) fn dc_half_section(&self, sections: usize) -> SpiceResult<(DenseComplex, Real)> {
        let divisor = sections
            .checked_mul(2)
            .ok_or_else(|| invalid("half-section count overflow"))?;
        let scale = f64::from(u32::try_from(divisor).map_err(|_| invalid("half-section scale"))?);
        let (operator, impedance_scale) = self.operator(0.)?;
        Ok((
            expm_pade13(&scaled(&operator, 1. / scale)?, self.limits.squarings)?,
            impedance_scale,
        ))
    }

    /// Exact coupled first-order telegrapher transfer up to numerical precision.
    ///
    /// States are input/output vectors [V, I'] with I' = Zref·I and I
    /// oriented in the positive propagation direction.
    /// # Errors
    /// Invalid frequency, exponential range, ill-conditioned Padé solve.
    pub fn transfer_at(&self, hz: Real) -> SpiceResult<(DenseComplex, Real)> {
        let (operator, scale) = self.operator(hz)?;
        Ok((expm_pade13(&operator, self.limits.squarings)?, scale))
    }

    /// A loss-bearing line with a large telegrapher generator is represented
    /// as a Norton two-port instead of the exponentially ill-conditioned
    /// full ABCD transfer. Lossless ideal constraints remain on the exact
    /// augmented transfer path, where a finite Y may not exist.
    /// # Errors
    /// Nonfinite frequency or RLGC matrix.
    pub(super) fn requires_norton(&self, hz: Real) -> SpiceResult<bool> {
        // If every conductor has zero series Z at this frequency, the
        // physical line imposes an exact voltage constraint. It has *no*
        // finite two-ended Y, so keep the augmented branch equations even
        // for arbitrarily large shunt attenuation/current.
        if self.r.data().iter().all(|&value| value == 0.)
            && (hz == 0. || self.l.data().iter().all(|&value| value == 0.))
        {
            return Ok(false);
        }
        if !self.r.data().iter().chain(self.g.data()).any(|&v| v != 0.) {
            return Ok(false);
        }
        let (operator, _) = self.operator(hz)?;
        Ok(operator_norm(&operator)? > 4.)
    }

    fn stable_admittance(&self, hz: Real, norm: Real) -> SpiceResult<DenseComplex> {
        let mut segment = self.clone();
        let mut steps = 0;
        let mut bound = norm;
        // Renormalize the *physical length* of a segment, never the
        // coefficients. The same validated model applies to every section.
        loop {
            if bound <= 4. {
                break;
            }
            if steps >= self.limits.squarings {
                return Err(invalid(
                    "passive long-line admittance exceeds segment work budget",
                ));
            }
            segment.length *= 0.5;
            if segment.length <= 0. || !segment.length.is_finite() {
                return Err(invalid("passive long-line segment length underflows"));
            }
            bound *= 0.5;
            steps += 1;
        }
        // Each short section has an accurate, well-conditioned B inverse.
        // Doubling via a passive Norton Schur complement needs no full-line
        // exp(+gamma*length), and remains finite as Y12 tends to zero.
        let mut admittance = segment.admittance_at(hz)?;
        for _ in 0..steps {
            admittance = double_norton(&admittance)?;
        }
        Ok(admittance)
    }

    /// Physical 2N × 2N nodal admittance relating both-end *entering* currents
    /// to voltages measured from the corresponding end's return.
    ///
    /// The B transfer block is inverted. Singular B indicates an ideal
    /// voltage constraint (e.g. zero-ohm DC conductor or half-wave resonance),
    /// and is rejected rather than replaced with fictitious zero admittance.
    /// # Errors
    /// Singular transfer or nonfinite physical coupling.
    pub fn admittance_at(&self, hz: Real) -> SpiceResult<DenseComplex> {
        // If physical Z=0, Vout=Vin is an *ideal constraint* and no finite
        // admittance Y exists. A floating-point Padé roundoff B≠0 must not
        // make an invalid enormous Y appear numerically invertible.
        if self.r.data().iter().all(|&value| value == 0.)
            && (hz == 0. || self.l.data().iter().all(|&value| value == 0.))
        {
            return Err(invalid(
                "ideal zero-series line has no finite port admittance",
            ));
        }
        if self.r.data().iter().chain(self.g.data()).any(|&v| v != 0.) {
            let (operator, _) = self.operator(hz)?;
            let norm = operator_norm(&operator)?;
            if norm > 4. {
                return self.stable_admittance(hz, norm);
            }
        }
        let conductors = self.conductors();
        let (transfer, scale) = self.transfer_at(hz)?;
        let mut block_a = DenseComplex::zeros(conductors)?;
        let mut block_b = DenseComplex::zeros(conductors)?;
        let mut block_c = DenseComplex::zeros(conductors)?;
        let mut block_d = DenseComplex::zeros(conductors)?;
        for row in 0..conductors {
            for col in 0..conductors {
                block_a.set(
                    row,
                    col,
                    transfer
                        .get(row, col)
                        .ok_or_else(|| invalid("transfer block A"))?,
                )?;
                block_b.set(
                    row,
                    col,
                    transfer
                        .get(row, conductors + col)
                        .ok_or_else(|| invalid("transfer block B"))?,
                )?;
                block_c.set(
                    row,
                    col,
                    transfer
                        .get(conductors + row, col)
                        .ok_or_else(|| invalid("transfer block C"))?,
                )?;
                block_d.set(
                    row,
                    col,
                    transfer
                        .get(conductors + row, conductors + col)
                        .ok_or_else(|| invalid("transfer block D"))?,
                )?;
            }
        }
        let inverse = block_b.inverse()?.ok_or_else(|| {
            invalid("singular voltage/current transfer: ideal or resonant line constraint")
        })?;
        let inva = inverse.mul(&block_a)?;
        let dinva = block_d.mul(&inva)?;
        let dinv = block_d.mul(&inverse)?;
        let mut result = DenseComplex::zeros(2 * conductors)?;
        for row in 0..conductors {
            for col in 0..conductors {
                let entries = [
                    (row, col, -inva.get(row, col).ok_or_else(|| invalid("Y11"))?),
                    (
                        row,
                        conductors + col,
                        inverse.get(row, col).ok_or_else(|| invalid("Y12"))?,
                    ),
                    (
                        conductors + row,
                        col,
                        dinva.get(row, col).ok_or_else(|| invalid("Y21"))?
                            - block_c
                                .get(row, col)
                                .ok_or_else(|| invalid("Y21 correction"))?,
                    ),
                    (
                        conductors + row,
                        conductors + col,
                        -dinv.get(row, col).ok_or_else(|| invalid("Y22"))?,
                    ),
                ];
                for (r, block_c, value) in entries {
                    let physical = value / Complex::real(scale);
                    if !physical.is_finite() {
                        return Err(invalid("nonfinite coupled physical Y"));
                    }
                    result.set(r, block_c, physical)?;
                }
            }
        }
        Ok(result)
    }
}

/// Infinity norm of the coupled generator bounds both the exponential
/// and the passive-segment splitting decision.
fn operator_norm(generator: &DenseComplex) -> SpiceResult<Real> {
    let order = generator.dim();
    let mut norm = 0.0_f64;
    for row in generator.data().chunks_exact(order.max(1)) {
        let sum = row.iter().map(|entry| entry.magnitude()).sum::<Real>();
        norm = norm.max(sum);
    }
    if !norm.is_finite() {
        return Err(invalid("nonfinite telegrapher generator norm"));
    }
    Ok(norm)
}

/// Extract one N×N block from a physical two-ended complex admittance.
fn block(
    admittance: &DenseComplex,
    row_offset: usize,
    col_offset: usize,
) -> SpiceResult<DenseComplex> {
    let conductors = admittance.dim() / 2;
    let mut result = DenseComplex::zeros(conductors)?;
    for row in 0..conductors {
        for col in 0..conductors {
            let value = admittance
                .get(row_offset + row, col_offset + col)
                .ok_or_else(|| invalid("passive Norton block index"))?;
            result.set(row, col, value)?;
        }
    }
    Ok(result)
}

/// Two identical adjacent passive N-conductor cable sections: eliminate
/// their common interface by a complex Norton Schur complement. This remains
/// well-conditioned even when the complete line's ABCD transfer explodes.
fn double_norton(admittance: &DenseComplex) -> SpiceResult<DenseComplex> {
    let conductors = admittance.dim() / 2;
    let y11 = block(admittance, 0, 0)?;
    let y12 = block(admittance, 0, conductors)?;
    let y21 = block(admittance, conductors, 0)?;
    let y22 = block(admittance, conductors, conductors)?;
    let inverse = y22
        .add(&y11)?
        .inverse()?
        .ok_or_else(|| invalid("singular internal interface of passive coupled line"))?;
    let left = y12.mul(&inverse)?;
    let right = y21.mul(&inverse)?;
    let doubled = [
        (0, 0, y11.sub(&left.mul(&y21)?)?),
        (0, conductors, scaled(&left.mul(&y12)?, -1.)?),
        (conductors, 0, scaled(&right.mul(&y21)?, -1.)?),
        (conductors, conductors, y22.sub(&right.mul(&y12)?)?),
    ];
    let mut result = DenseComplex::zeros(2 * conductors)?;
    for (r, c, matrix) in doubled {
        for row in 0..conductors {
            for col in 0..conductors {
                result.set(
                    r + row,
                    c + col,
                    matrix
                        .get(row, col)
                        .ok_or_else(|| invalid("doubled Norton block index"))?,
                )?;
            }
        }
    }
    if result.data().iter().any(|v| !v.is_finite()) {
        return Err(invalid("nonfinite passive long-line admittance"));
    }
    Ok(result)
}

/// Padé-13 complex matrix exponential with scaling-and-squaring.
///
/// The full noncommuting 2N×2N system is treated directly. The certified
/// theta-13 bound applies to any submultiplicative norm; here the infinity
/// norm is used as a conservative, cheaply computable choice.
fn expm_pade13(input: &DenseComplex, squarings: usize) -> SpiceResult<DenseComplex> {
    const B: [Real; 14] = [
        64_764_752_532_480_000.,
        32_382_376_266_240_000.,
        7_771_770_303_897_600.,
        1_187_353_796_428_800.,
        129_060_195_264_000.,
        10_559_470_521_600.,
        670_442_572_800.,
        33_522_128_640.,
        1_323_241_920.,
        40_840_800.,
        960_960.,
        16380.,
        182.,
        1.,
    ];
    let n = input.dim();
    let mut norm = 0.0_f64;
    for row in input.data().chunks_exact(n.max(1)) {
        let value = row.iter().map(|z| z.magnitude()).sum::<Real>();
        norm = norm.max(value);
    }
    if !norm.is_finite() {
        return Err(invalid("matrix exponential has nonfinite norm"));
    }
    let mut scaled_norm = norm;
    let mut steps = 0;
    loop {
        if scaled_norm <= 5.371_920_351_148_152 {
            break;
        }
        if steps >= squarings {
            return Err(invalid(
                "propagation exponential exceeds squaring work budget",
            ));
        }
        steps += 1;
        scaled_norm *= 0.5;
    }
    let divisor =
        2.0_f64.powi(-i32::try_from(steps).map_err(|_| invalid("scaling exponent overflow"))?);
    let a = scaled(input, divisor)?;
    let a2 = a.mul(&a)?;
    let a4 = a2.mul(&a2)?;
    let a6 = a4.mul(&a2)?;
    let identity = DenseComplex::identity(n)?;
    let combine = |terms: &[(&DenseComplex, Real)]| -> SpiceResult<DenseComplex> {
        let mut out = DenseComplex::zeros(n)?;
        for &(matrix, weight) in terms {
            for row in 0..n {
                for col in 0..n {
                    let old = out
                        .get(row, col)
                        .ok_or_else(|| invalid("Padé accumulator row"))?;
                    let entry = matrix
                        .get(row, col)
                        .ok_or_else(|| invalid("Padé term row"))?;
                    out.set(row, col, old + entry * Complex::real(weight))?;
                }
            }
        }
        Ok(out)
    };
    let inner_u = a6.mul(&combine(&[(&a6, B[13]), (&a4, B[11]), (&a2, B[9])])?)?;
    let u = a.mul(&inner_u.add(&combine(&[
        (&a6, B[7]),
        (&a4, B[5]),
        (&a2, B[3]),
        (&identity, B[1]),
    ])?)?)?;
    let inner_v = a6.mul(&combine(&[(&a6, B[12]), (&a4, B[10]), (&a2, B[8])])?)?;
    let v = inner_v.add(&combine(&[
        (&a6, B[6]),
        (&a4, B[4]),
        (&a2, B[2]),
        (&identity, B[0]),
    ])?)?;
    let inverse = v
        .sub(&u)?
        .inverse()?
        .ok_or_else(|| invalid("Padé exponential denominator is singular"))?;
    let mut result = inverse.mul(&v.add(&u)?)?;
    for _ in 0..steps {
        result = result.mul(&result)?;
    }
    if result.data().iter().any(|entry| !entry.is_finite()) {
        return Err(invalid("matrix exponential overflow"));
    }
    Ok(result)
}

fn scaled(matrix: &DenseComplex, factor: Real) -> SpiceResult<DenseComplex> {
    let mut out = DenseComplex::zeros(matrix.dim())?;
    for row in 0..matrix.dim() {
        for col in 0..matrix.dim() {
            let value = matrix
                .get(row, col)
                .ok_or_else(|| invalid("scalar matrix coordinate"))?;
            out.set(row, col, value * Complex::real(factor))?;
        }
    }
    Ok(out)
}
