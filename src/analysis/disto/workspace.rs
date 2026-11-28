//! Run-local Volterra kernels and reusable solve/output storage.

use std::collections::BTreeMap;

use crate::analysis::linear::LinearOutput;
use crate::devices::distortion::{CircuitDistortion, Product};
use crate::devices::{Circuit, LinearSystem, SourceKind};
use crate::maths::complex::ComplexMatrix;
use crate::primitives::{Complex, Real, SpiceError, SpiceResult};

fn buffer(n: usize) -> SpiceResult<Vec<Complex>> {
    let mut values = Vec::new();
    values.try_reserve_exact(n).map_err(|error| {
        SpiceError::circuit(format!("distortion workspace allocation: {error}"))
    })?;
    values.resize(n, Complex::ZERO);
    Ok(values)
}

/// Both source forcings are bound once, preserving C's device order.
pub(super) struct Inputs {
    first: Vec<Complex>,
    second: Option<Vec<Complex>>,
}

impl Inputs {
    pub(super) fn bind(
        system: &LinearSystem,
        distortion: &CircuitDistortion<'_>,
        intermodulation: bool,
    ) -> SpiceResult<Self> {
        let mut sources = BTreeMap::new();
        for source in &system.sources {
            sources
                .entry(source.name.to_ascii_lowercase())
                .or_insert(source);
        }
        let n = system.a.rows();
        let mut first = buffer(n)?;
        let mut second = intermodulation.then(|| buffer(n)).transpose()?;
        let mut has_second = false;
        // Tone order is deliberately outside the source loop, as before.
        for (tone, rhs) in
            std::iter::once((false, &mut first)).chain(second.iter_mut().map(|rhs| (true, rhs)))
        {
            for (name, f1, f2) in distortion.inputs() {
                let Some(input) = (if tone { *f2 } else { *f1 }) else {
                    continue;
                };
                has_second |= tone;
                let source = sources.get(&name.to_ascii_lowercase()).ok_or_else(|| {
                    SpiceError::circuit(format!("distortion input source {name} has no AC row"))
                })?;
                let phasor = match source.kind {
                    SourceKind::Voltage => input.phasor(),
                    // C's distortion current inputs have the opposite sign
                    // of their AC/DC stamps.
                    SourceKind::Current => {
                        let p = input.phasor();
                        Complex::new(-p.re, -p.im)
                    }
                };
                for &(row, sign) in &source.rows {
                    let target = rhs.get_mut(row).ok_or_else(|| {
                        SpiceError::circuit(format!(
                            "distortion input source {name} has invalid RHS row {row}"
                        ))
                    })?;
                    let updated = *target + Complex::new(sign * phasor.re, sign * phasor.im);
                    if !updated.is_finite() {
                        return Err(SpiceError::circuit(format!(
                            "distortion input source {name} has nonfinite RHS contribution"
                        )));
                    }
                    *target = updated;
                }
            }
        }
        if intermodulation && !has_second {
            return Err(super::unsupported(
                ".disto with f2overf1 needs a source with a distof2 input (distoan.c: No \
                 source with f2 distortion input)",
            ));
        }
        Ok(Self { first, second })
    }
}

struct SolveBuffers<'a> {
    circuit: &'a Circuit,
    system: &'a LinearSystem,
    scratch: Vec<Complex>,
    conjugate: Vec<Complex>,
    rhs: Vec<Complex>,
}

impl<'a> SolveBuffers<'a> {
    fn new(circuit: &'a Circuit, system: &'a LinearSystem) -> SpiceResult<Self> {
        let n = circuit.unknown_count();
        let internal = n
            .checked_add(1)
            .ok_or_else(|| SpiceError::circuit("distortion workspace dimension overflow"))?;
        if system.a.rows() != n
            || system.a.cols() != n
            || system.e.rows() != n
            || system.e.cols() != n
        {
            return Err(SpiceError::circuit("distortion operator dimensions"));
        }
        Ok(Self {
            circuit,
            system,
            scratch: buffer(internal)?,
            conjugate: buffer(n)?,
            rhs: buffer(n)?,
        })
    }

    fn input(&mut self, omega: Real, rhs: &[Complex], out: &mut Vec<Complex>) -> SpiceResult<()> {
        solve_at(
            self.circuit,
            self.system,
            omega,
            rhs,
            &mut self.scratch,
            &mut self.conjugate,
            out,
        )
    }

    fn product(
        &mut self,
        distortion: &CircuitDistortion<'_>,
        product: Product<'_>,
        omega: Real,
        out: &mut Vec<Complex>,
    ) -> SpiceResult<()> {
        self.rhs.fill(Complex::ZERO);
        distortion.add_products(product, omega, &mut self.rhs)?;
        solve_at(
            self.circuit,
            self.system,
            omega,
            &self.rhs,
            &mut self.scratch,
            &mut self.conjugate,
            out,
        )
    }
}

/// Preserve C's factorization sequence even when the RHS is zero. Fallback
/// matrices use positive omega, with conjugation on both sides of the solve.
fn solve_at(
    circuit: &Circuit,
    system: &LinearSystem,
    omega: Real,
    rhs: &[Complex],
    scratch: &mut Vec<Complex>,
    conjugate: &mut [Complex],
    out: &mut Vec<Complex>,
) -> SpiceResult<()> {
    if rhs.len() != system.a.rows() || conjugate.len() != rhs.len() {
        return Err(SpiceError::circuit("invalid distortion RHS"));
    }
    let (factor, flip) = if let Some(pattern) = circuit.matrix_pattern()
        && let Some(matrix) = ComplexMatrix::from_operators_grouped_on(
            &pattern,
            &system.a,
            &system.e,
            &system.e_groups,
            omega,
        )? {
        (matrix.factorize()?, false)
    } else {
        let factor = ComplexMatrix::from_operators_grouped(
            &system.a,
            &system.e,
            &system.e_groups,
            omega.abs(),
        )?
        .factorize()?;
        (factor, omega < 0.)
    };
    if rhs.iter().all(|value| *value == Complex::ZERO) {
        out.resize(rhs.len(), Complex::ZERO);
        out.fill(Complex::ZERO);
        return Ok(());
    }
    let load = if flip {
        for (target, value) in conjugate.iter_mut().zip(rhs) {
            *target = value.conj();
        }
        &*conjugate
    } else {
        rhs
    };
    factor.solve_into(load, scratch, out, false)?;
    if flip {
        for value in out {
            *value = value.conj();
        }
    }
    Ok(())
}

enum Products {
    Harmonics {
        third: Vec<Complex>,
    },
    Intermodulation {
        forcing: Vec<Complex>,
        second_input: Vec<Complex>,
        sum: Vec<Complex>,
        difference: Vec<Complex>,
        third: Vec<Complex>,
    },
}

/// The two legal schedules allocate only their own solution slots. None of
/// the kernel buffers is retained in a plot, and each is overwritten before
/// being used by a dependent solve at the next frequency.
pub(super) struct Workspace<'a> {
    first_input: Vec<Complex>,
    solve: SolveBuffers<'a>,
    first: Vec<Complex>,
    second: Vec<Complex>,
    products: Products,
    row: Vec<Complex>,
}

impl<'a> Workspace<'a> {
    pub(super) fn new(
        circuit: &'a Circuit,
        system: &'a LinearSystem,
        inputs: Inputs,
        plots: &mut [LinearOutput],
        points: usize,
    ) -> SpiceResult<Self> {
        let n = circuit.unknown_count();
        let width = n
            .checked_add(1)
            .and_then(|width| width.checked_add(circuit.reported_currents().len()))
            .ok_or_else(|| SpiceError::circuit("distortion output dimension overflow"))?;
        let intermodulation = inputs.second.is_some();
        if plots.len() != if intermodulation { 3 } else { 2 }
            || plots
                .iter()
                .any(|plot| plot.header.variable_count() != width)
        {
            return Err(SpiceError::circuit("invalid distortion plot dimensions"));
        }
        for plot in plots {
            plot.header.reserve_points(points)?;
        }
        let products = if let Some(forcing) = inputs.second {
            Products::Intermodulation {
                forcing,
                second_input: buffer(n)?,
                sum: buffer(n)?,
                difference: buffer(n)?,
                third: buffer(n)?,
            }
        } else {
            Products::Harmonics { third: buffer(n)? }
        };
        Ok(Self {
            first_input: inputs.first,
            solve: SolveBuffers::new(circuit, system)?,
            first: buffer(n)?,
            second: buffer(n)?,
            products,
            row: buffer(width)?,
        })
    }

    pub(super) fn frequency(
        &mut self,
        distortion: &CircuitDistortion<'_>,
        freq: Real,
        omega1: Real,
        omega2: Real,
        plots: &mut [LinearOutput],
    ) -> SpiceResult<()> {
        let Self {
            first_input,
            solve,
            first,
            second,
            products,
            row,
        } = self;
        solve.input(omega1, first_input, first)?;
        solve.product(
            distortion,
            Product::TwoF1 { h1: first },
            2. * omega1,
            second,
        )?;
        match (products, plots) {
            (Products::Harmonics { third }, [hd2, hd3]) => {
                solve.product(
                    distortion,
                    Product::ThreeF1 {
                        h1: first,
                        h2: second,
                    },
                    3. * omega1,
                    third,
                )?;
                append(hd2, row, freq, second, 2.)?;
                append(hd3, row, freq, third, 2.)
            }
            (
                Products::Intermodulation {
                    forcing,
                    second_input,
                    sum,
                    difference,
                    third,
                },
                [plus, minus, im3],
            ) => {
                solve.input(omega2, forcing, second_input)?;
                solve.product(
                    distortion,
                    Product::F1PlusF2 {
                        h1: first,
                        h1_f2: second_input,
                    },
                    omega1 + omega2,
                    sum,
                )?;
                solve.product(
                    distortion,
                    Product::F1MinusF2 {
                        h1: first,
                        h1_f2: second_input,
                    },
                    omega1 - omega2,
                    difference,
                )?;
                let twice_omega1 = 2. * omega1;
                solve.product(
                    distortion,
                    Product::TwoF1MinusF2 {
                        h1: first,
                        h1_f2: second_input,
                        h2: second,
                        h2_minus: difference,
                    },
                    twice_omega1 - omega2,
                    third,
                )?;
                append(plus, row, freq, sum, 4.)?;
                append(minus, row, freq, difference, 4.)?;
                append(im3, row, freq, third, 6.)
            }
            _ => Err(SpiceError::circuit("distortion schedule/plot mismatch")),
        }
    }
}

fn append(
    plot: &mut LinearOutput,
    row: &mut Vec<Complex>,
    frequency: Real,
    kernel: &[Complex],
    scale: Real,
) -> SpiceResult<()> {
    if kernel.len() + plot.reports.len() + 1 != plot.header.variable_count() {
        return Err(SpiceError::circuit("distortion output kernel dimensions"));
    }
    row.clear();
    row.push(Complex::real(frequency));
    row.extend(
        kernel
            .iter()
            .map(|value| Complex::new(scale * value.re, scale * value.im)),
    );
    plot.append_reports(row);
    if row.iter().any(|value| !value.is_finite()) {
        return Err(SpiceError::Numerical {
            context: ".disto".into(),
            message: "nonfinite distortion response".into(),
        });
    }
    plot.header.push_point(row)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bits(values: &[Complex]) -> Vec<(u64, u64)> {
        values
            .iter()
            .map(|v| (v.re.to_bits(), v.im.to_bits()))
            .collect()
    }

    #[test]
    fn fallback_conjugation_and_zero_rhs_reuse_the_same_storage() {
        // No circuit pattern: force the standalone matrix path. In particular
        // negative frequency must conjugate the forcing, not merely the result.
        let circuit = Circuit::new();
        let mut system = LinearSystem::new(2);
        system.a.add(0, 0, 2.).unwrap();
        system.a.add(0, 1, -1.).unwrap();
        system.a.add(1, 1, 3.).unwrap();
        system.e.add(1, 0, 0.25).unwrap();
        let mut scratch = buffer(3).unwrap();
        let mut conjugate = buffer(2).unwrap();
        let mut out = buffer(2).unwrap();
        let pointers = (scratch.as_ptr(), conjugate.as_ptr(), out.as_ptr());
        for omega in [2_f64, -2., 0., -8., 8.] {
            for rhs in [
                [Complex::new(1., 2.), Complex::new(-3., 4.)],
                [Complex::ZERO; 2],
            ] {
                // Independent reference using the previous allocating path.
                let matrix =
                    ComplexMatrix::from_operators(&system.a, &system.e, omega.abs()).unwrap();
                let lu = matrix.factorize().unwrap();
                let expected = if rhs == [Complex::ZERO; 2] {
                    vec![Complex::ZERO; 2]
                } else if omega < 0. {
                    lu.solve(&rhs.map(Complex::conj))
                        .unwrap()
                        .into_iter()
                        .map(Complex::conj)
                        .collect()
                } else {
                    lu.solve(&rhs).unwrap()
                };
                solve_at(
                    &circuit,
                    &system,
                    omega,
                    &rhs,
                    &mut scratch,
                    &mut conjugate,
                    &mut out,
                )
                .unwrap();
                assert_eq!(bits(&out), bits(&expected));
                assert_eq!(
                    (scratch.as_ptr(), conjugate.as_ptr(), out.as_ptr()),
                    pointers
                );
            }
        }
        assert!(
            solve_at(
                &circuit,
                &system,
                1.,
                &[Complex::ZERO],
                &mut scratch,
                &mut conjugate,
                &mut out
            )
            .is_err()
        );
        assert!(
            solve_at(
                &circuit,
                &system,
                1.,
                &[Complex::real(Real::NAN), Complex::ZERO],
                &mut scratch,
                &mut conjugate,
                &mut out
            )
            .is_err()
        );
        // A homogeneous singular problem still has to factorize and fail.
        system.a.clear();
        system.e.clear();
        assert!(
            solve_at(
                &circuit,
                &system,
                1.,
                &[Complex::ZERO; 2],
                &mut scratch,
                &mut conjugate,
                &mut out
            )
            .is_err()
        );
        assert!(SolveBuffers::new(&circuit, &system).is_err());
        assert!(buffer(usize::MAX).is_err());
    }
}
