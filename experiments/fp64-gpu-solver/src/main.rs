//! CPU baseline for independent, physically assembled AC ensembles.
#![forbid(unsafe_code)]

use std::error::Error;
use std::fmt::Write as _;
use std::hint::black_box;
use std::io;
use std::path::Path;
use std::time::{Duration, Instant};

use ngspice_rs::analysis::bias::{DcSettings, solve_dc_with};
use ngspice_rs::devices::{Circuit, LinearSystem, ModelContext};
use ngspice_rs::maths::complex::{ComplexLu, ComplexMatrix, IndependentComplexBatch};
use ngspice_rs::netlist::{Parser, source::parse_deck_text};
use ngspice_rs::primitives::Complex;

type Result<T> = std::result::Result<T, Box<dyn Error>>;
type Solutions = Vec<Vec<Complex>>;

const TOLERANCE: f64 = 1e-11;
const TRIALS: usize = 7;

fn invalid(message: &str) -> Box<dyn Error> {
    io::Error::other(message).into()
}

// Grounded RLC ladder/mesh, including true branch-current unknowns from the
// voltage source and inductors. The controlled source makes A nonsymmetric.
fn deck(side: u32, height: u32) -> Result<String> {
    let mut text =
        String::from("Independent AC ensemble\nVdrive in 0 DC 1 AC 1\nRdrive in n0 47\n");
    for y in 0..height {
        for x in 0..side {
            let i = y * side + x;
            writeln!(text, "Rload{i} n{i} 0 2k\nCload{i} n{i} 0 {}n", 10 + i % 13)?;
            if x + 1 < side {
                let j = i + 1;
                writeln!(text, "Rseries{i} n{i} m{i} 100\nLseries{i} m{i} n{j} 47u")?;
            }
            if y + 1 < height {
                writeln!(text, "Rcross{i} n{i} n{} 330", i + side)?;
            }
        }
    }
    text.push_str("Gfeedback n0 0 n1 0 0.0001\n.end\n");
    Ok(text)
}

fn assemble(text: &str) -> Result<(Circuit, LinearSystem)> {
    let netlist = Parser::new().parse_deck(&parse_deck_text(Path::new("ensemble.cir"), text))?;
    let mut circuit = Circuit::from_netlist(&netlist)?;
    circuit.finalize()?;
    let context = ModelContext::default();
    let bias = solve_dc_with(&circuit, &context, &DcSettings::default(), &[], None, None)?;
    let system = circuit.small_signal_system(&context, &bias.solution.values)?;
    // Only frequency-independent device stamps belong in this experiment.
    if !system.e_groups.is_empty() {
        return Err(invalid("unexpected grouped dynamic stamps"));
    }
    Ok((circuit, system))
}

fn scalar(matrices: &[ComplexMatrix], rhs: &[Complex]) -> Result<Solutions> {
    let mut previous: Option<ComplexLu> = None;
    let mut output = Vec::with_capacity(matrices.len());
    for matrix in matrices {
        let factor = matrix.factorize_independent(previous.as_ref())?;
        output.push(factor.solve(rhs)?);
        previous = Some(factor);
    }
    Ok(output)
}

fn packed(matrices: &[ComplexMatrix], rhs: &[Complex]) -> Result<Solutions> {
    let mut batch = IndependentComplexBatch::default();
    let mut output = Vec::with_capacity(matrices.len());
    for tile in matrices.chunks(4) {
        let sources = [rhs; 4];
        for solved in batch.solve(tile, &sources[..tile.len()], false)? {
            output.push(solved?);
        }
    }
    Ok(output)
}

// Componentwise backward error from the original physical stamps, without
// reading any factor entries or using the solver's matrix-vector code.
fn backward_error(
    system: &LinearSystem,
    omega: f64,
    rhs: &[Complex],
    x: &[Complex],
) -> Result<f64> {
    if x.len() != rhs.len() || x.iter().any(|value| !value.is_finite()) {
        return Err(invalid("invalid AC solution"));
    }
    let mut residual: Vec<_> = rhs.iter().map(|&value| -value).collect();
    let mut scale: Vec<_> = rhs.iter().map(|value| value.magnitude()).collect();
    for stamp in system.a.triplets() {
        let term = Complex::real(stamp.value) * x[stamp.col];
        residual[stamp.row] = residual[stamp.row] + term;
        scale[stamp.row] += term.magnitude();
    }
    for stamp in system.e.triplets() {
        let term = Complex::new(0., omega * stamp.value) * x[stamp.col];
        residual[stamp.row] = residual[stamp.row] + term;
        scale[stamp.row] += term.magnitude();
    }
    let mut worst = 0_f64;
    for (value, scale) in residual.iter().zip(scale) {
        let error = value.magnitude();
        if !error.is_finite() || !scale.is_finite() {
            return Err(invalid("nonfinite backward error"));
        }
        if scale != 0. {
            worst = worst.max(error / scale);
        } else if error != 0. {
            return Err(invalid("nonzero residual at zero scale"));
        }
    }
    Ok(worst)
}

fn verify(
    system: &LinearSystem,
    frequencies: &[f64],
    rhs: &[Complex],
    reference: &[Vec<Complex>],
    candidate: &[Vec<Complex>],
) -> Result<(f64, f64)> {
    if reference.len() != frequencies.len() || candidate.len() != frequencies.len() {
        return Err(invalid("point count mismatch"));
    }
    let mut residual = 0_f64;
    let mut difference = 0_f64;
    for ((&omega, expected), actual) in frequencies.iter().zip(reference).zip(candidate) {
        residual = residual.max(backward_error(system, omega, rhs, expected)?);
        residual = residual.max(backward_error(system, omega, rhs, actual)?);
        for (&a, &b) in expected.iter().zip(actual) {
            difference = difference.max((a - b).magnitude() / a.magnitude().max(1.));
        }
    }
    if residual > TOLERANCE || difference > TOLERANCE {
        return Err(
            io::Error::other(format!("residual={residual:e}, difference={difference:e}")).into(),
        );
    }
    Ok((residual, difference))
}

fn measure(mut solve: impl FnMut() -> Result<Solutions>) -> Result<(Duration, Solutions)> {
    black_box(solve()?);
    let mut times = [Duration::ZERO; TRIALS];
    let mut last = Vec::new();
    for elapsed in &mut times {
        let start = Instant::now();
        last = black_box(solve()?);
        *elapsed = start.elapsed();
    }
    times.sort_unstable();
    Ok((times[TRIALS / 2], last))
}

fn workload(side: u32, height: u32, count: u32, benchmark: bool) -> Result<()> {
    let (circuit, system) = assemble(&deck(side, height)?)?;
    let pattern = circuit
        .matrix_pattern()
        .ok_or_else(|| invalid("missing circuit pattern"))?;
    let rhs = system.ac_rhs();
    let frequencies: Vec<_> = (0..count)
        .map(|i| 2. * std::f64::consts::PI * 10_f64.powf(2. + 5. * f64::from(i) / f64::from(count)))
        .collect();
    let matrices = frequencies
        .iter()
        .map(|&omega| {
            ComplexMatrix::from_operators_grouped_on(
                &pattern,
                &system.a,
                &system.e,
                &system.e_groups,
                omega,
            )?
            .ok_or_else(|| invalid("AC stamp escaped fixed topology"))
        })
        .collect::<Result<Vec<_>>>()?;
    let (scalar_time, reference) = if benchmark {
        measure(|| scalar(&matrices, &rhs))?
    } else {
        (Duration::ZERO, scalar(&matrices, &rhs)?)
    };
    let (packed_time, candidate) = if benchmark {
        measure(|| packed(&matrices, &rhs))?
    } else {
        (Duration::ZERO, packed(&matrices, &rhs)?)
    };
    let (residual, difference) = verify(&system, &frequencies, &rhs, &reference, &candidate)?;
    println!(
        "mesh={side}x{height},unknowns={},points={count},scalar_ns={},packed_ns={},backward_error={residual:e},scaled_difference={difference:e}",
        circuit.unknown_count(),
        scalar_time.as_nanos(),
        packed_time.as_nanos(),
    );
    Ok(())
}

fn main() -> Result<()> {
    for (side, height) in [(8, 1), (8, 8)] {
        for count in [64, 256, 1024, 4096] {
            workload(side, height, count, true)?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn actual_rlc_circuits_match_scalar_and_original_physical_stamps() -> Result<()> {
        for (side, height) in [(8, 1), (8, 8)] {
            for count in [1, 3, 5, 17] {
                workload(side, height, count, false)?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod analytic {
    use super::*;

    #[test]
    fn series_rl_parallel_rc_matches_closed_form_transfer() -> Result<()> {
        let (circuit, system) = assemble(
            "RLC transfer\nVdrive in 0 DC 1 AC 1\nRseries in mid 47\nLseries mid out 47u\nRload out 0 2k\nCload out 0 22n\n.end\n",
        )?;
        let pattern = circuit
            .matrix_pattern()
            .ok_or_else(|| invalid("missing pattern"))?;
        let names = circuit.unknown_names();
        let output = names
            .iter()
            .position(|name| name == "out")
            .ok_or_else(|| invalid("missing output node"))?;
        let rhs = system.ac_rhs();
        let frequencies = [0., 1e2, 1e3, 1e4, 1e5, 1e6, 1e7];
        let omegas = frequencies.map(|frequency| 2. * std::f64::consts::PI * frequency);
        let matrices = omegas
            .iter()
            .map(|&omega| {
                ComplexMatrix::from_operators_grouped_on(
                    &pattern,
                    &system.a,
                    &system.e,
                    &system.e_groups,
                    omega,
                )?
                .ok_or_else(|| invalid("AC stamp escaped fixed topology"))
            })
            .collect::<Result<Vec<_>>>()?;
        for (&omega, x) in omegas.iter().zip(packed(&matrices, &rhs)?) {
            let z = Complex::new(47., omega * 47e-6);
            let y = Complex::new(1. / 2000., omega * 22e-9);
            let expected = Complex::real(1.) / (Complex::real(1.) + z * y);
            let actual = x[output];
            assert!((actual - expected).magnitude() <= TOLERANCE * expected.magnitude().max(1.));
            assert!(backward_error(&system, omega, &rhs, &x)? <= TOLERANCE);
        }
        Ok(())
    }
}
