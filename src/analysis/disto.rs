//! `.disto` — small-signal distortion analysis (Volterra series).
//!
//! C references (behaviour only, reimplemented): `distoan.c` (`DISTOan`: the
//! frequency loop, the solves and the plots), `cktdisto.c` (`CKTdisto`: the
//! source inputs and the device dispatch), `dkerproc.c` (`DkerProc`: the
//! output scaling), `dloadfns.c` (the kernel polynomials, see
//! [`crate::devices::distortion::CircuitDistortion::add_products`]), `dsetparm.c`/`inp2dot.c`
//! (the card) and the device routines summarised in
//! [`crate::devices::distortion`].
//!
//! ```text
//! .disto {dec|oct|lin} pts fstart fstop [f2overf1]
//! ```
//!
//! The operating point is solved as for `.ac` and every device describes its
//! nonlinearities there as Taylor polynomials. At every `f1` of C's sweep the
//! linearized system `A + j omega E` is solved for the first-order kernel
//! `H1(f1)`, driven by the `distof1` inputs as the phasors `0.5 mag e^(j
//! phase)`, then for each higher-order kernel with a right-hand side built
//! from the lower-order ones:
//!
//! * without `f2overf1`: `H2(f1, f1)` at `2 f1` and `H3(f1, f1, f1)` at `3 f1`,
//!   written (times 2, `DkerProc`) as the plots `DISTORTION - 2nd harmonic`
//!   and `DISTORTION - 3rd harmonic`;
//! * with `f2overf1`: the second input `H1(f2)` at `f2 = f2overf1 * fstart`,
//!   **fixed** over the sweep as in C (`distoan.c`: "keeping f2 const to be
//!   compatible with spectre"), driven by the `distof2` inputs, and
//!   `H2(f1, f1)`; then `H2(f1, f2)` at `f1 + f2`, `H2(f1, -f2)` at `f1 - f2`
//!   and `H3(f1, f1, -f2)` at `2 f1 - f2`, written (times 4, 4 and 6) as
//!   `DISTORTION - IM: f1+f2`, `DISTORTION - IM: f1-f2` and
//!   `DISTORTION - IM: 2f1-f2`.
//!
//! Each plot is complex, along the `frequency` scale `f1`, with every node
//! voltage and branch current exactly as an `.ac` plot. The sweep is C's own:
//! `dec`/`oct` multiply by `exp(ln(10 or 2)/pts)` from `fstart` while
//! `f <= fstop + delta*fstop*reltol`, and `lin` adds `(fstop - fstart)/(pts +
//! 1)` (so it measures `pts + 2` frequencies) while `f <= fstop +
//! delta*reltol`.

use crate::analysis::frequency::{Advance, FrequencyGrid};
use crate::analysis::linear::{number, plot, unsupported};
use crate::analysis::results::Plot;
use crate::analysis::{AnalysisContext, AnalysisPlan};
use crate::devices::Circuit;
use crate::devices::distortion::circuit_distortion;
use crate::primitives::{Real, SpiceResult, parse_spice_number};

mod workspace;
use workspace::{Inputs, Workspace};

/// C's default `CKTreltol`, the frequency-loop tolerance without `.option
/// reltol`.
/// The harmonic plot titles (`distoan.c`).
pub const HARMONIC_PLOTNAMES: [&str; 2] =
    ["DISTORTION - 2nd harmonic", "DISTORTION - 3rd harmonic"];
/// The intermodulation plot titles (`distoan.c`).
pub const INTERMODULATION_PLOTNAMES: [&str; 3] = [
    "DISTORTION - IM: f1+f2",
    "DISTORTION - IM: f1-f2",
    "DISTORTION - IM: 2f1-f2",
];

/// The frequency stepping of a `.disto` card (`DISTOAN.DstepType`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Stepping {
    Decade,
    Octave,
    Linear,
}

/// A parsed `.disto` card.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct DistoCard {
    stepping: Stepping,
    steps: u32,
    start: Real,
    stop: Real,
    f2_over_f1: Option<Real>,
}

impl DistoCard {
    pub(super) const fn plot_count(&self) -> usize {
        if self.f2_over_f1.is_some() { 3 } else { 2 }
    }
}

/// C's `INPgetValue(IF_INTEGER)`: `floor(value + 0.5)`.
fn integer(text: &str) -> SpiceResult<i64> {
    let value = parse_spice_number(text)
        .filter(|v| v.is_finite())
        .ok_or_else(|| unsupported(".disto expects an integer point count"))?;
    crate::primitives::c_integer(value)
        .ok_or_else(|| unsupported(".disto point count is out of range"))
}

pub(super) fn parse(arguments: &[String]) -> SpiceResult<DistoCard> {
    let syntax = || unsupported(".disto needs {dec|oct|lin} pts fstart fstop [f2overf1]");
    let args: Vec<_> = arguments.iter().map(String::as_str).collect();
    if !(4..=5).contains(&args.len()) {
        return Err(syntax());
    }
    let stepping = match args[0].to_ascii_lowercase().as_str() {
        "dec" => Stepping::Decade,
        "oct" => Stepping::Octave,
        "lin" => Stepping::Linear,
        other => {
            return Err(unsupported(format!(
                ".disto sweep must be dec, oct or lin, not '{other}'"
            )));
        }
    };
    let steps = integer(args[1])?;
    // dec/oct with no step never advance (`distoan.c` would loop forever); a
    // `lin` sweep with zero steps measures fstart and fstop.
    let minimum = i64::from(stepping != Stepping::Linear);
    if steps < minimum {
        return Err(unsupported(format!(
            ".disto needs at least {minimum} step(s) for this sweep, got {steps}"
        )));
    }
    let start = number(Some(args[2]), "start frequency")?;
    let stop = number(Some(args[3]), "stop frequency")?;
    if start <= 0. || stop <= 0. {
        // dsetparm.c: "Frequency of 0 is invalid".
        return Err(unsupported(format!(
            ".disto frequencies must be positive, got {start} .. {stop} (dsetparm.c)"
        )));
    }
    if stop < start {
        return Err(unsupported(format!(
            ".disto needs fstart <= fstop, got {start} .. {stop} (C writes empty plots)"
        )));
    }
    let f2_over_f1 = match args.get(4) {
        Some(text) => Some(number(Some(text), "f2overf1")?),
        None => None,
    };
    Ok(DistoCard {
        stepping,
        steps: u32::try_from(steps).map_err(|_| syntax())?,
        start,
        stop,
        f2_over_f1,
    })
}

/// `distoan.c`'s frequency sweep.
pub(super) fn frequencies(card: &DistoCard, reltol: Real) -> SpiceResult<Vec<Real>> {
    let n = Real::from(card.steps);
    let (delta, tolerance) = match card.stepping {
        Stepping::Decade => {
            let delta = (10f64.ln() / n).exp();
            (delta, delta * card.stop * reltol)
        }
        Stepping::Octave => {
            let delta = (2f64.ln() / n).exp();
            (delta, delta * card.stop * reltol)
        }
        Stepping::Linear => {
            let delta = (card.stop - card.start) / (n + 1.);
            (delta, delta * reltol)
        }
    };
    // Detect a rounded, stationary multiplier exactly, not approximately.
    let advance = match card.stepping {
        Stepping::Decade | Stepping::Octave if delta.to_bits() == 1.0_f64.to_bits() => {
            Advance::Single
        }
        Stepping::Decade | Stepping::Octave => Advance::Multiply(delta),
        Stepping::Linear if delta == 0. => Advance::Single,
        Stepping::Linear => Advance::Add(delta),
    };
    FrequencyGrid::new(card.start, card.stop, tolerance, advance, ".disto")?.collect()
}

fn output_plots(circuit: &Circuit, intermodulation: bool) -> SpiceResult<Vec<Plot>> {
    let titles: &[&str] = if intermodulation {
        &INTERMODULATION_PLOTNAMES
    } else {
        &HARMONIC_PLOTNAMES
    };
    titles
        .iter()
        .enumerate()
        .map(|(index, title)| {
            plot(
                circuit,
                &format!("disto{}", index + 1),
                title,
                Some(("frequency", "frequency")),
                true,
            )
        })
        .collect()
}

/// Runs `.disto` and returns its plots: the 2nd and 3rd harmonics, or the
/// three intermodulation products with `f2overf1`.
pub(super) fn run(
    circuit: &mut Circuit,
    request: &AnalysisPlan,
    context: &AnalysisContext,
    card: &DistoCard,
    grid: &[Real],
    include_bias: bool,
) -> SpiceResult<Vec<Plot>> {
    let settings = request.dc_settings().clone();
    circuit.finalize()?;
    let small = crate::analysis::ac::SmallSignal::prepare(circuit, request, context, settings)?;
    let mut retained = small.retained_bias(circuit, request, context, include_bias)?;
    if small.varies() {
        return Err(unsupported(
            ".disto of a circuit with a `hertz`-dependent device (distoan.c linearizes once, \
             at DC)",
        ));
    }
    let model = context.model_context();
    let distortion = circuit_distortion(circuit, &model, small.bias())?;
    let system = small.system();
    let intermodulation = card.f2_over_f1.is_some();
    let inputs = Inputs::bind(system, &distortion, intermodulation)?;
    let two_pi = 2. * std::f64::consts::PI;
    let omega2 = two_pi * card.start * card.f2_over_f1.unwrap_or(0.);
    let mut plots = output_plots(circuit, intermodulation)?;
    let mut workspace = Workspace::new(circuit, system, inputs, &mut plots, grid.len())?;
    for &freq in grid {
        workspace.frequency(&distortion, freq, two_pi * freq, omega2, &mut plots)?;
    }
    retained.extend(plots);
    Ok(retained)
}

#[cfg(test)]
mod tests {
    use super::{DistoCard, Stepping, frequencies};

    fn args(text: &str) -> Vec<String> {
        text.split_whitespace().map(str::to_owned).collect()
    }

    #[test]
    fn plot_count_follows_f2overf1() {
        use crate::analysis::AnalysisInput;
        use crate::primitives::AnalysisKind;
        let plot_count = |arguments: &[String]| {
            AnalysisInput::with_arguments(AnalysisKind::Distortion, arguments.iter().cloned())
                .resolve()
                .unwrap()
                .plot_count()
        };
        assert_eq!(plot_count(&args("dec 10 1k 100k")), 2);
        assert_eq!(plot_count(&args("dec 10 1k 100k 0.9")), 3);
        assert_eq!(plot_count(&args("dec 10 1k 100k rtol=1e-3")), 2);
    }

    #[test]
    fn sweeps_follow_distoan() {
        let card = |stepping, steps, start, stop| DistoCard {
            stepping,
            steps,
            start,
            stop,
            f2_over_f1: None,
        };
        // lin measures pts + 2 frequencies.
        let lin = frequencies(&card(Stepping::Linear, 3, 1e3, 4e3), 1e-3).unwrap();
        assert_eq!(lin, vec![1e3, 1.75e3, 2.5e3, 3.25e3, 4e3]);
        let dec = frequencies(&card(Stepping::Decade, 2, 1e3, 1e5), 1e-3).unwrap();
        assert_eq!(dec.len(), 5);
        assert!((dec[4] / 1e5 - 1.).abs() < 1e-12);
        let one = frequencies(&card(Stepping::Linear, 0, 1e3, 1e3), 1e-3).unwrap();
        assert_eq!(one, vec![1e3]);
        let oct = frequencies(&card(Stepping::Octave, 1, 1., 8.), 1e-3).unwrap();
        assert_eq!(oct.len(), 4);
    }
    #[test]
    fn distortion_source_rows_and_contributions_are_checked() {
        use crate::devices::distortion::circuit_distortion;
        use crate::devices::{Circuit, ModelContext};
        use crate::maths::Vector;
        use crate::netlist::{Parser, source::parse_deck_text};
        use std::path::Path;

        let deck = Parser::new()
            .parse_deck(&parse_deck_text(
                Path::new("input.cir"),
                "source test\nv1 a 0 0 distof1 2\nr1 a 0 1k\n.end\n",
            ))
            .unwrap();
        let mut circuit = Circuit::from_netlist(&deck).unwrap();
        let mut system = circuit.linear_system().unwrap();
        let model = ModelContext::default();
        let bias = Vector::zeros(circuit.unknown_count());
        let distortion = circuit_distortion(&circuit, &model, &bias).unwrap();
        assert!(super::Inputs::bind(&system, &distortion, false).is_ok());
        let [source] = system.sources.as_mut_slice() else {
            panic!("expected one source")
        };
        source.rows.push((usize::MAX, 1.));
        assert!(super::Inputs::bind(&system, &distortion, false).is_err());
        let [source] = system.sources.as_mut_slice() else {
            panic!("expected one source")
        };
        source.rows.pop();
        source.rows[0].1 = f64::INFINITY;
        assert!(super::Inputs::bind(&system, &distortion, false).is_err());
    }
}
