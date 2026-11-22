//! Opt-in live C comparison of the #94/#95 source functions.
//!
//! ```text
//! NGSPICE_BIN=/abs/path/to/ngspice \
//!   cargo test -p ngspice-rs --test c_source_functions -- --ignored
//! ```
//!
//! * **Waveform values.** Every source drives its own node (V directly, I through
//!   1 kohm), so C's node voltages *are* `vsrcload.c`/`isrcload.c` evaluations
//!   at C's own timepoints, with C's `CKTstep`/`CKTfinalTime` defaults. Each is
//!   compared with the port's [`ngspice_rs::devices::Waveform`] at the same instant
//!   (left or right limit: C evaluates a jump instant once). No integration is
//!   involved, so the bound is round-off level.
//! * **Operating point.** The same sources without DC values under `.op` give
//!   C's time-zero values.
//! * **Fourier.** A SIN-driven diode clipper (strong harmonics) is transformed
//!   by both engines' `.four`; THD and harmonic magnitudes are compared.
//! * **Documented divergences.** RC decks without marker sources (discontinuous
//!   repeated PWL, SIN/EXP/SFFM/AM delays, a fractional PULSE count) run in
//!   both engines; the worst `v(out)` difference must stay within the bounds
//!   stated in `docs/port/TRANSIENT.md`.
//!
//! Ordinary `cargo test` never needs C: everything here is `#[ignore]`d.

use ngspice_conformance::ngspice::Scratch;
use ngspice_conformance::process::{ProcessLimits, run};
use std::fs;
use std::path::Path;

use ngspice_rs::analysis::{AnalysisContext, Plot, RawFile, RunConfig, fourier};
use ngspice_rs::devices::{Circuit, Limit, SourceKind, TransientTiming};
use ngspice_rs::netlist::Parser;
use ngspice_rs::netlist::source::parse_deck_text;
use ngspice_rs::primitives::Real;

/// Each source drives node `n<k>`; I sources (`n<k>` to ground) pull current
/// through a 1 kohm load, so `v(n<k>) = -1k * I`.
const SOURCES: &str = "\
vsin1 n1 0 sin(0.5 2 1k 0.1m 300 45)
vsin2 n2 0 sin(1 1)
vsin3 n3 0 sine(0 1 0 0.3m)
vexp1 n4 0 exp(0 1)
vexp2 n5 0 exp(1 -1 0.3m 0.1m)
vexp3 n6 0 exp(0 1 0 0.2m 0.8m)
vsffm1 n7 0 sffm(0.1 1 5k 2 500 0.2m 30 60)
vsffm2 n8 0 sffm(0 1)
vsffm3 n9 0 sffm(0 1 10k 50 1k)
vam1 n10 0 am(0 1 0.5 1k 10k 0.1m 30 45)
vam2 n11 0 am(0.2 1)
vpulse1 n12 0 pulse(0 1 0.1m 20u 20u 0.1m 0.3m 2.5)
vpulse2 n13 0 pulse(0 1 0 20u 20u 0.1m 0.3m 3)
vpwl1 n14 0 pwl(0 0 0.3m 1 0.6m 0.5) r=0.3m td=0.1m
vpwl2 n15 0 pwl(0 0 0.3m 1 0.6m 0.5) td=0.2m
vpwl3 n16 0 pwl(0 0 0.2m 1 0.4m 0) r=0
isin n17 0 sin(0 1m 2k 0.05m 0 90)
iexp n18 0 exp(0 1m 0.1m 0.1m 0.5m 0.2m)
isffm n19 0 sffm(0 1m 4k 1 400)
iam n20 0 am(0 1m 0.5m 500 8k)
ipwl n21 0 pwl(0 0 0.1m 1m 0.2m 0) r=0 td=0.05m
ipulse n22 0 pulse(0 1m 0.1m 10u 10u 50u 0.2m 2)
";

fn loads() -> String {
    (1..=22).map(|k| format!("r{k} n{k} 0 1k\n")).collect()
}

/// Runs `cards` (title and `.end` added) under `analysis` in C, returning the
/// plot written by `write`, and the full output text.
fn run_c(tag: &str, cards: &str, analysis: &str, extra: &str) -> (Option<Plot>, String) {
    let binary = std::env::var_os("NGSPICE_BIN").expect("set absolute NGSPICE_BIN");
    assert!(
        Path::new(&binary).is_absolute(),
        "NGSPICE_BIN must be absolute"
    );
    let scratch =
        Scratch::new("c-source-functions").unwrap_or_else(|error| panic!("{tag}: {error}"));
    let dir = scratch.path();
    // An analysis command runs from a `.control` block and writes a rawfile;
    // without one the deck's own cards (e.g. `.tran` + `.four`) run in batch.
    let deck = if analysis.is_empty() {
        format!("c\n{cards}{extra}.end\n")
    } else {
        format!(
            "c\n{cards}{extra}.control\nset filetype=ascii\n{analysis}\nwrite result.raw\nquit\n.endc\n.end\n"
        )
    };
    fs::write(dir.join("c.cir"), deck).unwrap();
    let output = run(&binary, dir, &["-b", "c.cir"], ProcessLimits::ORACLE).unwrap();
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(output.status.success(), "{text}");
    let plot = fs::read_to_string(dir.join("result.raw"))
        .ok()
        .map(|raw| RawFile::parse(&raw).unwrap().plots[0].plot.clone());
    (plot, text)
}

fn rust_netlist(text: &str) -> ngspice_rs::netlist::ast::Netlist {
    Parser::new()
        .parse_deck(&parse_deck_text(Path::new("c.cir"), text))
        .unwrap()
}

#[test]
#[ignore = "requires absolute NGSPICE_BIN; runs a temporary C deck out of process"]
fn source_function_values_match_c_at_c_timepoints() {
    let (step, stop) = (10e-6, 2e-3);
    let cards = format!("{SOURCES}{}", loads());
    let (plot, text) = run_c("values", &cards, "tran 10u 2m", "");
    let plot = plot.unwrap_or_else(|| panic!("no C rawfile:\n{text}"));
    let netlist = rust_netlist(&format!("t\n{cards}.end\n"));
    let mut system = Circuit::from_netlist(&netlist)
        .unwrap()
        .linear_system()
        .unwrap();
    system
        .bind_transient_timing(&TransientTiming::new(step, stop).unwrap())
        .unwrap();
    let time: Vec<Real> = plot.column("time").unwrap().iter().map(|v| v.re).collect();
    assert!(time.len() > 200, "C wrote {} rows", time.len());
    let mut compared = 0;
    // The system lists sources in C's load order, so each is matched to the
    // node its own card drives.
    for source in &system.sources {
        let card = netlist
            .devices
            .iter()
            .find(|d| d.name.eq_ignore_ascii_case(&source.name))
            .unwrap();
        let node = format!("v({})", card.nodes[0].to_ascii_lowercase());
        let scale = match source.kind {
            SourceKind::Voltage => 1.,
            SourceKind::Current => -1e3,
        };
        let c = plot
            .column(&node)
            .unwrap_or_else(|| panic!("C has no {node}"));
        for (t, c) in time.iter().zip(&c) {
            // C's rawfile prints time with 16 significant digits, so at a jump
            // its sample may sit an ulp to either side: the left and right
            // limits and the values 1e-14 relative away are all admissible.
            // This also covers repeated-PWL boundaries, where C's single value
            // depends on how its landing time rounds (either limit); the
            // effect of the port always integrating to the left limit there
            // is bounded separately by the unmarked-deck test below.
            let delta = 1e-14 * t.abs();
            let candidates = [
                (*t, Limit::Left),
                (*t, Limit::Right),
                (t - delta, Limit::Left),
                (t + delta, Limit::Right),
            ]
            .map(|(at, limit)| scale * source.waveform.value_at(at, limit).unwrap());
            let error = candidates
                .iter()
                .map(|v| (v - c.re).abs())
                .fold(Real::INFINITY, Real::min);
            assert!(
                error <= 1e-9 * c.re.abs().max(1.),
                "{} at t={t:e}: C {:e}, port {candidates:?}",
                source.name,
                c.re
            );
            compared += 1;
        }
    }
    assert_eq!(system.sources.len(), 22);
    println!(
        "{compared} source values compared at {} C timepoints",
        time.len()
    );
}

#[test]
#[ignore = "requires absolute NGSPICE_BIN; runs a temporary C deck out of process"]
fn operating_points_match_c_time_zero_values() {
    let cards = format!("{SOURCES}{}", loads());
    let (plot, text) = run_c("op", &cards, "op", "");
    let c = plot.unwrap_or_else(|| panic!("no C rawfile:\n{text}"));
    let netlist = rust_netlist(&format!("t\n{cards}.op\n.end\n"));
    let config = RunConfig::from_netlist(&netlist).unwrap();
    let request = config.plan_for(&netlist.analyses[0]).unwrap();
    let mut circuit = config.circuit(&netlist).unwrap();
    let rust = request
        .run(&mut circuit, &AnalysisContext::default())
        .unwrap();
    for k in 1..=22 {
        let name = format!("v(n{k})");
        let (ours, theirs) = (
            rust.value(&name, 0).unwrap().re,
            c.value(&name, 0).unwrap().re,
        );
        assert!(
            (ours - theirs).abs() <= 1e-12 * theirs.abs().max(1.),
            "{name}: port {ours}, C {theirs}"
        );
    }
}

/// A 1 kHz SIN into a resistor-diode clipper: strongly distorted, so the THD
/// and harmonic magnitudes are meaningful comparisons.
const CLIPPER: &str = "v1 in 0 sin(0 2 1k)\nr1 in out 1k\nd1 out 0 dm\nc1 out 0 10n\n\
                       .model dm d(is=1e-14)\n";

#[test]
#[ignore = "requires absolute NGSPICE_BIN; runs a temporary C deck out of process"]
fn four_of_a_sin_driven_clipper_matches_c() {
    let text = format!("four\n{CLIPPER}.tran 1u 4m\n.four 1k v(out)\n.end\n");
    let parsed = Parser::new()
        .parse_deck_with_output(&parse_deck_text(Path::new("four.cir"), &text))
        .unwrap();
    let netlist = &parsed.netlist;
    let config = RunConfig::from_netlist(netlist).unwrap();
    let request = config.plan_for(&netlist.analyses[0]).unwrap();
    let mut circuit = config.circuit(netlist).unwrap();
    let plot = request.run(&mut circuit, &config.context()).unwrap();
    let ours = fourier::resolve(&plot, request.kind(), &parsed.fourier)
        .unwrap()
        .remove(0);

    let (_, output) = run_c("four", CLIPPER, "", ".tran 1u 4m\n.four 1k v(out)\n");
    let mut thd = None;
    let mut rows = Vec::new();
    for line in output.lines() {
        if let Some(rest) = line.trim().strip_prefix("No. Harmonics:") {
            thd = rest
                .split_once("THD:")
                .and_then(|(_, after)| after.split_whitespace().next())
                .and_then(|v| v.parse::<Real>().ok());
            continue;
        }
        let fields: Vec<_> = line.split_whitespace().collect();
        if let [order, frequency, magnitude, ..] = fields[..]
            && let (Ok(order), Ok(frequency), Ok(magnitude)) = (
                order.parse::<u32>(),
                frequency.parse::<Real>(),
                magnitude.parse::<Real>(),
            )
            && order > 0
            && frequency > 0.
        {
            rows.push((order, magnitude));
        }
    }
    let thd = thd.unwrap_or_else(|| panic!("C printed no THD:\n{output}"));
    // A clipped sine is far from clean: both engines must see it.
    assert!(thd > 5., "C THD {thd} %");
    let ours_percent = ours.thd * 100.;
    assert!(
        (ours_percent - thd).abs() <= 0.01 * thd + 0.05,
        "THD: port {ours_percent} %, C {thd} %"
    );
    assert_eq!(rows.len(), 9, "{output}");
    for (order, magnitude) in rows {
        let harmonic = ours
            .harmonics
            .iter()
            .find(|h| h.order == order)
            .unwrap_or_else(|| panic!("no harmonic {order}"));
        assert!(
            (harmonic.amplitude - magnitude).abs() <= 0.01 * magnitude + 1e-4,
            "harmonic {order}: port {}, C {magnitude}",
            harmonic.amplitude
        );
    }
}

/// Linear interpolation of `(time, value)` at `t` (clamped at the ends).
fn interpolate(time: &[Real], value: &[Real], t: Real) -> Real {
    let k = time.partition_point(|x| *x < t);
    if k == 0 {
        return value[0];
    }
    if k == time.len() {
        return value[k - 1];
    }
    if time[k] == t {
        return value[k];
    }
    let (a, b) = (time[k - 1], time[k]);
    value[k - 1] + (value[k] - value[k - 1]) * (t - a) / (b - a)
}

/// Documented divergences from C on ordinary decks (no marker sources):
/// `(deck, worst |v(out)| difference allowed)`. Each RC has tau = 0.1 ms.
///
/// * Discontinuous repeated PWL: at a repetition boundary C loads either the
///   end value or the restart value depending on how its accumulated landing
///   time rounds; the port always takes the left limit (measured 0.024-0.036 V
///   for a 1 V jump with 7-10 us steps).
/// * SIN/EXP/SFFM/AM delays and a fractional PULSE count: the port lands on the
///   corner or jump, C's `VSRCaccept` does not (measured 2e-5 to 9e-3 V).
const UNMARKED: &[(&str, &str, Real)] = &[
    ("sawtooth", "v1 in 0 pwl(0 0 1m 1) r=0\n.tran 10u 4m", 0.04),
    (
        "odd sawtooth",
        "v1 in 0 pwl(0 0 0.3m 1) r=0\n.tran 7u 3m",
        0.03,
    ),
    (
        "delayed partial repeat",
        "v1 in 0 pwl(0 0 0.3m 1 0.7m 0.2) r=0.3m td=0.05m\n.tran 10u 4m",
        0.035,
    ),
    (
        "exp",
        "v1 in 0 exp(0 1 0.2m 0.3m 1.5m 0.5m)\n.tran 5u 4m",
        2e-4,
    ),
    (
        "exp td2 < td1",
        "v1 in 0 exp(0 1 0.5m 0.2m 0.3m 0.1m)\n.tran 5u 4m",
        0.012,
    ),
    (
        "sin",
        "v1 in 0 sin(0.5 1 1k 0.2m 200 30)\n.tran 2u 3m",
        1e-4,
    ),
    (
        "sffm",
        "v1 in 0 sffm(0.5 1 5k 1 500 0.3m)\n.tran 1u 2m",
        3e-3,
    ),
    ("am", "v1 in 0 am(0.5 1 0.5 500 5k 0.3m)\n.tran 1u 2m", 3e-3),
    (
        "fractional pulse count",
        "v1 in 0 pulse(0 1 0.1m 20u 20u 0.2m 0.5m 1.3)\n.tran 10u 3m",
        2e-3,
    ),
];

#[test]
#[ignore = "requires absolute NGSPICE_BIN; runs temporary C decks out of process"]
fn unmarked_decks_diverge_from_c_only_within_documented_bounds() {
    for (k, (name, cards, bound)) in UNMARKED.iter().enumerate() {
        let (source, tran) = cards.split_once('\n').unwrap();
        let circuit_cards = format!("{source}\nr1 in out 1k\nc1 out 0 0.1u\n");
        let (plot, text) = run_c(
            &format!("unmarked{k}"),
            &circuit_cards,
            &tran.replace(".tran", "tran"),
            "",
        );
        let c = plot.unwrap_or_else(|| panic!("{name}: no C rawfile:\n{text}"));
        let netlist = rust_netlist(&format!("t\n{circuit_cards}{tran}\n.end\n"));
        let config = RunConfig::from_netlist(&netlist).unwrap();
        let request = config.plan_for(&netlist.analyses[0]).unwrap();
        let mut circuit = config.circuit(&netlist).unwrap();
        let ours = request
            .run(&mut circuit, &AnalysisContext::default())
            .unwrap();
        let column = |plot: &Plot, name: &str| -> Vec<Real> {
            plot.column(name).unwrap().iter().map(|v| v.re).collect()
        };
        let (port_time, port_out) = (column(&ours, "time"), column(&ours, "v(out)"));
        let worst = column(&c, "time")
            .iter()
            .zip(column(&c, "v(out)"))
            .map(|(t, v)| (v - interpolate(&port_time, &port_out, *t)).abs())
            .fold(0., Real::max);
        println!("{name}: worst v(out) difference {worst:.3e} V (bound {bound:e})");
        assert!(worst <= *bound, "{name}: {worst:e} > {bound:e}");
    }
}

#[test]
#[ignore = "requires absolute NGSPICE_BIN; runs a temporary C deck out of process"]
fn polar_ac_coefficients_and_phase_overrides_match_c() {
    use ngspice_rs::devices::ModelContext;

    let base = "v1 a 0 ac 2 0\ni1 b 0 ac 3 0\nr1 a 0 1\nr2 b 0 1\n";
    let mut circuit = Circuit::from_netlist(&rust_netlist(&format!("t\n{base}.end\n"))).unwrap();
    let context = ModelContext::default();
    // The huge phase distinguishes C's multiplication/division order
    // from multiplying by the pre-rounded degrees-to-radians factor.
    for (index, phase) in [123_456_789_012_345., -720.25, -0.].into_iter().enumerate() {
        let static_cards =
            format!("v1 a 0 ac 2 {phase}\ni1 b 0 ac 3 {phase}\nr1 a 0 1\nr2 b 0 1\n");
        let (c_initial, text) = run_c(
            &format!("ac-initial-{index}"),
            &static_cards,
            "ac lin 1 1k 1k",
            "",
        );
        let c_initial = c_initial.unwrap_or_else(|| panic!("missing AC plot: {text}"));
        let (c_altered, text) = run_c(
            &format!("ac-altered-{index}"),
            base,
            &format!("alter @v1[acphase] = {phase}\nalter @i1[acphase] = {phase}\nac lin 1 1k 1k"),
            "",
        );
        let c_altered = c_altered.unwrap_or_else(|| panic!("missing altered AC plot: {text}"));
        let initial = Circuit::from_netlist(&rust_netlist(&format!("t\n{static_cards}.end\n")))
            .unwrap()
            .linear_system()
            .unwrap();
        let overrides = context
            .with_instance_override(
                circuit
                    .instance_override("v1", "acphase", phase, &context)
                    .unwrap(),
            )
            .unwrap()
            .with_instance_override(
                circuit
                    .instance_override("i1", "acphase", phase, &context)
                    .unwrap(),
            )
            .unwrap();
        let swept = circuit.linear_system_with_context(&overrides).unwrap();
        for (name, node, sign) in [("v1", "v(a)", 1.), ("i1", "v(b)", -1.)] {
            let a = initial.sources.iter().find(|s| s.name == name).unwrap().ac;
            let b = swept.sources.iter().find(|s| s.name == name).unwrap().ac;
            assert_eq!(
                [a.re.to_bits(), a.im.to_bits()],
                [b.re.to_bits(), b.im.to_bits()]
            );
            for plot in [&c_initial, &c_altered] {
                let c = plot.column(node).unwrap()[0];
                let bound = 8. * Real::EPSILON * a.magnitude().max(1.);
                assert!(
                    (a.re - sign * c.re).abs() <= bound,
                    "{name} {phase}: real {a:?} != {c:?}"
                );
                assert!(
                    (a.im - sign * c.im).abs() <= bound,
                    "{name} {phase}: imag {a:?} != {c:?}"
                );
            }
        }
    }
}

/// Binary samples preserve the exact C solver times and forcing bits, unlike
/// the ASCII comparison above.
#[test]
#[ignore = "requires absolute NGSPICE_BIN"]
fn pwl_and_pulse_match_c_binary_samples_bit_for_bit() {
    use ngspice_conformance::ngspice::Ngspice;

    let deck = "Exact source samples\n\
vplain a 0 pwl(0 1.25 0.3m 2.75 0.6m 1.5)\n\
vshift b 0 pwl(0 1.25 0.3m 2.75 0.6m 1.5) td=0.2m\n\
vrepeat c 0 pwl(0 1.25 0.3m 2.75 0.6m 1.5) td=0.1m r=0.3m\n\
vpulse d 0 pulse(1.25 2.75 0.1m 20u 30u 0.1m 0.3m 2.5)\n\
ra a 0 1\nrb b 0 1\nrc c 0 1\nrd d 0 1\n.tran 10u 2m\n.end\n";
    let scratch = Scratch::new("waveform-bits").unwrap();
    fs::write(scratch.path().join("wave.cir"), deck).unwrap();
    Ngspice::pinned()
        .unwrap()
        .run(
            scratch.path(),
            &["-n", "-b", "-r", "result.raw", "wave.cir"],
        )
        .unwrap()
        .require_success("binary source samples")
        .unwrap();
    let raw = RawFile::load(scratch.path().join("result.raw")).unwrap();
    let plot = &raw.plots[0].plot;
    let times = plot.column("time").unwrap();
    assert!(times.len() > 200);
    let netlist = rust_netlist(deck);
    let mut system = Circuit::from_netlist(&netlist)
        .unwrap()
        .linear_system()
        .unwrap();
    system
        .bind_transient_timing(&TransientTiming::new(10e-6, 2e-3).unwrap())
        .unwrap();
    assert_eq!(system.sources.len(), 4);
    for (source_name, node) in [
        ("vplain", "v(a)"),
        ("vshift", "v(b)"),
        ("vrepeat", "v(c)"),
        ("vpulse", "v(d)"),
    ] {
        let source = system
            .sources
            .iter()
            .find(|s| s.name == source_name)
            .unwrap();
        let expected = plot.column(node).unwrap();
        assert_eq!(times.len(), expected.len());
        for (time, value) in times.iter().zip(expected) {
            let actual = source.waveform.value_at(time.re, Limit::Right).unwrap();
            assert_eq!(
                actual.to_bits(),
                value.re.to_bits(),
                "{source_name} at {:.17e}",
                time.re
            );
        }
    }
    println!("{} exact C binary source samples", 4 * times.len());
}

/// Independently specified nominal corners must appear in the actual C and
/// Rust trajectories. The expected lists are not generated by the shared
/// timeline; an incorrect resolver cannot make an incorrect solver pass.
#[test]
#[ignore = "requires absolute NGSPICE_BIN; runs a temporary C deck out of process"]
fn elaborated_waveform_timeline_matches_independent_c_landings() {
    const STEP: Real = 1e-5;
    const STOP: Real = 4e-3;
    let cases: [(&str, &str, &[Real]); 6] = [
        (
            "pulse-default",
            "v1 in 0 pulse(0 1 0.3m 0 0)\nr1 in 0 1k\n",
            &[0.3e-3, 0.31e-3, 0.32e-3],
        ),
        (
            "pulse-count",
            "v1 in 0 pulse(0 1 0.2m 20u 20u 0.1m 1m 2.5)\nr1 in 0 1k\n",
            &[
                0.2e-3, 0.22e-3, 0.32e-3, 0.34e-3, 1.2e-3, 1.22e-3, 1.32e-3, 1.34e-3, 2.2e-3,
                2.22e-3, 2.32e-3, 2.34e-3,
            ],
        ),
        (
            "pwl-repeat",
            "v1 in 0 pwl(0 0 0.5m 1 1m 0) r=0 td=0.2m\nr1 in 0 1k\n",
            &[
                0.2e-3, 0.7e-3, 1.2e-3, 1.7e-3, 2.2e-3, 2.7e-3, 3.2e-3, 3.7e-3,
            ],
        ),
        (
            "adjacent-pwl",
            "v1 in 0 pwl(0 0 0.3m 1 0.300001m 0 1m 0) r=-0 td=0\nr1 in 0 1k\n",
            &[
                0.3e-3,
                0.300_001e-3,
                1e-3,
                1.3e-3,
                1.300_001e-3,
                2e-3,
                2.3e-3,
                2.300_001e-3,
                3e-3,
                3.3e-3,
                3.300_001e-3,
            ],
        ),
        (
            "current-repeat",
            "i1 in 0 pwl(0 0 0.5m 1m 1m 0) r=-0 td=0.2m\nr1 in 0 1k\n",
            &[
                0.2e-3, 0.7e-3, 1.2e-3, 1.7e-3, 2.2e-3, 2.7e-3, 3.2e-3, 3.7e-3,
            ],
        ),
        (
            "nested-param",
            ".subckt inner b params: td=1m\nv1 b 0 pwl(0 0 10u 1 1.01m 1 1.03m 0) td={td}\n.ends\n.subckt outer a params: d=1m\nx1 a inner td={d}\n.ends\nx1 n outer d=0.5m\nx2 m outer d=1m\nr1 n 0 1k\nr2 m 0 1k\n",
            &[
                0.5e-3, 0.51e-3, 1e-3, 1.01e-3, 1.51e-3, 1.53e-3, 2.01e-3, 2.03e-3,
            ],
        ),
    ];
    for (tag, cards, expected) in cases {
        let (c, log) = run_c(tag, cards, "tran 10u 4m", "");
        let c = c.unwrap_or_else(|| panic!("{tag}: no C plot: {log}"));
        let netlist = rust_netlist(&format!("t\n{cards}.tran 10u 4m\n.end\n"));
        let config = RunConfig::from_netlist(&netlist).unwrap();
        let request = config.plan_for(&netlist.analyses[0]).unwrap();
        let mut circuit = config.circuit(&netlist).unwrap();
        let timing = TransientTiming::new(STEP, STOP).unwrap();
        let mut events = Vec::new();
        for source in circuit.source_timelines(timing) {
            let source = source.unwrap();
            let mut iter = source.waveform.breakpoints_in(0., STOP).unwrap();
            while let Some(event) = iter.next_event() {
                if event.kind.requires_landing() && event.time > 0. && event.time < STOP {
                    events.push(event.time);
                }
            }
        }
        events.sort_by(Real::total_cmp);
        events.dedup();
        assert_eq!(events.len(), expected.len(), "{tag}: {events:?}");
        for (&actual, &want) in events.iter().zip(expected) {
            assert!(
                (actual - want).abs() < 1e-15,
                "{tag}: {actual:e} != {want:e}"
            );
        }
        let rust = request.run(&mut circuit, &config.context()).unwrap();
        for (engine, plot) in [("C", c), ("Rust", rust)] {
            let times = plot.column("time").unwrap();
            for &want in expected {
                assert!(
                    times.iter().any(|time| (time.re - want).abs() < 1e-15),
                    "{tag}: {engine} missed corner {want:e}"
                );
            }
        }
    }
}

#[test]
#[ignore = "requires absolute NGSPICE_BIN; independent C timing oracle"]
fn nested_resolved_timeline_matches_independent_c_corners() {
    // Binary-exact times so these expected instants are independent of the
    // port's timing implementation. Two parameterized PWL delays, a zero-width
    // pulse whose rise/fall default to the .tran step, and coincident corners.
    let cards = "\
.param dt=0.125\n\
.subckt inner x y params: d=0\n\
v1 x 0 pwl(0 0 0.25 1 0.5 0) td={d}\n\
v2 y 0 pulse(0 1 0.25 0 0)\n\
.ends inner\n\
.subckt outer a b params: delay=0.5\n\
xinner a b inner d={delay}\n\
.ends outer\n\
x1 a b outer delay=0.5\n\
x2 c d outer delay=1\n\
r1 a 0 1k\nr2 b 0 1k\nr3 c 0 1k\nr4 d 0 1k\n";
    let (c_plot, msg) = run_c("nested-timeline", cards, "tran 0.125 2", "");
    let c_plot = c_plot.unwrap_or_else(|| panic!("no C plot: {msg}"));
    let netlist = rust_netlist(&format!("t\n{cards}.tran {{dt}} 2\n.end\n"));
    let config = RunConfig::from_netlist(&netlist).unwrap();
    let mut circuit = config.circuit(&netlist).unwrap();
    let plan = config.plan_for(&netlist.analyses[0]).unwrap();
    let timing = TransientTiming::new(0.125, 2.).unwrap();
    let sources = circuit
        .source_timelines(timing)
        .collect::<Result<Vec<_>, _>>()
        .unwrap();
    assert_eq!(sources.len(), 4);
    let mut exact_times = sources
        .iter()
        .flat_map(|source| source.waveform.breakpoints_in(0., 2.).unwrap())
        .filter(|t| *t > 0. && *t < 2.)
        .collect::<Vec<_>>();
    exact_times.sort_by(Real::total_cmp);
    exact_times.dedup();
    let expected = [0.25, 0.375, 0.5, 0.75, 1., 1.25, 1.5];
    assert_eq!(exact_times, expected);
    let rust_plot = plan.run(&mut circuit, &config.context()).unwrap();
    for (label, plot) in [("C", c_plot), ("Rust", rust_plot)] {
        let times = plot.column("time").unwrap();
        for event in expected {
            assert!(
                times.iter().any(|t| (t.re - event).abs() < 1e-12),
                "{label} did not accept expected corner {event}"
            );
        }
        // Numeric forcing checked against manually computed values at corners,
        // not by evaluating the shared Rust source descriptor again.
        for (node, t, value) in [
            ("v(a)", 0.75, 1.),
            ("v(c)", 1.25, 1.),
            ("v(b)", 0.375, 1.),
            ("v(d)", 0.5, 0.),
        ] {
            let index = times
                .iter()
                .position(|sample| (sample.re - t).abs() < 1e-12)
                .unwrap();
            let got = plot.value(node, index).unwrap().re;
            assert!(
                (got - value).abs() < 1e-9,
                "{label} {node} at {t}: {got} expected {value}"
            );
        }
    }
}
