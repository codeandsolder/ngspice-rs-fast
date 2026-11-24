//! Circuit-coupled VDMOS DC goldens from ngspice 47.
use ngspice_rs::analysis::{AnalysisContext, AnalysisInput};
use ngspice_rs::devices::Circuit;
use ngspice_rs::netlist::{Parser, source::parse_deck_text};
use ngspice_rs::primitives::{AnalysisKind, SpiceResult};
use std::path::Path;

fn run(source: &str) -> SpiceResult<ngspice_rs::analysis::Plot> {
    let card = source.split(".control").next().unwrap_or(source);
    let deck = parse_deck_text(Path::new("vdmos.cir"), &format!("{card}\n.op\n.end\n"));
    let parsed = Parser::new().parse_deck(&deck)?;
    let mut circuit = Circuit::from_netlist(&parsed)?;
    AnalysisInput::with_arguments(AnalysisKind::OperatingPoint, [""; 0])
        .resolve()?
        .run(&mut circuit, &AnalysisContext::default())
}
fn close(name: &str, actual: f64, c: f64) {
    let tolerance = 2e-4_f64.mul_add(c.abs(), 1e-5);
    assert!(
        (actual - c).abs() <= tolerance,
        "{name} Rust {actual:e}, ngspice47 {c:e}, tolerance {tolerance:e}"
    );
}
#[test]
fn circuit_coupled_vdmos_nmos_pmos_series_and_diode() {
    for (name, source, drain, supply_current) in [
        (
            "N",
            include_str!("fixtures/vdmos/n_load.cir"),
            0.06158834,
            -1.19384,
        ),
        (
            "P",
            include_str!("fixtures/vdmos/p_load.cir"),
            -0.126767,
            1.187323,
        ),
        (
            "series",
            include_str!("fixtures/vdmos/n_series.cir"),
            0.3009264,
            -1.16991,
        ),
        (
            "diode",
            include_str!("fixtures/vdmos/n_diode.cir"),
            -0.598882,
            0.01117514,
        ),
    ] {
        let plot = run(source).unwrap_or_else(|error| panic!("{name}: {error}"));
        let v = plot
            .value("v(drain)", 0)
            .unwrap_or_else(|| panic!("{name}: missing Vdrain"));
        let i = plot
            .value("i(vdd)", 0)
            .unwrap_or_else(|| panic!("{name}: missing source current"));
        close(&format!("{name} Vdrain"), v.re, drain);
        close(&format!("{name} supply current"), i.re, supply_current);
    }
}
#[test]
fn unsupported_thermal_network_cannot_be_silently_ignored() {
    let source = "VDMOS thermal refusal\nvdd d 0 1\nvg g 0 4\nm1 d g 0 dut thermal\n.model dut vdmos(vto=3 kp=20 rthjc=1 rthca=2 cthj=10u)\n";
    assert!(run(source).is_err());
}

fn ac(source: &str, frequency: &str) -> SpiceResult<ngspice_rs::analysis::Plot> {
    use ngspice_rs::analysis::RunConfig;
    let card = source.split(".control").next().unwrap_or(source);
    let deck = parse_deck_text(
        Path::new("vdmos-ac.cir"),
        &format!("{card}\n.ac lin 1 {frequency} {frequency}\n.end\n"),
    );
    let parsed = Parser::new().parse_deck(&deck)?;
    let config = RunConfig::from_netlist(&parsed)?;
    let request = config.plan_for(&parsed.analyses[0])?;
    let mut circuit = config.circuit(&parsed)?;
    request.run(&mut circuit, &config.context())
}

#[test]
fn native_circuit_coupled_small_signal_n_p_and_series() {
    for (name, source, real, imag) in [
        (
            "N",
            include_str!("fixtures/vdmos/n_ac.cir"),
            -0.0652793,
            7.197459e-5,
        ),
        (
            "P",
            include_str!("fixtures/vdmos/p_ac.cir"),
            -0.143518,
            1.673841e-4,
        ),
        (
            "series",
            include_str!("fixtures/vdmos/n_series_ac.cir"),
            -0.0777147,
            4.281995e-4,
        ),
    ] {
        let plot = ac(source, "100k").unwrap_or_else(|error| panic!("AC {name}: {error}"));
        let v = plot
            .value("v(drain)", 0)
            .unwrap_or_else(|| panic!("AC {name}: missing drain"));
        close(&format!("AC {name} real"), v.re, real);
        close(&format!("AC {name} imaginary"), v.im, imag);
    }
}

fn transient(source: &str, timing: &str) -> SpiceResult<ngspice_rs::analysis::Plot> {
    use ngspice_rs::analysis::RunConfig;
    let card = source.split(".control").next().unwrap_or(source);
    let deck = parse_deck_text(
        Path::new("vdmos-tran.cir"),
        &format!("{card}\n.tran {timing}\n.end\n"),
    );
    let parsed = Parser::new().parse_deck(&deck)?;
    let config = RunConfig::from_netlist(&parsed)?;
    let request = config.plan_for(&parsed.analyses[0])?;
    let mut circuit = config.circuit(&parsed)?;
    request.run(&mut circuit, &config.context())
}

fn interpolation(plot: &ngspice_rs::analysis::Plot, variable: &str, at: f64) -> f64 {
    for current in 1..plot.point_count() {
        let t0 = plot.value("time", current - 1).unwrap().re;
        let t1 = plot.value("time", current).unwrap().re;
        if t0 <= at && at <= t1 {
            let y0 = plot.value(variable, current - 1).unwrap().re;
            let y1 = plot.value(variable, current).unwrap().re;
            return (at - t0).mul_add((y1 - y0) / (t1 - t0), y0);
        }
    }
    panic!("requested time {at:e} not covered by transient result")
}

#[test]
fn circuit_coupled_n_p_gate_switching_matches_ngspice47_charge_transient() {
    for (name, source, expected) in [
        (
            "N",
            include_str!("fixtures/vdmos/n_switch.cir"),
            [
                12.0, 12.0022, 0.0637966, 0.0615883, 0.2853, 3.01956, 10.3514,
            ],
        ),
        (
            "P",
            include_str!("fixtures/vdmos/p_switch.cir"),
            [
                -12.0, -12.0022, -4.88255, -0.126767, -0.354727, -3.07363, -10.3614,
            ],
        ),
    ] {
        let plot = transient(source, "5n 6u 0 5n")
            .unwrap_or_else(|error| panic!("transient {name}: {error}"));
        assert!(
            plot.point_count() >= 100,
            "{name}: transient waveform truncated"
        );
        for (time, reference) in [0.5e-6, 1.03e-6, 1.2e-6, 2e-6, 3.03e-6, 3.3e-6, 5e-6]
            .into_iter()
            .zip(expected)
        {
            let actual = interpolation(&plot, "v(drain)", time);
            let tolerance = 4e-3_f64.mul_add(f64::abs(reference), 0.04);
            assert!(
                (actual - reference).abs() <= tolerance,
                "transient {name} t={time:e}: Rust {actual:e}, C47 {reference:e}, tol {tolerance:e}"
            );
        }
    }
}

#[test]
fn circuit_coupled_soft_recovery_n_p_matches_ngspice47_transient() {
    let times = [180e-9, 201e-9, 207e-9, 220e-9, 260e-9, 400e-9];
    let n_voltage = [
        -0.549495, -0.504010, 0.0924717, 0.0995240, 0.0998249, 0.0999947,
    ];
    let n_supply = [
        0.00168459,
        -0.280033,
        -0.0250944,
        -0.00158655,
        -0.000583648,
        -0.0000176238,
    ];
    for (name, deck, sign) in [
        ("N", include_str!("fixtures/vdmos/n_recovery.cir"), 1.0),
        ("P", include_str!("fixtures/vdmos/p_recovery.cir"), -1.0),
    ] {
        let plot = transient(deck, "0.5n 700n 0 0.5n")
            .unwrap_or_else(|error| panic!("recovery {name}: {error}"));
        assert!(
            plot.point_count() > 500,
            "{name}: truncated recovery waveform"
        );
        for (index, t) in times.into_iter().enumerate() {
            let v = interpolation(&plot, "v(d)", t);
            let i = interpolation(&plot, "i(vdrive)", t);
            let reference_v: f64 = sign * n_voltage[index];
            let reference_i: f64 = sign * n_supply[index];
            let v_tolerance = 0.015_f64.mul_add(reference_v.abs(), 0.002);
            let i_tolerance = 0.08_f64.mul_add(reference_i.abs(), 3e-6);
            assert!(
                (v - reference_v).abs() <= v_tolerance,
                "{name} reverse-recovery V(d) t={t:e}: Rust {v:e}, C {reference_v:e}, tol {v_tolerance:e}"
            );
            assert!(
                (i - reference_i).abs() <= i_tolerance,
                "{name} reverse-recovery I(vdrive) t={t:e}: Rust {i:e}, C {reference_i:e}, tol {i_tolerance:e}"
            );
        }
    }
}

#[test]
fn dynamic_soft_recovery_small_signal_matches_ngspice47() {
    for (name, source) in [
        ("N", include_str!("fixtures/vdmos/n_recovery_ac.cir")),
        ("P", include_str!("fixtures/vdmos/p_recovery_ac.cir")),
    ] {
        for (frequency, real_ref, imaginary_ref) in [
            ("1Meg", 0.9789868, -0.0131346),
            ("10Meg", 0.9518050, -0.0708756),
        ] {
            let plot = ac(source, frequency)
                .unwrap_or_else(|error| panic!("Qp AC {name} {frequency}: {error}"));
            let voltage = plot
                .value("v(d)", 0)
                .unwrap_or_else(|| panic!("Qp AC {name} {frequency}: no drain result"));
            let tolerance = 5e-4;
            assert!(
                (voltage.re - real_ref).abs() < tolerance,
                "Qp AC {name} {frequency}: real Rust {} C {real_ref}",
                voltage.re
            );
            assert!(
                (voltage.im - imaginary_ref).abs() < tolerance,
                "Qp AC {name} {frequency}: imag Rust {} C {imaginary_ref}",
                voltage.im
            );
        }
    }
}
