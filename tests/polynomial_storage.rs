//! Analytic witnesses for explicit producer-selected charge and flux models.
use std::path::Path;

use ngspice_rs::analysis::{AnalysisInput, Plot, RunConfig};
use ngspice_rs::netlist::{Parser, source::parse_deck_text};
use ngspice_rs::primitives::{AnalysisKind, SpiceResult};

fn run(body: &str, kind: AnalysisKind, arguments: &[&str]) -> SpiceResult<Plot> {
    let deck = parse_deck_text(
        Path::new("polynomial.cir"),
        &format!("polynomial storage\n{body}\n.end\n"),
    );
    let netlist = Parser::new().parse_deck(&deck)?;
    let config = RunConfig::from_netlist(&netlist)?;
    let mut circuit = config.circuit(&netlist)?;
    let mut request =
        AnalysisInput::with_arguments(kind, arguments.iter().copied().filter(|arg| *arg != "uic"));
    request.uic = arguments.contains(&"uic");
    config.plan(request)?.run(&mut circuit, &config.context())
}

fn close(got: f64, expected: f64, tolerance: f64) {
    assert!(
        (got - expected).abs() <= tolerance,
        "{got:e} != {expected:e} (tol {tolerance:e})"
    );
}

#[test]
fn dc_and_ac_use_the_bias_differential_not_stored_over_coordinate() -> SpiceResult<()> {
    let body = "vc a 0 dc -2 ac 1\nil 0 b dc -0.5 ac 1\nc1 a 0 cm 1u\nl1 b 0 lm 2m\n.model cm pspice_cap(c=2 vc1=0.2 vc2=0.07)\n.model lm pspice_ind(l=3 il1=-0.1 il2=0.02)";
    let op = run(body, AnalysisKind::OperatingPoint, &[])?;
    close(op.value("i(vc)", 0).unwrap().re, 0., 1e-14);
    close(op.value("v(b)", 0).unwrap().re, 0., 1e-14);
    let ac = run(body, AnalysisKind::Ac, &["lin", "1", "1000", "1000"])?;
    let omega = std::f64::consts::TAU * 1000.;
    close(
        ac.value("i(vc)", 0).unwrap().im,
        -omega * 2e-6 * (1. - 0.4 + 0.28),
        1e-10,
    );
    close(
        ac.value("v(b)", 0).unwrap().im,
        omega * 6e-3 * (1. + 0.05 + 0.005),
        1e-8,
    );
    Ok(())
}

#[test]
fn signed_and_zero_effective_storage_are_not_clamped() -> SpiceResult<()> {
    let ac = run(
        "v1 a 0 dc 1 ac 1\nc1 a 0 cm -1u\n.model cm pspice_cap(c=2 vc1=0.5)\nv2 b 0 dc 1 ac 1\nc2 b 0 zero 1u\n.model zero pspice_cap(c=0 vc1=1)",
        AnalysisKind::Ac,
        &["lin", "1", "1000", "1000"],
    )?;
    close(
        ac.value("i(v1)", 0).unwrap().im,
        std::f64::consts::TAU * 1000. * 3e-6,
        1e-10,
    );
    close(ac.value("i(v2)", 0).unwrap().im, 0., 1e-14);
    for body in [
        "v1 a 0 1\nc1 a 0 cm 0\n.model cm pspice_cap(vc1=1)",
        "v1 a 0 1\nc1 a 0 cm\n.model cm pspice_cap(vc1=1)",
        "i1 0 a 1\nl1 a 0 lm 1m\nl2 b 0 1m\nr1 b 0 1k\nk1 l1 l2 0.5\n.model lm pspice_ind(il1=1)",
    ] {
        assert!(
            run(body, AnalysisKind::OperatingPoint, &[]).is_err(),
            "{body}"
        );
    }
    Ok(())
}

#[test]
fn transient_ramps_integrate_the_polynomial_and_reverse_sign() -> SpiceResult<()> {
    let body = "vc a 0 PWL(0 -1 1m 1)\nil 0 b PWL(0 -1 1m 1)\nc1 a 0 cm 1u\nl1 b 0 lm 1m\n.model cm pspice_cap(vc1=0.2 vc2=0.3)\n.model lm pspice_ind(il1=0.2 il2=0.3)";
    for method in ["method=trap", "method=gear"] {
        let tran = run(
            body,
            AnalysisKind::Transient,
            &["1u", "1m", "0", "1u", method],
        )?;
        let mut count = 0;
        for index in 0..tran.point_count() {
            let time = tran.value("time", index).unwrap().re;
            if !(20e-6..=980e-6).contains(&time) {
                continue;
            }
            let x = -1. + 2000. * time;
            let coefficient = 1. + 0.2 * x + 0.3 * x * x;
            close(
                tran.value("i(vc)", index).unwrap().re,
                -2e-3 * coefficient,
                2e-5,
            );
            close(
                tran.value("v(b)", index).unwrap().re,
                2. * coefficient,
                2e-2,
            );
            count += 1;
        }
        assert!(count > 20);
    }
    Ok(())
}

#[test]
fn nonlinear_uic_uses_the_charge_at_the_prescribed_voltage() -> SpiceResult<()> {
    // I=C0*(1+v)*dv/dt; integral from initial v=1 gives
    // v(t) = sqrt(4 + 2*I*t/C0)-1.
    let tran = run(
        "i1 0 a 1m\nc1 a 0 cm 1u ic=1\n.model cm pspice_cap(vc1=1)",
        AnalysisKind::Transient,
        &["1u", "1m", "0", "1u", "uic"],
    )?;
    for index in 0..tran.point_count() {
        let time = tran.value("time", index).unwrap().re;
        close(
            tran.value("v(a)", index).unwrap().re,
            (4. + 2000. * time).sqrt() - 1.,
            5e-4,
        );
    }
    // V=L0*(1+i)*di/dt, with an explicit initial current.
    let tran = run(
        "v1 a 0 1\nl1 a 0 lm 1m ic=1\n.model lm pspice_ind(il1=1)",
        AnalysisKind::Transient,
        &["1u", "1m", "0", "1u", "uic"],
    )?;
    for index in 0..tran.point_count() {
        let time = tran.value("time", index).unwrap().re;
        close(
            tran.value("i(l1)", index).unwrap().re,
            (4. + 2000. * time).sqrt() - 1.,
            5e-4,
        );
    }
    Ok(())
}

#[test]
fn model_temperature_and_parallel_copy_coordinates() -> SpiceResult<()> {
    let body = "v1 a 0 dc 2 ac 1\nc1 a 0 cm 1u m=2\n.model cm pspice_cap(c=2 vc1=0.5 tc1=0.01 tnom=20 temp=30)\ni1 0 b dc 2 ac 1\nl1 b 0 lm 1m m=2\n.model lm pspice_ind(l=2 il1=0.5 tc1=0.01 tnom=20 temp=30)";
    let ac = run(body, AnalysisKind::Ac, &["lin", "1", "1000", "1000"])?;
    let omega = std::f64::consts::TAU * 1000.;
    close(ac.value("i(v1)", 0).unwrap().im, -omega * 8.8e-6, 1e-10);
    close(ac.value("v(b)", 0).unwrap().im, omega * 1.65e-3, 1e-8);
    Ok(())
}
