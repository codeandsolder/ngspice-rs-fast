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
    // Signed initial coordinates must not be interpreted as magnitudes.
    // D(x)=B*(1+x/4), so (x+4)^2 starts at 9 and increases at 8000/s.
    for (body, observed) in [
        (
            "i1 0 a 1m\nc1 a 0 cm 1u ic=-1\n.model cm pspice_cap(vc1=0.25)",
            "v(a)",
        ),
        (
            "v1 a 0 1\nl1 a 0 lm 1m ic=-1\n.model lm pspice_ind(il1=0.25)",
            "i(l1)",
        ),
    ] {
        let tran = run(
            body,
            AnalysisKind::Transient,
            &["1u", "1m", "0", "1u", "uic"],
        )?;
        for index in 0..tran.point_count() {
            let time = tran.value("time", index).unwrap().re;
            close(
                tran.value(observed, index).unwrap().re,
                (9. + 8000. * time).sqrt() - 4.,
                5e-4,
            );
        }
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

#[test]
fn actual_pspice_preparation_preserves_scoped_model_parameters()
-> Result<(), Box<dyn std::error::Error>> {
    use ngspice_rs_preprocess::{
        Cancellation, InputFormat, Options, Request, SourceInput, prepare,
    };
    let directory = tempfile::tempdir()?;
    let path = directory.path().join("source.cir");
    std::fs::write(
        &path,
        "scoped PSpice storage\nv1 a 0 dc 2 ac 1\nv2 b 0 dc 2 ac 1\n.subckt cell p n params: factor=1 slope=.1\n.model base CAP(C={factor} VC1={slope} T_MEASURED=20 T_ABS=25 TC1=.01)\n.model cm AKO:base CAP(VC2=.03 T_REL_LOCAL=5)\nc1 p n cm 1u\n.ends cell\nx1 a 0 cell factor=2 slope=.2\nx2 b 0 cell factor=3 slope=.4\n.ac lin 1 1k 1k\n.end\n",
    )?;
    let prepared = prepare(
        &Request {
            format: InputFormat::PspiceDeck,
            input: SourceInput::File(path),
            libraries: Vec::new(),
            search_paths: Vec::new(),
        },
        &Options::default(),
        &Cancellation::default(),
    )?;
    let text = std::str::from_utf8(&prepared.primary.bytes)?;
    let parsed = Parser::new().parse_deck(&parse_deck_text(Path::new("prepared.cir"), text))?;
    let config = RunConfig::from_netlist(&parsed)?;
    let mut circuit = config.circuit(&parsed)?;
    let plot = config.plans_for(&parsed.analyses)?[0].run(&mut circuit, &config.context())?;
    let omega = std::f64::consts::TAU * 1000.;
    close(
        plot.value("i(v1)", 0).unwrap().im,
        -omega * 2e-6 * 1.1 * 1.52,
        1e-10,
    );
    close(
        plot.value("i(v2)", 0).unwrap().im,
        -omega * 3e-6 * 1.1 * 1.92,
        1e-10,
    );
    Ok(())
}

#[test]
fn ramp_power_integrates_to_stored_energy() -> SpiceResult<()> {
    let tran = run(
        "vc a 0 PWL(0 0 1m 2)\nil 0 b PWL(0 0 1m 2)\nc1 a 0 cm 1u\nl1 b 0 lm 1m\n.model cm pspice_cap(vc1=0.2 vc2=0.3)\n.model lm pspice_ind(il1=0.2 il2=0.3)",
        AnalysisKind::Transient,
        &["1u", "1m", "0", "1u"],
    )?;
    let mut energy = [0.; 2];
    let mut previous = None;
    for index in 0..tran.point_count() {
        let t = tran.value("time", index).unwrap().re;
        let p = [
            -tran.value("v(a)", index).unwrap().re * tran.value("i(vc)", index).unwrap().re,
            tran.value("v(b)", index).unwrap().re * tran.value("i(l1)", index).unwrap().re,
        ];
        if let Some((last_t, last_p)) = previous {
            let last_p: [f64; 2] = last_p;
            for k in 0..2 {
                energy[k] += (t - last_t) * (p[k] + last_p[k]) / 2.;
            }
        }
        previous = Some((t, p));
    }
    let primitive = 0.5 * 4. + 0.2 / 3. * 8. + 0.3 / 4. * 16.;
    close(energy[0], 1e-6 * primitive, 2e-8);
    close(energy[1], 1e-3 * primitive, 2e-5);
    Ok(())
}

#[test]
fn candidate_and_rejected_trial_preserve_original_history() -> SpiceResult<()> {
    use ngspice_rs::devices::{AnalysisMode, LoadRequest, ModelContext};
    use ngspice_rs::maths::{
        SparseMatrix, Vector,
        integrator::{DEFAULT_XMU, IntegrationMethod, StepHistory},
    };
    let deck = parse_deck_text(
        Path::new("candidate.cir"),
        "candidate\nc1 a 0 cm 1u\n.model cm pspice_cap(vc1=0.2 vc2=0.3)\n.end\n",
    );
    let netlist = Parser::new().parse_deck(&deck)?;
    let config = RunConfig::from_netlist(&netlist)?;
    let mut circuit = config.circuit(&netlist)?;
    circuit.finalize()?;
    let target = circuit.bind_instance_target("c1", "capacitance")?;
    let original = ModelContext::default();
    let changed = original.with_instance_override(target.at(2e-6)?)?;
    let invalid = original.with_instance_override(target.at(0.)?)?;
    let mut history = circuit.state_history()?;
    let x = Vector::from_slice(&[1., 0.]);
    let coefficients =
        StepHistory::new().trial(IntegrationMethod::Trapezoidal, 1, 1e-6, DEFAULT_XMU)?;
    let n = circuit.unknown_count();
    let run = circuit.prepare_run();
    let request = LoadRequest {
        mode: AnalysisMode::OperatingPoint,
        solution: &x,
        model_context: &original,
        integration: None,
        history: &history,
        forcing: None,
    };
    let mut first = history.trial();
    run.load(
        &request,
        &mut SparseMatrix::new(n, n),
        &mut Vector::zeros(n),
        &mut first,
    )?;
    close(first.values()[0], 1.2e-6, 1e-20);
    circuit.accept_point(&x, None, &mut history, first)?;
    let accepted = history.accepted(1).unwrap().to_vec();
    let x = Vector::from_slice(&[2., 0.]);
    for (context, base) in [(&changed, 2e-6), (&original, 1e-6)] {
        let mut trial = history.trial();
        let request = LoadRequest {
            mode: AnalysisMode::Transient {
                time: 1e-6,
                dt: 1e-6,
            },
            solution: &x,
            model_context: context,
            integration: Some(&coefficients),
            history: &history,
            forcing: None,
        };
        run.load(
            &request,
            &mut SparseMatrix::new(n, n),
            &mut Vector::zeros(n),
            &mut trial,
        )?;
        close(trial.values()[0], base * 2. * (1. + 0.2 + 0.4), 1e-19);
        assert_eq!(history.accepted(1).unwrap(), accepted);
        let invalid = LoadRequest {
            model_context: &invalid,
            ..request
        };
        assert!(
            run.load(
                &invalid,
                &mut SparseMatrix::new(n, n),
                &mut Vector::zeros(n),
                &mut history.trial()
            )
            .is_err()
        );
    }
    Ok(())
}

#[test]
fn differential_crossings_are_signed_and_do_not_create_artificial_loading() -> SpiceResult<()> {
    let tran = run(
        "vc a 0 PWL(0 0 1m 2)\nil 0 b PWL(0 0 1m 2)\nc1 a 0 cm 1u\nl1 b 0 lm 1m\n.model cm pspice_cap(vc1=-1)\n.model lm pspice_ind(il1=-1)",
        AnalysisKind::Transient,
        &["1u", "1m", "0", "1u"],
    )?;
    let mut before = false;
    let mut after = false;
    for index in 0..tran.point_count() {
        let t = tran.value("time", index).unwrap().re;
        if !(20e-6..=980e-6).contains(&t) {
            continue;
        }
        let coefficient = 1. - 2000. * t;
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
        before |= t < 0.4e-3;
        after |= t > 0.6e-3;
    }
    assert!(before && after);
    Ok(())
}

#[test]
fn physical_current_observers_share_the_native_storage_branch() -> SpiceResult<()> {
    let ac = run(
        "vc a 0 dc 2 ac 1\nc1 a 0 cm 1u\n.model cm pspice_cap(vc1=0.5)\nbmon obs 0 v=i(c1)",
        AnalysisKind::Ac,
        &["lin", "1", "1000", "1000"],
    )?;
    let expected = std::f64::consts::TAU * 1000. * 2e-6;
    close(ac.value("i(c1)", 0).unwrap().im, expected, 1e-10);
    close(ac.value("v(obs)", 0).unwrap().im, expected, 1e-10);
    close(ac.value("i(vc)", 0).unwrap().im, -expected, 1e-10);
    Ok(())
}
