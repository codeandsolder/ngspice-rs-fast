//! Source-selected diode integration. The physical references are independent
//! SIMetrix Elements measurements and analytic electrical witnesses.
use std::{path::Path, sync::Arc};

use model_kernel_diode::MODEL_FAMILY;
use ngspice_rs::{
    analysis::{AnalysisContext, AnalysisInput, bias::solve_dc, newton::NewtonOptions},
    devices::{
        Circuit, ModelContext,
        noise::{NoiseContext, NoiseKind},
        nonlinear::Diode,
    },
    netlist::{Parser, ast::Netlist, semantic_diff, source::parse_deck_text, write_netlist},
    primitives::{AnalysisKind, Real},
};
use ngspice_rs_preprocess::{Cancellation, InputFormat, Options, Request, SourceInput, prepare};

fn parse(body: &str) -> Netlist {
    Parser::new()
        .parse_deck(&parse_deck_text(Path::new("diode.cir"), body))
        .unwrap()
}

fn projected(source: &str) -> Netlist {
    let dir = tempfile::tempdir().unwrap();
    let bundle = prepare(
        &Request {
            format: InputFormat::SimetrixDeck,
            input: SourceInput::Bytes {
                origin: dir.path().join("source.net"),
                bytes: Arc::from(source.as_bytes()),
            },
            libraries: Vec::new(),
            search_paths: Vec::new(),
        },
        &Options::default(),
        &Cancellation::default(),
    )
    .unwrap();
    for artifact in bundle.artifacts() {
        std::fs::write(dir.path().join(&artifact.filename), &artifact.bytes).unwrap();
    }
    Parser::new()
        .parse_file(dir.path().join(&bundle.primary.filename))
        .unwrap()
}

fn close(actual: Real, expected: Real, relative: Real, absolute: Real) {
    assert!(
        (actual - expected).abs() <= relative.mul_add(expected.abs(), absolute),
        "{actual:e} != {expected:e}"
    );
}

fn dc_current(circuit: &Circuit, context: &ModelContext, source: &str) -> Real {
    let options = NewtonOptions {
        reltol: 1e-10,
        abstol: 1e-14,
        vntol: 1e-10,
        ..NewtonOptions::default()
    };
    let solved = solve_dc(circuit, context, &options, &[], None, None).unwrap();
    let index = circuit
        .devices()
        .iter()
        .position(|device| device.name() == source)
        .unwrap();
    -solved
        .values
        .get(circuit.branch_rows(index).unwrap().start)
        .unwrap()
}

#[test]
fn source_projection_matches_measured_high_injection_and_recombination() {
    // Independently authored SIMetrix Elements 9.40d fixtures:
    // model-kernel-diode/tests/fixtures/vendor-authored.tsv, revision
    // 46cb5924387c5ed92cb03376453bd6140103f282, diode-equations-27-5-*.
    let netlist = projected(
        "* diode witness\n*#SIMETRIX\nV1 a 0 0.65\nV2 b 0 -3\nD1 a 0 DM\nD2 b 0 DM\n.model DM D(IS=1p N=1 ISR=100n NR=2 VJ=0.7 M=0.4 IKF=1m T_MEASURED=27)\n.op\n.end\n",
    );
    assert!(
        netlist
            .models
            .iter()
            .any(|model| model.base.eq_ignore_ascii_case(MODEL_FAMILY))
    );
    let context = ModelContext::default().with_gmin(1e-15);
    let circuit = Circuit::from_netlist_with_context(&netlist, &context).unwrap();
    close(
        dc_current(&circuit, &context, "v1"),
        0.02043710934,
        2e-8,
        1e-14,
    );
    close(
        dc_current(&circuit, &context, "v2"),
        -1.9465217221e-7,
        2e-8,
        1e-14,
    );
    let diodes: Vec<_> = circuit
        .devices()
        .iter()
        .filter(|device| device.designator() == 'd')
        .collect();
    assert_eq!(diodes.len(), 2);
    assert!(diodes[0].model_name().is_some());
    assert_eq!(
        diodes[0].prepared_batch_key(),
        diodes[1].prepared_batch_key()
    );
    assert!(diodes[0].prepared_batch_key().is_some());
}

#[test]
fn source_projection_matches_measured_complex_ac_and_native_d_is_unchanged() {
    // Independent witness ac-27-hm2v4-1000000.0 in vendor-authored.tsv.
    let netlist = projected(
        "* diode AC witness\n*#SIMETRIX\nV1 a 0 DC 0.65 AC 1\nD1 a 0 DM\n.model DM D(IS=1p ISR=100n NR=2 VJ=0.7 M=0.4 CJO=100p TT=20n)\n.ac lin 2 1Meg 2Meg\n.end\n",
    );
    let context = AnalysisContext {
        gmin: 1e-15,
        ..AnalysisContext::default()
    };
    let mut circuit =
        Circuit::from_netlist_with_context(&netlist, &context.model_context()).unwrap();
    let plots = AnalysisInput::with_arguments(AnalysisKind::Ac, ["lin", "2", "1meg", "2meg"])
        .resolve()
        .unwrap()
        .run_plots(&mut circuit, &context)
        .unwrap();
    let current = plots[0].value("i(v1)", 0).unwrap();
    close(current.re.hypot(current.im), 3.3757714353, 2e-7, 1e-12);
    close(
        (-current.im).atan2(-current.re).to_degrees(),
        7.1812044151,
        2e-7,
        1e-8,
    );
    let native = Circuit::from_netlist(&parse(
        "native\nV1 a 0 0.65\nD1 a 0 DM\n.model DM D(IS=1p)\n.end\n",
    ))
    .unwrap();
    let diode = native
        .devices()
        .iter()
        .find(|device| device.designator() == 'd')
        .unwrap();
    assert!((diode.as_ref() as &dyn std::any::Any).is::<Diode>());
    assert!(diode.prepared_batch_key().is_none());
}

#[test]
fn source_geometry_temperature_and_context_rebinding_preserve_artifact_sharing() {
    let netlist = parse(
        "geometry and scope\nV1 a 0 1\nV2 b 0 1\nV3 c 0 1\nD1 a 0 DM area=2 m=3 dtemp=5\nD2 b 0 DM area=4 m=2\nD3 c 0 DM temp=50 dtemp=12\n.model DM simetrix_diode_level1(IS=0 RS=100 T_ABS=35)\n.end\n",
    );
    let initial = ModelContext::default().with_gmin(1e-3);
    let circuit = Circuit::from_netlist_with_context(&netlist, &initial).unwrap();
    let diode = |name: &str| {
        circuit
            .devices()
            .iter()
            .find(|device| device.name() == name)
            .unwrap()
    };
    assert_eq!(
        diode("d1").prepared_batch_key(),
        diode("d2").prepared_batch_key()
    );
    assert_ne!(
        diode("d1").prepared_batch_key(),
        diode("d3").prepared_batch_key()
    );
    assert_eq!(diode("d1").terminals().len(), 3);
    assert_eq!(diode("d1").external_terminals().len(), 2);
    assert_eq!(
        diode("d1").observation_parameter("temp", &initial).unwrap(),
        Some(40.)
    );
    assert_eq!(
        diode("d3").observation_parameter("temp", &initial).unwrap(),
        Some(50.)
    );
    assert_eq!(
        diode("d1").observation_parameter("m", &initial).unwrap(),
        Some(3.)
    );
    let expected = |area: Real, multiplicity: Real, gmin: Real| {
        (1. / (multiplicity * gmin) + 100. / (area * multiplicity)).recip()
    };
    close(
        dc_current(&circuit, &initial, "v1"),
        expected(2., 3., initial.gmin),
        1e-10,
        1e-14,
    );
    let changed = initial.with_gmin(2e-3);
    close(
        dc_current(&circuit, &changed, "v1"),
        expected(2., 3., changed.gmin),
        1e-10,
        1e-14,
    );
    let target = circuit.bind_instance_target("d1", "m").unwrap();
    let candidate = changed
        .with_instance_override(target.at(5.).unwrap())
        .unwrap();
    close(
        dc_current(&circuit, &candidate, "v1"),
        expected(2., 5., changed.gmin),
        1e-10,
        1e-14,
    );
    close(
        dc_current(&circuit, &candidate, "v2"),
        expected(4., 2., changed.gmin),
        1e-10,
        1e-14,
    );
    close(
        dc_current(&circuit, &initial, "v1"),
        expected(2., 3., initial.gmin),
        1e-10,
        1e-14,
    );
}

#[test]
fn lexical_model_shadowing_and_actuals_select_distinct_physical_declarations() {
    let netlist = parse(
        "lexical diode\nV1 a 0 1\nV2 b 0 1\nV3 c 0 1\nD3 c 0 DM\nX1 a 0 cell params: r=100\nX2 b 0 cell params: r=200\n.model DM simetrix_diode_level1(IS=0 RS=999)\n.subckt cell p n params: r=300\n.model DM simetrix_diode_level1(IS=0 RS={r})\nD1 p n DM\n.ends cell\n.end\n",
    );
    let context = ModelContext::default().with_gmin(1e-3);
    let circuit = Circuit::from_netlist_with_context(&netlist, &context).unwrap();
    close(
        dc_current(&circuit, &context, "v1"),
        1. / 1100.,
        1e-10,
        1e-14,
    );
    close(
        dc_current(&circuit, &context, "v2"),
        1. / 1200.,
        1e-10,
        1e-14,
    );
    close(
        dc_current(&circuit, &context, "v3"),
        1. / 1999.,
        1e-10,
        1e-14,
    );
    let serialized = write_netlist(&netlist).unwrap();
    let roundtrip = parse(&serialized);
    assert!(semantic_diff(&netlist, &roundtrip).is_none());
}

#[test]
fn transient_charge_and_noise_are_physical() {
    let context = AnalysisContext {
        gmin: 1e-12,
        ..AnalysisContext::default()
    };
    let mut circuit = Circuit::from_netlist_with_context(&parse("linear charge witness\nV1 a 0 PWL(0 0 1u 0.1 2u 0.2)\nD1 a 0 DM\n.model DM simetrix_diode_level1(IS=0 CJO=1n M=0 VJ=1)\n.end\n"), &context.model_context()).unwrap();
    let plots = AnalysisInput::with_arguments(AnalysisKind::Transient, ["10n", "2u", "0", "10n"])
        .resolve()
        .unwrap()
        .run_plots(&mut circuit, &context)
        .unwrap();
    let mut checked = 0;
    for row in 0..plots[0].point_count() {
        let time = plots[0].value("time", row).unwrap().re;
        if (0.2e-6..0.8e-6).contains(&time) {
            let current = -plots[0].value("i(v1)", row).unwrap().re;
            close(current, 1e-4 + time * 1e5 * context.gmin, 1e-6, 1e-12);
            checked += 1;
        }
    }
    assert!(checked > 10);
    let netlist = parse(
        "noise witness\nV1 a 0 0.65\nD1 a 0 DM\n.model DM simetrix_diode_level1(IS=1p KF=1e-12 AF=1)\n.end\n",
    );
    let model_context = ModelContext::default().with_gmin(1e-15).with_frequency(1e3);
    let circuit = Circuit::from_netlist_with_context(&netlist, &model_context).unwrap();
    let solved = solve_dc(
        &circuit,
        &model_context,
        &NewtonOptions::default(),
        &[],
        None,
        None,
    )
    .unwrap();
    let device = circuit
        .devices()
        .iter()
        .find(|device| device.designator() == 'd')
        .unwrap();
    let mut noise = Vec::new();
    device
        .noise(
            &NoiseContext {
                model_context: &model_context,
                unknowns: circuit.unknowns(),
                bias: &solved.values,
                states: None,
                controls: &[],
            },
            &mut noise,
        )
        .unwrap();
    let current = dc_current(&circuit, &model_context, "v1").abs();
    let mut densities: Vec<_> = noise
        .iter()
        .map(|source| {
            let NoiseKind::Generalized { density, .. } = source.kind else {
                panic!("expected generalized source");
            };
            density
        })
        .collect();
    densities.sort_by(Real::total_cmp);
    assert_eq!(densities.len(), 2);
    close(densities[0], 2. * 1.602_191_8e-19 * current, 1e-9, 1e-30);
    close(densities[1], 1e-12 * current / 1e3, 1e-9, 1e-30);
}

#[test]
fn invalid_source_models_and_topology_changes_are_explicit() {
    for (model, instance) in [
        ("IS=1p LEVEL=1.2", ""),
        ("IS=1p unknown=1", ""),
        ("IS=1p", " area=-1"),
        ("IS=1p", " off"),
    ] {
        let body = format!(
            "invalid\nV1 a 0 0.2\nD1 a 0 DM{instance}\n.model DM simetrix_diode_level1({model})\n.end\n"
        );
        assert!(
            Parser::new()
                .parse_deck(&parse_deck_text(Path::new("invalid.cir"), &body))
                .and_then(|netlist| Circuit::from_netlist(&netlist))
                .is_err(),
            "{body}"
        );
    }
    let circuit = Circuit::from_netlist(&parse("topology\nV1 a 0 0.2\nD1 a 0 DM\n.model DM simetrix_diode_level1(IS=0 RS=100 TRS1=-0.015625 TNOM=27)\n.end\n")).unwrap();
    // 64 K times -1/64 is exactly -1 even with fused multiply-add.
    let hot = ModelContext::new(91., 27.);
    let result = solve_dc(&circuit, &hot, &NewtonOptions::default(), &[], None, None);
    assert!(
        result.is_err(),
        "temperature-induced zero series resistance changes topology"
    );
}

#[test]
fn series_device_observations_and_noise_transfer_use_external_terminals() {
    let context = AnalysisContext {
        gmin: 1e-3,
        ..AnalysisContext::default()
    };
    let mut circuit = Circuit::from_netlist_with_context(
        &parse("terminal observation\nV1 a 0 1\nD1 a 0 DM\n.model DM simetrix_diode_level1(IS=0 RS=100)\n.end\n"),
        &context.model_context(),
    ).unwrap();
    circuit
        .set_observations(&["@d1[i]".into(), "@d1[temp]".into(), "@d1[p]".into()])
        .unwrap();
    let plots =
        AnalysisInput::with_arguments(AnalysisKind::OperatingPoint, std::iter::empty::<&str>())
            .resolve()
            .unwrap()
            .run_plots(&mut circuit, &context)
            .unwrap();
    for column in ["@d1[i]", "@d1[p]"] {
        close(
            plots[0].value(column, 0).unwrap().re,
            1. / 1100.,
            1e-10,
            1e-14,
        );
    }
    close(plots[0].value("@d1[temp]", 0).unwrap().re, 27., 1e-12, 0.);

    // At zero bias only the physical series resistor and the external
    // resistor generate noise. GMIN is a numerical conductance, not a
    // thermal noise generator. Check the transfer from the hidden node.
    let mut circuit = Circuit::from_netlist_with_context(
        &parse("series noise\nV1 in 0 DC 0 AC 1\nR1 in out 1k\nD1 out 0 DM\n.model DM simetrix_diode_level1(IS=0 RS=100)\n.end\n"),
        &context.model_context(),
    ).unwrap();
    let plots = AnalysisInput::with_arguments(
        AnalysisKind::Noise,
        ["v(out)", "v1", "lin", "2", "1k", "2k"],
    )
    .resolve()
    .unwrap()
    .run_plots(&mut circuit, &context)
    .unwrap();
    let temperature = 300.15;
    let external_gain = 1100. / 2100_f64;
    let series_gain = 1000. / 2100_f64;
    let density = 4.
        * temperature
        * (ngspice_rs::devices::noise::BOLTZMANN * 1000. * external_gain.powi(2)
            + model_kernel_diode::BOLTZMANN * 100. * series_gain.powi(2));
    for row in 0..plots[0].point_count() {
        close(
            plots[0].value("onoise_spectrum", row).unwrap().re,
            density.sqrt(),
            1e-9,
            1e-18,
        );
    }
}

#[test]
fn direct_newton_limits_forward_and_avalanche_junctions_in_scalar_and_tiles() {
    use ngspice_rs::analysis::bias::{DcOutcome, DcSettings, DcStrategy, solve_dc_with};
    let model = ".model dm simetrix_diode_level1(IS=1p N=1 BV=5 IBV=1m NBV=1 RS=1k)";
    let context = ModelContext::default().with_gmin(1e-15);
    let settings = DcSettings {
        newton: NewtonOptions {
            reltol: 1e-10,
            abstol: 1e-14,
            vntol: 1e-10,
            ..NewtonOptions::default()
        },
        continuation: ngspice_rs::analysis::bias::ContinuationPolicy::default(),
    };
    let solve = |deck: &str| {
        let circuit = Circuit::from_netlist_with_context(&parse(deck), &context).unwrap();
        let result = solve_dc_with(&circuit, &context, &settings, &[], None, None).unwrap();
        assert_eq!(
            result.report.outcome,
            DcOutcome::Converged(DcStrategy::Direct)
        );
        circuit
            .devices()
            .iter()
            .enumerate()
            .filter(|(_, device)| device.designator() == 'v')
            .map(|(index, _)| {
                -result
                    .solution
                    .values
                    .get(circuit.branch_rows(index).unwrap().start)
                    .unwrap()
            })
            .collect::<Vec<_>>()
    };
    let forward = solve(&format!(
        "* forward\nv1 a 0 100\nd1 a 0 dm\n{model}\n.end\n"
    ))[0];
    let reverse = solve(&format!(
        "* reverse\nv2 b 0 -100\nd2 b 0 dm\n{model}\n.end\n"
    ))[0];
    let tiled = solve(&format!(
        "* tile\nv1 a 0 100\nv2 b 0 -100\nd1 a 0 dm\nd2 b 0 dm\n{model}\n.end\n"
    ));
    close(tiled[0], forward, 1e-10, 1e-14);
    close(tiled[1], reverse, 1e-10, 1e-14);
    // Independent load-line residual, with an analytic exponential junction.
    let vt = model_kernel_diode::K_OVER_Q * 300.15;
    for (source, current) in [(100., forward), (-100., reverse)] {
        let voltage = (-1000_f64).mul_add(current, source);
        let expected =
            1e-12 * (voltage / vt).exp_m1() - 1e-3 * ((-voltage - 5.) / vt).exp() + 1e-15 * voltage;
        close(current, expected, 1e-7, 1e-12);
    }
}

#[test]
fn measured_temperature_precedence_survives_native_binding() {
    // Independent producer fixture temperature-policy.tsv at 0.4 V.
    let netlist = parse(
        "* thermal\nv1 a 0 .4\nv2 b 0 .4\nv3 c 0 .4\nv4 d 0 .4\nd1 a 0 base dtemp=7\nd2 b 0 relative dtemp=7\nd3 c 0 absolute dtemp=7\nd4 d 0 absolute temp=50 dtemp=7\n.model base simetrix_diode_level1(IS=1p N=1 TNOM=27)\n.model relative simetrix_diode_level1(IS=1p N=1 TNOM=27 T_REL_GLOBAL=10)\n.model absolute simetrix_diode_level1(IS=1p N=1 TNOM=27 T_ABS=45 T_REL_GLOBAL=10)\n.end\n",
    );
    let context = ModelContext::new(65., 27.).with_gmin(1e-15);
    let circuit = Circuit::from_netlist_with_context(&netlist, &context).unwrap();
    for (source, expected) in [
        ("v1", 0.0002837231788),
        ("v2", 0.0006053763324),
        ("v3", 0.00005462818211),
        ("v4", 0.00004584224507),
    ] {
        close(
            dc_current(&circuit, &context, source),
            expected,
            2e-8,
            1e-14,
        );
    }
}

#[test]
fn candidate_session_reuses_geometry_but_rebuilds_series_topology() {
    use ngspice_rs::analysis::optimization::parameters::{CandidateTopology, CircuitParameterPlan};
    use ngspice_rs::netlist::eval::EvalBudget;
    let source = parse(
        "* candidate\n.param area=2 resistance=100\nv1 a 0 1\nd1 a 0 dm area={area}\n.model dm simetrix_diode_level1(IS=0 RS={resistance})\n.options gmin=1m\n.end\n",
    );
    let plan = CircuitParameterPlan::new(source, &["area", "resistance"]).unwrap();
    let mut session = plan.session().unwrap();
    let mut budget = EvalBudget::default();
    let check =
        |candidate: &ngspice_rs::analysis::optimization::parameters::CircuitCandidate<'_>,
         area: f64,
         resistance: f64,
         topology| {
            assert_eq!(candidate.topology(), topology);
            let circuit = candidate.circuit();
            let context = candidate.model_context();
            let diode = circuit
                .devices()
                .iter()
                .find(|device| device.designator() == 'd')
                .unwrap();
            assert_eq!(
                diode.terminals().len(),
                if resistance == 0. { 2 } else { 3 }
            );
            close(
                dc_current(circuit, &context, "v1"),
                (1000. + resistance / area).recip(),
                1e-10,
                1e-14,
            );
        };
    check(&session, 2., 100., CandidateTopology::Reused);
    session.update(&[4., 100.], &mut budget).unwrap();
    check(&session, 4., 100., CandidateTopology::Reused);
    assert!(session.update(&[-1., 100.], &mut budget).is_err());
    check(&session, 4., 100., CandidateTopology::Reused);
    session.update(&[4., 0.], &mut budget).unwrap();
    check(&session, 4., 0., CandidateTopology::Rebuilt);
    session.update(&[2., 100.], &mut budget).unwrap();
    check(&session, 2., 100., CandidateTopology::Reused);
}

#[test]
fn source_selected_temperature_sweep_matches_measured_precedence() {
    let netlist = projected(
        "* thermal sweep\n*#SIMETRIX\nv1 a 0 .4\nv2 b 0 .4\nd1 a 0 dm\nd2 b 0 dm temp=50 dtemp=7\n.model dm D(IS=1p N=1 TNOM=27)\n.dc temp 65 72 7\n.end\n",
    );
    let context = AnalysisContext {
        gmin: 1e-15,
        ..AnalysisContext::default()
    };
    let mut circuit =
        Circuit::from_netlist_with_context(&netlist, &context.model_context()).unwrap();
    let plots = AnalysisInput::with_arguments(
        AnalysisKind::DcSweep,
        ["temp", "65", "72", "7", "rtol=1e-10", "vntol=1e-12"],
    )
    .resolve()
    .unwrap()
    .run_plots(&mut circuit, &context)
    .unwrap();
    for (row, (temperature, current)) in [(65., 0.0001627732692), (72., 0.0002837231788)]
        .into_iter()
        .enumerate()
    {
        close(
            plots[0].value("temp-sweep", row).unwrap().re,
            temperature,
            0.,
            0.,
        );
        close(
            -plots[0].value("i(v1)", row).unwrap().re,
            current,
            2e-8,
            1e-14,
        );
        close(
            -plots[0].value("i(v2)", row).unwrap().re,
            0.00004584224507,
            2e-8,
            1e-14,
        );
    }
}

#[test]
fn measured_nonlinear_transient_currents_survive_companion_integration() {
    // Public fixture ramp-27-1-knee-rec: nonlinear depletion, diffusion,
    // recombination and high injection during both ramp directions.
    let netlist = projected(
        "* nonlinear transient\n*#SIMETRIX\nv1 a 0 PWL(0 -.7 100n -.7 300n .65 400n .65 600n -.7 700n -.7)\nd1 a 0 dm m=2\n.model dm D(IS=1p N=1.2 ISR=50n NR=1.8 VJ=.7 M=.4 IKF=10u CJO=80p FC=.5 TT=35n TNOM=27)\n.tran .25n 700n\n.end\n",
    );
    let context = AnalysisContext {
        gmin: 1e-15,
        ..AnalysisContext::default()
    };
    let mut circuit =
        Circuit::from_netlist_with_context(&netlist, &context.model_context()).unwrap();
    let plots =
        AnalysisInput::with_arguments(AnalysisKind::Transient, [".25n", "700n", "0", ".25n"])
            .resolve()
            .unwrap()
            .run_plots(&mut circuit, &context)
            .unwrap();
    let plot = &plots[0];
    let times = (0..plot.point_count())
        .map(|row| plot.value("time", row).unwrap().re)
        .collect::<Vec<_>>();
    for (sample, current) in [
        (200e-9, 0.0010652108),
        (250e-9, 0.0017486154),
        (350e-9, 0.0463989921),
        (450e-9, -0.001618402),
        (500e-9, -0.001065295),
        (550e-9, -0.000914098),
        (650e-9, -1.31986e-7),
    ] {
        let high = times.partition_point(|time| *time < sample);
        assert!(high > 0 && high < times.len());
        let low = high - 1;
        let fraction = (sample - times[low]) / (times[high] - times[low]);
        let before = -plot.value("i(v1)", low).unwrap().re;
        let after = -plot.value("i(v1)", high).unwrap().re;
        close(
            fraction.mul_add(after - before, before),
            current,
            2e-3,
            2e-9,
        );
    }
}

#[test]
fn measured_recombination_noise_matches_generalized_sources() {
    // Public fixture noise-27-1-recombination. The fixture amplitudes
    // measure physical local shot and total current noise, not a circuit
    // output transfer function (covered separately by the series test).
    let netlist = projected(
        "* measured noise\n*#SIMETRIX\nv1 a 0 .45\nd1 a 0 dm area=2 m=3\n.model dm D(IS=1p N=1.2 ISR=100n NR=2.3 VJ=.7 M=.4 CJO=100p TT=20n TNOM=27 KF=1p AF=1.3)\n.end\n",
    );
    let context = ModelContext::default().with_gmin(1e-15);
    let circuit = Circuit::from_netlist_with_context(&netlist, &context).unwrap();
    let solved = solve_dc(
        &circuit,
        &context,
        &NewtonOptions::default(),
        &[],
        None,
        None,
    )
    .unwrap();
    let diode = circuit
        .devices()
        .iter()
        .find(|device| device.designator() == 'd')
        .unwrap();
    close(
        dc_current(&circuit, &context, "v1"),
        0.0007839686942,
        2e-8,
        1e-14,
    );
    for (frequency, expected) in [(1., 8.123194e-9), (1e3, 2.57366e-10), (1e6, 1.78101e-11)] {
        let mut sources = Vec::new();
        diode
            .noise(
                &NoiseContext {
                    model_context: &context.with_frequency(frequency),
                    unknowns: circuit.unknowns(),
                    bias: &solved.values,
                    states: None,
                    controls: &[],
                },
                &mut sources,
            )
            .unwrap();
        let densities = sources
            .iter()
            .map(|source| match source.kind {
                NoiseKind::Generalized { density, .. } => density,
                _ => panic!("expected generalized source"),
            })
            .collect::<Vec<_>>();
        assert_eq!(densities.len(), 2);
        let shot = sources
            .iter()
            .position(|source| source.suffix == "_shot")
            .unwrap();
        close(densities[shot].sqrt(), 1.584972e-11, 2e-6, 1e-18);
        close(densities.iter().sum::<Real>().sqrt(), expected, 2e-6, 1e-18);
    }
}
