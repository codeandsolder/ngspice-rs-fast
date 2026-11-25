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
    netlist::{Parser, ast::Netlist, source::parse_deck_text},
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
        "lexical diode\nV1 a 0 1\nV2 b 0 1\nX1 a 0 cell params: r=100\nX2 b 0 cell params: r=200\n.model DM simetrix_diode_level1(IS=0 RS=999)\n.subckt cell p n params: r=300\n.model DM simetrix_diode_level1(IS=0 RS={r})\nD1 p n DM\n.ends cell\n.end\n",
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
    let circuit = Circuit::from_netlist(&parse("topology\nV1 a 0 0.2\nD1 a 0 DM\n.model DM simetrix_diode_level1(IS=0 RS=100 TRS1=-0.01 TNOM=27)\n.end\n")).unwrap();
    let hot = ModelContext::new(127., 27.);
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
