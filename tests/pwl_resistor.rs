//! Prepared physical PWL laws: producer-to-solver qualification against
//! independent SIMPLIS multiroot observations and analytic load lines.
use ngspice_rs::analysis::{AnalysisInput, Plot, RunConfig};
use ngspice_rs::devices::{Circuit, subckt::PreparedDeclarations};
use ngspice_rs::netlist::{Parser, ast::Netlist};
use ngspice_rs::primitives::AnalysisKind;
use ngspice_rs_preprocess::{
    Cancellation, InputFormat, Options, PreparedBundle, Request, SourceInput, prepare,
    pwl_resistor::NativeBinding,
};
use std::fs;

const VOLTAGE: &str = ".model MR VPWLR NSEG=4 X0=-1 Y0=-1 X1=0 Y1=2 X2=1 Y2=0 X3=2 Y3=2 X4=3 Y4=3";
const CURRENT: &str = ".model MC IPWLR NSEG=4 X0=-1 Y0=-1 X1=2 Y1=0 X2=0 Y2=1 X3=2 Y3=2 X4=3 Y4=3";

fn source(body: &str, format: InputFormat) -> anyhow::Result<(Netlist, PreparedBundle)> {
    let dir = tempfile::tempdir()?;
    let path = dir.path().join("source.cir");
    fs::write(&path, format!("* prepared PWL\n{body}\n.end\n"))?;
    let bundle = prepare(
        &Request {
            format,
            input: SourceInput::File(path),
            libraries: Vec::new(),
            search_paths: Vec::new(),
        },
        &Options::default(),
        &Cancellation::default(),
    )?;
    for artifact in bundle.artifacts() {
        fs::write(dir.path().join(&artifact.filename), &artifact.bytes)?;
    }
    let deck = Parser::new().parse_file(dir.path().join(&bundle.primary.filename))?;
    Ok((deck, bundle))
}

fn circuit(deck: &Netlist, bundle: &PreparedBundle) -> anyhow::Result<(Circuit, RunConfig)> {
    let config = RunConfig::from_netlist(deck)?;
    let circuit = config.prepared_circuit(
        deck,
        PreparedDeclarations {
            waveforms: &bundle.waveforms,
            pwl_resistors: &bundle.pwl_resistors,
        },
    )?;
    Ok((circuit, config))
}

fn run(
    circuit: &mut Circuit,
    config: &RunConfig,
    kind: AnalysisKind,
    args: &[&str],
) -> anyhow::Result<Plot> {
    Ok(config
        .plan(AnalysisInput::with_arguments(kind, args.iter().copied()))?
        .run(circuit, &config.context())?)
}

fn near(plot: &Plot, name: &str, index: usize, expected: f64) {
    let actual = plot
        .value(name, index)
        .unwrap_or_else(|| panic!("missing {name}"))
        .re;
    assert!(
        (actual - expected).abs() < 1e-8,
        "{name} at {index}: {actual:.17e} != {expected:.17e}"
    );
}

#[test]
fn actual_prepared_voltage_defined_multiroot_hints_are_advisory() -> anyhow::Result<()> {
    for (segment, root) in [(1, -1. / 3.), (2, 0.5), (3, 1.5), (4, 1.5)] {
        let (deck, bundle) = source(
            &format!("I1 0 p 1\n!R1 p 0 MR IC={segment}\n{VOLTAGE}"),
            InputFormat::ExpandedDeck,
        )?;
        let (mut c, config) = circuit(&deck, &bundle)?;
        let plot = run(&mut c, &config, AnalysisKind::OperatingPoint, &[])?;
        near(&plot, "v(p)", 0, root);
        let NativeBinding::CurrentLaw { source } = &bundle.pwl_resistors[0].binding else {
            anyhow::bail!("wrong branch");
        };
        near(&plot, &format!("i({})", source.to_ascii_lowercase()), 0, 1.);
    }
    Ok(())
}

#[test]
fn actual_prepared_current_defined_multiroot_hints_are_advisory() -> anyhow::Result<()> {
    for (segment, root) in [(1, -1. / 3.), (2, 0.5), (3, 1.5), (4, 1.5)] {
        let (deck, bundle) = source(
            &format!("V1 p 0 1\n!R1 p 0 MC IC={segment}\n{CURRENT}"),
            InputFormat::ExpandedDeck,
        )?;
        let (mut c, config) = circuit(&deck, &bundle)?;
        let plot = run(&mut c, &config, AnalysisKind::OperatingPoint, &[])?;
        near(&plot, "i(v1)", 0, -root);
        let NativeBinding::VoltageLaw { sensor, source, .. } = &bundle.pwl_resistors[0].binding
        else {
            anyhow::bail!("wrong branch");
        };
        for name in [sensor, source] {
            near(&plot, &format!("i({})", name.to_ascii_lowercase()), 0, root);
        }
    }
    Ok(())
}

#[test]
fn floating_shared_terminals_and_transient_startup() -> anyhow::Result<()> {
    let (deck, bundle) = source(
        &format!(
            "Vcm cm 0 10\nI1 cm a 1\nI2 cm b 1\n!R1 a cm MR IC=1\n!R2 b cm MR IC=3\n{VOLTAGE}"
        ),
        InputFormat::ExpandedDeck,
    )?;
    let (mut c, config) = circuit(&deck, &bundle)?;
    let plot = run(&mut c, &config, AnalysisKind::OperatingPoint, &[])?;
    near(&plot, "v(a)", 0, 10. - 1. / 3.);
    near(&plot, "v(b)", 0, 11.5);
    near(&plot, "i(vcm)", 0, 0.);
    let (mut c, config) = circuit(&deck, &bundle)?;
    let plot = run(&mut c, &config, AnalysisKind::Transient, &["1n", "10n"])?;
    for index in 0..plot.point_count() {
        near(&plot, "v(a)", index, 10. - 1. / 3.);
        near(&plot, "v(b)", index, 11.5);
    }
    Ok(())
}

fn expected(x: f64) -> f64 {
    if x <= 0. {
        3_f64.mul_add(x, 2.)
    } else if x <= 1. {
        (-2_f64).mul_add(x, 2.)
    } else if x <= 2. {
        2. * (x - 1.)
    } else {
        x
    }
}

#[test]
fn independent_sweeps_and_small_signal() -> anyhow::Result<()> {
    let (deck, bundle) = source(
        &format!(
            "Vref n 0 5\nVdrive p n 0\nIforce n q 0\n!R1 p n MR IC=2\n!R2 q n MC IC=3\n{VOLTAGE}\n{CURRENT}"
        ),
        InputFormat::ExpandedDeck,
    )?;
    let (mut c, config) = circuit(&deck, &bundle)?;
    for (source, signal, sign) in [("vdrive", "i(vdrive)", -1.), ("iforce", "v(q)", 1.)] {
        let plot = run(
            &mut c,
            &config,
            AnalysisKind::DcSweep,
            &[source, "-2", "4", "0.5"],
        )?;
        assert_eq!(plot.point_count(), 13);
        for index in 0..13_u32 {
            let x = f64::from(index) * 0.5 - 2.;
            near(
                &plot,
                signal,
                index as usize,
                if sign > 0. {
                    5. + expected(x)
                } else {
                    -expected(x)
                },
            );
        }
    }
    // A native prepared branch has the exact signed tangent at the bias;
    // interior negative resistance is not regularized away.
    let (deck, bundle) = source(
        &format!("V1 p 0 0.5 AC 1\n!R1 p 0 MR IC=1\n{VOLTAGE}"),
        InputFormat::ExpandedDeck,
    )?;
    let (mut c, config) = circuit(&deck, &bundle)?;
    let ac = run(&mut c, &config, AnalysisKind::Ac, &["lin", "1", "1k", "1k"])?;
    near(&ac, "i(v1)", 0, 2.);
    Ok(())
}

#[test]
fn sidecar_validation_and_candidate_isolation() -> anyhow::Result<()> {
    let (deck, mut bundle) = source(
        &format!("I1 0 p 1\n!R1 p 0 MR IC=1\n{VOLTAGE}"),
        InputFormat::ExpandedDeck,
    )?;
    for (hint, root) in [(1, -1. / 3.), (3, 1.5), (1, -1. / 3.)] {
        bundle.pwl_resistors[0].initial_segment = Some(hint);
        let (mut c, config) = circuit(&deck, &bundle)?;
        near(
            &run(&mut c, &config, AnalysisKind::OperatingPoint, &[])?,
            "v(p)",
            0,
            root,
        );
    }
    let original = bundle.pwl_resistors[0].clone();
    bundle.pwl_resistors.push(original.clone());
    assert!(circuit(&deck, &bundle).is_err());
    bundle.pwl_resistors.pop();
    for hint in [0, 5] {
        bundle.pwl_resistors[0].initial_segment = Some(hint);
        assert!(circuit(&deck, &bundle).is_err());
    }
    bundle.pwl_resistors[0] = original.clone();
    bundle.pwl_resistors[0].negative = "not-the-terminal".into();
    assert!(circuit(&deck, &bundle).is_err());
    bundle.pwl_resistors[0] = original.clone();
    bundle.pwl_resistors[0].knots[2].voltage = bundle.pwl_resistors[0].knots[1].voltage;
    assert!(circuit(&deck, &bundle).is_err());
    bundle.pwl_resistors[0] = original;
    bundle.pwl_resistors[0].artifact = "wrong.cir".into();
    assert!(circuit(&deck, &bundle).is_err());
    Ok(())
}

#[test]
fn specialized_occurrence_hints_and_current_probes() -> anyhow::Result<()> {
    let (deck, bundle) = source(
        &format!(
            ".INIT X1.!R1=2\n.INIT X2.!R1=3\nVref ref 0 5\nI1 ref a 1\nI2 ref b 1\nI3 ref c 1\nI4 ref d 1\nX1 a ref CHILD\nX2 b ref CHILD\nX3 c ref CHILD\nX4 d ref CHILD\n.subckt CHILD p n\n!R1 p n MR IC=1\n{VOLTAGE}\n.ends CHILD"
        ),
        InputFormat::SimplisDeck,
    )?;
    let (mut c, config) = circuit(&deck, &bundle)?;
    let plot = run(&mut c, &config, AnalysisKind::OperatingPoint, &[])?;
    for (node, root) in [("a", 0.5), ("b", 1.5), ("c", -1. / 3.), ("d", -1. / 3.)] {
        near(&plot, &format!("v({node})"), 0, 5. + root);
    }
    let (deck, bundle) = source(
        &format!(
            "V1 a 0 0.5\nV2 b 0 1\n!R1 a 0 MR IC=2\n!R2 b 0 MC IC=3\nF1 out 0 !R1 2\nF2 other 0 !R2 3\nRout out 0 1\nRother other 0 1\n{VOLTAGE}\n{CURRENT}"
        ),
        InputFormat::ExpandedDeck,
    )?;
    let (mut c, config) = circuit(&deck, &bundle)?;
    let plot = run(&mut c, &config, AnalysisKind::OperatingPoint, &[])?;
    near(&plot, "v(out)", 0, -2.);
    near(&plot, "v(other)", 0, -4.5);
    Ok(())
}

#[test]
fn physical_taylor_tangent_at_knot_uses_one_sided_germ() -> anyhow::Result<()> {
    use ngspice_rs::devices::{Limit, TransientTiming};
    use ngspice_rs::maths::Vector;
    for (model, card) in [(VOLTAGE, "MR"), (CURRENT, "MC")] {
        let (deck, bundle) = source(
            &format!("V1 p 0 0\n!R1 p 0 {card} IC=2\n{model}"),
            InputFormat::ExpandedDeck,
        )?;
        let (c, config) = circuit(&deck, &bundle)?;
        let name = match &bundle.pwl_resistors[0].binding {
            NativeBinding::CurrentLaw { source } | NativeBinding::VoltageLaw { source, .. } => {
                source
            }
        };
        let index = c
            .devices()
            .iter()
            .position(|device| device.name().eq_ignore_ascii_case(name))
            .unwrap();
        let device = &c.devices()[index];
        let positive = c.unknowns().node_row(device.terminals()[0]).unwrap();
        let branch = c.branch_rows(index).unwrap().start;
        let mut state = vec![Vector::zeros(c.unknown_count()); 3];
        let current_defined = card == "MC";
        let (independent, dependent) = if current_defined {
            (branch, positive)
        } else {
            (positive, branch)
        };
        state[0].set(dependent, 2.)?;
        state[1].set(independent, 1.)?;
        let prepared = c.prepare_run();
        let dae = prepared.prepare_physical_dae(
            &config.context().model_context(),
            TransientTiming::new(1e-6, 1e-3)?,
        )?;
        for (limit, slope) in [(Limit::Left, 3.), (Limit::Right, -2.)] {
            let result = dae.evaluate_taylor(&state, 0., 1e-6, limit)?;
            assert_eq!(result.residual[0].get(branch), Some(0.));
            assert_eq!(result.residual[1].get(branch), Some(-slope));
            assert_eq!(
                result.conductance[0].get(branch, independent).to_bits(),
                (-slope).to_bits()
            );
            assert_eq!(
                result.conductance[0].get(branch, dependent).to_bits(),
                1_f64.to_bits()
            );
            assert_eq!(result.residual[2].get(branch), Some(0.));
        }
    }
    Ok(())
}
