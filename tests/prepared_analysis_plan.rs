//! Exercise the actual producer/solver analysis artifact boundary.
use std::fs;

use model_analysis_plan::sweep::SweepTarget;
use model_preprocess_command::WorkBudget;
use ngspice_rs::analysis::prepared_plan::{NativeTarget, PreparedAnalysisPlan};
use ngspice_rs::devices::subckt::PreparedDeclarations;
use ngspice_rs::netlist::{Parser, ast::Netlist};
use ngspice_rs::primitives::{SpiceError, SpiceResult};
use ngspice_rs_preprocess::{
    Cancellation, InputFormat, Options, PreparedBundle, Request, SourceInput, prepare,
};

fn produce(
    text: &str,
    format: InputFormat,
) -> anyhow::Result<(tempfile::TempDir, PreparedBundle, Netlist)> {
    let dir = tempfile::tempdir()?;
    let original = dir.path().join("source.sp");
    fs::write(&original, text)?;
    let bundle = prepare(
        &Request {
            format,
            input: SourceInput::File(original),
            libraries: Vec::new(),
            search_paths: Vec::new(),
        },
        &Options::default(),
        &Cancellation::default(),
    )?;
    for artifact in bundle.artifacts() {
        fs::write(dir.path().join(&artifact.filename), &artifact.bytes)?;
    }
    let netlist = Parser::new().parse_file(&dir.path().join(&bundle.primary.filename))?;
    Ok((dir, bundle, netlist))
}

fn resolve(_: &model_analysis_plan::PlanEntry, target: &SweepTarget) -> SpiceResult<NativeTarget> {
    match target {
        SweepTarget::Parameter { name } => Ok(NativeTarget::Parameter(name.clone())),
        SweepTarget::Temperature => Ok(NativeTarget::Temperature),
        SweepTarget::Source { name } => Ok(NativeTarget::Source(name.clone())),
        _ => Err(SpiceError::circuit(
            "no explicit test mapping for source target",
        )),
    }
}

fn declarations(bundle: &PreparedBundle) -> PreparedDeclarations<'_> {
    PreparedDeclarations {
        waveforms: &bundle.waveforms,
        pwl_resistors: &bundle.pwl_resistors,
    }
}

#[test]
fn hspice_nested_transient_rebinds_timing_and_keeps_source_order() -> anyhow::Result<()> {
    let (_dir, bundle, netlist) = produce(
        "Nested source timing\n.param duration=10n\nv1 in 0 1\nr1 in out 1k\nc1 out 0 1p\n.tran 1n 'duration' 2n 30n START=12n SWEEP duration LIN 3 10n 20n SWEEP TEMP 25 75 25\n.option DELMAX=.25n INTERP=1\n.end\n",
        InputFormat::HspiceDeck,
    )?;
    let source = bundle.analysis_plan.as_ref().unwrap();
    let work = WorkBudget::default();
    let plan = PreparedAnalysisPlan::bind(netlist, source, declarations(&bundle), resolve, &work)?;
    let mut count = 0;
    plan.run(&work, |result| {
        assert_eq!(result.ordinal, count);
        assert_eq!(result.entry_index, 0);
        let duration = (10. + 5. * (count % 3) as f64) * 1e-9;
        let temp = 25. + 25. * (count / 3) as f64;
        assert!((result.coordinates[0].value - duration).abs() < 1e-22);
        assert!((result.coordinates[1].value - temp).abs() < 1e-12);
        let plot = &result.plots[0];
        let time = plot
            .column("time")
            .unwrap()
            .iter()
            .map(|value| value.re)
            .collect::<Vec<_>>();
        assert!(time[0] >= 12e-9 - 1e-22);
        assert!((time.last().unwrap() - 30e-9).abs() < 1e-22);
        assert!(time.iter().any(|&value| (value - duration).abs() < 1e-22) == (duration >= 12e-9));
        assert!(
            plot.column("v(out)")
                .unwrap()
                .iter()
                .all(|value| (value.re - 1.).abs() < 1e-8)
        );
        count += 1;
        Ok(())
    })?;
    assert_eq!(count, 9);
    Ok(())
}

#[test]
fn ltspice_step_rebinds_electrical_parameter_and_dc_continuation() -> anyhow::Result<()> {
    let (_dir, bundle, netlist) = produce(
        "Independent and continuation\n.param r=1k\nv1 in 0 1\nr1 in 0 {r}\n.step param r list 1k 2k\n.dc v1 0 2 1\n.end\n",
        InputFormat::LtspiceDeck,
    )?;
    let source = bundle.analysis_plan.as_ref().unwrap();
    let work = WorkBudget::default();
    let plan = PreparedAnalysisPlan::bind(netlist, source, declarations(&bundle), resolve, &work)?;
    let mut count = 0;
    plan.run(&work, |result| {
        let r = result.coordinates[0].value;
        let plot = &result.plots[0];
        for (index, voltage) in [0., 1., 2.].into_iter().enumerate() {
            assert!((plot.value("v(in)", index).unwrap().re - voltage).abs() < 1e-12);
            assert!((plot.value("i(v1)", index).unwrap().re + voltage / r).abs() < 1e-12);
        }
        count += 1;
        Ok(())
    })?;
    assert_eq!(count, 2);
    Ok(())
}

#[test]
fn unresolved_source_automatic_integration_is_not_replaced_by_native_guess() -> anyhow::Result<()> {
    let (_dir, bundle, netlist) = produce(
        "Automatic ceiling\nv1 in 0 1\nr1 in 0 1k\n.tran 1n 10n\n.end\n",
        InputFormat::HspiceDeck,
    )?;
    let source = bundle.analysis_plan.as_ref().unwrap();
    let work = WorkBudget::default();
    let plan = PreparedAnalysisPlan::bind(netlist, source, declarations(&bundle), resolve, &work)?;
    let mut emitted = false;
    let error = plan
        .run(&work, |_| {
            emitted = true;
            Ok(())
        })
        .unwrap_err();
    assert!(!emitted);
    assert!(
        error
            .to_string()
            .contains("automatic source integration ceiling"),
        "{error}"
    );
    Ok(())
}

#[test]
fn hspice_nested_ac_binds_shared_device_parameters_and_temperature() -> anyhow::Result<()> {
    let (_dir, bundle, netlist) = produce(
        "Nested AC\n.param r=1k\nv1 in 0 0 AC 1\nr1 in out {r} tc1=0.01\nr2 out 0 {r} tc1=0.01\nc1 out 0 1u\n.option tnom=25\n.ac LIN 3 1k 3k SWEEP r LIN 2 1k 2k SWEEP TEMP 25 75 50\n.end\n",
        InputFormat::HspiceDeck,
    )?;
    let work = WorkBudget::default();
    let plan = PreparedAnalysisPlan::bind(
        netlist,
        bundle.analysis_plan.as_ref().unwrap(),
        declarations(&bundle),
        resolve,
        &work,
    )?;
    let mut count = 0;
    plan.run(&work, |result| {
        let r = result.coordinates[0].value;
        let t = result.coordinates[1].value;
        let resistance = r * (1. + 0.01 * (t - 25.));
        let plot = &result.plots[0];
        let f = plot.column("frequency").unwrap();
        assert_eq!(f.len(), 3);
        for (index, frequency) in f.iter().enumerate() {
            let x = std::f64::consts::TAU * frequency.re * 1e-6 * resistance;
            let expected_re = 2. / (4. + x * x);
            let expected_im = -x / (4. + x * x);
            let actual = plot.value("v(out)", index).unwrap();
            assert!((actual.re - expected_re).abs() < 1e-10);
            assert!((actual.im - expected_im).abs() < 1e-10);
        }
        count += 1;
        Ok(())
    })?;
    assert_eq!(count, 4);
    Ok(())
}

#[test]
fn source_initialization_cannot_be_omitted_or_bound_for_the_wrong_analysis() -> anyhow::Result<()> {
    use ngspice_rs::analysis::prepared_plan::initialization_selection;
    use ngspice_rs::netlist::prepared_initialization::PreparedInitialConditions;
    use ngspice_rs::primitives::AnalysisKind;
    let (dir, bundle, mut netlist) = produce(
        include_str!("fixtures/passive-ic/analysis-op.net"),
        InputFormat::SimetrixDeck,
    )?;
    let source = bundle.analysis_plan.as_ref().unwrap();
    let work = WorkBudget::default();
    assert!(
        PreparedAnalysisPlan::bind(
            netlist.clone(),
            source,
            declarations(&bundle),
            resolve,
            &work
        )
        .is_err()
    );
    PreparedInitialConditions::attach(
        &mut netlist,
        dir.path(),
        bundle.initial_conditions.clone().unwrap(),
        [(AnalysisKind::Ac, false)],
    )?;
    assert!(
        PreparedAnalysisPlan::bind(
            netlist.clone(),
            source,
            declarations(&bundle),
            resolve,
            &work
        )
        .is_err()
    );
    PreparedInitialConditions::attach(
        &mut netlist,
        dir.path(),
        bundle.initial_conditions.clone().unwrap(),
        initialization_selection(source)?,
    )?;
    let plan = PreparedAnalysisPlan::bind(netlist, source, declarations(&bundle), resolve, &work)?;
    let mut count = 0;
    plan.run(&work, |result| {
        let plot = &result.plots[0];
        for (name, expected) in [("v(cp)", 7.8125), ("v(cn)", 4.375), ("v(lp)", -19.9999998)] {
            assert!((plot.value(name, 0).unwrap().re - expected).abs() < 1e-5);
        }
        count += 1;
        Ok(())
    })?;
    assert_eq!(count, 1);
    Ok(())
}

fn structural_plan(target: SweepTarget, values: &[&str]) -> model_analysis_plan::AnalysisPlan {
    use model_analysis_plan::{
        AnalysisDialect, AnalysisPlan, Directive, DirectiveOrigin, Disposition, PlanEntry,
        PlanKind,
        sweep::{Cardinality, CoordinateLaw, CoordinateOrder, SweepAxis, SweepExecution},
    };
    let mut plan = AnalysisPlan::new(
        AnalysisDialect::Native,
        PlanKind::Deck,
        Some("Structural DC".into()),
    );
    plan.entries.push(PlanEntry {
        origin: DirectiveOrigin::Generated {
            artifact: "structural.cir".into(),
            first_line: 1,
            last_line: 1,
        },
        dialect: AnalysisDialect::Native,
        scope: Vec::new(),
        command: ".DC".into(),
        statement: "Typed continuation witness".into(),
        directive: Directive::DcSweep {
            axes: vec![SweepAxis {
                target,
                law: CoordinateLaw::List {
                    values: values.iter().map(|value| (*value).to_owned()).collect(),
                },
                execution: SweepExecution::Continuation,
                order: CoordinateOrder::Declared,
                cardinality: Cardinality::Nonempty,
            }],
        },
        independent_axes: Vec::new(),
        parallelism_hint: None,
        disposition: Disposition::RequiresBinding,
    });
    plan
}

fn native_deck(text: &str) -> SpiceResult<Netlist> {
    use ngspice_rs::netlist::source::parse_deck_text;
    Parser::new().parse_deck(&parse_deck_text(
        std::path::Path::new("structural.cir"),
        text,
    ))
}

#[test]
fn structural_parameter_continuation_retains_switch_hysteresis() -> anyhow::Result<()> {
    let source = native_deck(
        "Structural DC\n.param control=0\nvc c 0 {control}\nvin in 0 1\nr1 in a 1k\ns1 a 0 c 0 sm\n.model sm sw(vt=1 vh=0.5 ron=10 roff=1meg)\n.end\n",
    )?;
    let plan = structural_plan(
        SweepTarget::Parameter {
            name: "control".into(),
        },
        &["0", "1", "2", "1", "1", "0", "1", "2"],
    );
    let work = WorkBudget::default();
    let bound = PreparedAnalysisPlan::bind(
        source,
        &plan,
        PreparedDeclarations::default(),
        resolve,
        &work,
    )?;
    let mut count = 0;
    bound.run(&work, |result| {
        let plot = &result.plots[0];
        let open = 1e6 / 1001000.;
        let closed = 10. / 1010.;
        assert_eq!(plot.point_count(), 8);
        for (index, expected) in [open, open, closed, closed, closed, open, open, closed]
            .into_iter()
            .enumerate()
        {
            assert!((plot.value("v(a)", index).unwrap().re - expected).abs() < 1e-10);
        }
        count += 1;
        Ok(())
    })?;
    assert_eq!(count, 1);
    Ok(())
}

#[test]
fn structural_model_continuation_rebuilds_every_model_use() -> anyhow::Result<()> {
    let source =
        native_deck("Model DC\nv1 in 0 1\nr1 in 0 rm\nr2 in 0 rm\n.model rm r(r=1k)\n.end\n")?;
    let plan = structural_plan(
        SweepTarget::ModelParameter {
            family: Some("r".into()),
            model: "rm".into(),
            parameter: "r".into(),
        },
        &["1k", "2k", "500", "1k"],
    );
    let work = WorkBudget::default();
    let bound = PreparedAnalysisPlan::bind(
        source,
        &plan,
        PreparedDeclarations::default(),
        |entry, target| match target {
            SweepTarget::ModelParameter {
                model, parameter, ..
            } => Ok(NativeTarget::Model {
                scope: Vec::new(),
                name: model.clone(),
                parameter: parameter.clone(),
            }),
            _ => resolve(entry, target),
        },
        &work,
    )?;
    bound.run(&work, |result| {
        let plot = &result.plots[0];
        assert_eq!(plot.point_count(), 4);
        for (index, r) in [1000., 2000., 500., 1000.].into_iter().enumerate() {
            assert!((plot.value("i(v1)", index).unwrap().re + 2. / r).abs() < 1e-12);
            assert!((plot.value("param-sweep", index).unwrap().re - r).abs() < 1e-12);
        }
        Ok(())
    })?;
    Ok(())
}
