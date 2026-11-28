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
