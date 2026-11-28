//! #272: removed storage is an output identity, not an electrical object.
use std::path::Path;

use ngspice_rs::{
    analysis::{
        AnalysisContext, AnalysisInput, Plot, PointSink, RawFile, RunConfig,
        optimization::parameters::CircuitParameterPlan,
    },
    devices::Circuit,
    netlist::{Parser, ast::Netlist, eval::EvalBudget, source::parse_deck_text},
    primitives::{AnalysisKind, Complex, SpiceResult},
};

const LIVE: &str = "v1 live 0 dc 1 ac 1\nr1 live 0 2\n";
const REMOVED: &str = "c1 ghost 0 2u m=0 pn_rseries=5 pn_cparallel=1u\nc2 live 0 2u m=0 pn_rseries=5 pn_cparallel=1u\nl1 orphan ghost 3m m=0 pn_rseries=4 pn_rparallel=7\n";
const ZERO: &[&str] = &["v(ghost)", "v(orphan)", "i(c1)", "i(c2)", "i(l1)"];

fn source(body: &str) -> SpiceResult<Netlist> {
    Parser::new().parse_deck(&parse_deck_text(
        Path::new("network.cir"),
        &format!("zero storage\n{body}\n.end\n"),
    ))
}

fn circuit(body: &str) -> SpiceResult<Circuit> {
    let source = source(body)?;
    RunConfig::from_netlist(&source)?.circuit(&source)
}

fn run(circuit: &mut Circuit, kind: AnalysisKind, args: &[&str]) -> SpiceResult<Plot> {
    AnalysisInput::with_arguments(kind, args.iter().copied())
        .resolve()?
        .run(circuit, &AnalysisContext::default())
}

fn assert_zero(plot: &Plot, names: &[&str]) {
    for name in names {
        for point in 0..plot.point_count() {
            assert_eq!(plot.value(name, point), Some(Complex::real(0.)), "{name}");
        }
    }
}

#[test]
fn independent_op_witness_and_no_physical_objects() -> SpiceResult<()> {
    let mut baseline = circuit(LIVE)?;
    let mut absent = circuit(&format!("{LIVE}{REMOVED}"))?;
    assert_eq!(absent.device_count(), baseline.device_count());
    assert_eq!(absent.nodes().len(), baseline.nodes().len());
    assert_eq!(absent.unknown_count(), baseline.unknown_count());
    assert_eq!(
        absent.state_history()?.len(),
        baseline.state_history()?.len()
    );
    let a = absent.linear_system()?;
    let b = baseline.linear_system()?;
    assert_eq!(a.a, b.a);
    assert_eq!(a.e, b.e);
    assert_eq!(a.sources.len(), b.sources.len());
    absent.set_observations(&[
        "i(c1)".into(),
        "i(r1)".into(),
        "@c1[i]".into(),
        "@l1[p]".into(),
        "@c2[m]".into(),
    ])?;
    let op = run(&mut absent, AnalysisKind::OperatingPoint, &[])?;
    assert_zero(&op, ZERO);
    assert_zero(&op, &["@c1[i]", "@l1[p]", "@c2[m]"]);
    // Independent LTspice 24.0.12 synthetic OP from producer #80.
    assert_eq!(op.value("v(live)", 0), Some(Complex::real(1.)));
    assert_eq!(op.value("i(r1)", 0), Some(Complex::real(0.5)));
    assert_eq!(op.value("i(v1)", 0), Some(Complex::real(-0.5)));
    Ok(())
}

#[derive(Default)]
struct Selected {
    columns: Vec<usize>,
    rows: Vec<Vec<Complex>>,
}
impl PointSink for Selected {
    fn begin(&mut self, header: &Plot) -> SpiceResult<()> {
        self.columns = ZERO
            .iter()
            .map(|name| header.variable_index(name).unwrap())
            .collect();
        self.columns.sort_unstable();
        Ok(())
    }
    fn selected_columns(&self) -> Option<&[usize]> {
        Some(&self.columns)
    }
    fn point(&mut self, values: &[Complex]) -> SpiceResult<()> {
        self.rows.push(values.to_vec());
        Ok(())
    }
    fn end(&mut self) -> SpiceResult<()> {
        Ok(())
    }
}

#[test]
fn all_analysis_phases_and_streamed_projections() -> SpiceResult<()> {
    for (kind, args) in [
        (AnalysisKind::Ac, vec!["lin", "3", "100", "10000"]),
        (AnalysisKind::DcSweep, vec!["v1", "0", "1", ".5"]),
        (AnalysisKind::Transient, vec!["100n", "3u", "0", "100n"]),
    ] {
        let body = format!("{LIVE}{REMOVED}");
        let mut baseline = circuit(LIVE)?;
        let mut absent = circuit(&body)?;
        let expected = run(&mut baseline, kind, &args)?;
        let plot = run(&mut absent, kind, &args)?;
        assert_zero(&plot, ZERO);
        assert_eq!(plot.point_count(), expected.point_count());
        for v in expected.variables() {
            for point in 0..plot.point_count() {
                assert_eq!(plot.value(&v.name, point), expected.value(&v.name, point));
            }
        }
        let mut selected = circuit(&body)?;
        let mut sink = Selected::default();
        AnalysisInput::with_arguments(kind, args.iter().copied())
            .resolve()?
            .run_into(&mut selected, &AnalysisContext::default(), &mut sink)?;
        assert_eq!(sink.rows.len(), plot.point_count());
        assert!(
            sink.rows
                .iter()
                .all(|row| row == &vec![Complex::real(0.); ZERO.len()])
        );
    }
    Ok(())
}

#[test]
fn hierarchical_and_behavioural_zero_reads() -> SpiceResult<()> {
    let mut c = circuit(&format!(
        "{LIVE}
.subckt removed p n copies=0
c1 ghost n 2u m={{copies}} pn_rseries=5
l1 p n 1m m={{copies}} pn_rseries=3
b1 out n v=i(c1)+i(l1)+v(p,ghost)
rout out n 1k
.ends
x1 live 0 removed
"
    ))?;
    let op = run(&mut c, AnalysisKind::OperatingPoint, &[])?;
    assert_zero(&op, &["i(c.x1.c1)", "i(l.x1.l1)", "v(x1.ghost)"]);
    assert_eq!(op.value("v(x1.out)", 0), Some(Complex::real(1.)));
    assert_eq!(c.device_count(), 4);
    assert!(c.nodes().get("x1.ghost").is_none());
    Ok(())
}

#[test]
fn parameter_candidates_rebuild_entire_network_across_zero() -> SpiceResult<()> {
    let body = |copies: f64| {
        format!(
            "{LIVE}.param copies={copies}\nc1 live 0 2u m={{copies}} pn_rseries=5 pn_rparallel=20 pn_cparallel=1u pn_report_dc=1\n"
        )
    };
    for initial in [0., 1.] {
        let plan = CircuitParameterPlan::new(source(&body(initial))?, &["copies"])?;
        let mut session = plan.session()?;
        for copies in [0., 1., 3., 0., 1.] {
            session.update(&[copies], &mut EvalBudget::default())?;
            let context = session.analysis_context();
            let candidate = session.circuit_mut();
            let mut reference = circuit(&body(copies))?;
            assert_eq!(candidate.device_count(), reference.device_count());
            assert_eq!(candidate.unknown_count(), reference.unknown_count());
            assert_eq!(
                candidate.state_history()?.len(),
                reference.state_history()?.len()
            );
            let request =
                AnalysisInput::with_arguments(AnalysisKind::Ac, ["lin", "3", "100", "10000"])
                    .resolve()?;
            let actual = request.run(candidate, &context)?;
            let expected = request.run(&mut reference, &AnalysisContext::default())?;
            for name in ["i(c1)", "i(v1)", "v(live)"] {
                for point in 0..expected.point_count() {
                    let a = actual.value(name, point).unwrap();
                    let b = expected.value(name, point).unwrap();
                    assert!(
                        (a - b).magnitude() < 1e-12,
                        "{initial} -> {copies}: {name} {a:?} {b:?}"
                    );
                }
            }
        }
    }
    Ok(())
}

#[test]
fn explicit_save_and_unknown_controls() -> SpiceResult<()> {
    let scratch = ngspice_conformance::ngspice::Scratch::new("absent-storage").unwrap();
    let input = scratch.path().join("input.cir");
    let raw = scratch.path().join("output.raw");
    std::fs::write(
        &input,
        format!("zero storage\n{LIVE}{REMOVED}.op\n.save v(ghost) i(c1) i(l1) @c2[i]\n.end\n"),
    )
    .unwrap();
    ngspice_rs::cli::simulate::run(&input, &raw, true)?;
    let file = RawFile::load(&raw)?;
    assert_zero(
        &file.plots[0].plot,
        &["v(ghost)", "i(c1)", "i(l1)", "@c2[i]"],
    );
    for bad in [
        "f1 out 0 c1 1\nr2 out 0 1k",
        "b1 out 0 v=i(missing)\nr2 out 0 1k",
        "C1 live 0 1u m=0 pn_rseries=0",
    ] {
        assert!(circuit(&format!("{LIVE}{REMOVED}{bad}")).is_err(), "{bad}");
    }
    let mut c = circuit(&format!("{LIVE}{REMOVED}"))?;
    assert!(c.set_observations(&["i(missing)".into()]).is_err());
    Ok(())
}
