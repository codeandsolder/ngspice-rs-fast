//! Reusable script session and resource bounds.
use std::path::Path;

use ngspice_rs::analysis::AnalysisInput;
use ngspice_rs::analysis::control::ControlSession;
use ngspice_rs::analysis::optimization::parameters::{CandidateTopology, CircuitParameterPlan};
use ngspice_rs::netlist::Parser;
use ngspice_rs::netlist::eval::EvalLimits;
use ngspice_rs::netlist::source::parse_deck_text;
use ngspice_rs::primitives::AnalysisKind;

fn divider() -> CircuitParameterPlan {
    let source = Parser::new()
        .parse_deck(&parse_deck_text(
            Path::new("control.cir"),
            "divider\n.param r=1k guard={1/(r-500)}\nv1 in 0 3\nr1 in out {r}\nr2 out 0 1k\n.end\n",
        ))
        .unwrap();
    CircuitParameterPlan::new(source, &["r"]).unwrap()
}

#[test]
fn repeated_parameter_runs_reuse_topology_and_reject_failed_edits() {
    let plan = divider();
    let mut session = ControlSession::new(&plan, EvalLimits::default()).unwrap();
    for r in [2000., 1000., 2000.] {
        assert_eq!(
            session.alter_parameter("R", r).unwrap(),
            CandidateTopology::Reused
        );
        let plots = session
            .run(AnalysisInput::new(AnalysisKind::OperatingPoint), 1, 100)
            .unwrap();
        let voltage = plots[0].value("v(out)", 0).unwrap().re;
        let analytic = 3000. / (r + 1000.);
        assert!((voltage - analytic).abs() < 1e-12);
    }
    assert!(session.alter_parameter("r", 500.).is_err());
    assert_eq!(session.parameter("r"), Some(2000.));
    let plots = session
        .run(AnalysisInput::new(AnalysisKind::OperatingPoint), 1, 100)
        .unwrap();
    assert!((plots[0].value("v(out)", 0).unwrap().re - 1.).abs() < 1e-12);
}

#[test]
fn plot_budget_failure_does_not_poison_next_run() {
    let plan = divider();
    let mut session = ControlSession::new(&plan, EvalLimits::default()).unwrap();
    assert!(
        session
            .run(AnalysisInput::new(AnalysisKind::OperatingPoint), 0, 100)
            .is_err()
    );
    assert!(
        session
            .run(AnalysisInput::new(AnalysisKind::OperatingPoint), 1, 0)
            .is_err()
    );
    assert!(
        session
            .run(AnalysisInput::new(AnalysisKind::OperatingPoint), 1, 100)
            .is_ok()
    );
}
