//! #34 DC continuation policy: bounded, typed, reported and physically honest.
//!
//! Every success must solve the *original* equations (full sources, zero
//! artificial nodal gmin); continuation never accepts device state, runs accept
//! hooks or edits a source value.
use ngspice_rs::analysis::bias::{
    AttemptOutcome, ContinuationPolicy, ContinuationSchedule, DEFAULT_GMIN_SCHEDULE, DcOutcome,
    DcReport, DcSettings, DcStrategy, GminSchedule, IterationBudget, IterationLimit,
    MAX_GMIN_STAGES, MAX_SOURCE_STEPS, MAX_TOTAL_ITERATIONS, NgspiceStepping, SourceStepping,
    Stepping, solve_dc, solve_dc_with,
};
use ngspice_rs::analysis::newton::{NewtonOptions, StepLimiting};
use ngspice_rs::analysis::{AnalysisContext, AnalysisInput, Plot, RunConfig};
use ngspice_rs::devices::{
    AnalysisMode, Circuit, Forcing, Limit, LoadRequest, ModelContext, TransientTiming,
};
use ngspice_rs::maths::{SparseMatrix, Vector};
use ngspice_rs::netlist::Parser;
use ngspice_rs::netlist::ast::Netlist;
use ngspice_rs::netlist::source::parse_deck_text;
use ngspice_rs::primitives::{AnalysisKind, SpiceError, SpiceResult};
use std::cell::Cell;
use std::path::Path;
use std::rc::Rc;

const VT: f64 = (1.38064852e-23 / 1.6021766208e-19) * 300.15;
/// 1 mA through a 1 mA-saturation diode: direct Newton needs more than four
/// iterations, while source stepping converges within four at every stage.
const DIFFICULT: &str = "i1 0 a 1m\nd1 a 0 dm\n.model dm d(is=1m)";
/// The same hard bias, with an AC unit drive so an `.ac` run must solve it first.
const DIFFICULT_AC: &str = "i1 0 a dc 1m ac 1\nd1 a 0 dm\n.model dm d(is=1m)";

/// Relative accuracy of a converged bias of [`DIFFICULT`] under ngspice's
/// default Newton tolerances: `NIiter` accepts once the update is within
/// `reltol |v| + vntol` (1e-3 relative plus 1e-6 V, 1.9e-5 V at this 18 mV
/// bias) and returns the iterate the last load used (`CKTrhsOld`), so the bias
/// is guaranteed only to about reltol. Measured in C ngspice (SPARSE): the
/// direct solve misses `vt ln 2` by 1.7e-9 relative (its AC `v(a)` misses
/// `vt / 2 mA` by 1.2e-9), but with `.options noopiter gminsteps=0
/// srcsteps=20` its last uniform source step stops at 0.017936336109075593,
/// 4.5e-4 relative from the root; the port returns the same value bit for bit.
const NIITER_ACCURACY: f64 = 1e-3;

fn parse(body: &str) -> Netlist {
    Parser::new()
        .parse_deck(&parse_deck_text(
            Path::new("dc.cir"),
            &format!("DC continuation\n{body}\n.end\n"),
        ))
        .unwrap()
}
fn circuit(body: &str) -> Circuit {
    Circuit::from_netlist(&parse(body)).unwrap()
}
fn close(a: f64, b: f64, relative: f64, absolute: f64) {
    assert!(
        (a - b).abs() <= relative * b.abs() + absolute,
        "{a:e} != {b:e}"
    );
}
/// Four iterations under the port's legacy global voltage-step damping: the
/// continuation-mechanics tests below were designed around that policy's
/// iteration counts (with ngspice's per-device limiting, the default, the
/// first `MODEINITJCT` stage alone spends more than four iterations).
fn bounded() -> NewtonOptions {
    NewtonOptions {
        max_iterations: 4,
        limiting: StepLimiting::Global,
        ..NewtonOptions::default()
    }
}
fn settings(newton: NewtonOptions, continuation: ContinuationPolicy) -> DcSettings {
    DcSettings {
        newton,
        continuation,
    }
}
fn custom_ladder(gmin: Vec<f64>, source: Option<SourceStepping>) -> ContinuationPolicy {
    ContinuationPolicy {
        schedule: ContinuationSchedule::Ladder {
            gmin: GminSchedule::new(gmin).unwrap(),
            source,
        },
        ..ContinuationPolicy::disabled()
    }
}

fn ladder_parts(policy: &ContinuationPolicy) -> (&[f64], Option<&SourceStepping>) {
    let ContinuationSchedule::Ladder { gmin, source } = &policy.schedule else {
        panic!("expected custom ladder");
    };
    (gmin.as_slice(), source.as_ref())
}

fn attempts(report: &DcReport) -> Vec<(DcStrategy, AttemptOutcome)> {
    report
        .attempts
        .iter()
        .map(|attempt| (attempt.strategy, attempt.outcome))
        .collect()
}
/// The unmodified diode equation `Id(v) = 1 mA`, including the fixed 1e-12 S
/// junction gmin, evaluated independently of the solver.
fn diode_residual(v: f64) -> f64 {
    1e-3 * ((v / VT).exp() - 1.) + 1e-12 * v - 1e-3
}
fn assert_original_equations_solved(report: &DcReport) {
    let last = report.stages.last().unwrap();
    assert!(last.converged() && last.is_unregularized(), "{last:?}");
    assert_eq!(
        report.total_iterations,
        report.stages.iter().map(|s| s.iterations).sum::<usize>()
    );
    assert!(report.total_iterations <= report.budget);
}
fn dc_rhs(c: &Circuit) -> Vec<f64> {
    c.small_signal_system(&ModelContext::default(), &Vector::zeros(c.unknown_count()))
        .unwrap()
        .dc_rhs(None)
        .unwrap()
        .as_slice()
        .to_vec()
}

/// `v(a)^3 = i` with a singular Jacobian at the zero seed. Counts accept hooks.
#[derive(Debug)]
struct Cubic {
    nodes: [ngspice_rs::primitives::NodeId; 2],
    accepted: Rc<Cell<usize>>,
}
impl ngspice_rs::devices::Device for Cubic {
    /// Like a C device's `DEVsetup`, allocate the conductance elements even
    /// though the load stamps zero at v = 0 (where `LoadGmin` then acts).
    fn setup_matrix(
        &self,
        setup: &mut ngspice_rs::devices::setup::MatrixSetup<'_>,
    ) -> SpiceResult<bool> {
        setup.conductance(self.nodes[0], self.nodes[1]);
        Ok(true)
    }
    fn name(&self) -> &str {
        "x1"
    }
    fn designator(&self) -> char {
        'x'
    }
    fn terminals(&self) -> &[ngspice_rs::primitives::NodeId] {
        &self.nodes
    }
    fn is_nonlinear(&self) -> bool {
        true
    }
    fn stamp(&self, context: &mut ngspice_rs::devices::StampContext<'_>) -> SpiceResult<()> {
        let [a, b] = self.nodes;
        let v = context.node_voltage(a) - context.node_voltage(b);
        let g = 3. * v * v;
        context.stamp(a, a, g)?;
        context.stamp(b, b, g)?;
        context.stamp(a, b, -g)?;
        context.stamp(b, a, -g)?;
        context.stamp_rhs(a, g * v - v * v * v)?;
        context.stamp_rhs(b, v * v * v - g * v)
    }
    fn assemble_small_signal(
        &self,
        context: &mut ngspice_rs::devices::LinearContext<'_>,
        bias: &Vector,
    ) -> SpiceResult<()> {
        let v = context
            .unknowns
            .node_row(self.nodes[0])
            .and_then(|r| bias.get(r))
            .unwrap_or(0.);
        context.nodal(self.nodes, 3. * v * v, false)
    }
    fn accept(&self, _: &ngspice_rs::devices::AcceptContext<'_>) {
        self.accepted.set(self.accepted.get() + 1);
    }
}
fn cubic(accepted: &Rc<Cell<usize>>) -> Circuit {
    let mut c = circuit("i1 0 a 1");
    let a = c.nodes().get("a").unwrap();
    c.add_device(Box::new(Cubic {
        nodes: [a, ngspice_rs::primitives::NodeId::GROUND],
        accepted: Rc::clone(accepted),
    }))
    .unwrap();
    c.finalize().unwrap();
    c
}

/// A nonlinear device that reports unimplemented physics rather than failing to
/// converge. Continuation must surface that error unchanged instead of retrying
/// it against a gmin or source-stepped equation set.
#[derive(Debug)]
struct Rejecting {
    nodes: [ngspice_rs::primitives::NodeId; 2],
    accepted: Rc<Cell<usize>>,
}
impl ngspice_rs::devices::Device for Rejecting {
    fn name(&self) -> &str {
        "x1"
    }
    fn designator(&self) -> char {
        'x'
    }
    fn terminals(&self) -> &[ngspice_rs::primitives::NodeId] {
        &self.nodes
    }
    fn is_nonlinear(&self) -> bool {
        true
    }
    fn stamp(&self, _: &mut ngspice_rs::devices::StampContext<'_>) -> SpiceResult<()> {
        Err(SpiceError::not_yet_ported(
            "x1 junction physics",
            "src/spicelib/devices/x/xload.c",
        ))
    }
    fn assemble_linear(
        &self,
        context: &mut ngspice_rs::devices::LinearContext<'_>,
    ) -> SpiceResult<()> {
        context.nodal(self.nodes, 1e-3, false)
    }
    fn accept(&self, _: &ngspice_rs::devices::AcceptContext<'_>) {
        self.accepted.set(self.accepted.get() + 1);
    }
}

#[test]
fn structural_device_errors_are_not_retried_by_continuation() {
    let accepted = Rc::new(Cell::new(0));
    let mut c = circuit("v1 a 0 1\nr1 a 0 1k");
    let a = c.nodes().get("a").unwrap();
    c.add_device(Box::new(Rejecting {
        nodes: [a, ngspice_rs::primitives::NodeId::GROUND],
        accepted: Rc::clone(&accepted),
    }))
    .unwrap();
    c.finalize().unwrap();
    let context = ModelContext::default();
    let failure = solve_dc_with(&c, &context, &DcSettings::default(), &[], None, None).unwrap_err();
    assert!(failure.error.is_not_yet_ported(), "{}", failure.error);
    assert_eq!(failure.report.outcome, DcOutcome::NonRetryableFailure);
    // Exactly the direct attempt and its single unregularized stage: the gmin and
    // source-stepping strategies in the default policy were never started.
    assert_eq!(failure.report.attempts.len(), 1);
    assert_eq!(failure.report.attempts[0].outcome, AttemptOutcome::Failed);
    assert_eq!(failure.report.stages.len(), 1);
    let stage = &failure.report.stages[0];
    assert_eq!(stage.strategy, DcStrategy::Direct);
    assert!(stage.is_unregularized(), "{stage:?}");
    assert_eq!(accepted.get(), 0);
    let ContinuationSchedule::Ngspice(stepping) = ContinuationPolicy::default().schedule else {
        panic!("the default policy is ngspice's CKTop schedule");
    };
    assert!(
        stepping.gmin != Stepping::Disabled && stepping.source != Stepping::Disabled,
        "this test only proves skipping when the default policy would otherwise continue"
    );
}

fn run_ac(options: &str, explicit: &[&str]) -> SpiceResult<Plot> {
    let netlist = parse(&format!("{DIFFICULT_AC}\n{options}"));
    let config = RunConfig::from_netlist(&netlist)?;
    let mut c = config.circuit(&netlist)?;
    let mut arguments = vec!["lin", "1", "1k", "1k"];
    arguments.extend(explicit);
    let request = config.plan(AnalysisInput::with_arguments(AnalysisKind::Ac, arguments))?;
    request.run(&mut c, &config.context())
}

#[test]
fn ac_bias_deck_continuation_options_use_last_set_wins_and_request_precedence() {
    // Deck continuation options with a four-iteration request budget (a deck
    // `itl1` below 100 is C's effective 100, see below): the same hard bias
    // that fails for `.op` fails here, with the same bounded diagnosis.
    let error = run_ac(
        ".options srcsteps=0 gminsteps=0",
        &["maxiter=4", "limiting=global"],
    )
    .unwrap_err();
    let message = error.to_string();
    assert!(message.contains("source stepping: disabled"), "{message}");
    assert!(message.contains("gmin stepping: disabled"), "{message}");
    // Deck source stepping alone rescues the budget, exactly as for `.op`, and
    // the AC point is linearized about the same physical full-source bias.
    let solved = run_ac(".options gminsteps=0", &["maxiter=4", "limiting=global"]).unwrap();
    close(
        solved.value("v(a)", 0).unwrap().re,
        1. / (2. * (1e-3 / VT)),
        NIITER_ACCURACY,
        1e-9,
    );
    // Last deck occurrence wins: the later `srcsteps=20` re-enables the strategy.
    assert!(
        run_ac(
            ".options gminsteps=0 srcsteps=0 srcsteps=20",
            &["maxiter=4", "limiting=global"]
        )
        .is_ok()
    );
    // An explicit request argument outranks the deck value.
    assert!(
        run_ac(
            ".options gminsteps=0",
            &["maxiter=4", "limiting=global", "srcsteps=0"]
        )
        .is_err()
    );
    // niiter.c raises a deck itl1=4 to 100, which the direct solve needs no
    // continuation for (C converges on this deck with itl1=4 as well).
    assert!(run_ac(".options itl1=4 srcsteps=0 gminsteps=0", &[]).is_ok());
}

#[test]
fn difficult_diode_fails_bounded_direct_newton_but_the_configured_policy_solves_it() {
    let c = circuit(DIFFICULT);
    let context = ModelContext::default();
    let run = |continuation| {
        solve_dc_with(
            &c,
            &context,
            &settings(bounded(), continuation),
            &[],
            None,
            None,
        )
    };

    // Direct Newton alone exhausts four iterations; nothing else is tried.
    let failure = run(ContinuationPolicy::disabled()).unwrap_err();
    assert!(matches!(failure.error, SpiceError::Numerical { .. }));
    assert_eq!(failure.report.outcome, DcOutcome::Exhausted);
    assert_eq!(
        attempts(&failure.report),
        [
            (DcStrategy::Direct, AttemptOutcome::Failed),
            (DcStrategy::GminStepping, AttemptOutcome::Disabled),
            (DcStrategy::SourceStepping, AttemptOutcome::Disabled),
        ]
    );
    assert_eq!(failure.report.stages.len(), 1);
    // NIiter fails after the solve that exceeds the limit: 4 + 1 passes.
    assert_eq!(failure.report.stages[0].iterations, 5);
    let message = failure.error.to_string();
    assert!(message.contains("gmin stepping: disabled"), "{message}");
    assert!(message.contains("source stepping: disabled"), "{message}");
    assert!(message.contains("iteration limit"), "{message}");

    // The gmin schedule alone cannot rescue it within the same iteration budget.
    let gmin_only = custom_ladder(DEFAULT_GMIN_SCHEDULE.to_vec(), None);
    let failure = run(gmin_only).unwrap_err();
    assert_eq!(failure.report.outcome, DcOutcome::Exhausted);
    assert_eq!(failure.report.stages.len(), 2);
    assert_eq!(failure.report.stages[1].gmin, 1e-3);
    assert!(!failure.report.stages[1].converged());
    assert_eq!(
        attempts(&failure.report)[1],
        (DcStrategy::GminStepping, AttemptOutcome::Failed)
    );
    assert!(
        failure
            .error
            .to_string()
            .contains("source stepping: disabled")
    );

    // Configured source stepping (no gmin stepping) succeeds in the same budget.
    let source_only = custom_ladder(Vec::new(), Some(SourceStepping::uniform(20, 1e-8).unwrap()));
    let solved = run(source_only).unwrap();
    let report = &solved.report;
    assert_eq!(
        report.outcome,
        DcOutcome::Converged(DcStrategy::SourceStepping)
    );
    assert_eq!(
        attempts(report),
        [
            (DcStrategy::Direct, AttemptOutcome::Failed),
            (DcStrategy::GminStepping, AttemptOutcome::Disabled),
            (DcStrategy::SourceStepping, AttemptOutcome::Converged),
        ]
    );
    assert_eq!(report.stages.len(), 1 + 21 + 1);
    assert!(!report.stages[0].converged());
    assert!(
        report.stages[1..]
            .iter()
            .all(|s| s.converged() && s.iterations <= 4)
    );
    assert_eq!(
        (report.stages[1].source_scale, report.stages[1].gmin),
        (0., 1e-8)
    );
    assert_eq!(
        (report.stages[21].source_scale, report.stages[21].gmin),
        (1., 1e-8)
    );
    assert!(!report.stages[21].is_unregularized());
    assert_original_equations_solved(report);
    // Success is checked against the unmodified equations, not the solver's own.
    let v = solved.solution.values.as_slice()[0];
    close(v, VT * 2_f64.ln(), NIITER_ACCURACY, 1e-12);
    close(diode_residual(v) + 1e-3, 1e-3, NIITER_ACCURACY, 1e-12);

    // The ladder policy tries gmin stepping first, then source stepping.
    let defaults = run(ContinuationPolicy::ladder()).unwrap();
    assert_eq!(
        attempts(&defaults.report),
        [
            (DcStrategy::Direct, AttemptOutcome::Failed),
            (DcStrategy::GminStepping, AttemptOutcome::Failed),
            (DcStrategy::SourceStepping, AttemptOutcome::Converged),
        ]
    );
    assert_original_equations_solved(&defaults.report);
    // The compatible wrapper is the same solve with the default (ngspice) policy.
    let ngspice = run(ContinuationPolicy::default()).unwrap();
    assert_original_equations_solved(&ngspice.report);
    let wrapped = solve_dc(&c, &context, &bounded(), &[], None, None).unwrap();
    assert_eq!(
        wrapped.values.as_slice(),
        ngspice.solution.values.as_slice()
    );
}

#[test]
fn configured_gmin_schedule_rescues_a_singular_jacobian_without_source_stepping() {
    let accepted = Rc::new(Cell::new(0));
    let c = cubic(&accepted);
    let context = ModelContext::default();
    let gmin_only = custom_ladder(DEFAULT_GMIN_SCHEDULE.to_vec(), None);
    let solved = solve_dc_with(
        &c,
        &context,
        &settings(NewtonOptions::default(), gmin_only),
        &[],
        None,
        None,
    )
    .unwrap();
    close(solved.solution.values.as_slice()[0], 1., 1e-8, 1e-12);
    assert_eq!(
        solved.report.outcome,
        DcOutcome::Converged(DcStrategy::GminStepping)
    );
    let scheduled: Vec<f64> = solved
        .report
        .stages
        .iter()
        .filter(|s| s.strategy == DcStrategy::GminStepping)
        .map(|s| s.gmin)
        .collect();
    let mut expected = DEFAULT_GMIN_SCHEDULE.to_vec();
    expected.push(0.);
    assert_eq!(scheduled, expected);
    assert!(!solved.report.stages[0].converged());
    assert_original_equations_solved(&solved.report);

    // A custom, shorter schedule is honoured exactly.
    let custom = custom_ladder(vec![1e-2, 1e-6], None);
    let solved = solve_dc_with(
        &c,
        &context,
        &settings(NewtonOptions::default(), custom),
        &[],
        None,
        None,
    )
    .unwrap();
    let gmins: Vec<f64> = solved.report.stages[1..].iter().map(|s| s.gmin).collect();
    assert_eq!(gmins, [1e-2, 1e-6, 0.]);
    close(solved.solution.values.as_slice()[0], 1., 1e-8, 1e-12);

    // With every schedule disabled the same singular start is a reported failure.
    let failure = solve_dc_with(
        &c,
        &context,
        &settings(NewtonOptions::default(), ContinuationPolicy::disabled()),
        &[],
        None,
        None,
    )
    .unwrap_err();
    assert_eq!(failure.report.outcome, DcOutcome::Exhausted);
    assert_eq!(failure.report.stages.len(), 1);
    assert_eq!(accepted.get(), 0);
}

#[test]
fn exhausted_schedules_and_total_budget_are_bounded_and_reported() {
    let c = circuit(DIFFICULT);
    let context = ModelContext::default();
    // The budget covers direct Newton (its 4-iteration limit plus the pass
    // NIiter makes before failing) plus two more: the gmin stage is cut
    // short, the attempt stops and nothing further runs.
    let policy = ContinuationPolicy {
        max_total_iterations: Some(IterationBudget::new(7).unwrap()),
        ..ContinuationPolicy::ladder()
    };
    let failure =
        solve_dc_with(&c, &context, &settings(bounded(), policy), &[], None, None).unwrap_err();
    assert_eq!(failure.report.outcome, DcOutcome::BudgetExhausted);
    assert_eq!(failure.report.budget, 7);
    assert_eq!(failure.report.total_iterations, 7);
    assert_eq!(failure.report.stages.len(), 2);
    assert_eq!(failure.report.stages[1].iterations, 2);
    assert_eq!(failure.report.attempts.len(), 2);
    assert!(
        failure.error.to_string().contains("budget"),
        "{}",
        failure.error
    );

    // A one-iteration budget leaves nothing for continuation.
    let policy = ContinuationPolicy {
        max_total_iterations: Some(IterationBudget::new(1).unwrap()),
        ..ContinuationPolicy::ladder()
    };
    let failure =
        solve_dc_with(&c, &context, &settings(bounded(), policy), &[], None, None).unwrap_err();
    assert_eq!(failure.report.outcome, DcOutcome::BudgetExhausted);
    assert!(failure.report.total_iterations <= 1);

    // Without an explicit budget the work is still bounded by
    // stages * (maxiter + 1), the passes NIiter makes before failing.
    let failure = solve_dc_with(
        &c,
        &context,
        &settings(bounded(), ContinuationPolicy::disabled()),
        &[],
        None,
        None,
    )
    .unwrap_err();
    assert_eq!(failure.report.budget, 5);
    let defaults = solve_dc_with(
        &c,
        &context,
        &settings(bounded(), ContinuationPolicy::ladder()),
        &[],
        None,
        None,
    )
    .unwrap();
    // Each solve: its limit plus the pass NIiter makes before failing.
    assert_eq!(defaults.report.budget, (1 + 11 + 22) * (4 + 1));
}

#[test]
fn default_budget_distinguishes_direct_and_continuation_iteration_limits() {
    use ngspice_rs::analysis::bias::MAX_ADAPTIVE_STAGES;

    let c = circuit("v1 a 0 1\nr1 a 0 1k");
    let context = ModelContext::default();
    let direct_limit = 4;
    let stage_limit = 7;
    let cases = [
        // One direct solve, two gmin stages plus their direct-limit
        // closing solve, three source stages plus a stage-limit closing solve.
        (
            custom_ladder(
                vec![1e-6, 1e-9],
                Some(SourceStepping::new(vec![0., 0.5, 1.], 1e-8).unwrap()),
            ),
            2,
            6,
        ),
        (
            ContinuationPolicy {
                skip_direct: true,
                ..custom_ladder(
                    vec![1e-6, 1e-9],
                    Some(SourceStepping::new(vec![0., 0.5, 1.], 1e-8).unwrap()),
                )
            },
            1,
            6,
        ),
        (
            ContinuationPolicy::ngspice(NgspiceStepping::new(2, 3, 10., 50).unwrap()),
            2,
            3 + 4,
        ),
        // Both dynamic gmin strategies have a direct-limit closing solve.
        // Gillespie may run its initial zero-source solve, eleven gmin
        // stages and all adaptive source steps, all at the stage limit.
        (
            ContinuationPolicy::default(),
            3,
            3 * MAX_ADAPTIVE_STAGES + 12,
        ),
        (ContinuationPolicy::disabled(), 1, 0),
    ];
    for (mut policy, direct_count, stage_count) in cases {
        policy.stage_max_iterations = Some(IterationLimit::new(stage_limit).unwrap());
        let expected = direct_count * (direct_limit + 1) + stage_count * (stage_limit + 1);
        let settings = settings(
            NewtonOptions {
                max_iterations: direct_limit,
                ..NewtonOptions::default()
            },
            policy,
        );
        let solved = solve_dc_with(&c, &context, &settings, &[], None, None).unwrap();
        assert_eq!(solved.report.budget, expected, "{settings:?}");
        assert_original_equations_solved(&solved.report);
    }
}

#[test]
fn invalid_schedules_scales_and_budgets_cannot_be_constructed() {
    let c = circuit(DIFFICULT);
    let context = ModelContext::default();
    let bad_gmin: [Vec<f64>; 8] = [
        vec![f64::NAN],
        vec![f64::INFINITY],
        vec![0.],
        vec![-1e-3],
        vec![2.],
        vec![1e-3, 1e-3],
        vec![1e-6, 1e-3],
        vec![1e-3; MAX_GMIN_STAGES + 1],
    ];
    for bad in bad_gmin {
        assert!(GminSchedule::new(bad).is_err());
    }
    let bad_scales: [&[f64]; 7] = [
        &[],
        &[f64::NAN],
        &[f64::INFINITY],
        &[-0.1, 1.],
        &[0., 1.5],
        &[0., 0.5, 0.5],
        &[0.5, 0.25],
    ];
    for bad in bad_scales {
        assert!(SourceStepping::new(bad.to_vec(), 1e-8).is_err());
    }
    for gmin in [f64::NAN, -1e-8, 2.] {
        assert!(SourceStepping::new(vec![0., 1.], gmin).is_err());
    }
    for total in [0, MAX_TOTAL_ITERATIONS + 1] {
        assert!(IterationBudget::new(total).is_err());
    }
    for stage in [0, ngspice_rs::analysis::newton::MAX_ITERATIONS + 1] {
        assert!(IterationLimit::new(stage).is_err());
    }
    assert!(SourceStepping::uniform(0, 1e-8).is_err());
    assert!(SourceStepping::uniform(MAX_SOURCE_STEPS + 1, 1e-8).is_err());
    assert!(SourceStepping::uniform(MAX_SOURCE_STEPS, 1e-8).is_ok());
    assert!(ContinuationPolicy::geometric_gmin(1e-3, 1., 3).is_err());
    assert!(ContinuationPolicy::geometric_gmin(1e-3, 0.5, 3).is_err());
    assert!(ContinuationPolicy::geometric_gmin(1e-3, f64::NAN, 3).is_err());
    assert!(ContinuationPolicy::geometric_gmin(1e-3, 10., MAX_GMIN_STAGES + 1).is_err());
    assert!(ContinuationPolicy::geometric_gmin(1e-3, 1e6, MAX_GMIN_STAGES).is_err());
    assert!(ContinuationPolicy::geometric_gmin(f64::NAN, 10., 3).is_err());
    // Newton options are validated by the same entry point.
    let broken = NewtonOptions {
        max_iterations: 0,
        ..NewtonOptions::default()
    };
    assert!(
        solve_dc_with(
            &c,
            &context,
            &settings(broken, ContinuationPolicy::default()),
            &[],
            None,
            None
        )
        .is_err()
    );
}

#[test]
fn ladder_policy_is_the_historical_schedule_and_geometric_builders_are_exact() {
    let default = ContinuationPolicy::ladder();
    let (gmin, source) = ladder_parts(&default);
    assert_eq!(gmin, DEFAULT_GMIN_SCHEDULE);
    let source = source.unwrap();
    assert_eq!(source.gmin(), 1e-8);
    assert_eq!(
        source.scales(),
        (0..=20).map(|k| f64::from(k) / 20.).collect::<Vec<_>>()
    );
    assert_eq!(default.max_total_iterations, None);
    assert_eq!(
        ContinuationPolicy::from_steps(None, None, None).unwrap(),
        default
    );
    assert_eq!(
        ContinuationPolicy::from_steps(Some(20), Some(10), Some(10.)).unwrap(),
        default
    );
    assert_eq!(
        ContinuationPolicy::geometric_gmin(1e-3, 2., 3)
            .unwrap()
            .as_slice(),
        [1e-3, 1e-3 / 2., 1e-3 / 4.]
    );
    let none = ContinuationPolicy::from_steps(Some(0), Some(0), None).unwrap();
    assert_eq!(none, ContinuationPolicy::disabled());
    let four = ContinuationPolicy::from_steps(Some(4), Some(3), None).unwrap();
    assert_eq!(ladder_parts(&four).0, &DEFAULT_GMIN_SCHEDULE[..3]);
    assert_eq!(
        ladder_parts(&four).1.unwrap().scales(),
        [0., 0.25, 0.5, 0.75, 1.]
    );
    // A factor alone rebuilds the default number of stages.
    let wide = ContinuationPolicy::from_steps(None, None, Some(100.)).unwrap();
    assert_eq!(ladder_parts(&wide).0.len(), 10);
    close(ladder_parts(&wide).0[1], 1e-5, 1e-14, 0.);
}

#[test]
fn impossible_ideal_source_loops_fail_boundedly_without_a_regularized_answer() {
    let context = ModelContext::default();
    // Nonlinear circuit: the KVL rows of two parallel ideal sources stay
    // singular under every artificial gmin and every source scale.
    let c = circuit("v1 a 0 1\nv2 a 0 2\nd1 a 0 dm\n.model dm d");
    for continuation in [
        ContinuationPolicy::default(),
        ContinuationPolicy::ladder(),
        ContinuationPolicy::disabled(),
    ] {
        let failure = solve_dc_with(
            &c,
            &context,
            &settings(NewtonOptions::default(), continuation),
            &[],
            None,
            None,
        )
        .unwrap_err();
        assert!(matches!(failure.report.outcome, DcOutcome::Exhausted));
        assert!(!failure.report.stages.is_empty());
        assert!(
            failure
                .report
                .stages
                .iter()
                .all(|s| !(s.converged() && s.is_unregularized()))
        );
        assert!(failure.report.total_iterations <= failure.report.budget);
    }
    // Linear circuits take CKTop's continuation too (ngspice tries gmin and
    // source stepping on a singular linear matrix), and still fail.
    let linear = circuit("v1 a 0 1\nv2 a 0 2");
    let failure =
        solve_dc_with(&linear, &context, &DcSettings::default(), &[], None, None).unwrap_err();
    assert_eq!(failure.report.outcome, DcOutcome::Exhausted);
    assert_eq!(failure.report.attempts[0].outcome, AttemptOutcome::Failed);
    assert!(
        solve_dc(
            &linear,
            &context,
            &NewtonOptions::default(),
            &[],
            None,
            None
        )
        .is_err()
    );
}

#[test]
fn final_unregularized_physical_failure_is_never_masked_by_a_regularized_solve() {
    // A floating diode has no DC reference: every gmin-regularized stage
    // converges (to zero), but the original equations are singular.
    let c = circuit("d1 a b dm\n.model dm d");
    // ngspice's schedule: every strategy fails at its original-equation solve.
    let failure = solve_dc_with(
        &c,
        &ModelContext::default(),
        &DcSettings::default(),
        &[],
        None,
        None,
    )
    .unwrap_err();
    let report = &failure.report;
    assert_eq!(report.outcome, DcOutcome::Exhausted);
    assert!(report.stages.iter().any(|s| s.gmin > 0. && s.converged()));
    assert!(
        report
            .stages
            .iter()
            .all(|s| !(s.is_unregularized() && s.converged()))
    );
    let ladder = DcSettings {
        continuation: ContinuationPolicy::ladder(),
        ..DcSettings::default()
    };
    let failure =
        solve_dc_with(&c, &ModelContext::default(), &ladder, &[], None, None).unwrap_err();
    let report = &failure.report;
    assert_eq!(report.outcome, DcOutcome::Exhausted);
    let gmin: Vec<_> = report
        .stages
        .iter()
        .filter(|s| s.strategy == DcStrategy::GminStepping)
        .collect();
    assert_eq!(gmin.len(), 11);
    assert!(gmin[..10].iter().all(|s| s.gmin > 0. && s.converged()));
    let last = gmin[10];
    assert!(last.is_unregularized() && !last.converged());
    assert!(report.stages.iter().any(|s| s.gmin > 0. && s.converged()));
    assert!(
        report
            .stages
            .iter()
            .all(|s| !(s.is_unregularized() && s.converged()))
    );
    let message = failure.error.to_string();
    assert!(message.contains("gmin stepping: failed"), "{message}");
    assert!(message.contains("source stepping: failed"), "{message}");
}

#[test]
fn easy_nonlinear_and_linear_solves_stay_direct_and_exact() {
    let context = ModelContext::default();
    let c = circuit("v1 a 0 0.6\nd1 a 0 dm\n.model dm d");
    let solved = solve_dc_with(&c, &context, &DcSettings::default(), &[], None, None).unwrap();
    assert_eq!(
        solved.report.outcome,
        DcOutcome::Converged(DcStrategy::Direct)
    );
    assert_eq!(
        attempts(&solved.report),
        [(DcStrategy::Direct, AttemptOutcome::Converged)]
    );
    assert_eq!(solved.report.stages.len(), 1);
    assert_original_equations_solved(&solved.report);
    close(solved.solution.values.as_slice()[0], 0.6, 1e-9, 1e-12);
    // A disabled policy changes nothing when direct Newton already converges.
    let direct = solve_dc_with(
        &c,
        &context,
        &settings(NewtonOptions::default(), ContinuationPolicy::disabled()),
        &[],
        None,
        None,
    )
    .unwrap();
    assert_eq!(
        direct.solution.values.as_slice(),
        solved.solution.values.as_slice()
    );
    assert_eq!(direct.solution.iterations, solved.solution.iterations);

    let linear = circuit("v1 a 0 1\nr1 a b 1k\nr2 b 0 1k");
    let solved = solve_dc_with(&linear, &context, &DcSettings::default(), &[], None, None).unwrap();
    assert_eq!(solved.report.stages.len(), 1);
    // CKTop's NIiter: MODEINITJCT, MODEINITFIX, then a converged MODEINITFLOAT.
    assert_eq!(solved.solution.iterations, 3);
    close(solved.solution.values.as_slice()[1], 0.5, 1e-12, 1e-15);
    // A linear circuit is solved by the same NIiter (three passes), so a
    // total budget of one stops it and one of four solves it.
    let budget = |total| ContinuationPolicy {
        max_total_iterations: Some(IterationBudget::new(total).unwrap()),
        stage_max_iterations: None,
        ..ContinuationPolicy::disabled()
    };
    let run = |total| {
        solve_dc_with(
            &linear,
            &context,
            &settings(NewtonOptions::default(), budget(total)),
            &[],
            None,
            None,
        )
    };
    assert!(run(1).is_err());
    let solved = run(4);
    assert!(solved.is_ok(), "{:?}", solved.err().map(|e| e.error));
}

#[test]
fn trial_state_belongs_to_the_solution_and_history_is_untouched() {
    let c = circuit("v1 a 0 0.3\nd1 a 0 dm\n.model dm d(cjo=1u vj=1)");
    let context = ModelContext::default();
    let history_before = c.state_history().unwrap();
    let rhs_before = dc_rhs(&c);
    let solved = solve_dc_with(&c, &context, &DcSettings::default(), &[], None, None).unwrap();
    assert_eq!(c.state_history().unwrap(), history_before);
    assert_eq!(dc_rhs(&c), rhs_before);
    let history = c.state_history().unwrap();
    let mut trial = history.trial();
    c.load(
        &LoadRequest {
            mode: AnalysisMode::OperatingPoint,
            solution: &solved.solution.values,
            model_context: &context,
            integration: None,
            history: &history,
            forcing: None,
        },
        &mut SparseMatrix::new(c.unknown_count(), c.unknown_count()),
        &mut Vector::zeros(c.unknown_count()),
        &mut trial,
    )
    .unwrap();
    assert_eq!(trial.values(), solved.solution.trial.values());
}

#[test]
fn continuation_never_accepts_state_or_edits_sources_on_success_or_failure() {
    let accepted = Rc::new(Cell::new(0));
    let c = cubic(&accepted);
    let context = ModelContext::default();
    let rhs_before = dc_rhs(&c);
    let history_before = c.state_history().unwrap();

    // Success needs gmin continuation; an override is applied to a temporary RHS.
    let solved = solve_dc_with(
        &c,
        &context,
        &DcSettings::default(),
        &[("I1", 8.)],
        None,
        None,
    )
    .unwrap();
    close(solved.solution.values.as_slice()[0], 2., 1e-8, 1e-12);
    assert_eq!(
        solved.report.outcome,
        DcOutcome::Converged(DcStrategy::GminStepping)
    );
    assert!(solved.report.stages.len() > 1);
    assert_eq!(accepted.get(), 0);
    assert_eq!(dc_rhs(&c), rhs_before);
    assert_eq!(c.state_history().unwrap(), history_before);

    // Failure (continuation disabled) leaves the same invariants.
    let failure = solve_dc_with(
        &c,
        &context,
        &settings(NewtonOptions::default(), ContinuationPolicy::disabled()),
        &[("i1", 8.)],
        None,
        None,
    )
    .unwrap_err();
    assert_eq!(failure.report.outcome, DcOutcome::Exhausted);
    assert_eq!(accepted.get(), 0);
    assert_eq!(dc_rhs(&c), rhs_before);

    // Forcing makes the solve C's MODETRANOP bias: a source loads its time
    // function at 0- (here the PULSE's 3 V) instead of its DC value, and the
    // circuit's sources are untouched.
    let pulsed = circuit("v1 a 0 dc 1 pulse(3 5 1m)\nr1 a 0 1k");
    let pulsed_rhs = dc_rhs(&pulsed);
    let forced = solve_dc_with(
        &pulsed,
        &context,
        &DcSettings::default(),
        &[],
        None,
        Some(Forcing {
            limit: Limit::Left,
            timing: TransientTiming::new(1e-4, 1e-2).unwrap(),
        }),
    )
    .unwrap();
    close(forced.solution.values.as_slice()[0], 3., 0., 0.);
    assert_eq!(dc_rhs(&pulsed), pulsed_rhs);
    let plain = solve_dc(
        &pulsed,
        &context,
        &NewtonOptions::default(),
        &[],
        None,
        None,
    )
    .unwrap();
    close(plain.values.as_slice()[0], 1., 0., 0.);

    // Only an explicit accept runs hooks, exactly once, after the solve.
    let plain = solve_dc(&c, &context, &NewtonOptions::default(), &[], None, None).unwrap();
    close(plain.values.as_slice()[0], 1., 1e-8, 1e-12);
    assert_eq!(accepted.get(), 0);
    c.accept_solution(&plain.values, None).unwrap();
    assert_eq!(accepted.get(), 1);

    // Invalid overrides/seeds are rejected before any stage.
    let bad_overrides: [&[(&str, f64)]; 3] = [
        &[("nosuch", 1.)],
        &[("i1", f64::NAN)],
        &[("i1", 1.), ("I1", 2.)],
    ];
    for overrides in bad_overrides {
        let failure =
            solve_dc_with(&c, &context, &DcSettings::default(), overrides, None, None).unwrap_err();
        assert_eq!(failure.report.outcome, DcOutcome::Rejected);
    }
    let seed = Vector::zeros(c.unknown_count() + 1);
    assert!(solve_dc_with(&c, &context, &DcSettings::default(), &[], Some(&seed), None).is_err());
}

fn request(arguments: &[&str]) -> AnalysisInput {
    AnalysisInput::with_arguments(AnalysisKind::OperatingPoint, arguments.iter().copied())
}

fn resolved_dc(input: AnalysisInput) -> SpiceResult<DcSettings> {
    input.resolve().map(|plan| plan.dc_settings().clone())
}

#[test]
fn ngspice_maximum_gmin_steps_include_both_ladder_endpoints() {
    let stepping = NgspiceStepping::new(MAX_GMIN_STAGES, 1, 1.01, 50).unwrap();
    let ladder = stepping.fixed_gmin_ladder(1e-12).unwrap().unwrap();
    assert_eq!(ladder.len(), MAX_GMIN_STAGES + 1);
    assert!(ladder.windows(2).all(|pair| pair[0] > pair[1]));
    close(ladder[0], 1e-12 * 1.01_f64.powi(100), 1e-12, 0.);
    close(*ladder.last().unwrap(), 1e-12, 1e-12, 0.);

    // Out-of-range counts cannot reach the execution planner.
    assert!(NgspiceStepping::new(usize::MAX, 1, 1.01, 50).is_err());
    assert!(NgspiceStepping::new(1, MAX_SOURCE_STEPS + 1, 1.01, 50).is_err());
    assert!(NgspiceStepping::new(1, 1, 1., 50).is_err());
    assert!(NgspiceStepping::new(1, 1, 10., 0).is_err());
    assert!(
        NgspiceStepping::default()
            .fixed_gmin_ladder(1e-12)
            .unwrap()
            .is_none()
    );
}

#[test]
fn request_arguments_resolve_and_invalid_ones_fail() {
    let resolved = resolved_dc(request(&[
        "continuation=ladder",
        "maxiter=4",
        "RTOL=1e-6",
        "srcsteps=5",
        "gminsteps=3",
        "gminfactor=100",
    ]))
    .unwrap();
    assert_eq!(resolved.newton.max_iterations, 4);
    assert_eq!(resolved.newton.reltol, 1e-6);
    let (gmin, source) = ladder_parts(&resolved.continuation);
    assert_eq!(gmin.len(), 3);
    for (got, want) in gmin.iter().zip([1e-3, 1e-5, 1e-7]) {
        close(*got, want, 1e-14, 0.);
    }
    assert_eq!(source.unwrap().scales(), [0., 0.2, 0.4, 0.6, 0.8, 1.]);
    let off = resolved_dc(request(&[
        "srcsteps=0",
        "gminsteps=0",
        "continuation=LADDER",
    ]))
    .unwrap();
    assert_eq!(off.continuation, ContinuationPolicy::disabled());
    // The default ngspice schedule takes C's counts.
    let ngspice = resolved_dc(request(&[
        "srcsteps=0",
        "gminsteps=3",
        "gminfactor=100",
        "adaptiter=60",
        "noopiter=1",
    ]))
    .unwrap();
    assert_eq!(
        ngspice.continuation.schedule,
        ContinuationSchedule::Ngspice(NgspiceStepping::new(3, 0, 100., 60).unwrap())
    );
    assert!(ngspice.continuation.skip_direct);
    assert_eq!(resolved_dc(request(&[])).unwrap(), DcSettings::default());
    // A complete .op plan validates positional arguments too.
    assert!(resolved_dc(request(&["lin", "1"])).is_err());
    for bad in [
        "srcsteps=1.5",
        "srcsteps=-1",
        "srcsteps=1001",
        "srcsteps=nan",
        "srcsteps=abc",
        "gminsteps=101",
        "gminsteps=0.5",
        "gminfactor=1",
        "gminfactor=0.5",
        "gminfactor=1e7",
        "gminfactor=inf",
        "maxiter=0",
        "maxiter=10001",
        "bogus=1",
    ] {
        assert!(resolved_dc(request(&[bad])).is_err(), "{bad}");
    }
    for duplicate in [
        ["srcsteps=1", "SRCSTEPS=2"],
        ["gminsteps=1", "gminsteps=2"],
        ["gminfactor=2", "gminfactor=3"],
        ["maxiter=1", "maxiter=2"],
    ] {
        assert!(resolved_dc(request(&duplicate)).is_err(), "{duplicate:?}");
    }
    // A factor/steps pair whose ladder underflows is invalid as a whole.
    assert!(
        resolved_dc(request(&[
            "gminfactor=1e6",
            "gminsteps=100",
            "continuation=ladder"
        ]))
        .is_err()
    );
    // ngspice's spice3_gmin ladder starts at gmin * factor^steps, so it is
    // checked against the circuit's junction gmin when the solve starts.
    let overflowing = resolved_dc(request(&["gminfactor=1e6", "gminsteps=100"]));
    let failure = solve_dc_with(
        &circuit(DIFFICULT),
        &ModelContext::default(),
        &overflowing.unwrap(),
        &[],
        None,
        None,
    )
    .unwrap_err();
    assert_eq!(failure.report.outcome, DcOutcome::Rejected);
    assert!(failure.report.stages.is_empty());
    for bad in [
        "continuation=dynamic",
        "adaptiter=0",
        "noopiter=2",
        "noopiter=0.5",
    ] {
        assert!(resolved_dc(request(&[bad])).is_err(), "{bad}");
    }
    assert!(resolved_dc(request(&["continuation=ladder", "adaptiter=50"])).is_err());
    // The single resolver accepts both Newton and continuation options together.
    let resolved = resolved_dc(request(&["srcsteps=3", "maxiter=7"])).unwrap();
    assert_eq!(resolved.newton.max_iterations, 7);
    assert_eq!(
        resolved.continuation.schedule,
        ContinuationSchedule::Ngspice(NgspiceStepping::new(1, 3, 10., 50).unwrap(),)
    );
}

fn run_op(options: &str, explicit: &[&str]) -> SpiceResult<Plot> {
    let netlist = parse(&format!("{DIFFICULT}\n{options}"));
    let config = RunConfig::from_netlist(&netlist)?;
    let mut c = config.circuit(&netlist)?;
    let request = config.plan(request(explicit))?;
    request.run(&mut c, &config.context())
}

#[test]
fn deck_options_resolve_with_last_set_wins_and_request_precedence() {
    // Defaults: nothing is added to a request.
    let plain = RunConfig::default();
    assert_eq!(
        plain.plan(request(&[])).unwrap(),
        request(&[]).resolve().unwrap()
    );
    let ok = |plot: SpiceResult<Plot>| plot.unwrap().value("v(a)", 0).unwrap().re;

    // A four-iteration budget (request `maxiter=4`; a deck `itl1` below 100 is
    // C's effective 100) with every deck continuation disabled fails and says
    // why.
    let error = run_op(
        ".options srcsteps=0 gminsteps=0",
        &["maxiter=4", "limiting=global"],
    )
    .unwrap_err();
    let message = error.to_string();
    assert!(message.contains("source stepping: disabled"), "{message}");
    assert!(message.contains("gmin stepping: disabled"), "{message}");
    // Source stepping alone (gmin disabled) rescues the same bounded budget.
    let v = ok(run_op(
        ".options gminsteps=0",
        &["maxiter=4", "limiting=global"],
    ));
    close(v, VT * 2_f64.ln(), NIITER_ACCURACY, 1e-12);
    // Deck default options (no DC options) use the default 200 iterations;
    // the direct solve is C's (1.7e-9 from the root, see NIITER_ACCURACY).
    close(ok(run_op("", &[])), VT * 2_f64.ln(), 1e-8, 1e-12);

    // Duplicate setters: last wins, in deck order, across option cards.
    let off_then_on = run_op(
        ".options gminsteps=0 srcsteps=0\n.options srcsteps=20",
        &["maxiter=4", "limiting=global"],
    );
    close(ok(off_then_on), VT * 2_f64.ln(), NIITER_ACCURACY, 1e-12);
    let on_then_off = run_op(
        ".options gminsteps=0 srcsteps=20\n.options srcsteps=0",
        &["maxiter=4", "limiting=global"],
    );
    assert!(on_then_off.is_err());
    let netlist = parse(&format!(
        "{DIFFICULT}\n.options srcsteps=0\n.options SRCSTEPS=7 itl1=150\n.options itl1=250"
    ));
    let config = RunConfig::from_netlist(&netlist).unwrap();
    assert_eq!(config.dc().srcsteps, Some(7));
    assert_eq!(config.dc().itl1, Some(250));
    assert_eq!(config.dc().gminsteps, None);
    assert_eq!(config.applied().len(), 4);

    // Explicit request arguments beat the deck.
    let deck = ".options itl1=100 gminsteps=0 srcsteps=0";
    close(ok(run_op(deck, &[])), VT * 2_f64.ln(), 1e-8, 1e-12);
    assert!(run_op(deck, &["maxiter=4", "limiting=global"]).is_err());
    close(
        ok(run_op(
            deck,
            &["maxiter=4", "limiting=global", "srcsteps=20"],
        )),
        VT * 2_f64.ln(),
        NIITER_ACCURACY,
        1e-12,
    );
    let merged = RunConfig::from_netlist(&parse(&format!("{DIFFICULT}\n{deck} gminfactor=5")))
        .unwrap()
        .plan(request(&["srcsteps=20", "MAXITER=9"]))
        .unwrap();
    assert_eq!(merged.dc_settings().newton.max_iterations, 9);
    assert_eq!(
        merged.dc_settings().continuation.schedule,
        ContinuationSchedule::Ngspice(NgspiceStepping::new(0, 20, 5.0, 50).unwrap(),)
    );
    // A request-only configuration (no deck) works the same way.
    assert!(
        run_op(
            "",
            &["maxiter=4", "limiting=global", "srcsteps=0", "gminsteps=0"]
        )
        .is_err()
    );
    close(
        ok(run_op("", &["maxiter=4", "limiting=global", "gminsteps=0"])),
        VT * 2_f64.ln(),
        NIITER_ACCURACY,
        1e-12,
    );
}

#[test]
fn deck_dc_options_are_validated_and_unimplemented_neighbours_still_fail() {
    let config = |options: &str| RunConfig::from_netlist(&parse(&format!("r1 a 0 1k\n{options}")));
    for bad in [
        ".options itl1=10001",
        ".options itl1=abc",
        ".options srcsteps=-1",
        ".options srcsteps=1001",
        ".options gminsteps=101",
        ".options gminfactor=1",
        ".options gminfactor=0.5",
        ".options gminfactor=abc",
        ".options gminfactor=1e7",
        ".options gminfactor=1e6 gminsteps=100",
        ".options itl1",
        ".options srcsteps",
    ] {
        let error = config(bad).expect_err(bad);
        assert!(!error.is_not_yet_ported(), "{bad}: {error}");
    }
    // niiter.c raises any limit below 100 (including 0) to 100.
    // IF_INTEGER: floor(0.5 + 2.5) = 3, raised to NIiter's 100.
    for low in [
        ".options itl1=0",
        ".options itl1=1",
        ".options itl1=99",
        ".options itl1=2.5",
    ] {
        assert_eq!(config(low).unwrap().dc().itl1, Some(100), "{low}");
    }
    // Disabling is valid for the count options only.
    let off = config(".options srcsteps=0 gminsteps=0").unwrap();
    assert_eq!((off.dc().srcsteps, off.dc().gminsteps), (Some(0), Some(0)));
    assert_eq!(
        off.dc().policy().unwrap(),
        ContinuationPolicy::from_ngspice_steps(Some(0), Some(0), None, None).unwrap()
    );
    // Junction gmin, itl2, itl4 and noopiter are implemented
    // (tests/run_config.rs, tests/convergence.rs); diagonal gshunt remains an
    // explicit gap.
    assert!(
        config(".options gshunt=1e-12")
            .unwrap_err()
            .is_not_yet_ported()
    );
    let skip = config(".options noopiter").unwrap();
    assert!(skip.dc().noopiter);
    assert!(skip.dc().policy().unwrap().skip_direct);
    assert!(
        !config(".options noopiter=1")
            .unwrap_err()
            .is_not_yet_ported()
    );
    let error = config(".options itl1=5\n.options itl1").unwrap_err();
    assert!(!error.is_not_yet_ported(), "{error}");
}

#[test]
fn companion_transient_reads_dc_options() {
    let config = RunConfig::from_netlist(&parse(
        "v1 a 0 1\nr1 a 0 1k\n.options itl1=50 srcsteps=4 gminsteps=3 gminfactor=20",
    ))
    .unwrap();
    // dctran.c computes the initial bias with CKTop(CKTdcMaxIter): the companion
    // driver receives the same settings as .op.
    let tran = config
        .plan(AnalysisInput::with_arguments(
            AnalysisKind::Transient,
            ["1u", "10u"],
        ))
        .unwrap();
    assert_eq!(tran.dc_settings().newton.max_iterations, 100);
    assert_eq!(
        tran.dc_settings().continuation.schedule,
        ContinuationSchedule::Ngspice(NgspiceStepping::new(3, 4, 20.0, 50).unwrap(),)
    );
    let mut c = circuit("v1 a 0 1\nr1 a 0 1k");
    tran.run(&mut c, &AnalysisContext::default()).unwrap();
    // DC analyses accept them; invalid explicit .tran bias settings fail.
    assert!(config.plan(request(&[])).is_ok());
    let error =
        AnalysisInput::with_arguments(AnalysisKind::Transient, ["1u", "10u", "srcsteps=-5"])
            .resolve()
            .and_then(|plan| plan.run(&mut c, &AnalysisContext::default()))
            .unwrap_err();
    assert!(error.to_string().contains("srcsteps"), "{error}");
}

#[test]
fn ac_bias_uses_the_configured_continuation() {
    let ac = |extra: &[&str]| {
        let mut c = circuit("i1 0 a dc 1m ac 1\nd1 a 0 dm\n.model dm d(is=1m)");
        let mut arguments = vec!["lin", "1", "1k", "1k"];
        arguments.extend(extra);
        AnalysisInput::with_arguments(AnalysisKind::Ac, arguments)
            .resolve()
            .and_then(|plan| plan.run(&mut c, &AnalysisContext::default()))
    };
    let solved = ac(&["maxiter=4", "limiting=global", "gminsteps=0"]).unwrap();
    let conductance = 1e-3 / VT;
    close(
        solved.value("v(a)", 0).unwrap().re,
        1. / (2. * conductance),
        NIITER_ACCURACY,
        1e-9,
    );
    assert!(ac(&["maxiter=4", "limiting=global", "gminsteps=0", "srcsteps=0"]).is_err());
    assert!(ac(&["srcsteps=nan"]).is_err());
    assert!(
        ac(&[
            "maxiter=4",
            "limiting=global",
            "gminsteps=0",
            "srcsteps=0",
            "srcsteps=5"
        ])
        .is_err()
    );
}

#[test]
fn typed_fixed_source_and_custom_ladder_close_at_their_own_stage_limit() {
    let c = circuit("v1 a 0 1\nr1 a 0 1k");
    let context = ModelContext::default();
    for (schedule, expected_scales, gmin) in [
        (
            ContinuationPolicy::from_ngspice_steps(Some(4), Some(0), None, None).unwrap(),
            vec![0., 0.25, 0.5, 0.75, 1.],
            0.,
        ),
        (
            custom_ladder(Vec::new(), Some(SourceStepping::uniform(4, 1e-8).unwrap())),
            vec![0., 0.25, 0.5, 0.75, 1., 1.],
            1e-8,
        ),
    ] {
        let continuation = ContinuationPolicy {
            skip_direct: true,
            stage_max_iterations: Some(IterationLimit::new(10).unwrap()),
            ..schedule
        };
        let settings = settings(
            NewtonOptions {
                max_iterations: 1,
                ..NewtonOptions::default()
            },
            continuation,
        );
        let solved = solve_dc_with(&c, &context, &settings, &[], None, None).unwrap();
        let report = &solved.report;
        assert_eq!(
            attempts(report),
            [
                (DcStrategy::Direct, AttemptOutcome::Disabled),
                (DcStrategy::GminStepping, AttemptOutcome::Disabled),
                (DcStrategy::SourceStepping, AttemptOutcome::Converged),
            ]
        );
        assert_eq!(report.budget, expected_scales.len() * 11);
        assert_eq!(
            report
                .stages
                .iter()
                .map(|s| s.source_scale)
                .collect::<Vec<_>>(),
            expected_scales
        );
        let (closing, warmup) = report.stages.split_last().unwrap();
        assert!(warmup.iter().all(|s| s.gmin == gmin));
        assert!(closing.is_unregularized());
        assert_original_equations_solved(report);
        close(solved.solution.values.as_slice()[0], 1., 1e-12, 1e-15);
    }
}

#[test]
fn c_step_encoding_is_decoded_once_into_bounded_modes() {
    assert_eq!(
        Stepping::<MAX_GMIN_STAGES>::from_count(0).unwrap(),
        Stepping::Disabled
    );
    assert_eq!(
        Stepping::<MAX_GMIN_STAGES>::from_count(1).unwrap(),
        Stepping::Adaptive
    );
    for count in [2, MAX_GMIN_STAGES] {
        let Stepping::Fixed(steps) = Stepping::<MAX_GMIN_STAGES>::from_count(count).unwrap() else {
            panic!("fixed count");
        };
        assert_eq!(steps.get(), count);
    }
    assert!(Stepping::<MAX_GMIN_STAGES>::from_count(MAX_GMIN_STAGES + 1).is_err());
    assert!(ngspice_rs::analysis::bias::FixedSteps::<MAX_GMIN_STAGES>::new(1).is_err());
}
