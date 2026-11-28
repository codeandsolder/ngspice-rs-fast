//! DC continuation with structural candidates. Every coordinate is validated
//! before solving; accepted state is transferred only between compatible
//! physical layouts. This uses the ordinary DC Newton/accept path, not OP runs.
use super::output::Columns;
use super::{
    AnalysisContext, AnalysisPlan, AxisTarget, Circuit, Complex, DcAxisTarget, GridPair,
    PointContext, Real, SpiceResult, SweepSolver, Vector, Written, plot, resolve_target,
    unsupported, warm_start_settings,
};
use crate::analysis::results::{Plot, PlotFlags};

type Candidate = (Circuit, AnalysisPlan, AnalysisContext);

fn scale(circuit: &Circuit, target: &AxisTarget) -> SpiceResult<(&'static str, &'static str)> {
    match target {
        AxisTarget::Typed(DcAxisTarget::Parameter(_) | DcAxisTarget::Model { .. }) => {
            Ok(("param-sweep", "notype"))
        }
        _ => {
            let written = Written::classify(circuit, target)?;
            Ok(resolve_target(circuit, &written)?.0.scale())
        }
    }
}

fn model(circuit: &Circuit, context: &AnalysisContext) -> crate::devices::ModelContext {
    let kind = circuit
        .requires_analysis_context()
        .then_some(crate::primitives::AnalysisKind::DcSweep);
    AnalysisContext {
        analysis_kind: kind,
        bias_analysis: kind,
        ..*context
    }
    .model_context()
}

fn validate(candidate: &mut Candidate) -> SpiceResult<()> {
    let (circuit, request, context) = candidate;
    circuit.finalize()?;
    circuit.validate_prepared_analysis(request.kind(), request.uic())?;
    if circuit.prepare_native_mixed()?.is_some() {
        return Err(unsupported(
            "structural DC sweep of a native mixed-signal circuit",
        ));
    }
    crate::analysis::initial::resolve(circuit, request)?;
    circuit
        .prepare_run()
        .validate_dc_inputs(&model(circuit, context))
}

/// Re-elaborate one candidate per structural coordinate. A validation pass
/// keeps failed coordinate preparation from leaking a partial result. The
/// union of columns is established before publishing the first solved row.
pub(in crate::analysis) fn run(
    request: &AnalysisPlan,
    mut candidate: impl FnMut([Real; 2]) -> SpiceResult<Candidate>,
) -> SpiceResult<Plot> {
    let super::super::plan::PlanKind::Dc { axes, .. } = request.analysis() else {
        return Err(unsupported("structural sweep requires DC continuation"));
    };
    if !axes.structural() {
        return Err(unsupported("no structural continuation target"));
    }
    let mut columns = Columns::default();
    let header = |circuit: &Circuit| {
        plot(
            circuit,
            "dc1",
            "DC transfer characteristic",
            Some(scale(circuit, &axes.axes[0].target)?),
            false,
            crate::devices::passive_network::ReportPhase::Steady,
        )
    };
    for coordinates in GridPair::new(&axes.grids).points() {
        let mut next = candidate(coordinates)?;
        validate(&mut next)?;
        columns.map(&header(&next.0)?.header);
    }
    let mut result = Plot::new(
        "dc1",
        "DC transfer characteristic",
        PlotFlags::Real,
        columns.variables.clone(),
    );
    result.reserve_points(axes.point_count())?;
    let mut row = vec![Complex::real(Real::NAN); result.variable_count()];
    let mut values_row = Vec::new();
    let mut previous_circuit: Option<Circuit> = None;
    let mut previous = Vector::zeros(0);
    let mut history: Option<crate::devices::StateHistory> = None;
    let mut workspace = crate::analysis::newton::CircuitNewtonWorkspace::default();
    let GridPair { inner, outer } = GridPair::new(&axes.grids);
    for outer_value in outer {
        let mut first = true;
        for inner_value in inner.clone() {
            let (mut circuit, request, context) = candidate([inner_value, outer_value])?;
            circuit.finalize()?;
            let hints = crate::analysis::initial::resolve(&circuit, &request)?;
            let nodes = crate::analysis::bias::NodeForcing {
                initial: Vec::new(),
                nodesets: crate::analysis::initial::forced_nodesets(&hints.nodesets),
            };
            let rebound = match (&previous_circuit, &history) {
                (Some(old), Some(history)) => circuit.continuation_history(old, history)?,
                _ => None,
            };
            let mut next_history = if let Some(rebound) = rebound {
                rebound
            } else {
                // Allocation or state interpretation changed. Do not copy
                // unidentified positional values into the new physical layout.
                previous = Vector::zeros(circuit.unknown_count());
                for hint in hints.nodesets {
                    previous.as_mut_slice()[hint.row] = hint.value;
                }
                first = true;
                circuit.independent_state_history()?
            };
            let settings = request.dc_settings();
            let warm = warm_start_settings(request.point_iterations(), settings);
            let point = PointContext {
                model: model(&circuit, &context),
                sources: [("", 0.); 2],
                source_count: 0,
            };
            let solver = SweepSolver {
                prepared: circuit.prepare_run(),
                settings,
                warm_settings: warm.as_ref(),
                nodes: &nodes,
            };
            let (x, trial) =
                solver.solve(&point, &previous, &next_history, first, &mut workspace)?;
            let load = crate::devices::LoadRequest {
                mode: crate::devices::AnalysisMode::DcSweep,
                solution: &x,
                model_context: &point.model,
                integration: None,
                history: &next_history,
                forcing: None,
            };
            let observed = solver
                .prepared
                .observe_real(&load, &[], None, Some(&trial))?;
            circuit.accept_point(&x, None, &mut next_history, trial)?;
            if std::mem::replace(&mut first, false) {
                next_history.retain_latest();
            }
            let output = header(&circuit)?;
            let mapping = columns.map(&output.header);
            row.fill(Complex::real(Real::NAN));
            values_row.clear();
            values_row.push(Complex::real(inner_value));
            values_row.extend(x.as_slice().iter().map(|&v| Complex::real(v)));
            output.append_reports(&mut values_row);
            values_row.extend(observed);
            for (index, value) in mapping.into_iter().zip(values_row.iter().copied()) {
                *row.get_mut(index).ok_or_else(|| {
                    unsupported("structural DC output schema changed after validation")
                })? = value;
            }
            result.push_point(&row)?;
            workspace.recycle_current(std::mem::replace(&mut previous, x));
            history = Some(next_history);
            previous_circuit = Some(circuit);
        }
    }
    Ok(result)
}
