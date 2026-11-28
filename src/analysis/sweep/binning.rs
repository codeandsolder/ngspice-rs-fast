//! Transactional sweep setup for geometry-dependent model declarations.
use super::{
    AnalysisContext, AnalysisPlan, AxisBinding, BoundAxis, Circuit, Complex, DcSweepPlan, GridPair,
    ModelContext, ParameterRoute, Real, SpiceResult, SweepSolver, SweepTarget, Vector, plot,
    point_context, unsupported, warm_start_settings,
};
use crate::analysis::results::{Plot, PlotFlags, Variable};
use crate::devices::circuit::GeometryEdit;
use crate::devices::models::ModelResolver;
use std::collections::BTreeMap;

pub(super) fn active(axes: &[BoundAxis]) -> bool {
    axes.iter()
        .any(|axis| matches!(axis.binding, AxisBinding::Geometry))
}

struct Candidate {
    circuit: Circuit,
    axes: Vec<BoundAxis>,
    base: ModelContext,
    key: [Option<u64>; 2],
}

fn geometry_edits(axes: &[BoundAxis], values: [Real; 2]) -> Vec<GeometryEdit<'_>> {
    axes.iter()
        .zip(values)
        .filter_map(|(axis, value)| {
            if let SweepTarget::InstanceParameter {
                instance,
                parameter,
                route: ParameterRoute::Geometry,
                ..
            } = &axis.spec.target
            {
                Some(GeometryEdit {
                    instance,
                    parameter,
                    value,
                })
            } else {
                None
            }
        })
        .collect()
}

fn candidate(
    baseline: &Circuit,
    resolver: &ModelResolver<'_>,
    axes: &[BoundAxis],
    values: [Real; 2],
    base: ModelContext,
) -> SpiceResult<Candidate> {
    let edits = geometry_edits(axes, values);
    let deck = baseline
        .binned_deck()
        .ok_or_else(|| unsupported("missing binned source graph"))?;
    let mut circuit = deck.candidate(baseline, resolver, &edits, &base)?;
    let base = circuit.rebind_sweep_context(baseline, base)?;
    let mut bound = axes.to_vec();
    for axis in &mut bound {
        let (name, keyword) = match &axis.spec.target {
            SweepTarget::InstanceParameter {
                instance,
                parameter,
                ..
            } => (instance.as_str(), parameter.as_str()),
            target => (target.name(), ""),
        };
        axis.binding = match axis.binding {
            AxisBinding::Resistor(_) => AxisBinding::Resistor(circuit.bind_resistor_target(name)?),
            AxisBinding::Source(_) => AxisBinding::Source(circuit.bind_source_target(name)?),
            AxisBinding::Device(_) => {
                AxisBinding::Device(circuit.bind_instance_target(name, keyword)?)
            }
            other => other,
        };
    }
    Ok(Candidate {
        circuit,
        axes: bound,
        base,
        key: key(axes, values),
    })
}

fn key(axes: &[BoundAxis], values: [Real; 2]) -> [Option<u64>; 2] {
    std::array::from_fn(|i| {
        axes.get(i)
            .filter(|axis| matches!(axis.binding, AxisBinding::Geometry))
            .map(|_| values[i].to_bits())
    })
}

fn advance(
    current: &mut Option<Candidate>,
    baseline: &Circuit,
    resolver: &ModelResolver<'_>,
    axes: &[BoundAxis],
    values: [Real; 2],
    base: ModelContext,
) -> SpiceResult<bool> {
    let next = key(axes, values);
    if let Some(previous) = current {
        if previous.key == next {
            return Ok(false);
        }
        let deck = baseline
            .binned_deck()
            .ok_or_else(|| unsupported("missing binned source graph"))?;
        if deck.try_update(
            baseline,
            &mut previous.circuit,
            resolver,
            &geometry_edits(axes, values),
            &base,
        )? {
            previous.key = next;
            return Ok(false);
        }
    }
    *current = Some(candidate(baseline, resolver, axes, values, base)?);
    Ok(true)
}

pub(super) fn validate_start(
    baseline: &Circuit,
    axes: &[BoundAxis],
    base: ModelContext,
) -> SpiceResult<()> {
    let resolver = baseline
        .binned_deck()
        .ok_or_else(|| unsupported("missing binned source graph"))?
        .resolver(base.geometry_scale)?;
    let values = [
        axes[0].spec.start,
        axes.get(1).map_or(0., |axis| axis.spec.start),
    ];
    let candidate = candidate(baseline, &resolver, axes, values, base)?;
    let point = point_context(&candidate.circuit, &candidate.axes, values, candidate.base)?;
    candidate
        .circuit
        .prepare_run()
        .validate_context(&point.model)
}

fn header(
    circuit: &Circuit,
    axes: &[BoundAxis],
) -> SpiceResult<crate::analysis::linear::LinearOutput> {
    plot(
        circuit,
        "dc1",
        "DC transfer characteristic",
        Some(axes[0].spec.target.scale()),
        false,
        crate::devices::passive_network::ReportPhase::Steady,
    )
}

/// Include every topology's columns in deterministic first-appearance order.
/// An internal unknown which does not exist at a point has no numeric value.
#[derive(Default)]
struct Columns {
    variables: Vec<Variable>,
    indices: BTreeMap<(String, String, bool, usize), usize>,
}

impl Columns {
    fn map(&mut self, plot: &Plot) -> Vec<usize> {
        let mut occurrences = BTreeMap::new();
        plot.variables()
            .iter()
            .map(|variable| {
                let identity = (
                    variable.name.clone(),
                    variable.unit.clone(),
                    variable.is_real,
                );
                let count = occurrences.entry(identity.clone()).or_insert(0usize);
                let key = (identity.0, identity.1, identity.2, *count);
                *count += 1;
                *self.indices.entry(key).or_insert_with(|| {
                    let index = self.variables.len();
                    self.variables.push(variable.clone());
                    index
                })
            })
            .collect()
    }
}

fn preflight(
    baseline: &Circuit,
    request: &AnalysisPlan,
    sweep: &DcSweepPlan,
    axes: &[BoundAxis],
    resolver: &ModelResolver<'_>,
    base: ModelContext,
) -> SpiceResult<Columns> {
    let source_axes =
        std::array::from_fn(|i| axes.get(i).is_some_and(|axis| axis.binding.is_source()));
    let mut columns = Columns::default();
    let mut current: Option<Candidate> = None;
    // Validate every distinct geometry and physical context before publishing a header.
    for values in GridPair::new(&sweep.grids)
        .physical_contexts(source_axes)
        .points()
    {
        if advance(&mut current, baseline, resolver, axes, values, base)? {
            let c = current
                .as_ref()
                .ok_or_else(|| unsupported("missing geometry candidate"))?;
            columns.map(&header(&c.circuit, axes)?.header);
            crate::analysis::initial::resolve(&c.circuit, request)?;
        }
        let c = current
            .as_ref()
            .ok_or_else(|| unsupported("missing geometry candidate"))?;
        let point = point_context(&c.circuit, &c.axes, values, c.base)?;
        c.circuit.prepare_run().validate_dc_inputs(&point.model)?;
    }
    Ok(columns)
}

fn projection(
    sink: &dyn crate::analysis::PointSink,
    result: &Plot,
) -> SpiceResult<Option<Vec<usize>>> {
    let selected = sink.selected_columns().map(<[usize]>::to_vec);
    if selected.as_ref().is_some_and(|indices| {
        indices
            .iter()
            .any(|&index| index >= result.variable_count())
            || indices.windows(2).any(|pair| pair[0] >= pair[1])
    }) {
        return Err(unsupported("invalid geometry sweep projection"));
    }
    Ok(selected)
}

fn emit(
    sink: &mut dyn crate::analysis::PointSink,
    result: &Plot,
    row: &[Complex],
    selected: Option<&[usize]>,
    projected: &mut Vec<Complex>,
) -> SpiceResult<()> {
    if let Some(selected) = selected {
        projected.clear();
        for &index in selected {
            projected.push(
                *row.get(index)
                    .ok_or_else(|| unsupported("invalid geometry sweep projection"))?,
            );
        }
        sink.point(projected)
    } else {
        result.check_point(row)?;
        sink.point(row)
    }
}

pub(super) fn run(
    baseline: &Circuit,
    request: &AnalysisPlan,
    context: &AnalysisContext,
    sweep: &DcSweepPlan,
    axes: &[BoundAxis],
    point_iterations: Option<usize>,
    sink: &mut dyn crate::analysis::PointSink,
) -> SpiceResult<()> {
    let base = context.model_context();
    let resolver = baseline
        .binned_deck()
        .ok_or_else(|| unsupported("missing binned source graph"))?
        .resolver(base.geometry_scale)?;
    let mut columns = preflight(baseline, request, sweep, axes, &resolver, base)?;
    let result = Plot::new(
        "dc1",
        "DC transfer characteristic",
        PlotFlags::Real,
        columns.variables.clone(),
    );
    sink.begin(&result)?;
    let selected = projection(sink, &result)?;
    let settings = request.dc_settings();
    let warm = warm_start_settings(point_iterations, settings);
    let mut current = None;
    let mut previous = Vector::zeros(0);
    let mut workspace = crate::analysis::newton::CircuitNewtonWorkspace::default();
    let mut history = None;
    let mut nodes = crate::analysis::bias::NodeForcing {
        initial: Vec::new(),
        nodesets: Vec::new(),
    };
    let mut mapping = Vec::new();
    let mut output = None;
    let mut values_row = Vec::new();
    let mut row = vec![Complex::real(Real::NAN); result.variable_count()];
    let mut projected = Vec::new();
    let GridPair { inner, outer } = GridPair::new(&sweep.grids);
    for outer_value in outer {
        let mut first = true;
        for inner_value in inner.clone() {
            let values = [inner_value, outer_value];
            if advance(&mut current, baseline, &resolver, axes, values, base)? {
                let c = current
                    .as_ref()
                    .ok_or_else(|| unsupported("missing geometry candidate"))?;
                let hints = crate::analysis::initial::resolve(&c.circuit, request)?;
                nodes.nodesets = crate::analysis::initial::forced_nodesets(&hints.nodesets);
                previous = Vector::zeros(c.circuit.unknown_count());
                for hint in hints.nodesets {
                    previous.as_mut_slice()[hint.row] = hint.value;
                }
                history = Some(c.circuit.state_history()?);
                let bound = header(&c.circuit, axes)?;
                mapping = columns.map(&bound.header);
                output = Some(bound);
                first = true;
            }
            let c = current
                .as_ref()
                .ok_or_else(|| unsupported("missing geometry candidate"))?;
            let history = history
                .as_mut()
                .ok_or_else(|| unsupported("missing geometry history"))?;
            let point = point_context(&c.circuit, &c.axes, values, c.base)?;
            let solver = SweepSolver {
                prepared: c.circuit.prepare_run(),
                settings,
                warm_settings: warm.as_ref(),
                nodes: &nodes,
            };
            let solution = solver.solve(&point, &previous, history, first, &mut workspace)?;
            let x = solution.values;
            let trial = solution.trial;
            let load = crate::devices::LoadRequest {
                mode: crate::devices::AnalysisMode::DcSweep,
                solution: &x,
                model_context: &point.model,
                integration: None,
                history,
                forcing: None,
            };
            let observed =
                solver
                    .prepared
                    .observe_real(&load, point.source_values(), None, Some(&trial))?;
            c.circuit.accept_point(&x, None, history, trial)?;
            if std::mem::replace(&mut first, false) {
                history.retain_latest();
            }
            row.fill(Complex::real(Real::NAN));
            values_row.clear();
            values_row.push(Complex::real(inner_value));
            values_row.extend(x.as_slice().iter().map(|v| Complex::real(*v)));
            output
                .as_ref()
                .ok_or_else(|| unsupported("missing geometry output"))?
                .append_reports(&mut values_row);
            values_row.extend(observed);
            for (index, value) in mapping.iter().copied().zip(values_row.iter().copied()) {
                *row.get_mut(index).ok_or_else(|| {
                    unsupported("geometry sweep schema changed after validation")
                })? = value;
            }
            emit(sink, &result, &row, selected.as_deref(), &mut projected)?;
            workspace.recycle_current(std::mem::replace(&mut previous, x));
            workspace.recycle_next(solution.next);
        }
    }
    sink.end()
}
