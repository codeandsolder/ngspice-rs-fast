//! Linear analysis orchestration. Nonlinear SPICE convergence policies stay separate.
use crate::analysis::results::{PlotFlags, Variable};
use crate::analysis::{AnalysisPlan, Plot};
use crate::devices::{
    BranchQuantity, Circuit,
    passive_network::{CurrentOutputLaw, ReportPhase},
};
use crate::primitives::{Complex, Real, SpiceError, SpiceResult, parse_spice_number};

pub(super) fn unsupported(feature: impl Into<String>) -> SpiceError {
    SpiceError::Unsupported {
        feature: feature.into(),
        location: None,
    }
}
pub(super) fn number(text: Option<&str>, name: &str) -> SpiceResult<Real> {
    text.and_then(parse_spice_number)
        .filter(|v| v.is_finite())
        .ok_or_else(|| unsupported(format!("expected finite {name}")))
}

/// The output schema and additional affine projections of the physical solution.
/// Only the output path sees these projections; they do not add MNA coordinates.
pub(super) struct LinearOutput {
    pub(super) header: Plot,
    pub(super) reports: Vec<CurrentOutputLaw>,
    first_unknown: usize,
}

impl LinearOutput {
    pub(super) fn append_reports(&self, point: &mut Vec<Complex>) {
        for &report in &self.reports {
            let value = report.evaluate(|row| point[self.first_unknown + row]);
            point.push(value);
        }
    }
}

pub(super) fn plot(
    circuit: &Circuit,
    name: &str,
    title: &str,
    scale: Option<(&str, &str)>,
    complex: bool,
    phase: ReportPhase,
) -> SpiceResult<LinearOutput> {
    let mut plot = Plot::builder(
        name,
        title,
        if complex {
            PlotFlags::Complex
        } else {
            PlotFlags::Real
        },
    );
    let var = |name: String, unit: &str| {
        if complex {
            Variable::complex(name, unit)
        } else {
            Variable::new(name, unit)
        }
    };
    if let Some((name, unit)) = scale {
        plot.push_variable(var(name.into(), unit));
    }
    // One vector per unknown, in row order: the order C's node list (and so
    // its rawfile) has, since rows are numbered as `CKTsetup` creates nodes.
    let mut names: Vec<Option<(String, &str)>> = vec![None; circuit.unknown_count()];
    for node in circuit.nodes().nodes() {
        if let Some(row) = circuit.unknowns().node_row(node.id) {
            let entry = names.get_mut(row).ok_or_else(|| {
                SpiceError::circuit(format!(
                    "node {} references nonexistent MNA row {row}",
                    node.name
                ))
            })?;
            *entry = Some((format!("v({})", node.name), "voltage"));
        }
    }
    for (index, device) in circuit.devices().iter().enumerate() {
        // Only physical branch currents use i(device). Native storage-rate
        // and normalized transfer coordinates carry their own quantities.
        // Legacy multiport reference models retain their C port labels.
        let rows = circuit.branch_rows(index).ok_or_else(|| {
            SpiceError::circuit(format!(
                "device {} is missing its MNA branch rows",
                device.name()
            ))
        })?;
        let report = circuit.current_report(index);
        for (local, row) in rows.enumerate() {
            let entry = names.get_mut(row).ok_or_else(|| {
                SpiceError::circuit(format!(
                    "device {} references nonexistent MNA branch row {row}",
                    device.name()
                ))
            })?;
            let quantity = device.branch_quantity(local);
            let name = match report {
                Some(report) if quantity == BranchQuantity::Current => &report.intrinsic,
                _ => device.name(),
            };
            *entry = Some(quantity.observation(name));
        }
    }
    for (name, unit) in names.into_iter().flatten() {
        plot.push_variable(var(name, unit));
    }
    let mut reports = Vec::new();
    for report in circuit.reported_currents() {
        plot.push_variable(var(format!("i({})", report.original), "current"));
        reports.push(report.law(phase)?);
    }
    for name in circuit.absent_storage_voltage_nodes() {
        plot.push_variable(var(format!("v({name})"), "voltage"));
        reports.push(CurrentOutputLaw::ZERO);
    }
    for name in circuit.absent_storage_currents() {
        plot.push_variable(var(format!("i({name})"), "current"));
        reports.push(CurrentOutputLaw::ZERO);
    }
    for observation in circuit.observations() {
        plot.push_variable(Variable::new(observation.name(), observation.unit()));
    }
    Ok(LinearOutput {
        header: plot.build(),
        reports,
        first_unknown: usize::from(scale.is_some()),
    })
}

pub(super) fn op(
    circuit: &mut Circuit,
    request: &AnalysisPlan,
    context: &crate::analysis::AnalysisContext,
) -> SpiceResult<Plot> {
    op_inner(circuit, request, context, None)
}

pub(super) fn mixed_op(
    circuit: &mut Circuit,
    request: &AnalysisPlan,
    context: &crate::analysis::AnalysisContext,
    mixed: &mut super::digital::MixedSignal,
) -> SpiceResult<Plot> {
    circuit.finalize()?;
    mixed.activate(circuit)?;
    let result = op_inner(circuit, request, context, Some(mixed));
    mixed.deactivate();
    result
}

fn op_inner(
    circuit: &mut Circuit,
    request: &AnalysisPlan,
    context: &crate::analysis::AnalysisContext,
    mixed: Option<&mut super::digital::MixedSignal>,
) -> SpiceResult<Plot> {
    // .ic is a transient-only constraint in C (cktload.c, MODETRANOP); a
    // .nodeset is forced in the MODEINITJCT/MODEINITFIX loads of a nonlinear
    // solve and cannot change a linear operating point. Both are validated.
    circuit.finalize()?;
    let model = context.model_context();
    let prepared = circuit.prepare_run();
    let start = super::bias::OperatingPointStart::resolve(circuit, request)?;
    let mut x = start
        .solve(&prepared, &model, request.dc_settings())?
        .values;
    if let Some(mixed) = mixed {
        let mut settled = false;
        for _ in 0..request.dc_settings().newton.max_iterations {
            let point = mixed.prepare_point(circuit, &x, 0., None)?;
            let changed = mixed.initial_changed(&point)?;
            mixed.commit_point(point);
            if !changed {
                mixed.accept_initial();
                settled = true;
                break;
            }
            x = start
                .solve_from(&prepared, &model, request.dc_settings(), &x)?
                .values;
        }
        if !settled {
            return Err(SpiceError::circuit(
                "mixed operating point failed to settle",
            ));
        }
    }
    let output = plot(
        circuit,
        "op1",
        "Operating Point",
        None,
        false,
        ReportPhase::Steady,
    )?;
    circuit.accept_solution(&x, None)?;
    let history = circuit.state_history()?;
    let values = prepared.observe_real(
        &crate::devices::LoadRequest {
            mode: crate::devices::AnalysisMode::OperatingPoint,
            solution: &x,
            model_context: &model,
            integration: None,
            history: &history,
            forcing: None,
        },
        &[],
        None,
        None,
    )?;
    let mut point: Vec<_> = x.as_slice().iter().map(|v| Complex::real(*v)).collect();
    output.append_reports(&mut point);
    point.extend(values);
    let mut plot = output.header;
    plot.push_point(&point)?;
    // dcop.c performs MODEINITSMSIG after CKTop. Publish its persistent
    // device fields for the next analysis, just as the AC preparation does.
    prepared.record_small_signal_state(&model, &x)?;
    Ok(plot)
}
