//! Linear analysis orchestration. Nonlinear SPICE convergence policies stay separate.
use crate::analysis::results::{PlotFlags, Variable};
use crate::analysis::{AnalysisPlan, Plot};
use crate::devices::Circuit;
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

pub(super) fn plot(
    circuit: &Circuit,
    name: &str,
    title: &str,
    scale: Option<(&str, &str)>,
    complex: bool,
) -> SpiceResult<Plot> {
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
        for (local, row) in rows.enumerate() {
            let entry = names.get_mut(row).ok_or_else(|| {
                SpiceError::circuit(format!(
                    "device {} references nonexistent MNA branch row {row}",
                    device.name()
                ))
            })?;
            *entry = Some(device.branch_quantity(local).observation(device.name()));
        }
    }
    for (name, unit) in names.into_iter().flatten() {
        plot.push_variable(var(name, unit));
    }
    for observation in circuit.observations() {
        plot.push_variable(Variable::new(observation.name(), observation.unit()));
    }
    Ok(plot.build())
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
    let mut plot = plot(circuit, "op1", "Operating Point", None, false)?;
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
    point.extend(values);
    plot.push_point(&point)?;
    // dcop.c performs MODEINITSMSIG after CKTop. Publish its persistent
    // device fields for the next analysis, just as the AC preparation does.
    prepared.record_small_signal_state(&model, &x)?;
    Ok(plot)
}
