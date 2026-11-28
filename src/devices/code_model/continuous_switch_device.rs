//! Native continuous switch binding and signed MNA residual/Jacobian.

use super::continuous_switch::{Material, Point};
use super::source::SourceModel;
use crate::devices::models::{PreparationKey, ResolvedModel};
use crate::devices::setup::MatrixSetup;
use crate::devices::traits::{ChargeContext, ControlReference, ConvergenceContext, MnaUnknowns};
use crate::devices::{
    BranchQuantity, Device, InitialVoltageLaw, InitialVoltageRelation, LinearContext, StampContext,
};
use crate::maths::Vector;
use crate::netlist::ast::{CodePortKind, CodeTerminal, DeviceInstance};
use crate::primitives::{NodeAppend, NodeId, Real, SpiceError, SpiceResult};

#[derive(Debug)]
enum Control {
    Voltage,
    Current,
    Named(ControlReference),
}

#[derive(Debug)]
struct ContinuousSwitch {
    name: String,
    model: String,
    material: Material,
    control: Control,
    // Output and control pairs; a named branch has no control nodes.
    terminals: [NodeId; 4],
}

#[derive(Clone, Copy, Debug)]
struct Coordinates {
    output: [Option<usize>; 2],
    input: [Option<usize>; 2],
    control: [Option<usize>; 2],
    ammeter: Option<usize>,
}

fn read_pair(rows: [Option<usize>; 2], values: &Vector) -> SpiceResult<Real> {
    let read = |row: Option<usize>| {
        row.map_or(Ok(0.), |row| {
            values
                .get(row)
                .ok_or_else(|| SpiceError::circuit("unbound continuous switch coordinate"))
        })
    };
    Ok(read(rows[0])? - read(rows[1])?)
}

impl Coordinates {
    fn sample(self, values: &Vector) -> SpiceResult<(Real, Real)> {
        Ok((
            read_pair(self.output, values)?,
            read_pair(self.control, values)?,
        ))
    }

    fn entries(
        self,
        g: Real,
        gm: Real,
        input_g: Real,
        mut emit: impl FnMut(usize, usize, Real) -> SpiceResult<()>,
    ) -> SpiceResult<()> {
        // Preserve the full Jacobian structure, even when the operating point
        // is on a constant-resistance segment or the output voltage is zero.
        let signed = |rows: [Option<usize>; 2]| [(rows[0], 1.), (rows[1], -1.)];
        for (row, sign) in signed(self.output) {
            let Some(row) = row else { continue };
            for (col, polarity) in signed(self.output) {
                if let Some(col) = col {
                    emit(row, col, sign * polarity * g)?;
                }
            }
            for (col, polarity) in signed(self.control) {
                if let Some(col) = col {
                    emit(row, col, sign * polarity * gm)?;
                }
            }
        }
        if input_g != 0. {
            for (row, sign) in signed(self.input) {
                let Some(row) = row else { continue };
                for (col, polarity) in signed(self.input) {
                    if let Some(col) = col {
                        emit(row, col, sign * polarity * input_g)?;
                    }
                }
            }
        }
        if let Some(branch) = self.ammeter {
            for (row, sign) in signed(self.input) {
                if let Some(row) = row {
                    emit(row, branch, sign)?;
                    emit(branch, row, sign)?;
                }
            }
        }
        Ok(())
    }
}

impl ContinuousSwitch {
    const fn input(&self) -> [NodeId; 2] {
        [self.terminals[2], self.terminals[3]]
    }

    fn coordinates(
        &self,
        node: impl Fn(NodeId) -> Option<usize>,
        branch: Option<usize>,
        controls: &[usize],
    ) -> SpiceResult<Coordinates> {
        let output = [node(self.terminals[0]), node(self.terminals[1])];
        let input = self.input().map(node);
        let (control, ammeter) = match &self.control {
            Control::Voltage => (input, None),
            Control::Current => {
                let row = branch
                    .ok_or_else(|| SpiceError::circuit("missing switch current input branch"))?;
                ([Some(row), None], Some(row))
            }
            Control::Named(_) => {
                let [row] = controls else {
                    return Err(SpiceError::circuit("missing named switch current input"));
                };
                ([Some(*row), None], None)
            }
        };
        Ok(Coordinates {
            output,
            input,
            control,
            ammeter,
        })
    }

    fn linearization(
        &self,
        coordinates: Coordinates,
        values: &Vector,
    ) -> SpiceResult<(Point, Real, Real)> {
        let (voltage, control) = coordinates.sample(values)?;
        let point = self.material.point(control)?;
        let gm = point.dg * voltage;
        if !voltage.is_finite() || !gm.is_finite() {
            return Err(SpiceError::circuit("nonfinite continuous switch Jacobian"));
        }
        Ok((point, gm, control))
    }

    fn mna_coordinates(
        &self,
        unknowns: &MnaUnknowns,
        branch: Option<usize>,
        controls: &[usize],
    ) -> SpiceResult<Coordinates> {
        self.coordinates(|node| unknowns.node_row(node), branch, controls)
    }
}

impl Device for ContinuousSwitch {
    fn name(&self) -> &str {
        &self.name
    }

    fn designator(&self) -> char {
        'a'
    }

    fn model_name(&self) -> Option<&str> {
        Some(&self.model)
    }

    fn terminals(&self) -> &[NodeId] {
        if matches!(self.control, Control::Named(_)) {
            &self.terminals[..2]
        } else {
            &self.terminals
        }
    }

    fn branch_currents(&self) -> usize {
        usize::from(matches!(self.control, Control::Current))
    }

    fn branch_quantity(&self, _branch: usize) -> BranchQuantity {
        BranchQuantity::InputCurrent
    }

    fn controlling_sources(&self) -> &[ControlReference] {
        match &self.control {
            Control::Named(reference) => std::slice::from_ref(reference),
            _ => &[],
        }
    }

    fn initial_voltage_relations(&self, _inductors_short: bool) -> Vec<InitialVoltageRelation> {
        if matches!(self.control, Control::Current) {
            vec![InitialVoltageRelation {
                terminals: self.input(),
                law: InitialVoltageLaw::Fixed(0.),
            }]
        } else {
            Vec::new()
        }
    }

    fn is_nonlinear(&self) -> bool {
        true
    }

    fn limits_voltage_steps(&self) -> bool {
        false
    }

    fn setup_matrix(&self, setup: &mut MatrixSetup<'_>) -> SpiceResult<bool> {
        let coordinates = self.coordinates(
            |node| setup.node(node),
            setup.branches.clone().next(),
            setup.controls,
        )?;
        coordinates.entries(0., 0., self.material.control_g, |row, col, _| {
            setup.element(Some(row), Some(col));
            Ok(())
        })?;
        Ok(true)
    }

    fn stamp(&self, context: &mut StampContext<'_>) -> SpiceResult<()> {
        if context.mode.is_ac() {
            return Err(SpiceError::circuit("continuous switch AC requires a bias"));
        }
        let coordinates = self.mna_coordinates(
            context.unknowns,
            context.branches.clone().next(),
            context.controls,
        )?;
        let (point, gm, control) = self.linearization(coordinates, context.solution)?;
        let rhs = gm * control;
        if !rhs.is_finite() {
            return Err(SpiceError::circuit("nonfinite continuous switch residual"));
        }
        coordinates.entries(point.g, gm, self.material.control_g, |row, col, value| {
            context.matrix.add(row, col, value)
        })?;
        // J*x - I(x) = dI/dc * c, since I = G(c) * V.
        for (row, sign) in [(coordinates.output[0], 1.), (coordinates.output[1], -1.)] {
            if let Some(row) = row {
                context.rhs.add_to(row, sign * rhs)?;
            }
        }
        Ok(())
    }

    fn convergence_test(&self, context: &ConvergenceContext<'_>) -> SpiceResult<bool> {
        let coordinates =
            self.mna_coordinates(context.unknowns, context.branch, context.controls)?;
        let (voltage, control) = coordinates.sample(context.solution)?;
        let (prior, gm, prior_control) = self.linearization(coordinates, context.previous)?;
        let actual = self.material.point(control)?.g * voltage;
        let predicted = gm.mul_add(control - prior_control, prior.g * voltage);
        if !actual.is_finite() || !predicted.is_finite() {
            return Err(SpiceError::circuit(
                "nonfinite continuous switch convergence residual",
            ));
        }
        let tolerance = context
            .reltol
            .mul_add(actual.abs().max(predicted.abs()), context.abstol);
        Ok((actual - predicted).abs() > tolerance)
    }

    fn assemble_small_signal(
        &self,
        context: &mut LinearContext<'_>,
        bias: &Vector,
    ) -> SpiceResult<()> {
        let coordinates =
            self.mna_coordinates(context.unknowns, context.branch, context.controls)?;
        let (point, gm, _) = self.linearization(coordinates, bias)?;
        coordinates.entries(point.g, gm, self.material.control_g, |row, col, value| {
            context.system.a.add(row, col, value)
        })
    }

    fn assemble_pole_zero(
        &self,
        context: &mut LinearContext<'_>,
        bias: &Vector,
    ) -> SpiceResult<()> {
        self.assemble_small_signal(context, bias)
    }

    fn assemble_nonlinear_taylor(
        &self,
        context: &mut crate::devices::physical_dae::TaylorContext<'_>,
    ) -> SpiceResult<()> {
        use crate::devices::physical_dae::TaylorControl;
        let coordinates =
            self.mna_coordinates(context.unknowns, context.branch, context.controls)?;
        let controls = [coordinates.output, coordinates.control]
            .map(|rows| TaylorControl { rows, polarity: 1. });
        let voltage = controls[0].input::<2>(0, context)?;
        let control = controls[1].input::<2>(1, context)?;
        let mut guards = Vec::new();
        let conductance = self.material.taylor(control, &mut guards)?;
        if let Some(trace) = &mut context.output.guards {
            trace.extend(context.device, guards)?;
        }
        let current = voltage.multiply(&conductance);
        context.local_law(coordinates.output, &current, &controls, 1., None)?;
        // The sensing branch and physical P-switch input conductance are
        // linear; reuse the exact same row incidence as scalar assembly.
        let zero_output = Coordinates {
            output: [None; 2],
            ..coordinates
        };
        zero_output.entries(0., 0., self.material.control_g, |row, column, value| {
            context.linear_static(Some(row), Some(column), value)
        })
    }

    fn assemble_nonlinear_charge(&self, _context: &mut ChargeContext<'_>) -> SpiceResult<()> {
        // No physical storage; the current-sensing branch is algebraic.
        Ok(())
    }

    fn noise(
        &self,
        _context: &crate::devices::noise::NoiseContext<'_>,
        _sources: &mut Vec<crate::devices::noise::NoiseSource>,
    ) -> SpiceResult<crate::devices::noise::DeviceNoise<'_>> {
        // XSPICE switch models declare no noise generator.
        Ok(crate::devices::noise::DeviceNoise::Noiseless)
    }
}

fn pair(terminal: &CodeTerminal, nodes: &mut NodeAppend<'_>) -> SpiceResult<[NodeId; 2]> {
    Ok([
        nodes.intern(&terminal.positive)?,
        nodes.intern(terminal.negative.as_deref().unwrap_or("0"))?,
    ])
}

pub(super) fn instantiate(
    schema: SourceModel,
    instance: &DeviceInstance,
    nodes: &mut NodeAppend<'_>,
    model: &ResolvedModel<'_>,
) -> SpiceResult<Box<dyn Device>> {
    let material = model.prepare(PreparationKey::Inputs, || {
        Material::compile(schema, model.card())
    })?;
    let [input, output] = instance.code_ports.as_slice() else {
        return Err(SpiceError::circuit("continuous switch port count mismatch"));
    };
    let input = &input.terminals[0];
    let control = match input.kind {
        CodePortKind::Voltage
        | CodePortKind::VoltageDifferential
        | CodePortKind::Conductance
        | CodePortKind::ConductanceDifferential => Control::Voltage,
        CodePortKind::Current | CodePortKind::CurrentDifferential => Control::Current,
        CodePortKind::BranchCurrent => Control::Named(ControlReference {
            name: input.positive.clone(),
            location: Some(instance.location.clone()),
        }),
        _ => return Err(SpiceError::circuit("invalid continuous switch control")),
    };
    nodes.transaction(|nodes| {
        let [p, n] = pair(&output.terminals[0], nodes)?;
        let [cp, cn] = if matches!(control, Control::Named(_)) {
            [NodeId::GROUND; 2]
        } else {
            pair(input, nodes)?
        };
        if matches!(control, Control::Current) && cp == cn {
            return Err(SpiceError::parse(
                instance.location.clone(),
                "shorted ideal switch current input",
            ));
        }
        Ok(Box::new(ContinuousSwitch {
            name: instance.name.clone(),
            model: model.card().name.clone(),
            material: *material,
            control,
            terminals: [p, n, cp, cn],
        }) as Box<dyn Device>)
    })
}
