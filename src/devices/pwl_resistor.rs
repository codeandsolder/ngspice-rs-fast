//! Prepared piecewise-linear two-terminal resistors. A segment hint is an
//! advisory first Newton tangent, not an electrical initial condition.
use std::{collections::BTreeMap, sync::Arc};

use crate::devices::{
    ConvergenceContext, Device, IterationPhase, LinearContext, MnaUnknowns, StampContext,
    models::{ModelFamily, PreparationKey, ResolvedModel},
    schema::finite_literal,
};
use crate::maths::{SparseMatrix, Vector};
use crate::netlist::ast::{DeviceInstance, ModelCard, ParameterAssignment};
use crate::primitives::{
    NodeAppend, NodeId, Real, SpiceError, SpiceResult, exact_nonnegative_integer,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Axis {
    Voltage,
    Current,
}

/// The physical line `y = slope*x + intercept`. The upper coordinate is
/// exclusive except on the final, extrapolating segment.
#[derive(Debug)]
struct Segment {
    upper: Real,
    slope: Real,
    intercept: Real,
}

#[derive(Debug)]
struct Curve {
    axis: Axis,
    segments: Vec<Segment>,
}

fn integer(parameter: &ParameterAssignment) -> SpiceResult<usize> {
    exact_nonnegative_integer(finite_literal(parameter)?).ok_or_else(|| {
        SpiceError::parse(
            parameter.location.clone(),
            format!("{} requires an exact nonnegative integer", parameter.name),
        )
    })
}

impl Curve {
    fn prepare(card: &ModelCard, axis: Axis) -> SpiceResult<Self> {
        let error = |message| SpiceError::parse(card.location.clone(), message);
        let mut coordinates = BTreeMap::new();
        let mut count = None;
        for parameter in &card.parameters {
            match parameter.name.as_str() {
                "level" => continue,
                "nseg" => {
                    if count.replace(integer(parameter)?).is_some() {
                        return Err(error("duplicate prepared PWL NSEG"));
                    }
                }
                name => {
                    if coordinates
                        .insert(name, finite_literal(parameter)?)
                        .is_some()
                    {
                        return Err(error("duplicate prepared PWL coordinate"));
                    }
                }
            }
        }
        let count = count
            .filter(|n| (2..=255).contains(n))
            .ok_or_else(|| error("prepared PWL NSEG must be in 2..=255"))?;
        let mut points = Vec::with_capacity(count + 1);
        for i in 0..=count {
            let x = coordinates
                .remove(format!("x{i}").as_str())
                .ok_or_else(|| error("missing prepared PWL voltage knot"))?;
            let y = coordinates
                .remove(format!("y{i}").as_str())
                .ok_or_else(|| error("missing prepared PWL current knot"))?;
            points.push(match axis {
                Axis::Voltage => (x, y),
                Axis::Current => (y, x),
            });
        }
        if !coordinates.is_empty() {
            return Err(error("unknown prepared PWL model parameter"));
        }
        let mut segments = Vec::with_capacity(count);
        for (index, window) in points.windows(2).enumerate() {
            let [(x0, y0), (x1, y1)] = *window else {
                return Err(error("missing prepared PWL endpoint"));
            };
            let dx = x1 - x0;
            let slope = (y1 - y0) / dx;
            let intercept = (-slope).mul_add(x0, y0);
            if dx <= 0.
                || !dx.is_finite()
                || !slope.is_finite()
                || !intercept.is_finite()
                || ((index == 0 || index + 1 == count) && slope <= 0.)
            {
                return Err(error(
                    "invalid prepared PWL independent coordinates or end slopes",
                ));
            }
            segments.push(Segment {
                upper: x1,
                slope,
                intercept,
            });
        }
        Ok(Self { axis, segments })
    }

    fn segment(&self, x: Real) -> &Segment {
        // Binary search, with the final segment extrapolated to +infinity.
        let index = self.segments.partition_point(|segment| x >= segment.upper);
        &self.segments[index.min(self.segments.len() - 1)]
    }

    fn coordinates(&self, voltage: Real, current: Real) -> (Real, Real) {
        match self.axis {
            Axis::Voltage => (voltage, current),
            Axis::Current => (current, voltage),
        }
    }

    fn coefficients(&self, segment: &Segment) -> (Real, Real) {
        match self.axis {
            Axis::Voltage => (-segment.slope, 1.),
            Axis::Current => (1., -segment.slope),
        }
    }
}

#[derive(Debug)]
struct PwlResistor {
    name: String,
    model: String,
    terminals: [NodeId; 2],
    curve: Arc<Curve>,
    initial: Option<usize>,
}

pub(super) fn instantiate(
    instance: &DeviceInstance,
    nodes: &mut NodeAppend<'_>,
    model: &ResolvedModel<'_>,
) -> SpiceResult<Box<dyn Device>> {
    let axis = match model.family() {
        ModelFamily::PreparedVpwlr => Axis::Voltage,
        ModelFamily::PreparedIpwlr => Axis::Current,
        _ => {
            return Err(SpiceError::parse(
                instance.location.clone(),
                "wrong prepared PWL resistor family",
            ));
        }
    };
    let curve = model.prepare(PreparationKey::Inputs, || {
        Curve::prepare(model.card(), axis)
    })?;
    let mut initial = None;
    for parameter in &instance.parameters {
        if parameter.name != "initial_segment" || initial.is_some() {
            return Err(SpiceError::parse(
                parameter.location.clone(),
                "unsupported or duplicate prepared PWL resistor instance parameter",
            ));
        }
        initial = Some(
            integer(parameter)?
                .checked_sub(1)
                .filter(|i| *i < curve.segments.len())
                .ok_or_else(|| {
                    SpiceError::parse(
                        parameter.location.clone(),
                        "initial_segment must select a defined segment (one-based)",
                    )
                })?,
        );
    }
    let [p, n] = instance.nodes.as_slice() else {
        return Err(SpiceError::parse(
            instance.location.clone(),
            "prepared PWL resistor requires two terminals",
        ));
    };
    Ok(Box::new(PwlResistor {
        name: instance.name.clone(),
        model: model.card().name.clone(),
        terminals: [nodes.intern(p)?, nodes.intern(n)?],
        curve,
        initial,
    }))
}

impl PwlResistor {
    fn voltage(&self, unknowns: &MnaUnknowns, solution: &Vector) -> Real {
        let potential = |node| {
            unknowns
                .node_row(node)
                .and_then(|row| solution.get(row))
                .unwrap_or(0.)
        };
        potential(self.terminals[0]) - potential(self.terminals[1])
    }

    fn stamp_matrix(
        &self,
        matrix: &mut SparseMatrix,
        unknowns: &MnaUnknowns,
        branch: usize,
        segment: &Segment,
    ) -> SpiceResult<()> {
        let (voltage, current) = self.curve.coefficients(segment);
        for (node, sign) in [(self.terminals[0], 1.), (self.terminals[1], -1.)] {
            if let Some(row) = unknowns.node_row(node) {
                matrix.add(row, branch, sign)?;
                matrix.add(branch, row, sign * voltage)?;
            }
        }
        matrix.add(branch, branch, current)
    }
}

impl Device for PwlResistor {
    fn name(&self) -> &str {
        &self.name
    }
    fn designator(&self) -> char {
        'r'
    }
    fn terminals(&self) -> &[NodeId] {
        &self.terminals
    }
    fn model_name(&self) -> Option<&str> {
        Some(&self.model)
    }
    fn branch_currents(&self) -> usize {
        1
    }
    fn findable_branch(&self) -> Option<usize> {
        Some(0)
    }
    fn is_nonlinear(&self) -> bool {
        true
    }
    fn limits_voltage_steps(&self) -> bool {
        false
    }
    fn has_start_settings(&self) -> bool {
        self.initial.is_some()
    }
    // Record that a load has occurred even on a failed factorization retry.
    fn state_count(&self) -> usize {
        1
    }
    fn setup_matrix(
        &self,
        setup: &mut crate::devices::setup::MatrixSetup<'_>,
    ) -> SpiceResult<bool> {
        let branch = setup.branch(0)?;
        setup.inductor(self.terminals[0], self.terminals[1], branch);
        Ok(true)
    }
    fn stamp(&self, context: &mut StampContext<'_>) -> SpiceResult<()> {
        let row = context.branch(0)?;
        let voltage = self.voltage(context.unknowns, context.solution);
        let current = context.row_value(row)?;
        let (x, _) = self.curve.coordinates(voltage, current);
        let hint = if context.mode.is_bias()
            && context.states.device_limiting()
            && context.states.phase() == IterationPhase::Junction
            && context.states.iterate(0).is_none()
        {
            self.initial
        } else {
            None
        };
        let segment = hint.map_or_else(|| self.curve.segment(x), |i| &self.curve.segments[i]);
        context.states.set(0, x)?;
        self.stamp_matrix(context.matrix, context.unknowns, row, segment)?;
        context.rhs.add_to(row, segment.intercept)
    }
    fn convergence_test(&self, context: &ConvergenceContext<'_>) -> SpiceResult<bool> {
        let branch = context
            .branch
            .ok_or_else(|| SpiceError::circuit("missing prepared PWL resistor branch"))?;
        let voltage = self.voltage(context.unknowns, context.solution);
        let current = context
            .solution
            .get(branch)
            .ok_or_else(|| SpiceError::circuit("missing prepared PWL resistor current"))?;
        let (x, y) = self.curve.coordinates(voltage, current);
        let segment = self.curve.segment(x);
        let target = segment.slope.mul_add(x, segment.intercept);
        let absolute = match self.curve.axis {
            Axis::Voltage => context.abstol,
            Axis::Current => context.vntol,
        };
        Ok(!target.is_finite()
            || (target - y).abs() > context.reltol.mul_add(target.abs().max(y.abs()), absolute))
    }
    fn assemble_small_signal(
        &self,
        context: &mut LinearContext<'_>,
        bias: &Vector,
    ) -> SpiceResult<()> {
        let branch = context
            .branch
            .ok_or_else(|| SpiceError::circuit("missing prepared PWL resistor branch"))?;
        let voltage = self.voltage(context.unknowns, bias);
        let current = bias
            .get(branch)
            .ok_or_else(|| SpiceError::circuit("missing prepared PWL resistor bias current"))?;
        let (x, _) = self.curve.coordinates(voltage, current);
        self.stamp_matrix(
            &mut context.system.a,
            context.unknowns,
            branch,
            self.curve.segment(x),
        )
    }
    fn assemble_pole_zero(
        &self,
        context: &mut LinearContext<'_>,
        bias: &Vector,
    ) -> SpiceResult<()> {
        self.assemble_small_signal(context, bias)
    }
    fn assemble_nonlinear_charge(
        &self,
        _context: &mut crate::devices::traits::ChargeContext<'_>,
    ) -> SpiceResult<()> {
        Ok(())
    }
}
