//! Prepared piecewise-linear two-terminal resistors. A segment hint is an
//! advisory first Newton tangent, not an electrical initial condition.
use model_normalize_ngspice::pwl_resistor::{Axis, NativeBinding, PreparedPwlResistor};
use std::{collections::BTreeMap, sync::Arc};

use crate::devices::{
    ConvergenceContext, Device, IterationPhase, LinearContext, MnaUnknowns, StampContext,
};
use crate::maths::{SparseMatrix, Vector};
use crate::netlist::ast::{DeviceInstance, Netlist, ParameterKind};
use crate::primitives::{NodeAppend, NodeId, NodeTable, Real, SpiceError, SpiceResult};

/// The physical line `y = slope*x + intercept`. The upper coordinate is
/// exclusive except on the final, extrapolating segment.
#[derive(Debug, PartialEq)]
struct Segment {
    upper: Real,
    slope: Real,
    intercept: Real,
}

#[derive(Debug, PartialEq)]
struct Curve {
    axis: Axis,
    segments: Vec<Segment>,
}

impl Curve {
    fn prepare(declaration: &PreparedPwlResistor) -> SpiceResult<Self> {
        declaration
            .validate()
            .map_err(|error| SpiceError::circuit(error.to_string()))?;
        let axis = declaration.axis;
        let points = declaration
            .knots
            .iter()
            .map(|knot| match axis {
                Axis::Voltage => (knot.voltage, knot.current),
                Axis::Current => (knot.current, knot.voltage),
            })
            .collect::<Vec<_>>();
        let count = points.len() - 1;
        let mut segments = Vec::with_capacity(count);
        for (index, window) in points.windows(2).enumerate() {
            let [(x0, y0), (x1, y1)] = *window else {
                return Err(SpiceError::circuit("missing prepared PWL endpoint"));
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
                return Err(SpiceError::circuit("unrepresentable prepared PWL segment"));
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

/// Cold, immutable law material shared by each occurrence of a declaration.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct BoundPwlResistor {
    curve: Arc<Curve>,
    initial: Option<usize>,
}

type Identity = (String, String, String);
pub(super) type BoundResistors = BTreeMap<String, BoundPwlResistor>;

/// Declaration index: artifact, lexical definition, emitted law anchor.
#[derive(Default)]
pub(super) struct Index(BTreeMap<Identity, BoundPwlResistor>);

fn identity(device: &DeviceInstance, definition: &str) -> Option<Identity> {
    Some((
        device.location.path().file_name()?.to_str()?.to_owned(),
        definition.to_ascii_lowercase(),
        device.name.to_ascii_lowercase(),
    ))
}

fn matches_nodes(card: &DeviceInstance, p: &str, n: &str, auto_gnd: bool) -> bool {
    card.nodes.as_slice()
        == [
            NodeTable::canonical_name(p, auto_gnd),
            NodeTable::canonical_name(n, auto_gnd),
        ]
}

fn validate_anchor(
    cards: &BTreeMap<Identity, Option<&DeviceInstance>>,
    key: Identity,
    positive: &str,
    negative: &str,
    output: Option<&str>,
    auto_gnd: bool,
) -> SpiceResult<()> {
    let card = cards.get(&key).and_then(|card| *card).ok_or_else(|| {
        SpiceError::circuit(format!(
            "missing or ambiguous prepared resistor anchor {}",
            key.2
        ))
    })?;
    let correct = if let Some(output) = output {
        card.designator == 'b'
            && matches!(card.parameters.as_slice(), [parameter]
            if parameter.name == output && matches!(parameter.kind, ParameterKind::Behavioural(_)))
    } else {
        card.designator == 'v'
            && matches!(card.parameters.as_slice(), [parameter]
            if parameter.name == "dc" && crate::primitives::parse_spice_number(&parameter.value) == Some(0.))
    };
    if !correct || !matches_nodes(card, positive, negative, auto_gnd) {
        return Err(SpiceError::parse(
            card.location.clone(),
            "incorrect prepared resistor anchor shape or orientation",
        ));
    }
    Ok(())
}

impl Index {
    pub(super) fn new(
        netlist: &Netlist,
        declarations: &[PreparedPwlResistor],
    ) -> SpiceResult<Self> {
        if declarations.is_empty() {
            return Ok(Self::default());
        }
        let mut cards = BTreeMap::new();
        for (definition, devices) in std::iter::once(("", netlist.devices.as_slice())).chain(
            netlist
                .subcircuits
                .iter()
                .map(|definition| (definition.name.as_str(), definition.devices.as_slice())),
        ) {
            for card in devices {
                if let Some(key) = identity(card, definition) {
                    cards
                        .entry(key)
                        .and_modify(|entry| *entry = None)
                        .or_insert(Some(card));
                }
            }
        }
        let mut index = BTreeMap::new();
        let mut seen = std::collections::BTreeSet::new();
        for declaration in declarations {
            if declaration.scope.len() > 1 {
                return Err(SpiceError::circuit(
                    "nested prepared resistor definition is unsupported",
                ));
            }
            let definition = declaration
                .scope
                .first()
                .map_or_else(String::new, |name| name.to_ascii_lowercase());
            let key = |name: &str| {
                (
                    declaration.artifact.clone(),
                    definition.clone(),
                    name.to_ascii_lowercase(),
                )
            };
            let curve = Arc::new(Curve::prepare(declaration)?);
            if !seen.insert(key(&declaration.device)) {
                return Err(SpiceError::circuit(
                    "duplicate prepared resistor declaration",
                ));
            }
            let mut positive = declaration.positive.as_str();
            let anchor = |name: &str, p: &str, n: &str, output: Option<&str>| {
                let key = key(name);
                validate_anchor(&cards, key, p, n, output, netlist.auto_gnd)
            };
            if let Some(sensor) = &declaration.series_sensor {
                anchor(&sensor.source, positive, &sensor.intermediate, None)?;
                positive = &sensor.intermediate;
            }
            let source = match &declaration.binding {
                NativeBinding::CurrentLaw { source } => {
                    anchor(source, positive, &declaration.negative, Some("i"))?;
                    source
                }
                NativeBinding::VoltageLaw {
                    sensor,
                    source,
                    intermediate,
                } => {
                    anchor(sensor, positive, intermediate, None)?;
                    anchor(source, intermediate, &declaration.negative, Some("v"))?;
                    source
                }
            };
            if index
                .insert(
                    key(source),
                    BoundPwlResistor {
                        curve,
                        initial: declaration.initial_segment.map(|segment| segment - 1),
                    },
                )
                .is_some()
            {
                return Err(SpiceError::circuit(
                    "duplicate prepared resistor law anchor",
                ));
            }
        }
        Ok(Self(index))
    }

    pub(super) fn get(
        &self,
        device: &DeviceInstance,
        definition: &str,
    ) -> Option<BoundPwlResistor> {
        self.0.get(&identity(device, definition)?).cloned()
    }
}

#[derive(Debug)]
struct PwlResistor {
    name: String,
    terminals: [NodeId; 2],
    curve: Arc<Curve>,
    initial: Option<usize>,
}

impl BoundPwlResistor {
    pub(super) fn bind(
        &self,
        instance: &DeviceInstance,
        nodes: &mut NodeAppend<'_>,
    ) -> SpiceResult<Box<dyn Device>> {
        let [p, n] = instance.nodes.as_slice() else {
            return Err(SpiceError::parse(
                instance.location.clone(),
                "prepared PWL resistor requires two terminals",
            ));
        };
        Ok(Box::new(PwlResistor {
            name: instance.name.clone(),
            terminals: [nodes.intern(p)?, nodes.intern(n)?],
            curve: Arc::clone(&self.curve),
            initial: self.initial,
        }))
    }
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
        'b'
    }
    fn terminals(&self) -> &[NodeId] {
        &self.terminals
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
