//! Bind a validated passive graph to ordinary RLC devices. The ideal real
//! port transform is power-conserving in every analysis.
use crate::analysis::rf_network::{
    invalid,
    realization::{Element, PassiveRealization},
};
use crate::devices::circuit::Circuit;
use crate::devices::elaboration::{ElaborationSize, reserve};
use crate::devices::linear::LinearContext;
use crate::devices::nport::PortTerminals;
use crate::devices::rlc::{Capacitor, Inductor, REACTIVE_STATE_COUNT, Resistor};
use crate::devices::traits::{Device, StampContext};
use crate::devices::{MnaUnknowns, setup::MatrixSetup};
use crate::maths::{SparseMatrix, Vector};
use crate::primitives::{NodeAppend, NodeId, NodeKind, SpiceResult};
use std::sync::Arc;

#[derive(Debug)]
struct PortTransform {
    name: String,
    graph: Arc<PassiveRealization>,
    ports: Vec<PortTerminals>,
    internal: Vec<NodeId>,
    terminals: Vec<NodeId>,
}

impl PortTransform {
    fn entries(
        &self,
        mut add: impl FnMut(usize, NodeId, f64) -> SpiceResult<()>,
    ) -> SpiceResult<()> {
        for (row, (&internal, basis)) in self
            .internal
            .iter()
            .zip(self.graph.basis.chunks_exact(self.graph.ports))
            .enumerate()
        {
            add(row, internal, 1.)?;
            for (&coefficient, port) in basis.iter().zip(&self.ports) {
                if coefficient != 0. {
                    add(row, port.positive, -coefficient)?;
                    add(row, port.negative, coefficient)?;
                }
            }
        }
        Ok(())
    }

    fn stamp_into(
        &self,
        matrix: &mut SparseMatrix,
        unknowns: &MnaUnknowns,
        branch: usize,
    ) -> SpiceResult<()> {
        self.entries(|row, node, value| {
            if let Some(col) = unknowns.node_row(node) {
                matrix.add(branch + row, col, value)?;
                matrix.add(col, branch + row, value)?;
            }
            Ok(())
        })
    }

    fn assemble(&self, ctx: &mut LinearContext<'_>, frequency: f64) -> SpiceResult<()> {
        self.graph
            .validate_context(frequency, ctx.model_context.temperature)?;
        self.stamp_into(
            &mut ctx.system.a,
            ctx.unknowns,
            ctx.branch
                .ok_or_else(|| invalid("missing realization port branches"))?,
        )
    }
}

impl Device for PortTransform {
    fn name(&self) -> &str {
        &self.name
    }
    fn designator(&self) -> char {
        'n'
    }
    fn terminals(&self) -> &[NodeId] {
        &self.terminals
    }
    fn branch_currents(&self) -> usize {
        self.graph.ports
    }
    fn small_signal_depends_on_frequency(&self) -> bool {
        true
    }

    fn setup_matrix(&self, setup: &mut MatrixSetup<'_>) -> SpiceResult<bool> {
        self.entries(|row, node, _| {
            let branch = Some(setup.branch(row)?);
            let node = setup.node(node);
            setup.element(branch, node);
            setup.element(node, branch);
            Ok(())
        })?;
        Ok(true)
    }

    fn stamp(&self, ctx: &mut StampContext<'_>) -> SpiceResult<()> {
        self.graph.validate_context(0., ctx.temperature)?;
        self.stamp_into(ctx.matrix, ctx.unknowns, ctx.branch(0)?)
    }

    fn assemble_linear(&self, ctx: &mut LinearContext<'_>) -> SpiceResult<()> {
        self.assemble(ctx, 0.)
    }

    fn assemble_small_signal(
        &self,
        ctx: &mut LinearContext<'_>,
        _bias: &Vector,
    ) -> SpiceResult<()> {
        self.assemble(ctx, ctx.model_context.frequency)
    }

    fn assemble_pole_zero(&self, ctx: &mut LinearContext<'_>, _bias: &Vector) -> SpiceResult<()> {
        self.assemble(ctx, 0.)
    }

    fn noise(
        &self,
        _ctx: &crate::devices::noise::NoiseContext<'_>,
        _sources: &mut Vec<crate::devices::noise::NoiseSource>,
    ) -> SpiceResult<crate::devices::noise::DeviceNoise<'_>> {
        Ok(crate::devices::noise::DeviceNoise::Noiseless)
    }

    fn distortion(
        &self,
        _ctx: &crate::devices::distortion::DistortionContext<'_>,
    ) -> SpiceResult<crate::devices::distortion::DeviceDistortion> {
        Ok(crate::devices::distortion::DeviceDistortion::Linear)
    }
}

fn build(
    name: &str,
    graph: Arc<PassiveRealization>,
    ports: Vec<PortTerminals>,
    nodes: &mut NodeAppend<'_>,
) -> SpiceResult<Vec<Box<dyn Device>>> {
    if ports
        .iter()
        .any(|p| nodes.node(p.positive).is_none() || nodes.node(p.negative).is_none())
    {
        return Err(invalid("realization port refers to an unknown node"));
    }
    let mut ids = Vec::new();
    reserve(&mut ids, graph.nodes, None)?;
    ids.push(NodeId::GROUND);
    for i in 1..graph.nodes {
        let node_name = format!("{name}#rf{i}");
        if nodes.get(&node_name).is_some() {
            return Err(invalid("realization internal node name collision"));
        }
        let id = nodes.intern(&node_name)?;
        nodes.set_kind(id, NodeKind::Internal)?;
        ids.push(id);
    }
    let internal = ids[1..=graph.ports].to_vec();
    let terminals = ports
        .iter()
        .flat_map(|p| [p.positive, p.negative])
        .chain(internal.iter().copied())
        .collect();
    let mut devices: Vec<Box<dyn Device>> = Vec::new();
    reserve(&mut devices, graph.edges.len() + 1, None)?;
    for (i, edge) in graph.edges.iter().enumerate() {
        let terminals = edge.nodes.map(|id| ids[id]);
        let element_name = format!("{name}#element{i}");
        let device: Box<dyn Device> = match edge.element {
            Element::Resistance(r) => Box::new(Resistor::new(element_name, terminals, r)?),
            Element::Inductance(l) => Box::new(Inductor::new(element_name, terminals, l, None)?),
            Element::Capacitance(c) => Box::new(Capacitor::new(element_name, terminals, c, None)?),
        };
        devices.push(device);
    }
    let transform = PortTransform {
        name: name.to_owned(),
        graph,
        ports,
        internal,
        terminals,
    };
    devices.insert(0, Box::new(transform));
    Ok(devices)
}

/// Atomically bind an immutable positive-element RF realization.
///
/// The simulator uses its native RLC state integration and the same circuit
/// topology for DC, AC, SP and transient. No runtime fitting occurs.
/// # Errors
/// Invalid ports, naming collisions, invalid elements or exhausted resources.
pub fn bind_realization(
    circuit: &mut Circuit,
    name: &str,
    graph: Arc<PassiveRealization>,
    ports: Vec<PortTerminals>,
) -> SpiceResult<()> {
    if name.is_empty()
        || ports.len() != graph.ports
        || ports.iter().any(|p| p.positive == p.negative)
    {
        return Err(invalid("invalid realization terminals"));
    }
    let reactive = graph
        .edges
        .iter()
        .filter(|e| !matches!(e.element, Element::Resistance(_)))
        .count();
    let size = ElaborationSize::new(
        graph.edges.len() + 1,
        graph.nodes - 1,
        reactive * REACTIVE_STATE_COUNT,
    );
    circuit.add_prepared_devices(size, |nodes| build(name, graph, ports, nodes))
}
