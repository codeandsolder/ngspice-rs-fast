//! Typed voltage relations and ordered, path-local consistency checks.
use super::{RowHint, failure};
use crate::devices::Circuit;
use crate::maths::Vector;
use crate::primitives::{NodeId, Real, SpiceResult};
use petgraph::graph::{EdgeIndex, NodeIndex, UnGraph};
use petgraph::unionfind::UnionFind;
use petgraph::visit::EdgeRef;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(super) struct AbsoluteRow(pub(super) usize);

/// Ground never occupies an MNA row.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum VoltageVertex {
    Ground,
    Node(AbsoluteRow),
}

impl VoltageVertex {
    pub(super) fn bind(circuit: &Circuit, node: NodeId) -> SpiceResult<Self> {
        Ok(circuit
            .unknowns()
            .checked_node_row(node)?
            .map_or(Self::Ground, |row| Self::Node(AbsoluteRow(row))))
    }

    const fn index(self) -> usize {
        match self {
            Self::Ground => 0,
            Self::Node(AbsoluteRow(row)) => row + 1,
        }
    }

    fn graph_node(self) -> NodeIndex<usize> {
        NodeIndex::new(self.index())
    }
}

#[derive(Clone, Copy)]
pub(super) enum Origin<'a> {
    Device(&'a str),
    Hint(&'a RowHint),
}

impl std::fmt::Display for Origin<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Device(name) => f.write_str(name),
            Self::Hint(hint) => write!(f, ".ic V({}) at {}", hint.node, hint.location),
        }
    }
}

#[derive(Clone, Copy)]
enum Voltage {
    Known(Real),
    Controlled,
}

#[derive(Clone, Copy)]
struct Relation<'a> {
    positive: VoltageVertex,
    voltage: Voltage,
    origin: Origin<'a>,
}

pub(super) enum Join {
    Tree,
    Cycle { expected: Real },
}

type RelationGraph<'a> = UnGraph<(), Relation<'a>, usize>;

#[derive(Clone, Copy, Default)]
struct ForestNode {
    parent: Option<(NodeIndex<usize>, EdgeIndex<usize>)>,
    depth: usize,
}

/// Root the numeric forest without storing weighted potentials. Joining
/// components re-roots only the smaller component (amortized O(n log n)).
/// A redundant relation then follows exactly its unique path, not an entire
/// connected component. Arithmetic is performed later in path order.
struct KnownForest {
    components: UnionFind<usize>,
    sizes: Vec<usize>,
    nodes: Vec<ForestNode>,
    pending: Vec<NodeIndex<usize>>,
    reverse: Vec<EdgeIndex<usize>>,
}

impl KnownForest {
    fn new(vertices: usize) -> Self {
        Self {
            components: UnionFind::new(vertices),
            sizes: vec![1; vertices],
            nodes: vec![ForestNode::default(); vertices],
            pending: Vec::new(),
            reverse: Vec::new(),
        }
    }

    fn equiv(&self, a: VoltageVertex, b: VoltageVertex) -> bool {
        self.components.equiv(a.index(), b.index())
    }

    fn connect(
        &mut self,
        graph: &RelationGraph<'_>,
        mut child: NodeIndex<usize>,
        mut parent: NodeIndex<usize>,
        edge: EdgeIndex<usize>,
    ) {
        let mut small = self.components.find(child.index());
        let mut large = self.components.find(parent.index());
        if self.sizes[small] > self.sizes[large] {
            std::mem::swap(&mut small, &mut large);
            std::mem::swap(&mut child, &mut parent);
        }
        let size = self.sizes[small] + self.sizes[large];
        self.components.union(small, large);
        self.sizes[self.components.find(small)] = size;

        self.nodes[child.index()] = ForestNode {
            parent: Some((parent, edge)),
            depth: self.nodes[parent.index()].depth + 1,
        };
        self.pending.push(child);
        while let Some(node) = self.pending.pop() {
            let entry = self.nodes[node.index()];
            for link in graph.edges(node) {
                let next = link.target();
                if !matches!(link.weight().voltage, Voltage::Known(_))
                    || entry.parent.is_some_and(|(parent, _)| parent == next)
                {
                    continue;
                }
                // Known edges are acyclic. The new parent is the only edge
                // leading outside the smaller component.
                self.nodes[next.index()] = ForestNode {
                    parent: Some((node, link.id())),
                    depth: entry.depth + 1,
                };
                self.pending.push(next);
            }
        }
    }

    fn ascend(&self, node: NodeIndex<usize>) -> SpiceResult<(NodeIndex<usize>, EdgeIndex<usize>)> {
        self.nodes[node.index()]
            .parent
            .ok_or_else(|| failure("constraint forest path reached an unrelated root"))
    }

    fn path(
        &mut self,
        mut a: NodeIndex<usize>,
        mut b: NodeIndex<usize>,
        output: &mut Vec<EdgeIndex<usize>>,
    ) -> SpiceResult<()> {
        output.clear();
        self.reverse.clear();
        while a != b {
            let left_depth = self.nodes[a.index()].depth;
            let right_depth = self.nodes[b.index()].depth;
            if left_depth >= right_depth {
                let (parent, edge) = self.ascend(a)?;
                output.push(edge);
                a = parent;
            }
            if right_depth >= left_depth {
                let (parent, edge) = self.ascend(b)?;
                self.reverse.push(edge);
                b = parent;
            }
        }
        output.extend(self.reverse.iter().rev().copied());
        Ok(())
    }
}

#[derive(Clone, Copy)]
struct Visit {
    node: NodeIndex<usize>,
    parent: Option<(usize, EdgeIndex<usize>)>,
}

/// A single graph records both numeric and controlled voltage constraints.
/// Known edges form a spanning forest; controlled edges add structural
/// rigidity but never contribute a guessed numeric voltage. Component
/// indexes answer connectivity queries without a traversal. Numeric cycle
/// checks use rooted-forest links; controlled paths use a targeted graph walk
/// with reused scratch, clearing only previously visited vertices.
pub(super) struct ConstraintGraph<'a> {
    graph: RelationGraph<'a>,
    known: KnownForest,
    rigid: UnionFind<usize>,
    visited: Vec<bool>,
    visits: Vec<Visit>,
    pending: Vec<usize>,
    path: Vec<EdgeIndex<usize>>,
}

impl<'a> ConstraintGraph<'a> {
    fn new(unknowns: usize) -> SpiceResult<Self> {
        let vertices = unknowns
            .checked_add(1)
            .ok_or_else(|| failure("constraint vertex count overflow"))?;
        let mut graph = UnGraph::default();
        for _ in 0..vertices {
            graph.add_node(());
        }
        Ok(Self {
            graph,
            known: KnownForest::new(vertices),
            rigid: UnionFind::new(vertices),
            visited: vec![false; vertices],
            visits: Vec::new(),
            pending: Vec::new(),
            path: Vec::new(),
        })
    }

    pub(super) fn for_circuit(
        circuit: &'a Circuit,
        rhs: &Vector,
        inductors_short: bool,
    ) -> SpiceResult<Self> {
        let mut graph = Self::new(circuit.unknown_count())?;
        if rhs.as_slice().len() != circuit.unknown_count() {
            return Err(failure(
                "constraint source vector does not match finalized circuit",
            ));
        }
        for device in circuit.devices() {
            let analog = (device.as_ref() as &dyn std::any::Any)
                .downcast_ref::<crate::devices::code_model::analog::AnalogCodeModel>();
            if (matches!(device.designator(), 'e' | 'h')
                || analog.is_some_and(|model| model.initial_constraint().is_none()))
                && let [positive, negative, ..] = device.terminals()
            {
                graph.add_controlled(
                    VoltageVertex::bind(circuit, *positive)?,
                    VoltageVertex::bind(circuit, *negative)?,
                    Origin::Device(device.name()),
                );
            }
        }
        // Independent ideal sources first, in original device order.
        // Source loops are left for the numerical solver, as before.
        for (index, device) in circuit.devices().iter().enumerate() {
            let Some(rows) = circuit.branch_rows(index).filter(|rows| rows.len() == 1) else {
                continue;
            };
            let value = match device.designator() {
                'v' => rhs.as_slice()[rows.start],
                'l' if inductors_short => 0.,
                'a' => {
                    let instance = device.as_ref() as &dyn std::any::Any;
                    if instance.is::<crate::devices::code_model::DacSource>() {
                        rhs.as_slice()[rows.start]
                    } else if let Some(value) = instance
                        .downcast_ref::<crate::devices::code_model::analog::AnalogCodeModel>()
                        .and_then(
                            crate::devices::code_model::analog::AnalogCodeModel::initial_constraint,
                        )
                    {
                        value
                    } else {
                        continue;
                    }
                }
                _ => continue,
            };
            // An RF port source is on the internal #res node.
            let (positive, negative) = match (device.terminals(), device.rf_port()) {
                ([_, negative, _], Some(port)) => (port.internal, *negative),
                ([positive, negative], None) => (*positive, *negative),
                ([positive, negative, ..], None) if device.designator() == 'a' => {
                    (*positive, *negative)
                }
                _ => continue,
            };
            graph.join(
                VoltageVertex::bind(circuit, positive)?,
                VoltageVertex::bind(circuit, negative)?,
                value,
                Origin::Device(device.name()),
            )?;
        }
        Ok(graph)
    }

    fn add_controlled(
        &mut self,
        positive: VoltageVertex,
        negative: VoltageVertex,
        origin: Origin<'a>,
    ) {
        self.rigid.union(positive.index(), negative.index());
        self.graph.add_edge(
            positive.graph_node(),
            negative.graph_node(),
            Relation {
                positive,
                voltage: Voltage::Controlled,
                origin,
            },
        );
    }

    pub(super) fn controlled_implies(&self, a: VoltageVertex, b: VoltageVertex) -> bool {
        !self.known.equiv(a, b) && self.rigid.equiv(a.index(), b.index())
    }

    pub(super) fn join(
        &mut self,
        positive: VoltageVertex,
        negative: VoltageVertex,
        value: Real,
        origin: Origin<'a>,
    ) -> SpiceResult<Join> {
        if !value.is_finite() {
            return Err(failure(format!("nonfinite voltage relation at {origin}")));
        }
        if self.known.equiv(positive, negative) {
            // The old traversal started at the positive terminal with +0,
            // followed the unique forest path and negated the final result.
            let expected = self.difference(positive, negative)?;
            if !expected.is_finite() {
                return Err(failure(format!(
                    "nonfinite voltage cycle at {origin} (path: {})",
                    self.path_description()
                )));
            }
            return Ok(Join::Cycle { expected });
        }
        self.rigid.union(positive.index(), negative.index());
        let edge = self.graph.add_edge(
            positive.graph_node(),
            negative.graph_node(),
            Relation {
                positive,
                voltage: Voltage::Known(value),
                origin,
            },
        );
        self.known.connect(
            &self.graph,
            positive.graph_node(),
            negative.graph_node(),
            edge,
        );
        Ok(Join::Tree)
    }

    /// Preserve the old full DFS's exact path-order arithmetic without
    /// visiting its unrelated branches or reassociating any sum.
    fn difference(&mut self, start: VoltageVertex, target: VoltageVertex) -> SpiceResult<Real> {
        self.known
            .path(start.graph_node(), target.graph_node(), &mut self.path)?;
        let mut at = start.graph_node();
        let mut potential = 0.;
        for &edge in &self.path {
            let relation = &self.graph[edge];
            let Voltage::Known(value) = relation.voltage else {
                return Err(failure("non-numeric edge in known voltage path"));
            };
            let (first, second) = self
                .graph
                .edge_endpoints(edge)
                .ok_or_else(|| failure("missing voltage path edge"))?;
            if at != first && at != second {
                return Err(failure("disconnected voltage path edge"));
            }
            potential = if at == relation.positive.graph_node() {
                potential - value
            } else {
                potential + value
            };
            at = if at == first { second } else { first };
        }
        Ok(-potential)
    }

    /// Controlled relations do not have a known numeric offset. Traverse the
    /// rigid graph only to identify the relevant constraint provenance.
    pub(super) fn controlled_path(
        &mut self,
        start: VoltageVertex,
        target: VoltageVertex,
    ) -> SpiceResult<String> {
        for visit in self.visits.drain(..) {
            self.visited[visit.node.index()] = false;
        }
        self.pending.clear();
        self.path.clear();
        self.visits.push(Visit {
            node: start.graph_node(),
            parent: None,
        });
        self.visited[start.index()] = true;
        self.pending.push(0);
        while let Some(index) = self.pending.pop() {
            let visit = self.visits[index];
            if visit.node == target.graph_node() {
                let mut at = index;
                while let Some((parent, edge)) = self.visits[at].parent {
                    self.path.push(edge);
                    at = parent;
                }
                self.path.reverse();
                return Ok(self.path_description());
            }
            for edge in self.graph.edges(visit.node) {
                let neighbor = edge.target();
                if self.visited[neighbor.index()] {
                    continue;
                }
                let next = self.visits.len();
                self.visits.push(Visit {
                    node: neighbor,
                    parent: Some((index, edge.id())),
                });
                self.visited[neighbor.index()] = true;
                self.pending.push(next);
                if neighbor == target.graph_node() {
                    break;
                }
            }
        }
        Err(failure("constraint component has no connecting path"))
    }

    pub(super) fn path_description(&self) -> String {
        if self.path.is_empty() {
            "same node".to_owned()
        } else {
            self.path
                .iter()
                .map(|&edge| self.graph[edge].origin.to_string())
                .collect::<Vec<_>>()
                .join(" -> ")
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use petgraph::visit::{Control, DfsEvent, depth_first_search};

    const fn vertex(row: usize) -> VoltageVertex {
        VoltageVertex::Node(AbsoluteRow(row))
    }

    fn known(
        graph: &mut ConstraintGraph<'_>,
        a: VoltageVertex,
        b: VoltageVertex,
        value: Real,
        name: &'static str,
    ) {
        assert!(matches!(
            graph.join(a, b, value, Origin::Device(name)).unwrap(),
            Join::Tree
        ));
    }

    fn cycle(graph: &mut ConstraintGraph<'_>, a: VoltageVertex, b: VoltageVertex) -> Real {
        match graph.join(a, b, 0., Origin::Device("redundant")).unwrap() {
            Join::Cycle { expected } => expected,
            Join::Tree => panic!("expected an existing relation"),
        }
    }

    #[test]
    fn traversal_order_preserves_reference_rounding_in_both_directions() {
        let mut graph = ConstraintGraph::new(4).unwrap();
        let [a, b, c] = [vertex(0), vertex(1), vertex(2)];
        known(&mut graph, a, b, 1e16, "v1");
        known(&mut graph, b, c, -1e16, "v2");
        known(&mut graph, c, VoltageVertex::Ground, 1., "v3");
        assert_eq!(
            cycle(&mut graph, a, VoltageVertex::Ground).to_bits(),
            1_f64.to_bits()
        );
        assert_eq!(graph.path_description(), "v1 -> v2 -> v3");
        assert_eq!(
            cycle(&mut graph, VoltageVertex::Ground, a).to_bits(),
            (-0_f64).to_bits()
        );
        assert_eq!(graph.path_description(), "v3 -> v2 -> v1");
        assert_eq!(
            cycle(&mut graph, VoltageVertex::Ground, VoltageVertex::Ground).to_bits(),
            (-0_f64).to_bits()
        );
        assert_eq!(graph.path_description(), "same node");
        assert!(!graph.controlled_implies(vertex(3), a));
    }

    /// Independent reference: the previous full DFS initialized every
    /// vertex's potential and traversed the entire known forest component.
    fn reference(graph: &ConstraintGraph<'_>, a: VoltageVertex, b: VoltageVertex) -> Real {
        let mut potential = vec![0.; graph.graph.node_count()];
        depth_first_search(&graph.graph, [a.graph_node()], |event| {
            if let DfsEvent::TreeEdge(from, to) = event {
                let relation = &graph.graph[graph.graph.find_edge(from, to).unwrap()];
                let Voltage::Known(value) = relation.voltage else {
                    panic!("known forest only");
                };
                potential[to.index()] = if relation.positive.graph_node() == from {
                    potential[from.index()] - value
                } else {
                    potential[from.index()] + value
                };
            }
            Control::<()>::Continue
        });
        -potential[b.index()]
    }

    #[test]
    fn path_local_walk_matches_old_full_traversal_bit_for_bit() {
        let size = 63;
        for reversed in [false, true] {
            let mut graph = ConstraintGraph::new(size).unwrap();
            for step in 0..size {
                let row = if reversed { size - step - 1 } else { step };
                let parent = if row == 0 {
                    VoltageVertex::Ground
                } else {
                    vertex((row - 1) / 2)
                };
                let value = [1e16, -1e16, 1., -1., 1e-10, -1e-10][row % 6];
                if row % 2 == 0 {
                    known(&mut graph, vertex(row), parent, value, "edge");
                } else {
                    known(&mut graph, parent, vertex(row), value, "edge");
                }
            }
            let vertices: Vec<_> = std::iter::once(VoltageVertex::Ground)
                .chain((0..size).map(vertex))
                .collect();
            for &a in &vertices {
                for &b in &vertices {
                    let expected = reference(&graph, a, b);
                    assert_eq!(
                        cycle(&mut graph, a, b).to_bits(),
                        expected.to_bits(),
                        "{a:?} to {b:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn controlled_rigidity_reports_only_the_path_between_requested_vertices() {
        let mut graph = ConstraintGraph::new(4).unwrap();
        let [a, b, c, d] = [vertex(0), vertex(1), vertex(2), vertex(3)];
        known(&mut graph, a, VoltageVertex::Ground, 2., "v1");
        graph.add_controlled(b, a, Origin::Device("e1"));
        graph.add_controlled(d, c, Origin::Device("e2"));
        assert!(graph.controlled_implies(b, VoltageVertex::Ground));
        assert_eq!(
            graph.controlled_path(b, VoltageVertex::Ground).unwrap(),
            "e1 -> v1"
        );
        assert!(!graph.controlled_implies(b, d));
        assert!(!graph.controlled_implies(a, VoltageVertex::Ground));
    }

    #[test]
    #[ignore = "manual workload: large redundant voltage-constraint graph"]
    fn redundant_constraint_graph_benchmark() {
        use std::time::Instant;
        const N: usize = 10_000;
        let mut graph = ConstraintGraph::new(N).unwrap();
        for row in 0..N {
            let parent = if row == 0 {
                VoltageVertex::Ground
            } else {
                vertex((row - 1) / 2)
            };
            known(&mut graph, vertex(row), parent, 1., "edge");
        }
        // Every constraint is redundant with one adjacent forest edge.
        // The previous DFS visited N+1 vertices for each relation.
        let started = Instant::now();
        let mut visited = 0;
        for row in 1..N {
            assert_eq!(cycle(&mut graph, vertex(row), vertex((row - 1) / 2)), 1.);
            visited += graph.path.len() + 1;
        }
        let local_time = started.elapsed();
        let started = Instant::now();
        for row in 1..N {
            assert_eq!(reference(&graph, vertex(row), vertex((row - 1) / 2)), 1.);
        }
        let full_time = started.elapsed();
        println!(
            "{} redundant constraints: {visited} path visits vs {} full DFS vertices; {local_time:?} vs {full_time:?}",
            N - 1,
            (N - 1) * (N + 1)
        );
        assert_eq!(visited, 2 * (N - 1));
    }
}
