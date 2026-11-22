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

#[derive(Clone, Copy)]
struct Visit {
    node: NodeIndex<usize>,
    parent: Option<(usize, EdgeIndex<usize>)>,
    potential: Real,
}

#[derive(Clone, Copy)]
enum PathKind {
    Known,
    Rigid,
}

/// A single graph records both numeric and controlled voltage constraints.
/// Known edges form a spanning forest; controlled edges add structural
/// rigidity but never contribute a guessed numeric voltage. Component
/// indexes answer connectivity queries without a traversal. Path scratch is
/// allocated once, cleared only for vertices reached by the previous walk.
pub(super) struct ConstraintGraph<'a> {
    graph: UnGraph<(), Relation<'a>, usize>,
    known: UnionFind<usize>,
    rigid: UnionFind<usize>,
    visited: Vec<bool>,
    visits: Vec<Visit>,
    pending: Vec<usize>,
    end: Option<usize>,
}

impl<'a> ConstraintGraph<'a> {
    pub(super) fn new(unknowns: usize) -> SpiceResult<Self> {
        let vertices = unknowns
            .checked_add(1)
            .ok_or_else(|| failure("constraint vertex count overflow"))?;
        let mut graph = UnGraph::default();
        for _ in 0..vertices {
            graph.add_node(());
        }
        Ok(Self {
            graph,
            known: UnionFind::new(vertices),
            rigid: UnionFind::new(vertices),
            visited: vec![false; vertices],
            visits: Vec::new(),
            pending: Vec::new(),
            end: None,
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
            if matches!(device.designator(), 'e' | 'h')
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
                _ => continue,
            };
            // An RF port source is on the internal #res node.
            let (positive, negative) = match (device.terminals(), device.rf_port()) {
                ([_, negative, _], Some(port)) => (port.internal, *negative),
                ([positive, negative], None) => (*positive, *negative),
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
        !self.known.equiv(a.index(), b.index()) && self.rigid.equiv(a.index(), b.index())
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
        if self.known.equiv(positive.index(), negative.index()) {
            // The old traversal started at the positive terminal with +0,
            // followed the unique forest path and negated the final result.
            let expected = -self.walk(positive, negative, PathKind::Known)?;
            if !expected.is_finite() {
                return Err(failure(format!(
                    "nonfinite voltage cycle at {origin} (path: {})",
                    self.path_description()
                )));
            }
            return Ok(Join::Cycle { expected });
        }
        self.known.union(positive.index(), negative.index());
        self.rigid.union(positive.index(), negative.index());
        self.graph.add_edge(
            positive.graph_node(),
            negative.graph_node(),
            Relation {
                positive,
                voltage: Voltage::Known(value),
                origin,
            },
        );
        Ok(Join::Tree)
    }

    /// Walk only until the requested endpoint is reached. For a known voltage
    /// path, its sequence of additions/subtractions is exactly the previous
    /// DFS's sequence, regardless of which unrelated branches are visited.
    fn walk(
        &mut self,
        start: VoltageVertex,
        target: VoltageVertex,
        kind: PathKind,
    ) -> SpiceResult<Real> {
        for visit in self.visits.drain(..) {
            self.visited[visit.node.index()] = false;
        }
        self.pending.clear();
        self.end = None;
        self.visits.push(Visit {
            node: start.graph_node(),
            parent: None,
            potential: 0.,
        });
        self.visited[start.index()] = true;
        self.pending.push(0);
        while let Some(index) = self.pending.pop() {
            let visit = self.visits[index];
            if visit.node == target.graph_node() {
                self.end = Some(index);
                return Ok(visit.potential);
            }
            for edge in self.graph.edges(visit.node) {
                let neighbor = edge.target();
                if self.visited[neighbor.index()] {
                    continue;
                }
                let relation = edge.weight();
                let potential = match relation.voltage {
                    Voltage::Known(value) => {
                        if relation.positive.graph_node() == visit.node {
                            visit.potential - value
                        } else {
                            visit.potential + value
                        }
                    }
                    Voltage::Controlled if matches!(kind, PathKind::Rigid) => 0.,
                    Voltage::Controlled => continue,
                };
                let next = self.visits.len();
                self.visits.push(Visit {
                    node: neighbor,
                    parent: Some((index, edge.id())),
                    potential,
                });
                self.visited[neighbor.index()] = true;
                if neighbor == target.graph_node() {
                    self.end = Some(next);
                    return Ok(potential);
                }
                self.pending.push(next);
            }
        }
        Err(failure("constraint component has no connecting path"))
    }

    pub(super) fn controlled_path(
        &mut self,
        start: VoltageVertex,
        target: VoltageVertex,
    ) -> SpiceResult<String> {
        self.walk(start, target, PathKind::Rigid)?;
        Ok(self.path_description())
    }

    pub(super) fn path_description(&self) -> String {
        let mut names = Vec::new();
        let mut index = self.end;
        while let Some(i) = index {
            let Some((parent, edge)) = self.visits[i].parent else {
                break;
            };
            names.push(self.graph[edge].origin.to_string());
            index = Some(parent);
        }
        names.reverse();
        if names.is_empty() {
            "same node".to_owned()
        } else {
            names.join(" -> ")
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
        let mut graph = ConstraintGraph::new(size).unwrap();
        for row in 0..size {
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
            visited += graph.visits.len();
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
        assert!(visited <= 4 * N);
    }
}
