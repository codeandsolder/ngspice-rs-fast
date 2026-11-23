//! Test-only helpers for classic MOS, JFET and MESFET fixtures.
//!
//! The model laws and fixture-specific tolerances remain in each test crate.
//! Keeping the sparse stamp traversal here makes the A*x-b sign and triplet
//! accumulation convention identical across all device families.
#![allow(dead_code)]

use ngspice_rs::analysis::{AnalysisContext, AnalysisInput, Plot, RunConfig};
use ngspice_rs::devices::{AnalysisMode, Circuit, LoadRequest, ModelContext, TrialState};
use ngspice_rs::maths::{SparseMatrix, Vector};
use ngspice_rs::netlist::Parser;
use ngspice_rs::netlist::ast::Netlist;
use ngspice_rs::netlist::source::parse_deck_text;
use ngspice_rs::primitives::{AnalysisKind, SpiceResult};
use std::path::Path;

pub(crate) struct Fixture {
    pub(crate) path: &'static str,
    pub(crate) title: &'static str,
}

impl Fixture {
    pub(crate) fn deck(&self, body: &str) -> SpiceResult<Netlist> {
        Parser::new().parse_deck(&parse_deck_text(
            Path::new(self.path),
            &format!("{}\n{body}\n.end\n", self.title),
        ))
    }

    pub(crate) fn circuit(&self, body: &str) -> SpiceResult<Circuit> {
        Circuit::from_netlist(&self.deck(body)?)
    }

    pub(crate) fn run_all(&self, body: &str) -> SpiceResult<Vec<Plot>> {
        let netlist = self.deck(body)?;
        let config = RunConfig::from_netlist(&netlist)?;
        let request = config.plan_for(&netlist.analyses[0])?;
        let mut circuit = config.circuit(&netlist)?;
        request.run_plots(&mut circuit, &config.context())
    }
}

pub(crate) fn run_request(
    circuit: &mut Circuit,
    kind: AnalysisKind,
    arguments: &[&str],
) -> SpiceResult<Plot> {
    AnalysisInput::with_arguments(kind, arguments.iter().copied())
        .resolve()
        .and_then(|plan| plan.run(circuit, &AnalysisContext::default()))
}

pub(crate) fn op(circuit: &mut Circuit) -> Plot {
    run_request(circuit, AnalysisKind::OperatingPoint, &[]).unwrap()
}

pub(crate) fn value(plot: &Plot, name: &str, point: usize) -> f64 {
    let index = plot
        .variable_index(name)
        .unwrap_or_else(|| panic!("missing {name}: {:?}", plot.variables()));
    plot.row(point).unwrap()[index].re
}

pub(crate) fn close(actual: f64, expected: f64, relative: f64, absolute: f64) {
    assert!(
        (actual - expected).abs() <= relative * expected.abs() + absolute,
        "{actual:e} != {expected:e}"
    );
}

pub(crate) fn bias(circuit: &Circuit) -> ngspice_rs::analysis::newton::NewtonSolution<TrialState> {
    ngspice_rs::analysis::bias::solve_dc(
        circuit,
        &ModelContext::default(),
        &Default::default(),
        &[],
        None,
        None,
    )
    .unwrap()
}

pub(crate) fn row(circuit: &Circuit, node: &str) -> usize {
    circuit
        .unknowns()
        .node_row(circuit.nodes().get(node).unwrap())
        .unwrap()
}

pub(crate) fn entry(matrix: &SparseMatrix, row: usize, col: usize) -> f64 {
    matrix
        .triplets()
        .iter()
        .filter(|t| t.row == row && t.col == col)
        .map(|t| t.value)
        .sum()
}

/// The native Newton residual: initial -b followed by each sparse A*x
/// contribution in stored triplet order, including repeated coordinates.
pub(crate) fn residual(a: &SparseMatrix, b: &Vector, x: &Vector) -> Vec<f64> {
    let mut result: Vec<f64> = b.as_slice().iter().map(|v| -v).collect();
    for triplet in a.triplets() {
        result[triplet.row] += triplet.value * x.as_slice()[triplet.col];
    }
    result
}

#[derive(Clone, Copy)]
pub(crate) enum JacobianMatrices {
    SmallSignal,
    SmallSignalAndNewton,
}

/// Check every node voltage column against a central difference of A(x)*x-b(x).
/// Explicit skip policies retain known C-model approximations without hiding
/// the other rows/columns or weakening their model-specific tolerances.
pub(crate) fn check_dc_jacobian(
    circuit: &Circuit,
    body: &str,
    relative: f64,
    absolute: f64,
    matrices: JacobianMatrices,
    skip_node: impl Fn(&str) -> bool,
) {
    let context = ModelContext::default();
    let history = circuit.state_history().unwrap();
    let solved = bias(circuit);
    let system = circuit
        .small_signal_system(&context, &solved.values)
        .unwrap();
    let n = circuit.unknown_count();
    let load = |x: &Vector| {
        let mut a = SparseMatrix::new(n, n);
        let mut b = Vector::zeros(n);
        let mut trial = history.trial();
        circuit
            .load(
                &LoadRequest {
                    mode: AnalysisMode::OperatingPoint,
                    solution: x,
                    model_context: &context,
                    integration: None,
                    history: &history,
                    forcing: None,
                },
                &mut a,
                &mut b,
                &mut trial,
            )
            .unwrap();
        let r = residual(&a, &b, x);
        (a, r)
    };
    let newton = match matrices {
        JacobianMatrices::SmallSignal => None,
        JacobianMatrices::SmallSignalAndNewton => Some(load(&solved.values).0),
    };
    for node in circuit.nodes().nodes() {
        let Some(col) = circuit.unknowns().node_row(node.id) else {
            continue;
        };
        if skip_node(&node.name) {
            continue;
        }
        let h = 1e-6;
        let mut low = solved.values.clone();
        let mut high = solved.values.clone();
        low.as_mut_slice()[col] -= h;
        high.as_mut_slice()[col] += h;
        let (_, rl) = load(&low);
        let (_, rh) = load(&high);
        for r in 0..n {
            let numerical = (rh[r] - rl[r]) / (2. * h);
            for analytic in std::iter::once(entry(&system.a, r, col))
                .chain(newton.iter().map(|a| entry(a, r, col)))
            {
                assert!(analytic.is_finite(), "{body}: row={r}, column={col}");
                assert!(
                    (numerical - analytic).abs() <= relative * analytic.abs() + absolute,
                    "{body}: row={r}, column={col}: {numerical:e} != {analytic:e}"
                );
            }
        }
    }
}

/// Preserve each bias grid's ordered primary/mirrored fixtures; model cards,
/// polarity and geometry are supplied by the individual test.
pub(crate) fn mirrored_bias_decks(
    biases: &[(f64, f64)],
    primary: impl Fn(f64, f64) -> String,
    mirrored: impl Fn(f64, f64) -> String,
) -> Vec<String> {
    let mut decks = Vec::with_capacity(biases.len() * 2);
    for &(vd, vg) in biases {
        decks.push(primary(vd, vg));
        decks.push(mirrored(vd, vg));
    }
    decks
}

#[cfg(test)]
mod tests {
    use super::{entry, residual};
    use ngspice_rs::maths::{SparseMatrix, Vector};

    #[test]
    fn repeated_sparse_triplets_preserve_signed_residual_and_index_order() {
        let mut a = SparseMatrix::new(3, 3);
        a.add(0, 2, 2.).unwrap();
        a.add(2, 0, -3.).unwrap();
        a.add(0, 2, 0.5).unwrap();
        a.add(1, 1, 4.).unwrap();
        let b = Vector::from_slice(&[1., -2., 3.]);
        let x = Vector::from_slice(&[4., 5., 6.]);
        assert_eq!(entry(&a, 0, 2), 2.5);
        assert_eq!(entry(&a, 2, 0), -3.);
        assert_eq!(residual(&a, &b, &x), [14., 22., -15.]);
    }
}
