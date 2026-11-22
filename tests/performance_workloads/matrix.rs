//! Deterministic manufactured-solution sparse kernels, independent of the
//! simulator's device equations. Stamps retain physical creation order.
use std::hint::black_box;
use std::sync::Arc;
use std::time::{Duration, Instant};

use ngspice_rs::maths::complex::{ComplexLu, ComplexMatrix};
use ngspice_rs::maths::{NiStep, Pattern, SparseLu, SparseMatrix, Vector, solve_ni};
use ngspice_rs::primitives::Complex;

use super::report;

struct Mesh {
    n: usize,
    stamps: Vec<(usize, usize, f64)>,
    pattern: Arc<Pattern>,
    exact: Vector,
}

impl Mesh {
    fn new(width: u32, height: u32) -> Self {
        let n = (width * height) as usize;
        let width = width as usize;
        let mut stamps = Vec::new();
        // Every node has a positive shunt. Edge conductances are nonuniform
        // but binary-exact: a well-conditioned, genuinely sparse Laplacian.
        for node in 0..n {
            stamps.push((node, node, 0.125));
            for other in [
                (node % width + 1 < width).then_some(node + 1),
                (node + width < n).then_some(node + width),
            ]
            .into_iter()
            .flatten()
            {
                let g = 0.5 + f64::from(u32::try_from(node % 7).unwrap()) / 8.;
                stamps.extend([
                    (node, node, g),
                    (other, other, g),
                    (node, other, -g),
                    (other, node, -g),
                ]);
            }
        }
        let pattern = Arc::new(Pattern::new(n, stamps.iter().map(|&(r, c, _)| (r, c))).unwrap());
        let exact = Vector::from(
            (0..n)
                .map(|i| f64::from(u32::try_from((i * 13) % 31).unwrap()) / 16. - 0.75)
                .collect::<Vec<_>>(),
        );
        Self {
            n,
            stamps,
            pattern,
            exact,
        }
    }

    fn stamp(&self, matrix: &mut SparseMatrix) {
        matrix.clear();
        for &(r, c, v) in &self.stamps {
            matrix.add(r, c, v).unwrap();
        }
        matrix.fold_duplicates();
    }
}

fn error(matrix: &SparseMatrix, rhs: &Vector, x: &Vector, exact: &Vector) -> (f64, f64) {
    let ax = matrix.mul_vector(x).unwrap();
    let forward = x
        .as_slice()
        .iter()
        .zip(exact.as_slice())
        .map(|(&a, &b)| (a - b).abs())
        .fold(0., f64::max);
    let residual = ax
        .as_slice()
        .iter()
        .zip(rhs.as_slice())
        .map(|(&a, &b)| (a - b).abs())
        .fold(0., f64::max);
    assert!(
        x.is_finite() && forward < 1e-10 && residual < 1e-10,
        "forward={forward:e}, residual={residual:e}"
    );
    (forward, residual)
}

fn ni(matrix: &SparseMatrix, rhs: &Vector, reorder: bool) -> Vector {
    match solve_ni(matrix, rhs, &mut None, reorder, 0.).unwrap() {
        NiStep::Solved(x) => x,
        NiStep::ZeroPivot => panic!("grounded positive-definite workload returned zero pivot"),
    }
}

pub(super) fn real(name: &str, width: u32, height: u32, repeats: u32) {
    let init = Instant::now();
    let mesh = Mesh::new(width, height);
    let mut bound = SparseMatrix::bound(mesh.pattern.clone()).unwrap();
    report(name, "pattern_setup", 1, init.elapsed());
    let mut triplet = SparseMatrix::new(mesh.n, mesh.n);
    mesh.stamp(&mut triplet);
    mesh.stamp(&mut bound);
    assert_eq!(triplet.triplets(), bound.triplets());
    let rhs = bound.mul_vector(&mesh.exact).unwrap();
    println!(
        "workload={name},n={},stamps={},pattern_nnz={},coefficient_nnz={}",
        mesh.n,
        mesh.stamps.len(),
        mesh.pattern.len(),
        bound.nnz()
    );

    for _ in 0..3 {
        let start = Instant::now();
        for _ in 0..repeats {
            mesh.stamp(black_box(&mut triplet));
        }
        report(name, "triplet_assembly", repeats, start.elapsed());
        let start = Instant::now();
        for _ in 0..repeats {
            mesh.stamp(black_box(&mut bound));
        }
        report(name, "bound_assembly", repeats, start.elapsed());

        let mut ordering_time = Duration::ZERO;
        let mut worst = (0.0_f64, 0.0_f64);
        for _ in 0..repeats {
            let start = Instant::now();
            let lu = SparseLu::new(black_box(&bound), None).unwrap();
            ordering_time += start.elapsed();
            let (f, r) = error(&bound, &rhs, &lu.solve(&rhs).unwrap(), &mesh.exact);
            worst = (worst.0.max(f), worst.1.max(r));
        }
        report(name, "fresh_order_factor", repeats, ordering_time);
        let factors = SparseLu::new(&bound, None).unwrap();
        let mut workspace = Vec::new();
        let start = Instant::now();
        let mut last = Vector::zeros(mesh.n);
        for _ in 0..repeats {
            last = black_box(
                factors
                    .solve_with_workspace(black_box(&rhs), &mut workspace)
                    .unwrap(),
            );
        }
        report(name, "retained_solve", repeats, start.elapsed());
        error(&bound, &rhs, &last, &mesh.exact);

        // Warm the exact pattern and exercise the two distinct NIiter paths.
        // Each changed diagonal forces new numeric factors; repeated values
        // after that refactor can retain the factors. Report requested calls,
        // not inferred internal instrumentation.
        let _ = ni(&bound, &rhs, true);
        let mut refactor_time = Duration::ZERO;
        let mut reuse_time = Duration::ZERO;
        for k in 0..repeats {
            bound
                .add(0, 0, if k % 2 == 0 { 0.0625 } else { -0.0625 })
                .unwrap();
            let rhs = bound.mul_vector(&mesh.exact).unwrap();
            let start = Instant::now();
            let changed = black_box(ni(&bound, &rhs, false));
            refactor_time += start.elapsed();
            let start = Instant::now();
            let same = black_box(ni(&bound, &rhs, false));
            reuse_time += start.elapsed();
            assert_eq!(same.as_slice(), changed.as_slice());
            let (f, r) = error(&bound, &rhs, &same, &mesh.exact);
            worst = (worst.0.max(f), worst.1.max(r));
        }
        report(
            name,
            "changed_coefficients_ni_solve",
            repeats,
            refactor_time,
        );
        report(name, "identical_coefficients_ni_solve", repeats, reuse_time);
        println!(
            "workload={name},forward_abs={:e},residual_abs={:e}",
            worst.0, worst.1
        );
        mesh.stamp(&mut bound);
    }
}

pub(super) fn frequency_ensemble(width: u32, height: u32, count: u32) {
    let name = "complex_frequency_ensemble";
    let mesh = Mesh::new(width, height);
    let mut g = SparseMatrix::new(mesh.n, mesh.n);
    mesh.stamp(&mut g);
    let mut c = SparseMatrix::new(mesh.n, mesh.n);
    for i in 0..mesh.n {
        c.add(i, i, 1e-6).unwrap();
    }
    c.fold_duplicates();
    let exact = mesh
        .exact
        .as_slice()
        .iter()
        .map(|&v| Complex::new(v, 0.25 - v))
        .collect::<Vec<_>>();
    println!(
        "workload={name},n={},g_nnz={},c_nnz={},frequencies={count}",
        mesh.n,
        g.nnz(),
        c.nnz()
    );
    for reuse in [false, true] {
        for _ in 0..3 {
            let mut previous: Option<ComplexLu> = None;
            let mut assemble = Duration::ZERO;
            let mut factor = Duration::ZERO;
            let mut solve = Duration::ZERO;
            let mut worst = 0.0_f64;
            let mut residual = 0.0_f64;
            for i in 0..count {
                let omega = 100. * (1. + f64::from(i)).powi(2);
                let start = Instant::now();
                let m = ComplexMatrix::from_operators(&g, &c, omega).unwrap();
                assemble += start.elapsed();
                // Build the right hand side independently of the complex
                // assembly / solve using the real coefficient operators.
                let mut rhs = vec![Complex::ZERO; mesh.n];
                for t in g.triplets() {
                    rhs[t.row] = rhs[t.row] + Complex::real(t.value) * exact[t.col];
                }
                for t in c.triplets() {
                    rhs[t.row] = rhs[t.row] + Complex::imaginary(t.value * omega) * exact[t.col];
                }
                let start = Instant::now();
                let lu = if let Some(old) = &previous {
                    m.factorize_reusing(old).unwrap()
                } else {
                    m.factorize().unwrap()
                };
                factor += start.elapsed();
                let start = Instant::now();
                let x = black_box(lu.solve(&rhs).unwrap());
                solve += start.elapsed();
                let mut ax = vec![Complex::ZERO; mesh.n];
                for t in g.triplets() {
                    ax[t.row] = ax[t.row] + Complex::real(t.value) * x[t.col];
                }
                for t in c.triplets() {
                    ax[t.row] = ax[t.row] + Complex::imaginary(t.value * omega) * x[t.col];
                }
                for ((&a, &b), (&want, &got)) in x.iter().zip(&exact).zip(rhs.iter().zip(&ax)) {
                    assert!(a.is_finite());
                    worst = worst.max((a - b).magnitude());
                    residual = residual.max((want - got).magnitude());
                }
                if reuse {
                    previous = Some(lu);
                }
            }
            assert!(worst < 1e-10 && residual < 1e-10);
            let mode = if reuse { "reuse_order" } else { "fresh_order" };
            report(name, &format!("{mode}_assembly"), count, assemble);
            report(name, &format!("{mode}_factor"), count, factor);
            report(name, &format!("{mode}_solve"), count, solve);
            println!(
                "workload={name},mode={mode},orders_requested={},reuse_requested={},forward_abs={worst:e},residual_abs={residual:e}",
                if reuse { 1 } else { count },
                if reuse { count - 1 } else { 0 }
            );
        }
    }
}
