//! Deterministic manufactured-solution sparse kernels, independent of the
//! simulator's device equations. Stamps retain physical creation order.
use std::hint::black_box;
use std::sync::Arc;
use std::time::{Duration, Instant};

use ngspice_rs::maths::complex::{ComplexLu, ComplexMatrix};
use ngspice_rs::maths::{
    FactorStorage, FactorWork, NiStep, NiWorkspace, Pattern, PatternTelemetry, SparseLu,
    SparseMatrix, Vector, solve_ni,
};
use ngspice_rs::primitives::Complex;

use super::report;

pub(super) fn report_storage(name: &str, phase: &str, storage: FactorStorage) {
    let fill_ratio = f64::from(u32::try_from(storage.factor_entries).unwrap())
        / f64::from(u32::try_from(storage.original_entries).unwrap());
    println!(
        "workload={name},phase={phase},external_order={},translated_order={},original_entries={},factor_entries={},fill_entries={},numeric_bytes={},fill_ratio={fill_ratio:.6}",
        storage.external_order,
        storage.translated_order,
        storage.original_entries,
        storage.factor_entries,
        storage.fill_entries(),
        storage.numeric_bytes
    );
}

pub(super) fn report_work(name: &str, phase: &str, work: FactorWork) {
    println!(
        "workload={name},phase={phase},order_attempts={},pivot_searches={},fixed_refactors={},successful_factors={},zero_pivots={},singular_failures={},nonfinite_failures={},resource_failures={},invalid_failures={},assembly_rejections={},zero_pivot_retries={}",
        work.order_attempts,
        work.pivot_searches,
        work.fixed_refactors,
        work.successful_factors,
        work.zero_pivots,
        work.singular_failures,
        work.nonfinite_failures,
        work.resource_failures,
        work.invalid_failures,
        work.assembly_rejections,
        work.zero_pivot_retries
    );
}

pub(super) fn report_cache(name: &str, phase: &str, telemetry: PatternTelemetry) {
    report_work(name, phase, telemetry.work.factor);
    println!(
        "workload={name},phase={phase},cache_totals=true,exact_input_hits={},cache_rhs_solves={},cache_rhs_failures={},real_to_complex={},complex_to_real={}",
        telemetry.work.exact_input_hits,
        telemetry.work.rhs_solves,
        telemetry.work.rhs_failures,
        telemetry.work.real_to_complex,
        telemetry.work.complex_to_real
    );
    if let Some(storage) = telemetry.storage {
        report_storage(name, phase, storage);
    } else {
        println!("workload={name},phase={phase},valid_factor_storage=false");
    }
}

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
    let mut x = Vector::default();
    match solve_ni(
        matrix,
        rhs,
        &mut NiWorkspace::default(),
        &mut x,
        reorder,
        0.,
    )
    .unwrap()
    {
        NiStep::Solved => x,
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
        let mut fresh_work = FactorWork::default();
        let mut worst = (0.0_f64, 0.0_f64);
        for _ in 0..repeats {
            let start = Instant::now();
            let lu = SparseLu::new(black_box(&bound), None).unwrap();
            ordering_time += start.elapsed();
            fresh_work += lu.factor_work();
            let (f, r) = error(&bound, &rhs, &lu.solve(&rhs).unwrap(), &mesh.exact);
            worst = (worst.0.max(f), worst.1.max(r));
        }
        report(name, "fresh_order_factor", repeats, ordering_time);
        let factors = SparseLu::new(&bound, None).unwrap();
        assert_eq!(fresh_work.successful_factors, u64::from(repeats));
        assert_eq!(fresh_work.pivot_searches, u64::from(repeats));
        report_work(name, "fresh_order_factor", fresh_work);
        report_storage(name, "fresh_order_factor", factors.factor_storage());
        println!(
            "workload={name},phase=fresh_order_factor,successful_snapshot_rhs_solves={repeats}"
        );
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
        println!("workload={name},phase=retained_solve,successful_snapshot_rhs_solves={repeats}");
        error(&bound, &rhs, &last, &mesh.exact);

        // Warm the exact pattern and exercise the two distinct NIiter paths.
        // Each changed diagonal forces new numeric factors; repeated values
        // after that refactor can retain the factors. The actual work is
        // read transactionally outside the timed loops.
        let _ = ni(&bound, &rhs, true);
        let warm = mesh.pattern.factor_telemetry();
        report_cache(name, "cache_after_warmup_cumulative", warm);
        let mut refactor_time = Duration::ZERO;
        let mut reuse_time = Duration::ZERO;
        for k in 0..repeats {
            bound
                .add(0, 0, if k.is_multiple_of(2) { 0.0625 } else { -0.0625 })
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
        let end = mesh.pattern.factor_telemetry();
        assert_eq!(
            end.work.factor.fixed_refactors - warm.work.factor.fixed_refactors,
            u64::from(repeats)
        );
        assert_eq!(
            end.work.factor.successful_factors - warm.work.factor.successful_factors,
            u64::from(repeats)
        );
        assert_eq!(
            end.work.exact_input_hits - warm.work.exact_input_hits,
            u64::from(repeats)
        );
        assert_eq!(
            end.work.rhs_solves - warm.work.rhs_solves,
            2 * u64::from(repeats)
        );
        assert_eq!(
            end.work.factor.order_attempts,
            warm.work.factor.order_attempts
        );
        assert_eq!(
            end.work.factor.pivot_searches,
            warm.work.factor.pivot_searches
        );
        report_cache(name, "cache_after_ni_repeats_cumulative", end);
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
            let mut work = FactorWork::default();
            let mut last_storage = None;
            let mut factor_entries_min = usize::MAX;
            let mut factor_entries_max = 0;
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
                work += lu.factor_work();
                let storage = lu.factor_storage();
                factor_entries_min = factor_entries_min.min(storage.factor_entries);
                factor_entries_max = factor_entries_max.max(storage.factor_entries);
                last_storage = Some(storage);
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
            assert_eq!(work.successful_factors, u64::from(count));
            report_work(name, mode, work);
            report_storage(name, mode, last_storage.unwrap());
            println!(
                "workload={name},mode={mode},factor_entries_min={factor_entries_min},factor_entries_max={factor_entries_max},successful_snapshot_rhs_solves={count},forward_abs={worst:e},residual_abs={residual:e}"
            );
        }
    }
}
