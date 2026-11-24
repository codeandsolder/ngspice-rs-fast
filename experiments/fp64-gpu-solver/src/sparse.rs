use std::collections::BTreeSet;

#[derive(Debug, Clone, Copy)]
struct Eliminate {
    pivot_diagonal: u32,
    lower: u32,
    pivot_row: u32,
    target_row: u32,
    first_update: u32,
    last_update: u32,
}

/// An arbitrary fixed CSR sparsity pattern with all exact symbolic fill,
/// a precomputed elimination schedule, and no per-candidate symbolic work.
pub struct Plan {
    pub n: usize,
    pub nnz: usize,
    pub row_offsets: Vec<u32>,
    columns: Vec<u32>,
    pub diagonal: Vec<u32>,
    original: Vec<BTreeSet<usize>>,
    operations: Vec<Eliminate>,
    updates: Vec<(u32, u32)>,
}

impl Plan {
    #[must_use]
    pub fn new(n: usize) -> Self {
        assert!((4..=256).contains(&n));
        let width = (n as f64).sqrt().floor() as usize;
        // Grid-like grounded conductance topology, plus nonlocal asymmetric
        // controlled couplings; these intentionally induce sparse LU fill.
        let original = (0..n)
            .map(|i| {
                let mut row = BTreeSet::from([i]);
                if i % width > 0 {
                    row.insert(i - 1);
                }
                if i % width + 1 < width && i + 1 < n {
                    row.insert(i + 1);
                }
                if i >= width {
                    row.insert(i - width);
                }
                if i + width < n {
                    row.insert(i + width);
                }
                if i % 3 == 0 {
                    row.insert((i * 7 + 11) % n);
                }
                row
            })
            .collect::<Vec<_>>();
        let mut rows = original.clone();
        for pivot in 0..n {
            let fill = rows[pivot]
                .range((pivot + 1)..)
                .copied()
                .collect::<Vec<_>>();
            for target in pivot + 1..n {
                if rows[target].contains(&pivot) {
                    rows[target].extend(fill.iter().copied());
                }
            }
        }
        let mut row_offsets = Vec::with_capacity(n + 1);
        let mut columns = Vec::new();
        row_offsets.push(0);
        for row in &rows {
            columns.extend(row.iter().map(|&column| u32::try_from(column).unwrap()));
            row_offsets.push(u32::try_from(columns.len()).unwrap());
        }
        let nnz = columns.len();
        let locate = |i: usize, j: usize| {
            let begin = row_offsets[i] as usize;
            let end = row_offsets[i + 1] as usize;
            let offset = columns[begin..end].binary_search(&(j as u32)).unwrap();
            u32::try_from(begin + offset).unwrap()
        };
        let diagonal = (0..n).map(|i| locate(i, i)).collect();
        let mut operations = Vec::new();
        let mut updates = Vec::new();
        for pivot in 0..n {
            for target in pivot + 1..n {
                if !rows[target].contains(&pivot) {
                    continue;
                }
                let first_update = u32::try_from(updates.len()).unwrap();
                for &column in rows[pivot].range((pivot + 1)..) {
                    updates.push((locate(pivot, column), locate(target, column)));
                }
                operations.push(Eliminate {
                    pivot_diagonal: locate(pivot, pivot),
                    lower: locate(target, pivot),
                    pivot_row: u32::try_from(pivot).unwrap(),
                    target_row: u32::try_from(target).unwrap(),
                    first_update,
                    last_update: u32::try_from(updates.len()).unwrap(),
                });
            }
        }
        Self {
            n,
            nnz,
            row_offsets,
            columns,
            diagonal,
            original,
            operations,
            updates,
        }
    }

    #[must_use]
    pub fn input_stride(&self) -> usize {
        self.nnz + self.n
    }

    #[must_use]
    pub fn output_stride(&self) -> usize {
        self.n + 1
    }

    #[must_use]
    pub fn initial_nonzeros(&self) -> usize {
        self.original.iter().map(BTreeSet::len).sum()
    }

    /// Compact immutable plan shared by every GPU lane. The initial four
    /// fields are N, NNZ, elimination count, and update count. Then CSR
    /// row pointers, columns, diagonal offsets, six-word elimination records
    /// and two-word update records.
    #[must_use]
    pub fn words(&self) -> Vec<u32> {
        let mut words = vec![
            u32::try_from(self.n).unwrap(),
            u32::try_from(self.nnz).unwrap(),
            u32::try_from(self.operations.len()).unwrap(),
            u32::try_from(self.updates.len()).unwrap(),
        ];
        words.extend_from_slice(&self.row_offsets);
        words.extend_from_slice(&self.columns);
        words.extend_from_slice(&self.diagonal);
        for step in &self.operations {
            words.extend([
                step.pivot_diagonal,
                step.lower,
                step.pivot_row,
                step.target_row,
                step.first_update,
                step.last_update,
            ]);
        }
        for &(source, target) in &self.updates {
            words.extend([source, target]);
        }
        words
    }

    /// Nonuniform physical MNA-like conductance coefficients. The RHS is
    /// constructed independently by multiplying A by the prescribed solution.
    #[must_use]
    pub fn generate(&self, count: usize) -> (Vec<f64>, Vec<f64>) {
        let mut input = vec![0.0; count * self.input_stride()];
        let mut truth = vec![0.0; count * self.n];
        for system in 0..count {
            let base = system * self.input_stride();
            let t = system * self.n;
            for (i, slot) in truth[t..t + self.n].iter_mut().enumerate() {
                *slot = 0.5 + ((system * 3 + i * 7) % 47) as f64 * 0.03125;
            }
            for i in 0..self.n {
                let mut sum = 0.0;
                let begin = self.row_offsets[i] as usize;
                let end = self.row_offsets[i + 1] as usize;
                for index in begin..end {
                    let column = self.columns[index] as usize;
                    if column == i || !self.original[i].contains(&column) {
                        continue;
                    }
                    let coefficient =
                        -0.05 - ((i * 11 + column * 17 + system * 7) % 23) as f64 * 0.006;
                    input[base + index] = coefficient;
                    sum += coefficient.abs();
                }
                input[base + self.diagonal[i] as usize] = 1.0 + sum;
            }
            for i in 0..self.n {
                let mut rhs = 0.0;
                let begin = self.row_offsets[i] as usize;
                let end = self.row_offsets[i + 1] as usize;
                for idx in begin..end {
                    rhs += input[base + idx] * truth[t + self.columns[idx] as usize];
                }
                input[base + self.nnz + i] = rhs;
            }
        }
        (input, truth)
    }

    /// Numerical factorization follows a precomputed no-pivot fill plan, and
    /// signals singular/ill-conditioned pivots instead of silently returning
    /// the last finite iterate.
    pub fn solve_cpu(&self, input: &[f64], output: &mut [f64]) -> usize {
        assert_eq!(input.len() % self.input_stride(), 0);
        let count = input.len() / self.input_stride();
        assert_eq!(output.len(), count * self.output_stride());
        let mut scratch = input.to_vec();
        let mut failures = 0;
        'systems: for system in 0..count {
            let base = system * self.input_stride();
            let result_base = system * self.output_stride();
            let a = &mut scratch[base..base + self.input_stride()];
            let solution = &mut output[result_base..result_base + self.output_stride()];
            for op in &self.operations {
                let pivot = a[op.pivot_diagonal as usize];
                if !pivot.is_finite() || pivot.abs() < 1e-12 {
                    solution.fill(f64::NAN);
                    solution[self.n] = 1.0;
                    failures += 1;
                    continue 'systems;
                }
                let factor = a[op.lower as usize] / pivot;
                a[op.lower as usize] = factor;
                a[self.nnz + op.target_row as usize] -=
                    factor * a[self.nnz + op.pivot_row as usize];
                for &(source, dest) in
                    &self.updates[op.first_update as usize..op.last_update as usize]
                {
                    a[dest as usize] -= factor * a[source as usize];
                }
            }
            for i in (0..self.n).rev() {
                let pivot = a[self.diagonal[i] as usize];
                if !pivot.is_finite() || pivot.abs() < 1e-12 {
                    solution.fill(f64::NAN);
                    solution[self.n] = 1.0;
                    failures += 1;
                    continue 'systems;
                }
                let mut value = a[self.nnz + i];
                for index in self.row_offsets[i] as usize..self.row_offsets[i + 1] as usize {
                    let column = self.columns[index] as usize;
                    if column > i {
                        value -= a[index] * solution[column];
                    }
                }
                solution[i] = value / pivot;
            }
            solution[self.n] = 0.0;
        }
        failures
    }

    #[must_use]
    pub fn validate(&self, input: &[f64], output: &[f64], truth: &[f64]) -> (f64, f64, usize) {
        assert_eq!(output.len() / self.output_stride() * self.n, truth.len());
        let mut residual = 0.0_f64;
        let mut error = 0.0_f64;
        let mut failures = 0;
        for system in 0..truth.len() / self.n {
            let base = system * self.input_stride();
            let target =
                &output[system * self.output_stride()..(system + 1) * self.output_stride()];
            if target[self.n] != 0.0 {
                failures += 1;
                continue;
            }
            if target[..self.n].iter().any(|value| !value.is_finite()) {
                return (f64::INFINITY, f64::INFINITY, failures);
            }
            for i in 0..self.n {
                let mut lhs = 0.0;
                for idx in self.row_offsets[i] as usize..self.row_offsets[i + 1] as usize {
                    lhs += input[base + idx] * target[self.columns[idx] as usize];
                }
                residual = residual.max((lhs - input[base + self.nnz + i]).abs());
                error = error.max((target[i] - truth[system * self.n + i]).abs());
            }
        }
        (residual, error, failures)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sparse_elimination_with_fill_reconstructs_independent_known_solutions() {
        for n in [7, 16, 32, 64] {
            let plan = Plan::new(n);
            assert!(plan.nnz >= plan.initial_nonzeros());
            for count in [1, 2, 17, 65] {
                let (input, truth) = plan.generate(count);
                let mut output = vec![f64::NAN; count * plan.output_stride()];
                assert_eq!(plan.solve_cpu(&input, &mut output), 0);
                let (r, e, fail) = plan.validate(&input, &output, &truth);
                assert!(r < 1e-11, "n={n} count={count}: residual={r:e}");
                assert!(e < 1e-11, "n={n} count={count}: error={e:e}");
                assert_eq!(fail, 0);
            }
        }
    }

    #[test]
    fn zero_pivot_is_reported_and_does_not_contaminate_other_candidates() {
        let plan = Plan::new(32);
        let (input, truth) = plan.generate(3);
        let mut bad = input.clone();
        bad[plan.diagonal[0] as usize] = 0.0;
        let mut output = vec![0.0; 3 * plan.output_stride()];
        assert_eq!(plan.solve_cpu(&bad, &mut output), 1);
        let (_, _, failures) = plan.validate(&bad, &output, &truth);
        assert_eq!(failures, 1);
        assert!(output[plan.n].is_sign_positive());
        assert_eq!(plan.solve_cpu(&input, &mut output), 0);
        assert_eq!(plan.validate(&input, &output, &truth).2, 0);
    }
}
