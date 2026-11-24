use std::simd::Simd;

pub const N: usize = 32;
pub const STRIDE: usize = 4 * N;

// Stable diagonally dominant, nonuniform grounded conductance ladders;
// synthesize RHS from the exact answer, not from our own solver.
#[must_use]
pub fn generate(count: usize) -> (Vec<f64>, Vec<f64>) {
    let mut matrices = vec![0.0; count * STRIDE];
    let mut truth = vec![0.0; count * N];
    for system in 0..count {
        for i in 0..N {
            let lower = if i == 0 {
                0.0
            } else {
                -0.25 - ((system * 17 + i * 13) % 19) as f64 * 0.01
            };
            let upper = if i + 1 == N {
                0.0
            } else {
                -0.18 - ((system * 7 + i * 11) % 17) as f64 * 0.01
            };
            let diag = 1.25 + lower.abs() + upper.abs() + (i % 7) as f64 * 0.03;
            let k = system * STRIDE + i * 4;
            matrices[k] = lower;
            matrices[k + 1] = diag;
            matrices[k + 2] = upper;
            truth[system * N + i] = 0.6 + ((system + i * 7) % 29) as f64 * 0.04;
        }
        for i in 0..N {
            let k = system * STRIDE + i * 4;
            let x = system * N + i;
            matrices[k + 3] = matrices[k + 1] * truth[x]
                + if i > 0 {
                    matrices[k] * truth[x - 1]
                } else {
                    0.0
                }
                + if i + 1 < N {
                    matrices[k + 2] * truth[x + 1]
                } else {
                    0.0
                };
        }
    }
    (matrices, truth)
}

pub fn scalar(input: &[f64], out: &mut [f64]) {
    assert_eq!(input.len(), out.len() * 4);
    assert_eq!(out.len() % N, 0);
    let mut cprime = [0.0; N];
    let mut dprime = [0.0; N];
    for (matrix, result) in input.chunks_exact(STRIDE).zip(out.chunks_exact_mut(N)) {
        let inv = matrix[1].recip();
        cprime[0] = matrix[2] * inv;
        dprime[0] = matrix[3] * inv;
        for i in 1..N {
            let k = i * 4;
            let a = matrix[k];
            let den = matrix[k + 1] - a * cprime[i - 1];
            cprime[i] = if i == N - 1 { 0.0 } else { matrix[k + 2] / den };
            dprime[i] = (matrix[k + 3] - a * dprime[i - 1]) / den;
        }
        result[N - 1] = dprime[N - 1];
        for i in (0..N - 1).rev() {
            result[i] = dprime[i] - cprime[i] * result[i + 1];
        }
    }
}

// SoA register lanes across independent systems, never SIMD across the
// dependent elimination chain within any one circuit.
pub fn simd4(input: &[f64], out: &mut [f64]) {
    assert_eq!(input.len(), out.len() * 4);
    assert_eq!(out.len() % N, 0);
    let groups = out.len() / (4 * N);
    let mut cp = [Simd::<f64, 4>::splat(0.0); N];
    let mut dp = [Simd::<f64, 4>::splat(0.0); N];
    for group in 0..groups {
        let load = |row: usize, field: usize| {
            Simd::from_array(std::array::from_fn(|lane| {
                input[(group * 4 + lane) * STRIDE + row * 4 + field]
            }))
        };
        let inv = Simd::splat(1.0) / load(0, 1);
        cp[0] = load(0, 2) * inv;
        dp[0] = load(0, 3) * inv;
        for i in 1..N {
            let a = load(i, 0);
            let den = load(i, 1) - a * cp[i - 1];
            cp[i] = if i == N - 1 {
                Simd::splat(0.0)
            } else {
                load(i, 2) / den
            };
            dp[i] = (load(i, 3) - a * dp[i - 1]) / den;
        }
        let mut solution = [Simd::splat(0.0); N];
        solution[N - 1] = dp[N - 1];
        for i in (0..N - 1).rev() {
            solution[i] = dp[i] - cp[i] * solution[i + 1];
        }
        for (i, value) in solution.iter().enumerate() {
            let lanes = value.to_array();
            for lane in 0..4 {
                out[(group * 4 + lane) * N + i] = lanes[lane];
            }
        }
    }
    let completed = groups * 4;
    scalar(&input[completed * STRIDE..], &mut out[completed * N..]);
}

#[must_use]
pub fn maximum_residual(input: &[f64], actual: &[f64], expected: &[f64]) -> (f64, f64) {
    assert_eq!(actual.len(), expected.len());
    let mut residual: f64 = 0.0;
    let mut deviation: f64 = 0.0;
    for (system, output) in actual.chunks_exact(N).enumerate() {
        if output.iter().any(|value| !value.is_finite()) {
            return (f64::INFINITY, f64::INFINITY);
        }
        for i in 0..N {
            let k = system * STRIDE + i * 4;
            let a = input[k + 1] * output[i]
                + if i > 0 { input[k] * output[i - 1] } else { 0.0 }
                + if i + 1 < N {
                    input[k + 2] * output[i + 1]
                } else {
                    0.0
                };
            residual = residual.max((a - input[k + 3]).abs());
            deviation = deviation.max((output[i] - expected[system * N + i]).abs());
        }
    }
    (residual, deviation)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn scalar_and_simd_reconstruct_independent_grounded_ladders() {
        for count in [1, 2, 3, 4, 5, 8, 17, 67] {
            let (input, truth) = generate(count);
            for solve in [scalar as fn(&[f64], &mut [f64]), simd4] {
                let mut output = vec![f64::NAN; truth.len()];
                solve(&input, &mut output);
                let (residual, deviation) = maximum_residual(&input, &output, &truth);
                assert!(residual < 1e-12, "count {count}: residual={residual}");
                assert!(deviation < 1e-12, "count {count}: deviation={deviation}");
            }
        }
    }
}
