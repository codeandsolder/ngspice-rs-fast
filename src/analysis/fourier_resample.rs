//! C-compatible moving polynomial interpolation (`maths/poly/interpolate.c`).
//! The absolute power basis and fit acceptance thresholds are intentional:
//! changing coordinates changes which degrees C falls back from.
use crate::maths::polynomial::evaluate;
use crate::primitives::{Real, SpiceError, SpiceResult};
fn failure(message: &str) -> SpiceError {
    SpiceError::Numerical {
        context: "Fourier interpolation".into(),
        message: message.into(),
    }
}

/// Scratch is allocated once per trace, not once per window or elimination row.
/// The coefficient tail deliberately survives a lower-degree fallback: C's final
/// interval evaluates at the original requested degree.
struct FitWorkspace {
    coefficients: Vec<Real>,
    matrix: Vec<Real>,
    rhs: Vec<Real>,
}

impl FitWorkspace {
    fn new(degree: usize) -> Self {
        let n = degree + 1;
        Self {
            coefficients: vec![0.; n],
            matrix: vec![0.; n * n],
            rhs: vec![0.; n],
        }
    }

    /// C `ft_polyfit` arithmetic, intentionally separate from the generic fitter:
    /// the acceptance test is sensitive to rounding in the absolute power basis.
    #[expect(
        clippy::suboptimal_flops,
        reason = "match C ft_polyfit rounding without FMA"
    )]
    fn fit_degree(&mut self, x: &[Real], y: &[Real]) -> bool {
        let n = x.len();
        let c = &mut self.coefficients[..n];
        if n == 2 {
            let dx = x[1] - x[0];
            c[0] = (x[1] * y[0] - x[0] * y[1]) / dx;
            c[1] = (y[1] - y[0]) / dx;
            return c.iter().all(|v| v.is_finite());
        }
        c.fill(0.);
        let matrix = &mut self.matrix[..n * n];
        let rhs = &mut self.rhs[..n];
        rhs.copy_from_slice(y);
        for (row, &t) in matrix.chunks_exact_mut(n).zip(x) {
            let mut power = 1.;
            for value in row {
                *value = power;
                power *= t;
            }
        }
        for col in 0..n {
            let mut pivot = col;
            for row in col + 1..n {
                if matrix[row * n + col].abs() > matrix[pivot * n + col].abs() {
                    pivot = row;
                }
            }
            if pivot != col {
                for k in 0..n {
                    matrix.swap(col * n + k, pivot * n + k);
                }
                rhs.swap(col, pivot);
            }
            if matrix[col * n + col] == 0. {
                return false;
            }
            for row in col + 1..n {
                eliminate(matrix, rhs, n, row, col);
            }
        }
        for col in (1..n).rev() {
            for row in (0..col).rev() {
                eliminate(matrix, rhs, n, row, col);
            }
        }
        for (i, value) in c.iter_mut().enumerate() {
            *value = rhs[i] / matrix[i * n + i];
        }
        c.iter().all(|v| v.is_finite())
            && x.iter().zip(y).all(|(&x, &y)| {
                let p = evaluate(c, x);
                let error = (p - y).abs();
                error <= 0.001 && error / p.abs().max(0.001) <= 0.001
            })
    }

    fn fit(&mut self, x: &[Real], y: &[Real], degree: usize, initial: bool) -> SpiceResult<usize> {
        let mut first = 0;
        for d in (1..=degree).rev() {
            if d < degree
                && (if initial {
                    d % 2 == 1
                } else {
                    (degree - d).is_multiple_of(2)
                })
            {
                first += 1;
            }
            if self.fit_degree(&x[first..first + d + 1], &y[first..first + d + 1]) {
                return Ok(d);
            }
        }
        Err(failure("no usable polynomial degree"))
    }
}

#[expect(
    clippy::suboptimal_flops,
    reason = "match C ft_polyfit elimination rounding without FMA"
)]
fn eliminate(matrix: &mut [Real], rhs: &mut [Real], n: usize, row: usize, col: usize) {
    let factor = matrix[row * n + col] / matrix[col * n + col];
    for k in 0..n {
        matrix[row * n + k] -= factor * matrix[col * n + k];
    }
    rhs[row] -= factor * rhs[col];
}

/// Resample onto an ascending physical grid with C's degree fallback and edge policy.
pub(super) fn resample(
    times: &[Real],
    values: &[Real],
    grid: &[Real],
    degree: usize,
) -> SpiceResult<Vec<Real>> {
    if times.len() != values.len()
        || times.len() <= degree
        || degree == 0
        || degree > 16
        || grid.len() < 2
        || times
            .iter()
            .chain(values)
            .chain(grid)
            .any(|x| !x.is_finite())
        || times.windows(2).any(|w| w[0] > w[1])
        || grid.windows(2).any(|w| w[0] >= w[1])
    {
        return Err(failure("invalid interpolation samples or degree"));
    }
    let middle = degree.div_ceil(2);
    let mut start = 0;
    while start + degree + 1 < times.len() && times[start + middle] < grid[0] {
        start += 1;
    }
    let mut x = vec![times[start]];
    let mut y = vec![values[start]];
    let mut end = start;
    while x.len() <= degree && end + 1 < times.len() {
        if times[end + 1] == times[end] {
            if x.len() == 1 {
                end += 1;
                y[0] = values[end];
                continue;
            }
            let k = x.len() - 1;
            x[k] -= (x[k] - x[k - 1]) * 0.001;
        }
        end += 1;
        x.push(times[end]);
        y.push(values[end]);
    }
    if x.len() <= degree {
        return Err(failure("too few distinct samples"));
    }
    let mut workspace = FitWorkspace::new(degree);
    let mut used = workspace.fit(&x, &y, degree, true)?;
    let mut out = Vec::with_capacity(grid.len());
    while out.len() < grid.len() && grid[out.len()] <= x[middle] {
        out.push(evaluate(&workspace.coefficients[..=used], grid[out.len()]));
    }
    for next in end + 1..times.len() {
        if out.len() == grid.len() {
            break;
        }
        let discarded = x[0];
        x.rotate_left(1);
        y.rotate_left(1);
        x[degree] = times[next];
        y[degree] = values[next];
        if x[degree] == x[degree - 1] {
            let width = if degree == 1 {
                x[0] - discarded
            } else {
                x[degree - 1] - x[degree - 2]
            };
            x[degree - 1] -= width * 0.001;
        }
        if next < times.len() - degree && x[middle] < grid[out.len()] {
            continue;
        }
        used = workspace.fit(&x, &y, degree, false)?;
        while out.len() < grid.len() && grid[out.len()] <= x[middle] {
            out.push(evaluate(&workspace.coefficients[..=used], grid[out.len()]));
        }
    }
    while out.len() < grid.len() {
        out.push(evaluate(&workspace.coefficients, grid[out.len()]));
    }
    if out.iter().any(|v| !v.is_finite()) {
        return Err(failure("nonfinite interpolated value"));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::resample;

    #[test]
    fn cubic_tail_matches_pinned_c_on_ill_conditioned_absolute_times() {
        // Last eleven samples of the live Fourier parity deck. Values below
        // come from pinned C 059066b5 ft_interpolate, not the Rust fitter.
        let time = [
            0.003990280000000205,
            0.003991280000000205,
            0.0039922800000002055,
            0.003993280000000206,
            0.003994280000000206,
            0.003995280000000206,
            0.003996280000000206,
            0.003997280000000206,
            0.003998280000000206,
            0.003999140000000103,
            0.004,
        ];
        let values = [
            0.43896539706326915,
            0.4452380318247266,
            0.4515128284949237,
            0.4577895393556249,
            0.4640679166130364,
            0.47034771240757456,
            0.4766286788236575,
            0.48291056789947884,
            0.4891931316368212,
            0.4945964869320709,
            0.499999999999999,
        ];
        let result = resample(&time, &values, &[0.0039949, 0.003995], 3).unwrap();
        let expected = [0.46796123867079586_f64, 0.4685892409220287];
        assert_eq!(
            result.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            expected.map(f64::to_bits)
        );
    }
}
