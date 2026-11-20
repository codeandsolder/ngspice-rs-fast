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
    fn fit_degree(&mut self, times: &[Real], samples: &[Real]) -> bool {
        let order = times.len();
        let coefficients = &mut self.coefficients[..order];
        if order == 2 {
            let dx = times[1] - times[0];
            coefficients[0] = (times[1] * samples[0] - times[0] * samples[1]) / dx;
            coefficients[1] = (samples[1] - samples[0]) / dx;
            return coefficients.iter().all(|value| value.is_finite());
        }
        coefficients.fill(0.);
        let matrix = &mut self.matrix[..order * order];
        let rhs = &mut self.rhs[..order];
        rhs.copy_from_slice(samples);
        for (row, &time) in matrix.chunks_exact_mut(order).zip(times) {
            let mut power = 1.;
            for value in row {
                *value = power;
                power *= time;
            }
        }
        for col in 0..order {
            let mut pivot = col;
            for row in col + 1..order {
                if matrix[row * order + col].abs() > matrix[pivot * order + col].abs() {
                    pivot = row;
                }
            }
            if pivot != col {
                for entry in 0..order {
                    matrix.swap(col * order + entry, pivot * order + entry);
                }
                rhs.swap(col, pivot);
            }
            if matrix[col * order + col] == 0. {
                return false;
            }
            for row in col + 1..order {
                eliminate(matrix, rhs, order, row, col);
            }
        }
        for col in (1..order).rev() {
            for row in (0..col).rev() {
                eliminate(matrix, rhs, order, row, col);
            }
        }
        for (index, value) in coefficients.iter_mut().enumerate() {
            *value = rhs[index] / matrix[index * order + index];
        }
        coefficients.iter().all(|value| value.is_finite())
            && times.iter().zip(samples).all(|(&time, &sample)| {
                let estimate = evaluate(coefficients, time);
                let error = (estimate - sample).abs();
                error <= 0.001 && error / estimate.abs().max(0.001) <= 0.001
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
            if self.fit_degree(&x[first..=first + d], &y[first..=first + d]) {
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

/// Detect coincident physical samples, not approximately equal times.
fn same_timestamp(left: Real, right: Real) -> bool {
    left.partial_cmp(&right) == Some(std::cmp::Ordering::Equal)
}

/// Move the older coincident sample back by C's fractional interval.
/// Keep the displacement rounded separately, as in `ft_interpolate`.
const fn retreat_duplicate(time: &mut Real, interval: Real) {
    let displacement = interval * 0.001;
    *time -= displacement;
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
    let mut window_times = vec![times[start]];
    let mut window_values = vec![values[start]];
    let mut end = start;
    while window_times.len() <= degree && end + 1 < times.len() {
        if same_timestamp(times[end + 1], times[end]) {
            if window_times.len() == 1 {
                end += 1;
                window_values[0] = values[end];
                continue;
            }
            let last = window_times.len() - 1;
            let width = window_times[last] - window_times[last - 1];
            retreat_duplicate(&mut window_times[last], width);
        }
        end += 1;
        window_times.push(times[end]);
        window_values.push(values[end]);
    }
    if window_times.len() <= degree {
        return Err(failure("too few distinct samples"));
    }
    let mut workspace = FitWorkspace::new(degree);
    let mut used = workspace.fit(&window_times, &window_values, degree, true)?;
    let mut out = Vec::with_capacity(grid.len());
    while out.len() < grid.len() && grid[out.len()] <= window_times[middle] {
        out.push(evaluate(&workspace.coefficients[..=used], grid[out.len()]));
    }
    for next in end + 1..times.len() {
        if out.len() == grid.len() {
            break;
        }
        let discarded = window_times[0];
        window_times.rotate_left(1);
        window_values.rotate_left(1);
        window_times[degree] = times[next];
        window_values[degree] = values[next];
        if same_timestamp(window_times[degree], window_times[degree - 1]) {
            let width = if degree == 1 {
                window_times[0] - discarded
            } else {
                window_times[degree - 1] - window_times[degree - 2]
            };
            retreat_duplicate(&mut window_times[degree - 1], width);
        }
        if next < times.len() - degree && window_times[middle] < grid[out.len()] {
            continue;
        }
        used = workspace.fit(&window_times, &window_values, degree, false)?;
        while out.len() < grid.len() && grid[out.len()] <= window_times[middle] {
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
    use super::{resample, same_timestamp};

    #[test]
    fn coincident_times_include_signed_zero_but_not_adjacent_floats() {
        assert!(same_timestamp(-0., 0.));
        assert!(!same_timestamp(1., 1_f64.next_up()));
    }

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
