//! Shared physical-axis validation and exact/linear sample lookup for postprocessing.
//!
//! Duplicate coordinates model a vertical edge: at an exact query every value
//! at that coordinate must agree. Between coordinates use the adjacent limits.
//! C-compatible polynomial Fourier resampling intentionally uses its own edge
//! policy, so it does not consume this linear lookup.

use super::results::Plot;
use crate::primitives::{Real, SpiceError, SpiceResult};

pub(super) enum AxisError {
    MissingColumn,
    MissingSample(usize),
    Nonfinite(usize),
    Descending(usize),
}

pub(super) fn column_names(plot: &Plot) -> String {
    if plot.variables.is_empty() {
        return "no vectors".to_owned();
    }
    plot.variables
        .iter()
        .map(|v| v.name.as_str())
        .collect::<Vec<_>>()
        .join(" ")
}

pub(super) fn ordered_values(plot: &Plot, name: &str) -> Result<Vec<Real>, AxisError> {
    let column = plot.variable_index(name).ok_or(AxisError::MissingColumn)?;
    let values = plot
        .points
        .iter()
        .enumerate()
        .map(|(index, point)| {
            let value = point.get(column).ok_or(AxisError::MissingSample(index))?;
            if !value.is_finite() {
                return Err(AxisError::Nonfinite(index));
            }
            Ok(value.re)
        })
        .collect::<Result<Vec<_>, _>>()?;
    if let Some(index) = values.windows(2).position(|w| w[1] < w[0]) {
        return Err(AxisError::Descending(index + 1));
    }
    Ok(values)
}

pub(super) enum SampleError {
    OutOfRange,
    Discontinuous,
    Nonfinite,
    Operand(SpiceError),
}

/// Linear interpolation in an ordered finite axis. Preserve the historical
/// nonfused operation order; a nonfinite result is an error.
pub(super) fn linear_sample(
    axis: &[Real],
    x: Real,
    mut value: impl FnMut(usize) -> SpiceResult<Real>,
) -> Result<Real, SampleError> {
    if !x.is_finite()
        || !axis
            .first()
            .zip(axis.last())
            .is_some_and(|(first, last)| *first <= x && x <= *last)
    {
        return Err(SampleError::OutOfRange);
    }
    let right = axis.partition_point(|t| *t < x);
    let mut get = |index| value(index).map_err(SampleError::Operand);
    if axis[right] <= x {
        // The lower bound is already >= x, so this is the exact coordinate,
        // including either encoding of zero. Find the entire duplicate group.
        let first = get(right)?;
        for (index, _) in axis
            .iter()
            .enumerate()
            .skip(right + 1)
            .take_while(|(_, time)| **time <= x)
        {
            // An unordered operand is not an agreement either.
            if get(index)?.partial_cmp(&first) != Some(std::cmp::Ordering::Equal) {
                return Err(SampleError::Discontinuous);
            }
        }
        return Ok(first);
    }
    let left = right - 1;
    let y0 = get(left)?;
    let y1 = get(right)?;
    let result = interpolate(axis[left], y0, axis[right], y1, x);
    if !result.is_finite() {
        return Err(SampleError::Nonfinite);
    }
    Ok(result)
}

pub(super) fn interpolate(x0: Real, y0: Real, x1: Real, y1: Real, x: Real) -> Real {
    y0 + (y1 - y0) * (x - x0) / (x1 - x0)
}

#[cfg(test)]
mod tests {
    use super::{SampleError, linear_sample};

    #[test]
    fn duplicate_coordinates_use_exact_equality_including_signed_zero() {
        let axis = [-1., -0., 0., 1.];
        let values = [-1., -0., 0., 1.];
        assert!(matches!(
            linear_sample(&axis, 0., |index| Ok(values[index])),
            Ok(value) if value.to_bits() == (-0_f64).to_bits()
        ));
        let distinct = f64::from_bits(1);
        assert!(matches!(
            linear_sample(&axis, -0., |index| Ok(if index == 2 {
                distinct
            } else {
                values[index]
            })),
            Err(SampleError::Discontinuous)
        ));
    }

    #[test]
    fn adjacent_representable_coordinates_are_not_duplicates() {
        let adjacent = f64::from_bits(1_f64.to_bits() + 1);
        let axis = [1., adjacent];
        assert!(matches!(
            linear_sample(&axis, adjacent, |index| Ok(if index == 0 { 2. } else { 3. })),
            Ok(value) if value.to_bits() == 3_f64.to_bits()
        ));
    }
}
