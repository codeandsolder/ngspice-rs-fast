//! Shared validated physical coordinates and postprocessing traversal policies.
//!
//! Equal coordinates retain every sample, in original order. Linear lookup
//! checks an entire coincident group; continuous integrals clip both edges;
//! measurement events scan the preceding segment but stop at the last stored
//! sample in the window. C polynomial Fourier fitting retains its own policy.

use std::borrow::Cow;
use std::ops::Range;

use super::results::Plot;
use crate::primitives::{Real, SpiceError, SpiceResult};

#[derive(Debug)]
pub(super) enum AxisError {
    MissingColumn,
    MissingSample(usize),
    Nonfinite(usize),
    Descending(usize),
}

pub(super) fn column_names(plot: &Plot) -> String {
    if plot.variables().is_empty() {
        return "no vectors".to_owned();
    }
    plot.variables()
        .iter()
        .map(|v| v.name.as_str())
        .collect::<Vec<_>>()
        .join(" ")
}

/// A finite, nondecreasing axis. Borrowed axes need no copy; extracting a
/// column from a plot owns just that column. Empty and singleton axes are
/// valid coordinates; each consumer decides whether it has enough samples.
pub(super) struct OrderedAxis<'a> {
    values: Cow<'a, [Real]>,
}

impl OrderedAxis<'static> {
    pub(super) fn from_plot(plot: &Plot, name: &str) -> Result<Self, AxisError> {
        let column = plot.variable_index(name).ok_or(AxisError::MissingColumn)?;
        let values = plot
            .points()
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
        // Preserve error precedence: missing/nonfinite column entries are
        // reported before a descending coordinate anywhere in the axis.
        Self::validate_order(&values)?;
        Ok(Self {
            values: Cow::Owned(values),
        })
    }
}

impl<'a> OrderedAxis<'a> {
    pub(super) fn borrowed(values: &'a [Real]) -> Result<Self, AxisError> {
        if let Some(index) = values.iter().position(|x| !x.is_finite()) {
            return Err(AxisError::Nonfinite(index));
        }
        Self::validate_order(values)?;
        Ok(Self {
            values: Cow::Borrowed(values),
        })
    }

    fn validate_order(values: &[Real]) -> Result<(), AxisError> {
        if let Some(index) = values.windows(2).position(|w| w[1] < w[0]) {
            return Err(AxisError::Descending(index + 1));
        }
        Ok(())
    }

    pub(super) fn as_slice(&self) -> &[Real] {
        &self.values
    }

    pub(super) fn len(&self) -> usize {
        self.values.len()
    }

    pub(super) fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    pub(super) fn lower_bound(&self, x: Real) -> usize {
        self.values.partition_point(|t| *t < x)
    }

    pub(super) fn has_duplicates(&self) -> bool {
        self.values.windows(2).any(|w| same_coordinate(w[0], w[1]))
    }

    /// The exact group is a range (including signed zeros), never a snapped
    /// coordinate. A bracket uses the last left and first right sample.
    fn locate(&self, x: Real) -> Option<Location> {
        if !x.is_finite()
            || !self
                .values
                .first()
                .zip(self.values.last())
                .is_some_and(|(first, last)| *first <= x && x <= *last)
        {
            return None;
        }
        let right = self.lower_bound(x);
        if self.values[right] <= x {
            let tail = &self.values[right + 1..];
            let end = right
                + 1
                + if tail.first().is_some_and(|t| *t <= x) {
                    tail.partition_point(|t| *t <= x)
                } else {
                    0
                };
            Some(Location::Exact(right..end))
        } else {
            Some(Location::Between(segment(&self.values, right - 1)))
        }
    }

    /// DERIV historically uses the first inclusive bracket, including a
    /// duplicate at the beginning of the axis. Return whether the query is
    /// exact so its operand group can be checked before selecting a stencil.
    pub(super) fn derivative_bracket(&self, x: Real) -> Option<(Segment, bool)> {
        if self.len() < 2 {
            return None;
        }
        match self.locate(x)? {
            Location::Exact(group) => Some((segment(&self.values, group.start.max(1) - 1), true)),
            Location::Between(pair) => Some((pair, false)),
        }
    }

    /// Closed window with at least one stored sample. Preserve the existing
    /// postprocessing refusal for windows wholly inside a sample bracket.
    /// Callers validate finite, ascending bounds clipped to the axis first.
    pub(super) fn sample_window(&self, from: Real, to: Real) -> Option<SampleWindow<'_>> {
        if !from.is_finite() || !to.is_finite() || from > to {
            return None;
        }
        let first = self.lower_bound(from);
        let end = self.values.partition_point(|t| *t <= to);
        if first >= end {
            return None;
        }
        Some(SampleWindow {
            values: &self.values,
            from,
            to,
            first,
            last: end - 1,
        })
    }

    /// Linear interpolation using the exact stored coordinate or the adjacent
    /// limits. The operand callback is evaluated in historical traversal order.
    pub(super) fn linear_sample(
        &self,
        x: Real,
        mut value: impl FnMut(usize) -> SpiceResult<Real>,
    ) -> Result<Real, SampleError> {
        let mut get = |index| value(index).map_err(SampleError::Operand);
        match self.locate(x).ok_or(SampleError::OutOfRange)? {
            Location::Exact(group) => {
                let first = get(group.start)?;
                for index in group.start + 1..group.end {
                    if !same_coordinate(get(index)?, first) {
                        return Err(SampleError::Discontinuous);
                    }
                }
                Ok(first)
            }
            Location::Between(pair) => {
                let result = interpolate(pair.x0, get(pair.left)?, pair.x1, get(pair.right())?, x);
                if !result.is_finite() {
                    return Err(SampleError::Nonfinite);
                }
                Ok(result)
            }
        }
    }
}

enum Location {
    Exact(Range<usize>),
    Between(Segment),
}

/// A pair of adjacent stored samples, retaining zero-width jump segments.
#[derive(Clone, Copy)]
pub(super) struct Segment {
    pub(super) left: usize,
    pub(super) x0: Real,
    pub(super) x1: Real,
}

impl Segment {
    pub(super) const fn right(self) -> usize {
        self.left + 1
    }

    pub(super) fn is_jump(self) -> bool {
        same_coordinate(self.x0, self.x1)
    }
}

const fn segment(values: &[Real], left: usize) -> Segment {
    Segment {
        left,
        x0: values[left],
        x1: values[left + 1],
    }
}

/// Shared exact coordinate identity. Signed zeros coincide, adjacent floats
/// do not, and unordered operands are not agreement.
pub(super) fn same_coordinate(left: Real, right: Real) -> bool {
    left.partial_cmp(&right) == Some(std::cmp::Ordering::Equal)
}

/// A nonempty stored-sample window with explicit policy-specific views.
#[derive(Clone, Copy)]
pub(super) struct SampleWindow<'a> {
    values: &'a [Real],
    from: Real,
    to: Real,
    first: usize,
    last: usize,
}

impl<'a> SampleWindow<'a> {
    pub(super) const fn first_sample(self) -> (usize, Real) {
        (self.first, self.values[self.first])
    }

    /// Raw sample extrema do not invent interpolated extrema at boundaries.
    pub(super) fn raw_samples(self) -> impl Iterator<Item = (usize, Real)> + 'a {
        self.values[self.first..=self.last]
            .iter()
            .copied()
            .enumerate()
            .map(move |(offset, x)| (self.first + offset, x))
    }

    /// Events consume the prior stored sample (if any), including a jump
    /// at FROM, and stop at the last in-window sample. C does not search the
    /// clipped right edge; measurement selection applies its own threshold.
    pub(super) fn event_segments(self) -> impl Iterator<Item = Segment> + 'a {
        (self.first.saturating_sub(1)..self.last).map(move |left| segment(self.values, left))
    }

    /// Continuous integrals use both interpolated edges, keeping each
    /// stored duplicate and reading each operand in the original order.
    pub(super) fn clipped_samples(
        self,
        value: impl Fn(usize) -> SpiceResult<Real> + 'a,
    ) -> impl Iterator<Item = SpiceResult<(Real, Real)>> + 'a {
        let before = (self.first > 0 && self.values[self.first] > self.from)
            .then(|| SamplePoint::Clipped(segment(self.values, self.first - 1), self.from));
        let after = (self.last + 1 < self.values.len() && self.values[self.last] < self.to)
            .then(|| SamplePoint::Clipped(segment(self.values, self.last), self.to));
        before
            .into_iter()
            .chain(self.raw_samples().map(|(i, x)| SamplePoint::Stored(i, x)))
            .chain(after)
            .map(move |point| match point {
                SamplePoint::Stored(i, x) => value(i).map(|y| (x, y)),
                SamplePoint::Clipped(pair, x) => Ok((
                    x,
                    interpolate(pair.x0, value(pair.left)?, pair.x1, value(pair.right())?, x),
                )),
            })
    }
}

enum SamplePoint {
    Stored(usize, Real),
    Clipped(Segment, Real),
}

pub(super) enum SampleError {
    OutOfRange,
    Discontinuous,
    Nonfinite,
    Operand(SpiceError),
}

fn interpolate(x0: Real, y0: Real, x1: Real, y1: Real, x: Real) -> Real {
    y0 + (y1 - y0) * (x - x0) / (x1 - x0)
}

#[cfg(test)]
mod tests {
    use super::{AxisError, OrderedAxis, SampleError};

    #[test]
    fn duplicate_coordinates_use_exact_equality_including_signed_zero() {
        let axis = OrderedAxis::borrowed(&[-1., -0., 0., 1.]).unwrap();
        let values = [-1., -0., 0., 1.];
        assert!(matches!(
            axis.linear_sample(0., |index| Ok(values[index])),
            Ok(value) if value.to_bits() == (-0_f64).to_bits()
        ));
        let distinct = f64::from_bits(1);
        assert!(matches!(
            axis.linear_sample(-0., |index| Ok(if index == 2 {
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
        let values = [1., adjacent];
        let axis = OrderedAxis::borrowed(&values).unwrap();
        assert!(matches!(
            axis.linear_sample(adjacent, |index| Ok(if index == 0 { 2. } else { 3. })),
            Ok(value) if value.to_bits() == 3_f64.to_bits()
        ));
        assert!(!axis.has_duplicates());
    }

    #[test]
    fn finite_validation_precedes_order_and_empty_axes_are_valid() {
        assert!(matches!(
            OrderedAxis::borrowed(&[2., 1., f64::NAN]),
            Err(AxisError::Nonfinite(2))
        ));
        assert!(matches!(
            OrderedAxis::borrowed(&[2., 1.]),
            Err(AxisError::Descending(1))
        ));
        let empty = OrderedAxis::borrowed(&[]).unwrap();
        assert!(empty.sample_window(0., 1.).is_none());
        assert!(matches!(
            empty.linear_sample(0., |_| Ok(0.)),
            Err(SampleError::OutOfRange)
        ));
        let single = OrderedAxis::borrowed(&[2.]).unwrap();
        assert_eq!(
            single
                .sample_window(2., 2.)
                .unwrap()
                .raw_samples()
                .collect::<Vec<_>>(),
            [(0, 2.)]
        );
        assert!(single.derivative_bracket(2.).is_none());
    }

    #[test]
    fn window_views_distinguish_raw_samples_edges_and_jumps() {
        let axis = OrderedAxis::borrowed(&[0., 1., 1., 2., 3.]).unwrap();
        let values = [0., 1., 10., 20., 30.];
        let window = axis.sample_window(0.5, 2.5).unwrap();
        assert_eq!(
            window.raw_samples().collect::<Vec<_>>(),
            [(1, 1.), (2, 1.), (3, 2.)]
        );
        let clipped = window
            .clipped_samples(|i| Ok(values[i]))
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        assert_eq!(
            clipped,
            [(0.5, 0.5), (1., 1.), (1., 10.), (2., 20.), (2.5, 25.)]
        );
        let event_segments = window
            .event_segments()
            .map(|s| (s.left, s.right(), s.is_jump()))
            .collect::<Vec<_>>();
        assert_eq!(event_segments, [(0, 1, false), (1, 2, true), (2, 3, false)]);
        assert!(axis.sample_window(2.25, 2.75).is_none());
        assert_eq!(
            axis.derivative_bracket(1.)
                .map(|(s, exact)| (s.left, exact)),
            Some((0, true))
        );
        assert_eq!(
            axis.derivative_bracket(1.5)
                .map(|(s, exact)| (s.left, exact)),
            Some((2, false))
        );
    }

    #[test]
    fn window_partitions_match_direct_scans_on_nonuniform_axes() {
        let mut seed = 0x1234_5678_u32;
        let mut advance = || {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            seed >> 24
        };
        for _ in 0..64 {
            let mut values = vec![0.];
            for _ in 0..31 {
                values.push(values.last().copied().unwrap() + f64::from(advance() % 5));
            }
            let axis = OrderedAxis::borrowed(&values).unwrap();
            for _ in 0..32 {
                let a = f64::from(advance() % 128) * 0.5;
                let b = f64::from(advance() % 128) * 0.5;
                let (from, to) = (a.min(b), a.max(b));
                let expected = values
                    .iter()
                    .copied()
                    .enumerate()
                    .filter(|&(_, x)| from <= x && x <= to)
                    .collect::<Vec<_>>();
                let got = axis
                    .sample_window(from, to)
                    .map(|w| w.raw_samples().collect::<Vec<_>>())
                    .unwrap_or_default();
                assert_eq!(got, expected);
            }
        }
    }
}
