//! The static transfer map of the XSPICE `pwl` code model, as `inpcom.c`
//! instantiates it for an E/G `TABLE` (`fraction=TRUE limit=TRUE`).
//!
//! C reference (behaviour only): `src/xspice/icm/analog/pwl/cfunc.mod`
//! (`cm_pwl`) and `src/xspice/cm/cmutil.c` (`cm_smooth_corner`). The table is
//! extended by one mirrored abscissa at each end whose ordinate repeats the
//! end value (`limit=TRUE`), every interior corner is replaced by a parabola
//! over `±fraction` of its shorter adjacent segment, and outside the first and
//! last midpoints the extended end segments (flat) apply.
//!
//! The model's iteration-to-iteration input limiting (`limit_x_value`, a
//! Newton aid that flags non-convergence while the input moves by more than
//! 30 % of a segment) is not reproduced: it changes the iteration path, not
//! the converged operating point.

use crate::primitives::{Real, SpiceError, SpiceResult};

/// A validated smoothed piecewise-linear transfer map.
#[derive(Debug, Clone, PartialEq)]
pub struct XspicePwl {
    x: Vec<Real>,
    y: Vec<Real>,
    fraction: Real,
    midpoints: Vec<Real>,
}

#[derive(Clone, Copy)]
enum Region {
    Segment {
        lower: usize,
        at: usize,
    },
    Corner {
        index: usize,
        domain: Real,
        lower_slope: Real,
        upper_slope: Real,
    },
}

impl XspicePwl {
    /// Builds the map from `(x, y)` points and the smoothing `fraction`.
    ///
    /// # Errors
    /// Fewer than two points, non-finite values, abscissas that do not
    /// strictly increase (the C model silently misbehaves on those) or a
    /// fraction outside the model's `input_domain` limits `[1e-12, 0.5]`.
    pub fn new(points: &[(Real, Real)], fraction: Real) -> SpiceResult<Self> {
        if points.len() < 2 {
            return Err(invalid("a TABLE needs at least two (x, y) points"));
        }
        if points.iter().any(|(x, y)| !x.is_finite() || !y.is_finite()) {
            return Err(invalid("TABLE points must be finite"));
        }
        if points.windows(2).any(|w| w[0].0 >= w[1].0) {
            return Err(invalid(
                "TABLE abscissas must strictly increase (the XSPICE pwl model has no \
                 ordering check and misbehaves otherwise)",
            ));
        }
        if !(1e-12..=0.5).contains(&fraction) {
            return Err(invalid("TABLE smoothing domain outside [1e-12, 0.5]"));
        }
        let n = points.len();
        let mut x = Vec::with_capacity(n + 2);
        let mut y = Vec::with_capacity(n + 2);
        x.push(mirrored_abscissa(points[0].0, points[1].0));
        y.push(points[0].1);
        for (px, py) in points {
            x.push(*px);
            y.push(*py);
        }
        x.push(mirrored_abscissa(points[n - 1].0, points[n - 2].0));
        y.push(points[n - 1].1);
        let midpoints = x
            .windows(2)
            .map(|pair| reference_midpoint(pair[0], pair[1]))
            .collect();
        Ok(Self {
            x,
            y,
            fraction,
            midpoints,
        })
    }

    /// The output and its derivative with respect to the input.
    #[must_use = "the returned pair supplies the transfer value and its exact input derivative"]
    pub fn evaluate(&self, input: Real) -> (Real, Real) {
        self.evaluate_region(input, self.region(input, 0.))
    }

    #[expect(
        clippy::float_cmp,
        reason = "the analytic side is selected only at an exact boundary"
    )]
    fn region(&self, input: Real, direction: Real) -> Region {
        let size = self.x.len();
        let less = |boundary| input < boundary || (input == boundary && direction < 0.);
        let greater = |boundary| input > boundary || (input == boundary && direction > 0.);
        // Preserve the native end-test priority and equal-midpoint choices.
        if less(self.midpoints[0]) || (input == self.midpoints[0] && direction == 0.) {
            return Region::Segment { lower: 0, at: 0 };
        }
        if greater(self.midpoints[size - 2])
            || (input == self.midpoints[size - 2] && direction == 0.)
        {
            return Region::Segment {
                lower: size - 2,
                at: size - 1,
            };
        }
        let i = 1 + self.midpoints[1..size - 2].partition_point(|&midpoint| {
            greater(midpoint) || (input == midpoint && direction == 0.)
        });
        let lower = self.x[i] - self.x[i - 1];
        let upper = self.x[i + 1] - self.x[i];
        let domain = self.fraction * lower.min(upper);
        if less(self.x[i] - domain) {
            Region::Segment {
                lower: i - 1,
                at: i,
            }
        } else if less(self.x[i] + domain) {
            Region::Corner {
                index: i,
                domain,
                lower_slope: (self.y[i] - self.y[i - 1]) / lower,
                upper_slope: (self.y[i + 1] - self.y[i]) / upper,
            }
        } else {
            Region::Segment { lower: i, at: i }
        }
    }

    fn evaluate_region(&self, input: Real, region: Region) -> (Real, Real) {
        match region {
            Region::Segment { lower, at } => self.segment(input, lower, at),
            Region::Corner {
                index,
                domain,
                lower_slope,
                upper_slope,
            } => smooth_corner(
                input,
                self.x[index],
                self.y[index],
                domain,
                lower_slope,
                upper_slope,
            ),
        }
    }

    pub(super) fn right_taylor(
        &self,
        input: &crate::maths::taylor::Taylor,
    ) -> SpiceResult<(crate::maths::taylor::Taylor, crate::maths::taylor::Taylor)> {
        use crate::maths::taylor::Taylor;
        let region = self.region(input.base(), input.direction());
        let (value, slope) = self.evaluate_region(input.base(), region);
        let curvature = match region {
            Region::Segment { .. } => 0.,
            Region::Corner {
                domain,
                lower_slope,
                upper_slope,
                ..
            } => (upper_slope - lower_slope) / (2. * domain),
        };
        let delta = input.clone().with_base(0.)?;
        let output = delta
            .scale(slope)?
            .add(&delta.mul(&delta)?.scale(curvature * 0.5)?)?
            .with_base(value)?;
        let slope = delta
            .scale(curvature)?
            .add(&Taylor::constant(slope, input.order()))?;
        Ok((output, slope))
    }

    fn segment(&self, input: Real, lower: usize, at: usize) -> (Real, Real) {
        let (x, y) = (&self.x, &self.y);
        let slope = (y[lower + 1] - y[lower]) / (x[lower + 1] - x[lower]);
        let offset = (input - x[at]) * slope;
        (y[at] + offset, slope)
    }
}

/// C extends an endpoint as `2 * endpoint - neighbour`, without fusion.
const fn mirrored_abscissa(endpoint: Real, neighbour: Real) -> Real {
    let twice_endpoint = 2. * endpoint;
    twice_endpoint - neighbour
}

/// Match the reference addition then division, including overflow and
/// subnormal rounding. A robust floating-point midpoint changes that contract.
const fn reference_midpoint(left: Real, right: Real) -> Real {
    let sum = left + right;
    sum / 2.
}

/// The parabola joining two slopes over `centre ± domain`, matching value and
/// slope at both ends (`cm_smooth_corner`).
fn smooth_corner(
    input: Real,
    centre: Real,
    value: Real,
    domain: Real,
    lower_slope: Real,
    upper_slope: Real,
) -> (Real, Real) {
    let x_upper = centre + domain;
    let upper_rise = upper_slope * domain;
    let y_upper = value + upper_rise;
    let a = ((upper_slope - lower_slope) / 4.) * (1. / domain);
    let upper_derivative = 2. * a * x_upper;
    let b = upper_slope - upper_derivative;
    let upper_quadratic = a * x_upper * x_upper;
    let upper_linear = b * x_upper;
    let c = y_upper - upper_quadratic - upper_linear;
    let quadratic = a * input * input;
    let linear = b * input;
    let derivative = 2. * a * input;
    (quadratic + linear + c, derivative + b)
}

fn invalid(message: &str) -> SpiceError {
    SpiceError::Unsupported {
        feature: message.to_owned(),
        location: None,
    }
}

#[cfg(test)]
mod tests {
    use super::{XspicePwl, reference_midpoint, smooth_corner};

    #[test]
    fn table_is_flat_outside_and_smooth_at_corners() {
        let map = XspicePwl::new(&[(0., 0.), (1., 2.), (2., 3.)], 0.1).unwrap();
        assert_eq!(map.evaluate(-5.), (0., 0.));
        assert_eq!(map.evaluate(10.), (3., 0.));
        // Linear away from corners.
        let (y, slope) = map.evaluate(0.5);
        assert!((y - 1.).abs() < 1e-15 && (slope - 2.).abs() < 1e-15);
        // C's operating point of `table {v(in)} = (0,0) (1,2) (2,3)` at 1 V.
        assert!((map.evaluate(1.).0 - 1.975).abs() < 1e-12);
        // Continuity and slope continuity at the parabola ends.
        for (edge, h) in [(0.9, 1e-9), (1.1, 1e-9), (-0.1, 1e-9), (0.1, 1e-9)] {
            let below = map.evaluate(edge - h);
            let above = map.evaluate(edge + h);
            assert!((below.0 - above.0).abs() < 1e-8, "{edge}");
            assert!((below.1 - above.1).abs() < 1e-6, "{edge}");
        }
    }

    #[test]
    fn derivative_matches_finite_differences() {
        let map = XspicePwl::new(&[(-1., -0.5), (0., 0.), (1., 2.), (3., 2.5)], 0.1).unwrap();
        let mut x = -2.;
        while x < 4. {
            let h = 1e-7;
            let numeric = (map.evaluate(x + h).0 - map.evaluate(x - h).0) / (2. * h);
            let (_, analytic) = map.evaluate(x);
            assert!(
                (numeric - analytic).abs() < 1e-5,
                "{x}: {numeric} {analytic}"
            );
            x += 0.0137;
        }
    }

    #[test]
    fn reference_midpoint_keeps_overflow_and_signed_zero() {
        assert!(reference_midpoint(f64::MAX, f64::MAX).is_infinite());
        assert_eq!(reference_midpoint(-0., -0.).to_bits(), (-0_f64).to_bits());
        assert_eq!(
            reference_midpoint(f64::from_bits(1), f64::from_bits(2)).to_bits(),
            2
        );
    }

    // Reproduce the reference's linear selection independently of the
    // production boundary cache and binary search.
    fn scan(map: &XspicePwl, input: f64) -> (f64, f64) {
        let size = map.x.len();
        let midpoint = |i| reference_midpoint(map.x[i], map.x[i + 1]);
        if input <= midpoint(0) {
            return map.segment(input, 0, 0);
        }
        if input >= midpoint(size - 2) {
            return map.segment(input, size - 2, size - 1);
        }
        for i in 1..size - 1 {
            if input >= midpoint(i) {
                continue;
            }
            let left = map.x[i] - map.x[i - 1];
            let right = map.x[i + 1] - map.x[i];
            let domain = map.fraction * left.min(right);
            return if input < map.x[i] - domain {
                map.segment(input, i - 1, i)
            } else if input < map.x[i] + domain {
                smooth_corner(
                    input,
                    map.x[i],
                    map.y[i],
                    domain,
                    (map.y[i] - map.y[i - 1]) / left,
                    (map.y[i + 1] - map.y[i]) / right,
                )
            } else {
                map.segment(input, i, i)
            };
        }
        map.segment(input, size - 2, size - 1)
    }

    #[test]
    fn interior_search_preserves_reference_scan_at_every_boundary() {
        let tables: &[&[(f64, f64)]] = &[
            &[(-5., 3.), (-2., 0.), (0., 2.), (1., 4.), (1.25, -2.)],
            &[(0., 0.), (2., 4.)],
            &[(-f64::MAX, 1.), (-1., 3.), (1., -2.), (f64::MAX, 4.)],
            &[(f64::MAX * 0.6, 1.), (f64::MAX * 0.8, 3.), (f64::MAX, 4.)],
            &[(0., 1.), (f64::from_bits(1), 2.), (f64::from_bits(2), 3.)],
        ];
        for points in tables {
            for fraction in [0.1, 0.5] {
                let map = XspicePwl::new(points, fraction).unwrap();
                let inputs = map
                    .midpoints
                    .iter()
                    .copied()
                    .flat_map(|value| [value.next_down(), value, value.next_up()])
                    .chain([f64::NEG_INFINITY, -0., 0., f64::INFINITY, f64::NAN]);
                for input in inputs {
                    let actual = map.evaluate(input);
                    let expected = scan(&map, input);
                    for (actual, expected) in [(actual.0, expected.0), (actual.1, expected.1)] {
                        if expected.is_nan() {
                            assert!(actual.is_nan());
                        } else {
                            assert_eq!(actual.to_bits(), expected.to_bits(), "{input}");
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn invalid_tables_are_rejected() {
        assert!(XspicePwl::new(&[(0., 0.)], 0.1).is_err());
        assert!(XspicePwl::new(&[(1., 0.), (0., 1.)], 0.1).is_err());
        assert!(XspicePwl::new(&[(0., 0.), (0., 1.)], 0.1).is_err());
        assert!(XspicePwl::new(&[(0., 0.), (1., f64::NAN)], 0.1).is_err());
    }
}
