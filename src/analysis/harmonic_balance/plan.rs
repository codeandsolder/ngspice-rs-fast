//! Deterministic, budget-checked harmonic lattice for torus-periodic signals.
//!
//! Exactly commensurate tones are locked explicitly. Silent near-integer
//! guessing is incorrect for genuine two-tone intermodulation studies.
use crate::primitives::{Complex, Real, SpiceError, SpiceResult};
use std::{f64::consts::TAU, sync::Arc};

/// A source tone and its requested harmonic range.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Tone {
    /// Adds an independent (potentially incommensurate) torus dimension.
    Independent {
        /// Fundamental Hz, finite and strictly positive.
        frequency_hz: Real,
        /// Included positive and negative integer harmonic orders.
        order: u16,
    },
    /// Exactly rationally related to an earlier tone.
    ///
    /// The ratio is explicit so independent close-spaced tones cannot be
    /// silently folded together using an arbitrary floating tolerance.
    Locked {
        /// Earlier tone ordinal, whose group may itself be locked.
        to: usize,
        /// Numerator of the exact rational relation.
        numerator: u32,
        /// Denominator of the exact rational relation.
        denominator: u32,
        /// Requested positive and negative harmonic bounds.
        order: u16,
    },
}

/// Bounds checked before allocating the lattice, sample grid or Fourier work.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HarmonicLimits {
    /// Maximum independent (torus) dimensions.
    pub dimensions: usize,
    /// Number of retained harmonic tuples.
    pub coefficients: usize,
    /// Number of oversampled real torus points.
    pub samples: usize,
    /// Upper bound for direct retained-coefficient × sample projections.
    pub dft_products: usize,
}

impl Default for HarmonicLimits {
    fn default() -> Self {
        Self {
            dimensions: 8,
            coefficients: 4096,
            samples: 16384,
            dft_products: 8_000_000,
        }
    }
}

fn invalid(message: impl Into<String>) -> SpiceError {
    SpiceError::Numerical {
        context: "harmonic balance lattice".into(),
        message: message.into(),
    }
}

fn checked_mul(a: usize, b: usize) -> SpiceResult<usize> {
    a.checked_mul(b)
        .ok_or_else(|| invalid("harmonic lattice index product overflow"))
}

fn gcd(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a
}

fn lcm(a: u64, b: u64) -> SpiceResult<u64> {
    (a / gcd(a, b))
        .checked_mul(b)
        .ok_or_else(|| invalid("locked-tone ratio LCM overflow"))
}

#[derive(Debug, Clone, Copy)]
struct Relation {
    root: usize,
    numerator: u64,
    denominator: u64,
    order: u16,
}

/// One independent torus dimension, after collapsing locked sources.
#[derive(Debug, Clone, PartialEq)]
pub struct Axis {
    /// Integer lattice fundamental in hertz, after locked-ratio reduction.
    pub frequency_hz: Real,
    /// Maximum retained magnitude along this dimension.
    pub order: usize,
    /// Number of grid points used for evaluation and projection.
    pub samples: usize,
}

/// One retained lattice mode. Negative-frequency conjugates are stored too.
#[derive(Debug, Clone, PartialEq)]
pub struct HarmonicPoint {
    /// Integer coordinates, in axis order.
    pub index: Vec<i32>,
    /// Signed physical frequency of the mode, in hertz.
    pub frequency_hz: Real,
    /// Ordinal of its negative-frequency conjugate.
    pub conjugate: usize,
}

/// Validated multidimensional harmonic and real-sample grid.
///
/// Mode order is lexicographic with the last axis fastest. The DC tuple
/// is kept once and every non-DC tuple has exactly one conjugate.
#[derive(Debug, Clone)]
pub struct HarmonicPlan {
    axes: Vec<Axis>,
    modes: Vec<HarmonicPoint>,
    sample_count: usize,
    /// Actual per-scalar forward plus inverse separable complex DFT products.
    transform_products: usize,
    /// Physical source frequencies after explicit ratio resolution.
    tones_hz: Vec<Real>,
    /// Per-tone group and signed integer weight, for load/forcing routing.
    weights: Vec<(usize, usize)>,
    /// Reusable e^(-j k theta) per axis, [harmonic][sample].
    twiddles: Vec<Arc<[Complex]>>,
}

impl HarmonicPlan {
    /// Construct a reusable bounded lattice and deterministic sample grid.
    ///
    /// Each dimension uses at least 4H+1 points: an explicit 2×
    /// nonlinearity oversampling policy, not a promise that arbitrary
    /// non-bandlimited device currents are alias-free. Convergence with
    /// increased harmonic order is an independent solver requirement.
    ///
    /// # Errors
    /// Nonfinite frequencies, invalid ratios, coincident independent
    /// frequency tuples, integer overflow, or violated work limits.
    pub fn new(tones: &[Tone], limits: HarmonicLimits) -> SpiceResult<Self> {
        if tones.is_empty() {
            return Err(invalid("at least one tone is required"));
        }
        let mut relations = Vec::new();
        let mut root_frequencies = Vec::new();
        for (i, tone) in tones.iter().enumerate() {
            let relation = match *tone {
                Tone::Independent {
                    frequency_hz,
                    order,
                } => {
                    if !frequency_hz.is_finite() || frequency_hz <= 0. || order == 0 {
                        return Err(invalid(format!(
                            "tone {i} requires a finite positive frequency and nonzero order"
                        )));
                    }
                    root_frequencies.push((i, frequency_hz));
                    Relation {
                        root: i,
                        numerator: 1,
                        denominator: 1,
                        order,
                    }
                }
                Tone::Locked {
                    to,
                    numerator,
                    denominator,
                    order,
                } => {
                    if to >= i || numerator == 0 || denominator == 0 || order == 0 {
                        return Err(invalid(format!(
                            "tone {i} needs a nonzero rational ratio and an earlier reference"
                        )));
                    }
                    let parent: Relation = relations[to];
                    let up = parent
                        .numerator
                        .checked_mul(u64::from(numerator))
                        .ok_or_else(|| invalid("locked-tone numerator overflow"))?;
                    let down = parent
                        .denominator
                        .checked_mul(u64::from(denominator))
                        .ok_or_else(|| invalid("locked-tone denominator overflow"))?;
                    let common = gcd(up, down);
                    Relation {
                        root: parent.root,
                        numerator: up / common,
                        denominator: down / common,
                        order,
                    }
                }
            };
            relations.push(relation);
        }
        if root_frequencies.len() > limits.dimensions {
            return Err(invalid("number of independent tone axes exceeds budget"));
        }
        let mut axes = Vec::with_capacity(root_frequencies.len());
        let mut tones_hz = Vec::with_capacity(tones.len());
        let mut weights = vec![(0, 0); tones.len()];
        for &(root, source_hz) in &root_frequencies {
            let related: Vec<_> = relations
                .iter()
                .enumerate()
                .filter(|(_, relation)| relation.root == root)
                .collect();
            let denominator_lcm = related
                .iter()
                .try_fold(1_u64, |acc, (_, relation)| lcm(acc, relation.denominator))?;
            let mut scaled = Vec::with_capacity(related.len());
            for &(ordinal, relation) in &related {
                let weight = relation
                    .numerator
                    .checked_mul(denominator_lcm / relation.denominator)
                    .ok_or_else(|| invalid("locked-tone integer weight overflow"))?;
                scaled.push((ordinal, weight, relation.order));
            }
            let group_gcd = scaled
                .iter()
                .fold(0, |acc, &(_, weight, _)| gcd(acc, weight));
            let fundamental = (source_hz / denominator_lcm as Real) * group_gcd as Real;
            if !fundamental.is_finite() || fundamental <= 0. {
                return Err(invalid(
                    "locked-tone fundamental outside finite positive range",
                ));
            }
            let mut order = 0_usize;
            for &(ordinal, weight, requested) in &scaled {
                let reduced = usize::try_from(weight / group_gcd)
                    .map_err(|_| invalid("locked-tone weight exceeds platform usize"))?;
                order = order
                    .checked_add(checked_mul(reduced, usize::from(requested))?)
                    .ok_or_else(|| invalid("locked-tone harmonic order overflow"))?;
                let tone_hz = fundamental * reduced as Real;
                if !tone_hz.is_finite() || tone_hz <= 0. {
                    return Err(invalid("locked-tone physical frequency overflows"));
                }
                weights[ordinal] = (axes.len(), reduced);
            }
            let sample_axis = order
                .checked_mul(4)
                .and_then(|value| value.checked_add(1))
                .ok_or_else(|| invalid("oversampled axis count overflow"))?;
            axes.push(Axis {
                frequency_hz: fundamental,
                order,
                samples: sample_axis,
            });
        }
        let mut sample_count = 1;
        let mut mode_count = 1;
        for axis in &axes {
            sample_count = checked_mul(sample_count, axis.samples)?;
            mode_count = checked_mul(
                mode_count,
                axis.order
                    .checked_mul(2)
                    .and_then(|value| value.checked_add(1))
                    .ok_or_else(|| invalid("harmonic axis size overflow"))?,
            )?;
        }
        // Count the multiplications performed by the actual separable
        // transforms, not the old O(samples × modes) direct DFT. Both the
        // forward and inverse passes have different intermediate shapes.
        let mut forward_length = sample_count;
        let mut forward_products = 0_usize;
        for axis in &axes {
            let modes = 2 * axis.order + 1;
            forward_length = checked_mul(forward_length / axis.samples, modes)?;
            let stage = checked_mul(forward_length, axis.samples)?;
            forward_products = forward_products
                .checked_add(stage)
                .ok_or_else(|| invalid("forward transform work overflow"))?;
        }
        let mut inverse_length = mode_count;
        let mut inverse_products = 0_usize;
        for axis in axes.iter().rev() {
            let modes = 2 * axis.order + 1;
            inverse_length = checked_mul(inverse_length / modes, axis.samples)?;
            let stage = checked_mul(inverse_length, modes)?;
            inverse_products = inverse_products
                .checked_add(stage)
                .ok_or_else(|| invalid("inverse transform work overflow"))?;
        }
        debug_assert_eq!(forward_length, mode_count);
        debug_assert_eq!(inverse_length, sample_count);
        let transform_products = forward_products
            .checked_add(inverse_products)
            .ok_or_else(|| invalid("total separable transform work overflow"))?;
        if mode_count > limits.coefficients
            || sample_count > limits.samples
            || transform_products > limits.dft_products
        {
            return Err(invalid(format!(
                "lattice requires {mode_count} coefficients, {sample_count} samples, {transform_products} separable DFT products; budget is {}/{}/{}",
                limits.coefficients, limits.samples, limits.dft_products
            )));
        }
        for relation in &relations {
            let (axis, weight) = weights
                .get(relation.root)
                .copied()
                .ok_or_else(|| invalid("missing independent-tone index"))?;
            let factor = relation.numerator as Real / relation.denominator as Real;
            let f = root_frequencies
                .iter()
                .find(|(root, _)| *root == relation.root)
                .ok_or_else(|| invalid("missing independent-tone frequency"))?
                .1
                * factor;
            if !f.is_finite() || f <= 0. || !axes[axis].frequency_hz.is_finite() || weight == 0 {
                return Err(invalid("tone frequency outside finite positive range"));
            }
            tones_hz.push(f);
        }
        let mut modes = Vec::with_capacity(mode_count);
        for flat in 0..mode_count {
            let mut remaining = flat;
            let mut index = vec![0_i32; axes.len()];
            let mut conjugate = 0_usize;
            let mut stride = 1_usize;
            for axis_id in (0..axes.len()).rev() {
                let axis = &axes[axis_id];
                let size = 2 * axis.order + 1;
                let digit = remaining % size;
                remaining /= size;
                index[axis_id] = i32::try_from(digit)
                    .and_then(|digit| i32::try_from(axis.order).map(|order| digit - order))
                    .map_err(|_| invalid("harmonic index exceeds signed i32"))?;
                conjugate += (2 * axis.order - digit) * stride;
                stride *= size;
            }
            let frequency_hz: Real = axes
                .iter()
                .zip(&index)
                .map(|(axis, &harmonic)| axis.frequency_hz * Real::from(harmonic))
                .sum();
            if !frequency_hz.is_finite() {
                return Err(invalid("harmonic frequency overflows"));
            }
            modes.push(HarmonicPoint {
                index,
                frequency_hz,
                conjugate,
            });
        }

        // A user-specified independent basis must be nondegenerate at the
        // actual retained harmonic orders. Explicit locked relations avoid
        // double-counting aliased physical frequencies.
        let mut sorted: Vec<_> = modes
            .iter()
            .enumerate()
            .map(|(ordinal, mode)| (mode.frequency_hz, ordinal))
            .collect();
        sorted.sort_by(|a, b| a.0.total_cmp(&b.0));
        let worst_scale = axes
            .iter()
            .map(|axis| axis.frequency_hz * axis.order as Real)
            .sum::<Real>();
        for pair in sorted.windows(2) {
            if (pair[1].0 - pair[0].0).abs() <= 8. * Real::EPSILON * worst_scale {
                return Err(invalid(format!(
                    "independent tones alias harmonics {:?} and {:?}; encode exact rational relations as locked tones",
                    modes[pair[0].1].index, modes[pair[1].1].index
                )));
            }
        }
        let mut twiddles = Vec::with_capacity(axes.len());
        for axis in &axes {
            let width = 2 * axis.order + 1;
            let mut table = Vec::with_capacity(width * axis.samples);
            for coefficient in 0..width {
                let harmonic = coefficient as i32 - axis.order as i32;
                for sample in 0..axis.samples {
                    let angle = -TAU * Real::from(harmonic) * sample as Real / axis.samples as Real;
                    let (sin, cos) = angle.sin_cos();
                    table.push(Complex::new(cos, sin));
                }
            }
            twiddles.push(Arc::from(table));
        }
        Ok(Self {
            axes,
            modes,
            sample_count,
            transform_products,
            tones_hz,
            weights,
            twiddles,
        })
    }

    /// Independent frequency axes in deterministic source order.
    #[must_use]
    pub fn axes(&self) -> &[Axis] {
        &self.axes
    }

    /// All retained signed-frequency lattice coefficients.
    #[must_use]
    pub fn modes(&self) -> &[HarmonicPoint] {
        &self.modes
    }

    /// Number of real torus samples per residual evaluation.
    #[must_use]
    pub const fn sample_count(&self) -> usize {
        self.sample_count
    }

    /// Exact estimated complex products for one forward and inverse
    /// separable transform at a single MNA unknown.
    #[must_use]
    pub const fn transform_products(&self) -> usize {
        self.transform_products
    }

    /// Original sources' physical frequencies in declaration order.
    #[must_use]
    pub fn tone_frequencies(&self) -> &[Real] {
        &self.tones_hz
    }

    /// Axis and canonical integer weight of an original source tone.
    #[must_use]
    pub fn tone_weight(&self, source: usize) -> Option<(usize, usize)> {
        self.weights.get(source).copied()
    }

    /// Return the deterministic flat ordinal of a full harmonic tuple.
    ///
    /// This is a checked row-major index, not a frequency search: distinct
    /// physical harmonic slots cannot alias or silently select another tone.
    #[must_use]
    pub fn mode_ordinal(&self, coordinates: &[i32]) -> Option<usize> {
        if coordinates.len() != self.axes.len() {
            return None;
        }
        let mut offset = 0_usize;
        for (&harmonic, axis) in coordinates.iter().zip(&self.axes) {
            let order = i32::try_from(axis.order).ok()?;
            if harmonic < -order || harmonic > order {
                return None;
            }
            offset = offset
                .checked_mul(2 * axis.order + 1)?
                .checked_add(usize::try_from(harmonic + order).ok()?)?;
        }
        Some(offset)
    }

    /// Resolve a requested signed harmonic of one original source tone.
    ///
    /// Locked ratios return their single physical axis slot. If two source
    /// excitations map to the same frequency the caller must add their
    /// complex amplitudes, never overwrite one of them.
    #[must_use]
    pub fn tone_harmonic_ordinal(&self, source: usize, harmonic: i32) -> Option<usize> {
        let (axis, weight) = self.tone_weight(source)?;
        let scaled = i64::from(harmonic).checked_mul(i64::try_from(weight).ok()?)?;
        let mut index = vec![0_i32; self.axes.len()];
        index[axis] = i32::try_from(scaled).ok()?;
        self.mode_ordinal(&index)
    }

    /// Separable dense DFT of a single axis, preserving row-major strides.
    ///
    /// Each axis is transformed exactly once. Reusing the bound twiddles
    /// reduces trigonometric evaluations from O(samples*coefficients) per
    /// Newton/Jv call to zero, and work from a product to a sum of axis
    /// transforms. The arithmetic still projects every requested mode.
    fn transform_axis(
        &self,
        input: &[Complex],
        shape: &[usize],
        axis_id: usize,
        inverse: bool,
    ) -> Vec<Complex> {
        let axis = &self.axes[axis_id];
        let input_width = shape[axis_id];
        let output_width = if inverse {
            axis.samples
        } else {
            2 * axis.order + 1
        };
        let inner = shape[axis_id + 1..].iter().product::<usize>();
        let outer = shape[..axis_id].iter().product::<usize>();
        let mut output = vec![Complex::ZERO; outer * output_width * inner];
        let table = &self.twiddles[axis_id];
        for prefix in 0..outer {
            for suffix in 0..inner {
                for destination in 0..output_width {
                    let mut value = Complex::ZERO;
                    for source in 0..input_width {
                        let (mode, sample) = if inverse {
                            (source, destination)
                        } else {
                            (destination, source)
                        };
                        let twiddle = table[mode * axis.samples + sample];
                        let phase = if inverse { twiddle.conj() } else { twiddle };
                        value =
                            value + input[(prefix * input_width + source) * inner + suffix] * phase;
                    }
                    if !inverse {
                        value = value / Complex::real(axis.samples as Real);
                    }
                    output[(prefix * output_width + destination) * inner + suffix] = value;
                }
            }
        }
        output
    }

    /// Project real torus samples to all full signed-frequency coefficients.
    ///
    /// # Errors
    /// Incorrect shape or non-finite input.
    pub fn project(&self, samples: &[Real]) -> SpiceResult<Vec<Complex>> {
        if samples.len() != self.sample_count || samples.iter().any(|x| !x.is_finite()) {
            return Err(invalid(
                "nonfinite or incorrectly shaped real torus samples",
            ));
        }
        let mut data = samples
            .iter()
            .copied()
            .map(Complex::real)
            .collect::<Vec<_>>();
        let mut shape = self
            .axes
            .iter()
            .map(|axis| axis.samples)
            .collect::<Vec<_>>();
        for axis in 0..self.axes.len() {
            data = self.transform_axis(&data, &shape, axis, false);
            shape[axis] = 2 * self.axes[axis].order + 1;
        }
        Ok(data)
    }

    /// Synthesize real torus samples from conjugate-symmetric coefficients.
    ///
    /// # Errors
    /// Shape, nonfinite inputs or conjugate-symmetry violation.
    pub fn synthesize(&self, coefficients: &[Complex]) -> SpiceResult<Vec<Real>> {
        if coefficients.len() != self.modes.len()
            || coefficients
                .iter()
                .any(|coefficient| !coefficient.is_finite())
        {
            return Err(invalid("nonfinite or incorrectly shaped harmonic spectrum"));
        }
        let scale = coefficients
            .iter()
            .map(|coefficient| coefficient.magnitude())
            .fold(0.0_f64, Real::max);
        for (i, (&coefficient, mode)) in coefficients.iter().zip(&self.modes).enumerate() {
            let neg = coefficients[mode.conjugate];
            let tolerance = 128. * Real::EPSILON * scale.max(Real::MIN_POSITIVE);
            if (coefficient.re - neg.re).abs() > tolerance
                || (coefficient.im + neg.im).abs() > tolerance
                || (i == mode.conjugate && coefficient.im != 0.)
            {
                return Err(invalid(
                    "nonconjugate harmonic spectrum cannot represent real state",
                ));
            }
        }
        let mut data = coefficients.to_vec();
        let mut shape = self
            .axes
            .iter()
            .map(|axis| 2 * axis.order + 1)
            .collect::<Vec<_>>();
        for axis in (0..self.axes.len()).rev() {
            data = self.transform_axis(&data, &shape, axis, true);
            shape[axis] = self.axes[axis].samples;
        }
        let mut result = Vec::with_capacity(self.sample_count);
        for value in data {
            if !value.is_finite() {
                return Err(invalid("nonfinite synthesized real sample"));
            }
            result.push(value.re);
        }
        Ok(result)
    }
}
