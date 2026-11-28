//! Reusable exact-length complex Fourier transforms.
//!
//! Power-of-two lengths use radix two directly; other lengths use Bluestein's
//! chirp convolution on a radix-two grid. The convolution padding is internal
//! and does not change the output frequencies or sample count.

use std::f64::consts::{PI, TAU};

use crate::primitives::{Complex, Real, SpiceError, SpiceResult};

fn failure(message: impl Into<String>) -> SpiceError {
    SpiceError::Numerical {
        context: "FFT".into(),
        message: message.into(),
    }
}

fn allocate(length: usize) -> SpiceResult<Vec<Complex>> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(length)
        .map_err(|error| failure(format!("allocating FFT buffer: {error}")))?;
    values.resize(length, Complex::ZERO);
    Ok(values)
}

fn real_count(value: usize) -> SpiceResult<Real> {
    u32::try_from(value)
        .map(Real::from)
        .map_err(|_| failure("FFT length exceeds exact index range"))
}

#[derive(Clone, Copy)]
struct Shape {
    length: usize,
    convolution: usize,
}

impl Shape {
    fn new(length: usize) -> SpiceResult<Self> {
        if length == 0 {
            return Err(failure("FFT needs at least one sample"));
        }
        let convolution = if length.is_power_of_two() {
            length
        } else {
            length
                .checked_mul(2)
                .and_then(|size| size.checked_sub(1))
                .and_then(usize::checked_next_power_of_two)
                .ok_or_else(|| failure("FFT convolution length overflow"))?
        };
        real_count(convolution)?;
        Ok(Self {
            length,
            convolution,
        })
    }

    const fn is_direct(self) -> bool {
        self.length == self.convolution
    }

    const fn work(self, transforms: usize) -> usize {
        let radix = self
            .convolution
            .saturating_mul(self.convolution.ilog2() as usize + 1);
        let (prepare, execute) = if self.is_direct() {
            (self.length, radix.saturating_add(self.length))
        } else {
            (
                radix.saturating_add(self.length),
                radix
                    .saturating_mul(2)
                    .saturating_add(self.convolution)
                    .saturating_add(self.length.saturating_mul(2)),
            )
        };
        prepare.saturating_add(execute.saturating_mul(transforms))
    }
}

struct Radix2 {
    size: usize,
    twiddles: Vec<Complex>,
}

impl Radix2 {
    fn new(size: usize) -> SpiceResult<Self> {
        let mut twiddles = allocate(size / 2)?;
        let denominator = real_count(size)?;
        for (index, twiddle) in twiddles.iter_mut().enumerate() {
            let (sin, cos) = (-TAU * real_count(index)? / denominator).sin_cos();
            *twiddle = Complex::new(cos, sin);
        }
        Ok(Self { size, twiddles })
    }

    fn forward(&self, values: &mut [Complex]) {
        if self.size == 1 {
            return;
        }
        let bits = self.size.ilog2();
        for index in 0..self.size {
            let reverse = index.reverse_bits() >> (usize::BITS - bits);
            if index < reverse {
                values.swap(index, reverse);
            }
        }
        for stage in 1..=bits {
            let width = 1usize << stage;
            let half = width / 2;
            let stride = self.size / width;
            for chunk in values.chunks_exact_mut(width) {
                let (left, right) = chunk.split_at_mut(half);
                for (index, (a, b)) in left.iter_mut().zip(right).enumerate() {
                    let rotated = *b * self.twiddles[index * stride];
                    let original = *a;
                    *a = original + rotated;
                    *b = original - rotated;
                }
            }
        }
    }
}

struct Bluestein {
    chirps: Vec<Complex>,
    kernel: Vec<Complex>,
    divisor: Real,
}

impl Bluestein {
    fn new(shape: Shape, radix: &Radix2) -> SpiceResult<Self> {
        let mut chirps = allocate(shape.length)?;
        let mut kernel = allocate(shape.convolution)?;
        let length =
            u32::try_from(shape.length).map_err(|_| failure("FFT chirp length overflow"))?;
        let period = 2 * u64::from(length);
        for (index, chirp) in (0..length).zip(&mut chirps) {
            // Reduce the integer phase before converting to floating point.
            // A direct k*k angle loses significant phase bits for long FFTs.
            let square = u64::from(index).pow(2) % period;
            let phase = u32::try_from(square).map_err(|_| failure("FFT chirp phase overflow"))?;
            let (sin, cos) = (-PI * Real::from(phase) / Real::from(length)).sin_cos();
            *chirp = Complex::new(cos, sin);
        }
        for (index, chirp) in chirps.iter().enumerate() {
            let value = chirp.conj();
            kernel[index] = value;
            if index != 0 {
                kernel[shape.convolution - index] = value;
            }
        }
        radix.forward(&mut kernel);
        Ok(Self {
            chirps,
            kernel,
            divisor: real_count(shape.convolution)?,
        })
    }

    fn forward(&self, radix: &Radix2, values: &mut [Complex], scratch: &mut [Complex]) {
        scratch.fill(Complex::ZERO);
        for ((destination, value), chirp) in scratch.iter_mut().zip(values.iter()).zip(&self.chirps)
        {
            *destination = *value * *chirp;
        }
        radix.forward(scratch);
        for (value, kernel) in scratch.iter_mut().zip(&self.kernel) {
            *value = (*value * *kernel).conj();
        }
        radix.forward(scratch);
        // The conjugated forward transform implements the inverse convolution
        // transform. Its radix length has already been checked in construction.
        let divisor = self.divisor;
        for ((destination, value), chirp) in values.iter_mut().zip(scratch).zip(&self.chirps) {
            *destination = Complex::new(value.re / divisor, -value.im / divisor) * *chirp;
        }
    }
}

/// Transform sign and normalization.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Direction {
    /// Negative-exponent DFT without normalization.
    Forward,
    /// Positive-exponent DFT divided by the exact transform length.
    Inverse,
}

/// A reusable transform plan for a nonempty real or complex sample sequence.
pub struct FftPlan {
    shape: Shape,
    radix: Radix2,
    bluestein: Option<Bluestein>,
}

impl FftPlan {
    /// Upper bound on work units for plan creation and the requested number of transforms.
    ///
    /// # Errors
    /// Zero length or an unrepresentable transform/convolution dimension.
    pub fn work_estimate(length: usize, transforms: usize) -> SpiceResult<usize> {
        Ok(Shape::new(length)?.work(transforms))
    }

    /// Construct a plan, checking the work allowance before allocating.
    ///
    /// # Errors
    /// Invalid dimension, insufficient work allowance or allocation failure.
    pub fn new(length: usize, work_limit: usize) -> SpiceResult<Self> {
        let shape = Shape::new(length)?;
        if shape.work(1) > work_limit {
            return Err(failure("FFT work budget exceeded"));
        }
        let radix = Radix2::new(shape.convolution)?;
        let bluestein = if shape.is_direct() {
            None
        } else {
            Some(Bluestein::new(shape, &radix)?)
        };
        Ok(Self {
            shape,
            radix,
            bluestein,
        })
    }

    /// Exact number of input/output samples.
    #[must_use = "the exact transform length is needed to size the input and output"]
    pub const fn length(&self) -> usize {
        self.shape.length
    }

    /// Number of complex scratch cells required for every execution.
    #[must_use = "the returned length is required to allocate sufficient execution scratch space"]
    pub const fn scratch_length(&self) -> usize {
        if self.shape.is_direct() {
            0
        } else {
            self.shape.convolution
        }
    }

    /// Allocate reusable execution scratch.
    ///
    /// # Errors
    /// Allocation failure.
    pub fn scratch(&self) -> SpiceResult<Vec<Complex>> {
        allocate(self.scratch_length())
    }

    /// Transform samples in place, without allocating.
    ///
    /// # Errors
    /// Incorrect buffer dimensions, nonfinite input or unrepresentable result.
    pub fn transform(
        &self,
        values: &mut [Complex],
        scratch: &mut [Complex],
        direction: Direction,
    ) -> SpiceResult<()> {
        if values.len() != self.length() || scratch.len() != self.scratch_length() {
            return Err(failure("FFT buffer dimensions do not match the plan"));
        }
        if values.iter().any(|value| !value.is_finite()) {
            return Err(failure("nonfinite FFT input"));
        }
        // Bound intermediate convolution growth, including the unnormalized
        // inverse radix transform. Scale only outside the ordinary range.
        let maximum = values.iter().fold(0_f64, |maximum, value| {
            maximum.max(value.re.abs()).max(value.im.abs())
        });
        let radix_size = real_count(self.shape.convolution)?;
        let growth = 4. * radix_size * radix_size * radix_size;
        let scale = if maximum > Real::MAX / growth
            || (maximum > 0. && maximum < Real::MIN_POSITIVE * radix_size)
        {
            maximum
        } else {
            1.
        };
        let inverse = direction == Direction::Inverse;
        for value in values.iter_mut() {
            *value = Complex::new(
                value.re / scale,
                if inverse {
                    -value.im / scale
                } else {
                    value.im / scale
                },
            );
        }
        if let Some(bluestein) = &self.bluestein {
            bluestein.forward(&self.radix, values, scratch);
        } else {
            self.radix.forward(values);
        }
        let divisor = if inverse {
            real_count(self.length())?
        } else {
            1.
        };
        for value in values {
            *value = Complex::new(
                (value.re / divisor) * scale,
                (if inverse { -value.im } else { value.im } / divisor) * scale,
            );
            if !value.is_finite() {
                return Err(failure("nonfinite FFT result"));
            }
        }
        Ok(())
    }
}
