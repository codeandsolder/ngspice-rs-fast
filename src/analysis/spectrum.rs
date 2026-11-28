//! Windowed spectra on a uniformly sampled time grid.

use std::f64::consts::TAU;

use crate::maths::fft::{Direction, FftPlan};
use crate::primitives::{Complex, Real, SpiceError, SpiceResult};

fn failure(message: impl Into<String>) -> SpiceError {
    SpiceError::Numerical {
        context: "spectrum".into(),
        message: message.into(),
    }
}

fn count(value: usize) -> SpiceResult<Real> {
    u32::try_from(value)
        .map(Real::from)
        .map_err(|_| failure("spectrum length exceeds index range"))
}

fn allocate<T: Default + Clone>(length: usize) -> SpiceResult<Vec<T>> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(length)
        .map_err(|error| failure(format!("allocating spectrum buffer: {error}")))?;
    values.resize(length, T::default());
    Ok(values)
}

/// Periodic windows, with the taper anchored at the final sample.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum SpectrumWindow {
    /// No taper.
    Rectangular,
    /// Hann (default).
    #[default]
    Hann,
    /// Hamming.
    Hamming,
    /// Bartlett/triangle.
    Triangle,
    /// Blackman.
    Blackman,
    /// Four-term Blackman-Harris.
    BlackmanHarris,
    /// Five-term flat top.
    FlatTop,
}

impl SpectrumWindow {
    /// Parse window names and common ngspice aliases.
    #[must_use = "the parsed window determines the spectral weighting"]
    pub fn parse(name: &str) -> Option<Self> {
        match name.to_ascii_lowercase().as_str() {
            "none" | "rectangular" => Some(Self::Rectangular),
            "hann" | "hanning" | "cosine" => Some(Self::Hann),
            "hamming" => Some(Self::Hamming),
            "triangle" | "bartlet" | "bartlett" => Some(Self::Triangle),
            "blackman" => Some(Self::Blackman),
            "blackmanharris" => Some(Self::BlackmanHarris),
            "flattop" => Some(Self::FlatTop),
            _ => None,
        }
    }

    fn weight(self, position: Real) -> Real {
        let angle = TAU * position;
        let c1 = angle.cos();
        match self {
            Self::Rectangular => 1.,
            Self::Hann => 0.5_f64.mul_add(-c1, 0.5),
            Self::Hamming => 0.46_f64.mul_add(-c1, 0.54),
            Self::Triangle => 2. * position.abs().min(1. - position.abs()),
            Self::Blackman => 0.08_f64.mul_add((2. * angle).cos(), 0.5_f64.mul_add(-c1, 0.42)),
            Self::BlackmanHarris => (-0.01168_f64).mul_add(
                (3. * angle).cos(),
                0.14128_f64.mul_add((2. * angle).cos(), 0.48829_f64.mul_add(-c1, 0.35875)),
            ),
            Self::FlatTop => {
                0.032_f64.mul_add(
                    (4. * angle).cos(),
                    (-0.388_f64).mul_add(
                        (3. * angle).cos(),
                        1.29_f64.mul_add((2. * angle).cos(), 1.93_f64.mul_add(-c1, 1.)),
                    ),
                ) / 4.64
            }
        }
    }
}

/// Frequency grid selection.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum SpectrumLength {
    /// Transform the actual sample count, even when odd or prime.
    #[default]
    Exact,
    /// Zero-pad to the next power of two (interpolates the frequency grid).
    Padded,
}

/// Output normalization.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum SpectrumNormalization {
    /// Complex peak amplitude, corrected for coherent window gain.
    #[default]
    Amplitude,
    /// Mean-square contribution of each bin, corrected for window energy.
    Power,
    /// Power per Hz (integrate over frequency to recover mean square).
    Density,
}

/// Select the spectrum of real or complex input signals.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpectrumLayout {
    /// Nonnegative frequencies; fold negative-frequency power/amplitude.
    Real,
    /// All frequencies in DFT order, including negative frequencies.
    Complex,
}

/// Spectrum transform settings.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SpectrumSettings {
    /// Sample window.
    pub window: SpectrumWindow,
    /// Transform grid.
    pub length: SpectrumLength,
    /// Output quantity.
    pub normalization: SpectrumNormalization,
}

/// Discrete window characteristics (independent of zero padding).
#[derive(Debug, Clone, Copy)]
pub struct WindowMetrics {
    /// Sum of weights divided by the actual sample count.
    pub coherent_gain: Real,
    /// Equivalent noise bandwidth in unpadded DFT bins.
    pub enbw_bins: Real,
    /// Equivalent noise bandwidth in Hz.
    pub enbw_hz: Real,
}

fn transform_length(samples: usize, length: SpectrumLength) -> SpiceResult<usize> {
    if samples < 2 {
        return Err(failure("spectrum needs at least two samples"));
    }
    match length {
        SpectrumLength::Exact => Ok(samples),
        SpectrumLength::Padded => samples
            .checked_next_power_of_two()
            .ok_or_else(|| failure("spectrum length overflow")),
    }
}

/// A reusable window and Fourier plan for a fixed sampling geometry.
pub struct SpectrumPlan {
    fft: FftPlan,
    weights: Vec<Real>,
    sum: Real,
    energy: Real,
    df: Real,
    dt: Real,
    layout: SpectrumLayout,
    normalization: SpectrumNormalization,
    metrics: WindowMetrics,
}

/// Caller-owned reusable spectrum buffers.
pub struct SpectrumWorkspace {
    values: Vec<Complex>,
    scratch: Vec<Complex>,
}

impl SpectrumPlan {
    /// Estimate work before allocating the window, plan or output.
    ///
    /// # Errors
    /// Invalid or unrepresentable transform length.
    pub fn work_estimate(
        samples: usize,
        length: SpectrumLength,
        signals: usize,
    ) -> SpiceResult<usize> {
        let size = transform_length(samples, length)?;
        Ok(FftPlan::work_estimate(size, signals)?
            .saturating_add(samples)
            .saturating_add(size.saturating_mul(signals)))
    }

    /// Build a spectrum plan, using the actual discrete window sums.
    ///
    /// # Errors
    /// Invalid dimensions, sample interval, degenerate window, work budget or allocation.
    pub fn new(
        samples: usize,
        dt: Real,
        layout: SpectrumLayout,
        settings: SpectrumSettings,
        work_limit: usize,
    ) -> SpiceResult<Self> {
        let size = transform_length(samples, settings.length)?;
        if Self::work_estimate(samples, settings.length, 1)? > work_limit {
            return Err(failure("spectrum work budget exceeded"));
        }
        let sample_count = count(samples)?;
        let df = (1. / count(size)?) / dt;
        if !dt.is_finite() || dt <= 0. || !df.is_finite() || df <= 0. {
            return Err(failure("invalid spectrum sample spacing"));
        }
        let mut weights = allocate::<Real>(samples)?;
        for (index, weight) in weights.iter_mut().enumerate() {
            *weight = settings
                .window
                .weight((count(index)? - (sample_count - 1.)) / sample_count);
        }
        let sum = weights.iter().sum::<Real>();
        let energy = weights.iter().map(|weight| weight * weight).sum::<Real>();
        let enbw_bins = (energy / sum) * (sample_count / sum);
        let enbw_hz = (enbw_bins / sample_count) / dt;
        if sum <= 0.
            || energy <= 0.
            || !enbw_bins.is_finite()
            || !enbw_hz.is_finite()
            || enbw_hz <= 0.
        {
            return Err(failure("invalid spectrum window normalization"));
        }
        Ok(Self {
            fft: FftPlan::new(size, work_limit)?,
            weights,
            sum,
            energy,
            df,
            dt,
            layout,
            normalization: settings.normalization,
            metrics: WindowMetrics {
                coherent_gain: sum / sample_count,
                enbw_bins,
                enbw_hz,
            },
        })
    }

    /// Discrete coherent gain and equivalent noise bandwidth.
    #[must_use = "the window metrics determine amplitude and bandwidth normalization"]
    pub const fn metrics(&self) -> WindowMetrics {
        self.metrics
    }

    /// Number of retained frequency bins.
    #[must_use = "the bin count determines the size of the resulting spectrum"]
    pub const fn bins(&self) -> usize {
        match self.layout {
            SpectrumLayout::Real => self.fft.length() / 2 + 1,
            SpectrumLayout::Complex => self.fft.length(),
        }
    }

    /// Frequency in Hz, with negative frequencies in the second half of a complex DFT.
    ///
    /// # Errors
    /// Out-of-range index or unrepresentable frequency.
    pub fn frequency(&self, index: usize) -> SpiceResult<Real> {
        if index >= self.bins() {
            return Err(failure("spectrum bin out of range"));
        }
        let size = self.fft.length();
        let frequency = if self.layout == SpectrumLayout::Complex && index > (size - 1) / 2 {
            -count(size - index)? * self.df
        } else {
            count(index)? * self.df
        };
        if !frequency.is_finite() {
            return Err(failure("spectrum frequency overflows"));
        }
        Ok(frequency)
    }

    /// Allocate buffers once for any number of signals.
    ///
    /// # Errors
    /// Allocation failure.
    pub fn workspace(&self) -> SpiceResult<SpectrumWorkspace> {
        Ok(SpectrumWorkspace {
            values: allocate(self.fft.length())?,
            scratch: self.fft.scratch()?,
        })
    }

    /// Compute a spectrum without allocation. Real-only plans reject complex samples.
    ///
    /// # Errors
    /// Wrong buffer dimensions, invalid input or unrepresentable output.
    pub fn transform(
        &self,
        input: &[Complex],
        workspace: &mut SpectrumWorkspace,
        output: &mut [Complex],
    ) -> SpiceResult<()> {
        if input.len() != self.weights.len()
            || workspace.values.len() != self.fft.length()
            || output.len() != self.bins()
        {
            return Err(failure("spectrum buffer dimensions do not match the plan"));
        }
        if input.iter().any(|value| {
            !value.is_finite() || (self.layout == SpectrumLayout::Real && !value.is_real())
        }) {
            return Err(failure("invalid spectrum samples"));
        }
        let maximum = input.iter().fold(0_f64, |maximum, value| {
            maximum.max(value.re.abs()).max(value.im.abs())
        });
        let size = count(self.fft.length())?;
        // Avoid overflowing an unnormalized DFT whose final normalized value
        // is representable. Keep ordinary samples unscaled.
        let scale = if maximum > Real::MAX / (4. * size)
            || (maximum > 0. && maximum < Real::MIN_POSITIVE * size)
        {
            maximum
        } else {
            1.
        };
        workspace.values.fill(Complex::ZERO);
        for ((value, sample), weight) in workspace.values.iter_mut().zip(input).zip(&self.weights) {
            *value = Complex::new((sample.re / scale) * weight, (sample.im / scale) * weight);
        }
        self.fft.transform(
            &mut workspace.values,
            &mut workspace.scratch,
            Direction::Forward,
        )?;
        for (index, (destination, value)) in output.iter_mut().zip(&workspace.values).enumerate() {
            let real_boundary = self.layout == SpectrumLayout::Real
                && (index == 0
                    || (self.fft.length().is_multiple_of(2) && index == self.fft.length() / 2));
            let fold = if self.layout == SpectrumLayout::Real && !real_boundary {
                2.
            } else {
                1.
            };
            let value = if real_boundary {
                Complex::real(value.re)
            } else {
                *value
            };
            *destination = match self.normalization {
                SpectrumNormalization::Amplitude => {
                    let factor = fold / self.sum;
                    Complex::new((value.re * factor) * scale, (value.im * factor) * scale)
                }
                SpectrumNormalization::Power | SpectrumNormalization::Density => {
                    let factor = (fold / self.energy).sqrt()
                        * if self.normalization == SpectrumNormalization::Power {
                            1. / size.sqrt()
                        } else {
                            self.dt.sqrt()
                        };
                    let re = (value.re * factor) * scale;
                    let im = (value.im * factor) * scale;
                    let amplitude = re.hypot(im);
                    Complex::real(amplitude * amplitude)
                }
            };
            if !destination.is_finite() {
                return Err(failure("spectrum output is not finite"));
            }
        }
        Ok(())
    }
}
