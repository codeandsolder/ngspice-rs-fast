//! Prepared reciprocal RF line kernels. Geometry and material extraction is
//! separate from this SI-unit frequency-domain interface.
use crate::maths::dense_complex::DenseComplex;
use crate::primitives::{Complex, SpiceError, SpiceResult};
use std::f64::consts::TAU;

pub(crate) fn invalid(message: &str) -> SpiceError {
    SpiceError::circuit(format!("prepared RF network: {message}"))
}

fn nonnegative(x: f64) -> bool {
    x.is_finite() && x >= 0.
}

/// An inclusive validated frequency interval in hertz.
#[derive(Debug, Clone, Copy)]
pub struct FrequencyBand {
    low: f64,
    high: f64,
}

impl FrequencyBand {
    /// # Errors
    /// Invalid or reversed frequency limits.
    pub fn new(low: f64, high: f64) -> SpiceResult<Self> {
        if !nonnegative(low) || !nonnegative(high) || high < low {
            return Err(invalid("invalid frequency interval"));
        }
        Ok(Self { low, high })
    }

    /// Lower and upper frequency, in Hz.
    #[must_use]
    pub const fn limits(self) -> [f64; 2] {
        [self.low, self.high]
    }

    fn contains(self, hz: f64) -> bool {
        (self.low..=self.high).contains(&hz)
    }
}

/// SI-unit per-metre coefficients of an independently propagating mode.
/// The skin term is the positive-real impedance k sqrt(s/(2 pi)); the
/// dielectric relaxation is a passive Debye term s delta_c/(1+s tau).
#[derive(Debug, Clone, Copy)]
pub struct ModeParameters {
    /// DC resistance, ohm/m at the reference temperature.
    pub resistance: f64,
    /// External inductance, H/m.
    pub inductance: f64,
    /// DC dielectric conductance, S/m.
    pub conductance: f64,
    /// High-frequency capacitance, F/m.
    pub capacitance: f64,
    /// Skin-effect coefficient, ohm/(m sqrt(Hz)), real part at 1 Hz.
    pub skin: f64,
    /// Additional low-frequency capacitance, F/m.
    pub relaxation_capacitance: f64,
    /// Debye relaxation time in seconds.
    pub relaxation_time: f64,
    /// Reference temperature in degrees Celsius.
    pub reference_temperature: f64,
    /// Linear DC-resistivity temperature coefficient, 1/K.
    pub resistance_temperature_coefficient: f64,
}

/// Validated constitutive coefficients, independent of circuit topology.
#[derive(Debug, Clone, Copy)]
pub struct LineMode(ModeParameters);

impl LineMode {
    /// # Errors
    /// Nonpassive, nonfinite or undefined material coefficients.
    pub fn new(p: ModeParameters) -> SpiceResult<Self> {
        if ![
            p.resistance,
            p.inductance,
            p.conductance,
            p.capacitance,
            p.skin,
            p.relaxation_capacitance,
            p.relaxation_time,
        ]
        .into_iter()
        .all(nonnegative)
            || p.inductance == 0.
            || p.capacitance == 0.
            || (p.relaxation_capacitance > 0. && p.relaxation_time == 0.)
            || !p.reference_temperature.is_finite()
            || p.reference_temperature <= -273.15
            || !p.resistance_temperature_coefficient.is_finite()
        {
            return Err(invalid("invalid passive RLGC mode"));
        }
        Ok(Self(p))
    }

    /// The immutable validated coefficients.
    #[must_use]
    pub const fn parameters(self) -> ModeParameters {
        self.0
    }

    /// Series impedance and shunt admittance per metre at frequency and
    /// temperature. The skin impedance has equal real/imaginary parts.
    /// # Errors
    /// Unrepresentable frequency, temperature or evaluated coefficients.
    pub fn rlgc(self, hz: f64, temperature: f64) -> SpiceResult<(Complex, Complex)> {
        let p = self.0;
        let resistivity = p
            .resistance_temperature_coefficient
            .mul_add(temperature - p.reference_temperature, 1.);
        if !nonnegative(hz)
            || !temperature.is_finite()
            || temperature <= -273.15
            || !resistivity.is_finite()
            || resistivity <= 0.
        {
            return Err(invalid("frequency or temperature outside material domain"));
        }
        // Multiplying omega by small material coefficients before forming
        // omega itself avoids an unnecessary overflow at large frequency.
        let skin = p.skin * hz.sqrt() * resistivity.sqrt();
        let series = Complex::new(
            p.resistance.mul_add(resistivity, skin),
            (p.inductance * hz).mul_add(TAU, skin),
        );
        let omega_c = (p.capacitance * hz) * TAU;
        let relaxation = if p.relaxation_capacitance == 0. {
            Complex::ZERO
        } else {
            let x = (p.relaxation_time * hz) * TAU;
            // At large omega*tau use the reciprocal form; the real part
            // tends to delta_c/tau, not infinity/infinity.
            if x > 1. {
                let inv = x.recip();
                let g = p.relaxation_capacitance / p.relaxation_time / inv.mul_add(inv, 1.);
                Complex::new(g, g * inv)
            } else {
                let b = (p.relaxation_capacitance * hz) * TAU / x.mul_add(x, 1.);
                Complex::new(b * x, b)
            }
        };
        let shunt = Complex::new(p.conductance + relaxation.re, omega_c + relaxation.im);
        if !series.is_finite() || !shunt.is_finite() {
            return Err(invalid("unrepresentable per-metre RLGC"));
        }
        Ok((series, shunt))
    }
}

fn scaled(x: Complex, scale: f64) -> Complex {
    Complex::new(x.re * scale, x.im * scale)
}

fn sqrt_magnitude(x: Complex) -> f64 {
    let hi = x.re.abs().max(x.im.abs());
    let lo = x.re.abs().min(x.im.abs());
    if hi == 0. {
        0.
    } else {
        hi.sqrt() * (lo / hi).hypot(1.).sqrt()
    }
}

fn roots_product(z: Complex, y: Complex) -> Complex {
    if z == Complex::ZERO || y == Complex::ZERO {
        return Complex::ZERO;
    }
    let magnitude = sqrt_magnitude(z) * sqrt_magnitude(y);
    // Use the complementary phase near the lossless axis, so an
    // arbitrarily small resistive part does not turn into cos(pi/2).
    if z.im == 0. && y.im == 0. {
        Complex::real(magnitude)
    } else if z.re == 0. && y.re == 0. {
        Complex::imaginary(magnitude)
    } else if z.re >= z.im && y.re >= y.im {
        let (sin, cos) = (0.5 * (z.phase() + y.phase())).sin_cos();
        Complex::new(magnitude * cos, magnitude * sin)
    } else {
        let (sin, cos) = (0.5 * (z.re.atan2(z.im) + y.re.atan2(y.im))).sin_cos();
        Complex::new(magnitude * sin, magnitude * cos)
    }
}

/// Both (1+exp(-x))/2 and 1-exp(-x), without subtracting nearly
/// equal real parts near a lossless resonance or the origin.
fn exponential_sums(x: Complex) -> (Complex, Complex) {
    let amplitude = (-x.re).exp();
    let lost_amplitude = -(-x.re).exp_m1();
    let (sin, cos) = (0.5 * x.im).sin_cos();
    let imaginary = amplitude * sin * cos;
    let plus = Complex::new(
        (amplitude * cos).mul_add(cos, 0.5 * lost_amplitude),
        -imaginary,
    );
    let minus = Complex::new(
        (2. * amplitude * sin).mul_add(sin, lost_amplitude),
        2. * imaginary,
    );
    (plus, minus)
}

/// (1-exp(-x))/x, including its analytic limit at zero.
fn exponential_quotient(x: Complex, difference: Complex) -> Complex {
    if x.magnitude() < 0.125 {
        let mut term = Complex::real(1.);
        let mut sum = term;
        for n in 2..=13 {
            term = scaled(term * x, -1. / f64::from(n));
            sum = sum + term;
        }
        sum
    } else {
        difference / x
    }
}

fn normalize_pair(a: Complex, b: Complex) -> SpiceResult<(Complex, Complex)> {
    let factor = a.re.abs().max(a.im.abs()).max(b.re.abs()).max(b.im.abs());
    if factor == 0. || !factor.is_finite() {
        return Err(invalid("unrepresentable modal equation"));
    }
    let divide = |x: Complex| Complex::new(x.re / factor, x.im / factor);
    Ok((divide(a), divide(b)))
}

/// Bounded even/odd modal equations. With currents into both ends:
/// common_current * (I1+I2) = common_voltage * (V1+V2);
/// differential_voltage * (V1-V2) = differential_current * (I1-I2).
#[derive(Debug, Clone, Copy)]
pub struct ModalRelation {
    /// Common-mode current coefficient.
    pub common_current: Complex,
    /// Common-mode voltage coefficient.
    pub common_voltage: Complex,
    /// Differential-mode voltage coefficient.
    pub differential_voltage: Complex,
    /// Differential-mode current coefficient.
    pub differential_current: Complex,
}

impl ModalRelation {
    fn new(z: Complex, y: Complex, length: f64) -> SpiceResult<Self> {
        let z = scaled(z, length);
        let y = scaled(y, length);
        let x = roots_product(z, y);
        if !z.is_finite() || !y.is_finite() || !x.is_finite() {
            return Err(invalid("unrepresentable total line coefficients"));
        }
        let (c, difference) = exponential_sums(x);
        let q = exponential_quotient(x, difference);
        let common_voltage = scaled(y, 0.5) * q;
        let differential_current = scaled(z, 0.5) * q;
        if !common_voltage.is_finite() || !differential_current.is_finite() {
            return Err(invalid("unrepresentable line relation"));
        }
        let (common_current, common_voltage) = normalize_pair(c, common_voltage)?;
        let (differential_voltage, differential_current) = normalize_pair(c, differential_current)?;
        Ok(Self {
            common_current,
            common_voltage,
            differential_voltage,
            differential_current,
        })
    }

    /// Even and odd power-wave reflection eigenvalues for a common real
    /// reference impedance. Unlike conversion through Y, this supports
    /// ideal shorts, through connections and lossless resonances.
    /// # Errors
    /// Invalid reference or singular/nonfinite wave equations.
    pub fn reflections(self, reference: f64) -> SpiceResult<[Complex; 2]> {
        if !reference.is_finite() || reference <= 0. {
            return Err(invalid("invalid wave reference"));
        }
        let common = scaled(self.common_voltage, reference);
        let differential = scaled(self.differential_voltage, reference);
        let even = (self.common_current - common) / (self.common_current + common);
        let odd =
            (self.differential_current - differential) / (self.differential_current + differential);
        if !even.is_finite() || !odd.is_finite() {
            return Err(invalid("singular wave relation"));
        }
        Ok([even, odd])
    }
}

/// A line with independently propagating modes and an orthogonal real
/// conductor-to-mode transform. Physical voltages and currents use the
/// same transform, preserving power. General noncommuting RLGC matrices
/// require a different prepared kernel, not an invented modal fit.
#[derive(Debug, Clone)]
pub struct PreparedLine {
    modes: Box<[LineMode]>,
    basis: Box<[f64]>,
    length: f64,
    band: FrequencyBand,
}

impl PreparedLine {
    /// Prepare a single-conductor line.
    /// # Errors
    /// Invalid length.
    pub fn single(mode: LineMode, length: f64, band: FrequencyBand) -> SpiceResult<Self> {
        Self::modal(vec![mode], vec![1.], length, band)
    }

    /// Prepare an orthogonal modal basis, row-major (mode, conductor).
    /// # Errors
    /// Invalid dimension, length or nonorthogonal basis.
    pub fn modal(
        modes: Vec<LineMode>,
        basis: Vec<f64>,
        length: f64,
        band: FrequencyBand,
    ) -> SpiceResult<Self> {
        let n = modes.len();
        if n == 0
            || n > 16
            || basis.len() != n * n
            || !nonnegative(length)
            || !basis.iter().all(|x| x.is_finite())
        {
            return Err(invalid("invalid modal line dimensions or length"));
        }
        for (i, a) in basis.chunks_exact(n).enumerate() {
            for (j, b) in basis.chunks_exact(n).enumerate() {
                let dot = a.iter().zip(b).map(|(x, y)| x * y).sum::<f64>();
                let expected = if i == j { 1. } else { 0. };
                if !dot.is_finite() || (dot - expected).abs() > 64. * f64::EPSILON {
                    return Err(invalid("modal basis must preserve power"));
                }
            }
        }
        Ok(Self {
            modes: modes.into_boxed_slice(),
            basis: basis.into_boxed_slice(),
            length,
            band,
        })
    }

    /// Number of conductors (ports are ordered near end, then far end).
    #[must_use]
    pub fn conductors(&self) -> usize {
        self.modes.len()
    }

    /// Physical length in metres.
    #[must_use]
    pub const fn length(&self) -> f64 {
        self.length
    }

    /// Validity range of the prepared model.
    #[must_use]
    pub const fn band(&self) -> FrequencyBand {
        self.band
    }

    /// Borrow the immutable mode coefficients.
    #[must_use]
    pub fn modes(&self) -> &[LineMode] {
        &self.modes
    }

    /// Row-major, orthogonal conductor-to-mode transform.
    #[must_use]
    pub fn basis(&self) -> &[f64] {
        &self.basis
    }

    /// Evaluate all independent line relations at one analysis point.
    /// DC uses the explicitly supplied static coefficients even for a
    /// microwave-only validity band, so the bias system is well defined.
    /// # Errors
    /// Invalid frequency or temperature, outside band, or numerical range.
    pub fn relations(&self, hz: f64, temperature: f64) -> SpiceResult<Vec<ModalRelation>> {
        if hz != 0. && !self.band.contains(hz) {
            return Err(invalid("frequency outside prepared validity band"));
        }
        self.modes
            .iter()
            .map(|m| {
                let (z, y) = m.rlgc(hz, temperature)?;
                ModalRelation::new(z, y, self.length)
            })
            .collect()
    }

    /// Power-wave scattering matrix at equal real reference impedances.
    /// # Errors
    /// Invalid reference, frequency, temperature, or singular relation.
    pub fn scattering(
        &self,
        hz: f64,
        temperature: f64,
        reference: f64,
    ) -> SpiceResult<DenseComplex> {
        let n = self.conductors();
        let values = self
            .relations(hz, temperature)?
            .into_iter()
            .map(|r| r.reflections(reference))
            .collect::<SpiceResult<Vec<_>>>()?;
        let mut matrix = DenseComplex::zeros(2 * n)?;
        for (basis, [even, odd]) in self.basis.chunks_exact(n).zip(values) {
            let reflect = scaled(even + odd, 0.5);
            let through = scaled(even - odd, 0.5);
            for (r, &a) in basis.iter().enumerate() {
                for (c, &b) in basis.iter().enumerate() {
                    for (row, col, v) in [
                        (r, c, reflect),
                        (r + n, c + n, reflect),
                        (r, c + n, through),
                        (r + n, c, through),
                    ] {
                        let old = matrix
                            .get(row, col)
                            .ok_or_else(|| invalid("invalid scattering index"))?;
                        matrix.set(row, col, old + scaled(v, a * b))?;
                    }
                }
            }
        }
        Ok(matrix)
    }
}
