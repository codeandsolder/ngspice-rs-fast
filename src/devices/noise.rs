//! Device noise generators for small-signal `.noise` analysis.
//!
//! C reference: every noisy device has a `DEVnoise` entry (`resnoise.c`,
//! `dionoise.c`, `bjtnoise.c`, `mesnoise.c`, `mos1noi.c`, `swnoise.c`, `cswnoise.c`) that
//! names its generators and evaluates them through `NevalSrc`/
//! `NevalSrcInstanceTemp` (`src/spicelib/analysis/nevalsrc.c`): every generator
//! is a current source between two nodes whose power spectral density is
//!
//! | kind | density (A²/Hz) | C |
//! | --- | --- | --- |
//! | thermal | `4 k T g` | `THERMNOISE` |
//! | shot | `2 q abs(I)` | `SHOTNOISE` |
//! | flicker | `numerator` over the device's frequency law ([`FlickerLaw`]) | `N_GAIN` times the device's own law |
//!
//! and the output noise it causes is that density times the squared magnitude
//! of the transfer impedance from the generator's node pair to the output, which
//! the analysis obtains from one adjoint solve per frequency
//! (`crate::analysis` `.noise`, C `NInzIter`).
//!
//! The port splits C's single `DEVnoise` callback in two:
//!
//! * a device describes its generators **at the bias point** through
//!   [`crate::devices::Device::noise`], returning a [`DeviceNoise`] description and appending generators to caller-owned
//!   storage. The model context includes frequency; adjoint vectors and
//!   integration state remain analysis-owned. A device
//!   that is noiseless in C (`DEVnoise = NULL`: C, L, V, I, E/F/G/H, K, B)
//!   returns [`DeviceNoise::Noiseless`]; the trait default is an explicit error,
//!   so a device whose noise is not ported can never be silently omitted;
//! * the analysis owns gains, densities, the `Nintegrate` integration history
//!   and the output naming (`onoise_<inst><suffix>`), and orders the instances
//!   as C's `CKTnoise` visits them ([`NoisePlan`]).
//!
//! The same pattern (a pure bias-point description returned by a default
//! erroring trait method, evaluated by the analysis) is meant for later
//! small-signal analyses with per-device hooks.

use crate::devices::{MnaUnknowns, ModelContext, SourceKind};
use crate::maths::Vector;
use crate::primitives::{Complex, NodeId, Real, SpiceError, SpiceResult};

mod layout;
pub(crate) use layout::BoundTransfer;
pub use layout::{BoundNoiseSource, NoiseInstance, NoisePlan};

/// ngspice `CONSTboltz` (J/K).
pub const BOLTZMANN: Real = 1.380_648_52e-23;
/// ngspice `CHARGE` (C).
pub const CHARGE: Real = 1.602_176_620_8e-19;
/// ngspice `CONSTCtoK`.
pub const CELSIUS_TO_KELVIN: Real = 273.15;

/// The noisy C device types, in the order of C's `DEVices[]` table
/// (`src/spicelib/devices/dev.c`), which is the order `CKTnoise` visits them
/// and therefore the order of the per-device output columns.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum NoiseFamily {
    /// `bjt` (`bjtnoise.c`).
    Bjt,
    /// `bsim3` (`b3noi.c`).
    Bsim3,
    /// `bsim4` (`b4noi.c`).
    Bsim4,
    /// `csw`, the W switch (`cswnoise.c`).
    CurrentSwitch,
    /// `dio` (`dionoise.c`).
    Diode,
    /// `mes`, the level-1 MESFET (`mesnoise.c`).
    Mes,
    /// `mos1` (`mos1noi.c`).
    Mos1,
    /// `mos2` (`mos2noi.c`).
    Mos2,
    /// `mos3` (`mos3noi.c`).
    Mos3,
    /// `mos9` (`mos9noi.c`).
    Mos9,
    /// `res` (`resnoise.c`).
    Resistor,
    /// `sw`, the S switch (`swnoise.c`).
    VoltageSwitch,
}

/// The physical law of one noise generator, evaluated at the bias point.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum NoiseKind {
    /// Thermal noise of a conductance: `4 k T g` (C `THERMNOISE`).
    Thermal {
        /// Conductance in siemens.
        conductance: Real,
        /// Noise temperature in kelvin: the instance temperature for
        /// `NevalSrcInstanceTemp`, the circuit temperature for `NevalSrc`.
        temperature: Real,
    },
    /// Shot noise of a DC current: `2 q abs(I)` (C `SHOTNOISE`).
    Shot {
        /// DC current in amperes.
        current: Real,
    },
    /// Flicker noise: `numerator` times the device's frequency law (C
    /// `N_GAIN` times the device's own 1/f expression).
    Flicker {
        /// The bias-dependent part of the device's flicker expression, the
        /// operand C evaluates before its frequency division.
        numerator: Real,
        /// How the device divides by frequency, in C's operation order.
        law: FlickerLaw,
    },
    /// `N_GAIN` times a factor the device computes at the analysis frequency
    /// (`b4noi.c`'s flicker noise: `noizDens = gain; noizDens *= factor`).
    /// Only devices that re-describe their noise at every frequency
    /// ([`crate::devices::Device::small_signal_depends_on_frequency`]) may
    /// use it.
    Gain {
        /// The factor C multiplies the gain by.
        factor: Real,
    },
    /// Two fully correlated thermal generators, C's `NevalSrc2(...,
    /// THERMNOISE, ...)` (`nevalsrc.c`, BSIM4 `tnoiMod=2`): the first between
    /// the source's own nodes with `conductance`, the second between
    /// `second` with `second_conductance`, shifted by `phase` radians.
    CorrelatedThermal {
        /// C's `param1`.
        conductance: Real,
        /// C's `node3`, `node4`.
        second: [NodeId; 2],
        /// C's `param2`.
        second_conductance: Real,
        /// C's `phi21`.
        phase: Real,
        /// Noise temperature in kelvin (C `CKTtemp`).
        temperature: Real,
    },
}

/// The frequency part of a device's flicker density, kept apart from the
/// numerator so the density is evaluated with exactly C's operand grouping
/// (`gain *= numerator / denominator(f)`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FlickerLaw {
    /// `resnoise.c`: `numerator / (area * pow(f, ef))`.
    Area {
        /// C `RESeffNoiseArea`.
        area: Real,
        /// C `RESef`.
        exponent: Real,
    },
    /// `bjtnoise.c`: `numerator / f`.
    PerHertz,
    /// `dionoise.c`: `numerator / f * m`.
    PerHertzTimes {
        /// C `DIOm`.
        multiplier: Real,
    },
    /// `mos1noi.c`: `numerator / (((fx * a) * b) * c)` with `fx = f`, or
    /// `pow(f, AF)` for NLEV 2 and 3.
    Channel {
        /// `Some(AF)` for C's `pow(data->freq, AF)`, `None` for plain `f`.
        exponent: Option<Real>,
        /// The three denominator factors after the frequency term.
        factors: [Real; 3],
    },
}

impl FlickerLaw {
    /// `numerator` over this law's denominator at `frequency`, grouped as C.
    #[must_use]
    pub fn apply(&self, numerator: Real, frequency: Real) -> Real {
        match *self {
            Self::Area { area, exponent } => numerator / (area * frequency.powf(exponent)),
            Self::PerHertz => numerator / frequency,
            Self::PerHertzTimes { multiplier } => numerator / frequency * multiplier,
            Self::Channel { exponent, factors } => {
                let f = exponent.map_or(frequency, |e| frequency.powf(e));
                numerator / (f * factors[0] * factors[1] * factors[2])
            }
        }
    }

    fn validate(&self) -> bool {
        match *self {
            Self::Area { area, exponent } => area.is_finite() && exponent.is_finite(),
            Self::PerHertz => true,
            Self::PerHertzTimes { multiplier } => multiplier.is_finite(),
            Self::Channel { exponent, factors } => {
                exponent.is_none_or(Real::is_finite) && factors.iter().all(|f| f.is_finite())
            }
        }
    }
}

impl NoiseKind {
    /// Evaluate a scalar generator at a squared transfer magnitude.
    ///
    /// Arithmetic follows `NevalSrc`, including signed scalar densities.
    ///
    /// # Errors
    ///
    /// A coherent generator requires two complex transfers. Invalid inputs
    /// or nonfinite densities also return an error.
    fn scalar_output_density(&self, gain: Real, frequency: Real) -> SpiceResult<Real> {
        if !gain.is_finite() || gain < 0. || !valid_frequency(frequency) || !self.validate() {
            return Err(noise_error("invalid scalar generator, gain or frequency"));
        }
        let density = match *self {
            Self::Thermal {
                conductance,
                temperature,
            } => gain * 4.0 * BOLTZMANN * temperature * conductance,
            Self::Shot { current } => gain * 2.0 * CHARGE * current.abs(),
            // C: `noizDens = gain; noizDens *= <numerator> / <denominator>`.
            Self::Flicker { numerator, law } => gain * law.apply(numerator, frequency),
            Self::Gain { factor } => gain * factor,
            Self::CorrelatedThermal { .. } => {
                return Err(SpiceError::circuit(
                    "coherent thermal noise requires two complex transfers",
                ));
            }
        };
        finite_density(density)
    }

    /// RFSPICE's independent-source amplitude before port transformation.
    ///
    /// Unlike the adjoint density evaluator, this omits `N_GAIN` generators
    /// and does not multiply by the squared output transfer.
    ///
    /// # Errors
    ///
    /// Invalid parameters, a nonfinite or negative density, or a coherent
    /// generator (the distinct RFSPICE `NevalSrc2` path is not yet ported).
    pub(crate) fn rf_amplitude(&self) -> SpiceResult<Option<Real>> {
        if !self.validate() {
            return Err(noise_error("invalid RFSPICE generator"));
        }
        let density = match *self {
            Self::Thermal {
                conductance,
                temperature,
            } => 4. * BOLTZMANN * temperature * conductance,
            Self::Shot { current } => 2. * CHARGE * current.abs(),
            Self::Flicker { .. } | Self::Gain { .. } => return Ok(None),
            // Do not substitute the ordinary NevalSrc2 expression: the RF
            // branch builds complex port amplitudes and the C implementation
            // even uses param1 for both amplitude scales.
            Self::CorrelatedThermal { .. } => {
                return Err(SpiceError::not_yet_ported(
                    "correlated (NevalSrc2) noise in S-parameter noise analysis",
                    "src/spicelib/analysis/nevalsrc.c (NevalSrc2, RFSPICE branch)",
                ));
            }
        };
        finite_density(density.sqrt()).map(Some)
    }

    fn validate(&self) -> bool {
        match *self {
            Self::Thermal {
                conductance,
                temperature,
            } => conductance.is_finite() && temperature.is_finite() && temperature > 0.0,
            Self::Shot { current } => current.is_finite(),
            Self::Flicker { numerator, law } => numerator.is_finite() && law.validate(),
            Self::Gain { factor } => factor.is_finite(),
            Self::CorrelatedThermal {
                conductance,
                second_conductance,
                phase,
                temperature,
                ..
            } => {
                conductance.is_finite()
                    && conductance >= 0.
                    && second_conductance.is_finite()
                    && second_conductance >= 0.
                    && phase.is_finite()
                    && temperature.is_finite()
                    && temperature > 0.0
            }
        }
    }
}

/// One noise generator: a current source between `nodes` (C's `node1`,
/// `node2` of `NevalSrc`), named by `suffix` after the instance name
/// (`onoise_<instance><suffix>`, e.g. `_thermal`, `_rb`, `_1overf`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NoiseSource {
    /// C's generator name suffix, including the leading underscore.
    pub suffix: &'static str,
    /// The first node pair, in reference order (sign matters for coherent noise).
    pub nodes: [NodeId; 2],
    /// The generator's law at the bias point.
    pub kind: NoiseKind,
}

const fn valid_frequency(frequency: Real) -> bool {
    frequency.is_finite() && frequency > 0.
}

fn noise_error(message: &str) -> SpiceError {
    SpiceError::Numerical {
        context: "noise transfer".into(),
        message: message.into(),
    }
}

fn finite_density(density: Real) -> SpiceResult<Real> {
    if density.is_finite() {
        Ok(density)
    } else {
        Err(noise_error("nonfinite noise density"))
    }
}

/// Squared transfer gain in the reference noise evaluator's operation order.
pub(crate) const fn squared_gain(transfer: Complex) -> Real {
    transfer.re * transfer.re + transfer.im * transfer.im
}

/// Coherent sum of both thermal generators, following `NevalSrc2`.
#[expect(
    clippy::suboptimal_flops,
    reason = "NevalSrc2 adds the first generator before the second generator's imaginary contribution"
)]
fn correlated_gain(
    first: Complex,
    second: Complex,
    conductance: Real,
    second_conductance: Real,
    phase: Real,
) -> Real {
    let first_scale = conductance.sqrt();
    let second_scale = second_conductance.sqrt();
    let in_phase = second_scale * phase.cos();
    let quadrature = second_scale * phase.sin();
    let real = first_scale * first.re + in_phase * second.re - quadrature * second.im;
    let imag = first_scale * first.im + in_phase * second.im + quadrature * second.re;
    squared_gain(Complex::new(real, imag))
}

impl NoiseSource {
    /// A generator.
    #[must_use]
    pub const fn new(suffix: &'static str, nodes: [NodeId; 2], kind: NoiseKind) -> Self {
        Self {
            suffix,
            nodes,
            kind,
        }
    }
}

impl NoiseKind {
    // The caller supplies the already bound primary and coherent transfers.
    fn output_density(
        &self,
        primary: Complex,
        secondary: Option<Complex>,
        frequency: Real,
    ) -> SpiceResult<Real> {
        if let Self::CorrelatedThermal {
            conductance,
            second_conductance,
            phase,
            temperature,
            ..
        } = *self
        {
            if !valid_frequency(frequency) || !self.validate() {
                return Err(noise_error("invalid coherent generator or frequency"));
            }
            let second = secondary.ok_or_else(|| noise_error("missing coherent transfer"))?;
            let gain = correlated_gain(primary, second, conductance, second_conductance, phase);
            return finite_density(4.0 * BOLTZMANN * temperature * gain);
        }
        self.scalar_output_density(squared_gain(primary), frequency)
    }
}

/// Borrowed identity of a device's contribution to `.noise`.
/// The device appends its ordered generators to the caller's scratch buffer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DeviceNoise<'a> {
    /// No generators and no output columns: C's `DEVnoise = NULL` devices, and
    /// a resistor with `noisy=0` (`resnoise.c` skips it entirely).
    Noiseless,
    /// Generators in C's output order, optionally followed by the instance
    /// total the analysis computes (C's empty suffix).
    Sources {
        /// The C device type, for C's visiting order.
        family: NoiseFamily,
        /// The instance's model name, `None` for C's default model of the
        /// type (a literal resistor). C visits the models of one type in
        /// reverse order of their creation and the instances of one model in
        /// reverse order of theirs ([`NoisePlan`]).
        model: Option<&'a str>,
        /// Whether C reports an instance total (empty suffix) after the
        /// generators: every multi-generator device does; the single-generator
        /// S/W switches name their generator with the empty suffix instead.
        total: bool,
    },
}

/// What [`crate::devices::Device::noise`] sees: the converged operating point
/// the small-signal analysis linearizes around.
#[derive(Debug, Clone, Copy)]
pub struct NoiseContext<'a> {
    /// Circuit/nominal temperature and junction `gmin`.
    pub model_context: &'a ModelContext,
    /// Node numbering, with ground eliminated.
    pub unknowns: &'a MnaUnknowns,
    /// The operating-point solution.
    pub bias: &'a Vector,
    /// This device's state slots at the small-signal load (C `CKTstate0` at
    /// `MODEINITSMSIG`), when the analysis supplies them.
    pub states: Option<&'a [Real]>,
    /// Branch rows of the devices named by
    /// [`crate::devices::Device::controlling_sources`].
    pub controls: &'a [usize],
}

impl NoiseContext<'_> {
    /// The operating-point voltage of `node`, zero at ground.
    #[must_use]
    pub fn voltage(&self, node: NodeId) -> Real {
        self.unknowns
            .node_row(node)
            .and_then(|row| self.bias.get(row))
            .unwrap_or(0.0)
    }

    /// The circuit temperature in kelvin (C `CKTtemp`).
    #[must_use]
    pub fn circuit_kelvin(&self) -> Real {
        self.model_context.temperature + CELSIUS_TO_KELVIN
    }
}

/// An independent source as a small-signal input reference (the `.noise`
/// input source; C checks `VSRCacGiven`/`ISRCacGiven`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InputSource {
    /// Voltage or current source.
    pub kind: SourceKind,
    /// Whether the card gave an `ac` value (even `ac 0`).
    pub ac_given: bool,
}

#[cfg(test)]
mod tests {
    use super::{
        BOLTZMANN, CHARGE, FlickerLaw, NoiseKind, NoiseSource, correlated_gain, squared_gain,
    };
    use crate::primitives::{Complex, NodeId, SpiceError};

    #[test]
    fn densities_follow_nevalsrc() {
        let thermal = NoiseKind::Thermal {
            conductance: 1e-3,
            temperature: 300.15,
        };
        assert_eq!(
            thermal.scalar_output_density(2.0, 10.0).unwrap(),
            2.0 * 4.0 * BOLTZMANN * 300.15 * 1e-3
        );
        let shot = NoiseKind::Shot { current: -1e-3 };
        assert_eq!(
            shot.scalar_output_density(1.0, 1.0).unwrap(),
            2.0 * CHARGE * 1e-3
        );
        let flicker = NoiseKind::Flicker {
            numerator: 1e-12,
            law: FlickerLaw::PerHertz,
        };
        assert_eq!(flicker.scalar_output_density(1.0, 100.0).unwrap(), 1e-14);
        let steep = NoiseKind::Flicker {
            numerator: 1e-12,
            law: FlickerLaw::Area {
                area: 1.0,
                exponent: 2.0,
            },
        };
        assert!((steep.scalar_output_density(1.0, 10.0).unwrap() - 1e-14).abs() < 1e-28);
    }

    #[test]
    fn coherent_noise_combines_transfers_before_squaring() {
        let first = Complex::new(2., -3.);
        let opposing = Complex::new(-2., 3.);
        assert_eq!(
            correlated_gain(first, opposing, 4., 4., 0.).to_bits(),
            0_f64.to_bits()
        );
        assert_eq!(correlated_gain(first, first, 4., 4., 0.), 208.);
        let second = Complex::new(1.25, -0.875);
        let angle = 0.37_f64;
        let in_phase = 3. * angle.cos();
        let quadrature = 3. * angle.sin();
        let real = 2. * first.re + in_phase * second.re - quadrature * second.im;
        let imag = 2. * first.im + in_phase * second.im + quadrature * second.re;
        let reference = real * real + imag * imag;
        assert_eq!(
            correlated_gain(first, second, 4., 9., angle).to_bits(),
            reference.to_bits()
        );
    }

    #[test]
    fn scalar_evaluation_rejects_coherent_sources_and_invalid_inputs() {
        let coherent = NoiseKind::CorrelatedThermal {
            conductance: 1.,
            second: [NodeId::new(2), NodeId::GROUND],
            second_conductance: 1.,
            phase: 0.,
            temperature: 300.,
        };
        assert!(matches!(
            coherent.scalar_output_density(1., 100.),
            Err(SpiceError::Circuit { .. })
        ));
        let scalar = NoiseKind::Gain { factor: 2. };
        for (gain, frequency) in [
            (-1., 1.),
            (f64::INFINITY, 1.),
            (f64::NAN, 1.),
            (1., 0.),
            (1., f64::NAN),
            (1., f64::INFINITY),
            (f64::MAX, 1.),
        ] {
            assert!(matches!(
                scalar.scalar_output_density(gain, frequency),
                Err(SpiceError::Numerical { .. })
            ));
        }
        // Signed scalar values remain meaningful to the C integration path.
        let signed = NoiseKind::Gain { factor: -0. };
        assert_eq!(
            signed.scalar_output_density(1., 1.).unwrap().to_bits(),
            (-0_f64).to_bits()
        );
        assert_eq!(
            squared_gain(Complex::new(-0., -0.)).to_bits(),
            0_f64.to_bits()
        );
    }

    #[test]
    fn coherent_source_evaluation_observes_both_ordered_transfers() {
        let first = [NodeId::new(1), NodeId::GROUND];
        let second = [NodeId::new(2), NodeId::GROUND];
        let source = NoiseSource::new(
            ".corl",
            first,
            NoiseKind::CorrelatedThermal {
                conductance: 4.,
                second,
                second_conductance: 4.,
                phase: 0.,
                temperature: 300.,
            },
        );
        let cancellation = |nodes| {
            if nodes == first {
                Complex::new(2., -3.)
            } else {
                assert_eq!(nodes, second);
                Complex::new(-2., 3.)
            }
        };
        assert_eq!(
            source
                .kind
                .output_density(cancellation(first), Some(cancellation(second)), 100.)
                .unwrap()
                .to_bits(),
            0_f64.to_bits()
        );
        assert_eq!(
            source
                .kind
                .output_density(Complex::new(2., -3.), Some(Complex::new(2., -3.)), 100.)
                .unwrap()
                .to_bits(),
            (4. * BOLTZMANN * 300. * 208.).to_bits()
        );
        assert!(
            source
                .kind
                .output_density(Complex::real(f64::MAX), Some(Complex::real(f64::MAX)), 100.)
                .is_err()
        );
        assert!(
            source
                .kind
                .output_density(cancellation(first), Some(cancellation(second)), 0.)
                .is_err()
        );

        let invalid = NoiseSource::new(
            ".corl",
            first,
            NoiseKind::CorrelatedThermal {
                conductance: -1.,
                second,
                second_conductance: 4.,
                phase: 0.,
                temperature: 300.,
            },
        );
        assert!(matches!(
            invalid
                .kind
                .output_density(cancellation(first), Some(cancellation(second)), 100.),
            Err(SpiceError::Numerical { .. })
        ));
    }

    #[test]
    fn rf_amplitude_uses_only_independent_thermal_and_shot_sources() {
        let thermal = NoiseKind::Thermal {
            conductance: 1e-3,
            temperature: 300.,
        };
        assert_eq!(
            thermal.rf_amplitude().unwrap(),
            Some((4. * BOLTZMANN * 300. * 1e-3).sqrt())
        );
        let shot = NoiseKind::Shot { current: -1e-3 };
        assert_eq!(
            shot.rf_amplitude().unwrap(),
            Some((2. * CHARGE * 1e-3).sqrt())
        );
        let flicker = NoiseKind::Flicker {
            numerator: 1e-12,
            law: FlickerLaw::PerHertz,
        };
        assert_eq!(flicker.rf_amplitude().unwrap(), None);
        assert_eq!(NoiseKind::Gain { factor: 1. }.rf_amplitude().unwrap(), None);
        let invalid = NoiseKind::Thermal {
            conductance: -1.,
            temperature: 300.,
        };
        assert!(invalid.rf_amplitude().is_err());
        let coherent = NoiseKind::CorrelatedThermal {
            conductance: 1.,
            second: [NodeId::new(2), NodeId::GROUND],
            second_conductance: 1.,
            phase: 0.,
            temperature: 300.,
        };
        assert!(matches!(
            coherent.rf_amplitude(),
            Err(SpiceError::NotYetPorted { .. })
        ));
    }

    #[test]
    fn generators_are_validated() {
        assert!(
            !NoiseKind::Thermal {
                conductance: 1.0,
                temperature: 0.0
            }
            .validate()
        );
        assert!(!NoiseKind::Shot { current: f64::NAN }.validate());
        assert!(
            NoiseKind::Flicker {
                numerator: 0.0,
                law: FlickerLaw::PerHertz
            }
            .validate()
        );
    }
}
