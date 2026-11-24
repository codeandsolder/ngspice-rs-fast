//! Offline quasi-static reference preparation. The solver does not consume
//! geometry or substrate-format syntax; it binds the resulting RLGC modes.

use super::{
    FrequencyBand, LineMode, ModeParameters, invalid, nonnegative, realization::PassiveRealization,
};
use crate::primitives::SpiceResult;
use std::f64::consts::{E, PI, TAU};

const LIGHT_SPEED: f64 = 299_792_458.;
const VACUUM_IMPEDANCE: f64 = 376.730_313_668;
const VACUUM_PERMEABILITY: f64 = VACUUM_IMPEDANCE / LIGHT_SPEED;

fn positive(x: f64) -> bool {
    x.is_finite() && x > 0.
}

/// Planar dielectric substrate in SI units, nonmagnetic.
#[derive(Debug, Clone, Copy)]
pub struct Substrate {
    /// Relative permittivity at the quasi-static reference.
    pub relative_permittivity: f64,
    /// Dielectric thickness in metres.
    pub height: f64,
}

impl Substrate {
    fn validate(self) -> SpiceResult<()> {
        if !(1. ..=128.).contains(&self.relative_permittivity) || !positive(self.height) {
            return Err(invalid("invalid or out-of-domain substrate"));
        }
        Ok(())
    }
}

/// Quasi-TEM impedance and velocity from a prepared cross section.
#[derive(Debug, Clone, Copy)]
pub struct QuasiStatic {
    /// Characteristic impedance in ohms.
    pub impedance: f64,
    /// Effective relative permittivity, defined by phase velocity.
    pub effective_permittivity: f64,
}

impl QuasiStatic {
    fn checked(impedance: f64, effective_permittivity: f64) -> SpiceResult<Self> {
        if !positive(impedance) || !positive(effective_permittivity) {
            return Err(invalid("unrepresentable quasi-static cross section"));
        }
        Ok(Self {
            impedance,
            effective_permittivity,
        })
    }

    /// Phase velocity in metres per second.
    #[must_use]
    pub fn velocity(self) -> f64 {
        LIGHT_SPEED / self.effective_permittivity.sqrt()
    }

    /// Construct the prepared RLGC mode from extracted geometry and
    /// separately specified loss/relaxation coefficients.
    /// # Errors
    /// Invalid geometry-derived or loss coefficients.
    pub fn prepare(self, loss: ModeLoss) -> SpiceResult<LineMode> {
        let velocity = self.velocity();
        LineMode::new(ModeParameters {
            inductance: self.impedance / velocity,
            capacitance: (1. / self.impedance) / velocity,
            resistance: loss.dc_resistance,
            skin: loss.skin,
            conductance: loss.conductance,
            relaxation_capacitance: loss.relaxation_capacitance,
            relaxation_time: loss.relaxation_time,
            reference_temperature: loss.reference_temperature,
            resistance_temperature_coefficient: loss.resistivity_coefficient,
        })
    }
}

/// Extracted conductor/dielectric loss coefficients in SI units.
///
/// A separate electromagnetic or empirical extraction can supply return
/// losses and roughness corrections without adding model-specific geometry
/// rules to the solver. A constant loss tangent is not silently extrapolated
/// into a causal time-domain model.
#[derive(Debug, Clone, Copy)]
pub struct ModeLoss {
    /// DC series resistance (ohm/m).
    pub dc_resistance: f64,
    /// Skin-effect coefficient (ohm/(m sqrt(Hz))).
    pub skin: f64,
    /// DC shunt conductance (S/m).
    pub conductance: f64,
    /// Extra low-frequency Debye capacitance (F/m).
    pub relaxation_capacitance: f64,
    /// Dielectric relaxation time (seconds).
    pub relaxation_time: f64,
    /// Reference temperature (degrees Celsius).
    pub reference_temperature: f64,
    /// Linear temperature coefficient of resistivity (1/K).
    pub resistivity_coefficient: f64,
}

impl Default for ModeLoss {
    fn default() -> Self {
        Self {
            dc_resistance: 0.,
            skin: 0.,
            conductance: 0.,
            relaxation_capacitance: 0.,
            relaxation_time: 0.,
            reference_temperature: 27.,
            resistivity_coefficient: 0.,
        }
    }
}

/// Coaxial cross section with a solid centre conductor and cylindrical shield.
#[derive(Debug, Clone, Copy)]
pub struct Coax {
    /// Centre-conductor radius in metres.
    pub inner_radius: f64,
    /// Shield inner radius in metres.
    pub shield_inner_radius: f64,
    /// Shield outer radius in metres.
    pub shield_outer_radius: f64,
    /// Relative permittivity of the homogeneous dielectric.
    pub relative_permittivity: f64,
}

impl Coax {
    fn validate(self) -> SpiceResult<()> {
        if !positive(self.inner_radius)
            || !positive(self.shield_inner_radius)
            || !positive(self.shield_outer_radius)
            || self.inner_radius >= self.shield_inner_radius
            || self.shield_inner_radius >= self.shield_outer_radius
            || !self.relative_permittivity.is_finite()
            || self.relative_permittivity < 1.
        {
            return Err(invalid("invalid coax cross section"));
        }
        Ok(())
    }

    /// Ideal TEM geometry: electrostatic capacitance and external
    /// inductance, with no conductor-penetration correction.
    /// # Errors
    /// Invalid dimensions, permittivity or numerical range.
    pub fn quasi_static(self) -> SpiceResult<QuasiStatic> {
        self.validate()?;
        let ratio = (self.shield_inner_radius - self.inner_radius) / self.inner_radius;
        let ratio_log = if ratio.is_finite() {
            ratio.ln_1p()
        } else {
            self.shield_inner_radius.ln() - self.inner_radius.ln()
        };
        QuasiStatic::checked(
            VACUUM_IMPEDANCE / TAU * ratio_log / self.relative_permittivity.sqrt(),
            self.relative_permittivity,
        )
    }

    /// Coax conductor losses using DC cross sections and the thick-metal
    /// skin-effect asymptote (nonmagnetic, smooth metal).
    ///
    /// The additive DC plus square-root loss law is a model, not the exact
    /// Bessel-function transition. Use an explicitly valid frequency band.
    /// # Errors
    /// Invalid material, cross section or numerical range.
    pub fn conductor_loss(
        self,
        conductivity: f64,
        reference_temperature: f64,
        resistivity_coefficient: f64,
    ) -> SpiceResult<ModeLoss> {
        self.validate()?;
        if !positive(conductivity)
            || !reference_temperature.is_finite()
            || reference_temperature <= -273.15
            || !resistivity_coefficient.is_finite()
        {
            return Err(invalid("invalid coax conductor material"));
        }
        let inner_area = PI * self.inner_radius * self.inner_radius;
        let shield_area = PI
            * (self.shield_outer_radius - self.shield_inner_radius)
            * (self.shield_outer_radius + self.shield_inner_radius);
        let dc_resistance =
            (conductivity * inner_area).recip() + (conductivity * shield_area).recip();
        let surface = (PI * VACUUM_PERMEABILITY / conductivity).sqrt();
        let skin = surface / TAU * (self.inner_radius.recip() + self.shield_inner_radius.recip());
        if !positive(dc_resistance) || !positive(skin) {
            return Err(invalid("unrepresentable coax conductor loss"));
        }
        Ok(ModeLoss {
            dc_resistance,
            skin,
            reference_temperature,
            resistivity_coefficient,
            ..ModeLoss::default()
        })
    }
}

fn homogeneous_impedance(width_ratio: f64) -> f64 {
    let shape = (TAU - 6.).mul_add((-(30.666 / width_ratio).powf(0.7528)).exp(), 6.);
    VACUUM_IMPEDANCE / TAU * (shape / width_ratio + (2. / width_ratio).hypot(1.)).ln()
}

fn hammerstad_ab(width_ratio: f64, permittivity: f64) -> (f64, f64) {
    let u4 = width_ratio.powi(4);
    let a = 1.
        + ((width_ratio / 52.).mul_add(width_ratio / 52., u4) / (u4 + 0.432)).ln() / 49.
        + (width_ratio / 18.1).powi(3).ln_1p() / 18.7;
    let b = 0.564 * ((permittivity - 0.9) / (permittivity + 3.)).powf(0.053);
    (a, b)
}

fn hammerstad_permittivity(width_ratio: f64, permittivity: f64) -> f64 {
    let (a, b) = hammerstad_ab(width_ratio, permittivity);
    (0.5 * (permittivity - 1.)).mul_add(
        (1. + 10. / width_ratio).powf(-a * b),
        permittivity.midpoint(1.),
    )
}

/// Single microstrip with Hammerstad–Jensen quasi-static correction.
#[derive(Debug, Clone, Copy)]
pub struct Microstrip {
    /// Trace width in metres.
    pub width: f64,
    /// Metal thickness in metres; zero selects the thin-strip model.
    pub thickness: f64,
}

impl Microstrip {
    /// Evaluate the quasi-static geometry, including the finite-thickness
    /// effective-width correction. This is not a full-wave dispersion law.
    /// # Errors
    /// Outside the supported geometry/material range.
    pub fn quasi_static(self, substrate: Substrate) -> SpiceResult<QuasiStatic> {
        substrate.validate()?;
        let width = self.width / substrate.height;
        let thickness = self.thickness / substrate.height;
        if !(0.01..=100.).contains(&width)
            || !nonnegative(thickness)
            || thickness > 0.1
            || thickness > width / 2.
        {
            return Err(invalid("microstrip outside quasi-static geometry domain"));
        }
        let er = substrate.relative_permittivity;
        let delta = if thickness == 0. {
            0.
        } else {
            thickness / PI * (4. * E * (6.517 * width).sqrt().tanh().powi(2) / thickness).ln_1p()
        };
        let air_width = width + delta;
        let dielectric_width = (0.5 * delta).mul_add(1. + (er - 1.).sqrt().cosh().recip(), width);
        let effective = hammerstad_permittivity(dielectric_width, er);
        let dielectric_impedance = homogeneous_impedance(dielectric_width);
        let impedance = dielectric_impedance / effective.sqrt();
        let corrected =
            effective * (homogeneous_impedance(air_width) / dielectric_impedance).powi(2);
        QuasiStatic::checked(impedance, corrected)
    }

    /// Quasi-static fringing capacitance for an open microstrip end.
    ///
    /// Uses the common Hammerstad effective-length approximation. This
    /// is a low-frequency capacitive correction, not a radiating open
    /// or full-wave model.
    /// # Errors
    /// Outside the supported open-end geometry domain or numerical range.
    pub fn open_end_capacitance(self, substrate: Substrate) -> SpiceResult<f64> {
        let cross = self.quasi_static(substrate)?;
        let width = self.width / substrate.height;
        if !(0.1..=10.).contains(&width) {
            return Err(invalid("open microstrip outside fringing formula domain"));
        }
        let er = cross.effective_permittivity;
        let length =
            0.412 * substrate.height * (er + 0.3) / (er - 0.258) * (width + 0.264) / (width + 0.8);
        let capacitance = length / cross.impedance / cross.velocity();
        if !positive(capacitance) {
            return Err(invalid("unrepresentable open-end capacitance"));
        }
        Ok(capacitance)
    }

    /// Quasi-static uncompensated right-angle microstrip corner.
    ///
    /// Equivalent series inductive arms and shunt capacitance from the
    /// Qucs corner model. Geometries with negative deembedded inductance
    /// are not realizable with these positive lumped elements.
    /// # Errors
    /// Outside model validity, or nonpassive equivalent parameters.
    pub fn right_angle_corner(
        self,
        substrate: Substrate,
        band: FrequencyBand,
    ) -> SpiceResult<PassiveRealization> {
        substrate.validate()?;
        let width_ratio = self.width / substrate.height;
        let er = substrate.relative_permittivity;
        if !(0.2..=6.).contains(&width_ratio)
            || !(2.36..=10.4).contains(&er)
            || band.limits()[1] * substrate.height > 12e6
            || self.thickness != 0.
        {
            return Err(invalid(
                "microstrip corner outside validated thin-strip domain",
            ));
        }
        let capacitance = 1e-12
            * self.width
            * (10.35_f64
                .mul_add(er, 2.5)
                .mul_add(width_ratio, 2.6_f64.mul_add(er, 5.64)));
        let inductance = 220e-9
            * substrate.height
            * (-1.35_f64).mul_add((-0.18 * width_ratio.powf(1.39)).exp(), 1.);
        PassiveRealization::junction(&[inductance; 2], capacitance, band)
    }

    /// Quasi-static width step with separate inductive arms.
    ///
    /// Uses the Qucs impedance-step capacitance and inductance formulas,
    /// with per-metre static inductance determining the arm split.
    /// # Errors
    /// Outside empirical validity or nonpassive equivalent parameters.
    pub fn step_to(
        self,
        other: Self,
        substrate: Substrate,
        band: FrequencyBand,
    ) -> SpiceResult<PassiveRealization> {
        let a = self.quasi_static(substrate)?;
        let b = other.quasi_static(substrate)?;
        let wide = self.width.max(other.width);
        let narrow = self.width.min(other.width);
        let ratio = wide / narrow;
        let er = substrate.relative_permittivity;
        if !(1.5..=3.5).contains(&ratio)
            || er > 10.
            || band.limits()[1] * substrate.height > 12e6
            || self.thickness != 0.
            || other.thickness != 0.
        {
            return Err(invalid(
                "microstrip width step outside validated thin-strip domain",
            ));
        }
        let log_er = er.log10();
        let capacitance = 1e-12
            * wide.sqrt()
            * narrow.sqrt()
            * (10.1_f64
                .mul_add(log_er, 2.33)
                .mul_add(ratio, -12.6_f64.mul_add(log_er, 3.17)));
        let delta = ratio - 1.;
        let inductance = 1e-9
            * substrate.height
            * (0.2_f64
                .mul_add(delta, 40.5)
                .mul_add(delta, -75. * ratio.log10()));
        let first = a.impedance / a.velocity();
        let second = b.impedance / b.velocity();
        let sum = first + second;
        PassiveRealization::junction(
            &[inductance * (first / sum), inductance * (second / sum)],
            capacitance,
            band,
        )
    }
}

fn agm(mut a: f64, mut b: f64) -> f64 {
    for _ in 0..32 {
        let next_a = a.midpoint(b);
        let next_b = a.sqrt() * b.sqrt();
        if (next_a - next_b).abs() <= 2. * f64::EPSILON * next_a {
            return next_a;
        }
        a = next_a;
        b = next_b;
    }
    a
}

// K(k)/K'(k) = AGM(1,k)/AGM(1,k'), avoiding evaluation of a
// divergent elliptic integral. The complementary modulus is evaluated
// independently by the conformal-map geometry.
fn elliptic_ratio(modulus: f64, complement: f64) -> f64 {
    agm(1., modulus) / agm(1., complement)
}

/// Grounded coplanar waveguide, zero metal thickness and infinite side grounds.
#[derive(Debug, Clone, Copy)]
pub struct GroundedCpw {
    /// Centre trace width in metres.
    pub width: f64,
    /// Gap to either side ground in metres.
    pub gap: f64,
}

impl GroundedCpw {
    /// Conductor-backed conformal-map model (Qucs equations 12.10–12.13).
    /// # Errors
    /// Outside the supported quasi-static geometry/material domain.
    pub fn quasi_static(self, substrate: Substrate) -> SpiceResult<QuasiStatic> {
        substrate.validate()?;
        let width = self.width / substrate.height;
        let gap = self.gap / substrate.height;
        if !(0.01..=20.).contains(&width) || !(0.001..=20.).contains(&gap) {
            return Err(invalid("grounded CPW outside quasi-static geometry domain"));
        }
        let total = 2_f64.mul_add(gap, width);
        let k = width / total;
        let kp = 2. * (gap * (width + gap)).sqrt() / total;
        let first = elliptic_ratio(k, kp);
        let a = PI * width / 4.;
        let b = PI * total / 4.;
        let kb = a.tanh() / b.tanh();
        // 1-tanh(a)^2/tanh(b)^2, expressed with sinh/cosh
        // so large normalized dimensions retain a nonzero complement.
        let kb_prime = ((b - a).sinh() * (b + a).sinh()).sqrt() / (a.cosh() * b.sinh());
        let second = elliptic_ratio(kb, kb_prime);
        let filling = second / (first + second);
        let effective = (substrate.relative_permittivity - 1.).mul_add(filling, 1.);
        QuasiStatic::checked(
            VACUUM_IMPEDANCE / 2. / effective.sqrt() / (first + second),
            effective,
        )
    }
}

/// Symmetric, zero-thickness pair of parallel microstrips.
#[derive(Debug, Clone, Copy)]
pub struct CoupledMicrostrip {
    /// Width of each trace in metres.
    pub width: f64,
    /// Edge-to-edge spacing in metres.
    pub gap: f64,
}

impl CoupledMicrostrip {
    /// Even and odd quasi-static modes (Kirschning–Jansen).
    ///
    /// The prepared line uses their orthonormal sum/difference transform.
    /// This geometry model assumes equal trace widths, unlike the general
    /// externally prepared orthogonal mode interface.
    /// # Errors
    /// Outside the documented model domain.
    pub fn quasi_static(self, substrate: Substrate) -> SpiceResult<[QuasiStatic; 2]> {
        substrate.validate()?;
        let width = self.width / substrate.height;
        let gap = self.gap / substrate.height;
        let er = substrate.relative_permittivity;
        if !(0.1..=10.).contains(&width) || !(0.1..=10.).contains(&gap) || er > 18. {
            return Err(invalid(
                "coupled microstrip outside Kirschning–Jansen domain",
            ));
        }
        let [even_epsilon, odd_epsilon] = coupled_permittivity(width, gap, er);
        let [even_correction, odd_correction] = coupled_impedance_correction(width, gap);
        let homogeneous = homogeneous_impedance(width);
        let impedance = |epsilon: f64, correction: f64| {
            homogeneous / epsilon.sqrt() / (-homogeneous / VACUUM_IMPEDANCE).mul_add(correction, 1.)
        };
        Ok([
            QuasiStatic::checked(impedance(even_epsilon, even_correction), even_epsilon)?,
            QuasiStatic::checked(impedance(odd_epsilon, odd_correction), odd_epsilon)?,
        ])
    }
}

fn coupled_permittivity(width: f64, gap: f64, er: f64) -> [f64; 2] {
    let gap_squared = gap * gap;
    let even_width = width.mul_add(
        (20. + gap_squared) / (10. + gap_squared),
        gap * (-gap).exp(),
    );
    let even = hammerstad_permittivity(even_width, er);
    let single = hammerstad_permittivity(width, er);
    let midpoint = er.midpoint(1.);
    let exponent = 0.694_f64.mul_add((-0.562 * width).exp(), 0.593);
    let scale = 0.747 * er / (0.15 + er);
    let decay = (-(scale - 0.207)).mul_add((-0.414 * width).exp(), scale);
    let correction = -0.7287 * (single - midpoint) * (-0.179 * width).exp_m1();
    let odd = (midpoint + correction - single).mul_add((-decay * gap.powf(exponent)).exp(), single);
    [even, odd]
}

/// Dimensionless empirical impedance corrections, equations 11.119–11.129.
fn coupled_impedance_correction(width: f64, gap: f64) -> [f64; 2] {
    let width_scale = 0.8695 * width.powf(0.194);
    let gap_scale = 0.189_f64.mul_add(gap.powf(2.31), 0.7519_f64.mul_add(gap, 1.));
    let gap_log = 10. * gap.ln();
    let even_exponent = 0.1975
        + (16.6 + (8.4 / gap).powi(6)).powf(-0.387)
        + (gap_log - (gap / 3.4).powi(10).ln_1p()) / 241.;
    let mixing = (-gap).exp();
    let width_power = width.powf(even_exponent);
    let even =
        2. * width_scale / gap_scale / mixing.mul_add(width_power, (2. - mixing) / width_power);
    let odd_scale = 1.14_f64.mul_add(
        (0.638 / (0.517_f64.mul_add(gap.powf(2.43), gap))).ln_1p(),
        1.794,
    );
    let odd_exponent = 0.2305
        + (gap_log - (gap / 5.8).powi(10).ln_1p()) / 281.3
        + (0.598 * gap.powf(1.154)).ln_1p() / 5.1;
    let ratio = 190_f64.mul_add(gap * gap, 10.) / 82.3_f64.mul_add(gap.powi(3), 1.);
    let narrow = (-0.95_f64).mul_add(gap.ln(), -6.5) - (gap / 0.15).powi(5);
    let correction_exponent = ratio.ln() * (narrow.exp() + 1. / 16.5);
    let odd = (odd_scale / gap_scale).mul_add(
        -(width.ln() * odd_exponent * width.powf(-correction_exponent)).exp(),
        even,
    );
    [even, odd]
}
