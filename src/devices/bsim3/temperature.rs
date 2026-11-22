//! Model-level temperature processing from `b3temp.c::BSIM3temp`.
//!
//! Before per-instance effective geometry, junction-capacitance clamping,
//! saturation currents and potentials follow C order and C constants. No load
//! equations are evaluated here.
// C-compatible numerical expressions are kept in their reference order:
// Clippy's fused multiply-add suggestions change IEEE-754 rounding.
#![expect(
    clippy::float_cmp,
    reason = "C device branches compare floating-point values exactly, including nonfinite cases"
)]
#![expect(
    clippy::too_many_lines,
    reason = "a complete translated C numerical routine remains contiguous for equation parity review"
)]
#![expect(
    clippy::suboptimal_flops,
    reason = "SPICE numerical reference parity requires the original non-fused C operation order"
)]

use super::Diagnostic;
use super::defaults::ModelDefaults;
use crate::devices::schema::temperature_kelvin;
use crate::primitives::{Real, SourceLoc, SpiceResult};

// `b3temp.c` local constants, deliberately not the NIST values of `const.h`.
const KB_OVER_Q: Real = 8.617_087e-5;
const EPS_SI: Real = 1.035_94e-10;
const EPS_OX: Real = 3.453_133e-11;
// `main.c`: CONSTvt0 = CONSTboltz * REFTEMP / CHARGE (300.15 K).
const CONST_BOLTZ: Real = 1.380_648_52e-23;
const CONST_CHARGE: Real = 1.602_176_620_8e-19;
const REF_TEMP: Real = 27.0 + 273.15;
const CONST_VT0: Real = CONST_BOLTZ * REF_TEMP / CONST_CHARGE;
// `const.h` spells this just below binary64 SQRT_2; do not change the bit pattern.
const CONST_ROOT2: Real = Real::from_bits(0x3ff6_a09e_667f_3bcc);
/// `b3temp.c`'s `Meter2Micron` equivalent used by the size-dependent setup.
pub(super) const METER2MICRON: Real = 1.0e6;

/// Model fields after `b3temp.c`'s temperature pass, plus the internal
/// constants the size-dependent setup needs.
#[derive(Debug, Clone, PartialEq)]
pub struct TemperatureModel {
    pub(super) pb: Real,
    pub(super) pbsw: Real,
    pub(super) pbswg: Real,
    pub(super) jct_temp_sat_cur_density: Real,
    pub(super) jct_sidewall_temp_sat_cur_density: Real,
    pub(super) unit_area_temp_jct_cap: Real,
    pub(super) unit_length_sidewall_temp_jct_cap: Real,
    pub(super) unit_length_gate_sidewall_temp_jct_cap: Real,
    pub(super) vcrit: Real,
    pub(super) factor1: Real,
    pub(super) vtm: Real,
    pub(super) vtm0: Real,
    pub(super) ni: Real,
    pub(super) eg0: Real,
    pub(super) eg: Real,
    pub(super) phi_b: Real,
    pub(super) phi_bsw: Real,
    pub(super) phi_bswg: Real,
    pub(super) temp: Real,
    pub(super) tnom: Real,
    /// Nonfatal `b3temp.c` messages in C order.
    pub warnings: Vec<Diagnostic>,
}

impl TemperatureModel {
    /// Apply C's model-level temperature pass. `temperature` is the circuit
    /// temperature in degrees Celsius (`CKTtemp` is Kelvin in C).
    ///
    /// # Errors
    /// Nonfinite or nonphysical temperature, or nonfinite derived quantities.
    pub fn new(
        defaults: &ModelDefaults,
        temperature: Real,
        location: &SourceLoc,
    ) -> SpiceResult<Self> {
        let temp = temperature_kelvin(temperature, location)?;
        let tnom = defaults.values.tnom.value;
        if tnom <= 0. || !temp.is_finite() {
            return Err(crate::primitives::SpiceError::parse(
                location.clone(),
                "BSIM3 temperature processing needs a positive TNOM",
            ));
        }
        let mut warnings = Vec::new();

        // Junction potentials are clamped before anything uses them.
        let mut clamp_potential = |given: Real, message: &'static str| {
            if given < 0.1 {
                warnings.push(Diagnostic::warning(
                    message,
                    "src/spicelib/devices/bsim3/b3temp.c",
                ));
                0.1
            } else {
                given
            }
        };
        let pb = clamp_potential(
            defaults.values.pb.value,
            "Given pb is less than 0.1. Pb is set to 0.1.",
        );
        let pbsw = clamp_potential(
            defaults.values.pbsw.value,
            "Given pbsw is less than 0.1. Pbsw is set to 0.1.",
        );
        let pbswg = clamp_potential(
            defaults.values.pbswg.value,
            "Given pbswg is less than 0.1. Pbswg is set to 0.1.",
        );

        let vcrit = CONST_VT0 * (CONST_VT0 / (CONST_ROOT2 * 1.0e-14)).ln();
        let factor1 = (EPS_SI / EPS_OX * defaults.values.tox.value).sqrt();

        let vtm0 = KB_OVER_Q * tnom;
        let eg0 = 1.16 - 7.02e-4 * tnom * tnom / (tnom + 1108.0);
        let ni = 1.45e10
            * (tnom / 300.15)
            * (tnom / 300.15).sqrt()
            * (21.556_598_1 - eg0 / (2.0 * vtm0)).exp();
        let vtm = KB_OVER_Q * temp;
        let eg = 1.16 - 7.02e-4 * temp * temp / (temp + 1108.0);

        let (jct_temp_sat_cur, jct_sidewall_temp_sat_cur) = if temp == tnom {
            (defaults.values.js.value, defaults.values.jsw.value)
        } else {
            let exponent = eg0 / vtm0 - eg / vtm + defaults.values.xti.value * (temp / tnom).ln();
            let scale = (exponent / defaults.values.nj.value).exp();
            (
                defaults.values.js.value * scale,
                defaults.values.jsw.value * scale,
            )
        };
        let jct_temp_sat_cur_density = jct_temp_sat_cur.max(0.);
        let jct_sidewall_temp_sat_cur_density = jct_sidewall_temp_sat_cur.max(0.);

        let delta_temp = temp - tnom;
        let mut scale_capacitance = |cap: Real, coefficient: Real, message: &'static str| {
            let scaled = coefficient * delta_temp;
            if scaled >= -1.0 {
                cap * (1.0 + scaled)
            } else {
                // C only overwrites the field when the base cap is positive;
                // otherwise a zero-initialized model field is left behind.
                if cap > 0.0 {
                    warnings.push(Diagnostic::warning(
                        message,
                        "src/spicelib/devices/bsim3/b3temp.c",
                    ));
                }
                0.0
            }
        };
        let unit_area_temp_jct_cap = scale_capacitance(
            defaults.values.cj.value,
            defaults.values.tcj.value,
            "Temperature effect has caused cj to be negative. Cj is clamped to zero.",
        );
        let unit_length_sidewall_temp_jct_cap = scale_capacitance(
            defaults.values.cjsw.value,
            defaults.values.tcjsw.value,
            "Temperature effect has caused cjsw to be negative. Cjsw is clamped to zero.",
        );
        let unit_length_gate_sidewall_temp_jct_cap = scale_capacitance(
            defaults.values.cjswg.value,
            defaults.values.tcjswg.value,
            "Temperature effect has caused cjswg to be negative. Cjswg is clamped to zero.",
        );

        let phi_b = (pb - defaults.values.tpb.value * delta_temp).max(0.01);
        let phi_bsw = (pbsw - defaults.values.tpbsw.value * delta_temp).max(0.01);
        let phi_bswg = (pbswg - defaults.values.tpbswg.value * delta_temp).max(0.01);
        if pb - defaults.values.tpb.value * delta_temp < 0.01 {
            warnings.push(Diagnostic::warning(
                "Temperature effect has caused pb to be less than 0.01. Pb is clamped to 0.01.",
                "src/spicelib/devices/bsim3/b3temp.c",
            ));
        }
        if pbsw - defaults.values.tpbsw.value * delta_temp <= 0.01 {
            warnings.push(Diagnostic::warning(
                "Temperature effect has caused pbsw to be less than 0.01. Pbsw is clamped to 0.01.",
                "src/spicelib/devices/bsim3/b3temp.c",
            ));
        }
        if pbswg - defaults.values.tpbswg.value * delta_temp <= 0.01 {
            warnings.push(Diagnostic::warning(
                "Temperature effect has caused pbswg to be less than 0.01. Pbswg is clamped to 0.01.",
                "src/spicelib/devices/bsim3/b3temp.c",
            ));
        }

        for (name, value) in [
            ("vcrit", vcrit),
            ("factor1", factor1),
            ("vtm", vtm),
            ("vtm0", vtm0),
            ("ni", ni),
            ("eg0", eg0),
            ("eg", eg),
            ("phi_b", phi_b),
            ("phi_bsw", phi_bsw),
            ("phi_bswg", phi_bswg),
            ("temp", temp),
            ("tnom", tnom),
        ] {
            if !value.is_finite() {
                return Err(crate::primitives::SpiceError::parse(
                    location.clone(),
                    format!("BSIM3 temperature processing produced nonfinite {name}"),
                ));
            }
        }
        Ok(Self {
            pb,
            pbsw,
            pbswg,
            jct_temp_sat_cur_density,
            jct_sidewall_temp_sat_cur_density,
            unit_area_temp_jct_cap,
            unit_length_sidewall_temp_jct_cap,
            unit_length_gate_sidewall_temp_jct_cap,
            vcrit,
            factor1,
            vtm,
            vtm0,
            ni,
            eg0,
            eg,
            phi_b,
            phi_bsw,
            phi_bswg,
            temp,
            tnom,
            warnings,
        })
    }

    /// A processed value by its stable name (a model keyword or an internal
    /// name such as `vcrit`, `phi_b`).
    #[must_use]
    pub fn get(&self, name: &str) -> Option<Real> {
        match name {
            "pb" => Some(self.pb),
            "pbsw" => Some(self.pbsw),
            "pbswg" => Some(self.pbswg),
            "jct_temp_sat_cur_density" => Some(self.jct_temp_sat_cur_density),
            "jct_sidewall_temp_sat_cur_density" => Some(self.jct_sidewall_temp_sat_cur_density),
            "unit_area_temp_jct_cap" => Some(self.unit_area_temp_jct_cap),
            "unit_length_sidewall_temp_jct_cap" => Some(self.unit_length_sidewall_temp_jct_cap),
            "unit_length_gate_sidewall_temp_jct_cap" => {
                Some(self.unit_length_gate_sidewall_temp_jct_cap)
            }
            "vcrit" => Some(self.vcrit),
            "factor1" => Some(self.factor1),
            "vtm" => Some(self.vtm),
            "vtm0" => Some(self.vtm0),
            "ni" => Some(self.ni),
            "eg0" => Some(self.eg0),
            "eg" => Some(self.eg),
            "phi_b" => Some(self.phi_b),
            "phi_bsw" => Some(self.phi_bsw),
            "phi_bswg" => Some(self.phi_bswg),
            "temp" => Some(self.temp),
            "tnom" => Some(self.tnom),
            _ => None,
        }
    }
}
