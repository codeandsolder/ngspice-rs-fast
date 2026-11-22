//! Model-only default processing from `b3set.c::BSIM3setup`, before instance
//! geometry, state/node allocation, temperature scaling or effective-size
//! `b3check.c` diagnostics. Keep this phase distinct from raw explicit inputs.
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::sync::OnceLock;

use super::{InputType, ModelParameters, Parameter, Value, defaults_data, descriptor, parameters};
use crate::devices::ModelContext;
use crate::devices::schema::temperature_kelvin;
use crate::primitives::{Real, SourceLoc, SpiceError, SpiceResult};

const EPS_OXIDE: Real = 3.453_133e-11;
// b3set.c uses this rounded constant, not the platform's full-precision pi.
// Preserve the exact binary64 value of b3set.c's deliberately rounded PI.
const SETUP_PI: Real = Real::from_bits(0x4009_21fb_5452_4550);

/// The generated descriptor remains authoritative; index its canonical
/// spelling and aliases once instead of scanning hundreds of rows at every
/// model-field access in setup and the numerical load. Mixed-case external
/// input retains the shared descriptor's ASCII-insensitive semantics.
fn model_descriptor(name: &str) -> Option<&'static Parameter> {
    static INDEX: OnceLock<HashMap<&'static str, &'static Parameter>> = OnceLock::new();
    let index = INDEX.get_or_init(|| {
        let mut index = HashMap::with_capacity(parameters::MODEL.len());
        for parameter in parameters::MODEL {
            index.entry(parameter.name).or_insert(parameter);
        }
        index
    });
    index.get(name).copied().or_else(|| {
        name.bytes()
            .any(|b| b.is_ascii_uppercase())
            .then(|| descriptor(parameters::MODEL, name))
            .flatten()
    })
}

/// A model selector reset performed by `b3set.c`, with the given flag retained.
/// Device-layer code returns diagnostics rather than printing them.
#[derive(Debug, Clone, PartialEq)]
pub struct ModelWarning {
    /// Canonical input keyword.
    pub parameter: &'static str,
    /// The value supplied to setup, after integer setter conversion.
    pub supplied: Real,
    /// The default selected by C's recovery rule.
    pub replacement: Real,
    /// Upstream behavior reference.
    pub c_reference: &'static str,
}

/// BSIM3 model defaults before geometry or temperature processing.
///
/// Model fields as the model-only part of `b3set.c` leaves them, not a
/// temperature- or geometry-processed transistor. Zero-initialized but ungiven
/// fields (such as K1/K2) remain distinguishable through [`Self::is_given`].
#[derive(Debug, Clone, PartialEq)]
pub struct ModelDefaults {
    numeric: BTreeMap<u16, Real>,
    given: BTreeSet<&'static str>,
    /// Effective NMOS (+1) or PMOS (-1), after ordered type flags.
    pub polarity: i8,
    /// VERSION's last string setter, or C's setup default `3.3.0`.
    pub version: String,
    /// C's `BSIM3cox`, calculated from TOX (F/m²).
    pub oxide_capacitance: Real,
    /// C's NQS selector reset warnings, in setup order.
    pub warnings: Vec<ModelWarning>,
}

impl ModelDefaults {
    /// A numeric model field, including setup defaults and zero initialization.
    /// Flags and VERSION are exposed separately; aliases use the same C ID.
    #[must_use]
    pub fn get(&self, name: &str) -> Option<Real> {
        let parameter = model_descriptor(name)?;
        self.numeric.get(&parameter.id).copied()
    }

    /// Read a C `IF_INTEGER` model field without silently saturating a float.
    ///
    /// Setter values are stored in the shared numeric projection as f64, but
    /// integer fields must remain representable in C's signed i32 domain.
    ///
    /// # Errors
    /// Missing, non-integer, nonfinite or out-of-range internal model fields.
    pub(super) fn integer(&self, name: &str) -> SpiceResult<i32> {
        let field = model_descriptor(name)
            .filter(|field| field.input_type == InputType::Integer)
            .ok_or_else(|| {
                SpiceError::circuit(format!("BSIM3 model field {name} is not an integer"))
            })?;
        let value = self
            .numeric
            .get(&field.id)
            .copied()
            .ok_or_else(|| SpiceError::circuit(format!("BSIM3 model defaults lost {name}")))?;
        // The IF_INTEGER setter has already applied C's floor(v + 0.5).
        // Validate the internal field rather than rounding it a second time.
        if value.fract() != 0. {
            return Err(SpiceError::circuit(format!(
                "BSIM3 integer model field {name} is fractional: {value}"
            )));
        }
        crate::primitives::exact_signed_integer(value).ok_or_else(|| {
            SpiceError::circuit(format!(
                "BSIM3 integer model field {name} is outside the C i32 range: {value}"
            ))
        })
    }

    /// C's given-bit state, not whether the final number differs from a
    /// default. Aliases (and the two type flags) share their actual C bit.
    #[must_use]
    pub fn is_given(&self, name: &str) -> bool {
        model_descriptor(name)
            .and_then(|p| p.c_given)
            .is_some_and(|flag| self.given.contains(flag))
    }

    /// Apply `b3temp.c`'s model-level temperature pass.
    ///
    /// # Errors
    /// Nonphysical temperature or nonfinite derived quantities.
    pub fn at_temperature(
        &self,
        temperature: Real,
        location: &SourceLoc,
    ) -> SpiceResult<crate::devices::bsim3::temperature::TemperatureModel> {
        crate::devices::bsim3::temperature::TemperatureModel::new(self, temperature, location)
    }

    fn set(&mut self, name: &str, value: Real) -> SpiceResult<()> {
        let p = model_descriptor(name)
            .ok_or_else(|| SpiceError::circuit(format!("unknown internal BSIM3 default {name}")))?;
        // Dependent defaults may legitimately be nonfinite when a given TOX is
        // not positive; `b3check.c` reports that in the instance phase, as C
        // does, so this projection must not pre-empt it with a different text.
        if !matches!(p.input_type, InputType::Real | InputType::Integer) {
            return Err(SpiceError::Numerical {
                context: "BSIM3 model defaults (bsim3/b3set.c)".into(),
                message: format!("nonfinite or nonnumeric default {name}={value}"),
            });
        }
        self.numeric.insert(p.id, value);
        Ok(())
    }

    fn default(&mut self, name: &str, value: Real) -> SpiceResult<()> {
        if !self.is_given(name) {
            self.set(name, value)?;
        }
        Ok(())
    }

    pub(super) fn number(&self, name: &str) -> SpiceResult<Real> {
        self.get(name).ok_or_else(|| {
            SpiceError::circuit(format!("missing internal BSIM3 model field {name}"))
        })
    }

    pub(super) fn from_parameters(
        input: &ModelParameters,
        context: &ModelContext,
    ) -> SpiceResult<Self> {
        context.validate(&input.location)?;
        let mut result = Self {
            numeric: BTreeMap::new(),
            given: BTreeSet::new(),
            polarity: input.polarity,
            version: "3.3.0".into(),
            oxide_capacitance: 0.,
            warnings: Vec::new(),
        };
        // GENmodel allocation zero-initializes fields not defaulted by setup.
        // These must NOT become given: b3temp.c needs the K1/K2/etc distinction.
        for p in parameters::MODEL.iter().filter(|p| !p.redundant) {
            if matches!(p.input_type, InputType::Real | InputType::Integer) {
                result.numeric.insert(p.id, 0.);
            }
        }
        // INPmakeMod applies the base NMOS/PMOS flag before tail setters.
        result.given.insert("BSIM3typeGiven");
        for applied in input.inputs.applied() {
            if let Some(flag) = applied.parameter.c_given {
                result.given.insert(flag);
            }
            match &applied.value {
                Value::Real(v) => {
                    result.set(applied.parameter.name, *v)?;
                }
                Value::Integer(v) => {
                    result.set(applied.parameter.name, Real::from(*v))?;
                }
                Value::String(v) => result.version.clone_from(v),
                Value::Flag(_) => {} // Effective type was computed in input order.
            }
        }
        for &(name, value) in defaults_data::LITERALS {
            result.default(name, value)?;
        }
        for name in ["nqsmod", "acnqsmod"] {
            let selector = result.integer(name)?;
            if !matches!(selector, 0 | 1) {
                result.set(name, 0.)?;
                result.warnings.push(ModelWarning {
                    parameter: name,
                    supplied: Real::from(selector),
                    replacement: 0.,
                    c_reference: "src/spicelib/devices/bsim3/b3set.c (NQS selector reset)",
                });
            }
        }
        let nmos = result.polarity > 0;
        for (name, n, p) in [
            ("vth0", 0.7, -0.7),
            ("u0", 0.067, 0.025),
            ("noia", 1e20, 9.9e18),
            ("noib", 5e4, 2.4e3),
            ("noic", -1.4e-12, 1.4e-12),
        ] {
            result.default(name, if nmos { n } else { p })?;
        }
        let mobility_three = result.integer("mobmod")? == 3;
        result.default("uc", if mobility_three { -0.0465 } else { -0.0465e-9 })?;
        result.default("uc1", if mobility_three { -0.056 } else { -0.056e-9 })?;
        result.default(
            "tnom",
            temperature_kelvin(context.nominal_temperature, &input.location)?,
        )?;
        for (to, from) in [
            ("toxm", "tox"),
            ("dsub", "drout"),
            ("llc", "ll"),
            ("lwc", "lw"),
            ("lwlc", "lwl"),
            ("wlc", "wl"),
            ("wwc", "ww"),
            ("wwlc", "wwl"),
            ("dwc", "wint"),
            ("dlc", "lint"),
            ("cjswg", "cjsw"),
            ("pbswg", "pbsw"),
            ("mjswg", "mjsw"),
        ] {
            result.default(to, result.number(from)?)?;
        }
        result.derive_capacitances()?;
        Ok(result)
    }

    // Retain the C rounding order in fringe and overlap calculations.
    fn derive_capacitances(&mut self) -> SpiceResult<()> {
        // A nonpositive TOX/TOXM or negative IJTH is reported by
        // `b3check.c` in the instance phase, with C's own message.
        let tox = self.number("tox")?;
        self.oxide_capacitance = EPS_OXIDE / tox;
        if !self.is_given("cf") {
            let fringe_ratio = 0.4e-6 / tox;
            let fringe_argument = 1. + fringe_ratio;
            let fringe = 2. * EPS_OXIDE / SETUP_PI * fringe_argument.ln();
            self.set("cf", fringe)?;
        }
        let overlap_length = self.number("dlc")?;
        let cox = self.oxide_capacitance;
        for (overlap, fringe) in [("cgdo", "cgdl"), ("cgso", "cgsl")] {
            if !self.is_given(overlap) {
                let value = if self.is_given("dlc") && overlap_length > 0. {
                    let overlap_capacitance = overlap_length * cox;
                    overlap_capacitance - self.number(fringe)?
                } else {
                    0.6 * self.number("xj")? * cox
                };
                self.set(overlap, value)?;
            }
        }
        if !self.is_given("cgbo") {
            self.set("cgbo", 2. * self.number("dwc")? * cox)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn indexed_model_descriptor_matches_generated_order_and_aliases() {
        for parameter in parameters::MODEL {
            assert!(
                parameter
                    .name
                    .bytes()
                    .all(|byte| !byte.is_ascii_uppercase())
            );
            for name in [
                parameter.name.to_owned(),
                parameter.name.to_ascii_uppercase(),
            ] {
                assert_eq!(
                    model_descriptor(&name),
                    descriptor(parameters::MODEL, &name),
                    "{name}"
                );
            }
        }
        assert_eq!(model_descriptor("missing"), None);
        assert_eq!(model_descriptor("MISSING"), None);
    }

    #[test]
    fn integer_model_fields_validate_type_and_c_bounds() {
        let mut model = ModelDefaults {
            numeric: BTreeMap::new(),
            given: BTreeSet::new(),
            polarity: 1,
            version: "3.3.0".into(),
            oxide_capacitance: 0.,
            warnings: Vec::new(),
        };
        for value in [i32::MIN, -1, 0, 3, i32::MAX] {
            model.set("capmod", Real::from(value)).unwrap();
            assert_eq!(model.integer("capmod").unwrap(), value);
        }
        for invalid in [
            Real::from(i32::MIN) - 1.,
            Real::from(i32::MAX) + 1.,
            1.25,
            Real::INFINITY,
            Real::NAN,
        ] {
            model.set("capmod", invalid).unwrap();
            assert!(model.integer("capmod").is_err(), "{invalid:?}");
        }
        assert!(model.integer("tox").is_err());
        assert!(model.integer("missing").is_err());
        assert!(model.integer("noimod").is_err());
    }
}
