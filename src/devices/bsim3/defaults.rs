//! Model-only default processing from `b3set.c::BSIM3setup`, before instance
//! geometry, state/node allocation, temperature scaling or effective-size
//! `b3check.c` diagnostics. Keep this phase distinct from raw explicit inputs.
use std::collections::HashMap;
use std::sync::OnceLock;

use super::{ModelParameters, Parameter, Value, defaults_data, descriptor, parameters};
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

/// One actual C model value and its given bit. Numeric zero initialization
/// is the `GENmodel` allocation rule, not a fallback for a missing parameter.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub(super) struct ModelField<T> {
    pub(super) value: T,
    pub(super) given: bool,
}

impl<T> ModelField<T> {
    fn assign(&mut self, value: T, given: bool) {
        self.value = value;
        self.given |= given;
    }

    fn apply_default(&mut self, value: T) {
        if !self.given {
            self.value = value;
        }
    }
}

// Numeric names and types come from the generated schema. Every canonical
// numeric parameter is present by construction; aliases are resolved only by
// the public keyword boundary. Integer selectors never inhabit f64 storage.
macro_rules! define_model_values {
    (real { $(($real_name:literal, $real:ident),)* } integer { $(($integer_name:literal, $integer:ident),)* }) => {
        #[derive(Debug, Clone, PartialEq, Default)]
        pub(super) struct ModelValues {
            $(pub(super) $real: ModelField<Real>,)*
            $(pub(super) $integer: ModelField<i32>,)*
        }

        impl ModelValues {
            fn get(&self, canonical: &str) -> Option<Real> {
                match canonical {
                    $($real_name => Some(self.$real.value),)*
                    $($integer_name => Some(Real::from(self.$integer.value)),)*
                    _ => None,
                }
            }

            fn given(&self, canonical: &str) -> bool {
                match canonical {
                    $($real_name => self.$real.given,)*
                    $($integer_name => self.$integer.given,)*
                    _ => false,
                }
            }

            fn set_number(&mut self, canonical: &str, value: Real, given: bool) -> SpiceResult<()> {
                match canonical {
                    $($real_name => self.$real.assign(value, given),)*
                    $($integer_name => {
                        let integer = crate::primitives::exact_signed_integer(value).ok_or_else(|| {
                            SpiceError::circuit(format!("BSIM3 integer model field {canonical} is outside the C i32 domain: {value}"))
                        })?;
                        self.$integer.assign(integer, given);
                    })*
                    _ => return Err(SpiceError::circuit(format!("unknown numeric BSIM3 model field {canonical}"))),
                }
                Ok(())
            }
        }
    };
}

defaults_data::model_fields!(define_model_values);

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
    pub(super) values: ModelValues,
    version_given: bool,
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
        self.values.get(parameter.canonical)
    }

    /// C's given-bit state, not whether the final number differs from a
    /// default. Aliases (and the two type flags) share their actual C bit.
    #[must_use]
    pub fn is_given(&self, name: &str) -> bool {
        model_descriptor(name).is_some_and(|parameter| {
            parameter.c_given.is_some()
                && match parameter.canonical {
                    "nmos" | "pmos" => true, // INPmakeMod always applies the type flag.
                    "version" => self.version_given,
                    canonical => self.values.given(canonical),
                }
        })
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

    pub(super) fn from_parameters(
        input: &ModelParameters,
        context: &ModelContext,
    ) -> SpiceResult<Self> {
        context.validate(&input.location)?;
        let mut result = Self {
            values: ModelValues::default(),
            version_given: false,
            polarity: input.polarity,
            version: "3.3.0".into(),
            oxide_capacitance: 0.,
            warnings: Vec::new(),
        };
        for applied in input.inputs.applied() {
            let canonical = applied.parameter.canonical;
            let given = applied.parameter.c_given.is_some();
            match &applied.value {
                Value::Real(v) => result.values.set_number(canonical, *v, given)?,
                Value::Integer(v) => result.values.set_number(canonical, Real::from(*v), given)?,
                Value::String(v) => {
                    result.version.clone_from(v);
                    result.version_given = given;
                }
                Value::Flag(_) => {} // Effective type was computed in input order.
            }
        }
        let values = &mut result.values;
        defaults_data::apply_literal_defaults!(values);
        for (name, field) in [
            ("nqsmod", &mut result.values.nqsmod),
            ("acnqsmod", &mut result.values.acnqsmod),
        ] {
            let selector = field.value;
            if !matches!(selector, 0 | 1) {
                field.value = 0;
                result.warnings.push(ModelWarning {
                    parameter: name,
                    supplied: Real::from(selector),
                    replacement: 0.,
                    c_reference: "src/spicelib/devices/bsim3/b3set.c (NQS selector reset)",
                });
            }
        }
        let nmos = result.polarity > 0;
        let values = &mut result.values;
        values.vth0.apply_default(if nmos { 0.7 } else { -0.7 });
        values.u0.apply_default(if nmos { 0.067 } else { 0.025 });
        values.noia.apply_default(if nmos { 1e20 } else { 9.9e18 });
        values.noib.apply_default(if nmos { 5e4 } else { 2.4e3 });
        values
            .noic
            .apply_default(if nmos { -1.4e-12 } else { 1.4e-12 });
        let mobility_three = values.mobmod.value == 3;
        values
            .uc
            .apply_default(if mobility_three { -0.0465 } else { -0.0465e-9 });
        values
            .uc1
            .apply_default(if mobility_three { -0.056 } else { -0.056e-9 });
        values.tnom.apply_default(temperature_kelvin(
            context.nominal_temperature,
            &input.location,
        )?);

        values.toxm.apply_default(values.tox.value);
        values.dsub.apply_default(values.drout.value);
        values.llc.apply_default(values.ll.value);
        values.lwc.apply_default(values.lw.value);
        values.lwlc.apply_default(values.lwl.value);
        values.wlc.apply_default(values.wl.value);
        values.wwc.apply_default(values.ww.value);
        values.wwlc.apply_default(values.wwl.value);
        values.dwc.apply_default(values.wint.value);
        values.dlc.apply_default(values.lint.value);
        values.cjswg.apply_default(values.cjsw.value);
        values.pbswg.apply_default(values.pbsw.value);
        values.mjswg.apply_default(values.mjsw.value);
        result.derive_capacitances();
        Ok(result)
    }

    // Retain the C rounding order in fringe and overlap calculations.
    fn derive_capacitances(&mut self) {
        // A nonpositive TOX/TOXM or negative IJTH is reported by
        // `b3check.c` in the instance phase, with C's own message.
        let tox = self.values.tox.value;
        self.oxide_capacitance = EPS_OXIDE / tox;
        if !self.values.cf.given {
            let fringe_ratio = 0.4e-6 / tox;
            let fringe_argument = 1. + fringe_ratio;
            let fringe = 2. * EPS_OXIDE / SETUP_PI * fringe_argument.ln();
            self.values.cf.value = fringe;
        }
        let overlap_length = self.values.dlc.value;
        let cox = self.oxide_capacitance;
        for (overlap, fringe) in [
            (&mut self.values.cgdo, self.values.cgdl.value),
            (&mut self.values.cgso, self.values.cgsl.value),
        ] {
            if !overlap.given {
                let value = if self.values.dlc.given && overlap_length > 0. {
                    let overlap_capacitance = overlap_length * cox;
                    overlap_capacitance - fringe
                } else {
                    0.6 * self.values.xj.value * cox
                };
                overlap.value = value;
            }
        }
        if !self.values.cgbo.given {
            self.values.cgbo.value = 2. * self.values.dwc.value * cox;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::InputType;
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
    fn typed_numeric_storage_matches_every_schema_field_and_given_bit() {
        let mut values = ModelValues::default();
        let mut given_flags = std::collections::BTreeSet::new();
        for parameter in parameters::MODEL.iter().filter(|p| !p.redundant) {
            if matches!(parameter.input_type, InputType::Real | InputType::Integer) {
                assert_eq!(values.get(parameter.canonical), Some(0.));
                assert!(!values.given(parameter.canonical));
                assert!(given_flags.insert(parameter.c_given.unwrap()));
                let value = Real::from(parameter.id);
                values.set_number(parameter.canonical, value, true).unwrap();
            }
        }
        for parameter in parameters::MODEL {
            if matches!(parameter.input_type, InputType::Real | InputType::Integer) {
                assert_eq!(
                    values.get(parameter.canonical),
                    Some(Real::from(parameter.id))
                );
                assert!(values.given(parameter.canonical));
            }
        }
        assert_eq!(values.get("missing"), None);
        assert!(values.set_number("missing", 1., true).is_err());
    }

    #[test]
    fn integer_fields_reject_nonrepresentable_values_at_assignment() {
        let mut values = ModelValues::default();
        for value in [i32::MIN, -1, 0, 3, i32::MAX] {
            values
                .set_number("capmod", Real::from(value), true)
                .unwrap();
            assert_eq!(values.capmod.value, value);
        }
        for invalid in [
            Real::from(i32::MIN) - 1.,
            Real::from(i32::MAX) + 1.,
            1.25,
            Real::INFINITY,
            Real::NAN,
        ] {
            assert!(
                values.set_number("capmod", invalid, true).is_err(),
                "{invalid:?}"
            );
            assert_eq!(values.capmod.value, i32::MAX);
        }
    }
}
