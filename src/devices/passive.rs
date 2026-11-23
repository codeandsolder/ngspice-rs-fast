//! Bounded model-backed R/C/L input projection and effective values.
//!
//! References: `res/{ressetup,restemp}.c`, `cap/{capsetup,captemp}.c`,
//! `ind/{indsetup,indtemp}.c`, `cap/capacld.c` and `ind/indacld.c`.
//! Geometry/precedence follows those routines, without C's missing-resistance
//! fallback, global geometry scale or ignored keywords. A tiny instance
//! resistance is clamped to `RESMIN` as `resparam.c` does (`rlc::res_resist`).
//! Ordinary setup and sensitivity replay share C-ordered passive arithmetic.
//!
//! ```
//! use std::path::Path;
//! use ngspice_rs::devices::{Circuit, ModelContext};
//! use ngspice_rs::netlist::{Parser, source::parse_deck_text};
//! let deck = parse_deck_text(Path::new("resistor.cir"),
//!     "model resistor\nr1 a 0 rm\n.model rm r(r=1k tc1=0.01)\n.end\n");
//! let netlist = Parser::new().parse_deck(&deck)?;
//! let mut circuit = Circuit::from_netlist(&netlist)?;
//! let hot = ModelContext::new(77.0, 27.0);
//! let system = circuit.linear_system_with_context(&hot)?;
//! assert_eq!(system.a.get(0, 0), 1.0 / 1500.0);
//! assert_eq!(circuit.linear_system()?.a.get(0, 0), 1.0 / 1000.0);
//! # Ok::<(), ngspice_rs::primitives::SpiceError>(())
//! ```

mod device;
pub(super) mod physics;

pub(super) use device::{instantiate, swept_literal, swept_resistor};
use physics::{
    CapacitorGeometry, CapacitorSetup, Coefficients, CoilGeometry, CoilSetup, SheetGeometry,
    TemperaturePoint, conductance, scaled,
};

use crate::devices::models::{ModelContext, ModelFamily, ResolvedModel};
use crate::devices::schema::{
    ScalarDomain as D, ScalarParameter, ScalarSchema, ScalarUnit as U, ScalarValues,
    temperature_kelvin,
};
use crate::devices::sensitivity::written;
use crate::netlist::ast::DeviceInstance;
use crate::primitives::{Real, SourceLoc, SpiceError, SpiceResult};

/// The three scalar passive backends. No unrelated model family can reach
/// the passive schema or be interpreted as an inductor by default.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum PassiveKind {
    Resistor,
    Capacitor,
    Inductor,
}

impl PassiveKind {
    const fn from_family(family: ModelFamily) -> Option<Self> {
        match family {
            ModelFamily::Resistor => Some(Self::Resistor),
            ModelFamily::Capacitor => Some(Self::Capacitor),
            ModelFamily::Inductor => Some(Self::Inductor),
            _ => None,
        }
    }

    const fn family(self) -> ModelFamily {
        match self {
            Self::Resistor => ModelFamily::Resistor,
            Self::Capacitor => ModelFamily::Capacitor,
            Self::Inductor => ModelFamily::Inductor,
        }
    }

    pub(super) const fn principal(self) -> ScalarParameter {
        match self {
            Self::Resistor => p("resistance", U::Ohm, D::Finite, None),
            Self::Capacitor => p("capacitance", U::Farad, D::NonNegative, None),
            Self::Inductor => p("inductance", U::Henry, D::NonNegative, None),
        }
    }

    const fn initial(self) -> Option<ScalarParameter> {
        match self {
            Self::Resistor => None,
            Self::Capacitor => Some(p("ic", U::Volt, D::Finite, None)),
            Self::Inductor => Some(p("ic", U::Ampere, D::Finite, None)),
        }
    }
}

/// Immutable validated input recipe, not a mutable temperature-adjusted cache.
///
/// Explicit instance values override model defaults; model-specific geometry
/// precedence is preserved. Derived values are recomputed for every context.
#[derive(Debug, Clone, PartialEq)]
pub struct PassiveParameters {
    kind: PassiveKind,
    nominal_value: Real,
    temperature: Option<Real>,
    nominal_temperature: Option<Real>,
    coefficients: Coefficients,
    exponential: Option<Real>,
    delta_temperature: Option<Real>,
    ac_resistance: Option<Real>,
    prepared: PreparedModel,
    scale: Real,
    multiplicity: Real,
    initial_condition: Option<Real>,
    /// Resistor `.noise` inputs (`resnoise.c`); the defaults for C and L.
    noisy: bool,
    kf: Real,
    af: Real,
    ef: Real,
    noise_area: Real,
    location: SourceLoc,
    /// The validated model and instance setters (schema defaults
    /// unlocated), for `.sens`, which replays C's own records
    /// ([`crate::devices::sensitivity`]).
    model_values: ScalarValues,
    instance_values: ScalarValues,
}

/// The setup-only model input projection. Geometry used by temperature and
/// per-point setters is read from this immutable prepared state.
#[derive(Debug, Clone, PartialEq)]
enum PreparedModel {
    Resistance {
        scalar: Option<Real>,
        geometry: SheetGeometry,
    },
    Capacitance {
        scalar: Option<Real>,
        default_width: Real,
        geometry: CapacitorGeometry,
    },
    Inductance {
        geometry: CoilGeometry,
        has_model_value: bool,
        has_coil: bool,
    },
}

impl PreparedModel {
    fn prepare(kind: PassiveKind, model: &ScalarValues) -> Self {
        match kind {
            PassiveKind::Resistor => Self::Resistance {
                scalar: optional(model, "r"),
                geometry: SheetGeometry {
                    length: value(model, "l", 10e-6),
                    width: value(model, "defw", 10e-6),
                    short: value(model, "short", 0.),
                    narrow: value(model, "narrow", 0.),
                    sheet: value(model, "rsh", 0.),
                },
            },
            PassiveKind::Capacitor => Self::Capacitance {
                scalar: optional(model, "cap"),
                default_width: value(model, "defw", 10e-6),
                geometry: CapacitorSetup {
                    cj: written(model, "cj"),
                    cjsw: value(model, "cjsw", 0.),
                    narrow: written(model, "narrow"),
                    short: written(model, "short"),
                    del: written(model, "del"),
                    dielectric: written(model, "di"),
                    thickness: written(model, "thick"),
                }
                .prepare(),
            },
            PassiveKind::Inductor => {
                let setup = CoilSetup {
                    inductance: written(model, "ind"),
                    cross_section: written(model, "csect"),
                    diameter: written(model, "dia"),
                    length: written(model, "length"),
                    turns: value(model, "nt", 0.),
                    permeability: value(model, "mu", 1.),
                };
                Self::Inductance {
                    geometry: setup.prepare(),
                    has_model_value: setup.inductance.is_some()
                        || (setup.turns > 0.
                            && setup.length.is_some()
                            && (setup.diameter.is_some() || setup.cross_section.is_some())),
                    has_coil: setup.length.is_some()
                        && (setup.diameter.is_some() || setup.cross_section.is_some()),
                }
            }
        }
    }

    /// Validate bounded geometry separately from C's arithmetic, retaining
    /// the original error location. Missing resistance/inductance is not
    /// silently substituted by a C warning fallback.
    fn nominal(&self, instance: &ScalarValues, location: &SourceLoc) -> SpiceResult<Real> {
        match *self {
            Self::Resistance {
                scalar,
                mut geometry,
            } => {
                geometry.length = value(instance, "l", geometry.length);
                geometry.width = value(instance, "w", geometry.width);
                if geometry.uses_sheet() {
                    let (length, width) = geometry.dimensions();
                    positive(length, location, "effective resistor length")?;
                    positive(width, location, "effective resistor width")?;
                    finite(geometry.resistance(), location, "sheet resistance")
                } else {
                    scalar.ok_or_else(|| SpiceError::parse(location.clone(), "resistor requires an instance resistance, model r or positive sheet geometry; no implicit 1 mOhm fallback"))
                }
            }
            Self::Capacitance {
                scalar,
                default_width,
                geometry,
            } => {
                if let Some(capacitance) = scalar {
                    return Ok(capacitance);
                }
                let length = optional(instance, "l").ok_or_else(|| {
                    SpiceError::parse(
                        location.clone(),
                        "capacitor geometry requires instance l; C setup does not apply model defl",
                    )
                })?;
                let width = value(instance, "w", default_width);
                let (effective_length, effective_width) = geometry.dimensions(length, width);
                positive(effective_length, location, "effective capacitor length")?;
                positive(effective_width, location, "effective capacitor width")?;
                finite(
                    geometry.capacitance(length, width),
                    location,
                    "geometry capacitance",
                )
            }
            Self::Inductance {
                geometry,
                has_model_value,
                has_coil,
            } => {
                let turns = optional(instance, "nt");
                if (turns.is_some() && !has_coil) || (turns.is_none() && !has_model_value) {
                    return Err(SpiceError::parse(
                        location.clone(),
                        "inductor requires an explicit inductance or complete coil geometry and turns",
                    ));
                }
                let nominal = geometry.inductance(None, turns);
                positive(nominal, location, "derived inductance")
            }
        }
    }
}

impl PassiveParameters {
    fn literal(kind: PassiveKind, value: Real, ic: Option<Real>) -> SpiceResult<Self> {
        let location = SourceLoc::new(std::path::PathBuf::from("<literal passive>"), 0, 0);
        let primary = kind.principal();
        let mut instance_values = instance_schema(kind).validate(&[], &location)?;
        instance_values.set(primary.name, value, primary.unit, location.clone())?;
        if let (Some(ic), Some(parameter)) = (ic, kind.initial()) {
            instance_values.set(parameter.name, ic, parameter.unit, location.clone())?;
        }
        let model_values = model_schema(kind).validate(&[], &location)?;
        Ok(Self {
            kind,
            nominal_value: value,
            temperature: None,
            nominal_temperature: None,
            coefficients: Coefficients {
                linear: 0.,
                quadratic: 0.,
            },
            exponential: None,
            delta_temperature: None,
            ac_resistance: None,
            prepared: PreparedModel::prepare(kind, &model_values),
            scale: 1.,
            multiplicity: 1.,
            initial_condition: ic,
            noisy: true,
            kf: 0.,
            af: 1.,
            ef: 1.,
            noise_area: 1.,
            model_values,
            instance_values,
            location,
        })
    }

    /// C's `DEVparam` then `DEVtemperature`, without rerunning `DEVsetup`.
    pub(crate) fn swept(
        &self,
        parameter: &str,
        value: Real,
        context: &ModelContext,
    ) -> SpiceResult<Self> {
        let mut p = self.clone();
        let primary = self.kind.principal().name;
        let definitions = instance_schema(self.kind).parameters();
        let definition = definitions
            .iter()
            .find(|d| d.name == parameter)
            .ok_or_else(|| {
                SpiceError::circuit(format!("passive parameter {parameter} cannot be swept"))
            })?;
        crate::devices::sweep::check_swept(
            "passive",
            parameter,
            value,
            match definition.domain {
                D::Positive => value > 0.,
                D::NonNegative => value >= 0.,
                D::NonZero => value != 0.,
                D::Temperature => value > -273.15,
                D::Finite => true,
            },
            "in its finite physical domain",
        )?;
        p.instance_values
            .set(parameter, value, definition.unit, self.location.clone())?;
        match parameter {
            name if name == primary => {
                let value = if self.kind == PassiveKind::Resistor {
                    crate::devices::rlc::clamp_resistance(value)
                } else {
                    value
                };
                valid_value(self.kind, value, &self.location)?;
                p.nominal_value = value;
                p.instance_values
                    .set(primary, value, definition.unit, self.location.clone())?;
            }
            "temp" => p.temperature = Some(value),
            "tc1" => p.coefficients.linear = value,
            "tc2" => p.coefficients.quadratic = value,
            "tce" => p.exponential = Some(value),
            "dtemp" => p.delta_temperature = Some(value),
            "ac" => p.ac_resistance = Some(value),
            "m" => p.multiplicity = value,
            "scale" => p.scale = value,
            "ic" => p.initial_condition = Some(value),
            "w" | "l" | "nt" => {
                // R/C geometry and instance coil turns are temperature-time
                // quantities. Prepared model geometry is not rerun by DEVparam.
                if optional(&p.instance_values, primary).is_none() {
                    p.nominal_value = p.prepared.nominal(&p.instance_values, &p.location)?;
                }
            }
            _ => {
                return Err(SpiceError::circuit(format!(
                    "passive parameter {parameter} cannot be swept"
                )));
            }
        }
        p.validate_effective(context)?;
        Ok(p)
    }

    /// Validated resistor/capacitor/inductor family.
    #[must_use]
    pub const fn family(&self) -> ModelFamily {
        self.kind.family()
    }
    /// Unadjusted single-instance R/C/L in ohms/farads/henries, respectively.
    /// Does not include temperature, scale or multiplicity.
    #[must_use]
    pub const fn nominal_value(&self) -> Real {
        self.nominal_value
    }
    /// Positive parallel multiplier (default 1).
    #[must_use]
    pub const fn multiplicity(&self) -> Real {
        self.multiplicity
    }
    /// A copy whose supplied scalar is `value`, with the same TC, TEMP/TNOM,
    /// scale and multiplicity. This is the immutable form of an explicit instance
    /// value, which outranks model/geometry bases; the original is unchanged.
    ///
    /// # Errors
    /// `value` is nonfinite, or violates the family's value domain (resistance
    /// must be nonzero with finite conductance; C/L must be nonnegative).
    pub fn with_nominal_value(&self, value: Real) -> SpiceResult<Self> {
        valid_value(self.kind, value, &self.location)?;
        let value = if self.kind == PassiveKind::Resistor {
            crate::devices::rlc::clamp_resistance(value)
        } else {
            value
        };
        let mut recipe = self.clone();
        let principal = self.kind.principal();
        recipe.nominal_value = value;
        recipe
            .instance_values
            .set(principal.name, value, principal.unit, self.location.clone())?;
        Ok(recipe)
    }
    /// The validated model setters (schema defaults have no location).
    #[must_use]
    pub const fn model_values(&self) -> &ScalarValues {
        &self.model_values
    }
    /// The validated instance setters (schema defaults have no location;
    /// resistor principal is clamped by the C setter).
    #[must_use]
    pub const fn instance_values(&self) -> &ScalarValues {
        &self.instance_values
    }
    /// Capacitor initial volts or inductor initial amperes, if given.
    /// Applied by the companion transient under `uic` (see
    /// `Device::storage_element`), ignored otherwise as in C.
    #[must_use]
    pub const fn initial_condition(&self) -> Option<Real> {
        self.initial_condition
    }

    /// The resistor's `.noise` description (`resnoise.c`; `ressetup.c`
    /// defaults `KF = 0`, `AF = EF = LF = WF = 1`, `noisy = 1`), for `model`.
    #[must_use]
    pub fn resistor_noise(&self, model: &str) -> crate::devices::ResistorNoise {
        crate::devices::ResistorNoise {
            model: Some(model.to_owned()),
            ..self.resistor_noise_parameters()
        }
    }

    const fn resistor_noise_parameters(&self) -> crate::devices::ResistorNoise {
        crate::devices::ResistorNoise {
            noisy: self.noisy,
            kf: self.kf,
            af: self.af,
            ef: self.ef,
            area: self.noise_area,
            multiplicity: self.multiplicity,
            temperature: self.temperature,
            model: None,
        }
    }

    /// Effective scalar value including temperature, scale and multiplicity.
    ///
    /// `f=1+TC1*dT+TC2*dT^2`; R/L use `nominal*f*scale/m`, C uses
    /// `nominal*f*scale*m`. R's polynomial uses C's Horner evaluation order.
    /// Instance TEMP overrides circuit TEMP and DTEMP; model TNOM overrides context TNOM.
    /// A written TCE selects the exponential resistor law. Temperature
    /// subtraction uses Kelvin conversions just like the C setters.
    ///
    /// # Errors
    /// Invalid context, nonpositive/nonfinite factor, overflow, underflow to
    /// zero, or nonfinite resistor conductance. No cache/state is mutated.
    pub fn effective_value(&self, context: &ModelContext) -> SpiceResult<Real> {
        self.effective_value_for(self.nominal_value, context)
    }

    fn effective_value_with_nominal(
        &self,
        nominal: Real,
        context: &ModelContext,
    ) -> SpiceResult<Real> {
        valid_value(self.kind, nominal, &self.location)?;
        // An explicitly supplied model-backed resistance follows the same
        // C RESparam floor as the actual instance replacement.
        let nominal = if self.kind == PassiveKind::Resistor {
            crate::devices::rlc::clamp_resistance(nominal)
        } else {
            nominal
        };
        self.effective_value_for(nominal, context)
    }

    fn effective_value_for(&self, nominal: Real, context: &ModelContext) -> SpiceResult<Real> {
        self.effective_scaled(self.scaled_value_for(nominal, context)?)
    }

    fn effective_scaled(&self, scaled: Real) -> SpiceResult<Real> {
        let value = if self.kind == PassiveKind::Capacitor {
            scaled * self.multiplicity
        } else {
            scaled / self.multiplicity
        };
        valid_value(self.kind, value, &self.location)?;
        Ok(value)
    }

    fn checked_scaled(&self, context: &ModelContext) -> SpiceResult<Real> {
        let scaled = self.scaled_value(context)?;
        self.effective_scaled(scaled)?;
        Ok(scaled)
    }

    /// The temperature-adjusted and scaled value before multiplicity: C's
    /// `INDinduct` for an inductor, which `MUTtemp` (`ind/muttemp.c`) uses
    /// for `M = k sqrt(|L1 L2|)` while the inductor itself stamps
    /// `INDinduct / m`.
    ///
    /// # Errors
    /// As [`Self::effective_value`].
    pub fn coupling_value(&self, context: &ModelContext) -> SpiceResult<Real> {
        let scaled = self.scaled_value(context)?;
        valid_value(self.kind, scaled, &self.location)?;
        Ok(scaled)
    }

    fn scaled_value(&self, context: &ModelContext) -> SpiceResult<Real> {
        self.scaled_value_for(self.nominal_value, context)
    }

    fn temperature_point(&self, context: &ModelContext) -> SpiceResult<TemperaturePoint> {
        context.validate(&self.location)?;
        let explicit = self
            .temperature
            .map(|temperature| temperature_kelvin(temperature, &self.location))
            .transpose()?;
        let circuit = temperature_kelvin(context.temperature, &self.location)?;
        let temperature = TemperaturePoint::resolve(explicit, self.delta_temperature, circuit);
        positive(
            temperature.effective(),
            &self.location,
            "instance temperature (kelvin)",
        )?;
        Ok(temperature)
    }

    fn factor(&self, context: &ModelContext) -> SpiceResult<Real> {
        let temperature = self.temperature_point(context)?;
        let nominal = temperature_kelvin(
            self.nominal_temperature
                .unwrap_or(context.nominal_temperature),
            &self.location,
        )?;
        let dt = temperature.difference(nominal);
        finite(dt, &self.location, "temperature difference")?;
        let factor = if self.kind == PassiveKind::Resistor {
            self.coefficients.resistor_factor(dt, self.exponential)
        } else {
            self.coefficients.reactive(dt)
        };
        positive(factor, &self.location, "temperature factor")
    }

    fn scaled_value_for(&self, nominal_value: Real, context: &ModelContext) -> SpiceResult<Real> {
        finite(
            scaled(nominal_value, self.factor(context)?, self.scale),
            &self.location,
            "scaled value",
        )
    }

    /// Ordinary loads use C's conductance expression directly, so the
    /// multiplier is not rounded once into an effective resistance.
    fn resistor_conductance(&self, context: &ModelContext, ac: bool) -> SpiceResult<Real> {
        let nominal = if ac {
            self.ac_resistance.unwrap_or(self.nominal_value)
        } else {
            self.nominal_value
        };
        let adjusted = self.scaled_value_for(nominal, context)?;
        valid_value(self.kind, adjusted / self.multiplicity, &self.location)?;
        finite(
            conductance(adjusted, self.multiplicity),
            &self.location,
            "resistor conductance",
        )
    }

    fn validate_effective(&self, context: &ModelContext) -> SpiceResult<()> {
        self.effective_value(context)?;
        if self.kind == PassiveKind::Resistor {
            self.resistor_conductance(context, false)?;
            self.resistor_conductance(context, true)?;
        }
        Ok(())
    }
}

impl ResolvedModel<'_> {
    /// Project bounded passive model/instance setters without changing the AST.
    /// All setters are validated even when a higher-precedence scalar makes a
    /// geometry branch unnecessary. Unsupported names never get ignored.
    ///
    /// # Errors
    /// Wrong family/reference/ports, unknown or invalid setters, missing values
    /// or geometry, and nonfinite/nonpositive effective geometry/base values.
    /// Temperature/scale/multiplicity derivation is checked by `effective_value`.
    pub fn passive_parameters(&self, instance: &DeviceInstance) -> SpiceResult<PassiveParameters> {
        let kind = PassiveKind::from_family(self.family()).ok_or_else(|| {
            SpiceError::parse(
                self.card().location.clone(),
                "passive schema requires an R/C/L model",
            )
        })?;
        let primary = kind.principal().name;
        if instance.nodes.len() != 2
            || !instance
                .designator
                .eq_ignore_ascii_case(&kind.family().designator())
            || !instance
                .model
                .as_deref()
                .is_some_and(|name| name.eq_ignore_ascii_case(&self.card().name))
        {
            return Err(SpiceError::parse(
                instance.location.clone(),
                "instance does not reference this two-terminal passive model",
            ));
        }
        let model = self.parameters(model_schema(kind))?;
        let mut instance_values =
            instance_schema(kind).validate(&instance.parameters, &instance.location)?;
        let location = &instance.location;
        let prepared = PreparedModel::prepare(kind, &model);
        let base = if let Some(value) = optional(&instance_values, primary) {
            if kind == PassiveKind::Resistor {
                crate::devices::rlc::res_resist(&instance.name, value)
            } else {
                value
            }
        } else {
            prepared.nominal(&instance_values, location)?
        };
        if kind == PassiveKind::Resistor
            && let Some(written) = instance_values.get(primary)
            && let Some(source) = &written.location
        {
            instance_values.set(primary, base, kind.principal().unit, source.clone())?;
        }
        valid_value(kind, base, location)?;
        // ressetup.c: RESeffNoiseArea uses the instance geometry (L/W
        // defaulting to the model's) only when the instance gives L or W.
        let noise_area = if kind == PassiveKind::Resistor
            && (optional(&instance_values, "l").is_some()
                || optional(&instance_values, "w").is_some())
        {
            let length =
                optional(&instance_values, "l").unwrap_or_else(|| value(&model, "l", 10e-6));
            let width =
                optional(&instance_values, "w").unwrap_or_else(|| value(&model, "defw", 10e-6));
            let (effective_length, effective_width) = SheetGeometry {
                length,
                width,
                short: value(&model, "short", 0.),
                narrow: value(&model, "narrow", 0.),
                sheet: value(&model, "rsh", 0.),
            }
            .dimensions();
            effective_length.powf(value(&model, "lf", 1.0))
                * effective_width.powf(value(&model, "wf", 1.0))
        } else {
            1.0
        };
        Ok(PassiveParameters {
            kind,
            nominal_value: base,
            temperature: optional(&instance_values, "temp"),
            nominal_temperature: optional(&model, "tnom"),
            coefficients: Coefficients {
                linear: optional(&instance_values, "tc1")
                    .unwrap_or_else(|| value(&model, "tc1", 0.)),
                quadratic: optional(&instance_values, "tc2")
                    .unwrap_or_else(|| value(&model, "tc2", 0.)),
            },
            exponential: optional(&instance_values, "tce").or_else(|| optional(&model, "tce")),
            delta_temperature: optional(&instance_values, "dtemp"),
            ac_resistance: optional(&instance_values, "ac"),
            prepared,
            scale: value(&instance_values, "scale", 1.0),
            multiplicity: value(&instance_values, "m", 1.0),
            initial_condition: optional(&instance_values, "ic"),
            // `noisy` is an IF_INTEGER setter: floor(value + 0.5).
            noisy: optional(&instance_values, "noisy").is_none_or(|v| (v + 0.5).floor() != 0.0),
            kf: value(&model, "kf", 0.0),
            af: value(&model, "af", 1.0),
            ef: value(&model, "ef", 1.0),
            noise_area,
            location: location.clone(),
            model_values: model,
            instance_values,
        })
    }
}

const fn p(name: &'static str, unit: U, domain: D, default: Option<Real>) -> ScalarParameter {
    ScalarParameter {
        name,
        unit,
        domain,
        default,
    }
}
fn model_schema(kind: PassiveKind) -> &'static ScalarSchema<'static> {
    const R: &[ScalarParameter] = &[
        p("r", U::Ohm, D::Positive, None),
        p("rsh", U::Ohm, D::NonNegative, Some(0.0)),
        p("defw", U::Metre, D::Positive, Some(10e-6)),
        p("l", U::Metre, D::Positive, Some(10e-6)),
        p("narrow", U::Metre, D::NonNegative, Some(0.0)),
        p("short", U::Metre, D::NonNegative, Some(0.0)),
        p("tc1", U::InverseKelvin, D::Finite, Some(0.0)),
        p("tc2", U::InverseKelvinSquared, D::Finite, Some(0.0)),
        p("tce", U::InverseKelvin, D::Finite, None),
        p("tnom", U::Celsius, D::Temperature, None),
        p("kf", U::Dimensionless, D::Finite, Some(0.0)),
        p("af", U::Dimensionless, D::Finite, Some(1.0)),
        p("ef", U::Dimensionless, D::Finite, Some(1.0)),
        p("lf", U::Dimensionless, D::Finite, Some(1.0)),
        p("wf", U::Dimensionless, D::Finite, Some(1.0)),
    ];
    const C: &[ScalarParameter] = &[
        p("cap", U::Farad, D::Positive, None),
        p("cj", U::FaradPerSquareMetre, D::NonNegative, Some(0.0)),
        p("cjsw", U::FaradPerMetre, D::NonNegative, Some(0.0)),
        p("defw", U::Metre, D::Positive, Some(10e-6)),
        p("narrow", U::Metre, D::NonNegative, Some(0.0)),
        p("short", U::Metre, D::NonNegative, Some(0.0)),
        p("del", U::Metre, D::NonNegative, None),
        p("thick", U::Metre, D::Positive, None),
        p("di", U::Dimensionless, D::NonNegative, None),
        p("tc1", U::InverseKelvin, D::Finite, Some(0.0)),
        p("tc2", U::InverseKelvinSquared, D::Finite, Some(0.0)),
        p("tnom", U::Celsius, D::Temperature, None),
    ];
    const L: &[ScalarParameter] = &[
        p("ind", U::Henry, D::Positive, None),
        p("csect", U::SquareMetre, D::Positive, None),
        p("dia", U::Metre, D::Positive, None),
        p("length", U::Metre, D::Positive, None),
        p("nt", U::Dimensionless, D::Positive, None),
        p("mu", U::Dimensionless, D::Positive, Some(1.)),
        p("tc1", U::InverseKelvin, D::Finite, Some(0.0)),
        p("tc2", U::InverseKelvinSquared, D::Finite, Some(0.0)),
        p("tnom", U::Celsius, D::Temperature, None),
    ];
    static R_SCHEMA: ScalarSchema<'static> = ScalarSchema::new(R);
    static C_SCHEMA: ScalarSchema<'static> = ScalarSchema::new(C);
    static L_SCHEMA: ScalarSchema<'static> = ScalarSchema::new(L);
    match kind {
        PassiveKind::Resistor => &R_SCHEMA,
        PassiveKind::Capacitor => &C_SCHEMA,
        PassiveKind::Inductor => &L_SCHEMA,
    }
}
fn instance_schema(kind: PassiveKind) -> &'static ScalarSchema<'static> {
    const TEMP: ScalarParameter = p("temp", U::Celsius, D::Temperature, None);
    const DTEMP: ScalarParameter = p("dtemp", U::Kelvin, D::Finite, None);
    const TC1: ScalarParameter = p("tc1", U::InverseKelvin, D::Finite, None);
    const TC2: ScalarParameter = p("tc2", U::InverseKelvinSquared, D::Finite, None);
    const M: ScalarParameter = p("m", U::Dimensionless, D::Positive, Some(1.));
    const SCALE: ScalarParameter = p("scale", U::Dimensionless, D::Positive, Some(1.));
    const L: ScalarParameter = p("l", U::Metre, D::Positive, None);
    const W: ScalarParameter = p("w", U::Metre, D::Positive, None);
    const R: &[ScalarParameter] = &[
        PassiveKind::Resistor.principal(),
        TEMP,
        DTEMP,
        TC1,
        TC2,
        M,
        SCALE,
        L,
        W,
        p("noisy", U::Dimensionless, D::Finite, None),
        p("tce", U::InverseKelvin, D::Finite, None),
        p("ac", U::Ohm, D::NonZero, None),
    ];
    const C: &[ScalarParameter] = &[
        PassiveKind::Capacitor.principal(),
        TEMP,
        DTEMP,
        TC1,
        TC2,
        M,
        SCALE,
        L,
        W,
        p("ic", U::Volt, D::Finite, None),
    ];
    const I: &[ScalarParameter] = &[
        PassiveKind::Inductor.principal(),
        TEMP,
        DTEMP,
        TC1,
        TC2,
        M,
        SCALE,
        p("ic", U::Ampere, D::Finite, None),
        p("nt", U::Dimensionless, D::Positive, None),
    ];
    static R_SCHEMA: ScalarSchema<'static> = ScalarSchema::new(R);
    static C_SCHEMA: ScalarSchema<'static> = ScalarSchema::new(C);
    static I_SCHEMA: ScalarSchema<'static> = ScalarSchema::new(I);
    match kind {
        PassiveKind::Resistor => &R_SCHEMA,
        PassiveKind::Capacitor => &C_SCHEMA,
        PassiveKind::Inductor => &I_SCHEMA,
    }
}
fn optional(values: &ScalarValues, name: &str) -> Option<Real> {
    values.get(name).map(|v| v.value)
}
fn value(values: &ScalarValues, name: &str, default: Real) -> Real {
    optional(values, name).unwrap_or(default)
}
fn finite(value: Real, location: &SourceLoc, name: &str) -> SpiceResult<Real> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err(SpiceError::parse(
            location.clone(),
            format!("nonfinite/overflow {name}"),
        ))
    }
}
fn positive(value: Real, location: &SourceLoc, name: &str) -> SpiceResult<Real> {
    finite(value, location, name)?;
    if value > 0.0 {
        Ok(value)
    } else {
        Err(SpiceError::parse(
            location.clone(),
            format!("{name} must be positive"),
        ))
    }
}
fn valid_value(kind: PassiveKind, value: Real, location: &SourceLoc) -> SpiceResult<()> {
    finite(value, location, "effective passive value")?;
    if kind == PassiveKind::Resistor {
        if value == 0.0 || !(1.0 / value).is_finite() {
            return Err(SpiceError::parse(
                location.clone(),
                "resistance must be nonzero with finite conductance",
            ));
        }
    } else if value < 0.0 {
        return Err(SpiceError::parse(
            location.clone(),
            "effective passive value must be nonnegative",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn literal_recipe_retains_typed_given_values() {
        for (kind, unit, initial_unit) in [
            (PassiveKind::Resistor, U::Ohm, None),
            (PassiveKind::Capacitor, U::Farad, Some(U::Volt)),
            (PassiveKind::Inductor, U::Henry, Some(U::Ampere)),
        ] {
            let recipe = PassiveParameters::literal(kind, 2., initial_unit.map(|_| 3.)).unwrap();
            let scalar = recipe.instance_values().get(kind.principal().name).unwrap();
            assert_eq!(scalar.unit, unit);
            assert_eq!(scalar.value.to_bits(), 2_f64.to_bits());
            assert!(scalar.location.is_some());
            let replacement = recipe.with_nominal_value(4.).unwrap();
            let replaced = replacement
                .instance_values()
                .get(kind.principal().name)
                .unwrap();
            assert_eq!(replaced.unit, unit);
            assert_eq!(replaced.value.to_bits(), 4_f64.to_bits());
            assert!(replaced.location.is_some());
            assert_eq!(replacement.nominal_value().to_bits(), 4_f64.to_bits());
            assert_eq!(recipe.nominal_value().to_bits(), 2_f64.to_bits());
            assert_eq!(
                recipe.instance_values().get(kind.principal().name),
                Some(scalar)
            );
            let initial = recipe.instance_values().get("ic");
            assert_eq!(replacement.instance_values().get("ic"), initial);
            assert_eq!(initial.map(|v| v.unit), initial_unit);
            if let Some(initial) = initial {
                assert_eq!(initial.value.to_bits(), 3_f64.to_bits());
                assert!(initial.location.is_some());
            }
        }
    }
}
