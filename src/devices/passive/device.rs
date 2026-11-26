//! Contextual input wrapper delegates effective values to existing R/C/L stamps.
use super::{PassiveKind, PassiveParameters};
use crate::devices::models::{ModelContext, ResolvedModel};
use crate::devices::noise::{DeviceNoise, NoiseContext, NoiseSource};
use crate::devices::rlc::{
    QUANTITY, REACTIVE_STATE_COUNT, ReactiveLoad, assemble_capacitor_ac, assemble_conductance,
    assemble_inductor_ac, resistor_noise, stamp_conductance,
};
use crate::devices::{
    Device, InductanceValue, LinearContext, Resistor, ResistorMetadata, ResistorNoise,
    ResistorOrigin, StampContext, StorageElement, StorageKind,
};
use crate::netlist::ast::DeviceInstance;
use crate::primitives::{NodeAppend, NodeId, Real, SpiceError, SpiceResult};

#[derive(Debug, Clone)]
struct ModelPassive {
    name: String,
    /// The model card's name, for C's `.noise` visiting order.
    model: Option<String>,
    // Programmatic literal resistors may carry an independent noise recipe.
    literal_noise: Option<Box<ResistorNoise>>,
    terminals: [NodeId; 2],
    parameters: PassiveParameters,
}

impl ModelPassive {
    const fn storage_kind(&self) -> Option<StorageKind> {
        match self.parameters.kind {
            PassiveKind::Capacitor => Some(StorageKind::Capacitor),
            PassiveKind::Inductor => Some(StorageKind::Inductor),
            PassiveKind::Resistor => None,
        }
    }
    fn reactive_load(&self, value: Real) -> ReactiveLoad<'_> {
        ReactiveLoad {
            name: &self.name,
            terminals: self.terminals,
            value,
            initial: self.parameters.initial_condition(),
        }
    }
}
impl Device for ModelPassive {
    fn validate_dc_inputs(&self, context: &crate::devices::ModelContext) -> SpiceResult<()> {
        match self.parameters.kind {
            PassiveKind::Resistor => self
                .parameters
                .resistor_conductance(context, false)
                .map(|_| ()),
            PassiveKind::Capacitor | PassiveKind::Inductor => {
                self.parameters.effective_value(context).map(|_| ())
            }
        }
    }

    /// `ressetup.c`, `capsetup.c` or `indsetup.c` by family.
    fn setup_matrix(
        &self,
        setup: &mut crate::devices::setup::MatrixSetup<'_>,
    ) -> SpiceResult<bool> {
        let [positive, negative] = self.terminals;
        match self.parameters.kind {
            PassiveKind::Resistor | PassiveKind::Capacitor => {
                setup.conductance(positive, negative);
            }
            PassiveKind::Inductor => {
                let branch = setup.branch(0)?;
                setup.inductor(positive, negative, branch);
            }
        }
        Ok(true)
    }
    fn observation_parameter(
        &self,
        keyword: &str,
        context: &ModelContext,
    ) -> SpiceResult<Option<Real>> {
        let p = &self.parameters;
        Ok(match keyword {
            "r" | "resistance" if p.kind == PassiveKind::Resistor => Some(p.scaled_value(context)?),
            "g" | "conductance" if p.kind == PassiveKind::Resistor => {
                Some(p.resistor_conductance(context, false)?)
            }
            "c" | "cap" | "capacitance" if p.kind == PassiveKind::Capacitor => {
                Some(p.effective_value(context)?)
            }
            "l" | "inductance" if p.kind == PassiveKind::Inductor => {
                Some(p.coupling_value(context)?)
            }
            "m" => Some(p.multiplicity),
            "scale" => Some(p.scale),
            "temp" => Some(p.temperature.unwrap_or(context.temperature)),
            "tc1" => Some(p.coefficients.linear),
            "tc2" => Some(p.coefficients.quadratic),
            "ic" => Some(p.initial_condition.unwrap_or(0.)),
            _ => p.instance_values.get(keyword).map(|v| v.value),
        })
    }
    fn parameter_geometry_scale(
        &self,
        key: crate::devices::sweep::ParameterKey,
    ) -> crate::devices::sweep::GeometryScale {
        use crate::devices::sweep::{GeometryScale, ParameterKey};
        if matches!(
            self.parameters.kind,
            PassiveKind::Resistor | PassiveKind::Capacitor
        ) && matches!(key, ParameterKey::Native("l" | "w"))
        {
            GeometryScale::Length
        } else {
            GeometryScale::Identity
        }
    }

    fn instance_parameter(&self, keyword: &str) -> Option<&'static str> {
        let keyword = if keyword.eq_ignore_ascii_case("r") {
            "resistance"
        } else if keyword.eq_ignore_ascii_case("c") || keyword.eq_ignore_ascii_case("cap") {
            "capacitance"
        } else {
            keyword
        };
        super::instance_schema(self.parameters.kind)
            .parameters()
            .iter()
            .find(|d| d.name != "noisy" && d.name.eq_ignore_ascii_case(keyword))
            .map(|d| d.name)
    }

    fn with_instance_parameter(
        &self,
        parameter: &str,
        value: Real,
        context: &ModelContext,
    ) -> SpiceResult<Box<dyn Device>> {
        let mut replacement = self.clone();
        replacement.parameters = self.parameters.swept(parameter, value, context)?;
        if let Some(noise) = replacement.literal_noise.as_deref_mut() {
            match parameter {
                "temp" => noise.temperature = Some(value),
                "m" => noise.multiplicity = value,
                _ => {}
            }
        }
        Ok(Box::new(replacement))
    }

    /// Linear in `.disto`: C gives this device no distortion routine
    /// (`DEVdisto = NULL`, `res`/`cap`/`ind` `*init.c`), so it enters only through its
    /// small-signal matrix.
    fn distortion(
        &self,
        _context: &crate::devices::distortion::DistortionContext<'_>,
    ) -> crate::primitives::SpiceResult<crate::devices::distortion::DeviceDistortion> {
        Ok(crate::devices::distortion::DeviceDistortion::Linear)
    }

    fn name(&self) -> &str {
        &self.name
    }
    fn designator(&self) -> char {
        self.parameters.family().designator()
    }
    fn terminals(&self) -> &[NodeId] {
        &self.terminals
    }
    fn branch_currents(&self) -> usize {
        usize::from(self.parameters.kind == PassiveKind::Inductor)
    }
    fn expression_current_branch(&self) -> Option<usize> {
        (self.parameters.kind == PassiveKind::Inductor).then_some(0)
    }
    fn state_count(&self) -> usize {
        match self.parameters.kind {
            PassiveKind::Capacitor | PassiveKind::Inductor => REACTIVE_STATE_COUNT,
            PassiveKind::Resistor => 0,
        }
    }
    /// The scalar capacitor/inductor's charge/flux slot (`captrunc.c`,
    /// `indtrunc.c`): the state layout is the delegated scalar device's.
    fn truncation_slot(&self) -> Option<usize> {
        self.storage_kind().map(|_| QUANTITY)
    }
    /// The effective (temperature/TC/scale/`m`-adjusted) value and the
    /// instance `ic=`, so `uic` seeds model-backed C/L exactly as literal ones.
    fn storage_element(&self, context: &ModelContext) -> Option<SpiceResult<StorageElement>> {
        let kind = self.storage_kind()?;
        Some(
            self.parameters
                .effective_value(context)
                .map(|value| StorageElement {
                    kind,
                    value,
                    initial: self.parameters.initial_condition(),
                }),
        )
    }
    fn stamp(&self, context: &mut StampContext<'_>) -> SpiceResult<()> {
        let model = ModelContext::new(context.temperature, context.nominal_temperature);
        match self.parameters.kind {
            PassiveKind::Resistor => stamp_conductance(
                self.terminals,
                self.parameters.resistor_conductance(&model, false)?,
                context,
            ),
            PassiveKind::Capacitor => self
                .reactive_load(self.parameters.effective_value(&model)?)
                .stamp_capacitor(context),
            PassiveKind::Inductor => self
                .reactive_load(self.parameters.effective_value(&model)?)
                .stamp_inductor(context),
        }
    }

    fn assemble_linear(&self, context: &mut LinearContext<'_>) -> SpiceResult<()> {
        match self.parameters.kind {
            PassiveKind::Resistor => assemble_conductance(
                self.terminals,
                self.parameters
                    .resistor_conductance(context.model_context, false)?,
                context,
            ),
            PassiveKind::Capacitor => self
                .reactive_load(self.parameters.effective_value(context.model_context)?)
                .assemble_capacitor(context),
            PassiveKind::Inductor => self
                .reactive_load(self.parameters.effective_value(context.model_context)?)
                .assemble_inductor(context),
        }
    }

    fn assemble_small_signal(
        &self,
        context: &mut LinearContext<'_>,
        _bias: &crate::maths::Vector,
    ) -> SpiceResult<()> {
        let p = &self.parameters;
        match p.kind {
            PassiveKind::Resistor => assemble_conductance(
                self.terminals,
                p.resistor_conductance(context.model_context, true)?,
                context,
            ),
            PassiveKind::Capacitor => {
                let value = p.checked_scaled(context.model_context)?;
                context.system.has_initial_conditions |= p.initial_condition.is_some();
                assemble_capacitor_ac(self.terminals, value, p.multiplicity, context)
            }
            PassiveKind::Inductor => {
                let value = p.checked_scaled(context.model_context)?;
                context.system.has_initial_conditions |= p.initial_condition.is_some();
                assemble_inductor_ac(self.terminals, value, p.multiplicity, context)
            }
        }
    }
    /// `resnoise.c` for a resistor (including geometry, multiplier and temp);
    /// C and L have no noise generators. Validate the effective value in
    /// every family, as for stamps and linear assembly.
    fn noise(
        &self,
        context: &NoiseContext<'_>,
        sources: &mut Vec<NoiseSource>,
    ) -> SpiceResult<DeviceNoise<'_>> {
        self.parameters.effective_value(context.model_context)?;
        Ok(if self.parameters.kind == PassiveKind::Resistor {
            let conductance = self
                .parameters
                .resistor_conductance(context.model_context, false)?;
            let temperature = self
                .parameters
                .temperature_point(context.model_context)?
                .effective();
            let model = self.model.as_deref().or_else(|| {
                self.literal_noise
                    .as_deref()
                    .and_then(|noise| noise.model.as_deref())
            });
            let mut evaluate = |noise: &ResistorNoise| {
                resistor_noise(
                    self.terminals,
                    conductance,
                    noise,
                    model,
                    context,
                    if self.literal_noise.is_some()
                        && self.parameters.temperature.is_none()
                        && self.parameters.delta_temperature.is_none()
                    {
                        None
                    } else {
                        Some(temperature)
                    },
                    sources,
                )
            };
            match self.literal_noise.as_deref() {
                Some(noise) => evaluate(noise),
                None => evaluate(&self.parameters.resistor_noise_parameters()),
            }
        } else {
            DeviceNoise::Noiseless
        })
    }
    fn inductance(&self, context: &ModelContext) -> Option<SpiceResult<InductanceValue>> {
        (self.parameters.kind == PassiveKind::Inductor).then(|| {
            Ok(InductanceValue {
                effective: self.parameters.effective_value(context)?,
                coupling_base: self.parameters.coupling_value(context)?,
                initial: self.parameters.initial_condition().unwrap_or(0.),
            })
        })
    }
    fn resistor_metadata(&self) -> Option<ResistorMetadata> {
        (self.parameters.kind == PassiveKind::Resistor).then(|| ResistorMetadata {
            origin: if self.model.is_some() {
                ResistorOrigin::ModelBacked
            } else {
                ResistorOrigin::Literal
            },
            supplied: self.parameters.nominal_value(),
            multiplicity: self.parameters.multiplicity(),
        })
    }
    fn resistor_effective(&self, supplied: Real, context: &ModelContext) -> SpiceResult<Real> {
        if self.parameters.kind != PassiveKind::Resistor {
            return Err(SpiceError::circuit(format!(
                "{} is not a resistor",
                self.name
            )));
        }
        self.parameters
            .effective_value_with_nominal(supplied, context)
    }

    /// `.sens`: C's passive records from the card's model and instance
    /// setters ([`crate::devices::sensitivity`]).
    fn sensitivity(
        &self,
        context: &ModelContext,
    ) -> SpiceResult<Box<dyn crate::devices::sensitivity::DeviceSensitivity + '_>> {
        use crate::devices::sensitivity::{
            PassiveInputs, ReactiveKind, ReactiveSensitivity, ResistorSensitivity,
        };
        let inputs = PassiveInputs::Validated {
            name: self.model.as_deref(),
            model: self.parameters.model_values(),
            instance: self.parameters.instance_values(),
        };
        match self.parameters.kind {
            PassiveKind::Resistor => Ok(Box::new(ResistorSensitivity::new(
                &self.name,
                self.terminals,
                inputs,
                context,
            ))),
            kind @ (PassiveKind::Capacitor | PassiveKind::Inductor) => {
                let kind = if kind == PassiveKind::Capacitor {
                    ReactiveKind::Capacitor
                } else {
                    ReactiveKind::Inductor
                };
                Ok(Box::new(ReactiveSensitivity::new(
                    kind,
                    &self.name,
                    self.terminals,
                    inputs,
                    context,
                )))
            }
        }
    }

    /// Pole-zero load: C `respzld.c`, `cappzld.c`, `indpzld.c` equals the AC load with `s` for `j omega`.
    fn assemble_pole_zero(
        &self,
        context: &mut crate::devices::linear::LinearContext<'_>,
        bias: &crate::maths::Vector,
    ) -> crate::primitives::SpiceResult<()> {
        self.assemble_small_signal(context, bias)
    }
}

pub(in crate::devices) fn instantiate(
    instance: &DeviceInstance,
    nodes: &mut NodeAppend<'_>,
    model: &ResolvedModel<'_>,
    context: &ModelContext,
) -> SpiceResult<Box<dyn Device>> {
    let parameters = model.passive_parameters(instance, context.geometry_scale)?;
    // The schema validates the IC and effective_value validates the scalar
    // before any terminals are interned.
    parameters.validate_effective(context)?;
    let terminals = nodes.intern_array(&instance.nodes)?;
    let device = ModelPassive {
        name: instance.name.clone(),
        model: Some(model.card().name.clone()),
        literal_noise: None,
        terminals,
        parameters,
    };
    Ok(Box::new(device))
}

/// Per-point literal R/C/L replacement, retaining temperature and multiplier setters.
pub(in crate::devices) fn swept_literal(
    name: &str,
    terminals: [NodeId; 2],
    kind: PassiveKind,
    value: Real,
    ic: Option<Real>,
    change: (&str, Real),
    context: &ModelContext,
) -> SpiceResult<Box<dyn Device>> {
    Ok(Box::new(ModelPassive {
        name: name.to_owned(),
        model: None,
        literal_noise: None,
        terminals,
        parameters: PassiveParameters::literal(kind, value, ic)?
            .swept(change.0, change.1, context)?,
    }))
}

/// A literal resistor can acquire contextual setters in a sweep without
/// losing the noise description of the original device.
pub(in crate::devices) fn swept_resistor(
    resistor: &Resistor,
    change: (&str, Real),
    context: &ModelContext,
) -> SpiceResult<Box<dyn Device>> {
    let [positive, negative] = resistor.terminals() else {
        return Err(SpiceError::circuit("invalid resistor terminals"));
    };
    let mut noise = resistor.noise_parameters().clone();
    match change.0 {
        "temp" => noise.temperature = Some(change.1),
        "m" => noise.multiplicity = change.1,
        _ => {}
    }
    Ok(Box::new(ModelPassive {
        name: resistor.name().to_owned(),
        model: None,
        literal_noise: Some(Box::new(noise)),
        terminals: [*positive, *negative],
        parameters: PassiveParameters::literal(PassiveKind::Resistor, resistor.resistance(), None)?
            .swept(change.0, change.1, context)?,
    }))
}
