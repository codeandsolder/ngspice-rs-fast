//! Explicit native `LTspice` idealized diode; standard D models are untouched.
//! The source front end selects this family instead of translating it to an
//! approximate XSPICE code model.

mod law;

use crate::maths::Vector;
use crate::netlist::ast::{DeviceInstance, ParameterAssignment, ParameterKind};
use crate::primitives::{NodeAppend, NodeId, Real, SourceLoc, SpiceError, SpiceResult};

use super::{
    Device, ModelContext, ResolvedModel, StampContext,
    limiting::{Linearization, previous_finite},
    linear::LinearContext,
    noise::{
        CELSIUS_TO_KELVIN, DeviceNoise, FlickerLaw, NoiseContext, NoiseFamily, NoiseKind,
        NoiseSource,
    },
    schema::finite_literal,
};
use law::{Branch, Point};

pub(super) const MODEL_FAMILY: &str = "ltspice_ideal_diode";

#[derive(Clone, Copy, Debug)]
struct Model {
    ron: Real,
    roff: Option<Real>,
    vfwd: Real,
    vrev: Option<Real>,
    rrev: Option<Real>,
    ilimit: Option<Real>,
    revilimit: Option<Real>,
    epsilon: Real,
    revepsilon: Real,
    kf: Real,
    af: Real,
}

impl Default for Model {
    fn default() -> Self {
        Self {
            ron: 1.,
            roff: None,
            vfwd: 0.,
            vrev: None,
            rrev: None,
            ilimit: None,
            revilimit: None,
            epsilon: 0.,
            revepsilon: 0.,
            kf: 0.,
            af: 1.,
        }
    }
}

fn invalid(location: &SourceLoc, name: &str, value: Real) -> SpiceError {
    SpiceError::parse(
        location.clone(),
        format!("invalid ideal diode {name}={value}"),
    )
}

fn positive(value: Real, location: &SourceLoc, name: &str) -> SpiceResult<Real> {
    if value.is_finite() && value > 0. {
        Ok(value)
    } else {
        Err(invalid(location, name, value))
    }
}

fn nonnegative(value: Real, location: &SourceLoc, name: &str) -> SpiceResult<Real> {
    if value.is_finite() && value >= 0. {
        Ok(value)
    } else {
        Err(invalid(location, name, value))
    }
}

fn unsupported(field: &ParameterAssignment) -> SpiceError {
    SpiceError::Unsupported {
        feature: format!("LTspice ideal diode parameter {}", field.name),
        location: Some(field.location.clone()),
    }
}

impl Model {
    fn read(model: &ResolvedModel<'_>) -> SpiceResult<Self> {
        let mut result = Self::default();
        for field in &model.card().parameters {
            let value = finite_literal(field)?;
            let location = &field.location;
            let name = field.name.as_str();
            match name {
                "ron" => result.ron = positive(value, location, name)?,
                "roff" => result.roff = Some(positive(value, location, name)?),
                "vfwd" => result.vfwd = nonnegative(value, location, name)?,
                "vrev" => result.vrev = Some(nonnegative(value, location, name)?),
                "rrev" => result.rrev = Some(positive(value, location, name)?),
                "ilimit" => result.ilimit = Some(positive(value, location, name)?),
                "revilimit" => result.revilimit = Some(positive(value, location, name)?),
                "epsilon" => result.epsilon = nonnegative(value, location, name)?,
                "revepsilon" => result.revepsilon = nonnegative(value, location, name)?,
                "kf" => result.kf = nonnegative(value, location, name)?,
                "af" => result.af = value,
                "level" => {} // Ordered level resolution already selected this family.
                // Semiconductor-only parameters have no effect on idealized conduction.
                "is" | "rs" | "n" | "tt" | "cjo" | "vj" | "m" | "eg" | "xti" | "fc" | "bv"
                | "ibv" | "isr" | "nr" | "ikf" | "tikf" | "tnom" | "trs1" | "trs2" => {}
                // Documented schematic ratings, not electrical setters.
                "vpk" | "ipk" | "iave" | "irms" | "diss" => {}
                _ => return Err(unsupported(field)),
            }
        }
        Ok(result)
    }

    fn prepare(self, gmin: Real) -> SpiceResult<Prepared> {
        let off = self.roff.map_or(gmin, Real::recip);
        if !gmin.is_finite()
            || gmin < 0.
            || !off.is_finite()
            || off < 0.
            || !self.ron.recip().is_finite()
            || self.rrev.is_some_and(|value| !value.recip().is_finite())
        {
            return Err(SpiceError::circuit(
                "nonrepresentable ideal diode conductance",
            ));
        }
        let forward = Branch::new(self.vfwd, off, self.ron.recip(), self.epsilon, self.ilimit)?;
        let reverse = self
            .vrev
            .map(|threshold| {
                Branch::new(
                    threshold,
                    off,
                    self.rrev.unwrap_or(self.ron).recip(),
                    self.revepsilon,
                    self.revilimit,
                )
            })
            .transpose()?;
        Ok(Prepared {
            off,
            floor: gmin / 1000.,
            forward,
            reverse,
        })
    }
}

#[derive(Clone, Copy, Debug)]
struct Prepared {
    off: Real,
    floor: Real,
    forward: Branch,
    reverse: Option<Branch>,
}

impl Prepared {
    fn has_limit(self) -> bool {
        self.forward.has_limit() || self.reverse.is_some_and(Branch::has_limit)
    }

    fn limit_trial(self, raw: Real, previous: Real) -> Real {
        if raw >= 0. {
            self.forward.limit_trial(raw, previous)
        } else {
            self.reverse
                .map_or(raw, |reverse| -reverse.limit_trial(-raw, -previous))
        }
    }

    fn point(self, voltage: Real) -> Point {
        if voltage > 0. || (voltage == 0. && self.forward.starts_at_zero()) {
            self.forward.point(voltage)
        } else if let Some(reverse) = self.reverse {
            let point = reverse.point(-voltage);
            Point {
                current: -point.current,
                conductance: point.conductance,
            }
        } else {
            Point {
                current: voltage * self.off,
                conductance: self.off,
            }
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct Instance {
    multiplier: Real,
    series: Real,
    area: Real,
    temperature: Option<Real>,
}

impl Default for Instance {
    fn default() -> Self {
        Self {
            multiplier: 1.,
            series: 1.,
            area: 1.,
            temperature: None,
        }
    }
}

impl Instance {
    fn set(&mut self, key: &str, value: Real, location: &SourceLoc) -> SpiceResult<()> {
        match key {
            "m" => self.multiplier = positive(value, location, key)?,
            "n" => self.series = positive(value, location, key)?,
            "area" => self.area = positive(value, location, key)?,
            "temp" if value.is_finite() && value > -273.15 => self.temperature = Some(value),
            _ => return Err(invalid(location, key, value)),
        }
        Ok(())
    }

    fn read(instance: &DeviceInstance) -> SpiceResult<Self> {
        let mut result = Self::default();
        for field in &instance.parameters {
            if field.name == "off" && field.kind == ParameterKind::Flag {
                continue; // No stored on/off state in the idealized family.
            }
            result.set(&field.name, finite_literal(field)?, &field.location)?;
        }
        Ok(result)
    }
}

/// Topology-bound memoryless idealized diode.
#[derive(Clone, Debug)]
pub(super) struct IdealDiode {
    name: String,
    model_name: String,
    location: SourceLoc,
    terminals: [NodeId; 2],
    model: Model,
    instance: Instance,
    prepared: Prepared,
    gmin: Real,
}

pub(super) fn instantiate(
    instance: &DeviceInstance,
    nodes: &mut NodeAppend<'_>,
    model: &ResolvedModel<'_>,
    context: &ModelContext,
) -> SpiceResult<Box<dyn Device>> {
    let inputs = Instance::read(instance)?;
    let law = Model::read(model)?;
    let prepared = law.prepare(context.gmin)?;
    let terminals = nodes.intern_array(&instance.nodes)?;
    Ok(Box::new(IdealDiode {
        name: instance.name.clone(),
        model_name: model.card().name.clone(),
        location: instance.location.clone(),
        terminals,
        model: law,
        instance: inputs,
        prepared,
        gmin: context.gmin,
    }))
}

impl IdealDiode {
    fn prepared_at(&self, context: &ModelContext) -> SpiceResult<Prepared> {
        if context.gmin.to_bits() == self.gmin.to_bits() {
            Ok(self.prepared)
        } else {
            self.model.prepare(context.gmin)
        }
    }

    fn point(&self, voltage: Real, context: &ModelContext) -> SpiceResult<Point> {
        self.scaled_point(voltage, self.prepared_at(context)?)
    }

    fn scaled_point(&self, voltage: Real, prepared: Prepared) -> SpiceResult<Point> {
        let local_voltage = voltage / self.instance.series;
        let local = prepared.point(local_voltage);
        let current = prepared
            .floor
            .mul_add(local_voltage, local.current * self.instance.multiplier);
        let conductance = law::ratio_product(
            local.conductance,
            self.instance.multiplier,
            self.instance.series,
        ) + prepared.floor / self.instance.series;
        if voltage.is_finite() && current.is_finite() && conductance.is_finite() {
            Ok(Point {
                current,
                conductance,
            })
        } else {
            Err(SpiceError::circuit(format!(
                "{}: nonrepresentable ideal diode bias",
                self.name
            )))
        }
    }
}

impl Device for IdealDiode {
    fn name(&self) -> &str {
        &self.name
    }
    fn model_name(&self) -> Option<&str> {
        Some(&self.model_name)
    }
    fn designator(&self) -> char {
        'd'
    }
    fn terminals(&self) -> &[NodeId] {
        &self.terminals
    }
    fn is_nonlinear(&self) -> bool {
        true
    }
    fn setup_matrix(&self, setup: &mut super::setup::MatrixSetup<'_>) -> SpiceResult<bool> {
        setup.conductance(self.terminals[0], self.terminals[1]);
        Ok(true)
    }
    fn validate_dc_inputs(&self, context: &ModelContext) -> SpiceResult<()> {
        self.prepared_at(context).map(|_| ())
    }
    fn state_count(&self) -> usize {
        usize::from(self.prepared.has_limit())
    }
    fn stamp(&self, context: &mut StampContext<'_>) -> SpiceResult<()> {
        let [a, c] = self.terminals;
        let raw = context.node_voltage(a) - context.node_voltage(c);
        let mut voltage = raw;
        let prepared = self.prepared_at(&context.model_context())?;
        if prepared.has_limit() {
            let states = &mut context.states;
            match Linearization::classify(states) {
                Linearization::Initial | Linearization::Limited => {
                    let previous = previous_finite(states, 0).unwrap_or(0.);
                    let local = prepared
                        .limit_trial(raw / self.instance.series, previous / self.instance.series);
                    if local.to_bits() != (raw / self.instance.series).to_bits() {
                        voltage = local * self.instance.series;
                        states.report_nonconvergence()?;
                    }
                }
                _ => {}
            }
            states.set(0, voltage)?;
        }
        let point = self.scaled_point(voltage, prepared)?;
        super::rlc::stamp_conductance(self.terminals, point.conductance, context)?;
        let rhs = point.conductance.mul_add(voltage, -point.current);
        context.stamp_rhs(a, rhs)?;
        context.stamp_rhs(c, -rhs)
    }
    fn assemble_small_signal(
        &self,
        context: &mut LinearContext<'_>,
        bias: &Vector,
    ) -> SpiceResult<()> {
        let voltage = |node| {
            context
                .unknowns
                .node_row(node)
                .and_then(|row| bias.get(row))
                .unwrap_or(0.)
        };
        let point = self.point(
            voltage(self.terminals[0]) - voltage(self.terminals[1]),
            context.model_context,
        )?;
        context.nodal(self.terminals, point.conductance, false)
    }
    fn assemble_pole_zero(
        &self,
        context: &mut LinearContext<'_>,
        bias: &Vector,
    ) -> SpiceResult<()> {
        self.assemble_small_signal(context, bias)
    }
    fn assemble_nonlinear_charge(
        &self,
        _: &mut super::traits::ChargeContext<'_>,
    ) -> SpiceResult<()> {
        // Idealized conduction has no stored charge.
        Ok(())
    }
    fn noise(
        &self,
        context: &NoiseContext<'_>,
        sources: &mut Vec<NoiseSource>,
    ) -> SpiceResult<DeviceNoise<'_>> {
        let [a, c] = self.terminals;
        let point = self.point(
            context.voltage(a) - context.voltage(c),
            context.model_context,
        )?;
        let temperature = self
            .instance
            .temperature
            .map_or_else(|| context.circuit_kelvin(), |t| t + CELSIUS_TO_KELVIN);
        let current = point.current.abs();
        let flicker = law::flicker_numerator(self.model.kf, self.model.af, current);
        if !flicker.is_finite() {
            return Err(SpiceError::circuit(
                "nonrepresentable ideal diode flicker noise",
            ));
        }
        sources.extend([
            NoiseSource::new(
                "_rs",
                self.terminals,
                NoiseKind::Thermal {
                    conductance: point.conductance,
                    temperature,
                },
            ),
            NoiseSource::new(
                "_id",
                self.terminals,
                NoiseKind::Shot {
                    current: current / self.instance.series,
                },
            ),
            NoiseSource::new(
                "_1overf",
                self.terminals,
                NoiseKind::Flicker {
                    numerator: flicker,
                    law: FlickerLaw::PerHertz,
                },
            ),
        ]);
        Ok(DeviceNoise::Sources {
            family: NoiseFamily::Diode,
            model: Some(&self.model_name),
            total: true,
        })
    }
    fn observation_parameter(
        &self,
        keyword: &str,
        context: &ModelContext,
    ) -> SpiceResult<Option<Real>> {
        Ok(match keyword {
            "area" => Some(self.instance.area),
            "m" => Some(self.instance.multiplier),
            "n" => Some(self.instance.series),
            "temp" => Some(self.instance.temperature.unwrap_or(context.temperature)),
            _ => None,
        })
    }
    fn observation_operating(
        &self,
        keyword: &str,
        context: &ModelContext,
        voltage: &dyn Fn(NodeId) -> Real,
        _: bool,
    ) -> SpiceResult<Option<Real>> {
        let bias = voltage(self.terminals[0]) - voltage(self.terminals[1]);
        let point = self.point(bias, context)?;
        Ok(match keyword {
            "vd" | "vj" => Some(bias),
            "id" | "i" | "cd" => Some(point.current),
            "gd" | "g" => Some(point.conductance),
            "cap" | "charge" => Some(0.),
            "p" | "power" => Some(bias * point.current),
            _ => None,
        })
    }
    fn instance_parameter(&self, keyword: &str) -> Option<&'static str> {
        ["m", "n", "area", "temp"]
            .into_iter()
            .find(|key| key.eq_ignore_ascii_case(keyword))
    }
    fn with_instance_parameter(
        &self,
        parameter: &str,
        value: Real,
        context: &ModelContext,
    ) -> SpiceResult<Box<dyn Device>> {
        let mut candidate = self.clone();
        candidate.instance.set(parameter, value, &self.location)?;
        candidate.prepared = self.prepared_at(context)?;
        candidate.gmin = context.gmin;
        Ok(Box::new(candidate))
    }
}
