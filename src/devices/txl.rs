//! Single lossy transmission lines (`Y`, `txl`).
//!
//! `inp2y.c` either expands a high-R/L line into three R and four C devices,
//! or binds two signal nodes (the written reference terminals are ignored by
//! C's Padé device). `txlsetup.c` constructs the Padé coefficients and
//! `txlload.c` stamps DC or recursive-convolution transient equations.
//! AC/SP are explicitly refused: C reuses `TXLload` with an unset delta.
//! See `docs/port/TXL.md` for the compatibility boundaries.
pub(in crate::devices) mod pade;
mod transient;
use super::elaboration::{ElaborationSize, reserve};
use crate::devices::linear::LinearContext;
use crate::devices::models::{ModelContext, ModelResolver, PreparationKey, ResolvedModel};
use crate::devices::rlc::{Capacitor, Resistor};
use crate::devices::schema::{ScalarDomain, ScalarParameter, ScalarSchema, ScalarUnit};
use crate::devices::traits::{AnalysisMode, Device, StampContext};
use crate::maths::Vector;
use crate::netlist::ast::{DeviceInstance, ModelCard, ParameterAssignment, ParameterKind};
use crate::primitives::{NodeAppend, NodeId, NodeTable, Real, SpiceError, SpiceResult};
use std::collections::BTreeSet;
pub use transient::{TxlHistory, TxlUpdate};

const fn scalar(name: &'static str, unit: ScalarUnit, domain: ScalarDomain) -> ScalarParameter {
    ScalarParameter {
        name,
        domain,
        unit,
        default: None,
    }
}
static MODEL: ScalarSchema<'static> = ScalarSchema::new(&[
    scalar("r", ScalarUnit::OhmPerMetre, ScalarDomain::NonNegative),
    scalar("l", ScalarUnit::HenryPerMetre, ScalarDomain::NonNegative),
    scalar("g", ScalarUnit::SiemensPerMetre, ScalarDomain::NonNegative),
    scalar("c", ScalarUnit::FaradPerMetre, ScalarDomain::Positive),
    scalar("length", ScalarUnit::Metre, ScalarDomain::Positive),
]);
static INSTANCE: ScalarSchema<'static> =
    ScalarSchema::new(&[scalar("len", ScalarUnit::Metre, ScalarDomain::Positive)]);
#[derive(Debug, Clone, Copy)]
struct Model {
    r: Real,
    l: Real,
    g: Real,
    c: Real,
    length: Real,
}
/// Immutable declaration data, separate from instance-specific length.
struct ModelInputs {
    model: Model,
    defaults: Vec<ParameterAssignment>,
}

impl ModelInputs {
    fn read(card: &ModelCard) -> SpiceResult<Self> {
        let mut setters = Vec::new();
        for p in &card.parameters {
            if p.kind == ParameterKind::Flag && p.name == "txl" {
                continue;
            }
            setters.push(p);
        }
        // `inpgmod.c::create_model`: an instance parameter on the `.model`
        // card (`len`) becomes a model-level instance default.
        let (setters, defaults) =
            crate::devices::schema::split_model_setters(setters, &MODEL, &INSTANCE);
        let values = MODEL.validate(setters, &card.location)?;
        let get = |key: &str| {
            values.get(key).map(|v| v.value).ok_or_else(|| {
                SpiceError::parse(
                    card.location.clone(),
                    format!("TXL model {}: {key} must be given (txlsetup.c)", card.name),
                )
            })
        };
        Ok(Self {
            model: Model {
                r: get("r")?,
                l: get("l")?,
                g: get("g")?,
                c: get("c")?,
                length: get("length")?,
            },
            defaults: defaults.into_iter().cloned().collect(),
        })
    }
}
/// A validated native line input. Its physical setup is deferred until the
/// batch has checked the aggregate resource plan.
pub(super) struct NativePlan {
    model: Model,
    length: Real,
    instance_length: Option<Real>,
}

impl NativePlan {
    fn prepare(instance: &DeviceInstance, model: &ResolvedModel<'_>) -> SpiceResult<Self> {
        if instance.nodes.len() != 4 {
            return Err(SpiceError::parse(
                instance.location.clone(),
                "TXL requires four written terminals (inp2y.c)",
            ));
        }
        let inputs = model.prepare(PreparationKey::Inputs, || ModelInputs::read(model.card()))?;
        let m = inputs.model;
        let values = INSTANCE.validate(
            inputs.defaults.iter().chain(&instance.parameters),
            &instance.location,
        )?;
        let given = values.get("len").map(|v| v.value);
        Ok(Self {
            model: m,
            length: given.unwrap_or(m.length),
            instance_length: given,
        })
    }
}

/// Two mutually exclusive outcomes of the parsed TXL setters. The high-R/L
/// case has no Y instance, only its seven generated R/C devices.
pub(super) enum TxlPlan {
    Native(NativePlan),
    Fallback(FallbackPlan),
}

impl TxlPlan {
    pub(super) fn prepare(
        instance: &DeviceInstance,
        models: &ModelResolver<'_>,
        context: &ModelContext,
    ) -> SpiceResult<Self> {
        context.validate(&instance.location)?;
        let resolved = models
            .resolve(instance)?
            .ok_or_else(|| SpiceError::circuit("TXL requires a model"))?;
        let native = NativePlan::prepare(instance, &resolved)?;
        let m = native.model;
        if m.r != 0. && m.l != 0. && m.r / m.l > 1.6e10 {
            // inp2y.c passes 3/(R*length) to the resistor's resistance
            // setter, not conductance. Retain the exact expression order.
            Ok(Self::Fallback(FallbackPlan {
                resistance: 3. / (m.r * native.length),
                capacitance: m.c * native.length / 6.,
            }))
        } else {
            Ok(Self::Native(native))
        }
    }

    pub(super) const fn size(&self) -> ElaborationSize {
        match self {
            Self::Native(_) => ElaborationSize::new(1, 0, 0),
            Self::Fallback(_) => FallbackPlan::SIZE,
        }
    }
}

pub(super) struct FallbackPlan {
    resistance: Real,
    capacitance: Real,
}

impl FallbackPlan {
    const SIZE: ElaborationSize = ElaborationSize::new(7, 2, 4 * super::rlc::REACTIVE_STATE_COUNT);

    pub(super) fn bind(
        self,
        instance: &DeviceInstance,
        nodes: &mut NodeAppend<'_>,
        deck_nodes: &BTreeSet<String>,
    ) -> SpiceResult<Vec<Box<dyn Device>>> {
        nodes.transaction(|nodes| {
            let terminals: [NodeId; 4] = nodes.intern_array(&instance.nodes)?;
            let mut line = [terminals[0], NodeId::GROUND, NodeId::GROUND, terminals[2]];
            for (k, node) in line.iter_mut().enumerate().take(3).skip(1) {
                let name = format!("txlnd{k}{}", instance.name);
                if nodes.get(&name).is_some()
                    || deck_nodes.contains(&NodeTable::canonical_name(&name, nodes.auto_gnd()))
                {
                    return Err(SpiceError::circuit(format!(
                        "TXL generated-node collision: {name}"
                    )));
                }
                *node = nodes.intern(&name)?;
            }
            let r = self.resistance;
            let c = self.capacitance;
            let mut devices: Vec<Box<dyn Device>> = Vec::new();
            reserve(&mut devices, Self::SIZE.devices, Some(&instance.location))?;
            for k in 0..3 {
                devices.push(Box::new(Resistor::new(
                    format!("txlres{}{}", k + 1, instance.name),
                    [line[k], line[k + 1]],
                    r,
                )?));
            }
            for (k, node) in line.iter().enumerate() {
                devices.push(Box::new(Capacitor::new(
                    format!("txlcap{}{}", k + 1, instance.name),
                    [*node, terminals[1]],
                    if k == 0 || k == 3 { c } else { 2. * c },
                    None,
                )?));
            }
            Ok(devices)
        })
    }
}
/// A Padé-path TXL instance. Its two branch currents point into the line
/// at the respective signal terminals, as `txlload.c` stamps them.
#[derive(Debug, Clone)]
pub struct Txl {
    name: String,
    terminals: [NodeId; 2],
    /// The instance `len` setter, if given. C's `@y1[length]` reports this
    /// field alone, so a model-level `length` reads as 0 (measured).
    instance_length: Option<Real>,
    resistance: Real,
    coefficients: pade::Pade,
}
impl NativePlan {
    pub(super) fn bind(
        self,
        instance: &DeviceInstance,
        nodes: &mut NodeAppend<'_>,
    ) -> SpiceResult<Box<dyn Device>> {
        let Self {
            model: m,
            length,
            instance_length,
        } = self;
        let coefficients = pade::Pade::new(m.r, m.l, m.g, m.c, length)?;
        let resistance = m.r * length;
        if !resistance.is_finite() {
            return Err(SpiceError::circuit("TXL R*length overflow"));
        }
        let ids: [NodeId; 4] = nodes.intern_array(&instance.nodes)?;
        Ok(Box::new(Txl {
            name: instance.name.clone(),
            terminals: [ids[0], ids[2]],
            instance_length,
            resistance,
            coefficients,
        }))
    }
}
impl Txl {
    fn unsupported(&self, analysis: &str) -> SpiceError {
        SpiceError::not_yet_ported(
            format!("{}: TXL {analysis}", self.name),
            "src/spicelib/devices/txl/txlinit.c: DEVacLoad reuses TXLload with an unset CKTdelta rather than a frequency-domain stamp; DEVpzLoad/DEVnoise/DEVdisto are NULL (docs/port/TXL.md, TX0)",
        )
    }
    /// TSTALLOC order shared by setup and each load; row 0 is never ground.
    const fn positions(
        p: Option<usize>,
        n: Option<usize>,
        a: usize,
        b: usize,
    ) -> [(Option<usize>, Option<usize>); 14] {
        let a = Some(a);
        let b = Some(b);
        [
            (p, p),
            (p, n),
            (n, p),
            (n, n),
            (a, p),
            (b, n),
            (n, b),
            (p, a),
            (a, a),
            (b, b),
            (a, n),
            (b, p),
            (a, b),
            (b, a),
        ]
    }
    fn dc_values(&self, gmin: Real) -> [Real; 14] {
        let g = 0.1 * gmin;
        [
            g,
            g,
            g,
            g,
            0.,
            -1.,
            1.,
            1.,
            1.,
            0.,
            0.,
            1.,
            1.,
            -self.resistance,
        ]
    }
}
impl Device for Txl {
    fn name(&self) -> &str {
        &self.name
    }
    fn designator(&self) -> char {
        'y'
    }
    fn terminals(&self) -> &[NodeId] {
        &self.terminals
    }
    fn branch_currents(&self) -> usize {
        2
    }
    fn delay_line(&self) -> Option<&dyn crate::devices::delay::DelayLine> {
        Some(self)
    }
    fn setup_matrix(
        &self,
        setup: &mut crate::devices::setup::MatrixSetup<'_>,
    ) -> SpiceResult<bool> {
        let (p, n) = (setup.node(self.terminals[0]), setup.node(self.terminals[1]));
        for (r, c) in Self::positions(p, n, setup.branch(0)?, setup.branch(1)?) {
            setup.element(r, c);
        }
        Ok(true)
    }
    fn stamp(&self, ctx: &mut StampContext<'_>) -> SpiceResult<()> {
        let p = ctx.unknowns.node_row(self.terminals[0]);
        let n = ctx.unknowns.node_row(self.terminals[1]);
        let a = ctx.branch(0)?;
        let b = ctx.branch(1)?;
        let (values, rhs) = match ctx.mode {
            AnalysisMode::OperatingPoint | AnalysisMode::DcSweep => {
                (self.dc_values(ctx.gmin), [0.; 2])
            }
            AnalysisMode::Ac { .. } => return Err(self.unsupported("AC/SP")),
            AnalysisMode::Transient { time, dt } => {
                let history = ctx.states.delay_history().ok_or_else(|| {
                    SpiceError::circuit("TXL transient requires accepted delay history")
                })?;
                self.transient_stamp(history, time, dt, ctx.gmin)?
            }
        };
        for ((r, c), v) in Self::positions(p, n, a, b).into_iter().zip(values) {
            if let (Some(r), Some(c)) = (r, c) {
                ctx.matrix.add(r, c, v)?;
            }
        }
        ctx.rhs.add_to(a, rhs[0])?;
        ctx.rhs.add_to(b, rhs[1])?;
        Ok(())
    }
    fn assemble_linear(&self, ctx: &mut LinearContext<'_>) -> SpiceResult<()> {
        let a = ctx
            .branch
            .ok_or_else(|| SpiceError::circuit("TXL missing branches"))?;
        let p = ctx.unknowns.node_row(self.terminals[0]);
        let n = ctx.unknowns.node_row(self.terminals[1]);
        for ((r, c), v) in Self::positions(p, n, a, a + 1)
            .into_iter()
            .zip(self.dc_values(ctx.model_context.gmin))
        {
            if let (Some(r), Some(c)) = (r, c) {
                ctx.system.a.add(r, c, v)?;
            }
        }
        Ok(())
    }
    fn assemble_small_signal(
        &self,
        ctx: &mut LinearContext<'_>,
        _bias: &Vector,
    ) -> SpiceResult<()> {
        // Source-resolution probes in a DC sweep have no bias state.
        if ctx.states.is_none() {
            self.assemble_linear(ctx)
        } else {
            Err(self.unsupported("AC/SP"))
        }
    }
    fn assemble_pole_zero(&self, _ctx: &mut LinearContext<'_>, _bias: &Vector) -> SpiceResult<()> {
        Err(self.unsupported("pole-zero"))
    }
    fn noise(
        &self,
        _ctx: &crate::devices::noise::NoiseContext<'_>,
        _sources: &mut Vec<crate::devices::noise::NoiseSource>,
    ) -> SpiceResult<crate::devices::noise::DeviceNoise<'_>> {
        Err(self.unsupported("noise"))
    }
    fn distortion(
        &self,
        _ctx: &crate::devices::distortion::DistortionContext<'_>,
    ) -> SpiceResult<crate::devices::distortion::DeviceDistortion> {
        Err(self.unsupported("distortion"))
    }
    /// `TXLask`: only `length`, which C reports from the instance field, so a
    /// model-level `length` reads as 0. The `in_node`/`out_node` entries C's
    /// table declares are refused by its own front end ("no such parameter"),
    /// so they are not exposed here either.
    fn observation_parameter(
        &self,
        keyword: &str,
        _ctx: &crate::devices::ModelContext,
    ) -> SpiceResult<Option<Real>> {
        Ok(match keyword {
            "length" => Some(self.instance_length.unwrap_or(0.)),
            _ => None,
        })
    }
}
