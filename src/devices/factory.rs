//! Scalar linear-device elaboration; unsupported parameters never disappear.
use crate::devices::models::ModelBackend;
use crate::devices::passive::PassiveKind;
use crate::devices::rlc::ResistorNoise;
use crate::devices::sources::{SourceExcitation, SourceInputs};
use crate::devices::{
    AmSpec, Capacitor, Device, ExpSpec, FunctionSpec, Inductor, PulseSpec, PwlSource, Resistor,
    SffmSpec, SineSpec, Waveform,
};
use crate::netlist::ast::{
    DeviceInstance, FunctionWaveform, ParameterKind, PositionedValue, SourceFunction,
    SourceWaveform, WaveformKind,
};
use crate::netlist::source::{Deck, LogicalLine};
use crate::netlist::{Parser, RawCard};
use crate::primitives::{
    NodeAppend, NodeKind, NodeTable, Real, SourceLoc, SpiceError, SpiceResult, parse_spice_number,
};

/// Designators [`instantiate`] builds from the card alone (the registry's
/// `ported` entries). `tests/registry_support.rs` elaborates one instance of
/// every designator to keep this list and the model catalog honest.
pub(super) const CARD_FACTORY: &[char] =
    &['r', 'c', 'l', 'v', 'i', 'e', 'f', 'g', 'h', 'b', 'k', 't'];

pub(super) fn from_card(card: &RawCard, nodes: &mut NodeTable) -> SpiceResult<Box<dyn Device>> {
    let deck = Deck {
        path: card.location.path().to_path_buf(),
        title: String::new(),
        title_location: card.location.clone(),
        lines: vec![LogicalLine {
            location: card.location.clone(),
            end_line: card.position().advance(card.raw.len()).location().line,
            continuations: 0,
            text: card.raw.clone(),
            source_map: card.source_map.clone(),
        }],
    };
    let netlist = Parser::new().parse_deck(&deck)?;
    let [instance] = netlist.devices.as_slice() else {
        return Err(SpiceError::parse(
            card.location.clone(),
            "device card must elaborate to exactly one instance",
        ));
    };
    nodes.transaction(|nodes| instantiate(instance, nodes))
}

/// Resolve and validate before any node interning. Bounded R/C/L and M4 D/Q/M
/// factories reject unsupported physics before committing staged nodes.
/// The append-only capability permits interning nodes and changing only
/// newly created node kinds. Circuit batch elaboration can roll back the
/// entire suffix without copying or changing pre-existing nodes.
pub(super) fn instantiate_with_models(
    instance: &DeviceInstance,
    nodes: &mut NodeAppend<'_>,
    models: &crate::devices::models::ModelResolver<'_>,
    context: &crate::devices::models::ModelContext,
) -> SpiceResult<Box<dyn Device>> {
    context.validate(&instance.location)?;
    let Some(model) = models.resolve(instance)? else {
        return instantiate(instance, nodes);
    };
    // Port counts are backend-specific. A three-terminal MOS cannot be
    // silently bound to a bulk-MOS, nor may VDMOS ignore a fourth terminal.
    if instance.designator == 'm' {
        let vdmos = model.levels().backend == ModelBackend::Vdmos;
        let expected = if vdmos { 3 } else { 4 };
        if instance.nodes.len() != expected {
            return Err(SpiceError::Unsupported {
                feature: format!(
                    "{}: {}-terminal {} connection required, got {}",
                    instance.name,
                    expected,
                    if vdmos { "VDMOS" } else { "bulk MOS" },
                    instance.nodes.len()
                ),
                location: Some(instance.location.clone()),
            });
        }
    }
    // The resolver has already selected a concrete backend. Exhaustive
    // dispatch means a new catalog backend cannot silently choose another.
    match model.levels().backend {
        ModelBackend::Resistor | ModelBackend::Capacitor | ModelBackend::Inductor => {
            crate::devices::passive::instantiate(instance, nodes, &model, context)
        }
        ModelBackend::Diode => {
            crate::devices::nonlinear::Diode::instantiate(instance, nodes, &model, context)
        }
        ModelBackend::Bjt => {
            crate::devices::bjt::Bjt::instantiate(instance, nodes, &model, context)
        }
        ModelBackend::Mos1 => {
            crate::devices::mos1::Mos1::instantiate(instance, nodes, &model, context)
        }
        ModelBackend::Mos2 => {
            crate::devices::mos2::Mos2::instantiate(instance, nodes, &model, context)
        }
        ModelBackend::Mos3 => {
            crate::devices::mos3::Mos3::instantiate(instance, nodes, &model, context)
        }
        ModelBackend::Mos6 => {
            crate::devices::mos6::Mos6::instantiate(instance, nodes, &model, context)
        }
        ModelBackend::Mos9 => {
            crate::devices::mos9::Mos9::instantiate(instance, nodes, &model, context)
        }
        ModelBackend::Bsim3 => crate::devices::bsim3::instantiate(instance, nodes, &model, context),
        ModelBackend::Bsim4 => crate::devices::bsim4::instantiate(instance, nodes, &model, context),
        ModelBackend::Vdmos => {
            crate::devices::vdmos::Vdmos::instantiate(instance, nodes, &model, context)
        }
        ModelBackend::Jfet => {
            crate::devices::jfet::Jfet::instantiate(instance, nodes, &model, context)
        }
        ModelBackend::Jfet2 => {
            crate::devices::jfet2::Jfet2::instantiate(instance, nodes, &model, context)
        }
        ModelBackend::Mesfet => {
            crate::devices::mesfet::Mesfet::instantiate(instance, nodes, &model, context)
        }
        ModelBackend::Mesa => {
            crate::devices::hfet::mesa::Mesa::instantiate(instance, nodes, &model, context)
        }
        ModelBackend::Hfet1 => {
            crate::devices::hfet::hfet1::Hfet1::instantiate(instance, nodes, &model, context)
        }
        ModelBackend::Hfet2 => {
            crate::devices::hfet::hfet2::Hfet2::instantiate(instance, nodes, &model, context)
        }
        ModelBackend::Switch | ModelBackend::CurrentSwitch => {
            crate::devices::switch::Switch::instantiate(instance, nodes, &model, context)
        }
        ModelBackend::Txl => crate::devices::txl::Txl::instantiate(instance, nodes, &model),
        ModelBackend::Ltra => {
            crate::devices::ltra::LossyLine::instantiate(instance, nodes, models, context)
                .map(|line| Box::new(line) as Box<dyn Device>)
        }
        ModelBackend::Urc => Err(SpiceError::circuit(format!(
            "URC {} is expanded by Circuit::add_instances, not built as one device",
            instance.name
        ))),
    }
}

pub(super) fn instantiate(
    instance: &DeviceInstance,
    nodes: &mut NodeAppend<'_>,
) -> SpiceResult<Box<dyn Device>> {
    if matches!(instance.designator, 'e'..='h') {
        return crate::devices::controlled::instantiate(instance, nodes);
    }
    if instance.designator == 'k' {
        return crate::devices::mutual::instantiate(instance);
    }
    if instance.designator == 't' {
        return crate::devices::tline::TransmissionLine::instantiate(instance, nodes);
    }
    // `a` instances only come from the front end's TABLE/POLY lowering
    // (crate::netlist::behavioural); user XSPICE cards are not parsed.
    if matches!(instance.designator, 'b' | 'a') {
        return crate::devices::behavioural::instantiate(instance, nodes);
    }
    if instance.model.is_some() || instance.nodes.len() != 2 {
        return Err(SpiceError::Unsupported {
            feature: format!("model/multiport device {}", instance.name),
            location: Some(instance.location.clone()),
        });
    }
    match instance.designator {
        'r' => instantiate_passive(instance, nodes, PassiveKind::Resistor),
        'c' => instantiate_passive(instance, nodes, PassiveKind::Capacitor),
        'l' => instantiate_passive(instance, nodes, PassiveKind::Inductor),
        'v' | 'i' => instantiate_source(instance, nodes),
        _ => Err(SpiceError::not_yet_ported(
            format!("device {}", instance.name),
            "src/spicelib/devices/",
        )),
    }
}

/// Build a four-terminal line's internal unknowns in C setup order. On a
/// generated-name collision, rollback only the newly interned nodes.
pub(super) fn line_terminals<const N: usize>(
    instance: &DeviceInstance,
    nodes: &mut NodeAppend<'_>,
    suffixes: &[&str],
    label: &str,
) -> SpiceResult<[crate::primitives::NodeId; N]> {
    use crate::primitives::NodeId;
    if instance.nodes.len() != 4 || N != 4 + suffixes.len() {
        return Err(SpiceError::circuit(
            "invalid transmission-line terminal layout",
        ));
    }
    nodes.transaction(|nodes| {
        let mut terminals = [NodeId::GROUND; N];
        for (slot, node) in instance.nodes.iter().enumerate() {
            terminals[slot] = nodes.intern(node)?;
        }
        for (slot, suffix) in suffixes.iter().enumerate() {
            let name = format!("{}#{suffix}", instance.name);
            if nodes.get(&name).is_some() {
                return Err(SpiceError::circuit(format!(
                    "{label} internal-node name collision: {name}"
                )));
            }
            let node = nodes.intern(&name)?;
            nodes.set_kind(node, NodeKind::Internal)?;
            terminals[4 + slot] = node;
        }
        Ok(terminals)
    })
}

/// Parse the shared scalar parameter syntax without conflating the two
/// factories' supported setter lists or dropping intermediate errors.
pub(super) fn scalar_number(p: &crate::netlist::ast::ParameterAssignment) -> SpiceResult<Real> {
    parse_spice_number(&p.value)
        .filter(|v| v.is_finite())
        .ok_or_else(|| SpiceError::Unsupported {
            feature: format!("non-finite or nonliteral {}={}", p.name, p.value),
            location: Some(p.location.clone()),
        })
}

fn unsupported_parameter(
    instance: &DeviceInstance,
    p: &crate::netlist::ast::ParameterAssignment,
) -> SpiceError {
    SpiceError::Unsupported {
        feature: format!("{} parameter {}", instance.name, p.name),
        location: Some(p.location.clone()),
    }
}

fn instantiate_passive(
    instance: &DeviceInstance,
    nodes: &mut NodeAppend<'_>,
    kind: PassiveKind,
) -> SpiceResult<Box<dyn Device>> {
    let mut value = 0.;
    let mut ic = None;
    let mut noisy = true;
    for p in &instance.parameters {
        if p.kind != ParameterKind::Scalar {
            return Err(SpiceError::not_yet_ported(
                format!(
                    "{}: {} setter '{}' runtime semantics",
                    p.location, instance.name, p.name
                ),
                "src/spicelib/parser/inpdpar.c",
            ));
        }
        match p.name.as_str() {
            name if name == kind.principal().name => {
                let number = scalar_number(p)?;
                // resparam.c: RES_RESIST raises tiny resistance to RESMIN.
                value = if kind == PassiveKind::Resistor {
                    crate::devices::rlc::res_resist(&instance.name, number)
                } else {
                    number
                };
            }
            "noisy" if kind == PassiveKind::Resistor => {
                // IF_INTEGER: floor(value + 0.5).
                noisy = (scalar_number(p)? + 0.5).floor() != 0.;
            }
            "ic" if kind != PassiveKind::Resistor => ic = Some(scalar_number(p)?),
            _ => return Err(unsupported_parameter(instance, p)),
        }
    }
    nodes.transaction(|nodes| {
        let terminals = [
            nodes.intern(&instance.nodes[0])?,
            nodes.intern(&instance.nodes[1])?,
        ];
        match kind {
            PassiveKind::Resistor => Resistor::new(&instance.name, terminals, value).map(|r| {
                Box::new(r.with_noise(ResistorNoise {
                    noisy,
                    ..ResistorNoise::default()
                })) as Box<dyn Device>
            }),
            PassiveKind::Capacitor => Capacitor::new(&instance.name, terminals, value, ic)
                .map(|c| Box::new(c) as Box<dyn Device>),
            PassiveKind::Inductor => Inductor::new(&instance.name, terminals, value, ic)
                .map(|l| Box::new(l) as Box<dyn Device>),
        }
    })
}

enum SourceSetter {
    Dc,
    AcMagnitude,
    AcPhase,
    DistortionMagnitude(usize),
    DistortionPhase(usize),
    Delay,
    Repeat,
    Port(PortSetter),
}

impl SourceSetter {
    fn parse(name: &str, voltage: bool) -> Option<Self> {
        Some(match name {
            "dc" => Self::Dc,
            "acmag" => Self::AcMagnitude,
            "acphase" => Self::AcPhase,
            "distof1mag" => Self::DistortionMagnitude(0),
            "distof1phase" => Self::DistortionPhase(0),
            "distof2mag" => Self::DistortionMagnitude(1),
            "distof2phase" => Self::DistortionPhase(1),
            "td" => Self::Delay,
            "r" => Self::Repeat,
            "portnum" if voltage => Self::Port(PortSetter::Number),
            "z0" if voltage => Self::Port(PortSetter::Impedance),
            "pwr" if voltage => Self::Port(PortSetter::Power),
            "freq" if voltage => Self::Port(PortSetter::Frequency),
            "phase" if voltage => Self::Port(PortSetter::Phase),
            _ => return None,
        })
    }
}

#[derive(Clone, Copy)]
enum PortSetter {
    Number,
    Impedance,
    Power,
    Frequency,
    Phase,
}

struct SourceSetters {
    inputs: SourceInputs,
    port: PortSetters,
    distortion: [Option<crate::devices::distortion::DistortionInput>; 2],
}

fn parse_source_setters(instance: &DeviceInstance) -> SpiceResult<SourceSetters> {
    let mut inputs = SourceInputs::default();
    // C applies waveform setters in order, so the last one wins.
    // PWL `td=` (any order) and `r=` (applies to the PWL already set), as the
    // ordered VSRC_TD/VSRC_R setters of vsrcpar.c/isrcpar.c.
    let mut pwl_delay: Option<(f64, SourceLoc)> = None;
    let mut pwl_repeat: Option<(f64, SourceLoc)> = None;
    let mut port = PortSetters::default();
    // `distof1`/`distof2` (magnitude, phase), each setter replacing both.
    let mut distortion: [Option<crate::devices::distortion::DistortionInput>; 2] = [None, None];
    for p in &instance.parameters {
        if let ParameterKind::Waveform(source) = &p.kind {
            if let Some((_, at)) = &pwl_repeat {
                // C keeps r='s knot index from the earlier PWL: refuse rather
                // than apply a stale repetition point.
                return Err(SpiceError::Unsupported {
                    feature: format!(
                        "{}: waveform setter '{}' after r= (set r= after the final PWL)",
                        instance.name, p.name
                    ),
                    location: Some(at.clone()),
                });
            }
            inputs.waveform = Some(source_waveform(&p.name, source, &p.location)?);
            // A later waveform setter replaces C's PORT function type.
            port.power_function = None;
            continue;
        }
        if p.kind != ParameterKind::Scalar {
            return Err(SpiceError::not_yet_ported(
                format!(
                    "{}: {} setter '{}' runtime semantics",
                    p.location, instance.name, p.name
                ),
                match instance.designator {
                    'v' => "src/spicelib/devices/vsrc/vsrcload.c",
                    'i' => "src/spicelib/devices/isrc/isrcload.c",
                    _ => "src/spicelib/parser/inpdpar.c",
                },
            ));
        }
        let setter = SourceSetter::parse(&p.name, instance.designator == 'v')
            .ok_or_else(|| unsupported_parameter(instance, p))?;
        let number = scalar_number(p)?;
        match setter {
            SourceSetter::AcMagnitude => {
                inputs.ac.get_or_insert_default().magnitude = Some(number);
            }
            SourceSetter::AcPhase => {
                inputs.ac.get_or_insert_default().phase = Some(number);
            }
            SourceSetter::DistortionMagnitude(index) => {
                // A new setter replaces both values (vsrcpar.c).
                distortion[index] = Some(crate::devices::distortion::DistortionInput {
                    magnitude: number,
                    phase: 0.,
                });
            }
            SourceSetter::DistortionPhase(index) => {
                distortion[index]
                    .get_or_insert(crate::devices::distortion::DistortionInput {
                        magnitude: 1.,
                        phase: 0.,
                    })
                    .phase = number;
            }
            SourceSetter::Delay => pwl_delay = Some((number, p.location.clone())),
            SourceSetter::Port(setter) => port.set(setter, number, &p.location)?,
            SourceSetter::Repeat => {
                validate_pwl_repeat(instance, p, inputs.waveform.as_ref(), number)?;
                pwl_repeat = Some((number, p.location.clone()));
            }
            SourceSetter::Dc => inputs.dc = Some(number),
        }
    }
    if pwl_delay.is_some() || pwl_repeat.is_some() {
        inputs.waveform = Some(pwl_options(inputs.waveform, pwl_delay, pwl_repeat)?);
    }
    Ok(SourceSetters {
        inputs,
        port,
        distortion,
    })
}

fn validate_pwl_repeat(
    instance: &DeviceInstance,
    p: &crate::netlist::ast::ParameterAssignment,
    waveform: Option<&Waveform>,
    number: Real,
) -> SpiceResult<()> {
    // C silently ignores r= when no PWL coefficients exist yet.
    let Some(Waveform::Pwl(knots)) = waveform else {
        return Err(SpiceError::Unsupported {
            feature: format!(
                "{}: r= without a preceding PWL setter (C ignores it)",
                instance.name
            ),
            location: Some(p.location.clone()),
        });
    };
    // VSRC_R/ISRC_R validate every r= as it is set (E_PARMVAL
    // aborts the deck), so an invalid earlier r= is an error even
    // when a later one is valid.
    knots
        .repeat_index(Some(number))
        .map_err(|error| SpiceError::Unsupported {
            feature: format!("{}: {error}", instance.name),
            location: Some(p.location.clone()),
        })?;
    Ok(())
}

fn instantiate_source(
    instance: &DeviceInstance,
    nodes: &mut NodeAppend<'_>,
) -> SpiceResult<Box<dyn Device>> {
    let SourceSetters {
        inputs,
        port,
        distortion,
    } = parse_source_setters(instance)?;
    let power_function = port.power_function.is_some();
    let rf = port.resolve(instance)?;
    let excitation = SourceExcitation::from_inputs(instance.designator == 'v', inputs)?;
    // Only topology binding now needs the append-only transaction.
    nodes.transaction(|nodes| {
        let terminals = [
            nodes.intern(&instance.nodes[0])?,
            nodes.intern(&instance.nodes[1])?,
        ];
        let source = excitation
            .bind(&instance.name, terminals)
            .with_distortion(distortion[0], distortion[1])?;
        let device: Box<dyn Device> = match rf {
            Some(rf) => Box::new(source.with_port(rf.bind(instance, nodes)?)?),
            // A PORT setter without an enabled port still selects C's
            // zero-amplitude function; no second tone calculation is needed.
            None => Box::new(source.with_power_tone(power_function.then_some((0., 0.)))),
        };
        Ok(device)
    })
}

struct PortInput {
    number: usize,
    z0: Real,
    power: Real,
    frequency: Real,
    phase: Real,
    power_function: bool,
}

impl PortInput {
    fn bind(
        self,
        instance: &DeviceInstance,
        nodes: &mut NodeAppend<'_>,
    ) -> SpiceResult<crate::devices::RfPort> {
        let name = format!("{}#res", instance.name);
        if nodes.get(&name).is_some() {
            return Err(SpiceError::circuit(format!(
                "RF port internal-node name collision: {name}"
            )));
        }
        let internal = nodes.intern(&name)?;
        nodes.set_kind(internal, NodeKind::Internal)?;
        Ok(crate::devices::RfPort {
            number: self.number,
            z0: self.z0,
            internal,
            power: self.power,
            frequency: self.frequency,
            phase: self.phase,
            power_function: self.power_function,
        })
    }
}

/// The ordered RFSPICE port setters of one V card (`vsrcpar.c`), resolved by
/// [`PortSetters::resolve`] with `vsrctemp.c`'s rules.
#[derive(Debug, Default)]
struct PortSetters {
    number: Option<(i64, SourceLoc)>,
    z0: Real,
    z0_given: bool,
    power: Option<Real>,
    frequency: Option<Real>,
    phase: Option<Real>,
    /// Where `pwr`/`freq` last selected the `PORT` function type, unless a
    /// later waveform setter replaced it.
    power_function: Option<SourceLoc>,
    any: Option<SourceLoc>,
}

impl PortSetters {
    fn set(&mut self, setter: PortSetter, number: Real, at: &SourceLoc) -> SpiceResult<()> {
        self.any.get_or_insert_with(|| at.clone());
        match setter {
            PortSetter::Number => {
                // INPgetValue(IF_INTEGER): (int) floor(0.5 + value).
                let rounded = crate::primitives::c_integer(number).ok_or_else(|| {
                    SpiceError::Unsupported {
                        feature: format!("portnum {number} is outside the C integer range"),
                        location: Some(at.clone()),
                    }
                })?;
                self.number = Some((rounded, at.clone()));
                // VSRC_PORTNUM: a port needs an impedance; take 50 ohms
                // unless a positive z0 was already set.
                if self.z0 <= 0. {
                    self.z0 = 50.;
                }
            }
            PortSetter::Impedance => {
                self.z0 = number;
                self.z0_given = true;
            }
            PortSetter::Power => {
                self.power = Some(number);
                self.power_function = Some(at.clone());
            }
            PortSetter::Frequency => {
                self.frequency = Some(number);
                self.power_function = Some(at.clone());
            }
            PortSetter::Phase => self.phase = Some(number),
        }
        Ok(())
    }

    /// `vsrctemp.c`: a source is a port when `portnum` was given, is positive
    /// and its `z0` (50 when not given) is positive. This resolves the
    /// setters before the port's internal node is allocated.
    fn resolve(self, instance: &DeviceInstance) -> SpiceResult<Option<PortInput>> {
        let Some(at) = self.any else {
            return Ok(None);
        };
        let z0 = if self.z0_given { self.z0 } else { 50. };
        let number = match &self.number {
            Some((number, _)) if *number > 0 && z0 > 0. => usize::try_from(*number).ok(),
            _ => None,
        };
        let Some(number) = number else {
            // Not a port: portnum/z0/phase have no effect in C. pwr/freq would
            // still switch the time function to PORT with an unset amplitude.
            return Ok(None);
        };
        if !z0.is_finite() {
            return Err(SpiceError::Unsupported {
                feature: format!("{}: non-finite port impedance z0", instance.name),
                location: Some(at),
            });
        }
        Ok(Some(PortInput {
            number,
            z0,
            power: self.power.unwrap_or(1e-3),
            frequency: self.frequency.unwrap_or(1e9),
            phase: self.phase.unwrap_or(0.),
            power_function: self.power_function.is_some(),
        }))
    }
}

/// Numeric text of a waveform field. `what` names the field in diagnostics.
fn field(value: &PositionedValue, what: &str) -> SpiceResult<f64> {
    parse_spice_number(&value.text)
        .filter(|v| v.is_finite())
        .ok_or_else(|| SpiceError::Unsupported {
            feature: format!("non-finite or nonliteral {what}={}", value.text),
            location: Some(value.location.clone()),
        })
}

/// Applies PWL `td=`/`r=` to the final waveform, which must be a PWL: C ignores
/// `td=` for any other function, which the port refuses instead.
fn pwl_options(
    waveform: Option<Waveform>,
    delay: Option<(f64, SourceLoc)>,
    repeat: Option<(f64, SourceLoc)>,
) -> SpiceResult<Waveform> {
    let at = delay.as_ref().or(repeat.as_ref()).map(|(_, at)| at.clone());
    let Some(Waveform::Pwl(knots)) = waveform else {
        return Err(SpiceError::Unsupported {
            feature: "td=/r= without a PWL waveform (C ignores them)".to_owned(),
            location: at,
        });
    };
    knots
        .with_timing(delay.map_or(0., |(d, _)| d), repeat.map(|(r, _)| r))
        .map(Waveform::Pwl)
        .map_err(|error| SpiceError::Unsupported {
            feature: error.to_string(),
            location: at,
        })
}

/// Converts a parsed PULSE/PWL/SIN/EXP/SFFM/AM setter into a device waveform.
/// Analysis-dependent defaults stay pending (`vsrcload.c`).
/// `setter` locates the whole setter for diagnostics without a field.
fn source_waveform(
    name: &str,
    source: &SourceWaveform,
    setter: &SourceLoc,
) -> SpiceResult<Waveform> {
    if WaveformKind::from_keyword(name) != Some(source.kind()) {
        return Err(SpiceError::parse(
            setter.clone(),
            format!("source waveform '{name}' does not match its typed coefficients"),
        ));
    }
    match source {
        SourceWaveform::Function(f) => function_waveform(f, setter),
        SourceWaveform::Pulse(p) => {
            let optional = |v: &Option<PositionedValue>, what| v.as_ref().map(|v| field(v, what));
            let spec = PulseSpec {
                initial: field(&p.initial, "pulse v1")?,
                pulsed: field(&p.pulsed, "pulse v2")?,
                delay: optional(&p.delay, "pulse td").transpose()?,
                rise: optional(&p.rise, "pulse tr").transpose()?,
                fall: optional(&p.fall, "pulse tf").transpose()?,
                width: optional(&p.width, "pulse pw").transpose()?,
                period: optional(&p.period, "pulse per").transpose()?,
                count: optional(&p.count, "pulse np").transpose()?,
            };
            let waveform = Waveform::PulseDefaults(spec);
            waveform
                .validate()
                .map_err(|e| waveform_error(&p.initial, &e))?;
            Ok(waveform)
        }
        SourceWaveform::Pwl(points) => {
            let knots = points
                .iter()
                .map(|p| Ok((field(&p.time, "pwl time")?, field(&p.value, "pwl value")?)))
                .collect::<SpiceResult<Vec<_>>>()?;
            PwlSource::plain(knots)
                .map(Waveform::Pwl)
                .map_err(|_| SpiceError::Unsupported {
                    feature: "PWL times must be nonnegative and strictly increasing".to_owned(),
                    location: Some(
                        points
                            .first()
                            .map_or_else(|| setter.clone(), |p| p.time.location.clone()),
                    ),
                })
        }
    }
}

fn function_waveform(f: &FunctionWaveform, setter: &SourceLoc) -> SpiceResult<Waveform> {
    // The parser bounds the field count; a hand-built AST may not.
    let most = f.function.fields().len();
    if !(2..=most).contains(&f.values.len()) {
        return Err(SpiceError::Unsupported {
            feature: format!(
                "{} with {} fields (needs 2 to {most})",
                f.function.keyword(),
                f.values.len()
            ),
            location: Some(
                f.values
                    .get(most)
                    .map_or_else(|| setter.clone(), |extra| extra.location.clone()),
            ),
        });
    }
    let at = &f.values[0];
    let fields = f
        .values
        .iter()
        .zip(f.function.fields())
        .map(|(value, what)| field(value, &format!("{} {what}", f.function.keyword())))
        .collect::<SpiceResult<Vec<_>>>()?;
    let get = |index: usize| fields.get(index).copied();
    let (v0, v1) = (fields[0], fields[1]);
    let spec = match f.function {
        SourceFunction::Sin => FunctionSpec::Sine(SineSpec {
            offset: v0,
            amplitude: v1,
            frequency: get(2),
            delay: get(3),
            damping: get(4),
            phase: get(5),
        }),
        SourceFunction::Exp => FunctionSpec::Exp(ExpSpec {
            initial: v0,
            pulsed: v1,
            rise_delay: get(2),
            rise_tau: get(3),
            fall_delay: get(4),
            fall_tau: get(5),
        }),
        SourceFunction::Sffm => FunctionSpec::Sffm(SffmSpec {
            offset: v0,
            amplitude: v1,
            carrier: get(2),
            index: get(3),
            signal: get(4),
            delay: get(5),
            signal_phase: get(6),
            carrier_phase: get(7),
        }),
        SourceFunction::Am => FunctionSpec::Am(AmSpec {
            offset: v0,
            carrier_amplitude: v1,
            modulation_amplitude: get(2),
            modulation_frequency: get(3),
            carrier_frequency: get(4),
            delay: get(5),
            modulation_phase: get(6),
            carrier_phase: get(7),
        }),
    };
    let waveform = Waveform::FunctionDefaults(spec);
    waveform.validate().map_err(|e| waveform_error(at, &e))?;
    Ok(waveform)
}

fn waveform_error(at: &PositionedValue, error: &SpiceError) -> SpiceError {
    SpiceError::Unsupported {
        feature: error.to_string(),
        location: Some(at.location.clone()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::netlist::source::parse_deck_text;
    use std::path::Path;

    #[test]
    fn source_setters_are_ordered_and_bad_ast_does_not_bind_nodes() {
        let deck = Parser::new()
            .parse_deck(&parse_deck_text(
                Path::new("source.cir"),
                "title\nv1 a 0 dc 0 ac 0 portnum=1 z0=75 pwr=1m\n.end\n",
            ))
            .unwrap();
        let instance = &deck.devices[0];
        let input = parse_source_setters(instance).unwrap();
        let ac = input
            .inputs
            .ac
            .expect("explicit AC magnitude must be retained");
        assert_eq!(ac.magnitude.map(f64::to_bits), Some(0_f64.to_bits()));
        assert_eq!(ac.phase.map(f64::to_bits), Some(0_f64.to_bits()));
        assert!(input.port.power_function.is_some());
        let mut nodes = NodeTable::new();
        let existing = nodes.intern("v1#res").unwrap();
        nodes.set_kind(existing, NodeKind::Internal);
        let before = nodes.nodes().to_vec();
        let error = nodes
            .transaction(|nodes| instantiate_source(instance, nodes))
            .unwrap_err();
        assert!(error.to_string().contains("collision"));
        assert_eq!(nodes.nodes(), before);

        let mut malformed = instance.clone();
        let mut invalid = malformed.parameters[0].clone();
        invalid.name = "acmag".into();
        invalid.value = "not-a-number".into();
        malformed.parameters.insert(0, invalid);
        let error = nodes
            .transaction(|nodes| instantiate_source(&malformed, nodes))
            .unwrap_err();
        assert!(
            error.to_string().contains("non-finite or nonliteral acmag"),
            "{error}"
        );
        assert_eq!(nodes.nodes(), before);

        let mut nodes = NodeTable::new();
        let source = nodes
            .transaction(|nodes| instantiate_source(instance, nodes))
            .unwrap();
        let port = source.rf_port().unwrap();
        assert_eq!(port.z0.to_bits(), 75_f64.to_bits());
        assert_eq!(port.internal, nodes.get("v1#res").unwrap());
    }
    #[test]
    fn waveform_alias_uses_shared_kind_and_mismatched_ast_is_rejected() {
        let deck = Parser::new()
            .parse_deck(&parse_deck_text(
                Path::new("waveform-alias.cir"),
                "title\nv1 a 0 sine(0 1 1k)\n.end\n",
            ))
            .unwrap();
        let mut instance = deck.devices[0].clone();
        assert!(
            NodeTable::new()
                .transaction(|nodes| instantiate_source(&instance, nodes))
                .is_ok()
        );
        let parameter = instance
            .parameters
            .iter_mut()
            .find(|p| matches!(p.kind, ParameterKind::Waveform(_)))
            .unwrap();
        parameter.name = "exp".to_owned();
        let mut nodes = NodeTable::new();
        let before = nodes.nodes().to_vec();
        let error = nodes
            .transaction(|nodes| instantiate_source(&instance, nodes))
            .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("does not match its typed coefficients")
        );
        assert_eq!(nodes.nodes(), before);
    }

    #[test]
    fn late_semiconductor_collisions_rollback_factory_and_enclosing_batch() {
        use crate::devices::{ModelContext, ModelResolver};
        let context = ModelContext::default();
        for (card, model) in [
            ("q1 c b e test", "npn rc=2 rb=10 re=1"),
            ("j1 d g s test", "njf rd=10 rs=20"),
            ("j1 d g s test", "njf level=2 rd=10 rs=20"),
            ("z1 d g s test", "nmf level=1 rd=10 rs=20"),
            ("z1 d g s test", "nmf level=2 rd=10 rs=20 rg=30"),
            ("z1 d g s test", "nhfet level=5 rd=10 rs=20 rg=30"),
            ("z1 d g s test", "nhfet level=6 rd=10 rs=20"),
            ("m1 d g s b test l=1u w=10u", "nmos level=1 rd=10 rs=20"),
            (
                "m1 d g s b test l=1u w=10u nrd=2 nrs=3",
                "nmos level=8 rsh=10",
            ),
            ("m1 d g s b test l=1u w=10u", "nmos level=14 rdsmod=1"),
            ("d1 a b test", "d rs=10"),
        ] {
            let deck = Parser::new()
                .parse_deck(&parse_deck_text(
                    Path::new("node-rollback.cir"),
                    &format!("t\n{card}\n.model test {model}\n.end\n"),
                ))
                .unwrap();
            let instance = &deck.devices[0];
            let models = ModelResolver::new(&deck.models).unwrap();
            let mut clean = NodeTable::new();
            let device = clean
                .transaction(|nodes| instantiate_with_models(instance, nodes, &models, &context))
                .unwrap_or_else(|error| panic!("{model}: {error}"));
            let collision = clean
                .nodes()
                .iter()
                .rev()
                .find(|node| node.kind == NodeKind::Internal)
                .unwrap_or_else(|| panic!("{model}: no internal nodes"))
                .name
                .clone();
            assert!(device.terminals().len() > instance.nodes.len());
            let mut table = NodeTable::new();
            // A pre-existing mixed-case internal name and an ordinary terminal.
            let existing = table.intern(&collision.to_ascii_uppercase()).unwrap();
            table.set_kind(existing, NodeKind::BranchCurrent);
            table.intern("keep").unwrap();
            let before = table.nodes().to_vec();
            table
                .transaction(|parent| {
                    let parent_id = parent.intern("parent")?;
                    let parent_before = parent.nodes().to_vec();
                    let error =
                        instantiate_with_models(instance, parent, &models, &context).unwrap_err();
                    assert!(error.to_string().contains("collision"), "{model}: {error}");
                    assert_eq!(parent.nodes(), parent_before, "{model}");
                    assert_eq!(parent.get(&collision), Some(existing));
                    assert_eq!(parent.get("PARENT"), Some(parent_id));
                    assert_eq!(parent.get("gnd"), Some(crate::primitives::NodeId::GROUND));
                    Err::<(), _>(SpiceError::circuit("enclosing batch failed"))
                })
                .unwrap_err();
            assert_eq!(table.nodes(), before, "{model}");
            // The enclosing circuit transaction rolls back a successfully
            // constructed device too, including its internal nodes.
            let mut empty = NodeTable::new();
            let before = empty.nodes().to_vec();
            empty
                .transaction(|parent| {
                    let device = instantiate_with_models(instance, parent, &models, &context)?;
                    assert_eq!(parent.nodes(), clean.nodes(), "{model}");
                    assert!(device.terminals().len() > instance.nodes.len());
                    Err::<(), _>(SpiceError::circuit("later card failed"))
                })
                .unwrap_err();
            assert_eq!(empty.nodes(), before);
        }
    }

    #[test]
    #[ignore = "opt-in semiconductor elaboration scaling benchmark"]
    fn semiconductor_node_transaction_scaling() {
        use crate::devices::{ModelContext, ModelResolver};
        use std::hint::black_box;
        use std::time::Instant;
        let deck = Parser::new()
            .parse_deck(&parse_deck_text(
                Path::new("node-bench.cir"),
                "t\nj1 d g s jm\n.model jm njf rd=10 rs=20\n.end\n",
            ))
            .unwrap();
        let resolver = ModelResolver::new(&deck.models).unwrap();
        let context = ModelContext::default();
        for count in [1_000_u32, 10_000, 100_000] {
            let instance_template = &deck.devices[0];
            let instances: Vec<_> = (0..count)
                .map(|index| {
                    let mut instance = instance_template.clone();
                    instance.name = format!("j{index:06}");
                    instance.nodes = ["d", "g", "s"]
                        .map(|prefix| format!("{prefix}{index:06}"))
                        .to_vec();
                    instance
                })
                .collect();
            let mut table = NodeTable::new();
            let start = Instant::now();
            table
                .transaction(|nodes| {
                    for instance in &instances {
                        black_box(instantiate_with_models(
                            instance, nodes, &resolver, &context,
                        )?);
                    }
                    Ok(())
                })
                .unwrap();
            let elapsed = start.elapsed();
            assert_eq!(table.len(), 1 + 5 * count as usize);
            // Every old clone copied each existing Node plus each map key/id
            // and both name strings. Count logical payload bytes, excluding
            // allocator and hash bucket overhead. New internal names are
            // fixed width, so the old cumulative copy traffic is exact.
            let n = u64::from(count);
            let copied_entries = n + 5 * n * (n - 1) / 2;
            let entry_size = (std::mem::size_of::<crate::primitives::Node>()
                + std::mem::size_of::<(String, crate::primitives::NodeId)>())
                as u64;
            let names_per_device = table.nodes()[1..6]
                .iter()
                .map(|node| node.name.len() as u64)
                .sum::<u64>();
            let cloned_bytes =
                entry_size * copied_entries + 2 * (n + names_per_device * n * (n - 1) / 2);
            println!(
                "{count} instances, {} nodes: transaction {elapsed:?}, old clone path {copied_entries} existing nodes plus map entries, >= {cloned_bytes} copied bytes, new path 0 existing entries copied",
                table.len()
            );
            if count <= 10_000 {
                let mut baseline = NodeTable::new();
                let start = Instant::now();
                for instance in &instances {
                    // Reconstruct the removed clone-and-replace operation,
                    // keeping the same factory body and model validation.
                    let mut cloned = baseline.clone();
                    cloned
                        .transaction(|nodes| {
                            instantiate_with_models(instance, nodes, &resolver, &context)
                        })
                        .unwrap();
                    baseline = cloned;
                }
                println!(
                    "{count} instances: clone-and-replace baseline {:?}",
                    start.elapsed()
                );
                assert_eq!(table.nodes(), baseline.nodes());
            }
        }
    }
}
