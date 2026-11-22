//! Scalar linear-device elaboration; unsupported parameters never disappear.
use crate::devices::models::ModelFamily;
use crate::devices::passive::PassiveKind;
use crate::devices::rlc::ResistorNoise;
use crate::devices::sources::{SourceExcitation, SourceInputs};
use crate::devices::{
    AmSpec, Capacitor, Device, ExpSpec, FunctionSpec, Inductor, PulseSpec, PwlSource, Resistor,
    SffmSpec, SineSpec, Waveform,
};
use crate::netlist::ast::{
    DeviceInstance, FunctionWaveform, ParameterKind, PositionedValue, SourceFunction,
    SourceWaveform,
};
use crate::netlist::source::{Deck, LogicalLine};
use crate::netlist::{Parser, RawCard};
use crate::primitives::{
    NodeKind, NodeTable, Real, SourceLoc, SpiceError, SpiceResult, parse_spice_number,
};

/// Designators [`instantiate`] builds from the card alone (the registry's
/// `ported` entries). `tests/registry_support.rs` elaborates one instance of
/// every designator to keep this list and [`ELABORATED`] honest.
pub(super) const CARD_FACTORY: &[char] =
    &['r', 'c', 'l', 'v', 'i', 'e', 'f', 'g', 'h', 'b', 'k', 't'];

/// Designators built only while elaborating a deck, with the subset built:
/// D/Q/M/S/W from a resolved `.model` card ([`instantiate_with_models`]) and X
/// by subcircuit expansion (`crate::devices::subckt`). The registry's `bounded` entries.
pub(super) const ELABORATED: &[(char, &str)] = &[
    ('d', "junction diode (dioload.c subset)"),
    ('q', "Gummel-Poon BJT level 1 (bjtload.c)"),
    (
        'm',
        "MOS1, MOS2, MOS3, MOS6 and MOS9 (levels 1, 2, 3, 6 and 9), plus \
         BSIM3v3.3 (levels 8/49, DC operating point only)",
    ),
    ('s', "voltage-controlled switch (companion .tran, no BDF)"),
    ('w', "current-controlled switch (companion .tran, no BDF)"),
    ('x', "expanded before device elaboration"),
    (
        'j',
        "JFET levels 1 (jfetload.c, Sydney B tail) and 2 (Parker-Skellern psmodel.c)",
    ),
    (
        'u',
        "uniform RC line expanded into R/C/D sections (urcsetup.c)",
    ),
    (
        'o',
        "lossy line: LC, RLC, RC and RG cases (ltraload.c; companion .tran, no BDF)",
    ),
    (
        'y',
        "single lossy transmission line (TXL): `ReadTxL`'s Padé DC load and \
         recursive-convolution transient, or `inp2y.c`'s three-pi fallback \
         expansion; Padé AC/SP/pole-zero are refused (C reuses its time-domain load)",
    ),
    (
        'z',
        "every C level: MES (level 1, Statz), MESA 2-4 (mesa1/mesa2/mesa3) and HFET1/HFET2 (levels 5/6)",
    ),
];

pub(super) fn from_card(card: &RawCard, nodes: &mut NodeTable) -> SpiceResult<Box<dyn Device>> {
    let deck = Deck {
        path: card.location.path().to_path_buf(),
        title: String::new(),
        title_location: card.location.clone(),
        lines: vec![LogicalLine {
            location: card.location.clone(),
            end_line: card.location.line,
            continuations: 0,
            text: card.raw.clone(),
        }],
    };
    let netlist = Parser::new().parse_deck(&deck)?;
    let [instance] = netlist.devices.as_slice() else {
        return Err(SpiceError::parse(
            card.location.clone(),
            "device card must elaborate to exactly one instance",
        ));
    };
    instantiate(instance, nodes)
}

/// Resolve and validate before any node interning. Bounded R/C/L and M4 D/Q/M
/// factories reject unsupported physics before committing staged nodes.
/// Every builtin must honor the append-only node contract: intern nodes and
/// change only the kinds of newly created nodes. Circuit batch elaboration
/// uses that contract to roll back an entire failed batch.
pub(super) fn instantiate_with_models(
    instance: &DeviceInstance,
    nodes: &mut NodeTable,
    models: &crate::devices::models::ModelResolver<'_>,
    context: &crate::devices::models::ModelContext,
) -> SpiceResult<Box<dyn Device>> {
    context.validate(&instance.location)?;
    let Some(model) = models.resolve(instance)? else {
        return instantiate(instance, nodes);
    };
    // The resolver validates supported levels. Keep dispatch explicit so
    // adding a new supported level cannot silently select a different backend.
    let unsupported_level = || {
        SpiceError::not_yet_ported(
            format!(
                "{}: model-backed {:?} level {} factory for '{}'",
                instance.location,
                model.family(),
                model.levels().selector,
                instance.name
            ),
            "src/spicelib/parser/inpdomod.c",
        )
    };
    match model.family() {
        ModelFamily::Resistor | ModelFamily::Capacitor | ModelFamily::Inductor => {
            crate::devices::passive::instantiate(instance, nodes, &model, context)
        }
        ModelFamily::Diode => {
            crate::devices::nonlinear::Diode::instantiate(instance, nodes, &model, context)
        }
        ModelFamily::Npn | ModelFamily::Pnp => {
            crate::devices::bjt::Bjt::instantiate(instance, nodes, &model, context)
        }
        ModelFamily::Nmos | ModelFamily::Pmos => match model.levels().selector {
            1 => crate::devices::mos1::Mos1::instantiate(instance, nodes, &model, context),
            2 => crate::devices::mos2::Mos2::instantiate(instance, nodes, &model, context),
            3 => crate::devices::mos3::Mos3::instantiate(instance, nodes, &model, context),
            6 => crate::devices::mos6::Mos6::instantiate(instance, nodes, &model, context),
            9 => crate::devices::mos9::Mos9::instantiate(instance, nodes, &model, context),
            8 | 49 => crate::devices::bsim3::instantiate(instance, nodes, &model, context),
            14 | 54 => crate::devices::bsim4::instantiate(instance, nodes, &model, context),
            _ => Err(unsupported_level()),
        },
        ModelFamily::Njf | ModelFamily::Pjf => match model.levels().selector {
            1 => crate::devices::jfet::Jfet::instantiate(instance, nodes, &model, context),
            2 => crate::devices::jfet2::Jfet2::instantiate(instance, nodes, &model, context),
            _ => Err(unsupported_level()),
        },
        ModelFamily::Nmf | ModelFamily::Pmf => match model.levels().selector {
            1 => crate::devices::mesfet::Mesfet::instantiate(instance, nodes, &model, context),
            2..=4 => {
                crate::devices::hfet::mesa::Mesa::instantiate(instance, nodes, &model, context)
            }
            5 => crate::devices::hfet::hfet1::Hfet1::instantiate(instance, nodes, &model, context),
            6 => crate::devices::hfet::hfet2::Hfet2::instantiate(instance, nodes, &model, context),
            _ => Err(unsupported_level()),
        },
        ModelFamily::Switch | ModelFamily::CurrentSwitch => {
            crate::devices::switch::Switch::instantiate(instance, nodes, &model, context)
        }
        ModelFamily::Txl => crate::devices::txl::Txl::instantiate(instance, nodes, &model),
        ModelFamily::Ltra => {
            crate::devices::ltra::LossyLine::instantiate(instance, nodes, models, context)
                .map(|line| Box::new(line) as Box<dyn Device>)
        }
        // URC elaborates to a ladder, not a single device.
        ModelFamily::Urc => Err(SpiceError::circuit(format!(
            "URC {} is expanded by Circuit::add_instances, not built as one device",
            instance.name
        ))),
    }
}

pub(super) fn instantiate(
    instance: &DeviceInstance,
    nodes: &mut NodeTable,
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

/// Execute an append-only node-table transaction. Builders may intern
/// nodes and assign newly created node kinds, but must not modify existing
/// nodes. An error discards just the appended suffix, not the entire table.
pub(super) fn append_nodes<T>(
    nodes: &mut NodeTable,
    build: impl FnOnce(&mut NodeTable) -> SpiceResult<T>,
) -> SpiceResult<T> {
    let mark = nodes.len();
    let result = build(nodes);
    if result.is_err() {
        nodes.truncate(mark);
    }
    result
}

/// Build a four-terminal line's internal unknowns in C setup order. On a
/// generated-name collision, rollback only the newly interned nodes.
pub(super) fn line_terminals<const N: usize>(
    instance: &DeviceInstance,
    nodes: &mut NodeTable,
    suffixes: &[&str],
    label: &str,
) -> SpiceResult<[crate::primitives::NodeId; N]> {
    use crate::primitives::NodeId;
    if instance.nodes.len() != 4 || N != 4 + suffixes.len() {
        return Err(SpiceError::circuit(
            "invalid transmission-line terminal layout",
        ));
    }
    append_nodes(nodes, |nodes| {
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
            nodes.set_kind(node, NodeKind::Internal);
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
    nodes: &mut NodeTable,
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
    append_nodes(nodes, |nodes| {
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
    PwlSource::new(knots.clone(), 0., Some(number)).map_err(|error| SpiceError::Unsupported {
        feature: format!("{}: {error}", instance.name),
        location: Some(p.location.clone()),
    })?;
    Ok(())
}

fn instantiate_source(
    instance: &DeviceInstance,
    nodes: &mut NodeTable,
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
    append_nodes(nodes, |nodes| {
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
        nodes: &mut NodeTable,
    ) -> SpiceResult<crate::devices::RfPort> {
        let name = format!("{}#res", instance.name);
        if nodes.get(&name).is_some() {
            return Err(SpiceError::circuit(format!(
                "RF port internal-node name collision: {name}"
            )));
        }
        let internal = nodes.intern(&name)?;
        nodes.set_kind(internal, NodeKind::Internal);
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
    PwlSource::new(knots, delay.map_or(0., |(d, _)| d), repeat.map(|(r, _)| r))
        .map(Waveform::PwlSource)
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
    match (name, source) {
        (name, SourceWaveform::Function(f)) if name == f.function.keyword() => {
            function_waveform(f, setter)
        }
        ("pulse", SourceWaveform::Pulse(p)) => {
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
        ("pwl", SourceWaveform::Pwl(points)) => {
            let knots = points
                .iter()
                .map(|p| Ok((field(&p.time, "pwl time")?, field(&p.value, "pwl value")?)))
                .collect::<SpiceResult<Vec<_>>>()?;
            let waveform = Waveform::Pwl(knots);
            waveform.validate().map_err(|_| SpiceError::Unsupported {
                feature: "PWL times must be nonnegative and strictly increasing".to_owned(),
                location: Some(
                    points
                        .first()
                        .map_or_else(|| setter.clone(), |p| p.time.location.clone()),
                ),
            })?;
            Ok(waveform)
        }
        _ => Err(SpiceError::not_yet_ported(
            format!("source waveform '{name}'"),
            "src/spicelib/devices/vsrc/vsrcload.c",
        )),
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
        assert!(ac.phase.is_none());
        assert!(input.port.power_function.is_some());
        let mut nodes = NodeTable::new();
        let existing = nodes.intern("v1#res").unwrap();
        nodes.set_kind(existing, NodeKind::Internal);
        let before = nodes.nodes().to_vec();
        let error = instantiate_source(instance, &mut nodes).unwrap_err();
        assert!(error.to_string().contains("collision"));
        assert_eq!(nodes.nodes(), before);

        let mut malformed = instance.clone();
        let mut invalid = malformed.parameters[0].clone();
        invalid.name = "acmag".into();
        invalid.value = "not-a-number".into();
        malformed.parameters.insert(0, invalid);
        let error = instantiate_source(&malformed, &mut nodes).unwrap_err();
        assert!(
            error.to_string().contains("non-finite or nonliteral acmag"),
            "{error}"
        );
        assert_eq!(nodes.nodes(), before);

        let mut nodes = NodeTable::new();
        let source = instantiate_source(instance, &mut nodes).unwrap();
        let port = source.rf_port().unwrap();
        assert_eq!(port.z0.to_bits(), 75_f64.to_bits());
        assert_eq!(port.internal, nodes.get("v1#res").unwrap());
    }
}
