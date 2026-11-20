//! Scalar linear-device elaboration; unsupported parameters never disappear.
use crate::devices::models::ModelFamily;
use crate::devices::passive::PassiveKind;
use crate::devices::rlc::ResistorNoise;
use crate::devices::{
    AmSpec, Capacitor, Device, ExpSpec, FunctionSpec, IndependentSource, Inductor, PulseSpec,
    PwlSource, Resistor, SffmSpec, SineSpec, Waveform,
};
use crate::netlist::ast::{
    DeviceInstance, FunctionWaveform, ParameterKind, PositionedValue, SourceFunction,
    SourceWaveform,
};
use crate::netlist::source::{Deck, LogicalLine};
use crate::netlist::{Parser, RawCard};
use crate::primitives::{
    Complex, NodeKind, NodeTable, Real, SourceLoc, SpiceError, SpiceResult, parse_spice_number,
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
            terminals[slot] = nodes.intern(node);
        }
        for (slot, suffix) in suffixes.iter().enumerate() {
            let name = format!("{}#{suffix}", instance.name);
            if nodes.get(&name).is_some() {
                return Err(SpiceError::circuit(format!(
                    "{label} internal-node name collision: {name}"
                )));
            }
            let node = nodes.intern(&name);
            nodes.set_kind(node, NodeKind::Internal);
            terminals[4 + slot] = node;
        }
        Ok(terminals)
    })
}

/// Parse the shared scalar parameter syntax without conflating the two
/// factories' supported setter lists or dropping intermediate errors.
fn scalar_number(p: &crate::netlist::ast::ParameterAssignment) -> SpiceResult<Real> {
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
            nodes.intern(&instance.nodes[0]),
            nodes.intern(&instance.nodes[1]),
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

struct SourceSetters {
    value: Option<Real>,
    mag: Real,
    phase: Real,
    ac_given: bool,
    waveform: Option<Waveform>,
    port: PortSetters,
    distortion: [Option<crate::devices::distortion::DistortionInput>; 2],
}

fn parse_source_setters(instance: &DeviceInstance) -> SpiceResult<SourceSetters> {
    let mut value = None;
    let mut mag = 0.;
    let mut phase: f64 = 0.;
    // C `VSRCacGiven`/`ISRCacGiven`: any AC setter, even `ac 0`.
    let mut ac_given = false;
    // C applies waveform setters in order, so the last one wins.
    let mut waveform = None;
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
            waveform = Some(source_waveform(&p.name, source, &p.location)?);
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
        let allowed = matches!(
            p.name.as_str(),
            "dc" | "acmag"
                | "acphase"
                | "r"
                | "td"
                | "distof1mag"
                | "distof1phase"
                | "distof2mag"
                | "distof2phase"
        ) || (instance.designator == 'v'
            && matches!(p.name.as_str(), "portnum" | "z0" | "pwr" | "freq" | "phase"));
        if !allowed {
            return Err(unsupported_parameter(instance, p));
        }
        let number = scalar_number(p)?;
        match p.name.as_str() {
            "acmag" => {
                mag = number;
                ac_given = true;
            }
            "acphase" => {
                phase = number;
                ac_given = true;
            }
            "distof1mag" | "distof2mag" | "distof1phase" | "distof2phase" => {
                let slot = &mut distortion[usize::from(p.name.starts_with("distof2"))];
                let input = slot.get_or_insert(crate::devices::distortion::DistortionInput {
                    magnitude: 1.,
                    phase: 0.,
                });
                if p.name.ends_with("mag") {
                    // A new setter replaces both values (`vsrcpar.c`).
                    *input = crate::devices::distortion::DistortionInput {
                        magnitude: number,
                        phase: 0.,
                    };
                } else {
                    input.phase = number;
                }
            }
            "td" => pwl_delay = Some((number, p.location.clone())),
            "portnum" | "z0" | "pwr" | "freq" | "phase" => {
                port.set(&p.name, number, &p.location)?;
            }
            "r" => {
                validate_pwl_repeat(instance, p, waveform.as_ref(), number)?;
                pwl_repeat = Some((number, p.location.clone()));
            }
            _ => value = Some(number),
        }
    }
    if pwl_delay.is_some() || pwl_repeat.is_some() {
        waveform = Some(pwl_options(waveform, pwl_delay, pwl_repeat)?);
    }
    Ok(SourceSetters {
        value,
        mag,
        phase,
        ac_given,
        waveform,
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
        value,
        mag,
        phase,
        ac_given,
        waveform,
        port,
        distortion,
    } = parse_source_setters(instance)?;
    let dc_given = value.is_some();
    let waveform_given = waveform.is_some();
    // An explicit DC value is kept apart from time forcing. Without one, DC
    // analyses see the waveform's time-zero level (vsrcload.c evaluates the
    // transient function at time 0 when DC is not given).
    let value = match (value, &waveform) {
        (Some(v), _) => v,
        (None, Some(w)) => w.time_zero()?,
        (None, None) => 0.,
    };
    // Intern in place and roll back on failure, so the caller's node table is
    // untouched by an error without cloning it per instance.
    append_nodes(nodes, |new_nodes| {
        let terminals = [
            new_nodes.intern(&instance.nodes[0]),
            new_nodes.intern(&instance.nodes[1]),
        ];
        let power_function = port.power_function.is_some();
        let rf = port.finish(instance, new_nodes)?;
        let tone = power_function.then(|| {
            rf.as_ref().map_or((0., 0.), |p| {
                (
                    // vsrctemp.c: sqrt(VSRCportPower * 4.0 * VSRCportZ0).
                    (p.power * 4. * p.z0).sqrt(),
                    2. * std::f64::consts::PI * p.frequency,
                )
            })
        });
        let device: Box<dyn Device> = {
            let source = IndependentSource::new(
                &instance.name,
                terminals,
                instance.designator == 'v',
                value,
                Complex::new(
                    mag * phase.to_radians().cos(),
                    mag * phase.to_radians().sin(),
                ),
                waveform.unwrap_or(Waveform::Constant(value)),
            )?
            .with_power_tone(tone)
            .with_ac_given(ac_given)
            .with_sensitivity_inputs(crate::devices::sensitivity::SourceInputs {
                dc_given,
                function_given: waveform_given,
                ac: ac_given.then_some((mag, phase)),
            })
            .with_distortion(distortion[0], distortion[1])?;
            // Only V cards accept port setters (parser and allow-list above).
            match rf {
                Some(rf) => Box::new(source.with_port(rf)?),
                None => Box::new(source),
            }
        };
        Ok(device)
    })
}

/// The ordered RFSPICE port setters of one V card (`vsrcpar.c`), resolved by
/// [`PortSetters::finish`] with `vsrctemp.c`'s rules.
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
    fn set(&mut self, name: &str, number: Real, at: &SourceLoc) -> SpiceResult<()> {
        self.any.get_or_insert_with(|| at.clone());
        match name {
            "portnum" => {
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
            "z0" => {
                self.z0 = number;
                self.z0_given = true;
            }
            "pwr" => {
                self.power = Some(number);
                self.power_function = Some(at.clone());
            }
            "freq" => {
                self.frequency = Some(number);
                self.power_function = Some(at.clone());
            }
            _ => self.phase = Some(number),
        }
        Ok(())
    }

    /// `vsrctemp.c`: a source is a port when `portnum` was given, is positive
    /// and its `z0` (50 when not given) is positive. Creates the port's
    /// internal `<name>#res` node in `nodes` (`vsrcset.c` `CKTmkVolt`).
    fn finish(
        self,
        instance: &DeviceInstance,
        nodes: &mut NodeTable,
    ) -> SpiceResult<Option<crate::devices::RfPort>> {
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
        let name = format!("{}#res", instance.name);
        if nodes.get(&name).is_some() {
            return Err(SpiceError::circuit(format!(
                "RF port internal-node name collision: {name}"
            )));
        }
        let internal = nodes.intern(&name);
        nodes.set_kind(internal, NodeKind::Internal);
        Ok(Some(crate::devices::RfPort {
            number,
            z0,
            internal,
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
        (_, SourceWaveform::Function(f)) => function_waveform(f, setter),
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
            if let Some(first) = points.first() {
                waveform.validate().map_err(|_| SpiceError::Unsupported {
                    feature: "PWL times must be nonnegative and strictly increasing".to_string(),
                    location: Some(first.time.location.clone()),
                })?;
            }
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
