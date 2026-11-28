//! Voltage- and current-controlled switches: S (model `sw`) and W (model `csw`).
//!
//! C references (read-only): `src/spicelib/parser/inp2s.c`, `inp2w.c`;
//! `src/spicelib/devices/sw/` (`sw.c`, `swsetup.c`, `swmparam.c`,
//! `swparam.c`, `swload.c`, `swacload.c`, `swtrunc.c`) and
//! `src/spicelib/devices/csw/` (`csw.c`, `cswsetup.c`, `cswmpar.c`,
//! `cswparam.c`, `cswload.c`, `cswacld.c`, `cswtrunc.c`).
//!
//! # Equations
//!
//! A switch is a conductance `g` between its first and second terminals,
//! `g = 1/RON` when closed and `g = 1/ROFF` when open (defaults: `RON` gives
//! 1 S, an omitted `ROFF` gives the circuit `gmin`, as `SW_OFF_CONDUCTANCE` is
//! `CKTgmin`). The control is `v(nc+) - v(nc-)` for S and the branch current of
//! the named voltage source for W (positive from its first terminal through it
//! to its second, as for F/H). Thresholds are `VT`/`VH` for S and `IT`/`IH` for
//! W, all defaulting to zero.
//!
//! # Discrete state
//!
//! Slot 0 holds the switch state with C's codes: `0` really off, `1` really
//! on, `2` off inside the hysteresis band, `3` on inside it; codes 1 and 3
//! stamp the on conductance. Slot 1 holds the control value. The state is
//! decided per Newton load from the [`IterationPhase`] exactly as
//! `SWload`/`CSWload` decide it from `MODEINITF`:
//!
//! * [`IterationPhase::Junction`]/[`IterationPhase::Fix`] (`MODEINITJCT`/
//!   `MODEINITFIX`): the instance `on`/`off` flag. An `on` switch is really on
//!   above `VT + |VH|` and otherwise on in the band; an `off` switch is really
//!   off below `VT - |VH|` and otherwise off in the band.
//! * [`IterationPhase::Predict`] (`MODEINITTRAN`/`MODEINITPRED`): for
//!   `VH > 0`, on above `VT + VH`, off below `VT - VH`, else the latest
//!   **accepted** state. For `VH <= 0` (including the default 0) the bounds
//!   are `VT - VH` and `VT + VH` and a value inside the band maps the accepted
//!   state as C does (S: band states kept, really-on becomes really-off and
//!   really-off really-on; W: really-on becomes off-in-band, really-off
//!   on-in-band).
//! * [`IterationPhase::Float`] (`MODEINITFLOAT`): the same bounds; inside a
//!   positive band S keeps the **previous iterate's** state (C `CKTstate0`)
//!   while W keeps the accepted state (C reads `CKTstate1` here), and the
//!   `VH <= 0` band maps the accepted state like S's `MODEINITFLOAT` branch
//!   (band states kept, really-on to off-in-band, really-off to on-in-band). A
//!   state different from the previous iterate's reports nonconvergence (C
//!   `CKTnoncon++`, "ensure one more iteration").
//!
//! Where C reads `CKTstate1` before any point was accepted (a plain `.op`),
//! its state vectors are zero-initialised, i.e. "really off"; the port uses
//! the same value. Only an accepted point ([`crate::devices::Circuit::accept_point`])
//! ever changes the accepted state; trial and rejected loads cannot.
//!
//! # Timestep control
//!
//! [`Device::timestep_limit`] follows `swtrunc.c`: while really off with the
//! control rising towards `VT + VH`, or (any other state) falling towards
//! `VT - VH`, the next step is limited to
//! `(0.75 (ref - c) +/- 0.05) / (c - c_accepted) * dt`. C's `cswtrunc.c` uses
//! the same rule with `5e-5`, but `CSWload` never stores the control value (the
//! store is commented out, a `FIXME` in C), so the C limit never applies to W;
//! the port reproduces that and W sets no limit.
//!
//! # AC
//!
//! `SWacLoad`/`CSWacLoad` and noise treat every nonzero state code,
//! including "off in band", as on. The port applies that rule to the
//! converged operating-point state rather than copying `CKTstate1` (still
//! zero after a plain operating point) into `CKTstate0` as `MODEINITSMSIG`
//! does in C. Assemblies without a supplied state use the instance flag's
//! DC conductance instead.

mod ltspice;

use crate::maths::Vector;
use crate::netlist::ast::{DeviceInstance, ParameterKind};
use crate::primitives::{NodeAppend, NodeId, Real, SpiceError, SpiceResult};

use crate::devices::noise::{DeviceNoise, NoiseContext, NoiseFamily, NoiseKind, NoiseSource};
use crate::devices::schema::{
    ScalarDomain, ScalarParameter, ScalarSchema, ScalarUnit, ScalarValues,
};
use crate::devices::state::IterationPhase;
use crate::devices::traits::{ControlReference, Device, StampContext, TruncationContext};
use crate::devices::{LinearContext, ModelContext, ModelFamily, ResolvedModel};

/// Which switch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SwitchKind {
    /// S: voltage-controlled switch, model `sw`.
    Voltage,
    /// W: current-controlled switch, model `csw`.
    Current,
}

impl SwitchKind {
    const fn designator(self) -> char {
        match self {
            Self::Voltage => 's',
            Self::Current => 'w',
        }
    }

    const fn base(self) -> &'static str {
        match self {
            Self::Voltage => "sw",
            Self::Current => "csw",
        }
    }
}

/// C's switch-state codes (`swload.c`: `REALLY_OFF = 0`, `REALLY_ON = 1`,
/// `HYST_OFF = 2`, `HYST_ON = 3`), stored as the value of state slot 0.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SwitchState {
    /// Open, control outside the hysteresis band.
    ReallyOff,
    /// Closed, control outside the hysteresis band.
    ReallyOn,
    /// Open, control inside the hysteresis band.
    HysteresisOff,
    /// Closed, control inside the hysteresis band.
    HysteresisOn,
}

impl SwitchState {
    /// The C code stored in the state vector.
    #[must_use = "the returned numeric code represents the switch state in device history"]
    pub const fn code(self) -> Real {
        match self {
            Self::ReallyOff => 0.,
            Self::ReallyOn => 1.,
            Self::HysteresisOff => 2.,
            Self::HysteresisOn => 3.,
        }
    }

    /// The state for a stored code.
    #[must_use = "the result decodes a valid stored switch state or rejects it"]
    pub fn from_code(code: Real) -> Option<Self> {
        match code {
            0. => Some(Self::ReallyOff),
            1. => Some(Self::ReallyOn),
            2. => Some(Self::HysteresisOff),
            3. => Some(Self::HysteresisOn),
            _ => None,
        }
    }

    /// True when the on conductance is stamped (`REALLY_ON` or `HYST_ON`).
    #[must_use = "the result determines which branch conductance is stamped"]
    pub const fn is_closed(self) -> bool {
        matches!(self, Self::ReallyOn | Self::HysteresisOn)
    }

    /// AC, PZ and noise in C regard every nonzero state code as on.
    const fn is_small_signal_closed(self) -> bool {
        !matches!(self, Self::ReallyOff)
    }

    /// Enter the negative/zero hysteresis band from the accepted state.
    /// Voltage-switch prediction swaps the outer states; the other paths
    /// map them into the opposite inner states.
    const fn enter_band(self, swap_outer: bool) -> Self {
        match self {
            Self::HysteresisOff | Self::HysteresisOn => self,
            Self::ReallyOn if swap_outer => Self::ReallyOff,
            Self::ReallyOff if swap_outer => Self::ReallyOn,
            Self::ReallyOn => Self::HysteresisOff,
            Self::ReallyOff => Self::HysteresisOn,
        }
    }
}

/// Validated model data at setup: thresholds and the explicit resistances.
#[derive(Debug, Clone, Copy, PartialEq)]
struct SwitchModel {
    threshold: Real,
    hysteresis: Real,
    /// `1/RON`, or C's default 1 S.
    on_conductance: Real,
    /// `1/ROFF`; `None` selects the load's `gmin` (`SW_OFF_CONDUCTANCE`).
    off_conductance: Option<Real>,
    /// Explicit source-selected constitutive law; None retains native SW/CSW.
    ltspice: Option<ltspice::Law>,
    /// Level-1 transient series inductance, zero for ordinary native switches.
    series_inductance: Real,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BandRegion {
    Below,
    Inside,
    Above,
}

impl SwitchModel {
    /// Strict comparisons: the endpoints belong to the hysteresis band.
    /// Keep C's signed-hysteresis addition/subtraction order.
    fn region(self, control: Real) -> BandRegion {
        let (lower, upper) = if self.hysteresis > 0. {
            (
                self.threshold - self.hysteresis,
                self.threshold + self.hysteresis,
            )
        } else {
            (
                self.threshold + self.hysteresis,
                self.threshold - self.hysteresis,
            )
        };
        if control > upper {
            BandRegion::Above
        } else if control < lower {
            BandRegion::Below
        } else {
            BandRegion::Inside
        }
    }
}

fn schema(kind: SwitchKind, ltspice: bool) -> &'static ScalarSchema<'static> {
    const SW: &[ScalarParameter] = &[
        ScalarParameter {
            name: "vt",
            unit: ScalarUnit::Volt,
            domain: ScalarDomain::Finite,
            default: Some(0.),
        },
        ScalarParameter {
            name: "vh",
            unit: ScalarUnit::Volt,
            domain: ScalarDomain::Finite,
            default: Some(0.),
        },
        ScalarParameter {
            name: "ron",
            unit: ScalarUnit::Ohm,
            domain: ScalarDomain::NonZero,
            default: None,
        },
        ScalarParameter {
            name: "roff",
            unit: ScalarUnit::Ohm,
            domain: ScalarDomain::NonZero,
            default: None,
        },
    ];
    const CSW: &[ScalarParameter] = &[
        ScalarParameter {
            name: "it",
            unit: ScalarUnit::Ampere,
            domain: ScalarDomain::Finite,
            default: Some(0.),
        },
        ScalarParameter {
            name: "ih",
            unit: ScalarUnit::Ampere,
            domain: ScalarDomain::Finite,
            default: Some(0.),
        },
        ScalarParameter {
            name: "ron",
            unit: ScalarUnit::Ohm,
            domain: ScalarDomain::NonZero,
            default: None,
        },
        ScalarParameter {
            name: "roff",
            unit: ScalarUnit::Ohm,
            domain: ScalarDomain::NonZero,
            default: None,
        },
    ];
    const LTSPICE: &[ScalarParameter] = &[
        SW[0],
        SW[1],
        ScalarParameter {
            domain: ScalarDomain::Positive,
            ..SW[2]
        },
        ScalarParameter {
            domain: ScalarDomain::Positive,
            ..SW[3]
        },
        ScalarParameter {
            name: "ilimit",
            unit: ScalarUnit::Ampere,
            domain: ScalarDomain::Finite,
            default: None,
        },
        ScalarParameter {
            name: "lser",
            unit: ScalarUnit::Henry,
            domain: ScalarDomain::Finite,
            default: Some(0.),
        },
        ScalarParameter {
            name: "vser",
            unit: ScalarUnit::Volt,
            domain: ScalarDomain::Finite,
            default: Some(0.),
        },
        ScalarParameter {
            name: "epsilon",
            unit: ScalarUnit::Volt,
            domain: ScalarDomain::Finite,
            default: Some(0.),
        },
    ];
    static LTSPICE_SCHEMA: ScalarSchema<'static> = ScalarSchema::new(LTSPICE);
    static SW_SCHEMA: ScalarSchema<'static> = ScalarSchema::new(SW);
    static CSW_SCHEMA: ScalarSchema<'static> = ScalarSchema::new(CSW);
    if ltspice {
        return &LTSPICE_SCHEMA;
    }
    match kind {
        SwitchKind::Voltage => &SW_SCHEMA,
        SwitchKind::Current => &CSW_SCHEMA,
    }
}

/// An S or W switch. See the [module documentation](self).
#[derive(Debug, Clone, PartialEq)]
pub struct Switch {
    name: String,
    /// The model card's name, for C's `.noise` visiting order.
    model_name: String,
    kind: SwitchKind,
    /// `[n+, n-]`, then `[nc+, nc-]` for S.
    terminals: Vec<NodeId>,
    /// W's controlling voltage source; empty for S.
    control: Vec<ControlReference>,
    model: SwitchModel,
    /// The last `on`/`off` flag (C `SWzero_stateGiven`): `true` for `on`.
    initially_on: bool,
    /// The model setters as written, for `.sens`.
    written: crate::devices::sensitivity::SwitchValues,
}

/// State slot of the switch state code.
const STATE: usize = 0;
/// State slot of the control value.
const CONTROL: usize = 1;

fn resolve_ltspice_law(
    model: &ResolvedModel<'_>,
    values: &ScalarValues,
    hysteresis: Real,
    oneway: bool,
) -> SpiceResult<ltspice::Law> {
    let card = model.card();
    let get = |name: &str| values.get(name).map(|value| value.value);
    let level = model.levels().selector;
    if level == 2 && get("lser").unwrap_or(0.) != 0. {
        return Err(SpiceError::Unsupported {
            feature: "LTspice level-2 switch does not support Lser".into(),
            location: Some(card.location.clone()),
        });
    }
    let offset = get("vser").unwrap_or(0.);
    let epsilon = get("epsilon").unwrap_or(0.);
    let direction = if level == 2 {
        let epsilon = epsilon.max(0.);
        if oneway || offset > 0. || epsilon > 0. {
            ltspice::Direction::OneWay { offset, epsilon }
        } else {
            ltspice::Direction::Bidirectional
        }
    } else {
        if oneway || epsilon != 0. {
            return Err(SpiceError::Unsupported {
                feature: "LTspice level-1 switch does not support oneway or epsilon".into(),
                location: Some(card.location.clone()),
            });
        }
        if offset == 0. {
            ltspice::Direction::Bidirectional
        } else {
            ltspice::Direction::Offset(offset)
        }
    };
    if level == 2 && hysteresis == 0. {
        return Err(SpiceError::Unsupported {
            feature: "LTspice level-2 switch with zero transition width".into(),
            location: Some(card.location.clone()),
        });
    }
    Ok(ltspice::Law::new(
        level,
        hysteresis,
        get("ilimit"),
        direction,
    ))
}

/// Read the model setters and preserve the written values for sensitivity.
fn resolve_switch_model(
    kind: SwitchKind,
    model: &ResolvedModel<'_>,
) -> SpiceResult<(SwitchModel, crate::devices::sensitivity::SwitchValues)> {
    let card = model.card();
    let is_ltspice = model.family() == ModelFamily::LtspiceSwitch;
    let mut setters = Vec::new();
    let mut oneway = false;
    for parameter in &card.parameters {
        if parameter.kind == ParameterKind::Flag && parameter.name == kind.base() {
            // SW_MOD_SW / CSW_CSW: "just says that this is a switch".
            continue;
        }
        if is_ltspice && parameter.kind == ParameterKind::Flag && parameter.name == "oneway" {
            oneway = true;
            continue;
        }
        if parameter.name.eq_ignore_ascii_case("level") {
            if is_ltspice {
                continue;
            }
            return Err(SpiceError::Unsupported {
                feature: format!(
                    "'level' on {} model '{}' (C ignores it with a warning)",
                    kind.base(),
                    card.name
                ),
                location: Some(parameter.location.clone()),
            });
        }
        setters.push(parameter);
    }
    let values = schema(kind, is_ltspice).validate(setters, &card.location)?;
    let (threshold, hysteresis) = match kind {
        SwitchKind::Voltage => ("vt", "vh"),
        SwitchKind::Current => ("it", "ih"),
    };
    let get = |name: &str| values.get(name).map(|value| value.value);
    let conductance = |name: &str| -> SpiceResult<Option<Real>> {
        let Some(value) = values.get(name) else {
            return Ok(None);
        };
        let g = 1. / value.value;
        if !g.is_finite() {
            return Err(SpiceError::parse(
                value
                    .location
                    .clone()
                    .unwrap_or_else(|| card.location.clone()),
                format!("{name}={} has no finite conductance", value.value),
            ));
        }
        Ok(Some(g))
    };
    let written_value = |name: &str| {
        values
            .get(name)
            .filter(|value| value.location.is_some())
            .map(|value| value.value)
    };
    let written = crate::devices::sensitivity::SwitchValues {
        threshold: written_value(threshold),
        hysteresis: written_value(hysteresis),
        on: written_value("ron"),
        off: written_value("roff"),
    };
    let hysteresis = get(hysteresis).unwrap_or(0.);
    let ltspice = is_ltspice
        .then(|| resolve_ltspice_law(model, &values, hysteresis, oneway))
        .transpose()?;
    let parameters = SwitchModel {
        threshold: get(threshold).unwrap_or(0.),
        hysteresis,
        on_conductance: conductance("ron")?.unwrap_or(1.),
        // Level 2 has an independent 10 MΩ default (verified at two gmin
        // values); level 1 and native switches use the circuit gmin.
        off_conductance: conductance("roff")?.or_else(|| {
            ltspice
                .filter(|law| law.transfer == ltspice::Transfer::Arctangent)
                .map(|_| 1e-7)
        }),
        ltspice,
        series_inductance: get("lser").unwrap_or(0.),
    };
    Ok((parameters, written))
}

impl Switch {
    /// Builds an S/W instance from its AST and resolved `sw`/`csw` model.
    /// Validation happens before any node is interned.
    pub(crate) fn instantiate(
        instance: &DeviceInstance,
        nodes: &mut NodeAppend<'_>,
        model: &ResolvedModel<'_>,
        _context: &ModelContext,
    ) -> SpiceResult<Box<dyn Device>> {
        let kind = match instance.designator {
            's' => SwitchKind::Voltage,
            'w' => SwitchKind::Current,
            other => {
                return Err(SpiceError::circuit(format!(
                    "'{other}' instance {} is not a switch",
                    instance.name
                )));
            }
        };
        let expected = if kind == SwitchKind::Voltage { 4 } else { 2 };
        if instance.nodes.len() != expected {
            return Err(SpiceError::parse(
                instance.location.clone(),
                format!(
                    "{} needs {expected} terminals, got {}",
                    instance.name,
                    instance.nodes.len()
                ),
            ));
        }
        let (parameters, written) = resolve_switch_model(kind, model)?;
        let card = model.card();
        let mut control = Vec::new();
        let mut initially_on = false;
        for parameter in &instance.parameters {
            match (&parameter.kind, parameter.name.as_str()) {
                (ParameterKind::Instance, "control") if kind == SwitchKind::Current => {
                    if !control.is_empty() {
                        return Err(SpiceError::parse(
                            parameter.location.clone(),
                            format!("{}: more than one controlling source", instance.name),
                        ));
                    }
                    control.push(ControlReference {
                        name: parameter.value.to_ascii_lowercase(),
                        location: Some(parameter.location.clone()),
                    });
                }
                (ParameterKind::Flag, "on") => initially_on = true,
                (ParameterKind::Flag, "off") => initially_on = false,
                _ => {
                    return Err(SpiceError::Unsupported {
                        feature: format!(
                            "{} parameter '{}' on a '{}' switch",
                            instance.name,
                            parameter.name,
                            kind.designator()
                        ),
                        location: Some(parameter.location.clone()),
                    });
                }
            }
        }
        if kind == SwitchKind::Current && control.is_empty() {
            return Err(SpiceError::parse(
                instance.location.clone(),
                format!("{} needs a controlling voltage source", instance.name),
            ));
        }
        let terminals = nodes.transaction(|staged| {
            let mut terminals = instance
                .nodes
                .iter()
                .map(|node| staged.intern(node))
                .collect::<SpiceResult<Vec<_>>>()?;
            if parameters.series_inductance != 0. {
                let label = format!("{}#lser", instance.name);
                if staged.get(&label).is_some() {
                    return Err(SpiceError::circuit(format!(
                        "hidden coordinate collision: {label}"
                    )));
                }
                let node = staged.intern(&label)?;
                staged.set_kind(node, crate::primitives::NodeKind::Internal)?;
                terminals.push(node);
            }
            Ok(terminals)
        })?;
        Ok(Box::new(Self {
            name: instance.name.clone(),
            model_name: card.name.clone(),
            kind,
            terminals,
            control,
            model: parameters,
            initially_on,
            written,
        }))
    }

    /// Which switch this is.
    #[must_use = "the returned kind identifies the switch control domain"]
    pub const fn kind(&self) -> SwitchKind {
        self.kind
    }

    /// The model card's name.
    #[must_use = "the returned name identifies the referenced switch model"]
    pub fn model_name(&self) -> &str {
        &self.model_name
    }

    /// The model setters as written (for `.sens`).
    pub(crate) const fn model_values(&self) -> crate::devices::sensitivity::SwitchValues {
        self.written
    }

    /// A copy with the given threshold, hysteresis and on/off conductances,
    /// for a `.sens` load.
    ///
    /// # Errors
    /// A nonfinite value.
    pub(crate) fn with_model(
        &self,
        threshold: Real,
        hysteresis: Real,
        on: Real,
        off: Real,
    ) -> SpiceResult<Self> {
        if ![threshold, hysteresis, on, off]
            .iter()
            .all(|v| v.is_finite())
        {
            return Err(SpiceError::circuit(format!(
                "{}: nonfinite perturbed switch model",
                self.name
            )));
        }
        Ok(Self {
            model: SwitchModel {
                threshold,
                hysteresis,
                on_conductance: on,
                off_conductance: Some(off),
                ltspice: None,
                series_inductance: 0.,
            },
            ..self.clone()
        })
    }

    /// `.sens` AC: whether `SWacLoad`/`CSWacLoad` stamps the on conductance
    /// after `cktsens.c`'s reset load, `SWload`/`CSWload` in `MODEINITFLOAT`
    /// with a zero control and zeroed state (`REALLY_OFF` in both
    /// `CKTstate0` and `CKTstate1`). The AC loads treat every non-zero state
    /// code as on.
    ///
    /// # Errors
    /// None in practice (the iterate is given).
    pub(crate) fn reset_closed(&self) -> SpiceResult<bool> {
        let state = self.decide(
            IterationPhase::Float,
            0.,
            SwitchState::ReallyOff,
            Some(SwitchState::ReallyOff),
        )?;
        Ok(state.is_small_signal_closed())
    }

    /// The control value at `solution`: `v(nc+) - v(nc-)` or the controlling
    /// branch current.
    fn control_value(
        &self,
        node: impl Fn(NodeId) -> Real,
        row: impl Fn(usize) -> SpiceResult<Real>,
        controls: &[usize],
    ) -> SpiceResult<Real> {
        match self.kind {
            SwitchKind::Voltage => Ok(node(self.terminals[2]) - node(self.terminals[3])),
            SwitchKind::Current => {
                let branch = controls.first().ok_or_else(|| {
                    SpiceError::circuit(format!(
                        "{}: controlling source is not bound to a branch row",
                        self.name
                    ))
                })?;
                row(*branch)
            }
        }
    }

    /// `SWload`/`CSWload` under `MODEINITJCT`/`MODEINITFIX`.
    fn flag_state(&self, control: Real) -> SwitchState {
        match (self.initially_on, self.model.region(control)) {
            (true, BandRegion::Above) => SwitchState::ReallyOn,
            (true, _) => SwitchState::HysteresisOn,
            (false, BandRegion::Below) => SwitchState::ReallyOff,
            (false, _) => SwitchState::HysteresisOff,
        }
    }

    /// The state for one load. `accepted` is C's `CKTstate1` (zero, "really
    /// off", before any accepted point); `iterate` is `CKTstate0` from the
    /// previous load of the solve.
    fn decide(
        &self,
        phase: IterationPhase,
        control: Real,
        accepted: SwitchState,
        iterate: Option<SwitchState>,
    ) -> SpiceResult<SwitchState> {
        // Select the state retained *inside* the band according to the
        // solver phase, independently of the physical control thresholds.
        let positive = self.model.hysteresis > 0. || self.model.ltspice.is_some();
        let inside = match phase {
            IterationPhase::Junction | IterationPhase::Fix => {
                return Ok(self.flag_state(control));
            }
            IterationPhase::Predict if positive => accepted,
            IterationPhase::Predict => accepted.enter_band(self.kind == SwitchKind::Voltage),
            IterationPhase::Float => {
                let iterate = iterate.ok_or_else(|| {
                    SpiceError::circuit(format!(
                        "{}: MODEINITFLOAT load without a previous iterate",
                        self.name
                    ))
                })?;
                if !positive {
                    accepted.enter_band(false)
                } else if self.kind == SwitchKind::Voltage {
                    iterate
                } else {
                    accepted
                }
            }
        };
        Ok(match self.model.region(control) {
            BandRegion::Above => SwitchState::ReallyOn,
            BandRegion::Below => SwitchState::ReallyOff,
            BandRegion::Inside => inside,
        })
    }

    fn conductance(&self, closed: bool, gmin: Real) -> Real {
        if closed {
            self.model.on_conductance
        } else {
            self.model.off_conductance.unwrap_or(gmin)
        }
    }

    /// Saved-state AC/noise semantics are intentionally different from the
    /// DC initialization rule used without a solved state. Evaluate the
    /// control lazily: a saved state needs no controlling branch lookup.
    fn small_signal_closed(
        &self,
        states: Option<&[Real]>,
        what: &str,
        control: impl FnOnce() -> SpiceResult<Real>,
    ) -> SpiceResult<bool> {
        if let Some(states) = states {
            let state = self
                .stored(states.get(self.switch_slot(STATE)).copied(), what)?
                .ok_or_else(|| {
                    SpiceError::circuit(format!("{}: missing {what} state", self.name))
                })?;
            Ok(state.is_small_signal_closed())
        } else {
            Ok(self.flag_state(control()?).is_closed())
        }
    }

    fn decode(&self, code: Real, what: &str) -> SpiceResult<SwitchState> {
        SwitchState::from_code(code).ok_or_else(|| SpiceError::Numerical {
            context: self.name.clone(),
            message: format!("invalid {what} switch state {code}"),
        })
    }

    fn stored(&self, value: Option<Real>, what: &str) -> SpiceResult<Option<SwitchState>> {
        value.map(|code| self.decode(code, what)).transpose()
    }

    const fn discrete(&self) -> bool {
        match self.model.ltspice {
            Some(law) => law.discrete(),
            None => true,
        }
    }

    fn load_state(
        &self,
        context: &mut StampContext<'_>,
        control: Real,
    ) -> SpiceResult<SwitchState> {
        let accepted = self
            .stored(
                context.states.accepted(1, self.switch_slot(STATE)),
                "accepted",
            )?
            .unwrap_or(SwitchState::ReallyOff);
        let iterate = self.stored(context.states.iterate(self.switch_slot(STATE)), "iterate")?;
        let phase = context.states.phase();
        let state = self.decide(phase, control, accepted, iterate)?;
        if phase == IterationPhase::Float && iterate != Some(state) {
            context.states.report_nonconvergence()?;
        }
        context.states.set(self.switch_slot(STATE), state.code())?;
        context.states.set(self.switch_slot(CONTROL), control)?;
        Ok(state)
    }

    fn series(&self) -> Option<crate::devices::rlc::ReactiveLoad<'_>> {
        self.terminals
            .get(4)
            .map(|&internal| crate::devices::rlc::ReactiveLoad {
                bias: None,
                name: &self.name,
                terminals: [self.terminals[0], internal],
                value: self.model.series_inductance,
                initial: None,
            })
    }

    const fn storage_slots(&self) -> usize {
        if self.model.series_inductance == 0. {
            0
        } else {
            crate::devices::rlc::REACTIVE_STATE_COUNT
        }
    }

    const fn switch_slot(&self, slot: usize) -> usize {
        self.storage_slots() + slot
    }

    fn conducting_terminals(&self) -> [NodeId; 2] {
        [
            self.terminals.get(4).copied().unwrap_or(self.terminals[0]),
            self.terminals[1],
        ]
    }

    fn ltspice_point(
        &self,
        law: ltspice::Law,
        voltage: Real,
        control: Real,
        closed: bool,
        gmin: Real,
    ) -> SpiceResult<ltspice::Point> {
        law.evaluate(
            self.ltspice_parameters(gmin),
            ltspice::Bias { voltage, control },
            closed,
        )
    }

    fn ltspice_parameters(&self, gmin: Real) -> ltspice::Conductance {
        ltspice::Conductance {
            on: self.model.on_conductance,
            off: self.model.off_conductance.unwrap_or(gmin),
            threshold: self.model.threshold,
            width: self.model.hysteresis.abs(),
            gmin,
        }
    }

    fn ltspice_closed(
        &self,
        law: ltspice::Law,
        states: Option<&[Real]>,
        control: Real,
    ) -> SpiceResult<bool> {
        if !law.discrete() {
            return Ok(false);
        }
        match self.model.region(control) {
            BandRegion::Above => return Ok(true),
            BandRegion::Below => return Ok(false),
            BandRegion::Inside => {}
        }
        if let Some(states) = states {
            let state = self
                .stored(states.get(self.switch_slot(STATE)).copied(), "LTspice bias")?
                .ok_or_else(|| SpiceError::circuit("missing LTspice bias state"))?;
            return Ok(state.is_closed());
        }
        Ok(self.initially_on)
    }

    fn ltspice_bias(&self, node: impl Fn(NodeId) -> Real) -> (Real, Real) {
        let [p, n] = self.conducting_terminals();
        (
            node(p) - node(n),
            node(self.terminals[2]) - node(self.terminals[3]),
        )
    }

    fn ltspice_entries(&self, jacobian: ltspice::Jacobian) -> [(NodeId, NodeId, Real); 8] {
        let [p, n] = self.conducting_terminals();
        let [cp, cn] = [self.terminals[2], self.terminals[3]];
        let g = jacobian.voltage;
        let gm = jacobian.control;
        [
            (p, p, g),
            (p, n, -g),
            (n, p, -g),
            (n, n, g),
            (p, cp, gm),
            (p, cn, -gm),
            (n, cp, -gm),
            (n, cn, gm),
        ]
    }

    fn stamp_ltspice(
        &self,
        context: &mut StampContext<'_>,
        voltage: Real,
        control: Real,
        point: ltspice::Point,
    ) -> SpiceResult<()> {
        context.stamp_fixed(self.ltspice_entries(point.jacobian))?;
        // A*x = b, with b = J*x - I(x).
        let rhs = point.jacobian.control.mul_add(
            control,
            point.jacobian.voltage.mul_add(voltage, -point.current),
        );
        let [p, n] = self.conducting_terminals();
        context.stamp_rhs(p, rhs)?;
        context.stamp_rhs(n, -rhs)
    }
}

impl Device for Switch {
    /// `swsetup.c`/`cswsetup.c`: `(p,p) (p,n) (n,p) (n,n)`.
    fn setup_matrix(
        &self,
        setup: &mut crate::devices::setup::MatrixSetup<'_>,
    ) -> SpiceResult<bool> {
        if let Some(series) = self.series() {
            let branch = setup.branch(0)?;
            setup.inductor(series.terminals[0], series.terminals[1], branch);
        }
        let [p, n] = self.conducting_terminals();
        setup.nodes(p, p);
        setup.nodes(p, n);
        setup.nodes(n, p);
        setup.nodes(n, n);
        if self.model.ltspice.is_some() {
            for row in [p, n] {
                for col in [self.terminals[2], self.terminals[3]] {
                    setup.nodes(row, col);
                }
            }
        }
        Ok(true)
    }
    /// Linear in `.disto`: C gives this device no distortion routine
    /// (`DEVdisto = NULL`, `sw/swinit.c`, `csw/cswinit.c`; the switch enters through its small-signal on/off conductance), so it enters only through its
    /// small-signal matrix.
    fn distortion(
        &self,
        _context: &crate::devices::distortion::DistortionContext<'_>,
    ) -> crate::primitives::SpiceResult<crate::devices::distortion::DeviceDistortion> {
        if self
            .model
            .ltspice
            .is_some_and(|law| !law.discrete() || law.limit.is_some())
        {
            return Err(SpiceError::Unsupported {
                feature: "LTspice nonlinear switch distortion".into(),
                location: None,
            });
        }
        Ok(crate::devices::distortion::DeviceDistortion::Linear)
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn designator(&self) -> char {
        self.kind.designator()
    }

    fn terminals(&self) -> &[NodeId] {
        &self.terminals
    }

    fn external_terminals(&self) -> &[NodeId] {
        let count = if self.kind == SwitchKind::Voltage {
            4
        } else {
            2
        };
        &self.terminals[..count]
    }

    fn branch_currents(&self) -> usize {
        usize::from(self.model.series_inductance != 0.)
    }

    fn expression_current_branch(&self) -> Option<usize> {
        (self.model.series_inductance != 0.).then_some(0)
    }

    fn truncation_slot(&self) -> Option<usize> {
        (self.model.series_inductance != 0.).then_some(crate::devices::rlc::QUANTITY)
    }

    fn storage_element(
        &self,
        _context: &ModelContext,
    ) -> Option<SpiceResult<crate::devices::traits::StorageElement>> {
        self.series().map(|series| {
            Ok(crate::devices::traits::StorageElement {
                kind: crate::devices::traits::StorageKind::Inductor,
                value: series.value,
                initial: None,
            })
        })
    }

    fn initial_voltage_relations(
        &self,
        inductors_short: bool,
    ) -> Vec<crate::devices::traits::InitialVoltageRelation> {
        if let Some(series) = self.series().filter(|_| inductors_short) {
            return vec![crate::devices::traits::InitialVoltageRelation {
                terminals: series.terminals,
                law: crate::devices::traits::InitialVoltageLaw::Fixed(0.),
            }];
        }
        Vec::new()
    }

    fn small_signal_uses_trial_state(&self) -> bool {
        self.model.ltspice.is_some() && self.discrete()
    }

    fn controlling_sources(&self) -> &[ControlReference] {
        &self.control
    }

    fn is_nonlinear(&self) -> bool {
        true
    }

    fn has_discrete_state(&self) -> bool {
        self.discrete()
    }

    fn state_count(&self) -> usize {
        self.storage_slots() + if self.discrete() { 2 } else { 0 }
    }

    fn stamp(&self, context: &mut StampContext<'_>) -> SpiceResult<()> {
        if context.mode.is_ac() {
            return Err(SpiceError::circuit(format!(
                "{}: switch AC needs small-signal assembly",
                self.name
            )));
        }
        let control = self.control_value(
            |node| context.node_voltage(node),
            |row| context.row_value(row),
            context.controls,
        )?;
        if let Some(law) = self.model.ltspice {
            if let Some(series) = self.series() {
                series.stamp_inductor(context)?;
            }
            let closed = law.discrete() && self.load_state(context, control)?.is_closed();
            let [p, n] = self.conducting_terminals();
            let voltage = context.node_voltage(p) - context.node_voltage(n);
            let point = self.ltspice_point(law, voltage, control, closed, context.gmin)?;
            return self.stamp_ltspice(context, voltage, control, point);
        }
        let state = self.load_state(context, control)?;
        let g = self.conductance(state.is_closed(), context.gmin);
        crate::devices::linear::nodal_stamp(
            context.matrix,
            context.unknowns,
            [self.terminals[0], self.terminals[1]],
            g,
        )
    }

    fn convergence_test(
        &self,
        context: &crate::devices::traits::ConvergenceContext<'_>,
    ) -> SpiceResult<bool> {
        let Some(law) = self.model.ltspice else {
            return Ok(false);
        };
        let (voltage, control) = self.ltspice_bias(|node| context.voltage(node));
        let closed = self.ltspice_closed(law, Some(context.states), control)?;
        let (prior_v, prior_c) = self.ltspice_bias(|node| {
            context
                .unknowns
                .node_row(node)
                .and_then(|row| context.previous.get(row))
                .unwrap_or(0.)
        });
        let prior_closed = self.ltspice_closed(law, Some(context.states), prior_c)?;
        let point =
            self.ltspice_point(law, voltage, control, closed, context.model_context.gmin)?;
        let prior = self.ltspice_point(
            law,
            prior_v,
            prior_c,
            prior_closed,
            context.model_context.gmin,
        )?;
        let predicted = prior.jacobian.control.mul_add(
            control - prior_c,
            prior
                .jacobian
                .voltage
                .mul_add(voltage - prior_v, prior.current),
        );
        let tol = context
            .reltol
            .mul_add(point.current.abs().max(predicted.abs()), context.abstol);
        Ok((predicted - point.current).abs() > tol)
    }

    fn assemble_nonlinear_taylor(
        &self,
        context: &mut crate::devices::physical_dae::TaylorContext<'_>,
    ) -> SpiceResult<()> {
        use crate::devices::physical_dae::TaylorControl;
        use crate::maths::taylor::differential::Differential;

        let Some(law) = self.model.ltspice.filter(|law| !law.discrete()) else {
            return Err(SpiceError::Unsupported {
                feature: format!("memoryless physical Taylor switch for {}", self.name),
                location: None,
            });
        };
        let node = |terminal| context.unknowns.node_row(terminal);
        let output = self.conducting_terminals().map(node);
        let input = match self.kind {
            SwitchKind::Voltage => [self.terminals[2], self.terminals[3]].map(node),
            SwitchKind::Current => {
                let row = context
                    .controls
                    .first()
                    .copied()
                    .ok_or_else(|| SpiceError::circuit("missing Taylor switch control row"))?;
                [Some(row), None]
            }
        };
        let controls = [
            TaylorControl {
                rows: output,
                polarity: 1.,
            },
            TaylorControl {
                rows: input,
                polarity: 1.,
            },
        ];
        let voltage = controls[0].input::<2>(0, context)?.value;
        let control = controls[1].input::<2>(1, context)?.value;
        let mut guards = context.output.guards.is_some().then(Vec::new);
        let point = law.taylor(
            self.ltspice_parameters(context.model.gmin),
            voltage,
            control,
            &mut guards,
        )?;
        let current = Differential {
            value: point.current,
            gradient: [Some(point.jacobian.voltage), Some(point.jacobian.control)],
        };
        context.local_law(output, &current, &controls, 1., None)?;
        if let (Some(trace), Some(guards)) = (&mut context.output.guards, guards) {
            trace.extend(context.device, guards)?;
        }
        if let Some(series) = self.series() {
            let branch = context
                .branch
                .ok_or_else(|| SpiceError::circuit("missing Taylor Lser branch"))?;
            let row = Some(branch);
            for (terminal, sign) in series.terminals.into_iter().zip([1., -1.]) {
                let terminal = context.unknowns.node_row(terminal);
                context.linear_static(terminal, row, sign)?;
                context.linear_static(row, terminal, sign)?;
            }
            context.linear_charge(row, row, -series.value)?;
        }
        Ok(())
    }

    fn assemble_nonlinear_charge(
        &self,
        context: &mut crate::devices::traits::ChargeContext<'_>,
    ) -> SpiceResult<()> {
        if self.model.ltspice.is_none() {
            return Err(SpiceError::Unsupported {
                feature: format!("nonlinear frequency-domain charge for {}", self.name),
                location: None,
            });
        }
        if let Some(series) = self.series() {
            let row = context
                .branch
                .ok_or_else(|| SpiceError::circuit("missing Lser branch"))?;
            let current = context
                .state
                .get(row)
                .ok_or_else(|| SpiceError::circuit("missing Lser current"))?;
            context.charge.add_to(row, -series.value * current)?;
            context.jacobian.add(row, row, -series.value)?;
        }
        Ok(())
    }

    /// `.sens`: C's switch model records ([`crate::devices::sensitivity`]).
    fn sensitivity(
        &self,
        context: &ModelContext,
    ) -> SpiceResult<Box<dyn crate::devices::sensitivity::DeviceSensitivity + '_>> {
        if self.model.ltspice.is_some() {
            return Err(SpiceError::Unsupported {
                feature: "LTspice switch model sensitivity".into(),
                location: None,
            });
        }
        Ok(Box::new(
            crate::devices::sensitivity::SwitchSensitivity::new(self, context),
        ))
    }

    fn assemble_small_signal(
        &self,
        context: &mut LinearContext<'_>,
        bias: &Vector,
    ) -> SpiceResult<()> {
        if let Some(law) = self.model.ltspice {
            if let Some(series) = self.series() {
                // Independently observed LTspice 24.0.12 AC ignores Lser,
                // although its transient equation includes the inductance.
                context.branch(series.terminals)?;
            }
            let (voltage, control) = self.ltspice_bias(|node| {
                context
                    .unknowns
                    .node_row(node)
                    .and_then(|row| bias.get(row))
                    .unwrap_or(0.)
            });
            let closed = self.ltspice_closed(law, context.states, control)?;
            let point =
                self.ltspice_point(law, voltage, control, closed, context.model_context.gmin)?;
            for (row, col, value) in self.ltspice_entries(point.small_signal) {
                if let (Some(row), Some(col)) = (
                    context.unknowns.node_row(row),
                    context.unknowns.node_row(col),
                ) {
                    context.system.a.add(row, col, value)?;
                }
            }
            return Ok(());
        }
        let closed = self.small_signal_closed(context.states, "small-signal", || {
            let unknowns = context.unknowns;
            self.control_value(
                |node| {
                    unknowns
                        .node_row(node)
                        .and_then(|row| bias.get(row))
                        .unwrap_or(0.)
                },
                |row| {
                    bias.get(row).ok_or_else(|| {
                        SpiceError::circuit("bias row out of range for a switch control")
                    })
                },
                context.controls,
            )
        })?;
        let g = self.conductance(closed, context.model_context.gmin);
        context.nodal([self.terminals[0], self.terminals[1]], g, false)
    }

    /// `swnoise.c`/`cswnoise.c`: one thermal generator (C's empty suffix, no
    /// separate total) of the on or off conductance at the circuit
    /// temperature, decided like the small-signal stamp: from the supplied
    /// small-signal state (any non-zero code is on, as `(int) CKTstate0` in
    /// C), or from the instance flag's `MODEINITJCT` state without one.
    fn noise(
        &self,
        context: &NoiseContext<'_>,
        sources: &mut Vec<NoiseSource>,
    ) -> SpiceResult<DeviceNoise<'_>> {
        let conductance = if let Some(law) = self.model.ltspice {
            let (voltage, control) = self.ltspice_bias(|node| context.voltage(node));
            let closed = self.ltspice_closed(law, context.states, control)?;
            self.ltspice_point(law, voltage, control, closed, context.model_context.gmin)?
                .small_signal
                .voltage
        } else {
            let closed = self.small_signal_closed(context.states, "noise", || {
                self.control_value(
                    |node| context.voltage(node),
                    |row| {
                        context.bias.get(row).ok_or_else(|| {
                            SpiceError::circuit("bias row out of range for a switch control")
                        })
                    },
                    context.controls,
                )
            })?;
            self.conductance(closed, context.model_context.gmin)
        };
        sources.extend([NoiseSource::new(
            "",
            self.conducting_terminals(),
            NoiseKind::Thermal {
                conductance,
                temperature: context.circuit_kelvin(),
            },
        )]);
        Ok(DeviceNoise::Sources {
            family: match self.kind {
                SwitchKind::Voltage => NoiseFamily::VoltageSwitch,
                SwitchKind::Current => NoiseFamily::CurrentSwitch,
            },
            model: Some(&self.model_name),
            total: false,
        })
    }

    fn timestep_limit(&self, context: &TruncationContext<'_>) -> SpiceResult<Option<Real>> {
        if !self.discrete() || self.kind == SwitchKind::Current {
            // cswtrunc.c reads a control slot CSWload never writes.
            return Ok(None);
        }
        let (Some(code), Some(control), Some(previous)) = (
            context.trial.get(self.switch_slot(STATE)).copied(),
            context.trial.get(self.switch_slot(CONTROL)).copied(),
            context
                .accepted
                .and_then(|a| a.get(self.switch_slot(CONTROL)))
                .copied(),
        ) else {
            return Ok(None);
        };
        let change = control - previous;
        let SwitchModel {
            threshold: t,
            hysteresis: h,
            ..
        } = self.model;
        // C tests `CKTstate0[SWswitchstate] == 0` (really off) only.
        let state = self.decode(code, "truncation")?;
        let max_change = if state == SwitchState::ReallyOff {
            let reference = t + h;
            if !(control < reference && change > 0.) {
                return Ok(None);
            }
            let scaled_distance = (reference - control) * 0.75;
            scaled_distance + 0.05
        } else {
            let reference = t - h;
            if !(control > reference && change < 0.) {
                return Ok(None);
            }
            let scaled_distance = (reference - control) * 0.75;
            scaled_distance - 0.05
        };
        Ok(Some(max_change / change * context.dt))
    }

    /// Pole-zero load: C `swpzload.c`/`cswpzld.c` (the `MODEINITSMSIG` state, as AC) equals the AC load with `s` for `j omega`.
    fn assemble_pole_zero(
        &self,
        context: &mut crate::devices::linear::LinearContext<'_>,
        bias: &crate::maths::Vector,
    ) -> crate::primitives::SpiceResult<()> {
        self.assemble_small_signal(context, bias)
    }
}

#[cfg(test)]
mod tests {
    use super::{Switch, SwitchKind, SwitchModel, SwitchState};
    use crate::devices::state::IterationPhase;
    use crate::primitives::NodeId;

    fn switch(kind: SwitchKind, threshold: f64, hysteresis: f64, on: bool) -> Switch {
        Switch {
            name: "s1".into(),
            model_name: "sw".into(),
            kind,
            terminals: vec![NodeId::GROUND; 4],
            control: Vec::new(),
            model: SwitchModel {
                threshold,
                hysteresis,
                on_conductance: 1.,
                off_conductance: None,
                ltspice: None,
                series_inductance: 0.,
            },
            initially_on: on,
            written: crate::devices::sensitivity::SwitchValues::default(),
        }
    }

    use SwitchState::{HysteresisOff, HysteresisOn, ReallyOff, ReallyOn};

    #[test]
    fn codes_round_trip() {
        for state in [ReallyOff, ReallyOn, HysteresisOff, HysteresisOn] {
            assert_eq!(SwitchState::from_code(state.code()), Some(state));
        }
        assert_eq!(SwitchState::from_code(-1.), None);
        assert!(ReallyOn.is_closed() && HysteresisOn.is_closed());
        assert!(!ReallyOff.is_closed() && !HysteresisOff.is_closed());
    }

    #[test]
    fn every_switch_phase_respects_the_signed_hysteresis_boundaries() {
        let states = [ReallyOff, ReallyOn, HysteresisOff, HysteresisOn];
        let predict_voltage = [ReallyOn, ReallyOff, HysteresisOff, HysteresisOn];
        let enter = [HysteresisOn, HysteresisOff, HysteresisOff, HysteresisOn];
        for kind in [SwitchKind::Voltage, SwitchKind::Current] {
            for h in [-0.5_f64, -0., 0., 0.5] {
                let t = 1.;
                let low = t - h.abs();
                let high = t + h.abs();
                let controls = [low.next_down(), low, t, high, high.next_up()];
                for initially_on in [false, true] {
                    let switch = switch(kind, t, h, initially_on);
                    for (i, accepted) in states.into_iter().enumerate() {
                        for iterate in states {
                            for (j, control) in controls.into_iter().enumerate() {
                                let flag = if initially_on {
                                    if j == 4 { ReallyOn } else { HysteresisOn }
                                } else if j == 0 {
                                    ReallyOff
                                } else {
                                    HysteresisOff
                                };
                                let predict = if h > 0. {
                                    accepted
                                } else if kind == SwitchKind::Voltage {
                                    predict_voltage[i]
                                } else {
                                    enter[i]
                                };
                                let float = if h <= 0. {
                                    enter[i]
                                } else if kind == SwitchKind::Voltage {
                                    iterate
                                } else {
                                    accepted
                                };
                                let outside = |inside| match j {
                                    0 => ReallyOff,
                                    4 => ReallyOn,
                                    _ => inside,
                                };
                                for (phase, expected) in [
                                    (IterationPhase::Junction, flag),
                                    (IterationPhase::Fix, flag),
                                    (IterationPhase::Predict, outside(predict)),
                                    (IterationPhase::Float, outside(float)),
                                ] {
                                    assert_eq!(
                                        switch
                                            .decide(phase, control, accepted, Some(iterate))
                                            .unwrap(),
                                        expected,
                                        "{kind:?}, {h}, {control}, {phase:?}, {accepted:?}, {iterate:?}"
                                    );
                                }
                            }
                        }
                    }
                    assert!(
                        switch
                            .decide(IterationPhase::Float, high.next_up(), ReallyOff, None)
                            .is_err()
                    );
                }
            }
        }
    }

    #[test]
    fn small_signal_state_and_fallback_use_distinct_conductance_rules() {
        for kind in [SwitchKind::Voltage, SwitchKind::Current] {
            let mut s = switch(kind, 1., 0.5, false);
            for state in [ReallyOff, ReallyOn, HysteresisOff, HysteresisOn] {
                let closed = s
                    .small_signal_closed(Some(&[state.code()]), "test", || {
                        panic!("saved state needs no control")
                    })
                    .unwrap();
                assert_eq!(closed, state != ReallyOff);
                assert_eq!(s.conductance(closed, 1e-9), if closed { 1. } else { 1e-9 });
            }
            assert!(!s.small_signal_closed(None, "test", || Ok(1.)).unwrap());
            assert!(s.small_signal_closed(Some(&[]), "test", || Ok(1.)).is_err());
            assert!(
                s.small_signal_closed(Some(&[4.]), "test", || Ok(1.))
                    .is_err()
            );
            s.model.off_conductance = Some(1e-6);
            assert_eq!(s.conductance(false, 1e-9), 1e-6);
        }
    }

    #[test]
    fn truncation_decodes_the_state_before_selecting_a_threshold() {
        use crate::devices::{Device, TruncationContext};
        let s = switch(SwitchKind::Voltage, 1., 0.5, false);
        let context = |trial| TruncationContext {
            trial,
            accepted: Some(&[0., 0.]),
            dt: 1.,
        };
        assert!(s.timestep_limit(&context(&[4., 1.])).is_err());
        assert!(s.timestep_limit(&context(&[f64::NAN, 1.])).is_err());
        assert!(s.timestep_limit(&context(&[0., 1.])).unwrap().is_some());
        assert!(s.timestep_limit(&context(&[2., 1.])).unwrap().is_none());
        let w = switch(SwitchKind::Current, 1., 0.5, false);
        assert!(w.timestep_limit(&context(&[0., 1.])).unwrap().is_none());
    }

    #[test]
    fn initial_flags_follow_swload_initjct() {
        let off = switch(SwitchKind::Voltage, 1., 0.5, false);
        assert_eq!(off.flag_state(0.4), ReallyOff);
        assert_eq!(off.flag_state(0.5), HysteresisOff);
        assert_eq!(off.flag_state(5.), HysteresisOff);
        let on = switch(SwitchKind::Voltage, 1., 0.5, true);
        assert_eq!(on.flag_state(1.6), ReallyOn);
        assert_eq!(on.flag_state(1.5), HysteresisOn);
        assert_eq!(on.flag_state(-5.), HysteresisOn);
        // Negative hysteresis uses |VH|.
        let negative = switch(SwitchKind::Voltage, 1., -0.5, true);
        assert_eq!(negative.flag_state(1.6), ReallyOn);
        assert_eq!(negative.flag_state(1.4), HysteresisOn);
    }

    #[test]
    fn positive_hysteresis_band_keeps_the_previous_state() {
        let s = switch(SwitchKind::Voltage, 1., 0.5, false);
        let decide = |phase, control, accepted, iterate| {
            s.decide(phase, control, accepted, iterate).unwrap()
        };
        // Rising: off until VT + VH is exceeded.
        assert_eq!(
            decide(IterationPhase::Predict, 1.4, ReallyOff, None),
            ReallyOff
        );
        assert_eq!(
            decide(IterationPhase::Predict, 1.51, ReallyOff, None),
            ReallyOn
        );
        // Falling: on until below VT - VH.
        assert_eq!(
            decide(IterationPhase::Predict, 0.6, ReallyOn, None),
            ReallyOn
        );
        assert_eq!(
            decide(IterationPhase::Predict, 0.49, ReallyOn, None),
            ReallyOff
        );
        // S float keeps the previous iterate inside the band, W the accepted state.
        assert_eq!(
            decide(IterationPhase::Float, 1., ReallyOff, Some(ReallyOn)),
            ReallyOn
        );
        let w = switch(SwitchKind::Current, 1., 0.5, false);
        assert_eq!(
            w.decide(IterationPhase::Float, 1., ReallyOff, Some(ReallyOn))
                .unwrap(),
            ReallyOff
        );
        assert!(
            s.decide(IterationPhase::Float, 1., ReallyOff, None)
                .is_err()
        );
    }

    #[test]
    fn zero_hysteresis_switches_at_the_threshold() {
        let s = switch(SwitchKind::Voltage, 2., 0., false);
        for phase in [IterationPhase::Predict, IterationPhase::Float] {
            assert_eq!(
                s.decide(phase, 2.001, ReallyOff, Some(ReallyOff)).unwrap(),
                ReallyOn
            );
            assert_eq!(
                s.decide(phase, 1.999, ReallyOn, Some(ReallyOn)).unwrap(),
                ReallyOff
            );
        }
        // Exactly at the threshold the C band rules map the accepted state.
        assert_eq!(
            s.decide(IterationPhase::Predict, 2., ReallyOn, None)
                .unwrap(),
            ReallyOff
        );
        assert_eq!(
            s.decide(IterationPhase::Float, 2., ReallyOn, Some(ReallyOn))
                .unwrap(),
            HysteresisOff
        );
        let w = switch(SwitchKind::Current, 2., 0., false);
        assert_eq!(
            w.decide(IterationPhase::Predict, 2., ReallyOn, None)
                .unwrap(),
            HysteresisOff
        );
        assert_eq!(
            w.decide(IterationPhase::Predict, 2., ReallyOff, None)
                .unwrap(),
            HysteresisOn
        );
        assert_eq!(
            w.decide(IterationPhase::Predict, 2., HysteresisOn, None)
                .unwrap(),
            HysteresisOn
        );
    }

    #[test]
    fn negative_hysteresis_band_maps_the_accepted_state() {
        let s = switch(SwitchKind::Voltage, 1., -0.5, false);
        // Band is [0.5, 1.5]; outside it the switch follows the control.
        assert_eq!(
            s.decide(IterationPhase::Predict, 1.6, ReallyOff, None)
                .unwrap(),
            ReallyOn
        );
        assert_eq!(
            s.decide(IterationPhase::Predict, 0.4, ReallyOn, None)
                .unwrap(),
            ReallyOff
        );
        assert_eq!(
            s.decide(IterationPhase::Predict, 1., ReallyOff, None)
                .unwrap(),
            ReallyOn
        );
        assert_eq!(
            s.decide(IterationPhase::Float, 1., ReallyOff, Some(ReallyOn))
                .unwrap(),
            HysteresisOn
        );
        assert_eq!(
            s.decide(IterationPhase::Float, 1., HysteresisOff, Some(ReallyOn))
                .unwrap(),
            HysteresisOff
        );
    }
}
