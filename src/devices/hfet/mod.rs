//! The macspice MESA/HFET family, `Z` levels 2–6.
//!
//! `mesa/`, `hfet1/` and `hfet2/` are three copies of one lineage: the same
//! `SPICEdev` shape, the same conditional internal nodes and the same
//! temperature/truncation/ask/pole-zero split. This module owns what they
//! share; one file per level holds its channel.
//!
//! C references, read as behaviour only: `parser/inp2z.c` and
//! `parser/inpdomod.c` for the card and the level map (0 or 1 `mes/`, 2–4
//! `mesa/`, 5 `hfet1/`, 6 `hfet2/`); `mesa/mesa.c` for `MESApTable` and
//! `MESAmPTable`; `mesa/mesaparam.c` and `mesamparam.c` for the setters;
//! `mesa/mesasetup.c` for the defaults and the internal nodes; and
//! `mesa/mesatemp.c` for the derived quantities.
//!
//! # C quirks reproduced deliberately
//!
//! - **MESA is NMF only.** `MESAsetup` prints "Only nmf model type supported,
//!   set to nmf" and forces the type for a `pmf` model, which then simulates
//!   as an N-channel device (checked against the reference binary). The port
//!   does not carry a polarity factor for this family at all, as C does not.
//! - **`td`/`ts` are inconsistent.** An explicit `td`/`ts` is Celsius and C
//!   stores `value + CONSTCtoK`; the *default* is `CKTtemp + dtemp`, i.e. a
//!   Celsius value used directly as kelvin by `mesatemp.c`. Both are
//!   reproduced, because the defaults feed `csatfs`, `vcrits` and every
//!   temperature coefficient.
//! - **`tf` defaults to `CKTtemp`** in kelvin, and `mesaacl.c` blends
//!   `tlambda` towards `tLambdahf` with `tanh((f - fl)/delf)`; with the
//!   default `delfo` of zero that blend is inactive.
//!
//! # Not ported
//!
//! `.op`, `.dc`, `.ac` and the analyses built on the small-signal assembly
//! (`.sp`, `.tf`) work: `mesaacl.c`'s conductances and gate capacitances are
//! in place and C-verified by the `m10_mesa_ac` fixture. `mesapzl.c`,
//! `mesatrunc.c` and the charge storage of `mesaload.c` belong to H1b, as do
//! the `ri`/`rf`/`rdi`/`rsi` resistance paths (issues #155-#157); each
//! refuses with its C reference rather than dropping terms.

pub(crate) mod hfet1;
pub(crate) mod hfet2;
mod lifecycle;
pub(crate) mod mesa;

use std::cell::Cell;

use crate::devices::initial::{InstanceInitial, split};
use crate::devices::jfet::{self, CHARGE, p};
use crate::devices::schema::{
    ScalarDomain as D, ScalarParameter as P, ScalarSchema, ScalarUnit as U,
};
use crate::devices::sensitivity::{ParameterScope, RecordPair, SensitivityRecord};
use crate::devices::{LinearContext, ModelFamily, ResolvedModel};
use crate::netlist::ast::DeviceInstance;
use crate::primitives::{NodeAppend, NodeId, NodeKind, Real, SpiceError, SpiceResult};

/// Carrier density and its voltage derivative for the common HFET/MESA
/// soft threshold. Evaluate log(1 + exp(x)/2) without losing weak-inversion
/// charge when exp(x)/2 is smaller than one ULP of 1.
fn soft_density(n0: Real, voltage: Real, thermal: Real) -> (Real, Real) {
    let x = voltage / thermal;
    let (logarithm, fraction) = if x <= 1. {
        let b = 0.5 * x.exp();
        (b.ln_1p(), b / (1. + b))
    } else {
        let reciprocal = 2. * (-x).exp();
        (
            (x - std::f64::consts::LN_2) + reciprocal.ln_1p(),
            1. / (1. + reciprocal),
        )
    };
    let density = 2. * n0 * logarithm;
    let slope = 2. * n0 / thermal * fraction;
    (density, slope)
}

/// Combine the two carrier populations in series, with derivative of
/// the second population equal to its density divided by thermal voltage.
/// Ratios avoid forming the product or squared sum of two large densities.
fn series_density(first: Real, slope: Real, second: Real, thermal: Real) -> (Real, Real) {
    let (density, first_fraction, second_fraction) = if first.abs() <= second.abs() {
        let ratio = first / second;
        let fraction = 1. / (1. + ratio);
        (first * fraction, ratio * fraction, fraction)
    } else {
        let ratio = second / first;
        let fraction = 1. / (1. + ratio);
        (second * fraction, fraction, ratio * fraction)
    };
    let derivative =
        (second_fraction * second_fraction).mul_add(slope, density / thermal * first_fraction);
    (density, derivative)
}

/// Staged node setup common to the three MESA/HFET levels.
/// The caller commits the transaction only after temperature derivation succeeds.
struct NodeSetup<'a, 'table> {
    staged: &'a mut NodeAppend<'table>,
    terminals: Vec<NodeId>,
    name: &'a str,
    family: &'static str,
}

impl<'a, 'table> NodeSetup<'a, 'table> {
    fn new(
        nodes: &'a mut NodeAppend<'table>,
        instance: &'a DeviceInstance,
        family: &'static str,
    ) -> SpiceResult<(Self, [NodeId; 3])> {
        let external = nodes.intern_array(&instance.nodes)?;
        Ok((
            Self {
                staged: nodes,
                terminals: external.to_vec(),
                name: &instance.name,
                family,
            },
            external,
        ))
    }

    /// C creates a prime only for a nonzero resistance; otherwise it aliases the parent.
    fn prime(&mut self, resistance: Real, suffix: &str, parent: NodeId) -> SpiceResult<NodeId> {
        if resistance == 0. {
            return Ok(parent);
        }
        let name = format!("{}#{suffix}", self.name);
        if self.staged.get(&name).is_some() {
            return Err(SpiceError::circuit(format!(
                "{}: {} internal-node name collision: {name}",
                self.name, self.family
            )));
        }
        let internal = self.staged.intern(&name)?;
        self.staged.set_kind(internal, NodeKind::Internal)?;
        self.terminals.push(internal);
        Ok(internal)
    }

    fn finish(self) -> Vec<NodeId> {
        self.terminals
    }
}

/// Last successful temperature derivation for an immutable model/instance pair.
/// Sweep and sensitivity replacements construct an empty cache. The key keeps
/// the exact context bits (including signed zero) and never retains errors.
#[derive(Debug, Clone)]
struct TemperatureCache<T: Copy, const N: usize> {
    value: Cell<Option<([u64; N], T)>>,
}

impl<T: Copy, const N: usize> TemperatureCache<T, N> {
    const fn new() -> Self {
        Self {
            value: Cell::new(None),
        }
    }

    fn get_or_compute(
        &self,
        temperatures: [Real; N],
        compute: impl FnOnce() -> SpiceResult<T>,
    ) -> SpiceResult<T> {
        let key = temperatures.map(Real::to_bits);
        if let Some((previous, value)) = self.value.get()
            && previous == key
        {
            return Ok(value);
        }
        let value = compute()?;
        self.value.set(Some((key, value)));
        Ok(value)
    }
}

/// C `CONSTCtoK`.
pub(super) const CELSIUS_TO_KELVIN: Real = 273.15;

/// Match the deck schema's temperature domain for explicit sweep inputs.
fn physical_temperature(celsius: Real) -> bool {
    let kelvin = celsius + CELSIUS_TO_KELVIN;
    kelvin.is_finite() && kelvin > 0.
}

/// Common HFET1/HFET2 scalar instance setters.
pub(super) const HFET_INSTANCE: &[P] = &[
    p("l", U::Metre, D::Positive, Some(1e-6)),
    p("w", U::Metre, D::Positive, Some(20e-6)),
    p("m", U::Dimensionless, D::Positive, Some(1.)),
    p("temp", U::Celsius, D::Temperature, None),
    p("dtemp", U::Dimensionless, D::Finite, Some(0.)),
];

/// The `IC` vector components (`hfetparam.c`): `ICVDS`, then `ICVGS`.
const HFET_IC_COMPONENTS: [&str; 2] = ["icvds", "icvgs"];

/// Validated HFET1/HFET2 instance setters (not MESA).
#[derive(Debug, Clone, Copy)]
pub(super) struct HfetInstance {
    pub(super) l: Real,
    pub(super) w: Real,
    pub(super) m: Real,
    /// Celsius as written (`hfetparam.c` adds 273.15); `None` is
    /// `CKTtemp + dtemp`, used as kelvin.
    pub(super) temp: Option<Real>,
    pub(super) dtemp: Real,
}

impl HfetInstance {
    pub(super) fn resolve(
        i: &DeviceInstance,
    ) -> SpiceResult<(
        InstanceInitial,
        Vec<crate::netlist::ast::ParameterAssignment>,
    )> {
        split(&i.parameters, &HFET_IC_COMPONENTS)
    }

    pub(super) fn validate(
        defaults: &[&crate::netlist::ast::ParameterAssignment],
        setters: &[crate::netlist::ast::ParameterAssignment],
        owner: &crate::primitives::SourceLoc,
    ) -> SpiceResult<Self> {
        let values = crate::devices::schema::with_defaults(
            &HFET_INSTANCE_SCHEMA,
            defaults,
            setters.iter(),
            owner,
        )?;
        Ok(Self {
            l: jfet::required(&values, "l")?,
            w: jfet::required(&values, "w")?,
            m: jfet::required(&values, "m")?,
            temp: values.get("temp").map(|v| v.value),
            dtemp: jfet::required(&values, "dtemp")?,
        })
    }

    /// The shared schema defines which real instance setters can be swept.
    fn sweep_parameter(keyword: &str) -> Option<&'static str> {
        HFET_INSTANCE
            .iter()
            .find(|field| field.name.eq_ignore_ascii_case(keyword))
            .map(|field| field.name)
    }

    fn swept(mut self, owner: &str, parameter: &str, value: Real) -> SpiceResult<Self> {
        let check =
            |ok, what| crate::devices::sweep::check_swept(owner, parameter, value, ok, what);
        match Self::sweep_parameter(parameter) {
            Some(field @ ("l" | "w" | "m")) => {
                check(value > 0., "finite and positive")?;
                match field {
                    "l" => self.l = value,
                    "w" => self.w = value,
                    _ => self.m = value,
                }
            }
            Some("temp") => {
                check(physical_temperature(value), "above absolute zero")?;
                self.temp = Some(value);
            }
            Some("dtemp") => {
                check(true, "finite")?;
                self.dtemp = value;
            }
            _ => {
                return Err(SpiceError::circuit(format!(
                    "{owner}: HFET instance parameter {parameter} cannot be swept"
                )));
            }
        }
        Ok(self)
    }

    fn sensitivity_record(
        &self,
        family: crate::devices::sensitivity::SensitivityFamily,
        initial: &InstanceInitial,
        circuit_kelvin: Real,
    ) -> SpiceResult<SensitivityRecord> {
        SensitivityRecord::new(family, ParameterScope::Instance)
            .with("l", self.l, true)?
            .with("w", self.w, true)?
            .with("m", self.m, true)?
            .with(
                "icvds",
                initial.values[0].unwrap_or(0.),
                initial.values[0].is_some(),
            )?
            .with(
                "icvgs",
                initial.values[1].unwrap_or(0.),
                initial.values[1].is_some(),
            )?
            .with(
                "temp",
                self.temperature_at_kelvin(circuit_kelvin),
                self.temp.is_some(),
            )?
            .with("dtemp", self.dtemp, true)
    }

    fn from_record(record: &SensitivityRecord) -> SpiceResult<Self> {
        Ok(Self {
            l: record.value("l")?,
            w: record.value("w")?,
            m: record.value("m")?,
            temp: given_celsius(record, "temp")?,
            dtemp: record.value("dtemp")?,
        })
    }

    /// C instance temperature in kelvin: an explicit TEMP overrides DTEMP.
    pub(super) fn temperature(&self, circuit_celsius: Real) -> Real {
        self.temperature_at_kelvin(circuit_celsius + CELSIUS_TO_KELVIN)
    }

    pub(super) fn temperature_at_kelvin(&self, circuit_kelvin: Real) -> Real {
        self.temp
            .map_or(circuit_kelvin + self.dtemp, |v| v + CELSIUS_TO_KELVIN)
    }
}

/// A temperature field is stored in Kelvin, but exposed in Celsius only when given.
fn given_celsius(record: &SensitivityRecord, keyword: &str) -> SpiceResult<Option<Real>> {
    Ok(record
        .given(keyword)
        .then(|| record.value(keyword))
        .transpose()?
        .map(|kelvin| kelvin - CELSIUS_TO_KELVIN))
}

/// Both single-temperature HFET families expose TEMP in Celsius.
fn ask_hfet(scope: ParameterScope, keyword: &str, records: RecordPair<'_>) -> SpiceResult<Real> {
    match scope {
        ParameterScope::Model => records.model.value(keyword),
        ParameterScope::Instance if keyword == "temp" => {
            Ok(records.instance.value(keyword)? - CELSIUS_TO_KELVIN)
        }
        ParameterScope::Instance => records.instance.value(keyword),
    }
}

/// The setter reapplies the Celsius offset even on the sensitivity restore.
fn set_hfet(
    scope: ParameterScope,
    keyword: &str,
    value: Real,
    model: &mut SensitivityRecord,
    instance: &mut SensitivityRecord,
) -> SpiceResult<()> {
    match scope {
        ParameterScope::Model => model.set(keyword, value),
        ParameterScope::Instance if keyword == "temp" => {
            instance.set(keyword, value + CELSIUS_TO_KELVIN)
        }
        ParameterScope::Instance => instance.set(keyword, value),
    }
}

type NodeStamp = (NodeId, NodeId, Real);

/// Assemble borrowed reset stamps in their original real-then-imaginary order.
/// Validate all addends before touching the matrix, including ground entries,
/// just as `ResetStamps::new` does. No temporary owned device or stamp vector.
fn stamp_reset(
    name: &str,
    context: &mut LinearContext<'_>,
    real: &[NodeStamp],
    imaginary: &[NodeStamp],
) -> SpiceResult<()> {
    if let Some((_, _, value)) = real
        .iter()
        .chain(imaginary)
        .find(|(_, _, value)| !value.is_finite())
    {
        return Err(SpiceError::Numerical {
            context: format!("AC sensitivity of {name}"),
            message: format!("the reset load gives a nonfinite matrix element {value}"),
        });
    }
    for &(a, b, value) in real {
        if let (Some(row), Some(col)) = (context.unknowns.node_row(a), context.unknowns.node_row(b))
        {
            context.system.a.add(row, col, value)?;
        }
    }
    for &(a, b, value) in imaginary {
        if let (Some(row), Some(col)) = (context.unknowns.node_row(a), context.unknowns.node_row(b))
        {
            context.dynamic_prescaled(row, col, value)?;
        }
    }
    Ok(())
}

/// Model setters of `MESAmPTable` with `mesasetup.c`'s defaults. `vto` is the
/// cardinal spelling and `vt0` its `IOPR` alias; `phib1`/`tphib`,
/// `xchi`/`tggr` and `tvto`/`alphat` are alias pairs of one C parameter, so
/// their defaults are absent here and resolved by [`Model::resolve`].
const MODEL: &[P] = &[
    p("vto", U::Volt, D::Finite, None),
    p("vt0", U::Volt, D::Finite, None),
    p("lambda", U::InverseVolt, D::NonNegative, Some(0.045)),
    p("lambdahf", U::InverseVolt, D::NonNegative, None),
    p("beta", U::AmperePerVoltSquared, D::Finite, Some(8.5e-3)),
    p("vs", U::MetrePerSecond, D::Positive, Some(1.5e5)),
    p("rd", U::Ohm, D::NonNegative, Some(0.)),
    p("rs", U::Ohm, D::NonNegative, Some(0.)),
    p("rg", U::Ohm, D::NonNegative, Some(0.)),
    p("ri", U::Ohm, D::NonNegative, Some(0.)),
    p("rf", U::Ohm, D::NonNegative, Some(0.)),
    p("rdi", U::Ohm, D::NonNegative, Some(0.)),
    p("rsi", U::Ohm, D::NonNegative, Some(0.)),
    // `mesamparam.c` stores PHIB/PHIB1 as `value * CHARGE`: the card is in
    // electron-volts, the model field in joules, like `HFETAphib`.
    p("phib", U::Volt, D::Finite, Some(0.5)),
    p("phib1", U::VoltPerKelvin, D::Finite, Some(0.)),
    p("tphib", U::VoltPerKelvin, D::Finite, None),
    p("astar", U::Dimensionless, D::NonNegative, Some(4.0e4)),
    p("ggr", U::SiemensPerMetre, D::NonNegative, Some(40.)),
    p("del", U::Dimensionless, D::NonNegative, Some(0.04)),
    p("xchi", U::InverseKelvin, D::Finite, None),
    p("tggr", U::InverseKelvin, D::Finite, None),
    p("n", U::Dimensionless, D::Finite, Some(1.)),
    p("eta", U::Dimensionless, D::Finite, Some(1.73)),
    p("m", U::Dimensionless, D::Finite, Some(2.5)),
    p("mc", U::Dimensionless, D::Finite, Some(3.0)),
    p("alpha", U::Dimensionless, D::Finite, Some(0.)),
    p("sigma0", U::Dimensionless, D::Finite, Some(0.081)),
    p("vsigmat", U::Volt, D::Finite, Some(1.01)),
    p("vsigma", U::Volt, D::Positive, Some(0.1)),
    p(
        "mu",
        U::SquareCentimetrePerVoltSecond,
        D::NonNegative,
        Some(0.23),
    ),
    p("theta", U::Dimensionless, D::Finite, Some(0.)),
    p(
        "mu1",
        U::SquareCentimetrePerVoltSecond,
        D::NonNegative,
        Some(0.),
    ),
    p(
        "mu2",
        U::SquareCentimetrePerVoltSecond,
        D::NonNegative,
        Some(0.),
    ),
    p("d", U::Metre, D::Positive, Some(0.12e-6)),
    p("nd", U::PerCubicCentimetre, D::Positive, Some(2.0e23)),
    p("du", U::Metre, D::Positive, Some(0.035e-6)),
    p("ndu", U::PerCubicCentimetre, D::Positive, Some(1e22)),
    p("th", U::Metre, D::NonNegative, Some(0.01e-6)),
    p("ndelta", U::PerCubicCentimetre, D::NonNegative, Some(6e24)),
    p("delta", U::Dimensionless, D::Finite, Some(5.0)),
    p("tc", U::InverseVolt, D::Finite, Some(0.)),
    p("tvto", U::VoltPerKelvin, D::Finite, None),
    p("alphat", U::VoltPerKelvin, D::Finite, None),
    p("tlambda", U::Kelvin, D::Finite, Some(Real::MAX)),
    p("teta0", U::Kelvin, D::Finite, Some(Real::MAX)),
    p("teta1", U::Kelvin, D::Finite, Some(0.)),
    p("tmu", U::Kelvin, D::Positive, Some(300.15)),
    p("xtm0", U::Dimensionless, D::Finite, Some(0.)),
    p("xtm1", U::Dimensionless, D::Finite, Some(0.)),
    p("xtm2", U::Dimensionless, D::Finite, Some(0.)),
    p("ks", U::Dimensionless, D::Finite, Some(0.)),
    p("vsg", U::Volt, D::Finite, Some(0.)),
    p("tf", U::Kelvin, D::NonZero, None),
    p("flo", U::Dimensionless, D::Finite, Some(0.)),
    p("delfo", U::Dimensionless, D::Finite, Some(0.)),
    p("ag", U::Dimensionless, D::Finite, Some(0.)),
    p("rtc1", U::InverseKelvin, D::Finite, Some(0.)),
    p("rtc2", U::InverseKelvinSquared, D::Finite, Some(0.)),
    p("zeta", U::Dimensionless, D::Finite, Some(1.)),
    p("level", U::Dimensionless, D::Finite, Some(2.)),
    p("nmax", U::Dimensionless, D::NonNegative, Some(2e16)),
    p("gamma", U::Dimensionless, D::Finite, Some(3.0)),
    p(
        "epsi",
        U::FaradPerMetre,
        D::Positive,
        Some(12.244 * 8.854_18e-12),
    ),
    p("cas", U::Dimensionless, D::Finite, Some(1.)),
    p("cbs", U::Dimensionless, D::Finite, Some(1.)),
];

/// Scalar instance setters of `MESApTable` (`mesasetup.c` and
/// `mesaparam.c`); `off` and `icvds`/`icvgs` are split off first. MESA has no
/// `area`.
const INSTANCE: &[P] = &[
    p("l", U::Metre, D::Positive, Some(1e-6)),
    p("w", U::Metre, D::Positive, Some(20e-6)),
    p("m", U::Dimensionless, D::Positive, Some(1.)),
    p("td", U::Celsius, D::Temperature, None),
    p("ts", U::Celsius, D::Temperature, None),
    p("dtemp", U::Dimensionless, D::Finite, Some(0.)),
];

/// The `IC` vector components (`mesaparam.c`): `ICVDS`, then `ICVGS`.
pub(super) const IC_COMPONENTS: [&str; 2] = ["icvds", "icvgs"];

/// Validated model card of a level-2 MESA device.
///
/// Only the setters `mesa1` and its temperature derivations read are stored.
/// `mesasetup.c`/`mesamparam.c` still accept the rest — they belong to the
/// level-3/4 channels (`mesa2`, `mesa3`) — so the schema validates them and C
/// ignores them at level 2 exactly as this port does: `beta`, `ndu`,
/// `ndelta`, `nmax`, `gamma`, `epsi`, `cas` and `cbs`. The [H1
/// reference notes](https://github.com/michaelnavazhylau/ngspice-rs/blob/main/docs/port/M10_SPILL_PLAN.md)
/// say which channel reads which.
#[derive(Debug, Clone, Copy)]
pub(super) struct Model {
    pub(super) threshold: Real,
    pub(super) lambda: Real,
    /// `mesatemp.c` uses the model's `lambda` when `lambdahf` is unset, and
    /// `mesaacl.c` blends the two through the `fl`/`delf` window.
    pub(super) lambdahf: Option<Real>,
    pub(super) vs: Real,
    pub(super) rd: Real,
    pub(super) rs: Real,
    pub(super) rg: Real,
    pub(super) ri: Real,
    pub(super) rf: Real,
    pub(super) rdi: Real,
    pub(super) rsi: Real,
    pub(super) phib: Real,
    pub(super) phib1: Real,
    pub(super) astar: Real,
    pub(super) ggr: Real,
    pub(super) del: Real,
    pub(super) xchi: Real,
    pub(super) n: Real,
    pub(super) eta: Real,
    pub(super) m: Real,
    pub(super) mc: Real,
    pub(super) alpha: Real,
    pub(super) sigma0: Real,
    pub(super) vsigmat: Real,
    pub(super) vsigma: Real,
    pub(super) mu: Real,
    pub(super) theta: Real,
    pub(super) mu1: Real,
    pub(super) mu2: Real,
    pub(super) d: Real,
    pub(super) nd: Real,
    /// Level-3/4 setters: `mesa2` reads `ndu`/`ndelta`/`du`/`th`/`beta`,
    /// `mesa3` reads `nmax`/`gamma`/`cas`/`cbs`/`epsi`, and levels 3-4 both
    /// read `du` through `mesatemp.c`'s `vpou`/`vpod`/`n0`.
    pub(super) beta: Real,
    pub(super) ndu: Real,
    pub(super) ndelta: Real,
    pub(super) nmax: Real,
    pub(super) gamma: Real,
    pub(super) epsi: Real,
    pub(super) cas: Real,
    pub(super) cbs: Real,
    pub(super) du: Real,
    pub(super) th: Real,
    pub(super) delta: Real,
    pub(super) tc: Real,
    pub(super) tvto: Real,
    pub(super) tlambda: Real,
    pub(super) teta0: Real,
    pub(super) teta1: Real,
    pub(super) tmu: Real,
    pub(super) xtm0: Real,
    pub(super) xtm1: Real,
    pub(super) xtm2: Real,
    pub(super) ks: Real,
    pub(super) vsg: Real,
    /// `mesasetup.c`'s `CKTtemp` default (kelvin) and the `mesaacl.c`
    /// frequency blend of `tlambda` with `tLambdahf`.
    pub(super) tf: Option<Real>,
    pub(super) flo: Real,
    pub(super) delfo: Real,
    pub(super) ag: Real,
    pub(super) tc1: Real,
    pub(super) tc2: Real,
    pub(super) zeta: Real,
}

impl Model {
    /// `MESAsetup`'s defaults, `mesamparam.c`'s setters and `inpdomod.c`'s
    /// level check. The `pmf` flag is accepted and ignored: C forces `NMF`
    /// with a warning, and the port reproduces the model rather than the
    /// message.
    pub(super) fn resolve<'a>(
        resolved: &'a ResolvedModel<'_>,
    ) -> SpiceResult<(Self, Vec<&'a crate::netlist::ast::ParameterAssignment>)> {
        let (_, scalars) = jfet::typed_scalars(
            resolved,
            resolved.family() == ModelFamily::Pmf,
            ["nmf", "pmf"],
        );
        // `inpgmod.c::create_model`: a setter outside the model table that is
        // a settable instance parameter becomes a model-level instance
        // default, applied before the instance card's own setters.
        let (scalars, defaults) =
            crate::devices::schema::split_model_setters(scalars, &MODEL_SCHEMA, &INSTANCE_SCHEMA);
        let values = MODEL_SCHEMA.validate(scalars, &resolved.card().location)?;
        let r = |name: &str| jfet::required(&values, name);
        let alias = |names: &[&str]| jfet::last_setter(&values, names);
        let model = Self {
            threshold: alias(&["vto", "vt0"]).unwrap_or(-1.26),
            lambda: r("lambda")?,
            lambdahf: alias(&["lambdahf"]),
            vs: r("vs")?,
            rd: r("rd")?,
            rs: r("rs")?,
            rg: r("rg")?,
            ri: r("ri")?,
            rf: r("rf")?,
            rdi: r("rdi")?,
            rsi: r("rsi")?,
            phib: CHARGE * r("phib")?,
            phib1: CHARGE * alias(&["phib1", "tphib"]).unwrap_or(0.),
            astar: r("astar")?,
            ggr: r("ggr")?,
            del: r("del")?,
            xchi: alias(&["xchi", "tggr"]).unwrap_or(0.033),
            n: r("n")?,
            eta: r("eta")?,
            m: r("m")?,
            mc: r("mc")?,
            alpha: r("alpha")?,
            sigma0: r("sigma0")?,
            vsigmat: r("vsigmat")?,
            vsigma: r("vsigma")?,
            mu: r("mu")?,
            theta: r("theta")?,
            mu1: r("mu1")?,
            mu2: r("mu2")?,
            d: r("d")?,
            nd: r("nd")?,
            beta: r("beta")?,
            ndu: r("ndu")?,
            ndelta: r("ndelta")?,
            nmax: r("nmax")?,
            gamma: r("gamma")?,
            epsi: r("epsi")?,
            cas: r("cas")?,
            cbs: r("cbs")?,
            du: r("du")?,
            th: r("th")?,
            delta: r("delta")?,
            tc: r("tc")?,
            tvto: alias(&["tvto", "alphat"]).unwrap_or(0.),
            tlambda: r("tlambda")?,
            teta0: r("teta0")?,
            teta1: r("teta1")?,
            tmu: r("tmu")?,
            xtm0: r("xtm0")?,
            xtm1: r("xtm1")?,
            xtm2: r("xtm2")?,
            ks: r("ks")?,
            vsg: r("vsg")?,
            tf: values.get("tf").map(|v| v.value),
            flo: r("flo")?,
            delfo: r("delfo")?,
            ag: r("ag")?,
            tc1: r("rtc1")?,
            tc2: r("rtc2")?,
            zeta: r("zeta")?,
        };
        if model.vsigma == 0. {
            return Err(SpiceError::circuit(format!(
                "{}: MESA model '{}' needs VSIGMA > 0 (mesa1 divides by it)",
                resolved.card().location,
                resolved.card().name
            )));
        }
        if model.d == 0. || model.du + model.th == 0. {
            return Err(SpiceError::circuit(format!(
                "{}: MESA model '{}' needs a nonzero device depth",
                resolved.card().location,
                resolved.card().name
            )));
        }
        Ok((model, defaults))
    }
}

/// Validated instance setters.
#[derive(Debug, Clone, Copy)]
pub(super) struct Instance {
    pub(super) l: Real,
    pub(super) w: Real,
    pub(super) m: Real,
    /// Celsius as written (`mesaparam.c` adds 273.15); `None` is
    /// `CKTtemp + dtemp` used as kelvin.
    pub(super) td: Option<Real>,
    pub(super) ts: Option<Real>,
    pub(super) dtemp: Real,
}

impl Instance {
    pub(super) fn resolve(
        i: &crate::netlist::ast::DeviceInstance,
    ) -> SpiceResult<(
        crate::devices::initial::InstanceInitial,
        Vec<crate::netlist::ast::ParameterAssignment>,
    )> {
        crate::devices::initial::split(&i.parameters, &IC_COMPONENTS)
    }

    pub(super) fn validate(
        defaults: &[&crate::netlist::ast::ParameterAssignment],
        setters: &[crate::netlist::ast::ParameterAssignment],
        owner: &crate::primitives::SourceLoc,
    ) -> SpiceResult<Self> {
        let values = crate::devices::schema::with_defaults(
            &INSTANCE_SCHEMA,
            defaults,
            setters.iter(),
            owner,
        )?;
        Ok(Self {
            l: jfet::required(&values, "l")?,
            w: jfet::required(&values, "w")?,
            m: jfet::required(&values, "m")?,
            td: values.get("td").map(|v| v.value),
            ts: values.get("ts").map(|v| v.value),
            dtemp: jfet::required(&values, "dtemp")?,
        })
    }
}

pub(super) static HFET_INSTANCE_SCHEMA: ScalarSchema<'static> = ScalarSchema::new(HFET_INSTANCE);

static INSTANCE_SCHEMA: ScalarSchema<'static> = ScalarSchema::new(INSTANCE);

static MODEL_SCHEMA: ScalarSchema<'static> = ScalarSchema::new(MODEL);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::primitives::NodeTable;

    #[test]
    fn soft_density_preserves_weak_inversion_and_avoids_exponential_overflow() {
        let (density, slope) = super::soft_density(2., -50., 1.);
        let asymptote = 2. * (-50_f64).exp();
        assert!((density / asymptote - 1.).abs() < 4. * f64::EPSILON);
        assert!((slope / asymptote - 1.).abs() < 4. * f64::EPSILON);
        let (density, slope) = super::soft_density(2., 1000., 1.);
        assert_eq!(density, 4. * (1000. - std::f64::consts::LN_2));
        assert_eq!(slope, 4.);
    }

    #[test]
    fn series_density_avoids_overflow_and_matches_the_analytic_derivative() {
        let (density, slope) = super::series_density(1e200, 2e200, 1e200, 0.5);
        assert_eq!(density, 5e199);
        assert_eq!(slope, 1e200);
        let (density, slope) = super::series_density(1e-200, 2e-200, 1e-200, 0.5);
        assert_eq!(density, 5e-201);
        assert_eq!(slope, 1e-200);
        let (density, slope) = super::series_density(1., 2., f64::INFINITY, 0.5);
        assert_eq!(density, 1.);
        assert_eq!(slope, 2.);
    }

    #[test]
    fn carrier_density_derivative_agrees_with_independent_central_difference() {
        let n0 = 1e15;
        let thermal = 0.026;
        let other_thermal = 0.04;
        let density = |v: Real| {
            let (first, slope) = soft_density(n0, v, thermal);
            let second = n0 * (v / other_thermal).exp();
            series_density(first, slope, second, other_thermal)
        };
        for voltage in [-1., -0.4, -0.1, 0., 0.1, 0.4, 1.] {
            let (_, slope) = density(voltage);
            let h = thermal * Real::EPSILON.cbrt();
            let numerical = (density(voltage + h).0 - density(voltage - h).0) / (2. * h);
            assert!((slope - numerical).abs() <= 2e-8 * slope.abs());
        }
    }

    #[test]
    #[ignore = "opt-in representative carrier-density value and derivative workload"]
    fn carrier_density_numeric_workload() {
        use std::hint::black_box;
        use std::time::Instant;

        const COUNT: u32 = 1_000_000;
        #[expect(
            clippy::suboptimal_flops,
            reason = "benchmark retains the historical non-fused arithmetic as the comparison workload"
        )]
        fn old(voltage: Real) -> (Real, Real) {
            let n0 = 1e15;
            let thermal = 0.026;
            let other_thermal = 0.04;
            let b = (voltage / thermal).exp();
            let nsn = 2. * n0 * (1. + 0.5 * b).ln();
            let nsc = n0 * (voltage / other_thermal).exp();
            let slope = n0 / thermal / (1. / b + 0.5);
            let denominator = nsc + nsn;
            (
                nsn * nsc / denominator,
                nsc * (nsc * slope + nsn * nsn / other_thermal) / denominator.powi(2),
            )
        }
        let new = |voltage: Real| {
            let (first, slope) = soft_density(1e15, voltage, 0.026);
            let second = 1e15 * (voltage / 0.04).exp();
            series_density(first, slope, second, 0.04)
        };
        // Identical repeating ordinary bias grid, independent of solver work.
        // Validate both maps before timing; the analytic extreme cases above
        // intentionally do not compare with inaccurate legacy values.
        for i in 0..=1000_u32 {
            let v = Real::from(i) / 1000. - 0.5;
            let a = old(v);
            let b = new(v);
            for (reference, actual) in [(a.0, b.0), (a.1, b.1)] {
                assert!((reference - actual).abs() <= 2e-7 * actual.abs());
            }
        }
        let time = |label: &str, evaluate: &dyn Fn(Real) -> (Real, Real)| {
            let start = Instant::now();
            for i in 0..COUNT {
                let v = Real::from(i % 1001) / 1000. - 0.5;
                black_box(evaluate(black_box(v)));
            }
            println!(
                "{label}: {COUNT} value/derivative pairs in {:?}",
                start.elapsed()
            );
        };
        for _ in 0..3 {
            time("legacy density", &old);
            time("conditioned density", &new);
        }
    }

    #[test]
    fn borrowed_reset_stamps_match_owned_assembly_and_validate_before_mutation() {
        use crate::devices::linear::LinearSystem;
        use crate::devices::sensitivity::{ResetStamp, ResetStamps, StampAt, StampPart};
        use crate::devices::traits::MnaUnknowns;
        use crate::devices::{Device, ModelContext};

        let mut nodes = NodeTable::new();
        let a = nodes.intern("a").unwrap();
        let b = nodes.intern("b").unwrap();
        let ground = NodeId::GROUND;
        let mut unknowns = MnaUnknowns::new();
        unknowns.rebuild(&nodes);
        let model_context = ModelContext::new(27., 27.);
        let real = [
            (a, a, 1e16),
            (a, a, -1e16),
            (a, a, 1.),
            (a, b, -2.),
            (ground, a, 5.),
        ];
        let imaginary = [(a, b, 3.), (a, b, -4.), (b, b, 0.), (a, ground, 6.)];
        let owned_stamps = real
            .into_iter()
            .map(|(a, b, value)| ResetStamp {
                at: StampAt::Nodes(a, b),
                value,
                part: StampPart::Real,
            })
            .chain(imaginary.into_iter().map(|(a, b, value)| ResetStamp {
                at: StampAt::Nodes(a, b),
                value,
                part: StampPart::Imaginary,
            }))
            .collect();
        let owned = ResetStamps::new("z1", 'z', &[a, b], owned_stamps).unwrap();
        let mut reference = LinearSystem::new(unknowns.len());
        let mut actual = LinearSystem::new(unknowns.len());
        for (system, reference_path) in [(&mut reference, true), (&mut actual, false)] {
            let mut context = LinearContext {
                model_context: &model_context,
                system,
                unknowns: &unknowns,
                branch: None,
                controls: &[],
                mutual: &[],
                states: None,
                pole_zero_statements: None,
                pole_zero_setups: 0,
            };
            if reference_path {
                owned.assemble_linear(&mut context).unwrap();
            } else {
                assert!(
                    stamp_reset("z1", &mut context, &real, &[(ground, ground, Real::NAN)]).is_err()
                );
                assert!(context.system.a.triplets().is_empty());
                assert!(context.system.e.triplets().is_empty());
                stamp_reset("z1", &mut context, &real, &imaginary).unwrap();
            }
        }
        assert_eq!(actual.a.triplets(), reference.a.triplets());
        assert_eq!(actual.e.triplets(), reference.e.triplets());
        assert_eq!(actual.e_groups, reference.e_groups);
    }

    #[test]
    fn temperature_cache_uses_full_exact_key_and_keeps_last_success() {
        let cache = TemperatureCache::<u32, 2>::new();
        assert_eq!(cache.get_or_compute([27., 27.], || Ok(1)).unwrap(), 1);
        assert_eq!(
            cache
                .get_or_compute([27., 27.], || panic!("unexpected recomputation"))
                .unwrap(),
            1
        );
        assert_eq!(cache.get_or_compute([27., 28.], || Ok(2)).unwrap(), 2);
        assert!(
            cache
                .get_or_compute([0., 28.], || Err(SpiceError::circuit("failed derivation")))
                .is_err()
        );
        assert_eq!(
            cache
                .get_or_compute([27., 28.], || panic!("lost successful snapshot"))
                .unwrap(),
            2
        );
        assert_eq!(cache.get_or_compute([0., 28.], || Ok(3)).unwrap(), 3);
        assert_eq!(cache.get_or_compute([-0., 28.], || Ok(4)).unwrap(), 4);
    }

    #[test]
    fn hfet_sensitivity_temperature_round_trips_given_flags() {
        let instance = HfetInstance {
            l: 1e-6,
            w: 20e-6,
            m: 2.,
            temp: None,
            dtemp: 12.3,
        };
        let initial = InstanceInitial {
            off: false,
            values: vec![Some(0.5), None],
        };
        let mut record = instance
            .sensitivity_record(
                crate::devices::sensitivity::SensitivityFamily::Hfet1,
                &initial,
                300.15,
            )
            .unwrap();
        assert!(!record.given("temp"));
        assert!(record.given("icvds"));
        assert!(!record.given("icvgs"));
        assert!(HfetInstance::from_record(&record).unwrap().temp.is_none());
        let mut model = SensitivityRecord::new(
            crate::devices::sensitivity::SensitivityFamily::Hfet1,
            ParameterScope::Model,
        );
        set_hfet(
            ParameterScope::Instance,
            "temp",
            -12.5,
            &mut model,
            &mut record,
        )
        .unwrap();
        let pair = RecordPair {
            model: &model,
            instance: &record,
        };
        assert_eq!(
            ask_hfet(ParameterScope::Instance, "temp", pair)
                .unwrap()
                .to_bits(),
            (-12.5_f64).to_bits()
        );
        assert_eq!(
            HfetInstance::from_record(&record)
                .unwrap()
                .temp
                .unwrap()
                .to_bits(),
            (-12.5_f64).to_bits()
        );
        assert!(record.given("temp"));
    }

    #[test]
    fn sweeps_match_deck_scalar_domains_for_all_hfet_families() {
        use crate::devices::{Circuit, ModelContext};
        use crate::netlist::{Parser, source::parse_deck_text};
        use std::path::Path;

        let context = ModelContext::default();
        for (base, level, temperatures) in [
            ("nmf", 2, &["td", "ts"][..]),
            ("nhfet", 5, &["temp"][..]),
            ("nhfet", 6, &["temp"][..]),
        ] {
            let deck = Parser::new()
                .parse_deck(&parse_deck_text(
                    Path::new("hfet-sweeps.cir"),
                    &format!(
                        "HFET sweeps\nz1 d g 0 test\n.model test {base} level={level}\n.end\n"
                    ),
                ))
                .unwrap();
            let circuit = Circuit::from_netlist(&deck).unwrap();
            let device = circuit.device("z1").unwrap();
            for field in ["l", "w", "m"] {
                let field = device.instance_parameter(field).unwrap();
                for invalid in [0., -1., Real::INFINITY, Real::NAN] {
                    assert!(
                        device
                            .with_instance_parameter(field, invalid, &context)
                            .is_err(),
                        "level {level} accepted {field}={invalid}"
                    );
                }
            }
            for field in temperatures {
                let field = device.instance_parameter(field).unwrap();
                for invalid in [-273.15, -300., Real::INFINITY, Real::NAN] {
                    assert!(
                        device
                            .with_instance_parameter(field, invalid, &context)
                            .is_err(),
                        "level {level} accepted {field}={invalid}"
                    );
                }
                let swept = device
                    .with_instance_parameter(field, 42., &context)
                    .unwrap();
                assert_eq!(swept.terminals(), device.terminals());
            }
            assert!(
                device
                    .with_instance_parameter("dtemp", Real::INFINITY, &context)
                    .is_err()
            );
            assert!(device.with_instance_parameter("m", 2., &context).is_ok());
        }
    }

    #[test]
    fn failed_temperature_setup_does_not_publish_nodes() {
        use crate::devices::models::ModelResolver;
        use crate::devices::{Device, ModelContext};
        use crate::netlist::{Parser, source::parse_deck_text};
        use std::path::Path;

        type Instantiate = fn(
            &DeviceInstance,
            &mut NodeAppend<'_>,
            &ResolvedModel<'_>,
            &ModelContext,
        ) -> SpiceResult<Box<dyn Device>>;
        for (base, level, instantiate) in [
            ("nmf", 2, mesa::Mesa::instantiate as Instantiate),
            ("nhfet", 5, hfet1::Hfet1::instantiate as Instantiate),
            ("nhfet", 6, hfet2::Hfet2::instantiate as Instantiate),
        ] {
            let deck = Parser::new().parse_deck(&parse_deck_text(
                Path::new("hfet-setup.cir"),
                &format!("Setup transaction\nz1 drain gate source test\n.model test {base} level={level} rs=10 rd=20\n.end\n"),
            )).unwrap();
            let instance = &deck.devices[0];
            let resolver = ModelResolver::new(&deck.models).unwrap();
            let resolved = resolver.resolve(instance).unwrap().unwrap();
            let mut nodes = NodeTable::new();
            let existing = nodes.intern("existing").unwrap();
            let count = nodes.len();
            let Err(error) = nodes.transaction(|nodes| {
                instantiate(
                    instance,
                    nodes,
                    &resolved,
                    &ModelContext::new(-CELSIUS_TO_KELVIN, 27.),
                )
            }) else {
                panic!("invalid temperature accepted for level {level}");
            };
            assert!(
                error.to_string().contains("temperature derivation"),
                "{error}"
            );
            assert_eq!(nodes.len(), count);
            assert_eq!(nodes.get("existing"), Some(existing));
            assert_eq!(nodes.get("drain"), None);
            assert_eq!(nodes.get("z1#source"), None);
            let device = nodes
                .transaction(|nodes| {
                    instantiate(instance, nodes, &resolved, &ModelContext::new(27., 27.))
                })
                .unwrap();
            assert_eq!(device.terminals().len(), 5);
            assert_eq!(device.terminals()[3], nodes.get("z1#source").unwrap());
            assert_eq!(device.terminals()[4], nodes.get("z1#drain").unwrap());
        }
    }

    #[test]
    fn hfet_temperature_uses_explicit_value_or_circuit_offset_in_c_order() {
        let mut instance = HfetInstance {
            l: 1e-6,
            w: 20e-6,
            m: 1.,
            temp: None,
            dtemp: 12.3,
        };
        let circuit_celsius: Real = 37.1;
        let circuit_kelvin = circuit_celsius + CELSIUS_TO_KELVIN;
        assert_eq!(
            instance.temperature(circuit_celsius).to_bits(),
            (circuit_kelvin + instance.dtemp).to_bits()
        );
        assert_eq!(
            instance.temperature_at_kelvin(circuit_kelvin).to_bits(),
            instance.temperature(circuit_celsius).to_bits()
        );
        instance.temp = Some(-11.7);
        assert_eq!(
            instance.temperature(circuit_celsius).to_bits(),
            (-11.7 + CELSIUS_TO_KELVIN).to_bits()
        );
        assert_eq!(
            instance.temperature_at_kelvin(circuit_kelvin).to_bits(),
            instance.temperature(circuit_celsius).to_bits()
        );
    }
}
