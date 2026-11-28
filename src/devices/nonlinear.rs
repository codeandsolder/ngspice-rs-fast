//! Junction equations, the level-1 diode and charge-aware trial stamping.
//!
//! C references: `dio/dioload.c`, `diosetup.c`, `diotemp.c`, `dioacld.c`,
//! `diompar.c`, `dioparam.c` and `NIintegrate`. The diode covers reverse
//! breakdown (BV/IBV/NBV/TCV), the full `diotemp.c` temperature laws (EG, XTI,
//! GAP1/GAP2, TLEV, TLEVC, DTEMP, TNOM), the sidewall junction (JSW, NS, CJSW,
//! VJSW, MJSW, FCS, PJ), recombination (ISR/NR), tunnelling (JTUN/JTUNSW/NTUN),
//! high-injection knees (IKF/IKR/IKP) and the transit-time/series-resistance
//! temperature coefficients, and the `.noise` generators of `dionoise.c`
//! (series-resistance thermal, junction shot and KF/AF flicker noise).
//! Soft reverse recovery uses an independent conservative charge state.
//! Separate sidewall series resistance, self-heating, level-3 geometry and SOA setters fail
//! with [`SpiceError::NotYetPorted`] before node interning; unknown setters
//! remain unsupported errors. The instance `off` flag follows `dioload.c`
//! (`MODEINITJCT`/`MODEINITFIX` hold at 0 V). The `uic` initial load starts
//! the junction at the external terminal voltage of the `.ic`/`.nodeset`
//! node vector (`diogetic.c`); the instance `ic=` is accepted but has no
//! effect, because C's setter (`dioparam.c`) never sets `DIOinitCondGiven`
//! and `diogetic.c` therefore always overwrites it.
//!
//! The diode limits its junction voltage as `dioload.c` does (`MODEINITJCT`
//! start at `tVcrit`, `DEVpnjlim`, reflected about BV in breakdown) through
//! [`crate::devices::limiting`]; C's predictor and bypass are not ported.
use crate::devices::limiting::{self, Limiter, Linearization};
use crate::devices::noise::{
    DeviceNoise, FlickerLaw, NoiseContext, NoiseFamily, NoiseKind, NoiseSource,
};
use crate::devices::schema::{
    ScalarDomain as Domain, ScalarParameter as Parameter, ScalarSchema, ScalarUnit as Unit,
    ScalarValues,
};
use crate::devices::{
    AnalysisMode, Device, DeviceState, LinearContext, MnaUnknowns, ModelContext, ResolvedModel,
    StampContext,
};
use crate::maths::{Coefficients, Vector};
use crate::netlist::ast::{DeviceInstance, ParameterAssignment};
use crate::primitives::{NodeAppend, NodeId, NodeKind, Real, SourceLoc, SpiceError, SpiceResult};

pub(super) mod charge;
mod recovery;
mod sens;
pub(super) mod taylor;
pub(super) mod thermal;
use thermal::{
    BOLTZMANN, CHARGE, DerivedCache, K_OVER_Q, PotentialTemperature, REFTEMP, TemperatureKey,
    band_gap, potential_shift,
};

/// A series branch between two aliases of one electrical node contributes
/// identically zero. Remove it before grouping diagonal conductances: a
/// large perturbed resistance inverse would otherwise round away the
/// intrinsic Jacobian terms before the sparse accumulator can cancel it.
#[inline]
pub(super) fn unaliased_conductance(a: NodeId, b: NodeId, conductance: Real) -> Real {
    if a == b { 0. } else { conductance }
}

/// Diode state slot of the limited junction voltage (C `DIOvoltage`).
const DIODE_CHARGE_SLOT: usize = 0;
const DIODE_VOLTAGE_SLOT: usize = 2;
/// ngspice default `CKTepsmin` (`cktntask.c`): IS floor and knee-current cutoff.
const EPSMIN: Real = 1e-28;
/// The diode breakdown derivation also depends on the circuit RELTOL.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct DiodeThermalKey {
    temperatures: TemperatureKey,
    reltol: u64,
}

impl DiodeThermalKey {
    const fn context(context: &ModelContext) -> Self {
        Self {
            temperatures: TemperatureKey::context(context),
            reltol: context.reltol.to_bits(),
        }
    }
}

const fn scalar(
    name: &'static str,
    unit: Unit,
    domain: Domain,
    default: Option<Real>,
) -> Parameter {
    Parameter {
        name,
        unit,
        domain,
        default,
    }
}

/// The supported diode model setters, canonical names only (aliases are folded
/// by [`MODEL_ALIASES`] first). Defaults follow `diosetup.c`; setters without a
/// default are C "given" flags or context/model-dependent defaults.
static MODEL: ScalarSchema<'static> = ScalarSchema::new(&[
    scalar("is", Unit::Ampere, Domain::Positive, Some(1e-14)),
    scalar("jsw", Unit::Ampere, Domain::NonNegative, None),
    scalar("n", Unit::Dimensionless, Domain::Positive, Some(1.)),
    scalar("ns", Unit::Dimensionless, Domain::Positive, Some(1.)),
    scalar("rs", Unit::Ohm, Domain::NonNegative, Some(0.)),
    scalar("trs", Unit::InverseKelvin, Domain::Finite, Some(0.)),
    scalar("trs2", Unit::InverseKelvinSquared, Domain::Finite, Some(0.)),
    scalar("tnom", Unit::Celsius, Domain::Temperature, None),
    scalar("tt", Unit::Second, Domain::NonNegative, Some(0.)),
    scalar("vp", Unit::Dimensionless, Domain::NonNegative, Some(0.)),
    scalar("qpscale", Unit::Dimensionless, Domain::Positive, Some(1e6)),
    scalar("ttt1", Unit::InverseKelvin, Domain::Finite, Some(0.)),
    scalar("ttt2", Unit::InverseKelvinSquared, Domain::Finite, Some(0.)),
    scalar("cjo", Unit::Farad, Domain::NonNegative, Some(0.)),
    scalar("vj", Unit::Volt, Domain::Positive, Some(1.)),
    scalar("m", Unit::Dimensionless, Domain::NonNegative, Some(0.5)),
    scalar("tm1", Unit::InverseKelvin, Domain::Finite, Some(0.)),
    scalar("tm2", Unit::InverseKelvinSquared, Domain::Finite, Some(0.)),
    scalar("fc", Unit::Dimensionless, Domain::NonNegative, Some(0.5)),
    scalar("cjsw", Unit::Farad, Domain::NonNegative, Some(0.)),
    scalar("vjsw", Unit::Volt, Domain::Positive, Some(1.)),
    scalar("mjsw", Unit::Dimensionless, Domain::NonNegative, Some(0.33)),
    scalar("fcs", Unit::Dimensionless, Domain::NonNegative, Some(0.5)),
    scalar("bv", Unit::Volt, Domain::Positive, None),
    scalar("ibv", Unit::Ampere, Domain::Positive, Some(1e-3)),
    scalar("nbv", Unit::Dimensionless, Domain::Positive, None),
    scalar("tcv", Unit::VoltPerKelvin, Domain::Finite, Some(0.)),
    scalar("tlev", Unit::Dimensionless, Domain::Finite, Some(0.)),
    scalar("tlevc", Unit::Dimensionless, Domain::Finite, Some(0.)),
    scalar("eg", Unit::ElectronVolt, Domain::Positive, None),
    scalar(
        "gap1",
        Unit::ElectronVoltPerKelvin,
        Domain::Finite,
        Some(7.02e-4),
    ),
    scalar("gap2", Unit::Kelvin, Domain::Finite, Some(1108.)),
    scalar("xti", Unit::Dimensionless, Domain::Finite, Some(3.)),
    scalar("cta", Unit::InverseKelvin, Domain::Finite, Some(0.)),
    scalar("ctp", Unit::InverseKelvin, Domain::Finite, Some(0.)),
    scalar("tpb", Unit::VoltPerKelvin, Domain::Finite, Some(0.)),
    scalar("tphp", Unit::VoltPerKelvin, Domain::Finite, Some(0.)),
    scalar("isr", Unit::Ampere, Domain::NonNegative, None),
    scalar("nr", Unit::Dimensionless, Domain::Positive, Some(2.)),
    scalar("ikf", Unit::Ampere, Domain::Finite, None),
    scalar("ikr", Unit::Ampere, Domain::Finite, None),
    scalar("ikp", Unit::Ampere, Domain::Finite, None),
    scalar("jtun", Unit::Ampere, Domain::NonNegative, None),
    scalar("jtunsw", Unit::Ampere, Domain::NonNegative, None),
    scalar("ntun", Unit::Dimensionless, Domain::Positive, Some(30.)),
    scalar("xtitun", Unit::Dimensionless, Domain::Finite, Some(3.)),
    scalar("keg", Unit::Dimensionless, Domain::Finite, Some(1.)),
    scalar("area", Unit::Dimensionless, Domain::Positive, Some(1.)),
    scalar("pj", Unit::Dimensionless, Domain::NonNegative, Some(0.)),
    // dionoise.c flicker law; diosetup.c defaults KF = 0, AF = 1.
    scalar("kf", Unit::Dimensionless, Domain::Finite, Some(0.)),
    scalar("af", Unit::Dimensionless, Domain::Finite, Some(1.)),
]);
/// `dio.c::DIOmPTable` aliases (`IOPR` entries share the canonical setter id, so
/// last-set precedence spans the alias and its canonical name).
const MODEL_ALIASES: &[(&str, &str)] = &[
    ("js", "is"),
    ("isw", "jsw"),
    ("tref", "tnom"),
    ("trs1", "trs"),
    ("cj0", "cjo"),
    ("cj", "cjo"),
    ("pb", "vj"),
    ("mj", "m"),
    ("cjp", "cjsw"),
    ("php", "vjsw"),
    ("ik", "ikf"),
    ("nz", "nbv"),
    ("vb", "bv"),
    ("vrb", "bv"),
    ("var", "bv"),
    ("ib", "ibv"),
    ("tbv1", "tcv"),
    ("ctc", "cta"),
    ("tvj", "tpb"),
];
/// Recognised C model setters whose physics is not ported yet.
const MODEL_PENDING: &[(&str, &str)] = &[
    (
        "rsw",
        "src/spicelib/devices/dio/diosetup.c, dioload.c (separate sidewall series resistance)",
    ),
    ("rth0", "src/spicelib/devices/dio/dioload.c (self-heating)"),
    ("cth0", "src/spicelib/devices/dio/dioload.c (self-heating)"),
    ("fv_max", "src/spicelib/devices/dio/diosoachk.c"),
    ("bv_max", "src/spicelib/devices/dio/diosoachk.c"),
    ("id_max", "src/spicelib/devices/dio/diosoachk.c"),
    ("te_max", "src/spicelib/devices/dio/diosoachk.c"),
    ("pd_max", "src/spicelib/devices/dio/diosoachk.c"),
    (
        "lm",
        "src/spicelib/devices/dio/diosetup.c (level 3 geometry)",
    ),
    (
        "lp",
        "src/spicelib/devices/dio/diosetup.c (level 3 geometry)",
    ),
    (
        "wm",
        "src/spicelib/devices/dio/diosetup.c (level 3 geometry)",
    ),
    (
        "wp",
        "src/spicelib/devices/dio/diosetup.c (level 3 geometry)",
    ),
    (
        "xom",
        "src/spicelib/devices/dio/diosetup.c (level 3 geometry)",
    ),
    (
        "xoi",
        "src/spicelib/devices/dio/diosetup.c (level 3 geometry)",
    ),
    (
        "xm",
        "src/spicelib/devices/dio/diosetup.c (level 3 geometry)",
    ),
    (
        "xp",
        "src/spicelib/devices/dio/diosetup.c (level 3 geometry)",
    ),
    (
        "xw",
        "src/spicelib/devices/dio/diosetup.c (level 3 geometry)",
    ),
];
static INSTANCE: ScalarSchema<'static> = ScalarSchema::new(&[
    scalar("area", Unit::Dimensionless, Domain::Positive, None),
    scalar("pj", Unit::Dimensionless, Domain::NonNegative, None),
    scalar("m", Unit::Dimensionless, Domain::Positive, Some(1.)),
    scalar("temp", Unit::Celsius, Domain::Temperature, None),
    scalar("dtemp", Unit::Kelvin, Domain::Finite, None),
]);
/// Recognised C instance setters whose behaviour is not ported yet.
const INSTANCE_PENDING: &[(&str, &str)] = &[
    (
        "thermal",
        "src/spicelib/devices/dio/dioload.c (self-heating)",
    ),
    (
        "w",
        "src/spicelib/devices/dio/diosetup.c (level 3 geometry)",
    ),
    (
        "l",
        "src/spicelib/devices/dio/diosetup.c (level 3 geometry)",
    ),
    (
        "lm",
        "src/spicelib/devices/dio/diosetup.c (level 3 geometry)",
    ),
    (
        "lp",
        "src/spicelib/devices/dio/diosetup.c (level 3 geometry)",
    ),
    (
        "wm",
        "src/spicelib/devices/dio/diosetup.c (level 3 geometry)",
    ),
    (
        "wp",
        "src/spicelib/devices/dio/diosetup.c (level 3 geometry)",
    ),
];

pub(crate) fn value(values: &ScalarValues, name: &str) -> SpiceResult<Real> {
    values
        .get(name)
        .map(|v| v.value)
        .ok_or_else(|| SpiceError::circuit(format!("missing nonlinear schema default {name}")))
}
/// An explicitly written setter (a schema default has no location).
fn given(values: &ScalarValues, name: &str) -> Option<Real> {
    values
        .get(name)
        .filter(|v| v.location.is_some())
        .map(|v| v.value)
}
/// Reject recognised-but-unported setters with their C reference, then fold
/// aliases onto canonical names in their original order.
fn canonical<'a>(
    assignments: impl IntoIterator<Item = &'a ParameterAssignment>,
    aliases: &[(&str, &str)],
    pending: &[(&str, &str)],
    owner: &str,
) -> SpiceResult<Vec<ParameterAssignment>> {
    let mut out = Vec::new();
    for assignment in assignments {
        let name = assignment.name.to_ascii_lowercase();
        if let Some((_, reference)) = pending.iter().find(|(p, _)| *p == name) {
            return Err(SpiceError::not_yet_ported(
                format!("{}: diode {owner} setter '{name}'", assignment.location),
                *reference,
            ));
        }
        let mut assignment = assignment.clone();
        if let Some((_, target)) = aliases.iter().find(|(alias, _)| *alias == name) {
            (*target).clone_into(&mut assignment.name);
        }
        out.push(assignment);
    }
    Ok(out)
}
/// `floor(raw + 0.5)` integer selector (`inpgval.c` `IF_INTEGER`) within `0..=max`.
fn selector(values: &ScalarValues, name: &str, max: u8, owner: &SourceLoc) -> SpiceResult<u8> {
    let raw = value(values, name)?;
    crate::primitives::c_integer(raw)
        .and_then(|rounded| u8::try_from(rounded).ok())
        .filter(|rounded| *rounded <= max)
        .ok_or_else(|| {
            SpiceError::parse(
                values
                    .get(name)
                    .and_then(|v| v.location.clone())
                    .unwrap_or_else(|| owner.clone()),
                format!("diode {name} must round to 0..={max}, got {raw}"),
            )
        })
}

/// A level-1 diode with an optional internal anode for series resistance.
/// Only the explicitly enumerated diode model/instance schema is supported.
#[derive(Debug)]
pub struct Diode {
    name: String,
    /// The model card's name, for C's `.noise` visiting order.
    model: String,
    terminals: Vec<NodeId>,
    junction: [NodeId; 2],
    recovery_node: Option<recovery::RecoveryBinding>,
    parameters: DiodeParameters,
    derived: DerivedCache<DiodeThermalKey, Thermal>,
    /// `off` and the instance `ic` (C `DIOoff`, `DIOinitCond`). The `ic`
    /// value is validated and kept but, exactly as in C, never used: see
    /// the `uic` start in `stamp`.
    initial: crate::devices::initial::InstanceInitial,
    /// The instance card, for diagnostics of swept replacements.
    location: SourceLoc,
    /// The validated model setters as written, shared by swept and perturbed copies.
    written_model: ScalarValues,
    /// A `.sens` stand-in: the load keeps C's nonfinite values (a knee
    /// current of zero divides by zero in `dioload.c`) instead of rejecting
    /// them, so the analysis can propagate C's NaN.
    lenient: bool,
}
/// `DIOsetup`'s saved quantities. `DIOtemp` must not reconstruct these from
/// perturbed model or instance parameters: the original setup is still live.
#[derive(Debug, Clone, Copy)]
struct DiodeSetup {
    conductance: Real,
    knees: [Real; 3],
}

impl DiodeSetup {
    fn conductance(rs: Real) -> Real {
        if rs == 0. { 0. } else { 1. / rs }
    }

    fn new(rs: Real, knees: [Real; 3], geometry: DiodeGeometry) -> Self {
        Self {
            conductance: Self::conductance(rs),
            knees: Self::scale_knees(knees, geometry),
        }
    }

    fn scale_knees(knees: [Real; 3], geometry: DiodeGeometry) -> [Real; 3] {
        [
            knees[0] * geometry.area * geometry.multiplier,
            knees[1] * geometry.area * geometry.multiplier,
            knees[2] * geometry.perimeter * geometry.multiplier,
        ]
    }
}

/// The C setup geometry, after applying model defaults and instance setters.
#[derive(Debug, Clone, Copy)]
struct DiodeGeometry {
    area: Real,
    perimeter: Real,
    multiplier: Real,
}

impl DiodeGeometry {
    fn setup(
        area: Option<Real>,
        perimeter: Option<Real>,
        multiplier: Option<Real>,
        defaults: [Real; 2],
    ) -> Self {
        let area = area.unwrap_or(defaults[0]);
        let multiplier = multiplier.unwrap_or(1.);
        Self {
            area: if area > 0. { area } else { 1. },
            perimeter: perimeter.unwrap_or(defaults[1]),
            multiplier: if multiplier > 0. { multiplier } else { 1. },
        }
    }
}

/// Explicit instance temperature is stored in Kelvin, not round-tripped in replay.
#[derive(Debug, Clone, Copy)]
struct DiodeInstanceParameters {
    geometry: DiodeGeometry,
    /// AREA, PJ and M given flags, in that order.
    geometry_given: [bool; 3],
    temperature: Option<Real>,
    dtemp: Option<Real>,
    width: Option<Real>,
    length: Option<Real>,
}

impl DiodeInstanceParameters {
    fn kelvin(self, circuit_celsius: Real) -> Real {
        self.temperature
            .unwrap_or_else(|| circuit_celsius + 273.15 + self.dtemp.unwrap_or(0.))
    }
}

const fn default_eg(tlev: u8) -> Real {
    if tlev == 2 { 1.16 } else { 1.11 }
}

/// Defaulted nominal model fields. Optional currents retain the C given flags.
#[derive(Debug, Clone, Copy)]
struct DiodeModelParameters {
    is: Real,
    /// JSW; `Some` (even zero) enables the sidewall current (C `DIOsatSWCurGiven`).
    jsw: Option<Real>,
    n: Real,
    ns: Real,
    /// NS given: the sidewall current has its own characteristic.
    ns_given: bool,
    rs: Option<Real>,
    trs1: Real,
    trs2: Real,
    tt: Real,
    vp: Real,
    qpscale: Real,
    ttt1: Real,
    ttt2: Real,
    cjo: Real,
    vj: Real,
    grading: Real,
    tm1: Real,
    tm2: Real,
    fc: Real,
    cjsw: Real,
    vjsw: Real,
    mjsw: Real,
    fcs: Real,
    bv: Option<Real>,
    ibv: Real,
    nbv: Real,
    tcv: Real,
    tlev: u8,
    tlevc: u8,
    eg: Real,
    gap1: Real,
    gap2: Real,
    xti: Real,
    cta: Real,
    ctp: Real,
    tpb: Real,
    tphp: Real,
    isr: Option<Real>,
    nr: Real,
    knee_given: [bool; 3],
    jtun: Option<Real>,
    jtunsw: Option<Real>,
    ntun: Real,
    xtitun: Real,
    keg: Real,
    nominal: Option<Real>,
    /// Flicker-noise coefficient and exponent (`dionoise.c`).
    kf: Real,
    af: Real,
}
/// The nominal model, current instance settings and saved setup state are
/// separate owners. `DIOtemp` derives a fresh thermal state from all three.
#[derive(Debug, Clone, Copy)]
struct DiodeParameters {
    model: DiodeModelParameters,
    instance: DiodeInstanceParameters,
    setup: DiodeSetup,
}
/// State-independent inputs for a single diode physical evaluation.
/// Sampling determines limiting and accepted-history decisions before
/// solving the pure junction equations and committing the ordered stamp.
#[derive(Debug, Clone, Copy)]
pub(crate) struct DiodeLoadSample {
    thermal: Thermal,
    limiter: Limiter,
    raw: Real,
    voltage: Real,
    c_jacobian: bool,
    lenient: bool,
}

impl DiodeLoadSample {
    /// Evaluate physical junction equations without mutating solver state.
    /// # Errors
    /// Nonfinite or unsupported junction physics.
    pub(crate) fn evaluate(self, gmin: Real) -> SpiceResult<DiodeLoadEvaluation> {
        let mut point = if self.lenient {
            self.thermal.point_unchecked(self.voltage, gmin)?
        } else {
            self.thermal.point(self.voltage, gmin)?
        };
        if self.c_jacobian {
            point.junction.conductance = point.ac_conductance;
        }
        Ok(DiodeLoadEvaluation {
            thermal: self.thermal,
            point,
            limiter: self.limiter,
            raw: self.raw,
            voltage: self.voltage,
        })
    }
}

/// Pure junction result of one diode load, including its independently
/// evaluated limiting decision. The caller stamps it in original C order.
#[derive(Debug, Clone, Copy)]
pub(crate) struct DiodeLoadEvaluation {
    thermal: Thermal,
    point: DiodePoint,
    limiter: Limiter,
    raw: Real,
    voltage: Real,
}

impl Diode {
    /// Gather immutable model and prior-state inputs with C's limiter
    /// semantics. This never writes a trial slot or MNA stamp.
    /// # Errors
    /// Invalid model input or missing accepted predictor history.
    pub(crate) fn sample_load(
        &self,
        model: &ModelContext,
        unknowns: &MnaUnknowns,
        solution: &Vector,
        states: &DeviceState<'_>,
        mode: AnalysisMode,
        integration: Option<&Coefficients>,
    ) -> SpiceResult<DiodeLoadSample> {
        if mode.is_ac() {
            return Err(SpiceError::circuit("diode AC needs small-signal assembly"));
        }
        let thermal = self.thermal(model)?;
        let mut limiter = Limiter::new(states);
        let node_voltage = |node| {
            unknowns
                .node_row(node)
                .and_then(|row| solution.get(row))
                .unwrap_or(0.)
        };
        // Preserve C's predictor extrapolation before limiting, including
        // a missing-state error and the DC sweep predictor ratio of one.
        let raw = match (states.phase(), limiter.mode()) {
            (crate::devices::IterationPhase::Predict, Linearization::Limited) => {
                let ratio = integration.map_or(1., Coefficients::predictor_ratio);
                states
                    .predicted(DIODE_VOLTAGE_SLOT, ratio)
                    .ok_or_else(|| SpiceError::circuit("missing accepted diode voltage"))?
            }
            _ => node_voltage(self.junction[0]) - node_voltage(self.junction[1]),
        };
        // C ignores instance ic= unless DIOinitCondGiven was separately set.
        let start = node_voltage(self.terminals[0]) - node_voltage(self.terminals[1]);
        let voltage = thermal.limit(&mut limiter, states, raw, start, self.initial.off);
        Ok(DiodeLoadSample {
            thermal,
            limiter,
            raw,
            voltage,
            c_jacobian: states.c_jacobian(),
            lenient: self.lenient,
        })
    }

    /// Ordered state and matrix commit of the previously independent diode
    /// physics. An error from a later diode is reported only after earlier
    /// devices have been stamped, exactly as in scalar load order.
    /// # Errors
    /// Invalid companion step, state access, or MNA bound element.
    pub(crate) fn commit_load(
        &self,
        context: &mut StampContext<'_>,
        evaluation: &DiodeLoadEvaluation,
    ) -> SpiceResult<()> {
        let mut limiter = evaluation.limiter;
        let p = &evaluation.point;
        let v = evaluation.voltage;
        let held = p.junction.current;
        let predicted_change = p.junction.conductance * (evaluation.raw - v);
        limiter.test_held(&context.states, &[(held, held + predicted_change)]);
        self.stamp_junction(context, &evaluation.thermal, p, limiter, v)
    }

    fn stamp_junction(
        &self,
        context: &mut StampContext<'_>,
        thermal: &Thermal,
        p: &DiodePoint,
        limiter: Limiter,
        v: Real,
    ) -> SpiceResult<()> {
        let gspr = if self.junction[0] == self.terminals[0] {
            0.
        } else {
            thermal.conductance
        };
        let (mut gd, mut cd) = (p.junction.conductance, p.junction.current);
        let recovery = self.recovery(thermal);
        let branch = if recovery.is_some() {
            charge::ChargeBranch::new(
                DIODE_CHARGE_SLOT,
                p.depletion_charge,
                p.depletion_capacitance,
            )
        } else {
            charge::ChargeBranch::new(DIODE_CHARGE_SLOT, p.junction.charge, p.junction.capacitance)
        };
        if let Some(coefficients) = context.integration {
            let crate::devices::AnalysisMode::Transient { dt, .. } = context.mode else {
                return Err(SpiceError::circuit("junction companion outside transient"));
            };
            if dt.to_bits() != coefficients.dt().to_bits() {
                return Err(SpiceError::circuit("junction timestep mismatch"));
            }
            let companion = branch.integrate(&mut context.states, coefficients)?;
            gd += companion.conductance;
            cd += companion.derivative;
        } else {
            if context.mode.is_transient() {
                return Err(SpiceError::circuit(
                    "junction transient needs companion integration",
                ));
            }
            branch.zero_derivative(&mut context.states)?;
        }
        branch.store(&mut context.states)?;
        let linearized_current = gd * v;
        let cdeq = cd - linearized_current;
        let [pos, neg] = [self.terminals[0], self.terminals[1]];
        let prime = self.junction[0];
        context.stamp_rhs(neg, cdeq)?;
        context.stamp_rhs(prime, -cdeq)?;
        context.stamp_fixed([
            (pos, pos, gspr),
            (neg, neg, gd),
            (prime, prime, gd + gspr),
            (pos, prime, -gspr),
            (neg, prime, -gd),
            (prime, pos, -gspr),
            (prime, neg, -gd),
        ])?;
        if let Some(recovery) = recovery {
            recovery.stamp(context, self.junction, p.junction, v)?;
        }
        limiter.finish(&mut context.states, &[(DIODE_VOLTAGE_SLOT, v)])
    }

    fn thermal(&self, context: &ModelContext) -> SpiceResult<Thermal> {
        self.derived
            .get_or_derive(DiodeThermalKey::context(context), || {
                self.parameters.thermal(context)
            })
    }

    pub(crate) fn instantiate(
        instance: &DeviceInstance,
        nodes: &mut NodeAppend<'_>,
        model: &ResolvedModel<'_>,
        context: &ModelContext,
    ) -> SpiceResult<Box<dyn Device>> {
        if instance.nodes.len() != 2 {
            return Err(SpiceError::circuit("diode needs two terminals"));
        }
        let card = model.card();
        // `inpgmod.c::create_model`: a `.model` setter outside the model table
        // that is a settable instance parameter (`area`, `m`, `pj`, `temp`,
        // `dtemp`) becomes a model-level instance default, applied before the
        // instance card's own setters.
        let (model_side, model_defaults) = crate::devices::schema::split_model_setters(
            card.parameters
                .iter()
                .filter(|p| !p.name.eq_ignore_ascii_case("level")),
            &MODEL,
            &INSTANCE,
        );
        let model_setters = canonical(model_side, MODEL_ALIASES, MODEL_PENDING, "model")?;
        let m = MODEL.validate(&model_setters, &card.location)?;
        let (initial, setters) = crate::devices::initial::split(&instance.parameters, &["ic"])?;
        let instance_setters = canonical(&setters, &[], INSTANCE_PENDING, "instance")?;
        let i = crate::devices::schema::with_defaults(
            &INSTANCE,
            &model_defaults,
            instance_setters.iter(),
            &instance.location,
        )?;
        let input = DiodeInput::prepare(
            &instance.name,
            &card.name,
            &instance.location,
            m,
            &i,
            initial,
            context,
        )?;
        nodes.transaction(|nodes| {
            let external = nodes.intern_array(&instance.nodes)?;
            Ok(Box::new(input.bind(nodes, external)?) as Box<dyn Device>)
        })
    }
}

/// Complete validated device data without terminal IDs. Both literal diode
/// cards and numeric URC-generated diodes use the same thermal validation
/// and binding, including the append-only internal-node setup.
struct DiodeInput {
    name: String,
    model: String,
    location: SourceLoc,
    parameters: DiodeParameters,
    derived: DerivedCache<DiodeThermalKey, Thermal>,
    initial: crate::devices::initial::InstanceInitial,
    written_model: ScalarValues,
}

impl DiodeInput {
    fn prepare(
        name: &str,
        model: &str,
        location: &SourceLoc,
        m: ScalarValues,
        i: &ScalarValues,
        initial: crate::devices::initial::InstanceInitial,
        context: &ModelContext,
    ) -> SpiceResult<Self> {
        let parameters = DiodeParameters::new(&m, i, location)?;
        let thermal = parameters.thermal(context)?;
        thermal.point(0., context.gmin)?;
        Ok(Self {
            name: name.to_owned(),
            model: model.to_owned(),
            location: location.clone(),
            parameters,
            derived: DerivedCache::from_value(DiodeThermalKey::context(context), thermal),
            initial,
            written_model: m,
        })
    }

    /// The caller owns the append-only topology transaction.
    fn bind(self, nodes: &mut NodeAppend<'_>, external: [NodeId; 2]) -> SpiceResult<Diode> {
        let mut terminals = external.to_vec();
        let positive = if self.parameters.model.rs.is_some_and(|rs| rs > 0.) {
            // diosetup.c: CKTmkVolt(ckt, &tmp, DIOname, "internal").
            let name = format!("{}#internal", self.name);
            if nodes.get(&name).is_some() {
                return Err(SpiceError::circuit("diode internal-node name collision"));
            }
            let prime = nodes.intern(&name)?;
            nodes.set_kind(prime, NodeKind::Internal)?;
            terminals.push(prime);
            prime
        } else {
            external[0]
        };
        let recovery_node = if self.parameters.model.vp > 0. && self.parameters.model.tt > 0. {
            let name = format!("{}#qp", self.name);
            if nodes.get(&name).is_some() {
                return Err(SpiceError::circuit("diode recovery-node name collision"));
            }
            let qp = nodes.intern(&name)?;
            nodes.set_kind(qp, NodeKind::Internal)?;
            terminals.push(qp);
            Some(recovery::RecoveryBinding::new(qp, &self.parameters.model))
        } else {
            None
        };
        Ok(Diode {
            name: self.name,
            model: self.model,
            terminals,
            junction: [positive, external[1]],
            recovery_node,
            parameters: self.parameters,
            derived: self.derived,
            initial: self.initial,
            location: self.location,
            written_model: self.written_model,
            lenient: false,
        })
    }
}

/// Number of state slots of a diode, including generated URC shunts.
pub(super) const DIODE_STATE_COUNT: usize = 3;

/// The shared generated URC diode model. Validate its numeric setters once;
/// each section supplies just its own area, with the same sensitivity
/// provenance and thermal checks as an ordinary diode card.
pub(super) struct UrcDiodeModel {
    name: String,
    values: ScalarValues,
}

impl UrcDiodeModel {
    pub(super) fn new(
        name: String,
        capacitance: Real,
        resistance: Real,
        current: Real,
        location: &SourceLoc,
    ) -> SpiceResult<Self> {
        let values = MODEL.validate_numeric(
            [
                ("cjo", capacitance, location),
                ("rs", resistance, location),
                ("is", current, location),
            ],
            location,
        )?;
        Ok(Self { name, values })
    }

    pub(super) fn bind_section(
        &self,
        name: &str,
        area: Real,
        location: &SourceLoc,
        nodes: &mut NodeAppend<'_>,
        terminals: [NodeId; 2],
        context: &ModelContext,
    ) -> SpiceResult<Box<dyn Device>> {
        let instance = INSTANCE.validate_numeric([("area", area, location)], location)?;
        let initial = crate::devices::initial::InstanceInitial {
            off: false,
            values: vec![None],
        };
        Ok(Box::new(
            DiodeInput::prepare(
                name,
                &self.name,
                location,
                self.values.clone(),
                &instance,
                initial,
                context,
            )?
            .bind(nodes, terminals)?,
        ))
    }
}
impl DiodeParameters {
    fn new(m: &ScalarValues, i: &ScalarValues, location: &SourceLoc) -> SpiceResult<Self> {
        let tlev = selector(m, "tlev", 2, location)?;
        let n = value(m, "n")?;
        let selectors = [tlev, selector(m, "tlevc", 1, location)?];
        let geometry = DiodeGeometry::setup(
            given(i, "area"),
            given(i, "pj"),
            given(i, "m"),
            [value(m, "area")?, value(m, "pj")?],
        );
        let instance = DiodeInstanceParameters {
            geometry,
            geometry_given: ["area", "pj", "m"].map(|key| given(i, key).is_some()),
            temperature: given(i, "temp").map(|v| v + 273.15),
            dtemp: given(i, "dtemp"),
            width: None,
            length: None,
        };
        let knees = ["ikf", "ikr", "ikp"].map(|key| given(m, key).unwrap_or(0.));
        let knee_given = knees.map(|k| k >= EPSMIN);
        let setup = DiodeSetup::new(given(m, "rs").unwrap_or(0.), knees, geometry);
        let p = Self::from_defaulted_model(
            |key| match key {
                "is" => Ok(value(m, key)?.max(EPSMIN)),
                "nbv" => Ok(given(m, key).unwrap_or(n)),
                "eg" => Ok(given(m, key).unwrap_or_else(|| default_eg(tlev))),
                _ => value(m, key),
            },
            |key| Ok(given(m, key)),
            selectors,
            instance,
            setup,
            knee_given,
            given(m, "tnom").map(|v| v + 273.15),
        )?;
        p.check(location)?;
        if p.instance.temperature.is_some() && i.get("dtemp").is_some() {
            return Err(SpiceError::parse(
                location.clone(),
                "diode has both temp= and dtemp= (C ignores dtemp); give one",
            ));
        }
        Ok(p)
    }

    /// Both the ordinary schema and the sensitivity records translate to
    /// the same defaulted nominal model, with saved setup quantities supplied
    /// separately. No inversion or geometry rescaling of those quantities.
    fn from_defaulted_model(
        read: impl Fn(&str) -> SpiceResult<Real>,
        given: impl Fn(&str) -> SpiceResult<Option<Real>>,
        selectors: [u8; 2],
        instance: DiodeInstanceParameters,
        setup: DiodeSetup,
        knee_given: [bool; 3],
        nominal: Option<Real>,
    ) -> SpiceResult<Self> {
        Ok(Self {
            model: DiodeModelParameters {
                // The setup stage applies the IS floor.
                is: read("is")?,
                jsw: given("jsw")?,
                n: read("n")?,
                ns: read("ns")?,
                ns_given: given("ns")?.is_some(),
                rs: given("rs")?,
                trs1: read("trs")?,
                trs2: read("trs2")?,
                tt: read("tt")?,
                vp: read("vp")?,
                qpscale: read("qpscale")?,
                ttt1: read("ttt1")?,
                ttt2: read("ttt2")?,
                cjo: read("cjo")?,
                vj: read("vj")?,
                grading: read("m")?,
                tm1: read("tm1")?,
                tm2: read("tm2")?,
                fc: read("fc")?,
                cjsw: read("cjsw")?,
                vjsw: read("vjsw")?,
                mjsw: read("mjsw")?,
                fcs: read("fcs")?,
                bv: given("bv")?,
                ibv: read("ibv")?,
                nbv: read("nbv")?,
                tcv: read("tcv")?,
                tlev: selectors[0],
                tlevc: selectors[1],
                eg: read("eg")?,
                gap1: read("gap1")?,
                gap2: read("gap2")?,
                xti: read("xti")?,
                cta: read("cta")?,
                ctp: read("ctp")?,
                tpb: read("tpb")?,
                tphp: read("tphp")?,
                isr: given("isr")?,
                nr: read("nr")?,
                knee_given,
                jtun: given("jtun")?,
                jtunsw: given("jtunsw")?,
                ntun: read("ntun")?,
                xtitun: read("xtitun")?,
                keg: read("keg")?,
                nominal,
                kf: read("kf")?,
                af: read("af")?,
            },
            instance,
            setup,
        })
    }

    /// The instance-independent consistency checks of [`Self::new`], shared
    /// with swept replacements ([`Diode::with_instance_parameter`]).
    fn check(&self, location: &SourceLoc) -> SpiceResult<()> {
        let p = self;
        let location = || location.clone();
        if p.model.grading >= 1. || p.model.fc >= 1. || p.model.mjsw >= 1. || p.model.fcs >= 1. {
            return Err(SpiceError::parse(
                location(),
                "diode requires 0 <= M,FC,MJSW,FCS < 1",
            ));
        }
        if !(p.instance.geometry.area * p.instance.geometry.multiplier).is_finite()
            || !(p.instance.geometry.perimeter * p.instance.geometry.multiplier).is_finite()
        {
            return Err(SpiceError::parse(
                location(),
                "diode needs finite area*m, pj*m",
            ));
        }
        // dioload.c evaluates the common-characteristic sidewall breakdown with
        // `vdsw`, which is only assigned for a separate sidewall (RSW), so C's
        // value there depends on stale solver state rather than on the junction.
        if p.model.bv.is_some()
            && !p.model.ns_given
            && p.model.jsw.unwrap_or(0.) * p.instance.geometry.perimeter > 0.
        {
            return Err(SpiceError::not_yet_ported(
                format!(
                    "{}: diode sidewall current (JSW*PJ > 0) sharing the bottom characteristic \
                     (NS not given) in breakdown (BV given)",
                    location()
                ),
                "src/spicelib/devices/dio/dioload.c (common-characteristic sidewall breakdown)",
            ));
        }
        // dioload.c/diotemp.c mix the temperature-adjusted (F1SW) and nominal
        // (F2SW, F3SW, charge below FCS*VJSW) sidewall grading under TM1/TM2.
        if p.model.cjsw * p.instance.geometry.perimeter > 0.
            && (p.model.tm1 != 0. || p.model.tm2 != 0.)
        {
            return Err(SpiceError::not_yet_ported(
                format!(
                    "{}: diode sidewall depletion charge with grading temperature \
                     coefficients TM1/TM2",
                    location()
                ),
                "src/spicelib/devices/dio/diotemp.c, dioload.c (DIOtGradingCoeffSW)",
            ));
        }
        Ok(())
    }

    /// Instance-scaled series conductance in C's order, after temperature correction.
    fn series_conductance(&self, dt: Real) -> SpiceResult<Real> {
        let conductance = self.setup.conductance
            * self.instance.geometry.area
            * self.instance.geometry.multiplier;
        if self.model.rs.is_some_and(|rs| rs != 0.) {
            let factor = temperature_polynomial(self.model.trs1, self.model.trs2, dt);
            if factor <= 0. || factor.is_nan() {
                return Err(SpiceError::Numerical {
                    context: "diode".into(),
                    message: "nonpositive temperature-adjusted series resistance".into(),
                });
            }
            // diosetup.c: DIOconductance = 1/RS; diotemp.c scales then divides.
            Ok(conductance / factor)
        } else {
            Ok(conductance)
        }
    }

    /// `diotemp.c::DIOtempUpdate` at this run's instance temperature.
    fn thermal(&self, context: &ModelContext) -> SpiceResult<Thermal> {
        // Low-level device callers can prepare without a circuit load.
        // Reject an invalid matching tolerance before publishing a derivation.
        if !(context.reltol.is_finite() && context.reltol > 0.) {
            return Err(SpiceError::circuit("invalid diode circuit RELTOL"));
        }
        let model = &self.model;
        let geometry = self.instance.geometry;
        // diotemp.c::DIOtemp: TEMP (dioparam.c adds CONSTCtoK), else the
        // circuit temperature in Kelvin (CKTtemp) plus DTEMP, in C's order.
        let t = self.instance.kelvin(context.temperature);
        let tn = self
            .model
            .nominal
            .unwrap_or(context.nominal_temperature + 273.15);
        if !t.is_finite() || t <= 0. || !tn.is_finite() || tn <= 0. {
            return Err(SpiceError::circuit("invalid diode temperature"));
        }
        let vt = K_OVER_Q * t;
        let vtnom = K_OVER_Q * tn;
        let dt = t - tn;
        let ln_ratio = (t / tn).ln();
        // diosetup.c/diotemp.c scale every instance quantity as
        // `value * AREA * M` (or `* PJ * M`), left to right.
        let scale = |v: Real| v * geometry.area * geometry.multiplier;
        let perimeter = |v: Real| v * geometry.perimeter * geometry.multiplier;

        let factor = temperature_polynomial(model.tm1, model.tm2, dt);
        let grading = model.grading * factor;

        // Band gap at T and TNOM: the fixed silicon law for TLEV 0/1, the
        // EG/GAP1/GAP2 law for TLEV 2.
        let (egfet, egfet1) = if model.tlev == 2 {
            (
                model.eg - model.gap1 * t * t / (t + model.gap2),
                model.eg - model.gap1 * tn * tn / (tn + model.gap2),
            )
        } else {
            (band_gap(t), band_gap(tn))
        };
        let fact2 = t / REFTEMP;
        let arg = -egfet / (2. * BOLTZMANN * t) + 1.115_087_7 / (BOLTZMANN * (REFTEMP + REFTEMP));
        let pbfact = potential_shift(vt, fact2, arg, CHARGE);
        let arg1 = -egfet1 / (BOLTZMANN * 2. * tn) + 1.115_087_7 / (2. * BOLTZMANN * REFTEMP);
        let fact1 = tn / REFTEMP;
        let pbfact1 = potential_shift(vtnom, fact1, arg1, CHARGE);
        let actual = PotentialTemperature::new(t, pbfact);
        let nominal = PotentialTemperature::new(tn, pbfact1);
        let (bottom, sidewall) = self.depletion_at_temperature(t, &actual, &nominal, grading);

        // Saturation currents: `n` is the emission coefficient, `xti` the
        // exponent and `k` the band-gap factor (KEG for tunnelling).
        // `k` is 1 for the plain laws (exact in every product) and KEG for
        // tunnelling. `base` is already instance-scaled (diotemp.c).
        let saturation = |base: Real, n: Real, xti: Real, k: Real| {
            let vte = n * vt;
            let temperature_exponent = xti / n * ln_ratio;
            let band_gap_exponent = if model.tlev == 2 {
                k * egfet1 / (n * vtnom) - k * egfet / vte
            } else {
                (t / tn - 1.) * k * model.eg / vte
            };
            let exponent = band_gap_exponent + temperature_exponent;
            base * exponent.exp()
        };
        let csat = saturation(scale(model.is), model.n, model.xti, 1.);
        let csatsw = saturation(perimeter(model.jsw.unwrap_or(0.)), model.ns, model.xti, 1.);
        let tunnel = |j: Option<Real>, s: &dyn Fn(Real) -> Real| {
            j.map(|j| saturation(s(j), model.ntun, model.xtitun, model.keg))
        };
        let breakdown = match model.bv {
            Some(bv) => Some(breakdown_voltage(
                {
                    let temperature_shift = model.tcv * dt;
                    if model.tlev == 0 {
                        bv - temperature_shift
                    } else {
                        bv * (1. - temperature_shift)
                    }
                },
                // Level 1: IBV scales with M only, not AREA.
                geometry.multiplier * model.ibv,
                csat + csatsw,
                vt,
                model.nbv,
                context.reltol,
            )?),
            None => None,
        };
        let conductance = self.series_conductance(dt)?;
        let thermal = Thermal {
            vt,
            n: model.n,
            ns: model.ns_given.then_some(model.ns),
            nbv: model.nbv,
            csat,
            csatsw: model.jsw.map(|_| csatsw),
            breakdown,
            recombination: model
                .isr
                .map(|isr| (saturation(scale(isr), model.nr, model.xti, 1.), model.nr)),
            tunnel: tunnel(model.jtun, &scale),
            tunnel_sidewall: tunnel(model.jtunsw, &perimeter),
            ntun: model.ntun,
            ikf: model.knee_given[0].then_some(self.setup.knees[0]),
            ikr: model.knee_given[1].then_some(self.setup.knees[1]),
            ikp: model.knee_given[2].then_some(self.setup.knees[2]),
            cjo: bottom.capacitance,
            bottom: bottom.law,
            cjsw: sidewall.capacitance,
            sidewall: sidewall.law,
            // diotemp.c: tt must stay positive.
            tt: model.tt * temperature_polynomial(model.ttt1, model.ttt2, dt).max(1e-3),
            conductance,
        };
        thermal.validate()?;
        Ok(thermal)
    }
    /// Prepare bottom and sidewall capacitances with their separate grading laws.
    fn depletion_at_temperature(
        &self,
        t: Real,
        actual: &PotentialTemperature,
        nominal: &PotentialTemperature,
        grading: Real,
    ) -> (PreparedDepletion, PreparedDepletion) {
        // Depletion capacitance/potential temperature laws (TLEVC 0 or 1).
        let depletion = |cap: Real, pot: Real, grading: Real, ct: Real, tp: Real| {
            if self.model.tlevc == 0 {
                let correction = actual.junction(nominal, pot);
                // The diode divides then multiplies, unlike MOS's two factors.
                let nominal_capacitance = cap / correction.nominal_denominator(grading);
                (
                    nominal_capacitance * correction.actual_factor(grading),
                    correction.potential,
                )
            } else {
                let shift = t - REFTEMP;
                let capacitance_scale = ct * shift;
                let potential_offset = tp * shift;
                (cap * (1. + capacitance_scale), pot - potential_offset)
            }
        };
        let (cjo, vj) = depletion(
            self.model.cjo * self.instance.geometry.area * self.instance.geometry.multiplier,
            self.model.vj,
            grading,
            self.model.cta,
            self.model.tpb,
        );
        let (cjsw, vjsw) = depletion(
            self.model.cjsw * self.instance.geometry.perimeter * self.instance.geometry.multiplier,
            self.model.vjsw,
            self.model.mjsw,
            self.model.ctp,
            self.model.tphp,
        );

        (
            PreparedDepletion {
                capacitance: cjo,
                law: DepletionLaw::new(vj, grading, self.model.fc),
            },
            PreparedDepletion {
                capacitance: cjsw,
                law: DepletionLaw::new(vjsw, self.model.mjsw, self.model.fcs),
            },
        )
    }
}

/// Temperature coefficient law for grading, transit time and series resistance.
/// Round each C term separately and keep the left-associative additions.
const fn temperature_polynomial(linear: Real, quadratic: Real, delta: Real) -> Real {
    let first_order = linear * delta;
    let second_order = quadratic * delta * delta;
    (1. + first_order) + second_order
}

/// `diotemp.c` breakdown matching: the knee voltage `xbv` at which the
/// reverse-breakdown exponential carries `cbv` at the temperature-adjusted
/// breakdown voltage `bv`, iterated with the resolved circuit RELTOL.
fn breakdown_voltage(
    bv: Real,
    cbv: Real,
    total: Real,
    vt: Real,
    nbv: Real,
    reltol: Real,
) -> SpiceResult<Real> {
    if cbv < total * bv / vt {
        // C keeps BV unmatched (its warning is TRACE-only).
        return Ok(bv);
    }
    let matched = |x: Real| {
        let exponential = ((bv - x) / (nbv * vt)).exp();
        total * (exponential - 1. + x / vt)
    };
    let tolerance = reltol * cbv;
    let normalized_current = 1. + cbv / total;
    let initial_offset = nbv * vt * normalized_current.ln();
    let mut x = bv - initial_offset;
    for _ in 0..25 {
        let offset = nbv * vt * (cbv / total + 1. - x / vt).ln();
        x = bv - offset;
        if (matched(x) - cbv).abs() <= tolerance {
            break;
        }
    }
    // After 25 iterations C keeps the last iterate (its warning is TRACE-only).
    if !x.is_finite() {
        return Err(SpiceError::Numerical {
            context: "diode".into(),
            message: "nonfinite breakdown matching".into(),
        });
    }
    Ok(x)
}

/// Capacitance and the continuation law prepared at the same temperature.
struct PreparedDepletion {
    capacitance: Real,
    law: DepletionLaw,
}

/// Temperature-adjusted, instance-scaled diode quantities (`diotemp.c` outputs).
#[derive(Debug, Clone, Copy)]
struct Thermal {
    vt: Real,
    n: Real,
    /// Sidewall emission coefficient when NS is given (own characteristic).
    ns: Option<Real>,
    nbv: Real,
    csat: Real,
    /// Sidewall saturation current when JSW is given.
    csatsw: Option<Real>,
    /// Matched breakdown voltage when BV is given.
    breakdown: Option<Real>,
    /// Recombination saturation current and NR when ISR is given.
    recombination: Option<(Real, Real)>,
    tunnel: Option<Real>,
    tunnel_sidewall: Option<Real>,
    ntun: Real,
    ikf: Option<Real>,
    ikr: Option<Real>,
    ikp: Option<Real>,
    cjo: Real,
    bottom: DepletionLaw,
    cjsw: Real,
    sidewall: DepletionLaw,
    tt: Real,
    /// Series conductance AREA*M/RS at temperature (0 without RS).
    conductance: Real,
}
/// One diode evaluation. The Newton current/conductance and charge/capacitance
/// are exact derivatives; `ac_*` are C's small-signal values (they differ only
/// with ISR, see [`Thermal::point`]).
#[derive(Debug, Clone, Copy)]
struct DiodePoint {
    junction: JunctionPoint,
    ac_conductance: Real,
    ac_capacitance: Real,
    depletion_charge: Real,
    depletion_capacitance: Real,
}
impl Thermal {
    /// `dioload.c`'s junction voltage for this load: `tVcrit` in
    /// `MODEINITJCT`, otherwise `DEVpnjlim` against the previous value, in
    /// the breakdown frame (`-(vd + BV)` with `nbv * vt`) when BV is given
    /// and `vd < min(0, -BV + 10 nbv vt)`. `tVcrit` uses the total (bottom
    /// plus sidewall) saturation current, as `diotemp.c` does without RSW.
    ///
    /// `start` is C's `uic` initial voltage (`DIOinitCond`) and `off` the
    /// instance flag: the `uic` initial load evaluates at `start`, and an
    /// `off` diode is held at zero in `MODEINITJCT` and `MODEINITFIX`.
    fn limit(
        &self,
        limiter: &mut Limiter,
        states: &crate::devices::DeviceState<'_>,
        raw: Real,
        start: Real,
        off: bool,
    ) -> Real {
        let vte = self.n * self.vt;
        let vcrit = limiting::critical_voltage(vte, self.csat + self.csatsw.unwrap_or(0.));
        if limiter.mode() == Linearization::InitialConditions {
            return start;
        }
        if limiter.holds_off(states, off) {
            return 0.;
        }
        if limiter.mode() == Linearization::Initial {
            return vcrit;
        }
        // dioload.c MODEINITTRAN: vd = *(CKTstate1 + DIOvoltage), unlimited.
        if limiter.mode() == Linearization::InitialTransient {
            return states.accepted(1, DIODE_VOLTAGE_SLOT).unwrap_or(raw);
        }
        let previous = limiter.previous(states, DIODE_VOLTAGE_SLOT);
        let vtebrk = self.nbv * self.vt;
        let breakdown_margin = 10. * vtebrk;
        match self.breakdown {
            Some(bv) if raw < (-bv + breakdown_margin).min(0.) => {
                let reflected = limiter.pn_junction(
                    -(raw + bv),
                    previous.map(|old| -(old + bv)),
                    vtebrk,
                    vcrit,
                );
                -(reflected + bv)
            }
            _ => limiter.pn_junction(raw, previous, vte, vcrit),
        }
    }

    fn validate(&self) -> SpiceResult<()> {
        let finite = [
            self.csat,
            self.csatsw.unwrap_or(0.),
            self.breakdown.unwrap_or(0.),
            self.recombination.map_or(0., |r| r.0),
            self.tunnel.unwrap_or(0.),
            self.tunnel_sidewall.unwrap_or(0.),
            self.cjo,
            self.bottom.potential,
            self.bottom.grading,
            self.cjsw,
            self.sidewall.potential,
            self.tt,
            self.conductance,
        ]
        .iter()
        .all(|v| v.is_finite());
        // VJ/M enter the bottom charge and the recombination generation
        // factor; VJSW only the sidewall charge.
        let bottom = self.cjo != 0. || self.recombination.is_some();
        if !finite
            || self.cjo < 0.
            || self.cjsw < 0.
            || (bottom
                && (self.bottom.potential <= 0. || !(0. ..1.).contains(&self.bottom.grading)))
            || (self.cjsw != 0. && self.sidewall.potential <= 0.)
        {
            return Err(SpiceError::Numerical {
                context: "diode".into(),
                message: "temperature-adjusted parameters out of domain \
                          (VJ, VJSW > 0, CJO, CJSW >= 0, 0 <= M < 1)"
                    .into(),
            });
        }
        Ok(())
    }

    /// `dioload.c` currents and equilibrium charges at junction voltage `vd`
    /// (no RSW or self-heating). Active recovery uses the separate depletion fields.
    fn point(&self, vd: Real, gmin: Real) -> SpiceResult<DiodePoint> {
        let point = self.point_unchecked(vd, gmin)?;
        point.junction.validate()?;
        if !point.ac_conductance.is_finite() || !point.ac_capacitance.is_finite() {
            return Err(SpiceError::Numerical {
                context: "junction".into(),
                message: "nonfinite small-signal junction values".into(),
            });
        }
        Ok(point)
    }

    /// [`Self::point`] without the final finiteness checks, for a `.sens`
    /// stand-in that must carry C's NaN.
    fn point_unchecked(&self, vd: Real, gmin: Real) -> SpiceResult<DiodePoint> {
        let vte = self.n * self.vt;
        let vtebrk = self.nbv * self.vt;
        let breakdown = self.breakdown.map(|bv| (bv, vtebrk));
        let forward = vd >= -3. * vte;

        let (mut cdsw, mut gdsw) = (0., 0.);
        if let Some(csatsw) = self.csatsw {
            (cdsw, gdsw) = match self.ns {
                Some(ns) => characteristic(vd, ns * self.vt, csatsw, breakdown)?,
                // Common characteristic. The breakdown branch is rejected at
                // instantiation for JSW*PJ > 0, so csatsw is zero there.
                None if forward || breakdown.is_none_or(|(bv, _)| vd >= -bv) => {
                    characteristic(vd, vte, csatsw, None)?
                }
                None => (0., 0.),
            };
        }
        let (mut cdb, mut gdb) = characteristic(vd, vte, self.csat, breakdown)?;
        // The exact derivative (Newton) and C's conductance (small signal,
        // `.tf`/`.sens`) differ only with ISR.
        let mut ac_gdb = gdb;
        if let Some((crec, nr)) = self.recombination {
            let vterec = nr * self.vt;
            // dioload.c: t1 = pow(1 - vd/VJ, 2) + 0.005, gen_fac = pow(t1, M/2).
            let generation = |v: Real| {
                // C uses pow(x, 2) here, not a product or powi.
                #[expect(
                    clippy::suboptimal_flops,
                    reason = "retain the reference pow operation"
                )]
                let squared = (1. - v / self.bottom.potential).powf(2.);
                let t1 = squared + 0.005;
                (t1, t1.powf(self.bottom.grading / 2.))
            };
            if forward {
                let evd = (vd / vterec).exp();
                let current = crec * (evd - 1.);
                let conductance = crec * evd / vterec;
                let (t1, factor) = generation(vd);
                let derivative = -self.bottom.grading
                    * (1. - vd / self.bottom.potential)
                    * t1.powf(self.bottom.grading / 2. - 1.);
                let generation_current = current * factor;
                cdb += generation_current;
                // d/dv of the generation factor carries 1/VJ.
                let generation_conductance = conductance * factor;
                let newton_correction = current * derivative / self.bottom.potential;
                gdb += generation_conductance + newton_correction;
                // dioload.c instead multiplies the already-scaled current and
                // omits 1/VJ; its root is unaffected, its AC conductance is.
                let ac_correction = generation_current * derivative;
                ac_gdb += generation_conductance + ac_correction;
            } else {
                // Reverse and breakdown: the constant value at -3*N*Vt,
                // `cdb = -csat*(..) + gen_fac*cdb_rec` (dioload.c).
                let (_, factor) = generation(-3. * vte);
                let exponential = (-3. * vte / vterec).exp();
                let reverse_current = factor * (crec * (exponential - 1.));
                cdb += reverse_current;
            }
        }
        let vtetun = self.ntun * self.vt;
        if let Some(tunnel) = self.tunnel_sidewall {
            let evd = (-vd / vtetun).exp();
            let reverse_current = tunnel * (evd - 1.);
            cdsw -= reverse_current;
            gdsw += tunnel * evd / vtetun;
        }
        if let Some(tunnel) = self.tunnel {
            let evd = (-vd / vtetun).exp();
            let reverse_current = tunnel * (evd - 1.);
            cdb -= reverse_current;
            let g = tunnel * evd / vtetun;
            gdb += g;
            ac_gdb += g;
        }
        let knee = if forward {
            self.ikf.filter(|_| cdb > 1e-18)
        } else {
            self.ikr.filter(|_| cdb < -1e-18)
        };
        if let Some(k) = knee {
            let current = cdb;
            (cdb, gdb) = high_injection(current, gdb, k, forward);
            ac_gdb = high_injection(current, ac_gdb, k, forward).1;
        }
        if let Some(k) = self.ikp.filter(|_| cdsw > 1e-18) {
            (cdsw, gdsw) = high_injection(cdsw, gdsw, k, true);
        }
        let shunt_current = gmin * vd;
        let current = cdb + cdsw + shunt_current;
        let conductance = gdb + gdsw + gmin;
        let ac_conductance = ac_gdb + gdsw + gmin;
        let (charge, capacitance) = self.bottom.charge(vd, self.cjo, None);
        let (charge_sw, capacitance_sw) = self.sidewall.charge(vd, self.cjsw, None);
        // dioload.c, in its order: diffusion (TT times the current including
        // gmin), bottom depletion, sidewall depletion.
        let diffusion_charge = self.tt * current;
        let diffusion_capacitance = self.tt * conductance;
        let ac_diffusion_capacitance = self.tt * ac_conductance;
        let point = DiodePoint {
            junction: JunctionPoint {
                current,
                conductance,
                charge: diffusion_charge + charge + charge_sw,
                capacitance: diffusion_capacitance + capacitance + capacitance_sw,
            },
            ac_conductance,
            ac_capacitance: ac_diffusion_capacitance + capacitance + capacitance_sw,
            depletion_charge: charge + charge_sw,
            depletion_capacitance: capacitance + capacitance_sw,
        };
        Ok(point)
    }
}

/// One exponential characteristic: forward exponential, C's cubic reverse
/// continuation below `-3*vte` and, with `breakdown = (xbv, n_bv*Vt)`, the
/// reverse-breakdown exponential below `-xbv` (`dioload.c`).
fn characteristic(
    v: Real,
    vte: Real,
    sat: Real,
    breakdown: Option<(Real, Real)>,
) -> SpiceResult<(Real, Real)> {
    match breakdown {
        Some((bv, vtebrk)) if v < -3. * vte && v < -bv => {
            let evrev = (-(bv + v) / vtebrk).exp();
            let result = (-sat * evrev, sat * evrev / vtebrk);
            if !result.0.is_finite() || !result.1.is_finite() {
                return Err(SpiceError::Numerical {
                    context: "junction".into(),
                    message: "reverse-breakdown exponential overflow".into(),
                });
            }
            Ok(result)
        }
        _ if sat == 0. => Ok((0., 0.)),
        _ => junction_current(v, vte, sat, ReverseBoundary::Exponential),
    }
}

/// `dioload.c` high-injection knee: `i/(1+sqrt(i/k))`, with `k` the positive
/// IKF/IKP (`forward`) or IKR (reverse, `i < 0`). Returns the limited current
/// and conductance in C's arithmetic.
fn high_injection(cdb: Real, gdb: Real, knee: Real, forward: bool) -> (Real, Real) {
    let knee_ratio = if forward { cdb / knee } else { cdb / (-knee) };
    let sqrt_ikx = knee_ratio.sqrt();
    let scaled_conductance = (1. + sqrt_ikx) * gdb;
    let correction = cdb * gdb / (2. * sqrt_ikx * knee);
    let twice_root = 2. * sqrt_ikx;
    // Preserve C's separate signed expressions (including signed zero).
    let (numerator, denominator) = if forward {
        (
            scaled_conductance - correction,
            1. + twice_root + cdb / knee,
        )
    } else {
        (
            scaled_conductance + correction,
            1. + twice_root - cdb / knee,
        )
    };
    (cdb / (1. + sqrt_ikx), numerator / denominator)
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct JunctionPoint {
    pub current: Real,
    pub conductance: Real,
    pub charge: Real,
    pub capacitance: Real,
}
impl JunctionPoint {
    pub(crate) fn validate(self) -> SpiceResult<()> {
        if [
            self.current,
            self.conductance,
            self.charge,
            self.capacitance,
        ]
        .iter()
        .any(|v| !v.is_finite())
            || self.capacitance < 0.
        {
            return Err(SpiceError::Numerical {
                context: "junction".into(),
                message: "nonfinite/out-of-domain junction equations".into(),
            });
        }
        Ok(())
    }
}
/// C's cubic reverse-bias continuation argument, with multiplication
/// grouped exactly as in dioload, bjtload, jfetload, mesload and mos3load.
pub(super) fn cubic_reverse_argument(v: Real, vt: Real) -> Real {
    let arg = 3. * vt / (v * std::f64::consts::E);
    arg * arg * arg
}

/// Current and conductance without gmin in the cubic reverse region.
pub(super) fn cubic_reverse_junction(v: Real, vt: Real, is: Real) -> (Real, Real) {
    let arg = cubic_reverse_argument(v, vt);
    (-is * (1. + arg), is * 3. * arg / v)
}

/// Current and conductance without gmin or an exponential clamp.
pub(super) fn exponential_junction(v: Real, vt: Real, is: Real) -> (Real, Real) {
    exponential_junction_at_argument(v / vt, vt, is)
}

/// Evaluate an exponential junction after the caller has applied its
/// model-specific reverse-region policy and optional exponent clamp.
pub(super) fn exponential_junction_at_argument(argument: Real, vt: Real, is: Real) -> (Real, Real) {
    let exp = argument.exp();
    if exp.is_finite() {
        // Around equilibrium the current is O(v); subtracting one from
        // exp(v/vt) would round it to zero long before it underflows.
        let current = if argument.abs() < 0.5 {
            is * argument.exp_m1()
        } else {
            is * (exp - 1.)
        };
        (current, is * exp / vt)
    } else if is > 0. && argument.is_finite() {
        // A small saturation current can cancel the exponential's
        // intermediate overflow. Delay multiplying the exponentials.
        let factor = (argument * 0.5).exp();
        let scaled = if factor.is_finite() {
            (is * factor) * factor
        } else {
            (is.ln() + argument).exp()
        };
        (scaled - is, scaled / vt)
    } else {
        (is * (exp - 1.), is * exp / vt)
    }
}

/// Which region owns the exact -3 Vt boundary in the reference load.
#[derive(Clone, Copy)]
pub(super) enum ReverseBoundary {
    Exponential,
    Cubic,
}

/// Checked junction current for diode and MESFET loads. The reference
/// models differ in their comparison at the reverse continuation boundary.
pub(super) fn junction_current(
    v: Real,
    vt: Real,
    is: Real,
    boundary: ReverseBoundary,
) -> SpiceResult<(Real, Real)> {
    let forward = match boundary {
        ReverseBoundary::Exponential => v >= -3. * vt,
        ReverseBoundary::Cubic => v > -3. * vt,
    };
    let result = if forward {
        exponential_junction(v, vt, is)
    } else {
        cubic_reverse_junction(v, vt, is)
    };
    if [is, vt, result.0, result.1].iter().any(|v| !v.is_finite()) || vt <= 0. || is <= 0. {
        return Err(SpiceError::Numerical {
            context: "junction".into(),
            message: "junction exponential/temperature overflow".into(),
        });
    }
    Ok(result)
}
/// Integral of the depletion capacitance, with its logarithmic limit
/// when the grading exponent is exactly one.
fn depletion_integral(logarithm: Real, grading: Real) -> Real {
    let exponent = 1. - grading;
    if exponent == 0. {
        -logarithm
    } else {
        -(exponent * logarithm).exp_m1() / exponent
    }
}

/// Reverse depletion charge and capacitance for the logarithmic grading law
/// in `dioload.c` and `bjtload.c` (not the MOS or JFET square-root shortcuts).
#[expect(
    clippy::float_cmp,
    reason = "specific grading exponent selects the equivalent exact square-root law"
)]
pub(super) fn reverse_depletion(v: Real, cz: Real, pot: Real, m: Real) -> (Real, Real) {
    if cz == 0. {
        return (cz * v, cz);
    }
    if m == 0.5 {
        let root = (1. - v / pot).sqrt();
        return (cz * (v / 1_f64.midpoint(root)), cz / root);
    }
    let logarithm = (-v / pot).ln_1p();
    (
        pot * cz * depletion_integral(logarithm, m),
        cz * (-m * logarithm).exp(),
    )
}

/// Temperature-adjusted depletion continuation shared by diode and BJT.
/// C computes these coefficients in diotemp/bjttemp, before evaluating load.
#[derive(Debug, Clone, Copy)]
pub(super) struct DepletionLaw {
    pub(super) potential: Real,
    pub(super) grading: Real,
    pub(super) threshold: Real,
    pub(super) f1: Real,
    pub(super) f2: Real,
    /// The normalized capacitance at the start of the forward continuation.
    pub(super) edge: Real,
}

impl DepletionLaw {
    pub(super) fn new(potential: Real, grading: Real, fc: Real) -> Self {
        let xfc = (-fc).ln_1p();
        Self {
            potential,
            grading,
            threshold: fc * potential,
            f1: potential * depletion_integral(xfc, grading),
            f2: ((1. + grading) * xfc).exp(),
            edge: 1. - fc,
        }
    }

    /// C's BJT BE/BC charge adds diffusion first, including before the
    /// second forward depletion term. BX and diode use depletion alone.
    /// Absence is distinct from adding +0 (signed zero and rounding).
    pub(super) fn charge(
        &self,
        v: Real,
        cz: Real,
        diffusion: Option<(Real, Real)>,
    ) -> (Real, Real) {
        let add_diffusion = |charge, capacitance| match diffusion {
            Some((diffusion_charge, diffusion_capacitance)) => (
                diffusion_charge + charge,
                diffusion_capacitance + capacitance,
            ),
            None => (charge, capacitance),
        };
        let potential = self.potential;
        let grading = self.grading;
        let threshold = self.threshold;
        if v < threshold {
            let (charge, capacitance) = reverse_depletion(v, cz, potential, grading);
            add_diffusion(charge, capacitance)
        } else {
            let czf2 = cz / self.f2;
            let delta = v - threshold;
            let slope = grading / potential;
            let edge = self.edge;
            let relative_capacitance = slope.mul_add(delta, edge);
            let (charge, capacitance) = add_diffusion(cz * self.f1, czf2 * relative_capacitance);
            // Integrate the linear capacitance from the forward threshold
            // instead of subtracting squares of large absolute voltages.
            let forward_charge = (czf2 * delta) * edge.midpoint(relative_capacitance);
            (charge + forward_charge, capacitance)
        }
    }
}

/// `NIintegrate` for a junction charge in `slot` (its derivative in
/// `slot + 1`): `(geq, ccap)` with `geq = ag0 cap` and `ccap` the charge
/// derivative, stored in the derivative slot.
#[inline]
pub(crate) fn junction_companion(
    context: &mut StampContext<'_>,
    coefficients: &crate::maths::integrator::Coefficients,
    p: JunctionPoint,
    slot: usize,
) -> SpiceResult<(Real, Real)> {
    let companion = charge::ChargeBranch::new(slot, p.charge, p.capacitance)
        .integrate(&mut context.states, coefficients)?;
    Ok((companion.conductance, companion.derivative))
}

impl Device for Diode {
    fn validate_dc_inputs(&self, context: &crate::devices::ModelContext) -> SpiceResult<()> {
        self.thermal(context).map(|_| ())
    }

    /// `diosetup.c` (without sidewall resistance or self-heating, which the
    /// port rejects); `posPrime` is the external anode when `RS = 0`.
    fn setup_matrix(
        &self,
        setup: &mut crate::devices::setup::MatrixSetup<'_>,
    ) -> SpiceResult<bool> {
        let [pos, neg] = [self.terminals[0], self.terminals[1]];
        let prime = self.junction[0];
        for (row, col) in [
            (pos, prime),
            (neg, prime),
            (prime, pos),
            (prime, neg),
            (pos, pos),
            (neg, neg),
            (prime, prime),
        ] {
            setup.nodes(row, col);
        }
        if let Some(binding) = self.recovery_node {
            let qp = binding.node;
            for (row, col) in [(qp, qp), (qp, prime), (qp, neg), (prime, qp), (neg, qp)] {
                setup.nodes(row, col);
            }
        }
        Ok(true)
    }
    fn observation_parameter(
        &self,
        key: &str,
        context: &ModelContext,
    ) -> SpiceResult<Option<Real>> {
        let p = &self.parameters;
        Ok(match key {
            "area" => Some(p.instance.geometry.area),
            "pj" | "perim" => Some(p.instance.geometry.perimeter),
            "m" => Some(p.instance.geometry.multiplier),
            "temp" => Some(p.instance.kelvin(context.temperature) - 273.15),
            "dtemp" => Some(p.instance.dtemp.unwrap_or(0.)),
            _ => None,
        })
    }
    fn observation_operating(
        &self,
        key: &str,
        context: &ModelContext,
        voltage: &dyn Fn(NodeId) -> Real,
        transient: bool,
    ) -> SpiceResult<Option<Real>> {
        // dioask.c reads the stored DIOcurrent and DIOconduct of the
        // completed load. In a real DC analysis, these are precisely the
        // temperature-adjusted junction evaluation, including gmin.
        // In transient they additionally contain the charge companion,
        // which cannot be reconstructed from instantaneous voltage alone.
        if !matches!(key, "id" | "c" | "vd" | "gd") {
            return Ok(None);
        }
        let vd = voltage(self.junction[0]) - voltage(self.junction[1]);
        if key == "vd" {
            return Ok(Some(vd));
        }
        if transient {
            return Ok(None);
        }
        let p = self.thermal(context)?.point(vd, context.gmin)?;
        Ok(match key {
            "id" | "c" => Some(p.junction.current),
            "gd" => Some(p.junction.conductance),
            _ => None,
        })
    }
    fn name(&self) -> &str {
        &self.name
    }
    fn designator(&self) -> char {
        'd'
    }
    fn terminals(&self) -> &[NodeId] {
        &self.terminals
    }
    fn external_terminals(&self) -> &[NodeId] {
        &self.terminals[..2]
    }
    fn constitutive_nodes(&self) -> &[NodeId] {
        self.recovery_node
            .as_ref()
            .map_or(&[], |binding| std::slice::from_ref(&binding.node))
    }
    fn is_nonlinear(&self) -> bool {
        true
    }
    fn has_start_settings(&self) -> bool {
        self.initial.off
    }
    fn state_count(&self) -> usize {
        // Charge, its derivative and the limited junction voltage.
        DIODE_STATE_COUNT + usize::from(self.recovery_node.is_some()) * 2
    }
    fn truncation_slot(&self) -> Option<usize> {
        Some(0)
    }
    fn truncation_slots(&self) -> Vec<usize> {
        if self.recovery_node.is_some() {
            vec![DIODE_CHARGE_SLOT, recovery::CHARGE_SLOT]
        } else {
            vec![DIODE_CHARGE_SLOT]
        }
    }
    /// `dio.c` `DIOpTable`: AREA, PJ (alias PERIM), M, TEMP and DTEMP, which
    /// `diotemp.c` re-derives completely (`dctrcurv.c` `DCTsetInstParam`).
    fn instance_parameter(&self, keyword: &str) -> Option<&'static str> {
        match keyword.to_ascii_lowercase().as_str() {
            "ic" => Some("ic"),
            "w" => Some("w"),
            "l" => Some("l"),
            "area" => Some("area"),
            "pj" | "perim" => Some("pj"),
            "m" => Some("m"),
            "temp" => Some("temp"),
            "dtemp" => Some("dtemp"),
            _ => None,
        }
    }
    /// `DIOparam` then `DIOtemp`: the swept setter replaces the instance value
    /// (AREA/PJ outrank the model's), every temperature-dependent quantity is
    /// re-derived from it, and the instance schema domains still apply. A
    /// swept DTEMP on a diode with an instance TEMP is rejected: C silently
    /// ignores DTEMP there (`diotemp.c`), as the card-level rule says.
    fn with_instance_parameter(
        &self,
        parameter: &str,
        value: Real,
        context: &ModelContext,
    ) -> SpiceResult<Box<dyn Device>> {
        let mut initial = self.initial.clone();
        if parameter == "ic" {
            crate::devices::sweep::check_swept(&self.name, parameter, value, true, "finite")?;
            initial.values[0] = Some(value);
        }
        let mut p = self.parameters;
        let domain = |ok: bool, what: &str| {
            if ok && value.is_finite() {
                Ok(())
            } else {
                Err(SpiceError::circuit(format!(
                    "{}: swept {parameter}={value} must be {what}",
                    self.name
                )))
            }
        };
        match parameter {
            "ic" => {}
            "w" => {
                domain(value >= 0., "nonnegative")?;
                p.instance.width = Some(value);
            }
            "l" => {
                domain(value >= 0., "nonnegative")?;
                p.instance.length = Some(value);
            }
            "area" => {
                domain(value > 0., "positive")?;
                p.instance.geometry.area = value;
                p.instance.geometry_given[0] = true;
            }
            "pj" => {
                domain(value >= 0., "nonnegative")?;
                p.instance.geometry.perimeter = value;
                p.instance.geometry_given[1] = true;
            }
            "m" => {
                domain(value > 0., "positive")?;
                p.instance.geometry.multiplier = value;
                p.instance.geometry_given[2] = true;
            }
            "temp" => {
                domain(value + 273.15 > 0., "above absolute zero")?;
                p.instance.temperature = Some(value + 273.15);
            }
            "dtemp" if p.instance.temperature.is_some() => {
                return Err(SpiceError::Unsupported {
                    feature: format!(
                        "{}: swept dtemp on a diode with an instance temp (C ignores dtemp)",
                        self.name
                    ),
                    location: Some(self.location.clone()),
                });
            }
            "dtemp" => {
                domain(true, "finite")?;
                p.instance.dtemp = Some(value);
            }
            _ => {
                return Err(SpiceError::circuit(format!(
                    "{}: diode parameter {parameter} cannot be swept",
                    self.name
                )));
            }
        }
        p.check(&self.location)?;
        let thermal = p.thermal(context)?;
        thermal.point(0., context.gmin)?;
        Ok(Box::new(Self {
            name: self.name.clone(),
            model: self.model.clone(),
            terminals: self.terminals.clone(),
            junction: self.junction,
            recovery_node: self.recovery_node,
            parameters: p,
            derived: DerivedCache::from_value(DiodeThermalKey::context(context), thermal),
            initial,
            location: self.location.clone(),
            written_model: self.written_model.clone(),
            lenient: false,
        }))
    }
    /// `.sens`: C's diode records ([`sens`]).
    fn sensitivity(
        &self,
        context: &ModelContext,
    ) -> SpiceResult<Box<dyn crate::devices::sensitivity::DeviceSensitivity + '_>> {
        Ok(Box::new(sens::DiodeSensitivity::new(self, context)))
    }
    fn stamp(&self, context: &mut StampContext<'_>) -> SpiceResult<()> {
        let sample = self.sample_load(
            &context.model_context(),
            context.unknowns,
            context.solution,
            &context.states,
            context.mode,
            context.integration,
        )?;
        let evaluation = sample.evaluate(context.gmin)?;
        self.commit_load(context, &evaluation)
    }
    fn convergence_test(
        &self,
        context: &crate::devices::traits::ConvergenceContext<'_>,
    ) -> SpiceResult<bool> {
        if self.recovery_node.is_none() {
            return Ok(false);
        }
        let thermal = self.thermal(context.model_context)?;
        let voltage = context.voltage(self.junction[0]) - context.voltage(self.junction[1]);
        let current = thermal
            .point(voltage, context.model_context.gmin)?
            .junction
            .current;
        self.recovery(&thermal).map_or(Ok(false), |recovery| {
            recovery.not_converged(context, current)
        })
    }
    fn physical_charge_histories(
        &self,
        visit: &mut dyn FnMut(crate::devices::traits::PhysicalChargeHistory) -> SpiceResult<()>,
    ) -> SpiceResult<()> {
        use crate::devices::traits::PhysicalChargeHistory;
        visit(PhysicalChargeHistory {
            slot: DIODE_CHARGE_SLOT,
            nodes: self.junction,
            scale: 1.,
        })?;
        if self.recovery_node.is_some() {
            visit(PhysicalChargeHistory {
                slot: recovery::CHARGE_SLOT,
                nodes: self.junction,
                scale: 1. - self.parameters.model.vp,
            })?;
        }
        Ok(())
    }

    fn assemble_nonlinear_taylor(
        &self,
        context: &mut crate::devices::physical_dae::TaylorContext<'_>,
    ) -> SpiceResult<()> {
        self.stamp_taylor(context)
    }

    fn assemble_nonlinear_charge(
        &self,
        context: &mut crate::devices::traits::ChargeContext<'_>,
    ) -> SpiceResult<()> {
        let voltage = context.voltage(self.junction[0]) - context.voltage(self.junction[1]);
        let thermal = self.thermal(context.model)?;
        let point = thermal.point(voltage, context.model.gmin)?;
        if let Some(recovery) = self.recovery(&thermal) {
            context.branch(
                self.junction,
                point.depletion_charge,
                point.depletion_capacitance,
            )?;
            recovery.assemble_charge(context, self.junction)
        } else {
            context.branch(
                self.junction,
                point.junction.charge,
                point.junction.capacitance,
            )
        }
    }

    /// `dioacld.c`: series conductance, C's small-signal junction conductance
    /// and `j*omega` times its small-signal capacitance at the bias point.
    fn assemble_small_signal(
        &self,
        context: &mut LinearContext<'_>,
        bias: &Vector,
    ) -> SpiceResult<()> {
        let v = |node| {
            context
                .unknowns
                .node_row(node)
                .and_then(|r| bias.get(r))
                .unwrap_or(0.)
        };
        let thermal = self.thermal(context.model_context)?;
        let p = thermal.point(
            v(self.junction[0]) - v(self.junction[1]),
            context.model_context.gmin,
        )?;
        if self.junction[0] != self.terminals[0] {
            context.nodal(
                [self.terminals[0], self.junction[0]],
                thermal.conductance,
                false,
            )?;
        }
        context.nodal(self.junction, p.ac_conductance, false)?;
        if let Some(recovery) = self.recovery(&thermal) {
            context.nodal(self.junction, p.depletion_capacitance, true)?;
            recovery.assemble_linear(context, self.junction, p.ac_conductance)
        } else {
            context.nodal(self.junction, p.ac_capacitance, true)
        }
    }

    /// Pole-zero load: C `diopzld.c` equals the AC load with `s` for `j omega`.
    fn assemble_pole_zero(
        &self,
        context: &mut crate::devices::linear::LinearContext<'_>,
        bias: &crate::maths::Vector,
    ) -> crate::primitives::SpiceResult<()> {
        self.assemble_small_signal(context, bias)
    }

    /// `diodset.c`/`diodisto.c`: the junction current's and charge's second-
    /// and third-order Taylor coefficients in the junction voltage, from C's
    /// own simplified distortion model rather than `dioload.c`: the ideal
    /// exponential of the total (bottom plus sidewall) saturation current,
    /// SPICE3's cubic reverse law below `-3 N Vt`, a breakdown exponential in
    /// `Vt` (not `NBV Vt`) below `-BV`, no ISR/IKF/IKR/tunnelling/gmin terms,
    /// the transit-time diffusion charge, and the depletion charges graded
    /// against the model's **unadjusted** VJ/VJSW below the temperature-
    /// adjusted `FC*VJ` (`DIOtDepCap`, which C applies to the sidewall too).
    /// The charge's `j omega` current is folded into the junction current's
    /// term ([`crate::devices::distortion::DistortionTerm::with_charge`]),
    /// and omitted when its second-order coefficient is zero, as
    /// `diodisto.c` does.
    fn distortion(
        &self,
        context: &crate::devices::distortion::DistortionContext<'_>,
    ) -> SpiceResult<crate::devices::distortion::DeviceDistortion> {
        use crate::devices::distortion::{
            Control, DeviceDistortion, DistortionTerm, Response, Taylor,
        };
        if self.recovery_node.is_some() {
            return Err(SpiceError::not_yet_ported(
                "diode soft-recovery distortion",
                "src/spicelib/devices/dio/diodisto.c",
            ));
        }
        let p = &self.parameters;
        let t = self.thermal(context.model_context)?;
        let vd = context.voltage(self.junction[0])? - context.voltage(self.junction[1])?;
        let csat = t.csat + t.csatsw.unwrap_or(0.);
        let vt = t.vt;
        let vte = t.n * vt;
        let tt = t.tt;
        let breakdown = t.breakdown.filter(|bv| *bv != 0.);
        let (g2, g3, cdiff2, cdiff3) = if vd >= -3. * vte {
            let evd = (vd / vte).exp();
            let gd = csat * evd / vte;
            let g2 = 0.5 * gd / vte;
            let g3 = g2 / 3. / vte;
            (g2, g3, g2 * tt, g3 * tt)
        } else if breakdown.is_none_or(|bv| vd >= -bv) {
            let arg = cubic_reverse_argument(vd, vte);
            let gd = csat * 3. * arg / vd;
            let g2 = -4. * gd / vd;
            (g2, 5. * g2 / vd, 0., 0.)
        } else {
            let bv = breakdown.unwrap_or(0.);
            let evrev = (-(bv + vd) / vt).exp();
            let gd = csat * evrev / vt;
            let g2 = -gd / 2. / vt;
            (g2, -g2 / 3. / vt, 0., 0.)
        };
        let depletion_cap = t.bottom.threshold;
        // `diotemp.c`: DIOtF2 = exp((1 + M(T)) ln(1 - FC)).
        let junction = |czero: Real, pot: Real, grading: Real, f2: Real| {
            if czero == 0. {
                (0., 0.)
            } else if vd < depletion_cap {
                let arg = 1. - vd / pot;
                let sarg = (-grading * arg.ln()).exp();
                let c1 = czero * sarg;
                let c2 = c1 / 2. / pot * grading / arg;
                let c3 = c2 / 3. / pot / arg * (grading + 1.);
                (c2, c3)
            } else {
                (czero / f2 / 2. / pot * grading, 0.)
            }
        };
        let f2 = t.bottom.f2;
        let f2_sw = t.sidewall.f2;
        let (cjunc2, cjunc3) = junction(t.cjo, p.model.vj, t.bottom.grading, f2);
        let (sw2, sw3) = junction(t.cjsw, p.model.vjsw, p.model.mjsw, f2_sw);
        let (cap2, cap3) = (cdiff2 + (cjunc2 + sw2), cdiff3 + (cjunc3 + sw3));
        let mut term = DistortionTerm::new(
            Response::Current,
            self.junction,
            vec![Control::between(self.junction[0], self.junction[1])],
            Taylor::single(g2, g3),
        );
        if cap2 != 0. {
            // diodisto.c folds the charge's `j omega` current into the
            // junction current's value before loading it.
            term = term.with_charge(Taylor::single(cap2, cap3));
        }
        let terms = vec![term];
        Ok(DeviceDistortion::Terms(terms))
    }

    /// `dionoise.c`: thermal noise of the series resistance at the instance
    /// temperature, shot noise `2 q abs(cd)` of the junction current (gmin
    /// current included, as C's `DIOcurrent`) and the flicker law
    /// `KF abs(cd/m)^AF m / f`. The `_rsw`/`_idsw`/`_1overfsw` generators of
    /// a separate sidewall (RSW, not ported) are zero, as C's are without RSW.
    fn noise(
        &self,
        context: &NoiseContext<'_>,
        sources: &mut Vec<NoiseSource>,
    ) -> SpiceResult<DeviceNoise<'_>> {
        let p = &self.parameters;
        let thermal = self.thermal(context.model_context)?;
        let vd = context.voltage(self.junction[0]) - context.voltage(self.junction[1]);
        let cd = thermal
            .point(vd, context.model_context.gmin)?
            .junction
            .current;
        let temperature = p.instance.kelvin(context.model_context.temperature);
        let m = p.instance.geometry.multiplier;
        // dionoise.c: `DIOfNcoef * exp(DIOfNexp * log(MAX(fabs(cd / DIOm),
        // N_MINLOG))) / freq * DIOm`.
        let flicker = p.model.kf * (p.model.af * (cd / m).abs().max(1e-38).ln()).exp();
        let [anode, cathode] = self.junction;
        let external = self.terminals[0];
        sources.extend([
            NoiseSource::new(
                "_rs",
                [anode, external],
                NoiseKind::Thermal {
                    conductance: thermal.conductance,
                    temperature,
                },
            ),
            NoiseSource::new("_id", self.junction, NoiseKind::Shot { current: cd }),
            NoiseSource::new(
                "_1overf",
                self.junction,
                NoiseKind::Flicker {
                    numerator: flicker,
                    law: FlickerLaw::PerHertzTimes { multiplier: m },
                },
            ),
            NoiseSource::new(
                "_rsw",
                [anode, external],
                NoiseKind::Thermal {
                    conductance: 0.,
                    temperature,
                },
            ),
            NoiseSource::new("_idsw", [anode, cathode], NoiseKind::Shot { current: 0. }),
            NoiseSource::new(
                "_1overfsw",
                [anode, cathode],
                NoiseKind::Flicker {
                    numerator: 0.,
                    law: FlickerLaw::PerHertzTimes { multiplier: m },
                },
            ),
        ]);
        Ok(DeviceNoise::Sources {
            family: NoiseFamily::Diode,
            model: Some(&self.model),
            total: true,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::devices::models::ModelResolver;
    use crate::netlist::Parser;
    use crate::netlist::source::parse_deck_text;
    use std::path::Path;

    fn deck(model: &str, instance: &str) -> crate::netlist::ast::Netlist {
        Parser::new()
            .parse_deck(&parse_deck_text(
                Path::new("diode.cir"),
                &format!("diode\nd1 a 0 dm {instance}\n.model dm d({model})\n.end\n"),
            ))
            .unwrap()
    }
    fn parameters(model: &str, instance: &str) -> SpiceResult<DiodeParameters> {
        let n = deck(model, instance);
        let resolver = ModelResolver::new(&n.models)?;
        let instance = &n.devices[0];
        let resolved = resolver.resolve(instance)?.unwrap();
        let setters = canonical(
            resolved
                .card()
                .parameters
                .iter()
                .filter(|p| !p.name.eq_ignore_ascii_case("level")),
            MODEL_ALIASES,
            MODEL_PENDING,
            "model",
        )?;
        let m = MODEL.validate(&setters, &resolved.card().location)?;
        let (_, setters) = crate::devices::initial::split(&instance.parameters, &["ic"])?;
        let setters = canonical(&setters, &[], INSTANCE_PENDING, "instance")?;
        let i = INSTANCE.validate(&setters, &instance.location)?;
        DiodeParameters::new(&m, &i, &instance.location)
    }
    fn thermal(model: &str, instance: &str, celsius: Real) -> Thermal {
        parameters(model, instance)
            .unwrap()
            .thermal(&ModelContext::new(celsius, 27.))
            .unwrap()
    }
    fn instantiate(model: &str, instance: &str) -> SpiceResult<Box<dyn Device>> {
        let n = deck(model, instance);
        let resolver = ModelResolver::new(&n.models)?;
        let resolved = resolver.resolve(&n.devices[0])?.unwrap();
        crate::primitives::NodeTable::new().transaction(|nodes| {
            Diode::instantiate(&n.devices[0], nodes, &resolved, &ModelContext::default())
        })
    }

    /// Central differences of current and charge against the analytic
    /// conductance and capacitance, away from C's region boundaries.
    fn assert_derivatives(t: &Thermal, voltages: &[Real]) {
        for &v in voltages {
            let h = 1e-6 * v.abs().max(1e-2);
            let p = t.point(v, 1e-12).unwrap().junction;
            let (plus, minus) = (
                t.point(v + h, 1e-12).unwrap().junction,
                t.point(v - h, 1e-12).unwrap().junction,
            );
            let g = (plus.current - minus.current) / (2. * h);
            let c = (plus.charge - minus.charge) / (2. * h);
            let current_scale = p.current.abs().max(plus.current.abs()) / h;
            assert!(
                (g - p.conductance).abs() <= 1e-5 * p.conductance.abs() + 1e-9 * current_scale,
                "{v}: dI/dV {g} vs {}",
                p.conductance
            );
            let charge_scale = p.charge.abs().max(plus.charge.abs()) / h;
            assert!(
                (c - p.capacitance).abs() <= 1e-5 * p.capacitance.abs() + 1e-9 * charge_scale,
                "{v}: dQ/dV {c} vs {}",
                p.capacitance
            );
        }
    }

    #[test]
    fn diode_taylor_reconstructs_physical_scalar_equations() -> SpiceResult<()> {
        use crate::maths::taylor::Jet;
        for (model, instance) in [
            ("is=1e-18 n=10", ""),
            (
                "is=1e-14 n=1.1 bv=5.1 ibv=5m nbv=1.4 tcv=2m cjo=30p vj=0.7 m=0.4 tt=5n",
                "",
            ),
            (
                "is=1e-14 jsw=2e-15 ns=1.6 cjsw=4p vjsw=0.8 mjsw=0.3 fcs=0.4 bv=5.1",
                "pj=10",
            ),
            (
                "is=1e-14 jsw=2e-15 cjsw=4p vjsw=0.8 mjsw=0.3 fcs=0.4 ikp=1u",
                "pj=10 m=2",
            ),
            ("isr=1e-11 nr=2.2 vj=0.8 m=0.4 cjo=5p", ""),
            (
                "jtun=1e-13 jtunsw=1e-14 ntun=20 xtitun=2 keg=0.9 bv=7",
                "pj=4",
            ),
            ("ikf=1m ikr=1n is=1e-12", ""),
            (
                "isr=1e-11 jtun=1e-12 ikf=1m ikr=1u bv=5.1 jsw=1e-15 ns=1.2",
                "pj=4",
            ),
        ] {
            for temperature in [-40., 27., 100.] {
                let thermal = thermal(model, instance, temperature);
                let edges = [
                    thermal.bottom.threshold,
                    thermal.sidewall.threshold,
                    -3. * thermal.n * thermal.vt,
                    -3. * thermal.ns.unwrap_or(thermal.n) * thermal.vt,
                    -thermal.breakdown.unwrap_or(Real::INFINITY),
                ];
                let voltages = [-9., -6.2, -5.4, -5., -3., -0.5, -0.05, 0.1, 0.4, 0.62, 0.75];
                for voltage in voltages.into_iter().chain(edges[..2].iter().copied()) {
                    for direction in [-1., 1.] {
                        // Each Taylor germ belongs to its own analytic
                        // interval. In particular C is continuous but its
                        // higher derivatives change at FC*VJ. Evaluating a
                        // right-hand germ at negative time is not a valid
                        // oracle at that boundary.
                        let step = edges
                            .iter()
                            .map(|edge| (edge - voltage) * direction)
                            .filter(|&distance| distance > 0.)
                            .fold(1e-4_f64, |step, distance| step.min(distance * 0.5));
                        let slope = step * direction;
                        let v = Jet::affine(12, voltage, slope);
                        let (current, charge) = thermal.taylor(&v, 1e-12, false)?;
                        for t in [0., 0.25, 0.5] {
                            let expected =
                                thermal.point(slope.mul_add(t, voltage), 1e-12)?.junction;
                            let at = |jet: &Jet| {
                                jet.coefficients()
                                    .iter()
                                    .rev()
                                    .fold(0_f64, |sum, coefficient| sum.mul_add(t, *coefficient))
                            };
                            let close = |kind: &str, actual: Real, expected: Real| {
                                assert!(
                                    (actual - expected).abs()
                                        <= 2e-11 * actual.abs().max(expected.abs()).max(1e-30),
                                    "{model} {temperature} {voltage} {slope} {t} {kind}: {actual:e} vs {expected:e}"
                                );
                            };
                            close("I", at(&current.value), expected.current);
                            close("G", at(&current.slope), expected.conductance);
                            close("Q", at(&charge.value), expected.charge);
                            close("C", at(&charge.slope), expected.capacitance);
                        }
                    }
                }
            }
        }
        Ok(())
    }

    #[test]
    fn diode_taylor_exponential_coefficients_survive_scaled_overflow() -> SpiceResult<()> {
        use crate::maths::taylor::Jet;
        let v = Jet::affine(8, 720., 0.25);
        let law = super::taylor::exponential(&v, 1., 1e-300);
        let scaled = 4.920_700_930_263_816e12;
        let mut expected = scaled;
        for (k, coefficient) in law.slope.coefficients().iter().enumerate() {
            if k != 0 {
                expected *= 0.25 / k as Real;
            }
            assert!((coefficient / expected - 1.).abs() < 2e-13);
        }
        Ok(())
    }

    #[test]
    fn junction_boundary_follows_each_reference_model() {
        for vt in [0.018_f64, 0.026, 0.04] {
            let is = 1.3e-14;
            let edge = -3. * vt;
            for (v, boundary, reverse) in [
                (edge.next_down(), ReverseBoundary::Exponential, true),
                (edge, ReverseBoundary::Exponential, false),
                (edge.next_up(), ReverseBoundary::Exponential, false),
                (edge.next_down(), ReverseBoundary::Cubic, true),
                (edge, ReverseBoundary::Cubic, true),
                (edge.next_up(), ReverseBoundary::Cubic, false),
            ] {
                let expected = if reverse {
                    let arg = 3. * vt / (v * std::f64::consts::E);
                    let arg = arg * arg * arg;
                    (-is * (1. + arg), is * 3. * arg / v)
                } else {
                    let ev = (v / vt).exp();
                    (is * (ev - 1.), is * ev / vt)
                };
                let actual = junction_current(v, vt, is, boundary).unwrap();
                assert!((actual.0 - expected.0).abs() <= 4. * Real::EPSILON * expected.0.abs());
                assert_eq!(actual.1.to_bits(), expected.1.to_bits());
            }
        }
    }

    #[test]
    fn depletion_matches_reference_and_preserves_diffusion_cancellation() {
        for (p, m, fc) in [(0.7, 0.33, 0.5), (1.1, 0.5, 0.8), (0.9, 0.999, 0.2)] {
            let law = DepletionLaw::new(p, m, fc);
            let fcp = fc * p;
            let xfc = (1. - fc).ln();
            let f1 = p * (1. - ((1. - m) * xfc).exp()) / (1. - m);
            let f2 = ((1. + m) * xfc).exp();
            let f3 = 1. - fc * (1. + m);
            for cz in [0., 2.3e-12, 7.7e-9] {
                for v in [-1., -0.0, fcp.next_down(), fcp, fcp.next_up(), fcp + 0.2] {
                    for diffusion in [None, Some((-0., -0.)), Some((-cz * f1, 1.2e-12))] {
                        let (q0, c0) = diffusion.unwrap_or((0., 0.));
                        let expected = if v < fcp {
                            let arg = 1. - v / p;
                            let sarg = (-m * arg.ln()).exp();
                            let q = p * cz * (1. - arg * sarg) / (1. - m);
                            let c = cz * sarg;
                            if diffusion.is_some() {
                                (q0 + q, c0 + c)
                            } else {
                                (q, c)
                            }
                        } else {
                            let czf2 = cz / f2;
                            let tail =
                                czf2 * (f3 * (v - fcp) + (m / (p + p)) * (v * v - fcp * fcp));
                            let c = czf2 * (f3 + m * v / p);
                            if diffusion.is_some() {
                                (q0 + cz * f1 + tail, c0 + c)
                            } else {
                                (cz * f1 + tail, c)
                            }
                        };
                        let actual = law.charge(v, cz, diffusion);
                        // The old subtraction is ill-conditioned when
                        // grading approaches one or diffusion cancels charge.
                        let charge_scale = (p * cz).abs().max(q0.abs());
                        assert!((actual.0 - expected.0).abs() <= 2e-12 * charge_scale);
                        assert!(
                            (actual.1 - expected.1).abs() <= 8. * Real::EPSILON * expected.1.abs()
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn junction_and_depletion_keep_the_linear_response_near_equilibrium() {
        for voltage in [-1e-30, 1e-30, -1e-20, 1e-20] {
            let (current, conductance) = exponential_junction(voltage, 0.026, 1e-14);
            let expected = voltage * (1e-14 / 0.026);
            assert!((current - expected).abs() <= 4. * Real::EPSILON * expected.abs());
            assert!((conductance - 1e-14 / 0.026).abs() <= 4. * Real::EPSILON * conductance);
            for grading in [0., 0.5, 0.999, 1.] {
                let (q, cap) = reverse_depletion(voltage, 2e-12, 0.7, grading);
                let expected = voltage * 2e-12;
                assert!((q - expected).abs() <= 4. * Real::EPSILON * expected.abs());
                assert_eq!(cap.to_bits(), (2e-12_f64).to_bits());
            }
        }
        let (current, conductance) = exponential_junction(720., 1., 1e-300);
        // exp(720)*1e-300, independently calculated in decimal.
        let expected = 4.920_700_930_263_816e12;
        assert!((current - expected).abs() < 2e-14 * expected);
        assert_eq!(current.to_bits(), conductance.to_bits());
        let law = DepletionLaw::new(0.7, 1., 0.5);
        let (q, c) = law.charge(0.35, 2e-12, None);
        assert!((q - 0.7 * 2e-12 * std::f64::consts::LN_2).abs() < 1e-27);
        assert!((c - 4e-12).abs() < 1e-27);
    }

    #[test]
    fn charge_derivative_and_current_jacobian_are_consistent() {
        let depletion = DepletionLaw::new(1., 0.5, 0.5);
        for v in [-1., -0.08, -0.02, 0., 0.49, 0.5, 0.51, 0.7] {
            let h = 1e-7;
            let (q, c) = depletion.charge(v, 2e-12, None);
            let derivative = (depletion.charge(v + h, 2e-12, None).0
                - depletion.charge(v - h, 2e-12, None).0)
                / (2. * h);
            assert!(
                (derivative - c).abs() < 1e-8 * c.abs(),
                "{v}: {q} {derivative} {c}"
            );
            let (i, g) = junction_current(v, 0.026, 1e-14, ReverseBoundary::Exponential).unwrap();
            let derivative = (junction_current(v + h, 0.026, 1e-14, ReverseBoundary::Exponential)
                .unwrap()
                .0
                - junction_current(v - h, 0.026, 1e-14, ReverseBoundary::Exponential)
                    .unwrap()
                    .0)
                / (2. * h);
            assert!(
                (derivative - g).abs() < 1e-7 * g.abs() + 4. * f64::EPSILON * 1e-14 / h,
                "{v}: {i} {derivative} {g}"
            );
        }
    }

    const SWEEP: &[Real] = &[
        -9., -6.2, -5.4, -5.0, -3., -0.5, -0.05, 0.1, 0.4, 0.62, 0.75,
    ];

    #[test]
    fn breakdown_matching_uses_circuit_tolerance_and_iteration_bound() {
        // Independently evaluated from the pinned diotemp.c recurrence.
        // The first four rows stop after 1, 3, 6 and 9 iterations; the
        // last retains iteration 25 when rounding prevents convergence.
        for (reltol, bits) in [
            (0.1, 0x4013_a3c0_04f4_949f),
            (1e-3, 0x4013_ae04_0454_e4e1),
            (1e-6, 0x4013_ae2a_032b_09da),
            (1e-9, 0x4013_ae2a_1443_b295),
            (1e-16, 0x4013_ae2a_144b_67f2),
        ] {
            assert_eq!(
                breakdown_voltage(5., 2e-9, 1e-11, 0.025_864, 1.3, reltol)
                    .unwrap()
                    .to_bits(),
                bits,
                "RELTOL={reltol}"
            );
        }
    }

    #[test]
    fn breakdown_jacobian_charge_and_matching() {
        for celsius in [-40., 27., 125.] {
            let t = thermal(
                "is=1e-14 n=1.1 bv=5.1 ibv=5m nbv=1.4 tcv=2m cjo=30p vj=0.7 m=0.4 tt=5n",
                "",
                celsius,
            );
            assert_derivatives(&t, SWEEP);
        }
        // At nominal temperature the knee carries IBV at V = -BV (diotemp.c).
        let t = thermal("is=1e-14 bv=5.1 ibv=5m nbv=1.4", "m=2", 27.);
        let i = t.point(-5.1, 0.).unwrap().junction.current;
        assert!((i + 2. * 5e-3).abs() < 1e-8 * 1e-2, "{i}");
        // TCV shifts the breakdown voltage linearly (TLEV 0) or relatively
        // (TLEV 1) before matching against the heated saturation current.
        for (tlev, tbv) in [(0, 10. - 1e-3 * 100.), (1, 10. * (1. - 1e-3 * 100.))] {
            let t = thermal(&format!("bv=10 tcv=1m tlev={tlev} ibv=1e-3"), "", 127.);
            let expected =
                breakdown_voltage(tbv, 1e-3, t.csat, t.vt, 1., crate::devices::DEFAULT_RELTOL)
                    .unwrap();
            assert_eq!(t.breakdown, Some(expected));
            assert!(expected < tbv && expected > tbv - 0.5);
        }
        // IBV below IS*BV/Vt keeps the unmatched BV, like C.
        assert_eq!(
            thermal("is=1e-6 bv=50 ibv=1e-6", "", 27.).breakdown,
            Some(50.)
        );
    }

    #[test]
    fn sidewall_jacobian_and_charge_for_both_characteristics() {
        for (model, instance) in [
            (
                "is=1e-14 jsw=2e-15 ns=1.6 cjsw=4p vjsw=0.8 mjsw=0.3 fcs=0.4 bv=5.1",
                "pj=10",
            ),
            (
                "is=1e-14 jsw=2e-15 cjsw=4p vjsw=0.8 mjsw=0.3 fcs=0.4 ikp=1u",
                "pj=10 m=2",
            ),
            ("cjsw=4p php=0.6 cjo=10p", "perim=3"),
        ] {
            for celsius in [-20., 27., 90.] {
                assert_derivatives(&thermal(model, instance, celsius), SWEEP);
            }
        }
        // PJ scales the sidewall saturation current and capacitance.
        let one = thermal("jsw=1e-15 cjsw=1p", "pj=1", 27.)
            .point(0.5, 0.)
            .unwrap();
        let ten = thermal("jsw=1e-15 cjsw=1p", "pj=10", 27.)
            .point(0.5, 0.)
            .unwrap();
        let bottom = thermal("", "", 27.).point(0.5, 0.).unwrap();
        let sidewall = |p: DiodePoint| p.junction.current - bottom.junction.current;
        assert!((sidewall(ten) / sidewall(one) - 10.).abs() < 1e-9);
        assert!((ten.junction.capacitance / one.junction.capacitance - 10.).abs() < 1e-9);
        // The model PJ is the instance default.
        let model_pj = thermal("jsw=1e-15 pj=10", "", 27.).point(0.5, 0.).unwrap();
        assert!((sidewall(model_pj) / sidewall(one) - 10.).abs() < 1e-9);
    }

    #[test]
    fn recombination_tunnelling_and_knee_jacobians() {
        for model in [
            "isr=1e-11 nr=2.2 vj=0.8 m=0.4 cjo=5p",
            "jtun=1e-13 jtunsw=1e-14 ntun=20 xtitun=2 keg=0.9 bv=7",
            "ikf=1m ikr=1n is=1e-12",
            "isr=1e-11 jtun=1e-12 ikf=1m ikr=1u bv=5.1 jsw=1e-15 ns=1.2",
        ] {
            for celsius in [-40., 27., 100.] {
                assert_derivatives(&thermal(model, "pj=4", celsius), SWEEP);
            }
        }
        // C's small-signal conductance equals the Newton derivative except for
        // the recombination term, whose dioload.c conductance omits 1/VJ and
        // reapplies the generation factor.
        let plain = thermal("ikf=1m jtun=1e-12 tt=1n", "", 27.)
            .point(0.6, 1e-12)
            .unwrap();
        assert_eq!(plain.ac_conductance, plain.junction.conductance);
        assert_eq!(plain.ac_capacitance, plain.junction.capacitance);
        let t = thermal("isr=1e-11 vj=0.8 m=0.4", "", 27.);
        let p = t.point(0.6, 0.).unwrap();
        let (vterec, vj, m, v) = (2. * t.vt, 0.8, 0.4, 0.6);
        let current = 1e-11 * ((v / vterec).exp() - 1.);
        let t1: Real = (1. - v / vj).powi(2) + 0.005;
        let factor = t1.powf(m / 2.);
        let c_derivative = -m * (1. - v / vj) * t1.powf(m / 2. - 1.);
        let bottom = t.csat * (v / t.vt).exp() / t.vt;
        let expected =
            bottom + 1e-11 * (v / vterec).exp() / vterec * factor + current * factor * c_derivative;
        assert!((p.ac_conductance - expected).abs() < 1e-12 * expected);
        assert!(p.ac_conductance != p.junction.conductance);
    }

    #[test]
    fn temperature_laws_follow_diotemp() {
        // Nominal temperature leaves VJ, CJO and IS unchanged.
        let t = thermal("cjo=10p vj=0.7 m=0.4 cjsw=1p vjsw=0.6", "pj=2", 27.);
        assert!((t.bottom.potential - 0.7).abs() < 1e-14 && (t.cjo - 10e-12).abs() < 1e-24);
        assert!((t.sidewall.potential - 0.6).abs() < 1e-14 && (t.cjsw - 2e-12).abs() < 1e-24);
        assert!((t.csat - 1e-14).abs() < 1e-28);
        // TLEV 0: IS(T) = IS exp((T/Tn - 1) EG/(N Vt) + XTI/N ln(T/Tn)).
        let (tk, tn) = (398.15, 300.15);
        let vt = K_OVER_Q * tk;
        let t = thermal("is=1e-14 n=1.2 eg=1.2 xti=2", "", 125.);
        let expected =
            1e-14 * ((tk / tn - 1.) * 1.2 / (1.2 * vt) + 2. / 1.2 * (tk / tn).ln()).exp();
        assert!((t.csat / expected - 1.).abs() < 1e-12);
        // DTEMP offsets the circuit temperature; TEMP replaces it.
        let offset = thermal("is=1e-14", "dtemp=98", 27.);
        assert!((offset.csat / thermal("is=1e-14", "", 125.).csat - 1.).abs() < 1e-12);
        let fixed = thermal("is=1e-14", "temp=125", -40.);
        assert!((fixed.csat / offset.csat - 1.).abs() < 1e-12);
        // TNOM moves the reference temperature.
        let tnom = thermal("is=1e-14 tnom=125", "", 125.);
        assert!((tnom.csat - 1e-14).abs() < 1e-28);
        // TLEVC 1: linear CTA/TPB laws about REFTEMP.
        let t = thermal(
            "cjo=10p vj=0.7 tlevc=1 cta=1m tpb=2m cjsw=1p vjsw=0.5 ctp=3m tphp=1m",
            "pj=1",
            77.,
        );
        assert!((t.cjo - 10e-12 * 1.05).abs() < 1e-24 && (t.bottom.potential - 0.6).abs() < 1e-12);
        assert!(
            (t.cjsw - 1e-12 * 1.15).abs() < 1e-24 && (t.sidewall.potential - 0.45).abs() < 1e-12
        );
        // TLEVC 0 lowers the junction potential and raises CJO when heated.
        let hot = thermal("cjo=10p vj=0.7 m=0.4", "", 125.);
        assert!(hot.bottom.potential < 0.7 && hot.cjo > 10e-12);
        // TLEV 2 uses the EG/GAP1/GAP2 band gap; TT/RS coefficients apply.
        let t = thermal(
            "tlev=2 eg=1.16 gap1=7.02e-4 gap2=1108 tt=1n ttt1=1m rs=10 trs1=2m",
            "",
            127.,
        );
        assert!((t.tt - 1.1e-9).abs() < 1e-21);
        assert!((t.conductance - 1. / (10. * 1.2)).abs() < 1e-12);
        for celsius in [-55., 27., 150.] {
            assert_derivatives(
                &thermal(
                    "tlev=2 tlevc=0 cjo=20p vj=0.75 m=0.45 tm1=1m isr=1e-12 bv=12",
                    "",
                    celsius,
                ),
                SWEEP,
            );
        }
    }

    #[test]
    fn aliases_keep_last_set_precedence() {
        let p = parameters("cj0=1p cjo=2p cj=3p pb=0.6 mj=0.3 vb=7 ib=1u nz=2", "").unwrap();
        assert_eq!(
            (p.model.cjo, p.model.vj, p.model.grading),
            (3e-12, 0.6, 0.3)
        );
        assert_eq!((p.model.bv, p.model.ibv, p.model.nbv), (Some(7.), 1e-6, 2.));
        let p = parameters("cjo=2p cj0=1p", "").unwrap();
        assert_eq!(p.model.cjo, 1e-12);
        // NBV defaults to N; EG to 1.11 eV, or 1.16 eV under TLEV 2.
        let p = parameters("n=1.3 tlev=2", "").unwrap();
        assert_eq!((p.model.nbv, p.model.eg), (1.3, 1.16));
        assert_eq!(parameters("", "").unwrap().model.eg, 1.11);
    }

    #[test]
    fn unported_and_invalid_setters_fail_explicitly() {
        for (model, instance) in [
            ("rsw=1", ""),
            ("rth0=10", ""),
            ("bv_max=10", ""),
            ("xom=1e4", ""),
            ("", "w=1u l=1u"),
            ("jsw=1e-15 bv=5", "pj=1"),
            ("cjsw=1p tm1=1m", "pj=1"),
        ] {
            let error = instantiate(model, instance).unwrap_err();
            assert!(error.is_not_yet_ported(), "{model} {instance}: {error}");
        }
        // JSW with NS (own characteristic) supports breakdown; JSW*PJ = 0 is
        // harmless under a shared characteristic.
        assert!(instantiate("jsw=1e-15 ns=1 bv=5", "pj=1").is_ok());
        assert!(instantiate("jsw=1e-15 bv=5", "").is_ok());
        // OFF and IC (#99) are ported.
        assert!(instantiate("", "off ic=0.5").is_ok());
        for (model, instance) in [
            ("tlev=3", ""),
            ("tlevc=2", ""),
            ("mjsw=1", ""),
            ("fcs=1", ""),
            ("bv=0", ""),
            ("", "temp=30 dtemp=2"),
            ("unknown=1", ""),
        ] {
            let error = instantiate(model, instance).unwrap_err();
            assert!(!error.is_not_yet_ported(), "{model} {instance}: {error}");
        }
    }
}
