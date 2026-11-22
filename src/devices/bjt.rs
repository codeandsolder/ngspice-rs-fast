//! Gummel-Poon level-1 bipolar junction transistor.
//!
//! C references (behaviour only, reimplemented):
//! - `bjt/bjt.c` / `bjtmpar.c`: model and instance setters, including aliases.
//! - `bjt/bjtsetup.c`: defaults and the internal collector/base/emitter nodes
//!   created for nonzero `RC`/`RB`/`RE`.
//! - `bjt/bjttemp.c`: temperature, area and `TLEV`/`TLEVC` scaling.
//! - `bjt/bjtload.c`: base charge `qb` (Early `VAF`/`VAR`, high injection
//!   `IKF`/`IKR`/`NKF`), `ISE`/`ISC` leakage, bias-dependent base resistance
//!   (`RB`/`RBM`/`IRB`), bias-dependent transit time (`XTF`/`VTF`/`ITF`),
//!   the split base-collector capacitance (`XCJC`) and the substrate junction
//!   (`ISS`/`NS`, `CJS`/`VJS`/`MJS`, `SUBS`).
//! - `bjt/bjtacld.c`: the small-signal stamp.
//! - `bjt/bjttrunc.c`: charges taking part in truncation-error control.
//!
//! DC/transient Newton and AC loads retain the actual C matrix. The
//! bias-dependent base resistance is stamped as conductance `gx` alone,
//! without the `d(gx)/dV` terms of the mathematical Jacobian; the residual
//! and fixed point of the Gummel-Poon equations are unchanged.
//!
//! Not ported, and rejected with [`SpiceError::NotYetPorted`]: excess phase
//! (`PTF` with `TF != 0`), Kull's quasi-saturation model (`RCO`, `VO`, `GAMMA`,
//! `QCO`, ...), safe-operating-area limits (`*_MAX`, `RTH0`). The `.noise`
//! generators of `bjtnoise.c` (resistor thermal, collector/base shot and
//! `KF`/`AF` flicker noise) are ported through [`Device::noise`].
//!
//! `.sens` (#164) replays `bjtsetup.c`/`bjttemp.c` on C's records and reloads
//! this device from them, so C's per-parameter `DEVsetup` (`ISE`/`ISC` above
//! 1e-4 become multiples of `IS`, `NKF`/`XCJC` clamp), `BJTemp`'s model-side
//! stores (`tnom`, `rbm` following `rb`, the `fc` clamp), the Celsius
//! `tnom`/`temp` pair and `sens_setp()`'s hard-coded `RCO` exclusion all show
//! up in the columns. A card with `RC != 0` is refused there: C's walk then
//! activates the unported quasi-saturation terms in its perturbed loads.
//!
//! Newton limiting follows `bjtload.c` through [`crate::devices::limiting`]:
//! `MODEINITJCT` starts at `vbe = tVcrit` (all zero for an `OFF` instance,
//! which is also held at zero through `MODEINITFIX`); the `uic` initial load
//! starts at `vbe = ICVBE`, `vbc = vbx = ICVBE - ICVCE`, `vsub = 0`, unset
//! components defaulted from the external terminals as `bjtgetic.c` does;
//! later loads apply `DEVpnjlim`
//! to `vbe`, `vbc` and `vsub`. C's `MODEINITPRED` extrapolation, bypass and
//! the quasi-saturation `vbcx`/`vrci` limits (a rejected model) are not ported.
// C-compatible numerical expressions are kept in their reference order:
// Clippy's fused multiply-add suggestions change IEEE-754 rounding.
#![expect(
    clippy::similar_names,
    reason = "C-translated equation symbols keep their original distinct names for source-level audit"
)]
#![expect(
    clippy::float_cmp,
    reason = "C device branches compare floating-point values exactly, including nonfinite cases"
)]
#![expect(
    clippy::too_many_lines,
    reason = "a complete translated C numerical routine remains contiguous for equation parity review"
)]
#![expect(
    clippy::suboptimal_flops,
    reason = "SPICE numerical reference parity requires the original non-fused C operation order"
)]

mod disto;
mod physics;

static INSTANCE_SCHEMA: ScalarSchema<'static> = ScalarSchema::new(INSTANCE);

static MODEL_SCHEMA: ScalarSchema<'static> = ScalarSchema::new(MODEL);

#[cfg(test)]
use physics::Point;
use physics::{Junction, Transport, c_point};

use crate::devices::limiting::{self, Limiter, Linearization};
use crate::devices::noise::{
    DeviceNoise, FlickerLaw, NoiseContext, NoiseFamily, NoiseKind, NoiseSource,
};
use crate::devices::nonlinear::DepletionLaw;
use crate::devices::nonlinear::charge::ChargeBranch;
use crate::devices::nonlinear::thermal::{
    BOLTZMANN, CELSIUS_TO_KELVIN, K_OVER_Q, PotentialTemperature, REFTEMP, TemperatureCache,
    TemperatureKey, band_gap, potential_shift,
};
use crate::devices::schema::{
    ScalarDomain as D, ScalarParameter as P, ScalarSchema, ScalarUnit as U, ScalarValues,
};
use crate::devices::sensitivity::{
    AcReset, ParameterScope, RecordPair, SensitivityFamily, SensitivityKernel, SensitivityLoad,
    SensitivityMode, SensitivityParameter, SensitivityRecord,
};
use crate::devices::{
    Device, LinearContext, ModelContext, ModelFamily, ResolvedModel, StampContext,
};
use crate::maths::Vector;
use crate::netlist::ast::{DeviceInstance, ParameterAssignment};
use crate::primitives::{NodeAppend, NodeId, NodeKind, Real, SpiceError, SpiceResult};

const fn p(name: &'static str, unit: U, domain: D, default: Option<Real>) -> P {
    P {
        name,
        unit,
        domain,
        default,
    }
}

/// The supported model setters. Defaults follow `bjtsetup.c`/`bjttemp.c`;
/// values without a default are "not given" (C `*Given` flags).
const MODEL: &[P] = &[
    p("is", U::Ampere, D::Positive, Some(1e-16)),
    p("ibe", U::Ampere, D::NonNegative, None),
    p("ibc", U::Ampere, D::NonNegative, None),
    p("bf", U::Dimensionless, D::Positive, Some(100.)),
    p("nf", U::Dimensionless, D::Positive, Some(1.)),
    p("vaf", U::Volt, D::NonNegative, Some(0.)),
    p("ikf", U::Ampere, D::NonNegative, Some(0.)),
    p("nkf", U::Dimensionless, D::Positive, None),
    p("ise", U::Ampere, D::NonNegative, Some(0.)),
    p("ne", U::Dimensionless, D::Positive, Some(1.5)),
    p("br", U::Dimensionless, D::Positive, Some(1.)),
    p("nr", U::Dimensionless, D::Positive, Some(1.)),
    p("var", U::Volt, D::NonNegative, Some(0.)),
    p("ikr", U::Ampere, D::NonNegative, Some(0.)),
    p("isc", U::Ampere, D::NonNegative, Some(0.)),
    p("nc", U::Dimensionless, D::Positive, Some(2.)),
    p("rb", U::Ohm, D::NonNegative, Some(0.)),
    p("irb", U::Ampere, D::NonNegative, Some(0.)),
    p("rbm", U::Ohm, D::NonNegative, None),
    p("re", U::Ohm, D::NonNegative, Some(0.)),
    p("rc", U::Ohm, D::NonNegative, Some(0.)),
    p("cje", U::Farad, D::NonNegative, Some(0.)),
    p("vje", U::Volt, D::Positive, Some(0.75)),
    p("mje", U::Dimensionless, D::NonNegative, Some(0.33)),
    p("tf", U::Second, D::NonNegative, Some(0.)),
    p("xtf", U::Dimensionless, D::NonNegative, Some(0.)),
    p("vtf", U::Volt, D::NonNegative, Some(0.)),
    p("itf", U::Ampere, D::NonNegative, Some(0.)),
    p("ptf", U::Dimensionless, D::Finite, Some(0.)),
    p("cjc", U::Farad, D::NonNegative, Some(0.)),
    p("vjc", U::Volt, D::Positive, Some(0.75)),
    p("mjc", U::Dimensionless, D::NonNegative, Some(0.33)),
    p("xcjc", U::Dimensionless, D::Finite, Some(1.)),
    p("tr", U::Second, D::NonNegative, Some(0.)),
    p("cjs", U::Farad, D::NonNegative, Some(0.)),
    p("vjs", U::Volt, D::Positive, Some(0.75)),
    p("mjs", U::Dimensionless, D::NonNegative, Some(0.)),
    p("xtb", U::Dimensionless, D::Finite, Some(0.)),
    p("eg", U::Volt, D::Positive, Some(1.11)),
    p("xti", U::Dimensionless, D::Finite, Some(3.)),
    p("fc", U::Dimensionless, D::NonNegative, Some(0.5)),
    p("iss", U::Ampere, D::NonNegative, None),
    p("ns", U::Dimensionless, D::Positive, Some(1.)),
    p("tnom", U::Celsius, D::Temperature, None),
    p("subs", U::Dimensionless, D::Finite, None),
    p("tlev", U::Dimensionless, D::Finite, Some(0.)),
    p("tlevc", U::Dimensionless, D::Finite, Some(0.)),
    p("tbf1", U::InverseKelvin, D::Finite, None),
    p("tbf2", U::InverseKelvinSquared, D::Finite, None),
    p("tbr1", U::InverseKelvin, D::Finite, None),
    p("tbr2", U::InverseKelvinSquared, D::Finite, None),
    p("tikf1", U::InverseKelvin, D::Finite, Some(0.)),
    p("tikf2", U::InverseKelvinSquared, D::Finite, Some(0.)),
    p("tikr1", U::InverseKelvin, D::Finite, Some(0.)),
    p("tikr2", U::InverseKelvinSquared, D::Finite, Some(0.)),
    p("tirb1", U::InverseKelvin, D::Finite, Some(0.)),
    p("tirb2", U::InverseKelvinSquared, D::Finite, Some(0.)),
    p("tnc1", U::InverseKelvin, D::Finite, Some(0.)),
    p("tnc2", U::InverseKelvinSquared, D::Finite, Some(0.)),
    p("tne1", U::InverseKelvin, D::Finite, Some(0.)),
    p("tne2", U::InverseKelvinSquared, D::Finite, Some(0.)),
    p("tnf1", U::InverseKelvin, D::Finite, Some(0.)),
    p("tnf2", U::InverseKelvinSquared, D::Finite, Some(0.)),
    p("tnr1", U::InverseKelvin, D::Finite, Some(0.)),
    p("tnr2", U::InverseKelvinSquared, D::Finite, Some(0.)),
    p("trb1", U::InverseKelvin, D::Finite, Some(0.)),
    p("trb2", U::InverseKelvinSquared, D::Finite, Some(0.)),
    p("trc1", U::InverseKelvin, D::Finite, Some(0.)),
    p("trc2", U::InverseKelvinSquared, D::Finite, Some(0.)),
    p("tre1", U::InverseKelvin, D::Finite, Some(0.)),
    p("tre2", U::InverseKelvinSquared, D::Finite, Some(0.)),
    p("trm1", U::InverseKelvin, D::Finite, Some(0.)),
    p("trm2", U::InverseKelvinSquared, D::Finite, Some(0.)),
    p("tvaf1", U::InverseKelvin, D::Finite, Some(0.)),
    p("tvaf2", U::InverseKelvinSquared, D::Finite, Some(0.)),
    p("tvar1", U::InverseKelvin, D::Finite, Some(0.)),
    p("tvar2", U::InverseKelvinSquared, D::Finite, Some(0.)),
    p("ctc", U::InverseKelvin, D::Finite, Some(0.)),
    p("cte", U::InverseKelvin, D::Finite, Some(0.)),
    p("cts", U::InverseKelvin, D::Finite, Some(0.)),
    p("tvjc", U::Volt, D::Finite, Some(0.)),
    p("tvje", U::Volt, D::Finite, Some(0.)),
    p("tvjs", U::Volt, D::Finite, Some(0.)),
    p("titf1", U::InverseKelvin, D::Finite, Some(0.)),
    p("titf2", U::InverseKelvinSquared, D::Finite, Some(0.)),
    p("ttf1", U::InverseKelvin, D::Finite, Some(0.)),
    p("ttf2", U::InverseKelvinSquared, D::Finite, Some(0.)),
    p("ttr1", U::InverseKelvin, D::Finite, Some(0.)),
    p("ttr2", U::InverseKelvinSquared, D::Finite, Some(0.)),
    p("tmje1", U::InverseKelvin, D::Finite, Some(0.)),
    p("tmje2", U::InverseKelvinSquared, D::Finite, Some(0.)),
    p("tmjc1", U::InverseKelvin, D::Finite, Some(0.)),
    p("tmjc2", U::InverseKelvinSquared, D::Finite, Some(0.)),
    p("tmjs1", U::InverseKelvin, D::Finite, Some(0.)),
    p("tmjs2", U::InverseKelvinSquared, D::Finite, Some(0.)),
    p("tns1", U::InverseKelvin, D::Finite, Some(0.)),
    p("tns2", U::InverseKelvinSquared, D::Finite, Some(0.)),
    p("tis1", U::InverseKelvin, D::Finite, Some(0.)),
    p("tis2", U::InverseKelvinSquared, D::Finite, Some(0.)),
    p("tise1", U::InverseKelvin, D::Finite, Some(0.)),
    p("tise2", U::InverseKelvinSquared, D::Finite, Some(0.)),
    p("tisc1", U::InverseKelvin, D::Finite, Some(0.)),
    p("tisc2", U::InverseKelvinSquared, D::Finite, Some(0.)),
    p("tiss1", U::InverseKelvin, D::Finite, Some(0.)),
    p("tiss2", U::InverseKelvinSquared, D::Finite, Some(0.)),
    // bjtnoise.c flicker law; bjtsetup.c defaults KF = 0, AF = 1.
    p("kf", U::Dimensionless, D::Finite, Some(0.)),
    p("af", U::Dimensionless, D::Finite, Some(1.)),
];

/// `bjt.c` alias setters (`IOPR`/`IOPAR`) and their canonical parameter.
const ALIASES: &[(&str, &str)] = &[
    ("tref", "tnom"),
    ("va", "vaf"),
    ("ik", "ikf"),
    ("nk", "nkf"),
    ("c2", "ise"),
    ("vb", "var"),
    ("c4", "isc"),
    ("pe", "vje"),
    ("me", "mje"),
    ("pc", "vjc"),
    ("mc", "mjc"),
    ("csub", "cjs"),
    ("ccs", "cjs"),
    ("ps", "vjs"),
    ("ms", "mjs"),
    ("trb", "trb1"),
    ("trc", "trc1"),
    ("tre", "tre1"),
];

/// Model setters C accepts whose physics is not ported, with the C file
/// that defines it.
const UNPORTED_MODEL: &[(&str, &str)] = &[
    ("rco", "Kull quasi-saturation, bjt/bjtload.c and bjttemp.c"),
    ("vo", "Kull quasi-saturation, bjt/bjtload.c and bjttemp.c"),
    (
        "gamma",
        "Kull quasi-saturation, bjt/bjtload.c and bjttemp.c",
    ),
    ("qco", "Kull quasi-saturation, bjt/bjtload.c and bjttemp.c"),
    ("quasimod", "Kull quasi-saturation, bjt/bjttemp.c"),
    ("vg", "Kull quasi-saturation, bjt/bjttemp.c"),
    ("cn", "Kull quasi-saturation, bjt/bjttemp.c"),
    ("d", "Kull quasi-saturation, bjt/bjttemp.c"),
    ("vbe_max", "BJT safe-operating-area check, bjt/bjtsoachk.c"),
    ("vbc_max", "BJT safe-operating-area check, bjt/bjtsoachk.c"),
    ("vce_max", "BJT safe-operating-area check, bjt/bjtsoachk.c"),
    ("pd_max", "BJT safe-operating-area check, bjt/bjtsoachk.c"),
    ("ic_max", "BJT safe-operating-area check, bjt/bjtsoachk.c"),
    ("ib_max", "BJT safe-operating-area check, bjt/bjtsoachk.c"),
    ("te_max", "BJT safe-operating-area check, bjt/bjtsoachk.c"),
    ("rth0", "BJT safe-operating-area check, bjt/bjtsoachk.c"),
];

const INSTANCE: &[P] = &[
    p("area", U::Dimensionless, D::Positive, Some(1.)),
    p("areab", U::Dimensionless, D::Positive, None),
    p("areac", U::Dimensionless, D::Positive, None),
    p("m", U::Dimensionless, D::Positive, Some(1.)),
    p("temp", U::Celsius, D::Temperature, None),
    p("dtemp", U::Celsius, D::Finite, Some(0.)),
];

/// The `IC` vector components (`bjtpar.c`): `ICVBE`, `ICVCE`.
const IC_COMPONENTS: [&str; 2] = ["icvbe", "icvce"];

fn canonical(parameter: &ParameterAssignment) -> ParameterAssignment {
    let mut parameter = parameter.clone();
    if let Some((_, name)) = ALIASES
        .iter()
        .find(|(alias, _)| alias.eq_ignore_ascii_case(&parameter.name))
    {
        (*name).clone_into(&mut parameter.name);
    }
    parameter
}

fn get(values: &ScalarValues, name: &str) -> SpiceResult<Real> {
    values
        .get(name)
        .map(|v| v.value)
        .ok_or_else(|| SpiceError::circuit(format!("missing BJT schema default {name}")))
}

/// An explicitly given value (C `*Given`), not a schema default.
fn given(values: &ScalarValues, name: &str) -> Option<Real> {
    values
        .get(name)
        .filter(|v| v.location.is_some())
        .map(|v| v.value)
}

/// A quadratic temperature coefficient pair `(c1, c2)` (`1 + c1 dt + c2 dt^2`).
type TempCo = [Real; 2];

fn poly(c: TempCo, dt: Real) -> Real {
    1. + c[0] * dt + c[1] * dt * dt
}

/// Validated model data at the parameter-measurement temperature.
#[derive(Debug, Clone, Copy)]
struct Model {
    is: Real,
    /// `IBE`/`IBC`, used only when both are given (`bjttemp.c`).
    ibe_ibc: Option<(Real, Real)>,
    /// Individual setters survive even when the physical model ignores a
    /// single-sided card. `.sens` can subsequently mark the partner given.
    ibe: Option<Real>,
    ibc: Option<Real>,
    bf: Real,
    nf: Real,
    vaf: Real,
    ikf: Real,
    nkf: Option<Real>,
    ise: Real,
    ne: Real,
    br: Real,
    nr: Real,
    var: Real,
    ikr: Real,
    isc: Real,
    nc: Real,
    rb: Real,
    irb: Real,
    rbm: Real,
    /// `BJTminBaseResistGiven`: `.sens` needs it because `bjttemp.c` copies
    /// `rb` into an ungiven `rbm` before every load.
    rbm_given: bool,
    re: Real,
    rc: Real,
    cje: Real,
    vje: Real,
    mje: Real,
    tf: Real,
    xtf: Real,
    vtf: Real,
    itf: Real,
    cjc: Real,
    vjc: Real,
    mjc: Real,
    xcjc: Real,
    tr: Real,
    cjs: Real,
    vjs: Real,
    mjs: Real,
    xtb: Real,
    eg: Real,
    xti: Real,
    fc: Real,
    iss: Option<Real>,
    ns: Real,
    /// Kelvin, `None` for the circuit nominal temperature.
    tnom: Option<Real>,
    tlev: u8,
    tlevc: u8,
    tbf: Option<TempCo>,
    tbr: Option<TempCo>,
    tikf: TempCo,
    tikr: TempCo,
    tirb: TempCo,
    tnc: TempCo,
    tne: TempCo,
    tnf: TempCo,
    tnr: TempCo,
    trb: TempCo,
    trc: TempCo,
    tre: TempCo,
    trm: TempCo,
    tvaf: TempCo,
    tvar: TempCo,
    titf: TempCo,
    ttf: TempCo,
    ttr: TempCo,
    tmje: TempCo,
    tmjc: TempCo,
    tmjs: TempCo,
    tns: TempCo,
    tis: TempCo,
    tise: TempCo,
    tisc: TempCo,
    tiss: TempCo,
    ctc: Real,
    cte: Real,
    cts: Real,
    tvjc: Real,
    tvje: Real,
    tvjs: Real,
    /// Flicker-noise coefficient and exponent (`bjtnoise.c`).
    kf: Real,
    af: Real,
}

fn integer_selector(values: &ScalarValues, name: &str, allowed: &[u8]) -> SpiceResult<u8> {
    let raw = get(values, name)?;
    allowed
        .iter()
        .copied()
        .find(|v| Real::from(*v) == raw)
        .ok_or_else(|| {
            SpiceError::circuit(format!(
                "BJT {name}={raw} is not one of the ported selectors {allowed:?}"
            ))
        })
}

impl Model {
    fn from_values(m: &ScalarValues) -> SpiceResult<Self> {
        let pair = |a: &str, b: &str| -> SpiceResult<TempCo> { Ok([get(m, a)?, get(m, b)?]) };
        // bjttemp.c: BF/BR use the TBF/TBR polynomial when either is given.
        let optional_pair = |a: &str, b: &str| -> Option<TempCo> {
            match (given(m, a), given(m, b)) {
                (None, None) => None,
                (x, y) => Some([x.unwrap_or(0.), y.unwrap_or(0.)]),
            }
        };
        let is = get(m, "is")?;
        // bjtsetup.c: a leakage current above 1e-4 is a multiple of IS.
        let leak = |name: &str| -> SpiceResult<Real> {
            let value = get(m, name)?;
            Ok(if value > 1e-4 { is * value } else { value })
        };
        let rb = get(m, "rb")?;
        let fc = get(m, "fc")?;
        let xcjc = get(m, "xcjc")?;
        let model = Self {
            is,
            ibe_ibc: given(m, "ibe").zip(given(m, "ibc")),
            ibe: given(m, "ibe"),
            ibc: given(m, "ibc"),
            bf: get(m, "bf")?,
            nf: get(m, "nf")?,
            vaf: get(m, "vaf")?,
            ikf: get(m, "ikf")?,
            // bjtsetup.c clamps NKF to at most 1.
            nkf: given(m, "nkf").map(|v| v.min(1.)),
            ise: leak("ise")?,
            ne: get(m, "ne")?,
            br: get(m, "br")?,
            nr: get(m, "nr")?,
            var: get(m, "var")?,
            ikr: get(m, "ikr")?,
            isc: leak("isc")?,
            nc: get(m, "nc")?,
            rb,
            irb: get(m, "irb")?,
            rbm: given(m, "rbm").unwrap_or(rb),
            rbm_given: given(m, "rbm").is_some(),
            re: get(m, "re")?,
            rc: get(m, "rc")?,
            cje: get(m, "cje")?,
            vje: get(m, "vje")?,
            mje: get(m, "mje")?,
            tf: get(m, "tf")?,
            xtf: get(m, "xtf")?,
            vtf: get(m, "vtf")?,
            itf: get(m, "itf")?,
            cjc: get(m, "cjc")?,
            vjc: get(m, "vjc")?,
            mjc: get(m, "mjc")?,
            // bjtsetup.c clamps XCJC into [0, 1].
            xcjc: xcjc.clamp(0., 1.),
            tr: get(m, "tr")?,
            cjs: get(m, "cjs")?,
            vjs: get(m, "vjs")?,
            mjs: get(m, "mjs")?,
            xtb: get(m, "xtb")?,
            eg: get(m, "eg")?,
            xti: get(m, "xti")?,
            // bjttemp.c limits FC to 0.9999.
            fc: fc.min(0.9999),
            iss: given(m, "iss"),
            ns: get(m, "ns")?,
            tnom: given(m, "tnom").map(|t| t + CELSIUS_TO_KELVIN),
            tlev: integer_selector(m, "tlev", &[0, 1, 3])?,
            tlevc: integer_selector(m, "tlevc", &[0, 1])?,
            tbf: optional_pair("tbf1", "tbf2"),
            tbr: optional_pair("tbr1", "tbr2"),
            tikf: pair("tikf1", "tikf2")?,
            tikr: pair("tikr1", "tikr2")?,
            tirb: pair("tirb1", "tirb2")?,
            tnc: pair("tnc1", "tnc2")?,
            tne: pair("tne1", "tne2")?,
            tnf: pair("tnf1", "tnf2")?,
            tnr: pair("tnr1", "tnr2")?,
            trb: pair("trb1", "trb2")?,
            trc: pair("trc1", "trc2")?,
            tre: pair("tre1", "tre2")?,
            trm: pair("trm1", "trm2")?,
            tvaf: pair("tvaf1", "tvaf2")?,
            tvar: pair("tvar1", "tvar2")?,
            titf: pair("titf1", "titf2")?,
            ttf: pair("ttf1", "ttf2")?,
            ttr: pair("ttr1", "ttr2")?,
            tmje: pair("tmje1", "tmje2")?,
            tmjc: pair("tmjc1", "tmjc2")?,
            tmjs: pair("tmjs1", "tmjs2")?,
            tns: pair("tns1", "tns2")?,
            tis: pair("tis1", "tis2")?,
            tise: pair("tise1", "tise2")?,
            tisc: pair("tisc1", "tisc2")?,
            tiss: pair("tiss1", "tiss2")?,
            ctc: get(m, "ctc")?,
            cte: get(m, "cte")?,
            cts: get(m, "cts")?,
            tvjc: get(m, "tvjc")?,
            tvje: get(m, "tvje")?,
            tvjs: get(m, "tvjs")?,
            kf: get(m, "kf")?,
            af: get(m, "af")?,
        };
        // Grading coefficients of 1 or more divide by zero in the depletion
        // charge; C clamps them (with tempco) to 0.999, the port additionally
        // rejects the raw value as malformed. FC >= 1 is rejected likewise.
        if model.mje >= 1. || model.mjc >= 1. || model.mjs >= 1. || fc >= 1. {
            return Err(SpiceError::circuit("BJT MJE/MJC/MJS/FC must be below 1"));
        }
        Ok(model)
    }
}

/// Instance geometry and temperature.
#[derive(Debug, Clone, Copy)]
struct Instance {
    area: Real,
    areab: Real,
    areac: Real,
    multiplier: Real,
    /// Kelvin, `None` for the circuit temperature plus `dtemp`.
    temp: Option<Real>,
    dtemp: Real,
}

/// Quantities at the instance temperature, scaled by area (`bjttemp.c`).
#[derive(Debug, Clone, Copy)]
struct Thermal {
    vt: Real,
    /// `BJTtSatCur` (`area * IS` at temperature), which sets `BJTtVcrit`.
    is: Real,
    is_be: Real,
    is_bc: Real,
    ise: Real,
    isc: Real,
    iss: Option<Real>,
    nf: Real,
    nr: Real,
    ne: Real,
    nc: Real,
    ns: Real,
    nkf: Option<Real>,
    bf: Real,
    br: Real,
    inv_vaf: Real,
    inv_var: Real,
    inv_ikf: Real,
    inv_ikr: Real,
    rb: Real,
    rbm: Real,
    irb: Real,
    gc: Real,
    ge: Real,
    tf: Real,
    tr: Real,
    xtf: Real,
    vtf_factor: Real,
    itf: Real,
    cje: Real,
    be: DepletionLaw,
    cjc: Real,
    bc: DepletionLaw,
    xcjc: Real,
    cjs: Real,
    vjs: Real,
    mjs: Real,
}

/// `bjttemp.c`'s silicon band-gap built-in-potential shift `pbfact` at `t`.
fn pbfact(t: Real) -> Real {
    let vt = K_OVER_Q * t;
    let egfet = band_gap(t);
    let arg = -egfet / (2. * BOLTZMANN * t) + 1.115_087_7 / (BOLTZMANN * (REFTEMP + REFTEMP));
    potential_shift(vt, t / REFTEMP, arg)
}

impl Model {
    #[expect(
        clippy::too_many_lines,
        reason = "keeps the reference numerical routine in the original statement and rounding order"
    )]
    fn thermal(
        &self,
        instance: &Instance,
        vertical: bool,
        context: &ModelContext,
    ) -> SpiceResult<Thermal> {
        let tnom = self
            .tnom
            .unwrap_or(context.nominal_temperature + CELSIUS_TO_KELVIN);
        let t = instance
            .temp
            .unwrap_or(context.temperature + CELSIUS_TO_KELVIN + instance.dtemp);
        if !(t.is_finite() && t > 0. && tnom.is_finite() && tnom > 0.) {
            return Err(SpiceError::circuit("invalid BJT temperature"));
        }
        let Instance {
            area, areab, areac, ..
        } = *instance;
        // Base/collector area of the base-collector and substrate junctions:
        // AREAB for a vertical device, AREAC for a lateral one (bjttemp.c).
        let (bc_area, sub_area) = if vertical {
            (areab, areac)
        } else {
            (areac, areab)
        };
        let dt = t - tnom;
        let vt = t * K_OVER_Q;
        let nominal = PotentialTemperature::new(tnom, pbfact(tnom));
        let actual = PotentialTemperature::new(t, pbfact(t));
        let ratlog = (t / tnom).ln();
        let ratio1 = t / tnom - 1.;
        let factlog = ratio1 * self.eg / vt + self.xti * ratlog;
        let (is, is_be, mut is_bc, mut iss) = if self.tlev == 3 {
            let exponent = poly(self.tis, dt);
            let is = area * self.is.powf(exponent);
            let (be, bc) = match self.ibe_ibc {
                Some((ibe, ibc)) => (area * ibe.powf(exponent), ibc.powf(exponent)),
                None => (is, is / area),
            };
            let iss = self.iss.map(|v| v.powf(poly(self.tiss, dt)));
            (is, be, bc, iss)
        } else {
            let is = area * self.is * factlog.exp();
            let (be, bc) = match self.ibe_ibc {
                Some((ibe, ibc)) => (
                    area * ibe * (factlog / self.nf).exp(),
                    ibc * (factlog / self.nr).exp(),
                ),
                None => (is, is / area),
            };
            let iss = self.iss.map(|v| v * (factlog / self.ns).exp());
            (is, be, bc, iss)
        };
        is_bc *= bc_area;
        if let Some(value) = iss.as_mut() {
            *value *= if self.ibe_ibc.is_some() {
                sub_area
            } else {
                area
            };
        }
        let bfactor = match self.tlev {
            0 => (ratlog * self.xtb).exp(),
            1 => 1. + self.xtb * dt,
            _ => 1.,
        };
        let bf = self
            .tbf
            .map_or(self.bf * bfactor, |c| self.bf * poly(c, dt));
        let br = self
            .tbr
            .map_or(self.br * bfactor, |c| self.br * poly(c, dt));
        let (ise, isc) = if self.tlev == 3 {
            (
                area * self.ise.powf(poly(self.tise, dt)),
                self.isc.powf(poly(self.tisc, dt)),
            )
        } else {
            (
                area * self.ise * (factlog / self.ne).exp() / bfactor,
                self.isc * (factlog / self.nc).exp() / bfactor,
            )
        };
        let isc = isc * bc_area;
        let inverse = |value: Real, c: TempCo| {
            if value == 0. {
                0.
            } else {
                1. / (value * poly(c, dt))
            }
        };
        // bjttemp.c clamps the temperature-adjusted grading to 0.999.
        let grading = |value: Real, c: TempCo| (value * poly(c, dt)).min(0.999);
        let mje = grading(self.mje, self.tmje);
        let mjc = grading(self.mjc, self.tmjc);
        let mjs = grading(self.mjs, self.tmjs);
        let junction = |cap: Real, pot: Real, m: Real, ct: Real, tv: Real| {
            if self.tlevc == 1 {
                (cap * (1. + ct * dt), pot - tv * dt)
            } else {
                let correction = actual.junction(&nominal, pot);
                let mut cap = cap / correction.nominal_denominator(m);
                cap *= correction.actual_factor(m);
                (cap, correction.potential)
            }
        };
        let (cje, vje) = junction(self.cje, self.vje, mje, self.cte, self.tvje);
        let (cjc, vjc) = junction(self.cjc, self.vjc, mjc, self.ctc, self.tvjc);
        let (cjs, vjs) = junction(self.cjs, self.vjs, mjs, self.cts, self.tvjs);
        let thermal = Thermal {
            vt,
            is,
            is_be,
            is_bc,
            ise,
            isc,
            iss,
            nf: self.nf * poly(self.tnf, dt),
            nr: self.nr * poly(self.tnr, dt),
            ne: self.ne * poly(self.tne, dt),
            nc: self.nc * poly(self.tnc, dt),
            ns: self.ns * poly(self.tns, dt),
            nkf: self.nkf,
            bf,
            br,
            inv_vaf: inverse(self.vaf, self.tvaf),
            inv_var: inverse(self.var, self.tvar),
            inv_ikf: inverse(self.ikf, self.tikf) / area,
            inv_ikr: inverse(self.ikr, self.tikr) / area,
            rb: self.rb * poly(self.trb, dt) / area,
            rbm: self.rbm * poly(self.trm, dt) / area,
            irb: self.irb * poly(self.tirb, dt) * area,
            gc: inverse(self.rc, self.trc) * area,
            ge: inverse(self.re, self.tre) * area,
            tf: self.tf * poly(self.ttf, dt),
            tr: self.tr * poly(self.ttr, dt),
            xtf: self.xtf,
            vtf_factor: if self.vtf == 0. {
                0.
            } else {
                1. / (self.vtf * 1.44)
            },
            itf: self.itf * poly(self.titf, dt) * area,
            cje: cje * area,
            be: DepletionLaw::new(vje, mje, self.fc),
            cjc: cjc * bc_area,
            bc: DepletionLaw::new(vjc, mjc, self.fc),
            xcjc: self.xcjc,
            cjs: cjs * sub_area,
            vjs,
            mjs,
        };
        thermal.validate()?;
        Ok(thermal)
    }
}

/// State slots (C `bjtdefs.h` names): each charge is followed by its
/// integrated current, as `NIintegrate` stores them, then the load's branch
/// voltages.
mod slot {
    /// `BJTqbe`, followed by `BJTcqbe`.
    pub(super) const QBE: usize = 0;
    /// `BJTqbc`, followed by `BJTcqbc`.
    pub(super) const QBC: usize = 2;
    /// `BJTqsub`, followed by `BJTcqsub`.
    pub(super) const QSUB: usize = 4;
    /// `BJTqbx`, followed by `BJTcqbx`.
    pub(super) const QBX: usize = 6;
    /// `BJTvbe` (limited).
    pub(super) const VBE: usize = 8;
    /// `BJTvbc` (limited).
    pub(super) const VBC: usize = 9;
    /// `BJTvsub` (limited).
    pub(super) const VSUB: usize = 10;
    /// `BJTvbx` (never limited).
    pub(super) const VBX: usize = 11;
    /// Number of slots.
    pub(super) const COUNT: usize = 12;
}
/// `bjtdefs.h` `VCRIT_DISABLED`: the substrate critical voltage without ISS.
const SUBSTRATE_VCRIT_DISABLED: Real = 50.;

impl Thermal {
    fn validate(&self) -> SpiceResult<()> {
        let values = [
            self.vt,
            self.is_be,
            self.is_bc,
            self.ise,
            self.isc,
            self.iss.unwrap_or(0.),
            self.nf,
            self.nr,
            self.ne,
            self.nc,
            self.ns,
            self.bf,
            self.br,
            self.inv_vaf,
            self.inv_var,
            self.inv_ikf,
            self.inv_ikr,
            self.rb,
            self.rbm,
            self.irb,
            self.gc,
            self.ge,
            self.tf,
            self.tr,
            self.itf,
            self.cje,
            self.be.potential,
            self.cjc,
            self.bc.potential,
            self.cjs,
            self.vjs,
            self.be.f1,
            self.be.f2,
            self.be.f3,
            self.bc.f1,
            self.bc.f2,
            self.bc.f3,
        ];
        if values.iter().any(|v| !v.is_finite())
            || [
                self.nf,
                self.nr,
                self.ne,
                self.nc,
                self.ns,
                self.be.potential,
                self.bc.potential,
                self.vjs,
            ]
            .iter()
            .any(|v| *v <= 0.)
            || self.bf == 0.
            || self.br == 0.
        {
            return Err(SpiceError::Numerical {
                context: "BJT temperature model".into(),
                message: "nonfinite or out-of-domain temperature-adjusted parameter".into(),
            });
        }
        Ok(())
    }
}

/// Normalized branch voltages: `vbe`, `vbc` across the intrinsic junctions,
/// `vbx` from the external base to the internal collector and `vsub` across
/// the substrate junction, each multiplied by the device polarity (and the
/// substrate geometry sign for `vsub`).
#[derive(Debug, Clone, Copy, Default, PartialEq)]
struct Bias {
    vbe: Real,
    vbc: Real,
    vbx: Real,
    vsub: Real,
}

/// The device's node roles. Primes are the internal (or, without the
/// corresponding resistance, external) nodes; `substrate_connection` is the
/// internal collector of a vertical device or the internal base of a lateral one.
#[derive(Debug, Clone, Copy)]
struct Nodes {
    c: NodeId,
    b: NodeId,
    e: NodeId,
    s: NodeId,
    cp: NodeId,
    bp: NodeId,
    ep: NodeId,
    sc: NodeId,
}

/// Gummel-Poon level-1 BJT (`bjtload.c`), NPN or PNP.
#[derive(Debug)]
pub struct Bjt {
    name: String,
    /// The model card's name, for C's `.noise` visiting order.
    model_name: String,
    /// External terminals (three, or four with an explicit substrate) followed
    /// by the internal nodes, in the order `bjtsetup.c` creates them.
    terminals: Vec<NodeId>,
    nodes: Nodes,
    pol: Real,
    /// `SUBS`: +1 vertical, -1 lateral.
    subs: Real,
    model: Model,
    instance: Instance,
    derived: TemperatureCache<Thermal>,
    /// `OFF` and `ICVBE`/`ICVCE` (C `BJToff`, `BJTicVBE`, `BJTicVCE`).
    initial: crate::devices::initial::InstanceInitial,
}

fn reject_unported(parameters: &[ParameterAssignment]) -> SpiceResult<()> {
    for parameter in parameters {
        if let Some((_, reference)) = UNPORTED_MODEL
            .iter()
            .find(|(name, _)| name.eq_ignore_ascii_case(&parameter.name))
        {
            return Err(SpiceError::not_yet_ported(
                format!(
                    "{}: BJT model parameter '{}'",
                    parameter.location, parameter.name
                ),
                format!("src/spicelib/devices/{reference}"),
            ));
        }
    }
    Ok(())
}

impl Bjt {
    pub(crate) fn instantiate(
        i: &DeviceInstance,
        nodes: &mut NodeAppend<'_>,
        model: &ResolvedModel<'_>,
        context: &ModelContext,
    ) -> SpiceResult<Box<dyn Device>> {
        if model.levels().backend != crate::devices::ModelBackend::Bjt
            || !(3..=4).contains(&i.nodes.len())
        {
            return Err(SpiceError::Unsupported {
                feature: "BJT backend requires a Gummel-Poon model and 3/4 terminals".into(),
                location: Some(i.location.clone()),
            });
        }
        let card = model.card();
        reject_unported(&card.parameters)?;
        // `inpgmod.c::create_model`: a `.model` setter outside the model table
        // that is a settable instance parameter becomes a model-level instance
        // default, applied before the instance card's own setters.
        let (model_side, model_defaults) = crate::devices::schema::split_model_setters(
            card.parameters
                .iter()
                .filter(|p| !p.name.eq_ignore_ascii_case("level")),
            &MODEL_SCHEMA,
            &INSTANCE_SCHEMA,
        );
        let assignments: Vec<_> = model_side.into_iter().map(canonical).collect();
        let values = MODEL_SCHEMA.validate(&assignments, &card.location)?;
        let parameters = Model::from_values(&values)?;
        let excess_phase = get(&values, "ptf")? * parameters.tf;
        if excess_phase != 0. {
            return Err(SpiceError::not_yet_ported(
                format!("{}: BJT excess phase (PTF with TF != 0)", card.location),
                "src/spicelib/devices/bjt/bjtload.c (excess phase), bjtacld.c",
            ));
        }
        let (initial, instance_setters) =
            crate::devices::initial::split(&i.parameters, &IC_COMPONENTS)?;
        let instance_values = crate::devices::schema::with_defaults(
            &INSTANCE_SCHEMA,
            &model_defaults,
            instance_setters.iter(),
            &i.location,
        )?;
        let area = get(&instance_values, "area")?;
        let instance = Instance {
            area,
            areab: given(&instance_values, "areab").unwrap_or(area),
            areac: given(&instance_values, "areac").unwrap_or(area),
            multiplier: get(&instance_values, "m")?,
            temp: given(&instance_values, "temp").map(|t| t + CELSIUS_TO_KELVIN),
            dtemp: get(&instance_values, "dtemp")?,
        };
        let pol = if model.family() == ModelFamily::Pnp {
            -1.
        } else {
            1.
        };
        // bjtsetup.c: SUBS other than +1/-1 takes the polarity default,
        // vertical for NPN and lateral for PNP.
        let subs = match given(&values, "subs") {
            Some(v) if v == 1. || v == -1. => v,
            _ => pol,
        };
        // Validate the temperature model and one evaluation before interning.
        let thermal = parameters.thermal(&instance, subs > 0., context)?;
        c_point(&thermal, Bias::default(), context.gmin)?;
        let (terminals, external, cp, bp, ep) = nodes.transaction(|staged| {
            let external: Vec<_> = i
                .nodes
                .iter()
                .map(|n| staged.intern(n))
                .collect::<SpiceResult<Vec<_>>>()?;
            let mut terminals = external.clone();
            let mut internal = |resistance: Real, suffix: &str, external: NodeId| {
                if resistance == 0. {
                    return Ok(external);
                }
                let name = format!("{}#{suffix}", i.name);
                if staged.get(&name).is_some() {
                    return Err(SpiceError::circuit(format!(
                        "BJT internal-node name collision: {name}"
                    )));
                }
                let node = staged.intern(&name)?;
                staged.set_kind(node, NodeKind::Internal)?;
                terminals.push(node);
                Ok(node)
            };
            // bjtsetup.c creates collCX, base, emitter in this order.
            let cp = internal(parameters.rc, "collCX", external[0])?;
            let bp = internal(parameters.rb, "base", external[1])?;
            let ep = internal(parameters.re, "emitter", external[2])?;
            Ok((terminals, external, cp, bp, ep))
        })?;
        let nodes = Nodes {
            c: external[0],
            b: external[1],
            e: external[2],
            s: external.get(3).copied().unwrap_or(NodeId::GROUND),
            cp,
            bp,
            ep,
            sc: if subs > 0. { cp } else { bp },
        };
        Ok(Box::new(Self {
            name: i.name.clone(),
            model_name: card.name.clone(),
            terminals,
            nodes,
            pol,
            subs,
            model: parameters,
            instance,
            derived: TemperatureCache::from_value(TemperatureKey::context(context), thermal),
            initial,
        }))
    }

    fn thermal(&self, context: &ModelContext) -> SpiceResult<Thermal> {
        self.derived
            .get_or_derive(TemperatureKey::context(context), || {
                self.model.thermal(&self.instance, self.subs > 0., context)
            })
    }

    /// `bjtload.c`'s `uic` initial bias (`MODEINITJCT` under `MODETRANOP |
    /// MODEUIC`): `vbe = type * ICVBE`, `vbc = vbx = vbe - type * ICVCE`,
    /// `vsub = 0`. An unset `ICVBE`/`ICVCE` is the external base-emitter /
    /// collector-emitter voltage of the solution (`bjtgetic.c`).
    fn uic_bias(&self, voltage: impl Fn(NodeId) -> Real) -> Bias {
        let n = self.nodes;
        let icvbe = self.initial.values[0].unwrap_or_else(|| voltage(n.b) - voltage(n.e));
        let icvce = self.initial.values[1].unwrap_or_else(|| voltage(n.c) - voltage(n.e));
        let vbe = self.pol * icvbe;
        let vbc = vbe - self.pol * icvce;
        Bias {
            vbe,
            vbc,
            vbx: vbc,
            vsub: 0.,
        }
    }

    const fn bias_nodes(&self) -> [NodeId; 6] {
        let n = self.nodes;
        [n.bp, n.ep, n.cp, n.b, n.s, n.sc]
    }

    fn bias_values(&self, [bp, ep, cp, b, s, sc]: [Real; 6]) -> Bias {
        Bias {
            vbe: self.pol * (bp - ep),
            vbc: self.pol * (bp - cp),
            vbx: self.pol * (b - cp),
            vsub: self.pol * self.subs * (s - sc),
        }
    }

    fn bias(&self, voltage: impl Fn(NodeId) -> Real) -> Bias {
        self.bias_values(self.bias_nodes().map(voltage))
    }

    /// `bjtload.c`'s branch voltages `(vbe, vbc, vbx, vsub)` for this load,
    /// in C's order of mode tests:
    ///
    /// - `MODEINITTRAN`: `vbe`, `vbc`, `vbx` from `CKTstate1`, `vsub` from
    ///   the solution (zero with `uic`, where C's `type*(ICVBE - ICVCE)` is
    ///   the `vbx` the `uic` load stored in `CKTstate1`);
    /// - the `uic` initial load (`MODEINITJCT | MODETRANOP | MODEUIC`):
    ///   `vbe = type*ICVBE`, `vbc = vbx = vbe - type*ICVCE`, `vsub = 0`;
    /// - `MODEINITJCT`: `vbe = tVcrit`, the rest zero (all zero when `OFF`,
    ///   which also holds them at zero through `MODEINITFIX`);
    /// - otherwise the solution's voltages (in `MODEINITPRED` the `DEVpred`
    ///   extrapolation of each stored voltage, with `xfact = 1` in a `.dc`
    ///   sweep where `dctrcurv.c` keeps `CKTdelta = CKTdeltaOld[1]`), then
    ///   `DEVpnjlim` of `vbe`, `vbc` against `tVcrit` and of `vsub` against
    ///   `tSubVcrit` (`VCRIT_DISABLED` without ISS), each relative to the
    ///   previous load's value (`CKTstate1` in `MODEINITPRED`).
    ///
    /// Bypass is never taken (`CKTbypass` defaults to off).
    fn voltages(
        &self,
        t: &Thermal,
        limiter: &mut Limiter,
        context: &StampContext<'_>,
    ) -> SpiceResult<(Real, Real, Real, Real)> {
        let states = &context.states;
        let voltage = |node| context.node_voltage(node);
        let raw = self.bias(voltage);
        let vcrit = limiting::critical_voltage(t.vt, t.is);
        match limiter.mode() {
            Linearization::InitialTransient => {
                let state1 = |slot, fallback| states.accepted(1, slot).unwrap_or(fallback);
                let vsub = if states.uic() { 0. } else { raw.vsub };
                return Ok((
                    state1(slot::VBE, raw.vbe),
                    state1(slot::VBC, raw.vbc),
                    state1(slot::VBX, raw.vbx),
                    vsub,
                ));
            }
            Linearization::InitialConditions => {
                let start = self.uic_bias(voltage);
                return Ok((start.vbe, start.vbc, start.vbx, 0.));
            }
            _ => {}
        }
        if limiter.holds_off(states, self.initial.off) {
            return Ok((0., 0., 0., 0.));
        }
        match limiter.mode() {
            Linearization::Initial => Ok((vcrit, 0., 0., 0.)),
            Linearization::Limited => {
                let (mut vbe, mut vbc, vbx, mut vsub) =
                    if states.phase() == crate::devices::IterationPhase::Predict {
                        let xfact = context
                            .integration
                            .map_or(1., crate::maths::Coefficients::predictor_ratio);
                        let predicted = limiting::predicted_values(
                            states,
                            [slot::VBE, slot::VBC, slot::VBX, slot::VSUB],
                            xfact,
                        )
                        .ok_or_else(|| SpiceError::circuit("missing accepted BJT voltages"))?;
                        predicted.into()
                    } else {
                        (raw.vbe, raw.vbc, raw.vbx, raw.vsub)
                    };
                let vt = t.vt;
                let sub_vcrit = t.iss.map_or(SUBSTRATE_VCRIT_DISABLED, |iss| {
                    limiting::critical_voltage(vt, iss)
                });
                vbe = limiter.pn_junction(vbe, limiter.previous(states, slot::VBE), vt, vcrit);
                vbc = limiter.pn_junction(vbc, limiter.previous(states, slot::VBC), vt, vcrit);
                vsub =
                    limiter.pn_junction(vsub, limiter.previous(states, slot::VSUB), vt, sub_vcrit);
                Ok((vbe, vbc, vbx, vsub))
            }
            _ => Ok((raw.vbe, raw.vbc, raw.vbx, raw.vsub)),
        }
    }
}

impl Device for Bjt {
    /// `bjtsetup.c` without quasi-saturation (`RCO`, which the port rejects),
    /// so `collCX` is the collector prime; a three-terminal device's
    /// substrate is ground and allocates nothing.
    fn setup_matrix(
        &self,
        setup: &mut crate::devices::setup::MatrixSetup<'_>,
    ) -> SpiceResult<bool> {
        let Nodes {
            c,
            b,
            e,
            s,
            cp,
            bp,
            ep,
            sc,
        } = self.nodes;
        let cx = cp;
        for (row, col) in [
            (c, cx),
            (b, bp),
            (e, ep),
            (cx, c),
            (cp, bp),
            (cp, ep),
            (bp, b),
            (bp, cp),
            (bp, ep),
            (ep, e),
            (ep, cp),
            (ep, bp),
            (c, c),
            (b, b),
            (e, e),
            (cp, cp),
            (bp, bp),
            (ep, ep),
            (s, s),
            (sc, s),
            (s, sc),
            (b, cp),
            (cp, b),
            (cx, cx),
        ] {
            setup.nodes(row, col);
        }
        Ok(true)
    }
    /// `bjt.c`'s `IFparm` tables under C's `sgen` walk (#164): the record
    /// replay of `bjtsetup.c`/`bjttemp.c`/`bjtmpar.c`/`bjtparam.c` with the
    /// load being this device rebuilt from those records.
    ///
    /// Refused for a card with `RC != 0`. `bjtsetup.c` then creates the
    /// internal `collCX` node, and C's walk marks `VO`/`GAMMA`/`QCO` given as
    /// it passes them, which switches on the Kull quasi-saturation terms of
    /// `bjtload.c` in the *perturbed* loads. The port does not model them (the
    /// card parameters are `UNPORTED_MODEL`), and the effect is not a rounding
    /// difference: measured on `rc=40`, `vbe` at 0.7 V, C answers
    /// `v(q1:gamma) = 6.52e-1` where the port has to answer 0. Every other
    /// column of that card agrees, so the refusal is narrow - and a card
    /// without `RC` has no `collCX` node for the terms to use, which is why
    /// the rest of the tables verify.
    fn sensitivity(
        &self,
        context: &ModelContext,
    ) -> SpiceResult<Box<dyn crate::devices::sensitivity::DeviceSensitivity + '_>> {
        if self.model.rc != 0. {
            return Err(SpiceError::not_yet_ported(
                format!("{}: BJT sensitivity with RC != 0", self.name),
                "src/spicelib/devices/bjt/bjtload.c's Kull quasi-saturation \
                 (BJTintCollResist/BJTepiSatVoltage/BJTepiDoping/BJTepiCharge)",
            ));
        }
        Ok(Box::new(BjtSensitivity::new(self, context)))
    }
    fn observation_parameter(
        &self,
        key: &str,
        context: &ModelContext,
    ) -> SpiceResult<Option<Real>> {
        let p = &self.instance;
        Ok(match key {
            "area" => Some(p.area),
            "areab" => Some(p.areab),
            "areac" => Some(p.areac),
            "m" => Some(p.multiplier),
            "temp" => Some(p.temp.map_or(context.temperature + p.dtemp, |t| t - 273.15)),
            "dtemp" => Some(p.dtemp),
            _ => None,
        })
    }
    fn name(&self) -> &str {
        &self.name
    }
    fn designator(&self) -> char {
        'q'
    }
    fn terminals(&self) -> &[NodeId] {
        &self.terminals
    }
    fn is_nonlinear(&self) -> bool {
        true
    }
    fn has_start_settings(&self) -> bool {
        self.initial.off
    }
    /// `qbe`, `qbc`, `qsub`, `qbx`, each followed by its current, then the
    /// limited `vbe`, `vbc`, `vsub` and the unlimited `vbx` (see [`slot`]).
    fn state_count(&self) -> usize {
        slot::COUNT
    }
    /// `bjttrunc.c`: `qbe`, `qbc`, `qsub`, and `qbx` only when `XCJC < 1`.
    fn truncation_slots(&self) -> Vec<usize> {
        if self.model.xcjc < 1. {
            vec![0, 2, 4, 6]
        } else {
            vec![0, 2, 4]
        }
    }
    /// `bjt.c` `BJTpTable`: AREA, AREAB, AREAC, M, TEMP and DTEMP, which
    /// `bjttemp.c`/`bjtload.c` re-derive (`dctrcurv.c` `DCTsetInstParam`).
    fn instance_parameter(&self, keyword: &str) -> Option<&'static str> {
        [
            "icvbe", "icvce", "area", "areab", "areac", "m", "temp", "dtemp",
        ]
        .into_iter()
        .find(|name| name.eq_ignore_ascii_case(keyword))
    }
    /// `BJTparam` then `BJTtemp`. A swept AREA leaves AREAB/AREAC at the
    /// values `bjtsetup.c` defaulted them to from the card's AREA, exactly as
    /// C does: only `BJTsetup` copies AREA into an ungiven AREAB/AREAC.
    fn with_instance_parameter(
        &self,
        parameter: &str,
        value: Real,
        context: &ModelContext,
    ) -> SpiceResult<Box<dyn Device>> {
        use crate::devices::sweep::check_swept;
        let mut initial = self.initial.clone();
        if let Some(index) = ["icvbe", "icvce"].iter().position(|k| *k == parameter) {
            check_swept(&self.name, parameter, value, true, "finite")?;
            initial.values[index] = Some(value);
        }
        let mut instance = self.instance;
        let positive = || check_swept(&self.name, parameter, value, value > 0., "positive");
        match parameter {
            "icvbe" | "icvce" => {}
            "area" => {
                positive()?;
                instance.area = value;
            }
            "areab" => {
                positive()?;
                instance.areab = value;
            }
            "areac" => {
                positive()?;
                instance.areac = value;
            }
            "m" => {
                positive()?;
                instance.multiplier = value;
            }
            "temp" => {
                check_swept(
                    &self.name,
                    parameter,
                    value,
                    value + CELSIUS_TO_KELVIN > 0.,
                    "above absolute zero",
                )?;
                instance.temp = Some(value + CELSIUS_TO_KELVIN);
            }
            "dtemp" => {
                check_swept(&self.name, parameter, value, true, "finite")?;
                instance.dtemp = value;
            }
            _ => {
                return Err(SpiceError::circuit(format!(
                    "{}: BJT parameter {parameter} cannot be swept",
                    self.name
                )));
            }
        }
        let device = Self {
            name: self.name.clone(),
            model_name: self.model_name.clone(),
            terminals: self.terminals.clone(),
            nodes: self.nodes,
            pol: self.pol,
            subs: self.subs,
            model: self.model,
            instance,
            derived: TemperatureCache::new(),
            initial,
        };
        c_point(&device.thermal(context)?, Bias::default(), context.gmin)?;
        Ok(Box::new(device))
    }
    /// `bjtload.c`: [`Bjt::voltages`], [`c_point`], the transient companions
    /// (`NIintegrate` of `qbe` and `qbc` folded into `gpi`, `gmu`, `cb`,
    /// `cc` and `geqcb *= ag0`, then `qsub`, `qbx`; `MODEINITTRAN` copies
    /// of the charges and currents into `CKTstate1`), and the right-hand
    /// side and matrix stamps in C's sequence and expressions.
    #[expect(
        clippy::too_many_lines,
        reason = "keeps the reference numerical routine in the original statement and rounding order"
    )]
    fn stamp(&self, context: &mut StampContext<'_>) -> SpiceResult<()> {
        if context.mode.is_ac() {
            return Err(SpiceError::circuit("BJT AC requires small-signal assembly"));
        }
        let model_context = context.model_context();
        let t = self.thermal(&model_context)?;
        let mut limiter = Limiter::new(&context.states);
        let (vbe, vbc, vbx, vsub) = self.voltages(&t, &mut limiter, context)?;
        let p = c_point(
            &t,
            Bias {
                vbe,
                vbc,
                vbx,
                vsub,
            },
            context.gmin,
        )?;
        let Transport {
            mut cc,
            mut cb,
            mut gpi,
            mut gmu,
            gm,
            go,
            gx,
        } = p.transport;
        let Junction {
            conductance: gdsub,
            current: cdsub,
            ..
        } = p.substrate;
        // BJTconvTest for an `off` instance held in MODEINITFIX: the held
        // currents against their linearization at the iterate's voltages.
        {
            let raw = self.bias(|node| context.node_voltage(node));
            let delvbe = raw.vbe - vbe;
            let delvbc = raw.vbc - vbc;
            let cchat = cc + (gm + go) * delvbe - (go + gmu) * delvbc;
            let cbhat = cb + gpi * delvbe + gmu * delvbc;
            limiter.test_held(&context.states, &[(cc, cchat), (cb, cbhat)]);
        }
        let be = ChargeBranch::new(slot::QBE, p.charges.qbe, p.charges.capbe);
        let bc = ChargeBranch::new(slot::QBC, p.charges.qbc, p.charges.capbc);
        let bx = ChargeBranch::new(slot::QBX, p.charges.qbx, p.charges.capbx);
        let sub = ChargeBranch::new(slot::QSUB, p.charges.qsub, p.charges.capsub);
        let mut geqcb = 0.;
        let (mut gcsub, mut geqbx) = (0., 0.);
        let (cqsub, cqbx);
        if let Some(coefficients) = context.integration {
            let crate::devices::AnalysisMode::Transient { dt, .. } = context.mode else {
                return Err(SpiceError::circuit(
                    "BJT charge companion outside transient",
                ));
            };
            if dt != coefficients.dt() {
                return Err(SpiceError::circuit("BJT charge timestep mismatch"));
            }
            let initial = context.states.phase() == crate::devices::IterationPhase::Predict
                && context.states.initial_transient();
            if initial {
                for branch in [be, bc, bx, sub] {
                    branch.seed(&mut context.states)?;
                }
            }
            let be_comp = be.integrate(&mut context.states, coefficients)?;
            geqcb = p.charges.geqcb * coefficients.ag()[0];
            gpi += be_comp.conductance;
            cb += be_comp.derivative;
            let bc_comp = bc.integrate(&mut context.states, coefficients)?;
            gmu += bc_comp.conductance;
            cb += bc_comp.derivative;
            cc -= bc_comp.derivative;
            if initial {
                be.seed_derivative(&mut context.states, be_comp.derivative)?;
                bc.seed_derivative(&mut context.states, bc_comp.derivative)?;
            }
            let sub_comp = sub.integrate(&mut context.states, coefficients)?;
            (gcsub, cqsub) = (sub_comp.conductance, sub_comp.derivative);
            let bx_comp = bx.integrate(&mut context.states, coefficients)?;
            (geqbx, cqbx) = (bx_comp.conductance, bx_comp.derivative);
            if initial {
                bx.seed_derivative(&mut context.states, cqbx)?;
                sub.seed_derivative(&mut context.states, cqsub)?;
            }
        } else {
            if context.mode.is_transient() {
                return Err(SpiceError::circuit(
                    "BJT transient needs companion integration",
                ));
            }
            for branch in [be, bc] {
                branch.zero_derivative(&mut context.states)?;
            }
            // C never writes `cqsub`/`cqbx` outside transient and
            // `MODEINITSMSIG`, yet reads them into `ceqsub`/`ceqbx`: a DC load
            // after a small-signal analysis of the same deck sees the
            // capacitances that analysis stored there.
            cqsub = sub.retain_derivative(&mut context.states)?;
            cqbx = bx.retain_derivative(&mut context.states)?;
        }
        for branch in [be, bc, bx, sub] {
            branch.store(&mut context.states)?;
        }
        // Load the current excitation vector and the matrix.
        let m = self.instance.multiplier;
        let ty = self.pol;
        let ttype = self.pol * self.subs;
        let gcpr = t.gc;
        let gepr = t.ge;
        let geqsub = gcsub + gdsub;
        let ceqsub = ttype * (cqsub + cdsub - vsub * geqsub);
        let ceqbx = ty * (cqbx - vbx * geqbx);
        let ceqbe = ty * (cc + cb - vbe * (gm + go + gpi) + vbc * (go - geqcb));
        let ceqbc = ty * (-cc + vbe * (gm + go) - vbc * (gmu + go));
        let Nodes {
            c,
            b,
            e,
            s,
            cp,
            bp,
            ep,
            sc,
        } = self.nodes;
        // Without quasi-saturation `collCX` is the collector prime.
        let cx = cp;
        context.stamp_rhs(b, m * (-ceqbx))?;
        context.stamp_rhs(cp, m * (ceqbx + ceqbc))?;
        context.stamp_rhs(sc, m * ceqsub)?;
        context.stamp_rhs(bp, m * (-ceqbe - ceqbc))?;
        context.stamp_rhs(ep, m * (ceqbe))?;
        context.stamp_rhs(s, m * (-ceqsub))?;
        context.stamp_fixed([
            (c, c, m * (gcpr)),
            (b, b, m * (gx + geqbx)),
            (e, e, m * (gepr)),
            (cp, cp, m * (gmu + go + geqbx)),
            (cx, cx, m * (gcpr)),
            (sc, sc, m * (geqsub)),
            (bp, bp, m * (gx + gpi + gmu + geqcb)),
            (ep, ep, m * (gpi + gepr + gm + go)),
            (c, cx, m * (-gcpr)),
            (b, bp, m * (-gx)),
            (e, ep, m * (-gepr)),
            (cx, c, m * (-gcpr)),
            (cp, bp, m * (-gmu + gm)),
            (cp, ep, m * (-gm - go)),
            (bp, b, m * (-gx)),
            (bp, cp, m * (-gmu - geqcb)),
            (bp, ep, m * (-gpi)),
            (ep, e, m * (-gepr)),
            (ep, cp, m * (-go + geqcb)),
            (ep, bp, m * (-gpi - gm - geqcb)),
            (s, s, m * (geqsub)),
            (sc, s, m * (-geqsub)),
            (s, sc, m * (-geqsub)),
            (b, cp, m * (-geqbx)),
            (cp, b, m * (-geqbx)),
        ])?;
        limiter.finish(
            &mut context.states,
            &[
                (slot::VBE, vbe),
                (slot::VBC, vbc),
                (slot::VSUB, vsub),
                (slot::VBX, vbx),
            ],
        )
    }
    /// `bjtacld.c` after C's `MODEINITSMSIG` load: the conductances of the
    /// last operating-point load (whose voltages are the solution's, the
    /// final load never limiting) and the capacitances at `vbe`, `vbc`, `vbx`,
    /// `vsub` of the solution, stamped in C's sequence. The real parts go to
    /// `A`; the `j omega` parts (C's `xc* = cap * omega`) go to `E` as
    /// capacitances.
    fn assemble_small_signal(
        &self,
        context: &mut LinearContext<'_>,
        bias: &Vector,
    ) -> SpiceResult<()> {
        let t = self.thermal(context.model_context)?;
        let unknowns = context.unknowns;
        let voltage = |node: NodeId| {
            unknowns
                .node_row(node)
                .and_then(|r| bias.get(r))
                .unwrap_or(0.)
        };
        let v = self.bias(voltage);
        let p = c_point(&t, v, context.model_context.gmin)?;
        let m = self.instance.multiplier;
        let gcpr = t.gc;
        let gepr = t.ge;
        let Transport {
            gpi,
            gmu,
            gm,
            go,
            gx,
            ..
        } = p.transport;
        let Junction {
            conductance: gdsub, ..
        } = p.substrate;
        // C's `xc*` before the `CKTomega` factor, which the complex assembly
        // applies term by term (`LinearContext::scaled_dynamic_sum`).
        let xcpi = p.charges.capbe;
        let xcmu = p.charges.capbc;
        let xcbx = p.charges.capbx;
        let xcsub = p.charges.capsub;
        let xcmcb = p.charges.geqcb;
        let Nodes {
            c,
            b,
            e,
            s,
            cp,
            bp,
            ep,
            sc,
        } = self.nodes;
        let cx = cp;
        let mut add = |row: NodeId, col: NodeId, value: Real| -> SpiceResult<()> {
            if let (Some(r), Some(k)) = (unknowns.node_row(row), unknowns.node_row(col)) {
                context.system.a.add(r, k, value)?;
            }
            Ok(())
        };
        add(c, c, m * (gcpr))?;
        add(b, b, m * (gx))?;
        add(e, e, m * (gepr))?;
        add(cp, cp, m * (gmu + go))?;
        add(cx, cx, m * (gcpr))?;
        add(bp, bp, m * (gx + gpi + gmu))?;
        add(ep, ep, m * (gpi + gepr + gm + go))?;
        add(c, cx, m * (-gcpr))?;
        add(b, bp, m * (-gx))?;
        add(e, ep, m * (-gepr))?;
        add(cx, c, m * (-gcpr))?;
        add(cp, bp, m * (-gmu + gm))?;
        add(cp, ep, m * (-gm - go))?;
        add(bp, b, m * (-gx))?;
        add(bp, cp, m * (-gmu))?;
        add(bp, ep, m * (-gpi))?;
        add(ep, e, m * (-gepr))?;
        add(ep, cp, m * (-go))?;
        add(ep, bp, m * (-gpi - gm))?;
        add(sc, sc, m * (gdsub))?;
        add(s, s, m * (gdsub))?;
        add(sc, s, m * (-gdsub))?;
        add(s, sc, m * (-gdsub))?;
        // The `j omega` parts in C's order, each element's terms one
        // `m * (x1 + x2 + ...)` sum of `cap * omega` values. C's excess-phase
        // `xgm` is zero (rejected at elaboration) and left out.
        for (row, col, terms) in [
            (b, b, &[xcbx][..]),
            (cp, cp, &[xcmu, xcbx][..]),
            (sc, sc, &[xcsub][..]),
            (bp, bp, &[xcpi, xcmu, xcmcb][..]),
            (ep, ep, &[xcpi][..]),
            (cp, bp, &[-xcmu][..]),
            (bp, cp, &[-xcmu, -xcmcb][..]),
            (bp, ep, &[-xcpi][..]),
            (ep, cp, &[xcmcb][..]),
            (ep, bp, &[-xcpi, -xcmcb][..]),
            (s, s, &[xcsub][..]),
            (sc, s, &[-xcsub][..]),
            (s, sc, &[-xcsub][..]),
            (b, cp, &[-xcbx][..]),
            (cp, b, &[-xcbx][..]),
        ] {
            context.scaled_dynamic_sum(row, col, terms, m)?;
        }
        Ok(())
    }
    /// Pole-zero load: C `bjtpzld.c` (excess phase is rejected at elaboration) equals the AC load with `s` for `j omega`.
    fn assemble_pole_zero(
        &self,
        context: &mut crate::devices::linear::LinearContext<'_>,
        bias: &crate::maths::Vector,
    ) -> crate::primitives::SpiceResult<()> {
        self.assemble_small_signal(context, bias)
    }

    /// `bjtload.c`'s `MODEINITSMSIG` load stores the substrate and B-X'
    /// capacitances in the `cqsub`/`cqbx` current slots, where every later
    /// non-transient load of the deck reads them back as companion currents
    /// (`ceqsub`, `ceqbx`); the other slots it writes are not read again.
    fn small_signal_state(
        &self,
        context: &crate::devices::distortion::DistortionContext<'_>,
        slots: &mut [Real],
    ) -> SpiceResult<()> {
        let thermal = self.thermal(context.model_context)?;
        let [
            base_prime,
            emitter_prime,
            collector_prime,
            base,
            substrate,
            substrate_connection,
        ] = self.bias_nodes();
        let voltage = |node| context.voltage(node);
        let bias = self.bias_values([
            voltage(base_prime)?,
            voltage(emitter_prime)?,
            voltage(collector_prime)?,
            voltage(base)?,
            voltage(substrate)?,
            voltage(substrate_connection)?,
        ]);
        let point = c_point(&thermal, bias, context.model_context.gmin)?;
        for (slot, value) in [
            (slot::QSUB + 1, point.charges.capsub),
            (slot::QBX + 1, point.charges.capbx),
        ] {
            if let Some(target) = slots.get_mut(slot) {
                *target = value;
            }
        }
        Ok(())
    }

    /// `bjtdset.c`/`bjtdisto.c` at the operating point (see the `disto`
    /// submodule for C's simplified distortion model).
    fn distortion(
        &self,
        context: &crate::devices::distortion::DistortionContext<'_>,
    ) -> SpiceResult<crate::devices::distortion::DeviceDistortion> {
        self.distortion_terms(context)
    }

    /// `bjtnoise.c` at the operating point: thermal noise of RC, RB (the
    /// bias-dependent `gx`) and RE at the instance temperature, shot noise of
    /// the collector current `cc` and base current `cb` (C's `BJTcc`/`BJTcb`,
    /// junction gmin included) and the flicker law `m KF abs(cb)^AF / f`
    /// between the internal base and emitter. The quasi-saturation `_rci`
    /// generator is zero: RCO is not ported, so C's `collCX` node is the
    /// internal collector.
    fn noise(
        &self,
        context: &NoiseContext<'_>,
        sources: &mut Vec<NoiseSource>,
    ) -> SpiceResult<DeviceNoise<'_>> {
        let thermal = self.thermal(context.model_context)?;
        let v = self.bias(|node| context.voltage(node));
        let p = c_point(&thermal, v, context.model_context.gmin)?;
        let m = self.instance.multiplier;
        let (cc, cb) = (p.transport.cc, p.transport.cb);
        let temperature = self
            .instance
            .temp
            .unwrap_or(context.model_context.temperature + CELSIUS_TO_KELVIN + self.instance.dtemp);
        let flicker = m * self.model.kf * (self.model.af * cb.abs().max(1e-38).ln()).exp();
        let n = self.nodes;
        let thermal_noise = |conductance: Real| NoiseKind::Thermal {
            conductance,
            temperature,
        };
        sources.extend([
            NoiseSource::new("_rc", [n.cp, n.c], thermal_noise(thermal.gc * m)),
            NoiseSource::new("_rci", [n.cp, n.cp], thermal_noise(0.)),
            NoiseSource::new("_rb", [n.bp, n.b], thermal_noise(p.transport.gx * m)),
            NoiseSource::new("_re", [n.ep, n.e], thermal_noise(thermal.ge * m)),
            NoiseSource::new("_ic", [n.cp, n.ep], NoiseKind::Shot { current: cc * m }),
            NoiseSource::new("_ib", [n.bp, n.ep], NoiseKind::Shot { current: cb * m }),
            NoiseSource::new(
                "_1overf",
                [n.bp, n.ep],
                NoiseKind::Flicker {
                    numerator: flicker,
                    law: FlickerLaw::PerHertz,
                },
            ),
        ]);
        Ok(DeviceNoise::Sources {
            family: NoiseFamily::Bjt,
            model: Some(&self.model_name),
            total: true,
        })
    }
}

// ---- `.sens` (#164) -------------------------------------------------------

/// `bjt.c`'s `BJTMpTable` in C's order, filtered the way `set_param()`
/// (`cktsgen.c`) and `sens_getp()` filter it.
///
/// A DC walk keeps the `IOP`/`IOPU` entries: the `IOPA` capacitance family
/// (`cje`, `vje`, `mje`, `tf`, `xtf`, `vtf`, `itf`, `ptf`, `cjc`, `vjc`,
/// `mjc`, `xcjc`, `tr`, `cjs`, `vjs`, `mjs`) is `IF_AC` and appears only in an
/// AC analysis, `type` is a string, `npn`/`pnp` are flags, `subs`/`tlev`/
/// `tlevc`/`quasimod` are integers, and the `IOPR` aliases (`tref`, `va`,
/// `ik`, `c2`, `vb`, `c4`, `pe`, `me`, `pc`, `mc`, `csub`, `ccs`, `ps`, `ms`)
/// are redundant.
///
/// The list deliberately keeps parameters the port refuses when given
/// (`rco`, `vo`, `gamma`, `qco`, `vg`, `cn`, `d`, the SOA limits and `rth0`):
/// C enumerates them, and their columns are exactly zero on every card
/// measured, so the port reports zero records for them rather than dropping
/// the columns. `rco` is also hard-coded out of `sens_setp()` (`cktsens.c`
/// skips BJT model parameter 147 because setting it allocates nodes).
const MODEL_PARAMETERS: &[SensitivityParameter] = &[
    SensitivityParameter::dc("tnom"),
    SensitivityParameter::dc("is"),
    SensitivityParameter::dc("ibe"),
    SensitivityParameter::dc("ibc"),
    SensitivityParameter::dc("bf"),
    SensitivityParameter::dc("nf"),
    SensitivityParameter::dc("vaf"),
    SensitivityParameter::dc("ikf"),
    SensitivityParameter::dc("ise"),
    SensitivityParameter::dc("ne"),
    SensitivityParameter::dc("br"),
    SensitivityParameter::dc("nr"),
    SensitivityParameter::dc("var"),
    SensitivityParameter::dc("ikr"),
    SensitivityParameter::dc("isc"),
    SensitivityParameter::dc("nc"),
    SensitivityParameter::dc("rb"),
    SensitivityParameter::dc("irb"),
    SensitivityParameter::dc("rbm"),
    SensitivityParameter::dc("re"),
    SensitivityParameter::dc("rc"),
    SensitivityParameter::ac("cje"),
    SensitivityParameter::ac("vje"),
    SensitivityParameter::ac("mje"),
    SensitivityParameter::ac("tf"),
    SensitivityParameter::ac("xtf"),
    SensitivityParameter::ac("vtf"),
    SensitivityParameter::ac("itf"),
    SensitivityParameter::ac("ptf"),
    SensitivityParameter::ac("cjc"),
    SensitivityParameter::ac("vjc"),
    SensitivityParameter::ac("mjc"),
    SensitivityParameter::ac("xcjc"),
    SensitivityParameter::ac("tr"),
    SensitivityParameter::ac("cjs"),
    SensitivityParameter::ac("vjs"),
    SensitivityParameter::ac("mjs"),
    SensitivityParameter::dc("xtb"),
    SensitivityParameter::dc("eg"),
    SensitivityParameter::dc("xti"),
    SensitivityParameter::dc("fc"),
    SensitivityParameter::dc("kf"),
    SensitivityParameter::dc("af"),
    SensitivityParameter::dc("iss"),
    SensitivityParameter::dc("ns"),
    SensitivityParameter::dc("rco"),
    SensitivityParameter::dc("vo"),
    SensitivityParameter::dc("gamma"),
    SensitivityParameter::dc("qco"),
    SensitivityParameter::dc("tbf1"),
    SensitivityParameter::dc("tbf2"),
    SensitivityParameter::dc("tbr1"),
    SensitivityParameter::dc("tbr2"),
    SensitivityParameter::dc("tikf1"),
    SensitivityParameter::dc("tikf2"),
    SensitivityParameter::dc("tikr1"),
    SensitivityParameter::dc("tikr2"),
    SensitivityParameter::dc("tirb1"),
    SensitivityParameter::dc("tirb2"),
    SensitivityParameter::dc("tnc1"),
    SensitivityParameter::dc("tnc2"),
    SensitivityParameter::dc("tne1"),
    SensitivityParameter::dc("tne2"),
    SensitivityParameter::dc("tnf1"),
    SensitivityParameter::dc("tnf2"),
    SensitivityParameter::dc("tnr1"),
    SensitivityParameter::dc("tnr2"),
    SensitivityParameter::dc("trb1"),
    SensitivityParameter::dc("trb2"),
    SensitivityParameter::dc("trc1"),
    SensitivityParameter::dc("trc2"),
    SensitivityParameter::dc("tre1"),
    SensitivityParameter::dc("tre2"),
    SensitivityParameter::dc("trm1"),
    SensitivityParameter::dc("trm2"),
    SensitivityParameter::dc("tvaf1"),
    SensitivityParameter::dc("tvaf2"),
    SensitivityParameter::dc("tvar1"),
    SensitivityParameter::dc("tvar2"),
    SensitivityParameter::dc("ctc"),
    SensitivityParameter::dc("cte"),
    SensitivityParameter::dc("cts"),
    SensitivityParameter::dc("tvjc"),
    SensitivityParameter::dc("tvje"),
    SensitivityParameter::dc("tvjs"),
    SensitivityParameter::dc("titf1"),
    SensitivityParameter::dc("titf2"),
    SensitivityParameter::dc("ttf1"),
    SensitivityParameter::dc("ttf2"),
    SensitivityParameter::dc("ttr1"),
    SensitivityParameter::dc("ttr2"),
    SensitivityParameter::dc("tmje1"),
    SensitivityParameter::dc("tmje2"),
    SensitivityParameter::dc("tmjc1"),
    SensitivityParameter::dc("tmjc2"),
    SensitivityParameter::dc("tmjs1"),
    SensitivityParameter::dc("tmjs2"),
    SensitivityParameter::dc("tns1"),
    SensitivityParameter::dc("tns2"),
    SensitivityParameter::dc("nkf"),
    SensitivityParameter::dc("tis1"),
    SensitivityParameter::dc("tis2"),
    SensitivityParameter::dc("tise1"),
    SensitivityParameter::dc("tise2"),
    SensitivityParameter::dc("tisc1"),
    SensitivityParameter::dc("tisc2"),
    SensitivityParameter::dc("tiss1"),
    SensitivityParameter::dc("tiss2"),
    SensitivityParameter::dc("vg"),
    SensitivityParameter::dc("cn"),
    SensitivityParameter::dc("d"),
    SensitivityParameter::dc("vbe_max"),
    SensitivityParameter::dc("vbc_max"),
    SensitivityParameter::dc("vce_max"),
    SensitivityParameter::dc("pd_max"),
    SensitivityParameter::dc("ic_max"),
    SensitivityParameter::dc("ib_max"),
    SensitivityParameter::dc("te_max"),
    SensitivityParameter::dc("rth0"),
];

/// `bjt.c`'s `BJTpTable` under the same filter: `off`/`sens_area` are flags,
/// `ic` is a vector and the probe entries are ask-only. `icvbe`/`icvce` are
/// `IOPAU` (`IF_AC`), so a DC walk reports six instance parameters.
const INSTANCE_PARAMETERS: &[SensitivityParameter] = &[
    SensitivityParameter::ac("icvbe"),
    SensitivityParameter::ac("icvce"),
    SensitivityParameter::dc("area"),
    SensitivityParameter::dc("areab"),
    SensitivityParameter::dc("areac"),
    SensitivityParameter::dc("m"),
    SensitivityParameter::dc("temp"),
    SensitivityParameter::dc("dtemp"),
];

/// `bjtsetup.c`/`bjttemp.c`/`bjtmpar.c`/`bjtparam.c` replayed on C's records,
/// with the load being the port's own [`Bjt`] rebuilt from them.
#[derive(Debug)]
pub(super) struct BjtSensitivity<'a> {
    device: &'a Bjt,
    circuit_kelvin: Real,
    nominal_kelvin: Real,
}

impl<'a> BjtSensitivity<'a> {
    fn new(device: &'a Bjt, context: &ModelContext) -> Self {
        Self {
            device,
            circuit_kelvin: context.temperature + CELSIUS_TO_KELVIN,
            nominal_kelvin: context.nominal_temperature + CELSIUS_TO_KELVIN,
        }
    }
}

impl SensitivityKernel for BjtSensitivity<'_> {
    fn family(&self) -> SensitivityFamily {
        SensitivityFamily::Bjt
    }

    fn model(&self) -> Option<&str> {
        Some(&self.device.model_name)
    }

    fn model_parameters(&self) -> &'static [SensitivityParameter] {
        MODEL_PARAMETERS
    }

    fn instance_parameters(&self) -> &'static [SensitivityParameter] {
        INSTANCE_PARAMETERS
    }

    #[expect(
        clippy::too_many_lines,
        reason = "keeps the reference numerical routine in the original statement and rounding order"
    )]
    fn records(&self) -> SpiceResult<(SensitivityRecord, SensitivityRecord)> {
        let (m, i) = (&self.device.model, &self.device.instance);
        let ibe = m.ibe.unwrap_or(0.);
        let ibc = m.ibc.unwrap_or(0.);
        let (ibe_given, ibc_given) = (m.ibe.is_some(), m.ibc.is_some());
        let npn = self.device.pol > 0.;
        let mut model = SensitivityRecord::new(self.family(), ParameterScope::Model);
        for (keyword, value, given) in [
            (
                "tnom",
                m.tnom.unwrap_or(self.nominal_kelvin),
                m.tnom.is_some(),
            ),
            ("is", m.is, true),
            ("ibe", ibe, ibe_given),
            ("ibc", ibc, ibc_given),
            ("bf", m.bf, true),
            ("nf", m.nf, true),
            ("vaf", m.vaf, true),
            ("ikf", m.ikf, true),
            ("ise", m.ise, true),
            ("ne", m.ne, true),
            ("br", m.br, true),
            ("nr", m.nr, true),
            ("var", m.var, true),
            ("ikr", m.ikr, true),
            ("isc", m.isc, true),
            ("nc", m.nc, true),
            ("rb", m.rb, true),
            ("irb", m.irb, true),
            ("rbm", m.rbm, m.rbm_given),
            ("re", m.re, true),
            ("rc", m.rc, true),
            // The `IF_AC` family, which only the AC reset load's excess
            // phase (`ptf`, `tf`) reads; their *given* flags gate only
            // `bjtsetup.c` defaults the values already hold.
            ("cje", m.cje, true),
            ("vje", m.vje, true),
            ("mje", m.mje, true),
            ("tf", m.tf, true),
            ("xtf", m.xtf, true),
            ("vtf", m.vtf, true),
            ("itf", m.itf, true),
            // Elaboration refuses a nonzero PTF.
            ("ptf", 0., true),
            ("cjc", m.cjc, true),
            ("vjc", m.vjc, true),
            ("mjc", m.mjc, true),
            ("xcjc", m.xcjc, true),
            ("tr", m.tr, true),
            ("cjs", m.cjs, true),
            ("vjs", m.vjs, true),
            ("mjs", m.mjs, true),
            ("xtb", m.xtb, true),
            ("eg", m.eg, true),
            ("xti", m.xti, true),
            ("fc", m.fc, true),
            ("kf", m.kf, true),
            ("af", m.af, true),
            ("iss", m.iss.unwrap_or(0.), m.iss.is_some()),
            ("ns", m.ns, true),
            // Parameters the port refuses when given, at `bjtsetup.c`'s
            // defaults: no load the port reproduces reads them, but `MDask`
            // reports these values, which size their (rounding-noise) AC
            // columns' perturbations.
            ("rco", 0.01, false),
            ("vo", 10., false),
            ("gamma", 1e-11, false),
            ("qco", 0., false),
            ("tbf1", m.tbf.map_or(0., |c| c[0]), m.tbf.is_some()),
            ("tbf2", m.tbf.map_or(0., |c| c[1]), m.tbf.is_some()),
            ("tbr1", m.tbr.map_or(0., |c| c[0]), m.tbr.is_some()),
            ("tbr2", m.tbr.map_or(0., |c| c[1]), m.tbr.is_some()),
            ("tikf1", m.tikf[0], true),
            ("tikf2", m.tikf[1], true),
            ("tikr1", m.tikr[0], true),
            ("tikr2", m.tikr[1], true),
            ("tirb1", m.tirb[0], true),
            ("tirb2", m.tirb[1], true),
            ("tnc1", m.tnc[0], true),
            ("tnc2", m.tnc[1], true),
            ("tne1", m.tne[0], true),
            ("tne2", m.tne[1], true),
            ("tnf1", m.tnf[0], true),
            ("tnf2", m.tnf[1], true),
            ("tnr1", m.tnr[0], true),
            ("tnr2", m.tnr[1], true),
            ("trb1", m.trb[0], true),
            ("trb2", m.trb[1], true),
            ("trc1", m.trc[0], true),
            ("trc2", m.trc[1], true),
            ("tre1", m.tre[0], true),
            ("tre2", m.tre[1], true),
            ("trm1", m.trm[0], true),
            ("trm2", m.trm[1], true),
            ("tvaf1", m.tvaf[0], true),
            ("tvaf2", m.tvaf[1], true),
            ("tvar1", m.tvar[0], true),
            ("tvar2", m.tvar[1], true),
            ("ctc", m.ctc, true),
            ("cte", m.cte, true),
            ("cts", m.cts, true),
            ("tvjc", m.tvjc, true),
            ("tvje", m.tvje, true),
            ("tvjs", m.tvjs, true),
            ("titf1", m.titf[0], true),
            ("titf2", m.titf[1], true),
            ("ttf1", m.ttf[0], true),
            ("ttf2", m.ttf[1], true),
            ("ttr1", m.ttr[0], true),
            ("ttr2", m.ttr[1], true),
            ("tmje1", m.tmje[0], true),
            ("tmje2", m.tmje[1], true),
            ("tmjc1", m.tmjc[0], true),
            ("tmjc2", m.tmjc[1], true),
            ("tmjs1", m.tmjs[0], true),
            ("tmjs2", m.tmjs[1], true),
            ("tns1", m.tns[0], true),
            ("tns2", m.tns[1], true),
            // `bjtsetup.c` defaults an ungiven NKF to 0.5.
            ("nkf", m.nkf.unwrap_or(0.5), m.nkf.is_some()),
            ("tis1", m.tis[0], true),
            ("tis2", m.tis[1], true),
            ("tise1", m.tise[0], true),
            ("tise2", m.tise[1], true),
            ("tisc1", m.tisc[0], true),
            ("tisc2", m.tisc[1], true),
            ("tiss1", m.tiss[0], true),
            ("tiss2", m.tiss[1], true),
            ("vg", 1.206, false),
            ("cn", if npn { 2.42 } else { 2.2 }, false),
            ("d", if npn { 0.87 } else { 0.52 }, false),
            ("vbe_max", 1e99, false),
            ("vbc_max", 1e99, false),
            ("vce_max", 1e99, false),
            ("pd_max", 1e99, false),
            ("ic_max", 1e99, false),
            ("ib_max", 1e99, false),
            ("te_max", 1e99, false),
            ("rth0", 0., false),
        ] {
            model = model.with(keyword, value, given)?;
        }
        let mut instance = SensitivityRecord::new(self.family(), ParameterScope::Instance);
        let ic = |k: usize| self.device.initial.values.get(k).copied().flatten();
        for (keyword, value, given) in [
            ("icvbe", ic(0).unwrap_or(0.), ic(0).is_some()),
            ("icvce", ic(1).unwrap_or(0.), ic(1).is_some()),
            ("area", i.area, true),
            ("areab", i.areab, true),
            ("areac", i.areac, true),
            ("m", i.multiplier, true),
            // `bjtparam.c` stores `temp` in kelvin.
            (
                "temp",
                i.temp.unwrap_or(self.circuit_kelvin + i.dtemp),
                i.temp.is_some(),
            ),
            ("dtemp", i.dtemp, true),
        ] {
            instance = instance.with(keyword, value, given)?;
        }
        self.setup(&mut model, &mut instance)?;
        self.temperature(&mut model, &mut instance)?;
        Ok((model, instance))
    }

    fn ask(
        &self,
        scope: ParameterScope,
        keyword: &str,
        records: RecordPair<'_>,
    ) -> SpiceResult<Real> {
        // `bjtmask.c`/`bjtask.c` answer in Celsius while `bjtmpar.c`/
        // `bjtparam.c` store kelvin, so the walk steps the Celsius value.
        match (scope, keyword) {
            (ParameterScope::Model, "tnom") => Ok(records.model.value("tnom")? - CELSIUS_TO_KELVIN),
            (ParameterScope::Instance, "temp") => {
                Ok(records.instance.value("temp")? - CELSIUS_TO_KELVIN)
            }
            (ParameterScope::Model, _) => records.model.value(keyword),
            (ParameterScope::Instance, _) => records.instance.value(keyword),
        }
    }

    fn set(
        &self,
        scope: ParameterScope,
        keyword: &str,
        value: Real,
        model: &mut SensitivityRecord,
        instance: &mut SensitivityRecord,
    ) -> SpiceResult<()> {
        match (scope, keyword) {
            (ParameterScope::Model, "tnom") => model.set(keyword, value + CELSIUS_TO_KELVIN),
            (ParameterScope::Instance, "temp") => instance.set(keyword, value + CELSIUS_TO_KELVIN),
            (ParameterScope::Model, _) => model.set(keyword, value),
            (ParameterScope::Instance, _) => instance.set(keyword, value),
        }
    }

    /// `bjtsetup.c`'s in-place rewrites, which `cktsens.c` re-runs for every
    /// parameter (`DEVsetup` sits inside the walk's loop):
    ///
    /// - a leakage current above 1e-4 is a multiple of `IS`
    ///   (`BJTleakBEcurrent = BJTsatCur * BJTleakBEcurrent`);
    /// - `NKF` above 1 is clamps to 1 and `XCJC` outside [0, 1] to the nearest
    ///   bound.
    ///
    /// The `!given` defaults are already in the records, and re-applying them
    /// would be a no-op: only the setter marking a field *given* can move it.
    fn setup(
        &self,
        model: &mut SensitivityRecord,
        _instance: &mut SensitivityRecord,
    ) -> SpiceResult<()> {
        for leak in ["ise", "isc"] {
            if model.given(leak) {
                let value = model.value(leak)?;
                if value > 1e-4 {
                    model.store(leak, model.value("is")? * value)?;
                }
            }
        }
        if model.given("nkf") {
            let value = model.value("nkf")?;
            if value > 1. {
                model.store("nkf", 1.)?;
            }
        }
        if model.given("xcjc") {
            let value = model.value("xcjc")?.clamp(0., 1.);
            model.store("xcjc", value)?;
        }
        Ok(())
    }

    /// `bjttemp.c`'s model-side stores, which `cktsens.c` re-runs before every
    /// base and perturbed load: an ungiven `tnom` follows `CKTnomTemp`, an
    /// ungiven `rbm` follows `rb` (so the `rb` column moves `rbm` too), and
    /// `fc` is clamped to 0.9999 when given and 0.5 otherwise. Everything else
    /// the routine writes is per instance and re-derived by the rebuilt
    /// device, which is also where the huge rest of `BJTemp` lives.
    fn temperature(
        &self,
        model: &mut SensitivityRecord,
        _instance: &mut SensitivityRecord,
    ) -> SpiceResult<()> {
        if !model.given("tnom") {
            model.store("tnom", self.nominal_kelvin)?;
        }
        if !model.given("rbm") {
            let rb = model.value("rb")?;
            model.store("rbm", rb)?;
        }
        if model.given("fc") {
            if model.value("fc")? > 0.9999 {
                model.store("fc", 0.9999)?;
            }
        } else {
            model.store("fc", 0.5)?;
        }
        Ok(())
    }

    fn load(
        &self,
        _setup: RecordPair<'_>,
        current: RecordPair<'_>,
        mode: SensitivityMode,
    ) -> SpiceResult<SensitivityLoad> {
        if mode == SensitivityMode::Ac {
            return Err(SpiceError::circuit(
                "a BJT's AC sensitivity load is its reset load",
            ));
        }
        Ok(SensitivityLoad::Replacement(Box::new(self.bjt(current)?)))
    }

    /// `BJTacLoad` after `cktsens.c`'s reset load: `gpi`, `gmu`, `gm`, `go`,
    /// `gx` and `gdsub` are the `CKTstate0` values `bjtload.c` stored at
    /// zero junction voltages with the reset records, the charge
    /// capacitances (`cqbe`, ... written only by a small-signal or transient
    /// load) are zero, and `RC`/`RE`, `m` and the excess-phase factor
    /// `PTF/(180/pi) TF` (`bjttemp.c`) come from `current`.
    fn reset_ac_load(
        &self,
        _setup: RecordPair<'_>,
        reset: AcReset<'_>,
        current: RecordPair<'_>,
    ) -> SpiceResult<SensitivityLoad> {
        let (m, i) = (current.model, current.instance);
        Ok(SensitivityLoad::Replacement(Box::new(ResetBjt {
            state: self.bjt(reset.state)?,
            current: self.bjt(current)?,
            multiplier: i.value("m")?,
            excess_phase: m.value("ptf")? / (180.0 / std::f64::consts::PI) * m.value("tf")?,
            omega: reset.omega,
        })))
    }
}

impl BjtSensitivity<'_> {
    /// The port's BJT with the records' DC parameters.
    #[expect(
        clippy::too_many_lines,
        reason = "keeps the reference numerical routine in the original statement and rounding order"
    )]
    fn bjt(&self, current: RecordPair<'_>) -> SpiceResult<Bjt> {
        let (m, i) = (current.model, current.instance);
        let r = |keyword: &str| m.value(keyword);
        let v = |keyword: &str| i.value(keyword);
        // Only the model fields C's DC load reads: the unported parameters and
        // the `IF_AC` capacitance family stay at the card's values, exactly as
        // C's own model struct keeps them through the walk.
        let mut model = self.device.model;
        model.tnom = m.given("tnom").then(|| r("tnom")).transpose()?;
        model.is = r("is")?;
        model.ibe = m.given("ibe").then(|| r("ibe")).transpose()?;
        model.ibc = m.given("ibc").then(|| r("ibc")).transpose()?;
        model.ibe_ibc = if m.given("ibe") && m.given("ibc") {
            Some((r("ibe")?, r("ibc")?))
        } else {
            None
        };
        model.bf = r("bf")?;
        model.nf = r("nf")?;
        model.vaf = r("vaf")?;
        model.ikf = r("ikf")?;
        model.ise = r("ise")?;
        model.ne = r("ne")?;
        model.br = r("br")?;
        model.nr = r("nr")?;
        model.var = r("var")?;
        model.ikr = r("ikr")?;
        model.isc = r("isc")?;
        model.nc = r("nc")?;
        model.rb = r("rb")?;
        model.irb = r("irb")?;
        model.rbm = r("rbm")?;
        model.rbm_given = m.given("rbm");
        model.re = r("re")?;
        model.rc = r("rc")?;
        model.xtb = r("xtb")?;
        model.eg = r("eg")?;
        model.xti = r("xti")?;
        model.fc = r("fc")?;
        model.kf = r("kf")?;
        model.af = r("af")?;
        model.iss = m.given("iss").then(|| r("iss")).transpose()?;
        model.ns = r("ns")?;
        model.tlev = self.device.model.tlev;
        model.tlevc = self.device.model.tlevc;
        model.tbf = if m.given("tbf1") || m.given("tbf2") {
            Some([r("tbf1")?, r("tbf2")?])
        } else {
            None
        };
        model.tbr = if m.given("tbr1") || m.given("tbr2") {
            Some([r("tbr1")?, r("tbr2")?])
        } else {
            None
        };
        model.tikf = [r("tikf1")?, r("tikf2")?];
        model.tikr = [r("tikr1")?, r("tikr2")?];
        model.tirb = [r("tirb1")?, r("tirb2")?];
        model.tnc = [r("tnc1")?, r("tnc2")?];
        model.tne = [r("tne1")?, r("tne2")?];
        model.tnf = [r("tnf1")?, r("tnf2")?];
        model.tnr = [r("tnr1")?, r("tnr2")?];
        model.trb = [r("trb1")?, r("trb2")?];
        model.trc = [r("trc1")?, r("trc2")?];
        model.tre = [r("tre1")?, r("tre2")?];
        model.trm = [r("trm1")?, r("trm2")?];
        model.tvaf = [r("tvaf1")?, r("tvaf2")?];
        model.tvar = [r("tvar1")?, r("tvar2")?];
        model.ctc = r("ctc")?;
        model.cte = r("cte")?;
        model.cts = r("cts")?;
        model.tvjc = r("tvjc")?;
        model.tvje = r("tvje")?;
        model.tvjs = r("tvjs")?;
        model.titf = [r("titf1")?, r("titf2")?];
        model.ttf = [r("ttf1")?, r("ttf2")?];
        model.ttr = [r("ttr1")?, r("ttr2")?];
        model.tmje = [r("tmje1")?, r("tmje2")?];
        model.tmjc = [r("tmjc1")?, r("tmjc2")?];
        model.tmjs = [r("tmjs1")?, r("tmjs2")?];
        model.tns = [r("tns1")?, r("tns2")?];
        model.tis = [r("tis1")?, r("tis2")?];
        model.tise = [r("tise1")?, r("tise2")?];
        model.tisc = [r("tisc1")?, r("tisc2")?];
        model.tiss = [r("tiss1")?, r("tiss2")?];
        model.nkf = m.given("nkf").then(|| r("nkf")).transpose()?;
        let instance = Instance {
            area: v("area")?,
            areab: v("areab")?,
            areac: v("areac")?,
            multiplier: v("m")?,
            // C's `BJTtemp` is kelvin and so is this field (`bjtparam.c`
            // stores `value + CONSTCtoK`); the record holds it in kelvin.
            temp: i.given("temp").then(|| v("temp")).transpose()?,
            dtemp: v("dtemp")?,
        };
        let device = Bjt {
            name: self.device.name.clone(),
            model_name: self.device.model_name.clone(),
            terminals: self.device.terminals.clone(),
            nodes: self.device.nodes,
            pol: self.device.pol,
            subs: self.device.subs,
            model,
            instance,
            derived: TemperatureCache::new(),
            initial: self.device.initial.clone(),
        };
        Ok(device)
    }
}

/// A BJT's AC load in `cktsens.c`'s sweep
/// ([`crate::devices::sensitivity::DeviceSensitivity::reset_ac_load`]).
#[derive(Debug)]
struct ResetBjt {
    /// Built from the reset load's records: the junction conductances.
    state: Bjt,
    /// Built from the current records: `RC` and `RE`.
    current: Bjt,
    /// `BJTm`.
    multiplier: Real,
    /// `BJTexcessPhaseFactor`.
    excess_phase: Real,
    /// `CKTomega`.
    omega: Real,
}

impl Device for ResetBjt {
    fn name(&self) -> &str {
        &self.current.name
    }

    fn designator(&self) -> char {
        'q'
    }

    fn terminals(&self) -> &[NodeId] {
        &self.current.terminals
    }

    fn stamp(&self, _context: &mut StampContext<'_>) -> SpiceResult<()> {
        Err(SpiceError::circuit(format!(
            "{}: an AC sensitivity stand-in has no DC load",
            self.current.name
        )))
    }

    /// `bjtacld.c`'s stamps in its order with zero capacitances: the real
    /// parts, then the excess phase's imaginary `xgm` parts.
    fn assemble_linear(&self, context: &mut LinearContext<'_>) -> SpiceResult<()> {
        use crate::devices::sensitivity::{ResetStamp, ResetStamps, StampAt, StampPart};
        let gmin = context.model_context.gmin;
        let t = self.state.thermal(context.model_context)?;
        let p = c_point(&t, Bias::default(), gmin)?;
        let current = self.current.thermal(context.model_context)?;
        let m = self.multiplier;
        let (gcpr, gepr) = (current.gc, current.ge);
        let Transport {
            gpi,
            gmu,
            mut gm,
            go,
            gx,
            ..
        } = p.transport;
        let Junction {
            conductance: gdsub, ..
        } = p.substrate;
        let mut xgm = 0.;
        let td = self.excess_phase;
        if td != 0. {
            let arg = td * self.omega;
            gm += go;
            xgm = -gm * arg.sin();
            gm = gm * arg.cos() - go;
        }
        let Nodes {
            c,
            b,
            e,
            s,
            cp,
            bp,
            ep,
            sc,
        } = self.current.nodes;
        let cx = cp;
        let real = [
            (c, c, m * (gcpr)),
            (b, b, m * (gx)),
            (e, e, m * (gepr)),
            (cp, cp, m * (gmu + go)),
            (cx, cx, m * (gcpr)),
            (bp, bp, m * (gx + gpi + gmu)),
            (ep, ep, m * (gpi + gepr + gm + go)),
            (c, cx, m * (-gcpr)),
            (b, bp, m * (-gx)),
            (e, ep, m * (-gepr)),
            (cx, c, m * (-gcpr)),
            (cp, bp, m * (-gmu + gm)),
            (cp, ep, m * (-gm - go)),
            (bp, b, m * (-gx)),
            (bp, cp, m * (-gmu)),
            (bp, ep, m * (-gpi)),
            (ep, e, m * (-gepr)),
            (ep, cp, m * (-go)),
            (ep, bp, m * (-gpi - gm)),
            (sc, sc, m * (gdsub)),
            (s, s, m * (gdsub)),
            (sc, s, m * (-gdsub)),
            (s, sc, m * (-gdsub)),
        ];
        let mut stamps: Vec<ResetStamp> = real
            .into_iter()
            .map(|(a, b, value)| ResetStamp {
                at: StampAt::Nodes(a, b),
                value,
                part: StampPart::Real,
            })
            .collect();
        if xgm != 0. {
            // `xcpi`, `xcmu` and `xcmcb` are zero: `m * (0 + xgm)`,
            // `m * (-0 + xgm)`, `m * (-xgm)`, `m * (-0 - xgm - 0)`.
            for (a, b, value) in [
                (ep, ep, m * xgm),
                (cp, bp, m * xgm),
                (cp, ep, m * (-xgm)),
                (ep, bp, m * (-xgm)),
            ] {
                stamps.push(ResetStamp {
                    at: StampAt::Nodes(a, b),
                    value,
                    part: StampPart::Imaginary,
                });
            }
        }
        ResetStamps::new(&self.current.name, 'q', &self.current.terminals, stamps)?
            .assemble_linear(context)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    pub(super) fn thermal(body: &str, temperature: Real, vertical: bool) -> Thermal {
        let deck = crate::netlist::source::parse_deck_text(
            std::path::Path::new("bjt.cir"),
            &format!("bjt\nq1 c b e qm\n.model qm npn({body})\n.end\n"),
        );
        let netlist = crate::netlist::Parser::new().parse_deck(&deck).unwrap();
        let card = &netlist.models[0];
        let assignments: Vec<_> = card.parameters.iter().map(canonical).collect();
        let values = MODEL_SCHEMA.validate(&assignments, &card.location).unwrap();
        let model = Model::from_values(&values).unwrap();
        let instance = Instance {
            area: 1.3,
            areab: 1.3,
            areac: 1.3,
            multiplier: 1.,
            temp: None,
            dtemp: 0.,
        };
        model
            .thermal(&instance, vertical, &ModelContext::new(temperature, 27.))
            .unwrap()
    }

    /// `bjtload.c`'s conductances and capacitances are the derivatives of
    /// its currents and charges (C's own convergence test relies on it):
    /// `cb` by `gpi`/`gmu`, `cc` by `gm + go`/`-(go + gmu)`, `qbe` by
    /// `capbe`/`geqcb`, `qbc`, `qbx`, `qsub` and `cdsub` by their own.
    fn check_jacobian(t: &Thermal) {
        let gmin = 1e-12;
        let h = 1e-7;
        let biases = [
            (0.65, -2.),
            (0.75, -0.3),
            (0.8, 0.2),
            (0.85, 0.7),
            (-0.4, 0.62),
            (0.1, -0.05),
            (-1.5, -3.),
        ];
        let close = |what: &str, analytic: Real, numeric: Real, value: Real| {
            let scale = analytic.abs().max(numeric.abs()).max(value.abs() * 1e-3);
            assert!(
                (analytic - numeric).abs() <= 2e-5 * scale + 1e-14,
                "{what}: {analytic:e} vs {numeric:e}"
            );
        };
        for (vbe, vbc) in biases {
            let at = |vbe: Real, vbc: Real| {
                c_point(
                    t,
                    Bias {
                        vbe,
                        vbc,
                        vbx: vbc,
                        vsub: -vbc,
                    },
                    gmin,
                )
                .unwrap()
            };
            let p = at(vbe, vbc);
            let d = |f: fn(&Point) -> Real, dvbe: Real, dvbc: Real| {
                (f(&at(vbe + dvbe, vbc + dvbc)) - f(&at(vbe - dvbe, vbc - dvbc))) / (2. * h)
            };
            let at_bias = format!("at ({vbe}, {vbc})");
            close(
                &format!("dcb/dvbe {at_bias}"),
                p.transport.gpi,
                d(|p| p.transport.cb, h, 0.),
                p.transport.cb,
            );
            close(
                &format!("dcb/dvbc {at_bias}"),
                p.transport.gmu,
                d(|p| p.transport.cb, 0., h),
                p.transport.cb,
            );
            close(
                &format!("dcc/dvbe {at_bias}"),
                p.transport.gm + p.transport.go,
                d(|p| p.transport.cc, h, 0.),
                p.transport.cc,
            );
            close(
                &format!("dcc/dvbc {at_bias}"),
                -(p.transport.go + p.transport.gmu),
                d(|p| p.transport.cc, 0., h),
                p.transport.cc,
            );
            close(
                &format!("dqbe/dvbe {at_bias}"),
                p.charges.capbe,
                d(|p| p.charges.qbe, h, 0.),
                p.charges.qbe,
            );
            close(
                &format!("dqbe/dvbc {at_bias}"),
                p.charges.geqcb,
                d(|p| p.charges.qbe, 0., h),
                p.charges.qbe,
            );
            close(
                &format!("dqbc/dvbc {at_bias}"),
                p.charges.capbc,
                d(|p| p.charges.qbc, 0., h),
                p.charges.qbc,
            );
            // Single-voltage junctions: vbx (qbx) and vsub (substrate).
            for v in [vbc, -vbc, 0.3, -0.7] {
                let single = |v: Real| {
                    c_point(
                        t,
                        Bias {
                            vbe,
                            vbc,
                            vbx: v,
                            vsub: v,
                        },
                        gmin,
                    )
                    .unwrap()
                };
                let (lo, mid, hi) = (single(v - h), single(v), single(v + h));
                let fd = |f: fn(&Point) -> Real| (f(&hi) - f(&lo)) / (2. * h);
                close(
                    &format!("dqbx at {v}"),
                    mid.charges.capbx,
                    fd(|p| p.charges.qbx),
                    mid.charges.qbx,
                );
                close(
                    &format!("dqsub at {v}"),
                    mid.charges.capsub,
                    fd(|p| p.charges.qsub),
                    mid.charges.qsub,
                );
                close(
                    &format!("dcdsub at {v}"),
                    mid.substrate.conductance,
                    fd(|p| p.substrate.current),
                    mid.substrate.current,
                );
            }
        }
    }

    #[test]
    fn bjtload_conductances_are_the_current_and_charge_derivatives() {
        for body in [
            // Ebers-Moll core with charges.
            "is=1e-15 bf=80 br=2 cje=2p cjc=1p tf=0.3n tr=5n",
            // qb: Early and high injection, default sqrt and NKF exponent.
            "is=1e-15 vaf=40 var=8 ikf=3m ikr=0.5m",
            "is=1e-15 vaf=40 var=8 ikf=3m ikr=0.5m nkf=0.7",
            // Leakage, including the multiple-of-IS spelling.
            "is=1e-15 ise=1e-13 ne=1.6 isc=50 nc=1.8",
            // Base resistance through qb and through IRB.
            "is=1e-15 vaf=40 ikf=3m rb=200 rbm=20",
            "is=1e-15 ikf=3m rb=200 rbm=20 irb=50u",
            // Bias-dependent transit time.
            "is=1e-15 vaf=40 ikf=3m tf=0.4n xtf=3 vtf=2 itf=20m cje=2p mje=0.4 fc=0.6",
            "is=1e-15 tf=0.4n xtf=3",
            "is=1e-15 tf=0.4n xtf=3 vtf=2",
            "is=1e-15 tf=0.4n xtf=3 itf=20m",
            "is=1e-15 tf=0.4n vtf=2 itf=20m",
            "is=1e-15 xtf=3 vtf=2 itf=20m",
            // Split base-collector charge and substrate junction.
            "is=1e-15 cjc=3p xcjc=0.4 mjc=0.5 cjs=2p mjs=0.3 vjs=0.6 iss=1e-17 ns=1.1",
        ] {
            check_jacobian(&thermal(body, 27., true));
            // Temperature scaling must keep the equations consistent too.
            check_jacobian(&thermal(body, 85., true));
        }
    }

    #[test]
    #[ignore = "matched-baseline kernel microbenchmark; run explicitly with --ignored --nocapture"]
    fn bjt_physics_benchmark() {
        use crate::devices::distortion::DistortionContext;
        use std::{hint::black_box, time::Instant};

        let model = "is=1e-15 bf=80 br=2 vaf=40 var=8 ikf=3m ikr=0.5m ise=1e-14 isc=2e-14 rb=200 rbm=20 irb=50u rc=5 re=2 tf=0.4n xtf=3 vtf=2 itf=20m cje=2p cjc=3p xcjc=0.4 cjs=2p mjs=0.3 iss=1e-17";
        let t = thermal(model, 85., true);
        let circuit = crate::devices::Circuit::from_netlist(
            &crate::netlist::Parser::new()
                .parse_deck(&crate::netlist::source::parse_deck_text(
                    std::path::Path::new("bjt-bench.cir"),
                    &format!(
                        "bjt bench
q1 c b e s qm area=1.3 m=2 temp=85
.model qm npn({model})
.end
"
                    ),
                ))
                .unwrap(),
        )
        .unwrap();
        let device = circuit.device("q1").unwrap();
        let model_context = ModelContext::new(85., 27.);
        let mut solution = Vector::zeros(circuit.unknown_count());
        for (name, voltage) in [
            ("b", 0.8),
            ("c", 2.),
            ("e", 0.),
            ("s", 0.2),
            ("q1#base", 0.75),
            ("q1#collcx", 1.95),
            ("q1#emitter", 0.01),
        ] {
            let node = circuit.nodes().get(name).unwrap();
            let row = circuit.unknowns().node_row(node).unwrap();
            solution.set(row, voltage).unwrap();
        }
        let distortion =
            DistortionContext::new(&model_context, circuit.unknowns(), &solution).unwrap();
        let biases = [
            Bias {
                vbe: 0.74,
                vbc: -1.2,
                vbx: -1.15,
                vsub: -1.75,
            },
            Bias {
                vbe: 0.8,
                vbc: 0.2,
                vbx: 0.25,
                vsub: -0.1,
            },
            Bias {
                vbe: -0.4,
                vbc: 0.62,
                vbx: 0.65,
                vsub: 0.3,
            },
            Bias {
                vbe: -0.2,
                vbc: -2.,
                vbx: -1.9,
                vsub: -2.,
            },
        ];
        // Warm the temperature cache before measuring the distortion setup.
        black_box(device.distortion(&distortion).unwrap());
        let mut load_times = Vec::new();
        let mut distortion_times = Vec::new();
        for _ in 0..7 {
            let started = Instant::now();
            for index in 0..100_000 {
                let bias = black_box(biases[index % biases.len()]);
                black_box(c_point(black_box(&t), bias, black_box(1e-12)).unwrap());
            }
            load_times.push(started.elapsed());
            let started = Instant::now();
            for _ in 0..20_000 {
                black_box(device.distortion(black_box(&distortion)).unwrap());
            }
            distortion_times.push(started.elapsed());
        }
        load_times.sort_unstable();
        distortion_times.sort_unstable();
        eprintln!(
            "BJT ordinary 100000 points: median {:?}, min {:?}; distortion 20000 setups: median {:?}, min {:?}",
            load_times[3], load_times[0], distortion_times[3], distortion_times[0]
        );
    }

    #[test]
    fn temperature_scaling_follows_bjttemp() {
        let nominal = thermal("is=1e-15 bf=100 xtb=1.5 cje=1p", 27., true);
        assert!((nominal.is_be - 1.3e-15).abs() < 1e-28);
        assert!((nominal.bf - 100.).abs() < 1e-12);
        let hot = thermal("is=1e-15 bf=100 xtb=1.5 cje=1p", 127., true);
        let (t, tnom) = (400.15, 300.15);
        let vt = K_OVER_Q * t;
        let factlog = (t / tnom - 1.) * 1.11 / vt + 3. * (t / tnom).ln();
        let expected = 1.3e-15 * factlog.exp();
        assert!((hot.is_be - expected).abs() < 1e-12 * expected);
        let bf = 100. * ((t / tnom).ln() * 1.5).exp();
        assert!((hot.bf - bf).abs() < 1e-12 * bf);
        // The built-in potential falls and the zero-bias capacitance rises.
        assert!(hot.be.potential < nominal.be.potential && hot.cje > nominal.cje);
        // TLEV=1 linear beta and TLEVC=1 linear capacitance/potential.
        let linear = thermal(
            "is=1e-15 bf=100 xtb=0.01 tlev=1 tlevc=1 cje=1p cte=1m vje=0.8 tvje=2m",
            127.,
            true,
        );
        assert!((linear.bf - 100. * (1. + 0.01 * 100.)).abs() < 1e-9);
        assert!((linear.cje - 1.3e-12 * 1.1).abs() < 1e-24);
        assert!((linear.be.potential - 0.6).abs() < 1e-12);
        // Polynomial resistance and Early-voltage coefficients.
        let poly = thermal("rb=100 trb1=1m rc=10 trc2=1e-5 vaf=50 tvaf1=-1m", 77., true);
        assert!((poly.rb - 100. * 1.05 / 1.3).abs() < 1e-9);
        assert!((poly.gc - 1.3 / (10. * 1.025)).abs() < 1e-12);
        assert!((poly.inv_vaf - 1. / (50. * 0.95)).abs() < 1e-15);
    }

    #[test]
    fn leakage_above_1e_minus_4_is_a_multiple_of_is() {
        let t = thermal("is=2e-15 ise=100 isc=1e-14", 27., true);
        assert!((t.ise - 1.3 * 2e-13).abs() < 1e-27);
        assert!((t.isc - 1.3e-14).abs() < 1e-27);
    }
}
