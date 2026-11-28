//! HFET2, the `Z` designator at level 6 (`hfet2/`, the `hfeta2` channel).
//!
//! C references, read as behaviour only: `hfet2/hfet2setup.c` (defaults and the
//! conditional internal nodes), `hfet2/hfet2temp.c` (`HFET2temp`),
//! `hfet2/hfet2load.c` (`HFET2load` and `hfeta2`), `hfet2/hfet2acl.c`,
//! `hfet2/hfet2pzl.c`, `hfet2/hfet2trunc.c`, `hfet2/hfet2getic.c`,
//! `hfet2/hfet2ask.c`, `hfet2/hfet2param.c` and `hfet2/hfet2mpar.c`.
//!
//! # Where it sits between the other two levels
//!
//! HFET2 is the smallest of the three lineage copies: 13 state slots, no
//! `#gate` node, no prime-prime pair, no `gatemod`, no `a1`/`a2` corrections
//! and no `cds` charge. Its load is the JFET-shaped one: `pnjlim` *and*
//! `fetlim` limiting (HFET1 has only the latter), a single-exponential gate
//! junction from `JS`/`GGR`, and a transient that folds the integration
//! companion into `ggs`/`ggd` directly (HFET1 keeps it in `ggspp`/`ggdpp`).
//! Like HFET1 it is two-polarity: every node difference is scaled by the type
//! and the loaded residual is scaled back.
//!
//! # C quirks reproduced deliberately
//!
//! - `HFET2temp` scales `vto` into `TVTO` but leaves `HFET2_VT1`/`VT2` on the
//!   *raw* threshold, and `hfeta2`'s `nsc` branch mixes the two
//!   (`exp((vgt + TVTO - VT2)/(ETA2*vt))`). Both are reproduced as written.
//! - `vcrit` is derived from the constant `1e-11` rather than a junction
//!   current (`hfet2temp.c`), so `pnjlim`'s critical voltage is fixed.
//! - `N01` is computed in C's temperature routine and read nowhere; the port
//!   does not carry it.
//! - The C identifiers for three matrix entries are mangled by a bad
//!   substitution (`HFET2sourcePriHFET2ourcePrimePtr`); they still map to
//!   `(sourcePrime, sourcePrime)`, `(sourcePrime, source)` and
//!   `(drainPrime, sourcePrime)` in `hfet2setup.c` and `hfet2load.c` alike,
//!   which is what the port stamps.
//! - `d2` and `eta2` gate the `nsc` branch through their *given* flags, which
//!   nothing ever clears, so `.sens` reports a topology jump for them unless
//!   the card gave both; `hfet2ask.c` has no `HFET2_M` case, so C reports no
//!   multiplier column.

#![expect(
    clippy::similar_names,
    reason = "C-translated equation symbols keep their original distinct names for source-level audit"
)]
#![expect(
    clippy::too_many_lines,
    reason = "a complete translated C numerical routine remains contiguous for equation parity review"
)]

use super::channel::{ChannelModel, ChannelTemperature};
use super::lifecycle::{ChargeInput, Linearization, Start, accepted, charge_step};
use super::{CELSIUS_TO_KELVIN, HfetInstance as Instance};
use super::{NodeSetup, TemperatureCache, ask_hfet, set_hfet};
use crate::devices::initial::InstanceInitial;
use crate::devices::jfet::{BOLTZMANN, CHARGE, p};
use crate::devices::schema::{
    ScalarDomain as D, ScalarParameter as P, ScalarSchema, ScalarUnit as U,
};
use crate::devices::sensitivity::{
    ParameterScope, RecordPair, ResetAcDevice, SensitivityFamily, SensitivityKernel,
    SensitivityParameter, SensitivityRecord, StampPart, node_stamps, reconstructed_reset_loads,
    stamp_reset,
};
use crate::devices::{
    AnalysisMode, Device, DeviceState, LinearContext, ModelContext, ResolvedModel, StampContext,
};
use crate::maths::Vector;
use crate::netlist::ast::DeviceInstance;
use crate::primitives::{NodeAppend, NodeId, Real, SpiceError, SpiceResult};

/// C `CONSTKoverQ`.
const K_OVER_Q: Real = BOLTZMANN / CHARGE;
/// `hfet2temp.c`'s fixed critical-current constant for `vcrit`.
const VCRIT_CURRENT: Real = 1e-11;

/// The state slots of `HFET2load`'s `CKTstate0`, in the port's own order. Each
/// charge is followed by its derivative.
mod slot {
    pub(super) const VGS: usize = 0;
    pub(super) const VGD: usize = 1;
    pub(super) const CG: usize = 2;
    pub(super) const CD: usize = 3;
    pub(super) const CGD: usize = 4;
    pub(super) const GM: usize = 5;
    pub(super) const GDS: usize = 6;
    pub(super) const GGS: usize = 7;
    pub(super) const GGD: usize = 8;
    pub(super) const QGS: usize = 9;
    pub(super) const CQGS: usize = 10;
    pub(super) const QGD: usize = 11;
    pub(super) const CQGD: usize = 12;
    pub(super) const COUNT: usize = 13;
}

/// `HFET2mPTable` with `hfet2setup.c`'s defaults. The type-dependent defaults
/// (`vt0`, `eta`, `mu`, `vs`) are resolved by [`Model::resolve`], and `vt1`/
/// `vt2` have theirs in `HFET2temp`.
const MODEL: &[P] = &[
    p("vt0", U::Volt, D::Finite, None),
    p("vto", U::Volt, D::Finite, None),
    p("cf", U::Farad, D::NonNegative, Some(0.)),
    p("d1", U::Metre, D::Positive, Some(0.03e-6)),
    p("d2", U::Metre, D::Positive, Some(0.2e-6)),
    p("del", U::Dimensionless, D::NonNegative, Some(0.04)),
    p("delta", U::Dimensionless, D::Finite, Some(3.0)),
    p("deltad", U::Metre, D::NonNegative, Some(4.5e-9)),
    p("di", U::Metre, D::Positive, Some(0.04e-6)),
    p(
        "epsi",
        U::FaradPerMetre,
        D::Positive,
        Some(12.244 * 8.854_18e-12),
    ),
    p("eta", U::Dimensionless, D::Finite, None),
    p("eta1", U::Dimensionless, D::Finite, Some(2.)),
    p("eta2", U::Dimensionless, D::Finite, Some(2.)),
    p("gamma", U::Dimensionless, D::Finite, Some(3.0)),
    p("ggr", U::SiemensPerMetre, D::NonNegative, Some(0.)),
    p("js", U::AmperePerSquareMetre, D::NonNegative, Some(0.)),
    p("klambda", U::Dimensionless, D::Finite, Some(0.)),
    p("kmu", U::Dimensionless, D::Finite, Some(0.)),
    p("knmax", U::Dimensionless, D::Finite, Some(0.)),
    p("kvto", U::Dimensionless, D::Finite, Some(0.)),
    p("lambda", U::InverseVolt, D::NonNegative, Some(0.15)),
    p("m", U::Dimensionless, D::Finite, Some(3.0)),
    p("mc", U::Dimensionless, D::Finite, Some(3.0)),
    p("mu", U::SquareCentimetrePerVoltSecond, D::NonNegative, None),
    p("n", U::Dimensionless, D::Finite, Some(5.0)),
    p("nmax", U::PerCubicCentimetre, D::NonNegative, Some(2e16)),
    p("p", U::Dimensionless, D::Finite, Some(1.)),
    p("rd", U::Ohm, D::NonNegative, Some(0.)),
    p("rs", U::Ohm, D::NonNegative, Some(0.)),
    p("rdi", U::Ohm, D::NonNegative, Some(0.)),
    p("rsi", U::Ohm, D::NonNegative, Some(0.)),
    p("sigma0", U::Dimensionless, D::Finite, Some(0.057)),
    p("vs", U::MetrePerSecond, D::Positive, None),
    p("vsigma", U::Volt, D::Positive, Some(0.1)),
    p("vsigmat", U::Volt, D::Finite, Some(0.3)),
    p("vt1", U::Volt, D::Finite, None),
    p("vt2", U::Volt, D::Finite, None),
    p("level", U::Dimensionless, D::Finite, Some(6.)),
];

/// The validated model card.
#[derive(Debug, Clone, Copy)]
pub(super) struct Model {
    channel: ChannelModel,
    /// `+1` for N, `-1` for P; `hfet2temp.c` stores `TYPE*VTO` in `TVTO`.
    pol: Real,
    threshold: Real,
    cf: Real,
    d2: Real,
    del: Real,
    delta: Real,
    deltad: Real,
    di: Real,
    ggr: Real,
    js: Real,
    klambda: Real,
    kmu: Real,
    knmax: Real,
    kvto: Real,
    lambda: Real,
    mu: Real,
    n: Real,
    nmax: Real,
    rd: Real,
    rs: Real,
    vt1: Option<Real>,
    vt2: Option<Real>,
    /// `hfet2setup.c`'s `Given` flags that `hfeta2` reads.
    eta2_given: bool,
    d2_given: bool,
}

impl Model {
    /// `HFET2setup`'s defaults, `hfet2mpar.c`'s setters and `inpdomod.c`'s
    /// level check. A `phfet` base (or the `phfet` flag) makes it p-type.
    pub(super) fn resolve<'a>(
        resolved: &'a ResolvedModel<'_>,
    ) -> SpiceResult<(Self, Vec<&'a crate::netlist::ast::ParameterAssignment>)> {
        let (pol, scalars) = crate::devices::jfet::typed_scalars(
            resolved,
            resolved.family() == crate::devices::ModelFamily::Pmf,
            ["nhfet", "phfet"],
        );
        // `inpgmod.c::create_model`: a setter outside the model table that is
        // a settable instance parameter becomes a model-level instance
        // default, applied before the instance card's own setters.
        let (scalars, defaults) = crate::devices::schema::split_model_setters(
            scalars,
            &MODEL_SCHEMA,
            &super::HFET_INSTANCE_SCHEMA,
        );
        let values = MODEL_SCHEMA.validate(scalars, &resolved.card().location)?;
        let r = |name: &str| crate::devices::jfet::required(&values, name);
        let alias = |names: &[&str]| crate::devices::jfet::last_setter(&values, names);
        let typed = |explicit: Option<Real>, n: Real, p: Real| {
            explicit.unwrap_or(if pol > 0. { n } else { p })
        };
        let model = Self {
            channel: ChannelModel {
                eta: typed(values.get("eta").map(|v| v.value), 1.28, 1.4),
                vs: typed(values.get("vs").map(|v| v.value), 1.5e5, 0.8e5),
                rsi: r("rsi")?,
                rdi: r("rdi")?,
                vsigmat: r("vsigmat")?,
                vsigma: r("vsigma")?,
                sigma0: r("sigma0")?,
                eta2: r("eta2")?,
                gamma: r("gamma")?,
                m: r("m")?,
                eta1: r("eta1")?,
                d1: r("d1")?,
                epsi: r("epsi")?,
                mc: r("mc")?,
                pm: r("p")?,
            },
            pol,
            threshold: typed(alias(&["vt0", "vto"]), 0.15, -0.15),
            cf: r("cf")?,
            d2: r("d2")?,
            del: r("del")?,
            delta: r("delta")?,
            deltad: r("deltad")?,
            di: r("di")?,
            ggr: r("ggr")?,
            js: r("js")?,
            klambda: r("klambda")?,
            kmu: r("kmu")?,
            knmax: r("knmax")?,
            kvto: r("kvto")?,
            lambda: r("lambda")?,
            mu: typed(values.get("mu").map(|v| v.value), 0.4, 0.03),
            n: r("n")?,
            nmax: r("nmax")?,
            rd: r("rd")?,
            rs: r("rs")?,
            vt1: values.get("vt1").map(|v| v.value),
            vt2: values.get("vt2").map(|v| v.value),
            eta2_given: values.get("eta2").is_some_and(|v| v.location.is_some()),
            d2_given: values.get("d2").is_some_and(|v| v.location.is_some()),
        };
        if model.channel.vsigma == 0. || model.di + model.deltad == 0. {
            return Err(SpiceError::circuit(format!(
                "{}: HFET2 model '{}' needs VSIGMA and DI nonzero (hfeta2 divides by them)",
                resolved.card().location,
                resolved.card().name
            )));
        }
        Ok((model, defaults))
    }
}

/// `hfet2temp.c`'s per-instance quantities.
#[derive(Debug, Clone, Copy)]
struct Derived {
    channel: ChannelTemperature,
    temp: Real,
    jslw: Real,
    ggrlw: Real,
    vcrit: Real,
    gdpr: Real,
    gspr: Real,
}

/// `hfet2setup.c`'s internal nodes.
#[derive(Debug, Clone, Copy)]
struct Nodes {
    d: NodeId,
    g: NodeId,
    s: NodeId,
    sp: NodeId,
    dp: NodeId,
}

impl Nodes {
    /// A resistance between aliased prepared nodes has no series branch.
    fn series(self, derived: &Derived) -> [Real; 2] {
        [
            if self.d == self.dp { 0. } else { derived.gdpr },
            if self.s == self.sp { 0. } else { derived.gspr },
        ]
    }
}

/// HFET2, `Z` level 6.
#[derive(Debug, Clone)]
pub(in crate::devices) struct Hfet2 {
    name: String,
    model_name: String,
    terminals: Vec<NodeId>,
    nodes: Nodes,
    model: Model,
    instance: Instance,
    initial: InstanceInitial,
    temperature_cache: TemperatureCache<Derived, 1>,
}

impl Hfet2 {
    pub(crate) fn instantiate(
        i: &DeviceInstance,
        nodes: &mut NodeAppend<'_>,
        resolved: &ResolvedModel<'_>,
        context: &ModelContext,
    ) -> SpiceResult<Box<dyn Device>> {
        if resolved.levels().backend != crate::devices::ModelBackend::Hfet2 {
            return Err(SpiceError::circuit(format!(
                "{}: HFET2 needs level 6",
                i.name
            )));
        }
        if i.nodes.len() != 3 {
            return Err(SpiceError::circuit(format!(
                "{}: HFET2 needs three terminals",
                i.name
            )));
        }
        let (model, model_defaults) = Model::resolve(resolved)?;
        let (initial, setters) = Instance::resolve(i)?;
        let instance = Instance::validate(&model_defaults, &setters, &i.location)?;
        nodes.transaction(|nodes| {
            let (mut topology, [d, g, s]) = NodeSetup::new(nodes, i, "HFET2")?;
            // C creates source and drain primes in this order.
            let sp = topology.prime(model.rs, "source", s)?;
            let dp = topology.prime(model.rd, "drain", d)?;
            let nodes_of = Nodes { d, g, s, sp, dp };
            let terminals = topology.finish();
            let device = Self {
                name: i.name.clone(),
                model_name: resolved.card().name.clone(),
                terminals,
                nodes: nodes_of,
                model,
                instance,
                initial,
                temperature_cache: TemperatureCache::new(),
            };
            device.derived(context.temperature)?;
            Ok(Box::new(device) as Box<dyn Device>)
        })
    }

    /// `hfet2temp.c`. `temperature` is Celsius (`CKTtemp`), while `TEMP` is
    /// kelvin.
    fn derived(&self, temperature: Real) -> SpiceResult<Derived> {
        self.temperature_cache
            .get_or_compute([temperature], || self.compute_derived(temperature))
    }

    fn compute_derived(&self, temperature: Real) -> SpiceResult<Derived> {
        let model = &self.model;
        let instance = &self.instance;
        let temp = self.instance.temperature(temperature);
        let nominal = temperature + CELSIUS_TO_KELVIN;
        let delta = temp - nominal;
        let vt = K_OVER_Q * temp;
        let area = instance.w * instance.l;
        let derived = Derived {
            channel: ChannelTemperature {
                vt,
                tmu: model.kmu.mul_add(-delta, model.mu),
                tvto: model.kvto.mul_add(-delta, model.pol * model.threshold),
                delta_sqr: model.delta * model.delta,
                n0: model.channel.epsi * model.channel.eta * vt
                    / 2.
                    / CHARGE
                    / (model.di + model.deltad),
                two_exponentials: model.eta2_given && model.d2_given,
                n02: if model.eta2_given {
                    model.channel.epsi * model.channel.eta2 * vt / 2. / CHARGE / model.d2
                } else {
                    0.
                },
                vt2: model.vt2.unwrap_or(model.threshold),
                nmax: model.knmax.mul_add(-delta, model.nmax),
                gchi0: CHARGE * instance.w * model.kmu.mul_add(-delta, model.mu) / instance.l,
                imax: CHARGE
                    * model.knmax.mul_add(-delta, model.nmax)
                    * model.channel.vs
                    * instance.w,
                tlambda: model.klambda.mul_add(delta, model.lambda),
                // C leaves VT1/VT2 on the raw threshold, unlike TVTO.
                vt1: model.vt1.unwrap_or(
                    model.threshold + CHARGE * model.nmax * model.di / model.channel.epsi,
                ),
                cf: model.cf,
            },
            temp,
            jslw: model.js * instance.l * instance.w / 2.,
            ggrlw: model.ggr * instance.l * instance.w / 2.,
            vcrit: vt * (vt / (std::f64::consts::SQRT_2 * VCRIT_CURRENT)).ln(),
            gdpr: if model.rd == 0. { 0. } else { 1. / model.rd },
            gspr: if model.rs == 0. { 0. } else { 1. / model.rs },
        };
        let _ = area;
        if [
            derived.temp,
            derived.channel.vt,
            derived.channel.tlambda,
            derived.channel.tmu,
            derived.channel.nmax,
            derived.channel.tvto,
            derived.channel.vt1,
            derived.channel.vt2,
            derived.channel.n0,
            derived.channel.gchi0,
            derived.channel.imax,
            derived.vcrit,
        ]
        .iter()
        .any(|v| !v.is_finite())
            || derived.channel.vt == 0.
        {
            return Err(SpiceError::circuit(format!(
                "{}: HFET2 temperature derivation is out of range (check DI, DELTAD, N, \
                 NMU and the instance temperature)",
                self.name
            )));
        }
        Ok(derived)
    }

    /// `hfeta2()`: the channel, its derivatives and the two gate
    /// capacitances. `vgs` is C's `vds > 0 ? vgs : vgd` argument; `vds` is the
    /// normalized (nonnegative) drain-source voltage.
    fn hfeta2(
        &self,
        derived: &Derived,
        vgs: Real,
        vds: Real,
    ) -> SpiceResult<super::channel::IntrinsicChannel> {
        let model = &self.model;
        let channel =
            super::channel::evaluate(&model.channel, &derived.channel, &self.instance, vgs, vds);
        if channel.saturation.is_some() && !channel.is_finite() {
            return Err(SpiceError::circuit(format!(
                "{}: nonfinite hfeta2 channel equations",
                self.name
            )));
        }
        Ok(channel)
    }
}

/// `HFET2load`'s evaluated device quantities at one bias.
#[derive(Debug, Clone, Copy)]
struct Point {
    vgs: Real,
    vgd: Real,
    vds: Real,
    cg: Real,
    cd: Real,
    cgd: Real,
    gm: Real,
    gds: Real,
    ggs: Real,
    ggd: Real,
    capgs: Real,
    capgd: Real,
}

impl Hfet2 {
    /// `HFET2load`'s initialization and limiting. `raw` and `ic` are already
    /// type-normalized.
    fn select_junctions(
        &self,
        derived: &Derived,
        states: &DeviceState<'_>,
        raw: [Real; 2],
        ic: [Real; 2],
    ) -> ([Real; 2], Start) {
        Start::select(
            states,
            raw,
            ic,
            self.initial.off,
            [slot::VGS, slot::VGD],
            [
                slot::CG,
                slot::CD,
                slot::GM,
                slot::GDS,
                slot::GGS,
                slot::GGD,
            ],
            |voltage, previous, history| {
                Start::limit_junctions(
                    voltage,
                    previous,
                    history,
                    [K_OVER_Q * derived.temp; 2],
                    [derived.vcrit; 2],
                    derived.channel.tvto,
                )
            },
        )
    }

    /// `HFET2load`'s device evaluation: the single-exponential gate junctions
    /// and the `hfeta2` channel.
    fn point(&self, derived: &Derived, vgs: Real, vgd: Real) -> SpiceResult<Point> {
        let model = &self.model;
        let vt = derived.channel.vt;
        let vtn = model.n * vt;
        let arg = -vgs * model.del / vt;
        let earg = arg.exp();
        let (current, conductance) =
            crate::devices::nonlinear::exponential_junction(vgs, vtn, derived.jslw);
        let ggs = (derived.ggrlw * earg).mul_add(1. - arg, conductance);
        let mut cg = (derived.ggrlw * vgs).mul_add(earg, current);
        let arg = -vgd * model.del / vt;
        let earg = arg.exp();
        let (current, conductance) =
            crate::devices::nonlinear::exponential_junction(vgd, vtn, derived.jslw);
        let ggd = (derived.ggrlw * earg).mul_add(1. - arg, conductance);
        let cgd = (derived.ggrlw * vgd).mul_add(earg, current);
        cg += cgd;
        let mut vds = vgs - vgd;
        let inverse = vds < 0.;
        if inverse {
            vds = -vds;
        }
        // C's argument is `vds > 0 ? vgs : vgd` after the normalization.
        let channel = self.hfeta2(derived, if vds > 0. { vgs } else { vgd }, vds)?;
        let (cdrain, capgs, capgd) = if inverse {
            (-channel.cdrain, channel.capgd, channel.capgs)
        } else {
            (channel.cdrain, channel.capgs, channel.capgd)
        };
        Ok(Point {
            vgs,
            vgd,
            vds: vgs - vgd,
            cg,
            cd: cdrain - cgd,
            cgd,
            gm: channel.gm,
            gds: channel.gds,
            ggs,
            ggd,
            capgs,
            capgd,
        })
    }
}

impl Device for Hfet2 {
    fn validate_dc_inputs(&self, context: &crate::devices::ModelContext) -> SpiceResult<()> {
        self.derived(context.temperature).map(|_| ())
    }

    fn name(&self) -> &str {
        &self.name
    }
    fn designator(&self) -> char {
        'z'
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
    fn state_count(&self) -> usize {
        slot::COUNT
    }
    /// `hfet2trunc.c`: both gate charges control the timestep.
    fn truncation_slots(&self) -> Vec<usize> {
        vec![slot::QGS, slot::QGD]
    }
    /// Noiseless: `hfet2init.c` leaves `DEVnoise = NULL`.
    fn noise(
        &self,
        _context: &crate::devices::noise::NoiseContext<'_>,
        _sources: &mut Vec<crate::devices::noise::NoiseSource>,
    ) -> SpiceResult<crate::devices::noise::DeviceNoise<'_>> {
        Ok(crate::devices::noise::DeviceNoise::Noiseless)
    }
    /// Linear in `.disto`: `hfet2init.c` leaves `DEVdisto = NULL`.
    fn distortion(
        &self,
        _context: &crate::devices::distortion::DistortionContext<'_>,
    ) -> SpiceResult<crate::devices::distortion::DeviceDistortion> {
        Ok(crate::devices::distortion::DeviceDistortion::Linear)
    }
    fn observation_parameter(
        &self,
        key: &str,
        context: &ModelContext,
    ) -> SpiceResult<Option<Real>> {
        Ok(self
            .instance
            .observation_parameter(key, &self.initial, context.temperature))
    }
    /// `hfet2ask.c`'s bias quantities, `m`-scaled as C stores them.
    fn observation_operating(
        &self,
        key: &str,
        context: &ModelContext,
        voltage: &dyn Fn(NodeId) -> Real,
        transient: bool,
    ) -> SpiceResult<Option<Real>> {
        let keys = [
            "vgs", "vgd", "cg", "cd", "cgd", "gm", "gds", "ggs", "ggd", "cs", "p",
        ];
        if matches!(key, "qgs" | "cqgs" | "qgd" | "cqgd") {
            return Err(SpiceError::not_yet_ported(
                format!("{}: HFET2 @{}[{key}]", self.name, self.name),
                "src/spicelib/devices/hfet2/hfet2ask.c (HFET2qgs reads CKTstate0)",
            ));
        }
        if !keys.contains(&key) {
            return Ok(None);
        }
        if transient {
            return Err(SpiceError::not_yet_ported(
                format!("{}: transient @{}[{key}]", self.name, self.name),
                "src/spicelib/devices/hfet2/hfet2load.c (charge companion)",
            ));
        }
        let derived = self.derived(context.temperature)?;
        let pol = self.model.pol;
        let vgs = pol * (voltage(self.nodes.g) - voltage(self.nodes.sp));
        let vgd = pol * (voltage(self.nodes.g) - voltage(self.nodes.dp));
        let point = self.point(&derived, vgs, vgd)?;
        let m = self.instance.m;
        Ok(Some(match key {
            "vgs" => point.vgs,
            "vgd" => point.vgd,
            "cg" => m * point.cg,
            "cd" => m * point.cd,
            "cgd" => m * point.cgd,
            "gm" => m * point.gm,
            "gds" => m * point.gds,
            "ggs" => m * point.ggs,
            "ggd" => m * point.ggd,
            "cs" => m * (-point.cd - point.cg),
            _ => {
                let v = |n: NodeId| voltage(n);
                m * f64::mul_add(
                    point.cd + point.cg,
                    -v(self.nodes.s),
                    f64::mul_add(point.cg, v(self.nodes.g), point.cd * v(self.nodes.d)),
                )
            }
        }))
    }
    fn instance_parameter(&self, keyword: &str) -> Option<&'static str> {
        Instance::sweep_parameter(keyword)
    }
    fn with_instance_parameter(
        &self,
        parameter: &str,
        value: Real,
        context: &ModelContext,
    ) -> SpiceResult<Box<dyn Device>> {
        let instance = self.instance.swept(&self.name, parameter, value)?;
        let device = Self {
            name: self.name.clone(),
            model_name: self.model_name.clone(),
            terminals: self.terminals.clone(),
            nodes: self.nodes,
            model: self.model,
            instance,
            initial: self.initial.clone(),
            temperature_cache: TemperatureCache::new(),
        };
        device.derived(context.temperature)?;
        Ok(Box::new(device))
    }

    /// `HFET2load`: the equations, their order and the stamps are C's.
    fn stamp(&self, context: &mut StampContext<'_>) -> SpiceResult<()> {
        if context.mode.is_ac() {
            return Err(SpiceError::circuit(format!(
                "{}: HFET2 AC needs small-signal assembly",
                self.name
            )));
        }
        let derived = self.derived(context.temperature)?;
        let m = self.instance.m;
        let pol = self.model.pol;
        let voltage = |n: NodeId| context.node_voltage(n);
        let raw = [
            pol * (voltage(self.nodes.g) - voltage(self.nodes.sp)),
            pol * (voltage(self.nodes.g) - voltage(self.nodes.dp)),
        ];
        let ic = [
            self.initial.values[0].unwrap_or_else(|| voltage(self.nodes.d) - voltage(self.nodes.s)),
            self.initial.values[1].unwrap_or_else(|| voltage(self.nodes.g) - voltage(self.nodes.s)),
        ];
        let ([vgs, vgd], start) = self.select_junctions(&derived, &context.states, raw, ic);
        let mut point = self.point(&derived, vgs, vgd)?;
        let (mut cg, mut cd, mut ggs, mut ggd) = (point.cg, point.cd, point.ggs, point.ggd);
        let gds = point.gds;
        let mut cgd = point.cgd;
        let (mut cgspp, mut cgdpp) = (0., 0.);
        let (mut qgs, mut qgd) = (0., 0.);
        if matches!(
            context.mode,
            AnalysisMode::Transient { .. } | AnalysisMode::DcSweep
        ) {
            let previous = [
                accepted(&context.states, slot::VGS, vgs),
                accepted(&context.states, slot::VGD, vgd),
            ];
            let step = charge_step(
                &mut context.states,
                context.mode,
                context.integration,
                "HFET2",
                [
                    ChargeInput::new(slot::QGS, vgs, previous[0], point.capgs),
                    ChargeInput::new(slot::QGD, vgd, previous[1], point.capgd),
                ],
            )?;
            [qgs, qgd] = step.charges;
            if let Some([gs, gd]) = step.companions {
                ggs += gs.conductance;
                cgspp = gs.derivative;
                cg += gs.derivative;
                ggd += gd.conductance;
                cgdpp = gd.derivative;
                cg += gd.derivative;
                cd -= gd.derivative;
                cgd += gd.derivative;
            }
        }
        point = Point {
            cg,
            cd,
            cgd,
            gds,
            ggs,
            ggd,
            ..point
        };
        start.report(
            &mut context.states,
            Linearization {
                voltage: [point.vgs, point.vgd],
                gate_current: point.cg,
                drain_current: point.cd,
                transconductance: point.gm,
                output_conductance: point.gds,
                source_conductance: point.ggs,
                drain_conductance: point.ggd,
            },
        )?;
        for (index, value) in [
            (slot::VGS, point.vgs),
            (slot::VGD, point.vgd),
            (slot::CG, point.cg),
            (slot::CD, point.cd),
            (slot::CGD, point.cgd),
            (slot::GM, point.gm),
            (slot::GDS, point.gds),
            (slot::GGS, point.ggs),
            (slot::GGD, point.ggd),
            (slot::QGS, qgs),
            (slot::CQGS, cgspp),
            (slot::QGD, qgd),
            (slot::CQGD, cgdpp),
        ] {
            context.states.set(index, value)?;
        }
        let n = self.nodes;
        let (ggs, ggd) = (point.ggs, point.ggd);
        let (gm, gds, cgd, cg) = (point.gm, point.gds, point.cgd, point.cg);
        let [gdpr, gspr] = n.series(&derived);
        let ceqgd = pol * f64::mul_add(ggd, -point.vgd, cgd);
        let ceqgs = pol * f64::mul_add(ggs, -point.vgs, cg - cgd);
        let cdreq = pol
            * f64::mul_add(
                gm,
                -point.vgs,
                f64::mul_add(gds, -point.vds, point.cd + cgd),
            );
        context.stamp_rhs(n.g, m * (-ceqgs - ceqgd))?;
        context.stamp_rhs(n.dp, m * (-cdreq + ceqgd))?;
        context.stamp_rhs(n.sp, m * (cdreq + ceqgs))?;
        for (row, col, value) in [
            (n.d, n.dp, -m * gdpr),
            (n.g, n.dp, -m * ggd),
            (n.g, n.sp, -m * ggs),
            (n.s, n.sp, -m * gspr),
            (n.dp, n.d, -m * gdpr),
            (n.dp, n.g, m * (gm - ggd)),
            (n.dp, n.sp, m * (-gds - gm)),
            (n.sp, n.g, m * (-ggs - gm)),
            (n.sp, n.s, -m * gspr),
            (n.sp, n.dp, -m * gds),
            (n.d, n.d, m * gdpr),
            (n.g, n.g, m * (ggd + ggs)),
            (n.s, n.s, m * gspr),
            (n.dp, n.dp, m * (gdpr + gds + ggd)),
            (n.sp, n.sp, m * (gspr + gds + gm + ggs)),
        ] {
            context.stamp(row, col, value)?;
        }
        Ok(())
    }

    /// `hfet2.c`'s `IFparm` tables under C's `sgen` walk (#164): the record
    /// replay of `hfet2temp.c`/`hfet2mpar.c`/`hfet2param.c` with the load being
    /// this device rebuilt from those records.
    ///
    /// Three C behaviours the replay reproduces, all verified against the
    /// reference binary:
    ///
    /// - `d2` and `eta2` gate the `nsc` topology branch through their *given*
    ///   flags (see [`MODEL_PARAMETERS`]), so their columns are honest
    ///   derivatives only when the card gave both, and given-flag jumps
    ///   otherwise.
    /// - `hfet2ask.c` answers the instance temperature in Celsius while
    ///   `hfet2param.c` stores it in kelvin.
    /// - `HFET2_VT1`/`VT2` are model fields that follow the raw `VTO` (not the
    ///   type-scaled `TVTO`) whenever they are not given.
    fn sensitivity(
        &self,
        context: &ModelContext,
    ) -> SpiceResult<Box<dyn crate::devices::sensitivity::DeviceSensitivity + '_>> {
        Ok(Box::new(Hfet2Sensitivity::new(self, context)))
    }

    /// `hfet2acl.c`: the bias conductances plus `j omega` times the gate
    /// capacitances.
    fn assemble_small_signal(
        &self,
        context: &mut LinearContext<'_>,
        bias: &Vector,
    ) -> SpiceResult<()> {
        let derived = self.derived(context.model_context.temperature)?;
        let row = |node: NodeId| context.unknowns.node_row(node);
        let voltage = |node: NodeId| row(node).and_then(|r| bias.get(r)).unwrap_or(0.);
        let pol = self.model.pol;
        let vgs = pol * (voltage(self.nodes.g) - voltage(self.nodes.sp));
        let vgd = pol * (voltage(self.nodes.g) - voltage(self.nodes.dp));
        let m = self.instance.m;
        let point = self.point(&derived, vgs, vgd)?;
        let (gm, gds, ggs, ggd) = (point.gm, point.gds, point.ggs, point.ggd);
        let n = self.nodes;
        let [gdpr, gspr] = n.series(&derived);
        for (a, b, value) in [
            (n.d, n.d, m * gdpr),
            (n.g, n.g, m * (ggd + ggs)),
            (n.s, n.s, m * gspr),
            (n.dp, n.dp, m * (gdpr + gds + ggd)),
            (n.sp, n.sp, m * (gspr + gds + gm + ggs)),
            (n.d, n.dp, -m * gdpr),
            (n.g, n.dp, -m * ggd),
            (n.g, n.sp, -m * ggs),
            (n.s, n.sp, -m * gspr),
            (n.dp, n.d, -m * gdpr),
            (n.dp, n.g, m * (gm - ggd)),
            (n.dp, n.sp, m * (-gds - gm)),
            (n.sp, n.g, m * (-ggs - gm)),
            (n.sp, n.s, -m * gspr),
            (n.sp, n.dp, -m * gds),
        ] {
            if let (Some(r), Some(c)) = (row(a), row(b)) {
                context.system.a.add(r, c, value)?;
            }
        }
        context.nodal([n.g, n.sp], m * point.capgs, true)?;
        context.nodal([n.g, n.dp], m * point.capgd, true)?;
        Ok(())
    }
    /// `hfet2pzl.c` is `hfet2acl.c` with `s` in place of `j omega`.
    fn assemble_pole_zero(
        &self,
        context: &mut LinearContext<'_>,
        bias: &Vector,
    ) -> SpiceResult<()> {
        self.assemble_small_signal(context, bias)
    }
}

// ---- `.sens` (#164) -------------------------------------------------------

/// `hfet2.c`'s `HFET2mPTable`, in C's order, filtered the way `set_param()`
/// (`cktsgen.c`) and `sens_getp()` filter it: `type` is `IF_STRING`,
/// `nhfet`/`phfet` are `IF_FLAG` and `vto` is the `IOPR` alias of `vt0`, so
/// `(IF_SET|IF_ASK|IF_REAL)` drops all four.
///
/// `d2` and `eta2` are ordinary entries here, but C's setters mark them
/// *given*, and that flag is a topology switch rather than a value:
/// `hfet2temp.c` fills `N02` under `eta2Given` alone, while `hfet2load.c` reads
/// the `nsc` branch under `eta2Given && d2Given`. The walk never clears a
/// given flag, so perturbing either of them can change which model the
/// perturbed load evaluates. See [`Hfet2Sensitivity::load`].
const MODEL_PARAMETERS: &[SensitivityParameter] = &[
    SensitivityParameter::dc("cf"),
    SensitivityParameter::dc("d1"),
    SensitivityParameter::dc("d2"),
    SensitivityParameter::dc("del"),
    SensitivityParameter::dc("delta"),
    SensitivityParameter::dc("deltad"),
    SensitivityParameter::dc("di"),
    SensitivityParameter::dc("epsi"),
    SensitivityParameter::dc("eta"),
    SensitivityParameter::dc("eta1"),
    SensitivityParameter::dc("eta2"),
    SensitivityParameter::dc("gamma"),
    SensitivityParameter::dc("ggr"),
    SensitivityParameter::dc("js"),
    SensitivityParameter::dc("klambda"),
    SensitivityParameter::dc("kmu"),
    SensitivityParameter::dc("knmax"),
    SensitivityParameter::dc("kvto"),
    SensitivityParameter::dc("lambda"),
    SensitivityParameter::dc("m"),
    SensitivityParameter::dc("mc"),
    SensitivityParameter::dc("mu"),
    SensitivityParameter::dc("n"),
    SensitivityParameter::dc("nmax"),
    SensitivityParameter::dc("p"),
    SensitivityParameter::dc("rd"),
    SensitivityParameter::dc("rdi"),
    SensitivityParameter::dc("rs"),
    SensitivityParameter::dc("rsi"),
    SensitivityParameter::dc("sigma0"),
    SensitivityParameter::dc("vs"),
    SensitivityParameter::dc("vsigma"),
    SensitivityParameter::dc("vsigmat"),
    SensitivityParameter::dc("vt0"),
    SensitivityParameter::dc("vt1"),
    SensitivityParameter::dc("vt2"),
];

/// `hfet2.c`'s `HFET2pTable` under the same filter: `off` is `IF_FLAG` and the
/// probe parameters are `OP` (ask only).
///
/// `m` passes the table filter but not the ask: `hfet2ask.c` has no
/// `HFET2_M` case (`hfetask.c` does have `HFETA_M`), so `sgen` drops the column
/// C never reports.
const INSTANCE_PARAMETERS: &[SensitivityParameter] = &[
    SensitivityParameter::dc("l"),
    SensitivityParameter::dc("w"),
    SensitivityParameter::dc("icvds"),
    SensitivityParameter::dc("icvgs"),
    SensitivityParameter::dc("temp"),
    SensitivityParameter::dc("dtemp"),
];

/// `hfet2temp.c`/`hfet2mpar.c`/`hfet2setup.c` replayed on C's records, with
/// the load being the port's own [`Hfet2`] rebuilt from them.
#[derive(Debug)]
pub(super) struct Hfet2Sensitivity<'a> {
    device: &'a Hfet2,
    circuit_kelvin: Real,
}

impl<'a> Hfet2Sensitivity<'a> {
    pub(super) fn new(device: &'a Hfet2, context: &ModelContext) -> Self {
        Self {
            device,
            circuit_kelvin: context.temperature + CELSIUS_TO_KELVIN,
        }
    }
}

impl SensitivityKernel for Hfet2Sensitivity<'_> {
    fn family(&self) -> SensitivityFamily {
        SensitivityFamily::Hfet2
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

    fn records(&self) -> SpiceResult<(SensitivityRecord, SensitivityRecord)> {
        let m = &self.device.model;
        let mut model = SensitivityRecord::new(self.family(), ParameterScope::Model);
        for (keyword, value, given) in [
            ("cf", m.cf, true),
            ("d1", m.channel.d1, true),
            ("d2", m.d2, m.d2_given),
            ("del", m.del, true),
            ("delta", m.delta, true),
            ("deltad", m.deltad, true),
            ("di", m.di, true),
            ("epsi", m.channel.epsi, true),
            ("eta", m.channel.eta, true),
            ("eta1", m.channel.eta1, true),
            ("eta2", m.channel.eta2, m.eta2_given),
            ("gamma", m.channel.gamma, true),
            ("ggr", m.ggr, true),
            ("js", m.js, true),
            ("klambda", m.klambda, true),
            ("kmu", m.kmu, true),
            ("knmax", m.knmax, true),
            ("kvto", m.kvto, true),
            ("lambda", m.lambda, true),
            ("m", m.channel.m, true),
            ("mc", m.channel.mc, true),
            ("mu", m.mu, true),
            ("n", m.n, true),
            ("nmax", m.nmax, true),
            ("p", m.channel.pm, true),
            ("rd", m.rd, true),
            ("rdi", m.channel.rdi, true),
            ("rs", m.rs, true),
            ("rsi", m.channel.rsi, true),
            ("sigma0", m.channel.sigma0, true),
            ("vs", m.channel.vs, true),
            ("vsigma", m.channel.vsigma, true),
            ("vsigmat", m.channel.vsigmat, true),
            // `vt0` is C's `VTO`: raw, never scaled by the type in place
            // (`hfet2temp.c` writes `TVTO` instead). `vt1`/`vt2` are the model
            // fields `hfet2setup.c` zeroes when not given, and `temperature`
            // below fills them the way `HFET2temp` does.
            ("vt0", m.threshold, true),
            ("vt1", m.vt1.unwrap_or(0.), m.vt1.is_some()),
            ("vt2", m.vt2.unwrap_or(0.), m.vt2.is_some()),
        ] {
            model = model.with(keyword, value, given)?;
        }
        let mut instance = self.device.instance.sensitivity_record(
            self.family(),
            &self.device.initial,
            self.circuit_kelvin,
        )?;
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
        ask_hfet(scope, keyword, records)
    }

    fn set(
        &self,
        scope: ParameterScope,
        keyword: &str,
        value: Real,
        model: &mut SensitivityRecord,
        instance: &mut SensitivityRecord,
    ) -> SpiceResult<()> {
        set_hfet(scope, keyword, value, model, instance)
    }

    /// `HFET2setup`'s record-visible part. Its model defaults are already in
    /// the records ([`Hfet2::records`] resolves the card), and its
    /// `HFET2_VT1 = 0`/`VT2 = 0` placeholders are overwritten by the
    /// `temperature` call that follows every setup, so nothing is left to do.
    fn setup(
        &self,
        _model: &mut SensitivityRecord,
        _instance: &mut SensitivityRecord,
    ) -> SpiceResult<()> {
        Ok(())
    }

    /// `hfet2temp.c`'s model-side arithmetic, which is the part that persists.
    ///
    /// `HFET2_VT1` and `VT2` are model fields (despite the `here->`-looking
    /// macros): when not given they follow the *raw* `VTO`, and the channel
    /// mixes them with the type-scaled `TVTO`. Everything else the routine
    /// writes is per instance and re-derived by the rebuilt device.
    fn temperature(
        &self,
        model: &mut SensitivityRecord,
        instance: &mut SensitivityRecord,
    ) -> SpiceResult<()> {
        let threshold = model.value("vt0")?;
        if !model.given("vt1") {
            let added = CHARGE * model.value("nmax")? * model.value("di")? / model.value("epsi")?;
            model.store("vt1", threshold + added)?;
        }
        if !model.given("vt2") {
            model.store("vt2", threshold)?;
        }
        let _ = instance;
        Ok(())
    }

    reconstructed_reset_loads!(
        /// `HFET2acLoad` after `cktsens.c`'s reset load: `gm`, `gds`, `ggs`,
        /// `ggd` are the `CKTstate0` values `hfet2load.c` stored at zero gate
        /// voltages with the reset records, the charge slots `qgs`/`qgd` are
        /// zero, and `RD`/`RS` and `m` come from `current`.
        hfet2
    );
}

impl Hfet2Sensitivity<'_> {
    /// The port's HFET2 with the records' parameters.
    fn hfet2(&self, current: RecordPair<'_>) -> SpiceResult<Hfet2> {
        let (m, i) = (current.model, current.instance);
        let r = |keyword: &str| m.value(keyword);
        // The record's given flags retain the C sensitivity walk's topology switches.
        let model = Model {
            channel: ChannelModel {
                eta: r("eta")?,
                vs: r("vs")?,
                rsi: r("rsi")?,
                rdi: r("rdi")?,
                vsigmat: r("vsigmat")?,
                vsigma: r("vsigma")?,
                sigma0: r("sigma0")?,
                eta2: r("eta2")?,
                gamma: r("gamma")?,
                m: r("m")?,
                eta1: r("eta1")?,
                d1: r("d1")?,
                epsi: r("epsi")?,
                mc: r("mc")?,
                pm: r("p")?,
            },
            pol: self.device.model.pol,
            threshold: r("vt0")?,
            cf: r("cf")?,
            d2: r("d2")?,
            del: r("del")?,
            delta: r("delta")?,
            deltad: r("deltad")?,
            di: r("di")?,
            ggr: r("ggr")?,
            js: r("js")?,
            klambda: r("klambda")?,
            kmu: r("kmu")?,
            knmax: r("knmax")?,
            kvto: r("kvto")?,
            lambda: r("lambda")?,
            mu: r("mu")?,
            n: r("n")?,
            nmax: r("nmax")?,
            rd: r("rd")?,
            rs: r("rs")?,
            vt1: Some(r("vt1")?),
            vt2: Some(r("vt2")?),
            eta2_given: m.given("eta2"),
            d2_given: m.given("d2"),
        };
        let instance = Instance::from_record(i)?;
        Ok(Hfet2 {
            name: self.device.name.clone(),
            model_name: self.device.model_name.clone(),
            terminals: self.device.terminals.clone(),
            nodes: self.device.nodes,
            model,
            instance,
            initial: self.device.initial.clone(),
            temperature_cache: TemperatureCache::new(),
        })
    }
}

impl ResetAcDevice for Hfet2 {
    /// `hfet2acl.c`'s real stamps; its `xgs`/`xgd` imaginary parts are zero.
    fn stamp_reset_ac(
        state: &Self,
        current: &Self,
        _omega: Real,
        context: &mut LinearContext<'_>,
    ) -> SpiceResult<()> {
        let temperature = context.model_context.temperature;
        // `type * (0 - 0)`: a signed zero, as C's at zero node voltages.
        let zero = state.model.pol * 0.;
        let point = state.point(&state.derived(temperature)?, zero, zero)?;
        let derived = current.derived(temperature)?;
        let (gm, gds, ggs, ggd) = (point.gm, point.gds, point.ggs, point.ggd);
        let m = current.instance.m;
        let n = current.nodes;
        let [gdpr, gspr] = n.series(&derived);
        let stamps = [
            (n.d, n.d, m * gdpr),
            (n.g, n.g, m * (ggd + ggs)),
            (n.s, n.s, m * gspr),
            (n.dp, n.dp, m * (gdpr + gds + ggd)),
            (n.sp, n.sp, m * (gspr + gds + gm + ggs)),
            (n.d, n.dp, -(m * gdpr)),
            (n.g, n.dp, -(m * ggd)),
            (n.g, n.sp, -(m * ggs)),
            (n.s, n.sp, -(m * gspr)),
            (n.dp, n.d, -(m * gdpr)),
            (n.dp, n.g, m * (-ggd + gm)),
            (n.dp, n.sp, m * (-gds - gm)),
            (n.sp, n.g, m * (-ggs - gm)),
            (n.sp, n.s, -(m * gspr)),
            (n.sp, n.dp, -(m * gds)),
        ];
        stamp_reset(
            &current.name,
            node_stamps(&stamps, StampPart::Real),
            context,
        )
    }
}

static MODEL_SCHEMA: ScalarSchema<'static> = ScalarSchema::new(MODEL);
