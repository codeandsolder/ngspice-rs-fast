//! `.sens` for BSIM4: `cktsens.c`'s generic perturbation replayed on C's
//! structs.
//!
//! C's BSIM4 registers no `DEVsen*` routine, so `sens_sens()` perturbs every
//! model and instance parameter `sgen` enumerates (`IF_SET|IF_ASK|IF_REAL`,
//! not redundant, and askable: `b4mask.c`/`b4ask.c` must answer it, see
//! [`super::asks`]) through `BSIM4mParam`/`BSIM4param`, `BSIM4setup`,
//! `BSIM4temp` and `BSIM4load`/`BSIM4acLoad` on the isolated instance. The
//! records are C's structs themselves: every scalar member of `sBSIM4model`,
//! and of `sBSIM4instance` followed by its `bsim4SizeDependParam` (C's
//! `pParam`), by C member name, so the ports of those routines run on them
//! unchanged.
//!
//! C behaviours the replay reproduces:
//!
//! * The asks read the stored member, and the setters convert again: TNOM is
//!   asked in kelvin and stored `+ 273.15`, so after its row the model's
//!   TNOM stays 273.15 K higher; NDEP above 1e20 and NSD/NGATE above 1e23
//!   (and their L/W/P coefficients) are stored `* 1e-6`, so a perturbation
//!   across the threshold jumps.
//! * `BSIM4setup` runs again before every parameter and is not idempotent:
//!   it creates an internal node only while its member is still zero, and
//!   otherwise aliases it. So from the first parameter on, the drain,
//!   source, gate and mid-gate primes it created alias their external nodes
//!   and `qNode` is ground (the body nodes are kept); the loads and
//!   `delta_Y`'s elements use those aliases. Where an alias is ground (a
//!   grounded source with series resistance, `trnqsMod`), the next re-run
//!   would create a node, which `cktsens.c` treats as a fatal internal error
//!   (`controlled_exit`); the port fails there with C's message.
//! * `BSIM4checkModel` does not run: its `static` model name was set by the
//!   deck's own temperature pass.
//! * AC: `CKTunsetup` resets the node members before each frequency's
//!   `CKTsetup`, which creates the nodes again. The reset load (zero
//!   voltages, no charge computation) writes the DC small-signal members and
//!   zeroes the C-V derivatives; `capbd`/`capbs`, which only a charge load
//!   writes, hold what an earlier analysis's `MODEINITSMSIG` load left (zero
//!   when none ran). Each parameter's AC load reads those, the reset's
//!   `CKTstate0`, and the setup/temperature members of the present records.
// C-compatible numerical expressions are kept in their reference order:
// Clippy's fused multiply-add suggestions change IEEE-754 rounding.
#![expect(
    clippy::suboptimal_flops,
    reason = "SPICE numerical reference parity requires the original non-fused C operation order"
)]
#![expect(
    non_snake_case,
    reason = "preserves upstream C identifiers for equation-by-equation comparison"
)] // C's member names.

use std::rc::Rc;
use std::sync::OnceLock;

use super::device::{Bsim4, ac_stamp, reset_load};
use super::fields::{Instance, Model, SizeParam};
use super::host::node;
use super::{Access, InputType, asks, elements, parameters, set, temp};
use crate::devices::schema::temperature_kelvin;
use crate::devices::sensitivity::{
    AcReset, Bsim4InstanceSnapshot, ParameterScope, RecordPair, SensitivityFamily,
    SensitivityKernel, SensitivityLoad, SensitivityMode, SensitivityParameter as P,
    SensitivityRecord as Record,
};
use crate::devices::{Device, LinearContext, ModelContext, StampContext};
use crate::primitives::{NodeId, Real, SpiceError, SpiceResult};

/// The internal nodes as `(own index, member)`: drain, source, gate and
/// mid-gate primes, the three body nodes and the NQS node.
const INTERNAL: [i32; 8] = [
    node::DP,
    node::SP,
    node::GP,
    node::GM,
    node::DB,
    node::BP,
    node::SB,
    node::Q,
];

/// The instance's internal node members in [`INTERNAL`] order.
const fn internal_members(h: &Instance) -> [i32; 8] {
    [
        h.dNodePrime,
        h.sNodePrime,
        h.gNodePrime,
        h.gNodeMid,
        h.dbNode,
        h.bNodePrime,
        h.sbNode,
        h.qNode,
    ]
}

/// The creation bits for the members that hold their own node.
fn created_bits(h: &Instance) -> u32 {
    internal_members(h)
        .into_iter()
        .zip(INTERNAL)
        .enumerate()
        .filter(|(_, (member, own))| member == own)
        .fold(0, |bits, (k, _)| bits | (1 << k))
}

/// `sgen`'s parameters of one table: real, settable and askable entries
/// that are not redundant, in table order.
fn table(rows: &'static [super::Parameter], asks: &[(&str, &str)]) -> Vec<P> {
    rows.iter()
        .filter(|p| {
            p.input_type == InputType::Real
                && p.access == Access::ReadWrite
                && !p.redundant
                && asks.iter().any(|(id, _)| *id == p.c_id)
        })
        .map(|p| P::dc(p.name))
        .collect()
}

fn model_table() -> &'static [P] {
    static TABLE: OnceLock<Vec<P>> = OnceLock::new();
    TABLE.get_or_init(|| table(parameters::MODEL, asks::MODEL))
}

fn instance_table() -> &'static [P] {
    static TABLE: OnceLock<Vec<P>> = OnceLock::new();
    TABLE.get_or_init(|| table(parameters::INSTANCE, asks::INSTANCE))
}

/// One BSIM4 instance as `.sens` sees it.
pub(super) struct Bsim4Sensitivity<'a> {
    device: &'a Bsim4,
    context: ModelContext,
    model: Model,
    instance: Instance,
    size: SizeParam,
}

impl std::fmt::Debug for Bsim4Sensitivity<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Bsim4Sensitivity")
            .field("device", &self.device.name)
            .finish_non_exhaustive()
    }
}

impl<'a> Bsim4Sensitivity<'a> {
    /// The device's structs as the deck's own setup and temperature pass
    /// left them.
    ///
    /// # Errors
    /// The device's setup failing.
    pub(crate) fn new(device: &'a Bsim4, context: &ModelContext) -> SpiceResult<Self> {
        let prepared = device.ensure(context.temperature, context.nominal_temperature)?;
        Ok(Self {
            device,
            context: *context,
            model: prepared.model.clone(),
            instance: device.live_copy(&prepared),
            size: prepared.size.clone(),
        })
    }

    fn kelvin(&self, celsius: Real) -> SpiceResult<Real> {
        temperature_kelvin(celsius, &self.device.location)
    }

    /// C node number 0: ground, or an external node tied to ground.
    fn grounded(&self, local: i32) -> bool {
        local == node::GROUND || self.device.node_id(local) == Some(NodeId::GROUND)
    }

    /// The structs of `pair` as one load sees them.
    fn structs(pair: RecordPair<'_>) -> SpiceResult<(Model, Instance, SizeParam)> {
        let snapshot = pair.instance.bsim4_instance()?;
        Ok((
            pair.model.bsim4_model()?.clone(),
            snapshot.instance.clone(),
            snapshot.size.clone(),
        ))
    }
}

#[derive(Clone, Copy)]
struct Accessor {
    parameter: &'static super::Parameter,
    ask: &'static str,
}

fn accessor(scope: ParameterScope, keyword: &str) -> SpiceResult<Accessor> {
    use std::collections::BTreeMap;
    static ACCESSORS: OnceLock<[BTreeMap<&'static str, Accessor>; 2]> = OnceLock::new();
    let tables = ACCESSORS.get_or_init(|| {
        [
            (parameters::MODEL, asks::MODEL),
            (parameters::INSTANCE, asks::INSTANCE),
        ]
        .map(|(rows, asks)| {
            rows.iter()
                .filter_map(|p| {
                    let (_, ask) = asks.iter().find(|(id, _)| *id == p.c_id)?;
                    Some((p.name, Accessor { parameter: p, ask }))
                })
                .collect()
        })
    });
    let index = match scope {
        ParameterScope::Model => 0,
        ParameterScope::Instance => 1,
    };
    tables[index].get(keyword).copied().ok_or_else(|| {
        SpiceError::circuit(format!("BSIM4 has no real ask/set parameter {keyword}"))
    })
}

impl SensitivityKernel for Bsim4Sensitivity<'_> {
    fn family(&self) -> SensitivityFamily {
        SensitivityFamily::Bsim4
    }

    fn model(&self) -> Option<&str> {
        Some(&self.device.model_name)
    }

    fn model_parameters(&self) -> &'static [P] {
        model_table()
    }

    fn instance_parameters(&self) -> &'static [P] {
        instance_table()
    }

    fn records(&self) -> SpiceResult<(Record, Record)> {
        Ok((
            Record::from_bsim4_model(self.model.clone()),
            Record::from_bsim4_instance(Bsim4InstanceSnapshot {
                instance: self.instance.clone(),
                size: self.size.clone(),
                unset: false,
                created: created_bits(&self.instance),
            }),
        ))
    }

    /// `BSIM4mAsk`/`BSIM4ask`: the member the ask reads, unconverted.
    fn ask(
        &self,
        scope: ParameterScope,
        keyword: &str,
        records: RecordPair<'_>,
    ) -> SpiceResult<Real> {
        let accessor = accessor(scope, keyword)?;
        let model = records.model.bsim4_model()?;
        let instance = records.instance.bsim4_instance()?;
        match scope {
            ParameterScope::Model => model.real(accessor.ask),
            ParameterScope::Instance => instance.instance.real(accessor.ask),
        }
        .ok_or_else(|| SpiceError::circuit(format!("BSIM4 cannot ask {keyword}")))
    }

    /// `BSIM4mParam`/`BSIM4param`: the setter's conversion, then the member
    /// and its given bit.
    fn set(
        &self,
        scope: ParameterScope,
        keyword: &str,
        value: Real,
        model: &mut Record,
        instance: &mut Record,
    ) -> SpiceResult<()> {
        // Resolve both typed fields before mutating either. A generated
        // descriptor cannot write an integer or a given bit as a real.
        let p = accessor(scope, keyword)?.parameter;
        let field = p
            .c_field
            .ok_or_else(|| SpiceError::circuit(format!("BSIM4 has no setter for {keyword}")))?;
        let model = model.bsim4_model_mut()?;
        let instance = instance.bsim4_instance_mut()?;
        let invalid =
            || SpiceError::circuit(format!("BSIM4 invalid real setter layout for {keyword}"));
        match scope {
            ParameterScope::Model => {
                let value = super::bsim4_model_real(p, value, &self.device.location)?;
                if model.real(field).is_none() {
                    return Err(invalid());
                }
                if let Some(given) = p.c_given {
                    *model.flag_mut(given).ok_or_else(invalid)? = true;
                }
                *model.real_mut(field).ok_or_else(invalid)? = value;
            }
            ParameterScope::Instance => {
                let h = &mut instance.instance;
                if h.real(field).is_none() {
                    return Err(invalid());
                }
                if let Some(given) = p.c_given {
                    *h.flag_mut(given).ok_or_else(invalid)? = true;
                }
                *h.real_mut(field).ok_or_else(invalid)? = value;
            }
        }
        Ok(())
    }

    /// `BSIM4unsetup`: it deletes an internal node only while its member
    /// still differs from the external node, then zeroes every member.
    /// `CKTunsetup` exits when a node the last `CKTsetup` created survives,
    /// which happens once a `.sens` setup re-run has aliased it.
    fn unsetup(&self, model: &mut Record, instance: &mut Record) -> SpiceResult<()> {
        model.bsim4_model()?;
        let snapshot = instance.bsim4_instance_mut()?;
        let h = &mut snapshot.instance;
        let created = snapshot.created;
        let survives = internal_members(h)
            .into_iter()
            .zip(INTERNAL)
            .enumerate()
            .any(|(k, (member, own))| created & (1 << k) != 0 && member != own);
        if survives {
            return Err(SpiceError::circuit(format!(
                "{}: Internal Error: incomplete CKTunsetup(), this will cause serious \
                 problems (cktsens.c's next frequency: BSIM4unsetup keeps the internal \
                 nodes a .sens setup re-run aliased, and C exits here)",
                self.device.name
            )));
        }
        snapshot.created = 0;
        h.dNodePrime = node::GROUND;
        h.sNodePrime = node::GROUND;
        h.gNodePrime = node::GROUND;
        h.gNodeMid = node::GROUND;
        h.dbNode = node::GROUND;
        h.bNodePrime = node::GROUND;
        h.sbNode = node::GROUND;
        h.qNode = node::GROUND;
        snapshot.unset = true;
        Ok(())
    }

    /// `BSIM4setup` on the isolated model and instance, run again: the
    /// defaults, then the node members as C's re-run leaves them.
    fn setup(&self, model: &mut Record, instance: &mut Record) -> SpiceResult<()> {
        let mut warnings = Vec::new();
        let mut m = model.bsim4_model()?.clone();
        set::setup_model(
            &mut m,
            self.kelvin(self.context.nominal_temperature)?,
            &mut warnings,
        )?;
        let snapshot = instance.bsim4_instance_mut()?;
        let mut h = snapshot.instance.clone();
        let unset = snapshot.unset;
        let previous = [
            h.dNodePrime,
            h.sNodePrime,
            h.gNodePrime,
            h.gNodeMid,
            h.qNode,
        ];
        set::setup_instance(&m, &mut h, self.device.noise_analysis, &mut warnings)?;
        // `if (create && field == 0) CKTmkVolt(...) else field = alias;`
        let primes = [
            (node::DP, h.dNode),
            (node::SP, h.sNode),
            (node::GP, h.gNodeExt),
            (node::GM, h.gNodeExt),
            (node::Q, node::GROUND),
        ];
        let fresh = [
            h.dNodePrime,
            h.sNodePrime,
            h.gNodePrime,
            h.gNodeMid,
            h.qNode,
        ];
        let mut result = [0; 5];
        for (k, ((own, alias), (was, now))) in primes
            .into_iter()
            .zip(previous.into_iter().zip(fresh))
            .enumerate()
        {
            let create = now == own;
            result[k] = if create && (unset || self.grounded(was)) {
                if !unset {
                    return Err(SpiceError::circuit(format!(
                        "{}: Internal Error: node allocation in DEVsetup() during \
                         sensitivity analysis (cktsens.c exits here): b4set.c would create \
                         an internal node again because its alias is ground",
                        self.device.name
                    )));
                }
                own
            } else {
                alias
            };
        }
        [
            h.dNodePrime,
            h.sNodePrime,
            h.gNodePrime,
            h.gNodeMid,
            h.qNode,
        ] = result;
        *model.bsim4_model_mut()? = m;
        if unset {
            snapshot.created = created_bits(&h);
        }
        snapshot.instance = h;
        snapshot.unset = false;
        Ok(())
    }

    /// `BSIM4temp` on the isolated model and instance, without
    /// `BSIM4checkModel` (its `static` model name is already set).
    fn temperature(&self, model: &mut Record, instance: &mut Record) -> SpiceResult<()> {
        let mut warnings = Vec::new();
        let mut m = model.bsim4_model()?.clone();
        let snapshot = instance.bsim4_instance_mut()?;
        let mut h = snapshot.instance.clone();
        let locals = temp::temperature_model(
            &mut m,
            self.kelvin(self.context.temperature)?,
            &mut warnings,
        );
        let before = h.clone();
        let size = match temp::temperature_instance(
            &mut m,
            &mut h,
            &locals,
            false,
            &self.device.model_name,
            &self.device.name,
            &mut warnings,
        ) {
            Ok(size) => size,
            // `sens_temp()` ignores BSIM4temp's E_BADPARM (`(void)`): the
            // model pass stands, the instance keeps its members, and its
            // `pParam` is the freshly allocated (zeroed) set with only the
            // geometry assigned before the failing check.
            Err(SpiceError::NotYetPorted { .. } | SpiceError::Unsupported { .. }) => {
                return Err(SpiceError::circuit(format!(
                    "{}: BSIM4 temperature pass failed under .sens",
                    self.device.name
                )));
            }
            Err(_) => {
                h = before;
                rejected_size(&m, &h)
            }
        };
        *model.bsim4_model_mut()? = m;
        snapshot.instance = h;
        snapshot.size = size;
        Ok(())
    }

    fn setup_elements(&self, setup: RecordPair<'_>) -> SpiceResult<Option<Vec<(NodeId, NodeId)>>> {
        let m = setup.model.bsim4_model()?;
        let h = &setup.instance.bsim4_instance()?.instance;
        Ok(Some(
            elements::matrix_elements(m, h)
                .into_iter()
                .filter_map(|(row, col)| {
                    Some((self.device.node_id(row)?, self.device.node_id(col)?))
                })
                .collect(),
        ))
    }

    /// `BSIM4load` (`MODEDCOP | MODEINITFLOAT`) on the present structs.
    fn load(
        &self,
        _setup: RecordPair<'_>,
        current: RecordPair<'_>,
        mode: SensitivityMode,
    ) -> SpiceResult<SensitivityLoad> {
        if mode == SensitivityMode::Ac {
            return Err(SpiceError::circuit(
                "BSIM4 AC sensitivity loads go through the reset load",
            ));
        }
        let (m, h, size) = Self::structs(current)?;
        Ok(SensitivityLoad::Replacement(Box::new(
            self.device.stand_in(m, h, size, &self.context),
        )))
    }

    fn stale_ac_fields(&self) -> Option<&'static str> {
        Some("src/spicelib/devices/bsim4/b4acld.c (capbd/capbs)")
    }

    /// `BSIM4acLoad` after `cktsens.c`'s reset load (see the module notes).
    fn reset_ac_load(
        &self,
        _setup: RecordPair<'_>,
        reset: AcReset<'_>,
        current: RecordPair<'_>,
    ) -> SpiceResult<SensitivityLoad> {
        if reset.transient.is_some() {
            return Err(SpiceError::not_yet_ported(
                format!(
                    "{}: BSIM4 .sens ac after .tran (the final transient load's capbd/capbs)",
                    self.device.name
                ),
                "src/spicelib/analysis/cktsens.c and src/spicelib/devices/bsim4/b4acld.c",
            ));
        }
        let (rm, rh, rsize) = Self::structs(reset.state)?;
        let (m, h, size) = Self::structs(current)?;
        Ok(SensitivityLoad::Replacement(Box::new(ResetBsim4 {
            device: self.device.stand_in(m, h, size, &self.context),
            original: self.device.stand_in(
                self.model.clone(),
                self.instance.clone(),
                self.size.clone(),
                &self.context,
            ),
            reset: (rm, rh, rsize),
            small_signal: reset.small_signal.cloned(),
            omega: reset.omega,
            temperature: self.kelvin(self.context.temperature)?,
            gmin: self.context.gmin,
        })))
    }
}

/// The `pParam` `b4temp.c` leaves when one of its effective-geometry checks
/// fails (lines after `if (Size_Not_Found)`): a zeroed allocation holding the
/// members assigned before the first failing check, in C's arithmetic.
fn rejected_size(m: &Model, h: &Instance) -> SizeParam {
    let mut p = SizeParam {
        Length: h.l,
        Width: h.w,
        NFinger: h.nf,
        ..SizeParam::default()
    };
    let Lnew = h.l + m.xl;
    let Wnew = h.w / h.nf + m.xw;
    let T0 = Lnew.powf(m.Lln);
    let T1 = Wnew.powf(m.Lwn);
    let tmp1 = m.Ll / T0 + m.Lw / T1 + m.Lwl / (T0 * T1);
    p.dl = m.Lint + tmp1;
    let tmp2 = m.Llc / T0 + m.Lwc / T1 + m.Lwlc / (T0 * T1);
    p.dlc = m.dlc + tmp2;
    let T2 = Lnew.powf(m.Wln);
    let T3 = Wnew.powf(m.Wwn);
    let tmp1 = m.Wl / T2 + m.Ww / T3 + m.Wwl / (T2 * T3);
    p.dw = m.Wint + tmp1;
    let tmp2 = m.Wlc / T2 + m.Wwc / T3 + m.Wwlc / (T2 * T3);
    p.dwc = m.dwc + tmp2;
    p.dwj = m.dwj + tmp2;
    p.leff = Lnew - 2.0 * p.dl;
    if p.leff <= 0.0 {
        return p;
    }
    p.weff = Wnew - 2.0 * p.dw;
    if p.weff <= 0.0 {
        return p;
    }
    p.leffCV = Lnew - 2.0 * p.dlc;
    if p.leffCV <= 0.0 {
        return p;
    }
    p.weffCV = Wnew - 2.0 * p.dwc;
    if p.weffCV <= 0.0 {
        return p;
    }
    p.weffCJ = Wnew - 2.0 * p.dwj;
    p
}

/// A `.sens` AC stand-in: `cktsens.c`'s reset load on the reset structs,
/// then `b4acld.c` on C's one struct as the later setup and temperature
/// calls left it.
struct ResetBsim4 {
    device: Bsim4,
    /// The deck's own structs, for an earlier small-signal load.
    original: Bsim4,
    reset: (Model, Instance, SizeParam),
    /// The operating point of an earlier `MODEINITSMSIG` load, if any.
    small_signal: Option<crate::maths::Vector>,
    omega: Real,
    temperature: Real,
    gmin: Real,
}

impl std::fmt::Debug for ResetBsim4 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ResetBsim4")
            .field("device", &self.device.name)
            .finish_non_exhaustive()
    }
}

impl Device for ResetBsim4 {
    fn name(&self) -> &str {
        self.device.name()
    }

    fn designator(&self) -> char {
        'm'
    }

    fn terminals(&self) -> &[NodeId] {
        self.device.terminals()
    }

    fn stamp(&self, _context: &mut StampContext<'_>) -> SpiceResult<()> {
        Err(SpiceError::circuit(format!(
            "{}: an AC sensitivity stand-in has no DC load",
            self.device.name()
        )))
    }

    fn assemble_small_signal(
        &self,
        context: &mut LinearContext<'_>,
        _bias: &crate::maths::Vector,
    ) -> SpiceResult<()> {
        let model_context = *context.model_context;
        let (rm, mut rh, rsize) = (
            self.reset.0.clone(),
            self.reset.1.clone(),
            self.reset.2.clone(),
        );
        if let Some(bias) = &self.small_signal {
            // capbd/capbs as the earlier analysis's MODEINITSMSIG load left
            // them, at the deck's operating point on the deck's structs.
            let unknowns = context.unknowns;
            let voltage = |id: NodeId| {
                unknowns
                    .node_row(id)
                    .and_then(|row| bias.get(row))
                    .unwrap_or(0.)
            };
            let prepared = self
                .original
                .ensure(model_context.temperature, model_context.nominal_temperature)?;
            let s0 = self.original.state_at(&prepared, &voltage, self.gmin)?;
            let (h, _) = self
                .original
                .small_signal_point(&prepared, s0, &voltage, self.gmin)?;
            rh.capbd = h.capbd;
            rh.capbs = h.capbs;
        }
        let s0 = reset_load(
            &self.device.nodes,
            &rm,
            &mut rh,
            &rsize,
            self.temperature,
            self.gmin,
        )?;
        let prepared = self
            .device
            .ensure(model_context.temperature, model_context.nominal_temperature)?;
        let prepared: &Rc<_> = &prepared;
        // Generated ownership projection is precisely the load-only member
        // set (LOAD_WRITES - SETUP/TEMPERATURE/CHECK_WRITES).
        let h = prepared.instance.with_previous_load_state(&rh);
        ac_stamp(
            context,
            &self.device.nodes,
            self.omega,
            s0,
            &prepared.model,
            &h,
            &prepared.size,
        )
    }
}
