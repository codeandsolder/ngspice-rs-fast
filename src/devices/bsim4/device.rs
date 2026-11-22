//! The BSIM4 device: `b4set.c`/`b4temp.c` setup and the `b4ld.c`,
//! `b4acld.c`, `b4pzld.c` and `b4cvtest.c` routines wired into the solver.
//!
//! The ported C routines see the circuit through [`super::host`]; this file
//! implements those traits over [`StampContext`] and [`LinearContext`] and
//! maps the port's analysis phases onto C's `CKTmode` bits:
//!
//! | Port | C `CKTmode` |
//! | --- | --- |
//! | `uic` initial load | `MODETRANOP \| MODEUIC \| MODEINITJCT` |
//! | `.op` phases Junction/Fix/Float | `MODEDCOP \| MODEINITJCT/FIX/FLOAT` |
//! | `.dc` sweep phases (Predict = warm start) | `MODEDCTRANCURVE \| ...` (`MODEINITPRED`) |
//! | `.tran` first point / later points | `MODETRAN \| MODEINITTRAN` / `MODEINITPRED`, then `MODEINITFLOAT` |
//! | small-signal and asks | `MODEDCOP \| MODEINITSMSIG` |
//!
//! C keeps one `BSIM4instance` whose fields each load overwrites and the
//! convergence test, the AC load and the asks read back. The device keeps
//! the same struct in a cell (the "live" instance): every load updates it,
//! as C's does. Small-signal assemblies and asks run C's `MODEINITSMSIG`
//! load on a copy, so they never disturb a running solve.
use std::cell::RefCell;
use std::rc::Rc;

use super::fields::{Instance, Model, SizeParam};
use super::host::{LoadCkt, SmallSignalCkt, mode, node, state};
use super::{Inputs, ModelParameters, acld, apply, cvtest, elements, load, noise, pzld, set, temp};
use crate::devices::linear::LinearContext;
use crate::devices::models::{ModelContext, ResolvedModel};
use crate::devices::schema::temperature_kelvin;
use crate::devices::state::IterationPhase;
use crate::devices::traits::{AnalysisMode, ConvergenceContext, Device, StampContext};
use crate::netlist::ast::DeviceInstance;
use crate::primitives::{NodeId, NodeKind, NodeTable, Real, SourceLoc, SpiceError, SpiceResult};

/// One temperature's setup: the model after `b4set.c` and `b4temp.c`, this
/// instance after both, and its size-dependent parameters.
pub(super) struct Prepared {
    pub(super) temperature: Real,
    pub(super) nominal: Real,
    /// Ordered instance setters applied after initial setup.
    swept: Rc<[SweptValue]>,
    pub(super) model: Model,
    pub(super) instance: Instance,
    pub(super) size: SizeParam,
    /// `b4set.c`/`b4temp.c`/`b4check.c` messages, in C order. Kept for
    /// inspection: the port has no channel that reports device warnings
    /// yet (C prints them).
    #[expect(
        dead_code,
        reason = "retains BSIM4 model diagnostics with their setup for a future public warning channel"
    )]
    pub(super) warnings: Vec<String>,
}

/// `BSIM4setup`'s result: the model, the instance and the messages so far.
struct SetUp {
    model: Model,
    instance: Instance,
    warnings: Vec<String>,
    /// Internal nodes C creates, `(local index, suffix)` in order.
    created: Vec<(i32, &'static str)>,
}

/// One BSIM4 instance.
pub(in crate::devices) struct Bsim4 {
    pub(super) name: String,
    pub(super) model_name: String,
    /// External drain, gate, source, bulk, then the internal nodes in
    /// `b4set.c`'s creation order.
    pub(super) terminals: Vec<NodeId>,
    /// Local node index ([`node`]) to circuit node. Uncreated internal
    /// indices are never looked up: the instance's node fields alias them
    /// to external indices, as in C.
    pub(super) nodes: [NodeId; node::COUNT],
    has_start_settings: bool,
    pub(super) location: SourceLoc,
    /// C's `noiseAnalGiven` at setup.
    pub(super) noise_analysis: bool,
    /// `.dc @inst[param]` values (C's `DCTsetInstParam`): `BSIM4param`
    /// stores them after `BSIM4setup`, and `BSIM4temp` follows. Each is the
    /// setter's C field and given bit with its value.
    swept: Rc<[SweptValue]>,
    /// C's structs, shared with this instance's `.dc @inst[param]`
    /// stand-ins (which C's `DCTsetInstParam` writes into the same struct).
    shared: Rc<RefCell<Shared>>,
}

/// A real setter and its given bit, with an exact floating-point cache key.
#[derive(Clone, Copy, PartialEq, Eq)]
struct SweptValue {
    field: &'static str,
    given: Option<&'static str>,
    bits: u64,
}

impl SweptValue {
    fn apply(self, instance: &mut Instance) -> SpiceResult<()> {
        let slot = instance.real_mut(self.field).ok_or_else(|| {
            SpiceError::circuit(format!("BSIM4 swept field {} has no member", self.field))
        })?;
        *slot = Real::from_bits(self.bits);
        if let Some(flag) = self.given.and_then(|given| instance.flag_mut(given)) {
            *flag = true;
        }
        Ok(())
    }
}

/// The C structs of one BSIM4 instance.
#[derive(Default)]
struct Shared {
    /// Recent setups by temperature, nominal temperature and swept values.
    setups: Vec<Rc<Prepared>>,
    /// C's instance struct as the last load left it, and its setup.
    live: Option<(Rc<Prepared>, Instance)>,
}

impl Shared {
    /// A fresh temperature snapshot carries the previous load-only members.
    fn live_copy(&self, prepared: &Rc<Prepared>) -> Instance {
        match &self.live {
            Some((owner, h)) if Rc::ptr_eq(owner, prepared) => h.clone(),
            Some((_, h)) => prepared.instance.with_previous_load_state(h),
            None => prepared.instance.clone(),
        }
    }

    /// Install a live instance only when the prepared snapshot changes.
    fn live_instance(&mut self, prepared: &Rc<Prepared>) -> SpiceResult<&mut Instance> {
        if !self
            .live
            .as_ref()
            .is_some_and(|(owner, _)| Rc::ptr_eq(owner, prepared))
        {
            self.live = Some((Rc::clone(prepared), self.live_copy(prepared)));
        }
        self.live
            .as_mut()
            .map(|(_, instance)| instance)
            .ok_or_else(|| SpiceError::circuit("BSIM4 live instance missing"))
    }
}

/// How many setups [`Shared`] keeps.
const SETUPS_KEPT: usize = 8;

impl std::fmt::Debug for Bsim4 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Bsim4")
            .field("name", &self.name)
            .field("model", &self.model_name)
            .finish_non_exhaustive()
    }
}

impl Bsim4 {
    /// Build an instance, running C's setup and temperature passes before
    /// interning any node, so a fatal `b4check.c` result creates nothing.
    ///
    /// # Errors
    /// Invalid input, setup or temperature failures, node name collisions.
    pub(crate) fn instantiate(
        instance: &DeviceInstance,
        nodes: &mut NodeTable,
        model: &ResolvedModel<'_>,
        context: &ModelContext,
    ) -> SpiceResult<Box<dyn Device>> {
        if instance.nodes.len() != 4 {
            return Err(SpiceError::parse(
                instance.location.clone(),
                "BSIM4 needs four terminals: drain, gate, source, bulk",
            ));
        }
        let params = super::model_parameters(model)?;
        let inputs = params.instance_parameters(instance, 1.)?;
        let has_start_settings = inputs
            .applied()
            .iter()
            .any(|a| matches!(a.parameter.canonical, "off" | "icvds" | "icvgs" | "icvbs"));
        let mut setup = Self::setup(&params, &inputs, &instance.location, context)?;
        let created = std::mem::take(&mut setup.created);
        let mut device = Self {
            name: instance.name.clone(),
            model_name: model.card().name.clone(),
            terminals: Vec::new(),
            nodes: [NodeId::GROUND; node::COUNT],
            has_start_settings,
            location: instance.location.clone(),
            noise_analysis: context.noise_analysis,
            swept: Rc::default(),
            shared: Rc::new(RefCell::new(Shared::default())),
        };
        let prepared = device.prepare(
            context.temperature,
            context.nominal_temperature,
            Some(setup),
        )?;
        let mut staged = nodes.clone();
        let external: Vec<NodeId> = instance
            .nodes
            .iter()
            .map(|n| staged.intern(n))
            .collect::<SpiceResult<Vec<_>>>()?;
        device.terminals.clone_from(&external);
        for (local, id) in [node::D, node::G_EXT, node::S, node::B]
            .into_iter()
            .zip(external)
        {
            let index = usize::try_from(local)
                .ok()
                .filter(|&i| i < node::COUNT)
                .ok_or_else(|| {
                    SpiceError::circuit("BSIM4 setup mapped an invalid external node")
                })?;
            device.nodes[index] = id;
        }
        for &(local, suffix) in &created {
            let name = format!("{}#{suffix}", instance.name);
            if staged.get(&name).is_some() {
                return Err(SpiceError::circuit(format!(
                    "{}: internal node name collision: {name}",
                    instance.name
                )));
            }
            let id = staged.intern(&name)?;
            staged.set_kind(id, NodeKind::Internal);
            let index = usize::try_from(local)
                .ok()
                .filter(|&i| i < node::COUNT)
                .ok_or_else(|| SpiceError::circuit("BSIM4 setup created an invalid node"))?;
            device.nodes[index] = id;
            device.terminals.push(id);
        }
        *nodes = staged;
        device.shared.borrow_mut().setups.push(Rc::new(prepared));
        Ok(Box::new(device))
    }

    /// `b4mpar.c`/`b4par.c` setters and `BSIM4setup` at this nominal
    /// temperature (C's `CKTnomTemp`, the TNOM default).
    fn setup(
        params: &ModelParameters,
        inputs: &Inputs,
        location: &SourceLoc,
        context: &ModelContext,
    ) -> SpiceResult<SetUp> {
        let mut warnings = Vec::new();
        let mut m = apply::model(params)?;
        set::setup_model(
            &mut m,
            temperature_kelvin(context.nominal_temperature, location)?,
            &mut warnings,
        )?;
        let mut h = apply::instance(inputs)?;
        h.dNode = node::D;
        h.gNodeExt = node::G_EXT;
        h.sNode = node::S;
        h.bNode = node::B;
        let created = set::setup_instance(&m, &mut h, context.noise_analysis, &mut warnings)?;
        Ok(SetUp {
            model: m,
            instance: h,
            warnings,
            created,
        })
    }

    /// `BSIM4temp` at these temperatures (Celsius) on C's structs.
    ///
    /// The first call runs `BSIM4setup` and `BSIM4temp` with
    /// `BSIM4checkModel`. C keeps one model and one instance struct and runs
    /// `BSIM4setup` once, while each later `BSIM4temp` (a new analysis
    /// temperature, `.dc temp`, a `.dc @inst[param]` point) works on those
    /// structs as the previous pass left them, and skips the check: its
    /// `static` model name is already set. So later calls start from the
    /// first call's model and instance, apply the swept values, and run the
    /// temperature pass unchecked. (The first call's nominal temperature is
    /// the one setup used; C does not re-run setup either.)
    fn prepare(
        &self,
        temperature: Real,
        nominal: Real,
        initial: Option<SetUp>,
    ) -> SpiceResult<Prepared> {
        let (mut model, mut instance, mut warnings, check) = if let Some(setup) = initial {
            (setup.model, setup.instance, setup.warnings, true)
        } else {
            let shared = self.borrow_shared()?;
            let first = shared.setups.first().ok_or_else(|| {
                SpiceError::circuit(format!("{}: missing initial BSIM4 setup", self.name))
            })?;
            (
                first.model.clone(),
                first.instance.clone(),
                Vec::new(),
                false,
            )
        };
        for &setter in self.swept.iter() {
            setter.apply(&mut instance)?;
        }
        let kelvin = temperature_kelvin(temperature, &self.location)?;
        let locals = temp::temperature_model(&mut model, kelvin, &mut warnings);
        let size = temp::temperature_instance(
            &mut model,
            &mut instance,
            &locals,
            check,
            &self.model_name,
            &self.name,
            &mut warnings,
        )
        .map_err(|e| match e {
            SpiceError::NotYetPorted { .. } => e,
            other => SpiceError::circuit(format!("{}: {other}", self.name)),
        })?;
        Ok(Prepared {
            temperature,
            nominal,
            swept: Rc::clone(&self.swept),
            model,
            instance,
            size,
            warnings,
        })
    }

    /// The setup for these temperatures (and this device's swept values),
    /// rebuilt when C would re-run `BSIM4temp`.
    pub(super) fn ensure(&self, temperature: Real, nominal: Real) -> SpiceResult<Rc<Prepared>> {
        let same =
            |p: &Prepared| p.matches_temperatures(temperature, nominal) && p.swept == self.swept;
        {
            let shared = self.borrow_shared()?;
            if let Some(prepared) = shared.setups.iter().find(|p| same(p)) {
                return Ok(Rc::clone(prepared));
            }
        }
        let prepared = Rc::new(self.prepare(temperature, nominal, None)?);
        let mut shared = self.borrow_shared_mut()?;
        // The first setup is C's struct as `BSIM4setup` and the checked
        // `BSIM4temp` left it; later passes start from it, so keep it.
        if shared.setups.len() >= SETUPS_KEPT {
            shared.setups.remove(1);
        }
        shared.setups.push(Rc::clone(&prepared));
        Ok(prepared)
    }

    fn borrow_shared(&self) -> SpiceResult<std::cell::Ref<'_, Shared>> {
        self.shared
            .try_borrow()
            .map_err(|_| SpiceError::circuit(format!("{}: reentrant BSIM4 access", self.name)))
    }

    fn borrow_shared_mut(&self) -> SpiceResult<std::cell::RefMut<'_, Shared>> {
        self.shared
            .try_borrow_mut()
            .map_err(|_| SpiceError::circuit(format!("{}: reentrant BSIM4 access", self.name)))
    }

    /// Any setup, for what does not depend on temperature (nodes, matrix
    /// elements, truncation slots).
    fn any_setup(&self) -> Option<Rc<Prepared>> {
        self.shared.try_borrow().ok()?.setups.last().cloned()
    }

    /// The circuit node of a local node index; `None` for ground.
    pub(super) fn node_id(&self, local: i32) -> Option<NodeId> {
        if local == node::GROUND {
            return None;
        }
        usize::try_from(local)
            .ok()
            .and_then(|i| self.nodes.get(i))
            .copied()
    }

    /// `h`'s node fields after `CKTpzSetup` has re-run `BSIM4setup`
    /// `setups` times. `b4set.c` creates an internal node only while its
    /// field is still zero; otherwise its `else` branch aliases it, so a
    /// re-run aliases every drain/source/gate/mid-gate prime it created back
    /// to its external node and resets `qNode` to ground, and a further
    /// re-run creates a node again only where that alias is ground (node 0).
    /// Such a "new" node reuses the internal node's own index: C's original
    /// node no longer has elements, and SPARSE numbers rows by first
    /// appearance. The body nodes have no `else` branch and are kept.
    fn pole_zero_instance(&self, h: &Instance, setups: u32) -> Instance {
        let grounded =
            |local: i32| local == node::GROUND || self.node_id(local) == Some(NodeId::GROUND);
        let rerun = |own: i32, current: i32, alias: i32| {
            let created = current == own;
            let mut value = current;
            for _ in 0..setups {
                value = if created && grounded(value) {
                    own
                } else {
                    alias
                };
            }
            value
        };
        let mut p = h.clone();
        p.dNodePrime = rerun(node::DP, h.dNodePrime, h.dNode);
        p.sNodePrime = rerun(node::SP, h.sNodePrime, h.sNode);
        p.gNodePrime = rerun(node::GP, h.gNodePrime, h.gNodeExt);
        p.gNodeMid = rerun(node::GM, h.gNodeMid, h.gNodeExt);
        p.qNode = rerun(node::Q, h.qNode, node::GROUND);
        p
    }

    /// A probe copies the live instance without changing the running solve.
    pub(super) fn live_copy(&self, prepared: &Rc<Prepared>) -> Instance {
        self.shared.try_borrow().map_or_else(
            |_| prepared.instance.clone(),
            |shared| shared.live_copy(prepared),
        )
    }

    /// C's `MODEINITSMSIG` load at `state0` (the operating point's state
    /// vector), leaving the small-signal values on the returned instance.
    pub(super) fn small_signal_point(
        &self,
        prepared: &Rc<Prepared>,
        state0: [Real; state::COUNT],
        voltage: &dyn Fn(NodeId) -> Real,
        gmin: Real,
    ) -> SpiceResult<(Instance, [Real; state::COUNT])> {
        let mut h = self.live_copy(prepared);
        let mut probe = Probe {
            nodes: &self.nodes,
            voltage,
            mode: mode::MODEDCOP | mode::MODEINITSMSIG,
            limiting: false,
            gmin,
            temperature: prepared.model_temperature(),
            s0: state0,
        };
        load::load(&prepared.model, &mut h, &prepared.size, &mut probe)?;
        Ok((h, probe.s0))
    }

    /// The `state0` C's last operating-point load leaves, reproduced at a
    /// solution: an unlimited `MODEINITFLOAT` load there.
    pub(super) fn state_at(
        &self,
        prepared: &Rc<Prepared>,
        voltage: &dyn Fn(NodeId) -> Real,
        gmin: Real,
    ) -> SpiceResult<[Real; state::COUNT]> {
        let mut h = self.live_copy(prepared);
        let mut probe = Probe {
            nodes: &self.nodes,
            voltage,
            mode: mode::MODEDCOP | mode::MODEINITFLOAT,
            limiting: false,
            gmin,
            temperature: prepared.model_temperature(),
            s0: [0.; state::COUNT],
        };
        load::load(&prepared.model, &mut h, &prepared.size, &mut probe)?;
        Ok(probe.s0)
    }
}

/// `cktsens.c`'s per-frequency reset load: `CKTsetup` zeroes the state
/// vectors and the solution, so `BSIM4load` runs (`MODEDCOP |
/// MODEINITFLOAT`, as `CKTop` left the mode) at all-zero node voltages and
/// writes its instance members and `CKTstate0` from there. Returns that
/// `CKTstate0`. Limiting is a no-op there (every voltage and its previous
/// value are zero).
pub(super) fn reset_load(
    nodes: &[NodeId; node::COUNT],
    model: &Model,
    h: &mut Instance,
    size: &SizeParam,
    temperature: Real,
    gmin: Real,
) -> SpiceResult<[Real; state::COUNT]> {
    let zero = |_: NodeId| 0.;
    let mut probe = Probe {
        nodes,
        voltage: &zero,
        mode: mode::MODEDCOP | mode::MODEINITFLOAT,
        limiting: false,
        gmin,
        temperature,
        s0: [0.; state::COUNT],
    };
    load::load(model, h, size, &mut probe)?;
    Ok(probe.s0)
}

/// `b4acld.c` on explicit structs at `omega`, reading `s0` as `CKTstate0`.
///
/// # Errors
/// As the AC load, or a matrix position out of range.
pub(super) fn ac_stamp(
    context: &mut LinearContext<'_>,
    nodes: &[NodeId; node::COUNT],
    omega: Real,
    s0: [Real; state::COUNT],
    model: &Model,
    h: &Instance,
    size: &SizeParam,
) -> SpiceResult<()> {
    let mut host = SmallSignalHost {
        context,
        nodes,
        omega,
        s0,
        error: None,
    };
    acld::ac_load(model, h, size, &mut host);
    host.error.map_or(Ok(()), Err)
}

impl Bsim4 {
    /// A copy of this device whose only setup is `model`/`instance`/`size`
    /// at `context`'s temperatures, with no load history: a `.sens`
    /// stand-in that loads exactly those structs (`cktsens.c` loads C's
    /// structs as its setter, setup and temperature calls left them).
    pub(super) fn stand_in(
        &self,
        model: Model,
        instance: Instance,
        size: SizeParam,
        context: &ModelContext,
    ) -> Self {
        let prepared = Prepared {
            temperature: context.temperature,
            nominal: context.nominal_temperature,
            swept: Rc::default(),
            model,
            instance,
            size,
            warnings: Vec::new(),
        };
        Self {
            name: self.name.clone(),
            model_name: self.model_name.clone(),
            terminals: self.terminals.clone(),
            nodes: self.nodes,
            has_start_settings: self.has_start_settings,
            location: self.location.clone(),
            noise_analysis: self.noise_analysis,
            swept: Rc::default(),
            shared: Rc::new(RefCell::new(Shared {
                setups: vec![Rc::new(prepared)],
                live: None,
            })),
        }
    }
}

impl Prepared {
    /// The exact cache key for C temperature passes. Both signed zeros compare equal.
    #[expect(
        clippy::float_cmp,
        reason = "reuse a prepared model only for exactly equal temperatures; retain IEEE signed-zero and NaN equality semantics"
    )]
    const fn matches_temperatures(&self, temperature: Real, nominal: Real) -> bool {
        self.temperature == temperature && self.nominal == nominal
    }

    /// `CKTtemp` in kelvin for the loads.
    fn model_temperature(&self) -> Real {
        self.temperature + 273.15
    }
}

/// C's `CKTmode` for a load, and whether limiting is allowed.
fn load_mode(context: &StampContext<'_>) -> SpiceResult<(u32, bool)> {
    let states = &context.states;
    if states.initial_conditions() {
        return Ok((mode::MODETRANOP | mode::MODEUIC | mode::MODEINITJCT, true));
    }
    let uic = if states.uic() { mode::MODEUIC } else { 0 };
    let base = match context.mode {
        AnalysisMode::OperatingPoint => mode::MODEDCOP,
        AnalysisMode::DcSweep => mode::MODEDCTRANCURVE,
        AnalysisMode::Transient { .. } => mode::MODETRAN,
        AnalysisMode::Ac { .. } => {
            return Err(SpiceError::circuit(
                "BSIM4 AC is assembled by the small-signal load, not stamped",
            ));
        }
    };
    if !states.device_limiting() {
        return Ok((uic | base | mode::MODEINITFLOAT, false));
    }
    let init = match states.phase() {
        IterationPhase::Junction => mode::MODEINITJCT,
        IterationPhase::Fix => mode::MODEINITFIX,
        IterationPhase::Float => mode::MODEINITFLOAT,
        IterationPhase::Predict if base == mode::MODETRAN && states.initial_transient() => {
            mode::MODEINITTRAN
        }
        IterationPhase::Predict => mode::MODEINITPRED,
    };
    Ok((uic | base | init, true))
}

/// [`LoadCkt`] over a Newton/transient load.
struct StampHost<'h, 'c> {
    context: &'h mut StampContext<'c>,
    nodes: &'h [NodeId; node::COUNT],
    mode: u32,
    limiting: bool,
    temperature: Real,
    s0: [Real; state::COUNT],
    error: Option<SpiceError>,
}

impl StampHost<'_, '_> {
    fn id(&self, local: i32) -> Option<NodeId> {
        if local == node::GROUND {
            return None;
        }
        usize::try_from(local)
            .ok()
            .and_then(|i| self.nodes.get(i))
            .copied()
    }

    fn fail(&mut self, error: SpiceError) {
        if self.error.is_none() {
            self.error = Some(error);
        }
    }

    fn accepted(&mut self, age: usize, slot: usize) -> Real {
        if let Some(value) = self.context.states.accepted(age, slot) {
            return value;
        }
        self.fail(SpiceError::circuit(format!(
            "BSIM4 load read CKTstate{age} slot {slot} with no accepted point"
        )));
        0.
    }

    /// Commit `CKTstate0` to the trial.
    fn finish(self) -> SpiceResult<()> {
        if let Some(error) = self.error {
            return Err(error);
        }
        for (slot, value) in self.s0.into_iter().enumerate() {
            self.context.states.set(slot, value)?;
        }
        Ok(())
    }
}

impl LoadCkt for StampHost<'_, '_> {
    fn mode(&self) -> u32 {
        self.mode
    }
    fn limiting(&self) -> bool {
        self.limiting
    }
    fn gmin(&self) -> f64 {
        self.context.gmin
    }
    fn temp(&self) -> f64 {
        self.temperature
    }
    fn ag(&self, i: usize) -> f64 {
        self.context
            .integration
            .and_then(|c| c.ag().get(i).copied())
            .unwrap_or(0.)
    }
    fn delta(&self) -> f64 {
        // dctrcurv.c leaves CKTdelta equal to every CKTdeltaOld, so DEVpred's
        // ratio is one in a sweep.
        self.context
            .integration
            .map_or(1., crate::maths::Coefficients::dt)
    }
    fn delta_old(&self, i: usize) -> f64 {
        self.context
            .integration
            .map_or(Some(1.), |c| c.delta_old(i))
            .unwrap_or(0.)
    }
    fn rhs_old(&self, node: i32) -> f64 {
        self.id(node).map_or(0., |id| self.context.node_voltage(id))
    }
    fn state0(&self, slot: usize) -> f64 {
        self.s0.get(slot).copied().unwrap_or(0.)
    }
    fn set_state0(&mut self, slot: usize, value: f64) {
        if let Some(entry) = self.s0.get_mut(slot) {
            *entry = value;
        }
    }
    fn state1(&self, slot: usize) -> f64 {
        // `&self`: a missing accepted point is reported at `finish` through
        // the `state1` read recorded by the load-specific accessors below.
        self.context.states.accepted(1, slot).unwrap_or(f64::NAN)
    }
    fn set_state1(&mut self, slot: usize, value: f64) {
        if let Err(error) = self.context.states.patch_state1(slot, value) {
            self.fail(error);
        }
    }
    fn state2(&self, slot: usize) -> f64 {
        // dctran.c copies state1 into state2 after the first step.
        self.context
            .states
            .accepted(2, slot)
            .or_else(|| self.context.states.accepted(1, slot))
            .unwrap_or(f64::NAN)
    }
    fn integrate(&mut self, cap: f64, slot: usize) -> SpiceResult<(f64, f64)> {
        let Some(coefficients) = self.context.integration else {
            // Outside transient C still integrates (the NQS charges in every
            // load, all charges in the `uic` initial load) with the zero
            // `CKTag` of a fresh circuit (`dctran.c` clears it before the
            // first step).
            let q1 = self.context.states.accepted(1, slot).unwrap_or(0.);
            let (geq, ceq, derivative) = zero_coefficient_integrate(cap, self.state0(slot), q1);
            self.set_state0(slot + 1, derivative);
            return Ok((geq, ceq));
        };
        let q0 = self.state0(slot);
        let mut history = crate::maths::integrator::ChargeHistory::new(q0);
        for age in 1..=coefficients.charge_history_len() {
            let q = self.accepted(age, slot);
            history.push(q)?;
        }
        let previous = if coefficients.needs_previous_derivative() {
            Some(self.accepted(1, slot + 1))
        } else {
            None
        };
        if let Some(error) = self.error.take() {
            return Err(error);
        }
        let companion = coefficients.integrate(&history, previous, cap)?;
        self.set_state0(slot + 1, companion.derivative);
        Ok((companion.conductance, companion.current))
    }
    fn noncon(&mut self) {
        if let Err(error) = self.context.states.report_nonconvergence() {
            self.fail(error);
        }
    }
    fn matrix(&mut self, row: i32, col: i32, value: f64) {
        let (Some(r), Some(c)) = (self.id(row), self.id(col)) else {
            return;
        };
        if let Err(error) = self.context.stamp(r, c, value) {
            self.fail(error);
        }
    }
    fn rhs(&mut self, node: i32, value: f64) {
        let Some(n) = self.id(node) else {
            return;
        };
        if let Err(error) = self.context.stamp_rhs(n, value) {
            self.fail(error);
        }
    }
}

/// `NIintegrate` (`niinteg.c`, trapezoidal order 1, C's defaults) with
/// `CKTag[0] = CKTag[1] = 0`: returns `(geq, ceq, state0[ccap])`. C keeps
/// whatever `CKTag` the last transient left; a fresh circuit's is zero,
/// which is what every non-transient analysis of a deck without an earlier
/// `.tran` sees.
#[expect(
    clippy::suboptimal_flops,
    reason = "preserve NIintegrate's separate multiply, sum and subtraction, including signed-zero and nonfinite behavior"
)]
fn zero_coefficient_integrate(cap: f64, q0: f64, q1: f64) -> (f64, f64, f64) {
    let ag = [0.0_f64; 2];
    let derivative = ag[0] * q0 + ag[1] * q1;
    (ag[0] * cap, derivative - ag[0] * q0, derivative)
}

/// [`LoadCkt`] for C's evaluation-only loads (`MODEINITSMSIG`, and the
/// unlimited evaluation that reproduces an operating point's `state0`):
/// stamps are discarded, nothing is integrated.
struct Probe<'p> {
    nodes: &'p [NodeId; node::COUNT],
    voltage: &'p dyn Fn(NodeId) -> Real,
    mode: u32,
    limiting: bool,
    gmin: Real,
    temperature: Real,
    s0: [Real; state::COUNT],
}

impl LoadCkt for Probe<'_> {
    fn mode(&self) -> u32 {
        self.mode
    }
    fn limiting(&self) -> bool {
        self.limiting
    }
    fn gmin(&self) -> f64 {
        self.gmin
    }
    fn temp(&self) -> f64 {
        self.temperature
    }
    fn ag(&self, _i: usize) -> f64 {
        0.
    }
    fn delta(&self) -> f64 {
        1.
    }
    fn delta_old(&self, _i: usize) -> f64 {
        1.
    }
    fn rhs_old(&self, node: i32) -> f64 {
        if node == node::GROUND {
            return 0.;
        }
        usize::try_from(node)
            .ok()
            .and_then(|i| self.nodes.get(i))
            .map_or(0., |&id| (self.voltage)(id))
    }
    fn state0(&self, slot: usize) -> f64 {
        self.s0.get(slot).copied().unwrap_or(0.)
    }
    fn set_state0(&mut self, slot: usize, value: f64) {
        if let Some(entry) = self.s0.get_mut(slot) {
            *entry = value;
        }
    }
    fn state1(&self, slot: usize) -> f64 {
        self.state0(slot)
    }
    fn set_state1(&mut self, _slot: usize, _value: f64) {}
    fn state2(&self, slot: usize) -> f64 {
        self.state0(slot)
    }
    fn integrate(&mut self, cap: f64, slot: usize) -> SpiceResult<(f64, f64)> {
        // As `StampHost` outside transient: C's NQS loads integrate with the
        // zero `CKTag` of a DC or small-signal analysis.
        let q = self.state0(slot);
        let (geq, ceq, derivative) = zero_coefficient_integrate(cap, q, q);
        self.set_state0(slot + 1, derivative);
        Ok((geq, ceq))
    }
    fn noncon(&mut self) {}
    fn matrix(&mut self, _row: i32, _col: i32, _value: f64) {}
    fn rhs(&mut self, _node: i32, _value: f64) {}
}

/// [`SmallSignalCkt`] over a small-signal or pole-zero assembly.
struct SmallSignalHost<'h, 'c> {
    context: &'h mut LinearContext<'c>,
    nodes: &'h [NodeId; node::COUNT],
    omega: Real,
    s0: [Real; state::COUNT],
    error: Option<SpiceError>,
}

impl SmallSignalHost<'_, '_> {
    fn rows(&self, row: i32, col: i32) -> Option<(usize, usize)> {
        let id = |local: i32| {
            if local == node::GROUND {
                return None;
            }
            usize::try_from(local)
                .ok()
                .and_then(|i| self.nodes.get(i))
                .copied()
        };
        let (r, c) = (id(row)?, id(col)?);
        Some((
            self.context.unknowns.node_row(r)?,
            self.context.unknowns.node_row(c)?,
        ))
    }

    fn record(&mut self, result: SpiceResult<()>) {
        if let Err(error) = result
            && self.error.is_none()
        {
            self.error = Some(error);
        }
    }
}

impl SmallSignalCkt for SmallSignalHost<'_, '_> {
    fn omega(&self) -> f64 {
        self.omega
    }
    fn state0(&self, slot: usize) -> f64 {
        self.s0.get(slot).copied().unwrap_or(0.)
    }
    fn real(&mut self, row: i32, col: i32, value: f64) {
        if let Some((r, c)) = self.rows(row, col) {
            if let Some(statements) = self.context.pole_zero_statements.as_deref_mut() {
                statements.push(crate::maths::pencil::PencilTerm {
                    row: r,
                    col: c,
                    conductance: Some(value),
                    capacitance: None,
                });
            }
            let result = self.context.system.a.add(r, c, value);
            self.record(result);
        }
    }
    fn imag(&mut self, row: i32, col: i32, value: f64) {
        if let Some((r, c)) = self.rows(row, col) {
            let result = self.context.dynamic_prescaled(r, c, value);
            self.record(result);
        }
    }
    fn cap(&mut self, row: i32, col: i32, value: f64) {
        if let Some((r, c)) = self.rows(row, col) {
            if let Some(statements) = self.context.pole_zero_statements.as_deref_mut() {
                statements.push(crate::maths::pencil::PencilTerm {
                    row: r,
                    col: c,
                    conductance: None,
                    capacitance: Some(value),
                });
            }
            let result = self.context.dynamic_group(r, c, &[value], 1., 1.);
            self.record(result);
        }
    }
}

impl Bsim4 {
    /// The operating point's `state0` for a small-signal assembly. The
    /// analyses hand devices the accepted-history vector (C's `CKTstate1`,
    /// which `swacload.c` reads), not `CKTstate0`; C's `CKTstate0` after
    /// `CKTop` is what its last load wrote at the iterate the analysis
    /// linearises around (`NewtonSolution::values`), so an unlimited load
    /// there reproduces it.
    fn operating_state(
        &self,
        prepared: &Rc<Prepared>,
        voltage: &dyn Fn(NodeId) -> Real,
        gmin: Real,
    ) -> SpiceResult<[Real; state::COUNT]> {
        self.state_at(prepared, voltage, gmin)
    }

    /// `b4acld.c` or `b4pzld.c` after the `MODEINITSMSIG` load.
    fn assemble(
        &self,
        context: &mut LinearContext<'_>,
        bias: &crate::maths::Vector,
        pole_zero: bool,
    ) -> SpiceResult<()> {
        let model_context = *context.model_context;
        let prepared = self.ensure(model_context.temperature, model_context.nominal_temperature)?;
        let unknowns = context.unknowns;
        let voltage = |node: NodeId| {
            unknowns
                .node_row(node)
                .and_then(|row| bias.get(row))
                .unwrap_or(0.)
        };
        let s0 = self.operating_state(&prepared, &voltage, model_context.gmin)?;
        let (h, s0) = self.small_signal_point(&prepared, s0, &voltage, model_context.gmin)?;
        // pzan.c loads MODEINITSMSIG before CKTpzSetup re-runs BSIM4setup.
        let h = if pole_zero {
            self.pole_zero_instance(&h, context.pole_zero_setups.max(1))
        } else {
            h
        };
        let omega = 2.0 * std::f64::consts::PI * model_context.frequency;
        let mut host = SmallSignalHost {
            context,
            nodes: &self.nodes,
            omega,
            s0,
            error: None,
        };
        if pole_zero {
            pzld::pz_load(&prepared.model, &h, &prepared.size, &mut host);
        } else {
            acld::ac_load(&prepared.model, &h, &prepared.size, &mut host);
        }
        host.error.map_or(Ok(()), Err)
    }

    /// `b4ask.c` at an operating point, on the instance and state the
    /// `MODEINITSMSIG` load leaves.
    fn ask(
        keyword: &str,
        prepared: &Prepared,
        h: &Instance,
        s0: &[Real; state::COUNT],
    ) -> Option<Real> {
        let p = super::descriptor(super::parameters::INSTANCE, keyword)?;
        let mi = h.mult_i * h.m;
        let mq = h.mult_q * h.m;
        let size = &prepared.size;
        Some(match p.c_id {
            "BSIM4_SOURCECONDUCT" => h.sourceConductance * mi,
            "BSIM4_DRAINCONDUCT" => h.drainConductance * mi,
            "BSIM4_VBD" => s0[state::VBD],
            "BSIM4_VBS" => s0[state::VBS],
            "BSIM4_VGS" => s0[state::VGS],
            "BSIM4_VDS" => s0[state::VDS],
            "BSIM4_CD" => h.cd * mi,
            "BSIM4_CBS" => h.cbs * mi,
            "BSIM4_CBD" => h.cbd * mi,
            "BSIM4_CSUB" => h.csub * mi,
            "BSIM4_IGIDL" => h.Igidl * mi,
            "BSIM4_IGISL" => h.Igisl * mi,
            "BSIM4_IGS" => h.Igs * mi,
            "BSIM4_IGD" => h.Igd * mi,
            "BSIM4_IGB" => h.Igb * mi,
            "BSIM4_IGCS" => h.Igcs * mi,
            "BSIM4_IGCD" => h.Igcd * mi,
            "BSIM4_GM" => h.gm * mi,
            "BSIM4_GDS" => h.gds * mi,
            "BSIM4_GMBS" => h.gmbs * mi,
            "BSIM4_GBD" => h.gbd * mi,
            "BSIM4_GBS" => h.gbs * mi,
            "BSIM4_QB" => h.qbulk * mq,
            "BSIM4_QG" => h.qgate * mq,
            "BSIM4_QS" => h.qsrc * mq,
            "BSIM4_QD" => h.qdrn * mq,
            "BSIM4_QINV" => h.qinv,
            "BSIM4_QDEF" => s0[state::QDEF],
            "BSIM4_GCRG" => h.gcrg * mi,
            "BSIM4_GTAU" => h.gtau,
            "BSIM4_CGGB" => h.cggb * mq,
            "BSIM4_CGDB" => h.cgdb * mq,
            "BSIM4_CGSB" => h.cgsb * mq,
            "BSIM4_CDGB" => h.cdgb * mq,
            "BSIM4_CDDB" => h.cddb * mq,
            "BSIM4_CDSB" => h.cdsb * mq,
            "BSIM4_CBGB" => h.cbgb * mq,
            "BSIM4_CBDB" => h.cbdb * mq,
            "BSIM4_CBSB" => h.cbsb * mq,
            "BSIM4_CSGB" => h.csgb * mq,
            "BSIM4_CSDB" => h.csdb * mq,
            "BSIM4_CSSB" => h.cssb * mq,
            "BSIM4_CGBB" => h.cgbb * mq,
            "BSIM4_CDBB" => h.cdbb * mq,
            "BSIM4_CSBB" => h.csbb * mq,
            "BSIM4_CBBB" => h.cbbb * mq,
            "BSIM4_CAPBD" => h.capbd * mq,
            "BSIM4_CAPBS" => h.capbs * mq,
            "BSIM4_VON" => h.von,
            "BSIM4_VDSAT" => h.vdsat,
            "BSIM4_QBS" => s0[state::QBS],
            "BSIM4_QBD" => s0[state::QBD],
            "BSIM4_VGSTEFF" => h.Vgsteff,
            "BSIM4_VDSEFF" => h.Vdseff,
            "BSIM4_CGSO" => h.cgso * mq,
            "BSIM4_CGDO" => h.cgdo * mq,
            "BSIM4_CGBO" => size.cgbo * mq,
            "BSIM4_WEFF" => size.weff,
            "BSIM4_LEFF" => size.leff,
            _ => return parameter_ask(p.c_id, h),
        })
    }
}

/// `b4ask.c`'s bias-independent instance values.
fn parameter_ask(c_id: &str, h: &Instance) -> Option<Real> {
    Some(match c_id {
        "BSIM4_L" => h.l,
        "BSIM4_W" => h.w,
        "BSIM4_M" => h.m,
        "BSIM4_MULT_I" => h.mult_i,
        "BSIM4_MULT_Q" => h.mult_q,
        "BSIM4_MULT_FN" => h.mult_fn,
        "BSIM4_NF" => h.nf,
        "BSIM4_MIN" => Real::from(h.min),
        "BSIM4_AS" => h.sourceArea,
        "BSIM4_AD" => h.drainArea,
        "BSIM4_PS" => h.sourcePerimeter,
        "BSIM4_PD" => h.drainPerimeter,
        "BSIM4_NRS" => h.sourceSquares,
        "BSIM4_NRD" => h.drainSquares,
        "BSIM4_OFF" => Real::from(h.off),
        "BSIM4_SA" => h.sa,
        "BSIM4_SB" => h.sb,
        "BSIM4_SD" => h.sd,
        "BSIM4_SCA" => h.sca,
        "BSIM4_SCB" => h.scb,
        "BSIM4_SCC" => h.scc,
        "BSIM4_SC" => h.sc,
        "BSIM4_RBSB" => h.rbsb,
        "BSIM4_RBDB" => h.rbdb,
        "BSIM4_RBPB" => h.rbpb,
        "BSIM4_RBPS" => h.rbps,
        "BSIM4_RBPD" => h.rbpd,
        "BSIM4_DELVTO" => h.delvto,
        "BSIM4_MULU0" => h.mulu0,
        "BSIM4_WNFLAG" => Real::from(h.wnflag),
        "BSIM4_DTEMP" => h.dtemp,
        "BSIM4_XGW" => h.xgw,
        "BSIM4_NGCON" => h.ngcon,
        "BSIM4_TRNQSMOD" => Real::from(h.trnqsMod),
        "BSIM4_ACNQSMOD" => Real::from(h.acnqsMod),
        "BSIM4_RBODYMOD" => Real::from(h.rbodyMod),
        "BSIM4_RGATEMOD" => Real::from(h.rgateMod),
        "BSIM4_GEOMOD" => Real::from(h.geoMod),
        "BSIM4_RGEOMOD" => Real::from(h.rgeoMod),
        "BSIM4_IC_VDS" => h.icVDS,
        "BSIM4_IC_VGS" => h.icVGS,
        "BSIM4_IC_VBS" => h.icVBS,
        _ => return None,
    })
}

impl Device for Bsim4 {
    fn name(&self) -> &str {
        &self.name
    }

    fn model_name(&self) -> Option<&str> {
        Some(&self.model_name)
    }

    fn designator(&self) -> char {
        'm'
    }

    fn terminals(&self) -> &[NodeId] {
        &self.terminals
    }

    fn is_nonlinear(&self) -> bool {
        true
    }

    fn has_start_settings(&self) -> bool {
        self.has_start_settings
    }

    fn state_count(&self) -> usize {
        state::COUNT
    }

    fn c_device_type(&self) -> crate::devices::setup::CDeviceType {
        crate::devices::setup::CDeviceType::Bsim
    }

    /// `BSIM4trunc`: `CKTterr` on qb, qg, qd, then qcdump (`trnqsMod`), qbs
    /// and qbd (`rbodyMod`) and qgmid (`rgateMod == 3`).
    fn truncation_slots(&self) -> Vec<usize> {
        let Some(prepared) = self.any_setup() else {
            return vec![state::QB, state::QG, state::QD];
        };
        let h = &prepared.instance;
        let mut slots = vec![state::QB, state::QG, state::QD];
        if h.trnqsMod != 0 {
            slots.push(state::QCDUMP);
        }
        if h.rbodyMod != 0 {
            slots.extend([state::QBS, state::QBD]);
        }
        if h.rgateMod == 3 {
            slots.push(state::QGMID);
        }
        slots
    }

    fn setup_matrix(
        &self,
        setup: &mut crate::devices::setup::MatrixSetup<'_>,
    ) -> SpiceResult<bool> {
        let Some(prepared) = self.any_setup() else {
            return Ok(false);
        };
        let instance = if setup.pole_zero {
            self.pole_zero_instance(&prepared.instance, setup.pole_zero_setups.max(1))
        } else {
            prepared.instance.clone()
        };
        for (row, col) in elements::matrix_elements(&prepared.model, &instance) {
            if let (Some(r), Some(c)) = (self.node_id(row), self.node_id(col)) {
                setup.nodes(r, c);
            }
        }
        Ok(true)
    }

    fn stamp(&self, context: &mut StampContext<'_>) -> SpiceResult<()> {
        let prepared = self.ensure(context.temperature, context.nominal_temperature)?;
        let (mode, limiting) = load_mode(context)?;
        let mut shared = self.borrow_shared_mut()?;
        let h = shared.live_instance(&prepared)?;
        if mode & mode::MODEUIC != 0 && context.states.initial_conditions() {
            // b4getic.c: unset IC components are the solution's terminal
            // voltages (CKTic runs only under MODEUIC).
            let v = |local: i32| {
                self.node_id(local)
                    .map_or(0., |id| context.node_voltage(id))
            };
            if !h.icVDSGiven {
                h.icVDS = v(h.dNode) - v(h.sNode);
            }
            if !h.icVGSGiven {
                h.icVGS = v(h.gNodeExt) - v(h.sNode);
            }
            if !h.icVBSGiven {
                h.icVBS = v(h.bNode) - v(h.sNode);
            }
        }
        let mut s0 = [0.; state::COUNT];
        for (slot, value) in s0.iter_mut().enumerate() {
            if let Some(previous) = context
                .states
                .iterate(slot)
                .or_else(|| context.states.carried(slot))
                .filter(|v| v.is_finite())
            {
                *value = previous;
            }
        }
        let temperature = prepared.model_temperature();
        let mut host = StampHost {
            context,
            nodes: &self.nodes,
            mode,
            limiting,
            temperature,
            s0,
            error: None,
        };
        load::load(&prepared.model, h, &prepared.size, &mut host)?;
        host.finish()
    }

    /// `BSIM4convTest` on the live instance the last load left, reading
    /// `CKTrhsOld` (the iterate that load used) as C does. `NIiter` runs it
    /// in `MODEINITFIX` too, where an `off` instance is not tested.
    fn convergence_test(&self, context: &ConvergenceContext<'_>) -> SpiceResult<bool> {
        let shared = self.borrow_shared()?;
        let Some((prepared, h)) = shared.live.as_ref() else {
            return Ok(true);
        };
        let rhs_old = |local: i32| {
            self.node_id(local)
                .and_then(|id| context.unknowns.node_row(id))
                .and_then(|row| context.previous.get(row))
                .unwrap_or(0.)
        };
        let state0 = |slot: usize| context.states.get(slot).copied().unwrap_or(0.);
        Ok(cvtest::conv_test(
            &prepared.model,
            h,
            &rhs_old,
            &state0,
            match context.phase {
                IterationPhase::Fix => mode::MODEINITFIX,
                _ => mode::MODEINITFLOAT,
            },
            context.reltol,
            context.abstol,
        ))
    }

    /// Every real instance setter of `b4.c`'s table (`DCTfindInstParam`
    /// accepts any settable instance parameter).
    fn instance_parameter(&self, keyword: &str) -> Option<&'static str> {
        let p = super::descriptor(super::parameters::INSTANCE, keyword)?;
        (p.input_type == super::InputType::Real
            && p.access != super::Access::Output
            && p.c_field.is_some())
        .then_some(p.canonical)
    }

    /// `BSIM4param` then `BSIM4temp` (`dctrcurv.c` `DCTsetInstParam`): the
    /// value is stored after `BSIM4setup`, so setup-derived quantities
    /// (instance defaults, the internal nodes) keep their setup values, as
    /// in C. Later temperature processing starts from the initial checked
    /// snapshot without rerunning `BSIM4checkModel`.
    fn with_instance_parameter(
        &self,
        parameter: &str,
        value: Real,
        context: &ModelContext,
    ) -> SpiceResult<Box<dyn Device>> {
        let p = super::descriptor(super::parameters::INSTANCE, parameter)
            .filter(|p| self.instance_parameter(p.name).is_some())
            .ok_or_else(|| {
                SpiceError::circuit(format!(
                    "{}: instance parameter {parameter} cannot be swept",
                    self.name
                ))
            })?;
        crate::devices::sweep::check_swept(&self.name, parameter, value, true, "finite")?;
        let mut swept = self.swept.to_vec();
        swept.push(SweptValue {
            field: p.c_field.unwrap_or_default(),
            given: p.c_given,
            bits: value.to_bits(),
        });
        let device = Self {
            name: self.name.clone(),
            model_name: self.model_name.clone(),
            terminals: self.terminals.clone(),
            nodes: self.nodes,
            has_start_settings: self.has_start_settings,
            location: self.location.clone(),
            noise_analysis: self.noise_analysis,
            swept: swept.into(),
            shared: Rc::clone(&self.shared),
        };
        device.ensure(context.temperature, context.nominal_temperature)?;
        Ok(Box::new(device))
    }

    fn observation_parameter(
        &self,
        key: &str,
        context: &ModelContext,
    ) -> SpiceResult<Option<Real>> {
        let Some(p) = super::descriptor(super::parameters::INSTANCE, key) else {
            return Ok(None);
        };
        let prepared = self.ensure(context.temperature, context.nominal_temperature)?;
        let h = &prepared.instance;
        let mi = h.mult_i * h.m;
        Ok(match p.c_id {
            "BSIM4_SOURCECONDUCT" => Some(h.sourceConductance * mi),
            "BSIM4_DRAINCONDUCT" => Some(h.drainConductance * mi),
            "BSIM4_WEFF" => Some(prepared.size.weff),
            "BSIM4_LEFF" => Some(prepared.size.leff),
            id => parameter_ask(id, h),
        })
    }

    fn observation_operating(
        &self,
        keyword: &str,
        context: &ModelContext,
        voltage: &dyn Fn(NodeId) -> Real,
        _transient: bool,
    ) -> SpiceResult<Option<Real>> {
        let prepared = self.ensure(context.temperature, context.nominal_temperature)?;
        let s0 = self.state_at(&prepared, voltage, context.gmin)?;
        let (h, s0) = self.small_signal_point(&prepared, s0, voltage, context.gmin)?;
        Ok(Self::ask(keyword, &prepared, &h, &s0))
    }

    /// `b4acld.c` reassembled at every frequency: its `x* = cap * omega`
    /// terms (and the NQS terms) are computed with C's `CKTomega`.
    fn small_signal_depends_on_frequency(&self) -> bool {
        true
    }

    fn assemble_small_signal(
        &self,
        context: &mut LinearContext<'_>,
        bias: &crate::maths::Vector,
    ) -> SpiceResult<()> {
        self.assemble(context, bias, false)
    }

    fn assemble_pole_zero(
        &self,
        context: &mut LinearContext<'_>,
        bias: &crate::maths::Vector,
    ) -> SpiceResult<()> {
        self.assemble(context, bias, true)
    }

    /// `.sens` (`cktsens.c`): C's generic perturbation of every askable real
    /// parameter, replayed on C's structs (see [`super::sens`]).
    fn sensitivity(
        &self,
        context: &ModelContext,
    ) -> SpiceResult<Box<dyn crate::devices::sensitivity::DeviceSensitivity + '_>> {
        Ok(Box::new(super::sens::Bsim4Sensitivity::new(self, context)?))
    }

    /// C's BSIM4 has no `DEVdisto` (`bsim4init.c`): `.disto` sees only its
    /// AC load.
    fn distortion(
        &self,
        _context: &crate::devices::distortion::DistortionContext<'_>,
    ) -> SpiceResult<crate::devices::distortion::DeviceDistortion> {
        Ok(crate::devices::distortion::DeviceDistortion::Linear)
    }

    /// `b4noi.c` at the analysis frequency, on the instance C's
    /// `MODEINITSMSIG` load leaves: the 14 generators in C's order, then the
    /// instance total.
    fn noise(
        &self,
        context: &crate::devices::noise::NoiseContext<'_>,
    ) -> SpiceResult<crate::devices::noise::DeviceNoise> {
        use crate::devices::noise::{DeviceNoise, NoiseFamily, NoiseKind, NoiseSource};
        let model_context = *context.model_context;
        let prepared = self.ensure(model_context.temperature, model_context.nominal_temperature)?;
        let m = &prepared.model;
        if !(0..=2).contains(&m.tnoiMod) || !(0..=1).contains(&m.fnoiMod) {
            return Err(SpiceError::not_yet_ported(
                format!(
                    "{}: BSIM4 noise with tnoiMod={} fnoiMod={}",
                    self.name, m.tnoiMod, m.fnoiMod
                ),
                "src/spicelib/devices/bsim4/b4noi.c (no case for this selector)",
            ));
        }
        let voltage = |node: NodeId| context.voltage(node);
        let s0 = self.operating_state(&prepared, &voltage, model_context.gmin)?;
        let (h, s0) = self.small_signal_point(&prepared, s0, &voltage, model_context.gmin)?;
        let temperature = context.circuit_kelvin();
        let generators = noise::generators(
            m,
            &h,
            &prepared.size,
            &|slot| s0.get(slot).copied().unwrap_or(0.),
            model_context.frequency,
            temperature,
        );
        let id = |local: i32| self.node_id(local).unwrap_or(NodeId::GROUND);
        let [dp, sp] = [id(h.dNodePrime), id(h.sNodePrime)];
        let sources = generators
            .into_iter()
            .zip(noise::NAMES)
            .map(|(generator, suffix)| {
                let (nodes, kind) = match generator {
                    noise::Source::Thermal(a, b, conductance) => (
                        [id(a), id(b)],
                        NoiseKind::Thermal {
                            conductance,
                            temperature,
                        },
                    ),
                    noise::Source::Shot(a, b, current) => {
                        ([id(a), id(b)], NoiseKind::Shot { current })
                    }
                    noise::Source::Gain(a, b, factor) => {
                        ([id(a), id(b)], NoiseKind::Gain { factor })
                    }
                    noise::Source::Zero => ([dp, sp], NoiseKind::Gain { factor: 0. }),
                    noise::Source::Correlated {
                        first,
                        g1,
                        second,
                        g2,
                        phase,
                    } => (
                        [id(first.0), id(first.1)],
                        NoiseKind::CorrelatedThermal {
                            conductance: g1,
                            second: [id(second.0), id(second.1)],
                            second_conductance: g2,
                            phase,
                            temperature,
                        },
                    ),
                };
                NoiseSource::new(suffix, nodes, kind)
            })
            .collect();
        Ok(DeviceNoise::Sources {
            family: NoiseFamily::Bsim4,
            model: Some(self.model_name.clone()),
            total: true,
            sources,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn prepared(length: Real) -> Rc<Prepared> {
        Rc::new(Prepared {
            temperature: 27.,
            nominal: 27.,
            swept: Rc::default(),
            model: Model::default(),
            instance: Instance {
                l: length,
                cgdo: length * 2.,
                cgso: length * 3.,
                ..Instance::default()
            },
            size: SizeParam::default(),
            warnings: Vec::new(),
        })
    }

    #[test]
    fn sweep_key_preserves_given_bits_and_signed_zero() {
        let first = SweptValue {
            field: "BSIM4l",
            given: Some("BSIM4lGiven"),
            bits: 0.0_f64.to_bits(),
        };
        let negative = SweptValue {
            bits: (-0.0_f64).to_bits(),
            ..first
        };
        let without_given = SweptValue {
            given: None,
            ..first
        };
        assert!(first != negative);
        assert!(first != without_given);
        let mut instance = Instance::default();
        negative.apply(&mut instance).unwrap();
        assert_eq!(instance.l.to_bits(), negative.bits);
        assert!(instance.lGiven);
    }

    #[test]
    fn temperature_cache_uses_exact_numeric_equality() {
        let setup = prepared(1e-6);
        assert!(setup.matches_temperatures(27., 27.));
        assert!(!setup.matches_temperatures(27.0_f64.next_up(), 27.));
        assert!(!setup.matches_temperatures(27., 27.0_f64.next_up()));
        assert!(!setup.matches_temperatures(f64::NAN, 27.));
    }

    #[test]
    fn ownership_projection_covers_every_instance_member() {
        let mut prepared = Instance::default();
        let mut previous = Instance::default();
        previous.seed_test_members();
        prepared.cgdo = -0.;
        previous.gm = f64::from_bits(0x7ff8_0000_0000_0001);
        let projected = prepared.with_previous_load_state(&previous);
        let fresh_values = prepared.to_values();
        let previous_values = previous.to_values();
        let actual = projected.to_values();
        for (index, name) in Instance::MEMBERS.iter().enumerate() {
            let carries = Instance::LOAD_WRITES.contains(name)
                && !Instance::SETUP_WRITES.contains(name)
                && !Instance::TEMPERATURE_WRITES.contains(name)
                && !Instance::CHECK_WRITES.contains(name);
            let expected = if carries {
                previous_values[index]
            } else {
                fresh_values[index]
            };
            assert_eq!(actual[index].to_bits(), expected.to_bits(), "{name}");
        }
        assert_eq!(projected.cgdo.to_bits(), (-0.0_f64).to_bits());
        assert_eq!(projected.gm.to_bits(), previous.gm.to_bits());
        let overlap: Vec<_> = Instance::LOAD_WRITES
            .iter()
            .copied()
            .filter(|member| Instance::TEMPERATURE_WRITES.contains(member))
            .collect();
        assert_eq!(overlap.len(), 2);
        assert!(overlap.contains(&"BSIM4cgdo"));
        assert!(overlap.contains(&"BSIM4cgso"));
    }

    #[test]
    fn live_snapshot_changes_keep_load_members_not_geometry() {
        let first = prepared(1e-6);
        let second = prepared(2e-6);
        let mut shared = Shared::default();
        let live = shared.live_instance(&first).unwrap();
        live.gm = 0.123;
        live.qgate = 0.456;
        live.cgdo = 123.;
        live.cgso = 456.;
        assert_eq!(
            shared.live_instance(&first).unwrap().gm.to_bits(),
            0.123_f64.to_bits()
        );
        let mut probe = shared.live_copy(&second);
        assert_eq!(probe.gm.to_bits(), 0.123_f64.to_bits());
        assert_eq!(probe.qgate.to_bits(), 0.456_f64.to_bits());
        assert_eq!(probe.l.to_bits(), 2e-6_f64.to_bits());
        assert_eq!(probe.cgdo.to_bits(), second.instance.cgdo.to_bits());
        assert_eq!(probe.cgso.to_bits(), second.instance.cgso.to_bits());
        probe.gm = 0.456;
        assert_eq!(probe.gm.to_bits(), 0.456_f64.to_bits());
        assert_eq!(shared.live_copy(&first).gm.to_bits(), 0.123_f64.to_bits());
        let live = shared.live_instance(&second).unwrap();
        assert_eq!(live.l.to_bits(), 2e-6_f64.to_bits());
        assert_eq!(live.gm.to_bits(), 0.123_f64.to_bits());
        assert_eq!(second.instance.gm.to_bits(), 0.0_f64.to_bits());
        live.gm = 0.789;
        live.cgdo = 789.;
        let live = shared.live_instance(&first).unwrap();
        assert_eq!(live.gm.to_bits(), 0.789_f64.to_bits());
        assert_eq!(live.cgdo.to_bits(), first.instance.cgdo.to_bits());
        assert_eq!(live.cgso.to_bits(), first.instance.cgso.to_bits());
    }
}
