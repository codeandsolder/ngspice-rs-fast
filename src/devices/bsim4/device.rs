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
use std::sync::Arc;

use super::fields::{Instance, Model, SizeParam};
use super::host::{LoadCkt, SmallSignalCkt, mode, node, state};
use super::{Inputs, ModelParameters, acld, apply, cvtest, elements, load, noise, pzld, set, temp};
use crate::devices::linear::LinearContext;
use crate::devices::models::{ModelContext, PreparationKey, ResolvedModel};
use crate::devices::schema::temperature_kelvin;
use crate::devices::state::IterationPhase;
use crate::devices::traits::{AnalysisMode, ConvergenceContext, Device, StampContext};
use crate::netlist::ast::DeviceInstance;
use crate::primitives::{NodeAppend, NodeId, NodeKind, Real, SourceLoc, SpiceError, SpiceResult};

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
        nodes: &mut NodeAppend<'_>,
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
        let prototype = Self::model_setup(
            model,
            &params,
            &instance.location,
            context.nominal_temperature,
        )?;
        let mut setup = Self::setup(&prototype, &inputs, context.noise_analysis)?;
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
        nodes.transaction(|staged| {
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
                staged.set_kind(id, NodeKind::Internal)?;
                let index = usize::try_from(local)
                    .ok()
                    .filter(|&i| i < node::COUNT)
                    .ok_or_else(|| SpiceError::circuit("BSIM4 setup created an invalid node"))?;
                device.nodes[index] = id;
                device.terminals.push(id);
            }
            Ok(())
        })?;
        device.shared.borrow_mut().setups.push(Rc::new(prepared));
        Ok(Box::new(device))
    }

    /// Model-only setters and defaults. Every consumer clones the prototype
    /// before running the instance setup and temperature/check passes.
    fn model_setup(
        resolved: &ResolvedModel<'_>,
        params: &ModelParameters,
        location: &SourceLoc,
        nominal: Real,
    ) -> SpiceResult<Arc<set::PreparedModel>> {
        let raw = resolved.prepare(PreparationKey::Inputs, || apply::model(params))?;
        let kelvin = temperature_kelvin(nominal, location)?;
        resolved.prepare(PreparationKey::Nominal(nominal.to_bits()), || {
            Ok(set::prepare_model((*raw).clone(), kelvin))
        })
    }

    fn setup(
        prototype: &set::PreparedModel,
        inputs: &Inputs,
        noise_analysis: bool,
    ) -> SpiceResult<set::PreparedSetup> {
        let mut h = apply::instance(inputs)?;
        h.dNode = node::D;
        h.gNodeExt = node::G_EXT;
        h.sNode = node::S;
        h.bNode = node::B;
        Ok(prototype.clone().prepare_instance(h, noise_analysis))
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
        initial: Option<set::PreparedSetup>,
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
        let p = super::DEVICE.instance.get(keyword)?;
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
        self.stamp_with_model(context, None)
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
        let p = super::DEVICE.instance.get(keyword)?;
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
        let p = super::DEVICE
            .instance
            .get(parameter)
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
        let Some(p) = super::DEVICE.instance.get(key) else {
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

    #[test]
    fn model_setup_prototypes_share_only_exact_nominal_context() {
        use crate::devices::ModelResolver;
        use crate::netlist::{Parser, source::parse_deck_text};
        let deck = Parser::new()
            .parse_deck(&parse_deck_text(
                std::path::Path::new("bsim4-cache.cir"),
                "BSIM4 cache\nm1 d g s 0 model l=1u w=10u\n.model model nmos level=54 mobmod=99\n.end\n",
            ))
            .unwrap();
        let resolver = ModelResolver::new(&deck.models).unwrap();
        let inst = &deck.devices[0];
        let resolved = resolver.resolve(inst).unwrap().unwrap();
        let params = super::super::model_parameters(&resolved).unwrap();
        let a = Bsim4::model_setup(&resolved, &params, &inst.location, 27.).unwrap();
        let b = Bsim4::model_setup(&resolved, &params, &inst.location, 42.).unwrap();
        let again = Bsim4::model_setup(&resolved, &params, &inst.location, 27.).unwrap();
        assert!(Arc::ptr_eq(&a, &again));
        assert!(!Arc::ptr_eq(&a, &b));
        let inputs = params.instance_parameters(inst, 1.).unwrap();
        let mut first = Bsim4::setup(&a, &inputs, false).unwrap();
        let different = Bsim4::setup(&b, &inputs, false).unwrap();
        let second = Bsim4::setup(&again, &inputs, false).unwrap();
        assert_ne!(first.model.tnom.to_bits(), different.model.tnom.to_bits());
        assert!(!first.warnings.is_empty());
        assert_eq!(first.warnings, different.warnings);
        let nominal = first.model.tnom.to_bits();
        first.model.tnom = 0.;
        assert_eq!(second.model.tnom.to_bits(), nominal);
        assert_eq!(first.warnings, second.warnings);
    }

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

impl Bsim4 {
    #[inline(always)]
    pub(in crate::devices) fn stamp_with_model(&self, context: &mut StampContext<'_>, baked: Option<(&Model, &SizeParam)>) -> SpiceResult<()> {
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
        let (model, size) = baked.unwrap_or((&prepared.model, &prepared.size));
        load::load(model, h, size, &mut host)?;
        host.finish()
    }

    pub(in crate::devices) fn emit_model(&self, context: &ModelContext) -> SpiceResult<String> {
        use std::fmt::Write;
        let prepared = self.ensure(context.temperature, context.nominal_temperature)?;
        let mut out = String::new();
        // version text is only setup/diagnostic data; the load uses v48intVersion.
        let result = (|| -> std::fmt::Result {
            writeln!(out, "static MODEL: crate::devices::bsim4::fields::Model = crate::devices::bsim4::fields::Model {{")?;
            writeln!(out, "r#type: {},", prepared.model.r#type)?;
writeln!(out, "mobMod: {},", prepared.model.mobMod)?;
writeln!(out, "cvchargeMod: {},", prepared.model.cvchargeMod)?;
writeln!(out, "capMod: {},", prepared.model.capMod)?;
writeln!(out, "dioMod: {},", prepared.model.dioMod)?;
writeln!(out, "trnqsMod: {},", prepared.model.trnqsMod)?;
writeln!(out, "acnqsMod: {},", prepared.model.acnqsMod)?;
writeln!(out, "fnoiMod: {},", prepared.model.fnoiMod)?;
writeln!(out, "tnoiMod: {},", prepared.model.tnoiMod)?;
writeln!(out, "rdsMod: {},", prepared.model.rdsMod)?;
writeln!(out, "rbodyMod: {},", prepared.model.rbodyMod)?;
writeln!(out, "rgateMod: {},", prepared.model.rgateMod)?;
writeln!(out, "perMod: {},", prepared.model.perMod)?;
writeln!(out, "geoMod: {},", prepared.model.geoMod)?;
writeln!(out, "mtrlMod: {},", prepared.model.mtrlMod)?;
writeln!(out, "mtrlCompatMod: {},", prepared.model.mtrlCompatMod)?;
writeln!(out, "gidlMod: {},", prepared.model.gidlMod)?;
writeln!(out, "igcMod: {},", prepared.model.igcMod)?;
writeln!(out, "igbMod: {},", prepared.model.igbMod)?;
writeln!(out, "tempMod: {},", prepared.model.tempMod)?;
writeln!(out, "binUnit: {},", prepared.model.binUnit)?;
writeln!(out, "paramChk: {},", prepared.model.paramChk)?;
writeln!(out, "version: String::new(),")?;
writeln!(out, "v48intVersion: {},", prepared.model.v48intVersion)?;
writeln!(out, "eot: f64::from_bits(0x{:016x}),", prepared.model.eot.to_bits())?;
writeln!(out, "vddeot: f64::from_bits(0x{:016x}),", prepared.model.vddeot.to_bits())?;
writeln!(out, "tempeot: f64::from_bits(0x{:016x}),", prepared.model.tempeot.to_bits())?;
writeln!(out, "leffeot: f64::from_bits(0x{:016x}),", prepared.model.leffeot.to_bits())?;
writeln!(out, "weffeot: f64::from_bits(0x{:016x}),", prepared.model.weffeot.to_bits())?;
writeln!(out, "ados: f64::from_bits(0x{:016x}),", prepared.model.ados.to_bits())?;
writeln!(out, "bdos: f64::from_bits(0x{:016x}),", prepared.model.bdos.to_bits())?;
writeln!(out, "toxe: f64::from_bits(0x{:016x}),", prepared.model.toxe.to_bits())?;
writeln!(out, "toxp: f64::from_bits(0x{:016x}),", prepared.model.toxp.to_bits())?;
writeln!(out, "toxm: f64::from_bits(0x{:016x}),", prepared.model.toxm.to_bits())?;
writeln!(out, "dtox: f64::from_bits(0x{:016x}),", prepared.model.dtox.to_bits())?;
writeln!(out, "epsrox: f64::from_bits(0x{:016x}),", prepared.model.epsrox.to_bits())?;
writeln!(out, "cdsc: f64::from_bits(0x{:016x}),", prepared.model.cdsc.to_bits())?;
writeln!(out, "cdscb: f64::from_bits(0x{:016x}),", prepared.model.cdscb.to_bits())?;
writeln!(out, "cdscd: f64::from_bits(0x{:016x}),", prepared.model.cdscd.to_bits())?;
writeln!(out, "cit: f64::from_bits(0x{:016x}),", prepared.model.cit.to_bits())?;
writeln!(out, "nfactor: f64::from_bits(0x{:016x}),", prepared.model.nfactor.to_bits())?;
writeln!(out, "xj: f64::from_bits(0x{:016x}),", prepared.model.xj.to_bits())?;
writeln!(out, "vsat: f64::from_bits(0x{:016x}),", prepared.model.vsat.to_bits())?;
writeln!(out, "at: f64::from_bits(0x{:016x}),", prepared.model.at.to_bits())?;
writeln!(out, "a0: f64::from_bits(0x{:016x}),", prepared.model.a0.to_bits())?;
writeln!(out, "ags: f64::from_bits(0x{:016x}),", prepared.model.ags.to_bits())?;
writeln!(out, "a1: f64::from_bits(0x{:016x}),", prepared.model.a1.to_bits())?;
writeln!(out, "a2: f64::from_bits(0x{:016x}),", prepared.model.a2.to_bits())?;
writeln!(out, "keta: f64::from_bits(0x{:016x}),", prepared.model.keta.to_bits())?;
writeln!(out, "ketac: f64::from_bits(0x{:016x}),", prepared.model.ketac.to_bits())?;
writeln!(out, "nsub: f64::from_bits(0x{:016x}),", prepared.model.nsub.to_bits())?;
writeln!(out, "phig: f64::from_bits(0x{:016x}),", prepared.model.phig.to_bits())?;
writeln!(out, "epsrgate: f64::from_bits(0x{:016x}),", prepared.model.epsrgate.to_bits())?;
writeln!(out, "easub: f64::from_bits(0x{:016x}),", prepared.model.easub.to_bits())?;
writeln!(out, "epsrsub: f64::from_bits(0x{:016x}),", prepared.model.epsrsub.to_bits())?;
writeln!(out, "ni0sub: f64::from_bits(0x{:016x}),", prepared.model.ni0sub.to_bits())?;
writeln!(out, "bg0sub: f64::from_bits(0x{:016x}),", prepared.model.bg0sub.to_bits())?;
writeln!(out, "tbgasub: f64::from_bits(0x{:016x}),", prepared.model.tbgasub.to_bits())?;
writeln!(out, "tbgbsub: f64::from_bits(0x{:016x}),", prepared.model.tbgbsub.to_bits())?;
writeln!(out, "ndep: f64::from_bits(0x{:016x}),", prepared.model.ndep.to_bits())?;
writeln!(out, "nsd: f64::from_bits(0x{:016x}),", prepared.model.nsd.to_bits())?;
writeln!(out, "phin: f64::from_bits(0x{:016x}),", prepared.model.phin.to_bits())?;
writeln!(out, "ngate: f64::from_bits(0x{:016x}),", prepared.model.ngate.to_bits())?;
writeln!(out, "gamma1: f64::from_bits(0x{:016x}),", prepared.model.gamma1.to_bits())?;
writeln!(out, "gamma2: f64::from_bits(0x{:016x}),", prepared.model.gamma2.to_bits())?;
writeln!(out, "vbx: f64::from_bits(0x{:016x}),", prepared.model.vbx.to_bits())?;
writeln!(out, "vbm: f64::from_bits(0x{:016x}),", prepared.model.vbm.to_bits())?;
writeln!(out, "xt: f64::from_bits(0x{:016x}),", prepared.model.xt.to_bits())?;
writeln!(out, "k1: f64::from_bits(0x{:016x}),", prepared.model.k1.to_bits())?;
writeln!(out, "kt1: f64::from_bits(0x{:016x}),", prepared.model.kt1.to_bits())?;
writeln!(out, "kt1l: f64::from_bits(0x{:016x}),", prepared.model.kt1l.to_bits())?;
writeln!(out, "kt2: f64::from_bits(0x{:016x}),", prepared.model.kt2.to_bits())?;
writeln!(out, "k2: f64::from_bits(0x{:016x}),", prepared.model.k2.to_bits())?;
writeln!(out, "k3: f64::from_bits(0x{:016x}),", prepared.model.k3.to_bits())?;
writeln!(out, "k3b: f64::from_bits(0x{:016x}),", prepared.model.k3b.to_bits())?;
writeln!(out, "w0: f64::from_bits(0x{:016x}),", prepared.model.w0.to_bits())?;
writeln!(out, "dvtp0: f64::from_bits(0x{:016x}),", prepared.model.dvtp0.to_bits())?;
writeln!(out, "dvtp1: f64::from_bits(0x{:016x}),", prepared.model.dvtp1.to_bits())?;
writeln!(out, "dvtp2: f64::from_bits(0x{:016x}),", prepared.model.dvtp2.to_bits())?;
writeln!(out, "dvtp3: f64::from_bits(0x{:016x}),", prepared.model.dvtp3.to_bits())?;
writeln!(out, "dvtp4: f64::from_bits(0x{:016x}),", prepared.model.dvtp4.to_bits())?;
writeln!(out, "dvtp5: f64::from_bits(0x{:016x}),", prepared.model.dvtp5.to_bits())?;
writeln!(out, "lpe0: f64::from_bits(0x{:016x}),", prepared.model.lpe0.to_bits())?;
writeln!(out, "lpeb: f64::from_bits(0x{:016x}),", prepared.model.lpeb.to_bits())?;
writeln!(out, "dvt0: f64::from_bits(0x{:016x}),", prepared.model.dvt0.to_bits())?;
writeln!(out, "dvt1: f64::from_bits(0x{:016x}),", prepared.model.dvt1.to_bits())?;
writeln!(out, "dvt2: f64::from_bits(0x{:016x}),", prepared.model.dvt2.to_bits())?;
writeln!(out, "dvt0w: f64::from_bits(0x{:016x}),", prepared.model.dvt0w.to_bits())?;
writeln!(out, "dvt1w: f64::from_bits(0x{:016x}),", prepared.model.dvt1w.to_bits())?;
writeln!(out, "dvt2w: f64::from_bits(0x{:016x}),", prepared.model.dvt2w.to_bits())?;
writeln!(out, "drout: f64::from_bits(0x{:016x}),", prepared.model.drout.to_bits())?;
writeln!(out, "dsub: f64::from_bits(0x{:016x}),", prepared.model.dsub.to_bits())?;
writeln!(out, "vth0: f64::from_bits(0x{:016x}),", prepared.model.vth0.to_bits())?;
writeln!(out, "eu: f64::from_bits(0x{:016x}),", prepared.model.eu.to_bits())?;
writeln!(out, "ucs: f64::from_bits(0x{:016x}),", prepared.model.ucs.to_bits())?;
writeln!(out, "ua: f64::from_bits(0x{:016x}),", prepared.model.ua.to_bits())?;
writeln!(out, "ua1: f64::from_bits(0x{:016x}),", prepared.model.ua1.to_bits())?;
writeln!(out, "ub: f64::from_bits(0x{:016x}),", prepared.model.ub.to_bits())?;
writeln!(out, "ub1: f64::from_bits(0x{:016x}),", prepared.model.ub1.to_bits())?;
writeln!(out, "uc: f64::from_bits(0x{:016x}),", prepared.model.uc.to_bits())?;
writeln!(out, "uc1: f64::from_bits(0x{:016x}),", prepared.model.uc1.to_bits())?;
writeln!(out, "ud: f64::from_bits(0x{:016x}),", prepared.model.ud.to_bits())?;
writeln!(out, "ud1: f64::from_bits(0x{:016x}),", prepared.model.ud1.to_bits())?;
writeln!(out, "up: f64::from_bits(0x{:016x}),", prepared.model.up.to_bits())?;
writeln!(out, "lp: f64::from_bits(0x{:016x}),", prepared.model.lp.to_bits())?;
writeln!(out, "u0: f64::from_bits(0x{:016x}),", prepared.model.u0.to_bits())?;
writeln!(out, "ute: f64::from_bits(0x{:016x}),", prepared.model.ute.to_bits())?;
writeln!(out, "ucste: f64::from_bits(0x{:016x}),", prepared.model.ucste.to_bits())?;
writeln!(out, "voff: f64::from_bits(0x{:016x}),", prepared.model.voff.to_bits())?;
writeln!(out, "tvoff: f64::from_bits(0x{:016x}),", prepared.model.tvoff.to_bits())?;
writeln!(out, "tnfactor: f64::from_bits(0x{:016x}),", prepared.model.tnfactor.to_bits())?;
writeln!(out, "teta0: f64::from_bits(0x{:016x}),", prepared.model.teta0.to_bits())?;
writeln!(out, "tvoffcv: f64::from_bits(0x{:016x}),", prepared.model.tvoffcv.to_bits())?;
writeln!(out, "minv: f64::from_bits(0x{:016x}),", prepared.model.minv.to_bits())?;
writeln!(out, "minvcv: f64::from_bits(0x{:016x}),", prepared.model.minvcv.to_bits())?;
writeln!(out, "voffl: f64::from_bits(0x{:016x}),", prepared.model.voffl.to_bits())?;
writeln!(out, "voffcvl: f64::from_bits(0x{:016x}),", prepared.model.voffcvl.to_bits())?;
writeln!(out, "delta: f64::from_bits(0x{:016x}),", prepared.model.delta.to_bits())?;
writeln!(out, "rdsw: f64::from_bits(0x{:016x}),", prepared.model.rdsw.to_bits())?;
writeln!(out, "rdswmin: f64::from_bits(0x{:016x}),", prepared.model.rdswmin.to_bits())?;
writeln!(out, "rdwmin: f64::from_bits(0x{:016x}),", prepared.model.rdwmin.to_bits())?;
writeln!(out, "rswmin: f64::from_bits(0x{:016x}),", prepared.model.rswmin.to_bits())?;
writeln!(out, "rsw: f64::from_bits(0x{:016x}),", prepared.model.rsw.to_bits())?;
writeln!(out, "rdw: f64::from_bits(0x{:016x}),", prepared.model.rdw.to_bits())?;
writeln!(out, "prwg: f64::from_bits(0x{:016x}),", prepared.model.prwg.to_bits())?;
writeln!(out, "prwb: f64::from_bits(0x{:016x}),", prepared.model.prwb.to_bits())?;
writeln!(out, "prt: f64::from_bits(0x{:016x}),", prepared.model.prt.to_bits())?;
writeln!(out, "eta0: f64::from_bits(0x{:016x}),", prepared.model.eta0.to_bits())?;
writeln!(out, "etab: f64::from_bits(0x{:016x}),", prepared.model.etab.to_bits())?;
writeln!(out, "pclm: f64::from_bits(0x{:016x}),", prepared.model.pclm.to_bits())?;
writeln!(out, "pdibl1: f64::from_bits(0x{:016x}),", prepared.model.pdibl1.to_bits())?;
writeln!(out, "pdibl2: f64::from_bits(0x{:016x}),", prepared.model.pdibl2.to_bits())?;
writeln!(out, "pdiblb: f64::from_bits(0x{:016x}),", prepared.model.pdiblb.to_bits())?;
writeln!(out, "fprout: f64::from_bits(0x{:016x}),", prepared.model.fprout.to_bits())?;
writeln!(out, "pdits: f64::from_bits(0x{:016x}),", prepared.model.pdits.to_bits())?;
writeln!(out, "pditsd: f64::from_bits(0x{:016x}),", prepared.model.pditsd.to_bits())?;
writeln!(out, "pditsl: f64::from_bits(0x{:016x}),", prepared.model.pditsl.to_bits())?;
writeln!(out, "pscbe1: f64::from_bits(0x{:016x}),", prepared.model.pscbe1.to_bits())?;
writeln!(out, "pscbe2: f64::from_bits(0x{:016x}),", prepared.model.pscbe2.to_bits())?;
writeln!(out, "pvag: f64::from_bits(0x{:016x}),", prepared.model.pvag.to_bits())?;
writeln!(out, "wr: f64::from_bits(0x{:016x}),", prepared.model.wr.to_bits())?;
writeln!(out, "dwg: f64::from_bits(0x{:016x}),", prepared.model.dwg.to_bits())?;
writeln!(out, "dwb: f64::from_bits(0x{:016x}),", prepared.model.dwb.to_bits())?;
writeln!(out, "b0: f64::from_bits(0x{:016x}),", prepared.model.b0.to_bits())?;
writeln!(out, "b1: f64::from_bits(0x{:016x}),", prepared.model.b1.to_bits())?;
writeln!(out, "alpha0: f64::from_bits(0x{:016x}),", prepared.model.alpha0.to_bits())?;
writeln!(out, "alpha1: f64::from_bits(0x{:016x}),", prepared.model.alpha1.to_bits())?;
writeln!(out, "beta0: f64::from_bits(0x{:016x}),", prepared.model.beta0.to_bits())?;
writeln!(out, "agidl: f64::from_bits(0x{:016x}),", prepared.model.agidl.to_bits())?;
writeln!(out, "bgidl: f64::from_bits(0x{:016x}),", prepared.model.bgidl.to_bits())?;
writeln!(out, "cgidl: f64::from_bits(0x{:016x}),", prepared.model.cgidl.to_bits())?;
writeln!(out, "egidl: f64::from_bits(0x{:016x}),", prepared.model.egidl.to_bits())?;
writeln!(out, "fgidl: f64::from_bits(0x{:016x}),", prepared.model.fgidl.to_bits())?;
writeln!(out, "kgidl: f64::from_bits(0x{:016x}),", prepared.model.kgidl.to_bits())?;
writeln!(out, "rgidl: f64::from_bits(0x{:016x}),", prepared.model.rgidl.to_bits())?;
writeln!(out, "agisl: f64::from_bits(0x{:016x}),", prepared.model.agisl.to_bits())?;
writeln!(out, "bgisl: f64::from_bits(0x{:016x}),", prepared.model.bgisl.to_bits())?;
writeln!(out, "cgisl: f64::from_bits(0x{:016x}),", prepared.model.cgisl.to_bits())?;
writeln!(out, "egisl: f64::from_bits(0x{:016x}),", prepared.model.egisl.to_bits())?;
writeln!(out, "fgisl: f64::from_bits(0x{:016x}),", prepared.model.fgisl.to_bits())?;
writeln!(out, "kgisl: f64::from_bits(0x{:016x}),", prepared.model.kgisl.to_bits())?;
writeln!(out, "rgisl: f64::from_bits(0x{:016x}),", prepared.model.rgisl.to_bits())?;
writeln!(out, "aigc: f64::from_bits(0x{:016x}),", prepared.model.aigc.to_bits())?;
writeln!(out, "bigc: f64::from_bits(0x{:016x}),", prepared.model.bigc.to_bits())?;
writeln!(out, "cigc: f64::from_bits(0x{:016x}),", prepared.model.cigc.to_bits())?;
writeln!(out, "aigsd: f64::from_bits(0x{:016x}),", prepared.model.aigsd.to_bits())?;
writeln!(out, "bigsd: f64::from_bits(0x{:016x}),", prepared.model.bigsd.to_bits())?;
writeln!(out, "cigsd: f64::from_bits(0x{:016x}),", prepared.model.cigsd.to_bits())?;
writeln!(out, "aigs: f64::from_bits(0x{:016x}),", prepared.model.aigs.to_bits())?;
writeln!(out, "bigs: f64::from_bits(0x{:016x}),", prepared.model.bigs.to_bits())?;
writeln!(out, "cigs: f64::from_bits(0x{:016x}),", prepared.model.cigs.to_bits())?;
writeln!(out, "aigd: f64::from_bits(0x{:016x}),", prepared.model.aigd.to_bits())?;
writeln!(out, "bigd: f64::from_bits(0x{:016x}),", prepared.model.bigd.to_bits())?;
writeln!(out, "cigd: f64::from_bits(0x{:016x}),", prepared.model.cigd.to_bits())?;
writeln!(out, "aigbacc: f64::from_bits(0x{:016x}),", prepared.model.aigbacc.to_bits())?;
writeln!(out, "bigbacc: f64::from_bits(0x{:016x}),", prepared.model.bigbacc.to_bits())?;
writeln!(out, "cigbacc: f64::from_bits(0x{:016x}),", prepared.model.cigbacc.to_bits())?;
writeln!(out, "aigbinv: f64::from_bits(0x{:016x}),", prepared.model.aigbinv.to_bits())?;
writeln!(out, "bigbinv: f64::from_bits(0x{:016x}),", prepared.model.bigbinv.to_bits())?;
writeln!(out, "cigbinv: f64::from_bits(0x{:016x}),", prepared.model.cigbinv.to_bits())?;
writeln!(out, "nigc: f64::from_bits(0x{:016x}),", prepared.model.nigc.to_bits())?;
writeln!(out, "nigbacc: f64::from_bits(0x{:016x}),", prepared.model.nigbacc.to_bits())?;
writeln!(out, "nigbinv: f64::from_bits(0x{:016x}),", prepared.model.nigbinv.to_bits())?;
writeln!(out, "ntox: f64::from_bits(0x{:016x}),", prepared.model.ntox.to_bits())?;
writeln!(out, "eigbinv: f64::from_bits(0x{:016x}),", prepared.model.eigbinv.to_bits())?;
writeln!(out, "pigcd: f64::from_bits(0x{:016x}),", prepared.model.pigcd.to_bits())?;
writeln!(out, "poxedge: f64::from_bits(0x{:016x}),", prepared.model.poxedge.to_bits())?;
writeln!(out, "toxref: f64::from_bits(0x{:016x}),", prepared.model.toxref.to_bits())?;
writeln!(out, "ijthdfwd: f64::from_bits(0x{:016x}),", prepared.model.ijthdfwd.to_bits())?;
writeln!(out, "ijthsfwd: f64::from_bits(0x{:016x}),", prepared.model.ijthsfwd.to_bits())?;
writeln!(out, "ijthdrev: f64::from_bits(0x{:016x}),", prepared.model.ijthdrev.to_bits())?;
writeln!(out, "ijthsrev: f64::from_bits(0x{:016x}),", prepared.model.ijthsrev.to_bits())?;
writeln!(out, "xjbvd: f64::from_bits(0x{:016x}),", prepared.model.xjbvd.to_bits())?;
writeln!(out, "xjbvs: f64::from_bits(0x{:016x}),", prepared.model.xjbvs.to_bits())?;
writeln!(out, "bvd: f64::from_bits(0x{:016x}),", prepared.model.bvd.to_bits())?;
writeln!(out, "bvs: f64::from_bits(0x{:016x}),", prepared.model.bvs.to_bits())?;
writeln!(out, "jtss: f64::from_bits(0x{:016x}),", prepared.model.jtss.to_bits())?;
writeln!(out, "jtsd: f64::from_bits(0x{:016x}),", prepared.model.jtsd.to_bits())?;
writeln!(out, "jtssws: f64::from_bits(0x{:016x}),", prepared.model.jtssws.to_bits())?;
writeln!(out, "jtsswd: f64::from_bits(0x{:016x}),", prepared.model.jtsswd.to_bits())?;
writeln!(out, "jtsswgs: f64::from_bits(0x{:016x}),", prepared.model.jtsswgs.to_bits())?;
writeln!(out, "jtsswgd: f64::from_bits(0x{:016x}),", prepared.model.jtsswgd.to_bits())?;
writeln!(out, "jtweff: f64::from_bits(0x{:016x}),", prepared.model.jtweff.to_bits())?;
writeln!(out, "njts: f64::from_bits(0x{:016x}),", prepared.model.njts.to_bits())?;
writeln!(out, "njtssw: f64::from_bits(0x{:016x}),", prepared.model.njtssw.to_bits())?;
writeln!(out, "njtsswg: f64::from_bits(0x{:016x}),", prepared.model.njtsswg.to_bits())?;
writeln!(out, "njtsd: f64::from_bits(0x{:016x}),", prepared.model.njtsd.to_bits())?;
writeln!(out, "njtsswd: f64::from_bits(0x{:016x}),", prepared.model.njtsswd.to_bits())?;
writeln!(out, "njtsswgd: f64::from_bits(0x{:016x}),", prepared.model.njtsswgd.to_bits())?;
writeln!(out, "xtss: f64::from_bits(0x{:016x}),", prepared.model.xtss.to_bits())?;
writeln!(out, "xtsd: f64::from_bits(0x{:016x}),", prepared.model.xtsd.to_bits())?;
writeln!(out, "xtssws: f64::from_bits(0x{:016x}),", prepared.model.xtssws.to_bits())?;
writeln!(out, "xtsswd: f64::from_bits(0x{:016x}),", prepared.model.xtsswd.to_bits())?;
writeln!(out, "xtsswgs: f64::from_bits(0x{:016x}),", prepared.model.xtsswgs.to_bits())?;
writeln!(out, "xtsswgd: f64::from_bits(0x{:016x}),", prepared.model.xtsswgd.to_bits())?;
writeln!(out, "tnjts: f64::from_bits(0x{:016x}),", prepared.model.tnjts.to_bits())?;
writeln!(out, "tnjtssw: f64::from_bits(0x{:016x}),", prepared.model.tnjtssw.to_bits())?;
writeln!(out, "tnjtsswg: f64::from_bits(0x{:016x}),", prepared.model.tnjtsswg.to_bits())?;
writeln!(out, "tnjtsd: f64::from_bits(0x{:016x}),", prepared.model.tnjtsd.to_bits())?;
writeln!(out, "tnjtsswd: f64::from_bits(0x{:016x}),", prepared.model.tnjtsswd.to_bits())?;
writeln!(out, "tnjtsswgd: f64::from_bits(0x{:016x}),", prepared.model.tnjtsswgd.to_bits())?;
writeln!(out, "vtss: f64::from_bits(0x{:016x}),", prepared.model.vtss.to_bits())?;
writeln!(out, "vtsd: f64::from_bits(0x{:016x}),", prepared.model.vtsd.to_bits())?;
writeln!(out, "vtssws: f64::from_bits(0x{:016x}),", prepared.model.vtssws.to_bits())?;
writeln!(out, "vtsswd: f64::from_bits(0x{:016x}),", prepared.model.vtsswd.to_bits())?;
writeln!(out, "vtsswgs: f64::from_bits(0x{:016x}),", prepared.model.vtsswgs.to_bits())?;
writeln!(out, "vtsswgd: f64::from_bits(0x{:016x}),", prepared.model.vtsswgd.to_bits())?;
writeln!(out, "xrcrg1: f64::from_bits(0x{:016x}),", prepared.model.xrcrg1.to_bits())?;
writeln!(out, "xrcrg2: f64::from_bits(0x{:016x}),", prepared.model.xrcrg2.to_bits())?;
writeln!(out, "lambda: f64::from_bits(0x{:016x}),", prepared.model.lambda.to_bits())?;
writeln!(out, "vtl: f64::from_bits(0x{:016x}),", prepared.model.vtl.to_bits())?;
writeln!(out, "lc: f64::from_bits(0x{:016x}),", prepared.model.lc.to_bits())?;
writeln!(out, "xn: f64::from_bits(0x{:016x}),", prepared.model.xn.to_bits())?;
writeln!(out, "vfbsdoff: f64::from_bits(0x{:016x}),", prepared.model.vfbsdoff.to_bits())?;
writeln!(out, "lintnoi: f64::from_bits(0x{:016x}),", prepared.model.lintnoi.to_bits())?;
writeln!(out, "tvfbsdoff: f64::from_bits(0x{:016x}),", prepared.model.tvfbsdoff.to_bits())?;
writeln!(out, "vfb: f64::from_bits(0x{:016x}),", prepared.model.vfb.to_bits())?;
writeln!(out, "gbmin: f64::from_bits(0x{:016x}),", prepared.model.gbmin.to_bits())?;
writeln!(out, "rbdb: f64::from_bits(0x{:016x}),", prepared.model.rbdb.to_bits())?;
writeln!(out, "rbsb: f64::from_bits(0x{:016x}),", prepared.model.rbsb.to_bits())?;
writeln!(out, "rbpb: f64::from_bits(0x{:016x}),", prepared.model.rbpb.to_bits())?;
writeln!(out, "rbps: f64::from_bits(0x{:016x}),", prepared.model.rbps.to_bits())?;
writeln!(out, "rbpd: f64::from_bits(0x{:016x}),", prepared.model.rbpd.to_bits())?;
writeln!(out, "rbps0: f64::from_bits(0x{:016x}),", prepared.model.rbps0.to_bits())?;
writeln!(out, "rbpsl: f64::from_bits(0x{:016x}),", prepared.model.rbpsl.to_bits())?;
writeln!(out, "rbpsw: f64::from_bits(0x{:016x}),", prepared.model.rbpsw.to_bits())?;
writeln!(out, "rbpsnf: f64::from_bits(0x{:016x}),", prepared.model.rbpsnf.to_bits())?;
writeln!(out, "rbpd0: f64::from_bits(0x{:016x}),", prepared.model.rbpd0.to_bits())?;
writeln!(out, "rbpdl: f64::from_bits(0x{:016x}),", prepared.model.rbpdl.to_bits())?;
writeln!(out, "rbpdw: f64::from_bits(0x{:016x}),", prepared.model.rbpdw.to_bits())?;
writeln!(out, "rbpdnf: f64::from_bits(0x{:016x}),", prepared.model.rbpdnf.to_bits())?;
writeln!(out, "rbpbx0: f64::from_bits(0x{:016x}),", prepared.model.rbpbx0.to_bits())?;
writeln!(out, "rbpbxl: f64::from_bits(0x{:016x}),", prepared.model.rbpbxl.to_bits())?;
writeln!(out, "rbpbxw: f64::from_bits(0x{:016x}),", prepared.model.rbpbxw.to_bits())?;
writeln!(out, "rbpbxnf: f64::from_bits(0x{:016x}),", prepared.model.rbpbxnf.to_bits())?;
writeln!(out, "rbpby0: f64::from_bits(0x{:016x}),", prepared.model.rbpby0.to_bits())?;
writeln!(out, "rbpbyl: f64::from_bits(0x{:016x}),", prepared.model.rbpbyl.to_bits())?;
writeln!(out, "rbpbyw: f64::from_bits(0x{:016x}),", prepared.model.rbpbyw.to_bits())?;
writeln!(out, "rbpbynf: f64::from_bits(0x{:016x}),", prepared.model.rbpbynf.to_bits())?;
writeln!(out, "rbsbx0: f64::from_bits(0x{:016x}),", prepared.model.rbsbx0.to_bits())?;
writeln!(out, "rbsby0: f64::from_bits(0x{:016x}),", prepared.model.rbsby0.to_bits())?;
writeln!(out, "rbdbx0: f64::from_bits(0x{:016x}),", prepared.model.rbdbx0.to_bits())?;
writeln!(out, "rbdby0: f64::from_bits(0x{:016x}),", prepared.model.rbdby0.to_bits())?;
writeln!(out, "rbsdbxl: f64::from_bits(0x{:016x}),", prepared.model.rbsdbxl.to_bits())?;
writeln!(out, "rbsdbxw: f64::from_bits(0x{:016x}),", prepared.model.rbsdbxw.to_bits())?;
writeln!(out, "rbsdbxnf: f64::from_bits(0x{:016x}),", prepared.model.rbsdbxnf.to_bits())?;
writeln!(out, "rbsdbyl: f64::from_bits(0x{:016x}),", prepared.model.rbsdbyl.to_bits())?;
writeln!(out, "rbsdbyw: f64::from_bits(0x{:016x}),", prepared.model.rbsdbyw.to_bits())?;
writeln!(out, "rbsdbynf: f64::from_bits(0x{:016x}),", prepared.model.rbsdbynf.to_bits())?;
writeln!(out, "tnoia: f64::from_bits(0x{:016x}),", prepared.model.tnoia.to_bits())?;
writeln!(out, "tnoib: f64::from_bits(0x{:016x}),", prepared.model.tnoib.to_bits())?;
writeln!(out, "tnoic: f64::from_bits(0x{:016x}),", prepared.model.tnoic.to_bits())?;
writeln!(out, "rnoia: f64::from_bits(0x{:016x}),", prepared.model.rnoia.to_bits())?;
writeln!(out, "rnoib: f64::from_bits(0x{:016x}),", prepared.model.rnoib.to_bits())?;
writeln!(out, "rnoic: f64::from_bits(0x{:016x}),", prepared.model.rnoic.to_bits())?;
writeln!(out, "ntnoi: f64::from_bits(0x{:016x}),", prepared.model.ntnoi.to_bits())?;
writeln!(out, "cgsl: f64::from_bits(0x{:016x}),", prepared.model.cgsl.to_bits())?;
writeln!(out, "cgdl: f64::from_bits(0x{:016x}),", prepared.model.cgdl.to_bits())?;
writeln!(out, "ckappas: f64::from_bits(0x{:016x}),", prepared.model.ckappas.to_bits())?;
writeln!(out, "ckappad: f64::from_bits(0x{:016x}),", prepared.model.ckappad.to_bits())?;
writeln!(out, "cf: f64::from_bits(0x{:016x}),", prepared.model.cf.to_bits())?;
writeln!(out, "vfbcv: f64::from_bits(0x{:016x}),", prepared.model.vfbcv.to_bits())?;
writeln!(out, "clc: f64::from_bits(0x{:016x}),", prepared.model.clc.to_bits())?;
writeln!(out, "cle: f64::from_bits(0x{:016x}),", prepared.model.cle.to_bits())?;
writeln!(out, "dwc: f64::from_bits(0x{:016x}),", prepared.model.dwc.to_bits())?;
writeln!(out, "dlc: f64::from_bits(0x{:016x}),", prepared.model.dlc.to_bits())?;
writeln!(out, "xw: f64::from_bits(0x{:016x}),", prepared.model.xw.to_bits())?;
writeln!(out, "xl: f64::from_bits(0x{:016x}),", prepared.model.xl.to_bits())?;
writeln!(out, "dlcig: f64::from_bits(0x{:016x}),", prepared.model.dlcig.to_bits())?;
writeln!(out, "dlcigd: f64::from_bits(0x{:016x}),", prepared.model.dlcigd.to_bits())?;
writeln!(out, "dwj: f64::from_bits(0x{:016x}),", prepared.model.dwj.to_bits())?;
writeln!(out, "noff: f64::from_bits(0x{:016x}),", prepared.model.noff.to_bits())?;
writeln!(out, "voffcv: f64::from_bits(0x{:016x}),", prepared.model.voffcv.to_bits())?;
writeln!(out, "acde: f64::from_bits(0x{:016x}),", prepared.model.acde.to_bits())?;
writeln!(out, "moin: f64::from_bits(0x{:016x}),", prepared.model.moin.to_bits())?;
writeln!(out, "tcj: f64::from_bits(0x{:016x}),", prepared.model.tcj.to_bits())?;
writeln!(out, "tcjsw: f64::from_bits(0x{:016x}),", prepared.model.tcjsw.to_bits())?;
writeln!(out, "tcjswg: f64::from_bits(0x{:016x}),", prepared.model.tcjswg.to_bits())?;
writeln!(out, "tpb: f64::from_bits(0x{:016x}),", prepared.model.tpb.to_bits())?;
writeln!(out, "tpbsw: f64::from_bits(0x{:016x}),", prepared.model.tpbsw.to_bits())?;
writeln!(out, "tpbswg: f64::from_bits(0x{:016x}),", prepared.model.tpbswg.to_bits())?;
writeln!(out, "dmcg: f64::from_bits(0x{:016x}),", prepared.model.dmcg.to_bits())?;
writeln!(out, "dmci: f64::from_bits(0x{:016x}),", prepared.model.dmci.to_bits())?;
writeln!(out, "dmdg: f64::from_bits(0x{:016x}),", prepared.model.dmdg.to_bits())?;
writeln!(out, "dmcgt: f64::from_bits(0x{:016x}),", prepared.model.dmcgt.to_bits())?;
writeln!(out, "xgw: f64::from_bits(0x{:016x}),", prepared.model.xgw.to_bits())?;
writeln!(out, "xgl: f64::from_bits(0x{:016x}),", prepared.model.xgl.to_bits())?;
writeln!(out, "rshg: f64::from_bits(0x{:016x}),", prepared.model.rshg.to_bits())?;
writeln!(out, "ngcon: f64::from_bits(0x{:016x}),", prepared.model.ngcon.to_bits())?;
writeln!(out, "lcdsc: f64::from_bits(0x{:016x}),", prepared.model.lcdsc.to_bits())?;
writeln!(out, "lcdscb: f64::from_bits(0x{:016x}),", prepared.model.lcdscb.to_bits())?;
writeln!(out, "lcdscd: f64::from_bits(0x{:016x}),", prepared.model.lcdscd.to_bits())?;
writeln!(out, "lcit: f64::from_bits(0x{:016x}),", prepared.model.lcit.to_bits())?;
writeln!(out, "lnfactor: f64::from_bits(0x{:016x}),", prepared.model.lnfactor.to_bits())?;
writeln!(out, "lxj: f64::from_bits(0x{:016x}),", prepared.model.lxj.to_bits())?;
writeln!(out, "lvsat: f64::from_bits(0x{:016x}),", prepared.model.lvsat.to_bits())?;
writeln!(out, "lat: f64::from_bits(0x{:016x}),", prepared.model.lat.to_bits())?;
writeln!(out, "la0: f64::from_bits(0x{:016x}),", prepared.model.la0.to_bits())?;
writeln!(out, "lags: f64::from_bits(0x{:016x}),", prepared.model.lags.to_bits())?;
writeln!(out, "la1: f64::from_bits(0x{:016x}),", prepared.model.la1.to_bits())?;
writeln!(out, "la2: f64::from_bits(0x{:016x}),", prepared.model.la2.to_bits())?;
writeln!(out, "lketa: f64::from_bits(0x{:016x}),", prepared.model.lketa.to_bits())?;
writeln!(out, "lketac: f64::from_bits(0x{:016x}),", prepared.model.lketac.to_bits())?;
writeln!(out, "lnsub: f64::from_bits(0x{:016x}),", prepared.model.lnsub.to_bits())?;
writeln!(out, "lndep: f64::from_bits(0x{:016x}),", prepared.model.lndep.to_bits())?;
writeln!(out, "lnsd: f64::from_bits(0x{:016x}),", prepared.model.lnsd.to_bits())?;
writeln!(out, "lphin: f64::from_bits(0x{:016x}),", prepared.model.lphin.to_bits())?;
writeln!(out, "lngate: f64::from_bits(0x{:016x}),", prepared.model.lngate.to_bits())?;
writeln!(out, "lgamma1: f64::from_bits(0x{:016x}),", prepared.model.lgamma1.to_bits())?;
writeln!(out, "lgamma2: f64::from_bits(0x{:016x}),", prepared.model.lgamma2.to_bits())?;
writeln!(out, "lvbx: f64::from_bits(0x{:016x}),", prepared.model.lvbx.to_bits())?;
writeln!(out, "lvbm: f64::from_bits(0x{:016x}),", prepared.model.lvbm.to_bits())?;
writeln!(out, "lxt: f64::from_bits(0x{:016x}),", prepared.model.lxt.to_bits())?;
writeln!(out, "lk1: f64::from_bits(0x{:016x}),", prepared.model.lk1.to_bits())?;
writeln!(out, "lkt1: f64::from_bits(0x{:016x}),", prepared.model.lkt1.to_bits())?;
writeln!(out, "lkt1l: f64::from_bits(0x{:016x}),", prepared.model.lkt1l.to_bits())?;
writeln!(out, "lkt2: f64::from_bits(0x{:016x}),", prepared.model.lkt2.to_bits())?;
writeln!(out, "lk2: f64::from_bits(0x{:016x}),", prepared.model.lk2.to_bits())?;
writeln!(out, "lk3: f64::from_bits(0x{:016x}),", prepared.model.lk3.to_bits())?;
writeln!(out, "lk3b: f64::from_bits(0x{:016x}),", prepared.model.lk3b.to_bits())?;
writeln!(out, "lw0: f64::from_bits(0x{:016x}),", prepared.model.lw0.to_bits())?;
writeln!(out, "ldvtp0: f64::from_bits(0x{:016x}),", prepared.model.ldvtp0.to_bits())?;
writeln!(out, "ldvtp1: f64::from_bits(0x{:016x}),", prepared.model.ldvtp1.to_bits())?;
writeln!(out, "ldvtp2: f64::from_bits(0x{:016x}),", prepared.model.ldvtp2.to_bits())?;
writeln!(out, "ldvtp3: f64::from_bits(0x{:016x}),", prepared.model.ldvtp3.to_bits())?;
writeln!(out, "ldvtp4: f64::from_bits(0x{:016x}),", prepared.model.ldvtp4.to_bits())?;
writeln!(out, "ldvtp5: f64::from_bits(0x{:016x}),", prepared.model.ldvtp5.to_bits())?;
writeln!(out, "llpe0: f64::from_bits(0x{:016x}),", prepared.model.llpe0.to_bits())?;
writeln!(out, "llpeb: f64::from_bits(0x{:016x}),", prepared.model.llpeb.to_bits())?;
writeln!(out, "ldvt0: f64::from_bits(0x{:016x}),", prepared.model.ldvt0.to_bits())?;
writeln!(out, "ldvt1: f64::from_bits(0x{:016x}),", prepared.model.ldvt1.to_bits())?;
writeln!(out, "ldvt2: f64::from_bits(0x{:016x}),", prepared.model.ldvt2.to_bits())?;
writeln!(out, "ldvt0w: f64::from_bits(0x{:016x}),", prepared.model.ldvt0w.to_bits())?;
writeln!(out, "ldvt1w: f64::from_bits(0x{:016x}),", prepared.model.ldvt1w.to_bits())?;
writeln!(out, "ldvt2w: f64::from_bits(0x{:016x}),", prepared.model.ldvt2w.to_bits())?;
writeln!(out, "ldrout: f64::from_bits(0x{:016x}),", prepared.model.ldrout.to_bits())?;
writeln!(out, "ldsub: f64::from_bits(0x{:016x}),", prepared.model.ldsub.to_bits())?;
writeln!(out, "lvth0: f64::from_bits(0x{:016x}),", prepared.model.lvth0.to_bits())?;
writeln!(out, "lua: f64::from_bits(0x{:016x}),", prepared.model.lua.to_bits())?;
writeln!(out, "lua1: f64::from_bits(0x{:016x}),", prepared.model.lua1.to_bits())?;
writeln!(out, "lub: f64::from_bits(0x{:016x}),", prepared.model.lub.to_bits())?;
writeln!(out, "lub1: f64::from_bits(0x{:016x}),", prepared.model.lub1.to_bits())?;
writeln!(out, "luc: f64::from_bits(0x{:016x}),", prepared.model.luc.to_bits())?;
writeln!(out, "luc1: f64::from_bits(0x{:016x}),", prepared.model.luc1.to_bits())?;
writeln!(out, "lud: f64::from_bits(0x{:016x}),", prepared.model.lud.to_bits())?;
writeln!(out, "lud1: f64::from_bits(0x{:016x}),", prepared.model.lud1.to_bits())?;
writeln!(out, "lup: f64::from_bits(0x{:016x}),", prepared.model.lup.to_bits())?;
writeln!(out, "llp: f64::from_bits(0x{:016x}),", prepared.model.llp.to_bits())?;
writeln!(out, "lu0: f64::from_bits(0x{:016x}),", prepared.model.lu0.to_bits())?;
writeln!(out, "leu: f64::from_bits(0x{:016x}),", prepared.model.leu.to_bits())?;
writeln!(out, "lucs: f64::from_bits(0x{:016x}),", prepared.model.lucs.to_bits())?;
writeln!(out, "lute: f64::from_bits(0x{:016x}),", prepared.model.lute.to_bits())?;
writeln!(out, "lucste: f64::from_bits(0x{:016x}),", prepared.model.lucste.to_bits())?;
writeln!(out, "lvoff: f64::from_bits(0x{:016x}),", prepared.model.lvoff.to_bits())?;
writeln!(out, "ltvoff: f64::from_bits(0x{:016x}),", prepared.model.ltvoff.to_bits())?;
writeln!(out, "ltnfactor: f64::from_bits(0x{:016x}),", prepared.model.ltnfactor.to_bits())?;
writeln!(out, "lteta0: f64::from_bits(0x{:016x}),", prepared.model.lteta0.to_bits())?;
writeln!(out, "ltvoffcv: f64::from_bits(0x{:016x}),", prepared.model.ltvoffcv.to_bits())?;
writeln!(out, "lminv: f64::from_bits(0x{:016x}),", prepared.model.lminv.to_bits())?;
writeln!(out, "lminvcv: f64::from_bits(0x{:016x}),", prepared.model.lminvcv.to_bits())?;
writeln!(out, "ldelta: f64::from_bits(0x{:016x}),", prepared.model.ldelta.to_bits())?;
writeln!(out, "lrdsw: f64::from_bits(0x{:016x}),", prepared.model.lrdsw.to_bits())?;
writeln!(out, "lrsw: f64::from_bits(0x{:016x}),", prepared.model.lrsw.to_bits())?;
writeln!(out, "lrdw: f64::from_bits(0x{:016x}),", prepared.model.lrdw.to_bits())?;
writeln!(out, "lprwg: f64::from_bits(0x{:016x}),", prepared.model.lprwg.to_bits())?;
writeln!(out, "lprwb: f64::from_bits(0x{:016x}),", prepared.model.lprwb.to_bits())?;
writeln!(out, "lprt: f64::from_bits(0x{:016x}),", prepared.model.lprt.to_bits())?;
writeln!(out, "leta0: f64::from_bits(0x{:016x}),", prepared.model.leta0.to_bits())?;
writeln!(out, "letab: f64::from_bits(0x{:016x}),", prepared.model.letab.to_bits())?;
writeln!(out, "lpclm: f64::from_bits(0x{:016x}),", prepared.model.lpclm.to_bits())?;
writeln!(out, "lpdibl1: f64::from_bits(0x{:016x}),", prepared.model.lpdibl1.to_bits())?;
writeln!(out, "lpdibl2: f64::from_bits(0x{:016x}),", prepared.model.lpdibl2.to_bits())?;
writeln!(out, "lpdiblb: f64::from_bits(0x{:016x}),", prepared.model.lpdiblb.to_bits())?;
writeln!(out, "lfprout: f64::from_bits(0x{:016x}),", prepared.model.lfprout.to_bits())?;
writeln!(out, "lpdits: f64::from_bits(0x{:016x}),", prepared.model.lpdits.to_bits())?;
writeln!(out, "lpditsd: f64::from_bits(0x{:016x}),", prepared.model.lpditsd.to_bits())?;
writeln!(out, "lpscbe1: f64::from_bits(0x{:016x}),", prepared.model.lpscbe1.to_bits())?;
writeln!(out, "lpscbe2: f64::from_bits(0x{:016x}),", prepared.model.lpscbe2.to_bits())?;
writeln!(out, "lpvag: f64::from_bits(0x{:016x}),", prepared.model.lpvag.to_bits())?;
writeln!(out, "lwr: f64::from_bits(0x{:016x}),", prepared.model.lwr.to_bits())?;
writeln!(out, "ldwg: f64::from_bits(0x{:016x}),", prepared.model.ldwg.to_bits())?;
writeln!(out, "ldwb: f64::from_bits(0x{:016x}),", prepared.model.ldwb.to_bits())?;
writeln!(out, "lb0: f64::from_bits(0x{:016x}),", prepared.model.lb0.to_bits())?;
writeln!(out, "lb1: f64::from_bits(0x{:016x}),", prepared.model.lb1.to_bits())?;
writeln!(out, "lalpha0: f64::from_bits(0x{:016x}),", prepared.model.lalpha0.to_bits())?;
writeln!(out, "lalpha1: f64::from_bits(0x{:016x}),", prepared.model.lalpha1.to_bits())?;
writeln!(out, "lbeta0: f64::from_bits(0x{:016x}),", prepared.model.lbeta0.to_bits())?;
writeln!(out, "lvfb: f64::from_bits(0x{:016x}),", prepared.model.lvfb.to_bits())?;
writeln!(out, "lagidl: f64::from_bits(0x{:016x}),", prepared.model.lagidl.to_bits())?;
writeln!(out, "lbgidl: f64::from_bits(0x{:016x}),", prepared.model.lbgidl.to_bits())?;
writeln!(out, "lcgidl: f64::from_bits(0x{:016x}),", prepared.model.lcgidl.to_bits())?;
writeln!(out, "legidl: f64::from_bits(0x{:016x}),", prepared.model.legidl.to_bits())?;
writeln!(out, "lfgidl: f64::from_bits(0x{:016x}),", prepared.model.lfgidl.to_bits())?;
writeln!(out, "lkgidl: f64::from_bits(0x{:016x}),", prepared.model.lkgidl.to_bits())?;
writeln!(out, "lrgidl: f64::from_bits(0x{:016x}),", prepared.model.lrgidl.to_bits())?;
writeln!(out, "lagisl: f64::from_bits(0x{:016x}),", prepared.model.lagisl.to_bits())?;
writeln!(out, "lbgisl: f64::from_bits(0x{:016x}),", prepared.model.lbgisl.to_bits())?;
writeln!(out, "lcgisl: f64::from_bits(0x{:016x}),", prepared.model.lcgisl.to_bits())?;
writeln!(out, "legisl: f64::from_bits(0x{:016x}),", prepared.model.legisl.to_bits())?;
writeln!(out, "lfgisl: f64::from_bits(0x{:016x}),", prepared.model.lfgisl.to_bits())?;
writeln!(out, "lkgisl: f64::from_bits(0x{:016x}),", prepared.model.lkgisl.to_bits())?;
writeln!(out, "lrgisl: f64::from_bits(0x{:016x}),", prepared.model.lrgisl.to_bits())?;
writeln!(out, "laigc: f64::from_bits(0x{:016x}),", prepared.model.laigc.to_bits())?;
writeln!(out, "lbigc: f64::from_bits(0x{:016x}),", prepared.model.lbigc.to_bits())?;
writeln!(out, "lcigc: f64::from_bits(0x{:016x}),", prepared.model.lcigc.to_bits())?;
writeln!(out, "laigsd: f64::from_bits(0x{:016x}),", prepared.model.laigsd.to_bits())?;
writeln!(out, "lbigsd: f64::from_bits(0x{:016x}),", prepared.model.lbigsd.to_bits())?;
writeln!(out, "lcigsd: f64::from_bits(0x{:016x}),", prepared.model.lcigsd.to_bits())?;
writeln!(out, "laigs: f64::from_bits(0x{:016x}),", prepared.model.laigs.to_bits())?;
writeln!(out, "lbigs: f64::from_bits(0x{:016x}),", prepared.model.lbigs.to_bits())?;
writeln!(out, "lcigs: f64::from_bits(0x{:016x}),", prepared.model.lcigs.to_bits())?;
writeln!(out, "laigd: f64::from_bits(0x{:016x}),", prepared.model.laigd.to_bits())?;
writeln!(out, "lbigd: f64::from_bits(0x{:016x}),", prepared.model.lbigd.to_bits())?;
writeln!(out, "lcigd: f64::from_bits(0x{:016x}),", prepared.model.lcigd.to_bits())?;
writeln!(out, "laigbacc: f64::from_bits(0x{:016x}),", prepared.model.laigbacc.to_bits())?;
writeln!(out, "lbigbacc: f64::from_bits(0x{:016x}),", prepared.model.lbigbacc.to_bits())?;
writeln!(out, "lcigbacc: f64::from_bits(0x{:016x}),", prepared.model.lcigbacc.to_bits())?;
writeln!(out, "laigbinv: f64::from_bits(0x{:016x}),", prepared.model.laigbinv.to_bits())?;
writeln!(out, "lbigbinv: f64::from_bits(0x{:016x}),", prepared.model.lbigbinv.to_bits())?;
writeln!(out, "lcigbinv: f64::from_bits(0x{:016x}),", prepared.model.lcigbinv.to_bits())?;
writeln!(out, "lnigc: f64::from_bits(0x{:016x}),", prepared.model.lnigc.to_bits())?;
writeln!(out, "lnigbacc: f64::from_bits(0x{:016x}),", prepared.model.lnigbacc.to_bits())?;
writeln!(out, "lnigbinv: f64::from_bits(0x{:016x}),", prepared.model.lnigbinv.to_bits())?;
writeln!(out, "lntox: f64::from_bits(0x{:016x}),", prepared.model.lntox.to_bits())?;
writeln!(out, "leigbinv: f64::from_bits(0x{:016x}),", prepared.model.leigbinv.to_bits())?;
writeln!(out, "lpigcd: f64::from_bits(0x{:016x}),", prepared.model.lpigcd.to_bits())?;
writeln!(out, "lpoxedge: f64::from_bits(0x{:016x}),", prepared.model.lpoxedge.to_bits())?;
writeln!(out, "lxrcrg1: f64::from_bits(0x{:016x}),", prepared.model.lxrcrg1.to_bits())?;
writeln!(out, "lxrcrg2: f64::from_bits(0x{:016x}),", prepared.model.lxrcrg2.to_bits())?;
writeln!(out, "llambda: f64::from_bits(0x{:016x}),", prepared.model.llambda.to_bits())?;
writeln!(out, "lvtl: f64::from_bits(0x{:016x}),", prepared.model.lvtl.to_bits())?;
writeln!(out, "lxn: f64::from_bits(0x{:016x}),", prepared.model.lxn.to_bits())?;
writeln!(out, "lvfbsdoff: f64::from_bits(0x{:016x}),", prepared.model.lvfbsdoff.to_bits())?;
writeln!(out, "ltvfbsdoff: f64::from_bits(0x{:016x}),", prepared.model.ltvfbsdoff.to_bits())?;
writeln!(out, "lcgsl: f64::from_bits(0x{:016x}),", prepared.model.lcgsl.to_bits())?;
writeln!(out, "lcgdl: f64::from_bits(0x{:016x}),", prepared.model.lcgdl.to_bits())?;
writeln!(out, "lckappas: f64::from_bits(0x{:016x}),", prepared.model.lckappas.to_bits())?;
writeln!(out, "lckappad: f64::from_bits(0x{:016x}),", prepared.model.lckappad.to_bits())?;
writeln!(out, "lcf: f64::from_bits(0x{:016x}),", prepared.model.lcf.to_bits())?;
writeln!(out, "lclc: f64::from_bits(0x{:016x}),", prepared.model.lclc.to_bits())?;
writeln!(out, "lcle: f64::from_bits(0x{:016x}),", prepared.model.lcle.to_bits())?;
writeln!(out, "lvfbcv: f64::from_bits(0x{:016x}),", prepared.model.lvfbcv.to_bits())?;
writeln!(out, "lnoff: f64::from_bits(0x{:016x}),", prepared.model.lnoff.to_bits())?;
writeln!(out, "lvoffcv: f64::from_bits(0x{:016x}),", prepared.model.lvoffcv.to_bits())?;
writeln!(out, "lacde: f64::from_bits(0x{:016x}),", prepared.model.lacde.to_bits())?;
writeln!(out, "lmoin: f64::from_bits(0x{:016x}),", prepared.model.lmoin.to_bits())?;
writeln!(out, "wcdsc: f64::from_bits(0x{:016x}),", prepared.model.wcdsc.to_bits())?;
writeln!(out, "wcdscb: f64::from_bits(0x{:016x}),", prepared.model.wcdscb.to_bits())?;
writeln!(out, "wcdscd: f64::from_bits(0x{:016x}),", prepared.model.wcdscd.to_bits())?;
writeln!(out, "wcit: f64::from_bits(0x{:016x}),", prepared.model.wcit.to_bits())?;
writeln!(out, "wnfactor: f64::from_bits(0x{:016x}),", prepared.model.wnfactor.to_bits())?;
writeln!(out, "wxj: f64::from_bits(0x{:016x}),", prepared.model.wxj.to_bits())?;
writeln!(out, "wvsat: f64::from_bits(0x{:016x}),", prepared.model.wvsat.to_bits())?;
writeln!(out, "wat: f64::from_bits(0x{:016x}),", prepared.model.wat.to_bits())?;
writeln!(out, "wa0: f64::from_bits(0x{:016x}),", prepared.model.wa0.to_bits())?;
writeln!(out, "wags: f64::from_bits(0x{:016x}),", prepared.model.wags.to_bits())?;
writeln!(out, "wa1: f64::from_bits(0x{:016x}),", prepared.model.wa1.to_bits())?;
writeln!(out, "wa2: f64::from_bits(0x{:016x}),", prepared.model.wa2.to_bits())?;
writeln!(out, "wketa: f64::from_bits(0x{:016x}),", prepared.model.wketa.to_bits())?;
writeln!(out, "wketac: f64::from_bits(0x{:016x}),", prepared.model.wketac.to_bits())?;
writeln!(out, "wnsub: f64::from_bits(0x{:016x}),", prepared.model.wnsub.to_bits())?;
writeln!(out, "wndep: f64::from_bits(0x{:016x}),", prepared.model.wndep.to_bits())?;
writeln!(out, "wnsd: f64::from_bits(0x{:016x}),", prepared.model.wnsd.to_bits())?;
writeln!(out, "wphin: f64::from_bits(0x{:016x}),", prepared.model.wphin.to_bits())?;
writeln!(out, "wngate: f64::from_bits(0x{:016x}),", prepared.model.wngate.to_bits())?;
writeln!(out, "wgamma1: f64::from_bits(0x{:016x}),", prepared.model.wgamma1.to_bits())?;
writeln!(out, "wgamma2: f64::from_bits(0x{:016x}),", prepared.model.wgamma2.to_bits())?;
writeln!(out, "wvbx: f64::from_bits(0x{:016x}),", prepared.model.wvbx.to_bits())?;
writeln!(out, "wvbm: f64::from_bits(0x{:016x}),", prepared.model.wvbm.to_bits())?;
writeln!(out, "wxt: f64::from_bits(0x{:016x}),", prepared.model.wxt.to_bits())?;
writeln!(out, "wk1: f64::from_bits(0x{:016x}),", prepared.model.wk1.to_bits())?;
writeln!(out, "wkt1: f64::from_bits(0x{:016x}),", prepared.model.wkt1.to_bits())?;
writeln!(out, "wkt1l: f64::from_bits(0x{:016x}),", prepared.model.wkt1l.to_bits())?;
writeln!(out, "wkt2: f64::from_bits(0x{:016x}),", prepared.model.wkt2.to_bits())?;
writeln!(out, "wk2: f64::from_bits(0x{:016x}),", prepared.model.wk2.to_bits())?;
writeln!(out, "wk3: f64::from_bits(0x{:016x}),", prepared.model.wk3.to_bits())?;
writeln!(out, "wk3b: f64::from_bits(0x{:016x}),", prepared.model.wk3b.to_bits())?;
writeln!(out, "ww0: f64::from_bits(0x{:016x}),", prepared.model.ww0.to_bits())?;
writeln!(out, "wdvtp0: f64::from_bits(0x{:016x}),", prepared.model.wdvtp0.to_bits())?;
writeln!(out, "wdvtp1: f64::from_bits(0x{:016x}),", prepared.model.wdvtp1.to_bits())?;
writeln!(out, "wdvtp2: f64::from_bits(0x{:016x}),", prepared.model.wdvtp2.to_bits())?;
writeln!(out, "wdvtp3: f64::from_bits(0x{:016x}),", prepared.model.wdvtp3.to_bits())?;
writeln!(out, "wdvtp4: f64::from_bits(0x{:016x}),", prepared.model.wdvtp4.to_bits())?;
writeln!(out, "wdvtp5: f64::from_bits(0x{:016x}),", prepared.model.wdvtp5.to_bits())?;
writeln!(out, "wlpe0: f64::from_bits(0x{:016x}),", prepared.model.wlpe0.to_bits())?;
writeln!(out, "wlpeb: f64::from_bits(0x{:016x}),", prepared.model.wlpeb.to_bits())?;
writeln!(out, "wdvt0: f64::from_bits(0x{:016x}),", prepared.model.wdvt0.to_bits())?;
writeln!(out, "wdvt1: f64::from_bits(0x{:016x}),", prepared.model.wdvt1.to_bits())?;
writeln!(out, "wdvt2: f64::from_bits(0x{:016x}),", prepared.model.wdvt2.to_bits())?;
writeln!(out, "wdvt0w: f64::from_bits(0x{:016x}),", prepared.model.wdvt0w.to_bits())?;
writeln!(out, "wdvt1w: f64::from_bits(0x{:016x}),", prepared.model.wdvt1w.to_bits())?;
writeln!(out, "wdvt2w: f64::from_bits(0x{:016x}),", prepared.model.wdvt2w.to_bits())?;
writeln!(out, "wdrout: f64::from_bits(0x{:016x}),", prepared.model.wdrout.to_bits())?;
writeln!(out, "wdsub: f64::from_bits(0x{:016x}),", prepared.model.wdsub.to_bits())?;
writeln!(out, "wvth0: f64::from_bits(0x{:016x}),", prepared.model.wvth0.to_bits())?;
writeln!(out, "wua: f64::from_bits(0x{:016x}),", prepared.model.wua.to_bits())?;
writeln!(out, "wua1: f64::from_bits(0x{:016x}),", prepared.model.wua1.to_bits())?;
writeln!(out, "wub: f64::from_bits(0x{:016x}),", prepared.model.wub.to_bits())?;
writeln!(out, "wub1: f64::from_bits(0x{:016x}),", prepared.model.wub1.to_bits())?;
writeln!(out, "wuc: f64::from_bits(0x{:016x}),", prepared.model.wuc.to_bits())?;
writeln!(out, "wuc1: f64::from_bits(0x{:016x}),", prepared.model.wuc1.to_bits())?;
writeln!(out, "wud: f64::from_bits(0x{:016x}),", prepared.model.wud.to_bits())?;
writeln!(out, "wud1: f64::from_bits(0x{:016x}),", prepared.model.wud1.to_bits())?;
writeln!(out, "wup: f64::from_bits(0x{:016x}),", prepared.model.wup.to_bits())?;
writeln!(out, "wlp: f64::from_bits(0x{:016x}),", prepared.model.wlp.to_bits())?;
writeln!(out, "wu0: f64::from_bits(0x{:016x}),", prepared.model.wu0.to_bits())?;
writeln!(out, "weu: f64::from_bits(0x{:016x}),", prepared.model.weu.to_bits())?;
writeln!(out, "wucs: f64::from_bits(0x{:016x}),", prepared.model.wucs.to_bits())?;
writeln!(out, "wute: f64::from_bits(0x{:016x}),", prepared.model.wute.to_bits())?;
writeln!(out, "wucste: f64::from_bits(0x{:016x}),", prepared.model.wucste.to_bits())?;
writeln!(out, "wvoff: f64::from_bits(0x{:016x}),", prepared.model.wvoff.to_bits())?;
writeln!(out, "wtvoff: f64::from_bits(0x{:016x}),", prepared.model.wtvoff.to_bits())?;
writeln!(out, "wtnfactor: f64::from_bits(0x{:016x}),", prepared.model.wtnfactor.to_bits())?;
writeln!(out, "wteta0: f64::from_bits(0x{:016x}),", prepared.model.wteta0.to_bits())?;
writeln!(out, "wtvoffcv: f64::from_bits(0x{:016x}),", prepared.model.wtvoffcv.to_bits())?;
writeln!(out, "wminv: f64::from_bits(0x{:016x}),", prepared.model.wminv.to_bits())?;
writeln!(out, "wminvcv: f64::from_bits(0x{:016x}),", prepared.model.wminvcv.to_bits())?;
writeln!(out, "wdelta: f64::from_bits(0x{:016x}),", prepared.model.wdelta.to_bits())?;
writeln!(out, "wrdsw: f64::from_bits(0x{:016x}),", prepared.model.wrdsw.to_bits())?;
writeln!(out, "wrsw: f64::from_bits(0x{:016x}),", prepared.model.wrsw.to_bits())?;
writeln!(out, "wrdw: f64::from_bits(0x{:016x}),", prepared.model.wrdw.to_bits())?;
writeln!(out, "wprwg: f64::from_bits(0x{:016x}),", prepared.model.wprwg.to_bits())?;
writeln!(out, "wprwb: f64::from_bits(0x{:016x}),", prepared.model.wprwb.to_bits())?;
writeln!(out, "wprt: f64::from_bits(0x{:016x}),", prepared.model.wprt.to_bits())?;
writeln!(out, "weta0: f64::from_bits(0x{:016x}),", prepared.model.weta0.to_bits())?;
writeln!(out, "wetab: f64::from_bits(0x{:016x}),", prepared.model.wetab.to_bits())?;
writeln!(out, "wpclm: f64::from_bits(0x{:016x}),", prepared.model.wpclm.to_bits())?;
writeln!(out, "wpdibl1: f64::from_bits(0x{:016x}),", prepared.model.wpdibl1.to_bits())?;
writeln!(out, "wpdibl2: f64::from_bits(0x{:016x}),", prepared.model.wpdibl2.to_bits())?;
writeln!(out, "wpdiblb: f64::from_bits(0x{:016x}),", prepared.model.wpdiblb.to_bits())?;
writeln!(out, "wfprout: f64::from_bits(0x{:016x}),", prepared.model.wfprout.to_bits())?;
writeln!(out, "wpdits: f64::from_bits(0x{:016x}),", prepared.model.wpdits.to_bits())?;
writeln!(out, "wpditsd: f64::from_bits(0x{:016x}),", prepared.model.wpditsd.to_bits())?;
writeln!(out, "wpscbe1: f64::from_bits(0x{:016x}),", prepared.model.wpscbe1.to_bits())?;
writeln!(out, "wpscbe2: f64::from_bits(0x{:016x}),", prepared.model.wpscbe2.to_bits())?;
writeln!(out, "wpvag: f64::from_bits(0x{:016x}),", prepared.model.wpvag.to_bits())?;
writeln!(out, "wwr: f64::from_bits(0x{:016x}),", prepared.model.wwr.to_bits())?;
writeln!(out, "wdwg: f64::from_bits(0x{:016x}),", prepared.model.wdwg.to_bits())?;
writeln!(out, "wdwb: f64::from_bits(0x{:016x}),", prepared.model.wdwb.to_bits())?;
writeln!(out, "wb0: f64::from_bits(0x{:016x}),", prepared.model.wb0.to_bits())?;
writeln!(out, "wb1: f64::from_bits(0x{:016x}),", prepared.model.wb1.to_bits())?;
writeln!(out, "walpha0: f64::from_bits(0x{:016x}),", prepared.model.walpha0.to_bits())?;
writeln!(out, "walpha1: f64::from_bits(0x{:016x}),", prepared.model.walpha1.to_bits())?;
writeln!(out, "wbeta0: f64::from_bits(0x{:016x}),", prepared.model.wbeta0.to_bits())?;
writeln!(out, "wvfb: f64::from_bits(0x{:016x}),", prepared.model.wvfb.to_bits())?;
writeln!(out, "wagidl: f64::from_bits(0x{:016x}),", prepared.model.wagidl.to_bits())?;
writeln!(out, "wbgidl: f64::from_bits(0x{:016x}),", prepared.model.wbgidl.to_bits())?;
writeln!(out, "wcgidl: f64::from_bits(0x{:016x}),", prepared.model.wcgidl.to_bits())?;
writeln!(out, "wegidl: f64::from_bits(0x{:016x}),", prepared.model.wegidl.to_bits())?;
writeln!(out, "wfgidl: f64::from_bits(0x{:016x}),", prepared.model.wfgidl.to_bits())?;
writeln!(out, "wkgidl: f64::from_bits(0x{:016x}),", prepared.model.wkgidl.to_bits())?;
writeln!(out, "wrgidl: f64::from_bits(0x{:016x}),", prepared.model.wrgidl.to_bits())?;
writeln!(out, "wagisl: f64::from_bits(0x{:016x}),", prepared.model.wagisl.to_bits())?;
writeln!(out, "wbgisl: f64::from_bits(0x{:016x}),", prepared.model.wbgisl.to_bits())?;
writeln!(out, "wcgisl: f64::from_bits(0x{:016x}),", prepared.model.wcgisl.to_bits())?;
writeln!(out, "wegisl: f64::from_bits(0x{:016x}),", prepared.model.wegisl.to_bits())?;
writeln!(out, "wfgisl: f64::from_bits(0x{:016x}),", prepared.model.wfgisl.to_bits())?;
writeln!(out, "wkgisl: f64::from_bits(0x{:016x}),", prepared.model.wkgisl.to_bits())?;
writeln!(out, "wrgisl: f64::from_bits(0x{:016x}),", prepared.model.wrgisl.to_bits())?;
writeln!(out, "waigc: f64::from_bits(0x{:016x}),", prepared.model.waigc.to_bits())?;
writeln!(out, "wbigc: f64::from_bits(0x{:016x}),", prepared.model.wbigc.to_bits())?;
writeln!(out, "wcigc: f64::from_bits(0x{:016x}),", prepared.model.wcigc.to_bits())?;
writeln!(out, "waigsd: f64::from_bits(0x{:016x}),", prepared.model.waigsd.to_bits())?;
writeln!(out, "wbigsd: f64::from_bits(0x{:016x}),", prepared.model.wbigsd.to_bits())?;
writeln!(out, "wcigsd: f64::from_bits(0x{:016x}),", prepared.model.wcigsd.to_bits())?;
writeln!(out, "waigs: f64::from_bits(0x{:016x}),", prepared.model.waigs.to_bits())?;
writeln!(out, "wbigs: f64::from_bits(0x{:016x}),", prepared.model.wbigs.to_bits())?;
writeln!(out, "wcigs: f64::from_bits(0x{:016x}),", prepared.model.wcigs.to_bits())?;
writeln!(out, "waigd: f64::from_bits(0x{:016x}),", prepared.model.waigd.to_bits())?;
writeln!(out, "wbigd: f64::from_bits(0x{:016x}),", prepared.model.wbigd.to_bits())?;
writeln!(out, "wcigd: f64::from_bits(0x{:016x}),", prepared.model.wcigd.to_bits())?;
writeln!(out, "waigbacc: f64::from_bits(0x{:016x}),", prepared.model.waigbacc.to_bits())?;
writeln!(out, "wbigbacc: f64::from_bits(0x{:016x}),", prepared.model.wbigbacc.to_bits())?;
writeln!(out, "wcigbacc: f64::from_bits(0x{:016x}),", prepared.model.wcigbacc.to_bits())?;
writeln!(out, "waigbinv: f64::from_bits(0x{:016x}),", prepared.model.waigbinv.to_bits())?;
writeln!(out, "wbigbinv: f64::from_bits(0x{:016x}),", prepared.model.wbigbinv.to_bits())?;
writeln!(out, "wcigbinv: f64::from_bits(0x{:016x}),", prepared.model.wcigbinv.to_bits())?;
writeln!(out, "wnigc: f64::from_bits(0x{:016x}),", prepared.model.wnigc.to_bits())?;
writeln!(out, "wnigbacc: f64::from_bits(0x{:016x}),", prepared.model.wnigbacc.to_bits())?;
writeln!(out, "wnigbinv: f64::from_bits(0x{:016x}),", prepared.model.wnigbinv.to_bits())?;
writeln!(out, "wntox: f64::from_bits(0x{:016x}),", prepared.model.wntox.to_bits())?;
writeln!(out, "weigbinv: f64::from_bits(0x{:016x}),", prepared.model.weigbinv.to_bits())?;
writeln!(out, "wpigcd: f64::from_bits(0x{:016x}),", prepared.model.wpigcd.to_bits())?;
writeln!(out, "wpoxedge: f64::from_bits(0x{:016x}),", prepared.model.wpoxedge.to_bits())?;
writeln!(out, "wxrcrg1: f64::from_bits(0x{:016x}),", prepared.model.wxrcrg1.to_bits())?;
writeln!(out, "wxrcrg2: f64::from_bits(0x{:016x}),", prepared.model.wxrcrg2.to_bits())?;
writeln!(out, "wlambda: f64::from_bits(0x{:016x}),", prepared.model.wlambda.to_bits())?;
writeln!(out, "wvtl: f64::from_bits(0x{:016x}),", prepared.model.wvtl.to_bits())?;
writeln!(out, "wxn: f64::from_bits(0x{:016x}),", prepared.model.wxn.to_bits())?;
writeln!(out, "wvfbsdoff: f64::from_bits(0x{:016x}),", prepared.model.wvfbsdoff.to_bits())?;
writeln!(out, "wtvfbsdoff: f64::from_bits(0x{:016x}),", prepared.model.wtvfbsdoff.to_bits())?;
writeln!(out, "wcgsl: f64::from_bits(0x{:016x}),", prepared.model.wcgsl.to_bits())?;
writeln!(out, "wcgdl: f64::from_bits(0x{:016x}),", prepared.model.wcgdl.to_bits())?;
writeln!(out, "wckappas: f64::from_bits(0x{:016x}),", prepared.model.wckappas.to_bits())?;
writeln!(out, "wckappad: f64::from_bits(0x{:016x}),", prepared.model.wckappad.to_bits())?;
writeln!(out, "wcf: f64::from_bits(0x{:016x}),", prepared.model.wcf.to_bits())?;
writeln!(out, "wclc: f64::from_bits(0x{:016x}),", prepared.model.wclc.to_bits())?;
writeln!(out, "wcle: f64::from_bits(0x{:016x}),", prepared.model.wcle.to_bits())?;
writeln!(out, "wvfbcv: f64::from_bits(0x{:016x}),", prepared.model.wvfbcv.to_bits())?;
writeln!(out, "wnoff: f64::from_bits(0x{:016x}),", prepared.model.wnoff.to_bits())?;
writeln!(out, "wvoffcv: f64::from_bits(0x{:016x}),", prepared.model.wvoffcv.to_bits())?;
writeln!(out, "wacde: f64::from_bits(0x{:016x}),", prepared.model.wacde.to_bits())?;
writeln!(out, "wmoin: f64::from_bits(0x{:016x}),", prepared.model.wmoin.to_bits())?;
writeln!(out, "pcdsc: f64::from_bits(0x{:016x}),", prepared.model.pcdsc.to_bits())?;
writeln!(out, "pcdscb: f64::from_bits(0x{:016x}),", prepared.model.pcdscb.to_bits())?;
writeln!(out, "pcdscd: f64::from_bits(0x{:016x}),", prepared.model.pcdscd.to_bits())?;
writeln!(out, "pcit: f64::from_bits(0x{:016x}),", prepared.model.pcit.to_bits())?;
writeln!(out, "pnfactor: f64::from_bits(0x{:016x}),", prepared.model.pnfactor.to_bits())?;
writeln!(out, "pxj: f64::from_bits(0x{:016x}),", prepared.model.pxj.to_bits())?;
writeln!(out, "pvsat: f64::from_bits(0x{:016x}),", prepared.model.pvsat.to_bits())?;
writeln!(out, "pat: f64::from_bits(0x{:016x}),", prepared.model.pat.to_bits())?;
writeln!(out, "pa0: f64::from_bits(0x{:016x}),", prepared.model.pa0.to_bits())?;
writeln!(out, "pags: f64::from_bits(0x{:016x}),", prepared.model.pags.to_bits())?;
writeln!(out, "pa1: f64::from_bits(0x{:016x}),", prepared.model.pa1.to_bits())?;
writeln!(out, "pa2: f64::from_bits(0x{:016x}),", prepared.model.pa2.to_bits())?;
writeln!(out, "pketa: f64::from_bits(0x{:016x}),", prepared.model.pketa.to_bits())?;
writeln!(out, "pketac: f64::from_bits(0x{:016x}),", prepared.model.pketac.to_bits())?;
writeln!(out, "pnsub: f64::from_bits(0x{:016x}),", prepared.model.pnsub.to_bits())?;
writeln!(out, "pndep: f64::from_bits(0x{:016x}),", prepared.model.pndep.to_bits())?;
writeln!(out, "pnsd: f64::from_bits(0x{:016x}),", prepared.model.pnsd.to_bits())?;
writeln!(out, "pphin: f64::from_bits(0x{:016x}),", prepared.model.pphin.to_bits())?;
writeln!(out, "pngate: f64::from_bits(0x{:016x}),", prepared.model.pngate.to_bits())?;
writeln!(out, "pgamma1: f64::from_bits(0x{:016x}),", prepared.model.pgamma1.to_bits())?;
writeln!(out, "pgamma2: f64::from_bits(0x{:016x}),", prepared.model.pgamma2.to_bits())?;
writeln!(out, "pvbx: f64::from_bits(0x{:016x}),", prepared.model.pvbx.to_bits())?;
writeln!(out, "pvbm: f64::from_bits(0x{:016x}),", prepared.model.pvbm.to_bits())?;
writeln!(out, "pxt: f64::from_bits(0x{:016x}),", prepared.model.pxt.to_bits())?;
writeln!(out, "pk1: f64::from_bits(0x{:016x}),", prepared.model.pk1.to_bits())?;
writeln!(out, "pkt1: f64::from_bits(0x{:016x}),", prepared.model.pkt1.to_bits())?;
writeln!(out, "pkt1l: f64::from_bits(0x{:016x}),", prepared.model.pkt1l.to_bits())?;
writeln!(out, "pkt2: f64::from_bits(0x{:016x}),", prepared.model.pkt2.to_bits())?;
writeln!(out, "pk2: f64::from_bits(0x{:016x}),", prepared.model.pk2.to_bits())?;
writeln!(out, "pk3: f64::from_bits(0x{:016x}),", prepared.model.pk3.to_bits())?;
writeln!(out, "pk3b: f64::from_bits(0x{:016x}),", prepared.model.pk3b.to_bits())?;
writeln!(out, "pw0: f64::from_bits(0x{:016x}),", prepared.model.pw0.to_bits())?;
writeln!(out, "pdvtp0: f64::from_bits(0x{:016x}),", prepared.model.pdvtp0.to_bits())?;
writeln!(out, "pdvtp1: f64::from_bits(0x{:016x}),", prepared.model.pdvtp1.to_bits())?;
writeln!(out, "pdvtp2: f64::from_bits(0x{:016x}),", prepared.model.pdvtp2.to_bits())?;
writeln!(out, "pdvtp3: f64::from_bits(0x{:016x}),", prepared.model.pdvtp3.to_bits())?;
writeln!(out, "pdvtp4: f64::from_bits(0x{:016x}),", prepared.model.pdvtp4.to_bits())?;
writeln!(out, "pdvtp5: f64::from_bits(0x{:016x}),", prepared.model.pdvtp5.to_bits())?;
writeln!(out, "plpe0: f64::from_bits(0x{:016x}),", prepared.model.plpe0.to_bits())?;
writeln!(out, "plpeb: f64::from_bits(0x{:016x}),", prepared.model.plpeb.to_bits())?;
writeln!(out, "pdvt0: f64::from_bits(0x{:016x}),", prepared.model.pdvt0.to_bits())?;
writeln!(out, "pdvt1: f64::from_bits(0x{:016x}),", prepared.model.pdvt1.to_bits())?;
writeln!(out, "pdvt2: f64::from_bits(0x{:016x}),", prepared.model.pdvt2.to_bits())?;
writeln!(out, "pdvt0w: f64::from_bits(0x{:016x}),", prepared.model.pdvt0w.to_bits())?;
writeln!(out, "pdvt1w: f64::from_bits(0x{:016x}),", prepared.model.pdvt1w.to_bits())?;
writeln!(out, "pdvt2w: f64::from_bits(0x{:016x}),", prepared.model.pdvt2w.to_bits())?;
writeln!(out, "pdrout: f64::from_bits(0x{:016x}),", prepared.model.pdrout.to_bits())?;
writeln!(out, "pdsub: f64::from_bits(0x{:016x}),", prepared.model.pdsub.to_bits())?;
writeln!(out, "pvth0: f64::from_bits(0x{:016x}),", prepared.model.pvth0.to_bits())?;
writeln!(out, "pua: f64::from_bits(0x{:016x}),", prepared.model.pua.to_bits())?;
writeln!(out, "pua1: f64::from_bits(0x{:016x}),", prepared.model.pua1.to_bits())?;
writeln!(out, "r#pub: f64::from_bits(0x{:016x}),", prepared.model.r#pub.to_bits())?;
writeln!(out, "pub1: f64::from_bits(0x{:016x}),", prepared.model.pub1.to_bits())?;
writeln!(out, "puc: f64::from_bits(0x{:016x}),", prepared.model.puc.to_bits())?;
writeln!(out, "puc1: f64::from_bits(0x{:016x}),", prepared.model.puc1.to_bits())?;
writeln!(out, "pud: f64::from_bits(0x{:016x}),", prepared.model.pud.to_bits())?;
writeln!(out, "pud1: f64::from_bits(0x{:016x}),", prepared.model.pud1.to_bits())?;
writeln!(out, "pup: f64::from_bits(0x{:016x}),", prepared.model.pup.to_bits())?;
writeln!(out, "plp: f64::from_bits(0x{:016x}),", prepared.model.plp.to_bits())?;
writeln!(out, "pu0: f64::from_bits(0x{:016x}),", prepared.model.pu0.to_bits())?;
writeln!(out, "peu: f64::from_bits(0x{:016x}),", prepared.model.peu.to_bits())?;
writeln!(out, "pucs: f64::from_bits(0x{:016x}),", prepared.model.pucs.to_bits())?;
writeln!(out, "pute: f64::from_bits(0x{:016x}),", prepared.model.pute.to_bits())?;
writeln!(out, "pucste: f64::from_bits(0x{:016x}),", prepared.model.pucste.to_bits())?;
writeln!(out, "pvoff: f64::from_bits(0x{:016x}),", prepared.model.pvoff.to_bits())?;
writeln!(out, "ptvoff: f64::from_bits(0x{:016x}),", prepared.model.ptvoff.to_bits())?;
writeln!(out, "ptnfactor: f64::from_bits(0x{:016x}),", prepared.model.ptnfactor.to_bits())?;
writeln!(out, "pteta0: f64::from_bits(0x{:016x}),", prepared.model.pteta0.to_bits())?;
writeln!(out, "ptvoffcv: f64::from_bits(0x{:016x}),", prepared.model.ptvoffcv.to_bits())?;
writeln!(out, "pminv: f64::from_bits(0x{:016x}),", prepared.model.pminv.to_bits())?;
writeln!(out, "pminvcv: f64::from_bits(0x{:016x}),", prepared.model.pminvcv.to_bits())?;
writeln!(out, "pdelta: f64::from_bits(0x{:016x}),", prepared.model.pdelta.to_bits())?;
writeln!(out, "prdsw: f64::from_bits(0x{:016x}),", prepared.model.prdsw.to_bits())?;
writeln!(out, "prsw: f64::from_bits(0x{:016x}),", prepared.model.prsw.to_bits())?;
writeln!(out, "prdw: f64::from_bits(0x{:016x}),", prepared.model.prdw.to_bits())?;
writeln!(out, "pprwg: f64::from_bits(0x{:016x}),", prepared.model.pprwg.to_bits())?;
writeln!(out, "pprwb: f64::from_bits(0x{:016x}),", prepared.model.pprwb.to_bits())?;
writeln!(out, "pprt: f64::from_bits(0x{:016x}),", prepared.model.pprt.to_bits())?;
writeln!(out, "peta0: f64::from_bits(0x{:016x}),", prepared.model.peta0.to_bits())?;
writeln!(out, "petab: f64::from_bits(0x{:016x}),", prepared.model.petab.to_bits())?;
writeln!(out, "ppclm: f64::from_bits(0x{:016x}),", prepared.model.ppclm.to_bits())?;
writeln!(out, "ppdibl1: f64::from_bits(0x{:016x}),", prepared.model.ppdibl1.to_bits())?;
writeln!(out, "ppdibl2: f64::from_bits(0x{:016x}),", prepared.model.ppdibl2.to_bits())?;
writeln!(out, "ppdiblb: f64::from_bits(0x{:016x}),", prepared.model.ppdiblb.to_bits())?;
writeln!(out, "pfprout: f64::from_bits(0x{:016x}),", prepared.model.pfprout.to_bits())?;
writeln!(out, "ppdits: f64::from_bits(0x{:016x}),", prepared.model.ppdits.to_bits())?;
writeln!(out, "ppditsd: f64::from_bits(0x{:016x}),", prepared.model.ppditsd.to_bits())?;
writeln!(out, "ppscbe1: f64::from_bits(0x{:016x}),", prepared.model.ppscbe1.to_bits())?;
writeln!(out, "ppscbe2: f64::from_bits(0x{:016x}),", prepared.model.ppscbe2.to_bits())?;
writeln!(out, "ppvag: f64::from_bits(0x{:016x}),", prepared.model.ppvag.to_bits())?;
writeln!(out, "pwr: f64::from_bits(0x{:016x}),", prepared.model.pwr.to_bits())?;
writeln!(out, "pdwg: f64::from_bits(0x{:016x}),", prepared.model.pdwg.to_bits())?;
writeln!(out, "pdwb: f64::from_bits(0x{:016x}),", prepared.model.pdwb.to_bits())?;
writeln!(out, "pb0: f64::from_bits(0x{:016x}),", prepared.model.pb0.to_bits())?;
writeln!(out, "pb1: f64::from_bits(0x{:016x}),", prepared.model.pb1.to_bits())?;
writeln!(out, "palpha0: f64::from_bits(0x{:016x}),", prepared.model.palpha0.to_bits())?;
writeln!(out, "palpha1: f64::from_bits(0x{:016x}),", prepared.model.palpha1.to_bits())?;
writeln!(out, "pbeta0: f64::from_bits(0x{:016x}),", prepared.model.pbeta0.to_bits())?;
writeln!(out, "pvfb: f64::from_bits(0x{:016x}),", prepared.model.pvfb.to_bits())?;
writeln!(out, "pagidl: f64::from_bits(0x{:016x}),", prepared.model.pagidl.to_bits())?;
writeln!(out, "pbgidl: f64::from_bits(0x{:016x}),", prepared.model.pbgidl.to_bits())?;
writeln!(out, "pcgidl: f64::from_bits(0x{:016x}),", prepared.model.pcgidl.to_bits())?;
writeln!(out, "pegidl: f64::from_bits(0x{:016x}),", prepared.model.pegidl.to_bits())?;
writeln!(out, "pfgidl: f64::from_bits(0x{:016x}),", prepared.model.pfgidl.to_bits())?;
writeln!(out, "pkgidl: f64::from_bits(0x{:016x}),", prepared.model.pkgidl.to_bits())?;
writeln!(out, "prgidl: f64::from_bits(0x{:016x}),", prepared.model.prgidl.to_bits())?;
writeln!(out, "pagisl: f64::from_bits(0x{:016x}),", prepared.model.pagisl.to_bits())?;
writeln!(out, "pbgisl: f64::from_bits(0x{:016x}),", prepared.model.pbgisl.to_bits())?;
writeln!(out, "pcgisl: f64::from_bits(0x{:016x}),", prepared.model.pcgisl.to_bits())?;
writeln!(out, "pegisl: f64::from_bits(0x{:016x}),", prepared.model.pegisl.to_bits())?;
writeln!(out, "pfgisl: f64::from_bits(0x{:016x}),", prepared.model.pfgisl.to_bits())?;
writeln!(out, "pkgisl: f64::from_bits(0x{:016x}),", prepared.model.pkgisl.to_bits())?;
writeln!(out, "prgisl: f64::from_bits(0x{:016x}),", prepared.model.prgisl.to_bits())?;
writeln!(out, "paigc: f64::from_bits(0x{:016x}),", prepared.model.paigc.to_bits())?;
writeln!(out, "pbigc: f64::from_bits(0x{:016x}),", prepared.model.pbigc.to_bits())?;
writeln!(out, "pcigc: f64::from_bits(0x{:016x}),", prepared.model.pcigc.to_bits())?;
writeln!(out, "paigsd: f64::from_bits(0x{:016x}),", prepared.model.paigsd.to_bits())?;
writeln!(out, "pbigsd: f64::from_bits(0x{:016x}),", prepared.model.pbigsd.to_bits())?;
writeln!(out, "pcigsd: f64::from_bits(0x{:016x}),", prepared.model.pcigsd.to_bits())?;
writeln!(out, "paigs: f64::from_bits(0x{:016x}),", prepared.model.paigs.to_bits())?;
writeln!(out, "pbigs: f64::from_bits(0x{:016x}),", prepared.model.pbigs.to_bits())?;
writeln!(out, "pcigs: f64::from_bits(0x{:016x}),", prepared.model.pcigs.to_bits())?;
writeln!(out, "paigd: f64::from_bits(0x{:016x}),", prepared.model.paigd.to_bits())?;
writeln!(out, "pbigd: f64::from_bits(0x{:016x}),", prepared.model.pbigd.to_bits())?;
writeln!(out, "pcigd: f64::from_bits(0x{:016x}),", prepared.model.pcigd.to_bits())?;
writeln!(out, "paigbacc: f64::from_bits(0x{:016x}),", prepared.model.paigbacc.to_bits())?;
writeln!(out, "pbigbacc: f64::from_bits(0x{:016x}),", prepared.model.pbigbacc.to_bits())?;
writeln!(out, "pcigbacc: f64::from_bits(0x{:016x}),", prepared.model.pcigbacc.to_bits())?;
writeln!(out, "paigbinv: f64::from_bits(0x{:016x}),", prepared.model.paigbinv.to_bits())?;
writeln!(out, "pbigbinv: f64::from_bits(0x{:016x}),", prepared.model.pbigbinv.to_bits())?;
writeln!(out, "pcigbinv: f64::from_bits(0x{:016x}),", prepared.model.pcigbinv.to_bits())?;
writeln!(out, "pnigc: f64::from_bits(0x{:016x}),", prepared.model.pnigc.to_bits())?;
writeln!(out, "pnigbacc: f64::from_bits(0x{:016x}),", prepared.model.pnigbacc.to_bits())?;
writeln!(out, "pnigbinv: f64::from_bits(0x{:016x}),", prepared.model.pnigbinv.to_bits())?;
writeln!(out, "pntox: f64::from_bits(0x{:016x}),", prepared.model.pntox.to_bits())?;
writeln!(out, "peigbinv: f64::from_bits(0x{:016x}),", prepared.model.peigbinv.to_bits())?;
writeln!(out, "ppigcd: f64::from_bits(0x{:016x}),", prepared.model.ppigcd.to_bits())?;
writeln!(out, "ppoxedge: f64::from_bits(0x{:016x}),", prepared.model.ppoxedge.to_bits())?;
writeln!(out, "pxrcrg1: f64::from_bits(0x{:016x}),", prepared.model.pxrcrg1.to_bits())?;
writeln!(out, "pxrcrg2: f64::from_bits(0x{:016x}),", prepared.model.pxrcrg2.to_bits())?;
writeln!(out, "plambda: f64::from_bits(0x{:016x}),", prepared.model.plambda.to_bits())?;
writeln!(out, "pvtl: f64::from_bits(0x{:016x}),", prepared.model.pvtl.to_bits())?;
writeln!(out, "pxn: f64::from_bits(0x{:016x}),", prepared.model.pxn.to_bits())?;
writeln!(out, "pvfbsdoff: f64::from_bits(0x{:016x}),", prepared.model.pvfbsdoff.to_bits())?;
writeln!(out, "ptvfbsdoff: f64::from_bits(0x{:016x}),", prepared.model.ptvfbsdoff.to_bits())?;
writeln!(out, "pcgsl: f64::from_bits(0x{:016x}),", prepared.model.pcgsl.to_bits())?;
writeln!(out, "pcgdl: f64::from_bits(0x{:016x}),", prepared.model.pcgdl.to_bits())?;
writeln!(out, "pckappas: f64::from_bits(0x{:016x}),", prepared.model.pckappas.to_bits())?;
writeln!(out, "pckappad: f64::from_bits(0x{:016x}),", prepared.model.pckappad.to_bits())?;
writeln!(out, "pcf: f64::from_bits(0x{:016x}),", prepared.model.pcf.to_bits())?;
writeln!(out, "pclc: f64::from_bits(0x{:016x}),", prepared.model.pclc.to_bits())?;
writeln!(out, "pcle: f64::from_bits(0x{:016x}),", prepared.model.pcle.to_bits())?;
writeln!(out, "pvfbcv: f64::from_bits(0x{:016x}),", prepared.model.pvfbcv.to_bits())?;
writeln!(out, "pnoff: f64::from_bits(0x{:016x}),", prepared.model.pnoff.to_bits())?;
writeln!(out, "pvoffcv: f64::from_bits(0x{:016x}),", prepared.model.pvoffcv.to_bits())?;
writeln!(out, "pacde: f64::from_bits(0x{:016x}),", prepared.model.pacde.to_bits())?;
writeln!(out, "pmoin: f64::from_bits(0x{:016x}),", prepared.model.pmoin.to_bits())?;
writeln!(out, "tnom: f64::from_bits(0x{:016x}),", prepared.model.tnom.to_bits())?;
writeln!(out, "cgso: f64::from_bits(0x{:016x}),", prepared.model.cgso.to_bits())?;
writeln!(out, "cgdo: f64::from_bits(0x{:016x}),", prepared.model.cgdo.to_bits())?;
writeln!(out, "cgbo: f64::from_bits(0x{:016x}),", prepared.model.cgbo.to_bits())?;
writeln!(out, "xpart: f64::from_bits(0x{:016x}),", prepared.model.xpart.to_bits())?;
writeln!(out, "cFringOut: f64::from_bits(0x{:016x}),", prepared.model.cFringOut.to_bits())?;
writeln!(out, "cFringMax: f64::from_bits(0x{:016x}),", prepared.model.cFringMax.to_bits())?;
writeln!(out, "sheetResistance: f64::from_bits(0x{:016x}),", prepared.model.sheetResistance.to_bits())?;
writeln!(out, "SjctSatCurDensity: f64::from_bits(0x{:016x}),", prepared.model.SjctSatCurDensity.to_bits())?;
writeln!(out, "DjctSatCurDensity: f64::from_bits(0x{:016x}),", prepared.model.DjctSatCurDensity.to_bits())?;
writeln!(out, "SjctSidewallSatCurDensity: f64::from_bits(0x{:016x}),", prepared.model.SjctSidewallSatCurDensity.to_bits())?;
writeln!(out, "DjctSidewallSatCurDensity: f64::from_bits(0x{:016x}),", prepared.model.DjctSidewallSatCurDensity.to_bits())?;
writeln!(out, "SjctGateSidewallSatCurDensity: f64::from_bits(0x{:016x}),", prepared.model.SjctGateSidewallSatCurDensity.to_bits())?;
writeln!(out, "DjctGateSidewallSatCurDensity: f64::from_bits(0x{:016x}),", prepared.model.DjctGateSidewallSatCurDensity.to_bits())?;
writeln!(out, "SbulkJctPotential: f64::from_bits(0x{:016x}),", prepared.model.SbulkJctPotential.to_bits())?;
writeln!(out, "DbulkJctPotential: f64::from_bits(0x{:016x}),", prepared.model.DbulkJctPotential.to_bits())?;
writeln!(out, "SbulkJctBotGradingCoeff: f64::from_bits(0x{:016x}),", prepared.model.SbulkJctBotGradingCoeff.to_bits())?;
writeln!(out, "DbulkJctBotGradingCoeff: f64::from_bits(0x{:016x}),", prepared.model.DbulkJctBotGradingCoeff.to_bits())?;
writeln!(out, "SbulkJctSideGradingCoeff: f64::from_bits(0x{:016x}),", prepared.model.SbulkJctSideGradingCoeff.to_bits())?;
writeln!(out, "DbulkJctSideGradingCoeff: f64::from_bits(0x{:016x}),", prepared.model.DbulkJctSideGradingCoeff.to_bits())?;
writeln!(out, "SbulkJctGateSideGradingCoeff: f64::from_bits(0x{:016x}),", prepared.model.SbulkJctGateSideGradingCoeff.to_bits())?;
writeln!(out, "DbulkJctGateSideGradingCoeff: f64::from_bits(0x{:016x}),", prepared.model.DbulkJctGateSideGradingCoeff.to_bits())?;
writeln!(out, "SsidewallJctPotential: f64::from_bits(0x{:016x}),", prepared.model.SsidewallJctPotential.to_bits())?;
writeln!(out, "DsidewallJctPotential: f64::from_bits(0x{:016x}),", prepared.model.DsidewallJctPotential.to_bits())?;
writeln!(out, "SGatesidewallJctPotential: f64::from_bits(0x{:016x}),", prepared.model.SGatesidewallJctPotential.to_bits())?;
writeln!(out, "DGatesidewallJctPotential: f64::from_bits(0x{:016x}),", prepared.model.DGatesidewallJctPotential.to_bits())?;
writeln!(out, "SunitAreaJctCap: f64::from_bits(0x{:016x}),", prepared.model.SunitAreaJctCap.to_bits())?;
writeln!(out, "DunitAreaJctCap: f64::from_bits(0x{:016x}),", prepared.model.DunitAreaJctCap.to_bits())?;
writeln!(out, "SunitLengthSidewallJctCap: f64::from_bits(0x{:016x}),", prepared.model.SunitLengthSidewallJctCap.to_bits())?;
writeln!(out, "DunitLengthSidewallJctCap: f64::from_bits(0x{:016x}),", prepared.model.DunitLengthSidewallJctCap.to_bits())?;
writeln!(out, "SunitLengthGateSidewallJctCap: f64::from_bits(0x{:016x}),", prepared.model.SunitLengthGateSidewallJctCap.to_bits())?;
writeln!(out, "DunitLengthGateSidewallJctCap: f64::from_bits(0x{:016x}),", prepared.model.DunitLengthGateSidewallJctCap.to_bits())?;
writeln!(out, "SjctEmissionCoeff: f64::from_bits(0x{:016x}),", prepared.model.SjctEmissionCoeff.to_bits())?;
writeln!(out, "DjctEmissionCoeff: f64::from_bits(0x{:016x}),", prepared.model.DjctEmissionCoeff.to_bits())?;
writeln!(out, "SjctTempExponent: f64::from_bits(0x{:016x}),", prepared.model.SjctTempExponent.to_bits())?;
writeln!(out, "DjctTempExponent: f64::from_bits(0x{:016x}),", prepared.model.DjctTempExponent.to_bits())?;
writeln!(out, "njtsstemp: f64::from_bits(0x{:016x}),", prepared.model.njtsstemp.to_bits())?;
writeln!(out, "njtsswstemp: f64::from_bits(0x{:016x}),", prepared.model.njtsswstemp.to_bits())?;
writeln!(out, "njtsswgstemp: f64::from_bits(0x{:016x}),", prepared.model.njtsswgstemp.to_bits())?;
writeln!(out, "njtsdtemp: f64::from_bits(0x{:016x}),", prepared.model.njtsdtemp.to_bits())?;
writeln!(out, "njtsswdtemp: f64::from_bits(0x{:016x}),", prepared.model.njtsswdtemp.to_bits())?;
writeln!(out, "njtsswgdtemp: f64::from_bits(0x{:016x}),", prepared.model.njtsswgdtemp.to_bits())?;
writeln!(out, "Lint: f64::from_bits(0x{:016x}),", prepared.model.Lint.to_bits())?;
writeln!(out, "Ll: f64::from_bits(0x{:016x}),", prepared.model.Ll.to_bits())?;
writeln!(out, "Llc: f64::from_bits(0x{:016x}),", prepared.model.Llc.to_bits())?;
writeln!(out, "Lln: f64::from_bits(0x{:016x}),", prepared.model.Lln.to_bits())?;
writeln!(out, "Lw: f64::from_bits(0x{:016x}),", prepared.model.Lw.to_bits())?;
writeln!(out, "Lwc: f64::from_bits(0x{:016x}),", prepared.model.Lwc.to_bits())?;
writeln!(out, "Lwn: f64::from_bits(0x{:016x}),", prepared.model.Lwn.to_bits())?;
writeln!(out, "Lwl: f64::from_bits(0x{:016x}),", prepared.model.Lwl.to_bits())?;
writeln!(out, "Lwlc: f64::from_bits(0x{:016x}),", prepared.model.Lwlc.to_bits())?;
writeln!(out, "Lmin: f64::from_bits(0x{:016x}),", prepared.model.Lmin.to_bits())?;
writeln!(out, "Lmax: f64::from_bits(0x{:016x}),", prepared.model.Lmax.to_bits())?;
writeln!(out, "Wint: f64::from_bits(0x{:016x}),", prepared.model.Wint.to_bits())?;
writeln!(out, "Wl: f64::from_bits(0x{:016x}),", prepared.model.Wl.to_bits())?;
writeln!(out, "Wlc: f64::from_bits(0x{:016x}),", prepared.model.Wlc.to_bits())?;
writeln!(out, "Wln: f64::from_bits(0x{:016x}),", prepared.model.Wln.to_bits())?;
writeln!(out, "Ww: f64::from_bits(0x{:016x}),", prepared.model.Ww.to_bits())?;
writeln!(out, "Wwc: f64::from_bits(0x{:016x}),", prepared.model.Wwc.to_bits())?;
writeln!(out, "Wwn: f64::from_bits(0x{:016x}),", prepared.model.Wwn.to_bits())?;
writeln!(out, "Wwl: f64::from_bits(0x{:016x}),", prepared.model.Wwl.to_bits())?;
writeln!(out, "Wwlc: f64::from_bits(0x{:016x}),", prepared.model.Wwlc.to_bits())?;
writeln!(out, "Wmin: f64::from_bits(0x{:016x}),", prepared.model.Wmin.to_bits())?;
writeln!(out, "Wmax: f64::from_bits(0x{:016x}),", prepared.model.Wmax.to_bits())?;
writeln!(out, "saref: f64::from_bits(0x{:016x}),", prepared.model.saref.to_bits())?;
writeln!(out, "sbref: f64::from_bits(0x{:016x}),", prepared.model.sbref.to_bits())?;
writeln!(out, "wlod: f64::from_bits(0x{:016x}),", prepared.model.wlod.to_bits())?;
writeln!(out, "ku0: f64::from_bits(0x{:016x}),", prepared.model.ku0.to_bits())?;
writeln!(out, "kvsat: f64::from_bits(0x{:016x}),", prepared.model.kvsat.to_bits())?;
writeln!(out, "kvth0: f64::from_bits(0x{:016x}),", prepared.model.kvth0.to_bits())?;
writeln!(out, "tku0: f64::from_bits(0x{:016x}),", prepared.model.tku0.to_bits())?;
writeln!(out, "llodku0: f64::from_bits(0x{:016x}),", prepared.model.llodku0.to_bits())?;
writeln!(out, "wlodku0: f64::from_bits(0x{:016x}),", prepared.model.wlodku0.to_bits())?;
writeln!(out, "llodvth: f64::from_bits(0x{:016x}),", prepared.model.llodvth.to_bits())?;
writeln!(out, "wlodvth: f64::from_bits(0x{:016x}),", prepared.model.wlodvth.to_bits())?;
writeln!(out, "lku0: f64::from_bits(0x{:016x}),", prepared.model.lku0.to_bits())?;
writeln!(out, "wku0: f64::from_bits(0x{:016x}),", prepared.model.wku0.to_bits())?;
writeln!(out, "pku0: f64::from_bits(0x{:016x}),", prepared.model.pku0.to_bits())?;
writeln!(out, "lkvth0: f64::from_bits(0x{:016x}),", prepared.model.lkvth0.to_bits())?;
writeln!(out, "wkvth0: f64::from_bits(0x{:016x}),", prepared.model.wkvth0.to_bits())?;
writeln!(out, "pkvth0: f64::from_bits(0x{:016x}),", prepared.model.pkvth0.to_bits())?;
writeln!(out, "stk2: f64::from_bits(0x{:016x}),", prepared.model.stk2.to_bits())?;
writeln!(out, "lodk2: f64::from_bits(0x{:016x}),", prepared.model.lodk2.to_bits())?;
writeln!(out, "steta0: f64::from_bits(0x{:016x}),", prepared.model.steta0.to_bits())?;
writeln!(out, "lodeta0: f64::from_bits(0x{:016x}),", prepared.model.lodeta0.to_bits())?;
writeln!(out, "web: f64::from_bits(0x{:016x}),", prepared.model.web.to_bits())?;
writeln!(out, "wec: f64::from_bits(0x{:016x}),", prepared.model.wec.to_bits())?;
writeln!(out, "kvth0we: f64::from_bits(0x{:016x}),", prepared.model.kvth0we.to_bits())?;
writeln!(out, "k2we: f64::from_bits(0x{:016x}),", prepared.model.k2we.to_bits())?;
writeln!(out, "ku0we: f64::from_bits(0x{:016x}),", prepared.model.ku0we.to_bits())?;
writeln!(out, "scref: f64::from_bits(0x{:016x}),", prepared.model.scref.to_bits())?;
writeln!(out, "wpemod: f64::from_bits(0x{:016x}),", prepared.model.wpemod.to_bits())?;
writeln!(out, "lkvth0we: f64::from_bits(0x{:016x}),", prepared.model.lkvth0we.to_bits())?;
writeln!(out, "lk2we: f64::from_bits(0x{:016x}),", prepared.model.lk2we.to_bits())?;
writeln!(out, "lku0we: f64::from_bits(0x{:016x}),", prepared.model.lku0we.to_bits())?;
writeln!(out, "wkvth0we: f64::from_bits(0x{:016x}),", prepared.model.wkvth0we.to_bits())?;
writeln!(out, "wk2we: f64::from_bits(0x{:016x}),", prepared.model.wk2we.to_bits())?;
writeln!(out, "wku0we: f64::from_bits(0x{:016x}),", prepared.model.wku0we.to_bits())?;
writeln!(out, "pkvth0we: f64::from_bits(0x{:016x}),", prepared.model.pkvth0we.to_bits())?;
writeln!(out, "pk2we: f64::from_bits(0x{:016x}),", prepared.model.pk2we.to_bits())?;
writeln!(out, "pku0we: f64::from_bits(0x{:016x}),", prepared.model.pku0we.to_bits())?;
writeln!(out, "Eg0: f64::from_bits(0x{:016x}),", prepared.model.Eg0.to_bits())?;
writeln!(out, "vtm: f64::from_bits(0x{:016x}),", prepared.model.vtm.to_bits())?;
writeln!(out, "vtm0: f64::from_bits(0x{:016x}),", prepared.model.vtm0.to_bits())?;
writeln!(out, "coxe: f64::from_bits(0x{:016x}),", prepared.model.coxe.to_bits())?;
writeln!(out, "coxp: f64::from_bits(0x{:016x}),", prepared.model.coxp.to_bits())?;
writeln!(out, "cof1: f64::from_bits(0x{:016x}),", prepared.model.cof1.to_bits())?;
writeln!(out, "cof2: f64::from_bits(0x{:016x}),", prepared.model.cof2.to_bits())?;
writeln!(out, "cof3: f64::from_bits(0x{:016x}),", prepared.model.cof3.to_bits())?;
writeln!(out, "cof4: f64::from_bits(0x{:016x}),", prepared.model.cof4.to_bits())?;
writeln!(out, "vcrit: f64::from_bits(0x{:016x}),", prepared.model.vcrit.to_bits())?;
writeln!(out, "factor1: f64::from_bits(0x{:016x}),", prepared.model.factor1.to_bits())?;
writeln!(out, "PhiBS: f64::from_bits(0x{:016x}),", prepared.model.PhiBS.to_bits())?;
writeln!(out, "PhiBSWS: f64::from_bits(0x{:016x}),", prepared.model.PhiBSWS.to_bits())?;
writeln!(out, "PhiBSWGS: f64::from_bits(0x{:016x}),", prepared.model.PhiBSWGS.to_bits())?;
writeln!(out, "SjctTempSatCurDensity: f64::from_bits(0x{:016x}),", prepared.model.SjctTempSatCurDensity.to_bits())?;
writeln!(out, "SjctSidewallTempSatCurDensity: f64::from_bits(0x{:016x}),", prepared.model.SjctSidewallTempSatCurDensity.to_bits())?;
writeln!(out, "SjctGateSidewallTempSatCurDensity: f64::from_bits(0x{:016x}),", prepared.model.SjctGateSidewallTempSatCurDensity.to_bits())?;
writeln!(out, "PhiBD: f64::from_bits(0x{:016x}),", prepared.model.PhiBD.to_bits())?;
writeln!(out, "PhiBSWD: f64::from_bits(0x{:016x}),", prepared.model.PhiBSWD.to_bits())?;
writeln!(out, "PhiBSWGD: f64::from_bits(0x{:016x}),", prepared.model.PhiBSWGD.to_bits())?;
writeln!(out, "DjctTempSatCurDensity: f64::from_bits(0x{:016x}),", prepared.model.DjctTempSatCurDensity.to_bits())?;
writeln!(out, "DjctSidewallTempSatCurDensity: f64::from_bits(0x{:016x}),", prepared.model.DjctSidewallTempSatCurDensity.to_bits())?;
writeln!(out, "DjctGateSidewallTempSatCurDensity: f64::from_bits(0x{:016x}),", prepared.model.DjctGateSidewallTempSatCurDensity.to_bits())?;
writeln!(out, "SunitAreaTempJctCap: f64::from_bits(0x{:016x}),", prepared.model.SunitAreaTempJctCap.to_bits())?;
writeln!(out, "DunitAreaTempJctCap: f64::from_bits(0x{:016x}),", prepared.model.DunitAreaTempJctCap.to_bits())?;
writeln!(out, "SunitLengthSidewallTempJctCap: f64::from_bits(0x{:016x}),", prepared.model.SunitLengthSidewallTempJctCap.to_bits())?;
writeln!(out, "DunitLengthSidewallTempJctCap: f64::from_bits(0x{:016x}),", prepared.model.DunitLengthSidewallTempJctCap.to_bits())?;
writeln!(out, "SunitLengthGateSidewallTempJctCap: f64::from_bits(0x{:016x}),", prepared.model.SunitLengthGateSidewallTempJctCap.to_bits())?;
writeln!(out, "DunitLengthGateSidewallTempJctCap: f64::from_bits(0x{:016x}),", prepared.model.DunitLengthGateSidewallTempJctCap.to_bits())?;
writeln!(out, "oxideTrapDensityA: f64::from_bits(0x{:016x}),", prepared.model.oxideTrapDensityA.to_bits())?;
writeln!(out, "oxideTrapDensityB: f64::from_bits(0x{:016x}),", prepared.model.oxideTrapDensityB.to_bits())?;
writeln!(out, "oxideTrapDensityC: f64::from_bits(0x{:016x}),", prepared.model.oxideTrapDensityC.to_bits())?;
writeln!(out, "em: f64::from_bits(0x{:016x}),", prepared.model.em.to_bits())?;
writeln!(out, "ef: f64::from_bits(0x{:016x}),", prepared.model.ef.to_bits())?;
writeln!(out, "af: f64::from_bits(0x{:016x}),", prepared.model.af.to_bits())?;
writeln!(out, "kf: f64::from_bits(0x{:016x}),", prepared.model.kf.to_bits())?;
writeln!(out, "vgsMax: f64::from_bits(0x{:016x}),", prepared.model.vgsMax.to_bits())?;
writeln!(out, "vgdMax: f64::from_bits(0x{:016x}),", prepared.model.vgdMax.to_bits())?;
writeln!(out, "vgbMax: f64::from_bits(0x{:016x}),", prepared.model.vgbMax.to_bits())?;
writeln!(out, "vdsMax: f64::from_bits(0x{:016x}),", prepared.model.vdsMax.to_bits())?;
writeln!(out, "vbsMax: f64::from_bits(0x{:016x}),", prepared.model.vbsMax.to_bits())?;
writeln!(out, "vbdMax: f64::from_bits(0x{:016x}),", prepared.model.vbdMax.to_bits())?;
writeln!(out, "vgsrMax: f64::from_bits(0x{:016x}),", prepared.model.vgsrMax.to_bits())?;
writeln!(out, "vgdrMax: f64::from_bits(0x{:016x}),", prepared.model.vgdrMax.to_bits())?;
writeln!(out, "vgbrMax: f64::from_bits(0x{:016x}),", prepared.model.vgbrMax.to_bits())?;
writeln!(out, "vbsrMax: f64::from_bits(0x{:016x}),", prepared.model.vbsrMax.to_bits())?;
writeln!(out, "vbdrMax: f64::from_bits(0x{:016x}),", prepared.model.vbdrMax.to_bits())?;
writeln!(out, "gidlclamp: f64::from_bits(0x{:016x}),", prepared.model.gidlclamp.to_bits())?;
writeln!(out, "idovvdsc: f64::from_bits(0x{:016x}),", prepared.model.idovvdsc.to_bits())?;
writeln!(out, "InstCount: {},", prepared.model.InstCount)?;
writeln!(out, "mobModGiven: {},", prepared.model.mobModGiven)?;
writeln!(out, "binUnitGiven: {},", prepared.model.binUnitGiven)?;
writeln!(out, "cvchargeModGiven: {},", prepared.model.cvchargeModGiven)?;
writeln!(out, "capModGiven: {},", prepared.model.capModGiven)?;
writeln!(out, "dioModGiven: {},", prepared.model.dioModGiven)?;
writeln!(out, "rdsModGiven: {},", prepared.model.rdsModGiven)?;
writeln!(out, "rbodyModGiven: {},", prepared.model.rbodyModGiven)?;
writeln!(out, "rgateModGiven: {},", prepared.model.rgateModGiven)?;
writeln!(out, "perModGiven: {},", prepared.model.perModGiven)?;
writeln!(out, "geoModGiven: {},", prepared.model.geoModGiven)?;
writeln!(out, "paramChkGiven: {},", prepared.model.paramChkGiven)?;
writeln!(out, "trnqsModGiven: {},", prepared.model.trnqsModGiven)?;
writeln!(out, "acnqsModGiven: {},", prepared.model.acnqsModGiven)?;
writeln!(out, "fnoiModGiven: {},", prepared.model.fnoiModGiven)?;
writeln!(out, "tnoiModGiven: {},", prepared.model.tnoiModGiven)?;
writeln!(out, "mtrlModGiven: {},", prepared.model.mtrlModGiven)?;
writeln!(out, "mtrlCompatModGiven: {},", prepared.model.mtrlCompatModGiven)?;
writeln!(out, "gidlModGiven: {},", prepared.model.gidlModGiven)?;
writeln!(out, "igcModGiven: {},", prepared.model.igcModGiven)?;
writeln!(out, "igbModGiven: {},", prepared.model.igbModGiven)?;
writeln!(out, "tempModGiven: {},", prepared.model.tempModGiven)?;
writeln!(out, "typeGiven: {},", prepared.model.typeGiven)?;
writeln!(out, "toxrefGiven: {},", prepared.model.toxrefGiven)?;
writeln!(out, "eotGiven: {},", prepared.model.eotGiven)?;
writeln!(out, "vddeotGiven: {},", prepared.model.vddeotGiven)?;
writeln!(out, "tempeotGiven: {},", prepared.model.tempeotGiven)?;
writeln!(out, "leffeotGiven: {},", prepared.model.leffeotGiven)?;
writeln!(out, "weffeotGiven: {},", prepared.model.weffeotGiven)?;
writeln!(out, "adosGiven: {},", prepared.model.adosGiven)?;
writeln!(out, "bdosGiven: {},", prepared.model.bdosGiven)?;
writeln!(out, "toxeGiven: {},", prepared.model.toxeGiven)?;
writeln!(out, "toxpGiven: {},", prepared.model.toxpGiven)?;
writeln!(out, "toxmGiven: {},", prepared.model.toxmGiven)?;
writeln!(out, "dtoxGiven: {},", prepared.model.dtoxGiven)?;
writeln!(out, "epsroxGiven: {},", prepared.model.epsroxGiven)?;
writeln!(out, "versionGiven: {},", prepared.model.versionGiven)?;
writeln!(out, "cdscGiven: {},", prepared.model.cdscGiven)?;
writeln!(out, "cdscbGiven: {},", prepared.model.cdscbGiven)?;
writeln!(out, "cdscdGiven: {},", prepared.model.cdscdGiven)?;
writeln!(out, "citGiven: {},", prepared.model.citGiven)?;
writeln!(out, "nfactorGiven: {},", prepared.model.nfactorGiven)?;
writeln!(out, "xjGiven: {},", prepared.model.xjGiven)?;
writeln!(out, "vsatGiven: {},", prepared.model.vsatGiven)?;
writeln!(out, "atGiven: {},", prepared.model.atGiven)?;
writeln!(out, "a0Given: {},", prepared.model.a0Given)?;
writeln!(out, "agsGiven: {},", prepared.model.agsGiven)?;
writeln!(out, "a1Given: {},", prepared.model.a1Given)?;
writeln!(out, "a2Given: {},", prepared.model.a2Given)?;
writeln!(out, "ketaGiven: {},", prepared.model.ketaGiven)?;
writeln!(out, "ketacGiven: {},", prepared.model.ketacGiven)?;
writeln!(out, "nsubGiven: {},", prepared.model.nsubGiven)?;
writeln!(out, "phigGiven: {},", prepared.model.phigGiven)?;
writeln!(out, "epsrgateGiven: {},", prepared.model.epsrgateGiven)?;
writeln!(out, "easubGiven: {},", prepared.model.easubGiven)?;
writeln!(out, "epsrsubGiven: {},", prepared.model.epsrsubGiven)?;
writeln!(out, "ni0subGiven: {},", prepared.model.ni0subGiven)?;
writeln!(out, "bg0subGiven: {},", prepared.model.bg0subGiven)?;
writeln!(out, "tbgasubGiven: {},", prepared.model.tbgasubGiven)?;
writeln!(out, "tbgbsubGiven: {},", prepared.model.tbgbsubGiven)?;
writeln!(out, "ndepGiven: {},", prepared.model.ndepGiven)?;
writeln!(out, "nsdGiven: {},", prepared.model.nsdGiven)?;
writeln!(out, "phinGiven: {},", prepared.model.phinGiven)?;
writeln!(out, "ngateGiven: {},", prepared.model.ngateGiven)?;
writeln!(out, "gamma1Given: {},", prepared.model.gamma1Given)?;
writeln!(out, "gamma2Given: {},", prepared.model.gamma2Given)?;
writeln!(out, "vbxGiven: {},", prepared.model.vbxGiven)?;
writeln!(out, "vbmGiven: {},", prepared.model.vbmGiven)?;
writeln!(out, "xtGiven: {},", prepared.model.xtGiven)?;
writeln!(out, "k1Given: {},", prepared.model.k1Given)?;
writeln!(out, "kt1Given: {},", prepared.model.kt1Given)?;
writeln!(out, "kt1lGiven: {},", prepared.model.kt1lGiven)?;
writeln!(out, "kt2Given: {},", prepared.model.kt2Given)?;
writeln!(out, "k2Given: {},", prepared.model.k2Given)?;
writeln!(out, "k3Given: {},", prepared.model.k3Given)?;
writeln!(out, "k3bGiven: {},", prepared.model.k3bGiven)?;
writeln!(out, "w0Given: {},", prepared.model.w0Given)?;
writeln!(out, "dvtp0Given: {},", prepared.model.dvtp0Given)?;
writeln!(out, "dvtp1Given: {},", prepared.model.dvtp1Given)?;
writeln!(out, "dvtp2Given: {},", prepared.model.dvtp2Given)?;
writeln!(out, "dvtp3Given: {},", prepared.model.dvtp3Given)?;
writeln!(out, "dvtp4Given: {},", prepared.model.dvtp4Given)?;
writeln!(out, "dvtp5Given: {},", prepared.model.dvtp5Given)?;
writeln!(out, "lpe0Given: {},", prepared.model.lpe0Given)?;
writeln!(out, "lpebGiven: {},", prepared.model.lpebGiven)?;
writeln!(out, "dvt0Given: {},", prepared.model.dvt0Given)?;
writeln!(out, "dvt1Given: {},", prepared.model.dvt1Given)?;
writeln!(out, "dvt2Given: {},", prepared.model.dvt2Given)?;
writeln!(out, "dvt0wGiven: {},", prepared.model.dvt0wGiven)?;
writeln!(out, "dvt1wGiven: {},", prepared.model.dvt1wGiven)?;
writeln!(out, "dvt2wGiven: {},", prepared.model.dvt2wGiven)?;
writeln!(out, "droutGiven: {},", prepared.model.droutGiven)?;
writeln!(out, "dsubGiven: {},", prepared.model.dsubGiven)?;
writeln!(out, "vth0Given: {},", prepared.model.vth0Given)?;
writeln!(out, "euGiven: {},", prepared.model.euGiven)?;
writeln!(out, "ucsGiven: {},", prepared.model.ucsGiven)?;
writeln!(out, "uaGiven: {},", prepared.model.uaGiven)?;
writeln!(out, "ua1Given: {},", prepared.model.ua1Given)?;
writeln!(out, "ubGiven: {},", prepared.model.ubGiven)?;
writeln!(out, "ub1Given: {},", prepared.model.ub1Given)?;
writeln!(out, "ucGiven: {},", prepared.model.ucGiven)?;
writeln!(out, "uc1Given: {},", prepared.model.uc1Given)?;
writeln!(out, "udGiven: {},", prepared.model.udGiven)?;
writeln!(out, "ud1Given: {},", prepared.model.ud1Given)?;
writeln!(out, "upGiven: {},", prepared.model.upGiven)?;
writeln!(out, "lpGiven: {},", prepared.model.lpGiven)?;
writeln!(out, "u0Given: {},", prepared.model.u0Given)?;
writeln!(out, "uteGiven: {},", prepared.model.uteGiven)?;
writeln!(out, "ucsteGiven: {},", prepared.model.ucsteGiven)?;
writeln!(out, "voffGiven: {},", prepared.model.voffGiven)?;
writeln!(out, "tvoffGiven: {},", prepared.model.tvoffGiven)?;
writeln!(out, "tnfactorGiven: {},", prepared.model.tnfactorGiven)?;
writeln!(out, "teta0Given: {},", prepared.model.teta0Given)?;
writeln!(out, "tvoffcvGiven: {},", prepared.model.tvoffcvGiven)?;
writeln!(out, "vofflGiven: {},", prepared.model.vofflGiven)?;
writeln!(out, "voffcvlGiven: {},", prepared.model.voffcvlGiven)?;
writeln!(out, "minvGiven: {},", prepared.model.minvGiven)?;
writeln!(out, "minvcvGiven: {},", prepared.model.minvcvGiven)?;
writeln!(out, "rdswGiven: {},", prepared.model.rdswGiven)?;
writeln!(out, "rdswminGiven: {},", prepared.model.rdswminGiven)?;
writeln!(out, "rdwminGiven: {},", prepared.model.rdwminGiven)?;
writeln!(out, "rswminGiven: {},", prepared.model.rswminGiven)?;
writeln!(out, "rswGiven: {},", prepared.model.rswGiven)?;
writeln!(out, "rdwGiven: {},", prepared.model.rdwGiven)?;
writeln!(out, "prwgGiven: {},", prepared.model.prwgGiven)?;
writeln!(out, "prwbGiven: {},", prepared.model.prwbGiven)?;
writeln!(out, "prtGiven: {},", prepared.model.prtGiven)?;
writeln!(out, "eta0Given: {},", prepared.model.eta0Given)?;
writeln!(out, "etabGiven: {},", prepared.model.etabGiven)?;
writeln!(out, "pclmGiven: {},", prepared.model.pclmGiven)?;
writeln!(out, "pdibl1Given: {},", prepared.model.pdibl1Given)?;
writeln!(out, "pdibl2Given: {},", prepared.model.pdibl2Given)?;
writeln!(out, "pdiblbGiven: {},", prepared.model.pdiblbGiven)?;
writeln!(out, "fproutGiven: {},", prepared.model.fproutGiven)?;
writeln!(out, "pditsGiven: {},", prepared.model.pditsGiven)?;
writeln!(out, "pditsdGiven: {},", prepared.model.pditsdGiven)?;
writeln!(out, "pditslGiven: {},", prepared.model.pditslGiven)?;
writeln!(out, "pscbe1Given: {},", prepared.model.pscbe1Given)?;
writeln!(out, "pscbe2Given: {},", prepared.model.pscbe2Given)?;
writeln!(out, "pvagGiven: {},", prepared.model.pvagGiven)?;
writeln!(out, "deltaGiven: {},", prepared.model.deltaGiven)?;
writeln!(out, "wrGiven: {},", prepared.model.wrGiven)?;
writeln!(out, "dwgGiven: {},", prepared.model.dwgGiven)?;
writeln!(out, "dwbGiven: {},", prepared.model.dwbGiven)?;
writeln!(out, "b0Given: {},", prepared.model.b0Given)?;
writeln!(out, "b1Given: {},", prepared.model.b1Given)?;
writeln!(out, "alpha0Given: {},", prepared.model.alpha0Given)?;
writeln!(out, "alpha1Given: {},", prepared.model.alpha1Given)?;
writeln!(out, "beta0Given: {},", prepared.model.beta0Given)?;
writeln!(out, "agidlGiven: {},", prepared.model.agidlGiven)?;
writeln!(out, "bgidlGiven: {},", prepared.model.bgidlGiven)?;
writeln!(out, "cgidlGiven: {},", prepared.model.cgidlGiven)?;
writeln!(out, "egidlGiven: {},", prepared.model.egidlGiven)?;
writeln!(out, "fgidlGiven: {},", prepared.model.fgidlGiven)?;
writeln!(out, "kgidlGiven: {},", prepared.model.kgidlGiven)?;
writeln!(out, "rgidlGiven: {},", prepared.model.rgidlGiven)?;
writeln!(out, "agislGiven: {},", prepared.model.agislGiven)?;
writeln!(out, "bgislGiven: {},", prepared.model.bgislGiven)?;
writeln!(out, "cgislGiven: {},", prepared.model.cgislGiven)?;
writeln!(out, "egislGiven: {},", prepared.model.egislGiven)?;
writeln!(out, "fgislGiven: {},", prepared.model.fgislGiven)?;
writeln!(out, "kgislGiven: {},", prepared.model.kgislGiven)?;
writeln!(out, "rgislGiven: {},", prepared.model.rgislGiven)?;
writeln!(out, "aigcGiven: {},", prepared.model.aigcGiven)?;
writeln!(out, "bigcGiven: {},", prepared.model.bigcGiven)?;
writeln!(out, "cigcGiven: {},", prepared.model.cigcGiven)?;
writeln!(out, "aigsdGiven: {},", prepared.model.aigsdGiven)?;
writeln!(out, "bigsdGiven: {},", prepared.model.bigsdGiven)?;
writeln!(out, "cigsdGiven: {},", prepared.model.cigsdGiven)?;
writeln!(out, "aigsGiven: {},", prepared.model.aigsGiven)?;
writeln!(out, "bigsGiven: {},", prepared.model.bigsGiven)?;
writeln!(out, "cigsGiven: {},", prepared.model.cigsGiven)?;
writeln!(out, "aigdGiven: {},", prepared.model.aigdGiven)?;
writeln!(out, "bigdGiven: {},", prepared.model.bigdGiven)?;
writeln!(out, "cigdGiven: {},", prepared.model.cigdGiven)?;
writeln!(out, "aigbaccGiven: {},", prepared.model.aigbaccGiven)?;
writeln!(out, "bigbaccGiven: {},", prepared.model.bigbaccGiven)?;
writeln!(out, "cigbaccGiven: {},", prepared.model.cigbaccGiven)?;
writeln!(out, "aigbinvGiven: {},", prepared.model.aigbinvGiven)?;
writeln!(out, "bigbinvGiven: {},", prepared.model.bigbinvGiven)?;
writeln!(out, "cigbinvGiven: {},", prepared.model.cigbinvGiven)?;
writeln!(out, "nigcGiven: {},", prepared.model.nigcGiven)?;
writeln!(out, "nigbinvGiven: {},", prepared.model.nigbinvGiven)?;
writeln!(out, "nigbaccGiven: {},", prepared.model.nigbaccGiven)?;
writeln!(out, "ntoxGiven: {},", prepared.model.ntoxGiven)?;
writeln!(out, "eigbinvGiven: {},", prepared.model.eigbinvGiven)?;
writeln!(out, "pigcdGiven: {},", prepared.model.pigcdGiven)?;
writeln!(out, "poxedgeGiven: {},", prepared.model.poxedgeGiven)?;
writeln!(out, "ijthdfwdGiven: {},", prepared.model.ijthdfwdGiven)?;
writeln!(out, "ijthsfwdGiven: {},", prepared.model.ijthsfwdGiven)?;
writeln!(out, "ijthdrevGiven: {},", prepared.model.ijthdrevGiven)?;
writeln!(out, "ijthsrevGiven: {},", prepared.model.ijthsrevGiven)?;
writeln!(out, "xjbvdGiven: {},", prepared.model.xjbvdGiven)?;
writeln!(out, "xjbvsGiven: {},", prepared.model.xjbvsGiven)?;
writeln!(out, "bvdGiven: {},", prepared.model.bvdGiven)?;
writeln!(out, "bvsGiven: {},", prepared.model.bvsGiven)?;
writeln!(out, "jtssGiven: {},", prepared.model.jtssGiven)?;
writeln!(out, "jtsdGiven: {},", prepared.model.jtsdGiven)?;
writeln!(out, "jtsswsGiven: {},", prepared.model.jtsswsGiven)?;
writeln!(out, "jtsswdGiven: {},", prepared.model.jtsswdGiven)?;
writeln!(out, "jtsswgsGiven: {},", prepared.model.jtsswgsGiven)?;
writeln!(out, "jtsswgdGiven: {},", prepared.model.jtsswgdGiven)?;
writeln!(out, "jtweffGiven: {},", prepared.model.jtweffGiven)?;
writeln!(out, "njtsGiven: {},", prepared.model.njtsGiven)?;
writeln!(out, "njtsswGiven: {},", prepared.model.njtsswGiven)?;
writeln!(out, "njtsswgGiven: {},", prepared.model.njtsswgGiven)?;
writeln!(out, "njtsdGiven: {},", prepared.model.njtsdGiven)?;
writeln!(out, "njtsswdGiven: {},", prepared.model.njtsswdGiven)?;
writeln!(out, "njtsswgdGiven: {},", prepared.model.njtsswgdGiven)?;
writeln!(out, "xtssGiven: {},", prepared.model.xtssGiven)?;
writeln!(out, "xtsdGiven: {},", prepared.model.xtsdGiven)?;
writeln!(out, "xtsswsGiven: {},", prepared.model.xtsswsGiven)?;
writeln!(out, "xtsswdGiven: {},", prepared.model.xtsswdGiven)?;
writeln!(out, "xtsswgsGiven: {},", prepared.model.xtsswgsGiven)?;
writeln!(out, "xtsswgdGiven: {},", prepared.model.xtsswgdGiven)?;
writeln!(out, "tnjtsGiven: {},", prepared.model.tnjtsGiven)?;
writeln!(out, "tnjtsswGiven: {},", prepared.model.tnjtsswGiven)?;
writeln!(out, "tnjtsswgGiven: {},", prepared.model.tnjtsswgGiven)?;
writeln!(out, "tnjtsdGiven: {},", prepared.model.tnjtsdGiven)?;
writeln!(out, "tnjtsswdGiven: {},", prepared.model.tnjtsswdGiven)?;
writeln!(out, "tnjtsswgdGiven: {},", prepared.model.tnjtsswgdGiven)?;
writeln!(out, "vtssGiven: {},", prepared.model.vtssGiven)?;
writeln!(out, "vtsdGiven: {},", prepared.model.vtsdGiven)?;
writeln!(out, "vtsswsGiven: {},", prepared.model.vtsswsGiven)?;
writeln!(out, "vtsswdGiven: {},", prepared.model.vtsswdGiven)?;
writeln!(out, "vtsswgsGiven: {},", prepared.model.vtsswgsGiven)?;
writeln!(out, "vtsswgdGiven: {},", prepared.model.vtsswgdGiven)?;
writeln!(out, "vfbGiven: {},", prepared.model.vfbGiven)?;
writeln!(out, "gbminGiven: {},", prepared.model.gbminGiven)?;
writeln!(out, "rbdbGiven: {},", prepared.model.rbdbGiven)?;
writeln!(out, "rbsbGiven: {},", prepared.model.rbsbGiven)?;
writeln!(out, "rbpsGiven: {},", prepared.model.rbpsGiven)?;
writeln!(out, "rbpdGiven: {},", prepared.model.rbpdGiven)?;
writeln!(out, "rbpbGiven: {},", prepared.model.rbpbGiven)?;
writeln!(out, "rbps0Given: {},", prepared.model.rbps0Given)?;
writeln!(out, "rbpslGiven: {},", prepared.model.rbpslGiven)?;
writeln!(out, "rbpswGiven: {},", prepared.model.rbpswGiven)?;
writeln!(out, "rbpsnfGiven: {},", prepared.model.rbpsnfGiven)?;
writeln!(out, "rbpd0Given: {},", prepared.model.rbpd0Given)?;
writeln!(out, "rbpdlGiven: {},", prepared.model.rbpdlGiven)?;
writeln!(out, "rbpdwGiven: {},", prepared.model.rbpdwGiven)?;
writeln!(out, "rbpdnfGiven: {},", prepared.model.rbpdnfGiven)?;
writeln!(out, "rbpbx0Given: {},", prepared.model.rbpbx0Given)?;
writeln!(out, "rbpbxlGiven: {},", prepared.model.rbpbxlGiven)?;
writeln!(out, "rbpbxwGiven: {},", prepared.model.rbpbxwGiven)?;
writeln!(out, "rbpbxnfGiven: {},", prepared.model.rbpbxnfGiven)?;
writeln!(out, "rbpby0Given: {},", prepared.model.rbpby0Given)?;
writeln!(out, "rbpbylGiven: {},", prepared.model.rbpbylGiven)?;
writeln!(out, "rbpbywGiven: {},", prepared.model.rbpbywGiven)?;
writeln!(out, "rbpbynfGiven: {},", prepared.model.rbpbynfGiven)?;
writeln!(out, "rbsbx0Given: {},", prepared.model.rbsbx0Given)?;
writeln!(out, "rbsby0Given: {},", prepared.model.rbsby0Given)?;
writeln!(out, "rbdbx0Given: {},", prepared.model.rbdbx0Given)?;
writeln!(out, "rbdby0Given: {},", prepared.model.rbdby0Given)?;
writeln!(out, "rbsdbxlGiven: {},", prepared.model.rbsdbxlGiven)?;
writeln!(out, "rbsdbxwGiven: {},", prepared.model.rbsdbxwGiven)?;
writeln!(out, "rbsdbxnfGiven: {},", prepared.model.rbsdbxnfGiven)?;
writeln!(out, "rbsdbylGiven: {},", prepared.model.rbsdbylGiven)?;
writeln!(out, "rbsdbywGiven: {},", prepared.model.rbsdbywGiven)?;
writeln!(out, "rbsdbynfGiven: {},", prepared.model.rbsdbynfGiven)?;
writeln!(out, "xrcrg1Given: {},", prepared.model.xrcrg1Given)?;
writeln!(out, "xrcrg2Given: {},", prepared.model.xrcrg2Given)?;
writeln!(out, "tnoiaGiven: {},", prepared.model.tnoiaGiven)?;
writeln!(out, "tnoibGiven: {},", prepared.model.tnoibGiven)?;
writeln!(out, "tnoicGiven: {},", prepared.model.tnoicGiven)?;
writeln!(out, "rnoiaGiven: {},", prepared.model.rnoiaGiven)?;
writeln!(out, "rnoibGiven: {},", prepared.model.rnoibGiven)?;
writeln!(out, "rnoicGiven: {},", prepared.model.rnoicGiven)?;
writeln!(out, "ntnoiGiven: {},", prepared.model.ntnoiGiven)?;
writeln!(out, "lambdaGiven: {},", prepared.model.lambdaGiven)?;
writeln!(out, "vtlGiven: {},", prepared.model.vtlGiven)?;
writeln!(out, "lcGiven: {},", prepared.model.lcGiven)?;
writeln!(out, "xnGiven: {},", prepared.model.xnGiven)?;
writeln!(out, "vfbsdoffGiven: {},", prepared.model.vfbsdoffGiven)?;
writeln!(out, "lintnoiGiven: {},", prepared.model.lintnoiGiven)?;
writeln!(out, "tvfbsdoffGiven: {},", prepared.model.tvfbsdoffGiven)?;
writeln!(out, "cgslGiven: {},", prepared.model.cgslGiven)?;
writeln!(out, "cgdlGiven: {},", prepared.model.cgdlGiven)?;
writeln!(out, "ckappasGiven: {},", prepared.model.ckappasGiven)?;
writeln!(out, "ckappadGiven: {},", prepared.model.ckappadGiven)?;
writeln!(out, "cfGiven: {},", prepared.model.cfGiven)?;
writeln!(out, "vfbcvGiven: {},", prepared.model.vfbcvGiven)?;
writeln!(out, "clcGiven: {},", prepared.model.clcGiven)?;
writeln!(out, "cleGiven: {},", prepared.model.cleGiven)?;
writeln!(out, "dwcGiven: {},", prepared.model.dwcGiven)?;
writeln!(out, "dlcGiven: {},", prepared.model.dlcGiven)?;
writeln!(out, "xwGiven: {},", prepared.model.xwGiven)?;
writeln!(out, "xlGiven: {},", prepared.model.xlGiven)?;
writeln!(out, "dlcigGiven: {},", prepared.model.dlcigGiven)?;
writeln!(out, "dlcigdGiven: {},", prepared.model.dlcigdGiven)?;
writeln!(out, "dwjGiven: {},", prepared.model.dwjGiven)?;
writeln!(out, "noffGiven: {},", prepared.model.noffGiven)?;
writeln!(out, "voffcvGiven: {},", prepared.model.voffcvGiven)?;
writeln!(out, "acdeGiven: {},", prepared.model.acdeGiven)?;
writeln!(out, "moinGiven: {},", prepared.model.moinGiven)?;
writeln!(out, "tcjGiven: {},", prepared.model.tcjGiven)?;
writeln!(out, "tcjswGiven: {},", prepared.model.tcjswGiven)?;
writeln!(out, "tcjswgGiven: {},", prepared.model.tcjswgGiven)?;
writeln!(out, "tpbGiven: {},", prepared.model.tpbGiven)?;
writeln!(out, "tpbswGiven: {},", prepared.model.tpbswGiven)?;
writeln!(out, "tpbswgGiven: {},", prepared.model.tpbswgGiven)?;
writeln!(out, "dmcgGiven: {},", prepared.model.dmcgGiven)?;
writeln!(out, "dmciGiven: {},", prepared.model.dmciGiven)?;
writeln!(out, "dmdgGiven: {},", prepared.model.dmdgGiven)?;
writeln!(out, "dmcgtGiven: {},", prepared.model.dmcgtGiven)?;
writeln!(out, "xgwGiven: {},", prepared.model.xgwGiven)?;
writeln!(out, "xglGiven: {},", prepared.model.xglGiven)?;
writeln!(out, "rshgGiven: {},", prepared.model.rshgGiven)?;
writeln!(out, "ngconGiven: {},", prepared.model.ngconGiven)?;
writeln!(out, "lcdscGiven: {},", prepared.model.lcdscGiven)?;
writeln!(out, "lcdscbGiven: {},", prepared.model.lcdscbGiven)?;
writeln!(out, "lcdscdGiven: {},", prepared.model.lcdscdGiven)?;
writeln!(out, "lcitGiven: {},", prepared.model.lcitGiven)?;
writeln!(out, "lnfactorGiven: {},", prepared.model.lnfactorGiven)?;
writeln!(out, "lxjGiven: {},", prepared.model.lxjGiven)?;
writeln!(out, "lvsatGiven: {},", prepared.model.lvsatGiven)?;
writeln!(out, "latGiven: {},", prepared.model.latGiven)?;
writeln!(out, "la0Given: {},", prepared.model.la0Given)?;
writeln!(out, "lagsGiven: {},", prepared.model.lagsGiven)?;
writeln!(out, "la1Given: {},", prepared.model.la1Given)?;
writeln!(out, "la2Given: {},", prepared.model.la2Given)?;
writeln!(out, "lketaGiven: {},", prepared.model.lketaGiven)?;
writeln!(out, "lketacGiven: {},", prepared.model.lketacGiven)?;
writeln!(out, "lnsubGiven: {},", prepared.model.lnsubGiven)?;
writeln!(out, "lndepGiven: {},", prepared.model.lndepGiven)?;
writeln!(out, "lnsdGiven: {},", prepared.model.lnsdGiven)?;
writeln!(out, "lphinGiven: {},", prepared.model.lphinGiven)?;
writeln!(out, "lngateGiven: {},", prepared.model.lngateGiven)?;
writeln!(out, "lgamma1Given: {},", prepared.model.lgamma1Given)?;
writeln!(out, "lgamma2Given: {},", prepared.model.lgamma2Given)?;
writeln!(out, "lvbxGiven: {},", prepared.model.lvbxGiven)?;
writeln!(out, "lvbmGiven: {},", prepared.model.lvbmGiven)?;
writeln!(out, "lxtGiven: {},", prepared.model.lxtGiven)?;
writeln!(out, "lk1Given: {},", prepared.model.lk1Given)?;
writeln!(out, "lkt1Given: {},", prepared.model.lkt1Given)?;
writeln!(out, "lkt1lGiven: {},", prepared.model.lkt1lGiven)?;
writeln!(out, "lkt2Given: {},", prepared.model.lkt2Given)?;
writeln!(out, "lk2Given: {},", prepared.model.lk2Given)?;
writeln!(out, "lk3Given: {},", prepared.model.lk3Given)?;
writeln!(out, "lk3bGiven: {},", prepared.model.lk3bGiven)?;
writeln!(out, "lw0Given: {},", prepared.model.lw0Given)?;
writeln!(out, "ldvtp0Given: {},", prepared.model.ldvtp0Given)?;
writeln!(out, "ldvtp1Given: {},", prepared.model.ldvtp1Given)?;
writeln!(out, "ldvtp2Given: {},", prepared.model.ldvtp2Given)?;
writeln!(out, "ldvtp3Given: {},", prepared.model.ldvtp3Given)?;
writeln!(out, "ldvtp4Given: {},", prepared.model.ldvtp4Given)?;
writeln!(out, "ldvtp5Given: {},", prepared.model.ldvtp5Given)?;
writeln!(out, "llpe0Given: {},", prepared.model.llpe0Given)?;
writeln!(out, "llpebGiven: {},", prepared.model.llpebGiven)?;
writeln!(out, "ldvt0Given: {},", prepared.model.ldvt0Given)?;
writeln!(out, "ldvt1Given: {},", prepared.model.ldvt1Given)?;
writeln!(out, "ldvt2Given: {},", prepared.model.ldvt2Given)?;
writeln!(out, "ldvt0wGiven: {},", prepared.model.ldvt0wGiven)?;
writeln!(out, "ldvt1wGiven: {},", prepared.model.ldvt1wGiven)?;
writeln!(out, "ldvt2wGiven: {},", prepared.model.ldvt2wGiven)?;
writeln!(out, "ldroutGiven: {},", prepared.model.ldroutGiven)?;
writeln!(out, "ldsubGiven: {},", prepared.model.ldsubGiven)?;
writeln!(out, "lvth0Given: {},", prepared.model.lvth0Given)?;
writeln!(out, "luaGiven: {},", prepared.model.luaGiven)?;
writeln!(out, "lua1Given: {},", prepared.model.lua1Given)?;
writeln!(out, "lubGiven: {},", prepared.model.lubGiven)?;
writeln!(out, "lub1Given: {},", prepared.model.lub1Given)?;
writeln!(out, "lucGiven: {},", prepared.model.lucGiven)?;
writeln!(out, "luc1Given: {},", prepared.model.luc1Given)?;
writeln!(out, "ludGiven: {},", prepared.model.ludGiven)?;
writeln!(out, "lud1Given: {},", prepared.model.lud1Given)?;
writeln!(out, "lupGiven: {},", prepared.model.lupGiven)?;
writeln!(out, "llpGiven: {},", prepared.model.llpGiven)?;
writeln!(out, "lu0Given: {},", prepared.model.lu0Given)?;
writeln!(out, "leuGiven: {},", prepared.model.leuGiven)?;
writeln!(out, "lucsGiven: {},", prepared.model.lucsGiven)?;
writeln!(out, "luteGiven: {},", prepared.model.luteGiven)?;
writeln!(out, "lucsteGiven: {},", prepared.model.lucsteGiven)?;
writeln!(out, "lvoffGiven: {},", prepared.model.lvoffGiven)?;
writeln!(out, "ltvoffGiven: {},", prepared.model.ltvoffGiven)?;
writeln!(out, "ltnfactorGiven: {},", prepared.model.ltnfactorGiven)?;
writeln!(out, "lteta0Given: {},", prepared.model.lteta0Given)?;
writeln!(out, "ltvoffcvGiven: {},", prepared.model.ltvoffcvGiven)?;
writeln!(out, "lminvGiven: {},", prepared.model.lminvGiven)?;
writeln!(out, "lminvcvGiven: {},", prepared.model.lminvcvGiven)?;
writeln!(out, "lrdswGiven: {},", prepared.model.lrdswGiven)?;
writeln!(out, "lrswGiven: {},", prepared.model.lrswGiven)?;
writeln!(out, "lrdwGiven: {},", prepared.model.lrdwGiven)?;
writeln!(out, "lprwgGiven: {},", prepared.model.lprwgGiven)?;
writeln!(out, "lprwbGiven: {},", prepared.model.lprwbGiven)?;
writeln!(out, "lprtGiven: {},", prepared.model.lprtGiven)?;
writeln!(out, "leta0Given: {},", prepared.model.leta0Given)?;
writeln!(out, "letabGiven: {},", prepared.model.letabGiven)?;
writeln!(out, "lpclmGiven: {},", prepared.model.lpclmGiven)?;
writeln!(out, "lpdibl1Given: {},", prepared.model.lpdibl1Given)?;
writeln!(out, "lpdibl2Given: {},", prepared.model.lpdibl2Given)?;
writeln!(out, "lpdiblbGiven: {},", prepared.model.lpdiblbGiven)?;
writeln!(out, "lfproutGiven: {},", prepared.model.lfproutGiven)?;
writeln!(out, "lpditsGiven: {},", prepared.model.lpditsGiven)?;
writeln!(out, "lpditsdGiven: {},", prepared.model.lpditsdGiven)?;
writeln!(out, "lpscbe1Given: {},", prepared.model.lpscbe1Given)?;
writeln!(out, "lpscbe2Given: {},", prepared.model.lpscbe2Given)?;
writeln!(out, "lpvagGiven: {},", prepared.model.lpvagGiven)?;
writeln!(out, "ldeltaGiven: {},", prepared.model.ldeltaGiven)?;
writeln!(out, "lwrGiven: {},", prepared.model.lwrGiven)?;
writeln!(out, "ldwgGiven: {},", prepared.model.ldwgGiven)?;
writeln!(out, "ldwbGiven: {},", prepared.model.ldwbGiven)?;
writeln!(out, "lb0Given: {},", prepared.model.lb0Given)?;
writeln!(out, "lb1Given: {},", prepared.model.lb1Given)?;
writeln!(out, "lalpha0Given: {},", prepared.model.lalpha0Given)?;
writeln!(out, "lalpha1Given: {},", prepared.model.lalpha1Given)?;
writeln!(out, "lbeta0Given: {},", prepared.model.lbeta0Given)?;
writeln!(out, "lvfbGiven: {},", prepared.model.lvfbGiven)?;
writeln!(out, "lagidlGiven: {},", prepared.model.lagidlGiven)?;
writeln!(out, "lbgidlGiven: {},", prepared.model.lbgidlGiven)?;
writeln!(out, "lcgidlGiven: {},", prepared.model.lcgidlGiven)?;
writeln!(out, "legidlGiven: {},", prepared.model.legidlGiven)?;
writeln!(out, "lfgidlGiven: {},", prepared.model.lfgidlGiven)?;
writeln!(out, "lkgidlGiven: {},", prepared.model.lkgidlGiven)?;
writeln!(out, "lrgidlGiven: {},", prepared.model.lrgidlGiven)?;
writeln!(out, "lagislGiven: {},", prepared.model.lagislGiven)?;
writeln!(out, "lbgislGiven: {},", prepared.model.lbgislGiven)?;
writeln!(out, "lcgislGiven: {},", prepared.model.lcgislGiven)?;
writeln!(out, "legislGiven: {},", prepared.model.legislGiven)?;
writeln!(out, "lfgislGiven: {},", prepared.model.lfgislGiven)?;
writeln!(out, "lkgislGiven: {},", prepared.model.lkgislGiven)?;
writeln!(out, "lrgislGiven: {},", prepared.model.lrgislGiven)?;
writeln!(out, "laigcGiven: {},", prepared.model.laigcGiven)?;
writeln!(out, "lbigcGiven: {},", prepared.model.lbigcGiven)?;
writeln!(out, "lcigcGiven: {},", prepared.model.lcigcGiven)?;
writeln!(out, "laigsdGiven: {},", prepared.model.laigsdGiven)?;
writeln!(out, "lbigsdGiven: {},", prepared.model.lbigsdGiven)?;
writeln!(out, "lcigsdGiven: {},", prepared.model.lcigsdGiven)?;
writeln!(out, "laigsGiven: {},", prepared.model.laigsGiven)?;
writeln!(out, "lbigsGiven: {},", prepared.model.lbigsGiven)?;
writeln!(out, "lcigsGiven: {},", prepared.model.lcigsGiven)?;
writeln!(out, "laigdGiven: {},", prepared.model.laigdGiven)?;
writeln!(out, "lbigdGiven: {},", prepared.model.lbigdGiven)?;
writeln!(out, "lcigdGiven: {},", prepared.model.lcigdGiven)?;
writeln!(out, "laigbaccGiven: {},", prepared.model.laigbaccGiven)?;
writeln!(out, "lbigbaccGiven: {},", prepared.model.lbigbaccGiven)?;
writeln!(out, "lcigbaccGiven: {},", prepared.model.lcigbaccGiven)?;
writeln!(out, "laigbinvGiven: {},", prepared.model.laigbinvGiven)?;
writeln!(out, "lbigbinvGiven: {},", prepared.model.lbigbinvGiven)?;
writeln!(out, "lcigbinvGiven: {},", prepared.model.lcigbinvGiven)?;
writeln!(out, "lnigcGiven: {},", prepared.model.lnigcGiven)?;
writeln!(out, "lnigbinvGiven: {},", prepared.model.lnigbinvGiven)?;
writeln!(out, "lnigbaccGiven: {},", prepared.model.lnigbaccGiven)?;
writeln!(out, "lntoxGiven: {},", prepared.model.lntoxGiven)?;
writeln!(out, "leigbinvGiven: {},", prepared.model.leigbinvGiven)?;
writeln!(out, "lpigcdGiven: {},", prepared.model.lpigcdGiven)?;
writeln!(out, "lpoxedgeGiven: {},", prepared.model.lpoxedgeGiven)?;
writeln!(out, "lxrcrg1Given: {},", prepared.model.lxrcrg1Given)?;
writeln!(out, "lxrcrg2Given: {},", prepared.model.lxrcrg2Given)?;
writeln!(out, "llambdaGiven: {},", prepared.model.llambdaGiven)?;
writeln!(out, "lvtlGiven: {},", prepared.model.lvtlGiven)?;
writeln!(out, "lxnGiven: {},", prepared.model.lxnGiven)?;
writeln!(out, "lvfbsdoffGiven: {},", prepared.model.lvfbsdoffGiven)?;
writeln!(out, "ltvfbsdoffGiven: {},", prepared.model.ltvfbsdoffGiven)?;
writeln!(out, "lcgslGiven: {},", prepared.model.lcgslGiven)?;
writeln!(out, "lcgdlGiven: {},", prepared.model.lcgdlGiven)?;
writeln!(out, "lckappasGiven: {},", prepared.model.lckappasGiven)?;
writeln!(out, "lckappadGiven: {},", prepared.model.lckappadGiven)?;
writeln!(out, "lcfGiven: {},", prepared.model.lcfGiven)?;
writeln!(out, "lclcGiven: {},", prepared.model.lclcGiven)?;
writeln!(out, "lcleGiven: {},", prepared.model.lcleGiven)?;
writeln!(out, "lvfbcvGiven: {},", prepared.model.lvfbcvGiven)?;
writeln!(out, "lnoffGiven: {},", prepared.model.lnoffGiven)?;
writeln!(out, "lvoffcvGiven: {},", prepared.model.lvoffcvGiven)?;
writeln!(out, "lacdeGiven: {},", prepared.model.lacdeGiven)?;
writeln!(out, "lmoinGiven: {},", prepared.model.lmoinGiven)?;
writeln!(out, "wcdscGiven: {},", prepared.model.wcdscGiven)?;
writeln!(out, "wcdscbGiven: {},", prepared.model.wcdscbGiven)?;
writeln!(out, "wcdscdGiven: {},", prepared.model.wcdscdGiven)?;
writeln!(out, "wcitGiven: {},", prepared.model.wcitGiven)?;
writeln!(out, "wnfactorGiven: {},", prepared.model.wnfactorGiven)?;
writeln!(out, "wxjGiven: {},", prepared.model.wxjGiven)?;
writeln!(out, "wvsatGiven: {},", prepared.model.wvsatGiven)?;
writeln!(out, "watGiven: {},", prepared.model.watGiven)?;
writeln!(out, "wa0Given: {},", prepared.model.wa0Given)?;
writeln!(out, "wagsGiven: {},", prepared.model.wagsGiven)?;
writeln!(out, "wa1Given: {},", prepared.model.wa1Given)?;
writeln!(out, "wa2Given: {},", prepared.model.wa2Given)?;
writeln!(out, "wketaGiven: {},", prepared.model.wketaGiven)?;
writeln!(out, "wketacGiven: {},", prepared.model.wketacGiven)?;
writeln!(out, "wnsubGiven: {},", prepared.model.wnsubGiven)?;
writeln!(out, "wndepGiven: {},", prepared.model.wndepGiven)?;
writeln!(out, "wnsdGiven: {},", prepared.model.wnsdGiven)?;
writeln!(out, "wphinGiven: {},", prepared.model.wphinGiven)?;
writeln!(out, "wngateGiven: {},", prepared.model.wngateGiven)?;
writeln!(out, "wgamma1Given: {},", prepared.model.wgamma1Given)?;
writeln!(out, "wgamma2Given: {},", prepared.model.wgamma2Given)?;
writeln!(out, "wvbxGiven: {},", prepared.model.wvbxGiven)?;
writeln!(out, "wvbmGiven: {},", prepared.model.wvbmGiven)?;
writeln!(out, "wxtGiven: {},", prepared.model.wxtGiven)?;
writeln!(out, "wk1Given: {},", prepared.model.wk1Given)?;
writeln!(out, "wkt1Given: {},", prepared.model.wkt1Given)?;
writeln!(out, "wkt1lGiven: {},", prepared.model.wkt1lGiven)?;
writeln!(out, "wkt2Given: {},", prepared.model.wkt2Given)?;
writeln!(out, "wk2Given: {},", prepared.model.wk2Given)?;
writeln!(out, "wk3Given: {},", prepared.model.wk3Given)?;
writeln!(out, "wk3bGiven: {},", prepared.model.wk3bGiven)?;
writeln!(out, "ww0Given: {},", prepared.model.ww0Given)?;
writeln!(out, "wdvtp0Given: {},", prepared.model.wdvtp0Given)?;
writeln!(out, "wdvtp1Given: {},", prepared.model.wdvtp1Given)?;
writeln!(out, "wdvtp2Given: {},", prepared.model.wdvtp2Given)?;
writeln!(out, "wdvtp3Given: {},", prepared.model.wdvtp3Given)?;
writeln!(out, "wdvtp4Given: {},", prepared.model.wdvtp4Given)?;
writeln!(out, "wdvtp5Given: {},", prepared.model.wdvtp5Given)?;
writeln!(out, "wlpe0Given: {},", prepared.model.wlpe0Given)?;
writeln!(out, "wlpebGiven: {},", prepared.model.wlpebGiven)?;
writeln!(out, "wdvt0Given: {},", prepared.model.wdvt0Given)?;
writeln!(out, "wdvt1Given: {},", prepared.model.wdvt1Given)?;
writeln!(out, "wdvt2Given: {},", prepared.model.wdvt2Given)?;
writeln!(out, "wdvt0wGiven: {},", prepared.model.wdvt0wGiven)?;
writeln!(out, "wdvt1wGiven: {},", prepared.model.wdvt1wGiven)?;
writeln!(out, "wdvt2wGiven: {},", prepared.model.wdvt2wGiven)?;
writeln!(out, "wdroutGiven: {},", prepared.model.wdroutGiven)?;
writeln!(out, "wdsubGiven: {},", prepared.model.wdsubGiven)?;
writeln!(out, "wvth0Given: {},", prepared.model.wvth0Given)?;
writeln!(out, "wuaGiven: {},", prepared.model.wuaGiven)?;
writeln!(out, "wua1Given: {},", prepared.model.wua1Given)?;
writeln!(out, "wubGiven: {},", prepared.model.wubGiven)?;
writeln!(out, "wub1Given: {},", prepared.model.wub1Given)?;
writeln!(out, "wucGiven: {},", prepared.model.wucGiven)?;
writeln!(out, "wuc1Given: {},", prepared.model.wuc1Given)?;
writeln!(out, "wudGiven: {},", prepared.model.wudGiven)?;
writeln!(out, "wud1Given: {},", prepared.model.wud1Given)?;
writeln!(out, "wupGiven: {},", prepared.model.wupGiven)?;
writeln!(out, "wlpGiven: {},", prepared.model.wlpGiven)?;
writeln!(out, "wu0Given: {},", prepared.model.wu0Given)?;
writeln!(out, "weuGiven: {},", prepared.model.weuGiven)?;
writeln!(out, "wucsGiven: {},", prepared.model.wucsGiven)?;
writeln!(out, "wuteGiven: {},", prepared.model.wuteGiven)?;
writeln!(out, "wucsteGiven: {},", prepared.model.wucsteGiven)?;
writeln!(out, "wvoffGiven: {},", prepared.model.wvoffGiven)?;
writeln!(out, "wtvoffGiven: {},", prepared.model.wtvoffGiven)?;
writeln!(out, "wtnfactorGiven: {},", prepared.model.wtnfactorGiven)?;
writeln!(out, "wteta0Given: {},", prepared.model.wteta0Given)?;
writeln!(out, "wtvoffcvGiven: {},", prepared.model.wtvoffcvGiven)?;
writeln!(out, "wminvGiven: {},", prepared.model.wminvGiven)?;
writeln!(out, "wminvcvGiven: {},", prepared.model.wminvcvGiven)?;
writeln!(out, "wrdswGiven: {},", prepared.model.wrdswGiven)?;
writeln!(out, "wrswGiven: {},", prepared.model.wrswGiven)?;
writeln!(out, "wrdwGiven: {},", prepared.model.wrdwGiven)?;
writeln!(out, "wprwgGiven: {},", prepared.model.wprwgGiven)?;
writeln!(out, "wprwbGiven: {},", prepared.model.wprwbGiven)?;
writeln!(out, "wprtGiven: {},", prepared.model.wprtGiven)?;
writeln!(out, "weta0Given: {},", prepared.model.weta0Given)?;
writeln!(out, "wetabGiven: {},", prepared.model.wetabGiven)?;
writeln!(out, "wpclmGiven: {},", prepared.model.wpclmGiven)?;
writeln!(out, "wpdibl1Given: {},", prepared.model.wpdibl1Given)?;
writeln!(out, "wpdibl2Given: {},", prepared.model.wpdibl2Given)?;
writeln!(out, "wpdiblbGiven: {},", prepared.model.wpdiblbGiven)?;
writeln!(out, "wfproutGiven: {},", prepared.model.wfproutGiven)?;
writeln!(out, "wpditsGiven: {},", prepared.model.wpditsGiven)?;
writeln!(out, "wpditsdGiven: {},", prepared.model.wpditsdGiven)?;
writeln!(out, "wpscbe1Given: {},", prepared.model.wpscbe1Given)?;
writeln!(out, "wpscbe2Given: {},", prepared.model.wpscbe2Given)?;
writeln!(out, "wpvagGiven: {},", prepared.model.wpvagGiven)?;
writeln!(out, "wdeltaGiven: {},", prepared.model.wdeltaGiven)?;
writeln!(out, "wwrGiven: {},", prepared.model.wwrGiven)?;
writeln!(out, "wdwgGiven: {},", prepared.model.wdwgGiven)?;
writeln!(out, "wdwbGiven: {},", prepared.model.wdwbGiven)?;
writeln!(out, "wb0Given: {},", prepared.model.wb0Given)?;
writeln!(out, "wb1Given: {},", prepared.model.wb1Given)?;
writeln!(out, "walpha0Given: {},", prepared.model.walpha0Given)?;
writeln!(out, "walpha1Given: {},", prepared.model.walpha1Given)?;
writeln!(out, "wbeta0Given: {},", prepared.model.wbeta0Given)?;
writeln!(out, "wvfbGiven: {},", prepared.model.wvfbGiven)?;
writeln!(out, "wagidlGiven: {},", prepared.model.wagidlGiven)?;
writeln!(out, "wbgidlGiven: {},", prepared.model.wbgidlGiven)?;
writeln!(out, "wcgidlGiven: {},", prepared.model.wcgidlGiven)?;
writeln!(out, "wegidlGiven: {},", prepared.model.wegidlGiven)?;
writeln!(out, "wfgidlGiven: {},", prepared.model.wfgidlGiven)?;
writeln!(out, "wkgidlGiven: {},", prepared.model.wkgidlGiven)?;
writeln!(out, "wrgidlGiven: {},", prepared.model.wrgidlGiven)?;
writeln!(out, "wagislGiven: {},", prepared.model.wagislGiven)?;
writeln!(out, "wbgislGiven: {},", prepared.model.wbgislGiven)?;
writeln!(out, "wcgislGiven: {},", prepared.model.wcgislGiven)?;
writeln!(out, "wegislGiven: {},", prepared.model.wegislGiven)?;
writeln!(out, "wfgislGiven: {},", prepared.model.wfgislGiven)?;
writeln!(out, "wkgislGiven: {},", prepared.model.wkgislGiven)?;
writeln!(out, "wrgislGiven: {},", prepared.model.wrgislGiven)?;
writeln!(out, "waigcGiven: {},", prepared.model.waigcGiven)?;
writeln!(out, "wbigcGiven: {},", prepared.model.wbigcGiven)?;
writeln!(out, "wcigcGiven: {},", prepared.model.wcigcGiven)?;
writeln!(out, "waigsdGiven: {},", prepared.model.waigsdGiven)?;
writeln!(out, "wbigsdGiven: {},", prepared.model.wbigsdGiven)?;
writeln!(out, "wcigsdGiven: {},", prepared.model.wcigsdGiven)?;
writeln!(out, "waigsGiven: {},", prepared.model.waigsGiven)?;
writeln!(out, "wbigsGiven: {},", prepared.model.wbigsGiven)?;
writeln!(out, "wcigsGiven: {},", prepared.model.wcigsGiven)?;
writeln!(out, "waigdGiven: {},", prepared.model.waigdGiven)?;
writeln!(out, "wbigdGiven: {},", prepared.model.wbigdGiven)?;
writeln!(out, "wcigdGiven: {},", prepared.model.wcigdGiven)?;
writeln!(out, "waigbaccGiven: {},", prepared.model.waigbaccGiven)?;
writeln!(out, "wbigbaccGiven: {},", prepared.model.wbigbaccGiven)?;
writeln!(out, "wcigbaccGiven: {},", prepared.model.wcigbaccGiven)?;
writeln!(out, "waigbinvGiven: {},", prepared.model.waigbinvGiven)?;
writeln!(out, "wbigbinvGiven: {},", prepared.model.wbigbinvGiven)?;
writeln!(out, "wcigbinvGiven: {},", prepared.model.wcigbinvGiven)?;
writeln!(out, "wnigcGiven: {},", prepared.model.wnigcGiven)?;
writeln!(out, "wnigbinvGiven: {},", prepared.model.wnigbinvGiven)?;
writeln!(out, "wnigbaccGiven: {},", prepared.model.wnigbaccGiven)?;
writeln!(out, "wntoxGiven: {},", prepared.model.wntoxGiven)?;
writeln!(out, "weigbinvGiven: {},", prepared.model.weigbinvGiven)?;
writeln!(out, "wpigcdGiven: {},", prepared.model.wpigcdGiven)?;
writeln!(out, "wpoxedgeGiven: {},", prepared.model.wpoxedgeGiven)?;
writeln!(out, "wxrcrg1Given: {},", prepared.model.wxrcrg1Given)?;
writeln!(out, "wxrcrg2Given: {},", prepared.model.wxrcrg2Given)?;
writeln!(out, "wlambdaGiven: {},", prepared.model.wlambdaGiven)?;
writeln!(out, "wvtlGiven: {},", prepared.model.wvtlGiven)?;
writeln!(out, "wxnGiven: {},", prepared.model.wxnGiven)?;
writeln!(out, "wvfbsdoffGiven: {},", prepared.model.wvfbsdoffGiven)?;
writeln!(out, "wtvfbsdoffGiven: {},", prepared.model.wtvfbsdoffGiven)?;
writeln!(out, "wcgslGiven: {},", prepared.model.wcgslGiven)?;
writeln!(out, "wcgdlGiven: {},", prepared.model.wcgdlGiven)?;
writeln!(out, "wckappasGiven: {},", prepared.model.wckappasGiven)?;
writeln!(out, "wckappadGiven: {},", prepared.model.wckappadGiven)?;
writeln!(out, "wcfGiven: {},", prepared.model.wcfGiven)?;
writeln!(out, "wclcGiven: {},", prepared.model.wclcGiven)?;
writeln!(out, "wcleGiven: {},", prepared.model.wcleGiven)?;
writeln!(out, "wvfbcvGiven: {},", prepared.model.wvfbcvGiven)?;
writeln!(out, "wnoffGiven: {},", prepared.model.wnoffGiven)?;
writeln!(out, "wvoffcvGiven: {},", prepared.model.wvoffcvGiven)?;
writeln!(out, "wacdeGiven: {},", prepared.model.wacdeGiven)?;
writeln!(out, "wmoinGiven: {},", prepared.model.wmoinGiven)?;
writeln!(out, "pcdscGiven: {},", prepared.model.pcdscGiven)?;
writeln!(out, "pcdscbGiven: {},", prepared.model.pcdscbGiven)?;
writeln!(out, "pcdscdGiven: {},", prepared.model.pcdscdGiven)?;
writeln!(out, "pcitGiven: {},", prepared.model.pcitGiven)?;
writeln!(out, "pnfactorGiven: {},", prepared.model.pnfactorGiven)?;
writeln!(out, "pxjGiven: {},", prepared.model.pxjGiven)?;
writeln!(out, "pvsatGiven: {},", prepared.model.pvsatGiven)?;
writeln!(out, "patGiven: {},", prepared.model.patGiven)?;
writeln!(out, "pa0Given: {},", prepared.model.pa0Given)?;
writeln!(out, "pagsGiven: {},", prepared.model.pagsGiven)?;
writeln!(out, "pa1Given: {},", prepared.model.pa1Given)?;
writeln!(out, "pa2Given: {},", prepared.model.pa2Given)?;
writeln!(out, "pketaGiven: {},", prepared.model.pketaGiven)?;
writeln!(out, "pketacGiven: {},", prepared.model.pketacGiven)?;
writeln!(out, "pnsubGiven: {},", prepared.model.pnsubGiven)?;
writeln!(out, "pndepGiven: {},", prepared.model.pndepGiven)?;
writeln!(out, "pnsdGiven: {},", prepared.model.pnsdGiven)?;
writeln!(out, "pphinGiven: {},", prepared.model.pphinGiven)?;
writeln!(out, "pngateGiven: {},", prepared.model.pngateGiven)?;
writeln!(out, "pgamma1Given: {},", prepared.model.pgamma1Given)?;
writeln!(out, "pgamma2Given: {},", prepared.model.pgamma2Given)?;
writeln!(out, "pvbxGiven: {},", prepared.model.pvbxGiven)?;
writeln!(out, "pvbmGiven: {},", prepared.model.pvbmGiven)?;
writeln!(out, "pxtGiven: {},", prepared.model.pxtGiven)?;
writeln!(out, "pk1Given: {},", prepared.model.pk1Given)?;
writeln!(out, "pkt1Given: {},", prepared.model.pkt1Given)?;
writeln!(out, "pkt1lGiven: {},", prepared.model.pkt1lGiven)?;
writeln!(out, "pkt2Given: {},", prepared.model.pkt2Given)?;
writeln!(out, "pk2Given: {},", prepared.model.pk2Given)?;
writeln!(out, "pk3Given: {},", prepared.model.pk3Given)?;
writeln!(out, "pk3bGiven: {},", prepared.model.pk3bGiven)?;
writeln!(out, "pw0Given: {},", prepared.model.pw0Given)?;
writeln!(out, "pdvtp0Given: {},", prepared.model.pdvtp0Given)?;
writeln!(out, "pdvtp1Given: {},", prepared.model.pdvtp1Given)?;
writeln!(out, "pdvtp2Given: {},", prepared.model.pdvtp2Given)?;
writeln!(out, "pdvtp3Given: {},", prepared.model.pdvtp3Given)?;
writeln!(out, "pdvtp4Given: {},", prepared.model.pdvtp4Given)?;
writeln!(out, "pdvtp5Given: {},", prepared.model.pdvtp5Given)?;
writeln!(out, "plpe0Given: {},", prepared.model.plpe0Given)?;
writeln!(out, "plpebGiven: {},", prepared.model.plpebGiven)?;
writeln!(out, "pdvt0Given: {},", prepared.model.pdvt0Given)?;
writeln!(out, "pdvt1Given: {},", prepared.model.pdvt1Given)?;
writeln!(out, "pdvt2Given: {},", prepared.model.pdvt2Given)?;
writeln!(out, "pdvt0wGiven: {},", prepared.model.pdvt0wGiven)?;
writeln!(out, "pdvt1wGiven: {},", prepared.model.pdvt1wGiven)?;
writeln!(out, "pdvt2wGiven: {},", prepared.model.pdvt2wGiven)?;
writeln!(out, "pdroutGiven: {},", prepared.model.pdroutGiven)?;
writeln!(out, "pdsubGiven: {},", prepared.model.pdsubGiven)?;
writeln!(out, "pvth0Given: {},", prepared.model.pvth0Given)?;
writeln!(out, "puaGiven: {},", prepared.model.puaGiven)?;
writeln!(out, "pua1Given: {},", prepared.model.pua1Given)?;
writeln!(out, "pubGiven: {},", prepared.model.pubGiven)?;
writeln!(out, "pub1Given: {},", prepared.model.pub1Given)?;
writeln!(out, "pucGiven: {},", prepared.model.pucGiven)?;
writeln!(out, "puc1Given: {},", prepared.model.puc1Given)?;
writeln!(out, "pudGiven: {},", prepared.model.pudGiven)?;
writeln!(out, "pud1Given: {},", prepared.model.pud1Given)?;
writeln!(out, "pupGiven: {},", prepared.model.pupGiven)?;
writeln!(out, "plpGiven: {},", prepared.model.plpGiven)?;
writeln!(out, "pu0Given: {},", prepared.model.pu0Given)?;
writeln!(out, "peuGiven: {},", prepared.model.peuGiven)?;
writeln!(out, "pucsGiven: {},", prepared.model.pucsGiven)?;
writeln!(out, "puteGiven: {},", prepared.model.puteGiven)?;
writeln!(out, "pucsteGiven: {},", prepared.model.pucsteGiven)?;
writeln!(out, "pvoffGiven: {},", prepared.model.pvoffGiven)?;
writeln!(out, "ptvoffGiven: {},", prepared.model.ptvoffGiven)?;
writeln!(out, "ptnfactorGiven: {},", prepared.model.ptnfactorGiven)?;
writeln!(out, "pteta0Given: {},", prepared.model.pteta0Given)?;
writeln!(out, "ptvoffcvGiven: {},", prepared.model.ptvoffcvGiven)?;
writeln!(out, "pminvGiven: {},", prepared.model.pminvGiven)?;
writeln!(out, "pminvcvGiven: {},", prepared.model.pminvcvGiven)?;
writeln!(out, "prdswGiven: {},", prepared.model.prdswGiven)?;
writeln!(out, "prswGiven: {},", prepared.model.prswGiven)?;
writeln!(out, "prdwGiven: {},", prepared.model.prdwGiven)?;
writeln!(out, "pprwgGiven: {},", prepared.model.pprwgGiven)?;
writeln!(out, "pprwbGiven: {},", prepared.model.pprwbGiven)?;
writeln!(out, "pprtGiven: {},", prepared.model.pprtGiven)?;
writeln!(out, "peta0Given: {},", prepared.model.peta0Given)?;
writeln!(out, "petabGiven: {},", prepared.model.petabGiven)?;
writeln!(out, "ppclmGiven: {},", prepared.model.ppclmGiven)?;
writeln!(out, "ppdibl1Given: {},", prepared.model.ppdibl1Given)?;
writeln!(out, "ppdibl2Given: {},", prepared.model.ppdibl2Given)?;
writeln!(out, "ppdiblbGiven: {},", prepared.model.ppdiblbGiven)?;
writeln!(out, "pfproutGiven: {},", prepared.model.pfproutGiven)?;
writeln!(out, "ppditsGiven: {},", prepared.model.ppditsGiven)?;
writeln!(out, "ppditsdGiven: {},", prepared.model.ppditsdGiven)?;
writeln!(out, "ppscbe1Given: {},", prepared.model.ppscbe1Given)?;
writeln!(out, "ppscbe2Given: {},", prepared.model.ppscbe2Given)?;
writeln!(out, "ppvagGiven: {},", prepared.model.ppvagGiven)?;
writeln!(out, "pdeltaGiven: {},", prepared.model.pdeltaGiven)?;
writeln!(out, "pwrGiven: {},", prepared.model.pwrGiven)?;
writeln!(out, "pdwgGiven: {},", prepared.model.pdwgGiven)?;
writeln!(out, "pdwbGiven: {},", prepared.model.pdwbGiven)?;
writeln!(out, "pb0Given: {},", prepared.model.pb0Given)?;
writeln!(out, "pb1Given: {},", prepared.model.pb1Given)?;
writeln!(out, "palpha0Given: {},", prepared.model.palpha0Given)?;
writeln!(out, "palpha1Given: {},", prepared.model.palpha1Given)?;
writeln!(out, "pbeta0Given: {},", prepared.model.pbeta0Given)?;
writeln!(out, "pvfbGiven: {},", prepared.model.pvfbGiven)?;
writeln!(out, "pagidlGiven: {},", prepared.model.pagidlGiven)?;
writeln!(out, "pbgidlGiven: {},", prepared.model.pbgidlGiven)?;
writeln!(out, "pcgidlGiven: {},", prepared.model.pcgidlGiven)?;
writeln!(out, "pegidlGiven: {},", prepared.model.pegidlGiven)?;
writeln!(out, "pfgidlGiven: {},", prepared.model.pfgidlGiven)?;
writeln!(out, "pkgidlGiven: {},", prepared.model.pkgidlGiven)?;
writeln!(out, "prgidlGiven: {},", prepared.model.prgidlGiven)?;
writeln!(out, "pagislGiven: {},", prepared.model.pagislGiven)?;
writeln!(out, "pbgislGiven: {},", prepared.model.pbgislGiven)?;
writeln!(out, "pcgislGiven: {},", prepared.model.pcgislGiven)?;
writeln!(out, "pegislGiven: {},", prepared.model.pegislGiven)?;
writeln!(out, "pfgislGiven: {},", prepared.model.pfgislGiven)?;
writeln!(out, "pkgislGiven: {},", prepared.model.pkgislGiven)?;
writeln!(out, "prgislGiven: {},", prepared.model.prgislGiven)?;
writeln!(out, "paigcGiven: {},", prepared.model.paigcGiven)?;
writeln!(out, "pbigcGiven: {},", prepared.model.pbigcGiven)?;
writeln!(out, "pcigcGiven: {},", prepared.model.pcigcGiven)?;
writeln!(out, "paigsdGiven: {},", prepared.model.paigsdGiven)?;
writeln!(out, "pbigsdGiven: {},", prepared.model.pbigsdGiven)?;
writeln!(out, "pcigsdGiven: {},", prepared.model.pcigsdGiven)?;
writeln!(out, "paigsGiven: {},", prepared.model.paigsGiven)?;
writeln!(out, "pbigsGiven: {},", prepared.model.pbigsGiven)?;
writeln!(out, "pcigsGiven: {},", prepared.model.pcigsGiven)?;
writeln!(out, "paigdGiven: {},", prepared.model.paigdGiven)?;
writeln!(out, "pbigdGiven: {},", prepared.model.pbigdGiven)?;
writeln!(out, "pcigdGiven: {},", prepared.model.pcigdGiven)?;
writeln!(out, "paigbaccGiven: {},", prepared.model.paigbaccGiven)?;
writeln!(out, "pbigbaccGiven: {},", prepared.model.pbigbaccGiven)?;
writeln!(out, "pcigbaccGiven: {},", prepared.model.pcigbaccGiven)?;
writeln!(out, "paigbinvGiven: {},", prepared.model.paigbinvGiven)?;
writeln!(out, "pbigbinvGiven: {},", prepared.model.pbigbinvGiven)?;
writeln!(out, "pcigbinvGiven: {},", prepared.model.pcigbinvGiven)?;
writeln!(out, "pnigcGiven: {},", prepared.model.pnigcGiven)?;
writeln!(out, "pnigbinvGiven: {},", prepared.model.pnigbinvGiven)?;
writeln!(out, "pnigbaccGiven: {},", prepared.model.pnigbaccGiven)?;
writeln!(out, "pntoxGiven: {},", prepared.model.pntoxGiven)?;
writeln!(out, "peigbinvGiven: {},", prepared.model.peigbinvGiven)?;
writeln!(out, "ppigcdGiven: {},", prepared.model.ppigcdGiven)?;
writeln!(out, "ppoxedgeGiven: {},", prepared.model.ppoxedgeGiven)?;
writeln!(out, "pxrcrg1Given: {},", prepared.model.pxrcrg1Given)?;
writeln!(out, "pxrcrg2Given: {},", prepared.model.pxrcrg2Given)?;
writeln!(out, "plambdaGiven: {},", prepared.model.plambdaGiven)?;
writeln!(out, "pvtlGiven: {},", prepared.model.pvtlGiven)?;
writeln!(out, "pxnGiven: {},", prepared.model.pxnGiven)?;
writeln!(out, "pvfbsdoffGiven: {},", prepared.model.pvfbsdoffGiven)?;
writeln!(out, "ptvfbsdoffGiven: {},", prepared.model.ptvfbsdoffGiven)?;
writeln!(out, "pcgslGiven: {},", prepared.model.pcgslGiven)?;
writeln!(out, "pcgdlGiven: {},", prepared.model.pcgdlGiven)?;
writeln!(out, "pckappasGiven: {},", prepared.model.pckappasGiven)?;
writeln!(out, "pckappadGiven: {},", prepared.model.pckappadGiven)?;
writeln!(out, "pcfGiven: {},", prepared.model.pcfGiven)?;
writeln!(out, "pclcGiven: {},", prepared.model.pclcGiven)?;
writeln!(out, "pcleGiven: {},", prepared.model.pcleGiven)?;
writeln!(out, "pvfbcvGiven: {},", prepared.model.pvfbcvGiven)?;
writeln!(out, "pnoffGiven: {},", prepared.model.pnoffGiven)?;
writeln!(out, "pvoffcvGiven: {},", prepared.model.pvoffcvGiven)?;
writeln!(out, "pacdeGiven: {},", prepared.model.pacdeGiven)?;
writeln!(out, "pmoinGiven: {},", prepared.model.pmoinGiven)?;
writeln!(out, "useFringeGiven: {},", prepared.model.useFringeGiven)?;
writeln!(out, "tnomGiven: {},", prepared.model.tnomGiven)?;
writeln!(out, "cgsoGiven: {},", prepared.model.cgsoGiven)?;
writeln!(out, "cgdoGiven: {},", prepared.model.cgdoGiven)?;
writeln!(out, "cgboGiven: {},", prepared.model.cgboGiven)?;
writeln!(out, "xpartGiven: {},", prepared.model.xpartGiven)?;
writeln!(out, "sheetResistanceGiven: {},", prepared.model.sheetResistanceGiven)?;
writeln!(out, "SjctSatCurDensityGiven: {},", prepared.model.SjctSatCurDensityGiven)?;
writeln!(out, "SjctSidewallSatCurDensityGiven: {},", prepared.model.SjctSidewallSatCurDensityGiven)?;
writeln!(out, "SjctGateSidewallSatCurDensityGiven: {},", prepared.model.SjctGateSidewallSatCurDensityGiven)?;
writeln!(out, "SbulkJctPotentialGiven: {},", prepared.model.SbulkJctPotentialGiven)?;
writeln!(out, "SbulkJctBotGradingCoeffGiven: {},", prepared.model.SbulkJctBotGradingCoeffGiven)?;
writeln!(out, "SsidewallJctPotentialGiven: {},", prepared.model.SsidewallJctPotentialGiven)?;
writeln!(out, "SGatesidewallJctPotentialGiven: {},", prepared.model.SGatesidewallJctPotentialGiven)?;
writeln!(out, "SbulkJctSideGradingCoeffGiven: {},", prepared.model.SbulkJctSideGradingCoeffGiven)?;
writeln!(out, "SunitAreaJctCapGiven: {},", prepared.model.SunitAreaJctCapGiven)?;
writeln!(out, "SunitLengthSidewallJctCapGiven: {},", prepared.model.SunitLengthSidewallJctCapGiven)?;
writeln!(out, "SbulkJctGateSideGradingCoeffGiven: {},", prepared.model.SbulkJctGateSideGradingCoeffGiven)?;
writeln!(out, "SunitLengthGateSidewallJctCapGiven: {},", prepared.model.SunitLengthGateSidewallJctCapGiven)?;
writeln!(out, "SjctEmissionCoeffGiven: {},", prepared.model.SjctEmissionCoeffGiven)?;
writeln!(out, "SjctTempExponentGiven: {},", prepared.model.SjctTempExponentGiven)?;
writeln!(out, "DjctSatCurDensityGiven: {},", prepared.model.DjctSatCurDensityGiven)?;
writeln!(out, "DjctSidewallSatCurDensityGiven: {},", prepared.model.DjctSidewallSatCurDensityGiven)?;
writeln!(out, "DjctGateSidewallSatCurDensityGiven: {},", prepared.model.DjctGateSidewallSatCurDensityGiven)?;
writeln!(out, "DbulkJctPotentialGiven: {},", prepared.model.DbulkJctPotentialGiven)?;
writeln!(out, "DbulkJctBotGradingCoeffGiven: {},", prepared.model.DbulkJctBotGradingCoeffGiven)?;
writeln!(out, "DsidewallJctPotentialGiven: {},", prepared.model.DsidewallJctPotentialGiven)?;
writeln!(out, "DGatesidewallJctPotentialGiven: {},", prepared.model.DGatesidewallJctPotentialGiven)?;
writeln!(out, "DbulkJctSideGradingCoeffGiven: {},", prepared.model.DbulkJctSideGradingCoeffGiven)?;
writeln!(out, "DunitAreaJctCapGiven: {},", prepared.model.DunitAreaJctCapGiven)?;
writeln!(out, "DunitLengthSidewallJctCapGiven: {},", prepared.model.DunitLengthSidewallJctCapGiven)?;
writeln!(out, "DbulkJctGateSideGradingCoeffGiven: {},", prepared.model.DbulkJctGateSideGradingCoeffGiven)?;
writeln!(out, "DunitLengthGateSidewallJctCapGiven: {},", prepared.model.DunitLengthGateSidewallJctCapGiven)?;
writeln!(out, "DjctEmissionCoeffGiven: {},", prepared.model.DjctEmissionCoeffGiven)?;
writeln!(out, "DjctTempExponentGiven: {},", prepared.model.DjctTempExponentGiven)?;
writeln!(out, "oxideTrapDensityAGiven: {},", prepared.model.oxideTrapDensityAGiven)?;
writeln!(out, "oxideTrapDensityBGiven: {},", prepared.model.oxideTrapDensityBGiven)?;
writeln!(out, "oxideTrapDensityCGiven: {},", prepared.model.oxideTrapDensityCGiven)?;
writeln!(out, "emGiven: {},", prepared.model.emGiven)?;
writeln!(out, "efGiven: {},", prepared.model.efGiven)?;
writeln!(out, "afGiven: {},", prepared.model.afGiven)?;
writeln!(out, "kfGiven: {},", prepared.model.kfGiven)?;
writeln!(out, "vgsMaxGiven: {},", prepared.model.vgsMaxGiven)?;
writeln!(out, "vgdMaxGiven: {},", prepared.model.vgdMaxGiven)?;
writeln!(out, "vgbMaxGiven: {},", prepared.model.vgbMaxGiven)?;
writeln!(out, "vdsMaxGiven: {},", prepared.model.vdsMaxGiven)?;
writeln!(out, "vbsMaxGiven: {},", prepared.model.vbsMaxGiven)?;
writeln!(out, "vbdMaxGiven: {},", prepared.model.vbdMaxGiven)?;
writeln!(out, "vgsrMaxGiven: {},", prepared.model.vgsrMaxGiven)?;
writeln!(out, "vgdrMaxGiven: {},", prepared.model.vgdrMaxGiven)?;
writeln!(out, "vgbrMaxGiven: {},", prepared.model.vgbrMaxGiven)?;
writeln!(out, "vbsrMaxGiven: {},", prepared.model.vbsrMaxGiven)?;
writeln!(out, "vbdrMaxGiven: {},", prepared.model.vbdrMaxGiven)?;
writeln!(out, "LintGiven: {},", prepared.model.LintGiven)?;
writeln!(out, "LlGiven: {},", prepared.model.LlGiven)?;
writeln!(out, "LlcGiven: {},", prepared.model.LlcGiven)?;
writeln!(out, "LlnGiven: {},", prepared.model.LlnGiven)?;
writeln!(out, "LwGiven: {},", prepared.model.LwGiven)?;
writeln!(out, "LwcGiven: {},", prepared.model.LwcGiven)?;
writeln!(out, "LwnGiven: {},", prepared.model.LwnGiven)?;
writeln!(out, "LwlGiven: {},", prepared.model.LwlGiven)?;
writeln!(out, "LwlcGiven: {},", prepared.model.LwlcGiven)?;
writeln!(out, "LminGiven: {},", prepared.model.LminGiven)?;
writeln!(out, "LmaxGiven: {},", prepared.model.LmaxGiven)?;
writeln!(out, "WintGiven: {},", prepared.model.WintGiven)?;
writeln!(out, "WlGiven: {},", prepared.model.WlGiven)?;
writeln!(out, "WlcGiven: {},", prepared.model.WlcGiven)?;
writeln!(out, "WlnGiven: {},", prepared.model.WlnGiven)?;
writeln!(out, "WwGiven: {},", prepared.model.WwGiven)?;
writeln!(out, "WwcGiven: {},", prepared.model.WwcGiven)?;
writeln!(out, "WwnGiven: {},", prepared.model.WwnGiven)?;
writeln!(out, "WwlGiven: {},", prepared.model.WwlGiven)?;
writeln!(out, "WwlcGiven: {},", prepared.model.WwlcGiven)?;
writeln!(out, "WminGiven: {},", prepared.model.WminGiven)?;
writeln!(out, "WmaxGiven: {},", prepared.model.WmaxGiven)?;
writeln!(out, "sarefGiven: {},", prepared.model.sarefGiven)?;
writeln!(out, "sbrefGiven: {},", prepared.model.sbrefGiven)?;
writeln!(out, "wlodGiven: {},", prepared.model.wlodGiven)?;
writeln!(out, "ku0Given: {},", prepared.model.ku0Given)?;
writeln!(out, "kvsatGiven: {},", prepared.model.kvsatGiven)?;
writeln!(out, "kvth0Given: {},", prepared.model.kvth0Given)?;
writeln!(out, "tku0Given: {},", prepared.model.tku0Given)?;
writeln!(out, "llodku0Given: {},", prepared.model.llodku0Given)?;
writeln!(out, "wlodku0Given: {},", prepared.model.wlodku0Given)?;
writeln!(out, "llodvthGiven: {},", prepared.model.llodvthGiven)?;
writeln!(out, "wlodvthGiven: {},", prepared.model.wlodvthGiven)?;
writeln!(out, "lku0Given: {},", prepared.model.lku0Given)?;
writeln!(out, "wku0Given: {},", prepared.model.wku0Given)?;
writeln!(out, "pku0Given: {},", prepared.model.pku0Given)?;
writeln!(out, "lkvth0Given: {},", prepared.model.lkvth0Given)?;
writeln!(out, "wkvth0Given: {},", prepared.model.wkvth0Given)?;
writeln!(out, "pkvth0Given: {},", prepared.model.pkvth0Given)?;
writeln!(out, "stk2Given: {},", prepared.model.stk2Given)?;
writeln!(out, "lodk2Given: {},", prepared.model.lodk2Given)?;
writeln!(out, "steta0Given: {},", prepared.model.steta0Given)?;
writeln!(out, "lodeta0Given: {},", prepared.model.lodeta0Given)?;
writeln!(out, "webGiven: {},", prepared.model.webGiven)?;
writeln!(out, "wecGiven: {},", prepared.model.wecGiven)?;
writeln!(out, "kvth0weGiven: {},", prepared.model.kvth0weGiven)?;
writeln!(out, "k2weGiven: {},", prepared.model.k2weGiven)?;
writeln!(out, "ku0weGiven: {},", prepared.model.ku0weGiven)?;
writeln!(out, "screfGiven: {},", prepared.model.screfGiven)?;
writeln!(out, "wpemodGiven: {},", prepared.model.wpemodGiven)?;
writeln!(out, "lkvth0weGiven: {},", prepared.model.lkvth0weGiven)?;
writeln!(out, "lk2weGiven: {},", prepared.model.lk2weGiven)?;
writeln!(out, "lku0weGiven: {},", prepared.model.lku0weGiven)?;
writeln!(out, "wkvth0weGiven: {},", prepared.model.wkvth0weGiven)?;
writeln!(out, "wk2weGiven: {},", prepared.model.wk2weGiven)?;
writeln!(out, "wku0weGiven: {},", prepared.model.wku0weGiven)?;
writeln!(out, "pkvth0weGiven: {},", prepared.model.pkvth0weGiven)?;
writeln!(out, "pk2weGiven: {},", prepared.model.pk2weGiven)?;
writeln!(out, "pku0weGiven: {},", prepared.model.pku0weGiven)?;
writeln!(out, "gidlclampGiven: {},", prepared.model.gidlclampGiven)?;
writeln!(out, "idovvdscGiven: {},", prepared.model.idovvdscGiven)?;
            writeln!(out, "}};")?;
            writeln!(out, "static SIZE: crate::devices::bsim4::fields::SizeParam = crate::devices::bsim4::fields::SizeParam {{")?;
            writeln!(out, "Width: f64::from_bits(0x{:016x}),", prepared.size.Width.to_bits())?;
writeln!(out, "Length: f64::from_bits(0x{:016x}),", prepared.size.Length.to_bits())?;
writeln!(out, "NFinger: f64::from_bits(0x{:016x}),", prepared.size.NFinger.to_bits())?;
writeln!(out, "cdsc: f64::from_bits(0x{:016x}),", prepared.size.cdsc.to_bits())?;
writeln!(out, "cdscb: f64::from_bits(0x{:016x}),", prepared.size.cdscb.to_bits())?;
writeln!(out, "cdscd: f64::from_bits(0x{:016x}),", prepared.size.cdscd.to_bits())?;
writeln!(out, "cit: f64::from_bits(0x{:016x}),", prepared.size.cit.to_bits())?;
writeln!(out, "nfactor: f64::from_bits(0x{:016x}),", prepared.size.nfactor.to_bits())?;
writeln!(out, "xj: f64::from_bits(0x{:016x}),", prepared.size.xj.to_bits())?;
writeln!(out, "vsat: f64::from_bits(0x{:016x}),", prepared.size.vsat.to_bits())?;
writeln!(out, "at: f64::from_bits(0x{:016x}),", prepared.size.at.to_bits())?;
writeln!(out, "a0: f64::from_bits(0x{:016x}),", prepared.size.a0.to_bits())?;
writeln!(out, "ags: f64::from_bits(0x{:016x}),", prepared.size.ags.to_bits())?;
writeln!(out, "a1: f64::from_bits(0x{:016x}),", prepared.size.a1.to_bits())?;
writeln!(out, "a2: f64::from_bits(0x{:016x}),", prepared.size.a2.to_bits())?;
writeln!(out, "keta: f64::from_bits(0x{:016x}),", prepared.size.keta.to_bits())?;
writeln!(out, "ketac: f64::from_bits(0x{:016x}),", prepared.size.ketac.to_bits())?;
writeln!(out, "nsub: f64::from_bits(0x{:016x}),", prepared.size.nsub.to_bits())?;
writeln!(out, "ndep: f64::from_bits(0x{:016x}),", prepared.size.ndep.to_bits())?;
writeln!(out, "nsd: f64::from_bits(0x{:016x}),", prepared.size.nsd.to_bits())?;
writeln!(out, "phin: f64::from_bits(0x{:016x}),", prepared.size.phin.to_bits())?;
writeln!(out, "ngate: f64::from_bits(0x{:016x}),", prepared.size.ngate.to_bits())?;
writeln!(out, "gamma1: f64::from_bits(0x{:016x}),", prepared.size.gamma1.to_bits())?;
writeln!(out, "gamma2: f64::from_bits(0x{:016x}),", prepared.size.gamma2.to_bits())?;
writeln!(out, "vbx: f64::from_bits(0x{:016x}),", prepared.size.vbx.to_bits())?;
writeln!(out, "vbi: f64::from_bits(0x{:016x}),", prepared.size.vbi.to_bits())?;
writeln!(out, "vbm: f64::from_bits(0x{:016x}),", prepared.size.vbm.to_bits())?;
writeln!(out, "xt: f64::from_bits(0x{:016x}),", prepared.size.xt.to_bits())?;
writeln!(out, "phi: f64::from_bits(0x{:016x}),", prepared.size.phi.to_bits())?;
writeln!(out, "litl: f64::from_bits(0x{:016x}),", prepared.size.litl.to_bits())?;
writeln!(out, "k1: f64::from_bits(0x{:016x}),", prepared.size.k1.to_bits())?;
writeln!(out, "kt1: f64::from_bits(0x{:016x}),", prepared.size.kt1.to_bits())?;
writeln!(out, "kt1l: f64::from_bits(0x{:016x}),", prepared.size.kt1l.to_bits())?;
writeln!(out, "kt2: f64::from_bits(0x{:016x}),", prepared.size.kt2.to_bits())?;
writeln!(out, "k2: f64::from_bits(0x{:016x}),", prepared.size.k2.to_bits())?;
writeln!(out, "k3: f64::from_bits(0x{:016x}),", prepared.size.k3.to_bits())?;
writeln!(out, "k3b: f64::from_bits(0x{:016x}),", prepared.size.k3b.to_bits())?;
writeln!(out, "w0: f64::from_bits(0x{:016x}),", prepared.size.w0.to_bits())?;
writeln!(out, "dvtp0: f64::from_bits(0x{:016x}),", prepared.size.dvtp0.to_bits())?;
writeln!(out, "dvtp1: f64::from_bits(0x{:016x}),", prepared.size.dvtp1.to_bits())?;
writeln!(out, "dvtp2: f64::from_bits(0x{:016x}),", prepared.size.dvtp2.to_bits())?;
writeln!(out, "dvtp3: f64::from_bits(0x{:016x}),", prepared.size.dvtp3.to_bits())?;
writeln!(out, "dvtp4: f64::from_bits(0x{:016x}),", prepared.size.dvtp4.to_bits())?;
writeln!(out, "dvtp5: f64::from_bits(0x{:016x}),", prepared.size.dvtp5.to_bits())?;
writeln!(out, "lpe0: f64::from_bits(0x{:016x}),", prepared.size.lpe0.to_bits())?;
writeln!(out, "lpeb: f64::from_bits(0x{:016x}),", prepared.size.lpeb.to_bits())?;
writeln!(out, "dvt0: f64::from_bits(0x{:016x}),", prepared.size.dvt0.to_bits())?;
writeln!(out, "dvt1: f64::from_bits(0x{:016x}),", prepared.size.dvt1.to_bits())?;
writeln!(out, "dvt2: f64::from_bits(0x{:016x}),", prepared.size.dvt2.to_bits())?;
writeln!(out, "dvt0w: f64::from_bits(0x{:016x}),", prepared.size.dvt0w.to_bits())?;
writeln!(out, "dvt1w: f64::from_bits(0x{:016x}),", prepared.size.dvt1w.to_bits())?;
writeln!(out, "dvt2w: f64::from_bits(0x{:016x}),", prepared.size.dvt2w.to_bits())?;
writeln!(out, "drout: f64::from_bits(0x{:016x}),", prepared.size.drout.to_bits())?;
writeln!(out, "dsub: f64::from_bits(0x{:016x}),", prepared.size.dsub.to_bits())?;
writeln!(out, "vth0: f64::from_bits(0x{:016x}),", prepared.size.vth0.to_bits())?;
writeln!(out, "ua: f64::from_bits(0x{:016x}),", prepared.size.ua.to_bits())?;
writeln!(out, "ua1: f64::from_bits(0x{:016x}),", prepared.size.ua1.to_bits())?;
writeln!(out, "ub: f64::from_bits(0x{:016x}),", prepared.size.ub.to_bits())?;
writeln!(out, "ub1: f64::from_bits(0x{:016x}),", prepared.size.ub1.to_bits())?;
writeln!(out, "uc: f64::from_bits(0x{:016x}),", prepared.size.uc.to_bits())?;
writeln!(out, "uc1: f64::from_bits(0x{:016x}),", prepared.size.uc1.to_bits())?;
writeln!(out, "ud: f64::from_bits(0x{:016x}),", prepared.size.ud.to_bits())?;
writeln!(out, "ud1: f64::from_bits(0x{:016x}),", prepared.size.ud1.to_bits())?;
writeln!(out, "up: f64::from_bits(0x{:016x}),", prepared.size.up.to_bits())?;
writeln!(out, "lp: f64::from_bits(0x{:016x}),", prepared.size.lp.to_bits())?;
writeln!(out, "u0: f64::from_bits(0x{:016x}),", prepared.size.u0.to_bits())?;
writeln!(out, "eu: f64::from_bits(0x{:016x}),", prepared.size.eu.to_bits())?;
writeln!(out, "ucs: f64::from_bits(0x{:016x}),", prepared.size.ucs.to_bits())?;
writeln!(out, "ute: f64::from_bits(0x{:016x}),", prepared.size.ute.to_bits())?;
writeln!(out, "ucste: f64::from_bits(0x{:016x}),", prepared.size.ucste.to_bits())?;
writeln!(out, "voff: f64::from_bits(0x{:016x}),", prepared.size.voff.to_bits())?;
writeln!(out, "tvoff: f64::from_bits(0x{:016x}),", prepared.size.tvoff.to_bits())?;
writeln!(out, "tnfactor: f64::from_bits(0x{:016x}),", prepared.size.tnfactor.to_bits())?;
writeln!(out, "teta0: f64::from_bits(0x{:016x}),", prepared.size.teta0.to_bits())?;
writeln!(out, "tvoffcv: f64::from_bits(0x{:016x}),", prepared.size.tvoffcv.to_bits())?;
writeln!(out, "minv: f64::from_bits(0x{:016x}),", prepared.size.minv.to_bits())?;
writeln!(out, "minvcv: f64::from_bits(0x{:016x}),", prepared.size.minvcv.to_bits())?;
writeln!(out, "vfb: f64::from_bits(0x{:016x}),", prepared.size.vfb.to_bits())?;
writeln!(out, "delta: f64::from_bits(0x{:016x}),", prepared.size.delta.to_bits())?;
writeln!(out, "rdsw: f64::from_bits(0x{:016x}),", prepared.size.rdsw.to_bits())?;
writeln!(out, "rds0: f64::from_bits(0x{:016x}),", prepared.size.rds0.to_bits())?;
writeln!(out, "rs0: f64::from_bits(0x{:016x}),", prepared.size.rs0.to_bits())?;
writeln!(out, "rd0: f64::from_bits(0x{:016x}),", prepared.size.rd0.to_bits())?;
writeln!(out, "rsw: f64::from_bits(0x{:016x}),", prepared.size.rsw.to_bits())?;
writeln!(out, "rdw: f64::from_bits(0x{:016x}),", prepared.size.rdw.to_bits())?;
writeln!(out, "prwg: f64::from_bits(0x{:016x}),", prepared.size.prwg.to_bits())?;
writeln!(out, "prwb: f64::from_bits(0x{:016x}),", prepared.size.prwb.to_bits())?;
writeln!(out, "prt: f64::from_bits(0x{:016x}),", prepared.size.prt.to_bits())?;
writeln!(out, "eta0: f64::from_bits(0x{:016x}),", prepared.size.eta0.to_bits())?;
writeln!(out, "etab: f64::from_bits(0x{:016x}),", prepared.size.etab.to_bits())?;
writeln!(out, "pclm: f64::from_bits(0x{:016x}),", prepared.size.pclm.to_bits())?;
writeln!(out, "pdibl1: f64::from_bits(0x{:016x}),", prepared.size.pdibl1.to_bits())?;
writeln!(out, "pdibl2: f64::from_bits(0x{:016x}),", prepared.size.pdibl2.to_bits())?;
writeln!(out, "pdiblb: f64::from_bits(0x{:016x}),", prepared.size.pdiblb.to_bits())?;
writeln!(out, "fprout: f64::from_bits(0x{:016x}),", prepared.size.fprout.to_bits())?;
writeln!(out, "pdits: f64::from_bits(0x{:016x}),", prepared.size.pdits.to_bits())?;
writeln!(out, "pditsd: f64::from_bits(0x{:016x}),", prepared.size.pditsd.to_bits())?;
writeln!(out, "pscbe1: f64::from_bits(0x{:016x}),", prepared.size.pscbe1.to_bits())?;
writeln!(out, "pscbe2: f64::from_bits(0x{:016x}),", prepared.size.pscbe2.to_bits())?;
writeln!(out, "pvag: f64::from_bits(0x{:016x}),", prepared.size.pvag.to_bits())?;
writeln!(out, "wr: f64::from_bits(0x{:016x}),", prepared.size.wr.to_bits())?;
writeln!(out, "dwg: f64::from_bits(0x{:016x}),", prepared.size.dwg.to_bits())?;
writeln!(out, "dwb: f64::from_bits(0x{:016x}),", prepared.size.dwb.to_bits())?;
writeln!(out, "b0: f64::from_bits(0x{:016x}),", prepared.size.b0.to_bits())?;
writeln!(out, "b1: f64::from_bits(0x{:016x}),", prepared.size.b1.to_bits())?;
writeln!(out, "alpha0: f64::from_bits(0x{:016x}),", prepared.size.alpha0.to_bits())?;
writeln!(out, "alpha1: f64::from_bits(0x{:016x}),", prepared.size.alpha1.to_bits())?;
writeln!(out, "beta0: f64::from_bits(0x{:016x}),", prepared.size.beta0.to_bits())?;
writeln!(out, "agidl: f64::from_bits(0x{:016x}),", prepared.size.agidl.to_bits())?;
writeln!(out, "bgidl: f64::from_bits(0x{:016x}),", prepared.size.bgidl.to_bits())?;
writeln!(out, "cgidl: f64::from_bits(0x{:016x}),", prepared.size.cgidl.to_bits())?;
writeln!(out, "egidl: f64::from_bits(0x{:016x}),", prepared.size.egidl.to_bits())?;
writeln!(out, "fgidl: f64::from_bits(0x{:016x}),", prepared.size.fgidl.to_bits())?;
writeln!(out, "kgidl: f64::from_bits(0x{:016x}),", prepared.size.kgidl.to_bits())?;
writeln!(out, "rgidl: f64::from_bits(0x{:016x}),", prepared.size.rgidl.to_bits())?;
writeln!(out, "agisl: f64::from_bits(0x{:016x}),", prepared.size.agisl.to_bits())?;
writeln!(out, "bgisl: f64::from_bits(0x{:016x}),", prepared.size.bgisl.to_bits())?;
writeln!(out, "cgisl: f64::from_bits(0x{:016x}),", prepared.size.cgisl.to_bits())?;
writeln!(out, "egisl: f64::from_bits(0x{:016x}),", prepared.size.egisl.to_bits())?;
writeln!(out, "fgisl: f64::from_bits(0x{:016x}),", prepared.size.fgisl.to_bits())?;
writeln!(out, "kgisl: f64::from_bits(0x{:016x}),", prepared.size.kgisl.to_bits())?;
writeln!(out, "rgisl: f64::from_bits(0x{:016x}),", prepared.size.rgisl.to_bits())?;
writeln!(out, "aigc: f64::from_bits(0x{:016x}),", prepared.size.aigc.to_bits())?;
writeln!(out, "bigc: f64::from_bits(0x{:016x}),", prepared.size.bigc.to_bits())?;
writeln!(out, "cigc: f64::from_bits(0x{:016x}),", prepared.size.cigc.to_bits())?;
writeln!(out, "aigsd: f64::from_bits(0x{:016x}),", prepared.size.aigsd.to_bits())?;
writeln!(out, "bigsd: f64::from_bits(0x{:016x}),", prepared.size.bigsd.to_bits())?;
writeln!(out, "cigsd: f64::from_bits(0x{:016x}),", prepared.size.cigsd.to_bits())?;
writeln!(out, "aigs: f64::from_bits(0x{:016x}),", prepared.size.aigs.to_bits())?;
writeln!(out, "bigs: f64::from_bits(0x{:016x}),", prepared.size.bigs.to_bits())?;
writeln!(out, "cigs: f64::from_bits(0x{:016x}),", prepared.size.cigs.to_bits())?;
writeln!(out, "aigd: f64::from_bits(0x{:016x}),", prepared.size.aigd.to_bits())?;
writeln!(out, "bigd: f64::from_bits(0x{:016x}),", prepared.size.bigd.to_bits())?;
writeln!(out, "cigd: f64::from_bits(0x{:016x}),", prepared.size.cigd.to_bits())?;
writeln!(out, "aigbacc: f64::from_bits(0x{:016x}),", prepared.size.aigbacc.to_bits())?;
writeln!(out, "bigbacc: f64::from_bits(0x{:016x}),", prepared.size.bigbacc.to_bits())?;
writeln!(out, "cigbacc: f64::from_bits(0x{:016x}),", prepared.size.cigbacc.to_bits())?;
writeln!(out, "aigbinv: f64::from_bits(0x{:016x}),", prepared.size.aigbinv.to_bits())?;
writeln!(out, "bigbinv: f64::from_bits(0x{:016x}),", prepared.size.bigbinv.to_bits())?;
writeln!(out, "cigbinv: f64::from_bits(0x{:016x}),", prepared.size.cigbinv.to_bits())?;
writeln!(out, "nigc: f64::from_bits(0x{:016x}),", prepared.size.nigc.to_bits())?;
writeln!(out, "nigbacc: f64::from_bits(0x{:016x}),", prepared.size.nigbacc.to_bits())?;
writeln!(out, "nigbinv: f64::from_bits(0x{:016x}),", prepared.size.nigbinv.to_bits())?;
writeln!(out, "ntox: f64::from_bits(0x{:016x}),", prepared.size.ntox.to_bits())?;
writeln!(out, "eigbinv: f64::from_bits(0x{:016x}),", prepared.size.eigbinv.to_bits())?;
writeln!(out, "pigcd: f64::from_bits(0x{:016x}),", prepared.size.pigcd.to_bits())?;
writeln!(out, "poxedge: f64::from_bits(0x{:016x}),", prepared.size.poxedge.to_bits())?;
writeln!(out, "xrcrg1: f64::from_bits(0x{:016x}),", prepared.size.xrcrg1.to_bits())?;
writeln!(out, "xrcrg2: f64::from_bits(0x{:016x}),", prepared.size.xrcrg2.to_bits())?;
writeln!(out, "lambda: f64::from_bits(0x{:016x}),", prepared.size.lambda.to_bits())?;
writeln!(out, "vtl: f64::from_bits(0x{:016x}),", prepared.size.vtl.to_bits())?;
writeln!(out, "xn: f64::from_bits(0x{:016x}),", prepared.size.xn.to_bits())?;
writeln!(out, "lc: f64::from_bits(0x{:016x}),", prepared.size.lc.to_bits())?;
writeln!(out, "tfactor: f64::from_bits(0x{:016x}),", prepared.size.tfactor.to_bits())?;
writeln!(out, "vfbsdoff: f64::from_bits(0x{:016x}),", prepared.size.vfbsdoff.to_bits())?;
writeln!(out, "tvfbsdoff: f64::from_bits(0x{:016x}),", prepared.size.tvfbsdoff.to_bits())?;
writeln!(out, "ku0: f64::from_bits(0x{:016x}),", prepared.size.ku0.to_bits())?;
writeln!(out, "kvth0: f64::from_bits(0x{:016x}),", prepared.size.kvth0.to_bits())?;
writeln!(out, "ku0temp: f64::from_bits(0x{:016x}),", prepared.size.ku0temp.to_bits())?;
writeln!(out, "rho_ref: f64::from_bits(0x{:016x}),", prepared.size.rho_ref.to_bits())?;
writeln!(out, "inv_od_ref: f64::from_bits(0x{:016x}),", prepared.size.inv_od_ref.to_bits())?;
writeln!(out, "kvth0we: f64::from_bits(0x{:016x}),", prepared.size.kvth0we.to_bits())?;
writeln!(out, "k2we: f64::from_bits(0x{:016x}),", prepared.size.k2we.to_bits())?;
writeln!(out, "ku0we: f64::from_bits(0x{:016x}),", prepared.size.ku0we.to_bits())?;
writeln!(out, "cgsl: f64::from_bits(0x{:016x}),", prepared.size.cgsl.to_bits())?;
writeln!(out, "cgdl: f64::from_bits(0x{:016x}),", prepared.size.cgdl.to_bits())?;
writeln!(out, "ckappas: f64::from_bits(0x{:016x}),", prepared.size.ckappas.to_bits())?;
writeln!(out, "ckappad: f64::from_bits(0x{:016x}),", prepared.size.ckappad.to_bits())?;
writeln!(out, "cf: f64::from_bits(0x{:016x}),", prepared.size.cf.to_bits())?;
writeln!(out, "clc: f64::from_bits(0x{:016x}),", prepared.size.clc.to_bits())?;
writeln!(out, "cle: f64::from_bits(0x{:016x}),", prepared.size.cle.to_bits())?;
writeln!(out, "vfbcv: f64::from_bits(0x{:016x}),", prepared.size.vfbcv.to_bits())?;
writeln!(out, "noff: f64::from_bits(0x{:016x}),", prepared.size.noff.to_bits())?;
writeln!(out, "voffcv: f64::from_bits(0x{:016x}),", prepared.size.voffcv.to_bits())?;
writeln!(out, "acde: f64::from_bits(0x{:016x}),", prepared.size.acde.to_bits())?;
writeln!(out, "moin: f64::from_bits(0x{:016x}),", prepared.size.moin.to_bits())?;
writeln!(out, "dw: f64::from_bits(0x{:016x}),", prepared.size.dw.to_bits())?;
writeln!(out, "dl: f64::from_bits(0x{:016x}),", prepared.size.dl.to_bits())?;
writeln!(out, "leff: f64::from_bits(0x{:016x}),", prepared.size.leff.to_bits())?;
writeln!(out, "weff: f64::from_bits(0x{:016x}),", prepared.size.weff.to_bits())?;
writeln!(out, "dwc: f64::from_bits(0x{:016x}),", prepared.size.dwc.to_bits())?;
writeln!(out, "dlc: f64::from_bits(0x{:016x}),", prepared.size.dlc.to_bits())?;
writeln!(out, "dwj: f64::from_bits(0x{:016x}),", prepared.size.dwj.to_bits())?;
writeln!(out, "leffCV: f64::from_bits(0x{:016x}),", prepared.size.leffCV.to_bits())?;
writeln!(out, "weffCV: f64::from_bits(0x{:016x}),", prepared.size.weffCV.to_bits())?;
writeln!(out, "weffCJ: f64::from_bits(0x{:016x}),", prepared.size.weffCJ.to_bits())?;
writeln!(out, "abulkCVfactor: f64::from_bits(0x{:016x}),", prepared.size.abulkCVfactor.to_bits())?;
writeln!(out, "cgso: f64::from_bits(0x{:016x}),", prepared.size.cgso.to_bits())?;
writeln!(out, "cgdo: f64::from_bits(0x{:016x}),", prepared.size.cgdo.to_bits())?;
writeln!(out, "cgbo: f64::from_bits(0x{:016x}),", prepared.size.cgbo.to_bits())?;
writeln!(out, "u0temp: f64::from_bits(0x{:016x}),", prepared.size.u0temp.to_bits())?;
writeln!(out, "vsattemp: f64::from_bits(0x{:016x}),", prepared.size.vsattemp.to_bits())?;
writeln!(out, "sqrtPhi: f64::from_bits(0x{:016x}),", prepared.size.sqrtPhi.to_bits())?;
writeln!(out, "phis3: f64::from_bits(0x{:016x}),", prepared.size.phis3.to_bits())?;
writeln!(out, "Xdep0: f64::from_bits(0x{:016x}),", prepared.size.Xdep0.to_bits())?;
writeln!(out, "sqrtXdep0: f64::from_bits(0x{:016x}),", prepared.size.sqrtXdep0.to_bits())?;
writeln!(out, "theta0vb0: f64::from_bits(0x{:016x}),", prepared.size.theta0vb0.to_bits())?;
writeln!(out, "thetaRout: f64::from_bits(0x{:016x}),", prepared.size.thetaRout.to_bits())?;
writeln!(out, "mstar: f64::from_bits(0x{:016x}),", prepared.size.mstar.to_bits())?;
writeln!(out, "VgsteffVth: f64::from_bits(0x{:016x}),", prepared.size.VgsteffVth.to_bits())?;
writeln!(out, "mstarcv: f64::from_bits(0x{:016x}),", prepared.size.mstarcv.to_bits())?;
writeln!(out, "voffcbn: f64::from_bits(0x{:016x}),", prepared.size.voffcbn.to_bits())?;
writeln!(out, "voffcbncv: f64::from_bits(0x{:016x}),", prepared.size.voffcbncv.to_bits())?;
writeln!(out, "rdswmin: f64::from_bits(0x{:016x}),", prepared.size.rdswmin.to_bits())?;
writeln!(out, "rdwmin: f64::from_bits(0x{:016x}),", prepared.size.rdwmin.to_bits())?;
writeln!(out, "rswmin: f64::from_bits(0x{:016x}),", prepared.size.rswmin.to_bits())?;
writeln!(out, "vfbsd: f64::from_bits(0x{:016x}),", prepared.size.vfbsd.to_bits())?;
writeln!(out, "cof1: f64::from_bits(0x{:016x}),", prepared.size.cof1.to_bits())?;
writeln!(out, "cof2: f64::from_bits(0x{:016x}),", prepared.size.cof2.to_bits())?;
writeln!(out, "cof3: f64::from_bits(0x{:016x}),", prepared.size.cof3.to_bits())?;
writeln!(out, "cof4: f64::from_bits(0x{:016x}),", prepared.size.cof4.to_bits())?;
writeln!(out, "cdep0: f64::from_bits(0x{:016x}),", prepared.size.cdep0.to_bits())?;
writeln!(out, "ToxRatio: f64::from_bits(0x{:016x}),", prepared.size.ToxRatio.to_bits())?;
writeln!(out, "Aechvb: f64::from_bits(0x{:016x}),", prepared.size.Aechvb.to_bits())?;
writeln!(out, "Bechvb: f64::from_bits(0x{:016x}),", prepared.size.Bechvb.to_bits())?;
writeln!(out, "ToxRatioEdge: f64::from_bits(0x{:016x}),", prepared.size.ToxRatioEdge.to_bits())?;
writeln!(out, "AechvbEdgeS: f64::from_bits(0x{:016x}),", prepared.size.AechvbEdgeS.to_bits())?;
writeln!(out, "AechvbEdgeD: f64::from_bits(0x{:016x}),", prepared.size.AechvbEdgeD.to_bits())?;
writeln!(out, "BechvbEdge: f64::from_bits(0x{:016x}),", prepared.size.BechvbEdge.to_bits())?;
writeln!(out, "ldeb: f64::from_bits(0x{:016x}),", prepared.size.ldeb.to_bits())?;
writeln!(out, "k1ox: f64::from_bits(0x{:016x}),", prepared.size.k1ox.to_bits())?;
writeln!(out, "k2ox: f64::from_bits(0x{:016x}),", prepared.size.k2ox.to_bits())?;
writeln!(out, "vfbzbfactor: f64::from_bits(0x{:016x}),", prepared.size.vfbzbfactor.to_bits())?;
writeln!(out, "dvtp2factor: f64::from_bits(0x{:016x}),", prepared.size.dvtp2factor.to_bits())?;
            writeln!(out, "}};")
        })();
        result.map_err(|_| SpiceError::circuit("cannot emit prepared model"))?;
        Ok(out)
    }
}
