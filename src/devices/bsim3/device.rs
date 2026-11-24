//! BSIM3v3.3 device: C-derived DC, charge, small-signal and noise loading.
//!
//! Conditional drain/source primes implement external sheet resistance.
//! ACM geometry, NQS and unsupported analysis combinations are refused
//! explicitly rather than silently producing incomplete stamps.
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
    clippy::many_single_char_names,
    reason = "physical-model equation symbols are kept consistent with the ngspice C reference"
)]

use std::cell::RefCell;
use std::rc::Rc;
use std::sync::Arc;

use super::defaults::ModelDefaults;
use super::setup::InstanceSetup;
use super::temperature::TemperatureModel;
use super::{Inputs, ModelParameters, Value, charges, dc};
use crate::devices::limiting::{fetlim, limvds, pnjlim};
use crate::devices::models::{ModelContext, PreparationKey, ResolvedModel};
use crate::devices::nonlinear::{JunctionPoint, junction_companion};
use crate::devices::state::{DeviceState, IterationPhase};
use crate::devices::traits::{Device, StampContext};
use crate::netlist::ast::DeviceInstance;
use crate::primitives::{NodeAppend, NodeId, NodeKind, Real, SourceLoc, SpiceError, SpiceResult};

/// State slots. The first block is `b3ld.c`'s DC state; the charge block
/// follows C's `bsim3def.h` order, with each integrated charge immediately
/// followed by its derivative slot (C `BSIM3qg`/`BSIM3cqg`, which
/// `NIintegrate` writes as `qcap+1`).
mod slot {
    pub(super) const VBS: usize = 0;
    pub(super) const VBD: usize = 1;
    pub(super) const VGS: usize = 2;
    pub(super) const VDS: usize = 3;
    pub(super) const VON: usize = 4;
    pub(super) const VDSAT: usize = 5;
    pub(super) const MODE: usize = 6;
    pub(super) const CD: usize = 7;
    pub(super) const CBS: usize = 8;
    pub(super) const CBD: usize = 9;
    pub(super) const GM: usize = 10;
    pub(super) const GDS: usize = 11;
    pub(super) const GMBS: usize = 12;
    pub(super) const GBD: usize = 13;
    pub(super) const GBS: usize = 14;
    pub(super) const GBGS: usize = 15;
    pub(super) const GBDS: usize = 16;
    pub(super) const GBBS: usize = 17;
    pub(super) const CSUB: usize = 18;
    /// `CKTstate0[BSIM3qb]` and its derivative `cqb`.
    pub(super) const QB: usize = 19;
    pub(super) const CQB: usize = 20;
    /// `CKTstate0[BSIM3qg]` and `cqg`.
    pub(super) const QG: usize = 21;
    pub(super) const CQG: usize = 22;
    /// `CKTstate0[BSIM3qd]` and `cqd`.
    pub(super) const QD: usize = 23;
    pub(super) const CQD: usize = 24;
    /// `CKTstate0[BSIM3qbs]`.
    pub(super) const QBS: usize = 25;
    /// `CKTstate0[BSIM3qbd]`.
    pub(super) const QBD: usize = 26;
    /// C `here->BSIM3qinv`, the inversion charge the noise model reads.
    pub(super) const QINV: usize = 27;
    pub(super) const COUNT: usize = 28;
}

/// `b3ld.c`'s `gc*` equivalent charge conductances (`ag0` times the
/// capacitance derivatives) for one mode, named as in C.
#[derive(Debug, Clone, Copy, Default)]
struct ChargeConductances {
    gcggb: Real,
    gcgdb: Real,
    gcgsb: Real,
    gcdgb: Real,
    gcddb: Real,
    gcdsb: Real,
    gcsgb: Real,
    gcsdb: Real,
    gcssb: Real,
    gcbgb: Real,
    gcbdb: Real,
    gcbsb: Real,
}

/// `devsup.c`'s thermal voltage at 300.15 K (`CONSTvt0` in `main.c`).
const CONST_VT0: Real = 1.380_648_52e-23 * (27.0 + 273.15) / 1.602_176_620_8e-19;

/// How this load treated the supplied voltages (C's start/limiting phases).
#[derive(Debug, Clone, Copy, PartialEq)]
enum Start {
    Conditions,
    Junction,
    HeldOff,
    Limited,
    Exact,
}

struct Cached {
    temperature: Real,
    nominal: Real,
    defaults: Arc<ModelDefaults>,
    temperature_model: TemperatureModel,
    setup: InstanceSetup,
}

/// One BSIM3 instance. Holds the defaulted model and effective parameters per
/// (temperature, nominal temperature), as C re-runs `BSIM3temp` on changes.
pub struct Bsim3 {
    name: String,
    model_name: String,
    /// External drain/gate/source/bulk, then any internal primes.
    terminals: Vec<NodeId>,
    /// Drain/source prime nodes, present when C's series conductance is nonzero.
    primes: [Option<NodeId>; 2],
    params: ModelParameters,
    inputs: Inputs,
    location: SourceLoc,
    polarity: Real,
    off: bool,
    ic: [Option<Real>; 3],
    cache: RefCell<Rc<Cached>>,
}

impl std::fmt::Debug for Bsim3 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Bsim3")
            .field("name", &self.name)
            .field("model", &self.model_name)
            .field("polarity", &self.polarity)
            .finish_non_exhaustive()
    }
}

impl Bsim3 {
    /// Build an instance, validating everything before interning nodes.
    ///
    /// # Errors
    /// Invalid schema/setup input, ACM geometry, RSH-created internal nodes,
    /// or a nonpositive effective size (`b3check.c`).
    pub(crate) fn instantiate(
        instance: &DeviceInstance,
        nodes: &mut NodeAppend<'_>,
        model: &ResolvedModel<'_>,
        context: &ModelContext,
    ) -> SpiceResult<Box<dyn Device>> {
        Ok(Box::new(Self::build(instance, nodes, model, context)?))
    }

    fn build(
        instance: &DeviceInstance,
        nodes: &mut NodeAppend<'_>,
        model: &ResolvedModel<'_>,
        context: &ModelContext,
    ) -> SpiceResult<Self> {
        if instance.nodes.len() != 4 {
            return Err(SpiceError::parse(
                instance.location.clone(),
                "BSIM3 needs four terminals: drain, gate, source, bulk",
            ));
        }
        let params = super::model_parameters(model)?;
        let inputs = params.instance_parameters(instance, 1.)?;
        context.validate(&params.location)?;
        let defaults = model.prepare(
            PreparationKey::Nominal(context.nominal_temperature.to_bits()),
            || params.model_defaults(context),
        )?;
        let temperature_model = defaults.at_temperature(context.temperature, &instance.location)?;
        let setup = InstanceSetup::new(&defaults, &temperature_model, &inputs, &instance.location)?;
        let off = inputs
            .get("off")
            .is_some_and(|applied| matches!(applied.value, Value::Flag(true)));
        let component = |name: &str| -> Option<Real> {
            inputs.get(name).map(|applied| match applied.value {
                Value::Real(value) => value,
                _ => 0.,
            })
        };
        let ic = [component("icvbs"), component("icvgs"), component("icvds")];
        let (terminals, primes) = nodes.transaction(|staged| {
            let mut terminals: Vec<NodeId> = instance
                .nodes
                .iter()
                .map(|n| staged.intern(n))
                .collect::<SpiceResult<Vec<_>>>()?;
            let mut primes = [None; 2];
            for (index, conductance, suffix) in [
                (0, setup.drain_conductance, "drain"),
                (1, setup.source_conductance, "source"),
            ] {
                if conductance <= 0. {
                    continue;
                }
                let name = format!("{}#{suffix}", instance.name);
                if staged.get(&name).is_some() {
                    return Err(SpiceError::circuit(format!(
                        "{}: internal node name collision: {name}",
                        instance.name
                    )));
                }
                let prime = staged.intern(&name)?;
                staged.set_kind(prime, NodeKind::Internal)?;
                terminals.push(prime);
                primes[index] = Some(prime);
            }
            Ok((terminals, primes))
        })?;
        let device = Self {
            name: instance.name.clone(),
            model_name: model.card().name.clone(),
            terminals,
            primes,
            polarity: Real::from(setup.polarity),
            params,
            inputs,
            location: instance.location.clone(),
            off,
            ic,
            cache: RefCell::new(Rc::new(Cached {
                temperature: context.temperature,
                nominal: context.nominal_temperature,
                defaults,
                temperature_model,
                setup,
            })),
        };
        Ok(device)
    }

    /// The setup for these temperatures, rebuilt when C would re-run
    /// `BSIM3temp`. Share the immutable snapshot so no cell borrow outlives
    /// the call and repeated loads do not clone the parameter tables.
    fn ensure(&self, temperature: Real, nominal: Real) -> SpiceResult<Rc<Cached>> {
        let mut cache = self
            .cache
            .try_borrow_mut()
            .map_err(|_| SpiceError::circuit(format!("{}: reentrant BSIM3 load", self.name)))?;
        if cache.temperature != temperature || cache.nominal != nominal {
            let context = ModelContext::new(temperature, nominal);
            // Only the nominal temperature affects model-only defaults.
            // Keep context validation at the model location even on reuse.
            let defaults = if cache.nominal == nominal {
                context.validate(&self.params.location)?;
                Arc::clone(&cache.defaults)
            } else {
                Arc::new(self.params.model_defaults(&context)?)
            };
            let temperature_model = defaults.at_temperature(temperature, &self.location)?;
            let setup =
                InstanceSetup::new(&defaults, &temperature_model, &self.inputs, &self.location)?;
            *cache = Rc::new(Cached {
                temperature,
                nominal,
                defaults,
                temperature_model,
                setup,
            });
        }
        Ok(Rc::clone(&cache))
    }

    /// C `b3getic.c`: under `uic`, unset IC components are the terminal
    /// voltages; otherwise they are zero (`b3set.c` defaults).
    fn start_conditions(
        &self,
        conditions: bool,
        voltage: impl Fn(NodeId) -> Real,
    ) -> [Option<Real>; 3] {
        let [d, g, s, b] = [0, 1, 2, 3].map(|k| self.terminals[k]);
        let mut result = self.ic;
        if conditions {
            let filled = [
                voltage(b) - voltage(s),
                voltage(g) - voltage(s),
                voltage(d) - voltage(s),
            ];
            for (value, fallback) in result.iter_mut().zip(filled) {
                if value.is_none() {
                    *value = Some(fallback);
                }
            }
        }
        result
    }

    /// C `b3ld.c`'s start and limiting block.
    fn select(
        cached: &Cached,
        states: &DeviceState<'_>,
        raw: [Real; 3],
        ic: [Option<Real>; 3],
        off: bool,
        polarity: Real,
        xfact: Real,
    ) -> ([Real; 3], Start, bool) {
        let scaled = [0, 1, 2].map(|k| polarity * ic[k].unwrap_or(0.));
        if states.initial_conditions() {
            let v = if off { [0.; 3] } else { scaled };
            return (v, Start::Conditions, true);
        }
        if !states.device_limiting() {
            return (raw, Start::Exact, false);
        }
        match states.phase() {
            IterationPhase::Junction => {
                let v = if off {
                    [0.; 3]
                } else if scaled.iter().all(|v| *v == 0.) {
                    // `b3ld.c`: Vbs=0, Vgs=type*Vth0+0.1, Vds=0.1.
                    [0., polarity.mul_add(cached.setup.vth0, 0.1), 0.1]
                } else {
                    scaled
                };
                (v, Start::Junction, true)
            }
            IterationPhase::Fix if off => ([0.; 3], Start::HeldOff, true),
            IterationPhase::Predict if states.initial_transient() => {
                // b3ld.c MODEINITTRAN: CKTstate1's voltages, unlimited, and
                // Check stays 1.
                match (
                    states.accepted(1, slot::VBS),
                    states.accepted(1, slot::VGS),
                    states.accepted(1, slot::VDS),
                ) {
                    (Some(vbs), Some(vgs), Some(vds)) => ([vbs, vgs, vds], Start::Junction, true),
                    _ => (raw, Start::Exact, false),
                }
            }
            IterationPhase::Predict => {
                // b3ld.c MODEINITPRED: DEVpred's extrapolation, limited against
                // CKTstate1 (copied into CKTstate0).
                let values = (
                    states.predicted(slot::VBS, xfact),
                    states.predicted(slot::VGS, xfact),
                    states.predicted(slot::VDS, xfact),
                    states.accepted(1, slot::VBS),
                    states.accepted(1, slot::VGS),
                    states.accepted(1, slot::VDS),
                    states.accepted(1, slot::VON),
                );
                let (
                    Some(vbs),
                    Some(vgs),
                    Some(vds),
                    Some(old_vbs),
                    Some(old_vgs),
                    Some(old_vds),
                    Some(von),
                ) = values
                else {
                    return (raw, Start::Exact, false);
                };
                let (v, check) = limit([vbs, vgs, vds], [old_vbs, old_vgs, old_vds], von, cached);
                (v, Start::Limited, check)
            }
            IterationPhase::Fix | IterationPhase::Float => {
                let previous = |slot: usize| states.iterate(slot).filter(|v| v.is_finite());
                let values = (
                    previous(slot::VBS),
                    previous(slot::VBD),
                    previous(slot::VGS),
                    previous(slot::VDS),
                    previous(slot::VON),
                );
                let (Some(old_vbs), Some(old_vbd), Some(old_vgs), Some(old_vds), Some(von)) =
                    values
                else {
                    return (raw, Start::Exact, false);
                };
                let (v, check) = limit(raw, [old_vbs, old_vgs, old_vds], von, cached);
                let _ = old_vbd;
                (v, Start::Limited, check)
            }
        }
    }

    fn evaluate_at(cached: &Cached, bias: dc::Bias, gmin: Real) -> SpiceResult<dc::DcEvaluation> {
        dc::DcModel {
            model: &cached.defaults,
            temperature: &cached.temperature_model,
            setup: &cached.setup,
        }
        .evaluate(bias, gmin)
    }
}

/// `b3ld.c`'s limiting block over `DEVfetlim`, `DEVlimvds` and `DEVpnjlim`.
fn limit(raw: [Real; 3], old: [Real; 3], von: Real, cached: &Cached) -> ([Real; 3], bool) {
    let [vbs_in, vgs_in, vds_in] = raw;
    let [old_vbs, old_vgs, old_vds] = old;
    let vcrit = cached.temperature_model.vcrit;
    let vgd_raw = vgs_in - vds_in;
    let mut check = false;
    let (vgs, vds);
    if old_vds >= 0. {
        let limited_vgs = fetlim(vgs_in, old_vgs, von);
        let limited_vds = limvds(limited_vgs - vgd_raw, old_vds);
        vgs = limited_vgs;
        vds = limited_vds;
    } else {
        let limited_vgd = fetlim(vgd_raw, old_vgs - old_vds, von);
        let limited_vds = -limvds(-(vgs_in - limited_vgd), -old_vds);
        vds = limited_vds;
        vgs = limited_vgd + limited_vds;
    }
    let vbs = if vds >= 0. {
        let result = pnjlim(vbs_in, old_vbs, CONST_VT0, vcrit);
        check |= result.limited;
        result.value
    } else {
        let vbd_in = vbs_in - vds_in;
        let result = pnjlim(vbd_in, vbs_in - old_vds, CONST_VT0, vcrit);
        check |= result.limited;
        result.value + vds
    };
    ([vbs, vgs, vds], check)
}

impl Device for Bsim3 {
    fn validate_dc_inputs(&self, context: &crate::devices::ModelContext) -> SpiceResult<()> {
        self.ensure(context.temperature, context.nominal_temperature)
            .map(|_| ())
    }

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
        self.off || self.ic.iter().any(Option::is_some)
    }

    fn state_count(&self) -> usize {
        slot::COUNT
    }

    fn c_device_type(&self) -> crate::devices::setup::CDeviceType {
        crate::devices::setup::CDeviceType::Bsim
    }

    /// C's BSIM3 has no `DEVdisto` (`bsim3init.c`): `.disto` sees only its
    /// AC load.
    fn distortion(
        &self,
        _context: &crate::devices::distortion::DistortionContext<'_>,
    ) -> SpiceResult<crate::devices::distortion::DeviceDistortion> {
        Ok(crate::devices::distortion::DeviceDistortion::Linear)
    }

    /// `BSIM3trunc` (`b3trunc.c`): `CKTterr` on qb, qg and qd.
    fn truncation_slots(&self) -> Vec<usize> {
        vec![slot::QB, slot::QG, slot::QD]
    }

    /// `BSIM3convTest` (`b3cvtest.c`) on the last load's state, reading
    /// `CKTrhsOld` (the iterate that load used) as C does. `NIiter` runs it
    /// in `MODEINITFIX` too, where an `off` instance is not tested.
    fn convergence_test(
        &self,
        context: &crate::devices::ConvergenceContext<'_>,
    ) -> SpiceResult<bool> {
        let at = |k: usize| context.states.get(k).copied().unwrap_or(0.);
        let rhs_old = |node: NodeId| {
            context
                .unknowns
                .node_row(node)
                .and_then(|row| context.previous.get(row))
                .unwrap_or(0.)
        };
        let [d, g, s, b] = [0, 1, 2, 3].map(|k| self.terminals[k]);
        let dp = self.primes[0].unwrap_or(d);
        let sp = self.primes[1].unwrap_or(s);
        let pol = self.polarity;
        let vbs = pol * (rhs_old(b) - rhs_old(sp));
        let vgs = pol * (rhs_old(g) - rhs_old(sp));
        let vds = pol * (rhs_old(dp) - rhs_old(sp));
        let vbd = vbs - vds;
        let vgd = vgs - vds;
        let vgdo = at(slot::VGS) - at(slot::VDS);
        let delvbs = vbs - at(slot::VBS);
        let delvbd = vbd - at(slot::VBD);
        let delvgs = vgs - at(slot::VGS);
        let delvds = vds - at(slot::VDS);
        let delvgd = vgd - vgdo;
        let (gbd, gbs, gm, gds, gmbs) = (
            at(slot::GBD),
            at(slot::GBS),
            at(slot::GM),
            at(slot::GDS),
            at(slot::GMBS),
        );
        let (gbbs, gbgs, gbds, csub) = (
            at(slot::GBBS),
            at(slot::GBGS),
            at(slot::GBDS),
            at(slot::CSUB),
        );
        let forward = at(slot::MODE) >= 0.;
        let mut cd = at(slot::CD) - at(slot::CBD);
        let cdhat = if forward {
            cd += csub;
            f64::mul_add(
                gds + gbds,
                delvds,
                f64::mul_add(
                    gm + gbgs,
                    delvgs,
                    f64::mul_add(gmbs + gbbs, delvbs, f64::mul_add(gbd, -delvbd, cd)),
                ),
            )
        } else {
            f64::mul_add(
                gds,
                -delvds,
                f64::mul_add(gm, delvgd, f64::mul_add(gmbs - gbd, delvbd, cd)),
            )
        };
        let fix = context.phase == crate::devices::IterationPhase::Fix;
        if !self.off || !fix {
            let max = |a: Real, b: Real| if a > b { a } else { b };
            let tol = context
                .reltol
                .mul_add(max(cdhat.abs(), cd.abs()), context.abstol);
            if (cdhat - cd).abs() >= tol {
                return Ok(true);
            }
            let (cbs, cbd) = (at(slot::CBS), at(slot::CBD));
            let cbhat = if forward {
                f64::mul_add(
                    gbds,
                    -delvds,
                    f64::mul_add(
                        gbgs,
                        -delvgs,
                        f64::mul_add(
                            gbs - gbbs,
                            delvbs,
                            f64::mul_add(gbd, delvbd, cbs + cbd - csub),
                        ),
                    ),
                )
            } else {
                f64::mul_add(
                    gbds,
                    delvds,
                    f64::mul_add(
                        gbgs,
                        -delvgd,
                        f64::mul_add(
                            gbd - gbbs,
                            delvbd,
                            f64::mul_add(gbs, delvbs, cbs + cbd - csub),
                        ),
                    ),
                )
            };
            let tol = context
                .reltol
                .mul_add(max(cbhat.abs(), (cbs + cbd - csub).abs()), context.abstol);
            if (cbhat - (cbs + cbd - csub)).abs() > tol {
                return Ok(true);
            }
        }
        Ok(false)
    }

    fn stamp(&self, context: &mut StampContext<'_>) -> SpiceResult<()> {
        if context.mode.is_ac() {
            return Err(SpiceError::not_yet_ported(
                format!("{}: BSIM3 small-signal load", self.name),
                "src/spicelib/devices/bsim3/b3acld.c",
            ));
        }
        let cached = self.ensure(context.temperature, context.nominal_temperature)?;
        let [d, g, s, b] = [0, 1, 2, 3].map(|k| self.terminals[k]);
        let dp = self.primes[0].unwrap_or(d);
        let sp = self.primes[1].unwrap_or(s);
        let voltage = |node: NodeId| context.node_voltage(node);
        let pol = self.polarity;
        let raw = [
            pol * (voltage(b) - voltage(sp)),
            pol * (voltage(g) - voltage(sp)),
            pol * (voltage(dp) - voltage(sp)),
        ];
        let ic = self.start_conditions(context.states.initial_conditions(), voltage);
        let ([vbs, vgs, vds], start, check) =
            Self::select(&cached, &context.states, raw, ic, self.off, pol, {
                // C `xfact = CKTdelta / CKTdeltaOld[1]`; one in a `.dc` sweep.
                context
                    .integration
                    .map_or(1., crate::maths::integrator::Coefficients::predictor_ratio)
            });
        let bias = dc::Bias { vbs, vgs, vds };
        let e = Self::evaluate_at(&cached, bias, context.gmin)?;
        let report = match start {
            Start::Junction => true,
            Start::Limited => check,
            Start::Conditions | Start::Exact | Start::HeldOff => false,
        };
        if report {
            context.states.report_nonconvergence()?;
        }
        for (slot, value) in [
            (slot::VBS, vbs),
            (slot::VBD, vbs - vds),
            (slot::VGS, vgs),
            (slot::VDS, vds),
            (slot::VON, e.von),
            (slot::VDSAT, e.vdsat),
            (slot::MODE, Real::from(e.mode)),
            (slot::CD, e.ids),
            (slot::CBS, e.cbs),
            (slot::CBD, e.cbd),
            (slot::GM, e.gm),
            (slot::GDS, e.gds),
            (slot::GMBS, e.gmbs),
            (slot::GBD, e.gbd),
            (slot::GBS, e.gbs),
            (slot::GBGS, e.gbgs),
            (slot::GBDS, e.gbds),
            (slot::GBBS, e.gbbs),
            (slot::CSUB, e.isub),
        ] {
            context.states.set(slot, value)?;
        }

        // b3ld.c's line900 terms with RD=RS=0 (so the primes are the external
        // nodes) and the DC-only zero charge terms.
        let m = cached.setup.multiplier;
        let vbd = vbs - vds;
        let ConductanceParts {
            gm_eff, fwd, rev, ..
        } = conductance_parts(&e);
        let (cdreq, ceqbs, ceqbd) = if e.mode >= 0 {
            (
                pol * f64::mul_add(
                    e.gmbs,
                    -vbs,
                    f64::mul_add(e.gm, -vgs, f64::mul_add(e.gds, -vds, e.ids)),
                ),
                0.,
                -pol * f64::mul_add(
                    e.gbbs,
                    -vbs,
                    f64::mul_add(e.gbgs, -vgs, f64::mul_add(e.gbds, -vds, e.isub)),
                ),
            )
        } else {
            let vgd = vgs - vds;
            (
                -pol * f64::mul_add(
                    e.gmbs,
                    -vbd,
                    f64::mul_add(e.gm, -vgd, f64::mul_add(e.gds, vds, e.ids)),
                ),
                -pol * f64::mul_add(
                    e.gbbs,
                    -vbd,
                    f64::mul_add(e.gbgs, -vgd, f64::mul_add(e.gbds, vds, e.isub)),
                ),
                0.,
            )
        };
        let (ceqbs, ceqbd) = if pol > 0. {
            (
                ceqbs + f64::mul_add(e.gbs, -vbs, e.cbs),
                ceqbd + f64::mul_add(e.gbd, -vbd, e.cbd),
            )
        } else {
            (
                ceqbs - f64::mul_add(e.gbs, -vbs, e.cbs),
                ceqbd - f64::mul_add(e.gbd, -vbd, e.cbd),
            )
        };
        let ConductanceParts {
            gmbs_eff,
            gbdpdp,
            gbdpg,
            gbdpb,
            gbdpsp,
            gbbdp,
            gbbsp,
            gbspg,
            gbspsp,
            gbspb,
            gbspdp,
            ..
        } = conductance_parts(&e);
        // b3ld.c's charge block. C computes it when the analysis needs charge
        // storage (`MODEDCTRANCURVE | MODEAC | MODETRAN | MODEINITSMSIG`, or
        // `MODETRANOP` with `MODEUIC`) and a plain `.op` here needs it too, so
        // asks and a later `.tran` start from C's values. A `capMod` or NQS
        // mode the port has not ported keeps working for DC-only physics and
        // refuses only where charges are required.
        let integration = context.integration;
        let transient = integration.is_some();
        let cap_mod = cached.defaults.values.capmod.value;
        let charge_ready = cap_mod == 3 && cached.setup.nqs_mod == 0 && cached.setup.acnqs_mod == 0;
        let charge = if transient || charge_ready {
            Some(
                charges::ChargeModel {
                    model: &cached.defaults,
                    temperature: &cached.temperature_model,
                    setup: &cached.setup,
                }
                .evaluate(&e, bias)?,
            )
        } else {
            None
        };
        let vgb = vgs - vbs;
        let (ceqq, gc) = if let Some(charge) = charge {
            stamp_charge(pol, context, charge, e.mode, vgb, vbd, vbs)?
        } else {
            for slot in [
                slot::QB,
                slot::CQB,
                slot::QG,
                slot::CQG,
                slot::QD,
                slot::CQD,
                slot::QBS,
                slot::QBD,
                slot::QINV,
            ] {
                context.states.set(slot, 0.)?;
            }
            ([0.; 3], ChargeConductances::default())
        };

        // C `b3ld.c`'s RHS and Y terms. The series conductances connect the
        // external drain/source to their primes and are zero when the primes
        // were not created.
        let gdr = cached.setup.drain_conductance;
        let gsr = cached.setup.source_conductance;
        context.stamp_rhs(g, -m * ceqq[0])?;
        context.stamp_rhs(b, -m * (ceqbs + ceqbd + ceqq[1]))?;
        context.stamp_rhs(dp, m * (ceqbd - cdreq - ceqq[2]))?;
        context.stamp_rhs(sp, m * (cdreq + ceqbs + ceqq[0] + ceqq[1] + ceqq[2]))?;
        context.stamp_fixed([
            (d, d, m * gdr),
            (d, dp, -m * gdr),
            (dp, d, -m * gdr),
            (dp, dp, m * (gdr + e.gds + e.gbd + rev + gbdpdp + gc.gcddb)),
            (dp, g, m * (gm_eff + gbdpg + gc.gcdgb)),
            (
                dp,
                b,
                -m * (e.gbd - gmbs_eff - gbdpb + gc.gcdgb + gc.gcddb + gc.gcdsb),
            ),
            (dp, sp, -m * (e.gds + fwd - gbdpsp - gc.gcdsb)),
            (s, s, m * gsr),
            (s, sp, -m * gsr),
            (sp, s, -m * gsr),
            (sp, sp, m * (gsr + e.gds + e.gbs + fwd + gbspsp + gc.gcssb)),
            (sp, g, m * (-gm_eff + gbspg + gc.gcsgb)),
            (
                sp,
                b,
                -m * (e.gbs + gmbs_eff - gbspb + gc.gcsgb + gc.gcsdb + gc.gcssb),
            ),
            (sp, dp, -m * (e.gds + rev - gbspdp - gc.gcsdb)),
            (
                b,
                b,
                m * (e.gbd + e.gbs - e.gbbs - gc.gcbgb - gc.gcbdb - gc.gcbsb),
            ),
            (b, g, m * (-e.gbgs + gc.gcbgb)),
            (b, dp, m * (-e.gbd + gbbdp + gc.gcbdb)),
            (b, sp, m * (-e.gbs + gbbsp + gc.gcbsb)),
            // C's gate row's charge conductances (`b3ld.c` line900). Its
            // source row's `gcs*` terms are already above; stamping them
            // twice would double the source's charge conductances.
            (g, g, m * gc.gcggb),
            (g, b, -m * (gc.gcggb + gc.gcgdb + gc.gcgsb)),
            (g, dp, m * gc.gcgdb),
            (g, sp, m * gc.gcgsb),
        ])?;
        Ok(())
    }

    /// `b3ask.c`'s bias-dependent values at the solution, times the
    /// multiplicity: the DC quantities, then the `capMod=3` charges and
    /// capacitance derivatives the charge block leaves on the instance.
    fn observation_operating(
        &self,
        keyword: &str,
        context: &ModelContext,
        voltage: &dyn Fn(NodeId) -> Real,
        _transient: bool,
    ) -> SpiceResult<Option<Real>> {
        let cached = self.ensure(context.temperature, context.nominal_temperature)?;
        let pol = self.polarity;
        let [d, g, s, b] = [0, 1, 2, 3].map(|k| self.terminals[k]);
        let dp = self.primes[0].unwrap_or(d);
        let sp = self.primes[1].unwrap_or(s);
        let bias = dc::Bias {
            vbs: pol * (voltage(b) - voltage(sp)),
            vgs: pol * (voltage(g) - voltage(sp)),
            vds: pol * (voltage(dp) - voltage(sp)),
        };
        let e = Self::evaluate_at(&cached, bias, context.gmin)?;
        let m = cached.setup.multiplier;
        if let Some(value) = e.ask(keyword) {
            return Ok(Some(value * m));
        }
        let lower = keyword.to_ascii_lowercase();
        if !CHARGE_ASKS.contains(&lower.as_str()) {
            return Ok(None);
        }
        let c = charges::ChargeModel {
            model: &cached.defaults,
            temperature: &cached.temperature_model,
            setup: &cached.setup,
        }
        .evaluate(&e, bias)?;
        let overlap = c.with_overlap(e.mode, bias.vgs - bias.vbs);
        let value = match lower.as_str() {
            "qg" => overlap.qg,
            "qb" => overlap.qb,
            "qd" => overlap.qd,
            "cgg" => c.cggb,
            "cgd" => c.cgdb,
            "cgs" => c.cgsb,
            "cdg" => c.cdgb,
            "cdd" => c.cddb,
            "cds" => c.cdsb,
            "cbg" => c.cbgb,
            "cbd" => c.cbdb,
            "cbs" => c.cbsb,
            "capbd" => c.capbd,
            "capbs" => c.capbs,
            _ => return Ok(None),
        };
        Ok(Some(value * m))
    }

    fn assemble_nonlinear_charge(
        &self,
        context: &mut crate::devices::traits::ChargeContext<'_>,
    ) -> SpiceResult<()> {
        let model = context.model;
        let cached = self.ensure(model.temperature, model.nominal_temperature)?;
        let [d, g, s, b] = [0, 1, 2, 3].map(|k| self.terminals[k]);
        let dp = self.primes[0].unwrap_or(d);
        let sp = self.primes[1].unwrap_or(s);
        let voltage = |node| context.voltage(node);
        let pol = self.polarity;
        let point = dc::Bias {
            vbs: pol * (voltage(b) - voltage(sp)),
            vgs: pol * (voltage(g) - voltage(sp)),
            vds: pol * (voltage(dp) - voltage(sp)),
        };
        let e = Self::evaluate_at(&cached, point, model.gmin)?;
        let c = charges::ChargeModel {
            model: &cached.defaults,
            temperature: &cached.temperature_model,
            setup: &cached.setup,
        }
        .evaluate(&e, point)?;
        let q = c.with_overlap(e.mode, point.vgs - point.vbs);
        let m = cached.setup.multiplier;
        for (node, charge) in [(g, q.qg), (dp, q.qd), (sp, -(q.qg + q.qb + q.qd))] {
            context.branch([node, b], pol * m * charge, 0.)?;
        }
        for (row, col, value) in terminal_capacitances(&c, e.mode, [g, b, dp, sp]) {
            if let (Some(r), Some(k)) = (
                context.unknowns.node_row(row),
                context.unknowns.node_row(col),
            ) {
                context.jacobian.add(r, k, m * value)?;
            }
        }
        Ok(())
    }

    /// `b3acld.c`: the DC conductances in `A` and the `j omega` capacitance
    /// terms in `E`, both at the operating point's bias.
    fn assemble_small_signal(
        &self,
        context: &mut crate::devices::linear::LinearContext<'_>,
        bias: &crate::maths::Vector,
    ) -> SpiceResult<()> {
        let model = context.model_context;
        let cached = self.ensure(model.temperature, model.nominal_temperature)?;
        let unknowns = context.unknowns;
        let voltage = |node: NodeId| {
            unknowns
                .node_row(node)
                .and_then(|row| bias.get(row))
                .unwrap_or(0.)
        };
        let pol = self.polarity;
        let [d, g, s, b] = [0, 1, 2, 3].map(|k| self.terminals[k]);
        let dp = self.primes[0].unwrap_or(d);
        let sp = self.primes[1].unwrap_or(s);
        let point = dc::Bias {
            vbs: pol * (voltage(b) - voltage(sp)),
            vgs: pol * (voltage(g) - voltage(sp)),
            vds: pol * (voltage(dp) - voltage(sp)),
        };
        let e = Self::evaluate_at(&cached, point, model.gmin)?;
        let c = charges::ChargeModel {
            model: &cached.defaults,
            temperature: &cached.temperature_model,
            setup: &cached.setup,
        }
        .evaluate(&e, point)?;
        let m = cached.setup.multiplier;
        let gdr = cached.setup.drain_conductance;
        let gsr = cached.setup.source_conductance;

        // b3acld.c's real terms, identical to `b3ld.c`'s conductance load.
        let ConductanceParts {
            gm_eff,
            gmbs_eff,
            fwd,
            rev,
            gbdpdp,
            gbdpg,
            gbdpb,
            gbdpsp,
            gbbdp,
            gbbsp,
            gbspg,
            gbspsp,
            gbspb,
            gbspdp,
        } = conductance_parts(&e);
        let real = [
            (d, d, gdr),
            (d, dp, -gdr),
            (dp, d, -gdr),
            (dp, dp, gdr + e.gds + e.gbd + rev + gbdpdp),
            (dp, g, gm_eff + gbdpg),
            (dp, b, -(e.gbd - gmbs_eff - gbdpb)),
            (dp, sp, -(e.gds + fwd - gbdpsp)),
            (s, s, gsr),
            (s, sp, -gsr),
            (sp, s, -gsr),
            (sp, sp, gsr + e.gds + e.gbs + fwd + gbspsp),
            (sp, g, -gm_eff + gbspg),
            (sp, b, -(e.gbs + gmbs_eff - gbspb)),
            (sp, dp, -(e.gds + rev - gbspdp)),
            (b, b, e.gbd + e.gbs - e.gbbs),
            (b, g, -e.gbgs),
            (b, dp, -e.gbd + gbbdp),
            (b, sp, -e.gbs + gbbsp),
        ];
        for (row, col, value) in real {
            if let (Some(r), Some(k)) = (unknowns.node_row(row), unknowns.node_row(col)) {
                context.system.a.add(r, k, m * value)?;
            }
        }

        // b3acld.c's `j omega` terms: C's `xc*` are `cap * omega`, and the
        // complex assembly applies the `omega` factor (see
        // `LinearContext::scaled_dynamic_sum`). The lone `gdsi`/`Gmbsi` terms
        // are zero without AC NQS, which is refused at evaluation.
        for (row, col, value) in terminal_capacitances(&c, e.mode, [g, b, dp, sp]) {
            context.scaled_dynamic_sum(row, col, &[value], m)?;
        }
        Ok(())
    }

    /// C `b3pzld.c` equals `b3acld.c` with `j omega` replaced by `s`: the
    /// same `A + s E` pencil.
    fn assemble_pole_zero(
        &self,
        context: &mut crate::devices::linear::LinearContext<'_>,
        bias: &crate::maths::Vector,
    ) -> SpiceResult<()> {
        self.assemble_small_signal(context, bias)
    }

    /// `b3noi.c`: the drain/source series-resistance thermal sources, the
    /// channel thermal source selected by `noiMod` and the flicker source.
    /// C computes `qinv` in the charge block first, so the noise model needs
    /// the `capMod=3` charge evaluation; other modes refuse rather than
    /// substituting a density.
    fn noise(
        &self,
        context: &crate::devices::noise::NoiseContext<'_>,
        sources: &mut Vec<crate::devices::noise::NoiseSource>,
    ) -> SpiceResult<crate::devices::noise::DeviceNoise<'_>> {
        use crate::devices::noise::{DeviceNoise, FlickerLaw, NoiseFamily, NoiseKind, NoiseSource};
        let model = context.model_context;
        let cached = self.ensure(model.temperature, model.nominal_temperature)?;
        let pol = self.polarity;
        let [d, g, s, b] = [0, 1, 2, 3].map(|k| self.terminals[k]);
        let dp = self.primes[0].unwrap_or(d);
        let sp = self.primes[1].unwrap_or(s);
        let voltage = |node: NodeId| context.voltage(node);
        let bias = dc::Bias {
            vbs: pol * (voltage(b) - voltage(sp)),
            vgs: pol * (voltage(g) - voltage(sp)),
            vds: pol * (voltage(dp) - voltage(sp)),
        };
        let e = Self::evaluate_at(&cached, bias, model.gmin)?;
        let charge = charges::ChargeModel {
            model: &cached.defaults,
            temperature: &cached.temperature_model,
            setup: &cached.setup,
        }
        .evaluate(&e, bias)?;
        let densities = super::noise::NoiseModel {
            model: &cached.defaults,
            setup: &cached.setup,
        }
        .densities(&e, &charge, context.circuit_kelvin())?;
        let thermal = |conductance: Real| NoiseKind::Thermal {
            conductance,
            temperature: context.circuit_kelvin(),
        };
        sources.extend([
            NoiseSource::new(".rd", [dp, d], thermal(densities.drain_conductance)),
            NoiseSource::new(".rs", [sp, s], thermal(densities.source_conductance)),
            NoiseSource::new(".id", [dp, sp], thermal(densities.channel)),
            NoiseSource::new(
                ".1overf",
                [dp, sp],
                NoiseKind::Flicker {
                    numerator: densities.flicker_numerator,
                    law: FlickerLaw::Channel {
                        exponent: Some(densities.flicker_exponent),
                        factors: densities.flicker_factors,
                    },
                },
            ),
        ]);
        Ok(DeviceNoise::Sources {
            family: NoiseFamily::Bsim3,
            model: Some(&self.model_name),
            total: true,
        })
    }
}

/// `b3ask.c` charge/C-V keywords the charge block leaves on the instance.
/// `qbs`/`qbd` and `cqg`/`cqb`/`cqd` are not in C's instance parameter table,
/// so they are not askable even though `b3ask.c` handles them.
const CHARGE_ASKS: [&str; 14] = [
    "qg", "qb", "qd", "cgg", "cgd", "cgs", "cdg", "cdd", "cds", "cbg", "cbd", "cbs", "capbd",
    "capbs",
];

/// C's mode-selected capacitance derivatives (`b3ld.c`/`b3acld.c`): the
/// inverse mode mirrors gate/drain and derives the drain row from the
/// node-charge sum.
#[derive(Debug, Clone, Copy)]
struct CapDerivatives {
    cggb: Real,
    cgdb: Real,
    cgsb: Real,
    cdgb: Real,
    cddb: Real,
    cdsb: Real,
    cbgb: Real,
    cbdb: Real,
    cbsb: Real,
}

/// `b3acld.c`'s mode branch over the intrinsic capacitance derivatives.
fn mode_capacitances(c: &charges::ChargeEvaluation, mode: i8) -> CapDerivatives {
    if mode >= 0 {
        CapDerivatives {
            cggb: c.cggb,
            cgdb: c.cgdb,
            cgsb: c.cgsb,
            cdgb: c.cdgb,
            cddb: c.cddb,
            cdsb: c.cdsb,
            cbgb: c.cbgb,
            cbdb: c.cbdb,
            cbsb: c.cbsb,
        }
    } else {
        CapDerivatives {
            cggb: c.cggb,
            cgdb: c.cgsb,
            cgsb: c.cgdb,
            cdgb: -(c.cdgb + c.cggb + c.cbgb),
            cddb: -(c.cdsb + c.cgsb + c.cbsb),
            cdsb: -(c.cddb + c.cgdb + c.cbdb),
            cbgb: c.cbgb,
            cbdb: c.cbsb,
            cbsb: c.cbdb,
        }
    }
}

/// Physical terminal charge derivatives shared by AC and harmonic balance.
/// The 16 entries retain the original AC stamping order.
fn terminal_capacitances(
    c: &charges::ChargeEvaluation,
    mode: i8,
    [g, b, dp, sp]: [NodeId; 4],
) -> [(NodeId, NodeId, Real); 16] {
    let cap = mode_capacitances(c, mode);
    let (cgdo, cgso, cgbo) = (c.cgdo, c.cgso, c.cgbo);
    let (capbd, capbs) = (c.capbd, c.capbs);
    let xcdgb = cap.cdgb - cgdo;
    let xcddb = cap.cddb + capbd + cgdo;
    let xcdsb = cap.cdsb;
    let xcsgb = -(cap.cggb + cap.cbgb + cap.cdgb + cgso);
    let xcsdb = -(cap.cgdb + cap.cbdb + cap.cddb);
    let xcssb = capbs + cgso - (cap.cgsb + cap.cbsb + cap.cdsb);
    let xcggb = cap.cggb + cgdo + cgso + cgbo;
    let xcgdb = cap.cgdb - cgdo;
    let xcgsb = cap.cgsb - cgso;
    let xcbgb = cap.cbgb - cgbo;
    let xcbdb = cap.cbdb - capbd;
    let xcbsb = cap.cbsb - capbs;
    [
        (g, g, xcggb),
        (b, b, -(xcbgb + xcbdb + xcbsb)),
        (dp, dp, xcddb),
        (sp, sp, xcssb),
        (g, b, -(xcggb + xcgdb + xcgsb)),
        (g, dp, xcgdb),
        (g, sp, xcgsb),
        (b, g, xcbgb),
        (b, dp, xcbdb),
        (b, sp, xcbsb),
        (dp, g, xcdgb),
        (dp, b, -(xcdgb + xcddb + xcdsb)),
        (dp, sp, xcdsb),
        (sp, g, xcsgb),
        (sp, b, -(xcsgb + xcsdb + xcssb)),
        (sp, dp, xcsdb),
    ]
}

/// `b3ld.c` line900 / `b3acld.c`'s mode-selected conductances and substrate
/// current derivatives, shared so the DC load and the small-signal load cannot
/// diverge.
#[derive(Debug, Clone, Copy)]
struct ConductanceParts {
    gm_eff: Real,
    gmbs_eff: Real,
    fwd: Real,
    rev: Real,
    gbdpdp: Real,
    gbdpg: Real,
    gbdpb: Real,
    gbdpsp: Real,
    gbbdp: Real,
    gbbsp: Real,
    gbspg: Real,
    gbspsp: Real,
    gbspb: Real,
    gbspdp: Real,
}

fn conductance_parts(e: &dc::DcEvaluation) -> ConductanceParts {
    if e.mode >= 0 {
        ConductanceParts {
            gm_eff: e.gm,
            gmbs_eff: e.gmbs,
            fwd: e.gm + e.gmbs,
            rev: 0.,
            gbdpdp: e.gbds,
            gbdpg: e.gbgs,
            gbdpb: e.gbbs,
            gbdpsp: -(e.gbgs + e.gbds + e.gbbs),
            gbbdp: -e.gbds,
            gbbsp: e.gbds + e.gbgs + e.gbbs,
            gbspg: 0.,
            gbspsp: 0.,
            gbspb: 0.,
            gbspdp: 0.,
        }
    } else {
        ConductanceParts {
            gm_eff: -e.gm,
            gmbs_eff: -e.gmbs,
            fwd: 0.,
            rev: e.gm + e.gmbs,
            gbdpdp: 0.,
            gbdpg: 0.,
            gbdpb: 0.,
            gbdpsp: 0.,
            gbbdp: e.gbds + e.gbgs + e.gbbs,
            gbbsp: -e.gbds,
            gbspg: e.gbgs,
            gbspsp: e.gbds,
            gbspb: e.gbbs,
            gbspdp: -(e.gbgs + e.gbds + e.gbbs),
        }
    }
}

/// `b3ld.c`'s charge state, `NIintegrate` companions and `gc*` equivalent
/// conductances at `c`. Returns the gate/bulk/drain equivalent charge currents
/// C's `line860` computes.
fn stamp_charge(
    polarity: Real,
    context: &mut StampContext<'_>,
    c: charges::ChargeEvaluation,
    mode: i8,
    vgb: Real,
    vbd: Real,
    vbs: Real,
) -> SpiceResult<([Real; 3], ChargeConductances)> {
    let overlap = c.with_overlap(mode, vgb);
    for (slot, value) in [
        (slot::QB, overlap.qb),
        (slot::QG, overlap.qg),
        (slot::QD, overlap.qd),
        (slot::QBS, c.qbs),
        (slot::QBD, c.qbd),
        (slot::QINV, c.qinv),
    ] {
        context.states.set(slot, value)?;
    }
    // C's two mode branches: the negative branch mirrors gate/drain and
    // derives the drain row from the node-charge sum.
    let CapDerivatives {
        cggb,
        cgsb,
        cgdb,
        cbgb,
        cbsb,
        cbdb,
        cdgb,
        cdsb,
        cddb,
    } = mode_capacitances(&c, mode);
    let ag0 = context
        .integration
        .map_or(0., |coefficients| coefficients.ag()[0]);
    let (cgdo, cgso, cgbo, capbd, capbs) = (c.cgdo, c.cgso, c.cgbo, c.capbd, c.capbs);
    let gc = if mode >= 0 {
        ChargeConductances {
            gcggb: (cggb + cgdo + cgso + cgbo) * ag0,
            gcgdb: (cgdb - cgdo) * ag0,
            gcgsb: (cgsb - cgso) * ag0,
            gcdgb: (cdgb - cgdo) * ag0,
            gcddb: (cddb + capbd + cgdo) * ag0,
            gcdsb: cdsb * ag0,
            gcsgb: -(cggb + cbgb + cdgb + cgso) * ag0,
            gcsdb: -(cgdb + cbdb + cddb) * ag0,
            gcssb: (capbs + cgso - (cgsb + cbsb + cdsb)) * ag0,
            gcbgb: (cbgb - cgbo) * ag0,
            gcbdb: (cbdb - capbd) * ag0,
            gcbsb: (cbsb - capbs) * ag0,
        }
    } else {
        ChargeConductances {
            gcggb: (cggb + cgdo + cgso + cgbo) * ag0,
            gcgdb: (cgsb - cgdo) * ag0,
            gcgsb: (cgdb - cgso) * ag0,
            gcdgb: -(cggb + cbgb + cdgb + cgdo) * ag0,
            gcddb: (capbd + cgdo - (cgsb + cbsb + cdsb)) * ag0,
            gcdsb: -(cgdb + cbdb + cddb) * ag0,
            gcsgb: (cdgb - cgso) * ag0,
            gcsdb: cdsb * ag0,
            gcssb: (cddb + capbs + cgso) * ag0,
            gcbgb: (cbgb - cgbo) * ag0,
            gcbdb: (cbsb - capbd) * ag0,
            gcbsb: (cbdb - capbs) * ag0,
        }
    };
    // `NIintegrate` for qb, qg and qd (`cap` is zero, so C discards geq).
    let (cqg, cqb, cqd) = if let Some(coefficients) = context.integration {
        let mut derivatives = [0.; 3];
        for (index, (slot, charge)) in [
            (slot::QB, overlap.qb),
            (slot::QG, overlap.qg),
            (slot::QD, overlap.qd),
        ]
        .into_iter()
        .enumerate()
        {
            let (_, derivative) = junction_companion(
                context,
                coefficients,
                JunctionPoint {
                    current: 0.,
                    conductance: 0.,
                    charge,
                    capacitance: 0.,
                },
                slot,
            )?;
            derivatives[index] = derivative;
        }
        (derivatives[1], derivatives[0], derivatives[2])
    } else {
        for slot in [slot::CQB, slot::CQG, slot::CQD] {
            context.states.set(slot, 0.)?;
        }
        (0., 0., 0.)
    };
    let mut ceqq = [
        f64::mul_add(
            gc.gcgsb,
            vbs,
            f64::mul_add(gc.gcgdb, vbd, f64::mul_add(gc.gcggb, -vgb, cqg)),
        ),
        f64::mul_add(
            gc.gcbsb,
            vbs,
            f64::mul_add(gc.gcbdb, vbd, f64::mul_add(gc.gcbgb, -vgb, cqb)),
        ),
        f64::mul_add(
            gc.gcdsb,
            vbs,
            f64::mul_add(gc.gcddb, vbd, f64::mul_add(gc.gcdgb, -vgb, cqd)),
        ),
    ];
    // `b3ld.c` negates the charge equivalents for negative polarity.
    if polarity < 0. {
        ceqq = ceqq.map(|value| -value);
    }
    Ok((ceqq, gc))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::devices::models::ModelResolver;
    use crate::netlist::{Parser, source::parse_deck_text};
    use std::path::Path;

    #[test]
    fn temperature_change_reuses_defaults_and_failed_setup_preserves_snapshot() {
        let deck = Parser::new()
            .parse_deck(&parse_deck_text(
                Path::new("bsim3-cache.cir"),
                "BSIM3 cache\nm1 d g s 0 model l=1u w=10u\n.model model nmos level=8\n.end\n",
            ))
            .unwrap();
        let resolver = ModelResolver::new(&deck.models).unwrap();
        let inst = &deck.devices[0];
        let resolved = resolver.resolve(inst).unwrap().unwrap();
        let device = crate::primitives::NodeTable::new()
            .transaction(|nodes| Bsim3::build(inst, nodes, &resolved, &ModelContext::new(27., 27.)))
            .unwrap();
        let initial = device.ensure(27., 27.).unwrap();
        assert!(Rc::ptr_eq(&initial, &device.ensure(27., 27.).unwrap()));
        let hot = device.ensure(127., 27.).unwrap();
        assert!(!Rc::ptr_eq(&initial, &hot));
        assert!(Arc::ptr_eq(&initial.defaults, &hot.defaults));
        assert_ne!(initial.setup.u0temp.to_bits(), hot.setup.u0temp.to_bits());
        assert!(device.ensure(-273.15, 27.).is_err());
        assert!(Rc::ptr_eq(&hot, &device.ensure(127., 27.).unwrap()));
        let new_nominal = device.ensure(127., 42.).unwrap();
        assert!(!Arc::ptr_eq(&hot.defaults, &new_nominal.defaults));
        let fresh = crate::primitives::NodeTable::new()
            .transaction(|nodes| {
                Bsim3::build(inst, nodes, &resolved, &ModelContext::new(127., 42.))
            })
            .unwrap();
        let fresh = fresh.ensure(127., 42.).unwrap();
        assert_eq!(new_nominal.defaults, fresh.defaults);
        assert_eq!(new_nominal.setup, fresh.setup);
        assert_eq!(new_nominal.temperature_model, fresh.temperature_model);
        // Build independent instances at A/B/A. Only the immutable model
        // defaults are shared; each instance retains its own setup state.
        let again = crate::primitives::NodeTable::new()
            .transaction(|nodes| Bsim3::build(inst, nodes, &resolved, &ModelContext::new(27., 27.)))
            .unwrap();
        let again = again.ensure(27., 27.).unwrap();
        assert!(Arc::ptr_eq(&initial.defaults, &again.defaults));
        assert!(!Rc::ptr_eq(&initial, &again));
        assert_eq!(initial.setup, again.setup);
    }
}
