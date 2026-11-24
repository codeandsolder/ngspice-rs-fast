//! Lossy transmission line `O` with an `ltra` model (C
//! `src/spicelib/devices/ltra/`, parser `src/spicelib/parser/inp2o.c`).
//!
//! ```text
//! Oname p1 n1 p2 n2 model [v1= i1= v2= i2=] [ic=v1[,i1[,v2[,i2]]]]
//! .model model ltra [r=] [l=] [g=] [c=] len= [rel=] [abs=] [compactrel=]
//!        [compactabs=] [steplimit|nosteplimit] [lininterp|quadinterp|mixedinterp]
//!        [truncdontcut] [truncnr] [nocontrol] [ltra]
//! ```
//!
//! # Cases (`ltraset.c`)
//!
//! C classifies the per-metre `R L G C` into special cases and simulates only
//! four of them; the port does the same, with C's errors for the others:
//!
//! | Case | Condition | Model |
//! | --- | --- | --- |
//! | LC | `R = G = 0`, `L, C > 0` | lossless line: delayed waves, attenuation 1 |
//! | RLC | `R > 0`, `G = 0`, `L, C > 0` | delayed waves times `exp(-beta TD)` plus convolutions with `h1'`, `h2`, `h3'` (Bessel-function impulse responses) |
//! | RC | `R > 0`, `G = L = 0`, `C > 0` | diffusion line: convolutions with the `erfc` responses, no delay |
//! | RG | `R > 0`, `G > 0`, `L = C = 0` | resistive line: the exact DC two-port at every analysis |
//! | RL | `R > 0`, `G = C = 0`, `L > 0` | error ("RL line not supported yet") |
//! | `G > 0` with `L` or `C` | | error ("Nonzero G (except RG) line not supported yet") |
//! | fewer than two nonzero | | error |
//!
//! # Unknowns and signs
//!
//! `ltraset.c` creates two current unknowns with `CKTmkVolt`, so they are
//! node unknowns named `<name>#i1` and `<name>#i2` (plots carry `v(o1#i1)`,
//! amperes typed `voltage`, as ngspice writes them). `i1`/`i2` are the
//! currents **into** the positive terminal of port 1/2. There are no internal
//! voltage nodes. With `Y0 = 1/Z0`:
//!
//! | Analysis | Equations | C |
//! | --- | --- | --- |
//! | DC (LC, RLC, RC) | `i1 + i2 = 0`, `v(p1) - v(p2) = R len i1` (C's DC load ignores the negative terminals) | `ltraload.c`, `MODEDC` |
//! | DC, transient (RG; AC refused, below) | `v1 = cosh(l sqrt(RG)) v2 - (1 + gmin) sqrt(R/G) sinh(.) i2`, `cosh(.) i2 - (1 + gmin) sqrt(G/R) sinh(.) v2 + i1 = 0` (as stamped by C) | `ltraload.c` |
//! | AC, `.sp` | `Y0 V1 - I1 = exp(-lambda len) (Y0 V2 + I2)` and symmetrically, `Y0`, `lambda` exact at each frequency | `ltraacld.c` |
//! | transient LC/RLC | `Y0 v1 - i1 (+ Y0 h1'(0) v1) = input1` | `ltraload.c` |
//! | transient RC | `-i1 + h1'(0) v1 - h2(0) i2 - h3'(0) v2 = input1` | `ltraload.c` |
//!
//! The AC coefficients are complex and frequency dependent; the line puts
//! their real part into `A` and their imaginary part divided by `omega` into
//! `E` at every frequency ([`Device::small_signal_depends_on_frequency`]), as
//! the lossless line does. At frequency zero it stamps its DC form.
//!
//! # Transient (`ltraload.c`, `ltraacct.c`, `ltratrun.c`, `ltramisc.c`)
//!
//! * **History.** C keeps every accepted time (`CKTtimePoints`) and the
//!   port voltages and currents there (`LTRAv1`, `LTRAi1`, `LTRAv2`,
//!   `LTRAi2`), never pruned: the convolutions run over the whole past. The
//!   port keeps the same samples in the device's [`DelayHistory`] (width
//!   4: `v1 i1 v2 i2`), starting with the operating point at `t = 0` and
//!   appending every accepted point; trials only read it.
//! * **Load.** At each trial time the line computes C's convolution
//!   coefficients ([`convolution::rlc_coefficients`],
//!   [`convolution::rc_coefficients`]) over the accepted times, stamps the
//!   coefficients of the unknown sample into the matrix and puts the
//!   convolution of the history (minus the operating-point initial
//!   conditions) and the delayed waves at `t - TD` (quadratic, linear or
//!   mixed interpolation, `LTRAquadInterp`/`LTRAlinInterp`) into the
//!   right-hand side. Within a transient proposal, both Newton and observation
//!   loads reuse these immutable history-dependent inputs. Rejected proposals
//!   discard them before retry; acceptance drops them before mutating history.
//! * **Breakpoints.** On acceptance the line requests a breakpoint at the
//!   previous accepted time plus `TD` when either attenuated wave
//!   `(v + Z0 i) exp(-beta TD)` changes slope by at least `rel max(|d1|,
//!   |d2|) + abs` and is not steady (C's `CHECK` with `50 reltol`,
//!   `50 abstol`). A request in the past is an error, as `CKTsetBreak`
//!   ("breakpoint in the past") makes it in C.
//! * **Step bound.** LC/RLC lines bound every step by `TD` (`steplimit`, the
//!   default) or, under `nosteplimit`, when the slope test fires for the
//!   trial; RLC lines also by `LTRAmaxSafeStep` unless `truncdontcut`. C's
//!   truncation-error control (`LTRAlteCalculate`) is unreachable from a
//!   deck (`LTRAlteConType` can only be `nocontrol`), and so is `truncnr`.
//!
//! # Deliberate divergences (explicit errors)
//!
//! * Negative `R`, `L`, `G`, `C`, a nonpositive or missing `len`, negative
//!   `rel`/`abs`/`compactrel`/`compactabs` (C computes square roots of
//!   negative numbers or a zero delay).
//! * A `level` or any setter outside `LTRAmPTable` on the model (C warns and
//!   ignores), instance-parameter defaults on the model card, a leading
//!   value on the instance (C ignores it).
//! * `.tran ... uic`, `.options trytocompact` (refused by the options
//!   layer), `.pz`, `.noise`, `.disto`, `.sens`, the diffsol BDF backend.
//! * RG lines in AC/`.sp`: C's `LTRAacLoad` loads the DC stamp of every LTRA
//!   model and returns at the first RG model, corrupting any other lossy
//!   line's AC load; the device cannot see the other lines, so it refuses.
//! * A nonfinite convolution input, which C's Bessel approximations produce
//!   once `beta t` exceeds about 700 (`exp` overflow); C continues with NaN.
//! * C reports `@o1[v1]` and the other initial conditions after a transient
//!   as the operating point it copied into them (`ltraload.c`
//!   `MODEINITTRAN`); the port offers no instance observations.

// C-compatible numerical expressions are kept in their reference order:
// Clippy's fused multiply-add suggestions change IEEE-754 rounding.
#![expect(
    clippy::manual_midpoint,
    reason = "C source midpoint arithmetic retains its original evaluation and rounding order"
)]
#![expect(
    clippy::too_many_lines,
    reason = "a complete translated C numerical routine remains contiguous for equation parity review"
)]
#![expect(
    clippy::many_single_char_names,
    reason = "physical-model equation symbols are kept consistent with the ngspice C reference"
)]
#![expect(
    clippy::suboptimal_flops,
    reason = "SPICE numerical reference parity requires the original non-fused C operation order"
)]

pub mod convolution;

use crate::devices::delay::{
    DelayContext, DelayHistory, DelayLine, DelaySamples, DelayUpdate, WaveformHistory,
    WaveformUpdate,
};
use crate::devices::linear::LinearContext;
use crate::devices::models::{ModelContext, ModelFamily, ModelResolver};
use crate::devices::schema::{ScalarDomain, ScalarParameter, ScalarSchema, ScalarUnit};
use crate::devices::traits::{AnalysisMode, Device, StampContext};
use crate::maths::Vector;
use crate::netlist::ast::{DeviceInstance, ParameterKind};
use crate::primitives::{
    NodeAppend, NodeId, Real, SourceLoc, SpiceError, SpiceResult, almost_equal_ulps,
    parse_spice_number,
};

/// C reference for errors.
const C_REFERENCE: &str = "src/spicelib/devices/ltra/";

/// Internal-node suffixes in `ltraset.c`'s creation order (`CKTmkVolt`).
pub const INTERNAL_SUFFIXES: [&str; 2] = ["i1", "i2"];

/// `LTRAmPTable` scalars (`ltra.c`). `r`, `l`, `g`, `c` default to 0, `len` is
/// required, `rel`/`abs` default to 1 (`ltraset.c`), and `compactrel`/
/// `compactabs` of 0 mean C's `reltol`/`abstol`.
static MODEL: ScalarSchema<'static> = ScalarSchema::new(&[
    scalar("r", ScalarUnit::OhmPerMetre, Some(0.)),
    scalar("l", ScalarUnit::HenryPerMetre, Some(0.)),
    scalar("g", ScalarUnit::SiemensPerMetre, Some(0.)),
    scalar("c", ScalarUnit::FaradPerMetre, Some(0.)),
    ScalarParameter {
        name: "len",
        unit: ScalarUnit::Metre,
        domain: ScalarDomain::Positive,
        default: None,
    },
    scalar("rel", ScalarUnit::Dimensionless, Some(1.)),
    scalar("abs", ScalarUnit::Dimensionless, Some(1.)),
    scalar("compactrel", ScalarUnit::Dimensionless, Some(0.)),
    scalar("compactabs", ScalarUnit::Dimensionless, Some(0.)),
]);

const fn scalar(name: &'static str, unit: ScalarUnit, default: Option<Real>) -> ScalarParameter {
    ScalarParameter {
        name,
        unit,
        domain: ScalarDomain::NonNegative,
        default,
    }
}

/// C's `LTRAspecialCase` of a simulated line.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LtraCase {
    /// Lossless (`R = G = 0`).
    Lc,
    /// Series loss only (`G = 0`).
    Rlc,
    /// Distributed RC (`L = G = 0`).
    Rc,
    /// Distributed RG (`L = C = 0`).
    Rg,
}

/// `LTRAhowToInterp`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Interpolation {
    /// `lininterp`.
    Linear,
    /// `quadinterp` (the default without `trytocompact`).
    Quadratic,
    /// `mixedinterp`: quadratic unless it leaves the range of the samples.
    Mixed,
}

/// The validated `ltra` model with `ltratemp.c`'s derived quantities.
#[derive(Debug, Clone, PartialEq)]
pub struct LtraModel {
    name: String,
    case: LtraCase,
    resist: Real,
    induct: Real,
    conduct: Real,
    capac: Real,
    length: Real,
    reltol: Real,
    abstol: Real,
    st_line_reltol: Real,
    st_line_abstol: Real,
    step_limit: bool,
    interpolation: Interpolation,
    trunc_dont_cut: bool,
    // ltratemp.c
    td: Real,
    imped: Real,
    admit: Real,
    alpha: Real,
    beta: Real,
    attenuation: Real,
    cbyr: Real,
    rclsqr: Real,
    int_h1dash: Real,
    int_h2: Real,
    int_h3dash: Real,
    // ltraload.c, MODEDC, RG case
    cosh_root_gr: Real,
    r_rs_lrgr_or_g: Real,
    r_gs_lrgr_or_r: Real,
}

impl LtraModel {
    /// Reads, validates and classifies an `ltra` model card (`LTRAmParam`,
    /// `ltraset.c`, `ltratemp.c`).
    ///
    /// # Errors
    /// Unknown setters, `level`, invalid values, a missing `len` and C's
    /// unsupported cases.
    pub fn from_card(card: &crate::netlist::ast::ModelCard) -> SpiceResult<Self> {
        let mut setters = Vec::with_capacity(card.parameters.len());
        let mut step_limit = true;
        let mut interpolation = Interpolation::Quadratic;
        let mut trunc_dont_cut = false;
        for parameter in &card.parameters {
            if parameter.kind == ParameterKind::Flag {
                match parameter.name.as_str() {
                    // LTRA_MOD_LTRA, the (default) LTRA_MOD_NOCONTROL and
                    // LTRA_MOD_TRUNCNR (read only by the unreachable
                    // truncation-error control) change nothing.
                    "ltra" | "nocontrol" | "truncnr" => {}
                    "steplimit" => step_limit = true,
                    "nosteplimit" => step_limit = false,
                    "lininterp" => interpolation = Interpolation::Linear,
                    "quadinterp" => interpolation = Interpolation::Quadratic,
                    "mixedinterp" => interpolation = Interpolation::Mixed,
                    "truncdontcut" => trunc_dont_cut = true,
                    other => {
                        return Err(SpiceError::parse(
                            parameter.location.clone(),
                            format!("unknown ltra model flag '{other}' (ltra.c LTRAmPTable)"),
                        ));
                    }
                }
                continue;
            }
            if parameter.name.eq_ignore_ascii_case("level") {
                return Err(SpiceError::Unsupported {
                    feature: "level on an ltra model (LTRAmPTable has no level; C ignores it)"
                        .into(),
                    location: Some(parameter.location.clone()),
                });
            }
            if matches!(parameter.name.as_str(), "v1" | "v2" | "i1" | "i2" | "ic") {
                return Err(SpiceError::Unsupported {
                    feature: format!(
                        "instance-parameter default '{}' on an ltra model (inpgmod.c defaults)",
                        parameter.name
                    ),
                    location: Some(parameter.location.clone()),
                });
            }
            setters.push(parameter);
        }
        let values = MODEL.validate(setters, &card.location)?;
        let get = |name: &str| values.get(name).map_or(0., |v| v.value);
        let Some(length) = values.get("len").map(|v| v.value) else {
            return Err(SpiceError::parse(
                card.location.clone(),
                format!("{}: lossy line length must be given (ltraset.c)", card.name),
            ));
        };
        let (resist, induct, conduct, capac) = (get("r"), get("l"), get("g"), get("c"));
        let fail = |message: &str| {
            Err(SpiceError::parse(
                card.location.clone(),
                format!("{}: {message} (ltraset.c)", card.name),
            ))
        };
        // ltraset.c's classification, in C's order (later tests win).
        // The supported classifications are mutually exclusive. Keep the
        // reference's explicit refusals ahead of the generic validation.
        let case = match (resist == 0., conduct == 0., induct == 0., capac == 0.) {
            (true, true, false, false) => Some(LtraCase::Lc),
            (false, true, false, false) => Some(LtraCase::Rlc),
            (false, true, true, false) => Some(LtraCase::Rc),
            (false, true, false, true) => return fail("RL line not supported yet"),
            (false, false, true, true) => Some(LtraCase::Rg),
            (_, false, false, _) | (_, false, _, false) => {
                return fail("Nonzero G (except RG) line not supported yet");
            }
            _ => None,
        };
        let nonzero = [resist, conduct, induct, capac]
            .iter()
            .filter(|v| **v != 0.)
            .count();
        if nonzero <= 1 {
            return fail("At least two of R,L,G,C must be specified and nonzero");
        }
        let Some(case) = case else {
            return fail("unclassified R, L, G, C combination");
        };
        let mut model = Self {
            name: card.name.clone(),
            case,
            resist,
            induct,
            conduct,
            capac,
            length,
            reltol: get("rel"),
            abstol: get("abs"),
            st_line_reltol: get("compactrel"),
            st_line_abstol: get("compactabs"),
            step_limit,
            interpolation,
            trunc_dont_cut,
            td: 0.,
            imped: 0.,
            admit: 0.,
            alpha: 0.,
            beta: 0.,
            attenuation: 0.,
            cbyr: 0.,
            rclsqr: 0.,
            int_h1dash: 0.,
            int_h2: 0.,
            int_h3dash: 0.,
            cosh_root_gr: 0.,
            r_rs_lrgr_or_g: 0.,
            r_gs_lrgr_or_r: 0.,
        };
        model.temperature();
        let derived = [
            model.td,
            model.imped,
            model.admit,
            model.attenuation,
            model.cbyr,
            model.rclsqr,
            model.cosh_root_gr,
            model.r_rs_lrgr_or_g,
            model.r_gs_lrgr_or_r,
        ];
        if derived.iter().any(|v| !v.is_finite()) {
            return fail("derived line constants are not finite");
        }
        Ok(model)
    }

    /// `ltratemp.c` (and the RG constants of `ltraload.c`'s DC branch).
    fn temperature(&mut self) {
        match self.case {
            LtraCase::Lc => {
                self.imped = (self.induct / self.capac).sqrt();
                self.admit = 1. / self.imped;
                self.td = (self.induct * self.capac).sqrt() * self.length;
                self.attenuation = 1.0;
            }
            LtraCase::Rlc => {
                self.imped = (self.induct / self.capac).sqrt();
                self.admit = 1. / self.imped;
                self.td = (self.induct * self.capac).sqrt() * self.length;
                self.alpha = 0.5 * (self.resist / self.induct);
                self.beta = self.alpha;
                self.attenuation = (-self.beta * self.td).exp();
                // alpha > 0 (R, L > 0).
                self.int_h1dash = -1.0;
                self.int_h2 = 1.0 - self.attenuation;
                self.int_h3dash = -self.attenuation;
            }
            LtraCase::Rc => {
                self.cbyr = self.capac / self.resist;
                self.rclsqr = self.resist * self.capac * self.length * self.length;
                self.int_h1dash = 0.0;
                self.int_h2 = 1.0;
                self.int_h3dash = 0.0;
            }
            LtraCase::Rg => {
                let mut dummy1 = self.length * (self.resist * self.conduct).sqrt();
                let dummy2 = (-dummy1).exp();
                dummy1 = dummy1.exp();
                self.cosh_root_gr = 0.5 * (dummy1 + dummy2);
                self.r_rs_lrgr_or_g = if self.conduct <= 1.0e-10 {
                    self.length * self.resist
                } else {
                    0.5 * (dummy1 - dummy2) * (self.resist / self.conduct).sqrt()
                };
                self.r_gs_lrgr_or_r = if self.resist <= 1.0e-10 {
                    self.length * self.conduct
                } else {
                    0.5 * (dummy1 - dummy2) * (self.conduct / self.resist).sqrt()
                };
            }
        }
    }

    /// The simulated case.
    #[must_use]
    pub const fn case(&self) -> LtraCase {
        self.case
    }

    /// Propagation delay `sqrt(L C) len` (LC and RLC; 0 otherwise).
    #[must_use]
    pub const fn td(&self) -> Real {
        self.td
    }

    /// Characteristic impedance `sqrt(L / C)` (LC and RLC; 0 otherwise).
    #[must_use]
    pub const fn z0(&self) -> Real {
        self.imped
    }
}

/// A lossy transmission line instance.
#[derive(Debug, Clone, PartialEq)]
pub struct LossyLine {
    name: String,
    /// `p1 n1 p2 n2`, then the current unknowns `i1 i2`.
    terminals: [NodeId; 6],
    model: LtraModel,
    /// `v1 i1 v2 i2` (used by C only under `uic`).
    initial: [Real; 4],
    location: SourceLoc,
}

fn invalid(location: &SourceLoc, message: impl Into<String>) -> SpiceError {
    SpiceError::Parse {
        location: location.clone(),
        message: message.into(),
    }
}

/// Actual work performed during a transient run, not estimated cache hits.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct LtraWork {
    /// Transient attempts, including rejected proposals.
    pub attempts: usize,
    /// Newton circuit loads within those attempts.
    pub newton_loads: usize,
    /// Non-RG transient LTRA stamps (Newton and isolated observations).
    pub line_loads: usize,
    /// RC/RLC coefficient preparations.
    pub coefficient_preparations: usize,
    /// Evaluations of the accepted waveform input (including LC interpolation).
    pub convolution_evaluations: usize,
}

/// Exact identity of the mathematical inputs to coefficient preparation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CoefficientKernel {
    Rc { cbyr: u64, rclsqr: u64 },
    Rlc { td: u64, alpha: u64, beta: u64 },
}

impl CoefficientKernel {
    const fn for_model(model: &LtraModel) -> Option<Self> {
        match model.case {
            LtraCase::Rc => Some(Self::Rc {
                cbyr: model.cbyr.to_bits(),
                rclsqr: model.rclsqr.to_bits(),
            }),
            LtraCase::Rlc => Some(Self::Rlc {
                td: model.td.to_bits(),
                alpha: model.alpha.to_bits(),
                beta: model.beta.to_bits(),
            }),
            LtraCase::Lc | LtraCase::Rg => None,
        }
    }

    fn prepare(
        self,
        times: &std::collections::VecDeque<Real>,
        time: Real,
        out: &mut convolution::Coefficients,
        work: &mut LtraWork,
    ) {
        work.coefficient_preparations += 1;
        let chop = 0.; // C's LTRAchopReltol has no deck setter.
        match self {
            Self::Rc { cbyr, rclsqr } => convolution::rc_coefficients(
                Real::from_bits(cbyr),
                Real::from_bits(rclsqr),
                time,
                times,
                chop,
                out,
            ),
            Self::Rlc { td, alpha, beta } => convolution::rlc_coefficients(
                Real::from_bits(td),
                Real::from_bits(alpha),
                Real::from_bits(beta),
                time,
                times,
                chop,
                out,
            ),
        }
    }
}

fn same_time_grid(a: DelaySamples<'_, 4>, b: DelaySamples<'_, 4>) -> bool {
    let (a, b) = (a.times(), b.times());
    a.len() == b.len() && a.iter().zip(b).all(|(a, b)| a.to_bits() == b.to_bits())
}

/// Analysis-owned scratch. Only the coefficient allocation and per-device
/// slot storage survive an attempt; no accepted data or prepared input does.
#[derive(Debug)]
pub(super) struct LtraWorkspace {
    coefficients: convolution::Coefficients,
    // A single coefficient family at a time: no model × history allocation.
    coefficient_source: Option<(usize, CoefficientKernel)>,
    coefficient_model: String,
    loads: std::collections::BTreeMap<usize, Option<Load>>,
    pub(super) work: LtraWork,
}

impl LtraWorkspace {
    pub(super) const fn new() -> Self {
        Self {
            coefficients: convolution::Coefficients::new(),
            coefficient_source: None,
            coefficient_model: String::new(),
            loads: std::collections::BTreeMap::new(),
            work: LtraWork {
                attempts: 0,
                newton_loads: 0,
                line_loads: 0,
                coefficient_preparations: 0,
                convolution_evaluations: 0,
            },
        }
    }

    pub(super) const fn begin(&mut self) {
        self.work.attempts += 1;
    }

    pub(super) fn finish(&mut self) {
        self.coefficient_source = None;
        self.coefficient_model.clear();
        // Discard logical inputs and the active coefficient range, not their
        // allocations. A retry always starts from an empty preparation.
        self.coefficients.reset(0);
        for slot in self.loads.values_mut() {
            *slot = None;
        }
    }

    fn load(
        &mut self,
        ordinal: usize,
        line: &LossyLine,
        history: &WaveformHistory,
        accepted: &crate::devices::StateHistory,
        time: Real,
    ) -> SpiceResult<Load> {
        self.work.line_loads += 1;
        if let Some(load) = self.loads.get(&ordinal).copied().flatten() {
            return Ok(load);
        }
        let samples = history.samples::<4>()?;
        if let Some(kernel) = CoefficientKernel::for_model(&line.model) {
            // Only consider sharing on the first visit to a line. Name and
            // exact coefficient arguments identify the same effective model
            // family; compare the complete immutable accepted time grid.
            let share = self.coefficient_model == line.model.name
                && self.coefficient_source.is_some_and(|(source, previous)| {
                    previous == kernel
                        && accepted
                            .delay(source)
                            .and_then(|history| history.as_waveform().ok())
                            .and_then(|history| history.samples::<4>().ok())
                            .is_some_and(|prior| same_time_grid(prior, samples))
                });
            if !share {
                kernel.prepare(
                    samples.times(),
                    time,
                    &mut self.coefficients,
                    &mut self.work,
                );
                self.coefficient_model.clone_from(&line.model.name);
                self.coefficient_source = Some((ordinal, kernel));
            }
        } else {
            self.coefficient_source = None;
            self.coefficients.reset(0);
        }
        // RHS histories are never shared, even for identical time grids.
        let load = line.evaluate_inputs(samples, time, &self.coefficients, &mut self.work)?;
        self.loads.insert(ordinal, Some(load));
        Ok(load)
    }
}

/// The convolution and interpolation state of one transient load.
#[derive(Debug, Clone, Copy)]
struct Load {
    /// Matrix coefficients of the unknown sample (`h1'`, `h2`, `h3'`).
    first: [Real; 3],
    /// `LTRAinput1`, `LTRAinput2`.
    inputs: [Real; 2],
}

/// The real stamp layouts differ by analysis and line case, but each is a
/// fixed-size batch. Keep them on the stack without padding or changing the
/// order in which C's load adds the individual terms.
type RealEntry = (NodeId, NodeId, Real);

enum RealEntries {
    Dc([RealEntry; 9]),
    Rg([RealEntry; 13]),
    Lc([RealEntry; 10]),
    Rlc([RealEntry; 14]),
    Rc([RealEntry; 16]),
}

impl RealEntries {
    const fn as_slice(&self) -> &[RealEntry] {
        match self {
            Self::Dc(entries) => entries,
            Self::Rg(entries) => entries,
            Self::Lc(entries) => entries,
            Self::Rlc(entries) => entries,
            Self::Rc(entries) => entries,
        }
    }

    fn stamp(&self, context: &mut StampContext<'_>) -> SpiceResult<()> {
        // Do not cache these positions as one fixed layout: DC and transient
        // have different layouts for the same line.
        for &(row, col, value) in self.as_slice() {
            context.stamp(row, col, value)?;
        }
        Ok(())
    }
}

impl LossyLine {
    /// Builds the line from its instance card and `ltra` model, interning
    /// the four terminals and the two current unknowns (nothing is interned
    /// on error).
    ///
    /// # Errors
    /// A missing or wrong model, invalid model or instance values, unknown
    /// setters or node-name collisions.
    pub fn instantiate(
        instance: &DeviceInstance,
        nodes: &mut NodeAppend<'_>,
        models: &ModelResolver<'_>,
        context: &ModelContext,
    ) -> SpiceResult<Self> {
        context.validate(&instance.location)?;
        if instance.nodes.len() != 4 {
            return Err(SpiceError::circuit(format!(
                "{}: a lossy transmission line has four terminals",
                instance.name
            )));
        }
        let Some(model) = models.resolve(instance)? else {
            return Err(invalid(
                &instance.location,
                format!(
                    "{}: a lossy line requires an ltra model (inp2o.c)",
                    instance.name
                ),
            ));
        };
        if model.family() != ModelFamily::Ltra {
            return Err(invalid(
                &instance.location,
                format!("{}: incorrect model type (inp2o.c)", instance.name),
            ));
        }
        let model = LtraModel::from_card(model.card())?;
        let mut initial = [0.; 4];
        let number = |text: &str, at: &SourceLoc, name: &str| {
            parse_spice_number(text)
                .filter(|v| v.is_finite())
                .ok_or_else(|| invalid(at, format!("{name} must be a finite number, not '{text}'")))
        };
        for p in &instance.parameters {
            match (&p.kind, p.name.as_str()) {
                (ParameterKind::InitialConditions(values), "ic") => {
                    for value in values {
                        let slot = match value.name.as_str() {
                            "v1" => 0,
                            "i1" => 1,
                            "v2" => 2,
                            "i2" => 3,
                            other => {
                                return Err(invalid(
                                    &value.value.location,
                                    format!("unexpected ic component {other}"),
                                ));
                            }
                        };
                        initial[slot] = number(&value.value.text, &value.value.location, "ic")?;
                    }
                }
                (ParameterKind::Scalar, name @ ("v1" | "i1" | "v2" | "i2")) => {
                    let slot = match name {
                        "v1" => 0,
                        "i1" => 1,
                        "v2" => 2,
                        _ => 3,
                    };
                    initial[slot] = number(&p.value, &p.location, name)?;
                }
                (ParameterKind::Scalar, other) => {
                    return Err(invalid(
                        &p.location,
                        format!("unknown lossy-line parameter {other} (LTRApTable)"),
                    ));
                }
                _ => {
                    return Err(SpiceError::not_yet_ported(
                        format!(
                            "{}: lossy-line setter '{}' of this form",
                            p.location, p.name
                        ),
                        "src/spicelib/devices/ltra/ltrapar.c",
                    ));
                }
            }
        }
        let terminals = crate::devices::factory::line_terminals::<6>(
            instance,
            nodes,
            &INTERNAL_SUFFIXES,
            "lossy-line",
        )?;
        Ok(Self {
            name: instance.name.clone(),
            terminals,
            model,
            initial,
            location: instance.location.clone(),
        })
    }

    /// The validated model.
    #[must_use]
    pub const fn model(&self) -> &LtraModel {
        &self.model
    }

    /// The instance `v1 i1 v2 i2` initial conditions.
    #[must_use]
    pub const fn initial_conditions(&self) -> [Real; 4] {
        self.initial
    }

    const fn pos1(&self) -> NodeId {
        self.terminals[0]
    }
    const fn neg1(&self) -> NodeId {
        self.terminals[1]
    }
    const fn pos2(&self) -> NodeId {
        self.terminals[2]
    }
    const fn neg2(&self) -> NodeId {
        self.terminals[3]
    }
    const fn ibr1(&self) -> NodeId {
        self.terminals[4]
    }
    const fn ibr2(&self) -> NodeId {
        self.terminals[5]
    }

    /// The DC entries (`ltraload.c`, `MODEDC`) with C's `gmin`.
    fn dc_entries(&self, gmin: Real) -> RealEntries {
        let m = &self.model;
        let (p1, n1, p2, n2, b1, b2) = (
            self.pos1(),
            self.neg1(),
            self.pos2(),
            self.neg2(),
            self.ibr1(),
            self.ibr2(),
        );
        match m.case {
            LtraCase::Rg => RealEntries::Rg([
                (b1, p1, 1.0),
                (b1, n1, -1.0),
                (b1, p2, -m.cosh_root_gr),
                (b1, n2, m.cosh_root_gr),
                (b1, b2, (1. + gmin) * m.r_rs_lrgr_or_g),
                (b2, b2, m.cosh_root_gr),
                (b2, p2, -(1. + gmin) * m.r_gs_lrgr_or_r),
                (b2, n2, (1. + gmin) * m.r_gs_lrgr_or_r),
                (b2, b1, 1.0),
                (p1, b1, 1.0),
                (n1, b1, -1.0),
                (p2, b2, 1.0),
                (n2, b2, -1.0),
            ]),
            LtraCase::Lc | LtraCase::Rlc | LtraCase::Rc => RealEntries::Dc([
                (p1, b1, 1.0),
                (n1, b1, -1.0),
                (p2, b2, 1.0),
                (n2, b2, -1.0),
                (b1, b1, 1.0),
                (b1, b2, 1.0),
                (b2, p1, 1.0),
                (b2, p2, -1.0),
                (b2, b1, -m.resist * m.length),
            ]),
        }
    }

    /// The transient matrix entries (`ltraload.c`, all non-DC cases but RG)
    /// given the first convolution coefficients.
    fn transient_entries(&self, first: [Real; 3], gmin: Real) -> RealEntries {
        let m = &self.model;
        let (p1, n1, p2, n2, b1, b2) = (
            self.pos1(),
            self.neg1(),
            self.pos2(),
            self.neg2(),
            self.ibr1(),
            self.ibr2(),
        );
        match m.case {
            LtraCase::Lc | LtraCase::Rlc => {
                let lc = [
                    (b1, p1, m.admit),
                    (b1, n1, -m.admit),
                    (b1, b1, -1.0),
                    (p1, b1, 1.0),
                    (n1, b1, -1.0),
                    (b2, p2, m.admit),
                    (b2, n2, -m.admit),
                    (b2, b2, -1.0),
                    (p2, b2, 1.0),
                    (n2, b2, -1.0),
                ];
                if m.case == LtraCase::Rlc {
                    let d = m.admit * first[0];
                    RealEntries::Rlc([
                        (b1, p1, d),
                        (b1, n1, -d),
                        (b2, p2, d),
                        (b2, n2, -d),
                        lc[0],
                        lc[1],
                        lc[2],
                        lc[3],
                        lc[4],
                        lc[5],
                        lc[6],
                        lc[7],
                        lc[8],
                        lc[9],
                    ])
                } else {
                    RealEntries::Lc(lc)
                }
            }
            LtraCase::Rc => {
                let [h1, h2, h3] = first;
                RealEntries::Rc([
                    (b1, b1, -1.0),
                    (p1, b1, 1.0),
                    (n1, b1, -1.0),
                    (b2, b2, -1.0),
                    (p2, b2, 1.0),
                    (n2, b2, -1.0),
                    (b1, p1, h1),
                    (b1, n1, -h1),
                    (b2, p2, h1),
                    (b2, n2, -h1),
                    (b1, b2, -h2),
                    (b2, b1, -h2),
                    (b1, p2, -h3),
                    (b1, n2, h3),
                    (b2, p1, -h3),
                    (b2, n1, h3),
                ])
            }
            LtraCase::Rg => self.dc_entries(gmin),
        }
    }

    /// `[v1, i1, v2, i2]` of a solution: the port voltages and the current
    /// unknowns (C reads them from `CKTrhsOld`).
    fn port_values(&self, context: &DelayContext<'_>) -> [Real; 4] {
        let v = |node| context.node_voltage(node);
        [
            v(self.pos1()) - v(self.neg1()),
            v(self.ibr1()),
            v(self.pos2()) - v(self.neg2()),
            v(self.ibr2()),
        ]
    }

    fn internal(&self, message: impl Into<String>) -> SpiceError {
        SpiceError::Numerical {
            context: self.name.clone(),
            message: message.into(),
        }
    }

    /// Evaluate the delayed waveform using C's strict lower bracket and
    /// last-two extrapolation, independent of coefficient preparation.
    fn delayed_values(&self, history: DelaySamples<'_, 4>, time: Real) -> SpiceResult<[Real; 4]> {
        let m = &self.model;
        let times = history.times();
        let index = times.len() - 1;
        let mut delayed = [0.; 4];
        let target = time - m.td;
        let mut i = times
            .partition_point(|&t| t < target)
            .checked_sub(1)
            .ok_or_else(|| {
                self.internal("cannot find the delayed time point (ltraload.c E_INTERN)")
            })?;
        if i == index {
            // Step larger than the delay: extrapolate from the last two.
            i = i.checked_sub(1).ok_or_else(|| {
                self.internal("the first time step exceeds the line delay (ltraload.c E_INTERN)")
            })?;
        }
        let (t2, t3) = (times[i], times[i + 1]);
        let middle = history.sample(i);
        let right = history.sample(i + 1);
        let quadratic = matches!(
            m.interpolation,
            Interpolation::Quadratic | Interpolation::Mixed
        );
        let qf = if i != 0 && quadratic {
            Some(
                convolution::quad_interp(target, times[i - 1], t2, t3)
                    .ok_or_else(|| self.internal("coincident history times"))?,
            )
        } else {
            None
        };
        let lf = if i == 0
            || matches!(
                m.interpolation,
                Interpolation::Mixed | Interpolation::Linear
            ) {
            Some(
                convolution::lin_interp(target, t2, t3)
                    .ok_or_else(|| self.internal("coincident history times"))?,
            )
        } else {
            None
        };
        let left = if qf.is_some() {
            history.sample(i - 1)
        } else {
            middle
        };
        for (k, value) in delayed.iter_mut().enumerate() {
            let (mut d, out_of_range) = if let Some([q1, q2, q3]) = qf {
                let (a, b, c) = (left[k], middle[k], right[k]);
                let d = a * q1 + b * q2 + c * q3;
                (d, d > a.max(b).max(c) || d < a.min(b).min(c))
            } else {
                (0., false)
            };
            // C falls back to linear for lininterp, at the first sample,
            // and for mixedinterp when the quadratic leaves the samples'
            // range (quadinterp keeps the out-of-range value).
            let linear = m.interpolation == Interpolation::Linear || i == 0 || out_of_range;
            if linear && !(i != 0 && m.interpolation == Interpolation::Quadratic) {
                let [l2, l3] = lf.ok_or_else(|| self.internal("missing linear interpolation"))?;
                d = middle[k] * l2 + right[k] * l3;
            }
            *value = d;
        }
        Ok(delayed)
    }

    /// Pure evaluation of `ltraload.c` for an isolated proposed load.
    fn transient_load(
        &self,
        history: &WaveformHistory,
        time: Real,
        coefficients: &mut convolution::Coefficients,
        work: &mut LtraWork,
    ) -> SpiceResult<Load> {
        let samples = history.samples::<4>()?;
        if let Some(kernel) = CoefficientKernel::for_model(&self.model) {
            kernel.prepare(samples.times(), time, coefficients, work);
        } else {
            coefficients.reset(0);
        }
        self.evaluate_inputs(samples, time, coefficients, work)
    }

    /// Apply independently prepared coefficients to this line's own history.
    fn evaluate_inputs(
        &self,
        history: DelaySamples<'_, 4>,
        time: Real,
        coefficients: &convolution::Coefficients,
        work: &mut LtraWork,
    ) -> SpiceResult<Load> {
        let tdover =
            matches!(self.model.case, LtraCase::Lc | LtraCase::Rlc) && time > self.model.td;
        let delayed = if tdover {
            self.delayed_values(history, time)?
        } else {
            [0.; 4]
        };
        work.convolution_evaluations += 1;
        let inputs = self.convolution_inputs(history, coefficients, tdover, delayed);
        if !(inputs.iter().all(|v| v.is_finite())
            && coefficients.first.iter().all(|v| v.is_finite()))
        {
            return Err(self.internal(format!(
                "nonfinite convolution at t = {time:e} (ltramisc.c Bessel/erfc terms overflow once \
                 beta t exceeds about 700; C continues with NaN)"
            )));
        }
        Ok(Load {
            first: coefficients.first,
            inputs,
        })
    }

    /// Apply the history sums in C's original reverse traversal and
    /// arithmetic order. No accepted or trial state is mutated.
    fn convolution_inputs(
        &self,
        history: DelaySamples<'_, 4>,
        coefficients: &convolution::Coefficients,
        tdover: bool,
        delayed: [Real; 4],
    ) -> [Real; 2] {
        let m = &self.model;
        let initial = history.sample(0);
        let [iv1, ii1, iv2, ii2] = initial;
        let [v1d, i1d, v2d, i2d] = delayed;
        let mut inputs = [0.; 2];
        match m.case {
            LtraCase::Rlc => {
                let c = coefficients;
                // h1' with v1 and v2.
                let seeds = if tdover {
                    [
                        [0.; 2],
                        [(i2d - ii2) * c.first[1], (i1d - ii1) * c.first[1]],
                        [(v2d - iv2) * c.first[2], (v1d - iv1) * c.first[2]],
                    ]
                } else {
                    [[0.; 2]; 3]
                };
                let [h1, h2, h3] = reduce_history(
                    history,
                    &c.weights,
                    initial,
                    if tdover { c.aux_index } else { 0 },
                    seeds,
                );
                let [mut d1, mut d2] = h1;
                d1 += iv1 * m.int_h1dash;
                d2 += iv2 * m.int_h1dash;
                d1 -= iv1 * c.first[0];
                d2 -= iv2 * c.first[0];
                inputs[0] -= d1 * m.admit;
                inputs[1] -= d2 * m.admit;
                // h2 with i2 and i1.
                let [mut d1, mut d2] = h2;
                d1 += ii2 * m.int_h2;
                d2 += ii1 * m.int_h2;
                inputs[0] += d1;
                inputs[1] += d2;
                // h3' with v2 and v1.
                let [mut d1, mut d2] = h3;
                d1 += iv2 * m.int_h3dash;
                d2 += iv1 * m.int_h3dash;
                inputs[0] += m.admit * d1;
                inputs[1] += m.admit * d2;
                self.lossless_inputs(&mut inputs, tdover, [iv1, ii1, iv2, ii2], delayed);
            }
            LtraCase::Lc => {
                self.lossless_inputs(&mut inputs, tdover, [iv1, ii1, iv2, ii2], delayed);
            }
            LtraCase::Rc => {
                let c = coefficients;
                let [h1, h2, h3] =
                    reduce_history(history, &c.weights, initial, c.aux_index, [[0.; 2]; 3]);
                let [mut d1, mut d2] = h1;
                d1 += iv1 * m.int_h1dash;
                d2 += iv2 * m.int_h1dash;
                d1 -= iv1 * c.first[0];
                d2 -= iv2 * c.first[0];
                inputs[0] -= d1;
                inputs[1] -= d2;
                let [mut d1, mut d2] = h2;
                d1 += ii2 * m.int_h2;
                d2 += ii1 * m.int_h2;
                d1 -= ii2 * c.first[1];
                d2 -= ii1 * c.first[1];
                inputs[0] += d1;
                inputs[1] += d2;
                let [mut d1, mut d2] = h3;
                d1 += iv2 * m.int_h3dash;
                d2 += iv1 * m.int_h3dash;
                d1 -= iv2 * c.first[2];
                d2 -= iv1 * c.first[2];
                inputs[0] += d1;
                inputs[1] += d2;
            }
            LtraCase::Rg => {}
        }
        inputs
    }

    /// The lossless-like part of `ltraload.c` (LC and the end of RLC).
    fn lossless_inputs(
        &self,
        inputs: &mut [Real; 2],
        tdover: bool,
        [iv1, ii1, iv2, ii2]: [Real; 4],
        [v1d, i1d, v2d, i2d]: [Real; 4],
    ) {
        let m = &self.model;
        if tdover {
            inputs[0] += m.attenuation * (v2d * m.admit + i2d);
            inputs[1] += m.attenuation * (v1d * m.admit + i1d);
        } else {
            inputs[0] += m.attenuation * (iv2 * m.admit + ii2);
            inputs[1] += m.attenuation * (iv1 * m.admit + ii1);
        }
    }

    /// `ltraacld.c`: the complex entries at angular frequency `omega > 0`, as
    /// `(row, column, re, im)`.
    fn ac_entries(&self, omega: Real) -> Option<[(NodeId, NodeId, Real, Real); 16]> {
        let m = &self.model;
        let (y0_r, y0_i, lambda_r, lambda_i) = match m.case {
            LtraCase::Lc => (m.admit, 0.0, 0.0, (m.induct * m.capac).sqrt() * omega),
            LtraCase::Rlc => {
                let theta = 0.5 * (m.resist / (omega * m.induct)).atan();
                let root = (m.resist * m.resist + omega * omega * m.induct * m.induct).sqrt();
                let mut mag = (omega * m.capac / root).sqrt();
                let y0 = (mag * theta.cos(), mag * theta.sin());
                let theta = std::f64::consts::PI / 2. - theta;
                mag *= root;
                (y0.0, y0.1, mag * theta.cos(), mag * theta.sin())
            }
            LtraCase::Rc => {
                let y = (0.5 * omega * m.cbyr).sqrt();
                let l = (0.5 * omega * m.resist * m.capac).sqrt();
                (y, y, l, l)
            }
            // RG lines stamp their DC load instead (stamp_small_signal).
            LtraCase::Rg => return None,
        };
        let exparg_r = -lambda_r * m.length;
        let exparg_i = -lambda_i * m.length;
        let explambda_r = exparg_r.exp() * exparg_i.cos();
        let explambda_i = exparg_r.exp() * exparg_i.sin();
        let y0exp_r = y0_r * explambda_r - y0_i * explambda_i;
        let y0exp_i = y0_r * explambda_i + y0_i * explambda_r;
        let (p1, n1, p2, n2, b1, b2) = (
            self.pos1(),
            self.neg1(),
            self.pos2(),
            self.neg2(),
            self.ibr1(),
            self.ibr2(),
        );
        Some([
            (b1, p1, y0_r, y0_i),
            (b1, n1, -y0_r, -y0_i),
            (b1, b1, -1.0, 0.0),
            (b1, p2, -y0exp_r, -y0exp_i),
            (b1, n2, y0exp_r, y0exp_i),
            (b1, b2, -explambda_r, -explambda_i),
            (b2, p2, y0_r, y0_i),
            (b2, n2, -y0_r, -y0_i),
            (b2, b2, -1.0, 0.0),
            (b2, p1, -y0exp_r, -y0exp_i),
            (b2, n1, y0exp_r, y0exp_i),
            (b2, b1, -explambda_r, -explambda_i),
            (p1, b1, 1.0, 0.0),
            (n1, b1, -1.0, 0.0),
            (p2, b2, 1.0, 0.0),
            (n2, b2, -1.0, 0.0),
        ])
    }

    /// Small-signal assembly: the DC form at frequency zero (and always for
    /// RG, as `LTRAacLoad` calls the DC load), else `ltraacld.c` with `A +=
    /// Re`, `E += Im / omega`.
    fn stamp_small_signal(&self, context: &mut LinearContext<'_>, dc: bool) -> SpiceResult<()> {
        let frequency = context.model_context.frequency;
        let unknowns = context.unknowns;
        let row = |node: NodeId| unknowns.node_row(node);
        if !dc && frequency != 0. && self.model.case == LtraCase::Rg {
            return Err(SpiceError::not_yet_ported(
                format!(
                    "AC analysis with RG lossy line {}: C's LTRAacLoad loads every LTRA model's \
                     DC stamp and returns early when it meets an RG model, which corrupts the \
                     AC load of any other lossy line in the circuit; the port cannot see the \
                     other lines from the device and refuses the RG line in AC",
                    self.name
                ),
                "src/spicelib/devices/ltra/ltraacld.c (LTRA_MOD_RG case)",
            ));
        }
        if dc || frequency == 0. || self.model.case == LtraCase::Rg {
            for &(r, c, value) in self.dc_entries(context.model_context.gmin).as_slice() {
                if let (Some(r), Some(c)) = (row(r), row(c)) {
                    context.system.a.add(r, c, value)?;
                }
            }
            return Ok(());
        }
        let omega = 2. * std::f64::consts::PI * frequency;
        for (r, c, re, im) in self
            .ac_entries(omega)
            .ok_or_else(|| self.internal("RG line requires DC matrix entries"))?
        {
            if !(re.is_finite() && im.is_finite()) {
                return Err(self.internal(format!(
                    "nonfinite AC coefficient at {frequency:e} Hz (ltraacld.c)"
                )));
            }
            if let (Some(r), Some(c)) = (row(r), row(c)) {
                context.system.a.add(r, c, re)?;
                if im != 0. {
                    context.system.e.add(r, c, im / omega)?;
                }
            }
        }
        Ok(())
    }

    fn unsupported(&self, analysis: &str) -> SpiceError {
        SpiceError::not_yet_ported(
            format!(
                "{analysis} with lossy transmission line {} (C has no LTRA routine for it)",
                self.name
            ),
            C_REFERENCE,
        )
    }

    /// `ltraacct.c`'s slope-change test on the attenuated waves of one port.
    fn slope_changed(&self, d1: Real, d2: Real) -> bool {
        (d1 - d2).abs() >= self.model.reltol * d1.abs().max(d2.abs()) + self.model.abstol
    }
}

impl Device for LossyLine {
    fn name(&self) -> &str {
        &self.name
    }

    fn designator(&self) -> char {
        'o'
    }

    fn terminals(&self) -> &[NodeId] {
        &self.terminals
    }

    fn small_signal_depends_on_frequency(&self) -> bool {
        // RG lines too, so that their AC refusal sees the frequency.
        true
    }

    fn delay_line(&self) -> Option<&dyn DelayLine> {
        Some(self)
    }

    fn stamp(&self, context: &mut StampContext<'_>) -> SpiceResult<()> {
        match context.mode {
            AnalysisMode::OperatingPoint | AnalysisMode::DcSweep => {
                self.dc_entries(context.gmin).stamp(context)?;
            }
            AnalysisMode::Transient { time, .. } => {
                if self.model.case == LtraCase::Rg {
                    // ltraload.c loads RG lines with their DC two-port always.
                    self.dc_entries(context.gmin).stamp(context)?;
                    return Ok(());
                }
                let history = context.states.delay_history().ok_or_else(|| {
                    SpiceError::circuit(format!(
                        "{}: transient load without a delay history (companion driver only)",
                        self.name
                    ))
                })?;
                let history = history.as_waveform()?;
                let load =
                    if let Some((workspace, accepted, ordinal)) = context.states.ltra_workspace() {
                        workspace.load(ordinal, self, history, accepted, time)?
                    } else {
                        // Explicit isolated low-level loads own one disposable evaluation.
                        self.transient_load(
                            history,
                            time,
                            &mut convolution::Coefficients::new(),
                            &mut LtraWork::default(),
                        )?
                    };
                self.transient_entries(load.first, context.gmin)
                    .stamp(context)?;
                context.stamp_rhs(self.ibr1(), load.inputs[0])?;
                context.stamp_rhs(self.ibr2(), load.inputs[1])?;
            }
            AnalysisMode::Ac { .. } => {
                return Err(SpiceError::circuit(format!(
                    "{}: the AC load is assembled by assemble_small_signal",
                    self.name
                )));
            }
        }
        Ok(())
    }

    fn assemble_linear(&self, context: &mut LinearContext<'_>) -> SpiceResult<()> {
        self.stamp_small_signal(context, true)
    }

    fn assemble_small_signal(
        &self,
        context: &mut LinearContext<'_>,
        _bias: &Vector,
    ) -> SpiceResult<()> {
        self.stamp_small_signal(context, false)
    }

    fn assemble_pole_zero(
        &self,
        _context: &mut LinearContext<'_>,
        _bias: &Vector,
    ) -> SpiceResult<()> {
        Err(SpiceError::Unsupported {
            feature: format!(
                "pole-zero analysis of lossy transmission line {}: exp(-lambda(s) len) is not a \
                 polynomial pencil, and C has no LTRA pole-zero load (DEVpzLoad is NULL in \
                 ltrainit.c; PZinit's transmission-line check finds only TRA lines, so ngspice \
                 silently leaves the lossy line out of the matrix)",
                self.name
            ),
            location: Some(self.location.clone()),
        })
    }

    fn noise(
        &self,
        _context: &crate::devices::noise::NoiseContext<'_>,
        _sources: &mut Vec<crate::devices::noise::NoiseSource>,
    ) -> SpiceResult<crate::devices::noise::DeviceNoise<'_>> {
        Err(self.unsupported(".noise"))
    }

    fn distortion(
        &self,
        _context: &crate::devices::distortion::DistortionContext<'_>,
    ) -> SpiceResult<crate::devices::distortion::DeviceDistortion> {
        Err(self.unsupported(".disto"))
    }
}

impl DelayLine for LossyLine {
    fn new_history(&self) -> DelayHistory {
        DelayHistory::waveform(4)
    }

    /// The `t = 0` sample (C stores the operating point at `CKTtimeIndex`
    /// 0 in `LTRAaccept` and copies it into `LTRAinitVolt/Cur` at the first
    /// `MODEINITTRAN` load).
    fn start(&self, context: &DelayContext<'_>) -> SpiceResult<DelayUpdate> {
        if context.initial_conditions {
            return Err(SpiceError::not_yet_ported(
                format!(".tran uic with lossy transmission line {}", self.name),
                "src/spicelib/devices/ltra/ltraload.c (MODEINITTRAN | MODEUIC)",
            ));
        }
        Ok(DelayUpdate::Waveform(WaveformUpdate {
            reset: Some(vec![(context.time, self.port_values(context).to_vec())]),
            ..WaveformUpdate::default()
        }))
    }

    /// `ltraacct.c` (without `trytocompact`): record the point, and request a
    /// breakpoint at the previous time plus `TD` where an attenuated wave
    /// changes slope.
    fn accept(
        &self,
        history: &DelayHistory,
        context: &DelayContext<'_>,
    ) -> SpiceResult<DelayUpdate> {
        let history = history.as_waveform()?;
        let m = &self.model;
        let values = self.port_values(context);
        let history = history.samples::<4>()?;
        let times = history.times();
        let count = times.len();
        let time = context.time;
        let wave = |v: Real, i: Real| (v + i * m.imped) * m.attenuation;
        // The new sample is index `count`; C's CKTtimeIndex.
        let previous = count - 1;
        let [pv1, pi1, pv2, pi2] = history.sample(previous);
        let [ov1, oi1, ov2, oi2] = history.sample(previous.saturating_sub(1));
        let v1 = wave(values[0], values[1]);
        let v2 = wave(pv1, pi1);
        let v3 = wave(ov1, oi1);
        let v4 = wave(values[2], values[3]);
        let v5 = wave(pv2, pi2);
        let v6 = wave(ov2, oi2);
        let h = time - times[previous];
        let d1 = (v1 - v2) / h;
        let d2 = if count < 2 {
            0.
        } else {
            (v2 - v3) / (times[previous] - times[previous - 1])
        };
        let d3 = (v4 - v5) / h;
        let d4 = if count < 2 {
            0.
        } else {
            (v5 - v6) / (times[previous] - times[previous - 1])
        };
        let check = |a: Real, b: Real, c: Real| {
            a.max(b).max(c) - a.min(b).min(c)
                >= (50.0 * (context.reltol / 3.0 * (a + b + c) + context.abstol)).abs()
        };
        let mut update = WaveformUpdate {
            append: Some((time, values.to_vec())),
            ..WaveformUpdate::default()
        };
        let fire = (self.slope_changed(d1, d2) && check(v1, v2, v3))
            || (self.slope_changed(d3, d4) && check(v4, v5, v6));
        if fire {
            let when = times[previous] + m.td;
            // CKTsetBreak: a request equal to now (3 ULPs) is ignored, one
            // before now is C's "breakpoint in the past" error.
            if !almost_equal_ulps(when, time, 3) {
                if when < time {
                    return Err(self.internal(format!(
                        "breakpoint in the past at {when:e} (now {time:e}; ltraacct.c \
                         CKTsetBreak fails the same way)"
                    )));
                }
                update.breakpoints.push(when);
            }
        }
        Ok(DelayUpdate::Waveform(update))
    }

    /// `ltratrun.c` (the reachable part: `LTRAlteConType` is always
    /// `nocontrol` from a deck).
    fn timestep_limit(
        &self,
        history: &DelayHistory,
        context: &DelayContext<'_>,
    ) -> SpiceResult<Option<Real>> {
        let history = history.as_waveform()?;
        let m = &self.model;
        if !matches!(m.case, LtraCase::Lc | LtraCase::Rlc) {
            return Ok(None);
        }
        let mut bound = if m.step_limit {
            Some(m.td)
        } else {
            let history = history.samples::<4>()?;
            let times = history.times();
            let count = times.len();
            if count < 2 {
                return Err(self.internal("delay history too short"));
            }
            let index = count - 1;
            let values = self.port_values(context);
            let wave = |v: Real, i: Real| (v * m.admit + i) * m.attenuation;
            let [pv1, pi1, pv2, pi2] = history.sample(index);
            let [ov1, oi1, ov2, oi2] = history.sample(index - 1);
            let i1 = wave(values[2], values[3]);
            let i2 = wave(pv2, pi2);
            let i3 = wave(ov2, oi2);
            let i4 = wave(values[0], values[1]);
            let i5 = wave(pv1, pi1);
            let i6 = wave(ov1, oi1);
            let h = context.time - times[index];
            let h_old = times[index] - times[index - 1];
            let (d1, d2) = ((i1 - i2) / h, (i2 - i3) / h_old);
            let (d3, d4) = ((i4 - i5) / h, (i5 - i6) / h_old);
            (self.slope_changed(d1, d2) || self.slope_changed(d3, d4)).then_some(m.td)
        };
        if m.case == LtraCase::Rlc && !m.trunc_dont_cut {
            // ltraset.c: compactrel/compactabs default to reltol/abstol.
            let reltol = if m.st_line_reltol == 0. {
                context.reltol
            } else {
                m.st_line_reltol
            };
            let abstol = if m.st_line_abstol == 0. {
                context.abstol
            } else {
                m.st_line_abstol
            };
            let safe = convolution::rlc_max_safe_step(m.td, m.alpha, m.beta, reltol, abstol);
            bound = Some(bound.map_or(safe, |b: Real| b.min(safe)));
        }
        Ok(bound)
    }
}

// Indexed traversal avoids constructing a split-ring iterator for short
// histories. The same scalar implementation also serves as the numerical
// reference for the vector kernel.
fn scalar_history_sums(
    history: DelaySamples<'_, 4>,
    weights: &[[Real; 3]],
    initial: [Real; 4],
    coupled_end: usize,
    mut sums: [[Real; 2]; 3],
) -> [[Real; 2]; 3] {
    for (k, components) in [[0, 2], [3, 1], [2, 0]].into_iter().enumerate() {
        let end = if k == 0 {
            weights.len() - 1
        } else {
            coupled_end
        };
        for i in (1..=end).rev() {
            let weight = weights[i][k];
            if weight != 0. {
                let row = history.sample(i);
                for (sum, component) in sums[k].iter_mut().zip(components) {
                    *sum += weight * (row[component] - initial[component]);
                }
            }
        }
    }
    sums
}

fn reduce_history(
    history: DelaySamples<'_, 4>,
    weights: &[[Real; 3]],
    initial: [Real; 4],
    coupled_end: usize,
    sums: [[Real; 2]; 3],
) -> [[Real; 2]; 3] {
    if weights.len() < 32 {
        scalar_history_sums(history, weights, initial, coupled_end, sums)
    } else {
        convolution::history_sums(history.rows(), weights, initial, coupled_end, sums)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::netlist::{Parser, source::parse_deck_text};
    use std::path::Path;

    fn line(model: &str) -> LossyLine {
        let deck = Parser::new()
            .parse_deck(&parse_deck_text(
                Path::new("history.cir"),
                &format!("history\no1 a 0 b 0 m\n.model m ltra {model}\n.end\n"),
            ))
            .unwrap();
        let models = ModelResolver::new(&deck.models).unwrap();
        crate::primitives::NodeTable::default()
            .transaction(|nodes| {
                LossyLine::instantiate(&deck.devices[0], nodes, &models, &ModelContext::default())
            })
            .unwrap()
    }

    fn waveform_history(count: i32, wrapped: bool) -> WaveformHistory {
        let row = |i| {
            let x = Real::from(i) * 1e-2;
            (
                Real::from(i) * 1e-11,
                vec![
                    x.sin(),
                    0.02 * x.cos(),
                    (0.7 * x).sin(),
                    0.02 * (0.7 * x).cos(),
                ],
            )
        };
        let mut history = WaveformHistory::new(4);
        history
            .apply(&WaveformUpdate {
                reset: Some((0..count).map(row).collect()),
                ..WaveformUpdate::default()
            })
            .unwrap();
        if wrapped {
            for i in count..count + count.div_euclid(2) {
                history
                    .apply(&WaveformUpdate {
                        drop_front: 1,
                        append: Some(row(i)),
                        ..WaveformUpdate::default()
                    })
                    .unwrap();
            }
        }
        history
    }

    #[test]
    fn batched_history_matches_scalar_order_and_delayed_cutoff() {
        for count in [1, 2, 7, 8, 9, 17, 31, 32, 33, 257] {
            for wrapped in [false, true] {
                let history = waveform_history(count, wrapped);
                let view = history.samples::<4>().unwrap();
                let initial = view.sample(0);
                let weights: Vec<_> = (0..count)
                    .map(|i| {
                        let value = if i % 4 == 0 {
                            -0.
                        } else {
                            10_f64.powi(i % 400 - 200)
                        };
                        [value, if i % 3 == 0 { -value } else { 0. }, -value]
                    })
                    .collect();
                let seeds = [[-0., 0.], [1., -2.], [3., -4.]];
                for end in [0, weights.len() / 2, weights.len() - 1] {
                    let expected = scalar_history_sums(view, &weights, initial, end, seeds);
                    let got = convolution::history_sums(view.rows(), &weights, initial, end, seeds);
                    assert_eq!(
                        got.map(|p| p.map(Real::to_bits)),
                        expected.map(|p| p.map(Real::to_bits))
                    );
                    assert_eq!(
                        reduce_history(view, &weights, initial, end, seeds)
                            .map(|p| p.map(Real::to_bits)),
                        expected.map(|p| p.map(Real::to_bits))
                    );
                }
            }
        }
    }

    #[test]
    fn zero_weights_do_not_propagate_overflow_or_change_signed_zero() {
        let rows = [[-Real::MAX, 1., Real::MAX, -1.]; 12];
        let initial = [Real::MAX, 0., -Real::MAX, 0.];
        let mut weights = [[0.; 3]; 12];
        weights[11] = [1., 2., -0.];
        weights[10] = [0., 0., 1.]; // Outside the coupled cutoff.
        let seeds = [[0., 0.], [0., 0.], [-0., -0.]];
        let sums = convolution::history_sums(rows.into_iter(), &weights, initial, 0, seeds);
        assert_eq!(sums[0], [Real::NEG_INFINITY, Real::INFINITY]);
        assert_eq!(sums[1].map(Real::to_bits), [0.; 2].map(Real::to_bits));
        assert_eq!(sums[2].map(Real::to_bits), seeds[2].map(Real::to_bits));
        let sums = convolution::history_sums(rows.into_iter(), &weights, initial, 11, seeds);
        assert_eq!(sums[1], [-2., 2.]);
        assert_eq!(sums[2], [Real::INFINITY, Real::NEG_INFINITY]);
    }

    #[test]
    #[ignore = "optimized history-only benchmark; run explicitly"]
    fn history_reduction_benchmark() {
        use std::{hint::black_box, time::Instant};
        for count in [8_i32, 128, 2048, 16384] {
            for wrapped in [false, true] {
                let history = waveform_history(count, wrapped);
                let view = history.samples::<4>().unwrap();
                let initial = view.sample(0);
                let now = history.last_time().unwrap() + 1e-11;
                let times = view.times();
                let mut coeff = convolution::Coefficients::new();
                for kind in ["RC", "RLC"] {
                    match kind {
                        "RC" => {
                            convolution::rc_coefficients(1e-12, 4e-8, now, times, 0., &mut coeff);
                        }
                        "RLC" => {
                            convolution::rlc_coefficients(
                                5e-9, 2e8, 2e8, now, times, 0., &mut coeff,
                            );
                        }
                        _ => unreachable!("fixed workload cases"),
                    }
                    let end = coeff.aux_index;
                    let seeds = [[0.; 2], [1e-3, -1e-3], [2e-3, -2e-3]];
                    let old = scalar_history_sums(view, &coeff.weights, initial, end, seeds);
                    let new = reduce_history(view, &coeff.weights, initial, end, seeds);
                    assert_eq!(
                        old.map(|p| p.map(Real::to_bits)),
                        new.map(|p| p.map(Real::to_bits))
                    );
                    let iterations = (1_048_576 / count).max(4);
                    for trial in 0..3 {
                        // Alternate order to reduce shared-host frequency and scheduling bias.
                        for vector in [trial % 2 == 0, trial % 2 != 0] {
                            let start = Instant::now();
                            for _ in 0..iterations {
                                if vector {
                                    black_box(reduce_history(
                                        black_box(view),
                                        black_box(&coeff.weights),
                                        initial,
                                        end,
                                        seeds,
                                    ));
                                } else {
                                    black_box(scalar_history_sums(
                                        black_box(view),
                                        black_box(&coeff.weights),
                                        initial,
                                        end,
                                        seeds,
                                    ));
                                }
                            }
                            println!(
                                "history_reduction,{kind},{count},{wrapped},{vector},{iterations},{}",
                                start.elapsed().as_nanos()
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn delayed_values_preserve_strict_bracket_and_mixed_fallback() {
        let mut line = line("l=1 c=1 len=1 mixedinterp");
        let mut history = WaveformHistory::new(4);
        // At t=2.5 the interpolated third component overshoots the [0,1]
        // interval, whereas the first component follows a quadratic.
        for (time, values) in [
            (0., vec![0., 0., 0., 0.]),
            (1., vec![1., 1., 1., -1.]),
            (2., vec![4., 2., 1., -2.]),
        ] {
            history
                .apply(&WaveformUpdate {
                    append: Some((time, values)),
                    ..WaveformUpdate::default()
                })
                .unwrap();
        }
        let view = history.samples::<4>().unwrap();
        assert_eq!(line.delayed_values(view, 2.).unwrap(), [1., 1., 1., -1.]);
        assert_eq!(line.delayed_values(view, 3.).unwrap(), [4., 2., 1., -2.]);
        assert_eq!(
            line.delayed_values(view, 2.5).unwrap(),
            [2.25, 1.5, 1., -1.5]
        );
        // Extrapolation uses the last two accepted samples in linear mode.
        line.model.interpolation = Interpolation::Linear;
        assert_eq!(line.delayed_values(view, 4.).unwrap(), [7., 3., 1., -3.]);
        assert!(line.delayed_values(view, 1.).is_err());
    }

    #[test]
    fn rejected_loads_leave_history_unchanged_for_every_line_case() {
        for model in [
            "l=1 c=1 len=1",
            "r=0.1 l=1 c=1 len=1",
            "r=1 c=1 len=1",
            "r=1 g=1 len=1",
        ] {
            let line = line(model);
            let mut history = WaveformHistory::new(4);
            for time in 0..4 {
                let t = Real::from(time);
                history
                    .apply(&WaveformUpdate {
                        append: Some((t, vec![t, 0.5 * t, -t, -0.5 * t])),
                        ..WaveformUpdate::default()
                    })
                    .unwrap();
            }
            let before = history.clone();
            let first = line
                .transient_load(
                    &history,
                    3.25,
                    &mut convolution::Coefficients::new(),
                    &mut LtraWork::default(),
                )
                .unwrap();
            let _discarded = line
                .transient_load(
                    &history,
                    3.75,
                    &mut convolution::Coefficients::new(),
                    &mut LtraWork::default(),
                )
                .unwrap();
            let retry = line
                .transient_load(
                    &history,
                    3.25,
                    &mut convolution::Coefficients::new(),
                    &mut LtraWork::default(),
                )
                .unwrap();
            assert_eq!(history, before);
            assert_eq!(
                first.first.map(Real::to_bits),
                retry.first.map(Real::to_bits)
            );
            assert_eq!(
                first.inputs.map(Real::to_bits),
                retry.inputs.map(Real::to_bits)
            );
        }
    }
}
