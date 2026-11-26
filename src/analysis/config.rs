//! Typed run configuration resolved from a deck's `.option` cards.
//!
//! C: `INPdoOpts()` (`inpdoopt.c`) applies settings left to right through
//! `CKTsetOpt()` (`cktsopt.c`); the front end first handles its own variables
//! (`spiceif.c::if_option`, `options.c`). The port differs deliberately: nothing
//! is applied to global state, unknown or unimplemented options are errors
//! instead of warnings, and every accepted name either takes effect or is a
//! documented no-op ([`RunConfig::ignored`]) justified by C behaviour.
//!
//! # Options with an effect
//!
//! | Option | Effect |
//! | --- | --- |
//! | `temp`, `tnom` | [`AnalysisContext`] circuit/nominal temperature (Celsius, finite, above absolute zero) |
//! | `gmin` | [`AnalysisContext::gmin`]: junction minimum conductance of every diode/BJT/MOS1 junction in all analyses (C `CKTgmin`, default 1e-12 S; finite, `>= 0`) |
//! | `rshunt`, `cshunt` | noiseless resistive and capacitive shunts on physical voltage nodes |
//! | `gshunt` | retained regularization conductance after nodal gmin recovery |
//! | `pivtol`, `pivrel` | [`AnalysisContext::pivot`]: SPARSE threshold-pivoting limits used whenever a circuit matrix is (re)ordered in any analysis (C `CKTpivotAbsTol`/`CKTpivotRelTol`, `SMPreorder`/`SMPcReorder`; defaults 1e-13 and 1e-3). `pivtol` must be finite and `>= 0`, `pivrel` in `(0, 1]`; C's `spOrderAndFactor` would silently use its default for other values |
//! | `reltol` | circuit relative tolerance, including diode temperature matching; default for Newton and transient truncation/convergence |
//! | `vntol` | voltage absolute tolerance (companion Newton test; DC/AC Newton) |
//! | `abstol` | branch-current absolute tolerance (companion truncation/Newton test; DC/AC Newton) |
//! | `chgtol`, `trtol` | companion local-truncation-error charge floor and overestimation factor |
//! | `minbreak` | minimum analog breakpoint spacing (seconds; zero selects the default) |
//! | `newtrunc`, `ltereltol`, `lteabstol`, `ltetrtol` | optional voltage predictor LTE bound |
//! | `method`, `maxord` | retained as [`RunConfig::method`]/[`RunConfig::maxord`] and forwarded to the companion driver (`trap`/`trapezoidal`/`gear`, `maxord` 1 to 6; `dctran.c` never runs above order 2, so 2 to 6 behave alike) |
//! | `xmu` | companion trapezoidal weighting (`nicomcof.c`, default 0.5, `0..=0.5`) |
//! | `itl1` | Newton iteration limit of the direct DC solve and of the gmin strategies' closing solve (`maxiter`) for `.op`/`.dc`/`.ac` and the companion `.tran` initial bias (C `CKTdcMaxIter`; `dcop.c`, `acan.c`, `dctran.c` call `CKTop`); default 100 |
//! | `itl2` | Newton limit of every other gmin/source-stepping stage of that DC bias (`stagemaxiter`, C `CKTdcTrcvMaxIter` in `cktop.c`); as written, the step adaptation of `dynamic_gmin`/`new_gmin`/`gillespie_src` (`adaptiter`, `iters <= itl2/4`); on `.dc` also the warm-started solve at every sweep point after the first (`trcvmaxiter`, `dctrcurv.c`), whose failure falls back to the full bias |
//! | `itl4` | companion `.tran` Newton iterations per timepoint (`tranmaxiter`; effective default 100 as in C) |
//! | `srcsteps` (alias `itl6`) | C's source stepping: `1` (default) `gillespie_src`, `n > 1` `spice3_src` with `n` equal increments, `0` disables (`0..=1000`) |
//! | `gminsteps`, `gminfactor` | C's gmin stepping: `1` (default) `dynamic_gmin` then `new_gmin`, `n > 1` `spice3_gmin` from `gmin * gminfactor^n`, `0` disables (`0..=100`); ratio `1 < factor <= 1e6`, default 10 |
//! | `noopiter` (flag) | skip the direct Newton attempt of every DC bias (C `CKTnoOpIter`) |
//!
//! `itl1`/`itl2`/`itl4` take integers in `0..=10000` and are stored as C's
//! *effective* limit `max(n, 100)`: `NIiter()` (`niiter.c`) raises every limit
//! below 100 to 100, so smaller values change nothing in C or here (C's
//! nominal `itl4` default of 10 and `itl2` default of 50 are effectively 100
//! too); the raw `itl2` is kept separately for the step adaptation. When a deck
//! sets `itl1` or `itl2`, the continuation stages use `itl2` (default 100) as in
//! C; without either, every stage uses `itl1`'s default 100, which is the same
//! number. As C's `IF_INTEGER` options, a non-integer is rounded with
//! `floor(0.5 + value)` (`itl4=2.5` is 3).
//!
//! The DC options reach `.op`, `.dc`, `.ac` and the companion `.tran` initial
//! bias. They follow C's `CKTop` strategies (#106); the port-only request keys
//! `continuation=ladder` and `limiting=global` select the port's earlier fixed
//! ladders and global damping instead. C's `OPtran` fallback,
//! `oldlimit`, predictor and bypass are not ported; see
//! `docs/port/DC_CONTINUATION.md`.
//!
//! # Documented no-ops
//!
//! Accepted, validated, recorded in [`RunConfig::ignored`] with the reason, and
//! without numerical effect in C or in this port:
//!
//! * flags `acct`, `noacct`, `list`, `nomod`, `nopage`, `node`, `opts`,
//!   `noinit`, `norefvalue`: front-end print controls (`spiceif.c::if_option`);
//! * `itl3`, `itl5`, `cptime`, `limtim`, `limpts`, `lvlcod`, `lvltim`: C ignores
//!   them (`OPTtbl` entries without `IF_SET`; `if_option` warns
//!   "unsupported"/"obsolete");
//! * `post`, `ingold` (flag or value): plain front-end variables nothing in
//!   ngspice reads;
//! * `indverbosity=N` (a non-negative integer): only selects which stderr
//!   diagnostics C's inductive-system check prints (`muttemp.c`); the port
//!   prints none and always rejects a coupled inductance matrix that is not
//!   positive semidefinite (`docs/port/MUTUAL_INDUCTANCE.md`);
//! * `bypass=0`: C's default (`cktntask.c`); this port never bypasses device
//!   evaluation. Any other `bypass` is `NotYetPorted`.
//!
//! Every other name from `cktsopt.c` (`oldlimit`, ...)
//! and the front-end variables with an effect (`filetype`, `numdgt`,
//! `savecurrents`, ...) is [`SpiceError::NotYetPorted`]; names
//! absent from both are parse errors. `no_auto_gnd` is a front-end variable in
//! C, not an `.option`: use `Parser::with_auto_gnd`.
//!
//! Static statistical realization accepts top-level literal `seed` and
//! `rndseed` at deck load. See `netlist::eval`; the run config validates
//! these frontend inputs without changing a prepared circuit's realization.
//!
//! # Expression values
//!
//! `{expr}` and `'expr'` values (C: numparam substitutes both) are evaluated by
//! [`RunConfig::from_netlist`] against the deck's top-level `.param` scope;
//! [`RunConfig::from_options`] has no scope and rejects them. `method` takes a
//! word, never an expression.
//!
//! `.ic` and `.nodeset` cards are not options: [`RunConfig::from_netlist`]
//! evaluates them against `.param` and [`RunConfig::plan`] attaches them to
//! every analysis request (`AnalysisInput::initial_conditions`/`nodesets`).
//!
//! # Precedence
//!
//! Highest first: explicit [`RunOverrides`] (temperatures) or explicit analysis
//! request arguments (`rtol=`, `vntol=`, `abstol=`, `chgtol=`, `trtol=`, `method=`,
//! `maxord=`, `xmu=`, `tranmaxiter=`, `maxsteps=`, `maxiter=`, `srcsteps=`,
//! `gminsteps=`, `gminfactor=`, `noopiter=`, `adaptiter=`, `trcvmaxiter=`),
//! then the deck's options, then driver defaults (27 C). Each name is resolved independently, so a request
//! `gminsteps=` combines with a deck `gminfactor`.
//! Defaults are ngspice's (reltol 1e-3, vntol 1e-6, abstol 1e-12, chgtol 1e-14,
//! trtol 7). A request's `backend=` is not interpreted here: the `.tran`
//! dispatcher accepts only the companion driver and rejects `backend=diffsol`
//! explicitly, whatever the deck's options.
//! Repeated options override in deck order, last wins (`itl6` and `srcsteps`
//! are one setting); giving one name both as a flag and with a value is a
//! conflict error. A [`RunConfig`] is a plain value computed per deck, so
//! nothing leaks between decks.

use crate::analysis::IntegrationFamily;
use crate::devices::Circuit;
use crate::netlist::ast::{AnalysisCard, Netlist, OptionCard, OptionSetting};
use crate::netlist::eval::{EvalBudget, ParamScope};
use crate::primitives::{Real, SourceLoc, SpiceError, SpiceResult, parse_spice_number};

use crate::analysis::{AnalysisContext, AnalysisInput, NodeCondition};

const C_REFERENCE: &str = "src/spicelib/analysis/cktsopt.c, src/spicelib/parser/inpdoopt.c";
const FRONTEND_REFERENCE: &str =
    "src/frontend/options.c, src/frontend/spiceif.c, src/frontend/inp.c";

/// Names in `cktsopt.c`'s `OPTtbl` that this port does not implement yet.
const KNOWN_UNIMPLEMENTED: &[&str] = &[
    "oldlimit",
    "numdgt",
    "defm",
    "defl",
    "defw",
    "defad",
    "defas",
    "badmos3",
    "trytocompact",
    "copynodesets",
    "nodedamping",
    "linesearch",
    "absdv",
    "reldv",
    "noopac",
    "epsmin",
    "sparse",
    "klu",
    "klu_memgrow_factor",
    "maxopalter",
    "maxevtiter",
    "noopalter",
    "ramptime",
    "convlimit",
    "convstep",
    "convabsstep",
    "autopartial",
];

/// Front-end variables that ngspice reads from `.options` with an effect on
/// output or setup, not implemented here.
const FRONTEND_UNIMPLEMENTED: &[&str] = &[
    "savecurrents",
    "scalm",
    "seedinfo",
    "interp",
    "warn",
    "measureprec",
    "rawfileprec",
    "strict_errorhandling",
];

/// Front-end print-control flags: no numerical effect in C (`spiceif.c`).
const PRINT_FLAGS: &[&str] = &[
    "acct",
    "noacct",
    "list",
    "nomod",
    "nopage",
    "node",
    "opts",
    "noinit",
    "norefvalue",
];
const PRINT_FLAG_REASON: &str = "front-end print control (spiceif.c if_option); no numerical \
                                 effect, and this port prints no such listing or accounting";

/// `OPTtbl` entries C parses but never applies (no `IF_SET`), with whether
/// the value must be an integer.
const IGNORED_BY_C: &[(&str, bool)] = &[
    ("itl3", true),
    ("itl5", true),
    ("cptime", false),
    ("limtim", true),
    ("limpts", true),
    ("lvlcod", true),
    ("lvltim", true),
];
const IGNORED_BY_C_REASON: &str = "ignored by ngspice (OPTtbl entry without IF_SET; \
                                   spiceif.c reports it unsupported or obsolete)";
const UNREAD_REASON: &str = "plain ngspice front-end variable that nothing reads";
const INDVERBOSITY_REASON: &str = "controls only the stderr diagnostics of C's inductive-system \
                                   check (muttemp.c, CKTindverbosity); this port never prints \
                                   them and always rejects a coupled inductance matrix that is \
                                   not positive semidefinite";
const BYPASS_REASON: &str = "bypass=0 is ngspice's default (cktntask.c); this port never \
                             bypasses device evaluation";

/// Highest `maxord` ngspice accepts (`cktsopt.c` clamps to 1..=6).
const MAX_ORD: u8 = 6;

/// Settings that explicitly beat the deck's `.option` temperatures.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct RunOverrides {
    /// Circuit temperature in Celsius, replacing a deck `temp`.
    pub temperature: Option<Real>,
    /// Nominal temperature in Celsius, replacing a deck `tnom`.
    pub nominal_temperature: Option<Real>,
}

/// Optional voltage-predictor truncation controls (`CKTtrunc`).
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct VoltageLteOptions {
    /// Whether voltage-controlled truncation is enabled.
    pub enabled: Option<bool>,
    /// Relative tolerance for voltage predictor error.
    pub relative: Option<Real>,
    /// Absolute tolerance in volts.
    pub absolute: Option<Real>,
    /// Predictor-error overestimation factor.
    pub factor: Option<Real>,
}

/// Deck-supplied transient settings. `None` leaves the driver default, so the
/// BDF defaults are unchanged unless a deck sets them.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct TransientSettings {
    /// `vntol` → voltage absolute tolerance.
    pub vntol: Option<Real>,
    /// `abstol` → branch-current absolute tolerance.
    pub abstol: Option<Real>,
    /// `chgtol` → companion charge/flux floor in truncation control.
    pub chgtol: Option<Real>,
    /// `trtol` → companion truncation-error overestimation factor.
    pub trtol: Option<Real>,
    /// Minimum spacing between analog breakpoints in seconds; zero selects the default.
    pub minbreak: Option<Real>,
    /// Voltage-based LTE controls, independent of charge tolerances.
    pub voltage_lte: VoltageLteOptions,
    /// `itl4` → companion Newton iterations per timepoint (`tranmaxiter`),
    /// stored as C's effective `max(itl4, 100)` (see [`NIITER_MIN_ITERATIONS`]).
    pub itl4: Option<usize>,
    /// `xmu` → companion trapezoidal weighting.
    pub xmu: Option<Real>,
}

/// `NIiter()` (`niiter.c`) raises every Newton iteration limit below this to
/// it; deck `itl1`/`itl2`/`itl4` values are stored as C's effective limit.
pub const NIITER_MIN_ITERATIONS: usize = 100;

/// Option names that configure the DC Newton/continuation solve, in no
/// particular order (`itl6` is stored as `srcsteps`).
const DC_OPTIONS: [&str; 8] = [
    "itl1",
    "itl2",
    "srcsteps",
    "itl6",
    "gminsteps",
    "gminfactor",
    "gshunt",
    "noopiter",
];

/// C's default `itl2` (`cktntask.c`: `TSKdcTrcvMaxIter = 50`) as `NIiter()`
/// applies it: raised to [`NIITER_MIN_ITERATIONS`].
pub const C_DEFAULT_ITL2_EFFECTIVE: usize = NIITER_MIN_ITERATIONS;

/// Deck-supplied DC settings (last occurrence wins). `None` leaves the default;
/// `Some(0)` disables source/gmin stepping.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct DcOptions {
    /// `itl1` → Newton iteration limit of the direct DC solve (C
    /// `CKTdcMaxIter`), stored as C's effective `max(itl1, 100)` (see
    /// [`NIITER_MIN_ITERATIONS`]).
    pub itl1: Option<usize>,
    /// `itl2` → Newton limit of every gmin/source-stepping stage (C
    /// `CKTdcTrcvMaxIter` in `cktop.c`) and of the `.dc` warm start at points
    /// after the first (`dctrcurv.c`), stored as C's effective `max(itl2, 100)`.
    pub itl2: Option<usize>,
    /// `itl2` as written: the adaptation base of the ngspice continuation
    /// strategies (`iters <= itl2 / 4`, [`crate::analysis::bias::NgspiceStepping`]).
    pub itl2_written: Option<usize>,
    /// `noopiter` → skip the direct Newton attempt of every DC bias (C
    /// `CKTnoOpIter`).
    pub noopiter: bool,
    /// `srcsteps`/`itl6` → equal source-stepping increments.
    pub srcsteps: Option<usize>,
    /// `gminsteps` → gmin-stepping stages.
    pub gminsteps: Option<usize>,
    /// `gminfactor` → ratio between consecutive gmin stages.
    pub gminfactor: Option<Real>,
    /// SPARSE continuation diagonal after nodal gmin stepping (siemens).
    pub gshunt: Option<Real>,
}

impl DcOptions {
    /// The continuation policy these options resolve to over the defaults.
    ///
    /// # Errors
    /// An invalid combination, e.g. a factor whose schedule underflows.
    pub fn policy(&self) -> SpiceResult<crate::analysis::bias::ContinuationPolicy> {
        let mut policy = crate::analysis::bias::ContinuationPolicy::from_ngspice_steps(
            self.srcsteps,
            self.gminsteps,
            self.gminfactor,
            self.itl2_written,
        )?;
        policy.skip_direct = self.noopiter;
        policy.gshunt = self.gshunt.unwrap_or(0.);
        Ok(policy)
    }
}

/// One accepted option occurrence, kept in deck order for diagnostics.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AppliedOption {
    /// Lowercased option name.
    pub name: String,
    /// Value text as written (an expression keeps its `{...}`/`'...'` text).
    pub value: String,
    /// Where the option name was written.
    pub location: SourceLoc,
}

/// An accepted option occurrence that has no effect, with the C-based reason.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IgnoredOption {
    /// Lowercased option name.
    pub name: String,
    /// Value text as written; `None` for a flag.
    pub value: Option<String>,
    /// Why it has no effect (C behaviour it mirrors).
    pub reason: &'static str,
    /// Where the option name was written.
    pub location: SourceLoc,
}

/// The original option text and, when applicable, its evaluated expression.
struct OptionInput<'a> {
    setting: &'a OptionSetting,
    evaluated: Option<Real>,
}

impl OptionInput<'_> {
    fn number(&self, what: &str) -> SpiceResult<Real> {
        let setting = self.setting;
        let name = setting.name.as_str();
        let value_location = setting
            .value
            .as_ref()
            .map_or(&setting.location, |value| &value.location);
        let parsed = match (self.evaluated, &setting.value) {
            (Some(value), _) => Some(value),
            (None, Some(value)) => parse_spice_number(&value.text),
            (None, None) => None,
        };
        parsed.filter(|v| v.is_finite()).ok_or_else(|| {
            SpiceError::parse(
                value_location.clone(),
                format!(
                    "option '{name}' needs {what}, not '{}'",
                    setting.value.as_ref().map_or("", |v| v.text.as_str())
                ),
            )
        })
    }

    fn whole(&self, lowest: u32, highest: u32) -> SpiceResult<u32> {
        let name = self.setting.name.as_str();
        let value_location = self
            .setting
            .value
            .as_ref()
            .map_or(&self.setting.location, |value| &value.location);
        let value = self.number("an integer")?;
        integer_value(value)
            .filter(|n| (lowest..=highest).contains(n))
            .ok_or_else(|| {
                SpiceError::parse(
                    value_location.clone(),
                    format!(
                        "option '{name}' must be an integer in {lowest}..={highest}, not {value}"
                    ),
                )
            })
    }

    fn whole_usize(&self, lowest: usize, highest: usize) -> SpiceResult<usize> {
        let name = self.setting.name.as_str();
        let value_location = self
            .setting
            .value
            .as_ref()
            .map_or(&self.setting.location, |value| &value.location);
        let value = self.number("an integer")?;
        integer_value(value)
            .and_then(|count| usize::try_from(count).ok())
            .filter(|count| (lowest..=highest).contains(count))
            .ok_or_else(|| {
                SpiceError::parse(
                    value_location.clone(),
                    format!(
                        "option '{name}' must be an integer in {lowest}..={highest}, not {value}"
                    ),
                )
            })
    }
}

/// Independent admission limits for aggregate analysis-plan preparation.
/// Each spectral/DC grid also retains its own reference-specific point cap.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PlanningLimits {
    /// Number of analysis jobs, including point-free and adaptive ones.
    pub max_jobs: usize,
    /// Total deterministic frequency and DC Cartesian points.
    pub max_grid_points: usize,
}

impl Default for PlanningLimits {
    fn default() -> Self {
        Self {
            max_jobs: 10_000,
            max_grid_points: 10_000_000,
        }
    }
}

/// Resolved, validated run configuration for one deck.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct RunConfig {
    fourier: Option<crate::analysis::fourier::FourierSettings>,
    raw_format: Option<crate::analysis::RawFormat>,
    squared_noise: bool,
    spice3_noise: bool,
    keep_op_info: bool,
    context: AnalysisContext,
    transient: TransientSettings,
    dc: DcOptions,
    method: Option<IntegrationFamily>,
    maxord: Option<(u8, SourceLoc)>,
    applied: Vec<AppliedOption>,
    ignored: Vec<IgnoredOption>,
    /// Evaluated `.ic` entries in deck order, attached to every request.
    initial_conditions: Vec<NodeCondition>,
    /// Evaluated `.nodeset` entries in deck order, attached to every request.
    nodesets: Vec<NodeCondition>,
    params: Option<std::sync::Arc<ParamScope>>,
}

impl RunConfig {
    /// Resolve all `.option` cards of a deck with no explicit overrides,
    /// evaluating `{expr}`/`'expr'` option values against top-level `.param`.
    ///
    /// # Errors
    /// Unknown, unimplemented, malformed, out-of-range or conflicting options,
    /// and option expressions that fail to evaluate.
    pub fn from_netlist(netlist: &Netlist) -> SpiceResult<Self> {
        let scope = std::sync::Arc::new(ParamScope::for_netlist(netlist)?);
        Self::from_netlist_with_scope(netlist, scope)
    }

    /// Resolve a candidate's options and node hints against an already
    /// evaluated immutable scope; no second dependency walk or parameter
    /// evaluation. Caller controls when setup-shape changes invalidate plans.
    pub(crate) fn from_netlist_with_scope(
        netlist: &Netlist,
        scope: std::sync::Arc<ParamScope>,
    ) -> SpiceResult<Self> {
        // Body front-end cards follow expanded instance/card order, not definition order.
        let expanded = if netlist.subcircuits.iter().any(|sub| {
            !sub.options.is_empty()
                || !sub.initial_conditions.is_empty()
                || !sub.nodesets.is_empty()
        }) {
            let literal = crate::netlist::elaborate::literalize_with(
                netlist,
                std::sync::Arc::clone(&scope),
                &mut EvalBudget::default(),
            )?;
            Some(crate::devices::subckt::expand_subcircuits(
                &literal.netlist,
                &literal.scope,
                crate::devices::subckt::SubcircuitLimits::default(),
            )?)
        } else {
            None
        };
        let options = expanded
            .as_ref()
            .map_or(netlist.options.as_slice(), |expanded| &expanded.options);
        let mut config = Self::resolve(options, &RunOverrides::default(), Some(&scope))?;
        let (initial, nodesets) = crate::netlist::elaborate::literalize_node_hints(
            netlist,
            &scope,
            &mut EvalBudget::default(),
        )?;
        let conditions = |cards: &[crate::netlist::ast::NodeHintCard]| {
            cards
                .iter()
                .flat_map(|card| &card.entries)
                .filter_map(|entry| {
                    Some(NodeCondition {
                        node: entry.node.clone(),
                        value: entry.literal()?,
                        location: entry.location.clone(),
                    })
                })
                .collect::<Vec<_>>()
        };
        config.initial_conditions = conditions(
            expanded
                .as_ref()
                .map_or(&initial, |expanded| &expanded.initial_conditions),
        );
        config.nodesets = conditions(
            expanded
                .as_ref()
                .map_or(&nodesets, |expanded| &expanded.nodesets),
        );
        config.params = Some(scope);
        Ok(config)
    }

    /// Replace only the already validated parameter environment. Numeric
    /// candidates with unchanged options and topology retain the same run
    /// settings but must expose their *current* parameter scope to objective
    /// measurement and subsequent analysis binding.
    pub(crate) fn with_parameter_scope(mut self, scope: std::sync::Arc<ParamScope>) -> Self {
        self.params = Some(scope);
        self
    }

    /// Explicit Fourier front-end settings.
    #[must_use = "the returned option distinguishes explicit Fourier settings from front-end defaults"]
    pub const fn fourier_settings(&self) -> Option<crate::analysis::fourier::FourierSettings> {
        self.fourier
    }

    /// Rawfile encoding selected by the front-end `set filetype` variable.
    #[must_use = "the returned option determines whether the deck specifies a rawfile encoding"]
    pub const fn raw_format(&self) -> Option<crate::analysis::RawFormat> {
        self.raw_format
    }

    /// The deck's resolved top-level `.param` scope (only for configs built by
    /// [`Self::from_netlist`]).
    #[must_use = "the returned option identifies the resolved top-level parameter scope when available"]
    pub fn params(&self) -> Option<&ParamScope> {
        self.params.as_deref()
    }

    /// Resolve option cards in order, then apply `overrides`.
    ///
    /// # Errors
    /// As [`Self::from_netlist`]; overrides must also be valid temperatures.
    /// Expression values are [`SpiceError::Unsupported`] here (no `.param`
    /// scope): use [`Self::from_netlist`].
    pub fn from_options(cards: &[OptionCard], overrides: &RunOverrides) -> SpiceResult<Self> {
        Self::resolve(cards, overrides, None)
    }

    fn resolve(
        cards: &[OptionCard],
        overrides: &RunOverrides,
        scope: Option<&ParamScope>,
    ) -> SpiceResult<Self> {
        let mut config = Self::default();
        let mut flagged = std::collections::BTreeSet::new();
        let mut valued = std::collections::BTreeSet::new();
        let mut budget = EvalBudget::default();
        for setting in cards.iter().flat_map(|card| &card.settings) {
            let seen = if setting.value.is_some() {
                &mut valued
            } else {
                &mut flagged
            };
            seen.insert(setting.name.as_str());
            if flagged.contains(setting.name.as_str()) && valued.contains(setting.name.as_str()) {
                return Err(SpiceError::parse(
                    setting.location.clone(),
                    format!(
                        "option '{}' is given both as a flag and with a value",
                        setting.name
                    ),
                ));
            }
            let evaluated = match (&setting.expression, scope) {
                (None, _) => None,
                // Unknown/unported names fail as such in `apply`, before any
                // evaluation error could mask them.
                (Some(_), _) if !accepted(&setting.name) => None,
                (Some(_), _) if setting.name == "method" => {
                    return Err(SpiceError::parse(
                        setting.location.clone(),
                        "option 'method' takes a word (trap, trapezoidal or gear), not an \
                         expression",
                    ));
                }
                (Some(expression), Some(scope)) => Some(
                    scope
                        .evaluate(expression, &mut budget)
                        .map_err(|error| match error {
                            SpiceError::Parse { location, message } => SpiceError::parse(
                                location,
                                format!(
                                    "{message}\n  while evaluating option '{}' at {}",
                                    setting.name, setting.location
                                ),
                            ),
                            other => other,
                        })?,
                ),
                (Some(_), None) => {
                    return Err(SpiceError::Unsupported {
                        feature: format!(
                            "expression value of option '{}' needs the deck's .param scope \
                             (use RunConfig::from_netlist)",
                            setting.name
                        ),
                        location: Some(setting.location.clone()),
                    });
                }
            };
            config.apply(setting, evaluated)?;
        }
        config.validate_continuation()?;
        let unknown = SourceLoc::new(std::path::PathBuf::from("<run overrides>"), 0, 0);
        if let Some(t) = overrides.temperature {
            config.context.temperature = temperature(t, &unknown, "temperature override")?;
        }
        if let Some(t) = overrides.nominal_temperature {
            config.context.nominal_temperature =
                temperature(t, &unknown, "nominal temperature override")?;
        }
        Ok(config)
    }

    fn validate_continuation(&self) -> SpiceResult<()> {
        // The last-set counts and factor must form one valid schedule together
        // (with the deck's junction gmin, which seeds C's `spice3_gmin` ladder).
        if let Err(error) = self.dc.policy().and_then(|policy| match policy.schedule {
            crate::analysis::bias::ContinuationSchedule::Ngspice(stepping) => stepping
                .fixed_gmin_ladder(policy.gmin_seed(self.context.gmin))
                .map(|_| ()),
            crate::analysis::bias::ContinuationSchedule::Ladder { .. } => Ok(()),
        }) {
            let location = self
                .applied
                .iter()
                .rev()
                .find(|option| DC_OPTIONS.contains(&option.name.as_str()))
                .map_or_else(
                    || SourceLoc::new(std::path::PathBuf::from("<options>"), 0, 0),
                    |option| option.location.clone(),
                );
            return Err(SpiceError::parse(location, error.to_string()));
        }
        Ok(())
    }

    /// Transactionally apply an executed control option card to this snapshot.
    /// The command has already resolved scalar expressions in its own scope.
    /// Runtime records retain only the latest setter of each changed name.
    pub(crate) fn with_control_options(&self, card: &OptionCard) -> SpiceResult<Self> {
        let mut next = self.clone();
        for setting in &card.settings {
            if setting.expression.is_some() {
                return Err(SpiceError::parse(
                    setting.location.clone(),
                    "unresolved control option",
                ));
            }
            next.applied.retain(|record| record.name != setting.name);
            next.ignored.retain(|record| record.name != setting.name);
            next.apply(setting, None)?;
        }
        next.validate_continuation()?;
        Ok(next)
    }

    /// Record a validated no-op occurrence.
    fn ignore(&mut self, setting: &OptionSetting, reason: &'static str) {
        self.ignored.push(IgnoredOption {
            name: setting.name.clone(),
            value: setting.value.as_ref().map(|value| value.text.clone()),
            reason,
            location: setting.location.clone(),
        });
    }

    /// Apply one setting; the input carries both its text and evaluated expression.
    fn apply(&mut self, setting: &OptionSetting, evaluated: Option<Real>) -> SpiceResult<()> {
        let input = OptionInput { setting, evaluated };
        if self.apply_ignored_option(&input)? || self.apply_frontend_option(&input)? {
            return Ok(());
        }
        let name = setting.name.as_str();
        let location = &setting.location;
        let supported = matches!(
            name,
            "temp"
                | "tnom"
                | "gmin"
                | "gshunt"
                | "rshunt"
                | "cshunt"
                | "scale"
                | "pivtol"
                | "pivrel"
                | "reltol"
                | "vntol"
                | "abstol"
                | "chgtol"
                | "trtol"
                | "minbreak"
                | "lteabstol"
                | "ltereltol"
                | "ltetrtol"
                | "method"
                | "maxord"
                | "xmu"
                | "itl1"
                | "itl2"
                | "itl4"
                | "srcsteps"
                | "itl6"
                | "gminsteps"
                | "gminfactor"
        );
        if !supported {
            if KNOWN_UNIMPLEMENTED.contains(&name) {
                return Err(SpiceError::not_yet_ported(
                    format!("{location}: option '{name}'"),
                    C_REFERENCE,
                ));
            }
            if FRONTEND_UNIMPLEMENTED.contains(&name) {
                return Err(SpiceError::not_yet_ported(
                    format!("{location}: front-end option '{name}'"),
                    FRONTEND_REFERENCE,
                ));
            }
            return Err(SpiceError::parse(
                location.clone(),
                if name == "no_auto_gnd" {
                    "unknown option 'no_auto_gnd' (a front-end variable; use Parser::with_auto_gnd \
                     or --no-auto-gnd)"
                        .to_owned()
                } else {
                    format!("unknown option '{name}'")
                },
            ));
        }
        let Some(value) = &setting.value else {
            return Err(SpiceError::parse(
                location.clone(),
                format!("option '{name}' requires a value"),
            ));
        };
        if !self.apply_solver_control(&input, value)? {
            self.apply_real_option(&input, value)?;
        }
        self.applied.push(AppliedOption {
            name: setting.name.clone(),
            value: value.text.clone(),
            location: location.clone(),
        });
        Ok(())
    }

    /// Validate the accepted settings that C does not use for computation.
    fn apply_ignored_option(&mut self, input: &OptionInput<'_>) -> SpiceResult<bool> {
        let setting = input.setting;
        let name = setting.name.as_str();
        let location = &setting.location;
        if PRINT_FLAGS.contains(&name) {
            if setting.value.is_some() {
                return Err(SpiceError::parse(
                    location.clone(),
                    format!("option '{name}' is a flag and takes no value"),
                ));
            }
            self.ignore(setting, PRINT_FLAG_REASON);
            return Ok(true);
        }
        if let Some((_, integral)) = IGNORED_BY_C.iter().find(|(n, _)| *n == name) {
            if setting.value.is_none() {
                return Err(SpiceError::parse(
                    location.clone(),
                    format!("option '{name}' requires a value"),
                ));
            }
            if *integral {
                input.whole(0, u32::MAX)?;
            } else {
                input.number("a finite number")?;
            }
            self.ignore(setting, IGNORED_BY_C_REASON);
            return Ok(true);
        }
        match name {
            "post" | "ingold" => {
                self.ignore(setting, UNREAD_REASON);
                Ok(true)
            }
            "indverbosity" => {
                if setting.value.is_none() {
                    return Err(SpiceError::parse(
                        location.clone(),
                        "option 'indverbosity' requires an integer value",
                    ));
                }
                input.whole(0, u32::MAX)?;
                self.ignore(setting, INDVERBOSITY_REASON);
                Ok(true)
            }
            "bypass" => {
                if setting.value.is_none() {
                    return Err(SpiceError::parse(
                        location.clone(),
                        "option 'bypass' requires an integer value",
                    ));
                }
                if input.whole(0, u32::MAX)? != 0 {
                    return Err(SpiceError::not_yet_ported(
                        format!("{location}: option 'bypass' other than 0 (device bypass)"),
                        C_REFERENCE,
                    ));
                }
                self.ignore(setting, BYPASS_REASON);
                Ok(true)
            }
            _ => Ok(false),
        }
    }

    /// Resolve front-end analysis/output controls and the bare Newton skip flag.
    fn apply_frontend_option(&mut self, input: &OptionInput<'_>) -> SpiceResult<bool> {
        let setting = input.setting;
        let name = setting.name.as_str();
        let location = &setting.location;
        match name {
            "seed" | "rndseed" => {
                crate::netlist::eval::statistics::parse_seed(setting)?;
                self.applied.push(AppliedOption {
                    name: setting.name.clone(),
                    value: setting
                        .value
                        .as_ref()
                        .map_or_else(String::new, |value| value.text.clone()),
                    location: location.clone(),
                });
            }
            "nfreqs" | "nperiods" | "polydegree" | "fourgridsize" => {
                let value = input.whole(
                    u32::from(name != "polydegree"),
                    match name {
                        "nfreqs" => 101,
                        "nperiods" => 100,
                        "polydegree" => 16,
                        _ => 100_000,
                    },
                )?;
                let settings = self.fourier.get_or_insert_with(Default::default);
                match name {
                    "nfreqs" => settings.nfreqs = value,
                    "nperiods" => settings.nperiods = value,
                    "polydegree" => settings.polydegree = value,
                    _ => settings.gridsize = value,
                }
            }
            "filetype" => {
                self.raw_format = Some(match setting.value.as_ref().map(|v| v.text.as_str()) {
                    Some("ascii") => crate::analysis::RawFormat::Ascii,
                    Some("binary") => crate::analysis::RawFormat::Binary,
                    _ => {
                        return Err(SpiceError::parse(
                            location.clone(),
                            "filetype requires ascii or binary",
                        ));
                    }
                });
            }
            "sqrnoise" => {
                if setting.value.is_some() {
                    return Err(SpiceError::Unsupported {
                        feature: "use the bare boolean set sqrnoise".into(),
                        location: Some(location.clone()),
                    });
                }
                self.squared_noise = true;
            }
            "ngbehavior" => {
                let value = setting.value.as_ref().map(|v| v.text.to_ascii_lowercase());
                if value.as_deref() != Some("s3") {
                    return Err(SpiceError::Unsupported {
                        feature: "only ngbehavior=s3 for MOS1 noise compatibility is supported"
                            .into(),
                        location: Some(location.clone()),
                    });
                }
                self.spice3_noise = true;
            }
            "keepopinfo" => {
                self.keep_op_info = setting.value.is_none() || input.number("0 or 1")? != 0.;
            }
            "newtrunc" => {
                if setting.value.is_some() {
                    return Err(SpiceError::parse(
                        location.clone(),
                        "option 'newtrunc' is a flag and takes no value",
                    ));
                }
                self.transient.voltage_lte.enabled = Some(true);
                self.applied.push(AppliedOption {
                    name: setting.name.clone(),
                    value: String::new(),
                    location: location.clone(),
                });
            }
            "noopiter" => {
                if setting.value.is_some() {
                    return Err(SpiceError::parse(
                        location.clone(),
                        "option 'noopiter' is a flag and takes no value",
                    ));
                }
                self.dc.noopiter = true;
                self.applied.push(AppliedOption {
                    name: setting.name.clone(),
                    value: String::new(),
                    location: location.clone(),
                });
            }
            _ => return Ok(false),
        }
        Ok(true)
    }

    /// Apply discrete solver controls and the continuation schedule.
    fn apply_solver_control(
        &mut self,
        input: &OptionInput<'_>,
        value: &crate::netlist::ast::PositionedValue,
    ) -> SpiceResult<bool> {
        let setting = input.setting;
        let name = setting.name.as_str();
        let location = &setting.location;
        match name {
            "method" => {
                self.method = Some(IntegrationFamily::from_name(&value.text).ok_or_else(|| {
                    SpiceError::parse(
                        value.location.clone(),
                        format!(
                            "method must be trap, trapezoidal or gear, not '{}'",
                            value.text
                        ),
                    )
                })?);
            }
            "maxord" => {
                let order = input
                    .number("an integer")
                    .ok()
                    .and_then(integer_value)
                    .and_then(|n| u8::try_from(n).ok())
                    .filter(|n| (1..=MAX_ORD).contains(n))
                    .ok_or_else(|| {
                        SpiceError::parse(
                            value.location.clone(),
                            format!(
                                "maxord must be an integer in 1..={MAX_ORD}, not '{}'",
                                value.text
                            ),
                        )
                    })?;
                self.maxord = Some((order, location.clone()));
            }
            "itl1" | "itl2" | "itl4" => {
                // niiter.c: `if (maxIter < 100) maxIter = 100;` applies to all
                // three (CKTop, dctrcurv and dctran call NIiter with them), so
                // C's effective limit is never below 100.
                let written = input.whole_usize(0, crate::analysis::newton::MAX_ITERATIONS)?;
                let count = written.max(NIITER_MIN_ITERATIONS);
                match name {
                    "itl1" => self.dc.itl1 = Some(count),
                    "itl2" => {
                        self.dc.itl2 = Some(count);
                        // The adaptive strategies divide the written value
                        // (`itl2 / 4`); a zero would make every stage "slow".
                        self.dc.itl2_written = Some(written.max(1));
                    }
                    _ => self.transient.itl4 = Some(count),
                }
            }
            "srcsteps" | "itl6" => {
                self.dc.srcsteps =
                    Some(input.whole_usize(0, crate::analysis::bias::MAX_SOURCE_STEPS)?);
            }
            "gminsteps" => {
                self.dc.gminsteps =
                    Some(input.whole_usize(0, crate::analysis::bias::MAX_GMIN_STAGES)?);
            }
            "gminfactor" => self.dc.gminfactor = Some(input.number("a finite number")?),
            _ => return Ok(false),
        }
        Ok(true)
    }

    /// Apply finite real-valued solver and temperature options.
    fn apply_geometry_option(
        &mut self,
        input: &OptionInput<'_>,
        value: &crate::netlist::ast::PositionedValue,
    ) -> SpiceResult<bool> {
        let name = input.setting.name.as_str();
        match name {
            "scale" => {
                let scale = input.number("a finite positive number")?;
                if scale <= 0. {
                    return Err(SpiceError::parse(
                        value.location.clone(),
                        "geometry scale must be positive",
                    ));
                }
                self.context.geometry_scale = scale;
            }
            "rshunt" | "cshunt" => {
                let number = input.number("a finite number")?;
                let mut shunts = self.context.shunts;
                if name == "rshunt" {
                    shunts.resistance = Some(number);
                } else {
                    shunts.capacitance = number;
                }
                shunts.validate().map_err(|error| {
                    SpiceError::parse(value.location.clone(), error.to_string())
                })?;
                self.context.shunts = shunts;
            }
            _ => return Ok(false),
        }
        Ok(true)
    }

    fn apply_real_option(
        &mut self,
        input: &OptionInput<'_>,
        value: &crate::netlist::ast::PositionedValue,
    ) -> SpiceResult<()> {
        if self.apply_geometry_option(input, value)? {
            return Ok(());
        }
        let setting = input.setting;
        let name = setting.name.as_str();
        let location = &setting.location;
        match name {
            "xmu" => {
                let xmu = input.number("a finite number")?;
                if !(0. ..=0.5).contains(&xmu) {
                    return Err(SpiceError::parse(
                        value.location.clone(),
                        format!("option 'xmu' must be in [0, 0.5], not {xmu}"),
                    ));
                }
                self.transient.xmu = Some(xmu);
            }
            "minbreak" => {
                let number = input.number("a finite number")?;
                if number < 0. {
                    return Err(SpiceError::parse(
                        value.location.clone(),
                        "option 'minbreak' must be nonnegative",
                    ));
                }
                self.transient.minbreak = Some(number);
            }
            "gshunt" => {
                let number = input.number("a finite number")?;
                if number < 0. {
                    return Err(SpiceError::parse(
                        value.location.clone(),
                        "option 'gshunt' must be nonnegative",
                    ));
                }
                self.dc.gshunt = Some(number);
            }
            "gmin" => {
                let gmin = input.number("a finite number")?;
                if gmin < 0. {
                    return Err(SpiceError::parse(
                        value.location.clone(),
                        format!("option 'gmin' must be >= 0, not {gmin}"),
                    ));
                }
                self.context.gmin = gmin;
            }
            "pivtol" | "pivrel" => {
                let mut pivot = self.context.pivot;
                let number = input.number("a finite number")?;
                if name == "pivtol" {
                    pivot.absolute = number;
                } else {
                    pivot.relative = number;
                }
                // spOrderAndFactor would silently keep its default for these.
                pivot
                    .validate()
                    .map_err(|message| SpiceError::parse(value.location.clone(), message))?;
                self.context.pivot = pivot;
            }
            "temp" => {
                self.context.temperature =
                    temperature(input.number("a finite number")?, location, name)?;
            }
            "tnom" => {
                self.context.nominal_temperature =
                    temperature(input.number("a finite number")?, location, name)?;
            }
            _ => {
                let number = input.number("a finite number")?;
                if number <= 0. {
                    return Err(SpiceError::parse(
                        location.clone(),
                        format!("option '{name}' must be positive"),
                    ));
                }
                match name {
                    "reltol" => self.context.reltol = number,
                    "vntol" => self.transient.vntol = Some(number),
                    "abstol" => self.transient.abstol = Some(number),
                    "chgtol" => self.transient.chgtol = Some(number),
                    "lteabstol" => self.transient.voltage_lte.absolute = Some(number),
                    "ltereltol" => self.transient.voltage_lte.relative = Some(number),
                    "ltetrtol" => self.transient.voltage_lte.factor = Some(number),
                    _ => self.transient.trtol = Some(number),
                }
            }
        }
        Ok(())
    }

    /// Resolved circuit temperatures, junction gmin, RELTOL and other
    /// physical settings shared by all analyses.
    #[must_use = "the returned context carries the resolved physical and numerical settings for the analysis"]
    pub const fn context(&self) -> AnalysisContext {
        self.context
    }

    /// Deck-supplied transient tolerances (unset fields keep driver defaults).
    #[must_use = "the returned options determine which transient defaults the deck replaces"]
    pub const fn transient(&self) -> &TransientSettings {
        &self.transient
    }

    /// Deck-supplied DC Newton/continuation settings (unset fields keep the
    /// defaults of [`crate::analysis::bias::DcSettings`]).
    #[must_use = "the returned options determine which Newton and continuation defaults the deck replaces"]
    pub const fn dc(&self) -> &DcOptions {
        &self.dc
    }

    /// Typed integration family, if the deck selected one.
    #[must_use = "the returned option distinguishes an explicit integration family from the default"]
    pub const fn method(&self) -> Option<IntegrationFamily> {
        self.method
    }

    /// Retained `maxord=` selection, if the deck gave one.
    #[must_use = "the returned option distinguishes an explicit order limit from the default"]
    pub fn maxord(&self) -> Option<u8> {
        self.maxord.as_ref().map(|(order, _)| *order)
    }

    /// Accepted options with an effect, in deck order (duplicates included).
    #[must_use = "the returned ordered options identify the effective deck settings including duplicate assignments"]
    pub fn applied(&self) -> &[AppliedOption] {
        &self.applied
    }

    /// Accepted documented no-op options, in deck order, with the reason each
    /// has no effect (see the module documentation).
    #[must_use = "the returned records explain which accepted deck options have no effect"]
    pub fn ignored(&self) -> &[IgnoredOption] {
        &self.ignored
    }

    /// Elaborate `netlist` at this configuration's temperatures.
    ///
    /// # Errors
    /// As [`Circuit::from_netlist_with_context`].
    pub fn circuit(&self, netlist: &Netlist) -> SpiceResult<Circuit> {
        Circuit::from_netlist_with_context(netlist, &self.context.model_context())
    }

    /// Resolve an analysis card with its deck's parameter scope and options.
    ///
    /// # Errors
    /// Invalid expressions, analysis syntax, ranges, or options.
    pub fn plan_for(&self, card: &AnalysisCard) -> SpiceResult<crate::analysis::AnalysisPlan> {
        if card.expressions.is_empty() {
            return self.plan(AnalysisInput::from(card));
        }
        let Some(scope) = &self.params else {
            return Err(SpiceError::Unsupported {
                feature: "braced analysis arguments need a RunConfig built with \\
                          RunConfig::from_netlist (no parameter scope here)"
                    .into(),
                location: Some(card.location.clone()),
            });
        };
        let literal = crate::netlist::elaborate::literalize_analysis(
            card,
            scope,
            &mut crate::netlist::eval::EvalBudget::default(),
        )?;
        self.plan(AnalysisInput::from(&literal))
    }

    /// Resolve all deck cards before executing any of them. Default limits
    /// admit at most 10,000 jobs and 10 million deterministic grid points.
    /// Adaptive transient stepping is governed separately by its own limits.
    ///
    /// # Errors
    /// Invalid cards or aggregate work-budget overruns.
    pub fn plans_for(
        &self,
        cards: &[AnalysisCard],
    ) -> SpiceResult<Vec<crate::analysis::AnalysisPlan>> {
        self.plans_for_with_limits(cards, PlanningLimits::default())
    }

    /// Resolve a deck with independent limits on job count and deterministic
    /// grid work. Checked addition rejects overflow before any execution.
    /// This budgets preparation and intended solves, not streamed rows or
    /// retained plots, which have their own independent resource policies.
    ///
    /// # Errors
    /// As the ordinary plan resolver, plus either aggregate budget overrun.
    pub fn plans_for_with_limits(
        &self,
        cards: &[AnalysisCard],
        limits: PlanningLimits,
    ) -> SpiceResult<Vec<crate::analysis::AnalysisPlan>> {
        if cards.len() > limits.max_jobs {
            return Err(crate::analysis::linear::unsupported(
                "aggregate analysis job limit exceeded",
            ));
        }
        let mut plans = Vec::with_capacity(cards.len());
        let mut requested = 0usize;
        for card in cards {
            let plan = self.plan_for(card)?;
            requested = requested
                .checked_add(plan.requested_grid_points())
                .filter(|count| *count <= limits.max_grid_points)
                .ok_or_else(|| {
                    crate::analysis::linear::unsupported(
                        "aggregate deterministic analysis grid point limit exceeded",
                    )
                })?;
            plans.push(plan);
        }
        Ok(plans)
    }

    /// Resolve explicit analysis input over this configuration's typed deck
    /// defaults. Explicit options and hints win; absent fields inherit deck
    /// settings, followed by the reference defaults.
    ///
    /// # Errors
    /// Invalid syntax, option combinations, ranges, or unsupported analyses.
    pub fn plan(&self, mut input: AnalysisInput) -> SpiceResult<crate::analysis::AnalysisPlan> {
        if input.initial_conditions.is_empty() {
            input
                .initial_conditions
                .clone_from(&self.initial_conditions);
        }
        if input.nodesets.is_empty() {
            input.nodesets.clone_from(&self.nodesets);
        }
        let location = input.location().cloned();
        crate::analysis::AnalysisPlan::resolve(
            input,
            self,
            crate::analysis::NoiseOptions {
                squared: self.squared_noise,
                spice3: self.spice3_noise,
            },
            self.keep_op_info,
        )
        .map_err(|error| match (error, location) {
            (
                SpiceError::Unsupported {
                    feature,
                    location: None,
                },
                location,
            ) => SpiceError::Unsupported { feature, location },
            (SpiceError::Numerical { context, message }, Some(location)) => {
                SpiceError::parse(location, format!("{context}: {message}"))
            }
            (error, _) => error,
        })
    }
}

/// Whether `apply` can accept `name` (with an effect or as a no-op).
fn accepted(name: &str) -> bool {
    PRINT_FLAGS.contains(&name)
        || IGNORED_BY_C.iter().any(|(ignored, _)| *ignored == name)
        || matches!(
            name,
            "nfreqs"
                | "nperiods"
                | "polydegree"
                | "fourgridsize"
                | "post"
                | "ingold"
                | "indverbosity"
                | "bypass"
                | "temp"
                | "tnom"
                | "gmin"
                | "gshunt"
                | "rshunt"
                | "cshunt"
                | "scale"
                | "pivtol"
                | "pivrel"
                | "reltol"
                | "vntol"
                | "abstol"
                | "chgtol"
                | "trtol"
                | "minbreak"
                | "lteabstol"
                | "ltereltol"
                | "ltetrtol"
                | "method"
                | "maxord"
                | "xmu"
                | "itl1"
                | "itl2"
                | "itl4"
                | "srcsteps"
                | "itl6"
                | "gminsteps"
                | "gminfactor"
        )
}

/// C's `IF_INTEGER` option conversion, `floor(0.5 + value)`
/// ([`crate::primitives::c_integer`]), limited to `u32`.
fn integer_value(value: Real) -> Option<u32> {
    crate::primitives::c_integer(value).and_then(|n| u32::try_from(n).ok())
}

fn temperature(value: Real, location: &SourceLoc, what: &str) -> SpiceResult<Real> {
    if value.is_finite() && value > -273.15 {
        Ok(value)
    } else {
        Err(SpiceError::parse(
            location.clone(),
            format!("{what} must be finite and above absolute zero (-273.15 C), got {value}"),
        ))
    }
}
