//! Analysis drivers: `.op`, `.dc`, `.ac`, `.tran`, `.tf` and friends.
//!
//! `.noise` runs [`crate::analysis::noise`] (two plots, see
//! [`crate::analysis::AnalysisPlan::run_plots`]); `.disto` runs [`crate::analysis::disto`] (two
//! harmonic or three intermodulation plots).
//!
//! Ported from `src/spicelib/analysis/`, which is where ngspice's job control
//! lives: `CKTdoJob()` in `cktdojob.c` dispatches on the analysis, `dctran.c`
//! drives DC sweeps and transient analysis, `dcop.c` the operating point,
//! `acan.c` the AC and noise analyses. Around them sit the loading
//! (`CKTload`), iteration (`CKTiter`) and convergence machinery.
//!
//! Drivers support linear R/C/L/V/I equations. Nonlinear analyses remain
//! unsupported; `.tran` runs the trap/Gear companion driver. The split
//! between [`AnalysisInput`] and the netlist AST is deliberate: the driver
//! layer does not need to know where a request came from, and the AST does not
//! need to know which analyses exist.

use crate::netlist::ast::AnalysisCard;
use crate::primitives::{AnalysisKind, Real, SourceLoc, SpiceResult};

/// One located and evaluated initial-condition or nodeset assignment.
///
/// One evaluated `.ic` or `.nodeset` entry, `V(node)=value`, as the analyses
/// receive it: the canonical node name, the finite value in volts and where the
/// entry was written. Duplicates are kept in deck order; the last entry for a
/// node wins (C: `INPpas3()` overwrites the node's `ic`/`nodeset` in turn).
#[derive(Debug, Clone, PartialEq)]
pub struct NodeCondition {
    /// Canonical (lowercased) node name.
    pub node: String,
    /// Value in volts.
    pub value: Real,
    /// Where the `V(node)=value` entry started.
    pub location: SourceLoc,
}

/// Noise representation and model compatibility settings for spectral analysis.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct NoiseOptions {
    /// Front-end `set sqrnoise`: report power densities and integrated powers.
    pub squared: bool,
    /// SPICE3-compatible MOS1 flicker law (`newcompat.s3`).
    pub spice3: bool,
}

/// Literal input to the analysis-plan resolver.
///
/// Arguments are kept as written at the input boundary. Their grammar differs per analysis — `.tran 1u
/// 10u 0 0.1u`, `.ac dec 10 1 1meg`, `.dc v1 0 5 0.1` — the plan resolver interprets them before any circuit load.
#[derive(Debug, Clone, PartialEq)]
pub struct AnalysisInput {
    /// Which analysis to run.
    pub kind: AnalysisKind,
    /// The card's arguments, as written. A `.tran` `uic` flag is **not** among
    /// them; see [`Self::uic`].
    pub arguments: Vec<String>,
    /// The `.tran` `uic` flag (use initial conditions), kept apart from the
    /// positional time arguments. The companion transient driver implements it
    /// (`dctran.c` `MODEUIC`).
    pub uic: bool,
    /// The deck's `.ic` entries in deck order (see [`NodeCondition`]). Enforced
    /// by the companion transient initial operating point unless `uic`; ignored
    /// by `.op`/`.dc`/`.ac`, exactly as C (`cktload.c`), but always validated.
    pub initial_conditions: Vec<NodeCondition>,
    /// The deck's `.nodeset` entries in deck order: convergence hints for the
    /// DC operating points. They cannot change the unique solution of a linear
    /// circuit; under transient `uic` C reuses them as initial node voltages.
    pub nodesets: Vec<NodeCondition>,
    /// Noise-specific representation and compatibility settings.
    pub noise: Option<NoiseOptions>,
    /// Retain the small-signal operating-point plot before spectral output.
    pub keep_op_info: Option<bool>,
    location: Option<SourceLoc>,
}

impl AnalysisInput {
    /// A request with no arguments.
    #[must_use]
    pub const fn new(kind: AnalysisKind) -> Self {
        Self {
            kind,
            arguments: Vec::new(),
            uic: false,
            initial_conditions: Vec::new(),
            nodesets: Vec::new(),
            noise: None,
            keep_op_info: None,
            location: None,
        }
    }

    /// A request with arguments.
    #[must_use]
    pub fn with_arguments(
        kind: AnalysisKind,
        arguments: impl IntoIterator<Item = impl Into<String>>,
    ) -> Self {
        Self {
            arguments: arguments.into_iter().map(Into::into).collect(),
            ..Self::new(kind)
        }
    }

    /// Resolve standalone literal input using the default deck settings.
    ///
    /// # Errors
    /// Invalid analysis syntax, ranges, or options.
    pub fn resolve(self) -> SpiceResult<crate::analysis::AnalysisPlan> {
        crate::analysis::RunConfig::default().plan(self)
    }

    pub(super) const fn location(&self) -> Option<&SourceLoc> {
        self.location.as_ref()
    }
}

impl From<&AnalysisCard> for AnalysisInput {
    fn from(card: &AnalysisCard) -> Self {
        // The tokenizer separates '=' even without surrounding whitespace.
        // Keep positional arguments unchanged and normalize named triples for
        // the driver API. Malformed assignments remain visible for rejection.
        let mut arguments = Vec::new();
        let mut i = 0;
        while i < card.arguments.len() {
            if i + 2 < card.arguments.len() && card.arguments[i + 1] == "=" {
                arguments.push(format!("{}={}", card.arguments[i], card.arguments[i + 2]));
                i += 3;
            } else {
                arguments.push(card.arguments[i].clone());
                i += 1;
            }
        }
        Self {
            arguments,
            uic: card.uic,
            location: Some(card.location.clone()),
            ..Self::new(card.kind)
        }
    }
}

/// Settings shared by every analysis.
///
/// Defaults match ngspice's, which prints `Doing analysis at TEMP = 27.000000
/// and TNOM = 27.000000` at the start of a run.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AnalysisContext {
    /// Circuit temperature in degrees Celsius.
    pub temperature: Real,
    /// Nominal temperature at which model parameters were measured.
    pub nominal_temperature: Real,
    /// Junction minimum conductance in siemens (`.option gmin`, C `CKTgmin`),
    /// default [`crate::devices::DEFAULT_GMIN`]. See
    /// [`crate::devices::ModelContext::gmin`].
    pub gmin: Real,
    /// SPARSE pivot thresholds of the circuit matrix (`.option pivtol`/
    /// `pivrel`, C `CKTpivotAbsTol`/`CKTpivotRelTol`), default
    /// [`crate::maths::PivotTolerances::DEFAULT`].
    pub pivot: crate::maths::PivotTolerances,
}

impl AnalysisContext {
    /// Copy temperature, junction-gmin and pivot settings into the lower-level
    /// device context. Validation occurs at circuit elaboration/equation assembly; no
    /// global state is used.
    #[must_use]
    pub const fn model_context(&self) -> crate::devices::ModelContext {
        crate::devices::ModelContext::new(self.temperature, self.nominal_temperature)
            .with_gmin(self.gmin)
            .with_pivot_tolerances(self.pivot)
    }
}

impl Default for AnalysisContext {
    fn default() -> Self {
        Self {
            temperature: 27.0,
            nominal_temperature: 27.0,
            gmin: crate::devices::DEFAULT_GMIN,
            pivot: crate::maths::PivotTolerances::DEFAULT,
        }
    }
}

/// Analyses with production drivers, for the devices and options documented
/// under `docs/port/`.
pub const DRIVERS: [AnalysisKind; 10] = [
    AnalysisKind::OperatingPoint,
    AnalysisKind::DcSweep,
    AnalysisKind::Ac,
    AnalysisKind::Transient,
    AnalysisKind::PoleZero,
    AnalysisKind::TransferFunction,
    AnalysisKind::SParameter,
    AnalysisKind::Noise,
    AnalysisKind::Distortion,
    AnalysisKind::Sensitivity,
];

/// Whether an analysis has a driver.
#[must_use]
pub fn has_driver(kind: AnalysisKind) -> bool {
    DRIVERS.contains(&kind)
}

/// What the port does with an analysis card (`spice-rs analyses`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AnalysisSupport {
    /// A production driver ([`crate::analysis::AnalysisPlan`]).
    Driver,
    /// Not an analysis of its own: evaluated over the plot of `of`, as
    /// `.four` is over the final `.tran` plot ([`crate::analysis::fourier`]).
    PostProcessor {
        /// The analysis whose plot is post-processed.
        of: AnalysisKind,
    },
    /// No driver: the card is refused.
    Missing,
}

/// The port's support for an analysis kind.
#[must_use]
pub fn support(kind: AnalysisKind) -> AnalysisSupport {
    if has_driver(kind) {
        AnalysisSupport::Driver
    } else if kind == AnalysisKind::Fourier {
        AnalysisSupport::PostProcessor {
            of: AnalysisKind::Transient,
        }
    } else {
        AnalysisSupport::Missing
    }
}
