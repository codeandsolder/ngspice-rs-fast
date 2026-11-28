//! Analysis drivers and result handling.
//!
//! | Module | Job | State |
//! | --- | --- | --- |
//! | [`results`] | [`Plot`], [`Variable`] and the flags that describe them | ported |
//! | [`selection`] | `.save`/`.print` output selection and its text rendering | ported for the bounded request subset in `docs/port/OUTPUT_SELECTION.md` |
//! | [`measure`] | `.measure`/`.meas` evaluation over the full plot and its text rendering | ported for the bounded operation subset in `docs/port/MEASURE.md` |
//! | [`fourier`] | `.four` Fourier/THD evaluation over the final complete period of a transient plot and its text rendering | ported for the bounded card subset in `docs/port/FOURIER.md` |
//! | [`rawfile`] | ngspice rawfile reading **and** writing, ASCII and binary | ported for the bounded layouts in `docs/port/RAWFILES.md` |
//! | [`batch`] | ngspice batch order, plot names and per-plot `.save`/`.print`/`.measure`/`.four` targeting for multi-analysis decks | ported (`docs/port/CLI.md`) |
//! | `pz` | `.pz` poles and zeros at the operating point (`pzan.c`), roots from [`crate::maths::pencil`] | ported (`docs/port/POLE_ZERO_ADR.md`) |
//! | [`noise`] | `.noise`: adjoint small-signal noise spectra and integrated totals | resistor, diode, BJT, MOS1 and switch generators (`docs/port/NOISE.md`) |
//! | [`driver`] | the typed [`AnalysisPlan`] and the `.op`/`.dc`/`.ac`/`.tran`/`.pz`/`.tf`/`.sp`/`.noise` drivers | linear DC/AC and the adaptive trap/Gear-2 companion transient |
//! | `tf` | `.tf` DC small-signal gain, input and output resistance at the bias point | ported (`tfanal.c`; see [`AnalysisPlan`]) |
//! | `sparam` | `.sp` S-parameter analysis over RF port sources (S, Y, Z) | ported, including `donoise` (`docs/port/SPARAM.md`) |
//!
//! The C equivalent is `src/spicelib/analysis/` (21,993 lines: the `CKT*`
//! job-control, loading and iteration machinery) and `src/frontend/rawfile.c`.
//!
//! ASCII rawfile I/O makes the port verifiable.
//!
//! `xtask` captures ASCII rawfiles from C and the test suite parses them, so
//! the format the port must reproduce
//! is pinned by data
//! rather than by prose. `cargo xtask golden verify` runs supported Rust fixtures
//! against that data without C. Binary rawfiles are read and written through the
//! byte-level API in [`rawfile`], whose layout, provenance and rejected variants
//! are documented in `docs/port/RAWFILES.md`. See `docs/port/VERIFICATION.md`,
//! `docs/port/NATIVE_SOLVER.md` and `docs/port/TRANSIENT.md` for current
//! limits. Bounded nonlinear D/Q/M DC/AC and charge-companion support is
//! documented in `docs/port/M4_NONLINEAR.md`; general physics/DAEs remain rejected; `.ic`/`.nodeset`/instance `ic=`/`uic` are implemented by the companion
//! transient driver only (`docs/port/TRANSIENT.md`); the CLI still inspects/parses only.

mod ac;
pub mod batch;
pub mod bias;
mod companion;
pub mod config;
pub mod control;
pub mod cosim;
pub mod digital;
pub mod disto;
pub mod driver;
pub mod fourier;
mod frequency;
pub mod harmonic_balance;
mod initial;
mod linear;
pub mod measure;
pub mod newton;
pub mod noise;
pub mod optimization;
mod ordered_samples;
mod output_probe;
pub mod periodic_noise;
mod periodic_orbit;
pub mod periodic_small_signal;
mod plan;
pub mod prepared_plan;
mod pss;
mod pz;
pub mod rawfile;
mod reduction;
pub mod results;
pub mod rf_network;
pub mod selection;
mod sens;
mod sparam;
pub mod spectrum;
pub mod sweep;
mod tf;
pub mod touchstone;

pub use companion::{
    OutputSampling, TimingSegment, TransientPlan, TransientSchedule, TransientStats,
    companion_transient, companion_transient_into, mixed_signal_transient,
    mixed_signal_transient_into,
};
pub use config::{
    AppliedOption, DcOptions, IgnoredOption, PlanningLimits, RunConfig, RunOverrides,
    TransientSettings,
};
pub use driver::{
    AnalysisContext, AnalysisInput, AnalysisSupport, DRIVERS, NodeCondition, NoiseOptions,
    has_driver, support,
};
pub use fourier::{FourierAnalysis, FourierWindow, Harmonic};
pub use measure::{MeasureEvents, MeasureSpan, Measurement};
pub use periodic_orbit::PeriodicOrbit;
pub use plan::{AnalysisPlan, IntegrationFamily};
pub use rawfile::{
    BinaryByteOrder, RawCollectPolicy, RawFile, RawFileReader, RawFileWriter, RawFormat, RawPlot,
    RawPlotRows, RawReadPolicy,
};
pub use rawfile::{FinishedPlot, RawFileTransaction};
pub use results::{ColumnView, Plot, PlotBuilder, PlotFlags, PlotSchema, PointSink, Variable};
pub use selection::{Selection, print_requests, write_requests};
pub use sparam::RfLimits;

/// The C reference for the analysis drivers.
///
/// Included in `NotYetPorted` errors to identify upstream analysis routines.
pub const C_REFERENCE_ANALYSIS: &str = "src/spicelib/analysis/cktdojob.c, dctran.c, dcop.c, acan.c";

/// Batch line-printer plots.
pub mod asciiplot;

pub mod observations;

mod fourier_resample;
