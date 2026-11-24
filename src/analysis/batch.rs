//! Batch scheduling of every analysis card in a deck, as `ngspice -b -r` does.
//!
//! A deck may request several analyses; ngspice's batch mode runs **all** of
//! them in one job and writes one plot per analysis into a single rawfile. This
//! module owns the parts of that behaviour that are not a driver's job:
//!
//! * **order** ([`schedule`]): C's `CKTdoJob()` (`src/spicelib/analysis/
//!   cktdojob.c`) walks the analysis table `analInfo[]`
//!   (`src/spicelib/analysis/analysis.c`) in its fixed order — options, `.ac`,
//!   `.dc`, `.op`, `.tran`, `.pz`, `.tf`, `.disto`, `.noise`, `.sens`, `.sp` — and for
//!   each type runs every job of that type. Jobs are *prepended* to the task's
//!   job list by `CKTnewAnal()` (`cktnewan.c`), so two cards of the same type run
//!   in **reverse deck order**. Deck order between different types is
//!   irrelevant: `.tran` before `.op` in the deck still runs `.op` first;
//! * **plot names** ([`ScheduledAnalysis::plot_name`]): `plot_add()`/
//!   `plot_alloc()` (`src/frontend/vectors.c`) name an in-memory plot with the
//!   `ft_plotabbrev()` abbreviation (`op`, `dc`, `ac`, `tran`, …) followed by the
//!   global `plot_num`, which starts at 1 and is incremented — and stays
//!   incremented — whenever the candidate name is already taken. A deck with
//!   `.ac`, two `.dc`, `.op` and `.tran` therefore produces `ac1 dc1 dc2 op2
//!   tran2`. A `.noise` card writes two plots, the spectrum and the integrated
//!   noise (`noise1 noise2`), unless its sweep is a single frequency, and a
//!   `.disto` card writes two harmonic or three intermodulation plots
//!   (`disto1 disto2 [disto3]`) ([`ScheduledAnalysis::extra_plot_names`]). The rawfile itself carries only the `Plotname:` header (`AC
//!   Analysis`, …), which each driver already writes;
//! * **which output card applies to which plot** ([`check_targets`],
//!   [`OutputRoutes`]): `.save` applies to every plot; `.print <type>`
//!   narrows and prints only the plots of that type (C's `ft_cktcoms()` prints
//!   one table per matching plot); `.measure <type>` and `.four` are evaluated
//!   against the **last executed** plot of their type, which is the plot C's
//!   `plot_cur`/`setcplot("tran")` select after the run.
//!
//! Divergences from C, all documented in `docs/port/CLI.md`:
//!
//! * C evaluates only the `.measure` cards whose type matches the **last**
//!   analysis that ran (`dosim()` calls `do_measure(ci_last_an)`) and skips the
//!   others silently; this port evaluates every card against the last plot of
//!   its own type instead of dropping it;
//! * a `.print`, `.measure` or `.four` card naming an analysis type the deck
//!   does not run is an error here (C prints `Error: .print: no ac analysis
//!   found.` and carries on, or skips the measurement);
//! * C keeps running the remaining analyses after one fails and writes the
//!   plots that succeeded; the port's callers publish nothing when any analysis
//!   fails.

use crate::netlist::ast::{FourierCard, MeasureCard, OutputCards, VectorRequest};
use crate::primitives::{AnalysisKind, SpiceError, SpiceResult};

use crate::analysis::fourier::{self, FourierAnalysis};
use crate::analysis::measure::Measurement;
use crate::analysis::optimization::objectives::{ActivePlot, ObjectiveRequests, PlotObjectivePlan};
use crate::analysis::results::Plot;
use crate::analysis::selection::{RowProjection, Selection};
use crate::primitives::Complex;
use std::sync::Arc;

/// The C reference for the batch order and plot naming.
pub const C_REFERENCE: &str = "src/spicelib/analysis/cktdojob.c (CKTdoJob), \
                               src/spicelib/analysis/analysis.c (analInfo), \
                               src/frontend/vectors.c (plot_add)";

/// One analysis card in the order a batch run executes it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScheduledAnalysis {
    /// The card's index in the deck's analysis list (`Netlist::analyses`).
    pub card_index: usize,
    /// The analysis type.
    pub kind: AnalysisKind,
    /// The in-memory plot name C gives the result, e.g. `tran1`.
    pub plot_name: String,
    /// True for the last executed plot of this type: the plot `.measure` cards
    /// of this type (and `.four`, for `.tran`) are evaluated against.
    pub last_of_kind: bool,
    /// The names of the further plots this card produces after
    /// [`Self::plot_name`], in order: `.noise` writes its `Integrated Noise`
    /// plot as a second plot (`noise2` after `noise1`) unless its sweep is a
    /// single frequency ([`crate::analysis::AnalysisPlan::plot_count`]); `.disto`
    /// writes its 3rd-harmonic plot after the 2nd, or its `f1-f2` and
    /// `2f1-f2` products after `f1+f2` ([`crate::analysis::AnalysisPlan::plot_count`]).
    /// Empty for every other analysis.
    pub extra_plot_names: Vec<String>,
    retain_bias: bool,
}

impl ScheduledAnalysis {
    /// Every plot name of this card, in rawfile order.
    pub fn plot_names(&self) -> impl Iterator<Item = &str> {
        std::iter::once(self.plot_name.as_str())
            .chain(self.extra_plot_names.iter().map(String::as_str))
    }
}

/// The position of `kind` in C's `analInfo[]` table, which is the order
/// `CKTdoJob()` runs analysis types in. Index 0 is the options pseudo-analysis.
///
/// `.four` is post-processing of a transient plot, not a job, so it sorts after
/// every real analysis; a driver lookup rejects it before anything runs.
#[must_use]
pub const fn job_order(kind: AnalysisKind) -> usize {
    match kind {
        AnalysisKind::Ac => 1,
        AnalysisKind::DcSweep => 2,
        AnalysisKind::OperatingPoint => 3,
        AnalysisKind::Transient => 4,
        AnalysisKind::PoleZero => 5,
        AnalysisKind::TransferFunction => 6,
        AnalysisKind::Distortion => 7,
        AnalysisKind::Noise => 8,
        AnalysisKind::Sensitivity => 9,
        // RFSPICE appends SPinfo after SENSinfo (and the PSS/SENSE2 entries,
        // which have no card here).
        AnalysisKind::PeriodicSteadyState => 10,
        AnalysisKind::SParameter => 11,
        AnalysisKind::HarmonicBalance => 12,
        AnalysisKind::Fourier => 13,
    }
}

/// The plot-type abbreviation `ft_plotabbrev()` (`src/frontend/typesdef.c`)
/// derives from the plot's name for each analysis type.
#[must_use]
pub const fn plot_abbreviation(kind: AnalysisKind) -> &'static str {
    match kind {
        AnalysisKind::OperatingPoint => "op",
        AnalysisKind::DcSweep => "dc",
        AnalysisKind::Ac => "ac",
        AnalysisKind::Transient => "tran",
        AnalysisKind::Noise => "noise",
        AnalysisKind::Distortion => "disto",
        AnalysisKind::PoleZero => "pz",
        AnalysisKind::Sensitivity => "sens",
        AnalysisKind::TransferFunction => "tf",
        AnalysisKind::Fourier => "four",
        // typesdef.c maps the "SP Analysis" plot name to "sp".
        AnalysisKind::SParameter => "sp",
        AnalysisKind::PeriodicSteadyState => "pss",
        AnalysisKind::HarmonicBalance => "hb",
    }
}

/// Resolved analyses in batch execution order, with C's plot names.
///
/// See the module documentation for the rules. An empty list schedules
/// nothing; whether that is an error is the caller's decision.
#[must_use]
pub fn schedule(plans: &[crate::analysis::AnalysisPlan]) -> Vec<ScheduledAnalysis> {
    let mut order: Vec<usize> = (0..plans.len()).collect();
    // Same type: reverse deck order (the job list is built by prepending).
    order.sort_by(|&a, &b| {
        job_order(plans[a].kind())
            .cmp(&job_order(plans[b].kind()))
            .then(b.cmp(&a))
    });
    // Generated names are lowercase; membership need not scan all prior plots.
    let mut names = std::collections::HashSet::with_capacity(order.len());
    let mut plot_num = 1_usize;
    let mut scheduled: Vec<ScheduledAnalysis> = order
        .iter()
        .map(|&card_index| {
            let plan = &plans[card_index];
            let kind = plan.kind();
            let abbreviation = plot_abbreviation(kind);
            let count = plan.plot_count();
            let bias = plan.retains_bias();
            let mut own = Vec::with_capacity(count);
            for index in 0..count {
                let abbreviation = if bias && index == 0 {
                    "op"
                } else {
                    abbreviation
                };
                let mut name = format!("{abbreviation}{plot_num}");
                while names.contains(&name) {
                    plot_num += 1;
                    name = format!("{abbreviation}{plot_num}");
                }
                names.insert(name.clone());
                own.push(name);
            }
            let plot_name = own.remove(0);
            ScheduledAnalysis {
                card_index,
                kind,
                plot_name,
                last_of_kind: false,
                extra_plot_names: own,
                retain_bias: bias,
            }
        })
        .collect();
    for index in 0..scheduled.len() {
        let kind = scheduled[index].kind;
        // Sorting makes all jobs of one kind contiguous.
        scheduled[index].last_of_kind = scheduled
            .get(index + 1)
            .is_none_or(|later| later.kind != kind);
    }
    scheduled
}

/// Checks, before anything runs, that every `.print`, `.measure` and `.four`
/// card targets an analysis type the schedule actually runs.
///
/// # Errors
///
/// [`SpiceError::Unsupported`], positioned at the card, for a `.print` or
/// `.measure` card whose analysis type is not scheduled, and for a `.four` card
/// in a deck that runs no `.tran`: such a card can never be honoured, and the
/// port never drops it silently.
pub fn check_targets(
    schedule: &[ScheduledAnalysis],
    output: &OutputCards,
    measures: &[MeasureCard],
    fourier: &[FourierCard],
) -> SpiceResult<()> {
    let mut counts = [0usize; AnalysisKind::ALL.len()];
    for entry in schedule {
        counts[kind_index(entry.kind)] += 1;
    }
    let runs = |kind| counts[kind_index(kind)] != 0;
    // `.sens` perturbs C's device structures in place and leaves them
    // perturbed-and-restored (given flags, setup-only values), and its name
    // filter is a global of the last `.sens` card parsed (`Sens_filter`,
    // `inp2dot.c`): a second `.sens`, or the `.sp` that C runs after it, would
    // see that mutated circuit. The port refuses such decks.
    if counts[kind_index(AnalysisKind::Sensitivity)] > 1 {
        return Err(SpiceError::Unsupported {
            feature: "more than one .sens card in a deck (C shares one name filter between them \
                      and runs the later ones on the circuit the first left perturbed; \
                      docs/port/SENSITIVITY.md)"
                .to_owned(),
            location: None,
        });
    }
    if runs(AnalysisKind::Sensitivity) && runs(AnalysisKind::SParameter) {
        return Err(SpiceError::Unsupported {
            feature: ".sp after .sens in one deck (C runs .sp on the device structures .sens \
                      left perturbed and restored; docs/port/SENSITIVITY.md)"
                .to_owned(),
            location: None,
        });
    }
    let scheduled = || {
        let kinds: Vec<String> = schedule
            .iter()
            .map(|entry| format!(".{}", entry.kind.as_str()))
            .collect();
        if kinds.is_empty() {
            "no analysis".to_owned()
        } else {
            kinds.join(" ")
        }
    };
    for print in &output.prints {
        if !runs(print.analysis) {
            return Err(SpiceError::Unsupported {
                feature: format!(
                    ".print {} names a different analysis than the deck runs ({}), so the \
                     request can never be honoured",
                    print.analysis.as_str(),
                    scheduled()
                ),
                location: Some(print.analysis_location.clone()),
            });
        }
    }
    for card in measures {
        if !runs(card.analysis) {
            return Err(SpiceError::Unsupported {
                feature: format!(
                    ".measure {} {}: the card names a .{} measurement; the deck runs {}, so \
                     the card can never be honoured",
                    card.analysis.as_str(),
                    card.name,
                    card.analysis.as_str(),
                    scheduled()
                ),
                location: Some(card.location.clone()),
            });
        }
    }
    if let Some(card) = fourier.first()
        && !runs(AnalysisKind::Transient)
    {
        return Err(SpiceError::Unsupported {
            feature: format!(
                ".four: the card transforms a .tran result (C selects the tran plot); the deck \
                 runs {}, so the card can never be honoured",
                scheduled()
            ),
            location: Some(card.location.clone()),
        });
    }
    Ok(())
}

/// Identity and borrowed consumers of one scheduled plot, independent of its
/// solver and storage representation.
pub struct PlannedOutput<'a> {
    /// Batch-allocated name in C plot order.
    pub name: &'a str,
    /// Analysis that produced this plot (including its optional bias).
    pub kind: AnalysisKind,
    selection_kind: AnalysisKind,
    routed: &'a RoutedOutputs<'a>,
    measurements: &'a [&'a MeasureCard],
    fourier: &'a [FourierCard],
}

impl PlannedOutput<'_> {
    /// Bind the rawfile output selection to the driver's schema.
    ///
    /// # Errors
    /// An unknown vector or an invalid component request.
    pub fn selection(&self, header: &Plot) -> SpiceResult<Selection> {
        Selection::resolve_refs(header, self.selection_kind, self.routed.write_requests())
    }

    pub(crate) fn bind_consumers(&self, header: &Plot) -> SpiceResult<BoundConsumers> {
        let kind = self.selection_kind;
        let mut requests = self.routed.print_requests().peekable();
        let table = if requests.peek().is_some() {
            Some(Selection::resolve_refs(header, kind, requests)?)
        } else {
            None
        };
        let graphs = self
            .routed
            .prints
            .iter()
            .filter(|card| card.ascii_plot && !card.requests.is_empty())
            .map(|card| {
                let selection =
                    Selection::resolve_refs(header, kind, card.requests.iter().copied())?;
                Ok(AsciiHistory {
                    plot: selection.header(header)?,
                    projection: selection.bind(header)?,
                })
            })
            .collect::<SpiceResult<Vec<_>>>()?;
        let objectives = if self.measurements.is_empty() && self.fourier.is_empty() {
            None
        } else {
            let plan = PlotObjectivePlan::bind(
                header,
                kind,
                ObjectiveRequests {
                    measurements: self
                        .measurements
                        .iter()
                        .map(|card| (*card).clone())
                        .collect(),
                    observations: Vec::new(),
                    fourier: self.fourier.to_vec(),
                    retained_cell_limit: MAX_REPORT_HISTORY_CELLS,
                },
            )?;
            let state = ActivePlot::new(&plan)?;
            Some((plan, state))
        };
        let retained_width = graphs
            .iter()
            .map(|graph| graph.plot.variables().len())
            .chain(
                objectives
                    .iter()
                    .map(|(plan, _)| plan.retained_columns().len()),
            )
            .try_fold(0usize, |total, width| {
                total.checked_add(width).ok_or_else(report_history_limit)
            })?;
        Ok(BoundConsumers {
            table,
            graphs,
            objectives,
            retained_width,
            rows: 0,
            row_text: String::new(),
        })
    }
}

const fn kind_index(kind: AnalysisKind) -> usize {
    job_order(kind) - 1
}

/// One deck-wide directive index. Same-kind jobs borrow the same routed lists
/// rather than repeating setup proportional to the number of directives.
pub struct OutputRoutes<'a> {
    regular: [RoutedOutputs<'a>; AnalysisKind::ALL.len()],
    bias: RoutedOutputs<'a>,
    noise_integrated: RoutedOutputs<'a>,
    measurements: [Vec<&'a MeasureCard>; AnalysisKind::ALL.len()],
    fourier: &'a [FourierCard],
}

impl<'a> OutputRoutes<'a> {
    /// Classify directives once in deck order and validate singleton noise.
    ///
    /// # Errors
    /// Unsupported targets and integrated consumers on single-frequency noise.
    pub fn new(
        schedule: &[ScheduledAnalysis],
        output: &'a OutputCards,
        measures: &'a [MeasureCard],
        fourier: &'a [FourierCard],
    ) -> SpiceResult<Self> {
        check_targets(schedule, output, measures, fourier)?;
        let saves: Arc<[_]> = output
            .saves
            .iter()
            .flat_map(|card| &card.requests)
            .collect();
        let mut regular = std::array::from_fn(|_| RoutedOutputs {
            saves: Arc::clone(&saves),
            prints: Vec::new(),
        });
        for card in &output.prints {
            regular[kind_index(card.analysis)].prints.push(RoutedPrint {
                ascii_plot: card.ascii_plot,
                requests: card.requests.iter().collect(),
            });
        }
        let mut measurements = std::array::from_fn(|_| Vec::new());
        for card in measures {
            measurements[kind_index(card.analysis)].push(card);
        }
        let bias = RoutedOutputs {
            saves,
            prints: Vec::new(),
        };
        let mut noise_integrated = RoutedOutputs {
            saves: Arc::from([]),
            prints: Vec::new(),
        };
        let mut noise = schedule
            .iter()
            .filter(|entry| entry.kind == AnalysisKind::Noise);
        if let Some(first) = noise.next() {
            let singleton = |entry: &ScheduledAnalysis| {
                entry.plot_names().count() - usize::from(entry.retain_bias) == 1
            };
            let has_singleton = singleton(first) || noise.any(singleton);
            let spectrum = &mut regular[kind_index(AnalysisKind::Noise)];
            let (saves, totals) = split_noise(spectrum.saves.iter().copied(), has_singleton)?;
            spectrum.saves = saves.into();
            noise_integrated.saves = totals.into();
            for card in &mut spectrum.prints {
                let (requests, totals) = split_noise(card.requests.iter().copied(), has_singleton)?;
                card.requests = requests;
                noise_integrated.prints.push(RoutedPrint {
                    ascii_plot: card.ascii_plot,
                    requests: totals,
                });
            }
        }
        Ok(Self {
            regular,
            bias,
            noise_integrated,
            measurements,
            fourier,
        })
    }

    /// Bias plots take only saves; noise totals route to the integrated result;
    /// measurement/Fourier selection preserves the last-of-kind policy.
    #[must_use]
    pub fn plan_outputs<'b>(&'b self, scheduled: &'b ScheduledAnalysis) -> Vec<PlannedOutput<'b>> {
        let kind = scheduled.kind;
        let has_integrated = kind == AnalysisKind::Noise
            && scheduled.plot_names().count() - usize::from(scheduled.retain_bias) > 1;
        scheduled
            .plot_names()
            .enumerate()
            .map(|(index, name)| {
                let bias = scheduled.retain_bias && index == 0;
                let integrated = has_integrated && index > usize::from(scheduled.retain_bias);
                let routed = if bias {
                    &self.bias
                } else if integrated {
                    &self.noise_integrated
                } else {
                    &self.regular[kind_index(kind)]
                };
                let measurements = if !bias && scheduled.last_of_kind {
                    &self.measurements[kind_index(kind)][..]
                } else {
                    &[]
                };
                let fourier = if !bias && scheduled.last_of_kind && kind == AnalysisKind::Transient
                {
                    self.fourier
                } else {
                    &[]
                };
                PlannedOutput {
                    name,
                    kind,
                    selection_kind: if bias {
                        AnalysisKind::OperatingPoint
                    } else {
                        kind
                    },
                    routed,
                    measurements,
                    fourier,
                }
            })
            .collect()
    }
}

struct RoutedPrint<'a> {
    ascii_plot: bool,
    requests: Vec<&'a VectorRequest>,
}

struct RoutedOutputs<'a> {
    saves: Arc<[&'a VectorRequest]>,
    prints: Vec<RoutedPrint<'a>>,
}

impl<'a> RoutedOutputs<'a> {
    fn write_requests(&self) -> impl Iterator<Item = &'a VectorRequest> + '_ {
        self.saves.iter().copied().chain(
            self.prints
                .iter()
                .flat_map(|card| card.requests.iter().copied()),
        )
    }

    fn print_requests(&self) -> impl Iterator<Item = &'a VectorRequest> + '_ {
        self.prints
            .iter()
            .filter(|card| !card.ascii_plot)
            .flat_map(|card| card.requests.iter().copied())
    }
}

fn split_noise<'a>(
    requests: impl Iterator<Item = &'a VectorRequest>,
    has_singleton: bool,
) -> SpiceResult<(Vec<&'a VectorRequest>, Vec<&'a VectorRequest>)> {
    let mut spectrum = Vec::new();
    let mut integrated = Vec::new();
    for request in requests {
        if matches!(request.vector, crate::netlist::ast::RequestedVector::All) {
            spectrum.push(request);
            integrated.push(request);
            continue;
        }
        let name = request.vector.name().to_ascii_lowercase();
        if name.contains("onoise_total") || name.contains("inoise_total") {
            if has_singleton {
                return Err(SpiceError::Unsupported {
                    feature: format!("{name}: single-frequency .noise has no integrated plot"),
                    location: Some(request.location.clone()),
                });
            }
            integrated.push(request);
        } else {
            spectrum.push(request);
        }
    }
    Ok((spectrum, integrated))
}

// Independent of output row count for online reductions and text tables.
// Deferred/Fourier/ASCII history has a 256 MiB logical cell budget per plot
// (16 Mi complex cells); exceeding it aborts the output transaction.
const MAX_REPORT_HISTORY_CELLS: usize = 16 * 1024 * 1024;

fn report_history_limit() -> SpiceError {
    SpiceError::Unsupported {
        feature: format!("report retained-cell limit of {MAX_REPORT_HISTORY_CELLS} exceeded"),
        location: None,
    }
}

struct AsciiHistory {
    plot: Plot,
    projection: RowProjection,
}

/// Bound downstream consumers; no solver or rawfile-specific state.
pub(crate) struct BoundConsumers {
    table: Option<Selection>,
    graphs: Vec<AsciiHistory>,
    objectives: Option<(PlotObjectivePlan, ActivePlot)>,
    retained_width: usize,
    rows: usize,
    row_text: String,
}

pub(crate) struct ObjectiveReports {
    pub(crate) measurements: Vec<Measurement>,
    pub(crate) fourier: Vec<FourierAnalysis>,
}

impl BoundConsumers {
    pub(crate) const fn has_text(&self) -> bool {
        self.table.is_some() || !self.graphs.is_empty()
    }

    pub(crate) const fn is_empty(&self) -> bool {
        !self.has_text() && self.objectives.is_none()
    }

    pub(crate) fn begin_text(
        &self,
        mut emit: impl FnMut(&str) -> SpiceResult<()>,
    ) -> SpiceResult<()> {
        if let Some(table) = &self.table {
            emit(&table.text_header())?;
        }
        Ok(())
    }

    pub(crate) fn point(
        &mut self,
        row: &[Complex],
        mut emit: impl FnMut(&str) -> SpiceResult<()>,
    ) -> SpiceResult<()> {
        let next = self.rows.checked_add(1).ok_or_else(report_history_limit)?;
        if next
            .checked_mul(self.retained_width)
            .is_none_or(|cells| cells > MAX_REPORT_HISTORY_CELLS)
        {
            return Err(report_history_limit());
        }
        if let Some(table) = &self.table {
            table.text_row(self.rows, row, &mut self.row_text)?;
            emit(&self.row_text)?;
        }
        for graph in &mut self.graphs {
            graph.plot.push_point(graph.projection.project(row)?)?;
        }
        if let Some((plan, state)) = &mut self.objectives {
            state.point(plan, row)?;
        }
        self.rows = next;
        Ok(())
    }

    pub(crate) fn finish(
        self,
        scope: &crate::netlist::eval::ParamScope,
        settings: Option<fourier::FourierSettings>,
        mut emit: impl FnMut(&str) -> SpiceResult<()>,
    ) -> SpiceResult<ObjectiveReports> {
        for graph in &self.graphs {
            super::asciiplot::render(&graph.plot, &mut emit)?;
        }
        let (measurements, fourier) = if let Some((plan, state)) = self.objectives {
            let result = state.finish(&plan, scope, settings);
            (result.measurements?, result.fourier?)
        } else {
            (Vec::new(), Vec::new())
        };
        Ok(ObjectiveReports {
            measurements,
            fourier,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::{job_order, schedule};
    use crate::netlist::ast::AnalysisCard;
    use crate::primitives::{AnalysisKind, SourceLoc};

    fn card(kind: AnalysisKind, line: u32) -> AnalysisCard {
        AnalysisCard {
            kind,
            arguments: match kind {
                AnalysisKind::Ac => "lin 3 1 3",
                AnalysisKind::DcSweep => "v1 0 1 0.1",
                AnalysisKind::Transient => "1u 10u",
                _ => "",
            }
            .split_whitespace()
            .map(str::to_owned)
            .collect(),
            expressions: Vec::new(),
            uic: false,
            uic_location: None,
            location: SourceLoc::new(std::path::PathBuf::from("deck.cir"), line, 1),
        }
    }

    fn plans(cards: &[AnalysisCard]) -> Vec<crate::analysis::AnalysisPlan> {
        crate::analysis::RunConfig::default()
            .plans_for(cards)
            .unwrap()
    }

    fn names(cards: &[AnalysisCard]) -> Vec<(usize, String, bool)> {
        schedule(&plans(cards))
            .into_iter()
            .map(|entry| (entry.card_index, entry.plot_name, entry.last_of_kind))
            .collect()
    }

    #[test]
    fn types_run_in_analinfo_order_whatever_the_deck_order() {
        use AnalysisKind::{Ac, DcSweep, OperatingPoint, Transient};
        let cards = [
            card(Transient, 2),
            card(OperatingPoint, 3),
            card(Ac, 4),
            card(DcSweep, 5),
        ];
        assert_eq!(
            names(&cards),
            [
                (2, "ac1".to_owned(), true),
                (3, "dc1".to_owned(), true),
                (1, "op1".to_owned(), true),
                (0, "tran1".to_owned(), true),
            ]
        );
    }

    #[test]
    fn same_type_cards_run_in_reverse_deck_order_and_bump_the_plot_number() {
        // Measured with ngspice-47: `.tran`, `.ac`, `.dc a`, `.op`, `.dc b`
        // lists `ac1 dc1 dc2 op2 tran2`, and `dc1` is the `.dc b` sweep.
        use AnalysisKind::{Ac, DcSweep, OperatingPoint, Transient};
        let cards = [
            card(Transient, 2),
            card(Ac, 3),
            card(DcSweep, 4),
            card(OperatingPoint, 5),
            card(DcSweep, 6),
        ];
        assert_eq!(
            names(&cards),
            [
                (1, "ac1".to_owned(), true),
                (4, "dc1".to_owned(), false),
                (2, "dc2".to_owned(), true),
                (3, "op2".to_owned(), true),
                (0, "tran2".to_owned(), true),
            ]
        );
    }

    #[test]
    fn a_single_card_and_an_empty_deck() {
        assert_eq!(
            names(&[card(AnalysisKind::OperatingPoint, 2)]),
            [(0, "op1".to_owned(), true)]
        );
        assert!(schedule(&[]).is_empty());
    }

    #[test]
    fn noise_cards_name_their_integrated_plot_too() {
        // `.noise` (decade), `.op`, `.noise` (one linear point): the later
        // card runs first and writes its spectrum only, so the plots are
        // `op1 noise1 noise2 noise3` (the naming the `noise_multi` capture
        // confirms against ngspice-47).
        let noise = |line, arguments: &str| {
            let mut card = card(AnalysisKind::Noise, line);
            card.arguments = arguments.split_whitespace().map(str::to_owned).collect();
            card
        };
        let cards = [
            noise(2, "v ( out ) v1 dec 3 100 1meg"),
            card(AnalysisKind::OperatingPoint, 3),
            noise(4, "v ( out ) v1 lin 1 10k 20k 1"),
        ];
        let entries = schedule(&plans(&cards));
        let names: Vec<Vec<&str>> = entries
            .iter()
            .map(|entry| entry.plot_names().collect())
            .collect();
        assert_eq!(
            names,
            [vec!["op1"], vec!["noise1"], vec!["noise2", "noise3"]]
        );
        assert!(!entries[1].last_of_kind && entries[2].last_of_kind);
    }

    #[test]
    fn disto_cards_name_every_plot() {
        // `.disto` (harmonics), `.disto` (IM), `.ac`: ngspice-47 runs `ac1`,
        // then the later IM card (`disto1..3`), then the harmonics
        // (`disto4 disto5`), as its `setplot` listing shows.
        let disto = |line, arguments: &str| {
            let mut card = card(AnalysisKind::Distortion, line);
            card.arguments = arguments.split_whitespace().map(str::to_owned).collect();
            card
        };
        let cards = [
            disto(2, "dec 2 1k 100k"),
            disto(3, "dec 2 1k 100k 0.9"),
            card(AnalysisKind::Ac, 4),
        ];
        let names: Vec<Vec<String>> = schedule(&plans(&cards))
            .iter()
            .map(|entry| entry.plot_names().map(str::to_owned).collect())
            .collect();
        assert_eq!(
            names,
            [
                vec!["ac1"],
                vec!["disto1", "disto2", "disto3"],
                vec!["disto4", "disto5"]
            ]
        );
    }

    #[test]
    fn consumers_are_planned_per_output_identity() {
        use crate::netlist::{Parser, source::parse_deck_text};
        use std::path::Path;
        let parsed = Parser::new().parse_deck_with_output(&parse_deck_text(
            Path::new("outputs.cir"),
            "outputs\nv1 in 0 dc 1 ac 1\nr1 in out 1k\nr2 out 0 1k\n.options keepopinfo\n.ac dec 3 1 100\n.tran 1u 10u\n.tran 1u 20u\n.noise v(out) v1 lin 4 1 4\n.print ac all\n.print noise onoise_spectrum\n.measure tran top max v(out)\n.end\n",
        )).unwrap();
        let config = crate::analysis::RunConfig::from_netlist(&parsed.netlist).unwrap();
        let schedule = schedule(&config.plans_for(&parsed.netlist.analyses).unwrap());
        let routes = super::OutputRoutes::new(
            &schedule,
            &parsed.output,
            &parsed.measurements,
            &parsed.fourier,
        )
        .unwrap();
        let outputs: Vec<_> = schedule
            .iter()
            .flat_map(|entry| routes.plan_outputs(entry))
            .collect();
        assert_eq!(
            outputs
                .iter()
                .map(|plan| plan
                    .routed
                    .prints
                    .iter()
                    .any(|print| !print.requests.is_empty())
                    || !plan.measurements.is_empty()
                    || !plan.fourier.is_empty())
                .collect::<Vec<_>>(),
            [false, true, false, true, false, true, false]
        );
        assert_eq!(outputs[0].selection_kind, AnalysisKind::OperatingPoint);
        assert_eq!(outputs[4].selection_kind, AnalysisKind::OperatingPoint);
        assert_eq!(outputs[5].kind, AnalysisKind::Noise);
        assert_eq!(outputs[5].routed.print_requests().count(), 1);
        assert_eq!(outputs[6].routed.print_requests().count(), 0);
    }

    #[test]
    fn full_retained_selection_moves_the_original_rows_after_printing() {
        use crate::analysis::{Plot, PlotFlags, Variable};
        use crate::netlist::{Parser, source::parse_deck_text};
        use crate::primitives::Complex;
        use std::path::Path;
        let parsed = Parser::new()
            .parse_deck_with_output(&parse_deck_text(
                Path::new("retained.cir"),
                "retained\nv1 in 0 1\nr1 in 0 1k\n.op\n.print op all\n.end\n",
            ))
            .unwrap();
        let config = crate::analysis::RunConfig::from_netlist(&parsed.netlist).unwrap();
        let schedule = schedule(&config.plans_for(&parsed.netlist.analyses).unwrap());
        let routes = super::OutputRoutes::new(&schedule, &parsed.output, &[], &[]).unwrap();
        let plans = routes.plan_outputs(&schedule[0]);
        let mut plot = Plot::new(
            "op1",
            "Operating Point",
            PlotFlags::Real,
            vec![Variable::new("v(in)", "voltage")],
        );
        let data = vec![Complex::real(1.)];
        plot.push_point(&data).unwrap();
        let row = plot.cells().as_ptr();
        let scope = crate::netlist::eval::ParamScope::root(&[]).unwrap();
        let mut consumers = plans[0].bind_consumers(&plot.header()).unwrap();
        let mut text = String::new();
        let mut emit = |line: &str| {
            text.push_str(line);
            Ok(())
        };
        consumers.begin_text(&mut emit).unwrap();
        for point in plot.rows() {
            consumers.point(point, &mut emit).unwrap();
        }
        consumers.finish(&scope, None, &mut emit).unwrap();
        assert!(text.contains("v(in)"));
        let selected = plans[0].selection(&plot).unwrap().into_plot(plot).unwrap();
        assert_eq!(selected.row(0).unwrap().as_ptr(), row);
    }

    #[test]
    fn many_jobs_share_directive_allocations() {
        use crate::netlist::{Parser, source::parse_deck_text};
        use std::path::Path;
        let mut deck = String::from("many jobs\nv1 in 0 1 ac 1\nr1 in 0 1k\n.save v(in)\n");
        for _ in 0..100 {
            deck.push_str(".tran 1u 10u\n.ac lin 2 1 2\n");
        }
        for index in 0..1000 {
            use std::fmt::Write as _;
            writeln!(
                deck,
                ".print tran v(in)\n.print ac v(in)\n.measure tran m{index} max v(in)"
            )
            .unwrap();
        }
        deck.push_str(".end\n");
        let parsed = Parser::new()
            .parse_deck_with_output(&parse_deck_text(Path::new("many.cir"), &deck))
            .unwrap();
        let config = crate::analysis::RunConfig::from_netlist(&parsed.netlist).unwrap();
        let scheduled = schedule(&config.plans_for(&parsed.netlist.analyses).unwrap());
        let routes =
            super::OutputRoutes::new(&scheduled, &parsed.output, &parsed.measurements, &[])
                .unwrap();
        let outputs: Vec<_> = scheduled
            .iter()
            .flat_map(|entry| routes.plan_outputs(entry))
            .collect();
        let allocations: std::collections::HashSet<_> = outputs
            .iter()
            .map(|output| std::ptr::from_ref(output.routed))
            .collect();
        assert_eq!(outputs.len(), 200);
        assert_eq!(allocations.len(), 2);
        assert_eq!(
            routes
                .regular
                .iter()
                .map(|route| route.prints.len())
                .sum::<usize>(),
            2000
        );
        assert_eq!(
            routes.measurements.iter().map(Vec::len).sum::<usize>(),
            1000
        );
        assert!(std::sync::Arc::ptr_eq(
            &outputs[0].routed.saves,
            &outputs[199].routed.saves
        ));
        assert_eq!(
            outputs
                .iter()
                .map(|output| output.measurements.len())
                .sum::<usize>(),
            1000
        );
        assert_eq!(outputs[199].measurements.len(), 1000);
    }

    #[test]
    fn wide_online_consumer_retains_no_history_and_prints_retained_bytes() {
        use crate::analysis::{Plot, PlotFlags, Variable};
        use crate::netlist::{Parser, source::parse_deck_text};
        use crate::primitives::Complex;
        use std::path::Path;
        let parsed = Parser::new().parse_deck_with_output(&parse_deck_text(
            Path::new("wide.cir"),
            "wide\nv1 in 0 1\nr1 in 0 1k\n.tran 1u 10u\n.print tran v(in)\n.measure tran top max v(hidden)\n.end\n",
        )).unwrap();
        let config = crate::analysis::RunConfig::from_netlist(&parsed.netlist).unwrap();
        let scheduled = schedule(&config.plans_for(&parsed.netlist.analyses).unwrap());
        let routes =
            super::OutputRoutes::new(&scheduled, &parsed.output, &parsed.measurements, &[])
                .unwrap();
        let planned = routes.plan_outputs(&scheduled[0]);
        let mut variables = vec![
            Variable::new("time", "time"),
            Variable::new("v(in)", "voltage"),
            Variable::new("v(hidden)", "voltage"),
        ];
        variables.extend((0..64).map(|index| Variable::new(format!("v(n{index})"), "voltage")));
        let mut plot = Plot::new("tran1", "Transient Analysis", PlotFlags::Real, variables);
        let mut consumers = planned[0].bind_consumers(&plot.header()).unwrap();
        assert_eq!(consumers.retained_width, 0);
        let mut text = String::new();
        let mut emit = |line: &str| {
            text.push_str(line);
            Ok(())
        };
        consumers.begin_text(&mut emit).unwrap();
        let mut row = vec![Complex::real(0.); 67];
        for index in 0..1000 {
            row[0] = Complex::real(f64::from(index));
            row[1] = Complex::real(2.);
            row[2] = Complex::real(f64::from(index));
            consumers.point(&row, &mut emit).unwrap();
            plot.push_point(&row).unwrap();
        }
        let scope = crate::netlist::eval::ParamScope::root(&[]).unwrap();
        let reports = consumers.finish(&scope, None, &mut emit).unwrap();
        assert_eq!(reports.measurements[0].value, 999.);
        assert_eq!(
            text,
            planned[0].selection(&plot).unwrap().to_text(&plot).unwrap()
        );
    }

    #[test]
    fn singleton_noise_rejects_integrated_consumers_before_rows() {
        use crate::netlist::{Parser, source::parse_deck_text};
        use std::path::Path;
        let parsed = Parser::new().parse_deck_with_output(&parse_deck_text(
            Path::new("single.cir"),
            "singleton\nv1 in 0 ac 1\nr1 in out 1k\nr2 out 0 1k\n.options keepopinfo\n.noise v(out) v1 lin 1 1 1\n.print noise v(onoise_total)\n.end\n",
        )).unwrap();
        let config = crate::analysis::RunConfig::from_netlist(&parsed.netlist).unwrap();
        let schedule = schedule(&config.plans_for(&parsed.netlist.analyses).unwrap());
        let Err(error) = super::OutputRoutes::new(
            &schedule,
            &parsed.output,
            &parsed.measurements,
            &parsed.fourier,
        ) else {
            panic!("accepted integrated noise consumer for a singleton")
        };
        assert!(error.to_string().contains("single-frequency"), "{error}");
    }

    #[test]
    fn the_job_order_is_c_analinfo() {
        let mut kinds = AnalysisKind::ALL.to_vec();
        kinds.sort_by_key(|kind| job_order(*kind));
        assert_eq!(
            kinds,
            [
                AnalysisKind::Ac,
                AnalysisKind::DcSweep,
                AnalysisKind::OperatingPoint,
                AnalysisKind::Transient,
                AnalysisKind::PoleZero,
                AnalysisKind::TransferFunction,
                AnalysisKind::Distortion,
                AnalysisKind::Noise,
                AnalysisKind::Sensitivity,
                AnalysisKind::PeriodicSteadyState,
                AnalysisKind::SParameter,
                AnalysisKind::HarmonicBalance,
                AnalysisKind::Fourier,
            ]
        );
    }
}
