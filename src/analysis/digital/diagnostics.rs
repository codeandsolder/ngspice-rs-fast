//! Accepted diagnostic history with cheap disposable event snapshots.
use crate::devices::code_model::{Drive, ModelDiagnostic};
use crate::primitives::Real;
use std::rc::Rc;
use std::sync::Arc;

/// Typed diagnostic produced by a native digital or mixed-domain model.
#[derive(Clone, Debug, PartialEq)]
pub enum DiagnosticDetail {
    /// A sequential timing constraint.
    Timing(ModelDiagnostic),
    /// Two successive drive arrivals have overlapping absolute windows.
    /// The relative interval factors out common upstream uncertainty.
    CumulativeAmbiguity {
        /// Earlier posted transition's absolute arrival bounds.
        earlier: super::timing::Bounds,
        /// Later posted transition's absolute arrival bounds.
        later: super::timing::Bounds,
        /// Later minus earlier, after causal cancellation.
        separation: super::timing::Bounds,
    },
    /// A voltage receiver encountered an uncovered region and applied
    /// its declared fallback policy.
    RangeGap {
        /// Whether the declared policy rejects this candidate.
        fatal: bool,
        /// Converged differential input voltage.
        voltage: Real,
        /// Selected output drive.
        drive: Drive,
        /// Cyclic scan index retained or selected by the fallback.
        range_index: usize,
    },
}

/// Model warning at a resolved event, retained only on transaction commit.
#[derive(Clone, Debug, PartialEq)]
pub struct EventDiagnostic {
    /// Flattened instance name, or a generated model ordinal for a model
    /// constructed directly through the event network API.
    pub model: Arc<str>,
    /// Physical time in seconds.
    pub time: Real,
    /// Delta-cycle ordinal at the physical time.
    pub delta: usize,
    /// Model-specific diagnostic data.
    pub detail: DiagnosticDetail,
}

impl EventDiagnostic {
    /// Access the sequential timing detail, when this is a timing diagnostic.
    #[must_use = "the returned detail describes the observed timing constraint"]
    pub const fn timing(&self) -> Option<&ModelDiagnostic> {
        match &self.detail {
            DiagnosticDetail::Timing(timing) => Some(timing),
            DiagnosticDetail::RangeGap { .. } | DiagnosticDetail::CumulativeAmbiguity { .. } => {
                None
            }
        }
    }

    pub(super) fn is_fatal(&self) -> bool {
        match &self.detail {
            DiagnosticDetail::Timing(timing) => timing.fatal,
            DiagnosticDetail::RangeGap { fatal, .. } => *fatal,
            DiagnosticDetail::CumulativeAmbiguity { .. } => false,
        }
    }
}

impl std::fmt::Display for EventDiagnostic {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.detail {
            DiagnosticDetail::Timing(detail) => {
                let possible = if detail.certainty == super::timing::Certainty::PossibleViolation {
                    "possible "
                } else {
                    ""
                };
                write!(
                    f,
                    "{}: {possible}{} violation at t={:e} (delta {}): input {} interval ",
                    self.model,
                    detail.kind,
                    self.time,
                    self.delta,
                    detail.input + 1
                )?;
                if detail.elapsed == detail.maximum_elapsed {
                    write!(f, "{:e}", detail.elapsed)?;
                } else {
                    write!(f, "[{:e},{:e}]", detail.elapsed, detail.maximum_elapsed)?;
                }
                write!(f, " < required {:e}", detail.minimum)
            }
            DiagnosticDetail::CumulativeAmbiguity {
                earlier,
                later,
                separation,
            } => write!(
                f,
                "{}: cumulative timing ambiguity at t={:e}: [{:e},{:e}] and [{:e},{:e}], relative separation [{:e},{:e}]",
                self.model,
                self.time,
                earlier.minimum(),
                earlier.maximum(),
                later.minimum(),
                later.maximum(),
                separation.minimum(),
                separation.maximum()
            ),
            DiagnosticDetail::RangeGap {
                fatal,
                voltage,
                drive,
                range_index,
            } => {
                write!(
                    f,
                    "{}: receiver voltage {:e} V outside all ranges at t={:e}",
                    self.model, voltage, self.time
                )?;
                if *fatal {
                    write!(f, "; rejected by gap policy")
                } else {
                    write!(f, "; fallback {:?}, scan index {}", drive, range_index)
                }
            }
        }
    }
}

const CHUNK: usize = 64;

#[derive(Clone, Debug)]
struct Chunk {
    previous: Option<Rc<Self>>,
    records: Vec<EventDiagnostic>,
}

/// Previewing and rejecting a model trial must not print or clone the full
/// warning history. Share complete chunks; copy only a partial tail on write.
#[derive(Clone, Default)]
pub(super) struct DiagnosticLog {
    tail: Option<Rc<Chunk>>,
    len: usize,
}

impl std::fmt::Debug for DiagnosticLog {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DiagnosticLog")
            .field("len", &self.len)
            .finish()
    }
}

impl DiagnosticLog {
    pub(super) fn push(&mut self, diagnostic: EventDiagnostic) {
        if self
            .tail
            .as_ref()
            .is_none_or(|tail| tail.records.len() == CHUNK)
        {
            self.tail = Some(Rc::new(Chunk {
                previous: self.tail.take(),
                records: Vec::with_capacity(CHUNK),
            }));
        }
        if let Some(tail) = &mut self.tail {
            Rc::make_mut(tail).records.push(diagnostic);
            self.len += 1;
        }
    }

    pub(super) fn records(&self) -> Vec<EventDiagnostic> {
        let mut chunks = Vec::new();
        let mut chunk = self.tail.as_deref();
        while let Some(current) = chunk {
            chunks.push(current.records.as_slice());
            chunk = current.previous.as_deref();
        }
        let mut records = Vec::with_capacity(self.len);
        for chunk in chunks.into_iter().rev() {
            records.extend_from_slice(chunk);
        }
        records
    }
}

impl Drop for DiagnosticLog {
    fn drop(&mut self) {
        // Avoid recursive stack growth when the last reference owns a long
        // sequence of complete chunks.
        let mut tail = self.tail.take();
        while let Some(chunk) = tail {
            let Ok(mut chunk) = Rc::try_unwrap(chunk) else {
                break;
            };
            tail = chunk.previous.take();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn snapshots_share_history_and_discard_trial_diagnostics() {
        let mut log = DiagnosticLog::default();
        let diagnostic = |delta| EventDiagnostic {
            model: Arc::from("a1"),
            time: 1.,
            delta,
            detail: DiagnosticDetail::Timing(ModelDiagnostic {
                fatal: false,
                kind: "setup",
                input: 0,
                elapsed: 0.,
                maximum_elapsed: 0.,
                certainty: super::super::timing::Certainty::DefiniteViolation,
                minimum: 1.,
            }),
        };
        for i in 0..(2 * CHUNK + 1) {
            log.push(diagnostic(i));
        }
        let mut trial = log.clone();
        trial.push(diagnostic(2 * CHUNK + 1));
        assert_eq!(trial.records().len(), 2 * CHUNK + 2);
        drop(trial);
        assert_eq!(
            log.records(),
            (0..(2 * CHUNK + 1)).map(diagnostic).collect::<Vec<_>>()
        );
    }
}
