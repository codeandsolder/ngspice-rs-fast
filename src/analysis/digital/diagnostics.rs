//! Accepted diagnostic history with cheap disposable event snapshots.
use crate::devices::code_model::ModelDiagnostic;
use crate::primitives::Real;
use std::rc::Rc;
use std::sync::Arc;

/// Timing warning at a resolved event, retained only on transaction commit.
#[derive(Clone, Debug, PartialEq)]
pub struct EventDiagnostic {
    /// Flattened instance name, or a generated model ordinal for a model
    /// constructed directly through the event network API.
    pub model: Arc<str>,
    /// Physical time in seconds.
    pub time: Real,
    /// Delta-cycle ordinal at the physical time.
    pub delta: usize,
    /// Constraint and measured interval.
    pub detail: ModelDiagnostic,
}

impl std::fmt::Display for EventDiagnostic {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}: {} violation at t={:e} (delta {}): input {} interval {:e} < required {:e}",
            self.model,
            self.detail.kind,
            self.time,
            self.delta,
            self.detail.input + 1,
            self.detail.elapsed,
            self.detail.minimum
        )
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
            detail: ModelDiagnostic {
                kind: "setup",
                input: 0,
                elapsed: 0.,
                minimum: 1.,
            },
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
