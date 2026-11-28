//! Explicit external source transformation at the physical read boundary.

use crate::primitives::{SourceLoc, SpiceError, SpiceResult};
use model_preprocess_command::SourceSession;
use std::path::Path;

pub use model_preprocess_command::{AdmissionConfig, Cancellation, CommandConfig, ProcessLimits};

/// Opt-in source admission settings and cancellation for a complete deck graph.
///
/// The resolver owns the graph cache: a command runs once for each unique
/// physical source, regardless of repeated includes or library sections.
#[derive(Clone, Debug, Default)]
pub struct SourceAdmission {
    config: AdmissionConfig,
    cancellation: Cancellation,
}

impl SourceAdmission {
    /// Configure bounded admission and its cancellation token.
    #[must_use = "the settings control the source graph admission"]
    pub const fn new(config: AdmissionConfig, cancellation: Cancellation) -> Self {
        Self {
            config,
            cancellation,
        }
    }

    pub(in crate::netlist) const fn transforms(&self) -> bool {
        self.config.command.is_some()
    }

    pub(in crate::netlist) fn check_cancelled(&self, location: &SourceLoc) -> SpiceResult<()> {
        if self.cancellation.is_cancelled() {
            Err(SpiceError::parse(
                location.clone(),
                "source processing cancelled",
            ))
        } else {
            Ok(())
        }
    }

    pub(super) fn read(
        &self,
        path: &Path,
        remaining: usize,
        location: &SourceLoc,
    ) -> SpiceResult<(String, usize)> {
        let fail = |error: &dyn std::fmt::Display| {
            SpiceError::parse(
                location.clone(),
                format!("source admission {}: {error}", path.display()),
            )
        };
        // Bound the original read before launching a process. The original
        // and replacement together must fit the graph's remaining budget.
        let mut config = self.config.clone();
        config.input_bytes = config.input_bytes.min(remaining);
        let command = config.command.take();
        let mut raw = SourceSession::new(config.clone(), self.cancellation.clone());
        let original = raw
            .load(path)
            .map_err(|error| fail(&format_args!("{error:#}")))?;
        let input_bytes = original.bytes.len();
        let (admitted, charged) = if let Some(command) = command {
            config.command = Some(command);
            config.process.output_bytes = config.process.output_bytes.min(remaining - input_bytes);
            let mut session = SourceSession::new(config, self.cancellation.clone());
            let replacement = session
                .admit_bytes(&original.path, original.bytes)
                .map_err(|error| fail(&format_args!("{error:#}")))?;
            let charged = input_bytes + replacement.bytes.len();
            (replacement, charged)
        } else {
            (original, input_bytes)
        };
        let text = std::str::from_utf8(&admitted.bytes).map_err(|error| {
            SpiceError::parse(location.clone(), format!("source is not UTF-8: {error}"))
        })?;
        let mut owned = String::new();
        owned
            .try_reserve_exact(text.len())
            .map_err(|error| fail(&error))?;
        owned.push_str(text);
        Ok((owned, charged))
    }
}
