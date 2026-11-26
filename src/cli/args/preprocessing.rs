//! Explicit CLI source admission options; never interpret an argument as shell code.

use crate::netlist::SourceAdmission;
use crate::netlist::source::admission::{AdmissionConfig, Cancellation, CommandConfig};
use std::path::PathBuf;

/// Literal executable and arguments supplied at the invocation boundary.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct PreprocessArgs {
    /// Absolute executable path; absent for ordinary plaintext admission.
    pub executable: Option<PathBuf>,
    /// Arguments preceding the source's canonical path.
    pub arguments: Vec<String>,
}

impl PreprocessArgs {
    pub(super) fn set_executable(&mut self, path: PathBuf) -> Result<(), String> {
        if self.executable.is_some() {
            return Err("--preprocess-command is given more than once".into());
        }
        self.executable = Some(path);
        Ok(())
    }

    pub(super) fn validate(&self, needs_netlist: bool) -> Result<(), String> {
        if let Some(executable) = &self.executable {
            if !needs_netlist {
                return Err("--preprocess-command requires a netlist command".into());
            }
            if !executable.is_absolute() {
                return Err("--preprocess-command requires an absolute executable path".into());
            }
        } else if !self.arguments.is_empty() {
            return Err("--preprocess-arg requires --preprocess-command".into());
        }
        Ok(())
    }

    pub(super) fn admission(&self) -> Option<SourceAdmission> {
        let executable = self.executable.clone()?;
        let command = CommandConfig {
            executable,
            arguments: self.arguments.iter().map(Into::into).collect(),
            working_directory: None,
        };
        Some(SourceAdmission::new(
            AdmissionConfig {
                command: Some(command),
                ..AdmissionConfig::default()
            },
            Cancellation::default(),
        ))
    }
}
