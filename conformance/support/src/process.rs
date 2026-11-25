//! Bounded, shell-free execution of reference processes.
use std::cell::Cell;
use std::ffi::OsStr;
use std::fmt;
use std::io::{self, Write};
use std::path::Path;
use std::time::{Duration, Instant};

use subprocess::{Exec, ExitStatus, Job, Redirection};

/// Limits for one invocation, including pipe drainage and process exit.
#[derive(Debug, Clone, Copy)]
pub struct ProcessLimits {
    /// Maximum elapsed time.
    pub timeout: Duration,
    /// Combined stdout and stderr byte budget.
    pub output_bytes: usize,
}

impl ProcessLimits {
    /// Default live simulation budget.
    pub const ORACLE: Self = Self {
        timeout: Duration::from_secs(180),
        output_bytes: 8 * 1024 * 1024,
    };

    /// Version discovery needs much less time and output.
    pub const VERSION: Self = Self {
        timeout: Duration::from_secs(10),
        output_bytes: 128 * 1024,
    };
}

/// Successfully captured process output, independent of its exit status.
#[derive(Debug, Clone)]
pub struct ProcessOutput {
    /// Exit status; success does not imply a valid rawfile.
    pub status: ExitStatus,
    /// Unmodified standard output.
    pub stdout: Vec<u8>,
    /// Unmodified standard error.
    pub stderr: Vec<u8>,
}

fn transcript(stdout: &[u8], stderr: &[u8]) -> String {
    format!(
        "--- stdout ---\n{}\n--- stderr ---\n{}",
        String::from_utf8_lossy(stdout),
        String::from_utf8_lossy(stderr)
    )
}

impl ProcessOutput {
    /// Lossy text view for diagnostics, retaining separate streams.
    #[must_use = "the combined transcript contains the subprocess diagnostic evidence"]
    pub fn transcript(&self) -> String {
        transcript(&self.stdout, &self.stderr)
    }

    /// Require a successful exit.
    ///
    /// # Errors
    /// Nonzero exit or signal, with captured diagnostics.
    pub fn require_success(&self, label: &str) -> Result<(), String> {
        if self.status.success() {
            Ok(())
        } else {
            Err(format!("{label}: {}\n{}", self.status, self.transcript()))
        }
    }
}

/// Which part of the process lifecycle failed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Failure {
    /// Invalid executable path or limits.
    Configuration,
    /// Process creation failed.
    Launch,
    /// Deadline elapsed before successful collection and exit.
    Timeout,
    /// Combined stream budget exceeded.
    OutputLimit,
    /// Pipe setup or reading failed.
    Capture,
    /// Waiting for child exit failed.
    Wait,
}

/// A failed invocation with the bounded output retained until failure.
pub struct ProcessError {
    /// Classification of the primary error.
    pub failure: Failure,
    /// Captured prefix of stdout.
    pub stdout: Vec<u8>,
    /// Captured prefix of stderr.
    pub stderr: Vec<u8>,
    invocation: String,
    detail: String,
    /// Error terminating or reaping the child, if cleanup failed.
    pub cleanup_error: Option<String>,
}

impl fmt::Display for ProcessError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}: {:?}: {}",
            self.invocation, self.failure, self.detail
        )?;
        if let Some(error) = &self.cleanup_error {
            write!(f, "\nchild cleanup failed: {error}")?;
        }
        write!(f, "\n{}", transcript(&self.stdout, &self.stderr))
    }
}

impl fmt::Debug for ProcessError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

impl std::error::Error for ProcessError {}

// The communicator polls pipes on Unix and cancels overlapped reads on
// Windows. Enforce the *shared* budget ourselves, including when the pipes
// remain continuously readable: a poll timeout alone is insufficient.
struct Budget {
    deadline: Instant,
    remaining: Cell<usize>,
    failure: Cell<Option<Failure>>,
}

struct Stream<'a> {
    budget: &'a Budget,
    bytes: &'a mut Vec<u8>,
}

impl Write for Stream<'_> {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        if Instant::now() >= self.budget.deadline {
            self.budget.failure.set(Some(Failure::Timeout));
            return Err(io::ErrorKind::TimedOut.into());
        }
        let count = bytes.len().min(self.budget.remaining.get());
        self.bytes.extend_from_slice(&bytes[..count]);
        self.budget
            .remaining
            .set(self.budget.remaining.get() - count);
        if count != bytes.len() {
            self.budget.failure.set(Some(Failure::OutputLimit));
            return Err(io::Error::other("combined stdout/stderr budget exceeded"));
        }
        Ok(count)
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

// subprocess normally waits indefinitely on Drop. Explicitly own the
// termination/reaping responsibility, including unwinding.
struct Running {
    job: Job,
    finished: bool,
}

impl Running {
    fn terminate(&mut self) -> Result<(), String> {
        if self.finished {
            return Ok(());
        }
        self.finished = true;
        // On a kernel-level cleanup failure, avoid subprocess's unbounded
        // implicit wait. Normal termination still explicitly reaps below.
        self.job.detach();
        // Kill a direct child, not an assumed process group.
        self.job.kill().map_err(|error| format!("kill: {error}"))?;
        match self.job.wait_timeout(Duration::from_secs(5)) {
            Ok(Some(_)) => Ok(()),
            Ok(None) => Err("child did not exit within termination grace period".into()),
            Err(error) => Err(format!("reap: {error}")),
        }
    }
}

impl Drop for Running {
    fn drop(&mut self) {
        if let Err(error) = self.terminate() {
            eprintln!("child cleanup failed: {error}");
        }
    }
}

/// Run a direct child with bounded concurrent capture of both streams.
///
/// The executable must be absolute. The deadline also covers a child that
/// closes its pipes but keeps running. On failure the direct child is killed
/// and reaped before returning (or the cleanup failure is reported).
///
/// # Errors
/// Invalid configuration, failed launch, timeout, exhaustion of the combined
/// stream budget, pipe I/O error or failed wait.
pub fn run(
    program: impl AsRef<Path>,
    directory: impl AsRef<Path>,
    args: &[impl AsRef<OsStr> + fmt::Debug],
    limits: ProcessLimits,
) -> Result<ProcessOutput, Box<ProcessError>> {
    run_with_environment(program, directory, args, limits, &[])
}

/// Run with explicit, per-child environment overrides (None removes a key).
///
/// # Errors
/// As in [run].
pub fn run_with_environment(
    program: impl AsRef<Path>,
    directory: impl AsRef<Path>,
    args: &[impl AsRef<OsStr> + fmt::Debug],
    limits: ProcessLimits,
    environment: &[(&str, Option<&str>)],
) -> Result<ProcessOutput, Box<ProcessError>> {
    let program = program.as_ref();
    let directory = directory.as_ref();
    let invocation = format!(
        "running {} {args:?} in {} (deadline {:?}, combined output limit {} bytes)",
        program.display(),
        directory.display(),
        limits.timeout,
        limits.output_bytes
    );
    let mut stdout = Vec::new();
    let mut stderr = Vec::new();
    let result = (|| {
        let deadline = Instant::now()
            .checked_add(limits.timeout)
            .filter(|_| program.is_absolute() && !limits.timeout.is_zero())
            .ok_or_else(|| {
                (
                    Failure::Configuration,
                    "expected absolute executable and representable nonzero timeout".into(),
                    None,
                )
            })?;
        let mut exec = Exec::cmd(program.as_os_str())
            .args(args.iter().map(AsRef::as_ref))
            .cwd(directory)
            .stdin(Redirection::Null)
            .stdout(Redirection::Pipe)
            .stderr(Redirection::Pipe);
        for &(key, value) in environment {
            exec = match value {
                Some(value) => exec.env(key, value),
                None => exec.env_remove(key),
            };
        }
        let job = exec
            .start()
            .map_err(|error| (Failure::Launch, error.to_string(), None))?;
        let mut child = Running {
            job,
            finished: false,
        };
        let result = collect(
            &mut child.job,
            deadline,
            limits.output_bytes,
            &mut stdout,
            &mut stderr,
        );
        match result {
            Ok(status) => {
                // A successful explicit wait has already reaped this child.
                child.finished = true;
                Ok(status)
            }
            Err((failure, detail)) => {
                let cleanup = child.terminate().err();
                Err((failure, detail, cleanup))
            }
        }
    })();
    match result {
        Ok(status) => Ok(ProcessOutput {
            status,
            stdout,
            stderr,
        }),
        Err((failure, detail, cleanup_error)) => Err(Box::new(ProcessError {
            failure,
            stdout,
            stderr,
            invocation,
            detail,
            cleanup_error,
        })),
    }
}

fn collect(
    child: &mut Job,
    deadline: Instant,
    output_bytes: usize,
    stdout: &mut Vec<u8>,
    stderr: &mut Vec<u8>,
) -> Result<ExitStatus, (Failure, String)> {
    let budget = Budget {
        deadline,
        remaining: Cell::new(output_bytes),
        failure: Cell::new(None),
    };
    let remaining = || deadline.saturating_duration_since(Instant::now());
    {
        let mut communication = child
            .communicate()
            .map_err(|error| (Failure::Capture, error.to_string()))?
            .limit_time(remaining());
        communication
            .read_to(
                Stream {
                    budget: &budget,
                    bytes: stdout,
                },
                Stream {
                    budget: &budget,
                    bytes: stderr,
                },
            )
            .map_err(|error| {
                let failure = budget.failure.get().unwrap_or_else(|| {
                    if error.kind() == io::ErrorKind::TimedOut {
                        Failure::Timeout
                    } else {
                        Failure::Capture
                    }
                });
                (failure, error.to_string())
            })?;
    }
    if remaining().is_zero() {
        return Err((
            Failure::Timeout,
            "deadline elapsed while capturing streams".into(),
        ));
    }
    child
        .wait_timeout(remaining())
        .map_err(|error| (Failure::Wait, error.to_string()))?
        .ok_or_else(|| {
            (
                Failure::Timeout,
                "deadline elapsed waiting for child exit".into(),
            )
        })
}
