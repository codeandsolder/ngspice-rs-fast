//! Locating and driving the C `ngspice` binary.
//!
//! The port never links against the C library; comparison data is produced by
//! running the reference binary out of process. This module owns the
//! details that make that work:
//!
//! 1. **ASCII output.** ngspice writes binary rawfiles unless the command
//!    `set filetype=ascii` has been issued, and the `-r` option writes before
//!    any control script runs. So the fixture is instrumented with a
//!    `.control` block that sets the file type explicitly and then issues
//!    `write`, rather than relying on `-r`.
//! 2. **A scratch directory per fixture.** The instrumented deck is written
//!    into `target/xtask/golden/<name>/` together with the rawfile, so that the
//!    `write` argument can be a bare file name and never needs quoting.
//! 3. **Multi-analysis decks.** `write` alone writes only the *current* plot.
//!    A deck with several analysis cards is instrumented to write every plot by
//!    its C name (`write f.raw ac1.all dc1.all op1.all tran1.all`), in batch
//!    order (a `.noise` card writes two plots, `noise1.all noise2.all`; a
//!    single `.noise` card is therefore instrumented this way too, and C's
//!    `write` then lists each plot's vectors in its own sorted order, which
//!    the name-based comparators ignore). The simulator-aware caller supplies
//!    the schedule and checks the captured plot count. The block ends with
//!    `quit`: without it, batch mode re-runs every analysis after `.endc` for a
//!    deck with `.op` (`ft_savedotargs()` registers an op-only save list, so the
//!    other analyses then fail with "no data saved").
//! 4. **XSPICE code models.** ngspice runs without `spinit`, so no code model
//!    is loaded. A fixture that needs one (E/G `TABLE` uses the `analog`
//!    library's `pwl`, `POLY` uses `spice2poly`) names it on a
//!    `* xtask-codemodels: <name>...` comment; the scratch directory then gets
//!    a `.spiceinit` with just those `codemodel` commands.

use crate::process::{self, ProcessLimits, ProcessOutput};
use std::ffi::{OsStr, OsString};
use std::fmt::{Debug, Write as _};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

/// Normalize the reference frequency axis of an incremental batch rawfile.
///
/// The pinned C oracle assigns only `IFvalue.rValue` for the frequency.
/// `OUTpData` writes `cValue` when the plot is complex, reading an
/// uninitialized imaginary component. Only the first column named frequency
/// has this known real-valued domain. Use its normalized sample both for
/// the absolute error and the relative tolerance.
#[must_use]
pub fn batch_reference_sample(column: usize, name: &str, mut sample: [f64; 2]) -> [f64; 2] {
    if column == 0 && name == "frequency" {
        sample[1] = 0.0;
    }
    sample
}

/// The C binary, and what it says it is.
#[derive(Debug, Clone)]
pub struct Ngspice {
    /// Absolute path to the resolved binary.
    pub path: PathBuf,
    /// The version string scraped from `--version`, e.g. `ngspice-47+`.
    pub version: String,
    /// Limits used for every simulation invocation.
    pub limits: ProcessLimits,
}

/// Explicit version-probe and simulation budgets.
#[derive(Debug, Clone, Copy)]
pub struct OracleLimits {
    /// Budget for discovering each candidate executable's version.
    pub version: ProcessLimits,
    /// Budget for a live simulation.
    pub simulation: ProcessLimits,
}

impl Default for OracleLimits {
    fn default() -> Self {
        Self {
            version: ProcessLimits::VERSION,
            simulation: ProcessLimits::ORACLE,
        }
    }
}

/// Finds the C binary.
///
/// `workspace` is the port's workspace root, where a `build/src/ngspice` from
/// building the C tree in this worktree would live.
///
/// # Errors
///
/// A message listing everywhere it looked. An explicitly selected executable
/// (including the environment override) must be usable; it never silently
/// falls back to a different oracle.
pub fn locate(
    explicit: Option<&Path>,
    workspace: &Path,
    limits: OracleLimits,
) -> Result<Ngspice, String> {
    let selected = explicit
        .map(Path::to_path_buf)
        .or_else(|| std::env::var_os("NGSPICE_BIN").map(PathBuf::from));
    let candidates = selected.map_or_else(
        || vec![PathBuf::from("build/src/ngspice"), PathBuf::from("ngspice")],
        |path| vec![path],
    );

    let mut tried = Vec::new();
    for candidate in candidates {
        // A bare name is resolved through `PATH`; anything else is taken
        // relative to the workspace root, not to the caller's directory, and is
        // made absolute so that it still resolves when ngspice runs in a scratch
        // directory.
        for path in candidate_paths(&candidate, workspace) {
            let Some(resolved) = executable_path(&path) else {
                tried.push(format!("{} (not a file)", path.display()));
                continue;
            };
            match version_of(&resolved, limits.version) {
                Ok(version) => {
                    return Ok(Ngspice {
                        path: resolved,
                        version,
                        limits: limits.simulation,
                    });
                }
                Err(error) => tried.push(format!("{} ({error})", resolved.display())),
            }
        }
    }

    Err(format!(
        "could not find a usable ngspice binary; tried:\n  {}\n\
         Pass --ngspice <PATH> or set NGSPICE_BIN.",
        tried.join("\n  ")
    ))
}

fn candidate_paths(candidate: &Path, workspace: &Path) -> Vec<PathBuf> {
    if candidate.components().count() == 1 {
        std::env::var_os("PATH").map_or_else(Vec::new, |path| {
            std::env::split_paths(&path)
                .map(|directory| directory.join(candidate))
                .collect()
        })
    } else {
        vec![workspace.join(candidate)]
    }
}

fn executable_path(path: &Path) -> Option<PathBuf> {
    let path = if cfg!(windows) && path.extension().is_none() && !path.is_file() {
        path.with_extension("exe")
    } else {
        path.to_path_buf()
    };
    if path.is_file() {
        fs::canonicalize(&path).ok()
    } else {
        None
    }
}

fn version_of(binary: &Path, limits: ProcessLimits) -> Result<String, String> {
    let output =
        process::run(binary, ".", &["--version"], limits).map_err(|error| error.to_string())?;
    output.require_success("--version")?;
    let text = output.transcript();
    text.split_whitespace()
        .find(|word| word.starts_with("ngspice-"))
        .map(|word| word.trim_end_matches(':').to_owned())
        .ok_or_else(|| format!("no 'ngspice-<version>' token in --version output\n{text}"))
}

/// Isolated scratch directory, removed on normal return and unwinding.
/// Set `NGSPICE_KEEP_SCRATCH=1` to preserve it explicitly for diagnosis.
#[derive(Debug)]
pub struct Scratch {
    path: PathBuf,
    keep: bool,
}

impl Scratch {
    /// Create a new directory; an existing path is never reused or removed.
    ///
    /// # Errors
    /// The tag is not a simple name, or the directory cannot be created.
    pub fn new(tag: &str) -> Result<Self, String> {
        Self::create_at(
            &std::env::temp_dir(),
            tag,
            std::env::var_os("NGSPICE_KEEP_SCRATCH").is_some_and(|value| value == "1"),
        )
    }

    fn create_at(root: &Path, tag: &str, keep: bool) -> Result<Self, String> {
        static ID: AtomicUsize = AtomicUsize::new(0);
        if tag.is_empty()
            || !tag
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_'))
        {
            return Err(format!("invalid scratch tag {tag:?}"));
        }
        let root = fs::canonicalize(root)
            .map_err(|error| format!("scratch root {}: {error}", root.display()))?;
        loop {
            let path = root.join(format!(
                "ngspice-{tag}-{}-{}",
                std::process::id(),
                ID.fetch_add(1, Ordering::Relaxed)
            ));
            match fs::create_dir(&path) {
                Ok(()) => return Ok(Self { path, keep }),
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {}
                Err(error) => return Err(format!("creating {}: {error}", path.display())),
            }
        }
    }

    /// Absolute directory path, valid for this scratch owner's lifetime.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        if self.keep {
            eprintln!("preserved ngspice scratch: {}", self.path.display());
        } else if let Err(error) = fs::remove_dir_all(&self.path) {
            eprintln!("cleanup of {} failed: {error}", self.path.display());
        }
    }
}

impl Ngspice {
    /// Resolve the explicitly pinned live-test binary, with no PATH fallback.
    ///
    /// # Errors
    /// Missing/relative `NGSPICE_BIN`, or the executable/version probe fails.
    pub fn pinned() -> Result<Self, String> {
        Self::pinned_selection(std::env::var_os("NGSPICE_BIN"))
    }

    fn pinned_selection(selected: Option<OsString>) -> Result<Self, String> {
        let path = selected
            .map(PathBuf::from)
            .ok_or_else(|| "set absolute NGSPICE_BIN for live C tests".to_owned())?;
        if !path.is_absolute() {
            return Err(format!("NGSPICE_BIN must be absolute: {}", path.display()));
        }
        locate(Some(&path), Path::new("."), OracleLimits::default())
    }

    /// Run C in the specified isolated directory, retaining both captured
    /// streams and the child status independently. The binary is absolute.
    ///
    /// # Errors
    /// The child could not be launched or its output could not be captured.
    pub fn run(
        &self,
        directory: &Path,
        args: &[impl AsRef<OsStr> + Debug],
    ) -> Result<ProcessOutput, String> {
        process::run(&self.path, directory, args, self.limits).map_err(|error| error.to_string())
    }
}

/// The result of capturing one fixture.
#[derive(Debug, Clone)]
pub struct Captured {
    /// The rawfile text, exactly as ngspice wrote it.
    pub rawfile: String,
    /// Where the instrumented deck was written, for debugging.
    pub deck_path: PathBuf,
    /// ngspice's stdout and stderr combined, for diagnostics.
    pub log: String,
}

/// Builds the instrumented deck: the fixture plus a `.control` block that writes
/// an ASCII rawfile called `plot_file`.
///
/// # Errors
///
/// A message when the fixture already contains a `.control` section, because
/// golden fixtures must be pure decks.
fn instrument(netlist: &str, plot_file: &str) -> Result<String, String> {
    instrument_with(
        netlist,
        &format!(".control\nset filetype=ascii\nrun\nwrite {plot_file}\n.endc\n"),
    )
}

/// Builds the instrumented deck for a multi-analysis fixture: every plot is
/// written by name, in `plots` order, into one ASCII rawfile `plot_file`.
///
/// # Errors
///
/// As [`instrument`].
fn instrument_plots(netlist: &str, plot_file: &str, plots: &[String]) -> Result<String, String> {
    let vectors: Vec<String> = plots.iter().map(|name| format!("{name}.all")).collect();
    instrument_with(
        netlist,
        &format!(
            ".control\nset filetype=ascii\nrun\nwrite {plot_file} {}\nquit\n.endc\n",
            vectors.join(" ")
        ),
    )
}

fn instrument_with(netlist: &str, control: &str) -> Result<String, String> {
    if netlist
        .lines()
        .any(|line| line.trim().to_ascii_lowercase().starts_with(".control"))
    {
        return Err(
            "fixture already has a .control section; golden fixtures must be pure decks".to_owned(),
        );
    }

    let mut out = String::with_capacity(netlist.len() + control.len());
    let mut inserted = false;
    for line in netlist.lines() {
        if !inserted && line.trim().eq_ignore_ascii_case(".end") {
            out.push_str(control);
            inserted = true;
        }
        out.push_str(line);
        out.push('\n');
    }
    if !inserted {
        out.push_str(control);
    }
    Ok(out)
}

/// Runs the C binary on one fixture and returns the ASCII rawfile.
///
/// # Errors
///
/// The caller supplies plot names in batch order for multi-plot decks.
///
/// Any failure to prepare the scratch directory, run the binary, or find a
/// non-empty rawfile afterwards. The message includes ngspice's output.
pub fn capture(
    ngspice: &Ngspice,
    netlist: &Path,
    scratch: &Path,
    plots: &[String],
) -> Result<Captured, String> {
    let name = crate::fixtures::fixture_id(netlist)?;
    let directory = std::path::absolute(scratch.join(name))
        .map_err(|error| format!("resolving scratch directory: {error}"))?;
    fs::create_dir_all(&directory)
        .map_err(|error| format!("creating {}: {error}", directory.display()))?;

    let plot_file = format!("{name}.raw");
    let raw_path = directory.join(&plot_file);
    remove_stale(&raw_path)?;

    let text = fs::read_to_string(netlist)
        .map_err(|error| format!("reading {}: {error}", netlist.display()))?;
    write_codemodel_init(ngspice, &text, &directory)?;
    let deck = if plots.len() > 1 {
        instrument_plots(&text, &plot_file, plots)?
    } else {
        instrument(&text, &plot_file)?
    };
    let deck_path = directory.join(format!("{name}.cir"));
    fs::write(&deck_path, &deck)
        .map_err(|error| format!("writing {}: {error}", deck_path.display()))?;

    let output = ngspice.run(&directory, &[OsStr::new("-b"), deck_path.as_os_str()])?;
    let log = output.transcript();
    if !output.status.success() {
        return Err(format!(
            "{} failed on {name} with {}\n--- ngspice output ---\n{}",
            ngspice.path.display(),
            output.status,
            tail(&log, 30)
        ));
    }

    let rawfile = fs::read_to_string(&raw_path).map_err(|error| {
        format!(
            "{name}: no rawfile at {} after a successful run ({error})\n\
             --- ngspice output ---\n{}",
            raw_path.display(),
            tail(&log, 30)
        )
    })?;
    if rawfile.trim().is_empty() {
        return Err(format!(
            "{name}: ngspice wrote an empty rawfile at {}\n{}",
            raw_path.display(),
            tail(&log, 30)
        ));
    }
    if !rawfile.contains("Values:") {
        return Err(format!(
            "{name}: the rawfile is not in ASCII form; the instrumented deck should have \
             issued 'set filetype=ascii'\n{}\n{}",
            raw_path.display(),
            tail(&log, 30)
        ));
    }

    Ok(Captured {
        rawfile,
        deck_path,
        log,
    })
}

/// The marker comment through which a fixture asks for XSPICE code models.
const CODEMODEL_MARKER: &str = "* xtask-codemodels:";

/// The XSPICE code models a fixture names on a `* xtask-codemodels: a b`
/// comment line (E/G `TABLE` needs `analog`, `POLY` needs `spice2poly`).
fn requested_codemodels(netlist: &str) -> Vec<String> {
    netlist
        .lines()
        .filter_map(|line| {
            let trimmed = line.trim();
            trimmed
                .get(..CODEMODEL_MARKER.len())
                .filter(|prefix| prefix.eq_ignore_ascii_case(CODEMODEL_MARKER))
                .map(|_| &trimmed[CODEMODEL_MARKER.len()..])
        })
        .flat_map(str::split_whitespace)
        .map(str::to_ascii_lowercase)
        .collect()
}

/// Where a code model library lives: `NGSPICE_CODEMODEL_DIR/<name>.cm`, the
/// C build tree next to the binary (`xspice/icm/<name>/<name>.cm`) or an
/// installed `../lib/ngspice/<name>.cm`.
fn codemodel_path(ngspice: &Ngspice, name: &str) -> Result<PathBuf, String> {
    let file = format!("{name}.cm");
    let mut candidates = Vec::new();
    if let Some(directory) = std::env::var_os("NGSPICE_CODEMODEL_DIR") {
        candidates.push(PathBuf::from(directory).join(&file));
    }
    if let Some(bin) = ngspice.path.parent() {
        candidates.push(bin.join("xspice/icm").join(name).join(&file));
        candidates.push(bin.join("../lib/ngspice").join(&file));
    }
    candidates
        .iter()
        .find(|path| path.is_file())
        .map(|path| fs::canonicalize(path).unwrap_or_else(|_| path.clone()))
        .ok_or_else(|| {
            format!(
                "XSPICE code model '{name}' not found; tried {} (set NGSPICE_CODEMODEL_DIR)",
                candidates
                    .iter()
                    .map(|path| path.display().to_string())
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        })
}

/// Writes (or removes) the scratch directory's `.spiceinit`, which ngspice
/// reads from its working directory: `codemodel` commands for the libraries
/// the fixture requests and nothing else, so other fixtures run exactly as
/// before (no spinit, no compatibility mode).
fn write_codemodel_init(ngspice: &Ngspice, netlist: &str, directory: &Path) -> Result<(), String> {
    let init = directory.join(".spiceinit");
    let models = requested_codemodels(netlist);
    if models.is_empty() {
        return remove_stale(&init);
    }
    let mut text = String::new();
    for name in &models {
        let path = codemodel_path(ngspice, name)?;
        writeln!(text, "codemodel {}", path.display())
            .map_err(|error| format!("formatting {}: {error}", init.display()))?;
    }
    fs::write(&init, text).map_err(|error| format!("writing {}: {error}", init.display()))
}

fn remove_stale(path: &Path) -> Result<(), String> {
    match fs::remove_file(path) {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(format!("removing stale {}: {error}", path.display())),
    }
}

fn tail(text: &str, lines: usize) -> String {
    let all: Vec<&str> = text.lines().collect();
    let skip = all.len().saturating_sub(lines);
    all[skip..].join("\n")
}

/// Compares two rawfiles, ignoring the `Date:` header.
///
/// The date is the only header ngspice fills in from the clock, so it is the
/// only one that cannot be reproduced. Everything else — including `Command:`,
/// which carries the ngspice version — is compared. Numeric layout ignores
/// trailing whitespace, but `Title:` preserves it: it is deck data and must
/// agree with the title checked by `tests/golden_rawfiles.rs`.
#[must_use]
pub fn rawfiles_match(left: &str, right: &str) -> bool {
    comparable(left) == comparable(right)
}

fn comparable(text: &str) -> Vec<&str> {
    text.lines()
        .map(|line| {
            if line.starts_with("Title:") {
                line
            } else {
                line.trim_end()
            }
        })
        .filter(|line| !line.starts_with("Date:"))
        .collect()
}

/// The first differing line, for a drift report.
#[must_use]
pub fn first_difference(left: &str, right: &str) -> Option<String> {
    let left = comparable(left);
    let right = comparable(right);
    for index in 0..left.len().max(right.len()) {
        match (left.get(index), right.get(index)) {
            (Some(a), Some(b)) if a == b => {}
            (Some(a), Some(b)) => {
                return Some(format!("line {}: golden {a:?} vs fresh {b:?}", index + 1));
            }
            (Some(a), None) => return Some(format!("line {}: golden {a:?} is missing", index + 1)),
            (None, Some(b)) => return Some(format!("line {}: extra line {b:?}", index + 1)),
            (None, None) => break,
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::{
        first_difference, instrument, instrument_plots, rawfiles_match, requested_codemodels,
    };

    #[test]
    fn batch_reference_preserves_every_component_except_frequency_axis_imaginary() {
        for imaginary in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, -1e213, 1e-300] {
            let sample = super::batch_reference_sample(0, "frequency", [10., imaginary]);
            assert_eq!(
                sample.map(f64::to_bits),
                [10.0_f64.to_bits(), 0.0_f64.to_bits()]
            );
            let tolerance = 1e-6 * sample[0].hypot(sample[1]) + 1e-12;
            assert!((11. - sample[0]).abs() > tolerance);
            assert!((10. - sample[0]).abs() <= tolerance);
            assert_eq!(
                super::batch_reference_sample(1, "frequency", [10., imaginary])[1].to_bits(),
                imaginary.to_bits()
            );
            assert_eq!(
                super::batch_reference_sample(0, "v(out)", [10., imaginary])[1].to_bits(),
                imaginary.to_bits()
            );
        }
    }

    #[test]
    fn scratch_names_are_unique_and_cleanup_survives_unwinding() {
        use super::Scratch;
        let root = Scratch::create_at(&std::env::temp_dir(), "scratch-root", false).unwrap();
        let first = Scratch::create_at(root.path(), "same", false).unwrap();
        let second = Scratch::create_at(root.path(), "same", false).unwrap();
        assert_ne!(first.path(), second.path());
        let path = first.path().to_path_buf();
        let result = std::panic::catch_unwind(move || {
            std::fs::write(first.path().join("data"), "test").unwrap();
            panic!("intentional test unwind");
        });
        assert!(result.is_err());
        assert!(!path.exists());
        assert!(second.path().exists());
        assert!(Scratch::create_at(root.path(), "../escape", false).is_err());
        let kept = Scratch::create_at(root.path(), "retained", true).unwrap();
        let kept_path = kept.path().to_path_buf();
        drop(kept);
        assert!(kept_path.exists());
    }

    #[cfg(unix)]
    #[test]
    fn runner_keeps_streams_status_and_launch_errors_separate() {
        let directory = super::Scratch::create_at(&std::env::temp_dir(), "child", false).unwrap();
        let binary = super::Ngspice {
            path: "/bin/sh".into(),
            version: "test".into(),
            limits: super::ProcessLimits::ORACLE,
        };
        let output = binary
            .run(
                directory.path(),
                &["-c", "printf hello; printf warning >&2; exit 7"],
            )
            .unwrap();
        assert_eq!(output.status.code(), Some(7));
        assert_eq!(output.stdout, b"hello");
        assert_eq!(output.stderr, b"warning");
        let error = output.require_success("child").unwrap_err();
        assert!(error.contains("hello") && error.contains("warning") && error.contains("child"));
        let missing = super::Ngspice {
            path: directory.path().join("missing-executable"),
            version: "test".into(),
            limits: super::ProcessLimits::ORACLE,
        };
        assert!(
            missing
                .run(directory.path(), &["-b"])
                .unwrap_err()
                .contains("missing-executable")
        );
    }

    #[test]
    fn fixtures_request_code_models_by_a_marker_comment() {
        let deck = "t\n* xtask-codemodels: spice2poly Analog\n* other\nr1 a 0 1\n.end\n";
        assert_eq!(requested_codemodels(deck), ["spice2poly", "analog"]);
        assert_eq!(requested_codemodels("t\nr1 a 0 1\n"), Vec::<String>::new());
    }

    #[test]
    fn live_oracle_requires_an_absolute_explicit_selection() {
        assert!(
            super::Ngspice::pinned_selection(None)
                .unwrap_err()
                .contains("set absolute NGSPICE_BIN")
        );
        assert!(
            super::Ngspice::pinned_selection(Some("ngspice".into()))
                .unwrap_err()
                .contains("must be absolute")
        );
        let directory = super::Scratch::new("pinned-selection").unwrap();
        let absent = directory.path().join("missing-c").into_os_string();
        assert!(
            super::Ngspice::pinned_selection(Some(absent))
                .unwrap_err()
                .contains("missing-c")
        );
    }

    #[test]
    fn missing_explicit_binary_never_falls_back() {
        let directory = super::Scratch::new("conformance-test").unwrap();
        let binary = directory.path().join("missing-ngspice");
        let error = super::locate(
            Some(&binary),
            directory.path(),
            super::OracleLimits::default(),
        )
        .unwrap_err();
        assert!(error.contains(&binary.display().to_string()), "{error}");
    }

    #[test]
    fn stale_init_is_removed_or_reported() {
        let directory = super::Scratch::new("conformance-test").unwrap();
        let binary = super::Ngspice {
            path: directory.path().join("missing-ngspice"),
            version: "test".to_owned(),
            limits: super::ProcessLimits::ORACLE,
        };
        let init = directory.path().join(".spiceinit");
        std::fs::write(&init, "codemodel obsolete.cm").unwrap();
        super::write_codemodel_init(&binary, DECK, directory.path()).unwrap();
        assert!(!init.exists());
        super::write_codemodel_init(&binary, DECK, directory.path()).unwrap();
        std::fs::create_dir(&init).unwrap();
        let error = super::write_codemodel_init(&binary, DECK, directory.path()).unwrap_err();
        assert!(
            error.contains("removing stale") && error.contains(".spiceinit"),
            "{error}"
        );
    }

    #[test]
    fn stale_rawfile_failure_precedes_process_launch() {
        let directory = super::Scratch::new("conformance-test").unwrap();
        let binary = super::Ngspice {
            path: directory.path().join("missing-ngspice"),
            version: "test".to_owned(),
            limits: super::ProcessLimits::ORACLE,
        };
        let deck = directory.path().join("fixture.cir");
        std::fs::write(&deck, DECK).unwrap();
        let scratch = directory.path().join("scratch");
        std::fs::create_dir_all(scratch.join("fixture/fixture.raw")).unwrap();
        let error = super::capture(&binary, &deck, &scratch, &[]).unwrap_err();
        assert!(
            error.contains("removing stale") && error.contains("fixture.raw"),
            "{error}"
        );
    }

    #[cfg(unix)]
    fn executable_fixture(directory: &std::path::Path, script: &str) -> std::path::PathBuf {
        use std::os::unix::fs::PermissionsExt as _;
        let path = directory.join("fake-ngspice");
        std::fs::write(&path, format!("#!/bin/sh\n{script}\n")).unwrap();
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o700)).unwrap();
        path
    }

    #[cfg(unix)]
    #[test]
    fn version_probe_has_its_own_budget_and_diagnostics() {
        use std::time::Duration;
        let scratch = super::Scratch::new("version-fixture").unwrap();
        let path = executable_fixture(scratch.path(), "printf 'ngspice-test:\\n'");
        let limits = super::OracleLimits {
            version: super::ProcessLimits {
                timeout: Duration::from_secs(2),
                output_bytes: 1024,
            },
            simulation: super::ProcessLimits::ORACLE,
        };
        let binary = super::locate(Some(&path), scratch.path(), limits).unwrap();
        assert_eq!(binary.version, "ngspice-test");
        let path = executable_fixture(
            scratch.path(),
            "printf 'probe started\\n' >&2; exec sleep 30",
        );
        let error = super::locate(Some(&path), scratch.path(), limits).unwrap_err();
        assert!(
            error.contains("Timeout") && error.contains("probe started"),
            "{error}"
        );
        let path = executable_fixture(scratch.path(), "printf 'not the oracle\\n'");
        let error = super::locate(Some(&path), scratch.path(), limits).unwrap_err();
        assert!(
            error.contains("no 'ngspice-<version>' token") && error.contains("not the oracle"),
            "{error}"
        );
    }

    #[cfg(unix)]
    #[test]
    fn capture_does_not_accept_failed_partial_rawfile() {
        use std::time::Duration;
        let scratch = super::Scratch::new("partial-rawfile").unwrap();
        let deck = scratch.path().join("fixture.cir");
        std::fs::write(&deck, DECK).unwrap();
        let output = scratch.path().join("capture");
        let mut binary = super::Ngspice {
            path: executable_fixture(
                scratch.path(),
                "printf 'Values:\\n' > fixture.raw; printf 'interrupted\\n' >&2; exec sleep 30",
            ),
            version: "test".into(),
            limits: super::ProcessLimits {
                timeout: Duration::from_secs(2),
                ..super::ProcessLimits::ORACLE
            },
        };
        let error = super::capture(&binary, &deck, &output, &[]).unwrap_err();
        assert!(
            error.contains("Timeout") && error.contains("interrupted"),
            "{error}"
        );
        assert_eq!(
            std::fs::read(output.join("fixture/fixture.raw")).unwrap(),
            b"Values:\n"
        );
        binary.path = executable_fixture(
            scratch.path(),
            "printf 'bad rawfile\\n' > fixture.raw; printf 'invalid format\\n' >&2",
        );
        let error = super::capture(&binary, &deck, &output, &[]).unwrap_err();
        assert!(
            error.contains("not in ASCII form") && error.contains("invalid format"),
            "{error}"
        );
    }

    const DECK: &str = "RC divider\nv1 in 0 dc 5\n.op\n.end\n";

    #[test]
    fn instrument_inserts_before_the_end_card() {
        let deck = instrument(DECK, "plot.raw").expect("instrumented");
        let lines: Vec<&str> = deck.lines().collect();
        assert_eq!(lines[0], "RC divider");
        assert_eq!(lines[1], "v1 in 0 dc 5");
        assert_eq!(lines[2], ".op");
        assert_eq!(lines[3], ".control");
        assert_eq!(lines[4], "set filetype=ascii");
        assert_eq!(lines[5], "run");
        assert_eq!(lines[6], "write plot.raw");
        assert_eq!(lines[7], ".endc");
        assert_eq!(lines[8], ".end");
    }

    #[test]
    fn multi_analysis_decks_write_every_plot_by_name_and_quit() {
        let deck = instrument_plots(
            "t\n.tran 1u 1m\n.op\n.end\n",
            "plot.raw",
            &["op1".to_owned(), "tran1".to_owned()],
        )
        .expect("instrumented");
        let lines: Vec<&str> = deck.lines().collect();
        assert_eq!(
            lines[3..],
            [
                ".control",
                "set filetype=ascii",
                "run",
                "write plot.raw op1.all tran1.all",
                "quit",
                ".endc",
                ".end",
            ]
        );
    }

    #[test]
    fn instrument_appends_when_there_is_no_end_card() {
        let deck = instrument("title\n.op\n", "plot.raw").expect("instrumented");
        assert!(deck.ends_with(".endc\n"));
        assert_eq!(deck.lines().count(), 7);
    }

    #[test]
    fn instrument_refuses_a_deck_that_already_has_control() {
        let error = instrument("t\n.control\nrun\n.endc\n.end\n", "plot.raw").unwrap_err();
        assert!(error.contains(".control"), "{error}");
    }

    #[test]
    fn comparison_ignores_only_the_date() {
        let a = "Title: t\nDate: Mon\nCommand: ngspice-47+\nValues:\n 0\t1.0\n";
        let b = "Title: t\nDate: Tue\nCommand: ngspice-47+\nValues:\n 0\t1.0\n";
        assert!(rawfiles_match(a, b));
        let spaced_title = b.replace("Title: t\n", "Title: t \n");
        assert!(!rawfiles_match(a, &spaced_title));
        assert!(
            first_difference(a, &spaced_title)
                .unwrap()
                .contains("line 1")
        );
        assert!(rawfiles_match(a, &b.replace("1.0\n", "1.0 \t\n")));

        let c = "Title: t\nDate: Mon\nCommand: ngspice-48\nValues:\n 0\t1.0\n";
        assert!(!rawfiles_match(a, c));
        assert!(first_difference(a, c).is_some_and(|text| text.contains("ngspice-48")));
    }

    #[test]
    fn difference_reports_missing_and_extra_lines() {
        let a = "Title: t\nValues:\n 0\t1.0\n";
        let b = "Title: t\nValues:\n 0\t1.0\n 1\t2.0\n";
        assert!(first_difference(a, b).is_some_and(|text| text.contains("extra line")));
        assert!(first_difference(b, a).is_some_and(|text| text.contains("missing")));
        assert_eq!(first_difference(a, a), None);
    }
}
