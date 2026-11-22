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

use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

/// The C binary, and what it says it is.
#[derive(Debug, Clone)]
pub struct Ngspice {
    /// Absolute path to the resolved binary.
    pub path: PathBuf,
    /// The version string scraped from `--version`, e.g. `ngspice-47+`.
    pub version: String,
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
pub fn locate(explicit: Option<&Path>, workspace: &Path) -> Result<Ngspice, String> {
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
            match version_of(&resolved) {
                Ok(version) => {
                    return Ok(Ngspice {
                        path: resolved,
                        version,
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

fn version_of(binary: &Path) -> Result<String, String> {
    let output = Command::new(binary)
        .arg("--version")
        .output()
        .map_err(|error| error.to_string())?;
    if !output.status.success() {
        return Err(format!("--version exited with {}", output.status));
    }
    let text = String::from_utf8_lossy(&output.stdout).into_owned()
        + &String::from_utf8_lossy(&output.stderr);
    text.split_whitespace()
        .find(|word| word.starts_with("ngspice-"))
        .map(|word| word.trim_end_matches(':').to_owned())
        .ok_or_else(|| "no 'ngspice-<version>' token in --version output".to_owned())
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

    let output = Command::new(&ngspice.path)
        .arg("-b")
        .arg(&deck_path)
        .current_dir(&directory)
        .output()
        .map_err(|error| format!("running {}: {error}", ngspice.path.display()))?;

    let log = String::from_utf8_lossy(&output.stdout).into_owned()
        + &String::from_utf8_lossy(&output.stderr);
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
        return Err(format!("{name}: ngspice wrote an empty rawfile"));
    }
    if !rawfile.contains("Values:") {
        return Err(format!(
            "{name}: the rawfile is not in ASCII form; the instrumented deck should have \
             issued 'set filetype=ascii'"
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
    fn fixtures_request_code_models_by_a_marker_comment() {
        let deck = "t\n* xtask-codemodels: spice2poly Analog\n* other\nr1 a 0 1\n.end\n";
        assert_eq!(requested_codemodels(deck), ["spice2poly", "analog"]);
        assert!(requested_codemodels("t\nr1 a 0 1\n").is_empty());
    }

    #[test]
    fn missing_explicit_binary_never_falls_back() {
        let directory = crate::Temp::new();
        let binary = directory.0.join("missing-ngspice");
        let error = super::locate(Some(&binary), &directory.0).unwrap_err();
        assert!(error.contains(&binary.display().to_string()), "{error}");
    }

    #[test]
    fn stale_init_is_removed_or_reported() {
        let directory = crate::Temp::new();
        let binary = super::Ngspice {
            path: directory.0.join("missing-ngspice"),
            version: "test".to_owned(),
        };
        let init = directory.0.join(".spiceinit");
        std::fs::write(&init, "codemodel obsolete.cm").unwrap();
        super::write_codemodel_init(&binary, DECK, &directory.0).unwrap();
        assert!(!init.exists());
        super::write_codemodel_init(&binary, DECK, &directory.0).unwrap();
        std::fs::create_dir(&init).unwrap();
        let error = super::write_codemodel_init(&binary, DECK, &directory.0).unwrap_err();
        assert!(
            error.contains("removing stale") && error.contains(".spiceinit"),
            "{error}"
        );
    }

    #[test]
    fn stale_rawfile_failure_precedes_process_launch() {
        let directory = crate::Temp::new();
        let binary = super::Ngspice {
            path: directory.0.join("missing-ngspice"),
            version: "test".to_owned(),
        };
        let deck = directory.0.join("fixture.cir");
        std::fs::write(&deck, DECK).unwrap();
        let scratch = directory.0.join("scratch");
        std::fs::create_dir_all(scratch.join("fixture/fixture.raw")).unwrap();
        let error = super::capture(&binary, &deck, &scratch, &[]).unwrap_err();
        assert!(
            error.contains("removing stale") && error.contains("fixture.raw"),
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
