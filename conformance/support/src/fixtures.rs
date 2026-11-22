//! C-reference selection for conformance tests and xtask.
//!
//! Most fixtures use one golden; explicitly documented libm-sensitive
//! fixtures may have independently captured platform or exact-input variants.
//! Capture and drift checks select the applicable reference; comparisons use
//! the observed envelope of documented captures. Missing data is an error.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

/// Fixture deck directory, relative to the workspace.
pub const NETLIST_DIR: &str = "conformance/netlists";

/// C-only decks whose decimal spellings produce the intended binary64 inputs.
pub const REFERENCE_INPUT_DIR: &str = "conformance/reference-inputs";

/// Reference directory, relative to the workspace.
pub const GOLDEN_DIR: &str = "conformance/golden";

/// Find the conformance workspace above a runtime directory.
#[must_use]
pub fn workspace_root_from(directory: &Path) -> Option<PathBuf> {
    directory
        .ancestors()
        .find(|root| {
            root.join("Cargo.toml").is_file()
                && root.join(NETLIST_DIR).is_dir()
                && root.join(GOLDEN_DIR).is_dir()
                && root.join("xtask/Cargo.toml").is_file()
        })
        .map(Path::to_path_buf)
}

/// Find the conformance workspace from the current working directory.
///
/// # Errors
/// The current directory is unavailable or is outside the workspace.
pub fn workspace_root() -> Result<PathBuf, String> {
    let directory =
        std::env::current_dir().map_err(|error| format!("current directory: {error}"))?;
    workspace_root_from(&directory).ok_or_else(|| {
        format!(
            "no ngspice conformance workspace above {}",
            directory.display()
        )
    })
}

/// Resolve the capture scratch root beneath the configured Cargo target directory.
///
/// Relative target directories are interpreted relative to the workspace.
#[must_use]
pub fn scratch_directory(root: &Path, target: Option<&Path>) -> PathBuf {
    root.join(target.unwrap_or_else(|| Path::new("target")))
        .join("xtask/golden")
}

/// Discover fixtures in deterministic order, with case-insensitive IDs.
///
/// # Errors
/// Directory access, invalid fixture names, duplicate IDs, or a missing selection.
pub fn netlist_paths_at(root: &Path, only: Option<&str>) -> Result<Vec<PathBuf>, String> {
    let directory = root.join(NETLIST_DIR);
    let entries = fs::read_dir(&directory)
        .map_err(|error| format!("reading {}: {error}", directory.display()))?;
    let mut paths = BTreeMap::new();
    for entry in entries {
        let path = entry
            .map_err(|error| format!("reading {}: {error}", directory.display()))?
            .path();
        if !path
            .extension()
            .is_some_and(|ext| ext.eq_ignore_ascii_case("cir"))
        {
            continue;
        }
        let id = fixture_id(&path)?.to_ascii_lowercase();
        if let Some(previous) = paths.insert(id, path.clone()) {
            return Err(format!(
                "duplicate fixture ID: {} and {}",
                previous.display(),
                path.display()
            ));
        }
    }
    if let Some(name) = only {
        let wanted = name.to_ascii_lowercase();
        let wanted = wanted.strip_suffix(".cir").unwrap_or(&wanted);
        return paths
            .remove(wanted)
            .map(|path| vec![path])
            .ok_or_else(|| format!("no fixture named '{name}' in {}", directory.display()));
    }
    if paths.is_empty() {
        return Err(format!("no .cir fixtures in {}", directory.display()));
    }
    Ok(paths.into_values().collect())
}

/// Extract the fixture's original case-preserving stem.
///
/// # Errors
/// The path does not have a nonempty UTF-8 file stem.
pub fn fixture_id(path: &Path) -> Result<&str, String> {
    path.file_stem()
        .and_then(|stem| stem.to_str())
        .filter(|stem| !stem.is_empty())
        .ok_or_else(|| format!("fixture {} has no file name", path.display()))
}

/// Whether C needs equivalent decimal spellings to round to the intended inputs.
fn exact_input(name: &str) -> bool {
    ["sens_hfet2", "sens_mos3", "sens_mos9"]
        .iter()
        .any(|fixture| name.eq_ignore_ascii_case(fixture))
}

/// The deck to give C when capturing a fixture. Rust runs the original deck.
///
/// # Errors
/// The original fixture has no valid ID.
pub fn reference_netlist_path(root: &Path, original: &Path) -> Result<PathBuf, String> {
    let name = fixture_id(original)?;
    Ok(if exact_input(name) {
        root.join(REFERENCE_INPUT_DIR)
            .join(format!("{}.cir", name.to_ascii_lowercase()))
    } else {
        original.to_path_buf()
    })
}

/// The C golden for `name` on the verifier/capture target.
///
/// Capture and byte-drift checks always use this specific path.
#[must_use]
pub fn golden_path(root: &Path, name: &str) -> PathBuf {
    path_for(
        root,
        name,
        cfg!(all(
            target_os = "linux",
            target_arch = "x86_64",
            target_env = "gnu"
        )),
    )
}

/// Every independently captured C reference used for numerical comparison.
///
/// The native platform's file comes first. All listed captures are required;
/// none is a fallback for a missing or malformed native reference.
#[must_use]
pub fn numerical_golden_paths(root: &Path, name: &str) -> Vec<PathBuf> {
    let native = golden_path(root, name);
    let mut paths = vec![native];
    for linux in [false, true] {
        let candidate = path_for(root, name, linux);
        if !paths.contains(&candidate) {
            paths.push(candidate);
        }
    }
    paths
}

/// Closest point in the envelope of independently observed reference values.
///
/// Identical captures reduce to the ordinary single-reference comparison.
/// Nonfinite or absent references must never make a comparison pass.
///
/// # Errors
/// Empty or nonfinite reference, or nonfinite candidate.
pub fn nearest_reference(
    actual: f64,
    references: impl IntoIterator<Item = f64>,
) -> Result<f64, String> {
    if !actual.is_finite() {
        return Err("nonfinite candidate".to_owned());
    }
    let mut references = references.into_iter();
    let first = references.next().ok_or("empty reference envelope")?;
    if !first.is_finite() {
        return Err("nonfinite reference".to_owned());
    }
    let (mut lower, mut upper) = (first, first);
    for reference in references {
        if !reference.is_finite() {
            return Err("nonfinite reference".to_owned());
        }
        lower = lower.min(reference);
        upper = upper.max(reference);
    }
    Ok(actual.clamp(lower, upper))
}

fn path_for(root: &Path, name: &str, linux_x86_64_gnu: bool) -> PathBuf {
    let mut directory = root.join(GOLDEN_DIR);
    // bjtsetup.c scales ISE=1e-3 by IS=1e-16, so cktsens.c's ISE
    // perturbation is 1e-25. Platform libm rounding is amplified by that
    // divisor: independent Linux and macOS C captures disagree on these residues.
    let name = name.to_ascii_lowercase();
    if exact_input(&name) {
        directory.push("exact-input");
    } else if name == "sens_bjt" && linux_x86_64_gnu {
        directory.push("linux-x86_64-gnu");
    }
    directory.join(format!("{name}.raw"))
}

#[cfg(test)]
mod tests {
    use super::{golden_path, path_for};
    use std::path::Path;

    #[test]
    fn discovery_handles_case_and_rejects_colliding_ids() {
        let root = crate::ngspice::Scratch::new("conformance-fixtures").unwrap();
        let directory = root.path().join(super::NETLIST_DIR);
        std::fs::create_dir_all(&directory).unwrap();
        let a = directory.join("a.CIR");
        let b = directory.join("b.cir");
        std::fs::write(&b, "").unwrap();
        std::fs::write(&a, "").unwrap();
        assert_eq!(
            super::netlist_paths_at(root.path(), None).unwrap(),
            [a.clone(), b]
        );
        assert_eq!(
            super::netlist_paths_at(root.path(), Some("A.CIR")).unwrap(),
            [a]
        );
        let collision = directory.join("A.cir");
        if !collision.exists() {
            std::fs::write(collision, "").unwrap();
            let error = super::netlist_paths_at(root.path(), Some("b")).unwrap_err();
            assert!(error.contains("duplicate fixture ID"), "{error}");
        }
    }

    #[test]
    fn scratch_respects_target_directory() {
        let root = Path::new("workspace");
        assert_eq!(
            super::scratch_directory(root, None),
            root.join("target/xtask/golden")
        );
        assert_eq!(
            super::scratch_directory(root, Some(Path::new("build"))),
            root.join("build/xtask/golden")
        );
        let absolute = std::env::temp_dir();
        assert_eq!(
            super::scratch_directory(root, Some(&absolute)),
            absolute.join("xtask/golden")
        );
    }

    #[test]
    fn only_the_documented_fixture_uses_a_platform_variant() {
        let root = Path::new("scratch");
        assert_eq!(
            path_for(root, "sens_bjt", true),
            root.join("conformance/golden/linux-x86_64-gnu/sens_bjt.raw")
        );
        assert_eq!(
            path_for(root, "SENS_BJT", true),
            path_for(root, "sens_bjt", true)
        );
        assert_eq!(
            path_for(root, "sens_bjt", false),
            root.join("conformance/golden/sens_bjt.raw")
        );
        for name in ["rc_divider", "sens_ac_bjt", "sens_ac_mos_tran"] {
            assert_eq!(path_for(root, name, true), path_for(root, name, false));
        }
    }

    #[test]
    fn numerical_comparison_requires_every_documented_capture() {
        let root = Path::new("workspace");
        let paths = super::numerical_golden_paths(root, "SENS_BJT");
        assert_eq!(paths[0], golden_path(root, "sens_bjt"));
        assert_eq!(paths.len(), 2);
        assert!(paths.contains(&path_for(root, "sens_bjt", false)));
        assert!(paths.contains(&path_for(root, "sens_bjt", true)));
        assert_eq!(
            super::numerical_golden_paths(root, "rc_divider"),
            [golden_path(root, "rc_divider")]
        );
    }

    #[test]
    fn exact_input_reference_decks_and_captures_are_selected_together() {
        let root = Path::new("scratch");
        for name in ["sens_hfet2", "sens_mos3", "sens_mos9"] {
            let original = root.join(super::NETLIST_DIR).join(format!("{name}.cir"));
            let reference = super::reference_netlist_path(root, &original).unwrap();
            assert_eq!(
                reference,
                root.join(super::REFERENCE_INPUT_DIR)
                    .join(format!("{name}.cir"))
            );
            for linux in [true, false] {
                assert_eq!(
                    path_for(root, name, linux),
                    root.join(super::GOLDEN_DIR)
                        .join("exact-input")
                        .join(format!("{name}.raw"))
                );
            }
        }
        let original = root.join(super::NETLIST_DIR).join("rc_divider.cir");
        assert_eq!(
            super::reference_netlist_path(root, &original).unwrap(),
            original
        );
    }

    #[test]
    fn envelope_is_the_observed_range_not_a_global_tolerance() {
        use super::nearest_reference;
        assert_eq!(
            nearest_reference(0.2, [-1., 1.]).unwrap().to_bits(),
            0.2_f64.to_bits()
        );
        assert_eq!(
            nearest_reference(2., [-1., 1.]).unwrap().to_bits(),
            1_f64.to_bits()
        );
        assert_eq!(
            nearest_reference(-2., [-1., 1.]).unwrap().to_bits(),
            (-1_f64).to_bits()
        );
        assert_eq!(
            nearest_reference(0.2, [0.1, 0.1]).unwrap().to_bits(),
            0.1_f64.to_bits()
        );
        assert!(nearest_reference(0., []).is_err());
        assert!(nearest_reference(0., [f64::NAN, 0.]).is_err());
        assert!(nearest_reference(f64::INFINITY, [0.]).is_err());
    }

    #[test]
    fn selection_does_not_depend_on_whether_the_file_exists() {
        let root = Path::new("not-an-existing-reference-directory");
        let selected = golden_path(root, "sens_bjt");
        assert!(!selected.exists());
        assert_eq!(
            selected,
            path_for(
                root,
                "sens_bjt",
                cfg!(all(
                    target_os = "linux",
                    target_arch = "x86_64",
                    target_env = "gnu"
                ))
            )
        );
    }
}
