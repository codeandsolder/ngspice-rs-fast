//! Golden-data capture and drift checks.
//!
//! Fixtures are the decks in `conformance/netlists/`; golden data is the ASCII
//! rawfile the C binary produces for each of them, stored in
//! `conformance/golden/`. See `docs/port/VERIFICATION.md` for what the fixtures
//! are allowed to contain and why.

use std::fs;
use std::path::{Path, PathBuf};

use ngspice_rs::analysis::RawFile;

use ngspice_conformance::{
    fixtures,
    ngspice::{self, Captured, Ngspice},
};

const USAGE: &str = "\
usage: cargo xtask golden <capture|check|list|verify> [OPTIONS]
       cargo xtask golden verify [--netlist <NAME>] (Rust only; no C invocation)

OPTIONS:
    --ngspice <PATH>   the C ngspice binary to drive
                       (default: $NGSPICE_BIN, then build/src/ngspice, then PATH)
    --netlist <NAME>   only this fixture, e.g. --netlist rc_divider
    --verbose          echo each ngspice invocation";

/// Options shared by `capture` and `check`.
#[derive(Debug, Default)]
struct Options {
    ngspice: Option<PathBuf>,
    netlist: Option<String>,
    verbose: bool,
}

impl Options {
    fn parse(arguments: &[String]) -> Result<Self, String> {
        let mut options = Self::default();
        let mut index = 0;
        while index < arguments.len() {
            match arguments[index].as_str() {
                "--ngspice" => {
                    index += 1;
                    let value = arguments
                        .get(index)
                        .ok_or_else(|| "usage: --ngspice needs a path".to_owned())?;
                    options.ngspice = Some(PathBuf::from(value));
                }
                "--netlist" => {
                    index += 1;
                    let value = arguments
                        .get(index)
                        .ok_or_else(|| "usage: --netlist needs a name".to_owned())?;
                    options.netlist = Some(value.clone());
                }
                "--verbose" => options.verbose = true,
                other => return Err(format!("usage: unknown option '{other}'")),
            }
            index += 1;
        }
        Ok(options)
    }
}

/// Entry point for `cargo xtask golden …`.
pub(super) fn main(root: &Path, arguments: &[String]) -> Result<(), String> {
    match arguments.first().map(String::as_str) {
        Some("capture") => capture(root, &Options::parse(&arguments[1..])?),
        Some("check") => check(root, &Options::parse(&arguments[1..])?),
        Some("list") => list(root),
        Some("verify") => super::verify::main(root, &arguments[1..]),
        _ => Err(format!("usage: unknown 'golden' subcommand\n\n{USAGE}")),
    }
}

fn scratch_directory(root: &Path) -> PathBuf {
    let target =
        std::env::var_os("CARGO_TARGET_DIR").or_else(|| std::env::var_os("CARGO_BUILD_TARGET_DIR"));
    fixtures::scratch_directory(root, target.as_deref().map(Path::new))
}

fn golden_path(root: &Path, netlist: &Path) -> Result<PathBuf, String> {
    Ok(fixtures::golden_path(root, fixtures::fixture_id(netlist)?))
}

fn locate(root: &Path, options: &Options) -> Result<Ngspice, String> {
    let binary = ngspice::locate(options.ngspice.as_deref(), root)?;
    println!("ngspice:  {} ({})", binary.path.display(), binary.version);
    Ok(binary)
}

// Scheduling and rawfile decoding use the simulator. The reference runner itself
// has no dependency on the implementation under test.
fn capture_fixture(binary: &Ngspice, netlist: &Path, scratch: &Path) -> Result<Captured, String> {
    let parsed = ngspice_rs::netlist::Parser::new()
        .parse_file(netlist)
        .map_err(|error| format!("parsing {}: {error}", netlist.display()))?;
    let config = ngspice_rs::analysis::RunConfig::from_netlist(&parsed)
        .map_err(|error| error.to_string())?;
    let plots: Vec<_> = ngspice_rs::analysis::batch::schedule_evaluated(&parsed.analyses, &config)
        .map_err(|error| error.to_string())?
        .iter()
        .flat_map(|entry| entry.plot_names().map(str::to_owned).collect::<Vec<_>>())
        .collect();
    let captured = ngspice::capture(binary, netlist, scratch, &plots)?;
    let actual = RawFile::parse(&captured.rawfile).map_err(|error| {
        format!(
            "{}: the captured rawfile does not parse: {error}",
            netlist.display()
        )
    })?;
    if plots.len() > 1 && actual.len() != plots.len() {
        return Err(format!(
            "{}: expected {} plots ({}), ngspice wrote {}",
            netlist.display(),
            plots.len(),
            plots.join(" "),
            actual.len()
        ));
    }
    Ok(captured)
}

fn describe(rawfile: &str) -> Result<String, String> {
    let parsed = RawFile::parse(rawfile).map_err(|error| error.to_string())?;
    let mut parts = Vec::new();
    for raw_plot in &parsed.plots {
        let plot = &raw_plot.plot;
        parts.push(format!(
            "{}: {} variable(s), {} point(s), {}",
            plot.plotname,
            plot.variable_count(),
            plot.point_count(),
            if plot.is_finite() {
                "all values finite"
            } else {
                "NON-FINITE VALUES"
            }
        ));
    }
    Ok(parts.join("; "))
}

fn capture(root: &Path, options: &Options) -> Result<(), String> {
    let binary = locate(root, options)?;
    let scratch = scratch_directory(root);
    fs::create_dir_all(&scratch)
        .map_err(|error| format!("creating {}: {error}", scratch.display()))?;

    let netlists = fixtures::netlist_paths_at(root, options.netlist.as_deref())?;
    let mut created = 0usize;
    let mut changed = 0usize;
    let mut unchanged = 0usize;

    for netlist in &netlists {
        if options.verbose {
            println!("capturing {}", netlist.display());
        }
        let captured = capture_fixture(&binary, netlist, &scratch)?;
        let target = golden_path(root, netlist)?;
        let name = fixtures::fixture_id(netlist)?;

        match fs::read_to_string(&target) {
            Ok(existing) if ngspice::rawfiles_match(&existing, &captured.rawfile) => {
                unchanged += 1;
                println!("  unchanged  {name}");
            }
            Ok(existing) => {
                changed += 1;
                let report = ngspice::first_difference(&existing, &captured.rawfile)
                    .unwrap_or_else(|| "differences are only in ignored headers".to_owned());
                println!("  changed    {name}  ({report})");
                write_golden(&target, &captured.rawfile)?;
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                created += 1;
                println!("  new        {name}");
                write_golden(&target, &captured.rawfile)?;
            }
            Err(error) => return Err(format!("reading {}: {error}", target.display())),
        }
        if options.verbose {
            println!("             deck: {}", captured.deck_path.display());
            println!(
                "             ngspice output: {} line(s)",
                captured.log.lines().count()
            );
        }
        println!("             {}", describe(&captured.rawfile)?);
    }

    println!(
        "\n{} fixture(s): {created} new, {changed} changed, {unchanged} unchanged",
        netlists.len()
    );
    Ok(())
}

fn write_golden(target: &Path, rawfile: &str) -> Result<(), String> {
    if let Some(parent) = target.parent() {
        fs::create_dir_all(parent)
            .map_err(|error| format!("creating {}: {error}", parent.display()))?;
    }
    fs::write(target, rawfile).map_err(|error| format!("writing {}: {error}", target.display()))
}

fn check(root: &Path, options: &Options) -> Result<(), String> {
    let binary = locate(root, options)?;
    let scratch = scratch_directory(root);
    fs::create_dir_all(&scratch)
        .map_err(|error| format!("creating {}: {error}", scratch.display()))?;

    let netlists = fixtures::netlist_paths_at(root, options.netlist.as_deref())?;
    let mut drifted = Vec::new();

    for netlist in &netlists {
        let captured = capture_fixture(&binary, netlist, &scratch)?;
        let target = golden_path(root, netlist)?;
        let name = fixtures::fixture_id(netlist)?;

        match fs::read_to_string(&target) {
            Ok(existing) if ngspice::rawfiles_match(&existing, &captured.rawfile) => {
                println!("  ok         {name}");
            }
            Ok(existing) => {
                let report = ngspice::first_difference(&existing, &captured.rawfile)
                    .unwrap_or_else(|| "differences are only in ignored headers".to_owned());
                println!("  DRIFT      {name}  ({report})");
                drifted.push(name.to_owned());
            }
            Err(error) => {
                println!("  MISSING    {name}  ({error})");
                drifted.push(name.to_owned());
            }
        }
    }

    if drifted.is_empty() {
        println!(
            "\n{} fixture(s) reproduce the committed goldens",
            netlists.len()
        );
        Ok(())
    } else {
        Err(format!(
            "drift in {} fixture(s): {}\n\
             If ngspice really did change its output, re-capture with \
             `cargo xtask golden capture` and review the resulting diff.",
            drifted.len(),
            drifted.join(", ")
        ))
    }
}

fn list(root: &Path) -> Result<(), String> {
    let netlists = fixtures::netlist_paths_at(root, None)?;
    println!("{:<22} {:<12} details", "fixture", "golden");
    for netlist in &netlists {
        let name = fixtures::fixture_id(netlist)?;
        let target = golden_path(root, netlist)?;
        match fs::read_to_string(&target) {
            Ok(rawfile) => println!("{name:<22} {:<12} {}", "present", describe(&rawfile)?),
            Err(_) => println!("{name:<22} {:<12} -", "missing"),
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::super::workspace_root;
    use super::{Options, golden_path};
    use ngspice_conformance::fixtures::netlist_paths_at;
    use std::path::PathBuf;

    #[test]
    fn options_are_parsed() {
        let options = Options::parse(&[
            "--ngspice".to_owned(),
            "/tmp/ngspice".to_owned(),
            "--netlist".to_owned(),
            "rc_divider".to_owned(),
            "--verbose".to_owned(),
        ])
        .expect("parses");
        assert_eq!(options.ngspice, Some(PathBuf::from("/tmp/ngspice")));
        assert_eq!(options.netlist.as_deref(), Some("rc_divider"));
        assert!(options.verbose);
    }

    #[test]
    fn missing_option_values_are_rejected() {
        assert!(Options::parse(&["--ngspice".to_owned()]).is_err());
        assert!(Options::parse(&["--netlist".to_owned()]).is_err());
        assert!(Options::parse(&["--nope".to_owned()]).is_err());
    }

    #[test]
    fn fixtures_are_discovered_and_filtered() {
        let paths = netlist_paths_at(&workspace_root(), None).expect("fixtures exist");
        assert!(!paths.is_empty());
        let first = paths[0]
            .file_stem()
            .and_then(|stem| stem.to_str())
            .expect("named")
            .to_owned();
        let filtered = netlist_paths_at(&workspace_root(), Some(&first)).expect("filter works");
        assert_eq!(filtered.len(), 1);
        // The `.cir` suffix is optional.
        let filtered = netlist_paths_at(&workspace_root(), Some(&format!("{first}.cir")))
            .expect("filter works");
        assert_eq!(filtered.len(), 1);
        assert!(netlist_paths_at(&workspace_root(), Some("definitely-not-a-fixture")).is_err());
    }

    #[test]
    fn goldens_are_named_after_their_fixture() {
        let path = golden_path(
            &workspace_root(),
            &PathBuf::from("conformance/netlists/rc_divider.cir"),
        )
        .expect("named");
        assert!(path.ends_with("conformance/golden/rc_divider.raw"));
    }
}
