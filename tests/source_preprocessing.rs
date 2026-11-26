//! External Command admission at the ordinary bounded source graph boundary.
#![cfg(unix)]

use ngspice_rs::analysis::{RawFile, RawFormat};
use ngspice_rs::cli::args::Args;
use ngspice_rs::cli::simulate::run_with_admission;
use ngspice_rs::netlist::source::admission::{
    AdmissionConfig, Cancellation, CommandConfig, ProcessLimits,
};
use ngspice_rs::netlist::{Parser, SourceAdmission, SourceLimits};
use std::fs;
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::time::Duration;

fn write(directory: &Path, name: &str, bytes: impl AsRef<[u8]>) -> PathBuf {
    let path = directory.join(name);
    fs::create_dir_all(path.parent().unwrap()).unwrap();
    fs::write(&path, bytes).unwrap();
    path
}

fn executable(directory: &Path, body: &str) -> PathBuf {
    let path = write(
        directory,
        "source-command",
        format!("#!/bin/sh\nset -eu\n{body}\n"),
    );
    fs::set_permissions(&path, fs::Permissions::from_mode(0o700)).unwrap();
    path.canonicalize().unwrap()
}

fn config(executable: PathBuf) -> AdmissionConfig {
    AdmissionConfig {
        command: Some(CommandConfig {
            executable,
            arguments: vec![],
            working_directory: None,
        }),
        ..AdmissionConfig::default()
    }
}

fn parse(
    path: &Path,
    config: AdmissionConfig,
    limits: SourceLimits,
) -> ngspice_rs::SpiceResult<ngspice_rs::netlist::ParsedDeck> {
    let admission = SourceAdmission::new(config, Cancellation::default());
    Parser::new().parse_file_with_admission(path, limits, Some(&admission))
}

#[test]
fn binary_input_linked_sections_and_nested_sources_are_admitted_once_per_graph() {
    let scratch = tempfile::tempdir().unwrap();
    let directory = scratch.path();
    let root = write(directory, "root.cir", [0xff, 0, 0x80]);
    let branch = write(directory, "linked/branch.inc", [0xfe, 0]);
    let parameter = write(directory, "linked/nested/parameter.inc", [0xfd, 0]);
    let library = write(directory, "linked/models.lib", [0xfc, 0]);
    let command = executable(
        directory,
        r#"
test "$#" = 2
test "$1" = 'literal ; $(not-a-command)'
test "$PWD" = "$(dirname "$2")"
cmp - "$2"
printf '%s\n' "$2" >> calls.log
case "$2" in
  */root.cir)
    printf '%s\n' 'transformed root' 'v1 in 0 2' '.include "linked/branch.inc"' '.lib "linked/models.lib" first' '.lib "linked/models.lib" second' '.op' '.dc v1 1 3 1' '.end'
    ;;
  */branch.inc) printf '%s\n' '.include "nested/parameter.inc"' ;;
  */parameter.inc) printf '%s\n' '.param rr=1k' ;;
  */models.lib) printf '%s\n' '.lib first' 'r1 in out {rr}' '.endl first' '.lib second' 'r2 out 0 {rr}' '.endl second' ;;
  *) exit 19 ;;
esac
"#,
    );
    let mut settings = config(command);
    settings
        .command
        .as_mut()
        .unwrap()
        .arguments
        .push("literal ; $(not-a-command)".into());
    assert!(Parser::new().parse_file(&root).is_err());
    let parsed = parse(&root, settings.clone(), SourceLimits::default()).unwrap();
    assert_eq!(parsed.netlist.title, "transformed root");
    assert_eq!(parsed.netlist.path, root);
    assert_eq!(
        parsed.netlist.params[0].location.path(),
        parameter.canonicalize().unwrap()
    );
    assert_eq!(
        parsed.netlist.devices[1].location.path(),
        library.canonicalize().unwrap()
    );
    assert_eq!(parsed.netlist.devices[1].location.line, 2);
    assert_eq!(parsed.netlist.devices[2].location.line, 5);
    for (log, count) in [
        ("calls.log", 1),
        ("linked/calls.log", 2),
        ("linked/nested/calls.log", 1),
    ] {
        let path = directory.join(log);
        assert_eq!(fs::read_to_string(&path).unwrap().lines().count(), count);
        fs::remove_file(path).unwrap();
    }
    let admission = SourceAdmission::new(settings, Cancellation::default());
    let output = directory.join("output.raw");
    run_with_admission(
        &root,
        &output,
        true,
        Some(RawFormat::Ascii),
        Some(&admission),
    )
    .unwrap();
    let raw = RawFile::load(output).unwrap();
    assert_eq!(raw.plots.len(), 2);
    let mut checked = 0;
    for plot in &raw.plots {
        let input = plot.plot.column("v(in)").unwrap();
        let output = plot.plot.column("v(out)").unwrap();
        for (input, output) in input.iter().zip(output) {
            assert!((output.re - input.re * 0.5).abs() < 1e-12);
            checked += 1;
        }
    }
    assert_eq!(checked, 4);
    let root_log = fs::read_to_string(directory.join("calls.log")).unwrap();
    let linked_log = fs::read_to_string(directory.join("linked/calls.log")).unwrap();
    let nested_log = fs::read_to_string(directory.join("linked/nested/calls.log")).unwrap();
    assert_eq!(
        root_log.lines().collect::<Vec<_>>(),
        [root.canonicalize().unwrap().to_str().unwrap()]
    );
    assert_eq!(
        linked_log.lines().collect::<Vec<_>>(),
        [
            branch.canonicalize().unwrap().to_str().unwrap(),
            library.canonicalize().unwrap().to_str().unwrap()
        ]
    );
    assert_eq!(
        nested_log.lines().collect::<Vec<_>>(),
        [parameter.canonicalize().unwrap().to_str().unwrap()]
    );
}

#[test]
fn aggregate_budget_charges_both_streams_and_repeated_section_visits() {
    let scratch = tempfile::tempdir().unwrap();
    let directory = scratch.path();
    let root_text = "title\n.lib shared.lib a\n.lib alias.lib b\n.op\n.end\n";
    let lib_text = ".lib a\nr1 a 0 1k\n.endl a\n.lib b\nr2 b 0 2k\n.endl b\n";
    let root = write(directory, "root.cir", root_text);
    let library = write(directory, "shared.lib", lib_text);
    std::os::unix::fs::symlink(library, directory.join("alias.lib")).unwrap();
    let command = executable(directory, "cat");
    let settings = config(command);
    let bytes = 2 * (root_text.len() + 2 * lib_text.len());
    let limits = SourceLimits {
        bytes,
        ..SourceLimits::default()
    };
    parse(&root, settings.clone(), limits).unwrap();
    let error = parse(
        &root,
        settings,
        SourceLimits {
            bytes: bytes - 1,
            ..limits
        },
    )
    .unwrap_err();
    assert!(error.to_string().contains("byte work limit"), "{error}");
}

#[test]
fn source_failures_are_positioned_and_leave_simulation_output_untouched() {
    let scratch = tempfile::tempdir().unwrap();
    let directory = scratch.path();
    let root = write(
        directory,
        "root.cir",
        "title\n.include child.inc\n.op\n.end\n",
    );
    let child = write(directory, "child.inc", "r1 out 0 1k\n");
    let command = executable(
        directory,
        r#"
case "$1" in
  */child.inc) printf 'external command rejected input' >&2; exit 7 ;;
  *) cat ;;
esac
"#,
    );
    let settings = config(command);
    let output = write(directory, "existing.raw", b"previous output");
    let admission = SourceAdmission::new(settings, Cancellation::default());
    let error = run_with_admission(
        &root,
        &output,
        true,
        Some(RawFormat::Ascii),
        Some(&admission),
    )
    .unwrap_err()
    .to_string();
    assert!(error.contains("external command rejected input"), "{error}");
    assert!(error.contains(child.to_str().unwrap()), "{error}");
    assert!(error.contains("root.cir:2"), "{error}");
    assert_eq!(fs::read(output).unwrap(), b"previous output");
}

#[test]
fn process_and_input_limits_and_cancellation_are_enforced() {
    let scratch = tempfile::tempdir().unwrap();
    let directory = scratch.path();
    let root = write(directory, "root.cir", "title\nr1 a 0 1k\n.op\n.end\n");
    let command = executable(directory, "printf 'started' > invoked; cat");
    let mut settings = config(command);
    settings.input_bytes = 2;
    let error = parse(&root, settings.clone(), SourceLimits::default())
        .unwrap_err()
        .to_string();
    assert!(error.contains("input limit"), "{error}");
    assert!(!directory.join("invoked").exists());
    settings.input_bytes = AdmissionConfig::default().input_bytes;
    let cancelled = Cancellation::default();
    cancelled.cancel();
    let admission = SourceAdmission::new(settings.clone(), cancelled);
    let error = Parser::new()
        .parse_file_with_admission(&root, SourceLimits::default(), Some(&admission))
        .unwrap_err()
        .to_string();
    assert!(error.contains("cancelled"), "{error}");
    assert!(!directory.join("invoked").exists());
    settings.process.output_bytes = 2;
    let error = parse(&root, settings.clone(), SourceLimits::default())
        .unwrap_err()
        .to_string();
    assert!(error.contains("stdout limit"), "{error}");
    settings.command.as_mut().unwrap().executable = executable(directory, "printf '\\377'");
    settings.process = ProcessLimits::default();
    let error = parse(&root, settings.clone(), SourceLimits::default())
        .unwrap_err()
        .to_string();
    assert!(error.contains("not UTF-8"), "{error}");
    settings.command.as_mut().unwrap().executable =
        executable(directory, "printf 'diagnostic beyond limit' >&2; cat");
    settings.process.diagnostic_bytes = 2;
    let error = parse(&root, settings.clone(), SourceLimits::default())
        .unwrap_err()
        .to_string();
    assert!(error.contains("stderr limit"), "{error}");
    settings.command.as_mut().unwrap().executable = executable(directory, "sleep 10");
    settings.process = ProcessLimits {
        timeout: Duration::from_millis(50),
        ..ProcessLimits::default()
    };
    let error = parse(&root, settings, SourceLimits::default())
        .unwrap_err()
        .to_string();
    assert!(error.contains("deadline"), "{error}");
}

#[test]
fn cli_options_are_literal_and_require_an_absolute_executable() {
    let args = Args::parse([
        "parse",
        "--preprocess-arg",
        "--literal",
        "--preprocess-command",
        "/command path",
        "file.cir",
    ])
    .unwrap();
    assert_eq!(
        args.preprocess.executable.as_deref(),
        Some(Path::new("/command path"))
    );
    assert_eq!(args.preprocess.arguments, ["--literal"]);
    for arguments in [
        vec!["parse", "--preprocess-arg", "x", "file.cir"],
        vec!["parse", "--preprocess-command", "relative", "file.cir"],
        vec!["devices", "--preprocess-command", "/bin/cat"],
        vec![
            "parse",
            "--preprocess-command",
            "/bin/cat",
            "--preprocess-command",
            "/bin/cat",
            "file.cir",
        ],
    ] {
        assert!(Args::parse(arguments).is_err());
    }
}

#[test]
fn cli_routes_share_external_admission() {
    let scratch = tempfile::tempdir().unwrap();
    let directory = scratch.path();
    let deck = write(directory, "deck.cir", [0xff]);
    let command = executable(
        directory,
        r#"
test "$1" = '--literal'
printf '%s\n' 'command output' 'v1 in 0 2' 'r1 in 0 1k' '.op' '.end'
"#,
    );
    let binary =
        std::env::var_os("CARGO_BIN_EXE_spice-rs").expect("Cargo supplies the test executable");
    for action in ["summary", "cards", "tokens", "parse", "simulate"] {
        let mut process = std::process::Command::new(&binary);
        process
            .arg(action)
            .arg("--preprocess-command")
            .arg(&command)
            .args(["--preprocess-arg", "--literal"]);
        if action == "simulate" {
            process.arg("--output").arg(directory.join("out.raw"));
        }
        let output = process.arg(&deck).output().unwrap();
        assert!(
            output.status.success(),
            "{action}: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
    let command = executable(
        directory,
        r#"
printf '%s\n' 'control output' 'v1 in 0 2' 'r1 in 0 1k' '.control' 'op' 'print v(in)' '.endc' '.end'
"#,
    );
    let output = std::process::Command::new(binary)
        .arg("control")
        .arg("--preprocess-command")
        .arg(command)
        .arg(deck)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn configured_root_alias_resolves_links_from_command_origin() {
    let scratch = tempfile::tempdir().unwrap();
    let directory = scratch.path();
    let actual = write(
        directory,
        "original/root.cir",
        "title\n.include child.inc\n.op\n.end\n",
    );
    let child = write(directory, "original/child.inc", "r1 a 0 1k\n");
    let alias = directory.join("root-link.cir");
    std::os::unix::fs::symlink(actual, &alias).unwrap();
    let command = executable(directory, "cat");
    let parsed = parse(&alias, config(command), SourceLimits::default()).unwrap();
    assert_eq!(parsed.netlist.path, alias);
    assert_eq!(
        parsed.netlist.devices[0].location.path(),
        child.canonicalize().unwrap()
    );
}
