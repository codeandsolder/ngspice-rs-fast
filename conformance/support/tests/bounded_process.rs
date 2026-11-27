//! Exercise the actual runner with a portable child test executable.
use ngspice_conformance::ngspice::Scratch;
use ngspice_conformance::process::{Failure, ProcessLimits, run};
use std::fs;
use std::io::{self, Write};
use std::time::{Duration, Instant};

const CHILD_ARGS: &[&str] = &[
    "--ignored",
    "--exact",
    "fixture_child",
    "--nocapture",
    "--quiet",
];

#[test]
#[ignore = "spawned by the runner tests"]
fn fixture_child() {
    let Ok(mode) = fs::read_to_string("fixture-mode") else {
        // Direct --include-ignored runs do not have a child fixture directory.
        return;
    };
    match mode.as_str() {
        "nonzero" => {
            io::stdout().write_all(b"stdout\xff").unwrap();
            io::stdout().flush().unwrap();
            io::stderr().write_all(b"stderr\xfe").unwrap();
            io::stderr().flush().unwrap();
            std::process::exit(7);
        }
        "flood" => {
            let mut out = io::stdout().lock();
            let mut err = io::stderr().lock();
            loop {
                out.write_all(&[b'o'; 4096]).unwrap();
                out.flush().unwrap();
                err.write_all(&[b'e'; 4096]).unwrap();
                err.flush().unwrap();
            }
        }
        "sleep" => {
            fs::write("partial.raw", b"partial rawfile").unwrap();
            io::stdout().write_all(b"started\n").unwrap();
            io::stdout().flush().unwrap();
            std::thread::sleep(Duration::from_secs(30));
            fs::write("finished", b"unexpected").unwrap();
        }
        _ => panic!("unknown fixture mode {mode}"),
    }
}

fn fixture(mode: &str) -> Scratch {
    let scratch = Scratch::new("bounded-process").unwrap();
    fs::write(scratch.path().join("fixture-mode"), mode).unwrap();
    scratch
}

#[test]
fn exact_budget_and_nonzero_status_preserve_raw_bytes() {
    let scratch = fixture("nonzero");
    let exe = std::env::current_exe().unwrap();
    let output = run(&exe, scratch.path(), CHILD_ARGS, ProcessLimits::VERSION).unwrap();
    assert_eq!(output.status.code(), Some(7));
    assert!(output.stdout.ends_with(b"stdout\xff"));
    assert!(output.stderr.ends_with(b"stderr\xfe"));
    let size = output.stdout.len() + output.stderr.len();
    let limits = ProcessLimits {
        output_bytes: size,
        ..ProcessLimits::VERSION
    };
    let exact = run(&exe, scratch.path(), CHILD_ARGS, limits).unwrap();
    assert_eq!(exact.status.code(), Some(7));
    let error = run(
        &exe,
        scratch.path(),
        CHILD_ARGS,
        ProcessLimits {
            output_bytes: size - 1,
            ..limits
        },
    )
    .unwrap_err();
    assert_eq!(error.failure, Failure::OutputLimit);
    assert_eq!(error.stdout.len() + error.stderr.len(), size - 1);
    assert!(error.cleanup_error.is_none(), "{error}");
    assert!(
        output
            .require_success("fixture")
            .unwrap_err()
            .contains("stderr")
    );
}

#[test]
fn interleaved_unbounded_output_is_stopped_at_combined_limit() {
    let scratch = fixture("flood");
    let limits = ProcessLimits {
        timeout: Duration::from_secs(10),
        output_bytes: 128 * 1024,
    };
    let error = run(
        std::env::current_exe().unwrap(),
        scratch.path(),
        CHILD_ARGS,
        limits,
    )
    .unwrap_err();
    assert_eq!(error.failure, Failure::OutputLimit);
    assert_eq!(error.stdout.len() + error.stderr.len(), limits.output_bytes);
    assert!(error.stdout.contains(&b'o') && error.stderr.contains(&b'e'));
    assert!(error.cleanup_error.is_none(), "{error}");
}

#[test]
fn timeout_retains_partial_evidence_and_reaps_child() {
    let scratch = fixture("sleep");
    let started = Instant::now();
    let error = run(
        std::env::current_exe().unwrap(),
        scratch.path(),
        CHILD_ARGS,
        ProcessLimits {
            timeout: Duration::from_secs(2),
            ..ProcessLimits::VERSION
        },
    )
    .unwrap_err();
    assert_eq!(error.failure, Failure::Timeout);
    assert!(started.elapsed() < Duration::from_secs(15));
    assert!(error.cleanup_error.is_none(), "{error}");
    assert!(String::from_utf8_lossy(&error.stdout).contains("started"));
    assert_eq!(
        fs::read(scratch.path().join("partial.raw")).unwrap(),
        b"partial rawfile"
    );
    assert!(!scratch.path().join("finished").exists());
}

#[test]
fn missing_executable_is_launch_failure() {
    let scratch = fixture("nonzero");
    let error = run(
        scratch.path().join("nonexistent"),
        scratch.path(),
        &["--version"],
        ProcessLimits::VERSION,
    )
    .unwrap_err();
    assert_eq!(error.failure, Failure::Launch);
    assert!(error.cleanup_error.is_none());
    assert!(error.stdout.is_empty() && error.stderr.is_empty());
}

#[cfg(unix)]
#[test]
fn deadline_covers_exit_after_both_streams_close() {
    let scratch = fixture("nonzero");
    let error = run(
        "/bin/sh",
        scratch.path(),
        &["-c", "exec 1>&- 2>&-; exec sleep 30"],
        ProcessLimits {
            timeout: Duration::from_millis(200),
            ..ProcessLimits::VERSION
        },
    )
    .unwrap_err();
    assert_eq!(error.failure, Failure::Timeout);
    assert!(error.cleanup_error.is_none(), "{error}");
}
