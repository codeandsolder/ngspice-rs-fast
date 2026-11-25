//! Process-level success/failure contract; deliberately unavailable C binary.
use std::process::{Command, Output};

fn verify(args: &[&str]) -> Output {
    Command::new(
        std::env::var_os("CARGO_BIN_EXE_xtask").expect("Cargo supplies the test executable"),
    )
    .args(["golden", "verify"])
    .args(args)
    .env("NGSPICE_BIN", "/definitely/unavailable/ngspice")
    .output()
    .expect("run xtask")
}

#[test]
fn default_and_selected_verification_need_no_c_binary() {
    let output = verify(&[]);
    assert!(output.status.success(), "{output:?}");
    let text = String::from_utf8(output.stdout).unwrap();
    assert!(text.contains("220 verified fixture(s), 0 unsupported fixture(s), 0 failure(s)"));
    // The default run above already verifies every fixture. Selection is
    // checked on a few representatives (exact name, case-insensitive name with
    // extension, a multi-analysis batch deck, an initial-condition deck), so
    // new fixtures do not need to be added here.
    for name in [
        "rc_divider",
        "RC_LOWPASS_AC.cir",
        "multi_analysis_rc",
        "m7_ic_latch_nodeset_op",
    ] {
        let output = verify(&["--netlist", name]);
        assert!(output.status.success(), "{output:?}");
        assert!(
            String::from_utf8(output.stdout)
                .unwrap()
                .contains("1 verified fixture(s), 0 unsupported fixture(s)")
        );
    }
}

#[test]
fn explicit_bad_requests_and_usage_errors_exit_nonzero() {
    // No committed fixture is excluded any more, so an unknown name is the
    // remaining "not verifiable" case, and it must never pass silently.
    for args in [
        vec!["--netlist", "unknown"],
        vec!["--netlist"],
        vec!["--ngspice", "ignored"],
        vec!["--netlist", "rc_divider", "--netlist", "rlc_series"],
    ] {
        assert!(!verify(&args).status.success(), "{args:?}");
    }
    let output = verify(&["--netlist", "unknown"]);
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(stderr.contains("no fixture named 'unknown'"), "{stderr}");
}

#[test]
fn workspace_is_selected_from_runtime_directory() {
    let root = ngspice_conformance::fixtures::workspace_root().unwrap();
    let binary = std::env::var_os("CARGO_BIN_EXE_xtask").unwrap();
    let output = Command::new(&binary)
        .current_dir(root.join("conformance/netlists"))
        .args(["golden", "verify", "--netlist", "rc_divider"])
        .output()
        .unwrap();
    assert!(output.status.success(), "{output:?}");
    let outside = std::env::temp_dir();
    if ngspice_conformance::fixtures::workspace_root_from(&outside).is_none() {
        let output = Command::new(binary)
            .current_dir(outside)
            .args(["golden", "verify", "--netlist", "rc_divider"])
            .output()
            .unwrap();
        assert!(!output.status.success(), "{output:?}");
        assert!(
            String::from_utf8_lossy(&output.stderr).contains("no ngspice conformance workspace")
        );
    }
}

#[test]
fn relocated_executable_uses_only_the_selected_checkout() {
    use std::{fs, path::PathBuf};

    let original = ngspice_conformance::fixtures::workspace_root().unwrap();
    let scratch = ngspice_conformance::ngspice::Scratch::new("relocated-xtask").unwrap();
    let root = scratch.path().join("relocated checkout");
    let nested = root.join("conformance/netlists");
    fs::create_dir_all(&nested).unwrap();
    fs::create_dir_all(root.join("conformance/golden")).unwrap();
    fs::create_dir_all(root.join("xtask")).unwrap();
    for relative in [
        "Cargo.toml",
        "xtask/Cargo.toml",
        "conformance/netlists/rc_divider.cir",
    ] {
        fs::copy(original.join(relative), root.join(relative)).unwrap();
    }

    // Run the already-built binary from a different path as well as a
    // different checkout. Neither location is known at compile time.
    let built = PathBuf::from(std::env::var_os("CARGO_BIN_EXE_xtask").unwrap());
    let binary = scratch.path().join(built.file_name().unwrap());
    fs::copy(built, &binary).unwrap();
    let verify_relocated = || {
        Command::new(&binary)
            .current_dir(&nested)
            .args(["golden", "verify", "--netlist", "rc_divider"])
            .env("NGSPICE_BIN", "/definitely/unavailable/ngspice")
            .output()
            .unwrap()
    };

    // The original checkout has a valid golden, but this checkout does not.
    // A compile-time root fallback would incorrectly make this succeed.
    let missing = verify_relocated();
    assert!(!missing.status.success(), "{missing:?}");
    assert!(
        String::from_utf8_lossy(&missing.stderr).contains("rc_divider"),
        "{missing:?}"
    );

    let relative = "conformance/golden/rc_divider.raw";
    fs::copy(original.join(relative), root.join(relative)).unwrap();
    let present = verify_relocated();
    assert!(present.status.success(), "{present:?}");
    assert!(
        String::from_utf8_lossy(&present.stdout)
            .contains("1 verified fixture(s), 0 unsupported fixture(s)")
    );
}
