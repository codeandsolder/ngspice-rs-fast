mod bsim3;
mod bsim4;
mod bsim4_fields;
mod ci;
mod compare;
mod golden;
mod ifparm;
mod snapshots;
mod tran;
mod verify;

use std::path::Path;
#[cfg(test)]
use std::path::PathBuf;
use std::process::ExitCode;

const USAGE: &str = "\
cargo xtask — workspace automation for the Rust ngspice port

USAGE:
    cargo xtask golden capture [OPTIONS]   run the C ngspice binary on every fixture
                                           and write conformance/golden/*.raw
    cargo xtask golden check [OPTIONS]     re-capture and report drift, without writing
    cargo xtask golden list                describe the committed fixtures
    cargo xtask golden verify [--netlist <NAME>]
                                           compare supported Rust results with C goldens
                                           (no C binary needed; never writes goldens)
    cargo xtask snapshots [--bless]        check (or with --bless, regenerate) the committed
                                           token/AST snapshots; Rust only, no C needed
    cargo xtask bsim3-schema --source <ngspice-checkout> [--check]
                                           regenerate/check BSIM3 interface metadata
    cargo xtask bsim4-schema --source <ngspice-checkout> [--check]
                                           regenerate/check BSIM4 interface metadata
    cargo xtask ci                         fmt --check, clippy -D warnings, test
    cargo xtask help

OPTIONS (golden capture/check):
    --ngspice <PATH>   the C ngspice binary to drive
                       (default: $NGSPICE_BIN, then build/src/ngspice, then PATH)
    --netlist <NAME>   only this fixture, e.g. --netlist rc_divider
    --verbose          echo each ngspice invocation
";

/// Resolve the fixture workspace from the runtime directory.
#[cfg(test)]
fn workspace_root() -> PathBuf {
    ngspice_conformance::fixtures::workspace_root().unwrap()
}

/// Run one workspace command against the supplied workspace root.
pub fn run(root: &Path, arguments: &[String]) -> ExitCode {
    let result = match arguments.first().map(String::as_str) {
        None | Some("help" | "-h" | "--help") => {
            println!("{USAGE}");
            Ok(())
        }
        Some("golden") => golden::main(root, &arguments[1..]),
        Some("snapshots") => snapshots::main(root, &arguments[1..]),
        Some("ci") => ci::main(root),
        Some("bsim3-schema") => bsim3::main(root, &arguments[1..]),
        Some("bsim4-schema") => bsim4::main(root, &arguments[1..]),
        Some(other) => Err(format!("unknown command '{other}'")),
    };

    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("xtask: {message}");
            if !message.starts_with("drift") && !message.starts_with("usage") {
                eprintln!();
                eprintln!("{USAGE}");
            }
            ExitCode::FAILURE
        }
    }
}
