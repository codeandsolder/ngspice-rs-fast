//! Workspace command-line entry point.

use std::process::ExitCode;

fn main() -> ExitCode {
    let arguments: Vec<String> = std::env::args().skip(1).collect();
    let root = match ngspice_conformance::fixtures::workspace_root() {
        Ok(root) => root,
        Err(error) => {
            eprintln!("xtask: {error}");
            return ExitCode::FAILURE;
        }
    };
    xtask::run(&root, &arguments)
}
