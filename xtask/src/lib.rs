//! Workspace automation for the port.
//!
//! Run it as `cargo xtask <command>` (the alias is defined in
//! `.cargo/config.toml`).
//!
//! `golden capture/check` drive C to capture/reproduce comparison data;
//! `golden verify` runs the supported Rust analyses against committed data
//! without invoking C. `snapshots` checks/blesses token/AST dumps. Nothing in the
//! simulator depends on this crate; it is a development tool. See
//! `docs/port/VERIFICATION.md`.

mod commands;

pub use commands::run;
