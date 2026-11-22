//! Development-only C reference tooling, independent of the simulator.
//!
//! Fixture discovery and reference selection are independent of simulation.
//! Child process capture uses portable, cancellable pipe I/O. Analysis scheduling and typed rawfile comparison
//! belong to their simulator-aware callers.

pub mod fixtures;
pub mod measurement;
pub mod ngspice;

pub mod process;
