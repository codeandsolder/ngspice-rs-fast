//! Development-only C reference tooling, independent of the simulator.
//!
//! Fixture discovery, reference selection and child-process mechanics need
//! only the standard library. Analysis scheduling and typed rawfile comparison
//! belong to their simulator-aware callers.

pub mod fixtures;
pub mod measurement;
pub mod ngspice;
