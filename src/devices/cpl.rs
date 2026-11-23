//! Coupled multiconductor line: immutable Maxwell geometry and modal planning.
//!
//! The entry point remains unported until the full DC and transient solver
//! is integrated. Do not expose an uncoupled approximation.

pub mod dc;
pub mod history;
pub mod modal;
pub mod model;
pub mod plan;
