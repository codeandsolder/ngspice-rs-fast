//! Forced large-signal harmonic balance in independent tone coordinates.
//!
//! Unlike shooting, the harmonic lattice does not integrate a startup
//! transient or assume that two close RF tones share a short period.

mod circuit;
mod plan;
mod projection;
mod solver;

pub use solver::{BalanceLimits, BalanceSolution, BalanceStats, HarmonicModel, Sample, solve};

pub(crate) use circuit::linearize_circuit_orbit;
pub use circuit::solve_circuit;
pub(super) use plan::HarmonicBalancePlan;
pub use plan::{Harmonic, HarmonicLattice, LatticeLimits, ToneAxis};
pub(crate) use solver::{OrbitSpectrum, linearize_orbit};

use crate::primitives::SpiceError;

fn failure(message: impl Into<String>) -> SpiceError {
    SpiceError::Numerical {
        context: "harmonic balance".into(),
        message: message.into(),
    }
}
