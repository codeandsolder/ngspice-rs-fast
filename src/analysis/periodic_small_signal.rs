//! Signed Floquet sideband coordinates for perturbations of a periodic orbit.

use super::harmonic_balance::HarmonicLattice;
use crate::primitives::{Real, SpiceError, SpiceResult};
use std::f64::consts::TAU;

fn failure(message: &str) -> SpiceError {
    SpiceError::Numerical {
        context: "periodic small signal".into(),
        message: message.into(),
    }
}

/// One complex perturbation coordinate, relative to an independent probe.
#[derive(Debug, Clone, PartialEq)]
pub struct Sideband {
    indices: Vec<i32>,
    frequency: Real,
}

impl Sideband {
    /// Integer offsets along the pump axes in lattice order.
    #[must_use]
    pub fn indices(&self) -> &[i32] {
        &self.indices
    }

    /// Signed physical frequency, in hertz.
    #[must_use]
    pub const fn frequency(&self) -> Real {
        self.frequency
    }
}

/// A single complex Floquet sector in ascending signed physical frequency.
/// The complex envelope is not folded onto positive frequencies: a real
/// perturbation also has the conjugate sector at the negative probe.
#[derive(Debug, Clone, PartialEq)]
pub struct SidebandPlan {
    probe: Real,
    bands: Vec<Sideband>,
    conjugate_overlap: bool,
}

fn coincides(a: Real, b: Real) -> bool {
    (a - b).abs() <= 8. * Real::EPSILON * a.abs().max(b.abs())
}

impl SidebandPlan {
    /// Derive signed sideband coordinates from the retained pump lattice.
    ///
    /// # Errors
    /// Invalid probe frequency, exceeded storage budget, unrepresentable
    /// shifted frequencies or collapsed distinct harmonic coordinates.
    pub fn new(lattice: &HarmonicLattice, probe: Real, max_sidebands: usize) -> SpiceResult<Self> {
        if !probe.is_finite() || lattice.coordinates() > max_sidebands {
            return Err(failure("invalid probe or sideband budget exceeded"));
        }
        let mut bands = Vec::new();
        bands
            .try_reserve_exact(lattice.coordinates())
            .map_err(|_| failure("cannot allocate sideband plan"))?;
        for harmonic in lattice.harmonics().iter().rev() {
            bands.push(Sideband {
                indices: harmonic.indices().iter().map(|&index| -index).collect(),
                frequency: probe - harmonic.frequency(),
            });
        }
        bands.push(Sideband {
            indices: vec![0; lattice.axes().len()],
            frequency: probe,
        });
        for harmonic in lattice.harmonics() {
            bands.push(Sideband {
                indices: harmonic.indices().to_vec(),
                frequency: probe + harmonic.frequency(),
            });
        }
        if bands.iter().any(|band| !(TAU * band.frequency).is_finite())
            || bands
                .windows(2)
                .any(|pair| coincides(pair[0].frequency, pair[1].frequency))
        {
            return Err(failure("nonfinite or indistinguishable shifted sidebands"));
        }
        // The reflected sector is already sorted by reversing the original
        // frequencies and changing their sign. A linear merge detects
        // phase-sensitive real-signal overlap without a quadratic search.
        let mut left = 0;
        let mut right = bands.len();
        let mut conjugate_overlap = false;
        while left < bands.len() && right > 0 {
            let a = bands[left].frequency;
            let b = -bands[right - 1].frequency;
            if coincides(a, b) {
                conjugate_overlap = true;
                break;
            }
            if a < b {
                left += 1;
            } else {
                right -= 1;
            }
        }
        Ok(Self {
            probe,
            bands,
            conjugate_overlap,
        })
    }

    /// Signed probe frequency, before pump offsets.
    #[must_use]
    pub const fn probe(&self) -> Real {
        self.probe
    }

    /// Ascending signed sideband coordinates.
    #[must_use]
    pub fn bands(&self) -> &[Sideband] {
        &self.bands
    }

    /// Whether the opposite-probe sector contains an overlapping frequency.
    /// Real-signal power or noise cannot be counted as independent sectors
    /// in this case; their coherent amplitudes must first be combined.
    #[must_use]
    pub const fn conjugate_overlap(&self) -> bool {
        self.conjugate_overlap
    }

    /// Locate the exact integer pump offset (not an approximate Hz key).
    #[must_use]
    pub fn locate(&self, indices: &[i32]) -> Option<usize> {
        self.bands.iter().position(|band| band.indices == indices)
    }
}
