//! Bounded, disposable nonlinear seed recovery for derivative arrays.
//! A probe is never an initial condition: it changes only the Newton
//! iterate, and the original physical equations and storage determine
//! whether the resulting right-hand germ is admissible.

use std::cell::Cell;

use super::{Array, Candidate, Constraint, CoreOutcome, LocalPoint, RightLimit, solve_core};
use crate::analysis::initial::{Tolerances, failure};
use crate::devices::physical_dae::PhysicalDae;
use crate::maths::Vector;
use crate::primitives::{Real, SpiceError, SpiceResult};

const MAX_DIRECTIONS: usize = 16;
const MAX_EVALUATIONS: usize = 4096;
const MAX_COEFFICIENT_VISITS: usize = 8_388_608;
const MAX_SEED_COEFFICIENTS: usize = 65_536;

pub(super) struct Budget {
    evaluations: Cell<usize>,
    coefficients: Cell<usize>,
}

impl Budget {
    fn new() -> Self {
        Self {
            evaluations: Cell::new(MAX_EVALUATIONS),
            coefficients: Cell::new(MAX_COEFFICIENT_VISITS),
        }
    }

    pub(super) fn charge(&self, state: &[Vector]) -> SpiceResult<()> {
        let cost = state[0].len().saturating_mul(state.len());
        let (Some(evaluations), Some(coefficients)) = (
            self.evaluations.get().checked_sub(1),
            self.coefficients.get().checked_sub(cost),
        ) else {
            self.evaluations.set(0);
            return Err(failure(
                "nonlinear DAE germ recovery exceeded its work budget",
            ));
        };
        self.evaluations.set(evaluations);
        self.coefficients.set(coefficients);
        Ok(())
    }

    fn exhausted(&self) -> bool {
        self.evaluations.get() == 0
    }
}

struct Search<'a, 'run, 'circuit> {
    dae: &'a PhysicalDae<'run, 'circuit>,
    seed: Vec<Vector>,
    initial: &'a [Constraint<'a>],
    local: LocalPoint<'a>,
    tol: Tolerances,
    iterations: usize,
    baseline_rank: usize,
    found: Option<RightLimit>,
}

impl Search<'_, '_, '_> {
    fn probe(&mut self, direction: &Vector, radius: Real) -> SpiceResult<()> {
        let width = self.seed[0].len();
        let mut state = self.seed.clone();
        let prefix = &direction.as_slice()[..2 * width];
        let magnitude = prefix.iter().fold(0_f64, |value, v| value.max(v.abs()));
        if magnitude == 0. {
            return Ok(());
        }
        // Relative scaling uses only coordinates moved by this direction.
        // The free direction is a seed, not a stamp or retained history.
        let reference = self
            .seed
            .iter()
            .take(2)
            .flat_map(Vector::as_slice)
            .zip(prefix)
            .filter(|(_, delta)| **delta != 0.)
            .fold(1_f64, |value, (x, _)| value.max(x.abs()));
        let factor = radius * reference / magnitude;
        for (jet, delta) in state
            .iter_mut()
            .zip(direction.as_slice().chunks_exact(width))
        {
            for (x, &d) in jet.as_mut_slice().iter_mut().zip(delta) {
                *x = factor.mul_add(d, *x);
            }
        }
        if state.iter().any(|jet| !jet.is_finite()) {
            return Ok(());
        }
        let result = solve_core(
            self.dae,
            state,
            self.initial,
            self.local,
            self.tol,
            self.iterations,
        );
        match result {
            Ok(CoreOutcome::Converged(candidate)) => self.admit(candidate),
            Ok(_) | Err(SpiceError::Numerical { .. } | SpiceError::Circuit { .. }) => {
                if self.local.recovery.is_some_and(Budget::exhausted) {
                    Err(failure(
                        "nonlinear DAE germ recovery exceeded its work budget",
                    ))
                } else {
                    Ok(())
                }
            }
            Err(error) => Err(error),
        }
    }

    fn admit(&mut self, candidate: Candidate) -> SpiceResult<()> {
        // At least one formerly free coordinate must become constrained
        // by the physical and mandatory equations themselves.
        if candidate.mandatory_rank <= self.baseline_rank {
            return Ok(());
        }
        if let Some(previous) = &self.found {
            if distinct(previous, &candidate.right, self.local.scale, self.tol) {
                return Err(failure(
                    "ambiguous nonlinear DAE right-limit germs: distinct physical states or rates satisfy the original equations",
                ));
            }
        } else {
            self.found = Some(candidate.right);
        }
        Ok(())
    }
}

fn distinct(a: &RightLimit, b: &RightLimit, scale: Real, tol: Tolerances) -> bool {
    // Compare dimensionless local Taylor coordinates, not raw derivatives.
    // The derivative-coordinate comparison includes a rounding floor
    // from its constant coefficient: generated differentiator signals
    // can be scaled by inverse seconds, even when observable voltages
    // and currents agree to binary64 precision. This is a comparison
    // between already verified roots, not a relaxed physical tolerance.
    a.state
        .as_slice()
        .iter()
        .zip(b.state.as_slice())
        .zip(a.rate.as_slice().iter().zip(b.rate.as_slice()))
        .any(|((&x, &y), (&dx, &dy))| {
            let value_scale = x.abs().max(y.abs());
            let absolute = tol.vntol.min(tol.abstol);
            let value_tolerance = tol.reltol.mul_add(value_scale, absolute);
            let (dx, dy) = (scale * dx, scale * dy);
            let rate_tolerance = tol
                .reltol
                .mul_add(dx.abs().max(dy.abs()), absolute)
                .max(32. * Real::EPSILON * value_scale);
            (x - y).abs() > value_tolerance || (dx - dy).abs() > rate_tolerance
        })
}

fn normalize(direction: &mut Vector, prefix: usize) {
    let magnitude = direction.as_slice()[..prefix]
        .iter()
        .fold(0_f64, |m, x| m.max(x.abs()));
    if magnitude != 0. {
        for value in direction.as_mut_slice() {
            *value /= magnitude;
        }
    }
}

/// Search symmetric nullspace directions at multiple scales. A successful
/// germ still needs full projected rank and original residual consistency.
/// This is a bounded local search, not global nonlinear root enumeration.
pub(super) fn recover(
    dae: &PhysicalDae<'_, '_>,
    seed: Vec<Vector>,
    initial: &[Constraint<'_>],
    local: LocalPoint<'_>,
    tol: Tolerances,
    iterations: usize,
) -> SpiceResult<RightLimit> {
    let width = seed[0].len();
    if width.saturating_mul(seed.len()) > MAX_SEED_COEFFICIENTS {
        return Err(failure(
            "nonlinear DAE germ recovery exceeded its seed size budget",
        ));
    }
    let budget = Budget::new();
    let local = LocalPoint {
        recovery: Some(&budget),
        ..local
    };
    let point = local.evaluate(dae, &seed)?;
    let array = Array::build(&point, &seed, local.scale, initial)?;
    let space = super::rank::probe_space(
        &array.matrix,
        &array.late,
        &array.optional,
        2 * width,
        MAX_DIRECTIONS,
    )?;
    let mut search = Search {
        dae,
        seed,
        initial,
        local,
        tol,
        iterations,
        baseline_rank: space.rank,
        found: None,
    };
    let mut directions = space.directions;
    for direction in &mut directions {
        normalize(direction, 2 * width);
    }
    // Coupled singular equations may have zero directional derivatives
    // along every coordinate axis separately (e.g. a product).
    let independent = directions.len();
    if independent > 1 {
        for alternating in [false, true] {
            let mut combined = Vector::zeros(directions[0].len());
            for (index, direction) in directions[..independent].iter().enumerate() {
                let sign: Real = if alternating && index % 2 != 0 {
                    -1.
                } else {
                    1.
                };
                for (sum, &x) in combined.as_mut_slice().iter_mut().zip(direction.as_slice()) {
                    *sum = sign.mul_add(x, *sum);
                }
            }
            normalize(&mut combined, 2 * width);
            directions.push(combined);
        }
    }
    for radius in [1., 0.125, 8.] {
        for direction in &directions {
            search.probe(direction, radius)?;
            search.probe(direction, -radius)?;
        }
    }
    search.found.ok_or_else(|| failure("nonlinear DAE germ recovery could not resolve a singular tangent within its bounded search; required coordinates may be underdetermined or noncausal"))
}
