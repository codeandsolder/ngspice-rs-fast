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
    const fn new() -> Self {
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

    const fn exhausted(&self) -> bool {
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
    targets: Vec<(usize, Real)>,
    found: Option<Candidate>,
}

impl Search<'_, '_, '_> {
    fn seed_at(&self, direction: &Vector, radius: Real) -> Vec<Vector> {
        let width = self.seed[0].len();
        let mut state = self.seed.clone();
        let prefix = &direction.as_slice()[..2 * width];
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
        let factor = radius * reference;
        for (jet, delta) in state
            .iter_mut()
            .zip(direction.as_slice().chunks_exact(width))
        {
            for (x, &d) in jet.as_mut_slice().iter_mut().zip(delta) {
                *x = factor.mul_add(d, *x);
            }
        }
        state
    }

    // Measure the response of previously unrepresented physical equations,
    // relative to their own nonzero residuals. A tiny high-order residual
    // can determine a large physical derivative; user tolerances must not
    // determine the seed-search scale.
    fn response(&self, direction: &Vector, radius: Real) -> SpiceResult<Real> {
        let state = self.seed_at(direction, radius);
        if state.iter().any(|jet| !jet.is_finite()) {
            return Ok(Real::INFINITY);
        }
        let point = match self.local.evaluate(self.dae, &state) {
            Ok(point) => point,
            Err(error @ (SpiceError::Circuit { .. } | SpiceError::Numerical { .. })) => {
                if self.local.recovery.is_some_and(Budget::exhausted) {
                    return Err(error);
                }
                return Ok(Real::INFINITY);
            }
            Err(error) => return Err(error),
        };
        let width = self.seed[0].len();
        let mut response: Real = 0.;
        for &(row, base) in &self.targets {
            let (degree, coordinate) = (row / width, row % width);
            let multiplier = Real::from(
                u32::try_from(degree + 1).map_err(|_| failure("invalid recovery degree"))?,
            ) / self.local.scale;
            let residual = multiplier.mul_add(
                point.charge[degree + 1].as_slice()[coordinate],
                point.residual[degree].as_slice()[coordinate],
            );
            response = response.max((residual / base - 1.).abs());
        }
        Ok(response)
    }

    /// Locate a response comparable with the outstanding residual by
    /// bracketing in binary exponent space. This handles microscopic
    /// high-order coefficients and large physical roots using the same
    /// criterion, without assuming a voltage or derivative magnitude.
    fn radius(&self, direction: &Vector, sign: Real) -> SpiceResult<Option<Real>> {
        const LIMIT: i32 = 512;
        const LOW: Real = 0.25;
        const HIGH: Real = 4.;
        let evaluate = |exponent: i32| self.response(direction, sign * Real::from(exponent).exp2());
        let first = evaluate(0)?;
        if (LOW..=HIGH).contains(&first) {
            return Ok(Some(sign));
        }
        let orientation = if first < LOW { 1 } else { -1 };
        let mut previous = 0;
        let mut distance = 1;
        let mut bracket = loop {
            let exponent = orientation * distance;
            let response = evaluate(exponent)?;
            if (LOW..=HIGH).contains(&response) {
                return Ok(Some(sign * Real::from(exponent).exp2()));
            }
            if (response < LOW) != (first < LOW) {
                break (previous.min(exponent), previous.max(exponent));
            }
            if distance == LIMIT {
                return Ok(None);
            }
            previous = exponent;
            distance = (2 * distance).min(LIMIT);
        };
        while bracket.1 - bracket.0 > 1 {
            let middle = i32::midpoint(bracket.0, bracket.1);
            let response = evaluate(middle)?;
            if (LOW..=HIGH).contains(&response) {
                return Ok(Some(sign * Real::from(middle).exp2()));
            }
            if response < LOW {
                bracket.0 = middle;
            } else {
                bracket.1 = middle;
            }
        }
        // A steep response can cross the target band within one exponent.
        // Use the lower-radius end, where the residual is still finite.
        Ok(Some(sign * Real::from(bracket.0).exp2()))
    }

    fn probe(&mut self, direction: &Vector, radius: Real) -> SpiceResult<()> {
        let state = self.seed_at(direction, radius);
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
            if distinct(previous, &candidate, self.local.scale, self.tol) {
                return Err(failure(
                    "ambiguous nonlinear DAE right-limit germs: distinct physical states or rates satisfy the original equations",
                ));
            }
        } else {
            self.found = Some(candidate);
        }
        Ok(())
    }
}

fn distinct(a: &Candidate, b: &Candidate, scale: Real, tol: Tolerances) -> bool {
    // Compare in the same normalized coordinates used by the physical
    // equations. Propagate row-local backward error through each selected
    // tangent basis, rather than interpreting amplified cancellation
    // in higher derivatives as another nonlinear branch.
    let width = a.right.state.len();
    let values = |point: &RightLimit, index: usize| {
        if index < width {
            point.state.as_slice()[index]
        } else {
            scale * point.rate.as_slice()[index - width]
        }
    };
    (0..2 * width).any(|index| {
        let (x, y) = (values(&a.right, index), values(&b.right, index));
        let tolerance = tol
            .reltol
            .mul_add(x.abs().max(y.abs()), tol.vntol.min(tol.abstol));
        let error = |candidate: &Candidate| {
            candidate
                .uncertainty
                .as_ref()
                .map_or(0., |bound| bound.as_slice()[index])
        };
        (x - y).abs() > tolerance.max(error(a) + error(b))
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
        targets: array.residual[..array.physical_rows]
            .iter()
            .zip(&array.scale)
            .zip(&space.independent)
            .enumerate()
            .filter_map(|(row, ((&residual, scale), &independent))| {
                (!independent && residual.abs() > scale.roundoff()).then_some((row, residual))
            })
            .collect(),
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
    for direction in &directions {
        for sign in [1., -1.] {
            if let Some(radius) = search.radius(direction, sign)? {
                search.probe(direction, radius)?;
                search.probe(direction, radius * 0.5)?;
                search.probe(direction, radius * 2.)?;
            }
        }
    }
    search.found.map(|candidate| candidate.right).ok_or_else(|| failure("nonlinear DAE germ recovery could not resolve a singular tangent within its bounded search; required coordinates may be underdetermined or noncausal"))
}
