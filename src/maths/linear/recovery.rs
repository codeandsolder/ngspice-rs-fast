//! Precision escalation for retained, unfolded physical operators.
//!
//! Use the same sparse factorization and both triangular solve orientations,
//! but keep the low word through assembly, elimination and residual updates.
//! No binary64 SIMD gather is permitted for the two-word scalar.
use std::sync::Arc;

use super::{OriginalStamp, OriginalStamps, PivotTolerances, numerical, order_fresh};
use crate::{
    maths::{
        Vector,
        equilibration::OriginalResidual,
        sparse13::{Scalar, Sparse},
        telemetry::PatternWork,
    },
    primitives::{ExactSum, SpiceResult, wide::Wide},
};

impl Scalar for Wide {
    const ZERO: Self = Self::ZERO;

    fn from_real(real: f64) -> Self {
        Self::from_real(real)
    }

    fn is_unit_real(self) -> bool {
        self.abs() == Self::ONE
    }

    fn mag(self) -> f64 {
        self.hi.abs()
    }

    fn real(self) -> f64 {
        self.hi
    }

    fn recip(self) -> Self {
        Self::ONE / self
    }

    fn mul(self, other: Self) -> Self {
        self * other
    }

    fn schur(self, _pivot: Self, _upper: Self, mult: Self, lower: Self, _normal: bool) -> Self {
        self - mult * lower
    }

    fn normal_multiplier(self) -> bool {
        self.finite() && self.hi.is_normal()
    }

    fn is_zero(self) -> bool {
        self == Self::ZERO
    }

    fn is_finite(self) -> bool {
        self.finite()
    }

    fn add_real(self, real: f64) -> Self {
        self + Self::from_real(real)
    }
}

/// The unfolded stamps and optional promoted factors belong to one numeric
/// frame. Even a rounded-input cache hit must invalidate both together.
#[derive(Debug, Clone)]
pub(super) struct RetainedOriginal {
    pub(super) stamps: Arc<OriginalStamps>,
    pivot: PivotTolerances,
    wide: Option<Arc<Sparse<Wide>>>,
}

impl RetainedOriginal {
    pub(super) fn new(stamps: OriginalStamps, pivot: PivotTolerances) -> Self {
        Self {
            stamps: Arc::new(stamps),
            pivot,
            wide: None,
        }
    }

    pub(super) const fn promoted(&self) -> bool {
        self.wide.is_some()
    }

    pub(super) fn solve_wide(
        &mut self,
        order: usize,
        rhs: &Vector,
        transpose: bool,
        work: &mut PatternWork,
    ) -> SpiceResult<Vector> {
        if self.wide.is_none() {
            let mut sparse = assemble_wide(order, &self.stamps)?;
            let factor = order_fresh("retained two-word sparse LU", &mut sparse, self.pivot, None);
            work.factor += sparse.last_attempt();
            factor?;
            self.wide = Some(Arc::new(sparse));
        }
        let sparse = self
            .wide
            .as_deref()
            .ok_or_else(|| numerical("retained sparse recovery", "missing promoted factors"))?;
        solve_checked(sparse, &self.stamps, rhs, transpose, work)
    }
}

fn relative_update(current: &mut [Wide], delta: &[Wide]) -> f64 {
    let mut relative = 0.0_f64;
    for (value, &change) in current.iter_mut().zip(delta) {
        let next = *value + change;
        if !next.finite() || !change.finite() {
            return f64::INFINITY;
        }
        let scale = value.hi.abs().max(next.hi.abs());
        if scale != 0. {
            relative = relative.max(change.hi.abs() / scale);
        }
        *value = next;
    }
    relative
}

fn assemble_wide(order: usize, stamps: &[OriginalStamp]) -> SpiceResult<Sparse<Wide>> {
    // Sorting indices rather than copying the physical stamps avoids one
    // superaccumulator per entry and preserves all duplicate contributions.
    let mut indices: Vec<_> = (0..stamps.len()).collect();
    indices.sort_unstable_by_key(|&index| {
        let (row, col, _) = stamps[index];
        (row, col)
    });
    let mut sparse = Sparse::new(order)?;
    for group in indices.chunk_by(|&a, &b| {
        let (r1, c1, _) = stamps[a];
        let (r2, c2, _) = stamps[b];
        (r1, c1) == (r2, c2)
    }) {
        let (row, col, _) = stamps[group[0]];
        let mut sum = ExactSum::new();
        for &index in group {
            sum.add_product(stamps[index].2, 1.);
        }
        let handle = sparse.get_element(row + 1, col + 1)?;
        sparse.set(handle, Wide::from_sum(sum));
    }
    Ok(sparse)
}

/// Check the correction in the promoted precision before narrowing the
/// result. Returning a finite but divergent iterate is not recovery.
fn solve_checked(
    sparse: &Sparse<Wide>,
    stamps: &[OriginalStamp],
    rhs: &Vector,
    transpose: bool,
    work: &mut PatternWork,
) -> SpiceResult<Vector> {
    let factors = sparse.factors()?;
    let original = OriginalResidual::real(rhs.len(), stamps, transpose)?;
    let wide_rhs: Vec<_> = rhs
        .as_slice()
        .iter()
        .copied()
        .map(Wide::from_real)
        .collect();
    let mut workspace = Vec::new();
    let mut solve = |rhs: &[Wide]| {
        let result = if transpose {
            factors.solve_transposed(rhs, &mut workspace)
        } else {
            factors.solve(rhs, &mut workspace)
        };
        work.record_rhs(&result);
        result
    };
    let mut current = solve(&wide_rhs)?;
    let mut residual = vec![Wide::ZERO; rhs.len()];
    for _ in 0..8 {
        original.wide_correction(rhs.as_slice(), &current, &mut residual)?;
        let delta = solve(&residual)?;
        if relative_update(&mut current, &delta) <= f64::EPSILON.sqrt() {
            return Ok(Vector::from(
                current.iter().map(|value| value.hi).collect::<Vec<_>>(),
            ));
        }
    }
    Err(numerical(
        "retained sparse recovery",
        "two-word refinement did not converge",
    ))
}
