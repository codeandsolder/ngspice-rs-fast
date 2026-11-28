//! Precision escalation for retained, unfolded physical operators.
//!
//! Use the same sparse factorization and both triangular solve orientations,
//! but keep the low word through assembly, elimination and residual updates.
//! No binary64 SIMD gather is permitted for the two-word scalar.
use std::sync::Arc;

use super::{PivotTolerances, numerical, order_fresh};
use crate::maths::Triplet;
use crate::{
    maths::{
        Vector,
        equilibration::OriginalResidual,
        sparse13::{Scalar, Sparse},
        telemetry::PatternWork,
    },
    primitives::{ExactSum, SpiceResult, Wide},
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
    pub(super) stamps: Arc<Vec<Triplet>>,
    pivot: PivotTolerances,
    wide: Option<Arc<Sparse<Wide>>>,
}

impl RetainedOriginal {
    pub(super) fn new(stamps: Vec<Triplet>, pivot: PivotTolerances) -> Self {
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
        out: &mut Vector,
    ) -> SpiceResult<()> {
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
        solve_checked(sparse, &self.stamps, rhs, transpose, work, out)
    }
}

pub(super) fn operator_entries(stamps: &[Triplet]) -> SpiceResult<Vec<(usize, usize, f64)>> {
    let mut entries = Vec::new();
    entries.try_reserve_exact(stamps.len()).map_err(|error| {
        numerical(
            "retained sparse refinement",
            format!("cannot reserve operator entries: {error}"),
        )
    })?;
    entries.extend(stamps.iter().map(|t| (t.row, t.col, t.value)));
    Ok(entries)
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

fn assemble_wide(order: usize, stamps: &[Triplet]) -> SpiceResult<Sparse<Wide>> {
    // Sorting indices rather than copying the physical stamps avoids one
    // superaccumulator per entry and preserves all duplicate contributions.
    let mut indices: Vec<_> = (0..stamps.len()).collect();
    indices.sort_unstable_by_key(|&index| {
        let stamp = stamps[index];
        (stamp.row, stamp.col)
    });
    let mut sparse = Sparse::new(order)?;
    for group in indices.chunk_by(|&a, &b| {
        let a = stamps[a];
        let b = stamps[b];
        (a.row, a.col) == (b.row, b.col)
    }) {
        let Triplet { row, col, .. } = stamps[group[0]];
        let mut sum = ExactSum::new();
        for &index in group {
            sum.add_product(stamps[index].value, 1.);
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
    stamps: &[Triplet],
    rhs: &Vector,
    transpose: bool,
    work: &mut PatternWork,
    out: &mut Vector,
) -> SpiceResult<()> {
    let factors = sparse.factors()?;
    let entries = operator_entries(stamps)?;
    let original = OriginalResidual::real(rhs.len(), &entries, transpose)?;
    let wide_rhs: Vec<_> = rhs
        .as_slice()
        .iter()
        .copied()
        .map(Wide::from_real)
        .collect();
    let mut workspace = Vec::new();
    let mut solve = |rhs: &[Wide], output: &mut Vec<Wide>| {
        let result = factors.solve_into(rhs, &mut workspace, output, transpose);
        work.record_rhs(&result);
        result
    };
    let mut current = Vec::new();
    solve(&wide_rhs, &mut current)?;
    let mut residual = vec![Wide::ZERO; rhs.len()];
    let mut delta = Vec::new();
    for _ in 0..8 {
        original.wide_correction(rhs.as_slice(), &current, &mut residual)?;
        solve(&residual, &mut delta)?;
        if relative_update(&mut current, &delta) <= f64::EPSILON.sqrt() {
            let storage = out.overwrite_storage();
            storage.clear();
            storage.extend(current.iter().map(|value| value.hi));
            return Ok(());
        }
    }
    Err(numerical(
        "retained sparse recovery",
        "two-word refinement did not converge",
    ))
}
