//! Complex sparse LU for AC through the SPARSE 1.3 port. Scalar conversion
//! happens only at this boundary.
//!
//! C: `SMPcLUfac`/`SMPcReorder`/`SMPcSolve`/`SMPcaSolve` (`spsmp.c`). As in
//! [`crate::maths::linear`], there is no condition estimate or residual check;
//! singular pivots and non-finite solutions are errors.
use std::sync::Arc;

use crate::maths::linear::{
    ComplexFactors, PivotTolerances, check_status, numerical, order_fresh, square,
};
use crate::maths::sparse13::{Sparse, Status};
use crate::maths::{Pattern, SparseMatrix};
use crate::primitives::{Complex, SpiceResult};

/// Capacitive contributions grouped at a single matrix element.
///
/// Contributions of `E` (a range of its unfolded triplets, all at one
/// position) that one device adds to an element as `scale * (x1 + x2 + ...)`,
/// each `x` being `cap * CKTomega`: `mos1acld.c` adds `xgd + xgs + xgb`,
/// `bjtacld.c` adds `m * (xcpi + xcmu + xcmcb)`.
#[derive(Debug, Clone, PartialEq)]
pub struct DynamicGroup {
    /// The contributions it replaces, as indices into `E`'s triplets (which
    /// hold `scale * term`, zero terms dropped).
    pub range: std::ops::Range<usize>,
    /// The unscaled terms (capacitances), in C's order.
    pub terms: Vec<f64>,
    /// The factor C applies to the sum (the instance multiplier `m`).
    pub scale: f64,
    /// The divisor C applies after scaling (`indacld.c`'s
    /// `CKTomega * INDinduct / m`); `1` for none.
    pub divisor: f64,
    /// The terms are imaginary parts that already include `CKTomega`
    /// (`jfet2acld.c`'s `PSacload` outputs), so they are summed unscaled.
    /// Only a load reassembled at every frequency may use them
    /// ([`crate::devices::Device::small_signal_depends_on_frequency`]).
    pub prescaled: bool,
}

/// `omega` times each of `e`'s contributions in stamp order, except that
/// the contributions of one of `groups` are scaled one by one, summed and
/// multiplied by the group's scale into a single value first
/// ([`DynamicGroup`]).
pub(crate) fn dynamic_terms(
    e: &SparseMatrix,
    groups: &[DynamicGroup],
    omega: f64,
) -> Vec<(usize, usize, f64)> {
    iter_dynamic_terms(e, groups, omega).collect()
}

fn iter_dynamic_terms<'a>(
    e: &'a SparseMatrix,
    groups: &'a [DynamicGroup],
    omega: f64,
) -> impl Iterator<Item = (usize, usize, f64)> + 'a {
    let triplets = e.triplets();
    let mut groups = groups.iter().peekable();
    let mut i = 0;
    std::iter::from_fn(move || {
        let t = triplets.get(i)?;
        while groups
            .next_if(|g| g.range.end <= i || g.range.is_empty())
            .is_some()
        {}
        let value = match groups.peek() {
            Some(g) if g.range.start == i && g.range.end <= triplets.len() => {
                let mut terms = g
                    .terms
                    .iter()
                    .map(|term| if g.prescaled { *term } else { omega * term });
                let mut sum = terms.next().unwrap_or(0.);
                for term in terms {
                    sum += term;
                }
                i = g.range.end;
                g.scale * sum / g.divisor
            }
            _ => {
                i += 1;
                omega * t.value
            }
        };
        Some((t.row, t.col, value))
    })
}

/// An assembled circuit matrix stores only coefficients in element-ID
/// order; standalone matrices retain their sorted coordinate triples.
#[derive(Debug, Clone)]
enum Assembly {
    Bound {
        pattern: Arc<Pattern>,
        values: Vec<Complex>,
    },
    Standalone(Vec<(usize, usize, Complex)>),
}

/// The coordinate order of a factor snapshot, shared without duplicating a
/// circuit pattern's positions. Standalone reuse can still compare against
/// either kind of previous snapshot.
#[derive(Debug, Clone)]
enum ComplexOrder {
    Bound(Arc<Pattern>),
    Standalone(Arc<Vec<(usize, usize)>>),
}

impl ComplexOrder {
    fn matches(&self, entries: &[(usize, usize, Complex)]) -> bool {
        let positions = entries.iter().map(|&(r, c, _)| (r, c));
        match self {
            Self::Bound(pattern) => (0..pattern.len())
                .map(|i| pattern.position(crate::maths::sparse::element_id(i)))
                .eq(positions),
            Self::Standalone(previous) => previous.iter().copied().eq(positions),
        }
    }
}

/// Complex assembled sparse matrix, independent of real stamping storage.
#[derive(Debug, Clone)]
pub struct ComplexMatrix {
    n: usize,
    assembly: Assembly,
    pivot: PivotTolerances,
    /// The circuit unknowns' names (from the real operator), for diagnostics.
    names: Option<crate::maths::UnknownNames>,
}
impl ComplexMatrix {
    /// Assembles `A + j omega E`, preserving cancellations and checking
    /// overflow. The pivot tolerances are `a`'s.
    /// # Errors
    /// Invalid dimensions, frequency or coefficients.
    pub fn from_operators(a: &SparseMatrix, e: &SparseMatrix, omega: f64) -> SpiceResult<Self> {
        Self::from_operators_grouped(a, e, &[], omega)
    }

    /// [`Self::from_operators`] with `groups` of `e`'s stamp-order
    /// contributions (see [`dynamic_terms`]) each added as one sum.
    /// # Errors
    /// As [`Self::from_operators`].
    pub fn from_operators_grouped(
        a: &SparseMatrix,
        e: &SparseMatrix,
        groups: &[DynamicGroup],
        omega: f64,
    ) -> SpiceResult<Self> {
        square(a.rows(), a.cols())?;
        if e.rows() != a.rows() || e.cols() != a.cols() || !omega.is_finite() || omega < 0. {
            return Err(numerical(
                "complex assembly",
                "operator dimensions/frequency invalid",
            ));
        }
        let mut assembled = std::collections::BTreeMap::new();
        let real = a
            .triplets()
            .iter()
            .map(|t| (t.row, t.col, Complex::real(t.value)));
        let imaginary = iter_dynamic_terms(e, groups, omega)
            .map(|(row, col, v)| (row, col, Complex::imaginary(v)));
        {
            for (row, col, v) in real.chain(imaginary) {
                if !v.is_finite() {
                    return Err(numerical(
                        "complex assembly",
                        "non-finite coefficient/overflow",
                    ));
                }
                let entry = assembled.entry((row, col)).or_insert(Complex::ZERO);
                *entry = *entry + v;
                if !entry.is_finite() {
                    return Err(numerical("complex assembly", "duplicate sum overflow"));
                }
            }
        }
        Ok(Self {
            n: a.rows(),
            assembly: Assembly::Standalone(
                assembled
                    .into_iter()
                    .filter(|(_, v)| *v != Complex::ZERO)
                    .map(|((r, c), v)| (r, c, v))
                    .collect(),
            ),
            pivot: a.pivot_tolerances(),
            names: a.unknown_names().cloned(),
        })
    }

    /// `CKTacLoad` into a circuit pattern: every element of `pattern` (zeros
    /// included, as ngspice allocates them at setup) holds the sum of `a`'s
    /// contributions as its real part and of `omega` times `e`'s as its
    /// imaginary part, each summed in stamp order (`omega` may be negative, as
    /// `CKTomega` is for `.disto`'s `f1 - f2`). `None` when a stamp lies
    /// outside the pattern.
    ///
    /// Factoring it ([`Self::factorize`]) shares the circuit's SPARSE frame
    /// and pivot order with its real solves, as `NIacIter` does.
    /// # Errors
    /// Invalid dimensions, frequency or coefficients.
    pub fn from_operators_on(
        pattern: &Arc<crate::maths::Pattern>,
        a: &SparseMatrix,
        e: &SparseMatrix,
        omega: f64,
    ) -> SpiceResult<Option<Self>> {
        Self::from_operators_grouped_on(pattern, a, e, &[], omega)
    }

    /// [`Self::from_operators_on`] with `groups` of `e`'s stamp-order
    /// contributions (see [`dynamic_terms`]) each added as one sum.
    /// # Errors
    /// As [`Self::from_operators_on`].
    pub fn from_operators_grouped_on(
        pattern: &Arc<crate::maths::Pattern>,
        a: &SparseMatrix,
        e: &SparseMatrix,
        groups: &[DynamicGroup],
        omega: f64,
    ) -> SpiceResult<Option<Self>> {
        let n = pattern.order();
        if a.rows() != n || a.cols() != n || e.rows() != n || e.cols() != n {
            return Err(numerical("complex assembly", "operator dimensions invalid"));
        }
        // `CKTomega` may be negative: `.disto` loads at `f1 - f2`.
        if !omega.is_finite() {
            return Err(numerical("complex assembly", "frequency invalid"));
        }
        let mut values = vec![Complex::ZERO; pattern.len()];
        // The bound real operator already has one accumulated value for
        // each element. Skip the row-major view and coordinate lookups if
        // it is bound to this exact pattern. Zero (including -0) was
        // omitted by the old triplet view, so keep the zero initialization.
        if let Some((source, real)) = a
            .bound_values()
            .filter(|(source, _)| Arc::ptr_eq(source, pattern))
        {
            debug_assert_eq!(source.len(), values.len());
            for (value, &re) in values.iter_mut().zip(real) {
                if !re.is_finite() {
                    return Err(numerical(
                        "complex assembly",
                        "non-finite coefficient/overflow",
                    ));
                }
                if re != 0.0 {
                    value.re = re;
                }
            }
        } else {
            for t in a.triplets() {
                let Some(id) = pattern.find(t.row, t.col) else {
                    return Ok(None);
                };
                let value = &mut values[id.index()];
                value.re += t.value;
                if !value.is_finite() {
                    return Err(numerical(
                        "complex assembly",
                        "non-finite coefficient/overflow",
                    ));
                }
            }
        }
        for (row, col, v) in iter_dynamic_terms(e, groups, omega) {
            let Some(id) = pattern.find(row, col) else {
                return Ok(None);
            };
            let value = &mut values[id.index()];
            value.im += v;
            if !value.is_finite() {
                return Err(numerical(
                    "complex assembly",
                    "non-finite coefficient/overflow",
                ));
            }
        }
        Ok(Some(Self {
            n,
            assembly: Assembly::Bound {
                pattern: Arc::clone(pattern),
                values,
            },
            pivot: a.pivot_tolerances(),
            names: a.unknown_names().cloned(),
        }))
    }

    /// Orders and factors a brand-new SPARSE matrix whose elements are
    /// created in `entries` order (for a bound matrix, the circuit's element
    /// order), with `SMPcReorder` alone: no `spMNA_Preorder` and no pivot
    /// order inherited from a real factorization. This is the matrix
    /// `sens_sens()` (`cktsens.c`) factors at each frequency, after
    /// `CKTunsetup`/`CKTsetup` rebuilt the circuit matrix and `NIacIter`
    /// reordered it.
    /// # Errors
    /// Structurally or numerically singular matrix.
    pub fn factorize_new_matrix(&self) -> SpiceResult<ComplexLu> {
        let (mut sparse, handles) = self.fresh_frame();
        if !sparse.fully_translated() {
            // A circuit's matrix follows C (see `maths::linear`).
            if self.names.is_none() && matches!(self.assembly, Assembly::Standalone(_)) {
                return Err(numerical(
                    "complex sparse LU",
                    "singular matrix: an unknown has no matrix entries",
                ));
            }
            crate::maths::linear::warn_untranslated(&sparse, self.names.as_ref());
        }
        let status = sparse.order_and_factor(self.pivot.relative, self.pivot.absolute);
        check_status("complex sparse LU", status, &sparse, self.names.as_ref())?;
        Ok(ComplexLu {
            n: self.n,
            order: self.order_key(),
            factors: ComplexFactors {
                handles: Arc::new(handles),
                sparse: Arc::new(sparse),
            },
        })
    }

    /// Orders and factors (`SMPpreOrder` + `SMPcReorder`).
    /// # Errors
    /// Structurally or numerically singular matrix.
    pub fn factorize(&self) -> SpiceResult<ComplexLu> {
        if let Assembly::Bound { pattern, values } = &self.assembly {
            let factors = crate::maths::linear::factor_shared_complex(
                pattern,
                values,
                self.pivot,
                self.names.as_ref(),
            )?;
            return Ok(ComplexLu {
                n: self.n,
                order: self.order_key(),
                factors,
            });
        }
        let (mut sparse, handles) = self.fresh_frame();
        order_fresh(
            "complex sparse LU",
            &mut sparse,
            self.pivot,
            self.names.as_ref(),
            self.names.is_some(),
        )?;
        Ok(ComplexLu {
            n: self.n,
            order: self.order_key(),
            factors: ComplexFactors {
                handles: Arc::new(handles),
                sparse: Arc::new(sparse),
            },
        })
    }

    fn fresh_frame(&self) -> (Sparse<Complex>, Vec<usize>) {
        let mut sparse = Sparse::new(self.n);
        let mut insert = |r, c, v| {
            let handle = sparse.get_element(r + 1, c + 1);
            sparse.set(handle, v);
            handle
        };
        let handles = match &self.assembly {
            Assembly::Bound { pattern, values } => values
                .iter()
                .enumerate()
                .map(|(index, &v)| {
                    let (r, c) = pattern.position(crate::maths::sparse::element_id(index));
                    insert(r, c, v)
                })
                .collect(),
            Assembly::Standalone(entries) => {
                entries.iter().map(|&(r, c, v)| insert(r, c, v)).collect()
            }
        };
        (sparse, handles)
    }

    fn order_key(&self) -> ComplexOrder {
        match &self.assembly {
            Assembly::Bound { pattern, .. } => ComplexOrder::Bound(Arc::clone(pattern)),
            Assembly::Standalone(entries) => ComplexOrder::Standalone(Arc::new(
                entries.iter().map(|&(r, c, _)| (r, c)).collect(),
            )),
        }
    }

    /// Refactors with the pivot order of `previous` when the pattern is
    /// identical (`SMPcLUfac`, `spFactor`), reordering after a zero pivot
    /// (`SMPcReorder`) as `NIacIter` does; otherwise orders afresh.
    /// # Errors
    /// Structurally or numerically singular matrix.
    pub fn factorize_reusing(&self, previous: &ComplexLu) -> SpiceResult<ComplexLu> {
        let Assembly::Standalone(entries) = &self.assembly else {
            // The circuit pattern carries the pivot order itself.
            return self.factorize();
        };
        if previous.n != self.n || !previous.order.matches(entries) {
            return self.factorize();
        }
        let mut sparse = (*previous.factors.sparse).clone();
        let load = |sparse: &mut Sparse<Complex>| {
            sparse.clear();
            for (handle, &(_, _, v)) in previous.factors.handles.iter().zip(entries) {
                sparse.set(*handle, v);
            }
        };
        load(&mut sparse);
        let mut status = sparse.factor();
        if status == Status::ZeroDiag {
            load(&mut sparse);
            status = sparse.order_and_factor(self.pivot.relative, self.pivot.absolute);
        }
        check_status("complex sparse LU", status, &sparse, self.names.as_ref())?;
        Ok(ComplexLu {
            n: self.n,
            order: previous.order.clone(),
            factors: ComplexFactors {
                handles: Arc::clone(&previous.factors.handles),
                sparse: Arc::new(sparse),
            },
        })
    }
}

/// Owned complex factors; subsequent real operator mutations cannot stale them.
#[derive(Debug)]
pub struct ComplexLu {
    n: usize,
    order: ComplexOrder,
    factors: ComplexFactors,
}
impl ComplexLu {
    /// Solves `A x = b` (`SMPcSolve`).
    /// # Errors
    /// Wrong RHS size, non-finite RHS or solution.
    pub fn solve(&self, rhs: &[Complex]) -> SpiceResult<Vec<Complex>> {
        self.solve_with(rhs, false)
    }

    /// Solves the transposed (not conjugated) system `A^T x = b` with the same
    /// factors. This is the adjoint solve of `.noise` (C `SMPcaSolve` in
    /// `NInzIter`).
    /// # Errors
    /// Wrong RHS size, non-finite RHS or solution.
    pub fn solve_transposed(&self, rhs: &[Complex]) -> SpiceResult<Vec<Complex>> {
        self.solve_with(rhs, true)
    }

    fn solve_with(&self, rhs: &[Complex], transposed: bool) -> SpiceResult<Vec<Complex>> {
        if rhs.len() != self.n || rhs.iter().any(|v| !v.is_finite()) {
            return Err(numerical("complex solve", "invalid RHS"));
        }
        let x = if transposed {
            self.factors.sparse.solve_transposed(rhs)
        } else {
            self.factors.sparse.solve(rhs)
        };
        if x.iter().any(|v| !v.is_finite()) {
            return Err(numerical(
                "complex solve",
                "non-finite solution/singular system",
            ));
        }
        Ok(x)
    }

    /// `spDeterminant`: `(mantissa, decimal exponent)`, with C's
    /// `1 <= max(|re|, |im|) < 10` normalization.
    #[must_use]
    pub fn determinant(&self) -> (Complex, i32) {
        self.factors.sparse.determinant()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bound_and_standalone_assembly_follow_stamp_order() {
        let pattern = Arc::new(Pattern::new(2, [(1, 1), (0, 1), (0, 0), (1, 0)]));
        let mut bound = SparseMatrix::bound(Arc::clone(&pattern));
        let mut triplets = SparseMatrix::new(2, 2);
        for (r, c, v) in [(0, 0, 3.), (1, 1, 4.), (0, 1, -0.), (1, 0, 2.), (0, 0, 1.)] {
            bound.add(r, c, v).unwrap();
            triplets.add(r, c, v).unwrap();
        }
        let mut e = SparseMatrix::new(2, 2);
        for v in [1e16, -1e16, 1.] {
            e.add(0, 1, v).unwrap();
        }
        e.add(1, 1, 0.5).unwrap();
        let groups = [
            DynamicGroup {
                range: 0..3,
                terms: vec![1e16, -1e16, 1.],
                scale: 3.,
                divisor: 2.,
                prescaled: true,
            },
            DynamicGroup {
                range: 3..4,
                terms: vec![0.5],
                scale: 1.,
                divisor: 1.,
                prescaled: false,
            },
        ];
        let values = |a: &SparseMatrix| {
            let matrix = ComplexMatrix::from_operators_grouped_on(&pattern, a, &e, &groups, -2.)
                .unwrap()
                .unwrap();
            let Assembly::Bound { values, .. } = matrix.assembly else {
                panic!("expected bound values")
            };
            values
        };
        assert_eq!(values(&bound), values(&triplets));
        assert_eq!(
            values(&bound),
            [
                Complex::new(4., -1.),
                Complex::imaginary(1.5),
                Complex::real(4.),
                Complex::real(2.)
            ]
        );
        let mut overflow = SparseMatrix::new(2, 2);
        overflow.add(0, 0, f64::MAX).unwrap();
        overflow.add(0, 0, f64::MAX).unwrap();
        assert!(ComplexMatrix::from_operators_on(&pattern, &overflow, &e, 1.).is_err());
    }
}
