//! Rust port of Kenneth Kundert's SPARSE 1.3, ngspice's native sparse LU.
//!
//! C: `src/maths/sparse/` (`spbuild.c`, `spfactor.c`, `spsolve.c`,
//! `sputils.c`), driven through `spsmp.c`. SPARSE (author Kenneth S.
//! Kundert, advising professor Alberto Sangiovanni-Vincentelli, UC Berkeley)
//! is distributed under the permissive Berkeley notice in its headers, so
//! unlike KLU it may be ported here. The full notice is preserved verbatim
//! in the source comments immediately below and in `NOTICE`.
//
// Verbatim upstream license notice:
// Copyright (c) 1985,86,87,88,89,90
// by Kenneth S. Kundert and the University of California.
//
// Permission to use, copy, modify, and distribute this software and
// its documentation for any purpose and without fee is hereby granted,
// provided that the copyright notices appear in all copies and
// supporting documentation and that the authors and the University of
// California are properly credited.  The authors and the University of
// California make no representations as to the suitability of this
// software for any purpose.  It is provided `as is', without express
// or implied warranty.
//!
//! The configuration matches ngspice's `spconfig.h`: `TRANSLATE`,
//! `EXPANDABLE` (storage is allocated for the external size up front, and
//! the internal size grows as `Translate` numbers new rows, as in C),
//! `DIAGONAL_PIVOTING`,
//! `MODIFIED_NODAL`, `TRANSPOSE`, `DETERMINANT`; no `MODIFIED_MARKOWITZ`,
//! scaling or condition estimation. The default relative threshold is
//! `1e-3` and `TIES_MULTIPLIER` 5 (`MAX_MARKOWITZ_TIES` only applies to
//! `MODIFIED_MARKOWITZ`).
//!
//! The orthogonal linked lists of C pointers become an element arena with
//! index links; a C `ElementPtr *` "address of a link" is a [`Slot`]. The
//! control flow of every routine follows the C function of the same name,
//! including the order of pivot candidates, tie breaking and fill-in creation,
//! so pivot sequences match SPARSE for the same element creation order.
//! Divergences: the factor loop always uses the direct (scatter/gather)
//! variant of `spFactor`, which performs the same arithmetic as the indirect
//! one; and `SearchForSingleton` stops on a structurally missing diagonal,
//! which is where ngspice's version either breaks or dereferences `NULL`.

#![expect(
    clippy::float_cmp,
    reason = "C device branches compare floating-point values exactly, including nonfinite cases"
)]

use super::linear::numerical;
use std::simd::{
    Mask, Select,
    cmp::{SimdPartialEq, SimdPartialOrd},
    f64x4,
    num::SimdFloat,
};

use crate::maths::linear::{BatchSolveFailure, RHS_BATCH_WIDTH, SolveBatchWorkspace};
use crate::primitives::{Complex, SpiceResult};

const NONE: usize = usize::MAX;
const DEFAULT_THRESHOLD: f64 = 1.0e-3;
const TIES_MULTIPLIER: i64 = 5;

/// Stable original element handle obtained only from a validated frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct SolverElement(usize);

fn sparse_reserve<T>(entries: &mut Vec<T>, additional: usize) -> SpiceResult<()> {
    entries.try_reserve(additional).map_err(|error| {
        numerical(
            "SPARSE storage",
            format!("cannot reserve {additional} entries: {error}"),
        )
    })
}

fn sparse_rows(size: usize) -> SpiceResult<usize> {
    // The external indexing uses a zero sentinel and the pivot workspace
    // also needs a trailing sentinel.
    let count = size
        .checked_add(2)
        .ok_or_else(|| numerical("SPARSE storage", "row sentinel count overflows"))?;
    std::alloc::Layout::array::<usize>(count)
        .map_err(|error| numerical("SPARSE storage", format!("row index capacity: {error}")))?;
    Ok(count - 1)
}

fn sparse_indices(size: usize) -> SpiceResult<Vec<usize>> {
    let mut indices = Vec::new();
    sparse_reserve(&mut indices, sparse_rows(size)?)?;
    indices.extend(0..=size);
    Ok(indices)
}

type LaneMask = Mask<i64, RHS_BATCH_WIDTH>;

/// One SIMD tile, split by component so scatter updates never repack
/// interleaved complex numbers. Real solves leave the imaginary plane empty.
#[derive(Debug, Clone, Copy)]
pub(super) struct LaneParts {
    re: f64x4,
    im: f64x4,
}

/// Scratch for independent numeric factor tiles. Reuses the packed matrix,
/// scatter rows and reciprocal storage across successive frequency tiles.
#[derive(Debug, Default)]
pub(super) struct IndependentFactorTileWorkspace {
    values: Vec<LaneParts>,
    scratch: Vec<LaneParts>,
    inverses: Vec<LaneParts>,
}

impl LaneParts {
    const ZERO: Self = Self {
        re: f64x4::splat(0.),
        im: f64x4::splat(0.),
    };

    fn from_lanes<T: Scalar>(values: [T; RHS_BATCH_WIDTH]) -> Self {
        Self {
            re: f64x4::from_array(values.map(Scalar::real)),
            im: f64x4::from_array(values.map(Scalar::imaginary)),
        }
    }

    fn get<T: Scalar>(self, lane: usize) -> T {
        T::from_parts(self.re[lane], self.im[lane])
    }

    fn set<T: Scalar>(&mut self, lane: usize, value: T) {
        self.re[lane] = value.real();
        self.im[lane] = value.imaginary();
    }

    fn select(mask: LaneMask, yes: Self, no: Self) -> Self {
        Self {
            re: mask.select(yes.re, no.re),
            im: mask.select(yes.im, no.im),
        }
    }
}

fn read_tile_row<T: Scalar>(work: &SolveBatchWorkspace<T>, index: usize) -> LaneParts {
    LaneParts {
        re: work.real[index],
        im: if T::HAS_IMAGINARY {
            work.imag[index]
        } else {
            f64x4::splat(0.)
        },
    }
}

fn write_tile_row<T: Scalar>(work: &mut SolveBatchWorkspace<T>, index: usize, value: LaneParts) {
    work.real[index] = value.re;
    if T::HAS_IMAGINARY {
        work.imag[index] = value.im;
    }
}

/// Keep the original numerator alongside the normalized multiplier.
/// Ordinary independent lanes use SIMD; exceptional lanes retain the
/// scalar cross-product recovery and its exact arithmetic order.
pub(super) struct BatchUpdate {
    numerator: LaneParts,
    multiplier: LaneParts,
    active: LaneMask,
    normal: LaneMask,
}

impl BatchUpdate {
    fn new<T: Scalar>(numerator: LaneParts, pivot: T, inverse: T) -> Self {
        let zero = f64x4::splat(0.);
        let active = numerator.re.simd_ne(zero) | numerator.im.simd_ne(zero);
        if !active.any() {
            return Self {
                numerator,
                multiplier: numerator,
                active,
                normal: LaneMask::splat(false),
            };
        }
        let candidate = T::batch_mul(numerator, inverse);
        let fast = T::batch_normal(candidate) & LaneMask::splat(!inverse.is_zero());
        let mut multiplier = LaneParts::select(active, candidate, numerator);
        let exceptional = active & !fast;
        if exceptional.any() {
            for lane in 0..RHS_BATCH_WIDTH {
                if exceptional.test(lane) {
                    let value = numerator.get::<T>(lane);
                    multiplier.set(lane, value.quotient(pivot, inverse));
                }
            }
        }
        Self {
            numerator,
            multiplier,
            active,
            normal: T::batch_normal(multiplier),
        }
    }

    fn recover<T: Scalar>(
        &self,
        old: LaneParts,
        ordinary: LaneParts,
        pivot: T,
        lower: T,
        fast: LaneMask,
    ) -> LaneParts {
        let mut next = LaneParts::select(self.active, ordinary, old);
        let exceptional = self.active & !fast;
        if exceptional.any() {
            for lane in 0..RHS_BATCH_WIDTH {
                if exceptional.test(lane) {
                    let value = old.get::<T>(lane).schur(
                        pivot,
                        self.numerator.get(lane),
                        self.multiplier.get(lane),
                        lower,
                        self.normal.test(lane),
                    );
                    next.set(lane, value);
                }
            }
        }
        next
    }
}

/// SIMD Schur update for distinct numeric matrices sharing one symbolic
/// traversal. Unlike multiple-RHS updates, each lane owns its own pivot
/// and reciprocal. Exceptional lanes use the ordinary scalar recovery.
#[derive(Clone, Copy)]
struct IndependentBatchUpdate {
    numerator: LaneParts,
    pivot: LaneParts,
    multiplier: LaneParts,
    normal: LaneMask,
    active: LaneMask,
}

impl IndependentBatchUpdate {
    fn new(numerator: LaneParts, pivot: LaneParts, inverse: LaneParts, enabled: LaneMask) -> Self {
        let zero = f64x4::splat(0.);
        let active = (numerator.re.simd_ne(zero) | numerator.im.simd_ne(zero)) & enabled;
        let mut multiplier = LaneParts {
            re: numerator.re * inverse.re - numerator.im * inverse.im,
            im: numerator.re * inverse.im + numerator.im * inverse.re,
        };
        let normal = (multiplier.re.abs() + multiplier.im.abs()).is_normal();
        let inverse_zero = inverse.re.simd_eq(zero) & inverse.im.simd_eq(zero);
        let exceptional = active & (!normal | inverse_zero);
        if exceptional.any() {
            for lane in 0..RHS_BATCH_WIDTH {
                if exceptional.test(lane) {
                    multiplier.set(
                        lane,
                        numerator
                            .get::<Complex>(lane)
                            .quotient(pivot.get::<Complex>(lane), inverse.get::<Complex>(lane)),
                    );
                }
            }
        }
        Self {
            numerator,
            pivot,
            multiplier,
            normal: (multiplier.re.abs() + multiplier.im.abs()).is_normal(),
            active,
        }
    }

    fn apply(self, acc: LaneParts, lower: LaneParts) -> LaneParts {
        let product = LaneParts {
            re: self.multiplier.re * lower.re - self.multiplier.im * lower.im,
            im: self.multiplier.re * lower.im + self.multiplier.im * lower.re,
        };
        let mut result = LaneParts {
            re: acc.re - product.re,
            im: acc.im - product.im,
        };
        let norm = result.re.abs() + result.im.abs();
        let scale = (acc.re.abs() + acc.im.abs()) * f64x4::splat(0.0625);
        let zero = f64x4::splat(0.);
        let lower_zero = lower.re.simd_eq(zero) & lower.im.simd_eq(zero);
        let finite = result.re.is_finite() & result.im.is_finite();
        let fast = (self.normal & norm.is_normal() & norm.simd_gt(scale)) | (finite & lower_zero);
        let exceptional = self.active & !fast;
        if exceptional.any() {
            for lane in 0..RHS_BATCH_WIDTH {
                if exceptional.test(lane) {
                    result.set(
                        lane,
                        acc.get::<Complex>(lane).schur(
                            self.pivot.get::<Complex>(lane),
                            self.numerator.get::<Complex>(lane),
                            self.multiplier.get::<Complex>(lane),
                            lower.get::<Complex>(lane),
                            self.normal.test(lane),
                        ),
                    );
                }
            }
        }
        LaneParts::select(self.active, result, acc)
    }
}

/// One real RHS or a SIMD tile of independent real RHS values. Both
/// instantiations perform the same rounded operations and safety checks.
trait RealGatherLane:
    Copy + std::ops::Add<Output = Self> + std::ops::Sub<Output = Self> + std::ops::Mul<Output = Self>
{
    type Valid: Copy
        + std::ops::BitAnd<Output = Self::Valid>
        + std::ops::BitOr<Output = Self::Valid>;
    fn broadcast(value: f64) -> Self;
    fn mask(value: bool) -> Self::Valid;
    fn all(valid: Self::Valid) -> bool;
    fn zero(self) -> Self::Valid;
    fn normal(self) -> Self::Valid;
    fn finite(self) -> Self::Valid;
    fn magnitude(self) -> Self;
    fn greater(self, other: Self) -> Self::Valid;
    fn choose(mask: Self::Valid, yes: Self, no: Self) -> Self;
}

impl RealGatherLane for f64 {
    type Valid = bool;
    fn broadcast(value: f64) -> Self {
        value
    }
    fn mask(value: bool) -> bool {
        value
    }
    fn all(valid: bool) -> bool {
        valid
    }
    fn zero(self) -> bool {
        self == 0.
    }
    fn normal(self) -> bool {
        self.is_normal()
    }
    fn finite(self) -> bool {
        self.is_finite()
    }
    fn magnitude(self) -> Self {
        self.abs()
    }
    fn greater(self, other: Self) -> bool {
        self > other
    }
    fn choose(mask: bool, yes: Self, no: Self) -> Self {
        if mask { yes } else { no }
    }
}

impl RealGatherLane for f64x4 {
    type Valid = LaneMask;
    fn broadcast(value: f64) -> Self {
        Self::splat(value)
    }
    fn mask(value: bool) -> LaneMask {
        LaneMask::splat(value)
    }
    fn all(valid: LaneMask) -> bool {
        valid.all()
    }
    fn zero(self) -> LaneMask {
        self.simd_eq(Self::splat(0.))
    }
    fn normal(self) -> LaneMask {
        self.is_normal()
    }
    fn finite(self) -> LaneMask {
        self.is_finite()
    }
    fn magnitude(self) -> Self {
        self.abs()
    }
    fn greater(self, other: Self) -> LaneMask {
        self.simd_gt(other)
    }
    fn choose(mask: LaneMask, yes: Self, no: Self) -> Self {
        mask.select(yes, no)
    }
}

/// Standard normalized backward row, with a local arithmetic error bound.
/// Retain the raw scatter route if normalization/product loses exponent
/// range or the sign/zero of the result is unresolved by rounded terms.
/// This is not a global forward-error certificate for an ill-conditioned A.
struct RealGatherRow<V: RealGatherLane> {
    value: V,
    magnitude_sum: V,
    error: V,
    valid: V::Valid,
}

impl<V: RealGatherLane> RealGatherRow<V> {
    fn new(numerator: V, inverse: f64) -> Self {
        let value = numerator * V::broadcast(inverse);
        let valid = numerator.zero() | (V::mask(inverse.is_normal()) & value.normal());
        let value = V::choose(numerator.zero(), numerator, value);
        let magnitude_sum = value.magnitude();
        Self {
            value,
            magnitude_sum,
            error: magnitude_sum * V::broadcast(2. * f64::EPSILON),
            valid,
        }
    }

    fn subtract<const TRACK_ERROR: bool>(&mut self, upper: f64, inverse: f64, solved: V) {
        if upper == 0. {
            return;
        }
        let coefficient = upper * inverse;
        let term = V::broadcast(coefficient) * solved;
        let zero = solved.zero();
        self.valid = self.valid & (zero | (V::mask(coefficient.is_normal()) & term.normal()));
        let term = V::choose(zero, V::broadcast(0.), term);
        self.value = V::choose(zero, self.value, self.value - term);
        self.magnitude_sum = self.magnitude_sum + term.magnitude();
        if TRACK_ERROR {
            // Reciprocal, coefficient, term and accumulation rounding, with
            // slack for the positive error-bound accumulation itself.
            self.error = self.error + self.magnitude_sum * V::broadcast(4. * f64::EPSILON);
        }
    }

    fn accumulate<const TRACK_ERROR: bool>(
        numerator: V,
        inverse: f64,
        terms: impl Iterator<Item = (f64, V)>,
    ) -> Self {
        let mut row = Self::new(numerator, inverse);
        for (upper, solved) in terms {
            row.subtract::<TRACK_ERROR>(upper, inverse, solved);
        }
        row
    }

    fn finish(self, error: V) -> (V, V::Valid) {
        let valid = self.valid
            & self.magnitude_sum.finite()
            & error.finite()
            & (self.value.normal() | self.value.zero())
            & (self.magnitude_sum.zero() | self.value.magnitude().greater(error));
        (self.value, valid)
    }

    fn precise(numerator: V, inverse: f64, terms: impl Iterator<Item = (f64, V)>) -> (V, V::Valid) {
        let row = Self::accumulate::<true>(numerator, inverse, terms);
        let error = row.error;
        row.finish(error)
    }

    /// Sufficient certificate for the existing running error bound. The
    /// rounded nonnegative magnitude prefixes never exceed the final sum S.
    /// For a range-valid nonzero row S is normal. Each rounded old increment
    /// is at most round(4*eps*S), even the initial 2*eps term. Rounding that
    /// product upwards costs at most 1/8 relative error (including subnormal
    /// products); summing at most 2^32 such terms adds less than 2^-20
    /// relative error. The factor of two covers both and the final rounding
    /// downwards, even at the minimum normal S. S=0 has exactly zero error.
    ///
    /// This bound is only a fast acceptance test, not a new fallback policy.
    /// Replay the original running certificate if any lane is uncertain, so
    /// normalized subtraction order, bits and raw-recovery decisions remain
    /// unchanged. Short rows cannot amortize the additional certificate.
    fn gather(
        numerator: V,
        inverse: f64,
        terms: impl ExactSizeIterator<Item = (f64, V)> + Clone,
    ) -> (V, V::Valid) {
        if let Ok(count) = u32::try_from(terms.len())
            && count > 1
        {
            let row = Self::accumulate::<false>(numerator, inverse, terms.clone());
            let error =
                row.magnitude_sum * V::broadcast(8. * f64::EPSILON * (f64::from(count) + 1.));
            let candidate = row.finish(error);
            if V::all(candidate.1) {
                return candidate;
            }
        }
        Self::precise(numerator, inverse, terms)
    }
}

/// Matrix entry type: `f64` for real systems, [`Complex`] for AC.
pub(super) trait Scalar:
    Copy + PartialEq + std::fmt::Debug + std::ops::Div<Output = Self>
{
    /// Additive identity.
    const ZERO: Self;
    const HAS_IMAGINARY: bool;
    fn imaginary(self) -> f64;
    fn from_parts(real: f64, imaginary: f64) -> Self;
    fn batch_mul(parts: LaneParts, value: Self) -> LaneParts;
    fn batch_normal(parts: LaneParts) -> LaneMask;
    /// C `ELEMENT_MAG`: `|re| + |im|`.
    fn mag(self) -> f64;
    /// Real part, used by `CountTwins`' `ABS(Real) == 1.0` test.
    fn real(self) -> f64;
    /// C `1.0 / x` or `CMPLX_RECIPROCAL`.
    fn recip(self) -> Self;
    /// C `x * y` or `CMPLX_MULT`.
    fn mul(self, other: Self) -> Self;
    /// Cache only a normal reciprocal. A finite raw pivot can have an
    /// unrepresentable reciprocal, or one with a subnormal significand.
    fn cached_recip(self) -> Self {
        let inverse = self.recip();
        if inverse.normal_multiplier() {
            inverse
        } else {
            Self::ZERO
        }
    }
    /// Normalize a raw triangular entry without requiring its reciprocal
    /// to be representable. Use the actual division at exponent limits.
    #[inline]
    fn quotient(self, pivot: Self, inverse: Self) -> Self {
        if inverse.is_zero() {
            return self / pivot;
        }
        let result = self.mul(inverse);
        if result.normal_multiplier() || self.is_zero() {
            result
        } else {
            self / pivot
        }
    }
    /// Update a Schur entry using the rounded multiplier only when well
    /// conditioned; otherwise recover the unnormalized cross-product.
    fn schur(self, pivot: Self, upper: Self, mult: Self, lower: Self, normal: bool) -> Self;
    /// Update four independently normalized RHS lanes, recovering unsafe
    /// ordinary results with the same scalar kernel.
    fn schur_batch(acc: LaneParts, pivot: Self, lower: Self, update: &BatchUpdate) -> LaneParts;
    /// Classify the multiplier once before traversing the lower column.
    fn normal_multiplier(self) -> bool;
    /// Whether the value is exactly zero (`== 0.0` on every part).
    fn is_zero(self) -> bool;
    /// Whether every scalar component is finite.
    fn is_finite(self) -> bool;
    /// `x.Real += r`.
    fn add_real(self, r: f64) -> Self;
}

impl Scalar for f64 {
    const ZERO: Self = 0.0;
    const HAS_IMAGINARY: bool = false;
    fn imaginary(self) -> f64 {
        0.
    }
    fn from_parts(real: f64, _imaginary: f64) -> Self {
        real
    }
    fn batch_mul(parts: LaneParts, value: Self) -> LaneParts {
        LaneParts {
            re: parts.re * f64x4::splat(value),
            im: f64x4::splat(0.),
        }
    }
    fn batch_normal(parts: LaneParts) -> LaneMask {
        parts.re.is_normal()
    }
    fn mag(self) -> f64 {
        self.abs()
    }
    fn real(self) -> f64 {
        self
    }
    fn recip(self) -> Self {
        1.0 / self
    }
    fn mul(self, other: Self) -> Self {
        self * other
    }
    #[inline]
    #[expect(
        clippy::suboptimal_flops,
        reason = "the rounded multiplier cannot supply a reliable cancellation residual; near cancellation uses the original cross-products instead"
    )]
    fn schur(self, pivot: Self, upper: Self, mult: Self, lower: Self, normal: bool) -> Self {
        let result = self - mult * lower;
        if result.is_finite()
            && ((normal && result.abs() > self.abs() * 0.0625) || upper == 0. || lower == 0.)
        {
            result
        } else if self.is_finite() && upper.is_finite() && lower.is_finite() {
            crate::primitives::value::real_schur(self, pivot, upper, lower)
        } else {
            result
        }
    }
    #[inline]
    fn schur_batch(acc: LaneParts, pivot: Self, lower: Self, update: &BatchUpdate) -> LaneParts {
        let result = acc.re - update.multiplier.re * f64x4::splat(lower);
        let fast = result.is_finite()
            & ((update.normal & result.abs().simd_gt(acc.re.abs() * f64x4::splat(0.0625)))
                | LaneMask::splat(lower == 0.));
        update.recover(
            acc,
            LaneParts {
                re: result,
                im: f64x4::splat(0.),
            },
            pivot,
            lower,
            fast,
        )
    }
    fn normal_multiplier(self) -> bool {
        self.is_normal()
    }
    fn is_zero(self) -> bool {
        self == 0.0
    }
    fn is_finite(self) -> bool {
        Self::is_finite(self)
    }
    fn add_real(self, r: f64) -> Self {
        self + r
    }
}

impl Scalar for Complex {
    const ZERO: Self = Self::ZERO;
    const HAS_IMAGINARY: bool = true;
    fn imaginary(self) -> f64 {
        self.im
    }
    fn from_parts(real: f64, imaginary: f64) -> Self {
        Self::new(real, imaginary)
    }
    fn batch_mul(parts: LaneParts, value: Self) -> LaneParts {
        let re = f64x4::splat(value.re);
        let im = f64x4::splat(value.im);
        LaneParts {
            re: parts.re * re - parts.im * im,
            im: parts.re * im + parts.im * re,
        }
    }
    fn batch_normal(parts: LaneParts) -> LaneMask {
        (parts.re.abs() + parts.im.abs()).is_normal()
    }
    fn mag(self) -> f64 {
        self.re.abs() + self.im.abs()
    }
    fn real(self) -> f64 {
        self.re
    }
    fn recip(self) -> Self {
        // CMPLX_RECIPROCAL (spdefs.h).
        let den = self;
        let result =
            if (den.re >= den.im && den.re > -den.im) || (den.re < den.im && den.re <= -den.im) {
                let r = den.im / den.re;
                let re = 1.0 / (r.mul_add(den.im, den.re));
                Self::new(re, -r * re)
            } else {
                let r = den.re / den.im;
                let im = -1.0 / (r.mul_add(den.re, den.im));
                Self::new(-r * im, im)
            };
        if !result.is_finite() || result == Self::ZERO {
            // The C ratio formula can overflow its denominator even for
            // a finite pivot with a representable reciprocal.
            Self::real(1.0) / den
        } else {
            result
        }
    }
    #[inline]
    #[expect(
        clippy::suboptimal_flops,
        reason = "ordinary normwise products use the cheap two-product sum; only exponent-range failures need the robust kernel"
    )]
    fn mul(self, b: Self) -> Self {
        // Matrix factors and triangular solves use normwise error bounds:
        // a normal finite complex product already meets that bound with
        // ordinary arithmetic. Recover exponent-range failures using the
        // componentwise robust public multiplication.
        let result = Self::new(
            self.re * b.re - self.im * b.im,
            self.re * b.im + self.im * b.re,
        );
        if (result.re.abs() + result.im.abs()).is_normal() || self.is_zero() || b.is_zero() {
            result
        } else {
            self * b
        }
    }
    #[inline]
    #[expect(
        clippy::suboptimal_flops,
        reason = "the cheap update estimates cancellation; a fused update using the rounded multiplier would not remove its rounding error"
    )]
    fn schur(self, pivot: Self, upper: Self, mult: Self, lower: Self, normal: bool) -> Self {
        // The ordinary Schur update is a three-term dot product.
        // Avoid separately compensating the normalized multiplier's product:
        // its error is dominated by the multiplier itself near cancellation.
        let product = Self::new(
            mult.re * lower.re - mult.im * lower.im,
            mult.re * lower.im + mult.im * lower.re,
        );
        let result = self - product;
        let norm = result.re.abs() + result.im.abs();
        let scale = (self.re.abs() + self.im.abs()) * 0.0625;
        if (normal && norm.is_normal() && norm > scale)
            || (result.is_finite() && (upper.is_zero() || lower.is_zero()))
        {
            result
        } else if self.is_finite() && upper.is_finite() && lower.is_finite() {
            crate::primitives::value::complex_schur(self, pivot, upper, lower)
        } else {
            result
        }
    }
    #[inline]
    fn schur_batch(acc: LaneParts, pivot: Self, lower: Self, update: &BatchUpdate) -> LaneParts {
        let product = Self::batch_mul(update.multiplier, lower);
        let result = LaneParts {
            re: acc.re - product.re,
            im: acc.im - product.im,
        };
        let norm = result.re.abs() + result.im.abs();
        let scale = (acc.re.abs() + acc.im.abs()) * f64x4::splat(0.0625);
        let fast = (update.normal & norm.is_normal() & norm.simd_gt(scale))
            | (result.re.is_finite() & result.im.is_finite() & LaneMask::splat(lower.is_zero()));
        update.recover(acc, result, pivot, lower, fast)
    }
    fn normal_multiplier(self) -> bool {
        (self.re.abs() + self.im.abs()).is_normal()
    }
    fn is_zero(self) -> bool {
        self.re == 0.0 && self.im == 0.0
    }
    fn is_finite(self) -> bool {
        Self::is_finite(self)
    }
    fn add_real(self, r: f64) -> Self {
        Self::new(self.re + r, self.im)
    }
}

/// SPARSE's error state (`spOKAY`, `spSMALL_PIVOT`, `spZERO_DIAG`,
/// `spSINGULAR`). ngspice maps `spSMALL_PIVOT` to `OK`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Status {
    Okay,
    SmallPivot,
    ZeroDiag,
    Singular,
    /// The factorization produced an unrepresentable numeric value.
    NonFinite,
    /// A traversal plan cannot be derived from the ordered topology.
    InvalidPlan,
    /// Coefficients must be reloaded before another factorization.
    NeedsReload,
    /// A required factor allocation exceeds storage capacity.
    Resource,
}

#[derive(Debug, Clone)]
struct Element {
    row: usize,
    col: usize,
    next_in_row: usize,
    next_in_col: usize,
}

/// Index width of an ordered traversal plan. Most circuits fit in u32;
/// retain usize support for larger standalone matrices, with the dispatch
/// outside all numeric loops.
trait PlanIndex: Copy {
    fn pack(index: usize) -> Option<Self>;
    fn unpack(self) -> usize;
}

impl PlanIndex for usize {
    fn pack(index: usize) -> Option<Self> {
        Some(index)
    }
    fn unpack(self) -> usize {
        self
    }
}

impl PlanIndex for u32 {
    fn pack(index: usize) -> Option<Self> {
        Self::try_from(index).ok()
    }
    fn unpack(self) -> usize {
        // The simulator supports 32- and 64-bit targets.
        self as usize
    }
}

/// A numeric traversal entry: stable element handle and internal row
/// (column runs) or internal column (row runs).
#[derive(Debug, Clone, Copy)]
struct RunEntry<I> {
    handle: I,
    coordinate: I,
}

impl<I: PlanIndex> RunEntry<I> {
    fn new(handle: usize, coordinate: usize) -> Option<Self> {
        Some(Self {
            handle: I::pack(handle)?,
            coordinate: I::pack(coordinate)?,
        })
    }

    fn indices(self) -> (usize, usize) {
        (self.handle.unpack(), self.coordinate.unpack())
    }
}

#[derive(Debug, Clone, Copy)]
struct AxisRun<I> {
    start: I,
    diagonal: I,
    end: I,
}

impl<I: PlanIndex> AxisRun<I> {
    fn new(start: usize, diagonal: usize, end: usize) -> Option<Self> {
        if !(start <= diagonal && diagonal < end) {
            return None;
        }
        Some(Self {
            start: I::pack(start)?,
            diagonal: I::pack(diagonal)?,
            end: I::pack(end)?,
        })
    }
}

/// One packed axis, with per-coordinate diagonal split points.
#[derive(Debug)]
struct AxisRuns<I> {
    runs: Vec<AxisRun<I>>,
    entries: Vec<RunEntry<I>>,
}

impl<I: PlanIndex> AxisRuns<I> {
    fn all(&self, coordinate: usize) -> &[RunEntry<I>] {
        let run = self.runs[coordinate - 1];
        &self.entries[run.start.unpack()..run.end.unpack()]
    }

    fn before_diagonal(&self, coordinate: usize) -> &[RunEntry<I>] {
        let run = self.runs[coordinate - 1];
        &self.entries[run.start.unpack()..run.diagonal.unpack()]
    }

    fn after_diagonal(&self, coordinate: usize) -> &[RunEntry<I>] {
        let run = self.runs[coordinate - 1];
        &self.entries[run.diagonal.unpack() + 1..run.end.unpack()]
    }
}

/// Immutable packed traversal, derived from linked topology after ordering.
#[derive(Debug)]
struct FactorRuns<I> {
    columns: AxisRuns<I>,
    rows: AxisRuns<I>,
}

#[derive(Debug)]
enum FactorPlan {
    Compact(FactorRuns<u32>),
    Wide(FactorRuns<usize>),
}

enum RefactorIssue {
    ZeroPivot(usize),
    NonFinite,
}

impl<I: PlanIndex> FactorRuns<I> {
    /// The fixed-order SPARSE scatter/gather refactor, after column one.
    fn refactor<T: Scalar>(
        &self,
        values: &mut [T],
        diag: &[usize],
        dest: &mut [T],
        inverses: &mut [T],
    ) -> Result<(), RefactorIssue> {
        for step in 2..=self.columns.runs.len() {
            for entry in self.columns.all(step) {
                let (handle, row) = entry.indices();
                dest[row] = values[handle];
            }
            for entry in self.columns.before_diagonal(step) {
                let (handle, row) = entry.indices();
                let upper = dest[row];
                values[handle] = upper;
                if upper.is_zero() {
                    continue;
                }
                let pivot = values[diag[row]];
                let mult = upper.quotient(pivot, inverses[row]);
                let normal = mult.normal_multiplier();
                for entry in self.columns.after_diagonal(row) {
                    let (handle, r) = entry.indices();
                    dest[r] = dest[r].schur(pivot, upper, mult, values[handle], normal);
                }
            }
            for entry in self.columns.after_diagonal(step) {
                let (handle, row) = entry.indices();
                values[handle] = dest[row];
            }
            if dest[step].is_zero() {
                return Err(RefactorIssue::ZeroPivot(step));
            }
            if !dest[step].is_finite() {
                return Err(RefactorIssue::NonFinite);
            }
            values[diag[step]] = dest[step];
            inverses[step] = dest[step].cached_recip();
        }
        Ok(())
    }

    /// Refactor distinct numeric matrices with one immutable symbolic plan.
    /// Coefficients and row scratch are packed by element/row into four SIMD
    /// lanes. Pivots are independent; exceptional Schur updates use the
    /// scalar range-preserving kernel for that lane.
    fn refactor_independent_tile(
        &self,
        values: &mut [LaneParts],
        diag: &[usize],
        scratch: &mut [LaneParts],
        inverses: &mut [LaneParts],
        statuses: &mut [Status; RHS_BATCH_WIDTH],
        count: usize,
    ) {
        for step in 2..=self.columns.runs.len() {
            for entry in self.columns.all(step) {
                let (handle, row) = entry.indices();
                scratch[row] = values[handle];
            }
            for entry in self.columns.before_diagonal(step) {
                let (handle, row) = entry.indices();
                let upper = scratch[row];
                values[handle] = upper;
                let pivot = values[diag[row]];
                let enabled = LaneMask::from_array(std::array::from_fn(|lane| {
                    lane < count && statuses[lane] == Status::Okay
                }));
                let update = IndependentBatchUpdate::new(upper, pivot, inverses[row], enabled);
                if update.active.any() {
                    for entry in self.columns.after_diagonal(row) {
                        let (handle, target) = entry.indices();
                        scratch[target] = update.apply(scratch[target], values[handle]);
                    }
                }
            }
            for entry in self.columns.after_diagonal(step) {
                let (handle, row) = entry.indices();
                values[handle] = scratch[row];
            }
            for (lane, status) in statuses.iter_mut().enumerate().take(count) {
                if *status != Status::Okay {
                    continue;
                }
                let pivot = scratch[step].get::<Complex>(lane);
                *status = if pivot.is_zero() {
                    Status::ZeroDiag
                } else if !pivot.is_finite() {
                    Status::NonFinite
                } else {
                    Status::Okay
                };
                if *status == Status::Okay {
                    values[diag[step]].set(lane, pivot);
                    inverses[step].set(lane, pivot.cached_recip());
                }
            }
        }
    }

    /// Solve independent, identically ordered complex factors in their
    /// packed lanes. The same elimination traversal serves normal and
    /// transposed solves without sharing numeric pivots.
    fn solve_independent_tile(
        &self,
        values: &[LaneParts],
        diag: &[usize],
        inverses: &[LaneParts],
        x: &mut [LaneParts],
        enabled: LaneMask,
        transposed: bool,
    ) {
        let axis = if transposed {
            &self.rows
        } else {
            &self.columns
        };
        for step in 1..=self.columns.runs.len() {
            let update =
                IndependentBatchUpdate::new(x[step], values[diag[step]], inverses[step], enabled);
            if update.active.any() {
                for entry in axis.after_diagonal(step) {
                    let (handle, row) = entry.indices();
                    x[row] = update.apply(x[row], values[handle]);
                }
            }
        }
        for step in (1..=self.columns.runs.len()).rev() {
            let update =
                IndependentBatchUpdate::new(x[step], values[diag[step]], inverses[step], enabled);
            if update.active.any() {
                for entry in axis.before_diagonal(step) {
                    let (handle, row) = entry.indices();
                    x[row] = update.apply(x[row], values[handle]);
                }
                x[step] = LaneParts::select(update.active, update.multiplier, x[step]);
            }
        }
    }

    fn forward<T: Scalar>(
        &self,
        values: &[T],
        diag: &[usize],
        inverses: &[T],
        x: &mut [T],
        transposed: bool,
    ) {
        let size = self.columns.runs.len();
        // A = L D^-1 U, where both L and U carry the raw pivot D.
        // Forward substitution keeps D*y rather than materializing y
        // into the result buffer. The normalized value is only an
        // approximation used for updating later rows.
        for i in 1..=size {
            let numerator = x[i];
            if numerator.is_zero() {
                continue;
            }
            let pivot = values[diag[i]];
            let mult = numerator.quotient(pivot, inverses[i]);
            let normal = mult.normal_multiplier();
            let entries = if transposed {
                self.rows.after_diagonal(i)
            } else {
                self.columns.after_diagonal(i)
            };
            for entry in entries {
                let (handle, r) = entry.indices();
                x[r] = x[r].schur(pivot, numerator, mult, values[handle], normal);
            }
        }
    }

    fn backward_raw<T: Scalar>(
        &self,
        values: &[T],
        diag: &[usize],
        inverses: &[T],
        x: &mut [T],
        transposed: bool,
    ) {
        let size = self.columns.runs.len();
        // Solve the other raw triangular factor in reverse. Scatter
        // immediately, while the unrounded numerator is still available.
        for i in (1..=size).rev() {
            let numerator = x[i];
            if numerator.is_zero() {
                continue;
            }
            let pivot = values[diag[i]];
            let mult = numerator.quotient(pivot, inverses[i]);
            let normal = mult.normal_multiplier();
            let entries = if transposed {
                self.rows.before_diagonal(i)
            } else {
                self.columns.before_diagonal(i)
            };
            for entry in entries {
                let (handle, r) = entry.indices();
                x[r] = x[r].schur(pivot, numerator, mult, values[handle], normal);
            }
            x[i] = mult;
        }
    }

    fn solve<T: Scalar>(
        &self,
        values: &[T],
        diag: &[usize],
        inverses: &[T],
        x: &mut [T],
        transposed: bool,
        raw_only: bool,
    ) -> bool {
        self.forward(values, diag, inverses, x, transposed);
        if !T::HAS_IMAGINARY && !raw_only {
            let axis = if transposed {
                &self.columns
            } else {
                &self.rows
            };
            for i in (1..=self.columns.runs.len()).rev() {
                let inverse = inverses[i].real();
                let terms = axis.after_diagonal(i).iter().map(|entry| {
                    let (handle, col) = entry.indices();
                    (values[handle].real(), x[col].real())
                });
                let (value, valid) = RealGatherRow::gather(x[i].real(), inverse, terms);
                if !valid {
                    return false;
                }
                x[i] = T::from_parts(value, 0.);
            }
        } else {
            self.backward_raw(values, diag, inverses, x, transposed);
        }
        true
    }

    fn solve_batch<T: Scalar>(
        &self,
        values: &[T],
        diag: &[usize],
        inverses: &[T],
        work: &mut SolveBatchWorkspace<T>,
        transposed: bool,
    ) -> LaneMask {
        let size = self.columns.runs.len();
        let axis = if transposed {
            &self.rows
        } else {
            &self.columns
        };
        for i in 1..=size {
            let pivot = values[diag[i]];
            let update = BatchUpdate::new(read_tile_row(work, i), pivot, inverses[i]);
            if update.active.any() {
                for entry in axis.after_diagonal(i) {
                    let (handle, r) = entry.indices();
                    let next =
                        T::schur_batch(read_tile_row(work, r), pivot, values[handle], &update);
                    write_tile_row(work, r, next);
                }
            }
        }
        if !T::HAS_IMAGINARY {
            let gather = if transposed {
                &self.columns
            } else {
                &self.rows
            };
            let mut valid = LaneMask::splat(true);
            for i in (1..=size).rev() {
                let inverse = inverses[i].real();
                let terms = gather.after_diagonal(i).iter().map(|entry| {
                    let (handle, col) = entry.indices();
                    (values[handle].real(), work.real[col])
                });
                let (value, row_valid) = RealGatherRow::gather(work.real[i], inverse, terms);
                work.real[i] = value;
                valid &= row_valid;
            }
            return valid;
        }
        for i in (1..=size).rev() {
            let pivot = values[diag[i]];
            let update = BatchUpdate::new(read_tile_row(work, i), pivot, inverses[i]);
            if update.active.any() {
                for entry in axis.before_diagonal(i) {
                    let (handle, r) = entry.indices();
                    let next =
                        T::schur_batch(read_tile_row(work, r), pivot, values[handle], &update);
                    write_tile_row(work, r, next);
                }
                write_tile_row(work, i, update.multiplier);
            }
        }
        LaneMask::splat(true)
    }
}

/// The address of a column (or row) link: a list head or an element's link.
#[derive(Debug, Clone, Copy)]
enum Slot {
    Head(usize),
    After(usize),
}

/// Linked topology and maps are immutable between structural edits.
/// Numeric snapshots and scalar transitions share these allocations;
/// ordering that changes their structure detaches only the writer.
#[derive(Debug, Clone)]
struct Topology {
    original_entries: usize,
    elements: Vec<Element>,
    diag: Vec<usize>,
    first_in_col: Vec<usize>,
    first_in_row: Vec<usize>,
    ext_to_int_row: Vec<usize>,
    ext_to_int_col: Vec<usize>,
    int_to_ext_row: Vec<usize>,
    int_to_ext_col: Vec<usize>,
}

/// Statically selected direction of an ordered sparse linked list.
///
/// A row list is ordered by its element columns; a column list by its rows.
/// Each orientation names its orthogonal list at the type level, so swapping
/// one axis cannot accidentally relink that same axis instead.
trait LinkedAxis {
    type Orthogonal: LinkedAxis;

    fn link(topology: &Topology, slot: Slot) -> usize;
    fn set_link(topology: &mut Topology, slot: Slot, element: usize);
    fn coordinate(element: &Element) -> usize;
    fn set_coordinate(element: &mut Element, coordinate: usize);
    fn successor(element: &Element) -> usize;
    fn set_successor(element: &mut Element, successor: usize);
    fn swap_maps(topology: &mut Topology, first: usize, second: usize);
}

struct AlongRows;
struct AlongColumns;

impl LinkedAxis for AlongRows {
    type Orthogonal = AlongColumns;

    fn link(topology: &Topology, slot: Slot) -> usize {
        topology.row_get(slot)
    }

    fn set_link(topology: &mut Topology, slot: Slot, element: usize) {
        topology.row_set(slot, element);
    }

    fn coordinate(element: &Element) -> usize {
        element.col
    }

    fn set_coordinate(element: &mut Element, coordinate: usize) {
        element.col = coordinate;
    }

    fn successor(element: &Element) -> usize {
        element.next_in_row
    }

    fn set_successor(element: &mut Element, successor: usize) {
        element.next_in_row = successor;
    }

    fn swap_maps(topology: &mut Topology, first: usize, second: usize) {
        topology.first_in_row.swap(first, second);
        topology.int_to_ext_row.swap(first, second);
        topology.ext_to_int_row[topology.int_to_ext_row[first]] = first;
        topology.ext_to_int_row[topology.int_to_ext_row[second]] = second;
    }
}

impl LinkedAxis for AlongColumns {
    type Orthogonal = AlongRows;

    fn link(topology: &Topology, slot: Slot) -> usize {
        topology.col_get(slot)
    }

    fn set_link(topology: &mut Topology, slot: Slot, element: usize) {
        topology.col_set(slot, element);
    }

    fn coordinate(element: &Element) -> usize {
        element.row
    }

    fn set_coordinate(element: &mut Element, coordinate: usize) {
        element.row = coordinate;
    }

    fn successor(element: &Element) -> usize {
        element.next_in_col
    }

    fn set_successor(element: &mut Element, successor: usize) {
        element.next_in_col = successor;
    }

    fn swap_maps(topology: &mut Topology, first: usize, second: usize) {
        topology.first_in_col.swap(first, second);
        topology.int_to_ext_col.swap(first, second);
        topology.ext_to_int_col[topology.int_to_ext_col[first]] = first;
        topology.ext_to_int_col[topology.int_to_ext_col[second]] = second;
    }
}

impl Topology {
    fn translate(
        &mut self,
        current_size: &mut usize,
        ext_row: usize,
        ext_col: usize,
    ) -> (usize, usize) {
        let mut int_row = self.ext_to_int_row[ext_row];
        if int_row == NONE {
            *current_size += 1;
            int_row = *current_size;
            self.ext_to_int_row[ext_row] = int_row;
            self.ext_to_int_col[ext_row] = int_row;
            self.int_to_ext_row[int_row] = ext_row;
            self.int_to_ext_col[int_row] = ext_row;
        }
        let mut int_col = self.ext_to_int_col[ext_col];
        if int_col == NONE {
            *current_size += 1;
            int_col = *current_size;
            self.ext_to_int_row[ext_col] = int_col;
            self.ext_to_int_col[ext_col] = int_col;
            self.int_to_ext_row[int_col] = ext_col;
            self.int_to_ext_col[int_col] = ext_col;
        }
        (int_row, int_col)
    }

    fn locate_in_col(&self, mut last: Slot, row: usize) -> (Slot, usize) {
        let mut e = self.col_get(last);
        while e != NONE {
            let r = self.elements[e].row;
            match r.cmp(&row) {
                std::cmp::Ordering::Less => {
                    last = Slot::After(e);
                    e = self.elements[e].next_in_col;
                }
                std::cmp::Ordering::Equal => return (last, e),
                std::cmp::Ordering::Greater => break,
            }
        }
        (last, NONE)
    }

    fn create_element(&mut self, row: usize, col: usize, last: Slot, rows_linked: bool) -> usize {
        let e = self.elements.len();
        self.elements.push(Element {
            row,
            col,
            next_in_row: NONE,
            next_in_col: self.col_get(last),
        });
        if row == col {
            self.diag[row] = e;
        }
        self.col_set(last, e);
        if rows_linked {
            let mut previous = NONE;
            let mut p = self.first_in_row[row];
            while p != NONE && self.elements[p].col < col {
                previous = p;
                p = self.elements[p].next_in_row;
            }
            if previous == NONE {
                self.elements[e].next_in_row = self.first_in_row[row];
                self.first_in_row[row] = e;
            } else {
                self.elements[e].next_in_row = self.elements[previous].next_in_row;
                self.elements[previous].next_in_row = e;
            }
        }
        e
    }

    /// Append to a sorted column in constant time, or mark it for deferred
    /// sorting if rows arrive in an arbitrary order. A missing tail with a
    /// nonempty head denotes an unordered column.
    fn append_original(&mut self, row: usize, col: usize, tails: &mut [usize]) -> usize {
        let tail = tails[col];
        let head = self.first_in_col[col];
        if head == NONE {
            let handle = self.create_element(row, col, Slot::Head(col), false);
            tails[col] = handle;
            handle
        } else if tail != NONE && row > self.elements[tail].row {
            let handle = self.create_element(row, col, Slot::After(tail), false);
            tails[col] = handle;
            handle
        } else {
            if row >= self.elements[head].row {
                tails[col] = NONE;
            }
            self.create_element(row, col, Slot::Head(col), false)
        }
    }

    /// Finalize only unordered columns. Stable handles remain in original
    /// creation order; the linked traversal is sorted by internal row.
    fn finish_original_columns(&mut self, tails: &[usize]) -> SpiceResult<()> {
        let mut run = Vec::new();
        for (col, &tail) in tails.iter().enumerate().skip(1) {
            if tail != NONE || self.first_in_col[col] == NONE {
                continue;
            }
            run.clear();
            let mut e = self.first_in_col[col];
            while e != NONE {
                if run.len() == run.capacity() {
                    sparse_reserve(&mut run, 1)?;
                }
                run.push(e);
                e = self.elements[e].next_in_col;
            }
            run.sort_unstable_by_key(|&handle| self.elements[handle].row);
            self.first_in_col[col] = run[0];
            for pair in run.windows(2) {
                if self.elements[pair[0]].row == self.elements[pair[1]].row {
                    return Err(numerical(
                        "SPARSE construction",
                        "duplicate original sparse element",
                    ));
                }
                self.elements[pair[0]].next_in_col = pair[1];
            }
            self.elements[run[run.len() - 1]].next_in_col = NONE;
        }
        Ok(())
    }

    fn col_get(&self, slot: Slot) -> usize {
        match slot {
            Slot::Head(col) => self.first_in_col[col],
            Slot::After(e) => self.elements[e].next_in_col,
        }
    }

    fn col_set(&mut self, slot: Slot, value: usize) {
        match slot {
            Slot::Head(col) => self.first_in_col[col] = value,
            Slot::After(e) => self.elements[e].next_in_col = value,
        }
    }

    fn row_get(&self, slot: Slot) -> usize {
        match slot {
            Slot::Head(row) => self.first_in_row[row],
            Slot::After(e) => self.elements[e].next_in_row,
        }
    }

    fn row_set(&mut self, slot: Slot, value: usize) {
        match slot {
            Slot::Head(row) => self.first_in_row[row] = value,
            Slot::After(e) => self.elements[e].next_in_row = value,
        }
    }

    /// Relink the two affected coordinates within an orthogonal linked list.
    ///
    /// Preserve the SPARSE case order exactly: missing first, missing second,
    /// adjacent elements and non-adjacent elements. In particular, direct
    /// coordinate updates and predecessor writes must remain in this order to
    /// preserve stable element handles and the singly linked sort invariant.
    fn exchange_list_elements<A: LinkedAxis>(
        &mut self,
        first: usize,
        e1: usize,
        second: usize,
        e2: usize,
        list: usize,
    ) {
        let mut preceding_first = Slot::Head(list);
        let mut cursor = A::link(self, preceding_first);
        while A::coordinate(&self.elements[cursor]) < first {
            preceding_first = Slot::After(cursor);
            cursor = A::link(self, preceding_first);
        }
        if e1 == NONE {
            let following_first = cursor;
            if A::coordinate(&self.elements[following_first]) != second {
                let preceding_second;
                loop {
                    let preceding = Slot::After(cursor);
                    cursor = A::link(self, preceding);
                    if A::coordinate(&self.elements[cursor]) >= second {
                        preceding_second = preceding;
                        break;
                    }
                }
                let following_second = A::successor(&self.elements[e2]);
                A::set_link(self, preceding_second, following_second);
                A::set_link(self, preceding_first, e2);
                A::set_successor(&mut self.elements[e2], following_first);
            }
            A::set_coordinate(&mut self.elements[e2], first);
        } else {
            let following_first = A::successor(&self.elements[e1]);
            if e2 == NONE {
                if following_first != NONE
                    && A::coordinate(&self.elements[following_first]) < second
                {
                    A::set_link(self, preceding_first, following_first);
                    let mut cursor = following_first;
                    let preceding_second;
                    loop {
                        let preceding = Slot::After(cursor);
                        cursor = A::link(self, preceding);
                        if cursor == NONE || A::coordinate(&self.elements[cursor]) >= second {
                            preceding_second = preceding;
                            break;
                        }
                    }
                    A::set_link(self, preceding_second, e1);
                    A::set_successor(&mut self.elements[e1], cursor);
                    A::set_link(self, preceding_first, following_first);
                }
                A::set_coordinate(&mut self.elements[e1], second);
            } else {
                if A::coordinate(&self.elements[following_first]) == second {
                    let following_second = A::successor(&self.elements[e2]);
                    A::set_successor(&mut self.elements[e1], following_second);
                    A::set_successor(&mut self.elements[e2], e1);
                    A::set_link(self, preceding_first, e2);
                } else {
                    let mut cursor = following_first;
                    let preceding_second;
                    loop {
                        let preceding = Slot::After(cursor);
                        cursor = A::link(self, preceding);
                        if A::coordinate(&self.elements[cursor]) >= second {
                            preceding_second = preceding;
                            break;
                        }
                    }
                    let following_second = A::successor(&self.elements[e2]);
                    A::set_link(self, preceding_first, e2);
                    A::set_successor(&mut self.elements[e2], following_first);
                    A::set_link(self, preceding_second, e1);
                    A::set_successor(&mut self.elements[e1], following_second);
                }
                A::set_coordinate(&mut self.elements[e1], second);
                A::set_coordinate(&mut self.elements[e2], first);
            }
        }
    }

    /// Exchange the selected axis while relinking only orthogonal lists.
    ///
    /// One monomorphized walk implements SPARSE row and column exchange.
    /// The linked ordering and the permutation-map mutation remain ordered
    /// exactly as in the original two routines.
    fn exchange_axis<A: LinkedAxis>(&mut self, mut first: usize, mut second: usize) {
        if first == second {
            return;
        }
        if first > second {
            std::mem::swap(&mut first, &mut second);
        }
        let mut p1 = A::link(self, Slot::Head(first));
        let mut p2 = A::link(self, Slot::Head(second));
        while p1 != NONE || p2 != NONE {
            let (orthogonal, e1, e2);
            if p1 == NONE {
                orthogonal = A::coordinate(&self.elements[p2]);
                (e1, e2) = (NONE, p2);
                p2 = A::successor(&self.elements[p2]);
            } else if p2 == NONE
                || A::coordinate(&self.elements[p1]) < A::coordinate(&self.elements[p2])
            {
                orthogonal = A::coordinate(&self.elements[p1]);
                (e1, e2) = (p1, NONE);
                p1 = A::successor(&self.elements[p1]);
            } else if A::coordinate(&self.elements[p1]) > A::coordinate(&self.elements[p2]) {
                orthogonal = A::coordinate(&self.elements[p2]);
                (e1, e2) = (NONE, p2);
                p2 = A::successor(&self.elements[p2]);
            } else {
                orthogonal = A::coordinate(&self.elements[p1]);
                (e1, e2) = (p1, p2);
                p1 = A::successor(&self.elements[p1]);
                p2 = A::successor(&self.elements[p2]);
            }
            self.exchange_list_elements::<A::Orthogonal>(first, e1, second, e2, orthogonal);
        }
        A::swap_maps(self, first, second);
    }

    /// SPARSE spcRowExchange, with statically bound row orientation.
    fn row_exchange(&mut self, first: usize, second: usize) {
        self.exchange_axis::<AlongRows>(first, second);
    }

    /// SPARSE spcColExchange, with statically bound column orientation.
    fn col_exchange(&mut self, first: usize, second: usize) {
        self.exchange_axis::<AlongColumns>(first, second);
    }
}

/// Parity of the row/column permutation accumulated during ordering.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PermutationParity {
    Even,
    Odd,
}

impl PermutationParity {
    const fn toggle(&mut self) {
        *self = match self {
            Self::Even => Self::Odd,
            Self::Odd => Self::Even,
        };
    }
}

/// Cached summaries of contiguous Markowitz-product blocks. They permit
/// skipping ranges that cannot contain the next reference-order candidate.
/// Dirty blocks are rebuilt on demand, never changing the candidate order.
#[derive(Debug, Clone, Copy)]
struct ProductSummary {
    min: i64,
    zeros: u64,
}

#[derive(Debug, Clone, Default)]
struct PivotProducts {
    values: Vec<i64>,
    summaries: Vec<Option<ProductSummary>>,
}

impl PivotProducts {
    const BLOCK: usize = u64::BITS as usize;
    // For a short search, walking the contiguous products is cheaper
    // than maintaining summaries. Amortize them over longer scans.
    const MIN_INDEXED_PRODUCTS: usize = Self::BLOCK * 8 + 2;

    fn new(len: usize) -> SpiceResult<Self> {
        Ok(Self {
            values: crate::primitives::filled_vec(len, 0, "SPARSE Markowitz products")?,
            summaries: if len > Self::MIN_INDEXED_PRODUCTS {
                crate::primitives::filled_vec(
                    len.div_ceil(Self::BLOCK),
                    None,
                    "SPARSE product summaries",
                )?
            } else {
                Vec::new()
            },
        })
    }

    fn set(&mut self, i: usize, value: i64) {
        if self.summaries.is_empty() {
            self.values[i] = value;
        } else if self.values[i] != value {
            self.values[i] = value;
            self.summaries[i / Self::BLOCK] = None;
        }
    }

    fn swap(&mut self, a: usize, b: usize) {
        let (x, y) = (self.values[a], self.values[b]);
        self.set(a, y);
        self.set(b, x);
    }

    fn summary(&mut self, block: usize) -> ProductSummary {
        *self.summaries[block].get_or_insert_with(|| {
            let start = block * Self::BLOCK;
            let values = &self.values[start..(start + Self::BLOCK).min(self.values.len())];
            let mut min = i64::MAX;
            let mut zeros = 0;
            for (i, &v) in values.iter().enumerate() {
                min = min.min(v);
                if v == 0 {
                    zeros |= 1_u64 << i;
                }
            }
            ProductSummary { min, zeros }
        })
    }

    /// Greatest index strictly before the cursor with product below limit.
    /// Sentinels ensure the search has a match in the C pivot routine.
    fn previous_below(&mut self, mut before: usize, limit: i64) -> Option<usize> {
        if self.summaries.is_empty() {
            return self.values[..before].iter().rposition(|&v| v < limit);
        }
        while before != 0 {
            let block = (before - 1) / Self::BLOCK;
            let start = block * Self::BLOCK;
            if (before - start != Self::BLOCK || self.summary(block).min < limit)
                && let Some(pos) = self.values[start..before].iter().rposition(|&v| v < limit)
            {
                return Some(start + pos);
            }
            before = start;
        }
        None
    }

    /// Greatest zero-product index strictly before the cursor.
    fn previous_zero(&mut self, mut before: usize) -> Option<usize> {
        if self.summaries.is_empty() {
            return self.values[..before].iter().rposition(|&v| v == 0);
        }
        while before != 0 {
            let block = (before - 1) / Self::BLOCK;
            let start = block * Self::BLOCK;
            let len = before - start;
            if len == Self::BLOCK {
                let mask = self.summary(block).zeros;
                if mask != 0 {
                    return Some(start + (u64::BITS - 1 - mask.leading_zeros()) as usize);
                }
            } else if let Some(pos) = self.values[start..before].iter().rposition(|&v| v == 0) {
                return Some(start + pos);
            }
            before = start;
        }
        None
    }

    #[cfg(test)]
    fn is_empty(&self) -> bool {
        self.values.is_empty()
    }
}

impl std::ops::Index<usize> for PivotProducts {
    type Output = i64;
    fn index(&self, index: usize) -> &i64 {
        &self.values[index]
    }
}

/// The legal combinations of topology, symbolic ordering and numeric factors.
#[derive(Debug, Clone)]
enum Lifecycle {
    ColumnsOnly,
    Unordered,
    /// Reuse the numbering of a partially changed order after singularity.
    PartialOrder,
    Symbolic(std::sync::Arc<FactorPlan>),
    Factored(std::sync::Arc<FactorPlan>),
    /// Full ordering required, but the old traversal still describes the topology.
    ReorderRequired(std::sync::Arc<FactorPlan>),
}

impl Lifecycle {
    const fn rows_linked(&self) -> bool {
        !matches!(self, Self::ColumnsOnly)
    }

    const fn needs_ordering(&self) -> bool {
        matches!(
            self,
            Self::ColumnsOnly | Self::Unordered | Self::ReorderRequired(_)
        )
    }

    const fn plan(&self) -> Option<&std::sync::Arc<FactorPlan>> {
        match self {
            Self::Symbolic(plan) | Self::Factored(plan) | Self::ReorderRequired(plan) => Some(plan),
            Self::ColumnsOnly | Self::Unordered | Self::PartialOrder => None,
        }
    }

    fn invalidate_numeric(&mut self) {
        if matches!(self, Self::Factored(_))
            && let Self::Factored(plan) = std::mem::replace(self, Self::Unordered)
        {
            *self = Self::Symbolic(plan);
        }
    }

    fn into_loaded(mut self) -> Self {
        self.invalidate_numeric();
        self
    }

    fn invalidate_structure(&mut self, original: bool) {
        *self = match self {
            Self::ColumnsOnly => Self::ColumnsOnly,
            Self::Unordered | Self::ReorderRequired(_) => Self::Unordered,
            _ if original => Self::Unordered,
            _ => Self::PartialOrder,
        };
    }

    fn require_ordering(&mut self) {
        *self = match std::mem::replace(self, Self::Unordered) {
            Self::ColumnsOnly => Self::ColumnsOnly,
            Self::Unordered | Self::PartialOrder => Self::Unordered,
            Self::Symbolic(plan) | Self::Factored(plan) | Self::ReorderRequired(plan) => {
                Self::ReorderRequired(plan)
            }
        };
    }

    fn finish(&mut self, plan: Option<FactorPlan>) -> Result<(), Status> {
        let plan = if let Some(plan) = plan {
            std::sync::Arc::new(plan)
        } else {
            match std::mem::replace(self, Self::Unordered) {
                Self::Symbolic(plan) | Self::Factored(plan) | Self::ReorderRequired(plan) => plan,
                _ => return Err(Status::InvalidPlan),
            }
        };
        *self = Self::Factored(plan);
        Ok(())
    }
}

/// Diagnostic payload of the latest operation.
#[derive(Debug, Clone, Copy)]
enum Diagnostic {
    Okay,
    SmallPivot(usize, usize),
    ZeroDiag(usize, usize),
    Singular(usize, usize),
    NonFinite,
    InvalidPlan,
    NeedsReload,
    Resource,
}

impl Diagnostic {
    const fn status(self) -> Status {
        match self {
            Self::Okay => Status::Okay,
            Self::SmallPivot(..) => Status::SmallPivot,
            Self::ZeroDiag(..) => Status::ZeroDiag,
            Self::Singular(..) => Status::Singular,
            Self::NonFinite => Status::NonFinite,
            Self::InvalidPlan => Status::InvalidPlan,
            Self::NeedsReload => Status::NeedsReload,
            Self::Resource => Status::Resource,
        }
    }

    const fn singular(self) -> (usize, usize) {
        match self {
            Self::ZeroDiag(row, col) | Self::Singular(row, col) => (row, col),
            _ => (0, 0),
        }
    }

    const fn small_pivot(self) -> Option<(usize, usize)> {
        match self {
            Self::SmallPivot(row, col) => Some((row, col)),
            _ => None,
        }
    }
}

/// Scratch for an ordering attempt, returned to the frame for allocation reuse.
/// A snapshot never needs the previous operation's counts or search sentinels.
#[derive(Debug, Default)]
struct Markowitz {
    row: Vec<i64>,
    col: Vec<i64>,
    products: PivotProducts,
    singletons: i64,
}

impl Markowitz {
    fn ensure_size(&mut self, size: usize) -> SpiceResult<()> {
        if self.row.len() <= size {
            let rows = sparse_rows(size)?;
            let row = crate::primitives::filled_vec(rows, 0, "SPARSE Markowitz row counts")?;
            let col = crate::primitives::filled_vec(rows, 0, "SPARSE Markowitz column counts")?;
            let products = PivotProducts::new(rows + 1)?;
            self.row = row;
            self.col = col;
            self.products = products;
        }
        Ok(())
    }

    fn product_of(&self, i: usize) -> i64 {
        self.row[i].saturating_mul(self.col[i])
    }

    const fn track_singleton(&mut self, new: i64, old: i64) {
        if (new == 0) != (old == 0) {
            if old == 0 {
                self.singletons -= 1;
            } else {
                self.singletons += 1;
            }
        }
    }
}

/// A SPARSE matrix frame (`struct MatrixFrame`) with 1-based internal and
/// external indices; external index 0 (ground) is never used here.
#[derive(Debug)]
pub(super) struct Sparse<T> {
    /// Internal order (C `Size`): the rows translated so far. An external
    /// index never referenced by an element has no internal row; as in
    /// ngspice, its solution entry is left holding the right-hand side.
    size: usize,
    /// External order (C `ExtSize`, here fixed by [`Self::new`]).
    ext_size: usize,
    /// Stable handles index topology and numeric storage independently.
    /// Reordering changes links, never moves element handles.
    topology: std::sync::Arc<Topology>,
    values: Vec<T>,
    /// Reciprocals of the raw pivots, indexed by internal row. Only valid
    /// with the numeric factors; reused across unique reload/refactor cycles.
    inverses: Vec<T>,
    lifecycle: Lifecycle,
    ordering_scratch: Markowitz,
    rel_threshold: f64,
    abs_threshold: f64,
    parity: PermutationParity,
    diagnostic: Diagnostic,
    last_attempt: Option<crate::maths::telemetry::FactorAttempt>,
}

impl<T: Clone> Clone for Sparse<T> {
    fn clone(&self) -> Self {
        Self {
            size: self.size,
            ext_size: self.ext_size,
            topology: self.topology.clone(),
            values: self.values.clone(),
            inverses: self.inverses.clone(),
            lifecycle: self.lifecycle.clone(),
            ordering_scratch: Markowitz::default(),
            rel_threshold: self.rel_threshold,
            abs_threshold: self.abs_threshold,
            parity: self.parity,
            diagnostic: self.diagnostic,
            last_attempt: self.last_attempt,
        }
    }
}

impl<T: Scalar> Sparse<T> {
    /// `spCreate(size)`.
    pub(crate) fn new(size: usize) -> SpiceResult<Self> {
        let rows = sparse_rows(size)?;
        let empty = || crate::primitives::filled_vec(rows, NONE, "SPARSE row index");
        let external = || {
            let mut indices = empty()?;
            indices[0] = 0;
            Ok::<_, crate::primitives::SpiceError>(indices)
        };
        Ok(Self {
            size: 0,
            ext_size: size,
            topology: std::sync::Arc::new(Topology {
                original_entries: 0,
                elements: Vec::new(),
                diag: empty()?,
                first_in_col: empty()?,
                first_in_row: empty()?,
                ext_to_int_row: external()?,
                ext_to_int_col: external()?,
                int_to_ext_row: sparse_indices(size)?,
                int_to_ext_col: sparse_indices(size)?,
            }),
            values: Vec::new(),
            inverses: Vec::new(),
            lifecycle: Lifecycle::ColumnsOnly,
            ordering_scratch: Markowitz::default(),
            rel_threshold: DEFAULT_THRESHOLD,
            abs_threshold: 0.0,
            parity: PermutationParity::Even,
            diagnostic: Diagnostic::Okay,
            last_attempt: None,
        })
    }

    /// Whether every external index has been translated (no empty unknown).
    pub(crate) const fn fully_translated(&self) -> bool {
        self.size == self.ext_size
    }

    /// External indices (1-based) that no element references, so have no
    /// internal row (C leaves them out of the matrix; see [`Self::solve_with_workspace`]).
    pub(crate) fn untranslated(&self) -> impl Iterator<Item = usize> + '_ {
        (1..=self.ext_size).filter(|&ext| self.topology.ext_to_int_row[ext] == NONE)
    }

    /// External row and column of the last singular/zero pivot.
    pub(crate) const fn where_singular(&self) -> (usize, usize) {
        self.diagnostic.singular()
    }

    /// External row and column of the first small pivot of the last
    /// ordering (status [`Status::SmallPivot`]), if any.
    pub(crate) const fn small_pivot(&self) -> Option<(usize, usize)> {
        self.diagnostic.small_pivot()
    }

    fn mag(&self, e: usize) -> f64 {
        self.values[e].mag()
    }

    // --- spbuild.c ------------------------------------------------------

    /// `spGetElement(ext_row, ext_col)`: the element handle, created if
    /// missing. Indices are external and 1-based.
    pub(super) fn get_element(&mut self, row: usize, col: usize) -> SpiceResult<SolverElement> {
        self.check_coordinate(row, col)?;
        if self.topology.ext_to_int_row[row] == NONE || self.topology.ext_to_int_col[col] == NONE {
            // Reserve the original's storage before publishing new translations.
            self.reserve_element_storage()?;
        }
        let (row, col) = self.translate(row, col);
        if row == col && self.topology.diag[row] != NONE {
            return Ok(SolverElement(self.topology.diag[row]));
        }
        let (last, found) = self.topology.locate_in_col(Slot::Head(col), row);
        if found != NONE {
            return Ok(SolverElement(found));
        }
        self.create_element(row, col, last, false)
            .map(SolverElement)
    }

    fn check_coordinate(&self, row: usize, col: usize) -> SpiceResult<()> {
        if row == 0 || col == 0 || row > self.ext_size || col > self.ext_size {
            return Err(numerical(
                "SPARSE construction",
                format!(
                    "external coordinate ({row}, {col}) outside 1..={}",
                    self.ext_size
                ),
            ));
        }
        Ok(())
    }

    /// Adds using the stable handle; no coordinate conversion in the load.
    pub(super) fn add(&mut self, element: SolverElement, value: T)
    where
        T: std::ops::Add<Output = T>,
    {
        self.note_numeric_write();
        let v = &mut self.values[element.0];
        *v = *v + value;
    }

    /// `Translate`: assigns internal indices in first-reference order.
    fn translate(&mut self, ext_row: usize, ext_col: usize) -> (usize, usize) {
        let row = self.topology.ext_to_int_row[ext_row];
        let col = self.topology.ext_to_int_col[ext_col];
        if row != NONE && col != NONE {
            return (row, col);
        }
        std::sync::Arc::make_mut(&mut self.topology).translate(&mut self.size, ext_row, ext_col)
    }

    /// `spcFindElementInCol`.
    fn find_element_in_col(&self, last: Slot, row: usize) -> usize {
        self.topology.locate_in_col(last, row).1
    }

    fn reserve_element_storage(&mut self) -> SpiceResult<()> {
        let topology = std::sync::Arc::make_mut(&mut self.topology);
        if topology.elements.len() == topology.elements.capacity() {
            sparse_reserve(&mut topology.elements, 1)?;
        }
        if self.values.len() == self.values.capacity() {
            sparse_reserve(&mut self.values, 1)?;
        }
        Ok(())
    }

    /// `spcCreateElement`. Reserve both arenas before mutating links.
    fn create_element(
        &mut self,
        row: usize,
        col: usize,
        last: Slot,
        fillin: bool,
    ) -> SpiceResult<usize> {
        self.reserve_element_storage()?;
        let topology = std::sync::Arc::make_mut(&mut self.topology);
        let e = topology.create_element(row, col, last, self.lifecycle.rows_linked());
        if !fillin {
            topology.original_entries += 1;
            self.note_numeric_write();
        }
        self.lifecycle.invalidate_structure(!fillin);
        self.values.push(T::ZERO);
        Ok(e)
    }

    /// `spcLinkRows`.
    fn link_rows(&mut self) {
        let topology = std::sync::Arc::make_mut(&mut self.topology);
        for col in (1..=self.size).rev() {
            let mut e = topology.first_in_col[col];
            while e != NONE {
                let row = topology.elements[e].row;
                topology.elements[e].col = col;
                topology.elements[e].next_in_row = topology.first_in_row[row];
                topology.first_in_row[row] = e;
                e = topology.elements[e].next_in_col;
            }
        }
        self.lifecycle = Lifecycle::Unordered;
    }

    /// `spClear`: zero every value, keeping structure and pivot order.
    pub(crate) fn clear(&mut self) {
        self.values.fill(T::ZERO);
        self.reset_numeric_status();
    }

    fn reset_numeric_status(&mut self) {
        self.lifecycle.invalidate_numeric();
        self.diagnostic = Diagnostic::Okay;
        self.last_attempt = None;
    }

    /// A write into existing LU storage revokes its factor view but does
    /// not restore the original coefficients. Likewise, stamping into a
    /// failed partial LU cannot repair it; clear/reload must happen first.
    fn note_numeric_write(&mut self) {
        if self.is_factored() {
            self.lifecycle.invalidate_numeric();
            self.diagnostic = Diagnostic::NeedsReload;
        }
    }

    const fn coefficients_loaded(&self) -> bool {
        !self.is_factored() && matches!(self.diagnostic, Diagnostic::Okay)
    }

    /// Build a fresh frame from unique, already assembled original
    /// elements in reference creation order (external indices are 1-based).
    /// Borrow the exclusively owned topology once and reserve both arenas.
    pub(super) fn from_originals(
        size: usize,
        entries: impl IntoIterator<Item = (usize, usize, T)>,
    ) -> SpiceResult<Self> {
        let mut sparse = Self::new(size)?;
        let entries = entries.into_iter();
        let capacity = entries.size_hint().0;
        let topology = std::sync::Arc::make_mut(&mut sparse.topology);
        sparse_reserve(&mut topology.elements, capacity)?;
        sparse_reserve(&mut sparse.values, capacity)?;
        let mut tails =
            crate::primitives::filled_vec(sparse_rows(size)?, NONE, "SPARSE column tails")?;
        for (row, col, value) in entries {
            if row == 0 || col == 0 || row > size || col > size {
                return Err(numerical(
                    "SPARSE construction",
                    format!("external coordinate ({row}, {col}) outside 1..={size}"),
                ));
            }
            if topology.elements.len() == topology.elements.capacity() {
                sparse_reserve(&mut topology.elements, 1)?;
            }
            if sparse.values.len() == sparse.values.capacity() {
                sparse_reserve(&mut sparse.values, 1)?;
            }
            let (r, c) = topology.translate(&mut sparse.size, row, col);
            let handle = topology.append_original(r, c, &mut tails);
            debug_assert_eq!(handle, sparse.values.len());
            sparse.values.push(value);
        }
        topology.finish_original_columns(&tails[..=sparse.size])?;
        topology.original_entries = topology.elements.len();
        Ok(sparse)
    }

    /// A pattern already contains its unique positions in creation order.
    pub(super) fn from_pattern(pattern: &crate::maths::Pattern, values: &[T]) -> SpiceResult<Self> {
        if pattern.len() != values.len() {
            return Err(numerical(
                "SPARSE construction",
                "pattern and coefficient count differ",
            ));
        }
        Self::from_originals(
            pattern.order(),
            pattern
                .positions()
                .zip(values)
                .map(|((r, c), &value)| (r + 1, c + 1, value)),
        )
    }

    /// Reload consecutive originals, clearing only the fill-in suffix.
    pub(super) fn reload_originals(&mut self, values: &[T]) {
        let (originals, fillins) = self.values.split_at_mut(values.len());
        originals.copy_from_slice(values);
        fillins.fill(T::ZERO);
        self.reset_numeric_status();
    }

    /// Reload from a folded stream without a temporary numeric vector.
    pub(super) fn reload_originals_iter(&mut self, values: impl ExactSizeIterator<Item = T>) {
        let (originals, fillins) = self.values.split_at_mut(values.len());
        for (out, value) in originals.iter_mut().zip(values) {
            *out = value;
        }
        fillins.fill(T::ZERO);
        self.reset_numeric_status();
    }

    /// Transfer one scalar frame's ordering; numeric coefficients are
    /// supplied by the caller, never copied from the old LU.
    fn into_retyped_values<U: Scalar>(self, values: Vec<U>) -> Sparse<U> {
        Sparse {
            size: self.size,
            ext_size: self.ext_size,
            values,
            inverses: Vec::new(),
            topology: self.topology,
            lifecycle: self.lifecycle.into_loaded(),
            ordering_scratch: self.ordering_scratch,
            rel_threshold: self.rel_threshold,
            abs_threshold: self.abs_threshold,
            parity: self.parity,
            diagnostic: Diagnostic::Okay,
            last_attempt: None,
        }
    }

    /// Fork the structural state, discarding scratch and old factors.
    fn fork_values<U: Scalar>(&self, values: Vec<U>) -> Sparse<U> {
        Sparse {
            size: self.size,
            ext_size: self.ext_size,
            topology: self.topology.clone(),
            values,
            inverses: Vec::new(),
            lifecycle: self.lifecycle.clone().into_loaded(),
            ordering_scratch: Markowitz::default(),
            rel_threshold: self.rel_threshold,
            abs_threshold: self.abs_threshold,
            parity: self.parity,
            diagnostic: Diagnostic::Okay,
            last_attempt: None,
        }
    }

    /// Construct the complete numeric buffer directly. The original
    /// handles are consecutive; only the fill-in suffix needs zeroing.
    fn loaded_values<U: Scalar>(&self, originals: impl ExactSizeIterator<Item = U>) -> Vec<U> {
        let total = self.values.len();
        assert!(
            originals.len() <= total,
            "more originals than sparse elements"
        );
        let mut values = Vec::with_capacity(total);
        values.extend(originals);
        values.resize(total, U::ZERO);
        values
    }

    pub(super) fn fork_loaded<U: Scalar>(
        &self,
        originals: impl ExactSizeIterator<Item = U>,
    ) -> Sparse<U> {
        let mut next = self.fork_values(self.loaded_values(originals));
        next.reset_numeric_status();
        next
    }

    /// Reload in place if the factor buffer is uniquely held, otherwise
    /// fork directly from the newly assembled values. No old LU or blanket
    /// zero pass is needed on the retained-snapshot path.
    pub(super) fn reload_shared(
        shared: &mut std::sync::Arc<Self>,
        originals: impl ExactSizeIterator<Item = T>,
    ) {
        if let Some(owned) = std::sync::Arc::get_mut(shared) {
            owned.reload_originals_iter(originals);
        } else {
            *shared = std::sync::Arc::new(shared.fork_loaded(originals));
        }
    }

    /// Switch scalar type, moving unique ordering storage or sharing it
    /// with a retained snapshot. The blank form is used before matching a
    /// real matrix against the cached order.
    pub(super) fn retype_shared<U: Scalar>(shared: std::sync::Arc<Self>) -> Sparse<U> {
        let values = vec![U::ZERO; shared.values.len()];
        match std::sync::Arc::try_unwrap(shared) {
            Ok(owned) => owned.into_retyped_values(values),
            Err(retained) => retained.fork_values(values),
        }
    }

    /// Retype and load in a single pass when the new assembly is known.
    pub(super) fn retype_shared_loaded<U: Scalar>(
        shared: std::sync::Arc<Self>,
        originals: impl ExactSizeIterator<Item = U>,
    ) -> Sparse<U> {
        let values = shared.loaded_values(originals);
        let mut next = match std::sync::Arc::try_unwrap(shared) {
            Ok(owned) => owned.into_retyped_values(values),
            Err(retained) => retained.fork_values(values),
        };
        next.reset_numeric_status();
        next
    }

    #[cfg(test)]
    fn into_retyped<U: Scalar>(self) -> Sparse<U> {
        let values = vec![U::ZERO; self.values.len()];
        self.into_retyped_values(values)
    }

    /// `LoadGmin` (`spsmp.c`): adds `gmin` to the real part of every
    /// internal diagonal element (after the MNA preorder's column swaps, so a
    /// voltage source's twin can be one), from the last row to the first.
    pub(crate) fn load_gmin(&mut self, gmin: f64) {
        if gmin == 0.0 {
            return;
        }
        self.note_numeric_write();
        for i in (1..=self.size).rev() {
            let diag = self.topology.diag[i];
            if diag != NONE {
                self.values[diag] = self.values[diag].add_real(gmin);
            }
        }
    }

    pub(super) fn values_are_finite(&self) -> bool {
        self.values.iter().all(|value| value.is_finite())
    }

    /// True only when a direct numerical refactor can use the established
    /// pivot order, rather than implicitly running the ordering algorithm.
    pub(super) const fn can_refactor(&self) -> bool {
        matches!(
            self.lifecycle,
            Lifecycle::Symbolic(_) | Lifecycle::Factored(_)
        )
    }

    /// Whether the frame holds LU factors (`Matrix->Factored`).
    pub(crate) const fn is_factored(&self) -> bool {
        matches!(self.lifecycle, Lifecycle::Factored(_))
    }

    /// Overwrites one element's value (`*element = value`).
    pub(super) fn set(&mut self, element: SolverElement, value: T) {
        self.note_numeric_write();
        self.values[element.0] = value;
    }

    /// One element's value.
    pub(super) fn value(&self, element: SolverElement) -> T {
        self.values[element.0]
    }

    /// `SMPcAddCol` (`spsmp.c`): adds every element of external column
    /// `addend` (fill-ins included) to the same row of column `accum`,
    /// creating the missing elements there as originals (which, once rows
    /// are linked, forces the next factorization to reorder). Untranslated
    /// columns are left alone.
    pub(crate) fn add_col(&mut self, accum: usize, addend: usize) -> SpiceResult<()>
    where
        T: std::ops::Add<Output = T>,
    {
        self.check_coordinate(accum, addend)?;
        let (accum, addend) = (
            self.topology.ext_to_int_col[accum],
            self.topology.ext_to_int_col[addend],
        );
        if accum == NONE || addend == NONE {
            return Ok(());
        }
        let mut next = self.topology.first_in_col[addend];
        let mut prev = Slot::Head(accum);
        let mut acc = self.topology.col_get(prev);
        while next != NONE {
            let row = self.topology.elements[next].row;
            while acc != NONE && self.topology.elements[acc].row < row {
                prev = Slot::After(acc);
                acc = self.topology.elements[acc].next_in_col;
            }
            if acc == NONE || self.topology.elements[acc].row > row {
                acc = self.create_element(row, accum, prev, false)?;
            }
            let value = self.values[next];
            self.add(SolverElement(acc), value);
            next = self.topology.elements[next].next_in_col;
        }
        Ok(())
    }

    /// `SMPcZeroCol` (`spsmp.c`): zeroes every element of external column
    /// `col`, keeping them in the structure.
    pub(crate) fn zero_col(&mut self, col: usize) -> SpiceResult<()> {
        self.check_coordinate(col, col)?;
        let col = self.topology.ext_to_int_col[col];
        if col == NONE {
            return Ok(());
        }
        self.note_numeric_write();
        let mut e = self.topology.first_in_col[col];
        while e != NONE {
            self.values[e] = T::ZERO;
            e = self.topology.elements[e].next_in_col;
        }
        Ok(())
    }

    /// `spMNA_Preorder`: swap columns so structural `±1` twins from voltage
    /// sources land on the diagonal before ordering.
    pub(crate) fn mna_preorder(&mut self) {
        if self.lifecycle.rows_linked() {
            return;
        }
        let size = self.size;
        let mut start_at = 1;
        loop {
            let mut another_pass = false;
            let mut swapped = false;
            // C's `StartAt` changes only take effect on the next pass.
            // Snapshot the range bound so changes within the pass cannot
            // affect its current iteration range.
            let pass_start = start_at;
            for j in pass_start..=size {
                if self.topology.diag[j] == NONE {
                    let (twins, t1, t2) = self.count_twins(j);
                    if twins == 1 {
                        self.swap_cols(t1, t2);
                        swapped = true;
                    } else if twins > 1 && !another_pass {
                        another_pass = true;
                        start_at = j;
                    }
                }
            }
            if another_pass {
                let mut j = start_at;
                while !swapped && j <= size {
                    if self.topology.diag[j] == NONE {
                        let (_, t1, t2) = self.count_twins(j);
                        self.swap_cols(t1, t2);
                        swapped = true;
                    }
                    j += 1;
                }
            }
            if !another_pass {
                break;
            }
        }
    }

    /// `CountTwins`: the twin pair found is the last one before a second.
    fn count_twins(&mut self, col: usize) -> (usize, usize, usize) {
        let topology = std::sync::Arc::make_mut(&mut self.topology);
        let mut twins = 0;
        let (mut out1, mut out2) = (NONE, NONE);
        let mut t1 = topology.first_in_col[col];
        while t1 != NONE {
            if self.values[t1].real().abs() == 1.0 {
                let row = topology.elements[t1].row;
                let mut t2 = topology.first_in_col[row];
                while t2 != NONE && topology.elements[t2].row != col {
                    t2 = topology.elements[t2].next_in_col;
                }
                if t2 != NONE && self.values[t2].real().abs() == 1.0 {
                    twins += 1;
                    if twins >= 2 {
                        return (twins, out1, out2);
                    }
                    topology.elements[t1].col = col;
                    topology.elements[t2].col = row;
                    out1 = t1;
                    out2 = t2;
                }
            }
            t1 = topology.elements[t1].next_in_col;
        }
        (twins, out1, out2)
    }

    /// `SwapCols`.
    fn swap_cols(&mut self, t1: usize, t2: usize) {
        let topology = std::sync::Arc::make_mut(&mut self.topology);
        let (c1, c2) = (topology.elements[t1].col, topology.elements[t2].col);
        topology.first_in_col.swap(c1, c2);
        topology.int_to_ext_col.swap(c1, c2);
        topology.ext_to_int_col[topology.int_to_ext_col[c2]] = c2;
        topology.ext_to_int_col[topology.int_to_ext_col[c1]] = c1;
        topology.diag[c1] = t2;
        topology.diag[c2] = t1;
        self.parity.toggle();
    }

    // --- spfactor.c -----------------------------------------------------

    /// `spOrderAndFactor(Matrix, NULL, rel, abs, DiagPivoting = YES)`.
    pub(crate) fn order_and_factor(&mut self, rel_threshold: f64, abs_threshold: f64) -> Status {
        use crate::maths::telemetry::{AttemptPath, FactorAttempt};

        let mut searched = false;
        let status = if self.coefficients_loaded() {
            let mut scratch = std::mem::take(&mut self.ordering_scratch);
            let status =
                self.order_with_scratch(rel_threshold, abs_threshold, &mut scratch, &mut searched);
            self.ordering_scratch = scratch;
            status
        } else {
            Status::NeedsReload
        };
        self.last_attempt = Some(FactorAttempt {
            path: AttemptPath::Order { searched },
            status,
        });
        status
    }

    /// The most recent actual factor attempt, not a lifetime count.
    pub(super) fn last_attempt(&self) -> crate::maths::FactorWork {
        let mut work = crate::maths::FactorWork::default();
        if let Some(attempt) = self.last_attempt {
            work.record(attempt);
        }
        work
    }

    /// Storage of this frame; callers expose it only for valid factors.
    pub(super) fn storage(&self) -> crate::maths::FactorStorage {
        crate::maths::FactorStorage {
            external_order: self.ext_size,
            translated_order: self.size,
            original_entries: self.topology.original_entries,
            factor_entries: self.values.len(),
            numeric_bytes: std::mem::size_of_val(self.values.as_slice())
                + std::mem::size_of_val(self.inverses.as_slice()),
        }
    }

    fn prepare_inverses(&mut self) -> SpiceResult<()> {
        let required = self.size + 1;
        let additional = required.saturating_sub(self.inverses.len());
        sparse_reserve(&mut self.inverses, additional)?;
        self.inverses.resize(required, T::ZERO);
        Ok(())
    }

    fn order_with_scratch(
        &mut self,
        rel_threshold: f64,
        abs_threshold: f64,
        scratch: &mut Markowitz,
        searched: &mut bool,
    ) -> Status {
        if !self.values_are_finite() {
            return self.nonfinite_factor();
        }
        if self.prepare_inverses().is_err() {
            return self.resource_failure();
        }
        let size = self.size;
        if rel_threshold > 0.0 && rel_threshold <= 1.0 {
            self.rel_threshold = rel_threshold;
        }
        if abs_threshold >= 0.0 {
            self.abs_threshold = abs_threshold;
        }
        let mut step = 1;
        let mut reorder = false;
        if self.lifecycle.needs_ordering() {
            if !self.lifecycle.rows_linked() {
                self.link_rows();
            }
        } else {
            // A failed partial reorder may have invalidated the plan.
            // Retain its numbering like C. The accepted prefix needs
            // no Markowitz counts; compute them only if a search begins.
            while step <= size {
                let pivot = self.topology.diag[step];
                if pivot == NONE {
                    reorder = true;
                    break;
                }
                let largest = self.find_largest_in_col(self.topology.elements[pivot].next_in_col);
                if largest * self.rel_threshold < self.mag(pivot) {
                    self.row_col_elimination(None, pivot);
                    if matches!(
                        self.diagnostic.status(),
                        Status::Singular | Status::NonFinite | Status::Resource
                    ) {
                        return self.diagnostic.status();
                    }
                } else {
                    reorder = true;
                    break;
                }
                step += 1;
            }
            if !reorder {
                return self.done();
            }
        }
        // Reordering always rebuilds all active counts. A snapshot fork
        // needs no copy of the previous ordering's scratch arrays.
        if scratch.ensure_size(size).is_err() {
            return self.resource_failure();
        }
        *searched = step <= size;
        self.count_markowitz(scratch, step);
        self.markowitz_products(scratch, step);
        while step <= size {
            let Some(pivot) = self.search_for_pivot(scratch, step) else {
                return self.matrix_is_singular(step);
            };
            self.exchange_rows_and_cols(scratch, pivot, step);
            self.row_col_elimination(Some(scratch), pivot);
            if matches!(
                self.diagnostic.status(),
                Status::Singular | Status::NonFinite | Status::Resource
            ) {
                return self.diagnostic.status();
            }
            self.update_markowitz_numbers(scratch, pivot);
            step += 1;
        }
        self.done()
    }

    fn done(&mut self) -> Status {
        let new = if self.lifecycle.plan().is_none() {
            match self.compile_plan() {
                Ok(plan) => Some(plan),
                Err(Status::Resource) => return self.resource_failure(),
                Err(_) => return self.invalid_plan(),
            }
        } else {
            None
        };
        if self.lifecycle.finish(new).is_err() {
            return self.invalid_plan();
        }
        self.diagnostic.status()
    }

    fn invalid_plan(&mut self) -> Status {
        self.lifecycle.require_ordering();
        self.diagnostic = Diagnostic::InvalidPlan;
        Status::InvalidPlan
    }

    fn compile_plan(&self) -> Result<FactorPlan, Status> {
        // Offsets include the exclusive end of the entry array, so both
        // counts (not just the largest existing handles) must fit.
        if u32::try_from(self.size).is_ok() && u32::try_from(self.topology.elements.len()).is_ok() {
            self.compile_runs().map(FactorPlan::Compact)
        } else {
            self.compile_runs().map(FactorPlan::Wide)
        }
    }

    fn compile_runs<I: PlanIndex>(&self) -> Result<FactorRuns<I>, Status> {
        let elements = &self.topology.elements;
        if self.values.len() != elements.len() || !self.lifecycle.rows_linked() {
            return Err(Status::InvalidPlan);
        }
        Ok(FactorRuns {
            columns: self.compile_axis(false)?,
            rows: self.compile_axis(true)?,
        })
    }

    fn compile_axis<I: PlanIndex>(&self, rows: bool) -> Result<AxisRuns<I>, Status> {
        let elements = &self.topology.elements;
        let mut runs = Vec::new();
        runs.try_reserve_exact(self.size)
            .map_err(|_| Status::Resource)?;
        let mut entries = Vec::new();
        entries
            .try_reserve_exact(elements.len())
            .map_err(|_| Status::Resource)?;
        let heads = if rows {
            &self.topology.first_in_row
        } else {
            &self.topology.first_in_col
        };
        for i in 1..=self.size {
            let start = entries.len();
            let diagonal = *self.topology.diag.get(i).ok_or(Status::InvalidPlan)?;
            let mut diagonal_offset = None;
            let mut e = *heads.get(i).ok_or(Status::InvalidPlan)?;
            let mut previous = 0;
            while e != NONE {
                let element = elements.get(e).ok_or(Status::InvalidPlan)?;
                let (axis, coordinate, next) = if rows {
                    (element.row, element.col, element.next_in_row)
                } else {
                    (element.col, element.row, element.next_in_col)
                };
                // Strict coordinate order excludes cycles and duplicates.
                if axis != i || coordinate <= previous || coordinate > self.size {
                    return Err(Status::InvalidPlan);
                }
                previous = coordinate;
                if coordinate == i {
                    if e != diagonal {
                        return Err(Status::InvalidPlan);
                    }
                    diagonal_offset = Some(entries.len());
                }
                if entries.len() == elements.len() {
                    return Err(Status::InvalidPlan);
                }
                entries.push(RunEntry::new(e, coordinate).ok_or(Status::InvalidPlan)?);
                e = next;
            }
            runs.push(
                AxisRun::new(
                    start,
                    diagonal_offset.ok_or(Status::InvalidPlan)?,
                    entries.len(),
                )
                .ok_or(Status::InvalidPlan)?,
            );
        }
        if entries.len() != elements.len() {
            return Err(Status::InvalidPlan);
        }
        Ok(AxisRuns { runs, entries })
    }

    #[cfg(test)]
    fn factor(&mut self) -> Status {
        self.factor_with_workspace(&mut Vec::new())
    }

    /// `spFactor`: refactor with the existing pivot order (no threshold
    /// test). Scratch belongs to the operation, not to factor snapshots.
    pub(super) fn factor_with_workspace(&mut self, workspace: &mut Vec<T>) -> Status {
        use crate::maths::telemetry::{AttemptPath, FactorAttempt};

        // C retains a partial order after a singular reordering. If
        // topology changed, no complete numeric traversal is available;
        // retry the threshold path, preserving C's existing pivot prefix.
        if !self.coefficients_loaded() {
            self.last_attempt = Some(FactorAttempt {
                path: AttemptPath::Rejected,
                status: Status::NeedsReload,
            });
            return Status::NeedsReload;
        }
        if !matches!(self.lifecycle, Lifecycle::Symbolic(_)) {
            return self.order_and_factor(0.0, 0.0);
        }
        if self.prepare_inverses().is_err() {
            let status = self.resource_failure();
            self.last_attempt = Some(FactorAttempt {
                path: AttemptPath::Refactor,
                status,
            });
            return status;
        }
        let status = self.refactor_with_workspace(workspace);
        self.last_attempt = Some(FactorAttempt {
            path: AttemptPath::Refactor,
            status,
        });
        status
    }

    fn refactor_with_workspace(&mut self, workspace: &mut Vec<T>) -> Status {
        let Lifecycle::Symbolic(plan) = &self.lifecycle else {
            return self.invalid_plan();
        };
        let size = self.size;
        if size == 0 {
            return self.done();
        }
        let first = self.topology.diag[1];
        if self.values[first].is_zero() {
            return self.refactor_zero_pivot(1);
        }
        let first_pivot = self.values[first];
        if !first_pivot.is_finite() {
            return self.nonfinite_factor();
        }
        self.inverses[1] = first_pivot.cached_recip();
        // Each column is scattered before use. Successful ordering
        // inserted every structural elimination target as a fill-in, so
        // the lower-column updates cannot read an unscattered row.
        // Existing scratch may hold the previous solve's internal RHS;
        // no blanket initialization is required.
        let required = size + 1;
        let additional = required.saturating_sub(workspace.len());
        if sparse_reserve(workspace, additional).is_err() {
            return self.resource_failure();
        }
        workspace.resize(size + 1, T::ZERO);
        let result = match plan.as_ref() {
            FactorPlan::Compact(runs) => runs.refactor(
                &mut self.values,
                &self.topology.diag,
                workspace,
                &mut self.inverses,
            ),
            FactorPlan::Wide(runs) => runs.refactor(
                &mut self.values,
                &self.topology.diag,
                workspace,
                &mut self.inverses,
            ),
        };
        match result {
            Ok(()) => {}
            Err(RefactorIssue::ZeroPivot(step)) => return self.refactor_zero_pivot(step),
            Err(RefactorIssue::NonFinite) => return self.nonfinite_factor(),
        }
        // Fixed-order factorization does not compare coefficient magnitudes
        // to select pivots. Validate the stored factors in one contiguous
        // pass, including the untouched subdiagonal entries of column one.
        // Every computed entry is stored; a nonfinite pivot is rejected above.
        if !self.values_are_finite() {
            return self.nonfinite_factor();
        }
        self.done()
    }

    fn refactor_zero_pivot(&mut self, step: usize) -> Status {
        // Preserve the distinction between singular and nonfinite input
        // even when an early zero pivot stops the factorization.
        if !self.values_are_finite() {
            return self.nonfinite_factor();
        }
        self.zero_pivot(step)
    }

    /// `CountMarkowitz` (no RHS).
    fn count_markowitz(&self, scratch: &mut Markowitz, step: usize) {
        for i in step..=self.size {
            let mut count = -1;
            let mut e = self.topology.first_in_row[i];
            while e != NONE && self.topology.elements[e].col < step {
                e = self.topology.elements[e].next_in_row;
            }
            while e != NONE {
                count += 1;
                e = self.topology.elements[e].next_in_row;
            }
            scratch.row[i] = count;
        }
        for i in step..=self.size {
            let mut count = -1;
            let mut e = self.topology.first_in_col[i];
            while e != NONE && self.topology.elements[e].row < step {
                e = self.topology.elements[e].next_in_col;
            }
            while e != NONE {
                count += 1;
                e = self.topology.elements[e].next_in_col;
            }
            scratch.col[i] = count;
        }
    }

    /// `MarkowitzProducts`.
    fn markowitz_products(&self, scratch: &mut Markowitz, step: usize) {
        scratch.singletons = 0;
        for i in step..=self.size {
            let product = scratch.row[i].saturating_mul(scratch.col[i]);
            scratch.products.set(i, product);
            if product == 0 {
                scratch.singletons += 1;
            }
        }
    }

    /// `SearchForPivot` with diagonal pivoting.
    fn search_for_pivot(&mut self, scratch: &mut Markowitz, step: usize) -> Option<usize> {
        if scratch.singletons != 0
            && let Some(pivot) = self.search_for_singleton(scratch, step)
        {
            return Some(pivot);
        }
        if let Some(pivot) = self.quickly_search_diagonal(scratch, step) {
            return Some(pivot);
        }
        if let Some(pivot) = self.search_diagonal(scratch, step) {
            return Some(pivot);
        }
        self.search_entire_matrix(scratch, step)
    }

    fn acceptable(&self, pivot: usize, step: usize) -> bool {
        let mag = self.mag(pivot);
        mag > self.abs_threshold
            && mag > self.rel_threshold * self.find_biggest_in_col_exclude(pivot, step)
    }

    /// `SearchForSingleton`.
    fn search_for_singleton(&self, scratch: &mut Markowitz, step: usize) -> Option<usize> {
        let size = self.size;
        scratch.products.set(size + 1, scratch.products[step]);
        let mut singletons = scratch.singletons;
        scratch.singletons -= 1;
        scratch.products.set(step - 1, 0);
        // C advances a pointer *after* testing each product. Track the
        // next index plus one instead: the first read is size + 1 and each
        // subsequent read moves left, preserving C's sentinel and pivot
        // priority without an isize that can become -1 at step == 1.
        let mut cursor = size + 2;
        while singletons > 0 {
            singletons -= 1;
            let Some(next) = scratch.products.previous_zero(cursor) else {
                break;
            };
            cursor = next;
            let mut i = cursor;
            if i < step {
                break;
            }
            if i > size {
                i = step;
            }
            let diag = self.topology.diag[i];
            if diag == NONE {
                // ngspice's off-diagonal singleton branches `break` (or would
                // dereference NULL) when no diagonal exists.
                break;
            }
            if self.acceptable(diag, step) {
                return Some(diag);
            }
        }
        scratch.singletons += 1;
        None
    }

    /// `QuicklySearchDiagonal` (without `MODIFIED_MARKOWITZ`).
    fn quickly_search_diagonal(&self, scratch: &mut Markowitz, step: usize) -> Option<usize> {
        let size = self.size;
        let mut chosen = NONE;
        let mut min = i64::MAX;
        scratch.products.set(size + 1, scratch.products[step]);
        scratch.products.set(step - 1, -1);
        let mut p = size + 2;
        while let Some(next) = scratch.products.previous_below(p, min) {
            p = next;
            let mut i = p;
            if i < step {
                break;
            }
            if i > size {
                i = step;
            }
            let diag = self.topology.diag[i];
            if diag == NONE {
                continue;
            }
            let magnitude = self.mag(diag);
            if magnitude <= self.abs_threshold {
                continue;
            }
            if scratch.products[p] == 1 {
                let mut other_row = self.topology.elements[diag].next_in_row;
                let mut other_col = self.topology.elements[diag].next_in_col;
                if other_row == NONE && other_col == NONE {
                    other_row = self.topology.first_in_row[i];
                    while other_row != NONE {
                        let c = self.topology.elements[other_row].col;
                        if c >= step && c != i {
                            break;
                        }
                        other_row = self.topology.elements[other_row].next_in_row;
                    }
                    other_col = self.topology.first_in_col[i];
                    while other_col != NONE {
                        let r = self.topology.elements[other_col].row;
                        if r >= step && r != i {
                            break;
                        }
                        other_col = self.topology.elements[other_col].next_in_col;
                    }
                }
                if other_row != NONE
                    && other_col != NONE
                    && self.topology.elements[other_row].col
                        == self.topology.elements[other_col].row
                {
                    let largest_off = self.mag(other_row).max(self.mag(other_col));
                    if magnitude >= largest_off {
                        return Some(diag);
                    }
                }
            }
            min = scratch.products[p];
            chosen = diag;
        }
        if chosen != NONE {
            let largest = self.find_biggest_in_col_exclude(chosen, step);
            if self.mag(chosen) <= self.rel_threshold * largest {
                return None;
            }
            return Some(chosen);
        }
        None
    }

    /// `SearchDiagonal`.
    fn search_diagonal(&self, scratch: &mut Markowitz, step: usize) -> Option<usize> {
        let size = self.size;
        let mut chosen = NONE;
        let mut min = i64::MAX;
        let mut ties = 0;
        let mut accepted_ratio = 0.0;
        scratch.products.set(size + 1, scratch.products[step]);
        for j in (step + 1..=size + 1).rev() {
            let product = scratch.products[j];
            if product > min {
                continue;
            }
            let i = if j > size { step } else { j };
            let diag = self.topology.diag[i];
            if diag == NONE {
                continue;
            }
            let magnitude = self.mag(diag);
            if magnitude <= self.abs_threshold {
                continue;
            }
            let largest = self.find_biggest_in_col_exclude(diag, step);
            if magnitude <= self.rel_threshold * largest {
                continue;
            }
            if product < min {
                chosen = diag;
                min = product;
                accepted_ratio = largest / magnitude;
                ties = 0;
            } else {
                ties += 1;
                let ratio = largest / magnitude;
                if ratio < accepted_ratio {
                    chosen = diag;
                    accepted_ratio = ratio;
                }
                if ties >= min.saturating_mul(TIES_MULTIPLIER) {
                    return Some(chosen);
                }
            }
        }
        (chosen != NONE).then_some(chosen)
    }

    /// `SearchEntireMatrix`. Sets `SmallPivot`/`Singular` like C.
    fn search_entire_matrix(&mut self, scratch: &Markowitz, step: usize) -> Option<usize> {
        let mut chosen = NONE;
        let mut largest_element = NONE;
        let mut largest_mag = 0.0;
        let mut min = i64::MAX;
        let mut ties = 0;
        let mut accepted_ratio = 0.0;
        for i in step..=self.size {
            let mut e = self.topology.first_in_col[i];
            while e != NONE && self.topology.elements[e].row < step {
                e = self.topology.elements[e].next_in_col;
            }
            let largest_in_col = self.find_largest_in_col(e);
            if largest_in_col == 0.0 {
                continue;
            }
            while e != NONE {
                let magnitude = self.mag(e);
                if magnitude > largest_mag {
                    largest_mag = magnitude;
                    largest_element = e;
                }
                let element = &self.topology.elements[e];
                let product = scratch.row[element.row].saturating_mul(scratch.col[element.col]);
                if product <= min
                    && magnitude > self.rel_threshold * largest_in_col
                    && magnitude > self.abs_threshold
                {
                    if product < min {
                        chosen = e;
                        min = product;
                        accepted_ratio = largest_in_col / magnitude;
                        ties = 0;
                    } else {
                        ties += 1;
                        let ratio = largest_in_col / magnitude;
                        if ratio < accepted_ratio {
                            chosen = e;
                            accepted_ratio = ratio;
                        }
                        if ties >= min.saturating_mul(TIES_MULTIPLIER) {
                            return Some(chosen);
                        }
                    }
                }
                e = self.topology.elements[e].next_in_col;
            }
        }
        if chosen != NONE {
            return Some(chosen);
        }
        if largest_mag == 0.0 {
            return None;
        }
        if self.diagnostic.small_pivot().is_none() {
            let element = &self.topology.elements[largest_element];
            self.diagnostic = Diagnostic::SmallPivot(
                self.topology.int_to_ext_row[element.row],
                self.topology.int_to_ext_col[element.col],
            );
        }
        Some(largest_element)
    }

    /// `FindLargestInCol` from `e` downwards.
    fn find_largest_in_col(&self, mut e: usize) -> f64 {
        let mut largest = 0.0;
        while e != NONE {
            let magnitude = self.mag(e);
            if magnitude > largest {
                largest = magnitude;
            }
            e = self.topology.elements[e].next_in_col;
        }
        largest
    }

    /// `FindBiggestInColExclude`.
    fn find_biggest_in_col_exclude(&self, pivot: usize, step: usize) -> f64 {
        let row = self.topology.elements[pivot].row;
        let col = self.topology.elements[pivot].col;
        let mut e = self.topology.first_in_col[col];
        while e != NONE && self.topology.elements[e].row < step {
            e = self.topology.elements[e].next_in_col;
        }
        let mut largest = if self.topology.elements[e].row == row {
            0.0
        } else {
            self.mag(e)
        };
        e = self.topology.elements[e].next_in_col;
        while e != NONE {
            let magnitude = self.mag(e);
            if magnitude > largest && self.topology.elements[e].row != row {
                largest = magnitude;
            }
            e = self.topology.elements[e].next_in_col;
        }
        largest
    }

    /// `ExchangeRowsAndCols`.
    fn exchange_rows_and_cols(&mut self, scratch: &mut Markowitz, pivot: usize, step: usize) {
        let row = self.topology.elements[pivot].row;
        let col = self.topology.elements[pivot].col;
        if row == step && col == step {
            return;
        }
        self.lifecycle.invalidate_structure(false);
        if row == col {
            self.row_exchange(scratch, step, row);
            self.col_exchange(scratch, step, col);
            scratch.products.swap(step, row);
            std::sync::Arc::make_mut(&mut self.topology)
                .diag
                .swap(row, step);
        } else {
            let old_step = scratch.products[step];
            let old_row = scratch.products[row];
            let old_col = scratch.products[col];
            if row != step {
                self.row_exchange(scratch, step, row);
                self.parity.toggle();
                scratch.products.set(row, scratch.product_of(row));
                scratch.track_singleton(scratch.products[row], old_row);
            }
            if col != step {
                self.col_exchange(scratch, step, col);
                self.parity.toggle();
                scratch.products.set(col, scratch.product_of(col));
                scratch.track_singleton(scratch.products[col], old_col);
                std::sync::Arc::make_mut(&mut self.topology).diag[col] =
                    self.find_element_in_col(Slot::Head(col), col);
            }
            if row != step {
                std::sync::Arc::make_mut(&mut self.topology).diag[row] =
                    self.find_element_in_col(Slot::Head(row), row);
            }
            std::sync::Arc::make_mut(&mut self.topology).diag[step] =
                self.find_element_in_col(Slot::Head(step), step);
            scratch.products.set(step, scratch.product_of(step));
            scratch.track_singleton(scratch.products[step], old_step);
        }
    }

    /// Exchange rows in the linked topology and Markowitz counts.
    fn row_exchange(&mut self, scratch: &mut Markowitz, a: usize, b: usize) {
        std::sync::Arc::make_mut(&mut self.topology).row_exchange(a, b);
        scratch.row.swap(a, b);
    }

    fn col_exchange(&mut self, scratch: &mut Markowitz, a: usize, b: usize) {
        std::sync::Arc::make_mut(&mut self.topology).col_exchange(a, b);
        scratch.col.swap(a, b);
    }

    /// `RealRowColElimination` / `ComplexRowColElimination`.
    fn row_col_elimination(&mut self, mut scratch: Option<&mut Markowitz>, pivot: usize) {
        if self.values[pivot].mag() == 0.0 {
            let row = self.topology.elements[pivot].row;
            self.matrix_is_singular(row);
            return;
        }
        let raw_pivot = self.values[pivot];
        if !raw_pivot.is_finite() {
            self.nonfinite_factor();
            return;
        }
        let inverse = raw_pivot.cached_recip();
        self.inverses[self.topology.elements[pivot].row] = inverse;
        let mut upper = self.topology.elements[pivot].next_in_row;
        while upper != NONE {
            let raw_upper = self.values[upper];
            let u = raw_upper.quotient(raw_pivot, inverse);
            let normal = u.normal_multiplier();
            let upper_col = self.topology.elements[upper].col;
            let mut above = upper;
            let mut sub = self.topology.elements[upper].next_in_col;
            let mut lower = self.topology.elements[pivot].next_in_col;
            while lower != NONE {
                let row = self.topology.elements[lower].row;
                while sub != NONE && self.topology.elements[sub].row < row {
                    above = sub;
                    sub = self.topology.elements[sub].next_in_col;
                }
                if sub == NONE || self.topology.elements[sub].row > row {
                    let Ok(element) = self.create_fillin(
                        scratch.as_deref_mut(),
                        row,
                        upper_col,
                        Slot::After(above),
                    ) else {
                        self.resource_failure();
                        return;
                    };
                    sub = element;
                }
                let l = self.values[lower];
                let value = self.values[sub].schur(raw_pivot, raw_upper, u, l, normal);
                if !value.is_finite() {
                    self.nonfinite_factor();
                    return;
                }
                self.values[sub] = value;
                above = sub;
                sub = self.topology.elements[sub].next_in_col;
                lower = self.topology.elements[lower].next_in_col;
            }
            upper = self.topology.elements[upper].next_in_row;
        }
    }

    /// `UpdateMarkowitzNumbers`.
    fn update_markowitz_numbers(&self, scratch: &mut Markowitz, pivot: usize) {
        let mut e = self.topology.elements[pivot].next_in_col;
        while e != NONE {
            let row = self.topology.elements[e].row;
            scratch.row[row] -= 1;
            scratch.products.set(row, scratch.product_of(row));
            if scratch.row[row] == 0 {
                scratch.singletons += 1;
            }
            e = self.topology.elements[e].next_in_col;
        }
        let mut e = self.topology.elements[pivot].next_in_row;
        while e != NONE {
            let col = self.topology.elements[e].col;
            scratch.col[col] -= 1;
            scratch.products.set(col, scratch.product_of(col));
            if scratch.col[col] == 0 && scratch.row[col] != 0 {
                scratch.singletons += 1;
            }
            e = self.topology.elements[e].next_in_row;
        }
    }

    /// `CreateFillin`: the elimination merge has already located the
    /// exact predecessor in its sorted column. Do not rescan its prefix.
    fn create_fillin(
        &mut self,
        scratch: Option<&mut Markowitz>,
        row: usize,
        col: usize,
        above: Slot,
    ) -> SpiceResult<usize> {
        debug_assert!(matches!(above, Slot::After(e) if self.topology.elements[e].row < row));
        debug_assert!({
            let next = self.topology.col_get(above);
            next == NONE || self.topology.elements[next].row > row
        });
        let element = self.create_element(row, col, above, true)?;
        if let Some(scratch) = scratch {
            scratch.row[row] += 1;
            scratch.products.set(row, scratch.product_of(row));
            if scratch.row[row] == 1 && scratch.col[row] != 0 {
                scratch.singletons -= 1;
            }
            scratch.col[col] += 1;
            scratch.products.set(col, scratch.product_of(col));
            if scratch.row[col] != 0 && scratch.col[col] == 1 {
                scratch.singletons -= 1;
            }
        }
        Ok(element)
    }

    fn resource_failure(&mut self) -> Status {
        self.lifecycle.require_ordering();
        self.diagnostic = Diagnostic::Resource;
        Status::Resource
    }

    fn nonfinite_factor(&mut self) -> Status {
        self.lifecycle.require_ordering();
        self.diagnostic = Diagnostic::NonFinite;
        Status::NonFinite
    }

    fn matrix_is_singular(&mut self, step: usize) -> Status {
        self.diagnostic = Diagnostic::Singular(
            self.topology.int_to_ext_row[step],
            self.topology.int_to_ext_col[step],
        );
        Status::Singular
    }

    fn zero_pivot(&mut self, step: usize) -> Status {
        self.diagnostic = Diagnostic::ZeroDiag(
            self.topology.int_to_ext_row[step],
            self.topology.int_to_ext_col[step],
        );
        Status::ZeroDiag
    }

    // --- spsolve.c ------------------------------------------------------

    /// Borrow proof that both the traversal and numeric factors are valid.
    pub(super) fn factors(&self) -> SpiceResult<FactorView<'_, T>> {
        match &self.lifecycle {
            Lifecycle::Factored(plan) => Ok(FactorView { frame: self, plan }),
            _ => Err(numerical(
                "SPARSE factors",
                "matrix has no valid numeric factors",
            )),
        }
    }

    #[cfg(test)]
    fn solve(&self, rhs: &[T]) -> Vec<T> {
        self.factors().unwrap().solve(rhs, &mut Vec::new()).unwrap()
    }

    #[cfg(test)]
    fn solve_transposed(&self, rhs: &[T]) -> Vec<T> {
        self.factors()
            .unwrap()
            .solve_transposed(rhs, &mut Vec::new())
            .unwrap()
    }
    /// Gather directly into the internal 1-based work vector without
    /// first zero-initializing every translated entry.
    fn gather_rhs(&self, rhs: &[T], x: &mut Vec<T>, transposed: bool) -> SpiceResult<()> {
        if rhs.len() != self.ext_size {
            return Err(numerical(
                "SPARSE solve",
                "RHS length differs from external matrix order",
            ));
        }
        let indices = if transposed {
            &self.topology.int_to_ext_col
        } else {
            &self.topology.int_to_ext_row
        };
        x.clear();
        sparse_reserve(x, self.size + 1)?;
        x.push(T::ZERO);
        x.extend(indices[1..=self.size].iter().map(|&ext| rhs[ext - 1]));
        Ok(())
    }

    /// Emit external-order values sequentially. Untranslated unknowns keep
    /// exactly their original RHS bits, including signed zeros.
    fn scatter_solution(
        &self,
        rhs: &[T],
        x: &[T],
        transposed: bool,
        result: &mut Vec<T>,
    ) -> SpiceResult<()> {
        let indices = if transposed {
            &self.topology.ext_to_int_row
        } else {
            &self.topology.ext_to_int_col
        };
        result.clear();
        sparse_reserve(result, rhs.len())?;
        result.extend(rhs.iter().zip(&indices[1..]).map(|(&value, &internal)| {
            if internal == NONE { value } else { x[internal] }
        }));
        Ok(())
    }
}

/// A view of immutable numeric factors, obtained only from Factored state.
/// The shared borrow forbids topology or coefficient mutation for its lifetime.
pub(super) struct FactorView<'a, T> {
    frame: &'a Sparse<T>,
    plan: &'a FactorPlan,
}

impl<T: Scalar> FactorView<'_, T> {
    pub(super) fn solve(&self, rhs: &[T], workspace: &mut Vec<T>) -> SpiceResult<Vec<T>> {
        let mut out = Vec::new();
        self.solve_into(rhs, workspace, &mut out, false)?;
        Ok(out)
    }

    pub(super) fn solve_transposed(
        &self,
        rhs: &[T],
        workspace: &mut Vec<T>,
    ) -> SpiceResult<Vec<T>> {
        let mut out = Vec::new();
        self.solve_into(rhs, workspace, &mut out, true)?;
        Ok(out)
    }

    pub(super) fn solve_into(
        &self,
        rhs: &[T],
        workspace: &mut Vec<T>,
        out: &mut Vec<T>,
        transposed: bool,
    ) -> SpiceResult<()> {
        self.solve_workspace(rhs, workspace, transposed)?;
        self.frame.scatter_solution(rhs, workspace, transposed, out)
    }

    fn solve_workspace(
        &self,
        rhs: &[T],
        workspace: &mut Vec<T>,
        transposed: bool,
    ) -> SpiceResult<()> {
        self.frame.gather_rhs(rhs, workspace, transposed)?;
        if !self.solve_loaded_workspace(workspace, transposed, false) {
            self.frame.gather_rhs(rhs, workspace, transposed)?;
            self.solve_loaded_workspace(workspace, transposed, true);
        }
        Ok(())
    }

    fn solve_loaded_workspace(
        &self,
        workspace: &mut [T],
        transposed: bool,
        raw_only: bool,
    ) -> bool {
        match self.plan {
            FactorPlan::Compact(runs) => runs.solve(
                &self.frame.values,
                &self.frame.topology.diag,
                &self.frame.inverses,
                workspace,
                transposed,
                raw_only,
            ),
            FactorPlan::Wide(runs) => runs.solve(
                &self.frame.values,
                &self.frame.topology.diag,
                &self.frame.inverses,
                workspace,
                transposed,
                raw_only,
            ),
        }
    }

    pub(super) fn solve_batch_into(
        &self,
        rhs: &[T],
        workspace: &mut SolveBatchWorkspace<T>,
        out: &mut Vec<T>,
        transposed: bool,
    ) -> Result<(), BatchSolveFailure> {
        let frame = self.frame;
        let n = frame.ext_size;
        out.clear();
        if rhs.is_empty() {
            return Ok(());
        }
        if n == 0 || !rhs.len().is_multiple_of(n) {
            return Err(numerical("SPARSE batch solve", "invalid RHS size").into());
        }
        if let Some(index) = rhs.iter().position(|v| !v.is_finite()) {
            return Err(BatchSolveFailure {
                rhs_index: Some(index / n),
                cause: numerical("SPARSE batch solve", "non-finite RHS"),
            });
        }
        let tile_size = n
            .checked_mul(RHS_BATCH_WIDTH)
            .ok_or_else(|| numerical("SPARSE batch solve", "RHS tile size overflows"))?;
        sparse_reserve(out, rhs.len())?;
        out.extend_from_slice(rhs);
        let gather = if transposed {
            &frame.topology.int_to_ext_col
        } else {
            &frame.topology.int_to_ext_row
        };
        let scatter = if transposed {
            &frame.topology.ext_to_int_row
        } else {
            &frame.topology.ext_to_int_col
        };
        for (tile, result) in rhs.chunks(tile_size).zip(out.chunks_mut(tile_size)) {
            if tile.len() < tile_size {
                for (rhs, vector) in tile.chunks_exact(n).zip(result.chunks_exact_mut(n)) {
                    self.solve_workspace(rhs, &mut workspace.scalar, transposed)?;
                    for (value, &internal) in vector.iter_mut().zip(&scatter[1..]) {
                        if internal != NONE {
                            *value = workspace.scalar[internal];
                        }
                    }
                }
                continue;
            }
            workspace.real.clear();
            sparse_reserve(&mut workspace.real, frame.size + 1)?;
            workspace.real.resize(frame.size + 1, LaneParts::ZERO.re);
            if T::HAS_IMAGINARY {
                workspace.imag.clear();
                sparse_reserve(&mut workspace.imag, frame.size + 1)?;
                workspace.imag.resize(frame.size + 1, LaneParts::ZERO.im);
            }
            for (internal, &external) in gather[1..=frame.size].iter().enumerate() {
                let parts = LaneParts::from_lanes(std::array::from_fn(|lane| {
                    tile[lane * n + external - 1]
                }));
                write_tile_row(workspace, internal + 1, parts);
            }
            let valid = match self.plan {
                FactorPlan::Compact(runs) => runs.solve_batch(
                    &frame.values,
                    &frame.topology.diag,
                    &frame.inverses,
                    workspace,
                    transposed,
                ),
                FactorPlan::Wide(runs) => runs.solve_batch(
                    &frame.values,
                    &frame.topology.diag,
                    &frame.inverses,
                    workspace,
                    transposed,
                ),
            };
            // Untranslated unknowns keep their original RHS, including signed zeros.
            for (lane, vector) in result.chunks_exact_mut(n).enumerate() {
                if !valid.test(lane) {
                    self.frame.gather_rhs(
                        &tile[lane * n..(lane + 1) * n],
                        &mut workspace.scalar,
                        transposed,
                    )?;
                    self.solve_loaded_workspace(&mut workspace.scalar, transposed, true);
                }
                for (value, &internal) in vector.iter_mut().zip(&scatter[1..]) {
                    if internal != NONE {
                        *value = if valid.test(lane) {
                            read_tile_row(workspace, internal).get(lane)
                        } else {
                            workspace.scalar[internal]
                        };
                    }
                }
            }
        }
        if let Some(index) = out.iter().position(|value| !value.is_finite()) {
            return Err(BatchSolveFailure {
                rhs_index: Some(index / n),
                cause: numerical("SPARSE batch solve", "non-finite solution/singular system"),
            });
        }
        Ok(())
    }

    /// Binary-scaled determinant, with `1 <= max(|component|) < 2`.
    /// Normalize each raw pivot before multiplying and renormalize the
    /// product after every step. Binary shifts do not perturb the mantissa
    /// as decimal scaling followed by logarithmic base conversion does.
    pub(super) fn determinant(&self) -> SpiceResult<(T, i64)>
    where
        T: DeterminantScale,
    {
        let mut exponent = 0_i64;
        let mut det = T::one();
        for i in 1..=self.frame.size {
            let pivot = self.frame.values[self.frame.topology.diag[i]];
            let (pivot, shift) = normalize_binary(pivot)?;
            exponent += i64::from(shift);
            let (next, shift) = normalize_binary(det * pivot)?;
            exponent += i64::from(shift);
            det = next;
        }
        if self.frame.parity == PermutationParity::Odd {
            det = -det;
        }
        Ok((det, exponent))
    }
}

fn normalize_binary<T: DeterminantScale>(value: T) -> SpiceResult<(T, i32)> {
    if !value.is_finite() {
        return Err(numerical(
            "SPARSE determinant",
            "non-finite determinant factor",
        ));
    }
    let exponent = crate::primitives::numeric::binary_exponent(value.norm())
        .ok_or_else(|| numerical("SPARSE determinant", "zero determinant factor"))?;
    Ok((value.scale_power(-exponent), exponent))
}

/// Scalar operations used only by the checked determinant accumulation.
pub(super) trait DeterminantScale:
    Scalar + std::ops::Neg<Output = Self> + std::ops::Mul<Output = Self>
{
    fn one() -> Self;
    fn norm(self) -> f64;
    fn scale_power(self, exponent: i32) -> Self;
}

impl DeterminantScale for f64 {
    fn one() -> Self {
        1.0
    }
    fn norm(self) -> f64 {
        self.abs()
    }
    fn scale_power(self, exponent: i32) -> Self {
        crate::primitives::numeric::scale_power(self, exponent)
    }
}

impl DeterminantScale for Complex {
    fn one() -> Self {
        Self::real(1.0)
    }
    fn norm(self) -> f64 {
        self.re.abs().max(self.im.abs())
    }
    fn scale_power(self, exponent: i32) -> Self {
        use crate::primitives::numeric::scale_power;
        Self::new(
            scale_power(self.re, exponent),
            scale_power(self.im, exponent),
        )
    }
}

/// Numeric per-lane factorization outcomes and ordered external solutions.
type IndependentTileSolutions = ([Status; RHS_BATCH_WIDTH], Vec<Option<Vec<Complex>>>);

impl Sparse<Complex> {
    /// Factor independent original coefficient streams using one immutable
    /// ordered symbolic topology. The reference's numeric factors are never
    /// read or mutated. On allocation or layout mismatch callers use scalar
    /// factorization, which has smaller peak scratch requirements.
    fn factor_independent_packed(
        reference: &Self,
        inputs: &[&[Complex]],
        workspace: &mut IndependentFactorTileWorkspace,
    ) -> Option<[Status; RHS_BATCH_WIDTH]> {
        if !(2..=RHS_BATCH_WIDTH).contains(&inputs.len()) {
            return None;
        }
        let plan = reference.lifecycle.plan()?;
        let topology = &reference.topology;
        let original_entries = topology.original_entries;
        if reference.size == 0 || inputs.iter().any(|values| values.len() != original_entries) {
            return None;
        }
        let values_len = reference.values.len();
        let size = reference.size;
        let IndependentFactorTileWorkspace {
            values,
            scratch,
            inverses,
        } = workspace;
        sparse_reserve(values, values_len.saturating_sub(values.len())).ok()?;
        sparse_reserve(scratch, (size + 1).saturating_sub(scratch.len())).ok()?;
        sparse_reserve(inverses, (size + 1).saturating_sub(inverses.len())).ok()?;
        values.resize(values_len, LaneParts::ZERO);
        scratch.resize(size + 1, LaneParts::ZERO);
        inverses.resize(size + 1, LaneParts::ZERO);
        for (index, packed) in values.iter_mut().enumerate() {
            *packed = LaneParts::from_lanes(std::array::from_fn(|lane| {
                if index < original_entries {
                    inputs.get(lane).map_or(Complex::ZERO, |input| input[index])
                } else {
                    Complex::ZERO
                }
            }));
        }
        let count = inputs.len();
        let mut status = [Status::Okay; RHS_BATCH_WIDTH];
        let first_pivot = topology.diag[1];
        for (lane, lane_status) in status.iter_mut().enumerate().take(count) {
            let pivot = values[first_pivot].get::<Complex>(lane);
            *lane_status = if pivot.is_zero() {
                Status::ZeroDiag
            } else if !pivot.is_finite() {
                Status::NonFinite
            } else {
                inverses[1].set(lane, pivot.cached_recip());
                Status::Okay
            };
        }
        match plan.as_ref() {
            FactorPlan::Compact(runs) => runs.refactor_independent_tile(
                values,
                &topology.diag,
                scratch,
                inverses,
                &mut status,
                count,
            ),
            FactorPlan::Wide(runs) => runs.refactor_independent_tile(
                values,
                &topology.diag,
                scratch,
                inverses,
                &mut status,
                count,
            ),
        }
        for (lane, status) in status.iter_mut().enumerate().take(count) {
            if *status == Status::Okay && values.iter().any(|v| !v.get::<Complex>(lane).is_finite())
            {
                *status = Status::NonFinite;
            }
        }
        Some(status)
    }

    /// Materialize independent immutable numeric factors for callers that
    /// must retain snapshots, perform multiple RHS or transposed solves.
    pub(super) fn factor_independent_tile(
        frames: &mut [Self],
        workspace: &mut IndependentFactorTileWorkspace,
    ) -> Option<[Status; RHS_BATCH_WIDTH]> {
        use crate::maths::telemetry::{AttemptPath, FactorAttempt};

        let first = frames.first()?;
        let plan = first.lifecycle.plan()?;
        if frames.iter().any(|frame| {
            !frame.coefficients_loaded()
                || !matches!(frame.lifecycle.plan(), Some(other)
                    if std::sync::Arc::ptr_eq(other, plan))
                || !std::sync::Arc::ptr_eq(&frame.topology, &first.topology)
        }) {
            return None;
        }
        let original_entries = first.topology.original_entries;
        let inputs: [&[Complex]; RHS_BATCH_WIDTH] = std::array::from_fn(|lane| {
            frames
                .get(lane)
                .map_or(&[][..], |frame| &frame.values[..original_entries])
        });
        let mut status =
            Self::factor_independent_packed(first, &inputs[..frames.len()], workspace)?;
        for (lane, frame) in frames.iter_mut().enumerate() {
            if status[lane] == Status::Okay {
                if frame.prepare_inverses().is_err() {
                    status[lane] = Status::Resource;
                } else {
                    for (dst, packed) in frame.values.iter_mut().zip(&workspace.values) {
                        *dst = packed.get::<Complex>(lane);
                    }
                    for (dst, packed) in frame.inverses.iter_mut().zip(&workspace.inverses) {
                        *dst = packed.get::<Complex>(lane);
                    }
                    status[lane] = frame.done();
                }
            }
            frame.last_attempt = Some(FactorAttempt {
                path: AttemptPath::Refactor,
                status: status[lane],
            });
        }
        Some(status)
    }

    /// Factor and solve independent numeric matrices without creating one
    /// scalar LU snapshot per lane. Each input is the original-element
    /// stream in the reference's order, and each RHS is in external order.
    /// None indicates an unsupported packed shape or exhausted scratch,
    /// in which case the caller can use independent scalar factors.
    pub(super) fn factor_and_solve_independent_tile(
        reference: &Self,
        inputs: &[&[Complex]],
        rhs: &[&[Complex]],
        workspace: &mut IndependentFactorTileWorkspace,
        transposed: bool,
    ) -> Option<IndependentTileSolutions> {
        if inputs.len() != rhs.len()
            || rhs.iter().any(|source| {
                source.len() != reference.ext_size || source.iter().any(|value| !value.is_finite())
            })
        {
            return None;
        }
        let mut status = Self::factor_independent_packed(reference, inputs, workspace)?;
        let count = inputs.len();
        let active = LaneMask::from_array(std::array::from_fn(|lane| {
            lane < count && status[lane] == Status::Okay
        }));
        let topology = &reference.topology;
        if active.any() {
            let gather = if transposed {
                &topology.int_to_ext_col
            } else {
                &topology.int_to_ext_row
            };
            for (internal, &external) in gather[1..=reference.size].iter().enumerate() {
                workspace.scratch[internal + 1] =
                    LaneParts::from_lanes(std::array::from_fn(|lane| {
                        rhs.get(lane)
                            .map_or(Complex::ZERO, |source| source[external - 1])
                    }));
            }
            match reference.lifecycle.plan()?.as_ref() {
                FactorPlan::Compact(runs) => runs.solve_independent_tile(
                    &workspace.values,
                    &topology.diag,
                    &workspace.inverses,
                    &mut workspace.scratch,
                    active,
                    transposed,
                ),
                FactorPlan::Wide(runs) => runs.solve_independent_tile(
                    &workspace.values,
                    &topology.diag,
                    &workspace.inverses,
                    &mut workspace.scratch,
                    active,
                    transposed,
                ),
            }
        }
        let scatter = if transposed {
            &topology.ext_to_int_row
        } else {
            &topology.ext_to_int_col
        };
        let mut output = Vec::new();
        output.try_reserve(count).ok()?;
        for lane in 0..count {
            if status[lane] != Status::Okay {
                output.push(None);
                continue;
            }
            let mut solution = Vec::new();
            solution.try_reserve(reference.ext_size).ok()?;
            for (index, &internal) in scatter[1..].iter().enumerate() {
                let value = if internal == NONE {
                    rhs[lane][index]
                } else {
                    workspace.scratch[internal].get::<Complex>(lane)
                };
                if !value.is_finite() {
                    status[lane] = Status::NonFinite;
                    break;
                }
                solution.push(value);
            }
            output.push((status[lane] == Status::Okay).then_some(solution));
        }
        Some((status, output))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build(n: usize, entries: &[(usize, usize, f64)]) -> Sparse<f64> {
        let mut m = Sparse::new(n).unwrap();
        for &(r, c, v) in entries {
            let e = m.get_element(r, c).unwrap();
            m.add(e, v);
        }
        m
    }

    fn residual(entries: &[(usize, usize, f64)], x: &[f64], b: &[f64]) -> f64 {
        let mut ax = vec![0.0; b.len()];
        for &(r, c, v) in entries {
            ax[r - 1] += v * x[c - 1];
        }
        ax.iter()
            .zip(b)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max)
    }

    #[test]
    fn sparse_construction_rejects_invalid_dimensions_and_external_coordinates() {
        assert!(Sparse::<f64>::new(usize::MAX).is_err());
        assert!(Sparse::<f64>::new(usize::MAX / size_of::<usize>()).is_err());
        assert!(Sparse::<f64>::from_originals(2, [(0, 1, 1.)]).is_err());
        assert!(Sparse::<f64>::from_originals(2, [(2, 3, 1.)]).is_err());
        let pattern = crate::maths::Pattern::new(2, [(0, 0), (1, 1)]).unwrap();
        assert!(Sparse::<f64>::from_pattern(&pattern, &[1.]).is_err());
        let mut sparse = Sparse::<f64>::new(2).unwrap();
        assert!(sparse.get_element(1, usize::MAX).is_err());
        assert!(sparse.get_element(0, 1).is_err());
        assert!(sparse.add_col(1, usize::MAX).is_err());
        assert!(sparse.zero_col(0).is_err());
        assert_eq!(sparse.size, 0);
        let first = sparse.get_element(2, 1).unwrap();
        sparse.set(first, 2.);
        let second = sparse.get_element(1, 1).unwrap();
        sparse.add(first, 3.);
        assert_eq!(sparse.get_element(2, 1).unwrap(), first);
        assert_ne!(second, first);
        assert_eq!(sparse.value(first), 5.);
    }

    /// Structural oracle independent of the linked-list exchange algorithm:
    /// every stable element handle has one expected matrix coordinate; the
    /// explicit per-axis permutations define the only allowed coordinate moves.
    fn assert_exchange_topology(
        topology: &Topology,
        n: usize,
        expected: &[(usize, usize)],
        row_permutation: &[usize],
        col_permutation: &[usize],
    ) {
        assert_eq!(topology.elements.len(), expected.len());
        let mut counted_rows = vec![0_usize; expected.len()];
        let mut counted_cols = vec![0_usize; expected.len()];
        for row in 1..=n {
            let mut prev_col = 0;
            let mut element = topology.first_in_row[row];
            let mut visited = 0;
            while element != NONE {
                assert!(element < expected.len(), "dangling row-list link at {row}");
                visited += 1;
                assert!(visited <= expected.len(), "row-list cycle at {row}");
                let node = &topology.elements[element];
                assert_eq!(node.row, row);
                assert!(node.col > prev_col, "row-list not ordered at {row}");
                prev_col = node.col;
                counted_rows[element] += 1;
                element = node.next_in_row;
            }
        }
        for col in 1..=n {
            let mut prev_row = 0;
            let mut element = topology.first_in_col[col];
            let mut visited = 0;
            while element != NONE {
                assert!(
                    element < expected.len(),
                    "dangling column-list link at {col}"
                );
                visited += 1;
                assert!(visited <= expected.len(), "column-list cycle at {col}");
                let node = &topology.elements[element];
                assert_eq!(node.col, col);
                assert!(node.row > prev_row, "column-list not ordered at {col}");
                prev_row = node.row;
                counted_cols[element] += 1;
                element = node.next_in_col;
            }
        }
        assert!(counted_rows.iter().all(|count| *count == 1));
        assert!(counted_cols.iter().all(|count| *count == 1));
        for (element, position) in topology.elements.iter().zip(expected) {
            assert_eq!((element.row, element.col), *position);
        }
        for i in 1..=n {
            assert_eq!(topology.int_to_ext_row[i], row_permutation[i]);
            assert_eq!(topology.int_to_ext_col[i], col_permutation[i]);
            assert_eq!(topology.ext_to_int_row[row_permutation[i]], i);
            assert_eq!(topology.ext_to_int_col[col_permutation[i]], i);
        }
    }

    #[test]
    fn row_and_column_exchange_match_reference_permutations_for_sparse_patterns() {
        let mut seed = 0x22ac_b703_ef84_a911_u64;
        let mut sample = || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            seed >> 32
        };
        for n in [1, 2, 3, 5, 9, 16] {
            for density in 0..5 {
                let mut matrix = Sparse::<f64>::new(n).unwrap();
                // Bind every coordinate to identity in the external/internal
                // maps. The off-diagonal pattern is independently randomized.
                for i in 1..=n {
                    matrix.get_element(i, i).unwrap();
                }
                for row in 1..=n {
                    for col in 1..=n {
                        if row != col && sample() % 5 <= density {
                            matrix.get_element(row, col).unwrap();
                        }
                    }
                }
                matrix.link_rows();
                let mut positions: Vec<_> = matrix
                    .topology
                    .elements
                    .iter()
                    .map(|node| (node.row, node.col))
                    .collect();
                let mut row_permutation: Vec<_> = (0..=n).collect();
                let mut col_permutation: Vec<_> = (0..=n).collect();
                assert_exchange_topology(
                    &matrix.topology,
                    n,
                    &positions,
                    &row_permutation,
                    &col_permutation,
                );
                for step in 0..(9 * n + 53) {
                    let a = if step % 17 == 0 {
                        1
                    } else {
                        sample() as usize % n + 1
                    };
                    let b = if step % 17 == 0 {
                        n
                    } else {
                        sample() as usize % n + 1
                    };
                    if step % 2 == 0 {
                        std::sync::Arc::make_mut(&mut matrix.topology).row_exchange(a, b);
                        for (row, _) in &mut positions {
                            if *row == a {
                                *row = b;
                            } else if *row == b {
                                *row = a;
                            }
                        }
                        row_permutation.swap(a, b);
                    } else {
                        std::sync::Arc::make_mut(&mut matrix.topology).col_exchange(a, b);
                        for (_, col) in &mut positions {
                            if *col == a {
                                *col = b;
                            } else if *col == b {
                                *col = a;
                            }
                        }
                        col_permutation.swap(a, b);
                    }
                    assert_exchange_topology(
                        &matrix.topology,
                        n,
                        &positions,
                        &row_permutation,
                        &col_permutation,
                    );
                }
            }
        }
    }

    #[test]
    fn single_element_and_empty_orthogonal_lists_exchange_without_losing_links() {
        let mut matrix = Sparse::<f64>::new(2).unwrap();
        let handle = matrix.get_element(1, 2).unwrap().0;
        matrix.link_rows();
        assert_eq!(matrix.topology.first_in_row[2], NONE);
        assert_eq!(matrix.topology.first_in_col[1], NONE);
        let topology = std::sync::Arc::make_mut(&mut matrix.topology);
        topology.row_exchange(1, 2);
        topology.col_exchange(1, 2);
        topology.row_exchange(2, 2);
        topology.col_exchange(1, 1);
        assert_exchange_topology(topology, 2, &[(2, 1)], &[0, 2, 1], &[0, 2, 1]);
        assert_eq!(topology.elements[handle].row, 2);
        assert_eq!(topology.elements[handle].col, 1);
    }

    #[test]
    fn unfactored_solve_rejects_without_touching_workspace() {
        let m = Sparse::<f64>::new(1).unwrap();
        let mut workspace = vec![42.0];
        assert!(
            m.factors()
                .and_then(|factors| factors.solve(&[1.0], &mut workspace))
                .is_err()
        );
        assert_eq!(workspace, vec![42.0]);
    }

    #[test]
    fn lifecycle_rejects_stale_factors_after_every_numeric_edit() {
        let entries = [(1, 1, 3.), (1, 2, 1.), (2, 1, 2.), (2, 2, 4.)];
        let input = entries.map(|(_, _, v)| v);
        let mut m = build(2, &entries);
        assert!(matches!(m.lifecycle, Lifecycle::ColumnsOnly));
        assert!(m.factors().is_err());
        assert_eq!(m.order_and_factor(1e-3, 0.), Status::Okay);
        assert!(matches!(m.lifecycle, Lifecycle::Factored(_)));
        let plan = m.lifecycle.plan().unwrap().clone();
        let initial = m.solve(&[4., 6.]);
        assert_eq!(m.factor(), Status::NeedsReload);
        assert_eq!(m.order_and_factor(1e-3, 0.), Status::NeedsReload);
        assert_eq!(m.solve(&[4., 6.]), initial);
        let handle = m.get_element(1, 1).unwrap();
        assert!(m.factors().is_ok(), "existing-handle lookup is read-only");
        let edits: [fn(&mut Sparse<f64>, SolverElement); 6] = [
            |frame: &mut Sparse<f64>, h| frame.set(h, 1.),
            |frame: &mut Sparse<f64>, h| frame.add(h, 1.),
            |frame: &mut Sparse<f64>, _| frame.load_gmin(1.),
            |frame: &mut Sparse<f64>, _| frame.zero_col(1).unwrap(),
            |frame: &mut Sparse<f64>, _| frame.add_col(1, 2).unwrap(),
            |frame: &mut Sparse<f64>, _| frame.clear(),
        ];
        for edit in edits {
            let saved = m.clone();
            let rhs = [4., 6.];
            let expected = saved.solve(&rhs);
            edit(&mut m, handle);
            assert!(m.factors().is_err());
            if !m.coefficients_loaded() {
                assert_eq!(m.factor(), Status::NeedsReload);
                assert_eq!(m.order_and_factor(1e-3, 0.), Status::NeedsReload);
            }
            assert!(matches!(m.lifecycle, Lifecycle::Symbolic(_)));
            assert!(std::sync::Arc::ptr_eq(m.lifecycle.plan().unwrap(), &plan));
            assert_eq!(saved.solve(&rhs), expected);
            m.reload_originals(&input);
            assert_eq!(m.factor(), Status::Okay);
            assert_eq!(m.solve(&rhs), expected);
        }
        let retyped: Sparse<Complex> = m.into_retyped();
        assert!(matches!(retyped.lifecycle, Lifecycle::Symbolic(_)));
        assert!(retyped.factors().is_err());
    }

    #[test]
    fn plan_construction_rejects_inconsistent_links_and_offsets() {
        let entries = [(1, 1, 3.), (1, 2, 1.), (2, 1, 2.), (2, 2, 4.)];
        let mut original = build(2, &entries);
        assert_eq!(original.order_and_factor(1e-3, 0.), Status::Okay);
        let corruptions: [fn(&mut Topology); 6] = [
            |topology: &mut Topology| topology.diag[1] = NONE,
            |topology: &mut Topology| topology.first_in_col[1] = topology.elements.len(),
            |topology: &mut Topology| {
                let head = topology.first_in_col[1];
                topology.elements[head].next_in_col = head;
            },
            |topology: &mut Topology| {
                let head = topology.first_in_col[1];
                topology.elements[head].row = 0;
            },
            |topology: &mut Topology| {
                let h = topology.diag[1];
                topology.elements[h].next_in_row = NONE;
            },
            |topology: &mut Topology| {
                let h = topology.diag[1];
                topology.elements[h].next_in_row = h;
            },
        ];
        for corrupt in corruptions {
            let mut frame = original.clone();
            corrupt(std::sync::Arc::make_mut(&mut frame.topology));
            frame.lifecycle = Lifecycle::Unordered;
            assert!(matches!(
                frame.compile_runs::<u32>(),
                Err(Status::InvalidPlan)
            ));
            assert!(matches!(
                frame.compile_runs::<usize>(),
                Err(Status::InvalidPlan)
            ));
        }
        assert!(AxisRun::<u32>::new(0, 0, 0).is_none());
        assert!(AxisRun::<usize>::new(3, 2, 4).is_none());
        if let Ok(max) = usize::try_from(u64::from(u32::MAX) + 1) {
            assert!(AxisRun::<u32>::new(0, 0, max).is_none());
            assert!(AxisRun::<usize>::new(0, 0, max).is_some());
        }
    }

    #[test]
    fn zero_diagonal_and_nonfinite_failures_have_distinct_retry_states() {
        let entries = [(1, 1, 2.), (2, 2, 3.)];
        let input = entries.map(|(_, _, v)| v);
        let mut m = build(2, &entries);
        assert_eq!(m.order_and_factor(1e-3, 0.), Status::Okay);
        let plan = m.lifecycle.plan().unwrap().clone();
        m.reload_originals(&[0., 3.]);
        assert_eq!(m.factor(), Status::ZeroDiag);
        assert_eq!(m.factor(), Status::NeedsReload);
        assert!(matches!(m.lifecycle, Lifecycle::Symbolic(_)));
        assert!(m.factors().is_err());
        assert_ne!(m.where_singular(), (0, 0));
        m.reload_originals(&[f64::NAN, 3.]);
        assert_eq!(m.factor(), Status::NonFinite);
        assert_eq!(m.order_and_factor(1e-3, 0.), Status::NeedsReload);
        assert!(matches!(m.lifecycle, Lifecycle::ReorderRequired(_)));
        assert!(m.factors().is_err());
        assert_eq!(m.where_singular(), (0, 0));
        m.reload_originals(&input);
        assert!(!m.can_refactor());
        assert_eq!(m.factor(), Status::Okay);
        assert!(std::sync::Arc::ptr_eq(m.lifecycle.plan().unwrap(), &plan));
        assert!(m.factors().is_ok());
    }

    #[test]
    fn pivot_product_summaries_match_reference_reverse_searches() {
        for size in [1, 63, 64, 323, 514, 515, 1027] {
            let mut products = PivotProducts::new(size).unwrap();
            let mut seed = 71_u64;
            for i in 0..6000 {
                seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                let at = ((seed >> 32) as usize) % size;
                let v = ((seed >> 48) % 17) as i64 - 2;
                products.set(at, v);
                if i % 11 == 0 {
                    products.swap(at, (at + 99) % size);
                }
                let before = ((seed >> 16) as usize) % (size + 1);
                let limit = ((seed >> 40) % 20) as i64 - 3;
                let expect = products.values[..before].iter().rposition(|&v| v < limit);
                assert_eq!(products.previous_below(before, limit), expect);
                let expect = products.values[..before].iter().rposition(|&v| v == 0);
                assert_eq!(products.previous_zero(before), expect);
            }
        }
    }

    #[test]
    fn solve_workspace_reuses_storage_and_keeps_external_permutation() {
        // Nodes 2 and 4 are deliberately absent, and the remaining two
        // have a zero diagonal forcing MNA/pivot exchange.
        let mut m = build(4, &[(3, 1, 2.), (1, 3, 4.), (3, 3, 1.)]);
        m.mna_preorder();
        assert_eq!(m.order_and_factor(1e-3, 0.), Status::Okay);
        let rhs = [8., -0., 6., -17.];
        let mut workspace = Vec::new();
        let first = m.factors().unwrap().solve(&rhs, &mut workspace).unwrap();
        let storage = workspace.as_ptr();
        assert_eq!(first[0], 2.);
        assert_eq!(first[2], 2.);
        assert_eq!(first[1].to_bits(), (-0.0_f64).to_bits());
        assert_eq!(first[3], -17.);
        let second = m
            .factors()
            .unwrap()
            .solve(&[4., 3., 4., 7.], &mut workspace)
            .unwrap();
        assert_eq!(storage, workspace.as_ptr());
        assert_eq!(second, [1.5, 3., 1., 7.]);
        let transposed = m.solve_transposed(&rhs);
        assert_eq!(transposed[0], 0.5);
        assert_eq!(transposed[2], 4.);
        assert_eq!(transposed[1].to_bits(), (-0.0_f64).to_bits());
        assert_eq!(transposed[3], -17.);
    }

    fn check_batch<T: Scalar>(matrix: &Sparse<T>, rhs: &[T], bits: impl Fn(T) -> [u64; 2]) {
        let factor = matrix.factors().unwrap();
        let n = matrix.ext_size;
        let mut batch = SolveBatchWorkspace::default();
        let mut out = Vec::new();
        let mut scalar = Vec::new();
        let mut expected = Vec::new();
        let mut storage = None;
        for transposed in [false, true] {
            for count in [9, 0, 1, 2, 3, 4, 5, 8, 7, 9] {
                let rhs = &rhs[..count * n];
                factor
                    .solve_batch_into(rhs, &mut batch, &mut out, transposed)
                    .unwrap();
                if let Some(storage) = storage {
                    assert_eq!(batch.real.as_ptr(), storage);
                } else {
                    storage = Some(batch.real.as_ptr());
                }
                let mut checked = Vec::new();
                for vector in rhs.chunks_exact(n) {
                    factor
                        .solve_into(vector, &mut scalar, &mut expected, transposed)
                        .unwrap();
                    checked.extend(expected.iter().copied().map(&bits));
                }
                assert_eq!(out.iter().copied().map(&bits).collect::<Vec<_>>(), checked);
            }
        }
        assert!(
            factor
                .solve_batch_into(&rhs[..n + 1], &mut batch, &mut out, false)
                .is_err()
        );
    }

    #[test]
    fn real_rhs_tiles_preserve_permutation_untranslated_signed_zero_and_range() {
        let mut matrix = build(
            6,
            &[
                (3, 1, 2.),
                (1, 3, 4.),
                (3, 3, 1.),
                (3, 5, 0.25),
                (5, 3, -0.5),
                (5, 5, 3.),
                (6, 6, 1e-200),
            ],
        );
        matrix.mna_preorder();
        assert!(matches!(
            matrix.order_and_factor(1e-3, 0.),
            Status::Okay | Status::SmallPivot
        ));
        let values = [
            0.,
            -0.,
            1.,
            -2.,
            1e-200,
            -1e100,
            1e-100,
            f64::MIN_POSITIVE,
            1e100,
        ];
        let rhs: Vec<_> = (0_usize..9)
            .flat_map(|lane| {
                [
                    values[lane],
                    -0.,
                    values[(lane + 1) % 9],
                    values[lane],
                    values[(lane + 2) % 9],
                    values[(lane + 3) % 9],
                ]
            })
            .collect();
        check_batch(&matrix, &rhs, |value| [value.to_bits(), 0]);
        // Invalid lanes cannot be silently ignored just because other RHS
        // are finite and the tile is not full.
        let mut invalid = rhs[..18].to_vec();
        invalid[12] = f64::NAN;
        let failure = matrix
            .factors()
            .unwrap()
            .solve_batch_into(
                &invalid,
                &mut SolveBatchWorkspace::default(),
                &mut Vec::new(),
                false,
            )
            .unwrap_err();
        assert_eq!(failure.rhs_index, Some(2));
    }

    #[test]
    fn batch_reports_overflowing_lane_and_recovers_workspace() {
        let mut matrix = build(1, &[(1, 1, f64::MIN_POSITIVE)]);
        assert!(matches!(
            matrix.order_and_factor(1e-3, 0.),
            Status::Okay | Status::SmallPivot
        ));
        let factor = matrix.factors().unwrap();
        let mut work = SolveBatchWorkspace::default();
        let mut output = Vec::new();
        for transposed in [false, true] {
            let failure = factor
                .solve_batch_into(&[0., 1e300, 1., -0.], &mut work, &mut output, transposed)
                .unwrap_err();
            assert_eq!(failure.rhs_index, Some(1));
            factor
                .solve_batch_into(&[0., 1., 0.5, -0.], &mut work, &mut output, transposed)
                .unwrap();
            assert_eq!(output[1], 1. / f64::MIN_POSITIVE);
            assert_eq!(output[3].to_bits(), (-0.0_f64).to_bits());
        }
    }

    #[test]
    fn complex_rhs_tiles_match_independent_solve_bits_at_exponent_limits() {
        let mut matrix = Sparse::<Complex>::new(6).unwrap();
        for (r, c, value) in [
            (1, 3, Complex::new(2., -0.25)),
            (3, 1, Complex::new(4., 0.5)),
            (3, 3, Complex::new(1., -1.)),
            (3, 5, Complex::new(0.125, -0.5)),
            (5, 3, Complex::new(-0.25, 0.5)),
            (5, 5, Complex::new(2., 3.)),
            (6, 6, Complex::new(f64::MAX, f64::MAX)),
        ] {
            let entry = matrix.get_element(r, c).unwrap();
            matrix.set(entry, value);
        }
        assert!(matches!(
            matrix.order_and_factor(1e-3, 0.),
            Status::Okay | Status::SmallPivot
        ));
        let values = [
            0.,
            -0.,
            1.,
            -2.,
            1e-200,
            -1e100,
            1e-100,
            f64::MIN_POSITIVE,
            1e100,
        ];
        let rhs: Vec<_> = (0_usize..9)
            .flat_map(|lane| {
                let value = |offset: usize| {
                    Complex::new(values[(lane + offset) % 9], values[(lane + offset + 1) % 9])
                };
                [
                    value(0),
                    Complex::new(-0., 0.),
                    value(1),
                    value(2),
                    value(3),
                    value(4),
                ]
            })
            .collect();
        check_batch(&matrix, &rhs, |z| [z.re.to_bits(), z.im.to_bits()]);
    }

    fn time_rhs_batches<T: Scalar>(matrix: &Sparse<T>, rhs: &[T], scalar: &str) {
        use std::hint::black_box;
        use std::time::Instant;

        let factor = matrix.factors().unwrap();
        let n = matrix.ext_size;
        let mut batch = SolveBatchWorkspace::default();
        let mut work = Vec::new();
        let mut single = Vec::new();
        let mut before = Vec::with_capacity(rhs.len());
        let mut after = Vec::with_capacity(rhs.len());
        for transposed in [false, true] {
            for count in [1, 2, 4, 8, 32, 64] {
                let rhs = &rhs[..n * count];
                let mut baseline = Vec::new();
                let mut batched = Vec::new();
                for _ in 0..5 {
                    let start = Instant::now();
                    for _ in 0..16 {
                        before.clear();
                        for row in black_box(rhs).chunks_exact(n) {
                            factor
                                .solve_into(row, &mut work, &mut single, transposed)
                                .unwrap();
                            before.extend_from_slice(&single);
                        }
                        black_box(&before);
                    }
                    baseline.push(start.elapsed());
                    let start = Instant::now();
                    for _ in 0..16 {
                        factor
                            .solve_batch_into(black_box(rhs), &mut batch, &mut after, transposed)
                            .unwrap();
                        black_box(&after);
                    }
                    batched.push(start.elapsed());
                    assert_eq!(before, after);
                }
                baseline.sort_unstable();
                batched.sort_unstable();
                println!(
                    "scalar={scalar} transpose={transposed} rhs={count} independent_ns={} batch_ns={}",
                    baseline[2].as_nanos(),
                    batched[2].as_nanos()
                );
            }
        }
    }

    #[test]
    #[ignore = "opt-in sparse multiple RHS kernel benchmark"]
    fn sparse_rhs_batch_benchmark() {
        let n = 512;
        let mut real = Sparse::<f64>::new(n).unwrap();
        let mut complex = Sparse::<Complex>::new(n).unwrap();
        for row in 1..=n {
            for (col, value) in [
                (row, 5.),
                (row.saturating_sub(1), -0.75),
                (row + 1, -0.5),
                (row.saturating_sub(16), -0.125),
                (row + 16, -0.25),
            ] {
                if !(1..=n).contains(&col) {
                    continue;
                }
                let h = real.get_element(row, col).unwrap();
                real.set(h, value);
                let h = complex.get_element(row, col).unwrap();
                complex.set(h, Complex::new(value, value * 0.25));
            }
        }
        assert_eq!(real.order_and_factor(1e-3, 0.), Status::Okay);
        assert_eq!(complex.order_and_factor(1e-3, 0.), Status::Okay);
        let rhs: Vec<_> = (0..n * 64)
            .map(|i| f64::from(u32::try_from((i * 13 + 7) % 127).unwrap()) / 64. - 1.)
            .collect();
        let imaginary: Vec<_> = rhs.iter().map(|&v| Complex::new(v, 0.25 - v)).collect();
        time_rhs_batches(&real, &rhs, "real");
        time_rhs_batches(&complex, &imaginary, "complex");
    }

    #[test]
    fn extreme_complex_reciprocals_and_invalid_factor_recovery() {
        let mut c = Sparse::<Complex>::new(1).unwrap();
        let handle = c.get_element(1, 1).unwrap();
        c.set(handle, Complex::new(f64::MAX, f64::MAX));
        assert!(matches!(
            c.order_and_factor(1e-3, 0.),
            Status::Okay | Status::SmallPivot
        ));
        let x = c.solve(&[Complex::real(1.)])[0];
        assert!(x.re > 0. && x.im < 0.);
        assert!((x.re * f64::MAX - 0.5).abs() < 1e-14);
        let mut m = build(2, &[(1, 1, 2.), (1, 2, 1.), (2, 1, 1.), (2, 2, 2.)]);
        assert_eq!(m.order_and_factor(1e-3, 0.), Status::Okay);
        m.clear();
        for (r, c, v) in [(1, 1, 1.), (1, 2, f64::MAX), (2, 1, f64::MAX), (2, 2, 1.)] {
            let h = m.get_element(r, c).unwrap();
            m.set(h, v);
        }
        assert_eq!(m.factor(), Status::NonFinite);
        assert!(!m.is_factored());
        assert!(m.factors().is_err());
        m.clear();
        for (r, c, v) in [(1, 1, 2.), (1, 2, 1.), (2, 1, 1.), (2, 2, 2.)] {
            let h = m.get_element(r, c).unwrap();
            m.set(h, v);
        }
        assert_eq!(m.factor(), Status::Okay);
        assert_eq!(m.solve(&[3., 3.]), [1., 1.]);
    }

    #[test]
    fn fixed_order_checks_every_stored_factor_and_early_zero_pivot() {
        let entries = [(1, 1, 2.), (1, 2, 1.), (2, 1, 1.), (2, 2, 2.)];
        let mut m = build(2, &entries);
        assert_eq!(m.order_and_factor(1e-3, 0.), Status::Okay);
        for (bad_row, bad_col, bad_value) in [
            (2, 1, f64::INFINITY),
            (1, 2, f64::NAN),
            (2, 2, f64::NEG_INFINITY),
        ] {
            m.clear();
            for (r, c, v) in entries {
                let h = m.get_element(r, c).unwrap();
                m.set(
                    h,
                    if (r, c) == (bad_row, bad_col) {
                        bad_value
                    } else {
                        v
                    },
                );
            }
            assert_eq!(m.factor(), Status::NonFinite);
            assert!(!m.is_factored());
            m.clear();
            for (r, c, v) in entries {
                let h = m.get_element(r, c).unwrap();
                m.set(h, v);
            }
            assert_eq!(m.factor(), Status::Okay);
        }
        m.clear();
        let h = m.get_element(2, 2).unwrap();
        m.set(h, f64::NAN);
        assert_eq!(m.factor(), Status::NonFinite);
    }

    #[test]
    fn determinant_rescales_extreme_pivots_without_nonfinite_loops() {
        for (a, b) in [(1e307, 1e308), (1e-307, 1e-308)] {
            let mut m = build(2, &[(1, 1, a), (2, 2, b)]);
            assert_eq!(m.order_and_factor(1e-3, 0.), Status::Okay);
            let (mantissa, exponent) = m.factors().unwrap().determinant().unwrap();
            let (am, ae) = normalize_binary(a).unwrap();
            let (bm, be) = normalize_binary(b).unwrap();
            let shift = i32::try_from(exponent - i64::from(ae + be)).unwrap();
            assert!((mantissa * 2_f64.powi(shift) / (am * bm) - 1.).abs() < 1e-14);
        }
        let tiny = f64::from_bits(1);
        let mut m = build(1, &[(1, 1, tiny)]);
        assert_eq!(m.order_and_factor(1e-3, 0.), Status::Okay);
        assert_eq!(m.factors().unwrap().determinant().unwrap(), (1., -1074));
        assert_eq!(m.solve(&[tiny]), [1.]);
        assert_eq!(m.solve_transposed(&[tiny]), [1.]);
        m.reload_originals(&[tiny * 2.]);
        assert_eq!(m.factor(), Status::Okay);
        assert_eq!(m.solve(&[tiny]), [0.5]);

        let mut m = Sparse::<Complex>::new(2).unwrap();
        for i in 1..=2 {
            let h = m.get_element(i, i).unwrap();
            m.set(h, Complex::new(1e307, 1e307));
        }
        assert_eq!(m.order_and_factor(1e-3, 0.), Status::Okay);
        let (mantissa, exponent) = m.factors().unwrap().determinant().unwrap();
        let (m, e) = normalize_binary(1e307).unwrap();
        let shift = i32::try_from(exponent - 2 * i64::from(e)).unwrap();
        let imag = mantissa.im * 2_f64.powi(shift);
        assert!((imag / (m * m) - 2.).abs() < 1e-14);
        assert!(mantissa.re.abs() < 1e-14);
    }

    #[test]
    fn raw_factors_do_not_require_representable_normalized_entries() {
        use crate::primitives::numeric::scale_power;
        let tiny = scale_power(1., -1000);
        let large = scale_power(1., 1000);
        // Retain the same diagonal order as the well-scaled factorization.
        // U12/D11 is 2^2000 or 2^-2000, but the Schur entry and solution
        // are finite. Check both fresh ordering and numerical refactor.
        let mut m = build(2, &[(1, 1, 2.), (1, 2, 1.), (2, 1, 1.), (2, 2, 2.)]);
        assert_eq!(m.order_and_factor(1e-3, 0.), Status::Okay);
        let c = |x| Complex::new(x, x);
        let mut complex = Sparse::from_originals(
            2,
            [(1, 1, c(2.)), (1, 2, c(1.)), (2, 1, c(1.)), (2, 2, c(2.))],
        )
        .unwrap();
        assert_eq!(complex.order_and_factor(1e-3, 0.), Status::Okay);
        for (a, b) in [(tiny, large), (large, tiny), (-tiny, large), (-large, tiny)] {
            m.reload_originals(&[a, b, a, b * 2.]);
            assert_eq!(m.factor(), Status::Okay);
            assert_eq!(m.solve(&[b, b * 2.]), [0., 1.]);
            assert_eq!(m.solve_transposed(&[a, b * 2.]), [0., 1.]);
            let mut fresh = build(2, &[(1, 1, a), (1, 2, b), (2, 1, a), (2, 2, b * 2.)]);
            assert_eq!(fresh.order_and_factor(1e-3, 0.), Status::Okay);
            assert_eq!(fresh.solve(&[b, b * 2.]), [0., 1.]);

            complex.reload_originals(&[c(a), c(b), c(a), c(b * 2.)]);
            assert_eq!(complex.factor(), Status::Okay);
            assert_eq!(
                complex.solve(&[c(b), c(b * 2.)]),
                [Complex::ZERO, Complex::real(1.)]
            );
            assert_eq!(
                complex.solve_transposed(&[c(a), c(b * 2.)]),
                [Complex::ZERO, Complex::real(1.)]
            );
            let mut fresh = Sparse::from_originals(
                2,
                [(1, 1, c(a)), (1, 2, c(b)), (2, 1, c(a)), (2, 2, c(b * 2.))],
            )
            .unwrap();
            assert_eq!(fresh.order_and_factor(1e-3, 0.), Status::Okay);
            assert_eq!(
                fresh.solve(&[c(b), c(b * 2.)]),
                [Complex::ZERO, Complex::real(1.)]
            );
        }

        let smallest = c(f64::from_bits(1));
        let mut m = Sparse::from_originals(1, [(1, 1, smallest)]).unwrap();
        assert_eq!(m.order_and_factor(1e-3, 0.), Status::Okay);
        assert_eq!(m.solve(&[smallest]), [Complex::real(1.)]);
        assert_eq!(m.solve_transposed(&[smallest]), [Complex::real(1.)]);
    }

    #[test]
    fn bulk_column_finalization_preserves_nonmonotonic_links() {
        // Translate all unknowns first, then insert a densely populated
        // column in deliberately unsorted internal row order.
        let mut entries: Vec<_> = (1..=8).map(|i| (i, i, 20.)).collect();
        entries.extend([7, 3, 8, 2, 6, 4, 5].map(|r| (r, 1, r as f64)));
        entries.extend([8, 6, 5, 4, 3, 1, 7].map(|r| (r, 2, -1.)));
        let a = Sparse::from_originals(8, entries.iter().copied()).unwrap();
        let b = build(8, &entries);
        assert_eq!(a.topology.first_in_col, b.topology.first_in_col);
        assert_eq!(a.topology.diag, b.topology.diag);
        for (a, b) in a.topology.elements.iter().zip(&b.topology.elements) {
            assert_eq!(
                (a.row, a.col, a.next_in_col, a.next_in_row),
                (b.row, b.col, b.next_in_col, b.next_in_row)
            );
        }
        assert_eq!(a.values, b.values);
    }

    #[test]
    fn bulk_column_builder_rejects_duplicate_originals() {
        let error = Sparse::from_originals(2, [(1, 2, 1.), (2, 2, 2.), (1, 2, 3.)]).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("duplicate original sparse element")
        );
    }

    #[test]
    fn batched_original_builder_preserves_translation_and_creation_order() {
        let entries = [
            (4, 2, 2.),
            (2, 1, 1.),
            (1, 4, 3.),
            (2, 2, 4.),
            (1, 1, 5.),
            (4, 4, 6.),
        ];
        let mut a = Sparse::from_originals(5, entries).unwrap();
        let mut b = build(5, &entries);
        assert_eq!(a.size, b.size);
        assert_eq!(a.topology.int_to_ext_row, b.topology.int_to_ext_row);
        assert_eq!(a.topology.int_to_ext_col, b.topology.int_to_ext_col);
        assert_eq!(a.topology.first_in_col, b.topology.first_in_col);
        assert_eq!(a.topology.diag, b.topology.diag);
        for (a, b) in a.topology.elements.iter().zip(&b.topology.elements) {
            assert_eq!(
                (a.row, a.col, a.next_in_col, a.next_in_row),
                (b.row, b.col, b.next_in_col, b.next_in_row)
            );
        }
        assert_eq!(a.values, b.values);
        a.mna_preorder();
        b.mna_preorder();
        assert_eq!(a.order_and_factor(1e-3, 0.), Status::Okay);
        assert_eq!(b.order_and_factor(1e-3, 0.), Status::Okay);
        assert_eq!(a.topology.int_to_ext_row, b.topology.int_to_ext_row);
        assert_eq!(a.topology.int_to_ext_col, b.topology.int_to_ext_col);
        assert!(
            a.values
                .iter()
                .map(|v| v.to_bits())
                .eq(b.values.iter().map(|v| v.to_bits()))
        );
        assert_eq!(
            a.solve(&[1., -0., 2., 3., 4.]),
            b.solve(&[1., -0., 2., 3., 4.])
        );
    }

    #[test]
    fn traversal_plan_matches_links_and_is_shared_until_topology_changes() {
        let entries = [
            (1, 2, 3.),
            (2, 1, 2.),
            (2, 2, 4.),
            (2, 3, 1.),
            (3, 3, 4.),
            (3, 4, 1.),
            (4, 4, 4.),
            (4, 1, 1.),
        ];
        let mut m = Sparse::from_originals(4, entries).unwrap();
        m.mna_preorder();
        assert_eq!(m.order_and_factor(1e-3, 0.), Status::Okay);
        let plan = std::sync::Arc::clone(m.lifecycle.plan().unwrap());
        let FactorPlan::Compact(runs) = plan.as_ref() else {
            panic!("small matrix uses the compact plan");
        };
        for i in 1..=m.size {
            let mut col = Vec::new();
            let mut e = m.topology.first_in_col[i];
            while e != NONE {
                col.push((e, m.topology.elements[e].row));
                e = m.topology.elements[e].next_in_col;
            }
            assert!(
                runs.columns
                    .all(i)
                    .iter()
                    .map(|entry| entry.indices())
                    .eq(col)
            );
            let mut row = Vec::new();
            let mut e = m.topology.first_in_row[i];
            while e != NONE {
                row.push((e, m.topology.elements[e].col));
                e = m.topology.elements[e].next_in_row;
            }
            assert!(
                runs.rows
                    .after_diagonal(i)
                    .iter()
                    .map(|entry| entry.indices())
                    .eq(row.iter().copied().filter(|&(_, col)| col > i))
            );
            assert!(
                runs.rows
                    .before_diagonal(i)
                    .iter()
                    .map(|entry| entry.indices())
                    .eq(row.iter().copied().filter(|&(_, col)| col < i))
            );
        }
        let rhs = [5., 6., 7., 8.];
        let x = m.solve(&rhs);
        let xt = m.solve_transposed(&rhs);
        m.reload_originals(&entries.map(|(_, _, v)| v));
        assert_eq!(m.factor(), Status::Okay);
        assert_eq!(m.solve(&rhs), x);
        assert_eq!(m.solve_transposed(&rhs), xt);
        assert!(std::sync::Arc::ptr_eq(m.lifecycle.plan().unwrap(), &plan));
        let mut c = m.into_retyped::<Complex>();
        assert!(std::sync::Arc::ptr_eq(c.lifecycle.plan().unwrap(), &plan));
        // Insert a missing structural entry after a completed ordering.
        let h = c.get_element(1, 4).unwrap();
        c.set(h, Complex::real(2.));
        assert!(c.lifecycle.plan().is_none());
    }

    #[test]
    fn compact_and_wide_runs_preserve_factor_and_solve_bits() {
        assert_eq!(std::mem::size_of::<RunEntry<u32>>(), 8);
        assert_eq!(std::mem::size_of::<AxisRun<u32>>(), 12);
        let entries = [
            (1, 1, 4.),
            (1, 2, 1.),
            (2, 2, 5.),
            (2, 3, 2.),
            (3, 1, -1.),
            (3, 3, 6.),
        ];
        let input = entries.map(|(_, _, v)| v);
        let mut narrow = build(3, &entries);
        assert_eq!(narrow.order_and_factor(1e-3, 0.), Status::Okay);
        let mut wide = narrow.clone();
        wide.lifecycle = Lifecycle::Factored(std::sync::Arc::new(FactorPlan::Wide(
            wide.compile_runs().unwrap(),
        )));
        narrow.reload_originals(&input);
        wide.reload_originals(&input);
        assert_eq!(narrow.factor(), Status::Okay);
        assert_eq!(wide.factor(), Status::Okay);
        assert!(
            narrow
                .values
                .iter()
                .map(|v| v.to_bits())
                .eq(wide.values.iter().map(|v| v.to_bits()))
        );
        let rhs = [2., -0., 1.];
        for (a, b) in [
            (narrow.solve(&rhs), wide.solve(&rhs)),
            (narrow.solve_transposed(&rhs), wide.solve_transposed(&rhs)),
        ] {
            assert!(
                a.iter()
                    .map(|v| v.to_bits())
                    .eq(b.iter().map(|v| v.to_bits()))
            );
        }
        let input = input.map(|v| Complex::new(v, v * 0.125));
        let mut narrow = narrow.into_retyped::<Complex>();
        let mut wide = wide.into_retyped::<Complex>();
        narrow.reload_originals(&input);
        wide.reload_originals(&input);
        assert_eq!(narrow.factor(), Status::Okay);
        assert_eq!(wide.factor(), Status::Okay);
        let bits = |z: &Complex| (z.re.to_bits(), z.im.to_bits());
        assert!(
            narrow
                .values
                .iter()
                .map(bits)
                .eq(wide.values.iter().map(bits))
        );
        let rhs = rhs.map(Complex::real);
        for (a, b) in [
            (narrow.solve(&rhs), wide.solve(&rhs)),
            (narrow.solve_transposed(&rhs), wide.solve_transposed(&rhs)),
        ] {
            assert!(a.iter().map(bits).eq(b.iter().map(bits)));
        }
    }

    #[test]
    fn original_handles_added_after_fill_keep_storage_attribution() {
        let entries = (1..=6).flat_map(|i| {
            let next = i % 6 + 1;
            [(i, i, 4.), (i, next, -1.), (next, i, -1.)]
        });
        let mut frame = Sparse::from_originals(6, entries).unwrap();
        assert_eq!(frame.order_and_factor(1e-3, 0.), Status::Okay);
        let initial = frame.storage();
        assert_eq!(initial.original_entries, 18);
        assert_eq!(initial.fill_entries(), 6);
        frame.clear();
        let (row, col) = (1..=6)
            .flat_map(|row| (1..=6).map(move |col| (row, col)))
            .find(|&(row, col)| {
                let r = frame.topology.ext_to_int_row[row];
                let c = frame.topology.ext_to_int_col[col];
                frame.topology.locate_in_col(Slot::Head(c), r).1 == NONE
            })
            .unwrap();
        frame.get_element(row, col).unwrap();
        let extended = frame.storage();
        assert_eq!(extended.original_entries, initial.original_entries + 1);
        assert_eq!(extended.factor_entries, initial.factor_entries + 1);
        assert_eq!(extended.fill_entries(), initial.fill_entries());
    }

    #[test]
    fn failed_reorder_cannot_publish_a_partial_traversal() {
        let entries = [
            (1, 1, 4.),
            (1, 2, 1.),
            (1, 3, 1.),
            (2, 1, 1.),
            (2, 2, 5.),
            (2, 3, 1.),
            (3, 1, 1.),
            (3, 2, 1.),
            (3, 3, 6.),
        ];
        let input = entries.map(|(_, _, v)| v);
        let mut a = build(3, &entries);
        assert_eq!(a.order_and_factor(1e-3, 0.), Status::Okay);
        let saved = a.clone();
        let rhs = [6., 7., 8.];
        let expected = saved.solve(&rhs);
        a.clear();
        // Force a threshold failure at the first old pivot, and allow
        // a different diagonal pivot before encountering singularity.
        a.values[a.topology.diag[2]] = 2.;
        assert_eq!(a.order_and_factor(1e-3, 0.), Status::Singular);
        assert!(!a.is_factored());
        assert!(!a.can_refactor());
        assert!(
            matches!(a.lifecycle, Lifecycle::PartialOrder),
            "C retains its partial ordering after singular failure"
        );
        assert!(a.lifecycle.plan().is_none());
        let mut threshold = a.clone();
        a.reload_originals(&input);
        threshold.reload_originals(&input);
        assert_eq!(threshold.order_and_factor(0., 0.), Status::Okay);
        assert_eq!(a.factor(), Status::Okay);
        let work = a.last_attempt();
        assert_eq!(work.order_attempts, 1);
        assert_eq!(work.fixed_refactors, 0);
        assert_eq!(work.successful_factors, 1);
        assert_eq!(a.topology.int_to_ext_row, threshold.topology.int_to_ext_row);
        assert_eq!(a.topology.int_to_ext_col, threshold.topology.int_to_ext_col);
        assert_eq!(a.values, threshold.values);
        let x = a.solve(&rhs);
        for v in x {
            assert!((v - 1.).abs() < 1e-14);
        }
        assert_eq!(saved.solve(&rhs), expected);
    }

    #[test]
    fn loaded_scalar_transition_preserves_original_bits_and_zero_fillins() {
        let entries = [
            (1, 1, 4.),
            (1, 2, 1.),
            (2, 2, 5.),
            (2, 3, 1.),
            (3, 3, 6.),
            (3, 1, 1.),
        ];
        let mut real = Sparse::from_originals(3, entries).unwrap();
        assert_eq!(real.order_and_factor(1e-3, 0.), Status::Okay);
        assert!(real.values.len() > entries.len());
        let real = std::sync::Arc::new(real);
        let before = real.solve(&[1.; 3]);
        let loaded = entries.map(|(_, _, v)| Complex::new(v, -0.));
        let complex =
            Sparse::retype_shared_loaded(std::sync::Arc::clone(&real), loaded.into_iter());
        assert!(!complex.is_factored());
        assert!(std::sync::Arc::ptr_eq(&real.topology, &complex.topology));
        assert!(
            complex.values[..loaded.len()]
                .iter()
                .zip(loaded)
                .all(|(a, b)| a.re.to_bits() == b.re.to_bits() && a.im.to_bits() == b.im.to_bits())
        );
        assert!(
            complex.values[loaded.len()..]
                .iter()
                .all(|v| v.re.to_bits() == 0 && v.im.to_bits() == 0)
        );
        assert_eq!(real.solve(&[1.; 3]), before);
    }

    #[test]
    fn reload_fork_preserves_snapshots_without_copying_numeric_scratch() {
        let entries = [(1, 1, 4.), (1, 2, 1.), (2, 1, 2.), (2, 2, 5.)];
        let rhs = [5., 7.];
        let mut a = build(2, &entries);
        assert_eq!(a.order_and_factor(1e-3, 0.), Status::Okay);
        a.reload_originals(&entries.map(|(_, _, v)| v));
        assert_eq!(a.factor(), Status::Okay);
        let saved = std::sync::Arc::new(a);
        let expected = saved.solve(&rhs);
        let mut changed = std::sync::Arc::clone(&saved);
        Sparse::reload_shared(&mut changed, [8., 1., 2., 5.].into_iter());
        let next = std::sync::Arc::get_mut(&mut changed).unwrap();
        assert_eq!(&next.values[..4], &[8., 1., 2., 5.]);
        assert!(next.values[4..].iter().all(|v| *v == 0.));
        assert!(next.ordering_scratch.row.is_empty());
        assert!(next.ordering_scratch.col.is_empty());
        assert!(next.ordering_scratch.products.is_empty());
        assert_eq!(next.factor(), Status::Okay);
        assert_ne!(next.solve(&rhs), expected);
        assert!(std::sync::Arc::ptr_eq(
            next.lifecycle.plan().unwrap(),
            saved.lifecycle.plan().unwrap()
        ));
        assert_eq!(saved.solve(&rhs), expected);
        // Force threshold failure and a new pivot order: deferred counts
        // must be allocated before searching and creating fill-ins.
        next.reload_originals(&[0., 1., 2., 5.]);
        assert_eq!(next.order_and_factor(1e-3, 0.), Status::Okay);
        assert!(!next.ordering_scratch.row.is_empty());
        assert_eq!(saved.solve(&rhs), expected);
        let c: Sparse<Complex> = Sparse::retype_shared(std::sync::Arc::clone(&saved));
        assert!(c.values.iter().all(|v| *v == Complex::ZERO));
        assert!(c.ordering_scratch.row.is_empty());
        assert_eq!(saved.solve(&rhs), expected);
    }

    #[test]
    fn snapshot_topology_is_shared_until_structural_mutation() {
        let mut a = build(3, &[(1, 1, 4.), (2, 2, 5.), (1, 2, 1.)]);
        assert_eq!(a.order_and_factor(1e-3, 0.), Status::Okay);
        let saved = a.clone();
        assert!(std::sync::Arc::ptr_eq(&a.topology, &saved.topology));
        let rhs = [5., 5., -0.];
        let answer = saved.solve(&rhs);
        a.reload_originals(&[6., 5., 1.]);
        assert_eq!(a.factor(), Status::Okay);
        assert!(std::sync::Arc::ptr_eq(&a.topology, &saved.topology));
        assert_ne!(a.solve(&rhs), answer);
        let mut complex = a.into_retyped::<Complex>();
        assert!(std::sync::Arc::ptr_eq(&complex.topology, &saved.topology));
        // Lookup of an already allocated element does not detach.
        complex.get_element(1, 1).unwrap();
        assert!(std::sync::Arc::ptr_eq(&complex.topology, &saved.topology));
        // Add a previously untranslated unknown after successful ordering.
        let h = complex.get_element(3, 3).unwrap();
        assert!(!std::sync::Arc::ptr_eq(&complex.topology, &saved.topology));
        complex.clear();
        for i in 1..=3 {
            let h = complex.get_element(i, i).unwrap();
            complex.set(h, Complex::real(i as f64));
        }
        complex.set(h, Complex::real(3.));
        assert_eq!(complex.order_and_factor(1e-3, 0.), Status::Okay);
        assert_eq!(
            complex.solve(&[Complex::real(1.), Complex::real(2.), Complex::real(3.)]),
            [Complex::real(1.); 3]
        );
        assert_eq!(saved.solve(&rhs), answer);
        assert_eq!(saved.untranslated().collect::<Vec<_>>(), [3]);
    }

    #[test]
    fn contiguous_original_reload_clears_fillins_and_preserves_handles() {
        let entries = [
            (1, 1, 4.),
            (1, 2, 1.),
            (2, 2, 4.),
            (2, 3, 1.),
            (3, 3, 4.),
            (3, 4, 1.),
            (4, 4, 4.),
            (4, 1, 1.),
        ];
        let mut m = Sparse::from_originals(4, entries).unwrap();
        assert_eq!(m.order_and_factor(1e-3, 0.), Status::Okay);
        assert!(m.values.len() > entries.len());
        let rhs = [5.; 4];
        let before = m.solve(&rhs);
        let ptr = m.values.as_ptr();
        let topology = m.topology.elements.as_ptr();
        let input = entries.map(|(_, _, v)| v);
        m.reload_originals(&input);
        assert!(!m.is_factored());
        assert_eq!(&m.values[..entries.len()], &input);
        assert!(m.values[entries.len()..].iter().all(|v| *v == 0.));
        assert_eq!(m.factor(), Status::Okay);
        assert_eq!(m.solve(&rhs), before);
        assert_eq!(m.values.as_ptr(), ptr);
        assert_eq!(m.topology.elements.as_ptr(), topology);
        m.reload_originals_iter(input.into_iter());
        assert_eq!(m.factor(), Status::Okay);
        assert_eq!(m.solve(&rhs), before);
    }

    #[test]
    fn scalar_transition_moves_structural_allocations() {
        let mut real = build(2, &[(1, 1, 2.), (1, 2, 1.), (2, 1, 3.), (2, 2, 4.)]);
        assert_eq!(real.order_and_factor(1e-3, 0.), Status::Okay);
        let elements = real.topology.elements.as_ptr();
        let diagonal = real.topology.diag.as_ptr();
        let rows = real.topology.int_to_ext_row.as_ptr();
        let columns = real.topology.int_to_ext_col.as_ptr();
        let complex = real.into_retyped::<Complex>();
        assert_eq!(complex.topology.elements.as_ptr(), elements);
        assert_eq!(complex.topology.diag.as_ptr(), diagonal);
        assert_eq!(complex.topology.int_to_ext_row.as_ptr(), rows);
        assert_eq!(complex.topology.int_to_ext_col.as_ptr(), columns);
        assert!(!complex.is_factored());
        assert!(complex.values.iter().all(|v| v.is_zero()));
        let real = complex.into_retyped::<f64>();
        assert_eq!(real.topology.elements.as_ptr(), elements);
        assert_eq!(real.topology.diag.as_ptr(), diagonal);
        assert_eq!(real.topology.int_to_ext_row.as_ptr(), rows);
        assert_eq!(real.topology.int_to_ext_col.as_ptr(), columns);
    }

    #[test]
    fn singleton_scan_preserves_c_sentinel_and_reverse_order() {
        // C places the current step's product at size+1. A singleton at
        // step must be returned from that sentinel before scanning other
        // positions. Otherwise the scan proceeds from the highest index.
        let m = build(4, &[(1, 1, 5.), (2, 2, 2.), (3, 3, 3.), (4, 4, 4.)]);
        let mut scratch = Markowitz::default();
        scratch.ensure_size(m.size).unwrap();
        for i in 1..=4 {
            scratch.products.set(i, 0);
        }
        scratch.singletons = 4;
        let pivot = m.search_for_singleton(&mut scratch, 1).unwrap();
        assert_eq!(m.topology.elements[pivot].row, 1);

        // The sentinel is not a singleton: the search selects the highest
        // remaining zero-product diagonal, never reading a negative index.
        for (i, v) in [1, 1, 0, 0].into_iter().enumerate() {
            scratch.products.set(i + 1, v);
        }
        scratch.singletons = 2;
        let pivot = m.search_for_singleton(&mut scratch, 1).unwrap();
        assert_eq!(m.topology.elements[pivot].row, 4);
    }

    #[test]
    fn solves_with_pivoting_and_mna_twins() {
        // Voltage source branch (row/col 3) with zero diagonal.
        let entries = [
            (1, 1, 1e-3),
            (1, 2, -1e-3),
            (2, 1, -1e-3),
            (2, 2, 3e-3),
            (1, 3, 1.0),
            (3, 1, 1.0),
        ];
        let mut m = build(3, &entries);
        m.mna_preorder();
        assert_eq!(m.order_and_factor(1e-3, 1e-13), Status::Okay);
        let b = [0.0, 0.0, 5.0];
        let x = m.solve(&b);
        assert!(residual(&entries, &x, &b) < 1e-12, "{x:?}");
        let xt = m.solve_transposed(&b);
        let transposed: Vec<_> = entries.iter().map(|&(r, c, v)| (c, r, v)).collect();
        assert!(residual(&transposed, &xt, &b) < 1e-12, "{xt:?}");
    }

    #[test]
    fn refactor_reuses_pivots_and_detects_zero_diagonal() {
        let entries = [(1, 1, 2.0), (1, 2, 1.0), (2, 1, 1.0), (2, 2, 3.0)];
        let mut m = build(2, &entries);
        assert_eq!(m.order_and_factor(1e-3, 0.0), Status::Okay);
        let handles: Vec<_> = entries
            .iter()
            .map(|&(r, c, _)| m.get_element(r, c).unwrap())
            .collect();
        m.clear();
        for (h, v) in handles.iter().zip([4.0, 1.0, 1.0, 5.0]) {
            m.set(*h, v);
        }
        assert_eq!(m.factor(), Status::Okay);
        let x = m.solve(&[5.0, 6.0]);
        assert!((4.0 * x[0] + x[1] - 5.0).abs() < 1e-14);
        m.clear();
        assert_eq!(m.factor(), Status::ZeroDiag);
    }

    #[test]
    fn fixed_order_refactor_does_not_read_previous_workspace_contents() {
        // A nonmonotonic, fill-producing original pattern, with exact
        // structural zeros deliberately retained in its assembly.
        let n = 30;
        let mut seed = 19_u64;
        let mut entries = Vec::new();
        let mut seen = std::collections::HashSet::new();
        for i in (1..=n).rev() {
            entries.push((i, i, 20.));
            seen.insert((i, i));
            for j in 0..5 {
                seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                let c = (seed >> 32) as usize % n + 1;
                if seen.insert((i, c)) {
                    entries.push((i, c, if j == 0 { -0. } else { 0.125 }));
                }
            }
        }
        let mut real = Sparse::from_originals(n, entries.iter().copied()).unwrap();
        assert_eq!(real.order_and_factor(1e-3, 0.), Status::Okay);
        assert!(real.values.len() > entries.len());
        let mut clean = real.clone();
        let mut poisoned = real.clone();
        let values = entries.iter().map(|&(_, _, v)| v * 1.5);
        clean.reload_originals_iter(values.clone());
        poisoned.reload_originals_iter(values);
        assert_eq!(clean.factor_with_workspace(&mut Vec::new()), Status::Okay);
        assert_eq!(
            poisoned.factor_with_workspace(&mut vec![f64::NAN; n + 1]),
            Status::Okay
        );
        assert!(
            clean
                .values
                .iter()
                .map(|v| v.to_bits())
                .eq(poisoned.values.iter().map(|v| v.to_bits()))
        );
        let mut complex = real.into_retyped::<Complex>();
        complex.reload_originals_iter(entries.iter().map(|&(_, _, v)| Complex::new(v, v * 0.25)));
        let mut poisoned = complex.clone();
        assert_eq!(complex.factor_with_workspace(&mut Vec::new()), Status::Okay);
        assert_eq!(
            poisoned.factor_with_workspace(&mut vec![Complex::new(f64::NAN, f64::NAN); n + 1]),
            Status::Okay
        );
        let bits = |v: &Complex| (v.re.to_bits(), v.im.to_bits());
        assert!(
            complex
                .values
                .iter()
                .map(bits)
                .eq(poisoned.values.iter().map(bits))
        );
    }

    #[test]
    fn refactor_workspace_survives_success_and_singularity() {
        let entries = [(1, 1, 3.), (1, 2, 1.), (2, 1, 1.), (2, 2, 4.)];
        let mut m = build(2, &entries);
        assert_eq!(m.order_and_factor(1e-3, 0.), Status::Okay);
        let handles: Vec<_> = entries
            .iter()
            .map(|&(r, c, _)| m.get_element(r, c).unwrap())
            .collect();
        let load = |m: &mut Sparse<f64>, values: &[f64]| {
            m.clear();
            for (&h, &v) in handles.iter().zip(values) {
                m.set(h, v);
            }
        };
        let mut workspace = Vec::new();
        load(&mut m, &[3., 1., 1., 4.]);
        assert_eq!(m.factor_with_workspace(&mut workspace), Status::Okay);
        let initial = m.solve(&[4., 5.]);
        let address = workspace.as_ptr();
        let capacity = workspace.capacity();
        load(&mut m, &[1., 1., 1., 1.]);
        assert_eq!(m.factor_with_workspace(&mut workspace), Status::ZeroDiag);
        load(&mut m, &[3., 1., 1., 4.]);
        assert_eq!(m.factor_with_workspace(&mut workspace), Status::Okay);
        assert_eq!(workspace.as_ptr(), address);
        assert_eq!(workspace.capacity(), capacity);
        assert_eq!(
            m.factors()
                .unwrap()
                .solve(&[4., 5.], &mut workspace)
                .unwrap(),
            initial
        );
        assert_eq!(workspace.as_ptr(), address);
    }

    /// External row 2 has no element: as with ngspice's `Translate`, the
    /// internal order is 2 and `spSolve` (in place in `SMPsolve`) leaves the
    /// right-hand side in the untranslated entry.
    #[test]
    fn untranslated_rows_keep_their_right_hand_side() {
        let entries = [(1, 1, 2.0), (1, 3, 1.0), (3, 1, 1.0), (3, 3, 4.0)];
        let mut m = build(3, &entries);
        assert!(!m.fully_translated());
        assert_eq!(m.untranslated().collect::<Vec<_>>(), [2]);
        m.mna_preorder();
        assert_eq!(m.order_and_factor(1e-3, 1e-13), Status::Okay);
        let x = m.solve(&[3.0, 7.5, 5.0]);
        assert_eq!(x[1], 7.5);
        assert!((2.0 * x[0] + x[2] - 3.0).abs() < 1e-15 && (x[0] + 4.0 * x[2] - 5.0).abs() < 1e-15);
        assert_eq!(m.solve_transposed(&[3.0, -1.0, 5.0])[1], -1.0);
        let (mantissa, exponent) = m.factors().unwrap().determinant().unwrap();
        assert!((mantissa * 2f64.powi(i32::try_from(exponent).unwrap()) - 7.0).abs() < 1e-12);
    }

    /// No element passes the absolute threshold, so `SearchEntireMatrix`
    /// takes the largest and reports `spSMALL_PIVOT` with its position.
    #[test]
    fn small_pivot_is_located() {
        let mut m = build(2, &[(1, 1, 1e-14), (2, 2, 2e-14)]);
        assert_eq!(m.order_and_factor(1e-3, 1e-13), Status::SmallPivot);
        assert_eq!(m.small_pivot(), Some((2, 2)));
        assert_eq!(m.solve(&[1e-14, 2e-14]), [1.0, 1.0]);
        let mut m = build(2, &[(1, 1, 1e-14), (2, 2, 2e-14)]);
        assert_eq!(m.order_and_factor(1e-3, 1e-16), Status::Okay);
        assert_eq!(m.small_pivot(), None);
    }

    #[test]
    fn normalized_gather_guard_is_lane_local() {
        let numerators = [6., 3., -0., f64::MIN_POSITIVE];
        let solutions = [1., 1., 0., 0.];
        let (values, valid) = RealGatherRow::gather(
            f64x4::from_array(numerators),
            0.5,
            [(3., f64x4::from_array(solutions)), (0., f64x4::splat(1.))].into_iter(),
        );
        assert_eq!(valid.to_array(), [true, false, true, false]);
        for i in 0..4 {
            let (value, scalar_valid) = RealGatherRow::gather(
                numerators[i],
                0.5,
                [(3., solutions[i]), (0., 1.)].into_iter(),
            );
            assert_eq!(valid.test(i), scalar_valid);
            assert_eq!(values[i].to_bits(), value.to_bits());
        }
        assert!(!RealGatherRow::gather(1., 2., [(f64::MAX, 0.5), (0., 1.)].into_iter()).1);
        assert!(!RealGatherRow::gather(1., 0.5, [(f64::MIN_POSITIVE, 1.), (0., 1.)].into_iter()).1);
    }

    #[test]
    fn coarse_gather_certificate_preserves_precise_decisions() {
        // A row that the coarse bound cannot certify but the old running
        // certificate accepts must retain its normalized result, not fall
        // back to raw scatter.
        let terms = [(1., 1. - 32. * f64::EPSILON), (1., 0.)];
        let old = RealGatherRow::precise(1., 1., terms.into_iter());
        assert!(old.1);
        let coarse = RealGatherRow::accumulate::<false>(1., 1., terms.into_iter());
        let bound = coarse.magnitude_sum * (24. * f64::EPSILON);
        assert!(!coarse.finish(bound).1);
        assert_eq!(RealGatherRow::gather(1., 1., terms.into_iter()), old);

        let mut seed = 0xc0a5_0a1e_f1ed_3b17_u64;
        let mut random = |wide: bool| {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            if wide {
                // Include subnormal values and the full finite exponent
                // range, including overflow/underflow in products.
                let value = f64::from_bits(seed);
                if value.is_finite() {
                    value
                } else {
                    f64::from_bits(seed & !(1_u64 << 52))
                }
            } else {
                f64::from(u32::try_from(seed >> 32).unwrap()) / 2_147_483_648. - 1.
            }
        };
        for length in [0, 1, 2, 3, 8, 31, 128, 1024] {
            for trial in 0..64 {
                let wide = trial % 2 == 0;
                let inverse = random(wide);
                let numerator = std::array::from_fn(|_| random(wide));
                let terms: Vec<_> = (0..length)
                    .map(|index| {
                        let upper = if index % 7 == 0 { 0. } else { random(wide) };
                        let solved = std::array::from_fn(|lane| {
                            if (index + lane) % 11 == 0 {
                                -0.
                            } else {
                                random(wide)
                            }
                        });
                        (upper, f64x4::from_array(solved))
                    })
                    .collect();
                let tile = f64x4::from_array(numerator);
                let (actual, valid) = RealGatherRow::gather(tile, inverse, terms.iter().copied());
                let (expected, old_valid) =
                    RealGatherRow::precise(tile, inverse, terms.iter().copied());
                assert_eq!(valid, old_valid);
                for lane in 0..RHS_BATCH_WIDTH {
                    let scalar_terms = terms.iter().map(|(u, x)| (*u, x[lane]));
                    let (scalar, scalar_valid) =
                        RealGatherRow::gather(numerator[lane], inverse, scalar_terms.clone());
                    let (old_scalar, old_scalar_valid) =
                        RealGatherRow::precise(numerator[lane], inverse, scalar_terms);
                    assert_eq!(scalar_valid, old_scalar_valid);
                    assert_eq!(valid.test(lane), scalar_valid);
                    for (a, b) in [
                        (actual[lane], expected[lane]),
                        (scalar, old_scalar),
                        (actual[lane], scalar),
                    ] {
                        assert!(a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan()));
                    }
                }
            }
        }
    }

    #[test]
    fn substitution_uses_raw_rhs_ratios_in_both_triangular_directions() {
        // Exact solution is (1/3, 0), although 1/3 is not representable.
        // A more accurate subtraction of 5 - 15*rounded(1/3) would
        // expose a false residual. Preserve the actual RHS relation.
        for (row, col) in [(2, 1), (1, 2)] {
            let mut m = build(2, &[(1, 1, 3.), (2, 2, 2.), (row, col, 15.)]);
            assert_eq!(m.order_and_factor(1e-3, 0.), Status::Okay);
            let solve = |rhs: &[f64]| {
                if row == 2 {
                    m.solve(rhs)
                } else {
                    m.solve_transposed(rhs)
                }
            };
            assert_eq!(solve(&[1., 5.]), [1. / 3., 0.]);
            let perturbed = 5.0_f64.next_up();
            assert_eq!(solve(&[1., perturbed]), [1. / 3., (perturbed - 5.) / 2.]);
            let rhs = [
                1., 5., 1., perturbed, -1., -5., 0., -0., 2., 10., -2., -10., 1., 4., -1., -4., 2.,
                9.,
            ];
            check_batch(&m, &rhs, |value| [value.to_bits(), 0]);
        }
        for (row, col) in [(1, 2), (2, 1)] {
            let mut m = build(2, &[(1, 1, 2.), (2, 2, 3.), (row, col, 15.)]);
            assert_eq!(m.order_and_factor(1e-3, 0.), Status::Okay);
            let solve = |rhs: &[f64]| {
                if row == 1 {
                    m.solve(rhs)
                } else {
                    m.solve_transposed(rhs)
                }
            };
            assert_eq!(solve(&[5., 1.]), [0., 1. / 3.]);
            let perturbed = 5.0_f64.next_up();
            assert_eq!(solve(&[perturbed, 1.]), [(perturbed - 5.) / 2., 1. / 3.]);
            let rhs = [
                5., 1., perturbed, 1., 7., 2., -0., 0., -5., -1., 10., 2., -7., -2., 1., 1., 5., 2.,
            ];
            check_batch(&m, &rhs, |value| [value.to_bits(), 0]);
        }
        let pivot = Complex::new(3., 7.);
        let rhs = Complex::new(1., 2.);
        let lower = pivot * Complex::real(5.);
        for (row, col) in [(2, 1), (1, 2)] {
            let mut m = Sparse::from_originals(
                2,
                [(1, 1, pivot), (2, 2, Complex::real(2.)), (row, col, lower)],
            )
            .unwrap();
            assert_eq!(m.order_and_factor(1e-3, 0.), Status::Okay);
            let b = [rhs, rhs * Complex::real(5.)];
            let x = if row == 2 {
                m.solve(&b)
            } else {
                m.solve_transposed(&b)
            };
            assert!((x[0] - rhs / pivot).magnitude() < 1e-15);
            assert_eq!(x[1], Complex::ZERO);
        }
    }

    #[test]
    fn proportional_rows_do_not_acquire_spurious_pivots() {
        for x in 1..=100 {
            for y in 1..=100 {
                for z in 1..=7 {
                    let (x, y, z) = (f64::from(x), f64::from(y), f64::from(z));
                    let entries = [(1, 1, x), (1, 2, y), (2, 1, z * x), (2, 2, z * y)];
                    let mut m = build(2, &entries);
                    assert_eq!(
                        m.order_and_factor(1e-3, 0.),
                        Status::Singular,
                        "{entries:?}"
                    );
                }
            }
        }
        let entries = [(1, 1, 3.), (1, 2, 7.), (2, 1, 9.), (2, 2, 22.)];
        let mut m = build(2, &entries);
        assert_eq!(m.order_and_factor(1e-3, 0.), Status::Okay);
        m.reload_originals(&[3., 7., 9., 21.]);
        assert_eq!(m.factor(), Status::ZeroDiag);
        m.reload_originals(&[3., 7., 9., 21.0_f64.next_up()]);
        assert_eq!(m.factor(), Status::Okay);
        let (mantissa, exponent) = m.factors().unwrap().determinant().unwrap();
        let expected = 3. * (21.0_f64.next_up() - 21.);
        let actual =
            crate::primitives::numeric::scale_power(mantissa, i32::try_from(exponent).unwrap());
        assert!((actual / expected - 1.).abs() < 1e-14);
    }

    #[test]
    fn complex_proportional_rows_and_nearby_nonsingular_rows() {
        for a in [
            Complex::new(3., 7.),
            Complex::new(7., -3.),
            Complex::new(1., 0.),
        ] {
            for b in [Complex::new(-5., 11.), Complex::new(13., 17.)] {
                for ratio in [
                    Complex::new(3., -2.),
                    Complex::new(0., 7.),
                    Complex::real(5.),
                ] {
                    let c = a * ratio;
                    let d = b * ratio;
                    let entries = [(1, 1, a), (1, 2, b), (2, 1, c), (2, 2, d)];
                    let mut m = Sparse::from_originals(2, entries).unwrap();
                    assert_eq!(
                        m.order_and_factor(1e-3, 0.),
                        Status::Singular,
                        "{entries:?}"
                    );
                    let different = Complex::new(d.re.next_up(), d.im);
                    let mut m = Sparse::from_originals(
                        2,
                        [(1, 1, a), (1, 2, b), (2, 1, c), (2, 2, different)],
                    )
                    .unwrap();
                    assert_eq!(m.order_and_factor(1e-3, 0.), Status::Okay);
                    m.reload_originals(&[a, b, c, d]);
                    assert_eq!(m.factor(), Status::ZeroDiag);
                }
            }
        }
    }

    #[test]
    fn singular_and_determinant() {
        let mut m = build(2, &[(1, 1, 1.0), (1, 2, 2.0), (2, 1, 2.0), (2, 2, 4.0)]);
        assert_eq!(m.order_and_factor(1e-3, 0.0), Status::Singular);
        let mut m = build(2, &[(1, 1, 3.0), (1, 2, 1.0), (2, 1, 2.0), (2, 2, 4.0)]);
        assert_eq!(m.order_and_factor(1e-3, 0.0), Status::Okay);
        let (mantissa, exponent) = m.factors().unwrap().determinant().unwrap();
        assert!((mantissa * 2f64.powi(i32::try_from(exponent).unwrap()) - 10.0).abs() < 1e-12);
    }

    #[test]
    fn complex_solve_matches_product() {
        let entries = [
            (1, 1, Complex::new(1.0, 2.0)),
            (1, 2, Complex::new(0.0, -1.0)),
            (2, 1, Complex::new(0.0, -1.0)),
            (2, 2, Complex::new(2.0, 0.5)),
        ];
        let mut m: Sparse<Complex> = Sparse::new(2).unwrap();
        for &(r, c, v) in &entries {
            let e = m.get_element(r, c).unwrap();
            m.add(e, v);
        }
        assert_eq!(m.order_and_factor(1e-3, 0.0), Status::Okay);
        let b = [Complex::real(1.0), Complex::new(0.0, 1.0)];
        let x = m.solve(&b);
        for row in 1..=2 {
            let mut sum = Complex::ZERO;
            for &(r, c, v) in &entries {
                if r == row {
                    sum = sum + v * x[c - 1];
                }
            }
            assert!((sum - b[row - 1]).magnitude() < 1e-14);
        }
    }

    #[test]
    fn larger_ladder_fills_and_solves() {
        let n = 200;
        let mut entries = Vec::new();
        for i in 1..=n {
            entries.push((i, i, 2.0 + 1e-3 * i as f64));
            if i > 1 {
                entries.push((i, i - 1, -1.0));
                entries.push((i - 1, i, -1.0));
            }
        }
        entries.push((1, n, 0.5));
        entries.push((n, 1, 0.25));
        let mut m = build(n, &entries);
        assert_eq!(m.order_and_factor(1e-3, 1e-13), Status::Okay);
        let b: Vec<_> = (0..n).map(|i| (i as f64).sin()).collect();
        let x = m.solve(&b);
        assert!(residual(&entries, &x, &b) < 1e-10);
    }
    #[test]
    #[ignore = "manual repeated reload/refactor benchmark"]
    fn sparse_reload_refactor_benchmark() {
        use std::{hint::black_box, time::Instant};
        const N: usize = 1024;
        const RUNS: u32 = 1000;
        let entries: Vec<_> = (1..=N)
            .flat_map(|i| [(i, i, 5.), (i, i % N + 1, -0.25), (i % N + 1, i, -0.5)])
            .collect();
        let originals: Vec<_> = entries.iter().map(|&(_, _, v)| v).collect();
        let mut m = Sparse::from_originals(N, entries).unwrap();
        assert_eq!(m.order_and_factor(1e-3, 0.), Status::Okay);
        let mut scratch = Vec::new();
        for _ in 0..3 {
            let start = Instant::now();
            for _ in 0..RUNS {
                m.reload_originals(black_box(&originals));
                assert_eq!(m.factor_with_workspace(&mut scratch), Status::Okay);
                black_box(&m);
            }
            let duration = start.elapsed();
            let rhs = vec![1.; N];
            let x = m.solve(&rhs);
            let checksum = x.iter().fold(0_u64, |sum, v| sum.wrapping_add(v.to_bits()));
            eprintln!(
                "reload/refactor n={N} runs={RUNS} elapsed={duration:?} checksum={checksum:016x}"
            );
        }
    }
}
