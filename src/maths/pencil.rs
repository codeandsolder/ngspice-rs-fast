//! Roots of `det(A + s E) = 0`: ngspice's pole-zero root search.
//!
//! A port of `CKTpzFindZeros` and its helpers
//! (`src/spicelib/analysis/cktpzstr.c`) with `NIpzSym2`/`NIpzMuller`
//! (`src/maths/ni/nipzmeth.c`). Each trial loads `A + s E` into one SPARSE
//! matrix (`maths::sparse13`), reorders it as `SMPcReorder(1e-30, 0)`
//! does (keeping the previous pivots when they pass the threshold test) and
//! takes the determinant (`SMPcDProd`: SPARSE's base-10 mantissa converted to
//! a base-2 mantissa/exponent pair). The search walks the real axis with the
//! trial list built by a frequency scan, brackets sign changes with a deflated
//! quadratic fit (`NIpzSym2`), deflates every found root out of the stored
//! trials (`CKTpzVerify`), and continues into the complex plane with Muller's
//! method from the curvature hint `NIpzK`.
//!
//! C's file-scope state (`CKTpzTrapped`, `NIpzK`, the guess bounds, move
//! history) lives in `Search`; `Last_Move` starts at zero for each search
//! instead of carrying over from a previous one. C's iteration-limit warning
//! goes to standard error and the roots found so far are returned, as in C.
//! The reference's aberration flag has no producer and its flatness counter is
//! debug-only. A determinant that vanishes at every trial is C's `E_SHORT`.

// C-compatible numerical expressions are kept in their reference order:
// Clippy's fused multiply-add suggestions change IEEE-754 rounding.
#![expect(
    clippy::float_cmp,
    reason = "C device branches compare floating-point values exactly, including nonfinite cases"
)]
#![expect(
    clippy::manual_midpoint,
    reason = "C source midpoint arithmetic retains its original evaluation and rounding order"
)]
#![expect(
    clippy::too_many_lines,
    reason = "a complete translated C numerical routine remains contiguous for equation parity review"
)]
#![expect(
    clippy::many_single_char_names,
    reason = "physical-model equation symbols are kept consistent with the ngspice C reference"
)]
#![expect(
    clippy::while_float,
    reason = "the C iterative floating-point loop must preserve its original termination and rounding order"
)]
#![expect(
    clippy::suboptimal_flops,
    reason = "SPICE numerical reference parity requires the original non-fused C operation order"
)]

use crate::maths::Matrix;
use crate::maths::linear::numerical;
use crate::maths::sparse13::{Sparse, Status};
use crate::primitives::{Complex, Real, SpiceResult};

mod search;

#[cfg(test)]
pub(crate) use search::check_c_trace;

/// The roots of a pencil, see [`pencil_roots`].
#[derive(Debug, Clone, PartialEq)]
pub struct PencilRoots {
    /// Roots in `PZpost` order: ascending real part, then imaginary part,
    /// repeated by multiplicity, each complex root followed by its conjugate.
    pub roots: Vec<Complex>,
}

/// One device's contribution to a pole-zero matrix element.
///
/// One device's pole-zero contribution to one matrix element, as a C
/// `DEVpzLoad` adds it: `*elt += g + c * s.real` and `*(elt + 1) += c *
/// s.imag` (`diopzld.c`), `*elt += g` alone (`respzld.c`) or `c * s.real`
/// alone (`cappzld.c`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PencilTerm {
    /// Zero-based matrix row.
    pub row: usize,
    /// Zero-based matrix column.
    pub col: usize,
    /// The `s`-independent part `g`, if the device adds one.
    pub conductance: Option<Real>,
    /// The coefficient `c` of `s`, if the device adds one.
    pub capacitance: Option<Real>,
}

impl PencilTerm {
    /// The value this term adds at `s`.
    fn at(&self, s: Cx) -> Complex {
        match (self.conductance, self.capacitance) {
            (Some(g), Some(c)) => Complex::new(g + c * s.re, c * s.im),
            (Some(g), None) => Complex::new(g, 0.0),
            (None, Some(c)) => Complex::new(c * s.re, c * s.im),
            (None, None) => Complex::ZERO,
        }
    }
}

/// Construct and load the pole-zero matrix at a trial complex frequency.
///
/// The matrix `CKTpzSetup` builds and `CKTpzLoad` (`cktpzld.c`) loads at
/// each trial `s`: the elements are created in order (the circuit's setup
/// elements, then the drive), every trial clears them, adds the device
/// [`PencilTerm`]s in load order, adds the solution column to the balance
/// column (`SMPcAddCol`), zeroes the solution column (`SMPcZeroCol`) and
/// stores the unit drive `+1`/`-1` in the input rows of the solution column.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct PencilLoad {
    /// The matrix order.
    pub order: usize,
    /// Zero-based `(row, col)` elements to create before loading, in
    /// creation order (SPARSE numbers rows and columns by first appearance).
    /// Term and drive positions not listed are created after these.
    pub elements: Vec<(usize, usize)>,
    /// Device contributions in load order.
    pub terms: Vec<PencilTerm>,
    /// The solution column (`PZsolution_col`), if any.
    pub solution: Option<usize>,
    /// The balance column (`PZbalance_col`), if any.
    pub balance: Option<usize>,
    /// The rows receiving `+1` and `-1` in the solution column
    /// (`PZdrive_pptr`/`PZdrive_nptr`).
    pub drive: [Option<usize>; 2],
    /// The relative pivot threshold the matrix holds. `CKTpzFindZeros`
    /// reorders with `SMPcReorder(matrix, 1e-30, 0.0)`, and a relative
    /// threshold outside `(0, 1]` keeps the one `ckt->CKTmatrix` stores: the
    /// `.options pivrel` (`CKTpivotRelTol`) of the operating point's
    /// `SMPreorder`. `0` keeps SPARSE's default, `1e-3`.
    pub relative_threshold: f64,
}

impl PencilLoad {
    /// A load of the plain pencil `A + s E` with one term per nonzero
    /// position and the elements created row by row.
    ///
    /// # Errors
    /// Non-square or mismatched matrices.
    pub fn from_dense(a: &Matrix, e: &Matrix) -> SpiceResult<Self> {
        let n = a.rows();
        if a.cols() != n || e.rows() != n || e.cols() != n {
            return Err(numerical(
                "pole-zero search",
                format!(
                    "needs square matrices of one size, got {}x{} and {}x{}",
                    a.rows(),
                    a.cols(),
                    e.rows(),
                    e.cols()
                ),
            ));
        }
        let mut terms = Vec::new();
        for row in 0..n {
            for col in 0..n {
                let g = a.get(row, col).unwrap_or(0.0);
                let c = e.get(row, col).unwrap_or(0.0);
                if g != 0.0 || c != 0.0 {
                    terms.push(PencilTerm {
                        row,
                        col,
                        conductance: Some(g),
                        capacitance: (c != 0.0).then_some(c),
                    });
                }
            }
        }
        Ok(Self {
            order: n,
            elements: terms.iter().map(|t| (t.row, t.col)).collect(),
            terms,
            ..Self::default()
        })
    }

    fn validate(&self) -> SpiceResult<()> {
        let n = self.order;
        if n == 0 {
            return Err(numerical("pole-zero search", "needs a nonempty matrix"));
        }
        let inside = |i: usize| i < n;
        let positions = self
            .elements
            .iter()
            .copied()
            .chain(self.terms.iter().map(|t| (t.row, t.col)));
        if !positions.into_iter().all(|(r, c)| inside(r) && inside(c))
            || ![self.solution, self.balance, self.drive[0], self.drive[1]]
                .into_iter()
                .flatten()
                .all(inside)
        {
            return Err(numerical(
                "pole-zero search",
                "matrix position out of range",
            ));
        }
        if self.terms.iter().any(|t| {
            !t.conductance.unwrap_or(0.0).is_finite() || !t.capacitance.unwrap_or(0.0).is_finite()
        }) {
            return Err(numerical("pole-zero search", "non-finite matrix entry"));
        }
        Ok(())
    }
}

/// `SPcomplex` arithmetic with the exact C macro formulas (`complex.h`).
#[derive(Debug, Clone, Copy, PartialEq)]
struct Cx {
    re: Real,
    im: Real,
}

impl Cx {
    const ZERO: Self = Self { re: 0.0, im: 0.0 };
    const fn new(re: Real, im: Real) -> Self {
        Self { re, im }
    }
    fn scale(self, k: Real) -> Self {
        Self::new(k * self.re, k * self.im)
    }
    /// `C_SUBEQ`.
    fn sub(self, b: Self) -> Self {
        Self::new(self.re - b.re, self.im - b.im)
    }
    /// `C_ADDEQ`.
    fn add(self, b: Self) -> Self {
        Self::new(self.re + b.re, self.im + b.im)
    }
    /// `C_MULEQ` / `C_MUL`.
    fn mul(self, b: Self) -> Self {
        Self::new(
            self.re * b.re - self.im * b.im,
            self.re * b.im + self.im * b.re,
        )
    }
    /// `C_DIV` / `C_DIVEQ`.
    fn div(self, b: Self) -> Self {
        // The C numerator's imaginary sum is the same finite binary64
        // operation as Complex::div (including signed zeros). Share its
        // exceptional exponent-scaled path as well.
        let quotient = Complex::new(self.re, self.im) / Complex::new(b.re, b.im);
        Self::new(quotient.re, quotient.im)
    }
    /// `C_MAG2`: squared modulus as the real part.
    fn mag2(self) -> Self {
        Self::new(self.re * self.re + self.im * self.im, 0.0)
    }
    /// `C_SQRT`.
    fn sqrt(self) -> Self {
        if self.im == 0.0 {
            if self.re < 0.0 {
                Self::new(0.0, (-self.re).sqrt())
            } else {
                Self::new(self.re.sqrt(), 0.0)
            }
        } else {
            let mag = self.re.hypot(self.im);
            let a = (mag - self.re) / 2.0;
            if a <= 0.0 {
                let re = mag.sqrt();
                Self::new(re, self.im / (2.0 * re))
            } else {
                let a = a.sqrt();
                Self::new(self.im / (2.0 * a), a)
            }
        }
    }
    fn is_zero(self) -> bool {
        self.re == 0.0 && self.im == 0.0
    }
}

/// `C_NORM`: scale `a` into `(0.5, 1]` per component, tracking the binary
/// exponent in `mag` (set to zero for a zero value).
fn c_norm(a: &mut Cx, mag: &mut i64) -> SpiceResult<()> {
    if !a.re.is_finite() || !a.im.is_finite() {
        return Err(numerical(
            "pole-zero search",
            "non-finite complex normalization",
        ));
    }
    if a.is_zero() {
        *mag = 0;
        return Ok(());
    }
    while a.re.abs() > 1.0 || a.im.abs() > 1.0 {
        *mag += 1;
        a.re /= 2.0;
        a.im /= 2.0;
    }
    while a.re.abs() <= 0.5 && a.im.abs() <= 0.5 {
        *mag -= 1;
        a.re *= 2.0;
        a.im *= 2.0;
    }
    Ok(())
}

/// `R_NORM`.
fn r_norm(a: &mut Real, mag: &mut i64) -> SpiceResult<()> {
    if !a.is_finite() {
        return Err(numerical(
            "pole-zero search",
            "non-finite real normalization",
        ));
    }
    if *a == 0.0 {
        *mag = 0;
        return Ok(());
    }
    while a.abs() > 1.0 {
        *mag += 1;
        *a /= 2.0;
    }
    while a.abs() < 0.5 {
        *mag -= 1;
        *a *= 2.0;
    }
    Ok(())
}

/// Materialize a scaled real in C's repeated doubling/halving order.
/// Stopping once zero or infinity is reached keeps the work bounded by
/// binary64's exponent range, even for enormous determinant exponents.
fn materialize_power(mut value: Real, exponent: i64) -> Real {
    for _ in 0..exponent.unsigned_abs() {
        if value == 0.0 || !value.is_finite() {
            break;
        }
        if exponent > 0 {
            value *= 2.0;
        } else {
            value /= 2.0;
        }
    }
    value
}

fn normalize_muller(a: &mut Cx, b: &mut Cx, c: &mut Cx) -> SpiceResult<()> {
    if [*a, *b, *c]
        .iter()
        .any(|z| !z.re.is_finite() || !z.im.is_finite())
    {
        return Err(numerical(
            "pole-zero search",
            "non-finite Muller coefficients",
        ));
    }
    while a.re.abs() > 1.0
        || a.im.abs() > 1.0
        || b.re.abs() > 1.0
        || b.im.abs() > 1.0
        || c.re.abs() > 1.0
        || c.im.abs() > 1.0
    {
        *a = a.scale(0.5);
        *b = b.scale(0.5);
        *c = c.scale(0.5);
    }
    Ok(())
}

/// `zaddeq`: `a * 2^amag = x * 2^xmag + y * 2^ymag`, normalized.
fn zaddeq(x: Real, xmag: i64, y: Real, ymag: i64) -> SpiceResult<(Real, i64)> {
    if !x.is_finite() || !y.is_finite() {
        return Err(numerical("pole-zero search", "non-finite scaled sum"));
    }
    let (mut x, mut y) = (x, y);
    let mut amag;
    if xmag > ymag {
        amag = xmag;
        if xmag > 50 + ymag {
            y = 0.0;
        } else {
            for _ in 0..(xmag - ymag) {
                y /= 2.0;
            }
        }
    } else {
        amag = ymag;
        if ymag > 50 + xmag {
            x = 0.0;
        } else {
            for _ in 0..(ymag - xmag) {
                x /= 2.0;
            }
        }
    }
    let mut a = x + y;
    r_norm(&mut a, &mut amag)?;
    Ok((a, amag))
}

fn sgn(x: Real) -> i32 {
    if x < 0.0 { -1 } else { i32::from(x != 0.0) }
}

/// C `logb`: the unbiased binary exponent of a nonzero finite value.
fn logb(x: Real) -> SpiceResult<Real> {
    crate::primitives::numeric::binary_exponent(x)
        .map(Real::from)
        .ok_or_else(|| numerical("pole-zero search", "invalid determinant component exponent"))
}

/// `SMPcDProd`: SPARSE's `mantissa * 10^p` as a base-2 mantissa/exponent.
fn base2_determinant(mantissa: Complex, p: i32) -> SpiceResult<(Cx, i64)> {
    let (mut re, mut im) = (mantissa.re, mantissa.im);
    let mut y = Real::from(p) * std::f64::consts::LN_10 / std::f64::consts::LN_2;
    let x = y.trunc();
    y -= x;
    let z = 2.0_f64.powf(y);
    re *= z;
    im *= z;
    // C treats a zero component as exponent zero, even when the other
    // component has a negative exponent. Do not drop that convention.
    let mut y = if re == 0.0 { 0.0 } else { logb(re)? };
    let z = if im == 0.0 { 0.0 } else { logb(im)? };
    if y < z {
        y = z;
    }
    let exponent = crate::primitives::exact_signed_integer(x + y)
        .ok_or_else(|| numerical("pole-zero search", "invalid determinant exponent"))?;
    let shift = crate::primitives::truncating_signed_integer::<i32>(y)
        .map(|shift| 2.0_f64.powi(-shift))
        .ok_or_else(|| numerical("pole-zero search", "invalid determinant shift"))?;
    Ok((Cx::new(re * shift, im * shift), exponent))
}

/// One compiled pencil load, retaining element handles and the SPARSE
/// pivot frame between trials. The root-search state does not own matrix
/// assembly or depend on the order of the handle vectors.
struct LoadedPencil<'a> {
    load: &'a PencilLoad,
    sparse: Sparse<Complex>,
    handles: Vec<super::sparse13::SolverElement>,
    drive: [Option<super::sparse13::SolverElement>; 2],
}

impl<'a> LoadedPencil<'a> {
    fn new(load: &'a PencilLoad) -> SpiceResult<Self> {
        let mut sparse = Sparse::new(load.order)?;
        for &(row, col) in &load.elements {
            sparse.get_element(row + 1, col + 1)?;
        }
        let handles = load
            .terms
            .iter()
            .map(|t| sparse.get_element(t.row + 1, t.col + 1))
            .collect::<SpiceResult<_>>()?;
        let mut drive = [None, None];
        if let Some(col) = load.solution {
            for (slot, row) in drive.iter_mut().zip(load.drive) {
                if let Some(row) = row {
                    *slot = Some(sparse.get_element(row + 1, col + 1)?);
                }
            }
        }
        // As in C, external rows not referenced by elements do not enter
        // the determinant.
        Ok(Self {
            load,
            sparse,
            handles,
            drive,
        })
    }

    /// `CKTpzLoad` + `SMPcReorder` + `SMPcDProd`; no MNA preorder.
    fn determinant_at(&mut self, s: Cx) -> SpiceResult<Option<(Cx, i64)>> {
        if !s.re.is_finite() || !s.im.is_finite() {
            return Err(numerical("pole-zero search", "non-finite trial frequency"));
        }
        let Self {
            load,
            sparse,
            handles,
            drive,
        } = self;
        sparse.clear();
        for (term, &handle) in load.terms.iter().zip(handles.iter()) {
            sparse.add(handle, term.at(s));
        }
        if let Some(solution) = load.solution {
            if let Some(balance) = load.balance {
                sparse.add_col(balance + 1, solution + 1)?;
            }
            sparse.zero_col(solution + 1)?;
        }
        for (element, value) in drive.iter().zip([1.0, -1.0]) {
            if let Some(element) = *element {
                let im = sparse.value(element).im;
                sparse.set(element, Complex::new(value, im));
            }
        }
        if !sparse.values_are_finite() {
            return Err(numerical("pole-zero search", "non-finite trial matrix"));
        }
        match sparse.order_and_factor(load.relative_threshold, 1.0e-30) {
            Status::Singular | Status::ZeroDiag => Ok(None),
            Status::Resource => Err(numerical(
                "pole-zero search",
                "sparse factor storage capacity exceeded",
            )),
            Status::NeedsReload => Err(numerical(
                "pole-zero search",
                "matrix coefficients must be reloaded before factorization",
            )),
            Status::InvalidPlan => Err(numerical(
                "pole-zero search",
                "ordered sparse topology lacks a valid traversal plan",
            )),
            Status::NonFinite => Err(numerical(
                "pole-zero search",
                "non-finite or unrepresentable LU factor",
            )),
            Status::Okay | Status::SmallPivot => {
                let (mantissa, exponent) = sparse.factors()?.determinant()?;
                base2_determinant(mantissa, exponent).map(Some)
            }
        }
    }
}

/// Roots of `det(A + s E)` found by ngspice's pole-zero search
/// ([`PencilLoad::from_dense`] + [`pencil_load_roots`]).
///
/// # Errors
/// As [`pencil_load_roots`], and non-square or mismatched matrices.
pub fn pencil_roots(a: &Matrix, e: &Matrix) -> SpiceResult<PencilRoots> {
    pencil_load_roots(&PencilLoad::from_dense(a, e)?)
}

/// Roots of the determinant of `load` as a function of `s`, found by
/// ngspice's pole-zero search (`CKTpzFindZeros`).
///
/// # Errors
/// An empty matrix, out-of-range positions or non-finite terms, a
/// determinant that vanishes at every trial (C `E_SHORT`, reported as a
/// "singular pencil"), or the search losing numerical stability (C
/// `E_PANIC`).
pub fn pencil_load_roots(load: &PencilLoad) -> SpiceResult<PencilRoots> {
    load.validate()?;
    search::find_zeros(load).map(|roots| PencilRoots { roots })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn matrix(rows: &[&[Real]]) -> Matrix {
        let mut m = Matrix::zeros(rows.len(), rows.len()).unwrap();
        for (r, row) in rows.iter().enumerate() {
            for (c, value) in row.iter().enumerate() {
                m.set(r, c, *value).unwrap();
            }
        }
        m
    }

    fn close(found: &[Complex], expected: &[Complex]) {
        assert_eq!(found.len(), expected.len(), "{found:?}");
        for (f, e) in found.iter().zip(expected) {
            assert!(
                (f.re - e.re).abs() <= 1e-6 * e.re.abs().max(1.0)
                    && (f.im - e.im).abs() <= 1e-6 * e.im.abs().max(1.0),
                "{found:?} vs {expected:?}"
            );
        }
    }

    #[test]
    fn huge_scaled_exponents_and_muller_coefficients_are_bounded() {
        assert!(materialize_power(1.0, i64::MAX).is_infinite());
        assert_eq!(materialize_power(1.0, i64::MIN), 0.0);
        assert_eq!(
            materialize_power(-1.0, i64::MIN).to_bits(),
            (-0.0_f64).to_bits()
        );
        let (_, high) = base2_determinant(Complex::real(1.), i32::MAX).unwrap();
        let (_, low) = base2_determinant(Complex::real(1.), i32::MIN).unwrap();
        assert!(logb(0.).is_err());
        assert!(logb(f64::INFINITY).is_err());
        assert!(base2_determinant(Complex::real(f64::NAN), 0).is_err());
        assert!(high > i64::from(i32::MAX));
        assert!(low < i64::from(i32::MIN));
        let (_, mag) = zaddeq(0.75, high, 0.75, low).unwrap();
        assert_eq!(mag, high);
        let mut a = Cx::new(2., -1.);
        let mut b = Cx::new(-4., 0.);
        let mut c = Cx::new(0.25, 0.);
        normalize_muller(&mut a, &mut b, &mut c).unwrap();
        assert_eq!(a, Cx::new(0.5, -0.25));
        assert_eq!(b, Cx::new(-1., 0.));
        assert_eq!(c, Cx::new(0.0625, 0.));
        c.re = f64::INFINITY;
        assert!(normalize_muller(&mut a, &mut b, &mut c).is_err());
        assert_eq!(logb(f64::from_bits(1)).unwrap(), -1074.);
        assert_eq!(
            logb(f64::from_bits(f64::MIN_POSITIVE.to_bits() - 1)).unwrap(),
            -1023.
        );
    }

    #[test]
    fn nonfinite_trial_and_normalization_are_errors() {
        let load = PencilLoad {
            order: 1,
            elements: vec![(0, 0)],
            terms: vec![PencilTerm {
                row: 0,
                col: 0,
                conductance: Some(1.),
                capacitance: Some(2.),
            }],
            ..PencilLoad::default()
        };
        let mut compiled = LoadedPencil::new(&load).unwrap();
        assert!(compiled.determinant_at(Cx::new(f64::MAX, 0.)).is_err());
        assert!(compiled.determinant_at(Cx::new(f64::INFINITY, 0.)).is_err());
        assert!(c_norm(&mut Cx::new(f64::INFINITY, 0.), &mut 0).is_err());
        let mut infinity = f64::INFINITY;
        let mut exponent = 0;
        assert!(r_norm(&mut infinity, &mut exponent).is_err());
        assert!(zaddeq(f64::MAX, 0, f64::MAX, 0).is_err());
        assert!(compiled.determinant_at(Cx::new(1., 0.)).unwrap().is_some());
    }

    #[test]
    fn real_roots_of_an_rc_ladder() {
        // det(A + sE) = (1 + s)(2 + s) for a diagonal pencil.
        let a = matrix(&[&[1.0, 0.0], &[0.0, 2.0]]);
        let e = matrix(&[&[1.0, 0.0], &[0.0, 1.0]]);
        let roots = pencil_roots(&a, &e).unwrap().roots;
        close(&roots, &[Complex::new(-2.0, 0.0), Complex::new(-1.0, 0.0)]);
    }

    #[test]
    fn complex_pair_of_an_lc_tank() {
        // s^2 + 0.2 s + 1: roots -0.1 +- j sqrt(0.99).
        let a = matrix(&[&[0.2, 1.0], &[-1.0, 0.0]]);
        let e = matrix(&[&[1.0, 0.0], &[0.0, 1.0]]);
        let roots = pencil_roots(&a, &e).unwrap().roots;
        let w = 0.99_f64.sqrt();
        close(&roots, &[Complex::new(-0.1, w), Complex::new(-0.1, -w)]);
    }

    #[test]
    fn a_pencil_without_finite_roots_has_none() {
        let a = matrix(&[&[1.0, 0.0], &[0.0, 1.0]]);
        let e = matrix(&[&[0.0, 0.0], &[0.0, 0.0]]);
        assert!(pencil_roots(&a, &e).unwrap().roots.is_empty());
    }

    #[test]
    fn a_determinant_vanishing_everywhere_is_c_e_short() {
        let a = matrix(&[&[1.0, 1.0], &[1.0, 1.0]]);
        let e = matrix(&[&[1.0, 1.0], &[1.0, 1.0]]);
        let error = pencil_roots(&a, &e).unwrap_err().to_string();
        assert!(error.contains("singular pencil"), "{error}");
    }
}
