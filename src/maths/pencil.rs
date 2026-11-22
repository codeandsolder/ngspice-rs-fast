//! Roots of `det(A + s E) = 0`: ngspice's pole-zero root search.
//!
//! A port of `CKTpzFindZeros` and its helpers
//! (`src/spicelib/analysis/cktpzstr.c`) with `NIpzSym2`/`NIpzMuller`
//! (`src/maths/ni/nipzmeth.c`). Each trial loads `A + s E` into one SPARSE
//! matrix (`maths::sparse13`), reorders it as `SMPcReorder(1e-30, 0)`
//! does (keeping the previous pivots when they pass the threshold test) and
//! takes the determinant (`SMPcDProd`: SPARSE's base-10 mantissa converted to
//! a base-2 mantissa/exponent pair). The search walks the real axis with the
//! trial list kept sorted by `s`, brackets sign changes with a deflated
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

const NITER_LIM: usize = 200;

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

/// Stable identity of a linked trial. IDs are allocated by the arena and
/// never reused, including after a trial is pruned during deflation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct TrialId(usize);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TrialKind {
    Sample,
    Minimum,
    Root(std::num::NonZeroU32),
}

#[derive(Debug, Clone, Copy)]
struct Trial {
    s: Cx,
    f_def: Cx,
    mag_def: i64,
    kind: TrialKind,
    count: u32,
    next: Option<TrialId>,
    prev: Option<TrialId>,
}

impl Trial {
    const fn new(s: Cx, f_def: Cx, mag_def: i64, kind: TrialKind) -> Self {
        Self {
            s,
            f_def,
            mag_def,
            kind,
            count: 0,
            next: None,
            prev: None,
        }
    }

    const fn multiplicity(&self) -> u32 {
        match self.kind {
            TrialKind::Root(multiplicity) => multiplicity.get(),
            TrialKind::Sample | TrialKind::Minimum => 0,
        }
    }

    const fn is_root(&self) -> bool {
        matches!(self.kind, TrialKind::Root(_))
    }
}

fn invalid_trial() -> crate::primitives::SpiceError {
    numerical("pole-zero search", "invalid or removed trial identity")
}

/// Owns linked trials and their sorted traversal. A proposed frequency
/// stays outside the arena until it is evaluated and linked; collisions
/// may return an existing ID without allocating an abandoned candidate.
#[derive(Default)]
struct TrialArena {
    entries: Vec<Option<Trial>>,
    head: Option<TrialId>,
    anchor: Option<TrialId>,
}

impl TrialArena {
    fn get(&self, id: TrialId) -> SpiceResult<&Trial> {
        self.entries
            .get(id.0)
            .and_then(Option::as_ref)
            .ok_or_else(invalid_trial)
    }

    fn get_mut(&mut self, id: TrialId) -> SpiceResult<&mut Trial> {
        self.entries
            .get_mut(id.0)
            .and_then(Option::as_mut)
            .ok_or_else(invalid_trial)
    }

    fn insert_after(&mut self, prev: Option<TrialId>, mut trial: Trial) -> SpiceResult<TrialId> {
        let next = match prev {
            Some(id) => self.get(id)?.next,
            None => self.head,
        };
        // Validate both links before changing the list.
        if let Some(id) = next {
            self.get(id)?;
        }
        let id = TrialId(self.entries.len());
        trial.prev = prev;
        trial.next = next;
        self.entries.push(Some(trial));
        if let Some(prev) = prev {
            self.get_mut(prev)?.next = Some(id);
        } else {
            self.head = Some(id);
        }
        if let Some(next) = next {
            self.get_mut(next)?.prev = Some(id);
        } else if prev.is_none() {
            self.anchor = Some(id);
        }
        Ok(id)
    }

    fn remove(&mut self, id: TrialId) -> SpiceResult<()> {
        let Trial { prev, next, .. } = *self.get(id)?;
        if let Some(prev) = prev {
            self.get_mut(prev)?.next = next;
        } else {
            self.head = next;
        }
        if let Some(next) = next {
            self.get_mut(next)?.prev = prev;
        }
        if self.anchor == Some(id) {
            self.anchor = next.or(prev);
        }
        self.entries[id.0] = None;
        Ok(())
    }

    fn triple(&self, ids: Triple) -> SpiceResult<[&Trial; 3]> {
        Ok([
            self.get(ids.left)?,
            self.get(ids.middle)?,
            self.get(ids.right)?,
        ])
    }

    fn frequency(&self, id: TrialId) -> SpiceResult<Cx> {
        Ok(self.get(id)?.s)
    }

    fn seek(&self, start: Option<TrialId>, direction: Seek) -> SpiceResult<Option<TrialId>> {
        let mut current = start;
        if direction != Seek::Anchor {
            current = self.neighbor(current, direction)?;
        }
        while let Some(id) = current {
            if self.get(id)?.kind == TrialKind::Sample {
                return Ok(Some(id));
            }
            current = self.neighbor(current, direction)?;
        }
        Ok(None)
    }

    fn neighbor(&self, id: Option<TrialId>, direction: Seek) -> SpiceResult<Option<TrialId>> {
        id.map_or(Ok(None), |id| {
            let t = self.get(id)?;
            Ok(match direction {
                Seek::Left => t.prev,
                Seek::Right | Seek::Anchor => t.next,
            })
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Side {
    Left,
    Right,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Seek {
    Left,
    Anchor,
    Right,
}

impl From<Side> for Seek {
    fn from(side: Side) -> Self {
        match side {
            Side::Left => Self::Left,
            Side::Right => Self::Right,
        }
    }
}

impl Seek {
    const fn guess(self) -> Real {
        match self {
            Self::Left => -1.0,
            Self::Anchor => 0.0,
            Self::Right => 1.0,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Move {
    None,
    Near(Side),
    Mid(Side),
    Far(Side),
}

impl Move {
    fn split_side(self) -> SpiceResult<Side> {
        match self {
            Self::Mid(Side::Left) | Self::Near(Side::Right) => Ok(Side::Left),
            Self::Mid(Side::Right) | Self::Near(Side::Left) => Ok(Side::Right),
            Self::None | Self::Far(_) => Err(numerical(
                "pole-zero search",
                "trapped search has no usable last move (C calls controlled_exit)",
            )),
        }
    }
}

/// The three numerical regions C denotes by CKTpzTrapped = 1, 2, 3.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Region {
    LeftCrossing,
    RightCrossing,
    Minimum,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Trap {
    Free,
    Bracket(Region),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Triple {
    left: TrialId,
    middle: TrialId,
    right: TrialId,
}

/// A temporary representation of C's three slots, used only to make list
/// transitions. The persistent neighborhood below is classified before a
/// strategy can consume it. In particular a complete bracket cannot lack a
/// required point, and a complex seed is distinct from a Muller triple.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
struct Window {
    left: Option<TrialId>,
    middle: Option<TrialId>,
    right: Option<TrialId>,
}

impl From<Triple> for Window {
    fn from(t: Triple) -> Self {
        Self {
            left: Some(t.left),
            middle: Some(t.middle),
            right: Some(t.right),
        }
    }
}

impl Window {
    const fn centered(middle: TrialId) -> Self {
        Self {
            left: None,
            middle: Some(middle),
            right: None,
        }
    }

    const fn triple(self) -> Option<Triple> {
        match (self.left, self.middle, self.right) {
            (Some(left), Some(middle), Some(right)) => Some(Triple {
                left,
                middle,
                right,
            }),
            _ => None,
        }
    }
}

/// Incomplete states are partitioned by C's fill priority (middle, right,
/// left). The slots outside the missing point retain their original meaning,
/// even after falling off either end of the trial list.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Incomplete {
    Middle {
        left: Option<TrialId>,
        right: Option<TrialId>,
    },
    Right {
        left: Option<TrialId>,
        middle: TrialId,
    },
    Left {
        middle: TrialId,
        right: TrialId,
    },
}

impl From<Incomplete> for Window {
    fn from(value: Incomplete) -> Self {
        match value {
            Incomplete::Middle { left, right } => Self {
                left,
                middle: None,
                right,
            },
            Incomplete::Right { left, middle } => Self {
                left,
                middle: Some(middle),
                right: None,
            },
            Incomplete::Left { middle, right } => Self {
                left: None,
                middle: Some(middle),
                right: Some(right),
            },
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct RealBracket(Triple);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ComplexSeed {
    First {
        latest: TrialId,
        previous: Option<TrialId>,
    },
    Gap {
        latest: TrialId,
        oldest: TrialId,
    },
}

impl From<ComplexSeed> for Window {
    fn from(seed: ComplexSeed) -> Self {
        match seed {
            ComplexSeed::First { latest, previous } => Self {
                left: Some(latest),
                middle: previous,
                right: None,
            },
            ComplexSeed::Gap { latest, oldest } => Self {
                left: Some(latest),
                middle: None,
                right: Some(oldest),
            },
        }
    }
}

impl ComplexSeed {
    const fn guess(self) -> (TrialId, Real) {
        match self {
            Self::First { latest, .. } => (latest, 1.0e8),
            Self::Gap { latest, .. } => (latest, 1.0e12),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Neighborhood {
    Incomplete(Incomplete),
    Real(RealBracket),
    ComplexSeed(ComplexSeed),
    ComplexTriple(Triple),
    Minimum {
        middle: TrialId,
        left: Option<TrialId>,
        right: Option<TrialId>,
    },
}

impl Neighborhood {
    fn classify(window: Window, arena: &TrialArena) -> SpiceResult<Self> {
        // Test the center minimum before the complex left point, exactly
        // as CKTpzStrat does. Do not reorder points or use the imaginary
        // parts of the other slots to change C's choice of kernel.
        let Window {
            left,
            middle,
            right,
        } = window;
        let center_is_minimum = middle
            .map(|id| arena.get(id))
            .transpose()?
            .is_some_and(|t| t.kind == TrialKind::Minimum);
        let left_is_complex = left
            .map(|id| arena.get(id))
            .transpose()?
            .is_some_and(|t| t.s.im != 0.0);
        if let Some(id) = right {
            arena.get(id)?;
        }
        if center_is_minimum {
            let middle = middle.ok_or_else(invalid_trial)?;
            return Ok(Self::Minimum {
                middle,
                left,
                right,
            });
        }
        if left_is_complex {
            let latest = left.ok_or_else(invalid_trial)?;
            return Ok(match (middle, right) {
                (Some(middle), Some(right)) => Self::ComplexTriple(Triple {
                    left: latest,
                    middle,
                    right,
                }),
                (previous, None) => Self::ComplexSeed(ComplexSeed::First { latest, previous }),
                (None, Some(oldest)) => Self::ComplexSeed(ComplexSeed::Gap { latest, oldest }),
            });
        }
        Ok(match (left, middle, right) {
            (Some(left), Some(middle), Some(right)) => Self::Real(RealBracket(Triple {
                left,
                middle,
                right,
            })),
            (left, None, right) => Self::Incomplete(Incomplete::Middle { left, right }),
            (left, Some(middle), None) => Self::Incomplete(Incomplete::Right { left, middle }),
            (None, Some(middle), Some(right)) => {
                Self::Incomplete(Incomplete::Left { middle, right })
            }
        })
    }

    fn window(self) -> Window {
        match self {
            Self::Incomplete(incomplete) => incomplete.into(),
            Self::Real(RealBracket(triple)) | Self::ComplexTriple(triple) => triple.into(),
            Self::ComplexSeed(seed) => seed.into(),
            Self::Minimum {
                middle,
                left,
                right,
            } => Window {
                left,
                middle: Some(middle),
                right,
            },
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Navigation {
    Fill(Incomplete),
    Skip { bracket: RealBracket, side: Side },
    Shift { bracket: RealBracket, side: Side },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Evaluation {
    Guess(Option<TrialId>),
    Quadratic {
        bracket: RealBracket,
        region: Region,
    },
    Split {
        edge: TrialId,
        middle: TrialId,
    },
    ComplexInit(TrialId),
    ComplexGuess(ComplexSeed),
    Muller(Triple),
}

impl Evaluation {
    const fn split(RealBracket(t): RealBracket, side: Side) -> Self {
        let edge = match side {
            Side::Left => t.left,
            Side::Right => t.right,
        };
        Self::Split {
            edge,
            middle: t.middle,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Strategy {
    Navigate(Navigation),
    Evaluate(Evaluation),
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

/// Curvature hint produced by NIpzSym2 and consumed by complex initialization.
#[derive(Default)]
struct Curvature {
    value: Real,
    exponent: i64,
}

impl Curvature {
    const fn usable(&self) -> bool {
        self.value != 0.0 && self.exponent > -10
    }
}

#[cfg(test)]
#[derive(Debug, Clone, Copy, PartialEq)]
enum TraceEvent {
    Proposal(Cx),
    Load(Cx),
    Determinant(Cx, i64),
}

/// One CKTpzFindZeros run with its independent scheduler and trial arena.
struct Search<'a> {
    pencil: LoadedPencil<'a>,
    arena: TrialArena,
    nzeros: usize,
    max_zeros: usize,
    seq_num: usize,
    guess_param: Real,
    high_guess: Real,
    low_guess: Real,
    last_move: Move,
    consec_moves: u32,
    niter: usize,
    trapped: Trap,
    curvature: Curvature,
    #[cfg(test)]
    trace: Vec<TraceEvent>,
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
    Search::new(load)?
        .find_zeros()
        .map(|roots| PencilRoots { roots })
}

impl<'a> Search<'a> {
    fn new(load: &'a PencilLoad) -> SpiceResult<Self> {
        Ok(Self {
            pencil: LoadedPencil::new(load)?,
            arena: TrialArena::default(),
            nzeros: 0,
            max_zeros: load.order,
            seq_num: 1,
            guess_param: 0.0,
            high_guess: -1.0,
            low_guess: 1.0,
            last_move: Move::None,
            consec_moves: 0,
            niter: 0,
            trapped: Trap::Free,
            curvature: Curvature::default(),
            #[cfg(test)]
            trace: Vec::new(),
        })
    }
}

impl Search<'_> {
    /// CKTpzFindZeros: navigate existing trials before proposing a new one.
    fn find_zeros(&mut self) -> SpiceResult<Vec<Complex>> {
        let mut neighborhood = self.reset()?;
        loop {
            let evaluation = loop {
                match self.strategy(neighborhood)? {
                    Strategy::Evaluate(evaluation) => break evaluation,
                    Strategy::Navigate(navigation) => {
                        if self.trapped != Trap::Free {
                            return Err(numerical(
                                "pole-zero search",
                                "trapped search attempted navigation",
                            ));
                        }
                        neighborhood = self.step(navigation)?;
                        let window = neighborhood.window();
                        if window.triple().is_none() {
                            break Evaluation::Guess(window.middle);
                        }
                    }
                }
            };
            self.niter += 1;
            let candidate = self.eval(evaluation)?;
            #[cfg(test)]
            self.trace.push(TraceEvent::Proposal(candidate));
            let new = self.run_trial(candidate, neighborhood.window().middle)?;
            neighborhood = match self.arena.get(new)?.kind {
                TrialKind::Root(_) => {
                    self.verify(new)?;
                    self.niter = 0;
                    self.reset()?
                }
                TrialKind::Minimum => Neighborhood::classify(Window::centered(new), &self.arena)?,
                TrialKind::Sample => self.update_neighborhood(neighborhood, new)?,
            };
            let spread = self.high_guess - self.low_guess;
            let window = neighborhood.window();
            let open = match (window.left, window.right) {
                (Some(left), Some(right)) if self.trapped == Trap::Free => {
                    self.arena.frequency(right)?.re - self.arena.frequency(left)?.re < 1e22
                }
                _ => true,
            };
            if !(spread < 1e40
                && self.nzeros < self.max_zeros
                && self.niter < NITER_LIM
                && spread < 1e35
                && open)
            {
                break;
            }
        }
        if self.nzeros + 1 >= self.seq_num {
            return Err(numerical(
                "pole-zero search",
                "singular pencil: the determinant vanished at every trial (C E_SHORT, \
                 \"The input signal is shorted on the way to the output\")",
            ));
        }
        if self.niter >= NITER_LIM {
            eprintln!(
                "warning: Pole-zero iteration limit reached; giving up after {} trials",
                self.seq_num
            );
        }
        // clear_trials(0) and PZpost: roots are already linked in C order.
        let mut roots = Vec::new();
        let mut current = self.arena.head;
        while let Some(id) = current {
            let trial = self.arena.get(id)?;
            for _ in 0..trial.multiplicity() {
                roots.push(Complex::new(trial.s.re, trial.s.im));
                if trial.s.im != 0.0 {
                    roots.push(Complex::new(trial.s.re, -trial.s.im));
                }
            }
            current = trial.next;
        }
        Ok(roots)
    }

    /// PZeval. Only candidate frequencies, not detached arena entries, exist
    /// before the determinant evaluation and collision handling.
    fn eval(&mut self, evaluation: Evaluation) -> SpiceResult<Cx> {
        self.seq_num += 1;
        let s = match evaluation {
            Evaluation::Guess(center) => {
                if self.high_guess < self.low_guess {
                    self.guess_param = 0.0;
                } else if self.guess_param > 0.0 {
                    self.guess_param = if self.high_guess > 0.0 {
                        self.high_guess * 10.0
                    } else {
                        1.0
                    };
                } else {
                    self.guess_param = if self.low_guess < 0.0 {
                        self.low_guess * 10.0
                    } else {
                        -1.0
                    };
                }
                if self.high_guess < self.guess_param {
                    self.high_guess = self.guess_param;
                }
                if self.low_guess > self.guess_param {
                    self.low_guess = self.guess_param;
                }
                let imag = center
                    .map(|id| self.arena.frequency(id))
                    .transpose()?
                    .map_or(0.0, |s| s.im);
                Cx::new(self.guess_param, imag)
            }
            Evaluation::Quadratic {
                bracket: RealBracket(triple),
                region,
            } => {
                let [t0, t1, t2] = self.arena.triple(triple)?;
                let mut trial = Cx::new(sym2(t0, t1, t2, region, &mut self.curvature)?, 0.0);
                let (s0, s1, s2) = (t0.s, t1.s, t2.s);
                match region {
                    Region::LeftCrossing => {
                        if trial.re < s0.re || trial.re > s1.re {
                            trial.re = (s0.re + s1.re) / 2.0;
                        }
                    }
                    Region::RightCrossing => {
                        if trial.re < s1.re || trial.re > s2.re {
                            trial.re = (s1.re + s2.re) / 2.0;
                        }
                    }
                    Region::Minimum => {
                        if trial.re <= s0.re
                            || (trial.re == s1.re && trial.im == s1.im)
                            || trial.re >= s2.re
                        {
                            trial.re = (s0.re + s2.re) / 2.0;
                            if trial.re == s1.re {
                                trial.re = if matches!(
                                    self.last_move,
                                    Move::Mid(Side::Left) | Move::Near(Side::Right)
                                ) {
                                    (s0.re + s1.re) / 2.0
                                } else {
                                    (s1.re + s2.re) / 2.0
                                };
                            }
                        }
                    }
                }
                trial
            }
            Evaluation::ComplexInit(minimum) => {
                let re = self.arena.frequency(minimum)?.re;
                let im = if self.curvature.usable() {
                    materialize_power(self.curvature.value, self.curvature.exponent)
                } else {
                    10000.0
                };
                self.curvature = Curvature::default();
                Cx::new(re, im)
            }
            Evaluation::ComplexGuess(seed) => {
                let (id, im) = seed.guess();
                Cx::new(self.arena.frequency(id)?.re, im)
            }
            Evaluation::Muller(triple) => muller(self.arena.triple(triple)?)?,
            Evaluation::Split { edge, middle } => Cx::new(
                (self.arena.frequency(edge)?.re + 2.0 * self.arena.frequency(middle)?.re) / 3.0,
                0.0,
            ),
        };
        if !s.re.is_finite() || !s.im.is_finite() {
            return Err(numerical("pole-zero search", "non-finite trial frequency"));
        }
        Ok(s)
    }

    /// CKTpzStrat. Each evaluation or navigation carries all of its inputs.
    fn strategy(&mut self, neighborhood: Neighborhood) -> SpiceResult<Strategy> {
        let mut trap = Trap::Free;
        let suggestion = match neighborhood {
            Neighborhood::Minimum { middle, .. } => {
                Strategy::Evaluate(Evaluation::ComplexInit(middle))
            }
            Neighborhood::ComplexSeed(seed) => Strategy::Evaluate(Evaluation::ComplexGuess(seed)),
            Neighborhood::ComplexTriple(triple) => Strategy::Evaluate(Evaluation::Muller(triple)),
            Neighborhood::Incomplete(incomplete) => {
                Strategy::Navigate(Navigation::Fill(incomplete))
            }
            Neighborhood::Real(bracket) => {
                let [t0, t1, t2] = self.arena.triple(bracket.0)?;
                let quadratic =
                    |region| Strategy::Evaluate(Evaluation::Quadratic { bracket, region });
                let skip = |side| Strategy::Navigate(Navigation::Skip { bracket, side });
                let shift = |side| Strategy::Navigate(Navigation::Shift { bracket, side });
                let split = |side| Strategy::Evaluate(Evaluation::split(bracket, side));
                let mut chosen;
                if sgn(t0.f_def.re) != sgn(t1.f_def.re) {
                    trap = Trap::Bracket(Region::LeftCrossing);
                    chosen = quadratic(Region::LeftCrossing);
                } else if sgn(t1.f_def.re) != sgn(t2.f_def.re) {
                    trap = Trap::Bracket(Region::RightCrossing);
                    chosen = quadratic(Region::RightCrossing);
                } else {
                    let (a, a_mag) = zaddeq(t1.f_def.re, t1.mag_def, -t0.f_def.re, t0.mag_def)?;
                    let (b, b_mag) = zaddeq(t2.f_def.re, t2.mag_def, -t1.f_def.re, t1.mag_def)?;
                    if self.trapped == Trap::Free {
                        let k1 = t1.s.re - t0.s.re;
                        let k2 = t2.s.re - t1.s.re;
                        chosen = if a_mag + 10 < t0.mag_def
                            && a_mag + 10 < t1.mag_def
                            && b_mag + 10 < t1.mag_def
                            && b_mag + 10 < t2.mag_def
                        {
                            if k1 > k2 {
                                skip(Side::Right)
                            } else {
                                skip(Side::Left)
                            }
                        } else if sgn(a) != -sgn(b) {
                            if a == 0.0 {
                                skip(Side::Left)
                            } else if b == 0.0 {
                                skip(Side::Right)
                            } else if sgn(a) == sgn(t1.f_def.re) {
                                shift(Side::Left)
                            } else {
                                shift(Side::Right)
                            }
                        } else if sgn(a) == -sgn(t1.f_def.re) {
                            trap = Trap::Bracket(Region::Minimum);
                            quadratic(Region::Minimum)
                        } else if k1 > k2 {
                            skip(Side::Right)
                        } else {
                            skip(Side::Left)
                        };
                    } else {
                        trap = Trap::Bracket(Region::Minimum);
                        chosen = if sgn(a) != sgn(b) {
                            quadratic(Region::Minimum)
                        } else if a_mag > b_mag || (a_mag == b_mag && a.abs() > b.abs()) {
                            split(Side::Left)
                        } else {
                            split(Side::Right)
                        };
                    }
                }
                if self.consec_moves >= 3 && self.trapped == trap {
                    chosen = split(self.last_move.split_side()?);
                    self.consec_moves = 0;
                }
                chosen
            }
        };
        self.trapped = trap;
        Ok(suggestion)
    }

    /// CKTpzRunTrial: evaluate a proposed frequency, or merge a collision
    /// with an existing linked trial. Deflation is accumulated in list order.
    fn run_trial(&mut self, mut s: Cx, middle: Option<TrialId>) -> SpiceResult<TrialId> {
        if s.im < 0.0 {
            s.im *= -1.0;
        }
        let mut shifted = false;
        let mut def_frac;
        let mut def_mag;
        let mut prev;
        let mut hit;
        loop {
            def_mag = 0;
            def_frac = Cx::new(1.0, 0.0);
            let was_shifted = shifted;
            shifted = false;
            prev = None;
            hit = None;
            let mut p = self.arena.head;
            while let Some(id) = p {
                let trial = *self.arena.get(id)?;
                let diff = trial.s.sub(s);
                if diff.re < 0.0 || (diff.re == 0.0 && diff.im < 0.0) {
                    prev = Some(id);
                }
                let root = trial.is_root();
                let (abstol, reltol) = if root { (1e-5, 1e-6) } else { (1e-20, 1e-12) };
                if diff.im == 0.0 && diff.re.abs() / (trial.s.re.abs() + abstol / reltol) < reltol {
                    if was_shifted
                        || trial.count >= 3
                        || !self.alter(&mut s, middle, abstol, reltol)?
                    {
                        self.arena.get_mut(id)?.count = 0;
                        hit = Some(id);
                        break;
                    }
                    self.arena.get_mut(id)?.count += 1;
                    shifted = true;
                    break;
                }
                if self.trapped == Trap::Free {
                    self.arena.get_mut(id)?.count = 0;
                }
                if root {
                    let mut diff = diff;
                    let mut diff_mag = 0;
                    c_norm(&mut diff, &mut diff_mag)?;
                    if diff.im != 0.0 {
                        diff = diff.mag2();
                        diff_mag *= 2;
                    }
                    c_norm(&mut diff, &mut diff_mag)?;
                    for _ in 0..trial.multiplicity() {
                        def_frac = def_frac.mul(diff);
                        def_mag += diff_mag;
                        c_norm(&mut def_frac, &mut def_mag)?;
                    }
                }
                p = trial.next;
            }
            if !shifted {
                break;
            }
        }

        let determinant = if let Some(hit) = hit {
            let trial = self.arena.get_mut(hit)?;
            match trial.kind {
                TrialKind::Root(multiplicity) => {
                    trial.kind = TrialKind::Root(multiplicity.checked_add(1).ok_or_else(|| {
                        numerical("pole-zero search", "root multiplicity overflow")
                    })?);
                    return Ok(hit);
                }
                TrialKind::Sample | TrialKind::Minimum
                    if self.trapped == Trap::Bracket(Region::Minimum)
                        && self.curvature.usable() =>
                {
                    trial.kind = TrialKind::Minimum;
                    return Ok(hit);
                }
                TrialKind::Sample | TrialKind::Minimum => None,
            }
        } else {
            #[cfg(test)]
            self.trace.push(TraceEvent::Load(s));
            let determinant = self.pencil.determinant_at(s)?;
            #[cfg(test)]
            if let Some((value, exponent)) = determinant {
                self.trace.push(TraceEvent::Determinant(value, exponent));
            }
            determinant
        };

        let trial = match determinant {
            Some((f_raw, mag_raw)) if !f_raw.is_zero() => {
                // PZnumswaps is always 1 after SMPcReorder.
                let mut f_def = f_raw.div(def_frac);
                let mut mag_def = mag_raw - def_mag;
                c_norm(&mut f_def, &mut mag_def)?;
                Trial::new(s, f_def, mag_def, TrialKind::Sample)
            }
            _ => Trial::new(s, Cx::ZERO, 0, TrialKind::Root(std::num::NonZeroU32::MIN)),
        };
        self.arena.insert_after(prev, trial)
    }

    /// CKTpzVerify always returns success in C. Count each occurrence of
    /// the root, deflate surviving trials and unlink those too close to it.
    fn verify(&mut self, new: TrialId) -> SpiceResult<()> {
        let root = self.arena.get(new)?;
        if !root.is_root() {
            return Err(numerical(
                "pole-zero search",
                "cannot verify a non-root trial",
            ));
        }
        self.nzeros += 1;
        let s = root.s;
        if s.im != 0.0 {
            self.nzeros += 1;
        }
        let mut current = self.arena.head;
        while let Some(id) = current {
            let trial = *self.arena.get(id)?;
            current = trial.next;
            if trial.is_root() {
                continue;
            }
            let mut diff = s.sub(trial.s);
            if s.im != 0.0 {
                diff = diff.mag2();
            }
            let tdiff = diff.re;
            if diff.re != 0.0 {
                let mut diff_mag = 0;
                c_norm(&mut diff, &mut diff_mag)?;
                diff_mag *= -1;
                let trial = self.arena.get_mut(id)?;
                trial.f_def = trial.f_def.div(diff);
                c_norm(&mut trial.f_def, &mut diff_mag)?;
                trial.mag_def += diff_mag;
            }
            if trial.s.im != 0.0 || tdiff.abs() / (s.re.abs() + 200.0) < 0.005 {
                self.arena.remove(id)?;
            } else {
                self.arena.get_mut(id)?.kind = TrialKind::Sample;
            }
        }
        Ok(())
    }

    /// pzseek also sets the sign of the next extrapolation guess, including
    /// when it encounters an empty list.
    fn seek(&mut self, start: Option<TrialId>, direction: Seek) -> SpiceResult<Option<TrialId>> {
        self.guess_param = direction.guess();
        self.arena.seek(start, direction)
    }

    /// CKTpzUpdateSet. Preserve the C slot updates and strict determinant
    /// tie-break; classify the result before it can schedule another kernel.
    fn update_neighborhood(
        &mut self,
        neighborhood: Neighborhood,
        new: TrialId,
    ) -> SpiceResult<Neighborhood> {
        let Window {
            left,
            middle,
            right,
        } = neighborhood.window();
        let trial = *self.arena.get(new)?;
        let n = trial.s;
        let mut movement = Move::None;
        let mut window = Window {
            left,
            middle,
            right,
        };
        if n.im != 0.0 {
            window = Window {
                left: Some(new),
                middle: left,
                right: middle,
            };
        } else if let Some(mid) = middle {
            let center = self.arena.get(mid)?;
            if right.is_none() && n.re > center.s.re {
                window.right = Some(new);
            } else if let Some(left) = left {
                if trial.kind == TrialKind::Minimum {
                    window.middle = Some(new);
                } else if n.re < self.arena.frequency(left)?.re {
                    window = Window {
                        left: Some(new),
                        middle: Some(left),
                        right: middle,
                    };
                    movement = Move::Far(Side::Left);
                } else {
                    let better = self.trapped == Trap::Free
                        || trial.mag_def < center.mag_def
                        || (trial.mag_def == center.mag_def
                            && trial.f_def.re.abs() < center.f_def.re.abs());
                    if n.re < center.s.re {
                        if better {
                            window.right = middle;
                            window.middle = Some(new);
                            movement = Move::Mid(Side::Left);
                        } else {
                            window.left = Some(new);
                            movement = Move::Near(Side::Left);
                        }
                    } else if right
                        .map(|id| self.arena.frequency(id))
                        .transpose()?
                        .is_some_and(|r| n.re < r.re)
                    {
                        if better {
                            window.left = middle;
                            window.middle = Some(new);
                            movement = Move::Mid(Side::Right);
                        } else {
                            window.right = Some(new);
                            movement = Move::Near(Side::Right);
                        }
                    } else {
                        window = Window {
                            left: middle,
                            middle: right,
                            right: Some(new),
                        };
                        movement = Move::Far(Side::Right);
                    }
                }
            } else {
                window.left = Some(new);
            }
        } else {
            window.middle = Some(new);
        }
        if self.trapped != Trap::Free && movement == self.last_move {
            self.consec_moves += 1;
        } else {
            self.consec_moves = 0;
        }
        self.last_move = movement;
        Neighborhood::classify(window, &self.arena)
    }

    /// CKTpzStep. A navigation cannot ask for an absent bracket endpoint;
    /// only initialization may fill a missing point.
    fn step(&mut self, navigation: Navigation) -> SpiceResult<Neighborhood> {
        let window = match navigation {
            Navigation::Fill(incomplete) => match incomplete {
                Incomplete::Middle { left, right } => Window {
                    left,
                    middle: self.seek(self.arena.anchor, Seek::Anchor)?,
                    right,
                },
                Incomplete::Right { left, middle } => Window {
                    left,
                    middle: Some(middle),
                    right: self.seek(Some(middle), Seek::Right)?,
                },
                Incomplete::Left { middle, right } => Window {
                    left: self.seek(Some(middle), Seek::Left)?,
                    middle: Some(middle),
                    right: Some(right),
                },
            },
            Navigation::Skip {
                bracket: RealBracket(t),
                side,
            } => {
                let mut window = Window::from(t);
                match side {
                    Side::Left => window.left = self.seek(Some(t.left), side.into())?,
                    Side::Right => window.right = self.seek(Some(t.right), side.into())?,
                }
                window
            }
            Navigation::Shift {
                bracket: RealBracket(t),
                side,
            } => match side {
                Side::Left => Window {
                    left: self.seek(Some(t.left), side.into())?,
                    middle: Some(t.left),
                    right: Some(t.middle),
                },
                Side::Right => Window {
                    left: Some(t.middle),
                    middle: Some(t.right),
                    right: self.seek(Some(t.right), side.into())?,
                },
            },
        };
        Neighborhood::classify(window, &self.arena)
    }

    /// CKTpzReset.
    fn reset(&mut self) -> SpiceResult<Neighborhood> {
        self.trapped = Trap::Free;
        self.consec_moves = 0;
        let middle = self.seek(self.arena.anchor, Seek::Anchor)?;
        let window = if middle.is_some() {
            let left = self.seek(middle, Seek::Left)?;
            let right = self.seek(middle, Seek::Right)?;
            Window {
                left,
                middle,
                right,
            }
        } else {
            Window::default()
        };
        Neighborhood::classify(window, &self.arena)
    }

    /// alter: move a candidate that collides with a linked trial.
    fn alter(
        &self,
        candidate: &mut Cx,
        nearto: Option<TrialId>,
        abstol: Real,
        reltol: Real,
    ) -> SpiceResult<bool> {
        let Some(near) = nearto else {
            return Ok(false);
        };
        let t = self.arena.get(near)?;
        let neighbor = |id: Option<TrialId>| -> SpiceResult<Option<Real>> {
            Ok(id
                .map(|id| self.arena.frequency(id))
                .transpose()?
                .map(|s| s.re))
        };
        let (left, right) = (neighbor(t.prev)?, neighbor(t.next)?);
        let root = t.is_root();
        let p1 = if self.trapped == Trap::Bracket(Region::RightCrossing) {
            t.s.re
        } else {
            let mut p1 = t.s.re;
            if root {
                p1 -= 1e-6 * t.s.re + 1e-5;
            }
            match left {
                Some(l) => p1 += l,
                None => p1 -= 10.0 * (p1.abs() + 1.0),
            }
            p1 / 2.0
        };
        let p2 = if self.trapped == Trap::Bracket(Region::LeftCrossing) {
            t.s.re
        } else {
            let mut p2 = t.s.re;
            if root {
                p2 += 1e-6 * t.s.re + 1e-5;
            }
            match right {
                Some(r) => p2 += r,
                None => p2 += 10.0 * (p2.abs() + 1.0),
            }
            p2 / 2.0
        };
        // C's precedence: |p - x| / |x| + abstol / reltol < reltol.
        if left.is_some_and(|l| (p1 - l).abs() / l.abs() + abstol / reltol < reltol)
            || right.is_some_and(|r| (p2 - r).abs() / r.abs() + abstol / reltol < reltol)
        {
            return Ok(false);
        }
        candidate.re =
            if self.trapped != Trap::Bracket(Region::RightCrossing) && t.s.re - p1 > p2 - t.s.re {
                p1
            } else {
                p2
            };
        Ok(true)
    }
}

/// `NIpzSym2`: deflated quadratic fit through three real trials.
fn sym2(
    t0: &Trial,
    t1: &Trial,
    t2: &Trial,
    region: Region,
    curvature: &mut Curvature,
) -> SpiceResult<Real> {
    let mut dx0 = t1.s.re - t0.s.re;
    let mut dx1 = t2.s.re - t1.s.re;
    let x0 = (t0.s.re + t1.s.re) / 2.0;
    let mut d2x = (t2.s.re - t0.s.re) / 2.0;

    let (mut a, mut a_mag) = zaddeq(t1.f_def.re, t1.mag_def, -t0.f_def.re, t0.mag_def)?;
    let mut tmag = 0;
    r_norm(&mut dx0, &mut tmag)?;
    a /= dx0;
    a_mag -= tmag;
    r_norm(&mut a, &mut a_mag)?;

    let (mut b, mut b_mag) = zaddeq(t2.f_def.re, t2.mag_def, -t1.f_def.re, t1.mag_def)?;
    tmag = 0;
    r_norm(&mut dx1, &mut tmag)?;
    b /= dx1;
    b_mag -= tmag;
    r_norm(&mut b, &mut b_mag)?;

    let (mut c, mut c_mag) = zaddeq(b, b_mag, -a, a_mag)?;
    tmag = 0;
    r_norm(&mut d2x, &mut tmag)?;
    c /= d2x;
    c_mag -= tmag;
    r_norm(&mut c, &mut c_mag)?;

    if c == 0.0 || ((a == 0.0 || c_mag < a_mag - 40) && (b == 0.0 || c_mag < b_mag - 40)) {
        if a == 0.0 {
            a = b;
            a_mag = b_mag;
        }
        if a == 0.0 {
            return Ok(t1.s.re);
        }
        let s = materialize_power(-t1.f_def.re / a, t1.mag_def - a_mag);
        return Ok(s + t1.s.re);
    }

    a /= c;
    r_norm(&mut a, &mut a_mag)?;
    a_mag -= c_mag;
    let mut diff = t1.s.re - x0;
    tmag = 0;
    r_norm(&mut diff, &mut tmag)?;
    (a, a_mag) = zaddeq(a, a_mag, diff, tmag)?;

    b = 2.0 * t1.f_def.re / c;
    b_mag = t1.mag_def - c_mag;
    r_norm(&mut b, &mut b_mag)?;

    let (mut disc, mut disc_mag) = zaddeq(a * a, 2 * a_mag, -b, b_mag)?;
    let complex = disc < 0.0;
    if complex {
        disc *= -1.0;
    }
    if disc_mag % 2 == 0 {
        disc = disc.sqrt();
    } else {
        disc = (2.0 * disc).sqrt();
        disc_mag -= 1;
    }
    disc_mag /= 2;
    if complex {
        if curvature.value == 0.0 {
            curvature.value = disc;
            curvature.exponent = disc_mag;
        }
        disc = 0.0;
        disc_mag = 0;
    }
    (c, c_mag) = if a * disc >= 0.0 {
        zaddeq(a, a_mag, disc, disc_mag)?
    } else {
        zaddeq(a, a_mag, -disc, disc_mag)?
    };
    if c == 0.0 {
        b = 0.0;
        b_mag = 0;
    } else {
        b /= c;
        b_mag -= c_mag;
    }
    (b, b_mag) = zaddeq(t1.s.re, 0, -b, b_mag)?;
    (c, c_mag) = zaddeq(t1.s.re, 0, -c, c_mag)?;
    b = materialize_power(b, b_mag);
    c = materialize_power(c, c_mag);
    let outside = |x: Real| x < t0.s.re || x > t2.s.re;
    Ok(if outside(b) {
        if outside(c) {
            match region {
                Region::LeftCrossing => (t0.s.re + t1.s.re) / 2.0,
                Region::RightCrossing => (t1.s.re + t2.s.re) / 2.0,
                Region::Minimum => {
                    if (t1.s.re - c).abs() < (t1.s.re - b).abs() {
                        (t1.s.re + c) / 2.0
                    } else {
                        (t1.s.re + b) / 2.0
                    }
                }
            }
        } else {
            c
        }
    } else if outside(c) || a > 0.0 {
        b
    } else {
        c
    })
}

/// `NIpzMuller`.
fn muller(t: [&Trial; 3]) -> SpiceResult<Cx> {
    let mut min = -999_999;
    let mut count = 0;
    let mut total = 0;
    for trial in t {
        if !trial.f_def.is_zero() {
            if min < trial.mag_def - 50 {
                min = trial.mag_def - 50;
            }
            total += trial.mag_def;
            count += 1;
        }
    }
    let mut magx = if count != 0 { total / count } else { total };
    if magx < min {
        magx = min;
    }
    let mut scale = [1.0; 3];
    for (i, trial) in t.iter().enumerate() {
        let mag = trial.mag_def - magx;
        scale[i] = if mag < -90 {
            0.0
        } else {
            materialize_power(1.0, mag)
        };
    }
    let f: [Cx; 3] = std::array::from_fn(|i| t[i].f_def.scale(scale[i]));
    let h0 = t[0].s.sub(t[1].s);
    let h1 = t[1].s.sub(t[2].s);
    let lambda = h0.div(h1);
    let mut delta = lambda;
    delta.re += 1.0;

    let mut a = f[2].mul(lambda);
    let mut c = f[1].mul(delta);
    a = a.sub(c);
    a = a.add(f[0]);
    a = a.mul(lambda);

    let mut b = lambda.mul(lambda);
    b = b.mul(f[2]);
    c = delta.mul(delta);
    c = c.mul(f[1]);
    b = b.sub(c);
    c = lambda.add(delta);
    c = c.mul(f[0]);
    b = b.add(c);

    c = delta.mul(f[0]);

    normalize_muller(&mut a, &mut b, &mut c)?;
    let mut d = b.mul(b);
    let e = a.scale(4.0).mul(c);
    d = d.sub(e);
    d = d.sqrt();
    let q = b.re * d.re + b.im * d.im;
    b = if q > 0.0 { b.add(d) } else { b.sub(d) };
    c = c.div(b.scale(-0.5));
    Ok(h0.mul(c).add(t[0].s))
}

#[cfg(test)]
pub(crate) fn check_c_trace(load: &PencilLoad, name: &str, trace: &str) {
    let mut search = Search::new(load).unwrap();
    let result = search.find_zeros();
    eprintln!(
        "TRACECASE {name} iterations={} result={result:?} expected_events={}",
        search.niter,
        trace.lines().count()
    );
    for event in search.trace {
        match event {
            TraceEvent::Proposal(s) => eprintln!("TRIAL {:.17e} {:.17e}", s.re, s.im),
            TraceEvent::Load(s) => eprintln!("LOAD {:.17e} {:.17e}", s.re, s.im),
            TraceEvent::Determinant(s, exp) => eprintln!("DET {:.17e} {:.17e} {exp}", s.re, s.im),
        }
    }
    eprintln!("TRACEEND {name}");
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
