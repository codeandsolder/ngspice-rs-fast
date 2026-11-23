//! Pole-zero scheduling and explicit trial-neighborhood transitions.
use super::{Cx, LoadedPencil, PencilLoad, c_norm, materialize_power, numerical, sgn, zaddeq};
use crate::primitives::{Complex, Real, SpiceResult};

mod arena;
mod kernels;

use arena::{Seek, Trial, TrialArena, TrialId, TrialKind};
use kernels::{Curvature, muller, sym2};

const NITER_LIM: usize = 200;

pub(super) fn find_zeros(load: &PencilLoad) -> SpiceResult<Vec<Complex>> {
    Search::new(load)?.find_zeros()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Side {
    Left,
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

/// The three numerical regions C denotes by `CKTpzTrapped` = 1, 2, 3.
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

impl Triple {
    fn trials(self, arena: &TrialArena) -> SpiceResult<[&Trial; 3]> {
        Ok([
            arena.get(self.left)?,
            arena.get(self.middle)?,
            arena.get(self.right)?,
        ])
    }
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
        let center = middle
            .map(|id| arena.get(id).map(|t| (id, t)))
            .transpose()?;
        let first = left.map(|id| arena.get(id).map(|t| (id, t))).transpose()?;
        if let Some(id) = right {
            arena.get(id)?;
        }
        if let Some((middle, trial)) = center
            && trial.kind == TrialKind::Minimum
        {
            return Ok(Self::Minimum {
                middle,
                left,
                right,
            });
        }
        if let Some((latest, trial)) = first
            && trial.s().im != 0.0
        {
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

#[cfg(test)]
#[derive(Debug, Clone, Copy, PartialEq)]
enum TraceEvent {
    Proposal(Cx),
    Load(Cx),
    Determinant(Cx, i64),
}

/// The actual polynomial deflation factor for a real root or a conjugate
/// pair. A squared modulus only agrees with this at real trial frequencies.
fn root_factor(root: Cx, trial: Cx) -> SpiceResult<(Cx, i64)> {
    let mut factor = root.sub(trial);
    let mut exponent = 0;
    c_norm(&mut factor, &mut exponent)?;
    if root.im != 0.0 {
        let mut conjugate = Cx::new(root.re, -root.im).sub(trial);
        let mut conjugate_exponent = 0;
        c_norm(&mut conjugate, &mut conjugate_exponent)?;
        factor = factor.mul(conjugate);
        exponent += conjugate_exponent;
        c_norm(&mut factor, &mut exponent)?;
    }
    Ok((factor, exponent))
}

/// One `CKTpzFindZeros` run with its independent scheduler and trial arena.
struct Search<'a> {
    pencil: LoadedPencil<'a>,
    arena: TrialArena,
    nzeros: usize,
    max_zeros: usize,
    regular: bool,
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

impl<'a> Search<'a> {
    fn new(load: &'a PencilLoad) -> SpiceResult<Self> {
        Ok(Self {
            pencil: LoadedPencil::new(load)?,
            arena: TrialArena::default(),
            nzeros: 0,
            max_zeros: load.degree_bound(),
            regular: false,
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
    /// `CKTpzFindZeros`: navigate existing trials before proposing a new one.
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
        if !self.regular {
            // A degree-one pencil can hit its exact root at the first
            // trial. Prove regularity at a distinct point before calling
            // it identically singular. A polynomial of degree <= d
            // cannot vanish at d+1 distinct points unless it is zero.
            let mut probe = 0.;
            for _ in 0..=self.max_zeros {
                if self
                    .pencil
                    .determinant_at(Cx::new(probe, 0.))?
                    .is_some_and(|(v, _)| !v.is_zero())
                {
                    self.regular = true;
                    break;
                }
                probe += 1.;
            }
        }
        if !self.regular {
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
        let mut current = self.arena.head();
        while let Some(id) = current {
            let trial = self.arena.get(id)?;
            for _ in 0..trial.multiplicity() {
                roots.push(Complex::new(trial.s().re, trial.s().im));
                if trial.s().im != 0.0 {
                    roots.push(Complex::new(trial.s().re, -trial.s().im));
                }
            }
            current = trial.next();
        }
        Ok(roots)
    }

    /// `PZeval`. Only candidate frequencies, not detached arena entries, exist
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
                let [t0, t1, t2] = triple.trials(&self.arena)?;
                let mut trial = Cx::new(sym2(t0, t1, t2, region, &mut self.curvature)?, 0.0);
                let (s0, s1, s2) = (t0.s(), t1.s(), t2.s());
                match region {
                    Region::LeftCrossing => {
                        if trial.re < s0.re || trial.re > s1.re {
                            trial.re = s0.re.midpoint(s1.re);
                        }
                    }
                    Region::RightCrossing => {
                        if trial.re < s1.re || trial.re > s2.re {
                            trial.re = s1.re.midpoint(s2.re);
                        }
                    }
                    Region::Minimum => {
                        if trial.re <= s0.re
                            || (trial.re == s1.re && trial.im == s1.im)
                            || trial.re >= s2.re
                        {
                            trial.re = s0.re.midpoint(s2.re);
                            if trial.re == s1.re {
                                trial.re = if matches!(
                                    self.last_move,
                                    Move::Mid(Side::Left) | Move::Near(Side::Right)
                                ) {
                                    s0.re.midpoint(s1.re)
                                } else {
                                    s1.re.midpoint(s2.re)
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
            Evaluation::Muller(triple) => muller(triple.trials(&self.arena)?)?,
            Evaluation::Split { edge, middle } => Cx::new(
                (self.arena.frequency(middle)?.re / 3.0)
                    .mul_add(2.0, self.arena.frequency(edge)?.re / 3.0),
                0.0,
            ),
        };
        if !s.re.is_finite() || !s.im.is_finite() {
            return Err(numerical("pole-zero search", "non-finite trial frequency"));
        }
        Ok(s)
    }

    /// `CKTpzStrat`. Each evaluation or navigation carries all of its inputs.
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
                let [t0, t1, t2] = bracket.0.trials(&self.arena)?;
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
                        let k1 = t1.s().re - t0.s().re;
                        let k2 = t2.s().re - t1.s().re;
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

    /// Evaluate the determinant after deflating every accepted root. The
    /// probe path uses the same factored pencil as ordinary trials.
    fn deflated_at(&mut self, s: Cx) -> SpiceResult<(Cx, i64)> {
        let determinant = self.pencil.determinant_at(s)?;
        #[cfg(test)]
        if let Some((value, exponent)) = determinant {
            self.trace.push(TraceEvent::Determinant(value, exponent));
        }
        let Some((mut value, mut exponent)) = determinant else {
            return Ok((Cx::ZERO, 0));
        };
        self.regular |= !value.is_zero();
        let mut current = self.arena.head();
        while let Some(id) = current {
            let trial = self.arena.get(id)?;
            if trial.is_root() {
                let (factor, shift) = root_factor(trial.s(), s)?;
                if factor.is_zero() {
                    return Err(numerical(
                        "pole-zero search",
                        "deflation at an existing root",
                    ));
                }
                for _ in 0..trial.multiplicity() {
                    value = value.div(factor);
                    exponent -= shift;
                    c_norm(&mut value, &mut exponent)?;
                }
            }
            current = trial.next();
        }
        Ok((value, exponent))
    }

    /// A coincident trial is not in itself evidence of a root. Check the
    /// local deflated determinant against displaced values, including
    /// the linear and quadratic terms of its local Taylor polynomial.
    fn coincident_root(&mut self, s: Cx) -> SpiceResult<bool> {
        let center = self.deflated_at(s)?;
        if center.0.is_zero() {
            return Ok(true);
        }
        let scale = s.re.hypot(s.im).max(1e-12);
        let step = scale * 2_f64.powi(-20);
        let left = self.deflated_at(Cx::new(s.re - step, s.im))?;
        let right = self.deflated_at(Cx::new(s.re + step, s.im))?;
        let exponent = [center, left, right]
            .into_iter()
            .filter(|(value, _)| !value.is_zero())
            .map(|(_, exponent)| exponent)
            .max()
            .unwrap_or_default();
        let aligned =
            |(value, shift): (Cx, i64)| value.scale(materialize_power(1., shift - exponent));
        let (center, left, right) = (aligned(center), aligned(left), aligned(right));
        let norm = |value: Cx| value.re.hypot(value.im);
        let slope = right.sub(left).scale(0.5);
        let curvature = right.add(left).scale(0.5).sub(center);
        let relative = 4. * 1e-12_f64.mul_add(scale, 1e-20) / step;
        // For a simple root this bounds the Newton correction. At an
        // unresolved repeated root the leading local term is quadratic.
        Ok(norm(center) <= relative * norm(slope)
            || norm(center) <= relative * relative * norm(curvature))
    }

    /// An already deflated simple root must not acquire another occurrence
    /// merely because the search revisits its neighborhood. A remaining
    /// multiplicity makes the deflated function tend to zero on both sides.
    fn remaining_multiplicity(&mut self, s: Cx) -> SpiceResult<bool> {
        let step = s.re.hypot(s.im).max(1e-12) * 2_f64.powi(-23);
        for direction in [-1.0_f64, 1.] {
            let far = self.deflated_at(Cx::new(direction.mul_add(step, s.re), s.im))?;
            let near = self.deflated_at(Cx::new(direction.mul_add(step * 0.5, s.re), s.im))?;
            if far.0.is_zero() {
                return Ok(false);
            }
            let ratio = materialize_power(
                near.0.re.hypot(near.0.im) / far.0.re.hypot(far.0.im),
                near.1 - far.1,
            );
            if ratio > 0.75 {
                return Ok(false);
            }
        }
        Ok(true)
    }

    /// The search's convergence radius is not a license to count a tiny
    /// imaginary perturbation as a second pole. Resolve an accepted candidate
    /// against the real determinant: a simple real root has a sign bracket and
    /// a localized Newton correction, while a genuine conjugate pair has no
    /// such nearby real crossing. An exact zero on the real axis also permits
    /// repeated roots (where the first derivative necessarily vanishes).
    ///
    /// All determinant probes use the already-deflated pencil. This is called
    /// only for candidates that have passed the ordinary residual test.
    fn real_axis_root(&mut self, s: Cx) -> SpiceResult<Option<Cx>> {
        let scale = s.re.abs().max(s.im);
        let ulp = s.re.abs().next_up() - s.re.abs();
        let step = (scale * 2_f64.powi(-20)).max(s.im * 32.).max(ulp * 32.);
        if !step.is_finite() || step == 0. {
            return Ok(None);
        }
        let (left_x, right_x) = (s.re - step, s.re + step);
        if !left_x.is_finite() || !right_x.is_finite() || left_x == s.re || right_x == s.re {
            return Ok(None);
        }
        let center = self.deflated_at(Cx::new(s.re, 0.))?;
        if center.0.is_zero() {
            // A true complex zero at the same real part must not be
            // discarded merely because a real zero also exists. When
            // the imaginary displacement is smaller than one real-axis
            // ULP, the two evaluations are not numerically resolvable:
            // the exact real zero is the supported classification.
            if s.im == 0. || s.im <= ulp || !self.deflated_at(s)?.0.is_zero() {
                return Ok(Some(Cx::new(s.re, 0.)));
            }
            return Ok(None);
        }
        let left = self.deflated_at(Cx::new(left_x, 0.))?;
        let right = self.deflated_at(Cx::new(right_x, 0.))?;
        let (difference, difference_exponent) = zaddeq(right.0.re, right.1, -left.0.re, left.1)?;
        let repeated = sgn(left.0.re) == sgn(right.0.re);
        let correction = if repeated {
            // A repeated real zero has no sign change. Fit a *local*
            // quadratic using the symmetric divided differences. Its
            // discriminant vanishes up to uncertainty in the representable
            // real coordinate. In contrast, conjugate pairs and two distinct
            // neighboring real roots have respectively positive/negative
            // nonzero vertex residuals.
            let (ends, ends_exponent) = zaddeq(left.0.re, left.1, right.0.re, right.1)?;
            let (curvature, curvature_exponent) =
                zaddeq(ends, ends_exponent, -center.0.re, center.1 + 1)?;
            if curvature == 0. {
                return Ok(None);
            }
            let slope = materialize_power(
                difference / curvature,
                difference_exponent - curvature_exponent,
            );
            let offset = materialize_power(center.0.re / curvature, center.1 - curvature_exponent);
            let normalized_vertex = offset - slope * slope / 8.;
            if !slope.is_finite()
                || !normalized_vertex.is_finite()
                || normalized_vertex.abs() > 16. * (ulp / step).powi(2)
            {
                return Ok(None);
            }
            -0.5 * step * slope
        } else {
            if difference == 0. {
                return Ok(None);
            }
            materialize_power(
                (-center.0.re / difference) * step,
                center.1 - difference_exponent + 1,
            )
        };
        let accepted_radius = 4e-12_f64
            .mul_add(scale.max(1e-12), 1e-20)
            .max(8. * s.im)
            .max(64. * ulp);
        if !correction.is_finite()
            || correction.abs() > accepted_radius
            || correction.abs() > step * 0.5
        {
            return Ok(None);
        }
        let root_x = s.re + correction;
        if !(left_x..=right_x).contains(&root_x) {
            return Ok(None);
        }
        let complex = if s.im != 0. {
            let value = self.deflated_at(s)?;
            if value.0.is_zero() && s.im > ulp {
                return Ok(None);
            }
            if !repeated && !value.0.is_zero() {
                // At a simple real root, a complex displacement produces
                // approximately i * Im(s) * D'(Re(s)). An actual complex
                // root has a much smaller residual than this.
                let complex_to_slope = materialize_power(
                    value.0.re.hypot(value.0.im) / difference.abs(),
                    value.1 - difference_exponent,
                );
                if complex_to_slope < s.im / (16. * step) {
                    return Ok(None);
                }
            }
            Some(value)
        } else {
            None
        };
        let refined = self.deflated_at(Cx::new(root_x, 0.))?;
        if repeated && let Some(complex) = complex {
            let residual_ratio = if refined.0.is_zero() {
                0.
            } else {
                materialize_power(
                    refined.0.re.hypot(refined.0.im) / complex.0.re.hypot(complex.0.im),
                    refined.1 - complex.1,
                )
            };
            if residual_ratio > 0.125 {
                return Ok(None);
            }
        }
        if root_x != s.re && !refined.0.is_zero() {
            let residual_ratio = materialize_power(
                refined.0.re.hypot(refined.0.im) / center.0.re.hypot(center.0.im),
                refined.1 - center.1,
            );
            if residual_ratio > 0.5 {
                // Around an exact real crossing the LU determinant can
                // already be at its rounding floor. A correction of only
                // a few representable coordinates need not monotonically
                // reduce that cancellation-limited residual.
                if correction.abs() <= 16. * ulp {
                    return Ok(Some(Cx::new(s.re, 0.)));
                }
                return Ok(None);
            }
        }
        Ok(Some(Cx::new(root_x, 0.)))
    }

    /// A root is linked in corrected frequency order, not in the original
    /// Muller proposal order. Classification is shared by both exact-zero
    /// trials and roots recognized through residual convergence.
    fn insert_root(&mut self, s: Cx) -> SpiceResult<TrialId> {
        let corrected = self.real_axis_root(s)?.unwrap_or(s);
        let mut previous = None;
        let mut current = self.arena.head();
        while let Some(id) = current {
            let trial = self.arena.get(id)?;
            let diff = trial.s().sub(corrected);
            if diff.re < 0. || (diff.re == 0. && diff.im < 0.) {
                previous = Some(id);
            }
            current = trial.next();
        }
        self.arena.insert_after(
            previous,
            Trial::new(
                corrected,
                Cx::ZERO,
                0,
                TrialKind::Root(std::num::NonZeroU32::MIN),
            ),
        )
    }

    /// Evaluate or merge a proposed trial. Preserve distinct roots and
    /// require a numerical residual check before declaring convergence.
    fn run_trial(&mut self, mut s: Cx, middle: Option<TrialId>) -> SpiceResult<TrialId> {
        const ABSTOL: Real = 1e-20;
        const RELTOL: Real = 1e-12;
        s.im = s.im.abs();
        let mut shifted = false;
        let mut attempts = 0;
        loop {
            let mut prev = None;
            let mut hit = None;
            let mut current = self.arena.head();
            while let Some(id) = current {
                let trial = *self.arena.get(id)?;
                let diff = trial.s().sub(s);
                if diff.re < 0. || (diff.re == 0. && diff.im < 0.) {
                    prev = Some(id);
                }
                if diff.re.hypot(diff.im) < RELTOL.mul_add(trial.s().re.hypot(trial.s().im), ABSTOL)
                {
                    hit = Some(id);
                    break;
                }
                if self.trapped == Trap::Free {
                    self.arena.get_mut(id)?.count = 0;
                }
                current = trial.next();
            }

            if let Some(id) = hit {
                let trial = *self.arena.get(id)?;
                if !shifted
                    && trial.count < 3
                    && s.im == 0.
                    && self.alter(&mut s, middle, ABSTOL, RELTOL)?
                {
                    self.arena.get_mut(id)?.count += 1;
                    shifted = true;
                    continue;
                }
                self.arena.get_mut(id)?.count = 0;
                match trial.kind {
                    TrialKind::Root(multiplicity) => {
                        if self.remaining_multiplicity(trial.s())? {
                            self.arena.get_mut(id)?.kind =
                                TrialKind::Root(multiplicity.checked_add(1).ok_or_else(|| {
                                    numerical("pole-zero search", "root multiplicity overflow")
                                })?);
                            return Ok(id);
                        }
                    }
                    TrialKind::Sample | TrialKind::Minimum => {
                        if self.coincident_root(s)? {
                            return self.insert_root(s);
                        }
                        if s.im == 0. && self.trapped == Trap::Bracket(Region::Minimum) {
                            self.arena.get_mut(id)?.kind = TrialKind::Minimum;
                            return Ok(id);
                        }
                    }
                }
                // A failed collision is just a stagnant sample. Move by an
                // increasing local step and evaluate a distinct frequency.
                attempts += 1;
                if attempts >= 32 {
                    return Err(numerical(
                        "pole-zero search",
                        "cannot separate coincident non-root trials",
                    ));
                }
                let step = RELTOL.mul_add(s.re.hypot(s.im), ABSTOL) * 2_f64.powi(attempts + 2);
                s.re += step;
                shifted = true;
                continue;
            }

            #[cfg(test)]
            self.trace.push(TraceEvent::Load(s));
            let (value, exponent) = self.deflated_at(s)?;
            if value.is_zero() {
                return self.insert_root(s);
            }
            return self
                .arena
                .insert_after(prev, Trial::new(s, value, exponent, TrialKind::Sample));
        }
    }

    /// `CKTpzVerify` always returns success in C. Count each occurrence of
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
        let s = root.s();
        if s.im != 0.0 {
            self.nzeros += 1;
        }
        let mut current = self.arena.head();
        while let Some(id) = current {
            let trial = *self.arena.get(id)?;
            current = trial.next();
            if trial.is_root() {
                continue;
            }
            let (diff, diff_mag) = root_factor(s, trial.s())?;
            let tdiff = materialize_power(diff.re, diff_mag);
            if trial.s().im != 0.0 || tdiff.abs() / (s.re.abs() + 200.0) < 0.005 {
                self.arena.remove(id)?;
            } else {
                let trial = self.arena.get_mut(id)?;
                let mut exponent = -diff_mag;
                trial.f_def = trial.f_def.div(diff);
                c_norm(&mut trial.f_def, &mut exponent)?;
                trial.mag_def += exponent;
                trial.kind = TrialKind::Sample;
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

    /// `CKTpzUpdateSet`. Preserve the C slot updates and strict determinant
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
        if trial.kind != TrialKind::Sample {
            return Err(numerical(
                "pole-zero search",
                "only sample trials update a neighborhood",
            ));
        }
        let n = trial.s();
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
            if right.is_none() && n.re > center.s().re {
                window.right = Some(new);
            } else if let Some(left) = left {
                if n.re < self.arena.frequency(left)?.re {
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
                    if n.re < center.s().re {
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

    /// `CKTpzStep`. A navigation cannot ask for an absent bracket endpoint;
    /// only initialization may fill a missing point.
    fn step(&mut self, navigation: Navigation) -> SpiceResult<Neighborhood> {
        let window = match navigation {
            Navigation::Fill(incomplete) => match incomplete {
                Incomplete::Middle { left, right } => Window {
                    left,
                    middle: self.seek(self.arena.anchor(), Seek::Anchor)?,
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

    /// `CKTpzReset`.
    fn reset(&mut self) -> SpiceResult<Neighborhood> {
        self.trapped = Trap::Free;
        self.consec_moves = 0;
        let middle = self.seek(self.arena.anchor(), Seek::Anchor)?;
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
        let (left, right) = (neighbor(t.prev())?, neighbor(t.next())?);
        let root = t.is_root();
        let p1 = if self.trapped == Trap::Bracket(Region::RightCrossing) {
            t.s().re
        } else {
            let mut p1 = t.s().re;
            if root {
                p1 -= reltol.mul_add(t.s().re.abs(), abstol);
            }
            match left {
                Some(l) => p1 += l,
                None => p1 = (-10.0_f64).mul_add(p1.abs() + 1.0, p1),
            }
            p1 / 2.0
        };
        let p2 = if self.trapped == Trap::Bracket(Region::LeftCrossing) {
            t.s().re
        } else {
            let mut p2 = t.s().re;
            if root {
                p2 += reltol.mul_add(t.s().re.abs(), abstol);
            }
            match right {
                Some(r) => p2 += r,
                None => p2 = 10.0_f64.mul_add(p2.abs() + 1.0, p2),
            }
            p2 / 2.0
        };
        if left.is_some_and(|l| (p1 - l).abs() < reltol.mul_add(l.abs(), abstol))
            || right.is_some_and(|r| (p2 - r).abs() < reltol.mul_add(r.abs(), abstol))
        {
            return Ok(false);
        }
        candidate.re = if self.trapped != Trap::Bracket(Region::RightCrossing)
            && t.s().re - p1 > p2 - t.s().re
        {
            p1
        } else {
            p2
        };
        Ok(true)
    }
}

#[cfg(test)]
pub(crate) fn report_c_trace(load: &PencilLoad, name: &str, trace: &str) {
    let mut search = Search::new(load).unwrap();
    let result = search.find_zeros();
    let first_difference = search.trace.iter().zip(trace.lines()).enumerate().find_map(
        |(index, (actual, expected))| {
            let mut fields = expected.split_whitespace();
            let kind = fields.next()?;
            let value = Cx::new(fields.next()?.parse().ok()?, fields.next()?.parse().ok()?);
            let same = match (actual, kind) {
                (TraceEvent::Proposal(got), "TRIAL") | (TraceEvent::Load(got), "LOAD") => {
                    got.re.to_bits() == value.re.to_bits() && got.im.to_bits() == value.im.to_bits()
                }
                (TraceEvent::Determinant(got, exponent), "DET") => {
                    got.re.to_bits() == value.re.to_bits()
                        && got.im.to_bits() == value.im.to_bits()
                        && Some(*exponent) == fields.next().and_then(|word| word.parse().ok())
                }
                _ => false,
            };
            (!same).then(|| format!("event {index}: Rust {actual:?}; C {expected}"))
        },
    );
    eprintln!(
        "{name}: {} Rust events / {} C events, first difference {first_difference:?}, roots {result:?}",
        search.trace.len(),
        trace.lines().count()
    );
    if search.niter >= NITER_LIM {
        for event in search
            .trace
            .iter()
            .take(30)
            .chain(search.trace.iter().rev().take(18).rev())
        {
            eprintln!("  {event:?}");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::maths::pencil::PencilTerm;

    #[test]
    fn real_axis_certificate_distinguishes_simple_repeated_and_damped_roots() {
        let location = 1e10;
        let noise = 2e-5;
        for gain in [1e-12, 1., 1e12] {
            let simple = PencilLoad {
                order: 1,
                elements: vec![(0, 0)],
                terms: vec![PencilTerm {
                    row: 0,
                    col: 0,
                    conductance: Some(location * gain),
                    capacitance: Some(gain),
                }],
                ..PencilLoad::default()
            };
            let mut search = Search::new(&simple).unwrap();
            let real = search
                .real_axis_root(Cx::new(-location - 0.02, noise))
                .unwrap()
                .unwrap();
            assert_eq!(real.im, 0.);
            assert!(
                (real.re + location).abs() <= 2e-6,
                "gain {gain:e}: refined {real:?}"
            );
            assert!(
                search
                    .real_axis_root(Cx::new(-location - 0.2, noise))
                    .unwrap()
                    .is_none(),
                "a distant real crossing is not the complex trial's root"
            );

            let pair = PencilLoad {
                order: 2,
                elements: vec![(0, 0), (0, 1), (1, 0), (1, 1)],
                terms: vec![
                    PencilTerm {
                        row: 0,
                        col: 0,
                        conductance: Some(location * gain),
                        capacitance: Some(gain),
                    },
                    PencilTerm {
                        row: 0,
                        col: 1,
                        conductance: Some(-noise * gain),
                        capacitance: None,
                    },
                    PencilTerm {
                        row: 1,
                        col: 0,
                        conductance: Some(noise * gain),
                        capacitance: None,
                    },
                    PencilTerm {
                        row: 1,
                        col: 1,
                        conductance: Some(location * gain),
                        capacitance: Some(gain),
                    },
                ],
                ..PencilLoad::default()
            };
            assert!(
                Search::new(&pair)
                    .unwrap()
                    .real_axis_root(Cx::new(-location, noise))
                    .unwrap()
                    .is_none(),
                "gain {gain:e}: a true near-real conjugate pair remains complex"
            );
            let repeated = PencilLoad {
                order: 2,
                elements: vec![(0, 0), (1, 1)],
                terms: vec![
                    simple.terms[0],
                    PencilTerm {
                        row: 1,
                        col: 1,
                        ..simple.terms[0]
                    },
                ],
                ..PencilLoad::default()
            };
            let split = PencilLoad {
                order: 2,
                elements: repeated.elements.clone(),
                terms: vec![
                    PencilTerm {
                        conductance: Some((location - noise) * gain),
                        ..repeated.terms[0]
                    },
                    PencilTerm {
                        conductance: Some((location + noise) * gain),
                        ..repeated.terms[1]
                    },
                ],
                ..PencilLoad::default()
            };
            assert!(
                Search::new(&split)
                    .unwrap()
                    .real_axis_root(Cx::new(-location, noise))
                    .unwrap()
                    .is_none(),
                "gain {gain:e}: two distinct nearby real roots must not be merged"
            );
            assert_eq!(
                Search::new(&repeated)
                    .unwrap()
                    .real_axis_root(Cx::new(-location, noise))
                    .unwrap(),
                Some(Cx::new(-location, 0.)),
                "gain {gain:e}: a repeated real root remains real"
            );
        }
    }

    #[test]
    fn deflation_is_a_polynomial_at_complex_frequencies() {
        let trial = Cx::new(2., 3.);
        let (factor, exponent) = root_factor(Cx::new(-1., 0.), trial).unwrap();
        assert_eq!(
            factor.scale(materialize_power(1., exponent)),
            Cx::new(-3., -3.)
        );
        let (factor, exponent) = root_factor(Cx::new(-1., 2.), trial).unwrap();
        // (-3-i) * (-3-5i) = 4+18i, not the squared modulus 10.
        assert_eq!(
            factor.scale(materialize_power(1., exponent)),
            Cx::new(4., 18.)
        );
    }

    fn load() -> PencilLoad {
        PencilLoad {
            order: 1,
            elements: vec![(0, 0)],
            terms: vec![PencilTerm {
                row: 0,
                col: 0,
                conductance: Some(1.0),
                capacitance: Some(1.0),
            }],
            ..PencilLoad::default()
        }
    }

    fn sample(re: Real, value: Real) -> Trial {
        Trial::new(Cx::new(re, 0.0), Cx::new(value, 0.0), 0, TrialKind::Sample)
    }

    fn bracket(arena: &mut TrialArena) -> RealBracket {
        let middle = arena.insert_after(None, sample(0.0, 0.5)).unwrap();
        let left = arena.insert_after(None, sample(-2.0, -0.75)).unwrap();
        let right = arena.insert_after(Some(middle), sample(2.0, 1.0)).unwrap();
        RealBracket(Triple {
            left,
            middle,
            right,
        })
    }

    #[test]
    fn shifts_preserve_slot_order_and_missing_endpoint_guess_priority() {
        let load = load();
        let mut search = Search::new(&load).unwrap();
        let RealBracket(t) = bracket(&mut search.arena);
        let neighborhood = search.reset().unwrap();
        assert_eq!(neighborhood, Neighborhood::Real(RealBracket(t)));
        assert_eq!(search.guess_param, 1.0);

        let moved = search
            .step(Navigation::Shift {
                bracket: RealBracket(t),
                side: Side::Left,
            })
            .unwrap();
        let missing_left = Incomplete::Left {
            middle: t.left,
            right: t.middle,
        };
        assert_eq!(moved, Neighborhood::Incomplete(missing_left));
        assert_eq!(search.guess_param, -1.0);
        assert_eq!(
            search.strategy(moved).unwrap(),
            Strategy::Navigate(Navigation::Fill(missing_left))
        );
        assert_eq!(search.step(Navigation::Fill(missing_left)).unwrap(), moved);

        let moved = search
            .step(Navigation::Skip {
                bracket: RealBracket(t),
                side: Side::Right,
            })
            .unwrap();
        assert_eq!(
            moved,
            Neighborhood::Incomplete(Incomplete::Right {
                left: Some(t.left),
                middle: t.middle
            })
        );
        assert_eq!(search.guess_param, 1.0);
    }

    #[test]
    fn complete_bracket_selects_sign_crossing_and_checked_stagnation_split() {
        let load = load();
        let mut search = Search::new(&load).unwrap();
        let bracket = bracket(&mut search.arena);
        let t = bracket.0;
        let n = Neighborhood::Real(bracket);
        assert_eq!(
            search.strategy(n).unwrap(),
            Strategy::Evaluate(Evaluation::Quadratic {
                bracket,
                region: Region::LeftCrossing
            })
        );
        assert_eq!(search.trapped, Trap::Bracket(Region::LeftCrossing));
        search.consec_moves = 3;
        search.last_move = Move::Near(Side::Right);
        assert_eq!(
            search.strategy(n).unwrap(),
            Strategy::Evaluate(Evaluation::Split {
                edge: t.left,
                middle: t.middle
            })
        );
        assert_eq!(search.consec_moves, 0);
        search.consec_moves = 3;
        search.last_move = Move::Far(Side::Right);
        assert!(search.strategy(n).is_err());

        search.arena.get_mut(t.left).unwrap().f_def.re = 1.0;
        search.arena.get_mut(t.right).unwrap().f_def.re = -1.0;
        search.consec_moves = 0;
        assert_eq!(
            search.strategy(n).unwrap(),
            Strategy::Evaluate(Evaluation::Quadratic {
                bracket,
                region: Region::RightCrossing
            })
        );
    }

    #[test]
    fn complex_history_and_minimum_precedence_are_explicit() {
        let load = load();
        let mut search = Search::new(&load).unwrap();
        let RealBracket(t) = bracket(&mut search.arena);
        let latest = search
            .arena
            .insert_after(
                Some(t.middle),
                Trial::new(Cx::new(0.0, 1.0), Cx::new(1.0, 0.0), 0, TrialKind::Sample),
            )
            .unwrap();
        let gap = ComplexSeed::Gap {
            latest,
            oldest: t.right,
        };
        let window = Window {
            left: Some(latest),
            middle: None,
            right: Some(t.right),
        };
        let n = Neighborhood::classify(window, &search.arena).unwrap();
        assert_eq!(n, Neighborhood::ComplexSeed(gap));
        assert_eq!(
            search.strategy(n).unwrap(),
            Strategy::Evaluate(Evaluation::ComplexGuess(gap))
        );
        assert_eq!(
            search.eval(Evaluation::ComplexGuess(gap)).unwrap(),
            Cx::new(0.0, 1.0e12)
        );

        let first = ComplexSeed::First {
            latest,
            previous: Some(t.middle),
        };
        let n = Neighborhood::classify(first.into(), &search.arena).unwrap();
        assert_eq!(n, Neighborhood::ComplexSeed(first));
        assert_eq!(
            search.eval(Evaluation::ComplexGuess(first)).unwrap(),
            Cx::new(0.0, 1.0e8)
        );

        let triple = Triple {
            left: latest,
            middle: t.middle,
            right: t.right,
        };
        let n = Neighborhood::classify(triple.into(), &search.arena).unwrap();
        assert_eq!(n, Neighborhood::ComplexTriple(triple));
        assert_eq!(
            search.strategy(n).unwrap(),
            Strategy::Evaluate(Evaluation::Muller(triple))
        );
        search.arena.get_mut(t.middle).unwrap().kind = TrialKind::Minimum;
        let n = Neighborhood::classify(triple.into(), &search.arena).unwrap();
        assert_eq!(
            search.strategy(n).unwrap(),
            Strategy::Evaluate(Evaluation::ComplexInit(t.middle))
        );
        assert_eq!(
            search.eval(Evaluation::ComplexInit(t.middle)).unwrap(),
            Cx::new(0.0, 10000.0)
        );
    }

    #[test]
    fn trapped_ties_keep_middle_and_removed_ids_are_errors() {
        let load = load();
        let mut search = Search::new(&load).unwrap();
        let RealBracket(t) = bracket(&mut search.arena);
        let n = Neighborhood::Real(RealBracket(t));
        search.trapped = Trap::Bracket(Region::LeftCrossing);
        let candidate = search
            .arena
            .insert_after(Some(t.left), sample(-0.5, -0.5))
            .unwrap();
        assert_eq!(
            search.update_neighborhood(n, candidate).unwrap(),
            Neighborhood::Real(RealBracket(Triple {
                left: candidate,
                ..t
            }))
        );
        assert_eq!(search.last_move, Move::Near(Side::Left));
        search.arena.get_mut(candidate).unwrap().mag_def = -1;
        assert_eq!(
            search.update_neighborhood(n, candidate).unwrap(),
            Neighborhood::Real(RealBracket(Triple {
                left: t.left,
                middle: candidate,
                right: t.middle
            }))
        );
        assert_eq!(search.last_move, Move::Mid(Side::Left));

        search.arena.remove(candidate).unwrap();
        assert!(search.update_neighborhood(n, candidate).is_err());
        assert!(
            search
                .eval(Evaluation::Split {
                    edge: candidate,
                    middle: t.middle
                })
                .is_err()
        );
        assert!(
            search
                .strategy(Neighborhood::Real(RealBracket(Triple {
                    left: candidate,
                    ..t
                })))
                .is_err()
        );
    }

    #[test]
    fn sample_collision_is_not_a_root_without_a_residual() {
        let load = load();
        let mut search = Search::new(&load).unwrap();
        let first = search.arena.insert_after(None, sample(1.0, 0.5)).unwrap();
        let candidate = search.run_trial(Cx::new(1.0 + 2e-13, 0.0), None).unwrap();
        assert!(!search.arena.get(candidate).unwrap().is_root());
        assert_ne!(
            search.arena.frequency(candidate).unwrap(),
            search.arena.frequency(first).unwrap()
        );
        assert!(search.coincident_root(Cx::new(-1., 0.)).unwrap());
        assert!(!search.coincident_root(Cx::new(1., 0.)).unwrap());
    }

    #[test]
    fn root_collisions_reuse_identity_and_verification_prunes_and_deflates() {
        let mut load = load();
        load.order = 2;
        load.elements.push((1, 1));
        load.terms.push(PencilTerm {
            row: 1,
            col: 1,
            conductance: Some(1.),
            capacitance: Some(1.),
        });
        let mut search = Search::new(&load).unwrap();
        let root = search.run_trial(Cx::new(-1.0, 0.0), None).unwrap();
        assert_eq!(search.arena.get(root).unwrap().multiplicity(), 1);
        search.verify(root).unwrap();
        assert_eq!(search.nzeros, 1);
        let repeated = search.run_trial(Cx::new(-1.0, 0.0), None).unwrap();
        assert_eq!(root, repeated);
        assert_eq!(search.arena.get(root).unwrap().multiplicity(), 2);
        assert_eq!(search.arena.head(), Some(root));
        assert_eq!(search.arena.get(root).unwrap().next(), None);
        let nearby = search
            .arena
            .insert_after(Some(root), sample(-0.99, 0.75))
            .unwrap();
        let complex = search
            .arena
            .insert_after(
                Some(nearby),
                Trial::new(Cx::new(0.0, 1.0), Cx::new(1.0, 0.0), 0, TrialKind::Sample),
            )
            .unwrap();
        let distant = search
            .arena
            .insert_after(Some(complex), sample(10.0, 1.0))
            .unwrap();
        search.arena.get_mut(distant).unwrap().kind = TrialKind::Minimum;
        assert!(search.verify(distant).is_err());
        assert!(
            search
                .update_neighborhood(
                    Neighborhood::Incomplete(Incomplete::Middle {
                        left: None,
                        right: None
                    }),
                    root
                )
                .is_err()
        );

        search.verify(root).unwrap();
        assert_eq!(search.nzeros, 2);
        assert!(search.arena.get(nearby).is_err());
        assert!(search.arena.get(complex).is_err());
        assert_eq!(search.arena.get(root).unwrap().next(), Some(distant));
        let d = search.arena.get(distant).unwrap();
        assert_eq!(d.prev(), Some(root));
        assert_eq!(d.kind, TrialKind::Sample);
        assert!((materialize_power(d.f_def.re, d.mag_def) + 1.0 / 11.0).abs() < 1e-16);
    }
}
