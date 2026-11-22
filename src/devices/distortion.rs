//! Device nonlinearities for small-signal `.disto` distortion analysis.
//!
//! C reference: `src/spicelib/analysis/distoan.c` drives a Volterra-series
//! analysis whose device hooks (`DEVdisto`: `dio/diodisto.c`,
//! `bjt/bjtdisto.c`, `mos1/mos1dist.c`) do two things:
//!
//! * `D_SETUP` (`diodset.c`, `bjtdset.c`, `mos1dset.c`) stores, at the
//!   operating point, the second- and third-order Taylor coefficients of every
//!   nonlinear branch current or charge with respect to up to three
//!   controlling voltages;
//! * `D_TWOF1`, `D_THRF1`, `D_F1PF2`, `D_F1MF2`, `D_2F1MF2` evaluate those
//!   polynomials on the lower-order Volterra kernels (the node solutions at the
//!   lower-order frequencies) through `dloadfns.c` and add the result, times
//!   `j omega` for a charge, to the right-hand side of the next solve.
//!
//! The port keeps that split. A device describes its nonlinearities **at the
//! bias point** through [`crate::devices::Device::distortion`], returning a
//! [`DeviceDistortion`]: a list of [`DistortionTerm`]s, each a Taylor
//! polynomial ([`Taylor`]) of a current or charge flowing between two nodes in
//! up to three [`Control`] voltages. The analysis owns the kernels, the
//! frequencies and the polynomial evaluation ([`CircuitDistortion::add_products`], C
//! `dloadfns.c`). Linear devices, which C distorts only through their AC
//! matrix, return [`DeviceDistortion::Linear`]; independent sources with
//! `distof1`/`distof2` inputs return [`DeviceDistortion::Input`]. The trait
//! default is an explicit `NotYetPorted` error, so a device whose distortion
//! is not ported can never be treated as linear by accident.
//!
//! `deriv::Dderivs` transcribes C's derivative arithmetic (`Dderivs`,
//! `src/maths/deriv/*.c`), which `bjtdset.c` uses to differentiate the
//! Gummel-Poon expressions, and the `dloadfns` module C's load functions.

use crate::devices::{Circuit, MnaUnknowns, ModelContext};
use crate::maths::Vector;
use crate::primitives::{Complex, NodeId, Real, SpiceError, SpiceResult};

pub(crate) mod deriv;
mod dloadfns;

/// A controlling voltage: the sum of the node differences `v(a) - v(b)` of
/// its pairs. Almost always one pair; C occasionally forms sums (the BJT's
/// `vbx` kernel is `(vb - vb') + (vb' - vc')`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Control {
    /// The `[a, b]` node pairs whose differences are summed.
    pub pairs: Vec<[NodeId; 2]>,
    /// Further pairs, summed like [`Self::pairs`] except in the `f1 - f2`
    /// product, where C reads their `H1(f2)` kernel **without** the
    /// conjugation that makes it `H1(-f2)` (`bjtdisto.c`'s `D_F1MF2` base-
    /// resistance kernel `i1hm2z` lacks the minus sign). Empty except for
    /// that BJT quirk.
    pub unconjugated_in_f1_minus_f2: Vec<[NodeId; 2]>,
    /// Pairs subtracted from the sum of [`Self::pairs`], as `mos1dist.c`
    /// forms `vbd` as `vbs - vds` from the kernels' differences (rather than
    /// `vb - vd`, which rounds differently). Empty for most controls.
    pub minus: Vec<[NodeId; 2]>,
}

impl Control {
    /// The voltage `v(a) - v(b)`.
    #[must_use]
    pub fn between(a: NodeId, b: NodeId) -> Self {
        Self::sum(vec![[a, b]])
    }

    /// The sum of the differences of `pairs`.
    #[must_use]
    pub const fn sum(pairs: Vec<[NodeId; 2]>) -> Self {
        Self {
            pairs,
            unconjugated_in_f1_minus_f2: Vec::new(),
            minus: Vec::new(),
        }
    }

    /// The difference `(v(a) - v(b)) - (v(c) - v(d))` of two node
    /// differences, evaluated in that order on the kernels.
    #[must_use]
    pub fn difference(plus: [NodeId; 2], minus: [NodeId; 2]) -> Self {
        Self {
            minus: vec![minus],
            ..Self::between(plus[0], plus[1])
        }
    }

    /// Adds `pairs` as [`Self::unconjugated_in_f1_minus_f2`] pairs.
    #[must_use]
    pub fn with_unconjugated_in_f1_minus_f2(mut self, pairs: Vec<[NodeId; 2]>) -> Self {
        self.unconjugated_in_f1_minus_f2.extend(pairs);
        self
    }
}

/// A control with node identities resolved once, in the original pair order.
type RowPair = [Option<usize>; 2];

#[derive(Debug)]
struct BoundControl {
    pairs: Vec<RowPair>,
    unconjugated: Vec<RowPair>,
    minus: Vec<RowPair>,
}

fn bind_pair(unknowns: &MnaUnknowns, [a, b]: [NodeId; 2]) -> SpiceResult<RowPair> {
    Ok([unknowns.checked_node_row(a)?, unknowns.checked_node_row(b)?])
}

impl BoundControl {
    fn bind(control: Control, unknowns: &MnaUnknowns) -> SpiceResult<Self> {
        let bind = |pairs: Vec<[NodeId; 2]>| {
            pairs
                .into_iter()
                .map(|pair| bind_pair(unknowns, pair))
                .collect::<SpiceResult<Vec<_>>>()
        };
        Ok(Self {
            pairs: bind(control.pairs)?,
            unconjugated: bind(control.unconjugated_in_f1_minus_f2)?,
            minus: bind(control.minus)?,
        })
    }

    fn sum_of(pairs: &[RowPair], x: &[Complex]) -> Complex {
        let at = |row: Option<usize>| row.map_or(Complex::ZERO, |row| x[row]);
        pairs
            .iter()
            .fold(Complex::ZERO, |sum, [a, b]| sum + (at(*a) - at(*b)))
    }

    fn signed(&self, x: &[Complex]) -> Complex {
        let sum = Self::sum_of(&self.pairs, x);
        if self.minus.is_empty() {
            sum
        } else {
            sum - Self::sum_of(&self.minus, x)
        }
    }

    fn value(&self, x: &[Complex]) -> Complex {
        self.signed(x) + Self::sum_of(&self.unconjugated, x)
    }

    fn minus_f2_value(&self, x: &[Complex]) -> Complex {
        self.signed(x).conj() + Self::sum_of(&self.unconjugated, x)
    }
}

/// Second- and third-order coefficients of a controlled function.
///
/// Second- and third-order Taylor coefficients of a function of the controls
/// `x`, `y`, `z` (C's `cxx`, ..., `cxyz` arguments of `dloadfns.c`): the
/// function's expansion is `xx x^2 + yy y^2 + ... + xy x y + ... + xxx x^3 +
/// ... + xyz x y z` beyond its linear part. Missing controls have zero
/// coefficients.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Taylor {
    /// Coefficient of the x² monomial.
    pub xx: Real,
    /// Coefficient of the y² monomial.
    pub yy: Real,
    /// Coefficient of the z² monomial.
    pub zz: Real,
    /// Coefficient of the xy monomial.
    pub xy: Real,
    /// Coefficient of the yz monomial.
    pub yz: Real,
    /// Coefficient of the xz monomial.
    pub xz: Real,
    /// Coefficient of the x³ monomial.
    pub xxx: Real,
    /// Coefficient of the y³ monomial.
    pub yyy: Real,
    /// Coefficient of the z³ monomial.
    pub zzz: Real,
    /// Coefficient of the x²y monomial.
    pub xxy: Real,
    /// Coefficient of the x²z monomial.
    pub xxz: Real,
    /// Coefficient of the xy² monomial.
    pub xyy: Real,
    /// Coefficient of the y²z monomial.
    pub yyz: Real,
    /// Coefficient of the xz² monomial.
    pub xzz: Real,
    /// Coefficient of the yz² monomial.
    pub yzz: Real,
    /// Coefficient of the xyz monomial.
    pub xyz: Real,
}

impl Taylor {
    /// A one-variable polynomial `c2 x^2 + c3 x^3` (C's `D1*` functions).
    #[must_use]
    pub fn single(c2: Real, c3: Real) -> Self {
        Self {
            xx: c2,
            xxx: c3,
            ..Self::default()
        }
    }

    const fn second(&self) -> [(usize, usize, Real); 6] {
        [
            (0, 0, self.xx),
            (1, 1, self.yy),
            (2, 2, self.zz),
            (0, 1, self.xy),
            (1, 2, self.yz),
            (0, 2, self.xz),
        ]
    }

    const fn third(&self) -> [(usize, usize, usize, Real); 10] {
        [
            (0, 0, 0, self.xxx),
            (1, 1, 1, self.yyy),
            (2, 2, 2, self.zzz),
            (0, 0, 1, self.xxy),
            (0, 0, 2, self.xxz),
            (0, 1, 1, self.xyy),
            (1, 1, 2, self.yyz),
            (0, 2, 2, self.xzz),
            (1, 2, 2, self.yzz),
            (0, 1, 2, self.xyz),
        ]
    }

    fn validate(&self, arity: usize) -> SpiceResult<()> {
        if !self.second().iter().all(|c| c.2.is_finite())
            || !self.third().iter().all(|c| c.3.is_finite())
        {
            return Err(SpiceError::circuit("nonfinite Taylor coefficient"));
        }
        if self
            .second()
            .iter()
            .any(|&(i, j, c)| c != 0. && (i >= arity || j >= arity))
            || self
                .third()
                .iter()
                .any(|&(i, j, k, c)| c != 0. && (i >= arity || j >= arity || k >= arity))
        {
            return Err(SpiceError::circuit(
                "Taylor coefficient refers to an absent control",
            ));
        }
        Ok(())
    }
}

/// Whether a term's polynomial is a current or a charge.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Response {
    /// A memoryless current.
    Current,
    /// A charge: its current is `j omega` times the polynomial, `omega` being
    /// the angular frequency of the response being solved for.
    Charge,
}

/// One nonlinearity: the higher-order part of a current or charge flowing
/// from `nodes[0]` to `nodes[1]` (C subtracts the evaluated polynomial from
/// the right-hand side at `nodes[0]` and adds it at `nodes[1]`).
#[derive(Debug, Clone, PartialEq)]
pub struct DistortionTerm {
    /// Current or charge.
    pub response: Response,
    /// The branch the response flows through.
    pub nodes: [NodeId; 2],
    /// The controls `x`, `y`, `z` (one to three).
    pub controls: Vec<Control>,
    /// The Taylor coefficients in those controls.
    pub taylor: Taylor,
    /// C quirk: `mos1dist.c`'s `D_2F1MF2` case reads its `H2(f1, f1)`
    /// kernel from the first-order `H1(f1)` vector (`r1H1ptr` where
    /// `r2H11ptr` is meant). Set by [`Self::with_first_order_im3_kernel`] for
    /// MOS1 terms so that `2f1 - f2` reproduces C.
    pub first_order_im3_kernel: bool,
    /// A charge polynomial in the same controls whose `j omega` current C
    /// folds into this (current) term's value before loading it
    /// (`diodisto.c`'s `temp += -omega * D1i(...)`), set by
    /// [`Self::with_charge`].
    pub charge: Option<Taylor>,
}

impl DistortionTerm {
    /// A term.
    #[must_use]
    pub const fn new(
        response: Response,
        nodes: [NodeId; 2],
        controls: Vec<Control>,
        taylor: Taylor,
    ) -> Self {
        Self {
            response,
            nodes,
            controls,
            taylor,
            first_order_im3_kernel: false,
            charge: None,
        }
    }

    /// Folds the charge polynomial `charge` into this term as `diodisto.c`
    /// does (see [`Self::charge`]).
    #[must_use]
    pub const fn with_charge(mut self, charge: Taylor) -> Self {
        self.charge = Some(charge);
        self
    }

    /// Marks the term with `mos1dist.c`'s `2f1 - f2` kernel quirk (see
    /// [`Self::first_order_im3_kernel`]).
    #[must_use]
    pub const fn with_first_order_im3_kernel(mut self) -> Self {
        self.first_order_im3_kernel = true;
        self
    }
}

/// A distortion input of an independent source (`distof1`/`distof2`,
/// `vsrcpar.c`/`isrcpar.c`): magnitude and phase in degrees.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DistortionInput {
    /// Magnitude (C defaults it to 1 when the setter has no value).
    pub magnitude: Real,
    /// Phase in degrees (default 0).
    pub phase: Real,
}

impl DistortionInput {
    /// The half-amplitude phasor C drives the first-order solve with
    /// (`cktdisto.c`: `0.5 * mag * (cos, sin)(pi * phase / 180)`).
    #[must_use]
    pub fn phasor(self) -> Complex {
        let angle = std::f64::consts::PI * self.phase / 180.;
        Complex::new(
            0.5 * self.magnitude * angle.cos(),
            0.5 * self.magnitude * angle.sin(),
        )
    }
}

/// What a device contributes to `.disto`.
#[derive(Debug, Clone, PartialEq)]
pub enum DeviceDistortion {
    /// Linear: the device enters only through its small-signal matrix (C
    /// devices without `DEVdisto`).
    Linear,
    /// An independent source with distortion inputs (it is otherwise linear).
    Input {
        /// The `distof1` input, if given.
        f1: Option<DistortionInput>,
        /// The `distof2` input, if given.
        f2: Option<DistortionInput>,
    },
    /// Nonlinear terms at the bias point (possibly none).
    Terms(Vec<DistortionTerm>),
}

/// A validated operating-point view for device distortion and carried state.
#[derive(Debug, Clone, Copy)]
pub struct DistortionContext<'a> {
    /// Circuit/nominal temperature and junction `gmin`.
    pub model_context: &'a ModelContext,
    unknowns: &'a MnaUnknowns,
    bias: &'a Vector,
}

impl<'a> DistortionContext<'a> {
    /// Construct a checked operating-point view.
    ///
    /// # Errors
    /// Bias dimensions or values do not match the numbering.
    pub fn new(
        model_context: &'a ModelContext,
        unknowns: &'a MnaUnknowns,
        bias: &'a Vector,
    ) -> SpiceResult<Self> {
        if bias.len() != unknowns.len() || !bias.is_finite() {
            return Err(SpiceError::circuit(
                "invalid distortion bias dimensions/values",
            ));
        }
        Ok(Self {
            model_context,
            unknowns,
            bias,
        })
    }

    /// The operating-point voltage, zero only for ground.
    ///
    /// # Errors
    /// A non-ground node is not mapped to the operating point.
    pub fn voltage(&self, node: NodeId) -> SpiceResult<Real> {
        match self.unknowns.checked_node_row(node)? {
            None => Ok(0.),
            Some(row) => self.bias.get(row).ok_or_else(|| {
                SpiceError::circuit(format!("missing bias row for distortion node {node}"))
            }),
        }
    }
}

/// One validated current/charge term in finalized MNA rows.
#[derive(Debug)]
struct BoundTerm {
    response: Response,
    rows: RowPair,
    controls: Vec<BoundControl>,
    taylor: Taylor,
    first_order_im3_kernel: bool,
    charge: Option<Taylor>,
}

impl BoundTerm {
    fn bind(term: DistortionTerm, unknowns: &MnaUnknowns) -> SpiceResult<Self> {
        let arity = term.controls.len();
        if !(1..=3).contains(&arity) {
            return Err(SpiceError::circuit(
                "a distortion term needs one to three controls",
            ));
        }
        term.taylor.validate(arity)?;
        if let Some(charge) = term.charge {
            if term.response != Response::Current {
                return Err(SpiceError::circuit(
                    "folded charge requires a current response",
                ));
            }
            charge.validate(arity)?;
        }
        Ok(Self {
            response: term.response,
            rows: bind_pair(unknowns, term.nodes)?,
            controls: term
                .controls
                .into_iter()
                .map(|control| BoundControl::bind(control, unknowns))
                .collect::<SpiceResult<_>>()?,
            taylor: term.taylor,
            first_order_im3_kernel: term.first_order_im3_kernel,
            charge: term.charge,
        })
    }

    fn values(&self, kernel: &[Complex]) -> dloadfns::Values {
        let mut values = [Complex::ZERO; 3];
        for (slot, control) in values.iter_mut().zip(&self.controls) {
            *slot = control.value(kernel);
        }
        values
    }

    fn minus_f2(&self, kernel: &[Complex]) -> dloadfns::Values {
        let mut values = [Complex::ZERO; 3];
        for (slot, control) in values.iter_mut().zip(&self.controls) {
            *slot = control.minus_f2_value(kernel);
        }
        values
    }
}

/// Bound distortion terms tied to the immutable finalized circuit borrow.
/// Neither topology nor the bound row interpretation can change while it lives.
pub struct CircuitDistortion<'a> {
    circuit: &'a Circuit,
    terms: Vec<BoundTerm>,
    inputs: Vec<(String, Option<DistortionInput>, Option<DistortionInput>)>,
}

impl CircuitDistortion<'_> {
    /// The source names and their first- and second-tone inputs.
    #[must_use]
    pub fn inputs(&self) -> &[(String, Option<DistortionInput>, Option<DistortionInput>)] {
        &self.inputs
    }

    /// Number of bound nonlinear terms.
    #[must_use]
    pub const fn term_count(&self) -> usize {
        self.terms.len()
    }
}

/// Collect and bind distortion terms once at the operating point.
///
/// # Errors
/// Stale numbering, invalid bias/device physics, unsupported distortion,
/// invalid node identities, control arity, or Taylor/input coefficients.
pub fn circuit_distortion<'a>(
    circuit: &'a Circuit,
    context: &ModelContext,
    bias: &Vector,
) -> SpiceResult<CircuitDistortion<'a>> {
    circuit.check_numbering()?;
    let mut result = CircuitDistortion {
        circuit,
        terms: Vec::new(),
        inputs: Vec::new(),
    };
    let distortion_context = DistortionContext::new(context, circuit.unknowns(), bias)?;
    // CKTdisto visits the devices in CKTload order.
    for index in circuit.load_order() {
        let device = &circuit.devices()[index];
        match device.distortion(&distortion_context)? {
            DeviceDistortion::Linear => {}
            DeviceDistortion::Input { f1, f2 } => {
                let finite = |input: Option<DistortionInput>| {
                    input.is_none_or(|i| {
                        i.magnitude.is_finite() && i.phase.is_finite() && i.phasor().is_finite()
                    })
                };
                if !finite(f1) || !finite(f2) {
                    return Err(SpiceError::circuit(format!(
                        "{}: nonfinite distortion input",
                        device.name()
                    )));
                }
                if f1.is_some() || f2.is_some() {
                    result.inputs.push((device.name().to_owned(), f1, f2));
                }
            }
            DeviceDistortion::Terms(terms) => {
                for term in terms {
                    let term = BoundTerm::bind(term, circuit.unknowns()).map_err(|error| {
                        SpiceError::circuit(format!("distortion of {}: {error}", device.name()))
                    })?;
                    result.terms.push(term);
                }
            }
        }
    }
    Ok(result)
}

/// Exactly the lower-order kernels required for one higher-order product.
/// Missing kernels are unrepresentable.
#[derive(Debug, Clone, Copy)]
pub enum Product<'a> {
    /// Second harmonic.
    TwoF1 {
        /// H1(f1).
        h1: &'a [Complex],
    },
    /// Third harmonic.
    ThreeF1 {
        /// H1(f1).
        h1: &'a [Complex],
        /// H2(f1,f1).
        h2: &'a [Complex],
    },
    /// Sum-frequency intermodulation.
    F1PlusF2 {
        /// H1(f1).
        h1: &'a [Complex],
        /// H1(f2).
        h1_f2: &'a [Complex],
    },
    /// Difference-frequency intermodulation.
    F1MinusF2 {
        /// H1(f1).
        h1: &'a [Complex],
        /// H1(f2), conjugated according to the control's C semantics.
        h1_f2: &'a [Complex],
    },
    /// Third-order intermodulation.
    TwoF1MinusF2 {
        /// H1(f1).
        h1: &'a [Complex],
        /// H1(f2).
        h1_f2: &'a [Complex],
        /// H2(f1,f1).
        h2: &'a [Complex],
        /// H2(f1,-f2).
        h2_minus: &'a [Complex],
    },
}

impl Product<'_> {
    fn validate(self, n: usize) -> SpiceResult<()> {
        let check = |kernel: &[Complex]| {
            if kernel.len() != n {
                return Err(SpiceError::circuit("distortion kernel size"));
            }
            if !kernel.iter().all(|value| value.is_finite()) {
                return Err(SpiceError::circuit("nonfinite distortion kernel"));
            }
            Ok(())
        };
        match self {
            Self::TwoF1 { h1 } => check(h1),
            Self::ThreeF1 { h1, h2 } => {
                check(h1)?;
                check(h2)
            }
            Self::F1PlusF2 { h1, h1_f2 } | Self::F1MinusF2 { h1, h1_f2 } => {
                check(h1)?;
                check(h1_f2)
            }
            Self::TwoF1MinusF2 {
                h1,
                h1_f2,
                h2,
                h2_minus,
            } => {
                check(h1)?;
                check(h1_f2)?;
                check(h2)?;
                check(h2_minus)
            }
        }
    }

    fn values(self, term: &BoundTerm) -> KernelValues {
        match self {
            Self::TwoF1 { h1 } => KernelValues::TwoF1(term.values(h1)),
            Self::ThreeF1 { h1, h2 } => KernelValues::ThreeF1(term.values(h1), term.values(h2)),
            Self::F1PlusF2 { h1, h1_f2 } => KernelValues::F12(term.values(h1), term.values(h1_f2)),
            Self::F1MinusF2 { h1, h1_f2 } => {
                KernelValues::F12(term.values(h1), term.minus_f2(h1_f2))
            }
            Self::TwoF1MinusF2 {
                h1,
                h1_f2,
                h2,
                h2_minus,
            } => {
                let x = term.values(h1);
                let b = term.values(h1_f2).map(Complex::conj);
                let x2 = if term.first_order_im3_kernel {
                    x
                } else {
                    term.values(h2)
                };
                KernelValues::TwoF1MinusF2(x, b, x2, term.values(h2_minus))
            }
        }
    }
}

/// A term's control values in the kernels a [`Product`] reads.
enum KernelValues {
    TwoF1(dloadfns::Values),
    ThreeF1(dloadfns::Values, dloadfns::Values),
    F12(dloadfns::Values, dloadfns::Values),
    TwoF1MinusF2(
        dloadfns::Values,
        dloadfns::Values,
        dloadfns::Values,
        dloadfns::Values,
    ),
}

impl KernelValues {
    fn evaluate(&self, t: &Taylor) -> Complex {
        match *self {
            Self::TwoF1(x) => dloadfns::two_f1(t, x),
            Self::ThreeF1(x, x2) => dloadfns::three_f1(t, x, x2),
            Self::F12(x, b) => dloadfns::f12(t, x, b),
            Self::TwoF1MinusF2(x, b, x2, m) => dloadfns::two_f1_minus_f2(t, x, b, x2, m),
        }
    }
}

impl CircuitDistortion<'_> {
    /// Adds every term's contribution to `rhs` for `product` (C `DEVdisto` with
    /// the `dloadfns.c` polynomial evaluators, [`dloadfns`]): with `X`, `B`,
    /// `X2`, `M` the control values in `H1(f1)`, `H1(+-f2)`, `H2(f1,f1)` and
    /// `H2(f1,-f2)`,
    ///
    /// | product | value of `c_ij v_i v_j` / `c_ijk v_i v_j v_k` |
    /// | --- | --- |
    /// | `2f1` | `X_i X_j` / 0 |
    /// | `3f1` | `X_i X2_j + X_j X2_i` / `X_i X_j X_k` |
    /// | `f1 +- f2` | `(X_i B_j + X_j B_i) / 2` / 0 |
    /// | `2f1 - f2` | `(2 (X_i M_j + X_j M_i) + B_i X2_j + B_j X2_i) / 3` / `(X_i X_j B_k + X_i X_k B_j + X_j X_k B_i) / 3` |
    ///
    /// A charge's value is multiplied by `j omega`, `omega` being the response's
    /// angular frequency (negative for `f1 - f2` below `f2`, as in C). The sums
    /// are evaluated in C's operand order and grouping, so the right-hand side
    /// rounds exactly as C's does.
    ///
    /// # Errors
    /// Invalid kernel/RHS dimensions, nonfinite inputs, or nonfinite frequency.
    pub fn add_products(
        &self,
        product: Product<'_>,
        omega: Real,
        rhs: &mut [Complex],
    ) -> SpiceResult<()> {
        let n = self.circuit.unknown_count();
        if rhs.len() != n || !rhs.iter().all(|value| value.is_finite()) {
            return Err(SpiceError::circuit(
                "invalid distortion right-hand side dimensions/values",
            ));
        }
        if !omega.is_finite() {
            return Err(SpiceError::circuit(
                "nonfinite distortion angular frequency",
            ));
        }
        product.validate(n)?;
        // No validation or node lookups remain in the stamping loop.
        for term in &self.terms {
            let inputs = product.values(term);
            let value = inputs.evaluate(&term.taylor);
            // A charge's current is `j omega` times its polynomial: C's `temp =
            // -omega * DFi(...)`, `itemp = omega * DFn(...)`.
            let (mut temp, mut itemp) = match term.response {
                Response::Current => (value.re, value.im),
                Response::Charge => (-omega * value.im, omega * value.re),
            };
            if let Some(charge) = &term.charge {
                // diodisto.c: `temp += -omega * D1i(...)`, `itemp += omega *
                // D1n(...)` onto the junction current's polynomial.
                let charge = inputs.evaluate(charge);
                let charge_real = -omega * charge.im;
                let charge_imag = omega * charge.re;
                temp += charge_real;
                itemp += charge_imag;
            }
            let [a, b] = term.rows;
            if let Some(row) = a {
                rhs[row].re -= temp;
                rhs[row].im -= itemp;
            }
            if let Some(row) = b {
                rhs[row].re += temp;
                rhs[row].im += itemp;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::Taylor;
    use super::deriv::Dderivs;

    /// A variable at `value` (C's `d_p` with its value set).
    fn var(index: usize, value: f64) -> Dderivs {
        let mut d = Dderivs::variable(index);
        d.value = value;
        d
    }

    /// Checks a derivative structure against central finite differences of
    /// its closed form.
    fn check(d: &Dderivs, f: impl Fn(f64, f64, f64) -> f64, at: [f64; 3]) {
        let h = 1e-3;
        let point = |dp: f64, dq: f64| f(at[0] + dp, at[1] + dq, at[2]);
        let value = point(0., 0.);
        assert!((d.value - value).abs() <= 1e-12 * value.abs().max(1.));
        let wanted = [
            (d.d1_p, (point(h, 0.) - point(-h, 0.)) / (2. * h)),
            (d.d1_q, (point(0., h) - point(0., -h)) / (2. * h)),
            (
                d.d2_p2,
                (point(h, 0.) - 2. * value + point(-h, 0.)) / (h * h),
            ),
            (
                d.d2_pq,
                (point(h, h) - point(h, -h) - point(-h, h) + point(-h, -h)) / (4. * h * h),
            ),
            (
                d.d3_p3,
                (point(2. * h, 0.) - 2. * point(h, 0.) + 2. * point(-h, 0.) - point(-2. * h, 0.))
                    / (2. * h * h * h),
            ),
        ];
        for (index, (got, want)) in wanted.into_iter().enumerate() {
            assert!(
                (got - want).abs() <= 1e-4 * want.abs().max(1e-3),
                "{index}: {got} vs {want}"
            );
        }
    }

    #[test]
    fn derivative_arithmetic_matches_closed_forms() {
        let at = [0.3, 0.7, -0.2];
        let p = var(0, at[0]);
        let q = var(1, at[1]);
        let r = var(2, at[2]);
        let mut q1 = q;
        q1.value += 1.;
        let f = p.mul(&q).plus(&r).exp().div(&q1.sqrt());
        check(&f, |p, q, r| (p * q + r).exp() / (q + 1.).sqrt(), at);
        let g = p.times(2.).tan().mul(&p.inv());
        check(&g, |p, _, _| (2. * p).tan() / p, at);
        // The mixed third derivative of p q r is exactly 1.
        assert_eq!(p.mul(&q).mul(&r).d3_pqr, 1.);
        // sqrtder.c: all derivatives vanish at a zero argument.
        assert_eq!(Dderivs::default().sqrt(), Dderivs::default());
        let t = Taylor::single(2., 3.);
        assert_eq!((t.xx, t.xxx, t.yy), (2., 3., 0.));
    }
}
