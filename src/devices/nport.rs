//! Physical small-signal N-port admittance from sampled Touchstone networks.
//!
//! Real positive references use incident/scattered *power* waves. The sampled
//! file is never treated as a causal time-domain model. Exact shorts or ideal
//! constraints that have no finite nodal Y remain explicit numerical errors.
use std::sync::Arc;

use crate::analysis::touchstone::{MixedPort, Parameter, Touchstone, TouchstoneLimits};
use crate::devices::linear::LinearContext;
use crate::devices::traits::{AnalysisMode, StampContext};
use crate::devices::{Device, MnaUnknowns};
use crate::maths::Vector;
use crate::maths::dense_complex::DenseComplex;
use crate::primitives::{Complex, NodeId, Real, SpiceError, SpiceResult};

fn invalid(message: impl std::fmt::Display) -> SpiceError {
    SpiceError::circuit(format!("measured N-port: {message}"))
}

/// Behavior outside the tabulated frequency interval.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Extrapolation {
    /// Reject frequencies without measured/interpolated samples.
    Reject,
    /// Hold the first or final measured matrix.
    Hold,
}
/// Physical positive and reference terminals of one port.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PortTerminals {
    /// Positive terminal.
    pub positive: NodeId,
    /// Negative/reference terminal, possibly ground.
    pub negative: NodeId,
}

/// Reusable measured linear N-port bound to circuit terminal identities.
#[derive(Debug, Clone)]
pub struct MeasuredNPort {
    name: String,
    network: Arc<Touchstone>,
    terminals: Vec<PortTerminals>,
    node_terminals: Vec<NodeId>,
    extrapolation: Extrapolation,
}

impl MeasuredNPort {
    /// Validate measured network and port attachments once, then retain them.
    /// # Errors
    /// Invalid network, mismatched port count or degenerate terminal pair.
    pub fn new(
        name: impl Into<String>,
        network: Arc<Touchstone>,
        terminals: Vec<PortTerminals>,
        extrapolation: Extrapolation,
    ) -> SpiceResult<Self> {
        network.validate(TouchstoneLimits::default())?;
        if terminals.len() != network.references.len()
            || terminals.iter().any(|p| p.positive == p.negative)
        {
            return Err(invalid(
                "port terminals must be distinct and match the measured port count",
            ));
        }
        let name = name.into();
        if name.is_empty() {
            return Err(invalid("measured device name cannot be empty"));
        }
        let mut node_terminals = Vec::new();
        node_terminals
            .try_reserve(terminals.len() * 2)
            .map_err(|_| invalid("terminal storage exhausted"))?;
        for port in &terminals {
            node_terminals.extend([port.positive, port.negative]);
        }
        Ok(Self {
            name,
            network,
            terminals,
            node_terminals,
            extrapolation,
        })
    }

    /// Borrow the immutable file data, including reference impedances.
    #[must_use]
    pub fn network(&self) -> &Touchstone {
        &self.network
    }

    /// Real-positive reference port count.
    #[must_use]
    pub fn port_count(&self) -> usize {
        self.terminals.len()
    }

    fn interpolate(&self, hz: Real) -> SpiceResult<Vec<Complex>> {
        if !hz.is_finite() || hz < 0. {
            return Err(invalid("frequency must be finite and nonnegative"));
        }
        let points = &self.network.points;
        let index = points.binary_search_by(|p| p.hz.total_cmp(&hz));
        match index {
            Ok(k) => Ok(points[k].matrix.clone()),
            Err(0) if self.extrapolation == Extrapolation::Hold => Ok(points[0].matrix.clone()),
            Err(k) if k == points.len() && self.extrapolation == Extrapolation::Hold => {
                Ok(points[k - 1].matrix.clone())
            }
            Err(k) if k == 0 || k == points.len() => {
                Err(invalid("frequency exceeds measured range"))
            }
            Err(k) => {
                let lo = &points[k - 1];
                let hi = &points[k];
                let fraction = (hz - lo.hz) / (hi.hz - lo.hz);
                if !fraction.is_finite() || !(0. ..=1.).contains(&fraction) {
                    return Err(invalid("frequency interpolation is numerically undefined"));
                }
                let mut data = Vec::new();
                data.try_reserve(lo.matrix.len())
                    .map_err(|_| invalid("cannot allocate interpolated matrix"))?;
                for (&a, &b) in lo.matrix.iter().zip(&hi.matrix) {
                    // Interpolate the rectangular phasor, never the wrapped
                    // phase angle. This preserves phase continuity across 360°.
                    let value = Complex::new(
                        a.re + fraction * (b.re - a.re),
                        a.im + fraction * (b.im - a.im),
                    );
                    if !value.is_finite() {
                        return Err(invalid("interpolation overflow"));
                    }
                    data.push(value);
                }
                Ok(data)
            }
        }
    }

    fn as_dense(&self, data: &[Complex]) -> SpiceResult<DenseComplex> {
        let n = self.port_count();
        let mut matrix = DenseComplex::zeros(n)?;
        for row in 0..n {
            for col in 0..n {
                matrix.set(row, col, data[row * n + col])?;
            }
        }
        Ok(matrix)
    }

    fn transform_matrix(&self, values: &DenseComplex) -> SpiceResult<DenseComplex> {
        let Some(order) = self.network.mixed_mode.as_ref() else {
            return Ok(values.clone());
        };
        let n = self.port_count();
        let mut modes = DenseComplex::zeros(n)?;
        for (row, axis) in order.iter().enumerate() {
            match *axis {
                MixedPort::Single(port) => modes.set(row, port, Complex::real(1.))?,
                MixedPort::Differential(a, b) => {
                    let factor = match self.network.parameter {
                        Parameter::S => std::f64::consts::FRAC_1_SQRT_2,
                        Parameter::Y => 1.,
                        Parameter::Z => 0.5,
                        _ => return Err(invalid("hybrid mixed-mode data is undefined")),
                    };
                    modes.set(row, a, Complex::real(factor))?;
                    modes.set(row, b, Complex::real(-factor))?;
                }
                MixedPort::Common(a, b) => {
                    let factor = match self.network.parameter {
                        Parameter::S => std::f64::consts::FRAC_1_SQRT_2,
                        Parameter::Y => 0.5,
                        Parameter::Z => 1.,
                        _ => return Err(invalid("hybrid mixed-mode data is undefined")),
                    };
                    modes.set(row, a, Complex::real(factor))?;
                    modes.set(row, b, Complex::real(factor))?;
                }
            }
        }
        let mut transposed = DenseComplex::zeros(n)?;
        for row in 0..n {
            for col in 0..n {
                transposed.set(row, col, modes.get(col, row).unwrap_or(Complex::ZERO))?;
            }
        }
        transposed.mul(values)?.mul(&modes)
    }

    /// Return the physical single-ended admittance I = Y V, in siemens.
    /// # Errors
    /// Missing sample, singular S->Y/Z->Y or H/G->Y conversion, or nonfinite
    /// interpolation/matrix arithmetic. No singularity is replaced with zero.
    pub fn admittance_at(&self, hz: Real) -> SpiceResult<DenseComplex> {
        let data = self.interpolate(hz)?;
        let native = self.as_dense(&data)?;
        let network = self.transform_matrix(&native)?;
        let n = self.port_count();
        let admittance = match self.network.parameter {
            Parameter::Y => network,
            Parameter::Z => network
                .inverse()?
                .ok_or_else(|| invalid("Z matrix is singular"))?,
            Parameter::S => {
                let identity = DenseComplex::identity(n)?;
                let right = identity.add(&network)?.inverse()?.ok_or_else(|| {
                    invalid("I+S is singular: ideal short/constraint requires a branch equation")
                })?;
                let mut y = identity.sub(&network)?.mul(&right)?;
                for row in 0..n {
                    for col in 0..n {
                        let r = self.network.references[row].sqrt()
                            * self.network.references[col].sqrt();
                        let z = y.get(row, col).unwrap_or(Complex::ZERO) / Complex::real(r);
                        if !z.is_finite() {
                            return Err(invalid("reference normalization overflow"));
                        }
                        y.set(row, col, z)?;
                    }
                }
                y
            }
            Parameter::H => {
                let [h11, h12, h21, h22] = [
                    network.get(0, 0),
                    network.get(0, 1),
                    network.get(1, 0),
                    network.get(1, 1),
                ]
                .map(|v| v.unwrap_or(Complex::ZERO));
                if h11 == Complex::ZERO {
                    return Err(invalid("hybrid H11 is a voltage constraint"));
                }
                let a = Complex::real(1.) / h11;
                let mut y = DenseComplex::zeros(2)?;
                y.set(0, 0, a)?;
                y.set(0, 1, -a * h12)?;
                y.set(1, 0, h21 * a)?;
                y.set(1, 1, h22 - h21 * a * h12)?;
                y
            }
            Parameter::G => {
                let [g11, g12, g21, g22] = [
                    network.get(0, 0),
                    network.get(0, 1),
                    network.get(1, 0),
                    network.get(1, 1),
                ]
                .map(|v| v.unwrap_or(Complex::ZERO));
                if g22 == Complex::ZERO {
                    return Err(invalid("hybrid G22 is a voltage constraint"));
                }
                let a = Complex::real(1.) / g22;
                let mut y = DenseComplex::zeros(2)?;
                y.set(0, 0, g11 - g12 * a * g21)?;
                y.set(0, 1, g12 * a)?;
                y.set(1, 0, -g21 * a)?;
                y.set(1, 1, a)?;
                y
            }
        };
        if admittance.data().iter().any(|x| !x.is_finite()) {
            return Err(invalid("non-finite physical admittance"));
        }
        Ok(admittance)
    }

    /// Expand physical node-pair currents into MNA stamps. A port current is
    /// positive into its positive terminal; ground contributes no row.
    /// # Errors
    /// Unbound node, invalid sample, nonfinite stamp or allocation failure.
    pub fn nodal_stamps_at(
        &self,
        hz: Real,
        unknowns: &MnaUnknowns,
    ) -> SpiceResult<Vec<(usize, usize, Complex)>> {
        let y = self.admittance_at(hz)?;
        let mut positions = Vec::new();
        for port in &self.terminals {
            positions.push((
                unknowns.checked_node_row(port.positive)?,
                unknowns.checked_node_row(port.negative)?,
            ));
        }
        let count = self
            .port_count()
            .checked_mul(self.port_count())
            .and_then(|n| n.checked_mul(4))
            .ok_or_else(|| invalid("stamp-count overflow"))?;
        let mut stamps = Vec::new();
        stamps
            .try_reserve(count)
            .map_err(|_| invalid("cannot allocate N-port stamp buffer"))?;
        for (r, &(rp, rn)) in positions.iter().enumerate() {
            for (c, &(cp, cn)) in positions.iter().enumerate() {
                let value = y.get(r, c).unwrap_or(Complex::ZERO);
                for (out, sign_out) in [(rp, 1.), (rn, -1.)] {
                    for (input, sign_in) in [(cp, 1.), (cn, -1.)] {
                        if let (Some(row), Some(col)) = (out, input) {
                            let contribution = Complex::real(sign_out * sign_in) * value;
                            if contribution != Complex::ZERO {
                                stamps.push((row, col, contribution));
                            }
                        }
                    }
                }
            }
        }
        Ok(stamps)
    }
}

impl Device for MeasuredNPort {
    fn name(&self) -> &str {
        &self.name
    }
    fn designator(&self) -> char {
        'n'
    }
    fn terminals(&self) -> &[NodeId] {
        &self.node_terminals
    }
    fn small_signal_depends_on_frequency(&self) -> bool {
        true
    }

    fn setup_matrix(
        &self,
        setup: &mut crate::devices::setup::MatrixSetup<'_>,
    ) -> SpiceResult<bool> {
        for output in &self.terminals {
            for input in &self.terminals {
                for row in [output.positive, output.negative] {
                    for col in [input.positive, input.negative] {
                        setup.nodes(row, col);
                    }
                }
            }
        }
        Ok(true)
    }

    fn stamp(&self, context: &mut StampContext<'_>) -> SpiceResult<()> {
        match context.mode {
            AnalysisMode::OperatingPoint | AnalysisMode::DcSweep => {
                let admittance = self.admittance_at(0.)?;
                if admittance.data().iter().any(|y| y.im != 0.) {
                    return Err(invalid(
                        "a nonzero imaginary DC admittance has no real-valued bias equation",
                    ));
                }
                for (row, col, value) in self.nodal_stamps_at(0., context.unknowns)? {
                    context.matrix.add(row, col, value.re)?;
                }
                Ok(())
            }
            AnalysisMode::Ac { .. } => Err(invalid(
                "complex AC stamps must be assembled through assemble_small_signal",
            )),
            AnalysisMode::Transient { .. } => Err(invalid(
                "transient analysis requires a verified causal rational macromodel",
            )),
        }
    }

    fn assemble_linear(&self, context: &mut LinearContext<'_>) -> SpiceResult<()> {
        let admittance = self.admittance_at(0.)?;
        if admittance.data().iter().any(|y| y.im != 0.) {
            return Err(invalid(
                "nonreal DC conductance cannot be used as a bias operator",
            ));
        }
        for (r, c, v) in self.nodal_stamps_at(0., context.unknowns)? {
            context.system.a.add(r, c, v.re)?;
        }
        Ok(())
    }

    fn assemble_small_signal(
        &self,
        context: &mut LinearContext<'_>,
        _bias: &Vector,
    ) -> SpiceResult<()> {
        let frequency = context.model_context.frequency;
        for (r, c, v) in self.nodal_stamps_at(frequency, context.unknowns)? {
            if v.re != 0. {
                context.system.a.add(r, c, v.re)?;
            }
            if v.im != 0. {
                context.dynamic_prescaled(r, c, v.im)?;
            }
        }
        Ok(())
    }
}
