//! General multiconductor telegrapher-equation transfer plan.
//!
//! Symmetric RLGC coupling is preserved as one noncommuting, 2N-dimensional
//! system. All matrices are physical per-unit-length coefficients.
use crate::maths::{Matrix, dense_complex::DenseComplex};
use crate::primitives::{Complex, Real, SpiceError, SpiceResult};

fn invalid(message: impl std::fmt::Display) -> SpiceError {
    SpiceError::Numerical {
        context: "coupled transmission line".to_owned(),
        message: message.to_string(),
    }
}

/// Matrix-memory and conductor-count limits for the coupled-line solver.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CoupledLineLimits {
    /// Highest allowed physical conductor count, excluding the returns.
    pub max_conductors: usize,
    /// Cap on the total dense complex exponential workspace in bytes.
    pub max_workspace_bytes: usize,
    /// Cap on exponent squaring iterations.
    pub max_squarings: usize,
}
impl Default for CoupledLineLimits {
    fn default() -> Self {
        Self {
            max_conductors: 32,
            max_workspace_bytes: 32 * 1024 * 1024,
            max_squarings: 128,
        }
    }
}
impl CoupledLineLimits {
    fn check(self, conductors: usize) -> SpiceResult<()> {
        if conductors == 0 || conductors > self.max_conductors {
            return Err(invalid("conductor count exceeds configured bounds"));
        }
        let bytes = conductors
            .checked_mul(2)
            .and_then(|v| v.checked_mul(v))
            .and_then(|v| v.checked_mul(std::mem::size_of::<Complex>()))
            .and_then(|v| v.checked_mul(16))
            .ok_or_else(|| invalid("matrix workspace allocation overflow"))?;
        if bytes > self.max_workspace_bytes {
            return Err(invalid("dense matrix workspace budget exceeded"));
        }
        Ok(())
    }
}

/// Positive-length, passive per-unit-length RLGC model.
#[derive(Debug, Clone)]
pub struct CoupledLinePlan {
    /// Physical length in the units used by the four per-length matrices.
    pub length: Real,
    /// Series resistance in ohms per length.
    pub r: Matrix,
    /// Series inductance in henries per length.
    pub l: Matrix,
    /// Shunt conductance in siemens per length.
    pub g: Matrix,
    /// Shunt capacitance in farads per length.
    pub c: Matrix,
    limits: CoupledLineLimits,
}
impl CoupledLinePlan {
    /// Expand C CPL model upper-triangular coefficients in row-major upper order.
    /// # Errors
    /// Wrong coefficient count, nonfinite coefficient or resource budget.
    pub fn unpack_symmetric(
        conductors: usize,
        values: &[Real],
        limits: CoupledLineLimits,
    ) -> SpiceResult<Matrix> {
        limits.check(conductors)?;
        let expected = conductors
            .checked_add(1)
            .and_then(|next| next.checked_mul(conductors))
            .map(|value| value / 2)
            .ok_or_else(|| invalid("triangular matrix dimension overflow"))?;
        if values.len() != expected || values.iter().any(|v| !v.is_finite()) {
            return Err(invalid(format!(
                "expected {expected} finite triangular values; got {}",
                values.len()
            )));
        }
        let mut result = Matrix::zeros(conductors, conductors)?;
        let mut index = 0;
        for row in 0..conductors {
            for col in row..conductors {
                result.set(row, col, values[index])?;
                result.set(col, row, values[index])?;
                index += 1;
            }
        }
        Ok(result)
    }

    /// Validate a general symmetric R/L/G/C cable with explicit length.
    ///
    /// Loss matrices R and G must be positive semidefinite; dynamic matrices
    /// L and C must be positive definite for distinct finite-speed modes.
    /// This permits cross-coupling, arbitrary modal rotations and zero losses.
    /// # Errors
    /// Invalid dimensions, active or singular material, allocation budget.
    pub fn new(
        length: Real,
        resistance: Matrix,
        inductance: Matrix,
        conductance: Matrix,
        capacitance: Matrix,
        limits: CoupledLineLimits,
    ) -> SpiceResult<Self> {
        let conductors = resistance.rows();
        limits.check(conductors)?;
        if !length.is_finite() || length <= 0. {
            return Err(invalid("cable length must be finite and positive"));
        }
        for (name, matrix, definite) in [
            ("R", &resistance, false),
            ("L", &inductance, true),
            ("G", &conductance, false),
            ("C", &capacitance, true),
        ] {
            if !matrix.is_square() || matrix.rows() != conductors {
                return Err(invalid(format!(
                    "{name} must be {conductors} by {conductors}"
                )));
            }
            let max = matrix
                .data()
                .iter()
                .fold(0.0_f64, |acc, &entry| acc.max(entry.abs()));
            if !max.is_finite() {
                return Err(invalid(format!("{name} contains nonfinite coefficients")));
            }
            for row in 0..conductors {
                for col in 0..row {
                    if matrix.get(row, col) != matrix.get(col, row) {
                        return Err(invalid(format!("{name} is nonsymmetric at ({row},{col})")));
                    }
                }
            }
            let values = matrix.symmetric_eigenvalues()?;
            let smallest = values
                .first()
                .copied()
                .ok_or_else(|| invalid("empty eigenspectrum"))?;
            let dimension = f64::from(
                u32::try_from(conductors).map_err(|_| invalid("dimension conversion overflow"))?,
            );
            let uncertainty = 128. * Real::EPSILON * dimension * max;
            if smallest < -uncertainty || (definite && smallest <= uncertainty) {
                return Err(invalid(format!(
                    "{name} is active or singular (lowest eigenvalue = {smallest:.7e})"
                )));
            }
        }
        Ok(Self {
            length,
            r: resistance,
            l: inductance,
            g: conductance,
            c: capacitance,
            limits,
        })
    }

    /// Parse the same upper-triangle per-length parameter arrays as C CPL.
    /// # Errors
    /// Invalid array lengths, material eigenvalues, or resource limits.
    pub fn from_packed(
        conductors: usize,
        length: Real,
        r: &[Real],
        l: &[Real],
        g: &[Real],
        c: &[Real],
        limits: CoupledLineLimits,
    ) -> SpiceResult<Self> {
        Self::new(
            length,
            Self::unpack_symmetric(conductors, r, limits)?,
            Self::unpack_symmetric(conductors, l, limits)?,
            Self::unpack_symmetric(conductors, g, limits)?,
            Self::unpack_symmetric(conductors, c, limits)?,
            limits,
        )
    }

    /// Count of physical coupled signal conductors.
    #[must_use]
    pub const fn conductors(&self) -> usize {
        self.r.rows()
    }

    fn operator(&self, hz: Real) -> SpiceResult<(DenseComplex, Real)> {
        if !hz.is_finite() || hz < 0. {
            return Err(invalid("frequency must be positive or zero"));
        }
        let omega = std::f64::consts::TAU * hz;
        if !omega.is_finite() {
            return Err(invalid("angular frequency overflow"));
        }
        let n = self.conductors();
        let mut series = Vec::new();
        let mut shunt = Vec::new();
        let (mut zmax, mut ymax) = (0.0_f64, 0.0_f64);
        series
            .try_reserve(n * n)
            .map_err(|_| invalid("series matrix allocation"))?;
        shunt
            .try_reserve(n * n)
            .map_err(|_| invalid("shunt matrix allocation"))?;
        for row in 0..n {
            let (mut zr, mut yr) = (0., 0.);
            for col in 0..n {
                let idx = row * n + col;
                let z = Complex::new(self.r.data()[idx], omega * self.l.data()[idx]);
                let y = Complex::new(self.g.data()[idx], omega * self.c.data()[idx]);
                if !z.is_finite() || !y.is_finite() {
                    return Err(invalid("nonfinite RLGC impedance or admittance"));
                }
                zr += z.magnitude();
                yr += y.magnitude();
                series.push(z);
                shunt.push(y);
            }
            zmax = zmax.max(zr);
            ymax = ymax.max(yr);
        }
        // Similarity transform state I' = Zref I, balancing Z and Y blocks.
        // Crucial for physical L~nH and C~pF, not a change to the line model.
        let impedance_scale = if zmax > 0. && ymax > 0. {
            (zmax / ymax).sqrt()
        } else {
            1.
        };
        if !impedance_scale.is_finite() || impedance_scale <= 0. {
            return Err(invalid("cannot normalize coupled wave impedances"));
        }
        let mut generator = DenseComplex::zeros(2 * n)?;
        for row in 0..n {
            for col in 0..n {
                let index = row * n + col;
                let upper = -series[index] * Complex::real(self.length / impedance_scale);
                let lower = -shunt[index] * Complex::real(self.length * impedance_scale);
                if !upper.is_finite() || !lower.is_finite() {
                    return Err(invalid("telegrapher matrix overflow"));
                }
                generator.set(row, n + col, upper)?;
                generator.set(n + row, col, lower)?;
            }
        }
        Ok((generator, impedance_scale))
    }

    /// Exact coupled first-order telegrapher transfer up to numerical precision.
    ///
    /// States are input/output vectors [V, I'] with I' = Zref·I and I
    /// oriented in the positive propagation direction.
    /// # Errors
    /// Invalid frequency, exponential range, ill-conditioned Padé solve.
    pub fn transfer_at(&self, hz: Real) -> SpiceResult<(DenseComplex, Real)> {
        let (operator, scale) = self.operator(hz)?;
        Ok((expm_pade13(&operator, self.limits.max_squarings)?, scale))
    }

    /// Physical 2N × 2N nodal admittance relating both-end *entering* currents
    /// to voltages measured from the corresponding end's return.
    ///
    /// The B transfer block is inverted. Singular B indicates an ideal
    /// voltage constraint (e.g. zero-ohm DC conductor or half-wave resonance),
    /// and is rejected rather than replaced with fictitious zero admittance.
    /// # Errors
    /// Singular transfer or nonfinite physical coupling.
    pub fn admittance_at(&self, hz: Real) -> SpiceResult<DenseComplex> {
        let conductors = self.conductors();
        let (transfer, scale) = self.transfer_at(hz)?;
        let mut block_a = DenseComplex::zeros(conductors)?;
        let mut block_b = DenseComplex::zeros(conductors)?;
        let mut block_c = DenseComplex::zeros(conductors)?;
        let mut block_d = DenseComplex::zeros(conductors)?;
        for row in 0..conductors {
            for col in 0..conductors {
                block_a.set(
                    row,
                    col,
                    transfer
                        .get(row, col)
                        .ok_or_else(|| invalid("transfer block A"))?,
                )?;
                block_b.set(
                    row,
                    col,
                    transfer
                        .get(row, conductors + col)
                        .ok_or_else(|| invalid("transfer block B"))?,
                )?;
                block_c.set(
                    row,
                    col,
                    transfer
                        .get(conductors + row, col)
                        .ok_or_else(|| invalid("transfer block C"))?,
                )?;
                block_d.set(
                    row,
                    col,
                    transfer
                        .get(conductors + row, conductors + col)
                        .ok_or_else(|| invalid("transfer block D"))?,
                )?;
            }
        }
        let inverse = block_b.inverse()?.ok_or_else(|| {
            invalid("singular voltage/current transfer: ideal or resonant line constraint")
        })?;
        let inva = inverse.mul(&block_a)?;
        let dinva = block_d.mul(&inva)?;
        let dinv = block_d.mul(&inverse)?;
        let mut result = DenseComplex::zeros(2 * conductors)?;
        for row in 0..conductors {
            for col in 0..conductors {
                let entries = [
                    (row, col, -inva.get(row, col).ok_or_else(|| invalid("Y11"))?),
                    (
                        row,
                        conductors + col,
                        inverse.get(row, col).ok_or_else(|| invalid("Y12"))?,
                    ),
                    (
                        conductors + row,
                        col,
                        dinva.get(row, col).ok_or_else(|| invalid("Y21"))?
                            - block_c
                                .get(row, col)
                                .ok_or_else(|| invalid("Y21 correction"))?,
                    ),
                    (
                        conductors + row,
                        conductors + col,
                        -dinv.get(row, col).ok_or_else(|| invalid("Y22"))?,
                    ),
                ];
                for (r, block_c, value) in entries {
                    let physical = value / Complex::real(scale);
                    if !physical.is_finite() {
                        return Err(invalid("nonfinite coupled physical Y"));
                    }
                    result.set(r, block_c, physical)?;
                }
            }
        }
        Ok(result)
    }
}

/// Padé-13 complex matrix exponential with scaling-and-squaring.
///
/// The full noncommuting 2N×2N system is treated directly. The certified
/// theta-13 bound applies to any submultiplicative norm; here the infinity
/// norm is used as a conservative, cheaply computable choice.
fn expm_pade13(input: &DenseComplex, max_squarings: usize) -> SpiceResult<DenseComplex> {
    const B: [Real; 14] = [
        64_764_752_532_480_000.,
        32_382_376_266_240_000.,
        7_771_770_303_897_600.,
        1_187_353_796_428_800.,
        129_060_195_264_000.,
        10_559_470_521_600.,
        670_442_572_800.,
        33_522_128_640.,
        1_323_241_920.,
        40_840_800.,
        960_960.,
        16380.,
        182.,
        1.,
    ];
    let n = input.dim();
    let mut norm = 0.0_f64;
    for row in input.data().chunks_exact(n.max(1)) {
        let value = row.iter().map(|z| z.magnitude()).sum::<Real>();
        norm = norm.max(value);
    }
    if !norm.is_finite() {
        return Err(invalid("matrix exponential has nonfinite norm"));
    }
    let mut scaled_norm = norm;
    let mut steps = 0;
    loop {
        if scaled_norm <= 5.371_920_351_148_152 {
            break;
        }
        if steps >= max_squarings {
            return Err(invalid(
                "propagation exponential exceeds squaring work budget",
            ));
        }
        steps += 1;
        scaled_norm *= 0.5;
    }
    let divisor =
        2.0_f64.powi(-i32::try_from(steps).map_err(|_| invalid("scaling exponent overflow"))?);
    let a = scaled(input, divisor)?;
    let a2 = a.mul(&a)?;
    let a4 = a2.mul(&a2)?;
    let a6 = a4.mul(&a2)?;
    let identity = DenseComplex::identity(n)?;
    let combine = |terms: &[(&DenseComplex, Real)]| -> SpiceResult<DenseComplex> {
        let mut out = DenseComplex::zeros(n)?;
        for &(matrix, weight) in terms {
            for row in 0..n {
                for col in 0..n {
                    let old = out
                        .get(row, col)
                        .ok_or_else(|| invalid("Padé accumulator row"))?;
                    let entry = matrix
                        .get(row, col)
                        .ok_or_else(|| invalid("Padé term row"))?;
                    out.set(row, col, old + entry * Complex::real(weight))?;
                }
            }
        }
        Ok(out)
    };
    let inner_u = a6.mul(&combine(&[(&a6, B[13]), (&a4, B[11]), (&a2, B[9])])?)?;
    let u = a.mul(&inner_u.add(&combine(&[
        (&a6, B[7]),
        (&a4, B[5]),
        (&a2, B[3]),
        (&identity, B[1]),
    ])?)?)?;
    let inner_v = a6.mul(&combine(&[(&a6, B[12]), (&a4, B[10]), (&a2, B[8])])?)?;
    let v = inner_v.add(&combine(&[
        (&a6, B[6]),
        (&a4, B[4]),
        (&a2, B[2]),
        (&identity, B[0]),
    ])?)?;
    let inverse = v
        .sub(&u)?
        .inverse()?
        .ok_or_else(|| invalid("Padé exponential denominator is singular"))?;
    let mut result = inverse.mul(&v.add(&u)?)?;
    for _ in 0..steps {
        result = result.mul(&result)?;
    }
    if result.data().iter().any(|entry| !entry.is_finite()) {
        return Err(invalid("matrix exponential overflow"));
    }
    Ok(result)
}

fn scaled(matrix: &DenseComplex, factor: Real) -> SpiceResult<DenseComplex> {
    let mut out = DenseComplex::zeros(matrix.dim())?;
    for row in 0..matrix.dim() {
        for col in 0..matrix.dim() {
            let value = matrix
                .get(row, col)
                .ok_or_else(|| invalid("scalar matrix coordinate"))?;
            out.set(row, col, value * Complex::real(factor))?;
        }
    }
    Ok(out)
}
