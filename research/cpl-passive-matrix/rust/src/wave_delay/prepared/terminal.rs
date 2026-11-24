//! General coupled-mode physical MNA branch equations from the
//! prepared passive scattering response.
//!
//! For normalized modal port voltage alpha and inward line current beta,
//! incident a=(alpha+beta)/2, reflected b=(alpha-beta)/2.
//! The prepared line gives b=b0+S*a; rearranging without inversion:
//!
//! Coefficients are (I+S) for inward current, (S-I) for voltage, RHS -2b0.
//!
//! Therefore branch-current unknowns remain valid even if I+S is
//! singular (an ideal short would instead impose a voltage constraint).
//! Physically, alpha=Tv*V and beta=Ti*I, where Tv and Ti are
//! inverse power-reciprocal modal voltage/current bases. Applying them
//! blockwise at both ports yields the exact row coefficients needed
//! for coupled MNA voltage differences and separate branch currents.

use super::PreparedWaveStep;

#[derive(Debug, Clone)]
pub struct ModalPortEquations {
    modes: usize,
    voltage: Vec<f64>,
    current: Vec<f64>,
    rhs: Vec<f64>,
}

/// Physical MNA device equations in signed port voltage differences
/// and physical inward line branch currents, ordered near then far.
#[derive(Debug, Clone)]
pub struct PhysicalPortEquations {
    modes: usize,
    voltage: Vec<f64>,
    current: Vec<f64>,
    rhs: Vec<f64>,
    voltage_from_physical: Vec<f64>,
    current_from_physical: Vec<f64>,
}

impl PreparedWaveStep {
    /// Construct inversion-free reciprocal terminal branch-current
    /// equations, not a potentially singular nodal admittance inverse.
    ///
    /// # Errors
    /// Empty or malformed prepared terminal affine response.
    pub fn modal_terminal_equations(&self) -> Result<ModalPortEquations, &'static str> {
        let latest = self.knots.last().ok_or("no prepared line endpoint")?;
        let width = 2 * self.modes;
        let mut current = vec![0.; width * width];
        let mut voltage = vec![0.; width * width];
        let mut rhs = vec![0.; width];
        for row in 0..width {
            let wave = if row < self.modes {
                &latest.near_out[row]
            } else {
                &latest.far_out[row - self.modes]
            };
            if wave.jac.len() != width
                || !wave.base.is_finite()
                || wave.jac.iter().any(|v| !v.is_finite())
            {
                return Err("invalid prepared modal reflection map");
            }
            rhs[row] = -2. * wave.base;
            for (col, &coeff) in wave.jac.iter().enumerate() {
                let identity = if row == col { 1. } else { 0. };
                voltage[row * width + col] = coeff - identity;
                current[row * width + col] = coeff + identity;
            }
        }
        Ok(ModalPortEquations {
            modes: self.modes,
            voltage,
            current,
            rhs,
        })
    }
}

impl ModalPortEquations {
    #[must_use]
    pub const fn modes(&self) -> usize {
        self.modes
    }
    #[must_use]
    pub fn voltage_matrix(&self) -> &[f64] {
        &self.voltage
    }
    #[must_use]
    pub fn current_matrix(&self) -> &[f64] {
        &self.current
    }
    #[must_use]
    pub fn rhs(&self) -> &[f64] {
        &self.rhs
    }

    /// Apply a full n*n real modal voltage and current inverse basis
    /// to *each* end of the same cable. The n conductors are coupled:
    /// these are matrices, not per-conductor scalar impedances.
    ///
    /// Input transformations use row-major orientation:
    /// Both transformations are row-major matrices from physical
    /// conductor coordinates to power-normalized modal coordinates.
    ///
    /// # Errors
    /// Incorrect basis dimension or nonfinite transformation product.
    pub fn project_physical(
        &self,
        voltage_from_physical: &[f64],
        current_from_physical: &[f64],
    ) -> Result<PhysicalPortEquations, &'static str> {
        let n = self.modes;
        let size = n.checked_mul(n).ok_or("modal basis dimension overflow")?;
        if voltage_from_physical.len() != size
            || current_from_physical.len() != size
            || voltage_from_physical
                .iter()
                .chain(current_from_physical)
                .any(|value| !value.is_finite())
        {
            return Err("invalid full physical modal transformation basis");
        }
        let width = 2 * n;
        let mut voltage = vec![0.; width * width];
        let mut current = vec![0.; width * width];
        for row in 0..width {
            for side in 0..2 {
                for col in 0..n {
                    let mut v = 0.;
                    let mut i = 0.;
                    for mode in 0..n {
                        let idx = row * width + side * n + mode;
                        v = self.voltage[idx].mul_add(voltage_from_physical[mode * n + col], v);
                        i = self.current[idx].mul_add(current_from_physical[mode * n + col], i);
                    }
                    if !v.is_finite() || !i.is_finite() {
                        return Err("modal to physical MNA stamp overflow");
                    }
                    voltage[row * width + side * n + col] = v;
                    current[row * width + side * n + col] = i;
                }
            }
        }
        Ok(PhysicalPortEquations {
            modes: n,
            voltage,
            current,
            rhs: self.rhs.clone(),
            voltage_from_physical: voltage_from_physical.to_vec(),
            current_from_physical: current_from_physical.to_vec(),
        })
    }
}
impl PhysicalPortEquations {
    /// Reconstruct the normalized waves that the converged physical
    /// terminal solution injects into the line: incident is half the
    /// sum of transformed conductor voltage and inward line current.
    /// Returned ordering is all near modes followed by all far modes.
    /// This closes the circuit solve -> history acceptance path.
    ///
    /// # Errors
    /// Incorrect port dimensions or nonfinite recovered waves.
    pub fn incident_from_physical(
        &self,
        terminal_voltage: &[f64],
        branch_current: &[f64],
    ) -> Result<Vec<f64>, &'static str> {
        let width = 2 * self.modes;
        if terminal_voltage.len() != width
            || branch_current.len() != width
            || terminal_voltage
                .iter()
                .chain(branch_current)
                .any(|x| !x.is_finite())
        {
            return Err("invalid solved physical coupled-line terminal state");
        }
        let mut incident = vec![0.; width];
        for end in 0..2 {
            for mode in 0..self.modes {
                let mut alpha = 0.;
                let mut beta = 0.;
                for conductor in 0..self.modes {
                    alpha = self.voltage_from_physical[mode * self.modes + conductor]
                        .mul_add(terminal_voltage[end * self.modes + conductor], alpha);
                    beta = self.current_from_physical[mode * self.modes + conductor]
                        .mul_add(branch_current[end * self.modes + conductor], beta);
                }
                let result = alpha.midpoint(beta);
                if !result.is_finite() {
                    return Err("nonfinite recovered modal incident wave");
                }
                incident[end * self.modes + mode] = result;
            }
        }
        Ok(incident)
    }

    #[must_use]
    pub const fn modes(&self) -> usize {
        self.modes
    }
    #[must_use]
    pub fn voltage_matrix(&self) -> &[f64] {
        &self.voltage
    }
    #[must_use]
    pub fn current_matrix(&self) -> &[f64] {
        &self.current
    }
    #[must_use]
    pub fn rhs(&self) -> &[f64] {
        &self.rhs
    }
}

/// One sparse MNA Jacobian coefficient. Grounded nodes are excluded
/// before constructing these triplets.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MnaTriplet {
    pub row: usize,
    pub col: usize,
    pub coefficient: f64,
}
/// One source contribution for a physical current-branch equation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MnaRhs {
    pub row: usize,
    pub value: f64,
}
#[derive(Debug)]
pub struct WaveMnaStamp {
    pub coefficients: Vec<MnaTriplet>,
    pub rhs: Vec<MnaRhs>,
}
impl PhysicalPortEquations {
    /// Stamp physical terminal equations using explicit branch currents.
    /// Each signal/return node receives signed KCL branch stamps and
    /// each branch equation receives the complete coupled physical
    /// voltage differences, branch-current coefficients and RHS.
    ///
    /// A missing node index denotes grounded SPICE node zero. Indexing
    /// and accumulation of repeated sparse locations are owned by
    /// the native matrix assembler, not this abstract coefficient stage.
    ///
    /// # Errors
    /// Incorrect port count, repeated branch unknown indices or
    /// exhausted sparse coefficient capacity.
    pub fn mna_stamp(
        &self,
        signal_nodes: &[Option<usize>],
        return_nodes: &[Option<usize>],
        branch_unknowns: &[usize],
    ) -> Result<WaveMnaStamp, &'static str> {
        let width = 2 * self.modes;
        if signal_nodes.len() != width
            || return_nodes.len() != width
            || branch_unknowns.len() != width
        {
            return Err("physical coupled-line terminal index count mismatch");
        }
        let mut sorted = branch_unknowns.to_vec();
        sorted.sort_unstable();
        if sorted.windows(2).any(|pair| pair[0] == pair[1]) {
            return Err("physical line branch unknown reused");
        }
        if signal_nodes
            .iter()
            .chain(return_nodes)
            .flatten()
            .any(|node| sorted.binary_search(node).is_ok())
        {
            return Err("physical line branch index overlaps a conductor node");
        }
        let mut coefficients = Vec::new();
        coefficients
            .try_reserve(width * width * 5)
            .map_err(|_| "physical coupled-line sparse matrix allocation failed")?;
        for col in 0..width {
            let branch = branch_unknowns[col];
            if let Some(node) = signal_nodes[col] {
                coefficients.push(MnaTriplet {
                    row: node,
                    col: branch,
                    coefficient: 1.,
                });
            }
            if let Some(node) = return_nodes[col] {
                coefficients.push(MnaTriplet {
                    row: node,
                    col: branch,
                    coefficient: -1.,
                });
            }
        }
        let mut rhs = Vec::with_capacity(width);
        for row in 0..width {
            let equation = branch_unknowns[row];
            for col in 0..width {
                let v = self.voltage[row * width + col];
                if v != 0. {
                    if let Some(signal) = signal_nodes[col] {
                        coefficients.push(MnaTriplet {
                            row: equation,
                            col: signal,
                            coefficient: v,
                        });
                    }
                    if let Some(ret) = return_nodes[col] {
                        coefficients.push(MnaTriplet {
                            row: equation,
                            col: ret,
                            coefficient: -v,
                        });
                    }
                }
                let i = self.current[row * width + col];
                if i != 0. {
                    coefficients.push(MnaTriplet {
                        row: equation,
                        col: branch_unknowns[col],
                        coefficient: i,
                    });
                }
            }
            rhs.push(MnaRhs {
                row: equation,
                value: self.rhs[row],
            });
        }
        Ok(WaveMnaStamp { coefficients, rhs })
    }
}

#[cfg(test)]
mod tests;
