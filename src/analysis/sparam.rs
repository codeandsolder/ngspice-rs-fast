//! `.sp` — S-parameter analysis over RF port sources.
//!
//! Ported from `src/spicelib/analysis/span.c` (`SPan`), `cktspdum.c`
//! (`CKTspCalcPowerWave`, `CKTspCalcSMatrix`, `CKTspDump`) and the port hooks
//! of `src/spicelib/devices/vsrc/vsrcacld.c` (`VSRCspinit`, `VSRCspupdate`),
//! an `RFSPICE` build option of ngspice:
//!
//! 1. the operating point and small-signal system are prepared exactly as for
//!    `.ac` ([`crate::analysis::ac::SmallSignal`]), on the same frequency grid;
//! 2. in `MODESP` every independent AC excitation is off (`VSRCacLoad` loads
//!    zero for every voltage source); at each frequency each port `j` in turn
//!    drives a unit voltage into its ideal source branch (`VSRCspupdate`)
//!    while every port keeps its series `z0` (see
//!    [`crate::devices::sources`]);
//! 3. for every port `i`, the terminal voltage `V` and the current `I` into
//!    the positive terminal give the power waves `a = ki (V + z0 I)` and
//!    `b = ki (V - z0 I)`, `ki = 1/(2 sqrt(z0))`, filling column `j` of `A`
//!    and `B`;
//! 4. `S = B A^-1`, `Z = Gn^-1 (E - S)^-1 (S Z0 + Z0) Gn` and
//!    `Y = Gn^-1 (S Z0 + Z0)^-1 (E - S) Gn` with `Z0 = diag(z0)` and
//!    `Gn = diag(2 ki)`.
//!
//! The plot (`SP Analysis`, complex) holds the frequency, the node voltages
//! and branch currents of the **last** port's excitation (C dumps
//! `CKTrhsOld` after the final `NIspSolve`), then `S_i_j`, `Y_i_j`, `Z_i_j`
//! (row `i` = response port, column `j` = driven port) and `v(Rbase)`, the
//! reference impedance of port 1. See `docs/port/SPARAM.md` for the
//! deliberate divergences: RF noise follows C's covariance path; AC current sources with a
//! nonzero imaginary phasor (which C's `span.c` copies into every port
//! excitation) are refused, and a Y or Z matrix that does not exist (a
//! singular conversion) is written as zeros, C's `cinverse` contract for an
//! exactly singular matrix.

use crate::analysis::ac::{GridKind, SmallSignal, frequency_grid};
use crate::analysis::linear::{plot, unsupported};
use crate::analysis::results::Variable;
use crate::analysis::{AnalysisPlan, Plot};
use crate::devices::Circuit;
use crate::devices::linear::SourceKind;
use crate::devices::noise::InstanceNoise;
use crate::maths::complex::ComplexLu;
use crate::maths::dense_complex::DenseComplex;
use crate::primitives::{Complex, Real, SpiceError, SpiceResult, filled_vec, parse_spice_number};

/// Resource limits for one retained RF sweep. They bound RF-specific
/// work and numeric storage; ordinary circuit elaboration and sparse
/// matrix assembly/factorization have their own resource constraints.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RfLimits {
    /// Weighted work units across all frequencies, including actual
    /// retained sparse entries traversed and noise-generator covariance work.
    pub work_units: usize,
    /// Bytes of retained complex cells and RF numeric workspace.
    pub numeric_bytes: usize,
    /// Maximum output variables, including the circuit's base columns.
    pub variables: usize,
}

impl Default for RfLimits {
    fn default() -> Self {
        Self {
            work_units: 500_000_000,
            numeric_bytes: 256 * 1024 * 1024,
            variables: 65_536,
        }
    }
}

fn resource(message: impl std::fmt::Display) -> SpiceError {
    unsupported(format!(".sp RF resource limit: {message}"))
}

fn product(terms: &[usize]) -> SpiceResult<usize> {
    terms
        .iter()
        .try_fold(1_usize, |a, &b| a.checked_mul(b))
        .ok_or_else(|| resource("dimension product overflows"))
}

fn sum(terms: &[usize]) -> SpiceResult<usize> {
    terms
        .iter()
        .try_fold(0_usize, |a, &b| a.checked_add(b))
        .ok_or_else(|| resource("dimension sum overflows"))
}

/// The known aggregate cost is checked before schema construction or
/// operating-point work. Factor fill-in and noise generators are charged
/// when they become known, before their RF loops execute.
struct RfBudget {
    remaining: usize,
    width: usize,
}

impl RfBudget {
    fn preflight(
        ports: usize,
        unknowns: usize,
        observations: usize,
        frequencies: usize,
        noise: bool,
        bias: bool,
        limits: RfLimits,
    ) -> SpiceResult<Self> {
        if ports == 0 {
            return Err(SpiceError::circuit(
                "no RF port is present (a V source with portnum), cannot run sp analysis",
            ));
        }
        let square = product(&[ports, ports])?;
        let width = sum(&[
            unknowns,
            observations,
            2,
            product(&[square, 3 + usize::from(noise)])?,
            4 * usize::from(noise && ports == 2),
        ])?;
        if width > limits.variables {
            return Err(resource(format!(
                "{width} variables exceed {}",
                limits.variables
            )));
        }
        // Thirteen dense port matrices (including constants and inverse
        // scratch); one additional covariance matrix and adjoint block
        // only for noise. RHS, internal work (sentinel), output, row and
        // bound impedance coefficient vectors.
        let retained = product(&[width, frequencies])?;
        let bias_cells = if bias {
            sum(&[unknowns, observations])?
        } else {
            0
        };
        let matrices = product(&[square, PortAlgebra::MATRIX_COUNT + usize::from(noise)])?;
        let adjoints = if noise {
            product(&[ports, unknowns])?
        } else {
            0
        };
        let numeric_cells = sum(&[
            retained,
            bias_cells,
            width,
            matrices,
            adjoints,
            product(&[unknowns, 3])?,
            1,
            observations,
            product(&[ports, 6 + 2 * usize::from(noise)])?,
        ])?;
        let bytes = product(&[numeric_cells, std::mem::size_of::<Complex>()])?;
        if bytes > limits.numeric_bytes {
            return Err(resource(format!(
                "{bytes} numeric bytes exceed {}",
                limits.numeric_bytes
            )));
        }
        // Three inversions (two row updates each), three full products,
        // diagonal/elementwise passes, wave formation and vector traversal.
        let dense = sum(&[product(&[9, square, ports])?, product(&[32, square])?])?;
        let rhs = product(&[4, unknowns, ports, 1 + usize::from(noise)])?;
        let base = product(&[frequencies, sum(&[dense, rhs, width])?])?;
        let mut budget = Self {
            remaining: limits.work_units,
            width,
        };
        budget.charge(base)?;
        Ok(budget)
    }

    fn charge(&mut self, units: usize) -> SpiceResult<()> {
        self.remaining = self.remaining.checked_sub(units).ok_or_else(|| {
            resource(format!(
                "{units} work units exceed {} remaining",
                self.remaining
            ))
        })?;
        Ok(())
    }

    fn charge_factor(
        &mut self,
        factor_entries: usize,
        ports: usize,
        transpose: bool,
    ) -> SpiceResult<()> {
        // Forward and backward traversal; the actual number of stored
        // factors is a stronger bound than the original MNA pattern nnz.
        self.charge(product(&[
            2,
            factor_entries,
            ports,
            1 + usize::from(transpose),
        ])?)
    }

    fn charge_noise(&mut self, ports: usize, sources: usize) -> SpiceResult<()> {
        // Per generator: two port^2 passes (Y transform and covariance)
        // and one voltage traversal. Charge even omitted generator kinds.
        self.charge(product(&[
            sources,
            ports,
            sum(&[product(&[2, ports])?, 2])?,
        ])?)
    }
}

/// One RF port bound to matrix rows.
#[derive(Debug, Clone)]
struct BoundPort {
    number: usize,
    z0: Complex,
    ki: Complex,
    inverse_z0: Complex,
    positive: Option<usize>,
    negative: Option<usize>,
    branch: usize,
}

impl BoundPort {
    fn voltage(&self, x: &[Complex]) -> Complex {
        let at = |row: Option<usize>| row.map_or(Complex::ZERO, |row| x[row]);
        at(self.positive) - at(self.negative)
    }
}

pub(super) fn parse(positional: &[String], reltol: Real) -> SpiceResult<(Vec<Real>, bool)> {
    if !matches!(positional.len(), 4 | 5) {
        return Err(unsupported(".sp lin|dec|oct points start stop [donoise]"));
    }
    let donoise = if let Some(flag) = positional.get(4) {
        // dot_sp() reads donoise as IF_INTEGER, (int) floor(0.5 + value), and
        // SPsetParm enables noise only for exactly 1.
        let value = parse_spice_number(flag)
            .filter(|v| v.is_finite())
            .ok_or_else(|| unsupported(format!(".sp donoise must be an integer, found {flag}")))?;
        (1. ..2.).contains(&(value + 0.5))
    } else {
        false
    };

    let grid = frequency_grid(positional, reltol, GridKind::Sp)?;
    Ok((grid, donoise))
}

/// Ports, grid and aggregate resource budget bound to the finalized
/// circuit's row numbering for this sweep.
struct RfPlan<'a> {
    ports: Vec<BoundPort>,
    grid: &'a [Real],
    unknowns: usize,
    budget: RfBudget,
    noise: bool,
}

impl<'a> RfPlan<'a> {
    fn bind(
        circuit: &Circuit,
        grid: &'a [Real],
        noise: bool,
        bias: bool,
        limits: RfLimits,
    ) -> SpiceResult<Self> {
        let count = circuit
            .devices()
            .iter()
            .filter(|device| device.rf_port().is_some())
            .count();
        let unknowns = circuit.unknown_count();
        let budget = RfBudget::preflight(
            count,
            unknowns,
            circuit.observations().len(),
            grid.len(),
            noise,
            bias,
            limits,
        )?;
        if grid
            .iter()
            .any(|&f| !(2. * std::f64::consts::PI * f).is_finite())
        {
            return Err(unsupported(".sp angular frequency is not finite"));
        }
        let ports = bind_ports(circuit, count)?;
        Ok(Self {
            ports,
            grid,
            unknowns,
            budget,
            noise,
        })
    }
}

pub(super) fn run(
    circuit: &mut Circuit,
    request: &AnalysisPlan,
    context: &crate::analysis::AnalysisContext,
    grid: &[Real],
    donoise: bool,
    include_bias: bool,
) -> SpiceResult<Vec<Plot>> {
    let settings = request.dc_settings().clone();
    circuit.finalize()?;
    let mut plan = RfPlan::bind(circuit, grid, donoise, include_bias, context.rf_limits)?;
    let mut small_signal = SmallSignal::prepare(circuit, request, context, settings)?;
    let mut plots = small_signal.retained_bias(circuit, request, context, include_bias)?;
    reject_imaginary_current_excitation(small_signal.system())?;
    let mut workspace = RfWorkspace::new(&plan)?;
    let mut plot = parameter_plot(circuit, plan.ports.len(), donoise, plan.budget.width)?;
    plot.reserve_points(plan.grid.len())?;
    let reference = plan.ports[0].z0;
    for &f in plan.grid {
        let factor = small_signal.at_frequency(circuit, context, f, |system| {
            let factor =
                crate::analysis::ac::ac_matrix(circuit, system, 2. * std::f64::consts::PI * f)?
                    .factorize()?;
            plan.budget.charge_factor(
                factor.factor_storage().factor_entries,
                plan.ports.len(),
                plan.noise,
            )?;
            workspace.solve_ports(&plan.ports, &factor, f)?;
            Ok(factor)
        })?;
        workspace.point.clear();
        workspace.point.push(Complex::real(f));
        workspace.point.extend_from_slice(&workspace.solve.solution);
        if !circuit.observations().is_empty() {
            workspace
                .point
                .extend(circuit.observe_parameters(&context.model_context().with_frequency(f))?);
        }
        for matrix in [
            &workspace.algebra.s,
            &workspace.algebra.y,
            &workspace.algebra.z,
        ] {
            workspace.point.extend_from_slice(matrix.data());
        }
        workspace.point.push(reference);
        if let Some(noise) = &mut workspace.noise {
            let generators = crate::devices::noise::circuit_noise(
                circuit,
                &context.model_context().with_frequency(f),
                small_signal.bias(),
                Some(small_signal.state()),
            )?;
            let sources = generators.iter().try_fold(0_usize, |total, instance| {
                sum(&[total, instance.sources.len()])
            })?;
            plan.budget.charge_noise(plan.ports.len(), sources)?;
            noise.evaluate(
                circuit,
                &plan.ports,
                &factor,
                &workspace.algebra.y,
                &generators,
                &mut workspace.solve,
            )?;
            noise.append_results(
                &plan.ports,
                &workspace.algebra.y,
                context,
                &mut workspace.point,
            )?;
        }
        plot.push_point(&workspace.point)?;
    }
    plots.push(plot);
    Ok(plots)
}

/// Fixed-dimensional port algebra. Every buffer is allocated once and
/// overwritten for each frequency, including singular-conversion outputs.
struct PortAlgebra {
    incident: DenseComplex,
    scattered: DenseComplex,
    s: DenseComplex,
    y: DenseComplex,
    z: DenseComplex,
    identity: DenseComplex,
    zref: DenseComplex,
    e_minus_s: DenseComplex,
    s_z0_plus_z0: DenseComplex,
    inverse: DenseComplex,
    inverse_scratch: DenseComplex,
    temp: DenseComplex,
    product: DenseComplex,
    z0: Vec<Complex>,
    gn: Vec<Complex>,
    gn_inverse: Vec<Complex>,
}

impl PortAlgebra {
    const MATRIX_COUNT: usize = 13;

    fn new(ports: &[BoundPort]) -> SpiceResult<Self> {
        let n = ports.len();
        let zeros = || DenseComplex::zeros(n);
        let z0 = ports.iter().map(|port| port.z0).collect::<Vec<_>>();
        let gn = ports
            .iter()
            .map(|port| Complex::real(2. * port.ki.re))
            .collect();
        let gn_inverse = ports
            .iter()
            .map(|port| Complex::real(1. / (2. * port.ki.re)))
            .collect();
        Ok(Self {
            incident: zeros()?,
            scattered: zeros()?,
            s: zeros()?,
            y: zeros()?,
            z: zeros()?,
            identity: DenseComplex::identity(n)?,
            zref: DenseComplex::diagonal(&z0)?,
            e_minus_s: zeros()?,
            s_z0_plus_z0: zeros()?,
            inverse: zeros()?,
            inverse_scratch: zeros()?,
            temp: zeros()?,
            product: zeros()?,
            z0,
            gn,
            gn_inverse,
        })
    }

    fn calculate(&mut self, frequency: Real) -> SpiceResult<()> {
        if !self
            .incident
            .inverse_into(&mut self.inverse, &mut self.inverse_scratch)?
        {
            return Err(SpiceError::Numerical {
                context: ".sp".to_owned(),
                message: format!("incident power-wave matrix is singular at {frequency} Hz"),
            });
        }
        self.scattered.mul_into(&self.inverse, &mut self.s)?;
        self.identity.sub_into(&self.s, &mut self.e_minus_s)?;
        self.s.right_diagonal_into(&self.z0, &mut self.temp)?;
        self.temp.add_into(&self.zref, &mut self.s_z0_plus_z0)?;
        // Preserve the original evaluation order. Singular conversion
        // yields zeros, not the stale result of a previous frequency.
        if self
            .e_minus_s
            .inverse_into(&mut self.inverse, &mut self.inverse_scratch)?
        {
            self.s_z0_plus_z0
                .right_diagonal_into(&self.gn, &mut self.temp)?;
            self.inverse.mul_into(&self.temp, &mut self.product)?;
            self.product
                .left_diagonal_into(&self.gn_inverse, &mut self.z)?;
        } else {
            self.z.clear();
        }
        if self
            .s_z0_plus_z0
            .inverse_into(&mut self.inverse, &mut self.inverse_scratch)?
        {
            self.e_minus_s
                .right_diagonal_into(&self.gn, &mut self.temp)?;
            self.inverse.mul_into(&self.temp, &mut self.product)?;
            self.product
                .left_diagonal_into(&self.gn_inverse, &mut self.y)?;
        } else {
            self.y.clear();
        }
        Ok(())
    }
}

struct SolveBuffers {
    rhs: Vec<Complex>,
    scratch: Vec<Complex>,
    solution: Vec<Complex>,
}

impl SolveBuffers {
    fn new(unknowns: usize) -> SpiceResult<Self> {
        let buffer = |length| filled_vec(length, Complex::ZERO, ".sp solve workspace");
        Ok(Self {
            rhs: buffer(unknowns)?,
            scratch: buffer(unknowns + 1)?,
            solution: buffer(unknowns)?,
        })
    }

    fn solve(&mut self, factor: &ComplexLu, transpose: bool) -> SpiceResult<()> {
        factor.solve_into(&self.rhs, &mut self.scratch, &mut self.solution, transpose)
    }
}

struct RfWorkspace {
    algebra: PortAlgebra,
    solve: SolveBuffers,
    point: Vec<Complex>,
    noise: Option<NoiseWorkspace>,
}

impl RfWorkspace {
    fn new(plan: &RfPlan<'_>) -> SpiceResult<Self> {
        let mut point = filled_vec(plan.budget.width, Complex::ZERO, ".sp point")?;
        point.clear();
        Ok(Self {
            algebra: PortAlgebra::new(&plan.ports)?,
            solve: SolveBuffers::new(plan.unknowns)?,
            point,
            noise: plan
                .noise
                .then(|| NoiseWorkspace::new(plan.ports.len(), plan.unknowns))
                .transpose()?,
        })
    }

    fn solve_ports(&mut self, ports: &[BoundPort], factor: &ComplexLu, f: Real) -> SpiceResult<()> {
        self.solve.rhs.fill(Complex::ZERO);
        for (column, driven) in ports.iter().enumerate() {
            // VSRCspupdate: only the active source branch has a unit RHS.
            self.solve.rhs[driven.branch] = Complex::real(1.);
            self.solve.solve(factor, false)?;
            self.solve.rhs[driven.branch] = Complex::ZERO;
            for (row, port) in ports.iter().enumerate() {
                let v = port.voltage(&self.solve.solution);
                let i = -self.solve.solution[port.branch];
                self.algebra
                    .incident
                    .set(row, column, port.ki * (v + port.z0 * i))?;
                self.algebra
                    .scattered
                    .set(row, column, port.ki * (v - port.z0 * i))?;
            }
        }
        self.algebra.calculate(f)
    }
}

/// Schema in the same order as the per-frequency point assembly.
fn parameter_plot(
    circuit: &Circuit,
    port_count: usize,
    donoise: bool,
    width: usize,
) -> SpiceResult<Plot> {
    let plot = plot(
        circuit,
        "sp1",
        "SP Analysis",
        Some(("frequency", "frequency")),
        true,
    )?;
    let base = plot.variables();
    let mut variables = Vec::new();
    variables
        .try_reserve_exact(width)
        .map_err(|e| resource(format!("allocating output schema: {e}")))?;
    variables.extend_from_slice(base);
    for (prefix, unit) in [("S", "s-param"), ("Y", "admittance"), ("Z", "impedance")] {
        for i in 1..=port_count {
            for j in 1..=port_count {
                variables.push(Variable::complex(format!("{prefix}_{i}_{j}"), unit));
            }
        }
    }
    variables.push(Variable::complex("v(Rbase)", "voltage"));
    if donoise {
        for i in 1..=port_count {
            for j in 1..=port_count {
                variables.push(Variable::complex(format!("i(Cy_{i}_{j})"), "current"));
            }
        }
        if port_count == 2 {
            for (name, unit) in [
                ("NF", "decibel"),
                ("SOpt", "notype"),
                ("NFmin", "decibel"),
                ("Rn", "impedance"),
            ] {
                variables.push(Variable::complex(name, unit));
            }
        }
    }
    if variables.len() != width {
        return Err(SpiceError::circuit(
            ".sp schema does not match bound row count",
        ));
    }
    Ok(Plot::new(
        plot.name.clone(),
        plot.plotname.clone(),
        plot.flags(),
        variables,
    ))
}

/// The circuit's RF ports in port-number order, bound to their rows.
/// `Circuit::finalize` has already checked the numbering is `1..=N`.
fn bind_ports(circuit: &Circuit, count: usize) -> SpiceResult<Vec<BoundPort>> {
    let mut ports = Vec::new();
    ports
        .try_reserve_exact(count)
        .map_err(|e| resource(format!("allocating port bindings: {e}")))?;
    for (index, device) in circuit.devices().iter().enumerate() {
        let Some(port) = device.rf_port() else {
            continue;
        };
        let branch = circuit
            .branch_rows(index)
            .and_then(|rows| (rows.len() == 1).then_some(rows.start))
            .ok_or_else(|| {
                SpiceError::circuit(format!("{}: RF port without a branch row", device.name()))
            })?;
        let row = |terminal: usize| {
            device
                .terminals()
                .get(terminal)
                .and_then(|node| circuit.unknowns().node_row(*node))
        };
        ports.push(BoundPort {
            number: port.number,
            z0: Complex::real(port.z0),
            ki: Complex::real(port.ki()),
            inverse_z0: Complex::real(1. / port.z0),
            positive: row(0),
            negative: row(1),
            branch,
        });
    }
    if ports.is_empty() {
        // span.c: "Error: No RF Port is present, cannot run sp analysis".
        return Err(SpiceError::circuit(
            "no RF port is present (a V source with portnum), cannot run sp analysis",
        ));
    }
    ports.sort_by_key(|port| port.number);
    if ports
        .iter()
        .enumerate()
        .any(|(index, port)| port.number != index + 1)
    {
        return Err(SpiceError::circuit("RF ports must be numbered 1..=N"));
    }
    Ok(ports)
}

/// `span.c` saves the AC load's RHS once and restores it before each port
/// excitation, but copies the *imaginary* RHS over the real one and starts
/// from a zero imaginary part. In `MODESP` voltage sources load nothing, so a
/// current source's AC phasor reaches the port solves only through its
/// imaginary part, as a real excitation: C's S-parameters then depend on an
/// unrelated source. The port refuses that case. A purely real current-source
/// phasor is discarded by the same copy, as here.
fn reject_imaginary_current_excitation(
    system: &crate::devices::linear::LinearSystem,
) -> SpiceResult<()> {
    if let Some(source) = system
        .sources
        .iter()
        .find(|s| s.kind == SourceKind::Current && s.ac.im != 0.)
    {
        return Err(unsupported(format!(
            "{}: .sp with an AC current source of nonzero imaginary phasor (span.c copies it \
             into every port excitation); remove its ac phase or the source",
            source.name
        )));
    }
    Ok(())
}

/// RFSPICE independent-generator covariance scratch. The transpose
/// solutions are kept in one contiguous port-major block.
struct NoiseWorkspace {
    adjoints: Vec<Complex>,
    voltage: Vec<Complex>,
    current: Vec<Complex>,
    cy: DenseComplex,
}

impl NoiseWorkspace {
    fn new(ports: usize, unknowns: usize) -> SpiceResult<Self> {
        let buffer = |length| filled_vec(length, Complex::ZERO, ".sp noise workspace");
        Ok(Self {
            adjoints: buffer(product(&[ports, unknowns])?)?,
            voltage: buffer(ports)?,
            current: buffer(ports)?,
            cy: DenseComplex::zeros(ports)?,
        })
    }

    fn evaluate(
        &mut self,
        circuit: &Circuit,
        ports: &[BoundPort],
        factor: &ComplexLu,
        y: &DenseComplex,
        generators: &[InstanceNoise],
        solve: &mut SolveBuffers,
    ) -> SpiceResult<()> {
        let count = ports.len();
        let n = circuit.unknown_count();
        // Preserve original assignment order even for coincident terminals.
        for (port, adjoint) in ports.iter().zip(self.adjoints.chunks_exact_mut(n)) {
            solve.rhs.fill(Complex::ZERO);
            if let Some(row) = port.positive {
                solve.rhs[row] = Complex::real(1.);
            }
            if let Some(row) = port.negative {
                solve.rhs[row] = Complex::real(-1.);
            }
            solve.solve(factor, true)?;
            adjoint.copy_from_slice(&solve.solution);
        }
        self.cy.clear();
        for instance in generators {
            for source in &instance.sources {
                let Some(amplitude) = source.kind.rf_amplitude()? else {
                    continue;
                };
                // Bind the two node rows once per generator, not per port.
                let rows = source.nodes.map(|node| circuit.unknowns().node_row(node));
                for (voltage, adjoint) in self.voltage.iter_mut().zip(self.adjoints.chunks_exact(n))
                {
                    let at = |row: Option<usize>| row.map_or(Complex::ZERO, |r| adjoint[r]);
                    *voltage = (at(rows[0]) - at(rows[1])) * Complex::real(amplitude);
                }
                for (i, current) in self.current.iter_mut().enumerate() {
                    *current = self.voltage[i] * ports[i].inverse_z0;
                    for (j, v) in self.voltage.iter().enumerate() {
                        *current = *current + y.get(i, j).unwrap_or(Complex::ZERO) * *v;
                    }
                }
                for i in 0..count {
                    for j in 0..count {
                        self.cy.set(
                            i,
                            j,
                            self.cy.get(i, j).unwrap_or(Complex::ZERO)
                                + self.current[i] * self.current[j].conj(),
                        )?;
                    }
                }
            }
        }
        Ok(())
    }

    fn append_results(
        &self,
        ports: &[BoundPort],
        y: &DenseComplex,
        context: &crate::analysis::AnalysisContext,
        point: &mut Vec<Complex>,
    ) -> SpiceResult<()> {
        use crate::devices::noise::BOLTZMANN;
        let count = ports.len();
        let cy = &self.cy;
        let start = point.len();
        point.extend_from_slice(cy.data());
        if count == 2 {
            let norm = Complex::real(4. * BOLTZMANN * (context.temperature + 273.15));
            let get = |i, j| cy.get(i, j).unwrap_or(Complex::ZERO) / norm;
            let y11 = y.get(0, 0).unwrap_or(Complex::ZERO);
            let y21 = y.get(1, 0).unwrap_or(Complex::ZERO);
            let mut rn = get(1, 1).re / y21.magnitude().powi(2);
            if rn.abs() < 1e-30 {
                rn = 1e-30;
            }
            let mut c22 = get(1, 1);
            if c22.re.abs() < 1e-30 && c22.im.abs() < 1e-30 {
                c22.re = 1e-30;
            }
            let ycor = y11 - (get(0, 1) / c22) * y21;
            let correlated_power = rn * (y11 - ycor).magnitude().powi(2);
            let gu = get(0, 0).re - correlated_power;
            let ycor_re_squared = ycor.re * ycor.re;
            let ys = Complex::new((ycor_re_squared + gu / rn).max(0.).sqrt(), -ycor.im);
            let y0 = ports[0].inverse_z0;
            let sopt = (y0 - ys) / (y0 + ys);
            let excess_noise = 2. * rn * (ycor.re + ys.re);
            let fmin = 1. + excess_noise;
            let mismatch_noise = rn / y0.re * (y0 - ys).magnitude().powi(2);
            let nf = fmin + mismatch_noise;
            point.extend([
                Complex::real(10. * nf.log10()),
                sopt,
                Complex::real(10. * fmin.log10()),
                Complex::real(rn),
            ]);
        }
        if point[start..].iter().any(|v| !v.is_finite()) {
            return Err(SpiceError::Numerical { context: "S-parameter noise".into(), message: "nonfinite covariance or undefined two-port noise parameters (check Y21 and port coupling)".into() });
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const UNLIMITED: RfLimits = RfLimits {
        work_units: usize::MAX,
        numeric_bytes: usize::MAX,
        variables: usize::MAX,
    };

    #[test]
    fn budget_preflight_checks_overflow_and_aggregate_work_before_allocation() {
        let check =
            |p, n, o, f, noise, limits| RfBudget::preflight(p, n, o, f, noise, false, limits);
        for (p, n, o, f) in [
            (usize::MAX, 2, 0, 1),
            (2, usize::MAX, 0, 1),
            (2, 2, usize::MAX, 1),
            (2, 2, 0, usize::MAX),
        ] {
            assert!(check(p, n, o, f, true, UNLIMITED).is_err());
        }
        assert!(check(0, 1, 0, 1, false, UNLIMITED).is_err());
        assert!(check(32, 96, 0, 100_000, true, RfLimits::default()).is_err());
        assert!(check(8, 24, 0, 64, true, RfLimits::default()).is_ok());
        assert!(check(32, 96, 0, 64, true, RfLimits::default()).is_ok());
        let without_noise = check(2, 6, 0, 3, false, UNLIMITED).unwrap();
        let with_noise = check(2, 6, 0, 3, true, UNLIMITED).unwrap();
        assert_eq!(without_noise.width, 20);
        assert_eq!(with_noise.width, 28);
        assert!(with_noise.remaining < without_noise.remaining);
        let work = usize::MAX - with_noise.remaining;
        let limits = RfLimits {
            work_units: work,
            ..UNLIMITED
        };
        let mut exact = check(2, 6, 0, 3, true, limits).unwrap();
        assert!(exact.charge(0).is_ok());
        assert!(exact.charge(1).is_err());
        assert!(
            check(
                2,
                6,
                0,
                3,
                true,
                RfLimits {
                    work_units: work - 1,
                    ..UNLIMITED
                }
            )
            .is_err()
        );
        assert!(
            check(
                2,
                6,
                0,
                3,
                true,
                RfLimits {
                    variables: 27,
                    ..UNLIMITED
                }
            )
            .is_err()
        );

        // Exact numeric bytes including retained bias, extra observations,
        // ports, row, thirteen algebra matrices and covariance, RHS/work/
        // result, two adjacent adjoints and noise scratch.
        let cells = 3 * 29 + 7 + 29 + 14 * 4 + 2 * 6 + 3 * 6 + 1 + 1 + 8 * 2;
        let bytes = cells * std::mem::size_of::<Complex>();
        assert!(
            RfBudget::preflight(
                2,
                6,
                1,
                3,
                true,
                true,
                RfLimits {
                    numeric_bytes: bytes - 1,
                    ..UNLIMITED
                }
            )
            .is_err()
        );
        assert!(
            RfBudget::preflight(
                2,
                6,
                1,
                3,
                true,
                true,
                RfLimits {
                    numeric_bytes: bytes,
                    ..UNLIMITED
                }
            )
            .is_ok()
        );
    }

    #[test]
    fn work_budget_charges_actual_factor_entries_and_noise_sources_transactionally() {
        let mut budget = RfBudget {
            remaining: 144,
            width: 0,
        };
        budget.charge_factor(10, 2, true).unwrap(); // 80
        assert_eq!(budget.remaining, 64);
        budget.charge_noise(2, 5).unwrap(); // 60
        assert_eq!(budget.remaining, 4);
        assert!(budget.charge_noise(2, 1).is_err());
        assert_eq!(budget.remaining, 4);
        assert!(budget.charge_factor(usize::MAX, 2, true).is_err());
        assert!(budget.charge_noise(usize::MAX, 1).is_err());
        assert_eq!(budget.remaining, 4);
    }

    #[test]
    fn conversions_clear_previous_results_when_a_later_frequency_is_singular() {
        let ports = [BoundPort {
            number: 1,
            z0: Complex::real(50.),
            ki: Complex::real(0.5 / 50_f64.sqrt()),
            inverse_z0: Complex::real(1. / 50.),
            positive: Some(0),
            negative: None,
            branch: 1,
        }];
        let mut algebra = PortAlgebra::new(&ports).unwrap();
        algebra.incident.set(0, 0, Complex::real(1.)).unwrap();
        algebra.scattered.set(0, 0, Complex::real(0.25)).unwrap();
        algebra.calculate(1.).unwrap();
        assert_ne!(algebra.z.data(), &[Complex::ZERO]);
        assert_ne!(algebra.y.data(), &[Complex::ZERO]);
        algebra.scattered.set(0, 0, Complex::real(1.)).unwrap();
        algebra.calculate(2.).unwrap();
        assert_eq!(algebra.z.data(), &[Complex::ZERO]);
        algebra.scattered.set(0, 0, Complex::real(-1.)).unwrap();
        algebra.calculate(3.).unwrap();
        assert_eq!(algebra.y.data(), &[Complex::ZERO]);
        algebra.incident.clear();
        assert!(algebra.calculate(4.).is_err());
        algebra.incident.set(0, 0, Complex::real(1.)).unwrap();
        algebra.scattered.set(0, 0, Complex::real(0.25)).unwrap();
        algebra.calculate(5.).unwrap();
        assert_ne!(algebra.z.data(), &[Complex::ZERO]);
        assert_ne!(algebra.y.data(), &[Complex::ZERO]);
    }
}
