//! Behavioural (arbitrary) sources: `Bname n+ n- v=expr | i=expr`, plus the
//! XSPICE `pwl`/`spice2poly` instances the front end generates for E/G/F/H
//! `TABLE` and `POLY` forms ([`crate::netlist::behavioural`]).
//!
//! C references (behaviour only): `src/spicelib/devices/asrc/` (`asrcset.c`,
//! `asrcload.c`, `asrcacld.c`, `asrcpar.c`, `asrctemp.c`, `asrcfbr.c`) and the
//! expression machinery of `src/spicelib/parser/` ([`program`]).
//!
//! # Equations
//!
//! With `f(x)` the expression over its quantities `x` (node voltages and
//! branch currents), `g = df/dx` at the present iterate `x0` and the
//! instance factor `k` (below), a load linearises `f` like `ASRCload`:
//!
//! | Output | Stamps (`A x = b`, ground rows/columns dropped) |
//! | --- | --- |
//! | `v=` (branch row `r`) | `A[n+][r] += 1`, `A[n-][r] -= 1`, `A[r][n+] += 1`, `A[r][n-] -= 1`, `A[r][x_i] -= k g_i`, `b[r] += k (f - g.x0)` |
//! | `i=` | `A[n+][x_i] += k g_i`, `A[n-][x_i] -= k g_i`, `b[n+] -= k (f - g.x0)`, `b[n-] += k (f - g.x0)` |
//!
//! so a voltage source enforces `v(n+) - v(n-) = k f(x)` and a current source
//! drives `k f(x)` from `n+` through itself to `n-` (the signs of V and I
//! sources). AC uses the same Jacobian at the bias point and no RHS
//! (`ASRCacLoad`). The factor is `k = m (1 + tc1 d + tc2 d^2)` with
//! `d = T + dtemp - 300.15 K` (`T` the instance `temp=` or the circuit
//! temperature); `reciproctc=1` inverts the temperature factor and
//! `reciprocm=1` divides by `m`.
//!
//! `time` is the transient time (0 in OP/DC and in the AC linearisation),
//! `temper` the circuit temperature and `hertz` the AC frequency (0 outside
//! AC; an AC analysis of a circuit using it re-solves the operating point at
//! every frequency, as `acan.c` does for `CKTvarHertz`). ngspice sets no breakpoints for B
//! sources, and neither does the port: a time-dependent expression is
//! followed by the ordinary truncation-error step control.
//!
//! A voltage B source's branch current may be sensed by `i(bname)` (C
//! `ASRCfindBr`). C would also invent an unstamped branch for a current B
//! source; the port reports that reference as not findable instead.

pub(super) mod dynamic;
pub mod program;
pub mod rlc;
pub mod xspice;

use crate::maths::{SparseMatrix, Vector};
use crate::netlist::ast::{DeviceInstance, ParameterKind};
use crate::netlist::bexpr::BehaviouralExpression;
use crate::primitives::{
    NodeAppend, NodeId, Real, SourceLoc, SpiceError, SpiceResult, parse_spice_number,
};

use crate::devices::linear::LinearContext;
use crate::devices::traits::{
    AnalysisMode, ControlReference, Device, MnaUnknowns, StampContext, StampRhs,
};
use program::{Environment, Evaluation, EvaluationView, Program, Quantity, Workspace};
use std::cell::RefCell;

/// C reference used in diagnostics.
pub const C_REFERENCE: &str = "src/spicelib/devices/asrc/ (asrcload.c, asrcacld.c, asrcpar.c), \
     src/spicelib/parser/inpptree.c, ptfuncs.c";

/// Whether the source imposes a voltage or drives a current.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BehaviouralOutput {
    /// `v=`: a voltage source with a branch-current unknown.
    Voltage,
    /// `i=`: a current source.
    Current,
}

/// The instance setters of `ASRCpTable` that scale the output.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BehaviouralScale {
    /// Multiplier `m` (default 1).
    pub m: Real,
    /// First-order temperature coefficient (1/K).
    pub tc1: Real,
    /// Second-order temperature coefficient (1/K^2).
    pub tc2: Real,
    /// Instance temperature in degrees Celsius, if given.
    pub temperature: Option<Real>,
    /// Temperature offset from the circuit (K), used without `temp=`.
    pub dtemp: Real,
    /// `reciproctc=1`: divide by the temperature factor instead.
    pub reciprocal_tc: bool,
    /// `reciprocm=1`: divide by `m` instead.
    pub reciprocal_m: bool,
}

impl Default for BehaviouralScale {
    fn default() -> Self {
        Self {
            m: 1.,
            tc1: 0.,
            tc2: 0.,
            temperature: None,
            dtemp: 0.,
            reciprocal_tc: false,
            reciprocal_m: false,
        }
    }
}

impl BehaviouralScale {
    /// The output factor at circuit temperature `circuit` (degrees Celsius),
    /// as `ASRCload` computes it.
    #[must_use = "the returned temperature factor scales the behavioral output"]
    pub fn factor(&self, circuit: Real) -> Real {
        let kelvin = self.temperature.unwrap_or(circuit) + 273.15;
        let difference = kelvin + self.dtemp - 300.15;
        let linear_temperature = self.tc1 * difference;
        let quadratic_temperature = self.tc2 * difference * difference;
        let mut factor = 1. + linear_temperature + quadratic_temperature;
        if self.reciprocal_tc {
            factor = 1. / factor;
        }
        if self.reciprocal_m {
            factor / self.m
        } else {
            factor * self.m
        }
    }
}

/// One variable of the compiled expression, bound to the circuit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Variable {
    Node(NodeId),
    /// Index into [`Device::controlling_sources`].
    Branch(usize),
}

/// Evaluation and the circuit bindings it was linearised around, in
/// expression-variable order. Ground quantities have no matrix column.
struct Linearization<'a> {
    evaluation: EvaluationView<'a>,
    values: &'a [Real],
    columns: &'a [Option<usize>],
}

impl Linearization<'_> {
    fn equivalent(&self, factor: Real) -> Real {
        let mut equivalent = self.evaluation.value;
        for (gradient, value) in self.evaluation.gradient.iter().zip(self.values) {
            equivalent -= value * gradient;
        }
        factor * equivalent
    }
}

/// Per-instance mutable scratch, never shared when a device is cloned.
/// The compiled plan and circuit bindings remain immutable and shareable.
#[derive(Debug, Default)]
struct Scratch {
    program: Workspace,
    values: Vec<Real>,
    columns: Vec<Option<usize>>,
}

impl Clone for Scratch {
    fn clone(&self) -> Self {
        Self::default()
    }
}

/// A behavioural source. See the [module documentation](self).
#[derive(Debug, Clone)]
pub struct Behavioural {
    name: String,
    designator: char,
    output: BehaviouralOutput,
    /// `[n+, n-]` followed by every other node the expression reads.
    terminals: Vec<NodeId>,
    variables: Vec<Variable>,
    controls: Vec<ControlReference>,
    program: Program,
    workspace: RefCell<Scratch>,
    scale: BehaviouralScale,
}

impl Behavioural {
    /// Which output this source has.
    #[must_use = "the result identifies the physical quantity driven by this source"]
    pub const fn output(&self) -> BehaviouralOutput {
        self.output
    }

    /// The output scaling setters.
    #[must_use = "the returned setters determine output scaling and sensitivity"]
    pub const fn scale(&self) -> &BehaviouralScale {
        &self.scale
    }

    /// A copy with other output scaling setters, for a `.sens` load.
    #[must_use = "the returned copy carries the replacement scale; the original is unchanged"]
    pub(crate) fn with_scale(&self, scale: BehaviouralScale) -> Self {
        Self {
            scale,
            ..self.clone()
        }
    }

    /// The compiled expression.
    #[must_use = "the returned program contains the bound expression evaluated by the source"]
    pub const fn program(&self) -> &Program {
        &self.program
    }

    /// Evaluate in the reusable instance scratch, borrowing all input,
    /// matrix-column and derivative buffers for the duration of the caller.
    /// Resolving columns for each assembly is necessary: MNA numbering can
    /// change after an analysis rebuild and no topology generation token is
    /// currently exposed by `MnaUnknowns`. No column allocations occur after
    /// the first load with a particular expression width.
    fn with_linearization<T>(
        &self,
        solution: &Vector,
        unknowns: &MnaUnknowns,
        controls: &[usize],
        time: Real,
        context: &crate::devices::ModelContext,
        use_result: impl FnOnce(&Linearization<'_>) -> SpiceResult<T>,
    ) -> SpiceResult<T> {
        let mut scratch = self.workspace.borrow_mut();
        scratch.columns.clear();
        for variable in &self.variables {
            let column = match *variable {
                Variable::Node(node) => unknowns.node_row(node),
                Variable::Branch(index) => Some(controls.get(index).copied().ok_or_else(|| {
                    SpiceError::circuit(format!(
                        "{}: controlling branch {} is not bound",
                        self.name, self.controls[index].name
                    ))
                })?),
            };
            scratch.columns.push(column);
        }

        let Scratch {
            program,
            values,
            columns,
        } = &mut *scratch;
        values.clear();
        values.extend(
            columns
                .iter()
                .map(|column| column.and_then(|row| solution.get(row)).unwrap_or(0.)),
        );
        let environment = Environment {
            values,
            time,
            temperature: context.temperature,
            gmin: context.gmin,
            // ifeval.c PTeval: hertz is CKTomega / 2PI,
            // with CKTomega = 2PI * frequency.
            frequency: 2.0 * std::f64::consts::PI * context.frequency / 2. / std::f64::consts::PI,
        };
        let evaluation = self
            .program
            .evaluate_with_workspace(&environment, program)
            .map_err(|error| self.evaluation_error(error))?;
        use_result(&Linearization {
            evaluation,
            values,
            columns,
        })
    }

    fn evaluation_error(&self, error: SpiceError) -> SpiceError {
        match error {
            SpiceError::Numerical { message, .. } => SpiceError::Numerical {
                context: format!("behavioural source {}", self.name),
                message,
            },
            other => other,
        }
    }

    /// Cold validation helper; all hot circuit evaluation uses the borrowed
    /// workspace, including convergence's derivative/domain checks.
    fn evaluate_values(
        &self,
        values: &[Real],
        time: Real,
        context: &crate::devices::ModelContext,
    ) -> SpiceResult<Evaluation> {
        self.program
            .evaluate(&Environment {
                values,
                time,
                temperature: context.temperature,
                gmin: context.gmin,
                frequency: 2.0 * std::f64::consts::PI * context.frequency
                    / 2.
                    / std::f64::consts::PI,
            })
            .map_err(|error| self.evaluation_error(error))
    }

    /// Stamps the linearised output. `rhs` is `None` for the AC Jacobian.
    fn stamp_into(
        &self,
        matrix: &mut SparseMatrix,
        rhs: Option<&mut StampRhs<'_>>,
        unknowns: &MnaUnknowns,
        branch: Option<usize>,
        linearization: &Linearization<'_>,
        factor: Real,
    ) -> SpiceResult<()> {
        let row = |node: NodeId| unknowns.node_row(node);
        let (p, n) = (row(self.terminals[0]), row(self.terminals[1]));
        let mut add = |r: Option<usize>, c: Option<usize>, value: Real| match (r, c) {
            (Some(r), Some(c)) => matrix.add(r, c, value),
            _ => Ok(()),
        };
        let equivalent = linearization.equivalent(factor);
        match self.output {
            BehaviouralOutput::Voltage => {
                let k = Some(branch.ok_or_else(|| {
                    SpiceError::circuit(format!("{}: missing branch row", self.name))
                })?);
                add(p, k, 1.)?;
                add(n, k, -1.)?;
                add(k, n, -1.)?;
                add(k, p, 1.)?;
                for (gradient, column) in linearization
                    .evaluation
                    .gradient
                    .iter()
                    .zip(linearization.columns)
                {
                    add(k, *column, -gradient * factor)?;
                }
                if let (Some(rhs), Some(k)) = (rhs, k) {
                    rhs.add_to(k, equivalent)?;
                }
            }
            BehaviouralOutput::Current => {
                for (gradient, column) in linearization
                    .evaluation
                    .gradient
                    .iter()
                    .zip(linearization.columns)
                {
                    add(p, *column, gradient * factor)?;
                    add(n, *column, -gradient * factor)?;
                }
                if let Some(rhs) = rhs {
                    if let Some(p) = p {
                        rhs.add_to(p, -equivalent)?;
                    }
                    if let Some(n) = n {
                        rhs.add_to(n, equivalent)?;
                    }
                }
            }
        }
        Ok(())
    }

    fn checked_factor(&self, temperature: Real) -> SpiceResult<Real> {
        let factor = self.scale.factor(temperature);
        if factor.is_finite() {
            Ok(factor)
        } else {
            Err(SpiceError::Numerical {
                context: format!("behavioural source {}", self.name),
                message: format!("non-finite output factor (m/tc1/tc2) {factor}"),
            })
        }
    }
}

impl Device for Behavioural {
    fn initial_voltage_relations(
        &self,
        _inductors_short: bool,
    ) -> Vec<crate::devices::InitialVoltageRelation> {
        (self.output == BehaviouralOutput::Voltage)
            .then(|| crate::devices::InitialVoltageRelation {
                terminals: [self.terminals[0], self.terminals[1]],
                law: crate::devices::InitialVoltageLaw::Controlled,
            })
            .into_iter()
            .collect()
    }

    fn validate_dc_inputs(&self, context: &crate::devices::ModelContext) -> SpiceResult<()> {
        let factor = self.checked_factor(context.temperature)?;
        // An expression depending only on the environment and ground has a
        // known value independent of any Newton iterate. Do not evaluate
        // expressions reading real nodes or branches at an arbitrary bias.
        if self
            .variables
            .iter()
            .all(|variable| matches!(variable, Variable::Node(node) if *node == NodeId::GROUND))
        {
            let values = vec![0.; self.variables.len()];
            let evaluation = self.evaluate_values(&values, 0., context)?;
            if !(factor * evaluation.value).is_finite() {
                return Err(SpiceError::circuit(format!(
                    "{}: nonfinite independent behavioural DC forcing",
                    self.name
                )));
            }
        }
        Ok(())
    }

    /// `asrc`: `inpcom.c` rewrites the E/G/F/H VALUE/TABLE/POLY forms into
    /// B sources before parsing.
    fn c_device_type(&self) -> crate::devices::setup::CDeviceType {
        crate::devices::setup::CDeviceType::Asrc
    }

    /// One trial slot: the expression value of the last load (C
    /// `ASRCprev_value`).
    fn state_count(&self) -> usize {
        1
    }

    /// `ASRCconvTest` (`asrcconv.c`): the expression at the new iterate
    /// against its value at the last load, within `reltol` and `vntol` (a
    /// voltage source) or `abstol` (a current source).
    fn convergence_test(
        &self,
        context: &crate::devices::ConvergenceContext<'_>,
    ) -> SpiceResult<bool> {
        self.with_linearization(
            context.solution,
            context.unknowns,
            context.controls,
            context.time,
            context.model_context,
            |linearization| {
                let rhs = linearization.evaluation.value;
                let prev = context.states.first().copied().unwrap_or(0.);
                let absolute = match self.output {
                    BehaviouralOutput::Voltage => context.vntol,
                    BehaviouralOutput::Current => context.abstol,
                };
                let relative = context.reltol * rhs.abs().max(prev.abs());
                let tol = relative + absolute;
                Ok((prev - rhs).abs() > tol)
            },
        )
    }

    /// `asrcset.c`: a voltage source's branch elements, then one column per
    /// expression variable (the branch row, or both output nodes).
    fn setup_matrix(
        &self,
        setup: &mut crate::devices::setup::MatrixSetup<'_>,
    ) -> SpiceResult<bool> {
        let (pos, neg) = (setup.node(self.terminals[0]), setup.node(self.terminals[1]));
        let branch = match self.output {
            BehaviouralOutput::Voltage => {
                let branch = Some(setup.branch(0)?);
                setup.element(pos, branch);
                setup.element(neg, branch);
                setup.element(branch, neg);
                setup.element(branch, pos);
                branch
            }
            BehaviouralOutput::Current => None,
        };
        for variable in &self.variables {
            let column = match *variable {
                Variable::Node(node) => setup.node(node),
                Variable::Branch(index) => {
                    Some(setup.controls.get(index).copied().ok_or_else(|| {
                        SpiceError::circuit(format!("{}: unbound controlling source", self.name))
                    })?)
                }
            };
            match self.output {
                BehaviouralOutput::Voltage => setup.element(branch, column),
                BehaviouralOutput::Current => {
                    setup.element(pos, column);
                    setup.element(neg, column);
                }
            }
        }
        Ok(true)
    }
    fn instance_parameter(&self, keyword: &str) -> Option<&'static str> {
        ["temp", "dtemp", "tc1", "tc2", "m"]
            .into_iter()
            .find(|k| k.eq_ignore_ascii_case(keyword))
    }
    fn with_instance_parameter(
        &self,
        parameter: &str,
        value: Real,
        context: &crate::devices::ModelContext,
    ) -> SpiceResult<Box<dyn Device>> {
        crate::devices::sweep::check_swept(
            &self.name,
            parameter,
            value,
            parameter != "temp" || value > -273.15,
            "finite and above absolute zero for temp",
        )?;
        let mut scale = self.scale;
        match parameter {
            "temp" => scale.temperature = Some(value),
            "dtemp" => scale.dtemp = value,
            "tc1" => scale.tc1 = value,
            "tc2" => scale.tc2 = value,
            "m" => scale.m = value,
            _ => return Err(SpiceError::circuit("unknown behavioural sweep setter")),
        }
        let copy = self.with_scale(scale);
        copy.checked_factor(context.temperature)?;
        Ok(Box::new(copy))
    }

    /// `.sens`: a B source's `ASRCpTable` setters
    /// ([`crate::devices::sensitivity`]). The XSPICE `spice2poly`/`pwl` code
    /// models TABLE/POLY lower to are refused: their `MIF` parameters are not
    /// ported.
    fn sensitivity(
        &self,
        context: &crate::devices::models::ModelContext,
    ) -> SpiceResult<Box<dyn crate::devices::sensitivity::DeviceSensitivity + '_>> {
        if self.designator != 'b' {
            return Err(crate::devices::sensitivity::not_ported(
                &self.name,
                self.designator,
                " (an XSPICE code model)",
            ));
        }
        Ok(Box::new(
            crate::devices::sensitivity::BehaviouralSensitivity::new(self, context),
        ))
    }

    /// Noiseless: C gives B sources no noise routine (`DEVnoise = NULL`,
    /// `src/spicelib/devices/asrc/asrcinit.c`), and the XSPICE `spice2poly`/
    /// `pwl` code models that TABLE/POLY lower to declare none of the noise
    /// parameters `MIFnoise` looks for (`src/xspice/mif/mifnoise.c`).
    fn noise(
        &self,
        _context: &crate::devices::noise::NoiseContext<'_>,
        _sources: &mut Vec<crate::devices::noise::NoiseSource>,
    ) -> crate::primitives::SpiceResult<crate::devices::noise::DeviceNoise<'_>> {
        Ok(crate::devices::noise::DeviceNoise::Noiseless)
    }

    /// Refused in `.disto`: neither C's B source (`asrcinit.c`) nor the XSPICE
    /// code models that `POLY`/`TABLE` lower to have a distortion routine, so
    /// C silently keeps only their small-signal linearization and drops their
    /// nonlinearity. The port does not reproduce that partial result.
    fn distortion(
        &self,
        _context: &crate::devices::distortion::DistortionContext<'_>,
    ) -> crate::primitives::SpiceResult<crate::devices::distortion::DeviceDistortion> {
        Err(SpiceError::Unsupported {
            feature: format!(
                "distortion analysis of {} ({}): C has no distortion routine for behavioural                  sources or XSPICE code models and would silently ignore their nonlinearity",
                self.name,
                if self.designator == 'a' {
                    "XSPICE POLY/TABLE code model"
                } else {
                    "B source"
                }
            ),
            location: None,
        })
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn designator(&self) -> char {
        self.designator
    }

    fn terminals(&self) -> &[NodeId] {
        &self.terminals
    }

    fn branch_currents(&self) -> usize {
        usize::from(self.output == BehaviouralOutput::Voltage)
    }

    fn controlling_sources(&self) -> &[ControlReference] {
        &self.controls
    }

    /// `ASRCfindBr`: a voltage B source's current can be sensed.
    fn findable_branch(&self) -> Option<usize> {
        (self.output == BehaviouralOutput::Voltage).then_some(0)
    }

    fn is_nonlinear(&self) -> bool {
        true
    }

    /// `ASRCload` applies no limiting.
    fn limits_voltage_steps(&self) -> bool {
        false
    }

    /// `hertz` makes the DC equations frequency dependent (C `CKTvarHertz`).
    fn depends_on_frequency(&self) -> bool {
        self.program.uses_frequency()
    }

    fn depends_on_time(&self) -> bool {
        self.program.uses_time()
    }

    fn stamp(&self, context: &mut StampContext<'_>) -> SpiceResult<()> {
        let time = match context.mode {
            AnalysisMode::Transient { time, .. } => time,
            AnalysisMode::OperatingPoint | AnalysisMode::DcSweep => 0.,
            AnalysisMode::Ac { .. } => {
                return Err(SpiceError::circuit(format!(
                    "{}: behavioural AC needs small-signal assembly",
                    self.name
                )));
            }
        };
        let factor = self.checked_factor(context.temperature)?;
        self.with_linearization(
            context.solution,
            context.unknowns,
            context.controls,
            time,
            &context.model_context(),
            |linearization| {
                // ASRCprev_value remains the raw, unscaled expression value.
                context.states.set(0, linearization.evaluation.value)?;
                let branch = (!context.branches.is_empty()).then_some(context.branches.start);
                self.stamp_into(
                    context.matrix,
                    Some(&mut context.rhs),
                    context.unknowns,
                    branch,
                    linearization,
                    factor,
                )
            },
        )
    }

    // The retained ASRCprev_value is an algebraic observation, not a charge.
    fn assemble_nonlinear_charge(
        &self,
        _context: &mut crate::devices::traits::ChargeContext<'_>,
    ) -> SpiceResult<()> {
        Ok(())
    }

    /// The bias-point Jacobian (`ASRCacLoad`): real, frequency independent.
    fn assemble_small_signal(
        &self,
        context: &mut LinearContext<'_>,
        bias: &Vector,
    ) -> SpiceResult<()> {
        let temperature = context.model_context.temperature;
        let factor = self.checked_factor(temperature)?;
        self.with_linearization(
            bias,
            context.unknowns,
            context.controls,
            0.,
            context.model_context,
            |linearization| {
                self.stamp_into(
                    &mut context.system.a,
                    None,
                    context.unknowns,
                    context.branch,
                    linearization,
                    factor,
                )
            },
        )
    }

    /// `ASRCpzLoad` (`asrcpzld.c`) stamps the same bias-point Jacobian as
    /// AC. The XSPICE instances the front end generates for `POLY`/`TABLE`
    /// (designator `a`) have no pole-zero load in C (`DEVpzLoad = NULL`, so
    /// C silently drops them) and a `hertz` expression has no defined
    /// pole-zero frequency: both are refused.
    fn assemble_pole_zero(
        &self,
        context: &mut LinearContext<'_>,
        bias: &Vector,
    ) -> SpiceResult<()> {
        if self.designator == 'a' {
            return Err(SpiceError::Unsupported {
                feature: format!(
                    "pole-zero analysis of XSPICE code-model instance {} (POLY/TABLE): C has no \
                     pole-zero load for code models and would drop it silently",
                    self.name
                ),
                location: None,
            });
        }
        if self.depends_on_frequency() {
            return Err(SpiceError::Unsupported {
                feature: format!(
                    "pole-zero analysis of {}: its expression reads `hertz`",
                    self.name
                ),
                location: None,
            });
        }
        self.assemble_small_signal(context, bias)
    }
}

fn unsupported(location: &SourceLoc, feature: impl Into<String>) -> SpiceError {
    SpiceError::Unsupported {
        feature: feature.into(),
        location: Some(location.clone()),
    }
}

fn literal(text: &str, location: &SourceLoc, what: &str) -> SpiceResult<Real> {
    parse_spice_number(text)
        .filter(|value| value.is_finite())
        .ok_or_else(|| unsupported(location, format!("non-finite or nonliteral {what}={text}")))
}

fn parse_parameters(
    instance: &DeviceInstance,
) -> SpiceResult<(BehaviouralOutput, &BehaviouralExpression, BehaviouralScale)> {
    let location = &instance.location;
    let mut expression: Option<(BehaviouralOutput, &BehaviouralExpression)> = None;
    let mut scale = BehaviouralScale::default();
    let mut temperature_given = false;
    let mut dtemp_given = false;
    for parameter in &instance.parameters {
        let at = &parameter.location;
        match (&parameter.kind, parameter.name.as_str()) {
            (ParameterKind::Behavioural(tree), "v" | "i") => {
                if expression.is_some() {
                    return Err(unsupported(
                        at,
                        format!("{}: more than one v=/i= expression", instance.name),
                    ));
                }
                let output = if parameter.name == "v" {
                    BehaviouralOutput::Voltage
                } else {
                    BehaviouralOutput::Current
                };
                expression = Some((output, tree));
            }
            (ParameterKind::Scalar, name) => {
                let value = literal(&parameter.value, at, name)?;
                match name {
                    "m" => scale.m = value,
                    "tc1" => scale.tc1 = value,
                    "tc2" => scale.tc2 = value,
                    "temp" => {
                        scale.temperature = Some(value);
                        temperature_given = true;
                    }
                    "dtemp" => {
                        scale.dtemp = value;
                        dtemp_given = true;
                    }
                    // IF_INTEGER flags: only the value 1 selects the reciprocal.
                    "reciproctc" => {
                        scale.reciprocal_tc = value.round().to_bits() == 1_f64.to_bits();
                    }
                    "reciprocm" => scale.reciprocal_m = value.round().to_bits() == 1_f64.to_bits(),
                    other => {
                        return Err(unsupported(
                            at,
                            format!("{} parameter {other}", instance.name),
                        ));
                    }
                }
            }
            (ParameterKind::Expression(_), name) => {
                return Err(unsupported(
                    at,
                    format!(
                        "non-literal {} parameter {name}={} (expressions must be literalized first)",
                        instance.name, parameter.value
                    ),
                ));
            }
            (_, name) => {
                return Err(unsupported(
                    at,
                    format!("{} parameter {name}", instance.name),
                ));
            }
        }
    }
    if temperature_given && dtemp_given {
        // ASRCtemp ignores dtemp (with a message) when temp is given.
        return Err(unsupported(
            location,
            format!(
                "{}: both temp= and dtemp= (C ignores dtemp with a message); give one",
                instance.name
            ),
        ));
    }
    let Some((output, tree)) = expression else {
        return Err(SpiceError::parse(
            location.clone(),
            format!(
                "{}: a B source needs v=expression or i=expression",
                instance.name
            ),
        ));
    };
    Ok((output, tree, scale))
}

/// Builds a B source (or a front-end generated XSPICE `a` instance) from its
/// literalized AST instance, binding nodes in `nodes` only on success.
pub(crate) fn instantiate(
    instance: &DeviceInstance,
    nodes: &mut NodeAppend<'_>,
) -> SpiceResult<Box<dyn Device>> {
    let location = &instance.location;
    if instance.model.is_some() || instance.nodes.len() != 2 {
        return Err(unsupported(
            location,
            format!("{} needs two nodes and no model", instance.name),
        ));
    }
    let (output, tree, scale) = parse_parameters(instance)?;
    Ok(Box::new(bind_expression(
        instance, nodes, output, tree, scale,
    )?))
}

fn bind_expression(
    instance: &DeviceInstance,
    nodes: &mut NodeAppend<'_>,
    output: BehaviouralOutput,
    tree: &BehaviouralExpression,
    scale: BehaviouralScale,
) -> SpiceResult<Behavioural> {
    let location = &instance.location;
    let program = Program::compile(&tree.root)?;
    nodes.transaction(|nodes| {
        let positive = nodes.intern(&instance.nodes[0])?;
        let negative = nodes.intern(&instance.nodes[1])?;
        if output == BehaviouralOutput::Voltage && positive == negative {
            // ASRCsetup: "instance %s is a shorted ASRC".
            return Err(unsupported(
                location,
                format!("instance {} is a shorted ASRC", instance.name),
            ));
        }
        let mut terminals = vec![positive, negative];
        let mut variables = Vec::with_capacity(program.quantities().len());
        let mut controls = Vec::new();
        for quantity in program.quantities() {
            match quantity {
                Quantity::Node(name) => {
                    let node = nodes.intern(name)?;
                    if !terminals.contains(&node) {
                        terminals.push(node);
                    }
                    variables.push(Variable::Node(node));
                }
                Quantity::Branch(name) => {
                    variables.push(Variable::Branch(controls.len()));
                    controls.push(ControlReference {
                        name: name.clone(),
                        location: Some(tree.span.start.clone()),
                    });
                }
            }
        }
        Ok(Behavioural {
            name: instance.name.clone(),
            designator: instance.designator,
            output,
            terminals,
            variables,
            controls,
            program,
            workspace: RefCell::default(),
            scale,
        })
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::netlist::{Parser, source::parse_deck_text};
    use std::path::Path;

    #[test]
    fn failed_behavioural_binding_preserves_existing_nodes() {
        let mut nodes = crate::primitives::NodeTable::new();
        nodes.intern("present").unwrap();
        let before = nodes.nodes().to_vec();
        let deck = Parser::new()
            .parse_deck(&parse_deck_text(
                Path::new("behavioural.cir"),
                "title\nb1 fresh fresh v=v(sense)\nb2 fresh 0 v=v(sense)+i(vx)\n.end\n",
            ))
            .unwrap();
        assert!(
            nodes
                .transaction(|nodes| instantiate(&deck.devices[0], nodes))
                .is_err()
        );
        assert_eq!(nodes.nodes(), before);
        let device = nodes
            .transaction(|nodes| instantiate(&deck.devices[1], nodes))
            .unwrap();
        assert_eq!(
            device.terminals(),
            &[NodeId::new(2), NodeId::GROUND, NodeId::new(3)]
        );
        assert_eq!(device.controlling_sources()[0].name, "vx");
        assert_eq!(nodes.get("sense"), Some(NodeId::new(3)));
        assert_eq!(nodes.get("vx"), None);
    }
}
