//! Behavioural passive constitutive equations with explicit storage semantics.
//!
//! For a physical passive the first branch unknown is current from positive
//! to negative. Generated derivative operators use their internal node row
//! directly, with a separate input storage coordinate but no dummy KCL
//! equation. Explicit charge and derivative laws separate the nonlinear
//! constitutive relation from the linear integration equation.
//! Differential C and L have a second algebraic unknown for the derivative
//! of terminal voltage or current. This retains an exact F+dQ/dt formulation
//! even when the coefficient depends on arbitrary controls.

use super::{
    Behavioural, BehaviouralOutput, BehaviouralScale, Linearization, bind_expression, dynamic,
    literal, unsupported,
};
use crate::devices::linear::LinearContext;
use crate::devices::noise::{
    BOLTZMANN, CELSIUS_TO_KELVIN, DeviceNoise, NoiseContext, NoiseCoordinate, NoiseFamily,
    NoiseSource,
};
use crate::devices::setup::MatrixSetup;
use crate::devices::traits::{
    AnalysisMode, ChargeContext, ControlReference, Device, MnaUnknowns, StampContext,
};
use crate::maths::{SparseMatrix, Vector};
use crate::netlist::ast::{DeviceInstance, ParameterKind};
use crate::primitives::{Complex, NodeAppend, NodeId, Real, SpiceError, SpiceResult};

/// Physical interpretation of the compiled scalar expression.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Law {
    Resistance,
    Capacitance,
    Charge,
    Inductance,
    Derivative,
}

impl Law {
    const fn differential(self) -> bool {
        matches!(self, Self::Capacitance | Self::Inductance)
    }

    const fn storage(self) -> bool {
        !matches!(self, Self::Resistance)
    }

    const fn branches(self) -> usize {
        match self {
            Self::Derivative | Self::Resistance => 1,
            Self::Capacitance | Self::Inductance | Self::Charge => 2,
        }
    }

    const fn explicit_storage(self) -> bool {
        matches!(self, Self::Charge | Self::Derivative)
    }

    /// Linear part and the variable (if any) multiplying the expression.
    /// All constitutive laws have the form L*x - expression*multiplier=0.
    const fn constitutive(self) -> ([Real; 4], Option<usize>) {
        match self {
            Self::Resistance => ([1., -1., 0., 0.], Some(2)),
            Self::Charge | Self::Derivative => ([0., 0., 0., 1.], None),
            Self::Capacitance => ([0., 0., 1., 0.], Some(3)),
            Self::Inductance => ([1., -1., 0., 0.], Some(3)),
        }
    }

    fn equations(self, values: [Real; 4], coefficient: Real) -> [Option<Equation>; 2] {
        let [positive, negative, current, auxiliary] = values;
        let voltage = positive - negative;
        let voltage_row = [1., -1., 0., 0.];
        let current_row = [0., 0., 1., 0.];
        let auxiliary_row = [0., 0., 0., 1.];
        let (mut direct, multiplier) = self.constitutive();
        let (expression, offset) = if let Some(index) = multiplier {
            direct[index] -= coefficient;
            (-values[index], 0.)
        } else {
            (-1., coefficient)
        };
        let constitutive = Algebraic {
            direct,
            expression,
            offset,
        };
        let evolution = match self {
            Self::Resistance => None,
            Self::Charge | Self::Derivative => Some(Equation {
                static_part: Algebraic::linear(if self == Self::Charge {
                    current_row
                } else {
                    voltage_row
                }),
                storage: Some(Stored {
                    value: -auxiliary,
                    direct: [0., 0., 0., -1.],
                }),
            }),
            Self::Capacitance => Some(Equation {
                static_part: Algebraic::linear(auxiliary_row),
                storage: Some(Stored {
                    value: -voltage,
                    direct: [-1., 1., 0., 0.],
                }),
            }),
            Self::Inductance => Some(Equation {
                static_part: Algebraic::linear(auxiliary_row),
                storage: Some(Stored {
                    value: -current,
                    direct: [0., 0., -1., 0.],
                }),
            }),
        };
        [
            Some(Equation {
                static_part: constitutive,
                storage: None,
            }),
            evolution,
        ]
    }
}

/// Direct coefficients, expression chain rule and analytical affine offset.
/// Linear terms cancel symbolically in J*x-F(x), not numerically.
#[derive(Debug, Clone, Copy)]
struct Algebraic {
    direct: [Real; 4],
    expression: Real,
    offset: Real,
}

impl Algebraic {
    const fn linear(direct: [Real; 4]) -> Self {
        Self {
            direct,
            expression: 0.,
            offset: 0.,
        }
    }
}

/// The evolved quantity is a homogeneous linear function of MNA
/// coordinates. Nonlinear storage is represented by a separate coordinate.
#[derive(Debug, Clone, Copy)]
struct Stored {
    value: Real,
    direct: [Real; 4],
}

#[derive(Debug, Clone, Copy)]
struct Equation {
    static_part: Algebraic,
    storage: Option<Stored>,
}

#[derive(Clone, Copy)]
struct BoundInput<'a> {
    solution: &'a Vector,
    unknowns: &'a MnaUnknowns,
    controls: &'a [usize],
    branch: usize,
    time: Real,
    model: &'a crate::devices::ModelContext,
}

struct Bound<'a, 'b> {
    columns: [Option<usize>; 4],
    values: [Real; 4],
    expression: &'a Linearization<'b>,
    factor: Real,
}

impl Bound<'_, '_> {
    fn direct_terms(&self, direct: [Real; 4]) -> impl Iterator<Item = (usize, Real, Real)> {
        self.columns
            .into_iter()
            .zip(self.values)
            .zip(direct)
            .filter_map(|((column, value), derivative)| {
                column
                    .filter(|_| derivative != 0.)
                    .map(|column| (column, value, derivative))
            })
    }

    fn expression_terms(&self, multiplier: Real) -> impl Iterator<Item = (usize, Real, Real)> {
        self.expression
            .columns
            .iter()
            .zip(self.expression.values)
            .zip(self.expression.evaluation.gradient)
            .filter_map(move |((&column, &value), &gradient)| {
                let derivative = multiplier * self.factor * gradient;
                column
                    .filter(|_| derivative != 0.)
                    .map(|column| (column, value, derivative))
            })
    }

    fn derivatives(&self, scalar: Algebraic) -> impl Iterator<Item = (usize, Real, Real)> {
        self.direct_terms(scalar.direct)
            .chain(self.expression_terms(scalar.expression))
    }

    fn matrix(&self, matrix: &mut SparseMatrix, row: usize, scalar: Algebraic) -> SpiceResult<()> {
        for (column, _, derivative) in self.derivatives(scalar) {
            matrix.add(row, column, derivative)?;
        }
        Ok(())
    }

    fn linear_matrix(
        &self,
        matrix: &mut SparseMatrix,
        row: usize,
        direct: [Real; 4],
        scale: Real,
    ) -> SpiceResult<()> {
        for (column, _, derivative) in self.direct_terms(direct) {
            matrix.add(row, column, derivative * scale)?;
        }
        Ok(())
    }

    fn affine(
        &self,
        matrix: &mut SparseMatrix,
        row: usize,
        scalar: Algebraic,
        multiplier: Real,
    ) -> SpiceResult<Real> {
        self.linear_matrix(matrix, row, scalar.direct, multiplier)?;
        let mut rhs = scalar.offset;
        for (column, value, derivative) in self.expression_terms(scalar.expression) {
            matrix.add(row, column, multiplier * derivative)?;
            rhs = derivative.mul_add(value, rhs);
        }
        Ok(rhs * multiplier)
    }
}

/// A compiled native behavioural passive. Reuses the ordinary behavioural
/// expression compiler, lexical bindings and derivative workspace.
#[derive(Debug, Clone)]
struct Passive {
    expression: Behavioural,
    law: Law,
    initial: Option<Real>,
    noisy: bool,
}

impl Passive {
    fn columns(&self, unknowns: &MnaUnknowns, branch: usize) -> [Option<usize>; 4] {
        [
            unknowns.node_row(self.expression.terminals[0]),
            unknowns.node_row(self.expression.terminals[1]),
            (self.law != Law::Derivative).then_some(branch),
            match self.law {
                Law::Derivative => Some(branch),
                Law::Resistance => None,
                _ => Some(branch + 1),
            },
        ]
    }

    fn with_bound<T>(
        &self,
        input: BoundInput<'_>,
        f: impl FnOnce(&Bound<'_, '_>, [Option<Equation>; 2]) -> SpiceResult<T>,
    ) -> SpiceResult<T> {
        let BoundInput {
            solution,
            unknowns,
            controls,
            branch,
            time,
            model,
        } = input;
        let factor = self.expression.checked_factor(model.temperature)?;
        let columns = self.columns(unknowns, branch);
        let values = columns.map(|column| column.and_then(|row| solution.get(row)).unwrap_or(0.));
        self.expression.with_linearization(
            solution,
            unknowns,
            controls,
            time,
            model,
            |expression| {
                let coefficient = factor * expression.evaluation.value;
                if !coefficient.is_finite() {
                    return Err(SpiceError::Numerical {
                        context: self.name().to_owned(),
                        message: "nonfinite constitutive coefficient".into(),
                    });
                }
                let bound = Bound {
                    columns,
                    values,
                    expression,
                    factor,
                };
                f(&bound, self.law.equations(values, coefficient))
            },
        )
    }

    fn rows(&self, unknowns: &MnaUnknowns, branch: Option<usize>) -> SpiceResult<[usize; 2]> {
        let branch = branch.ok_or_else(|| {
            SpiceError::circuit(format!("{}: missing constitutive branch", self.name()))
        })?;
        let primary = if self.law == Law::Derivative {
            unknowns
                .node_row(self.terminals()[0])
                .ok_or_else(|| SpiceError::circuit("missing derivative output row"))?
        } else {
            branch
        };
        let auxiliary = if self.law == Law::Derivative {
            branch
        } else {
            branch + 1
        };
        Ok([primary, auxiliary])
    }

    fn kcl(
        &self,
        matrix: &mut SparseMatrix,
        columns: [Option<usize>; 4],
        branch: usize,
    ) -> SpiceResult<()> {
        if self.law == Law::Derivative {
            return Ok(());
        }
        for (column, sign) in [(columns[0], 1.), (columns[1], -1.)] {
            if let Some(row) = column {
                matrix.add(row, branch, sign)?;
            }
        }
        Ok(())
    }
}

impl Device for Passive {
    fn name(&self) -> &str {
        self.expression.name()
    }
    fn designator(&self) -> char {
        self.expression.designator()
    }
    fn terminals(&self) -> &[NodeId] {
        self.expression.terminals()
    }
    fn control_branch(&self, target: &dyn Device) -> Option<usize> {
        target.expression_current_branch()
    }

    fn controlling_sources(&self) -> &[ControlReference] {
        self.expression.controlling_sources()
    }
    fn branch_currents(&self) -> usize {
        self.law.branches()
    }
    fn branch_quantity(&self, branch: usize) -> crate::devices::BranchQuantity {
        use crate::devices::BranchQuantity;
        match (self.law, branch) {
            (Law::Capacitance, 1) => BranchQuantity::VoltageRate,
            (Law::Inductance, 1) => BranchQuantity::CurrentRate,
            (Law::Charge, 1) => BranchQuantity::Charge,
            (Law::Derivative, _) => BranchQuantity::ExpressionState,
            _ => BranchQuantity::Current,
        }
    }
    fn branch_convergence(&self, branch: usize) -> crate::devices::ConvergenceKind {
        if self.law == Law::Derivative || (self.law.branches() == 2 && branch == 1) {
            crate::devices::ConvergenceKind::Constitutive
        } else {
            crate::devices::ConvergenceKind::Current
        }
    }
    fn findable_branch(&self) -> Option<usize> {
        (self.law != Law::Derivative).then_some(0)
    }
    fn is_nonlinear(&self) -> bool {
        true
    }
    fn limits_voltage_steps(&self) -> bool {
        false
    }
    fn depends_on_time(&self) -> bool {
        self.expression.depends_on_time()
    }
    fn depends_on_frequency(&self) -> bool {
        self.expression.depends_on_frequency()
    }
    fn state_count(&self) -> usize {
        if self.law.storage() { 2 } else { 0 }
    }
    fn truncation_slot(&self) -> Option<usize> {
        self.law.storage().then_some(0)
    }

    fn initial_coordinates(
        &self,
        _context: &crate::devices::ModelContext,
    ) -> SpiceResult<Vec<crate::devices::InitialCoordinate>> {
        Ok((self.law == Law::Inductance)
            .then(|| crate::devices::InitialCoordinate {
                coordinate: 0,
                equation: 1,
                value: self.initial.unwrap_or(0.),
                history: crate::devices::InitialCoordinateHistory::EquationCharge {
                    slot: 0,
                    scale: 1.,
                },
            })
            .into_iter()
            .collect())
    }

    fn initial_voltage_storage(
        &self,
        _context: &crate::devices::ModelContext,
    ) -> SpiceResult<Option<crate::devices::InitialVoltageStorage>> {
        Ok(
            (self.law == Law::Capacitance).then_some(crate::devices::InitialVoltageStorage {
                voltage: self.initial,
                slot: 0,
                scale: -1.,
                equation: Some(1),
            }),
        )
    }

    fn initial_expression_storage(&self) -> Option<crate::devices::InitialExpressionStorage> {
        use crate::devices::{InitialExpressionStorage, InitialStoragePolicy, InitialStorageRow};
        let (equation, gauge, policy) = match self.law {
            Law::Charge => (
                InitialStorageRow::Branch(1),
                InitialStorageRow::Branch(0),
                InitialStoragePolicy::Physical,
            ),
            Law::Derivative => (
                InitialStorageRow::Branch(0),
                InitialStorageRow::Signal(self.terminals()[0]),
                InitialStoragePolicy::Derived,
            ),
            _ => return None,
        };
        Some(InitialExpressionStorage {
            slot: 0,
            equation,
            gauge,
            voltage: self.initial,
            policy,
        })
    }

    fn validate_dc_inputs(&self, context: &crate::devices::ModelContext) -> SpiceResult<()> {
        self.expression.validate_dc_inputs(context)
    }

    fn convergence_test(
        &self,
        context: &crate::devices::ConvergenceContext<'_>,
    ) -> SpiceResult<bool> {
        let rows = self.rows(context.unknowns, context.branch)?;
        let branch = if self.law == Law::Derivative {
            rows[1]
        } else {
            rows[0]
        };
        self.with_bound(
            BoundInput {
                solution: context.solution,
                unknowns: context.unknowns,
                controls: context.controls,
                branch,
                time: context.time,
                model: context.model_context,
            },
            |bound, equations| {
                let [p, n, current, rate] = bound.values;
                let voltage = p - n;
                let coefficient = bound.factor * bound.expression.evaluation.value;
                let disagree = |actual: Real, expected: Real, absolute: Real| {
                    !actual.is_finite()
                        || !expected.is_finite()
                        || (actual - expected).abs()
                            > context
                                .reltol
                                .mul_add(actual.abs().max(expected.abs()), absolute)
                };
                let derivative = |quantity: Real| -> SpiceResult<Real> {
                    if let Some(integration) = context.integration {
                        let accepted = |age, slot| {
                            context.accepted(age, slot).ok_or_else(|| {
                                SpiceError::circuit("missing constitutive convergence history")
                            })
                        };
                        let mut history = crate::maths::integrator::ChargeHistory::new(quantity);
                        for age in 1..=integration.charge_history_len() {
                            history.push(accepted(age, 0)?)?;
                        }
                        let previous = integration
                            .needs_previous_derivative()
                            .then(|| accepted(1, 1))
                            .transpose()?;
                        Ok(integration.integrate(&history, previous, 1.)?.derivative)
                    } else {
                        Ok(0.)
                    }
                };
                let dynamic = || -> SpiceResult<Real> {
                    let storage = equations
                        .into_iter()
                        .flatten()
                        .find_map(|equation| equation.storage)
                        .ok_or_else(|| SpiceError::circuit("missing constitutive storage"))?;
                    derivative(storage.value)
                };
                Ok(match self.law {
                    Law::Resistance => disagree(voltage, coefficient * current, context.vntol),
                    Law::Charge | Law::Derivative => {
                        let mut scale = coefficient.abs().max(rate.abs());
                        for (_, value, derivative) in bound.derivatives(
                            equations[0]
                                .ok_or_else(|| {
                                    SpiceError::circuit("missing explicit storage relation")
                                })?
                                .static_part,
                        ) {
                            scale += (value * derivative).abs();
                        }
                        let rounding = 4. * Real::EPSILON * scale;
                        let (actual, absolute) = if self.law == Law::Charge {
                            (current, context.abstol)
                        } else {
                            (voltage, context.vntol)
                        };
                        disagree(rate, coefficient, rounding)
                            || disagree(actual, -dynamic()?, absolute)
                    }
                    Law::Capacitance => {
                        disagree(current, coefficient * rate, context.abstol)
                            || disagree(current, -coefficient * dynamic()?, context.abstol)
                    }
                    Law::Inductance => {
                        disagree(voltage, coefficient * rate, context.vntol)
                            || disagree(voltage, -coefficient * dynamic()?, context.vntol)
                    }
                })
            },
        )
    }

    fn setup_matrix(&self, setup: &mut MatrixSetup<'_>) -> SpiceResult<bool> {
        let pos = setup.node(self.terminals()[0]);
        let neg = setup.node(self.terminals()[1]);
        let current = if self.law == Law::Derivative {
            None
        } else {
            let current = Some(setup.branch(0)?);
            setup.element(pos, current);
            setup.element(neg, current);
            current
        };
        let auxiliary = match self.law {
            Law::Resistance => None,
            Law::Derivative => Some(setup.branch(0)?),
            _ => Some(setup.branch(1)?),
        };
        let primary = if self.law == Law::Derivative {
            pos
        } else {
            current
        };
        for row in [primary, auxiliary].into_iter().flatten() {
            let row = Some(row);
            for column in [pos, neg, current, auxiliary] {
                setup.element(row, column);
            }
            for variable in &self.expression.variables {
                let column = match *variable {
                    super::Variable::Node(node) => setup.node(node),
                    super::Variable::Branch(index) => Some(
                        *setup
                            .controls
                            .get(index)
                            .ok_or_else(|| SpiceError::circuit("unbound passive control"))?,
                    ),
                };
                setup.element(row, column);
            }
        }
        Ok(true)
    }

    fn stamp(&self, context: &mut StampContext<'_>) -> SpiceResult<()> {
        let time = match context.mode {
            AnalysisMode::Transient { time, .. } | AnalysisMode::Instantaneous { time } => time,
            AnalysisMode::OperatingPoint | AnalysisMode::DcSweep => 0.,
            AnalysisMode::Ac { .. } => {
                return Err(SpiceError::circuit(
                    "behavioural passive AC requires small-signal assembly",
                ));
            }
        };
        let rows = self.rows(context.unknowns, context.branches.clone().next())?;
        let branch = if self.law == Law::Derivative {
            rows[1]
        } else {
            rows[0]
        };
        let model = context.model_context();
        // An instance IC is a local terminal voltage at which to evaluate
        // the arbitrary charge law. Other controls retain their seeded
        // values. Only the disposable UIC history load uses this override.
        let seed = if self.law == Law::Charge
            && context.states.initial_conditions()
            && let Some(voltage) = self.initial
        {
            let mut state = context.solution.clone();
            context.unknowns.seed_terminal_voltage(
                &mut state,
                [self.expression.terminals[0], self.expression.terminals[1]],
                voltage,
            )?;
            Some(state)
        } else {
            None
        };
        self.with_bound(
            BoundInput {
                solution: seed.as_ref().unwrap_or(context.solution),
                unknowns: context.unknowns,
                controls: context.controls,
                branch,
                time,
                model: &model,
            },
            |bound, equations| {
                self.kcl(context.matrix, bound.columns, branch)?;
                for (index, equation) in equations.into_iter().enumerate() {
                    let Some(equation) = equation else {
                        continue;
                    };
                    let row = rows[index];
                    let mut rhs;
                    if let Some(storage) = equation.storage {
                        let quantity =
                            if self.law.explicit_storage() && context.states.initial_conditions() {
                                -bound.factor * bound.expression.evaluation.value
                            } else {
                                storage.value
                            };
                        let (history, multiplier) = dynamic::integrate(context, quantity)?;
                        if self.law.explicit_storage()
                            && matches!(context.mode, AnalysisMode::Transient { .. })
                        {
                            let [p, n, current, _] = bound.values;
                            // Store the independently solved physical rate,
                            // not a noisy difference of absolute charges.
                            let derivative = if self.law == Law::Charge {
                                -current
                            } else {
                                n - p
                            };
                            context.states.set(1, derivative)?;
                        }
                        // Keep the integration row in storage units when its
                        // derivative coefficient dominates. Multiplying the
                        // whole row by the same nonzero scalar preserves the
                        // DAE while avoiding large elimination multipliers.
                        let scale = if self.law.explicit_storage() {
                            crate::primitives::numeric::scale_power(
                                1.,
                                -crate::primitives::numeric::binary_exponent(
                                    multiplier.abs().max(1.),
                                )
                                .unwrap_or(0),
                            )
                        } else {
                            1.
                        };
                        rhs = bound.affine(context.matrix, row, equation.static_part, scale)?;
                        bound.linear_matrix(
                            context.matrix,
                            row,
                            storage.direct,
                            multiplier * scale,
                        )?;
                        rhs = history.mul_add(-scale, rhs);
                    } else {
                        rhs = bound.affine(context.matrix, row, equation.static_part, 1.)?;
                    }
                    context.rhs.add_to(row, rhs)?;
                }
                Ok(())
            },
        )
    }

    fn assemble_small_signal(
        &self,
        context: &mut LinearContext<'_>,
        bias: &Vector,
    ) -> SpiceResult<()> {
        let rows = self.rows(context.unknowns, context.branch)?;
        let branch = if self.law == Law::Derivative {
            rows[1]
        } else {
            rows[0]
        };
        self.with_bound(
            BoundInput {
                solution: bias,
                unknowns: context.unknowns,
                controls: context.controls,
                branch,
                time: 0.,
                model: context.model_context,
            },
            |bound, equations| {
                self.kcl(&mut context.system.a, bound.columns, branch)?;
                for (index, equation) in equations.into_iter().enumerate() {
                    let Some(equation) = equation else {
                        continue;
                    };
                    bound.matrix(&mut context.system.a, rows[index], equation.static_part)?;
                    if let Some(storage) = equation.storage {
                        bound.linear_matrix(
                            &mut context.system.e,
                            rows[index],
                            storage.direct,
                            1.,
                        )?;
                    }
                }
                Ok(())
            },
        )
    }

    fn assemble_pole_zero(
        &self,
        context: &mut LinearContext<'_>,
        bias: &Vector,
    ) -> SpiceResult<()> {
        if self.depends_on_frequency() || self.depends_on_time() {
            return Err(SpiceError::circuit(
                "time/frequency-dependent passive pole-zero model",
            ));
        }
        self.assemble_small_signal(context, bias)
    }

    fn assemble_nonlinear_taylor(
        &self,
        context: &mut crate::devices::physical_dae::TaylorContext<'_>,
    ) -> SpiceResult<()> {
        use crate::maths::taylor::Jet;
        let rows = self.rows(context.unknowns, context.branch)?;
        let branch = if self.law == Law::Derivative {
            rows[1]
        } else {
            rows[0]
        };
        let columns = self.columns(context.unknowns, branch);
        if self.law != Law::Derivative {
            context.linear_static(columns[0], Some(branch), 1.)?;
            context.linear_static(columns[1], Some(branch), -1.)?;
        }
        let (evaluation, controls) = self.expression.bound_taylor(context)?;
        let factor = self.expression.checked_factor(context.model.temperature)?;
        let (linear, multiplier) = self.law.constitutive();
        for (&column, coefficient) in columns.iter().zip(linear) {
            context.linear_static(Some(rows[0]), column, coefficient)?;
        }
        let coefficient = evaluation.value.scale(factor);
        let multiplier = if let Some(index) = multiplier {
            let column = columns[index];
            context.static_gradient(Some(rows[0]), column, &coefficient, -1.)?;
            context.variable(column)?
        } else {
            Jet::constant(context.time.order(), 1.)
        };
        context.static_value(Some(rows[0]), &coefficient.multiply(&multiplier), -1.)?;
        for (&column, gradient) in controls.iter().zip(&evaluation.gradient) {
            context.static_gradient(
                Some(rows[0]),
                column,
                &gradient.multiply(&multiplier),
                -factor,
            )?;
        }
        // The evolution equation is linear in the retained storage
        // coordinates, exactly as in the ordinary physical load.
        if let Some(evolution) = self.law.equations([0.; 4], 0.)[1] {
            for (&column, coefficient) in columns.iter().zip(evolution.static_part.direct) {
                context.linear_static(Some(rows[1]), column, coefficient)?;
            }
            if let Some(storage) = evolution.storage {
                for (&column, coefficient) in columns.iter().zip(storage.direct) {
                    context.linear_charge(Some(rows[1]), column, coefficient)?;
                }
            }
        }
        Ok(())
    }

    fn assemble_nonlinear_charge(&self, context: &mut ChargeContext<'_>) -> SpiceResult<()> {
        let rows = self.rows(context.unknowns, context.branch)?;
        let branch = if self.law == Law::Derivative {
            rows[1]
        } else {
            rows[0]
        };
        self.with_bound(
            BoundInput {
                solution: context.state,
                unknowns: context.unknowns,
                controls: context.controls,
                branch,
                time: context.time,
                model: context.model,
            },
            |bound, equations| {
                for (index, equation) in equations.into_iter().enumerate() {
                    if let Some(storage) = equation.and_then(|equation| equation.storage) {
                        let row = rows[index];
                        context.charge.add_to(row, storage.value)?;
                        bound.linear_matrix(context.jacobian, row, storage.direct, 1.)?;
                    }
                }
                Ok(())
            },
        )
    }

    fn noise(
        &self,
        context: &NoiseContext<'_>,
        sources: &mut Vec<NoiseSource>,
    ) -> SpiceResult<DeviceNoise<'_>> {
        if !self.noisy {
            return Ok(DeviceNoise::Noiseless);
        }
        let factor = self
            .expression
            .checked_factor(context.model_context.temperature)?;
        let resistance = self.expression.with_linearization(
            context.bias,
            context.unknowns,
            context.controls,
            0.,
            context.model_context,
            |linearization| Ok(factor * linearization.evaluation.value),
        )?;
        let scale = &self.expression.scale;
        let temperature = scale
            .temperature
            .unwrap_or(context.model_context.temperature)
            + scale.dtemp
            + CELSIUS_TO_KELVIN;
        if !resistance.is_finite()
            || resistance < 0.
            || !temperature.is_finite()
            || temperature <= 0.
        {
            return Err(SpiceError::Numerical {
                context: format!("thermal noise of {}", self.name()),
                message: "thermal resistance must be nonnegative and temperature positive".into(),
            });
        }
        // Noise is the series EMF of the physical resistance, not a
        // parallel current source that bypasses self-current feedback.
        sources.push(NoiseSource::generalized(
            "thermal",
            4. * BOLTZMANN * temperature * resistance,
            vec![(NoiseCoordinate::Branch(0), Complex::real(1.))],
        ));
        Ok(DeviceNoise::Sources {
            family: NoiseFamily::Resistor,
            model: None,
            total: true,
        })
    }
}

/// Bind a typed, already lexical-resolved passive expression.
pub(in crate::devices) fn instantiate(
    instance: &DeviceInstance,
    nodes: &mut NodeAppend<'_>,
) -> SpiceResult<Box<dyn Device>> {
    if instance.model.is_some() || instance.nodes.len() != 2 {
        return Err(unsupported(
            &instance.location,
            "behavioural passive needs two nodes and no model",
        ));
    }
    let mut expression = None;
    let mut scale = BehaviouralScale::default();
    let mut initial = None;
    let mut noisy = false;
    for parameter in &instance.parameters {
        match &parameter.kind {
            ParameterKind::Behavioural(tree) => {
                let law = match (instance.designator, parameter.name.as_str()) {
                    ('r', "resistance") => Law::Resistance,
                    ('c', "capacitance") => Law::Capacitance,
                    ('c', "charge") => Law::Charge,
                    ('l', "inductance") => Law::Inductance,
                    ('b', "__dynamic_derivative") => Law::Derivative,
                    _ => {
                        return Err(unsupported(
                            &parameter.location,
                            "invalid behavioural passive law",
                        ));
                    }
                };
                if expression.replace((law, tree)).is_some() {
                    return Err(unsupported(
                        &parameter.location,
                        "multiple behavioural passive laws",
                    ));
                }
            }
            ParameterKind::Scalar => {
                let value = literal(&parameter.value, &parameter.location, &parameter.name)?;
                match parameter.name.as_str() {
                    "m" => scale.m = value,
                    "tc1" => scale.tc1 = value,
                    "tc2" => scale.tc2 = value,
                    "temp" => scale.temperature = Some(value),
                    "dtemp" => scale.dtemp = value,
                    "noisy" if value == 0. || value.to_bits() == 1_f64.to_bits() => {
                        noisy = value != 0.;
                    }
                    "ic" => initial = Some(value),
                    _ => {
                        return Err(unsupported(
                            &parameter.location,
                            format!("behavioural passive setter {}", parameter.name),
                        ));
                    }
                }
            }
            _ => {
                return Err(unsupported(
                    &parameter.location,
                    format!("unresolved behavioural passive setter {}", parameter.name),
                ));
            }
        }
    }
    let (law, tree) =
        expression.ok_or_else(|| SpiceError::circuit("missing behavioural passive law"))?;
    if noisy && law != Law::Resistance {
        return Err(unsupported(
            &instance.location,
            "thermal noise is only defined for a resistive law",
        ));
    }
    if initial.is_some() && !(law.differential() || law == Law::Charge) {
        return Err(unsupported(
            &instance.location,
            "ic= applies to C/L storage laws",
        ));
    }
    scale.reciprocal_tc = matches!(law, Law::Capacitance | Law::Charge);
    scale.reciprocal_m = matches!(law, Law::Resistance | Law::Inductance);
    let expression = bind_expression(instance, nodes, BehaviouralOutput::Current, tree, scale)?;
    if law == Law::Derivative {
        nodes.set_kind(
            expression.terminals[0],
            crate::primitives::NodeKind::Internal,
        )?;
    }
    Ok(Box::new(Passive {
        expression,
        law,
        initial,
        noisy,
    }))
}
