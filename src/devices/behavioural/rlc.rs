//! Behavioural passive constitutive equations with explicit storage semantics.
//!
//! For a physical passive the first branch unknown is current from positive
//! to negative. Generated derivative operators use their internal node row
//! directly, with no branch or dummy KCL equation.
//! Differential C and L have a second algebraic unknown for the derivative
//! of terminal voltage or current. This retains an exact F+dQ/dt formulation
//! even when the coefficient depends on arbitrary controls.

use super::{
    Behavioural, BehaviouralOutput, BehaviouralScale, Linearization, bind_expression, dynamic,
    literal, unsupported,
};
use crate::devices::linear::LinearContext;
use crate::devices::noise::{DeviceNoise, NoiseContext, NoiseSource};
use crate::devices::setup::MatrixSetup;
use crate::devices::traits::{
    AnalysisMode, ChargeContext, ControlReference, Device, MnaUnknowns, StampContext,
};
use crate::maths::{SparseMatrix, Vector};
use crate::netlist::ast::{DeviceInstance, ParameterKind};
use crate::primitives::{NodeAppend, NodeId, Real, SpiceError, SpiceResult};

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
            Self::Derivative => 0,
            Self::Capacitance | Self::Inductance => 2,
            _ => 1,
        }
    }

    fn equations(self, values: [Real; 4], coefficient: Real) -> [Option<Equation>; 2] {
        let [positive, negative, current, rate] = values;
        let voltage = positive - negative;
        let (algebraic, storage) = match self {
            Self::Resistance => (
                Scalar {
                    value: voltage - coefficient * current,
                    direct: [1., -1., -coefficient, 0.],
                    expression: -current,
                },
                None,
            ),
            Self::Charge | Self::Derivative => (
                if self == Self::Charge {
                    Scalar {
                        value: current,
                        direct: [0., 0., 1., 0.],
                        expression: 0.,
                    }
                } else {
                    Scalar {
                        value: voltage,
                        direct: [1., -1., 0., 0.],
                        expression: 0.,
                    }
                },
                Some(Scalar {
                    value: -coefficient,
                    direct: [0.; 4],
                    expression: -1.,
                }),
            ),
            Self::Capacitance => (
                Scalar {
                    value: current - coefficient * rate,
                    direct: [0., 0., 1., -coefficient],
                    expression: -rate,
                },
                None,
            ),
            Self::Inductance => (
                Scalar {
                    value: voltage - coefficient * rate,
                    direct: [1., -1., 0., -coefficient],
                    expression: -rate,
                },
                None,
            ),
        };
        let derivative = match self {
            Self::Capacitance => Some(Scalar {
                value: -voltage,
                direct: [-1., 1., 0., 0.],
                expression: 0.,
            }),
            Self::Inductance => Some(Scalar {
                value: -current,
                direct: [0., 0., -1., 0.],
                expression: 0.,
            }),
            _ => None,
        };
        [
            Some(Equation {
                static_part: algebraic,
                storage,
            }),
            derivative.map(|storage| Equation {
                static_part: Scalar {
                    value: rate,
                    direct: [0., 0., 0., 1.],
                    expression: 0.,
                },
                storage: Some(storage),
            }),
        ]
    }
}

/// A scalar residual: direct derivatives plus the chain rule through the
/// compiled expression. Coincident rows are summed by the matrix writer.
#[derive(Debug, Clone, Copy)]
struct Scalar {
    value: Real,
    direct: [Real; 4],
    expression: Real,
}

#[derive(Debug, Clone, Copy)]
struct Equation {
    static_part: Scalar,
    storage: Option<Scalar>,
}

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
    fn derivatives(
        &self,
        scalar: Scalar,
        mut emit: impl FnMut(usize, Real, Real) -> SpiceResult<()>,
    ) -> SpiceResult<()> {
        for ((column, value), derivative) in
            self.columns.into_iter().zip(self.values).zip(scalar.direct)
        {
            if let Some(column) = column {
                emit(column, value, derivative)?;
            }
        }
        for ((&column, &value), &gradient) in self
            .expression
            .columns
            .iter()
            .zip(self.expression.values)
            .zip(self.expression.evaluation.gradient)
        {
            if let Some(column) = column {
                emit(column, value, scalar.expression * self.factor * gradient)?;
            }
        }
        Ok(())
    }

    fn matrix(&self, matrix: &mut SparseMatrix, row: usize, scalar: Scalar) -> SpiceResult<()> {
        self.derivatives(scalar, |column, _, derivative| {
            matrix.add(row, column, derivative)
        })
    }

    fn affine(
        &self,
        matrix: &mut SparseMatrix,
        row: usize,
        scalar: Scalar,
        multiplier: Real,
        rhs: &mut Real,
    ) -> SpiceResult<()> {
        self.derivatives(scalar, |column, value, derivative| {
            let derivative = multiplier * derivative;
            matrix.add(row, column, derivative)?;
            *rhs = derivative.mul_add(value, *rhs);
            Ok(())
        })
    }
}

/// A compiled native behavioural passive. Reuses the ordinary behavioural
/// expression compiler, lexical bindings and derivative workspace.
#[derive(Debug, Clone)]
struct Passive {
    expression: Behavioural,
    law: Law,
    initial: Option<Real>,
}

impl Passive {
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
        let columns = [
            unknowns.node_row(self.expression.terminals[0]),
            unknowns.node_row(self.expression.terminals[1]),
            (self.law != Law::Derivative).then_some(branch),
            self.law.differential().then_some(branch + 1),
        ];
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

    fn equation_row(&self, unknowns: &MnaUnknowns, branch: Option<usize>) -> SpiceResult<usize> {
        let row = if self.law == Law::Derivative {
            unknowns.node_row(self.terminals()[0])
        } else {
            branch
        };
        row.ok_or_else(|| SpiceError::circuit(format!("{}: missing constitutive row", self.name())))
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
    fn controlling_sources(&self) -> &[ControlReference] {
        self.expression.controlling_sources()
    }
    fn branch_currents(&self) -> usize {
        self.law.branches()
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

    fn validate_dc_inputs(&self, context: &crate::devices::ModelContext) -> SpiceResult<()> {
        self.expression.validate_dc_inputs(context)
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
        for index in 0..self.law.branches().max(1) {
            let row = if self.law == Law::Derivative {
                pos
            } else {
                Some(setup.branch(index)?)
            };
            for column in [pos, neg, current] {
                setup.element(row, column);
            }
            if self.law.differential() {
                let rate = Some(setup.branch(1)?);
                setup.element(row, rate);
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
            AnalysisMode::Transient { time, .. } => time,
            AnalysisMode::OperatingPoint | AnalysisMode::DcSweep => 0.,
            AnalysisMode::Ac { .. } => {
                return Err(SpiceError::circuit(
                    "behavioural passive AC requires small-signal assembly",
                ));
            }
        };
        let branch = self.equation_row(context.unknowns, context.branches.clone().next())?;
        let model = context.model_context();
        self.with_bound(
            BoundInput {
                solution: context.solution,
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
                    let row = branch + index;
                    let mut rhs = -equation.static_part.value;
                    bound.affine(context.matrix, row, equation.static_part, 1., &mut rhs)?;
                    if let Some(storage) = equation.storage {
                        let (derivative, multiplier) = dynamic::integrate(context, storage.value)?;
                        rhs -= derivative;
                        bound.affine(context.matrix, row, storage, multiplier, &mut rhs)?;
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
        let branch = self.equation_row(context.unknowns, context.branch)?;
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
                    bound.matrix(&mut context.system.a, branch + index, equation.static_part)?;
                    if let Some(storage) = equation.storage {
                        bound.matrix(&mut context.system.e, branch + index, storage)?;
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

    fn assemble_nonlinear_charge(&self, context: &mut ChargeContext<'_>) -> SpiceResult<()> {
        let branch = self.equation_row(context.unknowns, context.branch)?;
        self.with_bound(
            BoundInput {
                solution: context.state,
                unknowns: context.unknowns,
                controls: context.controls,
                branch,
                time: 0.,
                model: context.model,
            },
            |bound, equations| {
                for (index, equation) in equations.into_iter().enumerate() {
                    if let Some(storage) = equation.and_then(|equation| equation.storage) {
                        let row = branch + index;
                        context.charge.add_to(row, storage.value)?;
                        bound.matrix(context.jacobian, row, storage)?;
                    }
                }
                Ok(())
            },
        )
    }

    fn noise(
        &self,
        _context: &NoiseContext<'_>,
        _sources: &mut Vec<NoiseSource>,
    ) -> SpiceResult<DeviceNoise<'_>> {
        // Native behavioural passive expressions are noiseless unless an
        // explicit noise law is requested. Such requests are rejected at bind.
        Ok(DeviceNoise::Noiseless)
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
                    "noisy" if value.abs() == 0. => {}
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
    if initial.is_some() && !law.differential() {
        return Err(unsupported(
            &instance.location,
            "ic= applies to differential C/L, not an arbitrary charge law",
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
    }))
}
