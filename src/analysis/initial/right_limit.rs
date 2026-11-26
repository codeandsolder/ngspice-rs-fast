//! Right-limit initialization and continuous index reduction.
use super::{
    AbsoluteRow, ImpulseStorage, Tolerances, VoltageVertex,
    derivative_array::{DerivativeArray, InitialConstraint},
    failure,
};
use crate::devices::{Circuit, InitialStoragePolicy, LoadRequest, PreparedRun, StateHistory};
use crate::maths::{Coefficients, Vector};
use crate::primitives::{Real, SpiceResult};

/// Generated differentiators require actual right-hand derivatives rather
/// than the zero-rate DC state. Ordinary circuits retain their C startup.
pub(in crate::analysis) fn needs_right_limit(circuit: &Circuit) -> bool {
    circuit.devices().iter().any(|device| {
        device
            .initial_expression_storage()
            .is_some_and(|storage| storage.policy == InitialStoragePolicy::Derived)
    })
}

fn vertex_value(x: &Vector, vertex: VoltageVertex) -> Real {
    match vertex {
        VoltageVertex::Ground => 0.,
        VoltageVertex::Node(AbsoluteRow(row)) => x.as_slice()[row],
    }
}

fn derivative_constraints(
    storage: &ImpulseStorage,
    trajectory: &[Vector],
    charge: &crate::devices::physical_jet::ResidualJet,
) -> Vec<InitialConstraint> {
    let x = &trajectory[0];
    let mut constraints = Vec::new();
    for (&AbsoluteRow(row), coordinate) in &storage.fixed_coordinates {
        constraints.push(InitialConstraint {
            terms: vec![(row, 1.)],
            residual: x.as_slice()[row] - coordinate.value,
            optional: false,
        });
    }
    for cap in &storage.capacitors {
        constraints.push(InitialConstraint {
            terms: [(cap.positive, 1.), (cap.negative, -1.)]
                .into_iter()
                .filter_map(|(vertex, weight)| match vertex {
                    VoltageVertex::Ground => None,
                    VoltageVertex::Node(AbsoluteRow(row)) => Some((row, weight)),
                })
                .collect(),
            residual: vertex_value(x, cap.positive) - vertex_value(x, cap.negative) - cap.voltage,
            optional: false,
        });
    }
    for (&AbsoluteRow(row), item) in &storage.charges {
        constraints.push(InitialConstraint {
            terms: charge.jacobians[0]
                .triplets()
                .iter()
                .filter(|term| term.row == row)
                .map(|term| (term.col, term.value))
                .collect(),
            residual: charge.values[0].as_slice()[row] - item.target,
            optional: item.policy == InitialStoragePolicy::Derived,
        });
    }
    constraints
}

/// A physical integration quantity, in the exact units used by the
/// device's accepted history. A row of Q is a quantity only when its
/// device has declared the conversion; nodal Q sums several capacitors.
#[derive(Clone, Copy)]
enum Quantity {
    Voltage {
        positive: VoltageVertex,
        negative: VoltageVertex,
        factor: Real,
    },
    ChargeRow {
        row: usize,
        factor: Real,
    },
}

#[derive(Clone, Copy)]
struct Storage {
    slot: usize,
    quantity: Quantity,
    independent: bool,
    derived: bool,
}

impl Storage {
    fn terms(
        self,
        charge: &crate::devices::physical_jet::ResidualJet,
        degree: usize,
    ) -> Vec<(usize, Real)> {
        match self.quantity {
            Quantity::Voltage {
                positive,
                negative,
                factor,
            } if degree == 0 => [(positive, factor), (negative, -factor)]
                .into_iter()
                .filter_map(|(vertex, weight)| match vertex {
                    VoltageVertex::Ground => None,
                    VoltageVertex::Node(AbsoluteRow(row)) => Some((row, weight)),
                })
                .collect(),
            Quantity::Voltage { .. } => Vec::new(),
            Quantity::ChargeRow { row, factor } => charge.jacobians[degree]
                .triplets()
                .iter()
                .filter(|term| term.row == row)
                .map(|term| (term.col, factor * term.value))
                .collect(),
        }
    }

    fn value(
        self,
        trajectory: &[Vector],
        charge: &crate::devices::physical_jet::ResidualJet,
        degree: usize,
    ) -> Real {
        match self.quantity {
            Quantity::Voltage {
                positive,
                negative,
                factor,
            } => {
                factor
                    * (vertex_value(&trajectory[degree], positive)
                        - vertex_value(&trajectory[degree], negative))
            }
            Quantity::ChargeRow { row, factor } => factor * charge.values[degree].as_slice()[row],
        }
    }

    fn integration_constraint(
        self,
        trajectory: &[Vector],
        charge: &crate::devices::physical_jet::ResidualJet,
        scale: Real,
        history: &StateHistory,
        coefficients: &Coefficients,
    ) -> SpiceResult<InitialConstraint> {
        let n = trajectory[0].len();
        let q = self.value(trajectory, charge, 0);
        let rate = self.value(trajectory, charge, 1) / scale;
        let mut values = crate::maths::integrator::ChargeHistory::new(q);
        let accepted = |age, offset| {
            history
                .accepted(age)
                .and_then(|state| state.get(self.slot + offset))
                .copied()
                .ok_or_else(|| failure("missing independent right-limit history"))
        };
        for age in 1..=coefficients.charge_history_len() {
            values.push(accepted(age, 0)?)?;
        }
        let previous = coefficients
            .needs_previous_derivative()
            .then(|| accepted(1, 1))
            .transpose()?;
        let integrated = coefficients.integrate(&values, previous, 1.)?.derivative;
        let ag = coefficients.ag()[0];
        let inverse = (scale * ag).recip();
        let mut terms = self.terms(charge, 0);
        terms.extend(
            self.terms(charge, 1)
                .into_iter()
                .map(|(row, value)| (row, -value * inverse)),
        );
        terms.extend(
            self.terms(charge, 0)
                .into_iter()
                .map(|(row, value)| (n + row, -value * inverse)),
        );
        Ok(InitialConstraint {
            terms,
            residual: (integrated - rate) / ag,
            optional: false,
        })
    }
}

/// The physical quantities that are independently integrated are determined
/// by the initialized derivative array, never by expression syntax.
/// Dependent quantities are reconstructed from the differentiated DAE.
pub(in crate::analysis) struct RightLimitPlan {
    storage: Vec<Storage>,
    scale: Real,
    order: usize,
}

impl RightLimitPlan {
    fn from_initial(
        storage: &ImpulseStorage,
        retained: &[bool],
        scale: Real,
        order: usize,
    ) -> SpiceResult<Self> {
        let mut quantities = Vec::new();
        let mut independent = retained.iter().copied();
        for fixed in storage.fixed_coordinates.values() {
            quantities.push(Storage {
                slot: fixed.slot,
                quantity: Quantity::ChargeRow {
                    row: fixed.equation.0,
                    factor: fixed.history_scale,
                },
                independent: independent
                    .next()
                    .ok_or_else(|| failure("missing initial rank decision"))?,
                derived: false,
            });
        }
        for cap in &storage.capacitors {
            quantities.push(Storage {
                slot: cap.slot,
                quantity: Quantity::Voltage {
                    positive: cap.positive,
                    negative: cap.negative,
                    factor: cap.scale,
                },
                independent: independent
                    .next()
                    .ok_or_else(|| failure("missing initial rank decision"))?,
                derived: false,
            });
        }
        for (row, item) in &storage.charges {
            quantities.push(Storage {
                slot: item.slot,
                quantity: Quantity::ChargeRow {
                    row: row.0,
                    factor: 1.,
                },
                independent: independent
                    .next()
                    .ok_or_else(|| failure("missing initial rank decision"))?,
                derived: item.policy == InitialStoragePolicy::Derived,
            });
        }
        if independent.next().is_some() {
            return Err(failure("unexpected initial rank decision"));
        }
        Ok(Self {
            storage: quantities,
            scale,
            order,
        })
    }

    pub(in crate::analysis) fn independent(&self, slot: usize) -> bool {
        self.storage
            .iter()
            .find(|item| item.slot == slot)
            .is_none_or(|item| item.independent)
    }

    fn canonical(
        &self,
        trajectory: &[Vector],
        charge: &crate::devices::physical_jet::ResidualJet,
        scale: Real,
    ) -> Vec<(usize, Real)> {
        self.storage
            .iter()
            .flat_map(|item| {
                [
                    (item.slot, item.value(trajectory, charge, 0)),
                    (item.slot + 1, item.value(trajectory, charge, 1) / scale),
                ]
            })
            .collect()
    }

    pub(in crate::analysis) fn step(
        &self,
        prepared: &PreparedRun<'_>,
        system: &crate::devices::LinearSystem,
        context: &crate::devices::ModelContext,
        seed: &Vector,
        time: Real,
        limit: crate::devices::Limit,
        coefficients: &Coefficients,
        history: &StateHistory,
        tolerances: Tolerances,
        max_iterations: usize,
    ) -> SpiceResult<(Vector, Vec<(usize, Real)>)> {
        let n = seed.len();
        let scale = if limit == crate::devices::Limit::Left {
            -self.scale
        } else {
            self.scale
        };
        let required = (0..2 * n).collect::<Vec<_>>();
        let mut trajectory = vec![seed.clone()];
        trajectory.resize_with(self.order + 1, || Vector::zeros(n));
        let mut previous = Real::INFINITY;
        for _ in 0..max_iterations {
            let jet = prepared.physical_taylor_jet(context, system, &trajectory, time, scale)?;
            let constraints = self
                .storage
                .iter()
                .filter(|item| item.independent)
                .map(|item| {
                    item.integration_constraint(
                        &trajectory,
                        &jet.charge,
                        scale,
                        history,
                        coefficients,
                    )
                })
                .collect::<SpiceResult<Vec<_>>>()?;
            let array =
                DerivativeArray::assemble(&jet.static_part, &jet.charge, &constraints, scale)?;
            let correction = array
                .solve(&required, &[])?
                .ok_or_else(|| failure("right-limit DAE lost its independent storage rank"))?;
            let residual = residual_ratio(&array, &trajectory, &correction.retained, tolerances)?;
            let change = correction_size(&trajectory, &correction.coefficients);
            if residual <= 1. && (change <= 64. * Real::EPSILON || residual >= previous) {
                let canonical = self.canonical(&trajectory, &jet.charge, scale);
                return Ok((trajectory.remove(0), canonical));
            }
            if change == 0. {
                break;
            }
            apply_correction(&mut trajectory, correction.coefficients)?;
            previous = residual;
        }
        Err(failure(format!(
            "right-limit differentiated transient constraints did not converge at {time:e}"
        )))
    }
}

fn residual_ratio(
    array: &DerivativeArray,
    trajectory: &[Vector],
    retained: &[bool],
    tolerances: Tolerances,
) -> SpiceResult<Real> {
    array.residual_ratio(
        trajectory,
        retained,
        tolerances.reltol.min(1e-10),
        tolerances
            .abstol
            .min(tolerances.vntol)
            .min(tolerances.chgtol),
    )
}

fn correction_size(trajectory: &[Vector], correction: &[Vector]) -> Real {
    trajectory
        .iter()
        .zip(correction)
        .flat_map(|(x, dx)| {
            x.as_slice()
                .iter()
                .zip(dx.as_slice())
                .map(|(&value, &delta)| delta.abs() / (value.abs() + 1.))
        })
        .fold(0., Real::max)
}

fn apply_correction(trajectory: &mut [Vector], correction: Vec<Vector>) -> SpiceResult<()> {
    for (x, dx) in trajectory.iter_mut().zip(correction) {
        for (value, delta) in x.as_mut_slice().iter_mut().zip(dx.as_slice()) {
            *value += delta;
        }
        if !x.is_finite() {
            return Err(failure("nonfinite right-limit Newton correction"));
        }
    }
    Ok(())
}

/// A verified initial algebraic solution, with canonical history rates
/// and quantities. All evaluations use disposable coefficient-space state.
pub(in crate::analysis) struct RightInitial {
    pub(in crate::analysis) solution: Vector,
    pub(in crate::analysis) canonical: Vec<(usize, Real)>,
    pub(in crate::analysis) plan: RightLimitPlan,
}

pub(in crate::analysis) fn solve_right_limit(
    prepared: &PreparedRun<'_>,
    system: &crate::devices::LinearSystem,
    request: &LoadRequest<'_>,
    trial: &crate::devices::TrialState,
    uic: bool,
    scale: Real,
    tolerances: Tolerances,
    options: &crate::analysis::newton::NewtonOptions,
) -> SpiceResult<RightInitial> {
    let circuit = prepared.circuit();
    let mut storage =
        ImpulseStorage::collect(circuit, request.solution, request.model_context, trial)?;
    if !uic {
        // Instance ic= is only authoritative under UIC. Otherwise the
        // physical storage is fixed by the already converged left bias.
        for (&AbsoluteRow(row), coordinate) in &mut storage.fixed_coordinates {
            coordinate.value = request.solution.as_slice()[row];
        }
        for cap in &mut storage.capacitors {
            cap.voltage = vertex_value(request.solution, cap.positive)
                - vertex_value(request.solution, cap.negative);
        }
    }
    let n = circuit.unknown_count();
    // Require both the complete MNA point and its first derivative. This
    // forces enough prolongation even if an optional derived input seed
    // could temporarily determine a lower-order algebraic coordinate.
    let required = (0..n.saturating_mul(2)).collect::<Vec<_>>();
    let mut trajectory = vec![request.solution.clone(), Vector::zeros(n)];
    let max_order = 32_768_usize
        .checked_div(n.max(1))
        .unwrap_or(0)
        .saturating_sub(1)
        .min(64);
    if max_order == 0 {
        return Err(failure("right-limit derivative coordinate budget exceeded"));
    }
    for order in 1..=max_order {
        trajectory.resize_with(order + 1, || Vector::zeros(n));
        let mut previous = Real::INFINITY;
        for _ in 0..options.max_iterations {
            let jet = prepared.physical_taylor_jet(
                request.model_context,
                system,
                &trajectory,
                0.,
                scale,
            )?;
            let constraints = derivative_constraints(&storage, &trajectory, &jet.charge);
            let array =
                DerivativeArray::assemble(&jet.static_part, &jet.charge, &constraints, scale)?;
            let Some(correction) = array.solve(&required, &[])? else {
                break;
            };
            let residual = residual_ratio(&array, &trajectory, &correction.retained, tolerances)?;
            let change = correction_size(&trajectory, &correction.coefficients);
            if residual <= 1. && (change <= 64. * Real::EPSILON || residual >= previous) {
                let plan =
                    RightLimitPlan::from_initial(&storage, &correction.retained, scale, order)?;
                let canonical = plan.canonical(&trajectory, &jet.charge, scale);
                if !uic {
                    // A non-UIC run has actual left-side storage. A finite
                    // jump of a differentiated input is an impulse, not a
                    // valid classical right derivative across the boundary.
                    for item in plan.storage.iter().filter(|item| item.derived) {
                        let left = trial.values()[item.slot];
                        let right = item.value(&trajectory, &jet.charge, 0);
                        let tolerance = tolerances
                            .reltol
                            .mul_add(left.abs().max(right.abs()), tolerances.chgtol);
                        if (left - right).abs() > tolerance {
                            return Err(failure(
                                "non-UIC differentiated input jumps at t=0; its classical derivative includes an impulse",
                            ));
                        }
                    }
                }
                return Ok(RightInitial {
                    solution: trajectory.remove(0),
                    canonical,
                    plan,
                });
            }
            if change == 0. {
                return Err(failure(format!(
                    "inconsistent right-limit derivative constraints (residual ratio {residual:e}); an impulse or noncausal dependency may be required"
                )));
            }
            apply_correction(&mut trajectory, correction.coefficients)?;
            previous = residual;
        }
    }
    Err(failure(
        "right-limit DAE derivatives are not uniquely consistent within the bounded derivative array",
    ))
}
