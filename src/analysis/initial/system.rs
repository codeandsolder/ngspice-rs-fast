//! Ordered elimination of fixed storage coordinates and compact instantaneous MNA.
mod derivative_array;
mod rank;
use super::constraints::{AbsoluteRow, VoltageVertex};
use super::{CapacitorInitial, ImpulseStorage, Tolerances, failure};
use crate::devices::{Circuit, InitialStoragePolicy};
use crate::maths::{SparseMatrix, Vector};
use crate::primitives::{Real, SpiceResult};
pub(super) use derivative_array::{
    RightLimit, SolveOutcome, solve_event, solve_right_limit, solve_step,
};
use std::collections::BTreeSet;

#[derive(Clone, Copy)]
struct EquationId(usize);

#[derive(Clone, Copy)]
struct VariableId(usize);

#[derive(Clone, Copy)]
struct CompactRow(usize);

#[derive(Clone, Copy)]
struct CompactColumn(usize);

#[derive(Clone, Copy)]
enum VariableBinding {
    Fixed(Real),
    Unknown(VariableId),
}

#[derive(Clone, Copy)]
struct Binding {
    equation: Option<EquationId>,
    variable: VariableBinding,
}

impl Binding {
    fn free(&self) -> SpiceResult<(EquationId, VariableId)> {
        match (self.equation, self.variable) {
            (Some(equation), VariableBinding::Unknown(variable)) => Ok((equation, variable)),
            _ => Err(failure("voltage constraint names an eliminated coordinate")),
        }
    }
}

#[derive(Clone, Copy)]
enum EquationOrigin<'a> {
    Node(AbsoluteRow),
    Branch(AbsoluteRow),
    Capacitor(&'a str),
    Charge(&'a str),
    ChargeLaw(&'a str),
    Derived(AbsoluteRow),
}

struct Equation<'a> {
    origin: EquationOrigin<'a>,
    rhs: Real,
    scale: Real,
    fixed: Vec<AbsoluteRow>,
}

impl<'a> Equation<'a> {
    const fn new(origin: EquationOrigin<'a>, rhs: Real) -> Self {
        Self {
            origin,
            rhs,
            scale: rhs.abs(),
            fixed: Vec::new(),
        }
    }
}

/// MNA voltage lookup after eliminating fixed coordinates and empty columns.
pub(super) struct InstantaneousSolution {
    original: Vec<Option<Real>>,
    /// Coordinates in the un-compacted reduced system, including appended
    /// capacitor currents. Unused columns can be set to zero because no
    /// retained equation depends on them at this linearization.
    reduced: Vec<Real>,
    independent_derived: BTreeSet<AbsoluteRow>,
}

impl InstantaneousSolution {
    pub(super) fn retained_derived(&self, row: AbsoluteRow) -> bool {
        self.independent_derived.contains(&row)
    }

    pub(super) fn update(&self, initial: &Vector, scale: Real) -> Vector {
        Vector::from_slice(
            &initial
                .as_slice()
                .iter()
                .zip(&self.original)
                .map(|(&previous, value)| {
                    value.map_or(previous, |value| {
                        if scale >= 1. {
                            value
                        } else {
                            scale.mul_add(value - previous, previous)
                        }
                    })
                })
                .collect::<Vec<_>>(),
        )
    }

    pub(super) fn step_scale(&self, initial: &Vector, limit: Real, limited: &[bool]) -> Real {
        let step = initial
            .as_slice()
            .iter()
            .zip(&self.original)
            .zip(limited)
            .filter_map(|((&previous, value), &limited)| {
                limited.then_some(value.map_or(0., |value| (value - previous).abs()))
            })
            .fold(0., Real::max);
        if step > limit { limit / step } else { 1. }
    }

    pub(super) fn potential(&self, vertex: VoltageVertex) -> Option<Real> {
        match vertex {
            VoltageVertex::Ground => Some(0.),
            VoltageVertex::Node(AbsoluteRow(row)) => self.original.get(row).copied().flatten(),
        }
    }
}

/// Reduced system with explicit original-row bindings, equations in original
/// order, and free capacitor current variables appended in device order.
pub(super) struct InstantaneousSystem<'a> {
    bindings: Vec<Binding>,
    equations: Vec<Equation<'a>>,
    matrix: SparseMatrix,
}

struct InitialLayout<'a> {
    bindings: Vec<Binding>,
    equations: Vec<Equation<'a>>,
    is_branch: Vec<bool>,
    removed: Vec<bool>,
}

impl<'a> InitialLayout<'a> {
    fn bind(circuit: &Circuit, rhs: &Vector, storage: &'a ImpulseStorage) -> SpiceResult<Self> {
        let unknowns = circuit.unknown_count();
        if rhs.len() != unknowns {
            return Err(failure(
                "instantaneous matrix/vector size does not match finalized circuit",
            ));
        }
        if storage
            .fixed_coordinates
            .iter()
            .any(|(row, value)| row.0 >= unknowns || value.equation.0 >= unknowns)
        {
            return Err(failure(
                "fixed coordinate or evolution equation outside finalized circuit",
            ));
        }
        let mut is_branch = vec![false; unknowns];
        for index in 0..circuit.device_count() {
            let range = circuit
                .branch_rows(index)
                .ok_or_else(|| failure("missing finalized branch rows"))?;
            for row in is_branch
                .get_mut(range)
                .ok_or_else(|| failure("branch row outside finalized circuit"))?
            {
                *row = true;
            }
        }
        let mut removed = vec![false; unknowns];
        for (row, fixed) in &storage.fixed_coordinates {
            if !fixed.value.is_finite()
                || !is_branch[row.0]
                || !is_branch[fixed.equation.0]
                || std::mem::replace(&mut removed[fixed.equation.0], true)
            {
                return Err(failure("invalid or duplicate eliminated state equation"));
            }
        }
        let mut equations = Vec::new();
        let mut bindings = Vec::with_capacity(unknowns);
        let mut variables = 0;
        let charge_laws: std::collections::BTreeMap<_, _> = storage
            .charges
            .values()
            .filter(|charge| charge.policy == InitialStoragePolicy::Physical)
            .map(|charge| (charge.gauge, charge.name.as_str()))
            .collect();
        for (row, &kind) in is_branch.iter().enumerate() {
            let absolute = AbsoluteRow(row);
            let variable = storage.fixed_coordinates.get(&absolute).map_or_else(
                || {
                    let binding = VariableBinding::Unknown(VariableId(variables));
                    variables += 1;
                    binding
                },
                |fixed| VariableBinding::Fixed(fixed.value),
            );
            let equation = if removed[row] {
                None
            } else {
                let equation = EquationId(equations.len());
                let origin = storage.charges.get(&absolute).map_or_else(
                    || {
                        if let Some(&name) = charge_laws.get(&absolute) {
                            EquationOrigin::ChargeLaw(name)
                        } else if kind {
                            EquationOrigin::Branch(absolute)
                        } else {
                            EquationOrigin::Node(absolute)
                        }
                    },
                    |charge| match charge.policy {
                        InitialStoragePolicy::Physical => EquationOrigin::Charge(&charge.name),
                        InitialStoragePolicy::Derived => EquationOrigin::Derived(absolute),
                    },
                );
                equations.push(Equation::new(origin, rhs.as_slice()[row]));
                Some(equation)
            };
            bindings.push(Binding { equation, variable });
        }
        if equations.len() != variables {
            return Err(failure(
                "unbalanced fixed coordinates and released equations",
            ));
        }
        Ok(Self {
            bindings,
            equations,
            is_branch,
            removed,
        })
    }
}

impl<'a> InstantaneousSystem<'a> {
    pub(super) fn assemble(
        circuit: &Circuit,
        matrix: &SparseMatrix,
        rhs: &Vector,
        storage: &'a ImpulseStorage,
        kept: &[&'a CapacitorInitial],
    ) -> SpiceResult<Self> {
        let unknowns = circuit.unknown_count();
        if matrix.rows() != unknowns || matrix.cols() != unknowns {
            return Err(failure(
                "instantaneous matrix size does not match finalized circuit",
            ));
        }
        let InitialLayout {
            bindings,
            equations,
            is_branch,
            removed,
        } = InitialLayout::bind(circuit, rhs, storage)?;
        let mut replacements = vec![None; unknowns];
        for capacitor in kept {
            if let Some(row) = capacitor.equation
                && (row.0 >= unknowns
                    || removed[row.0]
                    || !is_branch[row.0]
                    || replacements[row.0].replace(*capacitor).is_some())
            {
                return Err(failure(
                    "invalid or duplicate voltage evolution replacement",
                ));
            }
        }
        let size = equations
            .len()
            .checked_add(
                kept.iter()
                    .filter(|capacitor| capacitor.equation.is_none())
                    .count(),
            )
            .ok_or_else(|| failure("instantaneous system size overflow"))?;
        let mut system = Self {
            bindings,
            equations,
            matrix: SparseMatrix::new(size, size),
        };

        // Retain the original contribution order for eliminated-column RHS
        // arithmetic. Remaining matrix duplicates are folded stably below.
        for triplet in matrix.triplets() {
            if replacements[triplet.row].is_some() {
                continue;
            }
            let Some(equation) = system.bindings[triplet.row].equation else {
                continue;
            };
            match system.bindings[triplet.col].variable {
                VariableBinding::Fixed(current) => {
                    let contribution = triplet.value * current;
                    let row = &mut system.equations[equation.0];
                    row.rhs -= contribution;
                    row.scale += contribution.abs();
                    row.fixed.push(AbsoluteRow(triplet.col));
                }
                VariableBinding::Unknown(variable) => {
                    system.add(equation, variable, triplet.value)?;
                }
            }
        }
        for capacitor in kept {
            let (equation, current) = if let Some(row) = capacitor.equation {
                let equation = system.bindings[row.0]
                    .equation
                    .ok_or_else(|| failure("missing voltage evolution equation"))?;
                system.equations[equation.0] = Equation::new(
                    EquationOrigin::Capacitor(&capacitor.name),
                    capacitor.voltage,
                );
                (equation, None)
            } else {
                let equation = EquationId(system.equations.len());
                let variable = VariableId(system.equations.len());
                system.equations.push(Equation::new(
                    EquationOrigin::Capacitor(&capacitor.name),
                    capacitor.voltage,
                ));
                (equation, Some(variable))
            };
            for (vertex, sign) in [(capacitor.positive, 1.), (capacitor.negative, -1.)] {
                if let VoltageVertex::Node(row) = vertex {
                    let (node_equation, node_variable) = system
                        .bindings
                        .get(row.0)
                        .ok_or_else(|| failure("capacitor voltage row outside finalized circuit"))?
                        .free()?;
                    if let Some(current) = current {
                        system.add(node_equation, current, sign)?;
                    }
                    system.add(equation, node_variable, sign)?;
                }
            }
        }
        Ok(system)
    }

    fn add(&mut self, row: EquationId, column: VariableId, value: Real) -> SpiceResult<()> {
        if !value.is_finite() {
            return Err(failure("nonfinite instantaneous coefficient"));
        }
        self.matrix.add(row.0, column.0, value)
    }

    fn check_empty_rows(
        &self,
        circuit: &Circuit,
        rows: &[bool],
        storage: &ImpulseStorage,
        tolerance: Tolerances,
    ) -> SpiceResult<()> {
        for (equation, &used) in self.equations.iter().zip(rows) {
            if used
                || matches!(
                    equation.origin,
                    EquationOrigin::Charge(_) | EquationOrigin::Derived(_)
                )
            {
                continue;
            }
            let absolute = match equation.origin {
                EquationOrigin::Node(_) => tolerance.abstol,
                EquationOrigin::Branch(_)
                | EquationOrigin::Capacitor(_)
                | EquationOrigin::Derived(_) => tolerance.vntol,
                EquationOrigin::Charge(_) | EquationOrigin::ChargeLaw(_) => tolerance.chgtol,
            };
            let relative_limit = tolerance.reltol * equation.scale;
            if equation.rhs.abs() > absolute + relative_limit {
                let names = circuit.unknown_names();
                let what = match equation.origin {
                    EquationOrigin::Node(row) => format!("currents into node {}", names[row.0]),
                    EquationOrigin::Branch(row) => format!("voltage relation {}", names[row.0]),
                    EquationOrigin::Capacitor(name) => {
                        format!("voltage relation of capacitor {name}")
                    }
                    EquationOrigin::Charge(name) | EquationOrigin::ChargeLaw(name) => {
                        format!("charge relation of {name}")
                    }
                    EquationOrigin::Derived(row) => format!("derived storage of {}", names[row.0]),
                };
                let mut fixed = equation.fixed.clone();
                fixed.sort_unstable();
                fixed.dedup();
                let who: Vec<_> = fixed
                    .iter()
                    .filter_map(|row| {
                        storage
                            .fixed_coordinates
                            .get(row)
                            .map(|current| current.name.as_str())
                    })
                    .collect();
                let who = if who.is_empty() {
                    String::new()
                } else {
                    format!(" (storage devices: {})", who.join(", "))
                };
                return Err(failure(format!(
                    "uic initial conditions are inconsistent with the circuit at t = 0+: {what} \
                     left over by fixed storage coordinates and sources is {:e} instead of 0, an \
                     impulse{who}",
                    equation.rhs
                )));
            }
        }
        Ok(())
    }

    /// Check the true residual at the point that was just reloaded, not
    /// the residual of the preceding Newton linearization. The appended
    /// capacitor currents are retained in the reduced coordinate vector.
    pub(super) fn residual_ratio(
        &self,
        solution: &InstantaneousSolution,
        state: &Vector,
        tolerance: Tolerances,
    ) -> (Real, bool) {
        if solution.reduced.len() != self.equations.len() {
            return (Real::INFINITY, false);
        }
        // An original coordinate can be inactive in one linearization
        // and active in the next. Verify the vector that was actually
        // reloaded; only appended free currents come from the reduced
        // solve's separate coordinates.
        let mut reduced = solution.reduced.clone();
        for (binding, &value) in self.bindings.iter().zip(state.as_slice()) {
            if let VariableBinding::Unknown(variable) = binding.variable {
                reduced[variable.0] = value;
            }
        }
        let mut values: Vec<_> = self
            .equations
            .iter()
            .map(|eq| (-eq.rhs, eq.rhs.abs(), 1_usize))
            .collect();
        for triplet in self.matrix.triplets() {
            let contribution = triplet.value * reduced[triplet.col];
            let (residual, scale, count) = &mut values[triplet.row];
            *residual += contribution;
            *scale += contribution.abs();
            *count += 1;
        }
        let mut ratio: Real = 0.;
        let mut roundoff = true;
        for (eq, (residual, scale, count)) in self.equations.iter().zip(values) {
            if matches!(
                eq.origin,
                EquationOrigin::Charge(_) | EquationOrigin::Derived(_)
            ) {
                continue;
            }
            let absolute = match eq.origin {
                EquationOrigin::Node(_) => tolerance.abstol,
                EquationOrigin::Branch(_)
                | EquationOrigin::Capacitor(_)
                | EquationOrigin::Derived(_) => tolerance.vntol,
                EquationOrigin::Charge(_) | EquationOrigin::ChargeLaw(_) => tolerance.chgtol,
            };
            if !residual.is_finite() || !scale.is_finite() {
                return (Real::INFINITY, false);
            }
            ratio = ratio.max(residual.abs() / tolerance.reltol.mul_add(scale, absolute));
            // Dot-product accumulation error, plus the rounded affine
            // residual and right-hand side from the device's local load.
            let count = u32::try_from(count).unwrap_or(u32::MAX);
            let rounding = (f64::from(count) + 4.) * Real::EPSILON * scale;
            roundoff &= residual.abs() <= rounding;
        }
        (ratio, roundoff)
    }

    /// Fold before identifying empty equations and variables. In particular,
    /// exactly cancelling stamps do not make a row/column nonempty.
    pub(super) fn solve(
        mut self,
        circuit: &Circuit,
        storage: &ImpulseStorage,
        tolerance: Tolerances,
    ) -> SpiceResult<Option<InstantaneousSolution>> {
        if self
            .equations
            .iter()
            .any(|eq| !eq.rhs.is_finite() || !eq.scale.is_finite())
        {
            return Err(failure(
                "nonfinite instantaneous right-hand side or constraint scale",
            ));
        }
        self.matrix.fold_duplicates();
        let size = self.equations.len();
        let mut row_used = vec![false; size];
        let mut column_used = vec![false; size];
        for triplet in self.matrix.triplets() {
            if !triplet.value.is_finite() {
                return Err(failure("nonfinite folded instantaneous coefficient"));
            }
            row_used[triplet.row] = true;
            column_used[triplet.col] = true;
        }
        // Even a structurally empty storage-current column can be fixed
        // as a gauge, e.g. a constant charge law between identical nodes.
        let gauge_variables: Vec<_> = storage
            .charges
            .values()
            .filter_map(|charge| match self.bindings[charge.gauge.0].variable {
                VariableBinding::Unknown(variable) => Some(variable.0),
                VariableBinding::Fixed(_) => None,
            })
            .collect();
        for &variable in &gauge_variables {
            column_used[variable] = true;
        }
        self.check_empty_rows(circuit, &row_used, storage, tolerance)?;

        let mut rhs = Vec::new();
        let rows: Vec<_> = self
            .equations
            .iter()
            .zip(&row_used)
            .map(|(eq, &used)| {
                used.then(|| {
                    let row = CompactRow(rhs.len());
                    rhs.push(eq.rhs);
                    row
                })
            })
            .collect();
        let mut columns = 0;
        let column_index: Vec<_> = column_used
            .into_iter()
            .map(|used| {
                used.then(|| {
                    let col = CompactColumn(columns);
                    columns += 1;
                    col
                })
            })
            .collect();
        if rhs.len() != columns && storage.charges.is_empty() {
            return Err(failure(format!(
                "uic initial conditions cannot be shown consistent: the instantaneous system has \
                 {} equations for {columns} unknowns (over- or under-determined by ideal sources, \
                 capacitors and inductors)",
                rhs.len()
            )));
        }
        if rhs.is_empty() && columns == 0 {
            return Ok(None);
        }

        let matrix = self.compact_matrix(rhs.len(), columns, &rows, &column_index)?;
        let (solution, independent_derived) = self.solve_compact(
            &matrix,
            &rhs,
            storage,
            &gauge_variables,
            &column_index,
            &rows,
        )?;
        let reduced = column_index
            .iter()
            .map(|column| column.map_or(0., |column| solution.as_slice()[column.0]))
            .collect();
        let original = self
            .bindings
            .iter()
            .map(|binding| match binding.variable {
                VariableBinding::Fixed(value) => Some(value),
                VariableBinding::Unknown(variable) => {
                    column_index[variable.0].map(|col| solution.as_slice()[col.0])
                }
            })
            .collect();
        Ok(Some(InstantaneousSolution {
            original,
            reduced,
            independent_derived,
        }))
    }

    fn compact_matrix(
        &self,
        row_count: usize,
        columns: usize,
        rows: &[Option<CompactRow>],
        column_index: &[Option<CompactColumn>],
    ) -> SpiceResult<SparseMatrix> {
        let mut matrix = SparseMatrix::new(row_count, columns);
        for triplet in self.matrix.triplets() {
            let Some(row) = rows[triplet.row] else {
                return Err(failure("nonempty row was not bound"));
            };
            let Some(col) = column_index[triplet.col] else {
                return Err(failure("nonempty column was not bound"));
            };
            matrix.add(row.0, col.0, triplet.value)?;
        }
        matrix.fold_duplicates();
        Ok(matrix)
    }

    fn solve_compact(
        &self,
        matrix: &SparseMatrix,
        rhs: &[Real],
        storage: &ImpulseStorage,
        gauge_variables: &[usize],
        column_index: &[Option<CompactColumn>],
        rows: &[Option<CompactRow>],
    ) -> SpiceResult<(Vector, BTreeSet<AbsoluteRow>)> {
        let derived: Vec<_> = self
            .equations
            .iter()
            .zip(rows)
            .filter_map(|(equation, row)| {
                if let (EquationOrigin::Derived(absolute), Some(row)) = (equation.origin, row) {
                    Some((absolute, row.0))
                } else {
                    None
                }
            })
            .collect();
        if !storage
            .charges
            .values()
            .any(|charge| charge.policy == InitialStoragePolicy::Derived)
        {
            match matrix.solve(&Vector::from_slice(rhs)) {
                Ok(solution) => return Ok((solution, BTreeSet::new())),
                Err(error) if storage.charges.is_empty() => {
                    return Err(failure(format!(
                        "uic initial conditions cannot be shown consistent at t = 0+ (the instantaneous constraint system is singular, possibly underdetermined or contradictory): {error}"
                    )));
                }
                Err(_) => {}
            }
        }
        // Mandatory algebraic and physical storage constraints win over
        // optional differentiator input values. A derived seed is retained
        // only if it provides an independent condition. Gauge coordinates
        // are introduced after all independent constraints.
        let gauges: Vec<_> = gauge_variables
            .iter()
            .filter_map(|&variable| column_index[variable].map(|column| column.0))
            .collect();
        let charges: Vec<_> = self
            .equations
            .iter()
            .zip(rows)
            .filter_map(|(equation, row)| {
                matches!(equation.origin, EquationOrigin::Charge(_))
                    .then_some(*row)
                    .flatten()
                    .map(|row| row.0)
            })
            .collect();
        let optional: Vec<_> = derived.iter().map(|(_, row)| *row).collect();
        let result = rank::complete(matrix, rhs, &gauges, &charges, &optional, matrix.cols())?;
        let retained = derived
            .into_iter()
            .filter_map(|(absolute, row)| result.independent[row].then_some(absolute))
            .collect();
        Ok((result.values, retained))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::netlist::Parser;
    use crate::netlist::source::parse_deck_text;
    use std::{collections::BTreeMap, path::Path};

    fn circuit() -> Circuit {
        let netlist = Parser::new()
            .parse_deck(&parse_deck_text(
                Path::new("instantaneous.cir"),
                "t\nr1 a 0 1k\nr2 b 0 1k\n.end\n",
            ))
            .unwrap();
        Circuit::from_netlist(&netlist).unwrap()
    }

    fn storage() -> ImpulseStorage {
        ImpulseStorage {
            fixed_coordinates: BTreeMap::new(),
            capacitors: Vec::new(),
            charges: BTreeMap::new(),
            local_charges: Vec::new(),
            path_voltages: Vec::new(),
        }
    }

    const TOL: Tolerances = Tolerances {
        reltol: 1e-3,
        vntol: 1e-6,
        abstol: 1e-12,
        chgtol: 1e-14,
    };

    #[test]
    fn exact_cancellation_is_compacted_before_classifying_empty_equations() {
        let circuit = circuit();
        let storage = storage();
        let mut matrix = SparseMatrix::new(2, 2);
        matrix.add(0, 0, 1.).unwrap();
        matrix.add(0, 0, -1.).unwrap();
        matrix.add(1, 1, 2.).unwrap();
        let rhs = Vector::from_slice(&[0., 6.]);
        let solution = InstantaneousSystem::assemble(&circuit, &matrix, &rhs, &storage, &[])
            .unwrap()
            .solve(&circuit, &storage, TOL)
            .unwrap()
            .unwrap();
        assert_eq!(solution.potential(VoltageVertex::Ground), Some(0.));
        assert_eq!(
            solution.potential(VoltageVertex::Node(AbsoluteRow(0))),
            None
        );
        assert_eq!(
            solution.potential(VoltageVertex::Node(AbsoluteRow(1))),
            Some(3.)
        );
        let rhs = Vector::from_slice(&[1., 6.]);
        let error = InstantaneousSystem::assemble(&circuit, &matrix, &rhs, &storage, &[])
            .unwrap()
            .solve(&circuit, &storage, TOL)
            .err()
            .unwrap()
            .to_string();
        assert!(
            error.contains("currents into node a") && error.contains("impulse"),
            "{error}"
        );
    }

    #[test]
    fn singular_but_consistent_constraints_are_not_diagnosed_as_an_impulse() {
        let circuit = circuit();
        let storage = storage();
        let mut matrix = SparseMatrix::new(2, 2);
        for row in 0..2 {
            for col in 0..2 {
                matrix.add(row, col, 1.).unwrap();
            }
        }
        let rhs = Vector::from_slice(&[2., 2.]);
        let error = InstantaneousSystem::assemble(&circuit, &matrix, &rhs, &storage, &[])
            .unwrap()
            .solve(&circuit, &storage, TOL)
            .err()
            .unwrap()
            .to_string();
        assert!(
            error.contains("cannot be shown consistent") && error.contains("singular"),
            "{error}"
        );
        assert!(!error.contains("impulse"), "{error}");
    }

    #[test]
    fn assembly_rejects_inconsistent_dimensions() {
        let circuit = circuit();
        assert!(
            InstantaneousSystem::assemble(
                &circuit,
                &SparseMatrix::new(3, 2),
                &Vector::zeros(2),
                &storage(),
                &[]
            )
            .is_err()
        );
        assert!(
            InstantaneousSystem::assemble(
                &circuit,
                &SparseMatrix::new(2, 2),
                &Vector::zeros(1),
                &storage(),
                &[]
            )
            .is_err()
        );
    }
}
