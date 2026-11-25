//! Ordered elimination of fixed storage coordinates and compact instantaneous MNA.
use super::constraints::{AbsoluteRow, VoltageVertex};
use super::{CapacitorInitial, ImpulseStorage, Tolerances, failure};
use crate::devices::Circuit;
use crate::maths::{SparseMatrix, Vector};
use crate::primitives::{Real, SpiceResult};

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
}

impl InstantaneousSolution {
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

impl<'a> InstantaneousSystem<'a> {
    pub(super) fn assemble(
        circuit: &Circuit,
        matrix: &SparseMatrix,
        rhs: &Vector,
        storage: &ImpulseStorage,
        kept: &[&'a CapacitorInitial],
    ) -> SpiceResult<Self> {
        let unknowns = circuit.unknown_count();
        if matrix.rows() != unknowns || matrix.cols() != unknowns || rhs.len() != unknowns {
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
        for (row, &kind) in is_branch.iter().enumerate() {
            let absolute = AbsoluteRow(row);
            let variable = if let Some(fixed) = storage.fixed_coordinates.get(&absolute) {
                VariableBinding::Fixed(fixed.value)
            } else {
                let binding = VariableBinding::Unknown(VariableId(variables));
                variables += 1;
                binding
            };
            let equation = if removed[row] {
                None
            } else {
                let equation = EquationId(equations.len());
                let origin = if kind {
                    EquationOrigin::Branch(absolute)
                } else {
                    EquationOrigin::Node(absolute)
                };
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
            if used {
                continue;
            }
            let absolute = match equation.origin {
                EquationOrigin::Node(_) => tolerance.abstol,
                EquationOrigin::Branch(_) | EquationOrigin::Capacitor(_) => tolerance.vntol,
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
        if rhs.len() != columns {
            return Err(failure(format!(
                "uic initial conditions cannot be shown consistent: the instantaneous system has \
                 {} equations for {columns} unknowns (over- or under-determined by ideal sources, \
                 capacitors and inductors)",
                rhs.len()
            )));
        }
        if rhs.is_empty() {
            return Ok(None);
        }

        let mut matrix = SparseMatrix::new(rhs.len(), columns);
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
        let solution = matrix.solve(&Vector::from_slice(&rhs)).map_err(|error| {
            failure(format!(
                "uic initial conditions cannot be shown consistent at t = 0+ (the \
                 instantaneous constraint system is singular, possibly underdetermined or \
                 contradictory): {error}"
            ))
        })?;
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
        Ok(Some(InstantaneousSolution { original }))
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
        }
    }

    const TOL: Tolerances = Tolerances {
        reltol: 1e-3,
        vntol: 1e-6,
        abstol: 1e-12,
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
