//! Indexed execution of the C-compatible symbolic value and derivatives.
//!
//! Intern instructions by exact operation and operand identity. Lower root
//! trees in evaluation order, keeping table data immutable, and only cache
//! finite successful results within a single evaluation environment.

use super::{
    Binary, Environment, EvaluationView, Node, Operator, PwlTable, Tree, Unary, XspicePwl, divide,
    power, power_h, pwr, unary_value,
};
use crate::primitives::{Real, SpiceError, SpiceResult};
use std::{collections::HashMap, sync::Arc};

mod derivative;
mod taylor;
#[cfg(test)]
use super::Evaluation;
pub(in crate::devices::behavioural) use taylor::TaylorEvaluation;

const MAX_INSTRUCTIONS: usize = 262_144;
/// Index of an instruction and its corresponding scratch value.
pub(super) type Slot = usize;

/// An instruction is also its exact interning key. Floating constants use
/// their bits (not floating equality); neither operand ordering nor operation
/// kind is changed. Table indices identify the original immutable table.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum Instruction {
    Constant(u64),
    Variable(usize),
    Time,
    Temperature,
    Frequency,
    Unary(Unary, Slot),
    Operator(Operator, Slot, Slot),
    Binary(Binary, Slot, Slot),
    Ternary(Slot, Slot, Slot),
    Pwl(Slot, usize),
    PwlDerivative(Slot, usize),
    Table(Slot, usize),
    TableSlope(Slot, usize),
}

/// Immutable, topologically ordered instructions with shared literal tables.
#[derive(Debug, Default, PartialEq)]
pub(super) struct Plan {
    instructions: Vec<Instruction>,
    has_ternary: bool,
    pwl: Vec<Arc<PwlTable>>,
    tables: Vec<Arc<XspicePwl>>,
}

/// Temporary structural interner; discarded when the program is finished.
#[derive(Default)]
pub(super) struct PlanBuilder {
    plan: Plan,
    instructions: HashMap<Instruction, Slot>,
    // The old tree-based regression oracle uses pointer identity only
    // while the corresponding tree allocations are alive.
    #[cfg(test)]
    source_slots: HashMap<*const Node, Slot>,
    #[cfg(test)]
    scratch_slots: HashMap<*const Node, Slot>,
    pwl: HashMap<*const PwlTable, usize>,
    tables: HashMap<*const XspicePwl, usize>,
}

fn table_index<T>(
    table: &Arc<T>,
    indices: &mut HashMap<*const T, usize>,
    tables: &mut Vec<Arc<T>>,
) -> usize {
    *indices.entry(Arc::as_ptr(table)).or_insert_with(|| {
        let index = tables.len();
        tables.push(Arc::clone(table));
        index
    })
}

impl PlanBuilder {
    pub(super) fn lower_value(&mut self, tree: &Tree) -> SpiceResult<Slot> {
        let root = self.lower(tree)?;
        #[cfg(test)]
        {
            self.source_slots = core::mem::take(&mut self.scratch_slots);
        }
        Ok(root)
    }

    #[cfg(test)]
    pub(super) fn lower_derivative(&mut self, tree: &Tree) -> SpiceResult<Slot> {
        let result = self.lower(tree);
        self.scratch_slots.clear();
        result
    }

    #[cfg(test)]
    pub(super) fn finish(self) -> Plan {
        self.plan
    }

    fn lower(&mut self, tree: &Tree) -> SpiceResult<Slot> {
        #[cfg(test)]
        let pointer = Arc::as_ptr(tree);
        #[cfg(test)]
        if let Some(&slot) = self
            .source_slots
            .get(&pointer)
            .or_else(|| self.scratch_slots.get(&pointer))
        {
            return Ok(slot);
        }
        let op = match &**tree {
            Node::Constant(value) => Instruction::Constant(value.to_bits()),
            Node::Variable(index) => Instruction::Variable(*index),
            Node::Time => Instruction::Time,
            Node::Temperature => Instruction::Temperature,
            Node::Frequency => Instruction::Frequency,
            Node::Unary(op, arg) => Instruction::Unary(*op, self.lower(arg)?),
            Node::Operator(op, a, b) => Instruction::Operator(*op, self.lower(a)?, self.lower(b)?),
            Node::Binary(op, a, b) => Instruction::Binary(*op, self.lower(a)?, self.lower(b)?),
            Node::Ternary(c, a, b) => {
                Instruction::Ternary(self.lower(c)?, self.lower(a)?, self.lower(b)?)
            }
            Node::Pwl(arg, table) => Instruction::Pwl(
                self.lower(arg)?,
                table_index(table, &mut self.pwl, &mut self.plan.pwl),
            ),
            Node::Table(arg, table) => Instruction::Table(
                self.lower(arg)?,
                table_index(table, &mut self.tables, &mut self.plan.tables),
            ),
            #[cfg(test)]
            Node::PwlDerivative(arg, table) => Instruction::PwlDerivative(
                self.lower(arg)?,
                table_index(table, &mut self.pwl, &mut self.plan.pwl),
            ),
            #[cfg(test)]
            Node::TableSlope(arg, table) => Instruction::TableSlope(
                self.lower(arg)?,
                table_index(table, &mut self.tables, &mut self.plan.tables),
            ),
        };
        let slot = self.intern(op)?;
        #[cfg(test)]
        self.scratch_slots.insert(pointer, slot);
        Ok(slot)
    }

    fn intern(&mut self, op: Instruction) -> SpiceResult<Slot> {
        let slot = if let Some(&slot) = self.instructions.get(&op) {
            slot
        } else {
            let slot = self.plan.instructions.len();
            if slot == MAX_INSTRUCTIONS {
                return Err(SpiceError::circuit(
                    "behavioural execution plan instruction limit exceeded",
                ));
            }
            self.plan.has_ternary |= matches!(op, Instruction::Ternary(..));
            self.plan.instructions.push(op);
            self.instructions.insert(op, slot);
            slot
        };
        Ok(slot)
    }
}

/// Per-evaluation scratch. The NaN marker is unambiguous because every
/// successful instruction yields a finite value. Capacity survives loads.
#[derive(Debug, Default)]
pub(in crate::devices::behavioural) struct Workspace {
    values: Vec<Real>,
    pending: Vec<Slot>,
    gradient: Vec<Real>,
}

// Scratch has no circuit state: a cloned device starts with empty scratch
// rather than copying obsolete cached values from its source device.
impl Clone for Workspace {
    fn clone(&self) -> Self {
        Self::default()
    }
}

impl Workspace {
    /// Transfer the one-shot evaluator's derivative buffer without copying.
    pub(super) fn into_gradient(self) -> Vec<Real> {
        self.gradient
    }
}

impl Plan {
    pub(super) fn evaluate<'w>(
        &self,
        root: Slot,
        derivatives: &[Slot],
        environment: &Environment<'_>,
        workspace: &'w mut Workspace,
    ) -> SpiceResult<EvaluationView<'w>> {
        workspace.values.resize(self.instructions.len(), Real::NAN);
        workspace.gradient.clear();
        if self.has_ternary {
            workspace.values.fill(Real::NAN);
            workspace.pending.clear();
        }
        let mut evaluator = Evaluator {
            plan: self,
            environment,
            fudge: environment.gmin * 1e-20,
            workspace,
        };
        let value = if self.has_ternary {
            evaluator.eval(root)?
        } else {
            for (slot, instruction) in self.instructions.iter().enumerate() {
                evaluator.workspace.values[slot] = evaluator.eval_ready(instruction)?;
            }
            evaluator.workspace.values[root]
        };
        evaluator.workspace.gradient.reserve(derivatives.len());
        for &slot in derivatives {
            let derivative = if self.has_ternary {
                evaluator.eval(slot)?
            } else {
                evaluator.workspace.values[slot]
            };
            evaluator.workspace.gradient.push(derivative);
        }
        Ok(EvaluationView {
            value,
            gradient: &evaluator.workspace.gradient,
        })
    }
}

struct Evaluator<'a, 'workspace> {
    plan: &'a Plan,
    environment: &'a Environment<'a>,
    fudge: Real,
    workspace: &'workspace mut Workspace,
}

impl Instruction {
    /// Return the first unmet dependency in the original traversal order.
    /// Never request a ternary's unselected branch.
    fn missing_input(self, values: &[Real]) -> Option<Slot> {
        let missing = |slot: Slot| values[slot].is_nan().then_some(slot);
        match self {
            Self::Unary(_, arg)
            | Self::Pwl(arg, _)
            | Self::PwlDerivative(arg, _)
            | Self::Table(arg, _)
            | Self::TableSlope(arg, _) => missing(arg),
            Self::Operator(_, a, b) | Self::Binary(_, a, b) => missing(a).or_else(|| missing(b)),
            Self::Ternary(c, a, b) => {
                missing(c).or_else(|| missing(if values[c] == 0. { b } else { a }))
            }
            _ => None,
        }
    }
}

fn out_of_range(node: &Instruction, arguments: &[Real]) -> SpiceError {
    let arguments: Vec<String> = arguments.iter().map(ToString::to_string).collect();
    SpiceError::Numerical {
        context: "behavioural source".into(),
        message: format!(
            "{} out of range for {}",
            arguments.join(", "),
            describe(node)
        ),
    }
}

impl Evaluator<'_, '_> {
    /// `PTeval` (`ifeval.c`): a `HUGE` function or operator result is an
    /// error, as in C; any other non-finite value is one too (C would carry
    /// on with a NaN or an infinity).
    fn eval(&mut self, root: Slot) -> SpiceResult<Real> {
        self.workspace.pending.push(root);
        while let Some(&slot) = self.workspace.pending.last() {
            if self.workspace.values[slot].is_nan() {
                let instruction = self.plan.instructions[slot];
                if let Some(input) = instruction.missing_input(&self.workspace.values) {
                    self.workspace.pending.push(input);
                    continue;
                }
                self.workspace.values[slot] = self.eval_ready(&instruction)?;
            }
            self.workspace.pending.pop();
        }
        Ok(self.workspace.values[root])
    }

    fn eval_ready(&self, node: &Instruction) -> SpiceResult<Real> {
        let read = |slot: &Slot| self.workspace.values[*slot];
        let value = match node {
            Instruction::Constant(bits) => Real::from_bits(*bits),
            Instruction::Variable(index) => self.environment.values[*index],
            Instruction::Time => self.environment.time,
            Instruction::Temperature => self.environment.temperature,
            Instruction::Frequency => self.environment.frequency,
            Instruction::Unary(function, argument) => {
                let x = read(argument);
                let value = unary_value(*function, x);
                if value == Real::INFINITY {
                    return Err(out_of_range(node, &[x]));
                }
                value
            }
            Instruction::Operator(op, lhs, rhs) => {
                let a = read(lhs);
                let b = read(rhs);
                let value = match op {
                    Operator::Plus => a + b,
                    Operator::Minus => a - b,
                    Operator::Times => a * b,
                    Operator::Divide => divide(a, b, self.fudge),
                    Operator::Power => power_h(a, b),
                };
                if value == Real::INFINITY {
                    return Err(out_of_range(node, &[a, b]));
                }
                value
            }
            Instruction::Binary(function, lhs, rhs) => {
                let a = read(lhs);
                let b = read(rhs);
                let value = match function {
                    Binary::Pow => power(a, b),
                    Binary::Pwr => pwr(a, b),
                    Binary::PowBaseSlope => super::power_base_slope(a, b, false),
                    Binary::PwrBaseSlope => super::power_base_slope(a, b, true),
                    Binary::PowExponentSlope => super::power_exponent_slope(a, b, false),
                    Binary::PwrExponentSlope => super::power_exponent_slope(a, b, true),
                    Binary::Min => {
                        if a > b {
                            b
                        } else {
                            a
                        }
                    }
                    Binary::Max => {
                        if a > b {
                            a
                        } else {
                            b
                        }
                    }
                };
                if value == Real::INFINITY {
                    return Err(out_of_range(node, &[a, b]));
                }
                value
            }
            Instruction::Ternary(condition, then, otherwise) => {
                if read(condition) == 0. {
                    read(otherwise)
                } else {
                    read(then)
                }
            }
            Instruction::Pwl(argument, table) => self.plan.pwl[*table].value(read(argument)),
            Instruction::PwlDerivative(argument, table) => {
                self.plan.pwl[*table].slope(read(argument))
            }
            Instruction::Table(argument, map) => self.plan.tables[*map].evaluate(read(argument)).0,
            Instruction::TableSlope(argument, map) => {
                self.plan.tables[*map].evaluate(read(argument)).1
            }
        };
        if !value.is_finite() {
            return Err(SpiceError::Numerical {
                context: "behavioural source".into(),
                message: format!("non-finite value {value} in {}", describe(node)),
            });
        }
        Ok(value)
    }
}

fn describe(node: &Instruction) -> String {
    match node {
        Instruction::Constant(_) => "a constant".into(),
        Instruction::Variable(_) => "a circuit quantity".into(),
        Instruction::Time => "time".into(),
        Instruction::Temperature => "temper".into(),
        Instruction::Frequency => "hertz".into(),
        Instruction::Unary(function, _) => format!("{}()", function.name()),
        Instruction::Operator(op, _, _) => format!(
            "'{}'",
            match op {
                Operator::Plus => "+",
                Operator::Minus => "-",
                Operator::Times => "*",
                Operator::Divide => "/",
                Operator::Power => "^",
            }
        ),
        Instruction::Binary(function, _, _) => format!(
            "{}()",
            match function {
                Binary::Pow => "pow",
                Binary::Pwr => "pwr",
                Binary::PowBaseSlope => "pow_base_slope",
                Binary::PwrBaseSlope => "pwr_base_slope",
                Binary::PowExponentSlope => "pow_exponent_slope",
                Binary::PwrExponentSlope => "pwr_exponent_slope",
                Binary::Min => "min",
                Binary::Max => "max",
            }
        ),
        Instruction::Ternary(..) => "'?:'".into(),
        Instruction::Pwl(..) => "pwl()".into(),
        Instruction::PwlDerivative(..) => "pwl_derivative()".into(),
        Instruction::Table(..) | Instruction::TableSlope(..) => "a TABLE".into(),
    }
}

#[cfg(test)]
mod tests {
    use super::super::{
        Compiler, MAX_SOURCE_DEPTH, MAX_SOURCE_NODES, Program, compile, constant, differentiate,
    };
    use super::*;
    use crate::netlist::bexpr::{BExpr, BExprKind};

    fn parse(text: &str) -> BExpr {
        use crate::netlist::Parser;
        use crate::primitives::SourceLoc;
        Parser::new()
            .parse_behavioural_expression(
                text,
                &SourceLoc::new(std::path::PathBuf::from("dag.cir"), 1, 1),
                true,
            )
            .unwrap()
            .root
    }

    fn environment(values: &[Real]) -> Environment<'_> {
        Environment {
            values,
            time: 0.25,
            temperature: 27.,
            gmin: 1e-12,
            frequency: 50.,
        }
    }

    // Reference traversal: always visit every operand of every reached root
    // in the old tree order, even if its slot was visited before. Only the
    // unchanged numerical instruction kernel is shared with the optimized path.
    fn uncached(
        evaluator: &mut Evaluator<'_, '_>,
        slot: Slot,
        visits: &mut usize,
    ) -> SpiceResult<Real> {
        *visits += 1;
        let instruction = evaluator.plan.instructions[slot];
        match instruction {
            Instruction::Unary(_, arg)
            | Instruction::Pwl(arg, _)
            | Instruction::PwlDerivative(arg, _)
            | Instruction::Table(arg, _)
            | Instruction::TableSlope(arg, _) => {
                uncached(evaluator, arg, visits)?;
            }
            Instruction::Operator(_, a, b) | Instruction::Binary(_, a, b) => {
                uncached(evaluator, a, visits)?;
                uncached(evaluator, b, visits)?;
            }
            Instruction::Ternary(c, a, b) => {
                let condition = uncached(evaluator, c, visits)?;
                uncached(evaluator, if condition == 0. { b } else { a }, visits)?;
            }
            _ => {}
        }
        let value = evaluator.eval_ready(&instruction)?;
        evaluator.workspace.values[slot] = value;
        Ok(value)
    }

    fn reference(
        program: &Program,
        environment: &Environment<'_>,
        workspace: &mut Workspace,
        visits: &mut usize,
    ) -> SpiceResult<Evaluation> {
        workspace
            .values
            .resize(program.plan.instructions.len(), Real::NAN);
        let mut evaluator = Evaluator {
            plan: &program.plan,
            environment,
            fudge: environment.gmin * 1e-20,
            workspace,
        };
        let value = uncached(&mut evaluator, program.root, visits)?;
        let gradient = program
            .derivatives
            .iter()
            .map(|&slot| uncached(&mut evaluator, slot, visits))
            .collect::<SpiceResult<Vec<_>>>()?;
        Ok(Evaluation { value, gradient })
    }

    fn bits(evaluation: &Evaluation) -> Vec<u64> {
        core::iter::once(evaluation.value)
            .chain(evaluation.gradient.iter().copied())
            .map(Real::to_bits)
            .collect()
    }

    fn tree_compiled(expr: &BExpr) -> Program {
        let mut compiler = Compiler::default();
        let value = Arc::new(compile(expr, &mut compiler).unwrap());
        let mut builder = PlanBuilder::default();
        let root = builder.lower_value(&value).unwrap();
        let derivatives = (0..compiler.quantities.len())
            .map(|variable| {
                builder
                    .lower_derivative(&differentiate(&value, variable))
                    .unwrap()
            })
            .collect();
        Program {
            plan: Arc::new(builder.finish()),
            root,
            derivatives,
            quantities: compiler.quantities,
            uses_frequency: compiler.uses_frequency,
            uses_time: compiler.uses_time,
        }
    }

    #[test]
    fn shared_plan_matches_uncached_value_gradient_and_error_order() {
        for text in [
            "pow(v(a),v(b))+pwr(v(a),v(b))+v(a)^v(b)",
            "min(v(a),v(b))+max(v(a),v(b))",
            "v(a)>0 ? sqrt(v(a)) : sqrt(-v(a))",
            "v(a)>0 ? log(-1) : v(b)*v(b)",
            "pwl(v(a)+v(b),0,0,1,2,2,1)",
            "pwl(v(a),2,1,1,2,0,0)+v(b)",
            "abs(v(a))+u(v(b))+floor(v(a))+hertz*v(b)",
            "sqrt(v(a))+log(v(b))",
            "v(a)*0+v(b)",
            "v(a)/(v(b)-v(b))",
            "v(a)+v(b)*(-0.0)",
            "v(a)>0 ? v(b) : sqrt(0)",
            "v(a)>0 ? sqrt(0) : v(b)",
            "v(a)+sqrt(0)",
            "v(a)+pow(0,v(b))",
            "v(a)+pwr(-2,v(b))",
            "v(a)+(-v(b))",
            "v(a)>0 ? min(v(a),v(b)) : max(v(a),v(b))",
            "v(a)>0 ? v(b) : -v(b)",
        ] {
            let expr = parse(text);
            let program = Program::compile(&expr).unwrap();
            let old = tree_compiled(&expr);
            assert_eq!(program.plan, old.plan, "{text}");
            assert_eq!(program.derivatives, old.derivatives, "{text}");
            let mut scratch = Workspace::default();
            let mut reference_scratch = Workspace::default();
            for point in [[0., -0.], [-0., 0.], [1., 1.], [-2., 2.], [0.25, -0.5]] {
                let env = environment(&point[..program.quantities().len()]);
                let actual = program.evaluate_with_workspace(&env, &mut scratch);
                let expected = reference(&old, &env, &mut reference_scratch, &mut 0);
                match (actual, expected) {
                    (Ok(actual), Ok(expected)) => {
                        assert_eq!(
                            bits(&Evaluation {
                                value: actual.value,
                                gradient: actual.gradient.to_vec()
                            }),
                            bits(&expected),
                            "{text} at {point:?}"
                        )
                    }
                    (Err(actual), Err(expected)) => assert_eq!(
                        actual.to_string(),
                        expected.to_string(),
                        "{text} at {point:?}"
                    ),
                    (actual, expected) => panic!("{text} at {point:?}: {actual:?} != {expected:?}"),
                }
            }
        }
    }

    #[test]
    fn sparse_symbolic_unaries_match_tree_compilation() {
        for name in [
            "abs", "acos", "acosh", "asin", "asinh", "atan", "atanh", "cos", "cosh", "exp", "log",
            "log10", "sgn", "sin", "sinh", "sqrt", "tan", "tanh", "u", "uramp", "ceil", "floor",
            "nint",
        ] {
            for input in ["v(b)", "v(a)+v(b)", "0", "-1", "1"] {
                let text = format!("v(a)+{name}({input})");
                let expr = parse(&text);
                let old = tree_compiled(&expr);
                let new = Program::compile(&expr).unwrap();
                assert_eq!(old.plan, new.plan, "{text}");
                assert_eq!(old.derivatives, new.derivatives, "{text}");
                for point in [[-2., -1.], [0., -0.], [0.25, 0.5], [1., 2.]] {
                    let env = environment(&point[..new.quantities().len()]);
                    match (old.evaluate(&env), new.evaluate(&env)) {
                        (Ok(old), Ok(new)) => assert_eq!(bits(&old), bits(&new), "{text}"),
                        (Err(old), Err(new)) => {
                            assert_eq!(old.to_string(), new.to_string(), "{text}")
                        }
                        (old, new) => panic!("{text}: {old:?} != {new:?}"),
                    }
                }
            }
        }
    }

    #[test]
    #[ignore = "opt-in symbolic compiler wall time and derivative allocation benchmark"]
    fn sparse_and_dense_compilation_benchmarks() {
        let sparse = parse(&balanced_sum(
            (0..192).map(|n| format!("sin(v(n{n}))")).collect(),
        ));
        let mut dense = parse(&balanced_sum(
            (0..80).map(|n| format!("{}*v(n{n})", n + 1)).collect(),
        ));
        for _ in 0..32 {
            dense = BExpr {
                span: dense.span.clone(),
                kind: BExprKind::Call {
                    name: "sin".into(),
                    arguments: vec![dense],
                },
            };
        }
        for (name, expr) in [("sparse", sparse), ("dense", dense)] {
            let mut old_samples = Vec::new();
            let mut new_samples = Vec::new();
            let mut old_allocations = 0;
            let mut slots = 0;
            for _ in 0..7 {
                super::super::derivative_reference::take_allocations();
                let now = std::time::Instant::now();
                let old = std::hint::black_box(tree_compiled(std::hint::black_box(&expr)));
                old_samples.push(now.elapsed());
                old_allocations = super::super::derivative_reference::take_allocations();
                let now = std::time::Instant::now();
                let new =
                    std::hint::black_box(Program::compile(std::hint::black_box(&expr)).unwrap());
                new_samples.push(now.elapsed());
                assert_eq!(old.plan, new.plan, "{name}");
                assert_eq!(old.derivatives, new.derivatives, "{name}");
                slots = new.plan.instructions.len();
                let point = vec![0.125; new.quantities().len()];
                let actual = new.evaluate(&environment(&point)).unwrap();
                assert_eq!(
                    bits(&old.evaluate(&environment(&point)).unwrap()),
                    bits(&actual)
                );
                if name == "sparse" {
                    assert!((actual.value - 192. * 0.125_f64.sin()).abs() < 1e-11);
                    for derivative in &actual.gradient {
                        assert!((derivative - 0.125_f64.cos()).abs() < 1e-12);
                    }
                }
            }
            old_samples.sort_unstable();
            new_samples.sort_unstable();
            eprintln!(
                "{name}: old {:?}, new {:?}; old derivative Arc allocations {old_allocations}, new derivative Arcs 0; retained slots {slots}",
                old_samples[3], new_samples[3]
            );
        }
    }

    #[test]
    fn workspace_resets_after_error_and_table_branches_stay_lazy() {
        use crate::netlist::bexpr::TableTransfer;
        let input = parse("v(a)+v(b)");
        let table = BExpr {
            span: input.span.clone(),
            kind: BExprKind::Table(Box::new(TableTransfer {
                input,
                points: vec![
                    (parse("0"), parse("0")),
                    (parse("1"), parse("2")),
                    (parse("2"), parse("1")),
                ],
                domain: 0.1,
            })),
        };
        let mut expr = parse("v(a)>0 ? log(-1) : 0");
        let BExprKind::Ternary { otherwise, .. } = &mut expr.kind else {
            panic!("expected ternary");
        };
        **otherwise = table;
        let program = Program::compile(&expr).unwrap();
        let old = tree_compiled(&expr);
        assert_eq!(program.plan, old.plan);
        let mut workspace = Workspace::default();
        let mut reference_scratch = Workspace::default();
        for values in [[1., 0.], [0., 0.], [-1., 2.], [0., 2.], [1., 1.], [-1., 4.]] {
            let env = environment(&values);
            let actual = program.evaluate_with_workspace(&env, &mut workspace);
            let expected = reference(&old, &env, &mut reference_scratch, &mut 0);
            match (actual, expected) {
                (Ok(actual), Ok(expected)) => {
                    assert_eq!(
                        bits(&Evaluation {
                            value: actual.value,
                            gradient: actual.gradient.to_vec()
                        }),
                        bits(&expected)
                    )
                }
                (Err(actual), Err(expected)) => {
                    assert_eq!(actual.to_string(), expected.to_string())
                }
                (actual, expected) => panic!("{actual:?} != {expected:?}"),
            }
        }
    }

    #[test]
    fn instruction_interning_does_not_merge_signed_zero() {
        let positive = constant(0.);
        let negative = constant(-0.);
        let mut builder = PlanBuilder::default();
        assert_ne!(
            builder.lower(&positive).unwrap(),
            builder.lower(&negative).unwrap()
        );
    }

    fn balanced_sum(mut terms: Vec<String>) -> String {
        while terms.len() > 1 {
            let mut combined = Vec::with_capacity(terms.len().div_ceil(2));
            let mut previous = terms.into_iter();
            while let Some(left) = previous.next() {
                combined.push(if let Some(right) = previous.next() {
                    format!("({left}+{right})")
                } else {
                    left
                });
            }
            terms = combined;
        }
        terms.pop().unwrap()
    }

    #[test]
    fn deep_admissible_expression_has_bounded_execution_stack() {
        let mut expr = parse("v(x)");
        for _ in 1..MAX_SOURCE_DEPTH {
            expr = BExpr {
                span: expr.span.clone(),
                kind: BExprKind::Call {
                    name: "sin".into(),
                    arguments: vec![expr],
                },
            };
        }
        let program = Program::compile(&expr).unwrap();
        let env = environment(&[0.25]);
        let actual = program.evaluate(&env).unwrap();
        let mut scratch = Workspace::default();
        let expected = reference(&program, &env, &mut scratch, &mut 0).unwrap();
        assert_eq!(bits(&actual), bits(&expected));
    }

    #[test]
    fn source_and_derivative_budgets_reject_oversized_expressions() {
        let mut deep = parse("v(x)");
        for _ in 0..MAX_SOURCE_DEPTH {
            deep = BExpr {
                span: deep.span.clone(),
                kind: BExprKind::Group(Box::new(deep)),
            };
        }
        assert!(
            Program::compile(&deep)
                .unwrap_err()
                .to_string()
                .contains("source size/depth limit")
        );

        let wide = balanced_sum((0..512).map(|n| format!("v(n{n})")).collect());
        let program = Program::compile(&parse(&wide)).unwrap();
        let evaluation = program.evaluate(&environment(&[1.; 512])).unwrap();
        assert_eq!(evaluation.value, 512.);
        assert_eq!(evaluation.gradient, vec![1.; 512]);

        let mut dense = parse(&balanced_sum(
            (0..1024).map(|n| format!("{}*v(n{n})", n + 1)).collect(),
        ));
        for _ in 0..140 {
            dense = BExpr {
                span: dense.span.clone(),
                kind: BExprKind::Call {
                    name: "sin".into(),
                    arguments: vec![dense],
                },
            };
        }
        let error = Program::compile(&dense).unwrap_err().to_string();
        assert!(
            error.contains("derivative expansion limit") || error.contains("instruction limit"),
            "{error}"
        );

        let number = parse("0");
        let mut pwl = parse("v(x)");
        pwl.kind = BExprKind::Call {
            name: "pwl".into(),
            arguments: vec![number; MAX_SOURCE_NODES],
        };
        assert!(
            Program::compile(&pwl)
                .unwrap_err()
                .to_string()
                .contains("source size/depth limit")
        );
    }

    #[test]
    fn large_literal_table_is_not_multiplied_into_derivative_budget() {
        use crate::netlist::bexpr::TableTransfer;
        let input = parse(&balanced_sum((0..16).map(|n| format!("v(n{n})")).collect()));
        let points = (0..5_000)
            .map(|n| {
                let x = parse(&n.to_string());
                (x.clone(), x)
            })
            .collect();
        let expr = BExpr {
            span: input.span.clone(),
            kind: BExprKind::Table(Box::new(TableTransfer {
                input,
                points,
                domain: 0.1,
            })),
        };
        let program = Program::compile(&expr).unwrap();
        let values = [1.; 16];
        let result = program.evaluate(&environment(&values)).unwrap();
        assert_eq!(result.value, 16.);
        assert_eq!(result.gradient, vec![1.; 16]);
    }

    #[test]
    fn shared_plan_reduces_repeated_work_and_reuses_scratch() {
        // A representative multi-input nonlinear source: a common nonlinear
        // transfer is multiplied by many independent sensed quantities.
        let common = "exp(sin(v(x))+cos(v(y)))+pow(v(x)+2,v(y))";
        let wide = balanced_sum((0..24).map(|n| format!("v(n{n})*({common})")).collect());
        measure_workload("wide", &parse(&wide));
        let mut deep = parse(&balanced_sum((0..12).map(|n| format!("v(n{n})")).collect()));
        for _ in 0..60 {
            deep = BExpr {
                span: deep.span.clone(),
                kind: BExprKind::Call {
                    name: "sin".into(),
                    arguments: vec![deep],
                },
            };
        }
        measure_workload("deep", &deep);
    }

    fn retained_tree_nodes(tree: &Tree, seen: &mut std::collections::HashSet<*const Node>) {
        if !seen.insert(Arc::as_ptr(tree)) {
            return;
        }
        match &**tree {
            Node::Unary(_, arg)
            | Node::Pwl(arg, _)
            | Node::PwlDerivative(arg, _)
            | Node::Table(arg, _)
            | Node::TableSlope(arg, _) => retained_tree_nodes(arg, seen),
            Node::Operator(_, a, b) | Node::Binary(_, a, b) => {
                retained_tree_nodes(a, seen);
                retained_tree_nodes(b, seen);
            }
            Node::Ternary(c, a, b) => {
                retained_tree_nodes(c, seen);
                retained_tree_nodes(a, seen);
                retained_tree_nodes(b, seen);
            }
            _ => {}
        }
    }

    fn measure_workload(label: &str, expr: &BExpr) {
        let started = std::time::Instant::now();
        let mut compiler = Compiler::default();
        let old_root = Arc::new(compile(expr, &mut compiler).unwrap());
        let old_derivatives: Vec<_> = (0..compiler.quantities.len())
            .map(|variable| differentiate(&old_root, variable))
            .collect();
        std::hint::black_box((&old_root, &old_derivatives));
        let old_compile_elapsed = started.elapsed();
        let mut old_nodes = std::collections::HashSet::new();
        for tree in core::iter::once(&old_root).chain(&old_derivatives) {
            retained_tree_nodes(tree, &mut old_nodes);
        }
        let started = std::time::Instant::now();
        let program = Program::compile(expr).unwrap();
        let compile_elapsed = started.elapsed();
        let inputs = vec![0.25; program.quantities().len()];
        let env = environment(&inputs);
        let mut scratch = Workspace::default();
        let mut reference_scratch = Workspace::default();
        let mut visits = 0;
        let expected = reference(&program, &env, &mut reference_scratch, &mut visits).unwrap();
        let actual = program.evaluate_with_workspace(&env, &mut scratch).unwrap();
        assert_eq!(
            bits(&Evaluation {
                value: actual.value,
                gradient: actual.gradient.to_vec()
            }),
            bits(&expected)
        );
        let unique = scratch
            .values
            .iter()
            .filter(|value| value.is_finite())
            .count();
        assert!(unique * 4 < visits, "{label}: {unique} vs {visits}");
        let capacity = (scratch.values.capacity(), scratch.pending.capacity());
        let started = std::time::Instant::now();
        for _ in 0..200 {
            std::hint::black_box(program.evaluate_with_workspace(&env, &mut scratch).unwrap());
        }
        let cached_elapsed = started.elapsed();
        let started = std::time::Instant::now();
        for _ in 0..200 {
            std::hint::black_box(
                reference(&program, &env, &mut reference_scratch, &mut 0).unwrap(),
            );
        }
        let uncached_elapsed = started.elapsed();
        assert_eq!(
            (scratch.values.capacity(), scratch.pending.capacity()),
            capacity
        );
        eprintln!(
            "behavioural DAG ({label}): compile {compile_elapsed:?} indexed vs {old_compile_elapsed:?} tree-only; {} retained slots vs {} individually allocated tree nodes; {unique} vs {visits} visits; 200 loads {cached_elapsed:?} indexed vs {uncached_elapsed:?} uncached",
            program.plan.instructions.len(),
            old_nodes.len()
        );
    }
}
