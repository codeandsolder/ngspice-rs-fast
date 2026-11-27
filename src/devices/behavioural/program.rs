//! Compiled behavioural expressions: value and analytic derivatives.
//!
//! C references (behaviour only): `src/spicelib/parser/inpptree.c`
//! (`PT_mkfnode`, `PTdifferentiate`, `prepare_PTF_PWL`), `ptfuncs.c` (the
//! function values and domain rules) and `ifeval.c` (`PTeval`). Like ngspice,
//! the port keeps the parse tree unsimplified, differentiates it symbolically
//! once per quantity with analytic rules for the actual value kernels and
//! constant folding/zero-one identities. The evaluation retains SPICE
//! value semantics, but does not reproduce inconsistent Jacobians. Sparse symbolic
//! partials are built directly in an indexed DAG; successful finite results are
//! reused only within one environment, with lazy branch traversal. That includes the deliberate non-derivatives
//! (`floor`, comparisons, `u` and `pwl_derivative` are zero or piecewise) and
//! the domain quirks of the value functions:
//!
//! - division adds `gmin * 1e-20` to the divisor away from zero (`PTdivide`),
//!   so `1/0` is `1e32` with the default gmin;
//! - `exp(x)` is `1e99` above `x = 227.9559242`; `log(0)` and `log10(0)` are
//!   `-1e99`; `log`, `log10` and `sqrt` of a negative number are errors;
//! - `sin`, `cos` and `tan` reduce their argument with C's
//!   `x - (int)(x / 2pi) * 2pi` (`pi` for `tan`) first;
//! - `^` is `pow(a, b)` for `a >= 0`, `pow(a, round(b))` for `a < 0` and a
//!   quasi-integer `b` (10 ulps), otherwise 0 (`PTpowerH`, default
//!   compatibility); `pow()` is the same except `pow(0, b) = 0` (`PTpower`);
//!   `pwr(a, b) = sign(a) |a|^b`;
//! - `u(0) = 0.5`; `nint` rounds half to even; `pwl()` extrapolates the end
//!   segments linearly, and its derivative uses the same selected segment as
//!   the value, including descending tables.
//!
//! Where C silently continues with a NaN (for example `acos(2)`) or an
//! infinity, the port stops with an explicit numerical error naming the
//! function. The PSPICE/HSPICE/LTspice compatibility variants (`newcompat`)
//! are not modelled: the reference binary runs without a compatibility mode.

use crate::netlist::bexpr::{BBinaryOp, BExpr, BExprKind, BUnaryOp, STATISTICAL_FUNCTIONS};
use crate::primitives::{Real, SpiceError, SpiceResult, almost_equal_ulps};

use std::collections::HashMap;
use std::sync::Arc;

use super::xspice::XspicePwl;

mod plan;
use plan::{Plan, PlanBuilder, Slot};
pub(super) use plan::{TaylorEvaluation, Workspace};

/// A circuit quantity an expression reads, in first-use order (C
/// `mkvnode`/`mkinode` deduplicate them the same way).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Quantity {
    /// A node voltage, by canonical node name.
    Node(String),
    /// The branch current of a named device (`i(name)`).
    Branch(String),
}

/// What an evaluation reads besides the variables.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Environment<'a> {
    /// Present values of the [`Program::quantities`], in order.
    pub values: &'a [Real],
    /// `time` in seconds (0 outside transient analysis, as `CKTtime`).
    pub time: Real,
    /// `temper`: the circuit temperature in degrees Celsius.
    pub temperature: Real,
    /// The circuit gmin; `PTdivide` adds `gmin * 1e-20` to divisors.
    pub gmin: Real,
    /// `hertz`: the analysis frequency in Hz (C `CKTomega / 2 pi`).
    pub frequency: Real,
}

/// A value and its partial derivatives with respect to each quantity.
#[derive(Debug, Clone, PartialEq)]
pub struct Evaluation {
    /// The expression value.
    pub value: Real,
    /// `d value / d quantity[i]`.
    pub gradient: Vec<Real>,
}

/// Workspace-backed value and derivatives, borrowed until the next evaluation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct EvaluationView<'a> {
    pub(super) value: Real,
    pub(super) gradient: &'a [Real],
}

/// One-argument functions (`inpptree.c` `funcs[]`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum Unary {
    Abs,
    Acos,
    Acosh,
    Asin,
    Asinh,
    Atan,
    Atanh,
    Cos,
    Cosh,
    Exp,
    Log,
    Log10,
    Sgn,
    Sin,
    Sinh,
    Sqrt,
    Tan,
    Tanh,
    Ustep,
    Uramp,
    Ceil,
    Floor,
    Nint,
    Uminus,
    Ustep2,
    Eq0,
    Ne0,
    Gt0,
    Lt0,
    Ge0,
    Le0,
}

impl Unary {
    fn from_name(name: &str) -> Option<Self> {
        Some(match name {
            "abs" => Self::Abs,
            "acos" => Self::Acos,
            "acosh" => Self::Acosh,
            "asin" => Self::Asin,
            "asinh" => Self::Asinh,
            "atan" => Self::Atan,
            "atanh" => Self::Atanh,
            "cos" => Self::Cos,
            "cosh" => Self::Cosh,
            "exp" => Self::Exp,
            "ln" | "log" => Self::Log,
            "log10" => Self::Log10,
            "sgn" => Self::Sgn,
            "sin" => Self::Sin,
            "sinh" => Self::Sinh,
            "sqrt" => Self::Sqrt,
            "tan" => Self::Tan,
            "tanh" => Self::Tanh,
            "u" => Self::Ustep,
            "uramp" => Self::Uramp,
            "ceil" => Self::Ceil,
            "floor" => Self::Floor,
            "nint" => Self::Nint,
            "u2" => Self::Ustep2,
            "eq0" => Self::Eq0,
            "ne0" => Self::Ne0,
            "gt0" => Self::Gt0,
            "lt0" => Self::Lt0,
            "ge0" => Self::Ge0,
            "le0" => Self::Le0,
            _ => return None,
        })
    }

    const fn name(self) -> &'static str {
        match self {
            Self::Abs => "abs",
            Self::Acos => "acos",
            Self::Acosh => "acosh",
            Self::Asin => "asin",
            Self::Asinh => "asinh",
            Self::Atan => "atan",
            Self::Atanh => "atanh",
            Self::Cos => "cos",
            Self::Cosh => "cosh",
            Self::Exp => "exp",
            Self::Log => "log",
            Self::Log10 => "log10",
            Self::Sgn => "sgn",
            Self::Sin => "sin",
            Self::Sinh => "sinh",
            Self::Sqrt => "sqrt",
            Self::Tan => "tan",
            Self::Tanh => "tanh",
            Self::Ustep => "u",
            Self::Uramp => "uramp",
            Self::Ceil => "ceil",
            Self::Floor => "floor",
            Self::Nint => "nint",
            Self::Uminus => "-",
            Self::Ustep2 => "u2",
            Self::Eq0 => "eq0",
            Self::Ne0 => "ne0",
            Self::Gt0 => "gt0",
            Self::Lt0 => "lt0",
            Self::Ge0 => "ge0",
            Self::Le0 => "le0",
        }
    }
}

/// Binary operators (`PT_PLUS` ... `PT_POWER`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum Operator {
    Plus,
    Minus,
    Times,
    Divide,
    Power,
}

/// Two-argument functions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum Binary {
    Power(PowerLaw, PowerPart),
    Min,
    Max,
}

/// Separate numerical laws selected explicitly by the prepared expression.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum PowerLaw {
    NativePow,
    NativePwr,
    SimetrixAbsolute,
    SimetrixCaret,
    SimetrixRealPart,
}

/// Value and the two analytic partials share one typed numerical law.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum PowerPart {
    Value,
    Base,
    Exponent,
}

impl PowerPart {
    const fn suffix(self) -> &'static str {
        match self {
            Self::Value => "",
            Self::Base => "_base_slope",
            Self::Exponent => "_exponent_slope",
        }
    }
}

impl PowerLaw {
    fn from_name(name: &str) -> Option<Self> {
        match name {
            "pow" => Some(Self::NativePow),
            "pwr" => Some(Self::NativePwr),
            "simetrix_abs_pow" => Some(Self::SimetrixAbsolute),
            "simetrix_caret_pow" => Some(Self::SimetrixCaret),
            "simetrix_pow" => Some(Self::SimetrixRealPart),
            _ => None,
        }
    }

    const fn name(self) -> &'static str {
        match self {
            Self::NativePow => "pow",
            Self::NativePwr => "pwr",
            Self::SimetrixAbsolute => "simetrix_abs_pow",
            Self::SimetrixCaret => "simetrix_caret_pow",
            Self::SimetrixRealPart => "simetrix_pow",
        }
    }

    fn evaluate(self, part: PowerPart, a: Real, b: Real) -> Real {
        match self {
            Self::NativePow | Self::NativePwr => {
                let signed = self == Self::NativePwr;
                match part {
                    PowerPart::Value if signed => pwr(a, b),
                    PowerPart::Value => power(a, b),
                    PowerPart::Base => power_base_slope(a, b, signed),
                    PowerPart::Exponent => power_exponent_slope(a, b, signed),
                }
            }
            Self::SimetrixAbsolute => absolute_power(part, a, b),
            Self::SimetrixCaret => caret_power(part, a, b),
            Self::SimetrixRealPart => real_part_power(part, a, b),
        }
    }
}

/// A shared subtree: C shares subtrees between the function and its
/// derivative trees (`PTdifferentiate`, `inc_usage`).
type Tree = Arc<Node>;

/// A parse-tree node (`INPparseNode`).
#[derive(Debug, Clone, PartialEq)]
enum Node {
    Constant(Real),
    Variable(usize),
    Time,
    Temperature,
    Frequency,
    Unary(Unary, Tree),
    Operator(Operator, Tree, Tree),
    Binary(Binary, Tree, Tree),
    Ternary(Tree, Tree, Tree),
    /// `pwl(x, x0, y0, x1, y1, ...)`: shared immutable literal points.
    Pwl(Tree, Arc<PwlTable>),
    /// `PTF_PWL_DERIVATIVE`: the slope `PTdifferentiate` builds for `pwl()`.
    #[cfg(test)]
    PwlDerivative(Tree, Arc<PwlTable>),
    /// The XSPICE `pwl` transfer of a lowered TABLE.
    Table(Tree, Arc<XspicePwl>),
    /// The slope of [`Node::Table`] (the code model's partial).
    #[cfg(test)]
    TableSlope(Tree, Arc<XspicePwl>),
}

#[cfg(test)]
impl Node {
    const fn constant(&self) -> Option<Real> {
        match self {
            Self::Constant(value) => Some(*value),
            _ => None,
        }
    }
}

// Bound recursive source lowering before constructing the value tree. Sparse
// derivative work and intermediate instruction budgets are enforced during lowering.
const MAX_SOURCE_DEPTH: usize = crate::netlist::expr::MAX_TREE_DEPTH;
const MAX_SOURCE_NODES: usize = crate::netlist::expr::MAX_TREE_NODES;

fn check_source_limits(root: &BExpr) -> SpiceResult<()> {
    let mut remaining = MAX_SOURCE_NODES;
    root.walk_bounded(
        1,
        &mut remaining,
        &mut |node, depth, remaining| {
            if depth > MAX_SOURCE_DEPTH || *remaining == 0 {
                return Err(unsupported(
                    node,
                    "behavioural expression source size/depth limit exceeded",
                ));
            }
            *remaining -= 1;
            Ok(())
        },
        // An unevaluated Value is rejected by compile(); its numparam tree
        // is not lowered into device instructions, so it cannot consume the
        // device compiler's source budget. Front-end AST construction counts
        // it against the shared TreeBudget before elaboration instead.
        &mut |_, _, _| Ok(()),
    )
}

/// A compiled behavioural value and its symbolic partial derivatives,
/// sharing one immutable, indexed execution plan.
#[derive(Debug, Clone, PartialEq)]
pub struct Program {
    plan: Arc<Plan>,
    root: Slot,
    derivatives: Vec<Slot>,
    quantities: Vec<Quantity>,
    uses_frequency: bool,
    uses_time: bool,
}

fn unsupported(expr: &BExpr, message: impl Into<String>) -> SpiceError {
    SpiceError::Unsupported {
        feature: message.into(),
        location: Some(expr.span.start.clone()),
    }
}

fn parse_error(expr: &BExpr, message: impl Into<String>) -> SpiceError {
    SpiceError::parse(expr.span.start.clone(), message)
}

impl Program {
    /// Compiles a front-end-resolved expression tree and lowers its sparse
    /// symbolic derivatives directly to the shared execution plan.
    ///
    /// # Errors
    /// Names the front end should have resolved (`.param` values, `.func`
    /// calls), unknown functions or wrong argument counts (positioned parse
    /// errors), non-literal or non-monotonic `pwl()` points (C:
    /// `prepare_PTF_PWL`), and [`SpiceError::NotYetPorted`] for `ddt()` and
    /// the statistical functions (`agauss`, `gauss`, `aunif`, `unif`,
    /// `limit`; must be realized by the front end). Expressions
    /// exceeding the source depth, size or symbolic expansion limits are
    /// explicitly rejected before unbounded work can occur.
    pub fn compile(expr: &BExpr) -> SpiceResult<Self> {
        check_source_limits(expr)?;
        let mut compiler = Compiler::default();
        let value_tree = Arc::new(compile(expr, &mut compiler)?);
        let mut builder = PlanBuilder::default();
        let root = builder.lower_value(&value_tree)?;
        let derivatives = builder.sparse_derivatives(root, compiler.quantities.len(), expr)?;
        let (plan, root, derivatives) = builder.finish_reachable(root, derivatives);
        Ok(Self {
            plan: Arc::new(plan),
            root,
            derivatives,
            quantities: compiler.quantities,
            uses_frequency: compiler.uses_frequency,
            uses_time: compiler.uses_time,
        })
    }

    /// True when the expression reads `hertz`.
    #[must_use = "the result identifies expressions that depend on analysis frequency"]
    pub const fn uses_frequency(&self) -> bool {
        self.uses_frequency
    }

    /// True when the expression reads explicit transient time.
    #[must_use = "the result identifies expressions that must be reevaluated as time advances"]
    pub const fn uses_time(&self) -> bool {
        self.uses_time
    }

    /// The circuit quantities the expression reads, in variable order.
    #[must_use = "the returned order binds expression variables to circuit quantities"]
    pub fn quantities(&self) -> &[Quantity] {
        &self.quantities
    }

    /// The value and every partial derivative at `environment`: `IFeval`
    /// evaluates the value tree, then each derivative tree (`PTeval`).
    ///
    /// # Errors
    /// [`SpiceError::Numerical`] when a function is evaluated outside its
    /// domain or any value/derivative is not finite.
    pub fn evaluate(&self, environment: &Environment<'_>) -> SpiceResult<Evaluation> {
        let mut workspace = Workspace::default();
        let value = self
            .evaluate_with_workspace(environment, &mut workspace)?
            .value;
        Ok(Evaluation {
            value,
            gradient: workspace.into_gradient(),
        })
    }

    pub(super) fn evaluate_taylor(
        &self,
        environment: &Environment<'_>,
        variables: &[crate::maths::taylor::Jet],
        time: &crate::maths::taylor::Jet,
        collect_guards: bool,
    ) -> SpiceResult<TaylorEvaluation> {
        if variables.len() != self.quantities.len() {
            return Err(SpiceError::circuit(
                "wrong number of behavioural Taylor variables",
            ));
        }
        self.plan
            .evaluate_taylor(self.root, environment, variables, time, collect_guards)
    }

    pub(super) fn evaluate_with_workspace<'w>(
        &self,
        environment: &Environment<'_>,
        workspace: &'w mut Workspace,
    ) -> SpiceResult<EvaluationView<'w>> {
        if environment.values.len() != self.quantities.len() {
            return Err(SpiceError::circuit(
                "behavioural expression evaluated with the wrong number of values",
            ));
        }
        self.plan
            .evaluate(self.root, &self.derivatives, environment, workspace)
    }
}

/// Borrowed AST names are scoped to compilation, not the resulting program.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum QuantityKey<'a> {
    Node(&'a str),
    Branch(&'a str),
}

/// Intern circuit quantities in first-use order and collect expression-wide
/// metadata in the same traversal. The lookup index is discarded after
/// lowering; only the ordered owned quantities survive in the program.
#[derive(Default)]
struct Compiler<'a> {
    indices: HashMap<QuantityKey<'a>, usize>,
    quantities: Vec<Quantity>,
    uses_frequency: bool,
    uses_time: bool,
}

impl<'a> Compiler<'a> {
    fn variable(&mut self, key: QuantityKey<'a>) -> Node {
        let index = *self.indices.entry(key).or_insert_with(|| {
            let index = self.quantities.len();
            self.quantities.push(match key {
                QuantityKey::Node(name) => Quantity::Node(name.to_owned()),
                QuantityKey::Branch(name) => Quantity::Branch(name.to_owned()),
            });
            index
        });
        Node::Variable(index)
    }
}

fn boxed<'a>(expr: &'a BExpr, compiler: &mut Compiler<'a>) -> SpiceResult<Tree> {
    compile(expr, compiler).map(Arc::new)
}

fn compile<'a>(expr: &'a BExpr, compiler: &mut Compiler<'a>) -> SpiceResult<Node> {
    Ok(match &expr.kind {
        BExprKind::Number { value, .. } => Node::Constant(*value),
        BExprKind::Name(name) => match name.as_str() {
            "time" => {
                compiler.uses_time = true;
                Node::Time
            }
            "temper" => Node::Temperature,
            "pi" => Node::Constant(std::f64::consts::PI),
            "e" => Node::Constant(std::f64::consts::E),
            "hertz" => {
                compiler.uses_frequency = true;
                Node::Frequency
            }
            other => {
                return Err(parse_error(
                    expr,
                    format!(
                        "unresolved name '{other}' in a behavioural expression (the front end \
                         substitutes .param values; elaborate the netlist first)"
                    ),
                ));
            }
        },
        BExprKind::Voltage { positive, negative } => {
            let plus = compiler.variable(QuantityKey::Node(positive));
            match negative {
                None => plus,
                Some(negative) => {
                    let minus = compiler.variable(QuantityKey::Node(negative));
                    Node::Operator(Operator::Minus, Arc::new(plus), Arc::new(minus))
                }
            }
        }
        BExprKind::Current(name) => compiler.variable(QuantityKey::Branch(name)),
        BExprKind::Unary { op, operand } => {
            let operand = boxed(operand, compiler)?;
            match op {
                BUnaryOp::Plus => Arc::unwrap_or_clone(operand),
                BUnaryOp::Minus => Node::Unary(Unary::Uminus, operand),
                BUnaryOp::Not => Node::Unary(Unary::Eq0, operand),
            }
        }
        BExprKind::Binary { op, lhs, rhs } => {
            let lhs = boxed(lhs, compiler)?;
            let rhs = boxed(rhs, compiler)?;
            compile_binary(*op, lhs, rhs)
        }
        BExprKind::Ternary {
            condition,
            then,
            otherwise,
        } => Node::Ternary(
            boxed(condition, compiler)?,
            boxed(then, compiler)?,
            boxed(otherwise, compiler)?,
        ),
        BExprKind::Group(inner) => compile(inner, compiler)?,
        BExprKind::Call { name, arguments } => call(expr, name, arguments, compiler)?,
        BExprKind::Value(_) => {
            return Err(parse_error(
                expr,
                "unevaluated numparam value in a behavioural expression (elaborate the netlist \
                 first)",
            ));
        }
        BExprKind::Table(table) => {
            let mut points = Vec::with_capacity(table.points.len());
            for (x, y) in &table.points {
                points.push((literal(x)?, literal(y)?));
            }
            let map = XspicePwl::new(&points, table.domain).map_err(|error| match error {
                SpiceError::Unsupported { feature, .. } => unsupported(expr, feature),
                other => other,
            })?;
            Node::Table(boxed(&table.input, compiler)?, Arc::new(map))
        }
    })
}

/// Lower a binary expression after compiling its operands in source order.
fn compile_binary(op: BBinaryOp, lhs: Tree, rhs: Tree) -> Node {
    let difference = |lhs, rhs| Arc::new(Node::Operator(Operator::Minus, lhs, rhs));
    match op {
        BBinaryOp::Add => Node::Operator(Operator::Plus, lhs, rhs),
        BBinaryOp::Sub => Node::Operator(Operator::Minus, lhs, rhs),
        BBinaryOp::Mul => Node::Operator(Operator::Times, lhs, rhs),
        BBinaryOp::Div => Node::Operator(Operator::Divide, lhs, rhs),
        BBinaryOp::Pow => Node::Operator(Operator::Power, lhs, rhs),
        BBinaryOp::Eq => Node::Unary(Unary::Eq0, difference(lhs, rhs)),
        BBinaryOp::Ne => Node::Unary(Unary::Ne0, difference(lhs, rhs)),
        BBinaryOp::Lt => Node::Unary(Unary::Lt0, difference(lhs, rhs)),
        BBinaryOp::Gt => Node::Unary(Unary::Gt0, difference(lhs, rhs)),
        BBinaryOp::Le => Node::Unary(Unary::Le0, difference(lhs, rhs)),
        BBinaryOp::Ge => Node::Unary(Unary::Ge0, difference(lhs, rhs)),
        // inpptree-parser.y: a && b is eq0(eq0(a) + eq0(b)), a || b is
        // ne0(ne0(a) + ne0(b)).
        BBinaryOp::And => Node::Unary(
            Unary::Eq0,
            Arc::new(Node::Operator(
                Operator::Plus,
                Arc::new(Node::Unary(Unary::Eq0, lhs)),
                Arc::new(Node::Unary(Unary::Eq0, rhs)),
            )),
        ),
        BBinaryOp::Or => Node::Unary(
            Unary::Ne0,
            Arc::new(Node::Operator(
                Operator::Plus,
                Arc::new(Node::Unary(Unary::Ne0, lhs)),
                Arc::new(Node::Unary(Unary::Ne0, rhs)),
            )),
        ),
    }
}

/// A literal number, possibly parenthesised and/or signed once
/// (`prepare_PTF_PWL` accepts a constant or `-constant`).
fn literal(expr: &BExpr) -> SpiceResult<Real> {
    match &expr.kind {
        BExprKind::Number { value, .. } => Ok(*value),
        BExprKind::Group(inner) => literal(inner),
        BExprKind::Unary {
            op: BUnaryOp::Plus,
            operand,
        } => literal(operand),
        BExprKind::Unary {
            op: BUnaryOp::Minus,
            operand,
        } => match &strip_groups(operand).kind {
            BExprKind::Number { value, .. } => Ok(-value),
            _ => Err(not_literal(expr)),
        },
        _ => Err(not_literal(expr)),
    }
}

fn strip_groups(expr: &BExpr) -> &BExpr {
    match &expr.kind {
        BExprKind::Group(inner) => strip_groups(inner),
        _ => expr,
    }
}

fn not_literal(expr: &BExpr) -> SpiceError {
    parse_error(
        expr,
        "PWL(expr, points...) only *literal* points are supported (C: prepare_PTF_PWL)",
    )
}

fn call<'a>(
    expr: &BExpr,
    name: &str,
    arguments: &'a [BExpr],
    compiler: &mut Compiler<'a>,
) -> SpiceResult<Node> {
    let arity = |expected: usize| -> SpiceResult<()> {
        if arguments.len() == expected {
            Ok(())
        } else {
            Err(parse_error(
                expr,
                format!(
                    "function '{name}' takes {expected} argument(s), found {}",
                    arguments.len()
                ),
            ))
        }
    };
    if let Some(function) = Unary::from_name(name) {
        arity(1)?;
        return Ok(Node::Unary(function, boxed(&arguments[0], compiler)?));
    }
    let two = |function: Binary, compiler: &mut Compiler<'a>| -> SpiceResult<Node> {
        arity(2)?;
        Ok(Node::Binary(
            function,
            boxed(&arguments[0], compiler)?,
            boxed(&arguments[1], compiler)?,
        ))
    };
    if let Some(law) = PowerLaw::from_name(name) {
        return two(Binary::Power(law, PowerPart::Value), compiler);
    }
    match name {
        "simetrix_abs_pow_product" => {
            arity(3)?;
            let factor = boxed(&arguments[0], compiler)?;
            let base = boxed(&arguments[1], compiler)?;
            let exponent = boxed(&arguments[2], compiler)?;
            Ok(absolute_power_product(factor, base, exponent))
        }
        "min" => two(Binary::Min, compiler),
        "max" => two(Binary::Max, compiler),
        "ternary_fcn" => {
            arity(3)?;
            Ok(Node::Ternary(
                boxed(&arguments[0], compiler)?,
                boxed(&arguments[1], compiler)?,
                boxed(&arguments[2], compiler)?,
            ))
        }
        "pwl" => pwl(expr, arguments, compiler),
        "ddt" => Err(SpiceError::not_yet_ported(
            format!(
                "{}: ddt() in a behavioural source (a stateful time derivative of \
                 iteration values)",
                expr.span.start
            ),
            "src/spicelib/parser/ptfuncs.c (PTddt)",
        )),
        name if STATISTICAL_FUNCTIONS.contains(&name) => Err(SpiceError::not_yet_ported(
            format!(
                "{}: {name}() in a behavioural source (a random value drawn once before \
                 the expression is parsed)",
                expr.span.start
            ),
            "src/frontend/inp.c (eval_agauss); src/frontend/inpcom.c \
             (inp_fix_agauss_in_param)",
        )),
        "pwl_derivative" => Err(unsupported(
            expr,
            "pwl_derivative() is internal to pwl() and has no table when called directly \
             (C dereferences a missing table)",
        )),
        _ => Err(parse_error(
            expr,
            format!("no such function '{name}' (src/spicelib/parser/inpptree.c PT_mkfnode)"),
        )),
    }
}

/// An explicit ordered product of a factor and an absolute-base power.
/// The source's zero-factor convention only masks a zero-base negative
/// power; finite products retain the ordinary product-rule Jacobian.
fn absolute_power_product(factor: Tree, base: Tree, exponent: Tree) -> Node {
    let zero = Arc::new(Node::Constant(0.));
    let factor_zero = Arc::new(Node::Unary(Unary::Eq0, Arc::clone(&factor)));
    let base_zero = Arc::new(Node::Unary(Unary::Eq0, Arc::clone(&base)));
    let exponent_negative = Arc::new(Node::Unary(Unary::Lt0, Arc::clone(&exponent)));
    let singular = Arc::new(Node::Ternary(
        base_zero,
        exponent_negative,
        Arc::clone(&zero),
    ));
    let masked = Arc::new(Node::Ternary(factor_zero, singular, Arc::clone(&zero)));
    let power = Arc::new(Node::Binary(
        Binary::Power(PowerLaw::SimetrixAbsolute, PowerPart::Value),
        base,
        exponent,
    ));
    let product = Arc::new(Node::Operator(Operator::Times, factor, power));
    Node::Ternary(masked, zero, product)
}

fn pwl<'a>(expr: &BExpr, arguments: &'a [BExpr], compiler: &mut Compiler<'a>) -> SpiceResult<Node> {
    let Some((input, points)) = arguments.split_first() else {
        return Err(parse_error(expr, "pwl() needs an input and points"));
    };
    if points.len() < 4 || points.len() % 2 != 0 {
        return Err(parse_error(
            expr,
            format!(
                "PWL(expr, points...) needs an even number (>= 4) of point values, found {} \
                 (C: prepare_PTF_PWL)",
                points.len()
            ),
        ));
    }
    let points = points
        .as_chunks::<2>()
        .0
        .iter()
        .map(|[x, y]| Ok((literal(x)?, literal(y)?)))
        .collect::<SpiceResult<Vec<_>>>()?;
    let table = PwlTable::new(points).ok_or_else(|| {
        parse_error(
            expr,
            "PWL(expr, points...) the abscissa of points must be monotonic \
             (C: prepare_PTF_PWL)",
        )
    })?;
    Ok(Node::Pwl(boxed(input, compiler)?, Arc::new(table)))
}

/// `PTpwl` supports both orders, but `PTpwl_derivative` always performs an
/// ascending search (even for descending point tables).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PwlOrder {
    Ascending,
    Descending,
}

impl PwlOrder {
    fn is_upper(self, x: Real, arg: Real) -> bool {
        match self {
            Self::Ascending => x > arg,
            Self::Descending => x < arg,
        }
    }

    fn ordered(self, pts: &[(Real, Real)]) -> bool {
        pts.windows(2).all(|w| match self {
            Self::Ascending => w[0].0 <= w[1].0,
            Self::Descending => w[0].0 >= w[1].0,
        })
    }
}

/// Validated literal coordinates, shared by value and slope nodes. The first
/// two abscissas must differ; subsequent equal abscissas are allowed by C.
#[derive(Debug, PartialEq)]
struct PwlTable {
    points: Vec<(Real, Real)>,
    order: PwlOrder,
    boundaries: Vec<Real>,
}

impl PwlTable {
    fn new(points: Vec<(Real, Real)>) -> Option<Self> {
        let [first, second, ..] = points.as_slice() else {
            return None;
        };
        let order = if first.0 < second.0 {
            PwlOrder::Ascending
        } else if first.0 > second.0 {
            PwlOrder::Descending
        } else {
            return None;
        };
        if !order.ordered(&points) {
            return None;
        }
        let mut boundaries: Vec<_> = points[1..points.len() - 1]
            .iter()
            .map(|point| point.0)
            .collect();
        if matches!(order, PwlOrder::Descending) {
            boundaries.reverse();
        }
        boundaries.dedup_by(|a, b| (*a).partial_cmp(&*b) == Some(std::cmp::Ordering::Equal));
        Some(Self {
            points,
            order,
            boundaries,
        })
    }

    /// The original C binary search, including extrapolation at both ends.
    /// Preserve its midpoint and strict comparison even for duplicate knots.
    fn segment(&self, arg: Real, order: PwlOrder) -> [(Real, Real); 2] {
        self.segment_by(|x| order.is_upper(x, arg))
    }

    fn segment_by(&self, upper_than_arg: impl FnMut(Real) -> bool) -> [(Real, Real); 2] {
        let lower = self.segment_index_by(upper_than_arg);
        [self.points[lower], self.points[lower + 1]]
    }

    fn segment_index_by(&self, mut upper_than_arg: impl FnMut(Real) -> bool) -> usize {
        let (mut lower, mut upper) = (0, self.points.len() - 1);
        while upper - lower > 1 {
            let middle = usize::midpoint(lower, upper);
            if upper_than_arg(self.points[middle].0) {
                upper = middle;
            } else {
                lower = middle;
            }
        }
        lower
    }

    /// `PTpwl` in the original floating-point operation order.
    fn value(&self, arg: Real) -> Real {
        let [(x0, y0), (x1, y1)] = self.segment(arg, self.order);
        y0 + (y1 - y0) * (arg - x0) / (x1 - x0)
    }

    /// Derivative of the selected interpolation/extrapolation segment.
    fn slope(&self, arg: Real) -> Real {
        let [(x0, y0), (x1, y1)] = self.segment(arg, self.order);
        (y1 - y0) / (x1 - x0)
    }
}

/// `PTpowerH` without a compatibility mode (the `^` operator).
fn power_h(a: Real, b: Real) -> Real {
    if a >= 0. {
        c_pow(a, b)
    } else if almost_equal_ulps(b.round_ties_even(), b, 10) {
        c_pow(a, b.round())
    } else {
        0.
    }
}

/// `PTpower` (the `pow()` function).
fn power(a: Real, b: Real) -> Real {
    if a == 0. {
        0.
    } else if a > 0. {
        c_pow(a, b)
    } else if almost_equal_ulps(b.round_ties_even(), b, 10) {
        c_pow(a, b.round())
    } else {
        0.
    }
}

/// `PTpwr` without a compatibility mode.
fn pwr(a: Real, b: Real) -> Real {
    if a < 0. { -c_pow(-a, b) } else { c_pow(a, b) }
}

/// Branch-local base derivative of the actual power value kernel.
/// Ordinary negative bases use the same quasi-integer exponent rule
/// as the value; signed powers instead differentiate |a|^b with an
/// even base slope. No absolute-value derivative is substituted for an
/// ordinary integer power.
fn power_base_slope(a: Real, b: Real, signed: bool) -> Real {
    if b == 0. {
        return 0.;
    }
    if a == 0. {
        return if b > 1. {
            0.
        } else if b.to_bits() == 1_f64.to_bits() {
            1.
        } else {
            Real::INFINITY
        };
    }
    if signed {
        return b * c_pow(a.abs(), b - 1.);
    }
    if a > 0. {
        return b * c_pow(a, b - 1.);
    }
    if almost_equal_ulps(b.round_ties_even(), b, 10) {
        let exponent = b.round();
        exponent * c_pow(a, exponent - 1.)
    } else {
        0.
    }
}

/// Partial with respect to the exponent. For ordinary negative bases
/// the value kernel is locally constant in each exponent branch
/// (rounded integer or zero); its branch-local partial is zero.
/// At a branch boundary no unique derivative exists.
fn power_exponent_slope(a: Real, b: Real, signed: bool) -> Real {
    if a == 0. || (!signed && a < 0.) {
        0.
    } else if signed {
        pwr(a, b) * a.abs().ln()
    } else {
        power(a, b) * a.ln()
    }
}

/// Source absolute-base law: both PWR and **, with the measured zero-base
/// Jacobian. Singular negative powers remain errors outside an explicit
/// ordered zero-factor product.
fn absolute_power(part: PowerPart, a: Real, b: Real) -> Real {
    if a == 0. {
        if b < 0. {
            return Real::INFINITY;
        }
        return if part == PowerPart::Value && b == 0. {
            1.
        } else {
            0.
        };
    }
    let magnitude = a.abs();
    match part {
        PowerPart::Value => c_pow(magnitude, b),
        PowerPart::Base if b == 0. => 0.,
        PowerPart::Base => a.signum() * b * c_pow(magnitude, b - 1.),
        PowerPart::Exponent => c_pow(magnitude, b) * magnitude.ln(),
    }
}

/// Source caret law. Negative fractional bases are outside its real domain.
/// The separately measured primal, base-slope and log floors at zero are
/// intentionally not shared with the absolute-base or complex-real laws.
fn caret_power(part: PowerPart, a: Real, b: Real) -> Real {
    let integral = b.fract() == 0.;
    if a < 0. && !integral {
        return Real::NAN;
    }
    let value = || {
        if a == 0. && !integral && b > 0. {
            c_pow(1e-308, b)
        } else {
            c_pow(a, b)
        }
    };
    match part {
        PowerPart::Value => value(),
        PowerPart::Base if b == 0. => 0.,
        PowerPart::Base if a == 0. && b > 1. => 0.,
        PowerPart::Base if a == 0. => b * c_pow(1e-200, b - 1.),
        PowerPart::Base => b * c_pow(a, b - 1.),
        PowerPart::Exponent => value() * a.max(1e-100).ln(),
    }
}

/// The real component of a principal complex power. The exact integer
/// branches preserve signs without trigonometric roundoff.
fn real_part_power(part: PowerPart, a: Real, b: Real) -> Real {
    if a == 0. {
        return match part {
            PowerPart::Value => c_pow(a, b),
            PowerPart::Base if b == 0. || b > 1. => 0.,
            PowerPart::Base => b * c_pow(1e-200, b - 1.),
            PowerPart::Exponent if b > 0. => 0.,
            PowerPart::Exponent => c_pow(a, b) * 1e-100_f64.ln(),
        };
    }
    let magnitude = a.abs();
    let (sin, cos) = if a > 0. {
        (0., 1.)
    } else if b.fract() == 0. {
        (0., if b.rem_euclid(2.) == 0. { 1. } else { -1. })
    } else {
        (core::f64::consts::PI * b.rem_euclid(2.)).sin_cos()
    };
    match part {
        PowerPart::Value => c_pow(magnitude, b) * cos,
        PowerPart::Base if b == 0. => 0.,
        PowerPart::Base => a.signum() * b * c_pow(magnitude, b - 1.) * cos,
        PowerPart::Exponent => {
            c_pow(magnitude, b) * magnitude.ln().mul_add(cos, -core::f64::consts::PI * sin)
        }
    }
}

/// C's `MODULUS(x, limit)`: `x - (int)(x / limit) * limit` (the `int`
/// conversion saturates, as on the reference platform).
fn modulus(x: Real, limit: Real) -> Real {
    #[expect(
        clippy::cast_possible_truncation,
        reason = "C MODULUS explicitly converts the quotient to a saturating i32 on this reference platform"
    )]
    let whole = (x / limit) as i32;
    let multiple = Real::from(whole) * limit;
    x - multiple
}

const TWO_PI: Real = 2. * std::f64::consts::PI;
const EXP_LIMIT: Real = 227.955_924_2;

/// C's `pow()`. The exponent is opaque to the optimiser so that a constant
/// exponent (`x^2`) is not rewritten to `x * x`: the platform `pow` is not
/// always correctly rounded, and C calls it.
fn c_pow(a: Real, b: Real) -> Real {
    a.powf(std::hint::black_box(b))
}

/// The `ptfuncs.c` one-argument functions. C's `HUGE` (out of range) is
/// [`Real::INFINITY`].
fn unary_value(function: Unary, x: Real) -> Real {
    match function {
        Unary::Abs => x.abs(),
        Unary::Sgn => {
            if x > 0. {
                1.
            } else if x < 0. {
                -1.
            } else {
                0.
            }
        }
        Unary::Acos => x.acos(),
        Unary::Acosh => x.acosh(),
        Unary::Asin => x.asin(),
        Unary::Asinh => x.asinh(),
        Unary::Atan => x.atan(),
        Unary::Atanh => x.atanh(),
        Unary::Cos => modulus(x, TWO_PI).cos(),
        Unary::Cosh => x.cosh(),
        Unary::Exp => {
            if x > EXP_LIMIT {
                1e99
            } else {
                x.exp()
            }
        }
        Unary::Log => {
            if x < 0. {
                Real::INFINITY
            } else if x == 0. {
                -1e99
            } else {
                x.ln()
            }
        }
        Unary::Log10 => {
            if x < 0. {
                Real::INFINITY
            } else if x == 0. {
                -1e99
            } else {
                x.log10()
            }
        }
        Unary::Sin => modulus(x, TWO_PI).sin(),
        Unary::Sinh => x.sinh(),
        Unary::Sqrt => {
            if x < 0. {
                Real::INFINITY
            } else {
                x.sqrt()
            }
        }
        Unary::Tan => modulus(x, std::f64::consts::PI).tan(),
        Unary::Tanh => x.tanh(),
        Unary::Ustep => {
            if x < 0. {
                0.
            } else if x > 0. {
                1.
            } else {
                0.5
            }
        }
        Unary::Uramp => {
            if x < 0. {
                0.
            } else {
                x
            }
        }
        Unary::Ceil => x.ceil(),
        Unary::Floor => x.floor(),
        Unary::Nint => x.round_ties_even(),
        Unary::Uminus => -x,
        Unary::Ustep2 => {
            if x <= 0. {
                0.
            } else if x <= 1. {
                x
            } else {
                1.
            }
        }
        Unary::Eq0 => flag(x == 0.),
        Unary::Ne0 => flag(x != 0.),
        Unary::Gt0 => flag(x > 0.),
        Unary::Lt0 => flag(x < 0.),
        Unary::Ge0 => flag(x >= 0.),
        Unary::Le0 => flag(x <= 0.),
    }
}

const fn flag(condition: bool) -> Real {
    if condition { 1. } else { 0. }
}

/// `PTdivide`: the divisor moves `fudge` away from zero; a zero divisor
/// after that is `HUGE`.
fn divide(a: Real, b: Real, fudge: Real) -> Real {
    let b = if b >= 0. { b + fudge } else { b - fudge };
    if b == 0. { Real::INFINITY } else { a / b }
}

#[cfg(test)]
mod derivative_reference;
#[cfg(test)]
use derivative_reference::{constant, differentiate};

#[cfg(test)]
pub(super) fn power_h_for_tests(a: Real, b: Real) -> Real {
    power_h(a, b)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn source_expr(kind: BExprKind, line: u32) -> BExpr {
        use crate::netlist::expr::SourceSpan;
        let loc = crate::primitives::SourceLoc::new(
            std::path::PathBuf::from("source-limit.cir"),
            line,
            1,
        );
        BExpr {
            kind,
            span: SourceSpan {
                start: loc.clone(),
                end: loc,
                logical: 0..1,
            },
        }
    }

    fn leaf(line: u32) -> BExpr {
        source_expr(
            BExprKind::Number {
                value: f64::from(line),
                spelling: line.to_string(),
            },
            line,
        )
    }

    #[test]
    fn shared_preflight_accepts_exact_limits_and_rejects_the_next_node() {
        // One Call wrapper + 99,999 leaves exactly fills the source-node
        // budget. All children are visited in stored order, not re-sorted.
        let mut tree = source_expr(
            BExprKind::Call {
                name: "test".into(),
                arguments: (0..MAX_SOURCE_NODES - 1).map(|_| leaf(2)).collect(),
            },
            1,
        );
        check_source_limits(&tree).unwrap();
        if let BExprKind::Call { arguments, .. } = &mut tree.kind {
            arguments.push(leaf(3));
        }
        let error = check_source_limits(&tree).unwrap_err();
        assert!(error.to_string().contains("source size/depth limit"));
        assert!(error.to_string().contains("source-limit.cir:3:1"));
    }

    #[test]
    fn shared_preflight_keeps_depth_boundary_and_first_error() {
        // Build the nesting iteratively to avoid a parser's syntactic
        // nesting limit and test only the compiler's structural bound.
        let mut at_limit = leaf(1);
        for _ in 1..MAX_SOURCE_DEPTH {
            at_limit = source_expr(BExprKind::Group(Box::new(at_limit)), 2);
        }
        check_source_limits(&at_limit).unwrap();
        let deeper = source_expr(BExprKind::Group(Box::new(at_limit)), 3);
        let error = check_source_limits(&deeper).unwrap_err();
        assert!(error.to_string().contains("source size/depth limit"));
        assert!(error.to_string().contains("source-limit.cir:1:1"));
    }

    #[test]
    fn unevaluated_numparam_value_is_not_lowered_into_device_program() {
        // Front-end construction counts the embedded numparam tree, but a
        // Value is terminal in the device compiler: compile rejects it
        // before ever lowering the hidden subtree into device instructions.
        use crate::netlist::expr::{Expr, ExprKind, ParameterExpression};
        let value = ParameterExpression {
            text: "unresolved".into(),
            braced: true,
            quoted: false,
            span: source_expr(BExprKind::Name(String::new()), 2).span,
            root: Expr {
                kind: ExprKind::Identifier("unresolved".into()),
                span: source_expr(BExprKind::Name(String::new()), 3).span,
            },
        };
        let tree = source_expr(BExprKind::Value(Box::new(value)), 1);
        check_source_limits(&tree).unwrap();
        let error = Program::compile(&tree).unwrap_err();
        assert!(error.to_string().contains("unevaluated numparam"));
    }

    #[test]
    fn ulps_comparison_matches_dawson() {
        assert!(almost_equal_ulps(2.0, 2.0, 10));
        assert!(almost_equal_ulps(2.0, 2.0 + 4. * f64::EPSILON, 10));
        assert!(!almost_equal_ulps(2.0, 2.0001, 10));
        assert!(almost_equal_ulps(-0.0, 0.0, 10));
    }

    #[test]
    fn power_follows_ptpowerh() {
        assert_eq!(power_h(-2., 2.), 4.);
        assert_eq!(power_h(-2., 2.5), 0.);
        assert_eq!(power_h(0., 0.), 1.);
        assert_eq!(power(0., 0.), 0.);
        assert_eq!(pwr(-2., 2.), -4.);
        assert_eq!(power_h_for_tests(4., 0.5), 2.);
    }

    #[test]
    fn pwl_extrapolates_and_searches_like_c() {
        let ascending = PwlTable::new(vec![(0., 0.), (1., 1.), (3., 5.)]).unwrap();
        assert_eq!(ascending.value(2.), 3.);
        assert_eq!(ascending.value(-1.), -1.);
        assert_eq!(ascending.value(4.), 7.);
        let descending = PwlTable::new(vec![(3., 5.), (1., 1.), (0., 0.)]).unwrap();
        assert_eq!(descending.value(2.), 3.);
        assert_eq!(ascending.slope(2.), 2.);
        assert_eq!(descending.slope(2.), 2.);

        let duplicate = PwlTable::new(vec![(0., 0.), (1., 1.), (1., 2.), (2., 5.)]).unwrap();
        assert_eq!(duplicate.value(1.), 2.);
        assert_eq!(duplicate.slope(1.), 3.);
        assert!(PwlTable::new(vec![(0., 0.), (0., 1.)]).is_none());
        assert!(PwlTable::new(vec![(0., 0.), (1., 1.), (0., 2.)]).is_none());
    }

    #[test]
    fn derivative_nodes_share_immutable_tables() {
        let input = Arc::new(Node::Operator(
            Operator::Plus,
            Arc::new(Node::Variable(0)),
            Arc::new(Node::Variable(1)),
        ));
        let points = vec![(0., 0.), (1., 2.), (2., 1.)];
        let pwl = Arc::new(PwlTable::new(points.clone()).unwrap());
        let root = Arc::new(Node::Pwl(Arc::clone(&input), Arc::clone(&pwl)));
        for variable in 0..2 {
            let derivative = differentiate(&root, variable);
            let Node::PwlDerivative(arg, table) = &*derivative else {
                panic!("expected PWL slope");
            };
            assert!(Arc::ptr_eq(arg, &input));
            assert!(Arc::ptr_eq(table, &pwl));
        }

        let map = Arc::new(XspicePwl::new(&points, 0.001).unwrap());
        let root = Arc::new(Node::Table(Arc::clone(&input), Arc::clone(&map)));
        for variable in 0..2 {
            let derivative = differentiate(&root, variable);
            let Node::TableSlope(arg, table) = &*derivative else {
                panic!("expected TABLE slope");
            };
            assert!(Arc::ptr_eq(arg, &input));
            assert!(Arc::ptr_eq(table, &map));
        }
    }

    #[test]
    fn modulus_reduces_like_c() {
        assert_eq!(modulus(7., TWO_PI), 7. - TWO_PI);
        assert_eq!(modulus(-7., TWO_PI), -7. + TWO_PI);
        assert_eq!(modulus(1., TWO_PI), 1.);
    }
}
