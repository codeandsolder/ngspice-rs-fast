//! Compiled behavioural expressions: value and analytic derivatives.
//!
//! C references (behaviour only): `src/spicelib/parser/inpptree.c`
//! (`PT_mkfnode`, `PTdifferentiate`, `prepare_PTF_PWL`), `ptfuncs.c` (the
//! function values and domain rules) and `ifeval.c` (`PTeval`). Like ngspice,
//! the port keeps the parse tree unsimplified, differentiates it symbolically
//! once per quantity with C's rules and its `mkb`/`mkf` simplifications
//! (constant folding, the 0/1 identities), and evaluates the value tree and
//! each derivative with the same operation order as C. The temporary trees
//! are lowered to a shared indexed DAG; successful finite results are
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
//!   segments linearly, and its derivative uses C's ascending-only search
//!   (`PTpwl_derivative`) even for descending tables.
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
pub(super) use plan::Workspace;
use plan::{Plan, PlanBuilder, Slot};

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
    Pow,
    Pwr,
    Min,
    Max,
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
    PwlDerivative(Tree, Arc<PwlTable>),
    /// The XSPICE `pwl` transfer of a lowered TABLE.
    Table(Tree, Arc<XspicePwl>),
    /// The slope of [`Node::Table`] (the code model's partial).
    TableSlope(Tree, Arc<XspicePwl>),
}

impl Node {
    const fn constant(&self) -> Option<Real> {
        match self {
            Self::Constant(value) => Some(*value),
            _ => None,
        }
    }
}

// Bound recursive lowering/differentiation before constructing any trees.
// The derivative rule for each source node performs a fixed amount of work,
// with each child differentiated at most once per quantity. The separate
// instruction cap also limits the size of the retained execution plan.
const MAX_SOURCE_DEPTH: usize = crate::netlist::expr::MAX_TREE_DEPTH;
const MAX_SOURCE_NODES: usize = crate::netlist::expr::MAX_TREE_NODES;
const MAX_DERIVATIVE_VISITS: usize = 131_072;

fn check_source_limits(root: &BExpr) -> SpiceResult<()> {
    fn walk(expr: &BExpr, depth: usize, remaining: &mut usize) -> SpiceResult<()> {
        if depth > MAX_SOURCE_DEPTH || *remaining == 0 {
            return Err(unsupported(
                expr,
                "behavioural expression source size/depth limit exceeded",
            ));
        }
        *remaining -= 1;
        let mut child = |node| walk(node, depth + 1, remaining);
        match &expr.kind {
            BExprKind::Unary { operand, .. } | BExprKind::Group(operand) => child(operand)?,
            BExprKind::Binary { lhs, rhs, .. } => {
                child(lhs)?;
                child(rhs)?;
            }
            BExprKind::Ternary {
                condition,
                then,
                otherwise,
            } => {
                child(condition)?;
                child(then)?;
                child(otherwise)?;
            }
            BExprKind::Call { arguments, .. } => {
                for argument in arguments {
                    child(argument)?;
                }
            }
            BExprKind::Table(table) => {
                child(&table.input)?;
                for (x, y) in &table.points {
                    child(x)?;
                    child(y)?;
                }
            }
            _ => {}
        }
        Ok(())
    }
    let mut remaining = MAX_SOURCE_NODES;
    walk(root, 1, &mut remaining)
}

fn tree_size(node: &Node) -> usize {
    1 + match node {
        Node::Unary(_, arg)
        | Node::Pwl(arg, _)
        | Node::PwlDerivative(arg, _)
        | Node::Table(arg, _)
        | Node::TableSlope(arg, _) => tree_size(arg),
        Node::Operator(_, a, b) | Node::Binary(_, a, b) => tree_size(a) + tree_size(b),
        Node::Ternary(c, a, b) => tree_size(c) + tree_size(a) + tree_size(b),
        _ => 0,
    }
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
    /// Compiles a front-end-resolved expression tree and builds its
    /// derivative trees, then lowers them to the shared execution plan.
    ///
    /// # Errors
    /// Names the front end should have resolved (`.param` values, `.func`
    /// calls), unknown functions or wrong argument counts (positioned parse
    /// errors), non-literal or non-monotonic `pwl()` points (C:
    /// `prepare_PTF_PWL`), and [`SpiceError::NotYetPorted`] for `ddt()` and
    /// the statistical functions (`agauss`, `gauss`, `aunif`, `unif`,
    /// `limit`; normally already rejected by the front end). Expressions
    /// exceeding the source depth, size or symbolic expansion limits are
    /// explicitly rejected before unbounded work can occur.
    pub fn compile(expr: &BExpr) -> SpiceResult<Self> {
        check_source_limits(expr)?;
        let mut compiler = Compiler::default();
        let value_tree = Arc::new(compile(expr, &mut compiler)?);
        if compiler.quantities.len() > MAX_DERIVATIVE_VISITS / tree_size(&value_tree) {
            return Err(unsupported(
                expr,
                "behavioural derivative expansion limit exceeded",
            ));
        }
        let mut builder = PlanBuilder::default();
        let root = builder.lower_value(&value_tree)?;
        let derivatives = (0..compiler.quantities.len())
            .map(|variable| builder.lower_derivative(&differentiate(&value_tree, variable)))
            .collect::<SpiceResult<_>>()?;
        Ok(Self {
            plan: Arc::new(builder.finish()),
            root,
            derivatives,
            quantities: compiler.quantities,
            uses_frequency: compiler.uses_frequency,
        })
    }

    /// True when the expression reads `hertz`.
    #[must_use]
    pub const fn uses_frequency(&self) -> bool {
        self.uses_frequency
    }

    /// The circuit quantities the expression reads, in variable order.
    #[must_use]
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
        self.evaluate_with_workspace(environment, &mut Workspace::default())
    }

    pub(super) fn evaluate_with_workspace(
        &self,
        environment: &Environment<'_>,
        workspace: &mut Workspace,
    ) -> SpiceResult<Evaluation> {
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
            "time" => Node::Time,
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
    match name {
        "pow" => two(Binary::Pow, compiler),
        "pwr" => two(Binary::Pwr, compiler),
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
        order.ordered(&points).then_some(Self { points, order })
    }

    /// The original C binary search, including extrapolation at both ends.
    /// Preserve its midpoint and strict comparison even for duplicate knots.
    fn segment(&self, arg: Real, order: PwlOrder) -> [(Real, Real); 2] {
        let (mut lower, mut upper) = (0, self.points.len() - 1);
        while upper - lower > 1 {
            let middle = usize::midpoint(lower, upper);
            if order.is_upper(self.points[middle].0, arg) {
                upper = middle;
            } else {
                lower = middle;
            }
        }
        [self.points[lower], self.points[upper]]
    }

    /// `PTpwl` in the original floating-point operation order.
    fn value(&self, arg: Real) -> Real {
        let [(x0, y0), (x1, y1)] = self.segment(arg, self.order);
        y0 + (y1 - y0) * (arg - x0) / (x1 - x0)
    }

    /// `PTpwl_derivative`: intentionally ignores the table's actual order.
    fn slope(&self, arg: Real) -> Real {
        let [(x0, y0), (x1, y1)] = self.segment(arg, PwlOrder::Ascending);
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

fn constant(value: Real) -> Tree {
    Arc::new(Node::Constant(value))
}

/// `mkcon(0.0)` / `mkcon(1.0)` for a variable.
fn zero() -> Tree {
    constant(0.)
}

/// `mkb()` for the arithmetic operators: constant folding (plain C
/// arithmetic and `pow`, not the `PT*` functions) and the 0/1 identities.
fn mk_operator(op: Operator, left: Tree, right: Tree) -> Tree {
    if let (Some(l), Some(r)) = (left.constant(), right.constant()) {
        return constant(match op {
            Operator::Times => l * r,
            Operator::Divide => l / r,
            Operator::Plus => l + r,
            Operator::Minus => l - r,
            Operator::Power => c_pow(l, r),
        });
    }
    let (l, r) = (left.constant(), right.constant());
    match op {
        Operator::Times => {
            // C's order: a zero factor (either side) wins, then a unit one.
            if l == Some(0.) {
                return left;
            } else if r == Some(0.) || l == Some(1.) {
                return right;
            } else if r == Some(1.) {
                return left;
            }
        }
        Operator::Divide => {
            if l == Some(0.) || r == Some(1.) {
                return left;
            }
        }
        Operator::Plus => {
            if l == Some(0.) {
                return right;
            } else if r == Some(0.) {
                return left;
            }
        }
        Operator::Minus => {
            if r == Some(0.) {
                return left;
            } else if l == Some(0.) {
                return mk_unary(Unary::Uminus, right);
            }
        }
        Operator::Power => {
            if r == Some(0.) {
                return constant(1.);
            } else if r == Some(1.) {
                return left;
            }
        }
    }
    Arc::new(Node::Operator(op, left, right))
}

/// `mkb(PT_TERN, condition, mkb(PT_COMMA, then, otherwise))`.
fn mk_ternary(condition: Tree, then: Tree, otherwise: Tree) -> Tree {
    if let Some(condition) = condition.constant() {
        return if condition == 0. { otherwise } else { then };
    }
    if let (Some(a), Some(b)) = (then.constant(), otherwise.constant())
        && a.partial_cmp(&b) == Some(std::cmp::Ordering::Equal)
    {
        return then;
    }
    Arc::new(Node::Ternary(condition, then, otherwise))
}

/// `mkf()` for a one-argument function: a constant argument is folded.
fn mk_unary(function: Unary, argument: Tree) -> Tree {
    argument.constant().map_or_else(
        || Arc::new(Node::Unary(function, argument)),
        |x| constant(unary_value(function, x)),
    )
}

/// `mkf(PTF_POW/PTF_PWR, mkb(PT_COMMA, a, b))`: never folded (a comma node
/// is not a constant).
fn mk_binary(function: Binary, a: &Tree, b: Tree) -> Tree {
    Arc::new(Node::Binary(function, a.clone(), b))
}

/// `mkb(PT_TIMES, ...)` shorthand.
fn times(left: Tree, right: Tree) -> Tree {
    mk_operator(Operator::Times, left, right)
}

/// `mkb(PT_DIVIDE, ...)` shorthand.
fn over(left: Tree, right: Tree) -> Tree {
    mk_operator(Operator::Divide, left, right)
}

/// `mkb(PT_POWER, u, mkcon(2.0))`.
fn squared(u: &Tree) -> Tree {
    mk_operator(Operator::Power, u.clone(), constant(2.))
}

/// `mkf(PTF_LOG, mkf(PTF_ABS, a))`.
fn log_abs(a: &Tree) -> Tree {
    mk_unary(Unary::Log, mk_unary(Unary::Abs, a.clone()))
}

/// The general power rule `f(a,b) * (b * (D(a) / a) + D(b) * log(abs(a)))`
/// of `PTdifferentiate` (`^`, `pow()`, `pwr()`), with `outer = f(a,b)`.
fn general_power(outer: Tree, a: &Tree, b: &Tree, variable: usize) -> Tree {
    times(
        outer,
        mk_operator(
            Operator::Plus,
            times(b.clone(), over(differentiate(a, variable), a.clone())),
            times(differentiate(b, variable), log_abs(a)),
        ),
    )
}

/// The derivative's two power modes, separate from general binary
/// functions: `^` and `pow()` have the ordinary rule, `pwr()` the signed rule.
#[derive(Clone, Copy, PartialEq, Eq)]
enum PowerRule {
    Ordinary,
    Signed,
}

impl PowerRule {
    const fn value_kernel(self) -> Binary {
        match self {
            Self::Ordinary => Binary::Pow,
            Self::Signed => Binary::Pwr,
        }
    }

    const fn exponent_kernel(self) -> Binary {
        match self {
            Self::Ordinary => Binary::Pwr,
            Self::Signed => Binary::Pow,
        }
    }
}

/// The three C power syntaxes share a derivative rule, but use different
/// function kernels when the exponent is constant.  `^` and `pow()` use
/// `pwr(a, b - 1)`; `pwr()` instead uses `pow(a, b - 1)`.
///
/// For variable exponents, only the `pow`/ `^` rule has C's special
/// constant-base case.  Keep the original mkb/mkf tree construction order;
/// it affects the values and the rounding of the evaluated Jacobian.
fn differentiate_power(rule: PowerRule, a: &Tree, b: &Tree, variable: usize) -> Tree {
    b.constant().map_or_else(
        || {
            if rule == PowerRule::Ordinary && a.constant().is_some() {
                times(
                    mk_binary(Binary::Pow, a, b.clone()),
                    times(differentiate(b, variable), log_abs(a)),
                )
            } else {
                general_power(mk_binary(rule.value_kernel(), a, b.clone()), a, b, variable)
            }
        },
        |exponent| {
            times(
                times(
                    constant(exponent),
                    mk_binary(rule.exponent_kernel(), a, constant(exponent - 1.)),
                ),
                differentiate(a, variable),
            )
        },
    )
}

/// The outer derivative of a unary function, constructed with C's tree
/// helpers. The chain rule is applied once by the main differentiator.
fn unary_slope(function: Unary, u: &Tree) -> Tree {
    let one = || constant(1.);
    match function {
        Unary::Abs => mk_unary(Unary::Sgn, u.clone()),
        Unary::Acos => over(
            constant(-1.),
            mk_unary(Unary::Sqrt, mk_operator(Operator::Minus, one(), squared(u))),
        ),
        Unary::Acosh => over(
            one(),
            mk_unary(Unary::Sqrt, mk_operator(Operator::Minus, squared(u), one())),
        ),
        Unary::Asin => over(
            one(),
            mk_unary(Unary::Sqrt, mk_operator(Operator::Minus, one(), squared(u))),
        ),
        Unary::Asinh => over(
            one(),
            mk_unary(Unary::Sqrt, mk_operator(Operator::Plus, squared(u), one())),
        ),
        Unary::Atan => over(one(), mk_operator(Operator::Plus, squared(u), one())),
        Unary::Atanh => over(one(), mk_operator(Operator::Minus, one(), squared(u))),
        Unary::Cos => mk_unary(Unary::Uminus, mk_unary(Unary::Sin, u.clone())),
        Unary::Cosh => mk_unary(Unary::Sinh, u.clone()),
        Unary::Exp => mk_unary(Unary::Exp, u.clone()),
        Unary::Log => over(one(), u.clone()),
        Unary::Log10 => over(constant(std::f64::consts::LOG10_E), u.clone()),
        Unary::Sin => mk_unary(Unary::Cos, u.clone()),
        Unary::Sinh => mk_unary(Unary::Cosh, u.clone()),
        Unary::Sqrt => over(one(), times(constant(2.), mk_unary(Unary::Sqrt, u.clone()))),
        Unary::Tan => mk_operator(
            Operator::Plus,
            one(),
            squared(&mk_unary(Unary::Tan, u.clone())),
        ),
        Unary::Tanh => mk_operator(
            Operator::Minus,
            one(),
            squared(&mk_unary(Unary::Tanh, u.clone())),
        ),
        Unary::Uramp => mk_unary(Unary::Ustep, u.clone()),
        Unary::Ustep2 => mk_operator(
            Operator::Minus,
            mk_unary(Unary::Ustep, u.clone()),
            mk_unary(Unary::Ustep, mk_operator(Operator::Minus, u.clone(), one())),
        ),
        Unary::Uminus => constant(-1.),
        Unary::Sgn
        | Unary::Ustep
        | Unary::Eq0
        | Unary::Ne0
        | Unary::Gt0
        | Unary::Lt0
        | Unary::Ge0
        | Unary::Le0
        | Unary::Floor
        | Unary::Ceil
        | Unary::Nint => zero(),
    }
}

/// `PTdifferentiate` (`inpptree.c`): the derivative tree of `p` with respect
/// to quantity `variable`, built with C's `mkb`/`mkf` simplifications so its
/// evaluation is C's operation for operation. The PSPICE/HSPICE/LTspice
/// `newcompat` variants are not modelled.
fn differentiate(p: &Tree, variable: usize) -> Tree {
    match &**p {
        Node::Constant(_) | Node::Time | Node::Temperature | Node::Frequency => zero(),
        Node::Variable(index) => constant(if *index == variable { 1. } else { 0. }),
        Node::Operator(op @ (Operator::Plus | Operator::Minus), a, b) => {
            mk_operator(*op, differentiate(a, variable), differentiate(b, variable))
        }
        // d(a * b) = d(a) * b + a * d(b)
        Node::Operator(Operator::Times, a, b) => mk_operator(
            Operator::Plus,
            times(differentiate(a, variable), b.clone()),
            times(a.clone(), differentiate(b, variable)),
        ),
        // d(a / b) = (d(a) * b - a * d(b)) / b^2
        Node::Operator(Operator::Divide, a, b) => over(
            mk_operator(
                Operator::Minus,
                times(differentiate(a, variable), b.clone()),
                times(a.clone(), differentiate(b, variable)),
            ),
            squared(b),
        ),
        Node::Operator(Operator::Power, a, b) | Node::Binary(Binary::Pow, a, b) => {
            differentiate_power(PowerRule::Ordinary, a, b, variable)
        }
        Node::Ternary(condition, then, otherwise) => mk_ternary(
            condition.clone(),
            differentiate(then, variable),
            differentiate(otherwise, variable),
        ),
        Node::Binary(function @ (Binary::Min | Binary::Max), a, b) => {
            // min(a,b)' = lt0(a-b) ? a' : b'; max uses gt0.
            let comparison = if *function == Binary::Min {
                Unary::Lt0
            } else {
                Unary::Gt0
            };
            mk_ternary(
                mk_unary(
                    comparison,
                    mk_operator(Operator::Minus, a.clone(), b.clone()),
                ),
                differentiate(a, variable),
                differentiate(b, variable),
            )
        }
        Node::Binary(Binary::Pwr, a, b) => differentiate_power(PowerRule::Signed, a, b, variable),
        Node::Unary(function, u) => times(unary_slope(*function, u), differentiate(u, variable)),
        Node::Pwl(u, table) => {
            let slope = u.constant().map_or_else(
                || Arc::new(Node::PwlDerivative(Arc::clone(u), Arc::clone(table))),
                |x| constant(table.slope(x)),
            );
            times(slope, differentiate(u, variable))
        }
        Node::Table(u, map) => {
            let slope = u.constant().map_or_else(
                || Arc::new(Node::TableSlope(Arc::clone(u), Arc::clone(map))),
                |x| constant(map.evaluate(x).1),
            );
            times(slope, differentiate(u, variable))
        }
        Node::PwlDerivative(u, _) | Node::TableSlope(u, _) => {
            times(zero(), differentiate(u, variable))
        }
    }
}

#[cfg(test)]
pub(super) fn power_h_for_tests(a: Real, b: Real) -> Real {
    power_h(a, b)
}

#[cfg(test)]
mod tests {
    use super::*;

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
        assert_eq!(descending.slope(2.), 1.);

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
