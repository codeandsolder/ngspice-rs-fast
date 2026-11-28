//! Front-end passes for behavioural sources (GitHub #79).
//!
//! Two jobs that ngspice's front end (`src/frontend/inpcom.c`) performs on
//! the card text before the circuit parser sees it:
//!
//! 1. [`lower_nonlinear_sources`] rewrites the nonlinear E/G/F/H forms into
//!    the primitive devices C builds (`inp_compat()`, `inp_poly_2g6_compat()`,
//!    `src/xspice/enh/enhtrans.c`):
//!
//!    | Form | Becomes |
//!    | --- | --- |
//!    | `e1 n+ n- value={f}` | `e1 n+ n- e1_int1 0 1` and `be1 e1_int1 0 v=f` |
//!    | `g1 n+ n- value={f} [m=k]` | `g1 n+ n- g1_int1 0 k` and `bg1 g1_int1 0 v=f` |
//!    | `e1 n+ n- table {f} = (x,y)...` | `e1 n+ n- e1_int1 0 1`, `be1 e1_int2 0 v=f` and the XSPICE `pwl` instance `ae1` from `e1_int2` to `e1_int1` |
//!    | `g1 ... table ... [m=k]` | the same with gain `k` (node names with `[`, `]`, `%` replaced by `_`) |
//!    | `e1 n+ n- nc+ nc- table=(...)` | `e1 n+ n- e1_int1 0 1` and `ae1` from `v(nc+,nc-)` (domain 0.001) |
//!    | one TABLE pair `(x0, y0)` | `v<name> <name>_int1 0 y0` instead of the B/A pair |
//!    | `e1 n+ n- poly(n) ...` (also F/G/H and the implicit `POLY(1)`) | native polynomial behavioural law retaining the original instance identity |
//!
//!    Generated TABLE instances (`ae1`) are designator-`a` devices carrying
//!    a behavioural voltage expression; POLY retains the source instance's
//!    identity and uses the same compiled expression kernel. The rewrite runs before
//!    subcircuit expansion, as in C, so generated names and nodes are renamed
//!    like any other (`x1.e1_int1`, `b.x1.be1`).
//!
//! 2. [`resolve_expression`] performs what numparam and macro expansion do to
//!    a B card: `.param` names become values, `.func` calls are expanded
//!    (with parenthesised arguments, `inp_expand_macro_in_str()`), numeric
//!    literals are rounded to the 11 significant digits `inp_modify_exp()`
//!    prints and numparam `{...}` values in `=pwl(` lines are evaluated.

use std::sync::Arc;

use crate::primitives::{Real, SourceLoc, SpiceError, SpiceResult};

use crate::netlist::ast::{DeviceInstance, Netlist, ParameterAssignment, ParameterKind, ScopeView};
use crate::netlist::bexpr::{
    BBinaryOp, BExpr, BExprKind, BUnaryOp, BehaviouralExpression, SPECIAL_NAMES,
    STATISTICAL_FUNCTIONS, TableTransfer, is_builtin_function, round_like_c_literal,
};
use crate::netlist::elaborate::format_literal;
use crate::netlist::eval::statistics::PreparedSite;
use crate::netlist::eval::{EvalBudget, FunctionScope, ParamScope};
use crate::netlist::expr::{
    BinaryOp, Expr, ExprKind, MAX_TREE_DEPTH, MAX_TREE_NODES, SourceSpan, TreeBudget, UnaryOp,
};

mod edits;
pub(super) mod forms;
mod poly;
use forms::{NonlinearForm, TableForm, TableInput, ValueForm};

/// C references of the front-end rewrites.
pub const C_REFERENCE: &str = "src/frontend/inpcom.c (inp_compat, inp_poly_2g6_compat, \
     inp_bsource_compat, inp_modify_exp); src/xspice/enh/enhtrans.c";

/// The XSPICE `pwl` `input_domain` `inp_compat()` uses for a TABLE.
pub const TABLE_DOMAIN: Real = 0.1;
/// The `input_domain` of the four-node `LTspice` TABLE form.
pub const LTSPICE_TABLE_DOMAIN: Real = 0.001;

/// Rewrites the nonlinear E/G/F/H forms of the root and every subcircuit
/// body into primitive devices (see the [module documentation](self)).
/// Everything else is copied unchanged; the input is never modified.
///
/// # Errors
/// A malformed nonlinear form (no expression, an odd TABLE point list, a
/// POLY without its controls), positioned at the card.
pub fn lower_nonlinear_sources(netlist: &Netlist) -> SpiceResult<Netlist> {
    edits::validate(ScopeView::from(netlist))?;
    let mut out = netlist.clone();
    edits::lower(&mut out)?;
    Ok(out)
}

fn lower_device(device: DeviceInstance, out: &mut Vec<DeviceInstance>) -> SpiceResult<()> {
    match NonlinearForm::validate(&device)? {
        Some(NonlinearForm::Value(form)) => out.extend(lower_value(&device, &form)),
        Some(NonlinearForm::Table(form)) => out.extend(lower_table(&device, &form)?),
        Some(NonlinearForm::Poly(form)) => out.push(poly::lower(&device, &form)?),
        None => out.push(device),
    }
    Ok(())
}

fn scalar(name: &str, value: &str, location: &SourceLoc) -> ParameterAssignment {
    ParameterAssignment {
        name: name.to_owned(),
        value: value.to_owned(),
        kind: ParameterKind::Scalar,
        location: location.clone(),
    }
}

fn instance(
    name: String,
    designator: char,
    nodes: Vec<String>,
    parameters: Vec<ParameterAssignment>,
    location: &SourceLoc,
) -> DeviceInstance {
    DeviceInstance {
        code_ports: Vec::new(),
        name,
        designator,
        nodes,
        model: None,
        parameters,
        location: location.clone(),
    }
}

fn malformed(device: &DeviceInstance, message: &str) -> SpiceError {
    SpiceError::parse(
        device.location.clone(),
        format!("{}: {message}", device.name),
    )
}

/// The controlled source driven by the generated internal node: gain 1 for
/// E, the `m=` value (default 1) for G (`inp_compat`, "find multiplier m").
fn driven_source(
    device: &DeviceInstance,
    internal: &str,
    setters: &[ParameterAssignment],
) -> DeviceInstance {
    let gain = setters
        .iter()
        .rev()
        .find(|p| p.name == "m")
        .filter(|_| device.designator == 'g')
        .map_or_else(
            || scalar("gain", "1", &device.location),
            |m| ParameterAssignment {
                name: "gain".to_owned(),
                ..m.clone()
            },
        );
    instance(
        device.name.clone(),
        device.designator,
        vec![
            device.nodes[0].clone(),
            device.nodes[1].clone(),
            internal.to_owned(),
            "0".to_owned(),
        ],
        vec![gain],
        &device.location,
    )
}

fn lower_value(device: &DeviceInstance, form: &ValueForm<'_>) -> [DeviceInstance; 2] {
    let internal = format!("{}_int1", device.name);
    let mut b_parameters = vec![ParameterAssignment {
        name: "v".to_owned(),
        ..form.input.clone()
    }];
    if device.designator == 'e' {
        b_parameters.extend_from_slice(form.setters);
    }
    [
        driven_source(device, &internal, form.setters),
        instance(
            format!("b{}", device.name),
            'b',
            vec![internal, "0".to_owned()],
            b_parameters,
            &device.location,
        ),
    ]
}

/// `inp_compat()` replaces `[`, `]` and `%` by `_` in the G TABLE names it
/// uses for nodes and the XSPICE instance.
fn sanitized(name: &str, designator: char) -> String {
    if designator == 'g' {
        name.replace(['[', ']', '%'], "_")
    } else {
        name.to_owned()
    }
}

fn generated(root: BExpr, text: String, location: &SourceLoc) -> ParameterKind {
    ParameterKind::Behavioural(Box::new(BehaviouralExpression {
        text,
        span: SourceSpan {
            start: location.clone(),
            end: location.clone(),
            logical: 0..0,
        },
        verbatim: true,
        root,
    }))
}

fn quantity(kind: BExprKind, location: &SourceLoc) -> BExpr {
    BExpr {
        kind,
        span: SourceSpan {
            start: location.clone(),
            end: location.clone(),
            logical: 0..0,
        },
    }
}

fn lower_table(device: &DeviceInstance, form: &TableForm<'_>) -> SpiceResult<Vec<DeviceInstance>> {
    let stok = sanitized(&device.name, device.designator);
    let int1 = format!("{stok}_int1");
    let int2 = format!("{stok}_int2");
    let mut out = vec![driven_source(device, &int1, form.setters)];
    if let [(_, y)] = form.points.as_slice() {
        out.push(constant_table_source(device, int1, y.parameter));
        return Ok(out);
    }
    let mut budget = TreeBudget::new(MAX_TREE_NODES);
    budget.charge(2, 2, &device.location)?; // Table root and input quantity.
    for (x, y) in &form.points {
        check_coefficient(x.parameter, 2, &mut budget)?;
        check_coefficient(y.parameter, 2, &mut budget)?;
    }
    let points: Vec<_> = form
        .points
        .iter()
        .map(|(x, y)| (x.expression(), y.expression()))
        .collect();
    let location = form.location;
    let (source, domain) = match form.input {
        TableInput::Voltage([positive, negative]) => (
            BExprKind::Voltage {
                positive: positive.clone(),
                negative: Some(negative.clone()),
            },
            LTSPICE_TABLE_DOMAIN,
        ),
        TableInput::Expression(input, expression) => {
            out.push(instance(
                format!("b{}", device.name),
                'b',
                vec![int2.clone(), "0".to_owned()],
                vec![ParameterAssignment {
                    name: "v".to_owned(),
                    value: expression.text.clone(),
                    kind: input.kind.clone(),
                    location: input.location.clone(),
                }],
                &device.location,
            ));
            (
                BExprKind::Voltage {
                    positive: int2,
                    negative: None,
                },
                TABLE_DOMAIN,
            )
        }
    };
    let text = format!(
        "xspice-pwl input_domain={domain} fraction=TRUE limit=TRUE ({} points)",
        points.len()
    );
    let root = quantity(
        BExprKind::Table(Box::new(TableTransfer {
            input: quantity(source, location),
            points,
            domain,
        })),
        location,
    );
    out.push(instance(
        format!("a{stok}"),
        'a',
        vec![int1, "0".to_owned()],
        vec![ParameterAssignment {
            name: "v".to_owned(),
            value: text.clone(),
            kind: generated(root, text, location),
            location: location.clone(),
        }],
        &device.location,
    ));
    Ok(out)
}

/// A single TABLE pair is a constant voltage source, independent of the input.
fn constant_table_source(
    device: &DeviceInstance,
    node: String,
    y: &ParameterAssignment,
) -> DeviceInstance {
    instance(
        format!("v{}", device.name),
        'v',
        vec![node, "0".to_owned()],
        vec![ParameterAssignment {
            name: "dc".to_owned(),
            ..y.clone()
        }],
        &device.location,
    )
}

/// A parameter value as the B-source parser sees it: numparam splices the
/// `"% 23.15e"` text into the expression, then the lexer reads the
/// decimal literal back with correct rounding.
fn spliced(value: Real) -> BExprKind {
    let spelling = format_literal(value);
    BExprKind::Number {
        value: crate::primitives::parse_spice_number(&spelling).unwrap_or(value),
        spelling,
    }
}

/// The generated coefficient is either a literal or a Value containing an
/// already parsed numparam tree. Count the wrapper too, before cloning it.
fn check_coefficient(
    parameter: &ParameterAssignment,
    depth: usize,
    budget: &mut TreeBudget,
) -> SpiceResult<usize> {
    budget.charge(depth, 1, &parameter.location)?;
    match &parameter.kind {
        ParameterKind::Expression(value) => value.root.check_tree(depth + 1, budget),
        _ => Ok(depth),
    }
}

/// Resolve a behavioural expression against a parameter scope.
///
/// Resolves a behavioural expression against a parameter scope: `.param`
/// names become numbers, `.func` calls are expanded, numparam `{...}` values
/// are evaluated and (outside verbatim text) literals are rounded like C's
/// `inp_modify_exp()`. Circuit quantities, `time`, `temper`, `hertz`, `pi`
/// and `e` stay symbolic.
///
/// # Errors
/// An undefined parameter, an unknown function (C: "no such function"), a
/// `.func` called with the wrong number of arguments, a bare name in a
/// verbatim (`=pwl(`) expression, a failing numparam value, or an expansion
/// larger than the node limit.
pub fn resolve_expression(
    expression: &BehaviouralExpression,
    scope: &ParamScope,
    budget: &mut EvalBudget,
) -> SpiceResult<BehaviouralExpression> {
    resolve_expression_at(expression, scope, budget, None)
}

pub(crate) fn resolve_expression_at(
    expression: &BehaviouralExpression,
    scope: &ParamScope,
    budget: &mut EvalBudget,
    site: Option<PreparedSite>,
) -> SpiceResult<BehaviouralExpression> {
    let mut resolver = Resolver {
        scope,
        functions: scope.functions().cloned(),
        budget,
        sampler: scope.sampler(&expression.span.start, site),
        prepared_frames: site
            .into_iter()
            .map(|site| (expression.span.clone(), site.expression))
            .collect(),
        exact: false,
        verbatim: expression.verbatim,
        syntax_only: false,
        nodes: 0,
        expansion_budget: TreeBudget::new(MAX_TREE_NODES),
    };
    let root = resolver.resolve(&expression.root, 1)?;
    Ok(BehaviouralExpression {
        text: expression.text.clone(),
        span: expression.span.clone(),
        verbatim: expression.verbatim,
        root,
    })
}

/// Use the same lexical classification for actual elaboration and for
/// inspecting a candidate's compiled constitutive dependencies.
pub(crate) fn classify_device_passives(
    device: &mut DeviceInstance,
    scope: &ParamScope,
    budget: &mut EvalBudget,
) -> SpiceResult<()> {
    for parameter in &mut device.parameters {
        if matches!(
            (device.designator, parameter.name.as_str()),
            ('r', "resistance") | ('c', "capacitance" | "charge") | ('l', "inductance")
        ) {
            classify_passive(parameter, scope, budget)?;
        }
    }
    Ok(())
}

/// Expand the exact macro language of a runtime law without substituting
/// numerical parameters. Discarded macro arguments are not dependencies;
/// names captured inside the body, including nested calls, are.
fn expand_symbolic(
    expression: &BehaviouralExpression,
    scope: &ParamScope,
    budget: &mut EvalBudget,
) -> SpiceResult<BExpr> {
    let mut resolver = Resolver {
        scope,
        functions: scope.functions().cloned(),
        budget,
        sampler: scope.sampler(&expression.span.start, None),
        prepared_frames: Vec::new(),
        exact: false,
        verbatim: expression.verbatim,
        syntax_only: true,
        nodes: 0,
        expansion_budget: TreeBudget::new(MAX_TREE_NODES),
    };
    resolver.resolve(&expression.root, 1)
}

/// Inspect the top-level laws *after* the same controlled-source lowering
/// and passive classification used by the circuit factory. This work is
/// performed once per parameter plan, never in a Newton iteration.
pub(crate) fn candidate_laws(
    netlist: &Netlist,
    scope: &ParamScope,
    budget: &mut EvalBudget,
) -> SpiceResult<Vec<BExpr>> {
    let mut lowered = lower_nonlinear_sources(netlist)?;
    let mut roots = Vec::new();
    for device in &mut lowered.devices {
        classify_device_passives(device, scope, budget)?;
        for parameter in &device.parameters {
            if let ParameterKind::Behavioural(expression) = &parameter.kind {
                roots.push(expand_symbolic(expression, scope, budget)?);
            }
        }
    }
    Ok(roots)
}

/// Classify a passive's primary expression after lexical macro expansion.
/// Numeric-only expressions remain in the numparam evaluator; runtime laws
/// use the behavioural evaluator. No static function is numerically evaluated
/// or rounded by the classification pass.
pub(crate) fn classify_passive(
    parameter: &mut ParameterAssignment,
    scope: &ParamScope,
    budget: &mut EvalBudget,
) -> SpiceResult<()> {
    if parameter.name == "charge" {
        return Ok(());
    }
    let functions = scope.functions().map(Arc::as_ref);
    let numeric;
    let expression = match &parameter.kind {
        ParameterKind::Expression(value) => {
            // A simple arithmetic parameter value cannot hide a runtime law.
            let potentially_runtime = value.root.preorder().any(|node| match &node.kind {
                ExprKind::Identifier(name) => matches!(name.as_str(), "time" | "temper" | "hertz"),
                ExprKind::UserCall { .. } => true,
                ExprKind::Call { function, .. } => {
                    functions.is_some_and(|scope| scope.get(function.name()).is_some())
                }
                _ => false,
            });
            if !potentially_runtime {
                return Ok(());
            }
            numeric = BehaviouralExpression {
                text: value.spelling(),
                span: value.span.clone(),
                verbatim: false,
                root: from_numparam(
                    &value.root,
                    &[],
                    &[],
                    1,
                    &mut TreeBudget::new(MAX_TREE_NODES),
                    functions,
                )?,
            };
            &numeric
        }
        ParameterKind::Behavioural(expression) => expression,
        _ => return Ok(()),
    };
    let root = expand_symbolic(expression, scope, budget)?;
    let mut runtime = false;
    root.visit(&mut |expr| {
        runtime |= match &expr.kind {
            BExprKind::Voltage { .. } | BExprKind::Current(_) => true,
            BExprKind::Name(name) => matches!(name.as_str(), "time" | "temper" | "hertz"),
            BExprKind::Call { name, .. } => name == "ddt",
            _ => false,
        };
    });
    if runtime {
        // The symbolic expansion classifies the law, but retain the original
        // call tree for realization: otherwise macro invocation identities
        // are lost before statistical calls can be bound.
        parameter.kind = ParameterKind::Behavioural(Box::new(expression.clone()));
    } else if matches!(parameter.kind, ParameterKind::Behavioural(_)) {
        // A locally shadowed probe call or an erased macro argument may
        // make the syntactically behavioural candidate an ordinary scalar.
        // Preserve the original numeric evaluation and its site metadata.
        let text = &expression.text;
        let inner = text.get(1..text.len().saturating_sub(1)).ok_or_else(|| {
            parse_error(
                &expression.span,
                "missing passive expression delimiters".into(),
            )
        })?;
        let mut value = crate::netlist::parse_parameter_expression(
            inner,
            &expression
                .span
                .start
                .at_column(expression.span.start.column + 1),
        )?;
        value.braced = true;
        value.quoted = text.starts_with('\'');
        parameter.kind = ParameterKind::Expression(Box::new(value));
    }
    Ok(())
}

/// Static argument arithmetic follows the same compiled expression semantics
/// as the surrounding behavioural law, but cannot read any circuit quantity.
fn statistical_argument(expr: &BExpr) -> SpiceResult<Real> {
    let mut dynamic = false;
    expr.visit(&mut |node| {
        dynamic |= matches!(
            &node.kind,
            BExprKind::Voltage { .. } | BExprKind::Current(_)
        ) || matches!(&node.kind, BExprKind::Name(name) if !matches!(name.as_str(),"pi"|"e"))
            || matches!(&node.kind, BExprKind::Call { name,.. } if name == "ddt");
    });
    if dynamic {
        return Err(parse_error(
            &expr.span,
            "statistical function requires static arguments, not a circuit-dependent value".into(),
        ));
    }
    let program = crate::devices::behavioural::program::Program::compile(expr)?;
    Ok(program
        .evaluate(&crate::devices::behavioural::program::Environment {
            values: &[],
            time: 0.,
            temperature: 0.,
            gmin: 0.,
            frequency: 0.,
        })?
        .value)
}

struct Resolver<'a> {
    scope: &'a ParamScope,
    functions: Option<Arc<FunctionScope>>,
    budget: &'a mut EvalBudget,
    sampler: crate::netlist::eval::statistics::Sampler<'a>,
    prepared_frames: Vec<(SourceSpan, model_statistics::Identity)>,
    exact: bool,
    verbatim: bool,
    syntax_only: bool,
    nodes: usize,
    expansion_budget: TreeBudget,
}

fn parse_error(span: &SourceSpan, message: String) -> SpiceError {
    SpiceError::parse(span.start.clone(), message)
}

impl Resolver<'_> {
    // Macro arguments retain their original source spans when textually
    // substituted. Resolve their prepared identity in the caller frame, not
    // the callee frame: repeated formal reads must not resample the argument.
    fn prepared_frame(&self, span: &SourceSpan) -> Option<(model_statistics::Identity, u64)> {
        self.prepared_frames
            .iter()
            .rev()
            .find_map(|(parent, identity)| {
                let start = (span.start.line, span.start.column);
                let end = (span.end.line, span.end.column);
                let parent_start = (parent.start.line, parent.start.column);
                let parent_end = (parent.end.line, parent.end.column);
                if span.start.path() != parent.start.path()
                    || start < parent_start
                    || end > parent_end
                {
                    return None;
                }
                span.relative_to(parent)
                    .map(|range| (*identity, range.start as u64))
            })
    }

    fn numeric_value(&self, value: Real) -> BExprKind {
        if self.exact {
            BExprKind::Number {
                value,
                spelling: format_literal(value),
            }
        } else {
            spliced(value)
        }
    }

    fn resolve(&mut self, expr: &BExpr, depth: usize) -> SpiceResult<BExpr> {
        self.nodes += 1;
        if self.nodes > MAX_TREE_NODES || depth > MAX_TREE_DEPTH {
            return Err(parse_error(
                &expr.span,
                format!(
                    "behavioural expression too large after .func expansion \
                     (more than {MAX_TREE_NODES} nodes or nesting deeper than {MAX_TREE_DEPTH})"
                ),
            ));
        }
        let boxed = |this: &mut Self, expr: &BExpr| this.resolve(expr, depth + 1).map(Box::new);
        let kind = match &expr.kind {
            BExprKind::Number { value, spelling } => BExprKind::Number {
                value: if self.verbatim || self.syntax_only || self.exact {
                    *value
                } else {
                    round_like_c_literal(*value)
                },
                spelling: spelling.clone(),
            },
            BExprKind::Name(name) => return self.name(expr, name),
            BExprKind::Voltage { positive, negative } => {
                if self
                    .functions
                    .as_deref()
                    .is_some_and(|scope| scope.get("v").is_some())
                {
                    return self.shadowed_probe(
                        expr,
                        "v",
                        std::iter::once(positive.as_str()).chain(negative.as_deref()),
                        depth,
                    );
                }
                expr.kind.clone()
            }
            BExprKind::Current(name) => {
                if self
                    .functions
                    .as_deref()
                    .is_some_and(|scope| scope.get("i").is_some())
                {
                    return self.shadowed_probe(expr, "i", std::iter::once(name.as_str()), depth);
                }
                expr.kind.clone()
            }
            BExprKind::Unary { op, operand } => BExprKind::Unary {
                op: *op,
                operand: boxed(self, operand)?,
            },
            BExprKind::Binary { op, lhs, rhs } => BExprKind::Binary {
                op: *op,
                lhs: boxed(self, lhs)?,
                rhs: boxed(self, rhs)?,
            },
            BExprKind::Ternary {
                condition,
                then,
                otherwise,
            } => BExprKind::Ternary {
                condition: boxed(self, condition)?,
                then: boxed(self, then)?,
                otherwise: boxed(self, otherwise)?,
            },
            BExprKind::Group(inner) => BExprKind::Group(boxed(self, inner)?),
            BExprKind::Call { name, arguments } => {
                return self.call(expr, name, arguments, depth);
            }
            BExprKind::Value(value) if !self.syntax_only => {
                let identity = self
                    .prepared_frame(&expr.span)
                    .map(|(identity, _)| identity);
                let evaluated =
                    self.scope
                        .evaluate_sampled(value, self.budget, &self.sampler, identity)?;
                self.numeric_value(evaluated)
            }
            BExprKind::Value(_) => expr.kind.clone(),
            BExprKind::Table(table) => {
                let input = self.resolve(&table.input, depth + 1)?;
                let mut points = Vec::with_capacity(table.points.len());
                for (x, y) in &table.points {
                    points.push((self.resolve(x, depth + 1)?, self.resolve(y, depth + 1)?));
                }
                BExprKind::Table(Box::new(TableTransfer {
                    input,
                    points,
                    domain: table.domain,
                }))
            }
        };
        Ok(BExpr {
            kind,
            span: expr.span.clone(),
        })
    }

    fn name(&self, expr: &BExpr, name: &str) -> SpiceResult<BExpr> {
        if self.syntax_only || SPECIAL_NAMES.contains(&name) {
            return Ok(expr.clone());
        }
        if self.verbatim {
            return Err(parse_error(
                &expr.span,
                format!(
                    "bare name '{name}' in a verbatim (=pwl) behavioural expression: C does not \
                     substitute it (the B-source parser fails); write {{{name}}}"
                ),
            ));
        }
        let value = self.scope.get(name).ok_or_else(|| {
            parse_error(
                &expr.span,
                format!("undefined parameter [{name}] in a behavioural expression"),
            )
        })?;
        Ok(BExpr {
            kind: self.numeric_value(value),
            span: expr.span.clone(),
        })
    }

    fn shadowed_probe<'a>(
        &mut self,
        expr: &BExpr,
        name: &str,
        operands: impl Iterator<Item = &'a str>,
        depth: usize,
    ) -> SpiceResult<BExpr> {
        let arguments = operands
            .map(|operand| {
                crate::netlist::Parser::new()
                    .parse_behavioural_expression(operand, &expr.span.start, false)
                    .map(|expression| expression.root)
            })
            .collect::<SpiceResult<Vec<_>>>()?;
        self.call(expr, name, &arguments, depth)
    }

    fn prepared_call(
        &mut self,
        expr: &BExpr,
        name: &str,
        arguments: &[BExpr],
        depth: usize,
    ) -> SpiceResult<BExpr> {
        let identity = self
            .prepared_frame(&expr.span)
            .map(|(identity, _)| identity);
        if identity.is_none() && self.sampler.prepared_site().is_some() {
            return Err(parse_error(
                &expr.span,
                "prepared statistical call has no static expression identity".into(),
            ));
        }
        let previous = std::mem::replace(&mut self.exact, true);
        let values = arguments
            .iter()
            .map(|argument| {
                let resolved = self.resolve(argument, depth + 1)?;
                statistical_argument(&resolved)
            })
            .collect::<SpiceResult<Vec<_>>>();
        self.exact = previous;
        let value = self
            .sampler
            .sample_prepared(name, &values?, identity, &expr.span.start)?;
        Ok(BExpr {
            kind: self.numeric_value(value),
            span: expr.span.clone(),
        })
    }

    fn call(
        &mut self,
        expr: &BExpr,
        name: &str,
        arguments: &[BExpr],
        depth: usize,
    ) -> SpiceResult<BExpr> {
        if !self.syntax_only && name.starts_with(model_statistics::PREPARED_PREFIX) {
            return self.prepared_call(expr, name, arguments, depth);
        }
        let functions = self.functions.clone();
        let definition = functions
            .as_deref()
            .and_then(|scope| scope.get(name))
            .map(|(definition, _)| definition);
        if let Some(definition) = definition {
            if definition.parameters.len() != arguments.len() {
                return Err(parse_error(
                    &expr.span,
                    format!(
                        "function '{name}' (defined at {}) takes {} argument(s), found {}",
                        definition.location,
                        definition.parameters.len(),
                        arguments.len()
                    ),
                ));
            }
            // inp_expand_macro_in_str(): the body replaces the call, each
            // formal by its parenthesised argument text.
            let body = from_numparam(
                &definition.body.root,
                &definition.parameters,
                arguments,
                depth + 1,
                &mut self.expansion_budget,
                functions.as_deref(),
            )?;
            let frame = self.prepared_frame(&expr.span).map(|(identity, offset)| {
                (
                    definition.body.span.clone(),
                    identity
                        .child(b"macro")
                        .child(definition.name.as_bytes())
                        .child(&offset.to_le_bytes()),
                )
            });
            let pushed = frame.is_some();
            self.prepared_frames.extend(frame);
            let expanded = self.resolve(&body, depth + 1);
            if pushed {
                self.prepared_frames.pop();
            }
            return Ok(BExpr {
                kind: BExprKind::Group(Box::new(expanded?)),
                span: expr.span.clone(),
            });
        }
        if !self.syntax_only && STATISTICAL_FUNCTIONS.contains(&name) {
            let mut values = Vec::with_capacity(arguments.len());
            for argument in arguments {
                let resolved = self.resolve(argument, depth + 1)?;
                values.push(statistical_argument(&resolved)?);
            }
            let value = self.sampler.sample(name, &values, &expr.span.start)?;
            return Ok(BExpr {
                kind: spliced(value),
                span: expr.span.clone(),
            });
        }
        if !self.syntax_only && !is_builtin_function(name) {
            return Err(SpiceError::parse(
                expr.span.start.clone(),
                format!(
                    "no such function '{name}' in a behavioural expression \
                     (src/spicelib/parser/inpptree.c PT_mkfnode)"
                ),
            ));
        }
        let mut resolved = Vec::with_capacity(arguments.len());
        for argument in arguments {
            resolved.push(self.resolve(argument, depth + 1)?);
        }
        Ok(BExpr {
            kind: BExprKind::Call {
                name: name.to_owned(),
                arguments: resolved,
            },
            span: expr.span.clone(),
        })
    }
}

/// Converts a numparam `.func` body into a behavioural tree, binding formals.
/// `v(name[, name])`/`i(name)` calls in a body name nodes/sources literally:
/// C does not substitute formals inside them (verified with the C binary).
fn from_numparam(
    expr: &Expr,
    formals: &[String],
    arguments: &[BExpr],
    depth: usize,
    budget: &mut TreeBudget,
    functions: Option<&FunctionScope>,
) -> SpiceResult<BExpr> {
    budget.charge(depth, 1, &expr.span.start)?;
    let mut convert = |expr: &Expr| {
        from_numparam(expr, formals, arguments, depth + 1, budget, functions).map(Box::new)
    };
    let kind = match &expr.kind {
        ExprKind::Number { value, spelling } => BExprKind::Number {
            value: *value,
            spelling: spelling.clone(),
        },
        ExprKind::Identifier(name) => {
            if let Some(index) = formals.iter().position(|formal| formal == name) {
                let argument = &arguments[index];
                argument.check_tree(depth + 1, budget)?;
                return Ok(BExpr {
                    kind: BExprKind::Group(Box::new(argument.clone())),
                    span: argument.span.clone(),
                });
            }
            BExprKind::Name(name.clone())
        }
        ExprKind::Unary { op, operand } => BExprKind::Unary {
            op: match op {
                UnaryOp::Plus => BUnaryOp::Plus,
                UnaryOp::Minus => BUnaryOp::Minus,
            },
            operand: convert(operand)?,
        },
        ExprKind::Binary { op, lhs, rhs } => BExprKind::Binary {
            op: match op {
                BinaryOp::Add => BBinaryOp::Add,
                BinaryOp::Sub => BBinaryOp::Sub,
                BinaryOp::Mul => BBinaryOp::Mul,
                BinaryOp::Div => BBinaryOp::Div,
                BinaryOp::Pow => BBinaryOp::Pow,
            },
            lhs: convert(lhs)?,
            rhs: convert(rhs)?,
        },
        ExprKind::Call {
            function,
            arguments: operands,
        } => BExprKind::Call {
            name: function.name().to_owned(),
            arguments: operands
                .iter()
                .map(|operand| {
                    from_numparam(operand, formals, arguments, depth + 1, budget, functions)
                })
                .collect::<SpiceResult<_>>()?,
        },
        ExprKind::Group(inner) => BExprKind::Group(convert(inner)?),
        ExprKind::UserCall {
            name,
            arguments: operands,
        } => {
            if functions.is_none_or(|scope| scope.get(name).is_none())
                && let Some(kind) = circuit_quantity(name, operands)
            {
                kind
            } else {
                BExprKind::Call {
                    name: name.clone(),
                    arguments: operands
                        .iter()
                        .map(|operand| {
                            from_numparam(operand, formals, arguments, depth + 1, budget, functions)
                        })
                        .collect::<SpiceResult<_>>()?,
                }
            }
        }
    };
    Ok(BExpr {
        kind,
        span: expr.span.clone(),
    })
}

fn literal_name(expr: &Expr) -> Option<String> {
    match &expr.kind {
        ExprKind::Identifier(name) => Some(name.clone()),
        ExprKind::Number { spelling, .. } => Some(spelling.to_ascii_lowercase()),
        _ => None,
    }
}

fn circuit_quantity(name: &str, arguments: &[Expr]) -> Option<BExprKind> {
    match (name, arguments) {
        ("v", [positive]) => Some(BExprKind::Voltage {
            positive: literal_name(positive)?,
            negative: None,
        }),
        ("v", [positive, negative]) => Some(BExprKind::Voltage {
            positive: literal_name(positive)?,
            negative: Some(literal_name(negative)?),
        }),
        ("i", [source]) => Some(BExprKind::Current(literal_name(source)?)),
        _ => None,
    }
}
