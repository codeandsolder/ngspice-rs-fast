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
    BBinaryOp, BExpr, BExprKind, BehaviouralExpression, SPECIAL_NAMES, STATISTICAL_FUNCTIONS,
    TableTransfer, is_builtin_function, round_like_c_literal,
};
use crate::netlist::elaborate::format_literal;
use crate::netlist::eval::statistics::PreparedSite;
use crate::netlist::eval::{EvalBudget, FunctionDef, FunctionScope, ParamScope};
use crate::netlist::expr::{Expr, ExprKind, MAX_TREE_NODES, SourceSpan, TreeBudget};

mod edits;
pub(super) mod forms;
use forms::{NonlinearForm, PolyControls, PolyForm, TableForm, TableInput, ValueForm};

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
        Some(NonlinearForm::Poly(form)) => out.push(lower_poly(&device, &form)?),
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
        scalar_alternative: None,
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

/// In-place SPICE2 monomial sequence (`nxtpwr` in
/// `src/xspice/icm/spice2poly/icm_spice2poly/cfunc.mod`).
///
/// Starts at the constant term. Each advance visits the next composition
/// in increasing total degree, then decreasing lexicographic order.
/// A single vector is sufficient to lower every coefficient.
struct PolyPowers {
    powers: Vec<u32>,
}

impl PolyPowers {
    fn new(dimension: usize) -> Option<Self> {
        (dimension != 0).then(|| Self {
            powers: vec![0; dimension],
        })
    }

    fn advance(&mut self) -> Option<&[u32]> {
        let last = self.powers.len() - 1;
        if let Some(index) = self.powers[..last].iter().rposition(|&power| power != 0) {
            // Move one power from the rightmost nonzero prefix element,
            // along with all remaining powers, to its next neighbour.
            let tail = self.powers[last];
            self.powers[last] = 0;
            self.powers[index] -= 1;
            self.powers[index + 1] = tail + 1;
        } else {
            // The previous degree is exhausted, or this is the constant term.
            let degree = self.powers[last].checked_add(1)?;
            self.powers[last] = 0;
            self.powers[0] = degree;
        }
        Some(&self.powers)
    }
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

fn lower_poly(device: &DeviceInstance, form: &PolyForm<'_>) -> SpiceResult<DeviceInstance> {
    let dimension = form.dimension;
    let location = &device.location;
    let controls: Vec<_> = match form.controls {
        PolyControls::Voltage(pairs) => pairs
            .iter()
            .map(|[positive, negative]| BExprKind::Voltage {
                positive: positive.clone(),
                negative: Some(negative.clone()),
            })
            .collect(),
        PolyControls::Current(sources) => sources
            .iter()
            .map(|parameter| BExprKind::Current(parameter.value.clone()))
            .collect(),
    };
    let mut coefficients = form.coefficients.iter();
    let first = coefficients
        .next()
        .ok_or_else(|| malformed(device, "POLY needs at least one coefficient"))?;
    let mut powers = PolyPowers::new(dimension)
        .ok_or_else(|| malformed(device, "POLY needs a positive integer dimension"))?;
    let mut coefficient_count = 1;
    let node = |kind: BExprKind| quantity(kind, location);
    let binary = |op, lhs: BExpr, rhs: BExpr| {
        node(BExprKind::Binary {
            op,
            lhs: Box::new(lhs),
            rhs: Box::new(rhs),
        })
    };
    let mut budget = TreeBudget::new(MAX_TREE_NODES);
    let mut height = check_coefficient(first.parameter, 1, &mut budget)?;
    let mut sum = first.expression();
    for coefficient in coefficients {
        coefficient_count += 1;
        let exponents = powers
            .advance()
            .ok_or_else(|| malformed(device, "POLY degree exceeds supported range"))?;
        // Calculate the exact height and node count of this monomial's
        // left-folded product before constructing any of its factors.
        let degree = exponents.iter().map(|&power| power as usize).sum::<usize>();
        let coefficient_height = check_coefficient(coefficient.parameter, 1, &mut budget)?;
        let term_height = coefficient_height.max(degree) + 1;
        height = height.max(term_height) + 1;
        // degree factors, degree-1 products, coefficient multiplication and sum.
        budget.charge(height, 2 * degree + 1, &coefficient.parameter.location)?;
        let mut product: Option<BExpr> = None;
        for (control, power) in controls.iter().zip(exponents) {
            for _ in 0..*power {
                let factor = node(control.clone());
                product = Some(match product {
                    None => factor,
                    Some(lhs) => binary(BBinaryOp::Mul, lhs, factor),
                });
            }
        }
        let term = binary(
            BBinaryOp::Mul,
            coefficient.expression(),
            product.ok_or_else(|| {
                malformed(device, "POLY nonconstant term has no controlling factors")
            })?,
        );
        sum = binary(BBinaryOp::Add, sum, term);
    }
    let output = if matches!(device.designator, 'e' | 'h') {
        "v"
    } else {
        "i"
    };
    let text = format!("spice2poly({dimension}) with {coefficient_count} coefficient(s)");
    let mut parameters = vec![ParameterAssignment {
        name: output.to_owned(),
        value: text.clone(),
        kind: generated(sum, text, location),
        location: location.clone(),
    }];
    parameters.extend_from_slice(form.setters);
    Ok(instance(
        device.name.clone(),
        device.designator,
        device.nodes[..2].to_vec(),
        parameters,
        location,
    ))
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
        functions: scope.functions().map(Arc::as_ref),
        budget,
        sampler: scope.sampler(&expression.span.start, site),
        prepared_frames: site
            .into_iter()
            .map(|site| (expression.span.clone(), site.expression))
            .collect(),
        exact: false,
        verbatim: expression.verbatim,
        syntax_only: false,
        expansion_budget: TreeBudget::new(MAX_TREE_NODES),
    };
    let root = resolver.resolve(&expression.root, 1, None)?;
    root.check_tree(1, &mut TreeBudget::new(MAX_TREE_NODES))?;
    Ok(BehaviouralExpression {
        text: expression.text.clone(),
        span: expression.span.clone(),
        verbatim: expression.verbatim,
        scalar_alternative: None,
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
        functions: scope.functions().map(Arc::as_ref),
        budget,
        sampler: scope.sampler(&expression.span.start, None),
        prepared_frames: Vec::new(),
        exact: false,
        verbatim: expression.verbatim,
        syntax_only: true,
        expansion_budget: TreeBudget::new(MAX_TREE_NODES),
    };
    let root = resolver.resolve(&expression.root, 1, None)?;
    root.check_tree(1, &mut TreeBudget::new(MAX_TREE_NODES))?;
    Ok(root)
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
    let selected = match &parameter.kind {
        ParameterKind::Expression(value) => {
            let Some(alternative) = &value.behavioural_alternative else {
                return Ok(());
            };
            match alternative
                .as_ref()
                .as_ref()
                .map_err(Clone::clone)
                .and_then(|expression| {
                    expand_symbolic(expression, scope, budget).map(|root| (expression, root))
                }) {
                Ok((expression, root)) if root.uses_runtime() => {
                    ParameterKind::Behavioural(Box::new(BehaviouralExpression {
                        scalar_alternative: None,
                        ..expression.clone()
                    }))
                }
                Ok(_) => return Ok(()),
                Err(error) => {
                    // A scalar-only operator or function can be legal when
                    // the expansion has no runtime dependencies. Use the
                    // scalar grammar only to decide whether that phase is
                    // still viable, never to construct a behavioural tree.
                    if has_runtime(
                        &value.root,
                        functions,
                        None,
                        1,
                        &mut TreeBudget::new(MAX_TREE_NODES),
                    )? {
                        return Err(error);
                    }
                    return Ok(());
                }
            }
        }
        ParameterKind::Behavioural(expression) => {
            match expand_symbolic(expression, scope, budget) {
                Ok(root) if root.uses_runtime() => {
                    ParameterKind::Behavioural(Box::new(BehaviouralExpression {
                        scalar_alternative: None,
                        ..expression.as_ref().clone()
                    }))
                }
                Ok(_) => {
                    // Directly written behavioural laws without a retained
                    // scalar alternative are already classified.
                    let Some(alternative) = &expression.scalar_alternative else {
                        return Ok(());
                    };
                    ParameterKind::Expression(Box::new(
                        alternative.as_ref().as_ref().map_err(Clone::clone)?.clone(),
                    ))
                }
                Err(error) => {
                    let Some(Ok(scalar)) = expression.scalar_alternative.as_deref() else {
                        return Err(error);
                    };
                    if has_runtime(
                        &scalar.root,
                        functions,
                        None,
                        1,
                        &mut TreeBudget::new(MAX_TREE_NODES),
                    )? {
                        return Err(error);
                    }
                    ParameterKind::Expression(Box::new(scalar.clone()))
                }
            }
        }
        _ => return Ok(()),
    };
    parameter.kind = selected;
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

/// Substitution retains the caller environment for each argument while
/// resolving free names through the enclosing active macro chain.
struct MacroFrame<'a, T> {
    def: &'a FunctionDef,
    scope: &'a FunctionScope,
    arguments: &'a [T],
    outer: Option<&'a Self>,
}

fn macro_definition<'a, T>(
    site: Option<&'a FunctionScope>,
    frame: Option<&'a MacroFrame<'a, T>>,
    name: &str,
) -> Option<(&'a FunctionDef, &'a FunctionScope)> {
    frame
        .map(|f| f.scope)
        .or(site)
        .and_then(|scope| scope.get(name))
        .or_else(|| site.and_then(|scope| scope.get(name)))
}

fn macro_argument<'a, T>(
    mut frame: Option<&'a MacroFrame<'a, T>>,
    name: &str,
) -> Option<(&'a T, Option<&'a MacroFrame<'a, T>>)> {
    while let Some(f) = frame {
        if let Some(index) = f.def.parameters.iter().position(|p| p == name) {
            return Some((&f.arguments[index], f.outer));
        }
        frame = f.outer;
    }
    None
}

fn check_macro_call<T>(
    def: &FunctionDef,
    arguments: &[T],
    mut frame: Option<&MacroFrame<'_, T>>,
    span: &SourceSpan,
) -> SpiceResult<()> {
    if def.parameters.len() != arguments.len() {
        return Err(parse_error(
            span,
            format!(
                "function '{}' (defined at {}) takes {} argument(s), found {}",
                def.name,
                def.location,
                def.parameters.len(),
                arguments.len()
            ),
        ));
    }
    while let Some(f) = frame {
        if std::ptr::eq(f.def, def) {
            return Err(parse_error(
                span,
                format!(
                    "recursive call of function '{}' (defined at {})",
                    def.name, def.location
                ),
            ));
        }
        frame = f.outer;
    }
    Ok(())
}

/// Classify the expanded scalar syntax without evaluating values, or
/// imposing behavioural operators on a scalar-only expression.
fn has_runtime(
    expr: &Expr,
    site: Option<&FunctionScope>,
    frame: Option<&MacroFrame<'_, Expr>>,
    depth: usize,
    budget: &mut TreeBudget,
) -> SpiceResult<bool> {
    budget.charge(depth, 1, &expr.span.start)?;
    let mut child = |node: &Expr| has_runtime(node, site, frame, depth + 1, budget);
    match &expr.kind {
        ExprKind::Number { .. } => Ok(false),
        ExprKind::Identifier(name) => {
            if let Some((argument, outer)) = macro_argument(frame, name) {
                has_runtime(argument, site, outer, depth + 1, budget)
            } else {
                Ok(matches!(name.as_str(), "time" | "temper" | "hertz"))
            }
        }
        ExprKind::Unary { operand, .. } | ExprKind::Group(operand) => child(operand),
        ExprKind::Binary { lhs, rhs, .. } => Ok(child(lhs)? || child(rhs)?),
        ExprKind::Ternary {
            condition,
            then,
            otherwise,
        } => Ok(child(condition)? || child(then)? || child(otherwise)?),
        ExprKind::Call {
            function,
            arguments,
        } => runtime_call(function.name(), arguments, expr, site, frame, depth, budget),
        ExprKind::UserCall { name, arguments } => {
            runtime_call(name, arguments, expr, site, frame, depth, budget)
        }
    }
}

fn runtime_call(
    name: &str,
    arguments: &[Expr],
    expr: &Expr,
    site: Option<&FunctionScope>,
    outer: Option<&MacroFrame<'_, Expr>>,
    depth: usize,
    budget: &mut TreeBudget,
) -> SpiceResult<bool> {
    if let Some((def, scope)) = macro_definition(site, outer, name) {
        check_macro_call(def, arguments, outer, &expr.span)?;
        let frame = MacroFrame {
            def,
            scope,
            arguments,
            outer,
        };
        // A behavioural-only definition must be resolved in that phase
        // before the final classification can be made.
        return match &def.body.scalar {
            Ok(body) => has_runtime(body, site, Some(&frame), depth + 1, budget),
            Err(_) => Ok(true),
        };
    }
    if matches!(name, "v" | "i" | "ddt") {
        return Ok(true);
    }
    for argument in arguments {
        if has_runtime(argument, site, outer, depth + 1, budget)? {
            return Ok(true);
        }
    }
    Ok(false)
}

struct Resolver<'a> {
    scope: &'a ParamScope,
    functions: Option<&'a FunctionScope>,
    budget: &'a mut EvalBudget,
    sampler: crate::netlist::eval::statistics::Sampler<'a>,
    prepared_frames: Vec<(SourceSpan, model_statistics::Identity)>,
    exact: bool,
    verbatim: bool,
    syntax_only: bool,
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

    fn resolve(
        &mut self,
        expr: &BExpr,
        depth: usize,
        frame: Option<&MacroFrame<'_, BExpr>>,
    ) -> SpiceResult<BExpr> {
        self.expansion_budget.charge(depth, 1, &expr.span.start)?;
        let boxed =
            |this: &mut Self, expr: &BExpr| this.resolve(expr, depth + 1, frame).map(Box::new);
        let kind = match &expr.kind {
            BExprKind::Number { value, spelling } => BExprKind::Number {
                value: if self.verbatim || self.syntax_only || self.exact {
                    *value
                } else {
                    round_like_c_literal(*value)
                },
                spelling: spelling.clone(),
            },
            BExprKind::Name(name) => {
                if let Some((argument, outer)) = macro_argument(frame, name) {
                    return Ok(BExpr {
                        kind: BExprKind::Group(Box::new(self.resolve(
                            argument,
                            depth + 1,
                            outer,
                        )?)),
                        span: argument.span.clone(),
                    });
                }
                return self.name(expr, name);
            }
            BExprKind::Voltage { positive, negative } => {
                if macro_definition(self.functions, frame, "v").is_some() {
                    return self.shadowed_probe(
                        expr,
                        "v",
                        std::iter::once(positive.as_str()).chain(negative.as_deref()),
                        depth,
                        frame,
                    );
                }
                expr.kind.clone()
            }
            BExprKind::Current(name) => {
                if macro_definition(self.functions, frame, "i").is_some() {
                    return self.shadowed_probe(
                        expr,
                        "i",
                        std::iter::once(name.as_str()),
                        depth,
                        frame,
                    );
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
                return self.call(expr, name, arguments, depth, frame);
            }
            BExprKind::Value(value) => {
                value
                    .root
                    .check_tree(depth + 1, &mut self.expansion_budget)?;
                if self.syntax_only {
                    expr.kind.clone()
                } else {
                    let identity = self
                        .prepared_frame(&expr.span)
                        .map(|(identity, _)| identity);
                    let evaluated =
                        self.scope
                            .evaluate_sampled(value, self.budget, &self.sampler, identity)?;
                    self.numeric_value(evaluated)
                }
            }
            BExprKind::Table(table) => {
                let input = self.resolve(&table.input, depth + 1, frame)?;
                let mut points = Vec::with_capacity(table.points.len());
                for (x, y) in &table.points {
                    points.push((
                        self.resolve(x, depth + 1, frame)?,
                        self.resolve(y, depth + 1, frame)?,
                    ));
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
        frame: Option<&MacroFrame<'_, BExpr>>,
    ) -> SpiceResult<BExpr> {
        let arguments = operands
            .map(|operand| {
                crate::netlist::Parser::new()
                    .parse_behavioural_expression(operand, &expr.span.start, false)
                    .map(|expression| expression.root)
            })
            .collect::<SpiceResult<Vec<_>>>()?;
        self.call(expr, name, &arguments, depth, frame)
    }

    fn prepared_call(
        &mut self,
        expr: &BExpr,
        name: &str,
        arguments: &[BExpr],
        depth: usize,
        frame: Option<&MacroFrame<'_, BExpr>>,
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
                let resolved = self.resolve(argument, depth + 1, frame)?;
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
        frame: Option<&MacroFrame<'_, BExpr>>,
    ) -> SpiceResult<BExpr> {
        if !self.syntax_only && name.starts_with(model_statistics::PREPARED_PREFIX) {
            return self.prepared_call(expr, name, arguments, depth, frame);
        }
        if let Some((definition, scope)) = macro_definition(self.functions, frame, name) {
            check_macro_call(definition, arguments, frame, &expr.span)?;
            let body = definition.body.behavioural.as_ref().map_err(Clone::clone)?;
            let inner = MacroFrame {
                def: definition,
                scope,
                arguments,
                outer: frame,
            };
            let identity_frame = self.prepared_frame(&expr.span).map(|(identity, offset)| {
                (
                    definition.body.span.clone(),
                    identity
                        .child(b"macro")
                        .child(definition.name.as_bytes())
                        .child(&offset.to_le_bytes()),
                )
            });
            let pushed = identity_frame.is_some();
            self.prepared_frames.extend(identity_frame);
            let expanded = self.resolve(body, depth + 1, Some(&inner));
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
                let resolved = self.resolve(argument, depth + 1, frame)?;
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
            resolved.push(self.resolve(argument, depth + 1, frame)?);
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

#[cfg(test)]
mod tests {
    use super::PolyPowers;

    fn poly_exponents(dimension: usize, count: usize) -> Vec<Vec<u32>> {
        let mut powers = PolyPowers::new(dimension).unwrap();
        (0..count)
            .map(|_| powers.advance().unwrap().to_vec())
            .collect()
    }

    /// Independent recursive composition enumerator, limited to the tiny
    /// reference cases; the production lowering never allocates these lists.
    fn reference_terms(dimension: usize, count: usize) -> Vec<Vec<u32>> {
        fn compose(
            prefix: &mut Vec<u32>,
            remaining: u32,
            dimension: usize,
            out: &mut Vec<Vec<u32>>,
            count: usize,
        ) {
            if out.len() == count {
                return;
            }
            if prefix.len() + 1 == dimension {
                prefix.push(remaining);
                out.push(prefix.clone());
                prefix.pop();
            } else {
                for power in (0..=remaining).rev() {
                    prefix.push(power);
                    compose(prefix, remaining - power, dimension, out, count);
                    prefix.pop();
                    if out.len() == count {
                        break;
                    }
                }
            }
        }
        let mut result = Vec::new();
        let mut degree = 1;
        while result.len() < count {
            compose(&mut Vec::new(), degree, dimension, &mut result, count);
            degree += 1;
        }
        result
    }

    #[test]
    fn poly_power_cursor_matches_compositions() {
        for dimension in [1, 2, 3, 4, 5, 6, 64] {
            assert_eq!(
                poly_exponents(dimension, 300),
                reference_terms(dimension, 300)
            );
        }
        assert!(PolyPowers::new(0).is_none());
        let mut overflow = PolyPowers {
            powers: vec![u32::MAX],
        };
        assert!(overflow.advance().is_none());
        let mut limit = PolyPowers {
            powers: vec![1, u32::MAX - 1],
        };
        assert_eq!(limit.advance(), Some([0, u32::MAX].as_slice()));
        assert!(limit.advance().is_none());
    }

    #[test]
    fn poly_terms_follow_the_spice2_order() {
        assert_eq!(
            poly_exponents(2, 5),
            vec![vec![1, 0], vec![0, 1], vec![2, 0], vec![1, 1], vec![0, 2]]
        );
        assert_eq!(
            poly_exponents(3, 9),
            vec![
                vec![1, 0, 0],
                vec![0, 1, 0],
                vec![0, 0, 1],
                vec![2, 0, 0],
                vec![1, 1, 0],
                vec![1, 0, 1],
                vec![0, 2, 0],
                vec![0, 1, 1],
                vec![0, 0, 2],
            ]
        );
        assert_eq!(poly_exponents(1, 3), vec![vec![1], vec![2], vec![3]]);
        assert_eq!(poly_exponents(2, 7)[5], vec![3, 0]);
    }
}
