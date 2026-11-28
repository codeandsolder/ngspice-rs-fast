//! Behavioural sources (GitHub #79): the B card and nonlinear E/G/F/H
//! grammars, positioned errors, writer round trips, front-end resolution of
//! `.param`/`.func`/numparam values and the `inpcom.c` lowering passes.
use std::path::Path;
use std::sync::Arc;

use ngspice_rs::netlist::ast::{DeviceInstance, Netlist, ParameterKind};
use ngspice_rs::netlist::behavioural::lower_nonlinear_sources;
use ngspice_rs::netlist::bexpr::{BExpr, BExprKind, BehaviouralExpression};
use ngspice_rs::netlist::elaborate::literalize;
use ngspice_rs::netlist::eval::{EvalBudget, ParamScope};
use ngspice_rs::netlist::source::parse_deck_text;
use ngspice_rs::netlist::{Parser, semantic_diff, write_netlist};
use ngspice_rs::primitives::SpiceError;

fn try_parse(body: &str) -> Result<Netlist, SpiceError> {
    Parser::new().parse_deck(&parse_deck_text(
        Path::new("in.cir"),
        &format!("behavioural\n{body}\n.end\n"),
    ))
}

fn parse(body: &str) -> Netlist {
    try_parse(body).unwrap_or_else(|error| panic!("{error}\n{body}"))
}

fn device(body: &str) -> DeviceInstance {
    parse(body).devices.last().unwrap().clone()
}

fn expression(device: &DeviceInstance, name: &str) -> BehaviouralExpression {
    let parameter = device
        .parameters
        .iter()
        .find(|p| p.name == name)
        .unwrap_or_else(|| panic!("{} has no {name}", device.name));
    match &parameter.kind {
        ParameterKind::Behavioural(expression) => (**expression).clone(),
        other => panic!("{name} is {other:?}"),
    }
}

fn names(device: &DeviceInstance) -> Vec<&str> {
    device.parameters.iter().map(|p| p.name.as_str()).collect()
}

/// A compact rendering of a tree, groups dropped.
fn shape(expr: &BExpr) -> String {
    match &expr.kind {
        BExprKind::Number { value, .. } => format!("{value}"),
        BExprKind::Name(name) => name.clone(),
        BExprKind::Voltage { positive, negative } => match negative {
            Some(negative) => format!("v({positive},{negative})"),
            None => format!("v({positive})"),
        },
        BExprKind::Current(name) => format!("i({name})"),
        BExprKind::Unary { op, operand } => format!("({op:?} {})", shape(operand)),
        BExprKind::Binary { op, lhs, rhs } => {
            format!("({} {} {})", shape(lhs), op.symbol(), shape(rhs))
        }
        BExprKind::Ternary {
            condition,
            then,
            otherwise,
        } => format!(
            "({} ? {} : {})",
            shape(condition),
            shape(then),
            shape(otherwise)
        ),
        BExprKind::Call { name, arguments } => format!(
            "{name}[{}]",
            arguments.iter().map(shape).collect::<Vec<_>>().join(",")
        ),
        BExprKind::Group(inner) => shape(inner),
        BExprKind::Value(value) => format!("{{{}}}", value.text),
        BExprKind::Table(table) => format!("table({}, {})", shape(&table.input), table.domain),
    }
}

#[test]
fn b_cards_parse_with_c_precedence_and_setters() {
    let b = device("B1 Out GND v = 2*V(In)^2 - -3**2 + i(Vin) m=2 tc1=1m");
    assert_eq!((b.name.as_str(), b.designator), ("b1", 'b'));
    assert_eq!(b.nodes, ["out", "0"]);
    assert_eq!(names(&b), ["v", "m", "tc1"]);
    let v = expression(&b, "v");
    assert_eq!(v.text, "2*V(In)^2 - -3**2 + i(Vin)");
    assert!(!v.verbatim);
    assert_eq!(
        shape(&v.root),
        "(((2 * (v(in) ^ 2)) - (Minus (3 ^ 2))) + i(vin))"
    );
    // The expression starts after `=`; the setter keeps its own position.
    assert_eq!(v.span.start.column, 16);
    assert_eq!(b.parameters[1].location.column, 43);

    let i = device("b2 0 o i={ v(a,b) > 0 ? 1m : 0 }");
    assert_eq!(
        shape(&expression(&i, "i").root),
        "((v(a,b) > 0) ? 0.001 : 0)"
    );
    // Braces and quotes are whitespace for C (inp_modify_exp), not groups.
    let quoted = device("b3 o 0 v='2*{1+1}'");
    assert_eq!(shape(&expression(&quoted, "v").root), "((2 * 1) + 1)");
    // A `=pwl(` card keeps numparam values in braces.
    let pwl = device("b4 o 0 v=pwl(v(in), 0, 0, {k*2}, 1)");
    let pwl = expression(&pwl, "v");
    assert!(pwl.verbatim);
    assert_eq!(shape(&pwl.root), "pwl[v(in),0,0,{k*2},1]");
}

#[test]
fn malformed_b_cards_are_positioned_errors() {
    for (body, message, column) in [
        ("b1 o 0 v=2*(v(in)+1", "expected ')'", 20),
        ("b1 o 0 v=2*v(in) 3", "unexpected '3'", 18),
        ("b1 o 0 v=1 i=2", "exactly one v= or i=", 12),
        ("b1 o 0 x=1", "expected v=expression", 8),
        ("b1 o 0 v=1 m", "unexpected 'm'", 12),
        ("b1 o 0 v=", "empty behavioural expression", 10),
        ("b1 o 0 v=v (a)", "without a space", 10),
        ("b1 o 0 v=i(a,b)", "i() takes one source name", 13),
        ("b1 o 0 v=1 = 2", "unexpected '='", 12),
    ] {
        let error = try_parse(body).unwrap_err();
        match &error {
            SpiceError::Parse {
                location,
                message: text,
            } => {
                assert!(text.contains(message), "{body}: {text}");
                assert_eq!(location.column, column, "{body}: {text}");
            }
            other => panic!("{body}: {other}"),
        }
    }
}

#[test]
fn nonlinear_controlled_forms_parse_into_ordered_setters() {
    let e = device("e1 o 0 value = {v(a)*2} m=3");
    assert_eq!(names(&e), ["value", "m"]);
    assert_eq!(e.nodes, ["o", "0"]);
    assert_eq!(expression(&e, "value").text, "{v(a)*2}");
    let g = device("g1 0 o cur='v(a)' m=2");
    assert_eq!(names(&g), ["value", "m"]);
    let table = device("g2 0 o TABLE {v(a)*2} = (-1,-1m) (0,0) (2,{k}) m=2");
    assert_eq!(names(&table), ["table", "x", "y", "x", "y", "x", "y", "m"]);
    assert!(matches!(
        table.parameters[6].kind,
        ParameterKind::Expression(_)
    ));
    let ltspice = device("e2 o 0 a b table=(0, 0, 1, 2)");
    assert_eq!(ltspice.nodes, ["o", "0", "a", "b"]);
    assert_eq!(names(&ltspice), ["table", "x", "y", "x", "y"]);
    assert!(matches!(ltspice.parameters[0].kind, ParameterKind::Flag));
    let poly = device("e3 o 0 poly(2) a 0 (b, 0) 1 2 3");
    assert_eq!(poly.nodes, ["o", "0", "a", "0", "b", "0"]);
    assert_eq!(names(&poly), ["poly", "coef", "coef", "coef"]);
    let fpoly = device("f1 0 o poly(2) v1 v2 0 1 1 m=2");
    assert_eq!(
        names(&fpoly),
        ["poly", "control", "control", "coef", "coef", "coef", "m"]
    );
    // inp_poly_2g6_compat: values after the gain make an implicit POLY(1)
    // whose first coefficient is the "gain".
    let implicit = device("e4 o 0 a b 0.5 1 0.25");
    assert_eq!(names(&implicit), ["poly", "coef", "coef", "coef"]);
    assert_eq!(implicit.parameters[0].value, "1");
    let implicit = device("h1 o 0 vx 1 2");
    assert_eq!(names(&implicit), ["poly", "control", "coef", "coef"]);
}

#[test]
fn malformed_nonlinear_forms_are_explicit_errors() {
    for (body, message) in [
        ("e1 o 0 poly(1) a 0", "at least one coefficient"),
        ("e1 o 0 poly(0) a 0 1", "positive integer POLY dimension"),
        ("e1 o 0 poly(1.5) a 0 1", "positive integer POLY dimension"),
        ("e1 o 0 poly(65) a 0 1", "positive integer POLY dimension"),
        ("e1 o 0 poly(1) a 0 1 2 m=2", "remove it"),
        ("e1 o 0 table {v(a)} = (0,0) (1)", "pairs"),
        ("e1 o 0 table (0,0) (1,1)", "in braces"),
        (
            "e1 o 0 table {v(a)} = (0,0) (1,1) m=2",
            "not accepted on an E TABLE",
        ),
        ("g1 o 0 a b table=(0,0,1,1)", "four-node G TABLE"),
        ("f1 o 0 value={v(a)}", "not a form"),
        ("g1 o 0 vol={v(a)}", "not a form"),
        ("e1 o 0 value={v(a)} junk", "unexpected 'junk'"),
    ] {
        let error = try_parse(body).unwrap_err();
        assert!(error.to_string().contains(message), "{body}: {error}");
    }
    let error = try_parse("e1 o 0 laplace {v(a)} = {1/(1+s)}").unwrap_err();
    assert!(error.is_not_yet_ported(), "{error}");
}

fn round_trip(body: &str) -> String {
    let first = parse(body);
    let written = write_netlist(&first).unwrap();
    let second = Parser::new()
        .parse_deck(&parse_deck_text(Path::new("out.cir"), &written))
        .unwrap_or_else(|error| panic!("{error}\n{written}"));
    if let Some(difference) = semantic_diff(&first, &second) {
        panic!("not equivalent: {difference}\n{written}");
    }
    assert_eq!(write_netlist(&second).unwrap(), written, "fixed point");
    written
}

#[test]
fn the_writer_round_trips_every_behavioural_form() {
    let written = round_trip(
        "b1 o 0 v = 2*v(in)^2 + {k} m=2 temp=30\nb2 0 o i='i(v1)*v(a, b) > 0 ? 1 : 2'\n\
         b3 o 0 v=pwl(time, 0, 0, {t1}, 1)\ne1 o 0 vol={v(a)*2} m=2\n\
         g1 0 o value={1m*v(a)} m=3\ne2 o 0 table {v(a)} = (0,0) (1,{k})\n\
         g2 0 o table {2*v(a)} (0,0) (1,1) m=2\ne3 o 0 a b table=(0, 0, 1, 2)\n\
         e4 o 0 poly(2) a 0 b 0 1 2 3\nf1 0 o poly(1) v1 0 1 m=2\nh1 o 0 vx 1 2",
    );
    for line in [
        "b1 o 0 v=2*v(in)^2 + {k} m=2 temp=30",
        "b2 0 o i='i(v1)*v(a, b) > 0 ? 1 : 2'",
        "b3 o 0 v=pwl(time, 0, 0, {t1}, 1)",
        "e1 o 0 value={v(a)*2} m=2",
        "g1 0 o value={1m*v(a)} m=3",
        "e2 o 0 table {v(a)} = (0, 0) (1, {k})",
        "g2 0 o table {2*v(a)} = (0, 0) (1, 1) m=2",
        "e3 o 0 a b table=(0, 0, 1, 2)",
        "e4 o 0 poly(2) a 0 b 0 1 2 3",
        "f1 0 o poly(1) v1 0 1 m=2",
        "h1 o 0 poly(1) vx 1 2",
    ] {
        assert!(written.lines().any(|l| l == line), "{line}\n{written}");
    }
}

fn resolved(body: &str) -> Netlist {
    literalize(&parse(body)).unwrap().netlist
}

#[test]
fn resolution_substitutes_parameters_and_expands_functions() {
    let netlist = resolved(
        ".param k=3 vt=0.025\n.func soft(x) {tanh(x)*k}\n.func vd(a) {v(a)-v(b)*a}\n\
         b1 o 0 v=k*v(in) + soft(1+1) + vd(2) + 1.23456789012345 + pi + time",
    );
    let v = expression(&netlist.devices[0], "v");
    // Preserve numparam's decimal text rewrite and the B-source's eleven
    // significant digits, then read both correctly rounded. The .func
    // argument is parenthesised and v(a) in a body names node a literally.
    assert_eq!(
        shape(&v.root),
        "((((((3 * v(in)) + (tanh[(1 + 1)] * 3)) + (v(a) - \
         (v(b) * 2))) + 1.2345678901) + pi) + time)"
    );
}

#[test]
fn verbatim_lines_evaluate_numparam_values_and_keep_full_precision() {
    let netlist = resolved(".param k=2\nb1 o 0 v=pwl(v(in), 0, 0, {k*1.5}, 1.23456789012345)");
    let v = expression(&netlist.devices[0], "v");
    assert_eq!(shape(&v.root), "pwl[v(in),0,0,3,1.23456789012345]");
    // C leaves bare names unsubstituted on such a line.
    let error = literalize(&parse(".param k=2\nb1 o 0 v=pwl(v(in), 0, 0, k, 1)")).unwrap_err();
    assert!(error.to_string().contains("bare name 'k'"), "{error}");
}

#[test]
fn resolution_errors_name_their_cause() {
    for (body, message) in [
        ("b1 o 0 v=undefinedp*2", "undefined parameter [undefinedp]"),
        ("b1 o 0 v=sqr(2)", "no such function 'sqr'"),
        (".func f(x) {x}\nb1 o 0 v=f(1, 2)", "takes 1 argument"),
    ] {
        let error = literalize(&parse(body)).unwrap_err();
        assert!(error.to_string().contains(message), "{body}: {error}");
    }
    // A .func may redefine a built-in function, as in C's macro expansion.
    let netlist = resolved(".func sin(x) {x*2}\nb1 o 0 v=sin(v(a))");
    assert_eq!(
        shape(&expression(&netlist.devices[0], "v").root),
        "(v(a) * 2)"
    );
}

#[test]
fn statistical_functions_are_realized_before_compiling_the_behavioural_law() {
    for body in [
        "b1 o 0 v=agauss(1,0.1,3)*v(1)",
        "b1 o 0 v=2*gauss(1,0.1,3)",
        "b1 o 0 i=aunif(1,0.1)*1m",
        "e1 o 0 value={unif(1,0.1)*v(1)}",
        "g1 o 0 value={limit(1,0.1)*v(1)}",
    ] {
        let source = lower_nonlinear_sources(&parse(body)).unwrap();
        let netlist = literalize(&source).unwrap().netlist;
        for instance in &netlist.devices {
            for parameter in &instance.parameters {
                if let ParameterKind::Behavioural(expression) = &parameter.kind {
                    expression.root.visit(&mut |node| {
                        assert!(!matches!(&node.kind,BExprKind::Call{name,..} if ngspice_rs::netlist::bexpr::STATISTICAL_FUNCTIONS.contains(&name.as_str())));
                    });
                }
            }
        }
    }
    let error = literalize(&parse("b1 o 0 v=agauss(v(1),0.1,3)"))
        .unwrap_err()
        .to_string();
    assert!(error.contains("static arguments"), "{error}");
    // A user .func of the same name is expanded first, as in C.
    let netlist = resolved(".func agauss(a,b,c) {a*2}\nb1 o 0 v=agauss(1,0.1,3)*v(x)");
    assert_eq!(
        shape(&expression(&netlist.devices[0], "v").root),
        "((1 * 2) * v(x))"
    );
}

#[test]
fn resolution_against_an_explicit_scope() {
    let netlist = parse(".param a=4\nb1 o 0 v=a*v(x)");
    let scope = ParamScope::for_netlist(&netlist).unwrap();
    let mut budget = EvalBudget::default();
    let v = expression(&netlist.devices[0], "v");
    let resolved =
        ngspice_rs::netlist::behavioural::resolve_expression(&v, &Arc::new(scope), &mut budget)
            .unwrap();
    assert_eq!(shape(&resolved.root), "(4 * v(x))");
}

fn lowered(body: &str) -> Netlist {
    lower_nonlinear_sources(&parse(body)).unwrap()
}

fn summary(netlist: &Netlist) -> Vec<String> {
    netlist
        .devices
        .iter()
        .map(|d| {
            format!(
                "{} {} [{}]",
                d.name,
                d.nodes.join(" "),
                d.parameters
                    .iter()
                    .map(|p| format!("{}={}", p.name, p.value))
                    .collect::<Vec<_>>()
                    .join(" ")
            )
        })
        .collect()
}

#[test]
fn value_forms_lower_onto_a_b_source_and_an_internal_node() {
    let netlist = lowered("e1 o 0 value={v(a)*2} m=3\ng1 0 p cur={v(a)} m=2");
    assert_eq!(
        summary(&netlist),
        [
            "e1 o 0 e1_int1 0 [gain=1]",
            "be1 e1_int1 0 [v={v(a)*2} m=3]",
            "g1 0 p g1_int1 0 [gain=2]",
            "bg1 g1_int1 0 [v={v(a)}]",
        ]
    );
}

#[test]
fn table_forms_lower_onto_the_xspice_pwl_transfer() {
    let netlist = lowered(
        "e1 o 0 table {v(a)} = (0,0) (1,2)\ng[1] 0 p table {v(a)} = (0, 3m) m=2\n\
         e2 q 0 a b table=(0, 0, 1, 1)",
    );
    let lines = summary(&netlist);
    assert_eq!(lines[0], "e1 o 0 e1_int1 0 [gain=1]");
    assert_eq!(lines[1], "be1 e1_int2 0 [v=v(a)]");
    assert!(lines[2].starts_with("ae1 e1_int1 0 [v=xspice-pwl input_domain=0.1"));
    // One pair is a constant source; G names lose `[`/`]` (inp_compat).
    assert_eq!(lines[3], "g[1] 0 p g_1__int1 0 [gain=2]");
    assert_eq!(lines[4], "vg[1] g_1__int1 0 [dc=3m]");
    assert_eq!(lines[5], "e2 q 0 e2_int1 0 [gain=1]");
    assert!(lines[6].starts_with("ae2 e2_int1 0 [v=xspice-pwl input_domain=0.001"));
    let ParameterKind::Behavioural(transfer) = &netlist.devices[6].parameters[0].kind else {
        panic!("not behavioural");
    };
    assert_eq!(shape(&transfer.root), "table(v(a,b), 0.001)");
}

#[test]
fn poly_forms_lower_onto_spice2poly_in_spice2_term_order() {
    let netlist = lowered("e1 o 0 poly(2) a 0 b 0 1 2 3 4 5 6\nf1 0 p poly(1) vx 0 2 m=3");
    assert_eq!(netlist.devices[0].name, "e1");
    assert_eq!(netlist.devices[0].designator, 'e');
    assert_eq!(netlist.devices[0].nodes, ["o", "0"]);
    let v = expression(&netlist.devices[0], "v");
    assert_eq!(
        shape(&v.root),
        "(((((1 + (2 * v(a,0))) + (3 * v(b,0))) + (4 * (v(a,0) * v(a,0)))) + \
         (5 * (v(a,0) * v(b,0)))) + (6 * (v(b,0) * v(b,0))))"
    );
    assert_eq!(names(&netlist.devices[1]), ["i", "m"]);
    assert_eq!(
        shape(&expression(&netlist.devices[1], "i").root),
        "(0 + (2 * i(vx)))"
    );
}

#[test]
fn poly_dimensions_use_one_validation_rule_across_frontend() {
    for spelling in ["2", "2.0", "2e0", "02", "+2"] {
        let body = format!("e1 o 0 poly({spelling}) a 0 b 0 1 2 3");
        let written = round_trip(&body);
        assert!(written.contains(&body), "{written}");
        let netlist = lowered(&body);
        assert_eq!(
            shape(&expression(&netlist.devices[0], "v").root),
            "((1 + (2 * v(a,0))) + (3 * v(b,0)))"
        );
    }
    // atoi in the C translator reads a decimal prefix, not a scaled SPICE
    // number; the port must not accept an inconsistent dimension.
    for spelling in ["2e1", "0.2e1", "1.5", "65", "0"] {
        let body = format!("e1 o 0 poly({spelling}) a 0 b 0 1 2 3");
        let error = try_parse(&body).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("positive integer POLY dimension"),
            "{error}"
        );
    }
    let mut malformed = parse("e1 o 0 poly(2) a 0 b 0 1 2 3");
    malformed.devices[0].parameters[0].value = "18446744073709551615".into();
    assert!(lower_nonlinear_sources(&malformed).is_err());
    assert!(write_netlist(&malformed).is_err());
}

#[test]
fn nonlinear_lowering_preserves_unresolved_probe_names_and_scope_boundaries() {
    let result = lowered(
        "r1 a 0 1k\nb1 out 0 v=i(r1)+i(r1)\n.subckt child a\nr1 a 0 2k\nb1 out 0 v=i(r1)\n.ends\nr2 b 0 2k\nb2 p 0 v=i(r2)+i(r1)+i(r2)",
    );
    assert_eq!(result.devices.len(), 4);
    assert_eq!(result.subcircuits[0].devices.len(), 2);
    assert_eq!(result.devices[0].nodes[0], "a");
    assert_eq!(
        expression(&result.devices[3], "v")
            .root
            .current_references(),
        ["r2", "r1", "r2"]
    );
    assert_eq!(
        expression(&result.subcircuits[0].devices[1], "v")
            .root
            .current_references(),
        ["r1"]
    );
}

#[test]
fn malformed_nonlinear_shapes_are_rejected_by_both_writer_and_lowering() {
    let table = parse("g1 out 0 table {v(a)} = (0,1) (2,3) m=4");
    let mut malformed = Vec::new();

    let mut duplicate_x = table.clone();
    let extra = duplicate_x.devices[0].parameters[1].clone();
    duplicate_x.devices[0].parameters.insert(2, extra);
    malformed.push(duplicate_x);

    let mut duplicate_input = table.clone();
    let extra = duplicate_input.devices[0].parameters[0].clone();
    duplicate_input.devices[0].parameters.insert(1, extra);
    malformed.push(duplicate_input);

    let mut misplaced_point = table.clone();
    misplaced_point.devices[0].parameters.swap(4, 5);
    malformed.push(misplaced_point);

    let mut bad_singleton = parse("e1 out 0 table {v(a)} = (0,3)");
    bad_singleton.devices[0].parameters[0].kind = ParameterKind::Flag;
    malformed.push(bad_singleton);

    let mut bad_y = parse("e1 out 0 table {v(a)} = (0,3)");
    bad_y.devices[0].parameters[2].kind = ParameterKind::Instance;
    malformed.push(bad_y);

    let mut bad_poly = parse("f1 out 0 poly(2) va vb 1 2 3 m=4");
    bad_poly.devices[0].parameters[1].kind = ParameterKind::Scalar;
    malformed.push(bad_poly);

    let mut bad_dimension = parse("e1 out 0 poly(1) in 0 1 2");
    bad_dimension.devices[0].parameters[0].kind = ParameterKind::Flag;
    malformed.push(bad_dimension);

    let mut duplicate_form = parse("e1 out 0 value={v(a)} m=2");
    let extra = duplicate_form.devices[0].parameters[0].clone();
    duplicate_form.devices[0].parameters.push(extra);
    malformed.push(duplicate_form);

    let mut misplaced_form = parse("e1 out 0 value={v(a)} m=2");
    misplaced_form.devices[0].parameters.swap(0, 1);
    malformed.push(misplaced_form);

    let mut bad_value = parse("e1 out 0 value={v(a)}");
    bad_value.devices[0].parameters[0].kind = ParameterKind::Scalar;
    malformed.push(bad_value);

    let mut four_node_g = parse("e1 out 0 in 0 table=(0,1,2,3)");
    four_node_g.devices[0].name = "g1".into();
    four_node_g.devices[0].designator = 'g';
    malformed.push(four_node_g);

    for deck in malformed {
        let err = lower_nonlinear_sources(&deck).expect_err("accepted malformed nonlinear source");
        assert!(matches!(err, SpiceError::Parse { .. }), "{err}");
        assert!(write_netlist(&deck).is_err());
    }
}

#[test]
fn duplicate_targets_are_rejected_at_physical_elaboration() {
    let input = parse("r1 a 0 1k\nr1 b 0 2k\nb1 out 0 v=i(r1)");
    let error = ngspice_rs::devices::Circuit::from_netlist(&input).unwrap_err();
    assert!(
        error.to_string().contains("duplicate instance name"),
        "{error}"
    );
}

#[test]
fn behavioural_expression_and_setters_use_separate_physical_and_logical_offsets() {
    let device = device("  B1 out 0 v=2*\n* comment\n  + v(in) m=3");
    let expr = expression(&device, "v");
    assert_eq!(expr.text, "2* v(in)");
    assert_eq!((expr.span.start.line, expr.span.start.column), (2, 14));
    assert_eq!((expr.span.end.line, expr.span.end.column), (4, 10));
    assert_eq!(expr.span.len(), expr.text.len());
    let setter = &device.parameters[1].location;
    assert_eq!((setter.line, setter.column), (4, 11));
}

#[test]
fn folded_current_references_survive_controlled_source_lowering() {
    let netlist =
        lowered("r1 a 0 1k\nr2 b 0 1k\ne1 out 0 value={max(1,\n* comment\n + i(r1)) +\n + i(r2)}");
    let source = netlist.devices.iter().find(|d| d.name == "be1").unwrap();
    assert_eq!(
        expression(source, "v").root.current_references(),
        ["r1", "r2"]
    );
}

#[test]
fn behavioural_flat_chains_are_bounded_before_recursive_consumers() {
    use ngspice_rs::netlist::expr::MAX_TREE_DEPTH;
    std::thread::Builder::new()
        .stack_size(2 * 1024 * 1024)
        .spawn(|| {
            for op in ["+", "-", "*", "/", "^", "**", "||", "&&", "==", "<"] {
                let accepted = format!("{}1", format!("1{op}").repeat(MAX_TREE_DEPTH - 1));
                let n = parse(&format!("b1 o 0 v={accepted}"));
                assert_eq!(n, n.clone());
                assert!(!shape(&expression(&n.devices[0], "v").root).is_empty());
                let written = write_netlist(&n).unwrap();
                let reparsed = Parser::new()
                    .parse_deck(&parse_deck_text(Path::new("in.cir"), &written))
                    .unwrap();
                assert!(semantic_diff(&n, &reparsed).is_none());
                assert!(literalize(&n).is_ok());
                assert!(
                    ngspice_rs::devices::Circuit::from_netlist(&n).is_ok(),
                    "{op}"
                );
                let excessive = format!("{}1", format!("1{op}").repeat(20_000));
                let error = try_parse(&format!("b1 o 0 v={excessive}")).unwrap_err();
                assert!(
                    error.to_string().contains("structural depth limit"),
                    "{error}"
                );
            }
        })
        .unwrap()
        .join()
        .unwrap();
}

#[test]
fn generated_and_embedded_expression_nodes_obey_the_same_limits() {
    use ngspice_rs::netlist::expr::{MAX_TREE_DEPTH, MAX_TREE_NODES};
    let chain = format!("{}1", "1+".repeat(MAX_TREE_DEPTH - 1));
    for body in [
        format!("b1 o 0 v=pwl(v(in),0,0,{{{chain}}},1)"),
        format!("b1 o 0 v=1?{chain}:0"),
        format!("b1 o 0 v=abs({chain})"),
    ] {
        let error = try_parse(&body).unwrap_err();
        assert!(
            error.to_string().contains("structural depth limit"),
            "{error}"
        );
    }
    let wide = format!("b1 o 0 v=f({}1)", "1,".repeat(MAX_TREE_NODES - 1));
    let error = try_parse(&wide).unwrap_err();
    assert!(error.to_string().contains("node limit"), "{error}");
    let embedded = format!(
        "b1 o 0 v=pwl(v(in),0,0,{}1)",
        "{1+1},".repeat(MAX_TREE_NODES / 4)
    );
    let error = try_parse(&embedded).unwrap_err();
    assert!(error.to_string().contains("node limit"), "{error}");
    let n = parse(&format!(
        "e1 o 0 poly(1) in 0 {}",
        "1 ".repeat(MAX_TREE_DEPTH)
    ));
    let error = lower_nonlinear_sources(&n).unwrap_err();
    assert!(
        error.to_string().contains("structural depth limit"),
        "{error}"
    );
}

#[test]
fn macro_substitution_checks_height_and_amplification_before_cloning() {
    use ngspice_rs::netlist::expr::MAX_TREE_DEPTH;
    let chain = format!("{}1", "1+".repeat(MAX_TREE_DEPTH - 2));
    let n = parse(&format!(".func f(x) {{x+1}}\nb1 o 0 v=f({chain})"));
    let error = literalize(&n).unwrap_err();
    assert!(
        error.to_string().contains("structural depth limit"),
        "{error}"
    );
    // 240 occurrences of a 511-node argument would exceed 100,000 nodes.
    let body = format!("{}x", "x+".repeat(239));
    let arg = format!("pwl(v(in),0,0,{}1)", "1,".repeat(510));
    let n = parse(&format!(".func f(x) {{{body}}}\nb1 o 0 v=f({arg})"));
    let error = literalize(&n).unwrap_err();
    assert!(error.to_string().contains("node limit"), "{error}");
}

#[test]
fn wide_valid_pwl_constructs_resolves_and_compiles_on_bounded_stack() {
    use std::fmt::Write as _;
    std::thread::Builder::new()
        .stack_size(2 * 1024 * 1024)
        .spawn(|| {
            let points = 8192;
            let mut body = "v1 in 0 1\nb1 o 0 v=pwl(v(in)".to_owned();
            for i in 0..points {
                write!(body, ",{i},{i}").unwrap();
            }
            body.push(')');
            let start = std::time::Instant::now();
            let n = parse(&body);
            let parsed = start.elapsed();
            assert_eq!(n, n.clone());
            let written = write_netlist(&n).unwrap();
            let reparsed = Parser::new()
                .parse_deck(&parse_deck_text(Path::new("in.cir"), &written))
                .unwrap();
            assert!(semantic_diff(&n, &reparsed).is_none());
            assert!(literalize(&n).is_ok());
            assert!(ngspice_rs::devices::Circuit::from_netlist(&n).is_ok());
            eprintln!(
                "PWL {points} points: parsed {parsed:?}, full pipeline {:?}",
                start.elapsed()
            );
        })
        .unwrap()
        .join()
        .unwrap();
}

#[test]
fn symbolic_depth_boundary_and_nested_syntax_compile_on_bounded_stack() {
    use ngspice_rs::netlist::expr::{MAX_NESTING, MAX_TREE_DEPTH};
    std::thread::Builder::new()
        .stack_size(2 * 1024 * 1024)
        .spawn(|| {
            let symbolic = format!("{}v(in)", "v(in)+".repeat(MAX_TREE_DEPTH - 1));
            let nested = [
                symbolic,
                format!("{}1{}", "(".repeat(MAX_NESTING), ")".repeat(MAX_NESTING)),
                format!("{}1", "1?1:".repeat(MAX_NESTING)),
            ];
            for text in nested {
                let n = parse(&format!("v1 in 0 1\nb1 o 0 v={text}"));
                assert!(ngspice_rs::devices::Circuit::from_netlist(&n).is_ok());
            }
            for text in [
                format!(
                    "{}1{}",
                    "(".repeat(MAX_NESTING + 1),
                    ")".repeat(MAX_NESTING + 1)
                ),
                format!("{}1", "1?1:".repeat(MAX_NESTING + 1)),
            ] {
                let error = try_parse(&format!("b1 o 0 v={text}")).unwrap_err();
                assert!(error.to_string().contains("nesting limit"), "{error}");
            }
        })
        .unwrap()
        .join()
        .unwrap();
}
