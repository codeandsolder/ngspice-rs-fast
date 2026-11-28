//! Behavioural sources (GitHub #79): the compiled expression's values and
//! analytic derivatives for every `inpptree.c` function (finite-difference
//! checks), C's value quirks, the `ASRCload` stamps and the consistency of
//! the Newton residual with its Jacobian, and explicit errors.
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use ngspice_rs::devices::behavioural::program::{Environment, Program, Quantity};
use ngspice_rs::devices::{AnalysisMode, Circuit, LoadRequest, ModelContext, Registry};
use ngspice_rs::maths::{SparseMatrix, Vector};
use ngspice_rs::netlist::ast::Netlist;
use ngspice_rs::netlist::source::parse_deck_text;
use ngspice_rs::netlist::{Parser, RawCard};
use ngspice_rs::primitives::{Real, SourceLoc, SpiceError};

fn origin() -> SourceLoc {
    SourceLoc::new(PathBuf::from("expr.cir"), 1, 1)
}

fn program(text: &str) -> Program {
    let parsed = Parser::new()
        .parse_behavioural_expression(text, &origin(), false)
        .unwrap_or_else(|error| panic!("{text}: {error}"));
    Program::compile(&parsed.root).unwrap_or_else(|error| panic!("{text}: {error}"))
}

fn values(program: &Program, point: &BTreeMap<&str, Real>) -> Vec<Real> {
    program
        .quantities()
        .iter()
        .map(|quantity| match quantity {
            Quantity::Node(name) | Quantity::Branch(name) => point[name.as_str()],
        })
        .collect()
}

fn environment(values: &[Real]) -> Environment<'_> {
    Environment {
        values,
        time: 1.25e-3,
        temperature: 27.,
        gmin: 1e-12,
        frequency: 50.,
    }
}

fn value(text: &str, point: &[(&str, Real)]) -> Real {
    let program = program(text);
    let point: BTreeMap<&str, Real> = point.iter().copied().collect();
    let values = values(&program, &point);
    program
        .evaluate(&environment(&values))
        .unwrap_or_else(|error| panic!("{text}: {error}"))
        .value
}

/// Central differences against the analytic gradient at every point.
fn assert_gradient(text: &str, points: &[&[(&str, Real)]]) {
    let program = program(text);
    for point in points {
        let point: BTreeMap<&str, Real> = point.iter().copied().collect();
        let x = values(&program, &point);
        let analytic = program
            .evaluate(&environment(&x))
            .unwrap_or_else(|error| panic!("{text} at {point:?}: {error}"));
        for (index, slope) in analytic.gradient.iter().enumerate() {
            let h = 1e-6 * x[index].abs().max(1.);
            let mut up = x.clone();
            up[index] += h;
            let mut down = x.clone();
            down[index] -= h;
            let f = |values: &[Real]| program.evaluate(&environment(values)).unwrap().value;
            let numeric = (f(&up) - f(&down)) / (2. * h);
            assert!(
                (numeric - slope).abs() <= 2e-6 * slope.abs().max(1.),
                "{text} at {point:?}, d/d{:?}: analytic {slope}, numeric {numeric}",
                program.quantities()[index]
            );
        }
    }
}

const POINTS: &[&[(&str, Real)]] = &[
    &[("a", 0.3), ("b", -0.7), ("vx", 2e-3)],
    &[("a", -0.45), ("b", 0.2), ("vx", -1.5e-3)],
    &[("a", 0.8), ("b", 0.55), ("vx", 0.5e-3)],
];

#[test]
fn quantity_interning_preserves_first_use_and_namespaces() {
    let compiled = program("v(x,y)+i(x)+2*v(y)+3*i(x)+v(x)");
    assert_eq!(
        compiled.quantities(),
        &[
            Quantity::Node("x".into()),
            Quantity::Node("y".into()),
            Quantity::Branch("x".into())
        ],
    );
    let result = compiled.evaluate(&environment(&[2., 3., 5.])).unwrap();
    assert_eq!(result.value, 27.);
    assert_eq!(result.gradient, [2., 1., 4.]);
    assert!(!compiled.uses_frequency());

    for text in [
        "hertz",
        "v(x)+hertz",
        "pwl(hertz,0,0,1,1)",
        "ternary_fcn(0,hertz,v(x))",
        "max(v(x),sin(hertz))",
    ] {
        assert!(program(text).uses_frequency(), "{text}");
    }
    assert!(!program("v(hertz)+i(hertz)").uses_frequency());
}

#[test]
fn every_function_has_a_derivative_matching_finite_differences() {
    for text in [
        // inpptree.c funcs[] in table order, each with an inner expression so
        // the chain rule is exercised.
        "abs(v(a) - 1.5*v(b))",
        "acos(0.9*v(a)*v(b))",
        "acosh(2 + v(a)^2 + v(b))",
        "asin(0.5*v(a) - 0.2*v(b))",
        "asinh(3*v(a)*v(b))",
        "atan(2*v(a) + v(b))",
        "atanh(0.6*v(a) + 0.3*v(b))",
        "cos(4*v(a) - v(b))",
        "cosh(v(a) + 2*v(b))",
        "exp(2*v(a) - v(b))",
        "ln(2 + v(a) + v(b))",
        "log(2 + v(a) + v(b))",
        "log10(2 + v(a)*v(b))",
        "sgn(v(a) - 5)*v(b)",
        "sin(3*v(a)*v(b))",
        "sinh(v(a) - v(b))",
        "sqrt(3 + v(a) + v(b))",
        "tan(v(a) + 0.5*v(b))",
        "tanh(2*v(a) - v(b))",
        "u(v(a) - 5) + u(v(a) + 5)*v(b)",
        "uramp(v(a) + 2*v(b) + 3)",
        "ceil(v(a)*7 + 0.5) * v(b)",
        "floor(v(a)*7 + 0.5) * v(b)",
        "nint(v(a)*7 + 0.25) * v(b)",
        "-v(a)*v(b)",
        "u2(0.3*v(a) + 0.5) + u2(v(b) + 5) + u2(v(b) - 5)",
        "pwl(v(a), -1, 2, 0, 0, 0.5, 1, 2, 3) * v(b)",
        "eq0(v(a)) + ne0(v(a)) + gt0(v(a)) + lt0(v(a)) + ge0(v(a)) + le0(v(a))",
        "pow(v(a) + 2, v(b) + 1)",
        "pow(v(a) + 2, 3)",
        "pow(2, v(b)*v(a))",
        "pow(v(a) - 3, 2)",
        "pwr(v(a) + 3, 2.5)",
        "pwr(v(a) + 2, v(b))",
        "min(v(a), v(b)) + max(2*v(a), v(b) + 1)",
        "ternary_fcn(v(a) > 0, v(a)^2, v(b)^3)",
        // Operators.
        "v(a) + v(b) - 2*v(a)*v(b)",
        "v(a) / (2 + v(b))",
        "(1 + v(a)) / v(b)",
        "(v(a) + 3)^(v(b) + 1)",
        "(v(a) + 3)^2.5",
        "(v(a) - 3)^2",
        "2^(v(a) - v(b))",
        "(v(a) + 2) ** 3",
        "v(a, b)^2 + v(b, a)",
        "v(a) > v(b) ? exp(v(a)) : cos(v(b))",
        "(v(a) < 0.5 && v(b) != 0) + (v(a) >= 0 || v(b) <= 0) + !(v(a) == v(b))",
        "i(vx)*1k + v(a)*i(vx)",
        "time*v(a) + temper*v(b) + hertz*v(a)*v(b)",
        "pi*v(a) + e*v(b)",
    ] {
        assert_gradient(text, POINTS);
    }
}

/// The `^` operator and `pow()` use the same branch-local partials, even
/// though their value functions diverge at a zero base. This covers constant
/// exponents, variable exponents and the special constant-base rule.
#[test]
fn caret_and_pow_share_identical_derivative_values() {
    let cases = [
        ("(v(a) + 3)^2.5", "pow(v(a) + 3, 2.5)"),
        ("(v(a) - 3)^3", "pow(v(a) - 3, 3)"),
        ("(v(a) + 3)^(v(b) + 1)", "pow(v(a) + 3, v(b) + 1)"),
        ("2^(v(a) - v(b))", "pow(2, v(a) - v(b))"),
    ];
    for (caret, function) in cases {
        let left = program(caret);
        let right = program(function);
        assert_eq!(left.quantities(), right.quantities());
        for point in POINTS {
            let names: BTreeMap<&str, Real> = point.iter().copied().collect();
            let x = values(&left, &names);
            let lhs = left.evaluate(&environment(&x)).unwrap();
            let rhs = right.evaluate(&environment(&x)).unwrap();
            assert_eq!(lhs.gradient.len(), rhs.gradient.len(), "{caret}");
            for (a, b) in lhs.gradient.iter().zip(&rhs.gradient) {
                assert_eq!(
                    a.to_bits(),
                    b.to_bits(),
                    "{caret} vs {function} at {point:?}"
                );
            }
        }
    }
}

#[test]
fn values_follow_the_c_functions() {
    let at = |text: &str| value(text, &[("a", 0.)]);
    // PTdivide moves the divisor by gmin * 1e-20 away from zero.
    assert_eq!(at("1/v(a)"), 1e32);
    assert_eq!(at("-1/v(a)"), -1e32);
    // PTlog/PTlog10: log(0) is -1e99; PTexp caps above 227.9559242.
    assert_eq!(at("log(v(a))"), -1e99);
    assert_eq!(at("log10(v(a))"), -1e99);
    assert_eq!(at("exp(300 + v(a))"), 1e99);
    // PTustep, PTuramp, PTustep2, nearbyint.
    assert_eq!(at("u(v(a))"), 0.5);
    assert_eq!(at("uramp(v(a) - 1)"), 0.);
    assert_eq!(at("u2(v(a) + 0.25)"), 0.25);
    assert_eq!(at("nint(2.5 + v(a)) + nint(3.5)"), 6.);
    // PTpowerH: a negative base needs a quasi-integer exponent; pow(0, b)=0.
    assert_eq!(at("(v(a) - 2)^2.5"), 0.);
    assert_eq!(at("(v(a) - 2)^3"), -8.);
    assert_eq!(at("v(a)^0"), 1.);
    assert_eq!(at("pow(v(a), 0)"), 0.);
    assert_eq!(at("pwr(v(a) - 2, 2)"), -4.);
    // Comparisons, logic and the ternary select by != 0.
    assert_eq!(
        at("(v(a) < 1) + (v(a) <> 0) + !v(a) + (1 && v(a)) + (0 || 2)"),
        3.
    );
    assert_eq!(at("v(a) ? 1 : 2"), 2.);
    // min/max/pwl.
    assert_eq!(at("min(v(a), -1) + max(v(a), 3)"), 2.);
    assert_eq!(at("pwl(v(a) - 2, 0, 0, 1, 1, 3, 5)"), -2.);
    assert_eq!(at("pwl(v(a) + 4, 0, 0, 1, 1, 3, 5)"), 7.);
    // Constants, time and temper.
    assert_eq!(
        at("pi + e + v(a)"),
        std::f64::consts::PI + std::f64::consts::E
    );
    assert_eq!(at("time*1000 + temper + v(a)"), 1.25 + 27.);
    assert_eq!(at("hertz + v(a)"), 50.);
    // sin/cos/tan reduce their argument like C's MODULUS before evaluating.
    let x: Real = 100.;
    let reduced =
        x - Real::from((x / (2. * std::f64::consts::PI)) as i32) * 2. * std::f64::consts::PI;
    assert_eq!(at("sin(100 + v(a))"), reduced.sin());
}

#[test]
fn pwl_value_and_slope_select_the_same_segment_in_either_order() {
    for text in [
        "pwl(v(a), 2, 3, 0.5, 1, 0, 0, -1, 2)",
        "pwl(v(a), -1, 2, 0, 0, 0.5, 1, 2, 3)",
    ] {
        let p = program(text);
        for (x, expected) in [
            (-2., -2.),
            (-0.3, -2.),
            (0.3, 2.),
            (0.8, 4. / 3.),
            (3., 4. / 3.),
        ] {
            let evaluation = p.evaluate(&environment(&[x])).unwrap();
            assert!(
                (evaluation.gradient[0] - expected).abs() < 1e-15,
                "{text} at {x}"
            );
            let h = 1e-6;
            let upper = p.evaluate(&environment(&[x + h])).unwrap().value;
            let lower = p.evaluate(&environment(&[x - h])).unwrap().value;
            assert!(
                ((upper - lower) / (2. * h) - expected).abs() < 1e-9,
                "{text} at {x}"
            );
        }
    }
}

#[test]
fn negative_signed_power_has_its_physical_nonzero_slope() {
    let program = program("pwr(v(a) - 3, 2.5)");
    let evaluation = program.evaluate(&environment(&[-0.3])).unwrap();
    assert!((evaluation.value + 3.3_f64.powf(2.5)).abs() < 1e-12);
    assert!((evaluation.gradient[0] - 2.5 * 3.3_f64.powf(1.5)).abs() < 1e-12);
}

#[test]
fn power_partials_follow_actual_value_branches() {
    for (expression, expected) in [
        ("v(a)**3", 0.6075),
        ("pow(v(a),3)", 0.6075),
        ("v(a)^2", -0.9),
        ("pow(v(a),2.5)", 0.),
        ("pwr(v(a),2)", 0.9),
        ("pwr(v(a),2.5)", 2.5 * 0.45_f64.powf(1.5)),
        ("pow(v(a),v(b))", 0.6075),
        ("pwr(v(a),v(b))", 0.6075),
    ] {
        let p = program(expression);
        let point = BTreeMap::from([("a", -0.45), ("b", 3.)]);
        let x = values(&p, &point);
        let evaluation = p.evaluate(&environment(&x)).unwrap();
        assert!(
            (evaluation.gradient[0] - expected).abs() < 1e-14,
            "{expression}: {:?}",
            evaluation.gradient
        );
        if evaluation.gradient.len() == 2 {
            let expected = if expression.starts_with("pwr") {
                evaluation.value * 0.45_f64.ln()
            } else {
                0.
            };
            assert!(
                (evaluation.gradient[1] - expected).abs() < 1e-14,
                "{expression}"
            );
        }
    }
}

#[test]
fn domain_errors_are_explicit_numerical_errors() {
    for text in [
        "sqrt(v(a) - 1)",
        "log(v(a) - 1)",
        "log10(v(a) - 1)",
        "acos(v(a) + 2)",
        "acosh(v(a))",
        "atanh(v(a) + 1)",
        "cosh(v(a) + 1000)",
    ] {
        let program = program(text);
        let values = vec![0.; program.quantities().len()];
        let error = program.evaluate(&environment(&values)).unwrap_err();
        assert!(
            matches!(error, SpiceError::Numerical { .. }),
            "{text}: {error}"
        );
    }
}

#[test]
fn unported_and_unknown_functions_are_rejected_at_compile_time() {
    let compile = |text: &str| {
        let parsed = Parser::new()
            .parse_behavioural_expression(text, &origin(), false)
            .unwrap();
        Program::compile(&parsed.root)
    };
    for text in [
        "ddt(v(a))",
        "gauss(1, 0.1, 3)",
        "agauss(1, 0.1, 3)",
        "aunif(1, 0.1)",
        "unif(1, 0.1)",
        "limit(1, 0.1)",
    ] {
        let error = compile(text).unwrap_err();
        assert!(error.is_not_yet_ported(), "{text}: {error}");
    }
    for (text, message) in [
        ("sqr(v(a))", "no such function 'sqr'"),
        ("sin(v(a), 2)", "takes 1 argument"),
        ("pow(v(a))", "takes 2 argument"),
        ("pwl(v(a), 0, 0, 1)", "even number"),
        ("pwl(v(a), 0, 0, 0, 1)", "monotonic"),
        ("pwl(v(a), 0, 0, 1, 1, 0.5, 2)", "monotonic"),
        ("pwl(v(a), 0, v(b), 1, 1)", "literal"),
        ("pwl_derivative(v(a))", "internal"),
        ("k*v(a)", "unresolved name 'k'"),
    ] {
        let error = compile(text).unwrap_err();
        assert!(error.to_string().contains(message), "{text}: {error}");
    }
}

fn deck(body: &str) -> Netlist {
    Parser::new()
        .parse_deck(&parse_deck_text(
            Path::new("behavioural.cir"),
            &format!("behavioural sources\n{body}\n.end\n"),
        ))
        .unwrap()
}

fn circuit(body: &str) -> ngspice_rs::primitives::SpiceResult<Circuit> {
    Circuit::from_netlist(&deck(body))
}

fn row(circuit: &Circuit, node: &str) -> usize {
    circuit
        .unknowns()
        .node_row(circuit.nodes().get(node).expect("node exists"))
        .expect("not ground")
}

fn branch(circuit: &Circuit, name: &str) -> usize {
    let index = circuit
        .devices()
        .iter()
        .position(|device| device.name() == name)
        .expect("device exists");
    circuit.branch_rows(index).unwrap().start
}

/// One operating-point load of `circuit` at `x`.
fn load(circuit: &Circuit, x: &Vector, context: &ModelContext) -> (SparseMatrix, Vector) {
    let n = circuit.unknown_count();
    let history = circuit.state_history().unwrap();
    let mut trial = history.trial();
    let mut a = SparseMatrix::new(n, n);
    let mut b = Vector::zeros(n);
    circuit
        .load(
            &LoadRequest {
                mode: AnalysisMode::OperatingPoint,
                solution: x,
                model_context: context,
                integration: None,
                history: &history,
                forcing: None,
            },
            &mut a,
            &mut b,
            &mut trial,
        )
        .unwrap();
    a.fold_duplicates();
    (a, b)
}

fn residual(circuit: &Circuit, x: &Vector) -> Vec<Real> {
    let (a, b) = load(circuit, x, &ModelContext::default());
    let mut r: Vec<Real> = b.as_slice().iter().map(|v| -v).collect();
    for t in a.triplets() {
        r[t.row] += t.value * x.as_slice()[t.col];
    }
    r
}

#[test]
fn newton_residuals_are_consistent_with_their_jacobians() {
    // The linearised load A(x0) x0 - b(x0) is the physical residual, and its
    // finite-difference Jacobian is A(x0) for every row a B source touches.
    let c = circuit(
        "vin in 0 dc 1\nrin in 0 1k\nvx x 0 dc 0.5\nrx x 0 2k\n\
         b1 o1 0 v=exp(v(in,x))*i(vx) + sqrt(1 + v(o2)^2) m=2 tc1=1m temp=40\n\
         r1 o1 0 1k\n\
         b2 o2 o3 i=1m*tanh(v(o1)/2) + 2m*pwl(v(in), 0,0, 1,2, 3,1) reciprocm=1 m=4\n\
         r2 o2 0 1k\nr3 o3 0 3k\n\
         e1 o4 0 value={v(o1)*v(o3) + i(b1)*100}\nr4 o4 0 1k",
    )
    .unwrap();
    let n = c.unknown_count();
    let mut x = Vector::zeros(n);
    for (row, value) in x.as_mut_slice().iter_mut().enumerate() {
        *value = 0.3 + 0.17 * row as Real - 0.05 * (row % 3) as Real;
    }
    let (a, _) = load(&c, &x, &ModelContext::default());
    let dense = |row: usize, col: usize| a.get(row, col);
    for col in 0..n {
        let h = 1e-6 * x.as_slice()[col].abs().max(1.);
        let mut up = x.clone();
        up.as_mut_slice()[col] += h;
        let mut down = x.clone();
        down.as_mut_slice()[col] -= h;
        let (ru, rd) = (residual(&c, &up), residual(&c, &down));
        for row in 0..n {
            let numeric = (ru[row] - rd[row]) / (2. * h);
            let analytic = dense(row, col);
            assert!(
                (numeric - analytic).abs() <= 1e-6 * analytic.abs().max(1.),
                "A[{row}][{col}]: analytic {analytic}, numeric {numeric}"
            );
        }
    }
}

#[test]
fn voltage_and_current_outputs_stamp_like_asrcload() {
    let c = circuit(
        "vin in 0 dc 1\nrin in 0 1k\nb1 o 0 v=3*v(in)^2 m=2\nr1 o 0 1k\n\
         b2 p q i=5m*v(in)^3\nrp p 0 1k\nrq q 0 1k",
    )
    .unwrap();
    let n = c.unknown_count();
    let mut x = Vector::zeros(n);
    let (input, o, p, q) = (row(&c, "in"), row(&c, "o"), row(&c, "p"), row(&c, "q"));
    x.as_mut_slice()[input] = 0.5;
    let (a, b) = load(&c, &x, &ModelContext::default());
    let k = branch(&c, "b1");
    // v(o) - m f = m (f - f' x0): f = 0.75, f' = 3 at 0.5, m = 2.
    assert_eq!(a.get(o, k), 1.);
    assert_eq!(a.get(k, o), 1.);
    assert_eq!(a.get(k, input), -6.);
    assert!((b.as_slice()[k] - 2. * (0.75 - 1.5)).abs() < 1e-15);
    // The current 5m v^3 flows from p through the source to q.
    let g = 15e-3 * 0.25;
    assert!((a.get(p, input) - g).abs() < 1e-18);
    assert!((a.get(q, input) + g).abs() < 1e-18);
    let equivalent = 5e-3 * 0.125 - g * 0.5;
    assert!((b.as_slice()[p] + equivalent).abs() < 1e-18);
    assert!((b.as_slice()[q] - equivalent).abs() < 1e-18);
}

#[test]
fn the_ac_jacobian_is_the_bias_point_derivative() {
    let c = circuit("vin in 0 dc 0.5\nrin in 0 1k\nb1 o 0 v=3*v(in)^2\nr1 o 0 1k\nb2 0 p i=2m*exp(v(in))\nrp p 0 1k").unwrap();
    let mut bias = Vector::zeros(c.unknown_count());
    bias.as_mut_slice()[row(&c, "in")] = 0.5;
    let system = c
        .small_signal_system(&ModelContext::default(), &bias)
        .unwrap();
    let (input, p) = (row(&c, "in"), row(&c, "p"));
    assert_eq!(system.a.get(branch(&c, "b1"), input), -3.);
    assert!((system.a.get(p, input) + 2e-3 * 0.5_f64.exp()).abs() < 1e-18);
    assert!(system.e.triplets().is_empty());
}

#[test]
fn temperature_factor_follows_asrcload() {
    let scale = ngspice_rs::devices::BehaviouralScale {
        m: 2.,
        tc1: 1e-2,
        tc2: 1e-4,
        temperature: None,
        dtemp: 3.,
        reciprocal_tc: false,
        reciprocal_m: false,
    };
    // d = 27 + 273.15 + 3 - 300.15 = 3.
    let factor = 2. * (1. + 3e-2 + 9e-4);
    assert!((scale.factor(27.) - factor).abs() < 1e-14);
    let reciprocal = ngspice_rs::devices::BehaviouralScale {
        reciprocal_tc: true,
        reciprocal_m: true,
        temperature: Some(37.),
        dtemp: 0.,
        ..scale
    };
    assert!((reciprocal.factor(0.) - 1. / (1. + 0.1 + 0.01) / 2.).abs() < 1e-14);
}

#[test]
fn invalid_instances_are_explicit_errors() {
    for (body, message) in [
        ("b1 a a v=1\nr1 a 0 1", "shorted ASRC"),
        ("b1 a 0 v=i(vx)\nr1 a 0 1", "unknown controlling source vx"),
        (
            "b1 a 0 v=1 temp=30 dtemp=2\nr1 a 0 1",
            "both temp= and dtemp=",
        ),
        ("b1 a 0 v=k\nr1 a 0 1", "undefined parameter [k]"),
        ("b1 a 0 v=sqr(2)\nr1 a 0 1", "no such function 'sqr'"),
    ] {
        let error = circuit(body).unwrap_err();
        assert!(error.to_string().contains(message), "{body}: {error}");
    }
}

#[test]
fn the_registry_builds_b_sources_from_cards() {
    let registry = Registry::with_builtins();
    assert!(
        registry
            .get('b')
            .is_some_and(|entry| entry.support == ngspice_rs::devices::DeviceSupport::Ported)
    );
    let deck = parse_deck_text(Path::new("b.cir"), "t\nb1 a 0 v=2*v(c)+i(v1)\n");
    let card = RawCard::parse(&deck.lines[0]).unwrap();
    let mut nodes = ngspice_rs::primitives::NodeTable::new();
    let device = registry.instantiate(&card, &mut nodes).unwrap();
    assert_eq!(device.designator(), 'b');
    assert_eq!(device.terminals().len(), 3);
    assert_eq!(device.branch_currents(), 1);
    assert_eq!(device.findable_branch(), Some(0));
    assert_eq!(device.controlling_sources()[0].name, "v1");
    assert!(device.is_nonlinear());
    assert!(!device.limits_voltage_steps());
}

#[test]
fn derivative_trees_evaluate_like_cs_ptdifferentiate() {
    // bsource_zero_tran.cir's `1m*log(v(in))/v(in)`: C's derivative tree
    // ((1m * (1/x)) * x - 1m*log(x)) / pow(x, 2) gives this Jacobian entry
    // (instrumented no-FMA reference, load 375); `pow(x, 2)` is libm's, not
    // `x * x`.
    let program = program("1.0000000000e-03*log(v(in))/v(in)");
    let x = [2.757_542_198_308_712_5];
    let evaluation = program.evaluate(&environment(&x)).unwrap();
    let c = -1.885_810_314_559_648_2e-6;
    if cfg!(all(target_os = "macos", target_arch = "aarch64")) {
        // The libm the C reference value was captured with.
        assert_eq!(evaluation.gradient, vec![c]);
    } else {
        // Another libm may round `log`/`pow` differently in the last bit.
        assert_eq!(evaluation.gradient.len(), 1);
        let ulps = ((evaluation.gradient[0] - c) / (c.abs() * f64::EPSILON)).abs();
        assert!(ulps <= 4., "{} vs C {c:e}", evaluation.gradient[0]);
    }
}

#[test]
fn behavioural_convergence_uses_physical_output_and_newton_defect() {
    use ngspice_rs::devices::ConvergenceRequest;

    // The same physical current expressed through different multipliers.
    // Also test a small multiplier: raw expression changes alone are not
    // a reason to reject an otherwise converged physical output.
    for (expression, multiplier, from, to, reject) in [
        ("v(a)^2", "1e9", 1e-6, 1.1e-6, true),
        ("1e9*v(a)^2", "1", 1e-6, 1.1e-6, true),
        ("v(a)^2", "1e-12", 1., 1.1, false),
        // Equal endpoint values do not imply a valid linearisation:
        // the tangent at -1 uV predicts a different current at +1 uV.
        ("v(a)^2", "1e9", -1e-6, 1e-6, true),
    ] {
        let c = circuit(&format!("r1 a 0 1k\nb1 a 0 i={expression} m={multiplier}")).unwrap();
        let n = c.unknown_count();
        let mut previous = Vector::zeros(n);
        previous.as_mut_slice()[row(&c, "a")] = from;
        let mut solution = previous.clone();
        solution.as_mut_slice()[row(&c, "a")] = to;
        let history = c.state_history().unwrap();
        let mut trial = history.trial();
        let model_context = ModelContext::default();
        c.load(
            &LoadRequest {
                mode: AnalysisMode::OperatingPoint,
                solution: &previous,
                model_context: &model_context,
                integration: None,
                history: &history,
                forcing: None,
            },
            &mut SparseMatrix::new(n, n),
            &mut Vector::zeros(n),
            &mut trial,
        )
        .unwrap();
        let result = c
            .convergence_test(&ConvergenceRequest {
                solution: &solution,
                previous: &previous,
                history: &history,
                trial: &trial,
                model_context: &model_context,
                time: 0.,
                integration: None,
                reltol: 1e-3,
                abstol: 1e-12,
                vntol: 1e-6,
            })
            .unwrap();
        assert_eq!(
            result, reject,
            "{expression} m={multiplier}: {from} -> {to}"
        );
    }
}

#[test]
fn behavioural_output_trials_do_not_replace_the_accepted_observation() {
    let c = circuit("r1 a 0 1k\nb1 a 0 i=v(a)^2 m=1e9").unwrap();
    let n = c.unknown_count();
    let index = c.devices().iter().position(|d| d.name() == "b1").unwrap();
    let slot = c.state_rows(index).unwrap().start;
    let mut history = c.state_history().unwrap();
    let context = ModelContext::default();
    let load_at = |v: Real, history: &ngspice_rs::devices::StateHistory| {
        let mut x = Vector::zeros(n);
        x.as_mut_slice()[row(&c, "a")] = v;
        let mut trial = history.trial();
        c.load(
            &LoadRequest {
                mode: AnalysisMode::OperatingPoint,
                solution: &x,
                model_context: &context,
                integration: None,
                history,
                forcing: None,
            },
            &mut SparseMatrix::new(n, n),
            &mut Vector::zeros(n),
            &mut trial,
        )
        .unwrap();
        (x, trial)
    };
    let (x, trial) = load_at(1e-6, &history);
    assert!((trial.values()[slot] - 1e-3).abs() < 1e-18);
    c.accept_point(&x, None, &mut history, trial).unwrap();
    let accepted = history.clone();
    for v in [2e-6, 3e-6, -4e-6] {
        let (_, rejected) = load_at(v, &history);
        assert!((rejected.values()[slot] - 1e9 * v.powi(2)).abs() < 1e-17);
        drop(rejected);
        assert_eq!(history, accepted);
    }
    let (x, trial) = load_at(5e-6, &history);
    c.accept_point(&x, None, &mut history, trial).unwrap();
    assert!((history.accepted(1).unwrap()[slot] - 0.025).abs() < 1e-17);
}

#[test]
fn series_probe_elimination_preserves_the_physical_residual_and_jacobian() {
    let direct = circuit("i1 0 out 1.1u\nr1 out 0 100\nb1 out 0 i=v(out)^3 m=1e9").unwrap();
    let probed =
        circuit("i1 0 out 1.1u\nr1 out 0 100\nvprobe out sense 0\nb1 sense 0 i=v(out)^3 m=1e9")
            .unwrap();
    let reduced = |c: &Circuit, v: Real| {
        let out = row(c, "out");
        let mut x = Vector::zeros(c.unknown_count());
        x.as_mut_slice()[out] = v;
        let sense = c
            .nodes()
            .get("sense")
            .map(|node| c.unknowns().node_row(node).unwrap());
        if let Some(sense) = sense {
            x.as_mut_slice()[sense] = v;
            x.as_mut_slice()[branch(c, "vprobe")] = 1e9 * v.powi(3);
        }
        let (a, _) = load(c, &x, &ModelContext::default());
        let r = residual(c, &x);
        let mut kcl = r[out];
        let mut derivative = a.get(out, out);
        if let Some(sense) = sense {
            let probe = branch(c, "vprobe");
            assert!(r[sense].abs() < 1e-18 && r[probe].abs() < 1e-18);
            assert_eq!(a.get(out, probe) + a.get(sense, probe), 0.);
            kcl += r[sense];
            derivative += a.get(sense, out) + a.get(out, sense) + a.get(sense, sense);
        }
        (kcl, derivative)
    };
    for v in [-7e-6_f64, 0., 1e-5, 2e-5] {
        let expected = v / 100. + 1e9 * v.powi(3) - 1.1e-6;
        let slope = 0.01 + 3e9 * v.powi(2);
        for c in [&direct, &probed] {
            let (r, j) = reduced(c, v);
            assert!((r - expected).abs() < 1e-18, "v={v:e}, r={r:e}");
            assert!((j - slope).abs() < 1e-12, "v={v:e}, j={j:e}");
            let h = 1e-10;
            let numerical = (reduced(c, v + h).0 - reduced(c, v - h).0) / (2. * h);
            assert!(
                (j - numerical).abs() < 1e-9,
                "v={v:e}, j={j:e}, fd={numerical:e}"
            );
        }
    }
}

#[test]
fn explicit_absolute_power_values_and_jacobians() {
    let compiled = program("simetrix_abs_pow(v(x),v(p))");
    for (base, exponent, expected, slope, exponent_slope) in [
        (-4., 0.5, 2., -0.25, 2. * 4_f64.ln()),
        (4., 0.5, 2., 0.25, 2. * 4_f64.ln()),
        (-2., 3., 8., -12., 8. * 2_f64.ln()),
        (0., 0.25, 0., 0., 0.),
        (0., 0.5, 0., 0., 0.),
        (0., 1., 0., 0., 0.),
        (0., 1.5, 0., 0., 0.),
        (0., 0., 1., 0., 0.),
    ] {
        let actual = compiled.evaluate(&environment(&[base, exponent])).unwrap();
        assert_eq!(actual.value, expected);
        assert_eq!(actual.gradient, [slope, exponent_slope]);
    }
    assert!(compiled.evaluate(&environment(&[0., -0.5])).is_err());
    assert_gradient("simetrix_abs_pow(v(a),v(b))", POINTS);
}

#[test]
fn explicit_caret_and_complex_real_powers_are_distinct() {
    let caret = program("simetrix_caret_pow(v(x),v(p))");
    let real = program("simetrix_pow(v(x),v(p))");
    for (base, exponent, expected, slope) in [
        (-2., 3., -8., 12.),
        (-2., 2., 4., -4.),
        (-2., -3., -0.125, -0.1875),
        (0., 1., 0., 1.),
        (0., 0., 1., 0.),
    ] {
        for compiled in [&caret, &real] {
            let actual = compiled.evaluate(&environment(&[base, exponent])).unwrap();
            assert_eq!(actual.value, expected);
            assert_eq!(actual.gradient[0], slope);
        }
    }
    let result = caret.evaluate(&environment(&[-2., 3.])).unwrap();
    assert!((result.gradient[1] - 1842.068_074_4).abs() < 1e-7);
    let result = real.evaluate(&environment(&[-2., 3.])).unwrap();
    assert!((result.gradient[1] + 5.545_177_444_5).abs() < 1e-10);
    assert!(caret.evaluate(&environment(&[-2., 0.5])).is_err());
    let result = real.evaluate(&environment(&[-2., 0.5])).unwrap();
    assert!((result.value - 8.673_617_379_9e-17).abs() < 1e-18);
    assert!((result.gradient[0] + 2.164_89e-17).abs() < 1e-18);
    let result = caret.evaluate(&environment(&[0., 0.5])).unwrap();
    assert!((result.value / 1e-154 - 1.).abs() < 1e-12);
    assert!((result.gradient[0] / 5e99 - 1.).abs() < 1e-12);
    assert_gradient("simetrix_pow(v(a),v(b))", POINTS);
    assert_gradient(
        "simetrix_caret_pow(v(a),v(b))",
        &[&[("a", 0.3), ("b", -0.7)]],
    );
    // Native compatibility policy remains selected by native spellings.
    assert_eq!(value("pow(0,0)", &[]), 0.);
    assert_eq!(value("pwr(-2,3)", &[]), -8.);
    assert_eq!(value("(-2)^.5", &[]), 0.);
}

#[test]
fn only_explicit_ordered_zero_factor_masks_absolute_power_singularity() {
    let compiled = program("simetrix_abs_pow_product(sgn(v(x)),v(x),-.5)");
    for (base, expected, slope) in [(0., 0., 0.), (4., 0.5, -0.0625), (-4., -0.5, -0.0625)] {
        let actual = compiled.evaluate(&environment(&[base])).unwrap();
        assert_eq!(actual.value, expected);
        assert_eq!(actual.gradient, [slope]);
    }
    let compiled = program("simetrix_abs_pow_product(v(f),v(x),v(p))");
    let result = compiled.evaluate(&environment(&[0., 4., 0.5])).unwrap();
    assert_eq!(result.value, 0.);
    assert_eq!(result.gradient, [2., 0., 0.]);
    assert!(compiled.evaluate(&environment(&[1., 0., -0.5])).is_err());
    let result = compiled.evaluate(&environment(&[0., 0., -0.5])).unwrap();
    assert_eq!(result.value, 0.);
    assert_eq!(result.gradient, [0., 0., 0.]);
    let ordinary = program("sgn(v(x))*simetrix_abs_pow(v(x),-.5)");
    assert!(ordinary.evaluate(&environment(&[0.])).is_err());
}
