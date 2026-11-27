//! Degenerate circuits through every analysis: each either fails as C
//! ngspice does (same class of error, naming the same unknown) or gives C's
//! answer. Expected values come from the live C binary with `.options sparse`
//! (`set numdgt=17`, `print all`); C's error classes from its batch output.
//!
//! Where C recovers from a failed operating point with its transient
//! fallback (`OPtran`, `optran.c`), which the port has not ported, the port
//! fails explicitly instead; those cases only check the singular unknown.

use ngspice_rs::analysis::{Plot, RunConfig};
use ngspice_rs::netlist::Parser;
use ngspice_rs::netlist::source::parse_deck_text;
use ngspice_rs::primitives::SpiceError;
use std::path::Path;

fn simulate(body: &str) -> Result<Plot, SpiceError> {
    let n = Parser::new()
        .parse_deck(&parse_deck_text(
            Path::new("robust.cir"),
            &format!("t\n{body}\n.end\n"),
        ))
        .unwrap_or_else(|error| panic!("{error}\n{body}"));
    let c = RunConfig::from_netlist(&n)?;
    let request = c.plan_for(&n.analyses[0])?;
    let mut circuit = c.circuit(&n)?;
    request.run(&mut circuit, &c.context())
}

fn value(plot: &Plot, name: &str) -> f64 {
    plot.value(name, plot.point_count() - 1)
        .unwrap_or_else(|| panic!("no {name}"))
        .re
}

/// The deck fails with a numerical error naming `node` as `NIiter`'s
/// warning does (`singular matrix:  check node <node>`).
fn singular_at(body: &str, node: &str) {
    let error = simulate(body).expect_err(body);
    assert!(
        matches!(error, SpiceError::Numerical { .. }),
        "{body}: {error}"
    );
    let text = error.to_string();
    assert!(
        text.contains(&format!("singular matrix:  check node {node}")),
        "{body}: {text}"
    );
}

const VLOOP: &str = "v1 a 0 1\nv2 a 0 2\nr1 a 0 1k";

/// Parallel voltage sources: C fails every analysis (`DC solution failed`,
/// `AC operating point failed`, `Transient op failed`) after warning
/// `singular matrix:  check node v1#branch`.
#[test]
fn voltage_source_loops_fail_in_every_analysis() {
    for analysis in [".op", ".dc v1 0 1 0.5", ".tran 1u 10u"] {
        singular_at(&format!("{VLOOP}\n{analysis}"), "v1#branch");
    }
    singular_at(
        "v1 a 0 1 ac 1\nv2 a 0 2\nr1 a 0 1k\n.ac dec 2 1 100",
        "v1#branch",
    );
    // Equal values are still a loop.
    singular_at("v1 a 0 1\nv2 a 0 1\nr1 a 0 1k\n.op", "v1#branch");
    // An inductor is a DC short: in parallel with a source it is a loop.
    singular_at("v1 a 0 1\nl1 a 0 1m\n.op", "l1#branch");
    // A VCVS sensing its own output with unit gain has no equation.
    singular_at(
        "v1 a 0 1\nr1 a 0 1k\ne1 b 0 b 0 1\nr2 b 0 1k\n.op",
        "e1#branch",
    );
    singular_at("v1 a 0 1\ne1 a 0 a 0 2\nr1 a 0 1k\n.op", "v1#branch");
}

/// Nodes without a DC path: singular at the operating point in C (which
/// then recovers through `OPtran`; the port reports the failure).
#[test]
fn floating_nodes_fail_at_the_operating_point() {
    singular_at("v1 a 0 1\nr1 a 0 1k\nr2 x y 1k\n.op", "x");
    singular_at("v1 a 0 1\nr1 a b 1k\nc1 b c 1u\nc2 c 0 1u\n.op", "c");
    singular_at("i1 0 a 1m\nc1 a 0 1u\n.tran 1u 10u", "a");
    // C: `doAnalyses: matrix is singular` after the operating point.
    singular_at("v1 a 0 0 ac 1\nr1 a 0 1k\nr2 x y 1k\n.ac dec 2 10 1k", "x");
}

/// With `uic` there is no operating point, and the capacitor companion
/// makes the transient matrix regular: C and the port both simulate.
#[test]
fn floating_capacitor_nodes_simulate_with_uic() {
    let plot = simulate("i1 0 a 1m\nc1 a 0 1u\n.tran 1u 10u uic").unwrap();
    // 1 mA into 1 uF for 10 us.
    let v = value(&plot, "v(a)");
    assert!((v - 1e-2).abs() < 1e-9, "{v}");
}

/// A node joined only to current sources has no matrix element, so SPARSE
/// never numbers it and C solves the rest, leaving the node's right-hand
/// side (the net injected current, here 2 mA - 1 mA) as its "voltage":
/// C prints `a = 1.00000000000000002e-03`, `b = 1.0`. The port matches it
/// (with a warning) in every analysis.
#[test]
fn current_source_cutset_keeps_the_right_hand_side_as_c_does() {
    let body = "i1 0 a 2m\ni2 a b 1m\nr1 b 0 1k";
    let plot = simulate(&format!("{body}\n.op")).unwrap();
    assert_eq!(value(&plot, "v(a)"), 1e-3);
    assert_eq!(value(&plot, "v(b)"), 1.0);
    let plot = simulate(&format!("{body}\n.tran 1u 10u")).unwrap();
    assert_eq!(value(&plot, "v(b)"), 1.0);
    simulate("i1 0 a 0 ac 2m\ni2 a b 0 ac 1m\nr1 b 0 1k\n.ac lin 1 1k 1k").unwrap();
}

/// A conductance below `pivtol` (1e-14 < 1e-13) is a small pivot: SPARSE
/// takes the largest element anyway (`spSMALL_PIVOT`, which ngspice treats
/// as `OK`), so C solves it exactly: `a = 1.0e+05`. The port warns.
#[test]
fn small_pivots_are_accepted_like_c() {
    for options in ["", ".options pivtol=1e-16"] {
        let plot = simulate(&format!("i1 0 a 1n\nr1 a 0 1e14\n{options}\n.op")).unwrap();
        assert_eq!(value(&plot, "v(a)"), 1e5, "{options}");
    }
}

/// `resparam.c` raises a zero resistance to `RESMIN = 1e-12` (with a
/// warning). The 1e12 S conductance then loses digits. C's reference
/// is `b = 9.99999999999999001e-01` and
/// `v1#branch = -9.7656250000000000e-04`. The branch-current result
/// suffers cancellation at 1e12 S, so compare within a few FP rounding
/// steps on its physical 1 V / 1 kohm current scale, not bit-for-bit.
#[test]
fn zero_resistance_is_clamped_like_c() {
    for r1 in ["r1 a b 0", "r1 a b rm r=0\n.model rm r"] {
        let plot = simulate(&format!("v1 a 0 1\n{r1}\nr2 b 0 1k\n.op")).unwrap();
        assert_eq!(value(&plot, "v(b)"), 9.999_999_999_999_99e-1, "{r1}");
        let branch_current = value(&plot, "i(v1)");
        let expected = -9.765_625e-4;
        let current_scale = 1e-3; // 1 V source into the 1 kohm load
        let bound = 8. * f64::EPSILON * current_scale;
        assert!(
            (branch_current - expected).abs() <= bound,
            "{r1}: branch current {branch_current:e}, C {expected:e}, bound {bound:e}"
        );
    }
}

/// Zero capacitance and inductance are legal in C: a capacitor that stamps
/// nothing and an inductor that is a short at every analysis.
#[test]
fn zero_capacitance_and_inductance_simulate() {
    let plot = simulate("v1 a 0 1\nr1 a b 1k\nl1 b c 0\nr2 c 0 1k\n.op").unwrap();
    assert_eq!(value(&plot, "v(c)"), 0.5);
    assert_eq!(value(&plot, "i(l1)"), 5e-4);
    let plot = simulate("v1 a 0 1\nr1 a b 1k\nc1 b 0 0\nr2 b 0 1k\n.op").unwrap();
    assert_eq!(value(&plot, "v(b)"), 0.5);
    for body in [
        "v1 a 0 pulse(0 1 1u 1u 1u 5u 20u)\nr1 a b 1k\nc1 b 0 0\n.tran 1u 20u",
        "v1 a 0 pulse(0 1 1u 1u 1u 5u 20u)\nr1 a b 1k\nl1 b 0 0\nr2 b 0 1k\n.tran 1u 20u",
        // M = k sqrt(L1 L2) = 0 for a zero inductance.
        "v1 a 0 pulse(0 1 1u 1u 1u 5u 20u)\nr1 a b 1k\nl1 b 0 0\nl2 c 0 1m\nr2 c 0 1k\n\
         k1 l1 l2 0.5\n.tran 1u 20u",
        "v1 a 0 0 ac 1\nr1 a b 1k\nc1 b 0 0\n.ac dec 2 10 1k",
    ] {
        simulate(body).unwrap_or_else(|error| panic!("{body}: {error}"));
    }
}

/// Optional quality telemetry must not overflow while the physical residual
/// ratio is finite, or erase cancelling entries before summation.
#[test]
fn original_backward_error_retains_extreme_real_products_and_small_residuals() {
    use ngspice_rs::maths::equilibration::OriginalResidual;

    let huge = f64::MAX;
    let tiny = f64::from_bits(1);
    let m = [(0, 0, huge)];
    let report = OriginalResidual::real(1, &m, false)
        .unwrap()
        .evaluate(&[huge], &[2.])
        .unwrap();
    assert!((report.maximum - 1. / 3.).abs() < 1e-15, "{report:?}");
    let mut correction = [0.];
    OriginalResidual::real(1, &m, false)
        .unwrap()
        .evaluate_with_correction(&[huge], &[2.], &mut correction)
        .unwrap();
    assert_eq!(correction, [-huge]);

    let m = [(0, 0, tiny)];
    let report = OriginalResidual::real(1, &m, false)
        .unwrap()
        .evaluate(&[tiny], &[2.])
        .unwrap();
    assert!((report.maximum - 1. / 3.).abs() < 1e-15, "{report:?}");
    OriginalResidual::real(1, &m, false)
        .unwrap()
        .evaluate_with_correction(&[tiny], &[2.], &mut correction)
        .unwrap();
    assert_eq!(correction, [-tiny]);

    let m = [(0, 0, huge), (0, 1, -huge)];
    let report = OriginalResidual::real(2, &m, false)
        .unwrap()
        .evaluate(&[0., 0.], &[2., 2.])
        .unwrap();
    assert_eq!(report.rows, [0., 0.]);
    assert_eq!(report.maximum, 0.);
    let m = [(0, 0, huge)];
    let ratio = OriginalResidual::real(1, &m, false).unwrap();
    assert_eq!(
        ratio
            .evaluate_with_correction(&[0.], &[3.], &mut correction)
            .unwrap()
            .maximum,
        1.
    );
    assert_eq!(correction, [f64::NEG_INFINITY]);
}

/// The very same original operator can evaluate normal, transposed, and
/// repeated RHS without allocating a new coefficient snapshot or refactoring.
#[test]
fn original_backward_error_uses_transposed_equations_and_many_rhs() {
    use ngspice_rs::maths::equilibration::OriginalResidual;

    let a = [(0, 0, 1.), (0, 1, 2.), (1, 0, 3.), (1, 1, 4.)];
    let normal = OriginalResidual::real(2, &a, false).unwrap();
    let transposed = OriginalResidual::real(2, &a, true).unwrap();
    let x1 = [1., 1.];
    let x2 = [2., -1.];
    let b1 = [3., 7.];
    let b2 = [0., 2.];
    let bt1 = [4., 6.];
    let bt2 = [-1., 0.];
    assert_eq!(normal.evaluate(&b1, &x1).unwrap().maximum, 0.);
    assert_eq!(transposed.evaluate(&bt1, &x1).unwrap().maximum, 0.);
    let pairs: [(&[f64], &[f64]); 2] = [(&b1, &x1), (&b2, &x2)];
    assert_eq!(
        normal
            .evaluate_many(&pairs)
            .unwrap()
            .into_iter()
            .map(|r| r.maximum)
            .collect::<Vec<_>>(),
        [0., 0.]
    );
    assert_eq!(transposed.evaluate(&bt2, &x2).unwrap().maximum, 0.);
    assert!(normal.evaluate(&bt1, &x1).unwrap().maximum > 0.);
    assert!(normal.evaluate(&[f64::NAN, 0.], &x1).is_err());
    assert!(normal.evaluate(&b1, &[1.]).is_err());
    assert!(OriginalResidual::real(2, &[(2, 0, 1.)], false).is_err());
    assert!(OriginalResidual::real(1, &[(0, 0, f64::INFINITY)], false).is_err());
}

/// Complex residuals retain their phase and products even where the
/// unnormalized imaginary product would exceed the binary64 exponent range.
#[test]
fn original_backward_error_inspects_complex_phases_and_transpose() {
    use ngspice_rs::maths::equilibration::OriginalResidual;
    use ngspice_rs::primitives::Complex;

    let huge = f64::MAX;
    let a = [(0, 0, Complex::new(huge, huge))];
    let report = OriginalResidual::complex(1, &a, false)
        .unwrap()
        .evaluate(&[Complex::ZERO], &[Complex::new(1., 1.)])
        .unwrap();
    assert_eq!(report.maximum, 0.5);

    let a = [
        (0, 0, Complex::new(1., 2.)),
        (0, 1, Complex::new(2., -1.)),
        (1, 0, Complex::new(3., 0.)),
        (1, 1, Complex::new(0., -1.)),
    ];
    let x = [Complex::new(1., 0.), Complex::new(0., 1.)];
    let b = [Complex::new(2., 4.), Complex::new(4., 0.)];
    let bt = [Complex::new(1., 5.), Complex::new(3., -1.)];
    assert_eq!(
        OriginalResidual::complex(2, &a, false)
            .unwrap()
            .evaluate(&b, &x)
            .unwrap()
            .maximum,
        0.
    );
    let transposed = OriginalResidual::complex(2, &a, true).unwrap();
    assert_eq!(transposed.evaluate(&bt, &x).unwrap().maximum, 0.);
    assert_eq!(transposed.evaluate_many(&[(&bt, &x)]).unwrap().len(), 1);
}
