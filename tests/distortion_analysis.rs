//! `.disto` (GitHub #104) through production APIs: parsed decks, `RunConfig`,
//! `AnalysisPlan::run_plots`. Accuracy is judged against the
//! closed-form Volterra kernels of a diode's exponential; the committed C
//! goldens are verified by `cargo xtask golden verify` and the opt-in live-C
//! comparison is `c_disto_reference.rs`.
use std::path::Path;

use ngspice_rs::analysis::{Plot, RunConfig};
use ngspice_rs::devices::noise::{BOLTZMANN, CHARGE};
use ngspice_rs::netlist::Parser;
use ngspice_rs::netlist::source::parse_deck_text;
use ngspice_rs::primitives::{Complex, SpiceResult};

const KELVIN: f64 = 300.15;
const GMIN: f64 = 1e-12;

fn run_at(deck: &str, index: usize) -> SpiceResult<Vec<Plot>> {
    let netlist = Parser::new().parse_deck(&parse_deck_text(Path::new("disto.cir"), deck))?;
    let config = RunConfig::from_netlist(&netlist)?;
    let request = config.plan_for(&netlist.analyses[index])?;
    let mut circuit = config.circuit(&netlist)?;
    request.run_plots(&mut circuit, &config.context())
}

fn run(deck: &str) -> Vec<Plot> {
    run_at(deck, 0).unwrap()
}

fn error_of(deck: &str) -> String {
    run_at(deck, 0).expect_err("expected failure").to_string()
}

fn column<'a>(plot: &'a Plot, name: &str) -> ngspice_rs::analysis::ColumnView<'a> {
    plot.column(name)
        .unwrap_or_else(|| panic!("{name} missing from {}", plot.plotname))
}

fn close(got: Complex, want: f64, relative: f64) {
    assert!(
        (got.re - want).abs() <= relative * want.abs() && got.im.abs() <= relative * want.abs(),
        "{got} != {want:e} (relative {relative:e})"
    );
}

/// A diode biased by a current source, with a parallel resistor:
/// `I0 = IS (exp(v / (N Vt)) - 1) + v / R + GMIN v`.
struct DiodeStage {
    i0: f64,
    r: f64,
    is: f64,
    n: f64,
}

impl DiodeStage {
    const STAGE: Self = Self {
        i0: 1e-3,
        r: 10e3,
        is: 1e-14,
        n: 1.5,
    };

    fn vte(&self) -> f64 {
        self.n * BOLTZMANN / CHARGE * KELVIN
    }

    /// The bias by bisection of the (monotonic) closed form.
    fn bias(&self) -> f64 {
        let vte = self.vte();
        let f = |v: f64| self.is * ((v / vte).exp() - 1.) + v / self.r + GMIN * v - self.i0;
        let (mut low, mut high) = (0., 2.);
        for _ in 0..200 {
            let mid = 0.5 * (low + high);
            if f(mid) > 0. {
                high = mid;
            } else {
                low = mid;
            }
        }
        0.5 * (low + high)
    }

    /// (total small-signal conductance, c2, c3) at the bias: the exponential's
    /// Taylor coefficients `IS e / (2 vte^2)` and `IS e / (6 vte^3)`.
    fn coefficients(&self) -> (f64, f64, f64) {
        let vte = self.vte();
        let e = (self.bias() / vte).exp();
        let gd = self.is * e / vte;
        (
            gd + GMIN + 1. / self.r,
            gd / (2. * vte),
            gd / (6. * vte * vte),
        )
    }

    fn deck(&self, inputs: &str, card: &str) -> String {
        format!(
            "diode stage\ni1 0 a dc {} {inputs}\nr1 a 0 {}\nd1 a 0 dm\n\
             .model dm d(is={} n={})\n.option reltol=1e-12\n{card}\n.end\n",
            self.i0, self.r, self.is, self.n
        )
    }
}

#[test]
fn diode_harmonics_follow_the_exponential_taylor_coefficients() {
    let stage = DiodeStage::STAGE;
    let amplitude = 2e-6;
    let plots = run(&stage.deck(&format!("distof1 {amplitude}"), ".disto dec 2 1k 100k"));
    assert_eq!(plots.len(), 2);
    assert_eq!(plots[0].plotname, "DISTORTION - 2nd harmonic");
    assert_eq!(plots[1].plotname, "DISTORTION - 3rd harmonic");
    let (g, c2, c3) = stage.coefficients();
    // cktdisto.c drives a current input with the opposite sign of its AC
    // stamp: `i1 0 a ... distof1 A` pulls 0.5 A out of `a`.
    let h1 = -0.5 * amplitude / g;
    let h2 = -c2 * h1 * h1 / g;
    let h3 = -(c2 * 2. * h1 * h2 + c3 * h1 * h1 * h1) / g;
    for plot in &plots {
        assert_eq!(column(plot, "frequency").len(), 5);
    }
    for value in column(&plots[0], "v(a)") {
        close(value, 2. * h2, 1e-9);
    }
    for value in column(&plots[1], "v(a)") {
        close(value, 2. * h3, 1e-9);
    }
    // The classical small-signal result: the second harmonic is
    // c2 a^2 / (2 g) for a fundamental of amplitude a = 2 h1 at the node.
    let a = 2. * h1;
    close(column(&plots[0], "v(a)")[0], -c2 * a * a / (2. * g), 1e-9);
    // Without a value, `distof1` drives magnitude 1 (vsrcpar.c).
    let unit = run(&stage.deck("distof1", ".disto lin 0 1k 1k"));
    let h1 = -0.5 / g;
    close(column(&unit[0], "v(a)")[0], -2. * c2 * h1 * h1 / g, 1e-9);
}

#[test]
fn diode_intermodulation_follows_the_volterra_kernels() {
    let stage = DiodeStage::STAGE;
    let (a, b) = (2e-6, 1e-6);
    let plots = run(&stage.deck(
        &format!("distof1 {a} distof2 {b}"),
        ".disto oct 1 1k 4k 0.9",
    ));
    let titles: Vec<&str> = plots.iter().map(|p| p.plotname.as_str()).collect();
    assert_eq!(
        titles,
        [
            "DISTORTION - IM: f1+f2",
            "DISTORTION - IM: f1-f2",
            "DISTORTION - IM: 2f1-f2"
        ]
    );
    let (g, c2, c3) = stage.coefficients();
    let (x, y) = (-0.5 * a / g, -0.5 * b / g);
    let sum = -c2 * x * y / g;
    let h2 = -c2 * x * x / g;
    // dloadfns.c D1n2F12: (c2 (4 X M + 2 B X2) + 3 c3 X^2 B) / 3.
    let third = -(c2 * (4. * x * sum + 2. * y * h2) + 3. * c3 * x * x * y) / 3. / g;
    for value in column(&plots[0], "v(a)") {
        close(value, 4. * sum, 1e-9);
    }
    for value in column(&plots[1], "v(a)") {
        close(value, 4. * sum, 1e-9);
    }
    for value in column(&plots[2], "v(a)") {
        close(value, 6. * third, 1e-9);
    }
}

#[test]
fn a_linear_circuit_has_no_distortion_and_c_names() {
    let plots = run(
        "rc\nv1 in 0 dc 1 ac 1 distof1 0.1\nr1 in out 1k\nc1 out 0 1u\n\
         .disto lin 2 1k 4k\n.end\n",
    );
    let names: Vec<&str> = plots[0]
        .variables()
        .iter()
        .map(|v| v.name.as_str())
        .collect();
    assert_eq!(names, ["frequency", "v(in)", "v(out)", "i(v1)"]);
    assert_eq!(plots[0].variables()[1].unit, "voltage");
    assert!(plots[0].flags().is_complex());
    // distoan.c steps a linear sweep by (stop - start) / (pts + 1).
    let freqs: Vec<f64> = column(&plots[0], "frequency")
        .iter()
        .map(|c| c.re)
        .collect();
    assert_eq!(freqs, [1e3, 2e3, 3e3, 4e3]);
    for plot in &plots {
        for name in ["v(in)", "v(out)", "i(v1)"] {
            assert!(column(plot, name).iter().all(|v| *v == Complex::ZERO));
        }
    }
}

#[test]
fn invalid_cards_and_unsupported_devices_are_refused() {
    let base = "d\nv1 in 0 dc 0.7 distof1 0.01\nr1 in a 1k\nd1 a 0 dm\n.model dm d\n";
    let cases = [
        (".disto dec 2 1k 10k 0.9", "distof2"),
        (".disto log 2 1k 10k", "dec, oct or lin"),
        (".disto dec 2 0 10k", "positive"),
        (".disto dec 2 10k 1k", "fstart <= fstop"),
        (".disto dec 0 1k 10k", "at least 1 step"),
        (".disto dec 2 1k", "needs {dec|oct|lin}"),
    ];
    for (card, message) in cases {
        let error = error_of(&format!("{base}{card}\n.end\n"));
        assert!(error.contains(message), "{card}: {error}");
    }
    let behavioural = error_of(&format!(
        "{base}b1 b 0 v=v(a)*v(a)\nrb b 0 1k\n.disto dec 2 1k 10k\n.end\n"
    ));
    assert!(behavioural.contains("B source"), "{behavioural}");
}

mod binding {
    use ngspice_rs::devices::distortion::{
        Control, DeviceDistortion, DistortionContext, DistortionTerm, Product, Response, Taylor,
        circuit_distortion,
    };
    use ngspice_rs::devices::{Circuit, Device, MnaUnknowns, ModelContext, StampContext};
    use ngspice_rs::maths::Vector;
    use ngspice_rs::primitives::{Complex, NodeId, NodeTable, SpiceResult};

    #[derive(Debug)]
    struct Terms {
        nodes: [NodeId; 2],
        terms: Vec<DistortionTerm>,
        probe: Option<NodeId>,
    }

    impl Device for Terms {
        fn name(&self) -> &str {
            "x1"
        }
        fn designator(&self) -> char {
            'x'
        }
        fn terminals(&self) -> &[NodeId] {
            &self.nodes
        }
        fn stamp(&self, _: &mut StampContext<'_>) -> SpiceResult<()> {
            Ok(())
        }
        fn distortion(&self, context: &DistortionContext<'_>) -> SpiceResult<DeviceDistortion> {
            if let Some(node) = self.probe {
                context.voltage(node)?;
            }
            Ok(DeviceDistortion::Terms(self.terms.clone()))
        }
    }

    fn term() -> DistortionTerm {
        DistortionTerm::new(
            Response::Current,
            [NodeId::new(1), NodeId::GROUND],
            vec![Control::between(NodeId::new(1), NodeId::GROUND)],
            Taylor::single(2., 3.),
        )
    }

    fn circuit(terms: Vec<DistortionTerm>, probe: Option<NodeId>) -> Circuit {
        let mut circuit = Circuit::new();
        let a = circuit.add_node("a").unwrap();
        circuit.add_node("b").unwrap();
        circuit.add_node("c").unwrap();
        circuit
            .add_device(Box::new(Terms {
                nodes: [a, NodeId::GROUND],
                terms,
                probe,
            }))
            .unwrap();
        circuit.finalize().unwrap();
        circuit
    }

    #[test]
    fn malformed_descriptions_fail_before_product_evaluation() {
        let invalid = NodeId::new(99);
        let mut cases = Vec::new();
        for index in 0..5 {
            let mut bad = term();
            match index {
                0 => bad.nodes[0] = invalid,
                1 => bad.controls[0].pairs[0][1] = invalid,
                2 => bad.controls[0].minus.push([NodeId::GROUND, invalid]),
                3 => bad.controls[0]
                    .unconjugated_in_f1_minus_f2
                    .push([invalid, NodeId::GROUND]),
                _ => bad.nodes[1] = invalid,
            }
            cases.push((bad, "unmapped MNA node"));
        }
        for arity in [0, 4] {
            let mut bad = term();
            bad.controls.resize(arity, bad.controls[0].clone());
            cases.push((bad, "one to three"));
        }
        for coefficients in [
            Taylor {
                yy: 1.,
                ..Taylor::default()
            },
            Taylor {
                xxz: 1.,
                ..Taylor::default()
            },
            Taylor {
                xyz: 1.,
                ..Taylor::default()
            },
        ] {
            let mut bad = term();
            bad.taylor = coefficients;
            cases.push((bad, "absent control"));
        }
        for invalid in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let mut bad = term();
            bad.taylor.xxx = invalid;
            cases.push((bad, "nonfinite Taylor"));
            cases.push((
                term().with_charge(Taylor::single(0., invalid)),
                "nonfinite Taylor",
            ));
        }
        cases.push((
            term().with_charge(Taylor {
                zz: 1.,
                ..Taylor::default()
            }),
            "absent control",
        ));
        let mut bad = term().with_charge(Taylor::single(1., 0.));
        bad.response = Response::Charge;
        cases.push((bad, "folded charge requires"));
        for (bad, expected) in cases {
            let c = circuit(vec![term(), bad], None);
            let error = circuit_distortion(&c, &ModelContext::default(), &Vector::zeros(3))
                .err()
                .unwrap();
            let message = error.to_string();
            assert!(
                message.contains("x1") && message.contains(expected),
                "{message}"
            );
        }
        let c = circuit(vec![term()], Some(invalid));
        assert!(circuit_distortion(&c, &ModelContext::default(), &Vector::zeros(3)).is_err());
    }

    #[test]
    fn ground_and_deferred_nodes_are_not_interchangeable() {
        let mut nodes = NodeTable::new();
        let a = nodes.intern("a").unwrap();
        let deferred = nodes.intern("deferred").unwrap();
        let mut unknowns = MnaUnknowns::new();
        unknowns.rebuild_deferring(&nodes, |id| id == deferred);
        let model = ModelContext::default();
        let bias = Vector::from_slice(&[2.]);
        let context = DistortionContext::new(&model, &unknowns, &bias).unwrap();
        assert_eq!(
            context.voltage(NodeId::GROUND).unwrap().to_bits(),
            0_f64.to_bits()
        );
        assert_eq!(context.voltage(a).unwrap(), 2.);
        assert!(context.voltage(deferred).is_err());
        assert!(context.voltage(NodeId::new(99)).is_err());
        assert!(DistortionContext::new(&model, &unknowns, &Vector::zeros(0)).is_err());
        assert!(
            DistortionContext::new(&model, &unknowns, &Vector::from_slice(&[f64::NAN])).is_err()
        );
        let c = Circuit::new();
        assert!(circuit_distortion(&c, &model, &Vector::zeros(0)).is_err());
    }

    #[test]
    fn every_product_rejects_bad_kernel_dimensions_without_touching_rhs() {
        let c = circuit(vec![term()], None);
        let bound = circuit_distortion(&c, &ModelContext::default(), &Vector::zeros(3)).unwrap();
        let good = [Complex::real(2.); 3];
        let short = [Complex::real(2.); 2];
        let bad = [Complex::ZERO, Complex::real(f64::NAN), Complex::ZERO];
        let initial = [Complex::new(-0., 1.), Complex::new(-3., -0.), Complex::ZERO];
        for bad_kernel in [&short[..], &bad[..]] {
            for product in [
                Product::TwoF1 { h1: bad_kernel },
                Product::ThreeF1 {
                    h1: &good,
                    h2: bad_kernel,
                },
                Product::F1PlusF2 {
                    h1: &good,
                    h1_f2: bad_kernel,
                },
                Product::F1MinusF2 {
                    h1: &good,
                    h1_f2: bad_kernel,
                },
                Product::TwoF1MinusF2 {
                    h1: &good,
                    h1_f2: bad_kernel,
                    h2: &good,
                    h2_minus: &good,
                },
                Product::TwoF1MinusF2 {
                    h1: &good,
                    h1_f2: &good,
                    h2: bad_kernel,
                    h2_minus: &good,
                },
                Product::TwoF1MinusF2 {
                    h1: &good,
                    h1_f2: &good,
                    h2: &good,
                    h2_minus: bad_kernel,
                },
            ] {
                let mut rhs = initial;
                assert!(bound.add_products(product, 1., &mut rhs).is_err());
                assert_eq!(
                    rhs.map(|v| [v.re.to_bits(), v.im.to_bits()]),
                    initial.map(|v| [v.re.to_bits(), v.im.to_bits()])
                );
            }
        }
        let mut rhs = initial;
        assert!(
            bound
                .add_products(Product::TwoF1 { h1: &good }, f64::NAN, &mut rhs)
                .is_err()
        );
        assert_eq!(rhs, initial);
        assert!(
            bound
                .add_products(Product::TwoF1 { h1: &short }, 1., &mut rhs[..2])
                .is_err()
        );
    }

    #[test]
    fn bound_rows_evaluate_all_arities_and_preserve_quirks() {
        let [a, b, c] = [1, 2, 3].map(NodeId::new);
        let ground = NodeId::GROUND;
        let mut two = term();
        two.controls.push(Control::between(b, ground));
        two.taylor = Taylor {
            xy: 1.,
            ..Taylor::default()
        };
        let mut three = two.clone();
        three.controls.push(Control::between(c, ground));
        three.taylor = Taylor {
            xyz: 1.,
            ..Taylor::default()
        };
        let one = term();
        let circuit = circuit(vec![one, two, three], None);
        let bound =
            circuit_distortion(&circuit, &ModelContext::default(), &Vector::zeros(3)).unwrap();
        assert_eq!(bound.term_count(), 3);
        let h1 = [Complex::real(2.), Complex::real(3.), Complex::real(4.)];
        let h2 = [Complex::real(1.); 3];
        let mut rhs = [Complex::ZERO; 3];
        bound
            .add_products(Product::TwoF1 { h1: &h1 }, 1., &mut rhs)
            .unwrap();
        assert_eq!(rhs, [Complex::real(-14.), Complex::ZERO, Complex::ZERO]);
        rhs.fill(Complex::ZERO);
        bound
            .add_products(Product::ThreeF1 { h1: &h1, h2: &h2 }, 1., &mut rhs)
            .unwrap();
        assert_eq!(rhs, [Complex::real(-61.), Complex::ZERO, Complex::ZERO]);
        // All values are exact integers: 8 + 24 plus 5 plus 24 = 61.
        // BJT's unconjugated term must remain unconjugated in f1-f2.
        let control =
            Control::between(a, ground).with_unconjugated_in_f1_minus_f2(vec![[b, ground]]);
        let quirk = DistortionTerm::new(
            Response::Current,
            [a, ground],
            vec![control],
            Taylor::single(1., 0.),
        )
        .with_charge(Taylor::single(1., 0.));
        let circuit = self::circuit(vec![quirk], None);
        let bound =
            circuit_distortion(&circuit, &ModelContext::default(), &Vector::zeros(3)).unwrap();
        let first = [Complex::real(1.), Complex::real(2.), Complex::ZERO];
        let second = [Complex::new(3., 4.), Complex::new(5., 6.), Complex::ZERO];
        rhs.fill(Complex::ZERO);
        bound
            .add_products(
                Product::F1MinusF2 {
                    h1: &first,
                    h1_f2: &second,
                },
                -2.,
                &mut rhs,
            )
            .unwrap();
        // x=3, b=8+j2; current=24+j6, charge current=12-j48.
        assert_eq!(rhs, [Complex::new(-36., 42.), Complex::ZERO, Complex::ZERO]);
        let circuit = self::circuit(vec![term().with_first_order_im3_kernel()], None);
        let bound =
            circuit_distortion(&circuit, &ModelContext::default(), &Vector::zeros(3)).unwrap();
        rhs.fill(Complex::ZERO);
        bound
            .add_products(
                Product::TwoF1MinusF2 {
                    h1: &h1,
                    h1_f2: &h2,
                    h2: &h2,
                    h2_minus: &h2,
                },
                1.,
                &mut rhs,
            )
            .unwrap();
        assert_eq!(rhs[0], Complex::real(-20.));
    }

    #[test]
    #[ignore = "opt-in distortion preparation and product evaluation timing"]
    fn bound_distortion_workload() {
        use std::hint::black_box;
        use std::time::Instant;
        let c = circuit(vec![term(); 4096], None);
        let model = ModelContext::default();
        let bias = Vector::zeros(3);
        let start = Instant::now();
        let bound = circuit_distortion(&c, &model, &bias).unwrap();
        println!(
            "distortion prepare: {} terms {} ns",
            bound.term_count(),
            start.elapsed().as_nanos()
        );
        let h1 = [Complex::real(2.); 3];
        let h2 = [Complex::real(1.); 3];
        let mut rhs = [Complex::ZERO; 3];
        let start = Instant::now();
        for _ in 0..1000 {
            rhs.fill(Complex::ZERO);
            black_box(&bound)
                .add_products(Product::ThreeF1 { h1: &h1, h2: &h2 }, 1., &mut rhs)
                .unwrap();
            black_box(&rhs);
        }
        println!(
            "distortion 1000 products: {} ns",
            start.elapsed().as_nanos()
        );
        assert_eq!(rhs[0], Complex::real(-32. * 4096.));
    }
}

#[test]
fn reactive_intermodulation_reuses_independent_kernels_on_both_sides_of_f2() {
    let stage = DiodeStage::STAGE;
    let (g, c2, c3) = stage.coefficients();
    let capacitance = 1e-7;
    let a = 2e-6;
    let b = 1e-6;
    // Choose a fixed f2 interior to the f1 grid, including a zero
    // difference frequency, and repeat the same request on the same circuit.
    let deck = stage
        .deck(&format!("distof1 {a} distof2 {b}"), ".disto lin 2 1k 4k 2")
        .replace(".option", "c1 a 0 100n\n.option");
    let netlist = Parser::new()
        .parse_deck(&parse_deck_text(Path::new("reactive.cir"), &deck))
        .unwrap();
    let config = RunConfig::from_netlist(&netlist).unwrap();
    let request = config.plan_for(&netlist.analyses[0]).unwrap();
    let mut circuit = config.circuit(&netlist).unwrap();
    let first = request.run_plots(&mut circuit, &config.context()).unwrap();
    let second = request.run_plots(&mut circuit, &config.context()).unwrap();
    let admittance = |f: f64| Complex::new(g, std::f64::consts::TAU * f * capacitance);
    let f2 = 2000.;
    let y = Complex::real(-0.5 * b) / admittance(f2);
    let close_complex = |got: Complex, want: Complex| {
        assert!(
            (got - want).magnitude() <= 1e-8 * want.magnitude(),
            "{got} != {want}"
        );
    };
    for (index, f1) in [1000., 2000., 3000., 4000.].into_iter().enumerate() {
        let x = Complex::real(-0.5 * a) / admittance(f1);
        let h2 = Complex::real(-c2) * x * x / admittance(2. * f1);
        let sum = Complex::real(-c2) * x * y / admittance(f1 + f2);
        let difference = Complex::real(-c2) * x * y.conj() / admittance(f1 - f2);
        let third = -(Complex::real(c2)
            * (Complex::real(4.) * x * difference + Complex::real(2.) * y.conj() * h2)
            + Complex::real(3. * c3) * x * x * y.conj())
            / Complex::real(3.)
            / admittance(2. * f1 - f2);
        for (plot, want) in first.iter().zip([
            Complex::real(4.) * sum,
            Complex::real(4.) * difference,
            Complex::real(6.) * third,
        ]) {
            assert_eq!(column(plot, "frequency")[index].re, f1);
            close_complex(column(plot, "v(a)")[index], want);
        }
    }
    for (a, b) in first.iter().zip(second.iter()) {
        assert_eq!(a.point_count(), b.point_count());
        for (a, b) in a.rows().zip(b.rows()) {
            for (a, b) in a.iter().zip(b.iter()) {
                assert_eq!(
                    (a.re.to_bits(), a.im.to_bits()),
                    (b.re.to_bits(), b.im.to_bits())
                );
            }
        }
    }
}
