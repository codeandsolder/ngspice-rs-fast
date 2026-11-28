//! Source-selected Berkeley BSIM1/2 physical and numerical integration.

use model_kernel_bsim::LegacyBsimKernel;
use ngspice_rs::{
    analysis::{AnalysisContext, AnalysisInput, bias::solve_dc, newton::NewtonOptions},
    devices::{
        Circuit, Device, ModelBackend, ModelContext, ModelResolver, noise::NoisePlan,
        prepared_model::PreparedDevice,
    },
    netlist::{Parser, ast::Netlist, source::parse_deck_text},
    primitives::{AnalysisKind, Real},
};
use std::path::Path;

const DECK: &str = include_str!("fixtures/legacy_bsim/dc-ac.cir");
const REFERENCE: &str = include_str!("fixtures/legacy_bsim/dc-ac-forward-equivalent.tsv");
const NATIVE_REFERENCE: &str = include_str!("fixtures/legacy_bsim/dc-ac.tsv");
const MODEL: &str = "tox=.024 vdd=5 vfb=-.9 phi=.75 k1=.75 k2=.02 n0=1.2 rsh=52 js=2e-5 cj=1e-3 cjsw=2e-10 pb=.74 pbsw=.83 mj=.5 mjsw=.33 cgdo=1.5e-10 cgso=2e-10 cgbo=4e-11 kf=2e-25 af=1.2";
const FAMILY1: &str = "mus=510 muz=400 eta=.02 u0=.1 u1=.06";
const FAMILY2: &str = "mus0=510 mu0=400 eta0=.02 ua0=.1 u10=.06 vbb=-3 ai0=.05 bi0=.5";
const GEOMETRY: &str = "l=8.3u w=17.6u m=2 ad=30p as=40p pd=13u ps=17u nrd=2.3 nrs=3.1";

fn parse(text: &str) -> Netlist {
    Parser::new()
        .parse_deck(&parse_deck_text(Path::new("legacy-bsim.cir"), text))
        .unwrap()
}

fn context() -> ModelContext {
    ModelContext::default().with_gmin(1e-15)
}

fn options() -> NewtonOptions {
    NewtonOptions {
        reltol: 1e-9,
        abstol: 1e-13,
        vntol: 1e-9,
        ..NewtonOptions::default()
    }
}

fn device<'a>(circuit: &'a Circuit, name: &str) -> &'a dyn Device {
    circuit
        .devices()
        .iter()
        .find(|d| d.name() == name)
        .map(|d| &**d as &dyn Device)
        .unwrap()
}

fn current(circuit: &Circuit, context: &ModelContext, name: &str) -> Real {
    let solved = solve_dc(circuit, context, &options(), &[], None, None).unwrap();
    let index = circuit
        .devices()
        .iter()
        .position(|d| d.name() == name)
        .unwrap();
    solved
        .values
        .get(circuit.branch_rows(index).unwrap().start)
        .unwrap()
}

fn close(label: &str, actual: Real, expected: Real, rel: Real, abs: Real) {
    assert!(
        (actual - expected).abs() <= rel.mul_add(expected.abs(), abs),
        "{label}: {actual:.17e} vs {expected:.17e}"
    );
}

#[test]
fn distinct_levels_and_hidden_series_rows_match_independent_ngspice() {
    let netlist = parse(DECK);
    let resolver = ModelResolver::new(&netlist.models).unwrap();
    for instance in netlist.devices.iter().filter(|d| d.designator == 'm') {
        let model = resolver.resolve(instance).unwrap().unwrap();
        assert_eq!(
            model.levels().backend,
            if instance.name.starts_with("m4") {
                ModelBackend::Bsim1
            } else {
                ModelBackend::Bsim2
            }
        );
    }
    let mut circuit = Circuit::from_netlist_with_context(&netlist, &context()).unwrap();
    for name in [
        "m4nf", "m4nr", "m4pf", "m4pr", "m5nf", "m5nr", "m5pf", "m5pr",
    ] {
        let d = device(&circuit, name);
        assert!((d as &dyn std::any::Any).is::<PreparedDevice<LegacyBsimKernel>>());
        assert_eq!(d.external_terminals().len(), 4);
        assert_eq!(d.terminals().len(), 6);
        assert_eq!(d.designator(), 'm');
    }
    assert_eq!(
        device(&circuit, "m4nf").prepared_batch_key(),
        device(&circuit, "m4nr").prepared_batch_key()
    );
    assert_ne!(
        device(&circuit, "m4nf").prepared_batch_key(),
        device(&circuit, "m5nf").prepared_batch_key()
    );
    assert_ne!(
        device(&circuit, "m4nf").prepared_batch_key(),
        device(&circuit, "m4pf").prepared_batch_key()
    );
    let dc = solve_dc(&circuit, &context(), &options(), &[], None, None).unwrap();
    let request = AnalysisInput::with_arguments(AnalysisKind::Ac, ["lin", "1", "1meg", "1meg"])
        .resolve()
        .unwrap();
    let plot = request
        .run(
            &mut circuit,
            &AnalysisContext {
                gmin: 1e-15,
                ..AnalysisContext::default()
            },
        )
        .unwrap();
    // Native reverse-mode AC stores the channel drain charge in csgb/cssb/csdb
    // but reads cdgb/cddb/cdsb. Independently swap the device's terminals,
    // asymmetric geometry and overlap coefficients to measure physical AC.
    // The original native reference is retained for direct DC validation.
    let native_dc: std::collections::BTreeMap<_, _> = NATIVE_REFERENCE
        .lines()
        .filter(|line| !line.starts_with('#'))
        .map(|line| {
            let mut columns = line.split('\t');
            (
                columns.next().unwrap(),
                columns.next().unwrap().parse::<Real>().unwrap(),
            )
        })
        .collect();
    let mut failures = Vec::new();
    for row in REFERENCE.lines().filter(|line| !line.starts_with('#')) {
        let [name, dc_ref, re_ref, im_ref] =
            row.split('\t').collect::<Vec<_>>().try_into().unwrap();
        let source = &name[2..name.len() - 1];
        let index = circuit
            .devices()
            .iter()
            .position(|d| d.name() == source)
            .unwrap();
        let actual_dc = dc
            .values
            .get(circuit.branch_rows(index).unwrap().start)
            .unwrap();
        close("direct native DC", actual_dc, native_dc[name], 2e-5, 2e-12);
        let actual_ac = plot.value(name, 0).unwrap();
        for (kind, actual, expected) in [
            ("dc", actual_dc, dc_ref),
            ("ac-re", actual_ac.re, re_ref),
            ("ac-im", actual_ac.im, im_ref),
        ] {
            let expected: Real = expected.parse().unwrap();
            let tolerance = 2e-5_f64.mul_add(expected.abs(), 2e-12);
            if (actual - expected).abs() > tolerance {
                failures.push(format!("{name} {kind}: {actual:.17e} vs {expected:.17e}"));
            }
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

fn deck(level: u8, extra: &str, model_extra: &str) -> String {
    let family = if level == 4 { FAMILY1 } else { FAMILY2 };
    format!(
        "legacy BSIM source binding\nvd d 0 .8\nvg g 0 2 ac 1\nvb b 0 -.2\nm1 d g 0 b mm {GEOMETRY} {extra}\n.model mm nmos(level={level} {MODEL} {family} {model_extra})\n.end\n"
    )
}

#[test]
fn source_setters_preserve_exact_candidate_and_topology_identity() {
    for level in [4, 5] {
        let netlist = parse(&deck(level, "", ""));
        let circuit = Circuit::from_netlist_with_context(&netlist, &context()).unwrap();
        let base = current(&circuit, &context(), "vd");
        let target = circuit.bind_instance_target("m1", "m").unwrap();
        let changed = context()
            .with_instance_override(target.at(4.).unwrap())
            .unwrap();
        close(
            "multiplicity",
            current(&circuit, &changed, "vd"),
            base * 2.,
            1e-10,
            1e-14,
        );
        assert_eq!(
            current(&circuit, &context(), "vd").to_bits(),
            base.to_bits()
        );
        let target = circuit.bind_instance_target("m1", "nrd").unwrap();
        let invalid = context()
            .with_instance_override(target.at(0.).unwrap())
            .unwrap();
        assert!(circuit.prepare_run().validate_context(&invalid).is_err());
        assert_eq!(
            current(&circuit, &context(), "vd").to_bits(),
            base.to_bits()
        );
        let rebuilt =
            Circuit::from_netlist_with_context(&parse(&deck(level, "nrd=0", "")), &context())
                .unwrap();
        assert_eq!(device(&rebuilt, "m1").terminals().len(), 5);
        assert_ne!(
            current(&rebuilt, &context(), "vd").to_bits(),
            base.to_bits()
        );
    }
}

#[test]
fn typed_source_defaults_flags_and_invalid_inputs() {
    for level in [4, 5] {
        let extra = if level == 4 { "xpart" } else { "xpart=1" };
        let circuit = Circuit::from_netlist_with_context(
            &parse(&deck(level, "off ic=.8,2,-.2", extra)),
            &context(),
        )
        .unwrap();
        let mos = device(&circuit, "m1");
        assert!(mos.has_start_settings());
        assert!(current(&circuit, &context(), "vd").is_finite());
        for (instance, model) in [
            ("l=0", ""),
            ("nrd=-1", ""),
            ("temp=40", ""),
            ("", "unknown=1"),
        ] {
            assert!(
                Circuit::from_netlist_with_context(
                    &parse(&deck(level, instance, model)),
                    &context()
                )
                .is_err()
            );
        }
    }
    // A model declared in a subcircuit is not shared with an unrelated
    // declaration carrying the same visible name.
    let mut text = String::from("legacy BSIM scopes\n");
    for (name, kp) in [("one", 510), ("two", 610)] {
        text.push_str(&format!(".subckt {name} d g b\nm1 d g 0 b mm {GEOMETRY}\n.model mm nmos(level=4 {MODEL} {FAMILY1} mus={kp})\n.ends\n"));
    }
    text.push_str("x1 d g b one\nx2 d g b two\nvd d 0 .8\nvg g 0 2\nvb b 0 -.2\n.end\n");
    let circuit = Circuit::from_netlist_with_context(&parse(&text), &context()).unwrap();
    let prepared: Vec<_> = circuit
        .devices()
        .iter()
        .filter(|d| d.designator() == 'm')
        .map(|d| d.prepared_batch_key())
        .collect();
    assert_eq!(prepared.len(), 2);
    assert_ne!(prepared[0], prepared[1]);
}

#[test]
fn physical_noise_generators_and_transient_charges_use_prepared_models() {
    for level in [4, 5] {
        let source = deck(level, "", "")
            .replace("vg g 0 2 ac 1", "vg g 0 PWL(0 1.5 1u 1.5 1.1u 2 3u 2) ac 1");
        let mut circuit = Circuit::from_netlist_with_context(&parse(&source), &context()).unwrap();
        let dc = solve_dc(&circuit, &context(), &options(), &[], None, None).unwrap();
        {
            let run = circuit.prepare_run();
            let noise =
                NoisePlan::bind(&run, &context().with_frequency(1000.), &dc.values, None).unwrap();
            assert_eq!(noise.source_count(), 4);
        }
        let request = AnalysisInput::with_arguments(
            AnalysisKind::Transient,
            [
                "20n",
                "3u",
                "0",
                "20n",
                "rtol=1e-8",
                "vntol=1e-10",
                "abstol=1e-14",
            ],
        )
        .resolve()
        .unwrap();
        let plot = request
            .run(
                &mut circuit,
                &AnalysisContext {
                    gmin: 1e-15,
                    reltol: 1e-8,
                    ..AnalysisContext::default()
                },
            )
            .unwrap();
        assert!(plot.point_count() > 20);
        for point in 0..plot.point_count() {
            assert!(plot.value("i(vd)", point).unwrap().re.is_finite());
            assert!(plot.value("i(vg)", point).unwrap().re.is_finite());
        }
        let first = plot.value("i(vd)", 0).unwrap().re;
        let last = plot.value("i(vd)", plot.point_count() - 1).unwrap().re;
        assert!(last.abs() > first.abs());
        let static_circuit =
            Circuit::from_netlist_with_context(&parse(&deck(level, "", "")), &context()).unwrap();
        close(
            "transient settled drain",
            last,
            current(&static_circuit, &context(), "vd"),
            2e-6,
            1e-12,
        );
    }
}

#[test]
fn transient_currents_match_independent_forward_equivalent() {
    let source = include_str!("fixtures/legacy_bsim/tran.cir");
    let reference = include_str!("fixtures/legacy_bsim/tran-forward-equivalent.tsv");
    let mut circuit = Circuit::from_netlist_with_context(&parse(source), &context()).unwrap();
    let request = AnalysisInput::with_arguments(
        AnalysisKind::Transient,
        [
            "0.5n",
            "3u",
            "0",
            "0.5n",
            "rtol=1e-9",
            "vntol=1e-10",
            "abstol=1e-14",
        ],
    )
    .resolve()
    .unwrap();
    let plot = request
        .run(
            &mut circuit,
            &AnalysisContext {
                gmin: 1e-15,
                reltol: 1e-9,
                ..AnalysisContext::default()
            },
        )
        .unwrap();
    let times: Vec<_> = (0..plot.point_count())
        .map(|index| plot.value("time", index).unwrap().re)
        .collect();
    let mut failures = Vec::new();
    for row in reference.lines().filter(|line| !line.starts_with('#')) {
        let [time, name, expected] = row.split('\t').collect::<Vec<_>>().try_into().unwrap();
        let time: Real = time.parse().unwrap();
        let expected: Real = expected.parse().unwrap();
        let right = times.partition_point(|value| *value < time);
        let left = right - 1;
        let fraction = (time - times[left]) / (times[right] - times[left]);
        let a = plot.value(name, left).unwrap().re;
        let b = plot.value(name, right).unwrap().re;
        let actual = fraction.mul_add(b - a, a);
        // The two solvers choose different integration histories. Check
        // physical terminal currents away from slope discontinuities.
        if (actual - expected).abs() > 5e-4_f64.mul_add(expected.abs(), 2e-10) {
            failures.push(format!(
                "{time:.9e} {name}: {actual:.17e} vs {expected:.17e}"
            ));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
fn physical_noise_matches_independent_native_dc_measurements() {
    use ngspice_rs::analysis::RunConfig;
    let reference = include_str!("fixtures/legacy_bsim/noise-dc.tsv");
    let model = MODEL.replace("rsh=52", "rsh=0");
    let temperature = 300.15;
    let thermal = 4. * 1.380_648_52e-23 * temperature;
    let cox = 3.453e-13_f64 / (0.024e-4);
    let area = 8.3e-6 * 17.6e-6;
    for row in reference.lines().filter(|line| !line.starts_with('#')) {
        let [level, polarity, direction, id, gm] =
            row.split('\t').collect::<Vec<_>>().try_into().unwrap();
        let id = id.parse::<Real>().unwrap();
        let gm = gm.parse::<Real>().unwrap();
        let sign = if polarity == "n" { 1. } else { -1. };
        let (vd, vg, vb) = if direction == "f" {
            (0.8, 2., -0.2)
        } else {
            (-0.6, 1.5, -1.)
        };
        let family = if level == "4" { FAMILY1 } else { FAMILY2 };
        // A unit transresistance CCVS observes the ideal drain voltage
        // source current: each physical drain-source generator sees
        // exactly 1 V/A transfer. No native noise state indices are used.
        let source = format!(
            "physical legacy BSIM noise\nvd d 0 {}\nvg g 0 {} ac 1\nvb b 0 {}\nm1 d g 0 b mm {GEOMETRY}\n.model mm {polarity}mos(level={level} {model} {family})\nh1 out 0 vd 1\n.options gmin=1e-15 reltol=1e-9 vntol=1e-10 abstol=1e-14\n.noise v(out) vg lin 3 1k 3k 1\n.end\n",
            sign * vd,
            sign * vg,
            sign * vb
        );
        let netlist = parse(&source);
        let config = RunConfig::from_netlist(&netlist).unwrap();
        let plan = config.plan_for(&netlist.analyses[0]).unwrap();
        let mut circuit = config.circuit(&netlist).unwrap();
        let plots = plan.run_plots(&mut circuit, &config.context()).unwrap();
        let spectrum = &plots[0];
        let channel = thermal * (2. / 3.) * gm.abs();
        for index in 0..3 {
            let frequency = spectrum.value("frequency", index).unwrap().re;
            let flicker = 2e-25 * 2. * (id.abs() / 2.).powf(1.2) / (frequency * area * cox.powi(2));
            let out = spectrum.value("onoise_spectrum", index).unwrap().re;
            close(
                "noise output PSD",
                out.powi(2),
                channel + flicker,
                3e-6,
                1e-35,
            );
        }
    }
}

#[test]
#[ignore = "opt-in prepared BSIM physical-load benchmark; timing is not an assertion"]
fn legacy_bsim_tile_load_throughput() {
    use ngspice_rs::devices::{AnalysisMode, LoadRequest};
    use ngspice_rs::maths::Vector;
    use std::{fmt::Write, hint::black_box, time::Instant};

    const LANES: usize = 128;
    const LOADS: u32 = 200;
    for (level, family) in [(4, FAMILY1), (5, FAMILY2)] {
        let build = |shared: bool| {
            let mut deck = String::from("legacy BSIM load benchmark\n");
            for lane in 0..LANES {
                let model = if shared {
                    "mm".to_owned()
                } else {
                    format!("mm{lane}")
                };
                if !shared || lane == 0 {
                    writeln!(deck, ".model {model} nmos(level={level} {MODEL} {family})").unwrap();
                }
                writeln!(deck, "vd{lane} d{lane} 0 .8\nvg{lane} g{lane} 0 2\nvb{lane} b{lane} 0 -.2\nm{lane} d{lane} g{lane} 0 b{lane} {model} {GEOMETRY}").unwrap();
            }
            deck.push_str(".end\n");
            let netlist = parse(&deck);
            let start = Instant::now();
            let circuit = Circuit::from_netlist_with_context(&netlist, &context()).unwrap();
            eprintln!(
                "level={level} shared={shared} binding={:?}",
                start.elapsed()
            );
            circuit
        };
        let tiled = build(true);
        let scalar = build(false);
        let measure = |circuit: &Circuit| {
            let context = context();
            let solution = solve_dc(circuit, &context, &options(), &[], None, None)
                .unwrap()
                .values;
            let run = circuit.prepare_run();
            let history = circuit.state_history().unwrap();
            let mut trial = history.trial();
            let mut matrix = circuit.new_matrix().unwrap();
            let mut rhs = Vector::zeros(circuit.unknown_count());
            let request = LoadRequest {
                mode: AnalysisMode::OperatingPoint,
                solution: &solution,
                model_context: &context,
                integration: None,
                history: &history,
                forcing: None,
            };
            run.load(&request, &mut matrix, &mut rhs, &mut trial)
                .unwrap();
            let start = Instant::now();
            for _ in 0..LOADS {
                matrix.clear();
                rhs.clear();
                run.load(&request, &mut matrix, &mut rhs, &mut trial)
                    .unwrap();
                black_box((&matrix, &rhs));
            }
            let elapsed = start.elapsed().as_secs_f64() * 1e6 / f64::from(LOADS);
            let snapshot: Vec<_> = matrix
                .bound_values()
                .unwrap()
                .1
                .iter()
                .chain(rhs.as_slice())
                .map(|value| value.to_bits())
                .collect();
            (elapsed, snapshot)
        };
        let mut tiles = Vec::new();
        let mut scalars = Vec::new();
        for round in 0..8 {
            let ((a, x), (b, y)) = if round % 2 == 0 {
                (measure(&tiled), measure(&scalar))
            } else {
                let b = measure(&scalar);
                (measure(&tiled), b)
            };
            assert_eq!(x, y);
            if round > 0 {
                tiles.push(a);
                scalars.push(b);
            }
        }
        tiles.sort_by(f64::total_cmp);
        scalars.sort_by(f64::total_cmp);
        eprintln!(
            "BSIM level {level}: {LANES} lanes; tiled {:.3} us/load, scalar {:.3} us/load",
            tiles[3], scalars[3]
        );
    }
}
