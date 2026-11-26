//! Static statistical realization is scoped, deterministic, and independent
//! of runtime Newton and candidate evaluation order.
use std::path::Path;

use ngspice_rs::analysis::config::RunConfig;
use ngspice_rs::analysis::optimization::parameters::CircuitParameterPlan;
use ngspice_rs::analysis::{AnalysisContext, AnalysisInput, Plot};
use ngspice_rs::devices::Circuit;
use ngspice_rs::netlist::{
    Parser,
    ast::Netlist,
    eval::{EvalBudget, ParamScope},
    source::parse_deck_text,
};
use ngspice_rs::primitives::{AnalysisKind, SpiceResult};

fn deck(body: &str) -> Netlist {
    Parser::new()
        .parse_deck(&parse_deck_text(
            Path::new("statistical.cir"),
            &format!("statistics\n{body}\n.end\n"),
        ))
        .unwrap()
}

fn op(circuit: &mut Circuit) -> SpiceResult<Plot> {
    AnalysisInput::with_arguments(AnalysisKind::OperatingPoint, std::iter::empty::<&str>())
        .resolve()?
        .run(circuit, &AnalysisContext::default())
}

#[test]
fn same_seed_same_draw_different_seed_different_draw() -> SpiceResult<()> {
    let source = |seed| {
        deck(&format!(
            ".options seed={seed}\n.param a=agauss(10,2,3) b=gauss(10,.2,3) c=aunif(10,2) d=unif(10,.2) e1=limit(10,2)"
        ))
    };
    let first = ParamScope::for_netlist(&source(41))?;
    let second = ParamScope::for_netlist(&source(41))?;
    let other = ParamScope::for_netlist(&source(42))?;
    for name in ["a", "b", "c", "d", "e1"] {
        assert_eq!(first.get(name), second.get(name));
    }
    for name in ["a", "b", "c", "d"] {
        assert_ne!(first.get(name), other.get(name));
    }
    for name in ["c", "d"] {
        assert!((8. ..=12.).contains(&first.get(name).unwrap()));
    }
    assert!([8., 12.].contains(&first.get("e1").unwrap()));
    Ok(())
}

#[test]
fn statistical_calls_in_macros_have_independent_occurrences_and_static_arguments() -> SpiceResult<()>
{
    let source = deck(
        ".options seed=13\n.func sample(a) {aunif(a,0.1)}\n.param x={sample(1)-sample(1)}\n.param a=agauss(2,0,3) b=gauss(4,0.1,0)\nb1 out 0 v=sample(1+2)\nb2 other 0 v=sample(1+2)",
    );
    let scope = ParamScope::for_netlist(&source)?;
    assert_ne!(scope.get("x"), Some(0.));
    assert_eq!(scope.get("a"), Some(2.));
    assert_eq!(scope.get("b"), Some(4.));
    let mut circuit = RunConfig::from_netlist(&source)?.circuit(&source)?;
    let plot = op(&mut circuit)?;
    let out = plot.value("v(out)", 0).unwrap().re;
    let other = plot.value("v(other)", 0).unwrap().re;
    assert!((2.9..=3.1).contains(&out) && (2.9..=3.1).contains(&other));
    assert_ne!(out, other);
    Ok(())
}

#[test]
fn subcircuit_instances_are_independent_but_shared_parameters_remain_correlated() -> SpiceResult<()>
{
    let source = deck(
        ".options seed=72\n.param common=aunif(1,0.1)\n.subckt cell a b c\n.param local=aunif(1,0.1)\nb1 a 0 v=local\nb2 b 0 v=local\nb3 c 0 v=common\n.ends\nx1 a1 b1 c1 cell\nx2 a2 b2 c2 cell",
    );
    let mut circuit = RunConfig::from_netlist(&source)?.circuit(&source)?;
    let plot = op(&mut circuit)?;
    let get = |name| plot.value(name, 0).unwrap().re;
    assert_eq!(get("v(a1)"), get("v(b1)"));
    assert_eq!(get("v(a2)"), get("v(b2)"));
    assert_ne!(get("v(a1)"), get("v(a2)"));
    assert_eq!(get("v(c1)"), get("v(c2)"));
    let mut rebuilt = RunConfig::from_netlist(&source)?.circuit(&source)?;
    let second = op(&mut rebuilt)?;
    for name in ["v(a1)", "v(a2)", "v(c1)"] {
        assert_eq!(get(name), second.value(name, 0).unwrap().re);
    }
    Ok(())
}

#[test]
fn candidate_aba_retains_the_normalized_realization_on_dependent_sites() -> SpiceResult<()> {
    let source = deck(
        ".options seed=17\n.param nominal=1k r={gauss(nominal,.1,3)}\nv1 in 0 1\nr1 in 0 {r}\nr2 in 0 {gauss(nominal,.1,3)}",
    );
    let plan = CircuitParameterPlan::new(source, &["nominal"])?;
    let mut session = plan.session()?;
    let run = |session: &mut ngspice_rs::analysis::optimization::parameters::CircuitCandidate<
        '_,
    >|
     -> SpiceResult<f64> {
        let context = session.analysis_context();
        Ok(
            AnalysisInput::with_arguments(AnalysisKind::OperatingPoint, std::iter::empty::<&str>())
                .resolve()?
                .run(session.circuit_mut(), &context)?
                .value("i(v1)", 0)
                .unwrap()
                .re,
        )
    };
    let a = run(&mut session)?;
    session.update(&[2000.], &mut EvalBudget::default())?;
    let b = run(&mut session)?;
    assert!(2.0_f64.mul_add(b, -a).abs() < 1e-12, "{a} {b}");
    session.update(&[1000.], &mut EvalBudget::default())?;
    assert_eq!(a, run(&mut session)?);
    Ok(())
}

#[test]
fn statistical_values_do_not_change_with_transient_time_or_newton_loads() -> SpiceResult<()> {
    let source = deck(".options seed=81\nb1 out 0 v=agauss(2,0.2,3)+1000*time\nr1 out 0 1k");
    let mut circuit = RunConfig::from_netlist(&source)?.circuit(&source)?;
    let bias = op(&mut circuit)?.value("v(out)", 0).unwrap().re;
    let plot = AnalysisInput::with_arguments(AnalysisKind::Transient, ["1u", "1m"])
        .resolve()?
        .run(&mut circuit, &AnalysisContext::default())?;
    for index in 0..plot.point_count() {
        let t = plot.value("time", index).unwrap().re;
        let actual = plot.value("v(out)", index).unwrap().re;
        assert!(1000.0_f64.mul_add(-t, actual - bias).abs() < 1e-10);
    }
    Ok(())
}

#[test]
fn behavioural_candidates_keep_draws_across_reelaboration_and_hierarchy() -> SpiceResult<()> {
    let source = deck(
        ".options seed=55\n.param nominal=10\n.subckt cell a b params: n=nominal\n.param local=gauss(n,.1,3)\nb1 a 0 v=local\nb2 b 0 v=gauss(n,.1,3)\n.ends\nx1 a1 b1 cell\nx2 a2 b2 cell",
    );
    let plan = CircuitParameterPlan::new(source, &["nominal"])?;
    let mut candidate = plan.session()?;
    let run = |candidate: &mut ngspice_rs::analysis::optimization::parameters::CircuitCandidate<
        '_,
    >|
     -> SpiceResult<[f64; 4]> {
        let context = candidate.analysis_context();
        let plot =
            AnalysisInput::with_arguments(AnalysisKind::OperatingPoint, std::iter::empty::<&str>())
                .resolve()?
                .run(candidate.circuit_mut(), &context)?;
        Ok(["v(a1)", "v(b1)", "v(a2)", "v(b2)"].map(|name| plot.value(name, 0).unwrap().re))
    };
    let first = run(&mut candidate)?;
    assert_ne!(first[0], first[2]);
    assert_ne!(first[1], first[3]);
    candidate.update(&[20.], &mut EvalBudget::default())?;
    let second = run(&mut candidate)?;
    for (a, b) in first.into_iter().zip(second) {
        assert!(2.0_f64.mul_add(-a, b).abs() < 1e-12, "{a} {b}");
    }
    candidate.update(&[10.], &mut EvalBudget::default())?;
    assert_eq!(first, run(&mut candidate)?);
    Ok(())
}

#[test]
fn prepared_parameters_and_device_fields_are_occurrence_bound() -> SpiceResult<()> {
    let body = ".param n=1000 s={simetrix_stat_unif(4294967295,4294967295,7,1,0,0,.2)}\n        .func draw(t) {simetrix_stat_unif(4294967295,4294967295,7,1,1,0,t)}\n        .param p={draw(.2)} q={draw(.2)-draw(.2)}\n        v1 in 0 1\nr1 in 0 {n*s}\nr2 in 0 {n*s}\nr3 in 0 {n*draw(.2)}\nr4 in 0 {n*draw(.2)}";
    let source = deck(body);
    let result = ngspice_rs::netlist::elaborate::literalize(&source)?;
    assert_ne!(result.scope.get("q"), Some(0.));
    let values = |result: &ngspice_rs::netlist::elaborate::ElaboratedNetlist| {
        result
            .netlist
            .devices
            .iter()
            .filter(|device| device.name.starts_with('r'))
            .map(|device| (device.name.clone(), device.parameters[0].value.clone()))
            .collect::<std::collections::BTreeMap<_, _>>()
    };
    let first = values(&result);
    assert_eq!(first["r1"], first["r2"]);
    assert_ne!(first["r3"], first["r4"]);
    let reordered = deck(&body.replace(
        "r3 in 0 {n*draw(.2)}\nr4 in 0 {n*draw(.2)}",
        "r4 in 0 {n*draw(.2)}\nr3 in 0 {n*draw(.2)}",
    ));
    assert_eq!(
        first,
        values(&ngspice_rs::netlist::elaborate::literalize(&reordered)?)
    );
    let plan = CircuitParameterPlan::new(source, &["n"])?;
    let mut session = plan.session()?;
    let a = op(session.circuit_mut())?.value("i(v1)", 0).unwrap().re;
    session.update(&[2000.], &mut EvalBudget::default())?;
    let context = session.analysis_context();
    let b = AnalysisInput::with_arguments(AnalysisKind::OperatingPoint, std::iter::empty::<&str>())
        .resolve()?
        .run(session.circuit_mut(), &context)?
        .value("i(v1)", 0)
        .unwrap()
        .re;
    assert!(2.0_f64.mul_add(b, -a).abs() < 1e-12, "{a} {b}");
    session.update(&[1000.], &mut EvalBudget::default())?;
    assert_eq!(a, op(session.circuit_mut())?.value("i(v1)", 0).unwrap().re);
    Ok(())
}

#[test]
fn prepared_model_draws_use_the_defining_scope_and_physical_occurrence() -> SpiceResult<()> {
    let source = deck(
        ".param tolerance=.2\n        .model shared d(is={1e-14*simetrix_stat_unifl(9,1,7,1,0,0,tolerance)} n={simetrix_stat_unif(9,1,7,1,1,0,tolerance)})\n        .subckt cell a b c\n.param tolerance=.4\n        .model local d(is={1e-14*simetrix_stat_unifl(9,1,7,1,0,0,tolerance)})\n        d1 a 0 shared\nd2 b 0 local\nd3 c 0 local\n.ends\n        x1 a1 b1 c1 cell\nx2 a2 b2 c2 cell\n        v1 a1 0 .5\nv2 b1 0 .5\nv3 c1 0 .5\nv4 a2 0 .5\nv5 b2 0 .5\nv6 c2 0 .5",
    );
    let result = ngspice_rs::netlist::elaborate::literalize(&source)?;
    let expanded = ngspice_rs::devices::subckt::expand_subcircuits(
        &result.netlist,
        &result.scope,
        ngspice_rs::devices::subckt::SubcircuitLimits::default(),
    )?;
    let parameter = |device: &str, field: &str| {
        let instance = expanded.devices.iter().find(|d| d.name == device).unwrap();
        let model = expanded
            .models
            .iter()
            .find(|m| Some(&m.name) == instance.model.as_ref())
            .unwrap();
        model
            .parameters
            .iter()
            .find(|p| p.name == field)
            .unwrap()
            .value
            .clone()
    };
    assert_eq!(parameter("d.x1.d1", "is"), parameter("d.x2.d1", "is"));
    assert_ne!(parameter("d.x1.d1", "n"), parameter("d.x2.d1", "n"));
    assert_eq!(parameter("d.x1.d2", "is"), parameter("d.x1.d3", "is"));
    assert_eq!(parameter("d.x2.d2", "is"), parameter("d.x2.d3", "is"));
    assert_ne!(parameter("d.x1.d2", "is"), parameter("d.x2.d2", "is"));
    let mut circuit = RunConfig::from_netlist(&source)?.circuit(&source)?;
    let plot = op(&mut circuit)?;
    assert_eq!(plot.value("i(v2)", 0), plot.value("i(v3)", 0));
    assert_ne!(plot.value("i(v2)", 0), plot.value("i(v5)", 0));
    Ok(())
}

#[test]
fn prepared_markers_reject_missing_lot_owner_and_unsupported_law() {
    for expression in [
        "simetrix_stat_unifl(9,1,7,1,0,0,.2)",
        "simetrix_stat_gausse(9,1,7,1,0,0,.2)",
        "simetrix_stat_unif(4294967296,1,7,1,0,0,.2)",
    ] {
        let source = deck(&format!(".param p={{{expression}}}"));
        assert!(ParamScope::for_netlist(&source).is_err(), "{expression}");
    }
}

#[test]
fn prepared_behavioural_draws_are_static_and_macro_arguments_are_shared() -> SpiceResult<()> {
    let source = deck(
        ".param tol=.2\n        .func draw(t) {simetrix_stat_unif(4294967295,4294967295,7,1,0,0,t)}\n        .func repeated(x) {x-x}\n        b1 a 0 v=draw(tol)+1000*time\n        b2 b 0 v=draw(tol)\n        b3 c 0 v=repeated(draw(tol))\n        b4 d 0 v=draw(tol)-draw(tol)\n        r1 a 0 1k\nr2 b 0 1k\nr3 c 0 1k\nr4 d 0 1k",
    );
    let mut circuit = RunConfig::from_netlist(&source)?.circuit(&source)?;
    let bias = op(&mut circuit)?;
    let a = bias.value("v(a)", 0).unwrap().re;
    assert_ne!(a, bias.value("v(b)", 0).unwrap().re);
    assert_eq!(0., bias.value("v(c)", 0).unwrap().re);
    assert_ne!(0., bias.value("v(d)", 0).unwrap().re);
    let plot = AnalysisInput::with_arguments(AnalysisKind::Transient, ["1u", "1m"])
        .resolve()?
        .run(&mut circuit, &AnalysisContext::default())?;
    for index in 0..plot.point_count() {
        let t = plot.value("time", index).unwrap().re;
        let actual = plot.value("v(a)", index).unwrap().re;
        assert!(1000.0_f64.mul_add(-t, actual - a).abs() < 1e-10);
    }
    Ok(())
}

#[test]
fn prepared_nested_models_preserve_the_actual_ancestor_owner() -> SpiceResult<()> {
    let source = deck(
        ".param t=.1\n        .func draw(t) {simetrix_stat_unifl(9,1,7,1,0,0,t)}\n        .model global d(is={1e-14*draw(t)})\n        .subckt inner a b\n.param t=.9\nd1 a 0 global\nd2 b 0 local\n.ends\n        .subckt outer a b c d\n.param t=.3\n.model local d(is={1e-14*draw(t)})\n        x1 a b inner\nx2 c d inner\n.ends\n        x1 a1 b1 c1 d1 outer\nx2 a2 b2 c2 d2 outer",
    );
    let literal = ngspice_rs::netlist::elaborate::literalize(&source)?;
    let expanded = ngspice_rs::devices::subckt::expand_subcircuits(
        &literal.netlist,
        &literal.scope,
        ngspice_rs::devices::subckt::SubcircuitLimits::default(),
    )?;
    let value = |name: &str| {
        let device = expanded.devices.iter().find(|d| d.name == name).unwrap();
        let model = expanded
            .models
            .iter()
            .find(|m| Some(&m.name) == device.model.as_ref())
            .unwrap();
        model
            .parameters
            .iter()
            .find(|p| p.name == "is")
            .unwrap()
            .value
            .parse::<f64>()
            .unwrap()
    };
    let global = value("d.x1.x1.d1");
    assert!((0.9e-14..=1.1e-14).contains(&global));
    assert_eq!(global, value("d.x1.x2.d1"));
    assert_eq!(global, value("d.x2.x1.d1"));
    assert_eq!(global, value("d.x2.x2.d1"));
    let local = value("d.x1.x1.d2");
    assert!((0.7e-14..=1.3e-14).contains(&local));
    assert_eq!(local, value("d.x1.x2.d2"));
    assert_ne!(local, value("d.x2.x1.d2"));
    assert_eq!(value("d.x2.x1.d2"), value("d.x2.x2.d2"));
    Ok(())
}

#[test]
fn prepared_full_width_words_affect_the_realization() -> SpiceResult<()> {
    let evaluate = |seed_high: u32, sample_high: u32| -> SpiceResult<f64> {
        let source = deck(&format!(
            ".param p={{simetrix_stat_unif(4294967295,{seed_high},4294967295,{sample_high},4294967295,4294967295,.2)}}"
        ));
        Ok(ParamScope::for_netlist(&source)?.get("p").unwrap())
    };
    let a = evaluate(u32::MAX, u32::MAX)?;
    assert_eq!(a, evaluate(u32::MAX, u32::MAX)?);
    assert_ne!(a, evaluate(u32::MAX - 1, u32::MAX)?);
    assert_ne!(a, evaluate(u32::MAX, u32::MAX - 1)?);
    Ok(())
}

#[test]
fn prepared_model_candidate_rebuild_and_parallel_sessions_match_fresh_elaboration()
-> SpiceResult<()> {
    let source = |nominal: f64| {
        deck(&format!(
            ".param nominal={nominal}\n.model varied d(is={{nominal*simetrix_stat_unif(9,1,7,1,0,0,.2)}})\n        v1 a 0 .5\nv2 b 0 .5\nd1 a 0 varied\nd2 b 0 varied"
        ))
    };
    let first = source(1e-14);
    let plan = CircuitParameterPlan::new(first, &["nominal"])?;
    let current = |nominal: f64| -> SpiceResult<(f64, f64)> {
        let independent = source(nominal);
        let mut circuit = RunConfig::from_netlist(&independent)?.circuit(&independent)?;
        let plot = op(&mut circuit)?;
        Ok((
            plot.value("i(v1)", 0).unwrap().re,
            plot.value("i(v2)", 0).unwrap().re,
        ))
    };
    std::thread::scope(|threads| -> SpiceResult<()> {
        let workers = [1e-14, 2e-14, 3e-14, 1e-14].map(|nominal| {
            let plan = &plan;
            let current = &current;
            threads.spawn(move || -> SpiceResult<()> {
                let mut session = plan.session()?;
                session.update(&[nominal], &mut EvalBudget::default())?;
                let plot = op(session.circuit_mut())?;
                let (a, b) = current(nominal)?;
                assert_eq!(
                    (a, b),
                    (
                        plot.value("i(v1)", 0).unwrap().re,
                        plot.value("i(v2)", 0).unwrap().re
                    )
                );
                assert_ne!(a, b);
                Ok(())
            })
        });
        for worker in workers {
            worker.join().unwrap()?;
        }
        Ok(())
    })
}

#[test]
fn prepared_resistor_draw_is_identical_across_op_ac_and_transient() -> SpiceResult<()> {
    let source = deck(
        "v1 in 0 dc 1 ac 1\n        r1 in out {1000*simetrix_stat_unif(9,1,7,1,0,0,.2)}\nr2 out 0 1k",
    );
    let mut circuit = RunConfig::from_netlist(&source)?.circuit(&source)?;
    let bias = op(&mut circuit)?.value("v(out)", 0).unwrap().re;
    let ac = AnalysisInput::with_arguments(AnalysisKind::Ac, ["lin", "2", "1", "10"])
        .resolve()?
        .run(&mut circuit, &AnalysisContext::default())?;
    for index in 0..ac.point_count() {
        let value = ac.value("v(out)", index).unwrap();
        assert!((value.re - bias).abs() < 1e-12);
        assert!(value.im.abs() < 1e-12);
    }
    let tran = AnalysisInput::with_arguments(AnalysisKind::Transient, ["1u", "1m"])
        .resolve()?
        .run(&mut circuit, &AnalysisContext::default())?;
    for index in 0..tran.point_count() {
        assert!((tran.value("v(out)", index).unwrap().re - bias).abs() < 1e-12);
    }
    assert_eq!(bias, op(&mut circuit)?.value("v(out)", 0).unwrap().re);
    Ok(())
}

#[test]
#[ignore = "requires NGSPICE_BIN; C consumes only the realized numeric deck"]
fn realized_prepared_values_match_c_operating_point() -> SpiceResult<()> {
    use ngspice_conformance::ngspice::{Ngspice, Scratch};
    let source = deck(
        ".model varied d(is={1e-14*simetrix_stat_unif(9,1,7,1,0,0,.2)})\n        v1 in 0 1\nr1 in out {1000*simetrix_stat_unif(9,1,7,1,0,0,.2)}\nr2 out 0 1k\n        vd diode 0 .5\nd1 diode 0 varied",
    );
    let literal = ngspice_rs::netlist::elaborate::literalize(&source)?;
    let expanded = ngspice_rs::devices::subckt::expand_subcircuits(
        &literal.netlist,
        &literal.scope,
        ngspice_rs::devices::subckt::SubcircuitLimits::default(),
    )?;
    let resistance = &expanded
        .devices
        .iter()
        .find(|d| d.name == "r1")
        .unwrap()
        .parameters[0]
        .value;
    let saturation = &expanded.models[0]
        .parameters
        .iter()
        .find(|p| p.name == "is")
        .unwrap()
        .value;
    let text = format!(
        "realized statistical fixture\nv1 in 0 1\nr1 in out {resistance}\nr2 out 0 1k\n        vd diode 0 .5\nd1 diode 0 varied\n.model varied d(is={saturation})\n        .control\nset numdgt=17\nop\nprint v(out) i(vd)\nquit\n.endc\n.end\n"
    );
    let scratch = Scratch::new("prepared-statistics").unwrap();
    std::fs::write(scratch.path().join("probe.cir"), text).unwrap();
    let oracle = Ngspice::pinned().unwrap();
    let output = oracle.run(scratch.path(), &["-b", "probe.cir"]).unwrap();
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(
        output.status.success(),
        "{stdout}\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let c = |name: &str| {
        stdout
            .lines()
            .find_map(|line| {
                let (key, value) = line.split_once('=')?;
                (key.trim() == name).then(|| value.trim().parse::<f64>().unwrap())
            })
            .unwrap()
    };
    let mut circuit = RunConfig::from_netlist(&source)?.circuit(&source)?;
    let plot = op(&mut circuit)?;
    for (name, relative, absolute) in [("v(out)", 1e-12, 1e-14), ("i(vd)", 1e-6, 1e-12)] {
        let want = c(name);
        let got = plot.value(name, 0).unwrap().re;
        assert!(
            (want - got).abs() <= f64::mul_add(relative, want.abs(), absolute),
            "{name}: C {want} Rust {got}"
        );
    }
    Ok(())
}

#[test]
fn prepared_model_bins_preserve_mixed_geometry_selection() -> SpiceResult<()> {
    let source = deck(
        "vd d 0 1\nvg g 0 1\nm1 d g 0 0 nch w=1u l=.7u\nm2 d g 0 0 nch w=1u l=2u\n.model nch.1 nmos level=8 lmin=.5u lmax=1u wmin=.5u wmax=5u vth0={.3*simetrix_stat_unif(9,1,7,1,0,0,.2)}\n.model nch.2 nmos level=8 lmin=1u lmax=5u wmin=.5u wmax=5u vth0=.5",
    );
    let circuit = RunConfig::from_netlist(&source)?.circuit(&source)?;
    let first = circuit.device("m1").unwrap().model_name().unwrap();
    let second = circuit.device("m2").unwrap().model_name().unwrap();
    assert!(first.ends_with(".1"));
    assert!(second.ends_with(".2"));
    assert_ne!(
        first.rsplit_once('.').unwrap().0,
        second.rsplit_once('.').unwrap().0
    );
    Ok(())
}

#[test]
fn prepared_runtime_passive_macros_retain_occurrence_paths() -> SpiceResult<()> {
    let source = deck(
        ".func law(t) {1000*simetrix_stat_unif(9,1,7,1,0,0,t)+time}\nv1 a 0 1\nv2 b 0 1\nr1 a 0 {law(.2)}\nr2 b 0 {law(.2)}",
    );
    let mut circuit = RunConfig::from_netlist(&source)?.circuit(&source)?;
    let bias = op(&mut circuit)?;
    let first = bias.value("i(v1)", 0).unwrap().re;
    let second = bias.value("i(v2)", 0).unwrap().re;
    assert!((-1. / 800. ..=-1. / 1200.).contains(&first));
    assert!((-1. / 800. ..=-1. / 1200.).contains(&second));
    assert_ne!(first, second);
    Ok(())
}
