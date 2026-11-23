//! #120: producer-side objective projections preserve the original numerical
//! solution, history and accepted-point grid while skipping device asks.
use std::path::{Path, PathBuf};

use ngspice_rs::analysis::optimization::objectives::{
    ObjectiveRequests, Observation, ObservationReduction, ObservationValue, PlotObjectivePlan,
    ScenarioPlan,
};
use ngspice_rs::analysis::{Plot, RunConfig, Variable};
use ngspice_rs::netlist::Parser;
use ngspice_rs::netlist::ast::{Netlist, RequestedVector, VectorRequest};
use ngspice_rs::netlist::eval::ParamScope;
use ngspice_rs::netlist::source::parse_deck_text;
use ngspice_rs::primitives::{AnalysisKind, SourceLoc};

fn deck(body: &str) -> Netlist {
    Parser::new()
        .parse_deck(&parse_deck_text(
            Path::new("observation-producer.cir"),
            &format!("Objective producer\n{body}\n.end\n"),
        ))
        .unwrap()
}

fn objective(name: &str, reduction: ObservationReduction) -> Observation {
    Observation {
        name: name.into(),
        vector: VectorRequest {
            vector: RequestedVector::Named {
                name: name.into(),
                component: None,
            },
            location: SourceLoc::new(PathBuf::from("observation-producer.cir"), 2, 1),
        },
        reduction,
    }
}

fn full_and_selected(body: &str, quantities: &[&str], wanted: &[&str]) {
    let netlist = deck(body);
    let config = RunConfig::from_netlist(&netlist).unwrap();
    let context = config.context();
    let scope = ParamScope::for_netlist(&netlist).unwrap();
    let analyses: Vec<_> = netlist
        .analyses
        .iter()
        .map(|card| config.plan_for(card).unwrap())
        .collect();
    assert_eq!(analyses.len(), 1);
    let analysis = &analyses[0];
    let mut full_circuit = config.circuit(&netlist).unwrap();
    full_circuit
        .set_observations(
            &quantities
                .iter()
                .map(|q| (*q).to_owned())
                .collect::<Vec<_>>(),
        )
        .unwrap();
    let full = analysis.run_plots(&mut full_circuit, &context).unwrap();
    assert_eq!(full.len(), 1);
    let full = &full[0];
    let requests = ObjectiveRequests {
        observations: wanted
            .iter()
            .map(|name| objective(name, ObservationReduction::Trace { max_points: 1000 }))
            .collect(),
        ..ObjectiveRequests::default()
    };
    let kind = analysis.kind();
    let plan = ScenarioPlan::new(vec![PlotObjectivePlan::bind(full, kind, requests).unwrap()]);
    assert!(plan.plots()[0].required_columns().len() < full.variable_count());
    let mut selected_circuit = config.circuit(&netlist).unwrap();
    selected_circuit
        .set_observations(
            &quantities
                .iter()
                .map(|q| (*q).to_owned())
                .collect::<Vec<_>>(),
        )
        .unwrap();
    let actual = plan
        .run(&analyses, &mut selected_circuit, &context, &scope)
        .unwrap();
    assert_eq!(actual.len(), 1);
    assert_eq!(actual[0].accepted_points, full.point_count());
    assert_eq!(actual[0].retained_cells, 0);
    for (expected_name, observed) in wanted.iter().zip(&actual[0].observations) {
        assert_eq!(&observed.name, expected_name);
        let ObservationValue::Trace(points) = observed.value.as_ref().unwrap() else {
            panic!("expected physical trace");
        };
        assert_eq!(points.len(), full.point_count());
        let source_index = full.variable_index(expected_name).unwrap();
        for (at, point) in points.iter().enumerate() {
            assert_eq!(point.value, full.row(at).unwrap()[source_index]);
            if kind != AnalysisKind::OperatingPoint {
                assert_eq!(point.coordinate, Some(full.row(at).unwrap()[0].re));
            }
        }
    }
}

#[test]
fn dc_sweep_uses_selected_device_columns_and_retains_each_accepted_point() {
    full_and_selected(
        "v1 n 0 0\nr1 n 0 1k\nr2 n 0 2k\n.dc v1 0 2 1",
        &["i(r1)", "@r1[p]", "@r2[i]", "@r1[current]", "@v1[dc]"],
        &["@r1[p]", "v(n)", "@v1[dc]"],
    );
}

#[test]
fn adaptive_transient_keeps_selected_trace_without_changing_accepted_grid() {
    full_and_selected(
        "v1 n 0 pulse(0 1 0 1n 1n 1u 2u)\nr1 n mid 1k\nc1 mid 0 1n\n.tran 0.1u 3u",
        &["i(r1)", "@r1[p]", "@r1[current]"],
        &["@r1[p]", "v(mid)"],
    );
}

#[test]
fn objective_only_ac_skips_an_unrequested_unsupported_spectral_ask() {
    let netlist = deck("v1 n 0 0 ac 1\nr1 n 0 1k\n.ac lin 3 1k 3k");
    let config = RunConfig::from_netlist(&netlist).unwrap();
    let context = config.context();
    let scope = ParamScope::for_netlist(&netlist).unwrap();
    let analysis = config.plan_for(&netlist.analyses[0]).unwrap();
    let mut reference_circuit = config.circuit(&netlist).unwrap();
    let reference = analysis
        .run_plots(&mut reference_circuit, &context)
        .unwrap();
    assert_eq!(reference.len(), 1);
    let source = &reference[0];
    let mut circuit = config.circuit(&netlist).unwrap();
    circuit.set_observations(&["@r1[i]".into()]).unwrap();
    // The full plot includes the unrequested ask and would reject it in AC.
    let mut variables = source.variables().to_vec();
    variables.extend(
        circuit
            .observations()
            .iter()
            .map(|observation| Variable::new(observation.name(), observation.unit())),
    );
    let header = Plot::new(
        source.name.clone(),
        source.plotname.clone(),
        source.flags(),
        variables,
    );
    let plan = ScenarioPlan::new(vec![
        PlotObjectivePlan::bind(
            &header,
            analysis.kind(),
            ObjectiveRequests {
                observations: vec![objective(
                    "v(n)",
                    ObservationReduction::Trace { max_points: 10 },
                )],
                ..ObjectiveRequests::default()
            },
        )
        .unwrap(),
    ]);
    assert_eq!(plan.plots()[0].required_columns(), &[0, 1]);
    let output = plan
        .run(&[analysis], &mut circuit, &context, &scope)
        .unwrap();
    let ObservationValue::Trace(points) = output[0].observations[0].value.as_ref().unwrap() else {
        panic!("expected selected AC trace");
    };
    assert_eq!(points.len(), source.point_count());
    for (index, point) in points.iter().enumerate() {
        assert_eq!(point.value, source.row(index).unwrap()[1]);
        assert_eq!(point.coordinate, Some(source.row(index).unwrap()[0].re));
    }
}

#[test]
fn malformed_compact_or_full_rows_poison_the_objective_sink() {
    use ngspice_rs::analysis::{PlotFlags, PointSink};
    let source = Plot::new(
        "op1",
        "Operating Point",
        PlotFlags::Real,
        vec![Variable::new("v(n)", "voltage")],
    );
    let plan = ScenarioPlan::new(vec![
        PlotObjectivePlan::bind(
            &source,
            AnalysisKind::OperatingPoint,
            ObjectiveRequests {
                observations: vec![objective("v(n)", ObservationReduction::Last)],
                ..ObjectiveRequests::default()
            },
        )
        .unwrap(),
    ]);
    let scope = ParamScope::root(&[]).unwrap();
    for negotiate in [false, true] {
        let mut sink = plan.sink(&scope);
        sink.begin(&source).unwrap();
        if negotiate {
            assert_eq!(sink.selected_columns(), Some([0_usize].as_slice()));
        }
        assert!(sink.point(&[]).is_err());
        assert!(
            sink.point(&[ngspice_rs::primitives::Complex::real(1.)])
                .is_err()
        );
        assert!(sink.end().is_err());
        assert!(sink.finish().is_err());
    }
}

/// Manually profile the actual driver and per-device load work rather than
/// comparing only already materialized plot projections. The numerical grid
/// and circuit are identical; source output and returned trace retention
/// are reported separately.
#[test]
#[ignore = "manual selected-vs-full producer benchmark"]
fn benchmark_real_producer_selection() {
    use std::fmt::Write as _;
    use std::hint::black_box;
    use std::time::{Duration, Instant};

    const RESISTORS: usize = 24;
    const REPEATS: usize = 3;
    let mut source = String::from("v1 n 0 0\n.dc v1 0 1 0.001\n");
    let mut queries = Vec::new();
    for index in 1..=RESISTORS {
        writeln!(source, "r{index} n 0 {}k", index).unwrap();
        queries.push(format!("i(r{index})"));
        queries.push(format!("@r{index}[p]"));
    }
    let netlist = deck(&source);
    let config = RunConfig::from_netlist(&netlist).unwrap();
    let context = config.context();
    let scope = ParamScope::for_netlist(&netlist).unwrap();
    let analysis = config.plan_for(&netlist.analyses[0]).unwrap();
    let mut baseline = config.circuit(&netlist).unwrap();
    baseline.set_observations(&queries).unwrap();
    let full_plot = analysis
        .run_plots(&mut baseline, &context)
        .unwrap()
        .remove(0);
    let plan = ScenarioPlan::new(vec![
        PlotObjectivePlan::bind(
            &full_plot,
            analysis.kind(),
            ObjectiveRequests {
                observations: vec![
                    objective("v(n)", ObservationReduction::Trace { max_points: 2000 }),
                    objective("@r1[p]", ObservationReduction::Trace { max_points: 2000 }),
                ],
                ..ObjectiveRequests::default()
            },
        )
        .unwrap(),
    ]);
    assert_eq!(plan.plots()[0].required_columns().len(), 3);
    let mut full_time = Duration::ZERO;
    let mut selected_time = Duration::ZERO;
    let mut retained_cells = 0;
    let mut accepted = 0;
    for i in 0..REPEATS {
        let mut full_circuit = config.circuit(&netlist).unwrap();
        full_circuit.set_observations(&queries).unwrap();
        let mut selected_circuit = config.circuit(&netlist).unwrap();
        selected_circuit.set_observations(&queries).unwrap();
        // Alternate order to reduce systematic heat/cache bias.
        if i % 2 == 0 {
            let start = Instant::now();
            let answer = plan
                .run(
                    std::slice::from_ref(&analysis),
                    &mut selected_circuit,
                    &context,
                    &scope,
                )
                .unwrap();
            selected_time += start.elapsed();
            accepted = answer[0].accepted_points;
            retained_cells = answer[0].retained_cells;
            black_box(answer);
            let start = Instant::now();
            black_box(analysis.run_plots(&mut full_circuit, &context).unwrap());
            full_time += start.elapsed();
        } else {
            let start = Instant::now();
            black_box(analysis.run_plots(&mut full_circuit, &context).unwrap());
            full_time += start.elapsed();
            let start = Instant::now();
            let answer = plan
                .run(
                    std::slice::from_ref(&analysis),
                    &mut selected_circuit,
                    &context,
                    &scope,
                )
                .unwrap();
            selected_time += start.elapsed();
            accepted = answer[0].accepted_points;
            retained_cells = answer[0].retained_cells;
            black_box(answer);
        }
    }
    assert_eq!(accepted, full_plot.point_count());
    assert_eq!(retained_cells, 0);
    eprintln!(
        "selected_producer n_resistors={RESISTORS} points={accepted}          full_width={} selected_width={} full_retained_cells={}          selected_raw_retained_cells={retained_cells}          full_isolated_device_loads_per_point={RESISTORS}          selected_isolated_device_loads_per_point=1          repeats={REPEATS} full_ms={:.3} selected_ms={:.3} speedup={:.3}",
        full_plot.variable_count(),
        plan.plots()[0].required_columns().len(),
        full_plot.cells().len(),
        full_time.as_secs_f64() * 1e3,
        selected_time.as_secs_f64() * 1e3,
        full_time.as_secs_f64() / selected_time.as_secs_f64(),
    );
}
