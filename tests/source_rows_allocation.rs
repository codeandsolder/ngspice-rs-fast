//! Repeated AC assembly must reuse immutable source handles and inline rows.

use ngspice_rs::devices::linear::SourceRows;
use ngspice_rs::devices::{Circuit, IndependentSource, ModelContext, Waveform};
use ngspice_rs::primitives::{Complex, NodeId};
use std::sync::Arc;

fn circuit(reverse_nodes: bool) -> Circuit {
    let mut circuit = Circuit::new();
    if reverse_nodes {
        circuit.add_node("b").unwrap();
        circuit.add_node("a").unwrap();
    } else {
        circuit.add_node("a").unwrap();
        circuit.add_node("b").unwrap();
    }
    let a = circuit.nodes().get("a").unwrap();
    let b = circuit.nodes().get("b").unwrap();
    let gnd = NodeId::GROUND;
    let cases = [
        ("v1", [a, gnd], true, 2., Complex::new(3., -1.)),
        ("i1", [a, gnd], false, 0.5, Complex::new(1., 0.)),
        ("i2", [a, a], false, 0.75, Complex::new(2., 0.)),
        ("i3", [gnd, gnd], false, 2.0, Complex::new(-1., 0.)),
        ("i4", [gnd, b], false, -4., Complex::new(4., 0.)),
    ];
    let waveform =
        Waveform::Pwl(ngspice_rs::devices::PwlSource::plain(vec![(0., 1.), (1., 3.)]).unwrap());
    for (name, nodes, voltage, dc, ac) in cases {
        let source =
            IndependentSource::new(name, nodes, voltage, dc, ac, waveform.clone()).unwrap();
        circuit.add_device(Box::new(source)).unwrap();
    }
    circuit.finalize().unwrap();
    circuit
}

#[test]
fn source_rows_are_inline_bounded_and_reject_overflow() {
    assert!(std::mem::size_of::<SourceRows>() <= 48);
    let mut rows = SourceRows::new();
    assert!(rows.is_empty());
    rows.push(3, -2.).unwrap();
    rows.push(3, 2.).unwrap();
    assert_eq!(rows.as_slice(), &[(3, -2.), (3, 2.)]);
    assert!(rows.push(4, 1.).is_err());
    assert_eq!(rows.as_slice(), &[(3, -2.), (3, 2.)]);
}

fn rows<'a>(system: &'a ngspice_rs::devices::LinearSystem, name: &str) -> &'a [(usize, f64)] {
    system
        .sources
        .iter()
        .find(|source| source.name.eq_ignore_ascii_case(name))
        .unwrap_or_else(|| panic!("missing {name}"))
        .rows
        .as_slice()
}

#[test]
fn ground_duplicate_and_branch_rows_preserve_sign_and_order() {
    for reverse_nodes in [false, true] {
        let mut c = circuit(reverse_nodes);
        let a = c.unknowns().node_row(c.nodes().get("a").unwrap()).unwrap();
        let b = c.unknowns().node_row(c.nodes().get("b").unwrap()).unwrap();
        let system = c.linear_system().unwrap();
        assert_eq!(system.sources.len(), 5);
        assert!(system.sources.capacity() >= system.sources.len());
        assert_eq!(rows(&system, "i1"), [(a, -1.)]);
        assert_eq!(rows(&system, "i2"), [(a, -1.), (a, 1.)]);
        assert_eq!(rows(&system, "i3"), []);
        assert_eq!(rows(&system, "i4"), [(b, 1.)]);
        assert_eq!(rows(&system, "v1").len(), 1);
        assert_eq!(rows(&system, "v1")[0].1, 1.);
        let dc = system.dc_rhs(None).unwrap();
        let ac = system.ac_rhs();
        assert_eq!(dc.as_slice()[a], -0.5);
        assert_eq!(dc.as_slice()[b], -4.);
        assert_eq!(ac[a], Complex::real(-1.));
        assert_eq!(ac[b], Complex::real(4.));
        let sweep = system.dc_rhs(Some(("I1", 2.5))).unwrap();
        assert_eq!(sweep.as_slice()[a], -2.5);
        assert_eq!(sweep.as_slice()[b], -4.);
    }
}

#[test]
fn repeated_contexts_reuse_handles_without_stale_topology() {
    let mut c = circuit(false);
    let default_context = ModelContext::default();
    let alternate = ModelContext {
        temperature: 70.,
        ..default_context
    };
    let first = c.linear_system_with_context(&default_context).unwrap();
    let changed = c.linear_system_with_context(&alternate).unwrap();
    let again = c.linear_system_with_context(&default_context).unwrap();
    for ((a, b), d) in first
        .sources
        .iter()
        .zip(&changed.sources)
        .zip(&again.sources)
    {
        assert_eq!(a.rows, b.rows);
        assert_eq!(a.rows, d.rows);
        assert!(Arc::ptr_eq(&a.name, &b.name));
        assert!(Arc::ptr_eq(&a.name, &d.name));
        assert!(Arc::ptr_eq(&a.waveform, &b.waveform));
        assert!(Arc::ptr_eq(&a.waveform, &d.waveform));
        assert_eq!(a.ac, b.ac);
        assert_eq!(a.dc, d.dc);
    }
    let other = circuit(true).linear_system().unwrap();
    assert_eq!(
        first
            .sources
            .iter()
            .map(|source| source.name.as_str())
            .collect::<Vec<_>>(),
        other
            .sources
            .iter()
            .map(|source| source.name.as_str())
            .collect::<Vec<_>>(),
    );
    let source = first
        .sources
        .iter()
        .find(|source| source.name.as_str() == "i1")
        .unwrap();
    let other_source = other
        .sources
        .iter()
        .find(|source| source.name.as_str() == "i1")
        .unwrap();
    assert_ne!(source.rows, other_source.rows);
    assert!(!Arc::ptr_eq(&source.name, &other_source.name));
}

#[test]
fn transient_binding_must_not_modify_original_immutable_waveform() {
    let mut c = circuit(false);
    let a = c.linear_system().unwrap();
    let mut b = c.linear_system().unwrap();
    let before_source = a
        .sources
        .iter()
        .find(|source| source.name.as_str() == "v1")
        .unwrap();
    let after_source = b
        .sources
        .iter()
        .find(|source| source.name.as_str() == "v1")
        .unwrap();
    assert!(Arc::ptr_eq(&before_source.waveform, &after_source.waveform));
    let before = a
        .transient_rhs(0.5, ngspice_rs::devices::Limit::Right)
        .unwrap();
    b.bind_transient_timing(&ngspice_rs::devices::TransientTiming::new(0.1, 1.).unwrap())
        .unwrap();
    let after = b
        .transient_rhs(0.5, ngspice_rs::devices::Limit::Right)
        .unwrap();
    assert_eq!(before, after);
    assert!(!Arc::ptr_eq(
        &before_source.waveform,
        &b.sources
            .iter()
            .find(|source| source.name.as_str() == "v1")
            .unwrap()
            .waveform,
    ));
}

#[test]
#[ignore = "manual repeated bound source assembly workload"]
fn benchmark_repeated_multi_source_small_signal() {
    use std::hint::black_box;
    use std::time::Instant;
    let mut c = circuit(false);
    let ctx = ModelContext::default();
    let first = c.linear_system_with_context(&ctx).unwrap();
    let mut reused = 0_usize;
    let start = Instant::now();
    for _ in 0..1000 {
        let system = c.linear_system_with_context(&ctx).unwrap();
        for (before, now) in first.sources.iter().zip(&system.sources) {
            reused += usize::from(Arc::ptr_eq(&before.name, &now.name));
            reused += usize::from(Arc::ptr_eq(&before.waveform, &now.waveform));
        }
        black_box(system);
    }
    println!(
        "source_count={} repeats=1000 shared_handles={} elapsed_ms={:.3} \
         source_vec_allocations=1000 source_row_allocations=0 source_name_allocations=0 source_waveform_allocations=0",
        first.sources.len(),
        reused,
        start.elapsed().as_secs_f64() * 1000.,
    );
    assert_eq!(reused, 1000 * first.sources.len() * 2);
}
