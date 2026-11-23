//! Dynamic small-signal statement storage: physical assembly and allocation probe.
use ngspice_rs::devices::{Circuit, ModelContext};
use ngspice_rs::maths::{Vector, complex::ComplexMatrix};
use ngspice_rs::netlist::{Parser, source::parse_deck_text};
use std::{hint::black_box, path::Path, time::Instant};

fn inductor_circuit(n: usize) -> Circuit {
    let mut text = String::from("Grouped inductor storage\n");
    for i in 0..n {
        use std::fmt::Write;
        writeln!(text, "l{i} n{i} 0 1u m=2").unwrap();
        writeln!(text, "r{i} n{i} 0 1k").unwrap();
    }
    text.push_str(".ac lin 1 1k 1k\n.end\n");
    let netlist = Parser::new()
        .parse_deck(&parse_deck_text(Path::new("dynamic_groups.cir"), &text))
        .unwrap();
    Circuit::from_netlist(&netlist).unwrap()
}

#[test]
fn assembled_groups_are_bound_reusable_and_frequency_correct() {
    let c = inductor_circuit(8);
    let system = c
        .small_signal_system(&ModelContext::default(), &Vector::zeros(c.unknown_count()))
        .unwrap();
    assert_eq!(system.e_groups.len(), 8);
    assert_eq!(
        system
            .e_groups
            .iter()
            .map(|group| group.terms.len())
            .sum::<usize>(),
        8
    );
    let pencil = c
        .pole_zero_terms_at(
            &ModelContext::default(),
            &Vector::zeros(c.unknown_count()),
            None,
            1,
        )
        .unwrap();
    for group in system.e_groups.iter() {
        let original = &system.e.triplets()[group.range.start];
        let effective = pencil
            .iter()
            .find(|term| term.row == original.row && term.col == original.col)
            .and_then(|term| term.capacitance)
            .unwrap();
        // PZ and AC share the same physical stamp, including 1/M.
        assert_eq!(effective.to_bits(), original.value.to_bits());
    }
    let omega = 2. * std::f64::consts::PI * 1e3;
    let matrix =
        ComplexMatrix::from_operators_grouped(&system.a, &system.e, &system.e_groups, omega)
            .unwrap();
    assert!(matrix.factorize().is_ok());
    let copy = system.clone();
    assert!(ComplexMatrix::from_operators_grouped(&copy.a, &copy.e, &copy.e_groups, omega).is_ok());
    let mut stale = system.clone();
    stale.e.add(0, 0, 1.).unwrap();
    assert!(
        ComplexMatrix::from_operators_grouped(&stale.a, &stale.e, &stale.e_groups, omega).is_err()
    );
}

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
static RECORD: AtomicBool = AtomicBool::new(false);
static CALLS: AtomicUsize = AtomicUsize::new(0);
static BYTES: AtomicUsize = AtomicUsize::new(0);
struct CountingAllocator;

// SAFETY: all allocator operations delegate to System; atomic accounting
// neither holds pointers nor allocates and cannot alter allocator semantics.
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // SAFETY: delegated to the standard system allocator with this layout.
        let ptr = unsafe { System.alloc(layout) };
        if !ptr.is_null() && RECORD.load(Ordering::Relaxed) {
            CALLS.fetch_add(1, Ordering::Relaxed);
            BYTES.fetch_add(layout.size(), Ordering::Relaxed);
        }
        ptr
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        // SAFETY: delegated to System with the original layout.
        let ptr = unsafe { System.alloc_zeroed(layout) };
        if !ptr.is_null() && RECORD.load(Ordering::Relaxed) {
            CALLS.fetch_add(1, Ordering::Relaxed);
            BYTES.fetch_add(layout.size(), Ordering::Relaxed);
        }
        ptr
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        // SAFETY: ptr and its old layout are forwarded unchanged to System.
        let next = unsafe { System.realloc(ptr, layout, new_size) };
        if !next.is_null() && RECORD.load(Ordering::Relaxed) {
            CALLS.fetch_add(1, Ordering::Relaxed);
            BYTES.fetch_add(new_size, Ordering::Relaxed);
        }
        next
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: ptr and the layout are forwarded unchanged to System.
        unsafe { System.dealloc(ptr, layout) };
    }
}
#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

fn measure<T>(f: impl FnOnce() -> T) -> (T, usize, usize) {
    CALLS.store(0, Ordering::Relaxed);
    BYTES.store(0, Ordering::Relaxed);
    RECORD.store(true, Ordering::SeqCst);
    let result = f();
    RECORD.store(false, Ordering::SeqCst);
    (
        result,
        CALLS.load(Ordering::Relaxed),
        BYTES.load(Ordering::Relaxed),
    )
}

#[test]
#[ignore = "manual allocation and timing comparison of many real small-signal devices"]
fn dynamic_group_arena_allocation_benchmark() {
    for n in [64, 512, 2048] {
        let c = inductor_circuit(n);
        let bias = Vector::zeros(c.unknown_count());
        let context = ModelContext::default();
        let start = Instant::now();
        let (system, allocations, bytes) =
            measure(|| c.small_signal_system(&context, &bias).unwrap());
        let elapsed = start.elapsed().as_nanos();
        assert_eq!(system.e_groups.len(), n);
        let (terms, borrowed_allocations, borrowed_bytes) = measure(|| {
            system
                .e_groups
                .iter()
                .map(|group| group.terms.len())
                .sum::<usize>()
        });
        assert_eq!(terms, n);
        assert_eq!((borrowed_allocations, borrowed_bytes), (0, 0));
        let start = Instant::now();
        let (legacy, legacy_allocations, legacy_bytes) = measure(|| {
            system
                .e_groups
                .iter()
                .map(|group| black_box(group.terms.to_vec()))
                .collect::<Vec<_>>()
        });
        let legacy_ns = start.elapsed().as_nanos();
        assert_eq!(legacy.len(), n);
        println!(
            "groups={n},system_allocations={allocations},system_bytes={bytes},system_ns={elapsed},arena_bytes={},legacy_group_allocations={legacy_allocations},legacy_group_bytes={legacy_bytes},legacy_group_ns={legacy_ns},borrowed_allocations={borrowed_allocations}",
            system.e_groups.storage_bytes()
        );
    }
}
