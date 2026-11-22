//! Trace the reference ordering of state writes, integration, convergence
//! and matrix/RHS stamps. Numerical goldens are additionally checked against C.
use super::super::{
    fields::{Instance, Model},
    host::{LoadCkt, mode, node, state},
    set, temp,
};
use crate::primitives::SpiceResult;

struct Trace {
    mode: u32,
    limiting: bool,
    voltage: [f64; node::COUNT],
    s0: [f64; state::COUNT],
    s1: [f64; state::COUNT],
    s2: [f64; state::COUNT],
    order: u64,
    values: u64,
    events: usize,
}

impl Trace {
    fn event(&mut self, kind: u64, row: u64, col: u64, value: f64) {
        for word in [kind, row, col] {
            self.order = (self.order ^ word).wrapping_mul(0x100_0000_01b3);
        }
        self.values = (self.values ^ value.to_bits()).wrapping_mul(0x100_0000_01b3);
        self.events += 1;
    }
}

impl LoadCkt for Trace {
    fn mode(&self) -> u32 {
        self.mode
    }
    fn limiting(&self) -> bool {
        self.limiting
    }
    fn gmin(&self) -> f64 {
        1e-12
    }
    fn temp(&self) -> f64 {
        300.15
    }
    fn ag(&self, i: usize) -> f64 {
        if self.mode & mode::MODETRAN != 0 {
            [1e11, -1e11, 0.][i]
        } else {
            0.
        }
    }
    fn delta(&self) -> f64 {
        1e-11
    }
    fn delta_old(&self, _: usize) -> f64 {
        2e-11
    }
    fn rhs_old(&self, node: i32) -> f64 {
        usize::try_from(node)
            .ok()
            .and_then(|index| self.voltage.get(index))
            .copied()
            .unwrap_or(0.)
    }
    fn state0(&self, slot: usize) -> f64 {
        self.s0[slot]
    }
    fn set_state0(&mut self, slot: usize, value: f64) {
        self.event(0, slot as u64, 0, value);
        self.s0[slot] = value;
    }
    fn state1(&self, slot: usize) -> f64 {
        self.s1[slot]
    }
    fn set_state1(&mut self, slot: usize, value: f64) {
        self.event(1, slot as u64, 0, value);
        self.s1[slot] = value;
    }
    fn state2(&self, slot: usize) -> f64 {
        self.s2[slot]
    }
    fn integrate(&mut self, cap: f64, slot: usize) -> SpiceResult<(f64, f64)> {
        self.event(2, slot as u64, 0, cap);
        let derivative = self.ag(0) * self.s0[slot] + self.ag(1) * self.s1[slot];
        self.s0[slot + 1] = derivative;
        Ok((self.ag(0) * cap, derivative - self.ag(0) * self.s0[slot]))
    }
    fn noncon(&mut self) {
        self.event(3, 0, 0, 0.);
    }
    fn matrix(&mut self, row: i32, col: i32, value: f64) {
        self.event(
            4,
            u64::from(row.cast_unsigned()),
            u64::from(col.cast_unsigned()),
            value,
        );
    }
    fn rhs(&mut self, node: i32, value: f64) {
        self.event(5, u64::from(node.cast_unsigned()), 0, value);
    }
}

fn prepared(
    version: &str,
    diode: i32,
    cap: i32,
    gate: i32,
    body: i32,
    nqs: i32,
) -> (Model, Instance, super::SizeParam, Vec<String>) {
    let model = Model {
        version: version.into(),
        versionGiven: true,
        dioMod: diode,
        dioModGiven: true,
        capMod: cap,
        capModGiven: true,
        rgateMod: gate,
        rgateModGiven: true,
        rbodyMod: body,
        rbodyModGiven: true,
        trnqsMod: nqs,
        trnqsModGiven: true,
        acnqsMod: nqs,
        acnqsModGiven: true,
        ..Model::default()
    };
    let instance = Instance {
        l: 0.5e-6,
        lGiven: true,
        w: 2e-6,
        wGiven: true,
        drainArea: 1e-12,
        drainAreaGiven: true,
        sourceArea: 1e-12,
        sourceAreaGiven: true,
        drainPerimeter: 2e-6,
        drainPerimeterGiven: true,
        sourcePerimeter: 2e-6,
        sourcePerimeterGiven: true,
        ..super::super::acld::tests::shaped(gate, body, nqs)
    };
    let set::PreparedSetup {
        mut model,
        mut instance,
        mut warnings,
        ..
    } = set::prepare_model(model, 300.15).prepare_instance(instance, false);
    let size = temp::temperature(
        &mut model,
        &mut instance,
        300.15,
        "nch",
        "m1",
        &mut warnings,
    )
    .unwrap();
    (model, instance, size, warnings)
}

impl Trace {
    const fn bias(drain: f64, gate: f64, body: f64) -> Self {
        Self {
            mode: mode::MODEDCTRANCURVE | mode::MODEINITFLOAT,
            limiting: false,
            voltage: [
                drain, gate, 0., body, drain, gate, gate, 0., body, body, body, 0.,
            ],
            s0: [0.; state::COUNT],
            s1: [0.; state::COUNT],
            s2: [0.; state::COUNT],
            order: 0xcbf2_9ce4_8422_2325,
            values: 0xcbf2_9ce4_8422_2325,
            events: 0,
        }
    }
}

#[test]
fn load_event_sequence_and_field_values() {
    // Baseline at 302ce72, before splitting the translated load.
    for (version, diode, cap, gate, body, nqs, events, order) in [
        ("4.8.0", 0, 0, 0, 0, 0, 351, 0x8d52_ac1f_70f3_f05e),
        ("4.8.1", 1, 1, 2, 1, 0, 496, 0x18d3_cb9c_9da8_6fc1),
        ("4.8.3", 2, 2, 3, 2, 1, 603, 0x1cab_191b_0771_eb2f),
    ] {
        let (model, mut instance, size, warnings) = prepared(version, diode, cap, gate, body, nqs);
        let seed = 0xcbf2_9ce4_8422_2325;
        let mut trace = Trace {
            mode: mode::MODEDCOP | mode::MODEINITFLOAT,
            limiting: false,
            voltage: [
                0.6, 0.9, 0., -0.1, 0.55, 0.88, 0.89, 0.01, -0.08, -0.07, -0.09, 0.,
            ],
            s0: [0.; state::COUNT],
            s1: [0.; state::COUNT],
            s2: [0.; state::COUNT],
            order: seed,
            values: seed,
            events: 0,
        };
        for (kind, limiting, reverse) in [
            (mode::MODEDCOP | mode::MODEINITJCT, true, false),
            (mode::MODEDCTRANCURVE | mode::MODEINITFLOAT, false, false),
            (mode::MODEDCOP | mode::MODEINITSMSIG, false, false),
            (mode::MODETRAN | mode::MODEINITTRAN, true, false),
            (mode::MODETRAN | mode::MODEINITPRED, true, false),
            (mode::MODETRAN | mode::MODEINITFLOAT, true, true),
        ] {
            trace.mode = kind;
            trace.limiting = limiting;
            if reverse {
                trace.voltage[node::DP as usize] = -0.25;
                trace.voltage[node::D as usize] = -0.3;
            }
            trace.s2 = trace.s1;
            trace.s1 = trace.s0;
            let prior = instance.to_values();
            super::load(&model, &mut instance, &size, &mut trace).unwrap();
            for ((name, value), before) in Instance::MEMBERS
                .iter()
                .zip(instance.to_values())
                .zip(prior)
            {
                if Instance::LOAD_WRITES.contains(name) {
                    trace.values = (trace.values ^ value.to_bits()).wrapping_mul(0x100_0000_01b3);
                } else {
                    assert_eq!(
                        value.to_bits(),
                        before.to_bits(),
                        "{version}: unexpected write to {name}"
                    );
                }
            }
        }
        assert_eq!(trace.events, events, "{version}: event count");
        assert_eq!(trace.order, order, "{version}: event sequence");
        if nqs == 0 {
            assert!(warnings.is_empty(), "{warnings:?}");
        } else {
            assert_eq!(
                warnings,
                [
                    "\nChecking parameters for BSIM 4.8 model nch",
                    "Warning: You've selected both Rg and charge deficit NQS; select one only.",
                ]
            );
        }
        eprintln!(
            "{version}: events={} order={:016x} values={:016x} warnings={warnings:?}",
            trace.events, trace.order, trace.values
        );
    }
}

/// Numerical invariants independent of the FMA-free C oracle. Evaluate the
/// isolated channel with no limiting or topology changes, so finite
/// differences interrogate the actual derivatives used by the load.
#[test]
fn charge_conservation_and_channel_derivatives_over_bias_grid() {
    fn conserved(values: [f64; 4], scale: f64, label: &str) {
        assert!(values.iter().all(|x| x.is_finite()), "{label}: {values:?}");
        let sum = values.iter().sum::<f64>();
        assert!(
            sum.abs() <= 8. * f64::EPSILON * scale,
            "{label}: {values:?}, sum = {sum:e}"
        );
    }

    for cap in 0..=2 {
        let (model, seed, size, _) = prepared("4.8.3", 2, cap, 0, 0, 0);
        let evaluate = |d, g, b| {
            let mut instance = seed.clone();
            super::load(&model, &mut instance, &size, &mut Trace::bias(d, g, b)).unwrap();
            instance
        };
        for g in [0.1, 0.45, 0.9, 1.6] {
            for d in [1e-5, 0.1, 1.2] {
                for b in [-0.7, -0.05] {
                    let center = evaluate(d, g, b);
                    let charge = [center.qgate, center.qdrn, center.qsrc, center.qbulk];
                    conserved(charge, charge.iter().map(|x| x.abs()).sum(), "charge");
                    let matrix = [
                        [center.cggb, center.cgdb, center.cgsb, center.cgbb],
                        [center.cdgb, center.cddb, center.cdsb, center.cdbb],
                        [center.csgb, center.csdb, center.cssb, center.csbb],
                        [center.cbgb, center.cbdb, center.cbsb, center.cbbb],
                    ];
                    // Some entries are themselves sums/differences of other
                    // matrix entries. Conservation is backward error against
                    // the entire capacitance matrix, not a relative error
                    // against a nearly cancelled single row or column.
                    let scale = matrix.iter().flatten().map(|x| x.abs()).sum();
                    for row in matrix {
                        conserved(row, scale, "capacitance row");
                    }
                    for col in 0..4 {
                        conserved(matrix.map(|row| row[col]), scale, "capacitance column");
                    }
                    const DELTA: f64 = 1e-6;
                    let derivative = |lo: f64, hi: f64| (hi - lo) / (2. * DELTA);
                    let gm = derivative(evaluate(d, g - DELTA, b).cd, evaluate(d, g + DELTA, b).cd);
                    let gds =
                        derivative(evaluate(d - DELTA, g, b).cd, evaluate(d + DELTA, g, b).cd);
                    let gmb =
                        derivative(evaluate(d, g, b - DELTA).cd, evaluate(d, g, b + DELTA).cd);
                    for (name, actual, expected) in [
                        ("gm", gm, center.gm),
                        ("gds", gds, center.gds),
                        ("gmb", gmb, center.gmbs),
                    ] {
                        assert!(actual.is_finite() && expected.is_finite());
                        assert!(
                            (actual - expected).abs()
                                <= 2e-4 * actual.abs().max(expected.abs()) + 1e-14,
                            "capmod {cap}, d={d:e}, g={g:e}, b={b:e}: {name}: finite difference {actual:e} vs analytic {expected:e}"
                        );
                    }
                }
            }
        }
    }
}

/// Exercise real prepared device loads rather than an isolated FMA loop.
/// Repeat the same workload for the original/current source, generic CPU
/// and native CPU builds. Host hashing is identical in every configuration.
#[test]
#[ignore = "manual BSIM4 throughput benchmark"]
fn benchmark_bsim4_arithmetic() {
    use std::{hint::black_box, time::Instant};
    const LOADS: u32 = 200_000;
    fn thread_ticks() -> Option<u64> {
        let stat = std::fs::read_to_string("/proc/thread-self/stat").ok()?;
        let (_, fields) = stat.rsplit_once(')')?;
        let mut fields = fields.split_whitespace().skip(11);
        let user = fields.next()?.parse::<u64>().ok()?;
        let system = fields.next()?.parse::<u64>().ok()?;
        Some(user + system)
    }
    for cap in 0..=2 {
        let (model, seed, size, _) = prepared("4.8.3", 2, cap, 0, 0, 0);
        let mut instance = seed.clone();
        let mut trace = Trace::bias(0.1, 0.45, -0.05);
        for batch in 0..5 {
            let cpu_start = thread_ticks();
            let start = Instant::now();
            for index in 0..LOADS {
                let v = f64::from(index % 101) / 100.;
                let d = 0.01 + 1.1 * v;
                let g = 0.1 + 1.4 * (1. - v);
                let b = -0.05 - 0.5 * v;
                trace.voltage = [d, g, 0., b, d, g, g, 0., b, b, b, 0.];
                super::load(
                    black_box(&model),
                    black_box(&mut instance),
                    black_box(&size),
                    black_box(&mut trace),
                )
                .unwrap();
            }
            black_box((&instance, &trace));
            let elapsed = start.elapsed();
            let ticks = cpu_start
                .zip(thread_ticks())
                .map(|(start, end)| end.saturating_sub(start));
            println!(
                "bsim4 capmod={cap} batch={batch} loads={LOADS} elapsed_ns={} cpu_ticks={ticks:?}",
                elapsed.as_nanos()
            );
        }
    }
}
