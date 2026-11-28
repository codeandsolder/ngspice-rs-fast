//! Trace the reference ordering of state writes, integration, convergence
//! and matrix/RHS stamps. Numerical goldens are additionally checked against C.
use super::super::{
    fields::{Instance, Model},
    host::{LoadCkt, mode, node, state},
    set, temp,
};
use crate::primitives::SpiceResult;

struct Trace {
    temperature: f64,
    mode: u32,
    limiting: bool,
    voltage: [f64; node::COUNT],
    s0: [f64; state::COUNT],
    s1: [f64; state::COUNT],
    s2: [f64; state::COUNT],
    order: u64,
    values: u64,
    events: usize,
    matrix: [[f64; node::COUNT]; node::COUNT],
    rhs: [f64; node::COUNT],
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
        self.temperature
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
        if row >= 0 && col >= 0 {
            self.matrix[row.cast_unsigned() as usize][col.cast_unsigned() as usize] += value;
        }
    }
    fn rhs(&mut self, node: i32, value: f64) {
        self.event(5, u64::from(node.cast_unsigned()), 0, value);
        if node >= 0 {
            self.rhs[node.cast_unsigned() as usize] += value;
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
        let seed = 0xcbf2_9ce4_8422_2325;
        let mut trace = Trace {
            temperature: 300.15,
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
            matrix: [[0.; node::COUNT]; node::COUNT],
            rhs: [0.; node::COUNT],
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

/// Independent constitutive-law test: no Newton solve, device limiting,
/// companion integration, or external source-current subtraction.
struct PhysicalFixture {
    temperature: f64,
    model: Model,
    instance: Instance,
    size: super::SizeParam,
}

struct PhysicalPoint {
    current: f64,
    current_gradient: [f64; 4],
    charge: [f64; 4],
    capacitance: [[f64; 4]; 4],
}

impl PhysicalFixture {
    fn new(polarity: i32, cap: i32, cv: i32, partition: f64) -> Self {
        let model = Model {
            r#type: polarity,
            typeGiven: true,
            capMod: cap,
            capModGiven: true,
            cvchargeMod: cv,
            cvchargeModGiven: true,
            xpart: partition,
            xpartGiven: true,
            ..Model::default()
        };
        Self::prepare(model, 300.15)
    }

    fn prepare(model: Model, temperature: f64) -> Self {
        let instance = Instance {
            l: 0.5e-6,
            lGiven: true,
            w: 2e-6,
            wGiven: true,
            ..super::super::acld::tests::shaped(0, 0, 0)
        };
        let set::PreparedSetup {
            mut model,
            mut instance,
            mut warnings,
            ..
        } = set::prepare_model(model, temperature).prepare_instance(instance, false);
        let size = temp::temperature(
            &mut model,
            &mut instance,
            temperature,
            "nch",
            "m1",
            &mut warnings,
        )
        .unwrap();
        Self {
            temperature,
            model,
            instance,
            size,
        }
    }

    /// Physical terminal order: G, D, S, B.
    fn evaluate(&self, [g, d, s, b]: [f64; 4]) -> PhysicalPoint {
        let mut instance = self.instance.clone();
        let mut trace = Trace {
            temperature: self.temperature,
            mode: mode::MODEDCTRANCURVE | mode::MODEINITFLOAT,
            limiting: false,
            voltage: [d, g, s, b, d, g, g, s, b, b, b, 0.],
            s0: [0.; state::COUNT],
            s1: [0.; state::COUNT],
            s2: [0.; state::COUNT],
            order: 0,
            values: 0,
            events: 0,
            matrix: [[0.; node::COUNT]; node::COUNT],
            rhs: [0.; node::COUNT],
        };
        let bias = super::bias::select(&self.model, &instance, &mut trace);
        let electro =
            super::channel::electrostatics(&self.model, &mut instance, &self.size, &trace, &bias)
                .unwrap();
        let transport =
            super::channel::transport(&self.model, &mut instance, &self.size, &bias, &electro)
                .unwrap();
        let _ = super::charge::intrinsic(
            &self.model,
            &mut instance,
            &self.size,
            &mut trace,
            &bias,
            &electro,
            &transport,
        )
        .unwrap();
        let mode = f64::from(instance.mode);
        let pol = f64::from(self.model.r#type);
        let gm = instance.gm;
        let gd = instance.gds;
        let gb = instance.gmbs;
        let mut gradient = [gm, gd, -(gm + gd + gb), gb];
        let mut q = [instance.qgate, instance.qdrn, instance.qsrc, instance.qbulk];
        let mut c = [
            [instance.cggb, instance.cgdb, instance.cgsb, instance.cgbb],
            [instance.cdgb, instance.cddb, instance.cdsb, instance.cdbb],
            [instance.csgb, instance.csdb, instance.cssb, instance.csbb],
            [instance.cbgb, instance.cbdb, instance.cbsb, instance.cbbb],
        ];
        if instance.mode < 0 {
            gradient.swap(1, 2);
            q.swap(1, 2);
            c.swap(1, 2);
            for row in &mut c {
                row.swap(1, 2);
            }
        }
        PhysicalPoint {
            current: pol * mode * transport.unscaled_drain_current,
            current_gradient: gradient.map(|x| mode * x),
            charge: q.map(|x| pol * x),
            capacitance: c,
        }
    }
}

fn richardson([minus, plus, minus_half, plus_half]: [f64; 4], h: f64) -> (f64, f64) {
    let coarse = (plus - minus) / (2. * h);
    let fine = (plus_half - minus_half) / h;
    ((4. * fine - coarse) / 3., (fine - coarse).abs())
}

fn record_error(
    errors: &mut Vec<String>,
    actual: f64,
    expected: f64,
    bound: f64,
    context: impl FnOnce() -> String,
) {
    if !actual.is_finite() || !expected.is_finite() || (actual - expected).abs() > bound {
        errors.push(format!(
            "{}: {actual:e} vs {expected:e}, bound {bound:e}",
            context()
        ));
    }
}

fn check_physical_point(
    fixture: &PhysicalFixture,
    v: [f64; 4],
    context: &str,
    errors: &mut Vec<String>,
) {
    let p = fixture.evaluate(v);
    let q_scale = p.charge.iter().map(|x| x.abs()).sum::<f64>();
    record_error(
        errors,
        p.charge.iter().sum(),
        0.,
        q_scale * 8. * f64::EPSILON,
        || format!("{context}: charge KCL"),
    );
    // The source row and bulk column are reconstructed from other entries.
    // Their rounding budget therefore depends on the whole matrix norm,
    // not the norm of a nearly cancelled row.
    let c_scale = p.capacitance.iter().flatten().map(|x| x.abs()).sum::<f64>();
    for (r, row) in p.capacitance.iter().enumerate() {
        record_error(
            errors,
            row.iter().sum(),
            0.,
            c_scale * 8. * f64::EPSILON,
            || format!("{context}: common mode row {r} {row:?}"),
        );
    }
    for col in 0..4 {
        let column = p.capacitance.map(|row| row[col]);
        record_error(
            errors,
            column.iter().sum(),
            0.,
            c_scale * 8. * f64::EPSILON,
            || format!("{context}: charge Jacobian KCL col {col}"),
        );
        let h = 1e-4;
        let samples = [-h, h, -h * 0.5, h * 0.5].map(|delta| {
            let mut shifted = v;
            shifted[col] += delta;
            fixture.evaluate(shifted)
        });
        let (reference, error) = richardson(samples.each_ref().map(|sample| sample.current), h);
        record_error(
            errors,
            p.current_gradient[col],
            reference,
            3e-5 * reference.abs() + error * 0.1 + 1e-13,
            || format!("{context}: current col {col}"),
        );
        for row in 0..4 {
            let (reference, error) =
                richardson(samples.each_ref().map(|sample| sample.charge[row]), h);
            record_error(
                errors,
                p.capacitance[row][col],
                reference,
                3e-5 * reference.abs() + error * 0.1 + 1e-21,
                || format!("{context}: charge row {row} col {col}"),
            );
        }
    }
}

#[test]
fn physical_channel_and_charge_partials_against_richardson() {
    let mut errors = Vec::new();
    for polarity in [-1, 1] {
        for cap in 0..=2 {
            for cv in 0..=1 {
                for partition in [0., 0.5, 1.] {
                    let fixture = PhysicalFixture::new(polarity, cap, cv, partition);
                    for normalized in [
                        [0.1, 0.05, 0., -0.1],
                        [0.7, 0.02, 0., -0.2],
                        [0.9, 0.4, 0., -0.1],
                        [1.2, 1.0, 0., -0.4],
                        [0.9, -0.2, 0., -0.5],
                    ] {
                        let v = normalized.map(|x| x * f64::from(polarity));
                        let context = format!(
                            "pol={polarity} cap={cap} cv={cv} xpart={partition} bias={normalized:?}"
                        );
                        check_physical_point(&fixture, v, &context, &mut errors);
                    }
                }
            }
        }
    }
    assert!(
        errors.is_empty(),
        "{} independent derivative/conservation failures:\n{}",
        errors.len(),
        errors.join("\n")
    );
}

#[test]
fn smoothed_drain_preserves_small_signal_and_saturation_limits() {
    let sat = 0.5;
    let delta = 0.01;
    // Independently evaluated with 70-digit decimal arithmetic from the
    // rationalized quadratic root (not the original subtractive formula).
    for (drain, expected) in [
        (1e-3, 9.803_543_928_288_431e-4),
        (1e-6, 9.803_921_191_697_806e-7),
        (1e-9, 9.803_921_568_250_522e-10),
        (1e-12, 9.803_921_568_627_074e-13),
        (1e-15, 9.803_921_568_627_451e-16),
        (1e-18, 9.803_921_568_627_452e-19),
    ] {
        let p = super::smooth_drain(sat, drain, delta).unwrap();
        assert!((p.effective - expected).abs() <= 8. * f64::EPSILON * expected);
        assert!((p.effective + p.drain_excess - drain).abs() <= 8. * f64::EPSILON * drain);
        assert!(p.saturation_partial > 0. && p.drain_partial > 0.);
    }
    let zero = super::smooth_drain(sat, 0., delta).unwrap();
    assert_eq!(zero.effective, 0.);
    assert_eq!(zero.drain_excess, 0.);
    assert_eq!(zero.saturation_partial, 0.);
    assert!((zero.drain_partial - sat / (sat + delta)).abs() < 4. * f64::EPSILON);
    for (sat, drain) in [
        (0.5, 0.1),
        (0.5, 0.49),
        (0.5, 0.5),
        (0.5, 1.2),
        (1e-8, 1.),
        (1., 1e-8),
    ] {
        let p = super::smooth_drain(sat, drain, delta).unwrap();
        let h = 1e-5 * sat.min(drain);
        let steps = [-h, h, -h * 0.5, h * 0.5];
        let (ds, es) = richardson(
            steps.map(|d| {
                super::smooth_drain(sat + d, drain, delta)
                    .unwrap()
                    .effective
            }),
            h,
        );
        let (dd, ed) = richardson(
            steps.map(|d| {
                super::smooth_drain(sat, drain + d, delta)
                    .unwrap()
                    .effective
            }),
            h,
        );
        assert!((p.saturation_partial - ds).abs() < 2e-8 * ds.abs() + es * 0.1 + 2e-10);
        assert!((p.drain_partial - dd).abs() < 2e-8 * dd.abs() + ed * 0.1 + 2e-10);
    }
}

#[test]
fn intrinsic_physics_preserves_orientation_common_mode_and_low_drain_limit() {
    for polarity in [-1, 1] {
        for cap in 0..=2 {
            let fixture = PhysicalFixture::new(polarity, cap, 0, 0.5);
            let pol = f64::from(polarity);
            let forward = [0.9 * pol, 0.3 * pol, 0., -0.2 * pol];
            let a = fixture.evaluate(forward);
            let mut reversed = forward;
            reversed.swap(1, 2);
            let b = fixture.evaluate(reversed);
            let common = fixture.evaluate(forward.map(|v| v + 0.125));
            assert!((a.current + b.current).abs() < 1e-13 * a.current.abs() + 1e-18);
            assert!((a.current - common.current).abs() < 1e-13 * a.current.abs() + 1e-18);
            let swap = [0, 2, 1, 3];
            for i in 0..4 {
                let q = a.charge[i];
                assert!((q - b.charge[swap[i]]).abs() < 1e-13 * q.abs() + 1e-28);
                assert!((q - common.charge[i]).abs() < 1e-13 * q.abs() + 1e-28);
                assert!(
                    (a.current_gradient[i] + b.current_gradient[swap[i]]).abs()
                        < 1e-12 * a.current_gradient[i].abs() + 1e-18
                );
                for j in 0..4 {
                    let c = a.capacitance[i][j];
                    assert!((c - b.capacitance[swap[i]][swap[j]]).abs() < 1e-12 * c.abs() + 1e-28);
                    assert!((c - common.capacitance[i][j]).abs() < 1e-12 * c.abs() + 1e-28);
                }
            }
            assert!(a.capacitance[0][0] > 0.);
            let zero = fixture.evaluate([0.9 * pol, 0., 0., -0.2 * pol]);
            assert_eq!(zero.current, 0.);
            assert!(zero.current_gradient[1] > 0.);
            assert_eq!(zero.current_gradient[0], 0.);
            assert_eq!(zero.current_gradient[3], 0.);
            for drain in [1e-8, 1e-12, 1e-16, 1e-18] {
                let p = fixture.evaluate([0.9 * pol, drain * pol, 0., -0.2 * pol]);
                let conductance = p.current / (drain * pol);
                assert!(
                    (conductance - zero.current_gradient[1]).abs()
                        < 1e-5 * zero.current_gradient[1],
                    "cap={cap} polarity={polarity} Vds={drain:e}: {conductance:e} vs {:e}",
                    zero.current_gradient[1]
                );
            }
        }
    }
}

#[test]
fn saturation_quadratic_has_regular_linear_limit() {
    for a in [0., 1e-20, 1e-12, 0.1, -0.1] {
        let b = 1_f64;
        let c = 0.125_f64;
        let root = super::SaturationRoot::new(&a, b, c).unwrap();
        let y = root.value;
        assert!((0.5 * a * y * y - b * y + c).abs() < 4. * f64::EPSILON);
        if a.abs() < 1e-15 {
            assert!((y - c / b).abs() < 4. * f64::EPSILON);
        }
        let h = 1e-5;
        let samples = [-h, h, -h * 0.5, h * 0.5];
        for (da, db, dc) in [(1_f64, 0_f64, 0_f64), (0., 1., 0.), (0., 0., 1.)] {
            let (expected, error) = richardson(
                samples.map(|d| {
                    super::SaturationRoot::new(
                        &da.mul_add(d, a),
                        db.mul_add(d, b),
                        dc.mul_add(d, c),
                    )
                    .unwrap()
                    .value
                }),
                h,
            );
            let derivative = root.derivative(&da, &db, &dc).unwrap();
            assert!((derivative - expected).abs() < 1e-9 * expected.abs() + error * 0.1 + 1e-12);
        }
    }
}

#[test]
fn depletion_charge_preserves_tiny_positive_charge_and_continuation() {
    let k1 = 0.4;
    for increment in [1e-3_f64, 1e-9, 1e-15, 1e-21, 1e-30] {
        let (charge, slope) = super::depletion_charge(k1, increment).unwrap();
        assert!(charge > 0.);
        let difference = charge / k1;
        assert!((difference * (k1 + difference) - increment).abs() < 8. * f64::EPSILON * increment);
        assert!(slope > 0. && slope <= 1.);
    }
    assert_eq!(super::depletion_charge(k1, 0.).unwrap(), (0., 1.));
    assert_eq!(super::depletion_charge(k1, -1e-30).unwrap(), (-1e-30, 1.));
    assert_eq!(super::depletion_charge(0., 1e-3).unwrap(), (0., 0.));
}

#[test]
fn channel_quadratic_is_regular_as_series_resistance_vanishes() {
    for polarity in [-1, 1] {
        let make = |resistance| {
            PhysicalFixture::prepare(
                Model {
                    r#type: polarity,
                    typeGiven: true,
                    rdsw: resistance,
                    rdswGiven: true,
                    rdswmin: 0.,
                    rdswminGiven: true,
                    a1: 0.,
                    a1Given: true,
                    a2: 1.,
                    a2Given: true,
                    ..Model::default()
                },
                300.15,
            )
        };
        let baseline = make(0.);
        for r in [1e-20, 1e-15, 1e-9] {
            let fixture = make(r);
            for bias in [
                [0.2, 0.1, 0., -0.1],
                [0.8, 0.3, 0., -0.2],
                [1.2, 1., 0., -0.4],
            ] {
                let v = bias.map(|x| x * f64::from(polarity));
                let a = baseline.evaluate(v);
                let b = fixture.evaluate(v);
                assert!(
                    (a.current - b.current).abs() < 1e-8 * a.current.abs() + 1e-25,
                    "polarity {polarity}, resistance {r:e}, bias {bias:?}"
                );
                for (expected, actual) in a.current_gradient.into_iter().zip(b.current_gradient) {
                    assert!((expected - actual).abs() < 1e-8 * expected.abs() + 1e-22);
                }
            }
        }
    }
}

#[test]
fn subthreshold_and_mobility_modes_have_independent_channel_derivatives() {
    let mut errors = Vec::new();
    for polarity in [-1, 1] {
        for mobility in 0..=6 {
            for temperature in [250., 400.] {
                let fixture = PhysicalFixture::prepare(
                    Model {
                        r#type: polarity,
                        typeGiven: true,
                        mobMod: mobility,
                        mobModGiven: true,
                        ..Model::default()
                    },
                    temperature,
                );
                for gate in [-0.4, -0.1, 0.2, 0.6, 1.1] {
                    let v = [gate, 0.1, 0., -0.2].map(|x| x * f64::from(polarity));
                    let p = fixture.evaluate(v);
                    for col in [0, 1, 3] {
                        let h = 1e-4;
                        let samples = [-h, h, -h * 0.5, h * 0.5].map(|delta| {
                            let mut x = v;
                            x[col] += delta;
                            fixture.evaluate(x).current
                        });
                        let (expected, error) = richardson(samples, h);
                        record_error(
                            &mut errors,
                            p.current_gradient[col],
                            expected,
                            1e-4 * expected.abs() + 0.2 * error + 1e-27,
                            || {
                                format!(
                                    "polarity {polarity}, mobility {mobility}, temperature {temperature}, gate {gate}, col {col}"
                                )
                            },
                        );
                    }
                }
            }
        }
    }
    assert!(
        errors.is_empty(),
        "{} errors:\n{}",
        errors.len(),
        errors.join("\n")
    );
}

#[test]
fn smooth_positive_retains_small_tail_and_slope() {
    // Rationalized result for x=-1e12, width²/4=0.0025.
    let (value, slope) = super::smooth_positive(-1e12, 0.0025).unwrap();
    assert!((value / 2.5e-15 - 1.).abs() < 1e-14);
    assert!((slope / 2.5e-27 - 1.).abs() < 1e-14);
    assert_eq!(super::smooth_positive(0., 0.).unwrap(), (0., 0.5));
    assert_eq!(super::smooth_positive(-2., 0.).unwrap(), (0., 0.));
    assert_eq!(super::smooth_positive(2., 0.).unwrap(), (2., 1.));
    for x in [-10., -0.2, 0., 0.2, 10.] {
        let (value, slope) = super::smooth_positive(x, 0.0025).unwrap();
        let h = 1e-4;
        let samples = [-h, h, -h * 0.5, h * 0.5]
            .map(|delta| super::smooth_positive(x + delta, 0.0025).unwrap().0);
        let (expected, error) = richardson(samples, h);
        assert!(value > 0.);
        assert!((slope - expected).abs() < 1e-8 * expected.abs() + error + 1e-12);
    }
}

fn junction_trace(temperature: f64, kind: u32, voltage: [f64; node::COUNT]) -> Trace {
    Trace {
        temperature,
        mode: kind | mode::MODEINITFLOAT,
        limiting: false,
        voltage,
        s0: [0.; state::COUNT],
        s1: [0.; state::COUNT],
        s2: [0.; state::COUNT],
        order: 0,
        values: 0,
        events: 0,
        matrix: [[0.; node::COUNT]; node::COUNT],
        rhs: [0.; node::COUNT],
    }
}

#[test]
fn junction_current_and_charge_keep_asymmetric_tiny_bias_terms() {
    use super::junction::{JunctionBias, charge, current};
    let mut fixture = PhysicalFixture::prepare(
        Model {
            dioMod: 1,
            dioModGiven: true,
            ..Model::default()
        },
        300.15,
    );
    let m = &mut fixture.model;
    let h = &mut fixture.instance;
    let p = &mut fixture.size;
    h.Aseff = 1e-12;
    h.Adeff = 2e-12;
    h.vjsmFwd = 1.;
    h.vjdmFwd = 1.;
    h.Pseff = 1e-6;
    h.Pdeff = 3e-6;
    h.nf = 1.;
    p.weffCJ = 2e-6;
    m.SjctTempSatCurDensity = 1.;
    m.DjctTempSatCurDensity = 3.;
    m.SjctSidewallTempSatCurDensity = 0.;
    m.DjctSidewallTempSatCurDensity = 0.;
    m.SjctGateSidewallTempSatCurDensity = 0.;
    m.DjctGateSidewallTempSatCurDensity = 0.;
    m.SunitAreaTempJctCap = 1.;
    m.DunitAreaTempJctCap = 2.;
    m.SunitLengthSidewallTempJctCap = 1e-6;
    m.DunitLengthSidewallTempJctCap = 2e-6;
    m.SunitLengthGateSidewallTempJctCap = 3e-6;
    m.DunitLengthGateSidewallTempJctCap = 4e-6;
    let mut trace = junction_trace(300.15, mode::MODEDCOP, [0.; node::COUNT]);
    let source_slope = 1e-12 / (m.vtm * m.SjctEmissionCoeff) + trace.gmin();
    let drain_slope = 6e-12 / (m.vtm * m.DjctEmissionCoeff) + trace.gmin();
    for sign in [-1., 1.] {
        let voltage = sign * 1e-18;
        let bias = JunctionBias {
            source: voltage,
            drain: -voltage,
        };
        current(m, h, p, &trace, bias).unwrap();
        charge(m, h, p, &mut trace, bias).unwrap();
        assert!((h.cbs / (source_slope * voltage) - 1.).abs() < 16. * f64::EPSILON);
        assert!((h.cbd / (-drain_slope * voltage) - 1.).abs() < 16. * f64::EPSILON);
        assert!((h.gbs / source_slope - 1.).abs() < 16. * f64::EPSILON);
        assert!((h.gbd / drain_slope - 1.).abs() < 16. * f64::EPSILON);
        let source_cap = 8e-12;
        let drain_cap = 18e-12;
        assert!(
            (trace.state0(state::QBS) / (source_cap * voltage) - 1.).abs() < 16. * f64::EPSILON
        );
        assert!(
            (trace.state0(state::QBD) / (-drain_cap * voltage) - 1.).abs() < 16. * f64::EPSILON
        );
        assert!((h.capbs / source_cap - 1.).abs() < 16. * f64::EPSILON);
        assert!((h.capbd / drain_cap - 1.).abs() < 16. * f64::EPSILON);
    }
}

fn assert_conserved(values: &[f64], label: &str) {
    let total: f64 = values.iter().sum();
    let magnitude: f64 = values.iter().map(|x| x.abs()).sum();
    assert!(
        total.abs() <= 256. * f64::EPSILON * magnitude + 1e-26,
        "{label}: residual={total:e}, magnitude={magnitude:e}, values={values:?}"
    );
}

#[test]
fn full_nonlinear_stamps_conserve_physical_rows_and_common_mode_with_aliases() {
    let q = usize::try_from(node::Q).unwrap();
    for polarity in [-1, 1] {
        for (gate, body, nqs, rds) in [(0, 0, 0, 0), (2, 1, 0, 1), (3, 2, 1, 1)] {
            let model = Model {
                r#type: polarity,
                typeGiven: true,
                igcMod: 1,
                igcModGiven: true,
                igbMod: 1,
                igbModGiven: true,
                rgateMod: gate,
                rgateModGiven: true,
                rbodyMod: body,
                rbodyModGiven: true,
                trnqsMod: nqs,
                trnqsModGiven: true,
                rdsMod: rds,
                rdsModGiven: true,
                agidl: 1e-8,
                agidlGiven: true,
                bgidl: 1e9,
                bgidlGiven: true,
                agisl: 2e-8,
                agislGiven: true,
                bgisl: 1e9,
                bgislGiven: true,
                ..Model::default()
            };
            let instance = Instance {
                l: 0.5e-6,
                lGiven: true,
                w: 2e-6,
                wGiven: true,
                sourceArea: 1e-12,
                sourceAreaGiven: true,
                drainArea: 2e-12,
                drainAreaGiven: true,
                sourcePerimeter: 2e-6,
                sourcePerimeterGiven: true,
                drainPerimeter: 3e-6,
                drainPerimeterGiven: true,
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
            for (drain, source) in [(0.6, 0.), (-0.2, 0.1), (1e-12, 0.)] {
                for kind in [mode::MODEDCOP, mode::MODETRAN] {
                    let p = f64::from(polarity);
                    let voltage = [
                        drain, 0.9, source, -0.2, drain, 0.89, 0.9, source, -0.2, -0.21, -0.19, 0.,
                    ]
                    .map(|x| p * x);
                    let mut trace = junction_trace(300.15, kind, voltage);
                    let mut instance = instance.clone();
                    super::load(&model, &mut instance, &size, &mut trace).unwrap();
                    let label = format!(
                        "p={polarity} gate={gate} body={body} nqs={nqs} drain={drain} kind={kind}"
                    );
                    assert_conserved(&trace.rhs[..q], &format!("{label} RHS KCL"));
                    for col in 0..node::COUNT {
                        let column: Vec<_> = trace.matrix[..q].iter().map(|r| r[col]).collect();
                        assert_conserved(&column, &format!("{label} column {col}"));
                    }
                    for (row, values) in trace.matrix.iter().enumerate() {
                        assert_conserved(&values[..q], &format!("{label} common mode row {row}"));
                    }
                }
            }
        }
    }
}

#[test]
fn channel_and_intrinsic_charge_share_physical_trajectory_and_partial_series() -> SpiceResult<()> {
    use crate::maths::taylor::{Jet, differential::Differential};

    fn at(jet: &Jet, t: f64) -> f64 {
        jet.coefficients()
            .iter()
            .rev()
            .fold(0., |sum: f64, coefficient| sum.mul_add(t, *coefficient))
    }

    fn assert_close(actual: f64, expected: f64, absolute: f64, context: &str) {
        assert!(
            actual.is_finite()
                && expected.is_finite()
                && (actual - expected).abs() <= absolute + 2e-6 * actual.abs().max(expected.abs()),
            "{context}: {actual:e} vs {expected:e}"
        );
    }

    for polarity in [-1, 1] {
        for mobility in 0..=6 {
            for cap in 0..=2 {
                let fixture = PhysicalFixture::prepare(
                    Model {
                        r#type: polarity,
                        typeGiven: true,
                        mobMod: mobility,
                        mobModGiven: true,
                        capMod: cap,
                        capModGiven: true,
                        cvchargeMod: mobility % 2,
                        cvchargeModGiven: true,
                        xpart: [0., 0.5, 1.][mobility as usize % 3],
                        xpartGiven: true,
                        ..Model::default()
                    },
                    300.15,
                );
                for initial in [
                    [0.35, 0.05, -0.3],
                    [1.2, 0.1, -0.2],
                    [1.4, 1.1, -0.3],
                    [1.2, -0.3, -0.7],
                ] {
                    let rates = [0.002, -0.001, 0.0007];
                    let variables = [0, 1, 2].map(|i| {
                        Differential::<3>::variable(i, Jet::affine(8, initial[i], rates[i]))
                            .unwrap()
                    });
                    let [vg, vd, vb] = variables;
                    let mut instance = fixture.instance.clone();
                    let e = super::channel::electrostatics_constitutive(
                        &fixture.model,
                        &mut instance,
                        &fixture.size,
                        fixture.temperature,
                        vb.clone(),
                        vg.clone(),
                        vd.clone(),
                    )?;
                    let transport = super::channel::transport_constitutive(
                        &fixture.model,
                        &mut instance,
                        &fixture.size,
                        [vb, vg, vd.clone(), vd, Differential::constant(8, 0.)],
                        &e,
                    )?;
                    let charge = super::charge::intrinsic_constitutive(
                        &fixture.model,
                        &mut instance,
                        &fixture.size,
                        true,
                        &e,
                        &transport,
                    )?;
                    let laws = [
                        transport.unscaled_drain_current,
                        charge.qgate,
                        charge.qbulk,
                        charge.qdrn,
                    ];
                    let scalar = |[vg, vd, vb]: [f64; 3]| -> SpiceResult<[f64; 4]> {
                        let mut instance = fixture.instance.clone();
                        let e = super::channel::electrostatics_constitutive(
                            &fixture.model,
                            &mut instance,
                            &fixture.size,
                            fixture.temperature,
                            vb,
                            vg,
                            vd,
                        )?;
                        let tr = super::channel::transport_constitutive(
                            &fixture.model,
                            &mut instance,
                            &fixture.size,
                            [vb, vg, vd, vd, 0.],
                            &e,
                        )?;
                        let q = super::charge::intrinsic_constitutive(
                            &fixture.model,
                            &mut instance,
                            &fixture.size,
                            true,
                            &e,
                            &tr,
                        )?;
                        Ok([tr.unscaled_drain_current, q.qgate, q.qbulk, q.qdrn])
                    };
                    for t in [0., 0.25, 0.5] {
                        let sample = std::array::from_fn(|i| rates[i].mul_add(t, initial[i]));
                        let values = scalar(sample)?;
                        for (row, law) in laws.iter().enumerate() {
                            let context = format!(
                                "pol={polarity} mob={mobility} cap={cap} bias={sample:?} row={row}"
                            );
                            let absolute = if row == 0 { 1e-14 } else { 1e-23 };
                            assert_close(at(&law.value, t), values[row], absolute, &context);
                            for column in 0..3 {
                                let h = 1e-4;
                                let samples = [-h, h, -0.5 * h, 0.5 * h].map(|delta| {
                                    let mut point = sample;
                                    point[column] += delta;
                                    scalar(point).unwrap()[row]
                                });
                                let (reference, error) = richardson(samples, h);
                                let actual =
                                    law.gradient[column].as_ref().map_or(0., |jet| at(jet, t));
                                assert_close(
                                    actual,
                                    reference,
                                    absolute + 0.1 * error,
                                    &format!("{context} col={column}"),
                                );
                            }
                        }
                    }
                }
            }
        }
    }
    Ok(())
}
