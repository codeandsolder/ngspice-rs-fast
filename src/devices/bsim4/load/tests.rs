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

/// Independent constitutive-law test: no Newton solve, device limiting,
/// companion integration, or external source-current subtraction.
struct PhysicalFixture {
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
        Self {
            model,
            instance,
            size,
        }
    }

    /// Physical terminal order: G, D, S, B.
    fn evaluate(&self, [g, d, s, b]: [f64; 4]) -> PhysicalPoint {
        let mut instance = self.instance.clone();
        let mut trace = Trace {
            mode: mode::MODEDCTRANCURVE | mode::MODEINITFLOAT,
            limiting: false,
            voltage: [d, g, s, b, d, g, g, s, b, b, b, 0.],
            s0: [0.; state::COUNT],
            s1: [0.; state::COUNT],
            s2: [0.; state::COUNT],
            order: 0,
            values: 0,
            events: 0,
        };
        let bias = super::bias::select(&self.model, &instance, &mut trace);
        let electro =
            super::channel::electrostatics(&self.model, &mut instance, &self.size, &trace, &bias);
        let transport =
            super::channel::transport(&self.model, &mut instance, &self.size, &bias, &electro);
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
        let p = super::smooth_drain(sat, drain, delta);
        assert!((p.effective - expected).abs() <= 8. * f64::EPSILON * expected);
        assert!((p.effective + p.drain_excess - drain).abs() <= 8. * f64::EPSILON * drain);
        assert!(p.saturation_partial > 0. && p.drain_partial > 0.);
    }
    let zero = super::smooth_drain(sat, 0., delta);
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
        let p = super::smooth_drain(sat, drain, delta);
        let h = 1e-5 * sat.min(drain);
        let steps = [-h, h, -h * 0.5, h * 0.5];
        let (ds, es) = richardson(
            steps.map(|d| super::smooth_drain(sat + d, drain, delta).effective),
            h,
        );
        let (dd, ed) = richardson(
            steps.map(|d| super::smooth_drain(sat, drain + d, delta).effective),
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
        let root = super::SaturationRoot::new(a, b, c);
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
                    super::SaturationRoot::new(da.mul_add(d, a), db.mul_add(d, b), dc.mul_add(d, c))
                        .value
                }),
                h,
            );
            let derivative = root.derivative(da, db, dc);
            assert!((derivative - expected).abs() < 1e-9 * expected.abs() + error * 0.1 + 1e-12);
        }
    }
}

#[test]
fn depletion_charge_preserves_tiny_positive_charge_and_continuation() {
    let k1 = 0.4;
    for increment in [1e-3_f64, 1e-9, 1e-15, 1e-21, 1e-30] {
        let (charge, slope) = super::depletion_charge(k1, increment);
        assert!(charge > 0.);
        let difference = charge / k1;
        assert!((difference * (k1 + difference) - increment).abs() < 8. * f64::EPSILON * increment);
        assert!(slope > 0. && slope <= 1.);
    }
    assert_eq!(super::depletion_charge(k1, 0.), (0., 1.));
    assert_eq!(super::depletion_charge(k1, -1e-30), (-1e-30, 1.));
    assert_eq!(super::depletion_charge(0., 1e-3), (0., 0.));
}
