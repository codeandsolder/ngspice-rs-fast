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

fn richardson(mut f: impl FnMut(f64) -> f64, h: f64) -> (f64, f64) {
    let coarse = (f(h) - f(-h)) / (2. * h);
    let fine = (f(h * 0.5) - f(-h * 0.5)) / h;
    ((4. * fine - coarse) / 3., (fine - coarse).abs())
}

#[test]
fn physical_channel_and_charge_partials_against_richardson() {
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
                        let p = fixture.evaluate(v);
                        let context = format!(
                            "pol={polarity} cap={cap} cv={cv} xpart={partition} bias={normalized:?}"
                        );
                        let q_scale = p.charge.iter().map(|x| x.abs()).fold(0., f64::max);
                        assert!(
                            p.charge.iter().sum::<f64>().abs() <= q_scale * 8. * f64::EPSILON,
                            "{context}: charge KCL"
                        );
                        for row in &p.capacitance {
                            let scale = row.iter().map(|x| x.abs()).sum::<f64>();
                            assert!(
                                row.iter().sum::<f64>().abs() <= scale * 8. * f64::EPSILON,
                                "{context}: common mode"
                            );
                        }
                        for col in 0..4 {
                            let column = p.capacitance.map(|row| row[col]);
                            let scale = column.iter().map(|x| x.abs()).sum::<f64>();
                            assert!(
                                column.iter().sum::<f64>().abs() <= scale * 8. * f64::EPSILON,
                                "{context}: charge Jacobian KCL"
                            );
                            let sample = |delta| {
                                let mut shifted = v;
                                shifted[col] += delta;
                                fixture.evaluate(shifted)
                            };
                            let (reference, error) =
                                richardson(|delta| sample(delta).current, 1e-4);
                            let actual = p.current_gradient[col];
                            assert!(
                                (actual - reference).abs()
                                    <= 3e-5 * reference.abs() + error * 0.1 + 1e-13,
                                "{context}: current col {col}: {actual:e} vs {reference:e}, truncation {error:e}"
                            );
                            for row in 0..4 {
                                let (reference, error) =
                                    richardson(|delta| sample(delta).charge[row], 1e-4);
                                let actual = p.capacitance[row][col];
                                assert!(
                                    (actual - reference).abs()
                                        <= 3e-5 * reference.abs() + error * 0.1 + 1e-21,
                                    "{context}: charge row {row} col {col}: {actual:e} vs {reference:e}, truncation {error:e}"
                                );
                            }
                        }
                    }
                }
            }
        }
    }
}
