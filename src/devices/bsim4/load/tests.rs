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
