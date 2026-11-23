use super::*;
use crate::devices::traits::MnaUnknowns;
use crate::maths::Vector;
use crate::primitives::NodeTable;

#[test]
fn rejected_trials_are_pure_and_the_clamp_survives_pruning() {
    let mut nodes = NodeTable::new();
    let terminals = [nodes.intern("in").unwrap(), nodes.intern("out").unwrap()];
    let mut unknowns = MnaUnknowns::new();
    unknowns.rebuild(&nodes);
    let solution = Vector::zeros(4);
    let device = Txl {
        name: "y1".into(),
        terminals,
        instance_length: None,
        resistance: 1.,
        coefficients: super::super::pade::Pade::new(1., 250e-9, 0., 100e-12, 1.).unwrap(),
    };
    let tau = device.coefficients.delay_ps * 1e-12;
    let mut context = DelayContext {
        solution: &solution,
        unknowns: &unknowns,
        time: 0.,
        steps: [tau; 3],
        min_break: 0.,
        reltol: 1e-3,
        abstol: 1e-12,
        initial_conditions: false,
        branches: 2..4,
    };
    let mut history = device.new_history();
    history.apply(&device.start(&context).unwrap()).unwrap();
    let before = history.clone();
    context.time = 0.5e-12;
    history
        .apply(&device.accept(&history, &context).unwrap())
        .unwrap();
    assert_eq!(
        history, before,
        "sub-picosecond acceptance does not advance TXL"
    );
    context.time = 0.;
    assert_eq!(device.timestep_limit(&history, &context).unwrap(), None);
    let first = device
        .transient_stamp(&history, 2. * tau, 2. * tau, 1e-12)
        .unwrap();
    device
        .transient_stamp(&history, 1.5 * tau, 1.5 * tau, 1e-12)
        .unwrap();
    assert_eq!(history, before, "a rejected load cannot change history");
    assert_eq!(
        first,
        device
            .transient_stamp(&history, 2. * tau, 2. * tau, 1e-12)
            .unwrap()
    );
    assert_eq!(device.timestep_limit(&history, &context).unwrap(), None);

    context.time = 2. * tau;
    context.steps[0] = 2. * tau;
    history
        .apply(&device.accept(&history, &context).unwrap())
        .unwrap();
    assert_eq!(
        device.timestep_limit(&history, &context).unwrap(),
        Some(0.9 * device.coefficients.delay_ps * 1e-12)
    );
    for _ in 0..10 {
        context.time += 0.8 * tau;
        context.steps[0] = 0.8 * tau;
        history
            .apply(&device.accept(&history, &context).unwrap())
            .unwrap();
    }
    assert!(
        history.as_txl().unwrap().samples.front().unwrap().tick.ps() > 2. * tau * 1e12,
        "the triggering gap is gone"
    );
    assert_eq!(
        device.timestep_limit(&history, &context).unwrap(),
        Some(0.9 * device.coefficients.delay_ps * 1e-12)
    );
    history.apply(&device.start(&context).unwrap()).unwrap();
    assert_eq!(
        device.timestep_limit(&history, &context).unwrap(),
        None,
        "restart resets the clamp"
    );
}

#[test]
fn picosecond_history_has_explicit_range_checks() {
    assert_eq!(Tick::from_seconds(1.9e-12).unwrap(), Tick(1));
    assert_eq!(Tick::from_seconds(0.).unwrap(), Tick(0));
    for time in [-1., 0.003, f64::INFINITY, f64::NAN] {
        assert!(Tick::from_seconds(time).is_err(), "{time}");
    }
}

#[test]
fn last_fractional_picosecond_fits_c_integer_cast() {
    let max = Real::from(i32::MAX);
    assert_eq!(
        Tick::from_seconds((max + 0.75) / 1e12).unwrap(),
        Tick(i32::MAX)
    );
    assert!(Tick::from_seconds((max + 1.) / 1e12).is_err());
    assert_eq!(Tick::from_seconds(0.99e-12).unwrap(), Tick(0));
    let min = Real::from(i32::MIN);
    assert_eq!(
        Tick::from_seconds((min - 0.75) / 1e12).unwrap(),
        Tick(i32::MIN)
    );
    assert!(Tick::from_seconds((min - 1.) / 1e12).is_err());
    assert_eq!(Tick::from_seconds(-1.9e-12).unwrap(), Tick(-1));
    assert!(Tick::accepted(-1.9e-12).is_err());
}

fn sample(tick: i32) -> Sample {
    let t = Real::from(tick);
    Sample {
        tick: Tick(tick),
        values: TerminalValues {
            voltage: [t, 2. * t],
            current: [-t, 3. * t],
        },
    }
}

fn empty_state() -> AcceptedState {
    AcceptedState {
        prehistory: [7., 8.],
        convolution: None,
        clamped: false,
    }
}

#[test]
fn wrapped_history_retains_exact_delay_brackets_and_prehistory() {
    let mut history = TxlHistory::default();
    let state = empty_state();
    let reset = TxlUpdate(Change::Reset {
        sample: sample(0),
        state,
    });
    history.check(&reset).unwrap();
    history.apply_validated(&reset);
    for tick in (10..=10_000).step_by(10) {
        let update = TxlUpdate(Change::Append {
            drop_front: history.drop_before(Real::from(tick) - 35.),
            sample: sample(tick),
            state,
        });
        history.check(&update).unwrap();
        history.apply_validated(&update);
    }
    assert_eq!(history.samples.front().unwrap().tick, Tick(9960));
    assert_eq!(history.drop_before(9970.), 0);
    assert_eq!(history.drop_before(9970.5), 1);
    let previous = Tick(10_000);
    let now = Tick(10_020);
    let (before, ratio) = history.delayed(0., previous, now).unwrap();
    assert_eq!(
        before,
        TerminalValues {
            voltage: [7., 8.],
            current: [0.; 2]
        }
    );
    assert_eq!(ratio, 0.);
    let (exact, ratio) = history.delayed(9980., previous, now).unwrap();
    assert_eq!(exact, sample(9980).values);
    assert_eq!(ratio, 0.);
    let (interpolated, _) = history.delayed(9985.5, previous, now).unwrap();
    assert_eq!(interpolated.voltage, [9985.5, 19971.]);
    assert_eq!(interpolated.current, [-9985.5, 29956.5]);
    let (future, ratio) = history.delayed(10005., previous, now).unwrap();
    assert_eq!(ratio, 0.25);
    assert_eq!(future, sample(10_000).values.map(|v| v * 0.75));

    let bad = TxlUpdate(Change::Append {
        drop_front: 0,
        sample: sample(10_000),
        state,
    });
    let before = history.clone();
    assert!(history.check(&bad).is_err());
    assert_eq!(history, before);
}

#[test]
fn recurrence_conjugate_slots_keep_c_initialization_update_and_sum_order() {
    use super::super::pade::ComplexTerm;
    let term = Term { c: 0.25, x: -2. };
    let model = Propagation::<3, 1>::Conjugate {
        real: [term],
        pair: ComplexTerm {
            residue: Complex::new(0.5, 0.125),
            pole: Complex::new(-3., 4.),
        },
    };
    let voltage = [1.25, -0.75];
    let old_pair = voltage.map(|v| Complex::new(-v * 0.5 / -3., -v * 0.125 / 4.));
    let mut state = PropagationState::initial(&model, Some(voltage));
    assert_eq!(
        state,
        PropagationState::Conjugate {
            real: [dc_term(term, voltage)],
            pair: old_pair
        }
    );
    let before = [0.25, 0.5];
    let after = [0.75, 1.25];
    let dt: Real = 0.125;
    let exp = (-3. * dt).exp();
    let exp = Complex::new(exp * (4. * dt).cos(), exp * (4. * dt).sin());
    let residue = Complex::new(0.5 * dt * 0.5, 0.5 * dt * 0.125);
    let pair: [Complex; 2] = std::array::from_fn(|k| {
        old_pair[k] * exp
            + residue * Complex::new(before[k] * exp.re + after[k], before[k] * exp.im)
    });
    let exp = (-2. * dt).exp();
    let real: [Real; 2] = std::array::from_fn(|k| {
        dc_term(term, voltage)[k] * exp + 0.5 * dt * term.c * (before[k] * exp + after[k])
    });
    let expected: [Real; 2] = std::array::from_fn(|k| real[k] + 2. * pair[k].re);
    assert_eq!(
        state
            .advance(&model, before, after, dt)
            .unwrap()
            .map(Real::to_bits),
        expected.map(Real::to_bits)
    );
    assert_eq!(state, PropagationState::Conjugate { real: [real], pair });
    state.correct(&model, after, 3., 0.25).unwrap();
    let correction = |v, c| 3. * 0.5e-12 * 0.25 * v * c;
    let real = std::array::from_fn(|k| real[k] + correction(after[k], term.c));
    let pair = std::array::from_fn(|k| {
        Complex::new(
            pair[k].re + correction(after[k], 0.5),
            pair[k].im + correction(after[k], 0.125),
        )
    });
    assert_eq!(state, PropagationState::Conjugate { real: [real], pair });
}

// Run explicitly with --ignored --nocapture; elapsed times are diagnostic,
// never acceptance thresholds. The former 31-value history is represented
// using the actual continuous waveform storage used before this refactor.
#[test]
#[ignore = "TXL long-history comparison"]
fn long_history_storage_and_search_comparison() {
    use crate::devices::delay::{WaveformHistory, WaveformUpdate};
    use std::hint::black_box;
    use std::time::Instant;

    const COUNT: i32 = 100_000;
    let mut old = WaveformHistory::new(31);
    let start = Instant::now();
    for i in 0..COUNT {
        let sample = sample(i);
        let mut values = vec![0.; 31];
        values[..2].copy_from_slice(&sample.values.voltage);
        values[2..4].copy_from_slice(&sample.values.current);
        old.apply(&WaveformUpdate {
            append: Some(((Real::from(i) + 0.25) / 1e12, values)),
            ..WaveformUpdate::default()
        })
        .unwrap();
    }
    let old_build = start.elapsed();
    let mut new = TxlHistory::default();
    let state = empty_state();
    let start = Instant::now();
    for i in 0..COUNT {
        let mut sample = sample(i);
        sample.tick = Tick::from_seconds((Real::from(i) + 0.25) / 1e12).unwrap();
        let change = if i == 0 {
            Change::Reset { sample, state }
        } else {
            Change::Append {
                drop_front: 0,
                sample,
                state,
            }
        };
        let update = TxlUpdate(change);
        new.check(&update).unwrap();
        new.apply_validated(&update);
    }
    let new_build = start.elapsed();
    let start = Instant::now();
    let mut old_sum = 0.;
    for i in 1..COUNT {
        let target = Real::from((i * 11) % (COUNT - 1)) + 0.25;
        let ticks = |index| (old.time(index).unwrap() * 1e12).trunc();
        let mut lo = 0;
        let mut hi = old.len() - 1;
        while lo < hi {
            let mid = (lo + hi).div_ceil(2);
            if ticks(mid) <= target {
                lo = mid;
            } else {
                hi = mid - 1;
            }
        }
        let left = std::array::from_fn::<Real, 31, _>(|k| old.value(lo, k).unwrap());
        let right = std::array::from_fn::<Real, 31, _>(|k| old.value(lo + 1, k).unwrap());
        let f = (target - ticks(lo)) / (ticks(lo + 1) - ticks(lo));
        old_sum += black_box(left[0] + f * (right[0] - left[0]));
    }
    let old_search = start.elapsed();
    let start = Instant::now();
    let mut new_sum = 0.;
    for i in 1..COUNT {
        let target = Real::from((i * 11) % (COUNT - 1)) + 0.25;
        let (value, _) = new.delayed(target, Tick(COUNT - 1), Tick(COUNT)).unwrap();
        new_sum += black_box(value.voltage[0]);
    }
    let new_search = start.elapsed();
    assert_eq!(old_sum.to_bits(), new_sum.to_bits());
    let old_bytes = old.len() * 32 * size_of::<Real>();
    let new_bytes = new.len() * size_of::<Sample>() + size_of::<AcceptedState>();
    assert!(new_bytes < old_bytes / 5);
    eprintln!(
        "retained payload: {old_bytes} -> {new_bytes} bytes; construction {old_build:?} -> {new_build:?}; search {old_search:?} -> {new_search:?}"
    );
}

#[test]
fn exact_delayed_tick_returns_stored_value_from_preceding_bracket() {
    // The bracket search retains C's strict comparison so the left sample
    // still precedes an exact tick. The interpolated endpoint must reproduce
    // the stored right value, not lose it by rounded cancellation.
    let mut left = sample(10);
    left.values.voltage[0] = 1e16;
    let mut right = sample(20);
    right.values.voltage[0] = 1.;
    let history = TxlHistory {
        samples: [left, right, sample(30)].into(),
        state: Some(empty_state()),
    };
    let (value, _) = history.delayed(20., Tick(30), Tick(40)).unwrap();
    assert_eq!(
        value.voltage[0].to_bits(),
        right.values.voltage[0].to_bits()
    );
    assert_ne!(value.voltage[0].to_bits(), 0.0_f64.to_bits());
}
