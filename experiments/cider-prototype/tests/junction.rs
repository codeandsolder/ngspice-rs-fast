//! Numerical feasibility regressions independent of the production engine.
use cider_junction_prototype::{C, Model, State};

fn equilibrium(model: &Model) -> State {
    model
        .solve(model.initial(300.).unwrap(), 0., 300., None)
        .unwrap()
        .0
}

fn ramp(model: &Model, mut state: State, target: f64) -> State {
    let start = state.bias();
    let steps = ((target - start).abs() / 0.025).ceil() as usize;
    for k in 1..=steps {
        state = model
            .solve(
                state,
                start + (target - start) * k as f64 / steps as f64,
                300.,
                None,
            )
            .unwrap()
            .0;
    }
    state
}

#[test]
fn c_oracle_dc_and_conservation() {
    let model = Model::abrupt(201, false);
    let eq = equilibrium(&model);
    let samples: Vec<(f64, f64)> = include_str!("../reference/abrupt-pn-dc.csv")
        .lines()
        .skip(1)
        .map(|line| {
            let (v, i) = line.split_once(',').unwrap();
            (v.parse().unwrap(), i.parse().unwrap())
        })
        .collect();
    assert_eq!(samples.len(), 39);
    for forward in [false, true] {
        let mut state = eq.clone();
        let mut side: Vec<_> = samples
            .iter()
            .copied()
            .filter(|(v, _)| v.abs() > 1e-12 && (*v > 0.) == forward)
            .collect();
        side.sort_by(|a, b| a.0.abs().total_cmp(&b.0.abs()));
        for (v, expected) in side {
            state = ramp(&model, state, v);
            let currents = model.currents(&state, None).unwrap();
            let center = currents[currents.len() / 2];
            assert!(
                (center - expected).abs() <= expected.abs() * 1e-7,
                "bias {v}: {center:e} vs {expected:e}"
            );
            assert!(currents.iter().all(|i| (i - center).abs() < 1e-13));
            let power = model.dissipation(&state).unwrap();
            assert!((power - v * center).abs() < 1e-12);
        }
    }
    assert!(
        model
            .currents(&eq, None)
            .unwrap()
            .iter()
            .all(|i| i.abs() < 1e-18)
    );
}

#[test]
fn homogeneous_analytic_admittance_on_nonuniform_mesh() {
    let positions = Model::abrupt(101, true).positions().to_vec();
    for doping in [-1e22, 0., 1e22] {
        let m = Model::new(positions.clone(), vec![doping; positions.len()]).unwrap();
        let state = ramp(&m, equilibrium(&m), 0.3);
        let dynamic = m.prepare_admittance(&state).unwrap();
        for freq in [0., 1., 1e4, 1e8, 1e12, 1e14] {
            let p = C::new(0., std::f64::consts::TAU * freq);
            let exact = m.uniform_admittance(doping, p);
            let (actual, spatial) = dynamic.evaluate(p).unwrap();
            assert!(
                (actual - exact).abs() / exact.abs() < 1e-9,
                "doping {doping} frequency {freq}"
            );
            assert!(spatial / exact.abs() < 1e-9);
        }
    }
}

#[test]
fn terminal_derivative_and_dynamic_be_agree() {
    let m = Model::abrupt(101, true);
    let accepted = ramp(&m, equilibrium(&m), 0.5);
    for dt in [1e-8, 1e-10, 1e-12] {
        let dynamic = m
            .prepare_admittance(&accepted)
            .unwrap()
            .evaluate(C::new(1. / dt, 0.))
            .unwrap()
            .0
            .re;
        let be = m
            .terminal_stamp(&accepted, Some((&accepted, dt)))
            .unwrap()
            .1;
        assert!((dynamic - be).abs() / be.abs() < 1e-9);
    }
    let dt = 2e-12;
    let bias = 0.49;
    let old = Some((&accepted, dt));
    let trial = m.solve(accepted.clone(), bias, 300., old).unwrap().0;
    let (current, g) = m.terminal_stamp(&trial, old).unwrap();
    let h = 1e-6;
    let plus = m.solve(trial.clone(), bias + h, 300., old).unwrap().0;
    let minus = m.solve(trial.clone(), bias - h, 300., old).unwrap().0;
    let fd = (m.currents(&plus, old).unwrap()[0] - m.currents(&minus, old).unwrap()[0]) / (2. * h);
    assert!((g - fd).abs() / fd.abs() < 1e-7);
    assert!(
        m.currents(&trial, old)
            .unwrap()
            .iter()
            .all(|i| (i - current).abs() < 1e-11)
    );
    assert_eq!(accepted.bias(), 0.5);
}

#[test]
fn series_mna_and_thermal_balance() {
    let m = Model::abrupt(101, true);
    let eq = equilibrium(&m);
    let (state, i, _, _) = m.series(&eq, 0.7, 1000., None).unwrap();
    assert!((i + (state.bias() - 0.7) / 1000.).abs() < 1e-10);
    let hot = ramp(&m, eq, 0.6);
    let (next, _, _, balance) = m.heat_step(&hot, 0.6, 1e-8, 300., 500., 2e-9).unwrap();
    assert!(next.temperature() > 300.);
    assert!(balance.abs() < 1e-8);
    assert_eq!(hot.temperature(), 300.);
}

#[test]
fn invalid_geometry_state_and_step_are_rejected() {
    assert!(Model::new(vec![0., 1., 1.], vec![0.; 3]).is_err());
    assert!(Model::new(vec![0., 1., 2.], vec![0., f64::NAN, 0.]).is_err());
    let a = Model::abrupt(11, false);
    let b = Model::abrupt(11, true);
    let state = equilibrium(&a);
    assert!(a.initial(0.).is_err());
    assert!(b.solve(state.clone(), 0., 300., None).is_err());
    assert!(b.currents(&state, None).is_err());
    assert!(
        a.solve(state.clone(), 0., 300., Some((&state, 0.)))
            .is_err()
    );
    assert!(
        a.prepare_admittance(&state)
            .unwrap()
            .evaluate(C::new(f64::NAN, 0.))
            .is_err()
    );
}

#[test]
fn c_oracle_complex_admittance() {
    let model = Model::abrupt(201, false);
    let eq = equilibrium(&model);
    let mut rows = 0;
    for bias in [-0.3, 0., 0.3, 0.55] {
        let state = ramp(&model, eq.clone(), bias);
        let prepared = model.prepare_admittance(&state).unwrap();
        for line in include_str!("../reference/abrupt-pn-ac.csv")
            .lines()
            .skip(1)
        {
            let fields: Vec<f64> = line.split(',').map(|v| v.parse().unwrap()).collect();
            if fields[0] != bias {
                continue;
            }
            let expected = C::new(fields[2], fields[3]);
            let (actual, spatial) = prepared
                .evaluate(C::new(0., std::f64::consts::TAU * fields[1]))
                .unwrap();
            // CIDER's reverse-bias low-frequency conductance has documented
            // cancellation; independent homogeneous and BE tests use tighter bounds.
            let tolerance = if bias <= 0. { 1e-4 } else { 1e-7 };
            assert!(
                (actual - expected).abs() / expected.abs() < tolerance,
                "bias {bias} frequency {}: {actual:?} vs {expected:?}",
                fields[1]
            );
            assert!(spatial / actual.abs() < 1e-9);
            rows += 1;
        }
    }
    assert_eq!(rows, 564);
}
