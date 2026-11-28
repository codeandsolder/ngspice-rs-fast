use cider_junction_prototype::{C, Model, State};
use std::time::Instant;

fn report(
    model: &Model,
    state: &State,
    old: Option<(&State, f64)>,
    iters: usize,
    res: f64,
) -> Result<(), String> {
    let currents = model.currents(state, old)?;
    let center = currents[currents.len() / 2];
    let error = currents
        .iter()
        .map(|i| (i - center).abs())
        .fold(0., f64::max);
    let (qn, qp, qleft) = model.stored(state)?;
    let e = model.midpoint_field(state)?;
    println!(
        "{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.9e},{},{:.3e},{:.12e},{:.12e}",
        state.bias(),
        center,
        error,
        qn,
        qp,
        qleft,
        state.temperature(),
        iters,
        res,
        model.dissipation(state)?,
        e
    );
    Ok(())
}
fn main() -> Result<(), String> {
    let args: Vec<_> = std::env::args().collect();
    let nodes = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(201);
    let mode = args.get(2).map(String::as_str).unwrap_or("dc");
    if nodes < 3 || nodes % 2 != 1 {
        return Err("mesh needs at least three, odd number of nodes".into());
    }
    let m = Model::abrupt(nodes, args.iter().any(|s| s == "clustered"));
    let start = Instant::now();
    let ambient = 300.;
    let (eq, n, r) = m.solve(m.initial(ambient)?, 0., ambient, None)?;
    println!(
        "bias_V,current_A,max_spatial_current_error_A,electron_charge_C,hole_charge_C,left_field_charge_C,temp_K,newton_iters,scaled_residual,dissipation_W,junction_field_V_m"
    );
    report(&m, &eq, None, n, r)?;
    if mode == "dc" {
        for range in [(1, -12i32, -1), (1, 26, 1)] {
            let mut state = eq.clone();
            for k in 1..=range.1.abs() {
                let v = 0.025 * f64::from(k * range.2);
                let (next, n, r) = m.solve(state, v, 300., None)?;
                report(&m, &next, None, n, r)?;
                state = next;
            }
        }
    } else if mode == "ac-tran" {
        let target = args
            .get(3)
            .and_then(|s| s.parse::<f64>().ok())
            .unwrap_or(0.3);
        let frequency = args
            .get(4)
            .and_then(|s| s.parse::<f64>().ok())
            .unwrap_or(1e9);
        let steps_per_cycle = args
            .get(5)
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(128);
        let cycles = args
            .get(6)
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(6);
        let amplitude = args
            .get(7)
            .and_then(|s| s.parse::<f64>().ok())
            .unwrap_or(1e-5);
        let mut state = eq;
        let steps = (target.abs() / 0.025).ceil() as usize;
        for k in 1..=steps {
            state = m
                .solve(state, target * k as f64 / steps as f64, 300., None)?
                .0;
        }
        if !frequency.is_finite()
            || frequency <= 0.
            || steps_per_cycle < 4
            || cycles == 0
            || !amplitude.is_finite()
            || amplitude == 0.
        {
            return Err("invalid AC transient parameters".into());
        }
        let dt = 1. / (frequency * steps_per_cycle as f64);
        let dphi = std::f64::consts::TAU / steps_per_cycle as f64;
        let dynamic = m.prepare_admittance(&state)?;
        let (expected, _) =
            dynamic.evaluate(C::new(2. * (dphi / 2.).sin().powi(2) / dt, dphi.sin() / dt))?;
        let (continuous, _) = dynamic.evaluate(C::new(0., std::f64::consts::TAU * frequency))?;
        let mut phasors = Vec::with_capacity(cycles);
        let mut acc = C::new(0., 0.);
        for k in 1..=cycles * steps_per_cycle {
            let phase = dphi * k as f64;
            let bias = target + amplitude * phase.cos();
            let next = m.solve(state.clone(), bias, 300., Some((&state, dt)))?.0;
            let i = m.currents(&next, Some((&state, dt)))?[nodes / 2];
            acc = acc + C::new(phase.cos(), -phase.sin()) * i;
            state = next;
            if k % steps_per_cycle == 0 {
                phasors.push(acc * (2. / (amplitude * steps_per_cycle as f64)));
                acc = C::new(0., 0.);
            }
        }
        for (i, y) in phasors.into_iter().enumerate() {
            println!(
                "time_ac bias={target} f={frequency:.6e} step={dt:.6e} amp={amplitude:.6e} cycle={} real={:.12e} imag={:.12e} be_relerr={:.6e} continuous_relerr={:.6e}",
                i + 1,
                y.re,
                y.im,
                (y - expected).abs() / expected.abs(),
                (y - continuous).abs() / continuous.abs()
            );
        }
        println!(
            "time_ac_expected discrete={:.12e},{:.12e} continuous={:.12e},{:.12e}",
            expected.re, expected.im, continuous.re, continuous.im
        );
    } else if mode == "uniform" {
        for doping in [-1., 0., 1.] {
            let uniform = Model::new(m.positions().to_vec(), vec![doping * 1e22; nodes])?;
            let conductance = uniform.uniform_admittance(doping * 1e22, C::new(0., 0.)).re;
            let capacitance = uniform.uniform_admittance(doping * 1e22, C::new(0., 1.)).im;
            let mut state = uniform.initial(300.)?;
            for target in [0., 0.1, 0.5, -0.5] {
                state = uniform.solve(state, target, 300., None)?.0;
                let current = uniform.currents(&state, None)?[nodes / 2];
                let prepared = uniform.prepare_admittance(&state)?;
                let mut worst = 0f64;
                for i in 0..=140 {
                    let f = 10f64.powf(i as f64 / 10.);
                    let s = C::new(0., std::f64::consts::TAU * f);
                    let exact = C::new(conductance, 0.) + s * capacitance;
                    let (actual, _) = prepared.evaluate(s)?;
                    worst = worst.max((actual - exact).abs() / exact.abs());
                }
                println!(
                    "uniform doping={doping} bias={target} I={current:.12e} exact_I={:.12e} G={conductance:.12e} C={capacitance:.12e} max_ac_error={worst:.6e}",
                    target * conductance
                );
            }
        }
    } else if mode == "ac" {
        let target = args
            .get(3)
            .and_then(|v| v.parse::<f64>().ok())
            .unwrap_or(0.);
        let mut state = eq;
        let steps = (target.abs() / 0.025).ceil() as usize;
        for i in 1..=steps {
            state = m
                .solve(state, target * i as f64 / steps as f64, 300., None)?
                .0;
        }
        let dynamic = m.prepare_admittance(&state)?;
        let (dc, _) = dynamic.evaluate(C::new(0., 0.))?;
        eprintln!(
            "dc G={:.12e}, AC zero G={:.12e}",
            m.terminal_stamp(&state, None)?.1,
            dc.re
        );
        for i in 0..=140 {
            let f = 10f64.powf(i as f64 / 10.);
            let (y, error) = dynamic.evaluate(C::new(0., std::f64::consts::TAU * f))?;
            println!(
                "ac,{target:.8e},{f:.12e},{:.12e},{:.12e},{:.12e}",
                y.re, y.im, error
            );
        }
        for dt in [1e-8, 1e-10, 1e-12] {
            let (y, _) = dynamic.evaluate(C::new(1. / dt, 0.))?;
            let (_, g) = m.terminal_stamp(&state, Some((&state, dt)))?;
            eprintln!(
                "BE dt={dt:.3e} dynamic={:.12e} BE={g:.12e} relerr={:.3e}",
                y.re,
                (y.re - g).abs() / g.abs()
            );
        }
    } else if mode == "stamp" {
        let mut state = eq.clone();
        for k in 1..=26 {
            state = m.solve(state, f64::from(k) * 0.025, 300., None)?.0;
            let (current, g) = m.terminal_stamp(&state, None)?;
            let step = 1e-6;
            let plus = m.solve(state.clone(), state.bias() + step, 300., None)?.0;
            let minus = m.solve(state.clone(), state.bias() - step, 300., None)?.0;
            let gd = (m.currents(&plus, None)?[nodes / 2] - m.currents(&minus, None)?[nodes / 2])
                / (2. * step);
            println!(
                "stamp bias={} I={:.12e} G={:.12e} finite_difference={:.12e} relerr={:.6e}",
                state.bias(),
                current,
                g,
                gd,
                (g - gd).abs() / gd.abs()
            );
        }
        let mut state = eq.clone();
        for k in 1..=10 {
            let source = f64::from(k) * 0.1;
            let (next, current, g, n) = m.series(&state, source, 1000., None)?;
            println!(
                "series source={} Vd={:.12e} I={:.12e} G={:.12e} circuit_iters={} kcl_A={:.12e}",
                source,
                next.bias(),
                current,
                g,
                n,
                current + (next.bias() - source) / 1000.
            );
            state = next;
        }
        // A transient MNA stamp contains both particle and displacement
        // current; compare against independently perturbed trial solutions.
        let accepted = state.clone();
        let dt = 1e-11;
        let trial = m
            .solve(
                accepted.clone(),
                accepted.bias() - 0.01,
                300.,
                Some((&accepted, dt)),
            )?
            .0;
        let (i, g) = m.terminal_stamp(&trial, Some((&accepted, dt)))?;
        let step = 1e-6;
        let plus = m
            .solve(
                trial.clone(),
                trial.bias() + step,
                300.,
                Some((&accepted, dt)),
            )?
            .0;
        let minus = m
            .solve(
                trial.clone(),
                trial.bias() - step,
                300.,
                Some((&accepted, dt)),
            )?
            .0;
        let gd = (m.currents(&plus, Some((&accepted, dt)))?[0]
            - m.currents(&minus, Some((&accepted, dt)))?[0])
            / (2. * step);
        println!(
            "transient_stamp Vd={} I={:.12e} G={:.12e} finite_difference={:.12e} relerr={:.6e}",
            trial.bias(),
            i,
            g,
            gd,
            (g - gd).abs() / gd.abs()
        );
    } else if mode == "thermal" {
        let mut state = eq;
        for k in 1..=24 {
            let (next, _, _) = m.solve(state, f64::from(k) * 0.025, ambient, None)?;
            state = next;
        }
        let rth = 500.;
        let cth = 2e-9;
        let dt = 1e-8;
        report(&m, &state, None, 0, 0.)?;
        let mut max_thermal_residual = 0f64;
        let mut max_energy = 0f64;
        for k in 1..=300 {
            let bias = if k <= 150 {
                0.6
            } else if k <= 174 {
                0.6 - 0.025 * f64::from(k - 150)
            } else {
                0.0
            };
            let (next, n, r, th) = m.heat_step(&state, bias, dt, ambient, rth, cth)?;
            max_thermal_residual = max_thermal_residual.max(th.abs());
            max_energy = max_energy.max(m.dissipation(&next)?.abs());
            report(&m, &next, Some((&state, dt)), n, r)?;
            state = next;
        }
        eprintln!(
            "thermal max balance residual W={max_thermal_residual:.4e}, peak dissipated W={max_energy:.4e}"
        );
    } else if mode == "tran" {
        let mut state = eq;
        for k in 1..=22 {
            let (next, _, _) = m.solve(state, f64::from(k) * 0.025, 300., None)?;
            state = next;
        }
        report(&m, &state, None, 0, 0.)?;
        // Constant ambient; voltage switches to reverse. Implicit BE carrier
        // state; displacement current included. No accepted-state mutation on failure.
        let dt = args
            .get(3)
            .and_then(|s| s.parse::<f64>().ok())
            .unwrap_or(10.)
            * 1e-12;
        if !dt.is_finite() || dt <= 0. {
            return Err("invalid time step".into());
        }
        let steps = (1e-9 / dt).round() as usize;
        for k in 1..=steps {
            let t = k as f64 * dt;
            let bias = if t < 2e-10 {
                0.55 - 0.85 * t / 2e-10
            } else {
                -0.3
            };
            let (next, n, r) = m.solve(state.clone(), bias, 300., Some((&state, dt)))?;
            report(&m, &next, Some((&state, dt)), n, r)?;
            state = next;
        }
    } else {
        return Err(format!("unknown mode {mode}"));
    }
    eprintln!(
        "nodes={nodes} elapsed_ms={:.3}",
        start.elapsed().as_secs_f64() * 1e3
    );
    Ok(())
}
