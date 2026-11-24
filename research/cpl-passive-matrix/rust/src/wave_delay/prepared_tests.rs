use super::*;

fn ideal_exchange(modes: usize) -> Vec<f64> {
    let width = 2 * modes;
    let mut scatter = vec![0.; width * width];
    for mode in 0..modes {
        scatter[mode * width + modes + mode] = 1.;
        scatter[(modes + mode) * width + mode] = 1.;
    }
    scatter
}
fn line(stages: usize, prune: bool) -> Result<WaveNetwork, &'static str> {
    let half = 1e-9 / (2. * f64::from(u32::try_from(stages).map_err(|_| "too many stages")?));
    WaveNetwork::new(
        vec![half],
        stages,
        ideal_exchange(1),
        vec![0.],
        vec![f64::INFINITY],
        prune,
    )
}

#[test]
fn one_global_step_causally_propagates_across_many_internal_delays() -> Result<(), &'static str> {
    for sections in [1, 4, 16, 32] {
        let mut model = line(sections, true)?;
        let prepare = model.prepare_step(2e-9)?;
        assert!(prepare.internal_steps() > sections);
        let b = prepare.evaluate(&[1.], &[0.])?;
        assert!(b.internal_steps() > sections);
        assert_eq!(b.reflected_near(), &[0.]);
        // A line with exactly 1ns of propagation transports the
        // PWL terminal ramp from 0 at t=0 to 1 at t=2ns:
        // at t=2ns the output is that input at t=1ns = 0.5.
        assert!(
            (b.transmitted_far()[0] - 0.5).abs() < 1e-8,
            "sections={sections}, wave={}",
            b.transmitted_far()[0]
        );
        let analytic = prepare.transmitted_jacobian();
        assert_eq!(analytic.len(), 2);
        assert!((analytic[0] - 0.5).abs() < 1e-8);
        assert_eq!(analytic[1], 0.);
        model.accept_prepared(b)?;
        assert!(model.energies().1 <= model.energies().0);
        let next = model.prepare_step(3e-9)?;
        let trial = next.evaluate(&[1.], &[0.])?;
        assert!(
            (trial.transmitted_far()[0] - 1.).abs() < 1e-8,
            "sections={sections}, second={:?}",
            trial.transmitted_far()
        );
    }
    Ok(())
}

#[test]
fn reused_preparation_has_exact_affine_newton_jacobian() -> Result<(), &'static str> {
    let model = line(8, true)?;
    let prepared = model.prepare_step(2e-9)?;
    let baseline = prepared.evaluate(&[0.], &[0.])?;
    let positive = prepared.evaluate(&[1.], &[0.])?;
    let negative = prepared.evaluate(&[-3.], &[0.])?;
    assert_eq!(baseline.transmitted_far(), &[0.]);
    assert_eq!(positive.reflected_near(), negative.reflected_near());
    let jac = prepared.transmitted_jacobian();
    assert!((positive.transmitted_far()[0] - baseline.transmitted_far()[0] - jac[0]).abs() < 1e-12);
    assert!(
        3.0_f64
            .mul_add(
                jac[0],
                negative.transmitted_far()[0] - baseline.transmitted_far()[0]
            )
            .abs()
            < 1e-12
    );
    assert_eq!(prepared.reflected_jacobian(), vec![0., 0.]);
    Ok(())
}

#[test]
fn prepared_one_big_step_agrees_with_several_individual_commits() -> Result<(), &'static str> {
    let global = 2e-9;
    let stages = 8;
    let mut macro_line = line(stages, true)?;
    let mut reference = line(stages, false)?;
    let prepared = macro_line.prepare_step(global)?;
    let batch = prepared.evaluate(&[1.], &[0.])?;
    let large_result = batch.transmitted_far()[0];
    macro_line.accept_prepared(batch)?;
    let count = 512_u32;
    for step in 1..=count {
        let time = global * (f64::from(step) / f64::from(count));
        let wave = [time / global];
        let trial = reference.trial(time, &wave, &[0.])?;
        reference.accept(trial)?;
    }
    let future_macro = macro_line.prepare_step(2.5e-9)?;
    let future_ref = reference.prepare_step(2.5e-9)?;
    let at_macro = future_macro.evaluate(&[1.], &[0.])?;
    let at_ref = future_ref.evaluate(&[1.], &[0.])?;
    assert!((large_result - 0.5).abs() < 1e-8);
    assert!((at_macro.transmitted_far()[0] - at_ref.transmitted_far()[0]).abs() < 3e-4);
    assert!(macro_line.retained_samples() < reference.retained_samples());
    Ok(())
}

#[test]
fn a_foreign_or_stale_prepared_batch_cannot_commit() -> Result<(), &'static str> {
    let mut line_a = line(4, true)?;
    let mut line_b = line(4, true)?;
    let proposal = line_a.prepare_step(1e-9)?;
    let foreign = proposal.evaluate(&[1.], &[0.])?;
    let initial = line_b.retained_samples();
    assert!(line_b.accept_prepared(foreign).is_err());
    assert_eq!(line_b.retained_samples(), initial);
    let stale = proposal.evaluate(&[1.], &[0.])?;
    let accepted = proposal.evaluate(&[0.25], &[0.])?;
    line_a.accept_prepared(accepted)?;
    let final_samples = line_a.retained_samples();
    assert!(line_a.accept_prepared(stale).is_err());
    assert_eq!(line_a.retained_samples(), final_samples);
    let next = line_a.prepare_step(2e-9)?;
    line_a.accept_prepared(next.evaluate(&[0.5], &[0.])?)?;
    Ok(())
}

#[test]
fn malformed_prepared_substep_rejects_all_wave_histories() -> Result<(), &'static str> {
    let mut network = line(4, true)?;
    let response = network.prepare_step(2e-9)?;
    let mut batch = response.evaluate(&[1.], &[0.])?;
    assert!(batch.samples.len() > 4);
    batch
        .samples
        .last_mut()
        .ok_or("missing substep")?
        .static_near
        .last_mut()
        .ok_or("missing last near scattering")?
        .first_mut()
        .ok_or("missing last modal coefficient")?
        .clone_from(&f64::NAN);
    let initial = (network.retained_samples(), network.energies());
    assert!(network.accept_prepared(batch).is_err());
    assert_eq!((network.retained_samples(), network.energies()), initial);
    let good = response.evaluate(&[0.5], &[0.])?;
    network.accept_prepared(good)?;
    Ok(())
}

#[test]
fn declared_future_source_front_is_not_leaked_by_macro_interpolation() -> Result<(), &'static str> {
    let mut model = WaveNetwork::new(
        vec![1e-10],
        8,
        ideal_exchange(1),
        vec![6e-10],
        vec![f64::INFINITY],
        true,
    )?;
    let trial = model.prepare_step(1.5e-9)?.evaluate(&[1.], &[0.])?;
    // Earliest transmitted signal has eight times two half-delays,
    // so near source at 0.6ns cannot reach far until 2.2ns.
    assert_eq!(trial.transmitted_far(), &[0.]);
    model.accept_prepared(trial)?;
    let later = model.prepare_step(3e-9)?.evaluate(&[1.], &[0.])?;
    assert!(later.transmitted_far()[0] > 0.);
    Ok(())
}

fn dense_passive_multimode(modes: usize, stages: usize) -> Result<WaveNetwork, &'static str> {
    let width = 2 * modes;
    let mut scattering = vec![0.; width * width];
    for row in 0..width {
        for col in 0..width {
            let same = row % modes == col % modes;
            let opposite = row / modes != col / modes;
            scattering[row * width + col] = if same && opposite {
                0.52
            } else if same {
                0.04
            } else if opposite {
                0.025
            } else {
                0.01
            };
        }
    }
    let delays = (0..modes)
        .map(|mode| match mode {
            0 => 1e-10,
            1 => 1.414_213_562_373_095e-10,
            _ => 1.732_050_807_568_877e-10,
        })
        .collect();
    WaveNetwork::new(
        delays,
        stages,
        scattering,
        vec![0.; modes],
        vec![f64::INFINITY; modes],
        true,
    )
}

#[test]
fn dense_coupled_three_wire_macrostep_carries_cross_mode_newton_jacobian()
-> Result<(), &'static str> {
    let mut network = dense_passive_multimode(3, 8)?;
    let response = network.prepare_step(7e-10)?;
    assert!(response.internal_steps() >= 7);
    let incident = [1., 0., 0.];
    let empty = [0., 0., 0.];
    let first = response.evaluate(&incident, &empty)?;
    let jac = response.reflected_jacobian();
    assert_eq!(jac.len(), 18);
    let alternate = response.evaluate(&[0.5, 0., 0.], &empty)?;
    for (mode, (&a, &b)) in first
        .reflected_near()
        .iter()
        .zip(alternate.reflected_near())
        .enumerate()
    {
        assert!(2.0_f64.mul_add(-b, a).abs() < 1e-11, "mode={mode}");
    }
    network.accept_prepared(first)?;
    assert!(network.energies().1 <= network.energies().0 + 1e-12);
    Ok(())
}

#[test]
fn dense_three_wire_event_fanout_remains_bounded_in_one_long_macrostep() -> Result<(), &'static str>
{
    for end in [9e-10, 1.2e-9, 1.5e-9, 1.8e-9, 2.1e-9, 2.4e-9, 2.7e-9, 3e-9] {
        let network = dense_passive_multimode(3, 8)?;
        let started = std::time::Instant::now();
        let prepared = network.prepare_step(end)?;
        eprintln!(
            "dense 3-wire stages=8 duration={end:.2e} steps={} elapsed={:?}",
            prepared.internal_steps(),
            started.elapsed()
        );
        assert!(
            prepared.internal_steps() < 200_000,
            "event scheduling has unacceptable combinatorial growth"
        );
    }
    Ok(())
}
