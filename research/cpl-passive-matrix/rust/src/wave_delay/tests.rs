use super::*;

mod physical_fixtures;

fn swapped(modes: usize) -> Vec<f64> {
    let width = 2 * modes;
    let mut scatter = vec![0.; width * width];
    for mode in 0..modes {
        scatter[mode * width + modes + mode] = 1.;
        scatter[(modes + mode) * width + mode] = 1.;
    }
    scatter
}

fn unit_lossless(stages: usize, prune: bool) -> Result<WaveNetwork, &'static str> {
    let half = 1e-9 / f64::from(u32::try_from(stages).map_err(|_| "stage overflow")?) / 2.;
    WaveNetwork::new(
        vec![half],
        stages,
        swapped(1),
        vec![0.],
        vec![f64::INFINITY],
        prune,
    )
}

#[test]
fn scalar_exact_wavefront_even_with_irregular_delayed_interpolation() -> Result<(), &'static str> {
    for stages in [1, 4, 16, 32] {
        let mut network = unit_lossless(stages, true)?;
        let half =
            1e-9 / (2. * f64::from(u32::try_from(stages).map_err(|_| "stage count out of range")?));
        let mut t = 0.;
        let mut before_front = 0.0_f64;
        let mut last_transmitted = 0.;
        for _ in 0..1024 {
            if t >= 2e-9 {
                break;
            }
            let next = half.mul_add(0.7, t).min(2e-9);
            let drive = (next / 1e-10).min(1.);
            let trial = network.trial(next, &[drive], &[0.])?;
            if next < 1e-9 {
                before_front = before_front.max(trial.transmitted_far()[0].abs());
            }
            assert_eq!(trial.reflected_near()[0], 0.);
            last_transmitted = trial.transmitted_far()[0];
            network.accept(trial)?;
            t = next;
        }
        assert_eq!(
            before_front, 0.,
            "{stages} cells leaked before physical LC front"
        );
        assert!((last_transmitted - 1.).abs() < 1e-11);
        let (input_energy, output_energy) = network.energies();
        assert!(output_energy <= input_energy);
    }
    Ok(())
}

fn coupled_scattering() -> Vec<f64> {
    // Reciprocal, strictly contractive 4x4 two-mode scattering. Every
    // absolute row sum is at most 0.945; both modal paths are coupled.
    vec![
        0.1, 0.015, 0.8, 0.03, 0.015, 0.1, 0.03, 0.8, 0.8, 0.03, 0.1, 0.015, 0.03, 0.8, 0.015, 0.1,
    ]
}

#[test]
fn coupled_two_mode_front_cross_talk_and_passivity() -> Result<(), &'static str> {
    let stages = 8;
    let half =
        1e-9 / (2. * f64::from(u32::try_from(stages).map_err(|_| "stage count out of range")?));
    let mut network = WaveNetwork::new(
        vec![half, 1.3 * half],
        stages,
        coupled_scattering(),
        vec![0., 0.],
        vec![f64::INFINITY, f64::INFINITY],
        true,
    )?;
    let mut t = 0.;
    let mut crosstalk = 0.0_f64;
    let mut largest_prefront = 0.0_f64;
    for _ in 0..1024 {
        if t >= 3e-9 {
            break;
        }
        let next = half.mul_add(0.25, t).min(3e-9);
        let trial = network.trial(next, &[1., 0.], &[0., 0.])?;
        if next < 1e-9 {
            for &value in trial.transmitted_far() {
                largest_prefront = largest_prefront.max(value.abs());
            }
        }
        crosstalk = crosstalk.max(trial.transmitted_far()[1].abs());
        network.accept(trial)?;
        t = next;
    }
    assert_eq!(largest_prefront, 0.);
    assert!(crosstalk > 1e-4, "real cross-mode transmission missing");
    let (input_energy, output_energy) = network.energies();
    assert!(output_energy <= input_energy);
    Ok(())
}

#[test]
fn rejects_active_and_nonreciprocal_static_scattering() {
    assert!(
        WaveNetwork::new(
            vec![1e-9],
            1,
            vec![0., 1.1, 1.1, 0.],
            vec![0.],
            vec![f64::INFINITY],
            true
        )
        .is_err()
    );
    assert!(
        WaveNetwork::new(
            vec![1e-9],
            1,
            vec![0., 1., 0.4, 0.],
            vec![0.],
            vec![f64::INFINITY],
            true
        )
        .is_err()
    );
    assert!(
        WaveNetwork::new(vec![0.], 1, swapped(1), vec![0.], vec![f64::INFINITY], true).is_err()
    );
}

#[test]
fn trial_rejection_preserves_histories_and_foreign_ownership() -> Result<(), &'static str> {
    let mut first = unit_lossless(4, true)?;
    let mut foreign = unit_lossless(4, true)?;
    let half = 1e-9 / 8.;
    let step = half * 0.5;
    let foreign_candidate = first.trial(step, &[1.], &[0.])?;
    assert!(foreign.accept(foreign_candidate).is_err());
    assert_eq!(foreign.retained_samples(), 10);
    let accepted = first.trial(step, &[1.], &[0.])?;
    let stale = first.trial(step, &[2.], &[0.])?;
    first.accept(accepted)?;
    assert!(first.accept(stale).is_err());
    assert!(first.trial(step, &[1.], &[0.]).is_err());
    assert!(first.trial(half.mul_add(2., step), &[1.], &[0.]).is_err());
    assert!(first.trial(step * 2., &[1.], &[1.]).is_err());
    assert!(first.trial(f64::NAN, &[1.], &[0.]).is_err());
    let next = first.trial(step * 2., &[1.], &[0.])?;
    first.accept(next)?;
    assert!(first.energies().0.is_finite());
    Ok(())
}

#[test]
fn pruning_is_identical_to_full_history_at_every_step() -> Result<(), &'static str> {
    let mut full = unit_lossless(8, false)?;
    let mut short = unit_lossless(8, true)?;
    let dt = 1e-9 / 8. / 8.;
    let mut t = 0.;
    for step in 1..=700_u32 {
        let next = dt * f64::from(step);
        let source = [(next / 1e-9 * 4.).sin()];
        let first = full.trial(next, &source, &[0.])?;
        let second = short.trial(next, &source, &[0.])?;
        assert_eq!(first.transmitted_far(), second.transmitted_far());
        assert_eq!(first.reflected_near(), second.reflected_near());
        full.accept(first)?;
        short.accept(second)?;
        t = next;
    }
    assert!(t > 1e-9);
    assert!(
        short.retained_samples() * 15 < full.retained_samples(),
        "bounded modal histories unexpectedly retained too many samples"
    );
    assert_eq!(full.energies(), short.energies());
    Ok(())
}

#[test]
fn two_modes_and_zero_shunt_loss_passivity_still_accept_exact_exchange() -> Result<(), &'static str>
{
    let mut network = WaveNetwork::new(
        vec![1e-10, 1.1e-10],
        1,
        swapped(2),
        vec![0., 0.],
        vec![f64::INFINITY, f64::INFINITY],
        true,
    )?;
    let trial = network.trial(1e-11, &[1., 0.], &[0., 0.])?;
    assert_eq!(trial.transmitted_far(), &[0., 0.]);
    network.accept(trial)?;
    Ok(())
}

#[test]
fn invalid_injected_future_wave_cannot_modify_accepted_state() -> Result<(), &'static str> {
    let mut network = WaveNetwork::new(
        vec![1e-10],
        2,
        swapped(1),
        vec![1e-9],
        vec![f64::INFINITY],
        true,
    )?;
    let samples_before = network.retained_samples();
    assert!(network.trial(1e-11, &[1.], &[0.]).is_err());
    assert!(network.trial(1e-11, &[0.], &[1.]).is_err());
    assert_eq!(network.retained_samples(), samples_before);
    let quiet = network.trial(1e-11, &[0.], &[0.])?;
    network.accept(quiet)?;
    Ok(())
}
