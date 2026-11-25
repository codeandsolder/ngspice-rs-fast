//! Independent Fourier and window normalization checks.
use std::f64::consts::TAU;

use ngspice_rs::Complex;
use ngspice_rs::analysis::spectrum::{
    SpectrumLayout, SpectrumLength, SpectrumNormalization, SpectrumPlan, SpectrumSettings,
    SpectrumWindow,
};
use ngspice_rs::maths::fft::{Direction, FftPlan};

fn near(actual: Complex, expected: Complex, tolerance: f64) {
    assert!(
        (actual - expected).magnitude() <= tolerance,
        "actual={actual:?} expected={expected:?}"
    );
}

fn direct(input: &[Complex], k: usize) -> Complex {
    input
        .iter()
        .enumerate()
        .fold(Complex::ZERO, |sum, (j, value)| {
            let phase = -TAU * f64::from(u32::try_from(j * k).unwrap())
                / f64::from(u32::try_from(input.len()).unwrap());
            sum + *value * Complex::new(phase.cos(), phase.sin())
        })
}

#[test]
fn exact_complex_fft_matches_independent_dft_and_round_trips() {
    for n in [1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 29, 32, 63, 64, 127] {
        let input: Vec<_> = (0..n)
            .map(|i| {
                let i = f64::from(i);
                Complex::new((0.71 * i).sin() + 0.3, (1.17 * i).cos() - 0.2)
            })
            .collect();
        let plan = FftPlan::new(input.len(), 100_000).unwrap();
        let mut workspace = plan.scratch().unwrap();
        let mut actual = input.clone();
        plan.transform(&mut actual, &mut workspace, Direction::Forward)
            .unwrap();
        for (k, value) in actual.iter().enumerate() {
            near(*value, direct(&input, k), 2e-11);
        }
        plan.transform(&mut actual, &mut workspace, Direction::Inverse)
            .unwrap();
        for (value, original) in actual.iter().zip(input) {
            near(*value, original, 2e-13);
        }
    }
}

fn spectrum(
    input: &[Complex],
    layout: SpectrumLayout,
    settings: SpectrumSettings,
) -> (SpectrumPlan, Vec<Complex>) {
    let plan = SpectrumPlan::new(input.len(), 0.125, layout, settings, 1_000_000).unwrap();
    let mut workspace = plan.workspace().unwrap();
    let mut output = vec![Complex::ZERO; plan.bins()];
    plan.transform(input, &mut workspace, &mut output).unwrap();
    (plan, output)
}

#[test]
fn odd_last_bin_is_not_nyquist() {
    for n in [3, 5, 17, 30, 31, 32, 33] {
        let k = n / 2;
        let input: Vec<_> = (0..n)
            .map(|j| Complex::real(2. * (TAU * f64::from(k * j) / f64::from(n)).cos()))
            .collect();
        let (plan, values) = spectrum(
            &input,
            SpectrumLayout::Real,
            SpectrumSettings {
                window: SpectrumWindow::Rectangular,
                ..SpectrumSettings::default()
            },
        );
        assert_eq!(values.len(), usize::try_from(k + 1).unwrap());
        near(*values.last().unwrap(), Complex::real(2.), 1e-12);
        assert!(
            (plan.frequency(values.len() - 1).unwrap() - 8. * f64::from(k) / f64::from(n)).abs()
                < 1e-14
        );
    }
}

#[test]
fn complex_signed_frequency_and_phase() {
    let n = 17;
    let input: Vec<_> = (0..n)
        .map(|j| {
            let phase = -TAU * 3. * f64::from(j) / f64::from(n);
            Complex::new(phase.cos(), phase.sin()) * Complex::new(2., -1.)
        })
        .collect();
    let (plan, values) = spectrum(
        &input,
        SpectrumLayout::Complex,
        SpectrumSettings {
            window: SpectrumWindow::Rectangular,
            ..SpectrumSettings::default()
        },
    );
    assert_eq!(values.len(), 17);
    for (k, value) in values.iter().enumerate() {
        near(
            *value,
            if k == 14 {
                Complex::new(2., -1.)
            } else {
                Complex::ZERO
            },
            1e-12,
        );
    }
    assert!((plan.frequency(14).unwrap() + 24. / 17.).abs() < 1e-14);
}

#[test]
fn finite_window_gain_is_exact_even_for_short_odd_lengths() {
    for window in [
        SpectrumWindow::Rectangular,
        SpectrumWindow::Hann,
        SpectrumWindow::Hamming,
        SpectrumWindow::Triangle,
        SpectrumWindow::Blackman,
        SpectrumWindow::BlackmanHarris,
        SpectrumWindow::FlatTop,
    ] {
        for n in [2, 3, 5, 16, 31, 64] {
            for length in [SpectrumLength::Exact, SpectrumLength::Padded] {
                let (plan, output) = spectrum(
                    &vec![Complex::real(2.5); n],
                    SpectrumLayout::Real,
                    SpectrumSettings {
                        window,
                        length,
                        ..SpectrumSettings::default()
                    },
                );
                near(output[0], Complex::real(2.5), 1e-12);
                let metrics = plan.metrics();
                assert!(metrics.coherent_gain > 0.);
                assert!(metrics.enbw_bins >= 1. - 1e-14);
                assert!(
                    (metrics.enbw_hz
                        - metrics.enbw_bins * 8. / f64::from(u32::try_from(n).unwrap()))
                    .abs()
                        < 1e-13
                );
            }
        }
    }
}

#[test]
fn power_and_density_obey_parseval_with_padding_and_windows() {
    for n in [5, 16, 31, 64] {
        for layout in [SpectrumLayout::Real, SpectrumLayout::Complex] {
            let input: Vec<_> = (0..n)
                .map(|j| {
                    let phase = TAU * f64::from(j) / f64::from(n);
                    Complex::new(
                        0.2 + phase.sin(),
                        if layout == SpectrumLayout::Complex {
                            phase.cos()
                        } else {
                            0.
                        },
                    )
                })
                .collect();
            for window in [
                SpectrumWindow::Rectangular,
                SpectrumWindow::Hann,
                SpectrumWindow::Triangle,
            ] {
                // Independent weights, using the periodic convention anchored at last sample.
                let weights: Vec<_> = (0..n)
                    .map(|j| {
                        let x = f64::from(n - 1 - j) / f64::from(n);
                        match window {
                            SpectrumWindow::Rectangular => 1.,
                            SpectrumWindow::Hann => (1. - (TAU * x).cos()) / 2.,
                            SpectrumWindow::Triangle => 2. * x.min(1. - x),
                            _ => unreachable!(),
                        }
                    })
                    .collect();
                let energy = weights.iter().map(|w| w * w).sum::<f64>();
                let expected = input
                    .iter()
                    .zip(&weights)
                    .map(|(v, w)| (v.magnitude() * w).powi(2))
                    .sum::<f64>()
                    / energy;
                for length in [SpectrumLength::Exact, SpectrumLength::Padded] {
                    for normalization in
                        [SpectrumNormalization::Power, SpectrumNormalization::Density]
                    {
                        let (plan, values) = spectrum(
                            &input,
                            layout,
                            SpectrumSettings {
                                window,
                                length,
                                normalization,
                            },
                        );
                        let total = values.iter().map(|v| v.re).sum::<f64>();
                        let integrated = if normalization == SpectrumNormalization::Density {
                            total * plan.frequency(1).unwrap()
                        } else {
                            total
                        };
                        assert!(
                            (integrated - expected).abs() < 1e-12,
                            "{window:?} {layout:?} {n} {length:?}"
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn spectrum_rejects_bad_dimensions_and_budget() {
    assert!(FftPlan::new(0, 100).is_err());
    assert!(FftPlan::new(127, 5).is_err());
    assert!(
        SpectrumPlan::new(
            5,
            0.,
            SpectrumLayout::Real,
            SpectrumSettings::default(),
            100_000
        )
        .is_err()
    );
    assert!(
        SpectrumPlan::new(5, 1., SpectrumLayout::Real, SpectrumSettings::default(), 1).is_err()
    );
    let plan = SpectrumPlan::new(
        5,
        1.,
        SpectrumLayout::Real,
        SpectrumSettings::default(),
        100_000,
    )
    .unwrap();
    let mut workspace = plan.workspace().unwrap();
    let mut out = vec![Complex::ZERO; 3];
    assert!(
        plan.transform(&[Complex::new(0., 1.); 5], &mut workspace, &mut out)
            .is_err()
    );
    assert!(
        plan.transform(&[Complex::real(f64::NAN); 5], &mut workspace, &mut out)
            .is_err()
    );
    assert!(
        plan.transform(&[Complex::ZERO; 4], &mut workspace, &mut out)
            .is_err()
    );
    assert!(plan.frequency(3).is_err());
}

#[test]
#[ignore = "requires NGSPICE_BIN for the independent native spectrum oracle"]
fn padded_or_exact_spectrum_agrees_with_native_ngspice() {
    use ngspice_conformance::ngspice::Scratch;
    use ngspice_conformance::process::{ProcessLimits, run};
    use ngspice_rs::analysis::RawFile;

    let binary = std::env::var_os("NGSPICE_BIN").expect("set absolute NGSPICE_BIN");
    let scratch = Scratch::new("c-fft-reference").unwrap();
    let directory = scratch.path();
    std::fs::write(
        directory.join("probe.cir"),
        concat!(
            "Independent spectral oracle\n.control\nset noaskquit\n",
            "setplot new\nlet time = vector(31) / 8\nsettype time time\nsetscale time\n",
            "let signal = 1.25 + 2 * sin(2 * pi * 5 * vector(31) / 31)\n",
            "set specwindow=none\nfft signal\nset filetype=ascii\n",
            "write spectrum.raw frequency signal\nquit\n.endc\n.end\n"
        ),
    )
    .unwrap();
    let result = run(
        binary,
        directory,
        &["-b", "probe.cir"],
        ProcessLimits::ORACLE,
    )
    .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let raw = RawFile::load(directory.join("spectrum.raw")).unwrap();
    let reference = &raw.plots[0].plot;
    let length = match reference.point_count() {
        16 => SpectrumLength::Exact,  // FFTW
        17 => SpectrumLength::Padded, // internal radix-two backend
        n => panic!("unexpected reference bin count: {n}"),
    };
    let input: Vec<_> = (0..31)
        .map(|j| Complex::real(1.25 + 2. * (TAU * 5. * f64::from(j) / 31.).sin()))
        .collect();
    let (plan, output) = spectrum(
        &input,
        SpectrumLayout::Real,
        SpectrumSettings {
            window: SpectrumWindow::Rectangular,
            length,
            ..SpectrumSettings::default()
        },
    );
    for (index, value) in output.iter().enumerate() {
        near(*value, reference.value("signal", index).unwrap(), 2e-12);
        near(
            Complex::real(plan.frequency(index).unwrap()),
            reference.value("frequency", index).unwrap(),
            2e-12,
        );
    }
}
