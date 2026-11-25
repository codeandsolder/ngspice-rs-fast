//! Analytic right-hand coefficients used by consistent DAE initialization.
use std::f64::consts::TAU;
use std::sync::Arc;

use ngspice_rs::devices::linear::SourceRows;
use ngspice_rs::devices::{
    AmSpec, ExpSpec, FunctionSpec, LinearSource, LinearSystem, Pulse, PwlSource, SffmSpec,
    SineSpec, SourceKind, TransientTiming, Waveform,
};
use ngspice_rs::primitives::Complex;

fn agree(actual: &[f64], expected: &[f64]) {
    assert_eq!(actual.len(), expected.len());
    for (&a, &b) in actual.iter().zip(expected) {
        assert!((a - b).abs() <= 1e-13 * b.abs().max(1.), "{a} != {b}");
    }
}

#[test]
fn piecewise_linear_sources_select_the_actual_right_segment() {
    let plain = PwlSource::plain(vec![(0., 2.), (0.001, 3.), (0.002, 7.)]).unwrap();
    agree(
        plain.initial_jet(3, 0.001).unwrap().coefficients(),
        &[2., 1., 0., 0.],
    );
    let shifted = plain.clone().with_timing(-0.001, None).unwrap();
    agree(
        shifted.initial_jet(3, 0.001).unwrap().coefficients(),
        &[3., 4., 0., 0.],
    );
    let repeated = plain.clone().with_timing(-0.002, Some(0.001)).unwrap();
    agree(
        repeated.initial_jet(3, 0.001).unwrap().coefficients(),
        &[3., 4., 0., 0.],
    );
    let held = plain.with_timing(0.001, None).unwrap();
    agree(
        held.initial_jet(3, 0.001).unwrap().coefficients(),
        &[2., 0., 0., 0.],
    );

    let pulse = Pulse::new(1., 5., 0., 0.002, 0.001, 0.003, 0.01).unwrap();
    agree(
        pulse.initial_jet(3, 0.001).unwrap().coefficients(),
        &[1., 2., 0., 0.],
    );
    let falling = Pulse::new(1., 5., 0., 0., 0.002, 0., 0.01).unwrap();
    agree(
        falling.initial_jet(3, 0.001).unwrap().coefficients(),
        &[5., -2., 0., 0.],
    );

    let step = Waveform::Step {
        before: 1.,
        after: -3.,
        time: 0.,
    };
    agree(
        step.initial_jet(3, 1.).unwrap().coefficients(),
        &[-3., 0., 0., 0.],
    );
}

#[test]
fn scaled_pwl_slope_does_not_overflow_the_endpoint_difference() {
    let pwl = PwlSource::plain(vec![(0., -1e308), (4., 1e308)]).unwrap();
    let jet = pwl.initial_jet(2, 1.).unwrap();
    assert_eq!(jet.coefficients(), &[-1e308, 5e307, 0.]);
}

#[test]
fn analytic_waveform_coefficients_match_closed_form() {
    let timing = TransientTiming::new(0.001, 1.).unwrap();
    let sine = FunctionSpec::Sine(SineSpec {
        offset: 1.,
        amplitude: 2.,
        frequency: Some(2. / TAU),
        delay: Some(0.),
        damping: Some(3.),
        phase: Some(0.),
    })
    .resolve(&timing)
    .unwrap();
    // 1+2*sin(2u)*exp(-3u).
    agree(
        sine.initial_jet(4, 1.).unwrap().coefficients(),
        &[1., 4., -12., 46. / 3., -10.],
    );

    let sffm = FunctionSpec::Sffm(SffmSpec {
        offset: 1.,
        amplitude: 2.,
        carrier: Some(4. / TAU),
        signal: Some(2. / TAU),
        index: Some(0.5),
        delay: Some(0.),
        carrier_phase: Some(0.),
        signal_phase: Some(0.),
    })
    .resolve(&timing)
    .unwrap();
    // 1+2*sin(4u+0.5*sin(2u)).
    agree(
        sffm.initial_jet(4, 1.).unwrap().coefficients(),
        &[1., 10., 0., -43., 0.],
    );

    let am = FunctionSpec::Am(AmSpec {
        offset: 1.,
        carrier_amplitude: 2.,
        modulation_amplitude: Some(3.),
        carrier_frequency: Some(4. / TAU),
        modulation_frequency: Some(2. / TAU),
        delay: Some(0.),
        carrier_phase: Some(0.),
        modulation_phase: Some(0.),
    })
    .resolve(&timing)
    .unwrap();
    // 1+(2+3*sin(2u))*sin(4u).
    agree(
        am.initial_jet(4, 1.).unwrap().coefficients(),
        &[1., 8., 24., -64. / 3., -80.],
    );

    let exp = FunctionSpec::Exp(ExpSpec {
        initial: 1.,
        pulsed: 5.,
        rise_delay: Some(0.),
        rise_tau: Some(1.),
        fall_delay: Some(2.),
        fall_tau: Some(1.),
    })
    .resolve(&timing)
    .unwrap();
    // Source EXP's zero rise delay defaults to the print step, so the
    // right-hand germ at t=0 is the held initial value.
    agree(
        exp.initial_jet(4, 1.).unwrap().coefficients(),
        &[1., 0., 0., 0., 0.],
    );
}

#[test]
fn signed_rhs_preserves_source_rows_and_coefficient_order() {
    let mut system = LinearSystem::new(3);
    for (name, rows, waveform) in [
        (
            "v1",
            vec![(0, 1.)],
            Waveform::Pwl(PwlSource::plain(vec![(0., 2.), (1., 3.)]).unwrap()),
        ),
        (
            "i1",
            vec![(0, -1.), (2, 1.)],
            Waveform::PortSeries {
                baseline: Box::new(Waveform::Constant(1.)),
                tones: vec![(2., 3.)],
            },
        ),
    ] {
        let mut bound = SourceRows::new();
        for (row, sign) in rows {
            bound.push(row, sign).unwrap();
        }
        system.sources.push(LinearSource {
            name: Arc::new(name.to_owned()),
            kind: SourceKind::Voltage,
            rows: bound,
            dc: 0.,
            ac: Complex::ZERO,
            waveform: Arc::new(waveform),
        });
    }
    let jet = system.initial_rhs_jet(3, 1.).unwrap();
    agree(jet[0].as_slice(), &[-1., 0., 3.]);
    agree(jet[1].as_slice(), &[1., 0., 0.]);
    agree(jet[2].as_slice(), &[9., 0., -9.]);
    agree(jet[3].as_slice(), &[0., 0., 0.]);
}
