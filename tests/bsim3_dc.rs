//! Offline sanity checks for the BSIM3 DC core. Numeric parity with C lives in
//! the opt-in `c_bsim3_dc_reference` suite.
use std::path::Path;

use ngspice_rs::devices::bsim3::{self, charges, dc, noise};
use ngspice_rs::devices::{ModelContext, ModelResolver};
use ngspice_rs::netlist::Parser;
use ngspice_rs::netlist::source::parse_deck_text;

fn prepared(
    card: &str,
    instance: &str,
) -> (
    bsim3::defaults::ModelDefaults,
    bsim3::temperature::TemperatureModel,
    bsim3::setup::InstanceSetup,
) {
    let text = format!("dc\nm1 d g 0 0 nn {instance}\n.model nn nmos level=8 {card}\n.end\n");
    let netlist = Parser::new()
        .parse_deck(&parse_deck_text(Path::new("dc.cir"), &text))
        .unwrap();
    let resolver = ModelResolver::new(&netlist.models).unwrap();
    let resolved = resolver.resolve(&netlist.devices[0]).unwrap().unwrap();
    let params = bsim3::model_parameters(&resolved).unwrap();
    let context = ModelContext::new(27., 27.);
    let defaults = params.model_defaults(&context).unwrap();
    let temperature = defaults
        .at_temperature(27., &netlist.devices[0].location)
        .unwrap();
    let inputs = params.instance_parameters(&netlist.devices[0], 1.).unwrap();
    let setup = bsim3::setup::InstanceSetup::new(
        &defaults,
        &temperature,
        &inputs,
        &netlist.devices[0].location,
    )
    .unwrap();
    (defaults, temperature, setup)
}

fn evaluate(card: &str, instance: &str, bias: dc::Bias) -> dc::DcEvaluation {
    let (defaults, temperature, setup) = prepared(card, instance);
    dc::DcModel {
        model: &defaults,
        temperature: &temperature,
        setup: &setup,
    }
    .evaluate(bias, ModelContext::new(27., 27.).gmin)
    .unwrap()
}

fn at(vgs: f64, vds: f64) -> dc::Bias {
    dc::Bias { vbs: 0., vgs, vds }
}

#[test]
fn zero_bias_leaves_only_the_junction_offsets() {
    let d = evaluate("", "l=1u w=10u", at(0., 0.));
    assert!(d.ids.abs() < 1e-18, "{}", d.ids);
    // C's junction model has no gmin offset at zero bias.
    assert_eq!(d.cbs, 0.);
    assert_eq!(d.cbd, 0.);
    assert!(d.gbs >= 1e-12 && d.gbd >= 1e-12);
    assert!(d.von.is_finite());
    assert_eq!(d.mode, 1);
}

#[test]
fn drain_current_rises_with_gate_and_saturates() {
    // A bare BSIM3 model has Vth around 1.8 V (C reports the same), so these
    // biases sit above threshold.
    let off = evaluate("", "l=1u w=10u", at(1.5, 1.0));
    let on = evaluate("", "l=1u w=10u", at(2.5, 1.0));
    assert!(on.ids > off.ids * 1e3, "{} {}", on.ids, off.ids);
    let linear = evaluate("", "l=1u w=10u", at(2.5, 0.05));
    let saturation = evaluate("", "l=1u w=10u", at(2.5, 1.0));
    assert!(saturation.ids > linear.ids);
    assert!(saturation.ids.is_finite() && saturation.gm > 0. && saturation.gds > 0.);
    // Long-channel saturation is only weakly dependent on Vds.
    let deeper = evaluate("", "l=1u w=10u", at(2.5, 2.0));
    assert!(deeper.ids > saturation.ids);
    assert!((deeper.ids - saturation.ids).abs() / saturation.ids < 0.2);
}

#[test]
fn inverse_mode_mirrors_the_terminals() {
    // Inverse mode swaps source and drain: C evaluates Vgs=vgd and Vbs=vbd,
    // so the same device driven with those forward biases must agree.
    let mirror = evaluate(
        "",
        "l=1u w=10u",
        dc::Bias {
            vbs: 0.8,
            vgs: 3.0,
            vds: 0.8,
        },
    );
    let reverse = evaluate(
        "",
        "l=1u w=10u",
        dc::Bias {
            vbs: 0.,
            vgs: 2.2,
            vds: -0.8,
        },
    );
    assert_eq!(mirror.mode, 1);
    assert_eq!(reverse.mode, -1);
    assert!(
        reverse.ids > 0.,
        "inverse current stays positive: {}",
        reverse.ids
    );
    assert!(
        (reverse.ids - mirror.ids).abs() / mirror.ids < 1e-12,
        "{} vs {}",
        reverse.ids,
        mirror.ids
    );
    assert_eq!(reverse.vds, 0.8);
}

#[test]
fn mobility_models_and_series_resistance_change_the_current() {
    let base = evaluate("", "l=0.5u w=5u", at(2.5, 1.0)).ids;
    let model2 = evaluate(
        "mobmod=2 ua=1e-9 ub=1e-18 uc=1e-11",
        "l=0.5u w=5u",
        at(2.5, 1.0),
    )
    .ids;
    let model3 = evaluate(
        "mobmod=3 ua=4.9e-10 ub=1e-18 uc=-3e-11",
        "l=0.5u w=5u",
        at(2.5, 1.0),
    )
    .ids;
    assert!((base - model2).abs() / base > 1e-3, "{base:e} {model2:e}");
    assert!((base - model3).abs() / base > 1e-3, "{base:e} {model3:e}");
    // RDSW creates the channel series resistance C folds into the load.
    // RSH/NRD/NRS only move the junction nodes, which the core does not see.
    let resistive = evaluate("rdsw=100", "l=1u w=10u", at(2.5, 0.4));
    let plain = evaluate("", "l=1u w=10u", at(2.5, 0.4));
    assert!(resistive.rds > 0., "{}", resistive.rds);
    assert!(resistive.ids < plain.ids);
}

#[test]
fn junction_diodes_follow_area_perimeter_and_ijth() {
    let card = "js=1e-5 jsw=1e-10";
    let area = evaluate(card, "l=1u w=10u ad=2p as=1p pd=4u ps=3u", at(0., 0.3));
    // vbd = -0.3 V: the drain diode is reverse biased but not clamped.
    assert!(area.cbd < 0. && area.cbs == 0.);
    let no_ijth = evaluate(
        "ijth=0 js=1e-5 jsw=1e-10",
        "l=1u w=10u ad=2p as=1p pd=4u ps=3u",
        at(0., 0.3),
    );
    // Both select C's exponential branch below breakdown.
    assert!((area.cbd - no_ijth.cbd).abs() <= 1e-12 * area.cbd.abs());
    // Larger area increases the saturation current and thus the magnitude.
    let bigger = evaluate(card, "l=1u w=10u ad=20p pd=4u", at(0., 0.3));
    assert!(bigger.cbd < area.cbd);
}

#[test]
fn ask_keywords_read_the_evaluated_values() {
    let d = evaluate("", "l=1u w=10u", at(2.5, 1.0));
    assert_eq!(d.ask("id"), Some(d.ids));
    assert_eq!(d.ask("vth"), Some(d.von));
    assert_eq!(d.ask("gm"), Some(d.gm));
    assert_eq!(d.ask("ibs"), Some(d.cbs));
    assert_eq!(d.ask("ibd"), Some(d.cbd));
    // `cbs`/`cbd` are capacitances in C, so they are not DC asks.
    assert_eq!(d.ask("cbs"), None);
    assert_eq!(d.ask("cgg"), None);
}

#[test]
#[ignore = "opt-in independent BSIM3 DC/CV/noise kernel measurements"]
fn arithmetic_workload() {
    use std::hint::black_box;
    use std::time::Instant;

    // Default and physically parameterized models exercise both weak and
    // strong inversion, mobility and saturation branches, all charge
    // partitions and both noise-law families without timing netlist setup.
    for (i, card) in [
        "capmod=3 xpart=0.5 noimod=1",
        "capmod=3 xpart=0.4 noimod=2 tox=4.1n nch=2.3549e17 vth0=0.3694 mobmod=2 u0=268.7 rdsw=100 ngate=1e20",
        "capmod=3 xpart=0.6 noimod=6 tox=4.1n nch=2.3549e17 vth0=0.3694 mobmod=3 u0=268.7 rdsw=20 alpha0=1e-8 beta0=2",
    ].into_iter().enumerate() {
        let (defaults, temperature, setup) = prepared(card, "l=0.18u w=2u ad=0.1p as=0.2p pd=2u ps=3u");
        let dc_model = dc::DcModel {model:&defaults, temperature:&temperature, setup:&setup};
        let charge_model = charges::ChargeModel {model:&defaults, temperature:&temperature, setup:&setup};
        let noise_model = noise::NoiseModel {model:&defaults, setup:&setup};
        let biases: Vec<_> = (0..32).flat_map(|gate| (0..16).flat_map(move |drain| (0..4).map(move |body| dc::Bias {
            vgs: f64::from(gate) / 10.,
            vds: f64::from(drain) / 10.,
            vbs: -f64::from(body) / 4.,
        }))).collect();
        let gmin = ModelContext::new(27.,27.).gmin;
        // Capture independent signatures off the timed path. The caller
        // compares each field against the same grid at the scalar baseline.
        for (j, &bias) in biases.iter().enumerate() {
            let d = dc_model.evaluate(bias,gmin).unwrap();
            let q = charge_model.evaluate(&d,bias).unwrap();
            let n = noise_model.densities(&d,&q,300.15).unwrap();
            let fields = [d.ids,d.gm,d.gds,d.gmbs,d.isub,d.vgsteff,d.vdseff,d.vdsat,d.cbs,d.cbd,q.qgate,q.qbulk,q.qdrn,q.qsrc,q.cggb,q.cgdb,q.cgsb,q.cdgb,q.cddb,q.cdsb,q.cbgb,q.cbdb,q.cbsb,q.capbs,q.capbd,q.qbs,q.qbd,n.channel,n.flicker_numerator];
            println!("bsim3_sample,{i},{j},{}",fields.map(|v| format!("{v:.17e}")).join(","));
        }
        for _ in 0..3 {
            let started = Instant::now();
            for _ in 0..32 {
                for &bias in &biases {
                    let d = dc_model.evaluate(black_box(bias),gmin).unwrap();
                    let q = charge_model.evaluate(&d,bias).unwrap();
                    black_box(noise_model.densities(&d,&q,300.15).unwrap());
                }
            }
            println!("bsim3_bench,{i},{},{}",biases.len()*32,started.elapsed().as_nanos());
        }
    }
}

#[test]
fn tiny_bias_retains_first_order_current_and_charge() {
    let (defaults, temperature, setup) = prepared(
        "vth0=0.4 tox=4.1n nch=2.35e17 ijth=0 js=1e-4 jsw=1e-10 cj=1e-3 cjsw=1e-10 mj=0.4 mjsw=0.5",
        "l=0.18u w=2u ad=1p as=1p pd=3u ps=3u",
    );
    let dc = dc::DcModel {
        model: &defaults,
        temperature: &temperature,
        setup: &setup,
    };
    let cv = charges::ChargeModel {
        model: &defaults,
        temperature: &temperature,
        setup: &setup,
    };
    let zero = dc.evaluate(at(1.5, 0.), 0.).unwrap();
    let zero_charge = cv.evaluate(&zero, at(1.5, 0.)).unwrap();
    // Bias far below an ulp of the exponential's value at zero or the
    // saturation voltage. The first-order small-signal limits are exact
    // here to many more digits than the asserted relative tolerance.
    for v in [1e-18, 1e-22, 1e-26] {
        let bias = dc::Bias {
            vbs: -v,
            vgs: 1.5,
            vds: v,
        };
        let d = dc.evaluate(bias, 0.).unwrap();
        let q = cv.evaluate(&d, bias).unwrap();
        assert!(d.vdseff > 0. && d.ids > 0. && d.cbs < 0. && q.qbs < 0.);
        for (got, want, name) in [
            (d.ids / v, zero.gds, "drain current slope"),
            (d.cbs / -v, zero.gbs, "junction DC slope"),
            (q.qbs / -v, zero_charge.capbs, "junction CV slope"),
        ] {
            assert!(
                (got - want).abs() <= 1e-10 * want.abs(),
                "{name}: {got:e} vs {want:e}"
            );
        }
    }
}

#[test]
fn saturation_root_has_continuous_small_resistance_limit() {
    // The general quadratic becomes linear as rds -> 0, lambda -> 1.
    // Direct subtraction of the two roots loses all useful digits there.
    let reference = evaluate("vth0=0.4 a1=0 a2=1 rdsw=0", "l=0.18u w=2u", at(1.5, 1.));
    for resistance in ["1e-8", "1e-12", "1e-16", "1e-20"] {
        let value = evaluate(
            &format!("vth0=0.4 a1=0 a2=1 rdsw={resistance}"),
            "l=0.18u w=2u",
            at(1.5, 1.),
        );
        for (got, want, name) in [
            (value.vdsat, reference.vdsat, "saturation voltage"),
            (value.ids, reference.ids, "drain current"),
            (value.gm, reference.gm, "transconductance"),
            (value.gds, reference.gds, "output conductance"),
            (value.gmbs, reference.gmbs, "body transconductance"),
        ] {
            assert!(
                (got - want).abs() <= 1e-6 * want.abs().max(1e-12),
                "{resistance} {name}: {got:e} vs {want:e}"
            );
        }
    }
}

#[test]
fn charge_conservation_and_gate_and_drain_derivatives() {
    for partition in ["0", "0.5", "1"] {
        let (defaults, temperature, setup) = prepared(
            &format!("vth0=0.45 tox=4.1n nch=2.35e17 xpart={partition}"),
            "l=0.5u w=4u",
        );
        let dc = dc::DcModel {
            model: &defaults,
            temperature: &temperature,
            setup: &setup,
        };
        let cv = charges::ChargeModel {
            model: &defaults,
            temperature: &temperature,
            setup: &setup,
        };
        let evaluate = |bias| {
            let d = dc.evaluate(bias, 0.).unwrap();
            let q = cv.evaluate(&d, bias).unwrap();
            (d, q)
        };
        for (gate, drain) in [(0.7, 0.1), (1.5, 0.2), (2.0, 1.0)] {
            let bias = dc::Bias {
                vbs: -0.2,
                vgs: gate,
                vds: drain,
            };
            let (d, q) = evaluate(bias);
            let sum = q.qgate + q.qbulk + q.qdrn + q.qsrc;
            let scale = q.qgate.abs() + q.qbulk.abs() + q.qdrn.abs() + q.qsrc.abs();
            assert!(sum.abs() <= 4. * f64::EPSILON * scale);
            let h = 1e-5;
            let (gp, gqp) = evaluate(dc::Bias {
                vgs: gate + h,
                ..bias
            });
            let (gm, gqm) = evaluate(dc::Bias {
                vgs: gate - h,
                ..bias
            });
            let (dp, dqp) = evaluate(dc::Bias {
                vds: drain + h,
                ..bias
            });
            let (dm, dqm) = evaluate(dc::Bias {
                vds: drain - h,
                ..bias
            });
            for (derivative, approximate, name, absolute) in [
                (d.gm, (gp.ids - gm.ids) / (2. * h), "gm", 1e-10),
                (d.gds, (dp.ids - dm.ids) / (2. * h), "gds", 1e-10),
                (q.cggb, (gqp.qgate - gqm.qgate) / (2. * h), "cgg", 1e-19),
                (q.cgdb, (dqp.qgate - dqm.qgate) / (2. * h), "cgd", 1e-19),
                (q.cbgb, (gqp.qbulk - gqm.qbulk) / (2. * h), "cbg", 1e-19),
                (q.cbdb, (dqp.qbulk - dqm.qbulk) / (2. * h), "cbd", 1e-19),
                (q.cdgb, (gqp.qdrn - gqm.qdrn) / (2. * h), "cdg", 1e-19),
                (q.cddb, (dqp.qdrn - dqm.qdrn) / (2. * h), "cdd", 1e-19),
            ] {
                assert!(
                    (derivative - approximate).abs()
                        <= absolute + 5e-4 * derivative.abs().max(approximate.abs()),
                    "{partition} {gate} {drain} {name}: {derivative:e} vs {approximate:e}"
                );
            }
        }
    }
}
