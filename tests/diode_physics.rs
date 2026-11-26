//! Diode physics (#86) through the production analysis paths, checked against
//! independent closed forms rather than C. C-golden comparisons of the same
//! physics (`m7_*` fixtures) live in `xtask golden verify`.
use ngspice_rs::analysis::{Plot, RunConfig};
use ngspice_rs::netlist::Parser;
use ngspice_rs::netlist::source::parse_deck_text;
use std::path::Path;

const K_OVER_Q: f64 = 1.38064852e-23 / 1.6021766208e-19;
const GMIN: f64 = 1e-12;

fn simulate(body: &str) -> Plot {
    let n = Parser::new()
        .parse_deck(&parse_deck_text(
            Path::new("diode_physics.cir"),
            &format!("diode physics\n{body}\n.end\n"),
        ))
        .unwrap();
    let config = RunConfig::from_netlist(&n).unwrap();
    let mut circuit = config.circuit(&n).unwrap();
    let request = config.plan_for(&n.analyses[0]).unwrap();
    request.run(&mut circuit, &config.context()).unwrap()
}
fn real(plot: &Plot, name: &str, point: usize) -> f64 {
    plot.value(name, point).unwrap().re
}

/// `diotemp.c` breakdown matching, stopped like C at RELTOL (`CKTreltol`).
fn matched_breakdown(bv: f64, ibv: f64, is: f64, vt: f64, nbv: f64, reltol: f64) -> f64 {
    let mut x = bv - nbv * vt * (1. + ibv / is).ln();
    for _ in 0..25 {
        x = bv - nbv * vt * (ibv / is + 1. - x / vt).ln();
        let matched = is * (((bv - x) / (nbv * vt)).exp() - 1. + x / vt);
        if (matched - ibv).abs() <= reltol * ibv {
            break;
        }
    }
    x
}

#[test]
fn zener_regulator_operating_point_satisfies_the_breakdown_law() {
    // Tight tolerances: with ngspice's defaults (reltol 1e-3, vntol 1e-6)
    // NIiter returns the iterate its last load used (CKTrhsOld), and C
    // ngspice (SPARSE) gives v(out) = 5.186604179174052 (the port's value
    // bit for bit), where the junction current misses the breakdown law by
    // 2.4e-3 relative (8e-5 V of the 5.19 V). RELTOL also stops diotemp.c's
    // breakdown matching, mirrored by `matched_breakdown`.
    let reltol = 1e-12;
    let p = simulate(
        "vin in 0 dc 12\nrs in out 100\ndz 0 out dz\nrl out 0 2k\n\
         .model dz d(is=5e-14 n=1.05 bv=5.1 ibv=5m nbv=1.3)\n\
         .options reltol=1e-12 vntol=1e-15 abstol=1e-18\n.op",
    );
    let vout = real(&p, "v(out)", 0);
    assert!((5.1..5.4).contains(&vout), "{vout}");
    // Current from anode (ground) to cathode (out) through the junction.
    let id = -((12. - vout) / 100. - vout / 2000.);
    let vt = K_OVER_Q * 300.15;
    let xbv = matched_breakdown(5.1, 5e-3, 5e-14, vt, 1.3, reltol);
    let vd = -vout;
    let law = -5e-14 * (-(xbv + vd) / (1.3 * vt)).exp() + GMIN * vd;
    assert!((id - law).abs() <= 1e-7 * law.abs(), "{id} vs {law}");
    // The knee: the junction carries IBV at V = -BV.
    let at_bv = -5e-14 * (-(xbv - 5.1) / (1.3 * vt)).exp();
    assert!((at_bv + 5e-3).abs() < 1e-6 * 5e-3, "{at_bv}");
}

#[test]
fn temperature_sweep_follows_the_saturation_current_law() {
    // A forward diode on a 1 mA current source with series resistance, TNOM
    // 35 C, EG/XTI and TRS: v = N Vt ln(I/IS(T) + 1) + I RS(T).
    let p = simulate(
        "i1 0 a dc 1m\nd1 a 0 dm\n\
         .model dm d(is=2e-14 n=1.05 eg=1.2 xti=2.5 tnom=35 rs=5 trs1=3m)\n\
         .options reltol=1e-12 vntol=1e-15 abstol=1e-18\n\
         .dc temp -40 125 15",
    );
    // Tight tolerances, as for the zener: with ngspice's defaults C ngspice
    // (SPARSE) sweeps v(a) = 0.8296523385601169, 0.8026055747083235, ...
    // (the port's values bit for bit), 1.4e-4 V from the law at -25 C.
    assert_eq!(p.point_count(), 12);
    let tn = 35. + 273.15;
    let mut previous = f64::INFINITY;
    for point in 0..p.point_count() {
        let celsius = -40. + 15. * point as f64;
        let t = celsius + 273.15;
        let vte = 1.05 * K_OVER_Q * t;
        let is = 2e-14 * ((t / tn - 1.) * 1.2 / vte + 2.5 / 1.05 * (t / tn).ln()).exp();
        let rs = 5. * (1. + 3e-3 * (t - tn));
        let v = real(&p, "v(a)", point);
        // Solve the junction (with gmin) for its voltage independently.
        let current = 1e-3;
        let mut vd = vte * (current / is + 1.).ln();
        for _ in 0..50 {
            let f = is * ((vd / vte).exp() - 1.) + GMIN * vd - current;
            vd -= f / (is * (vd / vte).exp() / vte + GMIN);
        }
        let expected = vd + current * rs;
        assert!(
            (v - expected).abs() < 1e-7,
            "{celsius} C: {v} vs {expected}"
        );
        assert!(v < previous, "forward voltage falls when heated");
        previous = v;
    }
}

#[test]
fn sidewall_current_and_charge_add_like_a_wider_bottom_junction() {
    // With a shared characteristic, JSW*PJ adds to IS*AREA and CJSW*PJ (same
    // potential and grading) to CJO*AREA, at any temperature.
    for temperature in ["27", "100"] {
        let p = simulate(&format!(
            ".options temp={temperature}\n\
             i1 0 a dc 1m ac 1\nd1 a 0 dsw pj=10\n\
             i2 0 b dc 1m ac 1\nd2 b 0 dsum\n\
             .model dsw d(is=1e-14 jsw=1e-15 cjo=10p cjsw=1p vj=0.8 vjsw=0.8 m=0.4 mjsw=0.4 \
             fc=0.5 fcs=0.5)\n\
             .model dsum d(is=2e-14 cjo=20p vj=0.8 m=0.4 fc=0.5)\n\
             .ac dec 3 1k 1g",
        ));
        for point in 0..p.point_count() {
            let (a, b) = (
                p.value("v(a)", point).unwrap(),
                p.value("v(b)", point).unwrap(),
            );
            assert!((a - b).magnitude() <= 1e-9 * b.magnitude(), "{a} vs {b}");
        }
    }
}

#[test]
fn transient_conserves_the_temperature_adjusted_depletion_charge() {
    // A reverse ramp charges only depletion charge (TLEVC 1 linear laws at
    // 77 C). The source's integrated current equals the closed-form charge
    // change of the bottom and sidewall junctions (leakage is negligible).
    let p = simulate(
        ".options temp=77\n\
         v1 in 0 pwl(0 0 1u 3 5u 3)\nr1 in k 1k\nd1 0 k dc pj=4\n\
         .model dc d(is=1e-20 cjo=20p vj=0.8 m=0.45 tlevc=1 cta=1.5m tpb=1.8m \
         cjsw=1p vjsw=0.6 mjsw=0.3 ctp=2m tphp=1m)\n\
         .tran 10n 5u 0 10n",
    );
    let time = p.column("time").unwrap();
    let current = p.column("i(v1)").unwrap();
    let mut delivered = 0.;
    for k in 1..time.len() {
        delivered += 0.5 * (current[k].re + current[k - 1].re) * (time[k].re - time[k - 1].re);
    }
    let dt = 77. + 273.15 - 300.15;
    let charge =
        |v: f64, c: f64, p: f64, m: f64| c * p * (1. - (1. - v / p).powf(1. - m)) / (1. - m);
    let vd = -real(&p, "v(k)", p.point_count() - 1);
    let bottom = charge(vd, 20e-12 * (1. + 1.5e-3 * dt), 0.8 - 1.8e-3 * dt, 0.45);
    let sidewall = charge(vd, 4e-12 * (1. + 2e-3 * dt), 0.6 - 1e-3 * dt, 0.3);
    // i(v1) flows from + through the source: the source delivers -i(v1).
    // The junction anode is ground, so its charge is Q(-v(k)).
    let expected = -(bottom + sidewall);
    assert!(
        (-delivered - expected).abs() < 2e-3 * expected.abs(),
        "{delivered} vs {expected}"
    );
}

#[test]
fn reused_circuit_rederives_breakdown_at_the_resolved_reltol() {
    use ngspice_rs::analysis::{AnalysisContext, AnalysisInput};
    use ngspice_rs::primitives::AnalysisKind;

    let n = Parser::new()
        .parse_deck(&parse_deck_text(
            Path::new("breakdown-reuse.cir"),
            "breakdown reuse\nv1 a 0 dc -5.1 ac 1\nd1 a 0 dm\n.model dm d(is=1e-11 bv=5 ibv=2e-9 nbv=1.3)\n.end\n",
        ))
        .unwrap();
    let config = RunConfig::from_netlist(&n).unwrap();
    // Intentionally keep Newton convergence identical at every physical
    // context, isolating the model-temperature derivation.
    let plan = config
        .plan(AnalysisInput::with_arguments(
            AnalysisKind::OperatingPoint,
            ["rtol=1e-12", "vntol=1e-15", "abstol=1e-18"],
        ))
        .unwrap();
    let mut circuit = config.circuit(&n).unwrap();
    let mut first = None;
    for (iteration, reltol) in [1e-3, 1e-9, 1e-3, 1e-9].into_iter().enumerate() {
        let context = AnalysisContext {
            reltol,
            ..config.context()
        };
        let plot = plan.run(&mut circuit, &context).unwrap();
        let current = real(&plot, "i(v1)", 0);
        let vt = K_OVER_Q * 300.15;
        let xbv = matched_breakdown(5., 2e-9, 1e-11, vt, 1.3, reltol);
        let expected = 1e-11 * ((5.1 - xbv) / (1.3 * vt)).exp() + 5.1 * GMIN;
        assert!(
            (current - expected).abs() <= 1e-8 * expected.abs() + 1e-18,
            "RELTOL={reltol}: {current} vs {expected}"
        );
        if iteration == 0 {
            first = Some(current);
        } else if iteration == 1 {
            assert!((current - first.unwrap()).abs() > 1e-3 * current.abs());
        } else if iteration == 2 {
            assert_eq!(current.to_bits(), first.unwrap().to_bits());
        }
    }

    for reltol in [0., -1., f64::NAN, f64::INFINITY] {
        let invalid = AnalysisContext {
            reltol,
            ..config.context()
        };
        assert!(plan.run(&mut circuit, &invalid).is_err());
    }
}

#[test]
fn soft_recovery_ac_has_the_physical_relaxation_pole() {
    // Independently eliminate the charge coordinate from
    // q + VP*TT*dq/dt = TT*I(v). The signed input admittance is
    // G * [1 + (1-VP)*s*TT/(1+VP*s*TT)] with CJO=0.
    for (temp, vp, scale) in [
        (27., 0., 1e6),
        (27., 0.01, 1e6),
        (27., 0.05, 1.),
        (27., 0.05, 1e6),
        (27., 0.05, 1e9),
        (27., 1., 1e6),
        (77., 0.05, 1e6),
    ] {
        let p = simulate(&format!(
            ".options temp={temp} reltol=1e-9 vntol=1e-12 abstol=1e-15\n\
             v1 a 0 dc 0.7 ac 1\nd1 a 0 dm\n\
             .model dm d(is=1e-14 tt=100n ttt1=0.003 ttt2=1e-5 vp={vp} qpscale={scale})\n\
             .ac dec 3 10k 10g"
        ));
        let t = temp + 273.15;
        let vt = K_OVER_Q * t;
        let is = 1e-14 * ((t / 300.15 - 1.) * 1.11 / vt + 3. * (t / 300.15).ln()).exp();
        let g = is * (0.7 / vt).exp() / vt + GMIN;
        let dt = temp - 27.;
        let tt = 100e-9 * (1. + 0.003 * dt + 1e-5 * dt * dt);
        for k in 0..p.point_count() {
            let wtt = std::f64::consts::TAU * real(&p, "frequency", k) * tt;
            let x = vp * wtt;
            let denom = 1. + x * x;
            let re = g * (1. + (1. - vp) * wtt * x / denom);
            let im = g * (1. - vp) * wtt / denom;
            let measured = p.value("i(v1)", k).unwrap();
            let bound = 2e-6 * re.hypot(im) + 1e-12;
            assert!(
                (measured.re + re).hypot(measured.im + im) < bound,
                "T={temp} VP={vp} scale={scale} wTT={wtt}: {measured:?} vs {}, {}",
                -re,
                -im
            );
        }
    }
}

#[test]
fn inactive_recovery_reduces_to_the_original_junction_without_an_auxiliary_node() {
    let run = |parameters: &str| {
        simulate(&format!(
            "v1 a 0 dc .7 ac 1\nd1 a 0 dm\n.model dm d(is=1e-14 cjo=1p {parameters})\n.ac dec 3 1k 1g"
        ))
    };
    for (lhs, rhs) in [("tt=100n", "tt=100n vp=0"), ("tt=0", "tt=0 vp=0.05")] {
        let a = run(lhs);
        let b = run(rhs);
        assert_eq!(a.column("i(v1)").unwrap(), b.column("i(v1)").unwrap());
        assert!(b.column("v(d1#qp)").is_none());
    }
}

#[test]
fn soft_recovery_transient_conserves_physical_charge_independent_of_coordinate_scale() {
    // An ideal voltage source holds the junction at 0.7 V before a finite
    // ramp to 0.2 V. The inhomogeneous recovery ODE can be integrated
    // analytically over the ramp, followed by exponential relaxation.
    let tt: f64 = 100e-9;
    let vp = 0.05;
    let tau = vp * tt;
    let vt = K_OVER_Q * 300.15;
    let saturation = 1e-14;
    let initial_v = 0.7;
    let final_v = 0.2;
    let ramp_start = 100e-9;
    let ramp_duration = 1e-9;
    let ramp_end = ramp_start + ramp_duration;
    let rate = (initial_v - final_v) / ramp_duration;
    let k = rate / vt;
    let decay = (-ramp_duration / tau).exp();
    let current = |v: f64| saturation * (v / vt).exp_m1() + GMIN * v;
    let initial_q = tt * current(initial_v);
    let exponential =
        saturation * (initial_v / vt).exp() * ((-k * ramp_duration).exp() - decay) / (1. - k * tau);
    let linear = GMIN * (final_v + rate * tau - (initial_v + rate * tau) * decay);
    let final_q = initial_q * decay + tt * (exponential - saturation * (1. - decay) + linear);
    let equilibrium = tt * current(final_v);
    let excess = |t: f64| (final_q - equilibrium) * (-(t - ramp_end) / tau).exp();
    for method in ["trap", "gear"] {
        for scale in [1., 1e6, 1e9] {
            let p = simulate(&format!(
                "v1 a 0 pwl(0 0.7 100n 0.7 101n 0.2 160n 0.2)\n\
                 d1 a 0 dm\n.model dm d(is=1e-14 tt=100n vp={vp} qpscale={scale})\n\
                 .options method={method} reltol=1e-6 vntol=1e-9 abstol=1e-14\n\
                 .tran 50p 160n 0 50p"
            ));
            let time = p.column("time").unwrap();
            let input = p.column("i(v1)").unwrap();
            let first = time.partition_point(|v| v.re < ramp_end + 0.25e-9);
            assert!(first + 1 < time.len());
            let mut integral = 0.;
            let conduction = current(final_v);
            for i in first..time.len() {
                let expected = (1. - vp) * excess(time[i].re) / tau;
                let recovery = input[i].re + conduction;
                assert!(
                    (recovery - expected).abs() <= 2e-3 * expected.abs() + 1e-8,
                    "{method} scale={scale} t={}: {recovery} vs {expected}",
                    time[i].re
                );
                if i > first {
                    let previous = input[i - 1].re + conduction;
                    integral += 0.5 * (recovery + previous) * (time[i].re - time[i - 1].re);
                }
            }
            let expected = (1. - vp) * (excess(time[first].re) - excess(time.last().unwrap().re));
            assert!(
                (integral - expected).abs() <= 2e-3 * expected.abs(),
                "{method} scale={scale}: integrated {integral} vs {expected}"
            );
        }
    }
}

#[test]
fn soft_recovery_approaches_the_instantaneous_diffusion_limit() {
    let run = |vp: f64| {
        simulate(&format!(
            "v1 a 0 dc .7 ac 1\nd1 a 0 dm\n.model dm d(is=1e-14 tt=100n vp={vp})\n.ac dec 3 1k 1meg"
        ))
    };
    let baseline = run(0.);
    let small = run(1e-8);
    for point in 0..baseline.point_count() {
        let want = baseline.value("i(v1)", point).unwrap();
        let actual = small.value("i(v1)", point).unwrap();
        assert!(
            (actual - want).magnitude() <= 2e-8 * want.magnitude(),
            "point={point}: {actual:?} vs {want:?}"
        );
    }
}

#[test]
fn soft_recovery_observations_are_external_terminal_quantities() {
    for rs in [0., 10.] {
        let n = Parser::new()
            .parse_deck(&parse_deck_text(
                Path::new("diode_recovery_observations.cir"),
                &format!(
                    "diode recovery observations\nv1 a 0 pulse(1 -1 20n 1n 1n 2u 5u)\n\
                     d1 a 0 dm\n.model dm d(is=1e-14 tt=100n vp=.05 rs={rs} cjo=1p)\n\
                     .options reltol=1e-6 vntol=1n abstol=1e-14\n\
                     .save v(a) i(v1) @d1[i] @d1[p]\n.tran 100p 100n 0 100p\n.end\n"
                ),
            ))
            .unwrap();
        let config = RunConfig::from_netlist(&n).unwrap();
        let mut circuit = config.circuit(&n).unwrap();
        circuit
            .set_observations(&["@d1[i]".into(), "@d1[p]".into()])
            .unwrap();
        let request = config.plan_for(&n.analyses[0]).unwrap();
        let plot = request.run(&mut circuit, &config.context()).unwrap();
        for point in 0..plot.point_count() {
            let current = real(&plot, "@d1[i]", point);
            let input = real(&plot, "i(v1)", point);
            let power = real(&plot, "@d1[p]", point);
            let voltage = real(&plot, "v(a)", point);
            assert!((current + input).abs() <= 1e-12 + 1e-6 * input.abs());
            assert!((power - voltage * current).abs() <= 1e-12 + 1e-6 * power.abs());
        }
    }
}
