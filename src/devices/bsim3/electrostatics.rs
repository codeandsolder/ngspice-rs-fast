//! Shared BSIM3 electrostatic preamble: body smoothing, threshold,
//! subthreshold slope, poly depletion and effective gate overdrive.
//!
//! The reference tangents retain b3ld.c's conventions. The physical
//! Taylor instantiation differentiates the same value operations, including
//! the actual body-bias clamp, rather than differentiating those tangents.
use std::cmp::Ordering;

use super::dc::DcModel;
use crate::maths::taylor::arithmetic::Constitutive;
use crate::primitives::{Real, SpiceResult};

const EXP_THRESHOLD: Real = 34.;
const MIN_EXP: Real = 1.713_908_431e-15;
const EPS_SI: Real = 1.035_94e-10;
const CHARGE_Q: Real = 1.602_19e-19;

pub(super) struct Body<S> {
    pub(super) effective: S,
    pub(super) d_effective: Real,
    pub(super) phis: S,
    pub(super) d_phis: Real,
    pub(super) sqrt_phis: S,
    pub(super) d_sqrt_phis: Real,
    pub(super) depletion: S,
    pub(super) d_depletion: Real,
}

fn body<S: Constitutive>(model: &DcModel<'_>, vbs: S) -> SpiceResult<Body<S>> {
    let derived = &model.setup.derived;
    let vbsc = derived.vbsc;
    let phi = derived.phi;
    let offset = vbs.clone().offset(-vbsc).offset(-0.001);
    let root = vbs
        .constant(0.004)
        .mul_add(&vbs.constant(-vbsc), &offset.clone().multiply(&offset))
        .sqrt()?;
    let raw = offset.clone().midpoint(&root).offset(vbsc);
    let d_effective = 1_f64.midpoint(offset.primal() / root.primal());
    let effective = if raw.clone().subtract(&vbs).compare(0.) == Some(Ordering::Less) {
        vbs
    } else {
        raw
    };
    let (phis, d_phis, sqrt_phis, d_sqrt_phis) = if effective.compare(0.) == Some(Ordering::Greater)
    {
        let ratio = effective
            .constant(phi)
            .divide(&effective.clone().offset(phi))?;
        let derivative = -ratio.primal() * ratio.primal();
        let phis = ratio.scale(phi);
        let sqrt_phis = effective
            .constant(derived.phis3)
            .divide(&effective.clone().affine(0.5, phi))?;
        let d_sqrt_phis = -0.5 * sqrt_phis.primal() * sqrt_phis.primal() / derived.phis3;
        (phis, derivative, sqrt_phis, d_sqrt_phis)
    } else {
        let phis = effective.clone().scale(-1.).offset(phi);
        let sqrt_phis = phis.clone().sqrt()?;
        let d_sqrt_phis = -0.5 / sqrt_phis.primal();
        (phis, -1., sqrt_phis, d_sqrt_phis)
    };
    let depletion = sqrt_phis
        .clone()
        .scale(derived.xdep0)
        .divide(&effective.constant(derived.sqrtphi))?;
    let d_depletion = (derived.xdep0 / derived.sqrtphi) * d_sqrt_phis;
    Ok(Body {
        effective,
        d_effective,
        phis,
        d_phis,
        sqrt_phis,
        d_sqrt_phis,
        depletion,
        d_depletion,
    })
}

/// Body dependence of the short-channel characteristic length.
fn body_factor<S: Constitutive>(effective: &S, coefficient: Real) -> SpiceResult<(S, Real)> {
    let bias = effective.clone().scale(coefficient);
    if bias.compare(-0.5) != Some(Ordering::Less) {
        Ok((bias.offset(1.), coefficient))
    } else {
        let reciprocal = bias.constant(1.).divide(&bias.clone().affine(8., 3.))?;
        let derivative = coefficient * reciprocal.primal() * reciprocal.primal();
        Ok((bias.affine(3., 1.).multiply(&reciprocal), derivative))
    }
}

/// Short/narrow-channel threshold factor and C's effective-body tangent.
fn rolloff<S: Constitutive>(
    body: &Body<S>,
    geometry: Real,
    coefficient: Real,
    factor: Real,
) -> SpiceResult<(S, Real)> {
    let (body_factor, derivative) = body_factor(&body.effective, coefficient)?;
    let root = body.depletion.clone().sqrt()?;
    let length = root.clone().scale(factor).multiply(&body_factor);
    let d_length = factor
        * root.primal().mul_add(
            derivative,
            0.5 / root.primal() * body_factor.primal() * body.d_depletion,
        );
    let exponent = length.constant(geometry).divide(&length)?;
    if exponent.compare(-EXP_THRESHOLD) == Some(Ordering::Greater) {
        let exponential = exponent.clone().exp();
        let d_exp = -exponent.primal() / length.primal() * exponential.primal() * d_length;
        let derivative = 4_f64.mul_add(exponential.primal(), 1.) * d_exp;
        let result = exponential.clone().multiply(&exponential.affine(2., 1.));
        Ok((result, derivative))
    } else {
        Ok((length.constant(MIN_EXP * 2_f64.mul_add(MIN_EXP, 1.)), 0.))
    }
}

pub(super) struct Threshold<S> {
    pub(super) voltage: S,
    pub(super) d_body: Real,
    pub(super) d_drain: Real,
    pub(super) rolloff: S,
    pub(super) slope: S,
    pub(super) d_slope_body: Real,
    pub(super) d_slope_drain: Real,
}

fn threshold<S: Constitutive>(
    model: &DcModel<'_>,
    body: &Body<S>,
    drain: &S,
) -> SpiceResult<Threshold<S>> {
    let p = model.setup;
    let b = &p.binned;
    let d = &p.derived;
    let t = model.temperature;
    let leff = p.leff;
    let phi = d.phi;
    let cox = model.model.oxide_capacitance;
    let tox = model.model.values.tox.value;
    let v0 = d.vbi - phi;
    let (theta, d_theta) = rolloff(body, -0.5 * b.dvt1 * leff, b.dvt2, t.factor1)?;
    let (width_theta, d_width_theta) =
        rolloff(body, -0.5 * b.dvt1w * p.weff * leff, b.dvt2w, t.factor1)?;
    let rolloff = theta.clone().scale(b.dvt0);
    let short_shift = rolloff.clone().scale(v0);
    let d_short_shift = b.dvt0 * d_theta * v0;
    let narrow_shift = width_theta.scale(b.dvt0w).scale(v0);
    let d_narrow_shift = b.dvt0w * d_width_theta * v0;
    let temp_ratio = (t.temp - t.tnom) / t.tnom;
    let length_ratio = b.nlx / leff;
    let root_offset = length_ratio / ((1. + length_ratio).sqrt() + 1.);
    let thermal_shift = body
        .effective
        .clone()
        .affine(b.kt2, b.kt1 + b.kt1l / leff)
        .affine(temp_ratio, d.k1ox * root_offset * d.sqrtphi);
    let width_factor = tox * phi / (p.weff + b.w0);

    let eta = body.effective.clone().affine(b.etab, b.eta0);
    let (eta, eta_tangent) = if eta.compare(1e-4) == Some(Ordering::Less) {
        let reciprocal = eta.constant(1.).divide(&eta.clone().affine(-2e4, 3.))?;
        let tangent = reciprocal.primal() * reciprocal.primal();
        (eta.scale(-1.).offset(2e-4).multiply(&reciprocal), tangent)
    } else {
        (eta, 1.)
    };
    let dibl = eta.scale(d.theta0vb0);
    let d_drain = -dibl.primal();
    let dibl_shift = dibl.multiply(drain);
    let base_threshold = b.k1.mul_add(-d.sqrtphi, Real::from(p.polarity) * p.vth0);
    let root_term = body.sqrt_phis.clone().affine(d.k1ox, base_threshold);
    let inner = body
        .effective
        .clone()
        .scale(-1.)
        .scale_add(d.k2ox, &root_term)
        .subtract(&short_shift)
        .subtract(&narrow_shift);
    let voltage = body
        .effective
        .clone()
        .affine(b.k3b, b.k3)
        .scale_add(width_factor, &inner)
        .add(&thermal_shift)
        .subtract(&dibl_shift);
    let d_body = b.kt2.mul_add(
        temp_ratio,
        (b.etab * drain.primal() * d.theta0vb0).mul_add(
            -eta_tangent,
            b.k3b.mul_add(
                width_factor,
                d.k1ox.mul_add(body.d_sqrt_phis, -d.k2ox) - d_short_shift - d_narrow_shift,
            ),
        ),
    );

    let slope_depletion = body
        .depletion
        .constant(b.nfactor * EPS_SI)
        .divide(&body.depletion)?;
    let slope_coupling = drain
        .clone()
        .scale_add(b.cdscd, &body.effective.clone().affine(b.cdscb, b.cdsc));
    let slope_argument = slope_coupling
        .clone()
        .mul_add(&theta, &slope_depletion)
        .offset(b.cit)
        .divide(&theta.constant(cox))?;
    let body_tangent = b.cdscb.mul_add(
        theta.primal(),
        slope_coupling.primal().mul_add(
            d_theta,
            -slope_depletion.primal() / body.depletion.primal() * body.d_depletion,
        ),
    ) / cox;
    let drain_tangent = b.cdscd * theta.primal() / cox;
    let (slope, factor) = if slope_argument.compare(-0.5) != Some(Ordering::Less) {
        (slope_argument.offset(1.), 1.)
    } else {
        let reciprocal = slope_argument
            .constant(1.)
            .divide(&slope_argument.clone().affine(8., 3.))?;
        let factor = reciprocal.primal() * reciprocal.primal();
        (slope_argument.affine(3., 1.).multiply(&reciprocal), factor)
    };
    Ok(Threshold {
        voltage,
        d_body,
        d_drain,
        rolloff,
        slope,
        d_slope_body: body_tangent * factor,
        d_slope_drain: drain_tangent * factor,
    })
}

/// Poly depletion, with both subtractive square-root offsets rationalized.
fn poly<S: Constitutive>(model: &DcModel<'_>, gate: S) -> SpiceResult<(S, Real)> {
    let p = model.setup;
    let threshold = p.vfb + p.derived.phi;
    let ngate = p.binned.ngate;
    if ngate > 1e18 && ngate < 1e25 && gate.compare(threshold) == Some(Ordering::Greater) {
        let cox = model.model.oxide_capacitance;
        let scale = 1e6 * CHARGE_Q * EPS_SI * ngate / (cox * cox);
        let excess = gate.clone().offset(-threshold);
        let root = excess
            .clone()
            .scale(2.)
            .divide(&gate.constant(scale))?
            .offset(1.)
            .sqrt()?;
        let drop = excess.scale(2.).divide(&root.clone().offset(1.))?;
        let energy = drop
            .clone()
            .scale(0.5)
            .multiply(&drop)
            .divide(&gate.constant(scale))?;
        let shift = energy.clone().scale(-1.).offset(1.07);
        let smooth = shift
            .clone()
            .mul_add(&shift, &gate.constant(0.224))
            .sqrt()?;
        let correction = energy
            .scale(2.24)
            .divide(&shift.clone().scale(-1.).offset(2.24).add(&smooth))?;
        let derivative = (0.5 * drop.primal() / (scale * root.primal()))
            .mul_add(-(1. + shift.primal() / smooth.primal()), 1.);
        Ok((gate.subtract(&correction), derivative))
    } else {
        Ok((gate, 1.))
    }
}

pub(super) struct Overdrive<S> {
    pub(super) gate: S,
    pub(super) d_gate: Real,
    pub(super) raw: S,
    pub(super) effective: S,
    pub(super) d_effective_gate: Real,
    pub(super) d_effective_drain: Real,
    pub(super) d_effective_body: Real,
}

fn overdrive<S: Constitutive>(
    model: &DcModel<'_>,
    threshold: &Threshold<S>,
    gate: S,
) -> SpiceResult<Overdrive<S>> {
    let (gate, d_gate) = poly(model, gate)?;
    let raw = gate.clone().subtract(&threshold.voltage);
    let n = &threshold.slope;
    let vtm = model.temperature.vtm;
    let p = model.setup;
    let cox = model.model.oxide_capacitance;
    let twice_nvt = n.clone().scale(2.).scale(vtm);
    let normalized = raw.clone().divide(&twice_nvt)?;
    let offset_argument = raw
        .constant(2.)
        .mul_add(&raw.constant(p.binned.voff), &raw.clone().scale(-1.))
        .divide(&twice_nvt)?;
    let (effective, d_effective_gate, d_effective_drain, d_effective_body) =
        if normalized.compare(EXP_THRESHOLD) == Some(Ordering::Greater) {
            (raw.clone(), d_gate, -threshold.d_drain, -threshold.d_body)
        } else if offset_argument.compare(EXP_THRESHOLD) == Some(Ordering::Greater) {
            let exponent = raw
                .clone()
                .offset(-p.binned.voff)
                .divide(&n.clone().scale(vtm))?;
            let effective = exponent.clone().exp().scale(vtm * p.derived.cdep0 / cox);
            let d_effective_gate = effective.primal() / (n.primal() * vtm);
            let d_effective_drain = -d_effective_gate
                * (exponent.primal() * vtm).mul_add(threshold.d_slope_drain, threshold.d_drain);
            let d_effective_body = -d_effective_gate
                * (exponent.primal() * vtm).mul_add(threshold.d_slope_body, threshold.d_body);
            (
                effective,
                d_effective_gate * d_gate,
                d_effective_drain,
                d_effective_body,
            )
        } else {
            let exponential = normalized.exp();
            let numerator = twice_nvt.clone().multiply(&exponential.clone().ln_1p()?);
            let d_num_gate = exponential.primal() / (1. + exponential.primal());
            let d_num_body = (numerator.primal() / n.primal()).mul_add(
                threshold.d_slope_body,
                -d_num_gate
                    * (raw.primal() / n.primal()).mul_add(threshold.d_slope_body, threshold.d_body),
            );
            let d_num_drain = (numerator.primal() / n.primal()).mul_add(
                threshold.d_slope_drain,
                -d_num_gate
                    * (raw.primal() / n.primal())
                        .mul_add(threshold.d_slope_drain, threshold.d_drain),
            );
            let d_den_gate = offset_argument
                .clone()
                .exp()
                .scale(-cox / (vtm * p.derived.cdep0));
            let denominator = twice_nvt.mul_add(&d_den_gate.clone().scale(-1.), &raw.constant(1.));
            let d_den_drain = ((denominator.primal() - 1.) / n.primal()).mul_add(
                threshold.d_slope_drain,
                -d_den_gate.primal()
                    * (2. * vtm * offset_argument.primal())
                        .mul_add(-threshold.d_slope_drain, threshold.d_drain),
            );
            let d_den_body = ((denominator.primal() - 1.) / n.primal()).mul_add(
                threshold.d_slope_body,
                -d_den_gate.primal()
                    * (2. * vtm * offset_argument.primal())
                        .mul_add(-threshold.d_slope_body, threshold.d_body),
            );
            let den = denominator.primal();
            let num = numerator.primal();
            let square = den * den;
            let effective = numerator.divide(&denominator)?;
            (
                effective,
                num.mul_add(-d_den_gate.primal(), den * d_num_gate) / square * d_gate,
                num.mul_add(-d_den_drain, den * d_num_drain) / square,
                num.mul_add(-d_den_body, den * d_num_body) / square,
            )
        };
    Ok(Overdrive {
        gate,
        d_gate,
        raw,
        effective,
        d_effective_gate,
        d_effective_drain,
        d_effective_body,
    })
}

pub(super) struct Electrostatics<S> {
    pub(super) body: Body<S>,
    pub(super) threshold: Threshold<S>,
    pub(super) overdrive: Overdrive<S>,
}

pub(super) fn evaluate<S: Constitutive>(
    model: &DcModel<'_>,
    vbs: S,
    vgs: S,
    vds: &S,
) -> SpiceResult<Electrostatics<S>> {
    let body = body(model, vbs)?;
    let threshold = threshold(model, &body, vds)?;
    let overdrive = overdrive(model, &threshold, vgs)?;
    Ok(Electrostatics {
        body,
        threshold,
        overdrive,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::devices::{
        ModelContext, ModelResolver,
        bsim3::{dc::Bias, model_parameters, setup::InstanceSetup},
    };
    use crate::maths::taylor::{Jet, differential::Differential};
    use crate::netlist::{Parser, source::parse_deck_text};
    use std::path::Path;

    type Law = Differential<3>;

    fn fields<S>(state: &Electrostatics<S>) -> [&S; 7] {
        [
            &state.body.effective,
            &state.body.phis,
            &state.body.sqrt_phis,
            &state.threshold.voltage,
            &state.threshold.slope,
            &state.overdrive.gate,
            &state.overdrive.effective,
        ]
    }

    fn sample(jet: &Jet, time: Real) -> Real {
        jet.coefficients()
            .iter()
            .rev()
            .fold(0., |value: Real, coefficient| {
                value.mul_add(time, *coefficient)
            })
    }

    fn close(actual: Real, expected: Real, absolute: Real, relative: Real) {
        assert!(
            (actual - expected).abs() <= absolute + relative * expected.abs(),
            "{actual:.17e} != {expected:.17e}"
        );
    }

    #[test]
    fn exact_electrostatic_jets_track_scalar_bias_trajectories() -> SpiceResult<()> {
        for (kind, threshold) in [("nmos", 0.6), ("pmos", -0.6)] {
            let text = format!(
                "dc\nm1 d g 0 b nn l=0.25u w=0.9u\n.model nn {kind} level=8 vth0={threshold} ngate=1e20 dvt0=2 dvt0w=0.1 dvt1w=1e6 dvt2=-0.04 dvt2w=-0.03\n.end\n"
            );
            let netlist = Parser::new().parse_deck(&parse_deck_text(Path::new("dc.cir"), &text))?;
            let resolver = ModelResolver::new(&netlist.models)?;
            let resolved = resolver.resolve(&netlist.devices[0])?.unwrap();
            let params = model_parameters(&resolved)?;
            let context = ModelContext::new(27., 27.);
            let defaults = params.model_defaults(&context)?;
            let temperature = defaults.at_temperature(27., &netlist.devices[0].location)?;
            let inputs = params.instance_parameters(&netlist.devices[0], 1.)?;
            let setup = InstanceSetup::new(
                &defaults,
                &temperature,
                &inputs,
                &netlist.devices[0].location,
            )?;
            let model = DcModel {
                model: &defaults,
                temperature: &temperature,
                setup: &setup,
            };
            let scalar = |[vbs, vgs, vds]: [Real; 3]| evaluate(&model, vbs, vgs, &vds);
            for bias in [
                [-0.8, -2., 0.1],
                [-0.2, 0.3, 0.05],
                [0.1, 0.9, 0.4],
                [-0.5, 1.7, 1.0],
                [0.2, 3.5, 1.3],
            ] {
                let rate = [0.01, -0.02, 0.03];
                let [body, gate, drain] = std::array::from_fn(|i| {
                    Law::variable(i, Jet::affine(8, bias[i], rate[i])).unwrap()
                });
                let jets = evaluate(&model, body, gate, &drain)?;
                let reference = model
                    .evaluate(
                        Bias {
                            vbs: bias[0],
                            vgs: bias[1],
                            vds: bias[2],
                        },
                        context.gmin,
                    )?
                    .cv;
                let scalar_fields = [
                    reference.vbseff,
                    reference.phis,
                    reference.sqrt_phis,
                    reference.vth,
                    reference.n,
                    reference.vgs_eff,
                    reference.vgsteff,
                ];
                for (jet, expected) in fields(&jets).into_iter().zip(scalar_fields) {
                    close(jet.value.value(), expected, 1e-14, 1e-12);
                }
                for time in [-0.02, 0.02] {
                    let local = std::array::from_fn(|i| rate[i].mul_add(time, bias[i]));
                    let expected = scalar(local)?;
                    for (jet, expected) in fields(&jets).into_iter().zip(fields(&expected)) {
                        close(sample(&jet.value, time), *expected, 1e-13, 1e-9);
                    }
                    // Numerical differencing only as a regression oracle, at
                    // off-branch smooth points. No derivative approximations
                    // are used by the constitutive evaluator.
                    for coordinate in 0..3 {
                        let mut plus = local;
                        let mut minus = local;
                        let step = 1e-5;
                        plus[coordinate] += step;
                        minus[coordinate] -= step;
                        let plus = scalar(plus)?;
                        let minus = scalar(minus)?;
                        for ((jet, upper), lower) in fields(&jets)
                            .into_iter()
                            .zip(fields(&plus))
                            .zip(fields(&minus))
                        {
                            let expected = (upper - lower) / (2. * step);
                            let actual = jet.gradient[coordinate]
                                .as_ref()
                                .map_or(0., |derivative| sample(derivative, time));
                            close(actual, expected, 2e-8, 2e-5);
                        }
                    }
                }
            }
        }
        Ok(())
    }
}
