//! Leakage and final current scaling in original C statement order.
#![expect(
    clippy::similar_names,
    reason = "C equation names preserve the reference distinctions"
)]
#![expect(
    clippy::float_cmp,
    reason = "C branch predicates compare floating point exactly"
)]
#![expect(
    clippy::too_many_lines,
    reason = "preserve cohesive C physics equation sequence"
)]
use super::{
    DELTA_3, DELTA_4, EXP_THRESHOLD, EXPL_THRESHOLD, Instance, MAX_EXP, MAX_EXPL, MIN_EXP,
    MIN_EXPL, Model, SizeParam,
};
use super::{
    bias::LoadBias,
    channel::{Electrostatics, Transport},
    reduction::residual,
};
use crate::devices::bsim_physics::thermal_softplus;
use crate::maths::taylor::{Constitutive, Local};
use crate::primitives::SpiceResult;
use std::cmp::Ordering;

/// Source/drain tunneling fractions. The two regularized fractions are
/// not renormalized: terminal incidence conserves each physical current.
fn gate_partition<S: Constitutive>(argument: S) -> SpiceResult<[S; 2]> {
    const REGULARIZATION: f64 = 1e-4;
    let above = argument.compare(EXP_THRESHOLD) == Some(Ordering::Greater);
    let below = argument.compare(-EXP_THRESHOLD) == Some(Ordering::Less);
    let c = |x| argument.constant(x);
    let (source, drain, source_slope, drain_slope) =
        if argument.primal().is_finite() && (above || below) {
            let exponential = if above { MAX_EXP } else { MIN_EXP };
            let difference = exponential - 1.;
            let inverse = c(1.).divide(&argument)?;
            let normalized = inverse
                .clone()
                .scale(2. * REGULARIZATION)
                .mul_add(&inverse, &c(1.));
            let source_scaled = inverse
                .clone()
                .scale(difference + REGULARIZATION)
                .offset(-1.)
                .divide(&normalized)?;
            let drain_scaled = inverse
                .clone()
                .scale(REGULARIZATION - difference)
                .offset(exponential)
                .divide(&normalized)?;
            let slope_factor = inverse.clone().multiply(&inverse).divide(&normalized)?;
            let source_slope = source_scaled
                .clone()
                .scale(-2.)
                .offset(-1.)
                .multiply(&slope_factor);
            let drain_slope = drain_scaled
                .clone()
                .scale(-2.)
                .offset(exponential)
                .multiply(&slope_factor);
            (
                source_scaled.multiply(&inverse),
                drain_scaled.multiply(&inverse),
                source_slope,
                drain_slope,
            )
        } else {
            let denominator = argument.clone().mul_add(&argument, &c(2. * REGULARIZATION));
            if argument.primal().abs() <= 0.5 {
                small_gate_partition(&argument, &denominator)?
            } else {
                let (exponential, derivative) = if above {
                    (c(MAX_EXP), c(0.))
                } else if below {
                    (c(MIN_EXP), c(0.))
                } else {
                    let exponential = argument.clone().exp();
                    (exponential.clone(), exponential)
                };
                let difference = exponential.clone().offset(-1.);
                let source = difference
                    .clone()
                    .add(&argument.clone().scale(-1.))
                    .offset(REGULARIZATION)
                    .divide(&denominator)?;
                let drain = argument
                    .clone()
                    .mul_add(&exponential, &difference.scale(-1.).offset(REGULARIZATION))
                    .divide(&denominator)?;
                let source_slope = argument
                    .clone()
                    .scale(-2.)
                    .mul_add(&source, &derivative.clone().offset(-1.))
                    .divide(&denominator)?;
                let drain_numerator = argument
                    .clone()
                    .offset(-1.)
                    .mul_add(&derivative, &exponential);
                let drain_slope = argument
                    .clone()
                    .scale(-2.)
                    .mul_add(&drain, &drain_numerator)
                    .divide(&denominator)?;
                (source, drain, source_slope, drain_slope)
            }
        };
    Ok([
        argument
            .clone()
            .compose_derivative(source.primal(), &source_slope),
        argument.compose_derivative(drain.primal(), &drain_slope),
    ])
}

/// Third-order exponential remainder x³ P(x). At |x|<=1/2 the error
/// after 15! is below binary64 rounding. The local slopes retain the
/// quadratic zero at equilibrium without subtracting nearly equal terms.
fn small_gate_partition<S: Constitutive>(
    argument: &S,
    denominator: &S,
) -> SpiceResult<(S, S, S, S)> {
    const COEFFICIENTS: [f64; 13] = [
        1. / 1_307_674_368_000.,
        1. / 87_178_291_200.,
        1. / 6_227_020_800.,
        1. / 479_001_600.,
        1. / 39_916_800.,
        1. / 3_628_800.,
        1. / 362_880.,
        1. / 40_320.,
        1. / 5_040.,
        1. / 720.,
        1. / 120.,
        1. / 24.,
        1. / 6.,
    ];
    let remainder = COEFFICIENTS
        .into_iter()
        .fold(argument.constant(0.), |acc, coefficient| {
            if cfg!(target_feature = "fma") {
                acc.mul_add(argument, &argument.constant(coefficient))
            } else {
                acc.multiply(argument).offset(coefficient)
            }
        });
    let square = argument.clone().divide(denominator)?.multiply(argument);
    let second = argument
        .clone()
        .mul_add(&remainder, &argument.constant(0.5));
    let drain_remainder = argument
        .clone()
        .offset(-1.)
        .mul_add(&remainder, &argument.constant(0.5));
    let source = argument
        .clone()
        .multiply(&square)
        .mul_add(&remainder, &argument.constant(0.5));
    let drain = argument
        .clone()
        .multiply(&square)
        .mul_add(&drain_remainder, &argument.constant(0.5));
    let source_slope = square
        .clone()
        .scale(-2.)
        .mul_add(&remainder, &second)
        .multiply(&square);
    let drain_slope = square
        .clone()
        .scale(-2.)
        .mul_add(
            &drain_remainder,
            &argument.clone().multiply(&second).offset(1.),
        )
        .multiply(&square);
    Ok((source, drain, source_slope, drain_slope))
}

/// Oxide tunneling exponent scale * (a-b*v)*(1+c*v). Factoring retains
/// both exact roots. Preserve the reference IEEE propagation for
/// nonfinite scalar inputs and the exact clipped physical derivatives.
fn tunneling_barrier<S: Constitutive>(
    voltage: S,
    scale: f64,
    a: f64,
    b: f64,
    c: f64,
) -> (S, Option<S>) {
    let (argument, polynomial_slope) = if [voltage.primal(), scale, a, b, c]
        .iter()
        .all(|v| v.is_finite())
    {
        let left = voltage.constant(-b).mul_add(&voltage, &voltage.constant(a));
        let right = voltage.constant(c).mul_add(&voltage, &voltage.constant(1.));
        let slope = voltage.constant(c).mul_add(&left, &right.clone().scale(-b));
        (left.multiply(&right).scale(scale), slope)
    } else {
        let linear = a * c - b;
        let quadratic = b * c;
        let argument = voltage
            .clone()
            .scale(-quadratic)
            .multiply(&voltage)
            .add(&voltage.clone().scale(linear))
            .offset(a)
            .scale(scale);
        let slope = voltage.clone().scale(-2. * quadratic).offset(linear);
        (argument, slope)
    };
    if argument.compare(EXP_THRESHOLD) == Some(Ordering::Greater) {
        (argument.constant(MAX_EXP), None)
    } else if argument.compare(-EXP_THRESHOLD) == Some(Ordering::Less) {
        (argument.constant(MIN_EXP), None)
    } else {
        let exponential = argument.exp();
        let slope = polynomial_slope.multiply(&exponential).scale(scale);
        (
            voltage.compose_derivative(exponential.primal(), &slope),
            Some(slope),
        )
    }
}

/// One edge's GIDL/GISL model coefficients.
struct EdgeLeakage {
    amplitude: f64,
    field: f64,
    body: f64,
    offset: f64,
    gate: f64,
    shift: f64,
    exponent: f64,
}

fn edge_leakage<S: Constitutive>(
    model: &Model,
    p: &SizeParam,
    voltage: S,
    gate: S,
    gate_partial: f64,
    junction: S,
    parameters: EdgeLeakage,
    oxide: f64,
) -> SpiceResult<Local<S, 3>> {
    let c = |x| Local::new(voltage.constant(x), [0.; 3]);
    let EdgeLeakage {
        amplitude,
        field,
        body,
        offset,
        gate: gate_factor,
        shift,
        exponent,
    } = parameters;
    if amplitude <= 0.
        || field <= 0.
        || if model.gidlMod == 0 {
            body <= 0.
        } else {
            body < 0.
        }
    {
        return Ok(c(0.));
    }
    let old_model = model.gidlMod == 0;
    let raw = Local::new(voltage.clone(), [0., 1., 0.]);
    let gate = Local::new(gate, [gate_partial, 0., 0.]);
    let gate_factor = if old_model { 1. } else { gate_factor };
    let material_shift = if model.mtrlMod == 0 { 0. } else { p.vfbsd };
    let drive = raw
        .add(&gate.scale(-gate_factor))
        .offset(material_shift - offset)
        .scale(oxide.recip());
    if drive.compare(0.) != Some(Ordering::Greater) {
        return Ok(c(0.));
    }
    let junction = Local::new(junction, [0., if old_model { -1. } else { 0. }, 1.]);
    if old_model && junction.compare(0.) == Some(Ordering::Greater) {
        return Ok(c(0.));
    }
    let limit = if old_model { 100. } else { EXPL_THRESHOLD };
    let min_exp = if old_model {
        3.720_075_976e-44
    } else {
        MIN_EXPL
    };
    let base = if drive.compare(field / limit) == Some(Ordering::Greater) {
        let field_exponential = c(field).divide(&drive)?;
        drive
            .multiply(&field_exponential.scale(-1.).exp())
            .scale(amplitude * p.weffCJ)
    } else {
        drive.scale(amplitude * p.weffCJ * min_exp)
    };
    if old_model {
        let cubic = junction
            .clone()
            .multiply(&junction)
            .multiply(&junction)
            .scale(-1.);
        let factor = cubic.clone().divide(&cubic.offset(body))?;
        return Ok(base.multiply(&factor));
    }

    let mut body_argument = junction.offset(-shift);
    if body_argument.compare(model.gidlclamp) == Some(Ordering::Greater) {
        // The reference stamp retains the body partial at the clamp;
        // the actual physical germ is constant there.
        body_argument.value = body_argument.value.constant(model.gidlclamp);
    }
    let sign = body_argument.compare(0.);
    let enhancement = if body_argument.primal() == 0. {
        // At a one-sided zero crossing the reciprocal exponential has
        // either a clipped upper limit or a flat zero germ.
        match sign {
            Some(Ordering::Less) if exponent > 0. => c(0.),
            Some(Ordering::Greater) if exponent < 0. => c(0.),
            Some(Ordering::Less | Ordering::Greater) if exponent == 0. => c(1.),
            _ => c(MAX_EXPL),
        }
    } else {
        let argument = c(exponent).divide(&body_argument)?;
        if argument.compare(EXPL_THRESHOLD) == Some(Ordering::Less) {
            argument.exp()
        } else {
            c(MAX_EXPL)
        }
    };
    Ok(base.multiply(&enhancement))
}

fn edge_tunneling<S: Constitutive>(
    voltage: S,
    shift: f64,
    amplitude: f64,
    exponent: f64,
    barrier: [f64; 3],
) -> SpiceResult<Local<S, 1>> {
    let voltage = Local::new(voltage, [1.]);
    let offset = voltage.clone().offset(-shift);
    let scale = offset.primal().abs().max(0.01);
    let normalized = offset.scale(scale.recip());
    let softened = normalized
        .clone()
        .multiply(&normalized)
        .offset((0.01 / scale).powi(2))
        .sqrt()?
        .scale(scale);
    let [a, b, c] = barrier;
    let (transmission, _) = tunneling_barrier(softened.clone(), exponent, a, b, c);
    Ok(voltage
        .multiply(&softened)
        .multiply(&transmission)
        .scale(amplitude))
}

/// Currents in the reference orientation, including physical finger
/// multiplicity. Gate-edge and GIDL/GISL currents keep their raw side
/// orientation; channel and bulk tunneling use the selected source.
pub(super) struct Leakage<S> {
    pub(super) channel: S,
    pub(super) substrate: S,
    pub(super) gidl: S,
    pub(super) gisl: S,
    pub(super) igcs: S,
    pub(super) igcd: S,
    pub(super) igs: S,
    pub(super) igd: S,
    pub(super) igb: S,
}

/// Reference load wrapper; all physical current values come from the
/// constitutive stage shared with Taylor evaluation.
pub(super) fn evaluate(
    model: &Model,
    here: &mut Instance,
    pParam: &SizeParam,
    bias: &LoadBias,
    electrostatics: &Electrostatics,
    transport: &Transport,
) -> SpiceResult<f64> {
    Ok(evaluate_constitutive(
        model,
        here,
        pParam,
        [bias.vbs, bias.vgs, bias.vds],
        electrostatics,
        transport,
    )?
    .channel)
}

pub(super) fn evaluate_constitutive<S: Constitutive>(
    model: &Model,
    here: &mut Instance,
    p: &SizeParam,
    [vbs, vgs, vds]: [S; 3],
    electro: &Electrostatics<S>,
    transport: &Transport<S>,
) -> SpiceResult<Leakage<S>> {
    let nf = here.nf;
    let oxide = if model.mtrlMod == 0 {
        3. * electro.toxe
    } else {
        model.epsrsub * electro.toxe / electro.epsrox
    };
    let vbd = vbs.clone().add(&vds.clone().scale(-1.));
    let vgd = vgs.clone().add(&vds.clone().scale(-1.));
    let gidl = edge_leakage(
        model,
        p,
        vds.clone(),
        electro.vgs_eff_raw.clone(),
        here.dvgs_eff_dvg,
        vbd,
        EdgeLeakage {
            amplitude: p.agidl,
            field: p.bgidl,
            body: p.cgidl,
            offset: p.egidl,
            gate: p.rgidl,
            shift: p.fgidl,
            exponent: p.kgidl,
        },
        oxide,
    )?
    .scale(nf);
    let gisl = edge_leakage(
        model,
        p,
        vds.scale(-1.),
        electro.vgd_eff_raw.clone(),
        here.dvgd_eff_dvg,
        vbs,
        EdgeLeakage {
            amplitude: p.agisl,
            field: p.bgisl,
            body: p.cgisl,
            offset: p.egisl,
            gate: p.rgisl,
            shift: p.fgisl,
            exponent: p.kgisl,
        },
        oxide,
    )?
    .scale(nf);
    [here.ggidlg, here.ggidld, here.ggidlb] = gidl.partials;
    [here.ggislg, here.ggisls, here.ggislb] = gisl.partials;
    here.ggidls = -(here.ggidlg + here.ggidld + here.ggidlb);
    here.ggisld = -(here.ggislg + here.ggisls + here.ggislb);
    let c = |x| Local::new(electro.Vgst.constant(x), [0.; 3]);
    let thermal = if model.tempMod < 2 {
        electro.Vtm
    } else {
        electro.Vtm0
    };
    let gate = Local::new(electro.Vgs.clone(), [1., 0., 0.]);
    let gate_effective = Local::new(electro.Vgs_eff.clone(), [electro.dVgs_eff_dVg, 0., 0.]);
    let body = Local::new(electro.Vbseff.clone(), [0., 0., 1.]);
    let raw_body = Local::new(electro.Vbs.clone(), [0., 0., 1.]);
    let overdrive = Local::new(
        electro.Vgsteff.clone(),
        [
            electro.dVgsteff_dVg,
            electro.dVgsteff_dVd,
            electro.dVgsteff_dVb,
        ],
    );
    let mut acc = c(0.);
    let mut inv = c(0.);
    if model.igcMod != 0 || model.igbMod != 0 {
        let flatband = here.vfbzb;
        let argument = body
            .clone()
            .add(&gate_effective.clone().scale(-1.))
            .offset(flatband - DELTA_3);
        (acc, _) = super::smooth_positive(argument, DELTA_3 * flatband.abs())?;
        if acc.compare(0.) == Some(Ordering::Less) {
            acc = c(0.);
        }
        let argument = gate_effective
            .clone()
            .add(&body.clone().scale(-1.))
            .add(&overdrive.clone().scale(-1.))
            .add(&acc)
            .offset(-flatband);
        inv = if p.k1ox == 0. {
            c(0.)
        } else if argument.compare(0.) == Some(Ordering::Less) {
            argument.scale(-1.)
        } else {
            super::depletion_charge(p.k1ox, argument)?.0
        };
        inv = inv.add(&overdrive);
    }
    let (igcs, igcd, igs, igd) = if model.igcMod != 0 {
        let auxiliary_argument = if model.igcMod == 1 {
            gate_effective
                .clone()
                .offset(-f64::from(model.r#type) * here.vth0)
        } else {
            let threshold = Local::new(
                electro.Vth.clone(),
                [0., electro.dVth_dVd, electro.dVth_dVb],
            );
            gate_effective.clone().add(&threshold.scale(-1.))
        };
        let (auxiliary, _, _) = thermal_softplus(&auxiliary_argument, &c(1.), thermal * p.nigc)?;
        let (barrier, _) = tunneling_barrier(inv.clone(), p.Bechvb, p.aigc, p.bigc, p.cigc);
        let igc = gate
            .clone()
            .multiply(&auxiliary)
            .multiply(&barrier)
            .scale(p.Aechvb * nf);
        let effective = Local::new(
            transport.Vdseff.clone(),
            [
                transport.dVdseff_dVg * electro.dVgsteff_dVg,
                transport
                    .dVdseff_dVg
                    .mul_add(electro.dVgsteff_dVd, transport.dVdseff_dVd),
                transport
                    .dVdseff_dVg
                    .mul_add(electro.dVgsteff_dVb, transport.dVdseff_dVb),
            ],
        );
        let partition_coefficient = if model.pigcdGiven {
            c(p.pigcd)
        } else {
            let width = overdrive.clone().offset(1e-20);
            let coefficient = c(-p.Bechvb).divide(&width)?.divide(&width)?;
            coefficient.multiply(&effective.clone().scale(-0.5).divide(&width)?.offset(1.))
        };
        let [source_fraction, drain_fraction] =
            gate_partition(partition_coefficient.multiply(&effective).scale(-1.))?;
        let source = igc.clone().multiply(&source_fraction);
        let drain = igc.multiply(&drain_fraction);
        let shift = p.vfbsd + p.vfbsdoff;
        let igs = edge_tunneling(
            vgs,
            shift,
            p.AechvbEdgeS * nf,
            p.BechvbEdge,
            [p.aigs, p.bigs, p.cigs],
        )?;
        let igd = edge_tunneling(
            vgd,
            shift,
            p.AechvbEdgeD * nf,
            p.BechvbEdge,
            [p.aigd, p.bigd, p.cigd],
        )?;
        (source, drain, igs, igd)
    } else {
        (
            c(0.),
            c(0.),
            Local::new(electro.Vgst.constant(0.), [0.]),
            Local::new(electro.Vgst.constant(0.), [0.]),
        )
    };
    let [here_g, here_d, here_b] = igcs.partials;
    here.gIgcsg = here_g;
    here.gIgcsd = here_d;
    here.gIgcsb = here_b * electro.dVbseff_dVb;
    let [here_g, here_d, here_b] = igcd.partials;
    here.gIgcdg = here_g;
    here.gIgcdd = here_d;
    here.gIgcdb = here_b * electro.dVbseff_dVb;
    here.gIgcss = -(here.gIgcsg + here.gIgcsd + here.gIgcsb);
    here.gIgcds = -(here.gIgcdg + here.gIgcdd + here.gIgcdb);
    here.gIgsg = igs.partials[0];
    here.gIgss = -here.gIgsg;
    here.gIgdg = igd.partials[0];
    here.gIgdd = -here.gIgdg;

    let igb = if model.igbMod != 0 {
        let gate_bulk = gate.add(&raw_body.scale(-1.));
        let accumulation_argument = body.add(&gate_effective.scale(-1.)).offset(here.vfbzb);
        let (acc_auxiliary, _, _) =
            thermal_softplus(&accumulation_argument, &c(1.), thermal * p.nigbacc)?;
        let prefactor = 4.97232e-7 * p.weff * p.leff * p.ToxRatio * nf;
        let exponent = -7.45669e11 * electro.toxe;
        let (barrier, _) = tunneling_barrier(acc, exponent, p.aigbacc, p.bigbacc, p.cigbacc);
        let accumulation = gate_bulk
            .clone()
            .multiply(&acc_auxiliary)
            .multiply(&barrier)
            .scale(prefactor);
        let inversion_argument = inv.clone().offset(-p.eigbinv);
        let (inv_auxiliary, _, _) =
            thermal_softplus(&inversion_argument, &c(1.), thermal * p.nigbinv)?;
        let (barrier, _) =
            tunneling_barrier(inv, exponent * 1.31724, p.aigbinv, p.bigbinv, p.cigbinv);
        accumulation.add(
            &gate_bulk
                .multiply(&inv_auxiliary)
                .multiply(&barrier)
                .scale(prefactor * 0.75610),
        )
    } else {
        c(0.)
    };
    let [gg, gd, gb] = igb.partials;
    here.gIgbg = gg;
    here.gIgbd = gd;
    here.gIgbb = gb * electro.dVbseff_dVb;
    here.gIgbs = -(here.gIgbg + here.gIgbd + here.gIgbb);
    let result = Leakage {
        channel: transport.unscaled_drain_current.clone().scale(nf),
        substrate: transport.substrate_current.clone().scale(nf),
        gidl: gidl.value,
        gisl: gisl.value,
        igcs: igcs.value,
        igcd: igcd.value,
        igs: igs.value,
        igd: igd.value,
        igb: igb.value,
    };
    here.cd = result.channel.primal();
    here.csub = result.substrate.primal();
    here.Igidl = result.gidl.primal();
    here.Igisl = result.gisl.primal();
    here.Igcs = result.igcs.primal();
    here.Igcd = result.igcd.primal();
    here.Igs = result.igs.primal();
    here.Igd = result.igd.primal();
    here.Igb = result.igb.primal();
    here.gds *= nf;
    here.gm *= nf;
    here.gmbs *= nf;
    here.IdovVds *= nf;
    here.gbbs *= nf;
    here.gbgs *= nf;
    here.gbds *= nf;
    // Noise auxiliaries are observations, not part of F or Q.
    if model.tnoiMod == 0 {
        let bulk = transport.Abulk0_Q.primal() * p.abulkCVfactor;
        let overdrive = electro.Vgsteff.primal();
        let saturation = overdrive / bulk;
        let drain = electro.Vds.primal();
        let effective = if drain == 0. {
            0.
        } else {
            super::smooth_drain(saturation, drain, DELTA_4)?.effective
        };
        let weighted = bulk * effective;
        let width = 12. * ((-0.5_f64).mul_add(weighted, overdrive) + 1e-20);
        let correction = weighted * (effective / width);
        here.qinv = transport.Coxeff.primal()
            * p.weffCV
            * nf
            * p.leffCV
            * residual([(overdrive, 1.), (-0.5, weighted), (bulk, correction)]);
    } else if model.tnoiMod == 2 {
        here.noiGd0 = nf * transport.beta.primal() * electro.Vgsteff.primal()
            / transport.gche.primal().mul_add(transport.Rds.primal(), 1.);
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tunneling_barrier_preserves_both_quadratic_roots_and_clamps() {
        let epsilon = f64::EPSILON;
        let voltage = 1. - epsilon;
        let scale = 1e30;
        // Exact (1+epsilon)*(1-epsilon) = 1-epsilon². The
        // expanded quadratic rounds away the residual at either root.
        let residual_factor = epsilon * epsilon;
        for (a, b, c, other_factor, polynomial_slope) in [
            (
                1.,
                1. + epsilon,
                1.5,
                1. + 1.5 * voltage,
                1.5 * residual_factor - (1. + epsilon) * (1. + 1.5 * voltage),
            ),
            (
                2.,
                1.,
                -(1. + epsilon),
                2. - voltage,
                -(1. + epsilon) * (2. - voltage) - residual_factor,
            ),
        ] {
            let expected = (scale * residual_factor * other_factor).exp();
            let (exponential, slope) = tunneling_barrier(voltage, scale, a, b, c);
            assert!((exponential / expected - 1.).abs() < 8. * f64::EPSILON);
            assert!(
                (slope.unwrap() / (expected * scale * polynomial_slope) - 1.).abs()
                    < 16. * f64::EPSILON
            );
        }
        assert_eq!(tunneling_barrier(0., 100., 1., 1., 1.), (MAX_EXP, None));
        assert_eq!(tunneling_barrier(0., -100., 1., 1., 1.), (MIN_EXP, None));
        let (exponential, slope) = tunneling_barrier(f64::INFINITY, 1., 1., 0., 1.);
        assert!(exponential.is_nan());
        assert!(slope.unwrap().is_nan());
    }

    #[test]
    fn partition_retains_the_tiny_derivative_at_half_partition() {
        for argument in [-1e-18, 1e-18] {
            let [source, drain] = gate_partition(Local::new(argument, [1.])).unwrap();
            assert_eq!(source.value, 0.5);
            assert_eq!(drain.value, 0.5);
            assert!((source.partials[0] / 2.5e-33 - 1.).abs() < 8. * f64::EPSILON);
            assert!((drain.partials[0] / 5e-33 - 1.).abs() < 8. * f64::EPSILON);
        }
    }

    #[test]
    fn partition_matches_independent_derivatives_across_clamp_regions() {
        for argument in [
            -100., -35., -33., -0.51, -0.49, -0.1, 0., 0.1, 0.49, 0.51, 33., 35., 100.,
        ] {
            let [source, drain] = gate_partition(Local::new(argument, [1.])).unwrap();
            let h = 1e-5;
            let [lo, hi, lo_half, hi_half] =
                [-h, h, -0.5 * h, 0.5 * h].map(|delta| gate_partition(argument + delta).unwrap());
            for (index, slope) in [source.partials[0], drain.partials[0]]
                .into_iter()
                .enumerate()
            {
                let difference = (4. * (hi_half[index] - lo_half[index]) / h
                    - (hi[index] - lo[index]) / (2. * h))
                    / 3.;
                assert!(
                    (slope - difference).abs() < 1e-6 * slope.abs() + 1e-8,
                    "{argument}: {slope:e}, {difference:e}"
                );
            }
        }
        let [source, drain] = gate_partition(1e200).unwrap();
        assert!(source.is_finite() && source < 0.);
        assert!(drain.is_finite() && drain > 0.);
    }
}
