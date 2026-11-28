//! Memoryless XSPICE switch resistance laws and their physical differential.
//! Unlike a threshold switch these have no trial or accepted discrete state.

use super::source::{SourceModel, numeric_scalar};
use crate::devices::physical_dae::LocalGuard;
use crate::maths::taylor::{Jet, differential::Differential};
use crate::netlist::ast::ModelCard;
use crate::primitives::{Real, SpiceError, SpiceResult};
use std::cmp::Ordering;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Law {
    CubicLog,
    LinearBounded,
    Exponential,
    Linear,
}

#[derive(Clone, Copy, Debug)]
pub(super) struct Material {
    law: Law,
    on: Real,
    off: Real,
    delta: Real,
    ron: Real,
    roff: Real,
    log_ratio: Real,
    limited: bool,
    pub(super) control_g: Real,
}

#[derive(Clone, Copy, Debug)]
pub(super) struct Point {
    pub(super) g: Real,
    pub(super) dg: Real,
}

impl Material {
    pub(super) fn compile(schema: SourceModel, card: &ModelCard) -> SpiceResult<Self> {
        schema.validate_parameters(card)?;
        let mut on = 1.;
        let mut off = 0.;
        let mut ron: Real = 1.;
        let mut roff: Real = 1e12;
        let mut control_r: Real = 1e12;
        let mut log = true;
        let mut limited = false;
        for parameter in &card.parameters {
            let value = numeric_scalar(parameter)?;
            match parameter.name.as_str() {
                "cntl_on" => on = value,
                "cntl_off" => off = value,
                "r_on" => ron = value,
                "r_off" => roff = value,
                "r_cntl_in" => control_r = value,
                "log" => log = matches!(value, 1.),
                "limit" => limited = matches!(value, 1.),
                _ => {
                    return Err(SpiceError::circuit(
                        "unregistered continuous switch parameter",
                    ));
                }
            }
        }
        let p = schema == SourceModel::PSwitch;
        // The reference raises R_on to 1 milliohm, bounds the P-switch
        // off resistance and expands an almost coincident P-switch threshold.
        ron = ron.max(1e-3);
        if p {
            roff = roff.min(1e12);
            if (on - off).abs() < 1e-12 {
                on += 0.001;
                off -= 0.001;
            }
        }
        let delta = on - off;
        if !delta.is_finite() || delta.abs() < 1e-12 || roff <= 0. || control_r <= 0. {
            return Err(SpiceError::parse(
                card.location.clone(),
                "invalid continuous switch resistance or coincident control thresholds",
            ));
        }
        let control_g = if p { control_r.recip() } else { 0. };
        if !control_g.is_finite() {
            return Err(SpiceError::parse(
                card.location.clone(),
                "unrepresentable control conductance",
            ));
        }
        Ok(Self {
            law: match (p, log) {
                (true, true) => Law::CubicLog,
                (true, false) => Law::LinearBounded,
                (false, true) => Law::Exponential,
                (false, false) => Law::Linear,
            },
            on,
            off,
            delta,
            ron,
            roff,
            log_ratio: if (ron - roff).abs() <= 0.5 * roff {
                ((ron - roff) / roff).ln_1p()
            } else {
                ron.ln() - roff.ln()
            },
            limited: p || limited,
            control_g,
        })
    }

    fn normalized(self, control: Real) -> Real {
        // Exact endpoint identity avoids cancellation in high-ratio laws.
        if control.partial_cmp(&self.on) == Some(Ordering::Equal) {
            1.
        } else if control.partial_cmp(&self.off) == Some(Ordering::Equal) {
            0.
        } else {
            (control - self.off) / self.delta
        }
    }

    pub(super) fn point(self, control: Real) -> SpiceResult<Point> {
        let local = self.local(self.normalized(control), Ordering::Equal);
        let point = local.law.point(self.delta);
        if !control.is_finite() || !point.g.is_finite() || !point.dg.is_finite() {
            return Err(SpiceError::circuit(
                "nonfinite continuous switch operating point",
            ));
        }
        Ok(point)
    }

    /// The scalar load and Taylor assembly use the same local polynomial in
    /// normalized control. Its constant term uses the stable endpoint form;
    /// the directional comparison selects the physical one-sided branch.
    fn local(self, t: Real, direction: Ordering) -> Local {
        let below = compare(t, 0., direction) == Ordering::Less;
        let above = compare(t, 1., direction) == Ordering::Greater;
        match self.law {
            Law::CubicLog => {
                if below {
                    return Local::flat(self.roff);
                }
                if compare(t, 1., direction) != Ordering::Less || self.ron > self.roff {
                    return Local::flat(self.ron);
                }
                if (matches!(t, 0.) && direction == Ordering::Equal) || matches!(self.log_ratio, 0.)
                {
                    return Local::flat(self.roff);
                }
                let s = 1. - t;
                let log_r = if t <= 0.5 {
                    self.log_ratio
                        .mul_add(t * t * 2.0_f64.mul_add(-t, 3.), self.roff.ln())
                } else {
                    (-self.log_ratio).mul_add(s * s * 2.0_f64.mul_add(-s, 3.), self.ron.ln())
                };
                Local::new(ResistanceGerm::Log {
                    g: self.endpoint_conductance(t, log_r),
                    coefficients: [
                        self.log_ratio * 6. * t * s,
                        self.log_ratio * 3. * 2_f64.mul_add(-t, 1.),
                        -2. * self.log_ratio,
                    ],
                })
            }
            Law::Exponential => {
                if self.limited {
                    if self.ron > self.roff || below {
                        return Local::flat(self.roff);
                    }
                    if above {
                        return Local::flat(self.ron);
                    }
                }
                if matches!(self.log_ratio, 0.) {
                    return Local::flat(self.ron);
                }
                let log_r = interpolate(t, self.roff.ln(), self.ron.ln(), self.log_ratio);
                let law = ResistanceGerm::Log {
                    g: self.endpoint_conductance(t, log_r),
                    coefficients: [self.log_ratio, 0., 0.],
                };
                if self.limited {
                    Local::new(law)
                } else {
                    Local::floor(law, log_r, self.log_ratio, 1e-9_f64.ln(), direction)
                }
            }
            Law::LinearBounded | Law::Linear => {
                let bounded = self.law == Law::LinearBounded;
                if self.limited {
                    if !bounded && self.ron > self.roff {
                        return Local::flat(self.roff);
                    }
                    if below {
                        return Local::flat(if bounded {
                            self.roff.max(1e-9)
                        } else {
                            self.roff
                        });
                    }
                    if above {
                        return Local::flat(self.ron);
                    }
                }
                let slope = self.ron - self.roff;
                if matches!(slope, 0.) {
                    return Local::flat(self.ron);
                }
                let r = interpolate(t, self.roff, self.ron, slope);
                let law = ResistanceGerm::Linear { r, slope };
                if bounded || !self.limited {
                    Local::floor(law, r, slope, 1e-9, direction)
                } else {
                    Local::new(law)
                }
            }
        }
    }

    fn endpoint_conductance(self, t: Real, log_r: Real) -> Real {
        if matches!(t, 0.) {
            self.roff.recip()
        } else if matches!(t, 1.) {
            self.ron.recip()
        } else {
            (-log_r).exp()
        }
    }

    pub(super) fn taylor<const N: usize>(
        self,
        control: Differential<N>,
        guards: &mut Option<Vec<LocalGuard>>,
    ) -> SpiceResult<Differential<N>> {
        let t = control
            .value
            .clone()
            .offset(-self.off)
            .scale(self.delta.recip())
            .with_value(self.normalized(control.value.value()));
        let local = self.local(t.value(), t.direction());
        let delta = t.clone().with_value(0.);
        if let Some(guards) = guards {
            if self.limited && t.value().is_finite() {
                guards.push(LocalGuard::interval(0, t, &[0., 1.]));
            }
            if let Some(floor) = local.floor {
                let argument = delta.clone().scale(floor.slope).offset(floor.value);
                if argument.check().is_ok() {
                    guards.push(LocalGuard::sign(1, 0, argument, floor.boundary));
                }
            }
        }
        let (g, slope) = local.law.taylor(&delta, self.delta)?;
        Ok(control.lift(g, &slope))
    }
}

/// Compare the accepted coordinate and then its first nonzero time derivative.
fn compare(value: Real, boundary: Real, direction: Ordering) -> Ordering {
    value
        .partial_cmp(&boundary)
        .map_or(Ordering::Equal, |order| {
            if order == Ordering::Equal {
                direction
            } else {
                order
            }
        })
}

fn interpolate(t: Real, off: Real, on: Real, slope: Real) -> Real {
    if t <= 0.5 {
        slope.mul_add(t, off)
    } else {
        (-slope).mul_add(1. - t, on)
    }
}

#[derive(Clone, Copy)]
enum ResistanceGerm {
    Flat(Real),
    Linear { r: Real, slope: Real },
    // Polynomial log(R) about the accepted normalized coordinate.
    Log { g: Real, coefficients: [Real; 3] },
}

impl ResistanceGerm {
    fn point(self, width: Real) -> Point {
        match self {
            Self::Flat(g) => Point { g, dg: 0. },
            Self::Linear { r, slope } => {
                let g = r.recip();
                Point {
                    g,
                    dg: -(slope / width) * g * g,
                }
            }
            Self::Log {
                g,
                coefficients: [slope, _, _],
            } => Point {
                g,
                dg: (-slope / width) * g,
            },
        }
    }

    fn taylor(self, delta: &Jet, width: Real) -> SpiceResult<(Jet, Jet)> {
        let point = self.point(width);
        let (g, slope) = match self {
            Self::Flat(g) => (
                Jet::constant(delta.order(), g),
                Jet::constant(delta.order(), 0.),
            ),
            Self::Linear { r, slope } => {
                let g = delta.clone().scale(slope).offset(r).reciprocal()?;
                let derivative = g.multiply(&g).scale(-slope / width);
                (g, derivative)
            }
            Self::Log {
                coefficients: [a, b, c],
                ..
            } => {
                let square = delta.multiply(delta);
                let log_delta = delta
                    .clone()
                    .scale(a)
                    .add(&square.clone().scale(b))
                    .add(&square.multiply(delta).scale(c));
                let g = log_delta.scale(-1.).exp_with_value(point.g);
                let log_slope = delta
                    .clone()
                    .scale(2. * b)
                    .add(&square.scale(3. * c))
                    .offset(a)
                    .scale(-width.recip());
                let derivative = g.multiply(&log_slope);
                (g, derivative)
            }
        };
        g.check()?;
        slope.check()?;
        Ok((g, slope))
    }
}

#[derive(Clone, Copy)]
struct FloorGuard {
    value: Real,
    slope: Real,
    boundary: Real,
}

#[derive(Clone, Copy)]
struct Local {
    law: ResistanceGerm,
    floor: Option<FloorGuard>,
}

impl Local {
    const fn new(law: ResistanceGerm) -> Self {
        Self { law, floor: None }
    }

    const fn flat(resistance: Real) -> Self {
        Self::new(ResistanceGerm::Flat(resistance.recip()))
    }

    fn floor(
        law: ResistanceGerm,
        value: Real,
        slope: Real,
        boundary: Real,
        direction: Ordering,
    ) -> Self {
        let orientation = if slope < 0. {
            direction.reverse()
        } else {
            direction
        };
        let law = if compare(value, boundary, orientation) == Ordering::Greater {
            law
        } else {
            ResistanceGerm::Flat(1e-9_f64.recip())
        };
        Self {
            law,
            floor: Some(FloorGuard {
                value,
                slope,
                boundary,
            }),
        }
    }
}
