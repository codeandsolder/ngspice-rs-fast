//! Memoryless XSPICE switch resistance laws and their physical differential.
//! Unlike a threshold switch these have no trial or accepted discrete state.

use super::source::{SourceModel, numeric_scalar};
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

    fn endpoint(resistance: Real) -> Point {
        Point {
            g: resistance.recip(),
            dg: 0.,
        }
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
        let t = self.normalized(control);
        let point = match self.law {
            Law::CubicLog => self.cubic(t),
            Law::LinearBounded | Law::Linear => self.linear(t),
            Law::Exponential => self.exponential(t),
        };
        if !control.is_finite() || !point.g.is_finite() || !point.dg.is_finite() {
            return Err(SpiceError::circuit(
                "nonfinite continuous switch operating point",
            ));
        }
        Ok(point)
    }

    fn cubic(self, t: Real) -> Point {
        if t < 0. {
            return Self::endpoint(self.roff);
        }
        if t >= 1. {
            return Self::endpoint(self.ron);
        }
        if self.ron > self.roff {
            // Native P-switch logarithmic minimum-resistance limiter.
            return Self::endpoint(self.ron);
        }
        if matches!(t, 0.) || matches!(self.log_ratio, 0.) {
            return Self::endpoint(self.roff);
        }
        // Smoothstep in log(R). Evaluate the nearer endpoint's expression
        // to retain accuracy at high ratios and at both joins.
        let s = 1. - t;
        let log_r = if t <= 0.5 {
            self.log_ratio
                .mul_add(t * t * (3. - 2. * t), self.roff.ln())
        } else {
            (-self.log_ratio).mul_add(s * s * (3. - 2. * s), self.ron.ln())
        };
        let g = (-log_r).exp();
        Point {
            g,
            dg: (-self.log_ratio * 6. * t * s / self.delta) * g,
        }
    }

    fn exponential(self, t: Real) -> Point {
        if self.limited {
            if self.ron > self.roff {
                // A-switch applies the minimum then the maximum limiter.
                return Self::endpoint(self.roff);
            }
            if t < 0. {
                return Self::endpoint(self.roff);
            }
            if t > 1. {
                return Self::endpoint(self.ron);
            }
        }
        if matches!(self.log_ratio, 0.) {
            return Self::endpoint(self.ron);
        }
        let log_r = if t <= 0.5 {
            self.log_ratio.mul_add(t, self.roff.ln())
        } else {
            (-self.log_ratio).mul_add(1. - t, self.ron.ln())
        };
        if !self.limited && log_r <= 1e-9_f64.ln() {
            return Self::endpoint(1e-9);
        }
        let g = if matches!(t, 0.) {
            self.roff.recip()
        } else if matches!(t, 1.) {
            self.ron.recip()
        } else {
            (-log_r).exp()
        };
        Point {
            g,
            dg: (-self.log_ratio / self.delta) * g,
        }
    }

    fn linear(self, t: Real) -> Point {
        let bounded = self.law == Law::LinearBounded;
        if self.limited {
            if !bounded && self.ron > self.roff {
                return Self::endpoint(self.roff);
            }
            if t < 0. {
                return Self::endpoint(if bounded {
                    self.roff.max(1e-9)
                } else {
                    self.roff
                });
            }
            if t > 1. {
                return Self::endpoint(self.ron);
            }
        }
        let difference = self.ron - self.roff;
        if matches!(difference, 0.) {
            return Self::endpoint(self.ron);
        }
        let r = if t <= 0.5 {
            difference.mul_add(t, self.roff)
        } else {
            (-difference).mul_add(1. - t, self.ron)
        };
        if (bounded || !self.limited) && r <= 1e-9 {
            return Self::endpoint(1e-9);
        }
        let g = r.recip();
        Point {
            g,
            dg: -(difference / self.delta) * g * g,
        }
    }
}
