//! Memoryless XSPICE switch resistance laws and their physical differential.
//! Unlike a threshold switch these have no trial or accepted discrete state.

use super::source::{SourceModel, numeric_scalar};
use crate::devices::physical_dae::LocalGuard;
use crate::maths::taylor::{
    constitutive::{Constitutive, Dual},
    differential::Differential,
};
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

    fn normalized<T: Constitutive>(self, control: T) -> T {
        // Preserve the exact scalar endpoint while retaining derivatives.
        let endpoint = match control.value().partial_cmp(&self.on) {
            Some(Ordering::Equal) => Some(1.),
            _ if control.value().partial_cmp(&self.off) == Some(Ordering::Equal) => Some(0.),
            _ => None,
        };
        let t = control.offset(-self.off).divide_constant(self.delta);
        if let Some(value) = endpoint {
            t.with_value(value)
        } else {
            t
        }
    }

    pub(super) fn point(self, control: Real) -> SpiceResult<Point> {
        let point = self.conductance(Dual::<1>::variable(0, control)?, |_, _, _| {})?;
        Ok(Point {
            g: point.value,
            dg: point.gradient[0],
        })
    }

    pub(super) fn taylor(
        self,
        control: Differential<2>,
        guards: &mut Vec<LocalGuard>,
    ) -> SpiceResult<Differential<2>> {
        self.conductance(control, |slot, argument, boundary| {
            guards.push(LocalGuard::sign(slot, 0, argument.value.clone(), boundary));
        })
    }

    fn conductance<T: Constitutive>(
        self,
        control: T,
        mut guard: impl FnMut(usize, &T, Real),
    ) -> SpiceResult<T> {
        control.check()?;
        let t = self.normalized(control);
        let point = match self.law {
            Law::CubicLog => self.cubic(t, &mut guard),
            Law::LinearBounded | Law::Linear => self.linear(t, &mut guard)?,
            Law::Exponential => self.exponential(t, &mut guard),
        };
        point.check()?;
        Ok(point)
    }

    fn endpoint<T: Constitutive>(t: &T, resistance: Real) -> T {
        t.constant_like(resistance.recip())
    }

    fn cubic<T: Constitutive>(self, t: T, guard: &mut impl FnMut(usize, &T, Real)) -> T {
        guard(0, &t, 0.);
        guard(1, &t, 1.);
        if t.compare(0.) == Ordering::Less {
            return Self::endpoint(&t, self.roff);
        }
        if t.compare(1.) != Ordering::Less {
            return Self::endpoint(&t, self.ron);
        }
        if self.ron > self.roff {
            return Self::endpoint(&t, self.ron);
        }
        if t.compare(0.) == Ordering::Equal || self.log_ratio == 0. {
            return Self::endpoint(&t, self.roff);
        }
        // Near each endpoint use its own logarithmic origin.
        let (fraction, span, origin) = if t.value() <= 0.5 {
            (t, self.log_ratio, self.roff.ln())
        } else {
            (t.scale(-1.).offset(1.), -self.log_ratio, self.ron.ln())
        };
        let smooth = fraction
            .clone()
            .square()
            .multiply(&fraction.affine(-2., 3.));
        smooth.affine(-span, -origin).exp()
    }

    fn bounded<T: Constitutive>(
        self,
        t: &T,
        bounded: bool,
        guard: &mut impl FnMut(usize, &T, Real),
    ) -> Option<T> {
        if !self.limited {
            return None;
        }
        if !bounded && self.ron > self.roff {
            return Some(Self::endpoint(t, self.roff));
        }
        guard(0, t, 0.);
        guard(1, t, 1.);
        if t.compare(0.) == Ordering::Less {
            Some(Self::endpoint(
                t,
                if bounded {
                    self.roff.max(1e-9)
                } else {
                    self.roff
                },
            ))
        } else if t.compare(1.) == Ordering::Greater {
            Some(Self::endpoint(t, self.ron))
        } else {
            None
        }
    }

    fn exponential<T: Constitutive>(self, t: T, guard: &mut impl FnMut(usize, &T, Real)) -> T {
        if let Some(point) = self.bounded(&t, false, guard) {
            return point;
        }
        if self.log_ratio == 0. {
            return Self::endpoint(&t, self.ron);
        }
        let endpoint = match t.value() {
            0. => Some(self.roff.recip()),
            1. => Some(self.ron.recip()),
            _ => None,
        };
        let log_r = if t.value() <= 0.5 {
            t.affine(self.log_ratio, self.roff.ln())
        } else {
            t.scale(-1.)
                .offset(1.)
                .affine(-self.log_ratio, self.ron.ln())
        };
        if !self.limited {
            let cutoff = 1e-9_f64.ln();
            guard(2, &log_r, cutoff);
            if log_r.compare(cutoff) != Ordering::Greater {
                return Self::endpoint(&log_r, 1e-9);
            }
        }
        let g = log_r.scale(-1.).exp();
        if let Some(value) = endpoint {
            g.with_value(value)
        } else {
            g
        }
    }

    fn linear<T: Constitutive>(
        self,
        t: T,
        guard: &mut impl FnMut(usize, &T, Real),
    ) -> SpiceResult<T> {
        let bounded = self.law == Law::LinearBounded;
        if let Some(point) = self.bounded(&t, bounded, guard) {
            return Ok(point);
        }
        let difference = self.ron - self.roff;
        if difference == 0. {
            return Ok(Self::endpoint(&t, self.ron));
        }
        let r = if t.value() <= 0.5 {
            t.affine(difference, self.roff)
        } else {
            t.scale(-1.).offset(1.).affine(-difference, self.ron)
        };
        if bounded || !self.limited {
            guard(2, &r, 1e-9);
            if r.compare(1e-9) != Ordering::Greater {
                return Ok(Self::endpoint(&r, 1e-9));
            }
        }
        r.reciprocal()
    }
}
