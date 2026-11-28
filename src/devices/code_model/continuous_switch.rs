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

/// Local normalized distance from the nearer endpoint. Branch decisions
/// use the original control coordinate, so rounding a long interval to
/// t=0 or t=1 cannot erase an interior germ.
struct Position<T> {
    distance: T,
    near_on: bool,
    off_side: Ordering,
    on_side: Ordering,
    anchor: Option<Real>,
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

    fn position<T: Constitutive>(self, control: T) -> Position<T> {
        let oriented = |order: Ordering| {
            if self.delta.is_sign_negative() {
                order.reverse()
            } else {
                order
            }
        };
        let off_side = oriented(control.compare(self.off));
        let on_side = oriented(control.compare(self.on));
        let midpoint = Real::midpoint(self.off, self.on);
        let near_on = oriented(
            control
                .value()
                .partial_cmp(&midpoint)
                .unwrap_or(Ordering::Equal),
        ) == Ordering::Greater;
        let anchor = if control.value().partial_cmp(&self.on) == Some(Ordering::Equal) {
            Some(self.ron.recip())
        } else if control.value().partial_cmp(&self.off) == Some(Ordering::Equal) {
            Some(self.roff.recip())
        } else {
            None
        };
        let distance = if near_on {
            control.scale(-1.).offset(self.on)
        } else {
            control.offset(-self.off)
        }
        .divide_constant(self.delta);
        Position {
            distance,
            near_on,
            off_side,
            on_side,
            anchor,
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
        if self.law == Law::CubicLog
            || (self.limited && (self.law == Law::LinearBounded || self.ron <= self.roff))
        {
            guard(0, &control, self.off);
            guard(1, &control, self.on);
        }
        let position = self.position(control);
        let point = match self.law {
            Law::CubicLog => self.cubic(position),
            Law::LinearBounded | Law::Linear => self.linear(position, &mut guard)?,
            Law::Exponential => self.exponential(position, &mut guard),
        };
        point.check()?;
        Ok(point)
    }

    fn endpoint<T: Constitutive>(t: &T, resistance: Real) -> T {
        t.constant_like(resistance.recip())
    }

    fn cubic<T: Constitutive>(self, position: Position<T>) -> T {
        let Position {
            distance,
            near_on,
            off_side,
            on_side,
            ..
        } = position;
        if off_side == Ordering::Less {
            return Self::endpoint(&distance, self.roff);
        }
        if on_side != Ordering::Less || self.ron > self.roff {
            return Self::endpoint(&distance, self.ron);
        }
        if off_side == Ordering::Equal || self.log_ratio == 0. {
            return Self::endpoint(&distance, self.roff);
        }
        // Smoothstep in log(R) about the nearer endpoint, without
        // computing 1-t from an already rounded normalized coordinate.
        let (span, origin) = if near_on {
            (-self.log_ratio, self.ron.ln())
        } else {
            (self.log_ratio, self.roff.ln())
        };
        let smooth = distance
            .clone()
            .square()
            .multiply(&distance.affine(-2., 3.));
        smooth.affine(-span, -origin).exp()
    }

    fn bounded<T: Constitutive>(self, position: &Position<T>, bounded: bool) -> Option<T> {
        if !self.limited {
            return None;
        }
        let distance = &position.distance;
        if !bounded && self.ron > self.roff {
            return Some(Self::endpoint(distance, self.roff));
        }
        if position.off_side == Ordering::Less {
            Some(Self::endpoint(
                distance,
                if bounded {
                    self.roff.max(1e-9)
                } else {
                    self.roff
                },
            ))
        } else if position.on_side == Ordering::Greater {
            Some(Self::endpoint(distance, self.ron))
        } else {
            None
        }
    }

    fn exponential<T: Constitutive>(
        self,
        position: Position<T>,
        guard: &mut impl FnMut(usize, &T, Real),
    ) -> T {
        if let Some(point) = self.bounded(&position, false) {
            return point;
        }
        if self.log_ratio == 0. {
            return Self::endpoint(&position.distance, self.ron);
        }
        let log_r = if position.near_on {
            position.distance.affine(-self.log_ratio, self.ron.ln())
        } else {
            position.distance.affine(self.log_ratio, self.roff.ln())
        };
        if !self.limited {
            let cutoff = 1e-9_f64.ln();
            guard(2, &log_r, cutoff);
            if log_r.compare(cutoff) != Ordering::Greater {
                return Self::endpoint(&log_r, 1e-9);
            }
        }
        let g = log_r.scale(-1.).exp();
        if let Some(value) = position.anchor {
            g.with_value(value)
        } else {
            g
        }
    }

    fn linear<T: Constitutive>(
        self,
        position: Position<T>,
        guard: &mut impl FnMut(usize, &T, Real),
    ) -> SpiceResult<T> {
        let bounded = self.law == Law::LinearBounded;
        if let Some(point) = self.bounded(&position, bounded) {
            return Ok(point);
        }
        let difference = self.ron - self.roff;
        if difference == 0. {
            return Ok(Self::endpoint(&position.distance, self.ron));
        }
        let r = if position.near_on {
            position.distance.affine(-difference, self.ron)
        } else {
            position.distance.affine(difference, self.roff)
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::maths::taylor::Jet;

    #[test]
    fn wide_control_interval_keeps_the_interior_germ_close_to_either_endpoint() -> SpiceResult<()> {
        for sign in [1., -1.] {
            let material = Material {
                law: Law::CubicLog,
                on: sign,
                off: -sign * 1e16,
                delta: sign * 1e16,
                ron: 10.,
                roff: 1000.,
                log_ratio: -100_f64.ln(),
                limited: true,
                control_g: 1e-12,
            };
            // (control-off)/delta rounds to exactly one. The physical
            // control is still half a volt inside the interval.
            let control = sign * 0.5;
            let point = material.point(control)?;
            let s = 0.5e-16;
            let g = (material.log_ratio * s * s * (3. - 2. * s) - material.ron.ln()).exp();
            let dg = -material.log_ratio * 6. * s * (1. - s) / material.delta * g;
            assert!((point.dg / dg - 1.).abs() < 1e-13);
            let mut guards = Vec::new();
            let inside = material.taylor(
                Differential::variable(0, Jet::affine(4, control, sign * 0.1))?,
                &mut guards,
            )?;
            assert!((inside.gradient[0].as_ref().unwrap().value() / dg - 1.).abs() < 1e-13);
            let exterior = material.taylor(
                Differential::variable(0, Jet::affine(4, sign * 1.5, sign * 0.1))?,
                &mut Vec::new(),
            )?;
            assert!(exterior.gradient[0].is_none());
        }
        Ok(())
    }
}
