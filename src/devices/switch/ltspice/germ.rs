//! Shared scalar and time-series arithmetic for the physical switch law.
//! Scalar evaluation is allocation-free; the time series retains all terms.

use crate::devices::physical_dae::LocalGuard;
use crate::maths::taylor::Jet;
use crate::primitives::{Real, SpiceResult};
use std::cmp::Ordering;

pub(super) trait Germ: Clone {
    fn value(&self) -> Real;
    fn direction(&self) -> Ordering;
    fn constant(&self, value: Real) -> Self;
    fn with_value(self, value: Real) -> Self;
    fn offset(self, value: Real) -> Self;
    fn scale(self, factor: Real) -> Self;
    fn add(self, rhs: &Self) -> Self;
    fn multiply(self, rhs: &Self) -> Self;
    fn divide(self, rhs: &Self) -> SpiceResult<Self>;
    fn exp(self) -> Self;
    fn compose_derivative(self, value: Real, slope: &Self) -> Self;
    fn guard(&self, slot: usize, boundaries: &[Real], guards: &mut Option<Vec<LocalGuard>>);

    fn compare(&self, boundary: Real) -> Ordering {
        match self.value().partial_cmp(&boundary) {
            Some(Ordering::Equal) => self.direction(),
            Some(order) => order,
            None => Ordering::Equal,
        }
    }

    fn abs(self) -> Self {
        if self.compare(0.) == Ordering::Less {
            self.scale(-1.)
        } else {
            self
        }
    }

    fn mul_add(self, rhs: &Self, add: &Self) -> Self {
        let value = self.value().mul_add(rhs.value(), add.value());
        self.multiply(rhs).add(add).with_value(value)
    }
}

impl Germ for Real {
    fn value(&self) -> Real {
        *self
    }
    fn direction(&self) -> Ordering {
        Ordering::Equal
    }
    fn constant(&self, value: Real) -> Self {
        value
    }
    fn with_value(self, value: Real) -> Self {
        value
    }
    fn offset(self, value: Real) -> Self {
        self + value
    }
    fn scale(self, factor: Real) -> Self {
        self * factor
    }
    fn add(self, rhs: &Self) -> Self {
        self + rhs
    }
    fn multiply(self, rhs: &Self) -> Self {
        self * rhs
    }
    fn divide(self, rhs: &Self) -> SpiceResult<Self> {
        Ok(self / rhs)
    }
    fn exp(self) -> Self {
        self.exp()
    }
    fn compose_derivative(self, value: Real, _slope: &Self) -> Self {
        value
    }
    fn guard(&self, _slot: usize, _boundaries: &[Real], _guards: &mut Option<Vec<LocalGuard>>) {}
}

impl Germ for Jet {
    fn value(&self) -> Real {
        self.value()
    }
    fn direction(&self) -> Ordering {
        self.direction()
    }
    fn constant(&self, value: Real) -> Self {
        Self::constant(self.order(), value)
    }
    fn with_value(self, value: Real) -> Self {
        self.with_value(value)
    }
    fn offset(self, value: Real) -> Self {
        self.offset(value)
    }
    fn scale(self, factor: Real) -> Self {
        self.scale(factor)
    }
    fn add(self, rhs: &Self) -> Self {
        self.add(rhs)
    }
    fn multiply(self, rhs: &Self) -> Self {
        Self::multiply(&self, rhs)
    }
    fn divide(self, rhs: &Self) -> SpiceResult<Self> {
        Self::divide(&self, rhs)
    }
    fn exp(self) -> Self {
        Self::exp(&self)
    }
    fn compose_derivative(self, value: Real, slope: &Self) -> Self {
        Self::compose_derivative(&self, value, slope)
    }
    fn guard(&self, slot: usize, boundaries: &[Real], guards: &mut Option<Vec<LocalGuard>>) {
        if let Some(guards) = guards
            && self.check().is_ok()
        {
            guards.push(LocalGuard::interval(slot, self.clone(), boundaries));
        }
    }
}

/// The stable derivative retains the exponentially small slope in saturation.
pub(super) fn sech_squared<T: Germ>(argument: &T) -> SpiceResult<T> {
    if argument.value().is_infinite() {
        return Ok(argument.constant(0.));
    }
    if (-2. * argument.value().abs()).exp() == 0. {
        return Ok(argument.constant(0.));
    }
    let tail = argument.clone().abs().scale(-2.).exp();
    let denom = tail.clone().offset(1.);
    tail.scale(4.).divide(&denom.clone().multiply(&denom))
}

pub(super) fn tanh<T: Germ>(argument: T) -> SpiceResult<T> {
    let value = argument.value().tanh();
    if argument.value().is_infinite() {
        return Ok(argument.constant(value));
    }
    let slope = sech_squared(&argument)?;
    Ok(argument.compose_derivative(value, &slope))
}

/// Angle and its derivative in the physical, unscaled coordinate.
pub(super) fn angle<T: Germ>(delta: T, width: Real) -> SpiceResult<(T, T)> {
    let value = delta.value().atan2(width);
    if delta.value().is_infinite() {
        return Ok((delta.constant(value), delta.constant(0.)));
    }
    let scale = delta.value().abs().max(width);
    let normalized_width = width / scale;
    let normalized_delta = delta.clone().divide(&delta.constant(scale))?;
    let denom = normalized_delta.clone().mul_add(
        &normalized_delta,
        &delta.constant(normalized_width * normalized_width),
    );
    let slope = delta
        .constant(normalized_width)
        .divide(&denom)?
        .divide(&delta.constant(scale))?;
    Ok((delta.compose_derivative(value, &slope), slope))
}
