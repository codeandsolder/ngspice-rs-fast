//! Realization of static statistical functions. A sample belongs to the
//! expression occurrence and expanded lexical instance, never a Newton
//! iteration, candidate evaluation order, or a mutable global generator.
use std::cell::Cell;

use crate::netlist::ast::OptionCard;
use crate::primitives::numeric::{scale_power, split_power};
use crate::primitives::random::{gaussian, mix, uniform};
use crate::primitives::{Real, SourceLoc, SpiceError, SpiceResult, parse_spice_number};

pub(super) const DEFAULT_SEED: u64 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct Realization {
    pub seed: u64,
    pub scope: u64,
}

impl Realization {
    pub(super) const fn root(seed: u64) -> Self {
        Self { seed, scope: 0 }
    }

    pub(super) const fn child(self, index: u64) -> Self {
        Self {
            seed: self.seed,
            scope: mix(self.scope ^ mix(index)),
        }
    }

    pub(super) const fn sampler(self, location: &SourceLoc) -> Sampler<'_> {
        Sampler {
            realization: self,
            location,
            key: Cell::new(None),
            next: Cell::new(0),
        }
    }
}

/// One root expression evaluation. Traversal order within the expression
/// distinguishes expanded calls at identical source spans. Each candidate
/// starts the same site stream, so nominal/variation changes preserve the
/// normalized random draw even if unrelated definitions are not evaluated.
#[expect(
    clippy::redundant_pub_crate,
    reason = "crate-restricted visibility is required by the unreachable_pub lint"
)]
pub(crate) struct Sampler<'a> {
    realization: Realization,
    location: &'a SourceLoc,
    key: Cell<Option<u64>>,
    next: Cell<u64>,
}

impl Sampler<'_> {
    fn key(&self) -> u64 {
        if let Some(key) = self.key.get() {
            return key;
        }
        // Hash the source site only when a statistical function actually
        // occurs. Ordinary numeric expressions pay no filename scan.
        let mut site = 0xcbf2_9ce4_8422_2325_u64;
        for &byte in self.location.path().as_os_str().as_encoded_bytes() {
            site = (site ^ u64::from(byte)).wrapping_mul(0x100_0000_01b3);
        }
        let key = mix(mix(
            mix(mix(self.realization.seed ^ site) ^ self.realization.scope)
                ^ u64::from(self.location.line),
        ) ^ self.location.column as u64);
        self.key.set(Some(key));
        key
    }

    pub(crate) fn sample(
        &self,
        name: &str,
        args: &[Real],
        location: &SourceLoc,
    ) -> SpiceResult<Real> {
        let index = self.next.get();
        self.next.set(index.checked_add(1).ok_or_else(|| {
            SpiceError::parse(location.clone(), "statistical call counter overflow")
        })?);
        let key = self.key();
        let unit = || 2_f64.mul_add(uniform(key, 0x5354_4154, index, 0), -1.);
        let value = match (name, args) {
            ("agauss" | "gauss", &[nominal, variation, sigma]) => {
                if variation <= 0. || sigma <= 0. {
                    nominal
                } else {
                    let draw = gaussian(key, 0x4741_5553, index);
                    if name == "gauss" {
                        affine_draw(nominal, &[nominal, variation, draw], sigma)
                    } else {
                        affine_draw(nominal, &[variation, draw], sigma)
                    }
                }
            }
            ("unif", &[nominal, variation]) => {
                affine_draw(nominal, &[nominal, variation, unit()], 1.)
            }
            ("aunif", &[nominal, variation]) => variation.mul_add(unit(), nominal),
            ("limit", &[nominal, variation]) => {
                nominal + if unit() > 0. { variation } else { -variation }
            }
            _ => {
                return Err(SpiceError::parse(
                    location.clone(),
                    format!(
                        "{name} requires {} static arguments",
                        if matches!(name, "gauss" | "agauss") {
                            3
                        } else {
                            2
                        }
                    ),
                ));
            }
        };
        if value.is_finite() {
            Ok(value)
        } else {
            Err(SpiceError::parse(
                location.clone(),
                format!("nonfinite statistical realization of {name}"),
            ))
        }
    }
}

/// Evaluate nominal + product(factors)/divisor without losing a finite
/// result to an intermediate overflow or underflow. The ordinary-scale
/// path retains a single fused final multiply/add. Extreme scales use
/// the same binary exponent kernels as complex physical arithmetic.
fn affine_draw(nominal: Real, factors: &[Real], divisor: Real) -> Real {
    if factors.contains(&0.) {
        return nominal;
    }
    let Some((draw, factors)) = factors.split_last() else {
        return nominal;
    };
    let product = factors.iter().product::<Real>();
    let scale = product / divisor;
    let result = scale.mul_add(*draw, nominal);
    if result.is_finite() && product.is_normal() && scale.is_normal() {
        return result;
    }
    let (mut mantissa, mut exponent) = split_power(divisor);
    mantissa = mantissa.recip();
    exponent = -exponent;
    for &factor in factors.iter().chain(std::iter::once(draw)) {
        let (m, e) = split_power(factor);
        mantissa *= m;
        exponent += e;
    }
    let (delta, normalization) = split_power(mantissa);
    exponent += normalization;
    if nominal == 0. {
        return scale_power(delta, exponent);
    }
    let (nominal, nominal_exponent) = split_power(nominal);
    let common = exponent.max(nominal_exponent);
    scale_power(
        scale_power(delta, exponent - common) + scale_power(nominal, nominal_exponent - common),
        common,
    )
}

/// The seed is a deck-loading input: it cannot depend on the parameters
/// whose realization it controls. Both spellings use the same last-wins
/// policy. Entropy-based seeds are deliberately not implicit.
#[expect(
    clippy::redundant_pub_crate,
    reason = "crate-restricted visibility is required by the unreachable_pub lint"
)]
pub(crate) fn seed(cards: &[OptionCard]) -> SpiceResult<u64> {
    let mut seed = DEFAULT_SEED;
    for setting in cards.iter().flat_map(|card| &card.settings) {
        if matches!(setting.name.as_str(), "seed" | "rndseed") {
            seed = parse_seed(setting)?;
        }
    }
    Ok(seed)
}

#[expect(
    clippy::redundant_pub_crate,
    reason = "crate-restricted visibility is required by the unreachable_pub lint"
)]
pub(crate) fn parse_seed(setting: &crate::netlist::ast::OptionSetting) -> SpiceResult<u64> {
    if setting.expression.is_some() {
        return Err(SpiceError::parse(
            setting.location.clone(),
            "statistical seed must be a literal before parameter evaluation",
        ));
    }
    let text = setting
        .value
        .as_ref()
        .map_or("", |value| value.text.as_str());
    let value = parse_spice_number(text)
        .and_then(crate::primitives::numeric::exact_nonnegative_integer::<u64>)
        .filter(|&seed| seed > 0)
        .ok_or_else(|| {
            SpiceError::parse(
                setting.location.clone(),
                "statistical seed must be a positive integer literal",
            )
        })?;
    Ok(value)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scaled_draws_preserve_representable_values_at_exponent_limits() {
        assert_eq!(affine_draw(1e308, &[1e308, 2., -0.25], 2.), 7.5e307);
        assert!((affine_draw(0., &[1e-300, 1e-300, 1e300], 1e-300) - 1.).abs() < 1e-14);
        // Cancellation can make the final value finite even if the
        // unscaled random perturbation exceeds f64::MAX.
        let result = affine_draw(1e308, &[1e308, 2., -1.], 1.);
        assert_eq!(result, -1e308);
    }

    #[test]
    fn independent_samples_match_distribution_moments() {
        let location = SourceLoc::new(std::path::PathBuf::from("statistical.cir"), 1, 1);
        let n = 50_000_u32;
        let sampler = Realization::root(87).sampler(&location);
        for (name, args, variance) in [
            ("agauss", vec![10., 3., 3.], 1_f64),
            ("gauss", vec![10., 0.3, 3.], 1.),
            ("aunif", vec![10., 3.], 3.),
            ("unif", vec![10., 0.3], 3.),
            ("limit", vec![10., 3.], 9.),
        ] {
            let mut sum = 0.;
            let mut square = 0.;
            let mut cross = 0.;
            let mut previous = 0.;
            for index in 0..n {
                let value = sampler.sample(name, &args, &location).unwrap() - 10.;
                sum += value;
                square += value * value;
                if index > 0 {
                    cross += value * previous;
                }
                previous = value;
            }
            let n = f64::from(n);
            let mean = sum / n;
            let variance_actual = (-mean).mul_add(mean, square / n);
            assert!(mean.abs() < variance.sqrt() * 0.025, "{name}: {mean}");
            assert!(
                (variance_actual / variance - 1.).abs() < 0.035,
                "{name}: {variance_actual}"
            );
            assert!(
                (cross / (n - 1.) / variance).abs() < 0.025,
                "{name}: {cross}"
            );
        }
    }
}
