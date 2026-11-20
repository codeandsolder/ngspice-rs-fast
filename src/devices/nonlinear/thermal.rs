//! Immutable operating-point derivations shared by classic devices.
//! These are independent of the Newton iterate and accepted/trial state.

use std::cell::{Ref, RefCell};

use crate::devices::ModelContext;
use crate::primitives::{Real, SpiceResult};

/// C `CONSTboltz` (J/K).
pub(in crate::devices) const BOLTZMANN: Real = 1.380_648_52e-23;
/// C `CHARGE` (C).
pub(in crate::devices) const CHARGE: Real = 1.602_176_620_8e-19;
/// C `CONSTKoverQ`.
pub(in crate::devices) const K_OVER_Q: Real = BOLTZMANN / CHARGE;
/// C `CONSTCtoK`.
pub(in crate::devices) const CELSIUS_TO_KELVIN: Real = 273.15;
/// C `REFTEMP` (27 degrees Celsius).
pub(in crate::devices) const REFTEMP: Real = 27. + CELSIUS_TO_KELVIN;

/// Silicon band gap at a temperature in Kelvin, as in C's classic models.
pub(in crate::devices) fn band_gap(kelvin: Real) -> Real {
    1.16 - (7.02e-4 * kelvin * kelvin) / (kelvin + 1108.)
}

/// Built-in potential shift from the C temperature routines.
pub(in crate::devices) fn potential_shift(vt: Real, relative: Real, arg: Real) -> Real {
    let logarithmic = 1.5 * relative.ln();
    let electrical = CHARGE * arg;
    -2. * vt * (logarithmic + electrical)
}

/// One temperature's shift and scale of semiconductor junction potential.
#[derive(Debug, Clone, Copy)]
pub(in crate::devices) struct PotentialTemperature {
    kelvin: Real,
    relative: Real,
    shift: Real,
}

impl PotentialTemperature {
    pub(in crate::devices) fn new(kelvin: Real, shift: Real) -> Self {
        Self {
            kelvin,
            relative: kelvin / REFTEMP,
            shift,
        }
    }

    fn reference_potential(&self, potential: Real) -> Real {
        (potential - self.shift) / self.relative
    }

    /// Transform a nominal-temperature potential to this temperature.
    pub(in crate::devices) fn correct(&self, nominal: &Self, potential: Real) -> Real {
        let scaled = self.relative * nominal.reference_potential(potential);
        scaled + self.shift
    }

    /// The nominal and actual grading-coefficient offsets of the C MOS,
    /// JFET and BJT temperature laws. Compute the shared base once; keep
    /// each model's subsequent capacitance multiplication/division order.
    pub(in crate::devices) fn junction(
        &self,
        nominal: &Self,
        potential: Real,
    ) -> JunctionCorrection {
        let pbo = nominal.reference_potential(potential);
        let gmaold = (potential - pbo) / pbo;
        let scaled = self.relative * pbo;
        let actual = scaled + self.shift;
        let gmanew = (actual - pbo) / pbo;
        let nominal_delta = 4e-4 * (nominal.kelvin - REFTEMP);
        let actual_delta = 4e-4 * (self.kelvin - REFTEMP);
        JunctionCorrection {
            potential: actual,
            nominal_offset: nominal_delta - gmaold,
            actual_offset: actual_delta - gmanew,
        }
    }
}

/// MOS/JFET silicon temperature values with the reference `kt + kt`
/// denominator (BJT/diode keep their own denominator grouping).
#[derive(Debug, Clone, Copy)]
pub(in crate::devices) struct SiliconTemperature {
    pub(in crate::devices) vt: Real,
    pub(in crate::devices) gap: Real,
    pub(in crate::devices) potential: PotentialTemperature,
}

impl SiliconTemperature {
    pub(in crate::devices) fn new(kelvin: Real) -> Self {
        let vt = kelvin * K_OVER_Q;
        let kt = BOLTZMANN * kelvin;
        let gap = band_gap(kelvin);
        let arg = -gap / (kt + kt) + 1.115_087_7 / (BOLTZMANN * (REFTEMP + REFTEMP));
        let shift = potential_shift(vt, kelvin / REFTEMP, arg);
        Self {
            vt,
            gap,
            potential: PotentialTemperature::new(kelvin, shift),
        }
    }
}

/// Temperature-adjusted junction potential and grading corrections.
#[derive(Debug, Clone, Copy)]
pub(in crate::devices) struct JunctionCorrection {
    pub(in crate::devices) potential: Real,
    nominal_offset: Real,
    actual_offset: Real,
}

impl JunctionCorrection {
    pub(in crate::devices) fn nominal_denominator(&self, grading: Real) -> Real {
        let correction = grading * self.nominal_offset;
        1. + correction
    }

    pub(in crate::devices) fn actual_factor(&self, grading: Real) -> Real {
        let correction = grading * self.actual_offset;
        1. + correction
    }

    /// The two separate MOS/JFET factors, not their rounded quotient:
    /// the C temperature routines multiply the capacitance by each in turn.
    pub(in crate::devices) fn factors(&self, grading: Real) -> [Real; 2] {
        [
            1. / self.nominal_denominator(grading),
            self.actual_factor(grading),
        ]
    }
}

/// Exact circuit and nominal temperature keys. Bitwise equality deliberately
/// distinguishes signed zero and never matches a different NaN payload.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::devices) struct TemperatureKey([u64; 2]);

impl TemperatureKey {
    pub(in crate::devices) const fn new(temperature: Real, nominal: Real) -> Self {
        Self([temperature.to_bits(), nominal.to_bits()])
    }

    pub(in crate::devices) const fn context(context: &ModelContext) -> Self {
        Self::new(context.temperature, context.nominal_temperature)
    }
}

/// One successful derivation for an immutable device's most recent context.
/// Replacements for parameter sweeps and sensitivity get fresh caches.
/// A failed derivation must neither be cached nor evict a valid entry.
#[derive(Debug)]
pub(in crate::devices) struct DerivedCache<K, T>(RefCell<Option<(K, T)>>);

pub(in crate::devices) type TemperatureCache<T> = DerivedCache<TemperatureKey, T>;

impl<K: Eq, T> DerivedCache<K, T> {
    pub(in crate::devices) const fn new() -> Self {
        Self(RefCell::new(None))
    }

    pub(in crate::devices) const fn from_value(key: K, value: T) -> Self {
        Self(RefCell::new(Some((key, value))))
    }
}

impl<K: Copy + Eq, T> DerivedCache<K, T> {
    /// Borrow on hot paths without copying a large operating descriptor.
    pub(in crate::devices) fn get(&self, key: K) -> Option<Ref<'_, T>> {
        Ref::filter_map(self.0.borrow(), |entry| {
            entry
                .as_ref()
                .and_then(|(stored, value)| (*stored == key).then_some(value))
        })
        .ok()
    }
}

impl<K: Copy + Eq, T: Copy> DerivedCache<K, T> {
    pub(in crate::devices) fn get_or_derive(
        &self,
        key: K,
        derive: impl FnOnce() -> SpiceResult<T>,
    ) -> SpiceResult<T> {
        if let Some(value) = self.get(key) {
            return Ok(*value);
        }
        let value = derive()?;
        // A caller holding a read of the previous key may still derive
        // another value; keep its borrowed entry intact until it is released.
        if let Ok(mut entry) = self.0.try_borrow_mut() {
            *entry = Some((key, value));
        }
        Ok(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::primitives::SpiceError;

    #[test]
    fn silicon_junction_correction_preserves_c_capacitance_factors() {
        let reference_shift = |t: Real| {
            let vt = t * K_OVER_Q;
            let kt = BOLTZMANN * t;
            let gap = 1.16 - (7.02e-4 * t * t) / (t + 1108.);
            let arg = -gap / (kt + kt) + 1.115_087_7 / (BOLTZMANN * (REFTEMP + REFTEMP));
            -2. * vt * (1.5 * (t / REFTEMP).ln() + CHARGE * arg)
        };
        let mut rounded_differently = false;
        for tnom in [255.15, 300.15, 335.15] {
            let nominal = SiliconTemperature::new(tnom);
            for temp in [233.15, 300.15, 373.15, 425.15] {
                let actual = SiliconTemperature::new(temp);
                for pot in [0.6, 0.77, 1.2] {
                    let pbo = (pot - reference_shift(tnom)) / (tnom / REFTEMP);
                    let potential = (temp / REFTEMP) * pbo + reference_shift(temp);
                    let gmaold = (pot - pbo) / pbo;
                    let gmanew = (potential - pbo) / pbo;
                    let correction = actual.potential.junction(&nominal.potential, pot);
                    assert_eq!(correction.potential.to_bits(), potential.to_bits());
                    assert_eq!(
                        actual.potential.correct(&nominal.potential, pot).to_bits(),
                        potential.to_bits()
                    );
                    for m in [0., 0.33, 0.5, 0.999] {
                        let nominal_denom = 1. + m * (4e-4 * (tnom - REFTEMP) - gmaold);
                        let actual_factor = 1. + m * (4e-4 * (temp - REFTEMP) - gmanew);
                        let factors = correction.factors(m);
                        assert_eq!(factors[0].to_bits(), (1. / nominal_denom).to_bits());
                        assert_eq!(factors[1].to_bits(), actual_factor.to_bits());
                        let c = 2.37e-12_f64;
                        let reference = (c * (1. / nominal_denom)) * actual_factor;
                        assert_eq!((c * factors[0] * factors[1]).to_bits(), reference.to_bits());
                        rounded_differently |=
                            reference.to_bits() != (c * (actual_factor / nominal_denom)).to_bits();
                    }
                }
            }
        }
        assert!(
            rounded_differently,
            "must exercise the previous quotient rounding difference"
        );
    }

    #[test]
    fn a_borrowed_derivation_survives_a_different_key() {
        let cache = DerivedCache::from_value(1, 10);
        let borrowed = cache.get(1).unwrap();
        assert_eq!(cache.get_or_derive(2, || Ok(20)).unwrap(), 20);
        assert_eq!(*borrowed, 10);
        drop(borrowed);
        assert_eq!(cache.get_or_derive(2, || Ok(21)).unwrap(), 21);
        assert!(cache.get(1).is_none());
    }

    #[test]
    fn temperature_cache_reuses_latest_success_and_never_caches_failure() {
        let cache = TemperatureCache::new();
        let cold = TemperatureKey::new(0., 27.);
        let minus_zero = TemperatureKey::new(-0., 27.);
        let hot = TemperatureKey::new(100., 27.);
        let nominal = TemperatureKey::new(100., 30.);
        assert_eq!(cache.get_or_derive(cold, || Ok(1)).unwrap(), 1);
        assert_eq!(
            cache.get_or_derive(cold, || panic!("cache miss")).unwrap(),
            1
        );
        assert_eq!(cache.get_or_derive(minus_zero, || Ok(2)).unwrap(), 2);
        assert!(
            cache
                .get_or_derive(hot, || Err(SpiceError::circuit("invalid")))
                .is_err()
        );
        assert_eq!(*cache.get(minus_zero).unwrap(), 2);
        assert!(cache.get(hot).is_none());
        assert_eq!(cache.get_or_derive(hot, || Ok(3)).unwrap(), 3);
        assert!(cache.get(minus_zero).is_none());
        assert_eq!(cache.get_or_derive(nominal, || Ok(4)).unwrap(), 4);
        assert_eq!(cache.get_or_derive(hot, || Ok(5)).unwrap(), 5);
    }
}
