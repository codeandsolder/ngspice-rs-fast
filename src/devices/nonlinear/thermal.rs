//! Immutable operating-point derivations shared by classic devices.
//! These are independent of the Newton iterate and accepted/trial state.

use std::cell::{Ref, RefCell};

use crate::devices::ModelContext;
use crate::primitives::{Real, SpiceResult};

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
        self.0.replace(Some((key, value)));
        Ok(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::primitives::SpiceError;

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
