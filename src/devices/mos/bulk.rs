//! Shared classic MOS bulk-junction temperature/geometry derivation.
//! The levels differ in C multiplication order and limiter scaling, not
//! in the underlying temperature or capacitance laws.

use super::{Common, Geometry, Junction, Multiplicity, ReverseLaw};
use crate::devices::nonlinear::thermal::SiliconTemperature;
use crate::primitives::Real;

/// Select the actual C arithmetic variant. Level 9 uses the MOS3
/// temperature and junction routines.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::devices) enum BulkLaw {
    Mos1,
    Mos2,
    Mos3,
    Mos6,
}

pub(in crate::devices) struct BulkJunctions {
    pub(in crate::devices) drain: Junction,
    pub(in crate::devices) source: Junction,
    pub(in crate::devices) factors: [Real; 4],
}

impl BulkLaw {
    fn bottom_cap(self, is_drain: bool, capacitance: Real, area: Real, m: Real) -> Real {
        // mos1temp.c has different product order for drain and source.
        if self == Self::Mos1 && is_drain {
            capacitance * m * area
        } else {
            capacitance * area * m
        }
    }

    pub(in crate::devices) const fn multiplicity(self) -> Multiplicity {
        match self {
            Self::Mos6 => Multiplicity::Stamp,
            _ => Multiplicity::Embedded,
        }
    }

    const fn reverse(self) -> ReverseLaw {
        match self {
            Self::Mos3 => ReverseLaw::Cubic,
            _ => ReverseLaw::Constant,
        }
    }

    pub(in crate::devices) fn junctions(
        self,
        model: &Common,
        geometry: &Geometry,
        nominal: &SiliconTemperature,
        actual: &SiliconTemperature,
    ) -> BulkJunctions {
        let scale = (-actual.gap / actual.vt + nominal.gap / nominal.vt).exp();
        let saturation = model.is * scale;
        let density = model.js * scale;
        let correction = actual.potential.junction(&nominal.potential, model.pb);
        let bottom = correction.factors(model.mj);
        let side = correction.factors(model.mjsw);
        let m = self.multiplicity().storage(geometry.m);
        let area_zero = geometry.ad == 0. || geometry.as_ == 0.;
        let use_density = density != 0. && !area_zero;
        // mos3temp.c tests nominal JS; mos3load.c tests the scaled density.
        let critical_density = if self == Self::Mos3 {
            model.js != 0. && !area_zero
        } else {
            use_density
        };
        let scaled = |c: Real, factor: [Real; 2]| c * factor[0] * factor[1];
        let junction = |is_drain: bool, given: Option<Real>, area: Real, perimeter: Real| {
            let current = if use_density {
                density * m * area
            } else {
                m * saturation
            };
            let root2 = std::f64::consts::SQRT_2 * m;
            let divisor = if critical_density {
                root2 * density * area
            } else {
                root2 * saturation
            };
            let vcrit = actual.vt * (actual.vt / divisor).ln();
            let bottom_cap = match (given, model.cj) {
                (Some(c), _) => scaled(c, bottom) * m,
                (None, Some(cj)) => self.bottom_cap(is_drain, scaled(cj, bottom), area, m),
                (None, None) => 0.,
            };
            let sidewall = model.cjsw.map_or(0., |c| scaled(c, side) * perimeter * m);
            Junction::new(
                current,
                vcrit,
                bottom_cap,
                sidewall,
                model.mj,
                model.mjsw,
                correction.potential,
                model.fc,
                self.reverse(),
            )
        };
        BulkJunctions {
            drain: junction(true, model.cbd, geometry.ad, geometry.pd),
            source: junction(false, model.cbs, geometry.as_, geometry.ps),
            factors: [bottom[0], bottom[1], side[0], side[1]],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mos1_drain_and_source_keep_distinct_products() {
        let mut differs = false;
        for c in [1.7e-4_f64, 2.3e-3, 3.77e-5] {
            for area in [3.1e-12, 7.2e-12, 9.3e-13] {
                for m in [1.7, 2.3, 3.9] {
                    let drain = (c * m * area).to_bits();
                    let source = (c * area * m).to_bits();
                    differs |= drain != source;
                    assert_eq!(BulkLaw::Mos1.bottom_cap(true, c, area, m).to_bits(), drain);
                    assert_eq!(
                        BulkLaw::Mos1.bottom_cap(false, c, area, m).to_bits(),
                        source
                    );
                    for law in [BulkLaw::Mos2, BulkLaw::Mos3, BulkLaw::Mos6] {
                        assert_eq!(law.bottom_cap(true, c, area, m).to_bits(), source);
                        assert_eq!(law.bottom_cap(false, c, area, m).to_bits(), source);
                    }
                }
            }
        }
        assert!(differs);
    }
}
