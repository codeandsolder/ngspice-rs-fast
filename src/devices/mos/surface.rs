//! Electrostatic process extraction for the classic MOS models.
//!
//! Preserve the reference C evaluation order, in particular the density
//! conversion and the silicon-permittivity product in the GAMMA derivation.

use super::{CHARGE, EPSILON_0, K_OVER_Q, aliased, band_gap};
use crate::devices::nonlinear::thermal::SiliconTemperature;
use crate::devices::schema::ScalarValues;
use crate::primitives::{Real, SpiceError, SpiceResult};

const INTRINSIC_DENSITY: Real = 1.45e16;
const EPSILON_SI: Real = 11.7 * EPSILON_0;

/// Raw optional setters, preserving the separate C Given flags.
#[derive(Debug, Clone, Copy, Default)]
pub(in crate::devices) struct SurfaceCard {
    pub(in crate::devices) vto: Option<Real>,
    pub(in crate::devices) gamma: Option<Real>,
    pub(in crate::devices) phi: Option<Real>,
    pub(in crate::devices) nsub: Option<Real>,
    pub(in crate::devices) tpg: Option<Real>,
    pub(in crate::devices) nss: Option<Real>,
}

/// C arithmetic variants: MOS1/2/6 use a fixed intrinsic density; MOS3/9
/// use a temperature-adjusted one.
#[derive(Debug, Clone, Copy)]
pub(in crate::devices) enum SurfaceLaw {
    Classic,
    Mos3,
}

/// Resolved nominal electrostatic parameters, separate from the card.
#[derive(Debug, Clone, Copy)]
pub(in crate::devices) struct Surface {
    pub(in crate::devices) phi: Real,
    pub(in crate::devices) gamma: Real,
    pub(in crate::devices) vto: Real,
}

impl SurfaceCard {
    pub(in crate::devices) fn from_values(values: &ScalarValues) -> Self {
        let get = |name: &str| values.get(name).map(|v| v.value);
        Self {
            vto: aliased(values, ["vto", "vt0"]),
            gamma: get("gamma"),
            phi: get("phi"),
            nsub: get("nsub"),
            tpg: get("tpg"),
            nss: get("nss"),
        }
    }

    /// MOS1/6 skip process extraction when TOX is missing or zero.
    pub(in crate::devices) fn defaults(self) -> Surface {
        Surface {
            phi: self.phi.unwrap_or(0.6),
            gamma: self.gamma.unwrap_or(0.),
            vto: self.vto.unwrap_or(0.),
        }
    }

    pub(in crate::devices) fn derive(
        self,
        law: SurfaceLaw,
        label: &str,
        tnom: Real,
        pol: Real,
        cox: Real,
    ) -> SpiceResult<Surface> {
        let mut surface = self.defaults();
        let Some(nsub) = self.nsub else {
            return Ok(surface);
        };
        let density = nsub * 1e6;
        let vtnom = tnom * K_OVER_Q;
        let gap = band_gap(tnom);
        let intrinsic = match law {
            SurfaceLaw::Classic => INTRINSIC_DENSITY,
            SurfaceLaw::Mos3 => {
                let nifact = (tnom / 300.) * (tnom / 300.).sqrt();
                let nifact = nifact * (0.5 * gap * ((1. / 300.) - (1. / tnom)) / K_OVER_Q).exp();
                INTRINSIC_DENSITY * nifact
            }
        };
        if density <= intrinsic {
            return Err(SpiceError::circuit(format!(
                "{label} NSUB={nsub} is below the intrinsic density ({}temp.c: Nsub < Ni)",
                label.to_ascii_lowercase()
            )));
        }
        if self.phi.is_none() {
            surface.phi = (2. * vtnom * (density / intrinsic).ln()).max(0.1);
        }
        let fermis = pol * 0.5 * surface.phi;
        let tpg = self.tpg.unwrap_or(1.);
        let wkfng = if tpg == 0. {
            3.2
        } else {
            let fermig = pol * tpg * 0.5 * gap;
            0.5f64.mul_add(gap, 3.25) - fermig
        };
        let wkfngs = wkfng - (0.5f64.mul_add(gap, 3.25) + fermis);
        if self.gamma.is_none() {
            let product = match law {
                SurfaceLaw::Classic => 2. * 11.70 * EPSILON_0 * CHARGE * nsub * 1e6,
                SurfaceLaw::Mos3 => 2. * EPSILON_SI * CHARGE * nsub * 1e6,
            };
            surface.gamma = product.sqrt() / cox;
        }
        if self.vto.is_none() {
            let vfb = wkfngs - self.nss.unwrap_or(0.) * 1e4 * CHARGE / cox;
            surface.vto = pol.mul_add(surface.gamma * surface.phi.sqrt() + surface.phi, vfb);
        }
        Ok(surface)
    }
}

impl Surface {
    /// Temperature-dependent surface potential, built-in and threshold voltage.
    /// DELVTO is a MOS3/9-only term, added even when it is zero.
    pub(in crate::devices) fn at_temperature(
        self,
        nominal: &SiliconTemperature,
        actual: &SiliconTemperature,
        pol: Real,
        delvto: Option<Real>,
    ) -> (Real, Real, Real) {
        let phi = actual.potential.correct(&nominal.potential, self.phi);
        let vto = delvto.map_or(self.vto, |delta| delta + self.vto);
        let vbi = (pol * 0.5).mul_add(
            phi - self.phi,
            pol.mul_add(-(self.gamma * self.phi.sqrt()), vto) + 0.5 * (nominal.gap - actual.gap),
        );
        let vto = (pol * self.gamma).mul_add(phi.sqrt(), vbi);
        (phi, vbi, vto)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn explicit_process_values_override_extraction_independently() {
        let card = SurfaceCard {
            nsub: Some(1.5e16),
            tpg: Some(-1.),
            nss: Some(3e10),
            ..SurfaceCard::default()
        };
        for law in [SurfaceLaw::Classic, SurfaceLaw::Mos3] {
            for pol in [1., -1.] {
                let derived = card.derive(law, "MOS", 330., pol, 1e-3).unwrap();
                assert!(derived.phi > 0. && derived.gamma > 0. && derived.vto.is_finite());
                let given = SurfaceCard {
                    phi: Some(0.8),
                    gamma: Some(0.4),
                    vto: Some(-0.2),
                    ..card
                }
                .derive(law, "MOS", 330., pol, 1e-3)
                .unwrap();
                assert_eq!((given.phi, given.gamma, given.vto), (0.8, 0.4, -0.2));
                let changed_phi = SurfaceCard {
                    phi: Some(0.8),
                    ..card
                }
                .derive(law, "MOS", 330., pol, 1e-3)
                .unwrap();
                assert_eq!(changed_phi.phi, 0.8);
                assert_eq!(changed_phi.gamma.to_bits(), derived.gamma.to_bits());
                assert_ne!(changed_phi.vto.to_bits(), derived.vto.to_bits());
                let changed_gamma = SurfaceCard {
                    gamma: Some(0.4),
                    ..card
                }
                .derive(law, "MOS", 330., pol, 1e-3)
                .unwrap();
                assert_eq!(changed_gamma.phi.to_bits(), derived.phi.to_bits());
                assert_eq!(changed_gamma.gamma, 0.4);
                assert_ne!(changed_gamma.vto.to_bits(), derived.vto.to_bits());
            }
        }
    }

    #[test]
    fn gamma_retains_level_specific_product_order() {
        for nsub in [1.23456789e14, 1.3456789e16, 9.87345e17] {
            let cox = 1.234e-3;
            let card = SurfaceCard {
                nsub: Some(nsub),
                ..SurfaceCard::default()
            };
            let reference = [
                (2. * 11.70 * EPSILON_0 * CHARGE * nsub * 1e6).sqrt() / cox,
                (2. * EPSILON_SI * CHARGE * nsub * 1e6).sqrt() / cox,
            ];
            for (law, expected) in [SurfaceLaw::Classic, SurfaceLaw::Mos3]
                .into_iter()
                .zip(reference)
            {
                let got = card.derive(law, "MOS", 300., 1., cox).unwrap();
                assert_eq!(got.gamma.to_bits(), expected.to_bits());
            }
        }
    }

    #[test]
    fn temperature_adjusted_intrinsic_density_is_level_specific() {
        let card = SurfaceCard {
            nsub: Some(1e11),
            ..SurfaceCard::default()
        };
        assert!(
            card.derive(SurfaceLaw::Classic, "MOS1", 500., 1., 1e-3)
                .is_ok()
        );
        assert!(
            card.derive(SurfaceLaw::Mos3, "MOS3", 500., 1., 1e-3)
                .is_err()
        );
    }
}
