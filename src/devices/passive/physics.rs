//! Arithmetic shared by ordinary passive loads and C-ordered sensitivity replay.
//!
//! Inputs distinguish written values from defaults. These kernels deliberately
//! leave validation and diagnostic attribution to the caller: sensitivity
//! perturbations may visit states disallowed by the ordinary bounded schema.

use crate::primitives::Real;

// ngspice const.h at the pinned C revision. Derive epsilon from the
// same constants and product order rather than using the older MOS constant.
const MU_ZERO: Real = 4. * std::f64::consts::PI * 1e-7;
const SPEED_OF_LIGHT: Real = 299_792_458.;
const EPS_ZERO: Real = 1. / (MU_ZERO * SPEED_OF_LIGHT * SPEED_OF_LIGHT);
const EPS_SIO2: Real = 3.9 * EPS_ZERO;

/// Resolve temperature and offset, both in kelvin. An explicit temperature
/// disables the offset even if it was written.
#[derive(Debug, Clone, Copy)]
pub(in crate::devices) struct TemperaturePoint {
    pub(in crate::devices) kelvin: Real,
    pub(in crate::devices) offset: Real,
}

impl TemperaturePoint {
    pub(in crate::devices) fn resolve(
        given: Option<Real>,
        offset: Option<Real>,
        circuit: Real,
    ) -> Self {
        given.map_or_else(
            || Self {
                kelvin: circuit,
                offset: offset.unwrap_or(0.),
            },
            |kelvin| Self { kelvin, offset: 0. },
        )
    }

    pub(in crate::devices) fn effective(self) -> Real {
        self.kelvin + self.offset
    }

    pub(in crate::devices) fn difference(self, nominal: Real) -> Real {
        self.effective() - nominal
    }
}

/// Independently overridden temperature coefficients.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(in crate::devices) struct Coefficients {
    pub(in crate::devices) linear: Real,
    pub(in crate::devices) quadratic: Real,
}

impl Coefficients {
    const fn resistor(self, difference: Real) -> Real {
        ((self.quadratic * difference) + self.linear) * difference + 1.
    }

    pub(in crate::devices) fn resistor_factor(
        self,
        difference: Real,
        exponential: Option<Real>,
    ) -> Real {
        exponential.map_or_else(
            || self.resistor(difference),
            |tce| 1.01_f64.powf(tce * difference),
        )
    }

    pub(in crate::devices) const fn reactive(self, difference: Real) -> Real {
        1. + self.linear * difference + self.quadratic * difference * difference
    }
}

/// C's separately rounded base, temperature factor and scale product.
pub(in crate::devices) const fn scaled(base: Real, factor: Real, scale: Real) -> Real {
    base * factor * scale
}

/// C stamps conductance directly, not the reciprocal of resistance divided by m.
pub(in crate::devices) const fn conductance(scaled: Real, m: Real) -> Real {
    m / scaled
}

/// Resistor dimensions and model corrections.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(in crate::devices) struct SheetGeometry {
    pub(in crate::devices) length: Real,
    pub(in crate::devices) width: Real,
    pub(in crate::devices) short: Real,
    pub(in crate::devices) narrow: Real,
    pub(in crate::devices) sheet: Real,
}

impl SheetGeometry {
    pub(in crate::devices) const fn dimensions(self) -> (Real, Real) {
        (self.length - 2. * self.short, self.width - 2. * self.narrow)
    }

    pub(in crate::devices) const fn uses_sheet(self) -> bool {
        self.length * self.width * self.sheet > 0.
    }

    pub(in crate::devices) const fn resistance(self) -> Real {
        let (length, width) = self.dimensions();
        length / width * self.sheet
    }
}

/// Written capacitor model geometry; defaults/derived quantities are prepared
/// at setup, not recomputed by temperature-time perturbations.
#[derive(Debug, Clone, Copy)]
pub(in crate::devices) struct CapacitorSetup {
    pub(in crate::devices) cj: Option<Real>,
    pub(in crate::devices) cjsw: Real,
    pub(in crate::devices) narrow: Option<Real>,
    pub(in crate::devices) short: Option<Real>,
    pub(in crate::devices) del: Option<Real>,
    pub(in crate::devices) dielectric: Option<Real>,
    pub(in crate::devices) thickness: Option<Real>,
}

/// Prepared capacitor geometry coefficients.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(in crate::devices) struct CapacitorGeometry {
    pub(in crate::devices) cj: Real,
    pub(in crate::devices) cjsw: Real,
    pub(in crate::devices) narrow: Real,
    pub(in crate::devices) short: Real,
}

impl CapacitorSetup {
    pub(in crate::devices) fn prepare(self) -> CapacitorGeometry {
        CapacitorGeometry {
            cj: self.cj.unwrap_or_else(|| {
                self.thickness
                    .filter(|thick| *thick > 0.)
                    .map_or(0., |thick| {
                        self.dielectric.map_or(EPS_SIO2, |di| di * EPS_ZERO) / thick
                    })
            }),
            cjsw: self.cjsw,
            narrow: self.narrow.unwrap_or_else(|| 2. * self.del.unwrap_or(0.)),
            short: self.short.unwrap_or_else(|| 2. * self.del.unwrap_or(0.)),
        }
    }
}

impl CapacitorGeometry {
    pub(in crate::devices) const fn dimensions(self, length: Real, width: Real) -> (Real, Real) {
        (length - self.short, width - self.narrow)
    }

    pub(in crate::devices) const fn capacitance(self, length: Real, width: Real) -> Real {
        let (length, width) = self.dimensions(length, width);
        self.cj * width * length + self.cjsw * 2. * (length + width)
    }
}

/// Written coil model setup fields.
#[derive(Debug, Clone, Copy)]
pub(in crate::devices) struct CoilSetup {
    pub(in crate::devices) inductance: Option<Real>,
    pub(in crate::devices) cross_section: Option<Real>,
    pub(in crate::devices) diameter: Option<Real>,
    pub(in crate::devices) length: Option<Real>,
    pub(in crate::devices) turns: Real,
    pub(in crate::devices) permeability: Real,
}

/// A coil's setup-only specific and model inductance.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(in crate::devices) struct CoilGeometry {
    pub(in crate::devices) cross_section: Real,
    pub(in crate::devices) specific: Real,
    pub(in crate::devices) model_inductance: Real,
}

impl CoilSetup {
    pub(in crate::devices) fn prepare(self) -> CoilGeometry {
        let cross_section = self.diameter.map_or_else(
            || self.cross_section.unwrap_or(0.),
            |dia| std::f64::consts::PI * dia * dia / 4.,
        );
        let mut specific = self
            .length
            .filter(|length| *length > 0.)
            .map_or(0., |length| {
                self.permeability * MU_ZERO * cross_section / length
            });
        if let Some(length) = self.length
            && (self.diameter.is_some() || self.cross_section.is_some())
        {
            specific *= lundin(length, cross_section);
        }
        CoilGeometry {
            cross_section,
            specific,
            model_inductance: self
                .inductance
                .unwrap_or(self.turns * self.turns * specific),
        }
    }
}

impl CoilGeometry {
    pub(in crate::devices) fn inductance(
        self,
        explicit: Option<Real>,
        turns: Option<Real>,
    ) -> Real {
        explicit
            .unwrap_or_else(|| turns.map_or(self.model_inductance, |nt| self.specific * nt * nt))
    }
}

/// `Lundin()` of `indsetup.c`: the correction factor of a solenoid.
/// Horner form avoids a separately rounded fourth power. For short
/// coils the reciprocal aspect ratio avoids overflowing its powers.
fn lundin(length: Real, csect: Real) -> Real {
    if csect < 1e-12 || length < 1e-6 {
        return 1.;
    }
    let x = (csect / std::f64::consts::PI).sqrt() * 2. / length;
    if x < 1. {
        let xx = x * x;
        let num = 0.017_108_f64.mul_add(xx, 0.383_901).mul_add(xx, 1.);
        let den = 0.258_952_f64.mul_add(xx, 1.);
        (-4. / (3. * std::f64::consts::PI)).mul_add(x, num / den)
    } else {
        let inverse = x.recip();
        let xx = inverse * inverse;
        let num = ((4. * x).ln() - 0.5) * 0.017_108_f64.mul_add(xx, 0.383_901).mul_add(xx, 1.);
        let den = 0.258_952_f64.mul_add(xx, 1.);
        let kk = (-0.000_801_f64)
            .mul_add(xx, 0.002_029)
            .mul_add(xx, 0.093_842)
            * xx;
        (2. / std::f64::consts::PI) * inverse * (num / den + kk)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lundin_matches_independent_polynomial_and_short_coil_limit() {
        let pi = std::f64::consts::PI;
        for x in [0.01_f64, 0.1, 0.999, 1., 2., 10., 1e20, 1e150] {
            let length = 2. / x;
            // Set length above the reference cutoff by using cross-section
            // rather than shrinking the coil out of the model's domain.
            let (length, area) = if length >= 1e-6 {
                (length, pi)
            } else {
                (2., pi * x * x)
            };
            let expected = if x < 1. {
                (1. + 0.383_901 * x.powi(2) + 0.017_108 * x.powi(4)) / (1. + 0.258_952 * x.powi(2))
                    - 4. * x / (3. * pi)
            } else {
                let inverse = x.recip();
                let numerator = ((4. * x).ln() - 0.5)
                    * (1. + 0.383_901 * inverse.powi(2) + 0.017_108 * inverse.powi(4));
                let denominator = 1. + 0.258_952 * inverse.powi(2);
                let correction = 0.093_842 * inverse.powi(2) + 0.002_029 * inverse.powi(4)
                    - 0.000_801 * inverse.powi(6);
                2. * (numerator / denominator + correction) / (pi * x)
            };
            assert!((lundin(length, area) - expected).abs() <= 8. * Real::EPSILON * expected.abs());
        }
    }
}
