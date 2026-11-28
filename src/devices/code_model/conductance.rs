//! Electrical realization of a digital drive. A grouped electrical table
//! is explicit, never an implicit strength-to-resistance conversion.

use super::{Drive, Logic, OutputRamp, RampLaw, Strength};
use crate::primitives::{Real, SpiceError, SpiceResult};
use std::sync::Arc;

const RANKS: usize = Strength::STRONG.rank() as usize;
const LEVELS: usize = 5;

#[derive(Clone, Debug)]
enum DriveTable {
    Grouped {
        values: [Real; 7],
        phases: Option<[Real; 4]>,
    },
    Ranked(Arc<[Real]>),
}

impl DriveTable {
    fn target(&self, drive: Drive) -> SpiceResult<Real> {
        let rank = usize::from(drive.strength.rank());
        let logic = drive.logic;
        match self {
            Self::Ranked(values) => Ok(values[if rank == 0 {
                0
            } else {
                1 + (rank - 1) * LEVELS + logic.index()
            }]),
            Self::Grouped { values, phases } => {
                if rank == 0 {
                    return Ok(values[6]);
                }
                let resistive = drive.strength != Strength::STRONG;
                if matches!(logic, Logic::Rising | Logic::Falling) {
                    let phases = phases.ok_or_else(|| SpiceError::circuit(
                        "directional timing ambiguity requires explicit R/F conductance entries"
                    ))?;
                    return Ok(
                        phases[usize::from(resistive) * 2 + usize::from(logic == Logic::Falling)]
                    );
                }
                Ok(values[usize::from(resistive) * 3 + logic.index()])
            }
        }
    }

    fn finite_resistances(&self) -> bool {
        let valid = |g: &Real| *g > 0. && g.recip().is_finite() && g.recip() > 0.;
        match self {
            Self::Grouped { values, phases } => {
                values.iter().all(valid) && phases.as_ref().is_none_or(|p| p.iter().all(valid))
            }
            Self::Ranked(values) => values.iter().all(valid),
        }
    }
}

/// Destination-specific duration and dimensionless duration/time-constant
/// ratio. The law settles exactly at the declared endpoint. Both are
/// supplied by the source binding; no dialect-specific defaults are inferred.
#[derive(Clone, Copy, Debug)]
pub(super) struct ResistanceSlew {
    duration: [Real; 6],
    shape: Real,
}

impl ResistanceSlew {
    pub(super) fn new(duration: [Real; 6], shape: Real) -> SpiceResult<Self> {
        if duration.iter().any(|d| !d.is_finite() || *d < 0.) || !shape.is_finite() || shape <= 0. {
            return Err(SpiceError::circuit(
                "invalid resistance slew duration or exponential shape",
            ));
        }
        Ok(Self { duration, shape })
    }

    fn duration(self, drive: Drive) -> Real {
        self.duration[if drive.strength == Strength::HIGH_IMPEDANCE {
            LEVELS
        } else {
            drive.logic.index()
        }]
    }
}

#[derive(Clone, Debug)]
pub(crate) struct ConductanceConfig {
    table: DriveTable,
    inverted: bool,
    slew: Option<ResistanceSlew>,
}

impl ConductanceConfig {
    pub(super) fn new(
        values: &[Real],
        ranked: bool,
        slew: Option<ResistanceSlew>,
    ) -> SpiceResult<Self> {
        if values.iter().any(|value| !value.is_finite() || *value < 0.) {
            return Err(SpiceError::circuit("invalid digital output conductance"));
        }
        let table = if ranked {
            if values.len() != 1 + RANKS * LEVELS {
                return Err(SpiceError::circuit(format!(
                    "ranked conductance table requires {} entries",
                    1 + RANKS * LEVELS
                )));
            }
            DriveTable::Ranked(values.into())
        } else {
            let (values, phases) = match values {
                [a, b, c, d, e, f, z] => ([*a, *b, *c, *d, *e, *f, *z], None),
                [a, b, c, d, e, f, z, sr, sf, rr, rf] => {
                    ([*a, *b, *c, *d, *e, *f, *z], Some([*sr, *sf, *rr, *rf]))
                }
                _ => {
                    return Err(SpiceError::circuit(
                        "logic conductance table requires seven or eleven entries",
                    ));
                }
            };
            DriveTable::Grouped { values, phases }
        };
        if slew.is_some_and(|s| s.duration.iter().any(|&d| d > 0.)) && !table.finite_resistances() {
            return Err(SpiceError::circuit(
                "resistance slew requires finite positive resistance at every declared state",
            ));
        }
        Ok(Self {
            table,
            inverted: false,
            slew,
        })
    }

    pub(super) fn with_inversion(mut self, inverted: bool) -> Self {
        self.inverted = inverted;
        self
    }

    fn physical_drive(&self, drive: Drive) -> Drive {
        if self.inverted {
            Drive {
                logic: drive.logic.invert(),
                ..drive
            }
        } else {
            drive
        }
    }

    pub(super) fn target(&self, drive: Drive) -> SpiceResult<Real> {
        self.table.target(self.physical_drive(drive))
    }

    pub(super) fn retarget(
        &self,
        ramp: OutputRamp,
        time: Real,
        drive: Drive,
    ) -> SpiceResult<OutputRamp> {
        let drive = self.physical_drive(drive);
        let target = self.table.target(drive)?;
        if ramp.target.to_bits() == target.to_bits() {
            return Ok(ramp);
        }
        let Some(slew) = self.slew else {
            return Ok(OutputRamp::initial(target));
        };
        let duration = slew.duration(drive);
        if duration == 0. {
            return Ok(OutputRamp::initial(target));
        }
        let end = time + duration;
        if !end.is_finite() || end <= time {
            return Err(SpiceError::circuit(
                "resistance slew outside representable time range",
            ));
        }
        let from = ramp.value(time);
        Ok(OutputRamp {
            start: time,
            end,
            from,
            target,
            law: RampLaw::Resistance { shape: slew.shape },
        })
    }
}

/// Normalize an exponential change in resistance over the declared
/// interval. Use both weights directly to avoid subtracting nearly
/// equal resistances near either endpoint.
pub(super) fn resistance_value(from: Real, to: Real, u: Real, shape: Real) -> Real {
    let relative_exp = |x: Real| if x == 0. { 1. } else { -(-x).exp_m1() / x };
    let x = shape * u;
    let y = shape * (1. - u);
    let normalization = relative_exp(shape);
    let r0 = from.recip();
    let r1 = to.recip();
    let w1 = (u * relative_exp(x)) / normalization;
    let w0_unscaled = ((1. - u) * relative_exp(y)) / normalization;
    let decay = (-x).exp();
    // A subnormal decay may lose a material contribution when the
    // starting resistance is very large. Evaluate that product in
    // the logarithmic domain only when necessary.
    let first = if decay < Real::MIN_POSITIVE && w0_unscaled > 0. {
        (r0.ln() + w0_unscaled.ln() - x).exp()
    } else {
        r0 * (w0_unscaled * decay)
    };
    let r = first + r1 * w1;
    r.clamp(r0.min(r1), r0.max(r1)).recip()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_rank_table_preserves_all_levels_and_disconnection() -> SpiceResult<()> {
        let values: Vec<Real> = (0..=RANKS * LEVELS).map(|i| i as Real * 1e-6).collect();
        let config = ConductanceConfig::new(&values, true, None)?;
        for rank in 0..=Strength::STRONG.rank() {
            for logic in [
                Logic::Zero,
                Logic::One,
                Logic::Unknown,
                Logic::Rising,
                Logic::Falling,
            ] {
                let drive = Drive {
                    logic,
                    strength: Strength::from_rank(rank)?,
                };
                let expected = if rank == 0 {
                    0
                } else {
                    1 + (usize::from(rank) - 1) * LEVELS + logic.index()
                };
                assert_eq!(config.target(drive)?, values[expected]);
                assert_eq!(
                    config.clone().with_inversion(true).target(drive)?,
                    config.target(Drive {
                        logic: logic.invert(),
                        ..drive
                    })?
                );
            }
        }
        Ok(())
    }

    #[test]
    fn resistance_ramp_retargets_from_current_value() -> SpiceResult<()> {
        let slew = ResistanceSlew::new([2., 4., 3., 1., 1., 0.], 2.)?;
        let config = ConductanceConfig::new(
            &[0.01, 0.001, 0.002, 0.01, 0.001, 0.002, 0.0001],
            false,
            Some(slew),
        )?;
        let ramp = config.retarget(OutputRamp::initial(0.01), 10., Drive::strong(Logic::One))?;
        assert_eq!(ramp.value(10.), 0.01);
        assert_eq!(ramp.value(14.), 0.001);
        let middle = ramp.value(12.);
        let expected = (100. + 900. * (-(-1_f64).exp_m1()) / (-(-2_f64).exp_m1())).recip();
        assert!((middle / expected - 1.).abs() < 1e-14);
        let reverse = config.retarget(ramp, 12., Drive::strong(Logic::Zero))?;
        assert_eq!(reverse.value(12.), middle);
        assert_eq!(reverse.value(14.), 0.01);
        assert_eq!(reverse.next_break(12.), Some(14.));
        assert!(ConductanceConfig::new(&[0., 1., 1., 1., 1., 1., 1.], false, Some(slew)).is_err());
        Ok(())
    }

    #[test]
    fn resistance_interpolation_stays_finite_across_extreme_shapes() {
        for shape in [Real::MIN_POSITIVE, 1e-10, 1., 100., 1e3, 1e300] {
            for u in [0.01, 0.1, 0.5, 0.9, 0.99] {
                let value = resistance_value(1e-200, 1e200, u, shape);
                assert!(
                    value.is_finite() && (1e-200..=1e200).contains(&value),
                    "{shape} {u} {value}"
                );
            }
        }
    }
}
