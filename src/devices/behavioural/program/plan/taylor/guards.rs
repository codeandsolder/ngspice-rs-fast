//! Collect only guards reached by the lazy physical value evaluation.
use super::{Binary, Evaluator, Instruction, Operator, Slot, Unary, discrete, discrete_limit};
use crate::devices::behavioural::program::{EXP_LIMIT, PowerLaw, PowerPart, unary_value};
use crate::devices::physical_dae::{LocalGuard, Region};
use crate::maths::taylor::Jet;
use crate::primitives::{Real, SpiceResult};
use std::cmp::Ordering;

impl Evaluator<'_> {
    pub(super) fn guards(&self) -> SpiceResult<Vec<LocalGuard>> {
        let mut result = Vec::new();
        for (slot, value) in self.values.iter().enumerate() {
            if value.is_none() {
                continue;
            }
            self.collect_guard(slot, self.plan.instructions[slot], &mut result)?;
        }
        Ok(result)
    }

    fn collect_guard(
        &self,
        slot: Slot,
        instruction: Instruction,
        out: &mut Vec<LocalGuard>,
    ) -> SpiceResult<()> {
        match instruction {
            Instruction::Unary(op, a) => {
                let x = self.read(a)?;
                match op {
                    Unary::Abs | Unary::Uramp => {
                        out.push(LocalGuard::sign(slot, 0, x.clone(), 0.));
                    }
                    Unary::Ustep2 => out.push(LocalGuard::interval(slot, x.clone(), &[0., 1.])),
                    Unary::Exp => out.push(LocalGuard::sign(slot, 0, x.clone(), EXP_LIMIT)),
                    Unary::Floor | Unary::Ceil | Unary::Nint => {
                        let value = discrete_limit(op, x);
                        let (lower, upper) = match op {
                            Unary::Floor => (value, value + 1.),
                            Unary::Ceil => (value - 1., value),
                            _ => (value - 0.5, value + 0.5),
                        };
                        out.push(LocalGuard {
                            slot,
                            operand: 0,
                            region: Region::Discrete(canonical_bits(value)),
                            argument: x.clone(),
                            boundaries: [Some(lower), Some(upper)],
                        });
                    }
                    op if discrete(op) => {
                        let germ = match x.sign() {
                            Ordering::Less => -1.,
                            Ordering::Equal => 0.,
                            Ordering::Greater => 1.,
                        };
                        let mut guard = LocalGuard::sign(slot, 0, x.clone(), 0.);
                        guard.region = Region::Discrete(canonical_bits(unary_value(op, germ)));
                        out.push(guard);
                    }
                    _ => {}
                }
            }
            Instruction::Ternary(c, ..) => {
                let x = self.read(c)?;
                let mut guard = LocalGuard::sign(slot, 0, x.clone(), 0.);
                guard.region = Region::Discrete(u64::from(x.sign() != Ordering::Equal));
                out.push(guard);
            }
            Instruction::Binary(Binary::Min | Binary::Max, a, b) => {
                let x = self.read(a)?.clone().add(&self.read(b)?.clone().scale(-1.));
                out.push(LocalGuard::sign(slot, 0, x, 0.));
            }
            Instruction::Operator(Operator::Divide, _, b) => {
                out.push(LocalGuard::sign(slot, 0, self.read(b)?.clone(), 0.));
            }
            Instruction::Operator(Operator::Power, a, b) => {
                self.power_guards(slot, PowerLaw::NativePow, a, b, out)?;
            }
            Instruction::Binary(Binary::Power(law, PowerPart::Value), a, b) => {
                self.power_guards(slot, law, a, b, out)?;
            }
            Instruction::Pwl(a, table) => {
                let x = self.read(a)?;
                let table = &self.plan.pwl[table];
                let upper = match table.order {
                    crate::devices::behavioural::program::PwlOrder::Ascending => Ordering::Less,
                    crate::devices::behavioural::program::PwlOrder::Descending => Ordering::Greater,
                };
                let mut guard = LocalGuard::interval(slot, x.clone(), &table.boundaries);
                guard.region = Region::Interval(
                    table.segment_index_by(|boundary| x.compare(boundary) == Some(upper)),
                );
                out.push(guard);
            }
            Instruction::Table(a, table) => {
                out.push(LocalGuard::interval(
                    slot,
                    self.read(a)?.clone(),
                    self.plan.tables[table].boundaries(),
                ));
            }
            _ => {}
        }
        Ok(())
    }

    fn power_guards(
        &self,
        slot: Slot,
        law: PowerLaw,
        a: Slot,
        b: Slot,
        out: &mut Vec<LocalGuard>,
    ) -> SpiceResult<()> {
        let base = self.read(a)?;
        let exponent = self.read(b)?;
        let integer =
            exponent.is_constant() && exponent.value() > 0. && exponent.value().fract() == 0.;
        let smooth = integer
            && match law {
                PowerLaw::NativePow | PowerLaw::SimetrixRealPart => true,
                PowerLaw::NativePwr => exponent.value().rem_euclid(2.).to_bits() == 1_f64.to_bits(),
                PowerLaw::SimetrixAbsolute => exponent.value().rem_euclid(2.) == 0.,
                // The measured SIMetrix caret regularization has an isolated
                // zero-base branch, even at an integral exponent.
                PowerLaw::SimetrixCaret => false,
            };
        if !smooth {
            out.push(LocalGuard::sign(slot, 0, base.clone(), 0.));
        }
        if law == PowerLaw::NativePow && base.sign() == Ordering::Less && !exponent.is_constant() {
            // Negative-base native pow has narrow quasi-integer windows.
            // Include their actual representable boundaries rather than
            // pretending a continuous exponent law exists.
            out.push(integer_window(slot, exponent));
        }
        Ok(())
    }
}

fn canonical_bits(value: Real) -> u64 {
    if value == 0. { 0 } else { value.to_bits() }
}

fn integer_window(slot: Slot, exponent: &Jet) -> LocalGuard {
    let center = exponent.value().round_ties_even();
    let inside = crate::primitives::almost_equal_ulps(center, exponent.value(), 10);
    let window = |center: Real, upper: bool| {
        (0..11).fold(center, |value, _| {
            if upper {
                value.next_up()
            } else {
                value.next_down()
            }
        })
    };
    let (lower, upper, region) = if inside {
        (
            window(center, false),
            window(center, true),
            canonical_bits(center),
        )
    } else {
        let floor = exponent.value().floor();
        (window(floor, true), window(floor + 1., false), u64::MAX)
    };
    LocalGuard {
        slot,
        operand: 1,
        region: Region::Discrete(region),
        argument: exponent.clone(),
        boundaries: [Some(lower), Some(upper)],
    }
}
