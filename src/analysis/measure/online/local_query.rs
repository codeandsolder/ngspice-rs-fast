//! Local sample geometry for a value/derivative query chosen by an event.
//! Preserve the raw first-inclusive derivative bracket, earlier duplicate
//! group, first three samples, and the selected query only.

use super::interpolate;
use crate::analysis::measure::{derivative_value, nonfinite, unsupported};
use crate::analysis::ordered_samples::same_coordinate;
use crate::netlist::ast::MeasureCard;
use crate::primitives::{Real, SpiceError, SpiceResult};
use std::sync::Arc;

#[derive(Clone)]
struct Datum {
    x: Real,
    value: Result<Real, Arc<SpiceError>>,
}

impl Datum {
    fn read(&self) -> SpiceResult<Real> {
        self.value
            .as_ref()
            .copied()
            .map_err(|error| (**error).clone())
    }
}

/// First-inclusive derivative bracket: j=1 uses the first three samples
/// (or the first two if the stream ends); otherwise the neighborhood ends
/// at j, with the preceding two raw samples.
#[derive(Clone)]
enum Geometry {
    Start,
    Local([Datum; 3]),
}

impl Geometry {
    fn new(previous: &[Option<Datum>; 2], current: &Datum) -> Self {
        if let [Some(a), Some(b)] = previous {
            Self::Local([a.clone(), b.clone(), current.clone()])
        } else {
            Self::Start
        }
    }

    fn derivative(
        &self,
        at: Real,
        first: &[Option<Datum>; 3],
        card: &MeasureCard,
    ) -> SpiceResult<Real> {
        match self {
            Self::Start => {
                let [Some(a), Some(b), c] = first else {
                    return Err(unsupported(
                        card,
                        "DERIV needs a bracketing interval".into(),
                    ));
                };
                let bracket = [(0, a.x), (1, b.x)];
                let triple = c.as_ref().map(|c| [(0, a.x), (1, b.x), (2, c.x)]);
                derivative_value(
                    at,
                    bracket,
                    triple,
                    |i| {
                        first
                            .get(i)
                            .and_then(Option::as_ref)
                            .ok_or_else(|| SpiceError::circuit("missing local derivative sample"))?
                            .read()
                    },
                    card,
                )
            }
            Self::Local(points) => derivative_value(
                at,
                [(1, points[1].x), (2, points[2].x)],
                Some([(0, points[0].x), (1, points[1].x), (2, points[2].x)]),
                |i| points[i].read(),
                card,
            ),
        }
    }
}

#[derive(Clone)]
enum LocalFailure {
    Discontinuous,
    Invalid(Arc<SpiceError>),
}

#[derive(Clone)]
struct Group {
    x: Real,
    value: Result<Real, LocalFailure>,
    geometry: Geometry,
}

impl Group {
    fn new(current: &Datum, geometry: Geometry) -> Self {
        Self {
            x: current.x,
            value: current.value.clone().map_err(LocalFailure::Invalid),
            geometry,
        }
    }

    fn include(&mut self, sample: &Datum) {
        if let Ok(first) = self.value {
            self.value = sample
                .value
                .clone()
                .map_err(LocalFailure::Invalid)
                .and_then(|value| {
                    if same_coordinate(first, value) {
                        Ok(first)
                    } else {
                        Err(LocalFailure::Discontinuous)
                    }
                });
        }
    }

    fn located(&self) -> Located {
        Located {
            geometry: self.geometry.clone(),
            value: LocationValue::Exact(self.value.clone()),
        }
    }
}

enum LocationValue {
    Exact(Result<Real, LocalFailure>),
    Between(Datum, Datum),
}

struct Located {
    geometry: Geometry,
    value: LocationValue,
}

fn exact_value(value: Result<Real, LocalFailure>, card: &MeasureCard) -> SpiceResult<Real> {
    value.map_err(|error| match error {
        LocalFailure::Discontinuous => {
            unsupported(card, "the operand is discontinuous at AT=".into())
        }
        LocalFailure::Invalid(error) => (*error).clone(),
    })
}

impl Located {
    fn evaluate(
        self,
        at: Real,
        derivative: bool,
        first: &[Option<Datum>; 3],
        card: &MeasureCard,
    ) -> SpiceResult<Real> {
        if derivative {
            if let LocationValue::Exact(value) = self.value {
                exact_value(value, card)?;
            }
            self.geometry.derivative(at, first, card)
        } else {
            let value = match self.value {
                LocationValue::Exact(value) => exact_value(value, card)?,
                LocationValue::Between(a, b) => interpolate(a.x, a.read()?, b.x, b.read()?, at),
            };
            if !value.is_finite() {
                return Err(nonfinite(
                    card,
                    "AT=",
                    "nonfinite interpolated value".into(),
                ));
            }
            Ok(value)
        }
    }
}

/// Only the first three, two most recent raw samples, two coordinate groups
/// and one selected location are kept. A later LAST crossing replaces the
/// location (including its pending duplicate validation) without retaining
/// the earlier event or any intervening rows.
pub(super) struct LocalHistory {
    first: [Option<Datum>; 3],
    raw: [Option<Datum>; 2],
    group: Option<Group>,
    preceding: Option<Group>,
    selected: Option<(Real, Located)>,
}

impl LocalHistory {
    pub(super) const fn new() -> Self {
        Self {
            first: [None, None, None],
            raw: [None, None],
            group: None,
            preceding: None,
            selected: None,
        }
    }

    pub(super) fn observe(&mut self, x: Real, value: SpiceResult<Real>) {
        let current = Datum {
            x,
            value: value.map_err(Arc::new),
        };
        if let Some(slot) = self.first.iter_mut().find(|slot| slot.is_none()) {
            *slot = Some(current.clone());
        }
        if let Some(group) = &mut self.group
            && same_coordinate(group.x, x)
        {
            group.include(&current);
            // The selected query may have landed on the first sample of a
            // group; validate later duplicates even after its crossing.
            if let Some((at, selected)) = &mut self.selected
                && same_coordinate(*at, x)
            {
                selected.value = LocationValue::Exact(group.value.clone());
            }
        } else {
            self.preceding = self.group.take();
            self.group = Some(Group::new(&current, Geometry::new(&self.raw, &current)));
        }
        self.raw[0] = self.raw[1].take();
        self.raw[1] = Some(current);
    }

    pub(super) fn select(&mut self, at: Real) {
        if self
            .selected
            .as_ref()
            .is_some_and(|(x, _)| same_coordinate(*x, at))
        {
            return;
        }
        for group in [&self.group, &self.preceding].into_iter().flatten() {
            if same_coordinate(group.x, at) {
                self.selected = Some((at, group.located()));
                return;
            }
        }
        if let [Some(left), Some(right)] = &self.raw
            && left.x < at
            && at < right.x
            && let Some(group) = &self.group
        {
            self.selected = Some((
                at,
                Located {
                    geometry: group.geometry.clone(),
                    value: LocationValue::Between(left.clone(), right.clone()),
                },
            ));
        }
    }

    pub(super) fn finish(
        self,
        at: Real,
        derivative: bool,
        card: &MeasureCard,
    ) -> SpiceResult<Real> {
        if derivative && self.first[1].is_none() {
            return Err(unsupported(
                card,
                "DERIV needs a bracketing interval".into(),
            ));
        }
        let located = self
            .selected
            .filter(|(x, _)| same_coordinate(*x, at))
            .ok_or_else(|| {
                unsupported(card, "AT= is outside the local measurement bracket".into())
            })?
            .1;
        located.evaluate(at, derivative, &self.first, card)
    }
}
