//! Shared electrical control ports for continuous and event-driven switches.

use crate::devices::traits::ControlReference;
use crate::maths::Vector;
use crate::netlist::ast::{CodePortKind, CodeTerminal};
use crate::primitives::{NodeAppend, NodeId, Real, SourceLoc, SpiceError, SpiceResult};

#[derive(Debug)]
pub(super) enum SwitchControl {
    Voltage,
    Current,
    Named(ControlReference),
}

pub(super) struct ControlRows {
    pub(super) coordinates: [Option<usize>; 2],
    pub(super) ammeter: Option<usize>,
}

pub(super) fn read_pair(rows: [Option<usize>; 2], values: &Vector) -> SpiceResult<Real> {
    let read = |row: Option<usize>| {
        row.map_or(Ok(0.), |row| {
            values
                .get(row)
                .ok_or_else(|| SpiceError::circuit("unbound switch control coordinate"))
        })
    };
    Ok(read(rows[0])? - read(rows[1])?)
}

impl SwitchControl {
    pub(super) fn parse(terminal: &CodeTerminal, location: &SourceLoc) -> SpiceResult<Self> {
        Ok(match terminal.kind {
            CodePortKind::Voltage
            | CodePortKind::VoltageDifferential
            | CodePortKind::Conductance
            | CodePortKind::ConductanceDifferential => Self::Voltage,
            CodePortKind::Current | CodePortKind::CurrentDifferential => Self::Current,
            CodePortKind::BranchCurrent => Self::Named(ControlReference {
                name: terminal.positive.clone(),
                location: Some(location.clone()),
            }),
            _ => {
                return Err(SpiceError::parse(
                    location.clone(),
                    "invalid switch control port",
                ));
            }
        })
    }

    pub(super) fn bind_nodes(
        &self,
        terminal: &CodeTerminal,
        nodes: &mut NodeAppend<'_>,
        location: &SourceLoc,
    ) -> SpiceResult<[NodeId; 2]> {
        if self.is_named() {
            return Ok([NodeId::GROUND; 2]);
        }
        let pair = super::source::analog_pair(terminal, nodes)?;
        if self.is_sensor() && pair[0] == pair[1] {
            return Err(SpiceError::parse(
                location.clone(),
                "shorted ideal switch current input",
            ));
        }
        Ok(pair)
    }

    pub(super) const fn is_sensor(&self) -> bool {
        matches!(self, Self::Current)
    }

    pub(super) const fn is_named(&self) -> bool {
        matches!(self, Self::Named(_))
    }

    pub(super) const fn is_current(&self) -> bool {
        !matches!(self, Self::Voltage)
    }

    pub(super) fn references(&self) -> &[ControlReference] {
        match self {
            Self::Named(reference) => std::slice::from_ref(reference),
            _ => &[],
        }
    }

    pub(super) fn rows(
        &self,
        input: [Option<usize>; 2],
        branch: Option<usize>,
        controls: &[usize],
    ) -> SpiceResult<ControlRows> {
        let (coordinates, ammeter) = match self {
            Self::Voltage => (input, None),
            Self::Current => {
                let row = branch
                    .ok_or_else(|| SpiceError::circuit("missing switch current input branch"))?;
                ([Some(row), None], Some(row))
            }
            Self::Named(_) => {
                let [row] = controls else {
                    return Err(SpiceError::circuit("missing named switch current input"));
                };
                ([Some(*row), None], None)
            }
        };
        Ok(ControlRows {
            coordinates,
            ammeter,
        })
    }
}
