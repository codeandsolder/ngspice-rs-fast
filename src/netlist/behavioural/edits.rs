//! Lower nonlinear controlled sources while retaining ordered source cards.
use super::lower_device;
use crate::netlist::ast::{
    DeviceInstance, Netlist, ScopeView, ScopedCard, ScopedCardKind, Subcircuit,
};
use crate::primitives::{SpiceError, SpiceResult};

pub(super) fn lower(netlist: &mut Netlist) -> SpiceResult<()> {
    lower_scope(
        &mut netlist.devices,
        &mut netlist.cards,
        &mut netlist.subcircuits,
    )
}

fn lower_scope(
    devices: &mut Vec<DeviceInstance>,
    cards: &mut Vec<ScopedCard>,
    children: &mut [Subcircuit],
) -> SpiceResult<()> {
    let originals = std::mem::take(devices);
    let mut out = Vec::with_capacity(originals.len());
    let mut ranges = Vec::with_capacity(originals.len());
    for device in originals {
        let start = out.len();
        lower_device(device, &mut out)?;
        ranges.push(start..out.len());
    }
    // Generated cards inherit the original card's provenance, but each
    // receives its own semantic ordinal. Include positions and all other
    // indexed collections remain unchanged.
    let original_cards = std::mem::take(cards);
    cards.reserve(original_cards.len() + out.len() - ranges.len());
    for card in original_cards {
        if let ScopedCardKind::Device(index) = card.kind {
            let range = ranges[index].clone();
            if let Some(last) = range.clone().next_back() {
                for index in range.start..last {
                    cards.push(ScopedCard {
                        kind: ScopedCardKind::Device(index),
                        ..card.clone()
                    });
                }
                cards.push(ScopedCard {
                    kind: ScopedCardKind::Device(last),
                    ..card
                });
            }
        } else {
            cards.push(card);
        }
    }

    *devices = out;
    for child in children {
        lower_scope(&mut child.devices, &mut child.cards, &mut child.subcircuits)?;
    }
    Ok(())
}

/// Validate all original scope ordinals before editing any of them.
pub(super) fn validate(scope: ScopeView<'_>) -> SpiceResult<()> {
    if !scope.cards.is_empty() {
        scope
            .validate_order()
            .map_err(|error| SpiceError::circuit(error.to_string()))?;
    }
    for child in scope.subcircuits {
        validate(ScopeView::from(child))?;
    }
    Ok(())
}
