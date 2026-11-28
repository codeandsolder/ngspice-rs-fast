//! Physical edits for an independent repetition, before topology allocation.
use std::borrow::Cow;
use std::collections::BTreeSet;

use crate::devices::subckt::ExpandedNetlist;
use crate::netlist::ast::{ModelCard, Netlist, ParameterAssignment, ParameterKind, Subcircuit};
use crate::primitives::{Real, SourceLoc, SpiceError, SpiceResult};

use super::{CandidateNumericOverride, Circuit, ModelContext};

/// An independent scalar edit in the emitted native namespace.
#[derive(Clone, Copy, Debug)]
pub enum CandidateEdit<'a> {
    /// Independent voltage or current source DC value.
    Source {
        /// Expanded native source name.
        name: &'a str,
        /// DC value in the source's native unit.
        value: Real,
    },
    /// A native parameter of a specific expanded device occurrence.
    Instance {
        /// Expanded native instance name.
        name: &'a str,
        /// Native scalar setter name.
        parameter: &'a str,
        /// Value in native card units.
        value: Real,
    },
    /// A native model parameter of a lexical declaration, shared by its uses.
    Model {
        /// Lexical declaration path, empty for root.
        scope: &'a [String],
        /// Native model declaration name.
        name: &'a str,
        /// Native scalar setter name.
        parameter: &'a str,
        /// Value in native card units.
        value: Real,
    },
}

fn unique<'a, T>(items: &'a mut [T], matches: impl Fn(&T) -> bool) -> SpiceResult<&'a mut T> {
    let mut found = items.iter_mut().filter(|item| matches(item));
    let item = found
        .next()
        .ok_or_else(|| SpiceError::circuit("unknown candidate edit target"))?;
    if found.next().is_some() {
        return Err(SpiceError::circuit("ambiguous candidate edit target"));
    }
    Ok(item)
}

fn models_at<'a>(
    models: &'a mut [ModelCard],
    subcircuits: &'a mut [Subcircuit],
    scope: &[String],
) -> SpiceResult<&'a mut [ModelCard]> {
    let Some((head, tail)) = scope.split_first() else {
        return Ok(models);
    };
    let sub = unique(subcircuits, |sub| sub.name.eq_ignore_ascii_case(head))?;
    models_at(&mut sub.models, &mut sub.subcircuits, tail)
}

fn assignment(
    parameter: &str,
    value: Real,
    location: &SourceLoc,
) -> SpiceResult<ParameterAssignment> {
    if !value.is_finite() || parameter.is_empty() {
        return Err(SpiceError::circuit("invalid scalar candidate edit"));
    }
    Ok(ParameterAssignment {
        name: parameter.to_ascii_lowercase(),
        value: value.to_string(),
        kind: ParameterKind::Scalar,
        location: location.clone(),
    })
}

pub(super) fn edit_models<'a>(
    netlist: &'a Netlist,
    edits: &[CandidateEdit<'_>],
) -> SpiceResult<Cow<'a, Netlist>> {
    let mut result = Cow::Borrowed(netlist);
    let mut seen = BTreeSet::new();
    for edit in edits {
        let CandidateEdit::Model {
            scope,
            name,
            parameter,
            value,
        } = *edit
        else {
            continue;
        };
        if !seen.insert((
            scope
                .iter()
                .map(|s| s.to_ascii_lowercase())
                .collect::<Vec<_>>(),
            name.to_ascii_lowercase(),
            parameter.to_ascii_lowercase(),
        )) {
            return Err(SpiceError::circuit("duplicate model candidate edit"));
        }
        let deck = result.to_mut();
        let models = models_at(&mut deck.models, &mut deck.subcircuits, scope)?;
        let model = unique(models, |model| model.name.eq_ignore_ascii_case(name))?;
        let setter = assignment(parameter, value, &model.location)?;
        // The level selector is also cached from the first lexical setter.
        if setter.name == "level" {
            model.parameters.retain(|p| p.name != "level");
            model.level = Some(value);
        }
        model.parameters.push(setter);
    }
    Ok(result)
}

pub(super) fn edit_instances<'a>(
    expanded: &mut ExpandedNetlist,
    edits: &[CandidateEdit<'a>],
) -> SpiceResult<Vec<(&'a str, Real)>> {
    let mut sources = Vec::new();
    let mut seen = BTreeSet::new();
    for edit in edits {
        let (name, parameter, value, source) = match *edit {
            CandidateEdit::Source { name, value } => (name, "dc", value, true),
            CandidateEdit::Instance {
                name,
                parameter,
                value,
            } => (name, parameter, value, false),
            CandidateEdit::Model { .. } => continue,
        };
        let device = unique(&mut expanded.devices, |device| {
            device.name.eq_ignore_ascii_case(name)
        })?;
        let setter = assignment(parameter, value, &device.location)?;
        let dc = source
            || (matches!(device.designator, 'v' | 'i')
                && (setter.name == "dc" || (device.designator == 'i' && setter.name == "c")));
        let key = if dc { "dc" } else { setter.name.as_str() };
        if !seen.insert((name.to_ascii_lowercase(), key.to_owned())) {
            return Err(SpiceError::circuit("duplicate instance candidate edit"));
        }
        if dc {
            sources.push((name, value));
            continue;
        }
        // These laws replace native card semantics. A scalar edit must not
        // appear to succeed while the authoritative prepared law ignores it.
        if expanded.prepared_resistors.contains_key(&device.name)
            || (expanded
                .prepared_waveforms
                .iter()
                .any(|(n, _)| n == &device.name)
                && !matches!(setter.name.as_str(), "acmag" | "acphase" | "m"))
            || (setter.name == "ic"
                && expanded
                    .prepared_initialization
                    .devices
                    .iter()
                    .any(|(n, _)| n == &device.name))
        {
            return Err(SpiceError::circuit(
                "candidate edit requires a source-prepared law binding",
            ));
        }
        device.parameters.push(setter);
    }
    Ok(sources)
}

pub(super) fn apply_sources(
    circuit: &mut Circuit,
    context: &ModelContext,
    sources: &[(&str, Real)],
) -> SpiceResult<()> {
    if sources.is_empty() {
        return Ok(());
    }
    let overrides = sources
        .iter()
        .map(|(name, value)| {
            circuit
                .source_override(name, *value, context)
                .map(CandidateNumericOverride::Instance)
        })
        .collect::<SpiceResult<Vec<_>>>()?;
    circuit.commit_candidate_overrides(context, overrides)
}
