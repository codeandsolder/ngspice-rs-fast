//! Literalization: replace evaluated numeric sites with finite scalars.
//!
//! [`literalize`] resolves the top-level `.param` cards ([`ParamScope`]) and
//! produces a *copy* of the [`Netlist`] in which every top-level
//! [`ParameterKind::Expression`] device/model parameter and every braced
//! analysis argument is replaced by its finite value, spelled as numparam
//! spells it (`"% 23.15e"`, see [`format_literal`]), so reading it back gives
//! ngspice's value. The input AST is
//! never modified; the copy keeps `.param` cards, locations and card order. The
//! original spelling and location of each replaced site are kept in
//! [`ElaboratedNetlist::sites`].
//!
//! Top-level `.ic`/`.nodeset` expression values are literalized too (the
//! ordered entries are available from [`ElaboratedNetlist::initial_conditions`]
//! and [`ElaboratedNetlist::nodesets`]).
//!
//! Subcircuit bodies are left untouched here (their expressions need formal
//! binding, which happens during expansion in `crate::devices::subckt`; this
//! pass does not reject subcircuit decks).
//! Bare names in device cards are not references (see
//! `docs/port/PARAM_EXPRESSIONS.md`), so terminals and model names are never
//! substituted.

use std::sync::Arc;

use crate::primitives::{Real, SourceLoc, SpiceError, SpiceResult};

use crate::netlist::ast::{
    AnalysisCard, DeviceInstance, ModelCard, Netlist, NodeHint, NodeHintCard, NodeHintValue,
    ParameterAssignment, ParameterKind, Subcircuit,
};
use crate::netlist::eval::{EvalBudget, FunctionScope, ParamScope};

/// Which site was literalized (indexes into the *input* netlist).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SiteKind {
    /// `Netlist::devices[device].parameters[parameter]`.
    DeviceParameter {
        /// Device index.
        device: usize,
        /// Parameter index.
        parameter: usize,
    },
    /// `Netlist::models[model].parameters[parameter]`.
    ModelParameter {
        /// Model index.
        model: usize,
        /// Parameter index.
        parameter: usize,
    },
    /// `Netlist::analyses[analysis].arguments[argument]`.
    AnalysisArgument {
        /// Analysis index.
        analysis: usize,
        /// Argument index.
        argument: usize,
    },
    /// `Netlist::initial_conditions[card].entries[entry]`.
    InitialCondition {
        /// `.ic` card index.
        card: usize,
        /// Entry index within the card.
        entry: usize,
    },
    /// `Netlist::nodesets[card].entries[entry]`.
    Nodeset {
        /// `.nodeset` card index.
        card: usize,
        /// Entry index within the card.
        entry: usize,
    },
}

/// One evaluated site with its original text.
#[derive(Debug, Clone, PartialEq)]
pub struct ResolvedSite {
    /// Where in the input netlist.
    pub kind: SiteKind,
    /// Original spelling, braces included when written.
    pub original: String,
    /// Location of the expression text.
    pub location: SourceLoc,
    /// The evaluated finite value before numparam formatting.
    pub value: Real,
    /// The literal inserted into the card by numparam.
    pub formatted: String,
    /// Correctly rounded read-back of the formatted literal, which may
    /// differ from the original evaluation because of decimal formatting.
    pub read_back: Real,
}

impl ResolvedSite {
    fn record(
        kind: SiteKind,
        original: String,
        location: SourceLoc,
        value: Real,
        formatted: &str,
    ) -> SpiceResult<Self> {
        let read_back = crate::primitives::parse_spice_number(formatted).ok_or_else(|| {
            SpiceError::parse(
                location.clone(),
                "formatted numeric value cannot be read back",
            )
        })?;
        Ok(Self {
            kind,
            original,
            location,
            value,
            formatted: formatted.to_owned(),
            read_back,
        })
    }
}

/// A literalized copy plus the evaluation record.
#[derive(Debug, Clone, PartialEq)]
pub struct ElaboratedNetlist {
    /// The netlist copy with evaluated sites replaced by scalars.
    pub netlist: Netlist,
    /// The resolved top-level parameter scope.
    pub scope: Arc<ParamScope>,
    /// Every replaced site in deck order.
    pub sites: Vec<ResolvedSite>,
}

/// One `.ic`/`.nodeset` entry with a finite value (see
/// [`ElaboratedNetlist::initial_conditions`]).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ResolvedNodeHint<'a> {
    /// Canonical node name.
    pub node: &'a str,
    /// Finite value in volts.
    pub value: Real,
    /// Where the entry (`V(node)=value`) started.
    pub location: &'a SourceLoc,
}

impl ElaboratedNetlist {
    /// Ordered `.ic` entries `(node, value, location)`, duplicates preserved
    /// (card order, then entry order). Every expression was evaluated by
    /// [`literalize`], so values are finite. Syntax only: no precedence among
    /// duplicates, `uic`, instance `ic=` or DC bias is applied here.
    #[must_use = "the returned entries carry resolved initial-condition values and order"]
    pub fn initial_conditions(&self) -> Vec<ResolvedNodeHint<'_>> {
        resolved(self.netlist.initial_conditions())
    }

    /// Ordered `.nodeset` entries, as [`Self::initial_conditions`]. A nodeset
    /// is a convergence hint, not a constraint.
    #[must_use = "the returned entries carry resolved convergence hints in declaration order"]
    pub fn nodesets(&self) -> Vec<ResolvedNodeHint<'_>> {
        resolved(self.netlist.nodesets())
    }
}

fn resolved<'a>(entries: impl Iterator<Item = &'a NodeHint>) -> Vec<ResolvedNodeHint<'a>> {
    // `literalize` replaces every expression, so `literal()` is always `Some`
    // for an elaborated netlist; an unevaluated entry is never reported with a
    // made-up value.
    entries
        .filter_map(|entry| {
            Some(ResolvedNodeHint {
                node: &entry.node,
                value: entry.literal()?,
                location: &entry.location,
            })
        })
        .collect()
}

/// Format an evaluated parameter for insertion back into a netlist.
///
/// Spell an evaluated value as numparam splices it back into the card
/// (`double_to_string`, `xpressn.c`: `"% 23.15e"`, without the padding).
/// Reading the text back with [`crate::primitives::parse_spice_number`]
/// correctly rounds the resulting decimal to binary64, which may differ
/// from `value` because of the formatting precision; integer arguments use C's
/// `floor(0.5 + x)` ([`crate::primitives::c_integer`]).
#[must_use = "the returned literal is the canonical emitted numerical spelling"]
pub fn format_literal(value: Real) -> String {
    crate::primitives::format_spice_number(value)
}

/// Resolve `.param` cards and literalize all top-level sites with default limits.
///
/// # Errors
/// Any parameter or site evaluation failure (undefined, cyclic, domain,
/// non-finite, over budget), with source locations.
pub fn literalize(netlist: &Netlist) -> SpiceResult<ElaboratedNetlist> {
    let mut budget = EvalBudget::for_netlist(netlist)?;
    let functions = FunctionScope::for_netlist(netlist, &budget)?;
    let scope = Arc::new(ParamScope::resolve_scoped(
        None,
        Some(functions),
        &[],
        &netlist.params,
        &mut budget,
    )?);
    literalize_with(netlist, scope, &mut budget)
}

/// Literalize against an already resolved scope.
///
/// # Errors
/// As [`literalize`].
pub fn literalize_with(
    netlist: &Netlist,
    scope: Arc<ParamScope>,
    budget: &mut EvalBudget,
) -> SpiceResult<ElaboratedNetlist> {
    let mut out = netlist.clone();
    let mut sites = Vec::new();
    literalize_device_models(
        &mut out.devices,
        &mut out.models,
        &scope,
        budget,
        Some(&mut sites),
    )?;
    for (ai, (before, card)) in netlist.analyses.iter().zip(&mut out.analyses).enumerate() {
        literalize_analysis_into(before, card, &scope, budget, Some((&mut sites, ai)))?;
    }
    literalize_hints(
        &mut out.initial_conditions,
        &scope,
        budget,
        Some(&mut sites),
        |c, e| SiteKind::InitialCondition { card: c, entry: e },
    )?;
    literalize_hints(
        &mut out.nodesets,
        &scope,
        budget,
        Some(&mut sites),
        |c, e| SiteKind::Nodeset { card: c, entry: e },
    )?;
    Ok(ElaboratedNetlist {
        netlist: out,
        scope,
        sites,
    })
}

fn literalize_device_models(
    devices: &mut [DeviceInstance],
    models: &mut [ModelCard],
    scope: &ParamScope,
    budget: &mut EvalBudget,
    mut sites: Option<&mut Vec<ResolvedSite>>,
) -> SpiceResult<()> {
    for (di, device) in devices.iter_mut().enumerate() {
        for parameter in &mut device.parameters {
            if matches!(
                (device.designator, parameter.name.as_str()),
                ('r', "resistance") | ('c', "capacitance" | "charge") | ('l', "inductance")
            ) {
                crate::netlist::behavioural::classify_passive(parameter, scope, budget)?;
            }
        }
        literalize_parameters(
            &mut device.parameters,
            scope,
            budget,
            sites.as_deref_mut(),
            |p| SiteKind::DeviceParameter {
                device: di,
                parameter: p,
            },
        )?;
    }
    for (mi, model) in models.iter_mut().enumerate() {
        literalize_parameters(
            &mut model.parameters,
            scope,
            budget,
            sites.as_deref_mut(),
            |p| SiteKind::ModelParameter {
                model: mi,
                parameter: p,
            },
        )?;
    }
    Ok(())
}

/// Instance-scoped semantic values, without copying lexical declarations or
/// creating a synthetic root deck.
pub(crate) struct LiteralizedBody {
    pub(crate) devices: Vec<DeviceInstance>,
    pub(crate) models: Vec<ModelCard>,
    pub(crate) initial_conditions: Vec<NodeHintCard>,
    pub(crate) nodesets: Vec<NodeHintCard>,
}

pub(crate) fn literalize_body(
    body: &Subcircuit,
    scope: &ParamScope,
    budget: &mut EvalBudget,
) -> SpiceResult<LiteralizedBody> {
    let mut values = LiteralizedBody {
        devices: body.devices.clone(),
        models: body.models.clone(),
        initial_conditions: body.initial_conditions.clone(),
        nodesets: body.nodesets.clone(),
    };
    literalize_device_models(&mut values.devices, &mut values.models, scope, budget, None)?;
    literalize_hints(
        &mut values.initial_conditions,
        scope,
        budget,
        None,
        |c, e| SiteKind::InitialCondition { card: c, entry: e },
    )?;
    literalize_hints(&mut values.nodesets, scope, budget, None, |c, e| {
        SiteKind::Nodeset { card: c, entry: e }
    })?;
    Ok(values)
}

/// Evaluate .ic and .nodeset entries without elaborating the whole deck.
///
/// Evaluates only the `.ic` and `.nodeset` entries against `scope`, returning
/// literalized copies of those cards (`.ic` first, `.nodeset` second) without
/// elaborating the rest of the deck. Order and duplicates are preserved.
///
/// # Errors
/// Any entry evaluation failure, located at the expression.
pub fn literalize_node_hints(
    netlist: &Netlist,
    scope: &ParamScope,
    budget: &mut EvalBudget,
) -> SpiceResult<(Vec<NodeHintCard>, Vec<NodeHintCard>)> {
    let mut initial = netlist.initial_conditions.clone();
    let mut nodesets = netlist.nodesets.clone();
    literalize_hints(&mut initial, scope, budget, None, |c, e| {
        SiteKind::InitialCondition { card: c, entry: e }
    })?;
    literalize_hints(&mut nodesets, scope, budget, None, |c, e| {
        SiteKind::Nodeset { card: c, entry: e }
    })?;
    Ok((initial, nodesets))
}

fn literalize_hints(
    cards: &mut [NodeHintCard],
    scope: &ParamScope,
    budget: &mut EvalBudget,
    mut sites: Option<&mut Vec<ResolvedSite>>,
    kind: impl Fn(usize, usize) -> SiteKind,
) -> SpiceResult<()> {
    for (ci, card) in cards.iter_mut().enumerate() {
        for (ei, hint) in card.entries.iter_mut().enumerate() {
            let NodeHintValue::Expression(expression) = &hint.value else {
                continue;
            };
            let value = scope
                .evaluate(expression, budget)
                .map_err(|error| match error {
                    SpiceError::Parse { location, message } => SpiceError::parse(
                        location,
                        format!(
                            "{message}\n  while evaluating the value of V({}) at {}",
                            hint.node, hint.location
                        ),
                    ),
                    other => other,
                })?;
            // Numparam's text, read back with correct rounding as the card is.
            let text = format_literal(value);
            let read_back = if let Some(sites) = sites.as_deref_mut() {
                let site = ResolvedSite::record(
                    kind(ci, ei),
                    if expression.braced {
                        format!("{{{}}}", expression.text)
                    } else {
                        expression.text.clone()
                    },
                    expression.span.start.clone(),
                    value,
                    &text,
                )?;
                let read_back = site.read_back;
                sites.push(site);
                read_back
            } else {
                crate::primitives::parse_spice_number(&text).unwrap_or(value)
            };
            hint.value = NodeHintValue::Literal {
                value: read_back,
                text,
            };
        }
    }
    Ok(())
}

fn literalize_parameters(
    parameters: &mut [ParameterAssignment],
    scope: &ParamScope,
    budget: &mut EvalBudget,
    mut sites: Option<&mut Vec<ResolvedSite>>,
    kind: impl Fn(usize) -> SiteKind,
) -> SpiceResult<()> {
    for (index, parameter) in parameters.iter_mut().enumerate() {
        if let ParameterKind::NumericVector(entries) = &mut parameter.kind {
            for entry in entries.iter_mut() {
                if let Some(expression) = entry.expression.take() {
                    let value =
                        scope
                            .evaluate(&expression, budget)
                            .map_err(|error| match error {
                                SpiceError::Parse { location, message } => SpiceError::parse(
                                    location,
                                    format!(
                                        "{message} while resolving {} vector coefficient at {}",
                                        parameter.name, entry.source.location
                                    ),
                                ),
                                other => other,
                            })?;
                    entry.source.text = format_literal(value);
                }
            }
            parameter.value = entries
                .iter()
                .map(|entry| entry.source.text.as_str())
                .collect::<Vec<_>>()
                .join(" ");
            continue;
        }
        if let ParameterKind::Behavioural(expression) = &parameter.kind {
            // Behavioural expressions stay symbolic: only their `.param`
            // names, `.func` calls and numparam values are resolved here.
            let resolved = crate::netlist::behavioural::resolve_expression(
                expression, scope, budget,
            )
            .map_err(|error| match error {
                SpiceError::Parse { location, message } => SpiceError::parse(
                    location,
                    format!(
                        "{message}\n  while resolving behavioural expression '{}' = {} at {}",
                        parameter.name, parameter.value, parameter.location
                    ),
                ),
                other => other,
            })?;
            parameter.kind = ParameterKind::Behavioural(Box::new(resolved));
            continue;
        }
        let ParameterKind::Expression(expression) = &parameter.kind else {
            continue;
        };
        let value = scope
            .evaluate(expression, budget)
            .map_err(|error| match error {
                SpiceError::Parse { location, message } => SpiceError::parse(
                    location,
                    format!(
                        "{message}\n  while evaluating parameter '{}' = {} at {}",
                        parameter.name, parameter.value, parameter.location
                    ),
                ),
                other => other,
            })?;
        let text = format_literal(value);
        if let Some(sites) = sites.as_deref_mut() {
            sites.push(ResolvedSite::record(
                kind(index),
                parameter.value.clone(),
                expression.span.start.clone(),
                value,
                &text,
            )?);
        }
        parameter.value = text;
        parameter.kind = ParameterKind::Scalar;
    }
    Ok(())
}

/// Replace the braced arguments of one analysis card with their values.
///
/// # Errors
/// Evaluation failures of the argument expressions.
pub fn literalize_analysis(
    card: &AnalysisCard,
    scope: &ParamScope,
    budget: &mut EvalBudget,
) -> SpiceResult<AnalysisCard> {
    let mut out = card.clone();
    literalize_analysis_into(card, &mut out, scope, budget, None)?;
    Ok(out)
}

/// Resolve into an existing copy of the input card so whole-deck
/// literalization does not clone the expression trees a second time.
fn literalize_analysis_into(
    card: &AnalysisCard,
    out: &mut AnalysisCard,
    scope: &ParamScope,
    budget: &mut EvalBudget,
    mut sites: Option<(&mut Vec<ResolvedSite>, usize)>,
) -> SpiceResult<()> {
    // Validate the whole index set before evaluation. A constructed card
    // must not silently overwrite an earlier resolved argument.
    let mut seen = vec![false; card.arguments.len()];
    for e in &card.expressions {
        let position = seen.get_mut(e.index).ok_or_else(|| {
            SpiceError::parse(
                e.expression.span.start.clone(),
                "analysis argument expression index is out of range",
            )
        })?;
        if std::mem::replace(position, true) {
            return Err(SpiceError::parse(
                e.expression.span.start.clone(),
                "duplicate analysis argument expression index",
            ));
        }
    }
    for e in &card.expressions {
        let argument = out.arguments.get_mut(e.index).ok_or_else(|| {
            SpiceError::parse(
                e.expression.span.start.clone(),
                "analysis argument expression index is out of range",
            )
        })?;
        let value = scope
            .evaluate(&e.expression, budget)
            .map_err(|error| match error {
                SpiceError::Parse { location, message } => SpiceError::parse(
                    location,
                    format!(
                        "{message}\n  while evaluating argument {} of the {:?} card at {}",
                        e.index + 1,
                        card.kind,
                        card.location
                    ),
                ),
                other => other,
            })?;
        let formatted = format_literal(value);
        if let Some((sites, analysis)) = &mut sites {
            sites.push(ResolvedSite::record(
                SiteKind::AnalysisArgument {
                    analysis: *analysis,
                    argument: e.index,
                },
                argument.clone(),
                e.expression.span.start.clone(),
                value,
                &formatted,
            )?);
        }
        *argument = formatted;
    }
    out.expressions.clear();
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::literalize;
    use crate::netlist::{Parser, source::parse_deck_text};
    use std::path::Path;

    #[test]
    fn malformed_analysis_expression_index_is_a_positioned_error() {
        let deck = parse_deck_text(
            Path::new("invalid-index.cir"),
            "Title\n.tran {1u} 2u\n.end\n",
        );
        let mut netlist = Parser::new().parse_deck(&deck).unwrap();
        netlist.analyses[0].expressions[0].index = usize::MAX;
        let error = literalize(&netlist).expect_err("out-of-range expression index");
        assert!(
            error
                .to_string()
                .contains("analysis argument expression index is out of range")
        );
        assert!(error.to_string().contains("invalid-index.cir:2:"));
    }
}
