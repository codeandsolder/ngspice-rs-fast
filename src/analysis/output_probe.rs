//! Shared, validated circuit-row binding for the `.tf` and `.sens` output
//! languages. This is a linear measurement functional, not the typed device
//! parameter observation layer used by other analyses.

use std::iter::Peekable;

use crate::devices::Circuit;
use crate::primitives::{NodeKind, NodeTable, SpiceError, SpiceResult};

/// The output expression as written in either analysis card.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum Output {
    Voltage { pos: String, neg: Option<String> },
    Current { source: String },
}

/// A copyable linear measurement: the external observation is
/// `x[pos] - x[neg]` or one branch-current row, in SPICE's sign convention.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Probe {
    Voltage {
        pos: Option<usize>,
        neg: Option<usize>,
    },
    Current {
        row: usize,
    },
}

/// A one-time validated binding, keeping canonical names and device index
/// for the transfer-function output-impedance and same-source rules.
#[derive(Debug)]
pub(super) struct BoundOutput {
    pub probe: Probe,
    pub names: [Option<String>; 2],
    pub device: Option<usize>,
}

/// Preserve analysis-specific diagnostics, despite sharing lookup semantics.
#[derive(Debug, Clone, Copy)]
pub(super) enum Card {
    Tf,
    Sens,
}

impl Card {
    const fn keyword(self) -> &'static str {
        match self {
            Self::Tf => ".tf",
            Self::Sens => ".sens",
        }
    }

    const fn reference(self) -> &'static str {
        match self {
            Self::Tf => "C: TFanal in tfanal.c",
            Self::Sens => "C: sens_sens in cktsens.c",
        }
    }

    fn not_found(self, message: impl std::fmt::Display) -> SpiceError {
        SpiceError::Circuit {
            message: format!("{message} ({})", self.reference()),
        }
    }
}

/// Parse the common `v(pos[,neg])` and `i(source)` prefix. The caller owns
/// the cursor and card-specific remaining arguments, so error order and
/// admissibility rules remain those of the containing analysis.
///
/// # Errors
/// The caller's syntax error for a missing/invalid token or output form.
pub(super) fn parse_output<'a>(
    tokens: &mut Peekable<impl Iterator<Item = &'a str>>,
    syntax: impl Fn(String) -> SpiceError,
) -> SpiceResult<Output> {
    let name = |token: Option<&str>, what: &str| -> SpiceResult<String> {
        match token {
            Some(text) if !matches!(text, "(" | ")" | ",") => Ok(text.to_ascii_lowercase()),
            Some(text) => Err(syntax(format!("expected {what}, found '{text}'"))),
            None => Err(syntax(format!("is missing {what}"))),
        }
    };
    let expect = |wanted: &str, token: Option<&str>| -> SpiceResult<()> {
        match token {
            Some(text) if text == wanted => Ok(()),
            Some(text) => Err(syntax(format!("expected '{wanted}', found '{text}'"))),
            None => Err(syntax(format!("is missing '{wanted}'"))),
        }
    };
    let kind = tokens
        .next()
        .ok_or_else(|| syntax("has no output variable".to_owned()))?
        .to_ascii_lowercase();
    match kind.as_str() {
        "v" => {
            expect("(", tokens.next())?;
            let pos = name(tokens.next(), "an output node")?;
            if tokens.peek().is_some_and(|text| *text == ",") {
                tokens.next();
            }
            let neg = if tokens.peek().is_some_and(|text| *text == ")") {
                None
            } else {
                Some(name(tokens.next(), "the negative output node")?)
            };
            expect(")", tokens.next())?;
            Ok(Output::Voltage { pos, neg })
        }
        "i" => {
            expect("(", tokens.next())?;
            let source = name(tokens.next(), "an output source")?;
            expect(")", tokens.next())?;
            Ok(Output::Current { source })
        }
        other => Err(syntax(format!(
            "output '{other}' is neither a voltage v(...) nor a current i(...)"
        ))),
    }
}

pub(super) fn device_index(circuit: &Circuit, name: &str) -> Option<usize> {
    circuit
        .devices()
        .iter()
        .position(|device| device.name().eq_ignore_ascii_case(name))
}

fn node_row(circuit: &Circuit, card: Card, name: &str) -> SpiceResult<(String, Option<usize>)> {
    let canonical = NodeTable::canonical_name(name, circuit.nodes().auto_gnd());
    let id = circuit.nodes().get(&canonical).ok_or_else(|| {
        card.not_found(format_args!(
            "{} output node {name} is not in the circuit",
            card.keyword()
        ))
    })?;
    let row = circuit.unknowns().node_row(id);
    let ground = circuit
        .nodes()
        .node(id)
        .is_some_and(|node| node.kind == NodeKind::Ground);
    if row.is_none() && !ground {
        return Err(card.not_found(format_args!(
            "{} output node {name} has no matrix row",
            card.keyword()
        )));
    }
    Ok((canonical, row))
}

/// Bind the linear output after the circuit is finalized. Valid current
/// outputs require the same CKTfndBranch-admissible device families and a
/// real branch row; no unknown node aliases are silently treated as ground.
///
/// # Errors
/// Unknown nodes/devices, non-ground nodes without rows, non-findable
/// branch currents and missing branch rows, all with card-specific context.
pub(super) fn bind(circuit: &Circuit, output: &Output, card: Card) -> SpiceResult<BoundOutput> {
    match output {
        Output::Voltage { pos, neg } => {
            let (pos_name, pos_row) = node_row(circuit, card, pos)?;
            let (neg_name, neg_row) = match neg {
                Some(neg) => {
                    let (canonical, row) = node_row(circuit, card, neg)?;
                    (Some(canonical), row)
                }
                None => (None, None),
            };
            Ok(BoundOutput {
                probe: Probe::Voltage {
                    pos: pos_row,
                    neg: neg_row,
                },
                names: [Some(pos_name), neg_name],
                device: None,
            })
        }
        Output::Current { source } => {
            let device = device_index(circuit, source).ok_or_else(|| {
                card.not_found(format_args!(
                    "{} output source {source} is not in the circuit",
                    card.keyword()
                ))
            })?;
            let branch = circuit.devices()[device].findable_branch().ok_or_else(|| {
                crate::analysis::linear::unsupported(format!(
                    "{} i({source}): the device has no findable branch current (only \
                     independent voltage sources, E/H sources and voltage B sources can be \
                     sensed, as by CKTfndBranch; C would silently read ground)",
                    card.keyword()
                ))
            })?;
            let row = circuit
                .branch_rows(device)
                .and_then(|mut rows| rows.nth(branch))
                .ok_or_else(|| card.not_found(format_args!("{source}: missing branch row")))?;
            Ok(BoundOutput {
                probe: Probe::Current { row },
                names: [None, None],
                device: Some(device),
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::netlist::{Parser, source::parse_deck_text};
    use std::path::Path;

    fn circuit(body: &str) -> Circuit {
        let text = format!("shared output probe\n{body}\n.end\n");
        let deck = Parser::new()
            .parse_deck(&parse_deck_text(Path::new("output-probes.cir"), &text))
            .unwrap();
        Circuit::from_netlist(&deck).unwrap()
    }

    fn voltage(pos: &str, neg: Option<&str>) -> Output {
        Output::Voltage {
            pos: pos.to_owned(),
            neg: neg.map(str::to_owned),
        }
    }

    #[test]
    fn both_cards_bind_identical_rows_and_canonical_ground_aliases() {
        let mut circuit = circuit("v1 IN 0 dc 1\nr1 in OUT 1k\nr2 out 0 2k\nvm out 0 0");
        circuit.finalize().unwrap();
        let measured = bind(&circuit, &voltage("OUT", Some("GND")), Card::Tf).unwrap();
        let sensitivity = bind(&circuit, &voltage("out", Some("0")), Card::Sens).unwrap();
        assert_eq!(measured.probe, sensitivity.probe);
        assert_eq!(
            measured.names,
            [Some("out".to_owned()), Some("0".to_owned())]
        );
        assert_eq!(sensitivity.names, measured.names);
        assert!(matches!(
            measured.probe,
            Probe::Voltage {
                pos: Some(_),
                neg: None
            }
        ));
        let default = bind(&circuit, &voltage("out", None), Card::Tf).unwrap();
        assert_eq!(default.probe, measured.probe);
        assert_eq!(default.names, [Some("out".to_owned()), None]);
        let explicit = bind(&circuit, &voltage("out", Some("0")), Card::Tf).unwrap();
        assert_eq!(explicit.names, measured.names);
        let branch = Output::Current {
            source: "VM".into(),
        };
        let tf = bind(&circuit, &branch, Card::Tf).unwrap();
        let sens = bind(&circuit, &branch, Card::Sens).unwrap();
        assert_eq!(tf.probe, sens.probe);
        assert_eq!(tf.device, sens.device);
        assert!(matches!(tf.probe, Probe::Current { .. }));
        assert_eq!(tf.names, [None, None]);
    }

    #[test]
    fn device_ordinals_track_the_actual_circuit_layout() {
        let mut first = circuit("v1 in 0 1\nr1 in out 1k\nvm out 0 0\nr2 out 0 2k");
        let mut reordered = circuit("r2 out 0 2k\nvm out 0 0\nr1 in out 1k\nv1 in 0 1");
        first.finalize().unwrap();
        reordered.finalize().unwrap();
        let expression = Output::Current {
            source: "vM".into(),
        };
        let a = bind(&first, &expression, Card::Tf).unwrap();
        let b = bind(&reordered, &expression, Card::Tf).unwrap();
        assert_ne!(a.device, b.device);
        for (circuit, result) in [(&first, a), (&reordered, b)] {
            let device = result.device.unwrap();
            let row = circuit.branch_rows(device).unwrap().next().unwrap();
            assert_eq!(result.probe, Probe::Current { row });
        }
    }

    #[test]
    fn invalid_outputs_keep_exact_card_context_and_no_false_ground() {
        let mut circuit = circuit("v1 in 0 1\nr1 in out 1k\nr2 out 0 2k");
        circuit.finalize().unwrap();
        for card in [Card::Tf, Card::Sens] {
            let label = card.keyword();
            let reference = card.reference();
            let missing = bind(&circuit, &voltage("NOPE", None), card).unwrap_err();
            assert_eq!(
                missing.to_string(),
                format!("circuit: {label} output node NOPE is not in the circuit ({reference})")
            );
            let absent_current = bind(
                &circuit,
                &Output::Current {
                    source: "vx".into(),
                },
                card,
            )
            .unwrap_err();
            assert_eq!(
                absent_current.to_string(),
                format!("circuit: {label} output source vx is not in the circuit ({reference})")
            );
            let rejected = bind(
                &circuit,
                &Output::Current {
                    source: "r1".into(),
                },
                card,
            )
            .unwrap_err()
            .to_string();
            assert!(rejected.contains(&format!("{label} i(r1)")));
            assert!(rejected.contains("no findable branch current"));
        }
    }

    #[test]
    fn prepared_rows_are_used_without_guessing_ground_or_branch_numbers() {
        // from_netlist already establishes a valid circuit numbering. The
        // binder must trust the actual node and branch maps, regardless of
        // whether the caller has redundantly called finalize().
        let circuit = circuit("v1 in 0 1\nr1 in out 1k\nr2 out 0 2k");
        for card in [Card::Tf, Card::Sens] {
            let node = bind(&circuit, &voltage("out", None), card).unwrap();
            assert!(matches!(
                node.probe,
                Probe::Voltage {
                    pos: Some(_),
                    neg: None
                }
            ));
            let ground = bind(&circuit, &voltage("out", Some("0")), card).unwrap();
            assert_eq!(ground.probe, node.probe);
            let branch = bind(
                &circuit,
                &Output::Current {
                    source: "v1".into(),
                },
                card,
            )
            .unwrap();
            let row = circuit
                .branch_rows(branch.device.unwrap())
                .unwrap()
                .next()
                .unwrap();
            assert_eq!(branch.probe, Probe::Current { row });
        }
    }

    #[test]
    fn parser_consumes_only_the_bounded_output_prefix() {
        for card in [Card::Tf, Card::Sens] {
            let syntax = |message| SpiceError::Circuit {
                message: format!("{} {message}", card.keyword()),
            };
            let mut tokens = ["V", "(", "OuT", ",", "GND", ")", "tail"]
                .into_iter()
                .peekable();
            let output = parse_output(&mut tokens, syntax).unwrap();
            assert_eq!(output, voltage("out", Some("gnd")));
            assert_eq!(tokens.next(), Some("tail"));
            for tokens in [
                vec!["i", "(", ")"],
                vec!["v", "out"],
                vec!["v", "(", "out", ",", ",", ")"],
            ] {
                let mut iter = tokens.into_iter().peekable();
                assert!(parse_output(&mut iter, syntax).is_err());
            }
        }
    }
}
