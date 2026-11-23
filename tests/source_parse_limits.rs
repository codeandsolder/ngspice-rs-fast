//! Shared source admission and lexical work budgets (#129).
use ngspice_rs::netlist::parser::{classify_deck_with_limits, load_classified_with_limits};
use ngspice_rs::netlist::source::parse_deck_text;
use ngspice_rs::netlist::{Parser, SourceLimits};
use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

struct Scratch(PathBuf);

impl Scratch {
    fn new() -> Self {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let path = std::env::temp_dir().join(format!(
            "spice-lexical-limits-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&path).unwrap();
        Self(path)
    }

    fn write(&self, name: &str, text: &str) -> PathBuf {
        let path = self.0.join(name);
        fs::write(&path, text).unwrap();
        path
    }
}
impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn limits_for(text: &str) -> SourceLimits {
    let deck = parse_deck_text(Path::new("limits.cir"), text);
    SourceLimits {
        files: 1,
        depth: 0,
        bytes: text.len(),
        physical_lines: text.lines().count(),
        cards: deck.len(),
        folded_cards: deck.len(),
        card_bytes: deck
            .lines
            .iter()
            .map(|line| line.text.len())
            .chain([deck.title.len()])
            .max()
            .unwrap(),
        tokens: ngspice_rs::netlist::classify_deck(&deck)
            .unwrap()
            .iter()
            .map(|card| card.tokens.len())
            .sum(),
    }
}

fn smaller(limits: SourceLimits) -> [(SourceLimits, &'static str); 7] {
    [
        (
            SourceLimits {
                files: limits.files - 1,
                ..limits
            },
            "file work",
        ),
        (
            SourceLimits {
                bytes: limits.bytes - 1,
                ..limits
            },
            "byte work",
        ),
        (
            SourceLimits {
                physical_lines: limits.physical_lines - 1,
                ..limits
            },
            "physical line work",
        ),
        (
            SourceLimits {
                folded_cards: limits.folded_cards - 1,
                ..limits
            },
            "folded card work",
        ),
        (
            SourceLimits {
                card_bytes: limits.card_bytes - 1,
                ..limits
            },
            "logical card byte",
        ),
        (
            SourceLimits {
                cards: limits.cards - 1,
                ..limits
            },
            "card work",
        ),
        (
            SourceLimits {
                tokens: limits.tokens - 1,
                ..limits
            },
            "token work",
        ),
    ]
}

#[test]
fn borrowed_text_and_file_have_identical_exact_limits() {
    let scratch = Scratch::new();
    let text = "Title\nr1 a 0\n+ 1k tc1=0.1\n.end\n";
    let path = scratch.write("limits.cir", text);
    let limits = limits_for(text);
    let memory = Parser::new().parse_text(&path, text, limits).unwrap();
    let file = Parser::new()
        .parse_file_with_limits_and_output(&path, limits)
        .unwrap();
    assert_eq!(memory, file);
    assert_eq!(file.netlist.devices.len(), 1);
    assert_eq!(limits.tokens, 8);
    assert_eq!(limits.physical_lines, 4);
    assert_eq!(limits.folded_cards, 2);
    let (deck, cards) = load_classified_with_limits(&path, limits).unwrap();
    assert_eq!(cards, classify_deck_with_limits(&deck, limits).unwrap());
    for (smaller, label) in smaller(limits) {
        for error in [
            Parser::new().parse_text(&path, text, smaller).unwrap_err(),
            Parser::new()
                .parse_file_with_limits(&path, smaller)
                .unwrap_err(),
            load_classified_with_limits(&path, smaller).unwrap_err(),
        ] {
            assert!(error.to_string().contains(label), "{label}: {error}");
        }
    }
}

#[test]
fn caller_owned_deck_charges_visible_storage_and_does_not_trust_edited_continuations() {
    let text = "Title\nr1 a 0\n+ 1k\n+ tc1=0.1\n.end\n";
    let mut deck = parse_deck_text(Path::new("limits.cir"), text);
    let limits = SourceLimits {
        bytes: deck.title.len() + deck.lines.iter().map(|line| line.text.len()).sum::<usize>(),
        ..limits_for(text)
    };
    deck.lines[0].continuations = 0;
    assert_eq!(
        Parser::new()
            .parse_deck_with_limits_and_output(&deck, limits)
            .unwrap()
            .netlist
            .devices
            .len(),
        1
    );
    assert_eq!(classify_deck_with_limits(&deck, limits).unwrap().len(), 2);
    for (smaller, label) in smaller(limits) {
        for error in [
            Parser::new()
                .parse_deck_with_limits_and_output(&deck, smaller)
                .unwrap_err(),
            classify_deck_with_limits(&deck, smaller).unwrap_err(),
        ] {
            assert!(error.to_string().contains(label), "{label}: {error}");
        }
    }
}

#[test]
fn lazy_library_sections_charge_storage_but_do_not_lex_unselected_ordinaries() {
    let scratch = Scratch::new();
    let text = "title\n.lib parts.lib used\n.lib parts.lib used\n.end\n";
    let root = scratch.write("main.cir", text);
    // The ignored card is deliberately malformed: eager tokenization fails
    // either syntax or the very small aggregate token budget.
    let library = format!(
        ".lib used\nr1 a 0 1k\n.endl used\n.lib unused\n{}{{\n.endl unused\n",
        ",".repeat(131_072)
    );
    scratch.write("parts.lib", &library);
    let limits = SourceLimits {
        depth: 1,
        files: 3,
        bytes: text.len() + 2 * library.len(),
        physical_lines: 10,
        folded_cards: 9,
        cards: 5,
        card_bytes: 131_073,
        tokens: 23,
    };
    let parsed = Parser::new().parse_file_with_limits(&root, limits).unwrap();
    assert_eq!(parsed.devices.len(), 2);
    assert!(Arc::ptr_eq(
        &parsed.cards[1].source,
        &parsed.cards[3].source
    ));
    // Replayed r1 uses four logical tokens even though its Arc is shared.
    for (smaller, label) in [
        (
            SourceLimits {
                tokens: 22,
                ..limits
            },
            "token work",
        ),
        (
            SourceLimits {
                physical_lines: 9,
                ..limits
            },
            "physical line work",
        ),
        (
            SourceLimits {
                folded_cards: 8,
                ..limits
            },
            "folded card work",
        ),
        (
            SourceLimits {
                card_bytes: 131_072,
                ..limits
            },
            "logical card byte",
        ),
        (
            SourceLimits {
                bytes: limits.bytes - 1,
                ..limits
            },
            "byte work",
        ),
    ] {
        let error = Parser::new()
            .parse_file_with_limits(&root, smaller)
            .unwrap_err();
        assert!(error.to_string().contains(label), "{label}: {error}");
    }
}

#[test]
fn late_lexical_exhaustion_does_not_mask_earlier_semantic_error() {
    let scratch = Scratch::new();
    let text = format!("title\n.width 80\n{}\n.end\n", ",".repeat(100_000));
    let path = scratch.write("bad.cir", &text);
    let deck = parse_deck_text(&path, &text);
    let limits = SourceLimits {
        tokens: 3,
        ..SourceLimits::default()
    };
    for error in [
        Parser::new().parse_text(&path, &text, limits).unwrap_err(),
        Parser::new()
            .parse_deck_with_limits_and_output(&deck, limits)
            .unwrap_err(),
        Parser::new()
            .parse_file_with_limits(&path, limits)
            .unwrap_err(),
    ] {
        assert!(error.is_not_yet_ported(), "{error}");
    }
    let text = "title\n.end\n,,,{\n";
    let path = scratch.write("end.cir", text);
    let limits = SourceLimits {
        cards: 1,
        tokens: 1,
        ..SourceLimits::default()
    };
    assert!(Parser::new().parse_text(&path, text, limits).is_ok());
    assert!(Parser::new().parse_file_with_limits(&path, limits).is_ok());
}

#[test]
fn excessive_tokens_are_rejected_at_the_first_unadmitted_physical_coordinate() {
    let path = Path::new("punctuation.cir");
    let text = format!("title\nr1 a 0\n+ {}\n", ",".repeat(100_000));
    let error = Parser::new()
        .parse_text(
            path,
            &text,
            SourceLimits {
                tokens: 6,
                ..SourceLimits::default()
            },
        )
        .unwrap_err();
    assert!(error.to_string().contains("punctuation.cir:3:6"), "{error}");
    assert!(error.to_string().contains("token work"), "{error}");
}

#[test]
fn default_policy_accepts_long_pwl_and_wide_flat_decks() {
    let mut text = "pwl\nv1 out 0 pwl(".to_owned();
    for i in 0..2048 {
        if i > 0 && i % 32 == 0 {
            text.push_str("\n+");
        }
        write!(text, " {i} {}", i % 2).unwrap();
    }
    text.push_str(")\nr1 out 0 1k\n.end\n");
    let parsed = Parser::new()
        .parse_text(Path::new("pwl.cir"), &text, SourceLimits::default())
        .unwrap();
    assert_eq!(parsed.netlist.devices.len(), 2);
    // The behavioral expression grammar permits a larger PWL than
    // ordinary source-vector fields. Preserve both independently.
    let mut behavioral = "wide behavioral PWL\nv1 in 0 1\nb1 o 0 v=pwl(v(in)".to_owned();
    for i in 0..8192 {
        write!(behavioral, ",{i},{i}").unwrap();
    }
    behavioral.push_str(")\n.end\n");
    assert_eq!(
        Parser::new()
            .parse_text(
                Path::new("behavioral.cir"),
                &behavioral,
                SourceLimits::default()
            )
            .unwrap()
            .netlist
            .devices
            .len(),
        2
    );
    let mut text = "flat\n".to_owned();
    for i in 0..100_000 {
        writeln!(text, "r{i} n{i} 0 1k").unwrap();
    }
    text.push_str(".end\n");
    let parsed = Parser::new()
        .parse_text(Path::new("flat.cir"), &text, SourceLimits::default())
        .unwrap();
    assert_eq!(parsed.netlist.devices.len(), 100_000);
}

fn punctuation_benchmark(tokens: usize) {
    let mut text = String::with_capacity(1_000_007);
    text.push_str("title\n");
    text.extend(std::iter::repeat_n(',', 1_000_000));
    text.push('\n');
    let start = std::time::Instant::now();
    let error = Parser::new()
        .parse_text(
            Path::new("punctuation.cir"),
            &text,
            SourceLimits {
                tokens,
                ..SourceLimits::default()
            },
        )
        .unwrap_err();
    assert_eq!(
        error.to_string().contains("token work"),
        tokens < 1_000_000,
        "{error}"
    );
    eprintln!(
        "punctuation=1000000; token budget={tokens}; elapsed={:?}",
        start.elapsed()
    );
}

#[test]
#[ignore = "measure bounded lexical allocation and runtime in a separate process"]
fn punctuation_budget_benchmark() {
    punctuation_benchmark(8192);
}

#[test]
#[ignore = "measure fully admitted lexical allocation in a separate process"]
fn punctuation_admitted_benchmark() {
    punctuation_benchmark(1_000_000);
}

#[test]
#[ignore = "measure bounded physical/card metadata allocation in a separate process"]
fn line_budget_benchmark() {
    let mut text = String::with_capacity(4_000_006);
    text.push_str("title\n");
    for _ in 0..1_000_000 {
        text.push_str(".op\n");
    }
    let start = std::time::Instant::now();
    let error = Parser::new()
        .parse_text(
            Path::new("lines.cir"),
            &text,
            SourceLimits {
                folded_cards: 8192,
                ..SourceLimits::default()
            },
        )
        .unwrap_err();
    assert!(error.to_string().contains("folded card work"), "{error}");
    eprintln!(
        "1,000,000 cards; folded 8192; elapsed={:?}",
        start.elapsed()
    );
}
