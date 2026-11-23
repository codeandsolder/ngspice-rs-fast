//! #13 production file API, bounded resolution and source provenance.
use ngspice_rs::netlist::ast::ScopedCardKind;
use ngspice_rs::netlist::source::parse_deck_text;
use ngspice_rs::netlist::{Parser, SourceLimits};
use ngspice_rs::primitives::SpiceError;
use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

static NEXT: AtomicUsize = AtomicUsize::new(0);
struct Files(PathBuf);
impl Files {
    fn new() -> Self {
        let path = std::env::temp_dir().join(format!(
            "spice-sources-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir_all(&path).unwrap();
        Self(path)
    }
    fn write(&self, name: &str, text: &str) -> PathBuf {
        let path = self.0.join(name);
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(&path, text).unwrap();
        path
    }
    fn parse(&self, name: &str) -> Result<ngspice_rs::netlist::ast::Netlist, SpiceError> {
        Parser::new().parse_file(self.0.join(name))
    }
}
impl Drop for Files {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

#[test]
fn syntax_only_retains_quoted_paths_without_io() {
    let deck = parse_deck_text(
        Path::new("absent.cir"),
        "title\n.inc 'missing dir/x.inc'\n.lib \"no.lib\" TT\n",
    );
    let n = Parser::new().parse_deck(&deck).unwrap();
    assert_eq!(n.includes[0].path, "missing dir/x.inc");
    assert_eq!(n.includes[0].path_spelling, "'missing dir/x.inc'");
    assert!(n.includes[0].resolved_path.is_none());
    assert_eq!(n.includes[1].section.as_deref(), Some("tt"));
    assert!(n.includes[1].selected_section.is_none());
    assert_eq!(n.cards[1].kind, ScopedCardKind::Include(1));
}

#[test]
fn relative_sources_expand_in_order_and_inherit_the_insertion_scope() {
    let f = Files::new();
    let root = f.write("main.cir", "Title\nr0 a 0 1k\n.subckt block a b\n.include 'dir/body.inc'\n.ends block\nX1 a 0 block\n.end\n.include missing\n");
    let body = f.write(
        "dir/body.inc",
        "Q1 c b e QM\n.include '../models.inc'\nr1 a b 2K\n+ tc1=0.01\n",
    );
    let models = f.write("models.inc", ".model QM NPN\n");
    let n = Parser::new().parse_file(&root).unwrap();
    assert_eq!(n.path, root);
    assert_eq!(
        n.devices
            .iter()
            .map(|d| d.name.as_str())
            .collect::<Vec<_>>(),
        ["r0", "x1"]
    );
    let s = &n.subcircuits[0];
    assert_eq!(
        s.devices
            .iter()
            .map(|d| d.name.as_str())
            .collect::<Vec<_>>(),
        ["q1", "r1"]
    );
    assert_eq!(n.models, []);
    assert_eq!(s.models[0].name, "qm");
    assert_eq!(
        s.cards.iter().map(|c| c.kind).collect::<Vec<_>>(),
        [
            ScopedCardKind::Include(0),
            ScopedCardKind::Device(0),
            ScopedCardKind::Include(1),
            ScopedCardKind::Model(0),
            ScopedCardKind::Device(1),
            ScopedCardKind::Ends
        ]
    );
    assert_eq!(
        s.includes[0].resolved_path.as_ref().unwrap(),
        &fs::canonicalize(body).unwrap()
    );
    assert_eq!(
        s.models[0].location.path(),
        fs::canonicalize(models).unwrap()
    );
    assert_eq!(
        s.models[0].location.line, 1,
        "fragment first line is not a title"
    );
    assert_eq!(
        s.cards[3]
            .include_chain
            .iter()
            .map(|l| l.line)
            .collect::<Vec<_>>(),
        [4, 2]
    );
    assert_eq!(
        s.devices[1].parameters[1].location.line, 4,
        "physical continuation line"
    );
    assert_eq!(s.devices[1].parameters[0].value, "2K");
}

#[test]
fn library_selects_only_requested_section_and_preserves_boundaries() {
    let f = Files::new();
    f.write(
        "main.cir",
        "title\n.lib 'libs/process.lib' tT\nr1 a 0 RES\n",
    );
    f.write("libs/process.lib", "* fragment comment\n.lib FF\n.include missing.inc\nmalformed {\n.endl ff\n.lib TT\n.lib 'nested.lib' BASE\n.endl TT\n");
    f.write("libs/nested.lib", ".lib base\n.model RES R RSH=1K\n.endl\n");
    let n = f.parse("main.cir").unwrap();
    assert_eq!(n.models[0].parameters[0].value, "1K");
    assert_eq!(n.devices[0].model.as_deref(), Some("res"));
    assert_eq!(n.includes.len(), 2);
    let section = n.includes[0].selected_section.as_ref().unwrap();
    assert_eq!(section.name, "tt");
    assert_eq!(section.opening.raw, ".lib TT");
    assert_eq!(section.opening.location.line, 6);
    assert_eq!(section.closing.raw, ".endl TT");
    assert_eq!(n.cards[2].include_chain.len(), 2);
}

#[test]
fn repeated_and_diamond_includes_are_not_cycles_or_deduplicated() {
    let f = Files::new();
    f.write(
        "main.cir",
        "title\n.include a.inc\n.include b.inc\n.include a.inc\n",
    );
    f.write("a.inc", ".include shared.inc\n");
    f.write("b.inc", ".include shared.inc\n");
    f.write("shared.inc", "r1 a 0 1K\n");
    let n = f.parse("main.cir").unwrap();
    assert_eq!(n.devices.len(), 3);
    assert_eq!(n.includes.len(), 6);
    assert_eq!(
        n.cards
            .iter()
            .filter(|c| matches!(c.kind, ScopedCardKind::Device(_)))
            .count(),
        3
    );
}

#[test]
fn empty_and_adjacent_library_sections_retain_exact_boundaries() {
    let f = Files::new();
    f.write(
        "main.cir",
        "title\n.lib 'process.lib' empty\n.lib 'process.lib' used\n.end\n",
    );
    f.write("process.lib", ".lib empty\n.endl empty\n.lib used\nR1 a 0\n+ 1k\n.endl used\n.lib other\nmalformed {\n.endl other\n");
    let n = f.parse("main.cir").unwrap();
    assert_eq!(n.devices.len(), 1);
    assert_eq!(n.devices[0].name, "r1");
    assert_eq!(n.devices[0].location.line, 4);
    for (include, start, end) in [(&n.includes[0], 1, 2), (&n.includes[1], 3, 6)] {
        let boundary = include.selected_section.as_ref().unwrap();
        assert_eq!(boundary.opening.location.line, start);
        assert_eq!(boundary.closing.location.line, end);
    }
    assert_eq!(n.cards[2].include_chain.len(), 1);
    assert_eq!(n.cards[2].include_chain[0].line, 3);

    // Boundary validation must continue beyond the selected section.
    f.write(
        "process.lib",
        ".lib empty\n.endl empty\n.lib other\n.endl wrong\n",
    );
    let error = f.parse("main.cir").unwrap_err();
    assert!(error.to_string().contains("mismatched .endl"), "{error}");
}
#[test]
fn canonical_file_and_section_dependencies_detect_cycles() {
    let f = Files::new();
    f.write("main.cir", "title\n.include a.inc\n");
    f.write("a.inc", ".include ./a.inc\n");
    assert!(
        f.parse("main.cir")
            .unwrap_err()
            .to_string()
            .contains("cycle")
    );
    f.write("a.inc", ".include b.inc\n");
    f.write("b.inc", ".include a.inc\n");
    let e = f.parse("main.cir").unwrap_err();
    assert!(e.to_string().contains("b.inc:1:1"), "{e}");
    f.write("main.cir", "title\n.lib lib.inc a\n");
    f.write(
        "lib.inc",
        ".lib a\n.lib lib.inc b\n.endl a\n.lib b\nr1 a 0 1k\n.endl b\n",
    );
    assert_eq!(
        f.parse("main.cir").unwrap().devices.len(),
        1,
        "distinct sections can share a file"
    );
    f.write(
        "lib.inc",
        ".lib a\n.lib lib.inc b\n.endl a\n.lib b\n.lib lib.inc a\n.endl b\n",
    );
    assert!(
        f.parse("main.cir")
            .unwrap_err()
            .to_string()
            .contains("cycle")
    );
}

#[cfg(unix)]
#[test]
fn symlink_aliases_cannot_bypass_cycle_detection() {
    let f = Files::new();
    f.write("main.cir", "title\n.include a.inc\n");
    let a = f.write("a.inc", ".include alias.inc\n");
    std::os::unix::fs::symlink(a, f.0.join("alias.inc")).unwrap();
    assert!(
        f.parse("main.cir")
            .unwrap_err()
            .to_string()
            .contains("cycle")
    );
}

#[test]
fn library_boundary_and_missing_source_errors_are_positioned() {
    let f = Files::new();
    f.write("main.cir", "title\n.lib parts.lib tt\n");
    for (text, message) in [
        (".lib other\n.endl\n", "not found"),
        (".lib tt\n", "missing .endl"),
        (".lib tt\n.endl ff\n", "mismatched"),
        (".endl tt\n", "unmatched"),
        (".lib tt\n.lib nested\n.endl\n.endl\n", "nested"),
        (".lib tt\n.endl\n.lib TT\n.endl\n", "duplicate"),
    ] {
        f.write("parts.lib", text);
        let e = f.parse("main.cir").unwrap_err();
        assert!(matches!(e, SpiceError::Parse { .. }), "{e}");
        assert!(e.to_string().contains(message), "{e}");
    }
    f.write("main.cir", "title\n.include missing.inc\n");
    let e = f.parse("main.cir").unwrap_err();
    assert!(e.to_string().contains("main.cir:2:1"), "{e}");
    f.write("missing.inc", "r1 a 0 \"unknown\"\n");
    let e = f.parse("main.cir").unwrap_err();
    assert!(e.is_not_yet_ported());
    assert!(e.to_string().contains("missing.inc:1:"), "{e}");
}

#[test]
fn file_depth_byte_and_card_work_limits_include_repeated_sources() {
    let f = Files::new();
    let root = f.write("main.cir", "title\n.include a.inc\n.include a.inc\n");
    f.write("a.inc", "r1 a 0 1k\n");
    let defaults = SourceLimits::default();
    for (limits, message) in [
        (
            SourceLimits {
                depth: 0,
                ..defaults
            },
            "depth limit",
        ),
        (
            SourceLimits {
                files: 2,
                ..defaults
            },
            "file work limit",
        ),
        (
            SourceLimits {
                bytes: 36,
                ..defaults
            },
            "byte work limit",
        ),
        (
            SourceLimits {
                cards: 3,
                ..defaults
            },
            "card work limit",
        ),
        (
            SourceLimits {
                depth: 65,
                ..defaults
            },
            "at most 64",
        ),
        (
            SourceLimits {
                files: 0,
                ..defaults
            },
            "file work limit",
        ),
    ] {
        let e = Parser::new()
            .parse_file_with_limits(&root, limits)
            .unwrap_err();
        assert!(e.to_string().contains(message), "{e}");
    }
    let limits = SourceLimits {
        depth: 1,
        files: 3,
        bytes: 56,
        cards: 4,
        ..defaults
    };
    assert_eq!(
        Parser::new()
            .parse_file_with_limits(root, limits)
            .unwrap()
            .devices
            .len(),
        2
    );
    f.write("a.inc", ".include b.inc\n");
    f.write("b.inc", "r1 a 0 1k\n");
    let e = Parser::new()
        .parse_file_with_limits(
            f.0.join("main.cir"),
            SourceLimits {
                depth: 1,
                ..defaults
            },
        )
        .unwrap_err();
    assert!(e.to_string().contains("depth limit"), "{e}");
    // A bounded read that cuts a multibyte token reports the byte budget,
    // rather than misreporting the truncated buffer as invalid UTF-8.
    let unicode = f.write("unicode.cir", "title\nÖ");
    let e = Parser::new()
        .parse_file_with_limits(
            unicode,
            SourceLimits {
                bytes: 7,
                ..defaults
            },
        )
        .unwrap_err();
    assert!(e.to_string().contains("byte work limit"), "{e}");
    fs::write(f.0.join("a.inc"), [0xff]).unwrap();
    let e = f.parse("main.cir").unwrap_err();
    assert!(e.to_string().contains("not UTF-8"), "{e}");
}

#[test]
fn end_and_error_order_survive_preprocessing() {
    let f = Files::new();
    // The unported card on line 2 wins over the missing include on line 3;
    // `.width` stands in for an unported front-end card.
    f.write("main.cir", "title\n.width 80\n.include absent\n");
    let e = f.parse("main.cir").unwrap_err();
    assert!(e.is_not_yet_ported());
    assert!(e.to_string().contains("main.cir:2:1"));
    f.write(
        "main.cir",
        "title\n.include a.inc\n.include absent\nmalformed {\n",
    );
    f.write("a.inc", "r1 a 0 1k\n.end\n.include absent\n");
    assert_eq!(f.parse("main.cir").unwrap().devices.len(), 1);
    f.write("main.cir", "title\n.subckt s\n.include a.inc\n.ends\n");
    assert!(
        f.parse("main.cir")
            .unwrap_err()
            .to_string()
            .contains(".end before .ends")
    );
}

#[test]
fn malformed_source_cards_require_complete_consumption() {
    for body in [
        ".include",
        ".include ''",
        ".include a b",
        ".lib",
        ".lib a ''",
        ".lib a tt extra",
        ".endl",
        ".lib tt",
    ] {
        let deck = parse_deck_text(Path::new("bad.cir"), &format!("title\n{body}\n"));
        assert!(Parser::new().parse_deck(&deck).is_err(), "{body}");
    }
}

/// C has no card cap (`inpcom.c`, `inp_readall`), so the default card budget
/// must accept a large flat deck: 150,000 cards was rejected by the former
/// 100,000-card default (a 50,000-stage RC ladder needs about 100,000).
#[test]
fn default_card_budget_accepts_a_large_flat_deck() {
    let f = Files::new();
    let mut deck = String::from("large flat deck\nv1 n0 0 1\n");
    for i in 0..149_997 {
        writeln!(deck, "r{i} n{i} n{} 1", i + 1).unwrap();
    }
    deck.push_str(".op\n.end\n");
    f.write("main.cir", &deck);
    let netlist = f.parse("main.cir").unwrap();
    assert_eq!(netlist.devices.len(), 149_998);
}

/// The budget still stops runaway repeated includes with an explicit error
/// naming the limit, before the file (1,024) or byte (16 MiB) budgets: 600
/// reads of a 2,000-card file are 1.2 million cards in about 12 MB.
#[test]
fn default_card_budget_stops_runaway_repeated_includes() {
    let f = Files::new();
    let mut body = String::new();
    for i in 0..2_000 {
        writeln!(body, "r{i} a 0 1").unwrap();
    }
    f.write("body.inc", &body);
    let mut deck = String::from("runaway includes\n");
    for _ in 0..600 {
        deck.push_str(".include body.inc\n");
    }
    deck.push_str(".end\n");
    f.write("main.cir", &deck);
    let error = f.parse("main.cir").unwrap_err();
    assert!(matches!(error, SpiceError::Parse { .. }), "{error}");
    assert!(
        error
            .to_string()
            .contains("source card work limit exceeded (1000000 cards"),
        "{error}"
    );
    assert!(error.to_string().contains("body.inc"), "{error}");
}

#[test]
fn included_and_library_continuations_use_fragment_physical_coordinates() {
    let f = Files::new();
    f.write("main.cir", "title\r\n  .include 'part.inc'\r\n.end\r\n");
    let part = f.write(
        "part.inc",
        "* comment\r\n  R1 α 0\r\n* between\r\n   + 1k\r\n",
    );
    let n = f.parse("main.cir").unwrap();
    let at = &n.devices[0].parameters[0].location;
    assert_eq!(at.path(), fs::canonicalize(&part).unwrap());
    assert_eq!((at.line, at.column), (4, 6));
    assert_eq!(
        (
            n.cards[1].include_chain[0].line,
            n.cards[1].include_chain[0].column
        ),
        (2, 3)
    );

    f.write("main.cir", "title\n  .lib 'models.lib' chosen\n.end\n");
    let library = f.write(
        "models.lib",
        ".lib ignored\n.endl ignored\n.lib chosen\n  R1 a 0 1k\n* comment\n  + tc1= ; absent\n.endl chosen\n",
    );
    let SpiceError::Parse { location, .. } = f.parse("main.cir").unwrap_err() else {
        panic!("expected located library parse error");
    };
    assert_eq!(location.path(), fs::canonicalize(library).unwrap());
    assert_eq!((location.line, location.column), (6, 9));
}

#[test]
fn repeated_sources_share_lexical_storage_not_scope_or_occurrence() {
    let f = Files::new();
    f.write("main.cir", "title\n.subckt first a\n.model local R RSH=1k\n.include body.inc\n.ends\n.subckt second a\n.model local R RSH=2k\n.include ./body.inc\n.ends\n");
    f.write("body.inc", "r1 a 0 local\n+ tc1=0.2\nr2 a 0 1k\n");
    let netlist = f.parse("main.cir").unwrap();
    let first = &netlist.subcircuits[0];
    let second = &netlist.subcircuits[1];
    assert_eq!(first.models[0].parameters[0].value, "1k");
    assert_eq!(second.models[0].parameters[0].value, "2k");
    assert_eq!(first.devices.len(), 2);
    assert_eq!(second.devices.len(), 2);
    for (left, right) in first.cards[2..4].iter().zip(&second.cards[2..4]) {
        assert!(Arc::ptr_eq(&left.source, &right.source));
        assert_ne!(left.include_chain, right.include_chain);
        assert_eq!(left.include_chain[0].line, 4);
        assert_eq!(right.include_chain[0].line, 8);
    }
    assert!(Arc::ptr_eq(
        &first.cards[2].include_chain,
        &first.cards[3].include_chain
    ));
    assert!(Arc::ptr_eq(
        &second.cards[2].include_chain,
        &second.cards[3].include_chain
    ));
    assert_eq!(first.devices[0].parameters.last().unwrap().location.line, 2);
}

#[test]
fn repeated_library_visits_share_sections_and_preserve_logical_work() {
    let f = Files::new();
    let repeats = 128;
    let width = 24;
    let mut root_text = String::from("title\n");
    let mut library = String::from(".lib used\n");
    for _ in 0..width {
        library.push_str("r1 a 0 1k\n");
    }
    library.push_str(".endl used\n.lib empty\n.endl empty\n");
    for _ in 0..repeats {
        root_text.push_str(".lib parts.lib used\n.lib parts.lib empty\n");
    }
    let root = f.write("main.cir", &root_text);
    f.write("parts.lib", &library);
    let limits = SourceLimits {
        files: 1 + 2 * repeats,
        bytes: root_text.len() + 2 * repeats * library.len(),
        cards: repeats * (width + 2),
        ..SourceLimits::default()
    };
    let netlist = Parser::new().parse_file_with_limits(&root, limits).unwrap();
    assert_eq!(netlist.devices.len(), repeats * width);
    assert_eq!(netlist.includes.len(), 2 * repeats);
    // 3,072 emitted device cards share 24 lexical allocations.
    for occurrence in netlist.cards.chunks_exact(width + 2) {
        for (first, later) in netlist.cards[1..=width].iter().zip(&occurrence[1..=width]) {
            assert!(Arc::ptr_eq(&first.source, &later.source));
            assert!(Arc::ptr_eq(
                &occurrence[1].include_chain,
                &later.include_chain
            ));
        }
    }
    let first = netlist.includes[0].selected_section.as_ref().unwrap();
    for include in netlist.includes.iter().step_by(2) {
        let section = include.selected_section.as_ref().unwrap();
        assert!(Arc::ptr_eq(&first.opening, &section.opening));
        assert!(Arc::ptr_eq(&first.closing, &section.closing));
    }
    for (smaller, expected) in [
        (
            SourceLimits {
                files: limits.files - 1,
                ..limits
            },
            "file work limit",
        ),
        (
            SourceLimits {
                bytes: limits.bytes - 1,
                ..limits
            },
            "byte work limit",
        ),
        (
            SourceLimits {
                cards: limits.cards - 1,
                ..limits
            },
            "card work limit",
        ),
    ] {
        let error = Parser::new()
            .parse_file_with_limits(&root, smaller)
            .unwrap_err();
        assert!(error.to_string().contains(expected), "{error}");
    }
}

#[test]
fn root_can_select_its_own_section_without_reinterpreting_the_title() {
    let f = Files::new();
    f.write(
        "main.cir",
        "title\n.lib main.cir body\n.end\n.lib body\nr1 a 0 1k\n.endl body\n",
    );
    let netlist = f.parse("main.cir").unwrap();
    assert_eq!(netlist.title, "title");
    assert_eq!(netlist.devices.len(), 1);
    assert_eq!(netlist.devices[0].location.line, 5);
    assert_eq!(
        netlist.includes[0]
            .selected_section
            .as_ref()
            .unwrap()
            .opening
            .location
            .line,
        4
    );
}
