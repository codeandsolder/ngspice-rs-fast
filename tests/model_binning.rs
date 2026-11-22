//! MOS model binning through the production parser, subcircuit expansion and
//! model resolver (#109). C bins only BSIM3/BSIM4/HiSIM2/HiSIM-HV models, none
//! of which this port simulates, so a selected bin is observed through the
//! explicit `NotYetPorted` error that names it; every C failure case is a
//! parse error. `tests/c_binning_reference.rs` checks the same selections
//! against the C binary.

use std::collections::BTreeSet;
use std::path::Path;

use ngspice_rs::analysis::RunConfig;
use ngspice_rs::devices::binning::{BinOptions, model_name_match};
use ngspice_rs::devices::{Circuit, ModelResolver};
use ngspice_rs::netlist::Parser;
use ngspice_rs::netlist::ast::{DeviceInstance, ModelCard, Netlist};
use ngspice_rs::netlist::source::parse_deck_text;
use ngspice_rs::primitives::SpiceError;

const BINS: &str = "\
.model nch.1 nmos level=8 lmin=0.5u lmax=1u wmin=0.5u wmax=5u vth0=0.3
.model nch.2 nmos level=8 lmin=1u lmax=5u wmin=0.5u wmax=5u vth0=0.5
.model nch.3 nmos level=8 lmin=0.1u lmax=1u wmin=5u wmax=20u vth0=0.7
";

fn parse(body: &str) -> Netlist {
    let text = format!("binning\nvd d 0 1\nvg g 0 1\n{body}.op\n.end\n");
    Parser::new()
        .parse_deck(&parse_deck_text(Path::new("bins.cir"), &text))
        .unwrap()
}

fn elaborate(body: &str) -> SpiceError {
    let netlist = parse(body);
    let config = RunConfig::from_netlist(&netlist).unwrap();
    config
        .circuit(&netlist)
        .expect_err("this deck must be rejected during resolution")
}

/// The model card the resolver selects for a root-level instance. Used where
/// an unused bin declaration would trip the circuit's unused-model check.
fn selected_resolved(body: &str) -> String {
    let netlist = parse(body);
    let resolver = ModelResolver::new(&netlist.models).unwrap();
    let device = netlist.device("m1").unwrap();
    resolver
        .resolve(device)
        .unwrap()
        .unwrap()
        .card()
        .name
        .clone()
}

/// The model card the binned MOS instance selected, read from the device.
fn selected(body: &str) -> String {
    let netlist = parse(body);
    let config = RunConfig::from_netlist(&netlist).unwrap();
    let circuit = config
        .circuit(&netlist)
        .unwrap_or_else(|error| panic!("{error}"));
    circuit
        .devices()
        .iter()
        .find_map(|device| device.model_name())
        .expect("the MOS device reports its model")
        .to_owned()
}

/// The model card of one named MOS device, after subcircuit expansion.
fn selected_named(body: &str, name: &str) -> String {
    let netlist = parse(body);
    let config = RunConfig::from_netlist(&netlist).unwrap();
    let circuit = config
        .circuit(&netlist)
        .unwrap_or_else(|error| panic!("{error}"));
    circuit
        .device(name)
        .unwrap_or_else(|| {
            panic!(
                "no device {name}; have {:?}",
                circuit
                    .devices()
                    .iter()
                    .map(|device| device.name())
                    .collect::<Vec<_>>()
            )
        })
        .model_name()
        .unwrap_or_else(|| panic!("{name} reports no model"))
        .to_owned()
}

fn rejected(body: &str) -> String {
    let error = elaborate(body);
    assert!(matches!(error, SpiceError::Parse { .. }), "{error}");
    error.to_string()
}

#[test]
fn selects_by_length_and_width_with_inclusive_edges() {
    assert_eq!(
        selected(&format!("m1 d g 0 0 nch w=1u l=0.7u\n{BINS}")),
        "nch.1"
    );
    assert_eq!(
        selected(&format!("m1 d g 0 0 nch w=1u l=2u\n{BINS}")),
        "nch.2"
    );
    assert_eq!(
        selected(&format!("m1 d g 0 0 nch w=10u l=0.5u\n{BINS}")),
        "nch.3"
    );
    // L=1u is inside nch.1 (lmax) and nch.2 (lmin): the last declared wins.
    assert_eq!(
        selected(&format!("m1 d g 0 0 nch w=1u l=1u\n{BINS}")),
        "nch.2"
    );
    // W=5u, L=1u is inside all three; nch.3 is declared last.
    assert_eq!(
        selected(&format!("m1 d g 0 0 nch w=5u l=1u\n{BINS}")),
        "nch.3"
    );
    // A 0.9 nm excess is still on the edge, 1.1 nm is not.
    assert_eq!(
        selected(&format!("m1 d g 0 0 nch w=1u l=5.0009u\n{BINS}")),
        "nch.2"
    );
    assert!(
        rejected(&format!("m1 d g 0 0 nch w=1u l=5.0011u\n{BINS}"))
            .contains("no bin with all of lmin/lmax/wmin/wmax")
    );
}

#[test]
fn declaration_order_breaks_overlap_and_multiplier_is_ignored() {
    let reversed = "\
.model nch.2 nmos level=8 lmin=1u lmax=5u wmin=0.5u wmax=5u
.model nch.1 nmos level=8 lmin=0.5u lmax=1u wmin=0.5u wmax=5u
";
    assert_eq!(
        selected(&format!("m1 d g 0 0 nch w=1u l=1u\n{reversed}")),
        "nch.1"
    );
    assert_eq!(
        selected(&format!("m1 d g 0 0 nch w=1u l=0.7u m=8\n{BINS}")),
        "nch.1"
    );
    // Forward references bin like any other declaration.
    assert_eq!(
        selected(&format!("{BINS}m1 d g 0 0 NCH W=1U L=0.7U\n")),
        "nch.1"
    );
}

#[test]
fn exact_names_win_and_suffixes_must_be_digits() {
    let exact = "\
.model nch.1 nmos level=8 lmin=0.5u lmax=2u wmin=0.5u wmax=5u
.model nch nmos level=8
";
    // The exact name wins over every bin.
    assert_eq!(
        selected_resolved(&format!("m1 d g 0 0 nch w=1u l=1u\n{exact}")),
        "nch"
    );
    assert_eq!(
        selected(
            "m1 d g 0 0 nch w=1u l=1u\n.model nch.01 nmos level=49 lmin=0.5u lmax=2u wmin=0.5u wmax=5u\n"
        ),
        "nch.01"
    );
    let netlist = Parser::new().parse_deck(&parse_deck_text(
        Path::new("bins.cir"),
        "t\nm1 d g 0 0 nch w=1u l=1u\n.model nch.a nmos level=8 lmin=0.5u lmax=2u wmin=0.5u wmax=5u\n.end\n",
    ));
    assert!(netlist.is_err());
    assert!(model_name_match("nch", "nch.12").is_some());
}

#[test]
fn c_failure_cases_are_explicit_parse_errors() {
    let level1 = ".model nch.1 nmos level=1 lmin=0.5u lmax=2u wmin=0.5u wmax=5u\n";
    let message = rejected(&format!("m1 d g 0 0 nch w=1u l=1u\n{level1}"));
    assert!(message.contains("ngspice bins only BSIM3"), "{message}");
    assert!(message.contains("'nch.1' (bins.cir:5:1)"), "{message}");
    let message = rejected(&format!("m1 d g 0 0 nch l=1u\n{BINS}"));
    assert!(message.contains("needs both l and w"), "{message}");
    let message = rejected(&format!("m1 d g 0 0 nch w=1u l=10u\n{BINS}"));
    assert!(message.contains(" W=1e-6;"), "{message}");
    // A candidate without all four bounds is skipped.
    let partial = ".model nch.1 nmos level=8 lmin=0.5u lmax=2u wmin=0.5u\n";
    assert!(rejected(&format!("m1 d g 0 0 nch w=1u l=1u\n{partial}")).contains("no bin"));
}

#[test]
fn bjt_references_are_never_binned() {
    let netlist = Parser::new().parse_deck(&parse_deck_text(
        Path::new("bins.cir"),
        "t\nq1 c b 0 qn\n.model qn.1 npn\n.end\n",
    ));
    assert!(netlist.is_err());
}

#[test]
fn subcircuit_bins_shadow_outer_models_like_subckt_c() {
    let local = "\
x1 d g inv
.subckt inv a b
m1 a b 0 0 nch w=1u l=1u
.model nch.1 nmos level=8 lmin=0.5u lmax=2u wmin=0.5u wmax=20u
.model nch.2 nmos level=8 lmin=0.5u lmax=2u wmin=0.5u wmax=20u
.ends
";
    // Root instances keep every root bin in use while the local set wins for
    // the subcircuit body; the last declared local bin wins there.
    let roots = "m2 d g 0 0 nch w=1u l=0.7u\nm3 d g 0 0 nch w=1u l=2u\nm4 d g 0 0 nch w=5u l=1u\n";
    let body = format!("{local}{BINS}{roots}");
    assert_eq!(selected_named(&body, "m.x1.m1"), "x1.nch.2");
    assert_eq!(selected_named(&body, "m2"), "nch.1");
    assert_eq!(selected_named(&body, "m4"), "nch.3");
    // Root bins are used when the body declares none.
    let global = "\
x1 d g inv
.subckt inv a b
m1 a b 0 0 nch w=1u l=0.7u
.ends
";
    let body = format!("{global}{BINS}{roots}");
    assert_eq!(selected_named(&body, "m.x1.m1"), "nch.1");
    // A local bin set captures the name even when none of it fits and the
    // root has a fitting bin or an exact model (C: "could not find a valid
    // modelname").
    let narrow = "\
x1 d g inv
.subckt inv a b
m1 a b 0 0 nch w=10u l=1u
.model nch.1 nmos level=8 lmin=0.5u lmax=2u wmin=0.5u wmax=2u
.ends
";
    let message = rejected(&format!("{narrow}{BINS}"));
    assert!(message.contains("model 'x1.nch'"), "{message}");
    let message = rejected(&format!("{narrow}.model nch nmos level=1\n"));
    assert!(message.contains("'x1.nch.1'"), "{message}");
    // An exact local model still shadows root bins.
    let exact = "\
x1 d g inv
.subckt inv a b
m1 a b 0 0 nch w=1u l=1u
.model nch nmos level=8
.ends
";
    // The local exact model shadows the root bins; a root instance still uses
    // them, so nothing is an unused declaration.
    let both = format!("{exact}{BINS}m2 d g 0 0 nch w=1u l=0.7u\n");
    assert_eq!(selected_named(&both, "m.x1.m1"), "x1.nch");
    assert_eq!(selected_named(&both, "m2"), "nch.1");
}

#[test]
fn root_models_may_not_join_a_local_bin_set() {
    let local = "\
x1 d g inv
.subckt inv a b
m1 a b 0 0 nch w=1u l=1u
.model nch.1 nmos level=8 lmin=0.5u lmax=2u wmin=0.5u wmax=20u
.ends
";
    for intruder in [
        ".model x1.nch.7 nmos level=8 lmin=0.5u lmax=2u wmin=0.5u wmax=20u\n",
        ".model x1.nch nmos level=8\n",
    ] {
        let message = rejected(&format!("{local}{intruder}"));
        assert!(
            message.contains("binned subcircuit models 'x1.nch.<n>'"),
            "{message}"
        );
    }
}

fn names<'a>(cards: impl Iterator<Item = &'a ModelCard>) -> Vec<String> {
    cards.map(|card| card.name.clone()).collect()
}

#[test]
fn resolver_exposes_candidates_and_selection() {
    let netlist = parse(&format!("m1 d g 0 0 nch w=2u l=2u\n{BINS}"));
    let resolver = ModelResolver::new(&netlist.models).unwrap();

    assert_eq!(
        names(resolver.bin_candidates("NCH")),
        ["nch.1", "nch.2", "nch.3"]
    );
    assert_eq!(
        names(resolver.declarations_for("nch")),
        ["nch.1", "nch.2", "nch.3"]
    );
    assert_eq!(names(resolver.declarations_for("nch.2")), ["nch.2"]);
    let mos = netlist
        .devices
        .iter()
        .find(|d| d.designator == 'm')
        .unwrap();
    assert_eq!(resolver.select_bin(mos).unwrap().unwrap().name, "nch.2");
    // `.options scale` is not yet a deck setting; the API takes it explicitly.
    let scaled = ModelResolver::new(&netlist.models)
        .unwrap()
        .with_bin_options(BinOptions {
            scale: 0.35,
            wnflag: false,
        });
    assert_eq!(scaled.select_bin(mos).unwrap().unwrap().name, "nch.1");
    // An exact-name or non-MOS reference is not a binning lookup.
    let vd = netlist.devices.iter().find(|d| d.name == "vd").unwrap();
    assert!(resolver.select_bin(vd).unwrap().is_none());
    assert!(Circuit::from_netlist(&netlist).is_ok());
}

#[test]
fn indexed_declarations_match_scan_for_dotted_and_case_insensitive_names() {
    let example = parse(BINS);
    let mut cards = Vec::new();
    for name in [
        "Nch.2",
        "nch.01",
        "NCH.2",
        "nch.",
        "nch.a",
        "nch.1a",
        "nch.2.0",
        "nch.2.1",
        "nch.2.01",
        "nch.2.0",
        "nch",
        "nch.１２",
        "µ.1",
        "µ.2",
        "Μ.1",
        "unrelated.42",
    ] {
        let mut card = example.models[0].clone();
        card.name = name.into();
        cards.push(card);
    }
    let resolver = ModelResolver::new(&cards).unwrap();
    let mut seen = BTreeSet::new();
    let first: Vec<_> = cards
        .iter()
        .filter(|card| seen.insert(card.name.to_ascii_lowercase()))
        .collect();
    for query in [
        "nch", "NCH", "nch.2", "NCH.2", "nch.01", "µ", "Μ", "missing", "nch.",
    ] {
        let mut exact = None;
        let mut candidates = Vec::new();
        for &card in &first {
            match model_name_match(query, &card.name) {
                Some(ngspice_rs::devices::binning::NameMatch::Exact) => exact = Some(card),
                Some(ngspice_rs::devices::binning::NameMatch::Bin) => candidates.push(card),
                None => {}
            }
        }
        assert_eq!(resolver.model(query), exact);
        assert_eq!(
            resolver.bin_candidates(query).collect::<Vec<_>>(),
            candidates
        );
        let expected: Vec<_> = exact.map_or(candidates, |card| vec![card]);
        assert_eq!(
            resolver.declarations_for(query).collect::<Vec<_>>(),
            expected
        );
    }
}

fn setter_value(device: &mut DeviceInstance, name: &str, value: &str) {
    device
        .parameters
        .iter_mut()
        .find(|p| p.name == name)
        .unwrap()
        .value = value.into();
}

fn corrupt_bound(card: &mut ModelCard) {
    card.parameters
        .iter_mut()
        .find(|p| p.name == "lmin")
        .unwrap()
        .value = "broken".into();
}

#[test]
fn lazy_bound_validation_respects_precedence_and_geometry_changes() {
    let mut deck = parse(&format!(
        "m1 d g 0 0 nch w=1u l=1u
{BINS}"
    ));
    let mut malformed = deck.models[0].clone();
    malformed.name = "nch.0".into();
    corrupt_bound(&mut malformed);
    let mut shadowed = deck.models[1].clone();
    shadowed.name = "NCH.2".into();
    corrupt_bound(&mut shadowed);
    deck.models.insert(0, malformed);
    deck.models.push(shadowed);

    let resolver = ModelResolver::new(&deck.models).unwrap();
    let mut device = deck.device("m1").unwrap().clone();
    device.model = Some("NCH".into());
    for length in ["1u", "2u", "0.7u", "10u", "2u", "10u", "1u"] {
        setter_value(&mut device, "l", length);
        let result = resolver.resolve(&device);
        match length {
            "0.7u" => assert_eq!(result.unwrap().unwrap().card().name, "nch.1"),
            "10u" => {
                let error = result.unwrap_err();
                let SpiceError::Parse { location, .. } = error else {
                    panic!("expected malformed bound: {error}");
                };
                assert_eq!(
                    location,
                    deck.models[0]
                        .parameters
                        .iter()
                        .find(|p| p.name == "lmin")
                        .unwrap()
                        .location
                );
            }
            _ => assert_eq!(result.unwrap().unwrap().card().name, "nch.2"),
        }
    }
    // Exact lookup bypasses geometry and bounds, even after the bound failure
    // has been cached; wrong-family diagnostics precede level validation.
    device.model = Some("nch.0".into());
    setter_value(&mut device, "l", "broken");
    assert_eq!(
        resolver.resolve(&device).unwrap().unwrap().card().name,
        "nch.0"
    );
    device.designator = 'q';
    assert!(
        resolver
            .resolve(&device)
            .unwrap_err()
            .to_string()
            .contains("wrong model family")
    );
}

#[test]
fn selector_failures_remain_declaration_local_and_follow_family_check() {
    let mut deck = parse(&format!(
        "m1 d g 0 0 nch w=1u l=1u
{BINS}"
    ));
    // Make nch.2's ordered level setter disagree with the raw first-level
    // cache. Selection of the bin remains valid; only resolution diagnoses it.
    deck.models[1].level = Some(49.);
    let resolver = ModelResolver::new(&deck.models).unwrap();
    let mut device = deck.device("m1").unwrap().clone();
    assert_eq!(resolver.select_bin(&device).unwrap().unwrap().name, "nch.2");
    let error = resolver.resolve(&device).unwrap_err();
    assert!(error.to_string().contains("raw level cache"));
    device.model = Some("NCH.2".into());
    device.designator = 'q';
    assert!(
        resolver
            .resolve(&device)
            .unwrap_err()
            .to_string()
            .contains("wrong model family")
    );
    device.designator = 'm';
    assert_eq!(resolver.resolve(&device).unwrap_err(), error);
    device.model = Some("nch.1".into());
    assert_eq!(
        resolver.resolve(&device).unwrap().unwrap().card().name,
        "nch.1"
    );
}
