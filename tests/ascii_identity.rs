//! The same ASCII identity rule is used by public model and plot lookups.

use ngspice_rs::analysis::{Plot, PlotFlags, Variable};
use ngspice_rs::devices::ModelResolver;
use ngspice_rs::netlist::Parser;
use ngspice_rs::netlist::source::parse_deck_text;
use ngspice_rs::primitives::ascii::folded;
use std::borrow::Cow;
use std::path::Path;

#[test]
fn folded_identifiers_borrow_canonical_names_without_touching_unicode() {
    for name in ["", "r1", "v(a)", "résistance", "Σprobe"] {
        let actual = folded(name);
        assert!(matches!(actual, Cow::Borrowed(_)));
        assert!(std::ptr::eq(actual.as_ref(), name));
    }
    for (before, after) in [
        ("R1", "r1"),
        ("V(A)", "v(a)"),
        ("ΣPROBE", "Σprobe"),
        ("ÉTAGE.2", "Étage.2"),
    ] {
        assert_eq!(folded(before), after);
        assert!(matches!(folded(before), Cow::Owned(_)));
    }
}

#[test]
fn plot_indexes_keep_first_match_case_insensitively() {
    let plot = Plot::new(
        "op1",
        "Operating point",
        PlotFlags::Real,
        ["v(Out)", "V(out)", "I(V1)", "v(ref)", "Σsense", "v(OUT2)"]
            .into_iter()
            .map(|name| Variable::new(name, "voltage"))
            .collect(),
    );
    for name in ["v(out)", "V(OUT)", "Out", "out"] {
        let index = if name.contains('(') {
            plot.variable_index(name)
        } else {
            plot.schema().voltage(name)
        };
        assert_eq!(index, Some(0), "{name}");
    }
    assert_eq!(plot.variable_index("i(v1)"), Some(2));
    assert_eq!(plot.variable_index("Σsense"), Some(4));
    assert_eq!(plot.variable_index("σsense"), None);
    assert_eq!(plot.variable_index("v(out2)"), Some(5));
}

#[test]
fn models_preserve_first_declaration_and_fold_lookup_only() {
    let parsed = Parser::new()
        .parse_deck(&parse_deck_text(
            Path::new("ascii-identity.cir"),
            "t\nd1 a 0 MIXED\n.model MiXeD d(is=1e-14)\n.model mixed npn(is=2e-14)\n.end\n",
        ))
        .unwrap();
    let resolver = ModelResolver::new(&parsed.models).unwrap();
    for name in ["MiXeD", "mixed", "MIXED"] {
        assert_eq!(resolver.model(name), Some(&parsed.models[0]));
    }
    assert_eq!(
        resolver
            .resolve(&parsed.devices[0])
            .unwrap()
            .unwrap()
            .card(),
        &parsed.models[0]
    );
}
