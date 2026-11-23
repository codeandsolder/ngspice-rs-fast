//! CPL model-vector expressions must pass through the same scoped numparam
//! literalization as scalar model and instance parameters.
use std::path::Path;

use ngspice_rs::netlist::Parser;
use ngspice_rs::netlist::ast::ParameterKind;
use ngspice_rs::netlist::elaborate::{SiteKind, literalize};
use ngspice_rs::netlist::source::parse_deck_text;

#[test]
fn vector_components_resolve_parameters_and_preserve_provenance() {
    let source = "CPL\n\
        .param m=-5n cap=1p distance=7\n\
        .model cm CPL L=(100n,{m},100n) C=(10p,{-cap},10p) length={distance}\n\
        P1 a b nret x y fret cm len={distance*2}\n\
        .end\n";
    let parsed = Parser::new()
        .parse_deck(&parse_deck_text(Path::new("cpl-scope.cir"), source))
        .unwrap();
    let original_model = &parsed.models[0];
    assert!(matches!(
        original_model.parameters[0].kind,
        ParameterKind::NumericVector(_)
    ));
    let result = literalize(&parsed).unwrap();
    let model = &result.netlist.models[0];
    for (index, expected) in [(0, 100e-9), (1, -5e-9), (2, 100e-9)] {
        let ParameterKind::NumericVector(values) = &model.parameters[0].kind else {
            panic!("typed CPL vector expected");
        };
        let value = ngspice_rs::primitives::parse_spice_number(&values[index].text).unwrap();
        assert!((value - expected).abs() <= 1e-24);
    }
    let ParameterKind::NumericVector(values) = &model.parameters[1].kind else {
        panic!("typed C expected");
    };
    assert_eq!(values.len(), 3);
    assert!(
        (ngspice_rs::primitives::parse_spice_number(&values[1].text).unwrap() + 1e-12).abs()
            < 1e-25
    );
    assert_eq!(
        result
            .sites
            .iter()
            .filter(|site| matches!(site.kind, SiteKind::ModelVectorComponent { .. }))
            .count(),
        2
    );
    assert!(result.sites.iter().any(|site| matches!(
        site.kind,
        SiteKind::ModelVectorComponent {
            model: 0,
            parameter: 0,
            component: 1
        }
    ) && site.original == "{m}"
        && site.location.line == 3));
    assert_eq!(result.sites.len(), 4);
    assert_eq!(
        result.netlist.devices[0].parameters[0].kind,
        ParameterKind::Scalar
    );
    // The source AST is untouched.
    let ParameterKind::NumericVector(source_values) = &original_model.parameters[0].kind else {
        panic!("original vector expected");
    };
    assert_eq!(source_values[1].text, "{m}");
}

#[test]
fn undefined_matrix_symbol_reports_precise_component_location() {
    let source = "CPL\n.model cm CPL L=(1n,{missing},1n) C=(1p,0,1p)\n.end\n";
    let parsed = Parser::new()
        .parse_deck(&parse_deck_text(Path::new("cpl-undefined.cir"), source))
        .unwrap();
    let error = literalize(&parsed).unwrap_err().to_string();
    assert!(error.contains("missing"), "{error}");
    assert!(error.contains("matrix 'l' component 2"), "{error}");
    assert!(error.contains("cpl-undefined.cir:2:"), "{error}");
}

#[test]
fn quoted_components_share_parameter_function_scope() {
    let source = "CPL\n\
        .param b=2n\n\
        .func incr(x) {x+b}\n\
        .model cm CPL L=('incr(4n)',0,'incr(4n)') C=(10p,0,10p)\n\
        .end\n";
    let parsed = Parser::new()
        .parse_deck(&parse_deck_text(Path::new("cpl-func.cir"), source))
        .unwrap();
    let result = literalize(&parsed).unwrap();
    let ParameterKind::NumericVector(values) = &result.netlist.models[0].parameters[0].kind else {
        panic!("model vector expected");
    };
    let value = ngspice_rs::primitives::parse_spice_number(&values[0].text).unwrap();
    assert!((value - 6e-9).abs() < 1e-24);
}
