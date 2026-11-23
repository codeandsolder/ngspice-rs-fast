//! Physical CPL model and P-instance binding without registering the
//! incomplete transient device.
use std::path::Path;

use ngspice_rs::devices::cpl::model::bind;
use ngspice_rs::devices::cpl::plan::CplMatrixKind;
use ngspice_rs::netlist::Parser;
use ngspice_rs::netlist::elaborate::literalize;
use ngspice_rs::netlist::source::parse_deck_text;

fn model(
    text: &str,
) -> (
    ngspice_rs::netlist::ast::DeviceInstance,
    ngspice_rs::netlist::ast::ModelCard,
) {
    let parsed = Parser::new()
        .parse_deck(&parse_deck_text(Path::new("cpl-bind.cir"), text))
        .unwrap();
    let literal = literalize(&parsed).unwrap();
    (
        literal.netlist.devices[0].clone(),
        literal.netlist.models[0].clone(),
    )
}

#[test]
fn binds_ordered_model_and_instance_setters_with_coupling_intact() {
    let (instance, card) = model(
        "CPL\n\
        .param len=4 cm=-1n\n\
        .model cable cpl length=1 R=(1,0,1) L=(100n,0,100n) C=(10p,-2p,10p) L=(200n,{cm},200n) R=(2,0,2) length=3\n\
        P1 a b nret x y fret cable len={len}\n\
        .end\n",
    );
    let plan = bind(&instance, &card).unwrap();
    assert_eq!(plan.conductors(), 2);
    assert_eq!(plan.length(), 4.);
    assert_eq!(plan.matrix(CplMatrixKind::Resistance).get(0, 0), Some(2.));
    assert_eq!(plan.matrix(CplMatrixKind::Resistance).get(0, 1), Some(0.));
    assert!((plan.matrix(CplMatrixKind::Inductance).get(0, 1).unwrap() + 1e-9).abs() < 1e-25);
    assert_eq!(
        plan.matrix(CplMatrixKind::Capacitance).get(0, 1),
        Some(-2e-12)
    );
    assert_eq!(plan.modal_delays().len(), 2);
}

#[test]
fn three_conductor_geometry_preserves_full_mutual_terms() {
    let (instance, card) = model(
        "3 lines\n\
        P1 a b c ref1 x y z ref2 cable\n\
        .model cable cpl length=3 R=(1,1,1) L=(50n,1n,2n,60n,3n,70n) G=(0,0,0) C=(20p,-1p,-2p,25p,-3p,30p)\n\
        .end\n",
    );
    let plan = bind(&instance, &card).unwrap();
    assert_eq!(plan.conductors(), 3);
    assert_eq!(plan.matrix(CplMatrixKind::Inductance).get(2, 0), Some(2e-9));
    assert_eq!(
        plan.matrix(CplMatrixKind::Capacitance).get(2, 1),
        Some(-3e-12)
    );
    assert_eq!(plan.modal_delays().len(), 3);
}

#[test]
fn rejects_matrix_shape_missing_length_and_nonpassive_coupling() {
    for (line, message) in [
        (
            "P1 a b 0 x y 0 cable\n.model cable cpl length=1 L=(1n,1n) C=(1p,0,1p)",
            "upper-triangular",
        ),
        (
            "P1 a b 0 x y 0 cable\n.model cable cpl L=(1n,0,1n) C=(1p,0,1p)",
            "length",
        ),
        (
            "P1 a b 0 x y 0 cable\n.model cable cpl length=1 R=(1,2,1) L=(1n,0,1n) C=(1p,0,1p)",
            "positive-semidefinite",
        ),
    ] {
        let (instance, card) = model(&format!("invalid\n{line}\n.end\n"));
        let error = bind(&instance, &card).unwrap_err().to_string();
        assert!(error.contains(message), "{message} missing from {error}");
        assert!(error.contains("cpl-bind.cir:"), "{error}");
    }
}

#[test]
fn rejects_unliteralized_components_and_wrong_model_identity() {
    let text = "unliteralized\nP1 a b 0 x y 0 cable\n.model cable cpl length=1 L=({foo},0,1n) C=(1p,0,1p)\n.end\n";
    let parsed = Parser::new()
        .parse_deck(&parse_deck_text(Path::new("cpl-bind.cir"), text))
        .unwrap();
    let error = bind(&parsed.devices[0], &parsed.models[0])
        .unwrap_err()
        .to_string();
    assert!(error.contains("must be literalized"), "{error}");
    let (mut instance, card) = model(
        "wrong model\nP1 a b 0 x y 0 cable\n.model cable cpl length=1 L=(1n,0,1n) C=(1p,0,1p)\n.end\n",
    );
    instance.model = Some("elsewhere".into());
    assert!(bind(&instance, &card).is_err());
}
