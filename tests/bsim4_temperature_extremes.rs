//! Checked real BSIM4 temperature preparation across difficult size-scaled
//! screening parameters, binning and temperature variation (#155).
//! The underlying screening function also has an independent 90-digit
//! decimal oracle in `devices::bsim4::temp::tests`.
use std::path::Path;

use ngspice_rs::devices::{ModelResolver, bsim4};
use ngspice_rs::netlist::{Parser, source::parse_deck_text};

#[test]
fn extreme_screening_model_preparation_is_finite_and_replayable() {
    let models = [
        (
            "ordinary",
            ".model nn nmos level=54 version=4.8.3",
            "l=0.1u w=1u",
        ),
        (
            "near_zero_screening",
            ".model nn nmos level=54 version=4.8.3 dsub=1e-9 drout=1e-9 dvt1w=1e-9 dvt1=1e-9 dvt0=0 dvt0w=0 pdiblc1=0",
            "l=0.1u w=1u",
        ),
        (
            "screening_zero",
            ".model nn nmos level=54 version=4.8.3 dsub=0 drout=0 dvt1w=0 dvt1=0 dvt0=0 dvt0w=0 pdiblc1=0",
            "l=0.1u w=1u nf=2",
        ),
        (
            "pmos_multibin",
            ".model nn pmos level=54 version=4.8.3 binunit=1 dsub=1e-7 drout=1e-7 ldsub=1e-15 wdrout=1e-15 dvt1w=0.1 dvt1=0.2",
            "l=0.2u w=3u nf=3",
        ),
    ];
    for (label, model_card, instance_card) in models {
        let source = format!(
            "BSIM4 temperature extremes\nv1 d 0 1.2\nvg g 0 0.6\nm1 d g 0 0 nn {instance_card}\n{model_card}\n.op\n.end\n"
        );
        let netlist = Parser::new()
            .parse_deck(&parse_deck_text(
                Path::new("bsim4-temp-extremes.cir"),
                &source,
            ))
            .unwrap();
        let instance = netlist.device("m1").unwrap();
        let models = ModelResolver::new(&netlist.models).unwrap();
        let parameters =
            bsim4::model_parameters(&models.resolve(instance).unwrap().unwrap()).unwrap();
        for temperature in [-35., 27., 150.] {
            let first = parameters
                .temperature_values(instance, temperature, 27.)
                .unwrap_or_else(|error| panic!("{label} at {temperature} C: {error}"));
            let again = parameters
                .temperature_values(instance, temperature, 27.)
                .unwrap();
            assert_eq!(first.0.len(), again.0.len(), "{label} model fields");
            for ((name, a), (same, b)) in first.0.iter().zip(&again.0) {
                assert_eq!(name, same);
                assert_eq!(
                    a.to_bits(),
                    b.to_bits(),
                    "{label} at {temperature} C: model {name}"
                );
            }
            assert_eq!(first.1.len(), again.1.len(), "{label} instance fields");
            for ((name, a), (same, b)) in first.1.iter().zip(&again.1) {
                assert_eq!(name, same);
                assert_eq!(
                    a.to_bits(),
                    b.to_bits(),
                    "{label} at {temperature} C: ask {name}"
                );
            }
            assert_eq!(first.2, again.2, "{label} diagnostic ordering");
            for quantity in ["leff", "weff", "cgbo"] {
                let value = first
                    .1
                    .iter()
                    .find(|(name, _)| *name == quantity)
                    .unwrap()
                    .1;
                assert!(
                    value.is_finite(),
                    "{label} at {temperature} C: {quantity} nonfinite"
                );
                if quantity != "cgbo" {
                    assert!(
                        value > 0.,
                        "{label} at {temperature} C: {quantity} nonpositive"
                    );
                }
            }
        }
    }
}
