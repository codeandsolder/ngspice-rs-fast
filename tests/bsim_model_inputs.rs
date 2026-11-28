//! Cross-family contract for ordered model and instance input dispatch.
use std::path::Path;

use ngspice_rs::devices::bsim3::{Inputs, Value};
use ngspice_rs::devices::{ModelResolver, bsim3, bsim4};
use ngspice_rs::netlist::{Parser, ast::Netlist, source::parse_deck_text};
use ngspice_rs::primitives::SpiceResult;

fn deck(level: u8, family: &str, model: &str, instance: &str) -> SpiceResult<Netlist> {
    Parser::new().parse_deck(&parse_deck_text(
        Path::new("ordered.cir"),
        &format!(
            "Ordered BSIM setters\nm1 d g s b nn {instance}\n.model nn {family} level={level} {model}\n.end\n"
        ),
    ))
}

fn prepare(netlist: &Netlist, level: u8, scale: f64) -> SpiceResult<(Inputs, i8, Inputs)> {
    let resolver = ModelResolver::new(&netlist.models)?;
    let device = &netlist.devices[0];
    let resolved = resolver.resolve(device)?.unwrap();
    if matches!(level, 8 | 49) {
        let model = bsim3::model_parameters(&resolved)?;
        let instance = model.instance_parameters(device, scale)?;
        Ok((model.inputs, model.polarity, instance))
    } else {
        let model = bsim4::model_parameters(&resolved)?;
        let instance = model.instance_parameters(device, scale)?;
        Ok((model.inputs, model.polarity, instance))
    }
}

fn close(inputs: &Inputs, name: &str, expected: f64) {
    let Value::Real(actual) = inputs.get(name).unwrap().value else {
        panic!("expected real setter {name}");
    };
    assert!((actual - expected).abs() <= expected.abs() * 1e-14 + 1e-30);
}

#[test]
fn interleaved_model_setters_and_defaults_preserve_order_and_provenance() {
    for level in [8, 49, 14, 54] {
        let netlist = deck(
            level,
            "nmos",
            "vth0=0.3 l=1u VTHO=0.4 l=2u w=4u m=7 pmos nmos pmos",
            "w=5u m=2 ic=(0.1 0.2 -0.3) icvgs=0.4 off",
        )
        .unwrap();
        let before = netlist.clone();
        let (model, polarity, instance) = prepare(&netlist, level, 2.).unwrap();
        assert_eq!(polarity, -1);
        close(&model, "vth0", 0.4);
        close(&model, "vtho", 0.4);
        assert_eq!(
            model
                .applied()
                .iter()
                .map(|p| p.parameter.name)
                .collect::<Vec<_>>(),
            ["vth0", "vtho", "pmos", "nmos", "pmos"]
        );
        assert_eq!(
            model.applied()[1].location,
            netlist.models[0].parameters[3].location
        );
        assert_eq!(
            instance
                .applied()
                .iter()
                .map(|p| p.parameter.name)
                .collect::<Vec<_>>(),
            [
                "w", "l", "l", "w", "m", "icvbs", "icvgs", "icvds", "icvgs", "off"
            ]
        );
        for (actual, source) in instance.applied()[..3].iter().zip([5, 4, 2]) {
            assert_eq!(
                actual.location,
                netlist.models[0].parameters[source].location
            );
        }
        close(&instance, "l", 2e-6);
        close(&instance, "w", 10e-6);
        close(&instance, "m", 2.);
        close(&instance, "icvds", 0.1);
        close(&instance, "icvgs", 0.4);
        close(&instance, "icvbs", -0.3);
        assert_eq!(instance.get("off").unwrap().value, Value::Flag(true));
        assert_eq!(netlist, before);
    }
}

#[test]
fn family_polarity_multiplier_and_errors_remain_explicit() {
    for level in [8, 49, 14, 54] {
        let netlist = deck(level, "pmos", "m=7", "").unwrap();
        let (model, polarity, instance) = prepare(&netlist, level, 1.).unwrap();
        assert_eq!(polarity, -1);
        assert!(model.applied().is_empty());
        assert!(instance.applied().is_empty());
        for scale in [0., -1., f64::NAN, f64::INFINITY] {
            assert!(prepare(&netlist, level, scale).is_err());
        }
        let override_model = deck(level, "pmos", "nmos", "").unwrap();
        assert_eq!(prepare(&override_model, level, 1.).unwrap().1, 1);
        for invalid in ["m={missing}", "gm=1", "bogus=1", "tnom=-273.15"] {
            let result =
                deck(level, "nmos", invalid, "").and_then(|netlist| prepare(&netlist, level, 1.));
            assert!(result.is_err(), "level {level}, {invalid}");
        }
    }
}
