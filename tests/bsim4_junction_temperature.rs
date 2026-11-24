//! End-to-end BSIM4 junction temperature warning behavior through the public
//! prepared-model API; the independent pre-extraction arithmetic oracle lives
//! beside the pure temperature kernel.
use std::path::Path;

use ngspice_rs::devices::{ModelResolver, bsim4};
use ngspice_rs::netlist::{Parser, source::parse_deck_text};

const SOURCE_WARNING: &str = "Warning: ijthsrev too small and set to 10 times IsbSat.";
const DRAIN_WARNING: &str = "Warning: ijthdrev too small and set to 10 times IdbSat.";

fn junction_warnings(
    polarity: &str,
    mode: i32,
    source_geometry: (&str, &str),
    drain_geometry: (&str, &str),
    temperature: f64,
    reverse_threshold: f64,
) -> Vec<String> {
    let (area_source, perimeter_source) = source_geometry;
    let (area_drain, perimeter_drain) = drain_geometry;
    let deck = format!(
        "BSIM4 junction temperature warning order\n\
         vdd drain 0 1\n\
         vgate gate 0 0.8\n\
         vsource source 0 0\n\
         vbody bulk 0 0\n\
         m1 drain gate source bulk nn l=0.2u w=2u nf=3 \
             as={area_source} ps={perimeter_source} \
             ad={area_drain} pd={perimeter_drain}\n\
         .model nn {polarity} level=54 version=4.8.3 diomod={mode} \
             jss=1e-4 jsd=2e-4 jsws=1e-10 jswd=2e-10 \
             ijthsrev={reverse_threshold:.17e} ijthdrev={reverse_threshold:.17e} \
             xjbvs=2 xjbvd=3 bvs=10 bvd=12\n\
         .end\n"
    );
    let netlist = Parser::new()
        .parse_deck(&parse_deck_text(
            Path::new("bsim4-junction-temp.cir"),
            &deck,
        ))
        .unwrap();
    let device = netlist.device("m1").unwrap();
    let resolver = ModelResolver::new(&netlist.models).unwrap();
    let model = bsim4::model_parameters(&resolver.resolve(device).unwrap().unwrap()).unwrap();
    let (_, _, warnings) = model.temperature_values(device, temperature, 27.).unwrap();
    warnings
        .into_iter()
        .filter(|warning| {
            warning.contains("ijthsrev too small") || warning.contains("ijthdrev too small")
        })
        .collect()
}

#[test]
fn source_warning_precedes_drain_for_both_polarities_and_temperatures() {
    for polarity in ["nmos", "pmos"] {
        for temperature in [-20., 27., 85.] {
            let warnings =
                junction_warnings(polarity, 2, ("1p", "2u"), ("3p", "4u"), temperature, 1e-30);
            assert_eq!(
                warnings,
                [SOURCE_WARNING, DRAIN_WARNING],
                "{polarity} at {temperature} C"
            );
        }
    }
}

#[test]
fn inactive_junction_does_not_emit_a_reverse_warning() {
    let source_only = junction_warnings("nmos", 2, ("1p", "2u"), ("0", "0"), 27., 1e-30);
    assert_eq!(source_only, [SOURCE_WARNING]);
    let drain_only = junction_warnings("nmos", 2, ("0", "0"), ("3p", "4u"), 27., 1e-30);
    assert_eq!(drain_only, [DRAIN_WARNING]);
}

#[test]
fn other_diode_modes_and_large_reverse_threshold_are_quiet() {
    for mode in [0, 1, 2] {
        for polarity in ["nmos", "pmos"] {
            assert!(
                junction_warnings(polarity, mode, ("1p", "2u"), ("3p", "4u"), 27., 1.,).is_empty(),
                "{polarity} diode mode {mode}"
            );
        }
    }
}
