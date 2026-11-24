//! Circuit-coupled VDMOS noise spectral references from native ngspice 47.
//! Four physical generators are resolved from the prepared model; the solver
//! owns adjoint transfer, frequencies, ordering and integrated observations.
use std::path::Path;

use ngspice_rs::analysis::{Plot, RunConfig};
use ngspice_rs::netlist::{Parser, source::parse_deck_text};
use ngspice_rs::primitives::SpiceResult;

fn run_reference(source: &str) -> SpiceResult<Vec<Plot>> {
    let cards = source.split(".control").next().unwrap_or(source);
    let deck = parse_deck_text(
        Path::new("vdmos-noise.cir"),
        &format!("{cards}\n.noise v(drain) vdd dec 4 1k 1Meg 1\n.end\n"),
    );
    let parsed = Parser::new().parse_deck(&deck)?;
    let config = RunConfig::from_netlist(&parsed)?;
    let request = config.plan_for(&parsed.analyses[0])?;
    let mut circuit = config.circuit(&parsed)?;
    request.run_plots(&mut circuit, &config.context())
}

fn near(name: &str, actual: f64, reference: f64) {
    assert!(
        (actual - reference).abs() <= 0.006 * reference.abs(),
        "{name}: Rust {actual:e}, ngspice47 {reference:e}"
    );
}

#[test]
fn n_p_vdmos_noise_matches_c47_individual_generators_and_total() {
    // ngspice47 `vdmosnoi.c`: thermal RD, RS, channel, and 1/f.
    // A drain resistor adds its independent thermal generator. Both N and P
    // versions use mirrored biases and `vto` and have the same spectrum.
    let expected_at_1k = [
        ("onoise_spectrum", 5.929591e-11),
        ("inoise_spectrum", 2.288872e-9),
        ("onoise_m1_rd", 4.344423e-11),
        ("onoise_m1_rs", 3.830353e-11),
        ("onoise_m1_id", 7.060840e-12),
        ("onoise_m1_1overf", 5.838111e-13),
        ("onoise_m1", 5.835031e-11),
    ];
    let expected_at_1meg = [
        ("onoise_spectrum", 5.929086e-11),
        ("inoise_spectrum", 2.288769e-9),
        ("onoise_m1_rd", 4.344421e-11),
        ("onoise_m1_rs", 3.830037e-11),
        ("onoise_m1_id", 7.060471e-12),
        ("onoise_m1_1overf", 1.846076e-14),
        ("onoise_m1", 5.834525e-11),
    ];
    for (kind, source) in [
        ("N", include_str!("fixtures/vdmos/n_noise.cir")),
        ("P", include_str!("fixtures/vdmos/p_noise.cir")),
    ] {
        let plots = run_reference(source).unwrap_or_else(|error| panic!("{kind} noise: {error}"));
        assert_eq!(plots.len(), 2, "{kind}: both noise plots required");
        let spectrum = &plots[0];
        assert_eq!(spectrum.point_count(), 13);
        for (index, reference) in [(0_usize, &expected_at_1k), (12, &expected_at_1meg)] {
            for &(column, expected) in reference {
                let observed = spectrum
                    .value(column, index)
                    .unwrap_or_else(|| panic!("{kind}: missing {column}"))
                    .re;
                near(&format!("{kind} {column} at {index}"), observed, expected);
            }
        }
    }
}
