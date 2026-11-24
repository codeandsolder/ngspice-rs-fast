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

/// Independent ngspice 47 device noise at both ends of its 13-point sweep.
/// The four contributions must remain separate, preserving C model/instance
/// multiplicity and the bias-dependent temperature of the thermal variant.
#[test]
fn electrothermal_and_multiplicity_noise_matches_ngspice47() {
    let multiplier_at_1k = [
        ("onoise_spectrum", 2.894535e-11),
        ("inoise_spectrum", 4.599442e-9),
        ("onoise_m1_rd", 2.215948e-11),
        ("onoise_m1_rs", 1.838298e-11),
        ("onoise_m1_id", 1.508072e-12),
        ("onoise_m1_1overf", 1.315865e-13),
        ("onoise_m1", 2.883172e-11),
    ];
    let multiplier_at_1meg = [
        ("onoise_spectrum", 2.894437e-11),
        ("inoise_spectrum", 4.599407e-9),
        ("onoise_m1_rd", 2.215946e-11),
        ("onoise_m1_rs", 1.838196e-11),
        ("onoise_m1_id", 1.508021e-12),
        ("onoise_m1_1overf", 4.160990e-15),
        ("onoise_m1", 2.883075e-11),
    ];
    let electrothermal_at_1k = [
        ("onoise_spectrum", 5.987219e-11),
        ("inoise_spectrum", 2.298501e-9),
        ("onoise_m1_rd", 4.382065e-11),
        ("onoise_m1_rs", 3.871099e-11),
        ("onoise_m1_id", 7.284347e-12),
        ("onoise_m1_1overf", 5.967101e-13),
        ("onoise_m1", 5.892544e-11),
    ];
    let electrothermal_at_1meg = [
        ("onoise_spectrum", 5.987941e-11),
        ("inoise_spectrum", 2.295411e-9),
        ("onoise_m1_rd", 4.381886e-11),
        ("onoise_m1_rs", 3.871579e-11),
        ("onoise_m1_id", 7.330526e-12),
        ("onoise_m1_1overf", 1.898925e-14),
        ("onoise_m1", 5.892997e-11),
    ];
    for (name, source, expected_first, expected_last) in [
        (
            "NMOS m=4",
            include_str!("fixtures/vdmos/n_multiplier.cir"),
            &multiplier_at_1k,
            &multiplier_at_1meg,
        ),
        (
            "PMOS m=4",
            include_str!("fixtures/vdmos/p_multiplier.cir"),
            &multiplier_at_1k,
            &multiplier_at_1meg,
        ),
        (
            "NMOS junction+case",
            include_str!("fixtures/vdmos/n_thermal.cir"),
            &electrothermal_at_1k,
            &electrothermal_at_1meg,
        ),
        (
            "PMOS junction+case",
            include_str!("fixtures/vdmos/p_thermal.cir"),
            &electrothermal_at_1k,
            &electrothermal_at_1meg,
        ),
    ] {
        let plots = run_reference(source).unwrap_or_else(|error| panic!("{name} noise: {error}"));
        assert_eq!(plots.len(), 2, "{name} must provide both noise plots");
        let spectrum = &plots[0];
        assert_eq!(spectrum.point_count(), 13, "{name}");
        // ngspice47 `vdmosacld.c` has a known electrothermal AC sign
        // inconsistency with its own DC power Jacobian. At 1 kHz this
        // changes the individual *adjoint* noise transfers by up to ~1.3%
        // for the self-heating test. At 1 MHz thermal feedback decouples and
        // all seven C47 outputs agree to ~1e-7 relative. Validate that C
        // reference at high frequency, not C's incorrect thermal AC sign.
        // The finite-difference DC/AC consistency is tested separately in
        // `vdmos_prepared::electrothermal_small_signal_is_derivative_of_its_dc_residual`.
        let samples = if name.contains("junction+case") {
            &[(12, expected_last)][..]
        } else {
            &[(0, expected_first), (12, expected_last)][..]
        };
        for &(index, expected) in samples {
            for &(variable, reference) in expected {
                let observed = spectrum
                    .value(variable, index)
                    .unwrap_or_else(|| panic!("{name}: missing {variable} at {index}"))
                    .re;
                near(
                    &format!("{name} {variable} at {index}"),
                    observed,
                    reference,
                );
            }
        }
    }
}

#[test]
fn prepared_vdmos_noise_is_separate_for_instances_sharing_one_model() {
    // Two devices share a model but have different instance multipliers, so
    // neither the physical bias nor any generated noise source may be cached
    // solely by model ID. These are C47 source-by-source references.
    let spectrum = &run_reference(include_str!("fixtures/vdmos/n_two_instances.cir")).unwrap()[0];
    assert_eq!(spectrum.point_count(), 13);
    for (index, reference) in [
        (
            0,
            [
                ("onoise_spectrum", 2.894532e-11),
                ("inoise_spectrum", 4.599437e-9),
                ("onoise_m1_rd", 1.107974e-11),
                ("onoise_m1_rs", 9.191491e-12),
                ("onoise_m1_id", 7.540361e-13),
                ("onoise_m1_1overf", 5.727637e-14),
                ("onoise_m1", 1.441583e-11),
                ("onoise_m2_rd", 1.919067e-11),
                ("onoise_m2_rs", 1.592013e-11),
                ("onoise_m2_id", 1.306029e-12),
                ("onoise_m2_1overf", 1.107257e-13),
                ("onoise_m2", 2.496899e-11),
            ],
        ),
        (
            12,
            [
                ("onoise_spectrum", 2.894437e-11),
                ("inoise_spectrum", 4.599407e-9),
                ("onoise_m1_rd", 1.107973e-11),
                ("onoise_m1_rs", 9.190981e-12),
                ("onoise_m1_id", 7.540103e-13),
                ("onoise_m1_1overf", 1.811176e-15),
                ("onoise_m1", 1.441538e-11),
                ("onoise_m2_rd", 1.919065e-11),
                ("onoise_m2_rs", 1.591925e-11),
                ("onoise_m2_id", 1.305984e-12),
                ("onoise_m2_1overf", 3.501333e-15),
                ("onoise_m2", 2.496816e-11),
            ],
        ),
    ] {
        for (variable, expected) in reference {
            near(
                &format!("two-instance {variable} at {index}"),
                spectrum.value(variable, index).unwrap().re,
                expected,
            );
        }
    }
}

#[test]
fn prepared_vdmos_noise_with_hot_ambient_thermal_and_multiplicity_matches_c47() {
    // At TEMP=75C the junction equilibrates at 75.35269C in C47. M=4
    // scales conductances and thermal network; this covers the combined
    // dimensions rather than only either independent parameter.
    let spectrum =
        &run_reference(include_str!("fixtures/vdmos/n_thermal_multiplier_hot.cir")).unwrap()[0];
    for (index, reference) in [
        (
            0,
            [
                ("onoise_spectrum", 3.126840e-11),
                ("inoise_spectrum", 4.716335e-9),
                ("onoise_m1_rd", 2.386967e-11),
                ("onoise_m1_rs", 1.988229e-11),
                ("onoise_m1_id", 2.041578e-12),
                ("onoise_m1_1overf", 1.651474e-13),
                ("onoise_m1", 3.113297e-11),
            ],
        ),
        (
            12,
            [
                ("onoise_spectrum", 3.126726e-11),
                ("inoise_spectrum", 4.715844e-9),
                ("onoise_m1_rd", 2.386962e-11),
                ("onoise_m1_rs", 1.988114e-11),
                ("onoise_m1_id", 2.042306e-12),
                ("onoise_m1_1overf", 5.224281e-15),
                ("onoise_m1", 3.113181e-11),
            ],
        ),
    ] {
        for (variable, expected) in reference {
            near(
                &format!("75C thermal m4 {variable} at {index}"),
                spectrum.value(variable, index).unwrap().re,
                expected,
            );
        }
    }
}

#[test]
fn high_power_electrothermal_noise_starts_from_c47_dc_equilibrium() {
    // Independent DC reference ensures the noise adjoint is linearizing the
    // same *heated* circuit as C47, not silently computing the correct noise
    // formulas around an incorrect initial thermal operating point.
    for (name, source, reference) in [
        (
            "NMOS",
            include_str!("fixtures/vdmos/n_thermal.cir"),
            [32.31293, 31.25034, -1.169720, 0.3028034],
        ),
        (
            "PMOS",
            include_str!("fixtures/vdmos/p_thermal.cir"),
            [32.31293, 31.25034, 1.169720, -0.3028034],
        ),
        (
            "NMOS m4, ambient 75C",
            include_str!("fixtures/vdmos/n_thermal_multiplier_hot.cir"),
            [75.35269, 75.28215, -1.192111, 0.07889489],
        ),
    ] {
        let cards = source.split(".control").next().unwrap_or(source);
        let deck = parse_deck_text(
            Path::new("vdmos-thermal-noise-op.cir"),
            &format!("{cards}\n.op\n.end\n"),
        );
        let parsed = Parser::new().parse_deck(&deck).unwrap();
        let config = RunConfig::from_netlist(&parsed).unwrap();
        let request = config.plan_for(&parsed.analyses[0]).unwrap();
        let mut circuit = config.circuit(&parsed).unwrap();
        let plot = &request.run_plots(&mut circuit, &config.context()).unwrap()[0];
        for (variable, c47) in ["v(junction)", "v(case)", "i(vdd)", "v(drain)"]
            .into_iter()
            .zip(reference)
        {
            let measured = plot.value(variable, 0).unwrap().re;
            assert!(
                (measured - c47).abs() <= 0.0001 * c47.abs() + 1e-6,
                "{name} {variable}: Rust {measured:e}, ngspice47 {c47:e}",
            );
        }
    }
}
