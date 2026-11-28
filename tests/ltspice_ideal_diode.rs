//! Independent `LTspice` 24.0.12 DC observations from the synthetic model
//! witness in issue #226, and analytic small-signal/transient checks.
use std::path::Path;

use ngspice_rs::{
    analysis::{AnalysisContext, AnalysisInput, bias::solve_dc, newton::NewtonOptions},
    devices::{Circuit, ModelContext, nonlinear::Diode},
    netlist::{Parser, ast::Netlist, source::parse_deck_text},
    primitives::{AnalysisKind, Real},
};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

fn parse(body: &str) -> Result<Netlist> {
    Ok(Parser::new().parse_deck(&parse_deck_text(Path::new("ideal-diode.cir"), body))?)
}

fn close(actual: Real, expected: Real, abs: Real, rel: Real) {
    assert!(
        (actual - expected).abs() <= rel.mul_add(expected.abs(), abs),
        "{actual:e} != {expected:e}"
    );
}

fn dc_current(circuit: &Circuit, context: &ModelContext, source: &str) -> Result<Real> {
    let options = NewtonOptions {
        reltol: 1e-10,
        abstol: 1e-14,
        vntol: 1e-10,
        ..NewtonOptions::default()
    };
    let solved = solve_dc(circuit, context, &options, &[], None, None)?;
    let index = circuit
        .devices()
        .iter()
        .position(|device| device.name() == source)
        .ok_or("source")?;
    let row = circuit.branch_rows(index).ok_or("source branch")?.start;
    Ok(-solved.values.get(row).ok_or("source current")?)
}

const BASE: &str = "ron=2 roff=100 vfwd=1 vrev=2 rrev=3";

#[test]
fn independent_measured_dc_points() -> Result<()> {
    let cases: &[(&str, &[(Real, Real)])] = &[
        (
            "",
            &[
                (1., 0.01),
                (1.2, 0.11),
                (2., 0.51),
                (-2., -0.02),
                (-3., -0.353_333_333_333),
            ],
        ),
        (
            "epsilon=0.4 revepsilon=0.6",
            &[
                (1.2, 0.0365),
                (1.4, 0.112),
                (2., 0.412),
                (-2.4, -0.067_111_111_111_1),
                (-3., -0.256_333_333_333),
            ],
        ),
        (
            "ilimit=0.25 revilimit=0.3",
            &[
                (1.2, 0.104_987_240_564),
                (2., 0.251_006_895_019),
                (4., 0.259_996_927_913),
                (-3., -0.261_336_440_09),
            ],
        ),
        (
            "epsilon=0.4 revepsilon=0.6 ilimit=0.25 revilimit=0.3",
            &[
                (1.2, 0.033_313_491_749_9),
                (1.4, 0.099_253_966_999_7),
                (2., 0.239_393_660_113),
                (-2.4, -0.062_810_270_350_3),
                (-3., -0.213_411_129_383),
            ],
        ),
    ];
    for (extra, points) in cases {
        for &(voltage, expected) in *points {
            let deck = parse(&format!(
                "ideal diode\nV1 a 0 {voltage}\nD1 a 0 dm\n.model dm ltspice_diode({BASE} {extra})\n.end\n"
            ))?;
            let context = ModelContext::default().with_gmin(1e-12);
            let circuit = Circuit::from_netlist_with_context(&deck, &context)?;
            close(
                dc_current(&circuit, &context, "v1")?,
                expected,
                1e-12,
                1e-10,
            );
        }
    }
    Ok(())
}

#[test]
fn small_signal_matches_the_tangent_and_has_no_storage() -> Result<()> {
    for (extra, voltage, expected) in [
        ("", 1.2, 0.5),
        ("epsilon=0.4", 1.2, 0.255),
        ("ilimit=0.25", 1.2, 0.5 / 0.4_f64.cosh().powi(2)),
        ("ilimit=0.25 epsilon=0.4", 1.2, 0.223_134_917_499_262_4),
        ("ilimit=0.25 epsilon=0.4", 1.4, 0.436_269_834_998_524_7),
        (
            "revepsilon=0.6 revilimit=0.3",
            -2.6,
            0.301_077_027_627_483_34,
        ),
    ] {
        let deck = parse(&format!(
            "AC\nV1 a 0 DC {voltage} AC 1\nD1 a 0 dm\n.model dm ltspice_diode({BASE} {extra})\n.end\n"
        ))?;
        let context = AnalysisContext {
            gmin: 1e-12,
            ..AnalysisContext::default()
        };
        let mut circuit = Circuit::from_netlist_with_context(&deck, &context.model_context())?;
        let plots = AnalysisInput::with_arguments(AnalysisKind::Ac, ["lin", "3", "1k", "1g"])
            .resolve()?
            .run_plots(&mut circuit, &context)?;
        assert_eq!(plots[0].point_count(), 3);
        for point in 0..plots[0].point_count() {
            let current = plots[0].value("i(v1)", point).ok_or("AC current")?;
            close(-current.re, expected, 1e-11, 1e-10);
            close(current.im, 0., 1e-14, 0.);
        }
    }
    Ok(())
}

#[test]
fn source_family_and_geometry() -> Result<()> {
    let native = Circuit::from_netlist(&parse("native\nD1 a 0 dm\n.model dm D(IS=1p)\n.end\n")?)?;
    assert!((native.devices()[0].as_ref() as &dyn std::any::Any).is::<Diode>());
    let deck = parse(
        "geometry\nV1 a 0 2\nV2 b 0 4\nV3 c 0 -3\nD1 a 0 dm m=3\nD2 b 0 dm m=3 n=2\nD3 c 0 other\n.model dm ltspice_diode(ron=2 roff=100 vfwd=1)\n.model other ltspice_diode(ron=2 roff=100 vrev=2)\n.end\n",
    )?;
    let context = ModelContext::default();
    let circuit = Circuit::from_netlist_with_context(&deck, &context)?;
    close(dc_current(&circuit, &context, "v1")?, 1.53, 1e-12, 1e-10);
    close(dc_current(&circuit, &context, "v2")?, 1.53, 1e-12, 1e-10);
    close(dc_current(&circuit, &context, "v3")?, -0.52, 1e-12, 1e-10);
    assert_eq!(
        circuit
            .devices()
            .iter()
            .find(|d| d.name() == "d1")
            .and_then(|d| d.model_name()),
        Some("dm")
    );
    Ok(())
}

#[test]
fn default_off_conductance_rebinds_with_gmin() -> Result<()> {
    let deck =
        parse("default leakage\nV1 a 0 -1\nD1 a 0 dm\n.model dm ltspice_diode(ron=2)\n.end\n")?;
    let context = ModelContext::default().with_gmin(1e-12);
    let circuit = Circuit::from_netlist_with_context(&deck, &context)?;
    // The independently measured parallel floor is Gmin/1000.
    close(dc_current(&circuit, &context, "v1")?, -1.001e-12, 0., 1e-10);
    let other = context.with_gmin(1e-5);
    close(dc_current(&circuit, &other, "v1")?, -1.001e-5, 0., 1e-10);
    Ok(())
}

#[test]
fn invalid_model_and_instance_fields_are_not_ignored() -> Result<()> {
    for parameters in ["ron=2 kf=-1p", "ron=2 unknown_parameter=1p"] {
        let deck = parse(&format!(
            "bad\nD1 a 0 dm\n.model dm ltspice_diode({parameters})\n.end\n"
        ))?;
        assert!(Circuit::from_netlist(&deck).is_err(), "{parameters}");
    }
    for fields in ["area=-1", "n=0", "temp=-300", "dtemp=1"] {
        let deck = parse(&format!(
            "bad\nD1 a 0 dm {fields}\n.model dm ltspice_diode(ron=2)\n.end\n"
        ))?;
        assert!(Circuit::from_netlist(&deck).is_err(), "{fields}");
    }
    Ok(())
}

#[test]
fn transient_has_exact_memoryless_current_through_both_knees() -> Result<()> {
    let deck = parse(
        "transient\nV1 a 0 PWL(0 -4 1u -4 3u 4 4u 4 6u -4)\nD1 a 0 dm\n.model dm ltspice_diode(ron=2 roff=100 vfwd=1 vrev=2 rrev=3 epsilon=0.4 revepsilon=0.6)\n.end\n",
    )?;
    let context = AnalysisContext {
        gmin: 1e-12,
        ..AnalysisContext::default()
    };
    let mut circuit = Circuit::from_netlist_with_context(&deck, &context.model_context())?;
    let plots = AnalysisInput::with_arguments(
        AnalysisKind::Transient,
        [
            "10n",
            "6u",
            "0",
            "10n",
            "rtol=1e-10",
            "abstol=1e-14",
            "vntol=1e-12",
        ],
    )
    .resolve()?
    .run_plots(&mut circuit, &context)?;
    let plot = &plots[0];
    assert!(plot.point_count() > 100);
    for row in 0..plot.point_count() {
        let voltage = plot.value("v(a)", row).ok_or("voltage")?.re;
        let current = -plot.value("i(v1)", row).ok_or("current")?.re;
        // Independent closed-form piecewise quadratic law, both ramp
        // directions and plateaus. No dynamic charge term is present.
        let expected = if voltage > 1.4 {
            (voltage - 1.4) / 2. + 0.112
        } else if voltage > 1. {
            0.6125_f64.mul_add((voltage - 1.).powi(2), 0.01 * voltage)
        } else if voltage < -2.6 {
            (voltage + 2.6) / 3. - 0.123
        } else if voltage < -2. {
            0.01_f64.mul_add(voltage, -((1. / 3. - 0.01) * (voltage + 2.).powi(2) / 1.2))
        } else {
            0.01 * voltage
        };
        close(current, expected, 2e-9, 1e-8);
    }
    Ok(())
}

#[test]
fn independent_c_common_regions_dc_sweep() -> Result<()> {
    let deck = parse(&format!(
        "common regions\nV1 a 0 0\nD1 a 0 dm\n.model dm ltspice_diode({BASE})\n.end\n"
    ))?;
    let context = AnalysisContext {
        gmin: 1e-12,
        ..AnalysisContext::default()
    };
    let mut circuit = Circuit::from_netlist_with_context(&deck, &context.model_context())?;
    let plots = AnalysisInput::with_arguments(AnalysisKind::DcSweep, ["v1", "-4", "4", "0.1"])
        .resolve()?
        .run_plots(&mut circuit, &context)?;
    let plot = &plots[0];
    let witness = include_str!("fixtures/ltspice_ideal_diode/common.tsv");
    assert_eq!(plot.point_count(), witness.lines().count());
    for (row, line) in witness.lines().enumerate() {
        let (voltage, current) = line.split_once('\t').ok_or("malformed witness")?;
        close(
            plot.value("v(a)", row).ok_or("voltage")?.re,
            voltage.parse()?,
            1e-12,
            1e-10,
        );
        close(
            -plot.value("i(v1)", row).ok_or("current")?.re,
            current.parse()?,
            1e-12,
            1e-10,
        );
    }
    Ok(())
}

#[test]
fn current_driven_limited_branches() -> Result<()> {
    for (current, expected) in [
        (0.2, 0.5_f64.mul_add((0.19_f64 / 0.25).atanh(), 1.)),
        (-0.25, 0.9_f64.mul_add(-(0.23_f64 / 0.3).atanh(), -2.)),
    ] {
        let deck = parse(&format!(
            "current drive\nI1 0 a {current}\nD1 a 0 dm\n.model dm ltspice_diode({BASE} ilimit=0.25 revilimit=0.3)\n.end\n"
        ))?;
        let context = AnalysisContext {
            gmin: 1e-12,
            ..AnalysisContext::default()
        };
        let mut circuit = Circuit::from_netlist_with_context(&deck, &context.model_context())?;
        let plots = AnalysisInput::new(AnalysisKind::OperatingPoint)
            .resolve()?
            .run_plots(&mut circuit, &context)?;
        close(
            plots[0].value("v(a)", 0).ok_or("voltage")?.re,
            expected,
            1e-8,
            1e-7,
        );
    }
    Ok(())
}

#[test]
fn vendor_threshold_and_level_selection() -> Result<()> {
    for line in include_str!("fixtures/ltspice_ideal_diode/ac-witness.tsv").lines() {
        let fields: Vec<_> = line.split('\t').collect();
        let [name, bias, instance, model, gmin, conductance] = fields.as_slice() else {
            return Err("invalid AC witness row".into());
        };
        let text = format!(
            "{name}\nV1 a 0 DC {bias} AC 1\nD1 a 0 dm {instance}\n.model dm ltspice_diode({model})\n.options gmin={gmin}\n.ac lin 3 1k 1meg\n.end\n"
        );
        let deck = parse(&text)?;
        let config = ngspice_rs::analysis::RunConfig::from_netlist(&deck)?;
        let plot = config
            .plan_for(&deck.analyses[0])?
            .run(&mut config.circuit(&deck)?, &config.context())?;
        for row in 0..plot.point_count() {
            let current = plot.value("i(v1)", row).ok_or("AC current")?;
            close(-current.re, conductance.parse()?, 0., 1e-10);
            close(current.im, 0., 0., 0.);
        }
    }
    Ok(())
}

#[test]
fn vendor_parallel_floor_across_all_branches() -> Result<()> {
    let rows: Vec<_> = include_str!("fixtures/ltspice_ideal_diode/floor-witness.tsv")
        .lines()
        .collect();
    for block in rows.chunks(17) {
        let first: Vec<_> = block[0].split('\t').collect();
        let [name, model, instance, ..] = first.as_slice() else {
            return Err("invalid DC witness row".into());
        };
        let text = format!(
            "{name}\nV1 a 0 0\nD1 a 0 dm {instance}\n.model dm ltspice_diode({model})\n.options gmin=1e-12 reltol=1e-10 abstol=1e-24 vntol=1e-12\n.dc V1 -4 4 0.5\n.end\n"
        );
        let deck = parse(&text)?;
        let config = ngspice_rs::analysis::RunConfig::from_netlist(&deck)?;
        let plot = config
            .plan_for(&deck.analyses[0])?
            .run(&mut config.circuit(&deck)?, &config.context())?;
        assert_eq!(plot.point_count(), block.len());
        for (row, line) in block.iter().enumerate() {
            let fields: Vec<_> = line.split('\t').collect();
            let [_, _, _, voltage, current] = fields.as_slice() else {
                return Err("invalid DC witness row".into());
            };
            close(
                plot.value("v(a)", row).ok_or("voltage")?.re,
                voltage.parse()?,
                1e-13,
                1e-12,
            );
            close(
                -plot.value("i(v1)", row).ok_or("current")?.re,
                current.parse()?,
                1e-25,
                1e-9,
            );
        }
    }
    Ok(())
}

#[test]
fn vendor_noise_spectrum_and_geometry() -> Result<()> {
    let rows: Vec<_> = include_str!("fixtures/ltspice_ideal_diode/noise-witness.tsv")
        .lines()
        .collect();
    for block in rows.chunks(3) {
        let first: Vec<_> = block[0].split('\t').collect();
        let [
            name,
            bias,
            model,
            instance,
            gmin,
            temp,
            voltage,
            _,
            impedance,
            ..,
        ] = first.as_slice()
        else {
            return Err("invalid noise witness row".into());
        };
        // Preserve the deliberately tiny conductance probes without the
        // default absolute sparse pivot threshold masking their physics.
        let pivot = if impedance.parse::<Real>()? > 1e12 {
            "pivtol=1e-30"
        } else {
            ""
        };
        let text = format!(
            "{name}\nI1 0 a DC {bias} AC 1\nD1 a 0 dm {instance}\n.model dm ltspice_diode({model})\n.options gmin={gmin} temp={temp} reltol=1e-10 abstol=1e-18 vntol=1e-12 {pivot}\n.op\n.ac lin 3 1k 1meg\n.noise V(a) I1 lin 3 1k 1meg 1\n.end\n"
        );
        let deck = parse(&text)?;
        let config = ngspice_rs::analysis::RunConfig::from_netlist(&deck)?;
        let mut circuit = config.circuit(&deck)?;
        let context = config.context();
        let op = config
            .plan_for(&deck.analyses[0])?
            .run(&mut circuit, &context)?;
        close(
            op.value("v(a)", 0).ok_or("OP voltage")?.re,
            voltage.parse()?,
            1e-10,
            1e-8,
        );
        let ac = config
            .plan_for(&deck.analyses[1])?
            .run(&mut circuit, &context)?;
        let plots = config
            .plan_for(&deck.analyses[2])?
            .run_plots(&mut circuit, &context)?;
        let spectrum = &plots[0];
        assert_eq!(spectrum.point_count(), block.len(), "{name}");
        for (row, line) in block.iter().enumerate() {
            let fields: Vec<_> = line.split('\t').collect();
            let [
                _,
                _,
                _,
                _,
                _,
                _,
                _,
                frequency,
                impedance,
                thermal,
                shot,
                flicker,
                total,
                input,
            ] = fields.as_slice()
            else {
                return Err("invalid noise witness row".into());
            };
            close(
                spectrum.value("frequency", row).ok_or("frequency")?.re,
                frequency.parse()?,
                1e-10,
                1e-12,
            );
            close(
                ac.value("v(a)", row).ok_or("AC impedance")?.re,
                impedance.parse()?,
                0.,
                1e-8,
            );
            for (column, expected) in [
                ("onoise_d1_rs", thermal),
                ("onoise_d1_id", shot),
                ("onoise_d1_1overf", flicker),
                ("onoise_d1", total),
                ("onoise_spectrum", total),
                ("inoise_spectrum", input),
            ] {
                let expected: Real = expected.parse()?;
                let got = spectrum.value(column, row).ok_or(column)?.re;
                // The two programs use slightly different physical constants.
                assert!(
                    (got - expected).abs() <= 2e-6 * expected.abs(),
                    "{name}: {column}: {got:.17e} != {expected:.17e}"
                );
            }
        }
    }
    Ok(())
}

#[test]
fn vendor_signed_parameters_defaults_and_unmatched_transitions() -> Result<()> {
    use std::fmt::Write as _;
    for block in include_str!("fixtures/ltspice_ideal_diode/edge-witness.tsv").split("\n\n") {
        let mut lines = block.lines();
        let (name, model) = lines
            .next()
            .ok_or("edge case header")?
            .split_once('\t')
            .ok_or("edge case header fields")?;
        let points = lines
            .map(|line| line.split_once('\t').ok_or("edge witness columns"))
            .collect::<std::result::Result<Vec<_>, _>>()?;
        // Independent voltage drives test the exact sampled biases, not a
        // regenerated floating-point sweep grid. This matters at a vendor
        // discontinuity: an ulp on either side can change current sharply.
        let mut text = format!(
            "{name}\n.model dm ltspice_diode({model})\n.options gmin=1e-12 reltol=1e-10 abstol=1e-14 vntol=1e-12\n"
        );
        for (index, (voltage, _)) in points.iter().enumerate() {
            writeln!(
                text,
                "V{index} a{index} 0 {voltage}\nD{index} a{index} 0 dm"
            )?;
        }
        text.push_str(".op\n.end\n");
        let deck = parse(&text)?;
        let config = ngspice_rs::analysis::RunConfig::from_netlist(&deck)?;
        let plot = config
            .plan_for(&deck.analyses[0])?
            .run(&mut config.circuit(&deck)?, &config.context())?;
        for (index, (voltage, current)) in points.iter().enumerate() {
            let expected: Real = current.parse()?;
            let actual = -plot
                .value(&format!("i(v{index})"), 0)
                .ok_or("edge diode current")?
                .re;
            assert!(
                (actual - expected).abs() <= 1e-9_f64.mul_add(expected.abs(), 2e-12),
                "{name} at {voltage}: {actual:.17e} != {expected:.17e}"
            );
        }
    }
    Ok(())
}

#[test]
fn vendor_signed_noise() -> Result<()> {
    // LTspice 24.0.12, voltage drive through 0.1 ohm. Independent OP,
    // gain and noise amplitudes at 1 kHz and 1 MHz.
    let rows = [
        (
            "ron=-2 roff=100 vfwd=1",
            2.,
            2.051_578_947_368_420_7,
            1.052_631_578_947_368_1,
            9.583_020_389_645_711e-12,
            4.279_399_222_815_901e-11,
            0.,
            0.,
        ),
        (
            "ron=2 roff=-100 vfwd=1",
            0.5,
            0.500_500_500_500_500_4,
            1.001_001_001_001_001,
            1.288_770_323_659_408_8e-12,
            4.008_731_246_280_551e-12,
            0.,
            0.,
        ),
        (
            "ron=-2 roff=-100 vfwd=1",
            2.,
            2.053_684_210_526_315_6,
            1.052_631_578_947_368_1,
            9.583_020_389_645_711e-12,
            4.365_860_466_700_237e-11,
            0.,
            0.,
        ),
        (
            "ron=2 roff=100 vfwd=1 kf=1e-10 af=-1",
            2.,
            1.951_428_571_428_570_8,
            0.952_380_952_380_952_2,
            8.670_351_781_108_042e-12,
            3.757_260_534_890_933e-11,
            4.321_358_157_014_43e-8,
            1.366_533_436_151_313e-9,
        ),
    ];
    for (model, bias, voltage, gain, thermal, shot, flicker_low, flicker_high) in rows {
        let deck = parse(&format!(
            "signed noise\nV1 in 0 DC {bias} AC 1\nR1 in out 0.1\nD1 out 0 dm\n.model dm ltspice_diode({model})\n.options gmin=1e-12 temp=27 reltol=1e-10 abstol=1e-18 vntol=1e-12\n.op\n.ac lin 3 1k 1meg\n.noise V(out) V1 lin 3 1k 1meg 1\n.end\n"
        ))?;
        let config = ngspice_rs::analysis::RunConfig::from_netlist(&deck)?;
        let mut circuit = config.circuit(&deck)?;
        let context = config.context();
        let op = config
            .plan_for(&deck.analyses[0])?
            .run(&mut circuit, &context)?;
        close(
            op.value("v(out)", 0).ok_or("OP voltage")?.re,
            voltage,
            1e-10,
            1e-8,
        );
        let ac = config
            .plan_for(&deck.analyses[1])?
            .run(&mut circuit, &context)?;
        let plots = config
            .plan_for(&deck.analyses[2])?
            .run_plots(&mut circuit, &context)?;
        let spectrum = &plots[0];
        for (row, flicker) in [(0, flicker_low), (2, flicker_high)] {
            close(ac.value("v(out)", row).ok_or("AC gain")?.re, gain, 0., 1e-8);
            for (column, expected) in [
                ("onoise_d1_rs", thermal),
                ("onoise_d1_id", shot),
                ("onoise_d1_1overf", flicker),
            ] {
                close(
                    spectrum.value(column, row).ok_or(column)?.re,
                    expected,
                    0.,
                    2e-6,
                );
            }
        }
    }
    Ok(())
}

#[test]
fn vendor_instance_geometry_boundaries() -> Result<()> {
    // Independently measured LTspice 24.0.12 AC and OP, imposed 2 V.
    // Zero/negative M or zero AREA disables the complete instance, even
    // its Gmin floor; a negative nonzero N reverses local voltage.
    for (fields, current, conductance) in [
        ("m=0", 0., 0.),
        ("m=-2", 0., 0.),
        ("area=0", 0., 0.),
        ("m=.5", 0.255_000_000_000_002, 0.250_000_000_000_001),
        ("n=-2", -0.010_000_000_000_001, -0.005_000_000_000_000_5),
        ("n=.5", 1.510_000_000_000_004, 1.000_000_000_000_002),
    ] {
        let deck = parse(&format!(
            "geometry boundary\nV1 a 0 DC 2 AC 1\nD1 a 0 dm {fields}\n.model dm ltspice_diode(ron=2 roff=100 vfwd=1 vrev=2 rrev=3)\n.options gmin=1e-12 reltol=1e-10 abstol=1e-18 vntol=1e-12\n.op\n.ac lin 3 1k 1meg\n.noise V(a) V1 lin 3 1k 1meg 1\n.end\n"
        ))?;
        let config = ngspice_rs::analysis::RunConfig::from_netlist(&deck)?;
        let mut circuit = config.circuit(&deck)?;
        let context = config.context();
        let op = config
            .plan_for(&deck.analyses[0])?
            .run(&mut circuit, &context)?;
        close(
            -op.value("i(v1)", 0).ok_or("OP source current")?.re,
            current,
            1e-13,
            1e-10,
        );
        let ac = config
            .plan_for(&deck.analyses[1])?
            .run(&mut circuit, &context)?;
        close(
            -ac.value("i(v1)", 0).ok_or("AC source current")?.re,
            conductance,
            1e-13,
            1e-10,
        );
        if fields == "n=-2" {
            // Vendor exposes NaN for the negative shot PSD. Diagnose it
            // instead of silently turning it into a positive density.
            assert!(
                config
                    .plan_for(&deck.analyses[2])?
                    .run_plots(&mut circuit, &context)
                    .is_err()
            );
        }
    }
    Ok(())
}
