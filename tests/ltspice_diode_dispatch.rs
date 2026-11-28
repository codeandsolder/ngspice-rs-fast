//! Independent LTspice source and physical dispatch witnesses.

use std::sync::Arc;

use anyhow::{Context, Result, ensure};
use ngspice_rs::{
    analysis::{Plot, RunConfig},
    devices::{Circuit, subckt::PreparedDeclarations},
    netlist::{Parser, ast::Netlist},
};
use ngspice_rs_preprocess::{Cancellation, InputFormat, Options, Request, SourceInput, prepare};

fn projected(source: &str) -> Result<(Netlist, RunConfig, Circuit)> {
    let dir = tempfile::tempdir()?;
    let bundle = prepare(
        &Request {
            format: InputFormat::LtspiceDeck,
            input: SourceInput::Bytes {
                origin: dir.path().join("source.cir"),
                bytes: Arc::from(source.as_bytes()),
            },
            libraries: Vec::new(),
            search_paths: Vec::new(),
        },
        &Options::default(),
        &Cancellation::default(),
    )?;
    for artifact in bundle.artifacts() {
        std::fs::write(dir.path().join(&artifact.filename), &artifact.bytes)?;
    }
    let deck = Parser::new().parse_file(dir.path().join(&bundle.primary.filename))?;
    let config = RunConfig::from_netlist(&deck)?;
    let circuit = config.prepared_circuit(
        &deck,
        PreparedDeclarations {
            waveforms: &bundle.waveforms,
            pwl_resistors: &bundle.pwl_resistors,
            ..Default::default()
        },
    )?;
    Ok((deck, config, circuit))
}

fn plot(source: &str) -> Result<Plot> {
    let (deck, config, mut circuit) = projected(source)?;
    Ok(config
        .plan_for(&deck.analyses[0])?
        .run(&mut circuit, &config.context())?)
}

fn close(name: &str, actual: f64, expected: f64, rel: f64, abs: f64) -> Result<()> {
    ensure!(
        (actual - expected).abs() <= rel.mul_add(expected.abs(), abs),
        "{name}: {actual:.17e} != {expected:.17e}"
    );
    Ok(())
}

#[test]
fn fourteen_source_models_select_the_independent_dc_laws() -> Result<()> {
    let measured = include_str!("fixtures/ltspice_diode/families-dc-compact.tsv");
    let plot = plot(include_str!("fixtures/ltspice_diode/families.cir"))?;
    ensure!(plot.point_count() == measured.lines().count() - 1);
    for (row, line) in measured.lines().skip(1).enumerate() {
        let data = line
            .split_ascii_whitespace()
            .map(str::parse::<f64>)
            .collect::<Result<Vec<_>, _>>()?;
        let [voltage, fn0, fn1, ideal0, ideal1] = data.as_slice() else {
            anyhow::bail!("invalid DC fixture width");
        };
        close("FN independent columns", *fn0, *fn1, 0., 0.)?;
        close("ideal independent columns", *ideal0, *ideal1, 0., 0.)?;
        close(
            "bias",
            plot.value("v(x)", row).context("missing bias")?.re,
            *voltage,
            0.,
            2e-12,
        )?;
        for index in 0..14 {
            let name = format!("i(v{index})");
            let current = plot.value(&name, row).with_context(|| name.clone())?.re;
            close(
                &name,
                current,
                if index < 12 { *fn0 } else { *ideal0 },
                3e-8,
                2e-13,
            )?;
        }
    }
    Ok(())
}

fn measured_ac(source: &str, measured: &str, count: usize) -> Result<()> {
    let plot = plot(source)?;
    let mut matched = 0;
    for field in measured.trim().split('\t').skip(1) {
        let (name, value) = field.split_once('=').context("invalid AC field")?;
        if !name.starts_with("I(V") || name == "I(Vx)" {
            continue;
        }
        let (real, imaginary) = value.split_once(',').context("invalid AC pair")?;
        let actual = plot
            .value(&name.to_ascii_lowercase(), 0)
            .with_context(|| name.to_owned())?;
        close(name, actual.re, real.parse()?, 3e-8, 1e-18)?;
        close(name, actual.im, imaginary.parse()?, 3e-8, 1e-18)?;
        matched += 1;
    }
    ensure!(matched == count);
    Ok(())
}

#[test]
fn rounded_boundaries_and_last_setter_dispatch() -> Result<()> {
    for (source, measured, count) in [
        (
            include_str!("fixtures/ltspice_diode/selector-ordered.cir"),
            include_str!("fixtures/ltspice_diode/selector-ordered-ac.tsv"),
            6,
        ),
        (
            include_str!("fixtures/ltspice_diode/selector-plain.cir"),
            include_str!("fixtures/ltspice_diode/selector-plain-ac.tsv"),
            15,
        ),
        (
            include_str!("fixtures/ltspice_diode/selector-ideal.cir"),
            include_str!("fixtures/ltspice_diode/selector-ideal-ac.tsv"),
            15,
        ),
    ] {
        measured_ac(source, measured, count)?;
    }
    Ok(())
}

#[test]
fn source_fn_noise_under_independent_current_bias() -> Result<()> {
    let measured = include_str!("fixtures/ltspice_diode/noise-measured.tsv");
    for line in include_str!("fixtures/ltspice_diode/noise-cases.tsv").lines() {
        let fields: Vec<_> = line.split('|').collect();
        let [name, bias, model, instance] = fields.as_slice() else {
            anyhow::bail!("invalid noise case");
        };
        let source = format!(
            "FN noise {name}\nI1 0 out DC {bias} AC 1\nD1 out 0 dm {instance}\n.model dm D(Level=2 {model})\n.options gmin=1e-12 reltol=1e-10 abstol=1e-14 vntol=1e-12\n.op\n.noise V(out) I1 lin 3 1k 1meg 1\n.end\n"
        );
        let (deck, config, mut circuit) = projected(&source)?;
        let context = config.context();
        let op = config
            .plan_for(&deck.analyses[0])?
            .run(&mut circuit, &context)?;
        let spectra = config
            .plan_for(&deck.analyses[1])?
            .run_plots(&mut circuit, &context)?;
        let spectrum = &spectra[0];
        let prefix = format!("noise-{name}\t");
        let mut rows = measured.lines().filter_map(|row| row.strip_prefix(&prefix));
        let expected = measured_value(rows.next().context("missing bias")?, "V(out)")?;
        close(
            name,
            op.value("v(out)", 0).context("missing bias voltage")?.re,
            expected,
            3e-8,
            1e-10,
        )?;
        let mut count = 0;
        for (index, row) in rows.enumerate() {
            close(
                name,
                spectrum
                    .value("frequency", index)
                    .context("missing frequency")?
                    .re,
                measured_value(row, "frequency")?,
                1e-12,
                0.,
            )?;
            for (column, field) in [
                ("onoise_spectrum", "V(onoise)"),
                ("inoise_spectrum", "inoise"),
            ] {
                close(
                    name,
                    spectrum
                        .value(column, index)
                        .with_context(|| column.to_owned())?
                        .re,
                    measured_value(row, field)?,
                    3e-7,
                    1e-22,
                )?;
            }
            count += 1;
        }
        ensure!(count == 3);
    }
    Ok(())
}

fn measured_value(line: &str, name: &str) -> Result<f64> {
    let (actual, value) = line
        .split('\t')
        .filter_map(|field| field.split_once('='))
        .find(|(key, _)| *key == name)
        .with_context(|| format!("missing measured {name}"))?;
    ensure!(actual == name);
    Ok(value.parse()?)
}

#[test]
fn fn_linear_ramp_stamps_terminal_charge() -> Result<()> {
    let plot = plot(include_str!("fixtures/ltspice_diode/transient.cir"))?;
    // Independent equation, validated against the raw vendor ramp samples.
    // The ramp is linear and the capacitance constant, so each solver sample
    // can be compared at its own accepted time without interpolating currents.
    let capacitance = 12. * 0.345_313_3 / (2. * 50.);
    let current = |voltage: f64, slope: f64| {
        let local = voltage / 2.;
        12. * local.signum() * 1e10 * (local / 50.).powi(2) * (-50. / local.abs()).exp()
            + capacitance * slope
    };
    let slope = |time: f64| {
        if time < 1. {
            1.
        } else if time < 2. {
            0.
        } else {
            -1.
        }
    };
    for row in include_str!("fixtures/ltspice_diode/transient-witness.tsv")
        .lines()
        .skip(1)
    {
        let data = row
            .split_ascii_whitespace()
            .map(str::parse::<f64>)
            .collect::<Result<Vec<_>, _>>()?;
        let [time, voltage, expected, _] = data.as_slice() else {
            anyhow::bail!("invalid transient reference");
        };
        close(
            "independent ramp",
            current(*voltage, slope(*time)),
            *expected,
            1e-9,
            1e-13,
        )?;
    }
    let mut count = 0;
    for row in 0..plot.point_count() {
        let time = plot.value("time", row).context("missing time")?.re;
        // Exact breakpoint samples use one-sided history. Compare only the
        // constant-slope interiors, rather than invent a breakpoint convention.
        if [0., 1., 2., 3.]
            .iter()
            .any(|knot| (time - knot).abs() < 0.01)
        {
            continue;
        }
        let voltage = plot.value("v(x)", row).context("missing voltage")?.re;
        let actual = -plot
            .value("i(vx)", row)
            .context("missing terminal current")?
            .re;
        close(
            "terminal ramp current",
            actual,
            current(voltage, slope(time)),
            1e-7,
            2e-8,
        )?;
        count += 1;
    }
    ensure!(count > 100);
    Ok(())
}

#[test]
fn ordinary_semiconductor_dispatch_keeps_the_measured_junction_and_charge() -> Result<()> {
    let source = include_str!("fixtures/ltspice_diode/ordinary.cir");
    let plot = plot(source)?;
    let measured = include_str!("fixtures/ltspice_diode/ordinary-dc.tsv");
    ensure!(plot.point_count() == measured.lines().count() - 1);
    for (row, line) in measured.lines().skip(1).enumerate() {
        let data = line
            .split_ascii_whitespace()
            .map(str::parse::<f64>)
            .collect::<Result<Vec<_>, _>>()?;
        ensure!(data.len() == 15);
        for (index, expected) in data[1..].iter().enumerate() {
            let name = format!("i(v{index})");
            let actual = plot.value(&name, row).with_context(|| name.clone())?.re;
            // LTspice solves an internal series node and reports branch current
            // with finite voltage accuracy; near leakage this introduces absolute
            // error from subtracting nearly equal voltages.
            close(&name, actual, *expected, 3e-7, 3e-14)?;
        }
    }
    measured_ac(
        include_str!("fixtures/ltspice_diode/ordinary-ac.cir"),
        include_str!("fixtures/ltspice_diode/ordinary-ac.tsv"),
        14,
    )
}

#[test]
fn sidewall_topologies_stamp_independent_series_nodes_and_charge() -> Result<()> {
    let cases: Vec<_> = include_str!("fixtures/ltspice_diode/side-cases.tsv")
        .lines()
        .collect();
    for row in include_str!("fixtures/ltspice_diode/side-ac-compact.tsv").lines() {
        let mut fields = row.split('\t');
        let bias = fields.next().context("missing bias")?;
        let expected: Vec<_> = fields.collect();
        ensure!(expected.len() == cases.len());
        let mut source = format!("sidewall AC\nVx x 0 DC {bias} AC 1\n");
        for (index, case) in cases.iter().enumerate() {
            let parts: Vec<_> = case.split('|').collect();
            let [_, model, instance] = parts.as_slice() else {
                anyhow::bail!("invalid sidewall case");
            };
            use std::fmt::Write as _;
            writeln!(
                source,
                "V{index} x a{index} 0\nD{index} a{index} 0 d{index} {instance}\n.model d{index} D({model})"
            )?;
        }
        source.push_str(
            ".options reltol=1e-10 abstol=1e-14 vntol=1e-12 gmin=1e-12\n.ac lin 1 1k 1k\n.end\n",
        );
        let plot = plot(&source)?;
        for (index, value) in expected.iter().enumerate() {
            let name = format!("i(v{index})");
            let (re, im) = value
                .split_once(',')
                .context("invalid complex measurement")?;
            let actual = plot.value(&name, 0).with_context(|| name.clone())?;
            close(&name, actual.re, re.parse()?, 3e-7, 1e-17)?;
            close(&name, actual.im, im.parse()?, 3e-7, 1e-17)?;
        }
    }
    Ok(())
}

#[test]
fn independent_sidewall_noise_sources_reach_the_terminal() -> Result<()> {
    let measured = include_str!("fixtures/ltspice_diode/noise-side-extra.tsv");
    for line in include_str!("fixtures/ltspice_diode/noise-side-extra-cases.tsv").lines() {
        let fields: Vec<_> = line.split('|').collect();
        let [name, model, instance] = fields.as_slice() else {
            anyhow::bail!("invalid semiconductor noise case");
        };
        let source = format!(
            "semiconductor noise {name}\nI1 0 out DC 5u AC 1\nD1 out 0 dm {instance}\n.model dm D({model})\n.options gmin=1e-12 reltol=1e-10 abstol=1e-14 vntol=1e-12\n.op\n.noise V(out) I1 lin 1 1k 1k 1\n.end\n"
        );
        if *name == "vp" {
            // Dynamic reverse recovery requires a separate state equation.
            // Issue #90 explicitly owns that independent physical law.
            ensure!(projected(&source).is_err());
            continue;
        }
        let (deck, config, mut circuit) = projected(&source)?;
        let context = config.context();
        let op = config
            .plan_for(&deck.analyses[0])?
            .run(&mut circuit, &context)?;
        let spectra = config
            .plan_for(&deck.analyses[1])?
            .run_plots(&mut circuit, &context)?;
        let spectrum = &spectra[0];
        let prefix = format!("noise-side-extra-{name}\t");
        let mut rows = measured.lines().filter_map(|row| row.strip_prefix(&prefix));
        let expected = measured_value(rows.next().context("missing bias")?, "V(out)")?;
        close(
            name,
            op.value("v(out)", 0).context("missing bias voltage")?.re,
            expected,
            3e-7,
            1e-10,
        )?;
        let noise = rows.next().context("missing noise measurement")?;
        for (column, field) in [
            ("onoise_spectrum", "V(onoise)"),
            ("inoise_spectrum", "inoise"),
        ] {
            close(
                name,
                spectrum
                    .value(column, 0)
                    .with_context(|| column.to_owned())?
                    .re,
                measured_value(noise, field)?,
                3e-7,
                1e-22,
            )?;
        }
    }
    Ok(())
}
