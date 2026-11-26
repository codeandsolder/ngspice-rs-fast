//! Independently measured source-prepared passive initialization witnesses.
//! Electrical fixtures and CSV values are pinned to the producer revision
//! recorded in fixtures/passive-ic/README.md. No circuit is hand rewritten.
use std::{fs, path::Path};

use ngspice_rs::{
    analysis::{Plot, RunConfig},
    netlist::{Parser, ast::Netlist, prepared_initialization::PreparedInitialConditions},
    primitives::{AnalysisKind, SpiceResult},
};
use ngspice_rs_preprocess::{Cancellation, InputFormat, Options, Request, SourceInput, prepare};

fn prepared(path: &Path) -> SpiceResult<Netlist> {
    let bundle = prepare(
        &Request {
            format: InputFormat::SimetrixDeck,
            input: SourceInput::File(path.to_path_buf()),
            libraries: Vec::new(),
            search_paths: Vec::new(),
        },
        &Options::default(),
        &Cancellation::default(),
    )
    .map_err(|e| ngspice_rs::primitives::SpiceError::circuit(e.to_string()))?;
    let dir = tempfile::tempdir().unwrap();
    for artifact in bundle.artifacts() {
        fs::write(dir.path().join(&artifact.filename), &artifact.bytes).unwrap();
    }
    let mut netlist = Parser::new().parse_file(dir.path().join(&bundle.primary.filename))?;
    PreparedInitialConditions::attach(
        &mut netlist,
        dir.path(),
        bundle.initial_conditions.unwrap(),
    )?;
    Ok(netlist)
}

fn run(path: &Path) -> SpiceResult<(AnalysisKind, Plot)> {
    let netlist = prepared(path)?;
    let config = RunConfig::from_netlist(&netlist)?;
    let mut circuit = config.circuit(&netlist)?;
    let request = config.plan_for(&netlist.analyses[0])?;
    Ok((
        request.kind(),
        request.run(&mut circuit, &config.context())?,
    ))
}

fn at(plot: &Plot, name: &str, time: f64) -> f64 {
    let times = plot.column("time").unwrap();
    let column = plot
        .column(name)
        .unwrap_or_else(|| panic!("missing vector {name}"));
    let upper = times
        .partition_point(|value| value.re < time)
        .min(times.len() - 1);
    let lower = upper.saturating_sub(1);
    if upper == lower || times[upper].re == time {
        return column[upper].re;
    }
    let fraction = (time - times[lower].re) / (times[upper].re - times[lower].re);
    fraction.mul_add(column[upper].re - column[lower].re, column[lower].re)
}

#[test]
fn source_prepared_passive_reference_matrix() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/passive-ic");
    let mut paths = fs::read_dir(&root)
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .filter(|path| path.extension().is_some_and(|ext| ext == "net"))
        .collect::<Vec<_>>();
    paths.sort();
    let mut errors = Vec::new();
    let mut cases = 0;
    let mut samples = 0;
    let mut max_error = 0_f64;
    for path in paths {
        let name = path.file_stem().unwrap().to_str().unwrap();
        let csv = [
            path.with_extension("csv"),
            path.with_extension("expected.csv"),
        ]
        .into_iter()
        .find(|path| path.exists());
        let result = run(&path);
        let Some(csv) = csv else {
            assert!(matches!(
                name,
                "icres-zero" | "parallel-hard" | "parallel-hard-equal"
            ));
            if result.is_ok() {
                errors.push(format!("{name}: invalid constraint unexpectedly solved"));
            }
            continue;
        };
        let (kind, plot) = match result {
            Ok(result) => result,
            Err(error) => {
                errors.push(format!("{name}: {error}"));
                continue;
            }
        };
        cases += 1;
        let table = fs::read_to_string(csv).unwrap();
        let mut lines = table.lines();
        let headers = lines
            .next()
            .unwrap()
            .split(',')
            .skip(1)
            .map(str::to_ascii_lowercase)
            .collect::<Vec<_>>();
        for (index, row) in lines.enumerate() {
            let values = row
                .split(',')
                .map(|text| text.parse::<f64>().unwrap())
                .collect::<Vec<_>>();
            for (name_column, expected) in headers.iter().zip(&values[1..]) {
                let actual = if kind == AnalysisKind::Transient {
                    at(&plot, name_column, values[0])
                } else {
                    let v = plot.value(name_column, index).unwrap();
                    if kind == AnalysisKind::Ac {
                        v.re.hypot(v.im)
                    } else {
                        v.re
                    }
                };
                let error = (actual - expected).abs();
                samples += 1;
                max_error = max_error.max(error);
                if error > 3e-6 {
                    errors.push(format!(
                        "{name}: {name_column} axis={} {actual:e} vs {expected:e} error={error:e}",
                        values[0]
                    ));
                }
            }
        }
    }
    eprintln!("reference: {cases} cases, {samples} values, max error {max_error:e} V");
    assert!(
        errors.is_empty(),
        "{} failures:\n{}",
        errors.len(),
        errors
            .iter()
            .take(35)
            .cloned()
            .collect::<Vec<_>>()
            .join("\n")
    );
    assert_eq!(cases, 35);
    assert_eq!(samples, 965);
}
