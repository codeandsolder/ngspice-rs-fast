//! Independently measured source-prepared passive initialization witnesses.
//! Electrical fixtures and CSV values are pinned to the producer revision
//! recorded in fixtures/passive-ic/README.md. No circuit is hand rewritten.
use std::{fs, path::Path};

use ngspice_rs::{
    analysis::{AnalysisInput, Plot, RunConfig},
    netlist::{Parser, ast::Netlist, prepared_initialization::PreparedInitialConditions},
    primitives::{AnalysisKind, SpiceResult},
};
use ngspice_rs_preprocess::{
    Cancellation, InputFormat, Options, PreparedBundle, Request, SourceInput,
    analysis_plan::{AnalysisKind as SourceAnalysis, Directive},
    prepare,
};

fn prepared(path: &Path) -> SpiceResult<(Netlist, Vec<AnalysisInput>)> {
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
    let plan = bundle.analysis_plan.as_ref().unwrap();
    let requests = plan
        .native_requests()
        .map(|entry| {
            let Directive::Analysis {
                analysis,
                arguments,
                uic,
            } = &entry.directive
            else {
                return Err(ngspice_rs::primitives::SpiceError::circuit(
                    "expected typed analysis request",
                ));
            };
            let kind = match analysis {
                SourceAnalysis::OperatingPoint => AnalysisKind::OperatingPoint,
                SourceAnalysis::DcSweep => AnalysisKind::DcSweep,
                SourceAnalysis::Ac => AnalysisKind::Ac,
                SourceAnalysis::Transient => AnalysisKind::Transient,
                SourceAnalysis::Noise => AnalysisKind::Noise,
                SourceAnalysis::TransferFunction => AnalysisKind::TransferFunction,
                SourceAnalysis::Pop | SourceAnalysis::Other => {
                    return Err(ngspice_rs::primitives::SpiceError::circuit(
                        "unsupported reference analysis",
                    ));
                }
            };
            let mut input = AnalysisInput::with_arguments(kind, arguments.clone());
            input.uic = *uic;
            Ok(input)
        })
        .collect::<SpiceResult<Vec<_>>>()?;
    let netlist = materialize(&bundle, &requests)?;
    Ok((netlist, requests))
}

fn materialize(bundle: &PreparedBundle, requests: &[AnalysisInput]) -> SpiceResult<Netlist> {
    let dir = tempfile::tempdir().unwrap();
    for artifact in bundle.artifacts() {
        fs::write(dir.path().join(&artifact.filename), &artifact.bytes).unwrap();
    }
    let mut netlist = Parser::new().parse_file(dir.path().join(&bundle.primary.filename))?;
    PreparedInitialConditions::attach_for_analyses(
        &mut netlist,
        dir.path(),
        bundle.initial_conditions.clone().unwrap(),
        requests.iter().map(|input| (input.kind, input.uic)),
    )?;
    Ok(netlist)
}

fn run(path: &Path) -> SpiceResult<(AnalysisKind, Plot)> {
    let (netlist, mut requests) = prepared(path)?;
    let config = RunConfig::from_netlist(&netlist)?;
    let mut circuit = config.circuit(&netlist)?;
    assert_eq!(requests.len(), 1);
    let input = requests.remove(0);
    // The C-native linear grid intentionally collapses two points into one.
    // Replay the two source-requested coordinates using native singleton
    // plans, without changing the electrical deck or its enclosing AC bias.
    if input.kind == AnalysisKind::Ac
        && input.arguments.len() == 4
        && input.arguments[0].eq_ignore_ascii_case("lin")
        && input.arguments[1] == "2"
    {
        let mut plots = input.arguments[2..].iter().map(|frequency| {
            let singleton = AnalysisInput::with_arguments(
                AnalysisKind::Ac,
                ["lin", "1", frequency.as_str(), frequency.as_str()],
            );
            config.plan(singleton)?.run(&mut circuit, &config.context())
        });
        let mut plot = plots.next().unwrap()?;
        let last = plots.next().unwrap()?;
        assert_eq!(last.point_count(), 1);
        plot.push_point(last.row(0).unwrap())?;
        return Ok((AnalysisKind::Ac, plot));
    }
    let request = config.plan(input)?;
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
            match result {
                Ok(_) => errors.push(format!("{name}: invalid constraint unexpectedly solved")),
                Err(error) => {
                    let message = error.to_string();
                    let expected = if name == "icres-zero" {
                        "icres"
                    } else {
                        "singular"
                    };
                    if !message.to_ascii_lowercase().contains(expected) {
                        errors.push(format!("{name}: wrong rejection: {message}"));
                    }
                }
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
            if kind == AnalysisKind::Ac {
                let axis = plot.value("frequency", index).map(|v| v.re);
                if axis != Some(values[0]) {
                    errors.push(format!(
                        "{name}: wrong AC coordinate {axis:?}, expected {}",
                        values[0]
                    ));
                }
            }
            for (name_column, expected) in headers.iter().zip(&values[1..]) {
                let actual = if kind == AnalysisKind::Transient {
                    at(&plot, name_column, values[0])
                } else {
                    let Some(v) = plot.value(name_column, index) else {
                        errors.push(format!("{name}: missing {name_column} at index {index}; {} points, columns {:?}", plot.point_count(), plot.variables()));
                        continue;
                    };
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

// These are the original producer #17 independent reactive-state witnesses.
// The solver consumes the emitted electrical artifacts and typed storage IC;
// no conversion from physical voltage/current is repeated here.
fn simplis_uic(source: &str, args: [&str; 4]) -> Plot {
    let dir = tempfile::tempdir().unwrap();
    let input = dir.path().join("input.net");
    fs::write(&input, source).unwrap();
    let bundle = prepare(
        &Request {
            format: InputFormat::SimplisDeck,
            input: SourceInput::File(input),
            libraries: Vec::new(),
            search_paths: Vec::new(),
        },
        &Options::default(),
        &Cancellation::default(),
    )
    .unwrap();
    assert_eq!(bundle.initial_conditions.as_ref().unwrap().devices.len(), 4);
    let mut request = AnalysisInput::with_arguments(AnalysisKind::Transient, args);
    request.uic = true;
    let netlist = materialize(&bundle, std::slice::from_ref(&request)).unwrap();
    let config = RunConfig::from_netlist(&netlist).unwrap();
    let mut circuit = config.circuit(&netlist).unwrap();
    config
        .plan(request)
        .unwrap()
        .run(&mut circuit, &config.context())
        .unwrap()
}

fn physical_charge(voltage: f64) -> f64 {
    if voltage < 0. {
        2.0_f64.mul_add(voltage, 4.)
    } else {
        4.0_f64.mul_add(voltage, 4.)
    }
}

#[test]
fn prepared_nonlinear_floating_capacitors_conserve_charge() {
    for (unit, current) in [("n", "1m"), ("p", "1u")] {
        let source = format!(
            r"* storage witness
Vbase n 0 0.75
I1 n a {current}
I2 n b {current}
I3 n c {current}
I4 n d {current}
X1 a n PARENT
X2 b n PARENT
X3 c n PARENT
X4 d n PARENT
.INIT V(X1.XC.!C1)=-2 V(X2.XC.!C1)=1
+ V(X3.XC.!C1)=3
.subckt PARENT p n
XC p n CELL
.INIT V(XC.!C1)=0
.subckt CELL p n
!C1 p n MC IC=1.5
.MODEL MC PWLC NSEG=2 X0=-1 Y0=2{unit} X1=0 Y1=4{unit} X2=2 Y2=12{unit}
.INIT V(!C1)=-0.5
.ends CELL
.ends PARENT
.end
"
        );
        let plot = simplis_uic(&source, ["10n", "10u", "0", "5n"]);
        let times = plot.column("time").unwrap();
        let reference = plot.column("v(n)").unwrap();
        let mut max_error = 0_f64;
        for (name, start) in [("v(a)", -2.), ("v(b)", 1.), ("v(c)", 3.), ("v(d)", 0.)] {
            let values = plot.column(name).unwrap();
            for point in 0..plot.point_count() {
                let voltage = values[point].re - reference[point].re;
                let expected = physical_charge(start) + times[point].re * 1e6;
                let error = (physical_charge(voltage) - expected).abs();
                max_error = max_error.max(error);
            }
        }
        eprintln!("{unit}: max normalized charge error {max_error:e}");
        assert!(max_error < 1e-7);
    }
}

#[test]
fn prepared_nonlinear_inductors_conserve_signed_flux_and_work() {
    let plot = simplis_uic(
        r"* signed flux witness
V1 a 0 1
V2 b 0 -1
!L1 a 0 ML IC=0
!L2 b 0 ML IC=0
.MODEL ML PWLL NSEG=2 X0=-2 Y0=-1m X1=0 Y1=3m X2=2 Y2=11m
.INIT I(!L1)=-1 I(!L2)=3
H1 ha 0 !L1 1
H2 hb 0 !L2 1
Rha ha 0 1k
Rhb hb 0 1k
Icap 0 c 1m
C1 c 0 1u IC=-5
.INIT V(C1)=2
Vlin e 0 2
L1 e 0 2m IC=9
.INIT I(L1)=-0.5
.end
",
        ["1u", "4m", "0", "0.5u"],
    );
    let flux = |current: f64| {
        if current < 0. {
            2e-3_f64.mul_add(current, 3e-3)
        } else {
            4e-3_f64.mul_add(current, 3e-3)
        }
    };
    let energy = |current: f64| {
        if current < 0. {
            1e-3 * current * current
        } else {
            2e-3 * current * current
        }
    };
    let (mut previous_time, mut previous_a, mut previous_b) = (0., -1., 3.);
    let (mut work_a, mut work_b) = (0., 0.);
    let times = plot.column("time").unwrap();
    let current_a = plot.column("i(v1)").unwrap();
    let current_b = plot.column("i(v2)").unwrap();
    let current_linear = plot.column("i(vlin)").unwrap();
    let observed_a = plot.column("v(ha)").unwrap();
    let observed_b = plot.column("v(hb)").unwrap();
    let capacitor = plot.column("v(c)").unwrap();
    for point in 0..plot.point_count() {
        let time = times[point].re;
        let a = -current_a[point].re;
        let b = -current_b[point].re;
        assert!(
            (flux(a) - flux(-1.) - time).abs() < 1e-10,
            "forward flux t={time} i={a}"
        );
        assert!(
            (flux(b) - flux(3.) + time).abs() < 1e-10,
            "reverse flux t={time} i={b}"
        );
        assert!(
            (observed_a[point].re - a).abs() < 1e-10 && (observed_b[point].re - b).abs() < 1e-10
        );
        let step = time - previous_time;
        work_a = (a + previous_a).mul_add(0.5 * step, work_a);
        work_b = (b + previous_b).mul_add(-0.5 * step, work_b);
        assert!((energy(a) - energy(-1.) - work_a).abs() < 1e-9);
        assert!((energy(b) - energy(3.) - work_b).abs() < 1e-9);
        assert!((capacitor[point].re - 1000_f64.mul_add(time, 2.)).abs() < 1e-7);
        assert!((-current_linear[point].re - 1000_f64.mul_add(time, -0.5)).abs() < 1e-7);
        (previous_time, previous_a, previous_b) = (time, a, b);
    }
}
