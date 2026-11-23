//! Explicit Fourier settings on analytic traces and C's exact saved sample grid.
use ngspice_conformance::ngspice::Scratch;
use ngspice_conformance::process::{ProcessLimits, run as run_process};
use ngspice_rs::analysis::fourier::FourierSettings;
use ngspice_rs::analysis::{Plot, PlotFlags, Variable, fourier};
use ngspice_rs::netlist::Parser;
use ngspice_rs::netlist::source::parse_deck_text;
use ngspice_rs::primitives::{AnalysisKind, Complex};
use std::fs;
use std::path::Path;
use std::process::Command;

fn cards() -> Vec<ngspice_rs::netlist::ast::FourierCard> {
    Parser::new()
        .parse_deck_with_output(&parse_deck_text(
            Path::new("four.cir"),
            "Four\n.four 1k v(out)\n.end\n",
        ))
        .unwrap()
        .fourier
}
#[test]
fn configured_fourier_recovers_multiple_periods_and_polynomial_interpolation() {
    let mut plot = Plot::builder("tran1", "Transient Analysis", PlotFlags::Real);
    plot.push_variable(Variable::new("time", "time"));
    plot.push_variable(Variable::new("v(out)", "voltage"));
    let mut plot = plot.build();
    for i in 0..=4000 {
        let t = i as f64 * 1e-6;
        let value = 0.5
            + (std::f64::consts::TAU * 1000. * t).sin()
            + 0.25 * (std::f64::consts::TAU * 2000. * t).sin();
        plot.push_point(&[Complex::real(t), Complex::real(value)])
            .unwrap();
    }
    for degree in [1, 2, 3, 5] {
        let settings = FourierSettings {
            nfreqs: 5,
            nperiods: 2,
            polydegree: degree,
            gridsize: 256,
        };
        let results =
            fourier::resolve_with_settings(&plot, AnalysisKind::Transient, &cards(), settings)
                .unwrap();
        let result = &results[0];
        assert_eq!(result.harmonics.len(), 4);
        assert_eq!(
            result
                .harmonics
                .iter()
                .map(|harmonic| harmonic.order)
                .collect::<Vec<_>>(),
            [1, 2, 3, 4]
        );
        assert_eq!(result.divisions, 512);
        assert!((result.dc - 0.5).abs() < 1e-7);
        assert!((result.harmonics[0].amplitude - 1.).abs() < 4e-6);
        assert!((result.thd - 0.25).abs() < 4e-6);
    }
    let invalid = FourierSettings {
        gridsize: 100000,
        nperiods: 100,
        ..Default::default()
    };
    assert!(
        fourier::resolve_with_settings(&plot, AnalysisKind::Transient, &cards(), invalid).is_err()
    );
}

/// Bound a Fourier coefficient's error against the exact sine on the
/// configured half-open grid. The interpolation remainder is bounded by
/// the (degree+1)th derivative times the product of local distances divided
/// by factorial; the L1 weight norm controls amplification of sample and
/// arithmetic roundoff. This is independent of C's ill-conditioned fit.
fn sine_coefficient_error_bound(plot: &Plot, settings: FourierSettings) -> f64 {
    let times: Vec<_> = plot.column("time").unwrap().iter().map(|v| v.re).collect();
    let values: Vec<_> = plot
        .column("v(out)")
        .unwrap()
        .iter()
        .map(|v| v.re)
        .collect();
    let omega = std::f64::consts::TAU * 1000.;
    let sample_error = times.iter().zip(&values)
        .map(|(&t, &v)| (v - (0.5 + (omega * t).sin())).abs())
        .fold(0., f64::max)
        // Account for the rounded phase used to evaluate the sample oracle.
        + 4. * f64::EPSILON * (omega * times.last().unwrap()).abs().max(1.);
    let order = settings.polydegree as usize + 1;
    let factorial = (1..=settings.polydegree + 1)
        .map(f64::from)
        .product::<f64>();
    let count = settings.gridsize * settings.nperiods;
    let to = *times.last().unwrap();
    let from = to - f64::from(settings.nperiods) / 1000.;
    let step = (to - from) / f64::from(count);
    let mut max_error = sample_error;
    for i in 0..count {
        let offset = f64::from(i) * step;
        let x = from + offset;
        let right = times.partition_point(|&t| t < x);
        if times[right] == x {
            continue;
        }
        let start = right
            .saturating_sub((order - 1).div_ceil(2))
            .min(times.len() - order);
        let nodes = &times[start..start + order];
        let truncation = nodes
            .iter()
            .map(|&t| omega * (x - t).abs())
            .product::<f64>()
            / factorial;
        let norm = nodes
            .iter()
            .enumerate()
            .map(|(j, &node)| {
                nodes
                    .iter()
                    .enumerate()
                    .filter(|&(k, _)| j != k)
                    .map(|(_, &other)| ((x - other) / (node - other)).abs())
                    .product::<f64>()
            })
            .sum::<f64>();
        let roundoff = 8. * f64::from(settings.polydegree + 1) * f64::EPSILON * 1.5;
        max_error = max_error.max(truncation + norm * (sample_error + roundoff));
    }
    // A harmonic is twice the normalized complex projection; account for
    // the small reduced-phase and compensated sum error as well.
    2. * max_error + 32. * f64::EPSILON
}

#[test]
#[ignore = "requires NGSPICE_BIN; compare all settings on the same physical samples"]
fn explicit_fourier_settings_match_c_or_analytic_sine_on_c_samples() {
    use ngspice_rs::analysis::RawFile;
    let binary = std::env::var("NGSPICE_BIN").expect("NGSPICE_BIN");
    assert!(Path::new(&binary).is_absolute());
    let scratch = Scratch::new("m9-fourier").unwrap();
    let dir = scratch.path();
    let raw = dir.join("samples.raw");
    let deck = dir.join("deck.cir");
    fs::write(
        &deck,
        "Four samples\nV1 out 0 sin(.5 1 1k)\nR1 out 0 1k\n.tran 1u 4m\n.end\n",
    )
    .unwrap();
    let run = run_process(
        &binary,
        dir,
        &[
            std::ffi::OsStr::new("-b"),
            std::ffi::OsStr::new("-r"),
            raw.as_os_str(),
            deck.as_os_str(),
        ],
        ProcessLimits::ORACLE,
    )
    .unwrap();
    assert!(
        run.status.success(),
        "{}",
        String::from_utf8_lossy(&run.stderr)
    );
    let plot = &RawFile::load(&raw).unwrap().plots[0].plot;
    for (index, settings) in [
        FourierSettings::default(),
        FourierSettings {
            nfreqs: 5,
            ..Default::default()
        },
        FourierSettings {
            nperiods: 2,
            ..Default::default()
        },
        FourierSettings {
            polydegree: 3,
            ..Default::default()
        },
        FourierSettings {
            polydegree: 2,
            ..Default::default()
        },
        FourierSettings {
            polydegree: 5,
            ..Default::default()
        },
        FourierSettings {
            polydegree: 7,
            ..Default::default()
        },
        FourierSettings {
            polydegree: 0,
            ..Default::default()
        },
        FourierSettings {
            gridsize: 512,
            ..Default::default()
        },
    ]
    .into_iter()
    .enumerate()
    {
        let control = dir.join(format!("read{index}.cir"));
        fs::write(&control,format!("Four compare\n.control\nload {}\nset nfreqs={} nperiods={} polydegree={} fourgridsize={}\nfourier 1k v(out)\nquit\n.endc\n.end\n",raw.display(),settings.nfreqs,settings.nperiods,settings.polydegree,settings.gridsize)).unwrap();
        let run = run_process(
            &binary,
            dir,
            &[std::ffi::OsStr::new("-b"), control.as_os_str()],
            ProcessLimits::ORACLE,
        )
        .unwrap();
        let text = String::from_utf8_lossy(&run.stdout);
        assert!(run.status.success(), "{text}");
        let ours =
            fourier::resolve_with_settings(plot, AnalysisKind::Transient, &cards(), settings)
                .unwrap();
        let rows: Vec<_> = text
            .lines()
            .filter_map(|line| {
                let mut fields = line.split_whitespace();
                let order = fields.next()?.parse::<usize>().ok()?;
                let _frequency = fields.next()?.parse::<f64>().ok()?;
                let magnitude = fields.next()?.parse::<f64>().ok()?;
                Some((order, magnitude))
            })
            .collect();
        assert_eq!(rows.len(), settings.nfreqs as usize, "{text}");
        // The pinned C high-degree absolute-power fit introduces spurious
        // harmonics (degree 5: up to 1.54e-6; degree 7: up to 2.16e-5).
        // Check those fits against the physical sine, not the inaccurate C
        // arithmetic. Every other setting retains its original C tolerance.
        let analytic_bound =
            (settings.polydegree >= 5).then(|| sine_coefficient_error_bound(plot, settings));
        for (order, magnitude) in rows {
            let value = if order == 0 {
                ours[0].dc
            } else {
                ours[0].harmonics[order - 1].amplitude
            };
            if let Some(bound) = analytic_bound {
                let expected = match order {
                    0 => 0.5,
                    1 => 1.,
                    _ => 0.,
                };
                assert!(
                    (value - expected).abs() <= bound,
                    "settings {settings:?}, row {order}: Rust {value}, analytic {expected}, bound {bound}, C {magnitude}"
                );
            } else {
                assert!(
                    (value - magnitude).abs() <= 5e-6 * magnitude.abs() + 1e-8,
                    "settings {settings:?}, row {order}: Rust {value}, C {magnitude}"
                );
            }
        }
    }
}

#[test]
fn fundamental_expressions_are_resolved_before_simulation() {
    use ngspice_rs::netlist::eval::ParamScope;
    let parsed = Parser::new().parse_deck_with_output(&parse_deck_text(
        Path::new("expr.cir"),
        "Expressions\n.param base=500\n.func twice(x) {2*x}\n.four {twice(base)} v(out)\n.four 'base*2' v(out)\n.end\n",
    )).unwrap();
    let written = ngspice_rs::netlist::write_netlist(&parsed.netlist).unwrap();
    let again = Parser::new()
        .parse_deck_with_output(&parse_deck_text(Path::new("again.cir"), &written))
        .unwrap();
    assert!(ngspice_rs::netlist::semantic_eq(
        &parsed.netlist,
        &again.netlist
    ));
    let scope = ParamScope::for_netlist(&parsed.netlist).unwrap();
    let resolved = fourier::resolve_cards(&parsed.fourier, &scope).unwrap();
    assert_eq!(resolved.len(), 2);
    for card in resolved {
        assert_eq!(card.fundamental, 1000.);
        assert!(card.fundamental_expression.is_none());
    }
    for expression in ["missing", "0", "-base", "1/0"] {
        let parsed = Parser::new()
            .parse_deck_with_output(&parse_deck_text(
                Path::new("expr.cir"),
                &format!("Expressions\n.four {{{expression}}} v(out)\n.end\n"),
            ))
            .unwrap();
        assert!(fourier::resolve_cards(&parsed.fourier, &scope).is_err());
    }
}

#[test]
fn cli_evaluates_fourier_parameters_and_preserves_output_on_failure() {
    let scratch = Scratch::new("m9-fourier").unwrap();
    let dir = scratch.path();
    let deck = dir.join("deck.cir");
    let raw = dir.join("out.raw");
    fs::write(&deck, "Expressions\n.param base=500\nV1 out 0 sin(0 1 1k)\nR1 out 0 1k\n.tran 1u 4m\n.four {2*base} v(out)\n.end\n").unwrap();
    let report = ngspice_rs::cli::simulate::run(&deck, &raw, true).unwrap();
    assert_eq!(
        report.plots.last().unwrap().fourier_results[0].fundamental,
        1000.
    );
    let original = fs::read(&raw).unwrap();
    fs::write(
        &deck,
        "Expressions\nV1 out 0 1\nR1 out 0 1k\n.tran 1u 4m\n.four {missing} v(out)\n.end\n",
    )
    .unwrap();
    assert!(ngspice_rs::cli::simulate::run(&deck, &raw, true).is_err());
    assert_eq!(fs::read(&raw).unwrap(), original);
}

#[test]
fn control_fourier_command_runs_through_the_process_interface() {
    let scratch = Scratch::new("m9-fourier").unwrap();
    let dir = scratch.path();
    let deck = dir.join("deck.cir");
    let raw = dir.join("out.raw");
    let text = "Control Fourier\nV1 out 0 sin(.5 1 1k)\nR1 out 0 1k\n.tran 1u 4m\n.control\nset nfreqs=5 nperiods=2 polydegree=3 fourgridsize=256\nrun\nfourier 1k v(out)\nquit\n.endc\n.end\n";
    fs::write(&deck, text).unwrap();
    let run = Command::new(
        std::env::var_os("CARGO_BIN_EXE_spice-rs").expect("Cargo supplies the test executable"),
    )
    .arg("simulate")
    .arg("--output")
    .arg(&raw)
    .arg(&deck)
    .output()
    .unwrap();
    assert!(
        run.status.success(),
        "{}",
        String::from_utf8_lossy(&run.stderr)
    );
    assert!(String::from_utf8_lossy(&run.stdout).contains("Fourier analysis for v(out)"));
    let report = ngspice_rs::cli::simulate::run(&deck, &raw, true).unwrap();
    let result = &report.plots[0].fourier_results[0];
    assert_eq!(result.harmonics.len(), 4);
    assert_eq!(result.divisions, 512);
    let parser = Parser::new();
    let parsed = parser
        .parse_deck_with_output(&parse_deck_text(Path::new("a.cir"), text))
        .unwrap();
    let written = ngspice_rs::netlist::write_netlist(&parsed.netlist).unwrap();
    let again = parser
        .parse_deck_with_output(&parse_deck_text(Path::new("b.cir"), &written))
        .unwrap();
    assert!(ngspice_rs::netlist::semantic_eq(
        &parsed.netlist,
        &again.netlist
    ));
    for commands in [
        "fourier 1k v(out)",
        "run\nrun",
        "quit",
        "run\n.endc\n.control\nrun",
        "run\nset nfreqs=5",
        "run\nquit\nfourier 1k v(out)",
    ] {
        let invalid = format!(
            "Bad\nV1 out 0 1\nR1 out 0 1k\n.tran 1u 4m\n.control\n{commands}\n.endc\n.end\n"
        );
        fs::write(&deck, invalid).unwrap();
        fs::write(&raw, b"previous").unwrap();
        assert!(ngspice_rs::cli::simulate::run(&deck, &raw, true).is_err());
        assert_eq!(fs::read(&raw).unwrap(), b"previous");
    }
}
