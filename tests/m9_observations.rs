//! Device observations at solver points, checked by C goldens and analytic KCL.
use ngspice_conformance::ngspice::Scratch;
use ngspice_rs::analysis::{Plot, RawFile};
use ngspice_rs::cli::simulate;
use ngspice_rs::primitives::Complex;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
fn root() -> PathBuf {
    ngspice_conformance::fixtures::workspace_root()
        .unwrap()
        .join("conformance/observations")
}
fn column(plot: &ngspice_rs::analysis::Plot, name: &str) -> Vec<Complex> {
    let i = plot
        .variable_index(name)
        .unwrap_or_else(|| panic!("missing {name}: {:?}", plot.variables()));
    plot.rows().map(|p| p[i]).collect()
}
fn compare(ours: &ngspice_rs::analysis::Plot, c: &ngspice_rs::analysis::Plot) {
    assert_eq!(ours.point_count(), c.point_count());
    for v in ours.variables() {
        let name = v.name.as_str();
        let c_name = if name.starts_with("i(") {
            format!("{}#branch", &name[2..name.len() - 1])
        } else {
            name.to_owned()
        };
        let other = c
            .variable_index(&c_name)
            .or_else(|| c.variable_index(name))
            .or_else(|| c.variable_index(&format!("i({name})")))
            .or_else(|| c.variable_index(&format!("v({name})")))
            .unwrap_or_else(|| panic!("missing {name}: {:?}", c.variables()));
        let i = ours.variable_index(&v.name).unwrap();
        for (a, b) in ours.rows().zip(c.rows()) {
            assert!(
                (a[i].re - b[other].re).abs() <= 1e-7 * b[other].re.abs() + 2e-11,
                "{}: {:?} {:?}",
                v.name,
                a[i],
                b[other]
            );
        }
    }
}
#[test]
fn saved_device_currents_and_parameters_match_committed_c_golden() {
    let scratch = Scratch::new("m9-observe-golden").unwrap();
    let dir = scratch.path();
    let raw = dir.join("out.raw");
    simulate::run(&root().join("m9_device_dc.cir"), &raw, true).unwrap();
    let ours = RawFile::load(&raw).unwrap();
    let c = RawFile::load(root().join("m9_device_dc.raw")).unwrap();
    compare(&ours.plots[0].plot, &c.plots[0].plot);
}
#[test]
fn companion_currents_use_integrated_charge_and_satisfy_kcl() {
    let scratch = Scratch::new("m9-observe-tran").unwrap();
    let dir = scratch.path();
    let deck = dir.join("deck.cir");
    let raw = dir.join("out.raw");
    fs::write(&deck,"RC observations\nV1 in 0 pulse(0 1 1u .1u .1u 1m 2m)\nR1 in out 1k\nC1 out 0 1u\n.tran 1u 20u\n.save v(in) v(out) i(r1) @c1[i] @r1[p] @c1[capacitance]\n.measure tran imax max @c1[i]\n.end\n").unwrap();
    let report = simulate::run(&deck, &raw, true).unwrap();
    assert!(report.plots[0].measurements[0].value > 0.0009);
    let file = RawFile::load(&raw).unwrap();
    let p = &file.plots[0].plot;
    let resistor = column(p, "i(r1)");
    let capacitor = column(p, "@c1[i]");
    let input = column(p, "v(in)");
    let output = column(p, "v(out)");
    let power = column(p, "@r1[p]");
    for k in 0..p.point_count() {
        assert!((resistor[k].re - capacitor[k].re).abs() < 2e-12);
        let drop = input[k].re - output[k].re;
        assert!((resistor[k].re - drop / 1000.).abs() < 1e-14);
        assert!((power[k].re - drop * resistor[k].re).abs() < 1e-14);
    }
}
#[test]
fn invalid_observations_fail_without_publishing() {
    let scratch = Scratch::new("m9-observe-atomic").unwrap();
    let dir = scratch.path();
    let deck = dir.join("deck.cir");
    let raw = dir.join("out.raw");
    for request in ["@r1[unknown]", "i(missing)", "@missing[m]"] {
        fs::write(&raw, b"previous").unwrap();
        fs::write(
            &deck,
            format!("Invalid\nV1 out 0 1\nR1 out 0 1k\n.op\n.save {request}\n.end\n"),
        )
        .unwrap();
        assert!(simulate::run(&deck, &raw, true).is_err());
        assert_eq!(fs::read(&raw).unwrap(), b"previous");
    }
}
// The capture oracle and a general live-C oracle serve different purposes:
// a platform's native C build can be valid for parity without reproducing
// another platform's binary64 rounding.
fn run_c_oracle(variable: &str, dir: &Path) -> RawFile {
    let binary = PathBuf::from(
        std::env::var_os(variable).unwrap_or_else(|| panic!("set {variable} to an absolute path")),
    );
    assert!(binary.is_absolute(), "{variable} must be absolute");
    let raw = dir.join("c.raw");
    let run = ngspice_conformance::process::run_with_environment(
        binary,
        dir,
        &[
            std::ffi::OsStr::new("-n"),
            std::ffi::OsStr::new("-b"),
            std::ffi::OsStr::new("-r"),
            raw.as_os_str(),
            root().join("m9_device_dc.cir").as_os_str(),
        ],
        ngspice_conformance::process::ProcessLimits::ORACLE,
        &[("LC_ALL", Some("C")), ("SPICE_ASCIIRAWFILE", None)],
    )
    .unwrap();
    assert!(
        run.status.success(),
        "{variable}: {}",
        String::from_utf8_lossy(&run.stderr)
    );
    RawFile::load(&raw).unwrap()
}

fn assert_plot_bits(actual: &Plot, expected: &Plot) {
    assert_eq!(actual.name, expected.name);
    assert_eq!(actual.plotname, expected.plotname);
    assert_eq!(actual.flags(), expected.flags());
    assert_eq!(actual.variables(), expected.variables());
    assert_eq!(actual.point_count(), expected.point_count());
    for (row, (actual, expected)) in actual.rows().zip(expected.rows()).enumerate() {
        assert_eq!(actual.len(), expected.len());
        for (column, (a, b)) in actual.iter().zip(expected).enumerate() {
            assert_eq!(
                [a.re.to_bits(), a.im.to_bits()],
                [b.re.to_bits(), b.im.to_bits()],
                "point {row}, column {column}: {a:?} != {b:?}"
            );
        }
    }
}

#[test]
#[should_panic(expected = "point 2, column 2")]
fn capture_comparison_rejects_single_ulp_drift() {
    let saved = RawFile::load(root().join("m9_device_dc.raw")).unwrap();
    let expected = &saved.plots[0].plot;
    let mut actual = expected.clone();
    let value = &mut actual.point_mut(2).unwrap()[2].re;
    *value = f64::from_bits(value.to_bits() - 1);
    assert_plot_bits(&actual, expected);
}

#[test]
#[should_panic(expected = "point 0, column 0")]
fn capture_comparison_distinguishes_signed_zero() {
    let saved = RawFile::load(root().join("m9_device_dc.raw")).unwrap();
    let expected = &saved.plots[0].plot;
    let mut actual = expected.clone();
    assert_eq!(actual.row(0).unwrap()[0].re.to_bits(), 0.0_f64.to_bits());
    actual.point_mut(0).unwrap()[0].re = -0.0;
    assert_plot_bits(&actual, expected);
}

#[test]
#[ignore = "requires absolute NGSPICE_CAPTURE_BIN; see conformance/observations/README.md"]
fn device_golden_matches_capture_oracle_bit_exactly() {
    let scratch = Scratch::new("m9-observe-capture-oracle").unwrap();
    let dir = scratch.path();
    let got = run_c_oracle("NGSPICE_CAPTURE_BIN", dir);

    let saved = RawFile::load(root().join("m9_device_dc.raw")).unwrap();
    assert_eq!(got.plots.len(), saved.plots.len());
    for (actual, expected) in got.plots.iter().zip(&saved.plots) {
        assert_eq!(actual.title, expected.title);
        assert_plot_bits(&actual.plot, &expected.plot);
    }
}

#[test]
#[ignore = "requires absolute NGSPICE_BIN"]
fn device_observations_match_live_c() {
    let scratch = Scratch::new("m9-observe-live-parity").unwrap();
    let dir = scratch.path();
    let c = run_c_oracle("NGSPICE_BIN", dir);
    let raw = dir.join("rust.raw");
    simulate::run(&root().join("m9_device_dc.cir"), &raw, true).unwrap();
    let ours = RawFile::load(&raw).unwrap();

    assert_eq!(ours.plots.len(), c.plots.len());
    for (actual, expected) in ours.plots.iter().zip(&c.plots) {
        compare(&actual.plot, &expected.plot);
    }
}

#[test]
fn parameter_sweeps_observe_each_context_and_ac_rejects_current_asks() {
    let scratch = Scratch::new("m9-observe-contexts").unwrap();
    let dir = scratch.path();
    let deck = dir.join("deck.cir");
    let raw = dir.join("out.raw");
    fs::write(&deck, "Sweep asks\nV1 out 0 1 ac 1\nR1 out 0 1k\n.dc r1 1k 2k 1k\n.save @r1[resistance] i(r1)\n.end\n").unwrap();
    simulate::run(&deck, &raw, true).unwrap();
    let file = RawFile::load(&raw).unwrap();
    let plot = &file.plots[0].plot;
    assert_eq!(
        column(plot, "@r1[resistance]"),
        [Complex::real(1000.), Complex::real(2000.)]
    );
    assert_eq!(
        column(plot, "i(r1)"),
        [Complex::real(0.001), Complex::real(0.0005)]
    );
    fs::write(
        &deck,
        "AC asks\nV1 out 0 1 ac 1\nR1 out 0 1k\n.ac lin 2 1 2\n.save @r1[resistance]\n.end\n",
    )
    .unwrap();
    simulate::run(&deck, &raw, true).unwrap();
    let file = RawFile::load(&raw).unwrap();
    assert!(
        column(&file.plots[0].plot, "@r1[resistance]")
            .iter()
            .all(|v| *v == Complex::real(1000.))
    );
    let previous = fs::read(&raw).unwrap();
    fs::write(
        &deck,
        "AC current\nV1 out 0 1 ac 1\nR1 out 0 1k\n.ac lin 2 1 2\n.save @r1[i]\n.end\n",
    )
    .unwrap();
    let error = simulate::run(&deck, &raw, true).unwrap_err();
    assert!(error.is_not_yet_ported(), "{error}");
    assert_eq!(fs::read(&raw).unwrap(), previous);
}

#[test]
fn device_observations_run_through_cli() {
    let scratch = Scratch::new("m9-observe-process").unwrap();
    let dir = scratch.path();
    let raw = dir.join("out.raw");
    let run = Command::new(
        std::env::var_os("CARGO_BIN_EXE_spice-rs").expect("Cargo supplies the test executable"),
    )
    .arg("simulate")
    .arg(root().join("m9_device_dc.cir"))
    .arg("--output")
    .arg(&raw)
    .output()
    .unwrap();
    assert!(
        run.status.success(),
        "{}",
        String::from_utf8_lossy(&run.stderr)
    );
    let ours = RawFile::load(&raw).unwrap();
    let c = RawFile::load(root().join("m9_device_dc.raw")).unwrap();
    compare(&ours.plots[0].plot, &c.plots[0].plot);
}

#[test]
fn swept_sources_keep_terminal_sign_multiplier_and_bound_units() {
    let scratch = Scratch::new("m9-observe-source-sweep").unwrap();
    let dir = scratch.path();
    let deck = dir.join("deck.cir");
    let raw = dir.join("out.raw");
    fs::write(&deck, "Swept current\ni1 0 a dc 1m m=2\nr1 a 0 1k\n.dc i1 -1m 1m 1m\n.save v(a) @i1[dc] @i1[i] i(i1) @i1[p] @r1[resistance] @r1[p]\n.end\n").unwrap();
    simulate::run(&deck, &raw, true).unwrap();
    let file = RawFile::load(&raw).unwrap();
    let plot = &file.plots[0].plot;
    assert_eq!(plot.point_count(), 3);
    for (name, unit, expected) in [
        ("v(a)", "voltage", [-2., 0., 2.]),
        ("@i1[dc]", "parameter", [-0.001, 0., 0.001]),
        ("@i1[i]", "current", [-0.002, 0., 0.002]),
        ("i(i1)", "current", [-0.002, 0., 0.002]),
        ("@i1[p]", "power", [-0.004, 0., -0.004]),
        ("@r1[resistance]", "resistance", [1000.; 3]),
        ("@r1[p]", "power", [0.004, 0., 0.004]),
    ] {
        let index = plot.variable_index(name).unwrap();
        assert_eq!(plot.variables()[index].unit, unit, "{name}");
        for (row, value) in plot.rows().zip(expected) {
            assert!((row[index].re - value).abs() < 1e-12, "{name}: {row:?}");
        }
    }
}
