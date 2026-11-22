//! Device observations at solver points, checked by C goldens and analytic KCL.
use ngspice_rs::analysis::{Plot, RawFile};
use ngspice_rs::cli::simulate;
use ngspice_rs::primitives::Complex;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("conformance/observations")
}
fn scratch(tag: &str) -> PathBuf {
    let p = std::env::temp_dir().join(format!("m9-observe-{}-{tag}", std::process::id()));
    fs::create_dir_all(&p).unwrap();
    p
}
fn column(plot: &ngspice_rs::analysis::Plot, name: &str) -> Vec<Complex> {
    let i = plot
        .variable_index(name)
        .unwrap_or_else(|| panic!("missing {name}: {:?}", plot.variables));
    plot.points.iter().map(|p| p[i]).collect()
}
fn compare(ours: &ngspice_rs::analysis::Plot, c: &ngspice_rs::analysis::Plot) {
    assert_eq!(ours.points.len(), c.points.len());
    for v in &ours.variables {
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
            .unwrap_or_else(|| panic!("missing {name}: {:?}", c.variables));
        let i = ours.variable_index(&v.name).unwrap();
        for (a, b) in ours.points.iter().zip(&c.points) {
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
    let dir = scratch("golden");
    let raw = dir.join("out.raw");
    simulate::run(&root().join("m9_device_dc.cir"), &raw, true).unwrap();
    let ours = RawFile::load(&raw).unwrap();
    let c = RawFile::load(root().join("m9_device_dc.raw")).unwrap();
    compare(&ours.plots[0].plot, &c.plots[0].plot);
    fs::remove_dir_all(dir).unwrap();
}
#[test]
fn companion_currents_use_integrated_charge_and_satisfy_kcl() {
    let dir = scratch("tran");
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
    for k in 0..p.points.len() {
        assert!((resistor[k].re - capacitor[k].re).abs() < 2e-12);
        let drop = input[k].re - output[k].re;
        assert!((resistor[k].re - drop / 1000.).abs() < 1e-14);
        assert!((power[k].re - drop * resistor[k].re).abs() < 1e-14);
    }
    fs::remove_dir_all(dir).unwrap();
}
#[test]
fn invalid_observations_fail_without_publishing() {
    let dir = scratch("atomic");
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
    fs::remove_dir_all(dir).unwrap();
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
    let run = Command::new(binary)
        .current_dir(dir)
        .env("LC_ALL", "C")
        .env_remove("SPICE_ASCIIRAWFILE")
        .args(["-n", "-b", "-r"])
        .arg(&raw)
        .arg(root().join("m9_device_dc.cir"))
        .output()
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
    assert_eq!(actual.flags, expected.flags);
    assert_eq!(actual.variables, expected.variables);
    assert_eq!(actual.points.len(), expected.points.len());
    for (row, (actual, expected)) in actual.points.iter().zip(&expected.points).enumerate() {
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
    let value = &mut actual.points[2][2].re;
    *value = f64::from_bits(value.to_bits() - 1);
    assert_plot_bits(&actual, expected);
}

#[test]
#[should_panic(expected = "point 0, column 0")]
fn capture_comparison_distinguishes_signed_zero() {
    let saved = RawFile::load(root().join("m9_device_dc.raw")).unwrap();
    let expected = &saved.plots[0].plot;
    let mut actual = expected.clone();
    assert_eq!(actual.points[0][0].re.to_bits(), 0.0_f64.to_bits());
    actual.points[0][0].re = -0.0;
    assert_plot_bits(&actual, expected);
}

#[test]
#[ignore = "requires absolute NGSPICE_CAPTURE_BIN; see conformance/observations/README.md"]
fn device_golden_matches_capture_oracle_bit_exactly() {
    let dir = scratch("capture-oracle");
    let got = run_c_oracle("NGSPICE_CAPTURE_BIN", &dir);
    fs::remove_dir_all(dir).unwrap();
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
    let dir = scratch("live-parity");
    let c = run_c_oracle("NGSPICE_BIN", &dir);
    let raw = dir.join("rust.raw");
    simulate::run(&root().join("m9_device_dc.cir"), &raw, true).unwrap();
    let ours = RawFile::load(&raw).unwrap();
    fs::remove_dir_all(dir).unwrap();
    assert_eq!(ours.plots.len(), c.plots.len());
    for (actual, expected) in ours.plots.iter().zip(&c.plots) {
        compare(&actual.plot, &expected.plot);
    }
}

#[test]
fn parameter_sweeps_observe_each_context_and_ac_rejects_current_asks() {
    let dir = scratch("contexts");
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
    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn device_observations_run_through_cli() {
    let dir = scratch("process");
    let raw = dir.join("out.raw");
    let run = Command::new(env!("CARGO_BIN_EXE_spice-rs"))
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
    fs::remove_dir_all(dir).unwrap();
}
