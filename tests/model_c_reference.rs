//! Model selection and opt-in live C setup / selector-alias comparisons.
use ngspice_conformance::ngspice::{Ngspice, Scratch, capture};
use ngspice_rs::analysis::{Plot, RawFile, RunConfig};
use ngspice_rs::devices::{ModelContext, ModelResolver};
use ngspice_rs::netlist::Parser;
use ngspice_rs::netlist::source::parse_deck_text;
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

const PROBE: &str = include_str!("../conformance/parser/model_schemas.cir");

#[test]
#[ignore = "requires NGSPICE_BIN; validates model schemas/selectors against C setup"]
fn model_resolver_and_diode_defaults_match_c_setup() {
    let netlist = Parser::new()
        .parse_deck(&parse_deck_text(Path::new("model_schemas.cir"), PROBE))
        .unwrap();
    let resolver = ModelResolver::new(&netlist.models).unwrap();
    let context = ModelContext::default();
    let mut expected = BTreeMap::new();
    for instance in &netlist.devices {
        if instance.designator != 'd' {
            continue;
        }
        let resolved = resolver.resolve(instance).unwrap().unwrap();
        let model = resolved.diode_parameters(&context).unwrap();
        let device = resolved
            .diode_instance_parameters(instance, &context)
            .unwrap();
        let name = &resolved.card().name;
        expected.insert(format!("@{name}[is]"), model.saturation_current);
        expected.insert(format!("@{name}[n]"), model.emission_coefficient);
        expected.insert(format!("@{name}[rs]"), model.series_resistance);
        expected.insert(
            format!("@{name}[tnom]"),
            model.nominal_temperature_kelvin - 273.15,
        );
        expected.insert(
            format!("@{name}[level]"),
            f64::from(resolved.levels().applied.unwrap()),
        );
        expected.insert(
            format!("@{}[temp]", instance.name),
            device.temperature_kelvin - 273.15,
        );
        expected.insert(format!("@{}[area]", instance.name), device.area);
    }
    assert_eq!(expected["@duplicate[is]"], 4e-14);
    assert_eq!(expected["@explicit[level]"], 1.0);
    assert_eq!(expected["@dexplicit[temp]"], 40.0);
    assert_eq!(
        resolver
            .resolve(netlist.device("q1").unwrap())
            .unwrap()
            .unwrap()
            .levels()
            .selector,
        2
    );
    assert_eq!(
        resolver
            .resolve(netlist.device("m1").unwrap())
            .unwrap()
            .unwrap()
            .levels()
            .selector,
        1
    );
    // These C-only queries ensure repeated selector-only level values did not
    // switch to the unavailable VBIC/BSIM backend. They are not Rust schemas.
    expected.insert("@qmod[bf]".into(), 100.0);
    expected.insert("@mmod[kp]".into(), 1e-3);
    let queries = expected.keys().cloned().collect::<Vec<_>>().join(" ");
    let text = PROBE.replace(
        ".end\n",
        &format!(".control\nset numdgt=17\nop\nprint {queries}\nquit\n.endc\n.end\n"),
    );
    struct Scratch(PathBuf);
    impl Drop for Scratch {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }
    let nonce = scratch_nonce();
    let scratch = Scratch(
        std::env::temp_dir().join(format!("spice-model-schema-{}-{nonce}", std::process::id())),
    );
    fs::create_dir_all(&scratch.0).unwrap();
    fs::write(scratch.0.join("probe.cir"), text).unwrap();
    let binary = PathBuf::from(std::env::var_os("NGSPICE_BIN").expect("set NGSPICE_BIN"));
    assert!(binary.is_absolute(), "NGSPICE_BIN must be an absolute path");
    let output = Command::new(binary)
        .args(["-b", "probe.cir"])
        .current_dir(&scratch.0)
        .output()
        .unwrap();
    let stdout = String::from_utf8(output.stdout).unwrap();
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(output.status.success(), "{stdout}\n{stderr}");
    let mut actual = BTreeMap::new();
    for line in stdout.lines() {
        if let Some((query, value)) = line.split_once(" = ")
            && query.starts_with('@')
        {
            actual.insert(query.to_owned(), value.trim().parse::<f64>().unwrap());
        }
    }
    assert_eq!(
        actual.len(),
        expected.len(),
        "missing queries:\n{stdout}\n{stderr}"
    );
    for (query, want) in expected {
        let got = actual[&query];
        assert!(
            got.is_finite() && (got - want).abs() <= 1e-12 * want.abs() + 1e-24,
            "{query}: Rust {want}, C {got}"
        );
    }
}

/// A process-local counter appended to scratch directory names.
///
/// `SystemTime::now()` is coarsened on some platforms (1 ms on macOS), so the
/// process id and the clock alone do not distinguish two tests of one binary
/// that start in the same tick - and tests run in parallel, so two of them
/// collided and `create_dir(...).unwrap()` panicked.
static SCRATCH_NONCE: AtomicU64 = AtomicU64::new(0);

fn scratch_nonce() -> u128 {
    let clock = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("the clock is after the epoch")
        .as_nanos();
    clock + u128::from(SCRATCH_NONCE.fetch_add(1, Ordering::Relaxed))
}

use std::sync::atomic::{AtomicU64, Ordering};

/// Alias decks use the same model and device names and explicit numerical
/// inputs; only the first level selector changes. Include both polarities.
const ALIAS_CASES: &[(&str, &str, &[u8])] = &[
    (
        "npn",
        "vc c 0 1.5\nvb b 0 0\nq1 c b 0 mdl\n.model mdl npn(is=1e-15 bf=100 vaf=50 level=LEVEL)\n.dc vb 0.5 0.8 0.05",
        &[0, 1, 2],
    ),
    (
        "pnp",
        "vc c 0 -1.5\nvb b 0 0\nq1 c b 0 mdl\n.model mdl pnp(is=1e-15 bf=100 vaf=50 level=LEVEL)\n.dc vb -0.5 -0.8 -0.05",
        &[0, 1, 2],
    ),
    (
        "nmos",
        "vd d 0 1.5\nvg g 0 0\nm1 d g 0 0 mdl l=1u w=5u\n.model mdl nmos(vto=0.5 kp=100u lambda=0.02 level=LEVEL)\n.dc vg 0 2 0.25",
        &[0, 1],
    ),
    (
        "pmos",
        "vd d 0 -1.5\nvg g 0 0\nm1 d g 0 0 mdl l=1u w=5u\n.model mdl pmos(vto=-0.5 kp=100u lambda=0.02 level=LEVEL)\n.dc vg 0 -2 -0.25",
        &[0, 1],
    ),
];

fn alias_deck(body: &str, level: u8) -> String {
    format!(
        "model selector alias\n{}\n.options reltol=1e-8\n.end\n",
        body.replace("LEVEL", &level.to_string())
    )
}

fn simulate_alias(text: &str) -> Plot {
    let netlist = Parser::new()
        .parse_deck(&parse_deck_text(Path::new("alias.cir"), text))
        .unwrap();
    let config = RunConfig::from_netlist(&netlist).unwrap();
    let mut circuit = config.circuit(&netlist).unwrap();
    config
        .plan_for(&netlist.analyses[0])
        .unwrap()
        .run(&mut circuit, &config.context())
        .unwrap()
}

#[test]
fn selector_aliases_use_identical_rust_physics() {
    for &(name, body, levels) in ALIAS_CASES {
        let expected = simulate_alias(&alias_deck(body, 1));
        assert!(expected.point_count() >= 7);
        for &level in levels {
            assert_eq!(
                simulate_alias(&alias_deck(body, level)),
                expected,
                "{name} level {level}"
            );
        }
    }
}

#[test]
#[ignore = "requires absolute NGSPICE_BIN; C's model selector alias dispatch"]
fn selector_aliases_match_pinned_c_dc_sweeps() {
    let c = Ngspice::pinned().unwrap();
    let scratch = Scratch::new("backend-aliases").unwrap();
    for &(name, body, levels) in ALIAS_CASES {
        for &level in levels {
            let text = alias_deck(body, level);
            let path = scratch.path().join(format!("{name}-{level}.cir"));
            fs::write(&path, &text).unwrap();
            let reference = capture(&c, &path, scratch.path(), &[]).unwrap();
            let plots = RawFile::parse(&reference.rawfile).unwrap();
            let reference = &plots.plots[0].plot;
            let rust = simulate_alias(&text);
            assert_eq!(
                rust.point_count(),
                reference.point_count(),
                "{name} level {level}"
            );
            let mut compared = 0;
            for variable in reference.variables() {
                let signal = variable.name.as_str();
                if !(signal.starts_with("v(") || signal.starts_with("i("))
                    || signal.contains("-sweep")
                {
                    continue;
                }
                compared += 1;
                for row in 0..reference.point_count() {
                    let want = reference.value(signal, row).unwrap();
                    let got = rust.value(signal, row).unwrap();
                    for (a, b) in [(got.re, want.re), (got.im, want.im)] {
                        assert!(
                            (a - b).abs() <= 1e-6 * b.abs() + 1e-12,
                            "{name} level {level} {signal}[{row}]: Rust {got:?}, C {want:?}"
                        );
                    }
                }
            }
            assert!(compared >= 3, "{name} level {level}");
        }
    }
}
