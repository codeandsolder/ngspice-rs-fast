//! Model selection and opt-in live C setup / selector-alias comparisons.
use ngspice_conformance::ngspice::{Ngspice, Scratch, capture};
use ngspice_conformance::process::{ProcessLimits, run};
use ngspice_rs::analysis::{Plot, RawFile, RunConfig};
use ngspice_rs::devices::{ModelContext, ModelResolver};
use ngspice_rs::netlist::Parser;
use ngspice_rs::netlist::source::parse_deck_text;
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

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
    let scratch = Scratch::new("model-c-reference").unwrap();
    fs::write(scratch.path().join("probe.cir"), text).unwrap();
    let binary = PathBuf::from(std::env::var_os("NGSPICE_BIN").expect("set NGSPICE_BIN"));
    assert!(binary.is_absolute(), "NGSPICE_BIN must be an absolute path");
    let output = run(
        binary,
        scratch.path(),
        &["-b", "probe.cir"],
        ProcessLimits::ORACLE,
    )
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
    (
        "r",
        "v1 a 0 1\nr1 a b mdl l=10u w=2u\nr2 b 0 1k\n.model mdl r(rsh=100 level=LEVEL)\n.dc v1 0.5 2 0.25",
        &[0, 1],
    ),
    (
        "njf",
        "vd d 0 3\nvg g 0 0\nj1 d g 0 mdl\n.model mdl njf(vto=-2 beta=1m lambda=0.02 b=0.8 level=LEVEL)\n.dc vg -2.5 0 0.25",
        &[0, 1],
    ),
    (
        "pjf",
        "vd d 0 -3\nvg g 0 0\nj1 d g 0 mdl\n.model mdl pjf(vto=-2 beta=1m lambda=0.02 b=0.8 level=LEVEL)\n.dc vg 2.5 0 -0.25",
        &[0, 1],
    ),
    (
        "nmf",
        "vd d 0 3\nvg g 0 0\nz1 d g 0 mdl\n.model mdl nmf(vto=-1.8 beta=3m lambda=0.05 alpha=2.5 b=0.4 level=LEVEL)\n.dc vg -2.5 0 0.25",
        &[0, 1],
    ),
    (
        "pmf",
        "vd d 0 -3\nvg g 0 0\nz1 d g 0 mdl\n.model mdl pmf(vto=-1.8 beta=3m lambda=0.05 alpha=2.5 b=0.4 level=LEVEL)\n.dc vg 2.5 0 -0.25",
        &[0, 1],
    ),
    (
        "bsim3_nmos",
        "vd d 0 1.2\nvg g 0 0\nm1 d g 0 0 mdl l=0.18u w=1u ad=1e-13 as=1e-13 pd=1.4u ps=1.4u\n.model mdl nmos(version=3.3.0 capmod=3 mobmod=1 tox=4.1n nch=2.3549e17 vth0=0.3694 u0=268.7 level=LEVEL)\n.dc vg 0 1.2 0.15",
        &[8, 49],
    ),
    (
        "bsim4_nmos",
        "vd d 0 1.2\nvg g 0 0\nm1 d g 0 0 mdl l=0.18u w=1u ad=1e-13 as=1e-13 pd=1.4u ps=1.4u\n.model mdl nmos(version=4.8.3 vth0=0.45 toxe=4n level=LEVEL)\n.dc vg 0 1.2 0.15",
        &[14, 54],
    ),
    (
        "bsim3_pmos",
        "vd d 0 -1.2\nvg g 0 0\nm1 d g 0 0 mdl l=0.18u w=1u ad=1e-13 as=1e-13 pd=1.4u ps=1.4u\n.model mdl pmos(version=3.3.0 capmod=3 mobmod=1 tox=4.1n nch=2.3549e17 vth0=-0.3694 u0=268.7 level=LEVEL)\n.dc vg 0 -1.2 -0.15",
        &[8, 49],
    ),
    (
        "bsim4_pmos",
        "vd d 0 -1.2\nvg g 0 0\nm1 d g 0 0 mdl l=0.18u w=1u ad=1e-13 as=1e-13 pd=1.4u ps=1.4u\n.model mdl pmos(version=4.8.3 vth0=-0.45 toxe=4n level=LEVEL)\n.dc vg 0 -1.2 -0.15",
        &[14, 54],
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
        let expected = simulate_alias(&alias_deck(body, levels[0]));
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
