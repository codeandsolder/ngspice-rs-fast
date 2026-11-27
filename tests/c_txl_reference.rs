//! Opt-in independent C capture of TXL's ordinary transient fixtures.
//!
//! The C batch rawfile and the committed C ASCII capture use different
//! variable orders (especially the two identically named i(y1) branches),
//! dates, and build headers. Check all numerical payload cells by variable
//! name and occurrence, excluding only the non-numerical capture metadata.
//! The Rust-to-golden comparison is separately enforced by xtask golden
//! verify; the two checks together exercise live C without recapturing data.

use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

use ngspice_conformance::ngspice::Scratch;
use ngspice_conformance::process::{ProcessLimits, run};
use ngspice_rs::analysis::{Plot, RawFile};

fn check_capture(name: &str) {
    let binary = std::env::var_os("NGSPICE_BIN").expect("set absolute NGSPICE_BIN");
    assert!(Path::new(&binary).is_absolute());
    let root = ngspice_conformance::fixtures::workspace_root().unwrap();
    let scratch = Scratch::new("c-txl-reference").unwrap();
    let directory = scratch.path();
    fs::copy(
        root.join(format!("conformance/netlists/{name}.cir")),
        directory.join("deck.cir"),
    )
    .unwrap();
    let result = run(
        binary,
        directory,
        &["-b", "-r", "capture.raw", "deck.cir"],
        ProcessLimits::ORACLE,
    )
    .unwrap();
    assert!(
        result.status.success(),
        "{name}: {}\n{}",
        String::from_utf8_lossy(&result.stdout),
        String::from_utf8_lossy(&result.stderr)
    );
    let actual = RawFile::parse_bytes(&fs::read(directory.join("capture.raw")).unwrap()).unwrap();
    let reference = RawFile::load(root.join(format!("conformance/golden/{name}.raw"))).unwrap();
    assert_eq!(actual.len(), reference.len(), "{name}: plot count");
    for (i, (actual, reference)) in actual.plots.iter().zip(&reference.plots).enumerate() {
        let (got, want): (&Plot, &Plot) = (&actual.plot, &reference.plot);
        assert_eq!(got.plotname, want.plotname, "{name}: plot {i}");
        assert_eq!(got.flags(), want.flags(), "{name}: plot {i}");
        assert_eq!(got.point_count(), want.point_count(), "{name}: plot {i}");
        assert_eq!(
            got.variable_count(),
            want.variable_count(),
            "{name}: plot {i}"
        );
        let mut indices: BTreeMap<&str, Vec<usize>> = BTreeMap::new();
        for (j, variable) in got.variables().iter().enumerate() {
            indices.entry(&variable.name).or_default().push(j);
        }
        let mut occurrence: BTreeMap<&str, usize> = BTreeMap::new();
        let mut permutation = Vec::with_capacity(want.variable_count());
        for variable in want.variables() {
            let seen = occurrence.entry(&variable.name).or_default();
            let column = *indices
                .get(variable.name.as_str())
                .and_then(|matches| matches.get(*seen))
                .unwrap_or_else(|| panic!("{name}: missing {} occurrence {}", variable.name, seen));
            assert_eq!(got.variables()[column].unit, variable.unit);
            permutation.push(column);
            *seen += 1;
        }
        for point in 0..want.point_count() {
            let c = got.row(point).unwrap();
            let capture = want.row(point).unwrap();
            for (j, &column) in permutation.iter().enumerate() {
                for (value, expected) in
                    [(c[column].re, capture[j].re), (c[column].im, capture[j].im)]
                {
                    let bound = 1e-14 * expected.abs() + 1e-300;
                    assert!(
                        (value - expected).abs() <= bound,
                        "{name}: plot {i}, point {point}, {}[{}]: C {value:e}, ASCII {expected:e}, bound {bound:e}",
                        want.variables()[j].name,
                        j
                    );
                }
            }
        }
    }
}

#[test]
#[ignore = "requires NGSPICE_BIN, original C batch captures against committed C ASCII references"]
fn c_txl_transient_captures_match_c_ascii_reference() {
    for name in ["m10_txl_lossless_tran", "m10_txl_tran", "m10_txl_tran_step"] {
        check_capture(name);
    }
}
