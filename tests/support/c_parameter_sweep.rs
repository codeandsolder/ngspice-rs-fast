//! Live-ngspice oracle for nested instance-parameter sweeps.
//!
//! ngspice 47's `.dc` accepts sources and resistors, not arbitrary
//! `@device[parameter]` axes. A fresh C circuit at each grid point is more
//! general than `alter`: some devices (including MESFET) do not recompute
//! their area-dependent setup state after an `alter` command.
use ngspice_conformance::ngspice::{Ngspice, Scratch};
use std::fs;

use ngspice_rs::analysis::{Plot, RawFile};
use ngspice_rs::cli::simulate;

/// A concrete instance parameter `@instance[keyword]` from a typed sweep.
fn parameter(axis: &str) -> (&str, &str) {
    let source = axis
        .strip_prefix('@')
        .expect("parameter sweep uses @instance[keyword]");
    let (instance, keyword) = source.split_once('[').expect("parameter sweep keyword");
    let keyword = keyword
        .strip_suffix(']')
        .expect("closing parameter bracket");
    assert!(!instance.is_empty() && !keyword.is_empty());
    assert!(
        instance
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '_')
    );
    assert!(
        keyword
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
    );
    (instance, keyword)
}

/// Build the same deck with its nested `.dc` replaced by one fresh C
/// operating point. Last setters override the original instance defaults.
fn point_deck(deck: &str, inner: (&str, f64), outer: (&str, f64), index: usize) -> String {
    let axes = [(parameter(inner.0), inner.1), (parameter(outer.0), outer.1)];
    let mut matched = [false; 2];
    let mut dc_count = 0;
    let mut output = String::new();
    for line in deck.lines() {
        if line.trim_start().to_ascii_lowercase().starts_with(".dc ") {
            dc_count += 1;
            continue;
        }
        output.push_str(line);
        let device = line.split_whitespace().next().unwrap_or("");
        for (i, ((instance, keyword), value)) in axes.iter().enumerate() {
            if device.eq_ignore_ascii_case(instance) {
                matched[i] = true;
                output.push_str(&format!(" {keyword}={value:.17e}"));
            }
        }
        output.push('\n');
    }
    assert_eq!(dc_count, 1, "expect exactly one nested DC sweep");
    assert!(
        matched.into_iter().all(|found| found),
        "swept device not found in circuit"
    );
    output.push_str(&format!(
        ".op\n.control\nset filetype=ascii\nrun\nwrite point-{index}.raw\nquit\n.endc\n.end\n"
    ));
    output
}

/// Run Rust's nested `.dc` and independent fresh C operating points for
/// each parameter combination. The inner parameter varies fastest.
pub(crate) fn nested(
    tag: &str,
    deck: &str,
    inner: (&str, &[f64]),
    outer: (&str, &[f64]),
) -> (Plot, Plot) {
    let binary = Ngspice::pinned().expect("pinned C reference");
    assert!(!inner.1.is_empty() && !outer.1.is_empty());
    assert!(inner.1.iter().chain(outer.1).all(|v| v.is_finite()));

    let scratch = Scratch::new(tag).expect("scratch directory");
    let dir = scratch.path();

    let mut reference: Option<Plot> = None;
    let mut index = 0;
    for outer_value in outer.1 {
        for inner_value in inner.1 {
            let text = point_deck(
                deck,
                (inner.0, *inner_value),
                (outer.0, *outer_value),
                index,
            );
            let input = format!("point-{index}.cir");
            fs::write(dir.join(&input), text).unwrap();
            let output = binary.run(dir, &["-b", &input]).expect("run C");
            let transcript = output.transcript();
            output
                .require_success(&format!("C point {index}"))
                .unwrap_or_else(|error| panic!("{error}"));
            assert!(output.status.success(), "C point {index}: {transcript}");
            assert!(
                !transcript.contains("Fatal error:") && !transcript.contains("Error:"),
                "C point {index}: {transcript}"
            );
            let raw = RawFile::load(dir.join(format!("point-{index}.raw")))
                .unwrap_or_else(|e| panic!("C point {index}: {e}\n{transcript}"));
            assert_eq!(raw.plots.len(), 1, "C point {index}: one OP plot");
            let sample = raw.plots.into_iter().next().unwrap().plot;
            assert_eq!(sample.plotname, "Operating Point");
            assert_eq!(sample.points.len(), 1);
            if let Some(reference) = &mut reference {
                assert_eq!(
                    reference.variables, sample.variables,
                    "C vectors changed at point {index}"
                );
                reference.points.push(sample.points[0].clone());
            } else {
                reference = Some(sample);
            }
            index += 1;
        }
    }
    let reference = reference.expect("nonempty sweep grid");

    fs::write(dir.join("r.cir"), format!("{deck}.end\n")).unwrap();
    simulate::run(&dir.join("r.cir"), &dir.join("r.raw"), true).unwrap();
    let raw = RawFile::load(dir.join("r.raw")).unwrap();
    assert_eq!(raw.plots.len(), 1);
    let ours = raw.plots.into_iter().next().unwrap().plot;
    assert_eq!(ours.points.len(), index);

    // Only the inner axis is written to the Rust DC plot. Check that its
    // order agrees with the explicit C point order, not merely its count.
    let axis = ours
        .variable_index("v(param-sweep)")
        .expect("inner parameter scale");
    for (point, expected) in ours.points.iter().zip(inner.1.iter().cycle()) {
        assert_eq!(point[axis].re, *expected, "parameter sweep point order");
    }

    (ours, reference)
}
