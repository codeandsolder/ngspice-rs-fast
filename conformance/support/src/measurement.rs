//! ngspice batch measurement reports, independent of simulator result types.
//!
//! The pinned C oracle emits "Measurements for DC/AC/SP/Transient Analysis"
//! sections terminated by a blank line; `.measure PARAM` (and the normal
//! second pass for scalar measurements) writes the same name/value rows.
//! The grammar follows `measure.c` and `com_measure2.c` at C commit
//! 059066b5. Only batch reports with analysis headers are supported.
//! Numerical failures may instead be reported on stderr as a failed card.
//! Never interpret the subsequent resource statistics as measurements.

use std::collections::BTreeMap;

/// Analyses for which ngspice emits batch measurement reports.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Analysis {
    /// DC sweep.
    Dc,
    /// AC sweep.
    Ac,
    /// S parameters.
    Sp,
    /// Transient analysis.
    Tran,
}

impl Analysis {
    fn header(text: &str) -> Option<Self> {
        match text {
            "DC" => Some(Self::Dc),
            "AC" => Some(Self::Ac),
            "SP" => Some(Self::Sp),
            "Transient" => Some(Self::Tran),
            _ => None,
        }
    }

    /// Recognizes the corresponding netlist analysis keyword.
    #[must_use = "the parsed analysis selects the measurement evaluation domain"]
    pub fn keyword(text: &str) -> Option<Self> {
        match text {
            "dc" => Some(Self::Dc),
            "ac" => Some(Self::Ac),
            "sp" => Some(Self::Sp),
            "tran" => Some(Self::Tran),
            _ => None,
        }
    }
}

/// Optional coordinates printed alongside the numerical result.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Coordinates {
    /// Extremum/sample position.
    pub at: Option<f64>,
    /// Signal value at the axis coordinate returned by `MIN_AT` or `MAX_AT`.
    pub with: Option<f64>,
    /// Start of the measurement window.
    pub from: Option<f64>,
    /// End of the measurement window.
    pub to: Option<f64>,
    /// Trigger position.
    pub trig: Option<f64>,
    /// Target position.
    pub targ: Option<f64>,
}

impl Coordinates {
    const fn valid(self, unit: Option<ReportedUnit>) -> bool {
        let Self {
            at,
            with,
            from,
            to,
            trig,
            targ,
        } = self;
        matches!(
            (at, with, from, to, trig, targ, unit),
            (None, _, None, None, None, None, None)
                | (Some(_), None, None, None, None, None, _)
                | (None, None, Some(_), Some(_), None, None, None)
                | (None, None, None, None, Some(_), Some(_), None)
        )
    }
}

/// One reported outcome, including failed scalar expressions.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Outcome {
    /// A finite numerical result.
    Value(f64),
    /// C printed `failed` instead of a value.
    Failed,
}

/// Units explicitly printed by ngspice's margin measurements.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReportedUnit {
    /// Phase margin, in degrees.
    Degrees,
    /// Gain margin, in decibels.
    Decibels,
}

/// One stdout row. Duplicates are retained, not collapsed into a map.
#[derive(Debug, Clone, PartialEq)]
pub struct Row {
    /// Measurement name.
    pub name: String,
    /// Reported value or failure.
    pub outcome: Outcome,
    /// Unit suffix, if explicitly present.
    pub unit: Option<ReportedUnit>,
    /// Supplementary axis coordinates.
    pub coordinates: Coordinates,
}

/// One measurement report in stdout order.
#[derive(Debug, Clone, PartialEq)]
pub struct Section {
    /// Analysis label emitted by C. The parser does not infer a different
    /// analysis from the numerical values or the Rust results.
    pub analysis: Analysis,
    /// Zero-based run as delimited by C's analysis-start banner, or 0
    /// for a transcript without banners.
    pub run: usize,
    /// Results in the order C prints them, including duplicate names.
    pub rows: Vec<Row>,
}

/// A failed measurement card printed on stderr and possibly absent from stdout.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Failure {
    /// Analysis named by the card.
    pub analysis: Analysis,
    /// Measurement name.
    pub name: String,
    /// Run, if unambiguously associable with a stdout section.
    /// Separate stdout/stderr capture cannot order a failure among repeated
    /// sections of the same analysis; those failures remain unresolved.
    pub run: Option<usize>,
    /// Entire diagnostic card line.
    pub diagnostic: String,
}

/// Parsed stdout sections and stderr failures. No outcome is discarded.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Report {
    /// Sections in source order.
    pub sections: Vec<Section>,
    /// Failed cards in diagnostic order.
    pub failures: Vec<Failure>,
}

fn parse_number(text: &str) -> Result<f64, String> {
    let value = text
        .parse::<f64>()
        .map_err(|_| format!("invalid measurement number {text:?}"))?;
    if !value.is_finite() {
        return Err(format!("nonfinite measurement number {text:?}"));
    }
    Ok(value)
}

fn parse_row(line: &str) -> Result<Row, String> {
    let (name, tail) = line
        .split_once('=')
        .ok_or_else(|| "expected name = value".to_owned())?;
    let name = name.trim();
    if name.is_empty() || name.chars().any(char::is_whitespace) {
        return Err(format!("invalid measurement name {name:?}"));
    }
    let mut words = tail.trim().splitn(2, char::is_whitespace);
    let value = words.next().unwrap_or_default();
    let outcome = if value == "failed" {
        Outcome::Failed
    } else {
        Outcome::Value(parse_number(value)?)
    };
    let mut rest = words.next().unwrap_or_default().trim();
    if matches!(outcome, Outcome::Failed) && !rest.is_empty() {
        return Err("unexpected metadata after a failed measurement".to_owned());
    }
    let mut unit = None;
    if let Some((prefix, tail)) = rest.split_once(char::is_whitespace) {
        unit = match prefix {
            "deg" => Some(ReportedUnit::Degrees),
            "dB" => Some(ReportedUnit::Decibels),
            _ => None,
        };
        if unit.is_some() {
            rest = tail.trim();
        }
    }
    let mut coordinates = Coordinates::default();
    while !rest.is_empty() {
        let (key, tail) = rest
            .split_once('=')
            .ok_or_else(|| format!("invalid coordinate {rest:?}"))?;
        let field = match key.trim() {
            "at" => &mut coordinates.at,
            "with" => &mut coordinates.with,
            "from" => &mut coordinates.from,
            "to" => &mut coordinates.to,
            "trig" => &mut coordinates.trig,
            "targ" => &mut coordinates.targ,
            key => return Err(format!("unknown measurement coordinate {key:?}")),
        };
        let mut words = tail.trim().splitn(2, char::is_whitespace);
        let value = parse_number(words.next().unwrap_or_default())?;
        if field.replace(value).is_some() {
            return Err(format!("duplicate coordinate {key:?}"));
        }
        rest = words.next().unwrap_or_default().trim();
    }
    if !coordinates.valid(unit) {
        return Err("incomplete or conflicting measurement metadata".to_owned());
    }
    Ok(Row {
        name: name.to_owned(),
        outcome,
        unit,
        coordinates,
    })
}

fn header(line: &str) -> Result<Option<Analysis>, String> {
    let Some(label) = line.strip_prefix("Measurements for ") else {
        return Ok(None);
    };
    let kind = label
        .strip_suffix(" Analysis")
        .and_then(Analysis::header)
        .ok_or_else(|| format!("unknown measurement section {line:?}"))?;
    Ok(Some(kind))
}

fn failed_card(line: &str) -> Result<Option<Failure>, String> {
    let mut words = line.split_whitespace();
    if !matches!(words.next(), Some(".meas" | ".measure")) || !line.ends_with("failed!") {
        return Ok(None);
    }
    let analysis = words
        .next()
        .and_then(Analysis::keyword)
        .ok_or_else(|| format!("unknown failed measurement analysis: {line}"))?;
    let name = words
        .next()
        .ok_or_else(|| format!("missing failed measurement name: {line}"))?;
    Ok(Some(Failure {
        analysis,
        name: name.to_owned(),
        run: None,
        diagnostic: line.to_owned(),
    }))
}

/// Parses the two separately captured streams of a C batch run.
///
/// # Errors
/// A malformed or unknown measurement section/row, coordinate or failed
/// measurement card. Lines outside report sections are not interpreted as rows.
pub fn parse(stdout: &str, stderr: &str) -> Result<Report, String> {
    let mut report = Report::default();
    let mut active: Option<Section> = None;
    let mut run = 0;
    let mut saw_banner = false;
    let mut saw_header_gap = false;
    for (index, line) in stdout.lines().enumerate() {
        let line = line.trim();
        if line.starts_with("Doing analysis at TEMP =") {
            report.sections.extend(active.take());
            run += usize::from(saw_banner);
            saw_banner = true;
        } else if let Some(analysis) = header(line)? {
            report.sections.extend(active.replace(Section {
                analysis,
                run,
                rows: Vec::new(),
            }));
            saw_header_gap = false;
        } else if let Some(section) = &mut active {
            if line.is_empty() {
                // C prints one blank line after the header, then a blank
                // separator after the rows (even when all cards failed).
                if saw_header_gap || !section.rows.is_empty() {
                    report.sections.extend(active.take());
                }
                saw_header_gap = true;
            } else {
                section.rows.push(parse_row(line).map_err(|error| {
                    format!("measurement report line {}: {error}: {line}", index + 1)
                })?);
            }
        }
    }
    report.sections.extend(active);
    for line in stderr.lines().map(str::trim) {
        if let Some(mut failure) = failed_card(line)? {
            let mut sections = report
                .sections
                .iter()
                .filter(|section| section.analysis == failure.analysis);
            if let Some(section) = sections.next()
                && sections.next().is_none()
            {
                failure.run = Some(section.run);
            }
            report.failures.push(failure);
        }
    }
    Ok(report)
}

/// Results produced by the simulator under test for one reported analysis/run.
#[derive(Debug, Clone, PartialEq)]
pub struct ActualSection {
    /// Expected analysis.
    pub analysis: Analysis,
    /// Zero-based expected run number.
    pub run: usize,
    /// Every result of this analysis, with the original name and value.
    pub values: Vec<(String, f64)>,
}

/// Compare complete ordered analysis/run sections.
///
/// Never compare a subset selected by the simulator under test. Within a
/// section, names rather than print ordering identify measurements;
/// duplicate names are errors.
///
/// `bound` returns the maximum allowed absolute error for a C reference value.
///
/// # Errors
/// Unexpected section/run, failed card or row, missing/extra/duplicate names,
/// nonfinite result, invalid bound, or an exceeded numerical bound.
pub fn compare(
    report: &Report,
    actual: &[ActualSection],
    bound: impl Fn(f64) -> f64,
) -> Result<(), String> {
    if !report.failures.is_empty() {
        return Err(format!("C failed measurements: {:?}", report.failures));
    }
    if report.sections.len() != actual.len() {
        return Err(format!(
            "different measurement section counts: C {}, Rust {}",
            report.sections.len(),
            actual.len()
        ));
    }
    if actual.is_empty() {
        return Err("neither engine measured anything".to_owned());
    }
    for (section, ours) in report.sections.iter().zip(actual) {
        if section.analysis != ours.analysis || section.run != ours.run {
            return Err(format!(
                "unexpected C section {:?} run {}; Rust {:?} run {}",
                section.analysis, section.run, ours.analysis, ours.run
            ));
        }
        compare_section(section, &ours.values, &bound)
            .map_err(|error| format!("{:?} run {}: {error}", section.analysis, section.run))?;
    }
    Ok(())
}

fn compare_section(
    section: &Section,
    actual: &[(String, f64)],
    bound: &impl Fn(f64) -> f64,
) -> Result<(), String> {
    let mut expected = BTreeMap::new();
    for row in &section.rows {
        let Outcome::Value(value) = row.outcome else {
            return Err(format!("C measurement {:?} failed", row.name));
        };
        if !value.is_finite() {
            return Err(format!("nonfinite C measurement {:?}", row.name));
        }
        if expected.insert(row.name.as_str(), value).is_some() {
            return Err(format!("duplicate C measurement {:?}", row.name));
        }
    }
    let mut seen = std::collections::BTreeSet::new();
    for (name, value) in actual {
        if !seen.insert(name.as_str()) {
            return Err(format!("duplicate Rust measurement {name:?}"));
        }
        let Some(reference) = expected.remove(name.as_str()) else {
            return Err(format!("extra Rust measurement {name:?}"));
        };
        let error = (value - reference).abs();
        let allowed = bound(reference);
        if !allowed.is_finite() || allowed < 0. {
            return Err(format!("invalid numerical bound {allowed} for {name:?}"));
        }
        if !value.is_finite() || error > allowed {
            return Err(format!(
                "{name}: Rust {value:e}, C {reference:e}, error {error:e}"
            ));
        }
    }
    if !expected.is_empty() {
        return Err(format!("missing Rust measurements: {expected:?}"));
    }
    if seen.is_empty() {
        return Err("neither engine measured anything".to_owned());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    // Extracted from pinned ngspice-47+ batch probes on 2026-11-22.
    const DC: &str = "  Measurements for DC Analysis\n\nstack               =  1.00000e+00 at=  2.00000e+00\narea                =   1.00000e+00 from=  0.00000e+00 to=  2.00000e+00\ncrossing            =   1.00000e+00\nfound               =  5.00000e-01\nscalar              =  2.00000e+00\ntwice               =  3.00000e+00\n\n\nTotal analysis time (seconds) = 0.000603765\nStack = 0 bytes.\n";
    const AC: &str = "  Measurements for AC Analysis\n\npeak                =  5.00000e-01 at=  1.00000e+04\natfrequency         =  5.00000e-01\n\n";
    const SP: &str = "  Measurements for SP Analysis\n\ntransmission        =  3.80952e-01 at=  1.00000e+03\natfreq              =  3.80952e-01\n\n";
    const TRAN: &str = "  Measurements for Transient Analysis\n\ndelta               =  2.00000e-03 targ=  2.00050e-03 trig=  5.00000e-07\nrms                 =   3.53657e-01 from=  1.00000e-03 to=  3.00000e-03\nexprval             =  7.07315e-01\n\n";

    fn one(
        analysis: Analysis,
        values: impl IntoIterator<Item = (String, f64)>,
    ) -> [ActualSection; 1] {
        [ActualSection {
            analysis,
            run: 0,
            values: values.into_iter().collect(),
        }]
    }

    #[test]
    fn compare_uses_all_sections_and_run_identities() {
        let banner = "Doing analysis at TEMP = 27.000000 and TNOM = 27.000000\n";
        let dc = "Measurements for DC Analysis\n\nstack = 1\n\n";
        let report = parse(&format!("{banner}{dc}{AC}{banner}{dc}"), "").unwrap();
        let mut actual = [
            ActualSection {
                analysis: Analysis::Dc,
                run: 0,
                values: vec![("stack".into(), 1.)],
            },
            ActualSection {
                analysis: Analysis::Ac,
                run: 0,
                values: vec![("peak".into(), 0.5), ("atfrequency".into(), 0.5)],
            },
            ActualSection {
                analysis: Analysis::Dc,
                run: 1,
                values: vec![("stack".into(), 1.)],
            },
        ];
        assert!(compare(&report, &actual, |_| 0.).is_ok());
        assert!(compare(&report, &actual[..2], |_| 0.).is_err());
        actual[2].run = 0;
        assert!(compare(&report, &actual, |_| 0.).is_err());
        actual[2].run = 1;
        actual[2].analysis = Analysis::Sp;
        assert!(compare(&report, &actual, |_| 0.).is_err());
    }

    #[test]
    fn pinned_reports_preserve_order_coordinates_and_stack() {
        let report = parse(&format!("{DC}{AC}{SP}{TRAN}"), "").unwrap();
        assert_eq!(
            report
                .sections
                .iter()
                .map(|s| s.analysis)
                .collect::<Vec<_>>(),
            [Analysis::Dc, Analysis::Ac, Analysis::Sp, Analysis::Tran]
        );
        let rows = &report.sections[0].rows;
        assert_eq!(
            rows.iter().map(|r| r.name.as_str()).collect::<Vec<_>>(),
            ["stack", "area", "crossing", "found", "scalar", "twice"]
        );
        assert_eq!(rows[0].coordinates.at, Some(2.));
        assert_eq!(rows[1].coordinates.from, Some(0.));
        assert_eq!(rows[1].coordinates.to, Some(2.));
        assert_eq!(rows[5].outcome, Outcome::Value(3.));
        assert_eq!(report.sections[3].rows[0].coordinates.trig, Some(5e-7));
        assert_eq!(report.sections[3].rows[0].coordinates.targ, Some(2.0005e-3));
    }

    #[test]
    fn repeated_runs_and_stderr_failures_are_not_conflated() {
        let banner = "Doing analysis at TEMP = 27.000000 and TNOM = 27.000000\n";
        let failure = "Error: measure nohit when(WHEN) : out of interval\n .meas dc nohit when v(in)=2 failed!\n";
        let single = parse(DC, failure).unwrap();
        assert_eq!(single.failures[0].run, Some(0));
        let repeated = parse(&format!("{banner}{DC}{banner}{DC}"), failure).unwrap();
        assert_eq!(repeated.sections.len(), 2);
        assert_eq!(repeated.sections[1].run, 1);
        assert_eq!(repeated.failures[0].run, None);
        assert!(compare(&single, &one(Analysis::Dc, []), |_| 0.).is_err());
    }

    #[test]
    fn extent_ordinates_and_margin_units_are_distinct_from_axis_coordinates() {
        let report = parse("Measurements for DC Analysis\n\nlow                 =  0.00000e+00 with=  0.00000e+00\nhigh                =  2.00000e+00 with=  2.00000e+00\nrange               =  2.00000e+00 from=  -1.00000e+99 to=  1.00000e+99\n\nMeasurements for AC Analysis\n\npm                  =  -7.02913e+00 deg at=  3.03708e+02\ngm                  =  -1.93801e+00 dB at=  2.75666e+02\n\n", "").unwrap();
        assert_eq!(report.sections[0].rows[1].coordinates.with, Some(2.));
        assert_eq!(report.sections[0].rows[1].coordinates.at, None);
        assert_eq!(report.sections[0].rows[2].coordinates.from, Some(-1e99));
        assert_eq!(report.sections[1].rows[0].unit, Some(ReportedUnit::Degrees));
        assert_eq!(report.sections[1].rows[0].coordinates.at, Some(303.708));
        assert_eq!(
            report.sections[1].rows[1].unit,
            Some(ReportedUnit::Decibels)
        );
    }

    #[test]
    fn failure_only_report_does_not_consume_the_resource_trailer() {
        let report = parse(
            "Measurements for DC Analysis\n\n\n\nTotal analysis time (seconds) = 0.00339845\nStack = 0 bytes.\n",
            "Error: measure nohit when(WHEN) : out of interval\n .meas dc nohit when v(in)=2 failed!\n",
        ).unwrap();
        assert!(report.sections[0].rows.is_empty());
        assert_eq!(report.failures[0].run, Some(0));
        assert_eq!(report.failures[0].name, "nohit");
        assert!(compare(&report, &one(Analysis::Dc, []), |_| 0.).is_err());
    }

    #[test]
    fn failures_and_duplicates_are_preserved_and_rejected() {
        let report = parse("Measurements for DC Analysis\n\nmissing = failed\ninvalid = failed\nduplicate = 1 at=1\nduplicate = 0 at=0\n\n", "").unwrap();
        assert_eq!(report.sections[0].rows.len(), 4);
        assert_eq!(report.sections[0].rows[0].outcome, Outcome::Failed);
        assert_eq!(
            report.sections[0].rows[2].name,
            report.sections[0].rows[3].name
        );
        assert!(compare(&report, &one(Analysis::Dc, []), |_| 0.).is_err());
        let duplicate = parse("Measurements for DC Analysis\n\nx = 1\nx = 1\n", "").unwrap();
        assert!(
            compare(&duplicate, &one(Analysis::Dc, [("x".into(), 1.)]), |_| 0.)
                .unwrap_err()
                .contains("duplicate C")
        );
    }

    #[test]
    fn comparison_checks_exact_name_set_without_changing_bounds() {
        let report = parse(AC, "").unwrap();
        let check = |rows| compare(&report, &one(Analysis::Ac, rows), |_| 1e-5);
        assert!(check(vec![("atfrequency".into(), 0.5), ("peak".into(), 0.5)]).is_ok());
        assert!(
            check(vec![("peak".into(), 0.5)])
                .unwrap_err()
                .contains("missing Rust")
        );
        assert!(
            check(vec![("unexpected".into(), 0.5)])
                .unwrap_err()
                .contains("extra Rust")
        );
        assert!(
            check(vec![("peak".into(), 0.5), ("peak".into(), 0.5)])
                .unwrap_err()
                .contains("duplicate Rust")
        );
        assert!(check(vec![("peak".into(), 0.5001), ("atfrequency".into(), 0.5)]).is_err());
        assert!(check(vec![("peak".into(), f64::NAN), ("atfrequency".into(), 0.5)]).is_err());
    }

    #[test]
    fn malformed_report_is_not_silently_filtered() {
        for row in [
            "bad",
            "x = ???",
            "x = NaN",
            "x = 1 at=2 at=3",
            "x = 1 unknown=3",
            "x = 1 at=",
            "x = 1 bytes.",
            "x = 1 from=2",
            "x = 1 at=2 with=3",
            "x = failed at=2",
            "x = 1 dB from=0 to=1",
        ] {
            assert!(
                parse(&format!("Measurements for DC Analysis\n\n{row}\n\n"), "").is_err(),
                "{row}"
            );
        }
        assert!(parse("Measurements for Unknown Analysis\n\nx = 1", "").is_err());
        assert!(parse("", ".meas wrong bad failed!").is_err());
        let empty = parse("", "").unwrap();
        assert!(compare(&empty, &[], |_| 0.).is_err());
    }
}
