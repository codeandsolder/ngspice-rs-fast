//! Numerical conformance policy, separate from C rawfile byte-drift checks.
//!
//! Names/units follow `src/frontend/rawfile.c`; Rust MNA column order is not
//! ngspice's vector order. Bounds retain the production DC/AC regression policy.

use std::collections::{BTreeMap, VecDeque};

use ngspice_rs::analysis::{Plot, PlotFlags};

#[derive(Debug, Clone, Copy)]
pub(super) struct Tolerance {
    relative: f64,
    absolute: f64,
}

pub(super) const DC: Tolerance = Tolerance {
    relative: 1e-12,
    absolute: 1e-15,
};
pub(super) const AC: Tolerance = Tolerance {
    relative: 1e-10,
    absolute: 1e-12,
};

/// Nonlinear bias/AC allow 1 ppm for independently converged bias Jacobians,
/// plus a 1 p-unit floor. The linear LU-only tolerances remain unchanged.
pub(super) const NONLINEAR: Tolerance = Tolerance {
    relative: 1e-6,
    absolute: 1e-12,
};

/// `.noise` (#100), linear circuits: the AC bound relative to each value, with
/// an absolute floor scaled to noise rather than to node voltages. Densities
/// are about 1e-9 V/sqrt(Hz) and integrated totals 1e-7 V and up, so a 1e-12
/// floor would hide whole generators; 1e-20 sits far below any physical
/// density yet above the residue C's adjoint leaves for a generator whose
/// transfer is exactly zero (e.g. about 2e-25 V/sqrt(Hz) for a resistor in
/// series with the current-source input).
pub(super) const NOISE: Tolerance = Tolerance {
    relative: 1e-10,
    absolute: 1e-20,
};

/// `.noise` of nonlinear circuits: the nonlinear 1 ppm bias bound (the
/// generators are evaluated at independently converged operating points)
/// with the noise floor of [`NOISE`].
pub(super) const NOISE_NONLINEAR: Tolerance = Tolerance {
    relative: 1e-6,
    absolute: 1e-20,
};

/// `.sens` (#102), linear circuits. A sensitivity is a forward difference
/// `dx/delta` with `delta = 1e-6 p` (C's `Sens_Delta`), so the rounding of the
/// solution `x` reaches it amplified by `1/delta`: for a parameter with a
/// strong effect that is about `1e-10` relative, which the 1e-8 bound covers
/// with margin. Parameters C perturbs without effect read exactly 0 (or a
/// rounding residue, e.g. about 4e-14 from the cosine of a 1e-6 degree AC
/// phase), which the 1e-12 floor accepts.
pub(super) const SENSITIVITY: Tolerance = Tolerance {
    relative: 1e-8,
    absolute: 1e-12,
};

/// `.sens` of nonlinear circuits: the nonlinear 1 ppm bias bound, with an
/// absolute floor for weak parameters, whose numerator `dx` is only a few
/// ulps of the node voltage: `ulp(x)/delta` is about `1e-16/2e-6 = 5e-11`
/// for an O(1) parameter of an O(1) V output (C's own result for a diode's
/// `nr` moves by 3e-10 between RELTOL 1e-6 and 1e-9 for that reason), which a
/// 1e-9 floor covers.
pub(super) const SENSITIVITY_NONLINEAR: Tolerance = Tolerance {
    relative: 1e-6,
    absolute: 1e-9,
};

/// `.sens` of MESA level 2 (#164): the nonlinear bound with a wider floor for
/// three columns. `mesatemp.c` computes its derived quantities once per
/// `DEVtemperature` and the load reuses them, while the port recomputes them
/// from the records at every load; the two agree to ~1e-16, but a column whose
/// answer is a difference of much larger terms amplifies that. Measured on
/// `sens_mesa` against the reference binary: `z::level` 3.4e-6 relative (the
/// perturbation disables the channel, so its answer is a difference of whole
/// device currents) and `z::teta1`/`z::xtm0` 4.6e-8/6e-9 absolute (parameters
/// level 2 does not read, where C reports its own rounding residue); every
/// other column is inside [`SENSITIVITY_NONLINEAR`]. The bound still rejects a
/// real model difference - it is 1e-5, and the MESA `tRsi` defect this fixture
/// family exposed sat at 1e-2..2e-1.
pub(super) const SENSITIVITY_CANCELLED: Tolerance = Tolerance {
    relative: 1e-5,
    absolute: 5e-8,
};

/// `.disto` (#104): the nonlinear 1 ppm operating-point bound (every Taylor
/// coefficient is evaluated at an independently converged bias), with an
/// absolute floor scaled to distortion products rather than node voltages.
/// Harmonic and intermodulation responses reach down to about 1e-16 V/A for
/// small branch currents, which a 1e-12 floor would hide entirely; 1e-18
/// stays below every such response yet above the residue (about 1e-28) a
/// node fixed by a voltage source keeps in the Rust LU solve where C writes
/// an exact zero.
pub(super) const DISTORTION: Tolerance = Tolerance {
    relative: 1e-6,
    absolute: 1e-18,
};

/// The original diode sweep was captured at C's default nonlinear RELTOL
/// (1e-3), including bypass. Independent junction/KCL checks establish that the
/// more accurate Rust root, not its equations, causes the 0.062% last-point
/// current difference. This bound applies only to that legacy golden; newly
/// demonstrated DC/AC decks retain NONLINEAR's 1 ppm requirement.
pub(super) const LEGACY_DIODE_DC: Tolerance = Tolerance {
    relative: 1e-3,
    absolute: 1e-12,
};

/// Transient bounds, by signal kind. Transient waveforms are compared at
/// shared physical times (see `tran.rs`), never step by step, so the bound must
/// cover two different integrators plus linear resampling of the denser side:
/// `|Rust - C| <= relative*|C| + absolute(kind)`.
///
/// * `relative = 1e-3` is ngspice's default `reltol`, the accuracy its own
///   local-truncation control targets per step; two independent solutions of
///   the same circuit can legitimately differ by about that much.
/// * `voltage_absolute = 1e-6` V is ngspice's default `vntol` (1 uV), the
///   natural voltage floor near zero crossings.
/// * `current_absolute = 1e-12` A is ngspice's default `abstol` (1 pA), the
///   natural current floor for the small branch currents of the linear decks.
///
/// These are the simulator's own default accuracy floors, not values fitted to
/// any fixture. Tighten only with evidence; never loosen to make a case pass.
///
/// `peak_relative` adds `peak_relative * max|C signal|` to the bound (0 for
/// [`TRAN`]).
#[derive(Debug, Clone, Copy)]
pub(super) struct TranTolerance {
    pub(super) peak_relative: f64,
    pub(super) relative: f64,
    pub(super) voltage_absolute: f64,
    pub(super) current_absolute: f64,
}

pub(super) const TRAN: TranTolerance = TranTolerance {
    peak_relative: 0.0,
    relative: 1e-3,
    voltage_absolute: 1e-6,
    current_absolute: 1e-12,
};

/// Pole-zero bound: each Rust root `r` matched to a C root `c` of the same
/// kind (pole or zero) must satisfy
/// `|r - c| <= relative * |c| + scale_relative * max|C root of the plot|`.
///
/// * `relative = 1e-6` is the relative tolerance C's own root search accepts
///   when it decides that a trial coincides with a found root
///   (`CKTpzRunTrial`, `cktpzstr.c`: `reltol = 1e-6` for `ISAROOT`), so C's
///   roots are not claimed to be more accurate than that. It also covers the
///   `~sqrt(eps)` split of an exact double root that any eigenvalue method
///   (and C's deflated Muller search) reports.
/// * `scale_relative = 1e-9` of the plot's largest C root (poles and zeros
///   together: the circuit's frequency scale) is the floor for roots at or
///   near the origin: C reports such a root as exactly zero (its determinant
///   LU is singular there), the port's eigenvalue is accurate to rounding of
///   the pencil's norm, which the largest root measures.
///
/// Roots are matched as unordered multisets (the order of equal real parts
/// and of perturbed multiple roots is not meaningful); never loosen these to
/// absorb a missing or extra root, which is a count mismatch.
#[derive(Debug, Clone, Copy)]
pub(super) struct RootTolerance {
    relative: f64,
    scale_relative: f64,
}

pub(super) const POLE_ZERO: RootTolerance = RootTolerance {
    relative: 1e-6,
    scale_relative: 1e-9,
};

/// The roots of one kind (`pole` or `zero`) of a pole-zero plot, in column
/// order.
fn root_set(
    plot: &Plot,
    row: &[ngspice_rs::primitives::Complex],
    kind: &str,
) -> Vec<(String, ngspice_rs::primitives::Complex)> {
    let prefix = format!("v({kind}(");
    plot.variables()
        .iter()
        .zip(row)
        .filter(|(variable, _)| variable.name.to_ascii_lowercase().starts_with(&prefix))
        .map(|(variable, value)| (variable.name.clone(), *value))
        .collect()
}

/// Find a perfect matching in reference-column order using breadth-first
/// augmenting paths. Only admissible edges enter the graph. The reported
/// maximum ratio is for the resulting assignment, not an optimized bottleneck.
/// Search is iterative (no root-count-dependent recursion), O(n³) time in the
/// dense worst case and O(n) additional space beyond the edge lists.
fn perfect_root_matching(edges: &[Vec<(usize, f64)>]) -> Result<f64, usize> {
    let n = edges.len();
    let mut reference_match: Vec<Option<(usize, f64)>> = vec![None; n];
    let mut candidate_match: Vec<Option<usize>> = vec![None; n];
    for start in 0..n {
        let mut parent = vec![None; n];
        let mut queue = VecDeque::from([start]);
        let mut free = None;
        'search: while let Some(reference) = queue.pop_front() {
            for &(candidate, ratio) in &edges[reference] {
                if parent[candidate].is_some() {
                    continue;
                }
                parent[candidate] = Some((reference, ratio));
                if let Some(next) = candidate_match[candidate] {
                    queue.push_back(next);
                } else {
                    free = Some(candidate);
                    break 'search;
                }
            }
        }
        let mut candidate = free.ok_or(start)?;
        while let Some((reference, ratio)) = parent[candidate] {
            candidate_match[candidate] = Some(reference);
            if let Some((previous, _)) = reference_match[reference].replace((candidate, ratio)) {
                candidate = previous;
            } else {
                break;
            }
        }
    }
    Ok(reference_match
        .into_iter()
        .flatten()
        .map(|(_, ratio)| ratio)
        .fold(0.0_f64, f64::max))
}

/// Compares a Rust pole-zero plot with a C one as two unordered root sets
/// (poles, zeros) under [`RootTolerance`]. The plot structure (name, flags,
/// variable names, units and real flags) must match exactly, so the root
/// counts must too. Returns a one-line summary.
#[expect(
    clippy::suboptimal_flops,
    reason = "retain the existing separately rounded pole-zero acceptance bounds; changing rounding can change golden classifications at the boundary"
)]
pub(super) fn roots(got: &Plot, want: &Plot, tolerance: RootTolerance) -> Result<String, String> {
    let got_columns = validate(got, "Rust")?;
    let want_columns = validate(want, "C")?;
    check_structure(got, want, &got_columns, &want_columns)?;
    if got.point_count() != 1 || want.point_count() != 1 {
        return Err(format!(
            "a pole-zero plot has one point: Rust {}, C {}",
            got.point_count(),
            want.point_count()
        ));
    }
    let mut worst_ratio: f64 = 0.0;
    let mut counts = Vec::new();
    let got_row = got
        .row(0)
        .ok_or_else(|| "Rust pole-zero plot has no row".to_owned())?;
    let want_row = want
        .row(0)
        .ok_or_else(|| "C pole-zero plot has no row".to_owned())?;
    let scale = want_row.iter().fold(0.0_f64, |m, z| m.max(z.magnitude()));
    for kind in ["pole", "zero"] {
        let (rust, c) = (root_set(got, got_row, kind), root_set(want, want_row, kind));
        if rust.len() != c.len() {
            return Err(format!(
                "{kind} count mismatch: Rust {}, C {}",
                rust.len(),
                c.len()
            ));
        }
        let bound = |c: ngspice_rs::primitives::Complex| {
            tolerance.relative * c.magnitude() + tolerance.scale_relative * scale
        };
        let mut edges = Vec::with_capacity(c.len());
        for (_, w) in &c {
            let limit = bound(*w);
            if !limit.is_finite() {
                return Err(format!("{kind}: nonfinite reference bound"));
            }
            let mut admissible = Vec::new();
            for (i, (_, r)) in rust.iter().enumerate() {
                let error = (*r - *w).magnitude();
                if limit == 0.0 {
                    if r == w {
                        admissible.push((i, 0.0));
                    }
                } else if error <= limit {
                    admissible.push((i, error / limit));
                }
            }
            edges.push(admissible);
        }
        let ratio = perfect_root_matching(&edges).map_err(|j| {
            format!(
                "{kind} mismatch: no perfect matching when including C {} = {:.17e}{:+.17e}j ({} admissible Rust candidates)",
                c[j].0, c[j].1.re, c[j].1.im, edges[j].len()
            )
        })?;
        worst_ratio = worst_ratio.max(ratio);
        counts.push(format!("{} {kind}(s)", c.len()));
    }
    Ok(format!(
        "{}, worst error {worst_ratio:.3e} of bound",
        counts.join(" + ")
    ))
}

pub(super) fn validate(plot: &Plot, label: &str) -> Result<BTreeMap<String, usize>, String> {
    if plot.is_empty() {
        return Err(format!("{label}: empty plot"));
    }
    let mut columns = BTreeMap::new();
    let mut seen: BTreeMap<String, usize> = BTreeMap::new();
    for (index, variable) in plot.variables().iter().enumerate() {
        let name = variable.name.to_ascii_lowercase();
        if name.is_empty() {
            return Err(format!("{label}: empty variable name"));
        }
        // C's rawfile can carry the same name twice: a Y (TXL) device has two
        // branch currents and `raw_write` prints both as `i(<device>)`. Such
        // repeats are identified by their occurrence, so both sides of a
        // comparison keep the same key for the same position.
        let count = seen.entry(name.clone()).or_default();
        *count += 1;
        let key = if *count == 1 {
            name
        } else {
            format!("{name}#{count}")
        };
        if columns.insert(key.clone(), index).is_some() {
            return Err(format!("{label}: ambiguous variable occurrence '{key}'"));
        }
    }
    for (index, point) in plot.rows().enumerate() {
        if point.len() != plot.variable_count() {
            return Err(format!("{label}: point {index} has invalid row shape"));
        }
        for (variable, value) in plot.variables().iter().zip(point) {
            if !value.is_finite() {
                return Err(format!(
                    "{label}: point {index}, {} has nonfinite value",
                    variable.name
                ));
            }
            if (plot.flags() == PlotFlags::Real || variable.is_real) && value.im != 0.0 {
                return Err(format!(
                    "{label}: point {index}, {} has imaginary data flagged real",
                    variable.name
                ));
            }
        }
    }
    Ok(columns)
}

/// Plot name/flags, variable-name set, units and real-vector flags must match
/// (shared by the point-wise and transient comparators).
pub(super) fn check_structure(
    got: &Plot,
    want: &Plot,
    got_columns: &BTreeMap<String, usize>,
    want_columns: &BTreeMap<String, usize>,
) -> Result<(), String> {
    if got.plotname != want.plotname || got.flags() != want.flags() {
        return Err(format!(
            "plot metadata mismatch: Rust '{}'/{}; C '{}'/{}",
            got.plotname,
            got.flags().as_rawfile(),
            want.plotname,
            want.flags().as_rawfile()
        ));
    }
    let missing: Vec<_> = want_columns
        .keys()
        .filter(|name| !got_columns.contains_key(*name))
        .collect();
    let extra: Vec<_> = got_columns
        .keys()
        .filter(|name| !want_columns.contains_key(*name))
        .collect();
    if !missing.is_empty() || !extra.is_empty() {
        return Err(format!(
            "variable mismatch: missing {missing:?}; extra {extra:?}"
        ));
    }
    for (name, &want_index) in want_columns {
        let actual = &got.variables()[got_columns[name]];
        let expected = &want.variables()[want_index];
        if actual.unit != expected.unit || actual.is_real != expected.is_real {
            return Err(format!(
                "variable metadata mismatch for '{name}': Rust {actual:?}; C {expected:?}"
            ));
        }
    }
    Ok(())
}

fn validate_alternate_captures(
    want: &Plot,
    want_columns: &BTreeMap<String, usize>,
    alternates: &[&Plot],
    axis: Option<&str>,
) -> Result<Vec<BTreeMap<String, usize>>, String> {
    alternates
        .iter()
        .map(|alternate| {
            let columns = validate(alternate, "alternate C")?;
            check_structure(want, alternate, want_columns, &columns)?;
            if alternate.point_count() != want.point_count() {
                return Err("alternate C point count mismatch".to_owned());
            }
            if let Some(axis) = axis {
                let primary = *want_columns.get(axis).ok_or("missing C axis")?;
                let other = *columns.get(axis).ok_or("missing alternate C axis")?;
                if !want
                    .rows()
                    .zip(alternate.rows())
                    .all(|(a, b)| a[primary] == b[other])
                {
                    return Err("independent C captures have different physical grids".to_owned());
                }
            }
            Ok(columns)
        })
        .collect()
}

/// Compare one production plot with committed independent C captures. Header dates,
/// commands, titles and internal plot IDs are not numerical metadata.
/// Axis identity is explicit in the fixture registry, not inferred from order.
/// Complex components each satisfy the ordinary bound against the nearest
/// point in the observed per-component C reference interval.
#[expect(
    clippy::suboptimal_flops,
    reason = "retain the existing separately rounded component acceptance bounds; changing rounding can change golden classifications at the boundary"
)]
pub(super) fn plots(
    got: &Plot,
    want: &Plot,
    alternates: &[&Plot],
    tolerance: Tolerance,
    axis: Option<&str>,
    resolution: Option<&Plot>,
) -> Result<(), String> {
    let got_columns = validate(got, "Rust")?;
    let want_columns = validate(want, "C")?;
    check_structure(got, want, &got_columns, &want_columns)?;
    if got.point_count() != want.point_count() {
        return Err(format!(
            "point count mismatch: Rust {}; C {}",
            got.point_count(),
            want.point_count()
        ));
    }
    let alternate_columns = validate_alternate_captures(want, &want_columns, alternates, axis)?;
    if let Some(resolution) = resolution {
        if resolution.point_count() != got.point_count()
            || resolution.variables() != got.variables()
            || resolution
                .cells()
                .iter()
                .any(|value| !value.is_finite() || value.re < 0. || value.im != 0.)
        {
            return Err("invalid sensitivity resolution diagnostic".to_owned());
        }
    }
    if let Some(axis) = axis {
        for (plot, columns, label) in [(got, &got_columns, "Rust"), (want, &want_columns, "C")] {
            let &column = columns
                .get(axis)
                .ok_or_else(|| format!("{label}: missing axis '{axis}'"))?;
            let mut previous = None;
            for (index, row) in plot.rows().enumerate() {
                let value = row[column];
                if value.im != 0.0 || value.re <= 0.0 || previous.is_some_and(|p| p >= value.re) {
                    return Err(format!(
                        "{label}: invalid positive increasing axis '{axis}' at point {index}"
                    ));
                }
                previous = Some(value.re);
            }
        }
    }
    let mut first = None;
    let mut worst = None;
    let mut worst_ratio = 0.0;
    let mut mismatches = 0usize;
    // C column order supplies stable, human-readable first-mismatch diagnostics.
    // The key identifies a repeated name by occurrence (`validate`), so a
    // device with two branch currents is compared position by position.
    let mut order: Vec<(&String, &usize)> = want_columns.iter().collect();
    order.sort_by_key(|(_, index)| **index);
    for (point, (actual, expected)) in got.rows().zip(want.rows()).enumerate() {
        let alternate_rows = alternates
            .iter()
            .map(|plot| plot.row(point).ok_or("missing alternate C point"))
            .collect::<Result<Vec<_>, _>>()?;
        for (key, column) in &order {
            let variable = &want.variables()[**column];
            let got = actual[got_columns[*key]];
            let want = expected[**column];
            for (component, got, reference) in [("re", got.re, want.re), ("im", got.im, want.im)] {
                let want = if alternates.is_empty() {
                    reference
                } else {
                    let observations =
                        alternate_rows
                            .iter()
                            .zip(&alternate_columns)
                            .map(|(row, columns)| {
                                let sample = row[columns[*key]];
                                if component == "re" {
                                    sample.re
                                } else {
                                    sample.im
                                }
                            });
                    ngspice_conformance::fixtures::nearest_reference(
                        got,
                        std::iter::once(reference).chain(observations),
                    )?
                };
                let error = (got - want).abs();
                let resolution_floor = resolution
                    .and_then(|plot| plot.row(point))
                    .map_or(0., |row| row[got_columns[*key]].re);
                let limit =
                    tolerance.relative * want.abs() + tolerance.absolute.max(resolution_floor);
                if error > limit {
                    mismatches += 1;
                    let report = format!(
                        "point {point}, {}.{component}: Rust {got:.17e}, C {want:.17e}, |error| {error:.6e} > bound {limit:.6e}",
                        variable.name
                    );
                    if first.is_none() {
                        first = Some(report.clone());
                    }
                    let ratio = error / limit;
                    if ratio > worst_ratio {
                        worst_ratio = ratio;
                        worst = Some(report);
                    }
                }
            }
        }
    }
    first.map_or(Ok(()), |first| {
        Err(format!(
            "{mismatches} component mismatch(es)\nfirst: {first}\nworst ({worst_ratio:.6e}x bound): {}",
            worst.as_deref().unwrap_or(&first)
        ))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ngspice_rs::analysis::Variable;
    use ngspice_rs::primitives::Complex;

    fn plot() -> Plot {
        Plot::new(
            "ac1",
            "AC Analysis",
            PlotFlags::Complex,
            vec![
                Variable::complex("frequency", "frequency"),
                Variable::complex("v(out)", "voltage"),
            ],
        )
        .with_rows(vec![
            vec![Complex::real(100.0), Complex::new(0.5, -0.25)],
            vec![Complex::real(1000.0), Complex::ZERO],
        ])
        .unwrap()
    }

    fn edit(
        plot: &Plot,
        change: impl FnOnce(&mut PlotFlags, &mut Vec<Variable>, &mut Vec<Vec<Complex>>),
    ) -> Plot {
        let mut flags = plot.flags();
        let mut variables = plot.variables().to_vec();
        let mut points = plot.rows().map(<[_]>::to_vec).collect::<Vec<_>>();
        change(&mut flags, &mut variables, &mut points);
        Plot::new(plot.name.clone(), plot.plotname.clone(), flags, variables)
            .with_rows(points)
            .unwrap()
    }

    fn compare(got: &Plot, want: &Plot) -> Result<(), String> {
        plots(got, want, &[], AC, Some("frequency"), None)
    }

    #[test]
    fn order_and_case_do_not_matter_but_metadata_does() {
        let want = plot();
        let mut got = edit(&want, |_, variables, points| {
            variables.swap(0, 1);
            variables[0].name = "V(OUT)".into();
            for row in points {
                row.swap(0, 1);
            }
        });
        got.name = "unrelated internal ID".into();
        compare(&got, &want).unwrap();
        got = edit(&got, |_, v, _| v[0].unit = "current".into());
        assert!(compare(&got, &want).unwrap_err().contains("metadata"));
        got = edit(&want, |_, v, _| v[1].is_real = true);
        assert!(compare(&got, &want).is_err());
        got = want.clone();
        got.plotname = "Operating Point".into();
        assert!(compare(&got, &want).is_err());
        got = edit(&want, |flags, _, _| *flags = PlotFlags::Real);
        assert!(compare(&got, &want).is_err());
    }

    #[test]
    fn missing_extra_renamed_and_duplicate_columns_fail() {
        let want = plot();
        for name in ["renamed", "FREQUENCY", ""] {
            let got = edit(&want, |_, v, _| v[1].name = name.into());
            assert!(compare(&got, &want).is_err());
        }
        let got = edit(&want, |_, v, rows| {
            v.pop();
            for row in rows {
                row.pop();
            }
        });
        assert!(compare(&got, &want).unwrap_err().contains("missing"));
        assert!(compare(&want, &got).unwrap_err().contains("extra"));
    }

    #[test]
    fn repeated_branch_names_compare_every_occurrence() {
        let want = edit(&plot(), |_, v, rows| {
            v.extend([
                Variable::complex("i(y1)", "current"),
                Variable::complex("i(y1)", "current"),
            ]);
            for row in rows {
                row.extend([Complex::real(0.02), Complex::real(-0.02)]);
            }
        });
        let mut got = edit(&want, |_, v, rows| {
            v.swap(0, 2);
            for row in rows {
                row.swap(0, 2);
            }
        });
        compare(&got, &want).unwrap();
        got.point_mut(0).unwrap()[3] = Complex::real(0.03);
        assert!(compare(&got, &want).unwrap_err().contains("i(y1)"));
        assert!(compare(&want, &got).is_err());
        got = edit(&want, |_, v, rows| {
            v.pop();
            for row in rows {
                row.pop();
            }
        });
        assert!(compare(&got, &want).unwrap_err().contains("missing"));
    }

    #[test]
    fn occurrence_keys_must_not_hide_a_literal_name() {
        let p = edit(&plot(), |_, v, rows| {
            v[0].name = "i(y1)".into();
            v[1].name = "i(y1)".into();
            v.push(Variable::complex("i(y1)#2", "current"));
            for row in rows {
                row.push(Complex::ZERO);
            }
        });
        assert!(validate(&p, "C").unwrap_err().contains("ambiguous"));
    }

    #[test]
    fn shape_empty_and_nonfinite_data_fail_on_either_side() {
        let want = plot();
        let mut cases = Vec::new();
        assert!(want.header().with_rows(vec![vec![Complex::ZERO]]).is_err());
        let mut bad = want.clone();
        bad.truncate_points(1);
        cases.push(bad);
        let mut bad = want.clone();
        bad.clear_points();
        cases.push(bad);
        cases.push(Plot::new(
            "ac1",
            "AC Analysis",
            PlotFlags::Complex,
            Vec::new(),
        ));
        for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            for imaginary in [false, true] {
                let mut bad = want.clone();
                if imaginary {
                    bad.point_mut(0).unwrap()[1].im = value;
                } else {
                    bad.point_mut(0).unwrap()[1].re = value;
                }
                cases.push(bad);
            }
        }
        for bad in cases {
            assert!(compare(&bad, &want).is_err());
            assert!(compare(&want, &bad).is_err());
        }
    }

    #[test]
    fn components_and_zero_use_relative_plus_absolute_bounds() {
        for tolerance in [DC, AC] {
            let want = plot();
            for (point, component) in [(0, "re"), (0, "im"), (1, "re"), (1, "im")] {
                let mut got = want.clone();
                let value = if component == "re" {
                    &mut got.point_mut(point).unwrap()[1].re
                } else {
                    &mut got.point_mut(point).unwrap()[1].im
                };
                let bound = tolerance.relative * value.abs() + tolerance.absolute;
                *value += 0.5 * bound;
                plots(&got, &want, &[], tolerance, None, None).unwrap();
                *if component == "re" {
                    &mut got.point_mut(point).unwrap()[1].re
                } else {
                    &mut got.point_mut(point).unwrap()[1].im
                } += 2.0 * bound;
                let error = plots(&got, &want, &[], tolerance, None, None).unwrap_err();
                assert!(error.contains(&format!("v(out).{component}")), "{error}");
                assert!(error.contains("first:") && error.contains("worst"));
            }
        }
    }

    #[test]
    fn first_and_worst_mismatches_are_distinct() {
        let want = plot();
        let mut got = want.clone();
        got.point_mut(0).unwrap()[1].re += 0.01;
        got.point_mut(1).unwrap()[1].im = 1.0;
        let report = compare(&got, &want).unwrap_err();
        assert!(report.contains("first: point 0, v(out).re"), "{report}");
        assert!(report.contains("x bound): point 1, v(out).im"), "{report}");
    }

    fn pole_zero(poles: &[Complex], zeros: &[Complex]) -> Plot {
        let mut plot = Plot::builder("pz1", "Pole-Zero Analysis", PlotFlags::Complex);
        for (index, _) in poles.iter().enumerate() {
            plot.push_variable(Variable::complex(
                format!("v(pole({}))", index + 1),
                "voltage",
            ));
        }
        for (index, _) in zeros.iter().enumerate() {
            plot.push_variable(Variable::complex(
                format!("v(zero({}))", index + 1),
                "voltage",
            ));
        }
        plot.build()
            .with_rows(vec![poles.iter().chain(zeros).copied().collect::<Vec<_>>()])
            .unwrap()
    }

    #[test]
    fn root_sets_match_unordered_within_relative_and_scale_bounds() {
        let p = [
            Complex::new(-1e3, 2e3),
            Complex::new(-1e3, -2e3),
            Complex::real(-5e6),
        ];
        let want = pole_zero(&p, &[Complex::ZERO, Complex::real(-1e4)]);
        // Reordered, and within the bound: 1e-7 relative, and a zero at
        // 1e-6 (below 1e-9 * 5e6 = 5e-3) for C's exact zero.
        let got = pole_zero(
            &[p[2] * Complex::real(1. + 1e-7), p[1], p[0]],
            &[Complex::real(-1e4), Complex::real(1e-6)],
        );
        assert!(
            roots(&got, &want, POLE_ZERO)
                .unwrap()
                .contains("3 pole(s) + 2 zero(s)")
        );
        let far = pole_zero(&p, &[Complex::real(1e-2), Complex::real(-1e4)]);
        assert!(
            roots(&far, &want, POLE_ZERO)
                .unwrap_err()
                .contains("zero mismatch")
        );
        let moved = pole_zero(
            &[p[0], p[1], Complex::real(-5e6 * (1. + 1e-5))],
            &[Complex::ZERO, Complex::real(-1e4)],
        );
        assert!(
            roots(&moved, &want, POLE_ZERO)
                .unwrap_err()
                .contains("pole mismatch")
        );
        // A missing root is a structure (count) failure, never a tolerance one.
        let fewer = pole_zero(&p[..2], &[Complex::ZERO, Complex::real(-1e4)]);
        assert!(roots(&fewer, &want, POLE_ZERO).is_err());
        let mut renamed = want.clone();
        renamed.plotname = "AC Analysis".into();
        assert!(roots(&renamed, &want, POLE_ZERO).is_err());
    }

    #[test]
    fn root_matching_reassigns_greedy_counterexample() {
        let reference = [Complex::real(1_000_000.0), Complex::real(1_000_001.5)];
        let candidates = [Complex::real(1_000_000.6), Complex::real(999_999.2)];
        for expected in [reference, [reference[1], reference[0]]] {
            for actual in [candidates, [candidates[1], candidates[0]]] {
                let want = pole_zero(&expected, &expected);
                let got = pole_zero(&actual, &actual);
                let report = roots(&got, &want, POLE_ZERO).unwrap();
                assert!(report.contains("2 pole(s) + 2 zero(s)"), "{report}");
            }
        }
        let impossible = pole_zero(&[candidates[0]; 2], &[]);
        // Both C roots can take the same candidate value here; use a
        // reference for which only one of the two candidates is admissible.
        let want = pole_zero(&reference, &[]);
        assert!(roots(&impossible, &want, POLE_ZERO).is_ok());
        let deficient = pole_zero(&[candidates[0], Complex::real(999_990.0)], &[]);
        let error = roots(&deficient, &want, POLE_ZERO).unwrap_err();
        assert!(error.contains("no perfect matching"), "{error}");
    }

    #[test]
    fn root_matching_zero_bounds_and_repeated_roots() {
        let want = pole_zero(&[Complex::ZERO; 3], &[Complex::ZERO; 2]);
        let got = pole_zero(&[Complex::real(-0.0); 3], &[Complex::ZERO; 2]);
        let report = roots(&got, &want, POLE_ZERO).unwrap();
        assert!(report.contains("worst error 0.000e0"), "{report}");
        let mut bad = got.clone();
        bad.point_mut(0).unwrap()[0] = Complex::real(f64::from_bits(1));
        assert!(
            roots(&bad, &want, POLE_ZERO)
                .unwrap_err()
                .contains("pole mismatch")
        );
        bad.point_mut(0).unwrap()[0] = Complex::real(f64::NAN);
        assert!(
            roots(&bad, &want, POLE_ZERO)
                .unwrap_err()
                .contains("nonfinite")
        );
        assert!(bad.header().with_rows(vec![vec![]]).is_err());

        let repeated = Complex::new(-1e6, 1e6);
        let nearby = Complex::new(-1_000_000.25, 999_999.75);
        let want = pole_zero(&[repeated, nearby, repeated], &[]);
        let got = pole_zero(&[nearby, repeated, nearby], &[]);
        assert!(roots(&got, &want, POLE_ZERO).is_ok());
    }

    fn exhaustive_matching(edges: &[Vec<(usize, f64)>], used: u32) -> bool {
        let Some((first, rest)) = edges.split_first() else {
            return true;
        };
        first.iter().any(|&(candidate, _)| {
            let bit = 1 << candidate;
            used & bit == 0 && exhaustive_matching(rest, used | bit)
        })
    }

    #[test]
    fn root_matching_agrees_with_exhaustive_assignments() {
        let graph = |size: usize, mask: u32| {
            (0..size)
                .map(|i| {
                    (0..size)
                        .filter(|&j| mask & (1 << (i * size + j)) != 0)
                        .map(|j| (j, 0.5))
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>()
        };
        for size in 0..=3 {
            for mask in 0..(1 << (size * size)) {
                let edges = graph(size, mask);
                assert_eq!(
                    perfect_root_matching(&edges).is_ok(),
                    exhaustive_matching(&edges, 0)
                );
            }
        }
        let mut seed = 0x1234_5678_u32;
        for _ in 0..512 {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let edges = graph(5, seed >> 7);
            assert_eq!(
                perfect_root_matching(&edges).is_ok(),
                exhaustive_matching(&edges, 0)
            );
        }
        // The low-ratio edge for C[0] must be displaced. The summary
        // must describe the final matching, not the initial greedy choice.
        let edges = [vec![(0, 0.1), (1, 0.8)], vec![(0, 0.9)]];
        assert!((perfect_root_matching(&edges).unwrap() - 0.9).abs() < f64::EPSILON);
    }

    #[test]
    fn axis_identity_values_and_monotonicity_are_checked() {
        let want = plot();
        assert!(plots(&want, &want, &[], AC, Some("time"), None).is_err());
        for value in [Complex::ZERO, Complex::real(-1.0), Complex::new(100.0, 1.0)] {
            let mut got = want.clone();
            got.point_mut(0).unwrap()[0] = value;
            assert!(compare(&got, &want).unwrap_err().contains("axis"));
        }
        let mut got = want.clone();
        got.point_mut(1).unwrap()[0] = got.row(0).unwrap()[0];
        assert!(compare(&got, &want).unwrap_err().contains("axis"));
        got = want.clone();
        got.point_mut(1).unwrap()[0].re += 1.0;
        assert!(compare(&got, &want).unwrap_err().contains("frequency.re"));
    }
}
