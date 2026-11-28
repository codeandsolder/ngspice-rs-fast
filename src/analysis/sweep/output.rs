//! Deterministic output projection for sweeps that can change circuit shape.
use super::{Complex, SpiceResult, unsupported};
use crate::analysis::results::{Plot, Variable};
use std::collections::BTreeMap;

/// Include every topology's columns in deterministic first-appearance order.
/// An internal unknown which does not exist at a point has no numeric value.
#[derive(Default)]
pub(super) struct Columns {
    pub(super) variables: Vec<Variable>,
    indices: BTreeMap<(String, String, bool, usize), usize>,
}

impl Columns {
    pub(super) fn map(&mut self, plot: &Plot) -> Vec<usize> {
        let mut occurrences = BTreeMap::new();
        plot.variables()
            .iter()
            .map(|variable| {
                let identity = (
                    variable.name.clone(),
                    variable.unit.clone(),
                    variable.is_real,
                );
                let count = occurrences.entry(identity.clone()).or_insert(0usize);
                let key = (identity.0, identity.1, identity.2, *count);
                *count += 1;
                *self.indices.entry(key).or_insert_with(|| {
                    let index = self.variables.len();
                    self.variables.push(variable.clone());
                    index
                })
            })
            .collect()
    }
}

pub(super) fn projection(
    sink: &dyn crate::analysis::PointSink,
    result: &Plot,
) -> SpiceResult<Option<Vec<usize>>> {
    let selected = sink.selected_columns().map(<[usize]>::to_vec);
    if selected.as_ref().is_some_and(|indices| {
        indices
            .iter()
            .any(|&index| index >= result.variable_count())
            || indices.windows(2).any(|pair| pair[0] >= pair[1])
    }) {
        return Err(unsupported("invalid geometry sweep projection"));
    }
    Ok(selected)
}

pub(super) fn emit(
    sink: &mut dyn crate::analysis::PointSink,
    result: &Plot,
    row: &[Complex],
    selected: Option<&[usize]>,
    projected: &mut Vec<Complex>,
) -> SpiceResult<()> {
    if let Some(selected) = selected {
        projected.clear();
        for &index in selected {
            projected.push(
                *row.get(index)
                    .ok_or_else(|| unsupported("invalid geometry sweep projection"))?,
            );
        }
        sink.point(projected)
    } else {
        result.check_point(row)?;
        sink.point(row)
    }
}
