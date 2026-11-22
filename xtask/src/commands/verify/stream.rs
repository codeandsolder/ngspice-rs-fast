//! Bounded sequential decoding of the selected C reference.
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

use ngspice_rs::analysis::{
    AnalysisPlan, Plot, RawFileReader, RawFormat, RawPlotRows, RawReadPolicy,
};
use ngspice_rs::devices::Circuit;

use super::{Gate, compare_plot, fixtures};

pub(super) struct Golden {
    path: PathBuf,
    reader: RawFileReader<BufReader<File>>,
}

impl Golden {
    pub(super) fn open(root: &Path, name: &str) -> Result<Self, String> {
        let path = fixtures::golden_path(root, name);
        let file =
            File::open(&path).map_err(|error| format!("reading {}: {error}", path.display()))?;
        Ok(Self {
            path,
            reader: RawFileReader::new(BufReader::new(file))
                .with_format(RawFormat::Ascii)
                .with_policy(RawReadPolicy::DEFAULT),
        })
    }

    pub(super) fn compare(
        &mut self,
        gate: &Gate,
        circuit: &Circuit,
        request: &AnalysisPlan,
        got: &Plot,
    ) -> Result<String, String> {
        let rows = self
            .reader
            .next_plot()
            .map_err(|error| format!("{}: {error}", self.path.display()))?
            .ok_or_else(|| format!("{}: missing C plot", self.path.display()))?;
        compare_plot(gate, circuit, request, got, rows)
            .map_err(|error| format!("{}: {error}", self.path.display()))
    }

    pub(super) fn finish(mut self) -> Result<(), String> {
        match self
            .reader
            .next_plot()
            .map_err(|error| format!("{}: {error}", self.path.display()))?
        {
            None => Ok(()),
            Some(_) => Err(format!("{}: unexpected extra C plot", self.path.display())),
        }
    }
}

// Roots and event-aware transient interpolation require random row access.
// Only the current expected plot is retained. The reader separately bounds
// header size and columns; charge the declared rows/cells before allocating.
pub(super) fn collect_plot(mut rows: RawPlotRows<'_, impl BufRead>) -> Result<Plot, String> {
    let policy = RawReadPolicy::DEFAULT.collect;
    let points = rows.point_count();
    let width = rows.header().plot.variable_count();
    let cells = points
        .checked_mul(width)
        .ok_or("C plot cell count overflows")?;
    if points > policy.rows || cells > policy.cells || width > policy.variables {
        return Err("C plot exceeds collection policy".into());
    }
    let mut values = Vec::new();
    values
        .try_reserve_exact(cells)
        .map_err(|error| error.to_string())?;
    while let Some(row) = rows.next_row().map_err(|error| error.to_string())? {
        values.extend_from_slice(row);
    }
    rows.finish()
        .map_err(|error| error.to_string())?
        .plot
        .with_cells(points, values)
        .map_err(|error| error.to_string())
}
