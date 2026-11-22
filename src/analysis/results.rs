//! Result plots: the data an analysis produces.
//!
//! Mirrors ngspice's `struct plot` from `src/include/ngspice/plot.h` and the
//! `struct dvec` it holds. ngspice stores an array of data vectors plus scale
//! information for the interactive plotter; the port keeps a flat table of
//! variables and points, which is what a rawfile contains, and leaves plotting
//! to whoever consumes the result.

use std::{borrow::Cow, collections::HashMap, sync::Arc};

use crate::primitives::{Complex, Real, SpiceError, SpiceResult};

/// Ordered output lifecycle for every plot produced by an analysis.
///
/// A successful plot is announced, receives borrowed rows in order, then ends.
/// Errors abort the analysis; no successful end event follows a failed row.
/// Drivers that intrinsically collect a plot transfer it by value instead.
pub trait PointSink {
    /// Announce a point-free header before producing the first row.
    ///
    /// # Errors
    /// Whatever the consumer reports; the analysis stops with that error.
    fn begin(&mut self, header: &Plot) -> SpiceResult<()>;

    /// One borrowed row, with one value per variable of the active header.
    ///
    /// # Errors
    /// Whatever the consumer reports; the analysis stops with that error.
    fn point(&mut self, values: &[Complex]) -> SpiceResult<()>;

    /// Complete the active plot, including when it has no rows.
    ///
    /// # Errors
    /// Whatever the consumer reports; the analysis stops with that error.
    fn end(&mut self) -> SpiceResult<()>;

    /// Transfer an already materialized plot. Consumers that retain it can
    /// take its packed cell allocation directly rather than copying or replaying.
    ///
    /// # Errors
    /// As the begin/point/end lifecycle.
    fn complete(&mut self, plot: Plot) -> SpiceResult<()> {
        self.begin(&plot.header())?;
        for point in plot.rows() {
            self.point(point)?;
        }
        self.end()
    }
}

pub(super) fn lifecycle_error(message: &str) -> SpiceError {
    SpiceError::Numerical {
        context: "analysis output".into(),
        message: message.into(),
    }
}

/// Collect only when requested by the caller, never inside a streaming driver.
#[derive(Default)]
pub(super) struct PlotCollector {
    plots: Vec<Plot>,
    active: Option<Plot>,
}

impl PlotCollector {
    pub(super) fn finish(self, expected: usize) -> SpiceResult<Vec<Plot>> {
        if self.active.is_some() || self.plots.len() != expected {
            return Err(lifecycle_error(
                "driver did not complete the planned number of plots",
            ));
        }
        Ok(self.plots)
    }
}

impl PointSink for PlotCollector {
    fn begin(&mut self, header: &Plot) -> SpiceResult<()> {
        if self.active.is_some() {
            return Err(lifecycle_error(
                "began a plot before completing the previous one",
            ));
        }
        self.active = Some(header.header());
        Ok(())
    }

    fn point(&mut self, values: &[Complex]) -> SpiceResult<()> {
        self.active
            .as_mut()
            .ok_or_else(|| lifecycle_error("point without an active plot"))?
            .push_point(values)
    }

    fn end(&mut self) -> SpiceResult<()> {
        let plot = self
            .active
            .take()
            .ok_or_else(|| lifecycle_error("end without an active plot"))?;
        self.complete(plot)
    }

    fn complete(&mut self, plot: Plot) -> SpiceResult<()> {
        if self.active.is_some() {
            return Err(lifecycle_error("completed a plot while another is active"));
        }
        self.plots.push(plot);
        Ok(())
    }
}

/// Whether a plot's data is real or complex.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlotFlags {
    /// Real data, as produced by `.op`, `.dc` and `.tran`.
    Real,
    /// Complex data, as produced by `.ac`, `.noise` and `.disto`.
    Complex,
}

impl PlotFlags {
    /// Interprets a rawfile `Flags:` line.
    ///
    /// ngspice writes `real`, `complex`, and for pole-zero analysis
    /// `real poles`. Only the presence of the word `complex` matters.
    #[must_use]
    pub fn parse(raw: &str) -> Self {
        if raw.to_ascii_lowercase().contains("complex") {
            Self::Complex
        } else {
            Self::Real
        }
    }

    /// The spelling used in a rawfile `Flags:` line.
    #[must_use]
    pub const fn as_rawfile(self) -> &'static str {
        match self {
            Self::Real => "real",
            Self::Complex => "complex",
        }
    }

    /// True when the data is complex.
    #[must_use]
    pub const fn is_complex(self) -> bool {
        matches!(self, Self::Complex)
    }
}

/// One column of a plot.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Variable {
    /// The name ngspice reports, e.g. `v(out)`, `i(v1)`, `frequency`.
    pub name: String,
    /// The unit ngspice reports: `voltage`, `current`, `time`, `frequency`,
    /// `device`, …
    pub unit: String,
    /// True when ngspice flags this data vector as real, i.e. `isreal(v)` in
    /// `src/frontend/rawfile.c`.
    ///
    /// Inside a `complex` plot, a vector flagged real is written as `re,0.0`,
    /// while a complex vector whose imaginary parts happen to be zero is written
    /// as `re,0.000000000000000e+00`. `.ac` and `.noise` produce complex vectors
    /// throughout, so their columns are not flagged real even when the imaginary
    /// part is zero — `frequency` and `v(in)` in an AC sweep, for instance.
    pub is_real: bool,
}

impl Variable {
    /// Builds a variable, flagged real.
    #[must_use]
    pub fn new(name: impl Into<String>, unit: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            unit: unit.into(),
            is_real: true,
        }
    }

    /// Builds a variable that ngspice does not flag as real: its values are
    /// written as `re,im` even when the imaginary part is zero.
    #[must_use]
    pub fn complex(name: impl Into<String>, unit: impl Into<String>) -> Self {
        Self {
            is_real: false,
            ..Self::new(name, unit)
        }
    }
}

/// Immutable ordered column identities and lookup indexes for a plot.
/// Duplicate names (such as the branch currents of a two-port device) are
/// allowed; lookup always selects the first occurrence.
#[derive(Debug, PartialEq, Eq)]
pub struct PlotSchema {
    flags: PlotFlags,
    variables: Vec<Variable>,
    names: HashMap<String, usize>,
    voltages: HashMap<String, usize>,
    currents: HashMap<String, usize>,
}

fn folded(name: &str) -> Cow<'_, str> {
    if name.bytes().any(|b| b.is_ascii_uppercase()) {
        Cow::Owned(name.to_ascii_lowercase())
    } else {
        Cow::Borrowed(name)
    }
}

impl PlotSchema {
    fn new(flags: PlotFlags, variables: Vec<Variable>) -> Self {
        let mut names = HashMap::with_capacity(variables.len());
        let mut voltages = HashMap::new();
        let mut currents = HashMap::new();
        for (index, variable) in variables.iter().enumerate() {
            let name = folded(&variable.name);
            if let Some(node) = name.strip_prefix("v(").and_then(|s| s.strip_suffix(')')) {
                voltages.entry(node.to_owned()).or_insert(index);
            }
            if let Some(device) = name.strip_prefix("i(").and_then(|s| s.strip_suffix(')')) {
                currents.entry(device.to_owned()).or_insert(index);
            }
            names.entry(name.into_owned()).or_insert(index);
        }
        Self {
            flags,
            variables,
            names,
            voltages,
            currents,
        }
    }

    /// Exact ASCII case-insensitive vector identity.
    #[must_use]
    pub fn index(&self, name: &str) -> Option<usize> {
        self.names.get(folded(name).as_ref()).copied()
    }

    /// Explicit voltage identity, without constructing a wrapper string.
    #[must_use]
    pub fn voltage(&self, node: &str) -> Option<usize> {
        self.voltages.get(folded(node).as_ref()).copied()
    }

    /// Explicit branch current identity.
    #[must_use]
    pub fn current(&self, device: &str) -> Option<usize> {
        self.currents.get(folded(device).as_ref()).copied()
    }

    /// Named requests prefer exact vector names, then voltage, current and
    /// finally the C branch-name alias.
    #[must_use]
    pub fn named(&self, vector: &str) -> Option<usize> {
        let name = folded(vector);
        self.names
            .get(name.as_ref())
            .or_else(|| self.voltages.get(name.as_ref()))
            .or_else(|| self.currents.get(name.as_ref()))
            .or_else(|| {
                let device = name.strip_suffix("#branch")?;
                (!device.is_empty())
                    .then(|| self.currents.get(device))
                    .flatten()
            })
            .copied()
    }

    /// Ordered columns of this schema.
    #[must_use]
    pub fn variables(&self) -> &[Variable] {
        &self.variables
    }

    /// Real or complex plot representation.
    #[must_use]
    pub const fn flags(&self) -> PlotFlags {
        self.flags
    }
}

/// Builder for a plot schema before any rows exist.
pub struct PlotBuilder {
    name: String,
    plotname: String,
    flags: PlotFlags,
    variables: Vec<Variable>,
}

impl PlotBuilder {
    /// Append a variable before sealing the schema.
    pub fn push_variable(&mut self, variable: Variable) {
        self.variables.push(variable);
    }

    /// Seal the schema and start collecting rows.
    #[must_use]
    pub fn build(self) -> Plot {
        Plot::new(self.name, self.plotname, self.flags, self.variables)
    }
}

/// The result of one analysis run.
#[derive(Debug, Clone, PartialEq)]
pub struct Plot {
    /// Internal plot name (not part of the column schema).
    pub name: String,
    /// Rawfile plot name.
    pub plotname: String,
    schema: Arc<PlotSchema>,
    /// Row-major rectangular storage. Each row has the immutable schema width.
    cells: Vec<Complex>,
    point_count: usize,
}

impl Plot {
    /// Construct a plot with a complete column schema and no points.
    #[must_use]
    pub fn new(
        name: impl Into<String>,
        plotname: impl Into<String>,
        flags: PlotFlags,
        variables: Vec<Variable>,
    ) -> Self {
        Self {
            name: name.into(),
            plotname: plotname.into(),
            schema: Arc::new(PlotSchema::new(flags, variables)),
            cells: Vec::new(),
            point_count: 0,
        }
    }

    /// Begin constructing a schema when the columns are generated incrementally.
    #[must_use]
    pub fn builder(
        name: impl Into<String>,
        plotname: impl Into<String>,
        flags: PlotFlags,
    ) -> PlotBuilder {
        PlotBuilder {
            name: name.into(),
            plotname: plotname.into(),
            flags,
            variables: Vec::new(),
        }
    }

    /// The immutable schema, shared by clones and bound projections.
    #[must_use]
    pub const fn schema(&self) -> &Arc<PlotSchema> {
        &self.schema
    }

    /// Ordered columns.
    #[must_use]
    pub fn variables(&self) -> &[Variable] {
        self.schema.variables()
    }

    /// Real or complex plot representation.
    #[must_use]
    pub fn flags(&self) -> PlotFlags {
        self.schema.flags()
    }

    /// All cells in row-major order, without row descriptors or allocations.
    #[must_use]
    pub fn cells(&self) -> &[Complex] {
        &self.cells
    }

    /// Borrow a row by index.
    #[must_use]
    pub fn row(&self, index: usize) -> Option<&[Complex]> {
        if index >= self.point_count {
            return None;
        }
        let width = self.variable_count();
        let start = index * width;
        Some(&self.cells[start..start + width])
    }

    /// Borrow every row in order, including the empty pole-zero result row.
    pub fn rows(&self) -> impl ExactSizeIterator<Item = &[Complex]> + DoubleEndedIterator + Clone {
        let width = self.variable_count();
        (0..self.point_count).map(move |index| {
            let start = index * width;
            &self.cells[start..start + width]
        })
    }

    /// Borrow a row mutably without allowing changes to its width.
    pub fn point_mut(&mut self, index: usize) -> Option<&mut [Complex]> {
        if index >= self.point_count {
            return None;
        }
        let width = self.variable_count();
        let start = index * width;
        Some(&mut self.cells[start..start + width])
    }

    /// Mutable fixed-width rows. A valid rootless pole-zero row has no cells.
    pub fn rows_mut(&mut self) -> impl DoubleEndedIterator<Item = &mut [Complex]> {
        let width = self.variable_count();
        let empty = if width == 0 { self.point_count } else { 0 };
        std::iter::once(&mut [][..])
            .take(empty)
            .chain(self.cells.chunks_exact_mut(width.max(1)))
    }

    /// Reserve space for additional rows, without modifying the row count.
    ///
    /// # Errors
    /// Invalid or overflowing shape, or allocation failure.
    pub fn reserve_points(&mut self, additional: usize) -> SpiceResult<()> {
        let count = self
            .point_count
            .checked_add(additional)
            .ok_or_else(|| self.storage_error("point count overflows usize"))?;
        self.cell_count(count)?;
        let cells = additional * self.variable_count();
        self.cells
            .try_reserve_exact(cells)
            .map_err(|error| self.storage_error(error))
    }

    /// Remove rows after the given length.
    pub fn truncate_points(&mut self, len: usize) {
        self.point_count = self.point_count.min(len);
        self.cells
            .truncate(self.point_count * self.variable_count());
    }

    /// Remove all rows, preserving the schema and reusable cell capacity.
    pub fn clear_points(&mut self) {
        self.truncate_points(0);
    }

    /// Remove a row by index, preserving the order of all other rows.
    /// Returns false if the index is out of bounds.
    pub fn remove_point(&mut self, index: usize) -> bool {
        if index >= self.point_count {
            return false;
        }
        let width = self.variable_count();
        let start = index * width;
        drop(self.cells.drain(start..start + width));
        self.point_count -= 1;
        true
    }

    /// Consume the plot and transfer its packed cells without allocating.
    #[must_use]
    pub fn into_cells(self) -> Vec<Complex> {
        self.cells
    }

    /// A point-free header sharing the same schema.
    #[must_use]
    pub fn header(&self) -> Self {
        Self {
            name: self.name.clone(),
            plotname: self.plotname.clone(),
            schema: Arc::clone(&self.schema),
            cells: Vec::new(),
            point_count: 0,
        }
    }

    /// Attach a rectangular owned cell buffer without copying it.
    ///
    /// # Errors
    /// Invalid shape, overflow or inconsistent cell count.
    pub fn with_cells(mut self, point_count: usize, cells: Vec<Complex>) -> SpiceResult<Self> {
        let expected = self.cell_count(point_count)?;
        if cells.len() != expected {
            return Err(self.storage_error(format_args!(
                "expected {expected} cells, got {}",
                cells.len()
            )));
        }
        self.cells = cells;
        self.point_count = point_count;
        Ok(self)
    }

    /// Populate a new table from rows of any slice-like representation.
    ///
    /// # Errors
    /// Ragged rows, invalid shape, overflow or allocation failure.
    pub fn with_rows(
        mut self,
        rows: impl IntoIterator<Item = impl AsRef<[Complex]>>,
    ) -> SpiceResult<Self> {
        self.clear_points();
        for row in rows {
            self.push_point(row.as_ref())?;
        }
        Ok(self)
    }

    /// Number of points, including the single empty row of a rootless PZ plot.
    #[must_use]
    pub const fn point_count(&self) -> usize {
        self.point_count
    }

    fn storage_error(&self, message: impl std::fmt::Display) -> SpiceError {
        SpiceError::Numerical {
            context: format!("plot {}", self.name),
            message: message.to_string(),
        }
    }

    pub(super) fn cell_count(&self, points: usize) -> SpiceResult<usize> {
        let width = self.variable_count();
        if width == 0 && !valid_empty_shape(&self.plotname, self.flags(), points) {
            return Err(self.storage_error(
                "zero variables with nonzero points outside a rootless pole-zero result",
            ));
        }
        points
            .checked_mul(width)
            .ok_or_else(|| self.storage_error("point count times column count overflows usize"))
    }

    /// Validate rectangular storage before serialization, including the
    /// zero-column rule (plot names can be edited by callers).
    pub(super) fn validate_shape(&self) -> SpiceResult<()> {
        let expected = self.cell_count(self.point_count)?;
        if expected == self.cells.len() {
            Ok(())
        } else {
            Err(self.storage_error("inconsistent rectangular cell count"))
        }
    }

    /// Number of columns.
    #[must_use]
    pub fn variable_count(&self) -> usize {
        self.variables().len()
    }

    /// True when the plot has no columns or no points.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.variables().is_empty() || self.point_count == 0
    }

    /// Append a borrowed row, copying its cells once into the packed table.
    ///
    /// # Errors
    /// Ragged row, invalid shape, overflow or allocation failure.
    pub fn push_point(&mut self, values: &[Complex]) -> SpiceResult<()> {
        self.check_point(values)?;
        let next = self
            .point_count
            .checked_add(1)
            .ok_or_else(|| self.storage_error("point count overflows usize"))?;
        self.cell_count(next)?;
        self.cells
            .try_reserve(values.len())
            .map_err(|error| self.storage_error(error))?;
        self.cells.extend_from_slice(values);
        self.point_count = next;
        Ok(())
    }

    /// Check that one row matches the immutable schema.
    ///
    /// # Errors
    /// Row width mismatch or invalid zero-column row.
    pub fn check_point(&self, values: &[Complex]) -> SpiceResult<()> {
        if values.len() != self.variable_count() {
            return Err(self.storage_error(format_args!(
                "point has {} value(s) but the plot has {} variable(s)",
                values.len(),
                self.variable_count()
            )));
        }
        self.cell_count(1)?;
        Ok(())
    }

    /// The index of a variable, matched case-insensitively.
    #[must_use]
    pub fn variable_index(&self, name: &str) -> Option<usize> {
        self.schema.index(name)
    }

    /// One value, addressed by variable name and point index.
    #[must_use]
    pub fn value(&self, variable: &str, point: usize) -> Option<Complex> {
        let column = self.variable_index(variable)?;
        self.row(point)?.get(column).copied()
    }

    /// Borrow a strided view of a named column without allocating.
    #[must_use]
    pub fn column(&self, variable: &str) -> Option<ColumnView<'_>> {
        self.column_at(self.variable_index(variable)?)
    }

    /// Borrow a strided view by column index.
    #[must_use]
    pub fn column_at(&self, column: usize) -> Option<ColumnView<'_>> {
        let width = self.variable_count();
        (column < width).then_some(ColumnView {
            cells: &self.cells,
            width,
            column,
        })
    }

    /// Largest absolute value anywhere in the plot.
    #[must_use]
    pub fn max_abs(&self) -> Real {
        self.cells
            .iter()
            .fold(0.0, |acc, value| acc.max(value.magnitude()))
    }

    /// True when every value is finite.
    #[must_use]
    pub fn is_finite(&self) -> bool {
        self.cells.iter().all(|value| value.is_finite())
    }
}

/// C's rootless .pz result is a real plot containing one empty row. Other
/// zero-column plots must be empty; no unbounded zero-cell stream is valid.
pub(super) fn valid_empty_shape(plotname: &str, flags: PlotFlags, points: usize) -> bool {
    points == 0 || (points == 1 && flags == PlotFlags::Real && plotname == "Pole-Zero Analysis")
}

/// A borrowed, strided column of the packed row-major plot table.
///
/// The view owns no cells. Indexing is in point order, not memory order.
#[derive(Clone, Copy)]
pub struct ColumnView<'a> {
    cells: &'a [Complex],
    width: usize,
    column: usize,
}

impl<'a> ColumnView<'a> {
    /// Number of values (rows).
    #[must_use]
    pub const fn len(&self) -> usize {
        self.cells.len() / self.width
    }

    /// Whether the column contains no values.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.cells.is_empty()
    }

    /// Borrow the value at a given row.
    #[must_use]
    pub fn get(&self, index: usize) -> Option<&'a Complex> {
        if index >= self.len() {
            return None;
        }
        self.cells.get(index * self.width + self.column)
    }

    /// Iterate over borrowed cells without copying the column.
    pub fn iter(&self) -> std::iter::StepBy<std::slice::Iter<'a, Complex>> {
        self.cells
            .get(self.column..)
            .unwrap_or_default()
            .iter()
            .step_by(self.width)
    }

    /// First value, if present.
    #[must_use]
    pub fn first(&self) -> Option<&'a Complex> {
        self.get(0)
    }

    /// Last value, if present.
    #[must_use]
    pub fn last(&self) -> Option<&'a Complex> {
        self.len().checked_sub(1).and_then(|index| self.get(index))
    }

    /// Binary search on a partitioned column without extracting a copy.
    pub fn partition_point(&self, mut predicate: impl FnMut(&Complex) -> bool) -> usize {
        let mut low = 0;
        let mut high = self.len();
        while low < high {
            let middle = low + (high - low) / 2;
            if predicate(&self[middle]) {
                low = middle + 1;
            } else {
                high = middle;
            }
        }
        low
    }
}

impl std::ops::Index<usize> for ColumnView<'_> {
    type Output = Complex;
    fn index(&self, index: usize) -> &Self::Output {
        // A valid column has positive width. Match slice indexing semantics.
        &self
            .cells
            .chunks_exact(self.width)
            .nth(index)
            .unwrap_or(&[])[self.column]
    }
}

impl<'a> IntoIterator for ColumnView<'a> {
    type Item = Complex;
    type IntoIter = std::iter::Copied<std::iter::StepBy<std::slice::Iter<'a, Complex>>>;
    fn into_iter(self) -> Self::IntoIter {
        self.iter().copied()
    }
}

impl<'a> IntoIterator for &ColumnView<'a> {
    type Item = &'a Complex;
    type IntoIter = std::iter::StepBy<std::slice::Iter<'a, Complex>>;
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl PartialEq for ColumnView<'_> {
    fn eq(&self, other: &Self) -> bool {
        self.iter().eq(other.iter())
    }
}

impl std::fmt::Debug for ColumnView<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_list().entries(self).finish()
    }
}

#[cfg(test)]
mod tests {
    use super::{Plot, PlotFlags, Variable};
    use crate::primitives::Complex;

    fn plot() -> Plot {
        let mut plot = Plot::new(
            "op1",
            "Operating Point",
            PlotFlags::Real,
            vec![
                Variable::new("v(out)", "voltage"),
                Variable::new("i(v1)", "current"),
            ],
        );
        plot.push_point(&[Complex::real(2.5), Complex::real(-2.5e-3)])
            .unwrap();
        plot
    }

    #[test]
    fn values_are_addressed_by_name() {
        let plot = plot();
        assert_eq!(plot.value("V(OUT)", 0), Some(Complex::real(2.5)));
        assert_eq!(plot.value("v(out)", 1), None);
        assert_eq!(plot.variable_index("i(v1)"), Some(1));
        assert!(
            plot.column("i(v1)")
                .unwrap()
                .iter()
                .copied()
                .eq([Complex::real(-2.5e-3)])
        );
        assert_eq!(plot.point_count(), 1);
        assert_eq!(plot.variable_count(), 2);
        assert!(!plot.is_empty());
        assert!(plot.is_finite());
        assert_eq!(plot.max_abs(), 2.5);
    }

    #[test]
    fn points_must_match_the_columns() {
        let mut plot = plot();
        let error = plot
            .push_point(&[Complex::real(1.0)])
            .expect_err("arity mismatch");
        assert!(error.to_string().contains("1 value(s)"));
    }

    #[test]
    fn ordered_column_index_preserves_first_occurrence_and_alias_precedence() {
        let plot = Plot::new(
            "op1",
            "Operating Point",
            PlotFlags::Real,
            vec![
                Variable::new("V(OUT)", "voltage"),
                Variable::new("v(out)", "voltage"),
                Variable::new("out", "device"),
                Variable::new("i(out)", "current"),
                Variable::new("v(aux)", "voltage"),
                Variable::new("i(aux)", "current"),
                Variable::new("i(v1)", "current"),
                Variable::new("v1#branch", "device"),
                Variable::new("i(v2)", "current"),
            ],
        );
        let schema = plot.schema();
        assert_eq!(schema.index("v(OUT)"), Some(0));
        assert_eq!(schema.voltage("OUT"), Some(0));
        assert_eq!(schema.named("OUT"), Some(2));
        assert_eq!(schema.current("OUT"), Some(3));
        assert_eq!(schema.named("AUX"), Some(4));
        assert_eq!(schema.named("V1#branch"), Some(7));
        assert_eq!(schema.named("v2#branch"), Some(8));
        assert_eq!(schema.named("missing"), None);
    }

    #[test]
    fn complete_rows_are_checked_before_storage() {
        let plot = plot();
        let header = plot.header();
        assert!(std::sync::Arc::ptr_eq(plot.schema(), header.schema()));
        assert_eq!(header.point_count(), 0);
        assert!(
            header
                .clone()
                .with_rows(vec![vec![Complex::ZERO; 2], vec![Complex::ZERO; 3]])
                .is_err()
        );
        assert!(header.with_rows(vec![vec![Complex::ZERO]]).is_err());
    }

    #[test]
    fn collection_transfers_allocations_and_enforces_plot_lifecycle() {
        use super::{PlotCollector, PointSink};
        let mut collector = PlotCollector::default();
        let retained = plot();
        let row = retained.row(0).unwrap().as_ptr();
        collector.complete(retained).unwrap();
        let header = plot().header();
        let next = vec![Complex::real(1.), Complex::real(2.)];

        collector.begin(&header).unwrap();
        collector.point(&next).unwrap();
        collector.end().unwrap();
        let plots = collector.finish(2).unwrap();
        assert_eq!(plots[0].row(0).unwrap().as_ptr(), row);
        assert_eq!(plots[1].cells(), next);

        let mut collector = PlotCollector::default();
        assert!(collector.end().is_err());
        assert!(collector.point(&[]).is_err());
        collector.begin(&header).unwrap();
        assert!(collector.begin(&header).is_err());
        assert!(collector.complete(plot()).is_err());
        assert!(collector.point(&[Complex::ZERO]).is_err());
        assert!(collector.finish(1).is_err());
        assert!(PlotCollector::default().finish(1).is_err());
    }

    #[test]
    fn packed_rows_and_strided_columns_preserve_shape_and_mutation() {
        use std::sync::Arc;
        let mut plot = plot().header();
        assert!(plot.column("v(out)").unwrap().is_empty());
        assert!(plot.row(0).is_none());
        plot.reserve_points(4).unwrap();
        for i in 0..4 {
            plot.push_point(&[Complex::real(f64::from(i)), Complex::new(-f64::from(i), 1.)])
                .unwrap();
        }
        assert_eq!(plot.cells().len(), 8);
        assert_eq!(plot.row(1).unwrap().as_ptr(), plot.cells()[2..].as_ptr());
        assert_eq!(plot.rows().len(), 4);
        assert_eq!(plot.rows().next_back(), plot.row(3));
        let column = plot.column("v(out)").unwrap();
        assert_eq!(column.len(), 4);
        assert_eq!(column.first(), Some(&Complex::real(0.)));
        assert_eq!(column.last(), Some(&Complex::real(3.)));
        assert!(column.get(usize::MAX).is_none());
        assert!(plot.column_at(2).is_none());
        assert_eq!(column.partition_point(|value| value.re < 2.5), 3);
        assert_eq!(
            column.iter().rev().copied().collect::<Vec<_>>(),
            [3., 2., 1., 0.].map(Complex::real)
        );
        assert_eq!(
            column.get(2).unwrap() as *const Complex,
            &plot.cells()[4] as *const Complex
        );
        let copy = plot.clone();
        assert_eq!(copy, plot);
        assert!(Arc::ptr_eq(copy.schema(), plot.schema()));
        assert_ne!(copy.cells().as_ptr(), plot.cells().as_ptr());
        let before = plot.clone();
        assert!(plot.push_point(&[]).is_err());
        assert!(plot.reserve_points(usize::MAX).is_err());
        assert_eq!(plot, before);
        assert!(plot.remove_point(1));
        assert!(!plot.remove_point(usize::MAX));
        assert_eq!(
            plot.column("v(out)")
                .unwrap()
                .into_iter()
                .collect::<Vec<_>>(),
            [0., 2., 3.].map(Complex::real)
        );
        for row in plot.rows_mut() {
            row[1].im = 2.;
        }
        assert_eq!(plot.point_mut(1).unwrap()[1], Complex::new(-2., 2.));
        plot.truncate_points(2);
        assert_eq!(plot.point_count(), 2);
        assert_eq!(plot.cells().len(), 4);
        plot.truncate_points(usize::MAX);
        assert_eq!(plot.point_count(), 2);
        plot.clear_points();
        assert_eq!(plot.rows().len(), 0);
        assert!(plot.cells().is_empty());
        assert!(Arc::ptr_eq(copy.schema(), plot.schema()));
    }

    #[test]
    fn checked_flat_attachment_and_capacity_failure_are_atomic() {
        let header = plot().header();
        assert!(header.clone().with_cells(usize::MAX, vec![]).is_err());
        assert!(
            header
                .clone()
                .with_cells(2, vec![Complex::ZERO; 3])
                .is_err()
        );
        let data = vec![Complex::ZERO; 4];
        let pointer = data.as_ptr();
        let packed = header.with_cells(2, data).unwrap();
        assert_eq!(packed.cells().as_ptr(), pointer);
        let mut narrow = Plot::new(
            "one",
            "one",
            PlotFlags::Complex,
            vec![Variable::complex("v(out)", "voltage")],
        );
        // Product fits usize, but impossible allocation exceeds isize::MAX.
        let huge = isize::MAX as usize / std::mem::size_of::<Complex>() + 1;
        assert!(narrow.reserve_points(huge).is_err());
        assert_eq!(narrow.point_count(), 0);
        narrow
            .push_point(&[Complex::new(f64::INFINITY, -0.)])
            .unwrap();
        assert!(!narrow.is_finite());
        assert_eq!(narrow.cells()[0].im.to_bits(), (-0.0_f64).to_bits());
    }

    #[test]
    fn zero_column_policy_preserves_only_the_rootless_c_result() {
        let mut empty = Plot::new("empty", "empty", PlotFlags::Real, vec![]);
        assert!(empty.push_point(&[]).is_err());
        assert!(empty.clone().with_cells(1, vec![]).is_err());
        assert!(empty.reserve_points(1).is_err());
        let mut rootless = Plot::new("pz1", "Pole-Zero Analysis", PlotFlags::Real, vec![]);
        rootless.reserve_points(1).unwrap();
        rootless.push_point(&[]).unwrap();
        assert_eq!(rootless.point_count(), 1);
        assert!(rootless.cells().is_empty());
        assert_eq!(rootless.row(0), Some([].as_slice()));
        assert_eq!(rootless.rows().count(), 1);
        assert_eq!(rootless.rows_mut().count(), 1);
        assert!(rootless.point_mut(0).unwrap().is_empty());
        assert!(rootless.push_point(&[]).is_err());
        assert!(rootless.clone().with_cells(2, vec![]).is_err());
        assert!(rootless.remove_point(0));
        assert_eq!(rootless.point_count(), 0);
        assert!(!rootless.remove_point(0));
    }

    #[test]
    fn flags_are_read_from_the_rawfile_spelling() {
        assert_eq!(PlotFlags::parse("real"), PlotFlags::Real);
        assert_eq!(PlotFlags::parse("complex"), PlotFlags::Complex);
        assert_eq!(PlotFlags::parse("real poles"), PlotFlags::Real);
        assert_eq!(PlotFlags::parse("COMPLEX"), PlotFlags::Complex);
        assert_eq!(PlotFlags::Complex.as_rawfile(), "complex");
        assert!(PlotFlags::Complex.is_complex());
        assert!(!PlotFlags::Real.is_complex());
    }
}
