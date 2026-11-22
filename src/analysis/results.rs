//! Result plots: the data an analysis produces.
//!
//! Mirrors ngspice's `struct plot` from `src/include/ngspice/plot.h` and the
//! `struct dvec` it holds. ngspice stores an array of data vectors plus scale
//! information for the interactive plotter; the port keeps a flat table of
//! variables and points, which is what a rawfile contains, and leaves plotting
//! to whoever consumes the result.

use std::{borrow::Cow, collections::HashMap, sync::Arc};

use crate::primitives::{Complex, Real, SpiceError, SpiceResult};

/// Receives an analysis's points as they are produced, in place of a [`Plot`]
/// storing them.
///
/// C's batch output (`outitf.c`: `OUTpBeginPlot`, then `OUTpData` per point)
/// writes every point to the rawfile as it is computed and keeps no plot in
/// memory; [`crate::analysis::Analysis::run_into`] offers the same, so a long
/// transient costs memory for one point rather than for the whole run.
pub trait PointSink {
    /// Called once, before the first point, with the plot's names, flags and
    /// variables. The header holds no points.
    ///
    /// # Errors
    ///
    /// Whatever the sink reports; the analysis stops with that error.
    fn begin(&mut self, header: &Plot) -> SpiceResult<()>;

    /// One point, one value per variable of the header, in point order.
    ///
    /// # Errors
    ///
    /// Whatever the sink reports; the analysis stops with that error.
    fn point(&mut self, values: &[Complex]) -> SpiceResult<()>;
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
    /// Each row has exactly the width of the immutable schema.
    points: Vec<Vec<Complex>>,
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
            points: Vec::new(),
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
    pub fn schema(&self) -> &Arc<PlotSchema> {
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

    /// Stored rows.
    #[must_use]
    pub fn points(&self) -> &[Vec<Complex>] {
        &self.points
    }

    /// Borrow a row mutably without allowing changes to its width.
    pub fn point_mut(&mut self, index: usize) -> Option<&mut [Complex]> {
        self.points.get_mut(index).map(Vec::as_mut_slice)
    }

    /// Mutable fixed-width rows.
    pub fn rows_mut(&mut self) -> impl Iterator<Item = &mut [Complex]> {
        self.points.iter_mut().map(Vec::as_mut_slice)
    }

    /// Reserve row capacity.
    pub fn reserve_points(&mut self, additional: usize) {
        self.points.reserve_exact(additional);
    }

    /// Remove rows after the given length.
    pub fn truncate_points(&mut self, len: usize) {
        self.points.truncate(len);
    }

    /// Remove all rows, preserving the schema.
    pub fn clear_points(&mut self) {
        self.points.clear();
    }

    /// Remove and return a row by index.
    pub fn remove_point(&mut self, index: usize) -> Option<Vec<Complex>> {
        (index < self.points.len()).then(|| self.points.remove(index))
    }

    /// Take the stored rows, leaving the immutable header.
    pub fn take_points(&mut self) -> Vec<Vec<Complex>> {
        std::mem::take(&mut self.points)
    }

    /// A point-free header sharing the same schema.
    #[must_use]
    pub fn header(&self) -> Self {
        Self {
            name: self.name.clone(),
            plotname: self.plotname.clone(),
            schema: Arc::clone(&self.schema),
            points: Vec::new(),
        }
    }

    /// Seal a complete array of rows, checking every row against the schema.
    ///
    /// # Errors
    /// Any row has the wrong number of values.
    pub fn with_points(mut self, points: Vec<Vec<Complex>>) -> SpiceResult<Self> {
        for row in &points {
            self.check_point(row)?;
        }
        self.points = points;
        Ok(self)
    }

    /// Number of points.
    #[must_use]
    pub const fn point_count(&self) -> usize {
        self.points.len()
    }

    /// Number of columns.
    #[must_use]
    pub fn variable_count(&self) -> usize {
        self.variables().len()
    }

    /// True when the plot has no columns or no points.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.variables().is_empty() || self.points.is_empty()
    }

    /// Appends a point.
    ///
    /// # Errors
    ///
    /// [`SpiceError::Numerical`] when the point does not have one value per
    /// variable.
    pub fn push_point(&mut self, values: Vec<Complex>) -> SpiceResult<()> {
        self.check_point(&values)?;
        self.points.push(values);
        Ok(())
    }

    /// Checks that `values` could be appended: one value per variable.
    ///
    /// # Errors
    ///
    /// As [`Plot::push_point`].
    pub fn check_point(&self, values: &[Complex]) -> SpiceResult<()> {
        if values.len() != self.variables().len() {
            return Err(SpiceError::Numerical {
                context: format!("plot {}", self.name),
                message: format!(
                    "point has {} value(s) but the plot has {} variable(s)",
                    values.len(),
                    self.variables().len()
                ),
            });
        }
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
        self.points.get(point)?.get(column).copied()
    }

    /// Every value of one variable.
    #[must_use]
    pub fn column(&self, variable: &str) -> Option<Vec<Complex>> {
        let column = self.variable_index(variable)?;
        self.points
            .iter()
            .map(|point| point.get(column).copied())
            .collect()
    }

    /// Largest absolute value anywhere in the plot.
    #[must_use]
    pub fn max_abs(&self) -> Real {
        self.points
            .iter()
            .flat_map(|point| point.iter())
            .fold(0.0, |acc, value| acc.max(value.magnitude()))
    }

    /// True when every value is finite.
    #[must_use]
    pub fn is_finite(&self) -> bool {
        self.points
            .iter()
            .flat_map(|point| point.iter())
            .all(|value| value.is_finite())
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
        plot.push_point(vec![Complex::real(2.5), Complex::real(-2.5e-3)])
            .unwrap();
        plot
    }

    #[test]
    fn values_are_addressed_by_name() {
        let plot = plot();
        assert_eq!(plot.value("V(OUT)", 0), Some(Complex::real(2.5)));
        assert_eq!(plot.value("v(out)", 1), None);
        assert_eq!(plot.variable_index("i(v1)"), Some(1));
        assert_eq!(plot.column("i(v1)"), Some(vec![Complex::real(-2.5e-3)]));
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
            .push_point(vec![Complex::real(1.0)])
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
                .with_points(vec![vec![Complex::ZERO; 2], vec![Complex::ZERO; 3]])
                .is_err()
        );
        assert!(header.with_points(vec![vec![Complex::ZERO]]).is_err());
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
