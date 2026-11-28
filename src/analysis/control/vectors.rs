//! Bounded plot transforms for a native control session.
//!
//! Linearization shares the measurement/Fourier ordered-axis geometry.
//! Spectra share reusable exact-length Fourier and window plans.

use crate::analysis::fourier_resample::PolynomialPlan;
use crate::analysis::ordered_samples::OrderedAxis;
use crate::analysis::spectrum::{
    SpectrumLayout, SpectrumLength, SpectrumNormalization, SpectrumPlan, SpectrumSettings,
};
use crate::analysis::{Plot, PlotFlags, Variable};
use crate::primitives::{Complex, Real, SpiceError, SpiceResult, exact_nonnegative_integer};

fn failure(message: impl Into<String>) -> SpiceError {
    SpiceError::Numerical {
        context: "control plot transform".into(),
        message: message.into(),
    }
}

/// Retention and cumulative computational limits for plot transformations.
#[derive(Debug, Clone, Copy)]
pub struct TransformBudget {
    /// Free retained cells, including the new plot's scale.
    pub cells: usize,
    /// Remaining interpolation and FFT work units.
    pub work: usize,
}

impl TransformBudget {
    fn charge(&mut self, work: usize) -> SpiceResult<()> {
        self.work = self
            .work
            .checked_sub(work)
            .ok_or_else(|| failure("transform work budget exceeded"))?;
        Ok(())
    }

    fn check_cells(&self, points: usize, width: usize) -> SpiceResult<usize> {
        points
            .checked_mul(width)
            .filter(|count| *count <= self.cells)
            .ok_or_else(|| failure("transform cell budget exceeded"))
    }
}

/// Requested transient output grid, independent of accepted solver points.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TimeGrid {
    /// First output time.
    pub start: Real,
    /// Requested end time.
    pub stop: Real,
    /// Requested output spacing.
    pub step: Real,
}

impl TimeGrid {
    fn coordinates(
        self,
        requested_points: Option<u32>,
        max_points: usize,
    ) -> SpiceResult<Vec<Real>> {
        let Self { start, stop, step } = self;
        let ratio = (stop - start) / step;
        if ![start, stop, step, ratio]
            .iter()
            .all(|value| value.is_finite())
            || step <= 0.
            || ratio < 1.
        {
            return Err(failure("invalid linearization grid"));
        }
        let tolerance = 32. * Real::EPSILON * ratio;
        let count = match requested_points {
            Some(count) => count,
            None => exact_nonnegative_integer::<u32>((ratio + tolerance).floor())
                .and_then(|count| count.checked_add(1))
                .ok_or_else(|| failure("linearization point count overflow"))?,
        };
        let intervals = count
            .checked_sub(1)
            .filter(|count| *count != 0)
            .ok_or_else(|| failure("linearization needs at least two points"))?;
        if Real::from(intervals) > ratio + tolerance {
            return Err(failure("linearization grid exceeds requested stop"));
        }
        let length =
            usize::try_from(count).map_err(|_| failure("linearization point count overflow"))?;
        if length > max_points {
            return Err(failure("linearization point budget exceeded"));
        }
        let mut values = allocate(length)?;
        for (i, value) in (0..count).zip(&mut values) {
            *value = if i == intervals && (Real::from(i) - ratio).abs() <= tolerance {
                stop
            } else {
                Real::from(i).mul_add(step, start)
            };
        }
        if values.windows(2).any(|pair| pair[0] >= pair[1]) {
            return Err(failure("linearization grid does not advance"));
        }
        Ok(values)
    }
}

fn allocate<T: Default + Clone>(length: usize) -> SpiceResult<Vec<T>> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(length)
        .map_err(|error| failure(format!("allocating transform buffer: {error}")))?;
    values.resize(length, T::default());
    Ok(values)
}

fn time_axis(plot: &Plot) -> SpiceResult<OrderedAxis<'static>> {
    if plot.point_count() < 2 {
        return Err(failure(
            "a transform needs a time plot with at least two points",
        ));
    }
    let time = plot
        .column_at(
            plot.variable_index("time")
                .ok_or_else(|| failure("missing time scale"))?,
        )
        .ok_or_else(|| failure("missing time scale"))?;
    if (&time).into_iter().any(|value| !value.is_real()) {
        return Err(failure("time scale must be real"));
    }
    OrderedAxis::from_plot(plot, "time")
        .map_err(|error| failure(format!("invalid time scale: {error:?}")))
}

fn vector_columns(plot: &Plot, names: &[String]) -> SpiceResult<Vec<usize>> {
    let time = plot
        .variable_index("time")
        .ok_or_else(|| failure("missing time scale"))?;
    let mut columns = Vec::new();
    if names.is_empty() {
        columns.extend((0..plot.variable_count()).filter(|index| *index != time));
    } else {
        for name in names {
            let index = plot
                .schema()
                .named(name)
                .ok_or_else(|| failure(format!("unknown transform vector {name}")))?;
            if index == time {
                return Err(failure("time scale cannot be transformed as a signal"));
            }
            if !columns.contains(&index) {
                columns.push(index);
            }
        }
    }
    if columns.is_empty() {
        return Err(failure("no signal vectors selected"));
    }
    Ok(columns)
}

fn trace_components(plot: &Plot, index: usize, imaginary: bool) -> SpiceResult<Vec<Real>> {
    let column = plot
        .column_at(index)
        .ok_or_else(|| failure("missing transform column"))?;
    let mut trace = allocate::<Real>(column.len())?;
    for (destination, value) in trace.iter_mut().zip(&column) {
        if !value.is_finite() || (plot.variables()[index].is_real && !value.is_real()) {
            return Err(failure("invalid transform input"));
        }
        *destination = if imaginary { value.im } else { value.re };
    }
    Ok(trace)
}

/// Resample selected real or complex transient vectors onto a uniform physical grid.
/// An empty vector list selects every signal (not the time scale).
///
/// # Errors
/// Invalid axis/grid, discontinuous exact sample, unknown vector, nonfinite
/// input or resource exhaustion. No extrapolation beyond accepted data.
pub fn linearize(
    plot: &Plot,
    grid: TimeGrid,
    requested_points: Option<u32>,
    vectors: &[String],
    budget: &mut TransformBudget,
) -> SpiceResult<Plot> {
    let columns = vector_columns(plot, vectors)?;
    let width = columns.len() + 1;
    let coordinates = grid.coordinates(requested_points, budget.cells / width)?;
    let length = coordinates.len();
    let cells = budget.check_cells(length, width)?;
    let axis = time_axis(plot)?;
    let (geometry_work, _) = PolynomialPlan::budget(axis.len(), length, 1);
    let components = columns
        .iter()
        .map(|&index| {
            if plot.variables()[index].is_real {
                1usize
            } else {
                2
            }
        })
        .sum::<usize>();
    // Geometry and per-vector sampling are both bounded, independently of
    // how many times a script destroys its retained output.
    budget.charge(
        geometry_work.saturating_add(axis.len().saturating_add(length).saturating_mul(components)),
    )?;
    let interpolation = PolynomialPlan::new(&axis, &coordinates, 1)?;
    let mut result = allocate::<Complex>(cells)?;
    for (row, &time) in result.chunks_exact_mut(width).zip(&coordinates) {
        row[0] = Complex::real(time);
    }
    for (column, source) in columns.iter().copied().enumerate() {
        let trace = trace_components(plot, source, false)?;
        let sampled = interpolation.sample(&trace)?;
        for (row, value) in result.chunks_exact_mut(width).zip(sampled) {
            row[column + 1] = Complex::real(value);
        }
        if !plot.variables()[source].is_real {
            let trace = trace_components(plot, source, true)?;
            for (row, value) in result
                .chunks_exact_mut(width)
                .zip(interpolation.sample(&trace)?)
            {
                row[column + 1].im = value;
            }
        }
    }
    let mut variables = vec![Variable::new("time", "time")];
    variables.extend(columns.iter().map(|&index| plot.variables()[index].clone()));
    Plot::new(
        "tran",
        "Transient Analysis (linearized)",
        if columns.len() == components {
            PlotFlags::Real
        } else {
            PlotFlags::Complex
        },
        variables,
    )
    .with_cells(length, result)
}

fn real_count(value: usize) -> SpiceResult<Real> {
    u32::try_from(value)
        .map(Real::from)
        .map_err(|_| failure("transform length exceeds exact index range"))
}

fn uniform_spacing(axis: &OrderedAxis<'_>) -> SpiceResult<Real> {
    let times = axis.as_slice();
    let first = times
        .first()
        .copied()
        .ok_or_else(|| failure("empty time axis"))?;
    let last = times
        .last()
        .copied()
        .ok_or_else(|| failure("empty time axis"))?;
    let step = (last - first) / real_count(times.len() - 1)?;
    if !step.is_finite() || step <= 0. {
        return Err(failure("invalid FFT time spacing"));
    }
    // Roundoff of a stored coordinate is not physical sample jitter.
    // Reject genuinely nonuniform input instead of treating adaptive solver
    // samples as equally spaced; users can linearize before fft.
    for pair in times.windows(2) {
        let tolerance =
            (8. * Real::EPSILON * pair[0].abs().max(pair[1].abs()).max(step)).min(step * 1e-4);
        if ((pair[1] - pair[0]) - step).abs() > tolerance {
            return Err(failure(
                "FFT needs uniformly spaced samples; linearize first",
            ));
        }
    }
    Ok(step)
}

/// Windowed spectrum of uniformly sampled real or complex signals.
/// # Errors
/// Invalid scale/vector, nonuniform samples, numerical or resource failure.
pub fn fft(
    plot: &Plot,
    vectors: &[String],
    settings: SpectrumSettings,
    budget: &mut TransformBudget,
) -> SpiceResult<Plot> {
    let columns = vector_columns(plot, vectors)?;
    let axis = time_axis(plot)?;
    let step = uniform_spacing(&axis)?;
    let length = axis.len();
    let size = match settings.length {
        SpectrumLength::Exact => length,
        SpectrumLength::Padded => length
            .checked_next_power_of_two()
            .ok_or_else(|| failure("FFT size overflow"))?,
    };
    let layout = if columns
        .iter()
        .any(|&index| !plot.variables()[index].is_real)
    {
        SpectrumLayout::Complex
    } else {
        SpectrumLayout::Real
    };
    let bins = if layout == SpectrumLayout::Real {
        size / 2 + 1
    } else {
        size
    };
    let width = columns.len() + 1;
    let cells = budget.check_cells(bins, width)?;
    let work = SpectrumPlan::work_estimate(length, settings.length, columns.len())?;
    budget.charge(work)?;
    let plan = SpectrumPlan::new(length, step, layout, settings, work)?;
    let mut workspace = plan.workspace()?;
    let mut trace = allocate::<Complex>(length)?;
    let mut spectrum = allocate::<Complex>(bins)?;
    let mut result = allocate::<Complex>(cells)?;
    for (index, row) in result.chunks_exact_mut(width).enumerate() {
        row[0] = Complex::real(plan.frequency(index)?);
    }
    for (column, source) in columns.iter().copied().enumerate() {
        let values = plot
            .column_at(source)
            .ok_or_else(|| failure("missing transform column"))?;
        for (destination, value) in trace.iter_mut().zip(&values) {
            *destination = *value;
        }
        plan.transform(&trace, &mut workspace, &mut spectrum)?;
        for (row, &value) in result.chunks_exact_mut(width).zip(&spectrum) {
            row[column + 1] = value;
        }
    }
    let mut variables = vec![Variable::new("frequency", "frequency")];
    let flags = if settings.normalization == SpectrumNormalization::Amplitude {
        variables.extend(columns.iter().map(|&index| {
            let source = &plot.variables()[index];
            Variable::complex(source.name.clone(), source.unit.clone())
        }));
        PlotFlags::Complex
    } else {
        variables.extend(columns.iter().map(|&index| {
            let source = &plot.variables()[index];
            let unit = if settings.normalization == SpectrumNormalization::Power {
                format!("{}^2", source.unit)
            } else {
                format!("{}^2/Hz", source.unit)
            };
            Variable::new(source.name.clone(), unit)
        }));
        PlotFlags::Real
    };
    Plot::new("spectrum", "Spectrum", flags, variables).with_cells(bins, result)
}

// A vector expression is bound once against the plot's schema and scalar
// namespace; evaluation of each row is an allocation-free stack traversal.
use crate::netlist::expr::{BinaryOp, Expr, ExprKind, Function, ParameterExpression, UnaryOp};

#[derive(Clone, Copy)]
enum Unary {
    Negate,
    Magnitude,
    Decibels,
    Phase,
    RealPart,
    ImaginaryPart,
    RealFunction(Function),
}

impl Unary {
    fn evaluate(self, value: Complex) -> SpiceResult<Complex> {
        let real = match self {
            Self::Negate => return Ok(-value),
            Self::Magnitude => value.magnitude(),
            Self::Decibels => 20. * value.magnitude().log10(),
            Self::Phase => value.phase(),
            Self::RealPart => value.re,
            Self::ImaginaryPart => value.im,
            Self::RealFunction(function) if value.is_real() => match function {
                Function::Sqr => value.re * value.re,
                Function::Sqrt => value.re.sqrt(),
                Function::Sin => value.re.sin(),
                Function::Cos => value.re.cos(),
                Function::Tan => value.re.tan(),
                Function::Exp => value.re.exp(),
                Function::Ln => value.re.ln(),
                Function::Log | Function::Log10 => value.re.log10(),
                Function::Floor => value.re.floor(),
                Function::Ceil => value.re.ceil(),
                _ => {
                    return Err(failure(format!(
                        "unsupported vector function {}",
                        function.name()
                    )));
                }
            },
            Self::RealFunction(function) => {
                return Err(failure(format!(
                    "{} requires a real vector",
                    function.name()
                )));
            }
        };
        Ok(Complex::real(real))
    }
}

enum VectorOp {
    Constant(Complex),
    Column(usize),
    Unary(Unary),
    Binary(BinaryOp),
}

struct VectorCompiler<'a, F> {
    plot: &'a Plot,
    root: &'a ParameterExpression,
    scalar: F,
    code: Vec<VectorOp>,
    uses_vector: bool,
}

impl<F: Fn(&str) -> Option<Real>> VectorCompiler<'_, F> {
    fn is_vector(&self, node: &Expr) -> bool {
        match &node.kind {
            ExprKind::Number { .. } => false,
            ExprKind::Identifier(name) => {
                (self.scalar)(name).is_none() && self.plot.schema().named(name).is_some()
            }
            ExprKind::Unary { operand, .. } | ExprKind::Group(operand) => self.is_vector(operand),
            ExprKind::Binary { lhs, rhs, .. } => self.is_vector(lhs) || self.is_vector(rhs),
            ExprKind::Ternary {
                condition,
                then,
                otherwise,
            } => self.is_vector(condition) || self.is_vector(then) || self.is_vector(otherwise),
            ExprKind::Call { arguments, .. } => {
                arguments.iter().any(|argument| self.is_vector(argument))
            }
            ExprKind::UserCall { name, arguments } => {
                matches!(name.as_str(), "v" | "i")
                    || arguments.iter().any(|argument| self.is_vector(argument))
            }
        }
    }

    fn probe(&mut self, argument: &Expr, voltage: bool) -> SpiceResult<()> {
        let name = probe_name(self.root, argument)?;
        let schema = self.plot.schema();
        let index = if voltage {
            schema.voltage(name)
        } else {
            schema.current(name)
        };
        if let Some(index) = index {
            self.code.push(VectorOp::Column(index));
        } else if voltage && name == "0" {
            self.code.push(VectorOp::Constant(Complex::ZERO));
        } else {
            return Err(failure(format!("unknown vector probe {name}")));
        }
        self.uses_vector = true;
        Ok(())
    }

    fn call(&mut self, name: &str, arguments: &[Expr]) -> SpiceResult<()> {
        match (name, arguments) {
            ("v", [node]) => self.probe(node, true),
            ("v", [positive, negative]) => {
                self.probe(positive, true)?;
                self.probe(negative, true)?;
                self.code.push(VectorOp::Binary(BinaryOp::Sub));
                Ok(())
            }
            ("i", [device]) => self.probe(device, false),
            (_, [argument]) => {
                let operation =
                    match name {
                        "mag" | "abs" => Unary::Magnitude,
                        "db" => Unary::Decibels,
                        "ph" | "phase" => Unary::Phase,
                        "real" => Unary::RealPart,
                        "imag" => Unary::ImaginaryPart,
                        _ => Unary::RealFunction(Function::from_name(name).ok_or_else(|| {
                            failure(format!("unsupported vector function {name}"))
                        })?),
                    };
                self.compile(argument)?;
                self.code.push(VectorOp::Unary(operation));
                Ok(())
            }
            _ => Err(failure(format!(
                "unsupported vector call {name} or argument count"
            ))),
        }
    }

    fn compile(&mut self, node: &Expr) -> SpiceResult<()> {
        match &node.kind {
            ExprKind::Number { value, .. } => {
                self.code.push(VectorOp::Constant(Complex::real(*value)));
            }
            ExprKind::Identifier(name) => {
                if let Some(value) = (self.scalar)(name) {
                    self.code.push(VectorOp::Constant(Complex::real(value)));
                } else {
                    let column = self
                        .plot
                        .schema()
                        .named(name)
                        .ok_or_else(|| failure(format!("unknown scalar or vector {name}")))?;
                    self.code.push(VectorOp::Column(column));
                    self.uses_vector = true;
                }
            }
            ExprKind::Group(operand) => self.compile(operand)?,
            ExprKind::Unary { op, operand } => {
                self.compile(operand)?;
                match op {
                    UnaryOp::Plus => {}
                    UnaryOp::Minus => self.code.push(VectorOp::Unary(Unary::Negate)),
                    UnaryOp::Not => {
                        return Err(failure("logical negation is unsupported for vectors"));
                    }
                }
            }
            ExprKind::Binary { op, lhs, rhs } => {
                self.compile(lhs)?;
                self.compile(rhs)?;
                self.code.push(VectorOp::Binary(*op));
            }
            ExprKind::Ternary { .. } => {
                return Err(failure("conditional vector expressions are unsupported"));
            }
            ExprKind::Call {
                function,
                arguments,
            } => self.call(function.name(), arguments)?,
            ExprKind::UserCall { name, arguments } => self.call(name, arguments)?,
        }
        Ok(())
    }
}

fn probe_name<'a>(root: &'a ParameterExpression, argument: &Expr) -> SpiceResult<&'a str> {
    let range = argument
        .span
        .relative_to(&root.span)
        .ok_or_else(|| failure("invalid probe source span"))?;
    root.text
        .get(range)
        .map(str::trim)
        .ok_or_else(|| failure("invalid probe name"))
}

pub(crate) fn current_probes(root: &ParameterExpression) -> SpiceResult<Vec<String>> {
    root.root
        .preorder()
        .filter_map(|node| match &node.kind {
            ExprKind::UserCall { name, arguments } if name == "i" => {
                Some(match arguments.as_slice() {
                    [argument] => probe_name(root, argument).map(|name| format!("i({name})")),
                    _ => Err(failure("current probe needs one device")),
                })
            }
            _ => None,
        })
        .collect()
}

fn binary(operation: BinaryOp, left: Complex, right: Complex) -> SpiceResult<Complex> {
    match operation {
        BinaryOp::Add => Ok(left + right),
        BinaryOp::Sub => Ok(left - right),
        BinaryOp::Mul => Ok(left * right),
        BinaryOp::Div => Ok(left / right),
        BinaryOp::Pow if left.is_real() && right.is_real() => {
            Ok(Complex::real(left.re.powf(right.re)))
        }
        BinaryOp::Pow => Err(failure("complex vector powers are unsupported")),
        _ => Err(failure("non-arithmetic vector operator is unsupported")),
    }
}

fn stack_pop(stack: &mut Vec<Complex>) -> SpiceResult<Complex> {
    stack
        .pop()
        .ok_or_else(|| failure("invalid bound vector expression stack"))
}

fn evaluate_row(
    code: &[VectorOp],
    row: &[Complex],
    stack: &mut Vec<Complex>,
) -> SpiceResult<Complex> {
    stack.clear();
    for instruction in code {
        let value = match *instruction {
            VectorOp::Constant(value) => value,
            VectorOp::Column(index) => *row
                .get(index)
                .ok_or_else(|| failure("stale vector column binding"))?,
            VectorOp::Unary(operation) => operation.evaluate(stack_pop(stack)?)?,
            VectorOp::Binary(operation) => {
                let right = stack_pop(stack)?;
                let left = stack_pop(stack)?;
                binary(operation, left, right)?
            }
        };
        if !value.is_finite() {
            return Err(failure("nonfinite vector expression value"));
        }
        stack.push(value);
    }
    let result = stack_pop(stack)?;
    if !stack.is_empty() {
        return Err(failure("unbalanced vector expression stack"));
    }
    Ok(result)
}

/// Evaluate a vector expression on the current plot.
///
/// Scalar-only expressions return `None` so callers can use the ordinary scalar evaluator.
/// Supported vector operands are named columns, v(node[, node]), i(source),
/// numeric literals and scalar bindings. Vector arithmetic uses complex
/// addition, subtraction, multiplication and division; real powers and
/// common magnitude/phase/real-valued functions are also supported.
///
/// # Errors
/// Unknown vector/function, invalid dimensions, nonfinite result or budget
/// exhaustion. The source plot is not mutated.
pub fn evaluate_vector(
    plot: &Plot,
    expression: &ParameterExpression,
    scalar: impl Fn(&str) -> Option<Real>,
    budget: &mut TransformBudget,
) -> SpiceResult<Option<Vec<Complex>>> {
    let mut compiler = VectorCompiler {
        plot,
        root: expression,
        scalar,
        code: Vec::new(),
        uses_vector: false,
    };
    if !compiler.is_vector(&expression.root) {
        return Ok(None);
    }
    compiler.compile(&expression.root)?;
    if !compiler.uses_vector {
        return Err(failure("vector expression lost its column binding"));
    }
    let code = compiler.code;
    let length = plot.point_count();
    budget.check_cells(length, 1)?;
    budget.charge(length.saturating_mul(code.len()))?;
    let mut values = allocate::<Complex>(length)?;
    let mut stack = Vec::with_capacity(code.len());
    for (destination, row) in values.iter_mut().zip(plot.rows()) {
        *destination = evaluate_row(&code, row, &mut stack)?;
    }
    Ok(Some(values))
}
