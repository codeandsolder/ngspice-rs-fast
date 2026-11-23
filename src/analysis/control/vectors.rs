//! Bounded plot transforms for a native control session.
//!
//! Linearization shares the measurement/Fourier ordered-axis geometry.
//! FFT uses a native radix-2 transform, with the single-sided normalization
//! and periodic window conventions of frontend/com_fft.c and maths/fft/fftext.c.

use std::f64::consts::TAU;

use crate::analysis::fourier_resample::PolynomialPlan;
use crate::analysis::ordered_samples::OrderedAxis;
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
    if plot.flags().is_complex() || plot.point_count() < 2 {
        return Err(failure(
            "a transform needs a real time plot with at least two points",
        ));
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

fn real_trace(plot: &Plot, index: usize) -> SpiceResult<Vec<Real>> {
    let column = plot
        .column_at(index)
        .ok_or_else(|| failure("missing transform column"))?;
    let mut trace = Vec::new();
    trace
        .try_reserve_exact(column.len())
        .map_err(|error| failure(format!("allocating transform trace: {error}")))?;
    for value in column.iter() {
        if !value.is_real() || !value.is_finite() {
            return Err(failure("non-real or nonfinite transform input"));
        }
        trace.push(value.re);
    }
    Ok(trace)
}

/// Resample selected real transient vectors onto a uniform physical grid.
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
    // Geometry and per-vector sampling are both bounded, independently of
    // how many times a script destroys its retained output.
    budget.charge(
        geometry_work.saturating_add(
            axis.len()
                .saturating_add(length)
                .saturating_mul(columns.len()),
        ),
    )?;
    let interpolation = PolynomialPlan::new(&axis, &coordinates, 1)?;
    let mut result = allocate::<Complex>(cells)?;
    for (row, &time) in result.chunks_exact_mut(width).zip(&coordinates) {
        row[0] = Complex::real(time);
    }
    for (column, source) in columns.iter().copied().enumerate() {
        let trace = real_trace(plot, source)?;
        let sampled = interpolation.sample(&trace)?;
        for (row, value) in result.chunks_exact_mut(width).zip(sampled) {
            row[column + 1] = Complex::real(value);
        }
    }
    let mut variables = vec![Variable::new("time", "time")];
    variables.extend(columns.iter().map(|&index| plot.variables()[index].clone()));
    Plot::new(
        "tran",
        "Transient Analysis (linearized)",
        PlotFlags::Real,
        variables,
    )
    .with_cells(length, result)
}

/// Periodic amplitude-normalized FFT window.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum SpectrumWindow {
    /// No taper (unity).
    Rectangular,
    /// Hann (ngspice default).
    #[default]
    Hann,
    /// Hamming.
    Hamming,
    /// Bartlett/triangle.
    Triangle,
    /// Blackman.
    Blackman,
    /// Four-term Blackman-Harris.
    BlackmanHarris,
    /// Five-term flat top.
    FlatTop,
}

impl SpectrumWindow {
    /// Parse the supported ngspice window names and aliases.
    #[must_use]
    pub fn parse(name: &str) -> Option<Self> {
        match name.to_ascii_lowercase().as_str() {
            "none" | "rectangular" => Some(Self::Rectangular),
            "hann" | "hanning" | "cosine" => Some(Self::Hann),
            "hamming" => Some(Self::Hamming),
            "triangle" | "bartlet" | "bartlett" => Some(Self::Triangle),
            "blackman" => Some(Self::Blackman),
            "blackmanharris" => Some(Self::BlackmanHarris),
            "flattop" => Some(Self::FlatTop),
            _ => None,
        }
    }

    fn weight(self, position: Real) -> Real {
        let angle = TAU * position;
        let c1 = angle.cos();
        match self {
            Self::Rectangular => 1.,
            Self::Hann => 1. - c1,
            Self::Hamming => 1. - (0.46 / 0.54) * c1,
            Self::Triangle => 4. * position.abs().min(1. - position.abs()),
            Self::Blackman => {
                (0.08 / 0.42_f64).mul_add((2. * angle).cos(), 1. - (0.50 / 0.42) * c1)
            }
            Self::BlackmanHarris => {
                let c2 = (2. * angle).cos();
                let c3 = (3. * angle).cos();
                (-0.01168 / 0.35875_f64).mul_add(
                    c3,
                    (0.14128 / 0.35875_f64).mul_add(c2, 1. - (0.48829 / 0.35875) * c1),
                )
            }
            Self::FlatTop => {
                let c2 = (2. * angle).cos();
                let c3 = (3. * angle).cos();
                let c4 = (4. * angle).cos();
                0.032_f64.mul_add(
                    c4,
                    (-0.388_f64).mul_add(c3, 1.29_f64.mul_add(c2, 1. - 1.93 * c1)),
                )
            }
        }
    }
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

struct Radix2 {
    size: usize,
    twiddles: Vec<Complex>,
}

impl Radix2 {
    fn new(size: usize) -> SpiceResult<Self> {
        let mut twiddles = allocate::<Complex>(size / 2)?;
        let denominator = real_count(size)?;
        for (index, twiddle) in twiddles.iter_mut().enumerate() {
            let (sin, cos) = (-TAU * real_count(index)? / denominator).sin_cos();
            *twiddle = Complex::new(cos, sin);
        }
        Ok(Self { size, twiddles })
    }

    fn transform(&self, values: &mut [Complex]) {
        let bits = self.size.ilog2();
        for index in 0..self.size {
            let reverse = index.reverse_bits() >> (usize::BITS - bits);
            if index < reverse {
                values.swap(index, reverse);
            }
        }
        for stage in 1..=bits {
            let width = 1usize << stage;
            let half = width / 2;
            let stride = self.size / width;
            for chunk in values.chunks_exact_mut(width) {
                let (left, right) = chunk.split_at_mut(half);
                for (index, (a, b)) in left.iter_mut().zip(right).enumerate() {
                    let rotated = *b * self.twiddles[index * stride];
                    let original = *a;
                    *a = original + rotated;
                    *b = original - rotated;
                }
            }
        }
    }
}

fn spectrum_trace(
    trace: &[Real],
    window: &[Real],
    plan: &Radix2,
    buffer: &mut [Complex],
    mut output: impl FnMut(usize, Complex) -> SpiceResult<()>,
) -> SpiceResult<()> {
    let count = real_count(trace.len())?;
    let maximum = trace
        .iter()
        .fold(0_f64, |maximum, value| maximum.max(value.abs()));
    let max_weight = window
        .iter()
        .fold(1_f64, |maximum, value| maximum.max(value.abs()));
    // In normal ranges keep the original data; range-scale extreme inputs
    // before a butterfly could overflow or erase an entire tiny signal.
    let scale = if maximum > Real::MAX / (count * max_weight)
        || (maximum > 0. && maximum < Real::MIN_POSITIVE * count)
    {
        maximum
    } else {
        1.
    };
    buffer.fill(Complex::ZERO);
    for ((destination, &sample), &weight) in buffer.iter_mut().zip(trace).zip(window) {
        *destination = Complex::real((sample / scale) * weight);
    }
    plan.transform(buffer);
    let nyquist = plan.size / 2;
    for (index, value) in buffer.iter().take(nyquist + 1).enumerate() {
        let boundary = index == 0 || index == nyquist;
        let factor = if boundary { 1. / count } else { 2. / count };
        let result = Complex::new(
            (value.re * factor) * scale,
            if boundary {
                0.
            } else {
                (value.im * factor) * scale
            },
        );
        if !result.is_finite() {
            return Err(failure("FFT spectrum is not finite"));
        }
        output(index, result)?;
    }
    Ok(())
}

/// Single-sided complex amplitude spectrum of uniform real samples.
/// Non-power-of-two lengths are zero padded; amplitude is normalized by
/// the actual sample count (not the padded length).
///
/// # Errors
/// Invalid scale/vector, nonuniform samples, numerical or resource failure.
pub fn fft(
    plot: &Plot,
    vectors: &[String],
    window: SpectrumWindow,
    budget: &mut TransformBudget,
) -> SpiceResult<Plot> {
    let columns = vector_columns(plot, vectors)?;
    let axis = time_axis(plot)?;
    let step = uniform_spacing(&axis)?;
    let length = axis.len();
    let size = length
        .checked_next_power_of_two()
        .ok_or_else(|| failure("FFT size overflow"))?;
    let bins = size / 2 + 1;
    let width = columns.len() + 1;
    let cells = budget.check_cells(bins, width)?;
    budget.charge(
        size.saturating_mul(size.ilog2() as usize + 1)
            .saturating_mul(columns.len()),
    )?;
    let sample_count = real_count(length)?;
    let span = step * real_count(size)?;
    let frequency_step = 1. / span;
    if !frequency_step.is_finite() || frequency_step <= 0. {
        return Err(failure("invalid FFT frequency spacing"));
    }
    let mut weights = allocate::<Real>(length)?;
    for (index, weight) in weights.iter_mut().enumerate() {
        *weight = window.weight((real_count(index)? - (sample_count - 1.)) / sample_count);
    }
    let plan = Radix2::new(size)?;
    let mut buffer = allocate::<Complex>(size)?;
    let mut result = allocate::<Complex>(cells)?;
    for (index, row) in result.chunks_exact_mut(width).enumerate() {
        let frequency = real_count(index)? * frequency_step;
        if !frequency.is_finite() {
            return Err(failure("FFT frequency overflows"));
        }
        row[0] = Complex::real(frequency);
    }
    for (column, source) in columns.iter().copied().enumerate() {
        let trace = real_trace(plot, source)?;
        spectrum_trace(&trace, &weights, &plan, &mut buffer, |index, value| {
            result[index * width + column + 1] = value;
            Ok(())
        })?;
    }
    let mut variables = vec![Variable::new("frequency", "frequency")];
    variables.extend(columns.iter().map(|&index| {
        let source = &plot.variables()[index];
        Variable::complex(source.name.clone(), source.unit.clone())
    }));
    Plot::new("spectrum", "Spectrum", PlotFlags::Complex, variables).with_cells(bins, result)
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
        let range = argument
            .span
            .relative_to(&self.root.span)
            .ok_or_else(|| failure("invalid probe source span"))?;
        let name = self
            .root
            .text
            .get(range)
            .ok_or_else(|| failure("invalid probe name"))?
            .trim();
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
                self.code.push(VectorOp::Constant(Complex::real(*value)))
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
                if *op == UnaryOp::Minus {
                    self.code.push(VectorOp::Unary(Unary::Negate));
                }
            }
            ExprKind::Binary { op, lhs, rhs } => {
                self.compile(lhs)?;
                self.compile(rhs)?;
                self.code.push(VectorOp::Binary(*op));
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

/// Evaluate a vector expression on the current plot. Scalar-only expressions
/// return None so callers can use the ordinary scalar evaluator instead.
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
