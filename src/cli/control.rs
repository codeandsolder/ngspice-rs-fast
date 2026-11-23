//! Bounded native control script execution.
//!
//! The circuit parser preserves positioned commands;
//! the entire script is compiled before the first simulation runs.

use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet};
use std::io::Write;
use std::path::Path;

use crate::analysis::control::vectors::{self, SpectrumWindow, TransformBudget};
use crate::analysis::control::{ControlPlot, ControlSession};
use crate::analysis::optimization::parameters::CircuitParameterPlan;
use crate::analysis::{Variable, measure};
use crate::netlist::ast::{AnalysisCard, MeasureCard, Netlist, ScopedCardKind};
use crate::netlist::eval::{EvalBudget, EvalLimits, ParamBinding, ParamScope};
use crate::netlist::parser::{control_analysis, control_measure};
use crate::netlist::{
    DotCommand, ParameterExpression, Parser, RawCard, SourceLimits, parse_parameter_expression,
};
use crate::primitives::{AnalysisKind, Real, SourceLoc, SpiceError, SpiceResult};

/// Independent work and storage bounds for one script invocation.
#[derive(Debug, Clone, Copy)]
pub struct ScriptLimits {
    /// Executed instructions, including each loop iteration/condition.
    pub steps: usize,
    /// Retained plot count.
    pub plots: usize,
    /// Retained scalar/complex plot cells.
    pub cells: usize,
    /// Cumulative deterministic analysis grid points.
    pub grid_points: usize,
    /// Cumulative interpolation/FFT computation budget.
    pub transform_work: usize,
    /// Number of distinct scalar variables.
    pub variables: usize,
    /// Printed bytes.
    pub output_bytes: usize,
    /// Scalar expression evaluation budget.
    pub expressions: EvalLimits,
}

impl Default for ScriptLimits {
    fn default() -> Self {
        Self {
            steps: 100_000,
            plots: 128,
            cells: 8_000_000,
            grid_points: 10_000_000,
            transform_work: 100_000_000,
            variables: 4096,
            output_bytes: 16_000_000,
            expressions: EvalLimits::default(),
        }
    }
}

/// Final script state; plots are retained in execution order until destroyed.
pub struct ScriptReport {
    /// Scalar control variables and measurement results, by folded name.
    pub variables: BTreeMap<String, Real>,
    /// Retained plots and their originating analysis kinds.
    pub plots: Vec<ControlPlot>,
    /// Instructions and loop iterations consumed.
    pub steps: usize,
}

type Scalar = Box<ParameterExpression>;

struct Instruction {
    location: SourceLoc,
    operation: Operation,
}

enum Operation {
    Let(String, Scalar),
    AlterParameter(String, Scalar),
    Reset,
    Run,
    Analysis(Box<AnalysisCard>),
    Measure(Box<MeasureCard>),
    Print(Vec<String>),
    Transform(Transform, Vec<String>),
    SetWindow(SpectrumWindow),
    SetPlot(String),
    Foreach(String, Vec<Scalar>, Vec<Instruction>),
    While(Condition, Vec<Instruction>),
    DestroyAll,
    Quit,
}

#[derive(Clone, Copy)]
enum Transform {
    Linearize(Option<u32>),
    Spectrum,
}

enum Condition {
    Nonzero(Scalar),
    Compare(Scalar, Comparison, Scalar),
}

#[derive(Clone, Copy)]
enum Comparison {
    Less,
    LessEqual,
    Equal,
    NotEqual,
    GreaterEqual,
    Greater,
}

impl Comparison {
    const fn matches(self, ordering: Ordering) -> bool {
        match self {
            Self::Less => matches!(ordering, Ordering::Less),
            Self::LessEqual => !matches!(ordering, Ordering::Greater),
            Self::Equal => matches!(ordering, Ordering::Equal),
            Self::NotEqual => !matches!(ordering, Ordering::Equal),
            Self::GreaterEqual => !matches!(ordering, Ordering::Less),
            Self::Greater => matches!(ordering, Ordering::Greater),
        }
    }
}

fn error(location: &SourceLoc, message: impl Into<String>) -> SpiceError {
    SpiceError::parse(location.clone(), message)
}

fn unsupported(location: &SourceLoc, command: &str) -> SpiceError {
    SpiceError::not_yet_ported(
        format!("{location}: control command {command}"),
        "src/frontend/control.c, commands.c",
    )
}

fn identifier(name: &str, location: &SourceLoc) -> SpiceResult<String> {
    let mut bytes = name.bytes();
    if !bytes
        .next()
        .is_some_and(|byte| byte.is_ascii_alphabetic() || byte == b'_')
        || !bytes.all(|byte| byte.is_ascii_alphanumeric() || byte == b'_')
    {
        return Err(error(location, format!("invalid scalar name {name:?}")));
    }
    Ok(name.to_ascii_lowercase())
}

// Numeric variable expansion is lexical, not a shell/text substitution.
// Bare names and $&name both resolve in the same scalar namespace.
fn expression(text: &str, location: &SourceLoc) -> SpiceResult<Scalar> {
    let text = text.trim();
    let text = text
        .strip_prefix('{')
        .and_then(|s| s.strip_suffix('}'))
        .unwrap_or(text);
    let mut normalized = String::with_capacity(text.len());
    let mut chars = text.chars().peekable();
    while let Some(ch) = chars.next() {
        if ch != '$' {
            normalized.push(ch);
            continue;
        }
        if chars.peek() == Some(&'&') {
            chars.next();
        }
        if !chars
            .peek()
            .is_some_and(|ch| ch.is_ascii_alphabetic() || *ch == '_')
            || normalized
                .chars()
                .next_back()
                .is_some_and(|ch| ch.is_ascii_alphanumeric() || ch == '_')
        {
            return Err(error(location, "invalid numeric variable substitution"));
        }
    }
    Ok(Box::new(parse_parameter_expression(&normalized, location)?))
}

fn assignment(text: &str, location: &SourceLoc) -> SpiceResult<(String, Scalar)> {
    let (name, value) = text
        .split_once('=')
        .ok_or_else(|| error(location, "expected name = expression"))?;
    Ok((
        identifier(name.trim(), location)?,
        expression(value, location)?,
    ))
}

fn condition(text: &str, location: &SourceLoc) -> SpiceResult<Condition> {
    let Some(index) = text.find(['<', '>', '=', '!']) else {
        return Ok(Condition::Nonzero(expression(text, location)?));
    };
    let (left, tail) = text.split_at(index);
    let (comparison, right) = [
        ("<=", Comparison::LessEqual),
        (">=", Comparison::GreaterEqual),
        ("==", Comparison::Equal),
        ("!=", Comparison::NotEqual),
        ("<>", Comparison::NotEqual),
        ("<", Comparison::Less),
        (">", Comparison::Greater),
        ("=", Comparison::Equal),
    ]
    .into_iter()
    .find_map(|(prefix, comparison)| tail.strip_prefix(prefix).map(|right| (comparison, right)))
    .ok_or_else(|| error(location, "invalid comparison"))?;
    Ok(Condition::Compare(
        expression(left, location)?,
        comparison,
        expression(right, location)?,
    ))
}

fn split_command(card: &RawCard) -> (&str, &str) {
    let text = card.raw.trim();
    text.split_once(char::is_whitespace)
        .map_or((text, ""), |(name, rest)| (name, rest.trim()))
}

fn empty_arguments(rest: &str, location: &SourceLoc) -> SpiceResult<()> {
    if rest.is_empty() {
        Ok(())
    } else {
        Err(error(location, "unexpected command arguments"))
    }
}

fn linearize_command(rest: &str, location: &SourceLoc) -> SpiceResult<Operation> {
    let mut points = None;
    let mut vectors = Vec::new();
    for word in rest.split_ascii_whitespace() {
        let word = word.to_ascii_lowercase();
        if let Some(count) = word.strip_prefix("np=") {
            let count = count
                .parse::<u32>()
                .ok()
                .filter(|count| *count >= 2)
                .ok_or_else(|| error(location, "linearize np needs an integer of at least two"))?;
            if points.replace(count).is_some() {
                return Err(error(location, "duplicate linearize np setting"));
            }
        } else {
            vectors.push(word);
        }
    }
    Ok(Operation::Transform(Transform::Linearize(points), vectors))
}

fn window_command(rest: &str, location: &SourceLoc) -> SpiceResult<Operation> {
    let (name, value) = rest
        .split_once('=')
        .ok_or_else(|| error(location, "expected set specwindow = name"))?;
    if !name.trim().eq_ignore_ascii_case("specwindow") {
        return Err(unsupported(location, name.trim()));
    }
    let value = value.trim().trim_matches('"');
    let window = SpectrumWindow::parse(value)
        .ok_or_else(|| unsupported(location, &format!("specwindow={value}")))?;
    Ok(Operation::SetWindow(window))
}

struct Compiler<'a> {
    cards: &'a [&'a RawCard],
    cursor: usize,
    auto_gnd: bool,
    tunables: &'a mut BTreeSet<String>,
}

impl Compiler<'_> {
    fn block(
        &mut self,
        opening: Option<&SourceLoc>,
        depth: usize,
    ) -> SpiceResult<Vec<Instruction>> {
        if depth > 32 {
            return Err(SpiceError::circuit("control nesting limit exceeded"));
        }
        let mut commands = Vec::new();
        while let Some(card) = self.cards.get(self.cursor).copied() {
            self.cursor += 1;
            let (name, rest) = split_command(card);
            if name.eq_ignore_ascii_case("end") {
                empty_arguments(rest, &card.location)?;
                if opening.is_none() {
                    return Err(error(&card.location, "end without a loop"));
                }
                return Ok(commands);
            }
            let operation = self.operation(card, name, rest, depth)?;
            commands.push(Instruction {
                location: card.location.clone(),
                operation,
            });
        }
        if let Some(location) = opening {
            return Err(error(location, "loop missing end"));
        }
        Ok(commands)
    }

    fn operation(
        &mut self,
        card: &RawCard,
        name: &str,
        rest: &str,
        depth: usize,
    ) -> SpiceResult<Operation> {
        let location = &card.location;
        match name.to_ascii_lowercase().as_str() {
            "let" => {
                let (name, value) = assignment(rest, location)?;
                Ok(Operation::Let(name, value))
            }
            "alterparam" => {
                let (name, value) = assignment(rest, location)?;
                self.tunables.insert(name.clone());
                Ok(Operation::AlterParameter(name, value))
            }
            "while" => Ok(Operation::While(
                condition(rest, location)?,
                self.block(Some(location), depth + 1)?,
            )),
            "foreach" => {
                let mut words = rest.split_ascii_whitespace();
                let name = identifier(words.next().unwrap_or_default(), location)?;
                let values = words
                    .map(|word| expression(word, location))
                    .collect::<SpiceResult<Vec<_>>>()?;
                if values.is_empty() {
                    return Err(error(location, "foreach requires at least one value"));
                }
                Ok(Operation::Foreach(
                    name,
                    values,
                    self.block(Some(location), depth + 1)?,
                ))
            }
            "meas" | "measure" => Ok(Operation::Measure(Box::new(control_measure(
                card,
                self.auto_gnd,
            )?))),
            "print" => {
                let values: Vec<_> = rest
                    .split_ascii_whitespace()
                    .map(str::to_ascii_lowercase)
                    .collect();
                if values.is_empty() {
                    return Err(error(location, "print needs a scalar or vector name"));
                }
                Ok(Operation::Print(values))
            }
            "linearize" => linearize_command(rest, location),
            "fft" => Ok(Operation::Transform(
                Transform::Spectrum,
                rest.split_ascii_whitespace()
                    .map(str::to_ascii_lowercase)
                    .collect(),
            )),
            "set" => window_command(rest, location),
            "setplot" => {
                if rest.is_empty() || rest.contains(char::is_whitespace) {
                    return Err(error(location, "setplot needs one retained plot name"));
                }
                Ok(Operation::SetPlot(rest.to_ascii_lowercase()))
            }
            "destroy" if rest.eq_ignore_ascii_case("all") => Ok(Operation::DestroyAll),
            "run" | "reset" | "quit" => {
                empty_arguments(rest, location)?;
                Ok(match name.to_ascii_lowercase().as_str() {
                    "run" => Operation::Run,
                    "reset" => Operation::Reset,
                    _ => Operation::Quit,
                })
            }
            _ => match AnalysisKind::parse(name) {
                Some(kind) if crate::analysis::has_driver(kind) && !name.starts_with('.') => Ok(
                    Operation::Analysis(Box::new(control_analysis(card, kind, self.auto_gnd)?)),
                ),
                _ => Err(unsupported(location, name)),
            },
        }
    }
}

struct Program {
    instructions: Vec<Instruction>,
    tunables: BTreeSet<String>,
}

impl Program {
    fn compile(netlist: &Netlist, auto_gnd: bool) -> SpiceResult<Self> {
        let mut program = Self {
            instructions: Vec::new(),
            tunables: BTreeSet::new(),
        };
        let mut cards = Vec::new();
        for scoped in &netlist.cards {
            if scoped.kind != ScopedCardKind::Control {
                continue;
            }
            let card = scoped.source.as_ref();
            match card.dot_command() {
                Some(DotCommand::Control) => cards.clear(),
                Some(DotCommand::Endc) => {
                    let mut compiler = Compiler {
                        cards: &cards,
                        cursor: 0,
                        auto_gnd,
                        tunables: &mut program.tunables,
                    };
                    program.instructions.extend(compiler.block(None, 0)?);
                    cards.clear();
                }
                _ => cards.push(card),
            }
        }
        if program.instructions.is_empty() {
            return Err(error(&netlist.location, "no executable .control commands"));
        }
        Ok(program)
    }
}

struct State<'a, 'w> {
    session: ControlSession<'a>,
    analyses: &'a [AnalysisCard],
    limits: ScriptLimits,
    variables: BTreeMap<String, Real>,
    pending: BTreeMap<String, Real>,
    plots: Vec<ControlPlot>,
    cells: usize,
    current: Option<usize>,
    next_plot_id: usize,
    window: SpectrumWindow,
    remaining_transform_work: usize,
    remaining_steps: usize,
    remaining_grid: usize,
    expressions: EvalBudget,
    output: ScriptOutput<'w>,
}

struct ScriptOutput<'a> {
    remaining: usize,
    writer: &'a mut dyn Write,
}

impl ScriptOutput<'_> {
    fn emit(&mut self, text: &str, location: &SourceLoc) -> SpiceResult<()> {
        self.remaining = self
            .remaining
            .checked_sub(text.len())
            .ok_or_else(|| error(location, "control output byte budget exceeded"))?;
        self.writer
            .write_all(text.as_bytes())
            .map_err(|failure| SpiceError::io(location.path(), &failure))
    }
}

impl State<'_, '_> {
    fn step(&mut self, location: &SourceLoc) -> SpiceResult<()> {
        self.remaining_steps = self
            .remaining_steps
            .checked_sub(1)
            .ok_or_else(|| error(location, "control instruction budget exceeded"))?;
        Ok(())
    }

    fn evaluate(&mut self, expression: &ParameterExpression) -> SpiceResult<Real> {
        self.session.evaluate(
            expression,
            |name| self.variables.get(name).copied(),
            &mut self.expressions,
        )
    }

    fn test(&mut self, condition: &Condition) -> SpiceResult<bool> {
        match condition {
            Condition::Nonzero(value) => Ok(self.evaluate(value)? != 0.),
            Condition::Compare(left, comparison, right) => {
                let left = self.evaluate(left)?;
                let right = self.evaluate(right)?;
                let ordering = left
                    .partial_cmp(&right)
                    .ok_or_else(|| SpiceError::circuit("nonfinite control comparison"))?;
                Ok(comparison.matches(ordering))
            }
        }
    }

    fn bind(&mut self, name: &str, value: Real, location: &SourceLoc) -> SpiceResult<()> {
        if !value.is_finite() {
            return Err(error(location, "nonfinite control variable"));
        }
        if !self.variables.contains_key(name) && self.variables.len() >= self.limits.variables {
            return Err(error(location, "control variable budget exceeded"));
        }
        self.variables.insert(name.to_owned(), value);
        Ok(())
    }

    fn plan_script_card(
        &mut self,
        card: &AnalysisCard,
    ) -> SpiceResult<crate::analysis::AnalysisPlan> {
        let mut card = card.clone();
        for argument in &mut card.arguments {
            if argument.starts_with('$') {
                let value = self.evaluate(&*expression(argument, &card.location)?)?;
                *argument = crate::netlist::elaborate::format_literal(value);
            }
        }
        for argument in std::mem::take(&mut card.expressions) {
            let value = self.evaluate(&argument.expression)?;
            let destination = card
                .arguments
                .get_mut(argument.index)
                .ok_or_else(|| error(&card.location, "stale control analysis argument"))?;
            *destination = crate::netlist::elaborate::format_literal(value);
        }
        self.session.plan(&card)
    }

    fn run_plans(
        &mut self,
        plans: &[crate::analysis::AnalysisPlan],
        location: &SourceLoc,
    ) -> SpiceResult<()> {
        if plans.is_empty() {
            return Err(error(location, "run requires deck analysis cards"));
        }
        let plots = self.session.collect_batch(
            plans,
            self.limits.plots - self.plots.len(),
            self.limits.cells - self.cells,
            &mut self.remaining_grid,
        )?;
        for plot in plots {
            self.retain(plot, location)?;
        }
        Ok(())
    }

    fn retain(&mut self, mut result: ControlPlot, location: &SourceLoc) -> SpiceResult<()> {
        if self.plots.len() >= self.limits.plots {
            return Err(error(location, "control plot count budget exceeded"));
        }
        let cells = self
            .cells
            .checked_add(result.plot.cells().len())
            .filter(|count| *count <= self.limits.cells)
            .ok_or_else(|| error(location, "control plot cell budget exceeded"))?;
        let prefix = if result.kind == AnalysisKind::Fourier {
            "spectrum"
        } else {
            crate::analysis::batch::plot_abbreviation(result.kind)
        };
        result.plot.name = format!("{prefix}{}", self.next_plot_id);
        self.next_plot_id = self
            .next_plot_id
            .checked_add(1)
            .ok_or_else(|| error(location, "control plot identifier overflow"))?;
        self.current = Some(self.plots.len());
        self.plots.push(result);
        self.cells = cells;
        Ok(())
    }

    fn current_plot(&self, location: &SourceLoc) -> SpiceResult<&ControlPlot> {
        self.current
            .and_then(|index| self.plots.get(index))
            .ok_or_else(|| error(location, "no current plot"))
    }

    fn select_plot(&mut self, name: &str, location: &SourceLoc) -> SpiceResult<()> {
        self.current = Some(
            self.plots
                .iter()
                .position(|plot| plot.plot.name.eq_ignore_ascii_case(name))
                .ok_or_else(|| error(location, format!("unknown plot {name}")))?,
        );
        Ok(())
    }

    fn transform(
        &mut self,
        transform: Transform,
        names: &[String],
        location: &SourceLoc,
    ) -> SpiceResult<()> {
        if self.plots.len() >= self.limits.plots {
            return Err(error(location, "control plot count budget exceeded"));
        }
        let current = self.current_plot(location)?;
        let mut budget = TransformBudget {
            cells: self.limits.cells - self.cells,
            work: self.remaining_transform_work,
        };
        let result = if let Transform::Linearize(points) = transform {
            let grid = current
                .grid
                .ok_or_else(|| error(location, "linearize needs a transient plot"))?;
            let plot = vectors::linearize(&current.plot, grid, points, names, &mut budget)?;
            ControlPlot {
                kind: AnalysisKind::Transient,
                grid: Some(grid),
                plot,
            }
        } else {
            let plot = vectors::fft(&current.plot, names, self.window, &mut budget)?;
            ControlPlot {
                kind: AnalysisKind::Fourier,
                grid: None,
                plot,
            }
        };
        self.remaining_transform_work = budget.work;
        self.retain(result, location)
    }

    fn assign(
        &mut self,
        name: &str,
        expression: &ParameterExpression,
        location: &SourceLoc,
    ) -> SpiceResult<()> {
        if let Some(index) = self.current {
            let plot = &self.plots[index].plot;
            let replaced = plot.variable_index(name);
            let old_cells = replaced.map_or(0, |_| plot.point_count());
            let mut budget = TransformBudget {
                cells: (self.limits.cells - self.cells).saturating_add(old_cells),
                work: self.remaining_transform_work,
            };
            let scope = self.session.scope()?;
            let values = vectors::evaluate_vector(
                plot,
                expression,
                |name| {
                    self.variables
                        .get(name)
                        .copied()
                        .or_else(|| scope.get(name))
                },
                &mut budget,
            )?;
            if let Some(values) = values {
                if matches!(name, "time" | "frequency") {
                    return Err(error(
                        location,
                        "a derived vector cannot replace the physical scale",
                    ));
                }
                let variable = if values.iter().all(|value| value.is_real()) {
                    Variable::new(name, "notype")
                } else {
                    Variable::complex(name, "notype")
                };
                self.plots[index].plot.set_column(variable, &values)?;
                self.cells += values.len() - old_cells;
                self.remaining_transform_work = budget.work;
                self.variables.remove(name);
                return Ok(());
            }
        }
        let value = self.evaluate(expression)?;
        self.bind(name, value, location)
    }

    fn measure(&mut self, card: &MeasureCard) -> SpiceResult<()> {
        let plot = self
            .plots
            .iter()
            .rev()
            .find(|plot| plot.kind == card.analysis)
            .map(|plot| &plot.plot)
            .ok_or_else(|| {
                error(
                    &card.location,
                    format!("no prior .{} plot", card.analysis.as_str()),
                )
            })?;
        let bindings: Vec<_> = self
            .variables
            .iter()
            .map(|(name, value)| ParamBinding {
                name: name.clone(),
                value: *value,
                location: card.location.clone(),
            })
            .collect();
        let scope = ParamScope::resolve(
            Some(std::sync::Arc::new(self.session.scope()?.clone())),
            &bindings,
            &[],
            &mut self.expressions,
        )?;
        let results =
            measure::resolve_with_scope(plot, card.analysis, std::slice::from_ref(card), &scope)?;
        for result in &results {
            self.bind(
                &result.name.to_ascii_lowercase(),
                result.value,
                &card.location,
            )?;
        }
        self.output
            .emit(&measure::to_text(&results), &card.location)
    }

    fn print(&mut self, values: &[String], location: &SourceLoc) -> SpiceResult<()> {
        for name in values {
            let scalar_name = name
                .strip_prefix("$&")
                .or_else(|| name.strip_prefix('$'))
                .unwrap_or(name);
            if let Some(value) = self
                .variables
                .get(scalar_name)
                .copied()
                .or_else(|| self.session.parameter(scalar_name))
            {
                self.output
                    .emit(&format!("{name} = {value:.15e}\n"), location)?;
                continue;
            }
            let qualified = name.split_once('.').and_then(|(prefix, vector)| {
                self.plots
                    .iter()
                    .find(|plot| plot.plot.name.eq_ignore_ascii_case(prefix))
                    .map(|plot| (&plot.plot, vector))
            });
            let (plot, vector) = if let Some(qualified) = qualified {
                qualified
            } else {
                let plot = self
                    .current
                    .and_then(|index| self.plots.get(index))
                    .ok_or_else(|| error(location, format!("no current plot for {name}")))?;
                (&plot.plot, name.as_str())
            };
            let index = plot
                .schema()
                .named(vector)
                .ok_or_else(|| error(location, format!("unknown vector {name}")))?;
            let column = plot
                .column_at(index)
                .ok_or_else(|| error(location, format!("missing vector column {name}")))?;
            for (index, value) in column.iter().enumerate() {
                let text = if plot.flags().is_complex() {
                    format!("{name}[{index}] = {:.15e},{:.15e}\n", value.re, value.im)
                } else {
                    format!("{name}[{index}] = {:.15e}\n", value.re)
                };
                self.output.emit(&text, location)?;
            }
        }
        Ok(())
    }

    fn execute_block(
        &mut self,
        commands: &[Instruction],
    ) -> SpiceResult<std::ops::ControlFlow<()>> {
        use std::ops::ControlFlow;
        for instruction in commands {
            self.step(&instruction.location)?;
            let location = &instruction.location;
            match &instruction.operation {
                Operation::Foreach(name, values, body) => {
                    let values = values
                        .iter()
                        .map(|value| self.evaluate(value))
                        .collect::<SpiceResult<Vec<_>>>()?;
                    for value in values {
                        self.step(location)?;
                        self.bind(name, value, location)?;
                        if self.execute_block(body)?.is_break() {
                            return Ok(ControlFlow::Break(()));
                        }
                    }
                }
                Operation::While(condition, body) => loop {
                    self.step(location)?;
                    if !self.test(condition)? {
                        break;
                    }
                    if self.execute_block(body)?.is_break() {
                        return Ok(ControlFlow::Break(()));
                    }
                },
                Operation::Quit => return Ok(ControlFlow::Break(())),
                operation => self.execute_command(operation, location)?,
            }
        }
        Ok(ControlFlow::Continue(()))
    }

    fn execute_command(&mut self, operation: &Operation, location: &SourceLoc) -> SpiceResult<()> {
        match operation {
            Operation::Let(name, value) => self.assign(name, value, location)?,
            Operation::AlterParameter(name, value) => {
                let value = self.evaluate(value)?;
                self.pending.insert(name.clone(), value);
            }
            Operation::Reset => {
                self.session.alter_parameters(
                    self.pending
                        .iter()
                        .map(|(name, value)| (name.as_str(), *value)),
                )?;
                self.pending.clear();
            }
            Operation::Run => {
                let plans = self
                    .analyses
                    .iter()
                    .map(|card| self.session.plan(card))
                    .collect::<SpiceResult<Vec<_>>>()?;
                self.run_plans(&plans, location)?;
            }
            Operation::Analysis(card) => {
                let plan = self.plan_script_card(card)?;
                self.run_plans(std::slice::from_ref(&plan), location)?;
            }
            Operation::Measure(card) => self.measure(card)?,
            Operation::Print(values) => self.print(values, location)?,
            Operation::Transform(transform, vectors) => {
                self.transform(*transform, vectors, location)?;
            }
            Operation::SetWindow(window) => self.window = *window,
            Operation::SetPlot(name) => self.select_plot(name, location)?,
            Operation::DestroyAll => {
                self.plots.clear();
                self.current = None;
                self.cells = 0;
            }
            Operation::Foreach(..) | Operation::While(..) | Operation::Quit => {
                return Err(error(location, "unexpected control-flow instruction"));
            }
        }
        Ok(())
    }
}

/// Compile and execute a deck's `.control` sections using bounded native commands.
///
/// Includes resolve with the ordinary source admission policy.
/// Unsupported commands are rejected during compilation, before any run.
///
/// # Errors
/// Malformed or unsupported script/deck, exhausted budgets, failed analysis
/// or output I/O. Text already written is not rolled back.
pub fn run_file(
    path: &Path,
    auto_gnd: bool,
    limits: ScriptLimits,
    output: &mut dyn Write,
) -> SpiceResult<ScriptReport> {
    let parsed =
        Parser::with_auto_gnd(auto_gnd).parse_control_file(path, SourceLimits::default())?;
    if !parsed.output.is_empty() || !parsed.measurements.is_empty() || !parsed.fourier.is_empty() {
        return Err(unsupported(
            &parsed.netlist.location,
            "mixed batch output cards in script mode",
        ));
    }
    let program = Program::compile(&parsed.netlist, auto_gnd)?;
    let tunables: Vec<_> = program.tunables.iter().map(String::as_str).collect();
    let analyses = parsed.netlist.analyses.clone();
    let plan = CircuitParameterPlan::new(parsed.netlist, &tunables)?;
    let session = ControlSession::new(&plan, limits.expressions)?;
    let mut state = State {
        session,
        analyses: &analyses,
        limits,
        variables: BTreeMap::new(),
        pending: BTreeMap::new(),
        plots: Vec::new(),
        cells: 0,
        current: None,
        next_plot_id: 1,
        window: SpectrumWindow::default(),
        remaining_transform_work: limits.transform_work,
        remaining_steps: limits.steps,
        remaining_grid: limits.grid_points,
        expressions: EvalBudget::new(limits.expressions),
        output: ScriptOutput {
            remaining: limits.output_bytes,
            writer: output,
        },
    };
    let _ = state.execute_block(&program.instructions)?;
    Ok(ScriptReport {
        variables: state.variables,
        plots: state.plots,
        steps: limits.steps - state.remaining_steps,
    })
}
