//! Borrowed, location-free semantic comparison of parsed decks.
//!
//! Comparison keeps typed values, ordered setters, expression shape, groups
//! and spelling, scope-local card indexes, include-chain depth and resolved
//! include paths. It ignores physical locations, source maps, lexical card
//! formatting, deck paths, include quoting and the raw vector spelling of
//! waveform/IC setters. Output/control cards retain their spelling because
//! root output requests live beside the netlist in the parsed deck.
//!
//! Both equality and diagnostics use one typed traversal. Matching decks
//! allocate nothing; an unsuccessful comparison records only the path to
//! the first differing field, never an owned, normalized AST.

use std::fmt;
use std::path::PathBuf;

use crate::netlist::ast::{
    AnalysisCard, ArgumentExpression, DeviceInstance, FourierCard, FuncCard, FuncParameter,
    FunctionWaveform, GlobalCard, GlobalNode, IncludeDirective, InitialCondition, LibrarySection,
    MeasureCard, MeasureEvent, MeasureRequest, MeasureStatistic, MeasureTransition, MeasureWindow,
    ModelCard, Netlist, NodeHint, NodeHintCard, NodeHintValue, OptionCard, OptionSetting,
    OutputCards, ParamAssignment, ParamCard, ParameterAssignment, ParameterKind, PositionedValue,
    PrintCard, PulseWaveform, PwlPoint, RequestedVector, SaveCard, ScopedCard, ScopedCardKind,
    SourceFunction, SourceWaveform, Subcircuit, VectorRequest,
};
use crate::netlist::bexpr::{
    BBinaryOp, BExpr, BExprKind, BUnaryOp, BehaviouralExpression, TableTransfer,
};
use crate::netlist::card::{CardKind, RawCard};
use crate::netlist::expr::{BinaryOp, Expr, ExprKind, Function, ParameterExpression, UnaryOp};
use crate::netlist::token::{Token, TokenKind};
use crate::primitives::{AnalysisKind, Real};

#[derive(Debug)]
enum Segment {
    Field(&'static str),
    Index(usize),
}

#[derive(Debug)]
enum Reason {
    Value,
    Variant,
    Presence,
    Length { left: usize, right: usize },
}

/// The path is accumulated inside-out only on failure.
#[derive(Debug)]
pub(super) struct Difference {
    path: Vec<Segment>,
    reason: Reason,
}

impl Difference {
    const fn new(reason: Reason) -> Self {
        Self {
            path: Vec::new(),
            reason,
        }
    }

    fn field(mut self, field: &'static str) -> Self {
        self.path.push(Segment::Field(field));
        self
    }

    fn index(mut self, index: usize) -> Self {
        self.path.push(Segment::Index(index));
        self
    }
}

impl fmt::Display for Difference {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (i, part) in self.path.iter().rev().enumerate() {
            match part {
                Segment::Field(field) => {
                    if i != 0 {
                        f.write_str(".")?;
                    }
                    f.write_str(field)?;
                }
                Segment::Index(index) => write!(f, "[{index}]")?,
            }
        }
        if self.path.is_empty() {
            f.write_str("value")?;
        }
        match self.reason {
            Reason::Value => f.write_str(": differs"),
            Reason::Variant => f.write_str(": different variant"),
            Reason::Presence => f.write_str(": present on only one side"),
            Reason::Length { left, right } => {
                write!(f, ": {left} entries on the left, {right} on the right")
            }
        }
    }
}

type Comparison = Result<(), Difference>;

pub(super) trait Semantic {
    fn compare_semantic(&self, other: &Self) -> Comparison;
}

/// Exact equality for fields with no lexical/location metadata. In
/// particular, real values use the same `PartialEq` semantics as the AST
/// (signed zero compares equal and NaN does not).
macro_rules! semantic_leaves {
    ($($ty:ty),+ $(,)?) => {
        $(
            impl Semantic for $ty {
                fn compare_semantic(&self, other: &Self) -> Comparison {
                    if PartialEq::eq(self, other) {
                        Ok(())
                    } else {
                        Err(Difference::new(Reason::Value))
                    }
                }
            }
        )+
    };
}

semantic_leaves!(
    String,
    bool,
    char,
    usize,
    u32,
    Real,
    PathBuf,
    AnalysisKind,
    SourceFunction,
    ScopedCardKind,
    RequestedVector,
    MeasureStatistic,
    MeasureTransition,
    CardKind,
    TokenKind,
    UnaryOp,
    BinaryOp,
    Function,
    BUnaryOp,
    BBinaryOp,
);

const fn compare_lengths(left: usize, right: usize) -> Comparison {
    if left == right {
        Ok(())
    } else {
        Err(Difference::new(Reason::Length { left, right }))
    }
}

impl<T: Semantic> Semantic for Vec<T> {
    fn compare_semantic(&self, other: &Self) -> Comparison {
        compare_lengths(self.len(), other.len())?;
        for (index, (left, right)) in self.iter().zip(other).enumerate() {
            left.compare_semantic(right)
                .map_err(|diff| diff.index(index))?;
        }
        Ok(())
    }
}

impl<T: Semantic> Semantic for Option<T> {
    fn compare_semantic(&self, other: &Self) -> Comparison {
        match (self, other) {
            (Some(left), Some(right)) => left.compare_semantic(right),
            (None, None) => Ok(()),
            _ => Err(Difference::new(Reason::Presence)),
        }
    }
}

impl<T: Semantic> Semantic for Box<T> {
    fn compare_semantic(&self, other: &Self) -> Comparison {
        (**self).compare_semantic(other)
    }
}

impl<T: Semantic, U: Semantic> Semantic for (T, U) {
    fn compare_semantic(&self, other: &Self) -> Comparison {
        self.0
            .compare_semantic(&other.0)
            .map_err(|diff| diff.index(0))?;
        self.1
            .compare_semantic(&other.1)
            .map_err(|diff| diff.index(1))
    }
}

// Exhaustive field patterns make the semantic/lexical classification explicit:
// adding a field to a compared AST struct requires classifying it here.
macro_rules! semantic_struct {
    ($ty:ty { $($field:ident),+ $(,)? } ignore { $($ignored:ident),* $(,)? }) => {
        impl Semantic for $ty {
            fn compare_semantic(&self, other: &Self) -> Comparison {
                let Self { $($field,)+ $($ignored: _,)* } = self;
                $(
                    $field.compare_semantic(&other.$field)
                        .map_err(|diff| diff.field(stringify!($field)))?;
                )+
                Ok(())
            }
        }
    };
}

// Scope order is compared first, including structural terminators and
// indexes, then the typed payload vectors in stable diagnostic order.
semantic_struct!(Netlist { title, cards, devices, models, subcircuits, analyses, includes, params, functions, options, globals, initial_conditions, nodesets } ignore { path, location });
semantic_struct!(Subcircuit { name, terminals, parameters, cards, devices, models, subcircuits, analyses, includes, params, functions, options, globals, initial_conditions, nodesets, output, measurements, fourier } ignore { end_location, location });
semantic_struct!(DeviceInstance { name, designator, nodes, model, parameters } ignore { location });
semantic_struct!(ModelCard { name, base, level, parameters } ignore { location });
semantic_struct!(IncludeDirective { path, resolved_path, section, selected_section } ignore { path_spelling, location });
semantic_struct!(LibrarySection { name } ignore { opening, closing });
semantic_struct!(ParamCard { assignments } ignore { location });
semantic_struct!(ParamAssignment { name, expression } ignore { name_span });
semantic_struct!(FuncCard { name, parameters, body } ignore { name_span, spelling, location });
semantic_struct!(FuncParameter { name } ignore { span });
semantic_struct!(ArgumentExpression { index, expression } ignore {});
semantic_struct!(OptionCard { settings } ignore { location });
semantic_struct!(OptionSetting { name, value, expression } ignore { location });
semantic_struct!(GlobalCard { nodes } ignore { location });
semantic_struct!(GlobalNode { name } ignore { location });
semantic_struct!(NodeHintCard { entries } ignore { location });
semantic_struct!(NodeHint { node, value } ignore { node_location, value_location, location });
semantic_struct!(PositionedValue { text } ignore { location });
semantic_struct!(InitialCondition { name, value } ignore {});
semantic_struct!(PulseWaveform { initial, pulsed, delay, rise, fall, width, period, count } ignore {});
semantic_struct!(FunctionWaveform { function, values } ignore {});
semantic_struct!(PwlPoint { time, value } ignore {});
semantic_struct!(OutputCards { saves, prints } ignore {});
semantic_struct!(SaveCard { requests } ignore { location });
semantic_struct!(PrintCard { ascii_plot, analysis, requests } ignore { analysis_location, location });
semantic_struct!(VectorRequest { vector } ignore { location });
semantic_struct!(MeasureCard { analysis, name, request } ignore { analysis_location, name_location, location });
semantic_struct!(MeasureWindow { from, to } ignore {});
semantic_struct!(ParameterExpression { text, braced, quoted, root } ignore { span });
semantic_struct!(Expr { kind } ignore { span });
semantic_struct!(BehaviouralExpression { text, verbatim, root } ignore { span });
semantic_struct!(BExpr { kind } ignore { span });
semantic_struct!(TableTransfer { input, points, domain } ignore {});
semantic_struct!(RawCard { raw, tokens, kind } ignore { location, source_map });

impl Semantic for Token {
    fn compare_semantic(&self, other: &Self) -> Comparison {
        // Deferred measurement cards retain their token shape and joined
        // offsets, but not the physical source map or token location.
        self.kind
            .compare_semantic(&other.kind)
            .map_err(|diff| diff.field("kind"))?;
        self.text
            .compare_semantic(&other.text)
            .map_err(|diff| diff.field("text"))?;
        self.raw_offset
            .compare_semantic(&other.raw_offset)
            .map_err(|diff| diff.field("raw_offset"))
    }
}

impl Semantic for AnalysisCard {
    fn compare_semantic(&self, other: &Self) -> Comparison {
        let Self {
            kind,
            arguments,
            expressions,
            uic,
            uic_location,
            location: _,
        } = self;
        kind.compare_semantic(&other.kind)
            .map_err(|diff| diff.field("kind"))?;
        arguments
            .compare_semantic(&other.arguments)
            .map_err(|diff| diff.field("arguments"))?;
        expressions
            .compare_semantic(&other.expressions)
            .map_err(|diff| diff.field("expressions"))?;
        uic.compare_semantic(&other.uic)
            .map_err(|diff| diff.field("uic"))?;
        uic_location
            .is_some()
            .compare_semantic(&other.uic_location.is_some())
            .map_err(|diff| diff.field("uic_location"))
    }
}

impl Semantic for FourierCard {
    fn compare_semantic(&self, other: &Self) -> Comparison {
        let Self {
            frontend_command,
            fundamental_expression,
            fundamental,
            fundamental_location: _,
            harmonics,
            harmonics_location,
            vectors,
            location: _,
        } = self;
        frontend_command
            .compare_semantic(&other.frontend_command)
            .map_err(|diff| diff.field("frontend_command"))?;
        fundamental_expression
            .compare_semantic(&other.fundamental_expression)
            .map_err(|diff| diff.field("fundamental_expression"))?;
        fundamental
            .compare_semantic(&other.fundamental)
            .map_err(|diff| diff.field("fundamental"))?;
        harmonics
            .compare_semantic(&other.harmonics)
            .map_err(|diff| diff.field("harmonics"))?;
        harmonics_location
            .is_some()
            .compare_semantic(&other.harmonics_location.is_some())
            .map_err(|diff| diff.field("harmonics_location"))?;
        vectors
            .compare_semantic(&other.vectors)
            .map_err(|diff| diff.field("vectors"))
    }
}

impl Semantic for ParameterAssignment {
    fn compare_semantic(&self, other: &Self) -> Comparison {
        let Self {
            name,
            value,
            kind,
            location: _,
        } = self;
        name.compare_semantic(&other.name)
            .map_err(|diff| diff.field("name"))?;
        kind.compare_semantic(&other.kind)
            .map_err(|diff| diff.field("kind"))?;
        // Only the structured components carry semantics for vectors.
        match kind {
            ParameterKind::Waveform(_) | ParameterKind::InitialConditions(_) => Ok(()),
            ParameterKind::Scalar
            | ParameterKind::Textual
            | ParameterKind::Flag
            | ParameterKind::Instance
            | ParameterKind::Expression(_)
            | ParameterKind::Behavioural(_) => value
                .compare_semantic(&other.value)
                .map_err(|diff| diff.field("value")),
        }
    }
}

impl Semantic for ScopedCard {
    fn compare_semantic(&self, other: &Self) -> Comparison {
        let Self {
            kind,
            source,
            include_chain,
        } = self;
        kind.compare_semantic(&other.kind)
            .map_err(|diff| diff.field("kind"))?;
        match kind {
            ScopedCardKind::Save(_)
            | ScopedCardKind::Print(_)
            | ScopedCardKind::Measure(_)
            | ScopedCardKind::Fourier(_)
            | ScopedCardKind::Control => {
                source
                    .raw
                    .compare_semantic(&other.source.raw)
                    .map_err(|diff| diff.field("source.raw"))?;
            }
            ScopedCardKind::Device(_)
            | ScopedCardKind::Model(_)
            | ScopedCardKind::Subcircuit(_)
            | ScopedCardKind::Analysis(_)
            | ScopedCardKind::Include(_)
            | ScopedCardKind::Options(_)
            | ScopedCardKind::Global(_)
            | ScopedCardKind::Param(_)
            | ScopedCardKind::Func(_)
            | ScopedCardKind::InitialCondition(_)
            | ScopedCardKind::Nodeset(_)
            | ScopedCardKind::End
            | ScopedCardKind::Ends => {}
        }
        compare_lengths(include_chain.len(), other.include_chain.len())
            .map_err(|diff| diff.field("include_chain"))
    }
}

// Exhaustive variant patterns also require callers to classify every new
// variant and field. Tuple and unit variants use the same mismatch path logic.
macro_rules! semantic_enum {
    ($ty:ty {
        $(
            $variant:ident
            $( ( $item:ident : $other_item:ident ) )?
            $( { $($field:ident : $other_field:ident),+ $(; $($ignored:ident),+)? } )?
        ),+ $(,)?
    }) => {
        impl Semantic for $ty {
            fn compare_semantic(&self, other: &Self) -> Comparison {
                match self {
                    $(
                        Self::$variant
                        $(($item))?
                        $({ $($field,)+ $($($ignored: _,)+)? })?
                        => {
                            let Self::$variant
                                $(($other_item))?
                                $({ $($field: $other_field,)+ $($($ignored: _,)+)? })?
                                = other else {
                                    return Err(Difference::new(Reason::Variant).field(stringify!($variant)));
                                };
                            $(
                                $item.compare_semantic($other_item)
                                    .map_err(|diff| diff.field(stringify!($item)).field(stringify!($variant)))?;
                            )?
                            $(
                                $(
                                    $field.compare_semantic($other_field)
                                        .map_err(|diff| diff.field(stringify!($field)).field(stringify!($variant)))?;
                                )+
                            )?
                            Ok(())
                        }
                    ),+
                }
            }
        }
    };
}

semantic_enum!(ExprKind {
    Number { value: rvalue, spelling: rspelling },
    Identifier(name: rname),
    Unary { op: rop, operand: roperand },
    Binary { op: rop, lhs: rlhs, rhs: rrhs },
    Call { function: rfunction, arguments: rarguments },
    Group(inner: rinner),
    UserCall { name: rname, arguments: rarguments },
});

semantic_enum!(BExprKind {
    Number { value: rvalue, spelling: rspelling },
    Name(name: rname),
    Voltage { positive: rpositive, negative: rnegative },
    Current(name: rname),
    Unary { op: rop, operand: roperand },
    Binary { op: rop, lhs: rlhs, rhs: rrhs },
    Ternary { condition: rcondition, then: rthen, otherwise: rotherwise },
    Call { name: rname, arguments: rarguments },
    Group(inner: rinner),
    Value(value: rvalue),
    Table(table: rtable),
});

semantic_enum!(ParameterKind {
    Scalar,
    Textual,
    Flag,
    Instance,
    Expression(expression: rexpression),
    Behavioural(expression: rexpression),
    InitialConditions(components: rcomponents),
    Waveform(waveform: rwaveform),
});

semantic_enum!(SourceWaveform {
    Pulse(pulse: rpulse),
    Function(function: rfunction),
    Pwl(points: rpoints),
});

semantic_enum!(NodeHintValue {
    Literal { text: rtext, value: rvalue },
    Expression(expression: rexpression),
});

semantic_enum!(MeasureRequest {
    Deferred { card: rcard, auto_gnd: rauto_gnd, template: rtemplate },
    Parameter(expression: rexpression),
    AtEvent { operand: roperand, event: revent, derivative: rderivative, window: rwindow },
    When { event: revent, window: rwindow },
    Find { operand: roperand, at: rat, window: rwindow; at_location },
    Statistic { statistic: rstatistic, operand: roperand, window: rwindow },
    TrigTarg { trig: rtrig, targ: rtarg, window: rwindow },
});

semantic_enum!(MeasureEvent {
    Delayed { event: revent, td: rtd },
    At { at: rat; location },
    VectorCrossing { operand: roperand, reference: rreference, transition: rtransition },
    Crossing { operand: roperand, value: rvalue, transition: rtransition; value_location },
});

/// Also used by the writer's expression/tree round-trip checks.
pub(super) fn equal<T: Semantic>(left: &T, right: &T) -> bool {
    left.compare_semantic(right).is_ok()
}

/// True when two decks differ only in location and the documented lexical fields.
#[must_use = "the predicate determines whether the decks have equivalent electrical semantics"]
pub fn semantic_eq(left: &Netlist, right: &Netlist) -> bool {
    equal(left, right)
}

/// Describes the first semantic difference, or `None` when the decks are
/// equal. The path is diagnostic text, not a stable serialization format.
#[must_use = "the returned diagnostic identifies the first semantic mismatch"]
pub fn semantic_diff(left: &Netlist, right: &Netlist) -> Option<String> {
    left.compare_semantic(right)
        .err()
        .map(|diff| diff.to_string())
}
