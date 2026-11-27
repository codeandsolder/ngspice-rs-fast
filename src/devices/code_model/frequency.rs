//! Native XSPICE `xfer`: a memoryless DC gain and a tabulated complex AC
//! response, not a causal transient realization. Interpolation is in Hz and
//! Cartesian coordinates, never in magnitude or phase.
use std::sync::Arc;

use super::rational::{PortLaw, transfer_port, transfer_ports};
use super::source::{SourceModel, numeric_scalar, numeric_vector};
use crate::devices::{
    AnalysisMode, BranchQuantity, Device, LinearContext, StampContext,
    models::{PreparationKey, ResolvedModel},
    noise::{DeviceNoise, NoiseContext, NoiseSource},
    setup::MatrixSetup,
};
use crate::maths::Vector;
use crate::netlist::ast::{DeviceInstance, ModelCard, ParameterAssignment};
use crate::primitives::{
    Complex, NodeAppend, NodeId, Real, SpiceError, SpiceResult, exact_nonnegative_integer,
};

fn invalid(message: &str) -> SpiceError {
    SpiceError::circuit(format!("xfer: {message}"))
}

fn integer(parameter: &ParameterAssignment) -> SpiceResult<usize> {
    exact_nonnegative_integer(numeric_scalar(parameter)?)
        .ok_or_else(|| invalid("span/offset must be nonnegative integers"))
}

#[derive(Debug)]
struct Table(Vec<(Real, Complex)>);

impl Table {
    fn compile(card: &ModelCard) -> SpiceResult<Self> {
        SourceModel::FrequencyTransfer.validate_parameters(card)?;
        let mut table = None;
        let mut cartesian = false;
        let mut db = true;
        let mut radians = false;
        let mut span = 3;
        let mut offset = 1;
        for parameter in &card.parameters {
            match parameter.name.as_str() {
                "table" => table = Some(numeric_vector(parameter)?),
                "r_i" => cartesian = numeric_scalar(parameter)? == 1.,
                "db" => db = numeric_scalar(parameter)? == 1.,
                "rad" => radians = numeric_scalar(parameter)? == 1.,
                "span" => span = integer(parameter)?,
                "offset" => offset = integer(parameter)?,
                _ => return Err(invalid("unsupported table parameter")),
            }
        }
        let values = table.ok_or_else(|| {
            invalid("inline table is required (prepare external data before binding)")
        })?;
        if offset < 1
            || span < 3
            || offset.checked_add(2).is_none_or(|end| end > span)
            || values.is_empty()
            || values.len() % span != 0
        {
            return Err(invalid("invalid table span, offset or length"));
        }
        let mut points = Vec::<(Real, Complex)>::with_capacity(values.len() / span);
        for row in values.chunks_exact(span) {
            let frequency = row[0];
            let (a, b) = (row[offset], row[offset + 1]);
            if frequency < 0.
                || points
                    .last()
                    .is_some_and(|(previous, _)| frequency <= *previous)
            {
                return Err(invalid(
                    "frequencies must be nonnegative and strictly increasing",
                ));
            }
            let gain = if cartesian {
                Complex::new(a, b)
            } else {
                let magnitude = if db { 10_f64.powf(a / 20.) } else { a };
                if magnitude < 0. || !magnitude.is_finite() {
                    return Err(invalid("invalid magnitude"));
                }
                let angle = if radians { b } else { b.to_radians() };
                let (sin, cos) = angle.sin_cos();
                Complex::new(magnitude * cos, magnitude * sin)
            };
            if !gain.is_finite() {
                return Err(invalid("nonfinite Cartesian gain"));
            }
            points.push((frequency, gain));
        }
        Ok(Self(points))
    }

    fn dc(&self) -> Real {
        self.0[0].1.re
    }

    fn at(&self, frequency: Real) -> SpiceResult<Complex> {
        if !frequency.is_finite() || frequency < 0. {
            return Err(invalid("invalid analysis frequency"));
        }
        let upper = self.0.partition_point(|(f, _)| *f <= frequency);
        if upper == 0 {
            return Ok(self.0[0].1);
        }
        let (lo, a) = self.0[upper - 1];
        if upper == self.0.len() || frequency == lo {
            return Ok(a);
        }
        let (hi, b) = self.0[upper];
        let t = (frequency - lo) / (hi - lo);
        // Avoid overflow in b-a for opposite-sign large finite ordinates.
        let interpolate = |a: Real, b: Real| {
            if a.is_sign_negative() == b.is_sign_negative() {
                t.mul_add(b - a, a)
            } else {
                (1. - t).mul_add(a, t * b)
            }
        };
        Ok(Complex::new(
            interpolate(a.re, b.re),
            interpolate(a.im, b.im),
        ))
    }
}

#[derive(Debug, Clone)]
struct FrequencyTransfer {
    name: String,
    model_name: String,
    terminals: [NodeId; 4],
    input: PortLaw,
    output: PortLaw,
    table: Arc<Table>,
}

impl FrequencyTransfer {
    fn entries(
        &self,
        branch: usize,
        nodes: [Option<usize>; 4],
        gain: Complex,
        mut emit: impl FnMut(usize, usize, Complex) -> SpiceResult<()>,
    ) -> SpiceResult<()> {
        let inputs = transfer_ports(
            self.input,
            self.output,
            branch,
            branch + 1,
            nodes,
            |row, col, value| {
                if let Some(col) = col {
                    emit(row, col, Complex::real(value))?;
                }
                Ok(())
            },
        )?;
        for (col, sign) in inputs {
            if let Some(col) = col {
                emit(branch, col, Complex::new(sign * gain.re, sign * gain.im))?;
            }
        }
        Ok(())
    }
}

impl Device for FrequencyTransfer {
    fn name(&self) -> &str {
        &self.name
    }
    fn model_name(&self) -> Option<&str> {
        Some(&self.model_name)
    }
    fn designator(&self) -> char {
        'a'
    }
    fn terminals(&self) -> &[NodeId] {
        &self.terminals
    }
    fn branch_currents(&self) -> usize {
        1 + usize::from(self.input == PortLaw::Current)
    }
    fn branch_quantity(&self, branch: usize) -> BranchQuantity {
        if branch == 0 {
            BranchQuantity::Current
        } else {
            BranchQuantity::InputCurrent
        }
    }
    fn findable_branch(&self) -> Option<usize> {
        Some(0)
    }
    fn small_signal_depends_on_frequency(&self) -> bool {
        true
    }
    fn setup_matrix(&self, setup: &mut MatrixSetup<'_>) -> SpiceResult<bool> {
        let branch = setup.branch(0)?;
        let nodes = self.terminals.map(|node| setup.node(node));
        self.entries(branch, nodes, Complex::real(1.), |row, col, _| {
            setup.element(Some(row), Some(col));
            Ok(())
        })?;
        Ok(true)
    }
    fn stamp(&self, context: &mut StampContext<'_>) -> SpiceResult<()> {
        if !matches!(
            context.mode,
            AnalysisMode::OperatingPoint | AnalysisMode::DcSweep
        ) {
            return Err(invalid(
                "only operating point, DC sweep and frequency-domain AC are supported",
            ));
        }
        let branch = context.branch(0)?;
        let nodes = self.terminals.map(|node| context.unknowns.node_row(node));
        self.entries(
            branch,
            nodes,
            Complex::real(self.table.dc()),
            |row, col, value| context.matrix.add(row, col, value.re),
        )
    }
    fn assemble_linear(&self, _context: &mut LinearContext<'_>) -> SpiceResult<()> {
        Err(invalid(
            "transient analysis requires a causal realization, not a frequency table",
        ))
    }
    fn assemble_small_signal(
        &self,
        context: &mut LinearContext<'_>,
        _bias: &Vector,
    ) -> SpiceResult<()> {
        let branch = context
            .branch
            .ok_or_else(|| invalid("missing output branch"))?;
        let nodes = self.terminals.map(|node| context.unknowns.node_row(node));
        let gain = self.table.at(context.model_context.frequency)?;
        self.entries(branch, nodes, gain, |row, col, value| {
            if value.re != 0. {
                context.system.a.add(row, col, value.re)?;
            }
            if value.im != 0. {
                context.dynamic_prescaled(row, col, value.im)?;
            }
            Ok(())
        })
    }
    fn noise(
        &self,
        _context: &NoiseContext<'_>,
        _sources: &mut Vec<NoiseSource>,
    ) -> SpiceResult<DeviceNoise<'_>> {
        Ok(DeviceNoise::Noiseless)
    }
}

pub(super) fn instantiate(
    instance: &DeviceInstance,
    nodes: &mut NodeAppend<'_>,
    model: &ResolvedModel<'_>,
) -> SpiceResult<Box<dyn Device>> {
    let [input, output] = instance.code_ports.as_slice() else {
        return Err(invalid("expected two scalar analog ports"));
    };
    let (input, ip, inn) = transfer_port(input)?;
    let (output, op, on) = transfer_port(output)?;
    let table = model.prepare(PreparationKey::Inputs, || Table::compile(model.card()))?;
    nodes.transaction(|nodes| {
        let terminals = [
            nodes.intern(op)?,
            nodes.intern(on)?,
            nodes.intern(ip)?,
            nodes.intern(inn)?,
        ];
        if (output == PortLaw::Voltage && terminals[0] == terminals[1])
            || (input == PortLaw::Current && terminals[2] == terminals[3])
        {
            return Err(invalid("shorted ideal transfer port"));
        }
        Ok(Box::new(FrequencyTransfer {
            name: instance.name.clone(),
            model_name: model.card().name.clone(),
            terminals,
            input,
            output,
            table,
        }) as Box<dyn Device>)
    })
}
