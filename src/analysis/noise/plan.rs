//! Fixed input/output transfer binding and reused adjoint-solve buffers.
use super::{N_MINGAIN, noise_failure};
use crate::devices::linear::SourceRows;
use crate::devices::noise::{BoundTransfer, squared_gain};
use crate::devices::{Circuit, LinearSystem, SourceKind};
use crate::maths::complex::ComplexLu;
use crate::primitives::{Complex, NodeId, Real, SpiceError, SpiceResult};

/// Unit excitations and the sole numerical solve workspace. Forward gain
/// is consumed before the same result buffer is overwritten by the adjoint.
pub(super) struct NoiseSolve<'a> {
    input: &'a str,
    input_slot: usize,
    input_kind: SourceKind,
    input_rows: SourceRows,
    input_rhs: Vec<Complex>,
    output: BoundTransfer,
    adjoint_rhs: Vec<Complex>,
    scratch: Vec<Complex>,
    solution: Vec<Complex>,
}

impl<'a> NoiseSolve<'a> {
    pub(super) fn bind(
        circuit: &Circuit,
        system: &LinearSystem,
        input: &'a str,
        output: [NodeId; 2],
    ) -> SpiceResult<Self> {
        let (input_slot, source) = system
            .sources
            .iter()
            .enumerate()
            .find(|(_, source)| source.name.eq_ignore_ascii_case(input))
            .ok_or_else(|| {
                SpiceError::circuit(format!("noise input source {input} has no AC row"))
            })?;
        let n = circuit.unknown_count();
        let mut input_rhs = vec![Complex::ZERO; n];
        for &(row, sign) in &source.rows {
            let value = input_rhs
                .get_mut(row)
                .ok_or_else(|| SpiceError::circuit("noise input row out of range"))?;
            *value = *value + Complex::real(sign);
        }
        let output = BoundTransfer::bind(circuit.unknowns(), output)?;
        let mut adjoint_rhs = vec![Complex::ZERO; n];
        let [positive, negative] = output.rows();
        if let Some(row) = positive {
            adjoint_rhs[row] = Complex::real(1.);
        }
        if let Some(row) = negative {
            adjoint_rhs[row] = Complex::real(-1.);
        }
        Ok(Self {
            input,
            input_slot,
            input_kind: source.kind,
            input_rows: source.rows,
            input_rhs,
            output,
            adjoint_rhs,
            scratch: Vec::new(),
            solution: Vec::new(),
        })
    }

    // Reassembly can change numeric parameters but cannot change the
    // topology-bound independent excitation behind this history.
    pub(super) fn validate(&self, system: &LinearSystem) -> SpiceResult<()> {
        if system.sources.get(self.input_slot).is_some_and(|source| {
            source.name.eq_ignore_ascii_case(self.input)
                && source.kind == self.input_kind
                && source.rows == self.input_rows
                && system.a.rows() == self.input_rhs.len()
        }) {
            Ok(())
        } else {
            Err(SpiceError::circuit(
                "bound noise input excitation changed during frequency sweep",
            ))
        }
    }

    pub(super) fn evaluate(&mut self, factor: &ComplexLu) -> SpiceResult<Real> {
        factor.solve_into(
            &self.input_rhs,
            &mut self.scratch,
            &mut self.solution,
            false,
        )?;
        let gain = squared_gain(self.output.difference(&self.solution));
        if !gain.is_finite() {
            return Err(noise_failure("nonfinite input transfer gain"));
        }
        factor.solve_into(
            &self.adjoint_rhs,
            &mut self.scratch,
            &mut self.solution,
            true,
        )?;
        Ok(1. / gain.max(N_MINGAIN))
    }

    pub(super) fn adjoint(&self) -> &[Complex] {
        &self.solution
    }
}
