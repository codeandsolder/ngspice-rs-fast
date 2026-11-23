//! Actual P-device MNA rows and DC loading, matched to cplload.c.
//!
//! C's DC mode uses only the diagonal R entries, even when the model
//! contains mutual series resistance and shunt conductance. This is a
//! reference-compatibility property of the DC equations, NOT a model
//! restriction: transient processing must retain all Maxwell matrices.
//!
//! The C setup clamps each per-unit-length R coefficient to at least 1e-4
//! before the DC load, including a zero-valued resistance. The validated
//! physical plan itself preserves the original Maxwell coefficients.
//! C also loads +0.1*gmin into all four near/far nodal matrix cells.
//! The written near/far return terminals do not enter C's DC stamps.

use super::plan::{CplMatrixKind, CplPlan};
use crate::devices::traits::{AnalysisMode, Device, StampContext};
use crate::maths::Vector;
use crate::primitives::{NodeId, Real, SpiceError, SpiceResult};

/// A physically validated coupled line with the C-compatible DC branch
/// equation layout. Native transient registration awaits modal convolution.
#[derive(Debug, Clone)]
pub struct Cpl {
    name: String,
    model: String,
    terminals: Vec<NodeId>,
    plan: CplPlan,
}

const MIN_REFERENCE_R_PER_UNIT: Real = 1e-4;

impl Cpl {
    /// Bind exactly 2n+2 written terminals and the corresponding immutable
    /// Maxwell matrices. Terminal order is: near signals, near reference,
    /// far signals, far reference.
    ///
    /// # Errors
    /// Invalid terminal count or series resistance overflow.
    pub fn new(
        name: impl Into<String>,
        model: impl Into<String>,
        terminals: Vec<NodeId>,
        plan: CplPlan,
    ) -> SpiceResult<Self> {
        if terminals.len() != 2 * plan.conductors() + 2 {
            return Err(SpiceError::circuit(
                "CPL P terminal count does not match physical model conductor order",
            ));
        }
        for index in 0..plan.conductors() {
            let resistance = plan
                .matrix(CplMatrixKind::Resistance)
                .get(index, index)
                .ok_or_else(|| SpiceError::circuit("missing CPL diagonal series resistance"))?
                .max(MIN_REFERENCE_R_PER_UNIT)
                * plan.length();
            if !resistance.is_finite() {
                return Err(SpiceError::circuit(
                    "CPL length times diagonal series resistance overflows",
                ));
            }
        }
        Ok(Self {
            name: name.into(),
            model: model.into(),
            terminals,
            plan,
        })
    }

    /// Immutable physical model, including mutual matrices and wavefronts.
    #[must_use]
    pub const fn plan(&self) -> &CplPlan {
        &self.plan
    }

    /// Emit the exact C DC MNA entries, in load order, without allocating.
    /// Node and branch rows are resolved once by the caller and ground is
    /// omitted rather than assigned an artificial row.
    fn dc_entries(
        &self,
        first_branch: usize,
        gmin: Real,
        node_row: impl Fn(NodeId) -> Option<usize>,
        mut add: impl FnMut(usize, usize, Real) -> SpiceResult<()>,
    ) -> SpiceResult<()> {
        let count = self.plan.conductors();
        for conductor in 0..count {
            let near = node_row(self.terminals[conductor]);
            let far = node_row(self.terminals[count + 1 + conductor]);
            let a = Some(first_branch + 2 * conductor);
            let b = Some(first_branch + 2 * conductor + 1);
            let resistance = self
                .plan
                .matrix(CplMatrixKind::Resistance)
                .get(conductor, conductor)
                .ok_or_else(|| SpiceError::circuit("missing CPL diagonal resistance"))?
                .max(MIN_REFERENCE_R_PER_UNIT)
                * self.plan.length();
            // CPLload, MODEDC: all four gmin entries have positive sign.
            // The final two branch equations enforce i_near = -i_far
            // and V_near - V_far = R_ii * length * i_near.
            let entries = [
                (near, near, 0.1 * gmin),
                (far, far, 0.1 * gmin),
                (far, near, 0.1 * gmin),
                (near, far, 0.1 * gmin),
                (near, a, 1.),
                (far, b, 1.),
                (a, a, 1.),
                (a, b, 1.),
                (b, near, 1.),
                (b, far, -1.),
                (b, a, -resistance),
            ];
            for (row, col, coefficient) in entries {
                if let (Some(row), Some(col)) = (row, col) {
                    add(row, col, coefficient)?;
                }
            }
        }
        Ok(())
    }

    fn not_yet_ported(&self, mode: &str) -> SpiceError {
        SpiceError::not_yet_ported(
            format!("{mode} of coupled transmission line {}", self.name),
            "src/spicelib/devices/cpl/cplsetup.c, cplload.c, cplinit.c",
        )
    }
}

impl Device for Cpl {
    fn name(&self) -> &str {
        &self.name
    }

    fn model_name(&self) -> Option<&str> {
        Some(&self.model)
    }

    fn designator(&self) -> char {
        'p'
    }

    fn terminals(&self) -> &[NodeId] {
        &self.terminals
    }

    fn branch_currents(&self) -> usize {
        2 * self.plan.conductors()
    }

    fn setup_matrix(
        &self,
        setup: &mut crate::devices::setup::MatrixSetup<'_>,
    ) -> SpiceResult<bool> {
        let unknowns = setup.unknowns;
        let first = setup.branch(0)?;
        self.dc_entries(
            first,
            0.,
            |node| unknowns.node_row(node),
            |row, col, _| {
                setup.element(Some(row), Some(col));
                Ok(())
            },
        )?;
        Ok(true)
    }

    fn stamp(&self, context: &mut StampContext<'_>) -> SpiceResult<()> {
        if !context.mode.is_dc() {
            return Err(self.not_yet_ported(match context.mode {
                AnalysisMode::Ac { .. } => "AC",
                AnalysisMode::Transient { .. } => "transient",
                AnalysisMode::OperatingPoint | AnalysisMode::DcSweep => "DC",
            }));
        }
        let first = context.branch(0)?;
        let unknowns = context.unknowns;
        self.dc_entries(
            first,
            context.gmin,
            |node| unknowns.node_row(node),
            |row, col, coefficient| context.matrix.add(row, col, coefficient),
        )
    }

    fn assemble_linear(
        &self,
        context: &mut crate::devices::linear::LinearContext<'_>,
    ) -> SpiceResult<()> {
        let first = context
            .branch
            .ok_or_else(|| SpiceError::circuit("missing CPL MNA branch rows"))?;
        let unknowns = context.unknowns;
        self.dc_entries(
            first,
            context.model_context.gmin,
            |node| unknowns.node_row(node),
            |row, col, coefficient| context.system.a.add(row, col, coefficient),
        )
    }

    fn assemble_small_signal(
        &self,
        _context: &mut crate::devices::linear::LinearContext<'_>,
        _bias: &Vector,
    ) -> SpiceResult<()> {
        Err(self.not_yet_ported("AC/SP"))
    }

    fn assemble_pole_zero(
        &self,
        _context: &mut crate::devices::linear::LinearContext<'_>,
        _bias: &Vector,
    ) -> SpiceResult<()> {
        Err(self.not_yet_ported("pole-zero"))
    }
}
