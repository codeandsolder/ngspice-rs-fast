//! Solver-owned integration of a signed storage residual. The expression
//! compiler supplies its value and Jacobian, never a mutable time history.

use crate::devices::nonlinear::charge::ChargeBranch;
use crate::devices::traits::{AnalysisMode, StampContext};
use crate::primitives::{Real, SpiceError, SpiceResult};

/// Integrated storage residual and its derivative multiplier for the current
/// trial. A rejected trial leaves accepted history unchanged.
pub(super) fn integrate(
    context: &mut StampContext<'_>,
    quantity: Real,
) -> SpiceResult<(Real, Real)> {
    let storage = ChargeBranch::new(0, quantity, 1.);
    let result = if let AnalysisMode::Transient { dt, .. } = context.mode {
        let coefficients = context.integration.ok_or_else(|| {
            SpiceError::circuit("dynamic residual requires integration coefficients")
        })?;
        if coefficients.dt().to_bits() != dt.to_bits() {
            return Err(SpiceError::circuit("dynamic residual timestep mismatch"));
        }
        let companion = storage.integrate(&mut context.states, coefficients)?;
        (companion.derivative, coefficients.ag()[0])
    } else {
        storage.zero_derivative(&mut context.states)?;
        (0., 0.)
    };
    storage.store(&mut context.states)?;
    Ok(result)
}
