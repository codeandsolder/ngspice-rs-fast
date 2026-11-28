//! Non-exact differential charge forms. The extra coordinates exist only in
//! the physical derivative system; the ordinary companion retains its own
//! discretization and accepted storage.
use super::{StorageBinding, TaylorContext, TaylorControl, absolute_gradients, negate_odd_vectors};
use crate::devices::linear::LinearSystem;
use crate::maths::{
    SparseMatrix, Vector,
    taylor::{Jet, differential::Differential},
};
use crate::primitives::{Real, SpiceError, SpiceResult};

pub(super) fn augment(
    linear: &LinearSystem,
    bindings: &[StorageBinding],
) -> SpiceResult<Option<LinearSystem>> {
    let count = bindings
        .iter()
        .filter(|binding| binding.coordinate.is_some())
        .count();
    if count == 0 {
        return Ok(None);
    }
    let width = linear
        .a
        .rows()
        .checked_add(count)
        .filter(|&width| width <= 1_048_576)
        .ok_or_else(|| SpiceError::circuit("physical history dimension exceeds budget"))?;
    let mut system = linear.clone();
    for matrix in [&mut system.a, &mut system.e] {
        let old = std::mem::replace(matrix, SparseMatrix::new(width, width));
        for term in old.triplets() {
            matrix.add(term.row, term.col, term.value)?;
        }
    }
    Ok(Some(system))
}

/// Coefficients and two independent partials of M(x,t) xdot.
pub struct RateTaylor {
    /// Residual coefficient of the rate term. The final degree is truncated.
    pub values: Vec<Vector>,
    /// Partial derivative with respect to the state, at fixed state rate.
    pub state_gradients: Vec<SparseMatrix>,
    /// Partial derivative with respect to the physical state rate.
    pub rate_gradients: Vec<SparseMatrix>,
}
impl RateTaylor {
    fn new(width: usize, degrees: usize) -> Self {
        Self {
            values: (0..degrees).map(|_| Vector::zeros(width)).collect(),
            state_gradients: (0..degrees)
                .map(|_| SparseMatrix::new(width, width))
                .collect(),
            rate_gradients: (0..degrees)
                .map(|_| SparseMatrix::new(width, width))
                .collect(),
        }
    }

    pub(super) fn validate(&mut self) -> SpiceResult<()> {
        for matrix in self
            .state_gradients
            .iter_mut()
            .chain(&mut self.rate_gradients)
        {
            matrix.fold_duplicates();
            if matrix
                .triplets()
                .iter()
                .any(|entry| !entry.value.is_finite())
            {
                return Err(SpiceError::circuit("nonfinite physical rate Jacobian"));
            }
        }
        if self.values.iter().any(|vector| !vector.is_finite()) {
            return Err(SpiceError::circuit("nonfinite physical rate residual"));
        }
        Ok(())
    }

    pub(super) fn reverse_odd(&mut self) -> SpiceResult<()> {
        negate_odd_vectors(&mut self.values);
        for matrices in [&mut self.state_gradients, &mut self.rate_gradients] {
            for matrix in matrices.iter_mut().skip(1).step_by(2) {
                let old =
                    std::mem::replace(matrix, SparseMatrix::new(matrix.rows(), matrix.cols()));
                for term in old.triplets() {
                    matrix.add(term.row, term.col, -term.value)?;
                }
            }
        }
        Ok(())
    }
}

fn derivative(jet: &Jet, scale: Real) -> SpiceResult<Jet> {
    let mut values = Vec::with_capacity(jet.order() + 1);
    let mut degree = 1.;
    for &value in &jet.coefficients()[1..] {
        values.push(degree * value / scale);
        degree += 1.;
    }
    // The highest coefficient is not determined by the available jet.
    // The derivative array consumes only degrees strictly below this one.
    values.push(0.);
    Jet::from_coefficients(values)
}

impl TaylorContext<'_> {
    /// qdot = C(x) vdot, with q a device-local independent history state.
    pub(crate) fn path_charge<const N: usize>(
        &mut self,
        slot: usize,
        scale: Real,
        control: TaylorControl,
        coefficient: &Differential<N>,
        controls: &[TaylorControl; N],
    ) -> SpiceResult<()> {
        let absolute = self
            .state_rows
            .start
            .checked_add(slot)
            .ok_or_else(|| SpiceError::circuit("physical path charge slot overflow"))?;
        let binding = self
            .bindings
            .binary_search_by_key(&absolute, |binding| binding.slot)
            .ok()
            .and_then(|i| self.bindings.get(i))
            .filter(|binding| binding.path == Some(control.rows))
            .ok_or_else(|| SpiceError::circuit("undeclared physical path charge"))?;
        let row = binding
            .coordinate
            .ok_or_else(|| SpiceError::circuit("missing path coordinate"))?;
        let charge = self.variable(Some(row))?;
        self.record_charge(slot, &charge, [(row, Jet::constant(self.time.order(), 1.))])?;
        self.linear_charge(Some(row), Some(row), 1.)?;
        if control.rows[0] != control.rows[1] {
            for (terminal, sign) in control.rows.into_iter().zip([1., -1.]) {
                self.linear_charge(terminal, Some(row), sign * scale)?;
            }
        }
        let voltage = self
            .variable(control.rows[0])?
            .add(&self.variable(control.rows[1])?.scale(-1.))
            .scale(control.polarity);
        let speed = if self.time.order() == 0 {
            Jet::constant(0, 0.)
        } else {
            derivative(&voltage, self.time.coefficients()[1])?
        };
        let value = coefficient.value.multiply(&speed).scale(-1.);
        let gradients = absolute_gradients(coefficient, controls);
        let rate = self
            .output
            .rate
            .get_or_insert_with(|| RateTaylor::new(self.state[0].len(), self.state.len()));
        for (vector, &term) in rate.values.iter_mut().zip(value.coefficients()) {
            vector.add_to(row, term)?;
        }
        for (column, partial) in gradients {
            let term = partial.multiply(&speed).scale(-1.);
            for (matrix, &value) in rate.state_gradients.iter_mut().zip(term.coefficients()) {
                matrix.add(row, column, value)?;
            }
        }
        if control.rows[0] != control.rows[1] {
            for (column, sign) in control
                .rows
                .into_iter()
                .zip([-control.polarity, control.polarity])
            {
                if let Some(column) = column {
                    for (matrix, &value) in rate
                        .rate_gradients
                        .iter_mut()
                        .zip(coefficient.value.coefficients())
                    {
                        matrix.add(row, column, sign * value)?;
                    }
                }
            }
        }
        Ok(())
    }
}
