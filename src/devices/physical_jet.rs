//! Physical F/Q Taylor stamps for consistent right-limit initialization.
use super::{ModelContext, traits::MnaUnknowns};
use crate::maths::{SparseMatrix, Vector, taylor::Taylor};
use crate::primitives::{Real, SpiceError, SpiceResult};

pub(crate) struct ResidualJet {
    pub(crate) values: Vec<Vector>,
    pub(crate) jacobians: Vec<SparseMatrix>,
}

impl ResidualJet {
    pub(crate) fn new(size: usize, order: usize) -> Self {
        Self {
            values: (0..=order).map(|_| Vector::zeros(size)).collect(),
            jacobians: (0..=order).map(|_| SparseMatrix::new(size, size)).collect(),
        }
    }

    pub(crate) fn value(
        &mut self,
        row: Option<usize>,
        value: &Taylor,
        weight: Real,
    ) -> SpiceResult<()> {
        if let Some(row) = row {
            for (target, coefficient) in self.values.iter_mut().zip(value.coefficients()) {
                target.add_to(row, weight * coefficient)?;
            }
        }
        Ok(())
    }

    pub(crate) fn derivative(
        &mut self,
        row: Option<usize>,
        column: Option<usize>,
        value: &Taylor,
        weight: Real,
    ) -> SpiceResult<()> {
        if let (Some(row), Some(column)) = (row, column) {
            for (target, coefficient) in self.jacobians.iter_mut().zip(value.coefficients()) {
                if *coefficient != 0. {
                    target.add(row, column, weight * coefficient)?;
                }
            }
        }
        Ok(())
    }

    pub(crate) fn linear(
        &mut self,
        row: Option<usize>,
        column: Option<usize>,
        coordinate: &Taylor,
        weight: Real,
    ) -> SpiceResult<()> {
        self.value(row, coordinate, weight)?;
        if let (Some(row), Some(column)) = (row, column) {
            if weight != 0. {
                self.jacobians[0].add(row, column, weight)?;
            }
        }
        Ok(())
    }

    pub(crate) fn finish(&mut self) {
        for jacobian in &mut self.jacobians {
            jacobian.fold_duplicates();
        }
    }
}

pub(crate) struct PhysicalJet {
    pub(crate) static_part: ResidualJet,
    pub(crate) charge: ResidualJet,
}

pub(crate) struct Binding<'a> {
    pub(crate) coordinates: &'a [Taylor],
    pub(crate) time: &'a Taylor,
    pub(crate) model: &'a ModelContext,
    pub(crate) unknowns: &'a MnaUnknowns,
    pub(crate) branch: Option<usize>,
    pub(crate) controls: &'a [usize],
    pub(crate) budget: &'a mut usize,
}

impl Binding<'_> {
    pub(crate) fn coordinate(&self, column: Option<usize>) -> SpiceResult<Taylor> {
        match column {
            Some(column) => self
                .coordinates
                .get(column)
                .cloned()
                .ok_or_else(|| SpiceError::circuit("unbound Taylor coordinate")),
            None => Ok(Taylor::constant(0., self.time.order())),
        }
    }
}
