//! Contiguous homogeneous model tiles with borrowed inputs and prepared data.
//! Family kernels may override the default scalar traversal with portable SIMD.

use crate::devices::prepared_model::{Capability, Kernel, Point, PointBuffer};
use crate::primitives::{Real, SpiceError, SpiceResult};

/// Persistent output buffers for one homogeneous model-family tile.
#[derive(Debug)]
pub struct BatchOutput {
    width: usize,
    lanes: usize,
    /// Terminal current per lane, then terminal.
    pub current: Vec<Real>,
    /// Terminal conductance per lane, row and column.
    pub conductance: Vec<Real>,
    /// Stored terminal charge per lane.
    pub charge: Vec<Real>,
    /// Charge Jacobian per lane, row and column.
    pub capacitance: Vec<Real>,
}

impl BatchOutput {
    /// Allocate one reusable output tile.
    ///
    /// # Errors
    /// Invalid shape or size overflow.
    pub fn new(width: usize, lanes: usize) -> SpiceResult<Self> {
        if width < 2 || lanes == 0 {
            return Err(SpiceError::circuit("invalid prepared tile dimensions"));
        }
        let terminals = width
            .checked_mul(lanes)
            .ok_or_else(|| SpiceError::circuit("prepared tile terminal count overflow"))?;
        let entries = terminals
            .checked_mul(width)
            .ok_or_else(|| SpiceError::circuit("prepared tile Jacobian count overflow"))?;
        let allocate = |count: usize| -> SpiceResult<Vec<Real>> {
            let mut values = Vec::new();
            values.try_reserve_exact(count).map_err(|error| {
                SpiceError::circuit(format!("cannot allocate prepared tile: {error}"))
            })?;
            values.resize(count, 0.);
            Ok(values)
        };
        Ok(Self {
            width,
            lanes,
            current: allocate(terminals)?,
            conductance: allocate(entries)?,
            charge: allocate(terminals)?,
            capacitance: allocate(entries)?,
        })
    }

    pub(crate) fn point(&mut self, lane: usize) -> SpiceResult<Point<'_>> {
        if lane >= self.lanes {
            return Err(SpiceError::circuit("prepared tile lane out of range"));
        }
        let from = lane * self.width;
        let to = from + self.width;
        let first = lane * self.width * self.width;
        let last = first + self.width * self.width;
        Ok(Point {
            current: &mut self.current[from..to],
            conductance: &mut self.conductance[first..last],
            charge: &mut self.charge[from..to],
            capacitance: &mut self.capacitance[first..last],
        })
    }

    /// Clear every output to a missing sentinel or a declared physical zero.
    pub fn reset(&mut self, charge: Capability) {
        self.current.fill(Real::NAN);
        self.conductance.fill(Real::NAN);
        let value = if charge == Capability::Zero {
            0.
        } else {
            Real::NAN
        };
        self.charge.fill(value);
        self.capacitance.fill(value);
    }

    /// Validate every lane without retaining intermediate kernel objects.
    ///
    /// # Errors
    /// Missing or nonconservative physical outputs.
    pub fn validate(&mut self, charge: Capability) -> SpiceResult<()> {
        for lane in 0..self.lanes {
            let width = self.width;
            self.point(lane)?.validate(width, charge)?;
        }
        Ok(())
    }

    /// Number of evaluated model instances.
    #[must_use]
    pub const fn lanes(&self) -> usize {
        self.lanes
    }
    /// Terminal count per instance.
    #[must_use]
    pub const fn width(&self) -> usize {
        self.width
    }
}

/// Borrowed homogeneous tile passed to statically dispatched family kernels.
pub struct BatchInput<'a, K: Kernel> {
    /// Immutable derived model/instance data in lane order.
    pub prepared: &'a [K::Prepared],
    /// Terminal voltages in lane-major order.
    pub voltages: &'a [Real],
    /// Validated physical channel capability.
    pub charge: Capability,
}

/// Complete a homogeneous tile without heap allocation in the hot path.
///
/// A specialized family can implement this call with portable SIMD, using the
/// same contiguous lane-major buffers. The default evaluates independent
/// scalar lanes and preserves their exact floating-point operation order.
///
/// # Errors
/// Shape/capability mismatch, invalid physics, missing or nonconservative data.
pub fn evaluate_tile<K: Kernel>(
    kernel: &K,
    input: &BatchInput<'_, K>,
    output: &mut BatchOutput,
) -> SpiceResult<()> {
    if input.prepared.len() != output.lanes || input.voltages.len() != output.width * output.lanes {
        return Err(SpiceError::circuit("prepared tile input shape mismatch"));
    }
    output.reset(input.charge);
    kernel.evaluate_batch(input.prepared, input.voltages, output)?;
    output.validate(input.charge)
}

/// Default scalar physical evaluation for a homogeneous prepared tile.
/// Model families may specialize their own `Kernel::evaluate_batch` without
/// changing the solver's borrowed-buffer contract.
///
/// # Errors
/// Invalid kernel physics or a mismatched lane input shape.
pub fn evaluate_scalar_tile<K: Kernel + ?Sized>(
    kernel: &K,
    prepared: &[K::Prepared],
    voltages: &[Real],
    output: &mut BatchOutput,
) -> SpiceResult<()> {
    if prepared.len() != output.lanes || voltages.len() != output.width * output.lanes {
        return Err(SpiceError::circuit("prepared scalar tile shape mismatch"));
    }
    for (lane, instance) in prepared.iter().enumerate() {
        let width = output.width;
        let voltage = &voltages[lane * width..(lane + 1) * width];
        kernel.evaluate(instance, voltage, &mut output.point(lane)?)?;
    }
    Ok(())
}

/// Single-lane bridge to validate a family point using a reusable workspace.
/// This is useful for conformance without creating temporary tile allocations.
///
/// # Errors
/// Kernel physics or nonconservative/missing outputs.
pub fn evaluate_one<K: Kernel>(
    kernel: &K,
    prepared: &K::Prepared,
    voltages: &[Real],
    charge: Capability,
    output: &mut PointBuffer,
) -> SpiceResult<()> {
    output.reset(charge);
    kernel.evaluate(prepared, voltages, &mut output.point())?;
    output.point().validate(voltages.len(), charge)
}
