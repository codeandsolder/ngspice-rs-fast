//! Causal finite-volume multiconductor RLGC transient approximation.
//!
//! Each spatial section is a symmetric π network: half shunt G/C,
//! one coupled series R/L, and a second half shunt G/C. Each section owns
//! charge and flux history; the accepted-history rule is exactly the same
//! as scalar capacitors and mutually coupled inductors. The spatial
//! approximation is a passive rational transmission-line network, rather
//! than sampled frequency-domain data or an instantaneous two-port.
//!
//! Internal nodal voltages and series currents are condensed into an affine
//! endpoint transfer. The external MNA contains only one branch-current row
//! per signal conductor; no additional circuit node or dense global block
//! is introduced by the spatial discretization.
use super::{CoupledLine, invalid};
use crate::devices::traits::{AnalysisMode, StampContext};
use crate::maths::{Coefficients, dense_complex::DenseComplex, integrator::ChargeHistory};
use crate::primitives::{Real, SpiceResult};

/// Six slots per section/conductor: q and dq of the left half-shunt C,
/// flux and dflux of the coupled series L, and q and dq of the right C.
const SLOTS_PER_SECTION_CONDUCTOR: usize = 6;
const LEFT: usize = 0;
const SERIES: usize = 2;
const RIGHT: usize = 4;

/// Real dense square matrix for the two-ended affine state propagator.
#[derive(Debug, Clone)]
struct Square {
    dimension: usize,
    entries: Vec<Real>,
}
impl Square {
    fn zeros(dimension: usize) -> SpiceResult<Self> {
        let elements = dimension
            .checked_mul(dimension)
            .ok_or_else(|| invalid("transient transfer square dimension overflow"))?;
        let mut entries = Vec::new();
        entries
            .try_reserve_exact(elements)
            .map_err(|_| invalid("transient matrix storage"))?;
        entries.resize(elements, 0.);
        Ok(Self { dimension, entries })
    }
    fn at(&self, row: usize, col: usize) -> Real {
        self.entries[row * self.dimension + col]
    }
    fn set(&mut self, row: usize, col: usize, value: Real) {
        self.entries[row * self.dimension + col] = value;
    }
    fn mul(&self, other: &Self) -> SpiceResult<Self> {
        if self.dimension != other.dimension {
            return Err(invalid("incompatible transient transfer dimensions"));
        }
        let size = self.dimension;
        let mut product = Self::zeros(size)?;
        for row in 0..size {
            for k in 0..size {
                let factor = self.at(row, k);
                if factor == 0. {
                    continue;
                }
                for col in 0..size {
                    let slot = &mut product.entries[row * size + col];
                    *slot = factor.mul_add(other.at(k, col), *slot);
                }
            }
        }
        if product.entries.iter().any(|value| !value.is_finite()) {
            return Err(invalid("transient transfer matrix overflow"));
        }
        Ok(product)
    }
    fn times(&self, vector: &[Real]) -> SpiceResult<Vec<Real>> {
        if self.dimension != vector.len() {
            return Err(invalid("transient matrix-vector dimension mismatch"));
        }
        let mut result = Vec::new();
        result
            .try_reserve_exact(vector.len())
            .map_err(|_| invalid("transient transfer vector allocation"))?;
        for row in 0..self.dimension {
            let mut total = 0.;
            for (col, &value) in vector.iter().enumerate() {
                total = self.at(row, col).mul_add(value, total);
            }
            if !total.is_finite() {
                return Err(invalid("transient transfer vector overflow"));
            }
            result.push(total);
        }
        Ok(result)
    }
}

/// One coupled π section at the current integration coefficients.
struct Cell {
    conductors: usize,
    scale: Real,
    series: Square,
    half_shunt: Square,
}
impl Cell {
    fn new(line: &CoupledLine, ag0: Real) -> SpiceResult<Self> {
        let conductors = line.conductors();
        let sections = line.sections;
        let length = line.plan.length
            / f64::from(
                u32::try_from(sections).map_err(|_| invalid("spatial section count overflow"))?,
            );
        if !ag0.is_finite() || ag0 <= 0. || !length.is_finite() || length <= 0. {
            return Err(invalid("invalid time-integration or spatial section width"));
        }
        let mut series = Square::zeros(conductors)?;
        let mut half_shunt = Square::zeros(conductors)?;
        let (mut series_norm, mut shunt_norm) = (0.0_f64, 0.0_f64);
        for row in 0..conductors {
            let (mut zr, mut yr) = (0., 0.);
            for col in 0..conductors {
                let idx = row * conductors + col;
                let z = length * ag0.mul_add(line.plan.l.data()[idx], line.plan.r.data()[idx]);
                let y =
                    0.5 * length * ag0.mul_add(line.plan.c.data()[idx], line.plan.g.data()[idx]);
                if !z.is_finite() || !y.is_finite() {
                    return Err(invalid("nonfinite transient RLGC companion coefficient"));
                }
                series.set(row, col, z);
                half_shunt.set(row, col, y);
                zr += z.abs();
                yr = 2.0_f64.mul_add(y.abs(), yr);
            }
            series_norm = series_norm.max(zr);
            shunt_norm = shunt_norm.max(yr);
        }
        // The same similarity change of current units used by frequency
        // stamping, here evaluated at the real integration coefficient ag0.
        let scale = if series_norm > 0. && shunt_norm > 0. {
            (series_norm / shunt_norm).sqrt()
        } else {
            1.
        };
        if !scale.is_finite() || scale <= 0. {
            return Err(invalid("cannot balance real transient cable impedance"));
        }
        for value in &mut series.entries {
            *value /= scale;
        }
        for value in &mut half_shunt.entries {
            *value *= scale;
        }
        Ok(Self {
            conductors,
            scale,
            series,
            half_shunt,
        })
    }
}

/// Apply a numerically real DC complex transfer to a real voltage/current
/// state. Any material imaginary leakage above roundoff is rejected.
fn apply_complex_real(matrix: &DenseComplex, state: &[Real]) -> SpiceResult<Vec<Real>> {
    if matrix.dim() != state.len() {
        return Err(invalid("DC section transfer size mismatch"));
    }
    let mut result = Vec::new();
    result
        .try_reserve_exact(state.len())
        .map_err(|_| invalid("DC transfer scratch"))?;
    for row in 0..state.len() {
        let mut value = 0.;
        for (col, &input) in state.iter().enumerate() {
            let factor = matrix
                .get(row, col)
                .ok_or_else(|| invalid("DC transfer index"))?;
            if factor.im.abs() > 1e-12 * factor.re.abs().max(1e-200) {
                return Err(invalid("DC line material yields imaginary propagation"));
            }
            value = factor.re.mul_add(input, value);
        }
        if !value.is_finite() {
            return Err(invalid("DC internal section state overflow"));
        }
        result.push(value);
    }
    Ok(result)
}

const fn slot(section: usize, conductor: usize, kind: usize, conductors: usize) -> usize {
    (section * conductors + conductor) * SLOTS_PER_SECTION_CONDUCTOR + kind
}
fn history_term(
    states: &crate::devices::DeviceState<'_>,
    coefficient: &Coefficients,
    slot: usize,
) -> SpiceResult<Real> {
    let mut history = ChargeHistory::new(0.);
    for age in 1..=coefficient.charge_history_len() {
        history.push(states.accepted(age, slot).ok_or_else(|| {
            invalid("missing accepted spatial charge/flux for transient integration")
        })?)?;
    }
    let previous = if coefficient.needs_previous_derivative() {
        Some(
            states
                .accepted(1, slot + 1)
                .ok_or_else(|| invalid("missing accepted spatial current/voltage derivative"))?,
        )
    } else {
        None
    };
    Ok(coefficient.integrate(&history, previous, 0.)?.current)
}
fn write_energy(
    states: &mut crate::devices::DeviceState<'_>,
    slot: usize,
    value: Real,
    history: Real,
    ag0: Real,
) -> SpiceResult<()> {
    states.set(slot, value)?;
    states.set(slot + 1, ag0.mul_add(value, history))
}

impl CoupledLine {
    /// Total charge/flux state slots for all spatial sections.
    #[must_use]
    pub(super) fn spatial_state_count(&self) -> usize {
        self.sections * self.conductors() * SLOTS_PER_SECTION_CONDUCTOR
    }

    /// Initialize steady state from the exact continuous DC transfer.
    /// Both section half-shunt charges and series fluxes are populated,
    /// including the correct mutually coupled per-unit-length matrices.
    pub(super) fn dc_spatial_state(&self, context: &mut StampContext<'_>) -> SpiceResult<()> {
        // Ordinary DC analyses may load without companion history, just as
        // capacitors and inductors do. A transient seed has bound slots.
        if context.states.is_empty() {
            return Ok(());
        }
        let count = self.conductors();
        if context.states.len() != self.spatial_state_count() {
            return Err(invalid("DC spatial-line state layout mismatch"));
        }
        if self.plan.requires_norton(0.)? {
            return self.dc_stable_state(context);
        }
        let (half, scale) = self.plan.dc_half_section(self.sections)?;
        let mut state = Vec::new();
        state
            .try_reserve_exact(2 * count)
            .map_err(|_| invalid("DC spatial input state"))?;
        for conductor in 0..count {
            let (positive, negative) = self.physical_pair(conductor);
            state.push(context.node_voltage(positive) - context.node_voltage(negative));
        }
        for conductor in 0..count {
            state.push(context.row_value(context.branch(conductor)?)?);
        }
        let width = self.plan.length
            / f64::from(u32::try_from(self.sections).map_err(|_| invalid("section count"))?);
        for section in 0..self.sections {
            let midpoint = apply_complex_real(&half, &state)?;
            let right = apply_complex_real(&half, &midpoint)?;
            for conductor in 0..count {
                let mut left_q = 0.;
                let mut series_flux = 0.;
                let mut right_q = 0.;
                for col in 0..count {
                    let index = conductor * count + col;
                    let capacitance = self.plan.c.data()[index] * width * 0.5;
                    let inductance = self.plan.l.data()[index] * width;
                    left_q = capacitance.mul_add(state[col], left_q);
                    right_q = capacitance.mul_add(right[col], right_q);
                    series_flux = inductance.mul_add(midpoint[count + col] / scale, series_flux);
                }
                for (kind, value) in [(LEFT, left_q), (SERIES, series_flux), (RIGHT, right_q)] {
                    let index = slot(section, conductor, kind, count);
                    context.states.set(index, value)?;
                    context.states.set(index + 1, 0.)?;
                }
            }
            state = right;
        }
        Ok(())
    }
}

// A positive-real Schur complement of adjacent passive spatial cells.
// This representation never constructs the exponentially growing long-line
// ABCD transfer; interior voltages are eliminated locally and stably.
impl Square {
    fn add(&self, other: &Self) -> SpiceResult<Self> {
        if self.dimension != other.dimension {
            return Err(invalid("Norton dimension mismatch"));
        }
        let mut result = Self::zeros(self.dimension)?;
        for ((target, &a), &b) in result
            .entries
            .iter_mut()
            .zip(&self.entries)
            .zip(&other.entries)
        {
            *target = a + b;
        }
        if result.entries.iter().any(|v| !v.is_finite()) {
            return Err(invalid("nonfinite Norton matrix sum"));
        }
        Ok(result)
    }
    fn sub(&self, other: &Self) -> SpiceResult<Self> {
        if self.dimension != other.dimension {
            return Err(invalid("Norton dimension mismatch"));
        }
        let mut result = Self::zeros(self.dimension)?;
        for ((target, &a), &b) in result
            .entries
            .iter_mut()
            .zip(&self.entries)
            .zip(&other.entries)
        {
            *target = a - b;
        }
        if result.entries.iter().any(|v| !v.is_finite()) {
            return Err(invalid("nonfinite Norton matrix difference"));
        }
        Ok(result)
    }
    fn negative(&self) -> Self {
        Self {
            dimension: self.dimension,
            entries: self.entries.iter().map(|v| -v).collect(),
        }
    }
    fn inverse(&self) -> SpiceResult<Self> {
        let count = self.dimension;
        let mut dense = crate::maths::Matrix::zeros(count, count)?;
        dense.data_mut().copy_from_slice(&self.entries);
        let lu = dense.lu_decompose()?;
        let mut inverse = Self::zeros(count)?;
        for col in 0..count {
            let mut unit = crate::maths::Vector::zeros(count);
            unit.set(col, 1.)?;
            let column = lu.solve(&unit)?;
            for row in 0..count {
                inverse.set(row, col, column.as_slice()[row]);
            }
        }
        if inverse.entries.iter().any(|v| !v.is_finite()) {
            return Err(invalid("nonfinite Norton inverse"));
        }
        Ok(inverse)
    }
    fn solve(&self, rhs: &[Real]) -> SpiceResult<Vec<Real>> {
        let mut dense = crate::maths::Matrix::zeros(self.dimension, self.dimension)?;
        dense.data_mut().copy_from_slice(&self.entries);
        Ok(dense
            .solve(&crate::maths::Vector::from(rhs.to_vec()))?
            .as_slice()
            .to_vec())
    }
}
fn sum_vectors(a: &[Real], b: &[Real]) -> SpiceResult<Vec<Real>> {
    if a.len() != b.len() {
        return Err(invalid("Norton vector size mismatch"));
    }
    let output = a.iter().zip(b).map(|(&x, &y)| x + y).collect::<Vec<_>>();
    if output.iter().any(|v| !v.is_finite()) {
        return Err(invalid("nonfinite Norton source sum"));
    }
    Ok(output)
}
fn diff_vectors(a: &[Real], b: &[Real]) -> SpiceResult<Vec<Real>> {
    if a.len() != b.len() {
        return Err(invalid("Norton vector size mismatch"));
    }
    let output = a.iter().zip(b).map(|(&x, &y)| x - y).collect::<Vec<_>>();
    if output.iter().any(|v| !v.is_finite()) {
        return Err(invalid("nonfinite Norton source difference"));
    }
    Ok(output)
}

#[derive(Debug, Clone)]
struct Norton {
    y11: Square,
    y12: Square,
    y21: Square,
    y22: Square,
    j1: Vec<Real>,
    j2: Vec<Real>,
}
impl Norton {
    fn cascade(&self, other: &Self) -> SpiceResult<Self> {
        // Shared interface K=Y22_left+Y11_right is SPD for a passive
        // well-posed spatial network under positive integration ag0.
        let inverse = self.y22.add(&other.y11)?.inverse()?;
        let left = self.y12.mul(&inverse)?;
        let right = other.y21.mul(&inverse)?;
        let forcing = sum_vectors(&self.j2, &other.j1)?;
        Ok(Self {
            y11: self.y11.sub(&left.mul(&self.y21)?)?,
            y12: left.mul(&other.y12)?.negative(),
            y21: right.mul(&self.y21)?.negative(),
            y22: other.y22.sub(&right.mul(&other.y12)?)?,
            j1: diff_vectors(&self.j1, &left.times(&forcing)?)?,
            j2: diff_vectors(&other.j2, &right.times(&forcing)?)?,
        })
    }
    fn interface(&self, tail: &Self, left: &[Real], right: &[Real]) -> SpiceResult<Vec<Real>> {
        let node = self.y22.add(&tail.y11)?;
        let source = sum_vectors(
            &sum_vectors(&self.y21.times(left)?, &tail.y12.times(right)?)?,
            &sum_vectors(&self.j2, &tail.j1)?,
        )?;
        node.solve(&source.iter().map(|v| -v).collect::<Vec<_>>())
    }
}
impl Cell {
    // A symmetric passive pi cell: Ileft=(S+H)Vleft-S Vright+Hleft-S Hseries
    // and Iright=-S Vleft+(S+H)Vright+Hright+S Hseries.
    // The physical series impedance is Zscaled * Zscale; Yhalf is Yscaled/Zscale.
    fn physical_norton(
        &self,
        series_inverse: &Square,
        hleft: &[Real],
        hseries: &[Real],
        hright: &[Real],
    ) -> SpiceResult<Norton> {
        let mut half = Square::zeros(self.conductors)?;
        for row in 0..self.conductors {
            for col in 0..self.conductors {
                half.set(row, col, self.half_shunt.at(row, col) / self.scale);
            }
        }
        let series_source = series_inverse.times(hseries)?;
        let diagonal = series_inverse.add(&half)?;
        Ok(Norton {
            y11: diagonal.clone(),
            y12: series_inverse.negative(),
            y21: series_inverse.negative(),
            y22: diagonal,
            j1: diff_vectors(hleft, &series_source)?,
            j2: sum_vectors(hright, &series_source)?,
        })
    }
    fn physical_series_inverse(&self) -> SpiceResult<Square> {
        let mut impedance = Square::zeros(self.conductors)?;
        for row in 0..self.conductors {
            for col in 0..self.conductors {
                impedance.set(row, col, self.series.at(row, col) * self.scale);
            }
        }
        impedance.inverse()
    }
}

impl CoupledLine {
    /// Passive, causal transient Norton load and all spatial trial states.
    /// No exponentially growing transfer-matrix product is formed.
    pub(super) fn stamp_stable_transient(
        &self,
        ctx: &mut StampContext<'_>,
        coeff: &Coefficients,
    ) -> SpiceResult<()> {
        let AnalysisMode::Transient { dt, .. } = ctx.mode else {
            return Err(invalid("spatial Norton load outside transient"));
        };
        if coeff.dt().total_cmp(&dt) != std::cmp::Ordering::Equal
            || ctx.states.len() != self.spatial_state_count()
        {
            return Err(invalid("spatial Norton timestep or history mismatch"));
        }
        if self.plan.r.data().iter().all(|&entry| entry == 0.)
            && self.plan.l.data().iter().all(|&entry| entry == 0.)
        {
            return self.stamp_ideal_series_transient(ctx, coeff);
        }
        let conductors = self.conductors();
        let cell = Cell::new(self, coeff.ag()[0])?;
        let inverse = cell.physical_series_inverse()?;
        let mut cells = Vec::new();
        cells
            .try_reserve_exact(self.sections)
            .map_err(|_| invalid("spatial Norton cell storage"))?;
        for section in 0..self.sections {
            let history = |kind: usize| -> SpiceResult<Vec<Real>> {
                (0..conductors)
                    .map(|i| history_term(&ctx.states, coeff, slot(section, i, kind, conductors)))
                    .collect()
            };
            cells.push(cell.physical_norton(
                &inverse,
                &history(LEFT)?,
                &history(SERIES)?,
                &history(RIGHT)?,
            )?);
        }
        let mut suffix = Vec::new();
        suffix
            .try_reserve_exact(self.sections)
            .map_err(|_| invalid("spatial Norton suffix storage"))?;
        suffix.push(
            cells
                .last()
                .ok_or_else(|| invalid("no spatial cells"))?
                .clone(),
        );
        for index in (0..self.sections - 1).rev() {
            let previous = suffix
                .last()
                .ok_or_else(|| invalid("empty Norton suffix"))?;
            suffix.push(cells[index].cascade(previous)?);
        }
        suffix.reverse();
        let equivalent = suffix
            .first()
            .ok_or_else(|| invalid("empty complete Norton line"))?;
        self.stamp_norton_ports(ctx, equivalent)?;
        self.reconstruct_norton_states(ctx, coeff, &inverse, &cells, &suffix)
    }

    fn stamp_norton_ports(&self, ctx: &mut StampContext<'_>, line: &Norton) -> SpiceResult<()> {
        let conductors = self.conductors();
        for i in 0..conductors {
            let branch = ctx.branch(i)?;
            let (ip, ir) = self.physical_pair(i);
            let (op, or) = self.physical_pair(conductors + i);
            let (input_positive_row, input_return_row) = (
                ctx.unknowns.checked_node_row(ip)?,
                ctx.unknowns.checked_node_row(ir)?,
            );
            if let Some(row) = input_positive_row {
                ctx.matrix.add(row, branch, 1.)?;
            }
            if let Some(row) = input_return_row {
                ctx.matrix.add(row, branch, -1.)?;
            }
            ctx.matrix.add(branch, branch, 1.)?;
            ctx.rhs.add_to(branch, line.j1[i])?;
            ctx.stamp_rhs(op, -line.j2[i])?;
            ctx.stamp_rhs(or, line.j2[i])?;
            for j in 0..conductors {
                let (left, lr) = self.physical_pair(j);
                let (right, rr) = self.physical_pair(conductors + j);
                // Iin - Y11 Vin - Y12 Vout = Jleft.
                for (positive, negative, coefficient) in [
                    (left, lr, -line.y11.at(i, j)),
                    (right, rr, -line.y12.at(i, j)),
                ] {
                    if let Some(col) = ctx.unknowns.checked_node_row(positive)? {
                        ctx.matrix.add(branch, col, coefficient)?;
                    }
                    if let Some(col) = ctx.unknowns.checked_node_row(negative)? {
                        ctx.matrix.add(branch, col, -coefficient)?;
                    }
                }
                // Iout=Y21 Vin+Y22 Vout+Jright, both terminal KCL signs.
                for (positive, negative, coefficient) in [
                    (left, lr, line.y21.at(i, j)),
                    (right, rr, line.y22.at(i, j)),
                ] {
                    ctx.stamp(op, positive, coefficient)?;
                    ctx.stamp(op, negative, -coefficient)?;
                    ctx.stamp(or, positive, -coefficient)?;
                    ctx.stamp(or, negative, coefficient)?;
                }
            }
        }
        Ok(())
    }

    fn reconstruct_norton_states(
        &self,
        ctx: &mut StampContext<'_>,
        coeff: &Coefficients,
        inverse: &Square,
        cells: &[Norton],
        suffix: &[Norton],
    ) -> SpiceResult<()> {
        let count = self.conductors();
        let mut left = Vec::new();
        let mut terminal_right = Vec::new();
        left.try_reserve_exact(count)
            .map_err(|_| invalid("input spatial voltages"))?;
        terminal_right
            .try_reserve_exact(count)
            .map_err(|_| invalid("output spatial voltages"))?;
        for i in 0..count {
            let (ip, ir) = self.physical_pair(i);
            let (op, or) = self.physical_pair(count + i);
            left.push(ctx.node_voltage(ip) - ctx.node_voltage(ir));
            terminal_right.push(ctx.node_voltage(op) - ctx.node_voltage(or));
        }
        let width = self.plan.length
            / f64::from(u32::try_from(self.sections).map_err(|_| invalid("spatial width"))?);
        for section in 0..self.sections {
            let right = if section + 1 < self.sections {
                cells[section].interface(&suffix[section + 1], &left, &terminal_right)?
            } else {
                terminal_right.clone()
            };
            let source = (0..count)
                .map(|i| history_term(&ctx.states, coeff, slot(section, i, SERIES, count)))
                .collect::<SpiceResult<Vec<_>>>()?;
            let mut potential_difference = Vec::new();
            for (i, &voltage) in left.iter().enumerate() {
                potential_difference.push(voltage - right[i] - source[i]);
            }
            let middle_current = inverse.times(&potential_difference)?;
            for i in 0..count {
                let (mut charge_left, mut flux, mut charge_right) = (0., 0., 0.);
                for j in 0..count {
                    let index = i * count + j;
                    let charge = self.plan.c.data()[index] * width * 0.5;
                    let inductance = self.plan.l.data()[index] * width;
                    charge_left = charge.mul_add(left[j], charge_left);
                    charge_right = charge.mul_add(right[j], charge_right);
                    flux = inductance.mul_add(middle_current[j], flux);
                }
                for (kind, value) in [(LEFT, charge_left), (SERIES, flux), (RIGHT, charge_right)] {
                    let index = slot(section, i, kind, count);
                    let historical = history_term(&ctx.states, coeff, index)?;
                    write_energy(&mut ctx.states, index, value, historical, coeff.ag()[0])?;
                }
            }
            left = right;
        }
        Ok(())
    }
}

/// Direct real DC Norton coefficients for one segment of the continuous
/// distributed cable, not its transient finite-volume approximation.
fn dc_segment(plan: &super::plan::CoupledLinePlan, length: Real) -> SpiceResult<Norton> {
    let mut segment = plan.clone();
    segment.length = length;
    let admittance = segment.admittance_at(0.)?;
    let conductors = plan.conductors();
    let mut blocks = [
        Square::zeros(conductors)?,
        Square::zeros(conductors)?,
        Square::zeros(conductors)?,
        Square::zeros(conductors)?,
    ];
    for row in 0..2 * conductors {
        for col in 0..2 * conductors {
            let entry = admittance
                .get(row, col)
                .ok_or_else(|| invalid("DC Norton coefficient missing"))?;
            if entry.im.abs() > 1e-12 * entry.re.abs().max(1e-200) {
                return Err(invalid("DC Norton coefficient has nonreal component"));
            }
            let position = 2 * usize::from(row >= conductors) + usize::from(col >= conductors);
            blocks[position].set(row % conductors, col % conductors, entry.re);
        }
    }
    let [y11, y12, y21, y22] = blocks;
    Ok(Norton {
        y11,
        y12,
        y21,
        y22,
        j1: vec![0.; conductors],
        j2: vec![0.; conductors],
    })
}

impl CoupledLine {
    /// Reconstruct physically damped DC spatial charge/flux from the two
    /// solved boundary potentials. A growing forward ABCD eigenmode is never
    /// propagated from a rounded input current into an exp(1000) overflow.
    fn dc_stable_state(&self, ctx: &mut StampContext<'_>) -> SpiceResult<()> {
        let conductors = self.conductors();
        let width = self.plan.length
            / f64::from(u32::try_from(self.sections).map_err(|_| invalid("DC section count"))?);
        let cell = dc_segment(&self.plan, width)?;
        let half = dc_segment(&self.plan, width * 0.5)?;
        let mut suffix = Vec::new();
        suffix
            .try_reserve_exact(self.sections)
            .map_err(|_| invalid("DC Norton suffix allocation"))?;
        suffix.push(cell.clone());
        for _ in 1..self.sections {
            let previous = suffix
                .last()
                .ok_or_else(|| invalid("DC Norton suffix empty"))?;
            suffix.push(cell.cascade(previous)?);
        }
        suffix.reverse();
        let mut left = Vec::new();
        let mut right_endpoint = Vec::new();
        left.try_reserve_exact(conductors)
            .map_err(|_| invalid("DC Norton left voltages"))?;
        right_endpoint
            .try_reserve_exact(conductors)
            .map_err(|_| invalid("DC Norton right voltages"))?;
        for i in 0..conductors {
            let (ip, ir) = self.physical_pair(i);
            let (op, or) = self.physical_pair(conductors + i);
            left.push(ctx.node_voltage(ip) - ctx.node_voltage(ir));
            right_endpoint.push(ctx.node_voltage(op) - ctx.node_voltage(or));
        }
        for section in 0..self.sections {
            let right = if section + 1 < self.sections {
                cell.interface(&suffix[section + 1], &left, &right_endpoint)?
            } else {
                right_endpoint.clone()
            };
            let midpoint = half.interface(&half, &left, &right)?;
            let mid_current = half
                .y21
                .times(&left)?
                .iter()
                .zip(half.y22.times(&midpoint)?)
                .map(|(&from_left, to_mid)| -from_left - to_mid)
                .collect::<Vec<_>>();
            for i in 0..conductors {
                let (mut qleft, mut flux, mut qright) = (0., 0., 0.);
                for j in 0..conductors {
                    let index = i * conductors + j;
                    let charge = width * 0.5 * self.plan.c.data()[index];
                    qleft = charge.mul_add(left[j], qleft);
                    qright = charge.mul_add(right[j], qright);
                    flux = (width * self.plan.l.data()[index]).mul_add(mid_current[j], flux);
                }
                for (kind, quantity) in [(LEFT, qleft), (SERIES, flux), (RIGHT, qright)] {
                    let at = slot(section, i, kind, conductors);
                    ctx.states.set(at, quantity)?;
                    ctx.states.set(at + 1, 0.)?;
                }
            }
            left = right;
        }
        Ok(())
    }
}

impl CoupledLine {
    /// A zero R/L series-impedance matrix imposes an ideal conductor
    /// constraint *for every mode*. Its distributed shunt G/C is exactly a
    /// lumped N-conductor shunt, for any spatial division and time method.
    /// Unlike a Norton inverse, this descriptor representation remains
    /// well-defined with singular Z and exact voltage equations.
    fn stamp_ideal_series_transient(
        &self,
        ctx: &mut StampContext<'_>,
        coeff: &Coefficients,
    ) -> SpiceResult<()> {
        let conductors = self.conductors();
        let ag0 = coeff.ag()[0];
        let section_width = self.plan.length
            / f64::from(u32::try_from(self.sections).map_err(|_| invalid("ideal section count"))?);
        let mut voltage = Vec::new();
        voltage
            .try_reserve_exact(conductors)
            .map_err(|_| invalid("ideal line shunt voltage allocation"))?;
        for i in 0..conductors {
            let (positive, negative) = self.physical_pair(conductors + i);
            voltage.push(ctx.node_voltage(positive) - ctx.node_voltage(negative));
        }
        let mut history = vec![0.; conductors];
        for section in 0..self.sections {
            for i in 0..conductors {
                let mut charge = 0.;
                for (j, &potential) in voltage.iter().enumerate() {
                    charge = (0.5 * section_width * self.plan.c.data()[i * conductors + j])
                        .mul_add(potential, charge);
                }
                for kind in [LEFT, RIGHT] {
                    let offset = slot(section, i, kind, conductors);
                    let previous = history_term(&ctx.states, coeff, offset)?;
                    history[i] += previous;
                    write_energy(&mut ctx.states, offset, charge, previous, ag0)?;
                }
                let series = slot(section, i, SERIES, conductors);
                let previous = history_term(&ctx.states, coeff, series)?;
                write_energy(&mut ctx.states, series, 0., previous, ag0)?;
            }
        }
        if history.iter().any(|value| !value.is_finite()) {
            return Err(invalid("ideal-line shunt history overflows"));
        }
        for i in 0..conductors {
            let (ip, ir) = self.physical_pair(i);
            let (op, or) = self.physical_pair(conductors + i);
            let branch = ctx.branch(i)?;
            for (node, coefficient) in [(ip, 1.), (ir, -1.), (op, -1.), (or, 1.)] {
                if let Some(row) = ctx.unknowns.checked_node_row(node)? {
                    ctx.matrix.add(row, branch, coefficient)?;
                }
            }
            for (node, coefficient) in [(op, 1.), (or, -1.), (ip, -1.), (ir, 1.)] {
                if let Some(col) = ctx.unknowns.checked_node_row(node)? {
                    ctx.matrix.add(branch, col, coefficient)?;
                }
            }
            ctx.stamp_rhs(op, -history[i])?;
            ctx.stamp_rhs(or, history[i])?;
            for j in 0..conductors {
                let (qp, qr) = self.physical_pair(conductors + j);
                let index = i * conductors + j;
                let shunt = self.plan.length
                    * ag0.mul_add(self.plan.c.data()[index], self.plan.g.data()[index]);
                if !shunt.is_finite() {
                    return Err(invalid("ideal-line distributed shunt overflows"));
                }
                ctx.stamp(op, qp, shunt)?;
                ctx.stamp(op, qr, -shunt)?;
                ctx.stamp(or, qp, -shunt)?;
                ctx.stamp(or, qr, shunt)?;
            }
        }
        Ok(())
    }
}
