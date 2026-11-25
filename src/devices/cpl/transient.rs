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
use crate::devices::traits::{AnalysisMode, MnaUnknowns, StampContext};
use crate::maths::{Coefficients, dense_complex::DenseComplex, integrator::ChargeHistory};
use crate::primitives::{Complex, Real, SpiceResult};

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
            series_norm.sqrt() / shunt_norm.sqrt()
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

impl Square {
    fn add(&self, other: &Self) -> SpiceResult<Self> {
        if self.dimension != other.dimension {
            return Err(invalid("hybrid dimension mismatch"));
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
            return Err(invalid("nonfinite hybrid matrix sum"));
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
            return Err(invalid("nonfinite hybrid inverse"));
        }
        Ok(inverse)
    }
    fn identity(dimension: usize) -> SpiceResult<Self> {
        let mut result = Self::zeros(dimension)?;
        for i in 0..dimension {
            result.set(i, i, 1.);
        }
        Ok(result)
    }
    fn transpose(&self) -> SpiceResult<Self> {
        let mut result = Self::zeros(self.dimension)?;
        for row in 0..self.dimension {
            for col in 0..self.dimension {
                result.set(col, row, self.at(row, col));
            }
        }
        Ok(result)
    }
}
fn sum_vectors(a: &[Real], b: &[Real]) -> SpiceResult<Vec<Real>> {
    if a.len() != b.len() {
        return Err(invalid("hybrid vector size mismatch"));
    }
    let output = a.iter().zip(b).map(|(&x, &y)| x + y).collect::<Vec<_>>();
    if output.iter().any(|v| !v.is_finite()) {
        return Err(invalid("nonfinite hybrid source sum"));
    }
    Ok(output)
}
fn diff_vectors(a: &[Real], b: &[Real]) -> SpiceResult<Vec<Real>> {
    if a.len() != b.len() {
        return Err(invalid("hybrid vector size mismatch"));
    }
    let output = a.iter().zip(b).map(|(&x, &y)| x - y).collect::<Vec<_>>();
    if output.iter().any(|v| !v.is_finite()) {
        return Err(invalid("nonfinite hybrid source difference"));
    }
    Ok(output)
}

/// Passive hybrid port equations, using balanced current units I'=scale*I:
/// Vleft = A Ileft' + B Vright + f,
/// Iright' = -B^T Ileft' + D Vright + g.
/// Both A and D are positive semidefinite at real nonnegative integration
/// frequency. In particular A may be zero or singular: ideal series
/// constraints require no inverse impedance, nor any artificial resistance.
#[derive(Debug, Clone)]
struct Hybrid {
    a: Square,
    b: Square,
    d: Square,
    f: Vec<Real>,
    g: Vec<Real>,
}

/// The shared port voltage is P Ileft' + Q Vright + h.
struct Interface {
    p: Square,
    q: Square,
    h: Vec<Real>,
}
impl Interface {
    fn voltage(&self, current: &[Real], right: &[Real]) -> SpiceResult<Vec<Real>> {
        sum_vectors(
            &sum_vectors(&self.p.times(current)?, &self.q.times(right)?)?,
            &self.h,
        )
    }
}
impl Hybrid {
    fn junction(&self, other: &Self) -> SpiceResult<Interface> {
        // I_mid_right = B_left^T I_left - D_left V_mid - g_left.
        // The positive-real interconnection is I + A_right D_left, even
        // when either constitutive matrix has an exact nullspace.
        let inverse = Square::identity(self.a.dimension)?
            .add(&other.a.mul(&self.d)?)?
            .inverse()?;
        let product = inverse.mul(&other.a)?;
        Ok(Interface {
            p: product.mul(&self.b.transpose()?)?,
            q: inverse.mul(&other.b)?,
            h: inverse.times(&diff_vectors(&other.f, &other.a.times(&self.g)?)?)?,
        })
    }
    fn cascade(&self, other: &Self) -> SpiceResult<Self> {
        self.cascade_at(other, &self.junction(other)?)
    }
    fn cascade_at(&self, other: &Self, interface: &Interface) -> SpiceResult<Self> {
        let right_b = other.b.transpose()?;
        Ok(Self {
            a: self.a.add(&self.b.mul(&interface.p)?)?,
            b: self.b.mul(&interface.q)?,
            d: other.d.add(&right_b.mul(&self.d)?.mul(&interface.q)?)?,
            f: sum_vectors(&self.f, &self.b.times(&interface.h)?)?,
            g: sum_vectors(
                &other.g,
                &right_b.times(&sum_vectors(&self.g, &self.d.times(&interface.h)?)?)?,
            )?,
        })
    }
    fn left_voltage(&self, current: &[Real], right: &[Real]) -> SpiceResult<Vec<Real>> {
        sum_vectors(
            &sum_vectors(&self.a.times(current)?, &self.b.times(right)?)?,
            &self.f,
        )
    }
    fn next_current(&self, current: &[Real], right: &[Real]) -> SpiceResult<Vec<Real>> {
        diff_vectors(
            &self.b.transpose()?.times(current)?,
            &sum_vectors(&self.d.times(right)?, &self.g)?,
        )
    }
    fn from_dc(transfer: &DenseComplex, conductors: usize) -> SpiceResult<Self> {
        let mut blocks = [
            Square::zeros(conductors)?,
            Square::zeros(conductors)?,
            Square::zeros(conductors)?,
        ];
        for row in 0..conductors {
            for col in 0..conductors {
                for (block, (r, c)) in blocks.iter_mut().zip([
                    (row, col),
                    (row, col + conductors),
                    (row + conductors, col),
                ]) {
                    let value = transfer
                        .get(r, c)
                        .ok_or_else(|| invalid("DC hybrid transfer index"))?;
                    if !value.is_finite() || value.im != 0. {
                        return Err(invalid("nonreal DC hybrid transfer"));
                    }
                    block.set(row, col, value.re);
                }
            }
        }
        let [voltage, series, shunt] = blocks;
        let inverse = voltage.inverse()?;
        Ok(Self {
            a: inverse.mul(&series)?.negative(),
            d: shunt.mul(&inverse)?.negative(),
            b: inverse,
            f: vec![0.; conductors],
            g: vec![0.; conductors],
        })
    }
    fn condense(cells: &[Self]) -> SpiceResult<(Self, Vec<Interface>)> {
        let (last, rest) = cells
            .split_last()
            .ok_or_else(|| invalid("empty spatial hybrid"))?;
        let mut network = last.clone();
        let mut interfaces = Vec::new();
        interfaces
            .try_reserve_exact(rest.len())
            .map_err(|_| invalid("hybrid interface allocation"))?;
        for cell in rest.iter().rev() {
            let interface = cell.junction(&network)?;
            network = cell.cascade_at(&network, &interface)?;
            interfaces.push(interface);
        }
        interfaces.reverse();
        Ok((network, interfaces))
    }
    fn stamps(
        &self,
        line: &CoupledLine,
        unknowns: &MnaUnknowns,
        first: usize,
        scale: Real,
    ) -> SpiceResult<Vec<(usize, usize, Complex)>> {
        let count = self.a.dimension;
        let mut entries = line.terminal_kcl(unknowns, first)?;
        let mut insert = |row: usize, col: Option<usize>, value: Real| {
            if let Some(col) = col
                && value != 0.
            {
                entries.push((row, col, Complex::real(value)));
            }
        };
        for i in 0..count {
            let branch = first + i;
            let output = first + count + i;
            let (ip, ir) = line.physical_pair(i);
            insert(branch, unknowns.checked_node_row(ip)?, 1.);
            insert(branch, unknowns.checked_node_row(ir)?, -1.);
            insert(output, Some(output), 1.);
            for j in 0..count {
                let (positive, negative) = line.physical_pair(count + j);
                let (positive, negative) = (
                    unknowns.checked_node_row(positive)?,
                    unknowns.checked_node_row(negative)?,
                );
                let input_branch = Some(first + j);
                insert(branch, input_branch, -scale * self.a.at(i, j));
                insert(branch, positive, -self.b.at(i, j));
                insert(branch, negative, self.b.at(i, j));
                insert(output, input_branch, self.b.at(j, i));
                let d = self.d.at(i, j) / scale;
                insert(output, positive, -d);
                insert(output, negative, d);
            }
        }
        if entries.iter().any(|(_, _, value)| !value.is_finite()) {
            return Err(invalid("nonfinite hybrid MNA coefficient"));
        }
        Ok(entries)
    }
}

impl Cell {
    fn coefficients(&self) -> SpiceResult<Hybrid> {
        let b = Square::identity(self.conductors)?
            .add(&self.series.mul(&self.half_shunt)?)?
            .inverse()?;
        let a = b.mul(&self.series)?;
        let d = self.half_shunt.add(&self.half_shunt.mul(&b)?)?;
        Ok(Hybrid {
            a,
            b,
            d,
            f: vec![0.; self.conductors],
            g: vec![0.; self.conductors],
        })
    }
    fn hybrid(
        &self,
        coefficients: &Hybrid,
        hleft: &[Real],
        hseries: &[Real],
        hright: &[Real],
    ) -> SpiceResult<Hybrid> {
        let hleft = hleft
            .iter()
            .map(|value| value * self.scale)
            .collect::<Vec<_>>();
        let hright = hright
            .iter()
            .map(|value| value * self.scale)
            .collect::<Vec<_>>();
        // (I+Z H)Vleft = Vright + Z Ileft' + hseries - Z hleft'.
        let f = coefficients
            .b
            .times(&diff_vectors(hseries, &self.series.times(&hleft)?)?)?;
        let g = sum_vectors(&sum_vectors(&hleft, &hright)?, &self.half_shunt.times(&f)?)?;
        Ok(Hybrid {
            f,
            g,
            ..coefficients.clone()
        })
    }
    fn middle_current(
        &self,
        current: &[Real],
        voltage: &[Real],
        hleft: &[Real],
    ) -> SpiceResult<Vec<Real>> {
        Ok(diff_vectors(current, &self.half_shunt.times(voltage)?)?
            .iter()
            .zip(hleft)
            .map(|(i, h)| i / self.scale - h)
            .collect())
    }
}

impl CoupledLine {
    #[must_use]
    pub(super) fn spatial_state_count(&self) -> usize {
        self.sections * self.conductors() * SLOTS_PER_SECTION_CONDUCTOR
    }
    fn section_width(&self) -> SpiceResult<Real> {
        Ok(self.plan.length
            / f64::from(
                u32::try_from(self.sections).map_err(|_| invalid("section count overflow"))?,
            ))
    }
    fn dc_hybrid(&self, length: Real) -> SpiceResult<(Hybrid, Real)> {
        let (transfer, scale, steps) = self.plan.bounded_transfer(0., length, 4.)?;
        let mut network = Hybrid::from_dc(&transfer, self.conductors())?;
        for _ in 0..steps {
            network = network.cascade(&network)?;
        }
        Ok((network, scale))
    }
    pub(super) fn dc_hybrid_stamps(
        &self,
        unknowns: &MnaUnknowns,
        first: usize,
    ) -> SpiceResult<Vec<(usize, usize, Complex)>> {
        let (network, scale) = self.dc_hybrid(self.plan.length)?;
        network.stamps(self, unknowns, first, scale)
    }
    fn boundary(&self, ctx: &StampContext<'_>, scale: Real) -> SpiceResult<(Vec<Real>, Vec<Real>)> {
        let count = self.conductors();
        let mut current = Vec::with_capacity(count);
        let mut voltage = Vec::with_capacity(count);
        for i in 0..count {
            let (positive, negative) = self.physical_pair(count + i);
            voltage.push(ctx.node_voltage(positive) - ctx.node_voltage(negative));
            current.push(scale * ctx.row_value(ctx.branch(i)?)?);
        }
        Ok((current, voltage))
    }
    fn section_energies(
        &self,
        width: Real,
        left: &[Real],
        current: &[Real],
        right: &[Real],
    ) -> impl Iterator<Item = (usize, [Real; 3])> {
        let count = self.conductors();
        (0..count).map(move |i| {
            let (mut qleft, mut flux, mut qright) = (0., 0., 0.);
            for j in 0..count {
                let index = i * count + j;
                let cap = 0.5 * width * self.plan.c.data()[index];
                qleft = cap.mul_add(left[j], qleft);
                qright = cap.mul_add(right[j], qright);
                flux = (width * self.plan.l.data()[index]).mul_add(current[j], flux);
            }
            (i, [qleft, flux, qright])
        })
    }
    /// Continuous DC bias, reconstructed by passive half-section elimination
    /// from physical input currents and far-end voltages, including singular
    /// series impedances. No exponentially growing forward state is used.
    pub(super) fn dc_spatial_state(&self, ctx: &mut StampContext<'_>) -> SpiceResult<()> {
        if ctx.states.is_empty() {
            return Ok(());
        }
        if ctx.states.len() != self.spatial_state_count() {
            return Err(invalid("DC spatial state layout mismatch"));
        }
        let width = self.section_width()?;
        let (half, scale) = self.dc_hybrid(0.5 * width)?;
        let cell = half.cascade(&half)?;
        let cells = vec![cell; self.sections];
        let (_, interfaces) = Hybrid::condense(&cells)?;
        let midpoint = half.junction(&half)?;
        let (mut current, endpoint) = self.boundary(ctx, scale)?;
        for (section, cell) in cells.iter().enumerate() {
            let right = if let Some(interface) = interfaces.get(section) {
                interface.voltage(&current, &endpoint)?
            } else {
                endpoint.clone()
            };
            let left = cell.left_voltage(&current, &right)?;
            let middle = midpoint.voltage(&current, &right)?;
            let mid_current = half
                .next_current(&current, &middle)?
                .iter()
                .map(|i| i / scale)
                .collect::<Vec<_>>();
            for (i, values) in self.section_energies(width, &left, &mid_current, &right) {
                for (kind, value) in [LEFT, SERIES, RIGHT].into_iter().zip(values) {
                    let offset = slot(section, i, kind, self.conductors());
                    ctx.states.set(offset, value)?;
                    ctx.states.set(offset + 1, 0.)?;
                }
            }
            current = cell.next_current(&current, &right)?;
        }
        Ok(())
    }
    fn section_history(
        &self,
        ctx: &StampContext<'_>,
        coeff: &Coefficients,
        section: usize,
        kind: usize,
    ) -> SpiceResult<Vec<Real>> {
        (0..self.conductors())
            .map(|i| {
                history_term(
                    &ctx.states,
                    coeff,
                    slot(section, i, kind, self.conductors()),
                )
            })
            .collect()
    }
    pub(super) fn stamp_stable_transient(
        &self,
        ctx: &mut StampContext<'_>,
        coeff: &Coefficients,
    ) -> SpiceResult<()> {
        let AnalysisMode::Transient { dt, .. } = ctx.mode else {
            return Err(invalid("spatial hybrid outside transient"));
        };
        if coeff.dt().total_cmp(&dt) != std::cmp::Ordering::Equal
            || ctx.states.len() != self.spatial_state_count()
        {
            return Err(invalid("spatial hybrid timestep/history mismatch"));
        }
        let cell = Cell::new(self, coeff.ag()[0])?;
        let coefficients = cell.coefficients()?;
        let mut cells = Vec::new();
        cells
            .try_reserve_exact(self.sections)
            .map_err(|_| invalid("spatial hybrid cell allocation"))?;
        for section in 0..self.sections {
            cells.push(cell.hybrid(
                &coefficients,
                &self.section_history(ctx, coeff, section, LEFT)?,
                &self.section_history(ctx, coeff, section, SERIES)?,
                &self.section_history(ctx, coeff, section, RIGHT)?,
            )?);
        }
        let (network, interfaces) = Hybrid::condense(&cells)?;
        let first = ctx.branch(0)?;
        for (row, col, value) in network.stamps(self, ctx.unknowns, first, cell.scale)? {
            ctx.matrix.add(row, col, value.re)?;
        }
        for i in 0..self.conductors() {
            ctx.rhs.add_to(first + i, network.f[i])?;
            ctx.rhs
                .add_to(first + self.conductors() + i, network.g[i] / cell.scale)?;
        }
        self.reconstruct_hybrid_states(ctx, coeff, &cell, &cells, &interfaces)
    }
    fn reconstruct_hybrid_states(
        &self,
        ctx: &mut StampContext<'_>,
        coeff: &Coefficients,
        cell: &Cell,
        cells: &[Hybrid],
        interfaces: &[Interface],
    ) -> SpiceResult<()> {
        let width = self.section_width()?;
        let (mut current, endpoint) = self.boundary(ctx, cell.scale)?;
        for (section, network) in cells.iter().enumerate() {
            let right = if let Some(interface) = interfaces.get(section) {
                interface.voltage(&current, &endpoint)?
            } else {
                endpoint.clone()
            };
            let left = network.left_voltage(&current, &right)?;
            let middle = cell.middle_current(
                &current,
                &left,
                &self.section_history(ctx, coeff, section, LEFT)?,
            )?;
            for (i, values) in self.section_energies(width, &left, &middle, &right) {
                for (kind, value) in [LEFT, SERIES, RIGHT].into_iter().zip(values) {
                    let offset = slot(section, i, kind, self.conductors());
                    let history = history_term(&ctx.states, coeff, offset)?;
                    write_energy(&mut ctx.states, offset, value, history, coeff.ag()[0])?;
                }
            }
            current = network.next_current(&current, &right)?;
        }
        Ok(())
    }
}
