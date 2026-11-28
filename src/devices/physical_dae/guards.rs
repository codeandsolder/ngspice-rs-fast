//! Branch observations of the executed compiled physical expression graph.
//! A predicted Taylor root is a probe hint, never proof of an event.

use super::{BranchTrace, ConstitutiveGuards, Guard, LocalGuard, Region};
use crate::maths::taylor::Jet;
use crate::primitives::{Real, SpiceError, SpiceResult};
use std::cmp::Ordering;

const MAX_GUARD_COEFFICIENTS: usize = 8_388_608;
// Aggregate arithmetic budget for one trace prediction, independent of the
// number of guards. Charge quadratic Bernstein conversion/subdivision work.
const MAX_ROOT_WORK: usize = 64 * 1024 * 1024;

struct RootWork(usize);

impl RootWork {
    fn charge(&mut self, operations: usize) -> SpiceResult<()> {
        self.0 = self.0.checked_sub(operations).ok_or_else(|| {
            SpiceError::circuit("physical branch root prediction budget exhausted")
        })?;
        Ok(())
    }
}

impl LocalGuard {
    pub(crate) fn sign(slot: usize, operand: u8, argument: Jet, boundary: Real) -> Self {
        Self {
            slot,
            operand,
            region: Region::Sign(argument.compare(boundary).unwrap_or(Ordering::Equal)),
            argument,
            boundaries: [Some(boundary), None],
        }
    }

    pub(crate) fn interval(slot: usize, argument: Jet, boundaries: &[Real]) -> Self {
        let index = boundaries
            .partition_point(|&boundary| argument.compare(boundary) != Some(Ordering::Less));
        Self {
            slot,
            operand: 0,
            region: Region::Interval(index),
            argument,
            boundaries: [
                index
                    .checked_sub(1)
                    .and_then(|i| boundaries.get(i))
                    .copied(),
                boundaries.get(index).copied(),
            ],
        }
    }

    fn predicted_root(
        &self,
        horizon: Real,
        reverse: bool,
        work: &mut RootWork,
    ) -> SpiceResult<Option<Real>> {
        let mut earliest = None;
        for &boundary in self.boundaries.iter().flatten() {
            work.charge(self.argument.coefficients().len())?;
            let mut coefficients = self.argument.coefficients().to_vec();
            coefficients[0] -= boundary;
            if reverse {
                for coefficient in coefficients.iter_mut().skip(1).step_by(2) {
                    *coefficient = -*coefficient;
                }
            }
            if let Some(root) = first_root(&coefficients, earliest.unwrap_or(horizon), work)? {
                earliest = Some(root);
            }
        }
        Ok(earliest)
    }
}

impl ConstitutiveGuards<'_> {
    /// Record the branch actually taken, including its equality convention.
    /// Slot identities belong to the device's constitutive law, independent
    /// of the active region or the order of the local Taylor expansion.
    fn split(
        &mut self,
        slot: usize,
        argument: &Jet,
        boundary: Real,
        lower: bool,
        active: bool,
    ) -> SpiceResult<()> {
        if active && let Some(trace) = self.trace.as_deref_mut() {
            trace.push(
                self.device,
                LocalGuard {
                    slot,
                    operand: 0,
                    region: Region::Interval(usize::from(!lower)),
                    argument: argument.clone(),
                    boundaries: [Some(boundary), None],
                },
            )?;
        }
        Ok(())
    }

    pub(crate) fn below(
        &mut self,
        slot: usize,
        argument: &Jet,
        boundary: Real,
        active: bool,
    ) -> SpiceResult<bool> {
        let lower = argument.compare(boundary).is_some_and(Ordering::is_lt);
        self.split(slot, argument, boundary, lower, active)?;
        Ok(lower)
    }

    pub(crate) fn above(
        &mut self,
        slot: usize,
        argument: &Jet,
        boundary: Real,
        active: bool,
    ) -> SpiceResult<bool> {
        let upper = argument.compare(boundary).is_some_and(Ordering::is_gt);
        self.split(slot, argument, boundary, !upper, active)?;
        Ok(upper)
    }
}

impl BranchTrace {
    fn push(&mut self, device: usize, local: LocalGuard) -> SpiceResult<()> {
        self.coefficients = self
            .coefficients
            .checked_add(local.argument.coefficients().len())
            .filter(|&count| count <= MAX_GUARD_COEFFICIENTS)
            .ok_or_else(|| SpiceError::circuit("physical branch guard resource limit"))?;
        self.guards
            .try_reserve(1)
            .map_err(|_| SpiceError::circuit("physical branch guard allocation failed"))?;
        self.guards.push(Guard { device, local });
        Ok(())
    }

    pub(crate) fn extend(&mut self, device: usize, guards: Vec<LocalGuard>) -> SpiceResult<()> {
        for local in guards {
            self.push(device, local)?;
        }
        Ok(())
    }

    pub(crate) const fn is_empty(&self) -> bool {
        self.guards.is_empty()
    }

    /// Trace order is the fixed device load order, then compiled slot order.
    /// Missing guards mean a formerly lazy branch is now active or vice versa.
    pub(crate) fn same_regions(&self, other: &Self) -> bool {
        self.guards.len() == other.guards.len()
            && self.guards.iter().zip(&other.guards).all(|(a, b)| {
                a.device == b.device
                    && a.local.slot == b.local.slot
                    && a.local.operand == b.local.operand
                    && a.local.region == b.local.region
            })
    }

    pub(crate) fn changed_device(&self, other: &Self) -> Option<usize> {
        for (a, b) in self.guards.iter().zip(&other.guards) {
            if a.device != b.device
                || a.local.slot != b.local.slot
                || a.local.operand != b.local.operand
                || a.local.region != b.local.region
            {
                return Some(a.device.min(b.device));
            }
        }
        if self.guards.len() == other.guards.len() {
            None
        } else {
            self.guards
                .get(other.guards.len())
                .or_else(|| other.guards.get(self.guards.len()))
                .map(|guard| guard.device)
        }
    }

    /// Earliest possible root of the local Taylor approximations, in seconds.
    /// The time scale is the one used to normalize the derivative array.
    pub(crate) fn predicted_offset(
        &self,
        duration: Real,
        scale: Real,
    ) -> SpiceResult<Option<Real>> {
        self.root_offset(duration, scale, false)
    }

    pub(crate) fn predicted_backward_offset(
        &self,
        duration: Real,
        scale: Real,
        already_left: bool,
    ) -> SpiceResult<Option<Real>> {
        self.root_offset(duration, scale, !already_left)
    }

    fn root_offset(&self, duration: Real, scale: Real, reverse: bool) -> SpiceResult<Option<Real>> {
        let mut work = RootWork(MAX_ROOT_WORK);
        let mut earliest = None;
        for guard in &self.guards {
            let horizon = earliest.unwrap_or(duration) / scale;
            let root = guard
                .local
                .predicted_root(horizon, reverse, &mut work)
                .map_err(|error| {
                    SpiceError::circuit(format!(
                        "physical branch guard device {} slot {}: {error}",
                        guard.device, guard.local.slot
                    ))
                })?;
            if let Some(root) = root {
                let offset = root * scale;
                if offset.is_finite() && offset > 0. && offset <= duration {
                    earliest = Some(offset);
                }
            }
        }
        Ok(earliest)
    }
}

/// First positive root of a Taylor polynomial over (0, horizon]. Strip
/// the root at the accepted origin; an event already crossed must not be
/// scheduled again. The coefficients are dimensionless local-time powers.
fn first_root(
    coefficients: &[Real],
    horizon: Real,
    work: &mut RootWork,
) -> SpiceResult<Option<Real>> {
    if !(horizon.is_finite() && horizon > 0.) {
        return Ok(None);
    }
    let Some(first) = coefficients.iter().position(|&value| value != 0.) else {
        return Ok(None);
    };
    let Some(last) = coefficients.iter().rposition(|&value| value != 0.) else {
        return Ok(None);
    };
    let coefficients = &coefficients[first..=last];
    match coefficients {
        [_] => return Ok(None),
        [a, b] => {
            let root = -a / b;
            return Ok((root > 0. && root <= horizon).then_some(root));
        }
        _ => {}
    }
    if coefficients.iter().all(|&value| value >= 0.)
        || coefficients.iter().all(|&value| value <= 0.)
    {
        return Ok(None);
    }
    work.charge(coefficients.len().saturating_mul(coefficients.len()))?;
    // Convert power coefficients on [0,1] to Bernstein coefficients.
    // Their convex hull encloses the polynomial, so intervals with one
    // strict sign need no subdivision. This is a hint from a truncated
    // local germ, not a bound on the actual nonlinear trajectory.
    let degree = coefficients.len() - 1;
    let mut power = 1.;
    let scaled = coefficients
        .iter()
        .map(|&value| {
            let result = if value == 0. { 0. } else { value * power };
            power *= horizon;
            result
        })
        .collect::<Vec<_>>();
    if scaled.iter().any(|value| !value.is_finite()) {
        return Ok(None);
    }
    let magnitude = scaled.iter().map(|value| value.abs()).fold(0., Real::max);
    if magnitude == 0. {
        return Ok(None);
    }
    let n = u32::try_from(degree)
        .map_err(|_| SpiceError::circuit("branch polynomial degree exceeds u32"))?;
    let mut bernstein = Vec::with_capacity(degree + 1);
    for i in 0..=n {
        let mut ratio = 1.;
        let mut sum = scaled[0] / magnitude;
        for k in 0..i {
            ratio *= Real::from(i - k) / Real::from(n - k);
            sum = ratio.mul_add(
                scaled[usize::try_from(k)
                    .map_err(|_| SpiceError::circuit("branch polynomial index overflow"))?
                    + 1]
                    / magnitude,
                sum,
            );
        }
        bernstein.push(sum);
    }
    Ok(isolate(bernstein, work)?.map(|fraction| fraction * horizon))
}

/// Left-first de Casteljau subdivision, bounded by both work and binary64
/// time resolution. No recursive call stack or unbounded root search.
fn isolate(coefficients: Vec<Real>, budget: &mut RootWork) -> SpiceResult<Option<Real>> {
    let mut pending = vec![(0., 1., coefficients)];
    for _ in 0..512 {
        let Some((lo, hi, coefficients)) = pending.pop() else {
            return Ok(None);
        };
        budget.charge(coefficients.len())?;
        let positive = coefficients.iter().any(|&value| value > 0.);
        let negative = coefficients.iter().any(|&value| value < 0.);
        if !(positive && negative) {
            if coefficients[coefficients.len() - 1] == 0. && hi > 0. {
                return Ok(Some(hi));
            }
            continue;
        }
        let mid = Real::midpoint(lo, hi);
        if hi - lo <= 8. * Real::EPSILON || mid <= lo || mid >= hi {
            return Ok(Some(mid));
        }
        let n = coefficients.len();
        budget.charge(n.saturating_mul(n))?;
        let mut work = coefficients;
        let mut left = Vec::with_capacity(n);
        let mut right = vec![0.; n];
        left.push(work[0]);
        right[n - 1] = work[n - 1];
        for k in 1..n {
            for i in 0..n - k {
                work[i] = Real::midpoint(work[i], work[i + 1]);
            }
            left.push(work[0]);
            right[n - k - 1] = work[n - k - 1];
        }
        pending.push((mid, hi, right));
        pending.push((lo, mid, left));
    }
    if pending.is_empty() {
        Ok(None)
    } else {
        Err(SpiceError::circuit(
            "physical branch Bernstein subdivision limit exhausted",
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn root(coefficients: &[Real], horizon: Real) -> Option<Real> {
        first_root(coefficients, horizon, &mut RootWork(MAX_ROOT_WORK)).unwrap()
    }

    #[test]
    fn constitutive_comparisons_select_exact_germs_and_filter_inactive_sites() {
        let mut up = BranchTrace::default();
        let mut down = BranchTrace::default();
        let rising = Jet::affine(2, 0.5, 1.);
        let falling = Jet::affine(2, 0.5, -1.);
        let mut guards = ConstitutiveGuards {
            trace: Some(&mut up),
            device: 12,
        };
        assert!(!guards.below(3, &rising, 0.5, true).unwrap());
        assert!(guards.above(8, &rising, 0.5, false).unwrap());
        let mut guards = ConstitutiveGuards {
            trace: Some(&mut down),
            device: 12,
        };
        assert!(guards.below(3, &falling, 0.5, true).unwrap());
        assert!(!guards.above(8, &falling, 0.5, false).unwrap());
        assert_eq!(up.guards.len(), 1);
        assert_eq!(down.guards.len(), 1);
        assert_eq!(up.changed_device(&down), Some(12));
        assert!(up.predicted_offset(1., 1.).unwrap().is_none());
        assert!(down.predicted_offset(1., 1.).unwrap().is_none());
    }

    #[test]
    fn prediction_budget_is_explicit() {
        assert!(first_root(&[0.1875, -1., 1.], 1., &mut RootWork(1)).is_err());
    }

    #[test]
    fn local_roots_include_a_pair_hidden_by_equal_endpoint_signs() {
        let found = root(&[0.1875, -1., 1.], 1.).unwrap();
        assert!((found - 0.25).abs() < 1e-13);
        let found = root(&[0., 0., -0.75, 1.], 1.).unwrap();
        assert!((found - 0.75).abs() < 1e-13);
        assert!(root(&[1., 0., 1.], 1.).is_none());
        assert!(root(&[0., 1.], 1.).is_none());
        assert_eq!(root(&[-0.5, 1.], 1.), Some(0.5));
    }
}
