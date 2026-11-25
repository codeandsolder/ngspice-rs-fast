//! Error-controlled, energy-contractive reduction of continuous PWL waves.
//!
//! Keep the endpoints of each protected interval exactly. A chord or a
//! two-segment tent replaces its interior only if it satisfies both a
//! componentwise interpolation error bound and the L2 Gram inequality.
//! The latter holds for every linear combination of affine trial columns,
//! not merely the currently estimated Newton endpoint.
use super::{Affine, Knot, Real};

fn product_integral(a: Real, b: Real, c: Real, d: Real) -> Real {
    // Integral of two linear functions over the unit interval.
    (2. * b).mul_add(d, b.mul_add(c, a.mul_add(d, 2. * a * c))) / 6.
}

fn chord(a: Real, b: Real, x: Real) -> Real {
    (b - a).mul_add(x, a)
}

fn tent(x: Real, split: Real) -> Real {
    if x <= split {
        x / split
    } else {
        (1. - x) / (1. - split)
    }
}

// Integral of a linear endpoint curve times a unit-height tent. Its
// squared integral is 1/3 for every interior peak location.
fn chord_tent(a: Real, b: Real, split: Real) -> Real {
    (a.mul_add(2. - split, b * (1. + split))) / 6.
}

// Pivoted semidefinite elimination. The Gram matrix has been scaled
// componentwise by the maximum magnitude of the original curve; the
// absolute slack here is only for accumulated binary64 arithmetic.
fn positive_semidefinite(matrix: &[Real], order: usize) -> bool {
    let roundoff = 128. * Real::EPSILON * Real::from(u32::try_from(order).unwrap_or(u32::MAX));
    if order == 1 {
        return matrix[0] >= -roundoff;
    }
    let mut a = matrix.to_vec();
    for k in 0..order {
        let mut pivot = k;
        for i in k + 1..order {
            if a[i * order + i] > a[pivot * order + pivot] {
                pivot = i;
            }
        }
        if a[pivot * order + pivot] <= roundoff {
            return (k..order).all(|i| (k..order).all(|j| a[i * order + j].abs() <= roundoff));
        }
        if pivot != k {
            for j in 0..order {
                a.swap(k * order + j, pivot * order + j);
            }
            for i in 0..order {
                a.swap(i * order + k, i * order + pivot);
            }
        }
        let diagonal = a[k * order + k];
        for i in k + 1..order {
            let factor = a[i * order + k] / diagonal;
            for j in i..order {
                let value = (-factor).mul_add(a[j * order + k], a[i * order + j]);
                a[i * order + j] = value;
                a[j * order + i] = value;
            }
        }
    }
    true
}

// A cubic matrix polynomial lies in the convex hull of its Bernstein
// coefficients. Subdivide a non-certifying hull rather than sampling
// the curve: every certified prefix is positive semidefinite.
fn bernstein_psd(coefficients: &[Vec<Real>; 4], order: usize, depth: u8) -> bool {
    if coefficients
        .iter()
        .all(|matrix| positive_semidefinite(matrix, order))
    {
        return true;
    }
    if depth == 0
        || !positive_semidefinite(&coefficients[0], order)
        || !positive_semidefinite(&coefficients[3], order)
    {
        return false;
    }
    let average = |a: &[Real], b: &[Real]| -> Vec<Real> {
        a.iter().zip(b).map(|(&a, &b)| a.midpoint(b)).collect()
    };
    let ab = average(&coefficients[0], &coefficients[1]);
    let bc = average(&coefficients[1], &coefficients[2]);
    let cd = average(&coefficients[2], &coefficients[3]);
    let abc = average(&ab, &bc);
    let bcd = average(&bc, &cd);
    let middle = average(&abc, &bcd);
    bernstein_psd(
        &[coefficients[0].clone(), ab, abc, middle.clone()],
        order,
        depth - 1,
    ) && bernstein_psd(
        &[middle, bcd, cd, coefficients[3].clone()],
        order,
        depth - 1,
    )
}

fn scalar_bernstein([a, b, c, d]: [Real; 4], depth: u8) -> bool {
    let slack = -128. * Real::EPSILON;
    if a.min(b).min(c).min(d) >= slack {
        return true;
    }
    if a.min(d) < slack || depth == 0 {
        return false;
    }
    let ab = a.midpoint(b);
    let bc = b.midpoint(c);
    let cd = c.midpoint(d);
    let abc = ab.midpoint(bc);
    let bcd = bc.midpoint(cd);
    let middle = abc.midpoint(bcd);
    scalar_bernstein([a, ab, abc, middle], depth - 1)
        && scalar_bernstein([middle, bcd, cd, d], depth - 1)
}

struct Reduction<'a> {
    knots: &'a [Knot],
    width: usize,
    indices: Vec<usize>,
    scales: Vec<Real>,
    error_weights: Vec<Real>,
    positions: Vec<Real>,
    split: Real,
    normalized: Vec<Vec<Real>>,
    tolerance: Real,
}

impl<'a> Reduction<'a> {
    fn new(knots: &'a [Knot], width: usize, tolerance: Real, known_scale: Real) -> Self {
        let mut indices = Vec::new();
        let mut scales = Vec::new();
        let mut error_weights = Vec::new();
        let active_width = knots
            .iter()
            .map(|knot| knot.wave.gains.len() + 1)
            .max()
            .unwrap_or(1);
        for index in 0..active_width {
            let scale = knots
                .iter()
                .map(|knot| knot.wave.component(index).abs())
                .fold(0., Real::max);
            if scale != 0. {
                indices.push(index);
                scales.push(scale);
                error_weights.push(if index == 0 {
                    scale / scale.max(known_scale)
                } else {
                    1.
                });
            }
        }
        let first = knots[0].time;
        let last = knots[knots.len() - 1].time;
        let duration = last - first;
        let split = (first.midpoint(last) - first) / duration;
        let positions = knots
            .iter()
            .map(|knot| (knot.time - first) / duration)
            .collect();
        let normalized = knots
            .iter()
            .map(|knot| {
                indices
                    .iter()
                    .zip(&scales)
                    .map(|(&index, &scale)| knot.wave.component(index) / scale)
                    .collect()
            })
            .collect();
        Self {
            knots,
            width,
            indices,
            scales,
            error_weights,
            positions,
            split,
            normalized,
            tolerance,
        }
    }

    fn endpoints(&self) -> (&[Real], &[Real]) {
        (
            &self.normalized[0],
            &self.normalized[self.normalized.len() - 1],
        )
    }

    fn midpoint(&self) -> Vec<Real> {
        let upper = self.positions.partition_point(|&x| x < self.split);
        let lower = upper - 1;
        let fraction =
            (self.split - self.positions[lower]) / (self.positions[upper] - self.positions[lower]);
        self.normalized[lower]
            .iter()
            .zip(&self.normalized[upper])
            .map(|(&a, &b)| chord(a, b, fraction))
            .collect()
    }

    fn error(&self, correction: &[Real], middle: &[Real]) -> (Real, usize) {
        let (a, b) = self.endpoints();
        let mut error = 0.;
        let mut split = self.knots.len() / 2;
        for (k, (&x, values)) in self
            .positions
            .iter()
            .zip(&self.normalized)
            .enumerate()
            .skip(1)
            .take(self.knots.len() - 2)
        {
            for i in 0..a.len() {
                let actual = correction[i].mul_add(tent(x, self.split), chord(a[i], b[i], x));
                let difference = (actual - values[i]).abs() * self.error_weights[i];
                if difference > error {
                    error = difference;
                    split = k;
                }
            }
        }
        for i in 0..a.len() {
            let difference = (correction[i] + chord(a[i], b[i], self.split) - middle[i]).abs()
                * self.error_weights[i];
            error = error.max(difference);
        }
        (error, split)
    }

    fn gram(&self) -> Vec<Real> {
        let order = self.indices.len();
        let mut gram = vec![0.; order * order];
        for (x, y) in self.positions.windows(2).zip(self.normalized.windows(2)) {
            let duration = x[1] - x[0];
            for i in 0..order {
                for j in i..order {
                    let value = product_integral(y[0][i], y[1][i], y[0][j], y[1][j]);
                    gram[i * order + j] = duration.mul_add(value, gram[i * order + j]);
                    gram[j * order + i] = gram[i * order + j];
                }
            }
        }
        gram
    }

    fn tent_projection(&self, middle: &[Real]) -> Vec<Real> {
        let (a, b) = self.endpoints();
        let mut dot = vec![0.; a.len()];
        for (x, y) in self.positions.windows(2).zip(self.normalized.windows(2)) {
            let mut integrate = |left: Real, right: Real, first: &[Real], last: &[Real]| {
                for i in 0..a.len() {
                    let r1 = first[i] - chord(a[i], b[i], left);
                    let r2 = last[i] - chord(a[i], b[i], right);
                    dot[i] = (right - left).mul_add(
                        product_integral(r1, r2, tent(left, self.split), tent(right, self.split)),
                        dot[i],
                    );
                }
            };
            if x[0] < self.split && x[1] > self.split {
                integrate(x[0], self.split, &y[0], middle);
                integrate(self.split, x[1], middle, &y[1]);
            } else {
                integrate(x[0], x[1], &y[0], &y[1]);
            }
        }
        for value in &mut dot {
            *value *= 3.;
        }
        dot
    }

    fn contractive(&self, gram: &[Real], correction: &[Real]) -> bool {
        let (a, b) = self.endpoints();
        let order = a.len();
        let mut difference = gram.to_vec();
        for i in 0..order {
            for j in i..order {
                let base = product_integral(a[i], b[i], a[j], b[j]);
                let cross = chord_tent(a[j], b[j], self.split).mul_add(
                    correction[i],
                    chord_tent(a[i], b[i], self.split) * correction[j],
                );
                let energy = base + cross + correction[i] * correction[j] / 3.;
                let value = difference[i * order + j] - energy;
                difference[i * order + j] = value;
                difference[j * order + i] = value;
            }
        }
        positive_semidefinite(&difference, order)
    }

    // Integrate the difference of the exact and reduced wave outer
    // products over every prefix. On each PWL segment this is a cubic
    // Gram polynomial. Certifying every prefix, not just the complete
    // interval, prevents the reducer from borrowing future wave energy.
    fn prefix_contracts(&self, correction: &[Real], middle: &[Real]) -> bool {
        let (begin, finish) = self.endpoints();
        let order = begin.len();
        let mut mass = vec![0.; order * order];
        let mut integrate = |left: Real, right: Real, first: &[Real], last: &[Real]| {
            let duration = (right - left) / 3.;
            if order == 1 {
                let approx_first =
                    correction[0].mul_add(tent(left, self.split), chord(begin[0], finish[0], left));
                let approx_last = correction[0]
                    .mul_add(tent(right, self.split), chord(begin[0], finish[0], right));
                let c0 = mass[0];
                let c1 =
                    duration.mul_add(first[0].mul_add(first[0], -approx_first * approx_first), c0);
                let c2 =
                    duration.mul_add(first[0].mul_add(last[0], -approx_first * approx_last), c1);
                let c3 = duration.mul_add(last[0].mul_add(last[0], -approx_last * approx_last), c2);
                mass[0] = c3;
                return scalar_bernstein([c0, c1, c2, c3], 8);
            }
            let approximated = |time: Real| -> Vec<Real> {
                (0..order)
                    .map(|i| {
                        correction[i]
                            .mul_add(tent(time, self.split), chord(begin[i], finish[i], time))
                    })
                    .collect()
            };
            let approx_first = approximated(left);
            let approx_last = approximated(right);
            let mut coefficients = [mass.clone(), mass.clone(), mass.clone(), mass.clone()];
            for i in 0..order {
                for j in i..order {
                    let entry = i * order + j;
                    let transpose = j * order + i;
                    let q0 = first[i].mul_add(first[j], -approx_first[i] * approx_first[j]);
                    let q1 = 0.5
                        * approx_last[i].mul_add(
                            -approx_first[j],
                            approx_first[i].mul_add(
                                -approx_last[j],
                                last[i].mul_add(first[j], first[i] * last[j]),
                            ),
                        );
                    let q2 = last[i].mul_add(last[j], -approx_last[i] * approx_last[j]);
                    let c1 = duration.mul_add(q0, mass[entry]);
                    let c2 = duration.mul_add(q1, c1);
                    let c3 = duration.mul_add(q2, c2);
                    for (matrix, value) in coefficients[1..].iter_mut().zip([c1, c2, c3]) {
                        matrix[entry] = value;
                        matrix[transpose] = value;
                    }
                }
            }
            let valid = bernstein_psd(&coefficients, order, 8);
            mass.clone_from(&coefficients[3]);
            valid
        };
        for (approx_first, approx_last) in self.positions.windows(2).zip(self.normalized.windows(2))
        {
            if approx_first[0] < self.split && approx_first[1] > self.split {
                if !integrate(approx_first[0], self.split, &approx_last[0], middle)
                    || !integrate(self.split, approx_first[1], middle, &approx_last[1])
                {
                    return false;
                }
            } else if !integrate(
                approx_first[0],
                approx_first[1],
                &approx_last[0],
                &approx_last[1],
            ) {
                return false;
            }
        }
        true
    }

    fn scalar_correction(&self, correction: Real, middle: &[Real]) -> Option<Real> {
        // For a same-sign scalar interval, the tent can be moved into
        // the exact waveform's pointwise magnitude envelope. This is
        // stronger than prefix contraction and avoids repeated Gram
        // searches for the overwhelmingly common accepted-wave case.
        let sign = if self.normalized.iter().all(|value| value[0] >= 0.) {
            1.
        } else if self.normalized.iter().all(|value| value[0] <= 0.) {
            -1.
        } else {
            return None;
        };
        let (a, b) = self.endpoints();
        let mut lower = Real::NEG_INFINITY;
        let mut upper = Real::INFINITY;
        for (x, value) in self
            .positions
            .iter()
            .copied()
            .zip(self.normalized.iter().map(Vec::as_slice))
            .chain(std::iter::once((self.split, middle)))
        {
            if x <= 0. || x >= 1. {
                continue;
            }
            let base = sign * chord(a[0], b[0], x);
            let weight = tent(x, self.split);
            lower = lower.max(-base / weight);
            upper = upper.min((sign.mul_add(value[0], -base)) / weight);
        }
        (lower <= upper).then(|| sign * (sign * correction).clamp(lower, upper))
    }

    fn wave(&self, correction: &[Real]) -> Affine {
        let (a, b) = self.endpoints();
        let mut wave = Affine::zero();
        for (i, (&index, &scale)) in self.indices.iter().zip(&self.scales).enumerate() {
            wave.set(
                index,
                (chord(a[i], b[i], self.split) + correction[i]) * scale,
                self.width,
            );
        }
        wave
    }

    fn choose(&self) -> Choice {
        if self.indices.is_empty() {
            return Choice::Chord;
        }
        if self.split <= 0. || self.split >= 1. {
            return Choice::Split(self.knots.len() / 2);
        }
        let zero = vec![0.; self.indices.len()];
        let middle = self.midpoint();
        let (error, mut split) = self.error(&zero, &middle);
        let gram = self.gram();
        if error <= self.tolerance
            && self.contractive(&gram, &zero)
            && self.prefix_contracts(&zero, &middle)
        {
            return Choice::Chord;
        }
        // A tent retains both endpoints. Its least-squares correction is
        // linear in the affine components. If that correction is not
        // contractive, move towards the minimum-energy tent (also linear)
        // until the Gram inequality is satisfied.
        let optimum = self.tent_projection(&middle);
        let (a, b) = self.endpoints();
        let minimum: Vec<_> = a
            .iter()
            .zip(b)
            .map(|(&a, &b)| -3. * chord_tent(a, b, self.split))
            .collect();
        let correction = if self.indices.len() == 1 {
            let q = chord_tent(a[0], b[0], self.split);
            let base = product_integral(a[0], b[0], a[0], b[0]);
            let discriminant = q.mul_add(q, (gram[0] - base) / 3.);
            if discriminant >= 0. {
                let radius = 3. * discriminant.sqrt();
                Some(vec![
                    optimum[0].clamp(minimum[0] - radius, minimum[0] + radius),
                ])
            } else {
                None
            }
        } else if self.contractive(&gram, &optimum) {
            Some(optimum)
        } else if self.contractive(&gram, &minimum) {
            let mut low = 0.;
            let mut high = 1.;
            let mut candidate = minimum.clone();
            for _ in 0..24 {
                let alpha = Real::midpoint(low, high);
                for ((value, &a), &b) in candidate.iter_mut().zip(&minimum).zip(&optimum) {
                    *value = chord(a, b, alpha);
                }
                if self.contractive(&gram, &candidate) {
                    low = alpha;
                } else {
                    high = alpha;
                }
            }
            Some(
                minimum
                    .iter()
                    .zip(optimum)
                    .map(|(&a, b)| chord(a, b, low))
                    .collect(),
            )
        } else {
            None
        };
        if let Some(mut correction) = correction {
            let (error, at) = self.error(&correction, &middle);
            split = at;
            if error <= self.tolerance && self.knots.len() > 3 {
                if self.prefix_contracts(&correction, &middle) {
                    return Choice::Tent(self.wave(&correction));
                }
                if correction.len() == 1 {
                    if let Some(value) = self.scalar_correction(correction[0], &middle) {
                        correction[0] = value;
                        if self.error(&correction, &middle).0 <= self.tolerance
                            && self.prefix_contracts(&correction, &middle)
                        {
                            return Choice::Tent(self.wave(&correction));
                        }
                    }
                } else if self.prefix_contracts(&minimum, &middle) {
                    let mut low = 0.;
                    let mut high = 1.;
                    let mut candidate = minimum.clone();
                    for _ in 0..16 {
                        let alpha = Real::midpoint(low, high);
                        for ((value, &a), &b) in candidate.iter_mut().zip(&minimum).zip(&correction)
                        {
                            *value = chord(a, b, alpha);
                        }
                        if self.prefix_contracts(&candidate, &middle) {
                            low = alpha;
                        } else {
                            high = alpha;
                        }
                    }
                    for ((value, &a), &b) in candidate.iter_mut().zip(&minimum).zip(&correction) {
                        *value = chord(a, b, low);
                    }
                    if self.error(&candidate, &middle).0 <= self.tolerance
                        && self.prefix_contracts(&candidate, &middle)
                    {
                        return Choice::Tent(self.wave(&candidate));
                    }
                }
            }
        }
        Choice::Split(split)
    }
}

enum Choice {
    Chord,
    Tent(Affine),
    Split(usize),
}

// Iterative subdivision rather than unbounded recursion. Every split
// removes at least one interior knot from the next candidate interval.
pub(super) fn reduce(
    knots: &[Knot],
    width: usize,
    tolerance: Real,
    known_scale: Real,
    output: &mut Vec<Knot>,
) {
    if knots.len() < 2 {
        return;
    }
    let mut pending = vec![(0, knots.len() - 1)];
    while let Some((first, last)) = pending.pop() {
        if last <= first + 1 {
            output.push(knots[last].clone());
            continue;
        }
        let candidate = Reduction::new(&knots[first..=last], width, tolerance, known_scale);
        match candidate.choose() {
            Choice::Chord => output.push(knots[last].clone()),
            Choice::Tent(wave) => {
                let time = knots[first].time.midpoint(knots[last].time);
                if time > knots[first].time && time < knots[last].time {
                    output.push(Knot { time, wave });
                    output.push(knots[last].clone());
                } else {
                    output.extend_from_slice(&knots[first + 1..=last]);
                }
            }
            Choice::Split(index) => {
                let split = first + index.clamp(1, last - first - 1);
                pending.push((split, last));
                pending.push((first, split));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::sample;
    use super::*;

    #[test]
    fn representable_peak_uses_its_actual_fraction_in_energy_integrals() {
        let origin = 1_f64;
        let times: Vec<_> = (0..=7)
            .map(|offset| Real::from_bits(origin.to_bits() + offset))
            .collect();
        let knots: Vec<_> = times
            .iter()
            .enumerate()
            .map(|(index, &time)| {
                let fraction = Real::from(u32::try_from(index).unwrap()) / 7.;
                Knot {
                    time,
                    wave: Affine::scalar(fraction * fraction),
                }
            })
            .collect();
        let candidate = Reduction::new(&knots, 1, 1., 0.);
        assert_ne!(candidate.split, 0.5);
        let peak = candidate.split;
        let correction = [0.2];
        let a = candidate.normalized[0][0];
        let b = candidate.normalized[knots.len() - 1][0];
        let height = chord(a, b, peak) + correction[0];
        let integrated = peak.mul_add(
            product_integral(a, height, a, height),
            (1. - peak) * product_integral(height, b, height, b),
        );
        let formula = product_integral(a, b, a, b)
            + 2. * chord_tent(a, b, peak) * correction[0]
            + correction[0] * correction[0] / 3.;
        assert!((integrated - formula).abs() < 1e-14);
    }

    #[test]
    fn whole_interval_energy_is_not_a_prefix_certificate() {
        let knots =
            [(0., 0.), (0.25, 0.), (0.5, 1.), (0.75, 2.), (1., 1.)].map(|(time, value)| Knot {
                time,
                wave: Affine::scalar(value),
            });
        let reduction = Reduction::new(&knots, 1, 1., 0.);
        let correction = [0.];
        assert!(reduction.contractive(&reduction.gram(), &correction));
        assert!(!reduction.prefix_contracts(&correction, &reduction.midpoint()));
    }

    #[test]
    fn affine_reduction_preserves_energy_for_arbitrary_endpoint_combinations() {
        // Two independent columns, with a deliberately cancelling
        // linear combination. Energy must contract for that combination
        // too, not just for each column in isolation.
        let knots: Vec<_> = (0..=64)
            .map(|i| {
                let time = Real::from(i) / 64.;
                let shape = time * (1. - time);
                let mut wave = Affine::scalar(time + 0.002 * shape);
                wave.set(1, time + 0.001 * shape, 3);
                wave.set(2, 1. - time + 0.003 * shape, 3);
                Knot { time, wave }
            })
            .collect();
        check_affine_reduction(&knots, false);
        let proportional: Vec<_> = knots
            .iter()
            .map(|knot| {
                let mut wave = knot.wave.clone();
                wave.set(1, wave.known, 3);
                wave.set(2, -2. * wave.known, 3);
                Knot {
                    time: knot.time,
                    wave,
                }
            })
            .collect();
        check_affine_reduction(&proportional, true);
    }

    fn check_affine_reduction(knots: &[Knot], must_reduce: bool) {
        let mut reduced = vec![knots[0].clone()];
        reduce(knots, 3, 1e-3, 0., &mut reduced);
        if must_reduce {
            assert!(reduced.len() < knots.len());
        }
        assert_eq!(reduced.first(), knots.first());
        assert_eq!(reduced.last(), knots.last());
        let mut times: Vec<_> = knots.iter().chain(&reduced).map(|knot| knot.time).collect();
        times.extend((0..=1024).map(|i| Real::from(i) / 1024.));
        super::super::sort_times(&mut times);
        for delta in [[-1., 0.], [1., 1.], [-4., 0.3], [0.7, -1.2], [0., 0.]] {
            let mut exact_energy = 0.;
            let mut reduced_energy = 0.;
            for times in times.windows(2) {
                let [a, b] = [times[0], times[1]];
                let x = sample(knots, a, 0.).unwrap().evaluate(&delta);
                let y = sample(knots, b, 0.).unwrap().evaluate(&delta);
                let u = sample(&reduced, a, 0.).unwrap().evaluate(&delta);
                let v = sample(&reduced, b, 0.).unwrap().evaluate(&delta);
                exact_energy += (b - a) * product_integral(x, y, x, y);
                reduced_energy += (b - a) * product_integral(u, v, u, v);
                assert!(reduced_energy <= exact_energy + 1e-12);
                assert!((u - x).abs() <= 1e-3 * (1. + delta[0].abs() + delta[1].abs()));
            }
        }
    }
}
