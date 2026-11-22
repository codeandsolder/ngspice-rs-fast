//! TXL coefficient construction, following `txlsetup.c`'s `ReadTxL`,
//! `y_pade`, `exp_pade`, `get_h3` and `update_h1C_c`.
//! Local arrays replace C's file-global scratch variables. Setup failures
//! return contextual errors instead of terminating the process.
#![expect(
    clippy::too_many_lines,
    reason = "a complete translated C numerical routine remains contiguous for equation parity review"
)]
#![expect(
    clippy::many_single_char_names,
    reason = "physical-model equation symbols are kept consistent with the ngspice C reference"
)]

use crate::primitives::{Complex, Real, SpiceError, SpiceResult};

#[derive(Debug, Clone, Copy, Default)]
pub(super) struct Term {
    pub c: Real,
    pub x: Real,
}
#[derive(Debug, Clone, Copy)]
pub(super) struct ComplexTerm {
    pub residue: Complex,
    pub pole: Complex,
}

#[derive(Debug, Clone)]
pub(super) enum Propagation<const N: usize, const R: usize> {
    Real([Term; N]),
    Conjugate { real: [Term; R], pair: ComplexTerm },
}
impl<const N: usize, const R: usize> Propagation<N, R> {
    fn from_c(terms: [Term; N], pair: bool) -> Self {
        if pair {
            Self::Conjugate {
                real: std::array::from_fn(|i| terms[i]),
                pair: ComplexTerm {
                    residue: Complex::new(terms[R].c, terms[R + 1].c),
                    pole: Complex::new(terms[R].x, terms[R + 1].x),
                },
            }
        } else {
            Self::Real(terms)
        }
    }

    // C sums the scalar storage slots, including the imaginary residue.
    pub(super) fn residue_sum(&self) -> Real {
        match self {
            Self::Real(terms) => terms.iter().map(|t| t.c).sum(),
            Self::Conjugate { real, pair } => real
                .iter()
                .map(|t| t.c)
                .chain([pair.residue.re, pair.residue.im])
                .sum(),
        }
    }
}

#[derive(Debug, Clone)]
pub(super) struct LossyPade {
    pub h1: [Term; 3],
    pub h2: Propagation<3, 1>,
    pub h3: Propagation<6, 4>,
    pub h1c: Real,
}

#[derive(Debug, Clone)]
pub(super) struct Pade {
    pub admit: Real,
    /// Delay in picoseconds (C's `taul`), not rounded to integer ticks.
    pub delay_ps: Real,
    pub lossy: Option<LossyPade>,
    pub atten2: Real,
    pub atten3: Real,
}
fn error(message: &str) -> SpiceError {
    SpiceError::Numerical {
        context: "TXL txlsetup.c Padé coefficients".into(),
        message: message.into(),
    }
}
fn eval(a: Real, b: Real, c: Real, x: Real) -> Real {
    a.mul_add(x, b).mul_add(x, c)
}

/// The two 3x4 Gauss-Jordan solves of `Gaussian_Elimination1/2`, including
/// their different absolute pivot limits and partial-row swap order.
///
/// The loops stay index-based because this is C's row/column algorithm: the
/// row swap uses the pivot row that partial pivoting just chose, and the
/// elimination reads the row it is updating.
fn eliminate(mut a: [[Real; 4]; 3], threshold: Real) -> SpiceResult<[Real; 3]> {
    for i in 0..3 {
        let mut pivot = i;
        for j in i + 1..3 {
            if a[j][i].abs() > a[pivot][i].abs() {
                pivot = j;
            }
        }
        if !a[pivot][i].is_finite() || a[pivot][i].abs() < threshold {
            return Err(error(
                "Gaussian elimination cannot choose a pivot; check model parameters",
            ));
        }
        if pivot != i {
            for k in i..4 {
                let value = a[i][k];
                a[i][k] = a[pivot][k];
                a[pivot][k] = value;
            }
        }
        let f = 1. / a[i][i];
        a[i][i] = 1.;
        for value in &mut a[i][i + 1..] {
            *value *= f;
        }
        for j in 0..3 {
            if i == j {
                continue;
            }
            let f = a[j][i];
            a[j][i] = 0.;
            for k in i + 1..4 {
                a[j][k] = (-f).mul_add(a[i][k], a[j][k]);
            }
        }
    }
    Ok([a[2][3], a[1][3], a[0][3]])
}
fn solve_series(
    b: [Real; 5],
    dc: Real,
    dc_complement: Real,
    threshold: Real,
) -> SpiceResult<([Real; 3], [Real; 3])> {
    let [b1, b2, b3, b4, b5] = b;
    let p = eliminate(
        [
            [dc_complement, b1, b2, -b3],
            [b1, b2, b3, -b4],
            [b2, b3, b4, -b5],
        ],
        threshold,
    )?;
    let [p1, p2, p3] = p;
    Ok((p, [p1 + b1, b1.mul_add(p1, p2) + b2, p3 * dc]))
}
/// Taylor coefficients of sqrt((1+a z)/(1+b z)). Differentiating the
/// defining function gives
/// 2(1+a z)(1+b z)y' = (a-b)y.
/// Equating powers avoids dividing any derivative by a-b, which becomes
/// arbitrarily ill-conditioned near a matched lossy line.
fn admittance_series(a: Real, b: Real) -> [Real; 5] {
    let delta = a - b;
    let sum = a + b;
    let product = a * b;
    let mut y = [0.; 6];
    y[0] = 1.;
    y[1] = 0.5 * delta;
    for n in 2_u16..=5 {
        let i = usize::from(n);
        let previous = Real::from(n - 1);
        let coefficient = (-2. * previous).mul_add(sum, delta);
        let second = -2. * Real::from(n - 2) * product;
        y[i] = coefficient.mul_add(y[i - 1], second * y[i - 2]) / (2. * Real::from(n));
    }
    std::array::from_fn(|i| y[i + 1])
}

/// Cardano seed followed by Newton refinement and stable quadratic
/// deflation. Solve in a dimensionless coordinate so neither the cubic
/// discriminant nor its residual need overflow for large physical rates.
fn roots([a, b, c]: [Real; 3], propagation: bool) -> SpiceResult<([Real; 3], bool)> {
    let scale = a.abs().max(b.abs().sqrt()).max(c.abs().cbrt());
    if !scale.is_finite() || scale == 0. {
        return Err(error("degenerate cubic scale"));
    }
    let a = a / scale;
    let b = (b / scale) / scale;
    let c = ((c / scale) / scale) / scale;
    let q = a.mul_add(a, -3. * b) / 9.;
    let p = (2. * a * a).mul_add(a, (-9. * a).mul_add(b, 27. * c)) / 54.;
    let d = (q * q).mul_add(q, -p * p);
    let mut x = if q == 0. && p == 0. {
        -a / 3.
    } else if d >= 0. {
        let root = q.sqrt();
        let angle = (p / (q * root)).clamp(-1., 1.).acos() / 3.;
        (-2. * root).mul_add(angle.cos(), -a / 3.)
    } else if p >= 0. {
        let t = ((-d).sqrt() + p).cbrt();
        -(t + q / t) - a / 3.
    } else {
        let t = ((-d).sqrt() - p).cbrt();
        t + q / t - a / 3.
    };
    // Retain the C choice of the real pole and bounded refinement.
    // Accept only updates that improve the actual polynomial residual.
    let residual = |x: Real| (x + a).mul_add(x, b).mul_add(x, c);
    for _ in 0..31 {
        let current = residual(x);
        let derivative = 3_f64.mul_add(x, 2. * a).mul_add(x, b);
        let next = x - current / derivative;
        if !next.is_finite() || residual(next).abs() >= current.abs() {
            break;
        }
        let converged = (next - x).abs() <= 2. * Real::EPSILON * next.abs();
        x = next;
        if converged {
            break;
        }
    }
    let quadratic_a = a + x;
    let quadratic_b = if x == 0. { b } else { -c / x };
    let d = quadratic_a.mul_add(quadratic_a, -4. * quadratic_b);
    let pair = d < 0.;
    let (y, z) = if pair {
        if !propagation {
            return Err(error(
                "two imaginary characteristic-admittance roots (C rejects this model)",
            ));
        }
        (-0.5 * quadratic_a, 0.5 * (-d).sqrt())
    } else {
        let d = d.sqrt();
        let y = -quadratic_a.midpoint(d.copysign(quadratic_a));
        let z = if y == 0. { 0. } else { quadratic_b / y };
        (y, z)
    };
    let roots = [x, y, z].map(|root| root * scale);
    if roots.iter().any(|x| !x.is_finite()) {
        return Err(error("nonfinite cubic roots"));
    }
    Ok((roots, pair))
}
fn residues(p: [Real; 3], q: [Real; 3], x: [Real; 3], pair: bool) -> [Real; 3] {
    let [p1, p2, p3] = p;
    let [a, b, c] = [q[0] - p1, q[1] - p2, q[2] - p3];
    let mut r = x.map(|x| eval(a, b, c, x) / eval(3., 2. * p1, p2, x));
    if pair {
        let z = Complex::new(x[1], x[2]);
        let num = Complex::real(a) * z * z + Complex::real(b) * z + Complex::real(c);
        let den = Complex::real(3.) * z * z + Complex::real(2. * p1) * z + Complex::real(p2);
        let v = num / den;
        r[1] = v.re;
        r[2] = v.im;
    }
    r
}
impl Pade {
    pub(super) fn new(r: Real, l: Real, g: Real, c: Real, length: Real) -> SpiceResult<Self> {
        let l = l.max(1e-12);
        let admit = (c / l).sqrt();
        let tau = (l * c).sqrt();
        let delay_ps = tau * length * 1e12;
        if !admit.is_finite() || admit <= 0. || !delay_ps.is_finite() || delay_ps <= 0. {
            return Err(error("nonfinite or degenerate line admittance/delay"));
        }
        if r / l < 5e5 && g < 1e-2 {
            return Ok(Self {
                admit,
                delay_ps,
                lossy: None,
                atten2: 1.,
                atten3: admit,
            });
        }
        // C-shaped temporary arrays are only used during coefficient construction.
        let mut h1 = [Term::default(); 3];
        let mut h2 = [Term::default(); 3];
        let mut h3 = [Term::default(); 6];
        let mut h1c = 0.;
        // Characteristic-admittance series in the reciprocal-frequency
        // variable. Preserve C's absolute pivot acceptance policy, but
        // compute the coefficients without the singular derivative formula.
        let a = g / c;
        let b = r / l;
        let delta = a - b;
        if !a.is_finite() || !b.is_finite() {
            return Err(error("nonfinite line rates"));
        }
        if delta == 0. {
            // Heaviside's distortionless condition R/L = G/C.
            // Z0 is constant and propagation is exp(-a*tau*length)
            // times the pure delay. No Padé poles or residues are needed.
            let attenuation = (-a * tau * length).exp();
            return Ok(Self {
                admit,
                delay_ps,
                lossy: None,
                atten2: attenuation,
                atten3: attenuation * admit,
            });
        }
        let dc = a.sqrt() / b.sqrt();
        let dc_complement = if dc > 0.5 && dc < 2. {
            (-delta / b) / (1. + dc)
        } else {
            1. - dc
        };
        let (p, q) = solve_series(admittance_series(a, b), dc, dc_complement, 1e-16)?;
        let (x, _) = roots(p, false)?;
        let cs = residues(p, q, x, false);
        for i in 0..3 {
            h1[i] = Term { c: cs[i], x: x[i] };
        }
        // exp_pade(): expansion of exp(-length*sqrt((sL+R)(sC+G))).
        let y1 = Real::midpoint(a, b);
        // a*b - midpoint(a,b)^2 = -((a-b)/2)^2 exactly.
        let half_delta = 0.5 * delta;
        let y2 = -half_delta * half_delta;
        let y3 = -3. * y1 * y2;
        let y4 = (-3. * y2).mul_add(y2, -4. * y1 * y3);
        let y5 = (-5. * y1).mul_add(y4, -10. * y2 * y3);
        let y6 = (-10. * y3).mul_add(y3, (-15. * y2).mul_add(y4, -6. * y1 * y5));
        let a0 = y1 * tau * length;
        let mut aa = [0.; 6];
        let mut power = tau * tau;
        for (i, (v, factor)) in [(y2, 2.), (y3, 6.), (y4, 24.), (y5, 120.), (y6, 720.)]
            .into_iter()
            .enumerate()
        {
            aa[i + 1] = -v * power / factor * length;
            power *= tau;
        }
        let mut bb = [0.; 6];
        bb[0] = 1.;
        bb[1] = aa[1];
        for i in 2_u16..6 {
            let row = usize::from(i);
            for j in 1..=i {
                let col = usize::from(j);
                bb[row] = (Real::from(j) * aa[col]).mul_add(bb[row - col], bb[row]);
            }
            bb[row] /= Real::from(i);
        }
        // midpoint(a,b)-sqrt(a*b) = (sqrt(a)-sqrt(b))^2/2.
        // Rationalize the root difference as well, so cases near a=b
        // retain their small propagation correction.
        let root_difference = delta / (a.sqrt() + b.sqrt());
        let exponent = 0.5 * root_difference * root_difference * tau * length;
        let dc = exponent.exp();
        let (mut p, mut q) = solve_series(
            [bb[1], bb[2], bb[3], bb[4], bb[5]],
            dc,
            -exponent.exp_m1(),
            1e-28,
        )?;
        for (i, divisor) in [tau, tau * tau, tau * tau * tau].into_iter().enumerate() {
            p[i] /= divisor;
            q[i] /= divisor;
        }
        let (x, pair) = roots(p, true)?;
        let cs = residues(p, q, x, pair);
        for i in 0..3 {
            h2[i] = Term { c: cs[i], x: x[i] };
        }
        let atten2 = (-a0).exp();
        let atten3 = atten2 * admit;
        // get_h3(): residues of the product H1*H2, before attenuation.
        for i in 0..3 {
            let t = h1[i];
            let mut sum = 0.;
            if pair {
                let [u, v, w] = h2;
                let single_pole = u.c / (t.x - u.x);
                let difference = t.x - v.x;
                let pair_numerator = v.c.mul_add(difference, -w.x * w.c);
                let denominator = difference.mul_add(difference, w.x * w.x);
                sum += 2_f64.mul_add(pair_numerator / denominator, single_pole);
            } else {
                for u in h2 {
                    sum += u.c / (t.x - u.x);
                }
            }
            h3[i] = Term {
                x: t.x,
                c: t.c.mul_add(sum, t.c),
            };
        }
        for i in 0..3 {
            let t = h2[i];
            let mut sum = 0.;
            for u in h1 {
                sum += u.c / (t.x - u.x);
            }
            h3[i + 3] = Term {
                x: t.x,
                c: t.c.mul_add(sum, t.c),
            };
        }
        if pair {
            let c = Complex::new(h2[1].c, h2[2].c);
            let x = Complex::new(h2[1].x, h2[2].x);
            let mut res = c;
            for u in h1 {
                res = res + (c / (x - Complex::real(u.x))) * Complex::real(u.c);
            }
            h3[4].c = res.re;
            h3[5].c = res.im;
        }
        for t in &mut h1 {
            t.c *= admit;
            h1c += t.c;
        }
        for t in &mut h2 {
            t.c *= atten2;
        }
        for t in &mut h3 {
            t.c *= atten3;
        }
        if h1
            .iter()
            .chain(&h2)
            .chain(&h3)
            .any(|t| !t.c.is_finite() || !t.x.is_finite() || t.x == 0.)
            || !h1c.is_finite()
        {
            return Err(error("nonfinite or degenerate Padé coefficients"));
        }
        Ok(Self {
            admit,
            delay_ps,
            lossy: Some(LossyPade {
                h1,
                h2: Propagation::from_c(h2, pair),
                h3: Propagation::from_c(h3, pair),
                h1c,
            }),
            atten2,
            atten3,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn characteristic_admittance_series_matches_binomial_and_nearly_equal_rates() {
        assert_eq!(admittance_series(0., 2.), [-1., 1.5, -2.5, 4.375, -7.875]);
        assert_eq!(admittance_series(2., 2.), [0.; 5]);
        let a = 2_f64.next_down();
        let b = 2.;
        let delta = a - b;
        let expected = [0.5, -1., 2., -4., 8.].map(|factor| factor * delta);
        for (actual, expected) in admittance_series(a, b).into_iter().zip(expected) {
            assert!((actual - expected).abs() <= 8. * Real::EPSILON * expected.abs());
        }
    }

    #[test]
    fn cubic_poles_have_small_residuals_across_physical_scales() {
        for scale in [1e-100, 1., 1e100] {
            let coefficients = [6. * scale, 11. * scale * scale, 6. * scale * scale * scale];
            let (poles, pair) = roots(coefficients, false).unwrap();
            assert!(!pair);
            let mut normalized = poles.map(|pole| pole / scale);
            normalized.sort_by(Real::total_cmp);
            for (pole, expected) in normalized.into_iter().zip([-3., -2., -1.]) {
                assert!((pole - expected).abs() <= 32. * Real::EPSILON);
            }
        }
        // (x + 2)(x^2 + 2x + 5), with poles -2 and -1 ± 2i.
        let (poles, pair) = roots([4., 9., 10.], true).unwrap();
        assert!(pair);
        for (actual, expected) in poles.into_iter().zip([-2., -1., 2.]) {
            assert!((actual - expected).abs() <= 32. * Real::EPSILON);
        }
    }

    #[test]
    fn matched_loss_rates_use_the_analytic_attenuated_delay() {
        let model = Pade::new(1., 1e-8, 0.01, 1e-10, 2.).unwrap();
        assert!(model.lossy.is_none());
        assert!((model.delay_ps - 2000.).abs() < 1e-12);
        assert!((model.atten2 - (-0.2_f64).exp()).abs() <= 2. * Real::EPSILON);
        assert_eq!(model.atten3, model.atten2 * model.admit);
    }

    #[test]
    fn representative_pade_has_finite_nodes_and_attenuation() {
        let model = Pade::new(12.45, 8.972e-9, 0., 0.468e-12, 16.).unwrap();
        assert!(model.lossy.is_some());
        assert!(model.delay_ps > 0.);
        assert!(model.atten2 > 0. && model.atten2 < 1.);
        assert_eq!(model.atten3, model.atten2 * model.admit);
    }
}
