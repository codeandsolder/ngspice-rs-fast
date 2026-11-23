//! TXL coefficient construction, following `txlsetup.c`'s `ReadTxL`,
//! `y_pade`, `exp_pade`, `get_h3` and `update_h1C_c`.
//! Local arrays replace C's file-global scratch variables. Setup failures
//! return contextual errors instead of terminating the process.
// C-compatible numerical expressions are kept in their reference order:
// Clippy's fused multiply-add suggestions change IEEE-754 rounding.
#![expect(
    clippy::manual_midpoint,
    reason = "C source midpoint arithmetic retains its original evaluation and rounding order"
)]
#![expect(
    clippy::too_many_lines,
    reason = "a complete translated C numerical routine remains contiguous for equation parity review"
)]
#![expect(
    clippy::many_single_char_names,
    reason = "physical-model equation symbols are kept consistent with the ngspice C reference"
)]
#![expect(
    clippy::imprecise_flops,
    reason = "the original C floating-point algorithm and libm operation sequence must remain intact"
)]
#![expect(
    clippy::suboptimal_flops,
    reason = "SPICE numerical reference parity requires the original non-fused C operation order"
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
    a * x * x + b * x + c
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
                a[j][k] -= f * a[i][k];
            }
        }
    }
    Ok([a[2][3], a[1][3], a[0][3]])
}
fn solve_series(b: [Real; 5], dc: Real, threshold: Real) -> SpiceResult<([Real; 3], [Real; 3])> {
    let [b1, b2, b3, b4, b5] = b;
    let p = eliminate(
        [[1. - dc, b1, b2, -b3], [b1, b2, b3, -b4], [b2, b3, b4, -b5]],
        threshold,
    )?;
    let [p1, p2, p3] = p;
    Ok((p, [p1 + b1, b1 * p1 + p2 + b2, p3 * dc]))
}
/// `find_roots`/`exp_find_roots`: Cardano seed, at most 31 Newton updates,
/// and stable quadratic deflation. For complex propagation roots the last
/// two entries hold the real and imaginary part of the conjugate pair.
fn roots([a, b, c]: [Real; 3], propagation: bool) -> SpiceResult<([Real; 3], bool)> {
    let q = (a * a - 3. * b) / 9.;
    let p = (2. * a * a * a - 9. * a * b + 27. * c) / 54.;
    let d = q * q * q - p * p;
    let mut x = if d >= 0. {
        -2. * q.sqrt() * ((p / (q * q.sqrt())).acos() / 3.).cos() - a / 3.
    } else if p > 0. {
        let t = ((-d).sqrt() + p).powf(1. / 3.);
        -(t + q / t) - a / 3.
    } else if p == 0. {
        -a / 3.
    } else {
        let t = ((-d).sqrt() - p).powf(1. / 3.);
        t + q / t - a / 3.
    };
    let backup = x;
    for i in 1..=32 {
        let t = x - (x * x * x + a * x * x + b * x + c) / (3. * x * x + 2. * a * x + b);
        if (t - x).abs() <= 5e-4 || !t.is_finite() {
            break;
        }
        if i == 32 {
            x = backup;
            break;
        }
        x = t;
    }
    let a = a + x;
    let b = -c / x;
    let d = a * a - 4. * b;
    let pair = d < 0.;
    let (y, z) = if pair {
        if !propagation {
            return Err(error(
                "two imaginary characteristic-admittance roots (C rejects this model)",
            ));
        }
        (-0.5 * a, 0.5 * (-d).sqrt())
    } else {
        let scale: Real = if propagation { 1e-16 } else { 1e-18 };
        let d = (d * scale).sqrt() / scale.sqrt();
        let y = if a >= 0. {
            -0.5 * (a + d)
        } else {
            -0.5 * (a - d)
        };
        (y, b / y)
    };
    if [x, y, z].iter().any(|x| !x.is_finite()) {
        return Err(error("nonfinite cubic roots"));
    }
    Ok(([x, y, z], pair))
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
        // Calculate the geometric quantities without allowing an intermediate
        // quotient or product to lose an exponent that the final square root
        // would recover. Keep the ordinary C evaluation when its intermediate
        // is normal: this also avoids needless extra rounding in that range.
        let ratio = c / l;
        let admit = if ratio.is_normal() {
            ratio.sqrt()
        } else {
            c.sqrt() / l.sqrt()
        };
        let product = l * c;
        let tau = if product.is_normal() {
            product.sqrt()
        } else {
            l.sqrt() * c.sqrt()
        };
        // Scaling seconds to picoseconds last can lose an otherwise finite
        // subnormal delay when tau*length underflows. Multiply the length by
        // the unit conversion first only when that intermediate is subnormal
        // or zero. If tau*length overflows, the final delay must overflow too.
        let transit = tau * length;
        let delay_ps = if transit.is_normal() {
            transit * 1e12
        } else {
            tau * (length * 1e12)
        };
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
        // mac(): derivatives of sqrt((1+a z)/(1+b z)), before factorials.
        let a = g / c;
        let b = r / l;
        let delta = a - b;
        let t = 3. * b * b - 2. * a * b - a * a;
        let y1 = 0.5 * delta;
        let y2 = 0.5 * t * y1 / delta;
        let y3 = ((3. * b * b + a * a) * y1 * y1 + 0.5 * t * y2) / delta;
        let y4 = ((3. * b * b - 3. * a * a) * y1 * y1 * y1
            + (9. * b * b + 3. * a * a) * y1 * y2
            + 0.5 * t * y3)
            / delta;
        let y2_squared = y2 * y2;
        let y5 = (12. * a * a * y1 * y1 * y1 * y1
            + y1 * y1 * y2 * (18. * b * b - 18. * a * a)
            + (9. * b * b + 3. * a * a) * (y2_squared + y1 * y3)
            + (3. * b * b + a * a) * y1 * y3
            + 0.5 * t * y4)
            / delta;
        let (p, q) = solve_series(
            [y1, y2 / 2., y3 / 6., y4 / 24., y5 / 120.],
            (a / b).sqrt(),
            1e-16,
        )?;
        let (x, _) = roots(p, false)?;
        let cs = residues(p, q, x, false);
        for i in 0..3 {
            h1[i] = Term { c: cs[i], x: x[i] };
        }
        // exp_pade(): expansion of exp(-length*sqrt((sL+R)(sC+G))).
        let y1 = 0.5 * (a + b);
        let ab = a * b;
        let y1_squared = y1 * y1;
        let y2 = ab - y1_squared;
        let y3 = -3. * y1 * y2;
        let y4 = -3. * y2 * y2 - 4. * y1 * y3;
        let y5 = -5. * y1 * y4 - 10. * y2 * y3;
        let y6 = -10. * y3 * y3 - 15. * y2 * y4 - 6. * y1 * y5;
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
                bb[row] += Real::from(j) * aa[col] * bb[row - col];
            }
            bb[row] /= Real::from(i);
        }
        let dc = (a0 - length * (r * g).sqrt()).exp();
        let (mut p, mut q) = solve_series([bb[1], bb[2], bb[3], bb[4], bb[5]], dc, 1e-28)?;
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
                let v_residue = v.c * t.x;
                let w_residue = w.x * w.c;
                let v_shifted_residue = v.x * v.c;
                let pair_numerator = v_residue - w_residue - v_shifted_residue;
                sum += single_pole
                    + 2. * pair_numerator / (t.x * t.x - 2. * v.x * t.x + v.x * v.x + w.x * w.x);
            } else {
                for u in h2 {
                    sum += u.c / (t.x - u.x);
                }
            }
            h3[i] = Term {
                x: t.x,
                c: t.c + t.c * sum,
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
                c: t.c + t.c * sum,
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
    use super::Pade;

    #[test]
    fn lossless_geometry_matches_independent_high_precision_square_roots() {
        // Reference bits: 750-digit Decimal arithmetic on the exact binary64
        // inputs, then correctly rounded once to binary64. These include
        // overflowing C intermediates, a subnormal ratio, a subnormal
        // product, and a transit product which rounds to zero before the
        // picosecond conversion but must not lose the final delay.
        for (l, c, length, admit_bits, delay_bits) in [
            (
                1e300,
                1e-100,
                1e-100,
                0x1668_7e92_154e_f7ac,
                0x426d_1a94_a200_0000,
            ),
            (
                1e160,
                1e160,
                1e-150,
                0x3ff0_0000_0000_0000,
                0x4480_f0cf_064d_d592,
            ),
            (
                1e-12,
                1e308,
                1.,
                0x6126_c2d4_256f_fcc3,
                0x6126_c2d4_256f_fcc3,
            ),
            (
                1e308,
                f64::from_bits(1),
                1e-300,
                0x0000_0000_02ae_7ad2,
                0x028d_12d4_1afc_a3c3,
            ),
            (
                1e-12,
                f64::from_bits(1),
                1e-164,
                0x1f9e_8480_0000_0000,
                0x0000_0000_0000_1193,
            ),
            (
                1e308,
                1e308,
                1e-308,
                0x3ff0_0000_0000_0000,
                0x426d_1a94_a1ff_ffff,
            ),
            (
                1e-12,
                1e308,
                1e-308,
                0x6126_c2d4_256f_fcc3,
                0x2134_756c_cb01_abfb,
            ),
        ] {
            let coefficients = Pade::new(0., l, 0., c, length).unwrap();
            assert!(coefficients.lossy.is_none());
            assert_eq!(coefficients.atten2.to_bits(), 1_f64.to_bits());
            assert_eq!(coefficients.atten3.to_bits(), coefficients.admit.to_bits());
            assert!(
                coefficients.admit.to_bits().abs_diff(admit_bits) <= 2,
                "L={l:e} C={c:e}: admittance {:016x} vs {admit_bits:016x}",
                coefficients.admit.to_bits()
            );
            assert!(
                coefficients.delay_ps.to_bits().abs_diff(delay_bits) <= 2,
                "L={l:e} C={c:e} length={length:e}: delay {:016x} vs {delay_bits:016x}",
                coefficients.delay_ps.to_bits()
            );
        }
    }

    #[test]
    fn ordinary_geometry_retains_the_original_c_evaluation() {
        let l = 250e-9;
        let c = 100e-12;
        let length = 1.;
        let coefficients = Pade::new(0., l, 0., c, length).unwrap();
        assert_eq!(coefficients.admit.to_bits(), (c / l).sqrt().to_bits());
        assert_eq!(
            coefficients.delay_ps.to_bits(),
            ((l * c).sqrt() * length * 1e12).to_bits()
        );
    }

    #[test]
    fn unrepresentable_final_quantities_still_fail() {
        for (l, c, length) in [
            (1e308, 1e308, 1.),
            (1e-12, 1e-300, 1e-300),
            (1e-12, f64::from_bits(1), 1e-180),
            (1e-12, 0., 1.),
        ] {
            assert!(Pade::new(0., l, 0., c, length).is_err());
        }
    }
}
