//! Deflated quadratic and Muller kernels with scaled numerical arithmetic.
use super::{Cx, Real, Region, SpiceResult, Trial, materialize_power};
use crate::maths::pencil::{normalize_muller, r_norm, zaddeq};

/// Curvature hint produced by `NIpzSym2` and consumed by complex initialization.
#[derive(Default)]
pub(super) struct Curvature {
    pub(super) value: Real,
    pub(super) exponent: i64,
}

impl Curvature {
    pub(super) const fn usable(&self) -> bool {
        self.value != 0.0 && self.exponent > -10
    }
}

/// `NIpzSym2`: deflated quadratic fit through three real trials.
pub(super) fn sym2(
    t0: &Trial,
    t1: &Trial,
    t2: &Trial,
    region: Region,
    curvature: &mut Curvature,
) -> SpiceResult<Real> {
    let mut dx0 = t1.s().re - t0.s().re;
    let mut dx1 = t2.s().re - t1.s().re;
    let x0 = t0.s().re.midpoint(t1.s().re);
    let mut d2x = (t2.s().re - t0.s().re) / 2.0;

    let (mut a, mut a_mag) = zaddeq(t1.f_def.re, t1.mag_def, -t0.f_def.re, t0.mag_def)?;
    let mut tmag = 0;
    r_norm(&mut dx0, &mut tmag)?;
    a /= dx0;
    a_mag -= tmag;
    r_norm(&mut a, &mut a_mag)?;

    let (mut b, mut b_mag) = zaddeq(t2.f_def.re, t2.mag_def, -t1.f_def.re, t1.mag_def)?;
    tmag = 0;
    r_norm(&mut dx1, &mut tmag)?;
    b /= dx1;
    b_mag -= tmag;
    r_norm(&mut b, &mut b_mag)?;

    let (mut c, mut c_mag) = zaddeq(b, b_mag, -a, a_mag)?;
    tmag = 0;
    r_norm(&mut d2x, &mut tmag)?;
    c /= d2x;
    c_mag -= tmag;
    r_norm(&mut c, &mut c_mag)?;

    if c == 0.0 || ((a == 0.0 || c_mag < a_mag - 40) && (b == 0.0 || c_mag < b_mag - 40)) {
        if a == 0.0 {
            a = b;
            a_mag = b_mag;
        }
        if a == 0.0 {
            return Ok(t1.s().re);
        }
        let s = materialize_power(-t1.f_def.re / a, t1.mag_def - a_mag);
        return Ok(s + t1.s().re);
    }

    a /= c;
    r_norm(&mut a, &mut a_mag)?;
    a_mag -= c_mag;
    let mut diff = t1.s().re - x0;
    tmag = 0;
    r_norm(&mut diff, &mut tmag)?;
    (a, a_mag) = zaddeq(a, a_mag, diff, tmag)?;

    b = 2.0 * t1.f_def.re / c;
    b_mag = t1.mag_def - c_mag;
    r_norm(&mut b, &mut b_mag)?;

    let (mut disc, mut disc_mag) = zaddeq(a * a, 2 * a_mag, -b, b_mag)?;
    let complex = disc < 0.0;
    if complex {
        disc *= -1.0;
    }
    if disc_mag % 2 == 0 {
        disc = disc.sqrt();
    } else {
        disc = (2.0 * disc).sqrt();
        disc_mag -= 1;
    }
    disc_mag /= 2;
    if complex {
        if curvature.value == 0.0 {
            curvature.value = disc;
            curvature.exponent = disc_mag;
        }
        disc = 0.0;
        disc_mag = 0;
    }
    (c, c_mag) = if a * disc >= 0.0 {
        zaddeq(a, a_mag, disc, disc_mag)?
    } else {
        zaddeq(a, a_mag, -disc, disc_mag)?
    };
    if c == 0.0 {
        b = 0.0;
        b_mag = 0;
    } else {
        b /= c;
        b_mag -= c_mag;
    }
    (b, b_mag) = zaddeq(t1.s().re, 0, -b, b_mag)?;
    (c, c_mag) = zaddeq(t1.s().re, 0, -c, c_mag)?;
    b = materialize_power(b, b_mag);
    c = materialize_power(c, c_mag);
    let outside = |x: Real| x < t0.s().re || x > t2.s().re;
    Ok(if outside(b) {
        if outside(c) {
            match region {
                Region::LeftCrossing => t0.s().re.midpoint(t1.s().re),
                Region::RightCrossing => t1.s().re.midpoint(t2.s().re),
                Region::Minimum => {
                    if (t1.s().re - c).abs() < (t1.s().re - b).abs() {
                        t1.s().re.midpoint(c)
                    } else {
                        t1.s().re.midpoint(b)
                    }
                }
            }
        } else {
            c
        }
    } else if outside(c) || a > 0.0 {
        b
    } else {
        c
    })
}

/// `NIpzMuller`.
pub(super) fn muller(t: [&Trial; 3]) -> SpiceResult<Cx> {
    let mut min = -999_999;
    let mut count = 0;
    let mut total = 0;
    for trial in t {
        if !trial.f_def.is_zero() {
            if min < trial.mag_def - 50 {
                min = trial.mag_def - 50;
            }
            total += trial.mag_def;
            count += 1;
        }
    }
    let mut magx = if count != 0 { total / count } else { total };
    if magx < min {
        magx = min;
    }
    let mut scale = [1.0; 3];
    for (i, trial) in t.iter().enumerate() {
        let mag = trial.mag_def - magx;
        scale[i] = if mag < -90 {
            0.0
        } else {
            materialize_power(1.0, mag)
        };
    }
    let f: [Cx; 3] = std::array::from_fn(|i| t[i].f_def.scale(scale[i]));
    let h0 = t[0].s().sub(t[1].s());
    let h1 = t[1].s().sub(t[2].s());
    let lambda = h0.div(h1);
    let mut delta = lambda;
    delta.re += 1.0;

    let mut a = f[2].mul(lambda);
    let mut c = f[1].mul(delta);
    a = a.sub(c);
    a = a.add(f[0]);
    a = a.mul(lambda);

    let mut b = lambda.mul(lambda);
    b = b.mul(f[2]);
    c = delta.mul(delta);
    c = c.mul(f[1]);
    b = b.sub(c);
    c = lambda.add(delta);
    c = c.mul(f[0]);
    b = b.add(c);

    c = delta.mul(f[0]);

    normalize_muller(&mut a, &mut b, &mut c)?;
    let mut d = b.mul(b);
    let e = a.scale(4.0).mul(c);
    d = d.sub(e);
    d = d.sqrt();
    let q = b.re.mul_add(d.re, b.im * d.im);
    b = if q > 0.0 { b.add(d) } else { b.sub(d) };
    c = c.div(b.scale(-0.5));
    Ok(h0.mul(c).add(t[0].s()))
}
