//! Counter-based independent random streams shared by elaboration and transient sources.
use super::Real;
const INV_53: Real = 1. / 9_007_199_254_740_992.;

pub(crate) const fn mix(mut value: u64) -> u64 {
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}
const fn draw(key: u64, stream: u64, index: u64, ordinal: u64) -> u64 {
    mix(key
        ^ mix(stream)
        ^ mix(index.wrapping_add(0x243f_6a88_85a3_08d3))
        ^ mix(ordinal.wrapping_add(0x1319_8a2e_0370_7344)))
}
/// Convert integer through 32-bit limbs; all binary64 grid integers
/// are exactly represented without relying on a narrowing cast.
pub(crate) fn real_u64(value: u64) -> Real {
    let hi = u32::try_from(value >> 32).unwrap_or_default();
    let lo = u32::try_from(value & u64::from(u32::MAX)).unwrap_or_default();
    4_294_967_296_f64.mul_add(Real::from(hi), Real::from(lo))
}
pub(crate) fn uniform(key: u64, stream: u64, index: u64, ordinal: u64) -> Real {
    (real_u64(draw(key, stream, index, ordinal) >> 11) + 0.5) * INV_53
}
pub(crate) fn gaussian(key: u64, stream: u64, index: u64) -> Real {
    let unit = uniform(key, stream, index, 0);
    let theta = std::f64::consts::TAU * uniform(key, stream, index, 1);
    (-2. * unit.ln()).sqrt() * theta.cos()
}
