//! Real leakage stamp contributions common to nonlinear, AC and PZ loads.
//! Keep additions in reference order even when local nodes alias.
use super::fields::Instance;

/// GIDL followed by GISL; the sum and multiplication order matches C.
#[expect(
    clippy::similar_names,
    reason = "retain the C drain/source leakage derivative names"
)]
pub(super) fn leakage(h: &Instance, mult_i: f64, mut add: impl FnMut(i32, i32, f64)) {
    let (dp, gp, sp, bp) = (h.dNodePrime, h.gNodePrime, h.sNodePrime, h.bNodePrime);
    let ggidld = h.ggidld;
    let ggidlg = h.ggidlg;
    let ggidlb = h.ggidlb;
    let ggislg = h.ggislg;
    let ggisls = h.ggisls;
    let ggislb = h.ggislb;

    // stamp gidl
    add(dp, dp, mult_i * ggidld);
    add(dp, gp, mult_i * ggidlg);
    add(dp, sp, -(mult_i * ((ggidlg + ggidld) + ggidlb)));
    add(dp, bp, mult_i * ggidlb);
    add(bp, dp, -(mult_i * ggidld));
    add(bp, gp, -(mult_i * ggidlg));
    add(bp, sp, mult_i * ((ggidlg + ggidld) + ggidlb));
    add(bp, bp, -(mult_i * ggidlb));
    // stamp gisl
    add(sp, dp, -(mult_i * ((ggisls + ggislg) + ggislb)));
    add(sp, gp, mult_i * ggislg);
    add(sp, sp, mult_i * ggisls);
    add(sp, bp, mult_i * ggislb);
    add(bp, dp, mult_i * ((ggislg + ggisls) + ggislb));
    add(bp, gp, -(mult_i * ggislg));
    add(bp, sp, -(mult_i * ggisls));
    add(bp, bp, -(mult_i * ggislb));
}
