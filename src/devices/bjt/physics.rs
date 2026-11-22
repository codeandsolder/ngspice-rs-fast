//! Ordinary BJT physics from `bjtload.c`, independent of matrix and state
//! orchestration. `bjtdset.c` has different junction and derivative rules;
//! its kernels remain separate in `disto`.
//!
//! Thermal parameters already include instance area; these stages do not
//! apply polarity or multiplicity. Bias is normalized by the caller.

#![expect(
    clippy::similar_names,
    reason = "C-translated equation symbols retain the original names for numerical review"
)]
#![expect(
    clippy::suboptimal_flops,
    reason = "SPICE numerical reference parity requires the original non-fused C operation order"
)]

use super::{Bias, Thermal};
use crate::devices::nonlinear::{cubic_reverse_junction, exponential_junction, reverse_depletion};
use crate::primitives::{Real, SpiceError, SpiceResult};

/// ngspice `MAX_EXP_ARG`, the substrate exponential clamp of `bjtload.c`.
const MAX_EXP_ARG: Real = 709.;

/// One junction's current and incremental conductance, before multiplicity.
#[derive(Debug, Clone, Copy)]
pub(super) struct Junction {
    pub(super) current: Real,
    pub(super) conductance: Real,
}

#[derive(Debug, Clone, Copy)]
struct IntrinsicJunctions {
    be: Junction,
    bc: Junction,
    be_leakage: Junction,
    bc_leakage: Junction,
}

/// Base charge and its two partial derivatives (`bjtload.c`).
#[derive(Debug, Clone, Copy)]
struct BaseCharge {
    qb: Real,
    dqbdve: Real,
    dqbdvc: Real,
}

/// Normalized currents and their linearization, including base resistance.
#[derive(Debug, Clone, Copy)]
pub(super) struct Transport {
    pub(super) cc: Real,
    pub(super) cb: Real,
    pub(super) gpi: Real,
    pub(super) gmu: Real,
    pub(super) gm: Real,
    pub(super) go: Real,
    pub(super) gx: Real,
}

/// Charge-storage elements and their Jacobian before transient integration.
#[derive(Debug, Clone, Copy)]
pub(super) struct Charges {
    pub(super) geqcb: Real,
    pub(super) capbe: Real,
    pub(super) capbc: Real,
    pub(super) capbx: Real,
    pub(super) capsub: Real,
    pub(super) qbe: Real,
    pub(super) qbc: Real,
    pub(super) qbx: Real,
    pub(super) qsub: Real,
}

/// Complete ordinary-load point, in normalized device orientation and
/// per-instance area scaling. The caller applies multiplicity once.
#[derive(Debug, Clone, Copy)]
pub(super) struct Point {
    pub(super) transport: Transport,
    pub(super) substrate: Junction,
    pub(super) charges: Charges,
}

/// `bjtload.c`'s junction exponential with the cubic reverse continuation
/// below `-3 vtn`, in C's arithmetic.
fn c_junction(v: Real, vtn: Real, is: Real) -> (Real, Real) {
    if v >= -3. * vtn {
        exponential_junction(v, vtn, is)
    } else {
        cubic_reverse_junction(v, vtn, is)
    }
}

fn intrinsic_junctions(t: &Thermal, bias: Bias, gmin: Real) -> IntrinsicJunctions {
    let Bias { vbe, vbc, .. } = bias;
    let vt = t.vt;
    let vte = t.ne * vt;
    let vtc = t.nc * vt;
    let vtn = vt * t.nf;
    let (cbe, gbe) = c_junction(vbe, vtn, t.is_be);
    let (mut cben, mut gben) = if t.ise == 0. {
        (0., 0.)
    } else {
        c_junction(vbe, vte, t.ise)
    };
    gben += gmin;
    cben += gmin * vbe;
    let vtn = vt * t.nr;
    let (cbc, gbc) = c_junction(vbc, vtn, t.is_bc);
    let (mut cbcn, mut gbcn) = if t.isc == 0. {
        (0., 0.)
    } else {
        c_junction(vbc, vtc, t.isc)
    };
    gbcn += gmin;
    cbcn += gmin * vbc;
    IntrinsicJunctions {
        be: Junction {
            current: cbe,
            conductance: gbe,
        },
        bc: Junction {
            current: cbc,
            conductance: gbc,
        },
        be_leakage: Junction {
            current: cben,
            conductance: gben,
        },
        bc_leakage: Junction {
            current: cbcn,
            conductance: gbcn,
        },
    }
}

fn substrate_junction(t: &Thermal, vsub: Real, gmin: Real) -> Junction {
    let vt = t.vt;
    let (gdsub, cdsub) = t.iss.map_or((gmin, gmin * vsub), |iss| {
        let vts = vt * t.ns;
        if vsub <= -3. * vts {
            let (current, conductance) = cubic_reverse_junction(vsub, vts, iss);
            (conductance + gmin, current + gmin * vsub)
        } else {
            let evsub = (vsub / vts).min(MAX_EXP_ARG).exp();
            (iss * evsub / vts + gmin, iss * (evsub - 1.) + gmin * vsub)
        }
    });
    Junction {
        current: cdsub,
        conductance: gdsub,
    }
}

/// Identical denominator in bjtload.c and bjtdset.c. Each caller
/// retains its own inverse and derivative operation order.
pub(super) fn early_denominator(t: &Thermal, vbe: Real, vbc: Real) -> Real {
    1. - t.inv_vaf * vbc - t.inv_var * vbe
}

/// Base current in C's addition order. Inputs already obey the
/// respective load or distortion scaling and junction policy.
pub(super) fn base_current(t: &Thermal, cbe: Real, cben: Real, cbc: Real, cbcn: Real) -> Real {
    cbe / t.bf + cben + cbc / t.br + cbcn
}

fn base_charge(t: &Thermal, bias: Bias, junctions: IntrinsicJunctions) -> BaseCharge {
    let Bias { vbe, vbc, .. } = bias;
    let Junction {
        current: cbe,
        conductance: gbe,
    } = junctions.be;
    let Junction {
        current: cbc,
        conductance: gbc,
    } = junctions.bc;
    // Base charge terms.
    let q1 = 1. / early_denominator(t, vbe, vbc);
    let (qb, dqbdve, dqbdvc);
    if t.inv_ikf == 0. && t.inv_ikr == 0. {
        qb = q1;
        dqbdve = q1 * qb * t.inv_var;
        dqbdvc = q1 * qb * t.inv_vaf;
    } else {
        let q2 = t.inv_ikf * cbe + t.inv_ikr * cbc;
        let x = 1. + 4. * q2;
        // C's MAX(0, x).
        let arg = if 0. > x { 0. } else { x };
        let mut sqarg = 1.;
        match t.nkf {
            None => {
                if arg != 0. {
                    sqarg = arg.sqrt();
                }
                qb = q1 * (1. + sqarg) / 2.;
                dqbdve = q1 * (qb * t.inv_var + t.inv_ikf * gbe / sqarg);
                dqbdvc = q1 * (qb * t.inv_vaf + t.inv_ikr * gbc / sqarg);
            }
            Some(nkf) => {
                if arg != 0. {
                    sqarg = arg.powf(nkf);
                }
                qb = q1 * (1. + sqarg) / 2.;
                dqbdve = q1 * (qb * t.inv_var + t.inv_ikf * gbe * 2. * sqarg * nkf / arg);
                dqbdvc = q1 * (qb * t.inv_vaf + t.inv_ikr * gbc * 2. * sqarg * nkf / arg);
            }
        }
    }
    BaseCharge { qb, dqbdve, dqbdvc }
}

fn transport(t: &Thermal, junctions: IntrinsicJunctions, base: BaseCharge) -> Transport {
    let Junction {
        current: cbe,
        conductance: gbe,
    } = junctions.be;
    let Junction {
        current: cbc,
        conductance: gbc,
    } = junctions.bc;
    let Junction {
        current: cben,
        conductance: gben,
    } = junctions.be_leakage;
    let Junction {
        current: cbcn,
        conductance: gbcn,
    } = junctions.bc_leakage;
    let BaseCharge { qb, dqbdve, dqbdvc } = base;
    let rbpi = t.rb - t.rbm;
    // Excess phase is rejected, so cc starts at zero and cex/gex are cbe/gbe.
    let mut cc = 0.;
    let cex = cbe;
    let gex = gbe;
    cc = cc + (cex - cbc) / qb - cbc / t.br - cbcn;
    let cb = base_current(t, cbe, cben, cbc, cbcn);
    let gx = if t.irb == 0. {
        t.rbm + rbpi / qb
    } else {
        let ratio = cb / t.irb;
        // C's MAX(cb/irb, 1e-9).
        let arg1 = if ratio > 1e-9 { ratio } else { 1e-9 };
        let arg2 = (-1. + (1. + 14.59025 * arg1).sqrt()) / 2.4317 / arg1.sqrt();
        let arg1 = arg2.tan();
        t.rbm + 3. * rbpi * (arg1 - arg2) / arg2 / arg1 / arg1
    };
    let gx = if gx == 0. { gx } else { 1. / gx };
    let gpi = gbe / t.bf + gben;
    let gmu = gbc / t.br + gbcn;
    let go = (gbc + (cex - cbc) * dqbdvc / qb) / qb;
    let gm = (gex - (cex - cbc) * dqbdve / qb) / qb - go;
    Transport {
        cc,
        cb,
        gpi,
        gmu,
        gm,
        go,
        gx,
    }
}

/// The forward diffusion charge and its two bias derivatives.
struct ForwardDiffusion {
    charge: Real,
    capacitance: Real,
    geqcb: Real,
}

fn forward_diffusion(t: &Thermal, bias: Bias, be: Junction, base: BaseCharge) -> ForwardDiffusion {
    let Bias { vbe, vbc, .. } = bias;
    let Junction {
        current: mut cbe,
        conductance: mut gbe,
    } = be;
    let BaseCharge { qb, dqbdve, dqbdvc } = base;
    let tf = t.tf;
    let xtf = t.xtf;
    let ovtf = t.vtf_factor;
    let xjtf = t.itf;
    let geqcb = if tf != 0. && vbe > 0. {
        let (argtf, arg2, arg3) = if xtf == 0. {
            (0., 0., 0.)
        } else {
            let argtf = if ovtf == 0. {
                xtf
            } else {
                xtf * (vbc * ovtf).exp()
            };
            let (argtf, arg2) = if xjtf == 0. {
                (argtf, argtf)
            } else {
                let temp = cbe / (cbe + xjtf);
                let argtf = argtf * temp * temp;
                (argtf, argtf * (3. - temp - temp))
            };
            (argtf, arg2, cbe * argtf * ovtf)
        };
        cbe = cbe * (1. + argtf) / qb;
        gbe = (gbe * (1. + arg2) - cbe * dqbdve) / qb;
        tf * (arg3 - cbe * dqbdvc) / qb
    } else {
        0.
    };
    ForwardDiffusion {
        charge: tf * cbe,
        capacitance: tf * gbe,
        geqcb,
    }
}

fn charges(t: &Thermal, bias: Bias, junctions: IntrinsicJunctions, base: BaseCharge) -> Charges {
    let Bias {
        vbe,
        vbc,
        vbx,
        vsub,
    } = bias;
    let Junction {
        current: cbc,
        conductance: gbc,
    } = junctions.bc;
    let tr = t.tr;
    let czbe = t.cje;
    let ctot = t.cjc;
    let czbc = ctot * t.xcjc;
    let czbx = ctot - czbc;
    let czsub = t.cjs;
    let ps = t.vjs;
    let xms = t.mjs;
    let ForwardDiffusion {
        charge,
        capacitance,
        geqcb,
    } = forward_diffusion(t, bias, junctions.be, base);
    let (qbe, capbe) = t.be.charge(vbe, czbe, Some((charge, capacitance)));
    let (qbc, capbc) = t.bc.charge(vbc, czbc, Some((tr * cbc, tr * gbc)));
    let (qbx, capbx) = t.bc.charge(vbx, czbx, None);
    let (qsub, capsub) = if vsub < 0. {
        reverse_depletion(vsub, czsub, ps, xms)
    } else {
        (
            vsub * czsub * (1. + xms * vsub / (2. * ps)),
            czsub * (1. + xms * vsub / ps),
        )
    };
    Charges {
        geqcb,
        capbe,
        capbc,
        capbx,
        capsub,
        qbe,
        qbc,
        qbx,
        qsub,
    }
}

/// Evaluate all stages, preserving C's order and the full nonfinite check
/// even when a caller (noise, AC or sensitivity reset) only consumes a subset.
pub(super) fn c_point(t: &Thermal, bias: Bias, gmin: Real) -> SpiceResult<Point> {
    let junctions = intrinsic_junctions(t, bias, gmin);
    let substrate = substrate_junction(t, bias.vsub, gmin);
    let base = base_charge(t, bias, junctions);
    let transport = transport(t, junctions, base);
    let charges = charges(t, bias, junctions, base);
    let Transport {
        cc,
        cb,
        gpi,
        gmu,
        gm,
        go,
        gx,
    } = transport;
    let Junction {
        conductance: gdsub,
        current: cdsub,
    } = substrate;
    let Charges {
        geqcb,
        capbe,
        capbc,
        capbx,
        capsub,
        qbe,
        qbc,
        qbx,
        qsub,
    } = charges;
    let finite = [
        cc, cb, gpi, gmu, gm, go, gx, gdsub, cdsub, geqcb, capbe, capbc, capbx, capsub, qbe, qbc,
        qbx, qsub,
    ]
    .iter()
    .all(|v| v.is_finite());
    if !finite {
        return Err(SpiceError::Numerical {
            context: "BJT".into(),
            message: "nonfinite Gummel-Poon current, charge or conductance".into(),
        });
    }
    Ok(Point {
        transport,
        substrate,
        charges,
    })
}

#[cfg(test)]
mod tests {
    use super::super::tests::thermal;
    use super::*;

    #[test]
    fn junction_policies_use_their_own_reverse_boundary_and_gmin() {
        let t = thermal("is=1e-15 ise=1e-14 ne=2 iss=2e-16 ns=1.3", 27., true);
        let gmin = 1e-12;
        let vtn = t.vt * t.nf;
        let boundary = -3. * vtn;
        for (voltage, expected) in [
            (boundary, exponential_junction(boundary, vtn, t.is_be)),
            (
                boundary - 1e-12,
                cubic_reverse_junction(boundary - 1e-12, vtn, t.is_be),
            ),
        ] {
            let actual = intrinsic_junctions(
                &t,
                Bias {
                    vbe: voltage,
                    ..Bias::default()
                },
                gmin,
            );
            assert_eq!(actual.be.current.to_bits(), expected.0.to_bits());
            assert_eq!(actual.be.conductance.to_bits(), expected.1.to_bits());
            let (leak, conductance) = exponential_junction(voltage, t.ne * t.vt, t.ise);
            assert_eq!(
                actual.be_leakage.current.to_bits(),
                (leak + gmin * voltage).to_bits()
            );
            assert_eq!(
                actual.be_leakage.conductance.to_bits(),
                (conductance + gmin).to_bits()
            );
        }
        let iss = t.iss.unwrap();
        let vts = t.vt * t.ns;
        let boundary = -3. * vts;
        let actual = substrate_junction(&t, boundary, gmin);
        let (current, conductance) = cubic_reverse_junction(boundary, vts, iss);
        assert_eq!(
            actual.current.to_bits(),
            (current + gmin * boundary).to_bits()
        );
        assert_eq!(actual.conductance.to_bits(), (conductance + gmin).to_bits());
    }
}
