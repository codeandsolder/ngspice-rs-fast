//! The slice of C's `CKTcircuit` the BSIM4 load routines read and write.
//!
//! `b4ld.c`, `b4acld.c` and `b4pzld.c` are ported statement by statement, so
//! they keep C's view of the circuit: a `CKTmode` bitmask, the previous
//! solution (`CKTrhsOld`), the state vectors `CKTstate0..2`, `NIintegrate`
//! and additions into matrix elements and the right-hand side. The device
//! (`device.rs`) implements these traits over the Rust solver's
//! [`crate::devices::StampContext`] and [`crate::devices::LinearContext`].
//!
//! Nodes are BSIM4-local indices ([`node`]), which the instance's node fields
//! (`dNodePrime`, `gNodeMid`, ...) hold exactly as C's hold circuit node
//! numbers: an internal node that `b4set.c` does not create aliases its
//! external node, as in C. Matrix stamps involving ground are dropped by the
//! host, as SPARSE's trash-can element absorbs them in C.
use crate::primitives::SpiceResult;

/// BSIM4-local node indices. `b4set.c` sets an instance's internal node
/// fields either to their own index or to the external node they alias.
pub(super) mod node {
    /// External drain (`BSIM4dNode`).
    pub(in super::super) const D: i32 = 0;
    /// External gate (`BSIM4gNodeExt`).
    pub(in super::super) const G_EXT: i32 = 1;
    /// External source (`BSIM4sNode`).
    pub(in super::super) const S: i32 = 2;
    /// External bulk (`BSIM4bNode`).
    pub(in super::super) const B: i32 = 3;
    /// Internal drain (`BSIM4dNodePrime`).
    pub(in super::super) const DP: i32 = 4;
    /// Internal gate (`BSIM4gNodePrime`).
    pub(in super::super) const GP: i32 = 5;
    /// Mid gate, `rgateMod == 3` (`BSIM4gNodeMid`).
    pub(in super::super) const GM: i32 = 6;
    /// Internal source (`BSIM4sNodePrime`).
    pub(in super::super) const SP: i32 = 7;
    /// Internal bulk (`BSIM4bNodePrime`).
    pub(in super::super) const BP: i32 = 8;
    /// Drain-side bulk, `rbodyMod` (`BSIM4dbNode`).
    pub(in super::super) const DB: i32 = 9;
    /// Source-side bulk, `rbodyMod` (`BSIM4sbNode`).
    pub(in super::super) const SB: i32 = 10;
    /// NQS charge node, `trnqsMod` (`BSIM4qNode`).
    pub(in super::super) const Q: i32 = 11;
    /// Number of local node indices.
    pub(in super::super) const COUNT: usize = 12;
    /// Ground: C's node 0, which an uncreated `qNode` holds. Stamps and
    /// right-hand-side terms at ground are discarded, as SPARSE's trash-can
    /// element discards them in C, and its voltage reads as zero.
    pub(in super::super) const GROUND: i32 = -1;
}

/// `CKTmode` bits (`cktdefs.h`), with C's values.
pub(super) mod mode {
    /// `MODETRAN`.
    pub(in super::super) const MODETRAN: u32 = 0x1;
    /// `MODEAC`.
    pub(in super::super) const MODEAC: u32 = 0x2;
    /// `MODEDCOP`.
    pub(in super::super) const MODEDCOP: u32 = 0x10;
    /// `MODETRANOP`.
    pub(in super::super) const MODETRANOP: u32 = 0x20;
    /// `MODEDCTRANCURVE`.
    pub(in super::super) const MODEDCTRANCURVE: u32 = 0x40;
    /// `MODEINITFLOAT`.
    pub(in super::super) const MODEINITFLOAT: u32 = 0x100;
    /// `MODEINITJCT`.
    pub(in super::super) const MODEINITJCT: u32 = 0x200;
    /// `MODEINITFIX`.
    pub(in super::super) const MODEINITFIX: u32 = 0x400;
    /// `MODEINITSMSIG`.
    pub(in super::super) const MODEINITSMSIG: u32 = 0x800;
    /// `MODEINITTRAN`.
    pub(in super::super) const MODEINITTRAN: u32 = 0x1000;
    /// `MODEINITPRED`.
    pub(in super::super) const MODEINITPRED: u32 = 0x2000;
    /// `MODEUIC`.
    pub(in super::super) const MODEUIC: u32 = 0x10000;
}

/// State-vector offsets (`bsim4def.h`'s `BSIM4states + n`).
pub(super) mod state {
    /// `BSIM4vbd`.
    pub(in super::super) const VBD: usize = 0;
    /// `BSIM4vbs`.
    pub(in super::super) const VBS: usize = 1;
    /// `BSIM4vgs`.
    pub(in super::super) const VGS: usize = 2;
    /// `BSIM4vds`.
    pub(in super::super) const VDS: usize = 3;
    /// `BSIM4vdbs`.
    pub(in super::super) const VDBS: usize = 4;
    /// `BSIM4vdbd`.
    pub(in super::super) const VDBD: usize = 5;
    /// `BSIM4vsbs`.
    pub(in super::super) const VSBS: usize = 6;
    /// `BSIM4vges`.
    pub(in super::super) const VGES: usize = 7;
    /// `BSIM4vgms`.
    pub(in super::super) const VGMS: usize = 8;
    /// `BSIM4vses`.
    pub(in super::super) const VSES: usize = 9;
    /// `BSIM4vdes`.
    pub(in super::super) const VDES: usize = 10;
    /// `BSIM4qb`.
    pub(in super::super) const QB: usize = 11;
    /// `BSIM4cqb`.
    pub(in super::super) const CQB: usize = 12;
    /// `BSIM4qg`.
    pub(in super::super) const QG: usize = 13;
    /// `BSIM4cqg`.
    pub(in super::super) const CQG: usize = 14;
    /// `BSIM4qd`.
    pub(in super::super) const QD: usize = 15;
    /// `BSIM4cqd`.
    pub(in super::super) const CQD: usize = 16;
    /// `BSIM4qgmid`.
    pub(in super::super) const QGMID: usize = 17;
    /// `BSIM4cqgmid`.
    pub(in super::super) const CQGMID: usize = 18;
    /// `BSIM4qbs`.
    pub(in super::super) const QBS: usize = 19;
    /// `BSIM4cqbs`.
    pub(in super::super) const CQBS: usize = 20;
    /// `BSIM4qbd`.
    pub(in super::super) const QBD: usize = 21;
    /// `BSIM4cqbd`.
    pub(in super::super) const CQBD: usize = 22;
    /// `BSIM4qcheq`.
    pub(in super::super) const QCHEQ: usize = 23;
    /// `BSIM4cqcheq`.
    pub(in super::super) const CQCHEQ: usize = 24;
    /// `BSIM4qcdump`.
    pub(in super::super) const QCDUMP: usize = 25;
    /// `BSIM4cqcdump`.
    pub(in super::super) const CQCDUMP: usize = 26;
    /// `BSIM4qdef`.
    pub(in super::super) const QDEF: usize = 27;
    /// `BSIM4qs`.
    pub(in super::super) const QS: usize = 28;
    /// `BSIM4numStates`.
    pub(in super::super) const COUNT: usize = 29;
}

/// What `b4ld.c` reads from and writes to `CKTcircuit`.
pub(super) trait LoadCkt {
    /// `CKTmode`, as [`mode`] bits.
    fn mode(&self) -> u32;
    /// Whether the load may limit its junction voltages (`DEVfetlim`,
    /// `DEVlimvds`, `DEVpnjlim`). Always true for C; false only for the
    /// port's exact (unlimited) re-evaluations, where the load evaluates the
    /// supplied voltages unchanged.
    fn limiting(&self) -> bool;
    /// `CKTgmin`.
    fn gmin(&self) -> f64;
    /// `CKTtemp` in kelvin.
    fn temp(&self) -> f64;
    /// `CKTag[i]` (zero outside transient).
    fn ag(&self, i: usize) -> f64;
    /// `CKTdelta` (zero outside transient).
    fn delta(&self) -> f64;
    /// `CKTdeltaOld[i]` (zero outside transient).
    fn delta_old(&self, i: usize) -> f64;
    /// `CKTrhsOld[node]`: the solution the load linearises around.
    fn rhs_old(&self, node: i32) -> f64;
    /// `CKTstate0[slot]`: the previous iterate's value until this load
    /// writes it, then the written value.
    fn state0(&self, slot: usize) -> f64;
    /// `*(CKTstate0 + slot) = value`.
    fn set_state0(&mut self, slot: usize, value: f64);
    /// `CKTstate1[slot]`, the last accepted point.
    fn state1(&self, slot: usize) -> f64;
    /// `*(CKTstate1 + slot) = value` (`MODEINITTRAN` only).
    fn set_state1(&mut self, slot: usize, value: f64);
    /// `CKTstate2[slot]`.
    fn state2(&self, slot: usize) -> f64;
    /// `NIintegrate(ckt, &geq, &ceq, cap, slot)`: integrates the charge in
    /// `state0[slot]` into `state0[slot + 1]` and returns `(geq, ceq)`.
    ///
    /// # Errors
    /// A short accepted history or nonfinite values (C's `E_ORDER` and
    /// `E_METHOD` cannot occur).
    fn integrate(&mut self, cap: f64, slot: usize) -> SpiceResult<(f64, f64)>;
    /// `ckt->CKTnoncon++`.
    fn noncon(&mut self);
    /// `*(here->XYPtr) += value` for the element at (`row`, `col`).
    fn matrix(&mut self, row: i32, col: i32, value: f64);
    /// `*(ckt->CKTrhs + node) += value`.
    fn rhs(&mut self, node: i32, value: f64);
}

/// What `b4acld.c` and `b4pzld.c` read and write. Both run after the
/// `MODEINITSMSIG` load has left the operating point's values on the
/// instance.
pub(super) trait SmallSignalCkt {
    /// `CKTomega` (`b4acld.c` only).
    fn omega(&self) -> f64;
    /// `CKTstates[0][slot]` / `CKTstate0[slot]` at the operating point.
    fn state0(&self, slot: usize) -> f64;
    /// `*(here->XYPtr) += value`: the real part.
    fn real(&mut self, row: i32, col: i32, value: f64);
    /// `*(here->XYPtr + 1) += value`: the imaginary part, already including
    /// `CKTomega` as `b4acld.c` computes it.
    fn imag(&mut self, row: i32, col: i32, value: f64);
    /// `b4pzld.c`'s `*(ptr) += value * s->real; *(ptr + 1) += value *
    /// s->imag`: `value` is the coefficient of `s` in the pencil `A + s E`.
    fn cap(&mut self, row: i32, col: i32, value: f64);
}
