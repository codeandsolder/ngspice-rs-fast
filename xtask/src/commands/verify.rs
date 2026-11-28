//! Rust-engine verification against committed C data. Never executes C or
//! writes a deck/rawfile. Extend the registry only with demonstrated coverage.

use std::collections::HashSet;
#[cfg(test)]
use std::fs;
use std::io::BufRead;
use std::path::Path;

#[cfg(test)]
use ngspice_rs::analysis::RawFile;
use ngspice_rs::analysis::{AnalysisPlan, RawPlotRows, RunConfig};
use ngspice_rs::netlist::Parser;
use ngspice_rs::primitives::AnalysisKind;

#[cfg(test)]
use super::workspace_root;
use super::{compare, tran};
use ngspice_conformance::fixtures;

mod stream;
use stream::Golden;

/// How a fixture's Rust result is compared with its C golden.
enum Gate {
    /// Point-wise by variable name (DC/AC), optionally along a sweep axis.
    Points {
        axis: Option<&'static str>,
        tolerance: compare::Tolerance,
        // Explicit equal physical coordinates introduced only by C's
        // redundant zero-voltage current-sense devices. Every reference
        // column is still compared; production circuits need no aliases.
        aliases: &'static [(&'static str, &'static str)],
    },
    /// Event-aware comparison on a shared physical time grid (`tran.rs`).
    Transient(compare::TranTolerance),
    /// Pole-zero plots: poles and zeros as unordered root sets
    /// (`compare::roots`).
    Roots(compare::RootTolerance),
    /// C exhausted its search before finding these independently verified
    /// real poles. Require the augmented root set, not either outcome.
    CompletedRoots {
        tolerance: compare::RootTolerance,
        additional_poles: &'static [f64],
    },
    /// [`Gate::Transient`] for transmission-line decks: every source
    /// breakpoint `b` also declares the breakpoints `b + offset` (inside the
    /// run) where a line's delayed wave changes slope. The offsets are
    /// derived by hand from the deck's `td` values and topology (which echoes
    /// exist depends on the terminations) and documented with the fixture;
    /// both simulators must land on each of them (`traacct.c` sets them as
    /// device breakpoints), never interpolated across.
    Echoes {
        tolerance: compare::TranTolerance,
        offsets: &'static [f64],
    },
}

struct Supported {
    name: &'static str,
    kind: AnalysisKind,
    gate: Gate,
}

/// Transient registry entry: `compare::TRAN`.
const fn tran(name: &'static str) -> Supported {
    fixture(name, Stage::transient())
}

const SUPPORTED: &[Supported] = &[
    fixture("bjt_ce", Stage::op(compare::NONLINEAR)),
    fixture("mos_inverter", Stage::op(compare::NONLINEAR)),
    fixture("diode_dc", Stage::dc(compare::LEGACY_DIODE_DC)),
    fixture("m4_diode_ac", Stage::ac(compare::NONLINEAR)),
    fixture("m4_bjt_ac", Stage::ac(compare::NONLINEAR)),
    fixture("m4_mos1_ac", Stage::ac(compare::NONLINEAR)),
    tran("m4_diode_tran"),
    tran("m4_bjt_tran"),
    tran("m4_mos1_tran"),
    // MOS1 completion (#88): Meyer gate charge (TOX), series resistance,
    // junction geometry, process extraction and temperature. The transient
    // decks bound the maximum step so that both simulators' discretization
    // error sits well inside `compare::TRAN`; the DC deck tightens RELTOL so
    // that C's own Newton stopping error does not exceed `NONLINEAR` (see
    // docs/port/VERIFICATION.md).
    tran("m7_mos1_inverter_tran"),
    tran("m7_mos1_ring_tran"),
    fixture("m7_mos1_meyer_ac", Stage::ac(compare::NONLINEAR)),
    fixture("m7_mos1_process_dc", Stage::dc(compare::NONLINEAR)),
    // MOS level 3 on the shared MOS shell (#89): short-channel threshold,
    // weak inversion, velocity saturation, channel-length modulation and
    // temperature (DC, tightened RELTOL as for MOS1), the small-signal stage
    // at 75 C, and a CMOS inverter with a bounded maximum step (transient).
    fixture("m10_mos3_dc", Stage::dc(compare::NONLINEAR)),
    fixture("m10_mos3_ac", Stage::ac(compare::NONLINEAR)),
    tran("m10_mos3_tran"),
    // MOS level 6 (Sakurai-Newton n-th power law) on the shared MOS shell
    // (#89): GAMMA1/SIGMA/LAMBDA1 threshold and CLM terms, NSUB/TOX process
    // extraction, PMOS series resistance and temperature (DC, tightened
    // RELTOL as for MOS1/MOS3), and a CMOS inverter with a bounded maximum
    // step (transient). C has no MOS6 AC load, so there is no AC fixture.
    fixture("m10_mos6_dc", Stage::dc(compare::NONLINEAR)),
    tran("m10_mos6_tran"),
    // MOS level 2 on the shared MOS shell (#89): Grove-Frohman and Baum
    // saturation, weak inversion, mobility degradation, channel-length
    // modulation and temperature (DC, tightened RELTOL as for MOS1/MOS3),
    // the small-signal stage at 75 C, and a CMOS inverter with a bounded
    // maximum step (transient).
    fixture("m10_mos2_dc", Stage::dc(compare::NONLINEAR)),
    fixture("m10_mos2_ac", Stage::ac(compare::NONLINEAR)),
    tran("m10_mos2_tran"),
    // MESA level 2 (#83, M10 H1a): the `mesa1` channel, `mesasetup.c`'s
    // `#gate` node (the second device carries it; the first has zero gate
    // resistance) and `mesaask.c`.
    fixture("m10_mesa_op", Stage::op(compare::NONLINEAR)),
    fixture("m10_mesa_dc", Stage::dc(compare::NONLINEAR)),
    // HFET1 (Z level 5, #83 spill H3): `hfeta`'s channel and both gate
    // junctions, the type-normalized `nhfet`/`phfet` polarities, the
    // `gatemod` cross terms, the `a1`/`a2` corrections and the `cds` charge.
    fixture("m10_hfet1_dc", Stage::dc(compare::NONLINEAR)),
    fixture("m10_hfet1_op", Stage::op(compare::NONLINEAR)),
    // A nonlinear device's AC: the bias Jacobian is converged independently
    // by each simulator, so this is `compare::NONLINEAR` (1 ppm) as for the
    // diode/BJT/MOS1 AC fixtures, not the linear `compare::AC`.
    fixture("m10_hfet1_ac", Stage::ac(compare::NONLINEAR)),
    pole_zero("m10_hfet1_pz"),
    // HFET2 (Z level 6, #169): `hfeta2`'s channel, the single-exponential gate
    // junctions, both polarities, the two-exponential `eta2`/`d2` branch and
    // the pnjlim/fetlim-limited load.
    fixture("m10_hfet2_op", Stage::op(compare::NONLINEAR)),
    fixture("m10_hfet2_dc", Stage::dc(compare::NONLINEAR)),
    fixture("m10_hfet2_ac", Stage::ac(compare::NONLINEAR)),
    Supported {
        name: "m10_hfet2_pz",
        kind: AnalysisKind::PoleZero,
        gate: Gate::CompletedRoots {
            tolerance: compare::POLE_ZERO,
            // Independent exact-rational expansion of the loaded pencil;
            // see POLE_ZERO_ADR.md and the production-path regression.
            additional_poles: &[-10_000_000_014.000_954],
        },
    },
    tran("m10_hfet2_tran"),
    // BSIM4 (#90): `bsim4/` translated statement by statement. Gate and body
    // resistance, the mid-gate/rbodymod=2/rdsmod=1 family, both polarities,
    // AC and pole-zero at C's MODEINITSMSIG bias, an NQS inverter chain and
    // binned models end to end (#109).
    fixture("m11_bsim4_op", Stage::op(compare::NONLINEAR)),
    fixture("m11_bsim4_dc", Stage::dc(compare::NONLINEAR)),
    fixture("m11_bsim4_ac", Stage::ac(compare::NONLINEAR)),
    pole_zero("m11_bsim4_pz"),
    tran("m11_bsim4_tran"),
    fixture("m11_bsim4_bin_dc", Stage::dc(compare::NONLINEAR)),
    // BSIM3v3.3 (#90): capMod=3 charges, both polarities.
    fixture("m11_bsim3_op", Stage::op(compare::NONLINEAR)),
    fixture("m11_bsim3_dc", Stage::dc(compare::NONLINEAR)),
    fixture("m11_bsim3_ac", Stage::ac(compare::NONLINEAR)),
    tran("m11_bsim3_tran"),
    tran("m10_hfet1_tran"),
    // MESA levels 3 and 4 (#168): `mesa2`'s two-population `nsa`/`nsb`
    // channel with the `vpou`/`vpod` threshold regions and `mesa3`'s
    // `nmax`/`gamma` channel with the `cas`/`cbs` capacitance split. Level 4
    // has no `.ac`/`.pz` fixture: C's load reads fields `mesa3` never stores
    // (see `tests/mesa.rs` and `docs/port/MESFET.md`).
    fixture("m10_mesa_l3_op", Stage::op(compare::NONLINEAR)),
    fixture("m10_mesa_l3_dc", Stage::dc(compare::NONLINEAR)),
    fixture("m10_mesa_l3_ac", Stage::ac(compare::NONLINEAR)),
    fixture("m10_mesa_l4_op", Stage::op(compare::NONLINEAR)),
    fixture("m10_mesa_l4_dc", Stage::dc(compare::NONLINEAR)),
    tran("m10_mesa_l4_tran"),
    tran("m10_mesa_tran"),
    // TXL (#170-#173). The fallback decks exercise the parser-time three-pi
    // expansion `inp2y.c` performs (its `3/(r*len)` resistance is C's own
    // dimensional quirk and is reproduced literally); the Pade decks exercise
    // `txlload.c`'s DC stamp and recursive convolution. `m10_txl_tran_step`
    // also covers C's `CKTmaxStep = 0.9*taul` clamp: its `tmax` exceeds
    // `taul`, so the first oversized step is taken and the rest are bounded.
    // `m10_txl3_tran` is the upstream three-line MOS driver.
    fixture("m10_txl_fb_op", Stage::op(compare::DC)),
    fixture("m10_txl_fb_dc", Stage::dc(compare::DC)),
    fixture("m10_txl_op", Stage::op(compare::DC)),
    fixture("m10_txl_dc", Stage::dc(compare::DC)),
    tran("m10_txl_fb_tran"),
    tran("m10_txl_lossless_tran"),
    tran("m10_txl_tran"),
    tran("m10_txl_early_tran"),
    tran("m10_txl_tran_step"),
    tran("m10_txl3_tran"),
    // The resistance paths (#155): `rd`/`rs` (the prime nodes, which C's
    // rawfile omits), `rg`, `ri`/`rf` (`#gs`/`#gd`) and the `rdi`/`rsi`
    // series pair, with the `rtc1`/`rtc2` temperature scaling in the second
    // fixture.
    fixture("m10_mesa_res_op", Stage::op(compare::NONLINEAR)),
    fixture("m10_mesa_res_dc", Stage::dc(compare::NONLINEAR)),
    // `mesaacl.c` with the resistance paths live: the same conductances
    // between the external, prime and prime-prime nodes, plus `j omega C`.
    fixture("m10_mesa_res_ac", Stage::ac(compare::AC)),
    // The same pair in the small-signal sweep: `mesaacl.c`'s conductances and
    // gate capacitances, with the `rg` device's `#gate` node in the AC plot as
    // it is in the DC one.
    fixture("m10_mesa_ac", Stage::ac(compare::AC)),
    // MOS level 9 (#89), mos3's decks with level-9 cards and geometry above
    // mos9temp.c's 1 um effective-size floor: DC (tightened RELTOL), the
    // small-signal stage at 75 C with an instance DTEMP, and the inverter.
    fixture("m10_mos9_dc", Stage::dc(compare::NONLINEAR)),
    fixture("m10_mos9_ac", Stage::ac(compare::NONLINEAR)),
    tran("m10_mos9_tran"),
    fixture("rc_divider", Stage::op(compare::DC)),
    fixture("rc_lowpass_ac", Stage::ac(compare::AC)),
    tran("rc_transient"),
    tran("m9_scoped_frontend"),
    fixture("rlc_series", Stage::op(compare::DC)),
    // Subcircuit elaboration (#18): one `X` instance is flattened through the
    // production `.op` path. The deck is a purely resistive divider, so it keeps
    // the same 1e-12 relative bound as the other linear operating points.
    fixture("subckt_divider", Stage::op(compare::DC)),
    // `.func` definitions and single-quoted values (#107): top-level and
    // body-local functions, quoted device/instance values, flattened through
    // the production `.op` path. Purely resistive, so the linear DC bound.
    fixture("func_quotes", Stage::op(compare::DC)),
    // M3 exit-gate fixtures (#48). All use `compare::TRAN`; the physical bound is
    // the simulator's own default accuracy (see `compare.rs`), no fixture-specific
    // tolerance exists. The companion driver (trap, or Gear-2 via
    // `.options method=gear`) runs each deck as written.
    tran("rl_pulse_tran"),
    tran("rc_gear_tran"),
    tran("rc_pwl_tran"),
    tran("rlc_series_tran"),
    tran("rlc_series_gear_tran"),
    // `maxord=6` (#98): C's dctran.c only toggles orders 1 and 2, so this is
    // the Gear-2 integration again (C data equal rlc_series_gear_tran).
    tran("rlc_series_gear_maxord6_tran"),
    tran("floating_cap_tran"),
    tran("coupled_cap_tran"),
    // Initialized-state fixtures (#27, #48): `uic` / instance `ic=` / `.ic`. With `uic` C writes no `t = 0`
    // row and adds a breakpoint at the `.tran` step; the Rust driver reproduces
    // both and the comparator (which only demands samples at source breakpoints)
    // needs no special case.
    tran("rc_ic_uic_tran"),
    tran("rlc_ic_uic_tran"),
    tran("rc_ic_node_tran"),
    tran("floating_cap_ic_tran"),
    // M6 source waveforms (#94, #95), companion trap driver under `compare::TRAN`.
    // C sets no breakpoints for SIN/EXP/SFFM/AM, so those decks are compared on
    // the grid only (`tran::breakpoints`); PULSE count and repeated PWL corners
    // are C breakpoints.
    tran("rc_sin_tran"),
    tran("rc_exp_tran"),
    tran("rc_sffm_am_tran"),
    tran("rc_pwl_repeat_tran"),
    tran("rc_pulse_count_tran"),
    fixture("rlc_series_ac", Stage::ac(compare::AC)), // `.option` coverage (#110/#107): `gmin={gj}` (evaluated against
    // `.param`) is the junction gmin of the reverse-biased diode/BJT junctions,
    // with itl1/itl2 and documented no-ops; the reverse region is nearly linear,
    // so the 1 ppm NONLINEAR bound applies.
    fixture("options_gmin_dc", Stage::dc(compare::NONLINEAR)),
    // Trapezoidal `xmu=0.2` and `itl4` on the companion driver.
    tran("options_xmu_tran"),
    // Linear controlled sources (#78): E/F/G/H in an operating point (with an
    // F/H controlled by an E branch and by a voltage source inside a
    // subcircuit), an AC sweep and a PULSE transient. All four devices are
    // linear, so the decks keep the linear DC/AC bounds and `compare::TRAN`.
    fixture("controlled_op", Stage::op(compare::DC)),
    fixture("controlled_ac", Stage::ac(compare::AC)),
    tran("controlled_tran"),
    // K mutual inductance (#80): a 1:2 transformer, a three-winding K inside a
    // subcircuit and a negative coupling in AC; a PULSE transformer transient
    // (trapezoidal companions on the coupled flux); and coupled `ic=` free
    // decay under `uic` with Gear-2. All linear: the linear AC bound and
    // `compare::TRAN`.
    fixture("transformer_ac", Stage::ac(compare::AC)),
    tran("transformer_tran"),
    tran("transformer_ic_uic_tran"),
    tran("transformer_model_uic_tran"),
    // S/W switches (#81): hysteresis bands, ON/OFF flags, gmin off conductance
    // and a W latch at an operating point, and a downward `.dc` sweep whose
    // points continue the previous point's accepted switch state. Switches
    // are nonlinear devices, so the nonlinear 1 ppm bound applies.
    fixture("switch_op", Stage::op(compare::NONLINEAR)),
    fixture("switch_dc", Stage::dc(compare::NONLINEAR)),
    // PULSE/SIN-controlled S, a self-controlled relaxation oscillator and W
    // switches sensing SIN/PULSE currents. Every plotted node is a source or
    // capacitor node, so no plotted value jumps between samples where a switch
    // flips.
    tran("switch_tran"),
    tran("switch_w_tran"),
    // A `.dc` with a decimal step: C's accumulated sweep values decide
    // switches at their thresholds (`dctrcurv.c` `value += step`).
    fixture("switch_dc_decimal", Stage::dc(compare::NONLINEAR)),
    // AC uses C's MODEINITSMSIG switch state (the zero CKTstate1: open), not
    // the operating point's.
    fixture("switch_ac", Stage::ac(compare::AC)),
    // Behavioural sources (#79): B sources over node voltages, branch
    // currents, time and temper in OP, a DC sweep, the AC linearisation at the
    // bias point and a time-dependent transient; E/G VALUE= lowered onto B
    // sources (also inside a subcircuit), E/G TABLE onto the XSPICE pwl
    // transfer and E/G/F/H POLY onto spice2poly (captured with those code
    // models, `* xtask-codemodels:`). Newton-solved, so the nonlinear 1 ppm
    // bound; the transient keeps `compare::TRAN`.
    fixture(
        "bsource_op",
        Stage::points_aliases(
            AnalysisKind::OperatingPoint,
            compare::NONLINEAR,
            &[("v(o1_vmeas_0)", "v(o1)"), ("i(v_b1)", "i(b1)")],
        ),
    ),
    fixture("bsource_dc", Stage::dc(compare::NONLINEAR)),
    fixture("bsource_ac", Stage::ac(compare::NONLINEAR)),
    tran("bsource_tran"),
    fixture(
        "evalue_op",
        Stage::points_aliases(
            AnalysisKind::OperatingPoint,
            compare::NONLINEAR,
            &[("v(o2_vmeas_0)", "v(o2)"), ("i(v_e2)", "i(e2)")],
        ),
    ),
    fixture("gtable_dc", Stage::dc(compare::NONLINEAR)),
    fixture(
        "epoly_dc",
        Stage::points_aliases(
            AnalysisKind::DcSweep,
            compare::NONLINEAR,
            &[
                ("i(a$poly$e1)", "i(e1)"),
                ("i(a$poly$e2)", "i(e2)"),
                ("i(a$poly$e3)", "i(e3)"),
                ("i(a$poly$h1)", "i(h1)"),
            ],
        ),
    ),
    // B sources whose sqrt/log/reciprocal/division slopes are ~1e32 (or whose
    // log() is -1e99) at the 0 V Newton start, in OP, a DC sweep and a
    // transient's initial point (#79); the Newton solve's balanced fallback.
    fixture("bsource_zero_op", Stage::op(compare::NONLINEAR)),
    fixture("bsource_zero_dc", Stage::dc(compare::NONLINEAR)),
    tran("bsource_zero_tran"),
    // Diode physics (#86): a Zener regulator swept through forward conduction
    // and reverse breakdown, recombination/tunnelling/knee/sidewall currents,
    // a `.dc temp` sweep of the EG/XTI/TLEV/TCV/DTEMP temperature laws, the
    // depletion-charge temperature laws and recombination small signal at
    // `.options temp=100`, and a SIN-driven Zener clipper with junction,
    // sidewall and diffusion charge. Newton-solved: the nonlinear 1 ppm bound;
    // the transient keeps `compare::TRAN`.
    fixture("m7_zener_dc", Stage::dc(compare::NONLINEAR)),
    fixture("m7_diode_physics_dc", Stage::dc(compare::NONLINEAR)),
    fixture("m7_diode_temp_dc", Stage::dc(compare::NONLINEAR)),
    // #97: `.dc @instance[parameter]` (C `param-sweep`) and the `res-sweep`
    // scale of a resistor target.
    fixture("m8_dc_param_diode", Stage::dc(compare::NONLINEAR)),
    fixture("m8_dc_param_mos1", Stage::dc(compare::NONLINEAR)),
    fixture("m8_dc_param_gain", Stage::dc(compare::DC)),
    fixture("m8_dc_res_temp", Stage::dc(compare::DC)),
    fixture("m7_diode_temp_ac", Stage::ac(compare::NONLINEAR)),
    tran("m7_zener_tran"),
    // Gummel-Poon BJT (#87): a Gummel plot (VBC = 0 sweep of VBE), nested
    // output characteristics (VCE inner, IB outer), a temperature sweep of
    // NPN/PNP/TLEV=3 devices with series resistances and a substrate junction,
    // and a CE amplifier's AC response and transient with every charge. The
    // DC/AC decks set `.option reltol=1e-8` so that C's convergence test (and
    // bypass) leave a result within the nonlinear 1 ppm bound of the shared
    // root; the transient keeps `compare::TRAN`.
    fixture("m7_bjt_gummel", Stage::dc(compare::NONLINEAR)),
    fixture("m7_bjt_output", Stage::dc(compare::NONLINEAR)),
    fixture("m7_bjt_temp", Stage::dc(compare::NONLINEAR)),
    fixture("m7_bjt_amp_ac", Stage::ac(compare::NONLINEAR)),
    tran("m7_bjt_amp_tran"),
    // Convergence parity (#106): a BJT latch whose operating point depends on
    // the Newton path (MODEINITJCT start, `DEVpnjlim`, `dynamic_gmin`), the
    // same latch under `.options noopiter` with each of ngspice's other
    // continuation strategies (`gillespie_src`, `spice3_gmin`, `spice3_src`;
    // `spice3_src` and the default land in different states), and a set/reset
    // transient through the regenerative switching. `.option reltol` is tight
    // enough that C's own stopping error stays inside the nonlinear 1 ppm
    // bound; the transient bounds the maximum step and keeps `compare::TRAN`.
    fixture("m7_conv_latch_op", Stage::op(compare::NONLINEAR)),
    fixture("m7_conv_latch_gillespie_op", Stage::op(compare::NONLINEAR)),
    fixture(
        "m7_conv_latch_spice3_gmin_op",
        Stage::op(compare::NONLINEAR),
    ),
    fixture("m7_conv_latch_spice3_src_op", Stage::op(compare::NONLINEAR)),
    tran("m7_conv_latch_tran"),
    // M10 0.5h: a 100-junction diode chain (MODEINITJCT start and DEVpnjlim on
    // every junction), once a `.op` failure where ngspice converges.
    fixture("diode100_op", Stage::op(compare::NONLINEAR)),
    // Nonlinear initial conditions (#99): a diode/RC `uic` start from the
    // `.ic` node vector (the diode's own `ic=` has no effect in C), a BJT
    // flip-flop whose `.ic` selects the state through the forced transient
    // operating point, a BJT flip-flop with an OFF transistor (held through
    // MODEINITJCT/MODEINITFIX until C's device convergence test hands over to
    // dynamic gmin), a MOS1 inverter pair started under `uic` from its IC
    // vectors, and a symmetric CMOS latch whose state a `.nodeset` or MOS1
    // `ic=` start voltages select. The transients keep `compare::TRAN`, the
    // operating points the nonlinear 1 ppm bound; decks tighten `reltol` (and
    // the MOS1 deck uses Gear-2) where C's own step error would be close to
    // the bound.
    tran("m7_ic_diode_uic_tran"),
    tran("m7_ic_bjt_flipflop_tran"),
    tran("m7_ic_bjt_off_tran"),
    tran("m7_ic_mos1_uic_tran"),
    fixture("m7_ic_latch_nodeset_op", Stage::op(compare::NONLINEAR)),
    fixture("m7_ic_latch_mos1_ic_op", Stage::op(compare::NONLINEAR)),
    // Lossless transmission lines (#84, M10 slice 3). The transient decks set
    // each line's `rel`/`abs` so that it lands on a breakpoint wherever a
    // wave changes slope; the echoes are listed per deck (see the decks'
    // comments). Both simulators take the same steps (the port follows
    // dctran.c/traacct.c/tratrunc.c), so `compare::TRAN` holds with margin.
    // The AC deck carries exp(-j omega td) exactly and keeps the linear AC
    // bound.
    Supported {
        name: "m10_tline_tran",
        kind: AnalysisKind::Transient,
        gate: Gate::Echoes {
            tolerance: compare::TRAN,
            // t2 (td = 1 ns, mismatched at both ends): every multiple of 1 ns;
            // t1 (td = 2 ns, matched) adds only b + 2 ns, already listed.
            offsets: &[
                1e-9, 2e-9, 3e-9, 4e-9, 5e-9, 6e-9, 7e-9, 8e-9, 9e-9, 10e-9, 11e-9,
            ],
        },
    },
    fixture("m10_tline_ac", Stage::ac(compare::AC)),
    Supported {
        name: "m10_tline_pulse",
        kind: AnalysisKind::Transient,
        gate: Gate::Echoes {
            tolerance: compare::TRAN,
            // t1 (1.25 ns, matched both ends): b + 1.25 ns; t2 (0.75 ns behind
            // the buffer, open end): b + 1.25 + 0.75 and b + 1.25 + 2*0.75.
            offsets: &[1.25e-9, 2e-9, 2.75e-9],
        },
    },
    // Pole-zero analysis (#103): C's Muller search against the port's
    // generalized eigenvalues, compared as unordered root sets under
    // `compare::POLE_ZERO`. Decks: a current-driven RLC ladder read on the
    // negative output node (swapped drive), a differential output (C's
    // column addition) with a zero at the origin, coupled inductors, a
    // capacitor across an ideal supply (an index-two block the port deflates),
    // and diode/MOS1 operating-point linearizations.
    pole_zero("pz_ladder_cur"),
    pole_zero("pz_bridge_diff"),
    pole_zero("pz_transformer"),
    pole_zero("pz_cv_loop"),
    pole_zero("pz_diode"),
    pole_zero("pz_mos1"),
    // The MESA load is `mesapzl.c` = `mesaacl.c` with `s` for `j omega`.
    pole_zero("m10_mesa_pz"),
    // `.tf` transfer function (#101): a passive ladder (inductor short,
    // capacitor open) and a current-driven E/G amplifier with a sensed current
    // output keep the linear DC bound; the Gummel-Poon stage, linearised at its
    // tightened-RELTOL operating point, the nonlinear 1 ppm bound.
    fixture("m8_tf_divider", Stage::transfer(compare::DC)),
    fixture("m8_tf_controlled", Stage::transfer(compare::DC)),
    fixture("m8_tf_bjt", Stage::transfer(compare::NONLINEAR)),
    // `.sens` (#102): C's finite-difference sensitivities, every parameter
    // `sgen` perturbs. Linear decks (DC, a hot deck with C's in-place
    // perturbation side effects, AC) under `compare::SENSITIVITY`; the diode
    // deck, at a tightened-RELTOL operating point, under
    // `compare::SENSITIVITY_NONLINEAR`.
    fixture("sens_divider", Stage::sensitivity_dc(compare::SENSITIVITY)),
    fixture(
        "sens_hot",
        Stage::sensitivity_dc(compare::SENSITIVITY_NONLINEAR),
    ),
    fixture(
        "sens_diode",
        Stage::sensitivity_dc(compare::SENSITIVITY_NONLINEAR),
    ),
    fixture("sens_ac", Stage::sensitivity_ac(compare::SENSITIVITY)),
    // AC `.sens` of nonlinear devices, switches and K (#102): every frequency
    // linearizes the devices at cktsens.c's zero-voltage reset load. Nonlinear
    // bound: the MESFET/HFET2 AC solves (and several rounding-noise columns of
    // repeated element contributions) differ from C by a few ulps.
    fixture(
        "sens_ac_reset",
        Stage::sensitivity_ac(compare::SENSITIVITY_NONLINEAR),
    ),
    fixture(
        "sens_ac_mutual",
        Stage::sensitivity_ac(compare::SENSITIVITY_NONLINEAR),
    ),
    fixture(
        "sens_ac_bjt",
        Stage::sensitivity_ac(compare::SENSITIVITY_NONLINEAR),
    ),
    fixture(
        "sens_ac_mos",
        Stage::sensitivity_ac(compare::SENSITIVITY_NONLINEAR),
    ),
    fixture(
        "sens_ac_fet",
        Stage::sensitivity_ac(compare::SENSITIVITY_NONLINEAR),
    ),
    fixture(
        "sens_ac_hetero",
        Stage::sensitivity_ac(compare::SENSITIVITY_NONLINEAR),
    ),
    // `.sens` of the `j` and `z` families (#164): C's `sgen` walk of the
    // `jfet.c`/`jfet2.c`/`mes.c`/`mesa.c`/`hfet1.c`/`hfet2.c` parameter
    // tables, replayed on the port's records. Same nonlinear bound as the
    // diode: weak parameters (`tnom`, `temp`) have sensitivities around
    // 1e-12, where a few ulps of the bias dominate the relative error.
    fixture(
        "sens_jfet",
        Stage::sensitivity_dc(compare::SENSITIVITY_NONLINEAR),
    ),
    fixture(
        "sens_jfet2",
        Stage::sensitivity_dc(compare::SENSITIVITY_NONLINEAR),
    ),
    fixture(
        "sens_mesfet",
        Stage::sensitivity_dc(compare::SENSITIVITY_NONLINEAR),
    ),
    // MESA level 2 (#164): the same walk over `mesa.c`'s 62 + 8 parameters, at
    // the cancellation floor `compare::SENSITIVITY_CANCELLED` documents for
    // three of them.
    fixture(
        "sens_mesa",
        Stage::sensitivity_dc(compare::SENSITIVITY_CANCELLED),
    ),
    // HFET1 (#164): `hfet1/hfet.c`'s 63 model + 7 instance parameters, with one
    // p-channel device so the walk re-runs `hfettemp.c`'s threshold-times-type
    // multiplication the way C does.
    fixture(
        "sens_hfet1",
        Stage::sensitivity_dc(compare::SENSITIVITY_NONLINEAR),
    ),
    // HFET2 (#164): `hfet2.c`'s 36 model + 6 instance parameters - `m` passes
    // the table filter but not `hfet2ask.c`'s switch - with `d2`/`eta2` given
    // so the `nsc` branch is on for both loads, a second n-channel instance at
    // `dtemp=10` and a p-channel at `temp=50`.
    fixture(
        "sens_hfet2",
        Stage::sensitivity_dc(compare::SENSITIVITY_NONLINEAR),
    ),
    // BJT (#164): `bjt.c`'s 103 model + 6 instance parameters (the `IF_AC`
    // capacitance family and the `IOPR` aliases are filtered, `icvbe`/`icvce`
    // are AC-only), an NPN with `dtemp=-10`, `ISE` above 1e-4 and a `BF`
    // polynomial plus a PNP at `dtemp=20` with `tnom` on the card.
    fixture(
        "sens_bjt",
        Stage::sensitivity_dc(compare::SENSITIVITY_NONLINEAR),
    ),
    // Classic MOS level 1: process-derived shared NMOS and explicit PMOS.
    fixture(
        "sens_mos1",
        Stage::sensitivity_dc(compare::SENSITIVITY_NONLINEAR),
    ),
    // Classic MOS level 2: process-derived shared NMOS and explicit PMOS.
    fixture(
        "sens_mos2",
        Stage::sensitivity_dc(compare::SENSITIVITY_NONLINEAR),
    ),
    // Classic MOS level 3: process-derived shared NMOS and explicit PMOS.
    fixture(
        "sens_mos3",
        Stage::sensitivity_dc(compare::SENSITIVITY_NONLINEAR),
    ),
    // Classic MOS level 6: process-derived shared NMOS and explicit PMOS.
    fixture(
        "sens_mos6",
        Stage::sensitivity_dc(compare::SENSITIVITY_NONLINEAR),
    ),
    // Classic MOS level 9: process-derived shared NMOS and explicit PMOS.
    fixture(
        "sens_mos9",
        Stage::sensitivity_dc(compare::SENSITIVITY_NONLINEAR),
    ),
    // MESA level 3 (#164): the `mesa2` walk, one device at the circuit
    // temperature and one at `dtemp=10`, both with `rtc1`/`rtc2` and the
    // `rdi`/`rsi` pair (which is where the channel's `tRsi` scaling showed).
    fixture(
        "sens_mesa3",
        Stage::sensitivity_dc(compare::SENSITIVITY_NONLINEAR),
    ),
    // MESA level 4 (#164): the same walk over the level-4 parameter table,
    // sensing the drain current (`sens_mesa` senses a ~1 uV gate node, where a
    // level-4 row is a cancellation-limited difference).
    fixture(
        "sens_mesa4",
        Stage::sensitivity_dc(compare::SENSITIVITY_NONLINEAR),
    ),
    // MESA level 4 at a shifted instance temperature (#164): the `rtc1`/`rtc2`
    // rows that used to come out at 0.808x C's, now the regression for
    // `mesa3`'s temperature-scaled `RS`.
    fixture(
        "sens_mesa4_ts",
        Stage::sensitivity_dc(compare::SENSITIVITY_NONLINEAR),
    ),
    // `.sp` (#105): linear port decks, so the AC bound applies to every
    // S/Y/Z entry, port node voltage and `v(rbase)`. `sp_rc` keeps a shunt
    // resistor so no Z/Y component is a rounding-level real part.
    fixture("sp_attenuator", Stage::s_parameter(compare::AC)),
    fixture("sp_rc", Stage::s_parameter(compare::AC)),
    // JFET level 1 (#82, M10 slice 1): output/transfer sweeps of both
    // polarities (inverse mode, forward gate bias, RD/RS, area/m, OFF), a
    // common-source AC stage and a temperature sweep under the nonlinear
    // 1 ppm bound (the DC decks tighten RELTOL so C's own Newton stopping
    // error stays below it), and a source follower/PJF stage transient under
    // `compare::TRAN` with `tmax` bounding both integrators' step error.
    fixture("m10_jfet_dc", Stage::dc(compare::NONLINEAR)),
    fixture("m10_jfet_ac", Stage::ac(compare::NONLINEAR)),
    tran("m10_jfet_tran"),
    fixture("m10_jfet_temp", Stage::dc(compare::NONLINEAR)),
    // JFET level 2, Parker-Skellern (#82, M10 slice 6): output/transfer
    // sweeps (subthreshold, power laws, drain feedback, self-heating,
    // breakdown, RD/RS, inverse mode, area/m, PJF), a dispersive (TAUG/TAUD)
    // common-source AC stage and a temperature sweep under the nonlinear
    // 1 ppm bound with tightened RELTOL as for level 1, and a gate-lag/
    // self-heating transient under `compare::TRAN`; that deck also tightens
    // RELTOL/VNTOL because `i(vin)` is the small difference of two nearly
    // equal node voltages over 1 k, which C's default Newton stopping error
    // on `v(g)` would otherwise exceed.
    fixture("m10_jfet2_dc", Stage::dc(compare::NONLINEAR)),
    fixture("m10_jfet2_ac", Stage::ac(compare::NONLINEAR)),
    tran("m10_jfet2_tran"),
    fixture("m10_jfet2_temp", Stage::dc(compare::NONLINEAR)),
    // URC lines (#85): urcsetup.c's R/C (and R/D) ladders with C's names. The
    // R/C decks are linear (AC bound, default TRAN); the ISPERL deck runs the
    // generated diodes and keeps the default TRAN bound.
    fixture("m10_urc_ac", Stage::ac(compare::AC)),
    tran("m10_urc_tran"),
    tran("m10_urc_diode_tran"),
    // MESFET level 1, Statz (#83, M10 wave 3): output/transfer sweeps of
    // both polarities (inverse mode, forward gate bias, RD/RS, area/m, VTO
    // alias, tail type flags) and a common-source AC stage under the
    // nonlinear 1 ppm bound (the DC deck tightens RELTOL/VNTOL/ABSTOL so C's
    // own Newton stopping error stays below it), and an NMF switching/PMF
    // stage transient under `compare::TRAN` with `tmax` 0.1 ns bounding both
    // integrators' step error on the Statz gate charge.
    fixture("m10_mesfet_dc", Stage::dc(compare::NONLINEAR)),
    fixture("m10_mesfet_ac", Stage::ac(compare::NONLINEAR)),
    tran("m10_mesfet_tran"),
    // Lossy lines (#85, LTRA): the port follows dctran.c, ltraacct.c and
    // ltratrun.c, so both simulators take the same timepoints and the
    // default TRAN bound holds with margin (worst 0.024 of it, from the RLC
    // line where ltramisc.c's coefficients branch on exact time equalities
    // at delayed source corners). No diffsol variants: the BDF backend
    // refuses delay lines.
    // The AC deck carries the exact complex two-ports.
    tran("m10_ltra_tran"),
    tran("m10_ltra_rc_tran"),
    fixture("m10_ltra_ac", Stage::ac(compare::AC)),
];

/// Pole-zero registry entry: `compare::POLE_ZERO`.
const fn pole_zero(name: &'static str) -> Supported {
    fixture(name, Stage::roots())
}
/// One plot of a multi-analysis fixture: the analysis type expected at this
/// position of the batch schedule and the gate its plot is compared under.
struct Stage {
    kind: AnalysisKind,
    gate: Gate,
}
impl Stage {
    const fn points(
        kind: AnalysisKind,
        axis: Option<&'static str>,
        tolerance: compare::Tolerance,
    ) -> Self {
        Self {
            kind,
            gate: Gate::Points {
                axis,
                tolerance,
                aliases: &[],
            },
        }
    }

    const fn op(tolerance: compare::Tolerance) -> Self {
        Self::points(AnalysisKind::OperatingPoint, None, tolerance)
    }

    const fn points_aliases(
        kind: AnalysisKind,
        tolerance: compare::Tolerance,
        aliases: &'static [(&'static str, &'static str)],
    ) -> Self {
        Self {
            kind,
            gate: Gate::Points {
                axis: None,
                tolerance,
                aliases,
            },
        }
    }

    const fn dc(tolerance: compare::Tolerance) -> Self {
        Self::points(AnalysisKind::DcSweep, None, tolerance)
    }

    const fn ac(tolerance: compare::Tolerance) -> Self {
        Self::points(AnalysisKind::Ac, Some("frequency"), tolerance)
    }

    const fn transfer(tolerance: compare::Tolerance) -> Self {
        Self::points(AnalysisKind::TransferFunction, None, tolerance)
    }

    const fn sensitivity_dc(tolerance: compare::Tolerance) -> Self {
        Self::points(AnalysisKind::Sensitivity, None, tolerance)
    }

    const fn sensitivity_ac(tolerance: compare::Tolerance) -> Self {
        Self::points(AnalysisKind::Sensitivity, Some("frequency"), tolerance)
    }

    const fn s_parameter(tolerance: compare::Tolerance) -> Self {
        Self::points(AnalysisKind::SParameter, Some("frequency"), tolerance)
    }

    const fn noise_spectrum(tolerance: compare::Tolerance) -> Self {
        Self::points(AnalysisKind::Noise, Some("frequency"), tolerance)
    }

    const fn noise_integrated(tolerance: compare::Tolerance) -> Self {
        Self::points(AnalysisKind::Noise, None, tolerance)
    }

    const fn distortion(tolerance: compare::Tolerance) -> Self {
        Self::points(AnalysisKind::Distortion, Some("frequency"), tolerance)
    }

    const fn transient() -> Self {
        Self {
            kind: AnalysisKind::Transient,
            gate: Gate::Transient(compare::TRAN),
        }
    }

    const fn roots() -> Self {
        Self {
            kind: AnalysisKind::PoleZero,
            gate: Gate::Roots(compare::POLE_ZERO),
        }
    }
}

/// Attach a name to a single-stage comparison policy. Batch entries use
/// the same stage constructors without an extra name.
const fn fixture(name: &'static str, stage: Stage) -> Supported {
    Supported {
        name,
        kind: stage.kind,
        gate: stage.gate,
    }
}

/// A multi-analysis fixture (#96): every analysis card of the deck runs in
/// ngspice batch order (`ngspice_rs::analysis::batch::schedule`) and plot `i` is
/// compared with plot `i` of the multi-plot C golden under `stages[i]`. The
/// plot count, the order and every plot name must match exactly; each stage
/// keeps the very tolerance a single-analysis fixture of that type uses.
struct Batch {
    name: &'static str,
    stages: &'static [Stage],
}

const NOISE_LINEAR_STAGES: [Stage; 2] = noise_stages(compare::NOISE);
const NOISE_NONLINEAR_STAGES: [Stage; 2] = noise_stages(compare::NOISE_NONLINEAR);

/// One distortion plot (#104) along `frequency`.
const DISTORTION_STAGE: Stage = Stage::distortion(compare::DISTORTION);
/// A deck with an intermodulation card (three plots, run first as the later
/// card) and a harmonic card (two plots).
const DISTORTION_IM_AND_HARMONIC_STAGES: [Stage; 5] = [
    DISTORTION_STAGE,
    DISTORTION_STAGE,
    DISTORTION_STAGE,
    DISTORTION_STAGE,
    DISTORTION_STAGE,
];

const BATCH: &[Batch] = &[
    // `.sens` (#102) after `.ac` and `.op`, with a name filter.
    Batch {
        name: "sens_multi",
        stages: &[
            Stage::ac(compare::AC),
            Stage::op(compare::DC),
            Stage::sensitivity_dc(compare::SENSITIVITY),
        ],
    },
    // Final transient instance fields survive `.sens`'s DC solve and resets.
    // Filtered to overlap/series/CJ columns: the linear sensitivity bound
    // catches substituting the DC-point capacitances for the final load's.
    Batch {
        name: "sens_ac_mos_tran",
        stages: &[
            Stage::op(compare::DC),
            Stage::transient(),
            Stage::sensitivity_ac(compare::SENSITIVITY),
        ],
    },
    // AC `.sens` (#102) after `.op`: the MOS AC loads read the capbd/capbs
    // the operating point's MODEINITSMSIG load left in the instances.
    Batch {
        name: "sens_ac_mos_op",
        stages: &[
            Stage::op(compare::DC),
            Stage::sensitivity_ac(compare::SENSITIVITY_NONLINEAR),
        ],
    },
    // `.noise` (#100): every `.noise` card writes a spectrum and an
    // integrated-noise plot, so the fixtures are batch fixtures. The RC deck
    // is linear (the AC-type bound with the noise floor); the diode, BJT and
    // MOS1 decks inherit the nonlinear 1 ppm operating-point bound.
    Batch {
        name: "noise_rc",
        stages: &NOISE_LINEAR_STAGES,
    },
    Batch {
        name: "noise_diode",
        stages: &NOISE_NONLINEAR_STAGES,
    },
    Batch {
        name: "noise_bjt",
        stages: &NOISE_NONLINEAR_STAGES,
    },
    Batch {
        name: "noise_mos1",
        stages: &NOISE_NONLINEAR_STAGES,
    },
    Batch {
        name: "m11_bsim4_noise",
        stages: &NOISE_NONLINEAR_STAGES,
    },
    // `.ac`, `.op` and two `.noise` cards with S/W switches: the later,
    // single-frequency card runs first and writes only its spectrum (`noise1`),
    // the decade card then writes `noise2` and `noise3`.
    Batch {
        name: "noise_multi",
        stages: &[
            Stage::ac(compare::NONLINEAR),
            Stage::op(compare::NONLINEAR),
            Stage::noise_spectrum(compare::NOISE_NONLINEAR),
            Stage::noise_spectrum(compare::NOISE_NONLINEAR),
            Stage::noise_integrated(compare::NOISE_NONLINEAR),
        ],
    },
    // `.disto` (#104): every card writes two harmonic or three
    // intermodulation plots, so the fixtures are batch fixtures. Each deck's
    // later IM card runs first (`disto1..3`), its harmonic card second
    // (`disto4 disto5`); `disto_multi` runs `.ac`, `.op` and a `lin` sweep.
    Batch {
        name: "disto_diode",
        stages: &DISTORTION_IM_AND_HARMONIC_STAGES,
    },
    Batch {
        name: "disto_bjt",
        stages: &DISTORTION_IM_AND_HARMONIC_STAGES,
    },
    Batch {
        name: "disto_mos1",
        stages: &DISTORTION_IM_AND_HARMONIC_STAGES,
    },
    Batch {
        name: "disto_multi",
        stages: &[
            Stage::ac(compare::NONLINEAR),
            Stage::op(compare::NONLINEAR),
            DISTORTION_STAGE,
            DISTORTION_STAGE,
        ],
    },
    Batch {
        name: "multi_analysis_rc",
        stages: &[
            Stage::ac(compare::AC),
            Stage::dc(compare::DC),
            Stage::op(compare::DC),
            Stage::transient(),
        ],
    },
    // The M6 exit gate: a SIN-driven K transformer into a G/E op-amp subcircuit
    // with `.param`/`.func`/`{expr}` values, a B limiter, an H current sense and
    // `.option reltol`, running every analysis card. Nonlinear (the B source), so
    // the DC-type plots and AC keep the nonlinear 1 ppm bound and the transient
    // `compare::TRAN`.
    Batch {
        name: "m6_gate",
        stages: &[
            Stage::ac(compare::NONLINEAR),
            Stage::dc(compare::NONLINEAR),
            Stage::op(compare::NONLINEAR),
            Stage::transient(),
        ],
    },
    // Convergence parity (#106): Schmitt triggers swept up and down through
    // their hysteresis (each `.dc` point warm-starts from the previous one,
    // as `dctrcurv.c` does, so the two sweeps follow different branches) and
    // an operating point inside the band, which C's (and the port's) CKTop
    // continuation settles on the middle branch for the BJT deck. ngspice
    // runs the sweeps before the operating point and the later `.dc` card
    // first. Nonlinear 1 ppm bound with tightened RELTOL/VNTOL.
    Batch {
        name: "m7_conv_bjt_schmitt",
        stages: &[
            Stage::dc(compare::NONLINEAR),
            Stage::dc(compare::NONLINEAR),
            Stage::op(compare::NONLINEAR),
        ],
    },
    Batch {
        name: "m7_conv_cmos_schmitt",
        stages: &[
            Stage::dc(compare::NONLINEAR),
            Stage::dc(compare::NONLINEAR),
            Stage::op(compare::NONLINEAR),
        ],
    },
    // `.tf` in a batch (#101): `.op` then two `.tf` cards in reverse deck
    // order, over a diode and a MOS1 stage; nonlinear 1 ppm bound.
    Batch {
        name: "m8_tf_batch",
        stages: &[
            Stage::op(compare::NONLINEAR),
            Stage::transfer(compare::NONLINEAR),
            Stage::transfer(compare::NONLINEAR),
        ],
    },
    // The former FMA-dependent exclusion is now covered by the
    // independently validated pole search (#91), including both PZ cards.
    Batch {
        name: "multi_analysis_pz",
        stages: &[
            Stage::ac(compare::AC),
            Stage::op(compare::DC),
            Stage::roots(),
            Stage::roots(),
        ],
    },
    // `.sp` (#105) beside `.ac`, `.op` and `.tran` on the same RF ports: the
    // series z0 of each port in every analysis, and `sp1` last in batch order.
    Batch {
        name: "sp_multi",
        stages: &[
            Stage::ac(compare::AC),
            Stage::op(compare::DC),
            Stage::transient(),
            Stage::s_parameter(compare::AC),
        ],
    },
];

/// The two plots of one `.noise` card (#100): the spectrum along `frequency`
/// and the one-point integrated noise.
const fn noise_stages(tolerance: compare::Tolerance) -> [Stage; 2] {
    [
        Stage::noise_spectrum(tolerance),
        Stage::noise_integrated(tolerance),
    ]
}

/// Fixtures whose deck the Rust engine does not reproduce. A requested excluded
/// fixture still fails the run, so an entry cannot be reported as a success by
/// accident.
const EXCLUDED: &[(&str, &str)] = &[];

pub(super) fn main(root: &Path, arguments: &[String]) -> Result<(), String> {
    let only = match arguments {
        [] => None,
        [flag, name] if flag == "--netlist" && !name.starts_with('-') => Some(name.as_str()),
        _ => return Err("usage: cargo xtask golden verify [--netlist <NAME>]".into()),
    };
    run(root, only)
}

fn run(root: &Path, only: Option<&str>) -> Result<(), String> {
    run_with_registries(root, only, SUPPORTED, BATCH, EXCLUDED)
}

/// [`run_with_registries`] without multi-analysis fixtures.
#[cfg(test)]
fn run_with_registry(
    root: &Path,
    only: Option<&str>,
    supported: &[Supported],
    excluded: &[(&str, &str)],
) -> Result<(), String> {
    run_with_registries(root, only, supported, &[], excluded)
}

/// The verification loop, parameterized by the fixture registry.
///
/// This signature lets a unit test drive the `requested unsupported fixture`
/// branch with a synthetic exclusion.
fn run_with_registries(
    root: &Path,
    only: Option<&str>,
    supported: &[Supported],
    batch: &[Batch],
    excluded: &[(&str, &str)],
) -> Result<(), String> {
    let paths = fixtures::netlist_paths_at(root, only)?;
    let mut verified = 0;
    let mut unsupported = 0;
    let mut failures = Vec::new();
    if only.is_none() {
        for name in supported
            .iter()
            .map(|fixture| fixture.name)
            .chain(batch.iter().map(|fixture| fixture.name))
        {
            if !paths.iter().any(|path| {
                path.file_stem()
                    .is_some_and(|stem| stem.eq_ignore_ascii_case(name))
            }) {
                failures.push(format!("missing supported fixture '{name}'"));
            }
        }
    }
    for path in paths {
        let name = path
            .file_stem()
            .and_then(|s| s.to_str())
            .ok_or("invalid fixture name")?;
        if let Some(fixture) = supported
            .iter()
            .find(|fixture| fixture.name.eq_ignore_ascii_case(name))
        {
            match fixture_result(root, &path, fixture) {
                Ok(details) => {
                    verified += 1;
                    println!("  verified   {name}");
                    for detail in details {
                        println!("             {detail}");
                    }
                }
                Err(error) => {
                    println!("  FAIL       {name}: {error}");
                    failures.push(format!("{name}: {error}"));
                }
            }
        } else if let Some(fixture) = batch
            .iter()
            .find(|fixture| fixture.name.eq_ignore_ascii_case(name))
        {
            match batch_result(root, &path, fixture) {
                Ok(details) => {
                    verified += 1;
                    println!("  verified   {name}");
                    for detail in details {
                        println!("             {detail}");
                    }
                }
                Err(error) => {
                    println!("  FAIL       {name}: {error}");
                    failures.push(format!("{name}: {error}"));
                }
            }
        } else if let Some((_, reason)) = excluded
            .iter()
            .find(|(fixture, _)| fixture.eq_ignore_ascii_case(name))
        {
            unsupported += 1;
            println!("  unsupported {name}: {reason}");
            if only.is_some() {
                failures.push(format!("requested unsupported fixture '{name}': {reason}"));
            }
        } else {
            failures.push(format!(
                "fixture '{name}' has no verification registry entry"
            ));
        }
    }
    println!(
        "\n{verified} verified fixture(s), {unsupported} unsupported fixture(s), {} failure(s); bounded coverage, not full corpus parity",
        failures.len()
    );
    if failures.is_empty() {
        Ok(())
    } else {
        Err(format!("verification failed: {}", failures.join("\n")))
    }
}

/// Detail lines for a verified fixture (one run: the deck as written).
fn fixture_result(root: &Path, path: &Path, fixture: &Supported) -> Result<Vec<String>, String> {
    Ok(vec![run_deck(root, path, fixture)?])
}

/// One Rust run of the deck compared with the committed C golden. Deck
/// `.option` cards are applied through `RunConfig`, exactly as for an ordinary
/// run.
fn run_deck(root: &Path, path: &Path, fixture: &Supported) -> Result<String, String> {
    let netlist = Parser::new().parse_file(path).map_err(|e| e.to_string())?;
    if netlist.analyses.len() != 1 {
        return Err(format!(
            "expected exactly one analysis, found {}",
            netlist.analyses.len()
        ));
    }
    let config = RunConfig::from_netlist(&netlist).map_err(|e| e.to_string())?;
    let request = config
        .plan_for(&netlist.analyses[0])
        .map_err(|error| error.to_string())?;
    let (mut plots, circuit, resolution) = run_plan_plots(&netlist, &config, &request, None)?;
    if plots.len() != 1 {
        return Err(format!(
            "expected one Rust plot, the driver produced {} (register multi-plot analyses as batch fixtures)",
            plots.len()
        ));
    }
    let got = plots.remove(0);
    if request.kind() != fixture.kind {
        return Err(format!(
            "registry expects {:?}, deck requests {:?}",
            fixture.kind,
            request.kind()
        ));
    }
    let mut golden = Golden::open(root, fixture.name)?;
    let detail = golden.compare(&fixture.gate, &circuit, &request, &got, resolution.as_ref())?;
    golden.finish()?;
    Ok(detail)
}

/// Every plot of a multi-analysis fixture, in batch order, against the
/// multi-plot C golden. Each analysis runs on a freshly elaborated circuit, as
/// `spice-rs simulate` does.
fn batch_result(root: &Path, path: &Path, fixture: &Batch) -> Result<Vec<String>, String> {
    let netlist = Parser::new().parse_file(path).map_err(|e| e.to_string())?;
    let config = RunConfig::from_netlist(&netlist).map_err(|e| e.to_string())?;
    let plans = config
        .plans_for(&netlist.analyses)
        .map_err(|e| e.to_string())?;
    let schedule = ngspice_rs::analysis::batch::schedule(&plans);
    let mut golden = Golden::open(root, fixture.name)?;
    let names: Vec<&str> = schedule
        .iter()
        .flat_map(ngspice_rs::analysis::batch::ScheduledAnalysis::plot_names)
        .collect();
    if names.len() != fixture.stages.len() {
        return Err(format!(
            "expected {} plots: the deck schedules {}",
            fixture.stages.len(),
            names.len()
        ));
    }
    let mut details = Vec::with_capacity(names.len());
    let mut stages = fixture.stages.iter();
    // Inherit the actual matrix and device load snapshots, as the CLI does;
    // analysis-kind markers alone cannot reproduce transient instance fields.
    let mut previous = None;
    for entry in &schedule {
        let request = &plans[entry.card_index];
        let (plots, circuit, resolution) =
            run_plan_plots(&netlist, &config, request, previous.as_ref())
                .map_err(|error| format!("{}: {error}", entry.plot_name))?;
        let own: Vec<&str> = entry.plot_names().collect();
        if plots.len() != own.len() {
            return Err(format!(
                "{}: the driver produced {} plot(s), the schedule names {}",
                entry.plot_name,
                plots.len(),
                own.len()
            ));
        }
        for (got, name) in plots.iter().zip(own) {
            let stage = stages.next().ok_or("more Rust plots than stages")?;
            if request.kind() != stage.kind {
                return Err(format!(
                    "{name}: registry expects {:?}, the batch schedule runs {:?}",
                    stage.kind,
                    request.kind()
                ));
            }
            let detail = golden
                .compare(&stage.gate, &circuit, request, got, resolution.as_ref())
                .map_err(|error| format!("{name}: {error}"))?;
            details.push(format!("{name} ({}): {detail}", got.plotname));
        }
        previous = Some(circuit);
    }
    golden.finish()?;
    Ok(details)
}

fn run_plan_plots(
    netlist: &ngspice_rs::netlist::ast::Netlist,
    config: &RunConfig,
    request: &AnalysisPlan,
    previous: Option<&ngspice_rs::devices::Circuit>,
) -> Result<
    (
        Vec<ngspice_rs::analysis::Plot>,
        ngspice_rs::devices::Circuit,
        Option<ngspice_rs::analysis::Plot>,
    ),
    String,
> {
    let mut circuit = config.circuit(netlist).map_err(|e| e.to_string())?;
    if let Some(previous) = previous {
        circuit.inherit_from(previous);
    }
    let (plots, resolution) = if request.kind() == AnalysisKind::Sensitivity {
        let (plot, resolution) = request
            .run_sensitivity_with_resolution(&mut circuit, &config.context())
            .map_err(|e| e.to_string())?;
        (vec![plot], Some(resolution))
    } else {
        (
            request
                .run_plots(&mut circuit, &config.context())
                .map_err(|e| e.to_string())?,
            None,
        )
    };
    circuit.note_completed(request.kind());
    // C's default save set omits device internal nodes by name
    // (`outitf.c`: `#internal`, `#source`, `#drain`, `#collector`,
    // `#collCX`, `#emitter`, `#base`) and saves every other node, including
    // an RF port's `#res`, a line's `#i1`/`#int1`, MESA's `#gate`/`#gs`/`#gd`
    // and BSIM4's `#gate`/`#midgate`/`#body`/`#charge`. Project only those;
    // every saved variable still goes through exact set checks.
    let internal: HashSet<_> = circuit
        .nodes()
        .nodes()
        .iter()
        .filter(|node| {
            node.kind == ngspice_rs::primitives::NodeKind::Internal
                && C_UNSAVED_SUFFIXES.iter().any(|suffix| {
                    // The port folds node names to lowercase (`#collcx`).
                    node.name
                        .to_ascii_lowercase()
                        .contains(&suffix.to_ascii_lowercase())
                })
        })
        .map(|node| format!("v({})", node.name))
        .collect();
    let plots = plots
        .into_iter()
        .map(|plot| project(plot, &internal))
        .collect::<Result<Vec<_>, _>>()?;
    let resolution = resolution
        .map(|plot| project(plot, &internal))
        .transpose()?;
    Ok((plots, circuit, resolution))
}

/// The internal-node name fragments `outitf.c` leaves out of C's default
/// save set.
const C_UNSAVED_SUFFIXES: [&str; 7] = [
    "#internal",
    "#source",
    "#drain",
    "#collector",
    "#collCX",
    "#emitter",
    "#base",
];

/// Projects one Rust plot onto C's default save set (see
/// [`run_plan_plots`]). Vector names already follow C, including the DC
/// sweep scales (`dctrcurv.c`).
fn project(
    got: ngspice_rs::analysis::Plot,
    internal: &HashSet<String>,
) -> Result<ngspice_rs::analysis::Plot, String> {
    let columns: Vec<_> = got
        .variables()
        .iter()
        .enumerate()
        .filter(|(_, variable)| !internal.contains(&variable.name))
        .map(|(index, _)| index)
        .collect();
    if columns.len() == got.variable_count() {
        return Ok(got);
    }
    let variables = columns
        .iter()
        .map(|&index| got.variables()[index].clone())
        .collect();
    let output = ngspice_rs::analysis::Plot::new(
        got.name.clone(),
        got.plotname.clone(),
        got.flags(),
        variables,
    );
    let width = got.variable_count();
    let points = got.point_count();
    let mut cells = got.into_cells();
    // The selected indices are strictly increasing; every destination is at
    // or before its source, so stable in-place compaction cannot overwrite
    // any unread cell.
    let mut destination = 0;
    for point in 0..points {
        for &column in &columns {
            cells[destination] = cells[point * width + column];
            destination += 1;
        }
    }
    cells.truncate(destination);
    output
        .with_cells(points, cells)
        .map_err(|error| error.to_string())
}

/// Preserve every C reference vector; append the explicitly certified
/// missing poles to its unordered reference root set.
fn complete_pole_reference(
    want: &ngspice_rs::analysis::Plot,
    additional_poles: &[f64],
) -> Result<ngspice_rs::analysis::Plot, String> {
    use ngspice_rs::analysis::Variable;
    use ngspice_rs::primitives::Complex;
    if want.point_count() != 1 {
        return Err("the pole-zero reference must have one point".to_owned());
    }
    let mut variables = want.variables().to_vec();
    let mut row = want
        .row(0)
        .ok_or("missing pole-zero reference row")?
        .to_vec();
    let poles = variables
        .iter()
        .filter(|v| v.name.starts_with("v(pole("))
        .count();
    for (index, &pole) in additional_poles.iter().enumerate() {
        variables.push(Variable::complex(
            format!("v(pole({}))", poles + index + 1),
            "voltage",
        ));
        row.push(Complex::real(pole));
    }
    let mut completed = ngspice_rs::analysis::Plot::new(
        want.name.clone(),
        want.plotname.clone(),
        want.flags(),
        variables,
    );
    completed
        .push_point(&row)
        .map_err(|error| error.to_string())?;
    Ok(completed)
}

/// Expand only explicitly declared reference-equivalent coordinates.
fn reference_aliases(
    plot: &ngspice_rs::analysis::Plot,
    reference: &ngspice_rs::analysis::Plot,
    aliases: &[(&str, &str)],
) -> Result<ngspice_rs::analysis::Plot, String> {
    let mut variables = Vec::with_capacity(plot.variables().len() + aliases.len());
    let mut columns = Vec::with_capacity(variables.capacity());
    // A native coordinate may have a differently named reference equivalent
    // or multiple redundant reference coordinates. Retain every native
    // column that is not explicitly represented under another name.
    for (index, variable) in plot.variables().iter().enumerate() {
        if !aliases
            .iter()
            .any(|&(_, original)| original == variable.name)
            || reference.variable_index(&variable.name).is_some()
        {
            variables.push(variable.clone());
            columns.push(index);
        }
    }
    for &(alias, original) in aliases {
        if variables.iter().any(|v| v.name == alias)
            || plot.variable_index(alias).is_some()
            || reference.variable_index(alias).is_none()
        {
            return Err(format!("invalid reference alias {alias}"));
        }
        let index = plot
            .variable_index(original)
            .ok_or_else(|| format!("missing physical coordinate {original} for {alias}"))?;
        let mut variable = plot.variables()[index].clone();
        alias.clone_into(&mut variable.name);
        variables.push(variable);
        columns.push(index);
    }
    let mut expanded =
        ngspice_rs::analysis::Plot::new(&plot.name, &plot.plotname, plot.flags(), variables);
    for row in plot.rows() {
        let data: Vec<_> = columns.iter().map(|&index| row[index]).collect();
        expanded
            .push_point(&data)
            .map_err(|error| error.to_string())?;
    }
    Ok(expanded)
}

/// Compares one Rust plot with one C plot under `gate`.
fn compare_plot(
    gate: &Gate,
    circuit: &ngspice_rs::devices::Circuit,
    request: &AnalysisPlan,
    got: &ngspice_rs::analysis::Plot,
    rows: RawPlotRows<'_, impl BufRead>,
    resolution: Option<&ngspice_rs::analysis::Plot>,
    alternates: &[ngspice_rs::analysis::Plot],
) -> Result<String, String> {
    if !alternates.is_empty() && !matches!(gate, Gate::Points { .. }) {
        return Err("independent reference envelopes require a point-wise gate".to_owned());
    }
    // A transmission-line deck is a transient comparison whose breakpoints
    // also include the declared echoes.
    let (echo_gate, offsets) = match *gate {
        Gate::Echoes { tolerance, offsets } => (Gate::Transient(tolerance), offsets),
        _ => (Gate::Transient(compare::TRAN), &[][..]),
    };
    let gate = if matches!(gate, Gate::Echoes { .. }) {
        &echo_gate
    } else {
        gate
    };
    match *gate {
        Gate::Points {
            axis,
            tolerance,
            aliases,
        } => {
            let expanded = if aliases.is_empty() {
                None
            } else {
                Some(reference_aliases(got, &rows.header().plot, aliases)?)
            };
            let got = expanded.as_ref().unwrap_or(got);
            compare::stream_points(got, rows, tolerance, axis, resolution, alternates)
                .map(|()| format!("{} point(s)", got.point_count()))
        }
        Gate::Roots(tolerance) => {
            let want = stream::collect_plot(rows)?;
            compare::roots(got, &want, tolerance)
        }
        Gate::CompletedRoots {
            tolerance,
            additional_poles,
        } => {
            let want = stream::collect_plot(rows)?;
            let completed = complete_pole_reference(&want, additional_poles)?;
            compare::roots(got, &completed, tolerance).map(|summary| {
                format!(
                    "{summary}; includes {} independently verified pole(s)",
                    additional_poles.len()
                )
            })
        }
        Gate::Transient(tolerance) => {
            let want = stream::collect_plot(rows)?;
            let transient = request.transient().ok_or("expected a transient plan")?;
            let (step, final_time) = (transient.step(), transient.stop());
            let timing = ngspice_rs::devices::TransientTiming::new(step, final_time)
                .map_err(|e| e.to_string())?;
            let mut breakpoints = tran::breakpoints(circuit, &timing)?;
            if !offsets.is_empty() {
                breakpoints = tran::echoes(&breakpoints, offsets, final_time)?;
            }
            // With `uic` C writes no t = 0 row: its first row is the first
            // accepted step. The comparison then starts at that time, which the
            // Rust plot must reproduce (`tran::Series::new`); without `uic`
            // both plots start at 0 as always.
            let start = if request.uic() {
                want.value("time", 0)
                    .map(|t| t.re)
                    .filter(|t| t.is_finite() && *t > 0.0)
                    .ok_or("uic golden has no positive first time")?
            } else {
                0.0
            };
            tran::transient(
                got,
                &want,
                tolerance,
                &tran::Grid {
                    start,
                    stop: final_time,
                    step,
                },
                &breakpoints,
            )
            .map(|summary| {
                format!(
                    "{} instants + {} breakpoint limits, worst error {:.3} of bound",
                    summary.instants, summary.limits, summary.worst_ratio
                )
            })
        }
        // Mapped to `Gate::Transient` with its offsets above.
        Gate::Echoes { .. } => Err("echo gate not mapped".into()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct Temp(std::path::PathBuf);
    impl Temp {
        fn divider() -> Self {
            Self::with(&["rc_divider"])
        }
        fn with(names: &[&str]) -> Self {
            static ID: AtomicUsize = AtomicUsize::new(0);
            let root = std::env::temp_dir().join(format!(
                "spice-verify-{}-{}",
                std::process::id(),
                ID.fetch_add(1, Ordering::Relaxed)
            ));
            fs::create_dir_all(root.join(fixtures::NETLIST_DIR)).unwrap();
            fs::create_dir_all(root.join(fixtures::GOLDEN_DIR)).unwrap();
            for name in names {
                let deck = format!("conformance/netlists/{name}.cir");
                fs::copy(workspace_root().join(&deck), root.join(&deck)).unwrap();
                let original = workspace_root().join(&deck);
                let reference =
                    fixtures::reference_netlist_path(&workspace_root(), &original).unwrap();
                if reference != original {
                    let destination =
                        fixtures::reference_netlist_path(&root, &root.join(&deck)).unwrap();
                    fs::create_dir_all(destination.parent().unwrap()).unwrap();
                    fs::copy(reference, destination).unwrap();
                }
                for target in fixtures::numerical_golden_paths(&root, name) {
                    fs::create_dir_all(target.parent().unwrap()).unwrap();
                    let relative = target.strip_prefix(&root).unwrap();
                    fs::copy(workspace_root().join(relative), &target).unwrap();
                }
            }
            Self(root)
        }
    }
    impl Drop for Temp {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn internal_projection_compacts_owned_cells_without_hiding_unsaved_columns() {
        use ngspice_rs::analysis::{Plot, PlotFlags, Variable};
        use ngspice_rs::primitives::Complex;
        let plot = Plot::new(
            "op",
            "Operating Point",
            PlotFlags::Real,
            ["v(a#internal)", "v(out)", "v(b#drain)", "v(other)"]
                .into_iter()
                .map(|name| Variable::new(name, "voltage"))
                .collect(),
        )
        .with_rows([[1., 2., 3., 4.], [5., 6., 7., 8.]].map(|row| row.map(Complex::real)))
        .unwrap();
        let allocation = plot.cells().as_ptr();
        let result = project(
            plot,
            &HashSet::from(["v(a#internal)".into(), "v(b#drain)".into()]),
        )
        .unwrap();
        assert_eq!(result.cells().as_ptr(), allocation);
        assert_eq!(
            result
                .variables()
                .iter()
                .map(|v| v.name.as_str())
                .collect::<Vec<_>>(),
            ["v(out)", "v(other)"]
        );
        assert_eq!(result.cells(), &[2., 4., 6., 8.].map(Complex::real));
    }

    // Separate ignored tests make RSS and allocator measurements comparable
    // across processes. Synthetic creation is identical in both modes.
    fn pointwise_memory_fixture(stages: usize, streaming: bool) {
        use ngspice_rs::analysis::{
            BinaryByteOrder, Plot, PlotFlags, RawFileReader, RawFileWriter, RawFormat, Variable,
        };
        use ngspice_rs::primitives::Complex;
        use std::io::BufReader;
        let root = Temp::with(&[]);
        let path = root.0.join("pointwise.raw");
        let mut got = Plot::new(
            "op",
            "Operating Point",
            PlotFlags::Real,
            (0..16)
                .map(|index| Variable::new(format!("v(n{index})"), "voltage"))
                .collect(),
        );
        got.reserve_points(65_536).unwrap();
        for point in 0..65_536_u32 {
            got.push_point(&[Complex::real(f64::from(point)); 16])
                .unwrap();
        }
        let got = RawFile::single(got);
        let mut writer =
            RawFileWriter::create(&path, RawFormat::Ascii, BinaryByteOrder::LittleEndian).unwrap();
        for _ in 0..stages {
            writer.write_plot(&got.plots[0]).unwrap();
        }
        writer.finish().unwrap();
        let actual = &got.plots[0].plot;
        if streaming {
            let mut reader = RawFileReader::new(BufReader::new(fs::File::open(&path).unwrap()));
            for _ in 0..stages {
                compare::stream_points(
                    actual,
                    reader.next_plot().unwrap().unwrap(),
                    compare::DC,
                    None,
                    None,
                    &[],
                )
                .unwrap();
            }
            assert!(reader.next_plot().unwrap().is_none());
        } else {
            let retained = RawFile::parse(&fs::read_to_string(&path).unwrap()).unwrap();
            assert_eq!(retained.plots.len(), stages);
            for expected in &retained.plots {
                compare::plots(actual, &expected.plot, &[], compare::DC, None, None).unwrap();
            }
        }
        eprintln!(
            "comparison input: {} bytes, {stages} stages",
            fs::metadata(&path).unwrap().len()
        );
    }

    #[test]
    #[ignore = "large pointwise resource benchmark; run alone"]
    fn memory_pointwise_retained() {
        pointwise_memory_fixture(1, false);
    }

    #[test]
    #[ignore = "large pointwise resource benchmark; run alone"]
    fn memory_pointwise_streamed() {
        pointwise_memory_fixture(1, true);
    }

    #[test]
    #[ignore = "large multistage resource benchmark; run alone"]
    fn memory_batch_retained() {
        pointwise_memory_fixture(4, false);
    }

    #[test]
    #[ignore = "large multistage resource benchmark; run alone"]
    fn memory_batch_streamed() {
        pointwise_memory_fixture(4, true);
    }

    fn transient_memory_fixture(streaming: bool) {
        use ngspice_rs::analysis::RawFileReader;
        use std::io::BufReader;
        let path = fixtures::golden_path(&workspace_root(), "m7_ic_mos1_uic_tran");
        let got = if streaming {
            let mut reader = RawFileReader::new(BufReader::new(fs::File::open(&path).unwrap()));
            let plot = stream::collect_plot(reader.next_plot().unwrap().unwrap()).unwrap();
            assert!(reader.next_plot().unwrap().is_none());
            plot
        } else {
            let mut file = RawFile::parse(&fs::read_to_string(&path).unwrap()).unwrap();
            assert_eq!(file.plots.len(), 1);
            file.plots.remove(0).plot
        };
        assert!(got.point_count() > 0);
        assert!(got.cells().iter().all(|value| value.is_finite()));
        eprintln!(
            "real transient input: {} bytes; {} points",
            fs::metadata(&path).unwrap().len(),
            got.point_count()
        );
    }

    #[test]
    #[ignore = "transient decoder resource benchmark; run alone"]
    fn memory_transient_retained() {
        transient_memory_fixture(false);
    }

    #[test]
    #[ignore = "transient decoder resource benchmark; run alone"]
    fn memory_transient_streamed() {
        transient_memory_fixture(true);
    }

    #[test]
    fn production_fixtures_pass_without_modifying_inputs() {
        let paths: Vec<_> = fixtures::netlist_paths_at(&workspace_root(), None)
            .unwrap()
            .into_iter()
            .flat_map(|p| {
                let paths = fixtures::numerical_golden_paths(
                    &workspace_root(),
                    p.file_stem().unwrap().to_str().unwrap(),
                );
                let reference = fixtures::reference_netlist_path(&workspace_root(), &p).unwrap();
                [p, reference].into_iter().chain(paths)
            })
            .collect();
        let before: Vec<_> = paths.iter().map(|p| fs::read(p).unwrap()).collect();
        run(&workspace_root(), None).unwrap();
        // The full run already verifies every fixture; a selected run per
        // registry kind covers `--netlist` dispatch without a second pass.
        run(&workspace_root(), Some(SUPPORTED[0].name)).unwrap();
        run(&workspace_root(), Some(BATCH[0].name)).unwrap();
        for (path, bytes) in paths.iter().zip(before) {
            assert_eq!(fs::read(path).unwrap(), bytes);
        }
    }

    #[test]
    fn reference_input_variants_change_only_numeric_spellings() {
        let root = workspace_root();
        for original in fixtures::netlist_paths_at(&root, None).unwrap() {
            let reference = fixtures::reference_netlist_path(&root, &original).unwrap();
            if reference == original {
                continue;
            }
            let a = fs::read_to_string(&original).unwrap();
            let b = fs::read_to_string(&reference).unwrap();
            assert_eq!(a.lines().count(), b.lines().count());
            let mut changed = 0;
            for (left, right) in a.lines().zip(b.lines()) {
                if left == right {
                    continue;
                }
                assert!(!left.starts_with('*'), "changed comment: {left}");
                let left: Vec<_> = left.split_ascii_whitespace().collect();
                let right: Vec<_> = right.split_ascii_whitespace().collect();
                assert_eq!(left.len(), right.len());
                for (x, y) in left.into_iter().zip(right) {
                    if x == y {
                        continue;
                    }
                    let (key_x, value_x) = x.split_once('=').unwrap_or(("", x));
                    let (key_y, value_y) = y.split_once('=').unwrap_or(("", y));
                    assert_eq!(key_x, key_y);
                    let number = ngspice_rs::primitives::value::parse_spice_number;
                    assert_eq!(
                        number(value_x).unwrap().to_bits(),
                        number(value_y).unwrap().to_bits(),
                        "{x} != {y} in {}",
                        original.display()
                    );
                    changed += 1;
                }
            }
            assert!(
                changed > 0,
                "redundant reference deck: {}",
                reference.display()
            );
        }
    }

    #[test]
    fn missing_exact_input_capture_does_not_fall_back_to_original() {
        let temp = Temp::with(&["sens_mos3"]);
        let historical = temp.0.join("conformance/golden/sens_mos3.raw");
        fs::copy(
            workspace_root().join("conformance/golden/sens_mos3.raw"),
            &historical,
        )
        .unwrap();
        let selected = fixtures::golden_path(&temp.0, "sens_mos3");
        let original = fs::read_to_string(&selected).unwrap();
        run(&temp.0, Some("sens_mos3")).unwrap();
        fs::remove_file(&selected).unwrap();
        let error = run(&temp.0, Some("sens_mos3")).unwrap_err();
        assert!(error.contains(&selected.display().to_string()), "{error}");

        let mut raw = RawFile::parse(&original).unwrap();
        let plot = &mut raw.plots[0].plot;
        let index = plot.variable_index("v(m1:tox)").unwrap();
        plot.point_mut(0).unwrap()[index] = ngspice_rs::primitives::Complex::ZERO;
        fs::write(selected, raw.to_ascii().unwrap()).unwrap();
        let error = run(&temp.0, Some("sens_mos3")).unwrap_err();
        assert!(error.contains("v(m1:tox)"), "{error}");
    }

    #[test]
    #[ignore = "requires the independent C oracle in NGSPICE_BIN"]
    fn capture_check_verify_use_the_same_selected_reference() {
        use super::super::golden;
        let names = [
            "rc_divider",
            "multi_analysis_rc",
            "sens_bjt",
            "sens_hfet2",
            "sens_mos3",
            "sens_mos9",
        ];
        let temp = Temp::with(&names);
        let root = &temp.0;
        let binary = std::env::var("NGSPICE_BIN").unwrap();
        assert!(Path::new(&binary).is_absolute());
        // Exercise case-insensitive deck IDs without perturbing the real fixtures.
        let netlists = root.join(fixtures::NETLIST_DIR);
        fs::rename(netlists.join("sens_bjt.cir"), netlists.join("SENS_BJT.CIR")).unwrap();
        for name in names {
            let selected = fixtures::golden_path(root, name);
            let historical = root.join(fixtures::GOLDEN_DIR).join(format!("{name}.raw"));
            let alternative = if name == "sens_bjt" && selected == historical {
                root.join("conformance/golden/linux-x86_64-gnu/sens_bjt.raw")
            } else {
                historical
            };
            let other_bytes = (name == "sens_bjt").then(|| fs::read(&alternative).unwrap());
            let original = root.join(fixtures::NETLIST_DIR).join(format!("{name}.cir"));
            let reference = fixtures::reference_netlist_path(root, &original).unwrap();
            if reference != original && name != "sens_bjt" {
                fs::write(&alternative, "not the selected reference").unwrap();
            }
            fs::remove_file(&selected).unwrap();
            let c_command = |command: &str| {
                [command, "--netlist", name, "--ngspice", &binary].map(str::to_owned)
            };
            assert!(golden::main(root, &c_command("check")).is_err());
            golden::main(root, &c_command("capture")).unwrap();
            let bytes = fs::read(&selected).unwrap();
            golden::main(root, &c_command("check")).unwrap();
            run(root, Some(name)).unwrap();
            assert_eq!(fs::read(&selected).unwrap(), bytes);
            if let Some(other_bytes) = other_bytes {
                assert_eq!(fs::read(alternative).unwrap(), other_bytes);
            } else if reference != original {
                assert_eq!(
                    fs::read_to_string(alternative).unwrap(),
                    "not the selected reference"
                );
            }
        }
    }

    #[test]
    fn every_bjt_capture_is_required_and_structurally_checked() {
        let temp = Temp::with(&["sens_bjt"]);
        let paths = fixtures::numerical_golden_paths(&temp.0, "sens_bjt");
        run(&temp.0, Some("sens_bjt")).unwrap();
        for path in &paths {
            let original = fs::read_to_string(path).unwrap();
            fs::remove_file(path).unwrap();
            let error = run(&temp.0, Some("sens_bjt")).unwrap_err();
            assert!(error.contains("reading"), "{error}");
            assert!(error.contains(&path.display().to_string()), "{error}");
            fs::write(path, "not a rawfile").unwrap();
            assert!(run(&temp.0, Some("sens_bjt")).is_err());
            fs::write(path, original).unwrap();
        }
        // A prediction outside the recorded interval must still fail, even
        // when both captures are present and their metadata is valid.
        for path in &paths {
            let mut raw = RawFile::parse(&fs::read_to_string(path).unwrap()).unwrap();
            let plot = &mut raw.plots[0].plot;
            let index = plot.variable_index("v(q1:rco)").unwrap();
            plot.point_mut(0).unwrap()[index] = ngspice_rs::primitives::Complex::real(1.);
            fs::write(path, raw.to_ascii().unwrap()).unwrap();
        }
        let error = run(&temp.0, Some("sens_bjt")).unwrap_err();
        assert!(error.contains("v(q1:rco)"), "{error}");
    }

    #[test]
    fn corrupted_transient_goldens_fail_the_companion_run() {
        let temp = Temp::with(&["floating_cap_tran"]);
        let raw = temp.0.join("conformance/golden/floating_cap_tran.raw");
        let original = fs::read_to_string(&raw).unwrap();
        run(&temp.0, Some("floating_cap_tran")).unwrap();
        // Perturb v(b) of point 600 (t ~ 6 ms, a smooth interior sample) by 1 mV:
        // far above the 1e-3 relative + 1 uV bound at v(b) ~ 1e-3 V.
        let mut lines: Vec<String> = original.lines().map(String::from).collect();
        let start = lines
            .iter()
            .position(|line| line.starts_with(" 600\t"))
            .unwrap();
        let name = lines
            .iter()
            .position(|l| l.ends_with("v(b)\tvoltage"))
            .unwrap();
        let column: usize = lines[name]
            .trim_start()
            .split('\t')
            .next()
            .unwrap()
            .parse()
            .unwrap();
        let line = &mut lines[start + column];
        let value: f64 = line.trim().parse().unwrap();
        *line = format!("\t{:.15e}", value + 1e-3);
        fs::write(&raw, lines.join("\n") + "\n").unwrap();
        let error = run(&temp.0, Some("floating_cap_tran")).unwrap_err();
        assert!(error.contains("transient mismatch"), "{error}");
    }

    #[test]
    fn requested_unsupported_unknown_and_bad_options_fail() {
        // Drive the excluded branch with a synthetic entry as well as the
        // committed exclusions below.
        let error = run_with_registry(
            &workspace_root(),
            Some("rc_divider"),
            &[],
            &[("rc_divider", "synthetic exclusion")],
        )
        .unwrap_err();
        assert!(
            error.contains("requested unsupported fixture 'rc_divider': synthetic exclusion"),
            "{error}"
        );
        for (name, _) in EXCLUDED {
            assert!(
                run(&workspace_root(), Some(name))
                    .unwrap_err()
                    .contains("requested unsupported")
            );
        }
        assert!(run(&workspace_root(), Some("unknown")).is_err());
        for args in [
            vec!["--netlist"],
            vec!["--ngspice", "anything"],
            vec!["--verbose"],
            vec!["--netlist", "--verbose"],
            vec!["--netlist", "rc_divider", "extra"],
        ] {
            assert!(
                main(
                    &workspace_root(),
                    &args.into_iter().map(String::from).collect::<Vec<_>>()
                )
                .is_err()
            );
        }
    }

    #[test]
    fn corrupted_missing_and_multi_plot_goldens_fail_through_verification() {
        let temp = Temp::divider();
        let raw = temp.0.join("conformance/golden/rc_divider.raw");
        let original = fs::read_to_string(&raw).unwrap();
        assert!(
            run(&temp.0, None)
                .unwrap_err()
                .contains("missing supported fixture")
        );
        run(&temp.0, Some("rc_divider")).unwrap();
        for text in [
            original.replace("2.500000000000000e+00", "2.600000000000000e+00"),
            original.replace("v(out)", "v(wrong)"),
            original.replace("2.500000000000000e+00", "NaN"),
            "not a rawfile".into(),
            original.trim_end().rsplit_once('\n').unwrap().0.to_owned(),
            format!("{original}trailing junk\n"),
            format!("{original}{original}"),
        ] {
            fs::write(&raw, text).unwrap();
            assert!(run(&temp.0, Some("rc_divider")).is_err());
        }
        fs::remove_file(&raw).unwrap();
        assert!(
            run(&temp.0, Some("rc_divider"))
                .unwrap_err()
                .contains("reading")
        );
    }

    #[test]
    fn deck_errors_are_not_silently_skipped() {
        let temp = Temp::divider();
        let deck = temp.0.join("conformance/netlists/rc_divider.cir");
        let original = fs::read_to_string(&deck).unwrap();
        for text in [
            original.replace(".op", ""),
            original.replace(".op", ".op\n.op"),
            original.replace(".op", ".ac lin 3 100 1k"),
            original.replace("r2 out 0 1k", "r2 out nowhere 1k"),
            original.replace("r2 out 0 1k", "r2 out 0 {expr}"),
        ] {
            fs::write(&deck, text).unwrap();
            assert!(run(&temp.0, Some("rc_divider")).is_err());
        }
        fs::write(
            temp.0.join("conformance/netlists/new_fixture.cir"),
            original,
        )
        .unwrap();
        assert!(
            run(&temp.0, Some("new_fixture"))
                .unwrap_err()
                .contains("no verification registry")
        );
    }

    #[test]
    fn multi_plot_goldens_are_verified_plot_by_plot_and_in_order() {
        let temp = Temp::with(&["multi_analysis_rc"]);
        let raw = temp.0.join("conformance/golden/multi_analysis_rc.raw");
        let deck = temp.0.join("conformance/netlists/multi_analysis_rc.cir");
        let original = fs::read_to_string(&raw).unwrap();
        let original_deck = fs::read_to_string(&deck).unwrap();
        run(&temp.0, Some("multi_analysis_rc")).unwrap();
        // The C plots, split at their `Title:` headers.
        let plots: Vec<String> =
            original
                .split_inclusive('\n')
                .fold(Vec::<String>::new(), |mut plots, line| {
                    if line.starts_with("Title:") || plots.is_empty() {
                        plots.push(String::new());
                    }
                    plots.last_mut().unwrap().push_str(line);
                    plots
                });
        assert_eq!(plots.len(), 4);
        let swapped = [&plots[1], &plots[0], &plots[2], &plots[3]]
            .map(String::as_str)
            .concat();
        let dropped = plots[..3].concat();
        let doubled = format!("{original}{}", plots[3]);
        // v(out) of the operating point: 4/3 V -> 1.4 V.
        let op_value = original.replacen("\t1.333333333333333e+00", "\t1.400000000000000e+00", 1);
        assert_ne!(op_value, original);
        for (label, text) in [
            ("swapped", swapped),
            ("dropped", dropped),
            ("doubled", doubled),
            ("op value", op_value),
        ] {
            fs::write(&raw, text).unwrap();
            let error = run(&temp.0, Some("multi_analysis_rc")).unwrap_err();
            assert!(error.contains("multi_analysis_rc"), "{label}: {error}");
        }
        fs::write(&raw, &original).unwrap();
        // A deck whose schedule no longer matches the registered stages fails
        // too, rather than being compared against the wrong plots.
        for text in [
            original_deck.replace(".op\n", ""),
            original_deck.replace(".op\n", ".op\n.op\n"),
        ] {
            fs::write(&deck, text).unwrap();
            assert!(run(&temp.0, Some("multi_analysis_rc")).is_err());
        }
    }
}
