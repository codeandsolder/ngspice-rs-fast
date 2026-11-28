//! Independent LTspice 24.0.12 synthetic startup witnesses. The native
//! deck explicitly selects bias topology, aggregate equations and history.

use std::{f64::consts::TAU, path::Path};

use ngspice_rs::{
    analysis::{Plot, RunConfig},
    netlist::{Parser, source::parse_deck_text},
    primitives::{Complex, SpiceResult},
};

fn simulate(body: &str) -> SpiceResult<Vec<Plot>> {
    let source = format!(
        "Independent terminal startup\n.options reltol=1e-8 abstol=1e-12 vntol=1e-10\n{body}\n.end\n"
    );
    let netlist =
        Parser::new().parse_deck(&parse_deck_text(Path::new("network-bias.cir"), &source))?;
    let config = RunConfig::from_netlist(&netlist)?;
    let mut circuit = config.circuit(&netlist)?;
    circuit.finalize()?;
    config
        .plans_for(&netlist.analyses)?
        .iter()
        .map(|plan| plan.run(&mut circuit, &config.context()))
        .collect()
}

fn close(got: Complex, want: Complex, tol: f64) {
    let error = (got - want).magnitude();
    assert!(error <= tol, "{got:?} != {want:?}; error {error:e}");
}

fn at(plot: &Plot, name: &str, index: usize) -> Complex {
    plot.value(name, index).unwrap()
}

#[test]
fn capacitor_terminal_bias_is_aggregate_and_replaces_parasitic_network() -> SpiceResult<()> {
    let plots = simulate(
        "v1 s 0 2 ac 1\n\
         r1 s a 10\nc1 a 0 2u ic=.75 pn_terminal_bias_op=1000 pn_exact_ic=1 pn_report_dc=1\n\
         r2 s b 10\nc2 b 0 2u m=3 ic=.75 pn_terminal_bias_op=1000 pn_exact_ic=1\n\
         r3 s c 10\nc3 c 0 2u ic=.75 pn_rseries=5 pn_terminal_bias_op=.2 pn_exact_ic=1\n\
         r4 s d 10\nc4 d 0 2u m=3 ic=.75 pn_rseries=5 pn_rparallel=20 pn_cparallel=1u pn_lseries=3m pn_rseries_shunt=20 pn_terminal_bias_op=.2 pn_exact_ic=1\n\
         r5 s e 10\nc5 e 0 2u ic=0 pn_rseries=5 pn_rparallel=20 pn_terminal_bias_op=.2 pn_exact_ic=1\n\
         .op\n.dc v1 2 2.1 .1\n.ac lin 1 10000 10000\n.op",
    )?;
    for (node, want) in [
        ("v(a)", 0.7501249875012499),
        ("v(b)", 0.7501249875012499),
        ("v(c)", 1.166666666666667),
        ("v(d)", 1.166666666666667),
        ("v(e)", 0.6666666666666666),
    ] {
        close(at(&plots[0], node, 0), Complex::real(want), 1e-9);
        close(at(&plots[3], node, 0), Complex::real(want), 1e-9);
    }
    close(
        at(&plots[0], "i(c1)", 0),
        Complex::real((2. - 0.7501249875012499) / 10.),
        1e-9,
    );
    for (node, want) in [
        ("v(a)", 2.),
        ("v(b)", 2.),
        ("v(c)", 2.),
        ("v(d)", 0.8),
        ("v(e)", 4. / 3.),
    ] {
        close(at(&plots[1], node, 0), Complex::real(want), 1e-9);
    }
    let omega = TAU * 10000.;
    for (node, c, r) in [("v(a)", 2e-6, 0.), ("v(b)", 6e-6, 0.), ("v(c)", 2e-6, 5.)] {
        let y = Complex::real(1.) / Complex::new(r, -1. / (omega * c));
        close(
            at(&plots[2], node, 0),
            Complex::real(1.) / (Complex::real(1.) + Complex::real(10.) * y),
            1e-9,
        );
    }
    Ok(())
}

#[test]
fn inductor_terminal_bias_keeps_winding_current_and_replaces_shunts() -> SpiceResult<()> {
    let plots = simulate(
        "v1 s 0 2 ac 1\n\
         r1 s a 10\nl1 a 0 3m ic=.1 pn_terminal_bias_op={1e-5+1e-12/3m} pn_exact_ic=1 pn_control=0 pn_report_dc=1\n\
         r2 s b 10\nl2 b 0 3m m=3 ic=.1 pn_terminal_bias_op={1e-5+3e-12/3m} pn_exact_ic=1\n\
         r3 s c 10\nl3 c 0 3m ic=.1 pn_rseries=5 pn_rparallel=20 pn_terminal_bias_op=.05 pn_exact_ic=1\n\
         r4 s d 10\nl4 d 0 3m m=3 ic=.1 pn_rseries=5 pn_rparallel=20 pn_cparallel=1u pn_terminal_bias_op=.15 pn_exact_ic=1\n\
         r5 s e 10\nl5 e 0 3m ic=0 pn_rseries=5 pn_rparallel=20 pn_terminal_bias_op=.05 pn_exact_ic=1\n\
         .op\n.dc v1 2 2.1 .1\n.ac lin 1 10000 10000",
    )?;
    for (node, want) in [
        ("v(a)", 0.9999000066663334),
        ("v(b)", 0.9999000000009999),
        ("v(c)", 0.6666666666666666),
        ("v(d)", 0.4),
        ("v(e)", 1.333333333333333),
    ] {
        close(at(&plots[0], node, 0), Complex::real(want), 1e-9);
    }
    close(
        at(&plots[0], "i(l1)", 0),
        Complex::real((2. - 0.9999000066663334) / 10.),
        1e-9,
    );
    close(
        at(&plots[0], "i(l1#intrinsic)", 0),
        Complex::real(0.1),
        1e-9,
    );
    close(at(&plots[1], "v(a)", 0), Complex::real(0.), 1e-9);
    close(at(&plots[1], "v(c)", 0), Complex::real(4. / 7.), 1e-9);
    close(
        at(&plots[2], "v(a)", 0),
        Complex::new(0., TAU * 1e4 * 3e-3) / Complex::new(10., TAU * 1e4 * 3e-3),
        1e-9,
    );
    Ok(())
}

#[test]
fn exact_history_is_distinct_from_soft_bias_and_native_ic() -> SpiceResult<()> {
    let plots = simulate(
        "v1 s 0 2\n\
         r1 s a 10\nc1 a 0 2u ic=.75 pn_terminal_bias_tran=1000 pn_exact_ic=1\n\
         r2 s b 10\nc2 b 0 2u m=3 ic=.75 pn_terminal_bias_tran=1000 pn_exact_ic=1\n\
         r3 s c 10\nl1 c 0 3m ic=.1 pn_terminal_bias_tran={1e-5+1e-12/3m} pn_exact_ic=1\n\
         r4 s d 10\nl2 d 0 3m m=3 ic=.1 pn_terminal_bias_tran={1e-5+3e-12/3m} pn_exact_ic=1\n\
         r5 s e 10\nc3 e 0 2u ic=.75\n\
         .tran 1n 100n 0 1n\n.tran 1n 100n 0 1n uic",
    )?;
    let tran = &plots[0];
    for (node, want) in [
        ("v(a)", 0.7501249875012499),
        ("v(b)", 0.7501249875012499),
        ("v(c)", 0.9999000066663334),
        ("v(d)", 0.9999000000009999),
        ("v(e)", 2.),
    ] {
        close(at(tran, node, 0), Complex::real(want), 1e-9);
    }
    for (plot, start) in [(&plots[0], 1), (&plots[1], 0)] {
        assert!(plot.point_count() > start);
        for index in start..plot.point_count() {
            let time = at(plot, "time", index).re;
            for (node, copies) in [("v(a)", 1.), ("v(b)", 3.)] {
                let v = 1.25_f64.mul_add(-(-time / (10. * 2e-6 * copies)).exp_m1(), 0.75);
                close(at(plot, node, index), Complex::real(v), 2e-7);
            }
            for (node, copies) in [("v(c)", 1.), ("v(d)", 3.)] {
                let v = (-time / (3e-3 / (10. * copies))).exp();
                close(at(plot, node, index), Complex::real(v), 2e-7);
            }
        }
    }
    Ok(())
}

#[test]
fn hierarchy_preserves_fractional_multiplicity_and_zero_target() -> SpiceResult<()> {
    let plots = simulate(
        "v1 s 0 2\n\
         r1 s a 10\nx1 a 0 cell copies=.5 init=0\n\
         r2 s b 10\nx2 b 0 cell copies=3 init=.75\n\
         .subckt cell p n params: copies=1 init=1\n\
         c1 p n 2u m={copies} ic={init} pn_rseries=5 pn_rparallel=20 pn_terminal_bias_op=.2 pn_exact_ic=1\n\
         .ends\n.op",
    )?;
    close(at(&plots[0], "v(a)", 0), Complex::real(2. / 3.), 1e-9);
    close(at(&plots[0], "v(b)", 0), Complex::real(7. / 6.), 1e-9);
    Ok(())
}

#[test]
fn parallel_capacitance_has_independent_exact_startup_voltage() -> SpiceResult<()> {
    let plots = simulate(
        "v1 s 0 2\n\
         r1 s a 10\nc1 a 0 2u ic=.75 pn_rseries=5 pn_cparallel=1u pn_parallel_ic=.75 pn_terminal_bias_tran=.2 pn_exact_ic=1\n\
         r2 s b 10\nc2 b 0 2u m=3 ic=.75 pn_rseries=5 pn_cparallel=1u pn_parallel_ic=.75 pn_terminal_bias_tran=.2 pn_exact_ic=1\n\
         r3 s c 10\nl1 c 0 3m ic=.1 pn_rseries=5 pn_rparallel=20 pn_cparallel=1u pn_terminal_bias_tran=.05 pn_exact_ic=1\n\
         .tran 1n 10n 0 1n\n.tran 1n 10n 0 1n uic",
    )?;
    for node in ["v(a)", "v(b)"] {
        close(at(&plots[0], node, 0), Complex::real(7. / 6.), 1e-9);
        for (plot, index) in [(&plots[0], 1), (&plots[1], 0)] {
            let time = at(plot, "time", index).re;
            let copies = if node == "v(a)" { 1. } else { 3. };
            let slope = 0.125_f64 / (copies * 1e-6);
            close(
                at(plot, node, index),
                Complex::real(slope.mul_add(time, 0.75)),
                1e-7,
            );
        }
    }
    close(at(&plots[0], "v(c)", 0), Complex::real(2. / 3.), 1e-9);
    close(at(&plots[0], "v(c)", 1), Complex::real(2. / 3.), 1e-7);
    let time = at(&plots[1], "time", 0).re;
    close(at(&plots[1], "v(c)", 0), Complex::real(1e5 * time), 1e-7);
    Ok(())
}

#[test]
fn derivative_startup_keeps_the_reported_bias_distinct_from_exact_storage() -> SpiceResult<()> {
    let plots = simulate(
        "v1 s 0 2\n\
         r1 s a 10\nc1 a 0 2u ic=.75 pn_terminal_bias_tran=1000 pn_exact_ic=1 pn_report_dc=1\n\
         v2 x 0 PWL(0 0 1u 1)\nb1 y 0 v=1u*ddt(v(x))\nr2 y 0 1k\n\
         .tran 1n 10n 0 1n",
    )?;
    let plot = &plots[0];
    close(at(plot, "v(a)", 0), Complex::real(0.7501249875012499), 1e-9);
    close(
        at(plot, "i(c1)", 0),
        Complex::real((2. - 0.7501249875012499) / 10.),
        1e-9,
    );
    for index in 1..plot.point_count() {
        let t = at(plot, "time", index).re;
        let v = 1.25_f64.mul_add(-(-t / 20e-6).exp_m1(), 0.75);
        close(at(plot, "v(a)", index), Complex::real(v), 1e-7);
        close(at(plot, "v(y)", index), Complex::real(1.), 1e-8);
    }
    Ok(())
}

#[test]
fn coupled_inductors_use_exact_initial_currents_for_both_fluxes() -> SpiceResult<()> {
    let plots = simulate(
        "v1 s 0 2\nr1 s a 10\nr2 s b 10\n\
         l1 a 0 3m ic=.1 pn_terminal_bias_tran=0 pn_exact_ic=1\n\
         l2 b 0 3m ic=.05 pn_terminal_bias_tran=0 pn_exact_ic=1\n\
         k1 l1 l2 .5\n.tran 1n 100n 0 1n",
    )?;
    let plot = &plots[0];
    for index in 0..plot.point_count() {
        let t = at(plot, "time", index).re;
        // Exact solution of L di/dt = 2 - 10i using common and differential modes.
        let common = 0.125_f64 * (-t / 450e-6).exp();
        let differential = 0.025 * (-t / 150e-6).exp();
        close(
            at(plot, "v(a)", index),
            Complex::real(10. * (common - differential)),
            2e-7,
        );
        close(
            at(plot, "v(b)", index),
            Complex::real(10. * (common + differential)),
            2e-7,
        );
    }
    Ok(())
}
