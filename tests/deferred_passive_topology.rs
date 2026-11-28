//! Occurrence-specific physical topology follows elaborated control references.
use std::path::Path;

use ngspice_rs::{
    analysis::{Plot, RunConfig},
    netlist::{Parser, source::parse_deck_text},
    primitives::{Complex, SpiceError, SpiceResult},
};

fn simulate(body: &str) -> SpiceResult<Plot> {
    let source = format!("Deferred network topology\n{body}\n.end\n");
    let netlist =
        Parser::new().parse_deck(&parse_deck_text(Path::new("deferred-network.cir"), &source))?;
    let config = RunConfig::from_netlist(&netlist)?;
    let mut circuit = config.circuit(&netlist)?;
    circuit.finalize()?;
    config
        .plans_for(&netlist.analyses)?
        .first()
        .ok_or_else(|| SpiceError::circuit("missing analysis"))?
        .run(&mut circuit, &config.context())
}

fn check(plot: &Plot, name: &str, expected: Complex) -> SpiceResult<()> {
    let actual = plot
        .value(name, 0)
        .ok_or_else(|| SpiceError::circuit(format!("missing {name}")))?;
    assert!(
        (actual - expected).magnitude() < 1e-8,
        "{name}: {actual:?} != {expected:?}"
    );
    Ok(())
}

#[test]
fn actual_current_controls_choose_series_shunt_before_folding() -> SpiceResult<()> {
    let sensed = Complex::new(0.602261102728891, 0.02998880634513171);
    let unsensed = Complex::new(0.5966083459066635, -0.04498320951769758);
    for (control, expected) in [
        ("", unsensed),
        (".save v(p) i(v1)", unsensed),
        (".func unused() {i(v1)}", unsensed),
        ("f1 q 0 v1 1\nr2 q 0 1k", sensed),
        ("h1 q 0 v1 1\nr2 q 0 1k", sensed),
        ("b1 q 0 v=i(v1)\nr2 q 0 1k", sensed),
        ("b1 q 0 v=0*i(v1)\nr2 q 0 1k", sensed),
        ("b1 q 0 v=0 ? i(v1) : 0\nr2 q 0 1k", sensed),
        (
            ".func read() {i(v1)}\n.func wrapper() {read()}\nb1 q 0 v=wrapper()\nr2 q 0 1k",
            sensed,
        ),
    ] {
        let plot = simulate(&format!(
            "v1 p 0 ac 1 pn_rseries=2 pn_ccontrol=1u pn_control=1 pn_report_dc=1\nr1 p 0 3\n{control}\n.ac lin 1 10k 10k"
        ))?;
        check(&plot, "v(p)", expected)?;
    }
    Ok(())
}

#[test]
fn coupling_and_initial_bias_use_resolved_series_topology() -> SpiceResult<()> {
    for coefficient in ["0", ".5"] {
        let plot = simulate(&format!(
            "v1 s 0 2\nr1 s p1 10\nr2 s p2 10\nr3 s p3 10\nl1 p1 0 3m pn_rseries_uncoupled=1m\nl2 p2 0 3m pn_rseries_uncoupled=1m\nl3 p3 0 3m pn_rseries_uncoupled=1m\nk1 l1 l2 {coefficient}\n.op"
        ))?;
        check(&plot, "v(p1)", Complex::real(0.))?;
        check(&plot, "v(p2)", Complex::real(0.))?;
        check(&plot, "v(p3)", Complex::real(0.0001999800019998000))?;
        let plot = simulate(&format!(
            "v1 s 0 2\nr1 s p1 10\nr2 s p2 10\nr3 s p3 10\nl1 p1 0 3m ic=.1 pn_rseries_uncoupled=1m pn_terminal_bias_op=0 pn_terminal_bias_zero_series=1e-5\nl2 p2 0 3m ic=.1 pn_rseries_uncoupled=1m pn_terminal_bias_op=0 pn_terminal_bias_zero_series=1e-5\nl3 p3 0 3m ic=.1 pn_rseries_uncoupled=1m pn_terminal_bias_op=0 pn_terminal_bias_zero_series=1e-5\nk1 l1 l2 {coefficient}\n.op"
        ))?;
        check(&plot, "v(p1)", Complex::real(0.9999000099990003))?;
        check(&plot, "v(p2)", Complex::real(0.9999000099990003))?;
        check(&plot, "v(p3)", Complex::real(1.))?;
    }
    Ok(())
}

#[test]
fn incompatible_topology_selectors_are_diagnostic() {
    for card in [
        "v1 p 0 1 pn_rseries=2 pn_ccontrol=1u pn_cparallel=0",
        "v1 p 0 1 pn_rseries=2 pn_ccontrol=1u pn_cseries_shunt=0",
        "l1 p 0 3m pn_rseries_uncoupled=1m pn_rseries=0",
        "l1 p 0 3m pn_terminal_bias_zero_series=1e-5",
    ] {
        assert!(simulate(&format!("{card}\nr1 p 0 10\n.op")).is_err());
    }
}
