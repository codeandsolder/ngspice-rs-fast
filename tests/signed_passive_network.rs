//! Signed constitutive coefficients are independent of thermal-generator laws.

use std::{f64::consts::TAU, path::Path};

use ngspice_rs::{
    analysis::{Plot, RunConfig},
    netlist::{Parser, source::parse_deck_text},
    primitives::{Complex, SpiceError, SpiceResult},
};

fn simulate(body: &str) -> SpiceResult<Vec<Plot>> {
    let source =
        format!("Signed storage\n.options reltol=1e-8 abstol=1e-12 vntol=1e-10\n{body}\n.end\n");
    let netlist =
        Parser::new().parse_deck(&parse_deck_text(Path::new("signed-network.cir"), &source))?;
    let config = RunConfig::from_netlist(&netlist)?;
    let mut circuit = config.circuit(&netlist)?;
    circuit.finalize()?;
    config
        .plans_for(&netlist.analyses)?
        .iter()
        .map(|plan| plan.run(&mut circuit, &config.context()))
        .collect()
}

fn first(body: &str) -> SpiceResult<Plot> {
    simulate(body)?
        .into_iter()
        .next()
        .ok_or_else(|| SpiceError::circuit("missing analysis"))
}

fn at(plot: &Plot, name: &str, index: usize) -> SpiceResult<Complex> {
    plot.value(name, index)
        .ok_or_else(|| SpiceError::circuit(format!("missing {name} at {index}")))
}

fn close(got: Complex, want: Complex, relative: f64, absolute: f64) {
    let error = (got - want).magnitude();
    assert!(
        error <= relative.mul_add(want.magnitude(), absolute),
        "{got:?} != {want:?}; error {error:e}"
    );
}

#[test]
fn signed_storage_and_loss_match_independent_ac_witnesses() -> SpiceResult<()> {
    // Independent LTspice 24.0.12, each 10 kHz, source AC 1 / DC 2,
    // a 10-ohm series drive and one storage network.
    for (card, re, im) in [
        (
            "c1 p 0 2u pn_rseries=-2",
            0.3716902792754108,
            -0.6249912365375416,
        ),
        (
            "c1 p 0 2u pn_rparallel=-20",
            0.2733529969952181,
            -0.6870110137575443,
        ),
        (
            "c1 p 0 2u pn_cparallel=-1u",
            0.7169568003248977,
            -0.4504772433683886,
        ),
        (
            "c1 p 0 2u pn_lseries=-3m pn_cparallel=2p",
            0.9974154863547918,
            -0.05077237373243457,
        ),
        (
            "c1 p 0 2u pn_lseries=3m pn_gseries_ac=-.05 pn_cparallel=2p",
            1.752517974113890,
            -0.4509694496139617,
        ),
        (
            "c1 p 0 2u pn_lseries=-3m pn_gseries_ac=-.05 pn_cparallel=2p",
            1.497007406959305,
            -0.5111864514526565,
        ),
        (
            "c1 p 0 2u pn_rseries=-2 pn_cparallel=-1u",
            0.8779491368181177,
            -0.6915044441942596,
        ),
        (
            "l1 p 0 3m pn_rseries=-2",
            0.9977524665457234,
            0.0529562594148951,
        ),
        (
            "l1 p 0 3m pn_rseries=1m pn_rparallel=-20",
            1.977733752088942,
            0.2098439503836435,
        ),
        (
            "l1 p 0 3m pn_rseries=1m pn_cparallel=-1u",
            0.6829362707329332,
            0.4653322776678149,
        ),
        (
            "c1 p 0 -2u pn_rseries=0",
            0.3877266367391514,
            0.4872316614323186,
        ),
        (
            "l1 p 0 -3m pn_rseries=0",
            0.9971934217275060,
            -0.0529027540955512,
        ),
        (
            "c1 p 0 2u m=-2 pn_rseries=0",
            0.1366764984976091,
            0.3435055068787721,
        ),
        (
            "l1 p 0 3m m=-2 pn_rseries=0",
            0.9888674202791085,
            -0.1049220919999672,
        ),
    ] {
        let plot = first(&format!(
            "v1 s 0 2 ac 1\nr1 s p 10\n{card}\n.ac lin 1 10k 10k"
        ))?;
        close(at(&plot, "v(p)", 0)?, Complex::new(re, im), 2e-8, 1e-10);
    }
    Ok(())
}

#[test]
fn signed_bias_and_damping_are_physical_equations() -> SpiceResult<()> {
    for (card, expected) in [
        (
            "c1 p 0 2u ic=.75 pn_rseries=-2 pn_terminal_bias_op=1000 pn_exact_ic=1",
            0.7501249875012499,
        ),
        (
            "c1 p 0 2u ic=.1 pn_rparallel=-20 pn_terminal_bias_op=1000",
            0.1001899810018998,
        ),
        (
            "l1 p 0 3m ic=.75 pn_rseries=-2 pn_terminal_bias_op={1e-12/3m}",
            -5.499999981666667,
        ),
        (
            "l1 p 0 3m ic=.1 pn_rparallel=-20 pn_terminal_bias_op=-.05",
            2.,
        ),
        ("c1 p 0 2u pn_gterminal_dc=-.05", 4.),
        ("c1 p 0 2u m=-2 pn_gterminal_dc=.025", 4.),
    ] {
        let plot = first(&format!("v1 s 0 2\nr1 s p 10\n{card}\n.op"))?;
        close(at(&plot, "v(p)", 0)?, Complex::real(expected), 2e-8, 1e-10);
    }
    Ok(())
}

#[test]
fn signed_noise_selection_never_changes_electrical_coefficients() -> SpiceResult<()> {
    for (device, want) in [
        ("c1 p 0 2u", 2.128643511035140e-10),
        ("l1 p 0 3m", 5.725653305296391e-10),
    ] {
        let body =
            format!("v1 s 0 ac 1\nr1 s p 10 noisy=0\n{device} pn_rseries=0 pn_rparallel=-20");
        let plot = first(&format!(
            "{body} pn_rparallel_noise_abs=1\n.noise v(p) v1 lin 2 10k 20k"
        ))?;
        close(
            at(&plot, "onoise_spectrum", 0)?,
            Complex::real(want),
            3e-4,
            1e-22,
        );
        close(
            at(&plot, "inoise_spectrum", 0)?,
            Complex::real(2.878896273035901e-10),
            3e-4,
            1e-22,
        );
        let invalid = first(&format!("{body}\n.noise v(p) v1 lin 2 10k 20k"));
        assert!(invalid.is_err_and(|e| e.to_string().contains("undefined thermal noise")));
        let quiet = first(&format!(
            "{body} pn_noiseless=1\n.noise v(p) v1 lin 2 10k 20k"
        ))?;
        close(at(&quiet, "onoise_spectrum", 0)?, Complex::real(0.), 0., 0.);
        let with_policy = first(&format!(
            "{body} pn_rparallel_noise_abs=1\n.ac lin 1 10k 10k"
        ))?;
        let without_policy = first(&format!("{body} pn_noiseless=1\n.ac lin 1 10k 10k"))?;
        assert_eq!(
            at(&with_policy, "v(p)", 0)?,
            at(&without_policy, "v(p)", 0)?
        );
    }
    for card in [
        "c1 p 0 2u pn_rseries=-2",
        "c1 p 0 2u pn_lseries=3m pn_gseries_ac=-.05 pn_gseries_noise=1",
        "c1 p 0 2u pn_lseries=3m pn_rseries_shunt=-20",
        "c1 p 0 2u pn_rstorage_shunt=-20",
    ] {
        let body = format!("v1 s 0 ac 1\nr1 s p 10 noisy=0\n{card}");
        assert!(
            first(&format!("{body}\n.noise v(p) v1 lin 2 10k 20k"))
                .is_err_and(|e| e.to_string().contains("undefined thermal noise"))
        );
        let plot = first(&format!(
            "{body} pn_noiseless=1\n.noise v(p) v1 lin 2 10k 20k"
        ))?;
        close(at(&plot, "onoise_spectrum", 0)?, Complex::real(0.), 0., 0.);
    }
    Ok(())
}

#[test]
fn signed_voltage_output_capacitances_match_explicit_impedance() -> SpiceResult<()> {
    let w = TAU * 1e4;
    for (key, expected) in [
        (
            "pn_cparallel",
            Complex::real(1.) / Complex::new(1.2, -2. * w * 1e-6),
        ),
        ("pn_cseries_shunt", {
            let z = Complex::real(1.) / Complex::new(0.5, -w * 1e-6);
            Complex::real(10.) / (Complex::real(10.) + z)
        }),
    ] {
        let plot = first(&format!(
            "v1 p 0 ac 1 pn_rseries=2 {key}=-1u\nr1 p 0 10\n.ac lin 1 10k 10k"
        ))?;
        close(at(&plot, "v(p)", 0)?, expected, 1e-10, 1e-12);
    }
    Ok(())
}

#[test]
fn signed_transient_storage_uses_exact_history_and_stable_signed_dynamics() -> SpiceResult<()> {
    // Negative storage and negative resistance cancel in the time constant.
    // The initial physical charge/flux retains its sign, not its magnitude.
    let cap = first(
        "v1 s 0 2\nr1 s p -10\nc1 p 0 -2u ic=.5 pn_rseries=0 pn_exact_ic=1\n.tran 100n 40u 0 100n",
    )?;
    let ind = first(
        "v1 s 0 2\nr1 s p -10\nl1 p 0 -3m ic=-.05 pn_rseries=0 pn_exact_ic=1\n.tran 1u 600u 0 1u",
    )?;
    for (plot, name, reference) in [
        (&cap, "v(p)", 1.796997918135468),
        (&ind, "i(l1)", -0.1796997450230323),
    ] {
        close(
            at(plot, name, plot.point_count() - 1)?,
            Complex::real(reference),
            2e-5,
            1e-7,
        );
    }
    for (plot, name, final_value, initial, tau) in [
        (&cap, "v(p)", 2.0_f64, 0.5, 20e-6),
        (&ind, "i(l1)", -0.2, -0.05, 300e-6),
    ] {
        for index in 1..plot.point_count() {
            let time = at(plot, "time", index)?.re;
            let expected = (initial - final_value).mul_add((-time / tau).exp(), final_value);
            close(at(plot, name, index)?, Complex::real(expected), 2e-4, 1e-6);
        }
    }
    Ok(())
}

#[test]
fn invalid_scaling_and_indefinite_coupling_remain_diagnostic() -> SpiceResult<()> {
    for card in [
        "c1 p 0 1e300 m=-1e300 pn_rseries=0",
        "c1 p 0 1e-300 m=-1e-300 pn_rseries=0",
        "l1 p 0 1e300 m=-1e-300 pn_rseries=0",
        "c1 p 0 1u m=-1e300 pn_rparallel=1e-300",
        "c1 p 0 1u pn_gterminal_dc=1e300 m=-1e300",
        "c1 p 0 1u pn_noiseless=2",
        "c1 p 0 1u pn_rparallel_noise_abs=-1",
        "c1 p 0 1u pn_rparallel=-10",
        "l1 p 0 -3m pn_rseries=1\nl2 p 0 1m\nk1 l1 l2 .1",
        "l1 p 0 -3m pn_rseries=1\nl2 p 0 -1m pn_rseries=1\nk1 l1 l2 .1",
    ] {
        assert!(
            first(&format!("v1 s 0 2\nr1 s p 10\n{card}\n.op")).is_err(),
            "{card}"
        );
    }
    Ok(())
}
