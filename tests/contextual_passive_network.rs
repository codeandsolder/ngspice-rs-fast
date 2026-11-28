//! Contextual storage uses one effective constitutive law for all analyses.
use std::path::Path;

use ngspice_rs::{
    analysis::{Plot, RunConfig},
    devices::ModelContext,
    netlist::{Parser, ast::Netlist, source::parse_deck_text, writer::write_netlist},
    primitives::{Complex, SpiceError, SpiceResult},
};

fn parse(body: &str) -> SpiceResult<Netlist> {
    Parser::new().parse_deck(&parse_deck_text(
        Path::new("contextual-storage.cir"),
        &format!(
            "Contextual storage\n.options reltol=1e-8 abstol=1e-12 vntol=1e-10\n{body}\n.end\n"
        ),
    ))
}

fn at(plot: &Plot, name: &str, index: usize) -> SpiceResult<Complex> {
    plot.value(name, index)
        .ok_or_else(|| SpiceError::circuit(format!("missing {name}")))
}

fn close(got: Complex, expected: Complex, relative: f64, absolute: f64) {
    let error = (got - expected).magnitude();
    assert!(
        error <= relative.mul_add(expected.magnitude(), absolute),
        "{got:?} != {expected:?}; error {error:e}"
    );
}

fn simulate(body: &str) -> SpiceResult<Plot> {
    let netlist = parse(body)?;
    let config = RunConfig::from_netlist(&netlist)?;
    let mut circuit = config.circuit(&netlist)?;
    circuit.finalize()?;
    config
        .plans_for(&netlist.analyses)?
        .first()
        .ok_or_else(|| SpiceError::circuit("missing analysis"))?
        .run(&mut circuit, &config.context())
}

#[test]
fn temperature_polynomials_match_independent_frequency_witnesses() -> SpiceResult<()> {
    for (card, ambient, nominal, expected) in [
        (
            "c1 p 0 2u pn_rseries=5 pn_storage_tc=.01",
            27.,
            27.,
            Complex::new(0.4797550849386705, -0.2759984995014461),
        ),
        (
            "c1 p 0 2u pn_rseries=5 pn_storage_tc=.01",
            127.,
            27.,
            Complex::new(0.3771577201505801, -0.1652140460077680),
        ),
        (
            "c1 p 0 2u pn_rseries=5 pn_storage_tc=.01 temp=27",
            127.,
            50.,
            Complex::new(0.5479296714032754, -0.3114685169755245),
        ),
        (
            "c1 p 0 2u pn_rseries=5 pn_storage_tc=.01 temp=127",
            27.,
            50.,
            Complex::new(0.3882872619512899, -0.1833467156548587),
        ),
        (
            "c1 p 0 2u pn_rseries=0 pn_storage_tc=(.01 .0001 .000001)",
            127.,
            27.,
            Complex::new(0.03807176084762064, -0.1913695427009801),
        ),
        (
            "l1 p 0 3m pn_rseries=5 pn_storage_tc=.01 pn_storage_tnom=27",
            27.,
            50.,
            Complex::new(0.9958048501119162, 0.05271780827644847),
        ),
        (
            "l1 p 0 3m pn_rseries=5 pn_storage_tc=.01 pn_storage_tnom=27",
            127.,
            50.,
            Complex::new(0.9989462392581887, 0.02648389604092536),
        ),
        (
            "l1 p 0 3m pn_rseries=1m pn_storage_tc=.01 pn_storage_tnom=50",
            127.,
            27.,
            Complex::new(0.9991023550498043, 0.02994577479745143),
        ),
        (
            "l1 p 0 3m pn_rseries=1m pn_storage_tc=.01 .0001 pn_storage_tnom=27",
            127.,
            50.,
            Complex::new(0.9996873468177956, 0.01767835309071601),
        ),
    ] {
        let plot = simulate(&format!(
            "v1 s 0 2 ac 1\nr1 s p 10\n{card}\n.options temp={ambient} tnom={nominal}\n.ac lin 1 10k 10k"
        ))?;
        close(at(&plot, "v(p)", 0)?, expected, 2e-8, 1e-10);
    }
    Ok(())
}

#[test]
fn contextual_leakage_and_storage_follow_signed_effective_value() -> SpiceResult<()> {
    for (base, m, tc, local, expected) in [
        (1e9, 1., 0.01, "", 1.96078431372549),
        (1e9, 1., 0.01, "temp=27", 1.98019801980198),
        (1e9, 1., -0.01, "", 2.),
        (-1e9, 1., -0.02, "", 1.98019801980198),
        (1e9, -2., -0.02, "", 1.96078431372549),
        (1e9, -2., 0.01, "", 2.),
    ] {
        let plot = simulate(&format!(
            "v1 s 0 2\nr1 s p 10\nc1 p 0 {base} m={m} pn_storage_tc={tc} {local} pn_gterminal_positive_storage_dc=1e-12 pn_gterminal_positive_storage_tran=1e-12\n.temp 127\n.op"
        ))?;
        close(at(&plot, "v(p)", 0)?, Complex::real(expected), 2e-9, 1e-12);
    }
    for tc in [0.01, -0.01, -0.02] {
        let plot = simulate(&format!(
            "v1 s 0 2\nr1 s p 10\nl1 p 0 1e-10 pn_rseries=5 pn_storage_tc={tc} pn_storage_tnom=27 pn_gterminal_dc=.01\n.temp 127\n.op"
        ))?;
        close(
            at(&plot, "v(p)", 0)?,
            Complex::real(0.6451612903225805),
            2e-9,
            1e-12,
        );
    }
    Ok(())
}

#[test]
fn one_elaboration_rebinds_temperature_and_nominal_context_without_topology_changes()
-> SpiceResult<()> {
    let netlist = parse(
        ".param tc=-.01\nv1 s 0 2 ac 1\nr1 s p 10\nc1 p 0 2u pn_storage_tc={tc} pn_gterminal_positive_storage_dc=1e-12\n.ac lin 1 10k 10k",
    )?;
    let config = RunConfig::from_netlist(&netlist)?;
    let mut circuit = config.circuit(&netlist)?;
    circuit.finalize()?;
    let initial_nodes = circuit.nodes().nodes().to_vec();
    let initial_devices = circuit.device_count();
    let plans = config.plans_for(&netlist.analyses)?;
    let plan = plans
        .first()
        .ok_or_else(|| SpiceError::circuit("missing AC plan"))?;
    for (temperature, nominal, effective) in [
        (27., 27., 2e-6),
        (127., 27., 0.),
        (227., 27., -2e-6),
        (127., 127., 2e-6),
        (27., 27., 2e-6),
    ] {
        let model = ModelContext::new(temperature, nominal);
        let primary = circuit
            .device("c1")
            .ok_or_else(|| SpiceError::circuit("missing primary"))?;
        let storage = primary
            .storage_element(&model)
            .ok_or_else(|| SpiceError::circuit("missing storage"))??;
        assert_eq!(storage.value, effective);
        assert_eq!(
            primary.observation_parameter("capacitance", &model)?,
            Some(effective)
        );
        let damping = circuit
            .device("rc1#storage_derived_damping")
            .ok_or_else(|| SpiceError::circuit("missing derived conductance"))?;
        assert_eq!(
            damping.observation_parameter("g", &model)?,
            Some(effective.max(0.) * 1e-12)
        );
        let mut context = config.context();
        context.temperature = temperature;
        context.nominal_temperature = nominal;
        let plot = plan.run(&mut circuit, &context)?;
        let expected =
            Complex::real(1.) / Complex::new(1., std::f64::consts::TAU * 1e5 * effective);
        close(at(&plot, "v(p)", 0)?, expected, 2e-9, 1e-12);
        assert_eq!(circuit.nodes().nodes(), initial_nodes);
        assert_eq!(circuit.device_count(), initial_devices);
    }
    Ok(())
}

#[test]
fn symbolic_vector_roundtrip_and_occurrence_parameters_are_preserved() -> SpiceResult<()> {
    let source = ".subckt cell p n params: k=.01\nc1 p n 2u pn_storage_tc=({k} 0 0 0 0 0 0 1e-16)\n.ends\nv1 s 0 ac 1\nr1 s a 10\nr2 s b 10\nx1 a 0 cell k=.01\nx2 b 0 cell k=-.01\n.temp 127\n.ac lin 1 10k 10k";
    let netlist = parse(source)?;
    let rendered = write_netlist(&netlist)?;
    assert!(rendered.contains("pn_storage_tc=("));
    let reparsed =
        Parser::new().parse_deck(&parse_deck_text(Path::new("roundtrip.cir"), &rendered))?;
    let config = RunConfig::from_netlist(&reparsed)?;
    let circuit = config.circuit(&reparsed)?;
    let model = ModelContext::new(127., 27.);
    for (name, effective) in [("c.x1.c1", 6e-6), ("c.x2.c1", 2e-6)] {
        let device = circuit
            .device(name)
            .ok_or_else(|| SpiceError::circuit(format!("missing {name}")))?;
        let storage = device
            .storage_element(&model)
            .ok_or_else(|| SpiceError::circuit("missing storage"))??;
        close(
            Complex::real(storage.value),
            Complex::real(effective),
            1e-12,
            1e-18,
        );
    }
    Ok(())
}

#[test]
fn only_main_storage_is_temperature_scaled_and_local_noise_temperature_is_used() -> SpiceResult<()>
{
    for (thermal, explicit, frequency) in [
        (
            "c1 p 0 2u pn_rseries=5 pn_lseries=3m pn_cparallel=1u pn_storage_tc=.01",
            "c1 p 0 4u pn_rseries=5 pn_lseries=3m pn_cparallel=1u",
            1e6,
        ),
        (
            "l1 p 0 3m pn_rseries=5 pn_rparallel=20 pn_cparallel=1u pn_storage_tc=.01 pn_storage_tnom=27",
            "l1 p 0 6m pn_rseries=5 pn_rparallel=20 pn_cparallel=1u",
            1e4,
        ),
    ] {
        let run = |card| {
            simulate(&format!(
                "v1 s 0 2 ac 1\nr1 s p 10\n{card}\n.temp 127\n.ac lin 1 {frequency} {frequency}"
            ))
        };
        let thermal = run(thermal)?;
        let explicit = run(explicit)?;
        close(
            at(&thermal, "v(p)", 0)?,
            at(&explicit, "v(p)", 0)?,
            2e-12,
            1e-14,
        );
    }
    for (card, cold, hot) in [
        (
            "c1 p 0 2u pn_rseries=5 pn_rparallel=20 pn_lseries=3m pn_gseries_ac=.05 pn_gseries_noise=1 pn_cparallel=1u pn_storage_tc=.01",
            1.898036047878347e-10,
            2.191526331574124e-10,
        ),
        (
            "l1 p 0 3m pn_rseries=5 pn_rparallel=20 pn_cparallel=1u pn_storage_tc=.01 pn_storage_tnom=27",
            1.793036704573182e-10,
            2.070291107454709e-10,
        ),
    ] {
        // Noise scales thermally at fixed storage: explicit nominal reference
        // follows the local temperature, preserving the unadjusted value.
        for (local, expected) in [(27., cold), (127., hot)] {
            let plot = simulate(&format!(
                "v1 s 0 ac 1\nr1 s p 10 noisy=0\n{card} temp={local} pn_storage_tnom={local}\n.temp 127\n.noise v(p) v1 lin 2 10k 20k"
            ))?;
            close(
                at(&plot, "onoise_spectrum", 0)?,
                Complex::real(expected),
                3e-4,
                1e-22,
            );
        }
    }
    Ok(())
}

#[test]
fn mutual_base_selection_is_independent_of_contextual_self_storage() -> SpiceResult<()> {
    let common = "v1 s 0 ac 1\nr1 s p 10\nl2 q 0 12m pn_rseries=0 temp=27 pn_storage_tc=.01 pn_storage_tnom=27\nk1 l1 l2 .5\nr2 q 0 100\n.temp 127\n.ac lin 1 100 100";
    for (selector, expected) in [
        (
            "pn_coupling_nominal=1",
            Complex::new(0.07379578375932881, 0.1590673616189068),
        ),
        (
            "pn_coupling_nominal=0",
            Complex::new(0.1037397687879336, 0.2244266146480546),
        ),
    ] {
        let plot = simulate(&format!(
            "{common}\nl1 p 0 3m pn_rseries=0 pn_storage_tc=.01 pn_storage_tnom=27 {selector}"
        ))?;
        close(at(&plot, "v(q)", 0)?, expected, 2e-8, 1e-10);
    }
    let invalid = simulate(&format!(
        "{common}\nl1 p 0 3m pn_rseries=0 pn_storage_tc=-.01 pn_storage_tnom=27 pn_coupling_nominal=1"
    ));
    assert!(invalid.is_err(), "nonzero mutual with zero self storage");
    Ok(())
}

#[test]
fn exact_initial_history_uses_the_adjusted_storage_coefficient() -> SpiceResult<()> {
    for (card, drive, name, final_value, initial, tau, end, step) in [
        (
            "c1 p 0 2u ic=.5 pn_rseries=0 pn_storage_tc=.01 pn_exact_ic=1",
            "10",
            "v(p)",
            2.0_f64,
            0.5,
            40e-6,
            "80u",
            "100n",
        ),
        (
            "l1 p 0 3m ic=.05 pn_rseries=0 pn_storage_tc=.01 pn_exact_ic=1",
            "10",
            "i(l1)",
            0.2,
            0.05,
            600e-6,
            "1200u",
            "1u",
        ),
    ] {
        let plot = simulate(&format!(
            "v1 s 0 2\nr1 s p {drive}\n{card}\n.temp 127\n.tran {step} {end} 0 {step}"
        ))?;
        for index in 1..plot.point_count() {
            let time = at(&plot, "time", index)?.re;
            let expected = (initial - final_value).mul_add((-time / tau).exp(), final_value);
            close(at(&plot, name, index)?, Complex::real(expected), 2e-4, 1e-6);
        }
    }
    Ok(())
}

#[test]
fn malformed_or_nonfinite_contextual_recipes_fail_even_when_removed() {
    for card in [
        "c1 p 0 1u pn_storage_tc=()",
        "c1 p 0 1u pn_storage_tc=(1,)",
        "c1 p 0 1u pn_storage_tc=.01 pn_storage_tnom=-300",
        "c1 p 0 1u m=0 pn_storage_tc=.01 pn_storage_tnom=-300",
        "c1 p 0 1u pn_storage_tc=1e300 1e300 1e300",
        "c1 p 0 1u pn_coupling_nominal=1",
        "l1 p 0 1m m=0 pn_coupling_nominal=2",
    ] {
        assert!(
            simulate(&format!("v1 p 0 1\n{card}\n.temp 127\n.op")).is_err(),
            "{card}"
        );
    }
}

#[test]
fn candidate_vector_rebinding_is_transactional_and_matches_fresh_elaboration() -> SpiceResult<()> {
    use ngspice_rs::{
        analysis::optimization::parameters::CircuitParameterPlan, netlist::eval::EvalBudget,
    };
    let body = "v1 s 0 2 ac 1\nr1 s p 10\nc1 p 0 2u pn_storage_tc={tc} {quadratic} pn_gterminal_positive_storage_dc=1e-12\n.temp 127\n.ac lin 1 10k 10k";
    let source = parse(&format!(".param tc=.01 quadratic=0\n{body}"))?;
    let plan = CircuitParameterPlan::new(source, &["tc", "quadratic"])?;
    let mut session = plan.session()?;
    for (linear, quadratic) in [
        (0.01, 0.),
        (-0.01, 0.),
        (-0.02, 0.),
        (-0.02, 0.0002),
        (0.01, 0.),
    ] {
        session.update(&[linear, quadratic], &mut EvalBudget::default())?;
        let model = session.model_context();
        let storage = session
            .circuit_mut()
            .device("c1")
            .ok_or_else(|| SpiceError::circuit("missing candidate primary"))?
            .storage_element(&model)
            .ok_or_else(|| SpiceError::circuit("missing candidate storage"))??;
        let independent = parse(&format!(".param tc={linear} quadratic={quadratic}\n{body}"))?;
        let config = RunConfig::from_netlist(&independent)?;
        let independent = config.circuit(&independent)?;
        let expected = independent
            .device("c1")
            .ok_or_else(|| SpiceError::circuit("missing independent primary"))?
            .storage_element(&model)
            .ok_or_else(|| SpiceError::circuit("missing independent storage"))??;
        close(
            Complex::real(storage.value),
            Complex::real(expected.value),
            1e-12,
            1e-18,
        );
    }
    assert!(
        session
            .update(&[f64::NAN, 0.], &mut EvalBudget::default())
            .is_err()
    );
    let model = session.model_context();
    let storage = session
        .circuit_mut()
        .device("c1")
        .ok_or_else(|| SpiceError::circuit("missing stable primary"))?
        .storage_element(&model)
        .ok_or_else(|| SpiceError::circuit("missing stable storage"))??;
    close(
        Complex::real(storage.value),
        Complex::real(4e-6),
        1e-12,
        1e-18,
    );
    Ok(())
}
