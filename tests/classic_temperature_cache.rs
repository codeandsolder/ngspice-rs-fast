//! Temperature changes must invalidate only derived immutable model data.
//! Compare loads of one reused elaboration against fresh elaborations.
use ngspice_rs::analysis::bias::solve_dc;
use ngspice_rs::devices::{Circuit, ModelContext};
use ngspice_rs::maths::SparseMatrix;
use ngspice_rs::netlist::{Parser, source::parse_deck_text};
use std::path::Path;

fn bits(values: &[f64]) -> Vec<u64> {
    values.iter().map(|v| v.to_bits()).collect()
}

fn stamps(matrix: &SparseMatrix) -> Vec<(usize, usize, u64)> {
    matrix
        .triplets()
        .iter()
        .map(|t| (t.row, t.col, t.value.to_bits()))
        .collect()
}

fn check(body: &str, small_signal: bool) {
    let n = Parser::new()
        .parse_deck(&parse_deck_text(
            Path::new("classic_temperature_cache.cir"),
            &format!("classic temperature cache\n{body}\n.end\n"),
        ))
        .unwrap();
    let reused = Circuit::from_netlist(&n).unwrap();
    // Include changed nominal temperature, a cache hit, a return to a
    // previously cached temperature, and return to elaboration temperature.
    for (temperature, nominal) in [
        (27., 27.),
        (97., 23.),
        (97., 23.),
        (-20., 40.),
        (97., 23.),
        (27., 27.),
    ] {
        let context = ModelContext::new(temperature, nominal);
        let fresh = Circuit::from_netlist_with_context(&n, &context).unwrap();
        let run =
            |c: &Circuit| solve_dc(c, &context, &Default::default(), &[], None, None).unwrap();
        let a = run(&reused);
        let b = run(&fresh);
        let label = format!("{body} at {temperature}/{nominal} C");
        assert_eq!(
            bits(a.values.as_slice()),
            bits(b.values.as_slice()),
            "{label}: DC"
        );
        assert_eq!(
            bits(a.trial.values()),
            bits(b.trial.values()),
            "{label}: state"
        );
        if !small_signal {
            // MOS6 has no DEVacLoad in the pinned C implementation.
            for (circuit, result) in [(&reused, &a), (&fresh, &b)] {
                let error = circuit
                    .small_signal_system_at(&context, &result.values, Some(result.trial.values()))
                    .unwrap_err();
                assert!(
                    error
                        .to_string()
                        .contains("C's level 6 registers no DEVacLoad")
                );
            }
            continue;
        }
        let ac_a = reused
            .small_signal_system_at(&context, &a.values, Some(a.trial.values()))
            .unwrap();
        let ac_b = fresh
            .small_signal_system_at(&context, &b.values, Some(b.trial.values()))
            .unwrap();
        assert_eq!(stamps(&ac_a.a), stamps(&ac_b.a), "{label}: conductance");
        assert_eq!(stamps(&ac_a.e), stamps(&ac_b.e), "{label}: capacitance");
    }
}

#[test]
fn diode_bjt_and_fets_rederive_only_for_their_context() {
    for body in [
        "v1 a 0 0.6\nd1 a 0 dm dtemp=7\n.model dm d(is=2e-14 rs=5 cjo=2p vj=0.8 m=0.4 tt=1n)",
        "vc c 0 2\nvb b 0 0.65\nq1 c b 0 qm dtemp=5\n.model qm npn(is=1e-15 bf=100 cje=2p cjc=1p tf=1n tr=2n rc=3 rb=10)",
        "vd d 0 3\nvg g 0 -0.5\nj1 d g 0 jm dtemp=7\n.model jm njf(level=1 beta=1m cgs=2p cgd=1p rd=5 rs=3)",
        "vd d 0 3\nvg g 0 -0.5\nj1 d g 0 jm temp=45\n.model jm njf(level=2 vto=-2 beta=1m cgs=2p cgd=1p rd=5 rs=3)",
        "vd d 0 3\nvg g 0 -0.5\nz1 d g 0 zm area=1.7\n.model zm nmf(beta=1m cgs=2p cgd=1p rd=5 rs=3)",
    ] {
        check(body, true);
    }
}

#[test]
fn all_classic_mos_levels_rederive_for_changed_temperatures() {
    for level in [1, 2, 3, 6, 9] {
        check(
            &format!(
                "vd d 0 3\nvg g 0 2\nvb b 0 -0.2\nm1 d g 0 b mm w=10u l=2u ad=5p as=7p pd=9u ps=11u m=1.7 dtemp=5\n.model mm nmos(level={level} vto=0.7 phi=0.8 gamma=0.4 cj=0.2m cjsw=0.3n mj=0.33 mjsw=0.44)"
            ),
            level != 6,
        );
    }
}
