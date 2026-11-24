//! Independent physical BSIM4 sensitivity references. The C-compatible
//! .sens oracle is covered separately in tests/bsim4_sens.rs.
use std::path::Path;

use ngspice_rs::analysis::RunConfig;
use ngspice_rs::netlist::{Parser, source::parse_deck_text};

fn run(threshold: f64, analysis: &str, variable: &str) -> f64 {
    let deck = format!(
        "BSIM4 physical VTH sensitivity\n\
         vdd supply 0 1.1\n\
         rload supply drain 2k\n\
         vgate gate 0 0.8\n\
         vsource source 0 0\n\
         vbody body 0 0\n\
         m1 drain gate source body mn l=0.12u w=2u\n\
         .model mn nmos level=54 version=4.8.3 vth0={threshold:.17e} toxe=1.4n u0=0.04\n\
         .options reltol=1e-11 vntol=1e-12 abstol=1e-18\n\
         {analysis}\n\
         .end\n"
    );
    let netlist = Parser::new()
        .parse_deck(&parse_deck_text(Path::new("bsim4-derivative.cir"), &deck))
        .unwrap();
    let config = RunConfig::from_netlist(&netlist).unwrap();
    let request = config.plan_for(&netlist.analyses[0]).unwrap();
    let mut circuit = config.circuit(&netlist).unwrap();
    let plot = request.run(&mut circuit, &config.context()).unwrap();
    plot.value(variable, 0).unwrap().re
}

#[test]
fn vth_sensitivity_matches_independent_central_richardson_derivative() {
    let vth0 = 0.42;
    let op = |threshold| run(threshold, ".op", "v(drain)");
    let h = 1e-3;
    let central = |step| (op(vth0 + step) - op(vth0 - step)) / (2. * step);
    let coarse = central(h);
    let fine = central(h * 0.5);
    let richardson = (4. * fine - coarse) / 3.;
    let physical = run(vth0, ".sens v(drain) m1:vth0 physical", "v(m1:vth0)");
    assert!(coarse.is_finite() && fine.is_finite() && physical.is_finite());
    assert!(
        richardson.abs() > 1e-5,
        "insensitive bias is not a useful physical reference: {richardson:.17e}"
    );
    let refinement = (coarse - fine).abs();
    assert!(
        refinement <= fine.abs() * 0.02 + 1e-5,
        "central derivative is not converged: coarse={coarse:.17e}, fine={fine:.17e}"
    );
    assert!(
        (physical - richardson).abs() <= richardson.abs() * 0.02 + 1e-4,
        "BSIM4 physical derivative {physical:.17e} versus independent Richardson {richardson:.17e}"
    );
}
