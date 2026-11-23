//! Opt-in, numeric-only BSIM4 DC linearization diagnostic for issue #154.
//!
//! This is not a replacement for the committed C oracle. The output is an
//! independently inspectable Jacobian and RHS at the converged Rust iterate.
//! No source values or acceptance bounds are changed.
use ngspice_rs::analysis::bias::solve_dc_with;
use ngspice_rs::analysis::{AnalysisInput, RunConfig};
use ngspice_rs::devices::{AnalysisMode, IterationPhase, LoadRequest};
use ngspice_rs::maths::{SparseMatrix, Vector};
use ngspice_rs::netlist::{Parser, source::parse_deck_text};
use ngspice_rs::primitives::AnalysisKind;
use std::path::Path;

fn dump(name: &str, body: &str) {
    let netlist = Parser::new()
        .parse_deck(&parse_deck_text(
            Path::new("bsim4-op-diagnostics.cir"),
            body,
        ))
        .unwrap();
    let config = RunConfig::from_netlist(&netlist).unwrap();
    let circuit = config.circuit(&netlist).unwrap();
    let plan = config
        .plan(AnalysisInput::new(AnalysisKind::OperatingPoint))
        .unwrap();
    let model = config.context().model_context();
    let solved = solve_dc_with(&circuit, &model, plan.dc_settings(), &[], None, None).unwrap();
    let point = &solved.solution;
    let n = circuit.unknown_count();
    let history = circuit.state_history().unwrap();
    let mut trial = history
        .trial_in(IterationPhase::Float, Some(&point.trial))
        .unwrap();
    let mut jacobian = SparseMatrix::new(n, n);
    let mut rhs = Vector::zeros(n);
    circuit
        .load(
            &LoadRequest {
                mode: AnalysisMode::OperatingPoint,
                solution: &point.values,
                model_context: &model,
                integration: None,
                history: &history,
                forcing: None,
            },
            &mut jacobian,
            &mut rhs,
            &mut trial,
        )
        .unwrap();
    let mut observer = config.circuit(&netlist).unwrap();
    observer
        .set_observations(
            &[
                "@m1[id]",
                "@m1[gm]",
                "@m1[igcd]",
                "@m1[igd]",
                "@m1[ibd]",
                "@m1[isub]",
            ]
            .map(ToOwned::to_owned),
        )
        .unwrap();
    let plan = config.plan_for(&netlist.analyses[0]).unwrap();
    let plots = plan.run_plots(&mut observer, &config.context()).unwrap();
    let observed = &plots[0];
    for key in [
        "@m1[id]",
        "@m1[gm]",
        "@m1[igcd]",
        "@m1[igd]",
        "@m1[ibd]",
        "@m1[isub]",
        "i(vdd)",
        "i(vb)",
    ] {
        if let Some(value) = observed.value(key, 0) {
            println!("BSIM4_ASK {name} {key} {:+.17e}", value.re);
        }
    }
    println!("MNA_BEGIN {name} {n} iterations={}", point.iterations);
    for triplet in jacobian.triplets() {
        println!(
            "MNA_T {} {} {:.17e}",
            triplet.row, triplet.col, triplet.value
        );
    }
    for (index, (&value, (&b, &next))) in point
        .values
        .as_slice()
        .iter()
        .zip(rhs.as_slice().iter().zip(point.next.as_slice()))
        .enumerate()
    {
        println!("MNA_R {index} {value:.17e} {b:.17e} {next:.17e}");
    }
    println!("MNA_END {name}");
}

#[test]
#[ignore = "manual: export BSIM4 converged DC Jacobian for independent high-precision linear solve"]
fn bsim4_amplifier_dc_matrix() {
    dump(
        "amplifier",
        r####"BSIM4 nmos amplifier op
vdd dd 0 1.0
rl dd d 1k
vg g 0 dc 0.7 ac 1
vb b 0 dc -0.1
m1 d g 0 b nch l=45n w=0.5u ad=5e-14 as=5e-14 pd=1.2u ps=1.2u
cl d 0 2f
.model nch nmos level=54 version=4.8.3 capmod=2 igcmod=1 igbmod=1 rgatemod=1 rbodymod=1
+ toxe=1.25e-9 toxp=1e-9 toxm=1.25e-9 epsrox=3.9 xl=-20e-9 lint=3.75e-9 wint=5e-9
+ vth0=0.46893 k1=0.4 dvt0=1 dvt1=2 dsub=0.1 minv=0.05 dvtp0=1e-9 dvtp1=0.1 xj=1.4e-8
+ ngate=2e23 ndep=3.24e18 nsd=2e20 voff=-0.13 nfactor=1.6 eta0=0.0125 vfb=-0.55 u0=0.049
+ ua=6e-10 ub=1.2e-18 vsat=130000 keta=0.04 pclm=0.02 pdiblc1=0.001 pdiblc2=0.001
+ pdiblcb=-0.005 drout=0.5 pvag=1e-20 delta=0.01 pscbe1=8.14e8 pscbe2=1e-7 fprout=0.2
+ pdits=0.08 pditsd=0.23 pditsl=2.3e6 rsh=5 rdsw=155 rsw=80 rdw=80 alpha0=0.074 alpha1=0.005
+ beta0=30 agidl=0.0002 bgidl=2.1e9 cgidl=0.0002 egidl=0.8 aigbacc=0.012 bigbacc=0.0028
+ cigbacc=0.002 aigbinv=0.014 bigbinv=0.004 cigbinv=0.004 eigbinv=1.1 nigbinv=3 aigc=0.012
+ bigc=0.0028 cigc=0.002 aigsd=0.012 bigsd=0.0028 cigsd=0.002 nigc=1 poxedge=1 pigcd=1
+ xrcrg1=12 xrcrg2=5 cgso=1.1e-10 cgdo=1.1e-10 cgbo=2.56e-11 cgdl=2.653e-10 cgsl=2.653e-10
+ ckappas=0.03 ckappad=0.03 kt1=-0.11 kt2=0.022 ute=-1.5 ua1=4.31e-9 ub1=7.61e-18
+ uc1=-5.6e-11 at=33000 jss=1e-4 jsws=1e-11 jswgs=1e-10 cjs=5e-4 cjsws=5e-10 cjswgs=3e-10
+ rshg=0.4 rbpb=5 rbpd=15 rbps=15 rbdb=15 rbsb=15
.op
.save all @m1[id] @m1[gm] @m1[gds] @m1[gmbs] @m1[ibd] @m1[ibs] @m1[igs] @m1[igd] @m1[igb] @m1[igcs] @m1[igcd] @m1[isub] @m1[vth] @m1[vdsat] @m1[vgs] @m1[vds] @m1[vbs]
.end"####,
    );
}

#[test]
#[ignore = "manual: export BSIM4 converged DC Jacobian for independent high-precision linear solve"]
fn bsim4_width_dc_matrix() {
    dump(
        "width",
        r####"BSIM4 nmos width sweep
vdd dd 0 1.0
rl dd d 1k
vg g 0 dc 0.7 ac 1
vb b 0 dc -0.1
m1 d g 0 b nch l=45n w=0.3u ad=5e-14 as=5e-14 pd=1.2u ps=1.2u
cl d 0 2f
.model nch nmos level=54 version=4.8.3 capmod=2 igcmod=1 igbmod=1 rgatemod=1 rbodymod=1
+ toxe=1.25e-9 toxp=1e-9 toxm=1.25e-9 epsrox=3.9 xl=-20e-9 lint=3.75e-9 wint=5e-9
+ vth0=0.46893 k1=0.4 dvt0=1 dvt1=2 dsub=0.1 minv=0.05 dvtp0=1e-9 dvtp1=0.1 xj=1.4e-8
+ ngate=2e23 ndep=3.24e18 nsd=2e20 voff=-0.13 nfactor=1.6 eta0=0.0125 vfb=-0.55 u0=0.049
+ ua=6e-10 ub=1.2e-18 vsat=130000 keta=0.04 pclm=0.02 pdiblc1=0.001 pdiblc2=0.001
+ pdiblcb=-0.005 drout=0.5 pvag=1e-20 delta=0.01 pscbe1=8.14e8 pscbe2=1e-7 fprout=0.2
+ pdits=0.08 pditsd=0.23 pditsl=2.3e6 rsh=5 rdsw=155 rsw=80 rdw=80 alpha0=0.074 alpha1=0.005
+ beta0=30 agidl=0.0002 bgidl=2.1e9 cgidl=0.0002 egidl=0.8 aigbacc=0.012 bigbacc=0.0028
+ cigbacc=0.002 aigbinv=0.014 bigbinv=0.004 cigbinv=0.004 eigbinv=1.1 nigbinv=3 aigc=0.012
+ bigc=0.0028 cigc=0.002 aigsd=0.012 bigsd=0.0028 cigsd=0.002 nigc=1 poxedge=1 pigcd=1
+ xrcrg1=12 xrcrg2=5 cgso=1.1e-10 cgdo=1.1e-10 cgbo=2.56e-11 cgdl=2.653e-10 cgsl=2.653e-10
+ ckappas=0.03 ckappad=0.03 kt1=-0.11 kt2=0.022 ute=-1.5 ua1=4.31e-9 ub1=7.61e-18
+ uc1=-5.6e-11 at=33000 jss=1e-4 jsws=1e-11 jswgs=1e-10 cjs=5e-4 cjsws=5e-10 cjswgs=3e-10
+ rshg=0.4 rbpb=5 rbpd=15 rbps=15 rbdb=15 rbsb=15
.op
.save all @m1[id] @m1[gm] @m1[gds] @m1[gmbs] @m1[ibd] @m1[ibs] @m1[igs] @m1[igd] @m1[igb] @m1[igcs] @m1[igcd] @m1[isub] @m1[vth] @m1[vdsat] @m1[vgs] @m1[vds] @m1[vbs]
.end"####,
    );
}
