//! BSIM4 (`bsim4/`, levels 14/54) end to end against values read from C.
//!
//! Every expected value below is C's: printed by a `-ffp-contract=off` build
//! of the reference ngspice (the FMA-free oracle the port is judged against,
//! see `docs/port/VERIFICATION.md`) from the same deck with `.options
//! sparse`. On macOS arm64, where that oracle was built, the port reproduces
//! them bit for bit; other platforms' `exp`/`log`/`pow` round differently in
//! the last place, so there the comparison is relative (and a transient is
//! compared on C's time grid by interpolation, since its step sequence can
//! differ).
use std::path::Path;

use ngspice_rs::analysis::{Plot, RunConfig};
use ngspice_rs::netlist::Parser;
use ngspice_rs::netlist::source::parse_deck_text;

/// Exact on the oracle's platform.
const EXACT: bool = cfg!(all(target_os = "macos", target_arch = "aarch64"));

fn plots(text: &str) -> Vec<Plot> {
    let netlist = Parser::new()
        .parse_deck(&parse_deck_text(Path::new("bsim4.cir"), text))
        .unwrap();
    let config = RunConfig::from_netlist(&netlist).unwrap();
    let request = config.plan_for(&netlist.analyses[0]).unwrap();
    let mut circuit = config.circuit(&netlist).unwrap();
    request.run_plots(&mut circuit, &config.context()).unwrap()
}

fn same(actual: f64, expected: f64, label: &str) {
    if EXACT {
        assert_eq!(
            actual.to_bits(),
            expected.to_bits(),
            "{label}: Rust {actual:e}, C {expected:e}"
        );
    } else {
        assert!(
            (actual - expected).abs() <= 1e-9 * expected.abs() + 1e-30,
            "{label}: Rust {actual:e}, C {expected:e}"
        );
    }
}

/// Expected samples: `(vector, [(point, re, im)])`.
type Expected<'a> = &'a [(&'a str, &'a [(usize, f64, f64)])];

/// Compare `expected` with `plot`.
fn check(plot: &Plot, expected: Expected<'_>) {
    for (name, points) in expected {
        for &(index, re, im) in *points {
            let value = plot
                .value(name, index)
                .unwrap_or_else(|| panic!("{name}[{index}] missing"));
            same(value.re, re, &format!("{name}[{index}].re"));
            same(value.im, im, &format!("{name}[{index}].im"));
        }
    }
}

#[test]
fn an_amplifier_operating_point_with_gate_and_body_resistance_matches_c() {
    let plots = plots(
        r"BSIM4 nmos amplifier op
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
.end
",
    );
    check(
        &plots[plots.len() - 1],
        &[
            ("v(d)", &[(0, 0.999978464811956, 0.0)]),
            ("v(m1#gate)", &[(0, 0.699999622060779, 0.0)]),
            ("v(m1#body)", &[(0, -0.09999999999773429, 0.0)]),
            ("v(m1#dbody)", &[(0, -0.09999999999061684, 0.0)]),
            ("i(vdd)", &[(0, -2.1535188044060835e-08, 0.0)]),
            ("i(vg)", &[(0, -1.446195998511257e-07, 0.0)]),
            ("i(vb)", &[(0, 1.2042380981291956e-12, 0.0)]),
        ],
    );
}

#[test]
fn a_pmos_sweep_with_rdsmod_matches_c() {
    let plots = plots(
        r"BSIM4 pmos dc sweep
vd d 0 -1.0
vg g 0 -0.8
vb b 0 0
m1 d g 0 b pch l=45n w=1u ad=1e-13 as=1e-13 pd=2.2u ps=2.2u
.model pch pmos level=54 version=4.8.3 capmod=2 igcmod=1 igbmod=1 rgatemod=0 rbodymod=0 rdsmod=1
+ toxe=1.25e-9 toxp=1e-9 toxm=1.25e-9 epsrox=3.9 xl=-20e-9 lint=3.75e-9 wint=5e-9
+ vth0=-0.49158 k1=0.4 dvt0=1 dvt1=2 dsub=0.1 minv=0.05 dvtp0=1e-9 dvtp1=0.1 xj=1.4e-8
+ ngate=2e23 ndep=3.24e18 nsd=2e20 voff=-0.13 nfactor=1.6 eta0=0.0125 vfb=0.55 u0=0.021
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
.dc vd 0 -1.1 -0.1 vg -0.2 -1.0 -0.4
.end
",
    );
    check(
        &plots[plots.len() - 1],
        &[
            (
                "i(vd)",
                &[
                    (0, -1.729138553784732e-11, 0.0),
                    (1, -4.14432854353719e-12, 0.0),
                    (2, 2.0011076129478056e-13, 0.0),
                    (3, 8.379980737105441e-13, 0.0),
                    (4, 6.552189346642479e-13, 0.0),
                    (5, 3.442312407342385e-12, 0.0),
                    (6, 1.1296196616994436e-11, 0.0),
                    (7, 2.978466952241465e-11, 0.0),
                    (8, 6.941328935250457e-11, 0.0),
                    (9, 1.494524382722462e-10, 0.0),
                    (10, 3.0454210507568913e-10, 0.0),
                    (11, 5.959515970987983e-10, 0.0),
                    (12, -5.579145161533376e-10, 0.0),
                    (13, -2.786304516104454e-10, 0.0),
                    (14, -1.358770630732753e-10, 0.0),
                    (15, -6.603967112744913e-11, 0.0),
                    (16, -3.402953006187426e-11, 0.0),
                    (17, -2.081051303504111e-11, 0.0),
                    (18, -1.629505905209605e-11, 0.0),
                    (19, -1.524164995603261e-11, 0.0),
                    (20, -1.4408030324375432e-11, 0.0),
                    (21, -9.125278657706737e-12, 0.0),
                    (22, 4.882289017515973e-12, 0.0),
                    (23, 3.862489147965498e-11, 0.0),
                    (24, -7.125027399479506e-09, 0.0),
                    (25, -3.6021352207461776e-09, 0.0),
                    (26, -1.2663828614228878e-09, 0.0),
                    (27, 1.0787903157588596e-09, 0.0),
                    (28, 4.865965706757347e-09, 0.0),
                    (29, 1.2792133201669498e-08, 0.0),
                    (30, 3.057096952842342e-08, 0.0),
                    (31, 7.043913697463156e-08, 0.0),
                    (32, 1.5802476304935253e-07, 0.0),
                    (33, 3.4549804974609577e-07, 0.0),
                    (34, 7.364305492415374e-07, 0.0),
                    (35, 1.5314678667530807e-06, 0.0),
                ],
            ),
            (
                "i(vg)",
                &[
                    (0, 6.760152965962064e-11, 0.0),
                    (1, 5.455459150196398e-11, 0.0),
                    (2, 5.03101538443401e-11, 0.0),
                    (3, 4.977226743242644e-11, 0.0),
                    (4, 5.0055048408305744e-11, 0.0),
                    (5, 4.7367954189968744e-11, 0.0),
                    (6, 3.961407104813541e-11, 0.0),
                    (7, 2.1225600954667763e-11, 0.0),
                    (8, -1.830300918467102e-11, 0.0),
                    (9, -9.82421392230373e-11, 0.0),
                    (10, -2.5323175960104474e-10, 0.0),
                    (11, -5.445411268722048e-10, 0.0),
                    (12, 1.115829526691582e-09, 0.0),
                    (13, 8.364390568565162e-10, 0.0),
                    (14, 6.937661999008064e-10, 0.0),
                    (15, 6.24034090362817e-10, 0.0),
                    (16, 5.921495762213918e-10, 0.0),
                    (17, 5.790984450142509e-10, 0.0),
                    (18, 5.748516732650395e-10, 0.0),
                    (19, 5.743124984472957e-10, 0.0),
                    (20, 5.745946494913678e-10, 0.0),
                    (21, 5.719073774857346e-10, 0.0),
                    (22, 5.64153650222389e-10, 0.0),
                    (23, 5.457655991429856e-10, 0.0),
                    (24, 1.425005900118617e-08, 0.0),
                    (25, 1.1116787431168584e-08, 0.0),
                    (26, 9.358287323147123e-09, 0.0),
                    (27, 8.385754489172261e-09, 0.0),
                    (28, 7.858456798547987e-09, 0.0),
                    (29, 7.579721122279244e-09, 0.0),
                    (30, 7.437257175971869e-09, 0.0),
                    (31, 7.367871555882864e-09, 0.0),
                    (32, 7.336677316163972e-09, 0.0),
                    (33, 7.324943196960631e-09, 0.0),
                    (34, 7.323041801668104e-09, 0.0),
                    (35, 7.326342548750556e-09, 0.0),
                ],
            ),
            (
                "i(vb)",
                &[
                    (0, -3.301875858392605e-11, 0.0),
                    (1, -3.3118877147982846e-11, 0.0),
                    (2, -3.3218879629817834e-11, 0.0),
                    (3, -3.331887968173078e-11, 0.0),
                    (4, -3.341887968282849e-11, 0.0),
                    (5, -3.3518879682625543e-11, 0.0),
                    (6, -3.361887968198996e-11, 0.0),
                    (7, -3.3718879680492435e-11, 0.0),
                    (8, -3.381887967728263e-11, 0.0),
                    (9, -3.391887967079998e-11, 0.0),
                    (10, -3.401887965824145e-11, 0.0),
                    (11, -3.411887963501587e-11, 0.0),
                    (12, -4.943843892162947e-16, 0.0),
                    (13, -1.0061293703860283e-13, 0.0),
                    (14, -2.0061541264769974e-13, 0.0),
                    (15, -3.0061546126834093e-13, 0.0),
                    (16, -4.006154608214964e-13, 0.0),
                    (17, -5.006154604310514e-13, 0.0),
                    (18, -6.006154610153435e-13, 0.0),
                    (19, -7.00615463254009e-13, 0.0),
                    (20, -8.006154694230046e-13, 0.0),
                    (21, -9.006154858205215e-13, 0.0),
                    (22, -1.0006163646132592e-12, 0.0),
                    (23, -1.1006638502334734e-12, 0.0),
                    (24, -4.202227077767961e-15, 0.0),
                    (25, -1.0432102773687872e-13, 0.0),
                    (26, -2.0432477878866375e-13, 0.0),
                    (27, -3.043289540113114e-13, 0.0),
                    (28, -4.0434109151165946e-13, 0.0),
                    (29, -5.043749452476945e-13, 0.0),
                    (30, -6.044653936535763e-13, 0.0),
                    (31, -7.046973299822197e-13, 0.0),
                    (32, -8.052678236766423e-13, 0.0),
                    (33, -9.06613075009503e-13, 0.0),
                    (34, -1.0590902154810267e-12, 0.0),
                    (35, -3.361328189121773e-12, 0.0),
                ],
            ),
        ],
    );
}

#[test]
fn an_amplifier_ac_sweep_matches_c() {
    let plots = plots(
        r"BSIM4 nmos amplifier ac
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
.ac dec 2 1e3 1e11
.end
",
    );
    check(
        &plots[plots.len() - 1],
        &[
            (
                "v(d)",
                &[
                    (0, -0.0008740762081596259, 9.666430974013617e-10),
                    (1, -0.0008740762081594857, 3.0567938722682862e-09),
                    (2, -0.0008740762081580856, 9.666430974013372e-09),
                    (3, -0.000874076208144084, 3.056793872267503e-08),
                    (4, -0.0008740762080040663, 9.66643097398859e-08),
                    (5, -0.0008740762066038884, 3.0567938721891256e-07),
                    (6, -0.0008740761926021094, 9.666430971510043e-07),
                    (7, -0.0008740760525843205, 3.056793864351272e-06),
                    (8, -0.0008740746524064674, 9.66643072365535e-06),
                    (9, -0.0008740606506315242, 3.0567930805661e-05),
                    (10, -0.0008739206332410655, 9.666405938250336e-05),
                    (11, -0.0008725204952326874, 0.00030567147040784954),
                    (12, -0.0008585227038413624, 0.0009663928039376061),
                    (13, -0.0007189027331518071, 0.0030488972977153264),
                    (14, 0.000642405300245558, 0.009422392461270496),
                    (15, 0.011483313202430536, 0.024279425926699034),
                    (16, 0.042465806662443796, 0.026920027669426153),
                ],
            ),
            (
                "v(m1#gate)",
                &[
                    (0, 0.999997909788349, -7.410286765916135e-12),
                    (1, 0.999997909788349, -2.3433384295297984e-11),
                    (2, 0.999997909788349, -7.410286765916132e-11),
                    (3, 0.999997909788349, -2.3433384295297864e-10),
                    (4, 0.9999979097883489, -7.41028676591575e-10),
                    (5, 0.9999979097883487, -2.343338429528574e-09),
                    (6, 0.9999979097883466, -7.410286765877412e-09),
                    (7, 0.9999979097883244, -2.3433384294073377e-08),
                    (8, 0.9999979097881029, -7.410286762043559e-08),
                    (9, 0.9999979097858878, -2.3433384172836346e-07),
                    (10, 0.9999979097637353, -7.41028637865929e-07),
                    (11, 0.9999979095422157, -2.343337204944658e-06),
                    (12, 0.999997907327576, -7.410248050156898e-06),
                    (13, 0.9999978852365452, -2.3432162841557948e-05),
                    (14, 0.9999976697232708, -7.406511941378693e-05),
                    (15, 0.9999959433892962, -0.00023336112370305742),
                    (16, 0.9999906563699871, -0.000730240208358531),
                ],
            ),
            (
                "i(vg)",
                &[
                    (0, -7.99825886854677e-07, -2.8355689155291346e-12),
                    (1, -7.99825886854677e-07, -8.96685623544566e-12),
                    (2, -7.99825886854677e-07, -2.8355689155291333e-11),
                    (3, -7.99825886854677e-07, -8.966856235445614e-11),
                    (4, -7.998258869101882e-07, -2.835568915528987e-10),
                    (5, -7.998258869656993e-07, -8.966856235440976e-10),
                    (6, -7.998258877983666e-07, -2.835568915514317e-09),
                    (7, -7.998258962915727e-07, -8.966856234977062e-09),
                    (8, -7.998259810571007e-07, -2.835568914047281e-08),
                    (9, -7.998268286568688e-07, -8.96685618858534e-08),
                    (10, -7.998353053761953e-07, -2.835568767344117e-07),
                    (11, -7.999200705155474e-07, -8.966851549533134e-07),
                    (12, -8.007675091903366e-07, -2.8355541008253446e-06),
                    (13, -8.092207097187831e-07, -8.966388842432892e-06),
                    (14, -8.916875239295408e-07, -2.8341244673642968e-05),
                    (15, -1.5522745039908692e-06, -8.92963483557618e-05),
                    (16, -3.575368627384279e-06, -0.0002794286511576013),
                ],
            ),
        ],
    );
}

#[test]
fn an_amplifier_noise_spectrum_matches_c() {
    let plots = plots(
        r"BSIM4 nmos amplifier noise
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
+ fnoimod=1 tnoimod=0
.noise v(d) vg dec 2 1e3 1e10
.end
",
    );
    check(
        &plots[0],
        &[
            (
                "onoise_spectrum",
                &[
                    (0, 4.200435751019782e-09, 0.0),
                    (1, 4.112908610305432e-09, 0.0),
                    (2, 4.084839805134725e-09, 0.0),
                    (3, 4.075923442293437e-09, 0.0),
                    (4, 4.0730997786805354e-09, 0.0),
                    (5, 4.0722064503268774e-09, 0.0),
                    (6, 4.071923913850702e-09, 0.0),
                    (7, 4.071834559298044e-09, 0.0),
                    (8, 4.071806256559703e-09, 0.0),
                    (9, 4.071796847010667e-09, 0.0),
                    (10, 4.071789277492136e-09, 0.0),
                    (11, 4.071740945178458e-09, 0.0),
                    (12, 4.071266365882909e-09, 0.0),
                    (13, 4.066532409921632e-09, 0.0),
                    (14, 4.020084603914306e-09, 0.0),
                ],
            ),
            (
                "inoise_spectrum",
                &[
                    (0, 4.805571541480647e-06, 0.0),
                    (1, 4.705434802922621e-06, 0.0),
                    (2, 4.6733222649919655e-06, 0.0),
                    (3, 4.663121363817155e-06, 0.0),
                    (4, 4.659890884198699e-06, 0.0),
                    (5, 4.658868609556773e-06, 0.0),
                    (6, 4.658542880235422e-06, 0.0),
                    (7, 4.6584157611854325e-06, 0.0),
                    (8, 4.658134487944534e-06, 0.0),
                    (9, 4.655636989427205e-06, 0.0),
                    (10, 4.6309781778595175e-06, 0.0),
                    (11, 4.404193705020654e-06, 0.0),
                    (12, 3.149520027188701e-06, 0.0),
                    (13, 1.298172130853297e-06, 0.0),
                    (14, 4.25664049922432e-07, 0.0),
                ],
            ),
        ],
    );
}

#[test]
fn correlated_induced_gate_noise_and_fnoimod0_match_c() {
    let plots = plots(
        r"BSIM4 nmos amplifier noise tnoimod=2 fnoimod=0
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
+ fnoimod=0 tnoimod=2 kf=1e-26 af=1.2
.noise v(d) vg dec 2 1e3 1e10
.end
",
    );
    check(
        &plots[0],
        &[
            (
                "onoise_spectrum",
                &[
                    (0, 3.663857290598721e-08, 0.0),
                    (1, 2.08766245058084e-08, 0.0),
                    (2, 1.2212971455994993e-08, 0.0),
                    (3, 7.648628269294776e-09, 0.0),
                    (4, 5.46203593036848e-09, 0.0),
                    (5, 4.557227311191266e-09, 0.0),
                    (6, 4.231022686308204e-09, 0.0),
                    (7, 4.122499406774636e-09, 0.0),
                    (8, 4.087581704540642e-09, 0.0),
                    (9, 4.076477053052454e-09, 0.0),
                    (10, 4.072954558859861e-09, 0.0),
                    (11, 4.071794078405971e-09, 0.0),
                    (12, 4.0709677827538394e-09, 0.0),
                    (13, 4.066123076595089e-09, 0.0),
                    (14, 4.019645216165719e-09, 0.0),
                ],
            ),
            (
                "inoise_spectrum",
                &[
                    (0, 4.19169090337187e-05, 0.0),
                    (1, 2.3884215484642896e-05, 0.0),
                    (2, 1.397243322865884e-05, 0.0),
                    (3, 8.750527921195575e-06, 0.0),
                    (4, 6.248924117772306e-06, 0.0),
                    (5, 5.213763969412587e-06, 0.0),
                    (6, 4.840562109810169e-06, 0.0),
                    (7, 4.716379295947439e-06, 0.0),
                    (8, 4.676181554448351e-06, 0.0),
                    (9, 4.66098827319331e-06, 0.0),
                    (10, 4.6323034902018906e-06, 0.0),
                    (11, 4.404251176512405e-06, 0.0),
                    (12, 3.1492890441332405e-06, 0.0),
                    (13, 1.2980414580679404e-06, 0.0),
                    (14, 4.2561752563576206e-07, 0.0),
                ],
            ),
        ],
    );
}

#[test]
fn an_instance_width_sweep_matches_c() {
    let deck = r"BSIM4 nmos width sweep
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
.dc @m1[w] 0.3u 1.2u 0.3u
.end
";
    let physical = plots(deck);
    let scaled_deck = deck
        .replace(
            "l=45n w=0.5u ad=5e-14 as=5e-14 pd=1.2u ps=1.2u",
            "l=90n w=1u ad=2e-13 as=2e-13 pd=2.4u ps=2.4u",
        )
        .replace(
            ".dc @m1[w] 0.3u 1.2u 0.3u",
            ".options scale=0.5\n.dc @m1[w] 0.6u 2.4u 0.6u",
        );
    let scaled = plots(&scaled_deck);
    let physical = &physical[physical.len() - 1];
    let scaled = &scaled[scaled.len() - 1];
    assert_eq!(physical.point_count(), scaled.point_count());
    for point in 0..physical.point_count() {
        for name in ["v(d)", "i(vdd)"] {
            let expected = physical.value(name, point).unwrap().re;
            let actual = scaled.value(name, point).unwrap().re;
            same(actual, expected, name);
        }
    }
    check(
        physical,
        &[
            (
                "v(d)",
                &[
                    (0, 0.9999885056493419, 0.0),
                    (1, 0.9999728684408109, 0.0),
                    (2, 0.9999540288732832, 0.0),
                    (3, 0.9999324977061741, 0.0),
                ],
            ),
            (
                "i(vdd)",
                &[
                    (0, -1.1494350658130051e-08, 0.0),
                    (1, -2.7131559188989934e-08, 0.0),
                    (2, -4.597112671675223e-08, 0.0),
                    (3, -6.75022938259693e-08, 0.0),
                ],
            ),
            (
                "i(vb)",
                &[
                    (0, 1.2023958217977082e-12, 0.0),
                    (1, 1.2052199516165985e-12, 0.0),
                    (2, 1.2083424538733567e-12, 0.0),
                    (3, 1.2117425118862712e-12, 0.0),
                ],
            ),
        ],
    );
}

#[test]
fn a_temperature_sweep_checks_the_model_once_as_c_does() {
    let plots = plots(
        r"BSIM4 nmos temperature sweep
vdd dd 0 1.0
rl dd d 1k
vg g 0 dc 0.7 ac 1
vb b 0 dc -0.1
m1 d g 0 b nch l=45n w=0.5u ad=5e-14 as=5e-14 pd=1.2u ps=1.2u 
cl d 0 2f
.model nch nmos level=54 version=4.8.3 binunit=1 paramchk=1 mobmod=0 capmod=2 igcmod=1 igbmod=1 geomod=1 diomod=1 rdsmod=0 rbodymod=1 rgatemod=1 permod=1 acnqsmod=0 trnqsmod=0 tnom=27 toxe=1.25e-9 toxp=1e-9 toxm=1.25e-9 dtox=0.25e-9 epsrox=3.9 wint=5e-9 lint=3.75e-9 ll=0 wl=0 lln=1 wln=1 lw=0 ww=0 lwn=1 wwn=1 lwl=0 wwl=0 xpart=0 toxref=1.25e-9 xl=-20e-9 vth0=0.46893 k1=0.4 k2=0 k3=0 k3b=0 w0=2.5e-6 dvt0=1 dvt1=2 dvt2=0 dvt0w=0 dvt1w=0 dvt2w=0 dsub=0.1 minv=0.05 voffl=0 dvtp0=1.0e-9 dvtp1=0.1 lpe0=0 lpeb=0 xj=1.4e-8 ngate=2e23 ndep=3.24e18 nsd=2e20 phin=0 cdsc=0 cdscb=0 cdscd=0 cit=0 voff=-0.13 nfactor=1.6 eta0=0.0125 etab=0 vfb=-0.55 u0=0.049 ua=6e-10 ub=1.2e-18 uc=0 vsat=130000 a0=1.0 ags=0 a1=0 a2=1.0 b0=0 b1=0 keta=0.04 dwg=0 dwb=0 pclm=0.02 pdiblc1=0.001 pdiblc2=0.001 pdiblcb=-0.005 drout=0.5 pvag=1e-20 delta=0.01 pscbe1=8.14e8 pscbe2=1e-7 fprout=0.2 pdits=0.08 pditsd=0.23 pditsl=2.3e6 rsh=5 rdsw=155 rsw=80 rdw=80 rdswmin=0 rdwmin=0 rswmin=0 prwg=0 prwb=0 wr=1 alpha0=0.074 alpha1=0.005 beta0=30 agidl=0.0002 bgidl=2.1e9 cgidl=0.0002 egidl=0.8 aigbacc=0.012 bigbacc=0.0028 cigbacc=0.002 nigbacc=1 aigbinv=0.014 bigbinv=0.004 cigbinv=0.004 eigbinv=1.1 nigbinv=3 aigc=0.012 bigc=0.0028 cigc=0.002 aigsd=0.012 bigsd=0.0028 cigsd=0.002 nigc=1 poxedge=1 pigcd=1 ntox=1 xrcrg1=12 xrcrg2=5 cgso=1.1e-10 cgdo=1.1e-10 cgbo=2.56e-11 cgdl=2.653e-10 cgsl=2.653e-10 ckappas=0.03 ckappad=0.03 acde=1 moin=15 noff=0.9 voffcv=0.02 kt1=-0.11 kt1l=0 kt2=0.022 ute=-1.5 ua1=4.31e-9 ub1=7.61e-18 uc1=-5.6e-11 prt=0 at=33000 fnoimod=1 tnoimod=0 jss=0.0001 jsws=1e-11 jswgs=1e-10 njs=1 ijthsfwd=0.01 ijthsrev=0.001 bvs=10 xjbvs=1 jsd=0.0001 jswd=1e-11 jswgd=1e-10 njd=1 ijthdfwd=0.01 ijthdrev=0.001 bvd=10 xjbvd=1 pbs=1 cjs=0.0005 mjs=0.5 pbsws=1 cjsws=5e-10 mjsws=0.33 pbswgs=1 cjswgs=3e-10 mjswgs=0.33 pbd=1 cjd=0.0005 mjd=0.5 pbswd=1 cjswd=5e-10 mjswd=0.33 pbswgd=1 cjswgd=5e-10 mjswgd=0.33 tpb=0.005 tcj=0.001 tpbsw=0.005 tcjsw=0.001 tpbswg=0.005 tcjswg=0.001 xtis=3 xtid=3 dmcg=0 dmci=0 dmdg=0 dmcgt=0 dwj=0 xgw=0 xgl=0 rshg=0.4 gbmin=1e-10 rbpb=5 rbpd=15 rbps=15 rbdb=15 rbsb=15 ngcon=1
+ tempmod=2
.dc temp -40 125 15
.end
",
    );
    check(
        &plots[plots.len() - 1],
        &[
            (
                "v(d)",
                &[
                    (0, 0.43809527111459046, 0.0),
                    (1, 0.44147176481339595, 0.0),
                    (2, 0.44466582830105505, 0.0),
                    (3, 0.4476927635510141, 0.0),
                    (4, 0.4505730725615641, 0.0),
                    (5, 0.4252891402508104, 0.0),
                    (6, 0.42881124696559336, 0.0),
                    (7, 0.43300092401969675, 0.0),
                    (8, 0.4366135642037559, 0.0),
                    (9, 0.4400921727589469, 0.0),
                    (10, 0.44345338102646586, 0.0),
                    (11, 0.4467114625502676, 0.0),
                ],
            ),
            (
                "i(vdd)",
                &[
                    (0, -0.0005619047288854095, 0.0),
                    (1, -0.0005585282351866041, 0.0),
                    (2, -0.000555334171698945, 0.0),
                    (3, -0.000552307236448986, 0.0),
                    (4, -0.0005494269274384359, 0.0),
                    (5, -0.0005747108597491896, 0.0),
                    (6, -0.0005711887530344067, 0.0),
                    (7, -0.0005669990759803033, 0.0),
                    (8, -0.0005633864357962441, 0.0),
                    (9, -0.0005599078272410531, 0.0),
                    (10, -0.0005565466189735341, 0.0),
                    (11, -0.0005532885374497324, 0.0),
                ],
            ),
            (
                "i(vg)",
                &[
                    (0, -8.409016333255082e-08, 0.0),
                    (1, -8.42714015236723e-08, 0.0),
                    (2, -8.445941601653573e-08, 0.0),
                    (3, -8.465018236369914e-08, 0.0),
                    (4, -8.484372804318241e-08, 0.0),
                    (5, -8.395420220130845e-08, 0.0),
                    (6, -8.408058527198392e-08, 0.0),
                    (7, -8.435077986090889e-08, 0.0),
                    (8, -8.455293837306144e-08, 0.0),
                    (9, -8.47555574057246e-08, 0.0),
                    (10, -8.495722292334307e-08, 0.0),
                    (11, -8.515673544096103e-08, 0.0),
                ],
            ),
        ],
    );
}

#[test]
fn pole_zero_with_cs_rerun_node_aliasing_matches_c() {
    let plots = plots(
        r"BSIM4 nmos amplifier pole-zero
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
+ rdsmod=1
.pz g 0 d 0 vol pz
.end
",
    );
    check(
        &plots[plots.len() - 1],
        &[
            ("v(pole(1))", &[(0, -6053252430954828.0, 0.0)]),
            ("v(pole(2))", &[(0, -355301868561303.2, 0.0)]),
            ("v(pole(3))", &[(0, -225166097544243.75, 0.0)]),
            ("v(pole(4))", &[(0, -390446593718.6479, 0.0)]),
            ("v(zero(1))", &[(0, -6470157081859989.0, 0.0)]),
            ("v(zero(2))", &[(0, -898737738135737.0, 0.0)]),
            ("v(zero(3))", &[(0, -224801883793448.03, 0.0)]),
            ("v(zero(4))", &[(0, -14569335743777.86, 0.0)]),
            ("v(zero(5))", &[(0, 5741117721.969103, 0.0)]),
        ],
    );
}

#[test]
fn a_cmos_inverter_with_nqs_transient_matches_c() {
    let plots = plots(
        r"BSIM4 cmos inverter transient
vdd dd 0 1.0
vin in 0 pulse(0 1 0.1n 20p 20p 0.4n 1n)
mp out in dd dd pch l=45n w=1u ad=1e-13 as=1e-13 pd=2.2u ps=2.2u
mn out in 0 0 nch l=45n w=0.5u ad=5e-14 as=5e-14 pd=1.2u ps=1.2u
cl out 0 2f
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
+ trnqsmod=1
.model pch pmos level=54 version=4.8.3 capmod=2 igcmod=1 igbmod=1 rgatemod=0 rbodymod=0 rdsmod=1
+ toxe=1.25e-9 toxp=1e-9 toxm=1.25e-9 epsrox=3.9 xl=-20e-9 lint=3.75e-9 wint=5e-9
+ vth0=-0.49158 k1=0.4 dvt0=1 dvt1=2 dsub=0.1 minv=0.05 dvtp0=1e-9 dvtp1=0.1 xj=1.4e-8
+ ngate=2e23 ndep=3.24e18 nsd=2e20 voff=-0.13 nfactor=1.6 eta0=0.0125 vfb=0.55 u0=0.021
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
.tran 10p 1.5n
.end
",
    );
    let plot = &plots[plots.len() - 1];
    let expected: Expected<'_> = &[
        (
            "v(out)",
            &[
                (0, 0.5345552189351839, 0.0),
                (14, 0.5345552189352244, 0.0),
                (28, 0.6881989829820164, 0.0),
                (42, 0.6848541975292551, 0.0),
                (56, 0.6816645059686987, 0.0),
                (70, 0.5256532405825242, 0.0),
                (84, 0.5254800293524405, 0.0),
                (98, 0.5255318455853101, 0.0),
                (112, 0.525583106337206, 0.0),
                (126, 0.5256338702543647, 0.0),
                (140, 0.6792442954649991, 0.0),
                (154, 0.6763060748253992, 0.0),
                (168, 0.6734968653963557, 0.0),
            ],
        ),
        (
            "i(vdd)",
            &[
                (0, -1.697706920225377e-08, 0.0),
                (14, -1.6977069205336608e-08, 0.0),
                (28, -1.616629551341362e-07, 0.0),
                (42, -8.429209560841021e-08, 0.0),
                (56, -4.40328057894451e-08, 0.0),
                (70, -3.525680549199971e-05, 0.0),
                (84, 1.278386328665683e-08, 0.0),
                (98, -7.758964010686127e-09, 0.0),
                (112, -2.0260616057776867e-08, 0.0),
                (126, -2.73869921454061e-08, 0.0),
                (140, -1.6024645828527004e-07, 0.0),
                (154, -8.258990564371876e-08, 0.0),
                (168, -4.219509411775154e-08, 0.0),
            ],
        ),
    ];
    // C's accepted time points at those indices.
    let indices = [0, 14, 28, 42, 56, 70, 84, 98, 112, 126, 140, 154, 168];
    let times = [
        0.0,
        7.28e-11,
        1.5499999999999998e-10,
        2.95e-10,
        4.3500000000000026e-10,
        5.4e-10,
        6.549999999999996e-10,
        7.949999999999991e-10,
        9.349999999999987e-10,
        1.0749999999999983e-09,
        1.155e-09,
        1.2949999999999996e-09,
        1.434999999999999e-09,
    ];
    if EXACT {
        for (&index, &time) in indices.iter().zip(&times) {
            same(plot.value("time", index).unwrap().re, time, "time");
        }
        check(plot, expected);
        return;
    }
    // Elsewhere: compare on C's grid by interpolating the port's waveform.
    let grid: Vec<f64> = (0..)
        .map_while(|i| plot.value("time", i).map(|v| v.re))
        .collect();
    for (name, points) in expected {
        for (&(_, value, _), &time) in points.iter().zip(&times) {
            let k = grid.partition_point(|&t| t < time).clamp(1, grid.len() - 1);
            let (t0, t1) = (grid[k - 1], grid[k]);
            let (v0, v1) = (
                plot.value(name, k - 1).unwrap().re,
                plot.value(name, k).unwrap().re,
            );
            let at = v0 + (v1 - v0) * (time - t0) / (t1 - t0);
            assert!(
                (at - value).abs() <= 1e-3 * value.abs() + 1e-6,
                "{name} at {time:e}: {at:e} vs {value:e}"
            );
        }
    }
}

#[test]
fn physical_taylor_preserves_terminal_incidence_and_split_charge_histories()
-> Result<(), ngspice_rs::primitives::SpiceError> {
    use ngspice_rs::devices::{Circuit, Limit, ModelContext, TransientTiming};
    use ngspice_rs::maths::Vector;

    fn close(actual: f64, expected: f64, absolute: f64) {
        assert!(
            (actual - expected).abs() <= absolute + 3e-7 * actual.abs().max(expected.abs()),
            "actual {actual:e}, expected {expected:e}"
        );
    }

    for (family, polarity) in [("nmos", 1.), ("pmos", -1.)] {
        for (cap, rgate, rbody, rds, gidl) in [
            (0, 0, 0, 0, 0),
            (1, 1, 1, 0, 1),
            (2, 2, 0, 1, 0),
            (2, 3, 2, 1, 1),
        ] {
            let text = format!(
                "BSIM4 physical terminal incidence\nm1 d g s b nch l=1u w=10u nf=2 m=3 nrd=2 nrs=3 ad=10p as=12p pd=12u ps=14u\n.model nch {family} level=54 capmod={cap} rgatemod={rgate} rbodymod={rbody} rdsmod={rds} gidlmod={gidl} igcmod=1 igbmod=1 rsh=10 rshg=2 rsw=100 rdw=100 rbpb=5 rbpd=15 rbps=15 rbdb=15 rbsb=15 cjs=1m cjd=2m cjsws=1n cjswd=1n cgdo=1n cgso=2n cgbo=1n agidl=2e-4 bgidl=2e8 cgidl=2e-4 egidl=0.1 agisl=2e-4 bgisl=2e8 cgisl=2e-4 egisl=0.1 alpha0=1e-8 beta0=0.1\n.end"
            );
            let deck = Parser::new()
                .parse_deck(&parse_deck_text(Path::new("bsim4-physical.cir"), &text))?;
            let mut c = Circuit::from_netlist(&deck)?;
            c.finalize()?;
            let prepared = c.prepare_run();
            let mut dae = prepared.prepare_physical_dae(
                &ModelContext::default(),
                TransientTiming::new(1e-6, 1e-3)?,
            )?;
            let nodes = c.device("m1").unwrap().terminals();
            let rows = nodes
                .iter()
                .map(|node| c.unknowns().node_row(*node).unwrap())
                .collect::<Vec<_>>();
            for (gate, drain, body) in [
                (0.5, 0.2, -0.3),
                (2.5, 1.8, -0.3),
                (2.0, -0.5, -0.8),
                (2.5, 0.3, 0.1),
            ] {
                let mut state = vec![Vector::zeros(c.unknown_count()); 9];
                for (i, &row) in rows.iter().enumerate() {
                    let name = &c.nodes().node(nodes[i]).unwrap().name;
                    let nominal = match name.as_str() {
                        "d" => drain + 0.001,
                        "g" => gate + 0.001,
                        "s" => -0.001,
                        "b" => body + 0.001,
                        _ if name.contains("drain") => drain,
                        _ if name.contains("gate") => gate,
                        _ if name.contains("source") => 0.,
                        _ if name.contains("body") => body,
                        _ => panic!("unexpected BSIM4 internal node {name}"),
                    };
                    let slope = [0.002, -0.003, 0.004, -0.001][i % 4];
                    state[0].set(row, polarity * nominal)?;
                    state[1].set(row, polarity * slope)?;
                    state[2].set(row, polarity * slope * 0.01)?;
                }
                let taylor = dae.evaluate_taylor(&state, 0., 1e-6, Limit::Right)?;
                for t in [0., 0.125, 0.5] {
                    let mut sample = Vector::zeros(c.unknown_count());
                    for &row in &rows {
                        sample.set(
                            row,
                            state.iter().rev().fold(0., |v: f64, coefficient| {
                                v.mul_add(t, coefficient.get(row).unwrap())
                            }),
                        )?;
                    }
                    let scalar = dae.evaluate(&sample, t * 1e-6, Limit::Right)?;
                    for &row in &rows {
                        let f = taylor
                            .residual
                            .iter()
                            .rev()
                            .fold(0., |v: f64, a| v.mul_add(t, a.get(row).unwrap()));
                        let q = taylor
                            .charge
                            .iter()
                            .rev()
                            .fold(0., |v: f64, a| v.mul_add(t, a.get(row).unwrap()));
                        close(f, scalar.residual.get(row).unwrap(), 1e-12);
                        close(q, scalar.charge.get(row).unwrap(), 1e-21);
                        for &col in &rows {
                            let g = taylor
                                .conductance
                                .iter()
                                .rev()
                                .fold(0., |v: f64, a| v.mul_add(t, a.get(row, col)));
                            let cap = taylor
                                .capacitance
                                .iter()
                                .rev()
                                .fold(0., |v: f64, a| v.mul_add(t, a.get(row, col)));
                            close(g, scalar.conductance.get(row, col), 1e-11);
                            close(cap, scalar.capacitance.get(row, col), 1e-20);
                        }
                    }
                }
                for (f, q) in taylor.residual.iter().zip(&taylor.charge) {
                    close(rows.iter().map(|&row| f.get(row).unwrap()).sum(), 0., 1e-10);
                    close(rows.iter().map(|&row| q.get(row).unwrap()).sum(), 0., 1e-21);
                }
            }
        }
    }
    Ok(())
}

#[test]
fn coupled_derivative_array_transient_uses_mos_physical_charge()
-> Result<(), ngspice_rs::primitives::SpiceError> {
    use ngspice_rs::devices::{Circuit, Limit, ModelContext, TransientTiming};
    use ngspice_rs::maths::Vector;
    fn close(actual: f64, expected: f64, absolute: f64, label: &str) {
        assert!(
            (actual - expected).abs() <= absolute + 2e-6 * actual.abs().max(expected.abs()),
            "{label}: actual {actual:e}, expected {expected:e}"
        );
    }

    for (level, cap) in [(8, 3), (54, 0), (54, 1), (54, 2)] {
        for (family, polarity) in [("nmos", 1.), ("pmos", -1.)] {
            for drain in [0.4, -0.4] {
                let initial = 1.2;
                let rate: f64 = 100.;
                let body = -0.7;
                let model = format!(
                    "m1 d g s b nch l=1u w=10u m=2 ad=10p as=12p pd=12u ps=14u\n.model nch {family} level={level} capmod={cap} cgdo=1n cgso=2n cgbo=1n"
                );
                let reference_deck = Parser::new().parse_deck(&parse_deck_text(
                    Path::new("mos-reference.cir"),
                    &format!("MOS physical reference\n{model}\n.end"),
                ))?;
                let mut reference = Circuit::from_netlist(&reference_deck)?;
                reference.finalize()?;
                let prepared = reference.prepare_run();
                let mut dae = prepared.prepare_physical_dae(
                    &ModelContext::default(),
                    TransientTiming::new(1e-6, 1e-3)?,
                )?;
                let rows = ["d", "g", "s", "b"].map(|name| {
                    reference
                        .unknowns()
                        .node_row(reference.nodes().get(name).unwrap())
                        .unwrap()
                });
                let scalar = |gate: f64| -> Result<Vector, ngspice_rs::primitives::SpiceError> {
                    let mut state = Vector::zeros(reference.unknown_count());
                    for (row, value) in rows.into_iter().zip([drain, gate, 0., body]) {
                        state.set(row, polarity * value)?;
                    }
                    Ok(state)
                };
                for method in ["gear", "trap"] {
                    let text = format!(
                        "MOS coupled derivative array\nvd d 0 {}\nvg g 0 PWL(0 {} 1m {})\nvs s 0 0\nvb b 0 {}\n{model}\nb1 first 0 v=1m*ddt(v(g))\nb2 out 0 v=1m*ddt(v(first))\nr1 out 0 1k\n.options method={method} reltol=1e-8 vntol=1n\n.tran 1u 20u 0 1u uic\n.end",
                        polarity * drain,
                        polarity * initial,
                        polarity * rate.mul_add(1e-3, initial),
                        polarity * body
                    );
                    let netlist = Parser::new()
                        .parse_deck(&parse_deck_text(Path::new("mos-coupled.cir"), &text))?;
                    let config = RunConfig::from_netlist(&netlist)?;
                    let mut c = config.circuit(&netlist)?;
                    let plot = config
                        .plan_for(&netlist.analyses[0])?
                        .run(&mut c, &config.context())?;
                    for i in 0..plot.point_count() {
                        let value = |name| plot.value(name, i).unwrap().re;
                        let time = value("time");
                        let gate = rate.mul_add(time, initial);
                        let sample = scalar(gate)?;
                        let residual = dae.evaluate(&sample, time, Limit::Right)?.residual.clone();
                        // Independent sampled charge derivative: not the
                        // Taylor derivative nor its analytic Jacobian.
                        let step = 1e-5;
                        let low = dae
                            .evaluate(&scalar(gate - step)?, time, Limit::Right)?
                            .charge
                            .clone();
                        let high = dae
                            .evaluate(&scalar(gate + step)?, time, Limit::Right)?
                            .charge
                            .clone();
                        for (row, source) in
                            rows.into_iter().zip(["i(vd)", "i(vg)", "i(vs)", "i(vb)"])
                        {
                            let displacement = (high.get(row).unwrap() - low.get(row).unwrap())
                                * rate
                                / (2. * step);
                            close(
                                value(source),
                                -residual.get(row).unwrap() - displacement,
                                1e-12,
                                source,
                            );
                        }
                        close(
                            value("v(first)"),
                            polarity * rate * 1e-3,
                            1e-10,
                            "first derivative",
                        );
                        close(value("v(out)"), 0., 1e-9, "second derivative");
                    }
                }
            }
        }
    }
    Ok(())
}

/// Exercise the branch-event machinery in a coupled derivative-array
/// transient, with a physical drain/source exchange in the middle of a
/// smooth forcing interval rather than at an independent source knot.
#[test]
fn coupled_derivative_array_crosses_channel_reversal()
-> Result<(), ngspice_rs::primitives::SpiceError> {
    use ngspice_rs::devices::{Circuit, Limit, ModelContext, TransientTiming};
    use ngspice_rs::maths::Vector;
    const GATE: f64 = 1.4;
    const BODY: f64 = -0.5;
    const INITIAL_DRAIN: f64 = -0.02;
    const RATE: f64 = 2000.;

    for (level, cap) in [(8, 3), (54, 0), (54, 1), (54, 2)] {
        for (family, polarity) in [("nmos", 1.), ("pmos", -1.)] {
            let model = format!(
                "m1 d g s b nch l=1u w=10u m=2 ad=10p as=12p pd=12u ps=14u\n.model nch {family} level={level} capmod={cap} cgdo=1n cgso=2n cgbo=1n"
            );
            let deck = Parser::new().parse_deck(&parse_deck_text(
                Path::new("mos-reversal-reference.cir"),
                &format!("MOS reversal reference\n{model}\n.end"),
            ))?;
            let mut reference = Circuit::from_netlist(&deck)?;
            reference.finalize()?;
            let prepared = reference.prepare_run();
            let mut dae = prepared.prepare_physical_dae(
                &ModelContext::default(),
                TransientTiming::new(1e-6, 40e-6)?,
            )?;
            let rows = ["d", "g", "s", "b"].map(|name| {
                reference
                    .unknowns()
                    .node_row(reference.nodes().get(name).unwrap())
                    .unwrap()
            });
            let scalar = |drain: f64| -> Result<Vector, ngspice_rs::primitives::SpiceError> {
                let mut state = Vector::zeros(reference.unknown_count());
                for (row, value) in rows.into_iter().zip([drain, GATE, 0., BODY]) {
                    state.set(row, polarity * value)?;
                }
                Ok(state)
            };

            for method in ["gear", "trap"] {
                let text = format!(
                    "MOS reversal derivative array\nvd d 0 PWL(0 {} 40u {})\nvg g 0 {}\nvs s 0 0\nvb b 0 {}\n{model}\nb1 first 0 v=1m*ddt(v(d))\nb2 out 0 v=1m*ddt(v(first))\nr1 out 0 1k\n.options method={method} reltol=1e-8 vntol=1n\n.tran 1u 20u 0 1u uic\n.end",
                    polarity * INITIAL_DRAIN,
                    polarity * RATE.mul_add(40e-6, INITIAL_DRAIN),
                    polarity * GATE,
                    polarity * BODY,
                );
                let netlist = Parser::new()
                    .parse_deck(&parse_deck_text(Path::new("mos-reversal.cir"), &text))?;
                let config = RunConfig::from_netlist(&netlist)?;
                let mut c = config.circuit(&netlist)?;
                let plot = config
                    .plan_for(&netlist.analyses[0])?
                    .run(&mut c, &config.context())?;
                let mut sides = [false; 2];
                for i in 0..plot.point_count() {
                    let value = |name| plot.value(name, i).unwrap().re;
                    let time = value("time");
                    let drain = RATE.mul_add(time, INITIAL_DRAIN);
                    sides[usize::from(drain >= 0.)] = true;
                    let actual = ["i(vd)", "i(vg)", "i(vs)", "i(vb)"].map(value);
                    let magnitude = actual.iter().map(|x| x.abs()).sum::<f64>();
                    assert!(actual.iter().all(|v| v.is_finite()));
                    assert!(actual.iter().sum::<f64>().abs() <= 1e-12 + 1e-9 * magnitude);
                    let point = dae.evaluate(&scalar(drain)?, time, Limit::Right)?;
                    let residual = point.residual.clone();
                    let charge = point.charge.clone();
                    // Use a second-order one-sided independent charge
                    // difference on the selected orientation. Never form
                    // a centered quotient across source/drain exchange.
                    let h = 1e-5 * if drain < 0. { -1. } else { 1. };
                    let q1 = dae
                        .evaluate(&scalar(drain + h)?, time, Limit::Right)?
                        .charge
                        .clone();
                    let q2 = dae
                        .evaluate(&scalar(drain + 2. * h)?, time, Limit::Right)?
                        .charge
                        .clone();
                    for (row, measured) in rows.into_iter().zip(actual) {
                        let q0 = charge.get(row).unwrap();
                        let derivative = (4. * (q1.get(row).unwrap() - q0)
                            - (q2.get(row).unwrap() - q0))
                            / (2. * h);
                        let expected = -residual.get(row).unwrap() - RATE * derivative;
                        assert!(
                            (measured - expected).abs()
                                <= 1e-12 + 2e-6 * measured.abs().max(expected.abs()),
                            "level={level} cap={cap} p={polarity} {method} t={time:e} row={row}: {measured:e} vs {expected:e}"
                        );
                    }
                    assert!((value("v(first)") - polarity * RATE * 1e-3).abs() <= 1e-9);
                    assert!(value("v(out)").abs() <= 1e-8);
                }
                assert_eq!(sides, [true; 2]);
            }
        }
    }
    Ok(())
}
