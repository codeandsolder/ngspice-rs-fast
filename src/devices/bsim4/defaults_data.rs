// Generated literal-default facts, not C implementation code.
// Source: ngspice 059066b532f7073fccf4173ef269309eafdc7f0c, bsim4/b4set.c.
// Dependent defaults and diagnostics are implemented separately.

/// Canonical model keyword and literal default, in C setup order.
#[rustfmt::skip]
pub(super) const LITERALS: &[(&str, f64)] = &[
    ("mobmod", 0.0),
    ("binunit", 1.0),
    ("paramchk", 1.0),
    ("diomod", 1.0),
    ("cvchargemod", 0.0),
    ("capmod", 2.0),
    ("rdsmod", 0.0),
    ("rbodymod", 0.0),
    ("rgatemod", 0.0),
    ("permod", 1.0),
    ("geomod", 0.0),
    ("fnoimod", 1.0),
    ("tnoimod", 0.0),
    ("trnqsmod", 0.0),
    ("acnqsmod", 0.0),
    ("mtrlmod", 0.0),
    ("mtrlcompatmod", 0.0),
    ("igcmod", 0.0),
    ("igbmod", 0.0),
    ("tempmod", 0.0),
    ("toxref", 3e-9),
    ("eot", 1.5e-9),
    ("tempeot", 300.15),
    ("leffeot", 1.0),
    ("weffeot", 10.0),
    ("ados", 1.0),
    ("bdos", 1.0),
    ("toxe", 3e-9),
    ("dtox", 0.0),
    ("epsrox", 3.9),
    ("cdsc", 0.00024),
    ("cdscb", 0.0),
    ("cdscd", 0.0),
    ("cit", 0.0),
    ("nfactor", 1.0),
    ("xj", 1.5e-7),
    ("vsat", 80000.0),
    ("at", 33000.0),
    ("a0", 1.0),
    ("ags", 0.0),
    ("a1", 0.0),
    ("a2", 1.0),
    ("keta", -0.047),
    ("nsub", 6e16),
    ("phig", 4.05),
    ("epsrgate", 11.7),
    ("easub", 4.05),
    ("epsrsub", 11.7),
    ("ni0sub", 14_500_000_000.0),
    ("bg0sub", 1.16),
    ("tbgasub", 0.000_702),
    ("tbgbsub", 1108.0),
    ("ndep", 1.7e17),
    ("nsd", 1e20),
    ("phin", 0.0),
    ("ngate", 0.0),
    ("vbm", -3.0),
    ("xt", 1.55e-7),
    ("kt1", -0.11),
    ("kt1l", 0.0),
    ("kt2", 0.022),
    ("k3", 80.0),
    ("k3b", 0.0),
    ("w0", 2.5e-6),
    ("lpe0", 1.74e-7),
    ("lpeb", 0.0),
    ("dvtp0", 0.0),
    ("dvtp1", 0.0),
    ("dvtp2", 0.0),
    ("dvtp3", 0.0),
    ("dvtp4", 0.0),
    ("dvtp5", 0.0),
    ("dvt0", 2.2),
    ("dvt1", 0.53),
    ("dvt2", -0.032),
    ("dvt0w", 0.0),
    ("dvt1w", 5_300_000.0),
    ("dvt2w", -0.032),
    ("drout", 0.56),
    ("vfb", -1.0),
    ("ua1", 1e-9),
    ("ub", 1e-19),
    ("ub1", -1e-18),
    ("ud", 0.0),
    ("ud1", 0.0),
    ("up", 0.0),
    ("lp", 1e-8),
    ("ute", -1.5),
    ("ucste", -0.004_775),
    ("voff", -0.08),
    ("voffl", 0.0),
    ("voffcvl", 0.0),
    ("minv", 0.0),
    ("minvcv", 0.0),
    ("fprout", 0.0),
    ("pdits", 0.0),
    ("pditsd", 0.0),
    ("pditsl", 0.0),
    ("delta", 0.01),
    ("rdswmin", 0.0),
    ("rdwmin", 0.0),
    ("rswmin", 0.0),
    ("rdsw", 200.0),
    ("rdw", 100.0),
    ("rsw", 100.0),
    ("prwg", 1.0),
    ("prwb", 0.0),
    ("prt", 0.0),
    ("eta0", 0.08),
    ("etab", -0.07),
    ("pclm", 1.3),
    ("pdiblc1", 0.39),
    ("pdiblc2", 0.0086),
    ("pdiblcb", 0.0),
    ("pscbe1", 424_000_000.0),
    ("pscbe2", 1e-5),
    ("pvag", 0.0),
    ("wr", 1.0),
    ("dwg", 0.0),
    ("dwb", 0.0),
    ("b0", 0.0),
    ("b1", 0.0),
    ("alpha0", 0.0),
    ("alpha1", 0.0),
    ("beta0", 0.0),
    ("gidlmod", 0.0),
    ("agidl", 0.0),
    ("bgidl", 2_300_000_000.0),
    ("cgidl", 0.5),
    ("egidl", 0.8),
    ("rgidl", 1.0),
    ("kgidl", 0.0),
    ("fgidl", 1.0),
    ("aigbacc", 0.0136),
    ("bigbacc", 0.00171),
    ("cigbacc", 0.075),
    ("aigbinv", 0.0111),
    ("bigbinv", 0.000_949),
    ("cigbinv", 0.006),
    ("nigc", 1.0),
    ("nigbinv", 3.0),
    ("nigbacc", 1.0),
    ("ntox", 1.0),
    ("eigbinv", 1.1),
    ("pigcd", 1.0),
    ("poxedge", 1.0),
    ("xrcrg1", 12.0),
    ("xrcrg2", 1.0),
    ("ijthsfwd", 0.1),
    ("ijthsrev", 0.1),
    ("tnoia", 1.5),
    ("tnoib", 3.5),
    ("tnoic", 0.0),
    ("rnoia", 0.577),
    ("rnoib", 0.5164),
    ("gidlclamp", -1e-5),
    ("idovvds", 1e-9),
    ("rnoic", 0.395),
    ("ntnoi", 1.0),
    ("lambda", 0.0),
    ("vtl", 200_000.0),
    ("xn", 3.0),
    ("lc", 5e-9),
    ("vfbsdoff", 0.0),
    ("tvfbsdoff", 0.0),
    ("tvoff", 0.0),
    ("tnfactor", 0.0),
    ("teta0", 0.0),
    ("tvoffcv", 0.0),
    ("lintnoi", 0.0),
    ("xjbvs", 1.0),
    ("bvs", 10.0),
    ("gbmin", 1e-12),
    ("rbdb", 50.0),
    ("rbpb", 50.0),
    ("rbsb", 50.0),
    ("rbps", 50.0),
    ("rbpd", 50.0),
    ("rbps0", 50.0),
    ("rbpsl", 0.0),
    ("rbpsw", 0.0),
    ("rbpsnf", 0.0),
    ("rbpd0", 50.0),
    ("rbpdl", 0.0),
    ("rbpdw", 0.0),
    ("rbpdnf", 0.0),
    ("rbpbx0", 100.0),
    ("rbpbxl", 0.0),
    ("rbpbxw", 0.0),
    ("rbpbxnf", 0.0),
    ("rbpby0", 100.0),
    ("rbpbyl", 0.0),
    ("rbpbyw", 0.0),
    ("rbpbynf", 0.0),
    ("rbsbx0", 100.0),
    ("rbsby0", 100.0),
    ("rbdbx0", 100.0),
    ("rbdby0", 100.0),
    ("rbsdbxl", 0.0),
    ("rbsdbxw", 0.0),
    ("rbsdbxnf", 0.0),
    ("rbsdbyl", 0.0),
    ("rbsdbyw", 0.0),
    ("rbsdbynf", 0.0),
    ("cgsl", 0.0),
    ("cgdl", 0.0),
    ("ckappas", 0.6),
    ("clc", 1e-7),
    ("cle", 0.6),
    ("vfbcv", -1.0),
    ("acde", 1.0),
    ("moin", 15.0),
    ("noff", 1.0),
    ("voffcv", 0.0),
    ("dmcg", 0.0),
    ("dmdg", 0.0),
    ("dmcgt", 0.0),
    ("xgw", 0.0),
    ("xgl", 0.0),
    ("rshg", 0.1),
    ("ngcon", 1.0),
    ("tcj", 0.0),
    ("tpb", 0.0),
    ("tcjsw", 0.0),
    ("tpbsw", 0.0),
    ("tcjswg", 0.0),
    ("tpbswg", 0.0),
    ("lcdsc", 0.0),
    ("lcdscb", 0.0),
    ("lcdscd", 0.0),
    ("lcit", 0.0),
    ("lnfactor", 0.0),
    ("lxj", 0.0),
    ("lvsat", 0.0),
    ("lat", 0.0),
    ("la0", 0.0),
    ("lags", 0.0),
    ("la1", 0.0),
    ("la2", 0.0),
    ("lketa", 0.0),
    ("lnsub", 0.0),
    ("lndep", 0.0),
    ("lnsd", 0.0),
    ("lphin", 0.0),
    ("lngate", 0.0),
    ("lvbm", 0.0),
    ("lxt", 0.0),
    ("lk1", 0.0),
    ("lkt1", 0.0),
    ("lkt1l", 0.0),
    ("lkt2", 0.0),
    ("lk2", 0.0),
    ("lk3", 0.0),
    ("lk3b", 0.0),
    ("lw0", 0.0),
    ("llpe0", 0.0),
    ("llpeb", 0.0),
    ("ldvtp0", 0.0),
    ("ldvtp1", 0.0),
    ("ldvtp2", 0.0),
    ("ldvtp3", 0.0),
    ("ldvtp4", 0.0),
    ("ldvtp5", 0.0),
    ("ldvt0", 0.0),
    ("ldvt1", 0.0),
    ("ldvt2", 0.0),
    ("ldvt0w", 0.0),
    ("ldvt1w", 0.0),
    ("ldvt2w", 0.0),
    ("ldrout", 0.0),
    ("ldsub", 0.0),
    ("lvth0", 0.0),
    ("lua", 0.0),
    ("lua1", 0.0),
    ("lub", 0.0),
    ("lub1", 0.0),
    ("luc", 0.0),
    ("luc1", 0.0),
    ("lud", 0.0),
    ("lud1", 0.0),
    ("lup", 0.0),
    ("llp", 0.0),
    ("lu0", 0.0),
    ("lute", 0.0),
    ("lucste", 0.0),
    ("lvoff", 0.0),
    ("lminv", 0.0),
    ("lminvcv", 0.0),
    ("lfprout", 0.0),
    ("lpdits", 0.0),
    ("lpditsd", 0.0),
    ("ldelta", 0.0),
    ("lrdsw", 0.0),
    ("lrdw", 0.0),
    ("lrsw", 0.0),
    ("lprwb", 0.0),
    ("lprwg", 0.0),
    ("lprt", 0.0),
    ("leta0", 0.0),
    ("letab", -0.0),
    ("lpclm", 0.0),
    ("lpdiblc1", 0.0),
    ("lpdiblc2", 0.0),
    ("lpdiblcb", 0.0),
    ("lpscbe1", 0.0),
    ("lpscbe2", 0.0),
    ("lpvag", 0.0),
    ("lwr", 0.0),
    ("ldwg", 0.0),
    ("ldwb", 0.0),
    ("lb0", 0.0),
    ("lb1", 0.0),
    ("lalpha0", 0.0),
    ("lalpha1", 0.0),
    ("lbeta0", 0.0),
    ("lagidl", 0.0),
    ("lbgidl", 0.0),
    ("lcgidl", 0.0),
    ("legidl", 0.0),
    ("lrgidl", 0.0),
    ("lkgidl", 0.0),
    ("lfgidl", 0.0),
    ("laigc", 0.0),
    ("lbigc", 0.0),
    ("lcigc", 0.0),
    ("laigs", 0.0),
    ("laigd", 0.0),
    ("laigsd", 0.0),
    ("lbigs", 0.0),
    ("lbigd", 0.0),
    ("lbigsd", 0.0),
    ("lcigs", 0.0),
    ("lcigd", 0.0),
    ("lcigsd", 0.0),
    ("laigbacc", 0.0),
    ("lbigbacc", 0.0),
    ("lcigbacc", 0.0),
    ("laigbinv", 0.0),
    ("lbigbinv", 0.0),
    ("lcigbinv", 0.0),
    ("lnigc", 0.0),
    ("lnigbinv", 0.0),
    ("lnigbacc", 0.0),
    ("lntox", 0.0),
    ("leigbinv", 0.0),
    ("lpigcd", 0.0),
    ("lpoxedge", 0.0),
    ("lxrcrg1", 0.0),
    ("lxrcrg2", 0.0),
    ("leu", 0.0),
    ("lucs", 0.0),
    ("lvfb", 0.0),
    ("llambda", 0.0),
    ("lvtl", 0.0),
    ("lxn", 0.0),
    ("lvfbsdoff", 0.0),
    ("ltvfbsdoff", 0.0),
    ("ltvoff", 0.0),
    ("ltnfactor", 0.0),
    ("lteta0", 0.0),
    ("ltvoffcv", 0.0),
    ("lcgsl", 0.0),
    ("lcgdl", 0.0),
    ("lckappas", 0.0),
    ("lckappad", 0.0),
    ("lclc", 0.0),
    ("lcle", 0.0),
    ("lcf", 0.0),
    ("lvfbcv", 0.0),
    ("lacde", 0.0),
    ("lmoin", 0.0),
    ("lnoff", 0.0),
    ("lvoffcv", 0.0),
    ("wcdsc", 0.0),
    ("wcdscb", 0.0),
    ("wcdscd", 0.0),
    ("wcit", 0.0),
    ("wnfactor", 0.0),
    ("wxj", 0.0),
    ("wvsat", 0.0),
    ("wat", 0.0),
    ("wa0", 0.0),
    ("wags", 0.0),
    ("wa1", 0.0),
    ("wa2", 0.0),
    ("wketa", 0.0),
    ("wnsub", 0.0),
    ("wndep", 0.0),
    ("wnsd", 0.0),
    ("wphin", 0.0),
    ("wngate", 0.0),
    ("wvbm", 0.0),
    ("wxt", 0.0),
    ("wk1", 0.0),
    ("wkt1", 0.0),
    ("wkt1l", 0.0),
    ("wkt2", 0.0),
    ("wk2", 0.0),
    ("wk3", 0.0),
    ("wk3b", 0.0),
    ("ww0", 0.0),
    ("wlpe0", 0.0),
    ("wlpeb", 0.0),
    ("wdvtp0", 0.0),
    ("wdvtp1", 0.0),
    ("wdvtp2", 0.0),
    ("wdvtp3", 0.0),
    ("wdvtp4", 0.0),
    ("wdvtp5", 0.0),
    ("wdvt0", 0.0),
    ("wdvt1", 0.0),
    ("wdvt2", 0.0),
    ("wdvt0w", 0.0),
    ("wdvt1w", 0.0),
    ("wdvt2w", 0.0),
    ("wdrout", 0.0),
    ("wdsub", 0.0),
    ("wvth0", 0.0),
    ("wua", 0.0),
    ("wua1", 0.0),
    ("wub", 0.0),
    ("wub1", 0.0),
    ("wuc", 0.0),
    ("wuc1", 0.0),
    ("wud", 0.0),
    ("wud1", 0.0),
    ("wup", 0.0),
    ("wlp", 0.0),
    ("wu0", 0.0),
    ("wute", 0.0),
    ("wucste", 0.0),
    ("wvoff", 0.0),
    ("wminv", 0.0),
    ("wminvcv", 0.0),
    ("wfprout", 0.0),
    ("wpdits", 0.0),
    ("wpditsd", 0.0),
    ("wdelta", 0.0),
    ("wrdsw", 0.0),
    ("wrdw", 0.0),
    ("wrsw", 0.0),
    ("wprwb", 0.0),
    ("wprwg", 0.0),
    ("wprt", 0.0),
    ("weta0", 0.0),
    ("wetab", 0.0),
    ("wpclm", 0.0),
    ("wpdiblc1", 0.0),
    ("wpdiblc2", 0.0),
    ("wpdiblcb", 0.0),
    ("wpscbe1", 0.0),
    ("wpscbe2", 0.0),
    ("wpvag", 0.0),
    ("wwr", 0.0),
    ("wdwg", 0.0),
    ("wdwb", 0.0),
    ("wb0", 0.0),
    ("wb1", 0.0),
    ("walpha0", 0.0),
    ("walpha1", 0.0),
    ("wbeta0", 0.0),
    ("wagidl", 0.0),
    ("wbgidl", 0.0),
    ("wcgidl", 0.0),
    ("wegidl", 0.0),
    ("wrgidl", 0.0),
    ("wkgidl", 0.0),
    ("wfgidl", 0.0),
    ("waigc", 0.0),
    ("wbigc", 0.0),
    ("wcigc", 0.0),
    ("waigs", 0.0),
    ("waigd", 0.0),
    ("waigsd", 0.0),
    ("wbigs", 0.0),
    ("wbigd", 0.0),
    ("wbigsd", 0.0),
    ("wcigs", 0.0),
    ("wcigd", 0.0),
    ("wcigsd", 0.0),
    ("waigbacc", 0.0),
    ("wbigbacc", 0.0),
    ("wcigbacc", 0.0),
    ("waigbinv", 0.0),
    ("wbigbinv", 0.0),
    ("wcigbinv", 0.0),
    ("wnigc", 0.0),
    ("wnigbinv", 0.0),
    ("wnigbacc", 0.0),
    ("wntox", 0.0),
    ("weigbinv", 0.0),
    ("wpigcd", 0.0),
    ("wpoxedge", 0.0),
    ("wxrcrg1", 0.0),
    ("wxrcrg2", 0.0),
    ("weu", 0.0),
    ("wucs", 0.0),
    ("wvfb", 0.0),
    ("wlambda", 0.0),
    ("wvtl", 0.0),
    ("wxn", 0.0),
    ("wvfbsdoff", 0.0),
    ("wtvfbsdoff", 0.0),
    ("wtvoff", 0.0),
    ("wtnfactor", 0.0),
    ("wteta0", 0.0),
    ("wtvoffcv", 0.0),
    ("wcgsl", 0.0),
    ("wcgdl", 0.0),
    ("wckappas", 0.0),
    ("wckappad", 0.0),
    ("wcf", 0.0),
    ("wclc", 0.0),
    ("wcle", 0.0),
    ("wvfbcv", 0.0),
    ("wacde", 0.0),
    ("wmoin", 0.0),
    ("wnoff", 0.0),
    ("wvoffcv", 0.0),
    ("pcdsc", 0.0),
    ("pcdscb", 0.0),
    ("pcdscd", 0.0),
    ("pcit", 0.0),
    ("pnfactor", 0.0),
    ("pxj", 0.0),
    ("pvsat", 0.0),
    ("pat", 0.0),
    ("pa0", 0.0),
    ("pags", 0.0),
    ("pa1", 0.0),
    ("pa2", 0.0),
    ("pketa", 0.0),
    ("pnsub", 0.0),
    ("pndep", 0.0),
    ("pnsd", 0.0),
    ("pphin", 0.0),
    ("pngate", 0.0),
    ("pvbm", 0.0),
    ("pxt", 0.0),
    ("pk1", 0.0),
    ("pkt1", 0.0),
    ("pkt1l", 0.0),
    ("pk2", 0.0),
    ("pkt2", 0.0),
    ("pk3", 0.0),
    ("pk3b", 0.0),
    ("pw0", 0.0),
    ("plpe0", 0.0),
    ("plpeb", 0.0),
    ("pdvtp0", 0.0),
    ("pdvtp1", 0.0),
    ("pdvtp2", 0.0),
    ("pdvtp3", 0.0),
    ("pdvtp4", 0.0),
    ("pdvtp5", 0.0),
    ("pdvt0", 0.0),
    ("pdvt1", 0.0),
    ("pdvt2", 0.0),
    ("pdvt0w", 0.0),
    ("pdvt1w", 0.0),
    ("pdvt2w", 0.0),
    ("pdrout", 0.0),
    ("pdsub", 0.0),
    ("pvth0", 0.0),
    ("pua", 0.0),
    ("pua1", 0.0),
    ("pub", 0.0),
    ("pub1", 0.0),
    ("puc", 0.0),
    ("puc1", 0.0),
    ("pud", 0.0),
    ("pud1", 0.0),
    ("pup", 0.0),
    ("plp", 0.0),
    ("pu0", 0.0),
    ("pute", 0.0),
    ("pucste", 0.0),
    ("pvoff", 0.0),
    ("pminv", 0.0),
    ("pminvcv", 0.0),
    ("pfprout", 0.0),
    ("ppdits", 0.0),
    ("ppditsd", 0.0),
    ("pdelta", 0.0),
    ("prdsw", 0.0),
    ("prdw", 0.0),
    ("prsw", 0.0),
    ("pprwb", 0.0),
    ("pprwg", 0.0),
    ("pprt", 0.0),
    ("peta0", 0.0),
    ("petab", 0.0),
    ("ppclm", 0.0),
    ("ppdiblc1", 0.0),
    ("ppdiblc2", 0.0),
    ("ppdiblcb", 0.0),
    ("ppscbe1", 0.0),
    ("ppscbe2", 0.0),
    ("ppvag", 0.0),
    ("pwr", 0.0),
    ("pdwg", 0.0),
    ("pdwb", 0.0),
    ("pb0", 0.0),
    ("pb1", 0.0),
    ("palpha0", 0.0),
    ("palpha1", 0.0),
    ("pbeta0", 0.0),
    ("pagidl", 0.0),
    ("pbgidl", 0.0),
    ("pcgidl", 0.0),
    ("pegidl", 0.0),
    ("prgidl", 0.0),
    ("pkgidl", 0.0),
    ("pfgidl", 0.0),
    ("paigc", 0.0),
    ("pbigc", 0.0),
    ("pcigc", 0.0),
    ("paigs", 0.0),
    ("paigd", 0.0),
    ("paigsd", 0.0),
    ("pbigs", 0.0),
    ("pbigd", 0.0),
    ("pbigsd", 0.0),
    ("pcigs", 0.0),
    ("pcigd", 0.0),
    ("pcigsd", 0.0),
    ("paigbacc", 0.0),
    ("pbigbacc", 0.0),
    ("pcigbacc", 0.0),
    ("paigbinv", 0.0),
    ("pbigbinv", 0.0),
    ("pcigbinv", 0.0),
    ("pnigc", 0.0),
    ("pnigbinv", 0.0),
    ("pnigbacc", 0.0),
    ("pntox", 0.0),
    ("peigbinv", 0.0),
    ("ppigcd", 0.0),
    ("ppoxedge", 0.0),
    ("pxrcrg1", 0.0),
    ("pxrcrg2", 0.0),
    ("peu", 0.0),
    ("pucs", 0.0),
    ("pvfb", 0.0),
    ("plambda", 0.0),
    ("pvtl", 0.0),
    ("pxn", 0.0),
    ("pvfbsdoff", 0.0),
    ("ptvfbsdoff", 0.0),
    ("ptvoff", 0.0),
    ("ptnfactor", 0.0),
    ("pteta0", 0.0),
    ("ptvoffcv", 0.0),
    ("pcgsl", 0.0),
    ("pcgdl", 0.0),
    ("pckappas", 0.0),
    ("pckappad", 0.0),
    ("pcf", 0.0),
    ("pclc", 0.0),
    ("pcle", 0.0),
    ("pvfbcv", 0.0),
    ("pacde", 0.0),
    ("pmoin", 0.0),
    ("pnoff", 0.0),
    ("pvoffcv", 0.0),
    ("gamma1", 0.0),
    ("lgamma1", 0.0),
    ("wgamma1", 0.0),
    ("pgamma1", 0.0),
    ("gamma2", 0.0),
    ("lgamma2", 0.0),
    ("wgamma2", 0.0),
    ("pgamma2", 0.0),
    ("vbx", 0.0),
    ("lvbx", 0.0),
    ("wvbx", 0.0),
    ("pvbx", 0.0),
    ("lint", 0.0),
    ("ll", 0.0),
    ("lln", 1.0),
    ("lw", 0.0),
    ("lwn", 1.0),
    ("lwl", 0.0),
    ("lmin", 0.0),
    ("lmax", 1.0),
    ("wint", 0.0),
    ("wl", 0.0),
    ("wln", 1.0),
    ("ww", 0.0),
    ("wwn", 1.0),
    ("wwl", 0.0),
    ("wmin", 0.0),
    ("wmax", 1.0),
    ("xl", 0.0),
    ("xw", 0.0),
    ("xpart", 0.0),
    ("rsh", 0.0),
    ("cjs", 0.0005),
    ("cjsws", 5e-10),
    ("jss", 0.0001),
    ("jsws", 0.0),
    ("jswgs", 0.0),
    ("pbs", 1.0),
    ("pbsws", 1.0),
    ("mjs", 0.5),
    ("mjsws", 0.33),
    ("njs", 1.0),
    ("xtis", 3.0),
    ("jtss", 0.0),
    ("jtssws", 0.0),
    ("jtsswgs", 0.0),
    ("jtweff", 0.0),
    ("njts", 20.0),
    ("njtssw", 20.0),
    ("njtsswg", 20.0),
    ("xtss", 0.02),
    ("xtssws", 0.02),
    ("xtsswgs", 0.02),
    ("tnjts", 0.0),
    ("tnjtssw", 0.0),
    ("tnjtsswg", 0.0),
    ("vtss", 10.0),
    ("vtssws", 10.0),
    ("vtsswgs", 10.0),
    ("noic", 8_750_000_000.0),
    ("em", 41_000_000.0),
    ("ef", 1.0),
    ("af", 1.0),
    ("kf", 0.0),
    ("vgs_max", 1e99),
    ("vgd_max", 1e99),
    ("vgb_max", 1e99),
    ("vds_max", 1e99),
    ("vbs_max", 1e99),
    ("vbd_max", 1e99),
    ("vgsr_max", 1e99),
    ("vgdr_max", 1e99),
    ("vgbr_max", 1e99),
    ("vbsr_max", 1e99),
    ("vbdr_max", 1e99),
    ("saref", 1e-6),
    ("sbref", 1e-6),
    ("wlod", 0.0),
    ("ku0", 0.0),
    ("kvsat", 0.0),
    ("kvth0", 0.0),
    ("tku0", 0.0),
    ("llodku0", 0.0),
    ("wlodku0", 0.0),
    ("llodvth", 0.0),
    ("wlodvth", 0.0),
    ("lku0", 0.0),
    ("wku0", 0.0),
    ("pku0", 0.0),
    ("lkvth0", 0.0),
    ("wkvth0", 0.0),
    ("pkvth0", 0.0),
    ("stk2", 0.0),
    ("lodk2", 1.0),
    ("steta0", 0.0),
    ("lodeta0", 1.0),
    ("web", 0.0),
    ("wec", 0.0),
    ("kvth0we", 0.0),
    ("k2we", 0.0),
    ("ku0we", 0.0),
    ("scref", 1e-6),
    ("wpemod", 0.0),
    ("lkvth0we", 0.0),
    ("lk2we", 0.0),
    ("lku0we", 0.0),
    ("wkvth0we", 0.0),
    ("wk2we", 0.0),
    ("wku0we", 0.0),
    ("pkvth0we", 0.0),
    ("pk2we", 0.0),
    ("pku0we", 0.0),
];

use super::fields::Model;

/// Independent given-guarded defaults, extracted from the model-loop body.
#[rustfmt::skip]
#[expect(clippy::too_many_lines, reason = "generated typed field assignments; no handwritten physics")]
pub(super) const fn apply_independent_defaults(m: &mut Model) {
    if !m.binUnitGiven { m.binUnit = 1; }
    if !m.paramChkGiven { m.paramChk = 1; }
    if !m.cvchargeModGiven { m.cvchargeMod = 0; }
    if !m.geoModGiven { m.geoMod = 0; }
    if !m.toxrefGiven { m.toxref = 3e-9; }
    if !m.eotGiven { m.eot = 1.5e-9; }
    if !m.tempeotGiven { m.tempeot = 300.15; }
    if !m.leffeotGiven { m.leffeot = 1.0; }
    if !m.weffeotGiven { m.weffeot = 10.0; }
    if !m.adosGiven { m.ados = 1.0; }
    if !m.bdosGiven { m.bdos = 1.0; }
    if !m.toxeGiven { m.toxe = 3e-9; }
    if !m.dtoxGiven { m.dtox = 0.0; }
    if !m.epsroxGiven { m.epsrox = 3.9; }
    if !m.cdscGiven { m.cdsc = 0.00024; }
    if !m.cdscbGiven { m.cdscb = 0.0; }
    if !m.cdscdGiven { m.cdscd = 0.0; }
    if !m.citGiven { m.cit = 0.0; }
    if !m.nfactorGiven { m.nfactor = 1.0; }
    if !m.xjGiven { m.xj = 1.5e-7; }
    if !m.vsatGiven { m.vsat = 80000.0; }
    if !m.atGiven { m.at = 33000.0; }
    if !m.a0Given { m.a0 = 1.0; }
    if !m.agsGiven { m.ags = 0.0; }
    if !m.a1Given { m.a1 = 0.0; }
    if !m.a2Given { m.a2 = 1.0; }
    if !m.ketaGiven { m.keta = -0.047; }
    if !m.nsubGiven { m.nsub = 6e16; }
    if !m.phigGiven { m.phig = 4.05; }
    if !m.epsrgateGiven { m.epsrgate = 11.7; }
    if !m.easubGiven { m.easub = 4.05; }
    if !m.epsrsubGiven { m.epsrsub = 11.7; }
    if !m.ni0subGiven { m.ni0sub = 14_500_000_000.0; }
    if !m.bg0subGiven { m.bg0sub = 1.16; }
    if !m.tbgasubGiven { m.tbgasub = 0.000_702; }
    if !m.tbgbsubGiven { m.tbgbsub = 1108.0; }
    if !m.ndepGiven { m.ndep = 1.7e17; }
    if !m.nsdGiven { m.nsd = 1e20; }
    if !m.phinGiven { m.phin = 0.0; }
    if !m.ngateGiven { m.ngate = 0.0; }
    if !m.vbmGiven { m.vbm = -3.0; }
    if !m.xtGiven { m.xt = 1.55e-7; }
    if !m.kt1Given { m.kt1 = -0.11; }
    if !m.kt1lGiven { m.kt1l = 0.0; }
    if !m.kt2Given { m.kt2 = 0.022; }
    if !m.k3Given { m.k3 = 80.0; }
    if !m.k3bGiven { m.k3b = 0.0; }
    if !m.w0Given { m.w0 = 2.5e-6; }
    if !m.lpe0Given { m.lpe0 = 1.74e-7; }
    if !m.lpebGiven { m.lpeb = 0.0; }
    if !m.dvtp0Given { m.dvtp0 = 0.0; }
    if !m.dvtp1Given { m.dvtp1 = 0.0; }
    if !m.dvtp2Given { m.dvtp2 = 0.0; }
    if !m.dvtp3Given { m.dvtp3 = 0.0; }
    if !m.dvtp4Given { m.dvtp4 = 0.0; }
    if !m.dvtp5Given { m.dvtp5 = 0.0; }
    if !m.dvt0Given { m.dvt0 = 2.2; }
    if !m.dvt1Given { m.dvt1 = 0.53; }
    if !m.dvt2Given { m.dvt2 = -0.032; }
    if !m.dvt0wGiven { m.dvt0w = 0.0; }
    if !m.dvt1wGiven { m.dvt1w = 5_300_000.0; }
    if !m.dvt2wGiven { m.dvt2w = -0.032; }
    if !m.droutGiven { m.drout = 0.56; }
    if !m.vfbGiven { m.vfb = -1.0; }
    if !m.ua1Given { m.ua1 = 1e-9; }
    if !m.ubGiven { m.ub = 1e-19; }
    if !m.ub1Given { m.ub1 = -1e-18; }
    if !m.udGiven { m.ud = 0.0; }
    if !m.ud1Given { m.ud1 = 0.0; }
    if !m.upGiven { m.up = 0.0; }
    if !m.lpGiven { m.lp = 1e-8; }
    if !m.uteGiven { m.ute = -1.5; }
    if !m.ucsteGiven { m.ucste = -0.004_775; }
    if !m.voffGiven { m.voff = -0.08; }
    if !m.vofflGiven { m.voffl = 0.0; }
    if !m.voffcvlGiven { m.voffcvl = 0.0; }
    if !m.minvGiven { m.minv = 0.0; }
    if !m.minvcvGiven { m.minvcv = 0.0; }
    if !m.fproutGiven { m.fprout = 0.0; }
    if !m.pditsGiven { m.pdits = 0.0; }
    if !m.pditsdGiven { m.pditsd = 0.0; }
    if !m.pditslGiven { m.pditsl = 0.0; }
    if !m.deltaGiven { m.delta = 0.01; }
    if !m.rdswminGiven { m.rdswmin = 0.0; }
    if !m.rdwminGiven { m.rdwmin = 0.0; }
    if !m.rswminGiven { m.rswmin = 0.0; }
    if !m.rdswGiven { m.rdsw = 200.0; }
    if !m.rdwGiven { m.rdw = 100.0; }
    if !m.rswGiven { m.rsw = 100.0; }
    if !m.prwgGiven { m.prwg = 1.0; }
    if !m.prwbGiven { m.prwb = 0.0; }
    if !m.prtGiven { m.prt = 0.0; }
    if !m.eta0Given { m.eta0 = 0.08; }
    if !m.etabGiven { m.etab = -0.07; }
    if !m.pclmGiven { m.pclm = 1.3; }
    if !m.pdibl1Given { m.pdibl1 = 0.39; }
    if !m.pdibl2Given { m.pdibl2 = 0.0086; }
    if !m.pdiblbGiven { m.pdiblb = 0.0; }
    if !m.pscbe1Given { m.pscbe1 = 424_000_000.0; }
    if !m.pscbe2Given { m.pscbe2 = 1e-5; }
    if !m.pvagGiven { m.pvag = 0.0; }
    if !m.wrGiven { m.wr = 1.0; }
    if !m.dwgGiven { m.dwg = 0.0; }
    if !m.dwbGiven { m.dwb = 0.0; }
    if !m.b0Given { m.b0 = 0.0; }
    if !m.b1Given { m.b1 = 0.0; }
    if !m.alpha0Given { m.alpha0 = 0.0; }
    if !m.alpha1Given { m.alpha1 = 0.0; }
    if !m.beta0Given { m.beta0 = 0.0; }
    if !m.gidlModGiven { m.gidlMod = 0; }
    if !m.agidlGiven { m.agidl = 0.0; }
    if !m.bgidlGiven { m.bgidl = 2_300_000_000.0; }
    if !m.cgidlGiven { m.cgidl = 0.5; }
    if !m.egidlGiven { m.egidl = 0.8; }
    if !m.rgidlGiven { m.rgidl = 1.0; }
    if !m.kgidlGiven { m.kgidl = 0.0; }
    if !m.fgidlGiven { m.fgidl = 1.0; }
    if !m.aigbaccGiven { m.aigbacc = 0.0136; }
    if !m.bigbaccGiven { m.bigbacc = 0.00171; }
    if !m.cigbaccGiven { m.cigbacc = 0.075; }
    if !m.aigbinvGiven { m.aigbinv = 0.0111; }
    if !m.bigbinvGiven { m.bigbinv = 0.000_949; }
    if !m.cigbinvGiven { m.cigbinv = 0.006; }
    if !m.nigcGiven { m.nigc = 1.0; }
    if !m.nigbinvGiven { m.nigbinv = 3.0; }
    if !m.nigbaccGiven { m.nigbacc = 1.0; }
    if !m.ntoxGiven { m.ntox = 1.0; }
    if !m.eigbinvGiven { m.eigbinv = 1.1; }
    if !m.pigcdGiven { m.pigcd = 1.0; }
    if !m.poxedgeGiven { m.poxedge = 1.0; }
    if !m.xrcrg1Given { m.xrcrg1 = 12.0; }
    if !m.xrcrg2Given { m.xrcrg2 = 1.0; }
    if !m.ijthsfwdGiven { m.ijthsfwd = 0.1; }
    if !m.ijthsrevGiven { m.ijthsrev = 0.1; }
    if !m.tnoiaGiven { m.tnoia = 1.5; }
    if !m.tnoibGiven { m.tnoib = 3.5; }
    if !m.tnoicGiven { m.tnoic = 0.0; }
    if !m.rnoiaGiven { m.rnoia = 0.577; }
    if !m.rnoibGiven { m.rnoib = 0.5164; }
    if !m.gidlclampGiven { m.gidlclamp = -1e-5; }
    if !m.idovvdscGiven { m.idovvdsc = 1e-9; }
    if !m.rnoicGiven { m.rnoic = 0.395; }
    if !m.ntnoiGiven { m.ntnoi = 1.0; }
    if !m.lambdaGiven { m.lambda = 0.0; }
    if !m.vtlGiven { m.vtl = 200_000.0; }
    if !m.xnGiven { m.xn = 3.0; }
    if !m.lcGiven { m.lc = 5e-9; }
    if !m.vfbsdoffGiven { m.vfbsdoff = 0.0; }
    if !m.tvfbsdoffGiven { m.tvfbsdoff = 0.0; }
    if !m.tvoffGiven { m.tvoff = 0.0; }
    if !m.tnfactorGiven { m.tnfactor = 0.0; }
    if !m.teta0Given { m.teta0 = 0.0; }
    if !m.tvoffcvGiven { m.tvoffcv = 0.0; }
    if !m.lintnoiGiven { m.lintnoi = 0.0; }
    if !m.xjbvsGiven { m.xjbvs = 1.0; }
    if !m.bvsGiven { m.bvs = 10.0; }
    if !m.gbminGiven { m.gbmin = 1e-12; }
    if !m.rbdbGiven { m.rbdb = 50.0; }
    if !m.rbpbGiven { m.rbpb = 50.0; }
    if !m.rbsbGiven { m.rbsb = 50.0; }
    if !m.rbpsGiven { m.rbps = 50.0; }
    if !m.rbpdGiven { m.rbpd = 50.0; }
    if !m.rbps0Given { m.rbps0 = 50.0; }
    if !m.rbpslGiven { m.rbpsl = 0.0; }
    if !m.rbpswGiven { m.rbpsw = 0.0; }
    if !m.rbpsnfGiven { m.rbpsnf = 0.0; }
    if !m.rbpd0Given { m.rbpd0 = 50.0; }
    if !m.rbpdlGiven { m.rbpdl = 0.0; }
    if !m.rbpdwGiven { m.rbpdw = 0.0; }
    if !m.rbpdnfGiven { m.rbpdnf = 0.0; }
    if !m.rbpbx0Given { m.rbpbx0 = 100.0; }
    if !m.rbpbxlGiven { m.rbpbxl = 0.0; }
    if !m.rbpbxwGiven { m.rbpbxw = 0.0; }
    if !m.rbpbxnfGiven { m.rbpbxnf = 0.0; }
    if !m.rbpby0Given { m.rbpby0 = 100.0; }
    if !m.rbpbylGiven { m.rbpbyl = 0.0; }
    if !m.rbpbywGiven { m.rbpbyw = 0.0; }
    if !m.rbpbynfGiven { m.rbpbynf = 0.0; }
    if !m.rbsbx0Given { m.rbsbx0 = 100.0; }
    if !m.rbsby0Given { m.rbsby0 = 100.0; }
    if !m.rbdbx0Given { m.rbdbx0 = 100.0; }
    if !m.rbdby0Given { m.rbdby0 = 100.0; }
    if !m.rbsdbxlGiven { m.rbsdbxl = 0.0; }
    if !m.rbsdbxwGiven { m.rbsdbxw = 0.0; }
    if !m.rbsdbxnfGiven { m.rbsdbxnf = 0.0; }
    if !m.rbsdbylGiven { m.rbsdbyl = 0.0; }
    if !m.rbsdbywGiven { m.rbsdbyw = 0.0; }
    if !m.rbsdbynfGiven { m.rbsdbynf = 0.0; }
    if !m.cgslGiven { m.cgsl = 0.0; }
    if !m.cgdlGiven { m.cgdl = 0.0; }
    if !m.ckappasGiven { m.ckappas = 0.6; }
    if !m.clcGiven { m.clc = 1e-7; }
    if !m.cleGiven { m.cle = 0.6; }
    if !m.vfbcvGiven { m.vfbcv = -1.0; }
    if !m.acdeGiven { m.acde = 1.0; }
    if !m.moinGiven { m.moin = 15.0; }
    if !m.noffGiven { m.noff = 1.0; }
    if !m.voffcvGiven { m.voffcv = 0.0; }
    if !m.dmcgGiven { m.dmcg = 0.0; }
    if !m.dmdgGiven { m.dmdg = 0.0; }
    if !m.dmcgtGiven { m.dmcgt = 0.0; }
    if !m.xgwGiven { m.xgw = 0.0; }
    if !m.xglGiven { m.xgl = 0.0; }
    if !m.rshgGiven { m.rshg = 0.1; }
    if !m.ngconGiven { m.ngcon = 1.0; }
    if !m.tcjGiven { m.tcj = 0.0; }
    if !m.tpbGiven { m.tpb = 0.0; }
    if !m.tcjswGiven { m.tcjsw = 0.0; }
    if !m.tpbswGiven { m.tpbsw = 0.0; }
    if !m.tcjswgGiven { m.tcjswg = 0.0; }
    if !m.tpbswgGiven { m.tpbswg = 0.0; }
    if !m.lcdscGiven { m.lcdsc = 0.0; }
    if !m.lcdscbGiven { m.lcdscb = 0.0; }
    if !m.lcdscdGiven { m.lcdscd = 0.0; }
    if !m.lcitGiven { m.lcit = 0.0; }
    if !m.lnfactorGiven { m.lnfactor = 0.0; }
    if !m.lxjGiven { m.lxj = 0.0; }
    if !m.lvsatGiven { m.lvsat = 0.0; }
    if !m.latGiven { m.lat = 0.0; }
    if !m.la0Given { m.la0 = 0.0; }
    if !m.lagsGiven { m.lags = 0.0; }
    if !m.la1Given { m.la1 = 0.0; }
    if !m.la2Given { m.la2 = 0.0; }
    if !m.lketaGiven { m.lketa = 0.0; }
    if !m.lnsubGiven { m.lnsub = 0.0; }
    if !m.lndepGiven { m.lndep = 0.0; }
    if !m.lnsdGiven { m.lnsd = 0.0; }
    if !m.lphinGiven { m.lphin = 0.0; }
    if !m.lngateGiven { m.lngate = 0.0; }
    if !m.lvbmGiven { m.lvbm = 0.0; }
    if !m.lxtGiven { m.lxt = 0.0; }
    if !m.lk1Given { m.lk1 = 0.0; }
    if !m.lkt1Given { m.lkt1 = 0.0; }
    if !m.lkt1lGiven { m.lkt1l = 0.0; }
    if !m.lkt2Given { m.lkt2 = 0.0; }
    if !m.lk2Given { m.lk2 = 0.0; }
    if !m.lk3Given { m.lk3 = 0.0; }
    if !m.lk3bGiven { m.lk3b = 0.0; }
    if !m.lw0Given { m.lw0 = 0.0; }
    if !m.llpe0Given { m.llpe0 = 0.0; }
    if !m.llpebGiven { m.llpeb = 0.0; }
    if !m.ldvtp0Given { m.ldvtp0 = 0.0; }
    if !m.ldvtp1Given { m.ldvtp1 = 0.0; }
    if !m.ldvtp2Given { m.ldvtp2 = 0.0; }
    if !m.ldvtp3Given { m.ldvtp3 = 0.0; }
    if !m.ldvtp4Given { m.ldvtp4 = 0.0; }
    if !m.ldvtp5Given { m.ldvtp5 = 0.0; }
    if !m.ldvt0Given { m.ldvt0 = 0.0; }
    if !m.ldvt1Given { m.ldvt1 = 0.0; }
    if !m.ldvt2Given { m.ldvt2 = 0.0; }
    if !m.ldvt0wGiven { m.ldvt0w = 0.0; }
    if !m.ldvt1wGiven { m.ldvt1w = 0.0; }
    if !m.ldvt2wGiven { m.ldvt2w = 0.0; }
    if !m.ldroutGiven { m.ldrout = 0.0; }
    if !m.ldsubGiven { m.ldsub = 0.0; }
    if !m.lvth0Given { m.lvth0 = 0.0; }
    if !m.luaGiven { m.lua = 0.0; }
    if !m.lua1Given { m.lua1 = 0.0; }
    if !m.lubGiven { m.lub = 0.0; }
    if !m.lub1Given { m.lub1 = 0.0; }
    if !m.lucGiven { m.luc = 0.0; }
    if !m.luc1Given { m.luc1 = 0.0; }
    if !m.ludGiven { m.lud = 0.0; }
    if !m.lud1Given { m.lud1 = 0.0; }
    if !m.lupGiven { m.lup = 0.0; }
    if !m.llpGiven { m.llp = 0.0; }
    if !m.lu0Given { m.lu0 = 0.0; }
    if !m.luteGiven { m.lute = 0.0; }
    if !m.lucsteGiven { m.lucste = 0.0; }
    if !m.lvoffGiven { m.lvoff = 0.0; }
    if !m.lminvGiven { m.lminv = 0.0; }
    if !m.lminvcvGiven { m.lminvcv = 0.0; }
    if !m.lfproutGiven { m.lfprout = 0.0; }
    if !m.lpditsGiven { m.lpdits = 0.0; }
    if !m.lpditsdGiven { m.lpditsd = 0.0; }
    if !m.ldeltaGiven { m.ldelta = 0.0; }
    if !m.lrdswGiven { m.lrdsw = 0.0; }
    if !m.lrdwGiven { m.lrdw = 0.0; }
    if !m.lrswGiven { m.lrsw = 0.0; }
    if !m.lprwbGiven { m.lprwb = 0.0; }
    if !m.lprwgGiven { m.lprwg = 0.0; }
    if !m.lprtGiven { m.lprt = 0.0; }
    if !m.leta0Given { m.leta0 = 0.0; }
    if !m.letabGiven { m.letab = -0.0; }
    if !m.lpclmGiven { m.lpclm = 0.0; }
    if !m.lpdibl1Given { m.lpdibl1 = 0.0; }
    if !m.lpdibl2Given { m.lpdibl2 = 0.0; }
    if !m.lpdiblbGiven { m.lpdiblb = 0.0; }
    if !m.lpscbe1Given { m.lpscbe1 = 0.0; }
    if !m.lpscbe2Given { m.lpscbe2 = 0.0; }
    if !m.lpvagGiven { m.lpvag = 0.0; }
    if !m.lwrGiven { m.lwr = 0.0; }
    if !m.ldwgGiven { m.ldwg = 0.0; }
    if !m.ldwbGiven { m.ldwb = 0.0; }
    if !m.lb0Given { m.lb0 = 0.0; }
    if !m.lb1Given { m.lb1 = 0.0; }
    if !m.lalpha0Given { m.lalpha0 = 0.0; }
    if !m.lalpha1Given { m.lalpha1 = 0.0; }
    if !m.lbeta0Given { m.lbeta0 = 0.0; }
    if !m.lagidlGiven { m.lagidl = 0.0; }
    if !m.lbgidlGiven { m.lbgidl = 0.0; }
    if !m.lcgidlGiven { m.lcgidl = 0.0; }
    if !m.legidlGiven { m.legidl = 0.0; }
    if !m.lrgidlGiven { m.lrgidl = 0.0; }
    if !m.lkgidlGiven { m.lkgidl = 0.0; }
    if !m.lfgidlGiven { m.lfgidl = 0.0; }
    if !m.laigcGiven { m.laigc = 0.0; }
    if !m.lbigcGiven { m.lbigc = 0.0; }
    if !m.lcigcGiven { m.lcigc = 0.0; }
    if !m.laigbaccGiven { m.laigbacc = 0.0; }
    if !m.lbigbaccGiven { m.lbigbacc = 0.0; }
    if !m.lcigbaccGiven { m.lcigbacc = 0.0; }
    if !m.laigbinvGiven { m.laigbinv = 0.0; }
    if !m.lbigbinvGiven { m.lbigbinv = 0.0; }
    if !m.lcigbinvGiven { m.lcigbinv = 0.0; }
    if !m.lnigcGiven { m.lnigc = 0.0; }
    if !m.lnigbinvGiven { m.lnigbinv = 0.0; }
    if !m.lnigbaccGiven { m.lnigbacc = 0.0; }
    if !m.lntoxGiven { m.lntox = 0.0; }
    if !m.leigbinvGiven { m.leigbinv = 0.0; }
    if !m.lpigcdGiven { m.lpigcd = 0.0; }
    if !m.lpoxedgeGiven { m.lpoxedge = 0.0; }
    if !m.lxrcrg1Given { m.lxrcrg1 = 0.0; }
    if !m.lxrcrg2Given { m.lxrcrg2 = 0.0; }
    if !m.leuGiven { m.leu = 0.0; }
    if !m.lucsGiven { m.lucs = 0.0; }
    if !m.lvfbGiven { m.lvfb = 0.0; }
    if !m.llambdaGiven { m.llambda = 0.0; }
    if !m.lvtlGiven { m.lvtl = 0.0; }
    if !m.lxnGiven { m.lxn = 0.0; }
    if !m.lvfbsdoffGiven { m.lvfbsdoff = 0.0; }
    if !m.ltvfbsdoffGiven { m.ltvfbsdoff = 0.0; }
    if !m.ltvoffGiven { m.ltvoff = 0.0; }
    if !m.ltnfactorGiven { m.ltnfactor = 0.0; }
    if !m.lteta0Given { m.lteta0 = 0.0; }
    if !m.ltvoffcvGiven { m.ltvoffcv = 0.0; }
    if !m.lcgslGiven { m.lcgsl = 0.0; }
    if !m.lcgdlGiven { m.lcgdl = 0.0; }
    if !m.lckappasGiven { m.lckappas = 0.0; }
    if !m.lckappadGiven { m.lckappad = 0.0; }
    if !m.lclcGiven { m.lclc = 0.0; }
    if !m.lcleGiven { m.lcle = 0.0; }
    if !m.lcfGiven { m.lcf = 0.0; }
    if !m.lvfbcvGiven { m.lvfbcv = 0.0; }
    if !m.lacdeGiven { m.lacde = 0.0; }
    if !m.lmoinGiven { m.lmoin = 0.0; }
    if !m.lnoffGiven { m.lnoff = 0.0; }
    if !m.lvoffcvGiven { m.lvoffcv = 0.0; }
    if !m.wcdscGiven { m.wcdsc = 0.0; }
    if !m.wcdscbGiven { m.wcdscb = 0.0; }
    if !m.wcdscdGiven { m.wcdscd = 0.0; }
    if !m.wcitGiven { m.wcit = 0.0; }
    if !m.wnfactorGiven { m.wnfactor = 0.0; }
    if !m.wxjGiven { m.wxj = 0.0; }
    if !m.wvsatGiven { m.wvsat = 0.0; }
    if !m.watGiven { m.wat = 0.0; }
    if !m.wa0Given { m.wa0 = 0.0; }
    if !m.wagsGiven { m.wags = 0.0; }
    if !m.wa1Given { m.wa1 = 0.0; }
    if !m.wa2Given { m.wa2 = 0.0; }
    if !m.wketaGiven { m.wketa = 0.0; }
    if !m.wnsubGiven { m.wnsub = 0.0; }
    if !m.wndepGiven { m.wndep = 0.0; }
    if !m.wnsdGiven { m.wnsd = 0.0; }
    if !m.wphinGiven { m.wphin = 0.0; }
    if !m.wngateGiven { m.wngate = 0.0; }
    if !m.wvbmGiven { m.wvbm = 0.0; }
    if !m.wxtGiven { m.wxt = 0.0; }
    if !m.wk1Given { m.wk1 = 0.0; }
    if !m.wkt1Given { m.wkt1 = 0.0; }
    if !m.wkt1lGiven { m.wkt1l = 0.0; }
    if !m.wkt2Given { m.wkt2 = 0.0; }
    if !m.wk2Given { m.wk2 = 0.0; }
    if !m.wk3Given { m.wk3 = 0.0; }
    if !m.wk3bGiven { m.wk3b = 0.0; }
    if !m.ww0Given { m.ww0 = 0.0; }
    if !m.wlpe0Given { m.wlpe0 = 0.0; }
    if !m.wlpebGiven { m.wlpeb = 0.0; }
    if !m.wdvtp0Given { m.wdvtp0 = 0.0; }
    if !m.wdvtp1Given { m.wdvtp1 = 0.0; }
    if !m.wdvtp2Given { m.wdvtp2 = 0.0; }
    if !m.wdvtp3Given { m.wdvtp3 = 0.0; }
    if !m.wdvtp4Given { m.wdvtp4 = 0.0; }
    if !m.wdvtp5Given { m.wdvtp5 = 0.0; }
    if !m.wdvt0Given { m.wdvt0 = 0.0; }
    if !m.wdvt1Given { m.wdvt1 = 0.0; }
    if !m.wdvt2Given { m.wdvt2 = 0.0; }
    if !m.wdvt0wGiven { m.wdvt0w = 0.0; }
    if !m.wdvt1wGiven { m.wdvt1w = 0.0; }
    if !m.wdvt2wGiven { m.wdvt2w = 0.0; }
    if !m.wdroutGiven { m.wdrout = 0.0; }
    if !m.wdsubGiven { m.wdsub = 0.0; }
    if !m.wvth0Given { m.wvth0 = 0.0; }
    if !m.wuaGiven { m.wua = 0.0; }
    if !m.wua1Given { m.wua1 = 0.0; }
    if !m.wubGiven { m.wub = 0.0; }
    if !m.wub1Given { m.wub1 = 0.0; }
    if !m.wucGiven { m.wuc = 0.0; }
    if !m.wuc1Given { m.wuc1 = 0.0; }
    if !m.wudGiven { m.wud = 0.0; }
    if !m.wud1Given { m.wud1 = 0.0; }
    if !m.wupGiven { m.wup = 0.0; }
    if !m.wlpGiven { m.wlp = 0.0; }
    if !m.wu0Given { m.wu0 = 0.0; }
    if !m.wuteGiven { m.wute = 0.0; }
    if !m.wucsteGiven { m.wucste = 0.0; }
    if !m.wvoffGiven { m.wvoff = 0.0; }
    if !m.wminvGiven { m.wminv = 0.0; }
    if !m.wminvcvGiven { m.wminvcv = 0.0; }
    if !m.wfproutGiven { m.wfprout = 0.0; }
    if !m.wpditsGiven { m.wpdits = 0.0; }
    if !m.wpditsdGiven { m.wpditsd = 0.0; }
    if !m.wdeltaGiven { m.wdelta = 0.0; }
    if !m.wrdswGiven { m.wrdsw = 0.0; }
    if !m.wrdwGiven { m.wrdw = 0.0; }
    if !m.wrswGiven { m.wrsw = 0.0; }
    if !m.wprwbGiven { m.wprwb = 0.0; }
    if !m.wprwgGiven { m.wprwg = 0.0; }
    if !m.wprtGiven { m.wprt = 0.0; }
    if !m.weta0Given { m.weta0 = 0.0; }
    if !m.wetabGiven { m.wetab = 0.0; }
    if !m.wpclmGiven { m.wpclm = 0.0; }
    if !m.wpdibl1Given { m.wpdibl1 = 0.0; }
    if !m.wpdibl2Given { m.wpdibl2 = 0.0; }
    if !m.wpdiblbGiven { m.wpdiblb = 0.0; }
    if !m.wpscbe1Given { m.wpscbe1 = 0.0; }
    if !m.wpscbe2Given { m.wpscbe2 = 0.0; }
    if !m.wpvagGiven { m.wpvag = 0.0; }
    if !m.wwrGiven { m.wwr = 0.0; }
    if !m.wdwgGiven { m.wdwg = 0.0; }
    if !m.wdwbGiven { m.wdwb = 0.0; }
    if !m.wb0Given { m.wb0 = 0.0; }
    if !m.wb1Given { m.wb1 = 0.0; }
    if !m.walpha0Given { m.walpha0 = 0.0; }
    if !m.walpha1Given { m.walpha1 = 0.0; }
    if !m.wbeta0Given { m.wbeta0 = 0.0; }
    if !m.wagidlGiven { m.wagidl = 0.0; }
    if !m.wbgidlGiven { m.wbgidl = 0.0; }
    if !m.wcgidlGiven { m.wcgidl = 0.0; }
    if !m.wegidlGiven { m.wegidl = 0.0; }
    if !m.wrgidlGiven { m.wrgidl = 0.0; }
    if !m.wkgidlGiven { m.wkgidl = 0.0; }
    if !m.wfgidlGiven { m.wfgidl = 0.0; }
    if !m.waigcGiven { m.waigc = 0.0; }
    if !m.wbigcGiven { m.wbigc = 0.0; }
    if !m.wcigcGiven { m.wcigc = 0.0; }
    if !m.waigbaccGiven { m.waigbacc = 0.0; }
    if !m.wbigbaccGiven { m.wbigbacc = 0.0; }
    if !m.wcigbaccGiven { m.wcigbacc = 0.0; }
    if !m.waigbinvGiven { m.waigbinv = 0.0; }
    if !m.wbigbinvGiven { m.wbigbinv = 0.0; }
    if !m.wcigbinvGiven { m.wcigbinv = 0.0; }
    if !m.wnigcGiven { m.wnigc = 0.0; }
    if !m.wnigbinvGiven { m.wnigbinv = 0.0; }
    if !m.wnigbaccGiven { m.wnigbacc = 0.0; }
    if !m.wntoxGiven { m.wntox = 0.0; }
    if !m.weigbinvGiven { m.weigbinv = 0.0; }
    if !m.wpigcdGiven { m.wpigcd = 0.0; }
    if !m.wpoxedgeGiven { m.wpoxedge = 0.0; }
    if !m.wxrcrg1Given { m.wxrcrg1 = 0.0; }
    if !m.wxrcrg2Given { m.wxrcrg2 = 0.0; }
    if !m.weuGiven { m.weu = 0.0; }
    if !m.wucsGiven { m.wucs = 0.0; }
    if !m.wvfbGiven { m.wvfb = 0.0; }
    if !m.wlambdaGiven { m.wlambda = 0.0; }
    if !m.wvtlGiven { m.wvtl = 0.0; }
    if !m.wxnGiven { m.wxn = 0.0; }
    if !m.wvfbsdoffGiven { m.wvfbsdoff = 0.0; }
    if !m.wtvfbsdoffGiven { m.wtvfbsdoff = 0.0; }
    if !m.wtvoffGiven { m.wtvoff = 0.0; }
    if !m.wtnfactorGiven { m.wtnfactor = 0.0; }
    if !m.wteta0Given { m.wteta0 = 0.0; }
    if !m.wtvoffcvGiven { m.wtvoffcv = 0.0; }
    if !m.wcgslGiven { m.wcgsl = 0.0; }
    if !m.wcgdlGiven { m.wcgdl = 0.0; }
    if !m.wckappasGiven { m.wckappas = 0.0; }
    if !m.wckappadGiven { m.wckappad = 0.0; }
    if !m.wcfGiven { m.wcf = 0.0; }
    if !m.wclcGiven { m.wclc = 0.0; }
    if !m.wcleGiven { m.wcle = 0.0; }
    if !m.wvfbcvGiven { m.wvfbcv = 0.0; }
    if !m.wacdeGiven { m.wacde = 0.0; }
    if !m.wmoinGiven { m.wmoin = 0.0; }
    if !m.wnoffGiven { m.wnoff = 0.0; }
    if !m.wvoffcvGiven { m.wvoffcv = 0.0; }
    if !m.pcdscGiven { m.pcdsc = 0.0; }
    if !m.pcdscbGiven { m.pcdscb = 0.0; }
    if !m.pcdscdGiven { m.pcdscd = 0.0; }
    if !m.pcitGiven { m.pcit = 0.0; }
    if !m.pnfactorGiven { m.pnfactor = 0.0; }
    if !m.pxjGiven { m.pxj = 0.0; }
    if !m.pvsatGiven { m.pvsat = 0.0; }
    if !m.patGiven { m.pat = 0.0; }
    if !m.pa0Given { m.pa0 = 0.0; }
    if !m.pagsGiven { m.pags = 0.0; }
    if !m.pa1Given { m.pa1 = 0.0; }
    if !m.pa2Given { m.pa2 = 0.0; }
    if !m.pketaGiven { m.pketa = 0.0; }
    if !m.pnsubGiven { m.pnsub = 0.0; }
    if !m.pndepGiven { m.pndep = 0.0; }
    if !m.pnsdGiven { m.pnsd = 0.0; }
    if !m.pphinGiven { m.pphin = 0.0; }
    if !m.pngateGiven { m.pngate = 0.0; }
    if !m.pvbmGiven { m.pvbm = 0.0; }
    if !m.pxtGiven { m.pxt = 0.0; }
    if !m.pk1Given { m.pk1 = 0.0; }
    if !m.pkt1Given { m.pkt1 = 0.0; }
    if !m.pkt1lGiven { m.pkt1l = 0.0; }
    if !m.pk2Given { m.pk2 = 0.0; }
    if !m.pkt2Given { m.pkt2 = 0.0; }
    if !m.pk3Given { m.pk3 = 0.0; }
    if !m.pk3bGiven { m.pk3b = 0.0; }
    if !m.pw0Given { m.pw0 = 0.0; }
    if !m.plpe0Given { m.plpe0 = 0.0; }
    if !m.plpebGiven { m.plpeb = 0.0; }
    if !m.pdvtp0Given { m.pdvtp0 = 0.0; }
    if !m.pdvtp1Given { m.pdvtp1 = 0.0; }
    if !m.pdvtp2Given { m.pdvtp2 = 0.0; }
    if !m.pdvtp3Given { m.pdvtp3 = 0.0; }
    if !m.pdvtp4Given { m.pdvtp4 = 0.0; }
    if !m.pdvtp5Given { m.pdvtp5 = 0.0; }
    if !m.pdvt0Given { m.pdvt0 = 0.0; }
    if !m.pdvt1Given { m.pdvt1 = 0.0; }
    if !m.pdvt2Given { m.pdvt2 = 0.0; }
    if !m.pdvt0wGiven { m.pdvt0w = 0.0; }
    if !m.pdvt1wGiven { m.pdvt1w = 0.0; }
    if !m.pdvt2wGiven { m.pdvt2w = 0.0; }
    if !m.pdroutGiven { m.pdrout = 0.0; }
    if !m.pdsubGiven { m.pdsub = 0.0; }
    if !m.pvth0Given { m.pvth0 = 0.0; }
    if !m.puaGiven { m.pua = 0.0; }
    if !m.pua1Given { m.pua1 = 0.0; }
    if !m.pubGiven { m.r#pub = 0.0; }
    if !m.pub1Given { m.pub1 = 0.0; }
    if !m.pucGiven { m.puc = 0.0; }
    if !m.puc1Given { m.puc1 = 0.0; }
    if !m.pudGiven { m.pud = 0.0; }
    if !m.pud1Given { m.pud1 = 0.0; }
    if !m.pupGiven { m.pup = 0.0; }
    if !m.plpGiven { m.plp = 0.0; }
    if !m.pu0Given { m.pu0 = 0.0; }
    if !m.puteGiven { m.pute = 0.0; }
    if !m.pucsteGiven { m.pucste = 0.0; }
    if !m.pvoffGiven { m.pvoff = 0.0; }
    if !m.pminvGiven { m.pminv = 0.0; }
    if !m.pminvcvGiven { m.pminvcv = 0.0; }
    if !m.pfproutGiven { m.pfprout = 0.0; }
    if !m.ppditsGiven { m.ppdits = 0.0; }
    if !m.ppditsdGiven { m.ppditsd = 0.0; }
    if !m.pdeltaGiven { m.pdelta = 0.0; }
    if !m.prdswGiven { m.prdsw = 0.0; }
    if !m.prdwGiven { m.prdw = 0.0; }
    if !m.prswGiven { m.prsw = 0.0; }
    if !m.pprwbGiven { m.pprwb = 0.0; }
    if !m.pprwgGiven { m.pprwg = 0.0; }
    if !m.pprtGiven { m.pprt = 0.0; }
    if !m.peta0Given { m.peta0 = 0.0; }
    if !m.petabGiven { m.petab = 0.0; }
    if !m.ppclmGiven { m.ppclm = 0.0; }
    if !m.ppdibl1Given { m.ppdibl1 = 0.0; }
    if !m.ppdibl2Given { m.ppdibl2 = 0.0; }
    if !m.ppdiblbGiven { m.ppdiblb = 0.0; }
    if !m.ppscbe1Given { m.ppscbe1 = 0.0; }
    if !m.ppscbe2Given { m.ppscbe2 = 0.0; }
    if !m.ppvagGiven { m.ppvag = 0.0; }
    if !m.pwrGiven { m.pwr = 0.0; }
    if !m.pdwgGiven { m.pdwg = 0.0; }
    if !m.pdwbGiven { m.pdwb = 0.0; }
    if !m.pb0Given { m.pb0 = 0.0; }
    if !m.pb1Given { m.pb1 = 0.0; }
    if !m.palpha0Given { m.palpha0 = 0.0; }
    if !m.palpha1Given { m.palpha1 = 0.0; }
    if !m.pbeta0Given { m.pbeta0 = 0.0; }
    if !m.pagidlGiven { m.pagidl = 0.0; }
    if !m.pbgidlGiven { m.pbgidl = 0.0; }
    if !m.pcgidlGiven { m.pcgidl = 0.0; }
    if !m.pegidlGiven { m.pegidl = 0.0; }
    if !m.prgidlGiven { m.prgidl = 0.0; }
    if !m.pkgidlGiven { m.pkgidl = 0.0; }
    if !m.pfgidlGiven { m.pfgidl = 0.0; }
    if !m.paigcGiven { m.paigc = 0.0; }
    if !m.pbigcGiven { m.pbigc = 0.0; }
    if !m.pcigcGiven { m.pcigc = 0.0; }
    if !m.paigbaccGiven { m.paigbacc = 0.0; }
    if !m.pbigbaccGiven { m.pbigbacc = 0.0; }
    if !m.pcigbaccGiven { m.pcigbacc = 0.0; }
    if !m.paigbinvGiven { m.paigbinv = 0.0; }
    if !m.pbigbinvGiven { m.pbigbinv = 0.0; }
    if !m.pcigbinvGiven { m.pcigbinv = 0.0; }
    if !m.pnigcGiven { m.pnigc = 0.0; }
    if !m.pnigbinvGiven { m.pnigbinv = 0.0; }
    if !m.pnigbaccGiven { m.pnigbacc = 0.0; }
    if !m.pntoxGiven { m.pntox = 0.0; }
    if !m.peigbinvGiven { m.peigbinv = 0.0; }
    if !m.ppigcdGiven { m.ppigcd = 0.0; }
    if !m.ppoxedgeGiven { m.ppoxedge = 0.0; }
    if !m.pxrcrg1Given { m.pxrcrg1 = 0.0; }
    if !m.pxrcrg2Given { m.pxrcrg2 = 0.0; }
    if !m.peuGiven { m.peu = 0.0; }
    if !m.pucsGiven { m.pucs = 0.0; }
    if !m.pvfbGiven { m.pvfb = 0.0; }
    if !m.plambdaGiven { m.plambda = 0.0; }
    if !m.pvtlGiven { m.pvtl = 0.0; }
    if !m.pxnGiven { m.pxn = 0.0; }
    if !m.pvfbsdoffGiven { m.pvfbsdoff = 0.0; }
    if !m.ptvfbsdoffGiven { m.ptvfbsdoff = 0.0; }
    if !m.ptvoffGiven { m.ptvoff = 0.0; }
    if !m.ptnfactorGiven { m.ptnfactor = 0.0; }
    if !m.pteta0Given { m.pteta0 = 0.0; }
    if !m.ptvoffcvGiven { m.ptvoffcv = 0.0; }
    if !m.pcgslGiven { m.pcgsl = 0.0; }
    if !m.pcgdlGiven { m.pcgdl = 0.0; }
    if !m.pckappasGiven { m.pckappas = 0.0; }
    if !m.pckappadGiven { m.pckappad = 0.0; }
    if !m.pcfGiven { m.pcf = 0.0; }
    if !m.pclcGiven { m.pclc = 0.0; }
    if !m.pcleGiven { m.pcle = 0.0; }
    if !m.pvfbcvGiven { m.pvfbcv = 0.0; }
    if !m.pacdeGiven { m.pacde = 0.0; }
    if !m.pmoinGiven { m.pmoin = 0.0; }
    if !m.pnoffGiven { m.pnoff = 0.0; }
    if !m.pvoffcvGiven { m.pvoffcv = 0.0; }
    if !m.gamma1Given { m.gamma1 = 0.0; }
    if !m.lgamma1Given { m.lgamma1 = 0.0; }
    if !m.wgamma1Given { m.wgamma1 = 0.0; }
    if !m.pgamma1Given { m.pgamma1 = 0.0; }
    if !m.gamma2Given { m.gamma2 = 0.0; }
    if !m.lgamma2Given { m.lgamma2 = 0.0; }
    if !m.wgamma2Given { m.wgamma2 = 0.0; }
    if !m.pgamma2Given { m.pgamma2 = 0.0; }
    if !m.vbxGiven { m.vbx = 0.0; }
    if !m.lvbxGiven { m.lvbx = 0.0; }
    if !m.wvbxGiven { m.wvbx = 0.0; }
    if !m.pvbxGiven { m.pvbx = 0.0; }
    if !m.LintGiven { m.Lint = 0.0; }
    if !m.LlGiven { m.Ll = 0.0; }
    if !m.LlnGiven { m.Lln = 1.0; }
    if !m.LwGiven { m.Lw = 0.0; }
    if !m.LwnGiven { m.Lwn = 1.0; }
    if !m.LwlGiven { m.Lwl = 0.0; }
    if !m.LminGiven { m.Lmin = 0.0; }
    if !m.LmaxGiven { m.Lmax = 1.0; }
    if !m.WintGiven { m.Wint = 0.0; }
    if !m.WlGiven { m.Wl = 0.0; }
    if !m.WlnGiven { m.Wln = 1.0; }
    if !m.WwGiven { m.Ww = 0.0; }
    if !m.WwnGiven { m.Wwn = 1.0; }
    if !m.WwlGiven { m.Wwl = 0.0; }
    if !m.WminGiven { m.Wmin = 0.0; }
    if !m.WmaxGiven { m.Wmax = 1.0; }
    if !m.xlGiven { m.xl = 0.0; }
    if !m.xwGiven { m.xw = 0.0; }
    if !m.xpartGiven { m.xpart = 0.0; }
    if !m.sheetResistanceGiven { m.sheetResistance = 0.0; }
    if !m.SunitAreaJctCapGiven { m.SunitAreaJctCap = 0.0005; }
    if !m.SunitLengthSidewallJctCapGiven { m.SunitLengthSidewallJctCap = 5e-10; }
    if !m.SjctSatCurDensityGiven { m.SjctSatCurDensity = 0.0001; }
    if !m.SjctSidewallSatCurDensityGiven { m.SjctSidewallSatCurDensity = 0.0; }
    if !m.SjctGateSidewallSatCurDensityGiven { m.SjctGateSidewallSatCurDensity = 0.0; }
    if !m.SbulkJctPotentialGiven { m.SbulkJctPotential = 1.0; }
    if !m.SsidewallJctPotentialGiven { m.SsidewallJctPotential = 1.0; }
    if !m.SbulkJctBotGradingCoeffGiven { m.SbulkJctBotGradingCoeff = 0.5; }
    if !m.SbulkJctSideGradingCoeffGiven { m.SbulkJctSideGradingCoeff = 0.33; }
    if !m.SjctEmissionCoeffGiven { m.SjctEmissionCoeff = 1.0; }
    if !m.SjctTempExponentGiven { m.SjctTempExponent = 3.0; }
    if !m.jtssGiven { m.jtss = 0.0; }
    if !m.jtsswsGiven { m.jtssws = 0.0; }
    if !m.jtsswgsGiven { m.jtsswgs = 0.0; }
    if !m.jtweffGiven { m.jtweff = 0.0; }
    if !m.njtsGiven { m.njts = 20.0; }
    if !m.njtsswGiven { m.njtssw = 20.0; }
    if !m.njtsswgGiven { m.njtsswg = 20.0; }
    if !m.xtssGiven { m.xtss = 0.02; }
    if !m.xtsswsGiven { m.xtssws = 0.02; }
    if !m.xtsswgsGiven { m.xtsswgs = 0.02; }
    if !m.tnjtsGiven { m.tnjts = 0.0; }
    if !m.tnjtsswGiven { m.tnjtssw = 0.0; }
    if !m.tnjtsswgGiven { m.tnjtsswg = 0.0; }
    if !m.vtssGiven { m.vtss = 10.0; }
    if !m.vtsswsGiven { m.vtssws = 10.0; }
    if !m.vtsswgsGiven { m.vtsswgs = 10.0; }
    if !m.oxideTrapDensityCGiven { m.oxideTrapDensityC = 8_750_000_000.0; }
    if !m.emGiven { m.em = 41_000_000.0; }
    if !m.efGiven { m.ef = 1.0; }
    if !m.afGiven { m.af = 1.0; }
    if !m.kfGiven { m.kf = 0.0; }
    if !m.vgsMaxGiven { m.vgsMax = 1e99; }
    if !m.vgdMaxGiven { m.vgdMax = 1e99; }
    if !m.vgbMaxGiven { m.vgbMax = 1e99; }
    if !m.vdsMaxGiven { m.vdsMax = 1e99; }
    if !m.vbsMaxGiven { m.vbsMax = 1e99; }
    if !m.vbdMaxGiven { m.vbdMax = 1e99; }
    if !m.vgsrMaxGiven { m.vgsrMax = 1e99; }
    if !m.vgdrMaxGiven { m.vgdrMax = 1e99; }
    if !m.vgbrMaxGiven { m.vgbrMax = 1e99; }
    if !m.vbsrMaxGiven { m.vbsrMax = 1e99; }
    if !m.vbdrMaxGiven { m.vbdrMax = 1e99; }
    if !m.sarefGiven { m.saref = 1e-6; }
    if !m.sbrefGiven { m.sbref = 1e-6; }
    if !m.wlodGiven { m.wlod = 0.0; }
    if !m.ku0Given { m.ku0 = 0.0; }
    if !m.kvsatGiven { m.kvsat = 0.0; }
    if !m.kvth0Given { m.kvth0 = 0.0; }
    if !m.tku0Given { m.tku0 = 0.0; }
    if !m.llodku0Given { m.llodku0 = 0.0; }
    if !m.wlodku0Given { m.wlodku0 = 0.0; }
    if !m.llodvthGiven { m.llodvth = 0.0; }
    if !m.wlodvthGiven { m.wlodvth = 0.0; }
    if !m.lku0Given { m.lku0 = 0.0; }
    if !m.wku0Given { m.wku0 = 0.0; }
    if !m.pku0Given { m.pku0 = 0.0; }
    if !m.lkvth0Given { m.lkvth0 = 0.0; }
    if !m.wkvth0Given { m.wkvth0 = 0.0; }
    if !m.pkvth0Given { m.pkvth0 = 0.0; }
    if !m.stk2Given { m.stk2 = 0.0; }
    if !m.lodk2Given { m.lodk2 = 1.0; }
    if !m.steta0Given { m.steta0 = 0.0; }
    if !m.lodeta0Given { m.lodeta0 = 1.0; }
    if !m.webGiven { m.web = 0.0; }
    if !m.wecGiven { m.wec = 0.0; }
    if !m.kvth0weGiven { m.kvth0we = 0.0; }
    if !m.k2weGiven { m.k2we = 0.0; }
    if !m.ku0weGiven { m.ku0we = 0.0; }
    if !m.screfGiven { m.scref = 1e-6; }
    if !m.lkvth0weGiven { m.lkvth0we = 0.0; }
    if !m.lk2weGiven { m.lk2we = 0.0; }
    if !m.lku0weGiven { m.lku0we = 0.0; }
    if !m.wkvth0weGiven { m.wkvth0we = 0.0; }
    if !m.wk2weGiven { m.wk2we = 0.0; }
    if !m.wku0weGiven { m.wku0we = 0.0; }
    if !m.pkvth0weGiven { m.pkvth0we = 0.0; }
    if !m.pk2weGiven { m.pk2we = 0.0; }
    if !m.pku0weGiven { m.pku0we = 0.0; }
}
