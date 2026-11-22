//! `b4set.c` (`BSIM4setup`): model defaults, then instance defaults and the
//! internal nodes C creates. Matrix elements are listed in `elements.rs`.
//!
//! Independent defaults are generated from the C model loop. Selector
//! normalization and dependent defaults retain C's precedence, including
//! its quirk that any VERSION beginning `4.8` selects the 4.8.3 equations.

use super::fields::{Instance, Model};
use super::geo::rdseff_geo;
use super::host::node;
use super::{InputType, ModelParameters, parameters};
use crate::devices::schema::temperature_kelvin;
use crate::primitives::{Real, SpiceError, SpiceResult};

/// Model-only setup before instance topology or temperature processing.
#[derive(Clone)]
pub(super) struct PreparedModel {
    model: Model,
    warnings: Vec<String>,
}

/// Complete instance setup, with internal node requests in C creation order.
pub(super) struct PreparedSetup {
    pub(super) model: Model,
    pub(super) instance: Instance,
    pub(super) warnings: Vec<String>,
    pub(super) created: Vec<(i32, &'static str)>,
}

pub(super) fn prepare_model(mut model: Model, nominal_temperature: f64) -> PreparedModel {
    let mut warnings = Vec::new();
    setup_model(&mut model, nominal_temperature, &mut warnings);
    PreparedModel { model, warnings }
}

impl PreparedModel {
    pub(super) fn prepare_instance(
        mut self,
        mut instance: Instance,
        noise_analysis: bool,
    ) -> PreparedSetup {
        let created = setup_instance(
            &self.model,
            &mut instance,
            noise_analysis,
            &mut self.warnings,
        );
        PreparedSetup {
            model: self.model,
            instance,
            warnings: self.warnings,
            created,
        }
    }
}

/// Independent defaults do not depend on any other member; their generated
/// typed assignments run before the selector and dependent-default stages.
fn setup_model(m: &mut Model, nominal_temperature: f64, warnings: &mut Vec<String>) {
    super::defaults_data::apply_independent_defaults(m);
    normalize_model_selectors(m, warnings);
    apply_dependent_defaults(m, nominal_temperature);
}

/// Integer selectors in setup order. The distinction between absent
/// inputs and explicitly invalid values controls whether C warns.
struct Selector<'a> {
    name: &'static str,
    value: &'a mut i32,
    given: bool,
    fallback: i32,
    upper: i32,
}

#[derive(Clone, Copy)]
enum SelectorOrigin {
    Model,
    Instance,
}

impl SelectorOrigin {
    const fn label(self) -> &'static str {
        match self {
            Self::Model => "default value:",
            Self::Instance => "global value",
        }
    }
}

impl<'a> Selector<'a> {
    const fn new(
        name: &'static str,
        value: &'a mut i32,
        given: bool,
        fallback: i32,
        upper: i32,
    ) -> Self {
        Self {
            name,
            value,
            given,
            fallback,
            upper,
        }
    }

    fn normalize(self, origin: SelectorOrigin, warnings: &mut Vec<String>) {
        if !self.given {
            *self.value = self.fallback;
        } else if !(0..=self.upper).contains(self.value) {
            *self.value = self.fallback;
            warnings.push(format!(
                "Warning: {} has been set to its {} {}.",
                self.name,
                origin.label(),
                self.fallback
            ));
        }
    }
}

/// Normalize model selectors in upstream diagnostic order.
fn normalize_model_selectors(m: &mut Model, warnings: &mut Vec<String>) {
    if !m.typeGiven {
        m.r#type = 1;
    }
    for selector in [
        Selector::new("mobMod", &mut m.mobMod, m.mobModGiven, 0, 6),
        Selector::new("dioMod", &mut m.dioMod, m.dioModGiven, 1, 2),
        Selector::new("capMod", &mut m.capMod, m.capModGiven, 2, 2),
        Selector::new("rdsMod", &mut m.rdsMod, m.rdsModGiven, 0, 1),
        Selector::new("rbodyMod", &mut m.rbodyMod, m.rbodyModGiven, 0, 2),
        Selector::new("rgateMod", &mut m.rgateMod, m.rgateModGiven, 0, 3),
        Selector::new("perMod", &mut m.perMod, m.perModGiven, 1, 1),
        Selector::new("fnoiMod", &mut m.fnoiMod, m.fnoiModGiven, 1, 1),
        Selector::new("tnoiMod", &mut m.tnoiMod, m.tnoiModGiven, 0, 2),
        Selector::new("trnqsMod", &mut m.trnqsMod, m.trnqsModGiven, 0, 1),
        Selector::new("acnqsMod", &mut m.acnqsMod, m.acnqsModGiven, 0, 1),
        Selector::new("mtrlMod", &mut m.mtrlMod, m.mtrlModGiven, 0, 1),
        Selector::new(
            "mtrlCompatMod",
            &mut m.mtrlCompatMod,
            m.mtrlCompatModGiven,
            0,
            1,
        ),
        Selector::new("igcMod", &mut m.igcMod, m.igcModGiven, 0, 2),
        Selector::new("igbMod", &mut m.igbMod, m.igbModGiven, 0, 1),
        Selector::new("tempMod", &mut m.tempMod, m.tempModGiven, 0, 3),
    ] {
        selector.normalize(SelectorOrigin::Model, warnings);
    }
    if !m.versionGiven {
        m.version = String::from("4.8.3");
    }
    // C's first prefix test subsumes its subsequent 4.8.2/4.8.1/4.8.0 tests.
    if !m.version.starts_with("4.8") {
        warnings.push("Warning: unknown BSIM4 version. Working now with BSIM4.8.3.".into());
    }
    m.v48intVersion = 483;
    if !m.wpemodGiven {
        m.wpemod = 0.0;
    } else if m.wpemod != 0.0 && m.wpemod.to_bits() != 1.0_f64.to_bits() {
        m.wpemod = 0.0;
        warnings.push("Warning: wpemod has been set to its default value: 0.".into());
    }
}

/// The remaining expressions depend on polarity, selectors, nominal
/// temperature or other already initialized model fields. Keep their C order.
fn apply_dependent_defaults(m: &mut Model, nominal_temperature: f64) {
    electrical_dependent_defaults(m);
    length_dependent_defaults(m);
    width_dependent_defaults(m);
    cross_dependent_defaults(m);
    geometry_defaults(m, nominal_temperature);
    junction_and_noise_defaults(m);
}

fn electrical_dependent_defaults(m: &mut Model) {
    if !m.vddeotGiven {
        m.vddeot = if m.r#type == 1 { 1.5 } else { -1.5 };
    }
    if !m.toxpGiven {
        m.toxp = m.toxe;
    }
    if !m.toxmGiven {
        m.toxm = m.toxe;
    }
    if !m.ketacGiven {
        m.ketac = m.keta;
    }
    if !m.dsubGiven {
        m.dsub = m.drout;
    }
    if !m.vth0Given {
        m.vth0 = if m.r#type == 1 { 0.7 } else { -0.7 };
    }
    if !m.euGiven {
        m.eu = if m.r#type == 1 { 1.67 } else { 1.0 };
    }
    if !m.ucsGiven {
        m.ucs = if m.r#type == 1 { 1.67 } else { 1.0 };
    }
    if m.v48intVersion <= 480 {
        if !m.uaGiven {
            m.ua = if m.mobMod == 2 { 1.0e-15 } else { 1.0e-9 };
        }
        if !m.ucGiven {
            m.uc = if m.mobMod == 1 { -0.0465 } else { -0.0465e-9 };
        }
        if !m.uc1Given {
            m.uc1 = if m.mobMod == 1 { -0.056 } else { -0.056e-9 };
        }
    } else {
        if !m.uaGiven {
            m.ua = if m.mobMod == 2 || m.mobMod == 6 {
                1.0e-15
            } else {
                1.0e-9
            };
        }
        if !m.ucGiven {
            m.uc = if m.mobMod == 1 || m.mobMod == 5 {
                -0.0465
            } else {
                -0.0465e-9
            };
        }
        if !m.uc1Given {
            m.uc1 = if m.mobMod == 1 || m.mobMod == 5 {
                -0.056
            } else {
                -0.056e-9
            };
        }
    }
    if !m.u0Given {
        m.u0 = if m.r#type == 1 { 0.067 } else { 0.025 };
    }
    leakage_defaults(m);
}

fn leakage_defaults(m: &mut Model) {
    if !m.agislGiven {
        m.agisl = m.agidl;
    }
    if !m.bgislGiven {
        m.bgisl = m.bgidl;
    }
    if !m.cgislGiven {
        m.cgisl = m.cgidl;
    }
    if !m.egislGiven {
        m.egisl = m.egidl;
    }
    if !m.rgislGiven {
        m.rgisl = m.rgidl;
    }
    if !m.kgislGiven {
        m.kgisl = m.kgidl;
    }
    if !m.fgislGiven {
        m.fgisl = m.fgidl;
    }
    if !m.aigcGiven {
        m.aigc = if m.r#type == 1 { 1.36e-2 } else { 9.80e-3 };
    }
    if !m.bigcGiven {
        m.bigc = if m.r#type == 1 { 1.71e-3 } else { 7.59e-4 };
    }
    if !m.cigcGiven {
        m.cigc = if m.r#type == 1 { 0.075 } else { 0.03 };
    }
    if m.aigsdGiven {
        m.aigd = m.aigsd;
        m.aigs = m.aigd;
    } else {
        m.aigsd = if m.r#type == 1 { 1.36e-2 } else { 9.80e-3 };
        if !m.aigsGiven {
            m.aigs = if m.r#type == 1 { 1.36e-2 } else { 9.80e-3 };
        }
        if !m.aigdGiven {
            m.aigd = if m.r#type == 1 { 1.36e-2 } else { 9.80e-3 };
        }
    }
    if m.bigsdGiven {
        m.bigd = m.bigsd;
        m.bigs = m.bigd;
    } else {
        m.bigsd = if m.r#type == 1 { 1.71e-3 } else { 7.59e-4 };
        if !m.bigsGiven {
            m.bigs = if m.r#type == 1 { 1.71e-3 } else { 7.59e-4 };
        }
        if !m.bigdGiven {
            m.bigd = if m.r#type == 1 { 1.71e-3 } else { 7.59e-4 };
        }
    }
    if m.cigsdGiven {
        m.cigd = m.cigsd;
        m.cigs = m.cigd;
    } else {
        m.cigsd = if m.r#type == 1 { 0.075 } else { 0.03 };
        if !m.cigsGiven {
            m.cigs = if m.r#type == 1 { 0.075 } else { 0.03 };
        }
        if !m.cigdGiven {
            m.cigd = if m.r#type == 1 { 0.075 } else { 0.03 };
        }
    }
    if !m.ijthdfwdGiven {
        m.ijthdfwd = m.ijthsfwd;
    }
    if !m.ijthdrevGiven {
        m.ijthdrev = m.ijthsrev;
    }
    if !m.xjbvdGiven {
        m.xjbvd = m.xjbvs;
    }
    if !m.bvdGiven {
        m.bvd = m.bvs;
    }
    if !m.ckappadGiven {
        m.ckappad = m.ckappas;
    }
    if !m.dmciGiven {
        m.dmci = m.dmcg;
    }
}

fn length_dependent_defaults(m: &mut Model) {
    if !m.lketacGiven {
        m.lketac = m.lketa;
    }
    if !m.lagislGiven {
        m.lagisl = m.lagidl;
    }
    if !m.lbgislGiven {
        m.lbgisl = m.lbgidl;
    }
    if !m.lcgislGiven {
        m.lcgisl = m.lcgidl;
    }
    if !m.legislGiven {
        m.legisl = m.legidl;
    }
    if !m.lrgislGiven {
        m.lrgisl = m.lrgidl;
    }
    if !m.lkgislGiven {
        m.lkgisl = m.lkgidl;
    }
    if !m.lfgislGiven {
        m.lfgisl = m.lfgidl;
    }
    if !m.aigsdGiven && (m.aigsGiven || m.aigdGiven) {
        if !m.laigsGiven {
            m.laigs = 0.0;
        }
        if !m.laigdGiven {
            m.laigd = 0.0;
        }
    } else {
        if !m.laigsdGiven {
            m.laigsd = 0.0;
        }
        m.laigd = m.laigsd;
        m.laigs = m.laigd;
    }
    if !m.bigsdGiven && (m.bigsGiven || m.bigdGiven) {
        if !m.lbigsGiven {
            m.lbigs = 0.0;
        }
        if !m.lbigdGiven {
            m.lbigd = 0.0;
        }
    } else {
        if !m.lbigsdGiven {
            m.lbigsd = 0.0;
        }
        m.lbigd = m.lbigsd;
        m.lbigs = m.lbigd;
    }
    if !m.cigsdGiven && (m.cigsGiven || m.cigdGiven) {
        if !m.lcigsGiven {
            m.lcigs = 0.0;
        }
        if !m.lcigdGiven {
            m.lcigd = 0.0;
        }
    } else {
        if !m.lcigsdGiven {
            m.lcigsd = 0.0;
        }
        m.lcigd = m.lcigsd;
        m.lcigs = m.lcigd;
    }
}

fn width_dependent_defaults(m: &mut Model) {
    if !m.wketacGiven {
        m.wketac = m.wketa;
    }
    if !m.wagislGiven {
        m.wagisl = m.wagidl;
    }
    if !m.wbgislGiven {
        m.wbgisl = m.wbgidl;
    }
    if !m.wcgislGiven {
        m.wcgisl = m.wcgidl;
    }
    if !m.wegislGiven {
        m.wegisl = m.wegidl;
    }
    if !m.wrgislGiven {
        m.wrgisl = m.wrgidl;
    }
    if !m.wkgislGiven {
        m.wkgisl = m.wkgidl;
    }
    if !m.wfgislGiven {
        m.wfgisl = m.wfgidl;
    }
    if !m.aigsdGiven && (m.aigsGiven || m.aigdGiven) {
        if !m.waigsGiven {
            m.waigs = 0.0;
        }
        if !m.waigdGiven {
            m.waigd = 0.0;
        }
    } else {
        if !m.waigsdGiven {
            m.waigsd = 0.0;
        }
        m.waigd = m.waigsd;
        m.waigs = m.waigd;
    }
    if !m.bigsdGiven && (m.bigsGiven || m.bigdGiven) {
        if !m.wbigsGiven {
            m.wbigs = 0.0;
        }
        if !m.wbigdGiven {
            m.wbigd = 0.0;
        }
    } else {
        if !m.wbigsdGiven {
            m.wbigsd = 0.0;
        }
        m.wbigd = m.wbigsd;
        m.wbigs = m.wbigd;
    }
    if !m.cigsdGiven && (m.cigsGiven || m.cigdGiven) {
        if !m.wcigsGiven {
            m.wcigs = 0.0;
        }
        if !m.wcigdGiven {
            m.wcigd = 0.0;
        }
    } else {
        if !m.wcigsdGiven {
            m.wcigsd = 0.0;
        }
        m.wcigd = m.wcigsd;
        m.wcigs = m.wcigd;
    }
}

fn cross_dependent_defaults(m: &mut Model) {
    if !m.pketacGiven {
        m.pketac = m.pketa;
    }
    if !m.pagislGiven {
        m.pagisl = m.pagidl;
    }
    if !m.pbgislGiven {
        m.pbgisl = m.pbgidl;
    }
    if !m.pcgislGiven {
        m.pcgisl = m.pcgidl;
    }
    if !m.pegislGiven {
        m.pegisl = m.pegidl;
    }
    if !m.prgislGiven {
        m.prgisl = m.prgidl;
    }
    if !m.pkgislGiven {
        m.pkgisl = m.pkgidl;
    }
    if !m.pfgislGiven {
        m.pfgisl = m.pfgidl;
    }
    if !m.aigsdGiven && (m.aigsGiven || m.aigdGiven) {
        if !m.paigsGiven {
            m.paigs = 0.0;
        }
        if !m.paigdGiven {
            m.paigd = 0.0;
        }
    } else {
        if !m.paigsdGiven {
            m.paigsd = 0.0;
        }
        m.paigd = m.paigsd;
        m.paigs = m.paigd;
    }
    if !m.bigsdGiven && (m.bigsGiven || m.bigdGiven) {
        if !m.pbigsGiven {
            m.pbigs = 0.0;
        }
        if !m.pbigdGiven {
            m.pbigd = 0.0;
        }
    } else {
        if !m.pbigsdGiven {
            m.pbigsd = 0.0;
        }
        m.pbigd = m.pbigsd;
        m.pbigs = m.pbigd;
    }
    if !m.cigsdGiven && (m.cigsGiven || m.cigdGiven) {
        if !m.pcigsGiven {
            m.pcigs = 0.0;
        }
        if !m.pcigdGiven {
            m.pcigd = 0.0;
        }
    } else {
        if !m.pcigsdGiven {
            m.pcigsd = 0.0;
        }
        m.pcigd = m.pcigsd;
        m.pcigs = m.pcigd;
    }
}

fn geometry_defaults(m: &mut Model, nominal_temperature: f64) {
    if !m.tnomGiven {
        m.tnom = nominal_temperature;
    }
    if !m.LlcGiven {
        m.Llc = m.Ll;
    }
    if !m.LwcGiven {
        m.Lwc = m.Lw;
    }
    if !m.LwlcGiven {
        m.Lwlc = m.Lwl;
    }
    if !m.WlcGiven {
        m.Wlc = m.Wl;
    }
    if !m.WwcGiven {
        m.Wwc = m.Ww;
    }
    if !m.WwlcGiven {
        m.Wwlc = m.Wwl;
    }
    if !m.dwcGiven {
        m.dwc = m.Wint;
    }
    if !m.dlcGiven {
        m.dlc = m.Lint;
    }
    if !m.dlcigGiven {
        m.dlcig = m.Lint;
    }
    if !m.dlcigdGiven {
        if m.dlcigGiven {
            m.dlcigd = m.dlcig;
        } else {
            m.dlcigd = m.Lint;
        }
    }
    if !m.dwjGiven {
        m.dwj = m.dwc;
    }
}

fn junction_and_noise_defaults(m: &mut Model) {
    if !m.DunitAreaJctCapGiven {
        m.DunitAreaJctCap = m.SunitAreaJctCap;
    }
    if !m.DunitLengthSidewallJctCapGiven {
        m.DunitLengthSidewallJctCap = m.SunitLengthSidewallJctCap;
    }
    if !m.SunitLengthGateSidewallJctCapGiven {
        m.SunitLengthGateSidewallJctCap = m.SunitLengthSidewallJctCap;
    }
    if !m.DunitLengthGateSidewallJctCapGiven {
        m.DunitLengthGateSidewallJctCap = m.SunitLengthGateSidewallJctCap;
    }
    if !m.DjctSatCurDensityGiven {
        m.DjctSatCurDensity = m.SjctSatCurDensity;
    }
    if !m.DjctSidewallSatCurDensityGiven {
        m.DjctSidewallSatCurDensity = m.SjctSidewallSatCurDensity;
    }
    if !m.DjctGateSidewallSatCurDensityGiven {
        m.DjctGateSidewallSatCurDensity = m.SjctGateSidewallSatCurDensity;
    }
    if !m.DbulkJctPotentialGiven {
        m.DbulkJctPotential = m.SbulkJctPotential;
    }
    if !m.DsidewallJctPotentialGiven {
        m.DsidewallJctPotential = m.SsidewallJctPotential;
    }
    if !m.SGatesidewallJctPotentialGiven {
        m.SGatesidewallJctPotential = m.SsidewallJctPotential;
    }
    if !m.DGatesidewallJctPotentialGiven {
        m.DGatesidewallJctPotential = m.SGatesidewallJctPotential;
    }
    if !m.DbulkJctBotGradingCoeffGiven {
        m.DbulkJctBotGradingCoeff = m.SbulkJctBotGradingCoeff;
    }
    if !m.DbulkJctSideGradingCoeffGiven {
        m.DbulkJctSideGradingCoeff = m.SbulkJctSideGradingCoeff;
    }
    if !m.SbulkJctGateSideGradingCoeffGiven {
        m.SbulkJctGateSideGradingCoeff = m.SbulkJctSideGradingCoeff;
    }
    if !m.DbulkJctGateSideGradingCoeffGiven {
        m.DbulkJctGateSideGradingCoeff = m.SbulkJctGateSideGradingCoeff;
    }
    if !m.DjctEmissionCoeffGiven {
        m.DjctEmissionCoeff = m.SjctEmissionCoeff;
    }
    if !m.DjctTempExponentGiven {
        m.DjctTempExponent = m.SjctTempExponent;
    }
    tunnelling_and_noise_defaults(m);
}

fn tunnelling_and_noise_defaults(m: &mut Model) {
    if !m.jtsdGiven {
        m.jtsd = m.jtss;
    }
    if !m.jtsswdGiven {
        m.jtsswd = m.jtssws;
    }
    if !m.jtsswgdGiven {
        m.jtsswgd = m.jtsswgs;
    }
    if !m.njtsdGiven {
        if m.njtsGiven {
            m.njtsd = m.njts;
        } else {
            m.njtsd = 20.0;
        }
    }
    if !m.njtsswdGiven {
        if m.njtsswGiven {
            m.njtsswd = m.njtssw;
        } else {
            m.njtsswd = 20.0;
        }
    }
    if !m.njtsswgdGiven {
        if m.njtsswgGiven {
            m.njtsswgd = m.njtsswg;
        } else {
            m.njtsswgd = 20.0;
        }
    }
    if !m.xtsdGiven {
        m.xtsd = m.xtss;
    }
    if !m.xtsswdGiven {
        m.xtsswd = m.xtssws;
    }
    if !m.xtsswgdGiven {
        m.xtsswgd = m.xtsswgs;
    }
    if !m.tnjtsdGiven {
        if m.tnjtsGiven {
            m.tnjtsd = m.tnjts;
        } else {
            m.tnjtsd = 0.0;
        }
    }
    if !m.tnjtsswdGiven {
        if m.tnjtsswGiven {
            m.tnjtsswd = m.tnjtssw;
        } else {
            m.tnjtsswd = 0.0;
        }
    }
    if !m.tnjtsswgdGiven {
        if m.tnjtsswgGiven {
            m.tnjtsswgd = m.tnjtsswg;
        } else {
            m.tnjtsswgd = 0.0;
        }
    }
    if !m.vtsdGiven {
        m.vtsd = m.vtss;
    }
    if !m.vtsswdGiven {
        m.vtsswd = m.vtssws;
    }
    if !m.vtsswgdGiven {
        m.vtsswgd = m.vtsswgs;
    }
    if !m.oxideTrapDensityAGiven {
        if m.r#type == 1 {
            m.oxideTrapDensityA = 6.25e41;
        } else {
            m.oxideTrapDensityA = 6.188e40;
        }
    }
    if !m.oxideTrapDensityBGiven {
        if m.r#type == 1 {
            m.oxideTrapDensityB = 3.125e26;
        } else {
            m.oxideTrapDensityB = 1.5e25;
        }
    }
}

/// Prepare instance defaults and selector overrides before choosing nodes.
fn setup_instance(
    m: &Model,
    h: &mut Instance,
    noise_analysis: bool,
    warnings: &mut Vec<String>,
) -> Vec<(i32, &'static str)> {
    instance_defaults(m, h);
    normalize_instance_selectors(m, h, warnings);
    layout_defaults(m, h);
    let plan = decide_topology(m, h, noise_analysis, warnings);
    apply_topology(h, plan)
}

fn instance_defaults(m: &Model, h: &mut Instance) {
    if !h.lGiven {
        h.l = 5.0e-6;
    }
    if !h.wGiven {
        h.w = 5.0e-6;
    }
    if !h.mGiven {
        h.m = 1.0;
    }
    if !h.mult_iGiven {
        h.mult_i = 1.0;
    }
    if !h.mult_qGiven {
        h.mult_q = 1.0;
    }
    if !h.mult_fnGiven {
        h.mult_fn = h.mult_i;
    }
    if !h.nfGiven {
        h.nf = 1.0;
    }
    if !h.minGiven {
        h.min = 0; // integer
    }
    if !h.icVDSGiven {
        h.icVDS = 0.0;
    }
    if !h.icVGSGiven {
        h.icVGS = 0.0;
    }
    if !h.icVBSGiven {
        h.icVBS = 0.0;
    }
    if !h.drainAreaGiven {
        h.drainArea = 0.0;
    }
    if !h.drainPerimeterGiven {
        h.drainPerimeter = 0.0;
    }
    if !h.drainSquaresGiven {
        h.drainSquares = 1.0;
    }
    if !h.sourceAreaGiven {
        h.sourceArea = 0.0;
    }
    if !h.sourcePerimeterGiven {
        h.sourcePerimeter = 0.0;
    }
    if !h.sourceSquaresGiven {
        h.sourceSquares = 1.0;
    }

    if !h.rbdbGiven {
        h.rbdb = m.rbdb; // in ohm
    }
    if !h.rbsbGiven {
        h.rbsb = m.rbsb;
    }
    if !h.rbpbGiven {
        h.rbpb = m.rbpb;
    }
    if !h.rbpsGiven {
        h.rbps = m.rbps;
    }
    if !h.rbpdGiven {
        h.rbpd = m.rbpd;
    }
    if !h.delvtoGiven {
        h.delvto = 0.0;
    }
    if !h.mulu0Given {
        h.mulu0 = 1.0;
    }
    if !h.xgwGiven {
        h.xgw = m.xgw;
    }
    if !h.ngconGiven {
        h.ngcon = m.ngcon;
    }
    if !h.dtempGiven {
        h.dtemp = 0.0;
    }
}

fn normalize_instance_selectors(m: &Model, h: &mut Instance, warnings: &mut Vec<String>) {
    // Unspecified selectors inherit model defaults without becoming "given".
    for selector in [
        Selector::new("rbodyMod", &mut h.rbodyMod, h.rbodyModGiven, m.rbodyMod, 2),
        Selector::new("rgateMod", &mut h.rgateMod, h.rgateModGiven, m.rgateMod, 3),
        Selector::new("trnqsMod", &mut h.trnqsMod, h.trnqsModGiven, m.trnqsMod, 1),
        Selector::new("acnqsMod", &mut h.acnqsMod, h.acnqsModGiven, m.acnqsMod, 1),
    ] {
        selector.normalize(SelectorOrigin::Instance, warnings);
    }
    if !h.geoModGiven {
        h.geoMod = m.geoMod;
    }
    if !h.rgeoModGiven {
        h.rgeoMod = 0;
    }
}

fn layout_defaults(m: &Model, h: &mut Instance) {
    // Stress effect.
    if !h.saGiven {
        h.sa = 0.0;
    }
    if !h.sbGiven {
        h.sb = 0.0;
    }
    if !h.sdGiven {
        h.sd = 2.0 * m.dmcg;
    }
    // Well proximity effect.
    if !h.scaGiven {
        h.sca = 0.0;
    }
    if !h.scbGiven {
        h.scb = 0.0;
    }
    if !h.sccGiven {
        h.scc = 0.0;
    }
    if !h.scGiven {
        h.sc = 0.0; // m
    }
}

enum GateNodes {
    External,
    Prime,
    PrimeAndMid,
}

struct TopologyPlan {
    series: [bool; 2],
    gate: GateNodes,
    body: bool,
    charge: bool,
}

#[derive(Clone, Copy)]
enum SeriesTerminal {
    Drain,
    Source,
}

/// C's series-resistance decision is made for drain and then source.
/// Keeping the call order also preserves geometry diagnostic order.
fn needs_prime(
    m: &Model,
    h: &Instance,
    terminal: SeriesTerminal,
    noise_analysis: bool,
    warnings: &mut Vec<String>,
) -> bool {
    if m.rdsMod != 0 || (m.tnoiMod == 1 && noise_analysis) {
        return true;
    }
    if m.sheetResistance <= 0.0 || m.sheetResistance.is_nan() {
        return false;
    }
    let (given, squares, side) = match terminal {
        SeriesTerminal::Drain => (h.drainSquaresGiven, h.drainSquares, 0),
        SeriesTerminal::Source => (h.sourceSquaresGiven, h.sourceSquares, 1),
    };
    if given {
        return squares > 0.0;
    }
    if h.rgeoMod == 0 {
        return false;
    }
    rdseff_geo(
        h.nf,
        h.geoMod,
        h.rgeoMod,
        h.min,
        h.w,
        m.sheetResistance,
        m.dmcg - m.dmcgt,
        m.dmci,
        m.dmdg - m.dmcgt,
        side,
        warnings,
    ) > 0.0
}

fn decide_topology(
    m: &Model,
    h: &Instance,
    noise_analysis: bool,
    warnings: &mut Vec<String>,
) -> TopologyPlan {
    let drain = needs_prime(m, h, SeriesTerminal::Drain, noise_analysis, warnings);
    let source = needs_prime(m, h, SeriesTerminal::Source, noise_analysis, warnings);
    let gate = match h.rgateMod {
        3 => GateNodes::PrimeAndMid,
        value if value > 0 => GateNodes::Prime,
        _ => GateNodes::External,
    };
    TopologyPlan {
        series: [drain, source],
        gate,
        body: matches!(h.rbodyMod, 1 | 2),
        charge: h.trnqsMod != 0,
    }
}

/// Apply the topology once, assigning all aliases and creating nodes in C order.
fn apply_topology(h: &mut Instance, plan: TopologyPlan) -> Vec<(i32, &'static str)> {
    let mut created = Vec::new();
    if plan.series[0] {
        h.dNodePrime = node::DP;
        created.push((node::DP, "drain"));
    } else {
        h.dNodePrime = h.dNode;
    }
    if plan.series[1] {
        h.sNodePrime = node::SP;
        created.push((node::SP, "source"));
    } else {
        h.sNodePrime = h.sNode;
    }
    if !matches!(plan.gate, GateNodes::External) {
        h.gNodePrime = node::GP;
        created.push((node::GP, "gate"));
    } else {
        h.gNodePrime = h.gNodeExt;
    }

    if matches!(plan.gate, GateNodes::PrimeAndMid) {
        h.gNodeMid = node::GM;
        created.push((node::GM, "midgate"));
    } else {
        h.gNodeMid = h.gNodeExt;
    }

    // Internal body nodes for the body resistance model.
    if plan.body {
        h.dbNode = node::DB;
        created.push((node::DB, "dbody"));
        h.bNodePrime = node::BP;
        created.push((node::BP, "body"));
        h.sbNode = node::SB;
        created.push((node::SB, "sbody"));
    } else {
        h.dbNode = h.bNode;
        h.bNodePrime = h.bNode;
        h.sbNode = h.bNode;
    }

    // NQS node.
    if plan.charge {
        h.qNode = node::Q;
        created.push((node::Q, "charge"));
    } else {
        h.qNode = node::GROUND;
    }
    created
}

impl ModelParameters {
    /// Every numeric model keyword with the value C's `BSIM4setup` model
    /// pass leaves in its member: the explicit setters, then `b4set.c`'s
    /// defaults at nominal temperature `nominal_temperature_celsius`.
    /// Aliases (`vtho`, ...) and ask-only, flag and string rows are
    /// omitted. Integer members are reported as reals. Values are before
    /// `b4temp.c`, which the C `@model[...]` ask sees after; this is a
    /// verification hook for the setup stage, not a simulation result.
    ///
    /// # Errors
    /// An invalid nominal temperature, or an internal member mismatch.
    pub fn setup_values(
        &self,
        nominal_temperature_celsius: Real,
    ) -> SpiceResult<Vec<(&'static str, Real)>> {
        let tnom = temperature_kelvin(nominal_temperature_celsius, &self.location)?;
        let mut prepared = prepare_model(super::apply::model(self)?, tnom);
        numeric_members(&mut prepared.model)
    }
}

/// `(keyword, value)` for every non-redundant numeric MODEL row.
fn numeric_members(m: &mut Model) -> SpiceResult<Vec<(&'static str, Real)>> {
    let mut values = Vec::new();
    for p in parameters::MODEL.iter().filter(|p| !p.redundant) {
        if !matches!(p.input_type, InputType::Real | InputType::Integer) {
            continue;
        }
        let Some(field) = p.c_field else { continue };
        let value = if let Some(v) = m.real_mut(field) {
            *v
        } else if let Some(v) = m.int_mut(field) {
            Real::from(*v)
        } else {
            return Err(SpiceError::circuit(format!(
                "BSIM4 model member {field} for {} is missing",
                p.name
            )));
        };
        values.push((p.name, value));
    }
    Ok(values)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn defaulted(polarity: i32) -> (Model, Vec<String>) {
        let mut m = Model {
            r#type: polarity,
            typeGiven: true,
            ..Model::default()
        };
        let mut warnings = Vec::new();
        setup_model(&mut m, 300.15, &mut warnings);
        (m, warnings)
    }

    #[test]
    fn literal_defaults_agree_with_the_generated_facts() {
        let (mut m, warnings) = defaulted(1);
        assert!(warnings.is_empty(), "{warnings:?}");
        let mut checked = 0;
        for &(keyword, expected) in super::super::defaults_data::LITERALS {
            let p = super::super::DEVICE.model.get(keyword).unwrap();
            let field = p.c_field.unwrap();
            let actual = if let Some(v) = m.real_mut(field) {
                *v
            } else {
                Real::from(*m.int_mut(field).unwrap())
            };
            assert_eq!(actual.to_bits(), expected.to_bits(), "{keyword}");
            checked += 1;
        }
        assert_eq!(checked, 777);
    }

    #[test]
    fn independent_default_stage_never_writes_given_parameters_or_given_bits() {
        let facts = super::super::defaults_data::LITERALS;
        let sentinel: Real = -123_456.25;
        let mut ungiven = Model::default();
        let mut given_model = Model::default();
        for &(keyword, _) in facts {
            let p = super::super::DEVICE.model.get(keyword).unwrap();
            let field = p.c_field.unwrap();
            let given = p.c_given.unwrap();
            for (m, explicitly_given) in [(&mut ungiven, false), (&mut given_model, true)] {
                if let Some(slot) = m.real_mut(field) {
                    *slot = sentinel;
                } else {
                    *m.int_mut(field).unwrap() = -123_456;
                }
                *m.flag_mut(given).unwrap() = explicitly_given;
            }
        }
        for m in [&mut ungiven, &mut given_model] {
            super::super::defaults_data::apply_independent_defaults(m);
        }
        let mut changed = 0;
        for &(keyword, expected) in facts {
            let p = super::super::DEVICE.model.get(keyword).unwrap();
            let field = p.c_field.unwrap();
            let given = p.c_given.unwrap();
            let (actual, original) = if let Some(slot) = ungiven.real_mut(field) {
                (*slot, sentinel)
            } else {
                (Real::from(*ungiven.int_mut(field).unwrap()), -123_456.)
            };
            if actual.to_bits() != original.to_bits() {
                changed += 1;
                assert_eq!(actual.to_bits(), expected.to_bits(), "{keyword}");
            }
            let explicit = if let Some(slot) = given_model.real_mut(field) {
                *slot
            } else {
                Real::from(*given_model.int_mut(field).unwrap())
            };
            assert_eq!(explicit.to_bits(), original.to_bits(), "{keyword}");
            assert!(!*ungiven.flag_mut(given).unwrap(), "{keyword}");
            assert!(*given_model.flag_mut(given).unwrap(), "{keyword}");
        }
        assert_eq!(changed, 733);
    }

    #[test]
    fn combined_and_separate_leakage_defaults_preserve_given_precedence() {
        let base = Model {
            aigsd: 0.012,
            aigsdGiven: true,
            aigs: 0.02,
            aigsGiven: true,
            aigd: 0.03,
            aigdGiven: true,
            laigsd: 2e-12,
            laigsdGiven: true,
            laigs: 3e-12,
            laigsGiven: true,
            laigd: 4e-12,
            laigdGiven: true,
            waigsd: 5e-12,
            waigsdGiven: true,
            paigsd: 6e-12,
            paigsdGiven: true,
            agidl: 7e-9,
            agidlGiven: true,
            lagidl: 8e-9,
            lagidlGiven: true,
            wagidl: 9e-9,
            wagidlGiven: true,
            pagidl: 1e-8,
            pagidlGiven: true,
            Lint: 1e-8,
            LintGiven: true,
            dlcig: 2e-8,
            dlcigGiven: true,
            ..Model::default()
        };
        let combined = prepare_model(base.clone(), 310.).model;
        assert_eq!([combined.aigs, combined.aigd], [0.012; 2]);
        assert_eq!([combined.laigs, combined.laigd], [2e-12; 2]);
        assert_eq!([combined.waigs, combined.waigd], [5e-12; 2]);
        assert_eq!([combined.paigs, combined.paigd], [6e-12; 2]);
        assert_eq!(
            [
                combined.agisl,
                combined.lagisl,
                combined.wagisl,
                combined.pagisl
            ],
            [7e-9, 8e-9, 9e-9, 1e-8]
        );
        assert_eq!(combined.dlcigd, 2e-8);
        let separate = prepare_model(
            Model {
                aigsdGiven: false,
                dlcigGiven: false,
                ..base
            },
            310.,
        )
        .model;
        assert_eq!([separate.aigs, separate.aigd], [0.02, 0.03]);
        assert_eq!([separate.laigs, separate.laigd], [3e-12, 4e-12]);
        assert_eq!(
            [
                separate.waigs,
                separate.waigd,
                separate.paigs,
                separate.paigd
            ],
            [0.; 4]
        );
        assert_eq!(separate.dlcigd, 1e-8);
    }

    #[test]
    fn polarity_and_dependent_defaults() {
        let (n, _) = defaulted(1);
        let (p, _) = defaulted(-1);
        assert_eq!((n.vth0, p.vth0), (0.7, -0.7));
        assert_eq!((n.u0, p.u0), (0.067, 0.025));
        assert_eq!(n.toxp, n.toxe);
        assert_eq!(n.tnom, 300.15);
        assert_eq!(n.version, "4.8.3");
        assert_eq!(n.v48intVersion, 483);
        assert_eq!(n.ua, 1.0e-9);
    }

    #[test]
    fn out_of_range_selectors_warn_as_c_does() {
        let mut m = Model {
            r#type: 1,
            typeGiven: true,
            capMod: 7,
            capModGiven: true,
            mtrlMod: 5,
            mtrlModGiven: true,
            version: "4.7".into(),
            versionGiven: true,
            ..Model::default()
        };
        let mut warnings = Vec::new();
        setup_model(&mut m, 300.15, &mut warnings);
        assert_eq!(m.capMod, 2);
        assert_eq!(m.mtrlMod, 0);
        assert_eq!(m.v48intVersion, 483);
        assert_eq!(
            warnings,
            [
                "Warning: capMod has been set to its default value: 2.",
                "Warning: mtrlMod has been set to its default value: 0.",
                "Warning: unknown BSIM4 version. Working now with BSIM4.8.3.",
            ]
        );
        // The first C prefix test makes all the more specific branches unreachable.
        for version in [
            "4.8",
            "4.80",
            "4.8.0",
            "4.8.1",
            "4.82",
            "4.8.3",
            "4.8-other",
        ] {
            let prepared = prepare_model(
                Model {
                    version: version.into(),
                    versionGiven: true,
                    ..Model::default()
                },
                300.15,
            );
            assert_eq!(prepared.model.v48intVersion, 483);
            assert!(prepared.warnings.is_empty(), "{version}");
        }
    }

    fn instance() -> Instance {
        Instance {
            dNode: node::D,
            gNodeExt: node::G_EXT,
            sNode: node::S,
            bNode: node::B,
            ..Instance::default()
        }
    }

    #[test]
    fn normalized_model_and_instance_selectors_keep_c_warning_order() {
        let mut model = Model {
            version: "unknown".into(),
            versionGiven: true,
            wpemod: 3.,
            wpemodGiven: true,
            ..Model::default()
        };
        let selectors = [
            ("mobMod", 0),
            ("dioMod", 1),
            ("capMod", 2),
            ("rdsMod", 0),
            ("rbodyMod", 0),
            ("rgateMod", 0),
            ("perMod", 1),
            ("fnoiMod", 1),
            ("tnoiMod", 0),
            ("trnqsMod", 0),
            ("acnqsMod", 0),
            ("mtrlMod", 0),
            ("mtrlCompatMod", 0),
            ("igcMod", 0),
            ("igbMod", 0),
            ("tempMod", 0),
        ];
        let mut expected = Vec::new();
        for (name, fallback) in selectors {
            *model.int_mut(&format!("BSIM4{name}")).unwrap() = -1;
            *model.flag_mut(&format!("BSIM4{name}Given")).unwrap() = true;
            expected.push(format!(
                "Warning: {name} has been set to its default value: {fallback}."
            ));
        }
        expected.push("Warning: unknown BSIM4 version. Working now with BSIM4.8.3.".into());
        expected.push("Warning: wpemod has been set to its default value: 0.".into());
        let mut prepared = prepare_model(model, 300.15);
        assert_eq!(prepared.warnings, expected);
        for (name, fallback) in selectors {
            assert_eq!(
                *prepared.model.int_mut(&format!("BSIM4{name}")).unwrap(),
                fallback
            );
            assert!(
                *prepared
                    .model
                    .flag_mut(&format!("BSIM4{name}Given"))
                    .unwrap()
            );
        }
        let h = Instance {
            rbodyMod: -1,
            rbodyModGiven: true,
            rgateMod: -1,
            rgateModGiven: true,
            trnqsMod: -1,
            trnqsModGiven: true,
            acnqsMod: -1,
            acnqsModGiven: true,
            ..instance()
        };
        let result = prepared.prepare_instance(h, false);
        assert_eq!(
            &result.warnings[expected.len()..],
            [
                "Warning: rbodyMod has been set to its global value 0.",
                "Warning: rgateMod has been set to its global value 0.",
                "Warning: trnqsMod has been set to its global value 0.",
                "Warning: acnqsMod has been set to its global value 0.",
            ]
        );
    }

    #[test]
    fn topology_order_and_aliases_cover_all_selector_combinations() {
        for rds in [0, 1] {
            for noise_mode in [0, 1, 2] {
                for gate in 0..=3 {
                    for body in 0..=2 {
                        for nqs in [0, 1] {
                            let prepared = prepare_model(
                                Model {
                                    rdsMod: rds,
                                    rdsModGiven: true,
                                    tnoiMod: noise_mode,
                                    tnoiModGiven: true,
                                    rgateMod: gate,
                                    rgateModGiven: true,
                                    rbodyMod: body,
                                    rbodyModGiven: true,
                                    trnqsMod: nqs,
                                    trnqsModGiven: true,
                                    sheetResistance: 10.,
                                    sheetResistanceGiven: true,
                                    ..Model::default()
                                },
                                300.15,
                            );
                            for noise_analysis in [false, true] {
                                for squares in
                                    [[false, false], [true, false], [false, true], [true, true]]
                                {
                                    let h = Instance {
                                        drainSquares: 2.,
                                        drainSquaresGiven: squares[0],
                                        sourceSquares: 2.,
                                        sourceSquaresGiven: squares[1],
                                        ..instance()
                                    };
                                    let result =
                                        prepared.clone().prepare_instance(h, noise_analysis);
                                    let forced = rds == 1 || (noise_mode == 1 && noise_analysis);
                                    assert_topology(
                                        &result,
                                        [
                                            forced || squares[0],
                                            forced || squares[1],
                                            gate != 0,
                                            gate == 3,
                                            body != 0,
                                            body != 0,
                                            body != 0,
                                            nqs != 0,
                                        ],
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    fn assert_topology(result: &PreparedSetup, flags: [bool; 8]) {
        let own = [
            node::DP,
            node::SP,
            node::GP,
            node::GM,
            node::DB,
            node::BP,
            node::SB,
            node::Q,
        ];
        let alias = [
            node::D,
            node::S,
            node::G_EXT,
            node::G_EXT,
            node::B,
            node::B,
            node::B,
            node::GROUND,
        ];
        let expected: Vec<_> = own
            .into_iter()
            .zip([
                "drain", "source", "gate", "midgate", "dbody", "body", "sbody", "charge",
            ])
            .zip(flags)
            .filter_map(|(node, create)| create.then_some(node))
            .collect();
        assert_eq!(result.created, expected);
        let h = &result.instance;
        let actual = [
            h.dNodePrime,
            h.sNodePrime,
            h.gNodePrime,
            h.gNodeMid,
            h.dbNode,
            h.bNodePrime,
            h.sbNode,
            h.qNode,
        ];
        for (((actual, own), alias), create) in actual.into_iter().zip(own).zip(alias).zip(flags) {
            assert_eq!(actual, if create { own } else { alias });
        }
        assert!(result.warnings.is_empty());
    }

    #[test]
    fn default_instance_creates_no_nodes() {
        let (m, _) = defaulted(1);
        let mut h = instance();
        let mut w = Vec::new();
        let created = setup_instance(&m, &mut h, false, &mut w);
        assert!(created.is_empty());
        assert_eq!((h.l, h.w, h.nf, h.m), (5.0e-6, 5.0e-6, 1.0, 1.0));
        assert_eq!(
            [h.dNodePrime, h.sNodePrime, h.gNodePrime, h.gNodeMid],
            [node::D, node::S, node::G_EXT, node::G_EXT]
        );
        assert_eq!([h.dbNode, h.bNodePrime, h.sbNode], [node::B; 3]);
        assert_eq!(h.qNode, node::GROUND);
        assert_eq!(h.sd, 2.0 * m.dmcg);
    }

    #[test]
    fn every_internal_node_in_c_creation_order() {
        let (mut m, _) = defaulted(1);
        m.rdsMod = 1;
        let mut h = instance();
        h.rgateMod = 3;
        h.rgateModGiven = true;
        h.rbodyMod = 1;
        h.rbodyModGiven = true;
        h.trnqsMod = 1;
        h.trnqsModGiven = true;
        let mut w = Vec::new();
        let created = setup_instance(&m, &mut h, false, &mut w);
        let suffixes: Vec<_> = created.iter().map(|c| c.1).collect();
        assert_eq!(
            suffixes,
            [
                "drain", "source", "gate", "midgate", "dbody", "body", "sbody", "charge"
            ]
        );
        assert!(w.is_empty());
    }

    #[test]
    fn invalid_instance_selector_falls_back_with_warning() {
        let (m, _) = defaulted(1);
        let mut h = instance();
        h.rbodyMod = 5;
        h.rbodyModGiven = true;
        let mut w = Vec::new();
        setup_instance(&m, &mut h, false, &mut w);
        assert_eq!(h.rbodyMod, 0);
        assert_eq!(w, ["Warning: rbodyMod has been set to its global value 0."]);
    }

    #[test]
    fn sheet_resistance_with_squares_or_rgeo_creates_primes() {
        let (mut m, _) = defaulted(1);
        m.sheetResistance = 10.0;
        let mut h = instance();
        h.drainSquares = 2.0;
        h.drainSquaresGiven = true;
        let mut w = Vec::new();
        let created = setup_instance(&m, &mut h, false, &mut w);
        assert_eq!(created, [(node::DP, "drain")]);
        // tnoiMod=1 creates both primes only for a deck with .noise.
        let (mut m, _) = defaulted(1);
        m.tnoiMod = 1;
        let mut h = instance();
        assert!(setup_instance(&m, &mut h, false, &mut w).is_empty());
        let mut h = instance();
        assert_eq!(setup_instance(&m, &mut h, true, &mut w).len(), 2);
    }
}
