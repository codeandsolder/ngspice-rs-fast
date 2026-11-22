//! Diodes under `.sens`: the `dio.c` tables, `diompar.c`/`dioparam.c`
//! setters, `diomask.c`/`dioask.c`, the defaults and setup-only quantities of
//! `diosetup.c` and `DIOtemp` (`diotemp.c`). The load itself is the port's
//! diode ([`Diode`]) built from C's records.
//!
//! `DIOsetup` runs once per parameter **before** the perturbation, so what it
//! alone derives does not follow the perturbed value: the series conductance
//! `1/RS` (an `rs` perturbation changes nothing) and the instance knee
//! currents `IKF*AREA*M`, `IKR*AREA*M`, `IKP*PJ*M`. A knee never given has an
//! instance knee of zero; perturbing it marks it given and `dioload.c`
//! divides by that zero, so C reports NaN, which the port reproduces. Setup
//! also defaults `NBV` to the `N` it sees and clears a knee's *given* flag
//! below `CKTepsmin`, and fields never given read C's defaults (`ISR` reads
//! `1e-14`: perturbing it turns the recombination current on, and restoring
//! it leaves it on for every later parameter of the model).

use super::thermal::TemperatureCache;
use super::{
    Diode, DiodeGeometry, DiodeInstanceParameters, DiodeParameters, DiodeSetup, EPSMIN, default_eg,
};
use crate::devices::sensitivity::{
    AcReset, CELSIUS_TO_KELVIN, ParameterScope, RecordPair, ResetStamp, ResetStamps,
    SensitivityFamily, SensitivityKernel, SensitivityLoad, SensitivityMode,
    SensitivityParameter as P, SensitivityRecord, StampAt, StampPart, written,
};
use crate::devices::{Device, LinearContext, StampContext};
use crate::primitives::{NodeId, Real, SpiceError, SpiceResult};

/// `DIOmPTable` entries `set_param()` accepts in a DC analysis (the `IF_AC`
/// `tt`, `ttt1`, `ttt2`, `cjo` follow in AC; `tnom`, `rsw` and the SOA and
/// self-heating limits are `IF_NONSENSE`, `level`/`tlev`/`tlevc` integers).
const MODEL: &[P] = &[
    P::dc("is"),
    P::dc("jsw"),
    P::dc("rs"),
    P::dc("trs"),
    P::dc("trs2"),
    P::dc("n"),
    P::dc("ns"),
    P::ac("tt"),
    P::ac("ttt1"),
    P::ac("ttt2"),
    P::ac("cjo"),
    P::dc("vj"),
    P::dc("m"),
    P::dc("tm1"),
    P::dc("tm2"),
    P::dc("cjp"),
    P::dc("php"),
    P::dc("mjsw"),
    P::dc("ikf"),
    P::dc("ikr"),
    P::dc("ikp"),
    P::dc("nbv"),
    P::dc("area"),
    P::dc("pj"),
    P::dc("eg"),
    P::dc("gap1"),
    P::dc("gap2"),
    P::dc("xti"),
    P::dc("cta"),
    P::dc("ctp"),
    P::dc("tpb"),
    P::dc("tphp"),
    P::dc("jtun"),
    P::dc("jtunsw"),
    P::dc("ntun"),
    P::dc("xtitun"),
    P::dc("keg"),
    P::dc("kf"),
    P::dc("af"),
    P::dc("fc"),
    P::dc("fcs"),
    P::dc("bv"),
    P::dc("ibv"),
    P::dc("tcv"),
    P::dc("isr"),
    P::dc("nr"),
    P::dc("vp"),
    P::dc("qpscale"),
    P::dc("lm"),
    P::dc("lp"),
    P::dc("wm"),
    P::dc("wp"),
    P::dc("xom"),
    P::dc("xoi"),
    P::dc("xm"),
    P::dc("xp"),
    P::dc("xw"),
];

/// `DIOpTable` entries `set_param()` accepts (`ic` is `IF_AC`).
const INSTANCE: &[P] = &[
    P::dc("temp"),
    P::dc("dtemp"),
    P::ac("ic"),
    P::dc("area"),
    P::dc("pj"),
    P::dc("w"),
    P::dc("l"),
    P::dc("m"),
    P::dc("lm"),
    P::dc("lp"),
    P::dc("wm"),
    P::dc("wp"),
];

/// Model fields beyond the table: `(keyword, port setter name)`.
const MODEL_FIELDS: &[&str] = &[
    "is",
    "jsw",
    "rs",
    "trs",
    "trs2",
    "n",
    "ns",
    "tt",
    "ttt1",
    "ttt2",
    "cjo",
    "vj",
    "m",
    "tm1",
    "tm2",
    "cjp",
    "php",
    "mjsw",
    "ikf",
    "ikr",
    "ikp",
    "nbv",
    "area",
    "pj",
    "eg",
    "gap1",
    "gap2",
    "xti",
    "cta",
    "ctp",
    "tpb",
    "tphp",
    "jtun",
    "jtunsw",
    "ntun",
    "xtitun",
    "keg",
    "kf",
    "af",
    "fc",
    "fcs",
    "bv",
    "ibv",
    "tcv",
    "isr",
    "nr",
    "vp",
    "qpscale",
    "lm",
    "lp",
    "wm",
    "wp",
    "xom",
    "xoi",
    "xm",
    "xp",
    "xw",
    "tnom",
    "tlev",
    "tlevc",
    "conductance",
];
const INSTANCE_FIELDS: &[&str] = &[
    "temp", "dtemp", "ic", "area", "pj", "w", "l", "m", "lm", "lp", "wm", "wp", "ikf", "ikr", "ikp",
];

/// The port's canonical setter name of a C model field (`cjp`/`php` are the
/// port's `cjsw`/`vjsw`, `trs` its `trs`).
fn port_name(keyword: &str) -> &str {
    match keyword {
        "cjp" => "cjsw",
        "php" => "vjsw",
        other => other,
    }
}

fn record_key(keyword: &str) -> &str {
    match keyword {
        "cjsw" => "cjp",
        "vjsw" => "php",
        other => other,
    }
}

/// A diode's `.sens` description.
#[derive(Debug)]
pub(in crate::devices) struct DiodeSensitivity<'a> {
    diode: &'a Diode,
    circuit_kelvin: Real,
    nominal_kelvin: Real,
}

impl<'a> DiodeSensitivity<'a> {
    pub(crate) fn new(diode: &'a Diode, context: &crate::devices::ModelContext) -> Self {
        Self {
            diode,
            circuit_kelvin: context.temperature + CELSIUS_TO_KELVIN,
            nominal_kelvin: context.nominal_temperature + CELSIUS_TO_KELVIN,
        }
    }
}

fn default(record: &mut SensitivityRecord, keyword: &str, value: Real) -> SpiceResult<()> {
    if record.given(keyword) {
        Ok(())
    } else {
        record.store(keyword, value)
    }
}

// Geometry, multiplier and effective knee currents belong to the instance.
fn setup_diode_instance(
    model: &SensitivityRecord,
    instance: &mut SensitivityRecord,
) -> SpiceResult<()> {
    let geometry = instance.given("w") || instance.given("l");
    let supplied = |key| -> SpiceResult<Option<Real>> {
        instance.given(key).then(|| instance.value(key)).transpose()
    };
    let defaults = if geometry {
        [1., 0.]
    } else {
        [model.value("area")?, model.value("pj")?]
    };
    let geometry =
        DiodeGeometry::setup(supplied("area")?, supplied("pj")?, supplied("m")?, defaults);
    instance.store("area", geometry.area)?;
    instance.store("pj", geometry.perimeter)?;
    instance.store("m", geometry.multiplier)?;
    let knees = DiodeSetup::scale_knees(
        [
            model.value("ikf")?,
            model.value("ikr")?,
            model.value("ikp")?,
        ],
        geometry,
    );
    for (key, value) in ["ikf", "ikr", "ikp"].into_iter().zip(knees) {
        instance.store(key, value)?;
    }
    Ok(())
}

impl SensitivityKernel for DiodeSensitivity<'_> {
    fn family(&self) -> SensitivityFamily {
        SensitivityFamily::Diode
    }

    fn model(&self) -> Option<&str> {
        Some(&self.diode.model)
    }

    fn model_parameters(&self) -> &'static [P] {
        MODEL
    }

    fn instance_parameters(&self) -> &'static [P] {
        INSTANCE
    }

    fn records(&self) -> SpiceResult<(SensitivityRecord, SensitivityRecord)> {
        let (m, i) = &*self.diode.written;
        let mut model = SensitivityRecord::new(self.family(), ParameterScope::Model);
        for keyword in MODEL_FIELDS {
            let value = written(m, port_name(keyword));
            let value = match (*keyword, value) {
                ("tnom", Some(celsius)) => Some(celsius + CELSIUS_TO_KELVIN),
                ("tlev", Some(_)) => Some(Real::from(self.diode.parameters.model.tlev)),
                ("tlevc", Some(_)) => Some(Real::from(self.diode.parameters.model.tlevc)),
                (_, value) => value,
            };
            model = model.with(keyword, value.unwrap_or(0.), value.is_some())?;
        }
        let mut instance = SensitivityRecord::new(self.family(), ParameterScope::Instance);
        for keyword in INSTANCE_FIELDS {
            let value = match *keyword {
                "ikf" | "ikr" | "ikp" => None,
                "ic" => self.diode.initial.values.first().copied().flatten(),
                "temp" => written(i, "temp").map(|t| t + CELSIUS_TO_KELVIN),
                other => written(i, other),
            };
            instance = instance.with(keyword, value.unwrap_or(0.), value.is_some())?;
        }
        self.setup(&mut model, &mut instance)?;
        self.temperature(&mut model, &mut instance)?;
        Ok((model, instance))
    }

    /// `DIOmAsk` reads `is` no lower than `CKTepsmin`; `DIOask` the
    /// temperature in Celsius.
    fn ask(
        &self,
        scope: ParameterScope,
        keyword: &str,
        records: RecordPair<'_>,
    ) -> SpiceResult<Real> {
        Ok(match (scope, keyword) {
            (ParameterScope::Model, "is") => records.model.value("is")?.max(EPSMIN),
            (ParameterScope::Model, _) => records.model.value(keyword)?,
            (ParameterScope::Instance, "temp") => {
                records.instance.value("temp")? - CELSIUS_TO_KELVIN
            }
            (ParameterScope::Instance, _) => records.instance.value(keyword)?,
        })
    }

    /// `DIOmParam`/`DIOparam`: `xom`/`xoi` are Angstrom, `temp` Celsius.
    fn set(
        &self,
        scope: ParameterScope,
        keyword: &str,
        value: Real,
        model: &mut SensitivityRecord,
        instance: &mut SensitivityRecord,
    ) -> SpiceResult<()> {
        match (scope, keyword) {
            (ParameterScope::Model, "xom" | "xoi") => model.set(keyword, value * 1e-10),
            (ParameterScope::Model, _) => model.set(keyword, value),
            (ParameterScope::Instance, "temp") => instance.set("temp", value + CELSIUS_TO_KELVIN),
            (ParameterScope::Instance, _) => instance.set(keyword, value),
        }
    }

    /// `DIOsetup` (level 1, no `scale` option).
    fn setup(
        &self,
        model: &mut SensitivityRecord,
        instance: &mut SensitivityRecord,
    ) -> SpiceResult<()> {
        for parameter in super::MODEL.parameters {
            if let Some(value) = parameter.default {
                default(model, record_key(parameter.name), value)?;
            }
        }
        for knee in ["ikf", "ikr", "ikp"] {
            if model.given(knee) && model.value(knee)? < EPSMIN {
                model.set_given(knee, false)?;
            }
        }
        let n = model.value("n")?;
        default(model, "nbv", n)?;
        default(model, "tlev", 0.)?;
        default(model, "tlevc", 0.)?;
        let tlev = if matches!(model.value("tlev")?, 2.) {
            2
        } else {
            0
        };
        default(model, "eg", default_eg(tlev))?;
        // Disabled physics still has C ask/set defaults in the sensitivity table.
        for (keyword, value) in [
            ("isr", 1e-14),
            ("jsw", 0.),
            ("jtun", 0.),
            ("jtunsw", 0.),
            ("vp", 0.),
            ("lm", 0.),
            ("lp", 0.),
            ("wm", 0.),
            ("wp", 0.),
            ("xom", 1e4),
            ("xoi", 1e4),
            ("xm", 0.),
            ("xp", 0.),
            ("xw", 0.),
        ] {
            default(model, keyword, value)?;
        }
        if !model.given("qpscale") || model.value("qpscale")? <= 0. {
            model.store("qpscale", 1e6)?;
        }
        if model.value("is")? < EPSMIN {
            model.store("is", EPSMIN)?;
        }
        default(model, "tnom", self.nominal_kelvin)?;
        let rs = model.value("rs")?;
        let conductance = DiodeSetup::conductance(if model.given("rs") { rs } else { 0. });
        model.store("conductance", conductance)?;
        setup_diode_instance(model, instance)
    }

    /// `DIOtemp`: the instance temperature (`CKTtemp + dtemp` unless given).
    fn temperature(
        &self,
        _model: &mut SensitivityRecord,
        instance: &mut SensitivityRecord,
    ) -> SpiceResult<()> {
        if !instance.given("dtemp") {
            instance.store("dtemp", 0.)?;
        }
        if !instance.given("temp") {
            let dtemp = instance.value("dtemp")?;
            instance.store("temp", self.circuit_kelvin + dtemp)?;
        }
        Ok(())
    }

    fn load(
        &self,
        _setup: RecordPair<'_>,
        current: RecordPair<'_>,
        mode: SensitivityMode,
    ) -> SpiceResult<SensitivityLoad> {
        if mode == SensitivityMode::Ac {
            return Err(SpiceError::circuit(
                "a diode's AC sensitivity load is its reset load",
            ));
        }
        Ok(SensitivityLoad::Replacement(Box::new(self.diode(current)?)))
    }

    /// `DIOacLoad` after `cktsens.c`'s reset load: the series conductance
    /// `DIOtConductance` of `current`, the junction conductance
    /// `CKTstate0[DIOconduct]` that `dioload.c` stored at `vd = 0` with the
    /// `state` records, and no capacitance (`DIOcapCurrent` is written only by
    /// a small-signal load, so it is still zero).
    fn reset_ac_load(
        &self,
        _setup: RecordPair<'_>,
        reset: AcReset<'_>,
        current: RecordPair<'_>,
    ) -> SpiceResult<SensitivityLoad> {
        Ok(SensitivityLoad::Replacement(Box::new(ResetDiode {
            state: self.diode(reset.state)?,
            current: self.diode(current)?,
        })))
    }
}

impl DiodeSensitivity<'_> {
    /// The port's diode with the records' parameters.
    fn diode(&self, current: RecordPair<'_>) -> SpiceResult<Diode> {
        let (m, i) = (current.model, current.instance);
        let selector = |key: &str, max: u8| -> SpiceResult<u8> {
            let value = m.value(key)?;
            crate::primitives::c_integer(value)
                .and_then(|rounded| u8::try_from(rounded).ok())
                .filter(|&rounded| rounded <= max)
                .ok_or_else(|| SpiceError::circuit(format!("diode {key}={value} in .sens")))
        };
        let instance = DiodeInstanceParameters {
            geometry: DiodeGeometry {
                area: i.value("area")?,
                perimeter: i.value("pj")?,
                multiplier: i.value("m")?,
            },
            temperature: Some(i.value("temp")?),
            dtemp: 0.,
        };
        let parameters = DiodeParameters::from_defaulted_model(
            |key| m.value(record_key(key)),
            |key| {
                m.given(record_key(key))
                    .then(|| m.value(record_key(key)))
                    .transpose()
            },
            [selector("tlev", 2)?, selector("tlevc", 1)?],
            instance,
            DiodeSetup {
                conductance: m.value("conductance")?,
                knees: [i.value("ikf")?, i.value("ikr")?, i.value("ikp")?],
            },
            ["ikf", "ikr", "ikp"].map(|key| m.given(key)),
            Some(m.value("tnom")?),
        )?;
        parameters.check(&self.diode.location)?;
        Ok(Diode {
            name: self.diode.name.clone(),
            model: self.diode.model.clone(),
            terminals: self.diode.terminals.clone(),
            junction: self.diode.junction,
            parameters,
            derived: TemperatureCache::new(),
            initial: self.diode.initial.clone(),
            location: self.diode.location.clone(),
            written: self.diode.written.clone(),
            lenient: true,
        })
    }
}

/// A diode's AC load in `cktsens.c`'s sweep
/// ([`crate::devices::sensitivity::DeviceSensitivity::reset_ac_load`]).
#[derive(Debug)]
struct ResetDiode {
    /// Built from the records of the reset load, which sets the junction
    /// conductance.
    state: Diode,
    /// Built from the current records, which set the series conductance.
    current: Diode,
}

impl Device for ResetDiode {
    fn name(&self) -> &str {
        &self.current.name
    }

    fn designator(&self) -> char {
        'd'
    }

    fn terminals(&self) -> &[NodeId] {
        &self.current.terminals
    }

    fn stamp(&self, _context: &mut StampContext<'_>) -> SpiceResult<()> {
        Err(SpiceError::circuit(format!(
            "{}: an AC sensitivity stand-in has no DC load",
            self.current.name
        )))
    }

    /// `dioacld.c`'s stamps in its order: `gspr` and `geq` (the `xceq`
    /// imaginary parts are zero).
    fn assemble_linear(&self, context: &mut LinearContext<'_>) -> SpiceResult<()> {
        let model = context.model_context;
        let gspr = self.current.thermal(model)?.conductance;
        let geq = self
            .state
            .thermal(model)?
            .point_unchecked(0., model.gmin)?
            .ac_conductance;
        let ([pos, neg], prime) = (
            [self.current.terminals[0], self.current.terminals[1]],
            self.current.junction[0],
        );
        let stamps = [
            (pos, pos, gspr),
            (neg, neg, geq),
            (prime, prime, geq + gspr),
            (pos, prime, -gspr),
            (neg, prime, -geq),
            (prime, pos, -gspr),
            (prime, neg, -geq),
        ];
        let stamps = stamps
            .into_iter()
            .map(|(a, b, value)| ResetStamp {
                at: StampAt::Nodes(a, b),
                value,
                part: StampPart::Real,
            })
            .collect();
        ResetStamps::new(&self.current.name, 'd', &self.current.terminals, stamps)?
            .assemble_linear(context)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::devices::{ModelContext, initial::InstanceInitial};
    use crate::primitives::{NodeTable, SourceLoc};
    use std::path::PathBuf;

    fn diode(model: &[(&str, Real)], instance: &[(&str, Real)], context: &ModelContext) -> Diode {
        let location = SourceLoc::new(PathBuf::from("diode-setup.cir"), 1, 1);
        let m = super::super::MODEL
            .validate_numeric(
                model.iter().map(|&(key, value)| (key, value, &location)),
                &location,
            )
            .unwrap();
        let i = super::super::INSTANCE
            .validate_numeric(
                instance.iter().map(|&(key, value)| (key, value, &location)),
                &location,
            )
            .unwrap();
        let input = super::super::DiodeInput::prepare(
            "d1",
            "dm",
            &location,
            m,
            i,
            InstanceInitial {
                off: false,
                values: vec![None],
            },
            context,
        )
        .unwrap();
        NodeTable::new()
            .transaction(|nodes| {
                let anode = nodes.intern("a")?;
                input.bind(nodes, [anode, NodeId::GROUND])
            })
            .unwrap()
    }

    #[test]
    fn setup_quantities_are_not_reconstructed_during_replay() {
        let context = ModelContext::new(79., 27.);
        let diode = diode(
            &[
                ("rs", 7.3),
                ("trs", 0.003),
                ("ikf", 1.37e-3),
                ("ikr", 2.19e-3),
                ("ikp", 1.23e-4),
                ("area", 2.3),
                ("pj", 1.2),
                ("n", 1.3),
                ("tlev", 1.7),
                ("tlevc", 0.7),
            ],
            &[("area", 3.7), ("pj", 0.), ("m", 2.1), ("dtemp", 0.13)],
            &context,
        );
        let kernel = DiodeSensitivity::new(&diode, &context);
        let (mut model, mut instance) = kernel.records().unwrap();
        let ordinary = diode.thermal(&context).unwrap();
        let replay = kernel
            .diode(RecordPair {
                model: &model,
                instance: &instance,
            })
            .unwrap();
        let original = replay.thermal(&context).unwrap();
        assert_eq!(ordinary.vt.to_bits(), original.vt.to_bits());
        assert_eq!(ordinary.csat.to_bits(), original.csat.to_bits());
        assert_eq!(
            ordinary.conductance.to_bits(),
            original.conductance.to_bits()
        );
        for (ordinary, replay) in [
            (ordinary.ikf, original.ikf),
            (ordinary.ikr, original.ikr),
            (ordinary.ikp, original.ikp),
        ] {
            assert_eq!(ordinary.map(Real::to_bits), replay.map(Real::to_bits));
        }
        assert_eq!(original.ikp.map(Real::to_bits), Some(0_f64.to_bits()));
        assert_eq!(replay.parameters.model.tlev, 2);
        assert_eq!(replay.parameters.model.tlevc, 1);
        assert_eq!(replay.parameters.model.nbv.to_bits(), 1.3_f64.to_bits());
        assert_eq!(replay.parameters.model.eg.to_bits(), 1.16_f64.to_bits());
        let saved = replay.parameters.setup;
        for (scope, keyword, value) in [
            (ParameterScope::Model, "rs", 0.),
            (ParameterScope::Model, "ikf", 0.1),
            (ParameterScope::Instance, "area", 2.7),
            (ParameterScope::Instance, "pj", 1.4),
            (ParameterScope::Instance, "m", 4.1),
        ] {
            kernel
                .set(scope, keyword, value, &mut model, &mut instance)
                .unwrap();
        }
        kernel.temperature(&mut model, &mut instance).unwrap();
        let changed = kernel
            .diode(RecordPair {
                model: &model,
                instance: &instance,
            })
            .unwrap();
        let thermal = changed.thermal(&context).unwrap();
        assert_eq!(
            changed.parameters.setup.conductance.to_bits(),
            saved.conductance.to_bits()
        );
        assert_eq!(
            changed.parameters.setup.knees.map(Real::to_bits),
            saved.knees.map(Real::to_bits)
        );
        // When RS is set to zero, C skips the temperature divisor but retains
        // DIOconductance from the last setup. Geometry is temperature-time.
        assert_eq!(
            thermal.conductance.to_bits(),
            (saved.conductance * 2.7 * 4.1).to_bits()
        );
        assert_eq!(
            thermal.ikf.map(Real::to_bits),
            Some(saved.knees[0].to_bits())
        );
        assert_eq!(thermal.ikp.map(Real::to_bits), Some(0_f64.to_bits()));

        kernel.setup(&mut model, &mut instance).unwrap();
        kernel.temperature(&mut model, &mut instance).unwrap();
        let refreshed = kernel
            .diode(RecordPair {
                model: &model,
                instance: &instance,
            })
            .unwrap();
        assert_eq!(
            refreshed.parameters.setup.conductance.to_bits(),
            0_f64.to_bits()
        );
        assert_eq!(
            refreshed.parameters.setup.knees[0].to_bits(),
            (0.1_f64 * 2.7 * 4.1).to_bits()
        );
        assert_eq!(
            refreshed.parameters.setup.knees[2].to_bits(),
            (1.23e-4_f64 * 1.4 * 4.1).to_bits()
        );
    }

    #[test]
    fn setup_floor_and_disabled_knee_keep_distinct_given_state() {
        let context = ModelContext::default();
        let diode = diode(
            &[("is", 1e-30), ("ikf", 1e-30)],
            &[("area", 2.), ("m", 3.)],
            &context,
        );
        let kernel = DiodeSensitivity::new(&diode, &context);
        let (mut model, mut instance) = kernel.records().unwrap();
        assert_eq!(model.value("is").unwrap().to_bits(), EPSMIN.to_bits());
        assert!(!model.given("ikf"));
        let saved = instance.value("ikf").unwrap();
        assert_eq!(saved.to_bits(), (1e-30_f64 * 2. * 3.).to_bits());
        kernel
            .set(ParameterScope::Model, "ikf", 0.1, &mut model, &mut instance)
            .unwrap();
        let replay = kernel
            .diode(RecordPair {
                model: &model,
                instance: &instance,
            })
            .unwrap();
        assert_eq!(
            replay.thermal(&context).unwrap().ikf.map(Real::to_bits),
            Some(saved.to_bits())
        );
    }
}
