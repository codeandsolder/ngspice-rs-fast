//! Resistors under `.sens`: `res.c` tables, `resparam.c`/`resmpar.c`
//! setters, `ressetup.c` defaults, `restemp.c` (`RESupdate_conduct`) and
//! `resload.c` (`RESload`, `RESacload`).

use super::{
    CELSIUS_TO_KELVIN, ParameterScope, PassiveInputs, RecordPair, SensitivityConductance,
    SensitivityFamily, SensitivityKernel, SensitivityParameter as P, SensitivityRecord,
    reconstructed_loads,
};
use crate::devices::passive::physics::{
    Coefficients, SheetGeometry, TemperaturePoint, conductance, scaled,
};
use crate::primitives::{NodeId, Real, SpiceResult};

/// `RESmPTable` entries `set_param()` accepts (`defw`, `l` and `tnom` are
/// `IF_NONSENSE`, `dw`/`dlr`/`tc1r`/`tc2r`/`res` redundant).
const MODEL: &[P] = &[
    P::dc("rsh"),
    P::dc("narrow"),
    P::dc("short"),
    P::dc("tc1"),
    P::dc("tc2"),
    P::dc("tce"),
    P::dc("kf"),
    P::dc("af"),
    P::dc("r"),
    P::dc("bv_max"),
    P::dc("lf"),
    P::dc("wf"),
    P::dc("ef"),
];

/// `RESpTable` entries `set_param()` accepts (`noisy` is an integer).
const INSTANCE: &[P] = &[
    P::principal("resistance", false),
    P::ac("ac"),
    P::dc("temp"),
    P::dc("dtemp"),
    P::dc("l"),
    P::dc("w"),
    P::dc("m"),
    P::dc("tc"),
    P::dc("tc2"),
    P::dc("tce"),
    P::dc("bv_max"),
    P::dc("scale"),
];

/// A resistor's `.sens` description, borrowing the device's immutable card.
#[derive(Debug)]
pub(in crate::devices) struct ResistorSensitivity<'a> {
    name: &'a str,
    terminals: [NodeId; 2],
    /// `CKTtemp`, `CKTnomTemp` in kelvin.
    circuit_kelvin: Real,
    nominal_kelvin: Real,
    inputs: PassiveInputs<'a>,
}

impl<'a> ResistorSensitivity<'a> {
    pub(crate) fn new(
        name: &'a str,
        terminals: [NodeId; 2],
        inputs: PassiveInputs<'a>,
        context: &crate::devices::ModelContext,
    ) -> Self {
        Self {
            name,
            terminals,
            circuit_kelvin: context.temperature + CELSIUS_TO_KELVIN,
            nominal_kelvin: context.nominal_temperature + CELSIUS_TO_KELVIN,
            inputs,
        }
    }
}

fn field(value: Option<Real>, default: Real) -> (Real, bool) {
    value.map_or((default, false), |value| (value, true))
}

impl SensitivityKernel for ResistorSensitivity<'_> {
    fn family(&self) -> SensitivityFamily {
        SensitivityFamily::Resistor
    }

    fn model(&self) -> Option<&str> {
        self.inputs.model_name()
    }

    fn model_parameters(&self) -> &'static [P] {
        MODEL
    }

    fn instance_parameters(&self) -> &'static [P] {
        INSTANCE
    }

    fn records(&self) -> SpiceResult<(SensitivityRecord, SensitivityRecord)> {
        let i = &self.inputs;
        let instance_value = |keyword| i.instance_value("resistance", keyword);
        let entry = |record: SensitivityRecord, keyword, value: Option<Real>, default| {
            let (value, given) = field(value, default);
            record.with(keyword, value, given)
        };
        let mut model = SensitivityRecord::new(self.family(), ParameterScope::Model);
        let (tnom, tnom_given) = field(i.model_value("tnom").map(|t| t + CELSIUS_TO_KELVIN), 0.);
        model = model.with("tnom", tnom, tnom_given)?;
        model = entry(model, "rsh", i.model_value("rsh"), 0.)?;
        model = entry(model, "defw", i.model_value("defw"), 0.)?;
        model = entry(model, "defl", i.model_value("l"), 0.)?;
        model = entry(model, "narrow", i.model_value("narrow"), 0.)?;
        model = entry(model, "short", i.model_value("short"), 0.)?;
        model = entry(model, "tc1", i.model_value("tc1"), 0.)?;
        model = entry(model, "tc2", i.model_value("tc2"), 0.)?;
        model = entry(model, "tce", i.model_value("tce"), 0.)?;
        model = entry(model, "kf", i.model_value("kf"), 0.)?;
        model = entry(model, "af", i.model_value("af"), 0.)?;
        model = entry(model, "r", i.model_value("r"), 0.)?;
        model = entry(model, "bv_max", None, 0.)?;
        model = entry(model, "lf", i.model_value("lf"), 0.)?;
        model = entry(model, "wf", i.model_value("wf"), 0.)?;
        model = entry(model, "ef", i.model_value("ef"), 0.)?;
        let mut instance = SensitivityRecord::new(self.family(), ParameterScope::Instance);
        instance = entry(instance, "resistance", instance_value("resistance"), 0.)?;
        instance = entry(instance, "ac", instance_value("ac"), 0.)?;
        let (temp, temp_given) = field(instance_value("temp").map(|t| t + CELSIUS_TO_KELVIN), 0.);
        instance = instance.with("temp", temp, temp_given)?;
        instance = entry(instance, "dtemp", instance_value("dtemp"), 0.)?;
        instance = entry(instance, "l", instance_value("l"), 0.)?;
        instance = entry(instance, "w", instance_value("w"), 0.)?;
        instance = entry(instance, "m", instance_value("m"), 0.)?;
        instance = entry(instance, "tc", instance_value("tc1"), 0.)?;
        instance = entry(instance, "tc2", instance_value("tc2"), 0.)?;
        instance = entry(instance, "tce", instance_value("tce"), 0.)?;
        instance = entry(instance, "bv_max", None, 0.)?;
        instance = entry(instance, "scale", instance_value("scale"), 0.)?;
        instance = instance
            .with("conduct", 0., false)?
            .with("acconduct", 0., false)?;
        self.setup(&mut model, &mut instance)?;
        self.temperature(&mut model, &mut instance)?;
        Ok((model, instance))
    }

    /// `RESmAsk`: `kf`/`af` read 0 unless given; `RESask`: `temp` in Celsius.
    fn ask(
        &self,
        scope: ParameterScope,
        keyword: &str,
        records: RecordPair<'_>,
    ) -> SpiceResult<Real> {
        match (scope, keyword) {
            (ParameterScope::Model, "kf" | "af") => Ok(if records.model.given(keyword) {
                records.model.value(keyword)?
            } else {
                0.
            }),
            (ParameterScope::Model, _) => records.model.value(keyword),
            (ParameterScope::Instance, "temp") => {
                Ok(records.instance.value("temp")? - CELSIUS_TO_KELVIN)
            }
            (ParameterScope::Instance, "ac") => records.instance.value("ac"),
            (ParameterScope::Instance, _) => records.instance.value(keyword),
        }
    }

    /// `RESparam`/`RESmParam`: `temp` from Celsius (below 1e-6 K reads 0),
    /// the resistance clamped away from zero (`RESMIN`), and a model `r`
    /// only when positive.
    fn set(
        &self,
        scope: ParameterScope,
        keyword: &str,
        value: Real,
        model: &mut SensitivityRecord,
        instance: &mut SensitivityRecord,
    ) -> SpiceResult<()> {
        match (scope, keyword) {
            (ParameterScope::Model, "r") => {
                if value > 0. {
                    model.set("r", value)?;
                }
                Ok(())
            }
            (ParameterScope::Model, _) => model.set(keyword, value),
            (ParameterScope::Instance, "temp") => {
                let kelvin = value + CELSIUS_TO_KELVIN;
                instance.set("temp", if kelvin < 1e-6 { 0. } else { kelvin })
            }
            (ParameterScope::Instance, "resistance") => {
                instance.set("resistance", crate::devices::rlc::clamp_resistance(value))
            }
            (ParameterScope::Instance, _) => instance.set(keyword, value),
        }
    }

    /// `RESsetup`'s defaults.
    fn setup(
        &self,
        model: &mut SensitivityRecord,
        instance: &mut SensitivityRecord,
    ) -> SpiceResult<()> {
        for (keyword, default) in [
            ("tnom", self.nominal_kelvin),
            ("rsh", 0.),
            ("defw", 10e-6),
            ("defl", 10e-6),
            ("tc1", 0.),
            ("tc2", 0.),
            ("tce", 0.),
            ("narrow", 0.),
            ("short", 0.),
            ("kf", 0.),
            ("af", 1.),
            ("lf", 1.),
            ("wf", 1.),
            ("ef", 1.),
            ("bv_max", 1e99),
        ] {
            if !model.given(keyword) {
                model.store(keyword, default)?;
            }
        }
        if !instance.given("w") {
            instance.store("w", model.value("defw")?)?;
        }
        if !instance.given("l") {
            instance.store("l", model.value("defl")?)?;
        }
        if !instance.given("scale") {
            instance.store("scale", 1.)?;
        }
        if !instance.given("m") {
            instance.store("m", 1.)?;
        }
        if !instance.given("bv_max") {
            instance.store("bv_max", model.value("bv_max")?)?;
        }
        Ok(())
    }

    /// `REStemp` and `RESupdate_conduct`.
    fn temperature(
        &self,
        model: &mut SensitivityRecord,
        instance: &mut SensitivityRecord,
    ) -> SpiceResult<()> {
        let temperature = TemperaturePoint::resolve(
            instance
                .given("temp")
                .then(|| instance.value("temp"))
                .transpose()?,
            instance
                .given("dtemp")
                .then(|| instance.value("dtemp"))
                .transpose()?,
            self.circuit_kelvin,
        );
        instance.store("temp", temperature.kelvin)?;
        instance.store("dtemp", temperature.offset)?;
        if !instance.given("resistance") {
            let geometry = SheetGeometry {
                length: instance.value("l")?,
                width: instance.value("w")?,
                short: model.value("short")?,
                narrow: model.value("narrow")?,
                sheet: model.value("rsh")?,
            };
            let resist = if geometry.uses_sheet() {
                geometry.resistance()
            } else if model.given("r") {
                model.value("r")?
            } else {
                1e-3
            };
            instance.store("resistance", resist)?;
        }
        let difference = temperature.difference(model.value("tnom")?);
        let pick = |instance_key: &str, model_key: &str| -> SpiceResult<Real> {
            if instance.given(instance_key) {
                instance.value(instance_key)
            } else {
                model.value(model_key)
            }
        };
        let coefficients = Coefficients {
            linear: pick("tc", "tc1")?,
            quadratic: pick("tc2", "tc2")?,
        };
        let exponential = (instance.given("tce") || model.given("tce"))
            .then(|| pick("tce", "tce"))
            .transpose()?;
        let factor = coefficients.resistor_factor(difference, exponential);
        if !instance.given("scale") {
            instance.store("scale", 1.)?;
        }
        let (m, scale) = (instance.value("m")?, instance.value("scale")?);
        let conduct = conductance(scaled(instance.value("resistance")?, factor, scale), m);
        instance.store("conduct", conduct)?;
        if instance.given("ac") {
            instance.store(
                "acconduct",
                conductance(scaled(instance.value("ac")?, factor, scale), m),
            )?;
        } else {
            instance.store("acconduct", conduct)?;
            instance.store("ac", instance.value("resistance")?)?;
        }
        Ok(())
    }

    reconstructed_loads!(
        ordinary,
        /// `RESload` stamps `RESconduct`; `RESacload` `RESacConduct` when `ac`
        /// was given, else `RESconduct`.
        replacement
    );
}

impl ResistorSensitivity<'_> {
    fn replacement(&self, current: RecordPair<'_>) -> SpiceResult<SensitivityConductance> {
        let conduct = current.instance.value("conduct")?;
        let ac = if current.instance.given("ac") {
            current.instance.value("acconduct")?
        } else {
            conduct
        };
        SensitivityConductance::new(self.name, 'r', self.terminals, conduct, ac)
    }
}
