//! Capacitors and inductors under `.sens`: `cap.c`/`ind.c` tables, the
//! setters of `capparam.c`/`capmpar.c`/`indparam.c`/`indmpar.c`, the defaults
//! of `capsetup.c`/`indsetup.c`, `captemp.c`/`indtemp.c` and the AC loads
//! `capacld.c`/`indacld.c`. Their DC loads do not depend on any parameter (a
//! capacitor stamps nothing at the operating point, an inductor a short), so
//! every DC sensitivity of theirs is exactly zero.

use super::{
    CELSIUS_TO_KELVIN, ParameterScope, PassiveInputs, RecordPair, SensitivityFamily,
    SensitivityKernel, SensitivityLoad, SensitivityMode, SensitivityParameter as P,
    SensitivityRecord,
};
use crate::devices::passive::physics::{
    CapacitorGeometry, CapacitorSetup, Coefficients, CoilGeometry, CoilSetup, TemperaturePoint,
    scaled,
};
use crate::devices::{Device, LinearContext, StampContext};
use crate::primitives::{NodeId, Real, SpiceError, SpiceResult};

/// `CAPmPTable` entries `set_param()` accepts.
const CAP_MODEL: &[P] = &[
    P::ac("cap"),
    P::ac("cj"),
    P::ac("cjsw"),
    P::ac("narrow"),
    P::ac("short"),
    P::ac("del"),
    P::ac("tc1"),
    P::ac("tc2"),
    P::ac("di"),
    P::ac("thick"),
    P::dc("bv_max"),
];

/// `CAPpTable` entries `set_param()` accepts.
const CAP_INSTANCE: &[P] = &[
    P::principal("capacitance", true),
    P::ac("ic"),
    P::dc("temp"),
    P::dc("dtemp"),
    P::ac("w"),
    P::ac("l"),
    P::dc("m"),
    P::dc("tc1"),
    P::dc("tc2"),
    P::dc("bv_max"),
    P::dc("scale"),
];

/// `INDmPTable` entries `set_param()` accepts (`tnom` is `IF_NONSENSE`).
const IND_MODEL: &[P] = &[
    P::ac("ind"),
    P::ac("tc1"),
    P::ac("tc2"),
    P::ac("csect"),
    P::ac("dia"),
    P::ac("length"),
    P::ac("nt"),
    P::ac("mu"),
];

/// `INDpTable` entries `set_param()` accepts.
const IND_INSTANCE: &[P] = &[
    P::principal("inductance", true),
    P::ac("ic"),
    P::dc("temp"),
    P::dc("dtemp"),
    P::dc("m"),
    P::dc("tc1"),
    P::dc("tc2"),
    P::dc("scale"),
    P::dc("nt"),
];

/// Which reactive element.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::devices) enum ReactiveKind {
    /// A capacitor.
    Capacitor,
    /// An inductor.
    Inductor,
}

/// A capacitor's or inductor's `.sens` description.
#[derive(Debug)]
pub(in crate::devices) struct ReactiveSensitivity<'a> {
    kind: ReactiveKind,
    name: &'a str,
    terminals: [NodeId; 2],
    circuit_kelvin: Real,
    nominal_kelvin: Real,
    inputs: PassiveInputs<'a>,
    hard_branch: bool,
}

impl<'a> ReactiveSensitivity<'a> {
    pub(crate) fn new(
        kind: ReactiveKind,
        name: &'a str,
        terminals: [NodeId; 2],
        inputs: PassiveInputs<'a>,
        context: &crate::devices::ModelContext,
        hard_branch: bool,
    ) -> Self {
        Self {
            kind,
            name,
            terminals,
            circuit_kelvin: context.temperature + CELSIUS_TO_KELVIN,
            nominal_kelvin: context.nominal_temperature + CELSIUS_TO_KELVIN,
            inputs,
            hard_branch,
        }
    }

    const fn fields(&self) -> (&'static [&'static str], &'static [&'static str]) {
        match self.kind {
            ReactiveKind::Capacitor => (
                &[
                    "tnom", "cap", "cj", "cjsw", "defw", "defl", "narrow", "short", "del", "tc1",
                    "tc2", "di", "thick", "bv_max",
                ],
                &[
                    "capacitance",
                    "ic",
                    "temp",
                    "dtemp",
                    "w",
                    "l",
                    "m",
                    "tc1",
                    "tc2",
                    "bv_max",
                    "scale",
                    "capac",
                ],
            ),
            ReactiveKind::Inductor => (
                &[
                    "tnom", "ind", "tc1", "tc2", "csect", "dia", "length", "nt", "mu", "spec",
                ],
                &[
                    "inductance",
                    "ic",
                    "temp",
                    "dtemp",
                    "m",
                    "tc1",
                    "tc2",
                    "scale",
                    "nt",
                    "induct",
                ],
            ),
        }
    }
}

impl SensitivityKernel for ReactiveSensitivity<'_> {
    fn family(&self) -> SensitivityFamily {
        match self.kind {
            ReactiveKind::Capacitor => SensitivityFamily::Capacitor,
            ReactiveKind::Inductor => SensitivityFamily::Inductor,
        }
    }

    fn model(&self) -> Option<&str> {
        self.inputs.model_name()
    }

    fn model_parameters(&self) -> &'static [P] {
        match self.kind {
            ReactiveKind::Capacitor => CAP_MODEL,
            ReactiveKind::Inductor => IND_MODEL,
        }
    }

    fn instance_parameters(&self) -> &'static [P] {
        match self.kind {
            ReactiveKind::Capacitor => CAP_INSTANCE,
            ReactiveKind::Inductor => IND_INSTANCE,
        }
    }

    fn records(&self) -> SpiceResult<(SensitivityRecord, SensitivityRecord)> {
        let (model_fields, instance_fields) = self.fields();
        let principal = match self.kind {
            ReactiveKind::Capacitor => "capacitance",
            ReactiveKind::Inductor => "inductance",
        };
        let build = |fields: &[&'static str], scope| {
            let mut record = SensitivityRecord::new(self.family(), scope);
            for keyword in fields {
                let value = match scope {
                    ParameterScope::Model => self.inputs.model_value(keyword),
                    ParameterScope::Instance => self.inputs.instance_value(principal, keyword),
                };
                let value = match (*keyword, value) {
                    ("temp" | "tnom", Some(celsius)) => Some(celsius + CELSIUS_TO_KELVIN),
                    (_, value) => value,
                };
                record = record.with(keyword, value.unwrap_or(0.), value.is_some())?;
            }
            Ok::<_, SpiceError>(record)
        };
        let mut model = build(model_fields, ParameterScope::Model)?;
        let mut instance = build(instance_fields, ParameterScope::Instance)?;
        // The instance value is kept as C's `CAPcapacinst`/`INDinductinst`.
        self.setup(&mut model, &mut instance)?;
        self.temperature(&mut model, &mut instance)?;
        Ok((model, instance))
    }

    /// `CAPask`: the capacitance is the temperature-adjusted `CAPcapac`
    /// times `m`; `INDask`: the inductance is `INDinduct`; `temp` in Celsius.
    fn ask(
        &self,
        scope: ParameterScope,
        keyword: &str,
        records: RecordPair<'_>,
    ) -> SpiceResult<Real> {
        let i = records.instance;
        match (scope, keyword) {
            (ParameterScope::Model, _) => records.model.value(keyword),
            (ParameterScope::Instance, "capacitance") => Ok(i.value("capac")? * i.value("m")?),
            (ParameterScope::Instance, "inductance") => i.value("induct"),
            (ParameterScope::Instance, "temp") => Ok(i.value("temp")? - CELSIUS_TO_KELVIN),
            (ParameterScope::Instance, _) => i.value(keyword),
        }
    }

    /// The setters: `temp` from Celsius; the instance value sets
    /// `CAPcapacinst`/`INDinductinst` (and defaults `m`).
    fn set(
        &self,
        scope: ParameterScope,
        keyword: &str,
        value: Real,
        model: &mut SensitivityRecord,
        instance: &mut SensitivityRecord,
    ) -> SpiceResult<()> {
        match (scope, keyword) {
            (ParameterScope::Model, _) => model.set(keyword, value),
            (ParameterScope::Instance, "temp") => instance.set("temp", value + CELSIUS_TO_KELVIN),
            (ParameterScope::Instance, "capacitance" | "inductance") => {
                instance.set(keyword, value)?;
                if !instance.given("m") {
                    instance.store("m", 1.)?;
                }
                Ok(())
            }
            (ParameterScope::Instance, _) => instance.set(keyword, value),
        }
    }

    /// `CAPsetup`/`INDsetup`: model defaults and the quantities only setup
    /// derives (a capacitor's `cj` from `di`/`thick`, `narrow`/`short` from
    /// `del`; an inductor's specific inductance and default `ind`).
    fn setup(
        &self,
        model: &mut SensitivityRecord,
        instance: &mut SensitivityRecord,
    ) -> SpiceResult<()> {
        let default = |record: &mut SensitivityRecord, keyword: &str, value: Real| {
            if record.given(keyword) {
                Ok(())
            } else {
                record.store(keyword, value)
            }
        };
        match self.kind {
            ReactiveKind::Capacitor => {
                for (keyword, value) in [
                    ("cap", 0.),
                    ("cjsw", 0.),
                    ("defw", 10e-6),
                    ("defl", 0.),
                    ("narrow", 0.),
                    ("short", 0.),
                    ("del", 0.),
                    ("tc1", 0.),
                    ("tc2", 0.),
                    ("tnom", self.nominal_kelvin),
                    ("di", 0.),
                    ("thick", 0.),
                    ("bv_max", 1e99),
                ] {
                    default(model, keyword, value)?;
                }
                let given = |key| model.given(key).then(|| model.value(key)).transpose();
                let geometry = CapacitorSetup {
                    cj: given("cj")?,
                    cjsw: model.value("cjsw")?,
                    narrow: given("narrow")?,
                    short: given("short")?,
                    del: given("del")?,
                    dielectric: given("di")?,
                    thickness: given("thick")?,
                }
                .prepare();
                model.store("cj", geometry.cj)?;
                model.store("narrow", geometry.narrow)?;
                model.store("short", geometry.short)?;
                default(instance, "l", 0.)?;
                let bv_max = model.value("bv_max")?;
                default(instance, "bv_max", bv_max)
            }
            ReactiveKind::Inductor => {
                for (keyword, value) in [
                    ("ind", 0.),
                    ("tnom", self.nominal_kelvin),
                    ("tc1", 0.),
                    ("tc2", 0.),
                    ("csect", 0.),
                    ("dia", 0.),
                    ("length", 0.),
                    ("nt", 0.),
                    ("mu", 1.),
                ] {
                    default(model, keyword, value)?;
                }
                let given = |key| model.given(key).then(|| model.value(key)).transpose();
                let geometry = CoilSetup {
                    inductance: given("ind")?,
                    cross_section: given("csect")?,
                    diameter: given("dia")?,
                    length: given("length")?,
                    turns: model.value("nt")?,
                    permeability: model.value("mu")?,
                }
                .prepare();
                model.store("csect", geometry.cross_section)?;
                model.store("spec", geometry.specific)?;
                model.store("ind", geometry.model_inductance)?;
                Ok(())
            }
        }
    }

    /// `CAPtemp`/`INDtemp`.
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
        if !instance.given("scale") {
            instance.store("scale", 1.)?;
        }
        if !instance.given("m") {
            instance.store("m", 1.)?;
        }
        let base = match self.kind {
            ReactiveKind::Capacitor => {
                if !instance.given("w") {
                    instance.store("w", model.value("defw")?)?;
                }
                if instance.given("capacitance") {
                    instance.value("capacitance")?
                } else if model.given("cap") {
                    model.value("cap")?
                } else {
                    CapacitorGeometry {
                        cj: model.value("cj")?,
                        cjsw: model.value("cjsw")?,
                        narrow: model.value("narrow")?,
                        short: model.value("short")?,
                    }
                    .capacitance(instance.value("l")?, instance.value("w")?)
                }
            }
            ReactiveKind::Inductor => {
                if !instance.given("nt") {
                    instance.store("nt", 0.)?;
                }
                let geometry = CoilGeometry {
                    cross_section: model.value("csect")?,
                    specific: model.value("spec")?,
                    model_inductance: model.value("ind")?,
                };
                geometry.inductance(
                    instance
                        .given("inductance")
                        .then(|| instance.value("inductance"))
                        .transpose()?,
                    instance
                        .given("nt")
                        .then(|| instance.value("nt"))
                        .transpose()?,
                )
            }
        };
        let difference = temperature.difference(model.value("tnom")?);
        let pick = |keyword: &str| -> SpiceResult<Real> {
            if instance.given(keyword) {
                instance.value(keyword)
            } else {
                model.value(keyword)
            }
        };
        let coefficients = Coefficients {
            linear: pick("tc1")?,
            quadratic: pick("tc2")?,
        };
        let factor = coefficients.reactive(difference);
        let value = scaled(base, factor, instance.value("scale")?);
        match self.kind {
            ReactiveKind::Capacitor => instance.store("capac", value),
            ReactiveKind::Inductor => instance.store("induct", value),
        }
    }

    /// DC: the device's own load (independent of every parameter). AC:
    /// `CAPacLoad` stamps `omega CAPcapac m`, `INDacLoad` `omega INDinduct /
    /// m` on the branch.
    fn load(
        &self,
        _setup: RecordPair<'_>,
        current: RecordPair<'_>,
        mode: SensitivityMode,
    ) -> SpiceResult<SensitivityLoad> {
        if mode == SensitivityMode::Dc {
            return Ok(SensitivityLoad::Original);
        }
        let i = current.instance;
        let m = i.value("m")?;
        let value = match self.kind {
            ReactiveKind::Capacitor => i.value("capac")?,
            ReactiveKind::Inductor => i.value("induct")?,
        };
        let effective = match self.kind {
            ReactiveKind::Capacitor => value * m,
            ReactiveKind::Inductor => value / m,
        };
        if !effective.is_finite() {
            return Err(SpiceError::Numerical {
                context: format!("sensitivity of {}", self.name),
                message: format!("nonfinite perturbed value {effective}"),
            });
        }
        Ok(SensitivityLoad::Replacement(Box::new(ReactiveStandIn {
            name: self.name.to_owned(),
            kind: self.kind,
            terminals: self.terminals,
            value,
            m,
            hard_branch: self.hard_branch,
        })))
    }
}

/// The AC stand-in of a perturbed capacitor or inductor.
#[derive(Debug)]
struct ReactiveStandIn {
    name: String,
    kind: ReactiveKind,
    terminals: [NodeId; 2],
    /// `CAPcapac` or `INDinduct`.
    value: Real,
    /// The instance multiplier.
    m: Real,
    hard_branch: bool,
}

impl Device for ReactiveStandIn {
    fn name(&self) -> &str {
        &self.name
    }

    fn designator(&self) -> char {
        match self.kind {
            ReactiveKind::Capacitor => 'c',
            ReactiveKind::Inductor => 'l',
        }
    }

    fn terminals(&self) -> &[NodeId] {
        &self.terminals
    }

    fn branch_currents(&self) -> usize {
        usize::from(self.kind == ReactiveKind::Inductor || self.hard_branch)
    }

    fn stamp(&self, _context: &mut StampContext<'_>) -> SpiceResult<()> {
        Err(SpiceError::circuit(format!(
            "{}: the sensitivity stand-in has no DC load",
            self.name
        )))
    }

    fn assemble_linear(&self, context: &mut LinearContext<'_>) -> SpiceResult<()> {
        match self.kind {
            ReactiveKind::Capacitor => {
                crate::devices::passive_bias::release_capacitor_branch(self.hard_branch, context)?;
                crate::devices::rlc::assemble_capacitor_ac(
                    self.terminals,
                    self.value,
                    self.m,
                    context,
                )
            }
            ReactiveKind::Inductor => crate::devices::rlc::assemble_inductor_ac(
                self.terminals,
                self.value,
                self.m,
                context,
            ),
        }
    }
}
