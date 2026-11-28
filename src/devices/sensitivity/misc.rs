//! Behavioural sources, switches and mutual inductance under `.sens`.
//!
//! * `asrc.c`: `temp`, `dtemp`, `tc1`, `tc2`, `m` (`asrcpar.c`,
//!   `asrcset.c`'s defaults, `asrctemp.c`, the `factor` of `asrcload.c`);
//! * `sw.c`/`csw.c`: the model's threshold, hysteresis and on/off
//!   resistances (`swmparam.c`/`cswmpar.c` store the conductances,
//!   `swsetup.c`/`cswsetup.c` default them; the switches have no
//!   temperature routine);
//! * `ind.c` (`MUTpTable`): the coupling `k` is `IF_AC`, so a K device has
//!   nothing to perturb in DC. In AC, `MUTask` reports `MUTfactor`
//!   (`M = k sqrt(|L1 L2|)`, `muttemp.c`) for `k`, so C's walk sets the
//!   coupling to `M + delta` and "restores" it to `M`.
//!
//! In an AC sweep these devices load what `cktsens.c`'s per-frequency reset
//! left ([`crate::devices::sensitivity::DeviceSensitivity::ac_load`]): a B source's `ASRCacValues`
//! are reallocated (zeroed) by every `ASRCsetup` and written only by a
//! small-signal load, so only a voltage source's branch incidences remain; a
//! switch keeps the state its `MODEINITFLOAT` load chose for a zero control.

use super::{
    AcReset, AcSensitivityLoad, CELSIUS_TO_KELVIN, DcSensitivityLoad, ParameterScope, RecordPair,
    ResetStamp, ResetStamps, SensitivityFamily, SensitivityKernel, SensitivityParameter as P,
    SensitivityRecord, StampAt, StampPart,
};
use crate::devices::{
    Behavioural, BehaviouralOutput, BehaviouralScale, Device, MutualInductance, Switch, SwitchKind,
};
use crate::primitives::{Real, SpiceResult};

/// Real stamps of `value` between nodes, in order.
fn real<const N: usize>(stamps: [(StampAt, Real); N]) -> [ResetStamp; N] {
    stamps.map(|(at, value)| ResetStamp {
        at,
        value,
        part: StampPart::Real,
    })
}

/// `ASRCpTable` entries `set_param()` accepts.
const ASRC: &[P] = &[
    P::dc("temp"),
    P::dc("dtemp"),
    P::dc("tc1"),
    P::dc("tc2"),
    P::dc("m"),
];

/// `SWmPTable` entries `set_param()` accepts.
const SW: &[P] = &[P::dc("vt"), P::dc("vh"), P::dc("ron"), P::dc("roff")];
/// `CSWmPTable` entries `set_param()` accepts.
const CSW: &[P] = &[P::dc("it"), P::dc("ih"), P::dc("ron"), P::dc("roff")];

/// A B source's `.sens` description.
#[derive(Debug)]
pub(in crate::devices) struct BehaviouralSensitivity<'a> {
    source: &'a Behavioural,
    circuit_kelvin: Real,
}

impl<'a> BehaviouralSensitivity<'a> {
    pub(crate) fn new(source: &'a Behavioural, context: &crate::devices::ModelContext) -> Self {
        Self {
            source,
            circuit_kelvin: context.temperature + CELSIUS_TO_KELVIN,
        }
    }
}

impl SensitivityKernel for BehaviouralSensitivity<'_> {
    fn family(&self) -> SensitivityFamily {
        SensitivityFamily::Asrc
    }

    fn model(&self) -> Option<&str> {
        None
    }

    fn model_parameters(&self) -> &'static [P] {
        &[]
    }

    fn instance_parameters(&self) -> &'static [P] {
        ASRC
    }

    fn records(&self) -> SpiceResult<(SensitivityRecord, SensitivityRecord)> {
        let scale = self.source.scale();
        let (temp, temp_given) = scale
            .temperature
            .map_or((0., false), |t| (t + CELSIUS_TO_KELVIN, true));
        let mut instance = SensitivityRecord::new(self.family(), ParameterScope::Instance)
            .with("temp", temp, temp_given)?
            .with("dtemp", scale.dtemp, scale.dtemp != 0.)?
            .with("tc1", scale.tc1, scale.tc1 != 0.)?
            .with("tc2", scale.tc2, scale.tc2 != 0.)?
            .with("m", scale.m, !matches!(scale.m, 1.))?;
        let mut model = SensitivityRecord::new(self.family(), ParameterScope::Model);
        self.temperature(&mut model, &mut instance)?;
        Ok((model, instance))
    }

    fn ask(
        &self,
        scope: ParameterScope,
        keyword: &str,
        records: RecordPair<'_>,
    ) -> SpiceResult<Real> {
        let value = records.instance.value(keyword)?;
        Ok(match (scope, keyword) {
            (ParameterScope::Instance, "temp") => value - CELSIUS_TO_KELVIN,
            _ => value,
        })
    }

    fn set(
        &self,
        _scope: ParameterScope,
        keyword: &str,
        value: Real,
        _model: &mut SensitivityRecord,
        instance: &mut SensitivityRecord,
    ) -> SpiceResult<()> {
        if keyword == "temp" {
            instance.set("temp", value + CELSIUS_TO_KELVIN)
        } else {
            instance.set(keyword, value)
        }
    }

    /// `ASRCsetup`'s defaults.
    fn setup(
        &self,
        _model: &mut SensitivityRecord,
        instance: &mut SensitivityRecord,
    ) -> SpiceResult<()> {
        for (keyword, value) in [("tc1", 0.), ("tc2", 0.), ("m", 1.)] {
            if !instance.given(keyword) {
                instance.store(keyword, value)?;
            }
        }
        Ok(())
    }

    /// `ASRCtemp`.
    fn temperature(
        &self,
        _model: &mut SensitivityRecord,
        instance: &mut SensitivityRecord,
    ) -> SpiceResult<()> {
        if instance.given("temp") {
            instance.store("dtemp", 0.)
        } else {
            instance.store("temp", self.circuit_kelvin)?;
            if instance.given("dtemp") {
                Ok(())
            } else {
                instance.store("dtemp", 0.)
            }
        }
    }

    fn dc_load(
        &self,
        _setup: RecordPair<'_>,
        current: RecordPair<'_>,
    ) -> SpiceResult<DcSensitivityLoad> {
        let i = current.instance;
        let scale = BehaviouralScale {
            m: i.value("m")?,
            tc1: i.value("tc1")?,
            tc2: i.value("tc2")?,
            temperature: Some(i.value("temp")? - CELSIUS_TO_KELVIN),
            dtemp: i.value("dtemp")?,
            ..*self.source.scale()
        };
        Ok(DcSensitivityLoad::replacement(
            self.source.with_scale(scale),
        ))
    }

    /// `ASRCacLoad` with the zeroed `ASRCacValues` of the reset: a voltage
    /// source's `+1, -1, -1, +1` branch incidences and zero derivatives, a
    /// current source nothing. The scale factor multiplies only those zeros.
    fn ac_load(
        &self,
        _setup: RecordPair<'_>,
        _reset: AcReset<'_>,
        _current: RecordPair<'_>,
    ) -> SpiceResult<AcSensitivityLoad> {
        let source = self.source;
        let [pos, neg] = [source.terminals()[0], source.terminals()[1]];
        match source.output() {
            BehaviouralOutput::Voltage => Ok(AcSensitivityLoad::stamp(ResetStamps::new(
                source.name(),
                real([
                    (StampAt::NodeBranch(pos), 1.),
                    (StampAt::NodeBranch(neg), -1.),
                    (StampAt::BranchNode(neg), -1.),
                    (StampAt::BranchNode(pos), 1.),
                ]),
            )?)),
            BehaviouralOutput::Current => Ok(AcSensitivityLoad::stamp(ResetStamps::new(
                source.name(),
                [],
            )?)),
        }
    }
}

/// An S or W switch's `.sens` description.
#[derive(Debug)]
pub(in crate::devices) struct SwitchSensitivity<'a> {
    switch: &'a Switch,
    gmin: Real,
}

impl<'a> SwitchSensitivity<'a> {
    pub(crate) const fn new(switch: &'a Switch, context: &crate::devices::ModelContext) -> Self {
        Self {
            switch,
            gmin: context.gmin,
        }
    }

    const fn keys(&self) -> (&'static str, &'static str) {
        match self.switch.kind() {
            SwitchKind::Voltage => ("vt", "vh"),
            SwitchKind::Current => ("it", "ih"),
        }
    }
}

impl SensitivityKernel for SwitchSensitivity<'_> {
    fn family(&self) -> SensitivityFamily {
        match self.switch.kind() {
            SwitchKind::Voltage => SensitivityFamily::VoltageSwitch,
            SwitchKind::Current => SensitivityFamily::CurrentSwitch,
        }
    }

    fn model(&self) -> Option<&str> {
        Some(self.switch.model_name())
    }

    fn model_parameters(&self) -> &'static [P] {
        match self.switch.kind() {
            SwitchKind::Voltage => SW,
            SwitchKind::Current => CSW,
        }
    }

    fn instance_parameters(&self) -> &'static [P] {
        &[]
    }

    fn records(&self) -> SpiceResult<(SensitivityRecord, SensitivityRecord)> {
        let (threshold, hysteresis) = self.keys();
        let values = self.switch.model_values();
        let mut model = SensitivityRecord::new(self.family(), ParameterScope::Model)
            .with(
                threshold,
                values.threshold.unwrap_or(0.),
                values.threshold.is_some(),
            )?
            .with(
                hysteresis,
                values.hysteresis.unwrap_or(0.),
                values.hysteresis.is_some(),
            )?
            .with("ron", values.on.unwrap_or(0.), values.on.is_some())?
            .with("roff", values.off.unwrap_or(0.), values.off.is_some())?
            .with("gon", values.on.map_or(0., |r| 1. / r), false)?
            .with("goff", values.off.map_or(0., |r| 1. / r), false)?;
        self.setup(
            &mut model,
            &mut SensitivityRecord::new(self.family(), ParameterScope::Instance),
        )?;
        Ok((
            model,
            SensitivityRecord::new(self.family(), ParameterScope::Instance),
        ))
    }

    /// `SWmParam`/`CSWmParam`: a resistance also stores its conductance.
    fn set(
        &self,
        _scope: ParameterScope,
        keyword: &str,
        value: Real,
        model: &mut SensitivityRecord,
        _instance: &mut SensitivityRecord,
    ) -> SpiceResult<()> {
        model.set(keyword, value)?;
        match keyword {
            "ron" => model.store("gon", 1. / value),
            "roff" => model.store("goff", 1. / value),
            _ => Ok(()),
        }
    }

    /// `SWsetup`/`CSWsetup`: threshold and hysteresis default to 0, the on
    /// conductance to 1 S and the off conductance to `gmin`.
    fn setup(
        &self,
        model: &mut SensitivityRecord,
        _instance: &mut SensitivityRecord,
    ) -> SpiceResult<()> {
        for keyword in <[&str; 2]>::from(self.keys()) {
            if !model.given(keyword) {
                model.store(keyword, 0.)?;
            }
        }
        if !model.given("ron") {
            model.store("gon", 1.)?;
            model.store("ron", 1.)?;
        }
        if !model.given("roff") {
            model.store("goff", self.gmin)?;
            model.store("roff", 1. / self.gmin)?;
        }
        Ok(())
    }

    fn dc_load(
        &self,
        _setup: RecordPair<'_>,
        current: RecordPair<'_>,
    ) -> SpiceResult<DcSensitivityLoad> {
        let (threshold, hysteresis) = self.keys();
        let m = current.model;
        Ok(DcSensitivityLoad::replacement(self.switch.with_model(
            m.value(threshold)?,
            m.value(hysteresis)?,
            m.value("gon")?,
            m.value("goff")?,
        )?))
    }

    /// `SWacLoad`/`CSWacLoad`: the on or off conductance of `current` for the
    /// state the reset load chose with the `state` threshold and hysteresis.
    fn ac_load(
        &self,
        _setup: RecordPair<'_>,
        reset: AcReset<'_>,
        current: RecordPair<'_>,
    ) -> SpiceResult<AcSensitivityLoad> {
        let (threshold, hysteresis) = self.keys();
        let closed = self
            .switch
            .with_model(
                reset.state.model.value(threshold)?,
                reset.state.model.value(hysteresis)?,
                reset.state.model.value("gon")?,
                reset.state.model.value("goff")?,
            )?
            .reset_closed()?;
        let g = current.model.value(if closed { "gon" } else { "goff" })?;
        let switch = self.switch;
        let [pos, neg] = [switch.terminals()[0], switch.terminals()[1]];
        Ok(AcSensitivityLoad::stamp(ResetStamps::new(
            switch.name(),
            real([
                (StampAt::Nodes(pos, pos), g),
                (StampAt::Nodes(pos, neg), -g),
                (StampAt::Nodes(neg, pos), -g),
                (StampAt::Nodes(neg, neg), g),
            ]),
        )?))
    }
}

/// A K coupling's `.sens` description: `MUTpTable`'s `k`, `MUTask`,
/// `MUTparam`, `MUTtemp` and `MUTacLoad`. The instance record holds `k`,
/// C's `MUTfactor` (`factor`) and the mirrored inductances and branch rows
/// of its inductors ([`crate::devices::sensitivity::DeviceSensitivity::coupled_inductors`]).
#[derive(Debug)]
pub(in crate::devices) struct MutualSensitivity<'a> {
    coupling: &'a MutualInductance,
}

impl<'a> MutualSensitivity<'a> {
    pub(crate) const fn new(coupling: &'a MutualInductance) -> Self {
        Self { coupling }
    }
}

impl SensitivityKernel for MutualSensitivity<'_> {
    fn family(&self) -> SensitivityFamily {
        SensitivityFamily::Mutual
    }

    fn model(&self) -> Option<&str> {
        None
    }

    fn model_parameters(&self) -> &'static [P] {
        &[]
    }

    /// `k` (`IOPAP`): perturbed only by an AC analysis.
    fn instance_parameters(&self) -> &'static [P] {
        const MUT: &[P] = &[P::principal("k", true)];
        MUT
    }

    fn records(&self) -> SpiceResult<(SensitivityRecord, SensitivityRecord)> {
        let instance = SensitivityRecord::new(self.family(), ParameterScope::Instance)
            .with("k", self.coupling.coefficient(), true)?
            .with("factor", 0., false)?
            .with("induct1", 0., false)?
            .with("induct2", 0., false)?
            .with_index("row1", 0)?
            .with_index("row2", 0)?;
        Ok((
            SensitivityRecord::new(self.family(), ParameterScope::Model),
            instance,
        ))
    }

    /// `MUTask` answers `MUTfactor` for `k`.
    fn ask(
        &self,
        _scope: ParameterScope,
        _keyword: &str,
        records: RecordPair<'_>,
    ) -> SpiceResult<Real> {
        records.instance.value("factor")
    }

    /// `MUTtemp`: `MUTfactor = k sqrt(|L1 L2|)` from the inductors'
    /// `INDinduct`.
    fn temperature(
        &self,
        _model: &mut SensitivityRecord,
        instance: &mut SensitivityRecord,
    ) -> SpiceResult<()> {
        let product = instance.value("induct1")? * instance.value("induct2")?;
        instance.store("factor", instance.value("k")? * product.abs().sqrt())
    }

    /// The circuit's own K device stamps nothing at DC.
    fn dc_load(
        &self,
        _setup: RecordPair<'_>,
        _current: RecordPair<'_>,
    ) -> SpiceResult<DcSensitivityLoad> {
        Ok(DcSensitivityLoad::Original)
    }

    /// `MUTacLoad`: `-= omega MUTfactor` in the imaginary parts of
    /// `(br1, br2)` and `(br2, br1)`.
    fn ac_load(
        &self,
        _setup: RecordPair<'_>,
        _reset: AcReset<'_>,
        current: RecordPair<'_>,
    ) -> SpiceResult<AcSensitivityLoad> {
        let i = current.instance;
        let factor = i.value("factor")?;
        let (row1, row2) = (i.index("row1")?, i.index("row2")?);
        let stamps = [(row1, row2), (row2, row1)].map(|(r, c)| ResetStamp {
            at: StampAt::Rows(r, c),
            value: -factor,
            part: StampPart::Reactive,
        });
        Ok(AcSensitivityLoad::stamp(ResetStamps::new(
            self.coupling.name(),
            stamps,
        )?))
    }

    fn coupled_inductors(&self) -> Option<[&str; 2]> {
        match self.coupling.inductors() {
            [first, second] => Some([first.name.as_str(), second.name.as_str()]),
            _ => None,
        }
    }
}
