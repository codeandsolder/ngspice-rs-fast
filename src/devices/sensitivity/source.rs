//! Independent sources under `.sens`: `vsrc.c`/`isrc.c` tables,
//! `vsrcpar.c`/`isrcpar.c` setters, `vsrctemp.c`/`isrctemp.c` and the DC/AC
//! loads (`vsrcload.c`, `vsrcacld.c`, `isrcload.c`, `isrcacld.c`).

use super::{
    ParameterScope, RecordPair, SensitivityFamily, SensitivityKernel, SensitivityParameter as P,
    SensitivityRecord, reconstructed_loads,
};
use crate::devices::IndependentSource;
use crate::devices::sources::{AcExcitation, AcInput};
use crate::primitives::{Complex, Real, SpiceResult};

/// `VSRCpTable` entries `set_param()` accepts (`portnum` is an integer).
const VOLTAGE: &[P] = &[
    P::principal("dc", false),
    P::principal("acmag", true),
    P::ac("acphase"),
    P::dc("z0"),
    P::dc("pwr"),
    P::dc("freq"),
    P::dc("phase"),
];

/// `ISRCpTable` entries `set_param()` accepts.
const CURRENT: &[P] = &[
    P::principal("dc", false),
    P::dc("m"),
    P::principal("acmag", true),
    P::ac("acphase"),
];

/// An independent source's replay, borrowing its actual excitation state.
#[derive(Debug)]
pub(in crate::devices) struct SourceSensitivity<'a> {
    source: &'a IndependentSource,
}

impl<'a> SourceSensitivity<'a> {
    pub(in crate::devices) const fn new(source: &'a IndependentSource) -> Self {
        Self { source }
    }

    fn ac_input(instance: &SensitivityRecord) -> SpiceResult<Option<AcInput>> {
        if !instance.given("ac") {
            return Ok(None);
        }
        Ok(Some(AcInput {
            magnitude: instance
                .given("acmag")
                .then(|| instance.value("acmag"))
                .transpose()?,
            phase: instance
                .given("acphase")
                .then(|| instance.value("acphase"))
                .transpose()?,
        }))
    }
}

impl SensitivityKernel for SourceSensitivity<'_> {
    fn family(&self) -> SensitivityFamily {
        if self.source.voltage {
            SensitivityFamily::VoltageSource
        } else {
            SensitivityFamily::CurrentSource
        }
    }

    fn model(&self) -> Option<&str> {
        None
    }

    fn model_parameters(&self) -> &'static [P] {
        &[]
    }

    fn instance_parameters(&self) -> &'static [P] {
        if self.source.is_current_probe() {
            &[]
        } else if self.source.voltage {
            VOLTAGE
        } else {
            CURRENT
        }
    }

    fn records(&self) -> SpiceResult<(SensitivityRecord, SensitivityRecord)> {
        let dc = self.source.dc;
        let ac = self.source.ac.input;
        let input = ac.unwrap_or_default();
        let mut instance = SensitivityRecord::new(self.family(), ParameterScope::Instance)
            .with("dc", if dc.given() { dc.value() } else { 0. }, dc.given())?
            .with(
                "acmag",
                input.magnitude.unwrap_or(0.),
                input.magnitude.is_some(),
            )?
            .with("acphase", input.phase.unwrap_or(0.), input.phase.is_some())?
            .with("ac", 0., ac.is_some())?
            .with(
                "m",
                self.source.multiplier.unwrap_or(1.),
                self.source.multiplier.is_some(),
            )?
            .with("z0", 0., false)?
            .with("pwr", 0., false)?
            .with("freq", 0., false)?
            .with("phase", 0., false)?
            .with("acreal", 0., false)?
            .with("acimag", 0., false)?;
        let mut model = SensitivityRecord::new(self.family(), ParameterScope::Model);
        self.temperature(&mut model, &mut instance)?;
        Ok((model, instance))
    }

    /// `VSRCparam`/`ISRCparam`: an AC magnitude or phase also sets
    /// `VSRCacGiven`.
    fn set(
        &self,
        _scope: ParameterScope,
        keyword: &str,
        value: Real,
        _model: &mut SensitivityRecord,
        instance: &mut SensitivityRecord,
    ) -> SpiceResult<()> {
        instance.set(keyword, value)?;
        if matches!(keyword, "acmag" | "acphase") {
            instance.set_given("ac", true)?;
        }
        Ok(())
    }

    /// `VSRCtemp`/`ISRCtemp`: the AC defaults, the phasor and the default `m`.
    fn temperature(
        &self,
        _model: &mut SensitivityRecord,
        instance: &mut SensitivityRecord,
    ) -> SpiceResult<()> {
        let input = Self::ac_input(instance)?;
        let (mag, phase) = input.map_or((0., 0.), AcInput::defaults);
        instance.store("acmag", mag)?;
        instance.store("acphase", phase)?;
        if !instance.given("m") {
            instance.store("m", 1.)?;
        }
        let ac = input.map_or(Ok(Complex::ZERO), AcInput::resolve)?;
        instance.store("acreal", ac.re)?;
        instance.store("acimag", ac.im)
    }

    reconstructed_loads!(
        ordinary,
        /// The DC load drives the DC value when given, otherwise the transient
        /// function's time-zero value (`dcValue`, zero, without a function); the
        /// AC load drives the phasor; a current source scales both by `m`. RF
        /// port setters (`z0`, `pwr`, `freq`, `phase`) do nothing on a source
        /// that is not a port.
        replacement
    );
}

impl SourceSensitivity<'_> {
    fn replacement(&self, current: RecordPair<'_>) -> SpiceResult<IndependentSource> {
        let i = current.instance;
        let dc = if i.given("dc") || !self.source.function_given {
            i.value("dc")?
        } else {
            self.source.dc.value()
        };
        let ac = AcExcitation {
            input: Self::ac_input(i)?,
            phasor: Complex::new(i.value("acreal")?, i.value("acimag")?),
        };
        let multiplier = (!self.source.voltage && i.given("m"))
            .then(|| i.value("m"))
            .transpose()?;
        self.source.sensitivity_copy(dc, ac, multiplier)
    }
}
