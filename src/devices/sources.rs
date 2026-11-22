//! Independent sources, with current positive from the first terminal to the second.
//!
//! A voltage source can also be an RFSPICE **port** (`portnum`/`z0`, C
//! `VSRCisPort`): `vsrcset.c` then creates an internal node `<name>#res`, puts
//! the ideal source between that node and the negative terminal, and
//! `vsrcload.c`/`vsrcacld.c` stamp the reference conductance `1/z0` between the
//! positive terminal and the internal node in **every** analysis. A port is
//! therefore a Thevenin source with series resistance `z0`, and its branch
//! current is the current through the ideal source from `<name>#res` to the
//! negative terminal. See `docs/port/SPARAM.md`.
use crate::devices::{AnalysisMode, Device, LinearContext, LinearSource, StampContext, Waveform};
use crate::primitives::{Complex, NodeId, Real, SpiceError, SpiceResult};

/// The RFSPICE port data of a voltage source (C `VSRCisPort` and its
/// `VSRCportNum`, `VSRCportZ0`, `VSRCportPower`, `VSRCportFreq`,
/// `VSRCportPhase` fields), with `vsrctemp.c`'s defaults applied.
#[derive(Debug, Clone, PartialEq)]
pub struct RfPort {
    /// The 1-based port index (`portnum`). Ports of a circuit must be numbered
    /// `1..=N` without gaps or duplicates (`vsrctemp.c`).
    pub number: usize,
    /// The positive, finite reference impedance `z0` in ohms (default 50).
    pub z0: Real,
    /// The internal node `<name>#res` between the series `z0` and the ideal
    /// source.
    pub internal: NodeId,
    /// `pwr` in watts (default 1 mW). Only the large-signal `PORT` time function
    /// uses it.
    pub power: Real,
    /// `freq` in hertz (default 1 GHz). Only the `PORT` time function uses it.
    pub frequency: Real,
    /// `phase` in degrees (default 0). Stored like C, which never uses it.
    pub phase: Real,
    /// True when `pwr`/`freq` selected C's large-signal `PORT` time function
    /// (`vsrcpar.c` sets `VSRCfunctionType = PORT`) and no later waveform setter
    /// replaced it.
    pub power_function: bool,
}

impl RfPort {
    /// The incident/scattered power-wave scale `ki = 1/(2 sqrt(z0))` of
    /// `vsrctemp.c` (`VSRCki`).
    #[must_use]
    pub fn ki(&self) -> Real {
        0.5 / self.z0.sqrt()
    }
}

/// AC polar input with independently optional magnitude and phase.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct AcInput {
    /// Magnitude, default 1 when AC is given.
    pub magnitude: Option<Real>,
    /// Degrees, default 0 when AC is given.
    pub phase: Option<Real>,
}

impl AcInput {
    pub(super) fn defaults(self) -> (Real, Real) {
        (self.magnitude.unwrap_or(1.), self.phase.unwrap_or(0.))
    }

    pub(super) fn resolve(self) -> SpiceResult<Complex> {
        let (magnitude, phase) = self.defaults();
        let radians = crate::devices::functions::phase_radians(phase);
        let ac = Complex::new(magnitude * radians.cos(), magnitude * radians.sin());
        if !ac.is_finite() {
            return Err(SpiceError::circuit("non-finite source excitation"));
        }
        Ok(ac)
    }
}

#[derive(Debug, Clone, Copy)]
pub(super) struct AcExcitation {
    pub(super) input: Option<AcInput>,
    pub(super) phasor: Complex,
}

impl AcExcitation {
    fn resolve(input: Option<AcInput>) -> SpiceResult<Self> {
        Ok(Self {
            input,
            phasor: input.map_or(Ok(Complex::ZERO), AcInput::resolve)?,
        })
    }

    // Do not round-trip a directly supplied rectangular excitation.
    fn rectangular(phasor: Complex) -> Self {
        Self {
            input: (phasor != Complex::ZERO).then(|| AcInput {
                magnitude: Some(phasor.magnitude()),
                phase: Some(phasor.im.atan2(phasor.re).to_degrees()),
            }),
            phasor,
        }
    }
}

/// Explicit DC is independent of time forcing. Derived DC may change when
/// the preceding PORT excitation changes; the original level is preserved.
#[derive(Debug, Clone, Copy)]
pub(super) enum DcExcitation {
    Explicit(Real),
    Waveform {
        time_zero: Real,
        accumulated: Option<Real>,
    },
}

impl DcExcitation {
    pub(super) const fn value(self) -> Real {
        match self {
            Self::Explicit(v)
            | Self::Waveform {
                accumulated: Some(v),
                ..
            } => v,
            Self::Waveform {
                time_zero,
                accumulated: None,
            } => time_zero,
        }
    }

    pub(super) const fn given(self) -> bool {
        matches!(self, Self::Explicit(_))
    }

    const fn accumulate(&mut self, value: Option<Real>) {
        if let Self::Waveform { accumulated, .. } = self {
            *accumulated = value;
        }
    }
}

/// Source card inputs after ordered setters, before coefficient resolution.
#[derive(Debug, Default)]
pub struct SourceInputs {
    /// Explicit DC level; omitted uses the waveform at time zero.
    pub dc: Option<Real>,
    /// AC inputs; omitted means no AC setter, unlike explicit zero magnitude.
    pub ac: Option<AcInput>,
    /// Time forcing; omitted uses the explicit DC level or implicit zero.
    pub waveform: Option<Waveform>,
}

/// Independent V or I source. AC, DC and transient excitations are distinct.
#[derive(Debug, Clone)]
pub struct IndependentSource {
    name: String,
    /// Inline `[positive, negative, internal]`. The third entry is exposed
    /// only when a port is present.
    terminals: [NodeId; 3],
    pub(super) voltage: bool,
    pub(super) dc: DcExcitation,
    pub(super) ac: AcExcitation,
    pub(super) multiplier: Option<Real>,
    pub(super) function_given: bool,
    /// The source's immutable original function. Only PORT accumulation
    /// has a second, derived waveform.
    waveform: Waveform,
    accumulated_waveform: Option<Box<Waveform>>,
    power_tone: Option<(Real, Real)>,
    port: Option<RfPort>,
    /// `distof1`/`distof2` distortion inputs (C `VSRCdF1given`, ...).
    distortion: [Option<crate::devices::distortion::DistortionInput>; 2],
}
/// Validated excitation before circuit topology is bound.
pub(super) struct SourceExcitation {
    voltage: bool,
    dc: DcExcitation,
    ac: AcExcitation,
    function_given: bool,
    waveform: Waveform,
}

impl SourceExcitation {
    fn rectangular(voltage: bool, dc: Real, ac: Complex, waveform: Waveform) -> SpiceResult<Self> {
        waveform.validate()?;
        Self::from_resolved(
            voltage,
            DcExcitation::Explicit(dc),
            AcExcitation::rectangular(ac),
            !matches!(waveform, Waveform::Constant(_)),
            waveform,
        )
    }

    pub(super) fn from_inputs(voltage: bool, inputs: SourceInputs) -> SpiceResult<Self> {
        let function_given = inputs.waveform.is_some();
        let waveform = inputs
            .waveform
            .unwrap_or_else(|| Waveform::Constant(inputs.dc.unwrap_or(0.)));
        waveform.validate()?;
        let dc = match inputs.dc {
            Some(value) => DcExcitation::Explicit(value),
            None => DcExcitation::Waveform {
                time_zero: waveform.time_zero()?,
                accumulated: None,
            },
        };
        Self::from_resolved(
            voltage,
            dc,
            AcExcitation::resolve(inputs.ac)?,
            function_given,
            waveform,
        )
    }

    fn from_resolved(
        voltage: bool,
        dc: DcExcitation,
        ac: AcExcitation,
        function_given: bool,
        waveform: Waveform,
    ) -> SpiceResult<Self> {
        if !dc.value().is_finite() || !ac.phasor.is_finite() {
            return Err(SpiceError::circuit("non-finite source excitation"));
        }
        Ok(Self {
            voltage,
            dc,
            ac,
            function_given,
            waveform,
        })
    }

    pub(super) fn bind(self, name: impl Into<String>, terminals: [NodeId; 2]) -> IndependentSource {
        IndependentSource {
            name: name.into(),
            terminals: [terminals[0], terminals[1], NodeId::GROUND],
            voltage: self.voltage,
            dc: self.dc,
            ac: self.ac,
            multiplier: None,
            function_given: self.function_given,
            accumulated_waveform: None,
            power_tone: None,
            waveform: self.waveform,
            port: None,
            distortion: [None, None],
        }
    }
}

impl IndependentSource {
    /// Creates an independent source. `voltage=false` selects a current source.
    /// A directly supplied rectangular AC coefficient is retained exactly;
    /// a nonzero coefficient counts as an AC excitation.
    /// # Errors
    /// Non-finite excitation or invalid waveform.
    pub fn new(
        name: impl Into<String>,
        terminals: [NodeId; 2],
        voltage: bool,
        dc: Real,
        ac: Complex,
        waveform: Waveform,
    ) -> SpiceResult<Self> {
        Ok(SourceExcitation::rectangular(voltage, dc, ac, waveform)?.bind(name, terminals))
    }

    /// Creates a source from polar AC inputs and explicit or derived DC.
    /// Keeps transient-dependent waveform defaults unresolved until analysis.
    /// # Errors
    /// Nonfinite resolved excitation or invalid waveform.
    pub fn from_inputs(
        name: impl Into<String>,
        terminals: [NodeId; 2],
        voltage: bool,
        inputs: SourceInputs,
    ) -> SpiceResult<Self> {
        Ok(SourceExcitation::from_inputs(voltage, inputs)?.bind(name, terminals))
    }

    /// Selects C's PORT cosine time function, including its zero-amplitude
    /// nonport form. Values are initialized by `VSRCtemp`.
    pub(crate) const fn with_power_tone(mut self, tone: Option<(Real, Real)>) -> Self {
        self.power_tone = tone;
        self
    }

    /// A copy with the replayed DC level, AC phasor and source multiplier.
    pub(super) fn sensitivity_copy(
        &self,
        dc: Real,
        ac: AcExcitation,
        multiplier: Option<Real>,
    ) -> SpiceResult<Self> {
        if !dc.is_finite() || !ac.phasor.is_finite() || multiplier.is_some_and(|m| !m.is_finite()) {
            return Err(SpiceError::circuit(format!(
                "{}: nonfinite perturbed source value",
                self.name
            )));
        }
        Ok(Self {
            dc: DcExcitation::Explicit(dc),
            ac,
            multiplier,
            ..self.clone()
        })
    }

    /// Records the `distof1` (`f1`) and `distof2` (`f2`) distortion inputs
    /// (`vsrcpar.c`/`isrcpar.c` `VSRC_D_F1`/`VSRC_D_F2`), used only by
    /// `.disto`.
    /// # Errors
    /// A nonfinite magnitude or phase.
    pub fn with_distortion(
        mut self,
        f1: Option<crate::devices::distortion::DistortionInput>,
        f2: Option<crate::devices::distortion::DistortionInput>,
    ) -> SpiceResult<Self> {
        if [f1, f2]
            .iter()
            .flatten()
            .any(|i| !i.magnitude.is_finite() || !i.phase.is_finite())
        {
            return Err(SpiceError::circuit(format!(
                "{}: non-finite distortion input",
                self.name
            )));
        }
        self.distortion = [f1, f2];
        Ok(self)
    }

    /// Whether the source has an AC excitation (C `VSRCacGiven`).
    #[must_use]
    pub const fn ac_given(&self) -> bool {
        self.ac.input.is_some()
    }

    /// Makes this voltage source an RF port (see the module documentation);
    /// the port's internal node becomes the third terminal.
    /// # Errors
    /// A current source, a port number of zero, a nonpositive or non-finite
    /// `z0`, non-finite `pwr`/`freq`/`phase`, or an internal node that is one
    /// of the external terminals.
    pub fn with_port(mut self, port: RfPort) -> SpiceResult<Self> {
        if !self.voltage {
            return Err(SpiceError::circuit(format!(
                "{}: only voltage sources can be RF ports",
                self.name
            )));
        }
        if port.number == 0
            || !(port.z0.is_finite() && port.z0 > 0.)
            || !(port.power.is_finite() && port.frequency.is_finite() && port.phase.is_finite())
        {
            return Err(SpiceError::circuit(format!(
                "{}: invalid RF port parameters",
                self.name
            )));
        }
        if self.terminals[..2].contains(&port.internal) {
            return Err(SpiceError::circuit(format!(
                "{}: the port's internal node must be distinct from its terminals",
                self.name
            )));
        }
        self.terminals[2] = port.internal;
        self.power_tone = port.power_function.then(|| {
            (
                // vsrctemp.c: sqrt(VSRCportPower * 4.0 * VSRCportZ0).
                (port.power * 4. * port.z0).sqrt(),
                2. * std::f64::consts::PI * port.frequency,
            )
        });
        self.port = Some(port);
        Ok(self)
    }

    fn effective_waveform(&self) -> &Waveform {
        self.accumulated_waveform
            .as_deref()
            .unwrap_or(&self.waveform)
    }

    /// The terminals of the ideal source: `[positive, negative]`, or
    /// `[#res, negative]` for a port.
    fn ideal(&self) -> [NodeId; 2] {
        [
            self.port
                .as_ref()
                .map_or(self.terminals[0], |port| port.internal),
            self.terminals[1],
        ]
    }
}

impl Device for IndependentSource {
    /// `vsrcset.c` (an RF port first stamps its reference resistor towards
    /// its internal node, whose branch replaces the positive terminal);
    /// current sources allocate nothing.
    fn setup_matrix(
        &self,
        setup: &mut crate::devices::setup::MatrixSetup<'_>,
    ) -> SpiceResult<bool> {
        if !self.voltage {
            return Ok(true);
        }
        let branch = setup.branch(0)?;
        if let Some(port) = &self.port {
            setup.conductance(self.terminals[0], port.internal);
        }
        let [positive, negative] = self.ideal();
        setup.voltage_branch(positive, negative, branch);
        if setup.pole_zero {
            // `VSRCpzSetup` (`vsrcpzs.c`) also allocates `(b,b)`, which
            // `vsrcpzld.c` loads for a source with an AC value.
            setup.element(Some(branch), Some(branch));
        }
        Ok(true)
    }
    fn observation_parameter(
        &self,
        keyword: &str,
        _context: &crate::devices::ModelContext,
    ) -> SpiceResult<Option<Real>> {
        Ok(match keyword {
            "dc" => Some(self.dc.value()),
            "m" if !self.voltage => Some(self.multiplier.unwrap_or(1.)),
            _ => None,
        })
    }
    fn observation_source_current(
        &self,
        time: Option<(Real, crate::devices::Forcing)>,
    ) -> SpiceResult<Option<Real>> {
        if self.voltage {
            return Ok(None);
        }
        let value = match time {
            Some((t, f)) => self
                .effective_waveform()
                .value_at_timed(t, f.limit, &f.timing)?,
            None => self.dc.value(),
        };
        Ok(Some(value * self.multiplier.unwrap_or(1.)))
    }
    fn observation_source_multiplier(&self) -> Real {
        self.multiplier.unwrap_or(1.)
    }

    fn instance_parameter(&self, keyword: &str) -> Option<&'static str> {
        match keyword.to_ascii_lowercase().as_str() {
            "acmag" => Some("acmag"),
            "acphase" => Some("acphase"),
            "m" if !self.voltage => Some("m"),
            _ => None,
        }
    }
    fn with_instance_parameter(
        &self,
        parameter: &str,
        value: Real,
        _context: &crate::devices::ModelContext,
    ) -> SpiceResult<Box<dyn Device>> {
        crate::devices::sweep::check_swept(&self.name, parameter, value, true, "finite")?;
        let mut copy = self.clone();
        match parameter {
            "acmag" | "acphase" => {
                let mut input = copy.ac.input.unwrap_or_default();
                if parameter == "acmag" {
                    input.magnitude = Some(value);
                } else {
                    input.phase = Some(value);
                }
                copy.ac = AcExcitation::resolve(Some(input))?;
            }
            "m" if !self.voltage => copy.multiplier = Some(value),
            // dctrcurv.c sets VSRCdcValue/ISRCdcValue and the dcGiven flag
            // for each swept point.
            "dc" => {
                copy.dc = DcExcitation::Explicit(value);
            }
            _ => {
                return Err(SpiceError::circuit(format!(
                    "source parameter {parameter} cannot be swept"
                )));
            }
        }
        Ok(Box::new(copy))
    }

    /// Noiseless: C's VSRC/ISRC have no noise routine (`DEVnoise = NULL`); the
    /// transient noise sources (`trnoise`/`trrandom`) are not ported.
    fn noise(
        &self,
        _context: &crate::devices::noise::NoiseContext<'_>,
    ) -> SpiceResult<crate::devices::noise::DeviceNoise> {
        Ok(crate::devices::noise::DeviceNoise::Noiseless)
    }

    /// `.sens`: C's VSRC/ISRC records ([`crate::devices::sensitivity`]). An
    /// RF port is refused: `VSRCtemp` re-registers ports on every call and
    /// the port's `z0`/`pwr`/`freq` perturbations are not ported.
    fn sensitivity(
        &self,
        _context: &crate::devices::models::ModelContext,
    ) -> SpiceResult<Box<dyn crate::devices::sensitivity::DeviceSensitivity + '_>> {
        if self.port.is_some() {
            return Err(crate::devices::sensitivity::not_ported(
                &self.name,
                self.designator(),
                " (an RF port source)",
            ));
        }
        Ok(Box::new(
            crate::devices::sensitivity::SourceSensitivity::new(self),
        ))
    }

    /// Linear, with its `distof1`/`distof2` inputs (`cktdisto.c` reads them
    /// for the first-order `D_RHSF1`/`D_RHSF2` solves).
    fn distortion(
        &self,
        _context: &crate::devices::distortion::DistortionContext<'_>,
    ) -> SpiceResult<crate::devices::distortion::DeviceDistortion> {
        Ok(match self.distortion {
            [None, None] => crate::devices::distortion::DeviceDistortion::Linear,
            [f1, f2] => crate::devices::distortion::DeviceDistortion::Input { f1, f2 },
        })
    }

    fn input_source(&self) -> Option<crate::devices::noise::InputSource> {
        Some(crate::devices::noise::InputSource {
            kind: if self.voltage {
                crate::devices::linear::SourceKind::Voltage
            } else {
                crate::devices::linear::SourceKind::Current
            },
            ac_given: self.ac_given(),
        })
    }

    fn name(&self) -> &str {
        &self.name
    }
    fn designator(&self) -> char {
        if self.voltage { 'v' } else { 'i' }
    }
    fn terminals(&self) -> &[NodeId] {
        &self.terminals[..if self.port.is_some() { 3 } else { 2 }]
    }
    fn branch_currents(&self) -> usize {
        usize::from(self.voltage)
    }
    /// A voltage source's current can control F/H sources (`VSRCfindBr`).
    fn findable_branch(&self) -> Option<usize> {
        self.voltage.then_some(0)
    }
    fn bind_voltage_predecessor(
        &mut self,
        previous: &Waveform,
        previous_dc: Real,
    ) -> SpiceResult<Option<(Waveform, Real)>> {
        if !self.voltage {
            return Ok(None);
        }
        self.dc.accumulate(None);
        self.accumulated_waveform = None;
        if let Some(tone) = self.power_tone {
            let (baseline, mut tones) = match previous {
                Waveform::PortSeries { baseline, tones } => (baseline.clone(), tones.clone()),
                other => (Box::new(other.clone()), Vec::new()),
            };
            tones.push(tone);
            let accumulated = Waveform::PortSeries { baseline, tones };
            accumulated.validate()?;
            self.accumulated_waveform = Some(Box::new(accumulated));
            self.dc.accumulate(Some(previous_dc + tone.0));
            if !self.dc.value().is_finite() {
                return Err(SpiceError::circuit("nonfinite PORT DC accumulation"));
            }
        }
        Ok(Some((self.effective_waveform().clone(), self.dc.value())))
    }
    fn rf_port(&self) -> Option<&RfPort> {
        self.port.as_ref()
    }
    /// The value `vsrcload.c`/`isrcload.c` load. In MODEDCOP/MODEDCTRANCURVE
    /// a given DC value is loaded as `dc * CKTsrcFact`. Otherwise the source
    /// function is evaluated at `CKTtime` (0 in DC modes), with the one-sided
    /// limit of `context.forcing`, and scaled by `CKTsrcFact` only in
    /// MODETRANOP (an operating-point load with a forcing context). A PORT
    /// adds its cosine to C's running `value`, which is the value the
    /// previous voltage source in load order loaded
    /// ([`StampContext::voltage_source_value`]). A port also stamps `1/z0`
    /// between its positive terminal and `#res`.
    fn stamp(&self, context: &mut StampContext<'_>) -> SpiceResult<()> {
        let (time, transient_op) = match context.mode {
            AnalysisMode::OperatingPoint | AnalysisMode::DcSweep => (0., context.forcing.is_some()),
            AnalysisMode::Transient { time, .. } => (time, false),
            AnalysisMode::Ac { .. } => {
                return Err(SpiceError::circuit(
                    "use linear equation assembly for AC sources",
                ));
            }
        };
        let dc_mode = !matches!(context.mode, AnalysisMode::Transient { .. });
        let value = if dc_mode && !transient_op && self.dc.given() {
            self.dc.value() * context.source_factor
        } else {
            let mut value = match (self.voltage, self.power_tone) {
                (true, Some((amplitude, two_pi_frequency))) => {
                    let tone = amplitude * (time * two_pi_frequency).cos();
                    *context.voltage_source_value + tone
                }
                _ => match context.forcing {
                    Some(forcing) => self.effective_waveform().value_at_timed(
                        time,
                        forcing.limit,
                        &forcing.timing,
                    )?,
                    // DC without a given value: the function at time 0,
                    // which the factory stored as `dc`.
                    None if dc_mode => self.dc.value(),
                    None => {
                        return Err(SpiceError::circuit(format!(
                            "{}: transient source load without a forcing context",
                            self.name
                        )));
                    }
                },
            };
            if transient_op {
                value *= context.source_factor;
            }
            value
        };
        if self.voltage {
            *context.voltage_source_value = value;
        }
        if let Some(port) = &self.port {
            let g0 = 1. / port.z0;
            let [positive, internal] = [self.terminals[0], port.internal];
            context.stamp(positive, positive, g0)?;
            context.stamp(internal, internal, g0)?;
            context.stamp(positive, internal, -g0)?;
            context.stamp(internal, positive, -g0)?;
        }
        if self.voltage {
            let branch = context.branch(0)?;
            crate::devices::linear::branch_stamp(
                context.matrix,
                context.unknowns,
                self.ideal(),
                branch,
            )?;
            context.rhs.add_to(branch, value)
        } else {
            context.stamp_rhs(self.terminals[0], -value * self.multiplier.unwrap_or(1.))?;
            context.stamp_rhs(self.terminals[1], value * self.multiplier.unwrap_or(1.))
        }
    }
    fn assemble_linear(&self, context: &mut LinearContext<'_>) -> SpiceResult<()> {
        if let Some(port) = &self.port {
            context.nodal([self.terminals[0], port.internal], 1. / port.z0, false)?;
        }
        let rows = if self.voltage {
            vec![(context.branch(self.ideal())?, 1.)]
        } else {
            [(self.terminals[0], -1.), (self.terminals[1], 1.)]
                .into_iter()
                .filter_map(|(node, sign)| {
                    context
                        .unknowns
                        .node_row(node)
                        .map(|r| (r, sign * self.multiplier.unwrap_or(1.)))
                })
                .collect()
        };
        context.system.sources.push(LinearSource {
            name: self.name.clone(),
            kind: if self.voltage {
                crate::devices::linear::SourceKind::Voltage
            } else {
                crate::devices::linear::SourceKind::Current
            },
            rows,
            dc: self.dc.value(),
            ac: self.ac.phasor,
            waveform: self.effective_waveform().clone(),
        });
        Ok(())
    }
    /// `VSRCpzLoad` (`vsrcpzld.c`): a voltage source without an AC value
    /// shorts its terminals as in AC; one with an AC value is removed (the
    /// KCL couplings of its current stay, its branch row becomes
    /// `i = 0`), since the pole-zero drive replaces the input. A current
    /// source has no pole-zero load in C and contributes nothing (open), as
    /// in the AC matrix. An RF port is refused: `vsrcpzld.c` stamps only the
    /// ideal source between `<name>#res` and the negative terminal and omits
    /// the port's `1/z0` (which `vsrcload.c`/`vsrcacld.c` stamp), leaving the
    /// positive terminal disconnected in C's pole-zero matrix.
    fn assemble_pole_zero(
        &self,
        context: &mut LinearContext<'_>,
        _bias: &crate::maths::Vector,
    ) -> SpiceResult<()> {
        if !self.voltage {
            return Ok(());
        }
        if self.port.is_some() {
            return Err(SpiceError::Unsupported {
                feature: format!(
                    "pole-zero analysis of RF port {}: C's vsrcpzld.c omits the port's z0",
                    self.name
                ),
                location: None,
            });
        }
        if !self.ac_given() {
            context.branch(self.ideal())?;
            return Ok(());
        }
        let branch = context
            .branch
            .ok_or_else(|| SpiceError::circuit("missing branch-row binding"))?;
        let [positive, negative] = self.ideal();
        for (node, sign) in [(positive, 1.), (negative, -1.)] {
            if let Some(row) = context.unknowns.node_row(node) {
                context.system.a.add(row, branch, sign)?;
            }
        }
        context.system.a.add(branch, branch, 1.)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn port_accumulation_rebuilds_from_predecessor_not_previous_result() {
        let original =
            Waveform::Pwl(crate::devices::PwlSource::plain(vec![(0., 1.), (1., 2.)]).unwrap());
        let mut source = IndependentSource::new(
            "v1",
            [NodeId::new(1), NodeId::GROUND],
            true,
            0.,
            Complex::ZERO,
            original.clone(),
        )
        .unwrap()
        .with_power_tone(Some((2., 3.)));
        let predecessor = Waveform::Constant(4.);
        let expected = Waveform::PortSeries {
            baseline: Box::new(predecessor.clone()),
            tones: vec![(2., 3.)],
        };
        for _ in 0..2 {
            let (actual, dc) = source
                .bind_voltage_predecessor(&predecessor, 4.)
                .unwrap()
                .unwrap();
            assert_eq!(actual, expected);
            assert_eq!(dc, 0.);
            assert_eq!(source.waveform, original);
        }
        source.power_tone = None;
        let (actual, dc) = source
            .bind_voltage_predecessor(&predecessor, 4.)
            .unwrap()
            .unwrap();
        assert_eq!(actual, original);
        assert_eq!(dc, 0.);
        assert!(source.accumulated_waveform.is_none());
    }
    fn coefficients(device: Box<dyn Device>) -> crate::devices::LinearSource {
        let mut circuit = crate::devices::Circuit::new();
        circuit.add_node("a").unwrap();
        circuit.add_device(device).unwrap();
        circuit
            .linear_system()
            .unwrap()
            .sources
            .into_iter()
            .next()
            .unwrap()
    }

    fn polar_source(
        voltage: bool,
        ac: Option<AcInput>,
        dc: Option<Real>,
        waveform: Option<Waveform>,
    ) -> IndependentSource {
        IndependentSource::from_inputs(
            if voltage { "v1" } else { "i1" }.to_owned(),
            [NodeId::new(1), NodeId::GROUND],
            voltage,
            SourceInputs { dc, ac, waveform },
        )
        .unwrap()
    }

    fn ac_bits(ac: Complex) -> [u64; 2] {
        [ac.re.to_bits(), ac.im.to_bits()]
    }

    #[test]
    fn source_ac_construction_sweep_and_replay_share_reference_coefficients() {
        use crate::devices::sensitivity::{
            ParameterScope, RecordPair, SensitivityLoad, SensitivityMode,
        };

        let context = crate::devices::ModelContext::default();
        for voltage in [true, false] {
            for (magnitude, phase) in [
                (2., 123_456_789_012_345.),
                (-0., -0.),
                (0., -720.25),
                (-3., 1e300),
            ] {
                let input = AcInput {
                    magnitude: Some(magnitude),
                    phase: Some(phase),
                };
                let source = polar_source(voltage, Some(input), None, None);
                let expected = ac_bits(source.ac.phasor);
                assert!(source.ac_given());
                let baseline = polar_source(
                    voltage,
                    Some(AcInput {
                        magnitude: Some(magnitude),
                        phase: None,
                    }),
                    None,
                    None,
                );
                let swept = baseline
                    .with_instance_parameter("acphase", phase, &context)
                    .unwrap();
                assert_eq!(ac_bits(coefficients(swept).ac), expected);
                let sensitivity = baseline.sensitivity(&context).unwrap();
                let (mut model, mut instance) = sensitivity.records().unwrap();
                sensitivity
                    .set(
                        ParameterScope::Instance,
                        "acphase",
                        phase,
                        &mut model,
                        &mut instance,
                    )
                    .unwrap();
                sensitivity.temperature(&mut model, &mut instance).unwrap();
                let records = RecordPair {
                    model: &model,
                    instance: &instance,
                };
                let SensitivityLoad::Replacement(replayed) = sensitivity
                    .load(records, records, SensitivityMode::Ac)
                    .unwrap()
                else {
                    panic!("expected replacement");
                };
                assert_eq!(ac_bits(coefficients(replayed).ac), expected);
            }
        }
    }

    #[test]
    fn omitted_ac_and_explicit_zero_are_distinct_and_overflow_is_rejected() {
        let context = crate::devices::ModelContext::default();
        let absent = polar_source(true, None, None, None);
        assert!(!absent.ac_given());
        let explicit_zero = polar_source(
            true,
            Some(AcInput {
                magnitude: Some(-0.),
                phase: Some(-0.),
            }),
            None,
            None,
        );
        assert!(explicit_zero.ac_given());
        assert_eq!(
            ac_bits(explicit_zero.ac.phasor),
            [(-0.0_f64).to_bits(), 0.0_f64.to_bits()]
        );
        let swept = absent
            .with_instance_parameter("acphase", -0.0, &context)
            .unwrap();
        assert_eq!(
            ac_bits(coefficients(swept).ac),
            [1.0_f64.to_bits(), (-0.0_f64).to_bits()]
        );
        assert!(
            absent
                .with_instance_parameter("acphase", Real::MAX, &context)
                .is_err()
        );
        assert!(
            AcInput {
                magnitude: Some(0.),
                phase: Some(Real::MAX)
            }
            .resolve()
            .is_err()
        );
        // The exact binary64 intermediate distinguishes C from to_radians().
        assert_eq!(
            crate::devices::functions::phase_radians(123_456_789_012_345.).to_bits(),
            2_154_727_452_205.378_4_f64.to_bits()
        );
    }

    #[test]
    fn explicit_dc_and_port_accumulation_do_not_overwrite_each_other() {
        let waveform = Waveform::Constant(4.);
        let mut source =
            polar_source(true, None, None, Some(waveform.clone())).with_power_tone(Some((2., 3.)));
        assert_eq!(source.dc.value(), 4.);
        assert!(!source.dc.given());
        source.bind_voltage_predecessor(&waveform, 4.).unwrap();
        assert_eq!(source.dc.value(), 6.);
        source.power_tone = None;
        source.bind_voltage_predecessor(&waveform, 7.).unwrap();
        assert_eq!(source.dc.value(), 4.);

        let mut explicit = polar_source(true, None, Some(-0.), Some(waveform.clone()))
            .with_power_tone(Some((2., 3.)));
        explicit.bind_voltage_predecessor(&waveform, 4.).unwrap();
        assert!(explicit.dc.given());
        assert_eq!(explicit.dc.value().to_bits(), (-0.0_f64).to_bits());
        let swept = explicit
            .with_instance_parameter("dc", 7., &crate::devices::ModelContext::default())
            .unwrap();
        assert_eq!(coefficients(swept).dc, 7.);
    }
    #[test]
    fn current_multiplier_is_replayed_once_with_its_given_state() {
        use crate::devices::sensitivity::{RecordPair, SensitivityLoad, SensitivityMode};
        let context = crate::devices::ModelContext::default();
        let source = polar_source(
            false,
            Some(AcInput {
                magnitude: Some(2.),
                phase: None,
            }),
            Some(3.),
            None,
        );
        let swept = source.with_instance_parameter("m", 2.5, &context).unwrap();
        let sensitivity = swept.sensitivity(&context).unwrap();
        let (model, instance) = sensitivity.records().unwrap();
        assert!(instance.given("m"));
        assert_eq!(instance.value("m").unwrap(), 2.5);
        let records = RecordPair {
            model: &model,
            instance: &instance,
        };
        let SensitivityLoad::Replacement(replayed) = sensitivity
            .load(records, records, SensitivityMode::Dc)
            .unwrap()
        else {
            panic!("expected replacement");
        };
        assert_eq!(
            replayed.observation_source_current(None).unwrap(),
            Some(7.5)
        );
        let linear = coefficients(replayed);
        assert_eq!(linear.dc, 3.);
        assert_eq!(linear.rows[0].1, -2.5);
        assert_eq!(linear.ac, Complex::new(2., 0.));
    }
}
