//! Resistor, capacitor and inductor — the first porting targets.
//!
//! Static and state-independent dynamic operators feed linear DC/AC through
//! [`Device::assemble_linear`]. Separately,
//! [`Device::stamp`] implements the SPICE trap/Gear companion models for C and
//! L from accepted charge/flux history (roadmap M3); the two paths never mix.
//! Each type follows its C counterpart:
//!
//! | Rust | C parser | C stamping |
//! | --- | --- | --- |
//! | [`Resistor`] | `src/spicelib/parser/inp2r.c` | `src/spicelib/devices/res/resload.c` |
//! | [`Capacitor`] | `src/spicelib/parser/inp2c.c` | `src/spicelib/devices/cap/capload.c` |
//! | [`Inductor`] | `src/spicelib/parser/inp2l.c` | `src/spicelib/devices/ind/indload.c` |
//!
//! Fields hold the supplied scalar values. [`crate::devices::passive`] projects bounded
//! model geometry, TC1/TC2, TEMP/TNOM, scale and multiplicity into effective
//! scalars before delegating here. Literal factory support is unchanged;
//! behavioural values, AC-only resistance and other advanced setters error
//! explicitly rather than silently modifying or omitting physics.

use crate::maths::{Coefficients, Companion};
use crate::primitives::{NodeId, Real, SpiceError, SpiceResult};

use crate::devices::noise::{
    CELSIUS_TO_KELVIN, DeviceNoise, FlickerLaw, NoiseContext, NoiseFamily, NoiseKind, NoiseSource,
};
use crate::devices::traits::{AnalysisMode, Device, StampContext};

/// State slots of a capacitor (`CAPqcap`, `CAPccap`) or inductor
/// (`INDflux`, `INDvolt`): the integrated quantity and its derivative.
pub(super) const QUANTITY: usize = 0;
pub(super) const REACTIVE_STATE_COUNT: usize = 2;
const DERIVATIVE: usize = 1;

/// The coefficients of a companion transient load, checked against the mode.
fn companion_coefficients<'a>(
    context: &StampContext<'a>,
    name: &str,
) -> SpiceResult<&'a Coefficients> {
    let AnalysisMode::Transient { dt, .. } = context.mode else {
        return Err(SpiceError::circuit(format!(
            "{name}: companion stamping needs a transient load"
        )));
    };
    let coefficients = context.integration.ok_or_else(|| {
        SpiceError::circuit(format!(
            "{name}: companion transient load without integration coefficients"
        ))
    })?;
    // These must be the very same trial timestep, not merely close.
    if coefficients.dt().partial_cmp(&dt) != Some(std::cmp::Ordering::Equal) {
        return Err(SpiceError::circuit(format!(
            "{name}: load timestep {dt} differs from integration timestep {}",
            coefficients.dt()
        )));
    }
    Ok(coefficients)
}

/// Integrates one element as `NIintegrate` does: `quantity` is the trial
/// charge/flux and `capacitance` the C or L it was derived with. Reads only
/// accepted history and touches nothing.
/// `(MODEINITTRAN, MODEINITPRED)` of a transient load.
fn transient_phase(context: &StampContext<'_>) -> (bool, bool) {
    let predict = context.states.phase() == crate::devices::IterationPhase::Predict;
    let initial = predict && context.states.initial_transient();
    (initial, predict && !initial)
}

fn integrate(
    context: &StampContext<'_>,
    coefficients: &Coefficients,
    quantity: Real,
    capacitance: Real,
    name: &str,
) -> SpiceResult<Companion> {
    // Composed devices may own additional non-reactive slots after this
    // pair. The shared scalar load reads/writes only its first two slots.
    if context.states.len() < REACTIVE_STATE_COUNT {
        return Err(SpiceError::circuit(format!(
            "{name}: companion load without its two state slots"
        )));
    }
    let needed = coefficients.charge_history_len();
    let mut history = crate::maths::integrator::ChargeHistory::new(quantity);
    for age in 1..=needed {
        history.push(context.states.accepted(age, QUANTITY).ok_or_else(|| {
            SpiceError::circuit(format!(
                "{name}: order {} needs {needed} accepted point(s), have {}",
                coefficients.order(),
                context.states.depth()
            ))
        })?)?;
    }
    let previous = if coefficients.needs_previous_derivative() {
        Some(
            context
                .states
                .accepted(1, DERIVATIVE)
                .ok_or_else(|| SpiceError::circuit(format!("{name}: no accepted derivative")))?,
        )
    } else {
        None
    };
    coefficients.integrate(&history, previous, capacitance)
}

/// Records a DC point's charge/flux with zero derivative, when the load
/// tracks state (the transient operating point, C `MODETRANOP`).
fn record_dc_state(context: &mut StampContext<'_>, quantity: Real) -> SpiceResult<()> {
    if context.states.is_empty() {
        return Ok(());
    }
    if !quantity.is_finite() {
        return Err(SpiceError::circuit("nonfinite charge/flux at DC"));
    }
    context.states.set(QUANTITY, quantity)?;
    context.states.set(DERIVATIVE, 0.0)
}

/// A resistor's `.noise` description (`resnoise.c`, with the `ressetup.c`
/// defaults): the `noisy` switch, the flicker law
/// `m KF abs(I/m)^AF / (area f^EF)` and the noise temperature.
#[derive(Debug, Clone, PartialEq)]
pub struct ResistorNoise {
    /// C `RESnoisy`: `false` (`noisy=0`) removes the resistor from `.noise`
    /// entirely, names included. Default `true`.
    pub noisy: bool,
    /// Use the magnitude of the signed conductance for the thermal generator.
    /// This is an explicit noise law for synthesized active series impedances,
    /// not the default resistor law.
    pub absolute_thermal_conductance: bool,
    /// Flicker coefficient `KF` (default 0, no flicker noise).
    pub kf: Real,
    /// Flicker current exponent `AF` (default 1).
    pub af: Real,
    /// Flicker frequency exponent `EF` (default 1).
    pub ef: Real,
    /// C `RESeffNoiseArea`: `(L - 2 SHORT)^LF (W - 2 NARROW)^WF` when the
    /// instance gives `l` or `w`, else 1.
    pub area: Real,
    /// The parallel multiplier `m` (default 1).
    pub multiplicity: Real,
    /// Instance `temp=` in Celsius; `None` for the circuit temperature.
    pub temperature: Option<Real>,
    /// The model name; `None` for C's default resistor model (a literal resistor).
    pub model: Option<String>,
}

impl Default for ResistorNoise {
    fn default() -> Self {
        Self {
            noisy: true,
            absolute_thermal_conductance: false,
            kf: 0.0,
            af: 1.0,
            ef: 1.0,
            area: 1.0,
            multiplicity: 1.0,
            temperature: None,
            model: None,
        }
    }
}

/// ngspice's smallest resistor magnitude (`RESMIN`, `resparam.c`).
pub(crate) const RES_MIN: Real = 1e-12;

/// The `RES_RESIST` setter (`resparam.c`): a resistance in `[0, RESMIN)` is
/// raised to `RESMIN`, one in `(-RESMIN, 0)` lowered to `-RESMIN`, each with
/// C's warning on standard error. Other values pass unchanged.
pub(crate) fn res_resist(name: &str, value: Real) -> Real {
    let clamped = clamp_resistance(value);
    if clamped.to_bits() != value.to_bits() {
        eprintln!("warning: Value of resistor {name} is too small, set to {clamped:.6e}");
    }
    clamped
}

/// The numerical part of the C resistance setter, without diagnostics.
pub(super) fn clamp_resistance(value: Real) -> Real {
    if (0.0..RES_MIN).contains(&value) {
        RES_MIN
    } else if value < 0.0 && value > -RES_MIN {
        -RES_MIN
    } else {
        value
    }
}

fn validate_resistance(name: &str, resistance: Real) -> SpiceResult<()> {
    if !resistance.is_finite() || resistance == 0.0 || !(1.0 / resistance).is_finite() {
        return Err(SpiceError::circuit(format!(
            "resistor {name}: resistance must be finite and nonzero with finite conductance"
        )));
    }
    Ok(())
}

/// A resistor, `r1 n1 n2 <value> [tc1=… tc2=…]`.
#[derive(Debug, Clone, PartialEq)]
pub struct Resistor {
    name: String,
    terminals: [NodeId; 2],
    /// Resistance in ohms, as parsed. A behavioural `R={expr}` value is not
    /// modelled yet.
    resistance: Real,
    /// What `.noise` sees; the default for a literal resistor.
    noise: ResistorNoise,
}

impl Resistor {
    /// Builds a resistor.
    ///
    /// # Errors
    ///
    /// [`SpiceError::Circuit`] for non-finite or zero resistance/conductance overflow.
    pub fn new(
        name: impl Into<String>,
        terminals: [NodeId; 2],
        resistance: Real,
    ) -> SpiceResult<Self> {
        let name = name.into();
        validate_resistance(&name, resistance)?;
        Ok(Self {
            name,
            terminals,
            resistance,
            noise: ResistorNoise::default(),
        })
    }

    /// The same resistor with the given `.noise` description.
    #[must_use = "the returned resistor carries the replacement noise description"]
    pub fn with_noise(self, noise: ResistorNoise) -> Self {
        Self { noise, ..self }
    }

    /// The `.noise` description.
    #[must_use = "the returned parameters determine the resistor noise law"]
    pub const fn noise_parameters(&self) -> &ResistorNoise {
        &self.noise
    }

    /// The resistance in ohms.
    #[must_use = "the returned resistance is the physical branch value in ohms"]
    pub const fn resistance(&self) -> Real {
        self.resistance
    }

    /// The conductance ngspice stamps, `1/R` (`resload.c` computes it as
    /// `1/model->RESconduct`).
    #[must_use = "the returned reciprocal resistance supplies the resistor stamp"]
    pub fn conductance(&self) -> Real {
        1.0 / self.resistance
    }
}

impl Device for Resistor {
    /// `ressetup.c`.
    fn setup_matrix(
        &self,
        setup: &mut crate::devices::setup::MatrixSetup<'_>,
    ) -> SpiceResult<bool> {
        setup.conductance(self.terminals[0], self.terminals[1]);
        Ok(true)
    }
    fn observation_parameter(
        &self,
        keyword: &str,
        context: &crate::devices::ModelContext,
    ) -> SpiceResult<Option<Real>> {
        Ok(match keyword {
            "r" | "resistance" => Some(self.resistance),
            "g" | "conductance" => Some(self.conductance()),
            "temp" => Some(context.temperature),
            "m" | "scale" => Some(1.),
            "tc1" | "tc2" | "dtemp" => Some(0.),
            _ => None,
        })
    }
    fn parameter_geometry_scale(
        &self,
        key: crate::devices::sweep::ParameterKey,
    ) -> crate::devices::sweep::GeometryScale {
        use crate::devices::sweep::{GeometryScale, ParameterKey};
        if matches!(key, ParameterKey::Native("l" | "w")) {
            GeometryScale::Length
        } else {
            GeometryScale::Identity
        }
    }

    fn instance_parameter(&self, keyword: &str) -> Option<&'static str> {
        [
            "resistance",
            "r",
            "temp",
            "tc1",
            "tc2",
            "w",
            "l",
            "m",
            "scale",
        ]
        .into_iter()
        .find(|k| k.eq_ignore_ascii_case(keyword))
        .map(|k| match k {
            "r" => "resistance",
            "c" | "cap" => "capacitance",
            other => other,
        })
    }
    fn with_instance_parameter(
        &self,
        parameter: &str,
        value: Real,
        context: &crate::devices::ModelContext,
    ) -> SpiceResult<Box<dyn Device>> {
        if parameter == "resistance" {
            validate_resistance(&self.name, value)?;
            return Ok(Box::new(Self {
                resistance: value,
                ..self.clone()
            }));
        }
        crate::devices::passive::swept_resistor(self, (parameter, value), context)
    }

    /// Linear in `.disto`: C gives this device no distortion routine
    /// (`DEVdisto = NULL`, `res/resinit.c`), so it enters only through its
    /// small-signal matrix.
    fn distortion(
        &self,
        _context: &crate::devices::distortion::DistortionContext<'_>,
    ) -> crate::primitives::SpiceResult<crate::devices::distortion::DeviceDistortion> {
        Ok(crate::devices::distortion::DeviceDistortion::Linear)
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn designator(&self) -> char {
        'r'
    }

    fn terminals(&self) -> &[NodeId] {
        &self.terminals
    }

    fn stamp(&self, context: &mut StampContext<'_>) -> SpiceResult<()> {
        stamp_conductance(self.terminals, 1. / self.resistance, context)
    }
    fn assemble_linear(
        &self,
        context: &mut crate::devices::linear::LinearContext<'_>,
    ) -> SpiceResult<()> {
        assemble_conductance(self.terminals, 1. / self.resistance, context)
    }

    fn resistor_metadata(&self) -> Option<crate::devices::sweep::ResistorMetadata> {
        Some(crate::devices::sweep::ResistorMetadata {
            origin: crate::devices::sweep::ResistorOrigin::Literal,
            supplied: self.resistance,
            multiplicity: 1.0,
        })
    }

    /// A literal resistor has no temperature or multiplicity law: the supplied
    /// scalar is the effective resistance, checked by the constructor's rules.
    fn resistor_effective(
        &self,
        supplied: Real,
        _context: &crate::devices::models::ModelContext,
    ) -> SpiceResult<Real> {
        validate_resistance(&self.name, supplied)?;
        Ok(supplied)
    }

    /// A literal resistor: C's default resistor model and the resistance
    /// alone given ([`crate::devices::sensitivity`]).
    fn sensitivity(
        &self,
        context: &crate::devices::models::ModelContext,
    ) -> SpiceResult<Box<dyn crate::devices::sensitivity::DeviceSensitivity + '_>> {
        Ok(Box::new(
            crate::devices::sensitivity::ResistorSensitivity::new(
                &self.name,
                self.terminals,
                crate::devices::sensitivity::PassiveInputs::Literal {
                    value: self.resistance,
                    initial_condition: None,
                },
                context,
            ),
        ))
    }

    /// Pole-zero load: C `respzld.c` (the `ac=` conductance when given, as `resacld.c`) equals the AC load with `s` for `j omega`.
    fn assemble_pole_zero(
        &self,
        context: &mut crate::devices::linear::LinearContext<'_>,
        bias: &crate::maths::Vector,
    ) -> crate::primitives::SpiceResult<()> {
        self.assemble_small_signal(context, bias)
    }

    /// `resnoise.c`: thermal noise `4 k T / R` at the instance temperature
    /// and the flicker law of [`ResistorNoise`], driven by the operating-point
    /// current (C `REScurrent`). `noisy=0` is noiseless.
    fn noise(
        &self,
        context: &NoiseContext<'_>,
        sources: &mut Vec<NoiseSource>,
    ) -> SpiceResult<DeviceNoise<'_>> {
        Ok(resistor_noise(
            self.terminals,
            1. / self.resistance,
            &self.noise,
            self.noise.model.as_deref(),
            context,
            None,
            sources,
        ))
    }
}

/// The four real nodal stamps in the existing load order. Their positions
/// never change for this two-terminal element, so a bound matrix can reuse
/// the verified batch instead of repeating per-node/element lookup.
fn stamp_nodal(
    context: &mut StampContext<'_>,
    [p, n]: [NodeId; 2],
    value: Real,
) -> SpiceResult<()> {
    context.stamp_fixed([(p, p, value), (p, n, -value), (n, p, -value), (n, n, value)])
}

pub(super) fn stamp_conductance(
    terminals: [NodeId; 2],
    conductance: Real,
    context: &mut StampContext<'_>,
) -> SpiceResult<()> {
    if context.mode.is_ac() {
        return Err(SpiceError::circuit("use complex equation assembly for AC"));
    }
    stamp_nodal(context, terminals, conductance)
}
pub(super) fn assemble_conductance(
    terminals: [NodeId; 2],
    conductance: Real,
    context: &mut crate::devices::linear::LinearContext<'_>,
) -> SpiceResult<()> {
    context.nodal(terminals, conductance, false)
}

/// Evaluate the resistor's generators at the operating point.
pub(super) fn resistor_noise<'a>(
    terminals: [NodeId; 2],
    conductance: Real,
    n: &ResistorNoise,
    model: Option<&'a str>,
    context: &NoiseContext<'_>,
    temperature_kelvin: Option<Real>,
    sources: &mut Vec<NoiseSource>,
) -> DeviceNoise<'a> {
    if !n.noisy {
        return DeviceNoise::Noiseless;
    }
    let current = (context.voltage(terminals[0]) - context.voltage(terminals[1])) * conductance;
    let temperature = temperature_kelvin.unwrap_or_else(|| {
        n.temperature
            .map_or_else(|| context.circuit_kelvin(), |t| t + CELSIUS_TO_KELVIN)
    });
    // resnoise.c: `RESm * RESfNcoef * pow(fabs(REScurrent / RESm), RESfNexp)
    // / (RESeffNoiseArea * pow(freq, RESef))`.
    let numerator = n.multiplicity * n.kf * (current / n.multiplicity).abs().powf(n.af);
    sources.extend([
        NoiseSource::new(
            "_thermal",
            terminals,
            NoiseKind::Thermal {
                conductance: if n.absolute_thermal_conductance {
                    conductance.abs()
                } else {
                    conductance
                },
                temperature,
            },
        ),
        NoiseSource::new(
            "_1overf",
            terminals,
            NoiseKind::Flicker {
                numerator,
                law: FlickerLaw::Area {
                    area: n.area,
                    exponent: n.ef,
                },
            },
        ),
    ]);
    DeviceNoise::Sources {
        family: NoiseFamily::Resistor,
        model,
        total: true,
    }
}

/// Borrowed inputs to the charge/flux equation kernels. The owning device
/// supplies a validated effective value and IC; no identity or history is
/// allocated here. The two families keep their distinct C stamp ordering.
pub(super) struct ReactiveLoad<'a> {
    pub(super) bias: Option<&'a crate::devices::passive_bias::PassiveInitialization>,
    pub(super) name: &'a str,
    pub(super) terminals: [NodeId; 2],
    pub(super) value: Real,
    pub(super) initial: Option<Real>,
}

impl ReactiveLoad<'_> {
    /// Perform `NIintegrate` and publish the trial charge/flux and derivative.
    /// MODEINITTRAN patches state1 before integration and its derivative
    /// afterwards; the trial owns those patches until acceptance.
    fn companion(
        &self,
        context: &mut StampContext<'_>,
        coefficients: &Coefficients,
        quantity: Real,
        initial: bool,
    ) -> SpiceResult<Companion> {
        if initial {
            context.states.patch_state1(QUANTITY, quantity)?;
        }
        let companion = integrate(context, coefficients, quantity, self.value, self.name)?;
        if initial {
            context
                .states
                .patch_state1(DERIVATIVE, companion.derivative)?;
        }
        context.states.set(QUANTITY, quantity)?;
        context.states.set(DERIVATIVE, companion.derivative)?;
        Ok(companion)
    }

    pub(super) fn stamp_capacitor(&self, context: &mut StampContext<'_>) -> SpiceResult<()> {
        let [positive, negative] = self.terminals;
        let v = context.node_voltage(positive) - context.node_voltage(negative);
        let charge = self.value * v;
        if let Some(bias) = self.bias {
            bias.stamp_capacitor(self.terminals, context)?;
        }
        if context.mode.is_dc() {
            return record_dc_state(context, charge);
        }
        if context.mode.is_ac() {
            return Err(SpiceError::circuit("use complex equation assembly for AC"));
        }
        let coefficients = companion_coefficients(context, self.name)?;
        // capload.c: MODEINITPRED keeps CKTstate1's charge; MODEINITTRAN
        // copies the charge, then the integrated current, into CKTstate1.
        let (initial, predicting) = transient_phase(context);
        let charge = if predicting {
            context
                .states
                .accepted(1, QUANTITY)
                .ok_or_else(|| SpiceError::circuit(format!("{}: no accepted charge", self.name)))?
        } else if initial && context.states.uic() {
            // cond1 (MODEUIC && MODEINITTRAN): vcap = CAPinitCond, which
            // CAPgetic defaults to the node voltages of the uic start vector
            // (the .ic values), the point this load sees.
            self.value * self.initial.unwrap_or(v)
        } else {
            charge
        };
        let companion = self.companion(context, coefficients, charge, initial)?;
        stamp_nodal(context, self.terminals, companion.conductance)?;
        context.stamp_rhs(positive, -companion.current)?;
        context.stamp_rhs(negative, companion.current)
    }
    pub(super) fn assemble_capacitor(
        &self,
        context: &mut crate::devices::linear::LinearContext<'_>,
    ) -> SpiceResult<()> {
        context.system.has_initial_conditions |= self.initial.is_some();
        if let Some(bias) = self.bias {
            bias.release_capacitor(context)?;
        }
        context.nodal(self.terminals, self.value, true)
    }

    pub(super) fn stamp_inductor(&self, context: &mut StampContext<'_>) -> SpiceResult<()> {
        let row = context.branch(0)?;
        let mut flux = self.value * context.row_value(row)?;
        for term in context.mutual {
            let mutual_flux = term.inductance * context.row_value(term.row)?;
            flux += mutual_flux;
        }
        if context.mode.is_dc() {
            record_dc_state(context, flux)?;
            if let Some(bias) = self.bias
                && bias.stamp_inductor(self.terminals, row, context)?
            {
                return Ok(());
            }
            return crate::devices::linear::branch_stamp(
                context.matrix,
                context.unknowns,
                self.terminals,
                row,
            );
        }
        if context.mode.is_ac() {
            return Err(SpiceError::circuit("use complex equation assembly for AC"));
        }
        let coefficients = companion_coefficients(context, self.name)?;
        // indload.c: MODEINITPRED keeps CKTstate1's flux (the mutual flux
        // is not added); MODEINITTRAN copies the flux, then the integrated
        // voltage, into CKTstate1.
        let (initial, predicting) = transient_phase(context);
        let flux = if predicting {
            context
                .states
                .accepted(1, QUANTITY)
                .ok_or_else(|| SpiceError::circuit(format!("{}: no accepted flux", self.name)))?
        } else if initial && context.states.uic() {
            // MODEUIC && MODEINITTRAN: INDinduct/m * INDinitCond, plus each
            // MUTfactor * the coupled inductor's INDinitCond.
            // INDgetic defaults INDinitCond to the uic start vector's value.
            let mut flux = self.value * self.initial.unwrap_or(context.row_value(row)?);
            for term in context.mutual {
                let mutual_flux = term.inductance * term.initial;
                flux += mutual_flux;
            }
            flux
        } else {
            flux
        };
        let companion = self.companion(context, coefficients, flux, initial)?;
        crate::devices::linear::branch_stamp(
            context.matrix,
            context.unknowns,
            self.terminals,
            row,
        )?;
        context.matrix.add(row, row, -companion.conductance)?;
        let ag0 = coefficients.ag()[0];
        for term in context.mutual {
            context.matrix.add(row, term.row, -ag0 * term.inductance)?;
        }
        context.rhs.add_to(row, companion.current)
    }
    /// `v+ - v- - L di/dt - sum(M di_k/dt) = 0`: the mutual terms are
    /// off-diagonal `E` entries between branch rows (`mutacld.c` stamps
    /// `-j omega M` there in AC).
    pub(super) fn assemble_inductor(
        &self,
        context: &mut crate::devices::linear::LinearContext<'_>,
    ) -> SpiceResult<()> {
        let branch = context.branch(self.terminals)?;
        // v+ - v- - L di/dt = 0, i positive from + to -.
        context.system.e.add(branch, branch, -self.value)?;
        for term in context.mutual {
            context.system.e.add(branch, term.row, -term.inductance)?;
        }
        context.system.has_initial_conditions |= self.initial.is_some();
        Ok(())
    }
}

/// `CAPacLoad`: multiply by frequency before applying the parallel multiplier.
pub(super) fn assemble_capacitor_ac(
    terminals: [NodeId; 2],
    value: Real,
    m: Real,
    context: &mut crate::devices::linear::LinearContext<'_>,
) -> SpiceResult<()> {
    let [p, n] = terminals;
    let terms = [value];
    context.scaled_dynamic_sum(p, p, &terms, m)?;
    context.scaled_dynamic_sum(n, n, &terms, m)?;
    context.scaled_dynamic_sum(p, n, &terms, -m)?;
    context.scaled_dynamic_sum(n, p, &terms, -m)
}

/// `INDacLoad`: multiply by frequency before dividing by the parallel multiplier.
pub(super) fn assemble_inductor_ac(
    terminals: [NodeId; 2],
    value: Real,
    m: Real,
    context: &mut crate::devices::linear::LinearContext<'_>,
) -> SpiceResult<()> {
    let branch = context.branch(terminals)?;
    context.dynamic_group(branch, branch, &[value], -1., m)?;
    for term in context.mutual {
        context.system.e.add(branch, term.row, -term.inductance)?;
    }
    Ok(())
}

/// A capacitor, `c1 n1 n2 <value> [ic=…]`.
#[derive(Debug, Clone, PartialEq)]
pub struct Capacitor {
    name: String,
    terminals: [NodeId; 2],
    /// Capacitance in farads.
    capacitance: Real,
    /// Initial voltage, from `ic=`; `None` when not given.
    initial_voltage: Option<Real>,
    bias: Option<Box<crate::devices::passive_bias::PassiveInitialization>>,
}

impl Capacitor {
    /// Builds a capacitor.
    ///
    /// # Errors
    ///
    /// [`SpiceError::Circuit`] for negative/non-finite capacitance or non-finite
    /// IC. A zero capacitance is accepted, as in C (it stamps nothing).
    pub fn new(
        name: impl Into<String>,
        terminals: [NodeId; 2],
        capacitance: Real,
        initial_voltage: Option<Real>,
    ) -> SpiceResult<Self> {
        let name = name.into();
        if !capacitance.is_finite()
            || capacitance < 0.0
            || initial_voltage.is_some_and(|v| !v.is_finite())
        {
            return Err(SpiceError::circuit(format!(
                "capacitor {name}: capacitance must be nonnegative and finite; initial voltage must be finite"
            )));
        }
        Ok(Self {
            name,
            terminals,
            capacitance,
            initial_voltage,
            bias: None,
        })
    }

    fn load(&self) -> ReactiveLoad<'_> {
        ReactiveLoad {
            name: &self.name,
            terminals: self.terminals,
            value: self.capacitance,
            initial: crate::devices::passive_bias::initial_value(
                self.bias.as_deref(),
                self.initial_voltage,
            ),
            bias: self.bias.as_deref(),
        }
    }

    /// The capacitance in farads.
    #[must_use = "the returned capacitance determines stored charge per volt"]
    pub const fn capacitance(&self) -> Real {
        self.capacitance
    }

    /// The `ic=` initial voltage, if any.
    #[must_use = "the result supplies the capacitor initial voltage when specified"]
    pub const fn initial_voltage(&self) -> Option<Real> {
        self.initial_voltage
    }
}

impl Device for Capacitor {
    /// `capsetup.c`.
    fn bind_passive_initialization(
        &mut self,
        policy: crate::devices::passive_bias::PassiveInitialization,
    ) -> SpiceResult<()> {
        policy.check_kind(crate::devices::traits::StorageKind::Capacitor)?;
        self.bias = Some(Box::new(policy));
        Ok(())
    }

    fn has_passive_bias(&self) -> bool {
        crate::devices::passive_bias::has_constraints(self.bias.as_deref())
    }

    fn setup_matrix(
        &self,
        setup: &mut crate::devices::setup::MatrixSetup<'_>,
    ) -> SpiceResult<bool> {
        setup.conductance(self.terminals[0], self.terminals[1]);
        if crate::devices::passive_bias::needs_branch(self.bias.as_deref()) {
            setup.inductor(self.terminals[0], self.terminals[1], setup.branch(0)?);
        }
        Ok(true)
    }
    fn observation_parameter(
        &self,
        keyword: &str,
        context: &crate::devices::ModelContext,
    ) -> SpiceResult<Option<Real>> {
        Ok(match keyword {
            "c" | "cap" | "capacitance" => Some(self.capacitance),
            "temp" => Some(context.temperature),
            "m" | "scale" => Some(1.),
            "tc1" | "tc2" | "dtemp" => Some(0.),
            _ => None,
        })
    }
    fn parameter_geometry_scale(
        &self,
        key: crate::devices::sweep::ParameterKey,
    ) -> crate::devices::sweep::GeometryScale {
        use crate::devices::sweep::{GeometryScale, ParameterKey};
        if matches!(key, ParameterKey::Native("l" | "w")) {
            GeometryScale::Length
        } else {
            GeometryScale::Identity
        }
    }

    fn instance_parameter(&self, keyword: &str) -> Option<&'static str> {
        [
            "capacitance",
            "c",
            "cap",
            "ic",
            "temp",
            "tc1",
            "tc2",
            "w",
            "l",
            "m",
            "scale",
        ]
        .into_iter()
        .find(|k| k.eq_ignore_ascii_case(keyword))
        .map(|k| match k {
            "r" => "resistance",
            "c" | "cap" => "capacitance",
            other => other,
        })
    }
    fn with_instance_parameter(
        &self,
        parameter: &str,
        value: Real,
        context: &crate::devices::ModelContext,
    ) -> SpiceResult<Box<dyn Device>> {
        let mut replacement = crate::devices::passive::swept_literal(
            &self.name,
            self.terminals,
            crate::devices::passive::PassiveKind::Capacitor,
            self.capacitance,
            self.initial_voltage,
            (parameter, value),
            context,
        )?;
        if let Some(policy) = &self.bias {
            replacement.bind_passive_initialization((**policy).clone())?;
        }
        Ok(replacement)
    }

    /// Linear in `.disto`: C gives this device no distortion routine
    /// (`DEVdisto = NULL`, `cap/capinit.c`), so it enters only through its
    /// small-signal matrix.
    fn distortion(
        &self,
        _context: &crate::devices::distortion::DistortionContext<'_>,
    ) -> crate::primitives::SpiceResult<crate::devices::distortion::DeviceDistortion> {
        Ok(crate::devices::distortion::DeviceDistortion::Linear)
    }

    /// Noiseless: C gives this device no noise routine (`DEVnoise = NULL`,
    /// `src/spicelib/devices/cap/capinit.c`).
    fn noise(
        &self,
        _context: &crate::devices::noise::NoiseContext<'_>,
        _sources: &mut Vec<crate::devices::noise::NoiseSource>,
    ) -> crate::primitives::SpiceResult<crate::devices::noise::DeviceNoise<'_>> {
        Ok(crate::devices::noise::DeviceNoise::Noiseless)
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn designator(&self) -> char {
        'c'
    }

    fn terminals(&self) -> &[NodeId] {
        &self.terminals
    }

    /// Reserve an exact voltage constraint branch only when a prepared
    /// analysis needs one. Native capacitors add no branch.
    fn branch_currents(&self) -> usize {
        usize::from(crate::devices::passive_bias::needs_branch(
            self.bias.as_deref(),
        ))
    }

    fn branch_quantity(&self, _branch: usize) -> crate::devices::BranchQuantity {
        crate::devices::BranchQuantity::BiasCurrent
    }

    /// Charge `q = C v` and current `dq/dt` (C `CAPqcap`/`CAPccap`).
    fn state_count(&self) -> usize {
        REACTIVE_STATE_COUNT
    }

    fn truncation_slot(&self) -> Option<usize> {
        Some(QUANTITY)
    }

    fn storage_element(
        &self,
        _context: &crate::devices::models::ModelContext,
    ) -> Option<SpiceResult<crate::devices::traits::StorageElement>> {
        Some(Ok(crate::devices::traits::StorageElement {
            kind: crate::devices::traits::StorageKind::Capacitor,
            value: self.capacitance,
            initial: self.load().initial,
        }))
    }

    /// `capload.c`: an open circuit at DC (recording `q = C v` when state
    /// is tracked); in transient, the Norton companion `i = geq v + ceq`
    /// from current from the first terminal to the second. `ic=` is not
    /// applied here; initial-condition policy belongs to the analysis.
    fn stamp(&self, context: &mut StampContext<'_>) -> SpiceResult<()> {
        self.load().stamp_capacitor(context)
    }
    fn assemble_linear(
        &self,
        context: &mut crate::devices::linear::LinearContext<'_>,
    ) -> SpiceResult<()> {
        self.load().assemble_capacitor(context)
    }

    /// `.sens`: C's capacitor records with the default model
    /// ([`crate::devices::sensitivity`]).
    fn sensitivity(
        &self,
        context: &crate::devices::models::ModelContext,
    ) -> SpiceResult<Box<dyn crate::devices::sensitivity::DeviceSensitivity + '_>> {
        use crate::devices::sensitivity::{PassiveInputs, ReactiveKind, ReactiveSensitivity};
        Ok(Box::new(ReactiveSensitivity::new(
            ReactiveKind::Capacitor,
            &self.name,
            self.terminals,
            PassiveInputs::Literal {
                value: self.capacitance,
                initial_condition: self.initial_voltage,
            },
            context,
            crate::devices::passive_bias::needs_branch(self.bias.as_deref()),
        )))
    }

    /// Pole-zero load: C `cappzld.c` equals the AC load with `s` for `j omega`.
    fn assemble_pole_zero(
        &self,
        context: &mut crate::devices::linear::LinearContext<'_>,
        bias: &crate::maths::Vector,
    ) -> crate::primitives::SpiceResult<()> {
        self.assemble_small_signal(context, bias)
    }
}

/// An inductor, `l1 n1 n2 <value> [ic=…]`.
///
/// An inductor is a short at DC, which makes it a voltage-source-like element:
/// it contributes a branch-current unknown, like `V` sources do.
#[derive(Debug, Clone, PartialEq)]
pub struct Inductor {
    name: String,
    terminals: [NodeId; 2],
    /// Inductance in henries.
    inductance: Real,
    /// Initial current, from `ic=`; `None` when not given.
    initial_current: Option<Real>,
    bias: Option<Box<crate::devices::passive_bias::PassiveInitialization>>,
}

impl Inductor {
    /// Builds an inductor.
    ///
    /// # Errors
    ///
    /// [`SpiceError::Circuit`] for negative/non-finite inductance or non-finite
    /// IC. A zero inductance is accepted, as in C (a short at every analysis).
    pub fn new(
        name: impl Into<String>,
        terminals: [NodeId; 2],
        inductance: Real,
        initial_current: Option<Real>,
    ) -> SpiceResult<Self> {
        let name = name.into();
        if !inductance.is_finite()
            || inductance < 0.0
            || initial_current.is_some_and(|v| !v.is_finite())
        {
            return Err(SpiceError::circuit(format!(
                "inductor {name}: inductance must be nonnegative and finite; initial current must be finite"
            )));
        }
        Ok(Self {
            name,
            terminals,
            inductance,
            initial_current,
            bias: None,
        })
    }

    fn load(&self) -> ReactiveLoad<'_> {
        ReactiveLoad {
            name: &self.name,
            terminals: self.terminals,
            value: self.inductance,
            initial: crate::devices::passive_bias::initial_value(
                self.bias.as_deref(),
                self.initial_current,
            ),
            bias: self.bias.as_deref(),
        }
    }

    /// The inductance in henries.
    #[must_use = "the returned inductance determines stored flux per ampere"]
    pub const fn inductance(&self) -> Real {
        self.inductance
    }

    /// The `ic=` initial current, if any.
    #[must_use = "the result supplies the inductor initial current when specified"]
    pub const fn initial_current(&self) -> Option<Real> {
        self.initial_current
    }
}

impl Device for Inductor {
    /// `indsetup.c`.
    fn bind_passive_initialization(
        &mut self,
        policy: crate::devices::passive_bias::PassiveInitialization,
    ) -> SpiceResult<()> {
        policy.check_kind(crate::devices::traits::StorageKind::Inductor)?;
        self.bias = Some(Box::new(policy));
        Ok(())
    }

    fn has_passive_bias(&self) -> bool {
        crate::devices::passive_bias::has_constraints(self.bias.as_deref())
    }

    fn setup_matrix(
        &self,
        setup: &mut crate::devices::setup::MatrixSetup<'_>,
    ) -> SpiceResult<bool> {
        let branch = setup.branch(0)?;
        setup.inductor(self.terminals[0], self.terminals[1], branch);
        Ok(true)
    }
    fn observation_parameter(
        &self,
        keyword: &str,
        context: &crate::devices::ModelContext,
    ) -> SpiceResult<Option<Real>> {
        Ok(match keyword {
            "l" | "inductance" => Some(self.inductance),
            "temp" => Some(context.temperature),
            "m" | "scale" => Some(1.),
            "tc1" | "tc2" | "dtemp" => Some(0.),
            _ => None,
        })
    }
    fn instance_parameter(&self, keyword: &str) -> Option<&'static str> {
        ["inductance", "ic", "temp", "tc1", "tc2", "m", "scale"]
            .into_iter()
            .find(|k| k.eq_ignore_ascii_case(keyword))
            .map(|k| match k {
                "r" => "resistance",
                "c" | "cap" => "capacitance",
                other => other,
            })
    }
    fn with_instance_parameter(
        &self,
        parameter: &str,
        value: Real,
        context: &crate::devices::ModelContext,
    ) -> SpiceResult<Box<dyn Device>> {
        let mut replacement = crate::devices::passive::swept_literal(
            &self.name,
            self.terminals,
            crate::devices::passive::PassiveKind::Inductor,
            self.inductance,
            self.initial_current,
            (parameter, value),
            context,
        )?;
        if let Some(policy) = &self.bias {
            replacement.bind_passive_initialization((**policy).clone())?;
        }
        Ok(replacement)
    }

    /// Linear in `.disto`: C gives this device no distortion routine
    /// (`DEVdisto = NULL`, `ind/indinit.c`), so it enters only through its
    /// small-signal matrix.
    fn distortion(
        &self,
        _context: &crate::devices::distortion::DistortionContext<'_>,
    ) -> crate::primitives::SpiceResult<crate::devices::distortion::DeviceDistortion> {
        Ok(crate::devices::distortion::DeviceDistortion::Linear)
    }

    /// Noiseless: C gives this device no noise routine (`DEVnoise = NULL`,
    /// `src/spicelib/devices/ind/indinit.c`).
    fn noise(
        &self,
        _context: &crate::devices::noise::NoiseContext<'_>,
        _sources: &mut Vec<crate::devices::noise::NoiseSource>,
    ) -> crate::primitives::SpiceResult<crate::devices::noise::DeviceNoise<'_>> {
        Ok(crate::devices::noise::DeviceNoise::Noiseless)
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn designator(&self) -> char {
        'l'
    }

    fn terminals(&self) -> &[NodeId] {
        &self.terminals
    }

    fn branch_currents(&self) -> usize {
        1
    }

    fn expression_current_branch(&self) -> Option<usize> {
        Some(0)
    }

    /// Flux `L i` and voltage `dflux/dt` (C `INDflux`/`INDvolt`).
    fn state_count(&self) -> usize {
        REACTIVE_STATE_COUNT
    }

    fn truncation_slot(&self) -> Option<usize> {
        Some(QUANTITY)
    }

    fn storage_element(
        &self,
        _context: &crate::devices::models::ModelContext,
    ) -> Option<SpiceResult<crate::devices::traits::StorageElement>> {
        Some(Ok(crate::devices::traits::StorageElement {
            kind: crate::devices::traits::StorageKind::Inductor,
            value: self.inductance,
            initial: self.load().initial,
        }))
    }

    fn inductance(
        &self,
        _context: &crate::devices::models::ModelContext,
    ) -> Option<SpiceResult<crate::devices::traits::InductanceValue>> {
        Some(Ok(crate::devices::traits::InductanceValue {
            effective: self.inductance,
            coupling_base: self.inductance,
            initial: self.load().initial.unwrap_or(0.),
        }))
    }

    /// `indload.c`: a short at DC (recording `flux = L i` when state is
    /// tracked); in transient, the branch row `v+ - v- - req i = veq` with
    /// `i` positive from the first terminal to the second. `ic=` is not
    /// applied here; initial-condition policy belongs to the analysis.
    ///
    /// With mutual coupling ([`StampContext::mutual`]) the flux is
    /// `L i + sum(M i_k)`, as `indload.c` accumulates it in `INDflux`, and
    /// the companion row gains `-ag0 M` in each coupled branch column (C
    /// `MUTbr1br2Ptr`/`MUTbr2br1Ptr`): `v+ - v- - ag0 (L i + sum(M i_k)) =
    /// veq`, with `veq` from the integrated coupled flux. At DC the coupled
    /// inductors stay shorts.
    fn stamp(&self, context: &mut StampContext<'_>) -> SpiceResult<()> {
        self.load().stamp_inductor(context)
    }
    fn assemble_linear(
        &self,
        context: &mut crate::devices::linear::LinearContext<'_>,
    ) -> SpiceResult<()> {
        self.load().assemble_inductor(context)
    }

    /// `.sens`: C's inductor records with the default model
    /// ([`crate::devices::sensitivity`]).
    fn sensitivity(
        &self,
        context: &crate::devices::models::ModelContext,
    ) -> SpiceResult<Box<dyn crate::devices::sensitivity::DeviceSensitivity + '_>> {
        use crate::devices::sensitivity::{PassiveInputs, ReactiveKind, ReactiveSensitivity};
        Ok(Box::new(ReactiveSensitivity::new(
            ReactiveKind::Inductor,
            &self.name,
            self.terminals,
            PassiveInputs::Literal {
                value: self.inductance,
                initial_condition: self.initial_current,
            },
            context,
            false,
        )))
    }

    /// Pole-zero load: C `indpzld.c` (with `mutpzld.c` mutual terms) equals the AC load with `s` for `j omega`.
    fn assemble_pole_zero(
        &self,
        context: &mut crate::devices::linear::LinearContext<'_>,
        bias: &crate::maths::Vector,
    ) -> crate::primitives::SpiceResult<()> {
        self.assemble_small_signal(context, bias)
    }
}

#[cfg(test)]
mod tests {
    use super::{Capacitor, Inductor, Resistor};
    use crate::devices::traits::Device;
    use crate::primitives::NodeId;

    fn nodes() -> [NodeId; 2] {
        [NodeId::new(1), NodeId::GROUND]
    }

    #[test]
    fn a_resistor_knows_its_conductance() {
        let resistor = Resistor::new("r1", nodes(), 1000.0).unwrap();
        assert_eq!(resistor.name(), "r1");
        assert_eq!(resistor.designator(), 'r');
        assert_eq!(resistor.terminals(), &nodes());
        assert_eq!(resistor.resistance(), 1000.0);
        assert_eq!(resistor.conductance(), 0.001);
        assert_eq!(resistor.branch_currents(), 0);
        assert!(!resistor.is_nonlinear());
    }

    #[test]
    fn a_capacitor_records_its_initial_condition() {
        let capacitor = Capacitor::new("c1", nodes(), 1e-6, Some(1.5)).unwrap();
        assert_eq!(capacitor.capacitance(), 1e-6);
        assert_eq!(capacitor.initial_voltage(), Some(1.5));
        assert_eq!(capacitor.designator(), 'c');
        assert_eq!(
            Capacitor::new("c1", nodes(), 1e-6, None)
                .unwrap()
                .initial_voltage(),
            None
        );
    }

    #[test]
    fn an_inductor_adds_a_branch_current() {
        let inductor = Inductor::new("l1", nodes(), 1e-3, Some(0.2)).unwrap();
        assert_eq!(inductor.inductance(), 1e-3);
        assert_eq!(inductor.initial_current(), Some(0.2));
        assert_eq!(inductor.designator(), 'l');
        assert_eq!(inductor.branch_currents(), 1);
    }

    #[test]
    fn non_finite_values_are_rejected_at_construction() {
        assert!(Resistor::new("r1", nodes(), f64::INFINITY).is_err());
        assert!(Capacitor::new("c1", nodes(), f64::NAN, None).is_err());
        assert!(Inductor::new("l1", nodes(), f64::INFINITY, None).is_err());
    }

    #[test]
    fn companion_stamping_requires_a_companion_transient_load() {
        let capacitor = Capacitor::new("c1", nodes(), 1e-6, None).unwrap();
        assert!(stamp_error(&capacitor).contains("without integration coefficients"));
        let inductor = Inductor::new("l1", nodes(), 1e-3, None).unwrap();
        assert!(stamp_error(&inductor).contains("missing branch-row binding"));
        assert_eq!(capacitor.state_count(), 2);
        assert_eq!(inductor.state_count(), 2);
    }

    fn stamp_error(device: &dyn Device) -> String {
        use crate::maths::{SparseMatrix, Vector};
        use crate::primitives::NodeTable;
        let nodes = NodeTable::new();
        let unknowns = crate::devices::traits::MnaUnknowns::new();
        let mut matrix = SparseMatrix::new(1, 1);
        let mut rhs = Vector::zeros(1);
        let solution = Vector::zeros(1);
        let mut context = crate::devices::traits::StampContext {
            matrix: &mut matrix,
            rhs: crate::devices::traits::StampRhs::dense(&mut rhs),
            unknowns: &unknowns,
            nodes: &nodes,
            solution: &solution,
            temperature: 27.0,
            nominal_temperature: 27.0,
            gmin: crate::devices::DEFAULT_GMIN,
            reltol: crate::devices::DEFAULT_RELTOL,
            frequency: 0.,
            mode: crate::devices::traits::AnalysisMode::Transient { time: 0., dt: 1e-6 },
            bias_analysis: None,
            branches: 0..0,
            controls: &[],
            mutual: &[],
            integration: None,
            states: crate::devices::state::DeviceState::none(),
            forcing: None,
            source_factor: 1.,
            voltage_source_value: &mut 0.,
        };
        device.stamp(&mut context).unwrap_err().to_string()
    }
}
