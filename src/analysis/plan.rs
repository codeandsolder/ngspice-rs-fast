//! Resolution of analysis options at the input boundary.
//!
//! Deck setters have already been evaluated and applied in source order.
//! Literal request overrides are decoded once into the same typed domains;
//! no deck option goes through a formatted argument on its way to a solver.

use crate::analysis::AnalysisInput;
use crate::analysis::bias::{
    ContinuationPolicy, DcSettings, IterationLimit, MAX_GMIN_STAGES, MAX_SOURCE_STEPS,
};
use crate::analysis::config::{C_DEFAULT_ITL2_EFFECTIVE, RunConfig};
use crate::analysis::linear::unsupported;
use crate::analysis::newton::{MAX_ITERATIONS, NewtonOptions, StepLimiting};
use crate::maths::integrator::{DEFAULT_XMU, IntegrationMethod};
use crate::primitives::{
    AnalysisKind, Real, SpiceResult, exact_nonnegative_integer, parse_spice_number,
};

const DEFAULT_MAX_STEPS: usize = 1_000_000;
const MAX_STEPS_LIMIT: usize = 10_000_000;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
enum Schedule {
    #[default]
    Ngspice,
    Ladder,
}

/// Integration method independent of the deck's separate maximum order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IntegrationFamily {
    /// Trapezoidal integration.
    Trapezoidal,
    /// Gear backward differentiation.
    Gear,
}

impl IntegrationFamily {
    pub(super) const fn from_name(text: &str) -> Option<Self> {
        match IntegrationMethod::parse(text, 2) {
            Some(IntegrationMethod::Trapezoidal) => Some(Self::Trapezoidal),
            Some(IntegrationMethod::Gear { .. }) => Some(Self::Gear),
            None => None,
        }
    }

    fn parse(text: &str) -> SpiceResult<Self> {
        Self::from_name(text).ok_or_else(|| {
            if text.eq_ignore_ascii_case("bdf") {
                unsupported("method=bdf: ngspice has no BDF integrator (the diffsol backend was removed); the companion methods are trap and gear")
            } else {
                unsupported(format!("unknown transient method '{text}'; expected trap, trapezoidal or gear"))
            }
        })
    }

    const fn with_order(self, order: u8) -> IntegrationMethod {
        match self {
            Self::Trapezoidal => IntegrationMethod::Trapezoidal,
            Self::Gear => IntegrationMethod::Gear { order },
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Default)]
struct SolverOverrides {
    reltol: Option<Real>,
    vntol: Option<Real>,
    abstol: Option<Real>,
    direct: Option<usize>,
    stage: Option<usize>,
    adapt: Option<usize>,
    source: Option<usize>,
    gmin_steps: Option<usize>,
    gmin_factor: Option<Real>,
    gshunt: Option<Real>,
    skip: Option<bool>,
    schedule: Schedule,
    limiting: Option<StepLimiting>,
}

impl SolverOverrides {
    fn resolve(self, config: &RunConfig) -> SpiceResult<DcSettings> {
        let dc = config.dc();
        let tran = config.transient();
        let mut newton = NewtonOptions::default();
        newton.reltol = self.reltol.unwrap_or_else(|| config.context().reltol);
        newton.vntol = self.vntol.or(tran.vntol).unwrap_or(newton.vntol);
        newton.abstol = self.abstol.or(tran.abstol).unwrap_or(newton.abstol);
        newton.max_iterations = self.direct.or(dc.itl1).unwrap_or(newton.max_iterations);
        newton.limiting = self.limiting.unwrap_or(newton.limiting);
        newton.validate()?;

        let source = self.source.or(dc.srcsteps);
        let gmin_steps = self.gmin_steps.or(dc.gminsteps);
        let factor = self.gmin_factor.or(dc.gminfactor);
        let stage = self.stage.or_else(|| {
            (dc.itl1.is_some() || dc.itl2.is_some())
                .then(|| dc.itl2.unwrap_or(C_DEFAULT_ITL2_EFFECTIVE))
        });
        let mut continuation = match self.schedule {
            Schedule::Ngspice => ContinuationPolicy::from_ngspice_steps(
                source,
                gmin_steps,
                factor,
                self.adapt.or(dc.itl2_written),
            )?,
            Schedule::Ladder => {
                if self.adapt.is_some() {
                    return Err(unsupported(
                        "adaptiter steers the ngspice schedule only, not continuation=ladder",
                    ));
                }
                ContinuationPolicy::from_steps(source, gmin_steps, factor)?
            }
        };
        continuation.stage_max_iterations = stage.map(IterationLimit::new).transpose()?;
        continuation.skip_direct = self.skip.unwrap_or(dc.noopiter);
        continuation.gshunt = self.gshunt.or(dc.gshunt).unwrap_or(0.);
        Ok(DcSettings {
            newton,
            continuation,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Default)]
struct TransientOverrides {
    method: Option<IntegrationFamily>,
    max_order: Option<u8>,
    chgtol: Option<Real>,
    trtol: Option<Real>,
    minbreak: Option<Real>,
    voltage_lte: crate::analysis::config::VoltageLteOptions,
    max_steps: Option<usize>,
    tran_max_iter: Option<usize>,
    xmu: Option<Real>,
}

/// Resolved voltage predictor tolerances, independent of charge tolerances.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct VoltageLte {
    pub relative: Real,
    pub absolute: Real,
    pub factor: Real,
}

impl VoltageLte {
    fn resolve(
        local: crate::analysis::config::VoltageLteOptions,
        deck: crate::analysis::config::VoltageLteOptions,
    ) -> Option<Self> {
        local
            .enabled
            .or(deck.enabled)
            .unwrap_or(false)
            .then(|| Self {
                relative: local.relative.or(deck.relative).unwrap_or(1e-3),
                absolute: local.absolute.or(deck.absolute).unwrap_or(1e-6),
                factor: local.factor.or(deck.factor).unwrap_or(500.),
            })
    }
}

/// The transient-only part of a resolved plan, independent of time bounds.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct TransientOptions {
    pub method: IntegrationMethod,
    pub max_order: u8,
    pub chgtol: Real,
    pub trtol: Real,
    pub minbreak: Real,
    pub voltage_lte: Option<VoltageLte>,
    pub max_steps: usize,
    pub tran_max_iter: usize,
    pub xmu: Real,
}

impl TransientOverrides {
    fn resolve(self, config: &RunConfig) -> SpiceResult<TransientOptions> {
        let deck = config.transient();
        let max_order = self.max_order.or_else(|| config.maxord()).unwrap_or(2);
        let method = self
            .method
            .or_else(|| config.method())
            .unwrap_or(IntegrationFamily::Trapezoidal)
            .with_order(max_order);
        method.validate_runtime()?;
        Ok(TransientOptions {
            method,
            max_order,
            chgtol: self.chgtol.or(deck.chgtol).unwrap_or(1e-14),
            trtol: self.trtol.or(deck.trtol).unwrap_or(7.),
            minbreak: self.minbreak.or(deck.minbreak).unwrap_or(0.),
            voltage_lte: VoltageLte::resolve(self.voltage_lte, deck.voltage_lte),
            max_steps: self.max_steps.unwrap_or(DEFAULT_MAX_STEPS),
            tran_max_iter: self.tran_max_iter.or(deck.itl4).unwrap_or(100),
            xmu: self.xmu.or(deck.xmu).unwrap_or(DEFAULT_XMU),
        })
    }
}

/// The analysis-specific options that cannot be passed to another kind.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) enum SpecificOptions {
    Ordinary,
    Dc {
        point_iterations: Option<usize>,
    },
    Transient(TransientOptions),
    HarmonicBalance {
        oversampling: u16,
        iterations: usize,
    },
}

/// One resolution of literal arguments and the deck's typed defaults.
#[derive(Debug, PartialEq)]
pub(super) struct ResolvedOptions {
    pub positional: Vec<String>,
    pub dc: DcSettings,
    pub specific: SpecificOptions,
}

/// Literal option decoding. The same explicit key cannot occur twice, even
/// with different case. Deck duplicates are independently last-writer-wins.
#[derive(Default)]
struct Overrides {
    solver: SolverOverrides,
    tran: TransientOverrides,
    point_iterations: Option<usize>,
    hb_oversampling: Option<u16>,
    hb_iterations: Option<usize>,
}

fn real(text: &str, key: &str) -> SpiceResult<Real> {
    parse_spice_number(text)
        .filter(|v| v.is_finite())
        .ok_or_else(|| unsupported(format!("nonfinite/nonliteral option {key}")))
}

fn positive(text: &str, key: &str) -> SpiceResult<Real> {
    let value = real(text, key)?;
    if value > 0. {
        Ok(value)
    } else {
        Err(unsupported(format!("option {key} must be positive")))
    }
}

pub(super) fn count(text: &str, key: &str, minimum: usize, maximum: usize) -> SpiceResult<usize> {
    exact_nonnegative_integer::<usize>(real(text, key)?)
        .filter(|n| (minimum..=maximum).contains(n))
        .ok_or_else(|| {
            unsupported(format!(
                "{key} must be an integer in {minimum}..={maximum}, not '{text}'"
            ))
        })
}

impl Overrides {
    fn set(&mut self, kind: AnalysisKind, key: &str, text: &str) -> SpiceResult<()> {
        if kind == AnalysisKind::HarmonicBalance {
            match key {
                "oversample" => {
                    self.hb_oversampling = Some(
                        u16::try_from(count(text, key, 2, 16)?)
                            .map_err(|_| unsupported("HB oversampling overflow"))?,
                    );
                    return Ok(());
                }
                "hbiter" => {
                    self.hb_iterations = Some(count(text, key, 1, 100)?);
                    return Ok(());
                }
                "abstol" | "vntol" => {}
                _ => return Err(unsupported(format!("HB option {key}"))),
            }
        }
        let solver = &mut self.solver;
        match key {
            "rtol" => solver.reltol = Some(positive(text, key)?),
            "vntol" => solver.vntol = Some(positive(text, key)?),
            "abstol" => solver.abstol = Some(positive(text, key)?),
            "maxiter" => solver.direct = Some(count(text, key, 1, MAX_ITERATIONS)?),
            "stagemaxiter" => solver.stage = Some(count(text, key, 1, MAX_ITERATIONS)?),
            "adaptiter" => solver.adapt = Some(count(text, key, 1, MAX_ITERATIONS)?),
            "srcsteps" => solver.source = Some(count(text, key, 0, MAX_SOURCE_STEPS)?),
            "gminsteps" => solver.gmin_steps = Some(count(text, key, 0, MAX_GMIN_STAGES)?),
            "gminfactor" => solver.gmin_factor = Some(real(text, key)?),
            "gshunt" => {
                let value = real(text, key)?;
                if value < 0. {
                    return Err(unsupported("option gshunt must be nonnegative"));
                }
                solver.gshunt = Some(value);
            }
            "noopiter" => solver.skip = Some(count(text, key, 0, 1)? == 1),
            "continuation" => {
                solver.schedule = match text.to_ascii_lowercase().as_str() {
                    "ngspice" => Schedule::Ngspice,
                    "ladder" => Schedule::Ladder,
                    _ => {
                        return Err(unsupported(format!(
                            "continuation must be 'ngspice' or 'ladder', not '{text}'"
                        )));
                    }
                };
            }
            "limiting" => {
                solver.limiting = Some(match text.to_ascii_lowercase().as_str() {
                    "device" => StepLimiting::Device,
                    "global" => StepLimiting::Global,
                    _ => {
                        return Err(unsupported(format!(
                            "limiting must be 'device' or 'global', not '{text}'"
                        )));
                    }
                });
            }
            "trcvmaxiter" if kind == AnalysisKind::DcSweep => {
                self.point_iterations = Some(count(text, key, 1, MAX_ITERATIONS)?);
            }
            _ if matches!(
                kind,
                AnalysisKind::Transient | AnalysisKind::PeriodicSteadyState
            ) =>
            {
                self.tran.set(key, text)?;
            }
            _ => return Err(unsupported(format!("convergence option {key}"))),
        }
        Ok(())
    }
}

impl TransientOverrides {
    fn set(&mut self, key: &str, text: &str) -> SpiceResult<()> {
        match key {
            "backend" if text.eq_ignore_ascii_case("companion") => {}
            "backend" if text.eq_ignore_ascii_case("diffsol") => {
                return Err(unsupported(
                    "backend=diffsol: the adaptive BDF backend was removed with diffsol; ngspice has \
                     no BDF integrator. Omit backend= to run the trapezoidal/Gear companion driver \
                     (dctran.c)",
                ));
            }
            "backend" => {
                return Err(unsupported(format!(
                    "unknown transient backend '{text}'; expected companion (default)"
                )));
            }
            "method" => self.method = Some(IntegrationFamily::parse(text)?),
            "maxord" => {
                self.max_order = Some(
                    text.parse::<u8>()
                        .ok()
                        .filter(|n| (1..=IntegrationMethod::MAX_GEAR_ORDER).contains(n))
                        .ok_or_else(|| {
                            unsupported(format!(
                                "maxord must be an integer in 1..={}, not '{text}'",
                                IntegrationMethod::MAX_GEAR_ORDER
                            ))
                        })?,
                );
            }
            "chgtol" => self.chgtol = Some(positive(text, key)?),
            "trtol" => self.trtol = Some(positive(text, key)?),
            "newtrunc" => self.voltage_lte.enabled = Some(count(text, key, 0, 1)? == 1),
            "ltereltol" => self.voltage_lte.relative = Some(positive(text, key)?),
            "lteabstol" => self.voltage_lte.absolute = Some(positive(text, key)?),
            "ltetrtol" => self.voltage_lte.factor = Some(positive(text, key)?),
            "minbreak" => {
                let value = real(text, key)?;
                if value < 0. {
                    return Err(unsupported("option minbreak must be nonnegative"));
                }
                self.minbreak = Some(value);
            }
            "maxsteps" => {
                self.max_steps = Some(
                    text.parse::<usize>()
                        .ok()
                        .filter(|n| (1..=MAX_STEPS_LIMIT).contains(n))
                        .ok_or_else(|| {
                            unsupported(format!(
                                "maxsteps must be an integer in 1..={MAX_STEPS_LIMIT}"
                            ))
                        })?,
                );
            }
            "tranmaxiter" => self.tran_max_iter = Some(count(text, key, 1, MAX_ITERATIONS)?),
            "xmu" => {
                self.xmu = Some(
                    parse_spice_number(text)
                        .filter(|v| (0. ..=0.5).contains(v))
                        .ok_or_else(|| {
                            unsupported(format!("xmu must be in [0, 0.5], not '{text}'"))
                        })?,
                );
            }
            _ => return Err(unsupported(format!("companion transient option '{key}'"))),
        }
        Ok(())
    }
}

pub(super) fn resolve_options(
    input: &AnalysisInput,
    config: &RunConfig,
) -> SpiceResult<ResolvedOptions> {
    let mut overrides = Overrides::default();
    let mut seen = std::collections::BTreeSet::new();
    let mut positional = Vec::with_capacity(input.arguments.len());
    for argument in &input.arguments {
        match argument.split_once('=') {
            Some((key, value)) => {
                let key = key.trim().to_ascii_lowercase();
                if !seen.insert(key.clone()) {
                    return Err(unsupported(format!("duplicate analysis option {key}")));
                }
                overrides.set(input.kind, &key, value.trim())?;
            }
            None => positional.push(argument.clone()),
        }
    }
    let dc = overrides.solver.resolve(config)?;
    let specific = match input.kind {
        AnalysisKind::Transient | AnalysisKind::PeriodicSteadyState => {
            SpecificOptions::Transient(overrides.tran.resolve(config)?)
        }
        AnalysisKind::DcSweep => SpecificOptions::Dc {
            point_iterations: overrides.point_iterations.or_else(|| config.dc().itl2),
        },
        AnalysisKind::HarmonicBalance => SpecificOptions::HarmonicBalance {
            oversampling: overrides.hb_oversampling.unwrap_or(3),
            iterations: overrides
                .hb_iterations
                .unwrap_or_else(|| super::harmonic_balance::BalanceLimits::default().iterations),
        },
        _ => SpecificOptions::Ordinary,
    };
    Ok(ResolvedOptions {
        positional,
        dc,
        specific,
    })
}

/// An analysis with its own validated input grammar and supported flags.
/// Circuit identities are intentionally still names until a circuit exists.
#[derive(Debug, Clone, PartialEq)]
pub(super) enum PlanKind {
    OperatingPoint,
    Dc {
        axes: crate::analysis::sweep::DcSweepPlan,
        point_iterations: Option<usize>,
    },
    Ac {
        frequencies: super::frequency::FrequencyPlan,
        retain_bias: bool,
    },
    Transient {
        settings: crate::analysis::TransientPlan,
        uic: bool,
    },
    PeriodicSteadyState {
        settings: super::pss::PssPlan,
        uic: bool,
    },
    HarmonicBalance(super::harmonic_balance::HarmonicBalancePlan),
    PoleZero {
        card: crate::analysis::pz::Card,
        retain_bias: bool,
    },
    TransferFunction(crate::analysis::tf::TransferRequest),
    SParameter {
        frequencies: super::frequency::FrequencyPlan,
        donoise: bool,
        retain_bias: bool,
    },
    Noise {
        card: crate::analysis::noise::NoiseCard,
        frequencies: super::frequency::FrequencyPlan,
        options: crate::analysis::NoiseOptions,
        retain_bias: bool,
    },
    Distortion {
        card: crate::analysis::disto::DistoCard,
        frequencies: super::frequency::FrequencyPlan,
        retain_bias: bool,
    },
    Sensitivity {
        card: crate::analysis::sens::SensCard,
        frequencies: Option<super::frequency::FrequencyPlan>,
    },
}

/// A resolved, immutable analysis execution plan.
///
/// All numeric request arguments and named analysis options have been
/// interpreted. Circuit-dependent node, source and device identities bind to
/// the circuit at execution; no driver interprets the original arguments.
#[derive(Debug, Clone, PartialEq)]
pub struct AnalysisPlan {
    analysis: PlanKind,
    dc: DcSettings,
    initial_conditions: Vec<crate::analysis::NodeCondition>,
    nodesets: Vec<crate::analysis::NodeCondition>,
}

fn validate_input(input: &AnalysisInput) -> SpiceResult<()> {
    let retains = matches!(
        input.kind,
        AnalysisKind::Ac
            | AnalysisKind::PoleZero
            | AnalysisKind::SParameter
            | AnalysisKind::Noise
            | AnalysisKind::Distortion
    );
    if input.uic
        && !matches!(
            input.kind,
            AnalysisKind::Transient | AnalysisKind::PeriodicSteadyState
        )
    {
        return Err(unsupported("uic is valid only for .tran or .pss"));
    }
    if input.noise.is_some() && input.kind != AnalysisKind::Noise {
        return Err(unsupported("noise options are valid only for .noise"));
    }
    if input.keep_op_info.is_some() && !retains {
        return Err(unsupported(
            "retaining the bias plot is not supported for this analysis",
        ));
    }
    for entry in input.initial_conditions.iter().chain(&input.nodesets) {
        if !entry.value.is_finite() {
            return Err(crate::primitives::SpiceError::parse(
                entry.location.clone(),
                format!("nonfinite node condition for V({})", entry.node),
            ));
        }
    }
    Ok(())
}

fn resolve_time_domain(
    kind: AnalysisKind,
    positional: &[String],
    specific: SpecificOptions,
    newton: &crate::analysis::newton::NewtonOptions,
    uic: bool,
) -> SpiceResult<PlanKind> {
    let SpecificOptions::Transient(options) = specific else {
        return Err(unsupported("missing transient options"));
    };
    if kind == AnalysisKind::Transient {
        Ok(PlanKind::Transient {
            settings: super::companion::TransientPlan::resolve(positional, options, newton)?,
            uic,
        })
    } else {
        Ok(PlanKind::PeriodicSteadyState {
            settings: super::pss::PssPlan::resolve(positional, options, newton)?,
            uic,
        })
    }
}

fn resolve_harmonic_balance(
    positional: &[String],
    specific: SpecificOptions,
) -> SpiceResult<PlanKind> {
    let SpecificOptions::HarmonicBalance {
        oversampling,
        iterations,
    } = specific
    else {
        return Err(unsupported("missing harmonic balance settings"));
    };
    Ok(PlanKind::HarmonicBalance(
        super::harmonic_balance::HarmonicBalancePlan::resolve(
            positional,
            oversampling,
            iterations,
        )?,
    ))
}

impl AnalysisPlan {
    pub(super) fn resolve(
        input: AnalysisInput,
        config: &RunConfig,
        deck_noise: crate::analysis::NoiseOptions,
        deck_keep_op_info: bool,
    ) -> SpiceResult<Self> {
        use crate::analysis::{ac, disto, noise, pz, sens, sparam, sweep, tf};
        let kind = input.kind;
        validate_input(&input)?;
        let retain_bias = input.keep_op_info.unwrap_or(deck_keep_op_info);
        let ResolvedOptions {
            positional,
            dc,
            specific,
        } = resolve_options(&input, config)?;
        let reltol = dc.newton.reltol;
        let analysis = match kind {
            AnalysisKind::OperatingPoint => {
                if !positional.is_empty() {
                    return Err(unsupported(".op arguments"));
                }
                PlanKind::OperatingPoint
            }
            AnalysisKind::DcSweep => {
                let SpecificOptions::Dc { point_iterations } = specific else {
                    return Err(unsupported("missing DC sweep options"));
                };
                PlanKind::Dc {
                    axes: sweep::parse_axes(&positional)?,
                    point_iterations,
                }
            }
            AnalysisKind::Ac => {
                if positional.len() != 4 {
                    return Err(unsupported(".ac lin|dec|oct points start stop"));
                }
                PlanKind::Ac {
                    frequencies: ac::frequency_grid(&positional, reltol, ac::GridKind::Ac)?,
                    retain_bias,
                }
            }
            AnalysisKind::Transient | AnalysisKind::PeriodicSteadyState => {
                resolve_time_domain(kind, &positional, specific, &dc.newton, input.uic)?
            }
            AnalysisKind::HarmonicBalance => resolve_harmonic_balance(&positional, specific)?,
            AnalysisKind::PoleZero => PlanKind::PoleZero {
                card: pz::parse(&positional)?,
                retain_bias,
            },
            AnalysisKind::TransferFunction => PlanKind::TransferFunction(tf::parse(
                &positional.iter().map(String::as_str).collect::<Vec<_>>(),
            )?),
            AnalysisKind::SParameter => {
                let (frequencies, donoise) = sparam::parse(&positional, reltol)?;
                PlanKind::SParameter {
                    frequencies,
                    donoise,
                    retain_bias,
                }
            }
            AnalysisKind::Noise => {
                let card = noise::parse(&positional)?;
                let frequencies = noise::frequency_grid(&card, reltol)?;
                PlanKind::Noise {
                    card,
                    frequencies,
                    options: input.noise.unwrap_or(deck_noise),
                    retain_bias,
                }
            }
            AnalysisKind::Distortion => {
                let card = disto::parse(&positional)?;
                let frequencies = disto::frequencies(&card, reltol)?;
                PlanKind::Distortion {
                    card,
                    frequencies,
                    retain_bias,
                }
            }
            AnalysisKind::Sensitivity => {
                let card = sens::parse(&positional)?;
                let frequencies = card.grid()?;
                PlanKind::Sensitivity { card, frequencies }
            }
            AnalysisKind::Fourier => {
                return Err(unsupported(format!(
                    ".four analysis has no driver in the port yet (see {})",
                    crate::analysis::C_REFERENCE_ANALYSIS,
                )));
            }
        };
        Ok(Self {
            analysis,
            dc,
            initial_conditions: input.initial_conditions,
            nodesets: input.nodesets,
        })
    }

    /// Bounded deterministic coordinate work requested by this plan.
    /// Adaptive transient stepping is separately governed by its own limits.
    pub(super) fn requested_grid_points(&self) -> usize {
        match &self.analysis {
            PlanKind::Dc { axes, .. } => axes.point_count(),
            PlanKind::PeriodicSteadyState { settings, .. } => settings.grid_points(),
            PlanKind::HarmonicBalance(settings) => settings.grid_points(),
            PlanKind::Ac { frequencies, .. }
            | PlanKind::SParameter { frequencies, .. }
            | PlanKind::Noise { frequencies, .. }
            | PlanKind::Distortion { frequencies, .. } => frequencies.len(),
            PlanKind::Sensitivity { frequencies, .. } => frequencies
                .as_ref()
                .map_or(0, super::frequency::FrequencyPlan::len),
            _ => 0,
        }
    }

    /// The kind encoded by the checked analysis payload.
    #[must_use = "the returned kind identifies the driver for the checked analysis payload"]
    pub const fn kind(&self) -> AnalysisKind {
        match self.analysis {
            PlanKind::OperatingPoint => AnalysisKind::OperatingPoint,
            PlanKind::Dc { .. } => AnalysisKind::DcSweep,
            PlanKind::Ac { .. } => AnalysisKind::Ac,
            PlanKind::Transient { .. } => AnalysisKind::Transient,
            PlanKind::PeriodicSteadyState { .. } => AnalysisKind::PeriodicSteadyState,
            PlanKind::HarmonicBalance(_) => AnalysisKind::HarmonicBalance,
            PlanKind::PoleZero { .. } => AnalysisKind::PoleZero,
            PlanKind::TransferFunction(_) => AnalysisKind::TransferFunction,
            PlanKind::SParameter { .. } => AnalysisKind::SParameter,
            PlanKind::Noise { .. } => AnalysisKind::Noise,
            PlanKind::Distortion { .. } => AnalysisKind::Distortion,
            PlanKind::Sensitivity { .. } => AnalysisKind::Sensitivity,
        }
    }

    /// The resolved Newton and continuation settings shared by bias solves.
    #[must_use = "the returned settings define the Newton and continuation policy for bias solves"]
    pub const fn dc_settings(&self) -> &DcSettings {
        &self.dc
    }

    /// Explicit or deck-supplied DC point iteration limit, if present.
    #[must_use = "the returned option determines whether a DC sweep point has an explicit iteration limit"]
    pub const fn point_iterations(&self) -> Option<usize> {
        match self.analysis {
            PlanKind::Dc {
                point_iterations, ..
            } => point_iterations,
            _ => None,
        }
    }

    /// The transient settings, if this is a transient plan.
    #[must_use = "the returned option distinguishes a checked transient payload from another analysis kind"]
    pub const fn transient(&self) -> Option<&crate::analysis::TransientPlan> {
        match &self.analysis {
            PlanKind::Transient { settings, .. } => Some(settings),
            _ => None,
        }
    }

    /// Whether this transient starts from the user initial conditions.
    #[must_use = "the result determines whether user initial conditions replace the ordinary startup bias solve"]
    pub const fn uic(&self) -> bool {
        matches!(
            self.analysis,
            PlanKind::Transient { uic: true, .. } | PlanKind::PeriodicSteadyState { uic: true, .. }
        )
    }

    /// Noise representation and model options, only for noise plans.
    #[must_use = "the returned option supplies noise representation and model settings only for noise analyses"]
    pub const fn noise_options(&self) -> Option<crate::analysis::NoiseOptions> {
        match self.analysis {
            PlanKind::Noise { options, .. } => Some(options),
            _ => None,
        }
    }

    /// Checked frequency points in the reference algorithm's order.
    #[must_use = "the returned iterator supplies the validated sweep frequencies in reference order"]
    pub fn frequencies(&self) -> Option<impl ExactSizeIterator<Item = Real> + Clone + '_> {
        match &self.analysis {
            PlanKind::Ac { frequencies, .. }
            | PlanKind::SParameter { frequencies, .. }
            | PlanKind::Noise { frequencies, .. }
            | PlanKind::Distortion { frequencies, .. } => Some(frequencies.values()),
            PlanKind::Sensitivity { frequencies, .. } => frequencies
                .as_ref()
                .map(super::frequency::FrequencyPlan::values),
            _ => None,
        }
    }

    /// Evaluated initial-condition assignments, in deck order.
    #[must_use = "the returned ordered assignments define the requested circuit initial conditions"]
    pub fn initial_conditions(&self) -> &[crate::analysis::NodeCondition] {
        &self.initial_conditions
    }

    /// Evaluated nodeset assignments, in deck order.
    #[must_use = "the returned ordered assignments define the requested bias-solve hints"]
    pub fn nodesets(&self) -> &[crate::analysis::NodeCondition] {
        &self.nodesets
    }

    /// Whether this plan retains its small-signal bias in multi-plot output.
    #[must_use = "the result determines whether the output sequence includes a small-signal bias plot"]
    pub const fn retains_bias(&self) -> bool {
        match self.analysis {
            PlanKind::Ac { retain_bias, .. }
            | PlanKind::PoleZero { retain_bias, .. }
            | PlanKind::SParameter { retain_bias, .. }
            | PlanKind::Noise { retain_bias, .. }
            | PlanKind::Distortion { retain_bias, .. } => retain_bias,
            _ => false,
        }
    }

    /// Exact number of plots emitted by this resolved multi-plot analysis.
    #[must_use = "the returned count determines how many plots the consumer must expect from this analysis"]
    pub const fn plot_count(&self) -> usize {
        let count = match &self.analysis {
            PlanKind::Noise { card, .. } => card.plot_count(),
            PlanKind::Distortion { card, .. } => card.plot_count(),
            PlanKind::PeriodicSteadyState { settings, .. } => settings.plot_count(),
            _ => 1,
        };
        count + if self.retains_bias() { 1 } else { 0 }
    }

    pub(super) const fn analysis(&self) -> &PlanKind {
        &self.analysis
    }

    /// Run the analysis and return its primary plot (without retaining bias).
    ///
    /// # Errors
    /// Circuit binding/elaboration or numerical errors.
    pub fn run(
        &self,
        circuit: &mut crate::devices::Circuit,
        context: &crate::analysis::AnalysisContext,
    ) -> SpiceResult<crate::analysis::Plot> {
        let mut collector = super::results::PlotCollector::default();
        self.execute(circuit, context, false, &mut collector)?;
        collector
            .finish(self.plot_count() - usize::from(self.retains_bias()))?
            .into_iter()
            .next()
            .ok_or_else(|| crate::primitives::SpiceError::Numerical {
                context: format!(".{} analysis", self.kind().as_str()),
                message: "the analysis returned no plot".into(),
            })
    }

    /// Run the analysis and return every plot in C output order.
    ///
    /// # Errors
    /// Circuit binding/elaboration or numerical errors.
    pub fn run_plots(
        &self,
        circuit: &mut crate::devices::Circuit,
        context: &crate::analysis::AnalysisContext,
    ) -> SpiceResult<Vec<crate::analysis::Plot>> {
        let mut collector = super::results::PlotCollector::default();
        self.run_into(circuit, context, &mut collector)?;
        collector.finish(self.plot_count())
    }

    /// Execute a sensitivity plan and return the ordinary plot plus a
    /// same-shaped diagnostic plot of estimated absolute comparison floors.
    /// The diagnostic uses each parameter's actual finite-difference denominator;
    /// it is not emitted to rawfiles and is not a certified derivative error bound.
    ///
    /// # Errors
    /// A non-sensitivity plan, or ordinary circuit/analysis failures.
    pub fn run_sensitivity_with_resolution(
        &self,
        circuit: &mut crate::devices::Circuit,
        context: &crate::analysis::AnalysisContext,
    ) -> SpiceResult<(crate::analysis::Plot, crate::analysis::Plot)> {
        if circuit.prepare_native_mixed()?.is_some() {
            return Err(unsupported("native mixed-signal sensitivity"));
        }
        let PlanKind::Sensitivity { card, frequencies } = &self.analysis else {
            return Err(unsupported("sensitivity resolution requires a .sens plan"));
        };
        circuit.validate_prepared_analysis(self.kind(), self.uic())?;
        let context = &crate::analysis::AnalysisContext {
            analysis_kind: circuit.requires_analysis_context().then(|| self.kind()),
            bias_analysis: circuit.requires_analysis_context().then(|| self.kind()),
            ..*context
        };
        let (plot, resolution) =
            super::sens::run(circuit, self, context, card, frequencies.as_ref(), true)?;
        let resolution = resolution.ok_or_else(|| unsupported("missing sensitivity resolution"))?;
        Ok((plot, resolution))
    }

    fn execute(
        &self,
        circuit: &mut crate::devices::Circuit,
        context: &crate::analysis::AnalysisContext,
        retain_bias: bool,
        sink: &mut dyn crate::analysis::PointSink,
    ) -> SpiceResult<()> {
        use crate::analysis::{ac, disto, linear, noise, pz, sens, sparam, sweep, tf};
        circuit.validate_prepared_analysis(self.kind(), self.uic())?;
        let context = &crate::analysis::AnalysisContext {
            analysis_kind: circuit.requires_analysis_context().then(|| self.kind()),
            bias_analysis: (!self.uic() && circuit.requires_analysis_context())
                .then(|| self.kind()),
            ..*context
        };
        if let Some(mut mixed) = circuit.prepare_native_mixed()? {
            let result = match &self.analysis {
                PlanKind::OperatingPoint => linear::mixed_op(circuit, self, context, &mut mixed)
                    .and_then(|plot| sink.complete(plot)),
                PlanKind::Transient { .. } => super::companion::mixed_signal_transient_into(
                    circuit, self, context, &mut mixed, sink,
                )
                .map(drop),
                _ => Err(unsupported(
                    "native mixed-signal analysis requires OP or transient",
                )),
            };
            circuit.record_digital_stats(result.as_ref().ok().map(|()| mixed.engine().stats()));
            return result;
        }
        // Large sweeps emit rows directly. Drivers whose numerical method
        // still requires a materialized output transfer ownership instead.
        let plots = match &self.analysis {
            PlanKind::OperatingPoint => vec![linear::op(circuit, self, context)?],
            PlanKind::Dc {
                axes,
                point_iterations,
            } => return sweep::run(circuit, self, context, axes, *point_iterations, sink),
            PlanKind::Ac { frequencies, .. } => {
                return ac::run(circuit, self, context, frequencies, retain_bias, sink);
            }
            PlanKind::Transient { .. } => {
                return crate::analysis::companion_transient_into(circuit, self, context, sink)
                    .map(|_| ());
            }
            PlanKind::PeriodicSteadyState { settings, .. } => {
                return super::pss::run(circuit, self, context, settings, None, sink).map(drop);
            }
            PlanKind::HarmonicBalance(settings) => {
                return settings.run(circuit, context, &self.dc.newton, sink);
            }
            PlanKind::PoleZero { card, .. } => pz::run(circuit, self, context, card, retain_bias)?,
            PlanKind::TransferFunction(card) => vec![tf::run(circuit, self, context, card)?],
            PlanKind::SParameter {
                frequencies,
                donoise,
                ..
            } => sparam::run(circuit, self, context, frequencies, *donoise, retain_bias)?,
            PlanKind::Noise {
                card, frequencies, ..
            } => {
                return noise::run(circuit, self, context, card, frequencies, retain_bias, sink);
            }
            PlanKind::Distortion {
                card, frequencies, ..
            } => disto::run(circuit, self, context, card, frequencies, retain_bias)?,
            PlanKind::Sensitivity { card, frequencies } => {
                vec![sens::run(circuit, self, context, card, frequencies.as_ref(), false)?.0]
            }
        };
        for plot in plots {
            sink.complete(plot)?;
        }
        Ok(())
    }

    /// Solve a periodic orbit and emit the configured plots, optionally
    /// starting from a previously converged orbit on the same topology.
    ///
    /// # Errors
    /// Non-PSS request, inconsistent warm start, or numerical/physical failure.
    pub fn run_periodic(
        &self,
        circuit: &mut crate::devices::Circuit,
        context: &crate::analysis::AnalysisContext,
        warm: Option<&super::PeriodicOrbit>,
        sink: &mut dyn crate::analysis::PointSink,
    ) -> SpiceResult<super::PeriodicOrbit> {
        if circuit.prepare_native_mixed()?.is_some() {
            return Err(unsupported("native mixed-signal periodic analysis"));
        }
        let PlanKind::PeriodicSteadyState { settings, .. } = &self.analysis else {
            return Err(unsupported("periodic orbit requires a .pss plan"));
        };
        circuit.validate_prepared_analysis(self.kind(), self.uic())?;
        let context = &crate::analysis::AnalysisContext {
            analysis_kind: circuit.requires_analysis_context().then(|| self.kind()),
            bias_analysis: (!self.uic() && circuit.requires_analysis_context())
                .then(|| self.kind()),
            ..*context
        };
        super::pss::run(circuit, self, context, settings, warm, sink)
    }

    /// Execute all planned outputs in order through a single sink lifecycle.
    /// Transient, DC, AC and noise spectrum rows are delivered as produced;
    /// other drivers transfer their completed plots. The sink controls retention.
    ///
    /// # Errors
    /// Circuit binding/elaboration, numerical or sink errors.
    pub fn run_into(
        &self,
        circuit: &mut crate::devices::Circuit,
        context: &crate::analysis::AnalysisContext,
        sink: &mut dyn crate::analysis::PointSink,
    ) -> SpiceResult<()> {
        self.execute(circuit, context, self.retains_bias(), sink)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::bias::{ContinuationSchedule, Stepping};
    use crate::netlist::{Parser, source::parse_deck_text};
    use std::path::Path;

    fn config(text: &str) -> RunConfig {
        let deck = parse_deck_text(
            Path::new("plan.cir"),
            &format!("plan\nr1 a 0 1k\n{text}\n.end\n"),
        );
        let netlist = Parser::new().parse_deck(&deck).unwrap();
        RunConfig::from_netlist(&netlist).unwrap()
    }

    #[test]
    fn typed_precedence_preserves_stage_and_adaptation_limits() {
        let config = config(
            ".options itl1=200 itl2=9 gminsteps=5 srcsteps=0\n.options itl1=300 srcsteps=8 reltol=1e-6",
        );
        let input = AnalysisInput::with_arguments(
            AnalysisKind::DcSweep,
            ["v1", "0", "1", "0.1", "MAXITER=7", "srcsteps=3"],
        );
        let options = resolve_options(&input, &config).unwrap();
        assert_eq!(options.positional, ["v1", "0", "1", "0.1"]);
        assert_eq!(options.dc.newton.max_iterations, 7);
        assert_eq!(options.dc.newton.reltol, 1e-6);
        assert_eq!(
            options
                .dc
                .continuation
                .stage_max_iterations
                .map(IterationLimit::get),
            Some(100)
        );
        let ContinuationSchedule::Ngspice(schedule) = options.dc.continuation.schedule else {
            panic!("wrong schedule")
        };
        assert_eq!(schedule.adapt_iterations(), 9);
        assert_eq!(schedule.source, Stepping::from_count(3).unwrap());
        assert_eq!(schedule.gmin, Stepping::from_count(5).unwrap());
        assert_eq!(
            options.specific,
            SpecificOptions::Dc {
                point_iterations: Some(100)
            }
        );
    }

    #[test]
    fn transient_overrides_are_typed_and_do_not_leak() {
        let config = config(".options method=gear maxord=4 itl4=200 chgtol=1p xmu=0.25");
        let input = AnalysisInput::with_arguments(
            AnalysisKind::Transient,
            ["1u", "1m", "maxord=3", "tranmaxiter=7"],
        );
        let options = resolve_options(&input, &config).unwrap();
        let SpecificOptions::Transient(tran) = options.specific else {
            panic!("wrong kind")
        };
        assert_eq!(tran.method, IntegrationMethod::Gear { order: 3 });
        assert_eq!(tran.tran_max_iter, 7);
        assert_eq!(tran.chgtol, 1e-12);
        assert_eq!(tran.xmu, 0.25);
        for key in ["method=gear", "tranmaxiter=20", "trcvmaxiter=1"] {
            let input = AnalysisInput::with_arguments(AnalysisKind::OperatingPoint, [key]);
            assert!(resolve_options(&input, &config).is_err(), "{key}");
        }
    }

    #[test]
    fn duplicate_invalid_and_cross_analysis_explicit_options_fail() {
        for args in [
            vec!["maxiter=1", "MAXITER=2"],
            vec!["srcsteps=-1"],
            vec!["continuation=ladder", "adaptiter=10"],
            vec!["rtol=0"],
            vec!["backend=diffsol"],
            vec!["trcvmaxiter=5"],
        ] {
            let input = AnalysisInput::with_arguments(AnalysisKind::Transient, args);
            assert!(resolve_options(&input, &RunConfig::default()).is_err());
        }
    }
}

#[cfg(test)]
mod boundary_tests {
    use super::*;
    use crate::analysis::{AnalysisContext, DRIVERS, NoiseOptions, has_driver};
    use crate::netlist::{Parser, source::parse_deck_text};
    use crate::primitives::SpiceError;
    use std::path::Path;

    #[test]
    fn every_supported_driver_has_a_resolvable_plan() {
        let inputs = [
            (AnalysisKind::OperatingPoint, ""),
            (AnalysisKind::DcSweep, "v1 0 1 0.1"),
            (AnalysisKind::Ac, "lin 3 1 3"),
            (AnalysisKind::Transient, "1u 1m"),
            (AnalysisKind::PoleZero, "in 0 out 0 vol pz"),
            (AnalysisKind::TransferFunction, "v ( out ) v1"),
            (AnalysisKind::SParameter, "lin 3 1 3"),
            (AnalysisKind::Noise, "v(out) v1 lin 3 1 3"),
            (AnalysisKind::Distortion, "dec 2 1k 10k"),
            (AnalysisKind::Sensitivity, "v ( out )"),
            (AnalysisKind::PeriodicSteadyState, "1k 1m out"),
            (AnalysisKind::HarmonicBalance, "1k 3"),
        ];
        assert_eq!(inputs.map(|(kind, _)| kind), DRIVERS);
        for (kind, arguments) in inputs {
            let plan = AnalysisInput::with_arguments(kind, arguments.split_whitespace())
                .resolve()
                .unwrap();
            assert!(has_driver(kind));
            assert_eq!(plan.kind(), kind);
        }
        let error = AnalysisInput::new(AnalysisKind::Fourier)
            .resolve()
            .unwrap_err();
        assert!(!error.is_not_yet_ported());
        assert!(error.to_string().contains("four"));
        assert!(!has_driver(AnalysisKind::Fourier));
    }

    #[test]
    fn invalid_dc_geometry_fails_without_a_circuit() {
        for arguments in [
            "v1 0 1 0",
            "v1 0 1 -1",
            "v1 -1e308 1e308 1",
            "v1 0 1e6 1",
            "v1 0 999 1 v2 0 100 1",
            "temp -300 0 100",
            "temp 0 1e-12 1e-15",
        ] {
            let error =
                AnalysisInput::with_arguments(AnalysisKind::DcSweep, arguments.split_whitespace())
                    .resolve()
                    .unwrap_err();
            assert!(
                matches!(error, SpiceError::Unsupported { .. }),
                "{arguments}: {error}"
            );
        }
    }

    #[test]
    fn irrelevant_flags_are_rejected_and_explicit_values_override_deck() {
        let deck = parse_deck_text(
            Path::new("flags.cir"),
            "flags\nv1 a 0 dc 0 ac 1\nr1 a 0 1k\n.control\nset sqrnoise\nset ngbehavior=s3\n.endc\n.options keepopinfo\n.noise v(a) v1 lin 3 1 3\n.op\n.end\n",
        );
        let netlist = Parser::new().parse_deck(&deck).unwrap();
        let config = RunConfig::from_netlist(&netlist).unwrap();
        let plan = config.plan_for(&netlist.analyses[0]).unwrap();
        assert_eq!(
            plan.noise_options(),
            Some(NoiseOptions {
                squared: true,
                spice3: true
            })
        );
        assert!(plan.retains_bias());
        assert_eq!(plan.plot_count(), 3);
        let mut input = AnalysisInput::from(&netlist.analyses[0]);
        input.noise = Some(NoiseOptions::default());
        input.keep_op_info = Some(false);
        let plan = config.plan(input).unwrap();
        assert_eq!(plan.noise_options(), Some(NoiseOptions::default()));
        assert!(!plan.retains_bias());
        assert_eq!(plan.plot_count(), 2);
        let mut op = AnalysisInput::new(AnalysisKind::OperatingPoint);
        assert_eq!(config.plan(op.clone()).unwrap().noise_options(), None);
        op.noise = Some(NoiseOptions::default());
        assert!(config.plan(op.clone()).is_err());
        op.noise = None;
        op.keep_op_info = Some(false);
        assert!(config.plan(op.clone()).is_err());
        op.keep_op_info = None;
        op.uic = true;
        assert!(config.plan(op).is_err());
    }

    #[test]
    fn programmatic_node_hints_reject_nonfinite_values_at_resolution() {
        let location =
            crate::primitives::SourceLoc::new(std::path::PathBuf::from("input.cir"), 7, 4);
        let mut input = AnalysisInput::new(AnalysisKind::OperatingPoint);
        input.nodesets.push(crate::analysis::NodeCondition {
            node: "a".into(),
            value: Real::NAN,
            location: location.clone(),
        });
        assert!(matches!(
            input.resolve(),
            Err(SpiceError::Parse { location: rejected, .. }) if rejected == location
        ));
    }

    #[test]
    fn invalid_deck_plan_diagnostics_retain_card_location() {
        let deck = parse_deck_text(
            Path::new("invalid.cir"),
            "invalid\nr1 a 0 1k\n.op\n.dc v1 0 1000000 1\n.end\n",
        );
        let netlist = Parser::new().parse_deck(&deck).unwrap();
        let config = RunConfig::from_netlist(&netlist).unwrap();
        let error = config.plans_for(&netlist.analyses).unwrap_err();
        assert!(
            matches!(
                error,
                SpiceError::Unsupported { location: Some(ref location), .. }
                if location == &netlist.analyses[1].location
            ),
            "{error}"
        );
        let plan = AnalysisInput::new(AnalysisKind::OperatingPoint)
            .resolve()
            .unwrap();
        let error = plan
            .run(
                &mut crate::devices::Circuit::new(),
                &AnalysisContext::default(),
            )
            .unwrap_err();
        assert!(error.to_string().contains("nonempty square"), "{error}");
    }
}
