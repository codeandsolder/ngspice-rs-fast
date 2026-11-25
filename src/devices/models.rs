//! Top-level model resolution, backend selectors and initial diode schemas.
//!
//! C: `inpmkmod.c::INPmakeMod` keeps the first declaration; `inpdomod.c` selects
//! a backend using `inpfindl.c::INPfindLev`'s first level. `inpgmod.c` applies
//! ordered setters. Resolution/validation are not proof of a simulation backend.
//! No scoped expansion, global state or nonlinear arithmetic lives here. MOS
//! binning (`inpgmod.c::INPgetModBin`) is a fallback of [`ModelResolver::resolve`]
//! whose selection rule lives in [`crate::devices::binning`].
//!
//! ```
//! use std::path::Path;
//! use ngspice_rs::devices::{ModelContext, ModelResolver};
//! use ngspice_rs::netlist::{Parser, source::parse_deck_text};
//! let deck = parse_deck_text(Path::new("diode.cir"),
//!     "diode inputs\nd1 a 0 mdl temp=40\n.model mdl d(is=2e-14 tnom=30)\n.end\n");
//! let netlist = Parser::new().parse_deck(&deck)?;
//! let resolver = ModelResolver::new(&netlist.models)?;
//! let model = resolver.resolve(&netlist.devices[0])?.expect("diode reference");
//! let context = ModelContext::default();
//! assert_eq!(model.diode_parameters(&context)?.saturation_current, 2e-14);
//! assert_eq!(model.diode_instance_parameters(&netlist.devices[0], &context)?
//!     .temperature_kelvin, 313.15);
//! // Validated inputs do not imply an available diode simulation factory.
//! # Ok::<(), ngspice_rs::primitives::SpiceError>(())
//! ```

use std::any::{Any, TypeId};
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use crate::devices::binning::{BinCandidate, BinOptions, BinOutcome, DeclarationIndex, select_bin};

use crate::netlist::ast::{DeviceInstance, ModelCard};
use crate::primitives::{Real, SourceLoc, SpiceError, SpiceResult};

use crate::devices::schema::{
    ScalarDomain, ScalarParameter, ScalarSchema, ScalarUnit, ScalarValues, finite_literal,
    temperature_kelvin,
};
use crate::devices::sweep::{
    InstanceOverride, MAX_INSTANCE_OVERRIDES, MAX_RESISTOR_OVERRIDES, ResistorOverride,
};

/// Parsed model family, including polarity without rewriting the raw AST.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelFamily {
    /// Scalar resistor (`r` or `res`).
    Resistor,
    /// Scalar capacitor.
    Capacitor,
    /// Scalar inductor.
    Inductor,
    /// Junction diode.
    Diode,
    /// Explicit source-selected SIMetrix level-1 diode.
    SimetrixDiodeLevel1,
    /// NPN classic BJT.
    Npn,
    /// PNP classic BJT.
    Pnp,
    /// NMOS (classic, BSIM3 and BSIM4 backends).
    Nmos,
    /// PMOS (classic, BSIM3 and BSIM4 backends).
    Pmos,
    /// Voltage-controlled switch (`sw`, `sw/sw.c`).
    Switch,
    /// Current-controlled switch (`csw`, `csw/csw.c`).
    CurrentSwitch,
    /// N-channel JFET (`njf`; level 1 `jfet/`, level 2 `jfet2/`).
    Njf,
    /// P-channel JFET (`pjf`; level 1 `jfet/`, level 2 `jfet2/`).
    Pjf,
    /// Uniform distributed RC line (`urc`, `urc/urc.c`), expanded into
    /// lumped R/C/D sections by [`crate::devices::urc`].
    Urc,
    /// Lossy transmission line (`ltra`, `ltra/ltra.c`), simulated by
    /// [`crate::devices::ltra`].
    Ltra,
    /// Single lossy transmission line (TXL, `inp2y.c`).
    Txl,
    /// Distributed multiconductor cable (`P`, C `cpl`).
    Cpl,
    /// N-channel MESFET/HFET family (`nmf`; level 1 `mes/`, levels 2-4
    /// `mesa/`, 5 `hfet1/`, 6 `hfet2/`).
    Nmf,
    /// P-channel MESFET/HFET family (`pmf`; levels as for [`Self::Nmf`]).
    Pmf,
    /// XSPICE rational transfer function (`s_xfer`).
    RationalTransfer,
}

impl ModelFamily {
    /// Family associated with an ordinary parsed model type.
    #[must_use = "the family determines the model interpretation and device dispatch"]
    pub fn from_base(base: &str) -> Option<Self> {
        match base.to_ascii_lowercase().as_str() {
            "r" | "res" => Some(Self::Resistor),
            "c" => Some(Self::Capacitor),
            "l" => Some(Self::Inductor),
            "d" => Some(Self::Diode),
            model_kernel_diode::MODEL_FAMILY => Some(Self::SimetrixDiodeLevel1),
            "npn" => Some(Self::Npn),
            "pnp" => Some(Self::Pnp),
            "nmos" => Some(Self::Nmos),
            "pmos" => Some(Self::Pmos),
            "sw" => Some(Self::Switch),
            "csw" => Some(Self::CurrentSwitch),
            "njf" => Some(Self::Njf),
            "pjf" => Some(Self::Pjf),
            "urc" => Some(Self::Urc),
            "ltra" => Some(Self::Ltra),
            "txl" => Some(Self::Txl),
            "cpl" | "cpllines" => Some(Self::Cpl),
            "s_xfer" => Some(Self::RationalTransfer),
            // `inpdomod.c` accepts the HFET spellings as the same families;
            // `level` then selects the device (5 is `hfet1/`, 6 `hfet2/`).
            "nmf" | "nhfet" => Some(Self::Nmf),
            "pmf" | "phfet" => Some(Self::Pmf),
            _ => None,
        }
    }

    /// Compatible instance designator. Node and model names have separate
    /// namespaces; model lookup never applies the `gnd` alias.
    #[must_use = "the designator selects the corresponding device family"]
    pub const fn designator(self) -> char {
        match self {
            Self::Resistor => 'r',
            Self::Capacitor => 'c',
            Self::Inductor => 'l',
            Self::Diode | Self::SimetrixDiodeLevel1 => 'd',
            Self::Npn | Self::Pnp => 'q',
            Self::Nmos | Self::Pmos => 'm',
            Self::Switch => 's',
            Self::CurrentSwitch => 'w',
            Self::Njf | Self::Pjf => 'j',
            Self::Urc => 'u',
            Self::Ltra => 'o',
            Self::Txl => 'y',
            Self::Cpl => 'p',
            Self::RationalTransfer => 'a',
            Self::Nmf | Self::Pmf => 'z',
        }
    }
}

/// Concrete device implementation chosen by the validated model selector.
/// Different raw selectors may name one implementation (C's `INPdomodel`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ModelBackend {
    /// Scalar resistor.
    Resistor,
    /// Scalar capacitor.
    Capacitor,
    /// Scalar inductor.
    Inductor,
    /// Junction diode.
    Diode,
    /// Explicit source-selected SIMetrix level-1 diode.
    SimetrixDiodeLevel1,
    /// Gummel-Poon.
    Bjt,
    /// MOS1.
    Mos1,
    /// MOS2.
    Mos2,
    /// MOS3.
    Mos3,
    /// MOS6.
    Mos6,
    /// MOS9.
    Mos9,
    /// BSIM3v3.3.
    Bsim3,
    /// BSIM4.8.
    Bsim4,
    /// JFET.
    Jfet,
    /// JFET2.
    Jfet2,
    /// MESFET.
    Mesfet,
    /// MESA.
    Mesa,
    /// HFET1.
    Hfet1,
    /// HFET2.
    Hfet2,
    /// Voltage-controlled switch.
    Switch,
    /// Current-controlled switch.
    CurrentSwitch,
    /// Uniform RC expansion.
    Urc,
    /// Lossy line.
    Ltra,
    /// TXL lossy line.
    Txl,
    /// Multiconductor coupled transmission line.
    Cpl,
    /// Native rational transfer model.
    RationalTransfer,
}

/// Authoritative backend catalog entry. Selector aliases are the C levels
/// supported by this implementation, not promises about every model parameter.
pub(crate) struct BackendSpec {
    pub(crate) backend: ModelBackend,
    pub(crate) name: &'static str,
    pub(crate) selectors: &'static [u8],
    family: Option<ModelFamily>,
}

impl BackendSpec {
    const fn new(backend: ModelBackend, name: &'static str, selectors: &'static [u8]) -> Self {
        Self {
            backend,
            name,
            selectors,
            family: None,
        }
    }

    const fn for_family(mut self, family: ModelFamily) -> Self {
        self.family = Some(family);
        self
    }
}

impl ModelBackend {
    /// Backend options for the model's compatible instance designator.
    /// Families differing only in polarity share the same selector catalog.
    pub(crate) const fn catalog(designator: char) -> &'static [BackendSpec] {
        match designator.to_ascii_lowercase() {
            'r' => const { &[BackendSpec::new(Self::Resistor, "Scalar resistor", &[0, 1])] },
            'c' => const { &[BackendSpec::new(Self::Capacitor, "Scalar capacitor", &[1])] },
            'l' => const { &[BackendSpec::new(Self::Inductor, "Scalar inductor", &[1])] },
            'd' => {
                const {
                    &[
                        BackendSpec::new(Self::Diode, "Junction diode", &[1]),
                        BackendSpec::new(
                            Self::SimetrixDiodeLevel1,
                            "SIMetrix diode (explicit simetrix_diode_level1 family)",
                            &[1],
                        )
                        .for_family(ModelFamily::SimetrixDiodeLevel1),
                    ]
                }
            }
            'q' => const { &[BackendSpec::new(Self::Bjt, "Gummel-Poon", &[0, 1, 2])] },
            'm' => {
                const {
                    &[
                        BackendSpec::new(Self::Mos1, "MOS1", &[0, 1]),
                        BackendSpec::new(Self::Mos2, "MOS2", &[2]),
                        BackendSpec::new(Self::Mos3, "MOS3", &[3]),
                        BackendSpec::new(Self::Mos6, "MOS6", &[6]),
                        BackendSpec::new(Self::Mos9, "MOS9", &[9]),
                        BackendSpec::new(Self::Bsim3, "BSIM3v3.3", &[8, 49]),
                        BackendSpec::new(Self::Bsim4, "BSIM4.8", &[14, 54]),
                    ]
                }
            }
            'j' => {
                const {
                    &[
                        BackendSpec::new(Self::Jfet, "JFET", &[0, 1]),
                        BackendSpec::new(Self::Jfet2, "JFET2", &[2]),
                    ]
                }
            }
            'z' => {
                const {
                    &[
                        BackendSpec::new(Self::Mesfet, "MESFET", &[0, 1]),
                        BackendSpec::new(Self::Mesa, "MESA", &[2, 3, 4]),
                        BackendSpec::new(Self::Hfet1, "HFET1", &[5]),
                        BackendSpec::new(Self::Hfet2, "HFET2", &[6]),
                    ]
                }
            }
            's' => {
                const {
                    &[BackendSpec::new(
                        Self::Switch,
                        "Voltage-controlled switch",
                        &[1],
                    )]
                }
            }
            'w' => {
                const {
                    &[BackendSpec::new(
                        Self::CurrentSwitch,
                        "Current-controlled switch",
                        &[1],
                    )]
                }
            }
            'u' => const { &[BackendSpec::new(Self::Urc, "Uniform RC expansion", &[1])] },
            'o' => const { &[BackendSpec::new(Self::Ltra, "Lossy line", &[1])] },
            'y' => const { &[BackendSpec::new(Self::Txl, "TXL lossy line", &[1])] },
            'p' => const { &[BackendSpec::new(Self::Cpl, "Coupled line", &[1])] },
            'a' => {
                const {
                    &[BackendSpec::new(
                        Self::RationalTransfer,
                        "Rational transfer",
                        &[1],
                    )]
                }
            }
            _ => &[],
        }
    }

    fn select(family: ModelFamily, selector: u8) -> Option<Self> {
        let catalog = Self::catalog(family.designator());
        // An explicitly selected family owns its selectors: it must never
        // fall back to a native backend when its requested level is absent.
        let explicit = catalog.iter().any(|spec| spec.family == Some(family));
        catalog
            .iter()
            .find(|spec| {
                spec.family == explicit.then_some(family) && spec.selectors.contains(&selector)
            })
            .map(|spec| spec.backend)
    }

    pub(super) const fn requires_model(designator: char) -> bool {
        !matches!(designator.to_ascii_lowercase(), 'r' | 'c' | 'l' | 'a')
            && !Self::catalog(designator).is_empty()
    }
}

/// Keep model-backend selection separate from model parameter setters.
///
/// Backend selection is distinct from ordered model setters. Only diode has a
/// `level` integer setter in the initial family tables; BJT/MOS/passive `level`
/// assignments are selector-only (inpgmod.c consumes them without setting data).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LevelSelection {
    /// First raw scalar level, or `None` if omitted. No AST mutation occurs.
    pub first_raw: Option<Real>,
    /// First rounded level for selector-based families (including aliases);
    /// fixed 1 for families whose C backend does not scan the level.
    pub selector: u8,
    /// Concrete implementation for the validated selector.
    pub backend: ModelBackend,
    /// Diode's last integer setter (`floor(raw + 0.5)`), defaulting to 1.
    /// `None` for families with no level setter.
    pub applied: Option<u8>,
}

/// Identity of the immutable preparation behind one resolved declaration.
/// A scalar schema is a static object with a stable address. Nominal keys
/// retain the exact binary64 value, including signed zero.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(super) enum PreparationKey {
    Inputs,
    Scalar(usize),
    Nominal(u64),
    PreparedTopology {
        series: bool,
        absolute_temperature: bool,
    },
}

type PreparationCell<T> = OnceLock<SpiceResult<Arc<T>>>;

/// Only immutable, declaration-scoped artifacts belong here. The value type
/// distinguishes the backend and preparation stage; the key distinguishes
/// schema/context variants of that stage. Never cache instance-local errors,
/// resettable models, geometry, or solver state.
#[derive(Default)]
pub(super) struct PreparationCache {
    entries: Mutex<HashMap<(TypeId, PreparationKey), Arc<dyn Any + Send + Sync>>>,
}

impl std::fmt::Debug for PreparationCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PreparationCache").finish_non_exhaustive()
    }
}

impl PreparationCache {
    pub(super) fn get_or_prepare<T: Send + Sync + 'static>(
        &self,
        key: PreparationKey,
        prepare: impl FnOnce() -> SpiceResult<T>,
    ) -> SpiceResult<Arc<T>> {
        let cell = {
            let mut entries = self
                .entries
                .lock()
                .map_err(|_| SpiceError::circuit("model preparation cache poisoned"))?;
            Arc::clone(
                entries
                    .entry((TypeId::of::<T>(), key))
                    .or_insert_with(|| Arc::new(PreparationCell::<T>::new())),
            )
        };
        let cell = cell
            .downcast::<PreparationCell<T>>()
            .map_err(|_| SpiceError::circuit("inconsistent model preparation cache type"))?;
        // Initialization may itself use another preparation stage. Drop the
        // map lock before running the closure; a single cell publishes once.
        cell.get_or_init(|| prepare().map(Arc::new)).clone()
    }
}

/// Declaration-local metadata. Validate family first, and defer level
/// validation until an instance has passed the family/designator check.
#[derive(Debug)]
struct Declaration<'a> {
    bin: BinCandidate<'a>,
    family: OnceLock<SpiceResult<ModelFamily>>,
    levels: OnceLock<SpiceResult<LevelSelection>>,
    preparation: OnceLock<Arc<PreparationCache>>,
}

impl<'a> Declaration<'a> {
    fn new(card: &'a ModelCard) -> Self {
        Self {
            bin: BinCandidate::new(card),
            family: OnceLock::new(),
            levels: OnceLock::new(),
            preparation: OnceLock::new(),
        }
    }

    fn family(&self) -> SpiceResult<ModelFamily> {
        self.family.get_or_init(|| family(self.bin.card)).clone()
    }

    fn levels(&self, family: ModelFamily) -> SpiceResult<LevelSelection> {
        self.levels
            .get_or_init(|| levels(self.bin.card, family))
            .clone()
    }
}

/// Immutable lookup over one deck's top-level declarations.
///
/// Duplicate names retain the first card as `INPmakeMod` does; all cards stay visible in the AST.
/// Selector and bin metadata are scoped to this borrowed deck, and validated
/// on demand. Immutable preparation artifacts are shared by exact declaration;
/// geometry-dependent results and mutable backend state remain instance-local.
#[derive(Debug)]
pub struct ModelResolver<'a> {
    declarations: DeclarationIndex<Declaration<'a>>,
    bin_options: BinOptions,
}

impl<'a> ModelResolver<'a> {
    /// Index model names independently of node symbols or backend availability.
    ///
    /// # Errors
    /// Empty model names in a programmatically constructed AST.
    pub fn new(cards: &'a [ModelCard]) -> SpiceResult<Self> {
        let mut declarations = DeclarationIndex::new();
        for card in cards {
            if card.name.is_empty() {
                return Err(SpiceError::parse(card.location.clone(), "empty model name"));
            }
            declarations.insert_first(&card.name, || Declaration::new(card));
        }
        Ok(Self {
            declarations,
            bin_options: BinOptions::default(),
        })
    }

    /// Replace the front-end settings binning reads (`scale`, `wnflag`).
    #[must_use = "discarding the returned index leaves the previous binning options in effect"]
    pub const fn with_bin_options(mut self, options: BinOptions) -> Self {
        self.bin_options = options;
        self
    }

    /// Binning candidates for a reference: first declarations named
    /// `<name>.<digits>`, in declaration order (no family filtering).
    pub fn bin_candidates<'b>(
        &'b self,
        name: &str,
    ) -> impl DoubleEndedIterator<Item = &'a ModelCard> + use<'a, 'b> {
        self.declarations
            .bins(name)
            .map(|declaration| declaration.bin.card)
    }

    /// Declarations a reference can bind: the exact card when one exists,
    /// otherwise every binning candidate. C's `inp_rem_unused_models`
    /// (`mark_all_binned`) treats all of them as used.
    pub fn declarations_for<'b>(
        &'b self,
        name: &str,
    ) -> impl Iterator<Item = &'a ModelCard> + use<'a, 'b> {
        let exact = self.model(name);
        let bins = exact.is_none().then(|| self.bin_candidates(name));
        exact.into_iter().chain(bins.into_iter().flatten())
    }

    /// The card `INPgetModBin` binds to an `M` instance whose model name has no
    /// exact declaration. `Ok(None)` when the instance is not a MOSFET, has an
    /// exact declaration, or has no `<name>.<digits>` candidate at all.
    ///
    /// # Errors
    /// C's "could not find a valid modelname" cases, as parse errors at the
    /// instance naming every candidate: no binnable family/level among the
    /// candidates, `l`/`w` not written on the instance, or no bin containing
    /// the geometry. Invalid geometry or bound values are also errors.
    pub fn select_bin(&self, instance: &DeviceInstance) -> SpiceResult<Option<&'a ModelCard>> {
        let Some(name) = &instance.model else {
            return Ok(None);
        };
        if !instance.designator.eq_ignore_ascii_case(&'m') || self.model(name).is_some() {
            return Ok(None);
        }
        self.select_declaration(instance, name)
            .map(|selected| selected.map(|declaration| declaration.bin.card))
    }

    /// Select one indexed declaration after exact-name resolution failed.
    fn select_declaration(
        &self,
        instance: &DeviceInstance,
        name: &str,
    ) -> SpiceResult<Option<&Declaration<'a>>> {
        if !instance.designator.eq_ignore_ascii_case(&'m') {
            return Ok(None);
        }
        let candidates = self
            .declarations
            .bins(name)
            .map(|declaration| (declaration, &declaration.bin));
        if candidates.len() == 0 {
            return Ok(None);
        }
        let reason = match select_bin(candidates, instance, &self.bin_options)? {
            BinOutcome::Selected(card) => return Ok(Some(card)),
            BinOutcome::NoBinnableCandidate => "ngspice bins only BSIM3 (level 8/49), BSIM4 \
                 (14/54), HiSIM2 (68) and HiSIM-HV (73) MOS models"
                .to_owned(),
            BinOutcome::MissingGeometry => {
                "binning needs both l and w written on the instance".to_owned()
            }
            BinOutcome::NoMatch(geometry) => format!(
                "no bin with all of lmin/lmax/wmin/wmax contains L={:e} W={:e}",
                geometry.length, geometry.width
            ),
        };
        let listed = self
            .bin_candidates(name)
            .map(|card| format!("'{}' ({})", card.name, card.location))
            .collect::<Vec<_>>()
            .join(", ");
        Err(SpiceError::parse(
            instance.location.clone(),
            format!(
                "model '{name}' referenced by {} is not defined and cannot be binned: \
                 {reason}; candidates {listed}",
                instance.name
            ),
        ))
    }

    /// Raw first declaration by case-insensitive name; no ground aliasing.
    #[must_use = "the returned declaration determines which model parameters are applied"]
    pub fn model(&self, name: &str) -> Option<&'a ModelCard> {
        self.declarations
            .get(name)
            .map(|declaration| declaration.bin.card)
    }

    /// Resolve an instance reference and validate its family and selector.
    /// Literal R/C/L/V/I may have no reference; D/Q/M require one. This does not
    /// validate arbitrary device keywords or make an unavailable factory work.
    ///
    /// # Errors
    /// Missing/wrong-family models, unsupported levels, invalid selector data.
    /// Diagnostics retain the instance/card/setter location as appropriate.
    pub fn resolve(&self, instance: &DeviceInstance) -> SpiceResult<Option<ResolvedModel<'a>>> {
        let Some(name) = &instance.model else {
            if ModelBackend::requires_model(instance.designator) {
                return Err(SpiceError::parse(
                    instance.location.clone(),
                    format!("device {} requires a model", instance.name),
                ));
            }
            return Ok(None);
        };
        let (declaration, binned) = if let Some(exact) = self.declarations.get(name) {
            (exact, false)
        } else {
            let declaration = self.select_declaration(instance, name)?.ok_or_else(|| {
                SpiceError::parse(
                    instance.location.clone(),
                    format!(
                        "model '{name}' referenced by {} is not defined",
                        instance.name
                    ),
                )
            })?;
            (declaration, true)
        };
        let card = declaration.bin.card;
        let family = declaration.family()?;
        if family.designator() != instance.designator.to_ascii_lowercase() {
            return Err(SpiceError::parse(
                instance.location.clone(),
                format!(
                    "wrong model family {family:?} for {}; model '{name}' declared at {}",
                    instance.name, card.location
                ),
            ));
        }
        let levels = declaration
            .levels(family)
            .map_err(|error| match (binned, error) {
                // Name the bin so an unported binnable family reads as such.
                (true, SpiceError::NotYetPorted { what, c_reference }) => {
                    SpiceError::not_yet_ported(
                        format!("{} binned to model '{}': {what}", instance.name, card.name),
                        c_reference,
                    )
                }
                (_, error) => error,
            })?;
        Ok(Some(ResolvedModel {
            card,
            family,
            levels,
            preparation: Arc::clone(declaration.preparation.get_or_init(Arc::default)),
        }))
    }
}

/// A resolved borrowed declaration. Parameter validation is an explicit,
/// device-owned schema operation, separate from lookup and selector choice.
#[derive(Debug, Clone)]
pub struct ResolvedModel<'a> {
    card: &'a ModelCard,
    family: ModelFamily,
    levels: LevelSelection,
    preparation: Arc<PreparationCache>,
}

impl<'a> ResolvedModel<'a> {
    /// Original first model card, with raw ordered setters and source locations.
    #[must_use = "the model card contains the selected declaration and parameters"]
    pub const fn card(&self) -> &'a ModelCard {
        self.card
    }

    /// Validated family.
    #[must_use = "the resolved family determines which device implementation is valid"]
    pub const fn family(&self) -> ModelFamily {
        self.family
    }

    /// Backend selection versus applied integer data.
    #[must_use = "the level selection determines the physical model variant"]
    pub const fn levels(&self) -> LevelSelection {
        self.levels
    }

    /// Shared immutable data for this exact declaration (not its name).
    pub(super) fn prepare<T: Send + Sync + 'static>(
        &self,
        key: PreparationKey,
        prepare: impl FnOnce() -> SpiceResult<T>,
    ) -> SpiceResult<Arc<T>> {
        self.preparation.get_or_prepare(key, prepare)
    }

    /// Apply a device-owned scalar schema, excluding `level` already consumed
    /// by selector/setter validation. Never ignores unknown physics keywords.
    ///
    /// # Errors
    /// The supplied schema's validation failures, with source provenance.
    pub fn parameters(&self, schema: &'static ScalarSchema<'static>) -> SpiceResult<ScalarValues> {
        let key = PreparationKey::Scalar(std::ptr::from_ref(schema).addr());
        self.prepare(key, || {
            schema.validate(
                self.card
                    .parameters
                    .iter()
                    .filter(|p| !p.name.eq_ignore_ascii_case("level")),
                &self.card.location,
            )
        })
        .map(|values| (*values).clone())
    }

    /// Validate the initial diode core: IS/N/RS/TNOM only. Defaults follow
    /// `diosetup.c`; temperature conversion follows `diompar.c::DIOmParam`.
    /// IS epsmin clamping and compatibility-mode RS substitutions are not
    /// applied by this input schema; numerical device setup remains separate.
    ///
    /// # Errors
    /// Wrong family, unknown setters, invalid ranges or context temperatures.
    pub fn diode_parameters(&self, context: &ModelContext) -> SpiceResult<DiodeModelParameters> {
        self.require_diode()?;
        context.validate(&self.card.location)?;
        let values = self.parameters(&DIODE_MODEL_SCHEMA)?;
        let nominal = values
            .get("tnom")
            .map_or(context.nominal_temperature, |v| v.value);
        Ok(DiodeModelParameters {
            saturation_current: required(&values, "is", &self.card.location)?,
            emission_coefficient: required(&values, "n", &self.card.location)?,
            series_resistance: required(&values, "rs", &self.card.location)?,
            nominal_temperature_kelvin: temperature_kelvin(
                nominal,
                values
                    .get("tnom")
                    .and_then(|v| v.location.as_ref())
                    .unwrap_or(&self.card.location),
            )?,
        })
    }

    /// Validate bounded diode AREA/TEMP setters. `diotemp.c::DIOtemp` uses
    /// circuit temperature when TEMP is omitted; TNOM does not set TEMP.
    /// Dtemp/multiplicity/geometry/IC/flags remain explicit gaps, not ignored.
    ///
    /// # Errors
    /// Mismatched instance/model, unknown setters, invalid ranges/context.
    pub fn diode_instance_parameters(
        &self,
        instance: &DeviceInstance,
        context: &ModelContext,
    ) -> SpiceResult<DiodeInstanceParameters> {
        self.require_diode()?;
        if !instance.designator.eq_ignore_ascii_case(&'d')
            || !instance
                .model
                .as_deref()
                .is_some_and(|name| name.eq_ignore_ascii_case(&self.card.name))
        {
            return Err(SpiceError::parse(
                instance.location.clone(),
                "instance does not reference this diode model",
            ));
        }
        context.validate(&instance.location)?;
        let values = DIODE_INSTANCE_SCHEMA.validate(&instance.parameters, &instance.location)?;
        let temperature = values.get("temp").map_or(context.temperature, |v| v.value);
        Ok(DiodeInstanceParameters {
            area: required(&values, "area", &instance.location)?,
            temperature_kelvin: temperature_kelvin(
                temperature,
                values
                    .get("temp")
                    .and_then(|v| v.location.as_ref())
                    .unwrap_or(&instance.location),
            )?,
        })
    }

    fn require_diode(&self) -> SpiceResult<()> {
        if self.family == ModelFamily::Diode {
            Ok(())
        } else {
            Err(SpiceError::parse(
                self.card.location.clone(),
                "diode schema requires a diode model",
            ))
        }
    }
}

/// Device-owned context, intentionally independent of analysis. Analysis
/// consumers can pass their temperature settings explicitly when factories land.
///
/// The context is an immutable per-point value: a typed DC sweep carries its
/// resistor and instance-parameter overrides here (see [`crate::devices::sweep`]) instead of mutating devices.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ModelContext {
    /// Use SPICE3 MOS1 flicker noise (`mos1noi.c`, newcompat.s3).
    pub spice3_noise: bool,
    /// Circuit temperature, degrees Celsius (instance TEMP default).
    pub temperature: Real,
    /// Nominal temperature, degrees Celsius (model TNOM default).
    pub nominal_temperature: Real,
    /// Per-point replacements of resistors' supplied scalars, applied by
    /// [`crate::devices::Circuit`] when it assembles or loads. Empty slots are `None`.
    pub resistor_overrides: [Option<ResistorOverride>; MAX_RESISTOR_OVERRIDES],
    /// Per-point replacements of device instance parameters (`.dc
    /// @inst[param]`), applied by [`crate::devices::Circuit`] when it assembles
    /// or loads. Empty slots are `None`.
    pub instance_overrides: [Option<InstanceOverride>; MAX_INSTANCE_OVERRIDES],
    /// Junction minimum conductance in siemens (C `CKTgmin`, `.option gmin`),
    /// added in parallel with every diode, BJT and MOS1 junction. It is not the
    /// artificial nodal continuation conductance of DC gmin stepping. Default
    /// [`DEFAULT_GMIN`]; must be finite and nonnegative.
    pub gmin: Real,
    /// Resolved circuit relative tolerance (C `CKTreltol`), used by device preparation.
    /// Independent of analysis-local Newton tolerance overrides.
    pub reltol: Real,
    /// The analysis frequency in Hz that a behavioural source's `hertz` reads
    /// (C `CKTomega / 2 pi`): the AC frequency while an AC analysis re-solves
    /// the operating point for a frequency-dependent circuit, 0 otherwise.
    pub frequency: Real,
    /// SPARSE pivot thresholds for every circuit matrix a load or assembly
    /// fills (C `CKTpivotAbsTol`/`CKTpivotRelTol`, `.option pivtol`/`pivrel`).
    pub pivot: crate::maths::PivotTolerances,
    /// C's `CKTsrcFact`: the source-stepping factor independent sources
    /// scale their loaded value by (1 outside source stepping).
    pub source_factor: Real,
    /// Whether the deck's task holds a `.noise` analysis: C's
    /// `noiseAnalGiven`, which `b4set.c` reads while it creates BSIM4's
    /// internal nodes (`tnoiMod=1` adds the drain/source primes for every
    /// analysis of such a deck). Set by
    /// [`crate::devices::Circuit::from_netlist_with_context`].
    pub noise_analysis: bool,
}

/// ngspice's default junction `gmin` (`cktntask.c`: `TSKgmin = 1e-12`).
pub const DEFAULT_GMIN: Real = 1e-12;
/// ngspice's default circuit relative tolerance (cktntask.c).
pub const DEFAULT_RELTOL: Real = 1e-3;

impl Default for ModelContext {
    fn default() -> Self {
        Self::new(27.0, 27.0)
    }
}
impl ModelContext {
    /// A context at explicit temperatures with no resistor overrides.
    #[must_use = "the new context carries the circuit and reference temperatures"]
    pub const fn new(temperature: Real, nominal_temperature: Real) -> Self {
        Self {
            temperature,
            nominal_temperature,
            resistor_overrides: [None; MAX_RESISTOR_OVERRIDES],
            instance_overrides: [None; MAX_INSTANCE_OVERRIDES],
            gmin: DEFAULT_GMIN,
            reltol: DEFAULT_RELTOL,
            frequency: 0.,
            spice3_noise: false,
            pivot: crate::maths::PivotTolerances::DEFAULT,
            source_factor: 1.,
            noise_analysis: false,
        }
    }

    /// This context with the matrix pivot thresholds replaced (validated on
    /// use, see [`crate::maths::PivotTolerances::validate`]).
    #[must_use = "discarding the returned context loses the requested pivot tolerances"]
    pub const fn with_pivot_tolerances(mut self, pivot: crate::maths::PivotTolerances) -> Self {
        self.pivot = pivot;
        self
    }

    /// This context with C's `CKTsrcFact` replaced ([`Self::source_factor`]).
    #[must_use = "discarding the returned context loses the requested source continuation factor"]
    pub const fn with_source_factor(mut self, factor: Real) -> Self {
        self.source_factor = factor;
        self
    }

    /// This context with the `hertz` frequency replaced (validated on use).
    #[must_use = "discarding the returned context loses the requested analysis frequency"]
    pub const fn with_frequency(mut self, frequency: Real) -> Self {
        self.frequency = frequency;
        self
    }

    /// This context with the circuit relative tolerance replaced (validated on use).
    #[must_use = "discarding the returned context loses the requested relative tolerance"]
    pub const fn with_reltol(mut self, reltol: Real) -> Self {
        self.reltol = reltol;
        self
    }

    /// This context with junction `gmin` replaced (validated on use, see
    /// [`Self::gmin`]).
    #[must_use = "discarding the returned context loses the requested minimum conductance"]
    pub const fn with_gmin(mut self, gmin: Real) -> Self {
        self.gmin = gmin;
        self
    }

    /// This context plus one resistor override, in the first free slot.
    ///
    /// # Errors
    /// [`SpiceError::Circuit`] when the same resistor is already overridden or
    /// every slot is used. The context itself is not changed on failure.
    pub fn with_resistor_override(mut self, target: ResistorOverride) -> SpiceResult<Self> {
        if self
            .resistor_overrides
            .iter()
            .flatten()
            .any(|existing| existing.device() == target.device())
        {
            return Err(SpiceError::circuit("duplicate resistor override"));
        }
        let slot = self
            .resistor_overrides
            .iter_mut()
            .find(|slot| slot.is_none())
            .ok_or_else(|| SpiceError::circuit("too many resistor overrides"))?;
        *slot = Some(target);
        Ok(self)
    }

    /// This context plus one instance-parameter override, in the first free
    /// slot. Two overrides may name the same device with different parameters.
    ///
    /// # Errors
    /// [`SpiceError::Circuit`] when the same parameter of the same device is
    /// already overridden or every slot is used. The context itself is not
    /// changed on failure.
    pub fn with_instance_override(mut self, target: InstanceOverride) -> SpiceResult<Self> {
        if self.instance_overrides.iter().flatten().any(|existing| {
            existing.device() == target.device() && existing.parameter() == target.parameter()
        }) {
            return Err(SpiceError::circuit("duplicate instance parameter override"));
        }
        let slot = self
            .instance_overrides
            .iter_mut()
            .find(|slot| slot.is_none())
            .ok_or_else(|| SpiceError::circuit("too many instance parameter overrides"))?;
        *slot = Some(target);
        Ok(self)
    }

    /// Exact identity of every immutable load setting. Ordinary floating
    /// equality does not distinguish signed zero, including swept values.
    pub(super) fn same_preparation(&self, other: &Self) -> bool {
        if self.spice3_noise != other.spice3_noise || self.noise_analysis != other.noise_analysis {
            return false;
        }
        let bits = |context: &Self| {
            [
                context.temperature,
                context.nominal_temperature,
                context.gmin,
                context.reltol,
                context.frequency,
                context.pivot.absolute,
                context.pivot.relative,
                context.source_factor,
            ]
            .map(Real::to_bits)
        };
        bits(self) == bits(other)
            && self.resistor_overrides.map(|slot| {
                slot.map(|value| (value.topology(), value.device(), value.supplied().to_bits()))
            }) == other.resistor_overrides.map(|slot| {
                slot.map(|value| (value.topology(), value.device(), value.supplied().to_bits()))
            })
            && self.instance_overrides.map(|slot| {
                slot.map(|value| {
                    (
                        value.topology(),
                        value.device(),
                        value.parameter(),
                        value.value().to_bits(),
                    )
                })
            }) == other.instance_overrides.map(|slot| {
                slot.map(|value| {
                    (
                        value.topology(),
                        value.device(),
                        value.parameter(),
                        value.value().to_bits(),
                    )
                })
            })
    }

    pub(crate) fn validate(&self, location: &SourceLoc) -> SpiceResult<()> {
        temperature_kelvin(self.temperature, location)?;
        temperature_kelvin(self.nominal_temperature, location)?;
        if !(self.frequency.is_finite() && self.frequency >= 0.) {
            return Err(SpiceError::parse(
                location.clone(),
                format!(
                    "analysis frequency must be finite and nonnegative, got {}",
                    self.frequency
                ),
            ));
        }
        if !(self.gmin.is_finite() && self.gmin >= 0.) {
            return Err(SpiceError::parse(
                location.clone(),
                format!(
                    "junction gmin must be finite and nonnegative, got {}",
                    self.gmin
                ),
            ));
        }
        if !(self.reltol.is_finite() && self.reltol > 0.) {
            return Err(SpiceError::parse(
                location.clone(),
                format!(
                    "circuit RELTOL must be finite and positive, got {}",
                    self.reltol
                ),
            ));
        }
        self.pivot
            .validate()
            .map_err(|message| SpiceError::parse(location.clone(), message))
    }
}

/// Validated diode model inputs, not diode simulation equations.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DiodeModelParameters {
    /// IS in amperes, strictly positive (default 1e-14 A).
    pub saturation_current: Real,
    /// N, strictly positive (default 1).
    pub emission_coefficient: Real,
    /// RS in ohms, nonnegative (default 0).
    pub series_resistance: Real,
    /// TNOM in Kelvin, strictly positive (context default 300.15 K).
    pub nominal_temperature_kelvin: Real,
}
/// Validated diode instance inputs, not topology/stamping or initialization.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DiodeInstanceParameters {
    /// Area scale, strictly positive (default 1).
    pub area: Real,
    /// TEMP in Kelvin, strictly positive (context default 300.15 K).
    pub temperature_kelvin: Real,
}

static DIODE_MODEL_SCHEMA: ScalarSchema<'static> = ScalarSchema::new(&[
    ScalarParameter {
        name: "is",
        unit: ScalarUnit::Ampere,
        domain: ScalarDomain::Positive,
        default: Some(1e-14),
    },
    ScalarParameter {
        name: "n",
        unit: ScalarUnit::Dimensionless,
        domain: ScalarDomain::Positive,
        default: Some(1.0),
    },
    ScalarParameter {
        name: "rs",
        unit: ScalarUnit::Ohm,
        domain: ScalarDomain::NonNegative,
        default: Some(0.0),
    },
    ScalarParameter {
        name: "tnom",
        unit: ScalarUnit::Celsius,
        domain: ScalarDomain::Temperature,
        default: None,
    },
]);
static DIODE_INSTANCE_SCHEMA: ScalarSchema<'static> = ScalarSchema::new(&[
    ScalarParameter {
        name: "area",
        unit: ScalarUnit::Dimensionless,
        domain: ScalarDomain::Positive,
        default: Some(1.0),
    },
    ScalarParameter {
        name: "temp",
        unit: ScalarUnit::Celsius,
        domain: ScalarDomain::Temperature,
        default: None,
    },
]);

fn required(values: &ScalarValues, name: &str, location: &SourceLoc) -> SpiceResult<Real> {
    values.get(name).map(|v| v.value).ok_or_else(|| {
        SpiceError::parse(
            location.clone(),
            format!("missing schema default for {name}"),
        )
    })
}
fn family(card: &ModelCard) -> SpiceResult<ModelFamily> {
    ModelFamily::from_base(&card.base).ok_or_else(|| SpiceError::Unsupported {
        feature: format!("model family '{}'", card.base),
        location: Some(card.location.clone()),
    })
}

fn rounded_level(raw: Real, location: &SourceLoc) -> SpiceResult<u8> {
    // Stricter than INPfindLev's warning/fallback and unsafe C casts. Reject
    // negative raw/nonfinite/out-of-range values, including invalid duplicates.
    // C's shared rounding rule is separate from the model-level range.
    match crate::primitives::c_integer(raw).and_then(|rounded| u8::try_from(rounded).ok()) {
        Some(rounded) if raw >= 0. && rounded <= 99 => Ok(rounded),
        _ => Err(SpiceError::parse(
            location.clone(),
            "model level must be finite, nonnegative and round into 0..=99",
        )),
    }
}

/// The C device directory `inpdomod.c` selects for an unported NMOS/PMOS
/// level.
const fn mos_level_reference(selector: u8) -> &'static str {
    match selector {
        4 => "src/spicelib/devices/bsim1/",
        5 => "src/spicelib/devices/bsim2/",
        10 | 58 => "src/spicelib/devices/bsimsoi/",
        55 => "src/spicelib/devices/bsim3soi_fd/",
        56 => "src/spicelib/devices/bsim3soi_dd/",
        57 => "src/spicelib/devices/bsim3soi_pd/",
        60 => "src/spicelib/devices/soi3/",
        68 => "src/spicelib/devices/hisim2/",
        73 => "src/spicelib/devices/hisimhv1/, hisimhv2/",
        _ => "src/spicelib/parser/inpdomod.c (no MOS device for this level)",
    }
}

fn levels(card: &ModelCard, family: ModelFamily) -> SpiceResult<LevelSelection> {
    let mut first = None;
    let mut last = None;
    for p in &card.parameters {
        if p.name.eq_ignore_ascii_case("level") {
            let raw = finite_literal(p)?;
            if family == ModelFamily::SimetrixDiodeLevel1 && raw.to_bits() != 1_f64.to_bits() {
                return Err(SpiceError::parse(
                    p.location.clone(),
                    format!("SIMetrix diode requires LEVEL=1, got {raw}"),
                ));
            }
            let integer = rounded_level(raw, &p.location)?;
            first.get_or_insert((raw, integer, &p.location));
            last = Some(integer);
        }
    }
    let raw = first.map(|(raw, _, _)| raw);
    if card.level != raw {
        return Err(SpiceError::parse(
            card.location.clone(),
            "raw level cache must equal the first ordered level setter",
        ));
    }
    let first_integer = first.map_or(1, |(_, integer, _)| integer);
    let location = first.map_or(&card.location, |(_, _, location)| location);
    // Preserve the first rounded selector, including aliases. Fixed-backend
    // families have separate level setter policy.
    let (selector, valid_setter, applied) = match family {
        ModelFamily::Npn
        | ModelFamily::Pnp
        | ModelFamily::Nmos
        | ModelFamily::Pmos
        | ModelFamily::Njf
        | ModelFamily::Pjf
        | ModelFamily::Nmf
        | ModelFamily::Pmf => (first_integer, true, None),
        ModelFamily::Resistor if card.base.eq_ignore_ascii_case("r") => (first_integer, true, None),
        ModelFamily::Diode => {
            let applied = last.unwrap_or(1);
            (1, applied == 1, Some(applied))
        }
        _ => (1, first_integer == 1, None),
    };
    let Some(backend) = ModelBackend::select(family, selector).filter(|_| valid_setter) else {
        if matches!(family, ModelFamily::Njf | ModelFamily::Pjf) {
            return Err(SpiceError::not_yet_ported(
                format!(
                    "{location}: {family:?} model '{}' level {selector}",
                    card.name
                ),
                "src/spicelib/parser/inpdomod.c (only JFET levels 1-2 exist in C)",
            ));
        }
        if matches!(family, ModelFamily::Nmf | ModelFamily::Pmf) {
            return Err(SpiceError::not_yet_ported(
                format!(
                    "{location}: {family:?} model '{}' level {selector}",
                    card.name
                ),
                "src/spicelib/parser/inpdomod.c (only MESFET levels 1-4 and HFET levels 5-6 exist in C)",
            ));
        }
        let setter_location = if family == ModelFamily::Diode {
            card.parameters
                .iter()
                .rev()
                .find(|p| p.name.eq_ignore_ascii_case("level"))
                .map_or(location, |p| &p.location)
        } else {
            location
        };
        let reference = if matches!(family, ModelFamily::Nmos | ModelFamily::Pmos) {
            mos_level_reference(selector)
        } else {
            "src/spicelib/parser/inpdomod.c, inpfindl.c, inpgval.c; src/spicelib/devices/dio/diosetup.c"
        };
        return Err(SpiceError::not_yet_ported(
            format!(
                "{setter_location}: {family:?} model '{}' selector {selector}, applied level {applied:?}",
                card.name
            ),
            reference,
        ));
    };
    match backend {
        ModelBackend::Bsim3 => {
            crate::devices::bsim3::select_version(card)?;
        }
        ModelBackend::Bsim4 => {
            crate::devices::bsim4::select_version(card)?;
        }
        _ => {}
    }
    Ok(LevelSelection {
        first_raw: raw,
        selector,
        backend,
        applied,
    })
}
