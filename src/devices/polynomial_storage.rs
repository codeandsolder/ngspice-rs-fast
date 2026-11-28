//! Prepared differential storage laws. The producer owns the scalar
//! constitutive equation; the solver owns temperature binding, topology,
//! exact physical residuals, and transactional integration history.
use model_normalize_ngspice::passive_storage::{PolynomialStorage, StoragePoint};

use crate::devices::{
    ConvergenceContext, Device, InitialCoordinate, InitialCoordinateHistory, InitialVoltageStorage,
    ModelContext, ModelFamily, ResolvedModel, StampContext,
    linear::LinearContext,
    noise::{DeviceNoise, NoiseContext, NoiseSource},
    nonlinear::charge::ChargeBranch,
    physical_dae::TaylorContext,
    schema::{
        ScalarDomain as D, ScalarParameter, ScalarSchema, ScalarUnit as U, ScalarValues,
        temperature_kelvin,
    },
    setup::MatrixSetup,
    traits::{AnalysisMode, ChargeContext, MnaUnknowns, VoltageStorageHistory},
};
use crate::maths::{SparseMatrix, Vector, taylor::Jet};
use crate::netlist::ast::DeviceInstance;
use crate::primitives::{NodeAppend, NodeId, Real, SourceLoc, SpiceError, SpiceResult};

const fn parameter(
    name: &'static str,
    unit: U,
    domain: D,
    default: Option<Real>,
) -> ScalarParameter {
    ScalarParameter {
        name,
        unit,
        domain,
        default,
    }
}

const TC1: ScalarParameter = parameter("tc1", U::InverseKelvin, D::Finite, None);
const TC2: ScalarParameter = parameter("tc2", U::InverseKelvinSquared, D::Finite, None);
const TEMP: ScalarParameter = parameter("temp", U::Celsius, D::Temperature, None);
const DTEMP: ScalarParameter = parameter("dtemp", U::Kelvin, D::Finite, None);
const TNOM: ScalarParameter = parameter("tnom", U::Celsius, D::Temperature, None);
const M: ScalarParameter = parameter("m", U::Dimensionless, D::Positive, Some(1.));
const SCALE: ScalarParameter = parameter("scale", U::Dimensionless, D::Positive, Some(1.));
static CAP_MODEL: ScalarSchema<'static> = ScalarSchema::new(&[
    parameter("c", U::Dimensionless, D::Finite, Some(1.)),
    parameter("vc1", U::InverseVolt, D::Finite, Some(0.)),
    parameter("vc2", U::InverseVoltSquared, D::Finite, Some(0.)),
    TC1,
    TC2,
    TEMP,
    DTEMP,
    TNOM,
]);
static IND_MODEL: ScalarSchema<'static> = ScalarSchema::new(&[
    parameter("l", U::Dimensionless, D::Finite, Some(1.)),
    parameter("il1", U::InverseAmpere, D::Finite, Some(0.)),
    parameter("il2", U::InverseAmpereSquared, D::Finite, Some(0.)),
    TC1,
    TC2,
    TEMP,
    DTEMP,
    TNOM,
]);
static CAP_INSTANCE: ScalarSchema<'static> = ScalarSchema::new(&[
    parameter("capacitance", U::Farad, D::NonZero, None),
    parameter("ic", U::Volt, D::Finite, None),
    TC1,
    TC2,
    TEMP,
    DTEMP,
    M,
    SCALE,
]);
static IND_INSTANCE: ScalarSchema<'static> = ScalarSchema::new(&[
    parameter("inductance", U::Henry, D::NonZero, None),
    parameter("ic", U::Ampere, D::Finite, None),
    TC1,
    TC2,
    TEMP,
    DTEMP,
    M,
    SCALE,
]);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Kind {
    Capacitor,
    Inductor,
}

impl Kind {
    const fn model_schema(self) -> &'static ScalarSchema<'static> {
        match self {
            Self::Capacitor => &CAP_MODEL,
            Self::Inductor => &IND_MODEL,
        }
    }
    const fn instance_schema(self) -> &'static ScalarSchema<'static> {
        match self {
            Self::Capacitor => &CAP_INSTANCE,
            Self::Inductor => &IND_INSTANCE,
        }
    }
    const fn principal(self) -> &'static str {
        match self {
            Self::Capacitor => "capacitance",
            Self::Inductor => "inductance",
        }
    }
}

fn scalar(values: &ScalarValues, key: &str) -> Option<Real> {
    values.get(key).map(|value| value.value)
}

#[derive(Debug, Clone)]
struct Recipe {
    kind: Kind,
    model: ScalarValues,
    instance: ScalarValues,
    location: SourceLoc,
}

#[derive(Debug, Clone, Copy)]
struct Prepared {
    law: PolynomialStorage,
    // These are derivative coefficients for exact Taylor lifting of the
    // producer's scalar evaluation (not a separate scalar constitutive law).
    base: Real,
    linear: Real,
    quadratic: Real,
}

impl Prepared {
    fn at(self, x: Real) -> SpiceResult<StoragePoint> {
        self.law
            .evaluate(x)
            .map_err(|error| SpiceError::circuit(error.to_string()))
    }

    fn taylor(self, x: Jet) -> SpiceResult<(Jet, Jet)> {
        if self.base == 0. {
            return Ok((Jet::constant(x.order(), 0.), Jet::constant(x.order(), 0.)));
        }
        let center = x.value();
        let point = self.at(center)?;
        let delta = x.offset(-center);
        let square = delta.multiply(&delta);
        let cubic = square.multiply(&delta);
        let second = self.base * self.quadratic;
        let curvature = (2. * second).mul_add(center, self.base * self.linear);
        let value = Jet::constant(delta.order(), point.stored)
            .add(&delta.clone().scale(point.differential))
            .add(&square.clone().scale(curvature / 2.))
            .add(&cubic.scale(second / 3.));
        let gradient = Jet::constant(delta.order(), point.differential)
            .add(&delta.scale(curvature))
            .add(&square.scale(second));
        value.check()?;
        gradient.check()?;
        Ok((value, gradient))
    }
}

impl Recipe {
    fn setting(&self, key: &str) -> Option<Real> {
        scalar(&self.instance, key).or_else(|| scalar(&self.model, key))
    }

    fn prepare(&self, context: &ModelContext) -> SpiceResult<Prepared> {
        let (multiplier, linear, quadratic) = match self.kind {
            Kind::Capacitor => ("c", "vc1", "vc2"),
            Kind::Inductor => ("l", "il1", "il2"),
        };
        let base = scalar(&self.instance, self.kind.principal()).ok_or_else(|| {
            SpiceError::parse(
                self.location.clone(),
                "polynomial storage requires an instance value",
            )
        })?;
        let temp = self
            .setting("temp")
            .unwrap_or_else(|| context.temperature + self.setting("dtemp").unwrap_or(0.));
        let nominal = scalar(&self.model, "tnom").unwrap_or(context.nominal_temperature);
        let delta = temperature_kelvin(temp, &self.location)?
            - temperature_kelvin(nominal, &self.location)?;
        let factor = self
            .setting("tc2")
            .unwrap_or(0.)
            .mul_add(delta, self.setting("tc1").unwrap_or(0.))
            .mul_add(delta, 1.);
        let base = base
            * scalar(&self.model, multiplier).unwrap_or(1.)
            * factor
            * self.setting("scale").unwrap_or(1.);
        let m = self.setting("m").unwrap_or(1.);
        let linear = scalar(&self.model, linear).unwrap_or(0.);
        let quadratic = scalar(&self.model, quadratic).unwrap_or(0.);
        // Parallel copies share voltage; total inductor current is m times
        // the current in one copy. Transform the polynomial's argument too.
        let (base, linear, quadratic) = match self.kind {
            Kind::Capacitor => (base * m, linear, quadratic),
            Kind::Inductor => (base / m, linear / m, quadratic / m / m),
        };
        let law = PolynomialStorage::new(base, linear, quadratic)
            .map_err(|error| SpiceError::parse(self.location.clone(), error.to_string()))?;
        Ok(Prepared {
            law,
            base,
            linear,
            quadratic,
        })
    }
}

#[derive(Debug, Clone)]
struct PolynomialDevice {
    name: String,
    model_name: String,
    terminals: [NodeId; 2],
    recipe: Recipe,
    prepared: Prepared,
}

impl PolynomialDevice {
    fn rows(&self, unknowns: &MnaUnknowns, branch: Option<usize>) -> SpiceResult<Rows> {
        Ok(Rows {
            terminals: self.terminals.map(|node| unknowns.node_row(node)),
            branch: branch
                .ok_or_else(|| SpiceError::circuit("missing polynomial storage branch"))?,
            kind: self.recipe.kind,
        })
    }

    fn initial(&self) -> Option<Real> {
        self.recipe.setting("ic")
    }

    fn with_recipe(&self, recipe: Recipe, context: &ModelContext) -> SpiceResult<Self> {
        Ok(Self {
            name: self.name.clone(),
            model_name: self.model_name.clone(),
            terminals: self.terminals,
            prepared: recipe.prepare(context)?,
            recipe,
        })
    }
}

#[derive(Clone, Copy)]
struct Rows {
    terminals: [Option<usize>; 2],
    branch: usize,
    kind: Kind,
}

impl Rows {
    const fn controls(self) -> [Option<usize>; 2] {
        match self.kind {
            Kind::Capacitor => self.terminals,
            Kind::Inductor => [Some(self.branch), None],
        }
    }

    fn coordinate(self, state: &Vector) -> SpiceResult<Real> {
        let [p, n] = self.controls();
        let value = |row: Option<usize>| {
            row.map_or(Ok(0.), |row| {
                state
                    .get(row)
                    .ok_or_else(|| SpiceError::circuit("missing storage coordinate"))
            })
        };
        Ok(value(p)? - value(n)?)
    }

    fn static_entries(self) -> [(Option<usize>, Option<usize>, Real); 5] {
        let [p, n] = self.terminals;
        let branch = Some(self.branch);
        let (v, i) = match self.kind {
            Kind::Capacitor => (0., 1.),
            Kind::Inductor => (1., 0.),
        };
        [
            (p, branch, 1.),
            (n, branch, -1.),
            (branch, p, v),
            (branch, n, -v),
            (branch, branch, i),
        ]
    }

    fn static_stamp(self, matrix: &mut SparseMatrix) -> SpiceResult<()> {
        for (row, column, value) in self.static_entries() {
            if let (Some(row), Some(column)) = (row, column)
                && value != 0.
            {
                matrix.add(row, column, value)?;
            }
        }
        Ok(())
    }

    fn dynamic_stamp(self, matrix: &mut SparseMatrix, gradient: Real) -> SpiceResult<()> {
        for (column, sign) in self.controls().into_iter().zip([-1., 1.]) {
            if let Some(column) = column {
                matrix.add(self.branch, column, sign * gradient)?;
            }
        }
        Ok(())
    }
}

impl Device for PolynomialDevice {
    fn name(&self) -> &str {
        &self.name
    }
    fn model_name(&self) -> Option<&str> {
        Some(&self.model_name)
    }
    fn designator(&self) -> char {
        match self.recipe.kind {
            Kind::Capacitor => 'c',
            Kind::Inductor => 'l',
        }
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
    fn state_count(&self) -> usize {
        2
    }
    fn truncation_slot(&self) -> Option<usize> {
        Some(0)
    }
    fn is_nonlinear(&self) -> bool {
        true
    }
    fn limits_voltage_steps(&self) -> bool {
        false
    }

    fn setup_matrix(&self, setup: &mut MatrixSetup<'_>) -> SpiceResult<bool> {
        let branch = Some(setup.branch(0)?);
        let [p, n] = self.terminals.map(|node| setup.node(node));
        for (row, col) in [
            (p, branch),
            (n, branch),
            (branch, p),
            (branch, n),
            (branch, branch),
        ] {
            setup.element(row, col);
        }
        Ok(true)
    }

    fn needs_context_preparation(&self) -> bool {
        true
    }
    fn reprepare_context(&self, context: &ModelContext) -> SpiceResult<Option<Box<dyn Device>>> {
        let prepared = self.recipe.prepare(context)?;
        if prepared.law == self.prepared.law {
            return Ok(None);
        }
        let mut next = self.clone();
        next.prepared = prepared;
        Ok(Some(Box::new(next)))
    }
    fn validate_dc_inputs(&self, context: &ModelContext) -> SpiceResult<()> {
        self.recipe.prepare(context).map(|_| ())
    }
    fn model_candidate_setters_are_stable(&self) -> bool {
        true
    }
    fn instance_parameter(&self, keyword: &str) -> Option<&'static str> {
        let key = match (self.recipe.kind, keyword) {
            (Kind::Capacitor, "c" | "cap") | (Kind::Inductor, "l") => self.recipe.kind.principal(),
            _ => keyword,
        };
        self.recipe
            .kind
            .instance_schema()
            .parameters()
            .iter()
            .find(|parameter| parameter.name.eq_ignore_ascii_case(key))
            .map(|parameter| parameter.name)
    }
    fn with_instance_parameter(
        &self,
        keyword: &str,
        value: Real,
        context: &ModelContext,
    ) -> SpiceResult<Box<dyn Device>> {
        let definition = self
            .recipe
            .kind
            .instance_schema()
            .parameters()
            .iter()
            .find(|p| p.name == keyword)
            .ok_or_else(|| SpiceError::circuit("unsupported polynomial storage setter"))?;
        if !definition.domain.accepts(value) {
            return Err(SpiceError::circuit("invalid polynomial storage setter"));
        }
        let mut recipe = self.recipe.clone();
        recipe.instance.set(
            definition.name,
            value,
            definition.unit,
            recipe.location.clone(),
        )?;
        Ok(Box::new(self.with_recipe(recipe, context)?))
    }

    fn initial_coordinates(&self, _context: &ModelContext) -> SpiceResult<Vec<InitialCoordinate>> {
        Ok(if self.recipe.kind == Kind::Inductor {
            vec![InitialCoordinate {
                coordinate: 0,
                equation: 0,
                value: self.initial().unwrap_or(0.),
                history: InitialCoordinateHistory::EquationCharge {
                    slot: 0,
                    scale: -1.,
                },
            }]
        } else {
            Vec::new()
        })
    }

    fn initial_voltage_storage(
        &self,
        context: &ModelContext,
    ) -> SpiceResult<Option<InitialVoltageStorage>> {
        if self.recipe.kind == Kind::Inductor {
            return Ok(None);
        }
        Ok(Some(InitialVoltageStorage {
            voltage: self.initial(),
            slot: 0,
            history: VoltageStorageHistory::Polynomial {
                law: self.recipe.prepare(context)?.law,
                equation_scale: -1.,
            },
            equation: Some(0),
        }))
    }

    fn stamp(&self, context: &mut StampContext<'_>) -> SpiceResult<()> {
        let rows = self.rows(context.unknowns, Some(context.branch(0)?))?;
        let coordinate = rows.coordinate(context.solution)?;
        let x = if context.states.initial_conditions() {
            self.initial().unwrap_or(coordinate)
        } else {
            coordinate
        };
        let point = self.prepared.at(x)?;
        let storage = ChargeBranch::new(0, point.stored, point.differential);
        rows.static_stamp(context.matrix)?;
        if let AnalysisMode::Transient { dt, .. } = context.mode {
            let integration = context
                .integration
                .ok_or_else(|| SpiceError::circuit("missing polynomial storage integration"))?;
            if dt.to_bits() != integration.dt().to_bits() {
                return Err(SpiceError::circuit("polynomial storage timestep mismatch"));
            }
            let companion = storage.integrate(&mut context.states, integration)?;
            rows.dynamic_stamp(context.matrix, companion.conductance)?;
            let rate = match self.recipe.kind {
                Kind::Capacitor => context.row_value(rows.branch)?,
                Kind::Inductor => {
                    context.node_voltage(self.terminals[0])
                        - context.node_voltage(self.terminals[1])
                }
            };
            context.states.set(1, rate)?;
            context.rhs.add_to(
                rows.branch,
                (-companion.conductance).mul_add(x, companion.derivative),
            )?;
        } else {
            storage.zero_derivative(&mut context.states)?;
        }
        storage.store(&mut context.states)
    }

    fn convergence_test(&self, context: &ConvergenceContext<'_>) -> SpiceResult<bool> {
        let rows = self.rows(context.unknowns, context.branch)?;
        let x = rows.coordinate(context.solution)?;
        let point = self.prepared.at(x)?;
        let actual = match self.recipe.kind {
            Kind::Capacitor => context
                .solution
                .get(rows.branch)
                .ok_or_else(|| SpiceError::circuit("missing polynomial branch current"))?,
            Kind::Inductor => {
                context.voltage(self.terminals[0]) - context.voltage(self.terminals[1])
            }
        };
        let absolute = match self.recipe.kind {
            Kind::Capacitor => context.abstol,
            Kind::Inductor => context.vntol,
        };
        let (expected, rounding) = if let Some(integration) = context.integration {
            let storage = ChargeBranch::new(0, point.stored, 1.);
            let companion =
                storage.companion(integration, |age, slot| context.accepted(age, slot))?;
            let roundoff = (8. * Real::EPSILON * companion.conductance.abs()).mul_add(
                point.stored.abs(),
                8. * Real::EPSILON * companion.current.abs(),
            );
            (companion.derivative, roundoff)
        } else {
            (0., 0.)
        };
        Ok((actual - expected).abs()
            > context
                .reltol
                .mul_add(actual.abs().max(expected.abs()), absolute + rounding))
    }

    fn assemble_small_signal(
        &self,
        context: &mut LinearContext<'_>,
        bias: &Vector,
    ) -> SpiceResult<()> {
        let rows = self.rows(context.unknowns, context.branch)?;
        rows.static_stamp(&mut context.system.a)?;
        rows.dynamic_stamp(
            &mut context.system.e,
            self.prepared.at(rows.coordinate(bias)?)?.differential,
        )
    }

    fn assemble_pole_zero(
        &self,
        context: &mut LinearContext<'_>,
        bias: &Vector,
    ) -> SpiceResult<()> {
        self.assemble_small_signal(context, bias)
    }

    fn assemble_nonlinear_charge(&self, context: &mut ChargeContext<'_>) -> SpiceResult<()> {
        let rows = self.rows(context.unknowns, context.branch)?;
        let point = self.prepared.at(rows.coordinate(context.state)?)?;
        context.charge.add_to(rows.branch, -point.stored)?;
        rows.dynamic_stamp(context.jacobian, point.differential)
    }

    fn assemble_nonlinear_taylor(&self, context: &mut TaylorContext<'_>) -> SpiceResult<()> {
        let rows = self.rows(context.unknowns, context.branch)?;
        for (row, col, value) in rows.static_entries() {
            context.linear_static(row, col, value)?;
        }
        let [p, n] = rows.controls();
        let coordinate = context.variable(p)?.add(&context.variable(n)?.scale(-1.));
        let (value, gradient) = self.prepared.taylor(coordinate)?;
        context.branch_taylor([None, Some(rows.branch)], [p, n], &value, &gradient, true)
    }

    fn noise(
        &self,
        _: &NoiseContext<'_>,
        _: &mut Vec<NoiseSource>,
    ) -> SpiceResult<DeviceNoise<'_>> {
        Ok(DeviceNoise::Noiseless)
    }

    fn physical_terminal_current(&self, _: &[Real], branches: &[Real]) -> Option<Real> {
        branches.first().copied()
    }
}

pub(super) fn instantiate(
    instance: &DeviceInstance,
    nodes: &mut NodeAppend<'_>,
    model: &ResolvedModel<'_>,
    context: &ModelContext,
) -> SpiceResult<Box<dyn Device>> {
    let kind = match model.family() {
        ModelFamily::PolynomialCapacitor => Kind::Capacitor,
        ModelFamily::PolynomialInductor => Kind::Inductor,
        _ => return Err(SpiceError::circuit("invalid polynomial storage family")),
    };
    let recipe = Recipe {
        kind,
        model: model.parameters(kind.model_schema())?,
        instance: kind
            .instance_schema()
            .validate(&instance.parameters, &instance.location)?,
        location: instance.location.clone(),
    };
    let prepared = recipe.prepare(context)?;
    Ok(Box::new(PolynomialDevice {
        name: instance.name.clone(),
        model_name: model.card().name.clone(),
        terminals: nodes.intern_array(&instance.nodes)?,
        recipe,
        prepared,
    }))
}
