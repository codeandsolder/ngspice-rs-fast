//! Native three-terminal VDMOS topology-bound stamping of prepared
//! `model-kernel-vdmos` physics from `spice-redox`.
//!
//! Source parsing, symbol resolution and physical derivatives stay upstream.
//! MNA bindings, Newton RHS, trial state and numerical integration belong here.

use std::sync::Arc;

use model_kernel_vdmos::{Bias, Evaluation, Instance, Polarity, PreparedModel, compile_ngspice};

use crate::devices::limiting::{Limiter, Linearization, critical_voltage};
use crate::devices::linear::LinearContext;
use crate::devices::models::{ModelContext, PreparationKey, ResolvedModel};
use crate::devices::noise::{
    DeviceNoise, FlickerLaw, NoiseContext, NoiseFamily, NoiseKind, NoiseSource,
};
use crate::devices::nonlinear::charge::ChargeBranch;
use crate::devices::traits::{Device, StampContext};
use crate::devices::{DeviceState, IterationPhase};
use crate::maths::Vector;
use crate::maths::integrator::{Coefficients, Companion};
use crate::netlist::ast::{DeviceInstance, ParameterKind};
use crate::primitives::{NodeAppend, NodeId, NodeKind, Real, SpiceError, SpiceResult};

/// Shared prepared physical model, independent of each transistor's geometry
/// and circuit-specific hidden nodes.
#[derive(Debug)]
pub struct Vdmos {
    name: String,
    model_name: String,
    terminals: [NodeId; 3],
    inner: [NodeId; 3],
    diode_prime: NodeId,
    qp: Option<NodeId>,
    thermal: Option<[NodeId; 2]>,
    model: Arc<PreparedModel>,
    multiplier: Real,
    temperature: Option<Real>,
    dtemp: Real,
}

/// Electrical dissipated power, analytically differentiated with respect to
/// the junction temperature and the bound physical electrical nodes.
#[derive(Debug, Clone, Copy)]
struct ThermalPower {
    value: Real,
    d_temperature: Real,
    d_electrical: [(NodeId, Real); 12],
}

// Charge and derivative pairs are followed by one junction-limiting slot.
const QGS: usize = 0;
const QGD: usize = 2;
const QDIO: usize = 4;
const DIODE_VOLTAGE: usize = 6;
const QP_CHARGE: usize = 7;
const TEMPERATURE_AFTER_QP: usize = 9;
const THERMAL_LOG_LIMIT: Real = 10.0; // C `vdmosload.c` -> `DEVlimitlog`

fn charge_companion(
    branch: ChargeBranch,
    states: &mut DeviceState<'_>,
    coefficients: Option<&Coefficients>,
    initial: bool,
) -> SpiceResult<Option<Companion>> {
    let companion = if let Some(coefficients) = coefficients {
        if initial {
            branch.seed(states)?;
        }
        let integrated = branch.integrate(states, coefficients)?;
        if initial {
            branch.seed_derivative(states, integrated.derivative)?;
        }
        Some(integrated)
    } else {
        branch.zero_derivative(states)?;
        None
    };
    branch.store(states)?;
    Ok(companion)
}

fn invalid(source: impl std::fmt::Display, what: impl std::fmt::Display) -> SpiceError {
    SpiceError::circuit(format!("{source}: {what}"))
}

fn scalar(parameter: &crate::netlist::ast::ParameterAssignment) -> SpiceResult<Real> {
    if parameter.kind != ParameterKind::Scalar {
        return Err(SpiceError::Unsupported {
            feature: format!(
                "VDMOS parameter {} requires a scalar literal",
                parameter.name
            ),
            location: Some(parameter.location.clone()),
        });
    }
    crate::primitives::parse_spice_number(&parameter.value)
        .filter(|value| value.is_finite())
        .ok_or_else(|| {
            invalid(
                &parameter.location,
                format!("invalid {} value", parameter.name),
            )
        })
}

impl Vdmos {
    /// Resolve one declaration into a shared prepared kernel and bind its
    /// immutable MNA terminals once, before any numerical analysis begins.
    ///
    /// # Errors
    ///
    /// Rejects unsupported physical model/instance options and invalid
    /// parameter domains, without substituting a generic MOS backend.
    pub(super) fn instantiate(
        instance: &DeviceInstance,
        nodes: &mut NodeAppend<'_>,
        resolved: &ResolvedModel<'_>,
        context: &ModelContext,
    ) -> SpiceResult<Box<dyn Device>> {
        let shared = resolved.prepare(
            PreparationKey::Nominal(context.nominal_temperature.to_bits()),
            || {
                let mut polarity = if resolved.card().base == "vdmosp" {
                    Polarity::P
                } else {
                    Polarity::N
                };
                let mut values = Vec::with_capacity(resolved.card().parameters.len());
                let mut explicit_tnom = false;
                for parameter in &resolved.card().parameters {
                    match parameter.name.as_str() {
                        "pchan" | "nchan" if parameter.kind == ParameterKind::Flag => {
                            polarity = if parameter.name == "pchan" {
                                Polarity::P
                            } else {
                                Polarity::N
                            };
                        }
                        "level" => {
                            if scalar(parameter)? != 1.0 {
                                return Err(invalid(
                                    &parameter.location,
                                    "VDMOS has only native level 1",
                                ));
                            }
                        }
                        name => {
                            explicit_tnom |= name == "tnom";
                            values.push((name, scalar(parameter)?));
                        }
                    }
                }
                let prepared = compile_ngspice(polarity, values)
                    .map_err(|error| invalid(&resolved.card().location, error))?;
                if explicit_tnom {
                    return Ok(prepared);
                }
                let mut model = *prepared.parameters();
                model.tnom = context.nominal_temperature + 273.15;
                model
                    .prepare()
                    .map_err(|error| invalid(&resolved.card().location, error))
            },
        )?;
        let mut multiplier = 1.0;
        let mut temperature = None;
        let mut dtemp = 0.0;
        let mut thermal_enabled = false;
        for parameter in &instance.parameters {
            match parameter.name.as_str() {
                "m" => multiplier = scalar(parameter)?,
                "temp" => temperature = Some(scalar(parameter)?),
                "dtemp" => dtemp = scalar(parameter)?,
                "thermal" if parameter.kind == ParameterKind::Flag => {
                    thermal_enabled = true;
                }
                other => {
                    return Err(SpiceError::Unsupported {
                        feature: format!(
                            "{}: unsupported VDMOS instance setter {other}",
                            instance.name
                        ),
                        location: Some(parameter.location.clone()),
                    });
                }
            }
        }
        if !multiplier.is_finite() || multiplier <= 0.0 {
            return Err(invalid(&instance.name, "nonpositive VDMOS multiplier"));
        }
        if thermal_enabled && !resolved.card().parameters.iter().any(|p| p.name == "rthjc") {
            return Err(invalid(
                &instance.name,
                "thermal operation needs explicitly given RTHJC",
            ));
        }
        if temperature.is_some() && dtemp != 0.0 {
            return Err(invalid(
                &instance.name,
                "dtemp with explicit temp is ambiguous",
            ));
        }
        let series = shared.parameters().series;
        if series.rd == 0.0 && series.quasi_saturation.is_some() {
            return Err(SpiceError::not_yet_ported(
                format!(
                    "{}: zero-ohm drain with bias-dependent quasi-saturation resistance",
                    instance.name
                ),
                "src/spicelib/devices/vdmos/vdmosload.c",
            ));
        }
        let device = nodes.transaction(|nodes| {
            let mut terminals = [NodeId::GROUND; 3];
            for (target, name) in terminals.iter_mut().zip(&instance.nodes) {
                *target = nodes.intern(name)?;
            }
            let thermal = if thermal_enabled {
                Some([
                    nodes.intern(&instance.nodes[3])?,
                    nodes.intern(&instance.nodes[4])?,
                ])
            } else {
                None
            };
            let mut node = |external: NodeId, suffix: &str, used: bool| -> SpiceResult<NodeId> {
                if !used {
                    return Ok(external);
                }
                let name = format!("{}#{suffix}", instance.name);
                if nodes.get(&name).is_some() {
                    return Err(SpiceError::circuit(format!(
                        "VDMOS internal node name collision: {name}"
                    )));
                }
                let inner = nodes.intern(&name)?;
                nodes.set_kind(inner, NodeKind::Internal)?;
                Ok(inner)
            };
            let inner = [
                node(terminals[0], "drain", series.rd > 0.0)?,
                node(terminals[1], "gate", series.rg > 0.0)?,
                node(terminals[2], "source", series.rs > 0.0)?,
            ];
            let diode_prime = node(terminals[2], "body_diode", series.rb > 0.0)?;
            let diode = shared.parameters().diode;
            let qp = if diode.vp > 0.0 && diode.tt > 0.0 {
                Some(node(NodeId::GROUND, "qp", true)?)
            } else {
                None
            };
            Ok(Self {
                name: instance.name.clone(),
                model_name: resolved.card().name.clone(),
                terminals,
                inner,
                diode_prime,
                qp,
                thermal,
                model: shared,
                multiplier,
                temperature,
                dtemp,
            })
        })?;
        Ok(Box::new(device))
    }

    fn ambient(&self, celsius: Real) -> Real {
        self.temperature.unwrap_or(celsius + self.dtemp)
    }

    fn bias_temperature(&self, voltage: &impl Fn(NodeId) -> Real, celsius: Real) -> Real {
        self.thermal
            .map_or(self.ambient(celsius), |thermal| voltage(thermal[0]))
            + 273.15
    }

    fn evaluation(
        &self,
        voltage: &impl Fn(NodeId) -> Real,
        celsius: Real,
    ) -> SpiceResult<Evaluation> {
        let [d, g, s] = self.inner.map(voltage);
        let [d_external, _, s_external] = self.terminals.map(voltage);
        let diode = voltage(self.diode_prime) - d_external;
        self.evaluate_bias(Bias {
            vgs: g - s,
            vds: d - s,
            vds_external: d_external - s_external,
            vdiode: diode,
            recovery_node: self.qp.map_or(0.0, |qp| {
                let polarity = self.model.parameters().polarity.sign();
                polarity * voltage(qp)
            }),
            temperature: self.bias_temperature(voltage, celsius),
        })
    }

    fn evaluate_bias(&self, bias: Bias) -> SpiceResult<Evaluation> {
        let device = self
            .model
            .bind(Instance {
                multiplier: self.multiplier,
                self_heating: self.thermal.is_some(),
            })
            .map_err(|error| invalid(&self.name, error))?;
        device
            .evaluate(bias)
            .map_err(|error| invalid(&self.name, format!("{error} at {bias:?}")))
    }

    fn diode_critical(&self, kelvin: Real) -> Real {
        // C's vdmostemp.c DIOtVcrit: the temperature-adjusted saturation
        // current is already part of the prepared diode's constitutive law.
        let model = self.model.parameters();
        let d = model.diode;
        let vt = 8.617_333_262_145e-5 * kelvin;
        let ratio = kelvin / model.tnom;
        let saturation = self.multiplier
            * d.is
            * (((ratio - 1.0) * d.eg / (d.n * vt) + (d.xti / d.n) * ratio.ln()).exp());
        critical_voltage(d.n * vt, saturation)
    }

    /// Native C `vdmosload.c` heat sources: channel and body diode, plus
    /// source/drain/body ohmic losses. The gate resistor and RDS shunt do not
    /// feed C's junction heat source. These derivatives are used by both the
    /// DC/transient Newton residual and the physical small-signal Jacobian.
    fn thermal_power(
        &self,
        evaluation: &Evaluation,
        voltage: &impl Fn(NodeId) -> Real,
    ) -> ThermalPower {
        let [de, _, se] = self.terminals;
        let [d, g, s] = self.inner;
        let dp = self.diode_prime;
        let heat = evaluation.heat;
        let rd = evaluation
            .series
            .drain
            .ohmic_power(voltage(de) - voltage(d));
        let rs = evaluation
            .series
            .source
            .ohmic_power(voltage(se) - voltage(s));
        let rb = evaluation
            .series
            .body
            .ohmic_power(voltage(se) - voltage(dp));
        ThermalPower {
            value: heat.intrinsic + rd.value + rs.value + rb.value,
            d_temperature: heat.d_temperature
                + rd.d_temperature
                + rs.d_temperature
                + rb.d_temperature,
            d_electrical: [
                (g, heat.d_vgs),
                (d, heat.d_vds),
                (s, -heat.d_vgs - heat.d_vds),
                (dp, heat.d_vdiode),
                (de, -heat.d_vdiode),
                (de, rd.d_voltage_drop + rd.d_vds_external),
                (d, -rd.d_voltage_drop),
                (se, -rd.d_vds_external),
                (se, rs.d_voltage_drop),
                (s, -rs.d_voltage_drop),
                (se, rb.d_voltage_drop),
                (dp, -rb.d_voltage_drop),
            ],
        }
    }

    /// Junction/case heat balance. The source kernel provides physical power
    /// and its derivatives; the circuit owns the temperature nodes and
    /// accepted/trial thermal-capacitor history.
    fn thermal_charge_slot(&self) -> usize {
        if self.qp.is_some() {
            TEMPERATURE_AFTER_QP
        } else {
            QP_CHARGE
        }
    }

    /// C `VDMOSdelTemp` is a distinct limiting-history state; it is not the
    /// heat-storage charge/derivative slot that transient integration owns.
    fn thermal_limit_slot(&self) -> usize {
        self.thermal_charge_slot() + 2
    }

    /// VDMOS's C `VDMOSvgs`/`VDMOSvds` limiting histories follow all charge,
    /// Qp and thermal slots; published charge/history offsets never change.
    fn fet_limit_slots(&self) -> [usize; 2] {
        let start = if self.thermal.is_some() {
            self.thermal_limit_slot() + 1
        } else if self.qp.is_some() {
            TEMPERATURE_AFTER_QP
        } else {
            QP_CHARGE
        };
        [start, start + 1]
    }

    /// The physically consistent derivative of the DC electrothermal MNA
    /// residual, including temperature-dependent electrical admittances.
    /// This is intentionally the derivative of `stamp_thermal`: ngspice 47
    /// `vdmosacld.c` reverses signs on its thermal heat-source derivatives
    /// versus `vdmosload.c`, contradicting its own DC finite differences.
    fn assemble_thermal_signal(
        &self,
        context: &mut LinearContext<'_>,
        bias: &Vector,
        evaluation: &Evaluation,
        thermal: [NodeId; 2],
    ) -> SpiceResult<()> {
        let voltage = |node| {
            context
                .unknowns
                .node_row(node)
                .and_then(|row| bias.get(row))
                .unwrap_or(0.0)
        };
        let [tj, tc] = thermal;
        let [de, _, se] = self.terminals;
        let [d, _, s] = self.inner;
        let dp = self.diode_prime;
        let power = self.thermal_power(evaluation, &voltage);
        let rd_drop = voltage(de) - voltage(d);
        let rs_drop = voltage(se) - voltage(s);
        let rb_drop = voltage(se) - voltage(dp);
        let jr = context
            .unknowns
            .node_row(tj)
            .ok_or_else(|| invalid(&self.name, "missing junction MNA row"))?;
        let cr = context
            .unknowns
            .node_row(tc)
            .ok_or_else(|| invalid(&self.name, "missing case MNA row"))?;
        let heat = evaluation.heat;
        let series = evaluation.series;
        stamp_ac_branch(
            context,
            d,
            s,
            [
                (tj, evaluation.channel.d_temperature),
                (NodeId::GROUND, 0.0),
                (NodeId::GROUND, 0.0),
            ],
        )?;
        stamp_ac_branch(
            context,
            dp,
            de,
            [
                (tj, evaluation.diode.current.d_temperature),
                (NodeId::GROUND, 0.0),
                (NodeId::GROUND, 0.0),
            ],
        )?;
        if de != d {
            stamp_ac_branch(
                context,
                de,
                d,
                [
                    (tj, series.drain.d_temperature * rd_drop),
                    (NodeId::GROUND, 0.0),
                    (NodeId::GROUND, 0.0),
                ],
            )?;
        }
        if se != s {
            stamp_ac_branch(
                context,
                se,
                s,
                [
                    (tj, series.source.d_temperature * rs_drop),
                    (NodeId::GROUND, 0.0),
                    (NodeId::GROUND, 0.0),
                ],
            )?;
        }
        if se != dp {
            stamp_ac_branch(
                context,
                se,
                dp,
                [
                    (tj, series.body.d_temperature * rb_drop),
                    (NodeId::GROUND, 0.0),
                    (NodeId::GROUND, 0.0),
                ],
            )?;
        }
        for (node, d_power) in power.d_electrical {
            if let Some(column) = context.unknowns.node_row(node) {
                context.system.a.add(jr, column, -d_power)?;
            }
        }
        let gjc = heat.junction_to_case;
        let gca = heat.case_to_ambient;
        context.system.a.add(jr, jr, gjc - power.d_temperature)?;
        context.system.a.add(jr, cr, -gjc)?;
        context.system.a.add(cr, jr, -gjc)?;
        context.system.a.add(cr, cr, gjc + gca)?;
        context.system.e.add(jr, jr, heat.junction_capacitance)?;
        let d_charge_dt =
            evaluation.diode.depletion.d_temperature + evaluation.diode.diffusion.d_temperature;
        for (node, sign) in [(dp, 1.0), (de, -1.0)] {
            if let Some(row) = context.unknowns.node_row(node) {
                context.system.e.add(row, jr, sign * d_charge_dt)?;
            }
        }
        if let Some(qp) = self.qp {
            let recovery = evaluation
                .diode
                .recovery
                .ok_or_else(|| invalid(&self.name, "Qp without recovery law"))?;
            let qr = context
                .unknowns
                .node_row(qp)
                .ok_or_else(|| invalid(&self.name, "missing Qp row"))?;
            let polarity = self.model.parameters().polarity.sign();
            context
                .system
                .a
                .add(qr, jr, polarity * recovery.d_temperature)?;
            context
                .system
                .e
                .add(qr, jr, polarity * recovery.stored_charge.d_temperature)?;
        }
        Ok(())
    }

    fn stamp_thermal(
        &self,
        context: &mut StampContext<'_>,
        evaluation: &Evaluation,
        temperature_nodes: [NodeId; 2],
        physical_diode_bias: Real,
    ) -> SpiceResult<()> {
        let [tj, tc] = temperature_nodes;
        let t = context.node_voltage(tj);
        let case = context.node_voltage(tc);
        let power = self.thermal_power(evaluation, &|node| context.node_voltage(node));
        let heat = evaluation.heat;
        let initial =
            context.states.phase() == IterationPhase::Predict && context.states.initial_transient();
        let comp = charge_companion(
            ChargeBranch::new(
                self.thermal_charge_slot(),
                heat.junction_capacitance * t,
                heat.junction_capacitance,
            ),
            &mut context.states,
            context.integration,
            initial,
        )?;
        let gjc = heat.junction_to_case;
        let gca = heat.case_to_ambient;
        let diode_limiting_delta = context.node_voltage(self.diode_prime)
            - context.node_voltage(self.terminals[0])
            - physical_diode_bias;
        let rj = gjc * (t - case) - power.value - heat.d_vdiode * diode_limiting_delta
            + comp.map_or(0.0, |charge| charge.derivative);
        stamp_row(
            context,
            tj,
            rj,
            [
                (
                    tj,
                    gjc - power.d_temperature + comp.map_or(0.0, |charge| charge.conductance),
                ),
                (tc, -gjc),
            ],
        )?;
        for (node, d_power) in power.d_electrical {
            context.stamp(tj, node, -d_power)?;
            context.stamp_rhs(tj, -d_power * context.node_voltage(node))?;
        }
        // The ideal C `cktTemp` ambient source is eliminated algebraically;
        // temperature is represented in °C at both public circuit nodes.
        let rc = gjc * (case - t) + gca * (case - self.ambient(context.temperature));
        stamp_row(context, tc, rc, [(tj, -gjc), (tc, gjc + gca)])
    }
}

/// Two-terminal current flowing from `positive` to `negative`, with a
/// Jacobian that is the exact derivative of the supplied current equation.
/// Three control voltages are enough for the VDMOS channel. No heap work.
fn stamp_branch(
    context: &mut StampContext<'_>,
    positive: NodeId,
    negative: NodeId,
    current: Real,
    derivatives: [(NodeId, Real); 3],
) -> SpiceResult<()> {
    let offset = derivatives.iter().fold(current, |sum, (node, derivative)| {
        derivative.mul_add(-context.node_voltage(*node), sum)
    });
    context.stamp_rhs(positive, -offset)?;
    context.stamp_rhs(negative, offset)?;
    for (node, derivative) in derivatives {
        context.stamp(positive, node, derivative)?;
        context.stamp(negative, node, -derivative)?;
    }
    Ok(())
}

/// A nonlinear MNA residual owned by one (possibly thermal) node.
fn stamp_row<const N: usize>(
    context: &mut StampContext<'_>,
    row: NodeId,
    residual: Real,
    derivatives: [(NodeId, Real); N],
) -> SpiceResult<()> {
    let offset = derivatives
        .iter()
        .fold(residual, |sum, (node, derivative)| {
            derivative.mul_add(-context.node_voltage(*node), sum)
        });
    context.stamp_rhs(row, -offset)?;
    for (column, derivative) in derivatives {
        context.stamp(row, column, derivative)?;
    }
    Ok(())
}

/// Thermal voltage is Celsius; its derivative couples an electrical branch
/// into the temperature node without creating a numerical thermal shunt.
fn stamp_temperature_control(
    context: &mut StampContext<'_>,
    thermal: NodeId,
    positive: NodeId,
    negative: NodeId,
    derivative: Real,
) -> SpiceResult<()> {
    let eq_offset = derivative * context.node_voltage(thermal);
    context.stamp(positive, thermal, derivative)?;
    context.stamp(negative, thermal, -derivative)?;
    context.stamp_rhs(positive, eq_offset)?;
    context.stamp_rhs(negative, -eq_offset)
}

fn resistor(
    context: &mut StampContext<'_>,
    positive: NodeId,
    negative: NodeId,
    conductance: Real,
) -> SpiceResult<()> {
    stamp_branch(
        context,
        positive,
        negative,
        conductance * (context.node_voltage(positive) - context.node_voltage(negative)),
        [
            (positive, conductance),
            (negative, -conductance),
            (NodeId::GROUND, 0.0),
        ],
    )
}

/// Bias-linearized current Jacobian, without the DC equivalent RHS.
fn stamp_ac_branch(
    context: &mut LinearContext<'_>,
    positive: NodeId,
    negative: NodeId,
    derivatives: [(NodeId, Real); 3],
) -> SpiceResult<()> {
    for (node, derivative) in derivatives {
        let Some(column) = context.unknowns.node_row(node) else {
            continue;
        };
        for (row, sign) in [(positive, 1.0), (negative, -1.0)] {
            if let Some(row) = context.unknowns.node_row(row) {
                context.system.a.add(row, column, sign * derivative)?;
            }
        }
    }
    Ok(())
}

impl Device for Vdmos {
    fn name(&self) -> &str {
        &self.name
    }
    fn designator(&self) -> char {
        'm'
    }
    fn model_name(&self) -> Option<&str> {
        Some(&self.model_name)
    }
    fn terminals(&self) -> &[NodeId] {
        &self.terminals
    }
    fn c_device_type(&self) -> crate::devices::setup::CDeviceType {
        crate::devices::setup::CDeviceType::Vdmos
    }
    fn is_nonlinear(&self) -> bool {
        true
    }
    fn state_count(&self) -> usize {
        self.fet_limit_slots()[1] + 1
    }
    fn truncation_slots(&self) -> Vec<usize> {
        let mut slots = vec![QGS, QGD, QDIO];
        if self.qp.is_some() {
            slots.push(QP_CHARGE);
        }
        if self.thermal.is_some() {
            slots.push(self.thermal_charge_slot());
        }
        slots
    }

    /// Source-owned white and flicker noise at the converged physical bias;
    /// MNA noise transfer, ordering and frequency integration stay here.
    fn noise(&self, context: &NoiseContext<'_>) -> SpiceResult<DeviceNoise> {
        let evaluated = self.evaluation(
            &|node| context.voltage(node),
            context.model_context.temperature,
        )?;
        let prepared = self
            .model
            .bind(Instance {
                multiplier: self.multiplier,
                self_heating: self.thermal.is_some(),
            })
            .map_err(|error| invalid(&self.name, error))?;
        let density = prepared
            .noise_density(&evaluated, 1.0)
            .map_err(|error| invalid(&self.name, error))?;
        let [d, _, s] = self.inner;
        let [de, _, se] = self.terminals;
        Ok(DeviceNoise::Sources {
            family: NoiseFamily::Vdmos,
            model: Some(self.model_name.clone()),
            total: true,
            sources: vec![
                NoiseSource::new(
                    "_rd",
                    [d, de],
                    NoiseKind::White {
                        density: density.drain,
                    },
                ),
                NoiseSource::new(
                    "_rs",
                    [s, se],
                    NoiseKind::White {
                        density: density.source,
                    },
                ),
                NoiseSource::new(
                    "_id",
                    [d, s],
                    NoiseKind::White {
                        density: density.channel,
                    },
                ),
                NoiseSource::new(
                    "_1overf",
                    [d, s],
                    NoiseKind::Flicker {
                        numerator: density.flicker,
                        law: FlickerLaw::PerHertz,
                    },
                ),
            ],
        })
    }

    fn observation_operating(
        &self,
        keyword: &str,
        context: &ModelContext,
        voltage: &dyn Fn(NodeId) -> Real,
        _transient: bool,
    ) -> SpiceResult<Option<Real>> {
        let eval = self.evaluation(&voltage, context.temperature)?;
        Ok(Some(match keyword {
            "id" => eval.channel.value.abs(),
            "gm" => eval.channel.d_vgs.abs(),
            "gds" => eval.channel.d_vds.abs(),
            "idio" => eval.diode.current.value,
            "cgs" => eval.qgs.d_voltage,
            "cgd" => eval.qgd.d_voltage,
            "cds" => eval.diode.depletion.d_voltage,
            _ => return Ok(None),
        }))
    }

    fn assemble_small_signal(
        &self,
        context: &mut LinearContext<'_>,
        bias: &Vector,
    ) -> SpiceResult<()> {
        let voltage = |node: NodeId| {
            context
                .unknowns
                .node_row(node)
                .and_then(|row| bias.get(row))
                .unwrap_or(0.0)
        };
        let result = self.evaluation(&voltage, context.model_context.temperature)?;
        let [d, g, s] = self.inner;
        let [de, ge, se] = self.terminals;
        let dp = self.diode_prime;
        stamp_ac_branch(
            context,
            d,
            s,
            [
                (g, result.channel.d_vgs),
                (d, result.channel.d_vds),
                (s, -result.channel.d_vgs - result.channel.d_vds),
            ],
        )?;
        let diode = result.diode.current.d_voltage + context.model_context.gmin;
        context.nodal([dp, de], diode, false)?;
        context.nodal(
            [dp, de],
            result.diode.depletion.d_voltage + result.diode.diffusion.d_voltage,
            true,
        )?;
        context.nodal([g, s], result.qgs.d_voltage, true)?;
        context.nodal([g, d], result.qgd.d_voltage, true)?;
        if let Some(qpnode) = self.qp {
            let recovery = result
                .diode
                .recovery
                .ok_or_else(|| invalid(&self.name, "missing Qp kernel"))?;
            let qp_row = context
                .unknowns
                .node_row(qpnode)
                .ok_or_else(|| invalid(&self.name, "missing Qp MNA row"))?;
            context.system.a.add(qp_row, qp_row, recovery.d_node)?;
            context
                .system
                .e
                .add(qp_row, qp_row, recovery.stored_charge.d_voltage)?;
            let polarity = self.model.parameters().polarity.sign();
            let coupling = polarity * recovery.d_voltage;
            for (node, value) in [(dp, coupling), (de, -coupling)] {
                if let Some(column) = context.unknowns.node_row(node) {
                    context.system.a.add(qp_row, column, value)?;
                }
            }
            // Source-to-drain body-diode charge-rate coupling; C VDMOSacLoad
            // adds +(1-VP)*s at the physical diode-positive/Qp matrix entry.
            let gain =
                polarity * recovery.diode_charge_rate_gain * recovery.stored_charge.d_voltage;
            for (node, sign) in [(dp, 1.0), (de, -1.0)] {
                if let Some(row) = context.unknowns.node_row(node) {
                    context.system.e.add(row, qp_row, sign * gain)?;
                }
            }
        }
        let series = result.series;
        if de != d {
            let conductance = series
                .drain
                .value
                .ok_or_else(|| invalid(&self.name, "missing drain resistance"))?;
            let drop = voltage(de) - voltage(d);
            let feedback = series.drain.d_vds_external * drop;
            stamp_ac_branch(
                context,
                de,
                d,
                [
                    (de, conductance + feedback),
                    (d, -conductance),
                    (se, -feedback),
                ],
            )?;
        }
        for (external, internal, conductance) in [
            (ge, g, series.gate.value),
            (se, s, series.source.value),
            (se, dp, series.body.value),
        ] {
            if external != internal {
                context.nodal(
                    [external, internal],
                    conductance.ok_or_else(|| invalid(&self.name, "missing series resistance"))?,
                    false,
                )?;
            }
        }
        context.nodal([de, se], series.shunt, false)?;
        if let Some(thermal) = self.thermal {
            self.assemble_thermal_signal(context, bias, &result, thermal)?;
        }
        Ok(())
    }

    fn stamp(&self, context: &mut StampContext<'_>) -> SpiceResult<()> {
        if context.mode.is_ac() {
            return Err(SpiceError::circuit(
                "VDMOS AC loads use small-signal assembly",
            ));
        }
        if context.mode.is_transient() && context.integration.is_none() {
            return Err(SpiceError::circuit(
                "VDMOS transient charge needs integration coefficients",
            ));
        }
        let [d, g, s] = self.inner;
        let [de, ge, se] = self.terminals;
        let dp = self.diode_prime;
        let mut limiter = Limiter::new(&context.states);
        let temperature_limit = self.thermal.map(|[junction, _]| {
            let raw = context.node_voltage(junction);
            match limiter.mode() {
                // C VDMOSload's MODEINITJCT starts delTemp at 0°C.
                Linearization::Initial => 0.0,
                Linearization::InitialTransient => context
                    .states
                    .accepted(1, self.thermal_limit_slot())
                    .unwrap_or(raw),
                _ => {
                    let previous = limiter.previous(&context.states, self.thermal_limit_slot());
                    limiter.temperature(raw, previous, THERMAL_LOG_LIMIT)
                }
            }
        });
        let kelvin =
            temperature_limit.unwrap_or_else(|| self.ambient(context.temperature)) + 273.15;
        let raw_vdiode = context.node_voltage(dp) - context.node_voltage(de);
        let polarity = if self.model.parameters().polarity == Polarity::N {
            1.0
        } else {
            -1.0
        };
        let raw_normal = polarity * raw_vdiode;
        let normal = match limiter.mode() {
            Linearization::Initial => 0.0,
            Linearization::InitialTransient => context
                .states
                .accepted(1, DIODE_VOLTAGE)
                .unwrap_or(raw_normal),
            _ => limiter.pn_junction(
                raw_normal,
                limiter.previous(&context.states, DIODE_VOLTAGE),
                8.617_333_262_145e-5 * kelvin * self.model.parameters().diode.n,
                self.diode_critical(kelvin),
            ),
        };
        let vdiode = polarity * normal;
        let raw_vgs = context.node_voltage(g) - context.node_voltage(s);
        let raw_vds = context.node_voltage(d) - context.node_voltage(s);
        let raw_normal_vgs = polarity * raw_vgs;
        let raw_normal_vds = polarity * raw_vds;
        let channel = self.model.parameters().channel;
        let normalized_vto =
            polarity * (channel.vth + channel.tcvth * (kelvin - self.model.parameters().tnom));
        let [vgs_slot, vds_slot] = self.fet_limit_slots();
        let (normal_vgs, normal_vds) = match limiter.mode() {
            Linearization::Initial => (normalized_vto + 0.1, 0.0),
            Linearization::InitialTransient => (
                context
                    .states
                    .accepted(1, vgs_slot)
                    .unwrap_or(raw_normal_vgs),
                context
                    .states
                    .accepted(1, vds_slot)
                    .unwrap_or(raw_normal_vds),
            ),
            _ => {
                let previous_vgs = limiter.previous(&context.states, vgs_slot);
                let previous_vds = limiter.previous(&context.states, vds_slot);
                match (previous_vgs, previous_vds) {
                    (Some(previous_vgs), Some(previous_vds)) if previous_vds >= 0.0 => {
                        let limited_vgs =
                            limiter.fet_gate(raw_normal_vgs, Some(previous_vgs), normalized_vto);
                        let limited_vds = limiter.drain_source(
                            limited_vgs - (raw_normal_vgs - raw_normal_vds),
                            Some(previous_vds),
                        );
                        (limited_vgs, limited_vds)
                    }
                    (Some(previous_vgs), Some(previous_vds)) => {
                        let limited_vgd = limiter.fet_gate(
                            raw_normal_vgs - raw_normal_vds,
                            Some(previous_vgs - previous_vds),
                            normalized_vto,
                        );
                        let limited_vds = -limiter
                            .drain_source(-(raw_normal_vgs - limited_vgd), Some(-previous_vds));
                        (limited_vgd + limited_vds, limited_vds)
                    }
                    _ => (raw_normal_vgs, raw_normal_vds),
                }
            }
        };
        let vgs = polarity * normal_vgs;
        let vds = polarity * normal_vds;
        let result = self.evaluate_bias(Bias {
            vgs,
            vds,
            vds_external: context.node_voltage(de) - context.node_voltage(se),
            vdiode,
            recovery_node: self
                .qp
                .map_or(0.0, |qp| polarity * context.node_voltage(qp)),
            temperature: kelvin,
        })?;
        let channel = result.channel;
        let drain_drop = context.node_voltage(de) - context.node_voltage(d);
        let initial =
            context.states.phase() == IterationPhase::Predict && context.states.initial_transient();
        let qs = ChargeBranch::new(QGS, result.qgs.value, result.qgs.d_voltage);
        let qd = ChargeBranch::new(QGD, result.qgd.value, result.qgd.d_voltage);
        let qdio = ChargeBranch::new(
            QDIO,
            result.diode.depletion.value + result.diode.diffusion.value,
            result.diode.depletion.d_voltage + result.diode.diffusion.d_voltage,
        );
        let gs = charge_companion(qs, &mut context.states, context.integration, initial)?;
        let gd = charge_companion(qd, &mut context.states, context.integration, initial)?;
        let diode_comp = charge_companion(qdio, &mut context.states, context.integration, initial)?;
        if let Some(qpnode) = self.qp {
            let recovery = result
                .diode
                .recovery
                .ok_or_else(|| invalid(&self.name, "Qp node without recovery law"))?;
            // C's internal Qp voltage is in physical polarity while the kernel
            // exposes a polarity-normalized charge. Multiplying the stored
            // charge and scalar residual by polarity gives one MNA voltage
            // variable independent of N/P conventions.
            let qpcharge = ChargeBranch::new(
                QP_CHARGE,
                polarity * recovery.stored_charge.value,
                recovery.stored_charge.d_voltage,
            );
            let qpcomp =
                charge_companion(qpcharge, &mut context.states, context.integration, initial)?;
            let qpvoltage = context.node_voltage(qpnode);
            let coupling = polarity * recovery.d_voltage;
            let diag = recovery.d_node + qpcomp.map_or(0.0, |charge| charge.conductance);
            let qpcurrent = polarity * recovery.residual
                + qpcomp.map_or(0.0, |charge| charge.derivative)
                + coupling * (raw_vdiode - vdiode);
            let offset = qpcurrent - diag * qpvoltage - coupling * context.node_voltage(dp)
                + coupling * context.node_voltage(de);
            context.stamp_rhs(qpnode, -offset)?;
            context.stamp(qpnode, qpnode, diag)?;
            context.stamp(qpnode, dp, coupling)?;
            context.stamp(qpnode, de, -coupling)?;
            if let Some([tj, _]) = self.thermal {
                // Qp relaxation and stored-charge rate both depend on Tj;
                // the same analytic kernel derivatives also feed AC.
                let ag0 = context.integration.map_or(0.0, |coeff| coeff.ag()[0]);
                let d_residual_dt = polarity
                    * (recovery.d_temperature + ag0 * recovery.stored_charge.d_temperature);
                context.stamp(qpnode, tj, d_residual_dt)?;
                context.stamp_rhs(qpnode, d_residual_dt * context.node_voltage(tj))?;
            }
            // Current from body diode's positive prime node to external drain.
            // The corrected kernel gain is normalized to physical direction.
            if let Some(charge) = qpcomp {
                let gain = polarity * recovery.diode_charge_rate_gain;
                stamp_branch(
                    context,
                    dp,
                    de,
                    gain * charge.derivative,
                    [
                        (qpnode, gain * charge.conductance),
                        (NodeId::GROUND, 0.0),
                        (NodeId::GROUND, 0.0),
                    ],
                )?;
                if let (Some([tj, _]), Some(coefficients)) = (self.thermal, context.integration) {
                    let d_charge_dt = polarity * recovery.stored_charge.d_temperature;
                    let d_gain_dt = polarity * recovery.d_gain_d_temperature;
                    let cross =
                        d_gain_dt * charge.derivative + gain * coefficients.ag()[0] * d_charge_dt;
                    stamp_temperature_control(context, tj, dp, de, cross)?;
                }
            }
        }
        if let Some(c) = gs {
            stamp_branch(
                context,
                g,
                s,
                c.derivative,
                [
                    (g, c.conductance),
                    (s, -c.conductance),
                    (NodeId::GROUND, 0.0),
                ],
            )?;
        }
        if let Some(c) = gd {
            stamp_branch(
                context,
                g,
                d,
                c.derivative,
                [
                    (g, c.conductance),
                    (d, -c.conductance),
                    (NodeId::GROUND, 0.0),
                ],
            )?;
        }
        // C `cdreq` linearizes about the limited VGS/VDS, not the raw
        // MNA values. Keep the correct equivalent-current RHS while the
        // Newton device limiter walks toward its physical operating point.
        let channel_equivalent =
            channel.value + channel.d_vgs * (raw_vgs - vgs) + channel.d_vds * (raw_vds - vds);
        stamp_branch(
            context,
            d,
            s,
            channel_equivalent,
            [
                (g, channel.d_vgs),
                (d, channel.d_vds),
                (s, -channel.d_vgs - channel.d_vds),
            ],
        )?;
        if let Some([tj, _]) = self.thermal {
            stamp_temperature_control(context, tj, d, s, channel.d_temperature)?;
        }
        let gd = result.diode.current.d_voltage
            + context.gmin
            + diode_comp.map_or(0.0, |charge| charge.conductance);
        // The equivalent current is linearized at the limited diode voltage,
        // including its independent charge companion and actual MNA nodes.
        let idio = result.diode.current.value
            + context.gmin * vdiode
            + diode_comp.map_or(0.0, |charge| charge.derivative)
            + gd * (raw_vdiode - vdiode);
        stamp_branch(
            context,
            dp,
            de,
            idio,
            [(dp, gd), (de, -gd), (NodeId::GROUND, 0.0)],
        )?;
        if let Some([tj, _]) = self.thermal {
            let dcharge_dt = context.integration.map_or(0.0, |coefficients| {
                coefficients.ag()[0]
                    * (result.diode.depletion.d_temperature + result.diode.diffusion.d_temperature)
            });
            stamp_temperature_control(
                context,
                tj,
                dp,
                de,
                result.diode.current.d_temperature + dcharge_dt,
            )?;
        }
        let series = result.series;
        if de != d {
            let gd = series
                .drain
                .value
                .ok_or_else(|| invalid(&self.name, "zero conductance on a bound drain branch"))?;
            let feedback = series.drain.d_vds_external * drain_drop;
            stamp_branch(
                context,
                de,
                d,
                gd * drain_drop,
                [(de, gd + feedback), (d, -gd), (se, -feedback)],
            )?;
            if let Some([tj, _]) = self.thermal {
                stamp_temperature_control(
                    context,
                    tj,
                    de,
                    d,
                    series.drain.d_temperature * drain_drop,
                )?;
            }
            // In a quasistatic voltage-dependent branch, the terminal-voltage
            // derivative is already included in the branch Jacobian above.
        }
        for (external, internal, conductance, dgt) in [
            (ge, g, series.gate.value, 0.0),
            (se, s, series.source.value, series.source.d_temperature),
            (se, dp, series.body.value, series.body.d_temperature),
        ] {
            if external != internal {
                if let Some([tj, _]) = self.thermal {
                    let drop = context.node_voltage(external) - context.node_voltage(internal);
                    stamp_temperature_control(context, tj, external, internal, dgt * drop)?;
                }
                resistor(
                    context,
                    external,
                    internal,
                    conductance
                        .ok_or_else(|| invalid(&self.name, "missing finite series resistance"))?,
                )?;
            }
        }
        resistor(context, de, se, series.shunt)?;
        if let Some(thermal) = self.thermal {
            self.stamp_thermal(context, &result, thermal, vdiode)?;
        }
        if let Some(temperature) = temperature_limit {
            limiter.finish(
                &mut context.states,
                &[
                    (DIODE_VOLTAGE, normal),
                    (vgs_slot, normal_vgs),
                    (vds_slot, normal_vds),
                    (self.thermal_limit_slot(), temperature),
                ],
            )
        } else {
            limiter.finish(
                &mut context.states,
                &[
                    (DIODE_VOLTAGE, normal),
                    (vgs_slot, normal_vgs),
                    (vds_slot, normal_vds),
                ],
            )
        }
    }
}
