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
use crate::devices::traits::{Device, StampContext};
use crate::maths::Vector;
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
    model: Arc<PreparedModel>,
    multiplier: Real,
    temperature: Option<Real>,
    dtemp: Real,
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
        for parameter in &instance.parameters {
            match parameter.name.as_str() {
                "m" => multiplier = scalar(parameter)?,
                "temp" => temperature = Some(scalar(parameter)?),
                "dtemp" => dtemp = scalar(parameter)?,
                // Self-heating needs a different topological network, never an
                // unconnected thermal source or an ignored instance flag.
                "thermal" => {
                    return Err(SpiceError::not_yet_ported(
                        format!("{}: electrothermal network", instance.name),
                        "src/spicelib/devices/vdmos/vdmosset.c",
                    ));
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
            Ok(Self {
                name: instance.name.clone(),
                model_name: resolved.card().name.clone(),
                terminals,
                inner,
                diode_prime,
                model: shared,
                multiplier,
                temperature,
                dtemp,
            })
        })?;
        Ok(Box::new(device))
    }

    fn kelvin(&self, celsius: Real) -> Real {
        self.temperature.unwrap_or(celsius + self.dtemp) + 273.15
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
            recovery_node: 0.0,
            temperature: self.kelvin(celsius),
        })
    }

    fn evaluate_bias(&self, bias: Bias) -> SpiceResult<Evaluation> {
        let device = self
            .model
            .bind(Instance {
                multiplier: self.multiplier,
                self_heating: false,
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
        1
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
        if self.model.parameters().diode.vp > 0.0 && self.model.parameters().diode.tt > 0.0 {
            return Err(SpiceError::not_yet_ported(
                format!(
                    "{}: reverse-recovery Qp small-signal hidden state",
                    self.name
                ),
                "src/spicelib/devices/vdmos/vdmosacld.c",
            ));
        }
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
        Ok(())
    }

    fn stamp(&self, context: &mut StampContext<'_>) -> SpiceResult<()> {
        if context.mode.is_ac() {
            return Err(SpiceError::circuit(
                "VDMOS AC loads use small-signal assembly",
            ));
        }
        if context.mode.is_transient() {
            return Err(SpiceError::not_yet_ported(
                format!(
                    "{}: VDMOS charge and reverse-recovery companion integration",
                    self.name
                ),
                "src/spicelib/devices/vdmos/vdmosload.c",
            ));
        }
        let [d, g, s] = self.inner;
        let [de, ge, se] = self.terminals;
        let dp = self.diode_prime;
        let kelvin = self.kelvin(context.temperature);
        let raw_vdiode = context.node_voltage(dp) - context.node_voltage(de);
        let polarity = if self.model.parameters().polarity == Polarity::N {
            1.0
        } else {
            -1.0
        };
        let mut limiter = Limiter::new(&context.states);
        let raw_normal = polarity * raw_vdiode;
        let normal = match limiter.mode() {
            Linearization::Initial => 0.0,
            _ => limiter.pn_junction(
                raw_normal,
                limiter.previous(&context.states, 0),
                8.617_333_262_145e-5 * kelvin * self.model.parameters().diode.n,
                self.diode_critical(kelvin),
            ),
        };
        let vdiode = polarity * normal;
        let result = self.evaluate_bias(Bias {
            vgs: context.node_voltage(g) - context.node_voltage(s),
            vds: context.node_voltage(d) - context.node_voltage(s),
            vds_external: context.node_voltage(de) - context.node_voltage(se),
            vdiode,
            recovery_node: 0.0,
            temperature: kelvin,
        })?;
        let channel = result.channel;
        let drain_drop = context.node_voltage(de) - context.node_voltage(d);
        stamp_branch(
            context,
            d,
            s,
            channel.value,
            [
                (g, channel.d_vgs),
                (d, channel.d_vds),
                (s, -channel.d_vgs - channel.d_vds),
            ],
        )?;
        let gd = result.diode.current.d_voltage + context.gmin;
        // The equivalent current is linearized at the limited diode voltage,
        // even though the matrix columns are the actual node coordinates.
        let idio = result.diode.current.value + context.gmin * vdiode + gd * (raw_vdiode - vdiode);
        stamp_branch(
            context,
            dp,
            de,
            idio,
            [(dp, gd), (de, -gd), (NodeId::GROUND, 0.0)],
        )?;
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
            // In a quasistatic voltage-dependent branch, the terminal-voltage
            // derivative is already included in the branch Jacobian above.
        }
        for (external, internal, conductance) in [
            (ge, g, series.gate.value),
            (se, s, series.source.value),
            (se, dp, series.body.value),
        ] {
            if external != internal {
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
        limiter.finish(&mut context.states, &[(0, normal)])
    }
}
