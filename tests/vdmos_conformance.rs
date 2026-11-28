//! Independent ngspice 47 VDMOS decks exercised through the current public
//! producer and generic prepared-device engine. The fixture adapter only
//! separates producer-owned declarations from the surrounding native deck;
//! it does not parse MOS model or instance syntax itself.
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;
use std::sync::Arc;

use model_kernel_vdmos::{Polarity, artifact::artifact, compile_ngspice};
use model_normalize_ngspice::{
    linked_source::RootKind,
    parameter_scope::{
        ModelLibrary,
        topology::{Limits, ModelVariant, Node, VdmosBinding},
    },
    vdmos_card::VdmosPolarity,
};
use ngspice_rs::{
    analysis::{AnalysisPlan, Plot, RunConfig},
    devices::{
        Circuit,
        prepared_model::{AmbientTemperature, InstanceTemperature, PreparedDevice},
    },
    netlist::{Parser, source::parse_deck_text},
    primitives::{Complex, SpiceError, SpiceResult},
};
use sha2::{Digest, Sha256};

#[path = "vdmos/noise.rs"]
mod noise;
#[path = "vdmos/performance.rs"]
mod performance;
#[path = "vdmos/reference.rs"]
mod reference;

struct Fixture {
    config: RunConfig,
    plan: AnalysisPlan,
    circuit: Circuit,
    thermal: Arc<BTreeSet<String>>,
}

/// Test-only projection to ngspice's reported absolute Celsius thermal
/// voltages; the generic prepared engine solves rises above circuit ambient.
struct ReferencePlot {
    plot: Plot,
    thermal: Arc<BTreeSet<String>>,
    ambient: f64,
}

impl ReferencePlot {
    fn point_count(&self) -> usize {
        self.plot.point_count()
    }

    fn value(&self, variable: &str, point: usize) -> Option<Complex> {
        let mut value = self.plot.value(variable, point)?;
        if self.thermal.contains(variable) {
            value.re += self.ambient;
        }
        Some(value)
    }
}

fn fixture_error(error: impl std::fmt::Display) -> SpiceError {
    SpiceError::circuit(format!("VDMOS reference fixture: {error}"))
}

fn compile_artifact(
    variant: &ModelVariant<'_>,
    self_heating: bool,
) -> SpiceResult<Arc<model_kernel_api::Artifact<model_kernel_vdmos::artifact::VdmosKernel>>> {
    let values = &variant.values;
    let polarity = match values.polarity {
        VdmosPolarity::N => Polarity::N,
        VdmosPolarity::P => Polarity::P,
    };
    let mut hash = Sha256::new();
    hash.update(values.name.as_bytes());
    hash.update([u8::from(self_heating), u8::from(polarity == Polarity::P)]);
    for (name, value) in &values.parameters {
        hash.update(name.as_bytes());
        hash.update([0]);
        hash.update(value.to_bits().to_le_bytes());
    }
    let mut fingerprint = [0; 16];
    fingerprint.copy_from_slice(&hash.finalize()[..16]);
    let model =
        compile_ngspice(polarity, values.parameters.iter().copied()).map_err(fixture_error)?;
    artifact(Arc::new(model), self_heating, fingerprint).map_err(fixture_error)
}

fn temperature(binding: &VdmosBinding, celsius: f64) -> InstanceTemperature {
    let instance = binding.instance;
    let ambient = if let Some(delta) = instance.dtemp_kelvin {
        AmbientTemperature::DeltaKelvin(delta)
    } else if let Some(absolute) = instance.temp_celsius {
        AmbientTemperature::AbsoluteKelvin(absolute + 273.15)
    } else {
        AmbientTemperature::Circuit
    };
    InstanceTemperature {
        circuit_celsius: celsius,
        ambient,
    }
}

fn prepare(source: &str, analysis: &str) -> SpiceResult<Fixture> {
    // These are flat, one-physical-line reference declarations. Use the
    // producer's source positions to remove *only* its model and device
    // declarations; the native parser remains responsible for all other cards.
    let cards = source.split(".control").next().unwrap_or(source);
    let library =
        ModelLibrary::parse_source(cards.as_bytes(), RootKind::Deck).map_err(fixture_error)?;
    let topology = library
        .instantiate(Limits::default())
        .map_err(fixture_error)?;
    assert_eq!(topology.frames.len(), 1, "flat reference deck required");
    assert!(
        !topology.mos.is_empty(),
        "reference requires prepared MOS instances"
    );
    let removed: BTreeSet<_> = library
        .models
        .iter()
        .map(|m| m.card.source_line)
        .chain(topology.mos.iter().map(|m| m.location.linked_line))
        .collect();
    let mut native = String::new();
    use std::fmt::Write as _;
    for (index, line) in cards.lines().enumerate() {
        assert!(
            !line.trim_start().starts_with('+'),
            "fixture requires one-line cards"
        );
        if !removed.contains(&(index + 1)) && !line.trim().eq_ignore_ascii_case(".end") {
            writeln!(native, "{line}").unwrap();
        }
    }
    writeln!(native, ".{analysis}\n.end").unwrap();
    let netlist =
        Parser::new().parse_deck(&parse_deck_text(Path::new("vdmos-reference.cir"), &native))?;
    let config = RunConfig::from_netlist(&netlist)?;
    let plan = config.plan_for(&netlist.analyses[0])?;
    let mut circuit = config.circuit(&netlist)?;
    let celsius = config.context().temperature;
    let mut artifacts = BTreeMap::new();
    let mut thermal = BTreeSet::new();
    for mos in &topology.mos {
        let binding = mos.vdmos.as_ref().expect("only VDMOS fixtures");
        assert!(!binding.instance.off && binding.instance.initial == [None; 2]);
        let artifact = match artifacts.entry(binding.artifact_key()) {
            std::collections::btree_map::Entry::Occupied(entry) => entry.into_mut(),
            std::collections::btree_map::Entry::Vacant(entry) => entry.insert(compile_artifact(
                &topology.models[binding.model],
                binding.self_heating,
            )?),
        };
        let mut terminals = Vec::new();
        for (index, node) in binding.terminals.iter().enumerate() {
            let name = match node {
                Node::Global(name) => name.clone(),
                Node::DeviceThermal { terminal, .. } => {
                    format!("{}-thermal-{terminal:?}", mos.name)
                }
                Node::Local { .. } => panic!("flat reference fixture has local node"),
            };
            if index >= 3 {
                thermal.insert(format!("v({name})"));
            }
            terminals.push(circuit.add_node(&name)?);
        }
        PreparedDevice::bind(
            &mut circuit,
            artifact,
            &mos.name,
            &terminals,
            &[binding.instance.multiplier],
            temperature(binding, celsius),
        )?;
    }
    circuit.finalize()?;
    Ok(Fixture {
        config,
        plan,
        circuit,
        thermal: Arc::new(thermal),
    })
}

fn run_plots(source: &str, analysis: &str) -> SpiceResult<Vec<ReferencePlot>> {
    let Fixture {
        config,
        plan,
        mut circuit,
        thermal,
    } = prepare(source, analysis)?;
    let context = config.context();
    let ambient = if analysis.starts_with("op") || analysis.starts_with("tran ") {
        context.temperature
    } else {
        0.
    };
    Ok(plan
        .run_plots(&mut circuit, &context)?
        .into_iter()
        .map(|plot| ReferencePlot {
            plot,
            thermal: Arc::clone(&thermal),
            ambient,
        })
        .collect())
}

fn run_one(source: &str, analysis: &str) -> SpiceResult<ReferencePlot> {
    let mut plots = run_plots(source, analysis)?;
    assert_eq!(plots.len(), 1);
    Ok(plots.remove(0))
}
