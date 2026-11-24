//! Research-only circuit source emitter, using the production device kernels.
use super::*;
use std::{any::Any, fmt::Write};

mod generated;

pub(super) use generated::{dispatch, matches};

fn typed<T: Device>(circuit: &Circuit, index: usize) -> SpiceResult<&T> {
    (&*circuit.devices[index] as &dyn Any)
        .downcast_ref::<T>()
        .ok_or_else(|| SpiceError::circuit("generated device type mismatch"))
}

pub(super) fn emit(circuit: &Circuit, case: usize, model: &ModelContext) -> SpiceResult<String> {
    let mut out = String::new();
    let result = (|| -> Result<(), Box<dyn std::error::Error>> {
        writeln!(out, "pub(super) fn check_{case}(circuit: &Circuit, model: &ModelContext) -> SpiceResult<bool> {{")?;
        writeln!(out, "    Ok(model.temperature.to_bits() == {} && model.nominal_temperature.to_bits() == {} && circuit.specialization_inventory() == vec![", model.temperature.to_bits(), model.nominal_temperature.to_bits())?;
        for (index, ty, name) in circuit.specialization_inventory() {
            writeln!(out, "        ({index}, {ty:?}, {name:?}.to_owned()),")?;
        }
        writeln!(out, "    ])\n}}")?;
        writeln!(out, "pub(super) fn case_{case}<const BAKE: bool>(circuit: &Circuit, request: &LoadRequest<'_>, matrix: &mut SparseMatrix, rhs: &mut Vector, trial: &mut TrialState, mutual: &[Vec<MutualTerm>], voltage_source_value: &mut Real) -> SpiceResult<()> {{")?;
        for (position, index) in circuit.load_order().enumerate() {
            let device = &*circuit.devices[index];
            let ty = device.concrete_type().replace("ngspice_rs::", "crate::").replace("bsim4::device::", "bsim4::");
            writeln!(out, "    circuit.stamp_one({position}, {index}, request, matrix, rhs, trial, mutual, voltage_source_value, |context| {{")?;
            writeln!(out, "        let device = typed::<{ty}>(circuit, {index})?;")?;
            if let Some(bsim) = (device as &dyn Any).downcast_ref::<crate::devices::bsim4::Bsim4>() {
                writeln!(out, "        if BAKE {{")?;
                out.push_str(&bsim.emit_model(model)?);
                writeln!(out, "            device.stamp_with_model(context, Some((&MODEL, &SIZE)))\n        }} else {{ device.stamp(context) }}")?;
            } else if let Some(resistor) = (device as &dyn Any).downcast_ref::<crate::devices::Resistor>() {
                let nodes = resistor.terminals();
                writeln!(out, "        if BAKE {{ crate::devices::rlc::stamp_conductance([NodeId::new({}), NodeId::new({})], f64::from_bits({}), context) }} else {{ device.stamp(context) }}", nodes[0].index(), nodes[1].index(), resistor.conductance().to_bits())?;
            } else {
                writeln!(out, "        device.stamp(context)")?;
            }
            writeln!(out, "    }})?;")?;
        }
        writeln!(out, "    Ok(())\n}}")?;
        Ok(())
    })();
    result.map_err(|error| SpiceError::circuit(format!("emit specialization: {error}")))?;
    Ok(out)
}
