//! Shared test-only contract for the two generated BSIM parameter interfaces.
//!
//! Expected access cases and schema sizes are independently specified by each
//! model test suite; the assertions never derive those expectations from the
//! same generated table they are checking.

use std::collections::BTreeSet;
use std::path::Path;

use ngspice_rs::devices::{ModelResolver, ResolvedModel};
use ngspice_rs::netlist::Parser;
use ngspice_rs::netlist::ast::{DeviceInstance, ParameterAssignment, ParameterKind};
use ngspice_rs::netlist::source::parse_deck_text;
use ngspice_rs::primitives::{SpiceError, SpiceResult};

use ngspice_rs::devices::bsim3::{Access, InputType, Inputs, Parameter, Value};

pub(super) struct SchemaContract {
    pub(super) family: &'static str,
    pub(super) level: u8,
    pub(super) model_version: &'static str,
    pub(super) model_table: &'static [Parameter],
    pub(super) instance_table: &'static [Parameter],
    pub(super) access_cases: &'static [(&'static str, Access)],
    pub(super) model_inputs: fn(&str) -> SpiceResult<Inputs>,
    pub(super) instance_inputs: fn(&str) -> SpiceResult<Inputs>,
    pub(super) instantiate: fn(&ResolvedModel<'_>, &DeviceInstance) -> SpiceResult<Inputs>,
}

impl SchemaContract {
    pub(super) fn assert_generated_tables(&self) {
        for (kind, table) in [
            ("model", self.model_table),
            ("instance", self.instance_table),
        ] {
            assert_eq!(
                table
                    .iter()
                    .map(|parameter| parameter.name)
                    .collect::<BTreeSet<_>>()
                    .len(),
                table.len(),
                "{} {kind} repeated keyword",
                self.family,
            );
            for parameter in table {
                let canonical = table
                    .iter()
                    .find(|candidate| candidate.name == parameter.canonical)
                    .unwrap_or_else(|| {
                        panic!(
                            "{} {kind} {} missing canonical",
                            self.family, parameter.name
                        )
                    });
                assert_eq!(
                    canonical.id, parameter.id,
                    "{} {kind} {}",
                    self.family, parameter.name
                );
                assert_eq!(
                    canonical.c_field, parameter.c_field,
                    "{} {kind} {}",
                    self.family, parameter.name
                );
                assert!(
                    parameter.c_id.starts_with(self.family),
                    "{} {kind} {} missing C ID prefix",
                    self.family,
                    parameter.name
                );
                if parameter.access != Access::Output
                    && parameter.input_type != InputType::RealVector
                {
                    assert!(
                        parameter.c_field.is_some(),
                        "{} {kind} {} missing setter field",
                        self.family,
                        parameter.name
                    );
                }
            }
        }
    }

    fn access_contract(table: &[Parameter], cases: &[(&str, Access)]) -> Result<(), String> {
        for &(name, expected) in cases {
            let Some(parameter) = table.iter().find(|row| row.name == name) else {
                return Err(format!(
                    "missing independently expected instance keyword {name}"
                ));
            };
            if parameter.access != expected {
                return Err(format!(
                    "{name}: expected {expected:?}, got {:?}",
                    parameter.access
                ));
            }
        }
        Ok(())
    }

    pub(super) fn assert_independent_access_expectations(&self) {
        Self::access_contract(self.instance_table, self.access_cases)
            .unwrap_or_else(|error| panic!("{} access contract: {error}", self.family));
        // Mutation-style sensitivity: this test must fail for an incorrect
        // instance-only or ask-only flag, even when row metadata is consistent
        // with the wrong access. Never infer expected access from the table.
        for &(name, expected) in self.access_cases {
            let mut mutated = self.instance_table.to_vec();
            let row = mutated
                .iter_mut()
                .find(|row| row.name == name)
                .unwrap_or_else(|| panic!("missing independent fixture {name}"));
            row.access = if expected == Access::Output {
                Access::ReadWrite
            } else {
                Access::Output
            };
            assert!(
                Self::access_contract(&mutated, self.access_cases).is_err(),
                "{} unexpectedly accepted wrong access for {name}",
                self.family,
            );
        }
    }

    pub(super) fn assert_model_setters(&self) {
        for parameter in self.model_table {
            let assignment = match parameter.input_type {
                InputType::Real => format!("{}=0.25", parameter.name),
                InputType::Integer => format!("{}=1.4", parameter.name),
                InputType::Flag => parameter.name.into(),
                InputType::String => format!("version={}", self.model_version),
                InputType::RealVector => panic!("{} model has no vector setter", self.family),
            };
            let inputs = (self.model_inputs)(&assignment).unwrap_or_else(|error| {
                panic!("{} model {}: {error}", self.family, parameter.name)
            });
            let value = &inputs
                .get(parameter.name)
                .unwrap_or_else(|| {
                    panic!(
                        "{} model {} missing applied setter",
                        self.family, parameter.name
                    )
                })
                .value;
            match parameter.input_type {
                InputType::Real => assert!(
                    matches!(value, Value::Real(v) if v.is_finite()),
                    "{} {}",
                    self.family,
                    parameter.name
                ),
                InputType::Integer => assert_eq!(
                    *value,
                    Value::Integer(1),
                    "{} {}",
                    self.family,
                    parameter.name
                ),
                InputType::Flag => assert_eq!(
                    *value,
                    Value::Flag(true),
                    "{} {}",
                    self.family,
                    parameter.name
                ),
                InputType::String => assert_eq!(
                    *value,
                    Value::String(self.model_version.into()),
                    "{} {}",
                    self.family,
                    parameter.name
                ),
                InputType::RealVector => unreachable!(),
            }
        }
    }

    pub(super) fn assert_instance_setters_and_asks(&self) {
        for parameter in self.instance_table {
            let assignment = match parameter.input_type {
                InputType::Real => format!("{}=0.25", parameter.name),
                InputType::Integer => format!("{}=1.4", parameter.name),
                InputType::Flag => parameter.name.into(),
                InputType::RealVector => "ic=(0.1 0.2 -0.3)".into(),
                InputType::String => panic!(
                    "{} {} unexpected string instance row",
                    self.family, parameter.name
                ),
            };
            if parameter.access == Access::Output {
                // Exercise the public typed-device boundary, not only parser
                // grammar rejection, with independently inserted AST input.
                let error = self.programmatic_ask(parameter.name).unwrap_err();
                assert!(
                    error.to_string().contains("ask-only"),
                    "{} {}: {error}",
                    self.family,
                    parameter.name
                );
                continue;
            }
            let inputs = (self.instance_inputs)(&assignment).unwrap_or_else(|error| {
                panic!("{} instance {}: {error}", self.family, parameter.name)
            });
            if parameter.input_type == InputType::RealVector {
                for (name, expected) in [("icvds", 0.1), ("icvgs", 0.2), ("icvbs", -0.3)] {
                    assert_eq!(
                        inputs.get(name).map(|v| &v.value),
                        Some(&Value::Real(expected)),
                        "{} {} IC {name}",
                        self.family,
                        parameter.name,
                    );
                }
            } else {
                assert!(
                    inputs.get(parameter.name).is_some(),
                    "{} {} missing",
                    self.family,
                    parameter.name
                );
            }
        }
    }

    fn programmatic_ask(&self, name: &str) -> SpiceResult<Inputs> {
        let deck = format!(
            "schema contract\nm1 d g s b nn\n.model nn nmos level={}\n.end\n",
            self.level,
        );
        let parsed = Parser::new().parse_deck(&parse_deck_text(
            Path::new("bsim-shared-contract.cir"),
            &deck,
        ))?;
        let resolver = ModelResolver::new(&parsed.models)?;
        let resolved = resolver
            .resolve(&parsed.devices[0])?
            .ok_or_else(|| SpiceError::circuit("missing contract model"))?;
        let mut device = parsed.devices[0].clone();
        device.parameters.push(ParameterAssignment {
            name: name.into(),
            value: "0.25".into(),
            kind: ParameterKind::Scalar,
            location: device.location.clone(),
        });
        (self.instantiate)(&resolved, &device)
    }
}
