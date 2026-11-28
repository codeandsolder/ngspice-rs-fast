//! Resolved physical passive startup constraints.
//!
//! Source-language policy
//! (BRANCH, analysis expressions, mode defaults) is resolved by preparation;
//! the engine consumes only the resulting voltage/current equations.
use std::collections::BTreeMap;

use crate::devices::{LinearContext, StampContext, StorageKind};
use crate::primitives::{AnalysisKind, NodeId, Real, SpiceError, SpiceResult};

/// A physical two-terminal bias equation, independent of the source dialect.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PassiveBias {
    /// Differential voltage; None denotes an exact voltage-source branch,
    /// Some denotes a finite source conductance in siemens.
    Voltage {
        /// Target differential voltage in volts.
        voltage: Real,
        /// Parallel conductance in siemens; None enforces the target exactly.
        conductance: Option<Real>,
    },
    /// Oriented current plus a parallel conductance. Zero conductance is
    /// an ideal current constraint; the existing inductor row is reused.
    Current {
        /// Oriented target current in amperes.
        current: Real,
        /// Parallel conductance in siemens.
        conductance: Real,
    },
}

impl PassiveBias {
    const fn kind(self) -> StorageKind {
        match self {
            Self::Voltage { .. } => StorageKind::Capacitor,
            Self::Current { .. } => StorageKind::Inductor,
        }
    }

    fn validate(self) -> bool {
        match self {
            Self::Voltage {
                voltage,
                conductance,
            } => {
                voltage.is_finite()
                    && conductance
                        .is_none_or(|g| g.is_finite() && g != 0. && (g * voltage).is_finite())
            }
            Self::Current {
                current,
                conductance,
            } => current.is_finite() && conductance.is_finite(),
        }
    }

    const fn needs_branch(self) -> bool {
        matches!(
            self,
            Self::Voltage {
                conductance: None,
                ..
            }
        )
    }
}

/// Immutable occurrence-specific constraints for the enclosing analysis.
/// The storage IC is independent of the selected bias equation.
#[derive(Debug, Clone, PartialEq)]
pub struct PassiveInitialization {
    kind: StorageKind,
    storage_ic: Option<Real>,
    bias: BTreeMap<AnalysisKind, PassiveBias>,
    hard_branch: bool,
    exact_history: bool,
    terminal_bias: bool,
}

impl PassiveInitialization {
    /// Validate and bind the physical equations once, never in Newton loads.
    ///
    /// # Errors
    /// A nonfinite target, invalid conductance, mismatched storage family,
    /// or duplicate enclosing analysis.
    pub fn new(
        kind: StorageKind,
        storage_ic: Option<Real>,
        constraints: impl IntoIterator<Item = (AnalysisKind, PassiveBias)>,
    ) -> SpiceResult<Self> {
        if storage_ic.is_some_and(|ic| !ic.is_finite()) {
            return Err(SpiceError::circuit(
                "nonfinite prepared storage initial condition",
            ));
        }
        let mut bias = BTreeMap::new();
        for (analysis, constraint) in constraints {
            if constraint.kind() != kind || !constraint.validate() {
                return Err(SpiceError::circuit(
                    "invalid prepared passive bias equation",
                ));
            }
            if bias.insert(analysis, constraint).is_some() {
                return Err(SpiceError::circuit(
                    "duplicate prepared passive bias analysis",
                ));
            }
        }
        let hard_branch = bias.values().copied().any(PassiveBias::needs_branch);
        Ok(Self {
            kind,
            storage_ic,
            bias,
            hard_branch,
            exact_history: false,
            terminal_bias: false,
        })
    }

    /// Select a bias network across the external terminal pair, independently
    /// of the physical series path and shunts. Bias conductances are aggregate.
    #[must_use = "the returned policy selects terminal rather than storage bias"]
    pub const fn with_terminal_bias(mut self) -> Self {
        self.terminal_bias = true;
        self
    }

    /// Prescribe the storage history at transient startup even without UIC.
    ///
    /// # Errors
    /// No explicit initial storage coordinate was supplied.
    pub fn with_exact_history(mut self) -> SpiceResult<Self> {
        if self.storage_ic.is_none() {
            return Err(SpiceError::circuit(
                "exact startup history requires a storage IC",
            ));
        }
        self.exact_history = true;
        Ok(self)
    }

    pub(super) const fn exact_initial(&self) -> Option<Real> {
        if self.exact_history {
            self.storage_ic
        } else {
            None
        }
    }

    pub(super) const fn terminal_bias(&self) -> bool {
        self.terminal_bias
    }

    /// For an external bias, the main capacitor is open during bias and
    /// the inductor carries the prescribed winding current. All terminal
    /// conductance belongs to the separate terminal equation.
    pub(super) fn primary_policy(&self) -> Self {
        let mut primary = self.clone();
        primary.terminal_bias = false;
        if self.terminal_bias {
            primary.hard_branch = false;
            match self.kind {
                StorageKind::Capacitor => primary.bias.clear(),
                StorageKind::Inductor => {
                    for constraint in primary.bias.values_mut() {
                        if let PassiveBias::Current { conductance, .. } = constraint {
                            *conductance = 0.;
                        }
                    }
                }
            }
        }
        primary
    }

    pub(super) fn check_kind(&self, kind: StorageKind) -> SpiceResult<()> {
        if self.kind != kind {
            return Err(SpiceError::circuit(
                "prepared bias does not match storage family",
            ));
        }
        Ok(())
    }

    pub(super) const fn storage_ic(&self) -> Option<Real> {
        self.storage_ic
    }

    pub(super) const fn has_constraints(&self) -> bool {
        !self.bias.is_empty()
    }

    pub(super) fn constrains(&self, analysis: AnalysisKind) -> bool {
        self.bias.contains_key(&analysis)
    }

    pub(super) const fn needs_branch(&self) -> bool {
        self.hard_branch
    }

    pub(super) fn active(&self, context: &StampContext<'_>) -> Option<PassiveBias> {
        if !context.mode.is_bias() {
            return None;
        }
        self.bias.get(&context.bias_analysis?).copied()
    }

    /// Bias-only capacitor current or voltage relation; keep a reserved
    /// exact branch algebraically neutral outside the selected bias phase.
    pub(super) fn stamp_capacitor(
        &self,
        terminals: [NodeId; 2],
        context: &mut StampContext<'_>,
    ) -> SpiceResult<()> {
        let active = self.active(context);
        if let Some(PassiveBias::Voltage {
            voltage,
            conductance: None,
        }) = active
        {
            let row = context.branch(0)?;
            crate::devices::linear::branch_stamp(context.matrix, context.unknowns, terminals, row)?;
            context.rhs.add_to(row, voltage)?;
        } else if self.needs_branch() {
            let row = context.branch(0)?;
            context.matrix.add(row, row, 1.)?;
        }
        if let Some(PassiveBias::Voltage {
            voltage,
            conductance: Some(g),
        }) = active
        {
            crate::devices::rlc::stamp_conductance(terminals, g, context)?;
            context.stamp_rhs(terminals[0], g * voltage)?;
            context.stamp_rhs(terminals[1], -g * voltage)?;
        }
        Ok(())
    }

    /// Replace a storage device's owned branch equation during bias.
    /// KCL always retains the physical terminal current. The ordinary
    /// constitutive equation is restored outside the selected bias.
    pub(super) fn stamp_owned_branch(
        &self,
        terminals: [NodeId; 2],
        row: usize,
        context: &mut StampContext<'_>,
    ) -> SpiceResult<bool> {
        let Some(constraint) = self.active(context) else {
            return Ok(false);
        };
        let (voltage, current, rhs) = match constraint {
            PassiveBias::Voltage {
                voltage,
                conductance: None,
            } => (1., 0., voltage),
            PassiveBias::Voltage {
                voltage,
                conductance: Some(g),
            } => (-g, 1., -g * voltage),
            PassiveBias::Current {
                current,
                conductance,
            } => (-conductance, 1., current),
        };
        for (node, sign) in [(terminals[0], 1.), (terminals[1], -1.)] {
            if let Some(node_row) = context.unknowns.node_row(node) {
                context.matrix.add(node_row, row, sign)?;
                if voltage != 0. {
                    context.matrix.add(row, node_row, sign * voltage)?;
                }
            }
        }
        if current != 0. {
            context.matrix.add(row, row, current)?;
        }
        context.rhs.add_to(row, rhs)?;
        Ok(true)
    }

    pub(super) fn stamp_terminal(
        &self,
        terminals: [NodeId; 2],
        context: &mut StampContext<'_>,
    ) -> SpiceResult<()> {
        match self.kind {
            StorageKind::Capacitor => self.stamp_capacitor(terminals, context),
            StorageKind::Inductor => {
                if let Some(PassiveBias::Current { conductance, .. }) = self.active(context) {
                    crate::devices::rlc::stamp_conductance(terminals, conductance, context)?;
                }
                Ok(())
            }
        }
    }

    /// The voltage-source branch does not persist electrically after bias.
    pub(super) fn release_capacitor(&self, context: &mut LinearContext<'_>) -> SpiceResult<()> {
        release_capacitor_branch(self.needs_branch(), context)
    }
}

/// Keep a reserved branch neutral in ordinary AC/linear and sensitivity loads.
pub(super) fn release_capacitor_branch(
    reserved: bool,
    context: &mut LinearContext<'_>,
) -> SpiceResult<()> {
    if reserved {
        let row = context
            .branch
            .ok_or_else(|| SpiceError::circuit("missing prepared capacitor bias branch"))?;
        context.system.a.add(row, row, 1.)?;
    }
    Ok(())
}

pub(super) fn bind(
    slot: &mut Option<Box<PassiveInitialization>>,
    kind: StorageKind,
    policy: PassiveInitialization,
) -> SpiceResult<()> {
    policy.check_kind(kind)?;
    *slot = Some(Box::new(policy));
    Ok(())
}

pub(super) fn has_active_constraints(
    policy: Option<&PassiveInitialization>,
    analysis: AnalysisKind,
) -> bool {
    policy.is_some_and(|policy| policy.constrains(analysis))
}

pub(super) fn initial_value(
    policy: Option<&PassiveInitialization>,
    native: Option<Real>,
) -> Option<Real> {
    policy.map_or(native, PassiveInitialization::storage_ic)
}

pub(super) fn has_constraints(policy: Option<&PassiveInitialization>) -> bool {
    policy.is_some_and(PassiveInitialization::has_constraints)
}

pub(super) fn needs_branch(policy: Option<&PassiveInitialization>) -> bool {
    policy.is_some_and(PassiveInitialization::needs_branch)
}

/// Physical voltage-to-ground bias and the separate UIC voltage hint.
#[derive(Debug, Clone, PartialEq)]
pub struct NodeInitialization {
    hint: Real,
    location: crate::primitives::SourceLoc,
    bias: BTreeMap<AnalysisKind, (Real, Real)>,
}

impl NodeInitialization {
    /// Each tuple contains an enclosing analysis, voltage and conductance.
    ///
    /// # Errors
    /// Nonfinite hint, nonpositive conductance, nonfinite Norton source,
    /// or duplicate enclosing analysis.
    pub fn new(
        hint: Real,
        location: crate::primitives::SourceLoc,
        constraints: impl IntoIterator<Item = (AnalysisKind, Real, Real)>,
    ) -> SpiceResult<Self> {
        if !hint.is_finite() {
            return Err(SpiceError::parse(location, "nonfinite prepared node hint"));
        }
        let mut bias = BTreeMap::new();
        for (analysis, voltage, conductance) in constraints {
            if conductance <= 0.
                || !(PassiveBias::Voltage {
                    voltage,
                    conductance: Some(conductance),
                })
                .validate()
            {
                return Err(SpiceError::parse(
                    location,
                    "invalid prepared node bias equation",
                ));
            }
            if bias.insert(analysis, (voltage, conductance)).is_some() {
                return Err(SpiceError::parse(
                    location,
                    "duplicate prepared node bias analysis",
                ));
            }
        }
        Ok(Self {
            hint,
            location,
            bias,
        })
    }

    pub(crate) const fn hint(&self) -> (Real, &crate::primitives::SourceLoc) {
        (self.hint, &self.location)
    }

    pub(super) fn constrains(&self, analysis: AnalysisKind) -> bool {
        self.bias.contains_key(&analysis)
    }

    pub(super) fn stamp(
        &self,
        analysis: AnalysisKind,
        row: usize,
        matrix: &mut crate::maths::SparseMatrix,
        rhs: &mut crate::maths::Vector,
    ) -> SpiceResult<()> {
        if let Some(&(voltage, conductance)) = self.bias.get(&analysis) {
            matrix.add(row, row, conductance)?;
            rhs.add_to(row, voltage * conductance)?;
        }
        Ok(())
    }
}
