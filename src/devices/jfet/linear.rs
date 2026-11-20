//! Shared small-signal and pole-zero loading for the classic gate devices.
//! Keep C statement order and model-specific multiplicity distinct from
//! the intrinsic evaluation and from transient charge companions.

use crate::devices::LinearContext;
use crate::primitives::{NodeId, Real, SpiceResult};

/// The two PZ reference routines share topology and statement order, but
/// MESpzLoad leaves parts of three diagonal conductances unscaled by M.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::devices) enum GatePencil {
    Jfet,
    Mesfet,
}

#[derive(Debug, Clone, Copy)]
enum GateCapacitance {
    None,
    Both,
    Drain,
    Source,
    NegativeDrain,
    NegativeSource,
}

impl GateCapacitance {
    fn pencil(self, m: Real, [capgs, capgd]: [Real; 2]) -> Option<Real> {
        match self {
            Self::None => None,
            Self::Both => Some(m * (capgd + capgs)),
            Self::Drain => Some(m * capgd),
            Self::Source => Some(m * capgs),
            Self::NegativeDrain => Some(-(m * capgd)),
            Self::NegativeSource => Some(-(m * capgs)),
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct GateEntry {
    row: NodeId,
    col: NodeId,
    conductance: Real,
    capacitance: GateCapacitance,
}

/// The C AC/PZ statement sequence. Capacitance identity is attached to
/// the statement, not inferred from node IDs: prime nodes can alias external
/// nodes, and each addition must retain its original position in the load.
fn gate_entries(
    terminals: &[NodeId],
    inner: [NodeId; 3],
    m: Real,
    [gdpr, gspr]: [Real; 2],
    [gm, gds, ggs, ggd]: [Real; 4],
    mes_pencil: bool,
) -> [GateEntry; 15] {
    use GateCapacitance as Cap;
    let [d, g, s] = [0, 1, 2].map(|k| terminals[k]);
    let [dp, _, sp] = inner;
    let [gg, dd, ss] = if mes_pencil {
        // mespzld.c's actual left-associated expressions. Do not correct
        // C's precedence mistake by scaling the whole diagonal.
        [
            m * ggd + ggs,
            m * gdpr + gds + ggd,
            m * gspr + gds + gm + ggs,
        ]
    } else {
        [
            m * (ggd + ggs),
            m * (gdpr + gds + ggd),
            m * (gspr + gds + gm + ggs),
        ]
    };
    [
        (d, d, m * gdpr, Cap::None),
        (g, g, gg, Cap::Both),
        (s, s, m * gspr, Cap::None),
        (dp, dp, dd, Cap::Drain),
        (sp, sp, ss, Cap::Source),
        (d, dp, -(m * gdpr), Cap::None),
        (g, dp, -(m * ggd), Cap::NegativeDrain),
        (g, sp, -(m * ggs), Cap::NegativeSource),
        (s, sp, -(m * gspr), Cap::None),
        (dp, d, -(m * gdpr), Cap::None),
        (dp, g, m * (-ggd + gm), Cap::NegativeDrain),
        (dp, sp, m * (-gds - gm), Cap::None),
        (sp, g, m * (-ggs - gm), Cap::NegativeSource),
        (sp, s, -(m * gspr), Cap::None),
        (sp, dp, -(m * gds), Cap::None),
    ]
    .map(|(row, col, conductance, capacitance)| GateEntry {
        row,
        col,
        conductance,
        capacitance,
    })
}

/// The real part of the gate-device AC loads, every stamp scaled by M.
pub(in crate::devices) fn ac_conductances(
    context: &mut LinearContext<'_>,
    terminals: &[NodeId],
    inner: [NodeId; 3],
    m: Real,
    series: [Real; 2],
    intrinsic: [Real; 4],
) -> SpiceResult<()> {
    for entry in gate_entries(terminals, inner, m, series, intrinsic, false) {
        if let (Some(r), Some(c)) = (
            context.unknowns.node_row(entry.row),
            context.unknowns.node_row(entry.col),
        ) {
            context.system.a.add(r, c, entry.conductance)?;
        }
    }
    Ok(())
}

/// The linearized intrinsic and series quantities of one gate device.
#[derive(Debug, Clone, Copy)]
pub(in crate::devices) struct GateLinearization {
    pub(in crate::devices) series: [Real; 2],
    pub(in crate::devices) intrinsic: [Real; 4],
    pub(in crate::devices) capacitances: [Real; 2],
}

impl GateLinearization {
    pub(in crate::devices) fn assemble_ac(
        self,
        context: &mut LinearContext<'_>,
        terminals: &[NodeId],
        inner: [NodeId; 3],
        m: Real,
    ) -> SpiceResult<()> {
        let [capgs, capgd] = self.capacitances;
        ac_gate_capacitances(context, inner, m, capgs, capgd)?;
        ac_conductances(context, terminals, inner, m, self.series, self.intrinsic)
    }

    pub(in crate::devices) fn assemble_pole_zero(
        self,
        context: &mut LinearContext<'_>,
        terminals: &[NodeId],
        inner: [NodeId; 3],
        m: Real,
        pencil: GatePencil,
    ) -> SpiceResult<()> {
        gate_pole_zero(context, terminals, inner, m, self, pencil)
    }
}

/// C's PZ routines add the real conductance and the complex s*C term
/// separately, even when multiple prime/external pointers alias. Retain
/// those statements rather than folding them into one g+s*C term.
/// JFET's m*(C*s) vs MES's (m*C)*s association cannot yet be expressed by
/// PencilTerm's single capacitance coefficient; both use the latter here.
fn gate_pole_zero(
    context: &mut LinearContext<'_>,
    terminals: &[NodeId],
    inner: [NodeId; 3],
    m: Real,
    GateLinearization {
        series,
        intrinsic,
        capacitances,
    }: GateLinearization,
    pencil: GatePencil,
) -> SpiceResult<()> {
    use crate::maths::pencil::PencilTerm;
    for entry in gate_entries(
        terminals,
        inner,
        m,
        series,
        intrinsic,
        pencil == GatePencil::Mesfet,
    ) {
        let (Some(row), Some(col)) = (
            context.unknowns.node_row(entry.row),
            context.unknowns.node_row(entry.col),
        ) else {
            continue;
        };
        let conductance = entry.conductance;
        context.system.a.add(row, col, conductance)?;
        if let Some(statements) = context.pole_zero_statements.as_deref_mut() {
            statements.push(PencilTerm {
                row,
                col,
                conductance: Some(conductance),
                capacitance: None,
            });
        }
        if let Some(capacitance) = entry.capacitance.pencil(m, capacitances) {
            context.system.e.add(row, col, capacitance)?;
            if let Some(statements) = context.pole_zero_statements.as_deref_mut() {
                statements.push(PencilTerm {
                    row,
                    col,
                    conductance: None,
                    capacitance: Some(capacitance),
                });
            }
        }
    }
    Ok(())
}

/// The gate capacitances' imaginary part of `jfetacld.c`/`jfet2acld.c`
/// (`xgs = capgs * omega`, `xgd = capgd * omega`), in C's order.
pub(in crate::devices) fn ac_gate_capacitances(
    context: &mut LinearContext<'_>,
    inner: [NodeId; 3],
    m: Real,
    capgs: Real,
    capgd: Real,
) -> SpiceResult<()> {
    let [dp, g, sp] = inner;
    for (row, col, terms, scale) in [
        (g, g, &[capgd, capgs][..], m),
        (dp, dp, &[capgd][..], m),
        (sp, sp, &[capgs][..], m),
        (g, dp, &[capgd][..], -m),
        (g, sp, &[capgs][..], -m),
        (dp, g, &[capgd][..], -m),
        (sp, g, &[capgs][..], -m),
    ] {
        context.scaled_dynamic_sum(row, col, terms, scale)?;
    }
    Ok(())
}
