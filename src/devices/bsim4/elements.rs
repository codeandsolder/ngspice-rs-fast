//! `b4set.c`'source `TSTALLOC` sequence: the matrix elements one instance makes,
//! in C'source creation order, which fixes SPARSE'source numbering and pivot ties.
use super::fields::{Instance, Model};

/// The `(row, column)` local node pairs `BSIM4setup` allocates for `instance`,
/// in order. Pairs involving [`super::host::node::GROUND`] are included, as
/// C makes (trash-can) elements for them too; the host skips them.
pub(super) fn matrix_elements(model: &Model, instance: &Instance) -> Vec<(i32, i32)> {
    let (drain, gate, source, body) = (
        instance.dNode,
        instance.gNodeExt,
        instance.sNode,
        instance.bNode,
    );
    let (drain_prime, gate_prime, gate_mid, source_prime, body_prime) = (
        instance.dNodePrime,
        instance.gNodePrime,
        instance.gNodeMid,
        instance.sNodePrime,
        instance.bNodePrime,
    );
    let (drain_body, source_body, charge) = (instance.dbNode, instance.sbNode, instance.qNode);
    let mut list = vec![
        (drain_prime, body_prime),
        (gate_prime, body_prime),
        (source_prime, body_prime),
        (body_prime, drain_prime),
        (body_prime, gate_prime),
        (body_prime, source_prime),
        (body_prime, body_prime),
        (drain, drain),
        (gate_prime, gate_prime),
        (source, source),
        (drain_prime, drain_prime),
        (source_prime, source_prime),
        (drain, drain_prime),
        (gate_prime, drain_prime),
        (gate_prime, source_prime),
        (source, source_prime),
        (drain_prime, source_prime),
        (drain_prime, drain),
        (drain_prime, gate_prime),
        (source_prime, gate_prime),
        (source_prime, source),
        (source_prime, drain_prime),
        (charge, charge),
        (charge, body_prime),
        (charge, drain_prime),
        (charge, source_prime),
        (charge, gate_prime),
        (drain_prime, charge),
        (source_prime, charge),
        (gate_prime, charge),
    ];
    if instance.rgateMod != 0 {
        list.extend([
            (gate, gate),
            (gate, gate_prime),
            (gate_prime, gate),
            (gate, drain_prime),
            (gate, source_prime),
            (gate, body_prime),
            (gate_mid, drain_prime),
            (gate_mid, gate_prime),
            (gate_mid, gate_mid),
            (gate_mid, gate),
            (gate_mid, source_prime),
            (gate_mid, body_prime),
            (drain_prime, gate_mid),
            (gate_prime, gate_mid),
            (gate, gate_mid),
            (source_prime, gate_mid),
            (body_prime, gate_mid),
        ]);
    }
    if instance.rbodyMod == 1 || instance.rbodyMod == 2 {
        list.extend([
            (drain_prime, drain_body),
            (source_prime, source_body),
            (drain_body, drain_prime),
            (drain_body, drain_body),
            (drain_body, body_prime),
            (drain_body, body),
            (body_prime, drain_body),
            (body_prime, body),
            (body_prime, source_body),
            (source_body, source_prime),
            (source_body, body_prime),
            (source_body, body),
            (source_body, source_body),
            (body, drain_body),
            (body, body_prime),
            (body, source_body),
            (body, body),
        ]);
    }
    if model.rdsMod != 0 {
        list.extend([
            (drain, gate_prime),
            (drain, source_prime),
            (drain, body_prime),
            (source, drain_prime),
            (source, gate_prime),
            (source, body_prime),
        ]);
    }
    list
}
