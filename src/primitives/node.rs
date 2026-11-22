//! Node identity, and the `gnd` → `0` rule.
//!
//! Ported from:
//!
//! - `src/frontend/inpcom.c`, `inp_fix_gnd_name()` — rewrites the node name
//!   `gnd` to `0`, unless the `no_auto_gnd` variable is set, in which case
//!   `gnd` stays an ordinary node. ngspice also inserts `.global gnd` so the
//!   name survives subcircuit flattening; the port models the aliasing here and
//!   leaves the `.global` insertion to the parser.
//! - `src/spicelib/devices/cktbindnode.c` — node creation and the node index
//!   space.

use std::borrow::Cow;
use std::collections::HashMap;
use std::fmt;

use crate::primitives::{SpiceError, SpiceResult};

/// Canonical name of the ground node.
pub const GROUND_NAME: &str = "0";

/// The node name ngspice rewrites to [`GROUND_NAME`] unless `no_auto_gnd` is set.
pub const GROUND_ALIAS: &str = "gnd";

/// A node's index in the circuit's node table.
///
/// Ids are dense and stable: id 0 is always ground.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct NodeId(u32);

impl NodeId {
    /// Ground, which is always node 0.
    pub const GROUND: Self = Self(0);

    /// Builds an id from a raw index.
    #[must_use]
    pub const fn new(index: u32) -> Self {
        Self(index)
    }

    /// Convert an index without truncating the compact identity.
    ///
    /// # Errors
    /// An index outside the representable node identity range.
    pub fn from_index(index: usize) -> SpiceResult<Self> {
        u32::try_from(index)
            .map(Self)
            .map_err(|_| SpiceError::circuit("node table exceeds compact identity range"))
    }

    /// The raw index, for use as a row/column in the MNA matrix.
    #[must_use]
    pub const fn index(self) -> usize {
        self.0 as usize
    }
}

impl fmt::Display for NodeId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "node {}", self.0)
    }
}

/// What a node is used for, which decides whether it adds an unknown to the
/// MNA system.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NodeKind {
    /// Ground. Never contributes an unknown.
    Ground,
    /// An externally visible terminal of the circuit.
    Terminal,
    /// A node introduced by the simulator, for instance inside a device.
    Internal,
    /// The branch-current unknown of a voltage source or inductor, named
    /// `<instance>#branch`.
    BranchCurrent,
}

impl NodeKind {
    /// Whether this node contributes a row/column to the MNA matrix.
    #[must_use]
    pub const fn is_unknown(self) -> bool {
        !matches!(self, Self::Ground)
    }
}

/// One node: its canonical name, its id, and what it is for.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Node {
    /// Index in the owning [`NodeTable`].
    pub id: NodeId,
    /// Canonical (lowercased) name as first written.
    pub name: String,
    /// What the node is used for.
    pub kind: NodeKind,
}

/// Maps node names to ids, case-insensitively.
///
/// SPICE node names are case-insensitive, so `In`, `IN` and `in` are one node.
/// Names are stored in canonical lowercase for display and rawfile output.
#[derive(Debug, Clone)]
pub struct NodeTable {
    nodes: Vec<Node>,
    // Only lookup/removal use this index; the Vec determines all node order.
    index: HashMap<String, NodeId>,
    auto_gnd: bool,
}

impl NodeTable {
    /// Creates a table containing only ground, with `gnd` aliasing enabled.
    #[must_use]
    pub fn new() -> Self {
        Self::with_auto_gnd(true)
    }

    /// Creates a table containing only ground.
    ///
    /// With `auto_gnd` disabled, `gnd` is an ordinary node name, matching
    /// ngspice when `no_auto_gnd` is set.
    #[must_use]
    pub fn with_auto_gnd(auto_gnd: bool) -> Self {
        let ground = Node {
            id: NodeId::GROUND,
            name: GROUND_NAME.to_owned(),
            kind: NodeKind::Ground,
        };
        let mut index = HashMap::new();
        index.insert(GROUND_NAME.to_owned(), NodeId::GROUND);
        Self {
            nodes: vec![ground],
            index,
            auto_gnd,
        }
    }

    /// Whether `gnd` is aliased to ground.
    #[must_use]
    pub const fn auto_gnd(&self) -> bool {
        self.auto_gnd
    }

    /// The canonical spelling of a node name: lowercased, with `gnd` folded into
    /// `0` when aliasing is on.
    #[must_use]
    pub fn canonical_name(name: &str, auto_gnd: bool) -> String {
        Self::canonical(name, auto_gnd).into_owned()
    }

    fn canonical_borrowed(name: &str, auto_gnd: bool) -> Option<&str> {
        if auto_gnd && name.eq_ignore_ascii_case(GROUND_ALIAS) {
            Some(GROUND_NAME)
        } else if name
            .bytes()
            .all(|b| b.is_ascii() && !b.is_ascii_uppercase())
        {
            Some(name)
        } else {
            None
        }
    }

    fn canonical(name: &str, auto_gnd: bool) -> Cow<'_, str> {
        if let Some(name) = Self::canonical_borrowed(name, auto_gnd) {
            return Cow::Borrowed(name);
        }
        // Unicode lowercase includes contextual mappings (e.g. final
        // sigma), so do not substitute per-character ASCII folding.
        let lowered = name.to_lowercase();
        if auto_gnd && lowered == GROUND_ALIAS {
            Cow::Borrowed(GROUND_NAME)
        } else {
            Cow::Owned(lowered)
        }
    }

    fn try_copy_name(name: &str) -> SpiceResult<String> {
        let mut owned = String::new();
        owned
            .try_reserve_exact(name.len())
            .map_err(|error| SpiceError::circuit(format!("cannot allocate node name: {error}")))?;
        owned.push_str(name);
        Ok(owned)
    }

    fn canonical_for_intern(name: &str, auto_gnd: bool) -> SpiceResult<Cow<'_, str>> {
        if let Some(name) = Self::canonical_borrowed(name, auto_gnd) {
            return Ok(Cow::Borrowed(name));
        }
        if name.is_ascii() {
            let mut owned = Self::try_copy_name(name)?;
            owned.make_ascii_lowercase();
            return Ok(Cow::Owned(owned));
        }
        // The standard library's contextual Unicode lowercase conversion
        // has no fallible variant. Preserve its exact spelling semantics.
        Ok(Self::canonical(name, auto_gnd))
    }

    /// Returns the id of `name`, creating the node if needed.
    ///
    /// The node starts out as a [`NodeKind::Terminal`]; use
    /// [`NodeTable::set_kind`] to mark simulator-internal nodes.
    ///
    /// # Errors
    /// The compact node identity space or node storage is exhausted. On error,
    /// existing identities and the name index are unchanged. Context-sensitive
    /// Unicode normalization uses the standard library's infallible allocation.
    pub fn intern(&mut self, name: &str) -> SpiceResult<NodeId> {
        let canonical = Self::canonical_for_intern(name, self.auto_gnd)?;
        if let Some(id) = self.index.get(canonical.as_ref()) {
            return Ok(*id);
        }
        let id = NodeId::from_index(self.nodes.len())?;
        // Reserve both containers and prepare both names before changing
        // either the ordered nodes or their lookup index.
        self.nodes
            .try_reserve(1)
            .map_err(|error| SpiceError::circuit(format!("cannot allocate node: {error}")))?;
        self.index
            .try_reserve(1)
            .map_err(|error| SpiceError::circuit(format!("cannot allocate node index: {error}")))?;
        let node_name = Self::try_copy_name(canonical.as_ref())?;
        let key = match canonical {
            Cow::Borrowed(name) => Self::try_copy_name(name)?,
            Cow::Owned(name) => name,
        };
        self.nodes.push(Node {
            id,
            name: node_name,
            kind: NodeKind::Terminal,
        });
        self.index.insert(key, id);
        Ok(id)
    }

    /// Intern a fixed terminal layout transactionally.
    pub(crate) fn intern_array<const N: usize>(
        &mut self,
        names: &[String],
    ) -> SpiceResult<[NodeId; N]> {
        if names.len() != N {
            return Err(SpiceError::circuit("invalid terminal layout"));
        }
        let mark = self.len();
        let mut ids = [NodeId::GROUND; N];
        for (id, name) in ids.iter_mut().zip(names) {
            match self.intern(name) {
                Ok(node) => *id = node,
                Err(error) => {
                    self.truncate(mark);
                    return Err(error);
                }
            }
        }
        Ok(ids)
    }

    /// Looks up a node without creating it.
    #[must_use]
    pub fn get(&self, name: &str) -> Option<NodeId> {
        let canonical = Self::canonical(name, self.auto_gnd);
        self.index.get(canonical.as_ref()).copied()
    }

    /// The node with this id.
    #[must_use]
    pub fn node(&self, id: NodeId) -> Option<&Node> {
        self.nodes.get(id.index())
    }

    /// Every node, ground first.
    #[must_use]
    pub fn nodes(&self) -> &[Node] {
        &self.nodes
    }

    /// Number of nodes, including ground.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Removes every node interned after the table had `len` entries, so a
    /// failed construction can roll back without cloning the whole table.
    pub(crate) fn truncate(&mut self, len: usize) {
        for node in self.nodes.drain(len.max(1)..) {
            self.index.remove(&node.name);
        }
    }

    /// True when the table holds nothing but ground.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.nodes.len() <= 1
    }

    /// Number of unknowns the MNA matrix needs for these nodes.
    #[must_use]
    pub fn unknown_count(&self) -> usize {
        self.nodes
            .iter()
            .filter(|node| node.kind.is_unknown())
            .count()
    }

    /// Changes what a node is used for.
    pub fn set_kind(&mut self, id: NodeId, kind: NodeKind) {
        if let Some(node) = self.nodes.get_mut(id.index()) {
            node.kind = kind;
        }
    }
}

impl Default for NodeTable {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::{GROUND_ALIAS, GROUND_NAME, NodeId, NodeKind, NodeTable};

    #[test]
    fn canonical_lookup_borrows_ascii_and_preserves_unicode_lowercase() {
        use std::borrow::Cow;

        assert!(matches!(
            NodeTable::canonical("v(out)#branch", true),
            Cow::Borrowed("v(out)#branch")
        ));
        assert!(matches!(
            NodeTable::canonical("GND", true),
            Cow::Borrowed("0")
        ));
        assert!(matches!(NodeTable::canonical("GND", false), Cow::Owned(_)));
        let mut table = NodeTable::new();
        // Final sigma uses context-sensitive Unicode lowercase.
        let greek = table.intern("ΟΣ").unwrap();
        assert_eq!(table.node(greek).unwrap().name, "ος");
        assert_eq!(table.get("ος"), Some(greek));
        assert_eq!(table.get("ΟΣ"), Some(greek));
        let expanded = table.intern("İ").unwrap();
        assert_eq!(table.get("i\u{307}"), Some(expanded));
    }

    #[test]
    fn fixed_terminal_layout_validates_before_interning() {
        let mut table = NodeTable::new();
        let existing = table.intern("Out").unwrap();
        let names = ["new".to_owned()];
        assert!(table.intern_array::<2>(&names).is_err());
        assert!(table.get("new").is_none());
        assert_eq!(table.len(), 2);
        let names = ["OUT".to_owned(), "new".to_owned(), "New".to_owned()];
        let ids = table.intern_array::<3>(&names).unwrap();
        assert_eq!(ids[0], existing);
        assert_eq!(ids[1], ids[2]);
        assert_eq!(table.len(), 3);
    }

    #[test]
    fn compact_identity_boundary_never_truncates() {
        let max = NodeId::new(u32::MAX).index();
        assert_eq!(NodeId::from_index(max).unwrap(), NodeId::new(u32::MAX));
        if let Some(overflow) = max.checked_add(1) {
            assert!(NodeId::from_index(overflow).is_err());
        }
    }

    #[test]
    fn rollback_removes_index_entries_and_reuses_dense_ids() {
        let mut table = NodeTable::new();
        let retained = table.intern("Keep").unwrap();
        let mark = table.len();
        let removed = table.intern("Remove").unwrap();
        table.intern("ΟΣ").unwrap();
        table.truncate(mark);
        assert_eq!(table.get("keep"), Some(retained));
        assert!(table.get("REMOVE").is_none());
        assert!(table.get("ος").is_none());
        assert!(table.node(removed).is_none());
        assert_eq!(table.intern("Replacement").unwrap(), removed);
        assert_eq!(table.intern("REPLACEMENT").unwrap(), removed);
        assert_eq!(table.intern("GND").unwrap(), NodeId::GROUND);
    }

    #[test]
    fn ground_is_node_zero() {
        let table = NodeTable::new();
        assert_eq!(table.len(), 1);
        assert!(table.is_empty());
        assert_eq!(table.nodes()[0].id, NodeId::GROUND);
        assert_eq!(table.node(NodeId::GROUND).unwrap().kind, NodeKind::Ground);
        assert_eq!(table.unknown_count(), 0);
    }

    #[test]
    fn names_are_case_insensitive_and_deduplicated() {
        let mut table = NodeTable::new();
        let first = table.intern("Out").unwrap();
        assert_eq!(table.intern("OUT").unwrap(), first);
        assert_eq!(table.intern("out").unwrap(), first);
        assert_eq!(table.len(), 2);
        assert_eq!(table.node(first).unwrap().name, "out");
    }

    #[test]
    fn gnd_is_aliased_to_ground_by_default() {
        let mut table = NodeTable::new();
        assert_eq!(table.intern(GROUND_ALIAS).unwrap(), NodeId::GROUND);
        assert_eq!(table.intern("GND").unwrap(), NodeId::GROUND);
        assert_eq!(table.len(), 1);
    }

    #[test]
    fn gnd_is_a_node_when_auto_gnd_is_off() {
        let mut table = NodeTable::with_auto_gnd(false);
        let gnd = table.intern("gnd").unwrap();
        assert_ne!(gnd, NodeId::GROUND);
        assert_eq!(table.len(), 2);
        assert_eq!(table.unknown_count(), 1);
    }

    #[test]
    fn canonical_name_only_folds_the_exact_alias() {
        assert_eq!(NodeTable::canonical_name("GND", true), GROUND_NAME);
        assert_eq!(NodeTable::canonical_name("gnd1", true), "gnd1");
        assert_eq!(NodeTable::canonical_name("gnd", false), "gnd");
    }

    #[test]
    fn unknown_count_excludes_ground_and_internal_nodes() {
        let mut table = NodeTable::new();
        let terminal = table.intern("1").unwrap();
        let internal = table.intern("x1#internal").unwrap();
        table.set_kind(internal, NodeKind::Internal);
        assert_eq!(table.unknown_count(), 2);
        table.set_kind(terminal, NodeKind::Terminal);
        assert_eq!(table.node(internal).unwrap().kind, NodeKind::Internal);
    }
}
