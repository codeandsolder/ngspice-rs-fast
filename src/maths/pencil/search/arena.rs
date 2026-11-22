//! Linked trial storage for the pole-zero search.
use super::{Cx, SpiceResult, numerical};

/// Stable identity of a linked trial. IDs are allocated by the arena and
/// never reused, including after a trial is pruned during deflation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct TrialId(usize);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum TrialKind {
    Sample,
    Minimum,
    Root(std::num::NonZeroU32),
}

#[derive(Debug, Clone, Copy)]
pub(super) struct Trial {
    s: Cx,
    pub(super) f_def: Cx,
    pub(super) mag_def: i64,
    pub(super) kind: TrialKind,
    pub(super) count: u32,
    next: Option<TrialId>,
    prev: Option<TrialId>,
}

impl Trial {
    pub(super) const fn new(s: Cx, f_def: Cx, mag_def: i64, kind: TrialKind) -> Self {
        Self {
            s,
            f_def,
            mag_def,
            kind,
            count: 0,
            next: None,
            prev: None,
        }
    }

    pub(super) const fn multiplicity(&self) -> u32 {
        match self.kind {
            TrialKind::Root(multiplicity) => multiplicity.get(),
            TrialKind::Sample | TrialKind::Minimum => 0,
        }
    }

    pub(super) const fn is_root(&self) -> bool {
        matches!(self.kind, TrialKind::Root(_))
    }
    pub(super) const fn s(&self) -> Cx {
        self.s
    }

    pub(super) const fn next(&self) -> Option<TrialId> {
        self.next
    }

    pub(super) const fn prev(&self) -> Option<TrialId> {
        self.prev
    }
}

fn invalid_trial() -> crate::primitives::SpiceError {
    numerical("pole-zero search", "invalid or removed trial identity")
}

/// Owns linked trials and their reference traversal. A proposed frequency
/// stays outside the arena until it is evaluated and linked; collisions
/// may return an existing ID without allocating an abandoned candidate.
#[derive(Default)]
pub(super) struct TrialArena {
    entries: Vec<Option<Trial>>,
    head: Option<TrialId>,
    anchor: Option<TrialId>,
}

impl TrialArena {
    pub(super) const fn head(&self) -> Option<TrialId> {
        self.head
    }

    pub(super) const fn anchor(&self) -> Option<TrialId> {
        self.anchor
    }

    pub(super) fn get(&self, id: TrialId) -> SpiceResult<&Trial> {
        self.entries
            .get(id.0)
            .and_then(Option::as_ref)
            .ok_or_else(invalid_trial)
    }

    pub(super) fn get_mut(&mut self, id: TrialId) -> SpiceResult<&mut Trial> {
        self.entries
            .get_mut(id.0)
            .and_then(Option::as_mut)
            .ok_or_else(invalid_trial)
    }

    pub(super) fn insert_after(
        &mut self,
        prev: Option<TrialId>,
        mut trial: Trial,
    ) -> SpiceResult<TrialId> {
        // A collision can stop the reference scan before reaching another
        // lower-frequency trial. Preserve that insertion point, rather than
        // imposing a total order that the numerical search does not require.
        if !trial.s.re.is_finite() || !trial.s.im.is_finite() {
            return Err(numerical("pole-zero search", "non-finite linked trial"));
        }
        let next = match prev {
            Some(id) => self.get(id)?.next,
            None => self.head,
        };
        if let Some(id) = next {
            self.get(id)?;
        }
        let id = TrialId(self.entries.len());
        trial.prev = prev;
        trial.next = next;
        self.entries.push(Some(trial));
        if let Some(prev) = prev {
            self.get_mut(prev)?.next = Some(id);
        } else {
            self.head = Some(id);
        }
        if let Some(next) = next {
            self.get_mut(next)?.prev = Some(id);
        } else if prev.is_none() {
            self.anchor = Some(id);
        }
        Ok(id)
    }

    pub(super) fn remove(&mut self, id: TrialId) -> SpiceResult<()> {
        let Trial { prev, next, .. } = *self.get(id)?;
        for neighbor in [prev, next].into_iter().flatten() {
            self.get(neighbor)?;
        }
        if let Some(prev) = prev {
            self.get_mut(prev)?.next = next;
        } else {
            self.head = next;
        }
        if let Some(next) = next {
            self.get_mut(next)?.prev = prev;
        }
        if self.anchor == Some(id) {
            self.anchor = next.or(prev);
        }
        self.entries[id.0] = None;
        Ok(())
    }

    pub(super) fn frequency(&self, id: TrialId) -> SpiceResult<Cx> {
        Ok(self.get(id)?.s)
    }

    pub(super) fn seek(
        &self,
        start: Option<TrialId>,
        direction: Seek,
    ) -> SpiceResult<Option<TrialId>> {
        let mut current = start;
        if direction != Seek::Anchor {
            current = self.neighbor(current, direction)?;
        }
        while let Some(id) = current {
            if self.get(id)?.kind == TrialKind::Sample {
                return Ok(Some(id));
            }
            current = self.neighbor(current, direction)?;
        }
        Ok(None)
    }

    fn neighbor(&self, id: Option<TrialId>, direction: Seek) -> SpiceResult<Option<TrialId>> {
        id.map_or(Ok(None), |id| {
            let t = self.get(id)?;
            Ok(match direction {
                Seek::Left => t.prev,
                Seek::Right | Seek::Anchor => t.next,
            })
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Seek {
    Left,
    Anchor,
    Right,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::primitives::Real;

    fn sample(re: Real, kind: TrialKind) -> Trial {
        Trial::new(Cx::new(re, 0.0), Cx::new(1.0, 0.0), 0, kind)
    }

    #[test]
    fn anchor_and_bidirectional_links_survive_pruning_without_id_reuse() {
        let mut arena = TrialArena::default();
        let center = arena
            .insert_after(None, sample(0.0, TrialKind::Sample))
            .unwrap();
        let left = arena
            .insert_after(None, sample(-1.0, TrialKind::Sample))
            .unwrap();
        let minimum = arena
            .insert_after(Some(center), sample(1.0, TrialKind::Minimum))
            .unwrap();
        let root = arena
            .insert_after(
                Some(minimum),
                sample(2.0, TrialKind::Root(std::num::NonZeroU32::MIN)),
            )
            .unwrap();
        let right = arena
            .insert_after(Some(root), sample(3.0, TrialKind::Sample))
            .unwrap();
        assert!(
            arena
                .insert_after(None, sample(Real::NAN, TrialKind::Sample))
                .is_err()
        );
        assert_eq!(arena.head(), Some(left));
        assert_eq!(arena.anchor(), Some(center));
        assert_eq!(arena.seek(Some(center), Seek::Right).unwrap(), Some(right));
        assert_eq!(arena.seek(Some(right), Seek::Left).unwrap(), Some(center));
        arena.remove(center).unwrap();
        assert_eq!(arena.anchor(), Some(minimum));
        assert_eq!(arena.get(left).unwrap().next(), Some(minimum));
        assert_eq!(arena.get(minimum).unwrap().prev(), Some(left));
        arena.remove(minimum).unwrap();
        assert_eq!(arena.anchor(), Some(root));
        arena.remove(root).unwrap();
        assert_eq!(arena.anchor(), Some(right));
        arena.remove(right).unwrap();
        assert_eq!(arena.anchor(), Some(left));
        arena.remove(left).unwrap();
        assert_eq!((arena.head(), arena.anchor()), (None, None));
        assert!(arena.remove(center).is_err());
        assert!(arena.get(TrialId(usize::MAX)).is_err());
        let new = arena
            .insert_after(None, sample(0.0, TrialKind::Sample))
            .unwrap();
        assert_ne!(new, center);
        assert_eq!(arena.anchor(), Some(new));
        assert!(arena.get(center).is_err());
    }
}
