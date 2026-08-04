use crate::{Error, Result};

/// One assignment change in an incremental update.
///
/// Change slices must be ordered by strictly increasing `node`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeltaChange {
    /// Zero-based node index.
    pub node: usize,
    /// Expected district label before the change.
    pub old: u16,
    /// District label after the change.
    pub new: u16,
}

/// Validate a delta against the assignment it claims to transform.
///
/// `current` is the authoritative pre-delta assignment, owned by the incremental state applying
/// the delta, so a delta produced against some other plan fails here instead of corrupting state.
pub(crate) fn validate_changes(current: &[u16], changes: &[DeltaChange]) -> Result<()> {
    let mut previous_node = None;
    for change in changes {
        let Some(&label) = current.get(change.node) else {
            return Err(Error::DeltaNodeOutOfRange {
                node: change.node,
                assignment_len: current.len(),
            });
        };
        if label != change.old {
            return Err(Error::DeltaOldLabelMismatch {
                node: change.node,
                expected: label,
                actual: change.old,
            });
        }
        if let Some(previous) = previous_node {
            if previous >= change.node {
                return Err(Error::DeltaNodesNotStrictlyIncreasing {
                    previous,
                    node: change.node,
                });
            }
        }
        previous_node = Some(change.node);
    }
    Ok(())
}

/// Write a validated delta's new labels into an owned assignment.
pub(crate) fn apply_changes(assignment: &mut [u16], changes: &[DeltaChange]) {
    for change in changes {
        assignment[change.node] = change.new;
    }
}

/// Materialize the post-delta assignment only when a metric's dense district storage must grow.
pub(crate) fn expanded_assignment(
    current: &[u16],
    changes: &[DeltaChange],
    district_slots: usize,
) -> Option<Vec<u16>> {
    changes
        .iter()
        .any(|change| change.new as usize >= district_slots)
        .then(|| {
            let mut assignment = current.to_vec();
            apply_changes(&mut assignment, changes);
            assignment
        })
}

/// Generation-stamped scratch marks that clear in O(1) by advancing the generation.
pub(crate) struct GenerationStamps {
    stamps: Vec<u64>,
    generation: u64,
}

impl GenerationStamps {
    pub(crate) fn new(len: usize) -> Self {
        Self {
            stamps: vec![0; len],
            generation: 0,
        }
    }

    /// Start a new generation, clearing all marks.
    pub(crate) fn advance(&mut self) {
        if self.generation == u64::MAX {
            self.stamps.fill(0);
            self.generation = 1;
        } else {
            self.generation += 1;
        }
    }

    pub(crate) fn mark(&mut self, index: usize) {
        self.stamps[index] = self.generation;
    }

    pub(crate) fn is_marked(&self, index: usize) -> bool {
        self.stamps[index] == self.generation
    }
}

pub(crate) struct PostDeltaLabels {
    new_label: Vec<u16>,
    changed: GenerationStamps,
}

impl PostDeltaLabels {
    pub(crate) fn new(node_count: usize) -> Self {
        Self {
            new_label: vec![0; node_count],
            changed: GenerationStamps::new(node_count),
        }
    }

    pub(crate) fn refresh(&mut self, changes: &[DeltaChange]) {
        self.changed.advance();
        for change in changes {
            self.changed.mark(change.node);
            self.new_label[change.node] = change.new;
        }
    }

    pub(crate) fn label(&self, before: &[u16], node: usize) -> u16 {
        if self.changed.is_marked(node) {
            self.new_label[node]
        } else {
            before[node]
        }
    }
}

#[cfg(test)]
#[path = "../tests/twodelta.rs"]
mod tests;
