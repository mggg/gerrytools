use crate::{DeltaChange, Error, Result, StreamSummary};
#[cfg(any(feature = "python", test))]
use rayon::prelude::*;
use sha2::{Digest, Sha256};
use std::collections::{HashMap, HashSet};

type Sha256Digest = [u8; 32];

#[derive(Default)]
pub(super) struct UniquenessCounts {
    plans: HashSet<Sha256Digest>,
    districts: HashSet<Sha256Digest>,
    current: Option<DistrictMembership>,
}

struct DistrictMembership {
    node_count: usize,
    nodes: HashMap<u16, Vec<u64>>,
    hashes: HashMap<u16, Sha256Digest>,
}

pub(super) struct AssignmentHashes {
    plan: Sha256Digest,
    districts: Vec<Sha256Digest>,
}

impl UniquenessCounts {
    pub(super) fn reset(&mut self, assignment: &[u16]) {
        let membership = DistrictMembership::new(assignment);
        let hashes = membership.assignment_hashes();
        self.current = Some(membership);
        self.observe_hashes(hashes);
    }

    pub(super) fn observe_changes(&mut self, changes: &[DeltaChange]) -> Result<()> {
        let hashes = self
            .current
            .as_mut()
            .expect("delta uniqueness tracking starts from a full assignment")
            .apply(changes)?;
        self.observe_hashes(hashes);
        Ok(())
    }

    pub(super) fn has_current(&self) -> bool {
        self.current.is_some()
    }

    pub(super) fn observe_hashes(&mut self, hashes: AssignmentHashes) {
        for digest in hashes.districts {
            self.districts.insert(digest);
        }
        self.plans.insert(hashes.plan);
    }

    pub(super) fn apply_to(&self, summary: &mut StreamSummary) -> Result<()> {
        summary.unique_plans = Some(
            self.plans
                .len()
                .try_into()
                .map_err(|_| Error::InvalidInput("unique plan count overflowed u64".into()))?,
        );
        summary.unique_districts = Some(
            self.districts
                .len()
                .try_into()
                .map_err(|_| Error::InvalidInput("unique district count overflowed u64".into()))?,
        );
        Ok(())
    }
}

#[cfg(any(feature = "python", test))]
pub(crate) fn count_assignments(assignments: &[Vec<u16>]) -> Result<(u64, u64)> {
    let hashes = assignments
        .par_iter()
        .map(|assignment| assignment_hashes(assignment))
        .collect::<Result<Vec<_>>>()?;
    let mut counts = UniquenessCounts::default();
    for hashes in hashes {
        counts.observe_hashes(hashes);
    }
    Ok((
        counts
            .plans
            .len()
            .try_into()
            .map_err(|_| Error::InvalidInput("unique plan count overflowed u64".into()))?,
        counts
            .districts
            .len()
            .try_into()
            .map_err(|_| Error::InvalidInput("unique district count overflowed u64".into()))?,
    ))
}

impl DistrictMembership {
    fn new(assignment: &[u16]) -> Self {
        let word_count = assignment.len().div_ceil(64);
        let mut nodes = HashMap::<u16, Vec<u64>>::new();
        for (node, &district) in assignment.iter().enumerate() {
            let words = nodes.entry(district).or_insert_with(|| vec![0; word_count]);
            words[node / 64] |= 1_u64 << (node % 64);
        }
        let hashes = nodes
            .iter()
            .map(|(&district, words)| (district, hash_bitset(words)))
            .collect();
        Self {
            node_count: assignment.len(),
            nodes,
            hashes,
        }
    }

    fn apply(&mut self, changes: &[DeltaChange]) -> Result<AssignmentHashes> {
        let mut touched = Vec::with_capacity(changes.len() * 2);
        for change in changes {
            if change.node >= self.node_count {
                return Err(Error::DeltaNodeOutOfRange {
                    node: change.node,
                    assignment_len: self.node_count,
                });
            }
            let old = self
                .nodes
                .get_mut(&change.old)
                .expect("validated delta old district is present");
            old[change.node / 64] &= !(1_u64 << (change.node % 64));
            let word_count = self.node_count.div_ceil(64);
            let new = self
                .nodes
                .entry(change.new)
                .or_insert_with(|| vec![0; word_count]);
            new[change.node / 64] |= 1_u64 << (change.node % 64);
            touched.extend([change.old, change.new]);
        }
        touched.sort_unstable();
        touched.dedup();
        for district in touched {
            let words = self
                .nodes
                .get(&district)
                .expect("every touched district has a bitset");
            if words.iter().all(|&word| word == 0) {
                self.nodes.remove(&district);
                self.hashes.remove(&district);
            } else {
                self.hashes.insert(district, hash_bitset(words));
            }
        }
        Ok(self.assignment_hashes())
    }

    fn assignment_hashes(&self) -> AssignmentHashes {
        plan_hashes(self.hashes.values().copied().collect())
    }
}

/// Hash ordered little-endian `u64` node indices per district, then hash the sorted district
/// digests per plan.
pub(super) fn assignment_hashes(assignment: &[u16]) -> Result<AssignmentHashes> {
    let mut hashers = HashMap::<u16, Sha256>::new();
    for (node, &district) in assignment.iter().enumerate() {
        let node = u64::try_from(node)
            .map_err(|_| Error::InvalidInput("node index overflowed u64".into()))?;
        hashers
            .entry(district)
            .or_default()
            .update(node.to_le_bytes());
    }
    let hashes = hashers
        .into_values()
        .map(|hasher| hasher.finalize().into())
        .collect::<Vec<Sha256Digest>>();
    Ok(plan_hashes(hashes))
}

fn hash_bitset(words: &[u64]) -> Sha256Digest {
    let mut hasher = Sha256::new();
    for (word_index, &word) in words.iter().enumerate() {
        let mut remaining = word;
        while remaining != 0 {
            let bit = remaining.trailing_zeros() as usize;
            hasher.update(((word_index * 64 + bit) as u64).to_le_bytes());
            remaining &= remaining - 1;
        }
    }
    hasher.finalize().into()
}

fn plan_hashes(mut hashes: Vec<Sha256Digest>) -> AssignmentHashes {
    hashes.sort_unstable();
    let mut plan_hasher = Sha256::new();
    for digest in &hashes {
        plan_hasher.update(digest);
    }
    AssignmentHashes {
        plan: plan_hasher.finalize().into(),
        districts: hashes,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counts_without_district_label_or_order_dependence() {
        let mut counts = UniquenessCounts::default();
        counts.reset(&[1, 1, 2, 2]);
        counts.reset(&[2, 2, 1, 1]);
        counts.reset(&[1, 2, 1, 2]);

        assert_eq!(counts.plans.len(), 2);
        assert_eq!(counts.districts.len(), 4);
    }

    #[test]
    fn delta_updates_match_full_assignment_hashes() {
        let mut counts = UniquenessCounts::default();
        counts.reset(&[1, 1, 2, 2]);
        counts
            .observe_changes(&[
                DeltaChange {
                    node: 1,
                    old: 1,
                    new: 2,
                },
                DeltaChange {
                    node: 2,
                    old: 2,
                    new: 1,
                },
            ])
            .unwrap();

        let expected = assignment_hashes(&[1, 2, 1, 2]).unwrap().plan;
        assert!(counts.plans.contains(&expected));
        assert_eq!(counts.plans.len(), 2);
        assert_eq!(counts.districts.len(), 4);
    }

    #[test]
    fn counts_independent_assignments() {
        let assignments = vec![vec![0, 0, 1], vec![1, 1, 0], vec![0, 1, 1]];

        assert_eq!(count_assignments(&assignments).unwrap(), (2, 4));
    }
}
