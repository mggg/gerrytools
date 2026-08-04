/// The number of distinct districts representable by the `u16` assignment format.
#[cfg(feature = "python")]
pub(crate) const MAX_DISTRICTS: u32 = u16::MAX as u32 + 1;

#[derive(Debug, Default)]
pub(crate) struct DistrictSet {
    words: Vec<u64>,
}

impl DistrictSet {
    fn reset(&mut self, district_slots: usize) {
        self.words.clear();
        self.words.resize(district_slots.div_ceil(64), 0);
    }

    fn insert(&mut self, district: u16) {
        let district = district as usize;
        self.words
            .resize(self.words.len().max(district / 64 + 1), 0);
        self.words[district / 64] |= 1_u64 << (district % 64);
    }

    fn remove(&mut self, district: u16) {
        let district = district as usize;
        self.words[district / 64] &= !(1_u64 << (district % 64));
    }
}

// Dense labels keep metric hot loops directly indexed. It is not expected for sparse labels to be
// common
pub(crate) fn district_capacity(assignment: &[u16]) -> usize {
    assignment
        .iter()
        .map(|&district| district as usize + 1)
        .max()
        .unwrap_or(0)
}

pub(crate) fn observed_districts(assignment: &[u16]) -> (usize, DistrictSet) {
    let district_slots = district_capacity(assignment);
    let mut observed = DistrictSet::default();
    observed.reset(district_slots);
    for &district in assignment {
        observed.insert(district);
    }
    (district_slots, observed)
}

pub(crate) fn district_ids(observed: &DistrictSet) -> Vec<u16> {
    let mut districts = Vec::new();
    for (word_index, &word) in observed.words.iter().enumerate() {
        let mut remaining = word;
        while remaining != 0 {
            let bit = remaining.trailing_zeros() as usize;
            districts.push((word_index * 64 + bit) as u16);
            remaining &= remaining - 1;
        }
    }
    districts
}

/// Per-district node counts plus the dynamically sized set of occupied districts.
#[derive(Debug, Default)]
pub(crate) struct DistrictOccupancy {
    node_counts: Vec<u32>,
    observed: DistrictSet,
}

impl DistrictOccupancy {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// Rebuild counts from a full assignment.
    pub(crate) fn reset(&mut self, assignment: &[u16]) {
        let district_slots = district_capacity(assignment);
        self.node_counts.clear();
        self.node_counts.resize(district_slots, 0);
        self.observed.reset(district_slots);
        for &district in assignment {
            self.node_counts[district as usize] += 1;
            self.observed.insert(district);
        }
    }

    /// Move one node between districts; callers skip no-op changes.
    pub(crate) fn apply(&mut self, old: u16, new: u16) {
        let required = usize::from(old.max(new)) + 1;
        self.node_counts
            .resize(self.node_counts.len().max(required), 0);
        self.node_counts[new as usize] += 1;
        let old_count = self.node_counts[old as usize]
            .checked_sub(1)
            .expect("occupancy underflow: a node left a district with no counted nodes");
        self.node_counts[old as usize] = old_count;
        self.observed.insert(new);
        if old_count == 0 {
            self.observed.remove(old);
        }
    }

    pub(crate) fn observed(&self) -> &DistrictSet {
        &self.observed
    }

    pub(crate) fn is_empty(&self, district: u16) -> bool {
        self.node_counts
            .get(district as usize)
            .copied()
            .unwrap_or(0)
            == 0
    }
}

#[cfg(test)]
#[path = "../tests/district.rs"]
mod tests;
