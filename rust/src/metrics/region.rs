use super::region_parts::{PartColumnState, MAX_PACKED_REGIONS};
use crate::adjacency::{build_csr_adjacency, validate_edge_nodes, CsrAdjacency};
use crate::scoring::delta::{apply_changes, expanded_assignment, validate_changes, DeltaChange};
use crate::scoring::district::{district_ids, observed_districts, DistrictOccupancy};
use crate::{Error, PlanTable, Result};
use std::collections::HashMap;

pub(crate) const NO_REGION: usize = usize::MAX;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RegionStatistic {
    Splits,
    Pieces,
    Parts,
}

#[derive(Debug)]
struct RegionColumn {
    regions: Vec<usize>,
    region_count: usize,
}

#[derive(Debug)]
/// Prepared categorical region columns for split, piece, or connected-part counts.
pub struct PreparedRegion {
    columns: Vec<RegionColumn>,
    node_count: usize,
    statistic: RegionStatistic,
    adjacency: Option<CsrAdjacency>,
}

impl PreparedRegion {
    /// Count regions assigned to more than one district.
    pub fn splits(columns: Vec<Vec<Option<u32>>>) -> Result<Self> {
        Self::prepare(columns, RegionStatistic::Splits, vec![])
    }

    /// Count occupied `(region, district)` pairs without testing their connectivity.
    pub fn pieces(columns: Vec<Vec<Option<u32>>>) -> Result<Self> {
        Self::prepare(columns, RegionStatistic::Pieces, vec![])
    }

    /// Count connected `(region, district)` components using the supplied graph edges.
    pub fn parts(columns: Vec<Vec<Option<u32>>>, edges: Vec<(u32, u32)>) -> Result<Self> {
        Self::prepare(columns, RegionStatistic::Parts, edges)
    }

    fn prepare(
        columns: Vec<Vec<Option<u32>>>,
        statistic: RegionStatistic,
        edges: Vec<(u32, u32)>,
    ) -> Result<Self> {
        let Some(first) = columns.first() else {
            return Err(Error::EmptyRegionMetric);
        };
        let node_count = first.len();
        validate_edge_nodes(node_count, &edges)?;
        let mut prepared = Vec::with_capacity(columns.len());
        for (column_index, column) in columns.into_iter().enumerate() {
            if column.len() != node_count {
                return Err(Error::RegionColumnLength {
                    column: column_index,
                    actual: column.len(),
                    expected: node_count,
                });
            }
            let mut dense_ids = HashMap::new();
            let mut regions = Vec::with_capacity(node_count);
            for region in column {
                let next = dense_ids.len();
                regions.push(region.map_or(NO_REGION, |raw| *dense_ids.entry(raw).or_insert(next)));
            }
            if statistic == RegionStatistic::Parts
                && u64::try_from(dense_ids.len()).map_or(true, |count| count > MAX_PACKED_REGIONS)
            {
                return Err(Error::InvalidInput(format!(
                    "region part column {column_index} has too many distinct regions"
                )));
            }
            prepared.push(RegionColumn {
                regions,
                region_count: dense_ids.len(),
            });
        }
        Ok(Self {
            columns: prepared,
            node_count,
            statistic,
            adjacency: (statistic == RegionStatistic::Parts)
                .then(|| build_csr_adjacency(node_count, &edges)),
        })
    }

    /// Return the required assignment length.
    pub fn node_count(&self) -> usize {
        self.node_count
    }

    /// Return the number of prepared region columns.
    pub fn column_count(&self) -> usize {
        self.columns.len()
    }

    /// Create incremental region-statistic state for an initial assignment.
    pub fn incremental(&self, assignment: &[u16]) -> Result<IncrementalRegion<'_>> {
        IncrementalRegion::new(self, assignment)
    }

    /// Score every prepared region column for an assignment.
    pub fn score(&self, assignment: &[u16]) -> Result<PlanTable> {
        if assignment.len() != self.node_count {
            return Err(Error::AssignmentLength {
                actual: assignment.len(),
                expected: self.node_count,
            });
        }
        let (district_slots, observed) = observed_districts(assignment);
        let values = match self.statistic {
            RegionStatistic::Splits => self
                .columns
                .iter()
                .map(|column| score_splits(column, assignment, district_slots) as f64)
                .collect(),
            RegionStatistic::Pieces => self
                .columns
                .iter()
                .map(|column| score_pieces(column, assignment, district_slots) as f64)
                .collect(),
            RegionStatistic::Parts => {
                let adjacency = self
                    .adjacency
                    .as_ref()
                    .expect("prepared part metrics always have adjacency");
                self.columns
                    .iter()
                    .map(|column| score_parts(column, assignment, adjacency) as f64)
                    .collect()
            }
        };
        PlanTable::new(district_ids(&observed), values)
    }

    pub(crate) fn kind(&self) -> &'static str {
        match self.statistic {
            RegionStatistic::Splits => "region_splits",
            RegionStatistic::Pieces => "region_pieces",
            RegionStatistic::Parts => "region_parts",
        }
    }
}

fn occupied_districts(
    column: &RegionColumn,
    assignment: &[u16],
    district_slots: usize,
) -> (Vec<u64>, usize) {
    if column.region_count == 0 {
        return (Vec::new(), 1);
    }
    let words_per_region = district_slots.div_ceil(64).max(1);
    let mut bitset = vec![0_u64; column.region_count * words_per_region];
    for (node, region) in column.regions.iter().enumerate() {
        if *region != NO_REGION {
            let district = assignment[node] as usize;
            let word = region * words_per_region + district / 64;
            bitset[word] |= 1_u64 << (district % 64);
        }
    }
    (bitset, words_per_region)
}

fn score_splits(column: &RegionColumn, assignment: &[u16], district_slots: usize) -> usize {
    let (bitset, words_per_region) = occupied_districts(column, assignment, district_slots);

    (0..column.region_count)
        .filter(|&region| {
            let start = region * words_per_region;
            bitset[start..start + words_per_region]
                .iter()
                .map(|word| word.count_ones())
                .sum::<u32>()
                > 1
        })
        .count()
}

fn score_pieces(column: &RegionColumn, assignment: &[u16], district_slots: usize) -> usize {
    occupied_districts(column, assignment, district_slots)
        .0
        .iter()
        .map(|word| word.count_ones() as usize)
        .sum()
}

fn score_parts(column: &RegionColumn, assignment: &[u16], adjacency: &CsrAdjacency) -> usize {
    let mut seen = vec![false; assignment.len()];
    let mut stack = Vec::new();
    let mut parts = 0;

    for node in 0..assignment.len() {
        let region = column.regions[node];
        if region == NO_REGION {
            continue;
        }
        if seen[node] {
            continue;
        }
        parts += 1;
        seen[node] = true;
        stack.push(node);
        while let Some(current) = stack.pop() {
            let incident =
                adjacency.offsets[current] as usize..adjacency.offsets[current + 1] as usize;
            for adjacency_index in incident {
                let neighbor = adjacency.neighbors[adjacency_index] as usize;
                if !seen[neighbor]
                    && column.regions[neighbor] == region
                    && assignment[neighbor] == assignment[node]
                {
                    seen[neighbor] = true;
                    stack.push(neighbor);
                }
            }
        }
    }
    parts
}

struct OccupancyColumnState {
    counts: Vec<u32>,
    distinct_counts: Vec<u32>,
    region_count: usize,
    district_slots: usize,
    splits: u32,
    pieces: usize,
}

impl OccupancyColumnState {
    fn new(region_count: usize) -> Self {
        Self {
            counts: Vec::new(),
            distinct_counts: vec![0; region_count],
            region_count,
            district_slots: 0,
            splits: 0,
            pieces: 0,
        }
    }

    fn reset(&mut self, district_slots: usize) {
        self.district_slots = district_slots;
        self.counts.clear();
        self.counts.resize(self.region_count * district_slots, 0);
        self.distinct_counts.fill(0);
        self.splits = 0;
        self.pieces = 0;
    }

    fn add(&mut self, region: usize, district: u16) {
        let count = &mut self.counts[region * self.district_slots + district as usize];
        if *count == 0 {
            if self.distinct_counts[region] == 1 {
                self.splits += 1;
            }
            self.distinct_counts[region] += 1;
            self.pieces += 1;
        }
        *count += 1;
    }

    fn remove(&mut self, region: usize, district: u16) {
        let count = &mut self.counts[region * self.district_slots + district as usize];
        *count -= 1;
        if *count == 0 {
            if self.distinct_counts[region] == 2 {
                self.splits -= 1;
            }
            self.distinct_counts[region] -= 1;
            self.pieces -= 1;
        }
    }
}

enum RegionColumnState {
    Occupancy(OccupancyColumnState),
    Parts(PartColumnState),
}

impl RegionColumnState {
    fn value(&self, statistic: RegionStatistic) -> f64 {
        match self {
            Self::Occupancy(state) => match statistic {
                RegionStatistic::Splits => state.splits as f64,
                RegionStatistic::Pieces => state.pieces as f64,
                RegionStatistic::Parts => unreachable!("part metrics have component state"),
            },
            Self::Parts(state) => state.parts() as f64,
        }
    }
}

/// Region statistics maintained across assignment changes.
pub struct IncrementalRegion<'a> {
    metric: &'a PreparedRegion,
    assignment: Vec<u16>,
    columns: Vec<RegionColumnState>,
    district_slots: usize,
    occupancy: DistrictOccupancy,
}

impl<'a> IncrementalRegion<'a> {
    fn new(metric: &'a PreparedRegion, assignment: &[u16]) -> Result<Self> {
        let mut state = Self {
            metric,
            assignment: Vec::new(),
            columns: metric
                .columns
                .iter()
                .map(|column| match metric.statistic {
                    RegionStatistic::Splits | RegionStatistic::Pieces => {
                        RegionColumnState::Occupancy(OccupancyColumnState::new(column.region_count))
                    }
                    RegionStatistic::Parts => {
                        RegionColumnState::Parts(PartColumnState::new(metric.node_count))
                    }
                })
                .collect(),
            district_slots: 0,
            occupancy: DistrictOccupancy::new(),
        };
        state.reset(assignment)?;
        Ok(state)
    }

    /// Replace the assignment and recompute every statistic from scratch.
    pub fn reset(&mut self, assignment: &[u16]) -> Result<()> {
        if assignment.len() != self.metric.node_count {
            return Err(Error::AssignmentLength {
                actual: assignment.len(),
                expected: self.metric.node_count,
            });
        }
        let (district_slots, _) = observed_districts(assignment);
        self.assignment.clear();
        self.assignment.extend_from_slice(assignment);
        self.district_slots = district_slots;
        self.occupancy.reset(assignment);

        for (prepared, state) in self.metric.columns.iter().zip(&mut self.columns) {
            match state {
                RegionColumnState::Occupancy(state) => {
                    state.reset(district_slots);
                    for (node, &district) in assignment.iter().enumerate() {
                        let region = prepared.regions[node];
                        if region != NO_REGION {
                            state.add(region, district);
                        }
                    }
                }
                RegionColumnState::Parts(state) => {
                    state.reset(
                        &prepared.regions,
                        assignment,
                        self.metric
                            .adjacency
                            .as_ref()
                            .expect("prepared part metrics always have adjacency"),
                    );
                }
            }
        }
        Ok(())
    }

    /// Apply a delta whose `old` labels are validated against this state's current assignment.
    pub fn update(&mut self, changes: &[DeltaChange]) -> Result<()> {
        validate_changes(&self.assignment, changes)?;
        self.update_trusted(None, changes)?;
        apply_changes(&mut self.assignment, changes);
        Ok(())
    }

    /// Apply a delta already validated against this state's current assignment.
    pub(crate) fn update_trusted(
        &mut self,
        canonical_assignment: Option<&[u16]>,
        changes: &[DeltaChange],
    ) -> Result<()> {
        let current = canonical_assignment.unwrap_or(&self.assignment);
        if let Some(assignment) = expanded_assignment(current, changes, self.district_slots) {
            return self.reset(&assignment);
        }

        for change in changes {
            if change.old == change.new {
                continue;
            }
            self.occupancy.apply(change.old, change.new);
        }

        for (prepared, state) in self.metric.columns.iter().zip(&mut self.columns) {
            match state {
                RegionColumnState::Occupancy(state) => {
                    for change in changes {
                        if change.old == change.new {
                            continue;
                        }
                        let region = prepared.regions[change.node];
                        if region != NO_REGION {
                            state.remove(region, change.old);
                            state.add(region, change.new);
                        }
                    }
                }
                RegionColumnState::Parts(state) => state.update(
                    &prepared.regions,
                    self.metric
                        .adjacency
                        .as_ref()
                        .expect("prepared part metrics always have adjacency"),
                    changes,
                ),
            }
        }
        Ok(())
    }

    /// Return the current value for every prepared region column.
    pub fn result(&self) -> PlanTable {
        let values = self
            .columns
            .iter()
            .map(|column| column.value(self.metric.statistic))
            .collect();
        PlanTable::new(district_ids(self.occupancy.observed()), values)
            .expect("prepared region state always has at least one plan value")
    }
}

#[cfg(test)]
#[path = "../tests/region.rs"]
mod tests;
