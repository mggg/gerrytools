use crate::scoring::delta::{apply_changes, expanded_assignment, validate_changes, DeltaChange};
use crate::scoring::district::{district_ids, observed_districts, DistrictOccupancy};
use crate::{DistrictTable, Error, Result};
use std::collections::HashMap;

const NO_REGION: usize = usize::MAX;

/// One or more district tallies grouped by a fixed region column.
#[derive(Debug)]
pub struct PreparedRegionTally {
    regions: Vec<usize>,
    region_ids: Vec<u32>,
    include_count: bool,
    values: Vec<Vec<f64>>,
}

impl PreparedRegionTally {
    /// Prepare optional unit counts and numeric sums grouped by region and district.
    ///
    /// Missing region IDs are excluded. Result columns use metric-major, then first-seen region
    /// order, with the count metric first when `include_count` is true.
    pub fn new(
        regions: Vec<Option<u32>>,
        include_count: bool,
        values: Vec<Vec<f64>>,
    ) -> Result<Self> {
        if !include_count && values.is_empty() {
            return Err(Error::InvalidInput(
                "tally-by-region requires a count or at least one value".into(),
            ));
        }
        for (value, column) in values.iter().enumerate() {
            if column.len() != regions.len() {
                return Err(Error::TallyByRegionValueLength {
                    value,
                    actual: column.len(),
                    expected: regions.len(),
                });
            }
            if let Some(node) = column.iter().position(|entry| !entry.is_finite()) {
                return Err(Error::NonFiniteTallyByRegionValue { value, node });
            }
        }

        let mut dense_ids = HashMap::new();
        let mut region_ids = Vec::new();
        let regions = regions
            .into_iter()
            .map(|region| {
                region.map_or(NO_REGION, |raw| {
                    if let Some(&dense) = dense_ids.get(&raw) {
                        dense
                    } else {
                        let dense = region_ids.len();
                        dense_ids.insert(raw, dense);
                        region_ids.push(raw);
                        dense
                    }
                })
            })
            .collect();
        Ok(Self {
            regions,
            region_ids,
            include_count,
            values,
        })
    }

    /// Return the required assignment length.
    pub fn node_count(&self) -> usize {
        self.regions.len()
    }

    /// Return the flattened metric-by-region result column count.
    pub fn column_count(&self) -> usize {
        self.metric_count() * self.region_count()
    }

    /// Return the number of count and value metrics.
    pub fn metric_count(&self) -> usize {
        usize::from(self.include_count) + self.values.len()
    }

    /// Return the number of distinct, non-missing regions.
    pub fn region_count(&self) -> usize {
        self.region_ids.len()
    }

    /// Raw region IDs in the first-seen order used for result columns.
    #[cfg(test)]
    fn region_ids(&self) -> &[u32] {
        &self.region_ids
    }

    /// Create incremental region-tally state for an initial assignment.
    pub fn incremental(&self, assignment: &[u16]) -> Result<IncrementalRegionTally<'_>> {
        IncrementalRegionTally::new(self, assignment)
    }

    /// Aggregate every prepared metric by region and observed district.
    pub fn score(&self, assignment: &[u16]) -> Result<DistrictTable> {
        if assignment.len() != self.node_count() {
            return Err(Error::AssignmentLength {
                actual: assignment.len(),
                expected: self.node_count(),
            });
        }
        let (district_slots, observed) = observed_districts(assignment);
        let districts = district_ids(&observed);
        let mut dense = vec![0.0; self.column_count() * district_slots];
        for (node, &region) in self.regions.iter().enumerate() {
            if region != NO_REGION {
                let district = assignment[node] as usize;
                if self.include_count {
                    dense[region * district_slots + district] += 1.0;
                }
                let first_value = usize::from(self.include_count);
                for (value, column) in self.values.iter().enumerate() {
                    let metric = first_value + value;
                    let offset = (metric * self.region_count() + region) * district_slots;
                    dense[offset + district] += column[node];
                }
            }
        }

        let mut values = Vec::with_capacity(self.column_count() * districts.len());
        for column in 0..self.column_count() {
            let offset = column * district_slots;
            values.extend(
                districts
                    .iter()
                    .map(|&district| dense[offset + district as usize]),
            );
        }
        Ok(DistrictTable::new(districts, values, self.column_count()))
    }
}

/// Region-by-district tallies maintained across assignment changes.
pub struct IncrementalRegionTally<'a> {
    metric: &'a PreparedRegionTally,
    assignment: Vec<u16>,
    totals: Vec<f64>,
    district_slots: usize,
    occupancy: DistrictOccupancy,
}

impl<'a> IncrementalRegionTally<'a> {
    fn new(metric: &'a PreparedRegionTally, assignment: &[u16]) -> Result<Self> {
        let mut state = Self {
            metric,
            assignment: Vec::new(),
            totals: Vec::new(),
            district_slots: 0,
            occupancy: DistrictOccupancy::new(),
        };
        state.reset(assignment)?;
        Ok(state)
    }

    /// Replace the assignment and recompute every tally from scratch.
    pub fn reset(&mut self, assignment: &[u16]) -> Result<()> {
        if assignment.len() != self.metric.node_count() {
            return Err(Error::AssignmentLength {
                actual: assignment.len(),
                expected: self.metric.node_count(),
            });
        }
        let (district_slots, _) = observed_districts(assignment);
        self.assignment.clear();
        self.assignment.extend_from_slice(assignment);
        self.district_slots = district_slots;
        self.totals.clear();
        self.totals
            .resize(self.metric.column_count() * district_slots, 0.0);
        self.occupancy.reset(assignment);

        for (node, &district) in assignment.iter().enumerate() {
            let region = self.metric.regions[node];
            if region != NO_REGION {
                if self.metric.include_count {
                    let offset = region * district_slots;
                    self.totals[offset + district as usize] += 1.0;
                }
                let first_value = usize::from(self.metric.include_count);
                for (value, column) in self.metric.values.iter().enumerate() {
                    let metric = first_value + value;
                    let offset = (metric * self.metric.region_count() + region) * district_slots;
                    self.totals[offset + district as usize] += column[node];
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
            let old = change.old as usize;
            let new = change.new as usize;
            self.occupancy.apply(change.old, change.new);

            let region = self.metric.regions[change.node];
            if region != NO_REGION {
                if self.metric.include_count {
                    let offset = region * self.district_slots;
                    self.totals[offset + old] -= 1.0;
                    self.totals[offset + new] += 1.0;
                }
                let first_value = usize::from(self.metric.include_count);
                for (value, column) in self.metric.values.iter().enumerate() {
                    let metric = first_value + value;
                    let offset =
                        (metric * self.metric.region_count() + region) * self.district_slots;
                    let entry = column[change.node];
                    self.totals[offset + old] -= entry;
                    self.totals[offset + new] += entry;
                }
            }
        }
        Ok(())
    }

    /// Return current metric-by-region columns for all observed districts.
    pub fn result(&self) -> DistrictTable {
        let districts = district_ids(self.occupancy.observed());
        let mut values = Vec::with_capacity(self.metric.column_count() * districts.len());
        for column in 0..self.metric.column_count() {
            let offset = column * self.district_slots;
            values.extend(
                districts
                    .iter()
                    .map(|&district| self.totals[offset + district as usize]),
            );
        }
        DistrictTable::new(districts, values, self.metric.column_count())
    }
}

#[cfg(test)]
#[path = "../tests/region_tally.rs"]
mod tests;
