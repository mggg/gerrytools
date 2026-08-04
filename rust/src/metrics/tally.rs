use crate::scoring::delta::{apply_changes, expanded_assignment, validate_changes, DeltaChange};
use crate::scoring::district::{district_ids, observed_districts, DistrictOccupancy};
use crate::{DistrictTable, Error, Result};

#[derive(Debug)]
/// Validated per-node numeric columns that can be summed by district.
pub struct PreparedTally {
    columns: Vec<Vec<f64>>,
    node_count: usize,
}

impl PreparedTally {
    /// Validate columns of equal length and prepare them for district aggregation.
    pub fn new(columns: Vec<Vec<f64>>) -> Result<Self> {
        let Some(first) = columns.first() else {
            return Err(Error::EmptyTally);
        };
        let node_count = first.len();

        for (column, values) in columns.iter().enumerate() {
            if values.len() != node_count {
                return Err(Error::TallyColumnLength {
                    column,
                    actual: values.len(),
                    expected: node_count,
                });
            }
            if values.iter().any(|value| !value.is_finite()) {
                return Err(Error::NonFiniteTallyValue { column });
            }
        }

        Ok(Self {
            columns,
            node_count,
        })
    }

    /// Return the required assignment length.
    pub fn node_count(&self) -> usize {
        self.node_count
    }

    /// Return the number of tally columns.
    pub fn column_count(&self) -> usize {
        self.columns.len()
    }

    /// Create incremental tally state for an initial assignment.
    pub fn incremental(&self, assignment: &[u16]) -> Result<IncrementalTally<'_>> {
        IncrementalTally::new(self, assignment)
    }

    /// Sum every column by the observed districts in `assignment`.
    pub fn score(&self, assignment: &[u16]) -> Result<DistrictTable> {
        if assignment.len() != self.node_count {
            return Err(Error::AssignmentLength {
                actual: assignment.len(),
                expected: self.node_count,
            });
        }

        let (district_slots, observed) = observed_districts(assignment);
        let districts = district_ids(&observed);
        let mut dense = vec![0.0; self.columns.len() * district_slots];

        for (column, values) in self.columns.iter().enumerate() {
            let offset = column * district_slots;
            for (node, &value) in values.iter().enumerate() {
                dense[offset + assignment[node] as usize] += value;
            }
        }

        let mut compact = Vec::with_capacity(self.columns.len() * districts.len());
        for column in 0..self.columns.len() {
            let offset = column * district_slots;
            compact.extend(
                districts
                    .iter()
                    .map(|&district| dense[offset + district as usize]),
            );
        }

        Ok(DistrictTable::new(districts, compact, self.columns.len()))
    }
}

/// District tallies maintained across assignment changes.
pub struct IncrementalTally<'a> {
    metric: &'a PreparedTally,
    assignment: Vec<u16>,
    totals: Vec<f64>,
    district_slots: usize,
    occupancy: DistrictOccupancy,
}

impl<'a> IncrementalTally<'a> {
    fn new(metric: &'a PreparedTally, assignment: &[u16]) -> Result<Self> {
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
        self.totals.clear();
        self.totals
            .resize(self.metric.columns.len() * district_slots, 0.0);
        self.occupancy.reset(assignment);

        for (node, &district) in assignment.iter().enumerate() {
            let district = district as usize;
            for (column, values) in self.metric.columns.iter().enumerate() {
                self.totals[column * district_slots + district] += values[node];
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
    ///
    /// Accepted drift bound: totals are updated with `+=`/`-=` and never resynced from scratch
    /// (TwoDelta snapshots verify the assignment, not the accumulators). Integer-valued tallies
    /// stay exact below 2^53; real-valued weights can drift over very long chains. The same
    /// trade-off applies to the other incremental accumulators (Polsby-Popper, cut edges).
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

            for (column, values) in self.metric.columns.iter().enumerate() {
                let offset = column * self.district_slots;
                let value = values[change.node];
                self.totals[offset + old] -= value;
                self.totals[offset + new] += value;
            }
        }
        Ok(())
    }

    /// Return the current tally columns for all observed districts.
    pub fn result(&self) -> DistrictTable {
        let districts = district_ids(self.occupancy.observed());
        let mut values = Vec::with_capacity(districts.len() * self.metric.columns.len());
        for column in 0..self.metric.columns.len() {
            let offset = column * self.district_slots;
            values.extend(
                districts
                    .iter()
                    .map(|&district| self.totals[offset + district as usize]),
            );
        }
        DistrictTable::new(districts, values, self.metric.columns.len())
    }
}

#[cfg(test)]
#[path = "../tests/tally.rs"]
mod tests;
