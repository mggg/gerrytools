use crate::{
    DeltaChange, DistrictTable, Error, IncrementalPolsbyPopper, PreparedPolsbyPopper, Result,
};

/// Validated unit geometry and topology for district Schwartzberg scores.
pub struct PreparedSchwartzberg {
    polsby_popper: PreparedPolsbyPopper,
}

impl PreparedSchwartzberg {
    /// Prepare the metric from unit areas, total perimeters, and shared edge perimeters.
    pub fn new(
        area_values: Vec<f64>,
        total_perimeter_values: Vec<f64>,
        edges: Vec<(u32, u32)>,
        shared_perimeters: Vec<f64>,
    ) -> Result<Self> {
        Ok(Self {
            polsby_popper: PreparedPolsbyPopper::new(
                area_values,
                total_perimeter_values,
                edges,
                shared_perimeters,
            )?,
        })
    }

    /// Prepare the metric from external boundary and shared edge perimeters.
    pub fn from_boundary_perimeters(
        area_values: Vec<f64>,
        boundary_perimeters: Vec<f64>,
        edges: Vec<(u32, u32)>,
        shared_perimeters: Vec<f64>,
    ) -> Result<Self> {
        Ok(Self {
            polsby_popper: PreparedPolsbyPopper::from_boundary_perimeters(
                area_values,
                boundary_perimeters,
                edges,
                shared_perimeters,
            )?,
        })
    }

    /// Decode projected unit geometries and prepare the metric.
    pub fn from_wkb<W: AsRef<[u8]>>(rows: &[W], edges: Vec<(u32, u32)>) -> Result<Self> {
        Ok(Self {
            polsby_popper: PreparedPolsbyPopper::from_wkb(rows, edges)?,
        })
    }

    /// Return the required assignment length.
    pub fn node_count(&self) -> usize {
        self.polsby_popper.node_count()
    }

    /// Create incremental Schwartzberg state for an initial assignment.
    pub fn incremental(&self, assignment: &[u16]) -> Result<IncrementalSchwartzberg<'_>> {
        Ok(IncrementalSchwartzberg {
            polsby_popper: self.polsby_popper.incremental(assignment)?,
        })
    }

    /// Score every observed district in an assignment.
    pub fn score(&self, assignment: &[u16]) -> Result<DistrictTable> {
        transform(self.polsby_popper.score(assignment)?)
    }
}

/// District Schwartzberg scores maintained across assignment changes.
pub struct IncrementalSchwartzberg<'a> {
    polsby_popper: IncrementalPolsbyPopper<'a>,
}

impl IncrementalSchwartzberg<'_> {
    /// Replace the assignment and recompute every score from scratch.
    pub fn reset(&mut self, assignment: &[u16]) -> Result<()> {
        self.polsby_popper.reset(assignment)
    }

    /// Apply changes after validating their old labels against the current assignment.
    pub fn update(&mut self, changes: &[DeltaChange]) -> Result<()> {
        self.polsby_popper.update(changes)
    }

    pub(crate) fn update_trusted(
        &mut self,
        canonical_assignment: Option<&[u16]>,
        changes: &[DeltaChange],
    ) -> Result<()> {
        self.polsby_popper
            .update_trusted(canonical_assignment, changes)
    }

    /// Return the current score for every observed district.
    pub fn result(&self) -> Result<DistrictTable> {
        transform(self.polsby_popper.result()?)
    }
}

fn transform(table: DistrictTable) -> Result<DistrictTable> {
    let scores = table
        .column(0)
        .expect("Polsby-Popper always returns one column")
        .iter()
        .map(|&score| schwartzberg_value(score))
        .collect::<Result<_>>()?;
    Ok(DistrictTable::new(table.district_ids().to_vec(), scores, 1))
}

// Defensive: Polsby-Popper already rejects scores beyond this shared tolerance at its source.
pub(crate) fn schwartzberg_value(score: f64) -> Result<f64> {
    if score <= 0.0 {
        return Err(Error::InvalidInput(
            "Polsby-Popper scores must be positive for Schwartzberg".into(),
        ));
    }
    if score > 1.0 + super::hull_metric::RATIO_SCORE_EPS {
        return Err(Error::InvalidInput(
            "Polsby-Popper scores cannot exceed one for Schwartzberg".into(),
        ));
    }
    Ok(1.0 / score.min(1.0).sqrt())
}

#[cfg(test)]
#[path = "../tests/schwartzberg.rs"]
mod tests;
