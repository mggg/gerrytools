use super::schwartzberg::schwartzberg_value;
use crate::{DeltaChange, DistrictTable, IncrementalPolsbyPopper, PreparedPolsbyPopper, Result};

/// Prepared Polsby-Popper and Schwartzberg scores sharing area and perimeter inputs.
pub struct PreparedAreaPerimeterMetrics {
    polsby_popper: PreparedPolsbyPopper,
}

impl PreparedAreaPerimeterMetrics {
    /// Prepare both metrics from unit areas, total perimeters, and shared edge perimeters.
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

    /// Prepare both metrics from external boundary and shared edge perimeters.
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

    /// Decode projected unit geometries and prepare both metrics.
    pub fn from_wkb<W: AsRef<[u8]>>(rows: &[W], edges: Vec<(u32, u32)>) -> Result<Self> {
        Ok(Self {
            polsby_popper: PreparedPolsbyPopper::from_wkb(rows, edges)?,
        })
    }

    /// Return the required assignment length.
    pub fn node_count(&self) -> usize {
        self.polsby_popper.node_count()
    }

    /// Create incremental state for an initial assignment.
    pub fn incremental(&self, assignment: &[u16]) -> Result<IncrementalAreaPerimeterMetrics<'_>> {
        Ok(IncrementalAreaPerimeterMetrics {
            polsby_popper: self.polsby_popper.incremental(assignment)?,
        })
    }

    /// Score Polsby-Popper followed by Schwartzberg for every observed district.
    pub fn score(&self, assignment: &[u16]) -> Result<DistrictTable> {
        combine(self.polsby_popper.score(assignment)?)
    }
}

/// Polsby-Popper and Schwartzberg scores maintained across assignment changes.
pub struct IncrementalAreaPerimeterMetrics<'a> {
    polsby_popper: IncrementalPolsbyPopper<'a>,
}

impl IncrementalAreaPerimeterMetrics<'_> {
    /// Replace the assignment and recompute both scores from scratch.
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

    /// Return current Polsby-Popper and Schwartzberg columns.
    pub fn result(&self) -> Result<DistrictTable> {
        combine(self.polsby_popper.result()?)
    }
}

fn combine(table: DistrictTable) -> Result<DistrictTable> {
    let polsby = table
        .column(0)
        .expect("Polsby-Popper always returns one column");
    let mut values = Vec::with_capacity(polsby.len() * 2);
    values.extend_from_slice(polsby);
    values.extend(
        polsby
            .iter()
            .map(|&score| schwartzberg_value(score))
            .collect::<Result<Vec<_>>>()?,
    );
    Ok(DistrictTable::new(table.district_ids().to_vec(), values, 2))
}
