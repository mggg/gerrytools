use super::hull_metric::RATIO_SCORE_EPS;
use crate::adjacency::{build_csr_adjacency, validate_edge_nodes, CsrAdjacency};
use crate::scoring::delta::{
    apply_changes, expanded_assignment, validate_changes, DeltaChange, GenerationStamps,
    PostDeltaLabels,
};
use crate::scoring::district::{district_ids, observed_districts, DistrictOccupancy};
use crate::{DistrictTable, Error, Result};

#[derive(Debug)]
/// Validated unit geometry and topology for district Polsby-Popper scores.
pub struct PreparedPolsbyPopper {
    node_count: usize,
    area_values: Vec<f64>,
    total_perimeter_values: Vec<f64>,
    edges: Vec<(u32, u32)>,
    shared_perimeters: Vec<f64>,
}

impl PreparedPolsbyPopper {
    /// Prepare the metric from unit areas, total perimeters, and shared edge perimeters.
    pub fn new(
        area_values: Vec<f64>,
        total_perimeter_values: Vec<f64>,
        edges: Vec<(u32, u32)>,
        shared_perimeters: Vec<f64>,
    ) -> Result<Self> {
        let node_count = area_values.len();
        validate_length(
            "total perimeter values",
            total_perimeter_values.len(),
            node_count,
        )?;
        validate_topology(node_count, &edges, &shared_perimeters)?;
        if area_values.iter().any(|value| !value.is_finite())
            || total_perimeter_values
                .iter()
                .any(|value| !value.is_finite())
            || shared_perimeters.iter().any(|value| !value.is_finite())
        {
            return Err(Error::NonFinitePolsbyPopperInput);
        }
        if let Some(node) = area_values.iter().position(|&area| area <= 0.0) {
            return Err(Error::InvalidInput(format!(
                "unit {node} has nonpositive area {}",
                area_values[node]
            )));
        }
        if let Some(node) = total_perimeter_values
            .iter()
            .position(|&perimeter| perimeter <= 0.0)
        {
            return Err(Error::InvalidInput(format!(
                "unit {node} has nonpositive total perimeter {}",
                total_perimeter_values[node]
            )));
        }
        if let Some(edge) = shared_perimeters.iter().position(|&shared| shared < 0.0) {
            return Err(Error::InvalidInput(format!(
                "edge {edge} has negative shared perimeter {}",
                shared_perimeters[edge]
            )));
        }
        let mut shared_by_node = vec![0.0; node_count];
        for (edge_index, &(u, v)) in edges.iter().enumerate() {
            let shared = shared_perimeters[edge_index];
            shared_by_node[u as usize] += shared;
            shared_by_node[v as usize] += shared;
        }
        if let Some(node) = shared_by_node
            .iter()
            .enumerate()
            .position(|(node, &shared)| {
                shared - total_perimeter_values[node]
                    > total_perimeter_values[node] * RATIO_SCORE_EPS
            })
        {
            return Err(Error::InvalidInput(format!(
                concat!(
                    "graph perimeter attributes are inconsistent: unit {} has total perimeter ",
                    "{}, but the shared perimeters of all graph edges connected to it sum to {}; ",
                    "the attributes may have been measured from different geometries or CRSs. ",
                    "Recompute every graph perimeter attribute from the same geometry in the ",
                    "same projected CRS, or use geometry-backed scoring"
                ),
                node, total_perimeter_values[node], shared_by_node[node]
            )));
        }

        Ok(Self {
            node_count,
            area_values,
            total_perimeter_values,
            edges,
            shared_perimeters,
        })
    }

    /// Prepare the metric from external boundary and shared edge perimeters.
    pub fn from_boundary_perimeters(
        area_values: Vec<f64>,
        boundary_perimeters: Vec<f64>,
        edges: Vec<(u32, u32)>,
        shared_perimeters: Vec<f64>,
    ) -> Result<Self> {
        let node_count = area_values.len();
        validate_length(
            "boundary perimeter values",
            boundary_perimeters.len(),
            node_count,
        )?;
        validate_topology(node_count, &edges, &shared_perimeters)?;
        if boundary_perimeters.iter().any(|value| !value.is_finite()) {
            return Err(Error::NonFinitePolsbyPopperInput);
        }
        if let Some(node) = boundary_perimeters
            .iter()
            .position(|&perimeter| perimeter < 0.0)
        {
            return Err(Error::InvalidInput(format!(
                "unit {node} has negative boundary perimeter {}",
                boundary_perimeters[node]
            )));
        }

        let mut total_perimeter_values = boundary_perimeters;
        for (edge_index, &(u, v)) in edges.iter().enumerate() {
            let shared = shared_perimeters[edge_index];
            total_perimeter_values[u as usize] += shared;
            total_perimeter_values[v as usize] += shared;
        }

        Self::new(
            area_values,
            total_perimeter_values,
            edges,
            shared_perimeters,
        )
    }

    /// Return the required assignment length.
    pub fn node_count(&self) -> usize {
        self.node_count
    }

    /// Create incremental Polsby-Popper state for an initial assignment.
    pub fn incremental(&self, assignment: &[u16]) -> Result<IncrementalPolsbyPopper<'_>> {
        IncrementalPolsbyPopper::new(self, assignment)
    }

    /// Score every observed district in an assignment.
    pub fn score(&self, assignment: &[u16]) -> Result<DistrictTable> {
        if assignment.len() != self.node_count {
            return Err(Error::AssignmentLength {
                actual: assignment.len(),
                expected: self.node_count,
            });
        }

        let (district_slots, observed) = observed_districts(assignment);
        let districts = district_ids(&observed);
        let mut area_by_district = vec![0.0; district_slots];
        let mut perimeter_by_district = vec![0.0; district_slots];

        for (node, &district) in assignment.iter().enumerate() {
            let district = district as usize;
            area_by_district[district] += self.area_values[node];
            perimeter_by_district[district] += self.total_perimeter_values[node];
        }

        for (edge_index, &(u, v)) in self.edges.iter().enumerate() {
            let district_u = assignment[u as usize] as usize;
            if district_u == assignment[v as usize] as usize {
                perimeter_by_district[district_u] -= 2.0 * self.shared_perimeters[edge_index];
            }
        }

        let mut scores = Vec::with_capacity(districts.len());
        for &district in &districts {
            let district_index = district as usize;
            scores.push(polsby_popper_score(
                district,
                area_by_district[district_index],
                perimeter_by_district[district_index],
            )?);
        }

        Ok(DistrictTable::new(districts, scores, 1))
    }
}

fn polsby_popper_score(district: u16, area: f64, perimeter: f64) -> Result<f64> {
    if perimeter <= 0.0 {
        return Err(Error::NonPositiveDistrictPerimeter {
            district,
            perimeter,
        });
    }
    let score = 4.0 * std::f64::consts::PI * area / perimeter.powi(2);
    // Graph inputs are caller-supplied, so a score meaningfully above 1 means the areas and
    // perimeters are mutually inconsistent, not float noise.
    if score > 1.0 + RATIO_SCORE_EPS {
        return Err(Error::ImpossibleScore {
            metric: "Polsby-Popper score",
            district,
            score,
        });
    }
    Ok(score)
}

/// District Polsby-Popper scores maintained across assignment changes.
pub struct IncrementalPolsbyPopper<'a> {
    metric: &'a PreparedPolsbyPopper,
    adjacency: CsrAdjacency,
    assignment: Vec<u16>,
    area_by_district: Vec<f64>,
    perimeter_by_district: Vec<f64>,
    occupancy: DistrictOccupancy,
    post_delta_labels: PostDeltaLabels,
    seen_edges: GenerationStamps,
}

impl<'a> IncrementalPolsbyPopper<'a> {
    fn new(metric: &'a PreparedPolsbyPopper, assignment: &[u16]) -> Result<Self> {
        let mut state = Self {
            metric,
            adjacency: build_csr_adjacency(metric.node_count, &metric.edges),
            assignment: Vec::new(),
            area_by_district: Vec::new(),
            perimeter_by_district: Vec::new(),
            occupancy: DistrictOccupancy::new(),
            post_delta_labels: PostDeltaLabels::new(metric.node_count),
            seen_edges: GenerationStamps::new(metric.edges.len()),
        };
        state.reset(assignment)?;
        Ok(state)
    }

    /// Replace the assignment and recompute every score from scratch.
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
        self.area_by_district.clear();
        self.area_by_district.resize(district_slots, 0.0);
        self.perimeter_by_district.clear();
        self.perimeter_by_district.resize(district_slots, 0.0);
        self.occupancy.reset(assignment);

        for (node, &district) in assignment.iter().enumerate() {
            let district = district as usize;
            self.area_by_district[district] += self.metric.area_values[node];
            self.perimeter_by_district[district] += self.metric.total_perimeter_values[node];
        }
        for (edge_index, &(u, v)) in self.metric.edges.iter().enumerate() {
            let district_u = assignment[u as usize] as usize;
            if district_u == assignment[v as usize] as usize {
                self.perimeter_by_district[district_u] -=
                    2.0 * self.metric.shared_perimeters[edge_index];
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

    pub(crate) fn update_trusted(
        &mut self,
        canonical_assignment: Option<&[u16]>,
        changes: &[DeltaChange],
    ) -> Result<()> {
        let current = canonical_assignment.unwrap_or(&self.assignment);
        if let Some(assignment) = expanded_assignment(current, changes, self.area_by_district.len())
        {
            return self.reset(&assignment);
        }

        for change in changes {
            if change.old == change.new {
                continue;
            }
            let old = change.old as usize;
            let new = change.new as usize;
            self.occupancy.apply(change.old, change.new);
            self.area_by_district[old] -= self.metric.area_values[change.node];
            self.area_by_district[new] += self.metric.area_values[change.node];
            self.perimeter_by_district[old] -= self.metric.total_perimeter_values[change.node];
            self.perimeter_by_district[new] += self.metric.total_perimeter_values[change.node];
        }

        self.post_delta_labels.refresh(changes);
        self.seen_edges.advance();
        let before = canonical_assignment.unwrap_or(&self.assignment);
        for change in changes {
            let incident = self.adjacency.offsets[change.node] as usize
                ..self.adjacency.offsets[change.node + 1] as usize;
            for adjacency_index in incident {
                let edge_index = self.adjacency.edge_indices[adjacency_index] as usize;
                if self.seen_edges.is_marked(edge_index) {
                    continue;
                }
                self.seen_edges.mark(edge_index);

                let (u, v) = self.metric.edges[edge_index];
                let u = u as usize;
                let v = v as usize;
                let before_u = before[u] as usize;
                let before_v = before[v] as usize;
                let after_u = self.post_delta_labels.label(before, u) as usize;
                let after_v = self.post_delta_labels.label(before, v) as usize;
                let shared = self.metric.shared_perimeters[edge_index];
                if before_u == before_v {
                    self.perimeter_by_district[before_u] += 2.0 * shared;
                }
                if after_u == after_v {
                    self.perimeter_by_district[after_u] -= 2.0 * shared;
                }
            }
        }
        Ok(())
    }

    /// Return the current score for every observed district.
    pub fn result(&self) -> Result<DistrictTable> {
        let districts = district_ids(self.occupancy.observed());
        let scores = districts
            .iter()
            .map(|&district| {
                let index = district as usize;
                polsby_popper_score(
                    district,
                    self.area_by_district[index],
                    self.perimeter_by_district[index],
                )
            })
            .collect::<Result<_>>()?;
        Ok(DistrictTable::new(districts, scores, 1))
    }
}

fn validate_length(input: &'static str, actual: usize, expected: usize) -> Result<()> {
    if actual != expected {
        return Err(Error::NumericInputLength {
            input,
            actual,
            expected,
        });
    }
    Ok(())
}

fn validate_topology(
    node_count: usize,
    edges: &[(u32, u32)],
    shared_perimeters: &[f64],
) -> Result<()> {
    if shared_perimeters.len() != edges.len() {
        return Err(Error::SharedPerimeterCount {
            actual: shared_perimeters.len(),
            expected: edges.len(),
        });
    }
    validate_edge_nodes(node_count, edges)
}

#[cfg(test)]
#[path = "../tests/polsby_popper.rs"]
mod tests;
