use crate::adjacency::{build_csr_adjacency, validate_edge_nodes, CsrAdjacency};
use crate::scoring::delta::{
    apply_changes, validate_changes, DeltaChange, GenerationStamps, PostDeltaLabels,
};
use crate::scoring::district::{district_ids, observed_districts, DistrictOccupancy};
use crate::{Error, PlanTable, Result};

// MkvChain stores full assignments, but Scorer derives changes between adjacent frames in each
// batch. Here a dense transition means that the sum of degrees of changed nodes is large relative
// to |E|; an edge with two changed endpoints is counted twice because the incremental loop visits
// it twice. I (Peter) ran benchmarks over 15 states and found that a full edge scan is faster once
// this estimate reaches ~1/4 of |E|.
const MKV_FULL_RESCAN_WORK_MULTIPLIER: usize = 4;

#[derive(Debug)]
/// A validated graph whose cut-edge count or weight can be scored by plan.
pub struct PreparedCutEdges {
    node_count: usize,
    edges: Vec<(u32, u32)>,
    edge_weights: Option<Vec<f64>>,
}

impl PreparedCutEdges {
    /// Prepare an unweighted graph, where every cut edge contributes one.
    pub fn new(node_count: usize, edges: Vec<(u32, u32)>) -> Result<Self> {
        Self::prepare(node_count, edges, None)
    }

    /// Prepare a graph with one finite contribution per edge.
    pub fn weighted(
        node_count: usize,
        edges: Vec<(u32, u32)>,
        edge_weights: Vec<f64>,
    ) -> Result<Self> {
        Self::prepare(node_count, edges, Some(edge_weights))
    }

    fn prepare(
        node_count: usize,
        edges: Vec<(u32, u32)>,
        edge_weights: Option<Vec<f64>>,
    ) -> Result<Self> {
        if let Some(weights) = &edge_weights {
            if weights.len() != edges.len() {
                return Err(Error::EdgeWeightCount {
                    actual: weights.len(),
                    expected: edges.len(),
                });
            }
            if let Some(edge) = weights.iter().position(|weight| !weight.is_finite()) {
                return Err(Error::NonFiniteEdgeWeight { edge });
            }
        }
        validate_edge_nodes(node_count, &edges)?;

        Ok(Self {
            node_count,
            edges,
            edge_weights,
        })
    }

    /// Return the required assignment length.
    pub fn node_count(&self) -> usize {
        self.node_count
    }

    /// Create incremental cut-edge state for an initial assignment.
    pub fn incremental(&self, assignment: &[u16]) -> Result<IncrementalCutEdges<'_>> {
        IncrementalCutEdges::new(self, assignment)
    }

    /// Return the cut-edge count or total cut-edge weight for an assignment.
    pub fn score(&self, assignment: &[u16]) -> Result<PlanTable> {
        if assignment.len() != self.node_count {
            return Err(Error::AssignmentLength {
                actual: assignment.len(),
                expected: self.node_count,
            });
        }
        let (_, observed) = observed_districts(assignment);
        let value = match &self.edge_weights {
            Some(weights) => self
                .edges
                .iter()
                .zip(weights)
                .filter(|((u, v), _)| assignment[*u as usize] != assignment[*v as usize])
                .map(|(_, weight)| weight)
                .sum(),
            None => self
                .edges
                .iter()
                .filter(|&&(u, v)| assignment[u as usize] != assignment[v as usize])
                .count() as f64,
        };
        PlanTable::new(district_ids(&observed), vec![value])
    }
}

/// A cut-edge score maintained across assignment changes.
pub struct IncrementalCutEdges<'a> {
    metric: &'a PreparedCutEdges,
    adjacency: CsrAdjacency,
    assignment: Vec<u16>,
    cut_value: f64,
    occupancy: DistrictOccupancy,
    post_delta_labels: PostDeltaLabels,
    seen_edges: GenerationStamps,
}

impl<'a> IncrementalCutEdges<'a> {
    fn new(metric: &'a PreparedCutEdges, assignment: &[u16]) -> Result<Self> {
        let mut state = Self {
            metric,
            adjacency: build_csr_adjacency(metric.node_count, &metric.edges),
            assignment: Vec::new(),
            cut_value: 0.0,
            occupancy: DistrictOccupancy::new(),
            post_delta_labels: PostDeltaLabels::new(metric.node_count),
            seen_edges: GenerationStamps::new(metric.edges.len()),
        };
        state.reset(assignment)?;
        Ok(state)
    }

    /// Replace the assignment and recompute the score from scratch.
    pub fn reset(&mut self, assignment: &[u16]) -> Result<()> {
        let result = self.metric.score(assignment)?;
        self.assignment.clear();
        self.assignment.extend_from_slice(assignment);
        self.cut_value = result.values()[0];
        self.occupancy.reset(assignment);
        Ok(())
    }

    /// True when the estimated incremental work makes a full edge rescan cheaper.
    fn use_full_rescan(&self, changes: &[DeltaChange]) -> bool {
        let incident_visits = changes
            .iter()
            .filter(|change| change.old != change.new)
            .map(|change| {
                (self.adjacency.offsets[change.node + 1] - self.adjacency.offsets[change.node])
                    as usize
            })
            .sum::<usize>();
        !self.metric.edges.is_empty()
            && incident_visits.saturating_mul(MKV_FULL_RESCAN_WORK_MULTIPLIER)
                >= self.metric.edges.len()
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
        for change in changes {
            if change.old == change.new {
                continue;
            }
            self.occupancy.apply(change.old, change.new);
        }
        self.post_delta_labels.refresh(changes);

        if self.use_full_rescan(changes) {
            let before = canonical_assignment.unwrap_or(&self.assignment);
            let labels = &self.post_delta_labels;
            let is_cut = |u: u32, v: u32| {
                labels.label(before, u as usize) != labels.label(before, v as usize)
            };
            self.cut_value = match &self.metric.edge_weights {
                Some(weights) => self
                    .metric
                    .edges
                    .iter()
                    .zip(weights)
                    .filter(|((u, v), _)| is_cut(*u, *v))
                    .map(|(_, weight)| weight)
                    .sum(),
                None => self
                    .metric
                    .edges
                    .iter()
                    .filter(|&&(u, v)| is_cut(u, v))
                    .count() as f64,
            };
            return Ok(());
        }

        self.seen_edges.advance();
        let before = canonical_assignment.unwrap_or(&self.assignment);
        let mut cut_value = self.cut_value;
        for change in changes {
            let incident = self.adjacency.offsets[change.node] as usize
                ..self.adjacency.offsets[change.node + 1] as usize;
            for adjacency_index in incident {
                let edge = self.adjacency.edge_indices[adjacency_index] as usize;
                if self.seen_edges.is_marked(edge) {
                    continue;
                }
                self.seen_edges.mark(edge);
                let (u, v) = self.metric.edges[edge];
                let u = u as usize;
                let v = v as usize;
                let was_cut = before[u] != before[v];
                let is_cut = self.post_delta_labels.label(before, u)
                    != self.post_delta_labels.label(before, v);
                if was_cut != is_cut {
                    let weight = self
                        .metric
                        .edge_weights
                        .as_ref()
                        .map_or(1.0, |weights| weights[edge]);
                    cut_value += if is_cut { weight } else { -weight };
                }
            }
        }
        self.cut_value = cut_value;
        Ok(())
    }

    /// Return the current cut-edge score.
    pub fn result(&self) -> PlanTable {
        PlanTable::new(
            district_ids(self.occupancy.observed()),
            vec![self.cut_value],
        )
        .expect("cut-edge state always has one plan value")
    }
}

#[cfg(test)]
#[path = "../tests/cut_edges.rs"]
mod tests;
