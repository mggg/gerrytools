//! Shared full-scan and incremental drivers for hull-style district metrics.

use crate::geometry::robust_convex_hull;
use crate::scoring::delta::{apply_changes, expanded_assignment, validate_changes, DeltaChange};
use crate::scoring::district::{district_ids, observed_districts, DistrictOccupancy, DistrictSet};
use crate::{DistrictTable, Error, Result};
use geo::{Coord, Polygon};

/// Tolerance for ratio-style scores exceeding their 1.0 ceiling; shared by the hull ratios
/// (which clamp within it) and Polsby-Popper/Schwartzberg (which reject beyond it).
pub(crate) const RATIO_SCORE_EPS: f64 = 1e-9;
/// Looser tolerance for state-clipped ratios, where boolean clipping adds overlay noise.
pub(crate) const CLIPPED_RATIO_SCORE_EPS: f64 = 1e-8;

pub(crate) fn district_hull(points: &mut [Coord<f64>]) -> Polygon<f64> {
    robust_convex_hull(points)
}

/// Compact one dense score-per-district slab into a single-column result table.
pub(crate) fn district_score_table(observed: &DistrictSet, scores: &[f64]) -> DistrictTable {
    let districts = district_ids(observed);
    let values = districts
        .iter()
        .map(|&district| scores[district as usize])
        .collect();
    DistrictTable::new(districts, values, 1)
}

/// Per-district scoring hook for metrics driven by district membership and area.
///
/// Equal `(nodes, area, district)` arguments must return bit-identical scores regardless of
/// prior scratch use. The drivers below supply canonical nodes and area.
pub(crate) trait HullScorer: Sync {
    /// Reusable per-call scratch (point buffers, rng state).
    type Scratch: Send;

    /// Fresh scratch; `secondary` selects the second half of the two-district parallel path.
    fn scratch(&self, secondary: bool) -> Self::Scratch;

    fn node_count(&self) -> usize;

    fn unit_area(&self, node: usize) -> f64;

    /// Score one district from its member nodes and accumulated area.
    fn score_district(
        &self,
        nodes: &[usize],
        area: f64,
        district: u16,
        scratch: &mut Self::Scratch,
    ) -> Result<f64>;
}

/// Load `nodes` into `buffer` in ascending order and return the district's area.
///
/// Membership uses `swap_remove`, and floating-point addition is not associative. Canonical
/// ordering prevents update history and chunk boundaries from changing low bits or table hashes.
fn canonical_nodes<S: HullScorer>(metric: &S, buffer: &mut Vec<usize>, nodes: &[usize]) -> f64 {
    buffer.clear();
    buffer.extend_from_slice(nodes);
    buffer.sort_unstable();
    buffer.iter().map(|&node| metric.unit_area(node)).sum()
}

/// Score every district of one assignment from scratch.
pub(crate) fn full_scan_score<S: HullScorer>(
    metric: &S,
    assignment: &[u16],
) -> Result<DistrictTable> {
    if assignment.len() != metric.node_count() {
        return Err(Error::AssignmentLength {
            actual: assignment.len(),
            expected: metric.node_count(),
        });
    }

    let (district_slots, observed) = observed_districts(assignment);
    let districts = district_ids(&observed);
    let mut nodes_by_district = vec![Vec::new(); district_slots];
    for (node, &district) in assignment.iter().enumerate() {
        nodes_by_district[district as usize].push(node);
    }

    let mut scratch = metric.scratch(false);
    let mut sorted_nodes = Vec::new();
    let mut scores = Vec::with_capacity(districts.len());
    for &district in &districts {
        let area = canonical_nodes(
            metric,
            &mut sorted_nodes,
            &nodes_by_district[district as usize],
        );
        scores.push(metric.score_district(&sorted_nodes, area, district, &mut scratch)?);
    }
    Ok(DistrictTable::new(districts, scores, 1))
}

/// Incremental district-membership engine shared by the hull-style metrics.
pub(crate) struct IncrementalHullMetric<'a, S: HullScorer> {
    metric: &'a S,
    assignment: Vec<u16>,
    scores: Vec<f64>,
    occupancy: DistrictOccupancy,
    nodes_by_district: Vec<Vec<usize>>,
    node_position: Vec<usize>,
    scratch: S::Scratch,
    scratch_b: S::Scratch,
    /// Canonical buffers for the parallel two-district path.
    sorted_nodes: Vec<usize>,
    sorted_nodes_b: Vec<usize>,
}

impl<'a, S: HullScorer> IncrementalHullMetric<'a, S> {
    pub(crate) fn new(metric: &'a S, assignment: &[u16]) -> Result<Self> {
        let mut state = Self {
            metric,
            assignment: Vec::new(),
            scores: Vec::new(),
            occupancy: DistrictOccupancy::new(),
            nodes_by_district: Vec::new(),
            node_position: vec![0; metric.node_count()],
            scratch: metric.scratch(false),
            scratch_b: metric.scratch(true),
            sorted_nodes: Vec::new(),
            sorted_nodes_b: Vec::new(),
        };
        state.reset(assignment)?;
        Ok(state)
    }

    pub(crate) fn reset(&mut self, assignment: &[u16]) -> Result<()> {
        if assignment.len() != self.metric.node_count() {
            return Err(Error::AssignmentLength {
                actual: assignment.len(),
                expected: self.metric.node_count(),
            });
        }
        let (district_slots, _) = observed_districts(assignment);
        self.assignment.clear();
        self.assignment.extend_from_slice(assignment);
        self.scores.clear();
        self.scores.resize(district_slots, 0.0);
        self.occupancy.reset(assignment);
        self.nodes_by_district.clear();
        self.nodes_by_district.resize_with(district_slots, Vec::new);

        for (node, &district) in assignment.iter().enumerate() {
            let district = district as usize;
            self.node_position[node] = self.nodes_by_district[district].len();
            self.nodes_by_district[district].push(node);
        }

        for district in district_ids(self.occupancy.observed()) {
            let area = canonical_nodes(
                self.metric,
                &mut self.sorted_nodes,
                &self.nodes_by_district[district as usize],
            );
            self.scores[district as usize] = self.metric.score_district(
                &self.sorted_nodes,
                area,
                district,
                &mut self.scratch,
            )?;
        }
        Ok(())
    }

    /// Apply a delta whose `old` labels are validated against this state's current assignment.
    pub(crate) fn update(&mut self, changes: &[DeltaChange]) -> Result<()> {
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
        if let Some(assignment) = expanded_assignment(current, changes, self.scores.len()) {
            return self.reset(&assignment);
        }

        let mut touched = Vec::with_capacity(changes.len() * 2);
        for change in changes {
            if change.old == change.new {
                continue;
            }
            self.remove_node(change.node, change.old)?;
            self.add_node(change.node, change.new);
            self.occupancy.apply(change.old, change.new);
            touched.push(change.old as usize);
            touched.push(change.new as usize);
        }
        touched.sort_unstable();
        touched.dedup();
        let mut recompute = Vec::with_capacity(touched.len());
        for district in touched {
            if self.occupancy.is_empty(district as u16) {
                self.scores[district] = 0.0;
            } else {
                recompute.push(district as u16);
            }
        }
        self.recompute_scores(&recompute)
    }

    pub(crate) fn result(&self) -> DistrictTable {
        district_score_table(self.occupancy.observed(), &self.scores)
    }

    fn remove_node(&mut self, node: usize, district: u16) -> Result<()> {
        let position = self.node_position[node];
        let nodes = &mut self.nodes_by_district[district as usize];
        if nodes.get(position) != Some(&node) {
            return Err(Error::IncrementalNodeMembership { node, district });
        }
        nodes.swap_remove(position);
        if position < nodes.len() {
            self.node_position[nodes[position]] = position;
        }
        Ok(())
    }

    fn add_node(&mut self, node: usize, district: u16) {
        let district = district as usize;
        self.node_position[node] = self.nodes_by_district[district].len();
        self.nodes_by_district[district].push(node);
    }

    fn recompute_scores(&mut self, districts: &[u16]) -> Result<()> {
        if let [left, right] = districts {
            let metric = self.metric;
            let nodes = &self.nodes_by_district;
            let scratch = &mut self.scratch;
            let scratch_b = &mut self.scratch_b;
            let sorted_nodes = &mut self.sorted_nodes;
            let sorted_nodes_b = &mut self.sorted_nodes_b;
            let left_area = canonical_nodes(metric, sorted_nodes, &nodes[*left as usize]);
            let right_area = canonical_nodes(metric, sorted_nodes_b, &nodes[*right as usize]);
            let (left_score, right_score) = rayon::join(
                || metric.score_district(sorted_nodes, left_area, *left, scratch),
                || metric.score_district(sorted_nodes_b, right_area, *right, scratch_b),
            );
            self.scores[*left as usize] = left_score?;
            self.scores[*right as usize] = right_score?;
            return Ok(());
        }

        for &district in districts {
            let area = canonical_nodes(
                self.metric,
                &mut self.sorted_nodes,
                &self.nodes_by_district[district as usize],
            );
            self.scores[district as usize] = self.metric.score_district(
                &self.sorted_nodes,
                area,
                district,
                &mut self.scratch,
            )?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod district_hull_tests {
    use super::*;
    use geo::IsConvex;

    #[test]
    fn district_hull_does_not_retrace_an_interior_vertex() {
        let mut points = vec![
            Coord::from((-2064.3649325436318, 2730.896316839746)),
            Coord::from((-1522.348449763663, 2011.6161501336799)),
            Coord::from((-820.3014135364234, 1079.9682661743652)),
            Coord::from((808.8160174928399, 564.9877784054688)),
            Coord::from((1432.8319847267508, 1035.2175378433947)),
            Coord::from((2048.7533488451854, 1499.347576319679)),
            Coord::from((2518.2306617694326, 2438.24986430806)),
            Coord::from((717.8452181749713, 4827.442044150855)),
            Coord::from((1259.86170095494, 4108.161877444789)),
            Coord::from((717.8452181749713, 4827.442044150855)),
            Coord::from((-18.68430479830249, 5804.849733583088)),
            Coord::from((-2009.3284380240657, 4304.791784677448)),
            Coord::from((-2355.8226115960265, 4043.689696845276)),
        ];

        let hull = district_hull(&mut points);

        assert!(hull.exterior().is_strictly_ccw_convex());
        assert_eq!(
            hull.exterior()
                .0
                .iter()
                .filter(|&&point| point == Coord::from((717.8452181749713, 4827.442044150855)))
                .count(),
            1,
        );
    }
}
