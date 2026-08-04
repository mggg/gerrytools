use super::hull_metric::{
    district_hull, full_scan_score, HullScorer, IncrementalHullMetric, RATIO_SCORE_EPS,
};
use crate::scoring::delta::DeltaChange;
use crate::{DistrictTable, Error, PreparedUnitHulls, Result, UnitHull};
use geo::{Area, Coord};
use std::sync::Arc;

#[derive(Debug)]
/// Prepared unit hulls for district area-to-convex-hull-area ratios.
pub struct PreparedConvexHullRatio {
    unit_hulls: Arc<PreparedUnitHulls>,
}

impl PreparedConvexHullRatio {
    /// Decode ordered, projected Polygon or MultiPolygon WKB rows into a prepared metric.
    pub fn from_wkb<W: AsRef<[u8]>>(rows: &[W]) -> Result<Self> {
        Ok(Self::from_unit_hulls(Arc::new(
            PreparedUnitHulls::from_wkb(rows)?,
        )))
    }

    /// Validate explicit unit hulls and prepare the metric.
    pub fn new(units: Vec<UnitHull>) -> Result<Self> {
        Ok(Self::from_unit_hulls(Arc::new(PreparedUnitHulls::new(
            units,
        )?)))
    }

    /// Prepare the metric from shared, already-validated unit hulls.
    pub fn from_unit_hulls(unit_hulls: Arc<PreparedUnitHulls>) -> Self {
        Self { unit_hulls }
    }

    /// Return a shared handle to the prepared unit hulls.
    pub fn unit_hulls(&self) -> Arc<PreparedUnitHulls> {
        Arc::clone(&self.unit_hulls)
    }

    /// Return the required assignment length.
    pub fn node_count(&self) -> usize {
        self.unit_hulls.node_count()
    }

    /// Create incremental convex-hull-ratio state for an initial assignment.
    pub fn incremental(&self, assignment: &[u16]) -> Result<IncrementalConvexHullRatio<'_>> {
        Ok(IncrementalConvexHullRatio(IncrementalHullMetric::new(
            self, assignment,
        )?))
    }

    /// Score every observed district in an assignment.
    pub fn score(&self, assignment: &[u16]) -> Result<DistrictTable> {
        full_scan_score(self, assignment)
    }
}

impl HullScorer for PreparedConvexHullRatio {
    type Scratch = Vec<Coord<f64>>;

    fn scratch(&self, _secondary: bool) -> Self::Scratch {
        Vec::new()
    }

    fn node_count(&self) -> usize {
        self.unit_hulls.node_count()
    }

    fn unit_area(&self, node: usize) -> f64 {
        self.unit_hulls.unit_area(node)
    }

    fn score_district(
        &self,
        nodes: &[usize],
        area: f64,
        district: u16,
        points: &mut Self::Scratch,
    ) -> Result<f64> {
        points.clear();
        for &node in nodes {
            points.extend(
                self.unit_hulls
                    .unit_hull_points(node)
                    .iter()
                    .map(|point| Coord {
                        x: point.x,
                        y: point.y,
                    }),
            );
        }
        convex_hull_ratio(points, area, district)
    }
}

/// District convex-hull ratios maintained across assignment changes.
pub struct IncrementalConvexHullRatio<'a>(IncrementalHullMetric<'a, PreparedConvexHullRatio>);

impl IncrementalConvexHullRatio<'_> {
    /// Replace the assignment and recompute every score from scratch.
    pub fn reset(&mut self, assignment: &[u16]) -> Result<()> {
        self.0.reset(assignment)
    }

    /// Apply a delta whose `old` labels are validated against this state's current assignment.
    pub fn update(&mut self, changes: &[DeltaChange]) -> Result<()> {
        self.0.update(changes)
    }

    pub(crate) fn update_trusted(
        &mut self,
        canonical_assignment: Option<&[u16]>,
        changes: &[DeltaChange],
    ) -> Result<()> {
        self.0.update_trusted(canonical_assignment, changes)
    }

    /// Return the current score for every observed district.
    pub fn result(&self) -> DistrictTable {
        self.0.result()
    }
}

fn convex_hull_ratio(points: &mut [Coord<f64>], district_area: f64, district: u16) -> Result<f64> {
    if !district_area.is_finite() || district_area <= 0.0 {
        return Err(Error::InvalidDistrictArea {
            district,
            area: district_area,
        });
    }
    let hull_area = district_hull(points).unsigned_area();
    if !hull_area.is_finite() || hull_area <= 0.0 {
        return Err(Error::InvalidEnclosureArea {
            metric: "convex-hull",
            district,
            area: hull_area,
        });
    }

    let score = district_area / hull_area;
    if score > 1.0 + RATIO_SCORE_EPS {
        return Err(Error::ImpossibleScore {
            metric: "convex-hull ratio",
            district,
            score,
        });
    }
    Ok(score.min(1.0))
}

#[cfg(test)]
#[path = "../tests/convex_hull_ratio.rs"]
mod tests;
