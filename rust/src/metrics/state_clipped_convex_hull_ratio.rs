use super::hull_metric::{
    district_hull, full_scan_score, HullScorer, IncrementalHullMetric, CLIPPED_RATIO_SCORE_EPS,
};
use crate::geometry::precise_intersection_area;
use crate::scoring::delta::DeltaChange;
use crate::{DistrictTable, Error, PreparedUnitHulls, Result};
use geo::{
    Area, BoundingRect, Coord, Covers, MultiPolygon, Point, Polygon, Triangle, TriangulateEarcut,
};
use rstar::{RTree, AABB};
use std::sync::Arc;

#[derive(Debug)]
/// Prepared unit hulls and state geometry for clipped convex-hull ratios.
pub struct PreparedStateClippedConvexHullRatio {
    unit_hulls: Arc<PreparedUnitHulls>,
    state: MultiPolygon<f64>,
    state_triangles: RTree<Triangle<f64>>,
}

impl PreparedStateClippedConvexHullRatio {
    pub(crate) fn from_validated_parts(
        unit_hulls: Arc<PreparedUnitHulls>,
        state: MultiPolygon<f64>,
    ) -> Self {
        let state_triangles = state
            .iter()
            .flat_map(TriangulateEarcut::earcut_triangles)
            .collect();
        Self {
            unit_hulls,
            state,
            state_triangles: RTree::bulk_load(state_triangles),
        }
    }

    /// Return a shared handle to the prepared unit hulls.
    pub fn unit_hulls(&self) -> Arc<PreparedUnitHulls> {
        Arc::clone(&self.unit_hulls)
    }

    /// Return the required assignment length.
    pub fn node_count(&self) -> usize {
        self.unit_hulls.node_count()
    }

    /// Create incremental state-clipped convex-hull state for an initial assignment.
    pub fn incremental(
        &self,
        assignment: &[u16],
    ) -> Result<IncrementalStateClippedConvexHullRatio<'_>> {
        Ok(IncrementalStateClippedConvexHullRatio(
            IncrementalHullMetric::new(self, assignment)?,
        ))
    }

    /// Score every observed district in an assignment.
    pub fn score(&self, assignment: &[u16]) -> Result<DistrictTable> {
        full_scan_score(self, assignment)
    }
}

impl HullScorer for PreparedStateClippedConvexHullRatio {
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
        state_clipped_convex_hull_ratio(points, area, district, &self.state, &self.state_triangles)
    }
}

/// State-clipped district convex-hull ratios maintained across assignment changes.
pub struct IncrementalStateClippedConvexHullRatio<'a>(
    IncrementalHullMetric<'a, PreparedStateClippedConvexHullRatio>,
);

impl IncrementalStateClippedConvexHullRatio<'_> {
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

fn state_clipped_convex_hull_ratio(
    points: &mut [Coord<f64>],
    district_area: f64,
    district: u16,
    state: &MultiPolygon<f64>,
    state_triangles: &RTree<Triangle<f64>>,
) -> Result<f64> {
    if !district_area.is_finite() || district_area <= 0.0 {
        return Err(Error::InvalidDistrictArea {
            district,
            area: district_area,
        });
    }
    let hull = district_hull(points);
    let hull_area = hull.unsigned_area();
    // Avoid quantized boolean-overlay error for the common case where no clipping is needed.
    let clipped_area = if state.covers(&hull) {
        hull_area
    } else {
        let fast_area = clipped_area(&hull, state_triangles);
        if fast_area <= hull_area {
            fast_area
        } else {
            precise_intersection_area(&MultiPolygon(vec![hull]), state)?
        }
    };
    if !clipped_area.is_finite() || clipped_area <= 0.0 {
        return Err(Error::InvalidEnclosureArea {
            metric: "state-clipped convex-hull",
            district,
            area: clipped_area,
        });
    }

    let score = district_area / clipped_area;
    if score > 1.0 + CLIPPED_RATIO_SCORE_EPS {
        return Err(Error::ImpossibleScore {
            metric: "state-clipped convex-hull ratio",
            district,
            score,
        });
    }
    Ok(score.min(1.0))
}

fn clipped_area(hull: &Polygon<f64>, state_triangles: &RTree<Triangle<f64>>) -> f64 {
    // General polygon overlay quantizes against the state-wide extent. Clip local triangles in
    // f64 instead so projected coordinates do not lose district-scale precision.
    let bounds = hull
        .bounding_rect()
        .expect("a district hull contains positive-area geometry");
    let envelope = AABB::from_corners(
        Point::new(bounds.min().x, bounds.min().y),
        Point::new(bounds.max().x, bounds.max().y),
    );
    state_triangles
        .locate_in_envelope_intersecting(&envelope)
        .map(|triangle| triangle_hull_intersection_area(*triangle, hull.exterior().0.as_slice()))
        .sum()
}

fn triangle_hull_intersection_area(triangle: Triangle<f64>, hull: &[Coord<f64>]) -> f64 {
    let mut clipped = triangle.to_array().to_vec();
    for edge in hull.windows(2) {
        let input = std::mem::take(&mut clipped);
        let Some(mut previous) = input.last().copied() else {
            break;
        };
        let mut previous_distance = cross(edge[0], edge[1], previous);
        for current in input {
            let current_distance = cross(edge[0], edge[1], current);
            let previous_inside = previous_distance >= 0.0;
            let current_inside = current_distance >= 0.0;
            if previous_inside != current_inside {
                let fraction = previous_distance / (previous_distance - current_distance);
                clipped.push(Coord {
                    x: previous.x + fraction * (current.x - previous.x),
                    y: previous.y + fraction * (current.y - previous.y),
                });
            }
            if current_inside {
                clipped.push(current);
            }
            previous = current;
            previous_distance = current_distance;
        }
    }
    polygon_area(&clipped)
}

fn cross(start: Coord<f64>, end: Coord<f64>, point: Coord<f64>) -> f64 {
    (end.x - start.x) * (point.y - start.y) - (end.y - start.y) * (point.x - start.x)
}

fn polygon_area(vertices: &[Coord<f64>]) -> f64 {
    let Some(origin) = vertices.first() else {
        return 0.0;
    };
    vertices
        .iter()
        .zip(vertices.iter().cycle().skip(1))
        .take(vertices.len())
        .map(|(left, right)| {
            (left.x - origin.x) * (right.y - origin.y) - (right.x - origin.x) * (left.y - origin.y)
        })
        .sum::<f64>()
        .abs()
        * 0.5
}

#[cfg(test)]
#[path = "../tests/state_clipped_convex_hull_ratio.rs"]
mod tests;
