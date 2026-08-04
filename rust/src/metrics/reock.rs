use super::hull_metric::{full_scan_score, HullScorer, IncrementalHullMetric, RATIO_SCORE_EPS};
use crate::scoring::delta::DeltaChange;
use crate::{Coordinate, DistrictTable, Error, PreparedUnitHulls, Result, UnitHull};
use std::sync::Arc;

/// Relative tolerances for the circle predicates (containment, collinearity), scaled by the
/// squared coordinate extent of the centered point cloud so they stay meaningful at
/// projected-CRS magnitudes.
const IN_CIRCLE_REL_EPS: f64 = 1e-12;
const COLLINEAR_DET_REL_EPS: f64 = 1e-12;
/// Base seed for per-district Welzl shuffles.
const REOCK_SHUFFLE_SEED: u64 = 0x9e37_79b9_7f4a_7c15;

#[derive(Debug)]
/// Prepared unit hulls for district Reock compactness scores.
pub struct PreparedReock {
    unit_hulls: Arc<PreparedUnitHulls>,
}

impl PreparedReock {
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

    /// Create incremental Reock state for an initial assignment.
    pub fn incremental(&self, assignment: &[u16]) -> Result<IncrementalReock<'_>> {
        Ok(IncrementalReock(IncrementalHullMetric::new(
            self, assignment,
        )?))
    }

    /// Score every observed district in an assignment.
    pub fn score(&self, assignment: &[u16]) -> Result<DistrictTable> {
        full_scan_score(self, assignment)
    }
}

impl HullScorer for PreparedReock {
    type Scratch = (Vec<Coordinate>, fastrand::Rng);

    fn scratch(&self, _secondary: bool) -> Self::Scratch {
        (Vec::new(), fastrand::Rng::with_seed(REOCK_SHUFFLE_SEED))
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
        (points, rng): &mut Self::Scratch,
    ) -> Result<f64> {
        points.clear();
        for &node in nodes {
            points.extend_from_slice(self.unit_hulls.unit_hull_points(node));
        }
        // Fix the shuffle so scratch history, chunking, and label encoding cannot change low bits.
        *rng = fastrand::Rng::with_seed(REOCK_SHUFFLE_SEED);
        reock_score(points, area, district, rng)
    }
}

/// District Reock scores maintained across assignment changes.
pub struct IncrementalReock<'a>(IncrementalHullMetric<'a, PreparedReock>);

impl IncrementalReock<'_> {
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

#[derive(Debug, Clone, Copy)]
struct Circle {
    center: Coordinate,
    radius: f64,
}

#[inline]
fn l2_sq_dist(p: Coordinate, q: Coordinate) -> f64 {
    (p.x - q.x).powi(2) + (p.y - q.y).powi(2)
}

fn diameter_circle(p: Coordinate, q: Coordinate) -> Circle {
    let center = Coordinate {
        x: (p.x + q.x) / 2.0,
        y: (p.y + q.y) / 2.0,
    };
    let radius = l2_sq_dist(p, q).sqrt() / 2.0;
    Circle { center, radius }
}

fn largest_diameter_circle(p: Coordinate, q: Coordinate, r: Coordinate) -> Circle {
    let mut endpoints = (p, q);
    for pair in [(p, r), (q, r)] {
        if l2_sq_dist(pair.0, pair.1) > l2_sq_dist(endpoints.0, endpoints.1) {
            endpoints = pair;
        }
    }
    diameter_circle(endpoints.0, endpoints.1)
}

/// Smallest pair-diameter circle containing the remaining point, for triples whose circumcircle
/// determinant vanished. A collinear triple's farthest pair always qualifies, so the Welzl
/// invariant (all three boundary points enclosed) is preserved.
fn collinear_fallback(p: Coordinate, q: Coordinate, r: Coordinate, tolerance_sq: f64) -> Circle {
    let mut best: Option<Circle> = None;
    for (a, b, c) in [(p, q, r), (p, r, q), (q, r, p)] {
        let candidate = diameter_circle(a, b);
        if in_circle(c, &candidate, tolerance_sq) {
            best = Some(match best {
                Some(current) if current.radius <= candidate.radius => current,
                _ => candidate,
            });
        }
    }
    best.unwrap_or_else(|| largest_diameter_circle(p, q, r))
}

#[inline]
fn determinant(p: Coordinate, q: Coordinate, r: Coordinate) -> f64 {
    p.x * (q.y - r.y) + q.x * (r.y - p.y) + r.x * (p.y - q.y)
}

fn circumcircle(p: Coordinate, q: Coordinate, r: Coordinate, collinear_det: f64) -> Option<Circle> {
    let det = determinant(p, q, r);
    if det.abs() <= collinear_det {
        return None;
    }

    let d = 2.0 * det;
    let ux = ((p.x.powi(2) + p.y.powi(2)) * (q.y - r.y)
        + (q.x.powi(2) + q.y.powi(2)) * (r.y - p.y)
        + (r.x.powi(2) + r.y.powi(2)) * (p.y - q.y))
        / d;
    let uy = ((p.x.powi(2) + p.y.powi(2)) * (r.x - q.x)
        + (q.x.powi(2) + q.y.powi(2)) * (p.x - r.x)
        + (r.x.powi(2) + r.y.powi(2)) * (q.x - p.x))
        / d;
    let center = Coordinate { x: ux, y: uy };
    let radius = l2_sq_dist(center, p).sqrt();
    Some(Circle { center, radius })
}

fn in_circle(point: Coordinate, circle: &Circle, tolerance_sq: f64) -> bool {
    l2_sq_dist(point, circle.center) <= circle.radius.powi(2) + tolerance_sq
}

fn bounding_box_center(points: &[Coordinate]) -> Coordinate {
    let mut min = points[0];
    let mut max = points[0];
    for point in points {
        min.x = min.x.min(point.x);
        min.y = min.y.min(point.y);
        max.x = max.x.max(point.x);
        max.y = max.y.max(point.y);
    }
    Coordinate {
        x: (min.x + max.x) / 2.0,
        y: (min.y + max.y) / 2.0,
    }
}

fn minimum_enclosing_circle_area(
    points: &mut [Coordinate],
    rng: &mut fastrand::Rng,
) -> Option<f64> {
    minimum_enclosing_circle(points, rng).map(|circle| std::f64::consts::PI * circle.radius.powi(2))
}

fn minimum_enclosing_circle(points: &mut [Coordinate], rng: &mut fastrand::Rng) -> Option<Circle> {
    if points.len() < 2 {
        return None;
    }

    // Work in coordinates centered on the bounding box: at projected-CRS magnitudes (~1e6-1e7)
    // the raw predicates lose most of their mantissa to cancellation.
    let offset = bounding_box_center(points);
    for point in points.iter_mut() {
        point.x -= offset.x;
        point.y -= offset.y;
    }
    let extent_sq = points
        .iter()
        .map(|point| point.x * point.x + point.y * point.y)
        .fold(0.0, f64::max);
    let tolerance_sq = IN_CIRCLE_REL_EPS * extent_sq;
    let collinear_det = COLLINEAR_DET_REL_EPS * extent_sq;

    rng.shuffle(points);
    let mut circle = diameter_circle(points[0], points[1]);
    for i in 0..points.len() {
        if in_circle(points[i], &circle, tolerance_sq) {
            continue;
        }

        circle = Circle {
            center: points[i],
            radius: 0.0,
        };
        for j in 0..i {
            if in_circle(points[j], &circle, tolerance_sq) {
                continue;
            }

            circle = diameter_circle(points[i], points[j]);
            for k in 0..j {
                if in_circle(points[k], &circle, tolerance_sq) {
                    continue;
                }
                circle = circumcircle(points[i], points[j], points[k], collinear_det)
                    .unwrap_or_else(|| {
                        collinear_fallback(points[i], points[j], points[k], tolerance_sq)
                    });
            }
        }
    }

    circle.center.x += offset.x;
    circle.center.y += offset.y;
    Some(circle)
}

fn reock_score(
    points: &mut [Coordinate],
    district_area: f64,
    district: u16,
    rng: &mut fastrand::Rng,
) -> Result<f64> {
    let circle_area = minimum_enclosing_circle_area(points, rng)
        .ok_or(Error::CannotComputeReockCircle { district })?;
    if !district_area.is_finite() || district_area <= 0.0 {
        return Err(Error::InvalidDistrictArea {
            district,
            area: district_area,
        });
    }
    if !circle_area.is_finite() || circle_area <= 0.0 {
        return Err(Error::InvalidEnclosureArea {
            metric: "minimum enclosing circle",
            district,
            area: circle_area,
        });
    }

    let score = district_area / circle_area;
    if score > 1.0 + RATIO_SCORE_EPS {
        return Err(Error::ImpossibleScore {
            metric: "Reock score",
            district,
            score,
        });
    }
    Ok(score.min(1.0))
}

#[cfg(test)]
#[path = "../tests/reock.rs"]
mod tests;
