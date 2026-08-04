use crate::{Error, Result};
use clipper2::{difference, intersect, FillRule, One, Path, Paths, Point};
use geo::{BoundingRect, Coord, LineString, MultiPolygon};

type IntegerPath = Path<One>;
type IntegerPaths = Paths<One>;

// Keep normalized coordinates well below Clipper2's i64 limit while retaining nearly all of an
// f64 mantissa. The validation tolerances are much wider than the remaining quantization step.
const NORMALIZED_HALF_SPAN: f64 = (1_u64 << 50) as f64;
const MAX_NORMALIZED_COORDINATE: f64 = (1_u64 << 52) as f64;

#[derive(Clone, Copy)]
enum Operation {
    Difference,
    Intersection,
}

pub(crate) fn precise_difference_area(
    subjects: &[MultiPolygon<f64>],
    clip: &MultiPolygon<f64>,
) -> Result<f64> {
    precise_area(subjects, std::slice::from_ref(clip), Operation::Difference)
}

pub(crate) fn precise_intersection_area(
    left: &MultiPolygon<f64>,
    right: &MultiPolygon<f64>,
) -> Result<f64> {
    precise_area(
        std::slice::from_ref(left),
        std::slice::from_ref(right),
        Operation::Intersection,
    )
}

fn precise_area(
    subjects: &[MultiPolygon<f64>],
    clips: &[MultiPolygon<f64>],
    operation: Operation,
) -> Result<f64> {
    let mut bounds: Option<geo::Rect<f64>> = None;
    for geometry in subjects.iter().chain(clips) {
        let rectangle = geometry
            .bounding_rect()
            .ok_or_else(|| geometry_error("geometry has no bounding rectangle"))?;
        bounds = Some(match bounds {
            None => rectangle,
            Some(current) => geo::Rect::new(
                Coord {
                    x: current.min().x.min(rectangle.min().x),
                    y: current.min().y.min(rectangle.min().y),
                },
                Coord {
                    x: current.max().x.max(rectangle.max().x),
                    y: current.max().y.max(rectangle.max().y),
                },
            ),
        });
    }
    let bounds = bounds.ok_or_else(|| geometry_error("precise overlay requires geometry"))?;
    let origin = Coord {
        x: (bounds.min().x + bounds.max().x) / 2.0,
        y: (bounds.min().y + bounds.max().y) / 2.0,
    };
    let half_span =
        ((bounds.max().x - bounds.min().x) / 2.0).max((bounds.max().y - bounds.min().y) / 2.0);
    if !half_span.is_finite() || half_span <= 0.0 {
        return Err(geometry_error("precise overlay has invalid bounds"));
    }
    let scale = NORMALIZED_HALF_SPAN / half_span;
    let subject_paths = geometry_paths(subjects, origin, scale)?;
    let clip_paths = geometry_paths(clips, origin, scale)?;
    let output = match operation {
        Operation::Difference => difference(subject_paths, clip_paths, FillRule::NonZero),
        Operation::Intersection => intersect(subject_paths, clip_paths, FillRule::NonZero),
    }
    .map_err(|error| geometry_error(format!("precise overlay failed: {error}")))?;
    let area = output.signed_area().abs() / scale.powi(2);
    if !area.is_finite() || area < 0.0 {
        return Err(geometry_error(format!(
            "precise overlay produced invalid area {area}"
        )));
    }
    Ok(area)
}

fn geometry_paths(
    geometries: &[MultiPolygon<f64>],
    origin: Coord<f64>,
    scale: f64,
) -> Result<IntegerPaths> {
    let mut paths = Vec::new();
    for geometry in geometries {
        for polygon in &geometry.0 {
            paths.push(ring_path(polygon.exterior(), true, origin, scale)?);
            for ring in polygon.interiors() {
                paths.push(ring_path(ring, false, origin, scale)?);
            }
        }
    }
    Ok(Paths::new(paths))
}

fn ring_path(
    ring: &LineString<f64>,
    positive: bool,
    origin: Coord<f64>,
    scale: f64,
) -> Result<IntegerPath> {
    let end = ring.0.len().saturating_sub(usize::from(ring.is_closed()));
    let mut path = ring.0[..end]
        .iter()
        .map(|coordinate| point64(*coordinate, origin, scale))
        .collect::<Result<Vec<_>>>()?;
    if path.len() < 3 {
        return Err(geometry_error(
            "precise overlay ring has fewer than three points",
        ));
    }
    let signed_area = ring_signed_area(&path);
    if (signed_area > 0.0) != positive {
        path.reverse();
    }
    Ok(Path::new(path))
}

fn point64(coordinate: Coord<f64>, origin: Coord<f64>, scale: f64) -> Result<Point<One>> {
    let x = ((coordinate.x - origin.x) * scale).round();
    let y = ((coordinate.y - origin.y) * scale).round();
    if !x.is_finite()
        || !y.is_finite()
        || x.abs() > MAX_NORMALIZED_COORDINATE
        || y.abs() > MAX_NORMALIZED_COORDINATE
    {
        return Err(geometry_error("precise overlay coordinate is out of range"));
    }
    Ok(Point::from_scaled(x as i64, y as i64))
}

fn ring_signed_area(path: &[Point<One>]) -> f64 {
    path.iter()
        .zip(path.iter().cycle().skip(1))
        .take(path.len())
        .map(|(left, right)| {
            left.x_scaled() as f64 * right.y_scaled() as f64
                - right.x_scaled() as f64 * left.y_scaled() as f64
        })
        .sum::<f64>()
        / 2.0
}

fn geometry_error(message: impl Into<String>) -> Error {
    Error::Geometry(message.into())
}
