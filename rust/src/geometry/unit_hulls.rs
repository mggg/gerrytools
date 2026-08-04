use crate::{Error, Result};
use std::collections::hash_map::DefaultHasher;
use std::hash::Hasher;
use std::sync::Arc;

#[derive(Clone, Copy, Debug, PartialEq)]
/// One point in a projected coordinate system.
pub struct Coordinate {
    /// Horizontal coordinate.
    pub x: f64,
    /// Vertical coordinate.
    pub y: f64,
}

#[derive(Debug)]
/// The area and convex-hull vertices of one graph unit, validated when prepared.
pub struct UnitHull {
    area: f64,
    points: Vec<Coordinate>,
}

impl UnitHull {
    /// Create an unvalidated unit hull.
    ///
    /// [`PreparedUnitHulls::new`] validates the area and points when the hull is prepared.
    pub fn new(area: f64, points: Vec<Coordinate>) -> Self {
        Self { area, points }
    }
}

#[derive(Debug)]
/// Validated unit areas and convex hulls packed in graph-node order.
pub struct PreparedUnitHulls {
    areas: Vec<f64>,
    offsets: Vec<usize>,
    points: Vec<Coordinate>,
}

impl PreparedUnitHulls {
    /// Decode ordered, projected Polygon or MultiPolygon WKB rows into packed unit hulls.
    pub fn from_wkb<W: AsRef<[u8]>>(rows: &[W]) -> Result<Self> {
        let units = rows
            .iter()
            .enumerate()
            .map(|(row, bytes)| super::unit_hull(row, bytes.as_ref()))
            .collect::<Result<_>>()?;
        Self::new(units)
    }

    /// Validate and pack unit hulls in graph-node order.
    pub fn new(units: Vec<UnitHull>) -> Result<Self> {
        let mut areas = Vec::with_capacity(units.len());
        let mut offsets = Vec::with_capacity(units.len() + 1);
        let mut points = Vec::with_capacity(units.iter().map(|unit| unit.points.len()).sum());
        offsets.push(0);

        for (unit, hull) in units.into_iter().enumerate() {
            if !hull.area.is_finite() || hull.area <= 0.0 {
                return Err(Error::InvalidGeometryArea {
                    unit,
                    area: hull.area,
                });
            }
            if hull.points.len() < 3 {
                return Err(Error::InvalidGeometryPointCount {
                    unit,
                    actual: hull.points.len(),
                });
            }
            if let Some(point) = hull
                .points
                .iter()
                .position(|point| !point.x.is_finite() || !point.y.is_finite())
            {
                return Err(Error::NonFiniteGeometryPoint { unit, point });
            }
            areas.push(hull.area);
            points.extend(hull.points);
            offsets.push(points.len());
        }

        Ok(Self {
            areas,
            offsets,
            points,
        })
    }

    /// Return the number of graph units.
    pub fn node_count(&self) -> usize {
        self.areas.len()
    }

    #[cfg(test)]
    fn areas(&self) -> &[f64] {
        &self.areas
    }

    #[cfg(test)]
    fn hull_offsets(&self) -> &[usize] {
        &self.offsets
    }

    #[cfg(test)]
    fn hull_points(&self) -> &[Coordinate] {
        &self.points
    }

    /// Return a unit's positive area.
    ///
    /// # Panics
    ///
    /// Panics when `node` is outside [`Self::node_count`].
    pub fn unit_area(&self, node: usize) -> f64 {
        self.areas[node]
    }

    /// Return a unit's convex-hull vertices without a repeated closing vertex.
    ///
    /// # Panics
    ///
    /// Panics when `node` is outside [`Self::node_count`].
    pub fn unit_hull_points(&self, node: usize) -> &[Coordinate] {
        &self.points[self.offsets[node]..self.offsets[node + 1]]
    }
}

/// Caches one decoded unit-hull set, fingerprinting the encoded rows so later registrations
/// cannot silently reuse hulls decoded from different geometry.
#[derive(Debug, Default)]
pub struct UnitHullCache {
    entry: Option<CachedUnitHulls>,
}

#[derive(Debug)]
struct CachedUnitHulls {
    hulls: Arc<PreparedUnitHulls>,
    row_count: usize,
    fingerprint: u64,
}

impl UnitHullCache {
    /// Create an empty cache.
    pub fn new() -> Self {
        Self::default()
    }

    /// Decode `rows` on first use; later calls verify the rows match before reusing the hulls.
    pub fn get_or_decode<W: AsRef<[u8]>>(&mut self, rows: &[W]) -> Result<Arc<PreparedUnitHulls>> {
        let fingerprint = rows_fingerprint(rows);
        if let Some(cached) = &self.entry {
            if cached.row_count != rows.len() || cached.fingerprint != fingerprint {
                return Err(Error::InvalidInput(
                    "geometry rows differ from the previously registered geometry".into(),
                ));
            }
            return Ok(Arc::clone(&cached.hulls));
        }
        let hulls = Arc::new(PreparedUnitHulls::from_wkb(rows)?);
        self.entry = Some(CachedUnitHulls {
            hulls: Arc::clone(&hulls),
            row_count: rows.len(),
            fingerprint,
        });
        Ok(hulls)
    }
}

fn rows_fingerprint<W: AsRef<[u8]>>(rows: &[W]) -> u64 {
    let mut hasher = DefaultHasher::new();
    for row in rows {
        let row = row.as_ref();
        hasher.write_usize(row.len());
        hasher.write(row);
    }
    hasher.finish()
}

#[cfg(test)]
#[path = "../tests/unit_hulls.rs"]
mod tests;
