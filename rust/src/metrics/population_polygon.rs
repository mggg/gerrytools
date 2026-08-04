use super::hull_metric::{district_hull, district_score_table};
use crate::geometry::decode_polygon;
use crate::geometry::precise_difference_area;
use crate::scoring::delta::{expanded_assignment, validate_changes, DeltaChange};
use crate::scoring::district::{district_ids, observed_districts, DistrictOccupancy};
use crate::{DistrictTable, Error, PreparedGeometry, PreparedUnitHulls, Result};
use geo::{Area, BoundingRect, Coord, Covers, Intersects, MultiPolygon};
use rayon::prelude::*;
use rstar::{RTree, RTreeObject, AABB};
use std::sync::Arc;

const POPULATION_COVERAGE_REL_EPS: f64 = 1e-12;

#[derive(Clone, Copy, Debug)]
struct PolygonData {
    order: usize,
    owner: usize,
    weight: f64,
}

#[derive(Clone, Copy, Debug)]
struct IndexedPolygon {
    envelope: AABB<[f64; 2]>,
    data: PolygonData,
}

impl RTreeObject for IndexedPolygon {
    type Envelope = AABB<[f64; 2]>;

    fn envelope(&self) -> Self::Envelope {
        self.envelope
    }
}

struct Observation {
    input_index: usize,
    owner: usize,
    geometry: MultiPolygon<f64>,
    /// Encoded-byte ordering makes non-associative owner totals independent of input order.
    wkb: Vec<u8>,
    weight: f64,
}

#[derive(Debug)]
struct PopulationSurface {
    owner_totals: Vec<f64>,
    geometries: Arc<[MultiPolygon<f64>]>,
    polygons: RTree<IndexedPolygon>,
}

#[derive(Debug)]
/// Prepared unit hulls and population polygons for population-polygon compactness.
pub struct PreparedPopulationPolygon {
    unit_hulls: Arc<PreparedUnitHulls>,
    surface: PopulationSurface,
}

impl PreparedPopulationPolygon {
    /// Test-only convenience: decode `rows` into unit hulls, then align one weight per node.
    #[cfg(test)]
    pub(crate) fn from_aligned_wkb<W: AsRef<[u8]>>(rows: &[W], weights: Vec<f64>) -> Result<Self> {
        Self::from_aligned_unit_hulls_and_wkb(
            Arc::new(PreparedUnitHulls::from_wkb(rows)?),
            rows,
            weights,
        )
    }

    /// Test-only convenience: decode `rows` into unit hulls, then attach the population surface.
    #[cfg(test)]
    pub(crate) fn from_wkb<W: AsRef<[u8]>, P: AsRef<[u8]>>(
        rows: &[W],
        population_rows: &[P],
        weights: Vec<f64>,
        owners: Vec<usize>,
    ) -> Result<Self> {
        Self::from_unit_hulls_and_wkb(
            Arc::new(PreparedUnitHulls::from_wkb(rows)?),
            rows,
            population_rows,
            weights,
            owners,
        )
    }

    /// Prepare an aligned surface (one population polygon and weight per graph node) against
    /// already-decoded unit hulls.
    pub fn from_aligned_unit_hulls_and_wkb<W: AsRef<[u8]>>(
        unit_hulls: Arc<PreparedUnitHulls>,
        rows: &[W],
        weights: Vec<f64>,
    ) -> Result<Self> {
        if rows.len() != unit_hulls.node_count() {
            return Err(Error::MetricNodeCount {
                metric: "population_polygon".into(),
                actual: rows.len(),
                expected: unit_hulls.node_count(),
            });
        }
        let owners = (0..rows.len()).collect();
        let observations = validate_observations(rows.len(), rows, weights, owners)?;
        Self::from_parts(unit_hulls, observations)
    }

    /// Validate a population surface (with explicit owners) against already-decoded unit hulls.
    pub fn from_unit_hulls_and_wkb<W: AsRef<[u8]>, P: AsRef<[u8]>>(
        unit_hulls: Arc<PreparedUnitHulls>,
        rows: &[W],
        population_rows: &[P],
        weights: Vec<f64>,
        owners: Vec<usize>,
    ) -> Result<Self> {
        if rows.len() != unit_hulls.node_count() {
            return Err(Error::MetricNodeCount {
                metric: "population_polygon".into(),
                actual: rows.len(),
                expected: unit_hulls.node_count(),
            });
        }
        let observations = validate_observations(rows.len(), population_rows, weights, owners)?;
        decode_and_validate_owners(rows, &observations)?;
        Self::from_parts(unit_hulls, observations)
    }

    /// Prepare an aligned surface from evaluator-lifetime decoded geometry.
    pub(crate) fn from_aligned_prepared_geometry(
        geometry: Arc<PreparedGeometry>,
        weights: Vec<f64>,
    ) -> Result<Self> {
        let node_count = geometry.node_count();
        validate_aligned_weights(node_count, &weights)?;
        let unit_hulls = geometry.unit_hulls()?;
        let geometries = geometry.geometries();
        let mut owner_totals = vec![0.0; node_count];
        let mut polygons = Vec::with_capacity(node_count);
        for (node, (&weight, polygon)) in weights.iter().zip(geometries.iter()).enumerate() {
            owner_totals[node] = weight;
            polygons.push(IndexedPolygon {
                envelope: polygon_envelope(polygon),
                data: PolygonData {
                    order: node,
                    owner: node,
                    weight,
                },
            });
        }
        Ok(Self {
            unit_hulls,
            surface: PopulationSurface {
                owner_totals,
                geometries,
                polygons: RTree::bulk_load(polygons),
            },
        })
    }

    /// Prepare an alternative population surface and infer its unique graph-unit owners.
    pub(crate) fn from_prepared_geometry<P>(
        geometry: Arc<PreparedGeometry>,
        population_rows: &[P],
        weights: Vec<f64>,
    ) -> Result<Self>
    where
        P: AsRef<[u8]> + Sync,
    {
        if population_rows.len() != weights.len() {
            return Err(Error::PopulationObservationLength {
                geometries: population_rows.len(),
                weights: weights.len(),
                owners: population_rows.len(),
            });
        }
        if population_rows.is_empty() {
            return Err(Error::EmptyPopulationSurface);
        }

        let unit_geometries = geometry.geometries();
        let mut decoded = population_rows
            .par_iter()
            .zip(&weights)
            .enumerate()
            .map(|(input_index, (bytes, &weight))| {
                if !weight.is_finite() || weight < 0.0 {
                    return Err(Error::InvalidPopulationWeight {
                        observation: input_index,
                        weight,
                    });
                }
                let bytes = bytes.as_ref();
                let polygon = decode_population_polygon(input_index, bytes)?;
                let area = polygon.unsigned_area();
                if !area.is_finite() || area <= 0.0 {
                    return Err(Error::InvalidPopulationGeometryArea {
                        observation: input_index,
                        area,
                    });
                }
                let envelope = polygon_envelope(&polygon);
                let tolerance = POPULATION_COVERAGE_REL_EPS * area.max(1.0);
                let mut covering = Vec::new();
                for (owner, owner_envelope) in geometry.unit_candidates(&envelope) {
                    if area - envelope_intersection_area(&envelope, &owner_envelope) > tolerance {
                        continue;
                    }
                    let owner_geometry = &unit_geometries[owner];
                    if owner_geometry == &polygon
                        || owner_geometry.covers(&polygon)
                        || precise_difference_area(std::slice::from_ref(&polygon), owner_geometry)?
                            <= tolerance
                    {
                        covering.push(owner);
                    }
                }
                if covering.len() != 1 {
                    return Err(Error::PopulationGeometryOwnerCount {
                        observation: input_index,
                        count: covering.len(),
                    });
                }
                Ok(Observation {
                    input_index,
                    owner: covering[0],
                    geometry: polygon,
                    wkb: bytes.to_vec(),
                    weight,
                })
            })
            .collect::<Vec<Result<_>>>()
            .into_iter()
            .collect::<Result<Vec<_>>>()?;
        if !decoded.iter().any(|observation| observation.weight > 0.0) {
            return Err(Error::NoPositivePopulation);
        }
        sort_observations(&mut decoded);
        Self::from_parts(geometry.unit_hulls()?, decoded)
    }

    /// Return a shared handle to the prepared unit hulls.
    pub fn unit_hulls(&self) -> Arc<PreparedUnitHulls> {
        Arc::clone(&self.unit_hulls)
    }

    /// Return the required assignment length.
    pub fn node_count(&self) -> usize {
        self.unit_hulls.node_count()
    }

    /// Score every observed district in an assignment.
    pub fn score(&self, assignment: &[u16]) -> Result<DistrictTable> {
        if assignment.len() != self.node_count() {
            return Err(Error::AssignmentLength {
                actual: assignment.len(),
                expected: self.node_count(),
            });
        }
        let (district_slots, observed) = observed_districts(assignment);
        let districts = district_ids(&observed);
        let mut nodes_by_district = vec![Vec::new(); district_slots];
        for (node, &district) in assignment.iter().enumerate() {
            nodes_by_district[district as usize].push(node);
        }

        let mut hull_points = Vec::new();
        let mut candidates = Vec::new();
        let mut scores = Vec::with_capacity(districts.len());
        for &district in &districts {
            scores.push(district_score(
                self,
                assignment,
                &nodes_by_district[district as usize],
                district,
                &mut hull_points,
                &mut candidates,
            )?);
        }
        Ok(DistrictTable::new(districts, scores, 1))
    }

    /// Create incremental population-polygon state for an initial assignment.
    pub fn incremental(&self, assignment: &[u16]) -> Result<IncrementalPopulationPolygon<'_>> {
        IncrementalPopulationPolygon::new(self, assignment)
    }

    fn from_parts(
        unit_hulls: Arc<PreparedUnitHulls>,
        observations: Vec<Observation>,
    ) -> Result<Self> {
        let mut owner_totals = vec![0.0; unit_hulls.node_count()];
        let mut polygons = Vec::with_capacity(observations.len());
        let mut geometries = Vec::with_capacity(observations.len());
        for (order, observation) in observations.into_iter().enumerate() {
            owner_totals[observation.owner] += observation.weight;
            if !owner_totals[observation.owner].is_finite() {
                return Err(Error::NonFinitePopulationOwnerTotal {
                    owner: observation.owner,
                });
            }
            let envelope = polygon_envelope(&observation.geometry);
            geometries.push(observation.geometry);
            polygons.push(IndexedPolygon {
                envelope,
                data: PolygonData {
                    order,
                    owner: observation.owner,
                    weight: observation.weight,
                },
            });
        }
        if !owner_totals.iter().sum::<f64>().is_finite() {
            return Err(Error::NonFinitePopulationTotal);
        }
        Ok(Self {
            unit_hulls,
            surface: PopulationSurface {
                owner_totals,
                geometries: geometries.into(),
                polygons: RTree::bulk_load(polygons),
            },
        })
    }
}

/// District population-polygon scores maintained across assignment changes.
pub struct IncrementalPopulationPolygon<'a> {
    metric: &'a PreparedPopulationPolygon,
    assignment: Vec<u16>,
    scores: Vec<f64>,
    occupancy: DistrictOccupancy,
    nodes_by_district: Vec<Vec<usize>>,
    positive_nodes_by_district: Vec<usize>,
    hull_points: Vec<Coord<f64>>,
    hull_points_b: Vec<Coord<f64>>,
    candidates: Vec<&'a IndexedPolygon>,
    candidates_b: Vec<&'a IndexedPolygon>,
}

impl<'a> IncrementalPopulationPolygon<'a> {
    fn new(metric: &'a PreparedPopulationPolygon, assignment: &[u16]) -> Result<Self> {
        let mut state = Self {
            metric,
            assignment: vec![0; metric.node_count()],
            scores: Vec::new(),
            occupancy: DistrictOccupancy::new(),
            nodes_by_district: Vec::new(),
            positive_nodes_by_district: Vec::new(),
            hull_points: Vec::new(),
            hull_points_b: Vec::new(),
            candidates: Vec::new(),
            candidates_b: Vec::new(),
        };
        state.reset(assignment)?;
        Ok(state)
    }

    /// Replace the assignment and recompute every score from scratch.
    pub fn reset(&mut self, assignment: &[u16]) -> Result<()> {
        if assignment.len() != self.metric.node_count() {
            return Err(Error::AssignmentLength {
                actual: assignment.len(),
                expected: self.metric.node_count(),
            });
        }
        let (district_slots, observed) = observed_districts(assignment);
        let mut scores = vec![0.0; district_slots];
        let mut nodes_by_district = vec![Vec::new(); district_slots];
        let mut positive_nodes_by_district = vec![0; district_slots];
        for (node, &district) in assignment.iter().enumerate() {
            nodes_by_district[district as usize].push(node);
            if self.metric.surface.owner_totals[node] > 0.0 {
                positive_nodes_by_district[district as usize] += 1;
            }
        }

        let mut hull_points = Vec::new();
        let mut candidates = Vec::new();
        for district in district_ids(&observed) {
            scores[district as usize] = district_score(
                self.metric,
                assignment,
                &nodes_by_district[district as usize],
                district,
                &mut hull_points,
                &mut candidates,
            )?;
        }
        self.assignment.copy_from_slice(assignment);
        self.scores = scores;
        self.occupancy.reset(assignment);
        self.nodes_by_district = nodes_by_district;
        self.positive_nodes_by_district = positive_nodes_by_district;
        Ok(())
    }

    /// Apply a delta whose `old` labels are validated against this state's current assignment.
    pub fn update(&mut self, changes: &[DeltaChange]) -> Result<()> {
        validate_changes(&self.assignment, changes)?;
        self.update_trusted(None, changes)
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
            touched.push(change.old);
            touched.push(change.new);
        }
        touched.sort_unstable();
        touched.dedup();

        let mut node_deltas = vec![0_isize; self.scores.len()];
        let mut positive_deltas = vec![0_isize; self.scores.len()];
        for change in changes {
            if change.old == change.new {
                continue;
            }
            node_deltas[change.old as usize] -= 1;
            node_deltas[change.new as usize] += 1;
            if self.metric.surface.owner_totals[change.node] > 0.0 {
                positive_deltas[change.old as usize] -= 1;
                positive_deltas[change.new as usize] += 1;
            }
        }
        for &district in &touched {
            let index = district as usize;
            let node_count = self.nodes_by_district[index].len() as isize + node_deltas[index];
            let positive_nodes =
                self.positive_nodes_by_district[index] as isize + positive_deltas[index];
            if node_count > 0 && positive_nodes == 0 {
                return Err(Error::InvalidDistrictPopulation {
                    kind: "owned",
                    district,
                    population: 0.0,
                });
            }
        }

        for change in changes {
            if change.old == change.new {
                continue;
            }
            self.remove_node(change.node, change.old)?;
            self.add_node(change.node, change.new);
            self.assignment[change.node] = change.new;
            if self.metric.surface.owner_totals[change.node] > 0.0 {
                self.positive_nodes_by_district[change.old as usize] -= 1;
                self.positive_nodes_by_district[change.new as usize] += 1;
            }
            self.occupancy.apply(change.old, change.new);
        }

        let mut recompute = Vec::with_capacity(touched.len());
        for district in touched {
            if self.occupancy.is_empty(district) {
                self.scores[district as usize] = 0.0;
            } else {
                recompute.push(district);
            }
        }
        self.recompute_scores(&recompute)
    }

    /// Return the current score for every observed district.
    pub fn result(&self) -> DistrictTable {
        district_score_table(self.occupancy.observed(), &self.scores)
    }

    fn remove_node(&mut self, node: usize, district: u16) -> Result<()> {
        let nodes = &mut self.nodes_by_district[district as usize];
        let position = nodes
            .binary_search(&node)
            .map_err(|_| Error::IncrementalNodeMembership { node, district })?;
        nodes.remove(position);
        Ok(())
    }

    fn add_node(&mut self, node: usize, district: u16) {
        let nodes = &mut self.nodes_by_district[district as usize];
        let position = nodes.binary_search(&node).unwrap_err();
        nodes.insert(position, node);
    }

    fn recompute_scores(&mut self, districts: &[u16]) -> Result<()> {
        if let [left, right] = districts {
            let metric = self.metric;
            let assignment = &self.assignment;
            let nodes = &self.nodes_by_district;
            let (left_score, right_score) = rayon::join(
                || {
                    district_score(
                        metric,
                        assignment,
                        &nodes[*left as usize],
                        *left,
                        &mut self.hull_points,
                        &mut self.candidates,
                    )
                },
                || {
                    district_score(
                        metric,
                        assignment,
                        &nodes[*right as usize],
                        *right,
                        &mut self.hull_points_b,
                        &mut self.candidates_b,
                    )
                },
            );
            self.scores[*left as usize] = left_score?;
            self.scores[*right as usize] = right_score?;
            return Ok(());
        }

        for &district in districts {
            self.scores[district as usize] = district_score(
                self.metric,
                &self.assignment,
                &self.nodes_by_district[district as usize],
                district,
                &mut self.hull_points,
                &mut self.candidates,
            )?;
        }
        Ok(())
    }
}

fn polygon_envelope(polygon: &MultiPolygon<f64>) -> AABB<[f64; 2]> {
    let bounds = polygon
        .bounding_rect()
        .expect("validated positive-area population geometry has bounds");
    AABB::from_corners(
        [bounds.min().x, bounds.min().y],
        [bounds.max().x, bounds.max().y],
    )
}

fn envelope_intersection_area(left: &AABB<[f64; 2]>, right: &AABB<[f64; 2]>) -> f64 {
    let lower = [
        left.lower()[0].max(right.lower()[0]),
        left.lower()[1].max(right.lower()[1]),
    ];
    let upper = [
        left.upper()[0].min(right.upper()[0]),
        left.upper()[1].min(right.upper()[1]),
    ];
    (upper[0] - lower[0]).max(0.0) * (upper[1] - lower[1]).max(0.0)
}

fn validate_aligned_weights(node_count: usize, weights: &[f64]) -> Result<()> {
    if weights.len() != node_count {
        return Err(Error::PopulationObservationLength {
            geometries: node_count,
            weights: weights.len(),
            owners: node_count,
        });
    }
    if weights.is_empty() {
        return Err(Error::EmptyPopulationSurface);
    }
    for (observation, &weight) in weights.iter().enumerate() {
        if !weight.is_finite() || weight < 0.0 {
            return Err(Error::InvalidPopulationWeight {
                observation,
                weight,
            });
        }
    }
    if !weights.iter().any(|&weight| weight > 0.0) {
        return Err(Error::NoPositivePopulation);
    }
    if !weights.iter().sum::<f64>().is_finite() {
        return Err(Error::NonFinitePopulationTotal);
    }
    Ok(())
}

fn decode_population_polygon(input_index: usize, bytes: &[u8]) -> Result<MultiPolygon<f64>> {
    decode_polygon(input_index, bytes).map_err(|error| {
        Error::Geometry(format!(
            "invalid population geometry at observation {input_index}: {error}"
        ))
    })
}

fn validate_observations<P: AsRef<[u8]>>(
    node_count: usize,
    population_rows: &[P],
    weights: Vec<f64>,
    owners: Vec<usize>,
) -> Result<Vec<Observation>> {
    if population_rows.len() != weights.len() || population_rows.len() != owners.len() {
        return Err(Error::PopulationObservationLength {
            geometries: population_rows.len(),
            weights: weights.len(),
            owners: owners.len(),
        });
    }
    if population_rows.is_empty() {
        return Err(Error::EmptyPopulationSurface);
    }

    let mut observations = Vec::with_capacity(population_rows.len());
    for (input_index, ((bytes, weight), owner)) in
        population_rows.iter().zip(weights).zip(owners).enumerate()
    {
        if !weight.is_finite() || weight < 0.0 {
            return Err(Error::InvalidPopulationWeight {
                observation: input_index,
                weight,
            });
        }
        if owner >= node_count {
            return Err(Error::PopulationOwnerOutOfRange {
                observation: input_index,
                owner,
                node_count,
            });
        }
        let bytes = bytes.as_ref();
        let geometry = decode_population_polygon(input_index, bytes)?;
        let area = geometry.unsigned_area();
        if !area.is_finite() || area <= 0.0 {
            return Err(Error::InvalidPopulationGeometryArea {
                observation: input_index,
                area,
            });
        }
        observations.push(Observation {
            input_index,
            owner,
            geometry,
            wkb: bytes.to_vec(),
            weight,
        });
    }
    if !observations
        .iter()
        .any(|observation| observation.weight > 0.0)
    {
        return Err(Error::NoPositivePopulation);
    }
    sort_observations(&mut observations);
    Ok(observations)
}

fn sort_observations(observations: &mut [Observation]) {
    observations.sort_by(|left, right| {
        left.owner
            .cmp(&right.owner)
            .then_with(|| left.wkb.cmp(&right.wkb))
            .then_with(|| left.weight.total_cmp(&right.weight))
    });
}

fn decode_and_validate_owners<W>(rows: &[W], observations: &[Observation]) -> Result<()>
where
    W: AsRef<[u8]>,
{
    let mut observation = 0;
    for (owner, bytes) in rows.iter().enumerate() {
        let geometry = decode_polygon(owner, bytes.as_ref())?;
        while observation < observations.len() && observations[observation].owner == owner {
            let item = &observations[observation];
            let area = item.geometry.unsigned_area();
            let uncovered_area =
                precise_difference_area(std::slice::from_ref(&item.geometry), &geometry)?;
            let tolerance = POPULATION_COVERAGE_REL_EPS * area.max(1.0);
            if uncovered_area > tolerance {
                return Err(Error::PopulationGeometryOutsideOwner {
                    observation: item.input_index,
                    owner,
                    uncovered_area,
                    tolerance,
                });
            }
            observation += 1;
        }
    }
    debug_assert_eq!(observation, observations.len());
    Ok(())
}

fn district_score<'a>(
    metric: &'a PreparedPopulationPolygon,
    assignment: &[u16],
    nodes: &[usize],
    district: u16,
    hull_points: &mut Vec<Coord<f64>>,
    candidates: &mut Vec<&'a IndexedPolygon>,
) -> Result<f64> {
    hull_points.clear();
    let mut numerator = 0.0;
    for &node in nodes {
        numerator += metric.surface.owner_totals[node];
        hull_points.extend(
            metric
                .unit_hulls
                .unit_hull_points(node)
                .iter()
                .map(|point| Coord {
                    x: point.x,
                    y: point.y,
                }),
        );
    }
    if !numerator.is_finite() || numerator <= 0.0 {
        return Err(Error::InvalidDistrictPopulation {
            kind: "owned",
            district,
            population: numerator,
        });
    }

    let hull = district_hull(hull_points);
    let bounds = hull
        .bounding_rect()
        .expect("a district contains at least one positive-area unit");
    candidates.clear();
    candidates.extend(metric.surface.polygons.locate_in_envelope_intersecting(
        &AABB::from_corners(
            [bounds.min().x, bounds.min().y],
            [bounds.max().x, bounds.max().y],
        ),
    ));
    candidates.sort_unstable_by_key(|polygon| polygon.data.order);

    // Owner coverage proves that every owned polygon intersects this hull.
    let mut denominator = numerator;
    for polygon in candidates.iter() {
        if assignment[polygon.data.owner] == district {
            continue;
        }
        if hull.intersects(&metric.surface.geometries[polygon.data.order]) {
            denominator += polygon.data.weight;
        }
    }
    if !denominator.is_finite() || denominator <= 0.0 {
        return Err(Error::InvalidDistrictPopulation {
            kind: "hull",
            district,
            population: denominator,
        });
    }
    let score = numerator / denominator;
    if !score.is_finite() || score <= 0.0 || score > 1.0 {
        return Err(Error::ImpossibleScore {
            metric: "population-polygon score",
            district,
            score,
        });
    }
    Ok(score)
}

#[cfg(test)]
#[path = "../tests/population_polygon.rs"]
mod tests;
