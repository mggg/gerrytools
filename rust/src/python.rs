use crate::{
    AssignmentSource, DistrictTable, Error, MetricRef, MetricScore, PlanScore, PlanTable,
    PreparedAreaPerimeterMetrics, PreparedConvexHullRatio, PreparedCutEdges,
    PreparedGeometry as RustPreparedGeometry, PreparedMetric, PreparedPolsbyPopper,
    PreparedPopulationPolygon, PreparedRegion, PreparedRegionTally, PreparedReock,
    PreparedSchwartzberg, PreparedStateClippedConvexHullRatio, PreparedTally, RunMetadata,
    RunWriter, Scorer, SharedTallyMetric, StreamOptions,
};
use pyo3::exceptions::{
    PyFileExistsError, PyFileNotFoundError, PyOSError, PyPermissionError, PyValueError,
};
use pyo3::prelude::*;
use std::io;
use std::path::PathBuf;
use std::sync::Arc;

type MetricRows = (Vec<u16>, Vec<Vec<Vec<f64>>>, Option<(u64, u64)>);
const PROGRESS_BATCH_SIZE: usize = 256;

#[derive(FromPyObject)]
#[pyo3(from_item_all)]
struct PythonStreamOptions {
    bendl_node_order_json: Option<String>,
    max_samples: Option<u64>,
    batch_size: usize,
    track_uniqueness: bool,
    progress: Option<Py<PyAny>>,
}

enum BackendMetric {
    Independent(PreparedMetric),
    SharedTally(SharedTallyMetric),
}

impl BackendMetric {
    fn kind(&self) -> &'static str {
        match self {
            Self::Independent(metric) => MetricRef::from(metric).kind(),
            Self::SharedTally(metric) => metric.kind(),
        }
    }

    fn column_count(&self) -> usize {
        match self {
            Self::Independent(metric) => MetricRef::from(metric).column_count(),
            Self::SharedTally(metric) => metric.column_count(),
        }
    }

    fn shape(&self) -> crate::TableShape {
        match self {
            Self::Independent(metric) => MetricRef::from(metric).shape(),
            Self::SharedTally(metric) => metric.shape(),
        }
    }

    fn register<'a>(&'a self, scorer: &mut Scorer<'a>, name: String) -> crate::Result<()> {
        match self {
            Self::Independent(metric) => scorer.add(name, metric),
            Self::SharedTally(metric) => scorer.add_shared(name, metric.clone()),
        }
    }

    fn accepts_kind(&self, kind: &str, columns: &[usize]) -> bool {
        match self {
            Self::Independent(PreparedMetric::AreaPerimeterMetrics(_)) => {
                (kind == "polsby_popper" && columns == [0])
                    || (kind == "schwartzberg" && columns == [1])
            }
            _ => kind == self.kind(),
        }
    }

    /// The kinds [`Self::accepts_kind`] accepts, for mismatch diagnostics. The combined
    /// area-perimeter metric is only ever projected as one of its two component kinds.
    fn expected_kinds(&self) -> String {
        match self {
            Self::Independent(PreparedMetric::AreaPerimeterMetrics(_)) => {
                "\"polsby_popper\" (column [0]) or \"schwartzberg\" (column [1])".into()
            }
            _ => format!("{:?}", self.kind()),
        }
    }
}

fn flatten_score(score: &MetricScore) -> Vec<f64> {
    match score {
        MetricScore::District(table) => (0..table.column_count())
            .flat_map(|column| {
                table
                    .column(column)
                    .expect("column index comes from the table's column count")
            })
            .copied()
            .collect(),
        MetricScore::Plan(table) => table.values().to_vec(),
    }
}

fn project_score(score: &MetricScore, columns: &[usize]) -> crate::Result<MetricScore> {
    match score {
        MetricScore::District(table) => {
            let mut values = Vec::with_capacity(columns.len() * table.district_ids().len());
            for &column in columns {
                values.extend_from_slice(table.column(column).ok_or_else(|| {
                    Error::InvalidInput(format!(
                        "district projection column {column} is out of range"
                    ))
                })?);
            }
            Ok(MetricScore::District(DistrictTable::new(
                table.district_ids().to_vec(),
                values,
                columns.len(),
            )))
        }
        MetricScore::Plan(table) => {
            let values = columns
                .iter()
                .map(|&column| {
                    table.values().get(column).copied().ok_or_else(|| {
                        Error::InvalidInput(format!(
                            "plan projection column {column} is out of range"
                        ))
                    })
                })
                .collect::<crate::Result<_>>()?;
            PlanTable::new(table.district_ids().to_vec(), values).map(MetricScore::Plan)
        }
    }
}

/// Validate run metadata against Python-side metrics and their column projections.
///
/// This is the projection-time counterpart of the engine `Scorer::validate_run_metadata`
/// (scoring/stream.rs): that validator checks registration identity (instance names, exact
/// subkeys); this one checks projected kinds and column ranges. Only the shape check is
/// genuinely shared, via [`crate::scoring::output::check_declared_shape`].
fn validate_projections(
    metrics: &[BackendMetric],
    metadata: &RunMetadata,
    projections: &[(usize, Vec<usize>)],
) -> crate::Result<()> {
    if metadata.metrics.len() != projections.len() {
        return Err(Error::InvalidInput(
            "run metadata and logical projections have different lengths".into(),
        ));
    }
    for (description, (source, columns)) in metadata.metrics.iter().zip(projections) {
        let metric = metrics.get(*source).ok_or_else(|| {
            Error::InvalidInput(format!("metric projection source {source} is out of range"))
        })?;
        if !metric.accepts_kind(&description.kind, columns) {
            return Err(Error::InvalidInput(format!(
                "run metric {:?} has kind {:?}; expected {}",
                description.instance,
                description.kind,
                metric.expected_kinds()
            )));
        }
        crate::scoring::output::check_declared_shape(description, metric.shape())?;
        if description.subkeys.len() != columns.len() {
            return Err(Error::InvalidInput(format!(
                "run metric {:?} declares {} subkeys; its projection has {} columns",
                description.instance,
                description.subkeys.len(),
                columns.len()
            )));
        }
        if columns
            .iter()
            .any(|&column| column >= metric.column_count())
        {
            return Err(Error::InvalidInput(format!(
                "run metric {:?} has an out-of-range column projection",
                description.instance
            )));
        }
    }
    Ok(())
}

fn value_error(error: Error) -> PyErr {
    PyValueError::new_err(error.to_string())
}

fn run_error(error: Error) -> PyErr {
    match error {
        Error::Io { kind, message } => match kind {
            io::ErrorKind::AlreadyExists => PyFileExistsError::new_err(message),
            io::ErrorKind::NotFound => PyFileNotFoundError::new_err(message),
            io::ErrorKind::PermissionDenied => PyPermissionError::new_err(message),
            _ => PyOSError::new_err(message),
        },
        Error::Output(message) => PyOSError::new_err(message),
        error => value_error(error),
    }
}

#[pyclass(module = "gerrytools._scoring_engine", frozen)]
struct PreparedGeometry {
    inner: Arc<RustPreparedGeometry>,
}

#[pymethods]
impl PreparedGeometry {
    #[new]
    fn new(py: Python<'_>, rows: Vec<Vec<u8>>) -> PyResult<Self> {
        let inner = py
            .detach(|| RustPreparedGeometry::from_wkb(&rows))
            .map(Arc::new)
            .map_err(value_error)?;
        Ok(Self { inner })
    }
}

#[pyfunction]
fn _prepare_geometry(py: Python<'_>, rows: Vec<Vec<u8>>) -> PyResult<PreparedGeometry> {
    let inner = py
        .detach(|| RustPreparedGeometry::from_validated_wkb(&rows))
        .map(Arc::new)
        .map_err(value_error)?;
    Ok(PreparedGeometry { inner })
}

#[pyclass(module = "gerrytools._scoring_engine")]
struct ScoringEngine {
    metrics: Vec<BackendMetric>,
    tally_bank: Option<PreparedTally>,
}

impl ScoringEngine {
    fn prepared_geometry(
        py: Python<'_>,
        geometry: &Py<PreparedGeometry>,
    ) -> Arc<RustPreparedGeometry> {
        Arc::clone(&geometry.borrow(py).inner)
    }

    fn scorer(&self) -> crate::Result<Scorer<'_>> {
        let mut scorer = Scorer::new();
        if let Some(tally) = &self.tally_bank {
            scorer.set_tally_bank(tally)?;
        }
        for (index, metric) in self.metrics.iter().enumerate() {
            metric.register(&mut scorer, format!("metric_{index}"))?;
        }
        Ok(scorer)
    }

    fn add_independent(&mut self, metric: crate::Result<PreparedMetric>) -> PyResult<()> {
        self.metrics
            .push(BackendMetric::Independent(metric.map_err(value_error)?));
        Ok(())
    }
}

#[pymethods]
impl ScoringEngine {
    #[new]
    fn new() -> Self {
        Self {
            metrics: Vec::new(),
            tally_bank: None,
        }
    }

    fn set_tally_bank(&mut self, columns: Vec<Vec<f64>>) -> PyResult<()> {
        if self.tally_bank.is_some() {
            return Err(PyValueError::new_err(
                "the scoring engine already has a tally bank",
            ));
        }
        self.tally_bank = Some(PreparedTally::new(columns).map_err(value_error)?);
        Ok(())
    }

    fn add_tally_projection(&mut self, columns: Vec<usize>) {
        self.metrics
            .push(BackendMetric::SharedTally(SharedTallyMetric::Projection(
                columns,
            )));
    }

    fn add_eguia(&mut self, party: usize, opposition: usize, benchmark: f64) -> PyResult<()> {
        let metric = SharedTallyMetric::eguia(party, opposition, benchmark).map_err(value_error)?;
        self.metrics.push(BackendMetric::SharedTally(metric));
        Ok(())
    }

    fn add_paired_derived(
        &mut self,
        kind: &str,
        party: usize,
        opposition: usize,
        turnout_model: &str,
    ) -> PyResult<()> {
        let metric = SharedTallyMetric::paired(kind, party, opposition, turnout_model)
            .map_err(value_error)?;
        self.metrics.push(BackendMetric::SharedTally(metric));
        Ok(())
    }

    fn add_population_derived(
        &mut self,
        kind: &str,
        population: usize,
        relative: bool,
    ) -> PyResult<()> {
        let metric =
            SharedTallyMetric::population(kind, population, relative).map_err(value_error)?;
        self.metrics.push(BackendMetric::SharedTally(metric));
        Ok(())
    }

    fn add_demographic_derived(
        &mut self,
        kind: &str,
        subgroup: usize,
        total: usize,
        threshold: f64,
    ) -> PyResult<()> {
        let metric = SharedTallyMetric::demographic(kind, subgroup, total, threshold)
            .map_err(value_error)?;
        self.metrics.push(BackendMetric::SharedTally(metric));
        Ok(())
    }

    fn add_cross_election_derived(
        &mut self,
        kind: &str,
        party: Vec<usize>,
        opposition: Vec<usize>,
        points_within: f64,
    ) -> PyResult<()> {
        let metric = SharedTallyMetric::cross_election(kind, party, opposition, points_within)
            .map_err(value_error)?;
        self.metrics.push(BackendMetric::SharedTally(metric));
        Ok(())
    }

    fn add_reock(&mut self, py: Python<'_>, geometry: Py<PreparedGeometry>) -> PyResult<()> {
        let geometry = Self::prepared_geometry(py, &geometry);
        let hulls = py.detach(|| geometry.unit_hulls()).map_err(value_error)?;
        let metric = PreparedReock::from_unit_hulls(hulls);
        self.metrics
            .push(BackendMetric::Independent(PreparedMetric::Reock(metric)));
        Ok(())
    }

    fn add_population_polygon(
        &mut self,
        py: Python<'_>,
        geometry: Py<PreparedGeometry>,
        population_rows: Vec<Vec<u8>>,
        weights: Vec<f64>,
    ) -> PyResult<()> {
        let geometry = Self::prepared_geometry(py, &geometry);
        let metric = py
            .detach(|| {
                PreparedPopulationPolygon::from_prepared_geometry(
                    geometry,
                    &population_rows,
                    weights,
                )
            })
            .map_err(value_error)?;
        self.metrics.push(BackendMetric::Independent(
            PreparedMetric::PopulationPolygon(metric),
        ));
        Ok(())
    }

    fn add_population_polygon_aligned(
        &mut self,
        py: Python<'_>,
        geometry: Py<PreparedGeometry>,
        weights: Vec<f64>,
    ) -> PyResult<()> {
        let geometry = Self::prepared_geometry(py, &geometry);
        let metric = py
            .detach(|| PreparedPopulationPolygon::from_aligned_prepared_geometry(geometry, weights))
            .map_err(value_error)?;
        self.metrics.push(BackendMetric::Independent(
            PreparedMetric::PopulationPolygon(metric),
        ));
        Ok(())
    }

    fn add_convex_hull_ratio(
        &mut self,
        py: Python<'_>,
        geometry: Py<PreparedGeometry>,
    ) -> PyResult<()> {
        let geometry = Self::prepared_geometry(py, &geometry);
        let hulls = py.detach(|| geometry.unit_hulls()).map_err(value_error)?;
        let metric = PreparedConvexHullRatio::from_unit_hulls(hulls);
        self.metrics
            .push(BackendMetric::Independent(PreparedMetric::ConvexHullRatio(
                metric,
            )));
        Ok(())
    }

    fn add_state_clipped_convex_hull_ratio(
        &mut self,
        py: Python<'_>,
        geometry: Py<PreparedGeometry>,
        state: Vec<u8>,
    ) -> PyResult<()> {
        let geometry = Self::prepared_geometry(py, &geometry);
        let metric = py
            .detach(|| {
                PreparedStateClippedConvexHullRatio::from_prepared_geometry_and_wkb(
                    &geometry, &state,
                )
            })
            .map_err(value_error)?;
        self.metrics.push(BackendMetric::Independent(
            PreparedMetric::StateClippedConvexHullRatio(metric),
        ));
        Ok(())
    }

    fn add_polsby_popper_geometry(
        &mut self,
        py: Python<'_>,
        geometry: Py<PreparedGeometry>,
    ) -> PyResult<()> {
        let geometry = Self::prepared_geometry(py, &geometry);
        self.add_independent(
            py.detach(|| {
                let (edges, shared) = geometry.rook_measurements()?.inputs();
                PreparedPolsbyPopper::new(
                    geometry.area_values(),
                    geometry.perimeter_values(),
                    edges,
                    shared,
                )
            })
            .map(PreparedMetric::PolsbyPopper),
        )
    }

    fn add_polsby_popper_graph_total(
        &mut self,
        areas: Vec<f64>,
        total_perimeters: Vec<f64>,
        edges: Vec<(u32, u32)>,
        shared_perimeters: Vec<f64>,
    ) -> PyResult<()> {
        self.add_independent(
            PreparedPolsbyPopper::new(areas, total_perimeters, edges, shared_perimeters)
                .map(PreparedMetric::PolsbyPopper),
        )
    }

    fn add_polsby_popper_graph_boundary(
        &mut self,
        areas: Vec<f64>,
        boundary_perimeters: Vec<f64>,
        edges: Vec<(u32, u32)>,
        shared_perimeters: Vec<f64>,
    ) -> PyResult<()> {
        self.add_independent(
            PreparedPolsbyPopper::from_boundary_perimeters(
                areas,
                boundary_perimeters,
                edges,
                shared_perimeters,
            )
            .map(PreparedMetric::PolsbyPopper),
        )
    }

    fn add_schwartzberg_geometry(
        &mut self,
        py: Python<'_>,
        geometry: Py<PreparedGeometry>,
    ) -> PyResult<()> {
        let geometry = Self::prepared_geometry(py, &geometry);
        self.add_independent(
            py.detach(|| {
                let (edges, shared) = geometry.rook_measurements()?.inputs();
                PreparedSchwartzberg::new(
                    geometry.area_values(),
                    geometry.perimeter_values(),
                    edges,
                    shared,
                )
            })
            .map(PreparedMetric::Schwartzberg),
        )
    }

    fn add_schwartzberg_graph_total(
        &mut self,
        areas: Vec<f64>,
        total_perimeters: Vec<f64>,
        edges: Vec<(u32, u32)>,
        shared_perimeters: Vec<f64>,
    ) -> PyResult<()> {
        self.add_independent(
            PreparedSchwartzberg::new(areas, total_perimeters, edges, shared_perimeters)
                .map(PreparedMetric::Schwartzberg),
        )
    }

    fn add_schwartzberg_graph_boundary(
        &mut self,
        areas: Vec<f64>,
        boundary_perimeters: Vec<f64>,
        edges: Vec<(u32, u32)>,
        shared_perimeters: Vec<f64>,
    ) -> PyResult<()> {
        self.add_independent(
            PreparedSchwartzberg::from_boundary_perimeters(
                areas,
                boundary_perimeters,
                edges,
                shared_perimeters,
            )
            .map(PreparedMetric::Schwartzberg),
        )
    }

    fn add_area_perimeter_metrics_geometry(
        &mut self,
        py: Python<'_>,
        geometry: Py<PreparedGeometry>,
    ) -> PyResult<()> {
        let geometry = Self::prepared_geometry(py, &geometry);
        self.add_independent(
            py.detach(|| {
                let (edges, shared) = geometry.rook_measurements()?.inputs();
                PreparedAreaPerimeterMetrics::new(
                    geometry.area_values(),
                    geometry.perimeter_values(),
                    edges,
                    shared,
                )
            })
            .map(PreparedMetric::AreaPerimeterMetrics),
        )
    }

    fn add_area_perimeter_metrics_graph_total(
        &mut self,
        areas: Vec<f64>,
        total_perimeters: Vec<f64>,
        edges: Vec<(u32, u32)>,
        shared_perimeters: Vec<f64>,
    ) -> PyResult<()> {
        self.add_independent(
            PreparedAreaPerimeterMetrics::new(areas, total_perimeters, edges, shared_perimeters)
                .map(PreparedMetric::AreaPerimeterMetrics),
        )
    }

    fn add_area_perimeter_metrics_graph_boundary(
        &mut self,
        areas: Vec<f64>,
        boundary_perimeters: Vec<f64>,
        edges: Vec<(u32, u32)>,
        shared_perimeters: Vec<f64>,
    ) -> PyResult<()> {
        self.add_independent(
            PreparedAreaPerimeterMetrics::from_boundary_perimeters(
                areas,
                boundary_perimeters,
                edges,
                shared_perimeters,
            )
            .map(PreparedMetric::AreaPerimeterMetrics),
        )
    }

    fn add_cut_edges(
        &mut self,
        node_count: usize,
        edges: Vec<(u32, u32)>,
        weights: Option<Vec<f64>>,
    ) -> PyResult<()> {
        let metric = match weights {
            Some(weights) => PreparedCutEdges::weighted(node_count, edges, weights),
            None => PreparedCutEdges::new(node_count, edges),
        };
        self.metrics
            .push(BackendMetric::Independent(PreparedMetric::CutEdges(
                metric.map_err(value_error)?,
            )));
        Ok(())
    }

    fn add_region_splits(&mut self, columns: Vec<Vec<Option<u32>>>) -> PyResult<()> {
        let metric = PreparedRegion::splits(columns).map_err(value_error)?;
        self.metrics
            .push(BackendMetric::Independent(PreparedMetric::Region(metric)));
        Ok(())
    }

    fn add_region_pieces(&mut self, columns: Vec<Vec<Option<u32>>>) -> PyResult<()> {
        let metric = PreparedRegion::pieces(columns).map_err(value_error)?;
        self.metrics
            .push(BackendMetric::Independent(PreparedMetric::Region(metric)));
        Ok(())
    }

    fn add_region_parts(
        &mut self,
        columns: Vec<Vec<Option<u32>>>,
        edges: Vec<(u32, u32)>,
    ) -> PyResult<()> {
        let metric = PreparedRegion::parts(columns, edges).map_err(value_error)?;
        self.metrics
            .push(BackendMetric::Independent(PreparedMetric::Region(metric)));
        Ok(())
    }

    fn add_tally_by_region(
        &mut self,
        regions: Vec<Option<u32>>,
        include_count: bool,
        values: Vec<Vec<f64>>,
    ) -> PyResult<()> {
        let metric =
            PreparedRegionTally::new(regions, include_count, values).map_err(value_error)?;
        self.metrics
            .push(BackendMetric::Independent(PreparedMetric::RegionTally(
                metric,
            )));
        Ok(())
    }

    #[pyo3(signature = (assignments, track_uniqueness=false, progress=None))]
    fn score_many(
        &self,
        py: Python<'_>,
        assignments: Vec<Vec<u16>>,
        track_uniqueness: bool,
        progress: Option<Py<PyAny>>,
    ) -> PyResult<MetricRows> {
        let scorer = self.scorer().map_err(value_error)?;
        let uniqueness = track_uniqueness
            .then(|| {
                py.detach(|| crate::scoring::uniqueness::count_assignments(&assignments))
                    .map_err(value_error)
            })
            .transpose()?;
        let Some(progress) = progress else {
            return py
                .detach(|| {
                    let (district_ids, rows) = scorer.score_batch(&assignments)?;
                    let rows = rows
                        .iter()
                        .map(|row| row.iter().map(flatten_score).collect())
                        .collect();
                    Ok((district_ids, rows, uniqueness))
                })
                .map_err(value_error);
        };
        let mut district_ids = None;
        let mut rows = Vec::with_capacity(assignments.len());
        for batch in assignments.chunks(PROGRESS_BATCH_SIZE) {
            let (batch_district_ids, batch_rows) = py
                .detach(|| scorer.score_batch(batch))
                .map_err(value_error)?;
            if district_ids
                .as_ref()
                .is_some_and(|expected| expected != &batch_district_ids)
            {
                return Err(value_error(Error::InvalidInput(
                    "district labels must be the same in every assignment".into(),
                )));
            }
            district_ids.get_or_insert(batch_district_ids);
            rows.extend(
                batch_rows
                    .iter()
                    .map(|row| row.iter().map(flatten_score).collect()),
            );
            progress.call1(py, (batch.len(),))?;
        }
        Ok((district_ids.unwrap_or_default(), rows, uniqueness))
    }

    /// Score a BEN-family stream into an atomic result directory.
    fn score_run(
        &self,
        py: Python<'_>,
        source_path: PathBuf,
        output_path: PathBuf,
        metadata_json: &str,
        stream_options: PythonStreamOptions,
        projections: Vec<(usize, Vec<usize>)>,
    ) -> PyResult<()> {
        let PythonStreamOptions {
            bendl_node_order_json,
            max_samples,
            batch_size,
            track_uniqueness,
            progress,
        } = stream_options;
        let metadata: RunMetadata = serde_json::from_str(metadata_json)
            .map_err(|error| PyValueError::new_err(format!("invalid run metadata: {error}")))?;
        validate_projections(&self.metrics, &metadata, &projections).map_err(value_error)?;

        let mut progress_error = None;
        let result = py.detach(|| {
            let source = AssignmentSource::open(source_path)?;
            source.validate_bundle_graph_node_order(bendl_node_order_json.as_deref())?;
            if let Some(total) = source.declared_sample_count() {
                let total = u64::try_from(total)
                    .map_err(|_| Error::InvalidInput("BENDL sample count is too large".into()))?;
                notify_progress(
                    &progress,
                    0,
                    Some(max_samples.map_or(total, |limit| limit.min(total))),
                    &mut progress_error,
                )?;
            }
            let scorer = self.scorer()?;
            let mut writer = RunWriter::new(output_path, metadata)?;
            let mut pending_samples = 0_u64;
            let mut pending_frames = 0_usize;
            let summary = scorer.score_stream(
                &source,
                StreamOptions {
                    max_samples,
                    batch_size,
                    track_uniqueness,
                },
                |score| {
                    let repetitions = u64::from(score.repetitions);
                    let metrics = projections
                        .iter()
                        .map(|(source, columns)| project_score(&score.metrics[*source], columns))
                        .collect::<crate::Result<_>>()?;
                    writer.push(&PlanScore {
                        sample_offset: score.sample_offset,
                        repetitions: score.repetitions,
                        accepted_index: score.accepted_index,
                        metrics,
                    })?;
                    if progress.is_some() {
                        pending_samples += repetitions;
                        pending_frames += 1;
                        if pending_frames >= batch_size {
                            notify_progress(&progress, pending_samples, None, &mut progress_error)?;
                            pending_samples = 0;
                            pending_frames = 0;
                        }
                    }
                    Ok(())
                },
            )?;
            notify_progress(&progress, pending_samples, None, &mut progress_error)?;
            writer.finish(summary)?;
            Ok(())
        });
        match progress_error {
            Some(error) => Err(error),
            None => result.map_err(run_error),
        }
    }
}

fn notify_progress(
    progress: &Option<Py<PyAny>>,
    amount: u64,
    total: Option<u64>,
    progress_error: &mut Option<PyErr>,
) -> crate::Result<()> {
    if amount == 0 && total.is_none() {
        return Ok(());
    }
    if let Some(callback) = progress {
        if let Err(error) = Python::attach(|py| callback.call1(py, (amount, total))) {
            let message = format!("progress callback failed: {error}");
            *progress_error = Some(error);
            return Err(Error::InvalidInput(message));
        }
    }
    Ok(())
}

#[pymodule]
fn _scoring_engine(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add("DEBUG_ASSERTIONS", cfg!(debug_assertions))?;
    module.add("MAX_DISTRICTS", crate::scoring::district::MAX_DISTRICTS)?;
    module.add_function(wrap_pyfunction!(_prepare_geometry, module)?)?;
    module.add_class::<PreparedGeometry>()?;
    module.add_class::<ScoringEngine>()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{flatten_score, project_score, validate_projections, BackendMetric, ScoringEngine};
    use crate::{
        MetricMetadata, MetricScore, PreparedAreaPerimeterMetrics, PreparedCutEdges,
        PreparedMetric, PreparedTally, RunMetadata, TableShape,
    };

    #[test]
    fn flattens_district_tables_column_major_and_plan_tables_directly() {
        let tally = PreparedTally::new(vec![vec![1.0, 2.0, 3.0], vec![10.0, 20.0, 30.0]]).unwrap();
        let district = MetricScore::District(tally.score(&[0, 0, 1]).unwrap());
        assert_eq!(flatten_score(&district), vec![3.0, 3.0, 30.0, 30.0]);

        let cut_edges = PreparedCutEdges::new(3, vec![(0, 1), (1, 2)]).unwrap();
        let plan = MetricScore::Plan(cut_edges.score(&[0, 0, 1]).unwrap());
        assert_eq!(flatten_score(&plan), vec![1.0]);
    }

    #[test]
    fn add_eguia_validates_the_benchmark_like_the_scoring_engine() {
        let mut scorer = ScoringEngine::new();
        assert!(scorer.add_eguia(0, 1, f64::NAN).is_err());
        assert!(scorer.add_eguia(0, 1, 1.5).is_err());
        assert!(scorer.metrics.is_empty());
        scorer.add_eguia(0, 1, 0.5).unwrap();
        assert_eq!(scorer.metrics.len(), 1);
    }

    #[test]
    fn area_perimeter_kind_mismatch_names_the_accepted_projection_kinds() {
        let metric = BackendMetric::Independent(PreparedMetric::AreaPerimeterMetrics(
            PreparedAreaPerimeterMetrics::new(vec![1.0; 2], vec![4.0; 2], vec![(0, 1)], vec![1.0])
                .unwrap(),
        ));
        let metadata_for = |kind: &str| {
            RunMetadata::new(
                None,
                vec![MetricMetadata::new(
                    kind,
                    "compactness",
                    TableShape::District,
                    vec!["value".into()],
                )],
            )
        };

        // The combined metric is only projected as its component kinds; naming the combined
        // kind itself must fail, and the diagnostic must name the kinds that would pass.
        let error = validate_projections(
            std::slice::from_ref(&metric),
            &metadata_for("area_perimeter_metrics"),
            &[(0, vec![0])],
        )
        .unwrap_err();
        let message = error.to_string();
        assert!(
            message.contains(
                "expected \"polsby_popper\" (column [0]) or \"schwartzberg\" (column [1])"
            ),
            "got {message}"
        );

        for (kind, column) in [("polsby_popper", 0), ("schwartzberg", 1)] {
            validate_projections(
                std::slice::from_ref(&metric),
                &metadata_for(kind),
                &[(0, vec![column])],
            )
            .unwrap();
        }
    }

    #[test]
    fn projects_metric_columns_in_requested_order() {
        let tally = PreparedTally::new(vec![vec![1.0, 2.0], vec![10.0, 20.0]]).unwrap();
        let score = MetricScore::District(tally.score(&[0, 1]).unwrap());
        assert_eq!(
            flatten_score(&project_score(&score, &[1, 0]).unwrap()),
            vec![10.0, 20.0, 1.0, 2.0]
        );

        let cut_edges = PreparedCutEdges::new(2, vec![(0, 1)]).unwrap();
        let score = MetricScore::Plan(cut_edges.score(&[0, 1]).unwrap());
        assert_eq!(
            flatten_score(&project_score(&score, &[0]).unwrap()),
            vec![1.0]
        );
    }
}
