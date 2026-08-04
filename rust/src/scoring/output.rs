use crate::{Error, MetricScore, PlanScore, Result, StreamSummary};
use arrow_array::{ArrayRef, Float64Array, RecordBatch, UInt16Array, UInt64Array};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use parquet::arrow::ArrowWriter;
use parquet::basic::Compression;
use parquet::file::properties::WriterProperties;
use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};
use std::collections::HashSet;
use std::fmt::Write as _;
use std::fs::{self, File};
use std::io::{self, Read};
use std::path::{Component, Path, PathBuf};
use std::sync::Arc;

const DEFAULT_BATCH_ROWS: usize = 1_024;
const PREFIX_COLUMN_COUNT: usize = 3;

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
/// The row and column organization of one metric's output table.
pub enum TableShape {
    /// One value per metric subkey and district.
    District,
    /// One value per metric subkey and plan.
    Plan,
    /// One value per metric, region, and district.
    Region,
}

#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
/// A region-axis label whose original string or integer type is preserved.
pub enum RegionLabelMetadata {
    /// A string-valued label.
    Str {
        /// Original region label.
        value: String,
    },
    /// An integer-valued label.
    Int {
        /// Original region label.
        value: i64,
    },
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
/// Metadata for the region dimension of a region-shaped result.
pub struct RegionAxisMetadata {
    /// Source column or semantic name of the region dimension.
    pub name: String,
    /// Region labels in result-column order.
    pub labels: Vec<RegionLabelMetadata>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
/// Metric and region dimensions for a region-shaped result.
pub struct RegionAxesMetadata {
    /// Metric-axis labels in result-column order.
    pub metric: Vec<String>,
    /// Region-axis metadata.
    pub region: RegionAxisMetadata,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(untagged)]
/// Axis metadata for ordinary or region-shaped metric output.
pub enum MetricAxesMetadata {
    /// Metric and region axes for a region-shaped result.
    Region(RegionAxesMetadata),
    /// The sole metric axis for a district- or plan-shaped result.
    Metric {
        /// Metric-axis labels in result-column order.
        metric: Vec<String>,
    },
}

/// Semantic description of one registered metric and its output columns.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct MetricMetadata {
    /// Stable metric kind used to interpret `options`.
    pub kind: String,
    /// Unique instance name and output-directory component.
    pub instance: String,
    /// Metric-specific serialized options.
    pub options: Map<String, Value>,
    /// Output table shape.
    pub shape: TableShape,
    /// Flattened value-column names in data order.
    pub subkeys: Vec<String>,
    /// Logical dtypes for the metric axis, each `"bool"`, `"float"`, or `"int"`.
    pub dtypes: Vec<String>,
    /// Axis labels needed to interpret `subkeys`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub axes: Option<MetricAxesMetadata>,
}

impl MetricMetadata {
    /// Create metadata with float dtypes and a metric axis matching `subkeys`.
    pub fn new(
        kind: impl Into<String>,
        instance: impl Into<String>,
        shape: TableShape,
        subkeys: Vec<String>,
    ) -> Self {
        let dtypes = vec!["float".into(); subkeys.len()];
        let metric = subkeys.clone();
        Self {
            kind: kind.into(),
            instance: instance.into(),
            options: Map::new(),
            shape,
            subkeys,
            dtypes,
            axes: Some(MetricAxesMetadata::Metric { metric }),
        }
    }

    /// Replace the metric-specific options.
    pub fn with_options(mut self, options: Map<String, Value>) -> Self {
        self.options = options;
        self
    }

    /// Replace the default metric axis with region axes.
    pub fn with_region_axes(mut self, axes: RegionAxesMetadata) -> Self {
        self.axes = Some(MetricAxesMetadata::Region(axes));
        self
    }

    /// Replace logical dtypes in metric-axis order.
    pub fn with_dtypes(mut self, dtypes: Vec<String>) -> Self {
        self.dtypes = dtypes;
        self
    }
}

/// Reject a run-metadata metric whose declared shape does not match the registered metric.
///
/// Shared by the engine validator (`Scorer::validate_run_metadata`) and the Python projection
/// validator (`validate_projections` in python.rs); their remaining checks differ on purpose
/// (registration identity vs projected kinds and columns) and stay separate.
pub(crate) fn check_declared_shape(
    description: &MetricMetadata,
    expected: TableShape,
) -> Result<()> {
    if description.shape != expected {
        return Err(Error::InvalidInput(format!(
            "run metric {:?} has shape {:?}; expected {expected:?}",
            description.instance, description.shape
        )));
    }
    Ok(())
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
/// Source and metric descriptions serialized into a completed run manifest.
pub struct RunMetadata {
    /// Optional assignment-source description.
    pub source: Option<String>,
    /// Metric descriptions in scorer and output order.
    pub metrics: Vec<MetricMetadata>,
}

impl RunMetadata {
    /// Create run metadata in metric result order.
    pub fn new(source: Option<String>, metrics: Vec<MetricMetadata>) -> Self {
        Self { source, metrics }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
/// Buffering options for [`RunWriter`].
pub struct RunWriterOptions {
    /// Maximum result rows buffered before writing a Parquet batch.
    pub batch_rows: usize,
}

impl Default for RunWriterOptions {
    fn default() -> Self {
        Self {
            batch_rows: DEFAULT_BATCH_ROWS,
        }
    }
}

/// Writes one versioned scoring run to a temporary sibling and publishes it with one rename.
pub struct RunWriter {
    output_path: PathBuf,
    temp_path: PathBuf,
    run_name: String,
    metadata: RunMetadata,
    writers: Vec<MetricWriter>,
    district_ids: Option<Vec<u16>>,
    samples: u64,
    accepted: u64,
    failed: bool,
    armed: bool,
}

impl RunWriter {
    /// Create a writer with default buffering options.
    pub fn new(output_path: impl AsRef<Path>, metadata: RunMetadata) -> Result<Self> {
        Self::with_options(output_path, metadata, RunWriterOptions::default())
    }

    /// Validate metadata and create a temporary sibling for an unpublished run.
    ///
    /// `output_path` must not exist. Dropping the returned writer before [`Self::finish`] removes
    /// the temporary run.
    pub fn with_options(
        output_path: impl AsRef<Path>,
        metadata: RunMetadata,
        options: RunWriterOptions,
    ) -> Result<Self> {
        let output_path = output_path.as_ref().to_path_buf();
        validate_metadata(&metadata, options)?;
        validate_output_path(&output_path)?;

        let parent = output_parent(&output_path);
        let run_name = encode_path_component(
            &output_path
                .file_name()
                .expect("validated output path has a file name")
                .to_string_lossy(),
        );
        validate_output_paths(&metadata, &run_name)?;
        fs::create_dir_all(parent)?;
        let temp_path = create_temp_directory(parent, &output_path)?;
        let writers = match create_metric_writers(
            &temp_path,
            &run_name,
            &metadata.metrics,
            options.batch_rows,
        ) {
            Ok(writers) => writers,
            Err(error) => {
                let _ = fs::remove_dir_all(&temp_path);
                return Err(error);
            }
        };

        Ok(Self {
            output_path,
            temp_path,
            run_name,
            metadata,
            writers,
            district_ids: None,
            samples: 0,
            accepted: 0,
            failed: false,
            armed: true,
        })
    }

    /// Append the next accepted frame in contiguous sample and accepted-index order.
    ///
    /// A failed push permanently invalidates the writer so partial data cannot be published.
    pub fn push(&mut self, score: &PlanScore) -> Result<()> {
        if self.failed {
            return Err(invalid_run("run writer is unusable after a failed push"));
        }
        let result = self.push_inner(score);
        if result.is_err() {
            self.failed = true;
        }
        result
    }

    fn push_inner(&mut self, score: &PlanScore) -> Result<()> {
        if score.accepted_index != self.accepted {
            return Err(invalid_run(format!(
                "accepted index is {}; expected {}",
                score.accepted_index, self.accepted
            )));
        }
        if score.sample_offset != self.samples {
            return Err(invalid_run(format!(
                "sample offset is {}; expected {}",
                score.sample_offset, self.samples
            )));
        }
        if score.repetitions == 0 {
            return Err(invalid_run(
                "a result row must represent at least one sample",
            ));
        }
        if score.metrics.len() != self.writers.len() {
            return Err(invalid_run(format!(
                "result has {} metrics; expected {}",
                score.metrics.len(),
                self.writers.len()
            )));
        }

        let district_ids = score
            .metrics
            .first()
            .expect("metadata validation requires at least one metric")
            .district_ids();
        validate_district_ids(district_ids)?;
        if !super::result::all_district_ids_match(district_ids, &score.metrics) {
            return Err(invalid_run(
                "metrics in one result row have different district sets",
            ));
        }
        if let Some(expected) = &self.district_ids {
            if expected != district_ids {
                return Err(invalid_run(format!(
                    "district set changed from {expected:?} to {district_ids:?}"
                )));
            }
        } else {
            self.district_ids = Some(district_ids.to_vec());
        }

        for (writer, metric) in self.writers.iter().zip(&score.metrics) {
            writer.validate(metric)?;
        }
        for (writer, metric) in self.writers.iter_mut().zip(&score.metrics) {
            writer.push(score, metric)?;
        }
        self.samples = self
            .samples
            .checked_add(score.repetitions as u64)
            .ok_or_else(|| invalid_run("sample count overflowed u64"))?;
        self.accepted = self
            .accepted
            .checked_add(1)
            .ok_or_else(|| invalid_run("accepted count overflowed u64"))?;
        Ok(())
    }

    /// Flush, checksum, and atomically publish a completed run.
    ///
    /// The summary counts must match all pushed frames. The destination is never overwritten.
    pub fn finish(mut self, summary: StreamSummary) -> Result<()> {
        if self.failed {
            return Err(invalid_run("cannot finish a run after a failed push"));
        }
        if summary.samples != self.samples || summary.accepted != self.accepted {
            return Err(invalid_run(format!(
                "stream summary is {summary:?}; writer observed {} samples and {} accepted frames",
                self.samples, self.accepted
            )));
        }
        validate_uniqueness_summary(summary, self.district_ids.as_deref().unwrap_or_default())?;

        for writer in std::mem::take(&mut self.writers) {
            writer.finish(self.district_ids.as_deref().unwrap_or_default())?;
        }
        let mut directories = HashSet::new();
        for metric in &self.metadata.metrics {
            for table in physical_tables(&self.run_name, metric) {
                if let Some(parent) = self.temp_path.join(table.relative_path).parent() {
                    if parent != self.temp_path {
                        directories.insert(parent.to_path_buf());
                    }
                }
            }
        }
        for directory in directories {
            sync_directory(&directory)?;
        }
        self.write_manifest(summary)?;
        sync_directory(&self.temp_path)?;
        publish_directory(&self.temp_path, &self.output_path)?;
        sync_directory(output_parent(&self.output_path))?;
        self.armed = false;
        Ok(())
    }

    fn write_manifest(&self, summary: StreamSummary) -> Result<()> {
        let metrics = self
            .metadata
            .metrics
            .iter()
            .map(|metric| -> Result<Value> {
                let tables = physical_tables(&self.run_name, metric)
                    .into_iter()
                    .map(|table| -> Result<Value> {
                        let (size, sha256) =
                            file_integrity(&self.temp_path.join(&table.relative_path))?;
                        Ok(json!({
                            "path": table.relative_path,
                            "subkeys": table.subkeys,
                            "size": size,
                            "sha256": sha256,
                        }))
                    })
                    .collect::<Result<Vec<_>>>()?;
                let mut description = json!({
                    "kind": metric.kind,
                    "instance": metric.instance,
                    "options": metric.options,
                    "shape": metric.shape,
                    "subkeys": metric.subkeys,
                    "dtypes": metric.dtypes,
                    "tables": tables,
                });
                if let Some(axes) = &metric.axes {
                    description
                        .as_object_mut()
                        .expect("metric description is an object")
                        .insert("axes".into(), json!(axes));
                }
                Ok(description)
            })
            .collect::<Result<Vec<_>>>()?;
        let mut summary_json = json!({
            "samples": summary.samples,
            "accepted": summary.accepted,
        });
        if let (Some(unique_plans), Some(unique_districts)) =
            (summary.unique_plans, summary.unique_districts)
        {
            let summary_json = summary_json
                .as_object_mut()
                .expect("run summary is a JSON object");
            summary_json.insert("unique_plans".into(), json!(unique_plans));
            summary_json.insert("unique_districts".into(), json!(unique_districts));
        }
        let manifest = json!({
            "format_version": 2,
            "source": self.metadata.source.as_ref().map(|path| json!({"path": path})),
            "summary": summary_json,
            "district_ids": self.district_ids.as_deref().unwrap_or_default(),
            "prefix_columns": [
                {"name": "sample_offset", "dtype": "uint64"},
                {"name": "repetitions", "dtype": "uint16"},
                {"name": "accepted_index", "dtype": "uint64"},
            ],
            "metrics": metrics,
        });
        let path = self
            .temp_path
            .join(format!("manifest__{}.json", self.run_name));
        let mut file = File::create(path)?;
        serde_json::to_writer_pretty(&mut file, &manifest)
            .map_err(|error| Error::Output(format!("manifest output error: {error}")))?;
        file.sync_all()?;
        Ok(())
    }
}

fn validate_uniqueness_summary(summary: StreamSummary, district_ids: &[u16]) -> Result<()> {
    let (unique_plans, unique_districts) = match (summary.unique_plans, summary.unique_districts) {
        (None, None) => return Ok(()),
        (Some(plans), Some(districts)) => (plans, districts),
        _ => {
            return Err(invalid_run(
                "unique plan and district counts must be present together",
            ))
        }
    };
    if summary.accepted == 0 {
        if unique_plans != 0 || unique_districts != 0 {
            return Err(invalid_run(
                "an empty run cannot contain unique plans or districts",
            ));
        }
        return Ok(());
    }
    if unique_plans == 0 || unique_plans > summary.accepted {
        return Err(invalid_run(
            "unique plan count must be between one and the accepted frame count",
        ));
    }
    let district_count = u64::try_from(district_ids.len())
        .map_err(|_| invalid_run("district count overflowed u64"))?;
    let maximum = summary
        .accepted
        .checked_mul(district_count)
        .ok_or_else(|| invalid_run("district occurrence count overflowed u64"))?;
    if unique_districts < district_count || unique_districts > maximum {
        return Err(invalid_run(format!(
            "unique district count must be between {district_count} and {maximum}"
        )));
    }
    Ok(())
}

fn file_integrity(path: &Path) -> Result<(u64, String)> {
    let mut file = File::open(path)?;
    let size = file.metadata()?.len();
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let count = file.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        hasher.update(&buffer[..count]);
    }
    Ok((size, format!("{:x}", hasher.finalize())))
}

#[cfg(any(target_os = "linux", target_os = "android", target_vendor = "apple"))]
fn publish_directory(source: &Path, destination: &Path) -> io::Result<()> {
    use rustix::fs::{renameat_with, RenameFlags, CWD};

    renameat_with(CWD, source, CWD, destination, RenameFlags::NOREPLACE)
        .map_err(std::io::Error::from)
}

#[cfg(not(any(target_os = "linux", target_os = "android", target_vendor = "apple")))]
fn publish_directory(source: &Path, destination: &Path) -> io::Result<()> {
    if destination.exists() {
        return Err(io::Error::new(
            io::ErrorKind::AlreadyExists,
            format!("output path {destination:?} already exists"),
        ));
    }
    fs::rename(source, destination)
}

impl Drop for RunWriter {
    fn drop(&mut self) {
        if self.armed {
            let _ = fs::remove_dir_all(&self.temp_path);
        }
    }
}

struct MetricWriter {
    shape: TableShape,
    subkeys: Vec<String>,
    tables: Vec<TableWriter>,
}

struct TableWriter {
    path: PathBuf,
    shape: TableShape,
    subkeys: Vec<String>,
    indices: Vec<usize>,
    batch_rows: usize,
    /// The schema is built once, when the first district ids are known, and reused per batch.
    writer: Option<(SchemaRef, ArrowWriter<File>)>,
    sample_offsets: Vec<u64>,
    repetitions: Vec<u16>,
    accepted_indices: Vec<u64>,
    values: Vec<Vec<f64>>,
}

impl MetricWriter {
    fn new(root: &Path, run_name: &str, metadata: &MetricMetadata, batch_rows: usize) -> Self {
        let tables = physical_tables(run_name, metadata)
            .into_iter()
            .map(|table| {
                TableWriter::new(
                    root.join(&table.relative_path),
                    metadata.shape,
                    table,
                    batch_rows,
                )
            })
            .collect();
        Self {
            shape: metadata.shape,
            subkeys: metadata.subkeys.clone(),
            tables,
        }
    }

    fn validate(&self, metric: &MetricScore) -> Result<()> {
        match (self.shape, metric) {
            (TableShape::District | TableShape::Region, MetricScore::District(table)) => {
                if table.column_count() != self.subkeys.len() {
                    return Err(invalid_run(format!(
                        "district metric has {} columns; metadata declares {}",
                        table.column_count(),
                        self.subkeys.len()
                    )));
                }
            }
            (TableShape::Plan, MetricScore::Plan(table)) => {
                if table.values().len() != self.subkeys.len() {
                    return Err(invalid_run(format!(
                        "plan metric has {} columns; metadata declares {}",
                        table.values().len(),
                        self.subkeys.len()
                    )));
                }
            }
            (expected, _) => {
                return Err(invalid_run(format!(
                    "metric result shape does not match {expected:?} metadata"
                )));
            }
        }
        Ok(())
    }

    fn push(&mut self, score: &PlanScore, metric: &MetricScore) -> Result<()> {
        for table in &mut self.tables {
            table.push(score, metric)?;
        }
        Ok(())
    }

    fn finish(self, district_ids: &[u16]) -> Result<()> {
        for table in self.tables {
            table.finish(district_ids)?;
        }
        Ok(())
    }
}

impl TableWriter {
    fn new(path: PathBuf, shape: TableShape, table: PhysicalTable, batch_rows: usize) -> Self {
        Self {
            path,
            shape,
            subkeys: table.subkeys,
            indices: table.columns,
            batch_rows,
            writer: None,
            sample_offsets: Vec::with_capacity(batch_rows),
            repetitions: Vec::with_capacity(batch_rows),
            accepted_indices: Vec::with_capacity(batch_rows),
            values: Vec::new(),
        }
    }

    fn push(&mut self, score: &PlanScore, metric: &MetricScore) -> Result<()> {
        match metric {
            MetricScore::District(table) => {
                let value_columns = self.indices.len() * table.district_ids().len();
                self.ensure_columns(value_columns);
                for (output, &subkey) in self.indices.iter().enumerate() {
                    for (district, value) in table
                        .column(subkey)
                        .expect("validated district-table column")
                        .iter()
                        .enumerate()
                    {
                        self.values[output * table.district_ids().len() + district].push(*value);
                    }
                }
            }
            MetricScore::Plan(table) => {
                self.ensure_columns(self.indices.len());
                for (output, &column) in self.indices.iter().enumerate() {
                    self.values[output].push(table.values()[column]);
                }
            }
        }

        self.sample_offsets.push(score.sample_offset);
        self.repetitions.push(score.repetitions);
        self.accepted_indices.push(score.accepted_index);
        if self.sample_offsets.len() == self.batch_rows {
            self.create_writer(metric.district_ids())?;
            self.flush()?;
        }
        Ok(())
    }

    fn ensure_columns(&mut self, count: usize) {
        if self.values.is_empty() {
            self.values = (0..count)
                .map(|_| Vec::with_capacity(self.batch_rows))
                .collect();
        }
        debug_assert_eq!(self.values.len(), count);
    }

    fn finish(mut self, district_ids: &[u16]) -> Result<()> {
        self.create_writer(district_ids)?;
        self.flush()?;
        let (_, mut writer) = self
            .writer
            .take()
            .expect("writer exists after create_writer");
        writer.finish()?;
        writer.inner().sync_all()?;
        Ok(())
    }

    /// Build the schema and open the Parquet writer once; later calls are no-ops.
    fn create_writer(&mut self, district_ids: &[u16]) -> Result<()> {
        if self.writer.is_some() {
            return Ok(());
        }
        let schema = schema(self.shape, &self.subkeys, district_ids);
        let properties = WriterProperties::builder()
            .set_compression(Compression::SNAPPY)
            .set_max_row_group_row_count(Some(self.batch_rows))
            .build();
        let writer = ArrowWriter::try_new(
            File::create(&self.path)?,
            Arc::clone(&schema),
            Some(properties),
        )?;
        self.writer = Some((schema, writer));
        Ok(())
    }

    fn flush(&mut self) -> Result<()> {
        if self.sample_offsets.is_empty() {
            return Ok(());
        }
        let (schema, writer) = self
            .writer
            .as_mut()
            .expect("writer exists before writing a batch");

        let mut arrays: Vec<ArrayRef> = vec![
            Arc::new(UInt64Array::from(std::mem::replace(
                &mut self.sample_offsets,
                Vec::with_capacity(self.batch_rows),
            ))),
            Arc::new(UInt16Array::from(std::mem::replace(
                &mut self.repetitions,
                Vec::with_capacity(self.batch_rows),
            ))),
            Arc::new(UInt64Array::from(std::mem::replace(
                &mut self.accepted_indices,
                Vec::with_capacity(self.batch_rows),
            ))),
        ];
        debug_assert_eq!(
            self.values.len(),
            schema.fields().len() - PREFIX_COLUMN_COUNT
        );
        for column in &mut self.values {
            arrays.push(Arc::new(Float64Array::from(std::mem::replace(
                column,
                Vec::with_capacity(self.batch_rows),
            ))));
        }
        let batch = RecordBatch::try_new(Arc::clone(schema), arrays)?;
        writer.write(&batch)?;
        Ok(())
    }
}

fn schema(shape: TableShape, subkeys: &[String], district_ids: &[u16]) -> SchemaRef {
    let mut fields = vec![
        Field::new("sample_offset", DataType::UInt64, false),
        Field::new("repetitions", DataType::UInt16, false),
        Field::new("accepted_index", DataType::UInt64, false),
    ];
    match shape {
        TableShape::District | TableShape::Region => {
            for subkey in subkeys {
                for district in district_ids {
                    fields.push(Field::new(
                        format!("{subkey}__district_{district}"),
                        DataType::Float64,
                        false,
                    ));
                }
            }
        }
        TableShape::Plan => {
            fields.extend(
                subkeys
                    .iter()
                    .map(|subkey| Field::new(subkey, DataType::Float64, false)),
            );
        }
    }
    Arc::new(Schema::new(fields))
}

fn validate_metadata(metadata: &RunMetadata, options: RunWriterOptions) -> Result<()> {
    if metadata.metrics.is_empty() {
        return Err(invalid_run("a run requires at least one metric"));
    }
    if options.batch_rows == 0 {
        return Err(invalid_run("batch_rows must be greater than zero"));
    }

    let mut instances = HashSet::new();
    for metric in &metadata.metrics {
        if metric.kind.is_empty() {
            return Err(invalid_run("metric kinds cannot be empty"));
        }
        validate_path_component(&metric.instance)?;
        if !instances.insert(&metric.instance) {
            return Err(invalid_run(format!(
                "metric instance {:?} is duplicated",
                metric.instance
            )));
        }
        if metric.subkeys.is_empty() && metric.shape != TableShape::Region {
            return Err(invalid_run(format!(
                "metric {:?} requires at least one subkey",
                metric.instance
            )));
        }
        validate_axes(metric)?;
        let dtype_count = metric_axis(metric).map_or(0, Vec::len);
        if metric.dtypes.len() != dtype_count
            || metric
                .dtypes
                .iter()
                .any(|dtype| !matches!(dtype.as_str(), "bool" | "float" | "int"))
        {
            return Err(invalid_run(format!(
                "metric {:?} has invalid logical dtypes",
                metric.instance
            )));
        }
        let mut subkeys = HashSet::new();
        for subkey in &metric.subkeys {
            if subkey.is_empty() {
                return Err(invalid_run(format!(
                    "metric {:?} has an empty subkey",
                    metric.instance
                )));
            }
            if !subkeys.insert(subkey) {
                return Err(invalid_run(format!(
                    "metric {:?} repeats subkey {subkey:?}",
                    metric.instance
                )));
            }
            if metric.shape == TableShape::Plan
                && matches!(
                    subkey.as_str(),
                    "sample_offset" | "repetitions" | "accepted_index"
                )
            {
                return Err(invalid_run(format!(
                    "plan subkey {subkey:?} conflicts with a prefix column"
                )));
            }
        }
    }
    Ok(())
}

fn validate_output_paths(metadata: &RunMetadata, run_name: &str) -> Result<()> {
    let mut paths = metadata
        .metrics
        .iter()
        .flat_map(|metric| physical_tables(run_name, metric))
        .map(|table| PathBuf::from(table.relative_path))
        .collect::<Vec<_>>();
    paths.push(PathBuf::from(format!("manifest__{run_name}.json")));
    for (index, left) in paths.iter().enumerate() {
        if paths[index + 1..]
            .iter()
            .any(|right| left == right || left.starts_with(right) || right.starts_with(left))
        {
            return Err(invalid_run(format!(
                "score output path {left:?} conflicts with another metric"
            )));
        }
    }
    Ok(())
}

fn validate_axes(metric: &MetricMetadata) -> Result<()> {
    match (metric.shape, &metric.axes) {
        (TableShape::Region, Some(MetricAxesMetadata::Region(axes))) => {
            if axes.metric.is_empty() {
                return Err(invalid_run(format!(
                    "region metric {:?} requires at least one metric-axis value",
                    metric.instance
                )));
            }
            if axes.region.name.is_empty() {
                return Err(invalid_run(format!(
                    "region metric {:?} requires a region-axis name",
                    metric.instance
                )));
            }
            let mut metrics = HashSet::new();
            for name in &axes.metric {
                if name.is_empty() || !metrics.insert(name) {
                    return Err(invalid_run(format!(
                        "region metric {:?} has empty or duplicate metric-axis values",
                        metric.instance
                    )));
                }
            }
            let mut labels = HashSet::new();
            if axes.region.labels.iter().any(|label| !labels.insert(label)) {
                return Err(invalid_run(format!(
                    "region metric {:?} has duplicate region labels",
                    metric.instance
                )));
            }
            let expected = axes
                .metric
                .iter()
                .flat_map(|name| {
                    (0..axes.region.labels.len())
                        .map(move |region| format!("{name}__region_{region}"))
                })
                .collect::<Vec<_>>();
            if metric.subkeys != expected {
                return Err(invalid_run(format!(
                    "region metric {:?} has subkeys that do not match its axes",
                    metric.instance
                )));
            }
        }
        (TableShape::Region, _) => {
            return Err(invalid_run(format!(
                "region metric {:?} requires axis metadata",
                metric.instance
            )));
        }
        (_, Some(MetricAxesMetadata::Region(_))) => {
            return Err(invalid_run(format!(
                "non-region metric {:?} cannot define region axes",
                metric.instance
            )));
        }
        (_, Some(MetricAxesMetadata::Metric { metric: columns })) => {
            if columns.is_empty()
                || columns.iter().any(String::is_empty)
                || columns.iter().collect::<HashSet<_>>().len() != columns.len()
                || columns != &metric.subkeys
            {
                return Err(invalid_run(format!(
                    "metric {:?} has metric-axis values that do not match its subkeys",
                    metric.instance
                )));
            }
        }
        (_, None) => {
            return Err(invalid_run(format!(
                "metric {:?} requires axis metadata",
                metric.instance
            )));
        }
    }
    Ok(())
}

fn metric_axis(metric: &MetricMetadata) -> Option<&Vec<String>> {
    match metric.axes.as_ref()? {
        MetricAxesMetadata::Region(axes) => Some(&axes.metric),
        MetricAxesMetadata::Metric { metric } => Some(metric),
    }
}

fn validate_output_path(output_path: &Path) -> Result<()> {
    if output_path.as_os_str().is_empty() || output_path.file_name().is_none() {
        return Err(invalid_run("output path must name a directory"));
    }
    if output_path.exists() {
        return Err(io::Error::new(
            io::ErrorKind::AlreadyExists,
            format!("output path {output_path:?} already exists"),
        )
        .into());
    }
    Ok(())
}

fn validate_path_component(value: &str) -> Result<()> {
    let mut components = Path::new(value).components();
    if value.is_empty()
        || !matches!(components.next(), Some(Component::Normal(_)))
        || components.next().is_some()
    {
        return Err(invalid_run(format!(
            "metric instance {value:?} must be one safe path component"
        )));
    }
    Ok(())
}

fn validate_district_ids(district_ids: &[u16]) -> Result<()> {
    if district_ids.windows(2).any(|pair| pair[0] >= pair[1]) {
        return Err(invalid_run(
            "district ids must be unique and strictly increasing",
        ));
    }
    Ok(())
}

fn create_temp_directory(parent: &Path, output_path: &Path) -> Result<PathBuf> {
    let name = output_path
        .file_name()
        .expect("validated output path has a file name")
        .to_string_lossy();
    for _ in 0..100 {
        let candidate = parent.join(format!(
            ".{name}.tmp-{}-{:016x}",
            std::process::id(),
            fastrand::u64(..)
        ));
        match fs::create_dir(&candidate) {
            Ok(()) => return Ok(candidate),
            Err(error) if error.kind() == io::ErrorKind::AlreadyExists => continue,
            Err(error) => return Err(error.into()),
        }
    }
    Err(io::Error::new(io::ErrorKind::AlreadyExists, "temporary path collision").into())
}

struct PhysicalTable {
    relative_path: String,
    subkeys: Vec<String>,
    columns: Vec<usize>,
}

fn physical_tables(run_name: &str, metric: &MetricMetadata) -> Vec<PhysicalTable> {
    if metric.kind == "tally" {
        return metric
            .subkeys
            .iter()
            .enumerate()
            .map(|(column, subkey)| PhysicalTable {
                relative_path: format!(
                    "{}/{}_tallies__{run_name}.parquet",
                    metric.instance,
                    encode_path_component(subkey)
                ),
                subkeys: vec![subkey.clone()],
                columns: vec![column],
            })
            .collect();
    }
    if metric.kind == "tally_by_region" {
        let (metric_axis, region_count) = match metric.axes.as_ref() {
            Some(MetricAxesMetadata::Region(axes)) => (&axes.metric, axes.region.labels.len()),
            _ => unreachable!("validated tally-by-region metadata has region axes"),
        };
        return metric_axis
            .iter()
            .enumerate()
            .map(|(metric_index, name)| {
                let start = metric_index * region_count;
                let end = start + region_count;
                PhysicalTable {
                    relative_path: format!(
                        "{}/{}_tallies_by_region__{run_name}.parquet",
                        metric.instance,
                        encode_path_component(name)
                    ),
                    subkeys: metric.subkeys[start..end].to_vec(),
                    columns: (start..end).collect(),
                }
            })
            .collect();
    }

    vec![PhysicalTable {
        relative_path: format!("{}__{run_name}.parquet", metric.instance),
        subkeys: metric.subkeys.clone(),
        columns: (0..metric.subkeys.len()).collect(),
    }]
}

fn encode_path_component(value: &str) -> String {
    let mut encoded = String::with_capacity(value.len());
    for byte in value.bytes() {
        if byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-' | b'.') {
            encoded.push(char::from(byte));
        } else {
            write!(&mut encoded, "%{byte:02X}").expect("writing to a string cannot fail");
        }
    }
    encoded
}

fn create_metric_writers(
    temp_path: &Path,
    run_name: &str,
    metadata: &[MetricMetadata],
    batch_rows: usize,
) -> Result<Vec<MetricWriter>> {
    metadata
        .iter()
        .map(|metric| {
            for table in physical_tables(run_name, metric) {
                if let Some(parent) = temp_path.join(table.relative_path).parent() {
                    fs::create_dir_all(parent)?;
                }
            }
            Ok(MetricWriter::new(temp_path, run_name, metric, batch_rows))
        })
        .collect()
}

/// Return an openable parent, treating the empty parent of a bare relative name as `.`.
fn output_parent(output_path: &Path) -> &Path {
    output_path
        .parent()
        .filter(|path| !path.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."))
}

fn sync_directory(path: &Path) -> Result<()> {
    #[cfg(unix)]
    File::open(path)?.sync_all()?;
    // Windows does not let std::fs::File open a directory. Metric files are already synced
    // before publication; skip the additional directory-metadata barrier there.
    #[cfg(not(unix))]
    let _ = path;
    Ok(())
}

fn invalid_run(message: impl Into<String>) -> Error {
    Error::InvalidInput(message.into())
}

#[cfg(test)]
#[path = "../tests/output.rs"]
mod tests;
