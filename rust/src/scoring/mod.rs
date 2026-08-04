pub(crate) mod delta;
pub(crate) mod district;
mod input;
pub(crate) mod output;
mod result;
pub(crate) mod uniqueness;

use self::delta::{apply_changes, validate_changes};
use self::result::all_district_ids_match;
use crate::{
    Error, IncrementalAreaPerimeterMetrics, IncrementalConvexHullRatio, IncrementalCutEdges,
    IncrementalPolsbyPopper, IncrementalPopulationPolygon, IncrementalRegion,
    IncrementalRegionTally, IncrementalReock, IncrementalSchwartzberg,
    IncrementalStateClippedConvexHullRatio, IncrementalTally, PreparedAreaPerimeterMetrics,
    PreparedConvexHullRatio, PreparedCutEdges, PreparedPolsbyPopper, PreparedPopulationPolygon,
    PreparedRegion, PreparedRegionTally, PreparedReock, PreparedSchwartzberg,
    PreparedStateClippedConvexHullRatio, PreparedTally, Result, SharedTallyMetric,
};
use ben::io::reader::{DecodeFrame, TwoDeltaFrameEvent};
use ben::BenVariant;
pub use delta::DeltaChange;
pub use input::AssignmentSource;
pub use output::{
    MetricAxesMetadata, MetricMetadata, RegionAxesMetadata, RegionAxisMetadata,
    RegionLabelMetadata, RunMetadata, RunWriter, RunWriterOptions, TableShape,
};
use rayon::prelude::*;
pub use result::{DistrictTable, MetricScore, PlanTable};
use std::path::Path;

const DEFAULT_BATCH_SIZE: usize = 256;
const MKVCHAIN_RESYNC_INTERVAL: u64 = 16;

#[derive(Debug, Clone, Copy)]
/// Controls bounded, batched scoring without changing metric definitions.
pub struct StreamOptions {
    /// Maximum expanded samples to score. The last frame's repetition count is truncated.
    pub max_samples: Option<u64>,
    /// Number of independent Standard frames in one bounded parallel batch.
    /// Other variants ignore this for scoring, but callers may use it for progress cadence.
    pub batch_size: usize,
    /// Compute label-invariant unique plan and district counts.
    pub track_uniqueness: bool,
}

impl Default for StreamOptions {
    fn default() -> Self {
        Self {
            max_samples: None,
            batch_size: DEFAULT_BATCH_SIZE,
            track_uniqueness: false,
        }
    }
}

#[derive(Debug, PartialEq)]
/// Scores for one accepted frame, with tables ordered like [`Scorer::metric_names`].
pub struct PlanScore {
    /// Zero-based expanded-sample offset of this frame.
    pub sample_offset: u64,
    /// Number of expanded samples represented by this frame.
    pub repetitions: u16,
    /// Zero-based accepted-frame index.
    pub accepted_index: u64,
    /// Metric results ordered like [`Scorer::metric_names`].
    pub metrics: Vec<MetricScore>,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
/// Counts actually emitted by a completed stream run.
pub struct StreamSummary {
    /// Number of expanded samples scored.
    pub samples: u64,
    /// Number of accepted frames scored.
    pub accepted: u64,
    /// Label-invariant unique plan count, when requested.
    pub unique_plans: Option<u64>,
    /// Label-invariant unique district count, when requested.
    pub unique_districts: Option<u64>,
}

macro_rules! metric_property {
    ($metric:ident, static $value:expr) => {{
        let _ = $metric;
        $value
    }};
    ($metric:ident, method $method:ident) => {
        $metric.$method()
    };
}

macro_rules! metric_result {
    ($state:ident, $shape:ident, direct) => {
        Ok(MetricScore::$shape($state.result()))
    };
    ($state:ident, $shape:ident, fallible) => {
        $state.result().map(MetricScore::$shape)
    };
}

macro_rules! define_metrics {
    (
        $(
            $variant:ident($prepared:ty, $incremental:ty) {
                kind: $kind_mode:ident $kind_value:tt,
                columns: $column_mode:ident $column_value:tt,
                shape: $shape:ident,
                table: $table:ident,
                result: $result_mode:ident,
            }
        ),* $(,)?
    ) => {
        /// A borrowed prepared metric, the single dispatch point for scoring.
        ///
        /// Every `&PreparedX` converts into a `MetricRef` via `From`, so [`Scorer::add`]
        /// accepts any prepared metric directly.
        #[derive(Clone, Copy)]
        pub enum MetricRef<'a> {
            $(
                #[doc = concat!("A borrowed [`", stringify!($prepared), "`].")]
                $variant(&'a $prepared),
            )*
        }

        /// An owned prepared metric, for callers that cannot hold borrows (e.g. Python).
        pub enum PreparedMetric {
            $(
                #[doc = concat!("An owned [`", stringify!($prepared), "`].")]
                $variant($prepared),
            )*
        }

        enum IncrementalState<'a> {
            $($variant($incremental),)*
        }

        $(impl<'a> From<&'a $prepared> for MetricRef<'a> {
            fn from(metric: &'a $prepared) -> Self {
                Self::$variant(metric)
            }
        })*

        impl<'a> From<&'a PreparedMetric> for MetricRef<'a> {
            fn from(metric: &'a PreparedMetric) -> Self {
                match metric {
                    $(PreparedMetric::$variant(metric) => Self::$variant(metric),)*
                }
            }
        }

        impl<'a> MetricRef<'a> {
            pub(crate) fn kind(self) -> &'static str {
                match self {
                    $(Self::$variant(metric) => {
                        metric_property!(metric, $kind_mode $kind_value)
                    },)*
                }
            }

            pub(crate) fn column_count(self) -> usize {
                match self {
                    $(Self::$variant(metric) => {
                        metric_property!(metric, $column_mode $column_value)
                    },)*
                }
            }

            pub(crate) fn shape(self) -> TableShape {
                match self {
                    $(Self::$variant(_) => TableShape::$shape,)*
                }
            }

            fn node_count(self) -> usize {
                match self {
                    $(Self::$variant(metric) => metric.node_count(),)*
                }
            }

            fn score(self, assignment: &[u16]) -> Result<MetricScore> {
                match self {
                    $(Self::$variant(metric) => {
                        metric.score(assignment).map(MetricScore::$table)
                    },)*
                }
            }

            fn incremental(self, assignment: &[u16]) -> Result<IncrementalState<'a>> {
                match self {
                    $(Self::$variant(metric) => Ok(IncrementalState::$variant(
                        metric.incremental(assignment)?,
                    )),)*
                }
            }
        }

        impl IncrementalState<'_> {
            fn reset(&mut self, assignment: &[u16]) -> Result<()> {
                match self {
                    $(Self::$variant(state) => state.reset(assignment),)*
                }
            }

            fn update_trusted(
                &mut self,
                canonical_assignment: &[u16],
                changes: &[DeltaChange],
            ) -> Result<()> {
                // Standalone states keep private assignments for their checked API. Scorer states
                // read this canonical pre-delta assignment and avoid copying the same labels.
                match self {
                    $(Self::$variant(state) => {
                        state.update_trusted(Some(canonical_assignment), changes)
                    },)*
                }
            }

            #[cfg(test)]
            fn update_checked(&mut self, changes: &[DeltaChange]) -> Result<()> {
                match self {
                    $(Self::$variant(state) => state.update(changes),)*
                }
            }

            fn result(&self) -> Result<MetricScore> {
                match self {
                    $(Self::$variant(state) => {
                        metric_result!(state, $table, $result_mode)
                    },)*
                }
            }
        }
    };
}

define_metrics!(
    Tally(PreparedTally, IncrementalTally<'a>) {
        kind: static "tally",
        columns: method column_count,
        shape: District,
        table: District,
        result: direct,
    },
    PolsbyPopper(PreparedPolsbyPopper, IncrementalPolsbyPopper<'a>) {
        kind: static "polsby_popper",
        columns: static 1,
        shape: District,
        table: District,
        result: fallible,
    },
    Schwartzberg(PreparedSchwartzberg, IncrementalSchwartzberg<'a>) {
        kind: static "schwartzberg",
        columns: static 1,
        shape: District,
        table: District,
        result: fallible,
    },
    AreaPerimeterMetrics(PreparedAreaPerimeterMetrics, IncrementalAreaPerimeterMetrics<'a>) {
        kind: static "area_perimeter_metrics",
        columns: static 2,
        shape: District,
        table: District,
        result: fallible,
    },
    Reock(PreparedReock, IncrementalReock<'a>) {
        kind: static "reock",
        columns: static 1,
        shape: District,
        table: District,
        result: direct,
    },
    PopulationPolygon(PreparedPopulationPolygon, IncrementalPopulationPolygon<'a>) {
        kind: static "population_polygon",
        columns: static 1,
        shape: District,
        table: District,
        result: direct,
    },
    ConvexHullRatio(PreparedConvexHullRatio, IncrementalConvexHullRatio<'a>) {
        kind: static "convex_hull_ratio",
        columns: static 1,
        shape: District,
        table: District,
        result: direct,
    },
    StateClippedConvexHullRatio(
        PreparedStateClippedConvexHullRatio,
        IncrementalStateClippedConvexHullRatio<'a>
    ) {
        kind: static "state_clipped_convex_hull_ratio",
        columns: static 1,
        shape: District,
        table: District,
        result: direct,
    },
    CutEdges(PreparedCutEdges, IncrementalCutEdges<'a>) {
        kind: static "cut_edges",
        columns: static 1,
        shape: Plan,
        table: Plan,
        result: direct,
    },
    Region(PreparedRegion, IncrementalRegion<'a>) {
        kind: method kind,
        columns: method column_count,
        shape: Plan,
        table: Plan,
        result: direct,
    },
    RegionTally(PreparedRegionTally, IncrementalRegionTally<'a>) {
        kind: static "tally_by_region",
        columns: method column_count,
        shape: Region,
        table: District,
        result: direct,
    },
);

#[derive(Clone)]
enum MetricSource<'a> {
    Independent(MetricRef<'a>),
    SharedTally(SharedTallyMetric),
}

impl MetricSource<'_> {
    fn kind(&self) -> &'static str {
        match self {
            Self::Independent(metric) => metric.kind(),
            Self::SharedTally(metric) => metric.kind(),
        }
    }

    fn column_count(&self) -> usize {
        match self {
            Self::Independent(metric) => metric.column_count(),
            Self::SharedTally(metric) => metric.column_count(),
        }
    }

    fn shape(&self) -> TableShape {
        match self {
            Self::Independent(metric) => metric.shape(),
            Self::SharedTally(metric) => metric.shape(),
        }
    }
}

struct MetricEntry<'a> {
    name: String,
    source: MetricSource<'a>,
    /// Column names in this metric's data order, when the caller declared them at registration.
    subkeys: Option<Vec<String>>,
}

enum LogicalState<'a> {
    Independent(Box<IncrementalState<'a>>),
    SharedTally(SharedTallyMetric),
}

struct ScorerState<'a> {
    assignment: Vec<u16>,
    tally: Option<IncrementalTally<'a>>,
    metrics: Vec<LogicalState<'a>>,
}

impl ScorerState<'_> {
    fn reset(&mut self, assignment: &[u16]) -> Result<()> {
        if let Some(tally) = &mut self.tally {
            tally.reset(assignment)?;
        }
        for metric in &mut self.metrics {
            if let LogicalState::Independent(metric) = metric {
                metric.reset(assignment)?;
            }
        }
        self.assignment.clear();
        self.assignment.extend_from_slice(assignment);
        Ok(())
    }

    fn update(&mut self, changes: &[DeltaChange]) -> Result<()> {
        // Validate before any metric can mutate, then advance the canonical assignment once.
        validate_changes(&self.assignment, changes)?;
        if let Some(tally) = &mut self.tally {
            tally.update_trusted(Some(&self.assignment), changes)?;
        }
        for metric in &mut self.metrics {
            if let LogicalState::Independent(metric) = metric {
                metric.update_trusted(&self.assignment, changes)?;
            }
        }
        apply_changes(&mut self.assignment, changes);
        Ok(())
    }

    /// Apply a snapshot frame's accompanying delta, then require the result to reproduce the
    /// decoded snapshot; a mismatch means the stream's delta and snapshot encodings disagree.
    fn apply_snapshot_delta(&mut self, snapshot: &[u16], changes: &[DeltaChange]) -> Result<()> {
        self.update(changes)?;
        if self.assignment.as_slice() != snapshot {
            return Err(Error::InvalidInput(
                "TwoDelta snapshot does not match the assignment produced by its delta".into(),
            ));
        }
        Ok(())
    }

    #[cfg(test)]
    fn update_checked(&mut self, changes: &[DeltaChange]) -> Result<()> {
        if let Some(tally) = &mut self.tally {
            tally.update(changes)?;
        }
        for metric in &mut self.metrics {
            if let LogicalState::Independent(metric) = metric {
                metric.update_checked(changes)?;
            }
        }
        apply_changes(&mut self.assignment, changes);
        Ok(())
    }

    fn result(&self) -> Result<Vec<MetricScore>> {
        let tally = self.tally.as_ref().map(IncrementalTally::result);
        self.metrics
            .iter()
            .map(|metric| match metric {
                LogicalState::Independent(metric) => metric.result(),
                LogicalState::SharedTally(metric) => metric.score(
                    tally
                        .as_ref()
                        .expect("shared tally metric requires shared tally state"),
                ),
            })
            .collect()
    }
}

/// An ordered collection of prepared metrics scored against the same assignment.
pub struct Scorer<'a> {
    metrics: Vec<MetricEntry<'a>>,
    tally_bank: Option<&'a PreparedTally>,
    node_count: Option<usize>,
}

impl<'a> Scorer<'a> {
    /// Create an empty scorer.
    pub fn new() -> Self {
        Self {
            metrics: Vec::new(),
            tally_bank: None,
            node_count: None,
        }
    }

    /// Register a prepared metric under a unique, nonempty name.
    pub fn add(&mut self, name: impl Into<String>, metric: impl Into<MetricRef<'a>>) -> Result<()> {
        let metric = metric.into();
        self.add_entry(
            name.into(),
            MetricSource::Independent(metric),
            metric.node_count(),
            None,
        )
    }

    /// Register a prepared metric together with its column names in data order.
    ///
    /// `score_run` then requires the run metadata to list exactly these subkeys in this order,
    /// so a reordered metadata column list cannot silently mislabel the output tables.
    pub fn add_with_subkeys(
        &mut self,
        name: impl Into<String>,
        metric: impl Into<MetricRef<'a>>,
        subkeys: Vec<String>,
    ) -> Result<()> {
        let name = name.into();
        let metric = metric.into();
        if subkeys.len() != metric.column_count() {
            return Err(Error::InvalidInput(format!(
                "metric {:?} registers {} subkeys; its results have {} columns",
                name,
                subkeys.len(),
                metric.column_count()
            )));
        }
        self.add_entry(
            name,
            MetricSource::Independent(metric),
            metric.node_count(),
            Some(subkeys),
        )
    }

    /// Install the district tally state shared by tally projections and formula metrics.
    pub fn set_tally_bank(&mut self, tally: &'a PreparedTally) -> Result<()> {
        if self.tally_bank.is_some() {
            return Err(Error::InvalidInput(
                "the scorer already has a shared tally bank".into(),
            ));
        }
        self.check_node_count("shared tally bank", tally.node_count())?;
        self.tally_bank = Some(tally);
        Ok(())
    }

    /// Register selected columns from the shared tally bank as a district table.
    pub fn add_tally(&mut self, name: impl Into<String>, columns: Vec<usize>) -> Result<()> {
        self.add_shared(name, SharedTallyMetric::Projection(columns))
    }

    /// Register Eguia's seat-share difference using two shared tally columns.
    pub fn add_eguia(
        &mut self,
        name: impl Into<String>,
        party: usize,
        opposition: usize,
        benchmark: f64,
    ) -> Result<()> {
        self.add_shared(
            name,
            SharedTallyMetric::eguia(party, opposition, benchmark)?,
        )
    }

    #[doc(hidden)]
    pub fn add_paired_derived(
        &mut self,
        name: impl Into<String>,
        kind: &str,
        party: usize,
        opposition: usize,
        turnout_model: &str,
    ) -> Result<()> {
        self.add_shared(
            name,
            SharedTallyMetric::paired(kind, party, opposition, turnout_model)?,
        )
    }

    #[doc(hidden)]
    pub fn add_population_derived(
        &mut self,
        name: impl Into<String>,
        kind: &str,
        population: usize,
        relative: bool,
    ) -> Result<()> {
        self.add_shared(
            name,
            SharedTallyMetric::population(kind, population, relative)?,
        )
    }

    #[doc(hidden)]
    pub fn add_demographic_derived(
        &mut self,
        name: impl Into<String>,
        kind: &str,
        subgroup: usize,
        total: usize,
        threshold: f64,
    ) -> Result<()> {
        self.add_shared(
            name,
            SharedTallyMetric::demographic(kind, subgroup, total, threshold)?,
        )
    }

    #[doc(hidden)]
    pub fn add_cross_election_derived(
        &mut self,
        name: impl Into<String>,
        kind: &str,
        party: Vec<usize>,
        opposition: Vec<usize>,
        points_within: f64,
    ) -> Result<()> {
        self.add_shared(
            name,
            SharedTallyMetric::cross_election(kind, party, opposition, points_within)?,
        )
    }

    pub(crate) fn add_shared(
        &mut self,
        name: impl Into<String>,
        metric: SharedTallyMetric,
    ) -> Result<()> {
        self.validate_tally_columns(&metric.required_columns())?;
        let node_count = self
            .tally_bank
            .expect("column validation requires a tally bank")
            .node_count();
        self.add_entry(
            name.into(),
            MetricSource::SharedTally(metric),
            node_count,
            None,
        )
    }

    fn add_entry(
        &mut self,
        name: String,
        source: MetricSource<'a>,
        node_count: usize,
        subkeys: Option<Vec<String>>,
    ) -> Result<()> {
        if name.is_empty() {
            return Err(Error::EmptyMetricName);
        }
        if self.metrics.iter().any(|entry| entry.name == name) {
            return Err(Error::DuplicateMetricName(name));
        }
        self.check_node_count(&name, node_count)?;
        self.metrics.push(MetricEntry {
            name,
            source,
            subkeys,
        });
        Ok(())
    }

    fn check_node_count(&mut self, name: &str, actual: usize) -> Result<()> {
        if let Some(expected) = self.node_count {
            if actual != expected {
                return Err(Error::MetricNodeCount {
                    metric: name.into(),
                    actual,
                    expected,
                });
            }
        } else {
            self.node_count = Some(actual);
        }
        Ok(())
    }

    fn validate_tally_columns(&self, columns: &[usize]) -> Result<()> {
        let tally = self.tally_bank.ok_or_else(|| {
            Error::InvalidInput("shared tally metrics require a tally bank".into())
        })?;
        if columns.is_empty() {
            return Err(Error::InvalidInput(
                "a tally projection requires at least one column".into(),
            ));
        }
        if columns.iter().any(|&column| column >= tally.column_count()) {
            return Err(Error::InvalidInput(
                "shared tally metric references an out-of-range column".into(),
            ));
        }
        Ok(())
    }

    /// Return registered metric names in result order.
    pub fn metric_names(&self) -> impl ExactSizeIterator<Item = &str> {
        self.metrics.iter().map(|entry| entry.name.as_str())
    }
}

impl Default for Scorer<'_> {
    fn default() -> Self {
        Self::new()
    }
}

fn delta_changes(changes: Vec<(u32, u16, u16)>) -> Vec<DeltaChange> {
    changes
        .into_iter()
        .map(|(node, old, new)| DeltaChange {
            node: node as usize,
            old,
            new,
        })
        .collect()
}

fn cap_repetitions(remaining: &mut Option<u64>, repetitions: u16) -> u16 {
    match remaining {
        Some(remaining) => {
            let keep = (*remaining).min(repetitions as u64);
            *remaining -= keep;
            keep as u16
        }
        None => repetitions,
    }
}

mod stream;

#[cfg(test)]
#[path = "../tests/stream.rs"]
mod tests;
