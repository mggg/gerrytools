//! Rust scoring engine for GerryTools.

mod adjacency;
mod error;
mod geometry;
mod metrics;
#[cfg(feature = "python")]
mod python;
mod scoring;

pub use error::{Error, Result};
pub(crate) use geometry::PreparedGeometry;
pub use geometry::{Coordinate, PreparedUnitHulls, UnitHull, UnitHullCache};
pub use metrics::{
    IncrementalAreaPerimeterMetrics, IncrementalConvexHullRatio, IncrementalCutEdges,
    IncrementalPolsbyPopper, IncrementalPopulationPolygon, IncrementalRegion,
    IncrementalRegionTally, IncrementalReock, IncrementalSchwartzberg,
    IncrementalStateClippedConvexHullRatio, IncrementalTally, PreparedAreaPerimeterMetrics,
    PreparedConvexHullRatio, PreparedCutEdges, PreparedPolsbyPopper, PreparedPopulationPolygon,
    PreparedRegion, PreparedRegionTally, PreparedReock, PreparedSchwartzberg,
    PreparedStateClippedConvexHullRatio, PreparedTally,
};
pub use scoring::{
    AssignmentSource, DeltaChange, DistrictTable, MetricAxesMetadata, MetricMetadata, MetricRef,
    MetricScore, PlanScore, PlanTable, PreparedMetric, RegionAxesMetadata, RegionAxisMetadata,
    RegionLabelMetadata, RunMetadata, RunWriter, RunWriterOptions, Scorer, StreamOptions,
    StreamSummary, TableShape,
};

pub(crate) use metrics::SharedTallyMetric;

#[cfg(test)]
#[path = "tests/support.rs"]
mod test_support;
