use std::fmt::{Display, Formatter};
use std::io;

#[derive(Debug, PartialEq)]
/// Errors returned while preparing metrics, scoring assignments, or writing results.
pub enum Error {
    /// A filesystem or stream operation failed.
    Io {
        /// Stable category of the underlying I/O error.
        kind: io::ErrorKind,
        /// Rendered underlying I/O error.
        message: String,
    },
    /// A public API received invalid data or an inconsistent configuration.
    InvalidInput(String),
    /// Geometry decoding, validation, or overlay failed.
    Geometry(String),
    /// Arrow, Parquet, or manifest serialization failed.
    Output(String),
    /// A scorer was used without a registered metric.
    EmptyScorer,
    /// A metric was registered with an empty name.
    EmptyMetricName,
    /// A metric name was registered more than once.
    DuplicateMetricName(String),
    /// A registered metric's node count differs from the scorer's established count.
    MetricNodeCount {
        /// Registered metric name.
        metric: String,
        /// Metric's node count.
        actual: usize,
        /// Scorer's established node count.
        expected: usize,
    },
    /// A TwoDelta stream emitted changes before its initial assignment.
    DeltaBeforeSnapshot,
    /// A compressed frame represents no samples.
    ZeroRepetitionFrame {
        /// Zero-based frame index.
        frame: u64,
    },
    /// A tally was prepared without any value columns.
    EmptyTally,
    /// A tally column's length differs from the first column.
    TallyColumnLength {
        /// Zero-based column index.
        column: usize,
        /// Column length.
        actual: usize,
        /// Required node count.
        expected: usize,
    },
    /// A tally column contains an infinite or NaN value.
    NonFiniteTallyValue {
        /// Zero-based column index.
        column: usize,
    },
    /// A numeric metric input has the wrong number of node or edge values.
    NumericInputLength {
        /// Input name used in the error message.
        input: &'static str,
        /// Input length.
        actual: usize,
        /// Required length.
        expected: usize,
    },
    /// Shared-perimeter values do not match the graph edge count.
    SharedPerimeterCount {
        /// Number of shared-perimeter values.
        actual: usize,
        /// Number of graph edges.
        expected: usize,
    },
    /// Edge-weight values do not match the graph edge count.
    EdgeWeightCount {
        /// Number of edge weights.
        actual: usize,
        /// Number of graph edges.
        expected: usize,
    },
    /// An edge weight is infinite or NaN.
    NonFiniteEdgeWeight {
        /// Zero-based edge index.
        edge: usize,
    },
    /// A graph edge references a node outside the prepared node range.
    EdgeNodeOutOfRange {
        /// First endpoint.
        u: u32,
        /// Second endpoint.
        v: u32,
        /// Exclusive upper bound for node IDs.
        node_count: usize,
    },
    /// A region metric was prepared without any region columns.
    EmptyRegionMetric,
    /// A region column's length differs from the first column.
    RegionColumnLength {
        /// Zero-based column index.
        column: usize,
        /// Column length.
        actual: usize,
        /// Required node count.
        expected: usize,
    },
    /// A region-tally value column has the wrong number of node values.
    TallyByRegionValueLength {
        /// Zero-based value-column index.
        value: usize,
        /// Column length.
        actual: usize,
        /// Required node count.
        expected: usize,
    },
    /// A region-tally value is infinite or NaN.
    NonFiniteTallyByRegionValue {
        /// Zero-based value-column index.
        value: usize,
        /// Zero-based node index.
        node: usize,
    },
    /// A Polsby-Popper area or perimeter input is infinite or NaN.
    NonFinitePolsbyPopperInput,
    /// A scored district has a zero or negative perimeter.
    NonPositiveDistrictPerimeter {
        /// District label.
        district: u16,
        /// Computed district perimeter.
        perimeter: f64,
    },
    /// A unit geometry has a non-finite or nonpositive area.
    InvalidGeometryArea {
        /// Zero-based unit index.
        unit: usize,
        /// Invalid area.
        area: f64,
    },
    /// A unit hull has fewer than three points.
    InvalidGeometryPointCount {
        /// Zero-based unit index.
        unit: usize,
        /// Number of hull points.
        actual: usize,
    },
    /// A unit hull contains an infinite or NaN coordinate.
    NonFiniteGeometryPoint {
        /// Zero-based unit index.
        unit: usize,
        /// Zero-based point index.
        point: usize,
    },
    /// Defensive: unit validation enforces at least 3 hull points, so scoring cannot reach this.
    CannotComputeReockCircle {
        /// District label.
        district: u16,
    },
    /// A scored district has a non-finite or nonpositive area.
    InvalidDistrictArea {
        /// District label.
        district: u16,
        /// Invalid district area.
        area: f64,
    },
    /// Invalid enclosing-shape area; `metric` names the shape in the rendered message.
    InvalidEnclosureArea {
        /// Enclosing shape used by the metric.
        metric: &'static str,
        /// District label.
        district: u16,
        /// Invalid enclosing area.
        area: f64,
    },
    /// Score outside its possible range; `metric` names the score in the rendered message.
    ImpossibleScore {
        /// Metric score name.
        metric: &'static str,
        /// District label.
        district: u16,
        /// Value outside the metric's defined range.
        score: f64,
    },
    /// A state geometry leaves a material part of a unit geometry uncovered.
    StateGeometryCoverage {
        /// Zero-based unit index.
        unit: usize,
        /// Area outside the state geometry.
        uncovered_area: f64,
        /// Maximum permitted overlay residue.
        tolerance: f64,
    },
    /// Population geometry, weight, and owner arrays have different lengths.
    PopulationObservationLength {
        /// Number of population geometries.
        geometries: usize,
        /// Number of population weights.
        weights: usize,
        /// Number of population owners.
        owners: usize,
    },
    /// A population surface contains no observations.
    EmptyPopulationSurface,
    /// A population observation has a negative, infinite, or NaN weight.
    InvalidPopulationWeight {
        /// Zero-based observation index.
        observation: usize,
        /// Invalid weight.
        weight: f64,
    },
    /// A population observation references a nonexistent owner node.
    PopulationOwnerOutOfRange {
        /// Zero-based observation index.
        observation: usize,
        /// Referenced owner node.
        owner: usize,
        /// Exclusive upper bound for owner nodes.
        node_count: usize,
    },
    /// A population observation geometry has a non-finite or nonpositive area.
    InvalidPopulationGeometryArea {
        /// Zero-based observation index.
        observation: usize,
        /// Invalid geometry area.
        area: f64,
    },
    /// A population geometry extends materially outside its owner geometry.
    PopulationGeometryOutsideOwner {
        /// Zero-based observation index.
        observation: usize,
        /// Owner node index.
        owner: usize,
        /// Area outside the owner geometry.
        uncovered_area: f64,
        /// Maximum permitted overlay residue.
        tolerance: f64,
    },
    /// A population geometry is not covered by exactly one evaluator geometry.
    PopulationGeometryOwnerCount {
        /// Zero-based observation index.
        observation: usize,
        /// Number of evaluator geometries that cover the observation.
        count: usize,
    },
    /// Summing an owner's population weights produced an infinite or NaN total.
    NonFinitePopulationOwnerTotal {
        /// Owner node index.
        owner: usize,
    },
    /// Summing all population weights produced an infinite or NaN total.
    NonFinitePopulationTotal,
    /// A population surface has no positive-weight observation.
    NoPositivePopulation,
    /// Invalid population sum; `kind` is "owned" or "hull" in the rendered message.
    InvalidDistrictPopulation {
        /// Population sum being validated, either `"owned"` or `"hull"`.
        kind: &'static str,
        /// District label.
        district: u16,
        /// Invalid population sum.
        population: f64,
    },
    /// Incremental state does not contain a node in its claimed district.
    IncrementalNodeMembership {
        /// Zero-based node index.
        node: usize,
        /// Claimed district label.
        district: u16,
    },
    /// An assignment's length differs from the prepared metric's node count.
    AssignmentLength {
        /// Assignment length.
        actual: usize,
        /// Required node count.
        expected: usize,
    },
    /// Legacy district-limit error retained for source compatibility.
    ///
    /// Dynamic district storage no longer emits this variant.
    DistrictLimitExceeded {
        /// Invalid district label.
        district: u16,
        /// Exclusive upper bound for district labels.
        limit: u16,
    },
    /// A delta references a node outside the current assignment.
    DeltaNodeOutOfRange {
        /// Referenced node index.
        node: usize,
        /// Current assignment length.
        assignment_len: usize,
    },
    /// A delta's old label differs from the current assignment.
    DeltaOldLabelMismatch {
        /// Referenced node index.
        node: usize,
        /// Label in the current assignment.
        expected: u16,
        /// Old label supplied by the delta.
        actual: u16,
    },
    /// Delta changes are duplicated or not ordered by increasing node index.
    DeltaNodesNotStrictlyIncreasing {
        /// Previous node index.
        previous: usize,
        /// Current node index.
        node: usize,
    },
}

impl Display for Error {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io { message, .. }
            | Self::InvalidInput(message)
            | Self::Geometry(message)
            | Self::Output(message) => formatter.write_str(message),
            Self::EmptyScorer => formatter.write_str("a scorer requires at least one metric"),
            Self::EmptyMetricName => formatter.write_str("metric names cannot be empty"),
            Self::DuplicateMetricName(name) => {
                write!(formatter, "metric name {name:?} is already registered")
            }
            Self::MetricNodeCount {
                metric,
                actual,
                expected,
            } => write!(
                formatter,
                "metric {metric:?} has {actual} nodes; expected {expected}"
            ),
            Self::DeltaBeforeSnapshot => {
                formatter.write_str("TwoDelta change appeared before its initial snapshot")
            }
            Self::ZeroRepetitionFrame { frame } => write!(
                formatter,
                "frame {frame} has a zero repetition count; every frame must represent at least \
                 one sample"
            ),
            Self::EmptyTally => write!(formatter, "a prepared tally requires at least one column"),
            Self::TallyColumnLength {
                column,
                actual,
                expected,
            } => write!(
                formatter,
                "tally column {column} has length {actual}; expected {expected}"
            ),
            Self::NonFiniteTallyValue { column } => {
                write!(
                    formatter,
                    "tally column {column} contains a non-finite value"
                )
            }
            Self::NumericInputLength {
                input,
                actual,
                expected,
            } => write!(
                formatter,
                "{input} has length {actual}; expected {expected}"
            ),
            Self::SharedPerimeterCount { actual, expected } => write!(
                formatter,
                "shared perimeter count is {actual}; expected {expected} graph edges"
            ),
            Self::EdgeWeightCount { actual, expected } => write!(
                formatter,
                "edge weight count is {actual}; expected {expected} graph edges"
            ),
            Self::NonFiniteEdgeWeight { edge } => {
                write!(formatter, "graph edge {edge} has a non-finite weight")
            }
            Self::EdgeNodeOutOfRange { u, v, node_count } => write!(
                formatter,
                "graph edge ({u}, {v}) references a node outside node count {node_count}"
            ),
            Self::EmptyRegionMetric => {
                formatter.write_str("a prepared region metric requires at least one column")
            }
            Self::RegionColumnLength {
                column,
                actual,
                expected,
            } => write!(
                formatter,
                "region column {column} has length {actual}; expected {expected}"
            ),
            Self::TallyByRegionValueLength {
                value,
                actual,
                expected,
            } => write!(
                formatter,
                "tally-by-region value {value} has length {actual}; expected {expected}"
            ),
            Self::NonFiniteTallyByRegionValue { value, node } => {
                write!(
                    formatter,
                    "tally-by-region value {value} at node {node} is not finite"
                )
            }
            Self::NonFinitePolsbyPopperInput => {
                write!(formatter, "Polsby-Popper inputs contain a non-finite value")
            }
            Self::NonPositiveDistrictPerimeter {
                district,
                perimeter,
            } => write!(
                formatter,
                "district {district} has nonpositive perimeter {perimeter}"
            ),
            Self::InvalidGeometryArea { unit, area } => {
                write!(formatter, "geometry unit {unit} has invalid area {area}")
            }
            Self::InvalidGeometryPointCount { unit, actual } => write!(
                formatter,
                "geometry unit {unit} has {actual} hull points; expected at least 3"
            ),
            Self::NonFiniteGeometryPoint { unit, point } => {
                write!(
                    formatter,
                    "geometry unit {unit} has a non-finite hull point at {point}"
                )
            }
            Self::CannotComputeReockCircle { district } => write!(
                formatter,
                "cannot compute a minimum enclosing circle for district {district}"
            ),
            Self::InvalidDistrictArea { district, area } => {
                write!(formatter, "district {district} has invalid area {area}")
            }
            Self::InvalidEnclosureArea {
                metric,
                district,
                area,
            } => write!(
                formatter,
                "district {district} has invalid {metric} area {area}"
            ),
            Self::ImpossibleScore {
                metric,
                district,
                score,
            } => write!(
                formatter,
                "district {district} has impossible {metric} {score}"
            ),
            Self::StateGeometryCoverage {
                unit,
                uncovered_area,
                tolerance,
            } => write!(
                formatter,
                "state geometry leaves area {uncovered_area} of geometry unit {unit} uncovered; \
                 tolerance is {tolerance}"
            ),
            Self::PopulationObservationLength {
                geometries,
                weights,
                owners,
            } => write!(
                formatter,
                "population surface has {geometries} geometries, {weights} weights, and {owners} \
                 owners"
            ),
            Self::EmptyPopulationSurface => {
                formatter.write_str("population surface requires at least one observation")
            }
            Self::InvalidPopulationWeight {
                observation,
                weight,
            } => write!(
                formatter,
                "population observation {observation} has invalid weight {weight}"
            ),
            Self::PopulationOwnerOutOfRange {
                observation,
                owner,
                node_count,
            } => write!(
                formatter,
                "population observation {observation} owner {owner} is outside node count \
                 {node_count}"
            ),
            Self::InvalidPopulationGeometryArea { observation, area } => write!(
                formatter,
                "population observation {observation} has invalid geometry area {area}"
            ),
            Self::PopulationGeometryOutsideOwner {
                observation,
                owner,
                uncovered_area,
                tolerance,
            } => write!(
                formatter,
                "population observation {observation} extends {uncovered_area} area outside owner \
                 geometry {owner}; tolerance is {tolerance}"
            ),
            Self::PopulationGeometryOwnerCount { observation, count } => write!(
                formatter,
                "alternative population geometry {observation} must be covered by exactly one \
                 evaluator geometry; found {count}"
            ),
            Self::NonFinitePopulationOwnerTotal { owner } => {
                write!(
                    formatter,
                    "population owner {owner} has a non-finite total weight"
                )
            }
            Self::NonFinitePopulationTotal => {
                formatter.write_str("population observations have a non-finite total weight")
            }
            Self::NoPositivePopulation => {
                formatter.write_str("population surface requires positive total weight")
            }
            Self::InvalidDistrictPopulation {
                kind,
                district,
                population,
            } => write!(
                formatter,
                "district {district} has invalid {kind} population {population}"
            ),
            Self::IncrementalNodeMembership { node, district } => write!(
                formatter,
                "node {node} is not in incremental district {district} state"
            ),
            Self::AssignmentLength { actual, expected } => write!(
                formatter,
                "assignment has length {actual}; expected {expected}"
            ),
            Self::DistrictLimitExceeded { district, limit } => write!(
                formatter,
                "district id {district} exceeds the current {limit}-district limit"
            ),
            Self::DeltaNodeOutOfRange {
                node,
                assignment_len,
            } => write!(
                formatter,
                "delta references node {node} outside assignment length {assignment_len}"
            ),
            Self::DeltaOldLabelMismatch {
                node,
                expected,
                actual,
            } => write!(
                formatter,
                "delta old label mismatch at node {node}: expected {expected}, got {actual}"
            ),
            Self::DeltaNodesNotStrictlyIncreasing { previous, node } => write!(
                formatter,
                "delta node IDs must be strictly increasing; got node {node} after {previous}"
            ),
        }
    }
}

impl std::error::Error for Error {}

impl From<io::Error> for Error {
    fn from(error: io::Error) -> Self {
        Self::Io {
            kind: error.kind(),
            message: error.to_string(),
        }
    }
}

impl From<arrow_schema::ArrowError> for Error {
    fn from(error: arrow_schema::ArrowError) -> Self {
        Self::Output(format!("Arrow output error: {error}"))
    }
}

impl From<parquet::errors::ParquetError> for Error {
    fn from(error: parquet::errors::ParquetError) -> Self {
        Self::Output(format!("Parquet output error: {error}"))
    }
}

/// Result type used by the scoring engine.
pub type Result<T> = std::result::Result<T, Error>;
