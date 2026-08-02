use super::*;
use crate::test_support::TempPath;
use crate::{DistrictTable, MetricScore, PlanTable};
use arrow_array::{Float64Array, UInt64Array};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use std::fs::File;

fn metadata() -> RunMetadata {
    let mut tally_options = Map::new();
    tally_options.insert("source".into(), json!("graph"));
    RunMetadata::new(
        Some("plans.bendl".into()),
        vec![
            MetricMetadata::new(
                "tally",
                "population",
                TableShape::District,
                vec!["total".into(), "vap".into()],
            )
            .with_options(tally_options),
            MetricMetadata::new(
                "cut_edges",
                "cut_edges",
                TableShape::Plan,
                vec!["count".into()],
            ),
        ],
    )
}

fn region_metadata(labels: Vec<RegionLabelMetadata>, subkeys: Vec<String>) -> MetricMetadata {
    MetricMetadata::new("tally_by_region", "counties", TableShape::Region, subkeys)
        .with_dtypes(vec!["int".into()])
        .with_region_axes(RegionAxesMetadata {
            metric: vec!["count".into()],
            region: RegionAxisMetadata {
                name: "county".into(),
                labels,
            },
        })
}

fn row(
    sample_offset: u64,
    repetitions: u16,
    accepted_index: u64,
    districts: Vec<u16>,
    district_values: Vec<f64>,
    plan_value: f64,
) -> PlanScore {
    PlanScore {
        sample_offset,
        repetitions,
        accepted_index,
        metrics: vec![
            MetricScore::District(DistrictTable::new(districts.clone(), district_values, 2)),
            MetricScore::Plan(PlanTable::new(districts, vec![plan_value]).unwrap()),
        ],
    }
}

fn batches(path: &Path) -> Vec<RecordBatch> {
    ParquetRecordBatchReaderBuilder::try_new(File::open(path).unwrap())
        .unwrap()
        .with_batch_size(1)
        .build()
        .unwrap()
        .map(|batch| batch.unwrap())
        .collect()
}

fn run_name(output: &TempPath) -> String {
    output
        .path()
        .file_name()
        .unwrap()
        .to_string_lossy()
        .to_string()
}

fn flat_table(output: &TempPath, metric: &str) -> PathBuf {
    output
        .path()
        .join(format!("{metric}__{}.parquet", run_name(output)))
}

fn manifest_path(output: &TempPath) -> PathBuf {
    output
        .path()
        .join(format!("manifest__{}.json", run_name(output)))
}

fn grouped_table(output: &TempPath, metric: &str, file: &str) -> PathBuf {
    let name = run_name(output);
    output
        .path()
        .join(format!("{metric}/{file}__{name}.parquet"))
}

#[test]
fn writes_version_two_district_and_plan_tables_atomically() {
    let output = TempPath::new("run");
    let mut writer = RunWriter::with_options(
        output.path(),
        metadata(),
        RunWriterOptions { batch_rows: 1 },
    )
    .unwrap();
    assert!(!output.path().exists());

    writer
        .push(&row(0, 2, 0, vec![1, 3], vec![10.0, 20.0, 7.0, 9.0], 4.0))
        .unwrap();
    assert!(!output.path().exists());
    writer
        .push(&row(2, 1, 1, vec![1, 3], vec![11.0, 21.0, 8.0, 10.0], 5.0))
        .unwrap();
    writer
        .finish(StreamSummary {
            samples: 3,
            accepted: 2,
            unique_plans: Some(2),
            unique_districts: Some(4),
        })
        .unwrap();

    let manifest: Value =
        serde_json::from_reader(File::open(manifest_path(&output)).unwrap()).unwrap();
    assert_eq!(manifest["format_version"], 2);
    assert_eq!(manifest["district_ids"], json!([1, 3]));
    assert_eq!(
        manifest["summary"],
        json!({
            "samples": 3,
            "accepted": 2,
            "unique_plans": 2,
            "unique_districts": 4,
        })
    );
    assert_eq!(manifest["metrics"][0]["shape"], "district");
    assert_eq!(manifest["metrics"][1]["shape"], "plan");
    assert_eq!(
        manifest["metrics"][0]["axes"],
        json!({"metric": ["total", "vap"]})
    );
    assert_eq!(manifest["metrics"][1]["axes"], json!({"metric": ["count"]}));
    assert_eq!(manifest["metrics"][0]["dtypes"], json!(["float", "float"]));
    assert_eq!(manifest["metrics"][1]["dtypes"], json!(["float"]));
    assert_eq!(manifest["metrics"][0]["options"]["source"], "graph");
    for metric in manifest["metrics"].as_array().unwrap() {
        for table in metric["tables"].as_array().unwrap() {
            let path = output.path().join(table["path"].as_str().unwrap());
            let (size, sha256) = file_integrity(&path).unwrap();
            assert_eq!(table["size"], size);
            assert_eq!(table["sha256"], sha256);
        }
    }

    let tally = batches(&grouped_table(&output, "population", "total_tallies"));
    assert_eq!(tally.len(), 2);
    assert_eq!(
        tally[0]
            .schema()
            .fields()
            .iter()
            .map(|field| field.name())
            .collect::<Vec<_>>(),
        [
            "sample_offset",
            "repetitions",
            "accepted_index",
            "total__district_1",
            "total__district_3",
        ]
    );
    assert_eq!(
        tally[1]
            .column(0)
            .as_any()
            .downcast_ref::<UInt64Array>()
            .unwrap()
            .value(0),
        2
    );
    assert_eq!(
        tally[1]
            .column(4)
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap()
            .value(0),
        21.0
    );
    let vap = batches(&grouped_table(&output, "population", "vap_tallies"));
    assert_eq!(
        vap[1]
            .column(4)
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap()
            .value(0),
        10.0
    );

    let plan = batches(&flat_table(&output, "cut_edges"));
    assert_eq!(
        plan[0]
            .schema()
            .fields()
            .iter()
            .map(|field| field.name())
            .collect::<Vec<_>>(),
        ["sample_offset", "repetitions", "accepted_index", "count"]
    );
    assert_eq!(
        plan[1]
            .column(3)
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap()
            .value(0),
        5.0
    );
}

#[test]
fn failed_push_poisoning_leaves_no_published_or_temporary_directory() {
    let output = TempPath::new("failed-run");
    let name = output
        .path()
        .file_name()
        .unwrap()
        .to_string_lossy()
        .to_string();
    let mut writer = RunWriter::new(output.path(), metadata()).unwrap();
    let bad_row = row(0, 1, 0, vec![1, 3], vec![10.0, 20.0, 7.0, 9.0], 4.0);
    let mut bad_row = bad_row;
    bad_row.metrics[1] = MetricScore::Plan(PlanTable::new(vec![1, 2], vec![4.0]).unwrap());

    assert!(writer.push(&bad_row).is_err());
    assert!(writer
        .push(&bad_row)
        .unwrap_err()
        .to_string()
        .contains("unusable"));
    assert!(writer
        .finish(StreamSummary {
            samples: 0,
            accepted: 0,
            unique_plans: None,
            unique_districts: None,
        })
        .unwrap_err()
        .to_string()
        .contains("failed push"));
    assert!(!output.path().exists());
    let prefix = format!(".{name}.tmp-");
    assert!(!std::fs::read_dir(output.path().parent().unwrap())
        .unwrap()
        .map(|entry| entry.unwrap())
        .any(|entry| entry.file_name().to_string_lossy().starts_with(&prefix)));
}

#[test]
fn rejects_a_district_set_change_across_rows() {
    let output = TempPath::new("changed-districts-run");
    let mut writer = RunWriter::new(output.path(), metadata()).unwrap();
    writer
        .push(&row(0, 1, 0, vec![1, 3], vec![10.0, 20.0, 7.0, 9.0], 4.0))
        .unwrap();

    // The row is internally consistent, so only the cross-row comparison can reject it.
    let error = writer
        .push(&row(1, 1, 1, vec![1, 2], vec![10.0, 20.0, 7.0, 9.0], 4.0))
        .unwrap_err();

    assert!(
        error.to_string().contains("district set changed"),
        "got {error}"
    );
    assert!(writer
        .push(&row(2, 1, 2, vec![1, 3], vec![10.0, 20.0, 7.0, 9.0], 4.0))
        .unwrap_err()
        .to_string()
        .contains("unusable"));
}

#[test]
fn zero_row_run_has_stable_empty_schemas() {
    let output = TempPath::new("empty-run");
    RunWriter::new(output.path(), metadata())
        .unwrap()
        .finish(StreamSummary {
            samples: 0,
            accepted: 0,
            unique_plans: None,
            unique_districts: None,
        })
        .unwrap();

    let tally = batches(&grouped_table(&output, "population", "total_tallies"));
    assert!(tally.is_empty());
    let tally_schema = ParquetRecordBatchReaderBuilder::try_new(
        File::open(grouped_table(&output, "population", "total_tallies")).unwrap(),
    )
    .unwrap()
    .schema()
    .clone();
    assert_eq!(tally_schema.fields().len(), 3);

    let plan_schema = ParquetRecordBatchReaderBuilder::try_new(
        File::open(flat_table(&output, "cut_edges")).unwrap(),
    )
    .unwrap()
    .schema()
    .clone();
    assert_eq!(
        plan_schema
            .fields()
            .iter()
            .map(|field| field.name())
            .collect::<Vec<_>>(),
        ["sample_offset", "repetitions", "accepted_index", "count"]
    );
}

#[test]
fn preserves_rows_immediately_around_the_flush_boundary() {
    for row_count in [2_u64, 3, 4] {
        let output = TempPath::new("run");
        let mut writer = RunWriter::with_options(
            output.path(),
            metadata(),
            RunWriterOptions { batch_rows: 3 },
        )
        .unwrap();
        for index in 0..row_count {
            writer
                .push(&row(
                    index,
                    1,
                    index,
                    vec![1, 3],
                    vec![index as f64, 20.0, 7.0, 9.0],
                    index as f64,
                ))
                .unwrap();
        }
        writer
            .finish(StreamSummary {
                samples: row_count,
                accepted: row_count,
                unique_plans: Some(row_count),
                unique_districts: Some(row_count * 2),
            })
            .unwrap();

        let path = flat_table(&output, "cut_edges");
        let builder = ParquetRecordBatchReaderBuilder::try_new(File::open(&path).unwrap()).unwrap();
        assert_eq!(
            builder.metadata().num_row_groups(),
            row_count.div_ceil(3) as usize
        );
        assert!(builder
            .metadata()
            .row_groups()
            .iter()
            .flat_map(|group| group.columns())
            .all(|column| column.compression() == Compression::SNAPPY));
        let offsets = batches(&path)
            .into_iter()
            .map(|batch| {
                batch
                    .column(0)
                    .as_any()
                    .downcast_ref::<UInt64Array>()
                    .unwrap()
                    .value(0)
            })
            .collect::<Vec<_>>();
        assert_eq!(offsets, (0..row_count).collect::<Vec<_>>());
    }
}

#[test]
fn validates_paths_metadata_and_summary_before_publication() {
    let output = TempPath::new("invalid-run");
    let invalid = RunMetadata::new(
        None,
        vec![MetricMetadata::new(
            "tally",
            "../escape",
            TableShape::District,
            vec!["population".into()],
        )],
    );
    assert!(RunWriter::new(output.path(), invalid)
        .err()
        .unwrap()
        .to_string()
        .contains("safe path component"));

    let collision_output = TempPath::new("collision-run");
    let collision = RunMetadata::new(
        None,
        vec![MetricMetadata::new(
            "tally",
            format!("manifest__{}.json", run_name(&collision_output)),
            TableShape::District,
            vec!["population".into()],
        )],
    );
    assert!(RunWriter::new(collision_output.path(), collision)
        .err()
        .unwrap()
        .to_string()
        .contains("conflicts"));
    assert!(!collision_output.path().exists());

    let writer = RunWriter::new(output.path(), metadata()).unwrap();
    assert!(writer
        .finish(StreamSummary {
            samples: 1,
            accepted: 1,
            unique_plans: None,
            unique_districts: None,
        })
        .unwrap_err()
        .to_string()
        .contains("writer observed"));
    assert!(!output.path().exists());

    std::fs::create_dir(output.path()).unwrap();
    File::create(output.path().join("keep")).unwrap();
    let error = RunWriter::new(output.path(), metadata()).err().unwrap();
    assert!(matches!(
        error,
        Error::Io {
            kind: io::ErrorKind::AlreadyExists,
            ..
        }
    ));
    assert!(output.path().join("keep").is_file());
}

#[test]
fn publication_does_not_replace_a_racing_output_directory() {
    let output = TempPath::new("racing-run");
    let writer = RunWriter::new(output.path(), metadata()).unwrap();
    std::fs::create_dir(output.path()).unwrap();

    let error = writer
        .finish(StreamSummary {
            samples: 0,
            accepted: 0,
            unique_plans: None,
            unique_districts: None,
        })
        .unwrap_err();

    assert!(error.to_string().contains("exist"));
    assert!(output.path().is_dir());
    assert!(std::fs::read_dir(output.path()).unwrap().next().is_none());
}

#[test]
fn validates_region_axes_before_creating_output() {
    let missing_axes = RunMetadata::new(
        None,
        vec![MetricMetadata::new(
            "tally_by_region",
            "counties",
            TableShape::Region,
            vec![],
        )],
    );
    let output = TempPath::new("missing-region-axes");
    assert!(RunWriter::new(output.path(), missing_axes)
        .err()
        .unwrap()
        .to_string()
        .contains("requires axis metadata"));
    assert!(!output.path().exists());

    let wrong_subkeys = RunMetadata::new(
        None,
        vec![region_metadata(
            vec![RegionLabelMetadata::Int { value: 1 }],
            vec!["count__region_1".into()],
        )],
    );
    let output = TempPath::new("wrong-region-subkeys");
    assert!(RunWriter::new(output.path(), wrong_subkeys)
        .err()
        .unwrap()
        .to_string()
        .contains("do not match its axes"));
    assert!(!output.path().exists());

    let duplicate_labels = RunMetadata::new(
        None,
        vec![region_metadata(
            vec![
                RegionLabelMetadata::Str { value: "1".into() },
                RegionLabelMetadata::Str { value: "1".into() },
            ],
            vec!["count__region_0".into(), "count__region_1".into()],
        )],
    );
    let output = TempPath::new("duplicate-region-labels");
    assert!(RunWriter::new(output.path(), duplicate_labels)
        .err()
        .unwrap()
        .to_string()
        .contains("duplicate region labels"));
    assert!(!output.path().exists());

    let empty_labels = RunMetadata::new(None, vec![region_metadata(vec![], vec![])]);
    let output = TempPath::new("empty-region-axis");
    RunWriter::new(output.path(), empty_labels)
        .unwrap()
        .finish(StreamSummary {
            samples: 0,
            accepted: 0,
            unique_plans: None,
            unique_districts: None,
        })
        .unwrap();
    assert!(grouped_table(&output, "counties", "count_tallies_by_region").is_file());
}
