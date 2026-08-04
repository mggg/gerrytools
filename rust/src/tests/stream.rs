use super::*;
use crate::test_support::TempPath;
use crate::{
    DistrictTable, MetricMetadata, PlanTable, PreparedPopulationPolygon, PreparedUnitHulls,
    RegionAxesMetadata, RegionAxisMetadata, RegionLabelMetadata, RunMetadata, TableShape, UnitHull,
};
use ben::io::bundle::format::AssignmentFormat;
use ben::io::bundle::BendlWriter;
use ben::io::writer::{BenStreamWriter, XzEncodeOptions};
use geo::{polygon, MultiPolygon};
use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::Arc;

fn plans() -> Vec<Vec<u16>> {
    let first = vec![1, 1, 2, 2];
    let second = vec![1, 2, 1, 2];
    let third = vec![2, 2, 1, 1];
    vec![
        first.clone(),
        first,
        second,
        third.clone(),
        third.clone(),
        third,
    ]
}

fn output_name(path: &Path) -> String {
    path.file_name().unwrap().to_string_lossy().to_string()
}

fn flat_output(path: &Path, metric: &str) -> PathBuf {
    path.join(format!("{metric}__{}.parquet", output_name(path)))
}

fn grouped_output(path: &Path, metric: &str, key: &str) -> PathBuf {
    path.join(metric)
        .join(format!("{key}__{}.parquet", output_name(path)))
}

fn output_manifest(path: &Path) -> PathBuf {
    path.join(format!("manifest__{}.json", output_name(path)))
}

fn county_tally_metadata() -> MetricMetadata {
    MetricMetadata::new(
        "tally_by_region",
        "county_totals",
        TableShape::Region,
        vec!["count__region_0".into(), "count__region_1".into()],
    )
    .with_dtypes(vec!["int".into()])
    .with_region_axes(RegionAxesMetadata {
        metric: vec!["count".into()],
        region: RegionAxisMetadata {
            name: "county".into(),
            labels: vec![
                RegionLabelMetadata::Int { value: 10 },
                RegionLabelMetadata::Int { value: 20 },
            ],
        },
    })
}

fn write_ben(path: &Path, variant: BenVariant) {
    write_ben_plans(path, variant, plans());
}

fn write_ben_plans(path: &Path, variant: BenVariant, plans: Vec<Vec<u16>>) {
    let mut writer = BenStreamWriter::for_ben(File::create(path).unwrap(), variant).unwrap();
    for plan in plans {
        writer.write_assignment(plan).unwrap();
    }
    writer.finish().unwrap();
}

#[test]
fn score_stream_rejects_a_scorer_without_metrics() {
    let input = TempPath::new("ben");
    write_ben(input.path(), BenVariant::Standard);
    let source = AssignmentSource::open(input.path()).unwrap();
    let tally = PreparedTally::new(vec![vec![1.0, 2.0, 3.0, 4.0]]).unwrap();
    let mut scorer = Scorer::new();
    scorer.set_tally_bank(&tally).unwrap();

    let error = scorer
        .score_stream(&source, StreamOptions::default(), |_| Ok(()))
        .unwrap_err();

    assert!(matches!(error, Error::EmptyScorer));
}

#[test]
fn twodelta_midstream_snapshot_scores_like_standard() {
    let standard = TempPath::new("ben");
    let twodelta = TempPath::new("ben");
    let plans = vec![vec![1, 1, 2, 2], vec![1, 2, 1, 2], vec![3, 3, 4, 4]];
    write_ben_plans(standard.path(), BenVariant::Standard, plans.clone());
    write_ben_plans(twodelta.path(), BenVariant::TwoDelta, plans);

    let (standard_summary, standard_rows) = score(standard.path(), None);
    let (twodelta_summary, twodelta_rows) = score(twodelta.path(), None);

    assert_eq!(twodelta_summary, standard_summary);
    assert_eq!(expanded(&standard_rows), expanded(&twodelta_rows));
}

#[test]
fn twodelta_snapshot_disagreeing_with_its_delta_is_rejected() {
    let cut_edges = PreparedCutEdges::new(4, vec![(0, 1), (1, 2), (2, 3)]).unwrap();
    let mut scorer = Scorer::new();
    scorer.add("cut_edges", &cut_edges).unwrap();
    let changes = [DeltaChange {
        node: 0,
        old: 1,
        new: 2,
    }];

    // The delta yields [2, 1, 2, 2]; the snapshot claims [2, 2, 2, 2].
    let mut state = scorer.incremental_state(&[1, 1, 2, 2]).unwrap();
    let error = state
        .apply_snapshot_delta(&[2, 2, 2, 2], &changes)
        .unwrap_err();
    assert!(error.to_string().contains("snapshot does not match"));

    let mut state = scorer.incremental_state(&[1, 1, 2, 2]).unwrap();
    state.apply_snapshot_delta(&[2, 1, 2, 2], &changes).unwrap();
    assert_eq!(state.assignment, [2, 1, 2, 2]);
}

fn write_xben(path: &Path, variant: BenVariant) {
    let mut writer =
        BenStreamWriter::for_xben(File::create(path).unwrap(), variant, XzEncodeOptions::new())
            .unwrap();
    for plan in plans() {
        writer.write_assignment(plan).unwrap();
    }
    writer.finish().unwrap();
}

fn write_bendl(path: &Path, xben: bool, variant: BenVariant) {
    let format = if xben {
        AssignmentFormat::Xben
    } else {
        AssignmentFormat::Ben
    };
    let writer = BendlWriter::new(File::create(path).unwrap(), format).unwrap();
    let mut session = writer.into_stream_session().unwrap();
    if xben {
        let mut stream =
            BenStreamWriter::for_xben(&mut session, variant, XzEncodeOptions::new()).unwrap();
        for plan in plans() {
            stream.write_assignment(plan).unwrap();
        }
        stream.finish().unwrap();
    } else {
        let mut stream = BenStreamWriter::for_ben(&mut session, variant).unwrap();
        for plan in plans() {
            stream.write_assignment(plan).unwrap();
        }
        stream.finish().unwrap();
    }
    session
        .finish_into_writer(plans().len() as i64)
        .finish()
        .unwrap();
}

fn square(x: f64) -> UnitHull {
    let point = |x, y| crate::Coordinate { x, y };
    UnitHull::new(
        1.0,
        vec![
            point(x, 0.0),
            point(x + 1.0, 0.0),
            point(x + 1.0, 1.0),
            point(x, 1.0),
        ],
    )
}

fn square_wkb(x: f64) -> Vec<u8> {
    let ring: [(f64, f64); 5] = [(x, 0.0), (x + 1.0, 0.0), (x + 1.0, 1.0), (x, 1.0), (x, 0.0)];
    let mut bytes = vec![1];
    bytes.extend_from_slice(&3_u32.to_le_bytes());
    bytes.extend_from_slice(&1_u32.to_le_bytes());
    bytes.extend_from_slice(&(ring.len() as u32).to_le_bytes());
    for (x, y) in ring {
        bytes.extend_from_slice(&x.to_le_bytes());
        bytes.extend_from_slice(&y.to_le_bytes());
    }
    bytes
}

fn score(path: &Path, max_samples: Option<u64>) -> (StreamSummary, Vec<PlanScore>) {
    score_with_batch(path, max_samples, 2)
}

fn score_with_batch(
    path: &Path,
    max_samples: Option<u64>,
    batch_size: usize,
) -> (StreamSummary, Vec<PlanScore>) {
    let tally = PreparedTally::new(vec![vec![1.0, 2.0, 3.0, 4.0]]).unwrap();
    let polsby = PreparedPolsbyPopper::new(
        vec![1.0; 4],
        vec![4.0; 4],
        vec![(0, 1), (1, 2), (2, 3)],
        vec![1.0; 3],
    )
    .unwrap();
    let unit_hulls = Arc::new(
        PreparedUnitHulls::new(vec![square(0.0), square(2.0), square(4.0), square(6.0)]).unwrap(),
    );
    let reock = PreparedReock::from_unit_hulls(Arc::clone(&unit_hulls));
    let geometry_rows = [
        square_wkb(0.0),
        square_wkb(2.0),
        square_wkb(4.0),
        square_wkb(6.0),
    ];
    let population_polygon = PreparedPopulationPolygon::from_unit_hulls_and_wkb(
        Arc::clone(&unit_hulls),
        &geometry_rows,
        &geometry_rows,
        vec![1.0, 2.0, 3.0, 4.0],
        vec![0, 1, 2, 3],
    )
    .unwrap();
    let convex_hull_ratio = PreparedConvexHullRatio::from_unit_hulls(unit_hulls);
    let cut_edges = PreparedCutEdges::new(4, vec![(0, 1), (1, 2), (2, 3)]).unwrap();
    let region_columns = vec![
        vec![Some(10), Some(10), Some(20), Some(20)],
        vec![Some(7), Some(8), Some(8), None],
    ];
    let region_splits = PreparedRegion::splits(region_columns.clone()).unwrap();
    let region_pieces = PreparedRegion::pieces(region_columns.clone()).unwrap();
    let region_parts =
        PreparedRegion::parts(region_columns.clone(), vec![(0, 1), (1, 2), (2, 3)]).unwrap();
    let region_tally = PreparedRegionTally::new(region_columns[0].clone(), true, vec![]).unwrap();
    let mut scorer = Scorer::new();
    scorer.add("population", &tally).unwrap();
    scorer.add("polsby", &polsby).unwrap();
    scorer.add("reock", &reock).unwrap();
    scorer
        .add("population_polygon", &population_polygon)
        .unwrap();
    scorer.add("convex_hull_ratio", &convex_hull_ratio).unwrap();
    scorer.add("cut_edges", &cut_edges).unwrap();
    scorer.add("region_splits", &region_splits).unwrap();
    scorer.add("region_pieces", &region_pieces).unwrap();
    scorer.add("region_parts", &region_parts).unwrap();
    scorer.add("region_tally", &region_tally).unwrap();
    assert_eq!(
        scorer.metric_names().collect::<Vec<_>>(),
        [
            "population",
            "polsby",
            "reock",
            "population_polygon",
            "convex_hull_ratio",
            "cut_edges",
            "region_splits",
            "region_pieces",
            "region_parts",
            "region_tally"
        ]
    );

    let source = AssignmentSource::open(path).unwrap();
    let mut rows = Vec::new();
    let summary = scorer
        .score_stream(
            &source,
            StreamOptions {
                max_samples,
                batch_size,
                track_uniqueness: true,
            },
            |row| {
                rows.push(row);
                Ok(())
            },
        )
        .unwrap();
    (summary, rows)
}

#[test]
fn oversized_batch_size_does_not_preallocate_the_requested_capacity() {
    let input = TempPath::new("ben");
    write_ben(input.path(), BenVariant::Standard);

    let (summary, rows) = score_with_batch(input.path(), None, usize::MAX);

    assert_eq!(summary.samples, plans().len() as u64);
    assert_eq!(rows.len(), plans().len());
}

#[test]
fn mkvchain_incremental_chunks_match_full_scoring_across_batch_boundaries() {
    let standard = TempPath::new("ben");
    let mkvchain = TempPath::new("ben");
    let distinct = [
        vec![1, 1, 2, 2],
        vec![1, 2, 1, 2],
        vec![2, 1, 2, 1],
        vec![2, 2, 1, 1],
    ];
    let plans = (0..40)
        .map(|index| distinct[index % distinct.len()].clone())
        .collect::<Vec<_>>();
    write_ben_plans(standard.path(), BenVariant::Standard, plans.clone());
    write_ben_plans(mkvchain.path(), BenVariant::MkvChain, plans);

    let (standard_summary, standard_rows) = score_with_batch(standard.path(), None, 32);
    let (mkvchain_summary, mkvchain_rows) = score_with_batch(mkvchain.path(), None, 32);

    assert_eq!(mkvchain_summary, standard_summary);
    assert_eq!(mkvchain_rows, standard_rows);
}

#[test]
fn mkvchain_scores_are_independent_of_batch_size_with_nonexact_tallies() {
    let input = TempPath::new("ben");
    let distinct = [
        vec![1, 1, 2, 2],
        vec![1, 2, 1, 2],
        vec![2, 1, 2, 1],
        vec![2, 2, 1, 1],
    ];
    let plans = (0..80)
        .map(|index| distinct[index % distinct.len()].clone())
        .collect::<Vec<_>>();
    write_ben_plans(input.path(), BenVariant::MkvChain, plans);

    let score_with_batch = |batch_size| {
        let tally = PreparedTally::new(vec![vec![0.1, 0.2, 0.3, 0.4]]).unwrap();
        let mut scorer = Scorer::new();
        scorer.add("tally", &tally).unwrap();
        let source = AssignmentSource::open(input.path()).unwrap();
        let mut rows = Vec::new();
        scorer
            .score_stream(
                &source,
                StreamOptions {
                    max_samples: None,
                    batch_size,
                    track_uniqueness: false,
                },
                |row| {
                    rows.push(row);
                    Ok(())
                },
            )
            .unwrap();
        rows
    };

    assert_eq!(score_with_batch(32), score_with_batch(40));
}

#[test]
fn mkvchain_scores_are_independent_of_rayon_thread_count() {
    let input = TempPath::new("ben");
    let distinct = [
        vec![1, 1, 2, 2],
        vec![1, 2, 1, 2],
        vec![2, 1, 2, 1],
        vec![2, 2, 1, 1],
    ];
    let plans = (0..40)
        .map(|index| distinct[index % distinct.len()].clone())
        .collect::<Vec<_>>();
    write_ben_plans(input.path(), BenVariant::MkvChain, plans);

    let score_with_threads = |threads| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap()
            .install(|| score_with_batch(input.path(), None, 32))
    };

    assert_eq!(score_with_threads(1), score_with_threads(4));
}

#[test]
fn shared_tally_bank_drives_tally_projections_and_eguia_once_per_transition() {
    let tally = PreparedTally::new(vec![
        vec![60.0, 10.0, 10.0, 20.0],
        vec![40.0, 20.0, 30.0, 10.0],
        vec![30.0, 20.0, 25.0, 25.0],
    ])
    .unwrap();
    let mut scorer = Scorer::new();
    scorer.set_tally_bank(&tally).unwrap();
    scorer.add_tally("reported", vec![2, 0]).unwrap();
    scorer.add_eguia("eguia", 0, 1, 0.5).unwrap();

    let first = vec![0, 0, 1, 1];
    let expected_first = vec![
        MetricScore::District(DistrictTable::new(
            vec![0, 1],
            vec![50.0, 50.0, 70.0, 30.0],
            2,
        )),
        MetricScore::Plan(PlanTable::new(vec![0, 1], vec![0.0]).unwrap()),
    ];
    assert_eq!(scorer.score_assignment(&first).unwrap(), expected_first);

    let second = vec![0, 1, 0, 1];
    let changes = [
        DeltaChange {
            node: 1,
            old: 0,
            new: 1,
        },
        DeltaChange {
            node: 2,
            old: 1,
            new: 0,
        },
    ];
    let mut state = scorer.incremental_state(&first).unwrap();
    state.update(&changes).unwrap();
    assert_eq!(
        state.result().unwrap(),
        scorer.score_assignment(&second).unwrap()
    );
    assert_eq!(
        state.result().unwrap()[1],
        MetricScore::Plan(PlanTable::new(vec![0, 1], vec![-0.5]).unwrap())
    );
}

#[test]
fn shared_derived_families_match_full_scoring_after_a_delta() {
    let tally = PreparedTally::new(vec![
        vec![60.0, 10.0, 10.0, 20.0],
        vec![40.0, 20.0, 30.0, 10.0],
        vec![45.0, 25.0, 30.0, 10.0],
        vec![35.0, 15.0, 10.0, 30.0],
        vec![30.0, 20.0, 25.0, 25.0],
        vec![20.0, 15.0, 5.0, 20.0],
    ])
    .unwrap();
    let mut scorer = Scorer::new();
    scorer.set_tally_bank(&tally).unwrap();
    scorer
        .add_paired_derived("bias", "partisan_bias", 0, 1, "observed")
        .unwrap();
    scorer
        .add_population_derived("deviation", "population_deviations", 4, false)
        .unwrap();
    scorer
        .add_demographic_derived("shares", "demographic_shares", 5, 4, 0.5)
        .unwrap();
    scorer
        .add_cross_election_derived(
            "wins",
            "party_wins_by_district",
            vec![0, 2],
            vec![1, 3],
            0.03,
        )
        .unwrap();

    let first = [0, 0, 1, 1];
    let second = [0, 1, 0, 1];
    let mut state = scorer.incremental_state(&first).unwrap();
    state
        .update(&[
            DeltaChange {
                node: 1,
                old: 0,
                new: 1,
            },
            DeltaChange {
                node: 2,
                old: 1,
                new: 0,
            },
        ])
        .unwrap();

    assert_eq!(
        state.result().unwrap(),
        scorer.score_assignment(&second).unwrap()
    );
}

#[test]
fn shared_tally_metric_registration_validates_dependencies() {
    let tally = PreparedTally::new(vec![vec![1.0, 2.0]]).unwrap();
    let mut scorer = Scorer::new();
    assert!(matches!(
        scorer.add_tally("missing", vec![0]),
        Err(Error::InvalidInput(message)) if message.contains("require a tally bank")
    ));
    scorer.set_tally_bank(&tally).unwrap();
    assert!(matches!(
        scorer.add_tally("invalid", vec![1]),
        Err(Error::InvalidInput(message)) if message.contains("out-of-range")
    ));
    assert!(matches!(
        scorer.add_eguia("invalid", 0, 0, f64::NAN),
        Err(Error::InvalidInput(message)) if message.contains("benchmark")
    ));
    assert!(matches!(
        scorer.add_paired_derived("invalid", "unknown", 0, 0, "equal"),
        Err(Error::InvalidInput(message)) if message.contains("unknown paired")
    ));
    assert!(matches!(
        scorer.add_demographic_derived("invalid", "districts_above_threshold", 0, 0, 2.0),
        Err(Error::InvalidInput(message)) if message.contains("threshold")
    ));
    assert!(matches!(
        scorer.add_cross_election_derived(
            "invalid",
            "aggregate_seats",
            vec![0],
            vec![0, 0],
            0.03,
        ),
        Err(Error::InvalidInput(message)) if message.contains("equal nonempty")
    ));
}

#[test]
fn validate_once_and_checked_updates_agree_without_mutating_on_malformed_deltas() {
    let tally = PreparedTally::new(vec![vec![1.0, 2.0, 3.0, 4.0]]).unwrap();
    let polsby = PreparedPolsbyPopper::new(
        vec![1.0; 4],
        vec![4.0; 4],
        vec![(0, 1), (1, 2), (2, 3)],
        vec![1.0; 3],
    )
    .unwrap();
    let unit_hulls = Arc::new(
        PreparedUnitHulls::new(vec![square(0.0), square(2.0), square(4.0), square(6.0)]).unwrap(),
    );
    let reock = PreparedReock::from_unit_hulls(Arc::clone(&unit_hulls));
    let geometry_rows = [
        square_wkb(0.0),
        square_wkb(2.0),
        square_wkb(4.0),
        square_wkb(6.0),
    ];
    let population_polygon = PreparedPopulationPolygon::from_unit_hulls_and_wkb(
        Arc::clone(&unit_hulls),
        &geometry_rows,
        &geometry_rows,
        vec![1.0, 2.0, 3.0, 4.0],
        vec![0, 1, 2, 3],
    )
    .unwrap();
    let convex_hull_ratio = PreparedConvexHullRatio::from_unit_hulls(Arc::clone(&unit_hulls));
    let state_clipped_convex_hull_ratio = PreparedStateClippedConvexHullRatio::from_validated_parts(
        Arc::clone(&unit_hulls),
        MultiPolygon(vec![polygon![
            (x: -1.0, y: -1.0),
            (x: 8.0, y: -1.0),
            (x: 8.0, y: 2.0),
            (x: -1.0, y: 2.0),
            (x: -1.0, y: -1.0),
        ]]),
    );
    let schwartzberg = PreparedSchwartzberg::new(
        vec![1.0; 4],
        vec![4.0; 4],
        vec![(0, 1), (1, 2), (2, 3)],
        vec![1.0; 3],
    )
    .unwrap();
    let area_perimeter = PreparedAreaPerimeterMetrics::new(
        vec![1.0; 4],
        vec![4.0; 4],
        vec![(0, 1), (1, 2), (2, 3)],
        vec![1.0; 3],
    )
    .unwrap();
    let cut_edges = PreparedCutEdges::new(4, vec![(0, 1), (1, 2), (2, 3)]).unwrap();
    let regions = vec![Some(10), Some(10), Some(20), Some(20)];
    let region_splits = PreparedRegion::splits(vec![regions.clone()]).unwrap();
    let region_pieces = PreparedRegion::pieces(vec![regions.clone()]).unwrap();
    let region_parts =
        PreparedRegion::parts(vec![regions.clone()], vec![(0, 1), (1, 2), (2, 3)]).unwrap();
    let region_tally = PreparedRegionTally::new(regions, true, vec![]).unwrap();
    let mut scorer = Scorer::new();
    scorer.add("tally", &tally).unwrap();
    scorer.add("polsby", &polsby).unwrap();
    scorer.add("reock", &reock).unwrap();
    scorer
        .add("population_polygon", &population_polygon)
        .unwrap();
    scorer.add("convex_hull_ratio", &convex_hull_ratio).unwrap();
    scorer
        .add(
            "state_clipped_convex_hull_ratio",
            &state_clipped_convex_hull_ratio,
        )
        .unwrap();
    scorer.add("schwartzberg", &schwartzberg).unwrap();
    scorer.add("area_perimeter", &area_perimeter).unwrap();
    scorer.add("cut", &cut_edges).unwrap();
    scorer.add("region_splits", &region_splits).unwrap();
    scorer.add("region_pieces", &region_pieces).unwrap();
    scorer.add("region_parts", &region_parts).unwrap();
    scorer.add("region_tally", &region_tally).unwrap();

    let plan_a = vec![1_u16, 1, 2, 2];
    let plan_b = vec![2_u16, 2, 1, 1];
    let plan_c = vec![1_u16, 2, 2, 1];
    let mut trusted = scorer.incremental_state(&plan_a).unwrap();
    let mut checked = scorer.incremental_state(&plan_a).unwrap();
    let initial_results = trusted.result().unwrap();
    assert_eq!(checked.result().unwrap(), initial_results);

    let cross_wired = [DeltaChange {
        node: 0,
        old: plan_b[0],
        new: 1,
    }];
    assert_eq!(
        trusted.update(&cross_wired),
        Err(Error::DeltaOldLabelMismatch {
            node: 0,
            expected: plan_a[0],
            actual: plan_b[0],
        })
    );
    assert_eq!(trusted.result().unwrap(), initial_results);

    let invalid = [
        DeltaChange {
            node: 0,
            old: plan_a[0],
            new: 2,
        },
        DeltaChange {
            node: 2,
            old: 0,
            new: 1,
        },
    ];
    assert_eq!(
        trusted.update(&invalid),
        Err(Error::DeltaOldLabelMismatch {
            node: 2,
            expected: plan_a[2],
            actual: 0,
        })
    );
    assert_eq!(trusted.result().unwrap(), initial_results);

    let mut current = plan_a;
    for next in [plan_b, plan_c] {
        let changes = current
            .iter()
            .zip(&next)
            .enumerate()
            .filter_map(|(node, (&old, &new))| {
                (old != new).then_some(DeltaChange { node, old, new })
            })
            .collect::<Vec<_>>();
        trusted.update(&changes).unwrap();
        checked.update_checked(&changes).unwrap();

        let fresh = scorer.score_assignment(&next).unwrap();
        assert_eq!(trusted.result().unwrap(), fresh);
        assert_eq!(checked.result().unwrap(), fresh);
        assert_eq!(trusted.assignment, next);
        assert_eq!(checked.assignment, next);
        current = next;
    }
}

fn expanded(rows: &[PlanScore]) -> Vec<String> {
    rows.iter()
        .flat_map(|row| std::iter::repeat_n(format!("{:?}", row.metrics), row.repetitions as usize))
        .collect()
}

#[test]
fn every_wire_and_variant_scores_the_same_samples() {
    let standard = TempPath::new("ben");
    write_ben(standard.path(), BenVariant::Standard);
    let (standard_summary, standard_rows) = score(standard.path(), None);
    assert_eq!(
        standard_summary,
        StreamSummary {
            samples: 6,
            accepted: 6,
            unique_plans: Some(2),
            unique_districts: Some(4),
        }
    );
    let expected = expanded(&standard_rows);

    for variant in [
        BenVariant::Standard,
        BenVariant::MkvChain,
        BenVariant::TwoDelta,
    ] {
        let ben = TempPath::new("ben");
        let xben = TempPath::new("xben");
        let bendl = TempPath::new("bendl");
        let xben_bendl = TempPath::new("bendl");
        write_ben(ben.path(), variant);
        write_xben(xben.path(), variant);
        write_bendl(bendl.path(), false, variant);
        write_bendl(xben_bendl.path(), true, variant);

        for path in [ben.path(), xben.path(), bendl.path(), xben_bendl.path()] {
            let (summary, rows) = score(path, None);
            assert_eq!(
                summary.samples,
                standard_summary.samples,
                "path={}",
                path.display()
            );
            assert_eq!(summary.unique_plans, Some(2), "path={}", path.display());
            assert_eq!(summary.unique_districts, Some(4), "path={}", path.display());
            assert_eq!(expanded(&rows), expected, "path={}", path.display());
        }
    }
}

#[test]
fn max_samples_caps_the_last_repetition_and_offsets_are_zero_based() {
    let mkvchain = TempPath::new("ben");
    write_ben(mkvchain.path(), BenVariant::MkvChain);

    let (summary, rows) = score(mkvchain.path(), Some(4));

    assert_eq!(
        summary,
        StreamSummary {
            samples: 4,
            accepted: 3,
            unique_plans: Some(2),
            unique_districts: Some(4),
        }
    );
    assert_eq!(
        rows.iter().map(|row| row.sample_offset).collect::<Vec<_>>(),
        [0, 2, 3]
    );
    assert_eq!(
        rows.iter().map(|row| row.repetitions).collect::<Vec<_>>(),
        [2, 1, 1]
    );
}

#[test]
fn max_samples_caps_standard_and_twodelta_variants_too() {
    // Standard frames always carry one repetition, so the cap lands on a frame boundary.
    let standard = TempPath::new("ben");
    write_ben(standard.path(), BenVariant::Standard);
    let (summary, rows) = score(standard.path(), Some(4));
    assert_eq!(
        summary,
        StreamSummary {
            samples: 4,
            accepted: 4,
            unique_plans: Some(2),
            unique_districts: Some(4),
        }
    );
    assert_eq!(
        rows.iter().map(|row| row.sample_offset).collect::<Vec<_>>(),
        [0, 1, 2, 3]
    );
    assert_eq!(
        rows.iter().map(|row| row.repetitions).collect::<Vec<_>>(),
        [1, 1, 1, 1]
    );

    // The TwoDelta writer merges the repeated plans into counted frames ([2, 1, 3]), so the
    // cap truncates the final frame's repetition count.
    let twodelta = TempPath::new("ben");
    write_ben(twodelta.path(), BenVariant::TwoDelta);
    let (summary, rows) = score(twodelta.path(), Some(4));
    assert_eq!(
        summary,
        StreamSummary {
            samples: 4,
            accepted: 3,
            unique_plans: Some(2),
            unique_districts: Some(4),
        }
    );
    assert_eq!(
        rows.iter().map(|row| row.sample_offset).collect::<Vec<_>>(),
        [0, 2, 3]
    );
    assert_eq!(
        rows.iter().map(|row| row.repetitions).collect::<Vec<_>>(),
        [2, 1, 1]
    );
}

fn single_tally_scorer_error(path: &Path) -> Error {
    let tally = PreparedTally::new(vec![vec![1.0; 4]]).unwrap();
    let mut scorer = Scorer::new();
    scorer.add("population", &tally).unwrap();
    let source = AssignmentSource::open(path).unwrap();
    scorer
        .score_stream(&source, StreamOptions::default(), |_| Ok(()))
        .unwrap_err()
}

#[test]
fn zero_repetition_frames_are_rejected_at_the_frame() {
    // The scoring-side guard names the offending frame instead of poisoning the writer late.
    assert_eq!(
        stream::ensure_positive_repetitions(0, 7),
        Err(Error::ZeroRepetitionFrame { frame: 7 })
    );
    assert_eq!(stream::ensure_positive_repetitions(1, 7), Ok(()));

    // A synthetic MkvChain frame declaring zero repetitions fails cleanly at decode time.
    let path = TempPath::new("ben");
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"MKVCHAIN BEN FILE");
    bytes.push(1); // max_val_bit_count
    bytes.push(1); // max_len_bit_count
    bytes.extend_from_slice(&1_u32.to_be_bytes()); // n_bytes
    bytes.push(0xFF); // payload
    bytes.extend_from_slice(&0_u16.to_be_bytes()); // count = 0
    std::fs::write(path.path(), &bytes).unwrap();

    let error = single_tally_scorer_error(path.path());
    assert!(matches!(
        &error,
        Error::Io {
            kind: std::io::ErrorKind::InvalidData,
            ..
        }
    ));
    assert!(error.to_string().contains("count"), "got {error}");
}

#[test]
fn truncated_raw_ben_streams_fail_cleanly_instead_of_panicking() {
    let path = TempPath::new("ben");
    write_ben(path.path(), BenVariant::Standard);
    let bytes = std::fs::read(path.path()).unwrap();
    // Every Standard frame here is at least 7 bytes, so cutting 2 bytes lands mid-frame.
    std::fs::write(path.path(), &bytes[..bytes.len() - 2]).unwrap();

    let error = single_tally_scorer_error(path.path());
    assert!(
        matches!(
            &error,
            Error::Io {
                kind: std::io::ErrorKind::UnexpectedEof,
                ..
            }
        ),
        "got {error}"
    );
}

#[test]
fn twodelta_delta_before_any_snapshot_fails_cleanly() {
    // A raw TwoDelta stream whose first frame is a delta (tag 0x01) has no anchor assignment.
    // The ben reader rejects it before scoring's own Error::DeltaBeforeSnapshot guard runs.
    let path = TempPath::new("ben");
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"TWODELTA BEN FILE");
    bytes.push(0x01); // delta frame tag
    bytes.extend_from_slice(&1_u16.to_be_bytes()); // pair a
    bytes.extend_from_slice(&2_u16.to_be_bytes()); // pair b
    bytes.push(8); // run-length bit width
    bytes.extend_from_slice(&1_u32.to_be_bytes()); // n_bytes
    bytes.push(4); // one run of length 4
    bytes.extend_from_slice(&1_u16.to_be_bytes()); // count
    std::fs::write(path.path(), &bytes).unwrap();

    let error = single_tally_scorer_error(path.path());
    assert!(matches!(&error, Error::Io { .. }));
    assert!(
        error
            .to_string()
            .contains("before an initial full-assignment frame"),
        "got {error}"
    );
}

#[test]
fn scorer_rejects_duplicate_names_and_mismatched_node_counts() {
    let two_nodes = PreparedTally::new(vec![vec![1.0, 2.0]]).unwrap();
    let three_nodes = PreparedTally::new(vec![vec![1.0, 2.0, 3.0]]).unwrap();
    let mut scorer = Scorer::new();
    assert_eq!(scorer.add("", &two_nodes), Err(Error::EmptyMetricName));
    scorer.add("population", &two_nodes).unwrap();

    assert_eq!(
        scorer.add("population", &two_nodes),
        Err(Error::DuplicateMetricName("population".into()))
    );
    assert_eq!(
        scorer.add("other", &three_nodes),
        Err(Error::MetricNodeCount {
            metric: "other".into(),
            actual: 3,
            expected: 2,
        })
    );
}

#[test]
fn scorer_writes_a_metadata_checked_atomic_run() {
    let input = TempPath::new("ben");
    let output = TempPath::new("run");
    write_ben(input.path(), BenVariant::Standard);
    let tally = PreparedTally::new(vec![vec![1.0, 2.0, 3.0, 4.0]]).unwrap();
    let cut_edges = PreparedCutEdges::new(4, vec![(0, 1), (1, 2), (2, 3)]).unwrap();
    let regions = PreparedRegion::splits(vec![
        vec![Some(10), Some(10), Some(20), Some(20)],
        vec![Some(7), Some(8), Some(8), None],
    ])
    .unwrap();
    let pieces = PreparedRegion::pieces(vec![
        vec![Some(10), Some(10), Some(20), Some(20)],
        vec![Some(7), Some(8), Some(8), None],
    ])
    .unwrap();
    let region_tally =
        PreparedRegionTally::new(vec![Some(10), Some(10), Some(20), Some(20)], true, vec![])
            .unwrap();
    let mut scorer = Scorer::new();
    scorer.add("population", &tally).unwrap();
    scorer.add("cut_edges", &cut_edges).unwrap();
    scorer.add("region_splits", &regions).unwrap();
    scorer.add("region_pieces", &pieces).unwrap();
    scorer.add("county_totals", &region_tally).unwrap();
    let source = AssignmentSource::open(input.path()).unwrap();
    let metadata = RunMetadata::new(
        Some(input.path().to_string_lossy().into_owned()),
        vec![
            MetricMetadata::new(
                "tally",
                "population",
                TableShape::District,
                vec!["total".into()],
            ),
            MetricMetadata::new(
                "cut_edges",
                "cut_edges",
                TableShape::Plan,
                vec!["count".into()],
            ),
            MetricMetadata::new(
                "region_splits",
                "region_splits",
                TableShape::Plan,
                vec!["county".into(), "municipality".into()],
            ),
            MetricMetadata::new(
                "region_pieces",
                "region_pieces",
                TableShape::Plan,
                vec!["county".into(), "municipality".into()],
            ),
            county_tally_metadata(),
        ],
    );

    let summary = scorer
        .score_run(&source, StreamOptions::default(), output.path(), metadata)
        .unwrap();

    assert_eq!(summary.samples, 6);
    assert_eq!(summary.accepted, 6);
    assert!(output_manifest(output.path()).is_file());
    assert!(grouped_output(output.path(), "population", "total_tallies").is_file());
    assert!(flat_output(output.path(), "cut_edges").is_file());
    assert!(flat_output(output.path(), "region_splits").is_file());
    assert!(flat_output(output.path(), "region_pieces").is_file());
    assert!(grouped_output(output.path(), "county_totals", "count_tallies_by_region").is_file());

    let bad_output = TempPath::new("bad-run");
    let bad_metadata = RunMetadata::new(
        None,
        vec![
            MetricMetadata::new(
                "reock",
                "population",
                TableShape::District,
                vec!["total".into()],
            ),
            MetricMetadata::new(
                "cut_edges",
                "cut_edges",
                TableShape::Plan,
                vec!["count".into()],
            ),
            MetricMetadata::new(
                "region_splits",
                "region_splits",
                TableShape::Plan,
                vec!["county".into(), "municipality".into()],
            ),
            MetricMetadata::new(
                "region_pieces",
                "region_pieces",
                TableShape::Plan,
                vec!["county".into(), "municipality".into()],
            ),
            county_tally_metadata(),
        ],
    );
    assert!(scorer
        .score_run(
            &source,
            StreamOptions::default(),
            bad_output.path(),
            bad_metadata,
        )
        .unwrap_err()
        .to_string()
        .contains("expected \"tally\""));
    assert!(!bad_output.path().exists());

    let wrong_shape_output = TempPath::new("wrong-shape-run");
    let wrong_shape = RunMetadata::new(
        None,
        vec![
            MetricMetadata::new(
                "tally",
                "population",
                TableShape::District,
                vec!["total".into()],
            ),
            MetricMetadata::new(
                "cut_edges",
                "cut_edges",
                TableShape::District,
                vec!["count".into()],
            ),
            MetricMetadata::new(
                "region_splits",
                "region_splits",
                TableShape::Plan,
                vec!["county".into(), "municipality".into()],
            ),
            MetricMetadata::new(
                "region_pieces",
                "region_pieces",
                TableShape::Plan,
                vec!["county".into(), "municipality".into()],
            ),
            county_tally_metadata(),
        ],
    );
    assert!(scorer
        .score_run(
            &source,
            StreamOptions::default(),
            wrong_shape_output.path(),
            wrong_shape,
        )
        .unwrap_err()
        .to_string()
        .contains("expected Plan"));
    assert!(!wrong_shape_output.path().exists());
}

#[test]
fn run_metadata_subkeys_must_match_the_registered_order() {
    let input = TempPath::new("ben");
    write_ben(input.path(), BenVariant::Standard);
    let source = AssignmentSource::open(input.path()).unwrap();
    let tally =
        PreparedTally::new(vec![vec![1.0, 2.0, 3.0, 4.0], vec![10.0, 20.0, 30.0, 40.0]]).unwrap();

    assert!(Scorer::new()
        .add_with_subkeys("population", &tally, vec!["pop".into()])
        .unwrap_err()
        .to_string()
        .contains("registers 1 subkeys"));

    let mut scorer = Scorer::new();
    scorer
        .add_with_subkeys("population", &tally, vec!["pop".into(), "vap".into()])
        .unwrap();
    let metadata_for = |subkeys: Vec<String>| {
        RunMetadata::new(
            None,
            vec![MetricMetadata::new(
                "tally",
                "population",
                TableShape::District,
                subkeys,
            )],
        )
    };

    let reordered_output = TempPath::new("reordered-run");
    let error = scorer
        .score_run(
            &source,
            StreamOptions::default(),
            reordered_output.path(),
            metadata_for(vec!["vap".into(), "pop".into()]),
        )
        .unwrap_err();
    assert!(error.to_string().contains("the scorer registered"));
    assert!(!reordered_output.path().exists());

    let output = TempPath::new("ordered-run");
    scorer
        .score_run(
            &source,
            StreamOptions::default(),
            output.path(),
            metadata_for(vec!["pop".into(), "vap".into()]),
        )
        .unwrap();
    assert!(grouped_output(output.path(), "population", "pop_tallies").is_file());
}

fn batch_scorer_metrics() -> (PreparedTally, PreparedCutEdges, PreparedRegion) {
    (
        PreparedTally::new(vec![vec![1.0, 2.0, 3.0]]).unwrap(),
        PreparedCutEdges::new(3, vec![(0, 1), (1, 2)]).unwrap(),
        PreparedRegion::splits(vec![vec![Some(0), Some(0), Some(1)]]).unwrap(),
    )
}

#[test]
fn score_batch_scores_mixed_district_and_plan_metrics_in_registration_order() {
    let (tally, cut_edges, region) = batch_scorer_metrics();
    let mut scorer = Scorer::new();
    scorer.add("tally", &tally).unwrap();
    scorer.add("cut_edges", &cut_edges).unwrap();
    scorer.add("region_splits", &region).unwrap();

    let (districts, rows) = scorer.score_batch(&[vec![0, 0, 1], vec![0, 1, 1]]).unwrap();

    assert_eq!(districts, vec![0, 1]);
    assert_eq!(
        rows,
        vec![
            vec![
                MetricScore::District(tally.score(&[0, 0, 1]).unwrap()),
                MetricScore::Plan(cut_edges.score(&[0, 0, 1]).unwrap()),
                MetricScore::Plan(region.score(&[0, 0, 1]).unwrap()),
            ],
            vec![
                MetricScore::District(tally.score(&[0, 1, 1]).unwrap()),
                MetricScore::Plan(cut_edges.score(&[0, 1, 1]).unwrap()),
                MetricScore::Plan(region.score(&[0, 1, 1]).unwrap()),
            ],
        ]
    );
    assert_eq!(
        rows[0][1],
        MetricScore::Plan(PlanTable::new(vec![0, 1], vec![1.0]).unwrap())
    );
    assert_eq!(
        rows[1][2],
        MetricScore::Plan(PlanTable::new(vec![0, 1], vec![1.0]).unwrap())
    );
}

#[test]
fn score_batch_rejects_assignment_batches_with_different_district_sets() {
    let (tally, cut_edges, region) = batch_scorer_metrics();
    let mut scorer = Scorer::new();
    scorer.add("tally", &tally).unwrap();
    scorer.add("cut_edges", &cut_edges).unwrap();
    scorer.add("region_splits", &region).unwrap();

    let error = scorer
        .score_batch(&[vec![0, 0, 1], vec![0, 1, 2]])
        .unwrap_err();

    assert!(error
        .to_string()
        .contains("district labels must be the same"));
    assert_eq!(
        Scorer::new().score_batch(&[vec![0, 0, 1]]).unwrap_err(),
        Error::EmptyScorer
    );
}
