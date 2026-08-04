use super::*;

fn expected_values(
    metric: &PreparedRegionTally,
    regions: &[Option<u32>],
    include_count: bool,
    input_values: &[Vec<f64>],
    assignment: &[u16],
) -> DistrictTable {
    let (_, observed) = observed_districts(assignment);
    let districts = district_ids(&observed);
    let mut values = Vec::new();
    for value in 0..metric.metric_count() {
        for &region in metric.region_ids() {
            for &district in &districts {
                values.push(
                    regions
                        .iter()
                        .zip(assignment)
                        .enumerate()
                        .filter_map(|(node, (candidate, &assigned))| {
                            if *candidate != Some(region) || assigned != district {
                                return None;
                            }
                            if include_count && value == 0 {
                                Some(1.0)
                            } else {
                                Some(input_values[value - usize::from(include_count)][node])
                            }
                        })
                        .sum(),
                );
            }
        }
    }
    DistrictTable::new(districts, values, metric.column_count())
}

#[test]
fn scores_counts_and_multiple_values_in_metric_then_region_order() {
    let regions = vec![Some(20), Some(10), Some(20), None, Some(10)];
    let assignment = [1, 2, 2, 3, 1];
    let metric = PreparedRegionTally::new(
        regions,
        true,
        vec![
            vec![1.0, 2.0, 4.0, 8.0, 16.0],
            vec![10.0, 20.0, 40.0, 80.0, 160.0],
        ],
    )
    .unwrap();

    assert_eq!(metric.region_ids(), &[20, 10]);
    assert_eq!(metric.metric_count(), 3);
    assert_eq!(metric.region_count(), 2);
    assert_eq!(
        metric.score(&assignment).unwrap().district_ids(),
        &[1, 2, 3]
    );
    let result = metric.score(&assignment).unwrap();
    assert_eq!(result.column(0), Some([1.0, 1.0, 0.0].as_slice()));
    assert_eq!(result.column(1), Some([1.0, 1.0, 0.0].as_slice()));
    assert_eq!(result.column(2), Some([1.0, 4.0, 0.0].as_slice()));
    assert_eq!(result.column(3), Some([16.0, 2.0, 0.0].as_slice()));
    assert_eq!(result.column(4), Some([10.0, 40.0, 0.0].as_slice()));
    assert_eq!(result.column(5), Some([160.0, 20.0, 0.0].as_slice()));
}

#[test]
fn missing_regions_do_not_hide_observed_districts() {
    let metric = PreparedRegionTally::new(vec![None, None], true, vec![]).unwrap();
    let result = metric.score(&[4, 7]).unwrap();

    assert!(metric.region_ids().is_empty());
    assert_eq!(result.district_ids(), &[4, 7]);
    assert_eq!(result.column_count(), 0);
    assert!(result.column(0).is_none());
    assert_eq!(result, metric.incremental(&[4, 7]).unwrap().result());
}

#[test]
fn handles_sparse_district_ids_across_dynamic_storage() {
    let metric = PreparedRegionTally::new(vec![Some(1), Some(1), Some(2)], true, vec![]).unwrap();
    let assignment = [0, 128, 499];
    let result = metric.score(&assignment).unwrap();

    assert_eq!(result.district_ids(), &[0, 128, 499]);
    assert_eq!(result.column(0), Some([1.0, 1.0, 0.0].as_slice()));
    assert_eq!(result.column(1), Some([0.0, 0.0, 1.0].as_slice()));
    assert_eq!(result, metric.incremental(&assignment).unwrap().result());
}

#[test]
fn validates_values_assignments_and_deltas_before_mutation() {
    assert_eq!(
        PreparedRegionTally::new(vec![Some(1), Some(2)], false, vec![vec![1.0]]).unwrap_err(),
        Error::TallyByRegionValueLength {
            value: 0,
            actual: 1,
            expected: 2,
        }
    );
    assert_eq!(
        PreparedRegionTally::new(vec![Some(1)], false, vec![vec![f64::NAN]]).unwrap_err(),
        Error::NonFiniteTallyByRegionValue { value: 0, node: 0 }
    );
    assert!(PreparedRegionTally::new(vec![Some(1)], false, vec![]).is_err());

    let metric = PreparedRegionTally::new(vec![Some(1), Some(2), Some(2)], true, vec![]).unwrap();
    assert_eq!(
        metric.score(&[1, 2]).unwrap_err(),
        Error::AssignmentLength {
            actual: 2,
            expected: 3,
        }
    );
    let assignment = [1, 1, 2];
    let mut state = metric.incremental(&assignment).unwrap();
    let before = state.result();
    let changes = [
        DeltaChange {
            node: 0,
            old: 1,
            new: 2,
        },
        DeltaChange {
            node: 2,
            old: 1,
            new: 3,
        },
    ];

    assert!(state.update(&changes).is_err());
    assert_eq!(state.result(), before);
}

#[test]
fn generated_incremental_multi_value_tallies_match_full_and_reference_scores() {
    for seed in 0..64 {
        let mut rng = fastrand::Rng::with_seed(20_000 + seed);
        let node_count = 20;
        let regions = (0..node_count)
            .map(|_| (rng.usize(0..5) != 0).then(|| [10, 20, 30][rng.usize(0..3)]))
            .collect::<Vec<_>>();
        let input_values = (0..3)
            .map(|_| {
                (0..node_count)
                    .map(|_| rng.u32(0..40) as f64 / 4.0 - 2.0)
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let metric = PreparedRegionTally::new(regions.clone(), true, input_values.clone()).unwrap();
        let mut assignment = (0..node_count)
            .map(|_| rng.usize(0..4) as u16)
            .collect::<Vec<_>>();
        let mut state = metric.incremental(&assignment).unwrap();

        for step in 0..100 {
            let mut nodes = (0..node_count).collect::<Vec<_>>();
            rng.shuffle(&mut nodes);
            let mut changed_nodes = nodes[..1 + rng.usize(0..5)].to_vec();
            changed_nodes.sort_unstable();
            let changes = changed_nodes
                .into_iter()
                .map(|node| DeltaChange {
                    node,
                    old: assignment[node],
                    new: rng.usize(0..4) as u16,
                })
                .collect::<Vec<_>>();

            state.update(&changes).unwrap();
            for change in changes {
                assignment[change.node] = change.new;
            }

            let full = metric.score(&assignment).unwrap();
            let expected = expected_values(&metric, &regions, true, &input_values, &assignment);
            assert_eq!(state.result(), full, "seed {seed}, step {step}");
            assert_eq!(state.result(), expected, "seed {seed}, step {step}");
        }
    }
}
