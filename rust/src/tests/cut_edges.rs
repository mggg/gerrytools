use super::*;

fn line_metric() -> PreparedCutEdges {
    PreparedCutEdges::new(4, vec![(0, 1), (1, 2), (2, 3)]).unwrap()
}

#[test]
fn scores_unweighted_edges_and_observes_isolated_nodes() {
    let metric = PreparedCutEdges::new(3, vec![(0, 1)]).unwrap();

    let result = metric.score(&[1, 2, 5]).unwrap();

    assert_eq!(result.district_ids(), &[1, 2, 5]);
    assert_eq!(result.values(), &[1.0]);
}

#[test]
fn weighted_scores_preserve_zero_and_negative_weights() {
    let metric =
        PreparedCutEdges::weighted(4, vec![(0, 1), (1, 2), (2, 3)], vec![0.0, -2.5, 4.0]).unwrap();

    assert_eq!(metric.score(&[1, 2, 1, 2]).unwrap().values(), &[1.5]);
    assert_eq!(metric.score(&[1, 1, 2, 2]).unwrap().values(), &[-2.5]);
}

#[test]
fn accepts_an_empty_edge_list_without_losing_district_metadata() {
    let metric = PreparedCutEdges::new(3, vec![]).unwrap();
    let result = metric.score(&[1, 2, 3]).unwrap();

    assert_eq!(result.district_ids(), &[1, 2, 3]);
    assert_eq!(result.values(), &[0.0]);
}

#[test]
fn validates_prepared_inputs() {
    assert_eq!(
        PreparedCutEdges::weighted(2, vec![(0, 1)], vec![]).unwrap_err(),
        Error::EdgeWeightCount {
            actual: 0,
            expected: 1,
        }
    );
    assert_eq!(
        PreparedCutEdges::weighted(2, vec![(0, 1)], vec![f64::NAN]).unwrap_err(),
        Error::NonFiniteEdgeWeight { edge: 0 }
    );
    assert_eq!(
        PreparedCutEdges::new(2, vec![(0, 2)]).unwrap_err(),
        Error::EdgeNodeOutOfRange {
            u: 0,
            v: 2,
            node_count: 2,
        }
    );
    assert!(PreparedCutEdges::new(2, vec![(0, 0)])
        .unwrap_err()
        .to_string()
        .contains("self-loop"));
    assert!(PreparedCutEdges::new(2, vec![(0, 1), (1, 0)])
        .unwrap_err()
        .to_string()
        .contains("duplicate edge"));
}

#[test]
fn incremental_update_handles_both_edge_endpoints_moving() {
    let metric = line_metric();
    let mut assignment = vec![1, 1, 2, 2];
    let changes = [
        DeltaChange {
            node: 1,
            old: 1,
            new: 2,
        },
        DeltaChange {
            node: 2,
            old: 2,
            new: 1,
        },
    ];
    let mut state = metric.incremental(&assignment).unwrap();

    state.update(&changes).unwrap();
    for change in changes {
        assignment[change.node] = change.new;
    }

    assert_eq!(state.result(), metric.score(&assignment).unwrap());
    assert_eq!(state.result().values(), &[3.0]);
}

#[test]
fn incremental_weighted_update_matches_full_recomputation() {
    let metric =
        PreparedCutEdges::weighted(4, vec![(0, 1), (1, 2), (2, 3)], vec![0.0, -2.5, 4.0]).unwrap();
    let mut assignment = vec![1, 1, 2, 2];
    let changes = [
        DeltaChange {
            node: 1,
            old: 1,
            new: 2,
        },
        DeltaChange {
            node: 2,
            old: 2,
            new: 1,
        },
    ];
    let mut state = metric.incremental(&assignment).unwrap();

    state.update(&changes).unwrap();
    for change in changes {
        assignment[change.node] = change.new;
    }

    assert_eq!(state.result(), metric.score(&assignment).unwrap());
}

#[test]
fn incremental_weighted_update_takes_the_delta_path_on_a_large_graph() {
    // A degree-2 node on a 19-edge path keeps incident_visits * 4 < |E|, so the weighted
    // per-edge delta branch runs instead of a full rescan.
    let edges: Vec<(u32, u32)> = (0..19).map(|node| (node, node + 1)).collect();
    let weights: Vec<f64> = (0..19).map(|edge| 0.5 + edge as f64).collect();
    let metric = PreparedCutEdges::weighted(20, edges, weights).unwrap();
    let mut assignment = vec![1; 20];
    assignment[10..].fill(2);
    let changes = [DeltaChange {
        node: 10,
        old: 2,
        new: 1,
    }];
    let mut state = metric.incremental(&assignment).unwrap();

    assert!(!state.use_full_rescan(&changes));
    state.update(&changes).unwrap();
    assignment[10] = 1;

    assert_eq!(state.result(), metric.score(&assignment).unwrap());
    // Cut edge moves from (9, 10) to (10, 11): weight 9.5 out, 10.5 in.
    assert_eq!(state.result().values(), &[10.5]);
}

#[test]
fn incremental_update_observes_an_isolated_node_move() {
    let metric = PreparedCutEdges::new(3, vec![(0, 1)]).unwrap();
    let mut assignment = vec![1, 2, 5];
    let changes = [DeltaChange {
        node: 2,
        old: 5,
        new: 6,
    }];
    let mut state = metric.incremental(&assignment).unwrap();

    state.update(&changes).unwrap();
    assignment[2] = 6;

    assert_eq!(state.result(), metric.score(&assignment).unwrap());
    assert_eq!(state.result().district_ids(), &[1, 2, 6]);
}

#[test]
fn incremental_update_clears_an_emptied_district() {
    let metric = line_metric();
    let mut assignment = vec![1, 2, 2, 2];
    let changes = [DeltaChange {
        node: 0,
        old: 1,
        new: 2,
    }];
    let mut state = metric.incremental(&assignment).unwrap();

    state.update(&changes).unwrap();
    assignment[0] = 2;

    assert_eq!(state.result(), metric.score(&assignment).unwrap());
    assert_eq!(state.result().district_ids(), &[2]);
}

#[test]
fn adaptive_update_chooses_both_exact_paths_at_the_work_boundary() {
    let below = PreparedCutEdges::new(6, vec![(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]).unwrap();
    let before = vec![1, 1, 2, 2, 2, 2];
    let mut after = before.clone();
    after[0] = 2;
    let changes = [DeltaChange {
        node: 0,
        old: 1,
        new: 2,
    }];
    let mut state = below.incremental(&before).unwrap();

    assert!(!state.use_full_rescan(&changes));
    state.update(&changes).unwrap();
    assert_eq!(state.result(), below.score(&after).unwrap());

    let boundary = PreparedCutEdges::new(5, vec![(0, 1), (1, 2), (2, 3), (3, 4)]).unwrap();
    let before = vec![1, 1, 2, 2, 2];
    let mut after = before.clone();
    after[0] = 2;
    let mut state = boundary.incremental(&before).unwrap();

    assert!(state.use_full_rescan(&changes));
    state.update(&changes).unwrap();
    assert_eq!(state.result(), boundary.score(&after).unwrap());
}

#[test]
fn rejects_a_delta_produced_against_a_different_plan() {
    let metric = line_metric();
    let plan_a = [1, 1, 2, 2];
    let mut state = metric.incremental(&plan_a).unwrap();
    // Valid against [1, 2, 1, 2] but cross-wired for a state that retained plan A.
    let cross_wired = [DeltaChange {
        node: 1,
        old: 2,
        new: 1,
    }];

    assert_eq!(
        state.update(&cross_wired),
        Err(Error::DeltaOldLabelMismatch {
            node: 1,
            expected: 1,
            actual: 2,
        })
    );
    assert_eq!(state.result(), metric.score(&plan_a).unwrap());
}

#[test]
fn invalid_delta_does_not_partially_update_cut_edges() {
    let metric = line_metric();
    let assignment = [1, 1, 2, 2];
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
            new: 2,
        },
    ];

    assert!(state.update(&changes).is_err());
    assert_eq!(state.result(), before);
}
