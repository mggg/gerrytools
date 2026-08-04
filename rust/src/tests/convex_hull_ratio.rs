use super::*;
use crate::Coordinate;

fn point(x: f64, y: f64) -> Coordinate {
    Coordinate { x, y }
}

fn square(x: f64, y: f64) -> UnitHull {
    UnitHull::new(1.0, square_points(x, y))
}

fn square_points(x: f64, y: f64) -> Vec<Coordinate> {
    vec![
        point(x, y),
        point(x + 1.0, y),
        point(x + 1.0, y + 1.0),
        point(x, y + 1.0),
    ]
}

fn assert_close(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() < 1e-12,
        "actual={actual}, expected={expected}"
    );
}

fn assert_same_scores(actual: &DistrictTable, expected: &DistrictTable) {
    assert_eq!(actual.district_ids(), expected.district_ids());
    for (&actual, &expected) in actual
        .column(0)
        .unwrap()
        .iter()
        .zip(expected.column(0).unwrap())
    {
        assert_close(actual, expected);
    }
}

#[test]
fn scores_hand_computable_square_l_shape_and_gap() {
    let metric = PreparedConvexHullRatio::new(vec![
        square(0.0, 0.0),
        square(1.0, 0.0),
        square(0.0, 1.0),
        square(3.0, 0.0),
    ])
    .unwrap();

    let result = metric.score(&[0, 0, 0, 1]).unwrap();
    assert_eq!(result.district_ids(), &[0, 1]);
    assert_close(result.column(0).unwrap()[0], 6.0 / 7.0);
    assert_close(result.column(0).unwrap()[1], 1.0);

    let gap = metric.score(&[0, 1, 1, 0]).unwrap();
    assert_close(gap.column(0).unwrap()[0], 0.5);
}

#[test]
fn rejects_impossible_score_and_invalid_assignment() {
    let impossible =
        PreparedConvexHullRatio::new(vec![UnitHull::new(2.0, square_points(0.0, 0.0))]).unwrap();
    assert!(matches!(
        impossible.score(&[4]),
        Err(Error::ImpossibleScore {
            metric: "convex-hull ratio",
            district: 4,
            ..
        })
    ));

    let metric = PreparedConvexHullRatio::new(vec![square(0.0, 0.0)]).unwrap();
    assert_eq!(
        metric.score(&[]).unwrap_err(),
        Error::AssignmentLength {
            actual: 0,
            expected: 1,
        }
    );
    assert_eq!(metric.score(&[500]).unwrap().district_ids(), &[500]);
}

#[test]
fn incremental_updates_match_fresh_scores_across_generated_moves() {
    let units = (0..5)
        .flat_map(|row| (0..6).map(move |column| square(column as f64, row as f64)))
        .collect();
    let metric = PreparedConvexHullRatio::new(units).unwrap();
    let mut assignment = (0..metric.node_count())
        .map(|node| (node % 4) as u16)
        .collect::<Vec<_>>();
    let mut incremental = metric.incremental(&assignment).unwrap();
    let mut seed = 0x5eed_cafe_u64;

    for _ in 0..500 {
        seed = seed
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let node = (seed as usize) % assignment.len();
        let old = assignment[node];
        let mut new = ((seed >> 32) % 4) as u16;
        if new == old {
            new = (new + 1) % 4;
        }
        let change = DeltaChange { node, old, new };

        incremental.update(&[change]).unwrap();
        assignment[node] = new;

        assert_same_scores(&incremental.result(), &metric.score(&assignment).unwrap());
    }
}

#[test]
fn incremental_clears_an_emptied_district() {
    let metric = PreparedConvexHullRatio::new(vec![square(0.0, 0.0), square(2.0, 0.0)]).unwrap();
    let mut assignment = vec![3, 7];
    let change = DeltaChange {
        node: 0,
        old: 3,
        new: 7,
    };
    let mut incremental = metric.incremental(&assignment).unwrap();

    incremental.update(&[change]).unwrap();
    assignment[0] = 7;

    assert_eq!(incremental.result().district_ids(), &[7]);
    assert_same_scores(&incremental.result(), &metric.score(&assignment).unwrap());
}
