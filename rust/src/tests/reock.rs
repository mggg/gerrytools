use super::*;

/// Absolute predicate tolerance for the brute-force oracle, which only sees small coordinates.
const BRUTE_EPS: f64 = 1e-9;

fn coord(x: f64, y: f64) -> Coordinate {
    Coordinate { x, y }
}

fn translated(points: &[Coordinate], dx: f64, dy: f64) -> Vec<Coordinate> {
    points
        .iter()
        .map(|point| coord(point.x + dx, point.y + dy))
        .collect()
}

fn square_points(x: f64, y: f64) -> Vec<Coordinate> {
    vec![
        coord(x, y),
        coord(x + 1.0, y),
        coord(x + 1.0, y + 1.0),
        coord(x, y + 1.0),
    ]
}

fn square_unit(x: f64, y: f64) -> UnitHull {
    UnitHull::new(1.0, square_points(x, y))
}

fn assert_close(actual: f64, expected: f64, epsilon: f64) {
    assert!(
        (actual - expected).abs() <= epsilon,
        "expected {expected}, got {actual}"
    );
}

fn circle_area(mut points: Vec<Coordinate>) -> Option<f64> {
    let mut rng = fastrand::Rng::with_seed(REOCK_SHUFFLE_SEED);
    minimum_enclosing_circle_area(&mut points, &mut rng)
}

fn brute_force_circle_area(points: &[Coordinate]) -> Option<f64> {
    let mut unique = Vec::new();
    for &point in points {
        if !unique
            .iter()
            .any(|&seen| l2_sq_dist(seen, point) <= BRUTE_EPS)
        {
            unique.push(point);
        }
    }
    if unique.len() < 2 {
        return None;
    }

    let mut best: Option<Circle> = None;
    for i in 0..unique.len() {
        for j in (i + 1)..unique.len() {
            let candidate = diameter_circle(unique[i], unique[j]);
            if unique
                .iter()
                .all(|&point| in_circle(point, &candidate, BRUTE_EPS))
            {
                best = Some(match best {
                    Some(current) if current.radius <= candidate.radius => current,
                    _ => candidate,
                });
            }
        }
    }
    for i in 0..unique.len() {
        for j in (i + 1)..unique.len() {
            for k in (j + 1)..unique.len() {
                let Some(candidate) = circumcircle(unique[i], unique[j], unique[k], BRUTE_EPS)
                else {
                    continue;
                };
                if unique
                    .iter()
                    .all(|&point| in_circle(point, &candidate, BRUTE_EPS))
                {
                    best = Some(match best {
                        Some(current) if current.radius <= candidate.radius => current,
                        _ => candidate,
                    });
                }
            }
        }
    }
    best.map(|circle| std::f64::consts::PI * circle.radius.powi(2))
}

#[test]
fn minimum_circle_matches_known_shapes() {
    assert_close(
        circle_area(square_points(0.0, 0.0)).unwrap(),
        std::f64::consts::PI / 2.0,
        1e-9,
    );
    assert_close(
        circle_area(vec![coord(0.0, 0.0), coord(4.0, 0.0), coord(0.0, 3.0)]).unwrap(),
        std::f64::consts::PI * 6.25,
        1e-9,
    );
    assert_close(
        circle_area(vec![coord(0.0, 0.0), coord(0.0, 2.0), coord(0.0, 5.0)]).unwrap(),
        std::f64::consts::PI * 6.25,
        1e-9,
    );
}

fn generated_cloud(seed: &mut u64, len: usize) -> Vec<Coordinate> {
    let mut points = Vec::new();
    for _ in 0..len {
        *seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let x = ((*seed >> 32) % 2000) as f64 / 10.0 - 100.0;
        *seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let y = ((*seed >> 32) % 2000) as f64 / 10.0 - 100.0;
        points.push(coord(x, y));
    }
    points
}

#[test]
fn minimum_circle_matches_brute_force_for_generated_clouds() {
    let mut seed = 0x5eed_cafe_u64;
    for case in 0..150 {
        let points = generated_cloud(&mut seed, 2 + case % 12);

        let expected = brute_force_circle_area(&points).unwrap();
        let actual = circle_area(points).unwrap();
        assert_close(actual, expected, 1e-7);
    }
}

#[test]
fn minimum_circle_is_translation_invariant_at_projected_crs_magnitudes() {
    for (points, expected) in [
        (square_points(0.0, 0.0), std::f64::consts::PI / 2.0),
        (
            vec![coord(0.0, 0.0), coord(4.0, 0.0), coord(0.0, 3.0)],
            std::f64::consts::PI * 6.25,
        ),
        (
            vec![coord(0.0, 0.0), coord(0.0, 2.0), coord(0.0, 5.0)],
            std::f64::consts::PI * 6.25,
        ),
    ] {
        let actual = circle_area(translated(&points, 6e6, 5e6)).unwrap();
        assert_close(actual, expected, 1e-9);
    }
}

#[test]
fn translated_clouds_match_the_untranslated_brute_force_oracle() {
    let mut seed = 0x5eed_cafe_u64;
    for case in 0..150 {
        let points = generated_cloud(&mut seed, 2 + case % 12);

        // Translation cannot change the area, so the small-coordinate oracle stays valid.
        let expected = brute_force_circle_area(&points).unwrap();
        let actual = circle_area(translated(&points, 6e6, 5e6)).unwrap();
        assert_close(actual, expected, 1e-6);
    }
}

#[test]
fn collinear_triple_with_outlier_at_projected_crs_magnitudes() {
    // A(0,0), B(4,0), C(8,0) are exactly collinear; outlier D(4,5) forces the circumcircle of
    // A, C, D with center (4, 0.9) and squared radius 16.81.
    let points = translated(
        &[
            coord(0.0, 0.0),
            coord(4.0, 0.0),
            coord(8.0, 0.0),
            coord(4.0, 5.0),
        ],
        6e6,
        5e6,
    );

    let mut rng = fastrand::Rng::with_seed(REOCK_SHUFFLE_SEED);
    let circle = minimum_enclosing_circle(&mut points.clone(), &mut rng).unwrap();
    assert_close(
        std::f64::consts::PI * circle.radius.powi(2),
        std::f64::consts::PI * 16.81,
        1e-6,
    );
    for &point in &points {
        assert!(l2_sq_dist(point, circle.center).sqrt() <= circle.radius * (1.0 + 1e-9));
    }
}

#[test]
fn returned_circle_contains_every_point_of_random_large_coordinate_clouds() {
    let mut seed = 0xfeed_beef_u64;
    for case in 0..150_u64 {
        let points = translated(
            &generated_cloud(&mut seed, 3 + case as usize % 15),
            6e6,
            5e6,
        );

        let mut rng = fastrand::Rng::with_seed(REOCK_SHUFFLE_SEED ^ case);
        let circle = minimum_enclosing_circle(&mut points.clone(), &mut rng).unwrap();
        for &point in &points {
            let distance = l2_sq_dist(point, circle.center).sqrt();
            assert!(
                distance <= circle.radius * (1.0 + 1e-9),
                "case {case}: point at distance {distance} outside radius {}",
                circle.radius
            );
        }
    }
}

#[test]
fn scores_hand_computable_sparse_districts() {
    let metric = PreparedReock::new(vec![
        square_unit(0.0, 0.0),
        square_unit(1.0, 0.0),
        square_unit(10.0, 0.0),
        square_unit(11.0, 0.0),
    ])
    .unwrap();

    let result = metric.score(&[3, 3, 7, 7]).unwrap();
    let expected = 8.0 / (5.0 * std::f64::consts::PI);
    assert_eq!(result.district_ids(), &[3, 7]);
    for &score in result.column(0).unwrap() {
        assert_close(score, expected, 1e-9);
    }
}

#[test]
fn validates_prepared_geometry() {
    assert_eq!(
        PreparedReock::new(vec![UnitHull::new(0.0, square_points(0.0, 0.0))]).unwrap_err(),
        Error::InvalidGeometryArea { unit: 0, area: 0.0 }
    );
    assert_eq!(
        PreparedReock::new(vec![UnitHull::new(1.0, vec![coord(0.0, 0.0)])]).unwrap_err(),
        Error::InvalidGeometryPointCount { unit: 0, actual: 1 }
    );
    assert_eq!(
        PreparedReock::new(vec![UnitHull::new(
            1.0,
            vec![coord(0.0, 0.0), coord(1.0, f64::NAN), coord(1.0, 1.0)],
        )])
        .unwrap_err(),
        Error::NonFiniteGeometryPoint { unit: 0, point: 1 }
    );
}

#[test]
fn rejects_impossible_score() {
    let metric = PreparedReock::new(vec![UnitHull::new(2.0, square_points(0.0, 0.0))]).unwrap();
    assert!(matches!(
        metric.score(&[4]),
        Err(Error::ImpossibleScore {
            metric: "Reock score",
            district: 4,
            ..
        })
    ));
}

#[test]
fn validates_assignment() {
    let metric = PreparedReock::new(vec![square_unit(0.0, 0.0)]).unwrap();
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
fn incremental_updates_match_full_recomputation() {
    let metric = PreparedReock::new(vec![
        square_unit(0.0, 0.0),
        square_unit(2.0, 1.0),
        square_unit(4.0, 0.0),
        square_unit(6.0, 1.0),
        square_unit(8.0, 0.0),
        square_unit(10.0, 1.0),
    ])
    .unwrap();
    let mut assignment = vec![1, 1, 1, 2, 2, 2];
    let changes = [
        DeltaChange {
            node: 1,
            old: 1,
            new: 2,
        },
        DeltaChange {
            node: 4,
            old: 2,
            new: 1,
        },
    ];
    let mut incremental = metric.incremental(&assignment).unwrap();

    incremental.update(&changes).unwrap();
    for change in changes {
        assignment[change.node] = change.new;
    }

    let expected = metric.score(&assignment).unwrap();
    let actual = incremental.result();
    assert_eq!(actual.district_ids(), expected.district_ids());
    for (&actual, &expected) in actual
        .column(0)
        .unwrap()
        .iter()
        .zip(expected.column(0).unwrap())
    {
        assert_close(actual, expected, 1e-8);
    }
}

#[test]
fn incremental_updates_remain_correct_at_projected_crs_magnitudes() {
    let units = (0..12)
        .map(|node| {
            UnitHull::new(
                1.0,
                translated(
                    &square_points((node % 6 * 2) as f64, (node / 6 * 3) as f64),
                    6e6,
                    5e6,
                ),
            )
        })
        .collect();
    let metric = PreparedReock::new(units).unwrap();
    let mut assignment = vec![1; 6];
    assignment.extend([2; 6]);
    let mut incremental = metric.incremental(&assignment).unwrap();

    for _ in 0..200 {
        let left = assignment
            .iter()
            .position(|&district| district == 1)
            .unwrap();
        let right = assignment
            .iter()
            .position(|&district| district == 2)
            .unwrap();
        let mut changes = [
            DeltaChange {
                node: left,
                old: 1,
                new: 2,
            },
            DeltaChange {
                node: right,
                old: 2,
                new: 1,
            },
        ];
        changes.sort_unstable_by_key(|change| change.node);

        incremental.update(&changes).unwrap();
        assignment[left] = 2;
        assignment[right] = 1;

        let expected = metric.score(&assignment).unwrap();
        let actual = incremental.result();
        assert_eq!(actual.district_ids(), expected.district_ids());
        for (&actual, &expected) in actual
            .column(0)
            .unwrap()
            .iter()
            .zip(expected.column(0).unwrap())
        {
            assert_close(actual, expected, 1e-8);
        }
    }
}

#[test]
fn incremental_clears_an_emptied_district() {
    let metric = PreparedReock::new(vec![square_unit(0.0, 0.0), square_unit(2.0, 0.0)]).unwrap();
    let mut assignment = vec![3, 7];
    let changes = [DeltaChange {
        node: 0,
        old: 3,
        new: 7,
    }];
    let mut incremental = metric.incremental(&assignment).unwrap();

    incremental.update(&changes).unwrap();
    assignment[0] = 7;

    assert_eq!(incremental.result().district_ids(), &[7]);
    let expected = metric.score(&assignment).unwrap();
    assert_close(
        incremental.result().column(0).unwrap()[0],
        expected.column(0).unwrap()[0],
        1e-8,
    );
}

#[test]
fn scoring_a_district_is_independent_of_what_ran_through_the_scratch_before_it() {
    let mut seed = 0xfeed_beef_u64;
    for case in 0..150_usize {
        let units = (0..4)
            .map(|offset| {
                UnitHull::new(
                    1.0,
                    translated(
                        &generated_cloud(&mut seed, 3 + (case + offset) % 15),
                        6e6,
                        5e6,
                    ),
                )
            })
            .collect();
        let metric = PreparedReock::new(units).unwrap();
        let nodes = [0_usize, 1, 2, 3];
        let area = 4.0;

        let mut scratch = metric.scratch(false);
        let first = metric
            .score_district(&nodes, area, 1, &mut scratch)
            .unwrap();
        // Advance the scratch the way scoring a neighbouring district would.
        metric
            .score_district(&nodes[..2], 2.0, 2, &mut scratch)
            .unwrap();
        let repeated = metric
            .score_district(&nodes, area, 1, &mut scratch)
            .unwrap();

        assert_eq!(
            first, repeated,
            "case {case}: score changed after the scratch was used elsewhere"
        );
    }
}

#[test]
fn scoring_is_bit_identical_when_only_the_district_label_changes() {
    let mut seed = 0x771e_1abe_u64;
    for case in 0..150_usize {
        let units = (0..4)
            .map(|offset| {
                UnitHull::new(
                    1.0,
                    translated(
                        &generated_cloud(&mut seed, 3 + (case + offset) % 15),
                        6e6,
                        5e6,
                    ),
                )
            })
            .collect();
        let metric = PreparedReock::new(units).unwrap();
        let nodes = [0_usize, 1, 2, 3];
        let mut scratch = metric.scratch(false);
        let first = metric.score_district(&nodes, 4.0, 1, &mut scratch).unwrap();
        let relabeled = metric.score_district(&nodes, 4.0, 7, &mut scratch).unwrap();

        assert_eq!(
            first, relabeled,
            "case {case}: changing only the district label changed the score"
        );
    }
}

#[test]
fn incremental_scores_are_bit_identical_to_full_recomputation() {
    // Exact equality pins canonical membership and area ordering across update histories.
    let units = (0..12)
        .map(|node| {
            UnitHull::new(
                1.0,
                translated(
                    &square_points((node % 6 * 2) as f64, (node / 6 * 3) as f64),
                    6e6,
                    5e6,
                ),
            )
        })
        .collect();
    let metric = PreparedReock::new(units).unwrap();
    let mut assignment = vec![1; 6];
    assignment.extend([2; 6]);
    let mut incremental = metric.incremental(&assignment).unwrap();

    let mut rng = fastrand::Rng::with_seed(0x5eed);
    for _ in 0..200 {
        let node = rng.usize(0..assignment.len());
        let old = assignment[node];
        let new = if old == 1 { 2 } else { 1 };
        if assignment
            .iter()
            .filter(|&&district| district == old)
            .count()
            == 1
        {
            continue; // never empty a district; that path has its own test
        }
        incremental
            .update(&[DeltaChange { node, old, new }])
            .unwrap();
        assignment[node] = new;

        let expected = metric.score(&assignment).unwrap();
        let actual = incremental.result();
        assert_eq!(actual.district_ids(), expected.district_ids());
        assert_eq!(actual.column(0).unwrap(), expected.column(0).unwrap());
    }
}
