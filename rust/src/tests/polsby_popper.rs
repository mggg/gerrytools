use super::*;

fn line_metric() -> PreparedPolsbyPopper {
    PreparedPolsbyPopper::new(
        vec![1.0; 4],
        vec![4.0; 4],
        vec![(0, 1), (1, 2), (2, 3)],
        vec![1.0; 3],
    )
    .unwrap()
}

#[test]
fn scores_known_example_with_sparse_district_ids() {
    let result = line_metric().score(&[3, 3, 7, 7]).unwrap();
    let expected = 2.0 * std::f64::consts::PI / 9.0;

    assert_eq!(result.district_ids(), &[3, 7]);
    assert_eq!(result.column_count(), 1);
    for score in result.column(0).unwrap() {
        assert!((score - expected).abs() < 1e-12);
    }
}

#[test]
fn derives_total_perimeters_from_boundary_values() {
    let metric = PreparedPolsbyPopper::from_boundary_perimeters(
        vec![1.0; 4],
        vec![3.0, 2.0, 2.0, 3.0],
        vec![(0, 1), (1, 2), (2, 3)],
        vec![1.0; 3],
    )
    .unwrap();

    assert_eq!(metric.total_perimeter_values, vec![4.0; 4]);
}

#[test]
fn validates_input_lengths_and_topology() {
    assert_eq!(
        PreparedPolsbyPopper::new(vec![1.0; 2], vec![4.0], vec![], vec![]).unwrap_err(),
        Error::NumericInputLength {
            input: "total perimeter values",
            actual: 1,
            expected: 2,
        }
    );
    assert_eq!(
        PreparedPolsbyPopper::new(vec![1.0; 2], vec![4.0; 2], vec![(0, 1)], vec![]).unwrap_err(),
        Error::SharedPerimeterCount {
            actual: 0,
            expected: 1,
        }
    );
    assert_eq!(
        PreparedPolsbyPopper::new(vec![1.0; 2], vec![4.0; 2], vec![(0, 5)], vec![1.0],)
            .unwrap_err(),
        Error::EdgeNodeOutOfRange {
            u: 0,
            v: 5,
            node_count: 2,
        }
    );
}

#[test]
fn validates_numeric_inputs() {
    assert_eq!(
        PreparedPolsbyPopper::new(vec![f64::NAN], vec![4.0], vec![], vec![]).unwrap_err(),
        Error::NonFinitePolsbyPopperInput
    );
    // Shared perimeters can still cancel a district's whole perimeter, so score validates too.
    assert_eq!(
        PreparedPolsbyPopper::new(vec![1.0; 2], vec![1.0; 2], vec![(0, 1)], vec![1.0])
            .unwrap()
            .score(&[4, 4])
            .unwrap_err(),
        Error::NonPositiveDistrictPerimeter {
            district: 4,
            perimeter: 0.0,
        }
    );
}

#[test]
fn rejects_nonpositive_areas_perimeters_and_negative_shared_perimeters() {
    assert!(matches!(
        PreparedPolsbyPopper::new(vec![0.0], vec![4.0], vec![], vec![]),
        Err(Error::InvalidInput(message)) if message.contains("nonpositive area")
    ));
    assert!(matches!(
        PreparedPolsbyPopper::new(vec![1.0], vec![0.0], vec![], vec![]),
        Err(Error::InvalidInput(message)) if message.contains("nonpositive total perimeter")
    ));
    assert!(matches!(
        PreparedPolsbyPopper::new(vec![1.0; 2], vec![4.0; 2], vec![(0, 1)], vec![-1.0]),
        Err(Error::InvalidInput(message)) if message.contains("negative shared perimeter")
    ));
    assert!(matches!(
        PreparedPolsbyPopper::from_boundary_perimeters(
            vec![1.0; 2],
            vec![-0.5; 2],
            vec![(0, 1)],
            vec![1e12],
        ),
        Err(Error::InvalidInput(message)) if message.contains("negative boundary perimeter")
    ));
    assert_eq!(
        PreparedPolsbyPopper::from_boundary_perimeters(vec![1.0], vec![f64::NAN], vec![], vec![],)
            .unwrap_err(),
        Error::NonFinitePolsbyPopperInput
    );
}

#[test]
fn rejects_summed_shared_perimeter_greater_than_unit_perimeter() {
    assert!(matches!(
        PreparedPolsbyPopper::new(
            vec![1.0; 3],
            vec![4.0; 3],
            vec![(0, 1), (0, 2)],
            vec![2.1, 2.1],
        ),
        Err(Error::InvalidInput(message))
            if message.contains("unit 0 has total perimeter 4")
                && message.contains("edges connected to it sum to 4.2")
                && message.contains("different geometries or CRSs")
                && message.contains("geometry-backed scoring")
    ));
}

#[test]
fn validates_assignments() {
    assert_eq!(
        line_metric().score(&[1]).unwrap_err(),
        Error::AssignmentLength {
            actual: 1,
            expected: 4,
        }
    );
    assert_eq!(
        line_metric().score(&[1, 1, 1, 500]).unwrap().district_ids(),
        &[1, 500]
    );
}

#[test]
fn rejects_impossible_scores_from_inconsistent_graph_inputs() {
    // Area 1 with perimeter 1 gives 4π, far above the 1.0 ceiling any real shape can reach,
    // so the caller-supplied graph inputs must be mutually inconsistent.
    let metric = PreparedPolsbyPopper::new(vec![1.0], vec![1.0], vec![], vec![]).unwrap();
    assert!(matches!(
        metric.score(&[0]).unwrap_err(),
        Error::ImpossibleScore {
            metric: "Polsby-Popper score",
            district: 0,
            ..
        }
    ));

    // The incremental path shares the guard.
    let incremental = metric.incremental(&[0]).unwrap();
    assert!(matches!(
        incremental.result().unwrap_err(),
        Error::ImpossibleScore {
            metric: "Polsby-Popper score",
            district: 0,
            ..
        }
    ));
}

#[test]
fn tolerates_float_noise_within_the_shared_ratio_epsilon() {
    // Score 1 + 5e-10 sits inside RATIO_SCORE_EPS and must not be rejected.
    let area = (1.0 + 5e-10) / (4.0 * std::f64::consts::PI);
    let metric = PreparedPolsbyPopper::new(vec![area], vec![1.0], vec![], vec![]).unwrap();

    let score = metric.score(&[0]).unwrap().column(0).unwrap()[0];

    assert!(score > 1.0 && score < 1.0 + 1e-9, "got {score}");
}

#[test]
fn incremental_updates_match_full_recomputation() {
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
    let mut incremental = metric.incremental(&assignment).unwrap();

    incremental.update(&changes).unwrap();
    for change in changes {
        assignment[change.node] = change.new;
    }

    let expected = metric.score(&assignment).unwrap();
    let actual = incremental.result().unwrap();
    assert_eq!(actual.district_ids(), expected.district_ids());
    for (&actual, &expected) in actual
        .column(0)
        .unwrap()
        .iter()
        .zip(expected.column(0).unwrap())
    {
        assert!((actual - expected).abs() < 1e-12);
    }
}

#[test]
fn incremental_clears_an_emptied_district() {
    let metric = line_metric();
    let mut assignment = vec![1, 2, 2, 2];
    let changes = [DeltaChange {
        node: 0,
        old: 1,
        new: 2,
    }];
    let mut incremental = metric.incremental(&assignment).unwrap();

    incremental.update(&changes).unwrap();
    assignment[0] = 2;

    assert_eq!(
        incremental.result().unwrap(),
        metric.score(&assignment).unwrap()
    );
    assert_eq!(incremental.result().unwrap().district_ids(), &[2]);
}
