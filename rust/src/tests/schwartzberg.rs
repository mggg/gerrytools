use crate::{DeltaChange, Error, PreparedAreaPerimeterMetrics, PreparedSchwartzberg};

fn metric() -> PreparedSchwartzberg {
    PreparedSchwartzberg::new(vec![1.0, 1.0], vec![4.0, 4.0], vec![(0, 1)], vec![1.0]).unwrap()
}

#[test]
fn transforms_full_and_incremental_polsby_popper_scores() {
    let metric = metric();
    let full = metric.score(&[0, 1]).unwrap();
    let expected = 2.0 / std::f64::consts::PI.sqrt();
    assert!((full.column(0).unwrap()[0] - expected).abs() < 1e-12);

    let mut incremental = metric.incremental(&[0, 1]).unwrap();
    incremental
        .update(&[DeltaChange {
            node: 1,
            old: 1,
            new: 0,
        }])
        .unwrap();
    assert_eq!(
        incremental.result().unwrap(),
        metric.score(&[0, 0]).unwrap()
    );
}

#[test]
fn rejects_nonpositive_inputs_through_the_shared_constructor() {
    assert!(matches!(
        PreparedSchwartzberg::new(vec![0.0, 1.0], vec![4.0, 4.0], vec![(0, 1)], vec![1.0]),
        Err(Error::InvalidInput(message)) if message.contains("nonpositive area")
    ));
    assert!(matches!(
        PreparedSchwartzberg::new(vec![1.0, 1.0], vec![4.0, 0.0], vec![(0, 1)], vec![1.0]),
        Err(Error::InvalidInput(message)) if message.contains("nonpositive total perimeter")
    ));
}

#[test]
fn combined_metric_returns_polsby_popper_then_schwartzberg_from_one_state() {
    let metric =
        PreparedAreaPerimeterMetrics::new(vec![1.0, 1.0], vec![4.0, 4.0], vec![(0, 1)], vec![1.0])
            .unwrap();

    let score = metric.score(&[0, 1]).unwrap();

    let polsby = std::f64::consts::PI / 4.0;
    assert_eq!(score.column_count(), 2);
    assert!((score.column(0).unwrap()[0] - polsby).abs() < 1e-12);
    assert!((score.column(1).unwrap()[0] - 1.0 / polsby.sqrt()).abs() < 1e-12);
}
