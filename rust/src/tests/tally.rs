use super::*;

#[test]
fn scores_multiple_columns_with_sparse_district_ids() {
    let tally = PreparedTally::new(vec![vec![1.0, 2.0, 3.0], vec![10.0, 20.0, 30.0]]).unwrap();

    let result = tally.score(&[1, 3, 1]).unwrap();

    assert_eq!(result.district_ids(), &[1, 3]);
    assert_eq!(result.column(0), Some(&[4.0, 2.0][..]));
    assert_eq!(result.column(1), Some(&[40.0, 20.0][..]));
    assert_eq!(result.column(2), None);
}

#[test]
fn accepts_an_empty_graph_without_inventing_a_district() {
    let tally = PreparedTally::new(vec![vec![]]).unwrap();
    let result = tally.score(&[]).unwrap();

    assert!(result.district_ids().is_empty());
    assert_eq!(result.column(0), Some(&[][..]));
    assert_eq!(result.column(1), None);
}

#[test]
fn validates_prepared_columns() {
    assert_eq!(PreparedTally::new(vec![]).unwrap_err(), Error::EmptyTally);
    assert_eq!(
        PreparedTally::new(vec![vec![1.0], vec![]]).unwrap_err(),
        Error::TallyColumnLength {
            column: 1,
            actual: 0,
            expected: 1,
        }
    );
    assert_eq!(
        PreparedTally::new(vec![vec![f64::NAN]]).unwrap_err(),
        Error::NonFiniteTallyValue { column: 0 }
    );
}

#[test]
fn validates_assignment_length() {
    let tally = PreparedTally::new(vec![vec![1.0, 2.0]]).unwrap();
    assert_eq!(
        tally.score(&[0]).unwrap_err(),
        Error::AssignmentLength {
            actual: 1,
            expected: 2,
        }
    );
}

#[test]
fn incremental_updates_match_full_recomputation() {
    let tally =
        PreparedTally::new(vec![vec![1.0, 2.0, 3.0, 4.0], vec![10.0, 20.0, 30.0, 40.0]]).unwrap();
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
    let mut incremental = tally.incremental(&assignment).unwrap();

    incremental.update(&changes).unwrap();
    for change in changes {
        assignment[change.node] = change.new;
    }

    assert_eq!(incremental.result(), tally.score(&assignment).unwrap());
}

#[test]
fn incremental_state_grows_when_a_delta_adds_a_high_district() {
    let tally = PreparedTally::new(vec![vec![1.0, 2.0, 3.0]]).unwrap();
    let mut incremental = tally.incremental(&[0, 0, 1]).unwrap();

    incremental
        .update(&[DeltaChange {
            node: 1,
            old: 0,
            new: 499,
        }])
        .unwrap();

    let result = incremental.result();
    assert_eq!(result.district_ids(), &[0, 1, 499]);
    assert_eq!(result.column(0), Some(&[1.0, 3.0, 2.0][..]));
    assert_eq!(incremental.totals.len(), 500);
}

#[test]
fn invalid_delta_does_not_partially_update_tallies() {
    let tally = PreparedTally::new(vec![vec![1.0, 2.0, 3.0]]).unwrap();
    let assignment = [1, 1, 2];
    let mut incremental = tally.incremental(&assignment).unwrap();
    let before = incremental.result();
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

    assert!(incremental.update(&changes).is_err());
    assert_eq!(incremental.result(), before);
}
