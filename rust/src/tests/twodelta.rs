use super::*;

#[test]
fn validates_delta_changes() {
    let before = [1, 1, 2];
    assert_eq!(
        validate_changes(
            &before,
            &[DeltaChange {
                node: 3,
                old: 2,
                new: 1,
            }],
        ),
        Err(Error::DeltaNodeOutOfRange {
            node: 3,
            assignment_len: 3,
        })
    );
    assert_eq!(
        validate_changes(
            &before,
            &[DeltaChange {
                node: 1,
                old: 2,
                new: 1,
            }],
        ),
        Err(Error::DeltaOldLabelMismatch {
            node: 1,
            expected: 1,
            actual: 2,
        })
    );
}

#[test]
fn accepts_delta_labels_across_dynamic_words() {
    assert!(validate_changes(
        &[127],
        &[DeltaChange {
            node: 0,
            old: 127,
            new: 500,
        }],
    )
    .is_ok());
}

#[test]
fn rejects_duplicate_nodes_before_state_can_be_corrupted() {
    let change = DeltaChange {
        node: 1,
        old: 1,
        new: 2,
    };
    assert_eq!(
        validate_changes(&[1, 1, 2], &[change, change]),
        Err(Error::DeltaNodesNotStrictlyIncreasing {
            previous: 1,
            node: 1,
        })
    );
}

#[test]
fn rejects_unsorted_changes() {
    let changes = [
        DeltaChange {
            node: 0,
            old: 1,
            new: 2,
        },
        DeltaChange {
            node: 2,
            old: 2,
            new: 1,
        },
        DeltaChange {
            node: 1,
            old: 1,
            new: 3,
        },
    ];

    assert_eq!(
        validate_changes(&[1, 1, 2], &changes),
        Err(Error::DeltaNodesNotStrictlyIncreasing {
            previous: 2,
            node: 1,
        })
    );
}

#[test]
fn reads_post_delta_labels_without_copying_assignment() {
    let before = [1, 1, 2];
    let changes = [DeltaChange {
        node: 1,
        old: 1,
        new: 2,
    }];
    let mut labels = PostDeltaLabels::new(before.len());
    labels.refresh(&changes);

    assert_eq!(labels.label(&before, 0), 1);
    assert_eq!(labels.label(&before, 1), 2);
}
