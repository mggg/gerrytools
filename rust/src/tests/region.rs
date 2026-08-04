use super::*;
use std::collections::HashSet;

fn region_columns() -> Vec<Vec<Option<u32>>> {
    vec![
        vec![Some(10), Some(10), Some(42), None],
        vec![Some(7), Some(8), Some(8), Some(7)],
    ]
}

fn grid_edges(width: usize, height: usize) -> Vec<(u32, u32)> {
    let mut edges = Vec::new();
    for row in 0..height {
        for column in 0..width {
            let node = (row * width + column) as u32;
            if column + 1 < width {
                edges.push((node, node + 1));
            }
            if row + 1 < height {
                edges.push((node, node + width as u32));
            }
        }
    }
    edges
}

fn reference_parts(regions: &[Option<u32>], edges: &[(u32, u32)], assignment: &[u16]) -> usize {
    fn find(parent: &mut [usize], node: usize) -> usize {
        let mut root = node;
        while parent[root] != root {
            root = parent[root];
        }
        let mut current = node;
        while parent[current] != current {
            let next = parent[current];
            parent[current] = root;
            current = next;
        }
        root
    }

    let mut parent = (0..assignment.len()).collect::<Vec<_>>();
    for &(u, v) in edges {
        let u = u as usize;
        let v = v as usize;
        if regions[u].is_some() && regions[u] == regions[v] && assignment[u] == assignment[v] {
            let root_u = find(&mut parent, u);
            let root_v = find(&mut parent, v);
            parent[root_u] = root_v;
        }
    }
    regions
        .iter()
        .enumerate()
        .filter_map(|(node, region)| region.map(|_| find(&mut parent, node)))
        .collect::<HashSet<_>>()
        .len()
}

#[test]
fn parts_count_connected_components_inside_the_region() {
    let assignment = [1, 2, 2, 2, 1, 1, 1, 1, 1, 1, 2, 2, 1, 2, 2, 2];
    let edges = grid_edges(4, 4);
    let metric =
        PreparedRegion::parts(vec![vec![Some(0); assignment.len()]], edges.clone()).unwrap();

    assert_eq!(edges.len(), 24);
    assert_eq!(metric.score(&assignment).unwrap().values(), &[3.0]);
}

#[test]
fn pieces_preserve_the_former_occupancy_count() {
    let assignment = [1, 2, 2, 2, 1, 1, 1, 1, 1, 1, 2, 2, 1, 2, 2, 2];
    let regions = vec![vec![Some(0); assignment.len()]];
    let pieces = PreparedRegion::pieces(regions.clone()).unwrap();
    let parts = PreparedRegion::parts(regions, grid_edges(4, 4)).unwrap();

    assert_eq!(pieces.score(&assignment).unwrap().values(), &[2.0]);
    assert_eq!(parts.score(&assignment).unwrap().values(), &[3.0]);
}

#[test]
fn parts_ignore_connections_outside_the_region() {
    let edges = vec![(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)];
    let regions = vec![None, Some(0), Some(0), Some(0), Some(0), None];
    let assignment = [2, 2, 1, 1, 2, 2];
    let metric = PreparedRegion::parts(vec![regions], edges).unwrap();

    assert_eq!(metric.score(&assignment).unwrap().values(), &[3.0]);
}

#[test]
fn disconnected_fixed_regions_count_as_multiple_parts() {
    let metric = PreparedRegion::parts(
        vec![vec![Some(0), Some(0), Some(0), Some(0)]],
        vec![(0, 1), (2, 3)],
    )
    .unwrap();

    assert_eq!(metric.score(&[1, 1, 1, 1]).unwrap().values(), &[2.0]);
}

#[test]
fn scores_multiple_region_columns_with_sparse_region_ids() {
    let assignment = [1, 2, 3, 3];
    let splits = PreparedRegion::splits(region_columns()).unwrap();
    let pieces = PreparedRegion::pieces(region_columns()).unwrap();
    let parts = PreparedRegion::parts(region_columns(), grid_edges(4, 1)).unwrap();

    let split_result = splits.score(&assignment).unwrap();
    let piece_result = pieces.score(&assignment).unwrap();
    let part_result = parts.score(&assignment).unwrap();

    assert_eq!(split_result.district_ids(), &[1, 2, 3]);
    assert_eq!(split_result.values(), &[1.0, 2.0]);
    assert_eq!(piece_result.values(), &[3.0, 4.0]);
    assert_eq!(part_result.values(), &[3.0, 4.0]);
}

#[test]
fn handles_district_ids_across_bitset_words() {
    let splits = PreparedRegion::splits(vec![vec![Some(0), Some(0), Some(1)]]).unwrap();
    let pieces = PreparedRegion::pieces(vec![vec![Some(0), Some(0), Some(1)]]).unwrap();
    let parts =
        PreparedRegion::parts(vec![vec![Some(0), Some(0), Some(1)]], grid_edges(3, 1)).unwrap();
    let mut assignment = [0, 64, 127];
    let mut piece_state = pieces.incremental(&assignment).unwrap();

    assert_eq!(splits.score(&assignment).unwrap().values(), &[1.0]);
    assert_eq!(pieces.score(&assignment).unwrap().values(), &[3.0]);
    assert_eq!(parts.score(&assignment).unwrap().values(), &[3.0]);

    let changes = [DeltaChange {
        node: 1,
        old: 64,
        new: 0,
    }];
    piece_state.update(&changes).unwrap();
    assignment[1] = 0;
    assert_eq!(piece_state.result(), pieces.score(&assignment).unwrap());
    assert_eq!(piece_state.result().values(), &[2.0]);
}

#[test]
fn missing_regions_contribute_no_region_scores() {
    let splits = PreparedRegion::splits(vec![vec![None, None]]).unwrap();
    let pieces = PreparedRegion::pieces(vec![vec![None, None]]).unwrap();
    let parts = PreparedRegion::parts(vec![vec![None, None]], grid_edges(2, 1)).unwrap();

    assert_eq!(splits.score(&[1, 2]).unwrap().values(), &[0.0]);
    assert_eq!(pieces.score(&[1, 2]).unwrap().values(), &[0.0]);
    assert_eq!(parts.score(&[1, 2]).unwrap().values(), &[0.0]);
}

#[test]
fn validates_region_columns() {
    assert_eq!(
        PreparedRegion::splits(vec![]).unwrap_err(),
        Error::EmptyRegionMetric
    );
    assert_eq!(
        PreparedRegion::parts(vec![vec![Some(0)], vec![]], vec![]).unwrap_err(),
        Error::RegionColumnLength {
            column: 1,
            actual: 0,
            expected: 1,
        }
    );
    assert_eq!(
        PreparedRegion::parts(vec![vec![Some(0)]], vec![(0, 1)]).unwrap_err(),
        Error::EdgeNodeOutOfRange {
            u: 0,
            v: 1,
            node_count: 1,
        }
    );
}

#[test]
fn accepts_empty_aligned_columns() {
    let splits = PreparedRegion::splits(vec![vec![], vec![]]).unwrap();
    let pieces = PreparedRegion::pieces(vec![vec![], vec![]]).unwrap();
    let parts = PreparedRegion::parts(vec![vec![], vec![]], vec![]).unwrap();
    let split_result = splits.score(&[]).unwrap();
    let piece_result = pieces.score(&[]).unwrap();
    let part_result = parts.score(&[]).unwrap();

    assert!(split_result.district_ids().is_empty());
    assert_eq!(split_result.values(), &[0.0, 0.0]);
    assert_eq!(piece_result, pieces.incremental(&[]).unwrap().result());
    assert_eq!(piece_result.values(), &[0.0, 0.0]);
    assert_eq!(part_result, parts.incremental(&[]).unwrap().result());
    assert_eq!(part_result.values(), &[0.0, 0.0]);
}

fn assert_incremental_matches_full(metric: PreparedRegion) {
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

    state.reset(&[2, 2, 2, 2]).unwrap();
    assert_eq!(state.result(), metric.score(&[2, 2, 2, 2]).unwrap());
}

#[test]
fn incremental_region_scalars_match_full_recomputation() {
    assert_incremental_matches_full(PreparedRegion::splits(region_columns()).unwrap());
    assert_incremental_matches_full(PreparedRegion::pieces(region_columns()).unwrap());
    assert_incremental_matches_full(
        PreparedRegion::parts(region_columns(), grid_edges(4, 1)).unwrap(),
    );
}

#[test]
fn incremental_parts_split_at_an_articulation_and_merge_back() {
    let metric =
        PreparedRegion::parts(vec![vec![Some(0), Some(0), Some(0)]], grid_edges(3, 1)).unwrap();
    let mut assignment = vec![1, 1, 1];
    let mut state = metric.incremental(&assignment).unwrap();

    let split = [DeltaChange {
        node: 1,
        old: 1,
        new: 2,
    }];
    state.update(&split).unwrap();
    assignment[1] = 2;
    assert_eq!(state.result().values(), &[3.0]);
    assert_eq!(state.result(), metric.score(&assignment).unwrap());

    let merge = [DeltaChange {
        node: 1,
        old: 2,
        new: 1,
    }];
    state.update(&merge).unwrap();
    assignment[1] = 1;
    assert_eq!(state.result().values(), &[1.0]);
    assert_eq!(state.result(), metric.score(&assignment).unwrap());
}

#[test]
fn simultaneous_changes_match_full_recomputation() {
    let metric = PreparedRegion::parts(vec![vec![Some(0); 5]], grid_edges(5, 1)).unwrap();
    let mut assignment = vec![1; 5];
    let changes = [
        DeltaChange {
            node: 1,
            old: 1,
            new: 2,
        },
        DeltaChange {
            node: 3,
            old: 1,
            new: 2,
        },
    ];
    let mut state = metric.incremental(&assignment).unwrap();

    state.update(&changes).unwrap();
    for change in changes {
        assignment[change.node] = change.new;
    }

    assert_eq!(state.result().values(), &[5.0]);
    assert_eq!(state.result(), metric.score(&assignment).unwrap());

    let followup = [DeltaChange {
        node: 2,
        old: 1,
        new: 2,
    }];
    state.update(&followup).unwrap();
    assignment[2] = 2;
    assert_eq!(state.result(), metric.score(&assignment).unwrap());
}

#[test]
fn invalid_delta_does_not_partially_update_regions() {
    let metric = PreparedRegion::splits(region_columns()).unwrap();
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

#[test]
fn invalid_delta_does_not_partially_update_part_components() {
    let metric = PreparedRegion::parts(vec![vec![Some(0); 4]], grid_edges(4, 1)).unwrap();
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

#[test]
fn generated_full_scores_match_an_independent_union_find_oracle() {
    for seed in 0..64 {
        let mut rng = fastrand::Rng::with_seed(seed);
        for node_count in 1..=12 {
            let mut edges = Vec::new();
            for u in 0..node_count {
                for v in u + 1..node_count {
                    if rng.usize(0..4) == 0 {
                        edges.push((u as u32, v as u32));
                    }
                }
            }
            let columns = (0..3)
                .map(|_| {
                    (0..node_count)
                        .map(|_| (rng.usize(0..5) != 0).then(|| rng.usize(0..3) as u32))
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>();
            let assignment = (0..node_count)
                .map(|_| rng.usize(0..4) as u16)
                .collect::<Vec<_>>();
            let expected = columns
                .iter()
                .map(|column| reference_parts(column, &edges, &assignment) as f64)
                .collect::<Vec<_>>();
            let metric = PreparedRegion::parts(columns.clone(), edges.clone()).unwrap();
            let actual = metric.score(&assignment).unwrap();
            let pieces = PreparedRegion::pieces(columns.clone())
                .unwrap()
                .score(&assignment)
                .unwrap();

            assert_eq!(actual.values(), expected, "seed {seed}, nodes {node_count}");
            for ((column, &parts), &piece_count) in
                columns.iter().zip(actual.values()).zip(pieces.values())
            {
                let tagged = column.iter().filter(|region| region.is_some()).count() as f64;
                let intersections = column
                    .iter()
                    .zip(&assignment)
                    .filter_map(|(region, &district)| region.map(|region| (region, district)))
                    .collect::<HashSet<_>>()
                    .len() as f64;
                assert_eq!(piece_count, intersections);
                assert!(intersections <= parts);
                assert!(parts <= tagged);
            }
        }
    }
}

#[test]
fn exhaustive_four_node_scores_match_the_union_find_oracle() {
    let possible_edges = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)];
    for edge_mask in 0_u8..(1 << possible_edges.len()) {
        let edges = possible_edges
            .iter()
            .enumerate()
            .filter(|(index, _)| edge_mask & (1 << index) != 0)
            .map(|(_, &edge)| edge)
            .collect::<Vec<_>>();
        for assignment_mask in 0_u8..16 {
            let assignment = (0..4)
                .map(|node| ((assignment_mask >> node) & 1) as u16)
                .collect::<Vec<_>>();
            for mut region_code in 0_u8..81 {
                let regions = (0..4)
                    .map(|_| {
                        let region = match region_code % 3 {
                            0 => None,
                            1 => Some(10),
                            _ => Some(20),
                        };
                        region_code /= 3;
                        region
                    })
                    .collect::<Vec<_>>();
                let expected = reference_parts(&regions, &edges, &assignment) as f64;
                let metric = PreparedRegion::parts(vec![regions], edges.clone()).unwrap();

                assert_eq!(metric.score(&assignment).unwrap().values(), &[expected]);
            }
        }
    }
}

#[test]
fn generated_incremental_updates_match_full_and_union_find_scores() {
    for seed in 0..128 {
        let mut rng = fastrand::Rng::with_seed(10_000 + seed);
        let node_count = 16;
        let mut edges = Vec::new();
        for u in 0..node_count {
            for v in u + 1..node_count {
                if rng.usize(0..5) == 0 {
                    edges.push((u as u32, v as u32));
                }
            }
        }
        let columns = (0..2)
            .map(|_| {
                (0..node_count)
                    .map(|_| (rng.usize(0..6) != 0).then(|| rng.usize(0..4) as u32))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let mut assignment = (0..node_count)
            .map(|_| rng.usize(0..4) as u16)
            .collect::<Vec<_>>();
        let metric = PreparedRegion::parts(columns.clone(), edges.clone()).unwrap();
        let mut state = metric.incremental(&assignment).unwrap();
        let piece_metric = PreparedRegion::pieces(columns.clone()).unwrap();
        let mut piece_state = piece_metric.incremental(&assignment).unwrap();

        for step in 0..250 {
            let mut nodes = (0..node_count).collect::<Vec<_>>();
            rng.shuffle(&mut nodes);
            let mut changed_nodes = nodes[..1 + rng.usize(0..4)].to_vec();
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
            piece_state.update(&changes).unwrap();
            for change in changes {
                assignment[change.node] = change.new;
            }

            let full = metric.score(&assignment).unwrap();
            let oracle = columns
                .iter()
                .map(|column| reference_parts(column, &edges, &assignment) as f64)
                .collect::<Vec<_>>();
            let piece_oracle = columns
                .iter()
                .map(|column| {
                    column
                        .iter()
                        .zip(&assignment)
                        .filter_map(|(region, &district)| region.map(|region| (region, district)))
                        .collect::<HashSet<_>>()
                        .len() as f64
                })
                .collect::<Vec<_>>();
            assert_eq!(state.result(), full, "seed {seed}, step {step}");
            assert_eq!(state.result().values(), oracle, "seed {seed}, step {step}");
            assert_eq!(
                piece_state.result(),
                piece_metric.score(&assignment).unwrap()
            );
            assert_eq!(piece_state.result().values(), piece_oracle);
        }
    }
}
