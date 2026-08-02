use super::*;
use crate::{DistrictTable, PreparedConvexHullRatio, PreparedReock};
use geo::{BooleanOps, ConvexHull, MultiPoint, Point};

fn polygon_wkb(rings: &[&[(f64, f64)]]) -> Vec<u8> {
    let mut bytes = vec![1];
    bytes.extend_from_slice(&3_u32.to_le_bytes());
    bytes.extend_from_slice(&(rings.len() as u32).to_le_bytes());
    for ring in rings {
        bytes.extend_from_slice(&(ring.len() as u32).to_le_bytes());
        for &(x, y) in *ring {
            bytes.extend_from_slice(&x.to_le_bytes());
            bytes.extend_from_slice(&y.to_le_bytes());
        }
    }
    bytes
}

fn encoded_square(little_endian: bool, type_code: u32, dimension: usize) -> Vec<u8> {
    let mut bytes = vec![u8::from(little_endian)];
    let push_u32 = |bytes: &mut Vec<u8>, value: u32| {
        let encoded = if little_endian {
            value.to_le_bytes()
        } else {
            value.to_be_bytes()
        };
        bytes.extend_from_slice(&encoded);
    };
    let push_f64 = |bytes: &mut Vec<u8>, value: f64| {
        let encoded = if little_endian {
            value.to_le_bytes()
        } else {
            value.to_be_bytes()
        };
        bytes.extend_from_slice(&encoded);
    };
    push_u32(&mut bytes, type_code);
    if type_code & 0x2000_0000 != 0 {
        push_u32(&mut bytes, 3857);
    }
    push_u32(&mut bytes, 1);
    push_u32(&mut bytes, 5);
    for (x, y) in [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0), (0.0, 0.0)] {
        push_f64(&mut bytes, x);
        push_f64(&mut bytes, y);
        for _ in 2..dimension {
            push_f64(&mut bytes, 0.0);
        }
    }
    bytes
}

fn square(x: f64, y: f64) -> Vec<u8> {
    polygon_wkb(&[&[
        (x, y),
        (x + 1.0, y),
        (x + 1.0, y + 1.0),
        (x, y + 1.0),
        (x, y),
    ]])
}

fn rectangle(x0: f64, y0: f64, x1: f64, y1: f64) -> Vec<u8> {
    polygon_wkb(&[&[(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)]])
}

fn multipolygon_wkb(polygons: &[Vec<u8>]) -> Vec<u8> {
    let mut bytes = vec![1];
    bytes.extend_from_slice(&6_u32.to_le_bytes());
    bytes.extend_from_slice(&(polygons.len() as u32).to_le_bytes());
    for polygon in polygons {
        bytes.extend_from_slice(polygon);
    }
    bytes
}

fn score(table: &DistrictTable) -> &[f64] {
    table.column(0).unwrap()
}

fn complete_geometry_hull(geometries: &[MultiPolygon<f64>], nodes: &[usize]) -> Polygon<f64> {
    MultiPolygon(
        nodes
            .iter()
            .flat_map(|&node| geometries[node].0.iter().cloned())
            .collect(),
    )
    .convex_hull()
}

fn prepared_geometry_hull(prepared: &PreparedUnitHulls, nodes: &[usize]) -> Polygon<f64> {
    MultiPoint(
        nodes
            .iter()
            .flat_map(|&node| prepared.unit_hull_points(node))
            .map(|coordinate| Point::new(coordinate.x, coordinate.y))
            .collect(),
    )
    .convex_hull()
}

fn assert_same_hull(
    geometries: &[MultiPolygon<f64>],
    prepared: &PreparedUnitHulls,
    nodes: &[usize],
) {
    let complete = complete_geometry_hull(geometries, nodes);
    let reconstructed = prepared_geometry_hull(prepared, nodes);
    assert!((complete.unsigned_area() - reconstructed.unsigned_area()).abs() < 1e-12);
    assert!(complete.xor(&reconstructed).unsigned_area().abs() < 1e-12);
}

#[test]
fn prepares_geometry_metrics_from_ordered_wkb() {
    let rows = [square(0.0, 0.0), square(1.0, 0.0)];
    let reock = PreparedReock::from_wkb(&rows).unwrap();
    let convex_hull_ratio = PreparedConvexHullRatio::from_wkb(&rows).unwrap();
    let state_clipped = PreparedStateClippedConvexHullRatio::from_wkb(
        &rows,
        &polygon_wkb(&[&[(0.0, 0.0), (2.0, 0.0), (2.0, 1.0), (0.0, 1.0), (0.0, 0.0)]]),
    )
    .unwrap();
    let polsby = PreparedPolsbyPopper::from_wkb(&rows, vec![(0, 1)]).unwrap();

    let reock_score = reock.score(&[0, 0]).unwrap();
    assert!((score(&reock_score)[0] - 2.0 / (1.25 * std::f64::consts::PI)).abs() < 1e-12);
    assert!((score(&convex_hull_ratio.score(&[0, 0]).unwrap())[0] - 1.0).abs() < 1e-12);
    assert!((score(&state_clipped.score(&[0, 0]).unwrap())[0] - 1.0).abs() < 1e-12);

    let combined = polsby.score(&[0, 0]).unwrap();
    assert!((score(&combined)[0] - 2.0 * std::f64::consts::PI / 9.0).abs() < 1e-12);
    let separate = polsby.score(&[0, 1]).unwrap();
    assert!(score(&separate)
        .iter()
        .all(|value| (*value - std::f64::consts::PI / 4.0).abs() < 1e-12));
}

#[test]
fn prepared_geometry_discovers_exact_rook_topology_and_caches_hulls() {
    let rows = [
        square(0.0, 0.0),
        square(1.0, 0.0),
        square(1.0, 1.0),
        square(3.0, 0.0),
    ];
    let geometry = PreparedGeometry::from_wkb(&rows).unwrap();

    let rook = geometry.rook_measurements().unwrap();
    let (edges, shared) = rook.inputs();
    assert_eq!(edges, [(0, 1), (1, 2)]);
    assert_eq!(shared, [1.0, 1.0]);

    let first = geometry.unit_hulls().unwrap();
    let second = geometry.unit_hulls().unwrap();
    assert!(Arc::ptr_eq(&first, &second));
}

#[test]
fn evaluator_rook_discovery_handles_split_and_zero_length_boundary_segments() {
    let left = rectangle(0.0, 0.0, 1.0, 2.0);
    let right = polygon_wkb(&[&[
        (1.0, 0.0),
        (2.0, 0.0),
        (2.0, 2.0),
        (1.0, 2.0),
        (1.0, 1.0),
        (1.0, 1.0),
        (1.0, 0.0),
    ]]);
    let geometry = PreparedGeometry::from_wkb(&[left, right]).unwrap();

    let (edges, shared) = geometry.rook_measurements().unwrap().inputs();
    assert_eq!(edges, [(0, 1)]);
    assert_eq!(shared, [2.0]);
}

#[test]
fn evaluator_rook_discovery_applies_overlap_and_shared_length_thresholds() {
    let shared = PreparedGeometry::from_wkb(&[
        rectangle(0.0, 0.0, 1.0, 1.0),
        rectangle(1.0, 0.0, 2.0, 2.0 * GEOMETRY_EPS),
    ])
    .unwrap();
    assert_eq!(shared.rook_measurements().unwrap().inputs().0, [(0, 1)]);

    let overlap = PreparedGeometry::from_wkb(&[
        rectangle(0.0, 0.0, 1.0, 1.0),
        rectangle(1.0 - 2.0 * GEOMETRY_EPS, 0.0, 2.0, 1.0),
    ])
    .unwrap()
    .rook_measurements()
    .unwrap_err();
    assert!(overlap.to_string().contains("overlap by area"));
}

#[test]
fn evaluator_rook_discovery_ignores_small_gaps_and_point_contacts() {
    let rows = [square(0.0, 0.0), square(1.0 + 5e-9, 0.0), square(-1.0, 1.0)];
    let geometry = PreparedGeometry::from_wkb(&rows).unwrap();

    let (edges, shared) = geometry.rook_measurements().unwrap().inputs();
    assert!(edges.is_empty(), "unexpected edges: {edges:?}");
    assert!(shared.is_empty());
}

#[test]
fn evaluator_rook_discovery_rejects_the_lowest_short_boundary() {
    let rows = [
        rectangle(0.0, 0.0, 1.0, 1.0),
        rectangle(1.0, 0.0, 2.0, 5e-9),
        rectangle(1.0, 0.5, 2.0, 0.5 + 5e-9),
    ];
    let error = PreparedGeometry::from_wkb(&rows)
        .unwrap()
        .rook_measurements()
        .unwrap_err();

    assert!(error
        .to_string()
        .contains("edge (0, 1) geometries have no shared boundary"));
}

#[test]
fn evaluator_rook_discovery_reports_overlap_before_boundary_errors() {
    let rows = [
        rectangle(0.0, 0.0, 2.0, 2.0),
        rectangle(1.0, 0.0, 3.0, 2.0),
        rectangle(2.0, 0.0, 3.0, 5e-9),
    ];
    let error = PreparedGeometry::from_wkb(&rows)
        .unwrap()
        .rook_measurements()
        .unwrap_err();

    assert!(error
        .to_string()
        .contains("edge (0, 1) geometries overlap by area"));
}

#[test]
fn evaluator_rook_results_and_errors_are_independent_of_rayon_threads() {
    let valid = [
        square(0.0, 0.0),
        square(1.0, 0.0),
        square(0.0, 1.0),
        square(1.0, 1.0),
    ];
    let invalid = [
        rectangle(0.0, 0.0, 1.0, 1.0),
        rectangle(1.0, 0.0, 2.0, 5e-9),
        rectangle(1.0, 0.5, 2.0, 0.5 + 5e-9),
    ];
    let run = |rows: &[Vec<u8>], threads| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap()
            .install(|| {
                PreparedGeometry::from_wkb(rows)
                    .and_then(|geometry| geometry.rook_measurements())
                    .map(|rook| rook.inputs())
                    .map_err(|error| error.to_string())
            })
    };

    assert_eq!(run(&valid, 1), run(&valid, 4));
    assert_eq!(run(&invalid, 1), run(&invalid, 4));
}

#[test]
fn prepared_geometry_row_errors_are_independent_of_rayon_threads() {
    let rows = [vec![0], square(0.0, 0.0), vec![0]];
    let run = |threads| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap()
            .install(|| PreparedGeometry::from_wkb(&rows).unwrap_err().to_string())
    };

    assert_eq!(run(1), run(4));
    assert!(run(1).contains("WKB row 0"));
}

#[test]
fn clipped_hull_preparation_checks_state_coverage_with_a_narrow_overlay_tolerance() {
    let rows = [square(0.0, 0.0)];
    let almost_covering = polygon_wkb(&[&[
        (0.0, 0.0),
        (1.0 - 5e-13, 0.0),
        (1.0 - 5e-13, 1.0),
        (0.0, 1.0),
        (0.0, 0.0),
    ]]);
    let metric = PreparedStateClippedConvexHullRatio::from_wkb(&rows, &almost_covering).unwrap();
    assert!((score(&metric.score(&[0]).unwrap())[0] - 1.0).abs() < 1e-12);

    let materially_short = polygon_wkb(&[&[
        (0.0, 0.0),
        (1.0 - 1e-6, 0.0),
        (1.0 - 1e-6, 1.0),
        (0.0, 1.0),
        (0.0, 0.0),
    ]]);
    assert!(matches!(
        PreparedStateClippedConvexHullRatio::from_wkb(&rows, &materially_short),
        Err(Error::StateGeometryCoverage { unit: 0, .. })
    ));

    // geo's default i32 overlay grid collapses this sliver even though its area is one hundred
    // times the promised tolerance. Coverage validation must use the higher-precision path.
    let above_tolerance_but_below_geo_resolution = polygon_wkb(&[&[
        (0.0, 0.0),
        (1.0 - 1e-10, 0.0),
        (1.0 - 1e-10, 1.0),
        (0.0, 1.0),
        (0.0, 0.0),
    ]]);
    assert!(matches!(
        PreparedStateClippedConvexHullRatio::from_wkb(
            &rows,
            &above_tolerance_but_below_geo_resolution,
        ),
        Err(Error::StateGeometryCoverage { unit: 0, .. })
    ));

    let protrusion_width = 2e-10;
    let protruding_unit = polygon_wkb(&[&[
        (0.0, 0.0),
        (1.0 + protrusion_width, 0.0),
        (1.0 + protrusion_width, 0.5),
        (1.0, 0.5),
        (1.0, 1.0),
        (0.0, 1.0),
        (0.0, 0.0),
    ]]);
    assert!(matches!(
        PreparedStateClippedConvexHullRatio::from_wkb(&[protruding_unit], &rows[0]),
        Err(Error::StateGeometryCoverage { unit: 0, .. })
    ));

    // The aggregate unary union must not erase an out-of-state protrusion before the precise
    // coverage check. A single-row case does not exercise that union behavior.
    let touching_rows = [
        rectangle(0.0, 0.0, 1.0, 1.0),
        rectangle(1.0, 0.0, 2.0 + protrusion_width, 1.0),
    ];
    let two_unit_state = rectangle(0.0, 0.0, 2.0, 1.0);
    assert!(matches!(
        PreparedStateClippedConvexHullRatio::from_wkb(&touching_rows, &two_unit_state),
        Err(Error::StateGeometryCoverage { unit: 1, .. })
    ));
}

#[test]
fn topology_rejects_material_overlap_below_the_default_overlay_grid() {
    let rows = [
        rectangle(0.0, 0.0, 1_000.0, 1_000.0),
        rectangle(1_000.0 - 1e-7, 0.0, 2_000.0, 1_000.0),
    ];
    let prepared = PreparedGeometry::from_wkb(&rows).unwrap();

    assert!(prepared
        .rook_measurements()
        .unwrap_err()
        .to_string()
        .contains("overlap by area"));
    assert!(PreparedPolsbyPopper::from_wkb(&rows, vec![(0, 1)])
        .unwrap_err()
        .to_string()
        .contains("overlap by area"));
}

#[test]
fn clipped_hull_preparation_rejects_units_inside_state_holes() {
    let state = polygon_wkb(&[
        &[(0.0, 0.0), (3.0, 0.0), (3.0, 3.0), (0.0, 3.0), (0.0, 0.0)],
        &[(1.0, 1.0), (1.0, 2.0), (2.0, 2.0), (2.0, 1.0), (1.0, 1.0)],
    ]);

    assert!(matches!(
        PreparedStateClippedConvexHullRatio::from_wkb(&[square(1.0, 1.0)], &state),
        Err(Error::StateGeometryCoverage { unit: 0, .. })
    ));
}

#[test]
fn clipped_hull_preparation_accepts_a_unit_with_the_same_hole_as_the_state() {
    let state = polygon_wkb(&[
        &[(0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (0.0, 4.0), (0.0, 0.0)],
        &[(1.0, 1.0), (1.0, 2.0), (2.0, 2.0), (2.0, 1.0), (1.0, 1.0)],
    ]);

    let metric =
        PreparedStateClippedConvexHullRatio::from_wkb(std::slice::from_ref(&state), &state)
            .unwrap();

    assert!((score(&metric.score(&[0]).unwrap())[0] - 1.0).abs() < 1e-12);
}

#[test]
fn clipped_hull_preparation_rejects_invalid_state_geometry() {
    let error =
        PreparedStateClippedConvexHullRatio::from_wkb(&[square(0.0, 0.0)], &polygon_wkb(&[]))
            .unwrap_err();

    assert!(error.to_string().contains("state geometry"));
}

#[test]
fn packed_unit_hulls_reconstruct_hulls_from_complete_complex_geometry() {
    let concave = polygon_wkb(&[&[
        (0.0, 0.0),
        (3.0, 0.0),
        (3.0, 1.0),
        (1.0, 1.0),
        (1.0, 3.0),
        (0.0, 3.0),
        (0.0, 0.0),
    ]]);
    let exterior = &[(4.0, 0.0), (8.0, 0.0), (8.0, 4.0), (4.0, 4.0), (4.0, 0.0)];
    let hole = &[(5.0, 1.0), (5.0, 2.0), (6.0, 2.0), (6.0, 1.0), (5.0, 1.0)];
    let rows = [
        concave,
        polygon_wkb(&[exterior, hole]),
        multipolygon_wkb(&[square(9.0, 0.0), square(11.0, 2.0)]),
    ];
    let geometries = rows
        .iter()
        .enumerate()
        .map(|(row, bytes)| decode_polygon(row, bytes))
        .collect::<Result<Vec<_>>>()
        .unwrap();
    let prepared = PreparedUnitHulls::from_wkb(&rows).unwrap();

    for nodes in [&[0][..], &[1], &[2], &[0, 1], &[0, 2], &[0, 1, 2]] {
        assert_same_hull(&geometries, &prepared, nodes);
    }
}

#[test]
fn generated_district_hulls_match_complete_unit_geometry() {
    let rows = (0..5)
        .flat_map(|row| (0..5).map(move |column| square(column as f64, row as f64)))
        .collect::<Vec<_>>();
    let geometries = rows
        .iter()
        .enumerate()
        .map(|(row, bytes)| decode_polygon(row, bytes))
        .collect::<Result<Vec<_>>>()
        .unwrap();
    let prepared = PreparedUnitHulls::from_wkb(&rows).unwrap();
    let mut seed = 0x9e37_79b9_7f4a_7c15_u64;

    for _ in 0..200 {
        let assignment = (0..rows.len())
            .map(|_| {
                seed = seed
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                ((seed >> 32) % 5) as usize
            })
            .collect::<Vec<_>>();
        for district in 0..5 {
            let nodes = assignment
                .iter()
                .enumerate()
                .filter_map(|(node, &label)| (label == district).then_some(node))
                .collect::<Vec<_>>();
            if !nodes.is_empty() {
                assert_same_hull(&geometries, &prepared, &nodes);
            }
        }
    }
}

#[test]
fn accepts_multipolygon_wkb() {
    let row = multipolygon_wkb(&[square(0.0, 0.0), square(2.0, 0.0)]);
    let reock = PreparedReock::from_wkb(std::slice::from_ref(&row)).unwrap();
    let convex_hull_ratio = PreparedConvexHullRatio::from_wkb(std::slice::from_ref(&row)).unwrap();
    let polsby = PreparedPolsbyPopper::from_wkb(std::slice::from_ref(&row), Vec::new()).unwrap();

    assert_eq!(reock.node_count(), 1);
    assert!((score(&convex_hull_ratio.score(&[0]).unwrap())[0] - 2.0 / 3.0).abs() < 1e-12);
    let result = polsby.score(&[0]).unwrap();
    assert!((score(&result)[0] - std::f64::consts::PI / 8.0).abs() < 1e-12);
}

#[test]
fn accepts_supported_byte_orders_dimensions_and_srid() {
    let encodings = [
        encoded_square(false, 3, 2),
        encoded_square(true, 1_003, 3),
        encoded_square(true, 2_003, 3),
        encoded_square(true, 3_003, 4),
        encoded_square(false, 0xe000_0003, 4),
    ];

    for row in encodings {
        let metric = PreparedPolsbyPopper::from_wkb(&[row], Vec::new()).unwrap();
        let result = metric.score(&[0]).unwrap();
        assert!((score(&result)[0] - std::f64::consts::PI / 4.0).abs() < 1e-12);
    }
}

#[test]
fn accepts_mixed_endian_nested_multipolygon_wkb() {
    let mut bytes = vec![0];
    bytes.extend_from_slice(&6_u32.to_be_bytes());
    bytes.extend_from_slice(&2_u32.to_be_bytes());
    bytes.extend_from_slice(&encoded_square(true, 3, 2));
    let mut right = encoded_square(false, 3, 2);
    for chunk in right[13..].chunks_exact_mut(16) {
        let x = f64::from_be_bytes(chunk[..8].try_into().unwrap()) + 2.0;
        chunk[..8].copy_from_slice(&x.to_be_bytes());
    }
    bytes.extend_from_slice(&right);

    let metric = PreparedPolsbyPopper::from_wkb(&[bytes], Vec::new()).unwrap();
    let result = metric.score(&[0]).unwrap();

    assert!((score(&result)[0] - std::f64::consts::PI / 8.0).abs() < 1e-12);
}

#[test]
fn shared_boundary_handles_different_segment_splits() {
    let left = square(0.0, 0.0);
    let right = polygon_wkb(&[&[
        (1.0, 0.0),
        (2.0, 0.0),
        (2.0, 1.0),
        (1.0, 1.0),
        (1.0, 0.5),
        (1.0, 0.0),
    ]]);
    let metric = PreparedPolsbyPopper::from_wkb(&[left, right], vec![(0, 1)]).unwrap();

    let combined = metric.score(&[0, 0]).unwrap();

    assert!((score(&combined)[0] - 2.0 * std::f64::consts::PI / 9.0).abs() < 1e-12);
}

#[test]
fn polsby_popper_perimeter_includes_holes() {
    let exterior = &[(0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (0.0, 4.0), (0.0, 0.0)];
    let hole = &[(1.0, 1.0), (1.0, 2.0), (2.0, 2.0), (2.0, 1.0), (1.0, 1.0)];
    let metric =
        PreparedPolsbyPopper::from_wkb(&[polygon_wkb(&[exterior, hole])], Vec::new()).unwrap();

    let result = metric.score(&[0]).unwrap();

    assert!((score(&result)[0] - 3.0 * std::f64::consts::PI / 20.0).abs() < 1e-12);
}

#[test]
fn rejects_a_zero_ring_polygon_unit_row() {
    let error = PreparedUnitHulls::from_wkb(&[polygon_wkb(&[])]).unwrap_err();
    assert!(matches!(error, Error::Geometry(_)), "got {error:?}");
}

#[test]
fn rejects_malformed_non_polygon_and_invalid_wkb() {
    let malformed = PreparedReock::from_wkb(&[vec![1, 2, 3]]).unwrap_err();
    assert!(malformed.to_string().contains("cannot be decoded"));

    let mut point = vec![1];
    point.extend_from_slice(&1_u32.to_le_bytes());
    point.extend_from_slice(&0.0_f64.to_le_bytes());
    point.extend_from_slice(&0.0_f64.to_le_bytes());
    let wrong_type = PreparedReock::from_wkb(&[point]).unwrap_err();
    assert!(wrong_type.to_string().contains("not a Polygon"));

    let bowtie = polygon_wkb(&[&[(0.0, 0.0), (1.0, 1.0), (0.0, 1.0), (1.0, 0.0), (0.0, 0.0)]]);
    let invalid = PreparedReock::from_wkb(&[bowtie]).unwrap_err();
    assert!(invalid.to_string().contains("is invalid"));
}

#[test]
fn rejects_wkb_counts_that_do_not_fit_before_the_decoder_can_allocate() {
    let prior_oom = [1, 0x65, 0x29, 0x2a, 1, 0x24, 0x16, 0xff, 0xff];
    assert!(PreparedReock::from_wkb(&[prior_oom]).is_err());

    let mut polygon = vec![1];
    polygon.extend_from_slice(&3_u32.to_le_bytes());
    polygon.extend_from_slice(&u32::MAX.to_le_bytes());
    assert!(PreparedReock::from_wkb(&[polygon]).is_err());

    let mut multipolygon = vec![1];
    multipolygon.extend_from_slice(&6_u32.to_le_bytes());
    multipolygon.extend_from_slice(&u32::MAX.to_le_bytes());
    assert!(PreparedPolsbyPopper::from_wkb(&[multipolygon], Vec::new()).is_err());
}

#[test]
fn rejects_trailing_wkb_bytes_and_underdeclared_multipolygon_counts() {
    let mut polygon = square(0.0, 0.0);
    polygon.extend_from_slice(b"trailing");
    assert!(PreparedReock::from_wkb(&[polygon]).is_err());

    let mut multipolygon = multipolygon_wkb(&[square(0.0, 0.0), square(2.0, 0.0)]);
    multipolygon[5..9].copy_from_slice(&1_u32.to_le_bytes());
    assert!(PreparedConvexHullRatio::from_wkb(&[multipolygon]).is_err());
}

#[test]
fn rejects_graph_edges_without_matching_geometry_boundaries() {
    let separated = [square(0.0, 0.0), square(2.0, 0.0)];
    let no_boundary = PreparedPolsbyPopper::from_wkb(&separated, vec![(0, 1)]).unwrap_err();
    assert!(no_boundary.to_string().contains("no shared boundary"));

    let overlapping = [square(0.0, 0.0), square(0.5, 0.0)];
    let overlap = PreparedPolsbyPopper::from_wkb(&overlapping, vec![(0, 1)]).unwrap_err();
    assert!(overlap.to_string().contains("overlap by area"));

    let out_of_range =
        PreparedPolsbyPopper::from_wkb(&[square(0.0, 0.0)], vec![(0, 1)]).unwrap_err();
    assert_eq!(
        out_of_range,
        Error::EdgeNodeOutOfRange {
            u: 0,
            v: 1,
            node_count: 1,
        }
    );
}

#[test]
fn geometry_tolerance_distinguishes_roundoff_from_real_gaps_and_overlaps() {
    let within = GEOMETRY_EPS / 2.0;
    let beyond = GEOMETRY_EPS * 2.0;

    assert!(PreparedPolsbyPopper::from_wkb(
        &[square(0.0, 0.0), square(1.0 + within, 0.0)],
        vec![(0, 1)],
    )
    .is_ok());
    assert!(PreparedPolsbyPopper::from_wkb(
        &[square(0.0, 0.0), square(1.0 - within, 0.0)],
        vec![(0, 1)],
    )
    .is_ok());

    let gap = PreparedPolsbyPopper::from_wkb(
        &[square(0.0, 0.0), square(1.0 + beyond, 0.0)],
        vec![(0, 1)],
    )
    .unwrap_err();
    assert!(gap.to_string().contains("no shared boundary"));
    let overlap = PreparedPolsbyPopper::from_wkb(
        &[square(0.0, 0.0), square(1.0 - beyond, 0.0)],
        vec![(0, 1)],
    )
    .unwrap_err();
    assert!(overlap.to_string().contains("overlap by area"));
}

#[test]
fn shared_boundary_tolerance_is_independent_of_segment_length() {
    let gap = GEOMETRY_EPS / 2.0;
    let tall_neighbors = [
        rectangle(0.0, 0.0, 1.0, 1_000.0),
        rectangle(1.0 + gap, 0.0, 2.0, 1_000.0),
    ];

    assert!(PreparedPolsbyPopper::from_wkb(&tall_neighbors, vec![(0, 1)]).is_ok());
}
