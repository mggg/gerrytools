use super::*;
use rstar::{Envelope, RTreeObject};

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

fn rectangle(x0: f64, y0: f64, x1: f64, y1: f64) -> Vec<u8> {
    polygon_wkb(&[&[(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)]])
}

fn values(table: &DistrictTable) -> &[f64] {
    table.column(0).unwrap()
}

#[test]
fn prepares_owner_totals_and_polygon_index() {
    let units = [rectangle(0.0, 0.0, 1.0, 1.0), rectangle(1.0, 0.0, 2.0, 1.0)];
    let population = [
        rectangle(1.1, 0.1, 1.9, 0.9),
        rectangle(0.5, 0.1, 0.9, 0.9),
        rectangle(0.1, 0.1, 0.4, 0.9),
    ];
    let metric = PreparedPopulationPolygon::from_wkb(
        &units,
        &population,
        vec![5.0, 2.0, 7.0],
        vec![1, 0, 0],
    )
    .unwrap();

    assert_eq!(metric.surface.owner_totals, [9.0, 5.0]);
    let envelope = AABB::from_corners([0.0, 0.0], [0.45, 1.0]);
    let indexed = metric
        .surface
        .polygons
        .locate_in_envelope_intersecting(&envelope)
        .map(|polygon| polygon.data.weight)
        .collect::<Vec<_>>();
    assert_eq!(indexed, [7.0]);
}

#[test]
fn aligned_surface_matches_the_checked_explicit_mapping() {
    let units = [rectangle(0.0, 0.0, 1.0, 1.0), rectangle(1.0, 0.0, 2.0, 1.0)];
    let aligned = PreparedPopulationPolygon::from_aligned_wkb(&units, vec![5.0, 7.0]).unwrap();
    let checked =
        PreparedPopulationPolygon::from_wkb(&units, &units, vec![5.0, 7.0], vec![0, 1]).unwrap();

    assert_eq!(aligned.surface.owner_totals, [5.0, 7.0]);
    assert_eq!(
        aligned.score(&[0, 1]).unwrap(),
        checked.score(&[0, 1]).unwrap()
    );
    assert_eq!(
        PreparedPopulationPolygon::from_aligned_wkb(&units, vec![5.0]).unwrap_err(),
        Error::PopulationObservationLength {
            geometries: 2,
            weights: 1,
            owners: 2,
        }
    );
}

#[test]
fn prepared_geometry_infers_unique_population_owners() {
    let units = [rectangle(0.0, 0.0, 1.0, 1.0), rectangle(1.0, 0.0, 2.0, 1.0)];
    let geometry = Arc::new(PreparedGeometry::from_wkb(&units).unwrap());
    let metric = PreparedPopulationPolygon::from_prepared_geometry(
        Arc::clone(&geometry),
        &[rectangle(1.1, 0.1, 1.9, 0.9), rectangle(0.1, 0.1, 0.9, 0.9)],
        vec![5.0, 7.0],
    )
    .unwrap();

    assert_eq!(metric.surface.owner_totals, [7.0, 5.0]);
    assert_eq!(values(&metric.score(&[0, 1]).unwrap()), [1.0, 1.0]);
}

#[test]
fn prepared_geometry_infers_ownership_from_coverage_not_a_representative_point() {
    let units = [rectangle(0.0, 0.0, 1.0, 1.0)];
    let geometry = Arc::new(PreparedGeometry::from_wkb(&units).unwrap());
    // Half of this 1e-12-area polygon is a tolerated overlay residue. Its center lies exactly on
    // the unit boundary, so representative-point containment would reject it.
    let population = [rectangle(1.0 - 5e-7, 0.5, 1.0 + 5e-7, 0.5 + 1e-6)];

    let metric =
        PreparedPopulationPolygon::from_prepared_geometry(geometry, &population, vec![3.0])
            .unwrap();

    assert_eq!(metric.surface.owner_totals, [3.0]);
}

#[test]
fn prepared_geometry_owner_inference_rejects_a_material_boundary_crossing() {
    let units = [rectangle(0.0, 0.0, 1.0, 1.0)];
    let geometry = Arc::new(PreparedGeometry::from_wkb(&units).unwrap());
    let population = [rectangle(0.5, 0.1, 1.5, 0.9)];

    assert_eq!(
        PreparedPopulationPolygon::from_prepared_geometry(geometry, &population, vec![1.0])
            .unwrap_err(),
        Error::PopulationGeometryOwnerCount {
            observation: 0,
            count: 0,
        }
    );
}

#[test]
fn prepared_geometry_owner_inference_resolves_thin_holes_above_tolerance() {
    let hole_width = 1e-10;
    let units = [polygon_wkb(&[
        &[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0), (0.0, 0.0)],
        &[
            (0.5 - hole_width / 2.0, 0.45),
            (0.5 - hole_width / 2.0, 0.55),
            (0.5 + hole_width / 2.0, 0.55),
            (0.5 + hole_width / 2.0, 0.45),
            (0.5 - hole_width / 2.0, 0.45),
        ],
    ])];
    let geometry = Arc::new(PreparedGeometry::from_wkb(&units).unwrap());
    let population = [rectangle(0.4, 0.4, 0.6, 0.6)];

    assert_eq!(
        PreparedPopulationPolygon::from_prepared_geometry(geometry, &population, vec![1.0])
            .unwrap_err(),
        Error::PopulationGeometryOwnerCount {
            observation: 0,
            count: 0,
        }
    );
}

#[test]
fn prepared_geometry_owner_inference_reports_the_lowest_nonunique_observation() {
    let duplicate_units = [rectangle(0.0, 0.0, 1.0, 1.0), rectangle(0.0, 0.0, 1.0, 1.0)];
    let geometry = Arc::new(PreparedGeometry::from_wkb(&duplicate_units).unwrap());
    let population = [rectangle(0.1, 0.1, 0.9, 0.9), rectangle(2.0, 2.0, 3.0, 3.0)];

    assert_eq!(
        PreparedPopulationPolygon::from_prepared_geometry(geometry, &population, vec![1.0, 1.0],)
            .unwrap_err(),
        Error::PopulationGeometryOwnerCount {
            observation: 0,
            count: 2,
        }
    );
}

#[test]
fn aligned_prepared_surface_shares_graph_geometry() {
    let units = [rectangle(0.0, 0.0, 1.0, 1.0), rectangle(1.0, 0.0, 2.0, 1.0)];
    let geometry = Arc::new(PreparedGeometry::from_wkb(&units).unwrap());
    let unit_geometries = geometry.geometries();
    let metric =
        PreparedPopulationPolygon::from_aligned_prepared_geometry(geometry, vec![5.0, 7.0])
            .unwrap();

    assert!(Arc::ptr_eq(&unit_geometries, &metric.surface.geometries));
}

#[test]
fn owner_validation_accepts_boundaries_and_rejects_exteriors_and_holes() {
    let units = [rectangle(0.0, 0.0, 1.0, 1.0), rectangle(1.0, 0.0, 2.0, 1.0)];
    PreparedPopulationPolygon::from_wkb(
        &units,
        &[
            rectangle(0.0, 0.25, 1.0, 0.75),
            rectangle(1.1, 0.1, 1.9, 0.9),
        ],
        vec![2.0, 3.0],
        vec![0, 1],
    )
    .unwrap();

    assert!(matches!(
        PreparedPopulationPolygon::from_wkb(
            &units,
            &[rectangle(1.1, 0.1, 1.9, 0.9)],
            vec![1.0],
            vec![0],
        ),
        Err(Error::PopulationGeometryOutsideOwner {
            observation: 0,
            owner: 0,
            ..
        })
    ));

    let with_hole = polygon_wkb(&[
        &[(0.0, 0.0), (3.0, 0.0), (3.0, 3.0), (0.0, 3.0), (0.0, 0.0)],
        &[(1.0, 1.0), (1.0, 2.0), (2.0, 2.0), (2.0, 1.0), (1.0, 1.0)],
    ]);
    assert!(matches!(
        PreparedPopulationPolygon::from_wkb(
            &[with_hole],
            &[rectangle(1.2, 1.2, 1.8, 1.8)],
            vec![1.0],
            vec![0],
        ),
        Err(Error::PopulationGeometryOutsideOwner { .. })
    ));
}

#[test]
fn validates_population_arrays_geometry_and_aggregated_totals() {
    let units = [rectangle(0.0, 0.0, 1.0, 1.0)];
    let population = [rectangle(0.1, 0.1, 0.9, 0.9)];
    assert_eq!(
        PreparedPopulationPolygon::from_wkb(&units, &population, vec![], vec![0]).unwrap_err(),
        Error::PopulationObservationLength {
            geometries: 1,
            weights: 0,
            owners: 1,
        }
    );
    assert_eq!(
        PreparedPopulationPolygon::from_wkb(&units, &Vec::<Vec<u8>>::new(), vec![], vec![],)
            .unwrap_err(),
        Error::EmptyPopulationSurface
    );
    assert!(matches!(
        PreparedPopulationPolygon::from_wkb(
            &units,
            &[rectangle(0.5, 0.1, 0.5, 0.9)],
            vec![1.0],
            vec![0],
        ),
        Err(Error::Geometry(_)) | Err(Error::InvalidPopulationGeometryArea { .. })
    ));
    assert!(matches!(
        PreparedPopulationPolygon::from_wkb(&units, &population, vec![-1.0], vec![0]),
        Err(Error::InvalidPopulationWeight { observation: 0, .. })
    ));
    assert_eq!(
        PreparedPopulationPolygon::from_wkb(&units, &population, vec![1.0], vec![1]).unwrap_err(),
        Error::PopulationOwnerOutOfRange {
            observation: 0,
            owner: 1,
            node_count: 1,
        }
    );
    assert_eq!(
        PreparedPopulationPolygon::from_wkb(&units, &population, vec![0.0], vec![0]).unwrap_err(),
        Error::NoPositivePopulation
    );
    assert_eq!(
        PreparedPopulationPolygon::from_wkb(
            &units,
            &[population[0].clone(), population[0].clone()],
            vec![f64::MAX, f64::MAX],
            vec![0, 0],
        )
        .unwrap_err(),
        Error::NonFinitePopulationOwnerTotal { owner: 0 }
    );
    assert_eq!(
        PreparedPopulationPolygon::from_wkb(
            &[rectangle(0.0, 0.0, 1.0, 1.0), rectangle(1.0, 0.0, 2.0, 1.0)],
            &[rectangle(0.1, 0.1, 0.9, 0.9), rectangle(1.1, 0.1, 1.9, 0.9),],
            vec![f64::MAX, f64::MAX],
            vec![0, 1],
        )
        .unwrap_err(),
        Error::NonFinitePopulationTotal
    );
}

#[test]
fn scores_disconnected_districts_and_full_weight_boundary_intersections() {
    let units = [
        rectangle(0.0, 0.0, 1.0, 1.0),
        rectangle(1.0, 0.0, 2.0, 1.0),
        rectangle(2.0, 0.0, 3.0, 1.0),
    ];
    let metric = PreparedPopulationPolygon::from_wkb(
        &units,
        &[
            rectangle(0.1, 0.1, 0.9, 0.9),
            rectangle(1.1, 0.1, 1.9, 0.9),
            rectangle(2.1, 0.1, 2.9, 0.9),
        ],
        vec![10.0, 20.0, 30.0],
        vec![0, 1, 2],
    )
    .unwrap();
    let table = metric.score(&[0, 1, 0]).unwrap();
    assert_eq!(table.district_ids(), [0, 1]);
    assert!((values(&table)[0] - 2.0 / 3.0).abs() < 1e-12);
    assert_eq!(values(&table)[1], 1.0);

    let boundary_metric = PreparedPopulationPolygon::from_wkb(
        &units[..2],
        &[
            rectangle(0.5, 0.25, 1.0, 0.75),
            rectangle(1.0, 0.25, 1.5, 0.75),
        ],
        vec![5.0, 7.0],
        vec![0, 1],
    )
    .unwrap();
    let boundary = boundary_metric.score(&[0, 1]).unwrap();
    assert!((values(&boundary)[0] - 5.0 / 12.0).abs() < 1e-12);
    assert!((values(&boundary)[1] - 7.0 / 12.0).abs() < 1e-12);
}

#[test]
fn rejects_zero_population_districts() {
    let units = [rectangle(0.0, 0.0, 1.0, 1.0), rectangle(1.0, 0.0, 2.0, 1.0)];
    let metric = PreparedPopulationPolygon::from_wkb(
        &units,
        &[rectangle(0.1, 0.1, 0.9, 0.9), rectangle(1.1, 0.1, 1.9, 0.9)],
        vec![1.0, 0.0],
        vec![0, 1],
    )
    .unwrap();
    assert_eq!(
        metric.score(&[0, 1]).unwrap_err(),
        Error::InvalidDistrictPopulation {
            kind: "owned",
            district: 1,
            population: 0.0,
        }
    );
}

#[test]
fn incremental_zero_population_error_does_not_mutate_state() {
    let units = [rectangle(0.0, 0.0, 1.0, 1.0), rectangle(1.0, 0.0, 2.0, 1.0)];
    let metric = PreparedPopulationPolygon::from_wkb(
        &units,
        &[rectangle(0.1, 0.1, 0.9, 0.9), rectangle(1.1, 0.1, 1.9, 0.9)],
        vec![1.0, 0.0],
        vec![0, 1],
    )
    .unwrap();
    let assignment = [0, 0];
    let mut state = metric.incremental(&assignment).unwrap();
    let before = state.result();

    assert_eq!(
        state.update(&[DeltaChange {
            node: 0,
            old: 0,
            new: 1,
        }]),
        Err(Error::InvalidDistrictPopulation {
            kind: "owned",
            district: 0,
            population: 0.0,
        })
    );
    assert_eq!(state.result(), before);
    assert_eq!(
        state.reset(&[0, 1]),
        Err(Error::InvalidDistrictPopulation {
            kind: "owned",
            district: 1,
            population: 0.0,
        })
    );
    assert_eq!(state.result(), before);
}

#[test]
fn generated_incremental_updates_match_fresh_scores() {
    let units = (0..16)
        .map(|node| {
            let x = (node % 4) as f64;
            let y = (node / 4) as f64;
            rectangle(x, y, x + 1.0, y + 1.0)
        })
        .collect::<Vec<_>>();
    let population = (0..16)
        .flat_map(|node| {
            let x = (node % 4) as f64;
            let y = (node / 4) as f64;
            [
                rectangle(x + 0.1, y + 0.1, x + 0.4, y + 0.9),
                rectangle(x + 0.6, y + 0.1, x + 0.9, y + 0.9),
            ]
        })
        .collect::<Vec<_>>();
    let weights = (0..32)
        .map(|index| (index % 11 + 1) as f64)
        .collect::<Vec<_>>();
    let owners = (0..16).flat_map(|node| [node, node]).collect::<Vec<_>>();
    let metric = PreparedPopulationPolygon::from_wkb(&units, &population, weights, owners).unwrap();
    let mut assignment = (0..16).map(|node| node as u16 % 4).collect::<Vec<_>>();
    let mut state = metric.incremental(&assignment).unwrap();
    let mut seed = 0x51a7_ec11_u64;

    for _ in 0..300 {
        seed = seed
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let node = seed as usize % assignment.len();
        let old = assignment[node];
        let new = (old + 1 + ((seed >> 16) % 3) as u16) % 4;
        let change = DeltaChange { node, old, new };
        state.update(&[change]).unwrap();
        assignment[node] = new;

        let incremental = state.result();
        let fresh = metric.score(&assignment).unwrap();
        assert_eq!(incremental.district_ids(), fresh.district_ids());
        for (actual, expected) in values(&incremental).iter().zip(values(&fresh)) {
            assert!((actual - expected).abs() < 1e-12);
        }
    }

    let before = state.result();
    let invalid = DeltaChange {
        node: 0,
        old: assignment[0] + 1,
        new: assignment[0],
    };
    assert!(matches!(
        state.update(&[invalid]),
        Err(Error::DeltaOldLabelMismatch { .. })
    ));
    assert_eq!(state.result(), before);
}

#[test]
fn generated_rtree_queries_match_direct_envelope_scans() {
    let units = [rectangle(-10.0, -10.0, 10.0, 10.0)];
    let mut seed = 0x5eed_51a7_u64;
    let population = (0..500)
        .map(|_| {
            seed = seed
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let x = ((seed >> 16) as u32 as f64 / u32::MAX as f64) * 18.0 - 9.0;
            seed = seed
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let y = ((seed >> 16) as u32 as f64 / u32::MAX as f64) * 18.0 - 9.0;
            rectangle(x, y, x + 0.1, y + 0.1)
        })
        .collect::<Vec<_>>();
    let metric =
        PreparedPopulationPolygon::from_wkb(&units, &population, vec![1.0; 500], vec![0; 500])
            .unwrap();

    for query in 0..200 {
        let lower = [
            -10.0 + query as f64 * 0.05,
            -8.0 + (query % 37) as f64 * 0.1,
        ];
        let upper = [lower[0] + 3.5, lower[1] + 2.25];
        let envelope = AABB::from_corners(lower, upper);
        let mut indexed = metric
            .surface
            .polygons
            .locate_in_envelope_intersecting(&envelope)
            .map(|polygon| polygon.data.order)
            .collect::<Vec<_>>();
        indexed.sort_unstable();
        let mut direct = metric
            .surface
            .polygons
            .iter()
            .filter_map(|polygon| {
                polygon
                    .envelope()
                    .intersects(&envelope)
                    .then_some(polygon.data.order)
            })
            .collect::<Vec<_>>();
        direct.sort_unstable();
        assert_eq!(indexed, direct);
    }
}
