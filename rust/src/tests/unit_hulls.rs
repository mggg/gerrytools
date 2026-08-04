use super::*;
use crate::{PreparedConvexHullRatio, PreparedReock};
use std::sync::Arc;

fn point(x: f64, y: f64) -> Coordinate {
    Coordinate { x, y }
}

fn square(x: f64) -> UnitHull {
    UnitHull::new(
        1.0,
        vec![
            point(x, 0.0),
            point(x + 1.0, 0.0),
            point(x + 1.0, 1.0),
            point(x, 1.0),
        ],
    )
}

#[test]
fn packs_ordered_areas_and_hull_points() {
    let prepared = PreparedUnitHulls::new(vec![square(0.0), square(5.0)]).unwrap();

    assert_eq!(prepared.node_count(), 2);
    assert_eq!(prepared.areas(), &[1.0, 1.0]);
    assert_eq!(prepared.hull_offsets(), &[0, 4, 8]);
    assert_eq!(prepared.hull_points().len(), 8);
    assert_eq!(prepared.unit_area(0), 1.0);
    assert_eq!(prepared.unit_area(1), 1.0);
    assert_eq!(
        prepared.unit_hull_points(0),
        &[
            point(0.0, 0.0),
            point(1.0, 0.0),
            point(1.0, 1.0),
            point(0.0, 1.0),
        ]
    );
    assert_eq!(
        prepared.unit_hull_points(1),
        &[
            point(5.0, 0.0),
            point(6.0, 0.0),
            point(6.0, 1.0),
            point(5.0, 1.0),
        ]
    );
}

#[test]
fn validates_every_unit_before_exposing_packed_data() {
    assert_eq!(
        PreparedUnitHulls::new(vec![UnitHull::new(0.0, square(0.0).points)]).unwrap_err(),
        Error::InvalidGeometryArea { unit: 0, area: 0.0 }
    );
    assert_eq!(
        PreparedUnitHulls::new(vec![UnitHull::new(1.0, vec![point(0.0, 0.0)])]).unwrap_err(),
        Error::InvalidGeometryPointCount { unit: 0, actual: 1 }
    );
    assert_eq!(
        PreparedUnitHulls::new(vec![UnitHull::new(
            1.0,
            vec![point(0.0, 0.0), point(1.0, f64::NAN), point(1.0, 1.0),],
        )])
        .unwrap_err(),
        Error::NonFiniteGeometryPoint { unit: 0, point: 1 }
    );
}

fn square_wkb(x: f64) -> Vec<u8> {
    let ring: [(f64, f64); 5] = [(x, 0.0), (x + 1.0, 0.0), (x + 1.0, 1.0), (x, 1.0), (x, 0.0)];
    let mut bytes = vec![1];
    bytes.extend_from_slice(&3_u32.to_le_bytes());
    bytes.extend_from_slice(&1_u32.to_le_bytes());
    bytes.extend_from_slice(&(ring.len() as u32).to_le_bytes());
    for (x, y) in ring {
        bytes.extend_from_slice(&x.to_le_bytes());
        bytes.extend_from_slice(&y.to_le_bytes());
    }
    bytes
}

#[test]
fn cache_reuses_identical_rows_without_redecoding() {
    let rows = [square_wkb(0.0), square_wkb(2.0)];
    let mut cache = UnitHullCache::new();

    let first = cache.get_or_decode(&rows).unwrap();
    let second = cache.get_or_decode(&rows).unwrap();

    assert!(Arc::ptr_eq(&first, &second));
}

#[test]
fn warm_cache_rejects_rows_that_differ_from_the_registered_geometry() {
    let rows = [square_wkb(0.0), square_wkb(2.0)];
    let mut cache = UnitHullCache::new();
    let hulls = cache.get_or_decode(&rows).unwrap();

    let expected =
        Error::InvalidInput("geometry rows differ from the previously registered geometry".into());
    assert_eq!(
        cache.get_or_decode(&[b"garbage".to_vec()]).unwrap_err(),
        expected
    );
    // Same row count and bytes, different order.
    assert_eq!(
        cache
            .get_or_decode(&[square_wkb(2.0), square_wkb(0.0)])
            .unwrap_err(),
        expected
    );
    // The original geometry is still served after a rejected mismatch.
    assert!(Arc::ptr_eq(&hulls, &cache.get_or_decode(&rows).unwrap()));
}

#[test]
fn geometry_metrics_reuse_the_same_prepared_resource() {
    let unit_hulls = Arc::new(PreparedUnitHulls::new(vec![square(0.0)]).unwrap());
    let metric = PreparedReock::from_unit_hulls(Arc::clone(&unit_hulls));
    let convex_hull_ratio = PreparedConvexHullRatio::from_unit_hulls(Arc::clone(&unit_hulls));

    assert!(Arc::ptr_eq(&unit_hulls, &metric.unit_hulls()));
    assert!(Arc::ptr_eq(&unit_hulls, &convex_hull_ratio.unit_hulls()));
}
