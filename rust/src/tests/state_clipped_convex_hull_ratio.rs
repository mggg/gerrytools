use super::*;
use crate::{Coordinate, PreparedUnitHulls, UnitHull};
use geo::algorithm::convex_hull::quick_hull;
use geo::{polygon, MultiPolygon};

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

fn rectangle_state(width: f64, height: f64) -> MultiPolygon<f64> {
    MultiPolygon(vec![polygon![
        (x: 0.0, y: 0.0),
        (x: width, y: 0.0),
        (x: width, y: height),
        (x: 0.0, y: height),
        (x: 0.0, y: 0.0),
    ]])
}

fn metric(units: Vec<UnitHull>, state: MultiPolygon<f64>) -> PreparedStateClippedConvexHullRatio {
    PreparedStateClippedConvexHullRatio::from_validated_parts(
        Arc::new(PreparedUnitHulls::new(units).unwrap()),
        state,
    )
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
fn clipping_removes_hull_area_outside_a_concave_state() {
    let state = MultiPolygon(vec![polygon![
        (x: 0.0, y: 0.0),
        (x: 2.0, y: 0.0),
        (x: 2.0, y: 1.0),
        (x: 1.0, y: 1.0),
        (x: 1.0, y: 2.0),
        (x: 0.0, y: 2.0),
        (x: 0.0, y: 0.0),
    ]]);
    let metric = metric(
        vec![square(0.0, 0.0), square(1.0, 0.0), square(0.0, 1.0)],
        state,
    );

    let result = metric.score(&[0, 0, 0]).unwrap();

    assert_close(result.column(0).unwrap()[0], 1.0);
}

#[test]
fn clipping_handles_a_state_with_disconnected_islands() {
    let state = MultiPolygon(vec![
        rectangle_state(1.0, 1.0).0.remove(0),
        polygon![
            (x: 2.0, y: 0.0),
            (x: 3.0, y: 0.0),
            (x: 3.0, y: 1.0),
            (x: 2.0, y: 1.0),
            (x: 2.0, y: 0.0),
        ],
    ]);
    let metric = metric(vec![square(0.0, 0.0), square(2.0, 0.0)], state);

    let result = metric.score(&[0, 0]).unwrap();

    assert_close(result.column(0).unwrap()[0], 1.0);
}

#[test]
fn contained_fractional_hull_at_projected_scale_is_exact() {
    let points = vec![
        point(514_642.841_378_144_75, 248_447.520_039_250_46),
        point(522_638.251_949_483_8, 248_447.520_039_250_46),
        point(522_638.251_949_483_8, 260_687.501_433_289_7),
        point(514_642.841_378_144_75, 260_687.501_433_289_7),
    ];
    let mut hull_points = points
        .iter()
        .map(|point| Coord {
            x: point.x,
            y: point.y,
        })
        .collect::<Vec<_>>();
    let area = Polygon::new(quick_hull(&mut hull_points), Vec::new()).unsigned_area();
    let metric = metric(
        vec![UnitHull::new(area, points)],
        rectangle_state(700_000.0, 500_000.0),
    );

    let result = metric.score(&[0]).unwrap();

    assert_eq!(result.column(0).unwrap()[0], 1.0);
}

#[test]
fn clipped_fractional_hull_at_projected_scale_is_exact() {
    let district = vec![
        point(298_611.135_271, 198_734.627_119),
        point(304_998.741_893, 198_734.627_119),
        point(304_998.741_893, 205_882.519_337),
        point(298_611.135_271, 205_882.519_337),
    ];
    let hole = polygon![
        (x: 300_219.428_751, y: 200_127.893_441),
        (x: 302_881.735_919, y: 200_127.893_441),
        (x: 302_881.735_919, y: 203_994.182_673),
        (x: 300_219.428_751, y: 203_994.182_673),
        (x: 300_219.428_751, y: 200_127.893_441),
    ]
    .exterior()
    .clone();
    let state = MultiPolygon(vec![Polygon::new(
        polygon![
            (x: 0.183_719, y: 0.294_113),
            (x: 700_000.781_337, y: 0.294_113),
            (x: 700_000.781_337, y: 500_000.619_871),
            (x: 0.183_719, y: 500_000.619_871),
            (x: 0.183_719, y: 0.294_113),
        ]
        .exterior()
        .clone(),
        vec![hole.clone()],
    )]);
    let mut hull_points = district
        .iter()
        .map(|point| Coord {
            x: point.x,
            y: point.y,
        })
        .collect::<Vec<_>>();
    let district_polygon = Polygon::new(quick_hull(&mut hull_points), vec![hole]);
    let metric = metric(
        vec![UnitHull::new(district_polygon.unsigned_area(), district)],
        state,
    );

    let result = metric.score(&[0]).unwrap();

    assert_close(result.column(0).unwrap()[0], 1.0);
}

#[test]
fn incremental_updates_match_fresh_scores_across_generated_moves() {
    let units = (0..5)
        .flat_map(|row| (0..6).map(move |column| square(column as f64, row as f64)))
        .collect();
    let metric = metric(units, rectangle_state(6.0, 5.0));
    let mut assignment = (0..metric.node_count())
        .map(|node| (node % 4) as u16)
        .collect::<Vec<_>>();
    let mut incremental = metric.incremental(&assignment).unwrap();
    let mut seed = 0x2d79_5eed_cafe_u64;

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
fn incremental_update_matches_fresh_score_when_state_clips_hulls() {
    let outer = rectangle_state(3.0, 3.0).0.remove(0).exterior().clone();
    let hole = polygon![
        (x: 1.0, y: 1.0),
        (x: 2.0, y: 1.0),
        (x: 2.0, y: 2.0),
        (x: 1.0, y: 2.0),
        (x: 1.0, y: 1.0),
    ]
    .exterior()
    .clone();
    let state = MultiPolygon(vec![Polygon::new(outer, vec![hole])]);
    let mut units = Vec::new();
    let mut assignment = Vec::new();
    for row in 0..3 {
        for column in 0..3 {
            if row == 1 && column == 1 {
                continue;
            }
            units.push(square(column as f64, row as f64));
            assignment.push(((row + column) % 2) as u16);
        }
    }
    let metric = metric(units, state);
    let mut incremental = metric.incremental(&assignment).unwrap();
    assert!(incremental.result().column(0).unwrap()[0] < 1.0);

    let change = DeltaChange {
        node: 0,
        old: assignment[0],
        new: 1,
    };
    incremental.update(&[change]).unwrap();
    assignment[0] = 1;

    let expected = metric.score(&assignment).unwrap();
    assert!(expected.column(0).unwrap().iter().any(|&score| score < 1.0));
    assert_same_scores(&incremental.result(), &expected);
}

#[test]
fn clipped_area_does_not_exceed_the_hull_at_large_coordinate_offsets() {
    let district_area = 0.003_903_208_804_775_862_3;
    let hull = vec![
        point(0.138_995_768_630_928_08, 19_999_999.948_839_653),
        point(-0.004_553_585_089_939_228_6, 19_999_999.973_594_55),
        point(0.0, 20_000_000.0),
        point(0.044_068_373_789_610_59, 19_999_999.992_400_467),
        point(0.143_549_353_720_867_3, 19_999_999.975_245_103),
    ];
    let state = MultiPolygon(vec![polygon![
        (x: 0.007_190_840_897_676_1, y: 19_999_999.778_554_045),
        (x: -0.011_264_271_050_327_6, y: 19_999_999.671_536_05),
        (x: -0.055_332_644_839_938_2, y: 19_999_999.679_135_583),
        (x: -0.036_877_532_891_934_5, y: 19_999_999.786_153_577),
        (x: -0.020_754_989_437_395_3, y: 19_999_999.879_645_415),
        (x: -0.013_256_549_196_895_2, y: 19_999_999.923_127_57),
        (x: -0.004_553_585_089_939_2, y: 19_999_999.973_594_55),
        (x: 0.0, y: 20_000_000.0),
        (x: 0.044_068_373_789_610_6, y: 19_999_999.992_400_467),
        (x: 0.107_691_438_019_239_7, y: 19_999_999.981_428_754),
        (x: 0.143_549_353_720_867_3, y: 19_999_999.975_245_103),
        (x: 0.187_434_110_935_623, y: 19_999_999.967_677_236),
        (x: 0.211_948_544_905_147_1, y: 19_999_999.963_449_754),
        (x: 0.207_394_959_815_207_9, y: 19_999_999.937_044_304),
        (x: 0.198_691_995_708_252, y: 19_999_999.886_577_323),
        (x: 0.191_193_555_467_751_8, y: 19_999_999.843_095_17),
        (x: 0.175_071_012_013_212_6, y: 19_999_999.749_603_33),
        (x: 0.156_615_900_065_208_9, y: 19_999_999.642_585_337),
        (x: 0.132_101_466_095_684_7, y: 19_999_999.646_812_82),
        (x: 0.088_216_708_880_929_1, y: 19_999_999.654_380_687),
        (x: 0.052_358_793_179_301_5, y: 19_999_999.660_564_337),
        (x: 0.070_813_905_127_305_2, y: 19_999_999.767_582_335),
        (x: 0.007_190_840_897_676_1, y: 19_999_999.778_554_045),
    ]]);
    let hull_polygon = Polygon::new(
        hull.iter()
            .map(|point| Coord {
                x: point.x,
                y: point.y,
            })
            .collect(),
        Vec::new(),
    );
    let minimum_score = district_area / hull_polygon.unsigned_area();

    let score = metric(vec![UnitHull::new(district_area, hull)], state)
        .score(&[0])
        .unwrap()
        .column(0)
        .unwrap()[0];

    assert!(score + 1e-12 >= minimum_score);
}

#[test]
fn rejects_impossible_score_and_invalid_assignment() {
    let impossible = metric(
        vec![UnitHull::new(2.0, square_points(0.0, 0.0))],
        rectangle_state(1.0, 1.0),
    );
    assert!(matches!(
        impossible.score(&[4]),
        Err(Error::ImpossibleScore {
            metric: "state-clipped convex-hull ratio",
            district: 4,
            ..
        })
    ));

    let metric = metric(vec![square(0.0, 0.0)], rectangle_state(1.0, 1.0));
    assert_eq!(
        metric.score(&[]).unwrap_err(),
        Error::AssignmentLength {
            actual: 0,
            expected: 1,
        }
    );
}
