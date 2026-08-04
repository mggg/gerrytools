import math

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import MultiPolygon, Point, Polygon, box
from shapely.ops import unary_union

from gerrytools.scoring import (
    ConvexHullRatio,
    CutEdges,
    PartisanBias,
    PlanEvaluator,
    PolsbyPopper,
    PopulationPolygon,
    RegionParts,
    RegionPieces,
    RegionSplits,
    Reock,
    StateClippedConvexHullRatio,
    Tally,
    TallyByRegion,
)
from tests._compactness_oracles import (
    convex_hull_scores,
    population_polygon_scores,
    reock_scores,
    state_clipped_convex_hull_scores,
)


def as_series(value: object) -> pd.Series:
    assert isinstance(value, pd.Series)
    return value


def as_frame(value: object) -> pd.DataFrame:
    assert isinstance(value, pd.DataFrame)
    return value


def grid_resources() -> tuple[nx.Graph, gpd.GeoDataFrame]:
    graph = nx.Graph()
    graph.add_node(0, population=10, area=1, boundary_perim=2, COUNTY="A", MUNI="L")
    graph.add_node(1, population=20, area=1, boundary_perim=2, COUNTY="A", MUNI="R")
    graph.add_node(2, population=30, area=1, boundary_perim=2, COUNTY="B", MUNI="L")
    graph.add_node(3, population=40, area=1, boundary_perim=2, COUNTY="B", MUNI="R")
    graph.add_edge(0, 1, shared_perim=1, cut_weight=10)
    graph.add_edge(0, 2, shared_perim=1, cut_weight=2)
    graph.add_edge(1, 3, shared_perim=1, cut_weight=4)
    graph.add_edge(2, 3, shared_perim=1, cut_weight=20)

    geometry = gpd.GeoDataFrame(
        {
            "population": [10, 20, 30, 40],
            "COUNTY": ["A", "A", "B", "B"],
            "MUNI": ["L", "R", "L", "R"],
        },
        geometry=[
            box(0, 1, 1, 2),
            box(1, 1, 2, 2),
            box(0, 0, 1, 1),
            box(1, 0, 2, 1),
        ],
        crs="EPSG:3857",
    )
    return graph, geometry


def population_frame(geometries, weights) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        {"population": weights},
        geometry=geometries,
        crs="EPSG:3857",
    )


def population_metric(geometries, weights) -> PopulationPolygon:
    return PopulationPolygon(
        "population",
        population_units=population_frame(geometries, weights),
    )


def test_all_native_base_metrics_share_one_plan_evaluator_result() -> None:
    graph, geometry = grid_resources()
    scorer = PlanEvaluator(graph, geometry=geometry)
    scorer.add_metric(Tally("population"))
    scorer.add_metric(PopulationPolygon("population"))
    scorer.add_metric(Reock())
    scorer.add_metric(ConvexHullRatio())
    scorer.add_metric(StateClippedConvexHullRatio(box(0, 0, 2, 2)))
    scorer.add_metric(PolsbyPopper(area_attr="area", result_name="polsby_graph"))
    scorer.add_metric(PolsbyPopper(result_name="polsby_geometry"))
    scorer.add_metric(CutEdges())
    scorer.add_metric(CutEdges(weight_attr="cut_weight", result_name="weighted_cut_edges"))
    scorer.add_metric(RegionSplits("COUNTY", "MUNI"))
    scorer.add_metric(RegionPieces("COUNTY", "MUNI"))
    scorer.add_metric(RegionParts("COUNTY", "MUNI"))
    scorer.add_metric(
        TallyByRegion(
            "COUNTY",
            {"population": "population"},
            include_count=True,
            result_name="county_totals",
        ),
    )

    result = scorer.evaluate(["north", "north", "south", "south"])

    assert result.metrics == (
        "population",
        "population_polygon",
        "reock",
        "convex_hull_ratio",
        "state_clipped_convex_hull_ratio",
        "polsby_graph",
        "polsby_geometry",
        "cut_edges",
        "weighted_cut_edges",
        "region_splits",
        "region_pieces",
        "region_parts",
        "county_totals",
    )

    np.testing.assert_allclose(result["population"], [30, 70])
    np.testing.assert_allclose(result["population_polygon"], [0.3, 0.7])
    np.testing.assert_allclose(result["reock"], [1.6 / math.pi] * 2)
    np.testing.assert_allclose(result["convex_hull_ratio"], [1, 1])
    np.testing.assert_allclose(
        result["state_clipped_convex_hull_ratio"],
        [1, 1],
    )
    expected_polsby_popper = [2 * math.pi / 9] * 2
    np.testing.assert_allclose(result["polsby_graph"], expected_polsby_popper)
    np.testing.assert_allclose(result["polsby_geometry"], expected_polsby_popper)

    county_totals = as_frame(result["county_totals"])
    assert county_totals.index.equals(pd.Index(["A", "B"], name="COUNTY"))
    assert county_totals.columns.equals(
        pd.MultiIndex.from_product(
            (["count", "population"], ["north", "south"]),
            names=("metric", "district"),
        )
    )
    assert county_totals.dtypes.tolist() == [
        np.dtype("int64"),
        np.dtype("int64"),
        np.dtype("float64"),
        np.dtype("float64"),
    ]
    np.testing.assert_allclose(county_totals, [[2, 0, 30, 0], [0, 2, 0, 70]])
    assert result.array("county_totals").shape == (2, 2, 2)

    assert type(result["cut_edges"]) is int
    assert result["cut_edges"] == 2
    assert type(result["weighted_cut_edges"]) is float
    assert result["weighted_cut_edges"] == 6.0
    region_splits = as_series(result["region_splits"])
    region_pieces = as_series(result["region_pieces"])
    region_parts = as_series(result["region_parts"])
    assert region_splits.index.tolist() == ["COUNTY", "MUNI"]
    assert region_splits.dtype == np.dtype("int64")
    assert region_pieces.dtype == np.dtype("int64")
    assert region_parts.dtype == np.dtype("int64")
    assert region_splits.tolist() == [0, 2]
    assert region_pieces.tolist() == [2, 4]
    assert region_parts.tolist() == [2, 4]
    assert not result.array("cut_edges").flags.writeable


def test_adding_a_metric_after_scoring_rebuilds_the_engine() -> None:
    graph, _ = grid_resources()
    scorer = PlanEvaluator(graph).add_metric(Tally("population"))
    scorer.evaluate([0, 0, 1, 1])

    result = scorer.add_metric(CutEdges()).evaluate([0, 0, 1, 1])

    np.testing.assert_allclose(result["population"], [30, 70])
    assert result["cut_edges"] == 2


def test_tally_by_region_can_omit_unit_counts() -> None:
    graph, _ = grid_resources()
    result = (
        PlanEvaluator(graph)
        .add_metric(TallyByRegion("COUNTY", {"population": "population"}))
        .evaluate(["north", "north", "south", "south"])
    )

    table = as_frame(result["tally_by_region"])
    assert table.columns.equals(
        pd.MultiIndex.from_product(
            (["population"], ["north", "south"]),
            names=("metric", "district"),
        )
    )
    assert table.dtypes.tolist() == [np.dtype("float64")] * 2
    np.testing.assert_allclose(table, [[30, 0], [0, 70]])


@pytest.mark.parametrize("district_count", [3, 6, 7])
def test_native_partisan_bias_treats_uniform_roundoff_as_ties(district_count: int) -> None:
    graph = nx.path_graph(district_count)
    nx.set_node_attributes(graph, {node: 2.0 for node in graph}, "party")
    nx.set_node_attributes(graph, {node: 3.0 for node in graph}, "opposition")

    result = (
        PlanEvaluator(graph)
        .add_metric(PartisanBias("party", "opposition"))
        .evaluate(range(district_count))
    )

    assert result["partisan_bias"] == 0.0


@pytest.mark.parametrize(
    ("offsets", "expected"),
    [
        ([0.5e-9, -0.25e-9, -0.25e-9], 0.0),
        ([1.5e-9, -0.75e-9, -0.75e-9], 1 / 6),
    ],
)
def test_native_partisan_bias_pins_the_tie_tolerance(offsets, expected: float) -> None:
    graph = nx.path_graph(3)
    shares = 0.5 + np.asarray(offsets)
    nx.set_node_attributes(graph, dict(enumerate(shares)), "party")
    nx.set_node_attributes(graph, dict(enumerate(1 - shares)), "opposition")

    result = PlanEvaluator(graph).add_metric(PartisanBias("party", "opposition")).evaluate(range(3))

    assert result["partisan_bias"] == pytest.approx(expected)


def test_tally_rejects_boolean_node_attributes() -> None:
    graph = nx.path_graph(2)
    nx.set_node_attributes(graph, {0: True, 1: False}, "population")

    with pytest.raises(ValueError, match="must be finite numeric"):
        PlanEvaluator(graph).add_metric(Tally("population")).evaluate([0, 1])


def test_geometry_metrics_require_geometry_when_registered() -> None:
    graph, _ = grid_resources()
    population = PopulationPolygon("population")

    with pytest.raises(RuntimeError, match="Reock requires geometry"):
        PlanEvaluator(graph).add_metric(Reock())
    with pytest.raises(RuntimeError, match="PopulationPolygon requires geometry"):
        PlanEvaluator(graph).add_metric(population)
    with pytest.raises(RuntimeError, match="ConvexHullRatio requires geometry"):
        PlanEvaluator(graph).add_metric(ConvexHullRatio())
    with pytest.raises(RuntimeError, match="StateClippedConvexHullRatio requires geometry"):
        PlanEvaluator(graph).add_metric(StateClippedConvexHullRatio(box(0, 0, 2, 2)))


def test_metric_descriptions_reject_ambiguous_or_empty_options() -> None:
    assert TallyByRegion("COUNTY", "population").columns == (("population", "population"),)
    assert TallyByRegion("COUNTY", ["population", "vap"]).columns == (
        ("population", "population"),
        ("vap", "vap"),
    )
    assert TallyByRegion("COUNTY", {"population": "TOTPOP"}).columns == (("population", "TOTPOP"),)
    with pytest.raises(ValueError, match="area_attr must be a nonempty graph attribute name"):
        PolsbyPopper(area_attr="")
    with pytest.raises(ValueError, match="perimeter_attr must be a nonempty graph attribute name"):
        PolsbyPopper(perimeter_attr="")
    with pytest.raises(
        ValueError,
        match="boundary_perimeter_attr must be a nonempty graph attribute name",
    ):
        PolsbyPopper(boundary_perimeter_attr="")
    with pytest.raises(
        ValueError, match="shared_perimeter_attr must be a nonempty graph attribute name"
    ):
        PolsbyPopper(shared_perimeter_attr="")
    with pytest.raises(ValueError, match="perimeter_attr or boundary_perimeter_attr"):
        PolsbyPopper(perimeter_attr="perim", boundary_perimeter_attr="boundary")
    with pytest.raises(ValueError, match="nonempty string"):
        CutEdges(weight_attr="")
    with pytest.raises(ValueError, match="RegionSplits requires"):
        RegionSplits()
    with pytest.raises(ValueError, match="nonempty string"):
        TallyByRegion("", include_count=True)
    with pytest.raises(ValueError, match="at least one column"):
        TallyByRegion("COUNTY")
    with pytest.raises(TypeError, match="string, iterable, mapping"):
        TallyByRegion("COUNTY", 1)  # type: ignore
    with pytest.raises(ValueError, match="column names"):
        TallyByRegion(
            "COUNTY",
            [("population", "population")],  # type: ignore
        )
    with pytest.raises(TypeError, match="bool"):
        TallyByRegion("COUNTY", include_count=1)  # type: ignore
    with pytest.raises(ValueError, match="cannot contain 'count'"):
        TallyByRegion("COUNTY", {"count": "population"}, include_count=True)
    with pytest.raises(ValueError, match="column names"):
        TallyByRegion("COUNTY", {"": "population"})
    with pytest.raises(ValueError, match="source columns"):
        TallyByRegion("COUNTY", {"population": ""})
    with pytest.raises(ValueError, match="population_col must be a nonempty"):
        PopulationPolygon("")
    with pytest.raises(TypeError, match="positional"):
        PopulationPolygon(
            "population",
            population_frame([box(0, 0, 1, 1)], [1]),  # type: ignore
        )
    empty = gpd.GeoDataFrame(
        {"population": []},
        geometry=[],
        crs="EPSG:3857",
    )
    with pytest.raises(ValueError, match="at least one observation"):
        PopulationPolygon("population", population_units=empty)
    missing_weight = gpd.GeoDataFrame(
        geometry=[box(0, 0, 1, 1)],
        crs="EPSG:3857",
    )
    with pytest.raises(ValueError, match="does not contain population column"):
        PopulationPolygon("population", population_units=missing_weight)
    invalid_weight = population_frame([box(0, 0, 1, 1)], [-1])
    with pytest.raises(ValueError, match="finite and nonnegative"):
        PopulationPolygon("population", population_units=invalid_weight)

    graph, geometry = grid_resources()
    missing_base_column = geometry.copy()
    del missing_base_column["population"]
    with pytest.raises(ValueError, match="has no 'population' attribute"):
        PlanEvaluator(graph, geometry=missing_base_column).add_metric(
            PopulationPolygon("population")
        ).evaluate([0, 0, 1, 1])
    invalid_base_column = geometry.copy()
    invalid_base_column["population"] = [10, 20, -1, 40]
    with pytest.raises(ValueError, match="cannot contain negative values"):
        PlanEvaluator(graph, geometry=invalid_base_column).add_metric(
            PopulationPolygon("population")
        ).evaluate([0, 0, 1, 1])
    zero_base_column = geometry.copy()
    zero_base_column["population"] = 0
    with pytest.raises(ValueError, match="must have a positive total"):
        PlanEvaluator(graph, geometry=zero_base_column).add_metric(
            PopulationPolygon("population")
        ).evaluate([0, 0, 1, 1])


def test_population_polygon_requires_projected_matching_polygon_geometry() -> None:
    no_crs = gpd.GeoDataFrame(
        {"population": [1]},
        geometry=[box(0, 0, 1, 1)],
    )
    with pytest.raises(ValueError, match="must have a CRS"):
        PopulationPolygon("population", population_units=no_crs)

    geographic = gpd.GeoDataFrame(
        {"population": [1]},
        geometry=[box(0, 0, 1, 1)],
        crs="EPSG:4326",
    )
    with pytest.raises(ValueError, match="projected CRS"):
        PopulationPolygon("population", population_units=geographic)

    points = gpd.GeoDataFrame(
        {"population": [1]},
        geometry=[Point(0.5, 0.5)],
        crs="EPSG:3857",
    )
    with pytest.raises(ValueError, match="only Polygon or MultiPolygon"):
        PopulationPolygon("population", population_units=points)

    graph = nx.empty_graph(1)
    graph_geometry = gpd.GeoDataFrame(geometry=[box(0, 0, 1, 1)], crs="EPSG:3857")
    population = population_frame([box(0, 0, 1, 1)], [1]).to_crs("EPSG:5070")
    with pytest.raises(ValueError, match="must use the same CRS"):
        PlanEvaluator(graph, geometry=graph_geometry).add_metric(
            PopulationPolygon("population", population_units=population)
        ).evaluate([0])


def test_population_polygon_base_path_uses_the_aligned_scorer_geodataframe() -> None:
    graph = nx.Graph()
    graph.add_nodes_from(["left", "right"])
    geometry = gpd.GeoDataFrame(
        {"node": ["right", "left"], "population": [3, 2]},
        geometry=[box(1, 0, 2, 1), box(0, 0, 1, 1)],
        crs="EPSG:3857",
    )
    scorer = PlanEvaluator(graph, geometry=geometry, node_id_column="node").add_metric(
        PopulationPolygon("population")
    )

    np.testing.assert_allclose(
        scorer.evaluate({"left": 0, "right": 1})["population_polygon"],
        [0.4, 0.6],
    )


def test_population_polygon_native_path_rejects_a_zero_population_district() -> None:
    graph = nx.path_graph(2)
    geometry = gpd.GeoDataFrame(
        {"population": [1, 0]},
        geometry=[box(0, 0, 1, 1), box(1, 0, 2, 1)],
        crs="EPSG:3857",
    )
    scorer = PlanEvaluator(graph, geometry=geometry).add_metric(PopulationPolygon("population"))

    with pytest.raises(ValueError, match="district 1 has invalid owned population 0"):
        scorer.evaluate([0, 1])


def test_population_polygon_matches_independent_shapely_oracle() -> None:
    geometries = [box(0, 0, 1, 1), box(1, 0, 2, 1), box(2, 0, 3, 1)]
    assignment = ["A", "B", "A"]
    population_geometries = [
        box(0.1, 0.1, 0.9, 0.9),
        box(1.1, 0.1, 1.9, 0.9),
        box(2.1, 0.1, 2.9, 0.9),
    ]
    weights = [10, 20, 30]
    owners = [0, 1, 2]
    graph = nx.empty_graph(len(geometries))
    frame = gpd.GeoDataFrame(geometry=geometries, crs="EPSG:3857")

    values = as_series(
        PlanEvaluator(graph, geometry=frame)
        .add_metric(population_metric(population_geometries, weights))
        .evaluate(assignment)["population_polygon"]
    )
    expected = population_polygon_scores(
        geometries,
        assignment,
        population_geometries,
        weights,
        owners,
    )

    np.testing.assert_allclose(
        values,
        [expected[district] for district in values.index],
        rtol=1e-12,
        atol=1e-12,
    )


def test_generated_population_polygon_scores_match_independent_oracle() -> None:
    geometries = [box(column, row, column + 1, row + 1) for row in range(4) for column in range(4)]
    population_geometries = [
        box(column + offset, row + 0.1, column + offset + 0.3, row + 0.9)
        for row in range(4)
        for column in range(4)
        for offset in (0.1, 0.6)
    ]
    owners = [node for node in range(16) for _ in range(2)]
    weights = [index % 13 + 1 for index in range(32)]
    graph = nx.empty_graph(len(geometries))
    frame = gpd.GeoDataFrame(geometry=geometries, crs="EPSG:3857")
    scorer = PlanEvaluator(graph, geometry=frame).add_metric(
        population_metric(population_geometries, weights)
    )
    rng = np.random.default_rng(0x50_50_1A)

    for _ in range(200):
        assignment = rng.integers(0, 5, len(geometries)).tolist()
        expected = population_polygon_scores(
            geometries,
            assignment,
            population_geometries,
            weights,
            owners,
        )
        values = as_series(scorer.evaluate(assignment)["population_polygon"])
        np.testing.assert_allclose(
            values,
            [expected[district] for district in values.index],
            rtol=1e-12,
            atol=1e-12,
        )


def test_population_polygon_native_path_preserves_observation_invariants() -> None:
    geometries = [box(0, 0, 1, 1), box(1, 0, 2, 1), box(2, 0, 3, 1)]
    assignment = ["A", "B", "A"]
    graph = nx.empty_graph(len(geometries))
    frame = gpd.GeoDataFrame(geometry=geometries, crs="EPSG:3857")

    def score(population_geometries, weights):
        return as_series(
            PlanEvaluator(graph, geometry=frame)
            .add_metric(population_metric(population_geometries, weights))
            .evaluate(assignment)["population_polygon"]
        ).to_numpy()

    expected = score(
        [box(0.1, 0.1, 0.9, 0.9), box(1.1, 0.1, 1.9, 0.9), box(2.1, 0.1, 2.9, 0.9)],
        [10, 20, 30],
    )
    reordered = score(
        [box(2.1, 0.1, 2.9, 0.9), box(0.1, 0.1, 0.9, 0.9), box(1.1, 0.1, 1.9, 0.9)],
        [30, 10, 20],
    )
    split = score(
        [
            box(0.1, 0.1, 0.9, 0.9),
            box(0.1, 0.1, 0.9, 0.9),
            box(1.1, 0.1, 1.9, 0.9),
            box(2.1, 0.1, 2.9, 0.9),
        ],
        [4, 6, 20, 30],
    )

    np.testing.assert_allclose(reordered, expected, rtol=0, atol=0)
    np.testing.assert_allclose(split, expected, rtol=0, atol=0)


def test_population_polygon_native_path_is_bit_stable_under_row_permutation() -> None:
    geometries = [box(0, 0, 1, 1), box(1, 0, 2, 1), box(2, 0, 3, 1)]
    population_geometries = [
        box(0.1, 0.1, 0.2, 0.2),
        box(0.3, 0.1, 0.4, 0.2),
        box(0.5, 0.1, 0.6, 0.2),
        box(1.1, 0.1, 1.9, 0.9),
        box(2.1, 0.1, 2.9, 0.9),
    ]
    weights = [1e16, 1.0, 1.0, 1e16, 1.0]
    graph = nx.Graph([(0, 1), (1, 2)])
    frame = gpd.GeoDataFrame(geometry=geometries, crs="EPSG:3857")

    def score(order: list[int]) -> np.ndarray:
        population = population_frame(
            [population_geometries[index] for index in order],
            [weights[index] for index in order],
        )
        result = (
            PlanEvaluator(graph, geometry=frame)
            .add_metric(PopulationPolygon("population", population_units=population))
            .evaluate({0: "A", 1: "B", 2: "A"})["population_polygon"]
        )
        return as_series(result).to_numpy()

    np.testing.assert_array_equal(score([0, 1, 2, 3, 4]), score([1, 2, 0, 3, 4]))


def test_population_polygon_native_path_handles_multipolygon_graph_units() -> None:
    geometries = [
        MultiPolygon([box(0, 0, 1, 1), box(3, 0, 4, 1)]),
        box(1.5, 0, 2.5, 1),
    ]
    assignment = ["outer", "inner"]
    population_geometries = [
        box(0.1, 0.1, 0.9, 0.9),
        box(3.1, 0.1, 3.9, 0.9),
        box(1.6, 0.1, 2.4, 0.9),
    ]
    weights = [4, 6, 5]
    owners = [0, 0, 1]
    graph = nx.empty_graph(len(geometries))
    frame = gpd.GeoDataFrame(geometry=geometries, crs="EPSG:3857")

    values = as_series(
        PlanEvaluator(graph, geometry=frame)
        .add_metric(population_metric(population_geometries, weights))
        .evaluate(assignment)["population_polygon"]
    )
    expected = population_polygon_scores(
        geometries,
        assignment,
        population_geometries,
        weights,
        owners,
    )

    np.testing.assert_allclose(
        values,
        [expected[district] for district in values.index],
        rtol=1e-12,
        atol=1e-12,
    )


def test_population_polygon_infers_unique_containing_graph_units() -> None:
    graph = nx.Graph()
    graph.add_nodes_from(["right", "left"])
    frame = gpd.GeoDataFrame(
        {
            "node": ["left", "right"],
            "geometry": [box(0, 0, 1, 1), box(1, 0, 2, 1)],
        },
        crs="EPSG:3857",
    )
    population = population_frame(
        [box(0.1, 0.1, 0.9, 0.9), box(1.1, 0.1, 1.9, 0.9)],
        [10, 20],
    )
    scorer = PlanEvaluator(graph, geometry=frame, node_id_column="node").add_metric(
        PopulationPolygon("population", population_units=population)
    )
    np.testing.assert_allclose(
        scorer.evaluate({"left": 0, "right": 1})["population_polygon"],
        [1, 1],
    )

    with pytest.raises(ValueError, match="covered by exactly one evaluator geometry; found 0"):
        outside = population_frame([box(3.1, 0.1, 3.9, 0.9)], [1])
        PlanEvaluator(graph, geometry=frame, node_id_column="node").add_metric(
            PopulationPolygon("population", population_units=outside)
        ).evaluate({"left": 0, "right": 1})
    with pytest.raises(ValueError, match="covered by exactly one evaluator geometry; found 0"):
        crossing = population_frame([box(0.5, 0.1, 1.5, 0.9)], [1])
        PlanEvaluator(graph, geometry=frame, node_id_column="node").add_metric(
            PopulationPolygon("population", population_units=crossing)
        ).evaluate({"left": 0, "right": 1})

    overlapping = gpd.GeoDataFrame(
        geometry=[box(0, 0, 2, 1), box(0, 0, 2, 1)],
        crs="EPSG:3857",
    )
    ambiguous = population_frame([box(0.1, 0.1, 0.9, 0.9)], [1])
    with pytest.raises(ValueError, match="covered by exactly one evaluator geometry; found 2"):
        PlanEvaluator(nx.empty_graph(2), geometry=overlapping).add_metric(
            PopulationPolygon("population", population_units=ambiguous)
        ).evaluate([0, 1])


def test_geometry_polsby_popper_matches_geos_for_oblique_and_multipart_boundaries() -> None:
    geometries = [
        Polygon([(0, 0), (2, 0), (2, 2), (0, 0)]),
        Polygon([(0, 0), (1, 1), (2, 2), (0, 2), (0, 0)]),
        MultiPolygon([box(10, 0, 11, 1), box(10, 2, 11, 3)]),
        box(11, 0, 12, 3),
    ]
    graph = nx.Graph([(0, 1), (2, 3)])
    frame = gpd.GeoDataFrame(geometry=geometries, crs="EPSG:3857")
    scorer = PlanEvaluator(graph, geometry=frame).add_metric(PolsbyPopper())

    for assignment in ([0, 0, 1, 1], [0, 1, 2, 3]):
        values = as_series(scorer.evaluate(assignment)["polsby_popper"])
        expected = []
        for district in values.index:
            merged = unary_union(
                [geometry for geometry, label in zip(geometries, assignment) if label == district]
            )
            expected.append(4 * math.pi * merged.area / merged.length**2)
        np.testing.assert_allclose(values, expected, rtol=1e-12, atol=1e-12)


def test_generated_geometry_polsby_popper_matches_dissolved_shapely_oracle() -> None:
    geometries = [box(column, row, column + 1, row + 1) for row in range(4) for column in range(4)]
    frame = gpd.GeoDataFrame(geometry=geometries, crs="EPSG:3857")
    scorer = PlanEvaluator(nx.empty_graph(len(frame)), geometry=frame).add_metric(PolsbyPopper())
    rng = np.random.default_rng(0x5015_BEEF)

    for _ in range(300):
        assignment = rng.integers(0, 5, len(frame)).tolist()
        values = as_series(scorer.evaluate(assignment)["polsby_popper"])
        expected = []
        for district in values.index:
            merged = unary_union(
                [geometry for geometry, label in zip(geometries, assignment) if label == district]
            )
            expected.append(4 * math.pi * merged.area / merged.length**2)
        np.testing.assert_allclose(values, expected, rtol=1e-12, atol=1e-12)


def test_explicit_geometry_crs_matches_preprojected_metrics() -> None:
    graph, projected = grid_resources()
    geographic = projected.to_crs("EPSG:4326")
    assignment = [0, 0, 1, 1]

    baseline = (
        PlanEvaluator(graph, geometry=projected)
        .add_metric(PolsbyPopper())
        .add_metric(Reock())
        .evaluate(assignment)
    )
    reprojected = (
        PlanEvaluator(graph, geometry=geographic, target_crs=projected.crs)
        .add_metric(PolsbyPopper())
        .add_metric(Reock())
        .evaluate(assignment)
    )

    np.testing.assert_allclose(reprojected["polsby_popper"], baseline["polsby_popper"])
    np.testing.assert_allclose(reprojected["reock"], baseline["reock"])


def test_reock_has_hand_computable_non_rectangle_scores() -> None:
    # The minimum enclosing circle of this obtuse triangle is the diameter circle of its longest
    # side (radius 2), not its bounding-box circumcircle (radius sqrt(17)/2), so a bbox-based
    # kernel cannot reproduce the score. The triangle's area is 2.
    obtuse_triangle = Polygon([(0, 0), (4, 0), (1, 1)])
    # Two separated unit squares share the circle through (0, 0) and (4, 1): radius^2 = 17/4.
    island_district = MultiPolygon([box(0, 0, 1, 1), box(3, 0, 4, 1)])
    graph = nx.empty_graph(2)
    frame = gpd.GeoDataFrame(geometry=[obtuse_triangle, island_district], crs="EPSG:3857")

    result = (
        PlanEvaluator(graph, geometry=frame).add_metric(Reock()).evaluate(["triangle", "islands"])
    )

    np.testing.assert_allclose(
        result["reock"],
        [2 / (4 * math.pi), 2 / (4.25 * math.pi)],
        rtol=1e-12,
    )


def test_reock_matches_independent_shapely_oracle_for_oblique_and_multipart_districts() -> None:
    geometries = [
        Polygon([(8, 0), (10, 0), (10, 2)]),
        Polygon([(0, 0), (4, 0), (1, 1)]),
        MultiPolygon([box(4, 0, 5, 1), box(6, 0, 7, 1)]),
        box(11, 0, 12, 3),
    ]
    graph = nx.empty_graph(len(geometries))
    frame = gpd.GeoDataFrame(geometry=geometries, crs="EPSG:3857")
    scorer = PlanEvaluator(graph, geometry=frame).add_metric(Reock())

    for assignment in (["L", "L", "R", "R"], ["A", "B", "C", "D"], ["L", "R", "L", "R"]):
        expected = reock_scores(geometries, assignment)
        values = as_series(scorer.evaluate(assignment)["reock"])
        np.testing.assert_allclose(
            values,
            [expected[district] for district in values.index],
            rtol=1e-12,
            atol=1e-12,
        )


def test_generated_reock_scores_match_independent_shapely_oracle() -> None:
    # Right triangles tile the grid so random districts get oblique, frequently disjoint shapes
    # whose minimum enclosing circles differ from their bounding-box circumcircles.
    geometries = []
    for row in range(4):
        for column in range(4):
            geometries.append(Polygon([(column, row), (column + 1, row), (column + 1, row + 1)]))
            geometries.append(Polygon([(column, row), (column + 1, row + 1), (column, row + 1)]))
    graph = nx.empty_graph(len(geometries))
    frame = gpd.GeoDataFrame(geometry=geometries, crs="EPSG:3857")
    scorer = PlanEvaluator(graph, geometry=frame).add_metric(Reock())
    rng = np.random.default_rng(0x2E0C_C1E5)

    for _ in range(200):
        assignment = rng.integers(0, 5, len(geometries)).tolist()
        expected = reock_scores(geometries, assignment)
        values = as_series(scorer.evaluate(assignment)["reock"])
        np.testing.assert_allclose(
            values,
            [expected[district] for district in values.index],
            rtol=1e-12,
            atol=1e-12,
        )


def test_reock_handles_nearly_collinear_sliver_districts() -> None:
    # Every sliver's vertices are nearly collinear along the main diagonal, stressing the
    # min-circle kernel's handling of almost-degenerate point sets.
    thickness = 1e-6
    slivers = [
        Polygon(
            [
                (step, step),
                (step + 1, step + 1),
                (step + 1, step + 1 + thickness),
                (step, step + thickness),
            ]
        )
        for step in range(6)
    ]
    graph = nx.empty_graph(len(slivers))
    frame = gpd.GeoDataFrame(geometry=slivers, crs="EPSG:3857")
    scorer = PlanEvaluator(graph, geometry=frame).add_metric(Reock())

    for assignment in ([0, 0, 0, 1, 1, 1], [0, 1, 0, 1, 0, 1], [0, 0, 1, 1, 2, 2]):
        expected = reock_scores(slivers, assignment)
        values = as_series(scorer.evaluate(assignment)["reock"])
        np.testing.assert_allclose(
            values,
            [expected[district] for district in values.index],
            # GEOS and the engine kernel disagree slightly on near-degenerate circumcircles.
            rtol=1e-8,
        )


def test_convex_hull_ratio_matches_independent_shapely_oracle() -> None:
    geometries = [
        box(0, 0, 1, 1),
        box(1, 0, 2, 1),
        box(0, 1, 1, 2),
        MultiPolygon([box(4, 0, 5, 1), box(6, 0, 7, 1)]),
    ]
    assignment = ["L", "L", "L", "R"]
    graph = nx.empty_graph(len(geometries))
    frame = gpd.GeoDataFrame(geometry=geometries, crs="EPSG:3857")

    values = as_series(
        PlanEvaluator(graph, geometry=frame)
        .add_metric(ConvexHullRatio())
        .evaluate(assignment)["convex_hull_ratio"]
    )
    expected = convex_hull_scores(geometries, assignment)

    np.testing.assert_allclose(
        values,
        [expected[district] for district in values.index],
        rtol=1e-12,
        atol=1e-12,
    )


def test_generated_convex_hull_ratios_match_independent_shapely_oracle() -> None:
    geometries = [
        box(column, row, column + 1, row + 1)
        for row in range(5)
        for column in range(5)
        if (row, column) != (2, 2)
    ]
    graph = nx.empty_graph(len(geometries))
    frame = gpd.GeoDataFrame(geometry=geometries, crs="EPSG:3857")
    scorer = PlanEvaluator(graph, geometry=frame).add_metric(ConvexHullRatio())
    rng = np.random.default_rng(0x5EED_CAFE)

    for _ in range(300):
        assignment = rng.integers(0, 5, len(geometries)).tolist()
        expected = convex_hull_scores(geometries, assignment)
        values = as_series(scorer.evaluate(assignment)["convex_hull_ratio"])
        np.testing.assert_allclose(
            values,
            [expected[district] for district in values.index],
            rtol=1e-12,
            atol=1e-12,
        )


def test_state_clipped_convex_hull_ratio_matches_independent_shapely_oracle() -> None:
    geometries = [
        box(0, 0, 1, 1),
        box(1, 0, 2, 1),
        box(0, 1, 1, 2),
        MultiPolygon([box(4, 0, 5, 1), box(6, 0, 7, 1)]),
    ]
    state = MultiPolygon(
        [
            Polygon([(0, 0), (2, 0), (2, 1), (1, 1), (1, 2), (0, 2)]),
            box(4, 0, 5, 1),
            box(6, 0, 7, 1),
        ]
    )
    assignment = ["L", "L", "L", "R"]
    graph = nx.empty_graph(len(geometries))
    frame = gpd.GeoDataFrame(geometry=geometries, crs="EPSG:3857")

    values = as_series(
        PlanEvaluator(graph, geometry=frame)
        .add_metric(StateClippedConvexHullRatio(state))
        .evaluate(assignment)["state_clipped_convex_hull_ratio"]
    )
    expected = state_clipped_convex_hull_scores(geometries, assignment, state)

    np.testing.assert_allclose(
        values,
        [expected[district] for district in values.index],
        rtol=1e-12,
        atol=1e-12,
    )


def test_state_clipped_convex_hull_ratio_excludes_state_holes() -> None:
    state = Polygon(
        [(0, 0), (4, 0), (4, 4), (0, 4)],
        holes=[[(1, 1), (2, 1), (2, 2), (1, 2)]],
    )
    graph = nx.empty_graph(1)
    frame = gpd.GeoDataFrame(geometry=[state], crs="EPSG:3857")
    result = (
        PlanEvaluator(graph, geometry=frame)
        .add_metric(ConvexHullRatio())
        .add_metric(StateClippedConvexHullRatio(state))
        .evaluate([0])
    )

    np.testing.assert_allclose(result["convex_hull_ratio"], [15 / 16])
    np.testing.assert_allclose(result["state_clipped_convex_hull_ratio"], [1])


def test_generated_state_clipped_ratios_match_independent_shapely_oracle() -> None:
    geometries = [
        box(column, row, column + 1, row + 1)
        for row in range(6)
        for column in range(6)
        if (row, column) not in {(2, 2), (2, 3), (3, 2), (3, 3)}
    ]
    state = unary_union(geometries)
    graph = nx.empty_graph(len(geometries))
    frame = gpd.GeoDataFrame(geometry=geometries, crs="EPSG:3857")
    scorer = (
        PlanEvaluator(graph, geometry=frame)
        .add_metric(ConvexHullRatio())
        .add_metric(StateClippedConvexHullRatio(state))
    )
    rng = np.random.default_rng(0x51A7_EC11)

    for _ in range(300):
        assignment = rng.integers(0, 5, len(geometries)).tolist()
        expected = state_clipped_convex_hull_scores(geometries, assignment, state)
        result = scorer.evaluate(assignment)
        standard = as_series(result["convex_hull_ratio"]).to_numpy()
        values = as_series(result["state_clipped_convex_hull_ratio"])
        np.testing.assert_allclose(
            values,
            [expected[district] for district in values.index],
            # Geo and GEOS polygon overlays may differ at their floating-point snap grids.
            rtol=1e-9,
            atol=1e-12,
        )
        assert np.all(standard <= values.to_numpy() + 1e-12)
        assert np.all((0 < values.to_numpy()) & (values.to_numpy() <= 1))


def test_state_clipped_convex_hull_ratio_rejects_invalid_state_geometry() -> None:
    with pytest.raises(TypeError, match="Shapely"):
        StateClippedConvexHullRatio(
            "not geometry"  # type: ignore
        )
    with pytest.raises(ValueError, match="Polygon or MultiPolygon"):
        StateClippedConvexHullRatio(Polygon().boundary)
    with pytest.raises(ValueError, match="nonempty, valid"):
        StateClippedConvexHullRatio(Polygon())
    with pytest.raises(ValueError, match="nonempty, valid"):
        StateClippedConvexHullRatio(Polygon([(0, 0), (1, 1), (1, 0), (0, 1), (0, 0)]))


def test_state_clipped_convex_hull_ratio_rejects_uncovered_units_during_preparation() -> None:
    graph, geometry = grid_resources()
    scorer = PlanEvaluator(graph, geometry=geometry).add_metric(
        StateClippedConvexHullRatio(box(0, 0, 1.5, 2))
    )

    with pytest.raises(ValueError, match="geometry unit 1 uncovered"):
        scorer.evaluate([0, 0, 1, 1])
