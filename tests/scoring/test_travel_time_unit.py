"""Focused tests for the travel-time scoring core and public API."""

from __future__ import annotations

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
import pytest
from gerrychain import Partition
from shapely.geometry import Point

import gerrytools.scoring as scoring
from gerrytools.scoring.travel_time import (
    CentroidWeightMode,
    PairwiseWeightMode,
    TravelTimeTable,
    centroid_travel_time,
    pairwise_travel_time,
    travel_time_scores,
)
from gerrytools.scoring.travel_time.district import (
    score_centroid_district,
    score_pairwise_district,
)


def corridor_table() -> TravelTimeTable:
    """Return four units on a ten-minute corridor."""
    times = np.array(
        [
            [0.0, 10.0, 20.0, 30.0],
            [10.0, 0.0, 10.0, 20.0],
            [20.0, 10.0, 0.0, 10.0],
            [30.0, 20.0, 10.0, 0.0],
        ]
    )
    return TravelTimeTable.from_matrix(("a", "b", "c", "d"), times)


def plan_graph() -> nx.Graph:
    """Return a graph with two districts' population attributes."""
    graph = nx.path_graph(("a", "b", "c", "d"))
    for node, population in zip(graph.nodes, (100.0, 0.0, 50.0, 25.0), strict=True):
        graph.nodes[node]["population"] = population
    return graph


def assignment() -> dict[str, str]:
    return {"a": "west", "b": "west", "c": "east", "d": "east"}


def plan_frame() -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        {
            "district": ["west", "west", "east", "east"],
            "population": [100.0, 0.0, 50.0, 25.0],
        },
        index=["a", "b", "c", "d"],
        geometry=[Point(i, 0) for i in range(4)],
        crs="EPSG:3857",
    )


def test_travel_time_table_is_immutable_and_has_value_equality() -> None:
    source = np.array([[0.0, 5.0], [5.0, 0.0]])
    table = TravelTimeTable.from_matrix(("a", "b"), source)
    source[0, 1] = 99.0

    assert table == TravelTimeTable.from_matrix(("a", "b"), np.array([[0.0, 5.0], [5.0, 0.0]]))
    assert table != TravelTimeTable.from_matrix(("a", "b"), np.array([[0.0, 6.0], [6.0, 0.0]]))
    assert table.travel_time("a", "b") == pytest.approx(5.0)
    with pytest.raises(ValueError, match="read-only"):
        table.times[0, 1] = 8.0


def test_directed_table_averages_opposite_routes() -> None:
    table = TravelTimeTable.from_directed_matrix(("a", "b"), np.array([[0.0, 5.0], [15.0, 0.0]]))
    assert table.travel_time("a", "b") == pytest.approx(10.0)
    assert table.travel_time("b", "a") == pytest.approx(10.0)


def test_disconnected_table_is_rejected_before_scoring() -> None:
    table = TravelTimeTable.from_matrix(("a", "b"), np.array([[0.0, np.nan], [np.nan, 0.0]]))
    with pytest.raises(ValueError, match="requires statewide connectivity"):
        pairwise_travel_time(plan_graph(), assignment(), od=table)


def test_public_functions_return_district_series() -> None:
    graph = plan_graph()
    ptt = pairwise_travel_time(graph, assignment(), od=corridor_table())
    ctt = centroid_travel_time(graph, assignment(), od=corridor_table())

    pd.testing.assert_series_equal(
        ptt,
        pd.Series(
            [10.0, 10.0],
            index=pd.Index(("west", "east"), dtype=object, name="district"),
            name="pairwise_travel_time",
        ),
    )
    pd.testing.assert_series_equal(
        ctt,
        pd.Series(
            [5.0, 5.0],
            index=pd.Index(("west", "east"), dtype=object, name="district"),
            name="centroid_travel_time",
        ),
    )


def test_unweighted_modes_do_not_require_population() -> None:
    graph = nx.path_graph(("a", "b", "c", "d"))
    assert pairwise_travel_time(graph, assignment(), od=corridor_table()).tolist() == [10.0, 10.0]
    assert centroid_travel_time(graph, assignment(), od=corridor_table()).tolist() == [5.0, 5.0]


def test_population_weighted_modes_are_raw_and_use_source_attribute() -> None:
    """Population weights scale terms without changing the trip-count denominator."""
    graph = plan_graph()
    ptt = pairwise_travel_time(
        graph,
        assignment(),
        od=corridor_table(),
        population_attr="population",
        weight=PairwiseWeightMode.POPULATION_SUM,
    )
    ctt = centroid_travel_time(
        graph,
        assignment(),
        od=corridor_table(),
        population_attr="population",
        weight=CentroidWeightMode.DESTINATION_POPULATION,
    )
    assert ptt["west"] == pytest.approx(1_000.0)
    assert ptt["east"] == pytest.approx(750.0)
    assert ctt["west"] == pytest.approx(0.0)
    assert ctt["east"] == pytest.approx(125.0)


def test_weighted_modes_require_population_attribute() -> None:
    with pytest.raises(ValueError, match="population_attr is required"):
        pairwise_travel_time(plan_graph(), assignment(), od=corridor_table(), weight="p*p")


def test_all_zero_weights_produce_consistent_zero_mean_and_maximum() -> None:
    populations = {"a": 0.0, "b": 0.0}
    pairwise = score_pairwise_district(
        ("a", "b"), populations, corridor_table(), PairwiseWeightMode.POPULATION_SUM
    )
    centroid = score_centroid_district(
        ("a", "b"),
        populations,
        corridor_table(),
        CentroidWeightMode.DESTINATION_POPULATION,
    )
    assert pairwise.mean == pairwise.maximum == 0.0
    assert centroid.mean == centroid.maximum == 0.0
    assert centroid.centroid == "a"


def test_duplicate_district_units_are_rejected() -> None:
    with pytest.raises(ValueError, match="must be unique"):
        score_pairwise_district(("a", "a"), None, corridor_table(), PairwiseWeightMode.NONE)


def test_invalid_weight_string_is_rejected_at_public_boundary() -> None:
    with pytest.raises(ValueError, match="Unknown pairwise weight mode"):
        pairwise_travel_time(plan_graph(), assignment(), od=corridor_table(), weight="pineapple")


def test_travel_time_scores_preserves_district_detail() -> None:
    result = travel_time_scores(plan_graph(), assignment(), od=corridor_table())
    assert list(result.columns) == ["ptt_mean", "ptt_max", "ctt_mean", "ctt_max", "centroid"]
    assert result.loc["west"].to_dict() == {
        "ptt_mean": 10.0,
        "ptt_max": 10.0,
        "ctt_mean": 5.0,
        "ctt_max": 10.0,
        "centroid": "a",
    }


def test_geodataframe_source_uses_assignment_and_population_columns() -> None:
    frame = plan_frame()
    result = pairwise_travel_time(
        frame,
        "district",
        od=corridor_table(),
        population_attr="population",
        weight="p+p",
    )
    assert result.to_dict() == {"west": 1_000.0, "east": 750.0}


def test_partition_source_uses_its_assignment_and_node_attributes() -> None:
    partition = Partition(plan_graph(), assignment())
    result = centroid_travel_time(
        partition,
        od=corridor_table(),
        population_attr="population",
        weight="p",
    )
    assert result.to_dict() == {"west": 0.0, "east": 125.0}


def test_assignment_must_exactly_match_graph_nodes() -> None:
    with pytest.raises(ValueError, match="must exactly match"):
        pairwise_travel_time(plan_graph(), {"a": "west"}, od=corridor_table())


def test_singleton_district_scores_are_nan() -> None:
    graph = plan_graph()
    result = travel_time_scores(
        graph,
        {"a": "one", "b": "two", "c": "two", "d": "two"},
        od=corridor_table(),
    )
    assert result.loc["one", ["ptt_mean", "ptt_max", "ctt_mean", "ctt_max"]].isna().all()
    assert pd.isna(result.loc["one", "centroid"])


def test_top_level_scoring_namespace_exports_primary_api() -> None:
    assert scoring.pairwise_travel_time is pairwise_travel_time
    assert scoring.centroid_travel_time is centroid_travel_time
    assert scoring.travel_time_scores is travel_time_scores
    assert scoring.TravelTimeTable is TravelTimeTable
