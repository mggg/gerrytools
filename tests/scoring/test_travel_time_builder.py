"""Tests for the reproducible OSM/Census travel-time table builder."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import LineString, MultiLineString, Point, Polygon

import gerrytools.scoring.travel_time.builder as builder_module
from gerrytools.scoring.travel_time import (
    FerryBuildSummary,
    build_osm_drive_graph,
    build_travel_time_table,
    build_travel_time_table_from_pbf,
    census_representative_points,
    load_travel_time_table,
    repair_nonpositive_edge_speeds,
    snap_units_to_graph,
)
from gerrytools.scoring.travel_time.builder import (
    add_vehicle_ferries,
    compute_directed_travel_times,
    parse_osm_duration_seconds,
)


def _directed_graph() -> nx.MultiDiGraph:
    graph = nx.MultiDiGraph()
    graph.add_edge(10, 20, travel_time=5.0)
    graph.add_edge(20, 10, travel_time=15.0)
    graph.add_edge(20, 30, travel_time=4.0)
    graph.add_edge(30, 20, travel_time=6.0)
    return graph


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("5", 300.0),
        ("00:05", 300.0),
        ("1:15", 4500.0),
        ("1:02:03", 3723.0),
        ("PT1H30M", 5400.0),
        (None, None),
        ("", None),
        ("not-a-duration", None),
        ("1:not-a-number", None),
        ("1:75", None),
        ("PT45S", 45.0),
    ],
)
def test_parse_osm_duration_seconds(value, expected):
    assert parse_osm_duration_seconds(value) == expected


def test_census_representative_points_uses_internal_point_columns():
    units = pd.DataFrame(
        {
            "GEOID20": ["001", "002"],
            "INTPTLAT20": ["38.5", "39.0"],
            "INTPTLON20": ["-98.0", "-97.5"],
        }
    )
    points = census_representative_points(units, id_column="GEOID20")
    assert points.crs is not None
    assert points.crs.to_epsg() == 4326
    assert points["GEOID20"].to_list() == ["001", "002"]
    assert points.geometry.iloc[0] == Point(-98.0, 38.5)


@pytest.mark.parametrize(
    "units",
    [
        pd.DataFrame(columns=["GEOID20", "INTPTLAT20", "INTPTLON20"]),
        pd.DataFrame({"GEOID20": ["x", "x"], "INTPTLAT20": [1, 2], "INTPTLON20": [1, 2]}),
        pd.DataFrame({"GEOID20": ["x"], "INTPTLAT20": [91], "INTPTLON20": [0]}),
    ],
)
def test_census_representative_points_rejects_invalid_rows(units):
    with pytest.raises(ValueError):
        census_representative_points(units, id_column="GEOID20")


def test_census_representative_points_rejects_missing_columns():
    with pytest.raises(KeyError, match="columns not found"):
        census_representative_points(pd.DataFrame({"id": [1]}), id_column="id")


@pytest.mark.parametrize("latitude", ["north", float("nan"), float("inf")])
def test_census_representative_points_rejects_invalid_coordinates(latitude):
    units = pd.DataFrame({"id": ["a"], "INTPTLAT20": [latitude], "INTPTLON20": [-98.0]})
    with pytest.raises(ValueError, match="coordinates"):
        census_representative_points(units, id_column="id")


def test_compute_directed_travel_times_preserves_directions_and_duplicate_snaps():
    directed = compute_directed_travel_times(
        _directed_graph(),
        {"a": 10, "b": 20, "same_as_b": 20, "c": 30},
        batch_size=2,
    )
    assert directed[0, 1] == pytest.approx(5.0)
    assert directed[1, 0] == pytest.approx(15.0)
    assert directed[1, 2] == pytest.approx(0.0)
    assert directed[0, 3] == pytest.approx(9.0)
    assert directed[3, 0] == pytest.approx(21.0)


def test_compute_directed_travel_times_uses_shortest_parallel_edge():
    graph = nx.MultiDiGraph()
    graph.add_edge(1, 2, travel_time=10.0)
    graph.add_edge(1, 2, travel_time=3.0)
    graph.add_edge(2, 1, travel_time=4.0)
    directed = compute_directed_travel_times(graph, {"a": 1, "b": 2})
    assert directed == pytest.approx(np.array([[0.0, 3.0], [4.0, 0.0]]))


def test_compute_directed_travel_times_supports_undirected_graphs():
    graph = nx.Graph()
    graph.add_edge(1, 2, travel_time=3.0)
    directed = compute_directed_travel_times(graph, {"a": 1, "b": 2})
    assert directed == pytest.approx(np.array([[0.0, 3.0], [3.0, 0.0]]))


@pytest.mark.parametrize(
    ("graph", "unit_to_node", "batch_size", "message"),
    [
        (nx.DiGraph(), {}, 32, "at least one unit"),
        (nx.DiGraph(), {"a": 1}, 0, "batch_size"),
        (nx.DiGraph([(1, 2)]), {"a": 3}, 32, "not in graph"),
        (nx.DiGraph(), {"a": 1}, 32, "not in graph"),
    ],
)
def test_compute_directed_travel_times_rejects_invalid_inputs(
    graph, unit_to_node, batch_size, message
):
    with pytest.raises((ValueError, KeyError), match=message):
        compute_directed_travel_times(graph, unit_to_node, batch_size=batch_size)


def test_compute_directed_travel_times_rejects_graph_without_weighted_edges():
    graph = nx.DiGraph()
    graph.add_node(1)
    with pytest.raises(ValueError, match="no weighted edges"):
        compute_directed_travel_times(graph, {"a": 1})


@pytest.mark.parametrize("bad_weight", [None, "slow", -1.0, float("inf")])
def test_compute_directed_travel_times_rejects_bad_edges(bad_weight):
    graph = nx.DiGraph()
    if bad_weight is None:
        graph.add_edge(1, 2)
    else:
        graph.add_edge(1, 2, travel_time=bad_weight)
    with pytest.raises(ValueError):
        compute_directed_travel_times(graph, {"a": 1, "b": 2})


def test_repair_nonpositive_edge_speeds_imputes_zero_speed_and_keeps_edge():
    graph = nx.MultiDiGraph()
    graph.add_edge(1, 2, length=1000.0, speed_kph=0.0, travel_time=float("inf"))
    graph.add_edge(2, 1, length=1000.0, speed_kph=50.0, travel_time=72.0)
    summary = repair_nonpositive_edge_speeds(graph, fallback_speed_kph=48.0)
    assert summary.repaired_speed == 1
    assert summary.repaired_travel_time == 1
    assert summary.inspected == 2
    assert graph[1][2][0]["speed_kph"] == 48.0
    assert graph[1][2][0]["speed_imputed"] is True
    assert np.isfinite(graph[1][2][0]["travel_time"])
    assert graph[1][2][0]["travel_time"] == pytest.approx(75.0)
    assert set(graph.edges()) == {(1, 2), (2, 1)}
    directed = compute_directed_travel_times(graph, {"a": 1, "b": 2})
    assert np.isfinite(directed).all()
    assert graph.graph["speed_repair"]["policy"] == "impute_fallback_speed_keep_edge"


def test_repair_nonpositive_edge_speeds_raises_when_length_missing():
    graph = nx.MultiDiGraph()
    graph.add_edge(1, 2, speed_kph=0.0, travel_time=float("inf"))
    with pytest.raises(ValueError, match="still have invalid travel_time"):
        repair_nonpositive_edge_speeds(graph)


def test_component_audit_captures_removed_edges_without_retaining_nodes():
    graph = nx.MultiDiGraph(crs="EPSG:3857")
    for node, x in [(1, 0.0), (2, 10.0), (10, 100.0), (11, 110.0)]:
        graph.add_node(node, x=x, y=0.0)
    graph.add_edge(1, 2, highway="primary", travel_time=1.0, length=10.0)
    graph.add_edge(2, 1, highway="primary", travel_time=1.0, length=10.0)
    graph.add_edge(10, 11, highway="service", travel_time=2.0, length=10.0)
    graph.add_edge(11, 10, highway="service", travel_time=2.0, length=10.0)

    summary, removed = builder_module._retain_principal_strong_component(
        graph, include_component_audit=True
    )

    assert set(graph) == {1, 2}
    assert summary["removed_node_count"] == 2
    assert summary["removed_edge_count"] == 2
    assert removed is not None
    assert len(removed) == 2
    assert set(removed["highway"]) == {"service"}
    assert removed.crs.to_epsg() == 3857
    assert removed.geometry.length.to_list() == pytest.approx([10.0, 10.0])


def test_component_audit_returns_empty_frame_when_nothing_removed():
    graph = nx.MultiDiGraph(crs="EPSG:3857")
    graph.add_node(1, x=0.0, y=0.0)
    graph.add_node(2, x=10.0, y=0.0)
    graph.add_edge(1, 2, travel_time=1.0, length=10.0)
    graph.add_edge(2, 1, travel_time=1.0, length=10.0)

    _, removed = builder_module._retain_principal_strong_component(
        graph, include_component_audit=True
    )

    assert removed is not None
    assert removed.empty
    assert removed.crs.to_epsg() == 3857


def test_build_travel_time_table_writes_validated_artifacts(tmp_path: Path):
    output = tmp_path / "scoring.npz"
    report_path = tmp_path / "report.json"
    audit_path = tmp_path / "directed.npz"
    result = build_travel_time_table(
        _directed_graph(),
        {"a": 10, "b": 20, "c": 30},
        snap_distances_m={"a": 10.0, "b": 20.0, "c": 30.0},
        ferry=FerryBuildSummary(found=1, included=1, estimated_duration=1),
        metadata={"state": "KS"},
        keep_directed=True,
        output_path=output,
        report_path=report_path,
        directed_audit_path=audit_path,
    )
    assert result.is_applicable
    assert result.table is not None
    assert result.table.travel_time("a", "b") == pytest.approx(10.0)
    assert result.report.asymmetric_pair_count == 3
    assert result.report.max_directional_difference_seconds == pytest.approx(12.0)
    assert result.report.mean_snap_distance_m == pytest.approx(20.0)
    assert result.directed_times is not None and not result.directed_times.flags.writeable
    assert dict(result.snap_distances_m) == {"a": 10.0, "b": 20.0, "c": 30.0}
    with pytest.raises(TypeError):
        cast(dict, result.snap_distances_m)["a"] = 0.0
    assert load_travel_time_table(output).metadata["state"] == "KS"
    assert json.loads(report_path.read_text())["status"] == "applicable"
    with np.load(audit_path, allow_pickle=False) as audit:
        assert audit["directed_times"][0, 1] == pytest.approx(5.0)


def test_disconnected_build_is_not_applicable_and_does_not_publish_table(tmp_path: Path):
    graph = nx.MultiDiGraph()
    graph.add_edge(1, 2, travel_time=1.0)
    graph.add_edge(2, 1, travel_time=1.0)
    graph.add_node(3)
    output = tmp_path / "must-not-exist.npz"
    report_path = tmp_path / "not-applicable.json"
    result = build_travel_time_table(
        graph,
        {"main-a": 1, "main-b": 2, "island": 3},
        output_path=output,
        report_path=report_path,
    )
    assert not result.is_applicable
    assert result.table is None
    assert result.report.status == "not_applicable"
    assert result.report.connectivity_component_sizes == (2, 1)
    assert not output.exists()
    assert json.loads(report_path.read_text())["status"] == "not_applicable"


def test_build_rejects_nonfinite_snap_distance():
    with pytest.raises(ValueError, match="finite"):
        build_travel_time_table(
            _directed_graph(), {"a": 10, "b": 20}, snap_distances_m={"a": np.nan}
        )


def test_single_unit_build_has_zero_directional_asymmetry():
    graph = nx.DiGraph()
    graph.add_edge(1, 2, travel_time=1.0)
    result = build_travel_time_table(graph, {"a": 1})
    assert result.report.asymmetric_pair_count == 0
    assert result.report.max_directional_difference_seconds == 0.0


def _fake_osmnx(monkeypatch: pytest.MonkeyPatch) -> None:
    def nearest_nodes(graph, X, Y, return_dist=False):
        _ = graph, Y
        nodes = np.array([1 if float(x) < 500 else 2 for x in np.atleast_1d(X)])
        if return_dist:
            return nodes, np.zeros(len(nodes))
        return nodes

    fake = SimpleNamespace(distance=SimpleNamespace(nearest_nodes=nearest_nodes))
    monkeypatch.setitem(sys.modules, "osmnx", fake)


def test_add_vehicle_ferries_uses_osm_duration_and_estimate(monkeypatch: pytest.MonkeyPatch):
    _fake_osmnx(monkeypatch)
    graph = nx.MultiDiGraph(crs="EPSG:3857")
    graph.add_node(1, x=0.0, y=0.0)
    graph.add_node(2, x=1000.0, y=0.0)
    ferries = gpd.GeoDataFrame(
        {
            "id": [10, 11, 12],
            "motor_vehicle": ["yes", "yes", "no"],
            "duration": ["00:05", None, "00:01"],
            "route": ["ferry", "ferry", "ferry"],
        },
        geometry=[
            LineString([(0, 0), (1000, 0)]),
            LineString([(0, 1), (1000, 1)]),
            LineString([(0, 2), (1000, 2)]),
        ],
        crs="EPSG:3857",
    )
    summary = add_vehicle_ferries(graph, ferries)
    assert summary == FerryBuildSummary(
        found=3,
        included=2,
        skipped_access=1,
        osm_duration=1,
        estimated_duration=1,
    )
    times = sorted(data["travel_time"] for _, _, data in graph.edges(data=True))
    assert times == pytest.approx([180.0, 180.0, 300.0, 300.0])


def test_add_vehicle_ferries_requires_positive_estimation_speed():
    graph = nx.MultiDiGraph(crs="EPSG:3857")
    with pytest.raises(ValueError, match="positive"):
        add_vehicle_ferries(
            graph, gpd.GeoDataFrame(geometry=[], crs="EPSG:3857"), ferry_speed_kph=0
        )


def test_add_vehicle_ferries_handles_empty_input_and_crs_validation():
    graph = nx.MultiDiGraph(crs="EPSG:3857")
    assert add_vehicle_ferries(graph, None) == FerryBuildSummary()
    with pytest.raises(ValueError, match="road graph"):
        add_vehicle_ferries(
            nx.MultiDiGraph(),
            gpd.GeoDataFrame(
                {"motor_vehicle": ["yes"]},
                geometry=[LineString([(0, 0), (1, 1)])],
                crs="EPSG:3857",
            ),
        )
    with pytest.raises(ValueError, match="ferry_features"):
        add_vehicle_ferries(
            graph,
            gpd.GeoDataFrame({"motor_vehicle": ["yes"]}, geometry=[LineString([(0, 0), (1, 1)])]),
        )


def test_add_vehicle_ferries_skips_ambiguous_and_invalid_geometry(
    monkeypatch: pytest.MonkeyPatch,
):
    _fake_osmnx(monkeypatch)
    graph = nx.MultiDiGraph(crs="EPSG:3857")
    ferries = gpd.GeoDataFrame(
        {
            "route": ["ferry", "ferry", "ferry"],
            "motor_vehicle": [None, "yes", "yes"],
            "vehicle": ["private", None, None],
        },
        geometry=[
            LineString([(0, 0), (1000, 0)]),
            Point(0, 0),
            LineString(),
        ],
        crs="EPSG:3857",
    )
    summary = add_vehicle_ferries(graph, ferries)
    assert summary.found == 3
    assert summary.included == 0
    assert summary.skipped_access == 1
    assert summary.skipped_geometry == 2


def test_add_vehicle_ferries_respects_oneway_and_multiline(
    monkeypatch: pytest.MonkeyPatch,
):
    _fake_osmnx(monkeypatch)
    graph = nx.MultiDiGraph(crs="EPSG:3857")
    graph.add_nodes_from([1, 2])
    ferries = gpd.GeoDataFrame(
        {
            "id": [1, 2],
            "ferry": ["primary", "primary"],
            "oneway": ["yes", "-1"],
            "duration": ["2", "3"],
        },
        geometry=[
            MultiLineString([LineString([(0, 0), (500, 0)]), LineString([(500, 0), (1000, 0)])]),
            LineString([(0, 1), (1000, 1)]),
        ],
        crs="EPSG:3857",
    )
    summary = add_vehicle_ferries(graph, ferries)
    assert summary.included == 2
    assert graph.has_edge(1, 2)
    assert graph.has_edge(2, 1)
    assert graph.number_of_edges() == 2


def test_add_vehicle_ferries_skips_route_whose_endpoints_snap_together(
    monkeypatch: pytest.MonkeyPatch,
):
    fake = SimpleNamespace(
        distance=SimpleNamespace(nearest_nodes=lambda graph, X, Y: np.array([1, 1]))
    )
    monkeypatch.setitem(sys.modules, "osmnx", fake)
    graph = nx.MultiDiGraph(crs="EPSG:3857")
    graph.add_node(1)
    ferries = gpd.GeoDataFrame(
        {"motor_vehicle": ["yes"]},
        geometry=[LineString([(0, 0), (100, 0)])],
        crs="EPSG:3857",
    )
    summary = add_vehicle_ferries(graph, ferries)
    assert summary.included == 0
    assert summary.skipped_geometry == 1


def test_snap_units_to_graph_returns_nodes_and_distances(monkeypatch: pytest.MonkeyPatch):
    _fake_osmnx(monkeypatch)
    graph = nx.MultiDiGraph(crs="EPSG:3857")
    points = gpd.GeoDataFrame(
        {"id": ["a", "b"]}, geometry=[Point(0, 0), Point(1000, 0)], crs="EPSG:3857"
    )
    unit_to_node, distances = snap_units_to_graph(graph, points, id_column="id")
    assert unit_to_node == {"a": 1, "b": 2}
    assert distances == {"a": 0.0, "b": 0.0}


def test_polygon_snapping_prefers_inside_node_then_exterior_boundary():
    graph = nx.MultiDiGraph(crs="EPSG:3857")
    graph.add_node("left", x=21.0, y=5.0)
    graph.add_node("right", x=41.0, y=5.0)
    graph.add_node("inside-far", x=2.0, y=5.0)
    graph.add_node("inside-near", x=8.0, y=5.0)
    points = gpd.GeoDataFrame(
        {"id": ["inside", "fallback"]},
        geometry=[Point(9.0, 5.0), Point(30.1, 5.0)],
        crs="EPSG:3857",
    )
    polygons = gpd.GeoDataFrame(
        {"id": ["inside", "fallback"]},
        geometry=[
            Polygon([(0, 0), (10, 0), (10, 10), (0, 10)]),
            Polygon([(30, 0), (40, 0), (40, 10), (30, 10)]),
        ],
        crs="EPSG:3857",
    )

    with pytest.warns(UserWarning, match="exterior-boundary fallback"):
        unit_to_node, distances = snap_units_to_graph(
            graph,
            points,
            id_column="id",
            unit_polygons=polygons,
        )

    assert unit_to_node == {"inside": "inside-near", "fallback": "right"}
    assert distances == pytest.approx({"inside": 1.0, "fallback": 10.9})


def test_polygon_fallback_ignores_interior_hole_boundaries():
    graph = nx.MultiDiGraph(crs="EPSG:3857")
    graph.add_node("in-hole", x=5.0, y=5.0)
    graph.add_node("outside-exterior", x=11.0, y=5.0)
    points = gpd.GeoDataFrame({"id": ["a"]}, geometry=[Point(5.0, 5.0)], crs="EPSG:3857")
    polygon = Polygon(
        [(0, 0), (10, 0), (10, 10), (0, 10)],
        holes=[[(4, 4), (6, 4), (6, 6), (4, 6)]],
    )
    polygons = gpd.GeoDataFrame({"id": ["a"]}, geometry=[polygon], crs="EPSG:3857")

    with pytest.warns(UserWarning, match="exterior-boundary fallback"):
        unit_to_node, distances = snap_units_to_graph(
            graph,
            points,
            id_column="id",
            unit_polygons=polygons,
        )

    assert unit_to_node == {"a": "outside-exterior"}
    assert distances == {"a": 6.0}


def test_polygon_snapping_requires_matching_unit_ids():
    graph = nx.MultiDiGraph(crs="EPSG:3857")
    graph.add_node(1, x=0.0, y=0.0)
    points = gpd.GeoDataFrame({"id": ["a"]}, geometry=[Point(0, 0)], crs="EPSG:3857")
    polygons = gpd.GeoDataFrame(
        {"id": ["b"]},
        geometry=[Polygon([(0, 0), (1, 0), (0, 1)])],
        crs="EPSG:3857",
    )

    with pytest.raises(ValueError, match="exactly the same unit IDs"):
        snap_units_to_graph(
            graph,
            points,
            id_column="id",
            unit_polygons=polygons,
        )


@pytest.mark.parametrize(
    ("graph", "points", "id_column", "message"),
    [
        (
            nx.MultiDiGraph(crs="EPSG:3857"),
            gpd.GeoDataFrame(geometry=[Point(0, 0)], crs="EPSG:3857"),
            "id",
            "ID column",
        ),
        (
            nx.MultiDiGraph(crs="EPSG:3857"),
            gpd.GeoDataFrame({"id": ["a"]}, geometry=[Point(0, 0)]),
            "id",
            "declare a CRS",
        ),
        (
            nx.MultiDiGraph(),
            gpd.GeoDataFrame({"id": ["a"]}, geometry=[Point(0, 0)], crs="EPSG:3857"),
            "id",
            "road graph",
        ),
        (
            nx.MultiDiGraph(crs="EPSG:3857"),
            gpd.GeoDataFrame(
                {"id": ["a", "a"]}, geometry=[Point(0, 0), Point(1, 1)], crs="EPSG:3857"
            ),
            "id",
            "unique",
        ),
        (
            nx.MultiDiGraph(crs="EPSG:3857"),
            gpd.GeoDataFrame(
                {"id": ["a"]}, geometry=[Polygon([(0, 0), (1, 0), (0, 1)])], crs="EPSG:3857"
            ),
            "id",
            "nonempty Point",
        ),
    ],
)
def test_snap_units_to_graph_validates_inputs(graph, points, id_column, message):
    with pytest.raises((KeyError, ValueError), match=message):
        snap_units_to_graph(graph, points, id_column=id_column)


def test_snap_units_to_graph_rejects_nonfinite_coordinates(monkeypatch: pytest.MonkeyPatch):
    _fake_osmnx(monkeypatch)
    graph = nx.MultiDiGraph(crs="EPSG:3857")
    points = gpd.GeoDataFrame(
        {"id": ["a"]},
        geometry=[Point(float("nan"), 0.0)],
        crs="EPSG:3857",
    )
    with pytest.raises(ValueError, match="coordinates must be finite"):
        snap_units_to_graph(graph, points, id_column="id")


def test_snap_units_to_graph_rejects_nonfinite_distance(monkeypatch: pytest.MonkeyPatch):
    fake = SimpleNamespace(
        distance=SimpleNamespace(
            nearest_nodes=lambda graph, X, Y, return_dist: (np.array([1]), np.array([np.inf]))
        )
    )
    monkeypatch.setitem(sys.modules, "osmnx", fake)
    graph = nx.MultiDiGraph(crs="EPSG:3857")
    points = gpd.GeoDataFrame({"id": ["a"]}, geometry=[Point(0, 0)], crs="EPSG:3857")
    with pytest.raises(ValueError, match="finite graph distance"):
        snap_units_to_graph(graph, points, id_column="id")


def test_build_osm_drive_graph_orchestrates_pyrosm_and_osmnx(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    pbf = tmp_path / "state.osm.pbf"
    pbf.write_bytes(b"fixture")
    calls: dict[str, object] = {}

    class FakeOSM:
        def __init__(self, path, **kwargs):
            calls["path"] = path
            calls["osm_kwargs"] = kwargs

        def get_network(self, **kwargs):
            calls["network"] = kwargs
            return pd.DataFrame({"id": [1]}), pd.DataFrame({"id": [2]})

        @staticmethod
        def to_graph(nodes, edges, **kwargs):
            calls["to_graph"] = kwargs
            graph = nx.MultiDiGraph(crs="EPSG:4326")
            graph.add_edge(1, 2, length=10.0, speed_kph=50.0)
            graph.add_edge(2, 1, length=10.0, speed_kph=50.0)
            graph.add_edge(2, 3, length=10.0, speed_kph=50.0)
            graph.add_edge(3, 2, length=10.0, speed_kph=50.0)
            graph.add_edge(10, 11, length=10.0, speed_kph=50.0)
            graph.add_edge(11, 10, length=10.0, speed_kph=50.0)
            return graph

        def get_data_by_custom_criteria(self, **kwargs):
            calls["ferries"] = kwargs
            return None

    def project_graph(graph):
        graph.graph["crs"] = "EPSG:3857"
        return graph

    def add_edge_speeds(graph, **kwargs):
        calls["add_edge_speeds"] = kwargs
        return graph

    def add_edge_travel_times(graph):
        for _, _, data in graph.edges(data=True):
            data["travel_time"] = 1.0
        return graph

    routing = SimpleNamespace(
        add_edge_speeds=add_edge_speeds,
        add_edge_travel_times=add_edge_travel_times,
    )
    monkeypatch.setitem(sys.modules, "pyrosm", SimpleNamespace(OSM=FakeOSM))
    monkeypatch.setitem(
        sys.modules, "osmnx", SimpleNamespace(routing=routing, project_graph=project_graph)
    )

    with pytest.warns(UserWarning, match="ferries were added before this filter"):
        graph, summary = build_osm_drive_graph(pbf)
    assert graph.graph["source_pbf"] == str(pbf.resolve())
    assert graph.graph["ferry_speed_kph"] == 20.0
    assert set(graph) == {1, 2, 3}
    assert graph.graph["connectivity_filter"] == {
        "method": "largest_strongly_connected_component",
        "original_strong_component_count": 2,
        "original_node_count": 5,
        "original_edge_count": 6,
        "retained_node_count": 3,
        "retained_edge_count": 4,
        "removed_node_count": 2,
        "removed_edge_count": 2,
    }
    assert summary == FerryBuildSummary()
    assert calls["osm_kwargs"] == {"keep_metadata": False}
    assert calls["add_edge_speeds"] == {"fallback": 48.0}
    assert calls["network"] == {"network_type": "driving", "nodes": True}
    assert cast(dict, calls["to_graph"])["retain_all"] is True
    assert cast(dict, calls["ferries"])["keep_relations"] is False
    assert graph.graph["speed_repair"]["repaired_speed"] == 0


def test_build_osm_drive_graph_wraps_pyrosm_none_type_parser_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    pbf = tmp_path / "connecticut.osm.pbf"
    pbf.write_bytes(b"fixture")

    class BrokenOSM:
        def __init__(self, path, **kwargs):
            self.path = path

        def get_network(self, **kwargs):
            raise TypeError(
                "int() argument must be a string, a bytes-like object or a real number, not 'NoneType'"
            )

    monkeypatch.setitem(sys.modules, "pyrosm", SimpleNamespace(OSM=BrokenOSM))
    monkeypatch.setitem(sys.modules, "osmnx", SimpleNamespace())
    with pytest.raises(RuntimeError, match="keep_metadata=False"):
        build_osm_drive_graph(pbf)


def test_open_pyrosm_reader_falls_back_only_for_unsupported_keyword(tmp_path: Path):
    pbf = tmp_path / "roads.osm.pbf"
    pbf.write_bytes(b"fixture")
    calls: list[tuple[str, dict[str, object]]] = []

    class LegacyOSM:
        def __init__(self, path, **kwargs):
            calls.append((path, kwargs))
            if kwargs:
                raise TypeError("unexpected keyword argument 'keep_metadata'")

    reader = builder_module._open_pyrosm_reader(pbf, LegacyOSM)

    assert isinstance(reader, LegacyOSM)
    assert calls == [(str(pbf), {"keep_metadata": False}), (str(pbf), {})]


def test_open_pyrosm_reader_does_not_mask_internal_type_error(tmp_path: Path):
    pbf = tmp_path / "roads.osm.pbf"
    pbf.write_bytes(b"fixture")

    class BrokenOSM:
        def __init__(self, path, **kwargs):
            raise TypeError("internal constructor failure")

    with pytest.raises(TypeError, match="internal constructor failure"):
        builder_module._open_pyrosm_reader(pbf, BrokenOSM)


def test_build_osm_drive_graph_rejects_missing_or_empty_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    with pytest.raises(FileNotFoundError):
        build_osm_drive_graph(tmp_path / "missing.pbf")

    pbf = tmp_path / "empty.osm.pbf"
    pbf.write_bytes(b"fixture")

    class EmptyOSM:
        def __init__(self, path, **kwargs):
            pass

        def get_network(self, **kwargs):
            return None, None

    monkeypatch.setitem(sys.modules, "pyrosm", SimpleNamespace(OSM=EmptyOSM))
    monkeypatch.setitem(sys.modules, "osmnx", SimpleNamespace())
    with pytest.raises(ValueError, match="No drivable roads"):
        build_osm_drive_graph(pbf)


def test_build_travel_time_table_from_pbf_forwards_inputs_and_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    points = gpd.GeoDataFrame(
        {"id": ["a"]},
        geometry=[Point(-98.0, 38.0)],
        crs="EPSG:4326",
    )
    graph = nx.MultiDiGraph(crs="EPSG:3857")
    graph.add_edge(1, 2, travel_time=1.0)
    ferry = FerryBuildSummary(found=1, included=1)
    expected = object()
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        builder_module,
        "build_osm_drive_graph",
        lambda path, ferry_speed_kph=20.0, fallback_speed_kph=48.0: (graph, ferry),
    )
    monkeypatch.setattr(
        builder_module,
        "snap_units_to_graph",
        lambda graph, points, *, id_column, unit_polygons: ({"a": 1}, {"a": 12.0}),
    )

    def fake_build(graph_arg, unit_to_node, **kwargs):
        captured["graph"] = graph_arg
        captured["units"] = unit_to_node
        captured.update(kwargs)
        return expected

    monkeypatch.setattr(builder_module, "build_travel_time_table", fake_build)
    result = build_travel_time_table_from_pbf(
        points,
        tmp_path / "state.osm.pbf",
        id_column="id",
        metadata={"state": "KS"},
        keep_directed=True,
    )
    assert result is expected
    assert captured["graph"] is graph
    assert captured["units"] == {"a": 1}
    assert captured["ferry"] == ferry
    assert captured["keep_directed"] is True
    assert captured["snap_distances_m"] == {"a": 12.0}
    metadata = cast(dict, captured["metadata"])
    assert metadata["state"] == "KS"
    assert metadata["symmetrization"] == "mean_of_directions"
    assert metadata["snapping"] == {
        "method": "nearest_drive_graph_node",
        "road_network": "principal_strong_component",
    }
    assert metadata["road_graph_filter"] == {}
    assert metadata["representative_point"] == {
        "method": "caller_supplied_points",
        "crs": "EPSG:4326",
    }


def test_report_metadata_and_result_mappings_are_read_only():
    result = build_travel_time_table(_directed_graph(), {"a": 10, "b": 20})
    with pytest.raises(TypeError):
        cast(dict, result.unit_to_node)["a"] = 99
    with pytest.raises(TypeError):
        cast(dict, result.report.metadata)["new"] = True
