import gc
import weakref
from typing import cast

import geopandas as gpd
import networkx as nx
import pandas as pd
import pytest
from gerrychain import Partition
from shapely.geometry import box

import gerrytools.scoring as scoring
from gerrytools.scoring import (
    ConvexHullRatio,
    CutEdges,
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
from gerrytools.scoring.single_plan._base import _expect


def test_single_plan_namespace_owns_convenience_functions() -> None:
    assert scoring.single_plan.tally is scoring.tally
    assert scoring.single_plan.polsby_popper is scoring.polsby_popper


def test_single_plan_result_boundary_rejects_unexpected_type() -> None:
    with pytest.raises(RuntimeError, match="returned Series; expected float or int"):
        _expect((float, int), pd.Series([1.0]))


def resources() -> tuple[nx.Graph, gpd.GeoDataFrame, Partition]:
    graph = nx.Graph()
    for node in ("c", "a", "d", "b"):
        graph.add_node(
            node,
            population=1_000,
            graph_only=5,
            region="stale",
            area=1,
            boundary_perim=2,
        )
    for left, right in (("a", "b"), ("a", "c"), ("b", "d"), ("c", "d")):
        graph.add_edge(left, right, shared_perim=1, weight=2)

    units = gpd.GeoDataFrame(
        {
            "node": ["d", "b", "c", "a"],
            "population": [40, 20, 30, 10],
            "vap": [32, 16, 24, 8],
            "region": ["B", "A", "B", "A"],
            "district": ["south", "north", "south", "north"],
        },
        geometry=[
            box(1, 0, 2, 1),
            box(1, 1, 2, 2),
            box(0, 0, 1, 1),
            box(0, 1, 1, 2),
        ],
        crs="EPSG:3857",
    )
    assignment = {"a": "north", "b": "north", "c": "south", "d": "south"}
    return graph, units, Partition(graph, assignment)


def expected_partition(
    graph: nx.Graph,
    geometry: gpd.GeoDataFrame | None,
    partition: Partition,
    metric,
):
    evaluator = PlanEvaluator(
        graph,
        geometry=geometry,
        node_id_column="node" if geometry is not None else None,
    )
    result = evaluator.add_metric(metric).evaluate(partition)
    return result[result.metrics[0]]


def expected_frame(frame: gpd.GeoDataFrame, metric):
    evaluator = PlanEvaluator(nx.empty_graph(tuple(frame.index)), geometry=frame)
    result = evaluator.add_metric(metric).evaluate(frame["district"].tolist())
    return result[result.metrics[0]]


def assert_same(actual, expected) -> None:
    if isinstance(expected, pd.DataFrame):
        assert isinstance(actual, pd.DataFrame)
        pd.testing.assert_frame_equal(actual, expected)
    elif isinstance(expected, pd.Series):
        assert isinstance(actual, pd.Series)
        pd.testing.assert_series_equal(actual, expected)
    elif isinstance(expected, float):
        assert actual == pytest.approx(expected)
    else:
        assert actual == expected


def test_partition_single_plan_functions_match_plan_evaluator() -> None:
    graph, units, partition = resources()
    state = box(0, 0, 2, 2)
    cases = [
        (
            scoring.tally(
                partition,
                columns=("population", "vap"),
                geometry=units,
                node_id_column="node",
            ),
            Tally("population", "vap"),
            True,
        ),
        (
            scoring.polsby_popper(partition),
            PolsbyPopper(),
            False,
        ),
        (
            scoring.polsby_popper(
                partition,
                geometry=units,
                node_id_column="node",
            ),
            PolsbyPopper(),
            True,
        ),
        (
            scoring.reock(partition, geometry=units, node_id_column="node"),
            Reock(),
            True,
        ),
        (
            scoring.convex_hull_ratio(partition, geometry=units, node_id_column="node"),
            ConvexHullRatio(),
            True,
        ),
        (
            scoring.state_clipped_convex_hull_ratio(
                partition,
                state_geometry=state,
                geometry=units,
                node_id_column="node",
            ),
            StateClippedConvexHullRatio(state),
            True,
        ),
        (
            scoring.population_polygon(
                partition,
                population_col="population",
                geometry=units,
                node_id_column="node",
            ),
            PopulationPolygon("population"),
            True,
        ),
        (
            scoring.cut_edges(partition, weight_attr="weight"),
            CutEdges("weight"),
            False,
        ),
        (
            scoring.region_splits(
                partition,
                region_attrs=("region",),
                geometry=units,
                node_id_column="node",
            ),
            RegionSplits("region"),
            True,
        ),
        (
            scoring.region_pieces(
                partition,
                region_attrs="region",
                geometry=units,
                node_id_column="node",
            ),
            RegionPieces("region"),
            True,
        ),
        (
            scoring.region_parts(
                partition,
                region_attrs="region",
                geometry=units,
                node_id_column="node",
            ),
            RegionParts("region"),
            True,
        ),
        (
            scoring.tally_by_region(
                partition,
                region_attr="region",
                columns={"population": "population"},
                include_count=True,
                geometry=units,
                node_id_column="node",
            ),
            TallyByRegion(
                "region",
                {"population": "population"},
                include_count=True,
            ),
            True,
        ),
    ]

    for actual, metric, uses_geometry in cases:
        expected = expected_partition(graph, units if uses_geometry else None, partition, metric)
        assert_same(actual, expected)


def test_geodataframe_single_plan_functions_match_plan_evaluator() -> None:
    _, units, _ = resources()
    frame = cast(gpd.GeoDataFrame, units.set_index("node"))
    state = box(0, 0, 2, 2)
    cases = [
        (
            scoring.tally(frame, "district", columns=("population", "vap")),
            Tally("population", "vap"),
        ),
        (scoring.polsby_popper(frame, "district"), PolsbyPopper()),
        (scoring.reock(frame, "district"), Reock()),
        (scoring.convex_hull_ratio(frame, "district"), ConvexHullRatio()),
        (
            scoring.state_clipped_convex_hull_ratio(
                frame,
                "district",
                state_geometry=state,
            ),
            StateClippedConvexHullRatio(state),
        ),
        (
            scoring.population_polygon(
                frame,
                "district",
                population_col="population",
            ),
            PopulationPolygon("population"),
        ),
        (
            scoring.region_splits(frame, "district", region_attrs="region"),
            RegionSplits("region"),
        ),
        (
            scoring.region_pieces(frame, "district", region_attrs="region"),
            RegionPieces("region"),
        ),
        (
            scoring.tally_by_region(
                frame,
                "district",
                region_attr="region",
                columns={"population": "population"},
                include_count=True,
            ),
            TallyByRegion(
                "region",
                {"population": "population"},
                include_count=True,
            ),
        ),
    ]

    for actual, metric in cases:
        assert_same(actual, expected_frame(frame, metric))


def test_geodataframe_assignment_forms_and_errors() -> None:
    _, units, _ = resources()
    frame = cast(gpd.GeoDataFrame, units.set_index("node"))
    sequence = frame["district"].tolist()
    mapping = dict(zip(frame.index, sequence, strict=True))
    expected = scoring.tally(frame, "district", columns="population")

    assert_same(scoring.tally(frame, sequence, columns="population"), expected)
    assert_same(scoring.tally(frame, mapping, columns="population"), expected)
    reordered = frame["district"].reindex(["d", "c", "b", "a"])
    assert_same(scoring.tally(frame, reordered, columns="population"), expected)

    with pytest.raises(ValueError, match="assignment column"):
        scoring.tally(frame, "missing", columns="population")
    with pytest.raises(ValueError, match="expected 4"):
        scoring.tally(frame, sequence[:-1], columns="population")
    with pytest.raises(ValueError, match="exactly match GeoDataFrame index"):
        scoring.tally(frame, {**mapping, "extra": "north"}, columns="population")
    with pytest.raises(ValueError, match="Series index must be unique"):
        scoring.tally(
            frame,
            pd.Series(sequence, index=["c", "a", "d", "d"]),
            columns="population",
        )
    missing = frame.copy()
    missing.loc["a", "district"] = None
    with pytest.raises(ValueError, match="missing district"):
        scoring.tally(missing, "district", columns="population")


def test_single_plan_graph_sources_and_topology_requirements() -> None:
    graph, units, partition = resources()
    frame = cast(gpd.GeoDataFrame, units.set_index("node"))
    assignment = {"a": "north", "b": "north", "c": "south", "d": "south"}

    assert scoring.cut_edges(graph, assignment, weight_attr="weight") == scoring.cut_edges(
        partition,
        weight_attr="weight",
    )
    assert_same(
        scoring.polsby_popper(graph, assignment),
        scoring.polsby_popper(partition),
    )
    assert scoring.region_parts(
        graph,
        assignment,
        region_attrs="region",
        geometry=units,
        node_id_column="node",
    ) == scoring.region_parts(
        partition,
        region_attrs="region",
        geometry=units,
        node_id_column="node",
    )
    assert_same(
        scoring.tally(
            graph, assignment, columns="population", geometry=units, node_id_column="node"
        ),
        scoring.tally(partition, columns="population", geometry=units, node_id_column="node"),
    )

    with pytest.raises(TypeError, match="requires a GerryChain Partition or graph"):
        scoring.cut_edges(frame)  # type: ignore
    with pytest.raises(TypeError, match="requires a GerryChain Partition or graph"):
        scoring.region_parts(
            frame,  # type: ignore
            region_attrs="region",
        )
    with pytest.raises(TypeError, match="graph source requires an assignment"):
        scoring.tally(graph, columns="population")
    with pytest.raises(TypeError, match="cannot be a GeoDataFrame column"):
        scoring.tally(graph, "district", columns="population")
    with pytest.raises(RuntimeError, match="requires geometry"):
        scoring.reock(partition)
    with pytest.raises(TypeError, match="supplies its own assignment"):
        scoring.reock(partition, [0, 0, 1, 1], geometry=units)
    with pytest.raises(TypeError, match="source must"):
        scoring.tally(
            object(),  # type: ignore
            columns="population",
        )


def test_single_plan_sources_reject_ambiguous_argument_shapes() -> None:
    graph, units, partition = resources()
    frame = cast(gpd.GeoDataFrame, units.set_index("node"))

    with pytest.raises(TypeError, match="GeoDataFrame source requires an assignment"):
        scoring.tally(frame, columns="population")
    with pytest.raises(ValueError, match="node_id_column and target_crs require geometry"):
        scoring.tally(partition, columns="population", node_id_column="node")
    with pytest.raises(TypeError, match="already a GeoDataFrame"):
        scoring.tally(frame, "district", columns="population", geometry=units)
    with pytest.raises(TypeError, match="node_id_column applies only"):
        scoring.tally(frame, "district", columns="population", node_id_column="node")
    with pytest.raises(TypeError, match="supplies its own assignment"):
        scoring.polsby_popper(partition, [0, 0, 1, 1])
    with pytest.raises(TypeError, match="district labels, not a GeoDataFrame"):
        scoring.polsby_popper(partition, units, geometry=units)
    with pytest.raises(TypeError, match="district labels, not a GeoDataFrame"):
        scoring.polsby_popper(graph, units)
    with pytest.raises(TypeError, match="district labels, not a GeoDataFrame"):
        scoring.polsby_popper(frame, units)


def test_partition_assignment_uses_original_labels_and_caches_graph_check() -> None:
    graph, units, partition = resources()
    reordered = nx.Graph()
    reordered.add_nodes_from(sorted(graph.nodes(data=True)))
    reordered.add_edges_from(graph.edges(data=True))
    evaluator = PlanEvaluator(reordered, geometry=units, node_id_column="node").add_metric(
        Tally("population")
    )

    expected = evaluator.evaluate({"a": "north", "b": "north", "c": "south", "d": "south"})[
        "population"
    ]
    actual = evaluator.evaluate(partition)["population"]
    assert_same(actual, expected)
    assert len(evaluator._verified_partition_graphs) == 1

    child = partition.flip({0: "north"})
    evaluator.evaluate_many([partition, child])
    assert len(evaluator._verified_partition_graphs) == 1

    foreign_graph = graph.copy()
    foreign_graph.remove_edge("a", "b")
    foreign = Partition(
        foreign_graph,
        {"a": "north", "b": "north", "c": "south", "d": "south"},
    )
    with pytest.raises(ValueError, match="same original node and edge sets"):
        evaluator.evaluate(foreign)


def test_partition_graph_cache_does_not_retain_independent_graphs() -> None:
    graph, units, _ = resources()
    evaluator = PlanEvaluator(graph, geometry=units, node_id_column="node").add_metric(
        Tally("population")
    )

    partition = Partition(
        graph.copy(),
        {"a": "north", "b": "north", "c": "south", "d": "south"},
    )
    cached_graph = weakref.ref(partition.graph.graph)
    evaluator.evaluate(partition)
    assert len(evaluator._verified_partition_graphs) == 1

    del partition
    gc.collect()

    assert cached_graph() is None
    assert len(evaluator._verified_partition_graphs) == 0


def test_plan_evaluator_series_assignment_uses_index_labels() -> None:
    graph, _, _ = resources()
    evaluator = PlanEvaluator(graph).add_metric(Tally("population"))
    assignment = pd.Series({"a": "north", "b": "north", "c": "south", "d": "south"}).reindex(
        ["d", "c", "b", "a"]
    )

    assert_same(
        evaluator.evaluate(assignment)["population"],
        evaluator.evaluate(assignment.to_dict())["population"],
    )


def test_geometry_is_authoritative_but_graph_polsby_uses_graph_measurements() -> None:
    graph, units, partition = resources()
    evaluator = PlanEvaluator(graph, geometry=units, node_id_column="node")
    population = evaluator.add_metric(Tally("population")).evaluate(partition)["population"]
    assert isinstance(population, pd.Series)
    assert population.sum() == 100

    with pytest.raises(ValueError, match="geometry row .* has no 'graph_only'"):
        PlanEvaluator(graph, geometry=units, node_id_column="node").add_metric(
            Tally("graph_only")
        ).evaluate([0, 0, 1, 1])

    graph_values = PlanEvaluator(graph, geometry=units, node_id_column="node").add_metric(
        PolsbyPopper(area_attr="area")
    )
    inferred_graph_values = PlanEvaluator(graph, geometry=units, node_id_column="node").add_metric(
        PolsbyPopper(area_attr="area")
    )
    without_geometry = PlanEvaluator(graph).add_metric(PolsbyPopper())
    assert_same(
        graph_values.evaluate(partition)["polsby_popper"],
        inferred_graph_values.evaluate(partition)["polsby_popper"],
    )
    assert_same(
        graph_values.evaluate(partition)["polsby_popper"],
        without_geometry.evaluate(partition)["polsby_popper"],
    )


@pytest.mark.parametrize("metric", [scoring.polsby_popper, scoring.schwartzberg])
def test_compactness_wrapper_graph_columns_override_available_geometry(metric) -> None:
    graph, units, _ = resources()
    assignment = {"a": "north", "b": "north", "c": "south", "d": "south"}
    expected = PlanEvaluator(graph).add_metric(PolsbyPopper(area_attr="area")).evaluate(assignment)

    actual = metric(
        graph,
        assignment,
        geometry=units,
        node_id_column="node",
        area_attr="area",
    )

    expected_values = cast(pd.Series, expected["polsby_popper"])
    if metric is scoring.schwartzberg:
        expected_values = (1 / expected_values.pow(0.5)).rename("schwartzberg")
    assert_same(actual, expected_values)


def test_geometry_polsby_derives_rook_edges_and_reuses_native_geometry() -> None:
    frame = gpd.GeoDataFrame(
        {"district": ["A", "A", "B"]},
        geometry=[
            box(0, 0, 1, 1),
            box(1, 0, 2, 1),
            box(1, 1, 2, 2),
        ],
        crs="EPSG:3857",
    )
    wrong: nx.Graph[int] = nx.Graph([(0, 2)])
    wrong.add_node(1)
    correct: nx.Graph[int] = nx.Graph([(0, 1), (1, 2)])

    wrong_evaluator = PlanEvaluator(wrong, geometry=frame).add_metric(PolsbyPopper())
    correct_evaluator = PlanEvaluator(correct, geometry=frame).add_metric(PolsbyPopper())
    assignment = dict(frame["district"])
    wrong_values = wrong_evaluator.evaluate(assignment)["polsby_popper"]
    correct_values = correct_evaluator.evaluate(assignment)["polsby_popper"]

    assert_same(wrong_values, correct_values)
    assert wrong_evaluator._resources is not None
    assert wrong_evaluator._resources.geometry is not None
    native = wrong_evaluator._resources.geometry.native

    wrong_evaluator.add_metric(Reock()).evaluate(assignment)

    assert wrong_evaluator._resources is not None
    assert wrong_evaluator._resources.geometry is not None
    assert wrong_evaluator._resources.geometry.native is native


def test_validated_geometry_snapshot_excludes_unrequested_columns() -> None:
    graph, units, assignment = resources()
    units["unused"] = ["discard"] * len(units)
    evaluator = PlanEvaluator(graph, geometry=units, node_id_column="node").add_metric(Reock())

    evaluator.evaluate(assignment)

    assert evaluator._resources is not None
    assert evaluator._resources.geometry is not None
    assert list(evaluator._resources.geometry.frame.columns) == [units.geometry.name]


def test_geometry_can_be_nonprojected_until_geometry_is_requested() -> None:
    graph, units, _ = resources()
    geographic = units.to_crs("EPSG:4326")
    evaluator = PlanEvaluator(graph, geometry=geographic, node_id_column="node").add_metric(
        Tally("population")
    )
    assert isinstance(evaluator.evaluate([0, 0, 1, 1])["population"], pd.Series)
    assert evaluator._resources is not None
    assert evaluator._resources.alignment is not None
    assert evaluator._resources.geometry is None
    assert ("geometry", "geometry") not in evaluator._resources.node_columns
    reprojected = PlanEvaluator(
        graph,
        geometry=geographic,
        node_id_column="node",
        target_crs="EPSG:5070",
    ).add_metric(Tally("population"))
    assert_same(
        reprojected.evaluate([0, 0, 1, 1])["population"],
        evaluator.evaluate([0, 0, 1, 1])["population"],
    )
    assert reprojected._resources is not None
    assert reprojected._resources.geometry is None

    with pytest.raises(ValueError, match="projected CRS"):
        PlanEvaluator(graph, geometry=geographic, node_id_column="node").add_metric(
            Reock()
        ).evaluate([0, 0, 1, 1])


def test_active_geometry_column_is_reserved_for_geometry_backed_metrics() -> None:
    graph, units, _ = resources()
    evaluator = PlanEvaluator(graph, geometry=units, node_id_column="node").add_metric(
        RegionSplits(str(units.geometry.name))
    )

    with pytest.raises(ValueError, match="cannot be used as a graph attribute"):
        evaluator.evaluate([0, 0, 1, 1])
    assert evaluator._resources is None
    assert evaluator._engine is None


def test_add_geometry_must_precede_metric_registration_without_mutating_evaluator() -> None:
    graph, units, _ = resources()
    evaluator = PlanEvaluator(graph).add_metric(Tally("population"))

    with pytest.raises(RuntimeError, match="before the first metric"):
        evaluator.add_geometry(units, node_id_column="node")

    values = evaluator.evaluate([0, 0, 1, 1])["population"]
    assert isinstance(values, pd.Series)
    assert values.sum() == 4_000


def test_add_geometry_validates_identifiers_and_snapshots_values() -> None:
    graph, units, _ = resources()

    with pytest.raises(ValueError, match="does not contain node ID column"):
        PlanEvaluator(graph, geometry=units, node_id_column="missing").add_metric(
            Tally("population")
        ).evaluate([0, 0, 1, 1])
    with pytest.raises(ValueError, match="must exactly match graph nodes"):
        PlanEvaluator(
            graph,
            geometry=cast(gpd.GeoDataFrame, units.iloc[:-1]),
            node_id_column="node",
        ).add_metric(Tally("population")).evaluate([0, 0, 1, 1])
    duplicated = units.copy()
    duplicated.loc[duplicated.index[-1], "node"] = "d"
    with pytest.raises(ValueError, match="must be unique"):
        PlanEvaluator(graph, geometry=duplicated, node_id_column="node").add_metric(
            Tally("population")
        ).evaluate([0, 0, 1, 1])

    evaluator = PlanEvaluator(graph, geometry=units, node_id_column="node").add_metric(
        Tally("population")
    )
    units["population"] = 0
    result = evaluator.evaluate([0, 0, 1, 1])["population"]
    assert isinstance(result, pd.Series)
    assert result.sum() == 0
    units["population"] = 100
    cached = evaluator.evaluate([0, 0, 1, 1])["population"]
    assert isinstance(cached, pd.Series)
    assert cached.sum() == 0

    with pytest.raises(RuntimeError, match="already been added"):
        PlanEvaluator(graph, geometry=units, node_id_column="node").add_geometry(
            units,
            node_id_column="node",
        )
