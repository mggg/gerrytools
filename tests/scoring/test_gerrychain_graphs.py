"""GerryChain graph inputs preserve scoring labels, attributes, and topology."""

from typing import Any, cast

import geopandas as gpd
import networkx as nx
import numpy as np
import pytest
import rustworkx as rx
from gerrychain import Graph, Partition
from gerrychain.graph import FrozenGraph
from shapely.geometry import box

from gerrytools.scoring import CutEdges, PlanEvaluator, Reock, Tally, cut_edges, tally


def graph_fixture() -> nx.Graph:
    graph = nx.path_graph(["west", ("middle", 1), 42])
    nx.set_node_attributes(graph, dict(zip(graph, [10, 20, 30], strict=True)), "population")
    nx.set_edge_attributes(graph, {edge: 2 for edge in graph.edges}, "weight")
    return graph


def as_gerrychain(graph: nx.Graph, backend: str) -> Graph | FrozenGraph:
    source = Graph.from_networkx(graph)
    if backend != "networkx":
        source = source.convert_from_nx_to_rx()
    return FrozenGraph(source) if backend == "frozen" else source


@pytest.mark.parametrize("backend", ["networkx", "rustworkx", "frozen"])
def test_gerrychain_graphs_match_networkx_and_single_plan_functions(backend: str) -> None:
    graph = graph_fixture()
    source = as_gerrychain(graph, backend)
    assignment = dict(zip(graph, [0, 0, 1], strict=True))
    scorer = PlanEvaluator(source).add_metric(Tally("population")).add_metric(CutEdges("weight"))
    expected = PlanEvaluator(graph).add_metric(Tally("population")).add_metric(CutEdges("weight"))
    for plan in (assignment, [0, 0, 1]):
        result = scorer.evaluate(plan)
        np.testing.assert_array_equal(result["population"], expected.evaluate(plan)["population"])
        assert result["cut_edges"] == 2
    np.testing.assert_array_equal(tally(source, assignment, columns="population"), [30, 30])
    assert cut_edges(source, assignment, weight_attr="weight") == 2
    partition = Partition(Graph.from_networkx(graph), assignment)
    np.testing.assert_array_equal(scorer.evaluate(partition)["population"], [30, 30])


@pytest.mark.parametrize("backend", ["networkx", "rustworkx", "frozen"])
def test_gerrychain_graph_geometry_matches_original_labels(backend: str) -> None:
    graph = graph_fixture()
    frame = gpd.GeoDataFrame(
        {"node": list(graph), "population": [1, 2, 4]},
        geometry=[box(i, 0, i + 1, 1) for i in range(3)],
        crs="EPSG:3857",
    ).iloc[::-1]
    scorer = PlanEvaluator(as_gerrychain(graph, backend), geometry=frame, node_id_column="node")
    scorer.add_metric(Tally("population")).add_metric(Reock())
    result = scorer.evaluate({42: 1, ("middle", 1): 0, "west": 0})
    np.testing.assert_array_equal(result["population"], [3, 4])
    np.testing.assert_allclose(result["reock"], [8 / (5 * np.pi), 2 / np.pi])


def test_rustworkx_subgraph_restores_original_labels() -> None:
    graph = graph_fixture()
    source = Graph.from_networkx(graph).convert_from_nx_to_rx().subgraph([1, 2])
    result = (
        PlanEvaluator(source).add_metric(Tally("population")).evaluate({("middle", 1): 0, 42: 1})
    )
    np.testing.assert_array_equal(result["population"], [20, 30])
    assert cut_edges(source, {("middle", 1): 0, 42: 1}) == 1


def test_native_rustworkx_graph_with_sparse_indices() -> None:
    native = rx.PyGraph(multigraph=False)
    native.add_nodes_from([{"population": 10}, {"population": 20}, {"population": 30}])
    native.remove_node(1)
    native.add_edge(0, 2, {"weight": 5})
    source = Graph.from_rustworkx(native)
    scorer = PlanEvaluator(source).add_metric(Tally("population"))
    np.testing.assert_array_equal(scorer.evaluate({0: 0, 2: 1})["population"], [10, 30])
    assert cut_edges(source, [0, 1], weight_attr="weight") == 5


def test_parallel_edges_are_not_silently_collapsed() -> None:
    native = rx.PyGraph(multigraph=True)
    native.add_nodes_from([{}, {}])
    native.add_edge(0, 1, {})
    native.add_edge(0, 1, {})
    with pytest.raises(ValueError, match="simple|parallel"):
        PlanEvaluator(Graph.from_rustworkx(native))


@pytest.mark.parametrize("graph", [nx.MultiGraph([(0, 1)]), nx.Graph([(0, 0)])])
def test_networkx_wrappers_retain_graph_validation(graph: nx.Graph) -> None:
    with pytest.raises(ValueError, match="undirected|simple|self-loop"):
        PlanEvaluator(Graph.from_networkx(graph))


def test_duplicate_original_labels_are_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    source = Graph.from_networkx(graph_fixture()).convert_from_nx_to_rx()
    monkeypatch.setattr(source, "original_nx_node_id_for_internal_node_id", lambda node: "same")
    with pytest.raises(ValueError, match="original node identifiers must be unique"):
        PlanEvaluator(source)


def test_rustworkx_attributes_are_snapshotted_at_construction() -> None:
    source = Graph.from_networkx(graph_fixture()).convert_from_nx_to_rx()
    scorer = PlanEvaluator(source).add_metric(Tally("population"))
    source.node_data(0)["population"] = 999
    np.testing.assert_array_equal(scorer.evaluate([0, 0, 1])["population"], [30, 30])


def test_unrelated_graph_input_is_rejected() -> None:
    with pytest.raises(TypeError, match="NetworkX or GerryChain"):
        PlanEvaluator(cast(Any, object()))
