"""Graph validation, canonicalization, and comparison helpers for :mod:`.recorded_chain`.

These cover everything RecordedChain does to a graph before and after a run: taking ownership
of the user's graph, reordering or JSON-canonicalizing it for encoding, and the structural
equality checks that guard the recorded file against mid-run mutation.
"""

from __future__ import annotations

import copy
import json
from typing import Any, Literal

import networkx as nx
import numpy as np
from binary_ensemble.graph import reorder
from gerrychain import Graph, Partition
from gerrychain.graph import FrozenGraph

GraphOrderName = Literal["mlc", "rcm", "key"]
GraphOrder = GraphOrderName | None

_RESERVED_NODE_KEY = "__networkx_node__"


def _normalize_numpy(value: Any) -> Any:
    """Recursively convert NumPy scalars in graph data to plain Python values."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _normalize_numpy(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_normalize_numpy(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_normalize_numpy(item) for item in value)
    return value


def _source_graph(graph: nx.Graph | Graph) -> nx.Graph:
    """Validate the user's graph and deep-copy it into a chain-owned NetworkX graph.

    Accepts a NetworkX graph or a NetworkX-backed GerryChain ``Graph``. The copy's graph, node,
    and edge attributes are normalized to plain Python scalars so JSON canonicalization and BENDL
    serialization later see exactly the values the chain holds.

    Raises:
        TypeError: If ``graph`` is frozen, not NetworkX-backed, directed, or a multigraph.
        ValueError: If a node or edge uses an attribute reserved by adjacency serialization.
    """
    if isinstance(graph, FrozenGraph):
        raise TypeError("RecordedChain does not accept a frozen GerryChain graph")
    if isinstance(graph, Graph):
        if not graph.is_nx_graph():
            raise TypeError("RecordedChain requires a NetworkX-backed GerryChain graph")
        graph = graph.get_nx_graph()
    if not isinstance(graph, nx.Graph):
        raise TypeError("RecordedChain requires a NetworkX graph")
    if graph.is_directed() or graph.is_multigraph():
        raise TypeError("RecordedChain requires a simple undirected NetworkX graph")
    for node, attributes in graph.nodes(data=True):
        marker = attributes.get(_RESERVED_NODE_KEY, node)
        if marker != node:
            raise ValueError(f"Node {node!r} uses reserved attribute {_RESERVED_NODE_KEY!r}")
        if "id" in attributes:
            raise ValueError(f"Node {node!r} uses reserved attribute 'id'")
    for u, v, attributes in graph.edges(data=True):
        if "id" in attributes:
            raise ValueError(f"Edge {(u, v)!r} uses reserved attribute 'id'")

    owned = copy.deepcopy(graph)
    owned.graph.update(_normalize_numpy(dict(owned.graph)))
    for _, attributes in owned.nodes(data=True):
        attributes.pop(_RESERVED_NODE_KEY, None)
        normalized = _normalize_numpy(dict(attributes))
        attributes.clear()
        attributes.update(normalized)
    for _, _, attributes in owned.edges(data=True):
        normalized = _normalize_numpy(dict(attributes))
        attributes.clear()
        attributes.update(normalized)
    return owned


def _prepare_graph(source: nx.Graph, order: GraphOrder, key: str | None) -> nx.Graph:
    """Reorder ``source`` for encoding, or JSON-canonicalize it when no order is requested.

    The ``order=None`` round trip through ``nx.adjacency_data`` applies the same value coercions
    BENDL storage does (tuples to lists, dict keys to strings), so the in-memory graph matches
    what a decoder will read back from the file.
    """
    try:
        if order is None:
            data = json.loads(json.dumps(nx.adjacency_data(source)))
            return nx.adjacency_graph(data)
        prepared, _ = reorder(source, sort=order, key=key)
        return prepared
    except BaseException as exc:
        stage = "no-order canonicalization" if order is None else f"{order!r} graph reordering"
        exc.add_note(f"RecordedChain failed during {stage}.")
        raise


def _comparable_graph(
    graph: nx.Graph,
    *,
    graph_attributes: bool = True,
    edge_attributes: bool = True,
) -> nx.Graph:
    """Rebuild ``graph`` for equality comparison, dropping the reserved node marker.

    ``graph_attributes=False`` and ``edge_attributes=False`` exclude those attribute sets from
    the comparison copy; the file-side checks exclude edge values because proposals may write
    scratch attributes into shared edge dicts (see ``RecordedChain._record``).
    """
    comparable = nx.Graph()
    if graph_attributes:
        comparable.graph.update(graph.graph)
    for node, attributes in graph.nodes(data=True):
        clean = {key: value for key, value in attributes.items() if key != _RESERVED_NODE_KEY}
        comparable.add_node(node, **clean)
    for u, v, attributes in graph.edges(data=True):
        comparable.add_edge(u, v, **(attributes if edge_attributes else {}))
    return comparable


def _assert_graph_equal(
    actual: nx.Graph,
    expected: nx.Graph,
    context: str,
    *,
    graph_attributes: bool = True,
    edge_attributes: bool = True,
) -> None:
    """Raise ``RuntimeError`` naming ``context`` unless the two graphs match.

    Node order matters: assignment vectors are positional, so two otherwise-equal graphs whose
    nodes enumerate differently are not interchangeable.
    """
    actual_clean = _comparable_graph(
        actual, graph_attributes=graph_attributes, edge_attributes=edge_attributes
    )
    expected_clean = _comparable_graph(
        expected, graph_attributes=graph_attributes, edge_attributes=edge_attributes
    )
    if list(actual_clean.nodes) != list(expected_clean.nodes) or not nx.utils.graphs_equal(
        actual_clean, expected_clean
    ):
        raise RuntimeError(f"{context} does not match RecordedChain.graph")


def _assert_original_edge_attributes_unchanged(
    actual: nx.Graph, expected: nx.Graph, context: str
) -> None:
    """Require every original edge attribute while permitting new proposal scratch keys."""
    missing = object()
    for u, v, expected_attributes in expected.edges(data=True):
        actual_attributes = actual.edges[u, v]
        changed = any(
            actual_attributes.get(key, missing) != value
            for key, value in expected_attributes.items()
        )
        if changed:
            raise RuntimeError(f"{context} does not match RecordedChain.graph")


def _execution_graph(partition: Partition) -> nx.Graph:
    """Reconstruct the NetworkX view of the graph ``partition`` actually executed on.

    Node and edge data are read back out of the partition's internal (rustworkx) storage under
    the original NetworkX node ids, so the result compares against ``RecordedChain.graph``.
    """
    graph = partition.graph.graph
    result = nx.Graph()
    for internal_id in graph.nodes:
        original_id = graph.original_nx_node_id_for_internal_node_id(internal_id)
        result.add_node(original_id, **dict(graph.node_data(internal_id)))
    for edge_id in graph.edge_indices:
        u, v = graph.get_edge_from_edge_id(edge_id)
        original_u = graph.original_nx_node_id_for_internal_node_id(u)
        original_v = graph.original_nx_node_id_for_internal_node_id(v)
        result.add_edge(original_u, original_v, **dict(graph.edge_data(edge_id)))
    return result
