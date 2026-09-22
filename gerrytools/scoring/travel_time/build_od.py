"""Build a TravelTimeTable from a routable NetworkX graph.

This is the thin bridge between a road network and scoring. It does not
download OSM; tests and callers supply a graph whose edges already have a
``travel_time`` (or other) weight attribute.
"""

from __future__ import annotations

from typing import Any, Hashable, Mapping

import networkx as nx
import numpy as np

from .od_table import TravelTimeTable


def build_od_from_graph(
    graph: nx.Graph,
    unit_to_node: Mapping[Any, Any],
    *,
    weight: str = "travel_time",
) -> TravelTimeTable:
    """All-pairs shortest-path times between units via their snapped graph nodes.

    Args:
        graph: Directed or undirected graph with numeric ``weight`` on edges.
        unit_to_node: Map each unit id → a node id present in ``graph``.
        weight: Edge attribute used as path length (default ``travel_time``).

    Returns:
        TravelTimeTable with unit order matching ``unit_to_node``'s insertion
        order (Python 3.7+ dict order). Unreachable pairs are ``NaN``.
    """
    unit_ids = list(unit_to_node.keys())
    snap_nodes = [unit_to_node[u] for u in unit_ids]
    missing = [n for n in snap_nodes if n not in graph]
    if missing:
        raise KeyError(f"Snap nodes not in graph: {missing!r}")

    # networkx silently treats an edge missing the `weight` attribute as
    # weight 1 rather than raising, which would quietly corrupt every travel
    # time computed through that edge (e.g. a `travel_time`-weighted graph
    # with one un-annotated edge would treat it as a 1-second hop). Fail
    # loudly instead.
    unweighted_edges = [(u, v) for u, v, data in graph.edges(data=True) if weight not in data]
    if unweighted_edges:
        raise ValueError(
            f"{len(unweighted_edges)} edge(s) missing the {weight!r} attribute "
            f"(e.g. {unweighted_edges[:5]!r}); networkx would silently treat "
            "these as weight 1, so refusing to build the table."
        )

    n = len(unit_ids)
    times = np.full((n, n), np.nan, dtype=float)
    np.fill_diagonal(times, 0.0)

    # Unique snaps: several units may share a node.
    unique_snaps = list(dict.fromkeys(snap_nodes))
    snap_od: dict[Hashable, dict[Hashable, float]] = {}
    for source in unique_snaps:
        lengths = nx.single_source_dijkstra_path_length(graph, source, weight=weight)
        snap_od[source] = {t: float(lengths[t]) for t in unique_snaps if t in lengths}

    for i, si in enumerate(snap_nodes):
        for j, sj in enumerate(snap_nodes):
            if i == j:
                continue
            t = snap_od.get(si, {}).get(sj)
            if t is not None:
                times[i, j] = t

    return TravelTimeTable.from_matrix(unit_ids, times)
