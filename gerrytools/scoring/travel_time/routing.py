"""Directed shortest-path routing for travel-time table construction."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Hashable

import networkx as nx
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from .od_table import UnitId


def _json_unit_ids(unit_ids: Sequence[UnitId]) -> list[str | int]:
    serializable: list[str | int] = []
    for unit_id in unit_ids:
        if isinstance(unit_id, str):
            serializable.append(unit_id)
        elif isinstance(unit_id, (int, np.integer)) and not isinstance(unit_id, bool):
            serializable.append(int(unit_id))
        else:
            raise TypeError("Travel-time unit IDs must be strings or integers")
    return serializable


def compute_directed_travel_times(
    graph: nx.Graph,
    unit_to_node: Mapping[UnitId, Hashable],
    *,
    weight: str = "travel_time",
    batch_size: int = 32,
) -> np.ndarray:
    """Compute a directed unit-to-unit matrix with batched SciPy Dijkstra.

    Args:
        graph (nx.Graph): Directed or undirected weighted road graph.
        unit_to_node (Mapping[UnitId, Hashable]): Selected road node for each unit.
        weight (str): Nonnegative edge attribute used as routing cost.
        batch_size (int): Number of source nodes routed in each SciPy call.

    Returns:
        np.ndarray: Directed ``float32`` matrix in ``unit_to_node`` iteration order.

    Raises:
        KeyError: If a selected road node is absent from the graph.
        ValueError: If inputs or routing edge weights are empty or invalid.
    """
    if not unit_to_node:
        raise ValueError("unit_to_node must contain at least one unit")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    missing_nodes = [node for node in unit_to_node.values() if node not in graph]
    if missing_nodes:
        raise KeyError(f"Snap nodes not in graph: {missing_nodes[:5]!r}")

    graph_nodes = list(graph.nodes)
    node_index = {node: index for index, node in enumerate(graph_nodes)}
    edge_weights: dict[tuple[int, int], float] = {}
    for u, v, data in graph.edges(data=True):
        if weight not in data:
            raise ValueError(f"Road edge {(u, v)!r} is missing {weight!r}")
        try:
            value = float(data[weight])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Road edge {(u, v)!r} has nonnumeric {weight!r}") from exc
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"Road edge {(u, v)!r} has invalid {weight!r}: {value!r}")
        key = (node_index[u], node_index[v])
        edge_weights[key] = min(edge_weights.get(key, math.inf), value)
        if not graph.is_directed():
            reverse = (key[1], key[0])
            edge_weights[reverse] = min(edge_weights.get(reverse, math.inf), value)
    if not edge_weights:
        raise ValueError("road graph has no weighted edges")

    rows = [key[0] for key in edge_weights]
    columns = [key[1] for key in edge_weights]
    values = list(edge_weights.values())
    sparse = csr_matrix((values, (rows, columns)), shape=(len(graph_nodes), len(graph_nodes)))

    snap_nodes = list(dict.fromkeys(unit_to_node.values()))
    snap_graph_indices = np.array([node_index[node] for node in snap_nodes], dtype=np.int64)
    snap_matrix = np.full((len(snap_nodes), len(snap_nodes)), np.nan, dtype=np.float32)
    for start in range(0, len(snap_nodes), batch_size):
        stop = min(start + batch_size, len(snap_nodes))
        distances = dijkstra(
            sparse,
            directed=True,
            indices=snap_graph_indices[start:stop],
            return_predecessors=False,
        )
        distances = np.atleast_2d(distances)[:, snap_graph_indices]
        distances[~np.isfinite(distances)] = np.nan
        snap_matrix[start:stop] = distances.astype(np.float32)
    np.fill_diagonal(snap_matrix, 0.0)

    snap_index = {node: index for index, node in enumerate(snap_nodes)}
    unit_snap_indices = np.array([snap_index[node] for node in unit_to_node.values()])
    directed = snap_matrix[np.ix_(unit_snap_indices, unit_snap_indices)]
    np.fill_diagonal(directed, 0.0)
    return directed


def _directional_statistics(directed: np.ndarray) -> tuple[int, float]:
    upper = np.triu_indices_from(directed, k=1)
    forward = directed[upper]
    reverse = directed.T[upper]
    finite = np.isfinite(forward) & np.isfinite(reverse)
    if not finite.any():
        return 0, 0.0
    differences = np.abs(forward[finite] - reverse[finite])
    return int(np.count_nonzero(differences > 1e-8)), float(np.max(differences, initial=0.0))


def _save_directed_audit(
    path: str | Path,
    unit_ids: Sequence[UnitId],
    directed: np.ndarray,
) -> None:
    serializable_ids = _json_unit_ids(unit_ids)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as file:
        np.savez_compressed(
            file,
            unit_ids_json=np.array(json.dumps(serializable_ids, separators=(",", ":"))),
            directed_times=directed.astype(np.float32),
            time_unit=np.array("seconds"),
        )
