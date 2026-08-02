"""Shared GeoDataFrame alignment and validation helpers."""

from collections.abc import Hashable, Iterable
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd


def _object_index(values: Iterable[Hashable], name: str) -> pd.Index:
    """Build an object index without expanding tuple-valued identifiers."""
    labels = tuple(values)
    if not any(isinstance(label, tuple) for label in labels):
        return pd.Index(labels, name=name)
    array = np.empty(len(labels), dtype=object)
    array[:] = labels
    return pd.Index(array, name=name)


def _require_projected_crs(frame: gpd.GeoDataFrame, context: str) -> None:
    """Require a projected CRS so downstream measurements are planar."""
    crs = frame.crs
    if crs is None or not crs.is_projected:
        raise ValueError(f"{context} requires a projected CRS")


def _require_valid_polygons(frame: gpd.GeoDataFrame, name: str) -> None:
    """Reject missing, empty, invalid, or non-polygonal geometries."""
    geometry = frame.geometry
    if geometry.isna().any():
        raise ValueError(f"{name} cannot contain missing geometries")
    if geometry.is_empty.any():
        raise ValueError(f"{name} cannot contain empty geometries")
    if not geometry.is_valid.all():
        raise ValueError(f"{name} geometries must be topologically valid")
    if not geometry.geom_type.isin(("Polygon", "MultiPolygon")).all():
        raise ValueError(f"{name} must contain only Polygon or MultiPolygon geometries")


def _alignment_positions(
    node_order: Iterable[Hashable],
    geometry_nodes: Iterable[Hashable],
    *,
    target_name: str = "node_order",
) -> tuple[int, ...]:
    """Return geometry row positions in node order after validating identifiers."""
    nodes = _object_index(node_order, "node")
    if nodes.empty:
        raise ValueError("node_order cannot be empty")
    if nodes.has_duplicates:
        raise ValueError("node_order cannot contain duplicate identifiers")
    if nodes.isna().any():
        raise ValueError("node_order cannot contain missing identifiers")

    geometry_index = _object_index(geometry_nodes, "node")
    if geometry_index.has_duplicates:
        raise ValueError("geometry node identifiers must be unique")
    if geometry_index.isna().any():
        raise ValueError("geometry node identifiers cannot be missing")

    missing = nodes[~nodes.isin(geometry_index)].tolist()
    unexpected = geometry_index[~geometry_index.isin(nodes)].tolist()
    if missing or unexpected:
        raise ValueError(
            f"geometry node identifiers must exactly match {target_name}; "
            f"missing={missing!r}, unexpected={unexpected!r}"
        )
    return tuple(int(value) for value in geometry_index.get_indexer(nodes))


def _validated_geometry_frame(
    frame: gpd.GeoDataFrame,
    *,
    crs: Any | None = None,
) -> gpd.GeoDataFrame:
    """Transform and validate a geometry frame whose rows are already aligned."""
    if frame.crs is None:
        raise ValueError("geometries must have a CRS")
    if crs is not None:
        frame = frame.to_crs(crs)
    _require_projected_crs(frame, "geometry scoring")
    _require_valid_polygons(frame, "geometries")
    return frame


def aligned_geodataframe(
    node_order: Iterable[Hashable],
    geometries: gpd.GeoDataFrame,
    *,
    node_column: str | None = None,
    crs: Any | None = None,
) -> gpd.GeoDataFrame:
    """Return a validated projected GeoDataFrame in graph-node order.

    Geometry rows are always matched to graph nodes by identifier. When ``node_column`` is
    omitted, the GeoDataFrame index is the identifier. Supplying a GeoDataFrame whose current
    order happens to match the graph is not enough unless its index also contains the node ids.

    Args:
        node_order (Iterable[Hashable]): Graph node identifiers in the order expected by the scoring
            engine.
        geometries (gpd.GeoDataFrame): One geometry row per graph node.
        node_column (str | None): Optional column containing graph node identifiers.
        crs (Any | None): Optional projected CRS to which the geometries are transformed. Without
            this argument, ``geometries`` must already use a projected CRS.

    Returns:
        gpd.GeoDataFrame: An aligned copy of ``geometries``.

    Raises:
        TypeError: If ``geometries`` is not a GeoDataFrame.
        ValueError: If identifiers are missing, duplicated, or do not match the node set; if CRS
            information is absent or geographic; or if a geometry is missing, empty, or invalid.
    """
    if not isinstance(geometries, gpd.GeoDataFrame):
        raise TypeError("geometries must be a GeoDataFrame")

    if node_column is None:
        geometry_nodes = _object_index(geometries.index, "node")
    else:
        if node_column not in geometries.columns:
            raise ValueError(f"geometries does not contain node column {node_column!r}")
        geometry_nodes = _object_index(geometries[node_column], "node")

    positions = _alignment_positions(node_order, geometry_nodes)
    return _validated_geometry_frame(geometries.iloc[list(positions)].copy(), crs=crs)
