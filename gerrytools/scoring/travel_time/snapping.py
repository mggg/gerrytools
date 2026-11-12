"""Representative-point preparation and road-node snapping."""

from __future__ import annotations

import importlib
import math
import warnings
from typing import Hashable, cast

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
from shapely import STRtree
from shapely.geometry import LineString, MultiLineString, MultiPolygon, Point, Polygon

from .od_table import UnitId


def census_representative_points(
    units: pd.DataFrame,
    *,
    id_column: str,
    latitude_column: str = "INTPTLAT20",
    longitude_column: str = "INTPTLON20",
) -> gpd.GeoDataFrame:
    """Create WGS84 points from Census internal-point columns.

    This works for any Census geography carrying the requested columns; it is
    not restricted to VTDs.

    Args:
        units (pd.DataFrame): Census geography rows containing IDs and internal points.
        id_column (str): Unique geographic-unit identifier column.
        latitude_column (str): Census internal-point latitude column.
        longitude_column (str): Census internal-point longitude column.

    Returns:
        gpd.GeoDataFrame: WGS84 point geometries aligned to the input rows.

    Raises:
        KeyError: If a required column is missing.
        ValueError: If IDs or coordinates are empty, duplicated, missing, or invalid.
    """
    required = [id_column, latitude_column, longitude_column]
    missing = [column for column in required if column not in units.columns]
    if missing:
        raise KeyError(f"Representative-point columns not found: {missing!r}")
    if len(units) == 0:
        raise ValueError("units must contain at least one row")

    ids = units[id_column].to_numpy()
    if bool(pd.isna(ids).any()) or len(set(ids.tolist())) != len(ids):
        raise ValueError(f"{id_column!r} must contain unique, non-missing unit IDs")
    try:
        latitudes = np.asarray(pd.to_numeric(units[latitude_column], errors="raise"), dtype=float)
        longitudes = np.asarray(pd.to_numeric(units[longitude_column], errors="raise"), dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("Representative-point coordinates must be numeric") from exc
    if not np.isfinite(latitudes).all() or not np.isfinite(longitudes).all():
        raise ValueError("Representative-point coordinates must be finite")
    if (
        not ((-90.0 <= latitudes) & (latitudes <= 90.0)).all()
        or not ((-180.0 <= longitudes) & (longitudes <= 180.0)).all()
    ):
        raise ValueError("Representative-point coordinates fall outside longitude/latitude bounds")

    return gpd.GeoDataFrame(
        {id_column: ids.tolist()},
        geometry=gpd.points_from_xy(longitudes, latitudes),
        crs="EPSG:4326",
    )


def snap_units_to_graph(
    graph: nx.MultiDiGraph,
    unit_points: gpd.GeoDataFrame,
    *,
    id_column: str,
    unit_polygons: gpd.GeoDataFrame | None = None,
) -> tuple[dict[UnitId, Hashable], dict[UnitId, float]]:
    """Snap representative points to connected graph nodes and report distances.

    Points may use any declared CRS; they are projected to the road-graph CRS
    before snapping. The graph from :func:`build_osm_drive_graph` is already
    restricted to its principal strongly connected component.

    Nodes incident only to service-road edges are ineligible for snapping,
    though routes may traverse those edges. With no ``unit_polygons``, each point
    is snapped to its nearest eligible graph node. When polygons are supplied, a
    unit first uses the contained eligible graph node
    nearest its representative point. If the unit contains no graph node, it
    instead uses the graph node nearest any exterior ring of the unit. Interior
    rings (polygon holes) are deliberately excluded from that fallback. Returned
    distances always measure representative-point-to-node distance, regardless
    of which selection rule was used.

    Args:
        graph (nx.MultiDiGraph): Projected retained road graph.
        unit_points (gpd.GeoDataFrame): Representative point for every unit.
        id_column (str): Unique unit identifier shared by points and polygons.
        unit_polygons (gpd.GeoDataFrame | None): Optional unit boundaries for inside-first
            snapping.

    Returns:
        tuple[dict[UnitId, Hashable], dict[UnitId, float]]: Selected node and point-to-node
            distance in meters for every unit.

    Raises:
        KeyError: If the unit identifier column is missing.
        TypeError: If supplied polygons are not a GeoDataFrame.
        ValueError: If coordinate systems, IDs, geometries, nodes, or distances are invalid.
        ImportError: If point-only snapping needs OSMnx and it is unavailable.
    """
    if id_column not in unit_points.columns:
        raise KeyError(f"Unit ID column not found: {id_column!r}")
    if unit_points.crs is None:
        raise ValueError("unit_points must declare a CRS")
    if graph.graph.get("crs") is None:
        raise ValueError("road graph must declare a CRS")
    ids = unit_points[id_column].to_list()
    if len(ids) == 0 or len(set(ids)) != len(ids) or bool(pd.isna(ids).any()):
        raise ValueError("unit_points must contain unique, non-missing IDs and at least one row")
    projected = unit_points.to_crs(graph.graph["crs"])
    if not all(
        isinstance(geometry, Point) and not geometry.is_empty for geometry in projected.geometry
    ):
        raise ValueError("unit_points geometry must contain nonempty Point values")
    xs = projected.geometry.x.to_numpy(dtype=float)
    ys = projected.geometry.y.to_numpy(dtype=float)
    if not np.isfinite(xs).all() or not np.isfinite(ys).all():
        raise ValueError("unit_points coordinates must be finite")

    if unit_polygons is not None:
        return _snap_units_with_polygons(
            graph,
            ids,
            xs,
            ys,
            unit_polygons,
            id_column=id_column,
        )

    try:
        ox = importlib.import_module("osmnx")
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("Install gerrytools[travel-time] to snap units to an OSM graph") from exc
    snap_ineligible = frozenset(graph.graph.get("snap_ineligible_nodes", ()))
    snap_graph = (
        graph
        if not snap_ineligible
        else nx.subgraph_view(graph, filter_node=lambda node: node not in snap_ineligible)
    )
    if snap_ineligible and snap_graph.number_of_nodes() == 0:
        raise ValueError("road graph contains no snap-eligible nodes")
    nearest, distances = ox.distance.nearest_nodes(
        snap_graph,
        X=xs,
        Y=ys,
        return_dist=True,
    )
    nodes = np.atleast_1d(nearest).tolist()
    distance_values = np.atleast_1d(distances).astype(float)
    if not np.isfinite(distance_values).all():
        raise ValueError("one or more units could not be snapped to a finite graph distance")
    return dict(zip(ids, nodes)), dict(zip(ids, distance_values.tolist()))


def _exterior_boundary(geometry: Polygon | MultiPolygon) -> LineString | MultiLineString:
    """Return only a polygon's exterior rings, including every multipart piece."""
    if isinstance(geometry, Polygon):
        return LineString(geometry.exterior.coords)
    return MultiLineString([LineString(part.exterior.coords) for part in geometry.geoms])


def _snap_units_with_polygons(
    graph: nx.MultiDiGraph,
    ids: list[UnitId],
    anchor_xs: np.ndarray,
    anchor_ys: np.ndarray,
    unit_polygons: gpd.GeoDataFrame,
    *,
    id_column: str,
) -> tuple[dict[UnitId, Hashable], dict[UnitId, float]]:
    """Apply inside-first node snapping with an exterior-boundary fallback."""
    if not isinstance(unit_polygons, gpd.GeoDataFrame):
        raise TypeError("unit_polygons must be a GeoDataFrame")
    if id_column not in unit_polygons.columns:
        raise KeyError(f"Unit ID column not found in unit_polygons: {id_column!r}")
    if unit_polygons.crs is None:
        raise ValueError("unit_polygons must declare a CRS")
    polygon_ids = unit_polygons[id_column].to_list()
    if (
        len(polygon_ids) == 0
        or len(set(polygon_ids)) != len(polygon_ids)
        or bool(pd.isna(polygon_ids).any())
    ):
        raise ValueError("unit_polygons must contain unique, non-missing IDs and at least one row")
    if set(polygon_ids) != set(ids):
        raise ValueError("unit_points and unit_polygons must contain exactly the same unit IDs")

    projected = unit_polygons.to_crs(graph.graph["crs"]).set_index(id_column)
    valid_polygons = projected.geometry.map(
        lambda geometry: (
            isinstance(geometry, (Polygon, MultiPolygon))
            and not geometry.is_empty
            and geometry.is_valid
        )
    )
    if not bool(valid_polygons.all()):
        raise ValueError(
            "unit_polygons geometry must contain valid, nonempty Polygon or MultiPolygon values"
        )

    snap_ineligible = frozenset(graph.graph.get("snap_ineligible_nodes", ()))
    node_ids = [node for node in graph.nodes if node not in snap_ineligible]
    if not node_ids:
        raise ValueError("road graph must contain at least one snap-eligible node")
    node_xs = np.array([graph.nodes[node].get("x") for node in node_ids], dtype=float)
    node_ys = np.array([graph.nodes[node].get("y") for node in node_ids], dtype=float)
    if not np.isfinite(node_xs).all() or not np.isfinite(node_ys).all():
        raise ValueError("road graph node coordinates must be finite")
    node_points = gpd.points_from_xy(node_xs, node_ys)
    tree = STRtree(node_points)

    unit_to_node: dict[UnitId, Hashable] = {}
    anchor_distances: dict[UnitId, float] = {}
    fallback_ids: list[UnitId] = []
    for unit_id, anchor_x, anchor_y in zip(ids, anchor_xs, anchor_ys):
        polygon = cast(Polygon | MultiPolygon, projected.loc[unit_id].geometry)
        contained = np.asarray(tree.query(polygon, predicate="intersects"), dtype=int)
        candidates = contained
        if candidates.size == 0:
            fallback_ids.append(unit_id)
            nearest, _ = tree.query_nearest(
                _exterior_boundary(polygon),
                all_matches=True,
                return_distance=True,
            )
            candidates = np.atleast_1d(nearest).astype(int)

        squared_anchor_distances = (node_xs[candidates] - anchor_x) ** 2 + (
            node_ys[candidates] - anchor_y
        ) ** 2
        minimum_distance = float(squared_anchor_distances.min())
        tied = candidates[
            np.isclose(squared_anchor_distances, minimum_distance, rtol=0.0, atol=1e-12)
        ]
        chosen_index = min(tied.tolist(), key=lambda index: repr(node_ids[index]))
        distance = math.hypot(node_xs[chosen_index] - anchor_x, node_ys[chosen_index] - anchor_y)
        if not math.isfinite(distance):
            raise ValueError("one or more units could not be snapped to a finite graph distance")
        unit_to_node[unit_id] = node_ids[chosen_index]
        anchor_distances[unit_id] = float(distance)

    if fallback_ids:
        preview = fallback_ids[:5]
        warnings.warn(
            f"{len(fallback_ids)} unit(s) contain no road node in the principal strongly "
            "connected component and used the nearest exterior-boundary fallback; "
            f"examples={preview!r}.",
            UserWarning,
            stacklevel=2,
        )
    return unit_to_node, anchor_distances
