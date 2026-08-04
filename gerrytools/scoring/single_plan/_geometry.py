"""Compactness and fixed-region functions for one plan."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import networkx as nx
from geopandas import GeoDataFrame
from gerrychain import Partition
from pandas import DataFrame, Series
from shapely.geometry.base import BaseGeometry

from ..evaluator import Assignment
from ..metrics import (
    ConvexHullRatio,
    CutEdges,
    PolsbyPopper,
    PopulationPolygon,
    RegionParts,
    RegionPieces,
    RegionSplits,
    Reock,
    Schwartzberg,
    StateClippedConvexHullRatio,
    TallyByRegion,
)
from ._base import GeoAssignment, SinglePlanSource, _columns, _evaluate, _expect


def cut_edges(
    source: Partition | nx.Graph,
    assignment: Assignment | None = None,
    *,
    weight_attr: str | None = None,
) -> int | float:
    """Count or weight cut edges in one partition or graph assignment.

    Args:
        source (Partition | nx.Graph): Partition or graph providing the plan topology.
        assignment (Assignment | None): Graph assignment mapping or node-ordered sequence. Omit
            for a Partition, which supplies its own assignment.
        weight_attr (str | None): Optional numeric edge attribute to sum over cut edges. When
            omitted, each cut edge contributes one.

    Returns:
        int | float: Cut-edge count when ``weight_attr`` is None; otherwise the summed edge weight.
    """
    return _expect(
        (int, float),
        _evaluate(source, assignment, CutEdges(weight_attr), topology_required=True),
    )


def polsby_popper(
    source: SinglePlanSource,
    assignment: GeoAssignment | None = None,
    *,
    geometry: GeoDataFrame | None = None,
    node_id_column: str | None = None,
    target_crs: Any | None = None,
    area_attr: str | None = None,
    perimeter_attr: str | None = None,
    boundary_perimeter_attr: str | None = None,
    shared_perimeter_attr: str | None = None,
) -> Series:
    """Calculate Polsby-Popper from partition graph measurements or geometry.

    ``polsby_popper(partition)`` uses graph measurements.
    ``polsby_popper(graph, assignment)`` uses graph measurements.
    ``polsby_popper(partition, geometry=frame)`` and ``polsby_popper(frame, assignment)`` use
    geometry.

    Args:
        source (SinglePlanSource): Partition, graph, or authoritative GeoDataFrame.
        assignment (GeoAssignment | None): District assignment for a graph or GeoDataFrame source.
            Omit for a Partition, which supplies its own assignment.
        geometry (GeoDataFrame | None, optional): Authoritative GeoDataFrame for a graph or
            Partition source. Defaults to None. Do not supply this when ``source`` is already a
            GeoDataFrame.
        node_id_column (str | None, optional): Geometry column containing the graph's node
            labels. Defaults to None.
        target_crs (Any | None, optional): Projected CRS to which geometry is transformed before
            scoring. Defaults to None.
        area_attr (str | None): Graph node attribute containing unit areas. Defaults to
            ``"area"`` for graph-backed scoring.
        perimeter_attr (str | None, optional): Graph node attribute containing total unit
            perimeters. Defaults to None. Mutually exclusive with ``boundary_perimeter_attr``.
        boundary_perimeter_attr (str | None): Graph node attribute containing exterior boundary
            perimeter. Defaults to ``"boundary_perim"`` when ``perimeter_attr`` is omitted.
        shared_perimeter_attr (str | None): Graph edge attribute containing shared boundary
            lengths. Defaults to ``"shared_perim"`` for graph-backed scoring.

    Returns:
        Series: District-indexed Polsby-Popper scores.
    """
    metric = PolsbyPopper(
        area_attr=area_attr,
        perimeter_attr=perimeter_attr,
        boundary_perimeter_attr=boundary_perimeter_attr,
        shared_perimeter_attr=shared_perimeter_attr,
    )
    return _expect(
        Series,
        _evaluate(
            source,
            assignment,
            metric,
            geometry=geometry,
            node_id_column=node_id_column,
            target_crs=target_crs,
        ),
    )


def schwartzberg(
    source: SinglePlanSource,
    assignment: GeoAssignment | None = None,
    *,
    geometry: GeoDataFrame | None = None,
    node_id_column: str | None = None,
    target_crs: Any | None = None,
    area_attr: str | None = None,
    perimeter_attr: str | None = None,
    boundary_perimeter_attr: str | None = None,
    shared_perimeter_attr: str | None = None,
) -> Series:
    """Calculate Schwartzberg from partition graph measurements or geometry.

    Source selection and graph-column defaults match :func:`polsby_popper`.

    Args:
        source (SinglePlanSource): Partition, graph, or authoritative GeoDataFrame.
        assignment (GeoAssignment | None): District assignment for a graph or GeoDataFrame source.
            Omit for a Partition, which supplies its own assignment.
        geometry (GeoDataFrame | None, optional): Authoritative GeoDataFrame for a graph or
            Partition source. Defaults to None. Do not supply this when ``source`` is already a
            GeoDataFrame.
        node_id_column (str | None, optional): Geometry column containing the graph's node
            labels. Defaults to None.
        target_crs (Any | None, optional): Projected CRS to which geometry is transformed before
            scoring. Defaults to None.
        area_attr (str | None): Graph node attribute containing unit areas. Defaults to
            ``"area"`` for graph-backed scoring.
        perimeter_attr (str | None, optional): Graph node attribute containing total unit
            perimeters. Defaults to None. Mutually exclusive with ``boundary_perimeter_attr``.
        boundary_perimeter_attr (str | None): Graph node attribute containing exterior boundary
            perimeter. Defaults to ``"boundary_perim"`` when ``perimeter_attr`` is omitted.
        shared_perimeter_attr (str | None): Graph edge attribute containing shared boundary
            lengths. Defaults to ``"shared_perim"`` for graph-backed scoring.

    Returns:
        Series: District-indexed Schwartzberg scores.
    """
    metric = Schwartzberg(
        area_attr=area_attr,
        perimeter_attr=perimeter_attr,
        boundary_perimeter_attr=boundary_perimeter_attr,
        shared_perimeter_attr=shared_perimeter_attr,
    )
    return _expect(
        Series,
        _evaluate(
            source,
            assignment,
            metric,
            geometry=geometry,
            node_id_column=node_id_column,
            target_crs=target_crs,
        ),
    )


def reock(
    source: SinglePlanSource,
    assignment: GeoAssignment | None = None,
    *,
    geometry: GeoDataFrame | None = None,
    node_id_column: str | None = None,
    target_crs: Any | None = None,
) -> Series:
    """Calculate Reock compactness for one plan using geometry.

    Args:
        source (SinglePlanSource): Partition, graph, or authoritative GeoDataFrame.
        assignment (GeoAssignment | None): District assignment. Required for graph and
            GeoDataFrame sources and omitted for a Partition. A GeoDataFrame also accepts an
            assignment column name.
        geometry (GeoDataFrame | None, optional): Authoritative geometry for a graph or Partition
            source. Defaults to None and is required unless ``source`` is a GeoDataFrame.
        node_id_column (str | None, optional): Geometry column containing the graph's node
            labels. Defaults to None.
        target_crs (Any | None, optional): Projected CRS to which geometry is transformed before
            scoring. Defaults to None.

    Returns:
        Series: District-indexed ratios of district area to minimum enclosing-circle area.
    """
    return _expect(
        Series,
        _evaluate(
            source,
            assignment,
            Reock(),
            geometry=geometry,
            node_id_column=node_id_column,
            target_crs=target_crs,
        ),
    )


def convex_hull_ratio(
    source: SinglePlanSource,
    assignment: GeoAssignment | None = None,
    *,
    geometry: GeoDataFrame | None = None,
    node_id_column: str | None = None,
    target_crs: Any | None = None,
) -> Series:
    """Calculate convex-hull compactness for one plan using geometry.

    Args:
        source (SinglePlanSource): Partition, graph, or authoritative GeoDataFrame.
        assignment (GeoAssignment | None): District assignment. Required for graph and
            GeoDataFrame sources and omitted for a Partition. A GeoDataFrame also accepts an
            assignment column name.
        geometry (GeoDataFrame | None, optional): Authoritative geometry for a graph or Partition
            source. Defaults to None and is required unless ``source`` is a GeoDataFrame.
        node_id_column (str | None, optional): Geometry column containing the graph's node
            labels. Defaults to None.
        target_crs (Any | None, optional): Projected CRS to which geometry is transformed before
            scoring. Defaults to None.

    Returns:
        Series: District-indexed ratios of district area to convex-hull area.
    """
    return _expect(
        Series,
        _evaluate(
            source,
            assignment,
            ConvexHullRatio(),
            geometry=geometry,
            node_id_column=node_id_column,
            target_crs=target_crs,
        ),
    )


def state_clipped_convex_hull_ratio(
    source: SinglePlanSource,
    assignment: GeoAssignment | None = None,
    *,
    state_geometry: BaseGeometry,
    geometry: GeoDataFrame | None = None,
    node_id_column: str | None = None,
    target_crs: Any | None = None,
) -> Series:
    """Calculate state-clipped convex-hull compactness for one plan.

    Args:
        source (SinglePlanSource): Partition, graph, or authoritative GeoDataFrame.
        assignment (GeoAssignment | None): District assignment. Required for graph and
            GeoDataFrame sources and omitted for a Partition. A GeoDataFrame also accepts an
            assignment column name.
        state_geometry (BaseGeometry): Nonempty, valid Polygon or MultiPolygon covering every
            scoring unit and using the same projected CRS.
        geometry (GeoDataFrame | None, optional): Authoritative geometry for a graph or Partition
            source. Defaults to None and is required unless ``source`` is a GeoDataFrame.
        node_id_column (str | None, optional): Geometry column containing the graph's node
            labels. Defaults to None.
        target_crs (Any | None, optional): Projected CRS to which unit geometry is transformed
            before scoring. Defaults to None.

    Returns:
        Series: District-indexed ratios of district area to state-clipped convex-hull area.
    """
    return _expect(
        Series,
        _evaluate(
            source,
            assignment,
            StateClippedConvexHullRatio(state_geometry),
            geometry=geometry,
            node_id_column=node_id_column,
            target_crs=target_crs,
        ),
    )


def population_polygon(
    source: SinglePlanSource,
    assignment: GeoAssignment | None = None,
    *,
    population_col: str,
    population_units: GeoDataFrame | None = None,
    geometry: GeoDataFrame | None = None,
    node_id_column: str | None = None,
    target_crs: Any | None = None,
) -> Series:
    """Calculate population-polygon compactness for one plan.

    Args:
        source (SinglePlanSource): Partition, graph, or authoritative GeoDataFrame.
        assignment (GeoAssignment | None): District assignment. Required for graph and
            GeoDataFrame sources and omitted for a Partition. A GeoDataFrame also accepts an
            assignment column name.
        population_col (str): Column containing finite, nonnegative population weights.
        population_units (GeoDataFrame | None): Optional finer projected population polygons
            containing ``population_col``. When omitted, weights and polygons come from the
            authoritative scoring geometry.
        geometry (GeoDataFrame | None, optional): Authoritative geometry for a graph or Partition
            source. Defaults to None and is required unless ``source`` is a GeoDataFrame.
        node_id_column (str | None, optional): Geometry column containing the graph's node
            labels. Defaults to None.
        target_crs (Any | None, optional): Projected CRS for scoring geometry. Defaults to None.

    Returns:
        Series: District-indexed population-polygon compactness scores.
    """
    return _expect(
        Series,
        _evaluate(
            source,
            assignment,
            PopulationPolygon(population_col, population_units=population_units),
            geometry=geometry,
            node_id_column=node_id_column,
            target_crs=target_crs,
        ),
    )


def region_splits(
    source: SinglePlanSource,
    assignment: GeoAssignment | None = None,
    *,
    region_attrs: str | Iterable[str],
    geometry: GeoDataFrame | None = None,
    node_id_column: str | None = None,
) -> int | Series:
    """Count split fixed regions for one plan.

    Args:
        source (SinglePlanSource): Partition, graph, or authoritative GeoDataFrame containing the
            fixed-region columns.
        assignment (GeoAssignment | None): District assignment. Required for graph and
            GeoDataFrame sources and omitted for a Partition. A GeoDataFrame also accepts an
            assignment column name.
        region_attrs (str | Iterable[str]): One fixed-region column, or several to score together.
        geometry (GeoDataFrame | None, optional): Authoritative GeoDataFrame for a graph or
            Partition source. Defaults to None.
        node_id_column (str | None, optional): Geometry column containing the graph's node
            labels. Defaults to None.

    Returns:
        int | Series: Split count for one region column, or a Series keyed by column name for
            several.
    """
    return _expect(
        (int, Series),
        _evaluate(
            source,
            assignment,
            RegionSplits(*_columns(region_attrs)),
            geometry=geometry,
            node_id_column=node_id_column,
        ),
    )


def region_pieces(
    source: SinglePlanSource,
    assignment: GeoAssignment | None = None,
    *,
    region_attrs: str | Iterable[str],
    geometry: GeoDataFrame | None = None,
    node_id_column: str | None = None,
) -> int | Series:
    """Count occupied fixed-region and district pairs for one plan.

    Args:
        source (SinglePlanSource): Partition, graph, or authoritative GeoDataFrame containing the
            fixed-region columns.
        assignment (GeoAssignment | None): District assignment. Required for graph and
            GeoDataFrame sources and omitted for a Partition. A GeoDataFrame also accepts an
            assignment column name.
        region_attrs (str | Iterable[str]): One fixed-region column, or several to score together.
        geometry (GeoDataFrame | None, optional): Authoritative GeoDataFrame for a graph or
            Partition source. Defaults to None.
        node_id_column (str | None, optional): Geometry column containing the graph's node
            labels. Defaults to None.

    Returns:
        int | Series: Pair count for one region column, or a Series keyed by column name for
            several.
    """
    return _expect(
        (int, Series),
        _evaluate(
            source,
            assignment,
            RegionPieces(*_columns(region_attrs)),
            geometry=geometry,
            node_id_column=node_id_column,
        ),
    )


def region_parts(
    source: Partition | nx.Graph,
    assignment: Assignment | None = None,
    *,
    region_attrs: str | Iterable[str],
    geometry: GeoDataFrame | None = None,
    node_id_column: str | None = None,
) -> int | Series:
    """Count connected fixed-region and district parts in one partition or graph assignment.

    Args:
        source (Partition | nx.Graph): Partition or graph providing the plan topology.
        assignment (Assignment | None): Graph assignment mapping or node-ordered sequence. Omit
            for a Partition, which supplies its own assignment.
        region_attrs (str | Iterable[str]): One fixed-region column, or several to score together.
        geometry (GeoDataFrame | None, optional): Authoritative GeoDataFrame containing the region
            columns and aligned to ``source``. Defaults to None.
        node_id_column (str | None, optional): Geometry column containing the graph's node
            labels. Defaults to None.

    Returns:
        int | Series: Connected-part count for one region column, or a Series keyed by column name
            for several.
    """
    return _expect(
        (int, Series),
        _evaluate(
            source,
            assignment,
            RegionParts(*_columns(region_attrs)),
            geometry=geometry,
            node_id_column=node_id_column,
            topology_required=True,
        ),
    )


def tally_by_region(
    source: SinglePlanSource,
    assignment: GeoAssignment | None = None,
    *,
    region_attr: str,
    columns: str | Iterable[str] | Mapping[str, str] | None = None,
    include_count: bool = False,
    geometry: GeoDataFrame | None = None,
    node_id_column: str | None = None,
) -> DataFrame:
    """Sum named unit columns by fixed region and proposed district for one plan.

    Args:
        source (SinglePlanSource): Partition, graph, or authoritative GeoDataFrame containing the
            region and tally columns.
        assignment (GeoAssignment | None): District assignment. Required for graph and
            GeoDataFrame sources and omitted for a Partition. A GeoDataFrame also accepts an
            assignment column name.
        region_attr (str): Column containing fixed-region labels. Units with missing labels are
            omitted.
        columns (str | Iterable[str] | Mapping[str, str] | None): Columns to sum. A mapping assigns
            output names to source columns. May be None only when ``include_count`` is True.
        include_count (bool, optional): Include a leading ``"count"`` tally of units in each
            region-district pair. Defaults to False.
        geometry (GeoDataFrame | None, optional): Authoritative GeoDataFrame for a graph or
            Partition source. Defaults to None.
        node_id_column (str | None, optional): Geometry column containing the graph's node
            labels. Defaults to None.

    Returns:
        DataFrame: Fixed-region-indexed table with metric and district column levels.
    """
    return _expect(
        DataFrame,
        _evaluate(
            source,
            assignment,
            TallyByRegion(region_attr, columns, include_count=include_count),
            geometry=geometry,
            node_id_column=node_id_column,
        ),
    )
