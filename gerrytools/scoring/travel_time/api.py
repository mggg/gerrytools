"""GerryTools-compatible public functions for travel-time scoring."""

from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

import networkx as nx
import pandas as pd
from geopandas import GeoDataFrame
from gerrychain import Partition
from pandas import DataFrame, Series

from ..evaluator import (
    Assignment,
    _assignment_mapping,
    _is_missing,
    _partition_assignment,
    _partition_graph,
)
from ..single_plan._base import GeoAssignment, SinglePlanSource, _geodataframe_assignment
from .district import score_centroid_district, score_pairwise_district
from .od_table import TravelTimeTable, UnitId
from .weights import (
    CentroidWeightMode,
    PairwiseWeightMode,
    normalize_centroid_weight,
    normalize_pairwise_weight,
)


@dataclass(frozen=True, slots=True)
class _ResolvedPlan:
    units: tuple[UnitId, ...]
    labels: tuple[Hashable, ...]
    attributes: Mapping[UnitId, Mapping[str, Any]]


def _validated_labels(
    units: Sequence[UnitId],
    assignment: Assignment,
) -> tuple[Hashable, ...]:
    mapping = _assignment_mapping(assignment)
    if mapping is not None:
        missing = [unit for unit in units if unit not in mapping]
        unexpected = [unit for unit in mapping if unit not in set(units)]
        if missing or unexpected:
            raise ValueError(
                "assignment keys must exactly match source units; "
                f"missing={missing!r}, unexpected={unexpected!r}"
            )
        values = [mapping[unit] for unit in units]
    else:
        values = list(assignment)
        if len(values) != len(units):
            raise ValueError(f"assignment has {len(values)} values; expected {len(units)}")

    labels: list[Hashable] = []
    for value in values:
        if _is_missing(value):
            raise ValueError("assignment cannot contain missing district labels")
        try:
            hash(value)
        except TypeError as exc:
            raise TypeError(f"district labels must be hashable; got {value!r}") from exc
        labels.append(cast(Hashable, value))
    return tuple(labels)


def _resolve_plan(
    source: SinglePlanSource,
    assignment: GeoAssignment | None,
) -> _ResolvedPlan:
    if isinstance(assignment, GeoDataFrame):
        raise TypeError("assignment must contain district labels, not a GeoDataFrame")

    if isinstance(source, Partition):
        if assignment is not None:
            raise TypeError("a Partition supplies its own assignment")
        graph = _partition_graph(source)
        units = tuple(graph.nodes)
        labels = _validated_labels(units, _partition_assignment(source))
        attributes = {unit: dict(graph.nodes[unit]) for unit in units}
    elif isinstance(source, nx.Graph):
        if assignment is None:
            raise TypeError("a graph source requires an assignment")
        if isinstance(assignment, str):
            raise TypeError("a graph assignment cannot be a GeoDataFrame column name")
        units = tuple(source.nodes)
        labels = _validated_labels(units, assignment)
        attributes = {unit: dict(source.nodes[unit]) for unit in units}
    elif isinstance(source, GeoDataFrame):
        if not source.index.is_unique:
            raise ValueError("GeoDataFrame index must be unique")
        units = tuple(cast(Sequence[UnitId], source.index.tolist()))
        labels = _validated_labels(units, _geodataframe_assignment(source, assignment))
        attributes = {unit: source.loc[unit].to_dict() for unit in units}
    else:
        raise TypeError("source must be a GerryChain Partition, graph, or GeoDataFrame")

    if not units:
        raise ValueError("source must contain at least one unit")
    return _ResolvedPlan(units, labels, attributes)


def _parts(plan: _ResolvedPlan) -> dict[Hashable, list[UnitId]]:
    parts: dict[Hashable, list[UnitId]] = {}
    for unit, district in zip(plan.units, plan.labels, strict=True):
        parts.setdefault(district, []).append(unit)
    return parts


def _populations(
    plan: _ResolvedPlan,
    population_attr: str | None,
) -> dict[UnitId, float]:
    if population_attr is None:
        raise ValueError("population_attr is required for population-weighted travel time")
    if not isinstance(population_attr, str) or not population_attr:
        raise ValueError("population_attr must be a nonempty string")
    missing = [unit for unit in plan.units if population_attr not in plan.attributes[unit]]
    if missing:
        raise ValueError(
            f"population attribute {population_attr!r} is missing for units {missing!r}"
        )
    return {unit: cast(float, plan.attributes[unit][population_attr]) for unit in plan.units}


def _district_index(parts: Mapping[Hashable, Sequence[UnitId]]) -> pd.Index:
    return pd.Index(tuple(parts), dtype=object, name="district", tupleize_cols=False)


def pairwise_travel_time(
    source: SinglePlanSource,
    assignment: GeoAssignment | None = None,
    *,
    od: TravelTimeTable,
    population_attr: str | None = None,
    weight: PairwiseWeightMode | str = PairwiseWeightMode.NONE,
) -> Series:
    """Calculate mean pairwise travel time for each district.

    Args:
        source (SinglePlanSource): Partition, graph, or GeoDataFrame containing the plan units.
        assignment (GeoAssignment | None): District assignment for graph and GeoDataFrame
            sources. A GeoDataFrame also accepts an assignment column name. Omit for a Partition.
        od (TravelTimeTable): Statewide-connected unit-to-unit travel-time table.
        population_attr (str | None): Node attribute or GeoDataFrame column used by population-
            weighted modes. Unweighted scoring does not require it.
        weight (PairwiseWeightMode | str): ``"none"``, ``"p+p"``, or ``"p*p"``.

    Returns:
        Series: District-indexed mean pairwise travel times.
    """
    mode = normalize_pairwise_weight(weight)
    plan = _resolve_plan(source, assignment)
    parts = _parts(plan)
    populations = None if mode is PairwiseWeightMode.NONE else _populations(plan, population_attr)
    values = [
        score_pairwise_district(units, populations, od, mode).mean for units in parts.values()
    ]
    return Series(values, index=_district_index(parts), name="pairwise_travel_time", dtype=float)


def centroid_travel_time(
    source: SinglePlanSource,
    assignment: GeoAssignment | None = None,
    *,
    od: TravelTimeTable,
    population_attr: str | None = None,
    weight: CentroidWeightMode | str = CentroidWeightMode.NONE,
) -> Series:
    """Calculate mean centroid travel time for each district.

    Args:
        source (SinglePlanSource): Partition, graph, or GeoDataFrame containing the plan units.
        assignment (GeoAssignment | None): District assignment for graph and GeoDataFrame
            sources. A GeoDataFrame also accepts an assignment column name. Omit for a Partition.
        od (TravelTimeTable): Statewide-connected unit-to-unit travel-time table.
        population_attr (str | None): Node attribute or GeoDataFrame column used by destination-
            population weighting. Unweighted scoring does not require it.
        weight (CentroidWeightMode | str): ``"none"`` or ``"p"``.

    Returns:
        Series: District-indexed mean centroid travel times.
    """
    mode = normalize_centroid_weight(weight)
    plan = _resolve_plan(source, assignment)
    parts = _parts(plan)
    populations = None if mode is CentroidWeightMode.NONE else _populations(plan, population_attr)
    values = [
        score_centroid_district(units, populations, od, mode).mean for units in parts.values()
    ]
    return Series(values, index=_district_index(parts), name="centroid_travel_time", dtype=float)


def travel_time_scores(
    source: SinglePlanSource,
    assignment: GeoAssignment | None = None,
    *,
    od: TravelTimeTable,
    population_attr: str | None = None,
    pairwise_weight: PairwiseWeightMode | str = PairwiseWeightMode.NONE,
    centroid_weight: CentroidWeightMode | str = CentroidWeightMode.NONE,
) -> DataFrame:
    """Calculate pairwise and centroid travel-time details together.

    Args:
        source (SinglePlanSource): Partition, graph, or GeoDataFrame containing the plan units.
        assignment (GeoAssignment | None): District assignment for graph and GeoDataFrame
            sources. A GeoDataFrame also accepts an assignment column name. Omit for a Partition.
        od (TravelTimeTable): Statewide-connected unit-to-unit travel-time table.
        population_attr (str | None): Node attribute or GeoDataFrame column used whenever either
            weighting mode requires population.
        pairwise_weight (PairwiseWeightMode | str): Pairwise weighting policy.
        centroid_weight (CentroidWeightMode | str): Centroid weighting policy.

    Returns:
        DataFrame: District-indexed ``ptt_mean``, ``ptt_max``, ``ctt_mean``, ``ctt_max``, and
            ``centroid`` columns.
    """
    pairwise_mode = normalize_pairwise_weight(pairwise_weight)
    centroid_mode = normalize_centroid_weight(centroid_weight)
    plan = _resolve_plan(source, assignment)
    parts = _parts(plan)
    needs_population = (
        pairwise_mode is not PairwiseWeightMode.NONE or centroid_mode is not CentroidWeightMode.NONE
    )
    populations = _populations(plan, population_attr) if needs_population else None

    rows: list[dict[str, Any]] = []
    for units in parts.values():
        pairwise = score_pairwise_district(units, populations, od, pairwise_mode)
        centroid = score_centroid_district(units, populations, od, centroid_mode)
        rows.append(
            {
                "ptt_mean": pairwise.mean,
                "ptt_max": pairwise.maximum,
                "ctt_mean": centroid.mean,
                "ctt_max": centroid.maximum,
                "centroid": centroid.centroid,
            }
        )
    return DataFrame(rows, index=_district_index(parts))
