"""Shared source normalization for single-plan functions."""

from __future__ import annotations

from collections.abc import Hashable, Iterable
from typing import Any, TypeAlias, TypeVar, cast, overload

import networkx as nx
from geopandas import GeoDataFrame
from gerrychain import Partition
from pandas import DataFrame, Series

from ..evaluator import (
    Assignment,
    PlanEvaluator,
    _assignment_mapping,
    _is_missing,
    _partition_assignment,
    _partition_graph,
    _result_name,
)
from ..metrics import Metric

SinglePlanSource: TypeAlias = Partition | nx.Graph | GeoDataFrame


SinglePlanResult: TypeAlias = float | int | Series | DataFrame


GeoAssignment: TypeAlias = Assignment | str
_ResultT = TypeVar("_ResultT", bound=SinglePlanResult)


@overload
def _expect(expected: type[_ResultT], result: SinglePlanResult) -> _ResultT: ...


@overload
def _expect(expected: tuple[type[_ResultT], ...], result: SinglePlanResult) -> _ResultT: ...


def _expect(
    expected: type[_ResultT] | tuple[type[_ResultT], ...],
    result: SinglePlanResult,
) -> _ResultT:
    """Validate the runtime result at the single-plan API boundary."""
    if not isinstance(result, expected):
        names = (
            " or ".join(item.__name__ for item in expected)
            if isinstance(expected, tuple)
            else expected.__name__
        )
        raise RuntimeError(f"metric returned {type(result).__name__}; expected {names}")
    return cast("_ResultT", result)


def _columns(values: str | Iterable[str]) -> tuple[str, ...]:
    return (values,) if isinstance(values, str) else tuple(values)


def _geodataframe_assignment(
    frame: GeoDataFrame,
    assignment: GeoAssignment | None,
) -> list[Hashable]:
    if assignment is None:
        raise TypeError("a GeoDataFrame source requires an assignment")
    if isinstance(assignment, str):
        if assignment not in frame.columns:
            raise ValueError(f"GeoDataFrame does not contain assignment column {assignment!r}")
        values = list(frame[assignment])
    elif (mapping := _assignment_mapping(assignment)) is not None:
        missing = [node for node in frame.index if node not in mapping]
        unexpected = [node for node in mapping if node not in frame.index]
        if missing or unexpected:
            raise ValueError(
                "assignment keys must exactly match GeoDataFrame index; "
                f"missing={missing!r}, unexpected={unexpected!r}"
            )
        values = [mapping[node] for node in frame.index]
    else:
        values = list(assignment)
        if len(values) != len(frame):
            raise ValueError(f"assignment has {len(values)} values; expected {len(frame)}")
    if any(_is_missing(value) for value in values):
        raise ValueError("assignment cannot contain missing district labels")
    return values


def _evaluate(
    source: SinglePlanSource,
    assignment: GeoAssignment | None,
    metric: Metric,
    *,
    geometry: GeoDataFrame | None = None,
    node_id_column: str | None = None,
    target_crs: Any | None = None,
    topology_required: bool = False,
) -> SinglePlanResult:
    if isinstance(assignment, GeoDataFrame):
        raise TypeError("assignment must contain district labels, not a GeoDataFrame")
    if isinstance(source, Partition):
        if assignment is not None:
            raise TypeError("a Partition supplies its own assignment")
        if geometry is None and (node_id_column is not None or target_crs is not None):
            raise ValueError("node_id_column and target_crs require geometry")
        evaluator = PlanEvaluator(
            _partition_graph(source),
            geometry=geometry,
            node_id_column=node_id_column,
            target_crs=target_crs,
        )
        plan: Assignment = _partition_assignment(source)
    elif isinstance(source, nx.Graph):
        if assignment is None:
            raise TypeError("a graph source requires an assignment")
        if isinstance(assignment, str):
            raise TypeError("a graph assignment cannot be a GeoDataFrame column name")
        evaluator = PlanEvaluator(
            source,
            geometry=geometry,
            node_id_column=node_id_column,
            target_crs=target_crs,
        )
        plan = assignment
    elif isinstance(source, GeoDataFrame):
        if topology_required:
            raise TypeError(f"{metric._kind} requires a GerryChain Partition or graph")
        if geometry is not None:
            raise TypeError("do not supply geometry when the source is already a GeoDataFrame")
        if node_id_column is not None:
            raise TypeError("node_id_column applies only to Partition geometry alignment")
        nodes = tuple(source.index)
        evaluator = PlanEvaluator(nx.empty_graph(nodes), geometry=source, target_crs=target_crs)
        plan = _geodataframe_assignment(source, assignment)
    else:
        raise TypeError("source must be a GerryChain Partition, graph, or GeoDataFrame")

    # Match evaluator registration, including explicit result names.
    name = _result_name(metric)
    return evaluator.add_metric(metric).evaluate(plan)[name]
