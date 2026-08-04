"""Shared implementation support for metric descriptions."""

from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, ClassVar, Literal, TypeAlias

from .._types import _Dtype, _ResultShape

if TYPE_CHECKING:
    from gerrytools._scoring_engine import ScoringEngine

    from ..evaluator import PlanEvaluator

_ColumnResource: TypeAlias = tuple[Literal["graph", "geometry"], str]


@dataclass(frozen=True, slots=True)
class _ResourceSpec:
    """Immutable set of source resources required by a metric collection."""

    topology: bool = False
    node_columns: frozenset[_ColumnResource] = frozenset()
    edge_columns: frozenset[str] = frozenset()
    region_columns: frozenset[_ColumnResource] = frozenset()
    alignment: bool = False
    geometry: bool = False
    fixed_values: frozenset[Hashable] = frozenset()

    def __post_init__(self) -> None:
        geometry_column = any(
            source == "geometry" for source, _ in self.node_columns | self.region_columns
        )
        if geometry_column or self.geometry:
            object.__setattr__(self, "alignment", True)
        if self.edge_columns:
            object.__setattr__(self, "topology", True)

    def __or__(self, other: "_ResourceSpec") -> "_ResourceSpec":
        return _ResourceSpec(
            topology=self.topology or other.topology,
            node_columns=self.node_columns | other.node_columns,
            edge_columns=self.edge_columns | other.edge_columns,
            region_columns=self.region_columns | other.region_columns,
            alignment=self.alignment or other.alignment,
            geometry=self.geometry or other.geometry,
            fixed_values=self.fixed_values | other.fixed_values,
        )

    def contains(self, other: "_ResourceSpec") -> bool:
        """Return whether every resource in ``other`` is present."""
        return (
            (self.topology or not other.topology)
            and self.node_columns.issuperset(other.node_columns)
            and self.edge_columns.issuperset(other.edge_columns)
            and self.region_columns.issuperset(other.region_columns)
            and (self.alignment or not other.alignment)
            and (self.geometry or not other.geometry)
            and self.fixed_values.issuperset(other.fixed_values)
        )


@dataclass(frozen=True, slots=True)
class _OutputSpec:
    shape: _ResultShape
    columns: tuple[Hashable, ...]
    dtypes: tuple[_Dtype, ...]
    regions: tuple[Hashable, ...] = ()
    region_name: str | None = None

    def __post_init__(self) -> None:
        if len(self.columns) != len(self.dtypes):
            raise ValueError("metric output columns and dtypes must have equal length")
        if self.shape == _ResultShape.REGION:
            if self.region_name is None:
                raise ValueError("region output requires a region axis name")
        elif self.regions or self.region_name is not None:
            raise ValueError("only region output can define a region axis")

    @property
    def value_count(self) -> int:
        """Number of flat engine columns represented by this output."""
        if self.shape == _ResultShape.REGION:
            return len(self.columns) * len(self.regions)
        return len(self.columns)


@dataclass(frozen=True, slots=True, kw_only=True)
class _MetricBase:
    """Implementation-sharing base for the concrete metric descriptors in this module.

    ``result_name`` changes only the public result key. It does not affect equality so differently
    named registrations can still share prepared engine work.
    """

    _kind: ClassVar[str]
    result_name: str | None = field(default=None, compare=False)

    def _default_name(self) -> str:
        return self._kind

    def _validate(self, evaluator: PlanEvaluator) -> None:
        pass

    def _resources(self, evaluator: PlanEvaluator) -> _ResourceSpec:
        return _ResourceSpec(
            node_columns=frozenset(
                evaluator._ordinary_column_resource(key) for key in self._tally_keys()
            )
        )

    def _merge(self, other: _MetricBase) -> _MetricBase | None:
        return None

    def _options(self) -> dict[str, object]:
        return {}

    def _stream_options(self, evaluator: PlanEvaluator) -> dict[str, object]:
        """Options recorded in streamed manifests, with evaluator-dependent defaults resolved."""
        del evaluator
        return self._options()

    def _tally_keys(self) -> tuple[str, ...]:
        return ()

    def _column_indices(self, available: tuple[Hashable, ...]) -> tuple[int, ...]:
        return tuple(range(len(available)))

    def _result_columns(
        self,
        available: tuple[Hashable, ...],
        indices: tuple[int, ...],
    ) -> tuple[Hashable, ...]:
        return tuple(available[index] for index in indices)

    def _prepare(self, backend: ScoringEngine, evaluator: PlanEvaluator) -> _OutputSpec:
        raise NotImplementedError


def _keys(values: tuple[str, ...], metric: str, kind: str = "column") -> tuple[str, ...]:
    if not values or any(not isinstance(value, str) or not value for value in values):
        raise ValueError(f"{metric} requires at least one nonempty string {kind}")
    if len(set(values)) != len(values):
        raise ValueError(f"{metric} {kind}s cannot repeat")
    return values


def _merged_keys(left: tuple[str, ...], right: tuple[str, ...]) -> tuple[str, ...]:
    return left + tuple(key for key in right if key not in left)


@dataclass(frozen=True, slots=True, init=False)
class _KeyedMetric(_MetricBase):
    keys: tuple[str, ...]

    def __init__(self, *columns: str, result_name: str | None = None) -> None:
        object.__setattr__(self, "result_name", result_name)
        object.__setattr__(self, "keys", _keys(columns, type(self).__name__))

    def _column_indices(self, available: tuple[Hashable, ...]) -> tuple[int, ...]:
        return tuple(available.index(key) for key in self.keys)
