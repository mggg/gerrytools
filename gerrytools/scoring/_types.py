"""Shared scoring result value types."""

from __future__ import annotations

import re
from collections.abc import Callable, Hashable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Literal, TypeAlias, TypedDict

import numpy as np
import pandas as pd
from numpy.typing import NDArray

_Dtype: TypeAlias = Literal["bool", "float", "int"]
_EvaluationValue: TypeAlias = bool | float | int | pd.Series | pd.DataFrame


class _ResultShape(StrEnum):
    """Logical axes carried by a scoring result.

    Plan results have sample and metric axes. District results add a district axis, while region
    results add region and district axes.
    """

    DISTRICT = "district"
    PLAN = "plan"
    REGION = "region"


@dataclass(frozen=True, slots=True)
class EvaluationSummary:
    """Counts produced by a batch or streaming evaluation.

    ``samples`` counts plan occurrences, including streaming frame repetitions. ``accepted``
    counts result rows and equals ``samples`` for batch evaluation. Unique counts ignore district
    labels and are ``None`` when tracking was not requested.

    Attributes:
        samples (int): Number of plan occurrences processed.
        accepted (int): Number of result rows retained.
        unique_plans (int | None): Distinct plans, or None when not tracked. Defaults to None.
        unique_districts (int | None): Distinct district sets, or None when not tracked. Defaults
            to None.
    """

    samples: int
    accepted: int
    unique_plans: int | None = None
    unique_districts: int | None = None

    def __repr__(self) -> str:
        fields = [f"samples={self.samples!r}", f"accepted={self.accepted!r}"]
        if self.unique_plans is not None:
            fields.append(f"unique_plans={self.unique_plans!r}")
        if self.unique_districts is not None:
            fields.append(f"unique_districts={self.unique_districts!r}")
        return f"{type(self).__name__}({', '.join(fields)})"


@dataclass(frozen=True, slots=True)
class _MetricResult:
    """One immutable metric result in canonical axis order.

    ``values`` has shape ``(sample, metric)`` for plan results, ``(sample, metric, district)`` for
    district results, and ``(sample, metric, region, district)`` for region results. Axis labels
    and dtypes follow the corresponding array axes. Region metadata is present only for region
    results, and plan results have no district labels.
    """

    values: NDArray[np.float64]
    shape: _ResultShape
    columns: tuple[Hashable, ...]
    districts: tuple[Hashable, ...]
    dtypes: tuple[_Dtype, ...]
    regions: tuple[Hashable, ...] = ()
    region_name: str | None = None

    def __post_init__(self) -> None:
        if self.shape == _ResultShape.REGION:
            expected = (len(self.columns), len(self.regions), len(self.districts))
            valid = (
                self.region_name is not None
                and self.values.ndim == 4
                and self.values.shape[1:] == expected
            )
        elif self.region_name is not None or self.regions:
            valid = False
        elif self.shape == _ResultShape.DISTRICT:
            expected = (len(self.columns), len(self.districts))
            valid = self.values.ndim == 3 and self.values.shape[1:] == expected
        else:
            valid = (
                not self.districts
                and self.values.ndim == 2
                and self.values.shape[1] == len(self.columns)
            )
        if not valid or len(self.dtypes) != len(self.columns):
            raise ValueError("metric result values and axis metadata do not agree")
        base_array = self.values
        while True:
            parent = base_array.base
            if not isinstance(parent, np.ndarray):
                break
            base_array = parent
        base_array.setflags(write=False)
        self.values.setflags(write=False)


@dataclass(frozen=True, slots=True)
class _RunTable:
    """One physical Parquet table belonging to a streamed metric."""

    path: Path
    subkeys: tuple[str, ...]
    size: int
    sha256: str


@dataclass(frozen=True, slots=True)
class _RunMetric:
    """Validated manifest metadata for one streamed metric.

    ``tables`` contains the physical files in subkey order. ``columns`` are the logical metric
    labels; region metadata is present only for region-shaped results.
    """

    name: str
    shape: _ResultShape
    tables: tuple[_RunTable, ...]
    subkeys: tuple[str, ...]
    columns: tuple[str, ...]
    dtypes: tuple[_Dtype, ...]
    regions: tuple[Hashable, ...] = ()
    region_name: str | None = None


class _StreamRunOptions(TypedDict):
    """Named options passed from the Python evaluator to the Rust stream scorer."""

    bendl_node_order_json: str | None
    max_samples: int | None
    batch_size: int
    track_uniqueness: bool
    progress: Callable[[int, int | None], object] | None


_RESULT_NAME = re.compile(r"[A-Za-z0-9_.-]+")


def is_valid_metric_name(name: object) -> bool:
    """Whether ``name`` is safe as a metric instance and output path component."""
    return (
        isinstance(name, str)
        and name not in {".", ".."}
        and name.casefold() != "manifest.json"
        and _RESULT_NAME.fullmatch(name) is not None
    )
