"""Semantic results returned by plan scoring."""

from __future__ import annotations

import hashlib
import inspect
import json
import numbers
import os
import warnings
from collections.abc import Hashable, Iterable, Iterator, Mapping
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import BinaryIO, Literal, TypeAlias, cast, overload

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from numpy.typing import NDArray

from ._manifest import find_manifest_path as _find_manifest_path
from ._manifest import parse_manifest as _parse_manifest
from ._types import (
    EvaluationSummary,
    _Dtype,
    _EvaluationValue,
    _MetricResult,
    _ResultShape,
    _RunMetric,
    _RunTable,
)

_ReadOperation = Literal["frames", "raw", "read"]
_PandasResult: TypeAlias = pd.Series | pd.DataFrame
_PANDAS_DTYPES = {"bool": "bool", "float": "float64", "int": "int64"}
_DTYPE_WIDTHS = {"bool": 1, "float": 8, "int": 8}
_PREFIX_COLUMNS = ("sample_offset", "repetitions", "accepted_index")
_WARNING_BYTES = 2 * 1024**3
_ERROR_BYTES = 8 * 1024**3
_FOOTER_EXPANSION = 16


class EvaluationMemoryError(MemoryError):
    """A result read rejected before its predicted peak exceeds the safety limit.

    Args:
        message (str): Human-readable error description.
        estimated_bytes (int): Predicted peak memory use in bytes.
        limit_bytes (int): Configured safety limit in bytes.

    Attributes:
        estimated_bytes (int): Predicted peak memory use in bytes.
        limit_bytes (int): Configured safety limit in bytes.
    """

    def __init__(self, message: str, estimated_bytes: int, limit_bytes: int) -> None:
        super().__init__(message)
        self.estimated_bytes = estimated_bytes
        self.limit_bytes = limit_bytes


def _index(values: Iterable[Hashable], name: str) -> pd.Index:
    labels = tuple(values)
    array = np.empty(len(labels), dtype=object)
    array[:] = labels
    return pd.Index(array, name=name)


def _metric(results: Mapping[str, _MetricResult], name: str) -> _MetricResult:
    try:
        return results[name]
    except KeyError:
        available = ", ".join(repr(key) for key in results)
        raise KeyError(f"unknown metric {name!r}; available: {available}") from None


def _scalar(value: np.float64, dtype: _Dtype) -> bool | float | int:
    if dtype == "bool":
        return bool(value)
    if dtype == "int":
        return int(value)
    return float(value)


def _cast_columns(
    frame: pd.DataFrame,
    columns: Iterable[Hashable],
    dtypes: Iterable[_Dtype],
) -> pd.DataFrame:
    return frame.astype(
        {column: _PANDAS_DTYPES[dtype] for column, dtype in zip(columns, dtypes, strict=True)}
    )


def _writable(values: NDArray[np.float64]) -> NDArray[np.float64]:
    """Copy frozen result values before wrapping them in mutable pandas objects."""
    return np.array(values)


class _Evaluation(Mapping[str, _EvaluationValue]):
    """Read-only logical metric mapping shared by one-plan and many-plan results."""

    _single: bool

    def __init__(self, results: Mapping[str, _MetricResult]) -> None:
        self._results = dict(results)

    @property
    def metrics(self) -> tuple[str, ...]:
        """Logical metric names in registration order."""
        return tuple(self._results)

    def __iter__(self) -> Iterator[str]:
        return iter(self._results)

    def __len__(self) -> int:
        return len(self._results)

    def array(self, name: str) -> NDArray[np.float64]:
        """Return an immutable array ordered like the axes of ``result[name]``.

        Args:
            name (str): Registered metric name.

        Returns:
            NDArray[np.float64]: Metric values in canonical sample, metric, region, and district
            axis order, with axes omitted when they do not apply.

        Raises:
            KeyError: If ``name`` is not present in the result.
        """
        values = _metric(self._results, name).values
        return (values[0] if self._single else values).view()


class PlanEvalResult(_Evaluation):
    """Read-only semantic metric values for one plan.

    Instances are normally returned by :meth:`PlanEvaluator.evaluate`.

    Args:
        results (Mapping[str, _MetricResult]): Internal metric results produced by the evaluator.

    Raises:
        ValueError: If a metric contains anything other than one result row.
    """

    _single = True

    def __init__(self, results: Mapping[str, _MetricResult]) -> None:
        if any(len(result.values) != 1 for result in results.values()):
            raise ValueError("a plan evaluation requires exactly one row per metric")
        super().__init__(results)

    def __getitem__(self, name: str) -> _EvaluationValue:
        result = _metric(self._results, name)
        values = result.values[0]
        columns = _index(result.columns, "metric")
        if result.shape == _ResultShape.REGION:
            assert result.region_name is not None
            regions = _index(result.regions, result.region_name)
            districts = _index(result.districts, "district")
            result_columns = pd.MultiIndex.from_product(
                (columns, districts),
                names=("metric", "district"),
            )
            frame = pd.DataFrame(
                values.transpose(1, 0, 2).reshape(len(regions), len(result_columns)),
                index=regions,
                columns=result_columns,
            )
            dtypes: Iterable[_Dtype] = (dtype for dtype in result.dtypes for _ in result.districts)
            return _cast_columns(frame, result_columns, dtypes)

        if result.shape == _ResultShape.PLAN:
            if len(result.columns) == 1:
                return _scalar(values[0], result.dtypes[0])
            typed_values = [
                _scalar(value, dtype) for value, dtype in zip(values, result.dtypes, strict=True)
            ]
            return pd.Series(typed_values, index=columns, name=name)

        districts = _index(result.districts, "district")
        if len(result.columns) == 1:
            return pd.Series(
                _writable(values[0]),
                index=districts,
                name=name,
                dtype=_PANDAS_DTYPES[result.dtypes[0]],
            )
        frame = pd.DataFrame(values.T, index=districts, columns=columns)
        return _cast_columns(frame, result.columns, result.dtypes)


class ManyPlanEvalResult(_Evaluation):
    """Read-only semantic metric values for an ordered collection of plans.

    Instances are normally returned by :meth:`PlanEvaluator.evaluate_many`.

    Args:
        results (Mapping[str, _MetricResult]): Internal metric results produced by the evaluator.
        sample_ids (Iterable[Hashable] | None, optional): Unique labels for result rows. Defaults
            to a zero-based range.
        summary (EvaluationSummary | None, optional): Batch counts, or None to derive them from the
            result rows. Defaults to None.

    Raises:
        ValueError: If metrics have inconsistent row counts or sample labels are invalid.
    """

    _single = False

    def __init__(
        self,
        results: Mapping[str, _MetricResult],
        sample_ids: Iterable[Hashable] | None = None,
        *,
        summary: EvaluationSummary | None = None,
    ) -> None:
        self._initialize(results, sample_ids, summary, sample_index=None)

    @classmethod
    def _from_index(
        cls,
        results: Mapping[str, _MetricResult],
        sample_index: pd.Index,
    ) -> ManyPlanEvalResult:
        result = cls.__new__(cls)
        result._initialize(results, None, None, sample_index=sample_index)
        return result

    def _initialize(
        self,
        results: Mapping[str, _MetricResult],
        sample_ids: Iterable[Hashable] | None,
        summary: EvaluationSummary | None,
        *,
        sample_index: pd.Index | None,
    ) -> None:
        row_counts = {len(result.values) for result in results.values()}
        if not row_counts:
            raise ValueError("many-plan evaluation requires at least one metric result")
        if len(row_counts) != 1:
            raise ValueError("many-plan metrics must contain the same number of plans")
        count = row_counts.pop()
        if summary is None:
            summary = EvaluationSummary(count, count)
        elif summary.samples != count or summary.accepted != count:
            raise ValueError("batch evaluation summary must match its number of plans")
        self.summary = summary
        if sample_index is not None:
            if sample_ids is not None or len(sample_index) != count:
                raise ValueError("sample index must match the result rows")
            self._samples = sample_index
        elif sample_ids is None:
            self._samples = pd.RangeIndex(count, name="sample")
        else:
            self._samples = _index(sample_ids, "sample")
            if len(self._samples) != count:
                raise ValueError(f"sample_ids has {len(self._samples)} values; expected {count}")
            try:
                for sample_id in self._samples:
                    hash(sample_id)
            except TypeError:
                raise ValueError("sample_ids must contain unique hashable values") from None
            if self._samples.has_duplicates:
                raise ValueError("sample_ids must contain unique hashable values")
        super().__init__(results)

    def __getitem__(self, name: str) -> _PandasResult:
        result = _metric(self._results, name)
        values = result.values
        if result.shape == _ResultShape.REGION:
            assert result.region_name is not None
            regions = _index(result.regions, result.region_name)
            districts = _index(result.districts, "district")
            row_index = pd.MultiIndex.from_product(
                (self._samples, regions),
                names=(self._samples.name, result.region_name),
            )
            columns = pd.MultiIndex.from_product(
                (_index(result.columns, "metric"), districts),
                names=("metric", "district"),
            )
            frame = pd.DataFrame(
                values.transpose(0, 2, 1, 3).reshape(len(row_index), len(columns)),
                index=row_index,
                columns=columns,
            )
            dtypes: Iterable[_Dtype] = (dtype for dtype in result.dtypes for _ in result.districts)
            return _cast_columns(frame, columns, dtypes)

        if result.shape == _ResultShape.PLAN:
            if len(result.columns) == 1:
                return pd.Series(
                    _writable(values[:, 0]),
                    index=self._samples,
                    name=name,
                    dtype=_PANDAS_DTYPES[result.dtypes[0]],
                )
            columns = _index(result.columns, "metric")
            frame = pd.DataFrame(values, index=self._samples, columns=columns)
            return _cast_columns(frame, result.columns, result.dtypes)

        districts = _index(result.districts, "district")
        if len(result.columns) == 1:
            return pd.DataFrame(
                _writable(values[:, 0, :]),
                index=self._samples,
                columns=districts,
                dtype=_PANDAS_DTYPES[result.dtypes[0]],
            )
        columns = pd.MultiIndex.from_product(
            (_index(result.columns, "metric"), districts),
            names=("metric", "district"),
        )
        frame = pd.DataFrame(values.reshape(len(values), -1), index=self._samples, columns=columns)
        dtypes: Iterable[_Dtype] = (dtype for dtype in result.dtypes for _ in result.districts)
        return _cast_columns(frame, columns, dtypes)


class EnsembleEvalResult:
    """Read a completed streamed ensemble evaluation and reconstruct logical metric results.

    Use :meth:`open` to validate a published ensemble result directory before reading its metrics.

    Direct construction accepts already validated internal metadata; callers should normally use
    :meth:`open`.

    Args:
        path (Path): Ensemble result directory.
        summary (EvaluationSummary): Validated evaluation counts.
        districts (tuple[int, ...]): District labels in stored column order.
        metrics (tuple[_RunMetric, ...]): Validated internal metric metadata.
    """

    def __init__(
        self,
        path: Path,
        summary: EvaluationSummary,
        districts: tuple[int, ...],
        metrics: tuple[_RunMetric, ...],
    ) -> None:
        self.path = path
        self.summary = summary
        self._districts = districts
        self._metric_metadata = {metric.name: metric for metric in metrics}
        self._frames: pd.DataFrame | None = None

    @classmethod
    def open(cls, path: str | os.PathLike[str]) -> "EnsembleEvalResult":
        """Open and validate a successfully published ensemble result.

        Args:
            path (str | os.PathLike[str]): Ensemble result directory.

        Returns:
            EnsembleEvalResult: Validated lazy reader for the result.

        Raises:
            OSError: If the manifest or metric files cannot be read.
            ValueError: If the manifest or result layout is malformed or inconsistent.
        """
        run_path = Path(path)
        manifest_path = _find_manifest_path(run_path)
        try:
            with manifest_path.open() as file:
                manifest = json.load(file)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid evaluation manifest JSON: {error}") from error
        summary, districts, metrics = _parse_manifest(run_path, manifest)
        return cls(run_path, summary, districts, metrics)

    @property
    def metrics(self) -> tuple[str, ...]:
        """Logical metric names in registration order."""
        return tuple(self._metric_metadata)

    @property
    def frames(self) -> pd.DataFrame:
        """Accepted stream frames indexed by ``accepted``."""
        if self._frames is None:
            metric = next(iter(self._metric_metadata.values()))
            table = _read_eager_table(
                metric,
                self.summary,
                self._districts,
                "frames",
                columns=list(_PREFIX_COLUMNS),
                allow_large=False,
            )
            self._frames = _validated_frames(table, self.summary)
        frames = self._frames
        assert frames is not None
        return frames.copy()

    def raw(self, name: str, *, allow_large: bool = False) -> pd.DataFrame:
        """Read one metric's physical Parquet table.

        Args:
            name (str): Registered metric name.
            allow_large (bool, optional): Whether to permit an eager read above the memory safety
                limit. Defaults to False.

        Returns:
            pd.DataFrame: Physical rows, including frame metadata and stored metric columns.

        Raises:
            EvaluationMemoryError: If the estimated eager read is too large and ``allow_large`` is
                False.
            KeyError: If ``name`` is not a metric in this result.
            TypeError: If ``allow_large`` is not a Boolean.
            ValueError: If the stored table is malformed or inconsistent with the manifest.
        """
        _validate_bool(allow_large, "allow_large")
        metric = _run_metric(self._metric_metadata, name)
        table = _read_eager_table(
            metric,
            self.summary,
            self._districts,
            "raw",
            allow_large=allow_large,
        )
        self._cache_or_compare_frames(metric, table)
        return table

    @overload
    def read(
        self,
        name: str,
        *,
        expand_repetitions: bool = False,
        allow_large: bool = False,
        return_type: Literal["series"],
    ) -> pd.Series: ...

    @overload
    def read(
        self,
        name: str,
        *,
        expand_repetitions: bool = False,
        allow_large: bool = False,
        return_type: Literal["dataframe"],
    ) -> pd.DataFrame: ...

    @overload
    def read(
        self,
        name: str,
        *,
        expand_repetitions: bool = False,
        allow_large: bool = False,
        return_type: None = None,
    ) -> _PandasResult: ...

    def read(
        self,
        name: str,
        *,
        expand_repetitions: bool = False,
        allow_large: bool = False,
        return_type: Literal["series", "dataframe"] | None = None,
    ) -> _PandasResult:
        """Read one logical metric, optionally expanding repeated stream frames.

        Args:
            name (str): Registered metric name.
            expand_repetitions (bool, optional): Whether to repeat accepted values by each frame's
                repetition count. Defaults to False.
            allow_large (bool, optional): Whether to permit an eager read above the memory safety
                limit. Defaults to False.
            return_type (Literal["series", "dataframe"] | None, optional): Requested pandas result
                shape. Series results become one-column DataFrames; one-column DataFrames become
                Series. Defaults to None.

        Returns:
            pd.Series | pd.DataFrame: Values in the metric's logical result shape.

        Raises:
            EvaluationMemoryError: If the estimated eager read is too large and ``allow_large`` is
                False.
            KeyError: If ``name`` is not a metric in this result.
            TypeError: If a Boolean option has an incompatible type or a multi-column result is
                requested as a Series.
            ValueError: If ``return_type`` is invalid or the stored table is malformed or
                inconsistent with the manifest.
        """
        _validate_bool(expand_repetitions, "expand_repetitions")
        _validate_bool(allow_large, "allow_large")
        if return_type not in (None, "series", "dataframe"):
            raise ValueError("return_type must be 'series', 'dataframe', or None")
        metric = _run_metric(self._metric_metadata, name)
        table = _read_eager_table(
            metric,
            self.summary,
            self._districts,
            "read",
            expand_repetitions=expand_repetitions,
            allow_large=allow_large,
        )
        self._cache_or_compare_frames(metric, table)
        values = _physical_values(table)
        if expand_repetitions:
            repetitions = table["repetitions"].to_numpy(dtype=np.intp)
            values = np.repeat(values, repetitions, axis=0)
        sample_name = "sample" if expand_repetitions else "accepted"
        index = pd.RangeIndex(len(values), name=sample_name)
        result = _semantic_value(
            metric.name,
            metric,
            self._districts,
            values,
            index,
        )
        if return_type == "series" and isinstance(result, pd.DataFrame):
            if len(result.columns) != 1:
                raise TypeError(
                    f"metric {name!r} produced a DataFrame with {len(result.columns)} columns; "
                    "cannot return a Series"
                )
            return result.iloc[:, 0]
        if return_type == "dataframe" and isinstance(result, pd.Series):
            return result.to_frame()
        return result

    def iter_raw_batches(
        self,
        name: str,
        *,
        batch_size: int = 1_024,
        allow_large: bool = False,
    ) -> Iterator[pd.DataFrame]:
        """Yield bounded physical-table batches for one metric.

        Args:
            name (str): Registered metric name.
            batch_size (int, optional): Maximum physical rows per batch. Defaults to 1,024.
            allow_large (bool, optional): Whether one batch may exceed the memory safety limit.
                Defaults to False.

        Yields:
            pd.DataFrame: Physical rows with frame metadata and stored metric columns.

        Raises:
            EvaluationMemoryError: If one batch is too large and ``allow_large`` is False.
            KeyError: If ``name`` is not a metric in this result.
            TypeError: If ``allow_large`` is not a Boolean.
            ValueError: If ``batch_size`` or the stored table is invalid.
        """
        batch_size = _validate_batch_options(batch_size, allow_large)
        metric = _run_metric(self._metric_metadata, name)
        expected_offset = 0
        expected_accepted = 0
        with _open_metric_readers(
            metric,
            self.summary,
            self._districts,
            "raw",
            allow_large=allow_large,
            batch_size=batch_size,
        ) as readers:
            for table in _iter_physical_batches(
                metric,
                readers,
                self._districts,
                batch_size,
            ):
                _, expected_offset, expected_accepted = _validated_frame_batch(
                    table,
                    expected_offset,
                    expected_accepted,
                )
                table.index = pd.RangeIndex(
                    expected_accepted - len(table),
                    expected_accepted,
                )
                output = table
                del table
                yield output
                del output
        _validate_iterator_totals(expected_offset, expected_accepted, self.summary)

    def iter_frame_batches(
        self,
        *,
        batch_size: int = 1_024,
        allow_large: bool = False,
    ) -> Iterator[pd.DataFrame]:
        """Yield bounded accepted-frame batches without populating the eager cache.

        Args:
            batch_size (int, optional): Maximum accepted frames per batch. Defaults to 1,024.
            allow_large (bool, optional): Whether one batch may exceed the memory safety limit.
                Defaults to False.

        Yields:
            pd.DataFrame: Accepted frame metadata indexed by accepted-frame number.

        Raises:
            EvaluationMemoryError: If one batch is too large and ``allow_large`` is False.
            TypeError: If ``allow_large`` is not a Boolean.
            ValueError: If ``batch_size`` or the stored frame table is invalid.
        """
        batch_size = _validate_batch_options(batch_size, allow_large)
        metric = next(iter(self._metric_metadata.values()))
        expected_offset = 0
        expected_accepted = 0
        with _open_metric_readers(
            metric,
            self.summary,
            self._districts,
            "frames",
            allow_large=allow_large,
            batch_size=batch_size,
        ) as readers:
            for table in _iter_physical_batches(
                metric,
                readers,
                self._districts,
                batch_size,
                frames_only=True,
            ):
                output, expected_offset, expected_accepted = _validated_frame_batch(
                    table,
                    expected_offset,
                    expected_accepted,
                    return_frame=True,
                )
                assert output is not None
                del table
                yield output
                del output
        _validate_iterator_totals(expected_offset, expected_accepted, self.summary)

    def iter_batches(
        self,
        name: str,
        *,
        batch_size: int = 1_024,
        expand_repetitions: bool = False,
        allow_large: bool = False,
    ) -> Iterator[_PandasResult]:
        """Yield bounded semantic batches for one metric.

        Args:
            name (str): Registered metric name.
            batch_size (int, optional): Maximum logical rows per batch. Defaults to 1,024.
            expand_repetitions (bool, optional): Whether to repeat accepted values by each frame's
                repetition count. Defaults to False.
            allow_large (bool, optional): Whether one batch may exceed the memory safety limit.
                Defaults to False.

        Yields:
            pd.Series | pd.DataFrame: Values in the metric's logical result shape.

        Raises:
            EvaluationMemoryError: If one batch is too large and ``allow_large`` is False.
            KeyError: If ``name`` is not a metric in this result.
            TypeError: If a Boolean option has an incompatible type.
            ValueError: If ``batch_size`` or the stored table is invalid.
        """
        batch_size = _validate_batch_options(batch_size, allow_large)
        _validate_bool(expand_repetitions, "expand_repetitions")
        metric = _run_metric(self._metric_metadata, name)
        expected_offset = 0
        expected_accepted = 0
        sample_start = 0
        with _open_metric_readers(
            metric,
            self.summary,
            self._districts,
            "read",
            expand_repetitions=expand_repetitions,
            allow_large=allow_large,
            batch_size=batch_size,
        ) as readers:
            for table in _iter_physical_batches(
                metric,
                readers,
                self._districts,
                batch_size,
            ):
                accepted_start = expected_accepted
                _, expected_offset, expected_accepted = _validated_frame_batch(
                    table,
                    expected_offset,
                    expected_accepted,
                )
                values = _physical_values(table)
                if not expand_repetitions:
                    index = pd.RangeIndex(
                        accepted_start,
                        expected_accepted,
                        name="accepted",
                    )
                    output = _semantic_value(name, metric, self._districts, values, index)
                    del table, values
                    yield output
                    del output
                    continue

                repetitions = table["repetitions"].to_numpy(dtype=np.intp, copy=True)
                del table
                # The estimate keeps this accepted-value buffer live while repetitions are split.
                row = 0
                remaining = int(repetitions[0]) if len(repetitions) else 0
                while row < len(values):
                    logical_rows = min(batch_size, expected_offset - sample_start)
                    buffer = np.empty((logical_rows, values.shape[1]), dtype=np.float64)
                    filled = 0
                    while filled < logical_rows:
                        take = min(remaining, logical_rows - filled)
                        buffer[filled : filled + take] = values[row]
                        filled += take
                        remaining -= take
                        if remaining == 0:
                            row += 1
                            if row < len(values):
                                remaining = int(repetitions[row])
                    index = pd.RangeIndex(
                        sample_start,
                        sample_start + logical_rows,
                        name="sample",
                    )
                    sample_start += logical_rows
                    output = _semantic_value(name, metric, self._districts, buffer, index)
                    del buffer
                    yield output
                    del output
                del values, repetitions
        _validate_iterator_totals(expected_offset, expected_accepted, self.summary)

    def _cache_or_compare_frames(self, metric: _RunMetric, table: pd.DataFrame) -> None:
        frames = _validated_frames(table, self.summary)
        if self._frames is None:
            self._frames = frames
        elif not frames.equals(self._frames):
            raise ValueError(
                f"metric {metric.name!r} frame columns disagree with the ensemble result"
            )


def _run_metric(metrics: Mapping[str, _RunMetric], name: str) -> _RunMetric:
    try:
        return metrics[name]
    except KeyError:
        available = ", ".join(repr(key) for key in metrics)
        raise KeyError(f"unknown metric {name!r}; available: {available}") from None


def _read_eager_table(
    metric: _RunMetric,
    summary: EvaluationSummary,
    districts: tuple[int, ...],
    operation: _ReadOperation,
    *,
    columns: list[str] | None = None,
    expand_repetitions: bool = False,
    allow_large: bool,
) -> pd.DataFrame:
    with _open_metric_readers(
        metric,
        summary,
        districts,
        operation,
        expand_repetitions=expand_repetitions,
        allow_large=allow_large,
    ) as readers:
        if columns is not None:
            parquet = readers[0][1]
            return parquet.read(columns=columns, use_threads=False).to_pandas(use_threads=False)
        tables = [
            parquet.read(
                columns=list(_PREFIX_COLUMNS) + _table_value_columns(metric, table, districts),
                use_threads=False,
            ).to_pandas(use_threads=False)
            for table, parquet in readers
        ]
        return _combine_physical_tables(metric, tables)


@contextmanager
def _open_metric_readers(
    metric: _RunMetric,
    summary: EvaluationSummary,
    districts: tuple[int, ...],
    operation: _ReadOperation,
    *,
    expand_repetitions: bool = False,
    allow_large: bool,
    batch_size: int | None = None,
) -> Iterator[list[tuple[_RunTable, pq.ParquetFile]]]:
    selected = metric.tables[:1] if operation == "frames" else metric.tables
    with ExitStack() as stack:
        verified = [
            (table, *stack.enter_context(_verified_metric_file(metric, table)))
            for table in selected
        ]
        footer_length = sum(footer for _, _, footer in verified)
        estimate = _estimate_memory(
            metric,
            summary,
            districts,
            operation,
            expand_repetitions=expand_repetitions,
            footer_length=footer_length,
            batch_size=batch_size,
        )
        minimum = (
            _estimate_memory(
                metric,
                summary,
                districts,
                operation,
                expand_repetitions=expand_repetitions,
                footer_length=footer_length,
                batch_size=1,
            )
            if batch_size is not None
            else estimate
        )
        warned = _enforce_memory(
            metric,
            operation,
            estimate,
            allow_large,
            iterator=batch_size is not None,
            can_reduce_batch=minimum < _ERROR_BYTES <= estimate,
        )
        readers = [
            (table, _validated_parquet_file(file, metric, table, summary, districts))
            for table, file, _ in verified
        ]
        if batch_size is not None:
            largest_row_group = max(
                (
                    parquet.metadata.row_group(index).num_rows
                    for _, parquet in readers
                    for index in range(parquet.metadata.num_row_groups)
                ),
                default=0,
            )
            batch_rows = min(summary.accepted, batch_size)
            physical_rows = min(summary.accepted, max(batch_rows, largest_row_group))
            estimate = _estimate_memory(
                metric,
                summary,
                districts,
                operation,
                expand_repetitions=expand_repetitions,
                footer_length=footer_length,
                batch_size=batch_size,
                physical_rows=physical_rows,
            )
            minimum_rows = min(summary.accepted, max(min(summary.accepted, 1), largest_row_group))
            minimum = _estimate_memory(
                metric,
                summary,
                districts,
                operation,
                expand_repetitions=expand_repetitions,
                footer_length=footer_length,
                batch_size=1,
                physical_rows=minimum_rows,
            )
            _enforce_memory(
                metric,
                operation,
                estimate,
                allow_large,
                iterator=True,
                can_reduce_batch=minimum < _ERROR_BYTES <= estimate,
                warned=warned,
            )
        yield readers


@contextmanager
def _verified_metric_file(
    metric: _RunMetric,
    table: _RunTable,
) -> Iterator[tuple[BinaryIO, int]]:
    with table.path.open("rb") as file:
        size = os.fstat(file.fileno()).st_size
        if size != table.size:
            raise ValueError(f"metric {metric.name!r} table failed its integrity check")
        digest = hashlib.file_digest(file, "sha256").hexdigest()
        if digest != table.sha256:
            raise ValueError(f"metric {metric.name!r} table failed its integrity check")
        if size < 12:
            raise ValueError(f"metric {metric.name!r} has an invalid Parquet footer")
        file.seek(-8, os.SEEK_END)
        tail = file.read(8)
        if len(tail) != 8 or tail[4:] != b"PAR1":
            raise ValueError(f"metric {metric.name!r} has invalid Parquet footer magic")
        footer_length = int.from_bytes(tail[:4], "little")
        if footer_length > size - 12:
            raise ValueError(f"metric {metric.name!r} has an invalid Parquet footer length")
        file.seek(0)
        yield file, footer_length


def _validated_parquet_file(
    file: BinaryIO,
    metric: _RunMetric,
    table: _RunTable,
    summary: EvaluationSummary,
    districts: tuple[int, ...],
) -> pq.ParquetFile:
    parquet = pq.ParquetFile(file)
    if parquet.metadata.num_rows != summary.accepted:
        raise ValueError(f"metric {metric.name!r} Parquet row count disagrees with its manifest")
    schema = parquet.schema_arrow
    expected_count = len(_PREFIX_COLUMNS) + _value_column_count(
        metric,
        districts,
        table.subkeys,
    )
    if len(schema) != expected_count:
        raise ValueError(f"metric {metric.name!r} Parquet columns disagree with its manifest")
    prefix_types = (pa.uint64(), pa.uint16(), pa.uint64())
    for index, (name, dtype) in enumerate(zip(_PREFIX_COLUMNS, prefix_types, strict=True)):
        field = schema.field(index)
        if field.name != name:
            raise ValueError(f"metric {metric.name!r} Parquet columns disagree with its manifest")
        if field.type != dtype:
            raise ValueError(f"metric {metric.name!r} Parquet physical dtypes are unsupported")
    for index, name in enumerate(
        _value_column_names(metric, districts, table.subkeys),
        len(_PREFIX_COLUMNS),
    ):
        field = schema.field(index)
        if field.name != name:
            raise ValueError(f"metric {metric.name!r} Parquet columns disagree with its manifest")
        if field.type != pa.float64():
            raise ValueError(f"metric {metric.name!r} Parquet physical dtypes are unsupported")
    return parquet


def _value_column_names(
    metric: _RunMetric,
    districts: tuple[int, ...],
    subkeys: tuple[str, ...] | None = None,
) -> Iterator[str]:
    subkeys = metric.subkeys if subkeys is None else subkeys
    if metric.shape == _ResultShape.PLAN:
        yield from subkeys
        return
    for subkey in subkeys:
        for district in districts:
            yield f"{subkey}__district_{district}"


def _value_columns(metric: _RunMetric, districts: tuple[int, ...]) -> list[str]:
    return list(_value_column_names(metric, districts))


def _table_value_columns(
    metric: _RunMetric,
    table: _RunTable,
    districts: tuple[int, ...],
) -> list[str]:
    return list(_value_column_names(metric, districts, table.subkeys))


def _value_column_count(
    metric: _RunMetric,
    districts: tuple[int, ...],
    subkeys: tuple[str, ...] | None = None,
) -> int:
    subkeys = metric.subkeys if subkeys is None else subkeys
    return len(subkeys) if metric.shape == _ResultShape.PLAN else len(subkeys) * len(districts)


def _combine_physical_tables(metric: _RunMetric, tables: list[pd.DataFrame]) -> pd.DataFrame:
    first = tables[0]
    prefix = first.loc[:, list(_PREFIX_COLUMNS)]
    values = [first.iloc[:, len(_PREFIX_COLUMNS) :]]
    for table in tables[1:]:
        if not table.loc[:, list(_PREFIX_COLUMNS)].equals(prefix):
            raise ValueError(
                f"metric {metric.name!r} frame columns disagree across physical tables"
            )
        values.append(table.iloc[:, len(_PREFIX_COLUMNS) :])
    return pd.concat([prefix, *values], axis="columns")


def _iter_physical_batches(
    metric: _RunMetric,
    readers: list[tuple[_RunTable, pq.ParquetFile]],
    districts: tuple[int, ...],
    batch_size: int,
    *,
    frames_only: bool = False,
) -> Iterator[pd.DataFrame]:
    iterators = [
        parquet.iter_batches(
            batch_size=batch_size,
            columns=(
                list(_PREFIX_COLUMNS)
                if frames_only
                else list(_PREFIX_COLUMNS) + _table_value_columns(metric, table, districts)
            ),
            use_threads=False,
        )
        for table, parquet in readers
    ]
    while True:
        batches = []
        ended = 0
        for iterator in iterators:
            try:
                batches.append(next(iterator))
            except StopIteration:
                ended += 1
        if ended:
            if ended != len(iterators):
                raise ValueError(
                    f"metric {metric.name!r} physical tables have inconsistent batches"
                )
            return
        tables = [batch.to_pandas(use_threads=False) for batch in batches]
        output = tables[0] if frames_only else _combine_physical_tables(metric, tables)
        del batches, tables
        yield output
        del output


def _estimate_memory(
    metric: _RunMetric,
    summary: EvaluationSummary,
    districts: tuple[int, ...],
    operation: _ReadOperation,
    *,
    expand_repetitions: bool = False,
    footer_length: int = 0,
    batch_size: int | None = None,
    physical_rows: int | None = None,
) -> int:
    iterator = batch_size is not None
    accepted_rows = summary.accepted if batch_size is None else min(summary.accepted, batch_size)
    if physical_rows is None:
        physical_rows = accepted_rows
    logical_rows = (
        summary.samples
        if batch_size is None and expand_repetitions
        else min(summary.samples, batch_size)
        if batch_size is not None and expand_repetitions
        else accepted_rows
    )
    value_columns = _value_column_count(metric, districts)
    table_count = 1 if operation == "frames" else len(metric.tables)
    physical = physical_rows * (18 * table_count + 8 * value_columns)
    prefix_physical = physical_rows * 18 * table_count
    accepted_float = accepted_rows * 8 * value_columns
    logical_float = logical_rows * 8 * value_columns
    logical_final = logical_rows * _logical_width(metric, districts)
    frame_cache = accepted_rows * 16
    validation = accepted_rows * 9
    footer = _FOOTER_EXPANSION * footer_length

    if operation == "read":
        index = _semantic_index_bytes(metric, districts, logical_rows)
        total = 2 * physical + accepted_float + logical_float + logical_final + validation + index
        if iterator:
            total += logical_final + index
        else:
            total += frame_cache
    elif operation == "raw":
        index = 16 * (len(_PREFIX_COLUMNS) + value_columns)
        total = 2 * physical + validation + index
        if iterator:
            total += accepted_rows * (18 + 8 * value_columns) + index
        else:
            total += frame_cache
    else:
        index = 32
        total = 2 * prefix_physical + validation + index
        total += 2 * frame_cache
        if iterator:
            total += index
    return (5 * (total + footer) + 3) // 4


def _logical_width(metric: _RunMetric, districts: tuple[int, ...]) -> int:
    multiplier = 1
    if metric.shape == _ResultShape.DISTRICT:
        multiplier = len(districts)
    elif metric.shape == _ResultShape.REGION:
        multiplier = len(districts) * len(metric.regions)
    return multiplier * sum(_DTYPE_WIDTHS[dtype] for dtype in metric.dtypes)


def _semantic_index_bytes(
    metric: _RunMetric,
    districts: tuple[int, ...],
    rows: int,
) -> int:
    columns = (
        len(metric.columns)
        if metric.shape == _ResultShape.PLAN
        else len(metric.columns) * len(districts)
    )
    if metric.shape == _ResultShape.REGION:
        return 16 * (rows * len(metric.regions) + columns)
    return 16 * columns


def _enforce_memory(
    metric: _RunMetric,
    operation: _ReadOperation,
    estimated_bytes: int,
    allow_large: bool,
    *,
    iterator: bool = False,
    can_reduce_batch: bool = False,
    warned: bool = False,
) -> bool:
    if estimated_bytes < _WARNING_BYTES:
        return warned
    if iterator:
        method = {
            "frames": "iter_frame_batches",
            "raw": "iter_raw_batches",
            "read": "iter_batches",
        }[operation]
        label = f"{method}()" if operation == "frames" else f"{method}({metric.name!r})"
    else:
        label = "frames" if operation == "frames" else f"{operation}({metric.name!r})"
    estimate = _format_bytes(estimated_bytes)
    if not warned:
        _warn_external(f"EnsembleEvalResult.{label} may use approximately {estimate} of memory")
        warned = True
    if estimated_bytes < _ERROR_BYTES or allow_large:
        return warned
    if can_reduce_batch:
        advice = "use a smaller batch_size"
    elif iterator:
        advice = "use allow_large=True"
    elif operation == "frames":
        advice = "use iter_frame_batches()"
    elif operation == "raw":
        advice = "use iter_raw_batches() or allow_large=True"
    else:
        advice = "use iter_batches() or allow_large=True"
    message = (
        f"EnsembleEvalResult.{label} is estimated to use {estimate}, exceeding the "
        f"{_format_bytes(_ERROR_BYTES)} limit; {advice}"
    )
    raise EvaluationMemoryError(message, estimated_bytes, _ERROR_BYTES)


def _warn_external(message: str) -> None:
    frame = inspect.currentframe()
    stacklevel = 1
    while frame is not None and frame.f_globals.get("__name__") in {__name__, "contextlib"}:
        stacklevel += 1
        frame = frame.f_back
    del frame
    warnings.warn(message, UserWarning, stacklevel=stacklevel)


def _format_bytes(value: int) -> str:
    return f"{value / 1024**3:.2f} GiB"


def _physical_values(table: pd.DataFrame) -> NDArray[np.float64]:
    return table.iloc[:, len(_PREFIX_COLUMNS) :].to_numpy(dtype=np.float64, copy=True)


def _semantic_value(
    name: str,
    metric: _RunMetric,
    districts: tuple[int, ...],
    flat_values: NDArray[np.float64],
    index: pd.Index,
) -> _PandasResult:
    row_count = len(flat_values)
    if metric.shape == _ResultShape.REGION:
        values = flat_values.reshape(
            row_count,
            len(metric.columns),
            len(metric.regions),
            len(districts),
        )
    elif metric.shape == _ResultShape.DISTRICT:
        values = flat_values.reshape(row_count, len(metric.columns), len(districts))
    else:
        values = flat_values.reshape(row_count, len(metric.columns))
    result = _MetricResult(
        values,
        metric.shape,
        metric.columns,
        districts if metric.shape != _ResultShape.PLAN else (),
        metric.dtypes,
        metric.regions,
        metric.region_name,
    )
    return ManyPlanEvalResult._from_index({name: result}, index)[name]


def _validate_bool(value: object, name: str) -> None:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a boolean")


def _validate_batch_options(batch_size: object, allow_large: object) -> int:
    if (
        not isinstance(batch_size, numbers.Integral)
        or isinstance(batch_size, (bool, np.bool_))
        or batch_size < 1
    ):
        raise ValueError("batch_size must be a positive integer")
    _validate_bool(allow_large, "allow_large")
    return int(batch_size)


def _integer_array(series: pd.Series, label: str) -> NDArray[np.integer]:
    values = series.to_numpy(copy=False)
    if not np.issubdtype(values.dtype, np.integer) or np.issubdtype(values.dtype, np.bool_):
        raise ValueError(f"evaluation table {label} must contain nonnegative integers")
    if np.issubdtype(values.dtype, np.signedinteger) and np.any(values < 0):
        raise ValueError(f"evaluation table {label} must contain nonnegative integers")
    return cast("NDArray[np.integer]", values)


def _validated_frame_batch(
    table: pd.DataFrame,
    expected_offset: int,
    expected_accepted: int,
    *,
    return_frame: bool = False,
) -> tuple[pd.DataFrame | None, int, int]:
    if list(table.columns[: len(_PREFIX_COLUMNS)]) != list(_PREFIX_COLUMNS):
        raise ValueError("evaluation table has unsupported prefix columns")
    offsets = _integer_array(cast("pd.Series", table["sample_offset"]), "sample_offset")
    repetitions = _integer_array(cast("pd.Series", table["repetitions"]), "repetitions")
    accepted = _integer_array(cast("pd.Series", table["accepted_index"]), "accepted_index")
    if len(table) == 0:
        frame = (
            pd.DataFrame(
                {
                    "sample_offset": np.array([], dtype=np.int64),
                    "repetitions": np.array([], dtype=np.int64),
                },
                index=pd.RangeIndex(expected_accepted, expected_accepted, name="accepted"),
            )
            if return_frame
            else None
        )
        return frame, expected_offset, expected_accepted
    if int(offsets[0]) != expected_offset:
        raise ValueError("evaluation table sample offsets disagree with repetitions")
    if int(accepted[0]) != expected_accepted:
        raise ValueError("evaluation table accepted indexes are not contiguous")
    if np.any(repetitions == 0):
        raise ValueError("evaluation table repetitions must be positive")
    if len(table) > 1:
        if np.any(offsets[1:] <= offsets[:-1]) or not np.array_equal(
            offsets[1:] - offsets[:-1], repetitions[:-1]
        ):
            raise ValueError("evaluation table sample offsets disagree with repetitions")
        if np.any(accepted[1:] <= accepted[:-1]) or not np.all(accepted[1:] - accepted[:-1] == 1):
            raise ValueError("evaluation table accepted indexes are not contiguous")
    next_offset = int(offsets[-1]) + int(repetitions[-1])
    next_accepted = expected_accepted + len(table)
    frame = None
    if return_frame:
        offset_dtype = np.uint64 if int(offsets[-1]) > np.iinfo(np.int64).max else np.int64
        frame = pd.DataFrame(
            {
                "sample_offset": offsets.astype(offset_dtype, copy=True),
                "repetitions": repetitions.astype(np.int64, copy=True),
            },
            index=pd.RangeIndex(expected_accepted, next_accepted, name="accepted"),
        )
    return frame, next_offset, next_accepted


def _validate_iterator_totals(
    sample_offset: int,
    accepted: int,
    summary: EvaluationSummary,
) -> None:
    if accepted != summary.accepted:
        raise ValueError("evaluation table row count disagrees with the manifest summary")
    if sample_offset != summary.samples:
        raise ValueError("evaluation table sample offsets disagree with repetitions")


def _validated_frames(table: pd.DataFrame, summary: EvaluationSummary) -> pd.DataFrame:
    if len(table) != summary.accepted:
        raise ValueError("evaluation table row count disagrees with the manifest summary")
    frames, sample_offset, accepted = _validated_frame_batch(table, 0, 0, return_frame=True)
    assert frames is not None
    _validate_iterator_totals(sample_offset, accepted, summary)
    return frames
