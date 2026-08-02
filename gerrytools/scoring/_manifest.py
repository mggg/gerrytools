"""Validation and parsing for streamed scoring manifests."""

from __future__ import annotations

import numbers
import re
from collections.abc import Hashable
from pathlib import Path, PurePosixPath
from typing import cast

import numpy as np

from ._types import (
    EvaluationSummary,
    _Dtype,
    _ResultShape,
    _RunMetric,
    _RunTable,
    is_valid_metric_name,
)

_PANDAS_DTYPES = {"bool": "bool", "float": "float64", "int": "int64"}


def preferred_manifest_path(path: Path) -> Path:
    """Return the manifest path used for a newly written run."""
    return path / f"manifest__{_encode_path_component(path.name)}.json"


def find_manifest_path(path: Path) -> Path:
    """Find a current, renamed, or version-one run manifest."""
    preferred = preferred_manifest_path(path)
    if preferred.is_file():
        return preferred
    legacy = path / "manifest.json"
    if legacy.is_file():
        return legacy
    candidates = [candidate for candidate in path.glob("manifest__*.json") if candidate.is_file()]
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        raise ValueError("evaluation run contains multiple possible manifests")
    return preferred


def parse_manifest(
    path: Path,
    manifest: object,
) -> tuple[EvaluationSummary, tuple[int, ...], tuple[_RunMetric, ...]]:
    """Parse a scoring manifest after validating all paths and logical axes."""
    if not isinstance(manifest, dict):
        raise ValueError("evaluation manifest must be an object")
    data = cast("dict[str, object]", manifest)
    format_version = data.get("format_version")
    if isinstance(format_version, bool) or format_version not in {1, 2}:
        raise ValueError("evaluation manifest must use format version 1 or 2")
    format_version = cast(int, format_version)
    summary_data = data.get("summary")
    if not isinstance(summary_data, dict):
        raise ValueError("evaluation manifest requires a summary")
    summary_data = cast("dict[str, object]", summary_data)
    samples = _nonnegative_int(summary_data.get("samples"), "summary.samples")
    accepted = _nonnegative_int(summary_data.get("accepted"), "summary.accepted")
    if accepted > samples:
        raise ValueError("evaluation manifest accepted frames cannot exceed samples")
    has_unique_plans = "unique_plans" in summary_data
    has_unique_districts = "unique_districts" in summary_data
    if has_unique_plans != has_unique_districts:
        raise ValueError("evaluation manifest unique plan and district counts must appear together")
    unique_plans = (
        _nonnegative_int(summary_data["unique_plans"], "summary.unique_plans")
        if has_unique_plans
        else None
    )
    unique_districts = (
        _nonnegative_int(summary_data["unique_districts"], "summary.unique_districts")
        if has_unique_districts
        else None
    )

    expected_prefix = [
        {"name": "sample_offset", "dtype": "uint64"},
        {"name": "repetitions", "dtype": "uint16"},
        {"name": "accepted_index", "dtype": "uint64"},
    ]
    if data.get("prefix_columns") != expected_prefix:
        raise ValueError("evaluation manifest has unsupported prefix columns")

    district_data = data.get("district_ids")
    if not isinstance(district_data, list):
        raise ValueError("evaluation manifest requires district_ids")
    districts = tuple(_nonnegative_int(value, "district id") for value in district_data)
    if len(set(districts)) != len(districts):
        raise ValueError("evaluation manifest district_ids must be unique")
    if unique_plans is not None:
        if accepted == 0:
            if unique_plans != 0 or unique_districts != 0:
                raise ValueError("an empty evaluation run cannot contain unique plans or districts")
        else:
            if not 1 <= unique_plans <= accepted:
                raise ValueError(
                    "evaluation manifest unique plans must be between one and accepted frames"
                )
            assert unique_districts is not None
            if not len(districts) <= unique_districts <= accepted * len(districts):
                raise ValueError("evaluation manifest unique district count is inconsistent")
    summary = EvaluationSummary(samples, accepted, unique_plans, unique_districts)

    metric_data = data.get("metrics")
    if not isinstance(metric_data, list) or not metric_data:
        raise ValueError("evaluation manifest requires at least one metric")
    metrics = tuple(_parse_run_metric(path, value, format_version) for value in metric_data)
    if len({metric.name for metric in metrics}) != len(metrics):
        raise ValueError("evaluation manifest metric names must be unique")
    return summary, districts, metrics


def _parse_run_metric(path: Path, value: object, format_version: int) -> _RunMetric:
    if not isinstance(value, dict):
        raise ValueError("evaluation manifest metric entries must be objects")
    data = cast("dict[str, object]", value)
    name = data.get("instance")
    if not is_valid_metric_name(name):
        raise ValueError("evaluation manifest contains an invalid metric name")
    assert isinstance(name, str)
    try:
        shape = _ResultShape(data.get("shape"))
    except (TypeError, ValueError):
        raise ValueError(f"metric {name!r} has an unsupported shape") from None
    subkeys = _string_tuple(data.get("subkeys"), f"metric {name!r} subkeys")
    axes = data.get("axes")
    regions: tuple[Hashable, ...] = ()
    region_name: str | None = None
    if shape == _ResultShape.REGION:
        if not isinstance(axes, dict):
            raise ValueError(f"region metric {name!r} requires axis metadata")
        axes = cast("dict[str, object]", axes)
        columns = _string_tuple(axes.get("metric"), f"metric {name!r} axis")
        if not columns:
            raise ValueError(f"region metric {name!r} requires metric-axis values")
        region = axes.get("region")
        if not isinstance(region, dict):
            raise ValueError(f"region metric {name!r} requires a region axis")
        region = cast("dict[str, object]", region)
        region_name_value = region.get("name")
        if not isinstance(region_name_value, str) or not region_name_value:
            raise ValueError(f"region metric {name!r} requires a region-axis name")
        region_name = region_name_value
        labels = region.get("labels")
        if not isinstance(labels, list):
            raise ValueError(f"region metric {name!r} requires region labels")
        regions = tuple(_region_label(label, name) for label in labels)
        if len(set(regions)) != len(regions):
            raise ValueError(f"region metric {name!r} has duplicate region labels")
        expected_subkeys = tuple(
            f"{column}__region_{region_index}"
            for column in columns
            for region_index in range(len(regions))
        )
        if subkeys != expected_subkeys:
            raise ValueError(f"region metric {name!r} subkeys disagree with its axes")
    else:
        if not isinstance(axes, dict):
            raise ValueError(f"metric {name!r} requires axis metadata")
        axes = cast("dict[str, object]", axes)
        if axes.get("region") is not None:
            raise ValueError(f"non-region metric {name!r} cannot define region axes")
        columns = _string_tuple(axes.get("metric"), f"metric {name!r} axis")
        if not subkeys or columns != subkeys:
            raise ValueError(f"metric {name!r} axis values disagree with its subkeys")

    tables = (
        _parse_v1_table(path, data, name, subkeys)
        if format_version == 1
        else _parse_v2_tables(path, data, name, data.get("kind"), subkeys, columns)
    )
    return _RunMetric(
        name,
        shape,
        tables,
        subkeys,
        columns,
        _dtype_tuple(data.get("dtypes"), name, len(columns)),
        regions,
        region_name,
    )


def _parse_v1_table(
    path: Path,
    data: dict[str, object],
    name: str,
    subkeys: tuple[str, ...],
) -> tuple[_RunTable, ...]:
    expected = f"{name}/scores.parquet"
    if data.get("table") != expected:
        raise ValueError(f"metric {name!r} has an unsafe or unsupported table path")
    return (
        _run_table(
            path,
            name,
            expected,
            subkeys,
            data.get("table_size"),
            data.get("table_sha256"),
        ),
    )


def _parse_v2_tables(
    path: Path,
    data: dict[str, object],
    name: str,
    kind: object,
    subkeys: tuple[str, ...],
    columns: tuple[str, ...],
) -> tuple[_RunTable, ...]:
    entries = data.get("tables")
    if not isinstance(kind, str) or not kind or not isinstance(entries, list) or not entries:
        raise ValueError(f"metric {name!r} requires physical tables")
    tables = []
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError(f"metric {name!r} has an invalid physical table")
        entry = cast("dict[str, object]", entry)
        table_subkeys = _string_tuple(entry.get("subkeys"), f"metric {name!r} table subkeys")
        tables.append(
            _run_table(
                path,
                name,
                entry.get("path"),
                table_subkeys,
                entry.get("size"),
                entry.get("sha256"),
            )
        )
    result = tuple(tables)
    if tuple(subkey for table in result for subkey in table.subkeys) != subkeys:
        raise ValueError(f"metric {name!r} tables disagree with its subkeys")
    relative_paths = tuple(table.path.relative_to(path).as_posix() for table in result)
    if len(set(relative_paths)) != len(relative_paths):
        raise ValueError(f"metric {name!r} repeats a physical table path")
    if (
        len(result) == 1
        and result[0].subkeys == subkeys
        and relative_paths == (f"{name}/scores.parquet",)
    ):
        return result
    if kind == "tally":
        run_names = (
            {
                _grouped_table_run_name(relative, name, f"{_encode_path_component(subkey)}_tallies")
                for relative, subkey in zip(relative_paths, subkeys, strict=True)
            }
            if len(result) == len(subkeys)
            else {None}
        )
        if (
            None in run_names
            or len(run_names) != 1
            or any(
                table.subkeys != (subkey,) for table, subkey in zip(result, subkeys, strict=True)
            )
        ):
            raise ValueError(f"tally metric {name!r} has an unsupported table layout")
    elif kind == "tally_by_region":
        run_names = (
            {
                _grouped_table_run_name(
                    relative,
                    name,
                    f"{_encode_path_component(column)}_tallies_by_region",
                )
                for relative, column in zip(relative_paths, columns, strict=True)
            }
            if len(result) == len(columns)
            else {None}
        )
        region_count = len(subkeys) // len(columns)
        if (
            None in run_names
            or len(run_names) != 1
            or len(result) != len(columns)
            or any(
                table.subkeys
                != subkeys[column_index * region_count : (column_index + 1) * region_count]
                for column_index, table in enumerate(result)
            )
        ):
            raise ValueError(f"tally-by-region metric {name!r} has an unsupported table layout")
    elif (
        len(result) != 1
        or len(PurePosixPath(relative_paths[0]).parts) != 1
        or not PurePosixPath(relative_paths[0]).name.startswith(f"{name}__")
        or PurePosixPath(relative_paths[0]).suffix != ".parquet"
        or not PurePosixPath(relative_paths[0]).stem.removeprefix(f"{name}__")
    ):
        raise ValueError(f"metric {name!r} has an unsupported table layout")
    return result


def _grouped_table_run_name(relative: str, directory: str, stem: str) -> str | None:
    path = PurePosixPath(relative)
    if len(path.parts) != 2 or path.parent.as_posix() != directory:
        return None
    prefix = f"{stem}__"
    if not path.name.startswith(prefix) or path.suffix != ".parquet":
        return None
    return path.stem.removeprefix(prefix) or None


def _encode_path_component(value: str) -> str:
    encoded = []
    for byte in value.encode():
        encoded.append(
            chr(byte) if byte < 128 and (chr(byte).isalnum() or byte in b"_-.") else f"%{byte:02X}"
        )
    return "".join(encoded)


def _run_table(
    root: Path,
    metric: str,
    relative: object,
    subkeys: tuple[str, ...],
    size: object,
    sha256: object,
) -> _RunTable:
    if not isinstance(relative, str) or "\\" in relative:
        raise ValueError(f"metric {metric!r} has an unsafe or unsupported table path")
    parsed = PurePosixPath(relative)
    if parsed.is_absolute() or any(part in {"", ".", ".."} for part in parsed.parts):
        raise ValueError(f"metric {metric!r} has an unsafe or unsupported table path")
    table = root.joinpath(*parsed.parts)
    if not table.is_file():
        raise FileNotFoundError(f"metric {metric!r} table does not exist: {table}")
    table_size = _nonnegative_int(size, f"metric {metric!r} table size")
    if not isinstance(sha256, str) or re.fullmatch(r"[0-9a-f]{64}", sha256) is None:
        raise ValueError(f"metric {metric!r} requires a valid table SHA-256 digest")
    return _RunTable(table, subkeys, table_size, sha256)


def _string_tuple(value: object, label: str) -> tuple[str, ...]:
    if (
        not isinstance(value, list)
        or any(not isinstance(item, str) or not item for item in value)
        or len(set(value)) != len(value)
    ):
        raise ValueError(f"{label} must contain unique nonempty strings")
    return cast("tuple[str, ...]", tuple(value))


def _dtype_tuple(value: object, name: str, count: int) -> tuple[_Dtype, ...]:
    if (
        not isinstance(value, list)
        or len(value) != count
        or any(dtype not in _PANDAS_DTYPES for dtype in value)
    ):
        raise ValueError(f"metric {name!r} has invalid logical dtypes")
    return cast("tuple[_Dtype, ...]", tuple(value))


def _region_label(value: object, name: str) -> Hashable:
    if not isinstance(value, dict) or set(value) != {"kind", "value"}:
        raise ValueError(f"region metric {name!r} has an invalid region label")
    data = cast("dict[str, object]", value)
    kind = data["kind"]
    label = data["value"]
    if kind == "str" and isinstance(label, str):
        return label
    if kind == "int" and isinstance(label, int) and not isinstance(label, bool):
        return label
    raise ValueError(f"region metric {name!r} has an invalid region label")


def _nonnegative_int(value: object, label: str) -> int:
    if not isinstance(value, numbers.Integral) or isinstance(value, (bool, np.bool_)) or value < 0:
        raise ValueError(f"evaluation manifest {label} must be a nonnegative integer")
    return int(value)
