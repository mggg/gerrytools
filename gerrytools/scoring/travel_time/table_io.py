"""Versioned persistence for symmetric travel-time scoring tables."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np

from .od_table import TravelTimeTable

TABLE_SCHEMA_VERSION = 1


class TimeUnit(StrEnum):
    """Supported units for persisted travel-time values."""

    SECONDS = "seconds"
    MINUTES = "minutes"


@dataclass(frozen=True, slots=True)
class TravelTimeArtifact:
    """Loaded scoring table with declared units and user metadata.

    Args:
        table (TravelTimeTable): Validated symmetric travel-time table.
        time_unit (TimeUnit): Unit of every value in the table.
        metadata (Mapping[str, Any]): JSON-compatible user metadata.
    """

    table: TravelTimeTable
    time_unit: TimeUnit
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


def _serializable_id(value: Any, *, label: str) -> str | int:
    if isinstance(value, str):
        return value
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        return int(value)
    raise TypeError(
        f"Serialized {label} must be a string or integer; got {value!r} ({type(value).__name__})"
    )


def _atomic_npz(
    output: Path,
    *,
    unit_ids_json: np.ndarray,
    times: np.ndarray,
    metadata_json: np.ndarray,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=output.parent,
        prefix=f".{output.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as file:
            np.savez_compressed(
                file,
                unit_ids_json=unit_ids_json,
                times=times,
                metadata_json=metadata_json,
            )
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def save_travel_time_table(
    path: str | Path,
    table: TravelTimeTable,
    *,
    time_unit: TimeUnit | str = TimeUnit.SECONDS,
    metadata: Mapping[str, Any] | None = None,
) -> None:
    """Save a connected scoring table as a compressed NPZ artifact.

    The artifact stores only numeric arrays and JSON strings, so loading never enables Python
    pickle. Matrix values are stored as ``float32``.

    Args:
        path (str | Path): Destination ``.npz`` path.
        table (TravelTimeTable): Statewide-connected scoring table.
        time_unit (TimeUnit | str): ``"seconds"`` or ``"minutes"``.
        metadata (Mapping[str, Any] | None): Optional JSON-compatible user metadata.

    Raises:
        TypeError: If unit identifiers or metadata cannot be serialized safely.
        ValueError: If the table is disconnected or ``time_unit`` is unsupported.
    """
    table.validate_connected()
    try:
        resolved_unit = TimeUnit(time_unit)
    except ValueError as exc:
        raise ValueError("time_unit must be 'seconds' or 'minutes'") from exc

    unit_ids_json = json.dumps(
        [_serializable_id(unit, label="travel-time unit ID") for unit in table.unit_ids],
        allow_nan=False,
        separators=(",", ":"),
    )
    document = {
        "schema_version": TABLE_SCHEMA_VERSION,
        "time_unit": resolved_unit.value,
        "symmetrization": "mean_of_directions",
        "matrix_dtype": "float32",
        "metadata": dict(metadata or {}),
    }
    try:
        metadata_json = json.dumps(document, allow_nan=False, separators=(",", ":"))
    except (TypeError, ValueError) as exc:
        raise TypeError("travel-time metadata must be JSON-serializable") from exc

    _atomic_npz(
        Path(path),
        unit_ids_json=np.array(unit_ids_json),
        times=table.times.astype(np.float32),
        metadata_json=np.array(metadata_json),
    )


def load_travel_time_table(path: str | Path) -> TravelTimeArtifact:
    """Load and validate a scoring-table artifact without enabling pickle.

    Args:
        path (str | Path): Existing ``.npz`` artifact.

    Returns:
        TravelTimeArtifact: Validated table, time unit, and metadata.
    """
    with np.load(Path(path), allow_pickle=False) as data:
        required = {"unit_ids_json", "times", "metadata_json"}
        missing = required.difference(data.files)
        if missing:
            raise ValueError(f"Travel-time artifact is missing fields: {sorted(missing)!r}")
        unit_ids_raw = json.loads(str(data["unit_ids_json"].item()))
        metadata_raw = json.loads(str(data["metadata_json"].item()))
        stored_times = np.array(data["times"], copy=True)

    if not isinstance(unit_ids_raw, list) or not all(
        isinstance(unit_id, (str, int)) and not isinstance(unit_id, bool)
        for unit_id in unit_ids_raw
    ):
        raise ValueError("Travel-time artifact unit_ids_json must contain string or integer IDs")
    if not isinstance(metadata_raw, dict):
        raise ValueError("Travel-time artifact metadata_json must contain an object")
    if metadata_raw.get("schema_version") != TABLE_SCHEMA_VERSION:
        raise ValueError(
            "Unsupported travel-time artifact schema version: "
            f"{metadata_raw.get('schema_version')!r}"
        )
    if metadata_raw.get("symmetrization") != "mean_of_directions":
        raise ValueError("Travel-time artifact must declare mean-of-directions symmetrization")
    try:
        time_unit = TimeUnit(metadata_raw.get("time_unit"))
    except ValueError as exc:
        raise ValueError(
            f"Unsupported travel-time unit: {metadata_raw.get('time_unit')!r}"
        ) from exc
    if metadata_raw.get("matrix_dtype") != "float32":
        raise ValueError(
            "Unsupported travel-time matrix dtype declaration: "
            f"{metadata_raw.get('matrix_dtype')!r}"
        )
    if stored_times.dtype != np.dtype(np.float32):
        raise ValueError(
            f"Travel-time artifact times matrix must use float32; got {stored_times.dtype}"
        )
    user_metadata = metadata_raw.get("metadata")
    if not isinstance(user_metadata, dict):
        raise ValueError("Travel-time artifact metadata field must contain an object")

    table = TravelTimeTable.from_matrix(unit_ids_raw, stored_times)
    table.validate_connected()
    return TravelTimeArtifact(table=table, time_unit=time_unit, metadata=user_metadata)
