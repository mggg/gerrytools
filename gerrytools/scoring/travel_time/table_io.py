"""Safe, versioned serialization for symmetric travel-time scoring tables."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal, Mapping, cast

import numpy as np

from .od_table import TravelTimeTable

SCHEMA_VERSION = 1
TimeUnit = Literal["seconds", "minutes"]


@dataclass(frozen=True, slots=True)
class TravelTimeArtifact:
    """A loaded scoring table together with its declared units and metadata."""

    table: TravelTimeTable
    time_unit: TimeUnit
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


def _serializable_unit_ids(table: TravelTimeTable) -> list[str | int]:
    unit_ids: list[str | int] = []
    for unit_id in table.unit_ids:
        if isinstance(unit_id, str):
            unit_ids.append(unit_id)
        elif isinstance(unit_id, (int, np.integer)) and not isinstance(unit_id, bool):
            unit_ids.append(int(unit_id))
        else:
            raise TypeError(
                "Serialized travel-time unit IDs must be strings or integers; "
                f"got {unit_id!r} ({type(unit_id).__name__})"
            )
    return unit_ids


def save_travel_time_table(
    path: str | Path,
    table: TravelTimeTable,
    *,
    time_unit: TimeUnit = "seconds",
    metadata: Mapping[str, Any] | None = None,
) -> None:
    """Save a symmetric scoring table as a compressed, pickle-free NPZ artifact.

    Matrix values are stored as ``float32``. Metadata must be JSON-serializable
    and is kept separate from reserved schema fields.
    """
    table.validate_connected()
    if time_unit not in ("seconds", "minutes"):
        raise ValueError("time_unit must be 'seconds' or 'minutes'")

    unit_ids_json = json.dumps(
        _serializable_unit_ids(table), allow_nan=False, separators=(",", ":")
    )
    document = {
        "schema_version": SCHEMA_VERSION,
        "time_unit": time_unit,
        "symmetrization": "mean_of_directions",
        "matrix_dtype": "float32",
        "metadata": dict(metadata or {}),
    }
    try:
        metadata_json = json.dumps(document, allow_nan=False, separators=(",", ":"))
    except (TypeError, ValueError) as exc:
        raise TypeError("travel-time metadata must be JSON-serializable") from exc

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as file:
        np.savez_compressed(
            file,
            unit_ids_json=np.array(unit_ids_json),
            times=table.times.astype(np.float32),
            metadata_json=np.array(metadata_json),
        )


def load_travel_time_table(path: str | Path) -> TravelTimeArtifact:
    """Load and validate a scoring-table artifact without enabling pickle."""
    with np.load(Path(path), allow_pickle=False) as data:
        required = {"unit_ids_json", "times", "metadata_json"}
        missing = required.difference(data.files)
        if missing:
            raise ValueError(f"Travel-time artifact is missing fields: {sorted(missing)!r}")
        unit_ids_raw = json.loads(str(data["unit_ids_json"].item()))
        metadata_raw = json.loads(str(data["metadata_json"].item()))
        times = np.array(data["times"], dtype=float, copy=True)

    if not isinstance(unit_ids_raw, list) or not all(
        isinstance(unit_id, (str, int)) and not isinstance(unit_id, bool)
        for unit_id in unit_ids_raw
    ):
        raise ValueError("Travel-time artifact unit_ids_json must contain string or integer IDs")
    if not isinstance(metadata_raw, dict):
        raise ValueError("Travel-time artifact metadata_json must contain an object")
    if metadata_raw.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            "Unsupported travel-time artifact schema version: "
            f"{metadata_raw.get('schema_version')!r}"
        )
    if metadata_raw.get("symmetrization") != "mean_of_directions":
        raise ValueError("Travel-time artifact must declare mean-of-directions symmetrization")
    time_unit = metadata_raw.get("time_unit")
    if time_unit not in ("seconds", "minutes"):
        raise ValueError(f"Unsupported travel-time unit: {time_unit!r}")
    user_metadata = metadata_raw.get("metadata")
    if not isinstance(user_metadata, dict):
        raise ValueError("Travel-time artifact metadata field must contain an object")

    table = TravelTimeTable.from_matrix(unit_ids_raw, times)
    table.validate_connected()
    return TravelTimeArtifact(
        table=table,
        time_unit=cast(TimeUnit, time_unit),
        metadata=user_metadata,
    )
