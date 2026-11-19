"""Per-unit snapping provenance stored beside a travel-time table."""

from __future__ import annotations

import json
import math
import os
import tempfile
from collections.abc import Hashable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, TypeVar

from .table_io import _serializable_id

SNAPPING_SCHEMA_VERSION = 1
_UnitId = TypeVar("_UnitId", bound=Hashable)


@dataclass(frozen=True, slots=True)
class UnitSnap:
    """Road node selected to represent one geographic unit.

    Args:
        unit_id (str | int): Geographic unit identifier.
        road_node_id (str | int): Selected road-graph node identifier.
        snap_distance_m (float): Distance from the representative point to the selected node.
    """

    unit_id: str | int
    road_node_id: str | int
    snap_distance_m: float


@dataclass(frozen=True, slots=True)
class SnappingArtifact:
    """Validated snapping records and user metadata.

    Args:
        records (tuple[UnitSnap, ...]): One record per geographic unit.
        metadata (Mapping[str, Any]): JSON-compatible user metadata.
    """

    records: tuple[UnitSnap, ...]
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    @property
    def unit_to_node(self) -> Mapping[str | int, str | int]:
        """Return an immutable unit-to-road-node mapping."""
        return MappingProxyType({record.unit_id: record.road_node_id for record in self.records})

    @property
    def snap_distances_m(self) -> Mapping[str | int, float]:
        """Return an immutable unit-to-distance mapping."""
        return MappingProxyType({record.unit_id: record.snap_distance_m for record in self.records})


def default_snapping_path(table_path: str | Path) -> Path:
    """Return the conventional companion path for a table artifact.

    Args:
        table_path (str | Path): Travel-time table path.

    Returns:
        Path: Path ending in ``.snaps.json`` beside the table.
    """
    path = Path(table_path)
    return path.with_suffix(".snaps.json")


def _records_from_mappings(
    unit_to_node: Mapping[_UnitId, Hashable],
    snap_distances_m: Mapping[_UnitId, float],
) -> tuple[UnitSnap, ...]:
    missing = [unit for unit in unit_to_node if unit not in snap_distances_m]
    unexpected = [unit for unit in snap_distances_m if unit not in unit_to_node]
    if missing or unexpected:
        raise ValueError(
            "snapping mappings must contain the same unit IDs; "
            f"missing distances={missing!r}, unexpected distances={unexpected!r}"
        )

    records: list[UnitSnap] = []
    for unit, node in unit_to_node.items():
        unit_id = _serializable_id(unit, label="snapping unit ID")
        road_node_id = _serializable_id(node, label="road node ID")
        raw_distance = snap_distances_m[unit]
        if isinstance(raw_distance, (bool, complex)):
            raise ValueError(
                f"snap distance for unit {unit!r} must be a finite, nonnegative number"
            )
        try:
            distance = float(raw_distance)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"snap distance for unit {unit!r} must be a finite, nonnegative number"
            ) from exc
        if not math.isfinite(distance) or distance < 0:
            raise ValueError(
                f"snap distance for unit {unit!r} must be a finite, nonnegative number"
            )
        records.append(UnitSnap(unit_id, road_node_id, distance))
    return tuple(records)


def _validated_document(
    records: tuple[UnitSnap, ...],
    metadata: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if len({record.unit_id for record in records}) != len(records):
        raise ValueError("snapping artifact unit IDs must be unique")
    return {
        "schema_version": SNAPPING_SCHEMA_VERSION,
        "records": [
            {
                "unit_id": record.unit_id,
                "road_node_id": record.road_node_id,
                "snap_distance_m": record.snap_distance_m,
            }
            for record in records
        ],
        "metadata": dict(metadata or {}),
    }


def _atomic_json(output: Path, document: Mapping[str, Any]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=output.parent,
        prefix=f".{output.name}.",
        suffix=".tmp",
        text=True,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as file:
            json.dump(document, file, allow_nan=False, separators=(",", ":"))
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def save_snapping_artifact(
    path: str | Path,
    unit_to_node: Mapping[_UnitId, Hashable],
    snap_distances_m: Mapping[_UnitId, float],
    *,
    metadata: Mapping[str, Any] | None = None,
) -> None:
    """Save per-unit snapping provenance as versioned JSON.

    Args:
        path (str | Path): Destination ``.json`` path.
        unit_to_node (Mapping[Hashable, Hashable]): Selected road node for every unit.
        snap_distances_m (Mapping[Hashable, float]): Snap distance in meters for every unit.
        metadata (Mapping[str, Any] | None): Optional JSON-compatible user metadata.

    Raises:
        TypeError: If identifiers or metadata are not JSON-compatible.
        ValueError: If mappings disagree or distances are invalid.
    """
    records = _records_from_mappings(unit_to_node, snap_distances_m)
    document = _validated_document(records, metadata)
    try:
        json.dumps(document, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise TypeError("snapping metadata must be JSON-serializable") from exc
    _atomic_json(Path(path), document)


def load_snapping_artifact(path: str | Path) -> SnappingArtifact:
    """Load and validate per-unit snapping provenance.

    Args:
        path (str | Path): Existing snapping JSON artifact.

    Returns:
        SnappingArtifact: Validated records and metadata.
    """
    with Path(path).open(encoding="utf-8") as file:
        document = json.load(file)
    if not isinstance(document, dict):
        raise ValueError("snapping artifact must contain a JSON object")
    if document.get("schema_version") != SNAPPING_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported snapping artifact schema version: {document.get('schema_version')!r}"
        )
    raw_records = document.get("records")
    if not isinstance(raw_records, list):
        raise ValueError("snapping artifact records must contain a list")
    metadata = document.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("snapping artifact metadata must contain an object")

    unit_to_node: dict[str | int, str | int] = {}
    snap_distances: dict[str | int, float] = {}
    required = {"unit_id", "road_node_id", "snap_distance_m"}
    for index, raw_record in enumerate(raw_records):
        if not isinstance(raw_record, dict) or set(raw_record) != required:
            raise ValueError(f"snapping record {index} must contain exactly {sorted(required)!r}")
        unit = _serializable_id(raw_record["unit_id"], label="snapping unit ID")
        if unit in unit_to_node:
            raise ValueError(f"snapping artifact contains duplicate unit ID {unit!r}")
        node = _serializable_id(raw_record["road_node_id"], label="road node ID")
        unit_to_node[unit] = node
        snap_distances[unit] = raw_record["snap_distance_m"]

    records = _records_from_mappings(unit_to_node, snap_distances)
    return SnappingArtifact(records=records, metadata=metadata)
