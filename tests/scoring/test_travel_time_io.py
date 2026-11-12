"""Persistence and provenance tests for travel-time artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

from gerrytools.scoring.travel_time import (
    SnappingArtifact,
    TimeUnit,
    TravelTimeArtifact,
    TravelTimeTable,
    default_snapping_path,
    load_snapping_artifact,
    load_travel_time_table,
    save_snapping_artifact,
    save_travel_time_table,
)


def small_table() -> TravelTimeTable:
    return TravelTimeTable.from_matrix(
        ("001", "002"),
        np.array([[0.0, 12.5], [12.5, 0.0]]),
    )


def test_table_artifact_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "travel_times.npz"
    save_travel_time_table(
        path,
        small_table(),
        time_unit=TimeUnit.MINUTES,
        metadata={"state": "KS", "geography": "vtd2020"},
    )

    artifact = load_travel_time_table(path)
    assert isinstance(artifact, TravelTimeArtifact)
    assert artifact.table == small_table()
    assert artifact.time_unit is TimeUnit.MINUTES
    assert artifact.metadata == {"state": "KS", "geography": "vtd2020"}
    with pytest.raises(TypeError):
        cast(Any, artifact.metadata)["state"] = "UT"


def test_table_save_rejects_disconnected_table(tmp_path: Path) -> None:
    disconnected = TravelTimeTable.from_matrix(
        ("mainland", "island"),
        np.array([[0.0, np.nan], [np.nan, 0.0]]),
    )
    with pytest.raises(ValueError, match="requires statewide connectivity"):
        save_travel_time_table(tmp_path / "invalid.npz", disconnected)


@pytest.mark.parametrize(
    ("table", "time_unit", "metadata", "error"),
    [
        (small_table(), "hours", {}, ValueError),
        (TravelTimeTable.from_matrix((1.5,), np.zeros((1, 1))), "seconds", {}, TypeError),
        (small_table(), "seconds", {"bad": object()}, TypeError),
    ],
)
def test_table_save_rejects_unsupported_values(
    tmp_path: Path,
    table: TravelTimeTable,
    time_unit: str,
    metadata: dict[str, Any],
    error: type[Exception],
) -> None:
    with pytest.raises(error):
        save_travel_time_table(
            tmp_path / "invalid.npz",
            table,
            time_unit=time_unit,
            metadata=metadata,
        )


def _write_npz_parts(
    path: Path,
    *,
    unit_ids: object = ("001", "002"),
    metadata: object = None,
    include_times: bool = True,
    times: np.ndarray | None = None,
) -> None:
    if metadata is None:
        metadata = {
            "schema_version": 1,
            "time_unit": "seconds",
            "symmetrization": "mean_of_directions",
            "matrix_dtype": "float32",
            "metadata": {},
        }
    values: dict[str, Any] = {
        "unit_ids_json": np.array(json.dumps(unit_ids)),
        "metadata_json": np.array(json.dumps(metadata)),
    }
    if include_times:
        values["times"] = (
            np.array([[0.0, 5.0], [5.0, 0.0]], dtype=np.float32) if times is None else times
        )
    np.savez_compressed(path, **values)


def test_table_load_rejects_missing_fields(tmp_path: Path) -> None:
    path = tmp_path / "missing.npz"
    _write_npz_parts(path, include_times=False)
    with pytest.raises(ValueError, match="missing fields"):
        load_travel_time_table(path)


@pytest.mark.parametrize(
    ("unit_ids", "metadata", "message"),
    [
        ([True, 1], None, "unit_ids_json"),
        (["001", "002"], [], "metadata_json"),
        (["001", "002"], {"schema_version": 999}, "schema version"),
        (
            ["001", "002"],
            {"schema_version": 1, "symmetrization": "minimum"},
            "mean-of-directions",
        ),
        (
            ["001", "002"],
            {
                "schema_version": 1,
                "symmetrization": "mean_of_directions",
                "time_unit": "hours",
            },
            "travel-time unit",
        ),
        (
            ["001", "002"],
            {
                "schema_version": 1,
                "symmetrization": "mean_of_directions",
                "time_unit": "seconds",
                "matrix_dtype": "float64",
            },
            "matrix dtype declaration",
        ),
    ],
)
def test_table_load_rejects_invalid_schema(
    tmp_path: Path,
    unit_ids: object,
    metadata: object,
    message: str,
) -> None:
    path = tmp_path / "invalid.npz"
    _write_npz_parts(path, unit_ids=unit_ids, metadata=metadata)
    with pytest.raises(ValueError, match=message):
        load_travel_time_table(path)


def test_table_load_rejects_non_float32_matrix(tmp_path: Path) -> None:
    path = tmp_path / "float64.npz"
    times = np.array([[0.0, 5.0], [5.0, 0.0]], dtype=np.float64)
    _write_npz_parts(path, times=times)
    with pytest.raises(ValueError, match="times matrix must use float32"):
        load_travel_time_table(path)


def test_table_write_is_atomic(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from gerrytools.scoring.travel_time import table_io

    path = tmp_path / "travel_times.npz"
    path.write_bytes(b"existing artifact")

    def fail_after_partial_write(file, **values) -> None:
        del values
        file.write(b"partial replacement")
        raise OSError("simulated disk failure")

    monkeypatch.setattr(table_io.np, "savez_compressed", fail_after_partial_write)
    with pytest.raises(OSError, match="simulated disk failure"):
        save_travel_time_table(path, small_table())

    assert path.read_bytes() == b"existing artifact"
    assert list(tmp_path.iterdir()) == [path]


def test_snapping_artifact_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "travel_times.snaps.json"
    save_snapping_artifact(
        path,
        {"001": 123456, "002": "synthetic:7"},
        {"001": 18.4, "002": 8420.7},
        metadata={"state": "KS"},
    )

    artifact = load_snapping_artifact(path)
    assert isinstance(artifact, SnappingArtifact)
    assert dict(artifact.unit_to_node) == {"001": 123456, "002": "synthetic:7"}
    assert dict(artifact.snap_distances_m) == {"001": 18.4, "002": 8420.7}
    assert artifact.metadata == {"state": "KS"}
    with pytest.raises(TypeError):
        cast(Any, artifact.unit_to_node)["001"] = 0


def test_default_snapping_path_replaces_table_suffix() -> None:
    assert default_snapping_path("data/ks.npz") == Path("data/ks.snaps.json")


def test_snapping_save_requires_matching_units(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="same unit IDs"):
        save_snapping_artifact(
            tmp_path / "snaps.json",
            {"001": 123},
            {"002": 5.0},
        )


@pytest.mark.parametrize("distance", [float("nan"), float("inf"), -1.0, True, "far"])
def test_snapping_save_rejects_invalid_distance(tmp_path: Path, distance: object) -> None:
    with pytest.raises(ValueError, match="finite, nonnegative"):
        save_snapping_artifact(
            tmp_path / "snaps.json",
            {"001": 123},
            cast(Any, {"001": distance}),
        )


def test_snapping_load_rejects_duplicate_units(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "records": [
                    {"unit_id": "001", "road_node_id": 1, "snap_distance_m": 5.0},
                    {"unit_id": "001", "road_node_id": 2, "snap_distance_m": 6.0},
                ],
                "metadata": {},
            }
        )
    )
    with pytest.raises(ValueError, match="duplicate unit ID"):
        load_snapping_artifact(path)


def test_snapping_load_rejects_unknown_schema(tmp_path: Path) -> None:
    path = tmp_path / "future.json"
    path.write_text(json.dumps({"schema_version": 99, "records": [], "metadata": {}}))
    with pytest.raises(ValueError, match="schema version"):
        load_snapping_artifact(path)


def test_snapping_write_is_atomic(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from gerrytools.scoring.travel_time import provenance

    path = tmp_path / "travel_times.snaps.json"
    path.write_text("existing artifact\n")

    def fail_after_partial_write(document, file, **kwargs) -> None:
        del document, kwargs
        file.write("partial replacement")
        raise OSError("simulated disk failure")

    monkeypatch.setattr(provenance.json, "dump", fail_after_partial_write)
    with pytest.raises(OSError, match="simulated disk failure"):
        save_snapping_artifact(path, {"001": 123}, {"001": 5.0})

    assert path.read_text() == "existing artifact\n"
    assert list(tmp_path.iterdir()) == [path]
