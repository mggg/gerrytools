"""Regression tests for the shared image-snapshot helper (tests/_image_snapshots.py)."""

from pathlib import Path

import pytest
from PIL import Image

import tests._image_snapshots as image_snapshots
from tests._image_snapshots import assert_image_snapshot


def _solid_image(color: str) -> Image.Image:
    return Image.new("RGB", (4, 4), color)


def test_missing_golden_fails_and_writes_candidate(tmp_path: Path, monkeypatch):
    # Regression: a missing baseline used to be silently created and pass, so a deleted
    # golden let any image through.
    monkeypatch.setattr(image_snapshots, "UPDATE", False)
    snapshots_dir = tmp_path / "snapshots"
    artifacts_dir = tmp_path / "artifacts"

    with pytest.raises(pytest.fail.Exception, match="Missing golden snapshot"):
        assert_image_snapshot(
            img=_solid_image("red"),
            name="brand_new",
            snapshots_dir=snapshots_dir,
            artifacts_dir=artifacts_dir,
        )

    assert (artifacts_dir / "brand_new.new.png").exists()
    assert not (snapshots_dir / "brand_new.png").exists()


def test_update_mode_still_writes_baseline(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(image_snapshots, "UPDATE", True)
    snapshots_dir = tmp_path / "snapshots"

    assert_image_snapshot(
        img=_solid_image("red"),
        name="fresh",
        snapshots_dir=snapshots_dir,
        artifacts_dir=tmp_path / "artifacts",
    )

    assert (snapshots_dir / "fresh.png").exists()


def test_matching_golden_passes_in_verify_mode(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(image_snapshots, "UPDATE", False)
    snapshots_dir = tmp_path / "snapshots"
    snapshots_dir.mkdir()
    _solid_image("red").save(snapshots_dir / "same.png")

    assert_image_snapshot(
        img=_solid_image("red"),
        name="same",
        snapshots_dir=snapshots_dir,
        artifacts_dir=tmp_path / "artifacts",
    )


def test_small_white_canvas_drift_passes_perceptual_comparison(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(image_snapshots, "UPDATE", False)
    snapshots_dir = tmp_path / "snapshots"
    snapshots_dir.mkdir()
    golden = Image.new("RGB", (20, 20), "white")
    golden.paste("black", (5, 5, 15, 15))
    golden.save(snapshots_dir / "canvas.png")
    current = Image.new("RGB", (20, 21), "white")
    current.paste("black", (5, 5, 15, 15))

    assert_image_snapshot(
        img=current,
        name="canvas",
        snapshots_dir=snapshots_dir,
        artifacts_dir=tmp_path / "artifacts",
    )


def test_material_visual_change_still_fails(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(image_snapshots, "UPDATE", False)
    snapshots_dir = tmp_path / "snapshots"
    snapshots_dir.mkdir()
    _solid_image("red").save(snapshots_dir / "changed.png")

    with pytest.raises(AssertionError, match="snapshot mismatch"):
        assert_image_snapshot(
            img=_solid_image("blue"),
            name="changed",
            snapshots_dir=snapshots_dir,
            artifacts_dir=tmp_path / "artifacts",
        )
