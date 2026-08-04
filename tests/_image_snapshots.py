import os
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageChops

_PERCEPTUAL_SIZE = (64, 64)
_PERCEPTUAL_MEAN_TOLERANCE = 0.015
_PERCEPTUAL_MAX_TOLERANCE = 0.10
_CONTENT_SIZE_TOLERANCE = 0.02

UPDATE = os.environ.get("UPDATE_SNAPSHOTS", "").strip() in {
    "1",
    "true",
    "TRUE",
    "yes",
    "YES",
}


def _to_rgba(img: Image.Image) -> Image.Image:
    return img.convert("RGBA")


def _diff_metrics(a: Image.Image, b: Image.Image) -> dict[str, float]:
    """Return mean/max absolute diff in [0,1]."""
    a = _to_rgba(a)
    b = _to_rgba(b)
    if a.size != b.size:
        return {"mean": 1.0, "max": 1.0}
    diff = ImageChops.difference(a, b)
    arr = np.asarray(diff).astype(np.float32) / 255.0
    return {"mean": float(arr.mean()), "max": float(arr.max())}


def _trim_and_normalize(img: Image.Image) -> tuple[Image.Image, tuple[int, int]]:
    """Trim white canvas drift and make a small perceptual comparison image."""
    rgba = _to_rgba(img)
    white = Image.new("RGBA", rgba.size, "white")
    white.alpha_composite(rgba)
    rgb = white.convert("RGB")
    bounds = ImageChops.difference(rgb, Image.new("RGB", rgb.size, "white")).getbbox()
    content = rgb.crop(bounds) if bounds is not None else rgb
    return content.resize(_PERCEPTUAL_SIZE, Image.Resampling.LANCZOS), content.size


def _sizes_are_close(first: tuple[int, int], second: tuple[int, int]) -> bool:
    return all(
        abs(left - right) / max(left, right) <= _CONTENT_SIZE_TOLERANCE
        for left, right in zip(first, second, strict=True)
    )


def assert_image_snapshot(
    *,
    img: Image.Image,
    name: str,
    snapshots_dir: Path,
    artifacts_dir: Path,
    tol_mean: float = 0.0025,
    tol_max: float = 0.08,
) -> None:
    """
    Compare `img` against golden `name.png` in `snapshots_dir` with tolerances.
    On failure, writes new/golden/diff into `artifacts_dir`. A missing golden only writes a new
    baseline in update mode (UPDATE_SNAPSHOTS=1); in verify mode it fails, so a deleted or
    never-committed baseline cannot silently pass.
    """
    if not snapshots_dir.is_absolute():
        snapshots_dir = Path(__file__).parent.parent / snapshots_dir
    snapshots_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    golden = snapshots_dir / f"{name}.png"

    if UPDATE:
        img.save(golden)
        return

    if not golden.exists():
        candidate = artifacts_dir / f"{name}.new.png"
        img.save(candidate)
        pytest.fail(
            f"Missing golden snapshot {golden}. Candidate image written to {candidate}; "
            "review it and run `task snapshots` (UPDATE_SNAPSHOTS=1) to create the baseline."
        )

    golden_img = Image.open(golden)
    metrics = _diff_metrics(img, golden_img)
    if metrics["mean"] <= tol_mean and metrics["max"] <= tol_max:
        return

    normalized_img, content_size = _trim_and_normalize(img)
    normalized_golden, golden_content_size = _trim_and_normalize(golden_img)
    perceptual_metrics = _diff_metrics(normalized_img, normalized_golden)
    perceptual_match = (
        _sizes_are_close(content_size, golden_content_size)
        and perceptual_metrics["mean"] <= _PERCEPTUAL_MEAN_TOLERANCE
        and perceptual_metrics["max"] <= _PERCEPTUAL_MAX_TOLERANCE
    )
    if not perceptual_match:
        new_path = artifacts_dir / f"{name}.new.png"
        gold_path = artifacts_dir / f"{name}.golden.png"
        diff_path = artifacts_dir / f"{name}.diff.png"

        img.save(new_path)
        golden_img.save(gold_path)
        ImageChops.difference(normalized_img, normalized_golden).save(diff_path)

        raise AssertionError(
            f"Image snapshot mismatch for {name}: "
            f"mean={metrics['mean']:.6f} max={metrics['max']:.6f} "
            f"(tol_mean={tol_mean}, tol_max={tol_max}); "
            f"perceptual_mean={perceptual_metrics['mean']:.6f} "
            f"perceptual_max={perceptual_metrics['max']:.6f}; "
            f"content_size={content_size} golden_content_size={golden_content_size}. "
            f"Artifacts: {artifacts_dir}"
        )
