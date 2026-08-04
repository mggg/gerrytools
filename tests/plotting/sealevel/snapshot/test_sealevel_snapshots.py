from pathlib import Path

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from gerrytools.plotting.data.sealevel import SeaLevelPlot
from tests._image_snapshots import assert_image_snapshot
from tests.plotting._snapshot_utils import RNG_SEED, render_plot

SNAPSHOTS_DIR = Path(__file__).with_name("image_snapshots")


def make_sealevel_data(rng: np.random.Generator) -> dict[str, float]:
    categories = ["Cat A", "Cat B", "Cat C", "Cat D", "Cat E"]
    return {c: float(v) for c, v in zip(categories, rng.uniform(0.3, 0.9, len(categories)))}


class TestSeaLevelSnapshots:
    @pytest.mark.snapshot
    def test_sealevel_basic_snapshot(self, tmp_path):
        rng = np.random.default_rng(RNG_SEED)
        data = make_sealevel_data(rng)

        plot = SeaLevelPlot(
            figure_size=(8, 5),
            dpi=100,
            xlabel="Category",
            ylabel="Score",
            jitter_rng_seed=RNG_SEED,
        )
        plot.add_dataset(data, linecolor="denim", name="Ensemble")
        img = render_plot(plot, tmp_path)

        assert_image_snapshot(
            img=img,
            name="sealevel_basic",
            snapshots_dir=SNAPSHOTS_DIR,
            artifacts_dir=tmp_path / "snapshot_artifacts",
        )

    @pytest.mark.snapshot
    def test_sealevel_multiple_series_snapshot(self, tmp_path):
        rng = np.random.default_rng(RNG_SEED)
        data_a = make_sealevel_data(rng)
        data_b = make_sealevel_data(rng)

        plot = SeaLevelPlot(
            figure_size=(8, 5),
            dpi=100,
            legend=True,
            jitter_rng_seed=RNG_SEED,
        )
        plot.add_dataset(data_a, linecolor="denim", name="Plan A")
        plot.add_dataset(data_b, linecolor="alizarin", name="Plan B")
        img = render_plot(plot, tmp_path)

        assert_image_snapshot(
            img=img,
            name="sealevel_multiple_series",
            snapshots_dir=SNAPSHOTS_DIR,
            artifacts_dir=tmp_path / "snapshot_artifacts",
        )
