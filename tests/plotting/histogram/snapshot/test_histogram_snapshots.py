from pathlib import Path

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from gerrytools.plotting.data.histogram import Histogram
from tests._image_snapshots import assert_image_snapshot
from tests.plotting._snapshot_utils import RNG_SEED, render_plot

SNAPSHOTS_DIR = Path(__file__).with_name("image_snapshots")


def make_histogram_data(rng: np.random.Generator) -> list[float]:
    return rng.normal(0.55, 0.12, 200).tolist()


class TestHistogramSnapshots:
    @pytest.mark.snapshot
    def test_histogram_basic_snapshot(self, tmp_path):
        rng = np.random.default_rng(RNG_SEED)
        data = make_histogram_data(rng)

        plot = Histogram(figure_size=(8, 5), dpi=100, xlabel="Value", ylabel="Count")
        plot.add_dataset(data, facecolor="denim", name="Distribution")
        img = render_plot(plot, tmp_path)

        assert_image_snapshot(
            img=img,
            name="histogram_basic",
            snapshots_dir=SNAPSHOTS_DIR,
            artifacts_dir=tmp_path / "snapshot_artifacts",
        )

    @pytest.mark.snapshot
    def test_histogram_overlay_snapshot(self, tmp_path):
        rng = np.random.default_rng(RNG_SEED)
        data_a = make_histogram_data(rng)
        data_b = rng.normal(0.65, 0.1, 200).tolist()

        plot = Histogram(figure_size=(8, 5), dpi=100, legend=True)
        plot.add_dataset(data_a, facecolor="denim", facealpha=0.6, name="Group A")
        plot.add_dataset(data_b, facecolor="alizarin", facealpha=0.6, name="Group B")
        img = render_plot(plot, tmp_path)

        assert_image_snapshot(
            img=img,
            name="histogram_overlay",
            snapshots_dir=SNAPSHOTS_DIR,
            artifacts_dir=tmp_path / "snapshot_artifacts",
        )

    @pytest.mark.snapshot
    def test_histogram_density_snapshot(self, tmp_path):
        rng = np.random.default_rng(RNG_SEED)
        data = make_histogram_data(rng)

        plot = Histogram(figure_size=(8, 5), dpi=100)
        plot.as_density()
        plot.add_dataset(data, facecolor="applegreen", name="Density")
        img = render_plot(plot, tmp_path)

        assert_image_snapshot(
            img=img,
            name="histogram_density",
            snapshots_dir=SNAPSHOTS_DIR,
            artifacts_dir=tmp_path / "snapshot_artifacts",
        )
