from pathlib import Path

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from gerrytools.plotting.data.violin import ViolinPlot
from tests._image_snapshots import assert_image_snapshot
from tests.plotting._snapshot_utils import RNG_SEED, render_plot

SNAPSHOTS_DIR = Path(__file__).with_name("image_snapshots")


def make_violin_data(rng: np.random.Generator) -> dict[str, list[float]]:
    return {
        "Plan 1": rng.normal(0.55, 0.1, 100).tolist(),
        "Plan 2": rng.normal(0.60, 0.12, 100).tolist(),
        "Plan 3": rng.normal(0.48, 0.08, 100).tolist(),
    }


class TestViolinPlotSnapshots:
    @pytest.mark.snapshot
    def test_violinplot_basic_snapshot(self, tmp_path):
        rng = np.random.default_rng(RNG_SEED)
        data = make_violin_data(rng)

        plot = ViolinPlot(figure_size=(8, 5), dpi=100, xlabel="Plan", ylabel="Score")
        plot.add_dataset(data, facecolor="denim", name="Ensemble")
        img = render_plot(plot, tmp_path)

        assert_image_snapshot(
            img=img,
            name="violinplot_basic",
            snapshots_dir=SNAPSHOTS_DIR,
            artifacts_dir=tmp_path / "snapshot_artifacts",
        )

    @pytest.mark.snapshot
    def test_violinplot_multiple_series_snapshot(self, tmp_path):
        rng = np.random.default_rng(RNG_SEED)
        data_a = make_violin_data(rng)
        data_b = make_violin_data(rng)

        plot = ViolinPlot(figure_size=(8, 5), dpi=100, legend=True)
        plot.add_dataset(data_a, facecolor="denim", name="Method A")
        plot.add_dataset(data_b, facecolor="alizarin", name="Method B")
        img = render_plot(plot, tmp_path)

        assert_image_snapshot(
            img=img,
            name="violinplot_multiple_series",
            snapshots_dir=SNAPSHOTS_DIR,
            artifacts_dir=tmp_path / "snapshot_artifacts",
        )
