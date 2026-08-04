from pathlib import Path

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from gerrytools.plotting.data.scatterplot import ScatterPlot
from tests._image_snapshots import assert_image_snapshot
from tests.plotting._snapshot_utils import RNG_SEED, render_plot

SNAPSHOTS_DIR = Path(__file__).with_name("image_snapshots")


def make_scatter_data(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    x = rng.uniform(0, 1, 60)
    y = 0.4 * x + rng.normal(0, 0.05, 60)
    return x, y


class TestScatterPlotSnapshots:
    @pytest.mark.snapshot
    def test_scatter_basic_snapshot(self, tmp_path):
        rng = np.random.default_rng(RNG_SEED)
        x, y = make_scatter_data(rng)

        plot = ScatterPlot(figure_size=(7, 6), dpi=100, xlabel="X", ylabel="Y")
        plot.add_series(x=x.tolist(), y=y.tolist(), name="Points")
        img = render_plot(plot, tmp_path)

        assert_image_snapshot(
            img=img,
            name="scatter_basic",
            snapshots_dir=SNAPSHOTS_DIR,
            artifacts_dir=tmp_path / "snapshot_artifacts",
        )

    @pytest.mark.snapshot
    def test_scatter_multiple_series_snapshot(self, tmp_path):
        rng = np.random.default_rng(RNG_SEED)
        x_a, y_a = make_scatter_data(rng)
        x_b, y_b = make_scatter_data(rng)

        plot = ScatterPlot(figure_size=(7, 6), dpi=100, legend=True)
        plot.add_series(x=x_a.tolist(), y=y_a.tolist(), name="Series A", markerfacecolor="denim")
        plot.add_series(x=x_b.tolist(), y=y_b.tolist(), name="Series B", markerfacecolor="alizarin")
        img = render_plot(plot, tmp_path)

        assert_image_snapshot(
            img=img,
            name="scatter_multiple_series",
            snapshots_dir=SNAPSHOTS_DIR,
            artifacts_dir=tmp_path / "snapshot_artifacts",
        )
