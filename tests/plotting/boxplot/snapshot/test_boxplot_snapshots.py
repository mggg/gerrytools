from pathlib import Path

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from gerrytools.plotting.data.boxplot import BoxPlot
from gerrytools.plotting.mpl.marker_options import PointMarkerOptions
from tests._image_snapshots import assert_image_snapshot
from tests.plotting._snapshot_utils import RNG_SEED, render_plot

SNAPSHOTS_DIR = Path(__file__).with_name("image_snapshots")


def make_boxplot_data(rng: np.random.Generator) -> dict[str, list[float]]:
    return {
        "District A": rng.uniform(0.3, 0.9, 80).tolist(),
        "District B": rng.uniform(0.4, 0.85, 80).tolist(),
        "District C": rng.uniform(0.2, 0.75, 80).tolist(),
        "District D": rng.uniform(0.5, 0.95, 80).tolist(),
    }


class TestBoxPlotSnapshots:
    @pytest.mark.snapshot
    def test_boxplot_basic_snapshot(self, tmp_path):
        rng = np.random.default_rng(RNG_SEED)
        data = make_boxplot_data(rng)

        plot = BoxPlot(figure_size=(8, 5), dpi=100, xlabel="District", ylabel="Score")
        plot.add_dataset(data, facecolor="denim", name="Series A")
        img = render_plot(plot, tmp_path)

        assert_image_snapshot(
            img=img,
            name="boxplot_basic",
            snapshots_dir=SNAPSHOTS_DIR,
            artifacts_dir=tmp_path / "snapshot_artifacts",
        )

    @pytest.mark.snapshot
    def test_boxplot_multiple_series_snapshot(self, tmp_path):
        rng = np.random.default_rng(RNG_SEED)
        data_a = make_boxplot_data(rng)
        data_b = make_boxplot_data(rng)

        plot = BoxPlot(
            figure_size=(8, 5),
            dpi=100,
            xlabel="District",
            ylabel="Score",
            legend=True,
        )
        plot.add_dataset(data_a, facecolor="denim", name="Ensemble A")
        plot.add_dataset(data_b, facecolor="alizarin", name="Ensemble B")
        img = render_plot(plot, tmp_path)

        assert_image_snapshot(
            img=img,
            name="boxplot_multiple_series",
            snapshots_dir=SNAPSHOTS_DIR,
            artifacts_dir=tmp_path / "snapshot_artifacts",
        )

    @pytest.mark.snapshot
    def test_boxplot_with_fliers_snapshot(self, tmp_path):
        rng = np.random.default_rng(RNG_SEED)
        data = make_boxplot_data(rng)

        flier_opts = PointMarkerOptions(
            marker="x",
            markersize=4.0,
            markerfacecolor="alizarin",
            markeredgecolor="alizarin",
        )
        plot = BoxPlot(figure_size=(8, 5), dpi=100)
        plot.add_dataset(
            data,
            facecolor="applegreen",
            showfliers=True,
            flier_options=flier_opts,
            name="With Fliers",
        )
        img = render_plot(plot, tmp_path)

        assert_image_snapshot(
            img=img,
            name="boxplot_with_fliers",
            snapshots_dir=SNAPSHOTS_DIR,
            artifacts_dir=tmp_path / "snapshot_artifacts",
        )

    @pytest.mark.snapshot
    def test_boxplot_with_vlines_snapshot(self, tmp_path):
        rng = np.random.default_rng(RNG_SEED)
        data = make_boxplot_data(rng)

        plot = BoxPlot(
            figure_size=(8, 5),
            dpi=100,
            title="Boxplot with Group Lines",
        )
        plot.display_group_separators(True)
        plot.add_dataset(data, facecolor="amber", name="Series")
        img = render_plot(plot, tmp_path)

        assert_image_snapshot(
            img=img,
            name="boxplot_with_vlines",
            snapshots_dir=SNAPSHOTS_DIR,
            artifacts_dir=tmp_path / "snapshot_artifacts",
        )
