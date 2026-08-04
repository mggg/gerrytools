import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

from gerrytools.plotting.data.sealevel import SeaLevelPlot


# ==================
# == CONSTRUCTION ==
# ==================
class TestSeaLevelBuildPreconditions:
    def test_no_labels_raises_valueerror(self):
        sl = SeaLevelPlot()
        with pytest.raises(ValueError, match="No labels"):
            sl.ax

    def test_no_datasets_raises_valueerror(self):
        sl = SeaLevelPlot()
        sl._labels = ["A"]
        with pytest.raises(ValueError, match="No sealevel sets"):
            sl.ax


# ======================
# == CATEGORY CENTERS ==
# ======================


class TestSeaLevelCategoryCenters:
    def test_centers_are_1_indexed(self):
        sl = SeaLevelPlot()
        sl.add_dataset({"A": 0.5, "B": 0.7, "C": 0.3})
        centers = sl._category_centers
        np.testing.assert_array_equal(centers, [1.0, 2.0, 3.0])

    def test_no_labels_returns_empty(self):
        sl = SeaLevelPlot()
        assert len(sl._category_centers) == 0


# ==========
# == GRID ==
# ==========


class TestSeaLevelGrid:
    """The shared tri-state grid still toggles gridlines at build time."""

    def _plot(self):
        sl = SeaLevelPlot(legend=False)
        sl.add_dataset({"A": 0.5, "B": 0.7})
        return sl

    def test_display_grid_true_draws_gridlines(self):
        sl = self._plot()
        sl.display_grid(True)
        ax = sl.ax
        assert any(line.get_visible() for line in ax.get_xgridlines())

    def test_display_grid_false_hides_gridlines(self):
        sl = self._plot()
        sl.display_grid(True)
        _ = sl.ax
        sl.display_grid(False)
        ax = sl.ax
        assert not any(line.get_visible() for line in ax.get_xgridlines())

    def test_grid_none_leaves_external_grid_alone(self):
        sl = self._plot()
        ax = sl.ax
        ax.grid(True)
        sl.title = "force rebuild"
        ax = sl.ax
        assert any(line.get_visible() for line in ax.get_xgridlines())
        assert sl.grid is None
