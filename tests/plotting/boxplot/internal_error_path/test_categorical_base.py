"""Tests for categorical base edge cases."""

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest
from matplotlib.patches import Patch

from gerrytools.plotting.data.boxplot import BoxPlot


# ======================================
# == TICK LABEL MISMATCH RETURNS NONE ==
# ======================================
class TestCategoricalBaseTickLabelMismatch:
    """The default tick-label helper returns `None` when counts differ."""

    def test_custom_tick_locations_use_position_aware_labels(self):
        bp = BoxPlot(legend=False)
        bp.add_dataset({"A": [1.0, 2.0], "B": [3.0, 4.0]})
        bp.set_xticks(locations=[0.5, 1.0, 1.5, 2.0, 2.5])
        ax = bp.ax
        assert [tick.get_text() for tick in ax.get_xticklabels()] == ["", "A", "", "B", ""]

    def test_custom_category_positions_do_not_reindex_labels(self):
        bp = BoxPlot(legend=False)
        bp.add_dataset({"alpha": [1.0], "beta": [2.0], "gamma": [3.0]})
        bp.set_xticks(locations=[2.0, 3.0])

        assert [tick.get_text() for tick in bp.ax.get_xticklabels()] == ["beta", "gamma"]
        assert bp.ax.get_xlim() == (0.5, 3.5)

    def test_default_labels_applied_when_count_matches(self):
        """When tick count matches label count, _default_x_tick_labels returns labels."""
        bp = BoxPlot(legend=False)
        bp.add_dataset({"A": [1.0, 2.0], "B": [3.0, 4.0]})
        # Default tick locations should match (2 categories → 2 ticks)
        ax = bp.ax
        tick_labels = [t.get_text() for t in ax.get_xticklabels()]
        assert "A" in tick_labels or "B" in tick_labels


def test_patch_legend_matches_dataset_edgewidth():
    bp = BoxPlot()
    bp.add_dataset({"A": [1.0, 2.0]}, name="set", edgewidth=3.0)

    handle = bp._legend_handles[0]
    assert isinstance(handle, Patch)
    assert handle.get_linewidth() == 3.0


# =================================
# == ALL-NAN POINTSET IS SKIPPED ==
# =================================
class TestCategoricalBaseAllNaNPointset:
    """All-NaN pointsets are skipped cleanly."""

    def test_all_nan_pointset_is_skipped(self):
        """A pointset with only NaN values is ignored during drawing."""
        bp = BoxPlot(legend=False)
        bp.add_dataset({"A": [1.0, 2.0], "B": [3.0, 4.0]})
        # Add a pointset where all values are NaN for every label
        bp.add_pointset({"A": float("nan"), "B": float("nan")})
        ax = bp.ax
        assert not any(line.get_marker() == "o" for line in ax.lines)

    def test_partial_nan_pointset_draws_non_nan_points(self):
        """A pointset with some NaN values only draws the non-NaN ones."""
        bp = BoxPlot(legend=False)
        bp.add_dataset({"A": [1.0, 2.0], "B": [3.0, 4.0]})
        bp.add_pointset({"A": 1.5, "B": float("nan")})
        ax = bp.ax
        point_lines = [line for line in ax.lines if line.get_marker() == "o"]
        assert len(point_lines) == 1
        assert np.asarray(point_lines[0].get_ydata()).tolist() == [1.5]


# ================================
# == POINTSET OFFSET VALIDATION ==
# ================================
class TestCategoricalBasePointsetOffsetValidation:
    """Non-finite x offsets fail at add time instead of silently drawing nothing."""

    def test_nan_x_offset_raises_valueerror(self):
        bp = BoxPlot(legend=False)
        bp.add_dataset({"A": [1.0, 2.0]})
        with pytest.raises(ValueError, match="x_offset must be finite"):
            bp.add_pointset({"A": 1.5}, x_offset=float("nan"))

    def test_infinite_x_offset_raises_valueerror(self):
        bp = BoxPlot(legend=False)
        bp.add_dataset({"A": [1.0, 2.0]})
        with pytest.raises(ValueError, match="x_offset must be finite"):
            bp.add_pointset({"A": 1.5}, x_offset=float("-inf"))

    def test_finite_x_offset_still_accepted(self):
        bp = BoxPlot(legend=False)
        bp.add_dataset({"A": [1.0, 2.0]})
        bp.add_pointset({"A": 1.5}, x_offset=0.1)
        assert bp._pointset_data_list[0].x_offset == 0.1
