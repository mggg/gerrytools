"""Tests for ViolinPlot behavior.

Covers: construction, data addition, label sync, build preconditions,
property accessors, legend handles.
"""

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest
from matplotlib.collections import PolyCollection

from gerrytools.plotting.data.violin import ViolinPlot


def _violin_bodies(ax):
    return [collection for collection in ax.collections if isinstance(collection, PolyCollection)]


class TestViolinPlotConstruction:
    def test_default_construction(self):
        vp = ViolinPlot()
        assert vp.group_width == 0.7
        assert vp.width_scale == 0.8

    def test_custom_width_params(self):
        vp = ViolinPlot(group_width=0.5, width_scale=0.6)
        assert vp.group_width == 0.5
        assert vp.width_scale == 0.6


class TestViolinPlotDataAddition:
    def test_add_single_dataset(self):
        vp = ViolinPlot()
        vp.add_dataset({"A": [1, 2, 3], "B": [4, 5, 6]})
        assert len(vp._violinplot_data_list) == 1
        assert vp._labels == ["A", "B"]

    def test_add_multiple_datasets_same_labels(self):
        vp = ViolinPlot()
        vp.add_dataset({"A": [1, 2], "B": [3, 4]})
        vp.add_dataset({"A": [5, 6], "B": [7, 8]})
        assert len(vp._violinplot_data_list) == 2

    def test_different_labels_raises_valueerror(self):
        vp = ViolinPlot()
        vp.add_dataset({"A": [1], "B": [2]})
        with pytest.raises(ValueError, match="labels must match"):
            vp.add_dataset({"X": [3], "Y": [4]})

    def test_add_extra_labels_with_different_labels(self):
        vp = ViolinPlot()
        vp.add_dataset({"A": [1], "B": [2]})
        vp.add_dataset({"A": [3], "C": [4]}, add_extra_labels=True)
        assert vp._labels == ["A", "B", "C"]

    def test_auto_name_increments(self):
        vp = ViolinPlot()
        vp.add_dataset({"A": [1]})
        vp.add_dataset({"A": [2]})
        assert vp._violinplot_data_list[0].name == "Set 1"
        assert vp._violinplot_data_list[1].name == "Set 2"


class TestViolinPlotBuildPreconditions:
    def test_no_labels_raises_valueerror(self):
        vp = ViolinPlot()
        with pytest.raises(ValueError, match="No labels"):
            vp.ax

    def test_no_datasets_raises_valueerror(self):
        vp = ViolinPlot()
        vp._labels = ["A"]
        with pytest.raises(ValueError, match="No violinplot sets"):
            vp.ax


class TestViolinPlotProperties:
    def test_group_width_setter(self):
        vp = ViolinPlot()
        vp.group_width = 0.9
        assert vp.group_width == 0.9

    def test_width_scale_setter(self):
        vp = ViolinPlot()
        vp.width_scale = 0.5
        assert vp.width_scale == 0.5


class TestViolinPlotLegend:
    def test_legend_includes_violin_handles(self):
        vp = ViolinPlot()
        vp.add_dataset({"A": [1, 2, 3]}, name="Ensemble")
        handles = vp._legend_handles
        labels = [h.get_label() for h in handles]
        assert "Ensemble" in labels

    def test_legend_includes_pointset_handles(self):
        vp = ViolinPlot()
        vp.add_dataset({"A": [1, 2, 3]})
        vp.add_pointset({"A": 1.5}, name="Enacted")
        handles = vp._legend_handles
        labels = [h.get_label() for h in handles]
        assert "Enacted" in labels


class TestViolinPlotActualBuilds:
    """Smoke tests for supported build configurations that only promise not to raise."""

    def test_build_with_pointset_overlay(self):
        vp = ViolinPlot()
        vp.add_dataset({"A": [1.0, 2.0, 3.0], "B": [4.0, 5.0, 6.0]})
        vp.add_pointset({"A": 2.0, "B": 5.0}, name="Enacted")
        ax = vp.ax
        assert ax is not None

    def test_build_with_group_vlines(self):
        vp = ViolinPlot()
        vp.display_group_separators(True)
        vp.add_dataset({"A": [1.0, 2.0, 3.0]})
        ax = vp.ax
        assert ax is not None

    def test_build_category_tick_labels_populated(self):
        vp = ViolinPlot()
        vp.add_dataset({"Group1": [1.0, 2.0, 3.0], "Group2": [4.0, 5.0, 6.0]})
        ax = vp.ax
        tick_labels = [t.get_text() for t in ax.get_xticklabels()]
        assert "Group1" in tick_labels

    def test_custom_tick_positions_do_not_receive_category_labels(self):
        vp = ViolinPlot()
        vp.add_dataset({"A": [1.0, 2.0], "B": [3.0, 4.0]})
        vp.set_xticks(locations=[0.0, 10.0])

        assert [tick.get_text() for tick in vp.ax.get_xticklabels()] == ["", ""]

    def test_unlabeled_data_uses_numeric_tick_labels(self):
        vp = ViolinPlot()
        vp.add_dataset([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        ax = vp.ax
        tick_labels = [t.get_text() for t in ax.get_xticklabels()]
        assert tick_labels == ["0", "1"]


# ======================================
# == EMPTY VALUE LISTS IN VIOLIN DATA ==
# ======================================
class TestViolinPlotEmptyValueLists:
    """Empty value lists are skipped without breaking the build."""

    def test_one_empty_category_is_skipped_silently(self):
        """A single empty category is skipped while other categories still draw."""
        vp = ViolinPlot(legend=False)
        vp.add_dataset({"A": [], "B": [1.0, 2.0, 3.0, 4.0]})
        ax = vp.ax
        assert ax is not None

    def test_all_empty_categories_skips_entire_set(self):
        """An all-empty dataset is skipped cleanly."""
        vp = ViolinPlot(legend=False)
        vp.add_dataset({"A": [], "B": []})
        ax = vp.ax
        assert ax is not None


class TestViolinNonFiniteFiltering:
    def test_nan_bearing_category_still_renders_body(self):
        """Regression: a NaN in a category emptied its KDE body entirely."""
        vp = ViolinPlot(legend=False)
        vp.add_dataset({"A": [1.0, 2.0, float("nan"), 3.0, 2.5], "B": [4.0, 5.0, 6.0, 5.5]})
        bodies = _violin_bodies(vp.ax)
        assert len(bodies) == 2
        assert all(len(body.get_paths()) > 0 for body in bodies)
        assert all(np.asarray(body.get_paths()[0].vertices).size > 0 for body in bodies)

    def test_nan_only_category_keeps_slot_but_draws_no_body(self):
        vp = ViolinPlot(legend=False)
        vp.add_dataset({"A": [float("nan"), float("nan")], "B": [4.0, 5.0, 6.0, 5.5]})
        bodies = _violin_bodies(vp.ax)
        assert len(bodies) == 1
        assert vp._labels == ["A", "B"]
