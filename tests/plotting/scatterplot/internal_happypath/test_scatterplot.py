"""Tests for ScatterPlot behavior.

Covers: add_series input modes, add_point, validation, build,
legend handles.
"""

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

from gerrytools.plotting.data.scatterplot import ScatterPlot
from gerrytools.plotting.mpl.marker_options import PointMarkerOptions


# ==================
# == CONSTRUCTION ==
# ==================
class TestScatterPlotConstruction:
    def test_default_construction(self):
        sp = ScatterPlot()
        assert sp._scatter_data_list == []
        assert sp.legend is False


# =================
# == ADD SCATTER ==
# =================
class TestAddScatter:
    def test_add_scatter_with_x_and_y(self):
        sp = ScatterPlot()
        sp.add_series(x=[1.0, 2.0], y=[3.0, 4.0])
        assert len(sp._scatter_data_list) == 1

    def test_add_scatter_with_xy_pairs(self):
        sp = ScatterPlot()
        sp.add_series(xy_pairs=[(1.0, 2.0), (3.0, 4.0)])
        assert len(sp._scatter_data_list) == 1
        sd = sp._scatter_data_list[0]
        np.testing.assert_array_equal(sd.x, [1.0, 3.0])
        np.testing.assert_array_equal(sd.y, [2.0, 4.0])

    def test_add_scatter_xy_pairs_and_x_raises_valueerror(self):
        sp = ScatterPlot()
        with pytest.raises(ValueError, match="not both"):
            sp.add_series(x=[1.0], xy_pairs=[(1.0, 2.0)])

    def test_add_scatter_empty_xy_pairs_raises_valueerror(self):
        sp = ScatterPlot()
        with pytest.raises(ValueError, match="must not be empty"):
            sp.add_series(xy_pairs=[])

    def test_add_scatter_xy_pairs_and_y_raises_valueerror(self):
        sp = ScatterPlot()
        with pytest.raises(ValueError, match="not both"):
            sp.add_series(y=[1.0], xy_pairs=[(1.0, 2.0)])

    def test_add_scatter_neither_x_nor_xy_raises_valueerror(self):
        sp = ScatterPlot()
        with pytest.raises(ValueError, match="must be provided"):
            sp.add_series()

    def test_add_scatter_x_only_no_y_raises_valueerror(self):
        sp = ScatterPlot()
        with pytest.raises(ValueError, match="must be provided"):
            sp.add_series(x=[1.0])

    def test_add_scatter_y_only_no_x_raises_valueerror(self):
        sp = ScatterPlot()
        with pytest.raises(ValueError, match="must be provided"):
            sp.add_series(y=[1.0])

    def test_add_scatter_mismatched_lengths_raises_valueerror(self):
        sp = ScatterPlot()
        with pytest.raises(ValueError, match="same shape"):
            sp.add_series(x=[1.0, 2.0], y=[3.0])

    def test_add_scatter_empty_arrays_raises_valueerror(self):
        sp = ScatterPlot()
        with pytest.raises(ValueError, match="not be empty"):
            sp.add_series(x=[], y=[])

    def test_add_scatter_with_label(self):
        sp = ScatterPlot()
        sp.add_series(x=[1.0], y=[2.0], name="Point A")
        assert sp._scatter_data_list[0].name == "Point A"

    def test_add_scatter_without_label(self):
        sp = ScatterPlot()
        sp.add_series(x=[1.0], y=[2.0])
        assert sp._scatter_data_list[0].name is None

    def test_default_markeredgecolor_is_none_string(self):
        sp = ScatterPlot()
        sp.add_series(x=[1.0], y=[2.0])
        # When markeredgecolor is None at the call site, it becomes "none"
        sd = sp._scatter_data_list[0]
        assert isinstance(sd.marker_options.markeredgecolor, str)
        assert sd.marker_options.markeredgecolor.lower() == "none"

    def test_explicit_markeredgecolor_is_preserved(self):
        sp = ScatterPlot()
        sp.add_series(x=[1.0], y=[2.0], markeredgecolor="red")
        sd = sp._scatter_data_list[0]
        assert sd.marker_options.markeredgecolor != "none"
        assert sd.marker_options.markeredgealpha == 1.0
        assert sd.marker_options.markeredgewidth == 0.8

    def test_explicit_none_colors_remove_fill_and_edge(self):
        sp = ScatterPlot()
        sp.add_series(
            x=[1.0],
            y=[2.0],
            markerfacecolor=None,
            markeredgecolor=None,
        )

        marker = sp._scatter_data_list[0].marker_options
        assert marker.markerfacecolor == "none"
        assert marker.markerfacealpha == 0.0
        assert marker.markeredgecolor == "none"
        assert marker.markeredgealpha == 0.0
        assert marker.markeredgewidth == 0.0

    def test_explicit_zero_markeredgewidth_keeps_edge_hidden(self):
        sp = ScatterPlot()
        sp.add_series(x=[1.0], y=[2.0], markeredgecolor="red", markeredgewidth=0)
        assert sp._scatter_data_list[0].marker_options.markeredgewidth == 0

    def test_marker_options_are_snapshotted_when_added(self):
        options = PointMarkerOptions(markersize=5)
        sp = ScatterPlot()
        sp.add_series(x=[1.0], y=[2.0], marker_options=options)

        options.markersize = 99

        assert sp._scatter_data_list[0].marker_options.markersize == 5


# ===============
# == ADD POINT ==
# ===============
class TestAddPoint:
    def test_add_single_point(self):
        sp = ScatterPlot()
        sp.add_point(0.5, 0.5, name="Center")
        assert len(sp._scatter_data_list) == 1
        sd = sp._scatter_data_list[0]
        np.testing.assert_array_equal(sd.x, [0.5])
        np.testing.assert_array_equal(sd.y, [0.5])
        assert sd.name == "Center"

    def test_add_multiple_points(self):
        sp = ScatterPlot()
        sp.add_point(0.0, 0.0, name="Origin")
        sp.add_point(1.0, 1.0, name="Corner")
        assert len(sp._scatter_data_list) == 2

    def test_markeredgecolor_alone_makes_edge_visible(self):
        sp = ScatterPlot()
        sp.add_point(0.5, 0.5, name="Center", markeredgecolor="black")

        marker = sp._scatter_data_list[0].marker_options
        assert marker.markeredgecolor != "none"
        assert marker.markeredgealpha == 1.0
        assert marker.markeredgewidth == 0.8

    def test_explicit_zero_markeredgewidth_keeps_edge_hidden(self):
        sp = ScatterPlot()
        sp.add_point(
            0.5,
            0.5,
            name="Center",
            markeredgecolor="black",
            markeredgewidth=0,
        )

        assert sp._scatter_data_list[0].marker_options.markeredgewidth == 0

    def test_explicit_zero_markeredgealpha_keeps_edge_transparent(self):
        sp = ScatterPlot()
        sp.add_point(
            0.5,
            0.5,
            name="Center",
            markeredgecolor="black",
            markeredgealpha=0,
        )

        assert sp._scatter_data_list[0].marker_options.markeredgealpha == 0


# =======================
# == BUILD AND DRAWING ==
# =======================
class TestScatterPlotBuild:
    def test_build_with_no_data_raises(self):
        sp = ScatterPlot()
        with pytest.raises(ValueError, match="No data added yet"):
            _ = sp.ax

    def test_build_with_data_succeeds(self):
        sp = ScatterPlot()
        sp.add_series(x=[1.0, 2.0, 3.0], y=[4.0, 5.0, 6.0], name="data")
        ax = sp.ax
        assert ax is not None


# ====================
# == LEGEND HANDLES ==
# ====================
class TestScatterPlotLegend:
    def test_labeled_scatter_appears_in_legend(self):
        sp = ScatterPlot()
        sp.add_series(x=[1.0], y=[2.0], name="A")
        handles = sp._legend_handles
        labels = [h.get_label() for h in handles]
        assert "A" in labels

    def test_unlabeled_scatter_excluded_from_legend(self):
        sp = ScatterPlot()
        sp.add_series(x=[1.0], y=[2.0])  # no label
        handles = sp._legend_handles
        assert len(handles) == 0

    def test_mix_of_labeled_and_unlabeled(self):
        sp = ScatterPlot()
        sp.add_series(x=[1.0], y=[2.0], name="Labeled")
        sp.add_series(x=[3.0], y=[4.0])  # no label
        handles = sp._legend_handles
        assert len(handles) == 1
        assert handles[0].get_label() == "Labeled"
