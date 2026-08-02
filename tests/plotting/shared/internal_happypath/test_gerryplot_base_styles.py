import matplotlib

matplotlib.use("Agg")

import os
import tempfile

import pytest
from matplotlib.colors import to_rgba
from matplotlib.patches import Patch

from gerrytools.plotting.data.scatterplot import ScatterPlot
from gerrytools.plotting.mpl.axis_title_style import AxisLabelStyle, TitleStyle
from gerrytools.plotting.mpl.tick_style import TickStyle


def _make_plot(**kwargs):
    """Create a minimal ScatterPlot with some data for testing base methods."""
    sp = ScatterPlot(**kwargs)
    sp.add_series(x=[0.0, 1.0], y=[0.0, 1.0])
    return sp


# ====================
# == VERTICAL LINES ==
# ====================
class TestLabelAndTitleStyles:
    def test_set_axis_label_style_stores_x_style(self):
        sp = _make_plot()
        sp.set_axis_label_style("x", fontsize=14, fontweight="bold")
        assert sp._xaxis.label.style is not None
        assert isinstance(sp._xaxis.label.style, AxisLabelStyle)

    def test_set_axis_label_style_stores_y_style(self):
        sp = _make_plot()
        sp.set_axis_label_style("y", fontsize=12)
        assert sp._yaxis.label.style is not None

    def test_set_title_style_stores_style(self):
        sp = _make_plot()
        sp.set_title_style(fontsize=18, loc="left")
        assert sp._title_text.style is not None
        assert isinstance(sp._title_text.style, TitleStyle)

    def test_xlabel_rendered_on_build(self):
        sp = ScatterPlot(xlabel="X Label")
        sp.add_series(x=[0, 1], y=[0, 1])
        ax = sp.ax
        assert ax.get_xlabel() == "X Label"

    def test_ylabel_rendered_on_build(self):
        sp = ScatterPlot(ylabel="Y Label")
        sp.add_series(x=[0, 1], y=[0, 1])
        ax = sp.ax
        assert ax.get_ylabel() == "Y Label"

    def test_title_rendered_on_build(self):
        sp = ScatterPlot(title="My Title")
        sp.add_series(x=[0, 1], y=[0, 1])
        ax = sp.ax
        assert ax.get_title() == "My Title"

    def test_none_xlabel_not_rendered(self):
        sp = ScatterPlot(xlabel=None)
        sp.add_series(x=[0, 1], y=[0, 1])
        ax = sp.ax
        assert ax.get_xlabel() == ""


# =================
# == TICK STYLES ==
# =================


class TestTickStyles:
    def test_set_tick_style_x(self):
        sp = _make_plot()
        sp.set_tick_style("x", size=14, rotation=45)
        assert sp._xaxis.tick_style is not None
        assert isinstance(sp._xaxis.tick_style, TickStyle)
        assert sp._xaxis.tick_style.size == 14
        assert sp._xaxis.tick_style.rotation == 45

    def test_set_tick_style_y(self):
        sp = _make_plot()
        sp.set_tick_style("y", size=12)
        assert sp._yaxis.tick_style is not None
        assert sp._yaxis.tick_style.size == 12


# ======================
# == FRAME VISIBILITY ==
# ======================


class TestFrameVisibility:
    def test_show_or_hide_frame_stores_settings(self):
        sp = _make_plot()
        sp.set_frame_visibility(top=False, right=False, bottom=True, left=True)
        assert sp._frame_visibility["top"] is False
        assert sp._frame_visibility["right"] is False
        assert sp._frame_visibility["bottom"] is True
        assert sp._frame_visibility["left"] is True

    def test_frame_applied_on_build(self):
        sp = _make_plot()
        sp.set_frame_visibility(top=False, right=False)
        ax = sp.ax
        assert ax.spines["top"].get_visible() is False
        assert ax.spines["right"].get_visible() is False
        assert ax.spines["bottom"].get_visible() is True
        assert ax.spines["left"].get_visible() is True


# ==================================
# == ANNOTATION ARROWS (DEFERRED) ==
# ==================================


class TestGerryPlotBaseBuildWithStyles:
    """Tests that trigger the _apply_deferred_label_styles path during build."""

    def test_build_with_xlabel_style(self):
        sp = _make_plot(xlabel="x")
        sp.set_axis_label_style("x", fontsize=12.0)
        ax = sp.ax
        assert ax.get_xlabel() == "x"

    def test_build_with_ylabel_style(self):
        sp = _make_plot(ylabel="y")
        sp.set_axis_label_style("y", fontsize=12.0)
        ax = sp.ax
        assert ax.get_ylabel() == "y"

    def test_build_with_title_style(self):
        sp = _make_plot(title="T")
        sp.set_title_style(fontsize=14.0)
        ax = sp.ax
        assert ax.get_title() == "T"

    def test_build_with_minor_x_tick_style(self):
        sp = _make_plot()
        sp.set_tick_style("x", ticktype="minor")
        ax = sp.ax
        # Smoke test: an axis with no explicit minor ticks accepts minor-only styling.
        assert ax is not None

    def test_build_with_minor_y_tick_style(self):
        sp = _make_plot()
        sp.set_tick_style("y", ticktype="minor")
        ax = sp.ax
        # Smoke test: an axis with no explicit minor ticks accepts minor-only styling.
        assert ax is not None

    def test_build_with_scalar_vertical_line_covers_real_branch(self):
        sp = _make_plot()
        sp.add_vertical_lines(0.5)
        ax = sp.ax
        assert len(ax.lines) >= 1

    def test_build_with_horizontal_band(self):
        sp = _make_plot()
        sp.add_horizontal_band(0.3, 0.7, bandcolor="blue")
        ax = sp.ax
        assert ax.patches[-1].get_facecolor() == to_rgba("blue")

    def test_build_with_horizontal_band_zero_linewidth_uses_none_edgecolor(self):
        """linewidth=0.0 triggers the edgecolor='none' branch in _draw_horizontals."""
        sp = _make_plot()
        sp.add_horizontal_band(0.3, 0.7, linewidth=0.0)
        ax = sp.ax
        assert to_rgba(ax.patches[-1].get_edgecolor())[3] == 0

    def test_build_with_vertical_band_zero_linewidth_uses_none_edgecolor(self):
        sp = _make_plot()
        sp.add_vertical_band(0.3, 0.7, linewidth=0.0)
        ax = sp.ax
        assert to_rgba(ax.patches[-1].get_edgecolor())[3] == 0

    def test_build_with_named_band_no_linecolor_in_legend(self):
        """Named band with linewidth=0 should produce a 'none' edgecolor legend handle."""
        sp = _make_plot()
        sp.legend = True
        sp.add_horizontal_band(0.3, 0.7, linewidth=0.0, name="My Band")
        _ = sp.ax
        handle = next(handle for handle in sp._legend_handles if handle.get_label() == "My Band")
        assert isinstance(handle, Patch)
        assert to_rgba(handle.get_edgecolor())[3] == 0

    def test_save_to_tempfile(self):
        sp = ScatterPlot()
        sp.add_series(x=[1.0], y=[2.0])
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            tmppath = f.name
        try:
            sp.save(tmppath)
            assert os.path.getsize(tmppath) > 0
        finally:
            os.unlink(tmppath)

    def test_arrow_length_with_explicit_arrowtail_raises(self):
        from gerrytools.plotting.data import ArrowPlacement, LabelArrowOptions

        sp = ScatterPlot()
        placement_with_tail = ArrowPlacement(arrowtail=(0.3, 0.3))
        with pytest.raises(ValueError, match="arrowtail"):
            sp.add_label_arrow(
                arrowtip=(0.5, 0.5),
                direction="right",
                arrow_options=LabelArrowOptions(
                    arrow_length=10.0,
                    placement=placement_with_tail,
                ),
            )
