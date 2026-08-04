import matplotlib

matplotlib.use("Agg")


import pytest

from gerrytools.plotting.data.scatterplot import ScatterPlot
from tests.plotting._typing_utils import as_any


def _make_plot():
    """Create a minimal ScatterPlot with some data for testing base methods."""
    sp = ScatterPlot()
    sp.add_series(x=[0.0, 1.0], y=[0.0, 1.0])
    return sp


# ====================
# == VERTICAL LINES ==
# ====================
class TestVerticalLines:
    def test_add_single_vertical_line(self):
        sp = _make_plot()
        sp.add_vertical_lines(0.5)
        assert len(sp._annotations.vertical_lines) == 1

    def test_add_multiple_vertical_lines_from_list(self):
        sp = _make_plot()
        sp.add_vertical_lines([0.2, 0.4, 0.6])
        assert len(sp._annotations.vertical_lines) == 1
        # The single _LineData stores all three values
        assert sp._annotations.vertical_lines[0].values == (0.2, 0.4, 0.6)

    def test_add_vertical_lines_from_int(self):
        sp = _make_plot()
        sp.add_vertical_lines(5)
        assert sp._annotations.vertical_lines[0].values == (5.0,)

    def test_string_x_values_raises_typeerror(self):
        sp = _make_plot()
        with pytest.raises(TypeError, match="string"):
            sp.add_vertical_lines(as_any("bad"))

    def test_bool_x_values_raises_typeerror(self):
        sp = _make_plot()
        with pytest.raises(TypeError, match="bool"):
            sp.add_vertical_lines(True)

    def test_named_vertical_line(self):
        sp = _make_plot()
        sp.add_vertical_lines(0.5, name="Threshold")
        assert sp._annotations.vertical_lines[0].name == "Threshold"


# ======================
# == HORIZONTAL LINES ==
# ======================


class TestHorizontalLines:
    def test_add_single_horizontal_line(self):
        sp = _make_plot()
        sp.add_horizontal_lines(0.5)
        assert len(sp._annotations.horizontal_lines) == 1

    def test_add_multiple_horizontal_lines(self):
        sp = _make_plot()
        sp.add_horizontal_lines([0.25, 0.75])
        assert sp._annotations.horizontal_lines[0].values == (0.25, 0.75)

    def test_string_y_values_raises_typeerror(self):
        sp = _make_plot()
        with pytest.raises(TypeError, match="string"):
            sp.add_horizontal_lines(as_any("bad"))

    def test_bool_y_values_raises_typeerror(self):
        sp = _make_plot()
        with pytest.raises(TypeError, match="bool"):
            sp.add_horizontal_lines(True)


# ====================
# == VERTICAL BANDS ==
# ====================


class TestVerticalBands:
    def test_add_vertical_band(self):
        sp = _make_plot()
        sp.add_vertical_band(0.3, 0.7)
        assert len(sp._annotations.vertical_bands) == 1
        assert sp._annotations.vertical_bands[0].lower_bound == 0.3
        assert sp._annotations.vertical_bands[0].upper_bound == 0.7

    def test_vertical_band_auto_sorts_bounds(self):
        sp = _make_plot()
        sp.add_vertical_band(0.9, 0.1)
        assert sp._annotations.vertical_bands[0].lower_bound == 0.1
        assert sp._annotations.vertical_bands[0].upper_bound == 0.9

    def test_named_vertical_band(self):
        sp = _make_plot()
        sp.add_vertical_band(0.3, 0.7, name="Confidence Interval")
        assert sp._annotations.vertical_bands[0].name == "Confidence Interval"


# ======================
# == HORIZONTAL BANDS ==
# ======================


class TestHorizontalBands:
    def test_add_horizontal_band(self):
        sp = _make_plot()
        sp.add_horizontal_band(0.4, 0.6)
        assert len(sp._annotations.horizontal_bands) == 1

    def test_horizontal_band_auto_sorts_bounds(self):
        sp = _make_plot()
        sp.add_horizontal_band(0.8, 0.2)
        assert sp._annotations.horizontal_bands[0].lower_bound == 0.2
        assert sp._annotations.horizontal_bands[0].upper_bound == 0.8


# ===================
# == CLEAR METHODS ==
# ===================


class TestClearMethods:
    def test_clear_verticals(self):
        sp = _make_plot()
        sp.add_vertical_lines(0.5)
        sp.add_vertical_band(0.3, 0.7)
        sp.clear_verticals()
        assert len(sp._annotations.vertical_lines) == 0
        assert len(sp._annotations.vertical_bands) == 0

    def test_clear_horizontals(self):
        sp = _make_plot()
        sp.add_horizontal_lines(0.5)
        sp.add_horizontal_band(0.3, 0.7)
        sp.clear_horizontals()
        assert len(sp._annotations.horizontal_lines) == 0
        assert len(sp._annotations.horizontal_bands) == 0

    def test_clear_annotation_arrows(self):
        sp = _make_plot()
        sp.add_text_arrow((0.5, 0.5), "right")
        assert len(sp._annotations.annotation_arrows) == 1
        sp.clear_annotation_arrows()
        assert len(sp._annotations.annotation_arrows) == 0


# =====================
# == TICK MANAGEMENT ==
# =====================


class TestAxisLimits:
    def test_set_xlim(self):
        sp = _make_plot()
        sp.set_xlim(0.0, 10.0)
        assert sp._xaxis.limits == (0.0, 10.0)

    def test_set_ylim(self):
        sp = _make_plot()
        sp.set_ylim(-5.0, 5.0)
        assert sp._yaxis.limits == (-5.0, 5.0)

    def test_limits_applied_on_build(self):
        sp = _make_plot()
        sp.set_xlim(-1.0, 2.0)
        sp.set_ylim(-1.0, 2.0)
        ax = sp.ax
        assert ax.get_xlim() == (-1.0, 2.0)
        assert ax.get_ylim() == (-1.0, 2.0)


# ============================
# == LABEL AND TITLE STYLES ==
# ============================
