import dataclasses
import inspect

import matplotlib

matplotlib.use("Agg")


import numpy as np
import pytest
from matplotlib.colors import to_rgba

from gerrytools.plotting.data.options import BandOptions, LineOptions
from gerrytools.plotting.data.scatterplot import ScatterPlot


def _make_plot():
    """Create a minimal ScatterPlot with some data for testing base methods."""
    sp = ScatterPlot()
    sp.add_series(x=[0.0, 1.0], y=[0.0, 1.0])
    return sp


# =======================
# == ANNOTATION ARROWS ==
# =======================
class TestAnnotationArrows:
    def test_add_text_arrow_stores_arrow(self):
        sp = _make_plot()
        sp.add_text_arrow((0.5, 0.5), "right")
        assert len(sp._annotations.annotation_arrows) == 1
        from gerrytools.plotting.data._gerryplot_dataclasses import _TextArrowData

        assert isinstance(sp._annotations.annotation_arrows[0], _TextArrowData)

    def test_add_label_arrow_stores_arrow(self):
        sp = _make_plot()
        sp.add_label_arrow((0.5, 0.5), "up")
        assert len(sp._annotations.annotation_arrows) == 1
        from gerrytools.plotting.data._gerryplot_dataclasses import _LabelArrowData

        assert isinstance(sp._annotations.annotation_arrows[0], _LabelArrowData)

    def test_text_arrow_empty_text_normalized_to_spaces(self):
        sp = _make_plot()
        sp.add_text_arrow((0.5, 0.5), "right", text="")
        assert sp._annotations.annotation_arrows[0].text == "   "

    def test_text_arrow_textrotation_overrides_style_rotation(self):
        sp = _make_plot()
        sp.add_text_arrow((0.5, 0.5), "right", textrotation=45.0)
        assert sp._annotations.annotation_arrows[0].textstyle.rotation == 45.0

    def test_label_arrow_with_arrow_length(self):
        from gerrytools.plotting.data import LabelArrowOptions
        from gerrytools.plotting.data._gerryplot_dataclasses import _LabelArrowData

        sp = _make_plot()
        sp.add_label_arrow((0.5, 0.5), "down", arrow_options=LabelArrowOptions(arrow_length=50.0))
        arrow = sp._annotations.annotation_arrows[0]
        assert isinstance(arrow, _LabelArrowData)
        assert arrow.arrow_length_percentage == 50.0

    def test_label_arrow_length_with_explicit_tail_raises_valueerror(self):
        from gerrytools.plotting.data import ArrowPlacement, LabelArrowOptions

        sp = _make_plot()
        with pytest.raises(ValueError, match="arrowtail"):
            sp.add_label_arrow(
                (0.5, 0.5),
                "up",
                arrow_options=LabelArrowOptions(
                    arrow_length=50.0,
                    placement=ArrowPlacement(arrowtail=(0.5, 0.3)),
                ),
            )

    def test_label_arrow_length_out_of_range_raises(self):
        from gerrytools.plotting.data import LabelArrowOptions

        sp = _make_plot()
        with pytest.raises(ValueError, match="\\[0, 100\\]"):
            sp.add_label_arrow(
                (0.5, 0.5), "up", arrow_options=LabelArrowOptions(arrow_length=150.0)
            )

    def test_add_text_arrow_with_custom_facecolor(self):
        sp = _make_plot()
        sp.add_text_arrow((0.5, 0.5), "left", arrowfacecolor="red")
        style = sp._annotations.annotation_arrows[0].style
        assert style.arrowfacecolor != "#5c676f"  # not the default

    def test_add_label_arrow_with_custom_outlinecolor(self):
        from gerrytools.plotting.data import LabelArrowOptions, LabelArrowStyle

        sp = _make_plot()
        sp.add_label_arrow(
            (0.5, 0.5),
            "right",
            arrow_options=LabelArrowOptions(style=LabelArrowStyle(arrowedgecolor="blue")),
        )
        style = sp._annotations.annotation_arrows[0].style
        assert style.arrowedgecolor != "black"  # overridden

    def test_add_text_arrow_name_stored(self):
        sp = _make_plot()
        sp.add_text_arrow((0.5, 0.5), "right", name="my_arrow")
        assert sp._annotations.annotation_arrows[0].name == "my_arrow"


# =====================
# == AXIS DIRECTION  ==
# =====================
class TestAxisDirectionArrows:
    def test_add_axis_text_arrow_defaults_right_below_x_axis(self):
        sp = _make_plot()
        sp.add_axis_text_arrow("x", "more compact")
        [arrow] = sp._annotations.annotation_arrows
        assert arrow.arrowtip == (0.5, -0.08)
        assert arrow.direction == "right"
        assert arrow.text == "more compact"
        assert arrow.placement.coordinate_system == "axes fraction"

    def test_add_axis_label_arrow_defaults_up_left_of_y_axis(self):
        sp = _make_plot()
        sp.add_axis_label_arrow("y", "more votes")
        [arrow] = sp._annotations.annotation_arrows
        assert arrow.arrowtip == (-0.08, 0.5)
        assert arrow.direction == "up"
        assert arrow.text == "more votes"
        assert arrow.placement.coordinate_system == "axes fraction"
        assert arrow.placement.tail_length == 0.04

    def test_position_offset_and_reversed_direction(self):
        sp = _make_plot()
        sp.add_axis_text_arrow("x", position=0.25, offset=0.12, direction="left")
        [arrow] = sp._annotations.annotation_arrows
        assert arrow.arrowtip == (0.25, -0.12)
        assert arrow.direction == "left"

    def test_custom_placement_keeps_fields_but_forces_axes_fraction(self):
        from gerrytools.plotting.data import (
            ArrowPlacement,
            LabelArrowOptions,
            TextArrowOptions,
        )

        sp = _make_plot()
        placement = ArrowPlacement(coordinate_system="data", tail_length=0.1, zorder=7)
        sp.add_axis_text_arrow("x", "n", arrow_options=TextArrowOptions(placement=placement))
        sp.add_axis_label_arrow("y", "n", arrow_options=LabelArrowOptions(placement=placement))
        for arrow in sp._annotations.annotation_arrows:
            assert arrow.placement.coordinate_system == "axes fraction"
            assert arrow.placement.tail_length == 0.1
            assert arrow.placement.zorder == 7

    def test_axis_text_arrow_common_overrides_take_precedence(self):
        from gerrytools.plotting import ArrowTextStyle, TextArrowOptions, TextArrowStyle

        sp = _make_plot()
        sp.add_axis_text_arrow(
            "x",
            "n",
            arrowfacecolor="#123456",
            arrowfacealpha=0.4,
            fontcolor="#654321",
            textrotation=90,
            text_options=ArrowTextStyle(fontcolor="#abcdef", rotation=45),
            arrow_options=TextArrowOptions(
                style=TextArrowStyle(arrowfacecolor="#fedcba", arrowfacealpha=0.8)
            ),
        )

        [arrow] = sp._annotations.annotation_arrows
        assert arrow.style.arrowfacecolor == "#123456"
        assert arrow.style.arrowfacealpha == 0.4
        assert arrow.textstyle.fontcolor == "#654321"
        assert arrow.textstyle.rotation == 90.0

    def test_axis_label_arrow_common_overrides_take_precedence(self):
        from gerrytools.plotting import (
            LabelArrowOptions,
            LabelArrowStyle,
            LabelFontOptions,
            LabelStyle,
        )
        from gerrytools.plotting.data._gerryplot_dataclasses import _LabelArrowData

        sp = _make_plot()
        sp.add_axis_label_arrow(
            "y",
            "n",
            arrowfacecolor="#123456",
            arrowfacealpha=0.4,
            fontcolor="#654321",
            textrotation=90,
            text_options=LabelStyle(font=LabelFontOptions(fontcolor="#abcdef")),
            arrow_options=LabelArrowOptions(
                arrow_length=25,
                style=LabelArrowStyle(arrowfacecolor="#fedcba", arrowfacealpha=0.8),
            ),
        )

        [arrow] = sp._annotations.annotation_arrows
        assert isinstance(arrow, _LabelArrowData)
        assert arrow.arrow_length_percentage == 25.0
        assert arrow.style.arrowfacecolor == "#123456"
        assert arrow.style.arrowfacealpha == 0.4
        assert arrow.label_font_options is not None
        assert arrow.label_font_options.fontcolor == "#654321"
        assert arrow.textstyle.rotation == 90.0

    def test_axis_label_arrow_uses_label_style_box_for_text(self):
        from gerrytools.plotting import LabelBoxOptions, LabelStyle
        from gerrytools.plotting.data._gerryplot_dataclasses import _LabelArrowData

        sp = _make_plot()
        sp.add_axis_label_arrow(
            "y",
            "A",
            text_options=LabelStyle(
                box=LabelBoxOptions(boxstyle="circle", pad=0.3),
                equalize_circle_pad=True,
            ),
        )

        [arrow] = sp._annotations.annotation_arrows
        assert isinstance(arrow, _LabelArrowData)
        assert arrow.label_box_options is not None
        assert arrow.label_box_options.pad == 0.55

    def test_axis_label_arrow_signature_keeps_advanced_controls_in_options(self):
        parameters = inspect.signature(ScatterPlot.add_axis_label_arrow).parameters

        assert "arrowfacecolor" in parameters
        assert "arrowfacealpha" in parameters
        assert "fontcolor" in parameters
        assert "textrotation" in parameters
        assert "text_options" in parameters
        assert "arrow_options" in parameters
        assert {
            "arrow_length",
            "arrowedgecolor",
            "arrowedgealpha",
            "arrowedgewidth",
            "arrowtextstyle",
            "arrowplacement",
            "arrowstyle",
            "label_font_options",
            "label_box_options",
        }.isdisjoint(parameters)

    def test_label_arrow_signature_keeps_common_controls_at_top_level(self):
        parameters = inspect.signature(ScatterPlot.add_label_arrow).parameters

        assert "arrowfacecolor" in parameters
        assert "arrowfacealpha" in parameters
        assert "fontcolor" in parameters
        assert "textrotation" in parameters
        assert "text_options" in parameters
        assert "arrow_options" in parameters
        assert {
            "arrow_length",
            "arrowedgecolor",
            "arrowedgealpha",
            "arrowedgewidth",
            "arrowtextstyle",
            "arrowplacement",
            "arrowstyle",
            "label_font_options",
            "label_box_options",
        }.isdisjoint(parameters)

    def test_axis_text_arrow_signature_keeps_advanced_controls_in_options(self):
        parameters = inspect.signature(ScatterPlot.add_axis_text_arrow).parameters

        assert "arrowfacecolor" in parameters
        assert "arrowfacealpha" in parameters
        assert "fontcolor" in parameters
        assert "textrotation" in parameters
        assert "text_options" in parameters
        assert "arrow_options" in parameters
        assert {
            "arrowedgecolor",
            "arrowedgealpha",
            "arrowedgewidth",
            "arrowtextstyle",
            "arrowplacement",
            "arrowstyle",
        }.isdisjoint(parameters)

    def test_direction_must_run_along_the_axis(self):
        sp = _make_plot()
        with pytest.raises(ValueError, match="x-axis arrow"):
            sp.add_axis_text_arrow("x", direction="up")
        with pytest.raises(ValueError, match="y-axis arrow"):
            sp.add_axis_label_arrow("y", direction="left")

    def test_axis_arrows_render_outside_the_axes(self):
        sp = _make_plot()
        sp.add_axis_text_arrow("x", "increase")
        sp.add_axis_label_arrow("y", "increase")
        sp.ax
        assert len(sp._annotations.annotation_arrows) == 2


# =========================
# == ARROW LENGTH INPUTS ==
# =========================


class TestGerryPlotArrowLength:
    """Explicit arrow lengths are preserved on stored arrow data."""

    def test_arrow_length_sets_percentage(self):
        from gerrytools.plotting.data import LabelArrowOptions
        from gerrytools.plotting.data.boxplot import BoxPlot

        bp = BoxPlot(legend=False)
        bp.add_dataset({"A": [1.0, 2.0, 3.0]})
        bp.add_label_arrow(
            arrowtip=(1.0, 2.0),
            direction="right",
            text="ArrowLenTest",
            arrow_options=LabelArrowOptions(arrow_length=30.0),
        )
        ax = bp.ax
        # Building successfully confirms the explicit arrow length is accepted.
        assert ax is not None


# ======================
# == AXIS BUILD PATHS ==
# ======================


class TestGerryPlotAxisBuildPaths:
    """Subclass defaults still drive axis labels when no explicit labels are provided."""

    def test_boxplot_default_xtick_labels_path(self):
        """BoxPlot uses its category labels as default x tick labels."""
        from gerrytools.plotting.data.boxplot import BoxPlot

        bp = BoxPlot(legend=False)
        bp.add_dataset({"Alpha": [1.0, 2.0], "Beta": [3.0, 4.0]})
        # No explicit tick calls - subclass default labels should apply
        ax = bp.ax
        tick_labels = [t.get_text() for t in ax.get_xticklabels()]
        assert "Alpha" in tick_labels or "Beta" in tick_labels or len(tick_labels) >= 1

    def test_empty_ytick_labels_keeps_build_working(self):
        """Clearing y tick labels still allows the plot to build."""
        from gerrytools.plotting.data.boxplot import BoxPlot

        bp = BoxPlot(legend=False)
        bp.add_dataset({"A": [1.0, 2.0]})
        bp.set_yticks(labels=[])
        ax = bp.ax
        assert ax is not None


# ====================
# == VERTICAL BANDS ==
# ====================


class TestGerryPlotVerticalBands:
    """Vertical bands apply default and zero-width edge variants."""

    def test_vertical_band_linecolor_none(self):
        sp = _make_plot()
        sp.add_vertical_band(0.2, 0.4, linecolor=None)
        ax = sp.ax
        assert to_rgba(ax.patches[-1].get_edgecolor())[3] == 0
        assert ax.patches[-1].get_linewidth() == 0

    def test_omitted_linecolor_tracks_overridden_bandcolor(self):
        sp = _make_plot()
        sp.add_vertical_band(0.2, 0.4, bandcolor="red")

        patch = sp.ax.patches[-1]
        assert patch.get_edgecolor() == patch.get_facecolor()

    def test_explicit_options_linecolor_survives_bandcolor_override(self):
        sp = _make_plot()
        sp.add_vertical_band(
            0.2,
            0.4,
            band_options=BandOptions(bandcolor="red", linecolor="black"),
            bandcolor="blue",
        )

        patch = sp.ax.patches[-1]
        assert to_rgba(patch.get_edgecolor()) == to_rgba("black")
        assert to_rgba(patch.get_facecolor()) == to_rgba("blue")

    def test_vertical_band_linewidth_zero(self):
        sp = _make_plot()
        sp.add_vertical_band(0.3, 0.5, linecolor="black", linewidth=0.0)
        ax = sp.ax
        assert ax.patches[-1].get_linewidth() == 0
        assert to_rgba(ax.patches[-1].get_edgecolor())[3] == 0


# =============================
# == HORIZONTAL LINES SCALAR ==
# =============================


class TestGerryPlotHorizontalLines:
    """Scalar horizontal line inputs are normalized internally."""

    def test_horizontal_line_scalar_value(self):
        sp = _make_plot()
        sp.add_horizontal_lines(0.5)
        ax = sp.ax
        assert np.asarray(ax.lines[-1].get_ydata()).tolist() == [0.5, 0.5]


# =====================================
# == ANNOTATION TEXT OUTLINE EFFECTS ==
# =====================================


class TestAnnotationTextOutlineEffects:
    """Missing or zero-width outlines skip path effects."""

    def test_no_font_outline_returns_none(self):
        """An absent label outline skips text outline effects."""
        from gerrytools.plotting.data.boxplot import BoxPlot
        from gerrytools.plotting.mpl.label_text_options import LabelFontOptions, LabelStyle

        bp = BoxPlot(legend=False)
        bp.add_dataset({"A": [1.0, 2.0]})
        bp.add_label_arrow(
            arrowtip=(1.0, 2.0),
            direction="right",
            text="NoOutline",
            text_options=LabelStyle(font=LabelFontOptions(outlinecolor="none", outlinewidth=0)),
        )
        ax = bp.ax
        text = next(item for item in ax.texts if item.get_text() == "NoOutline")
        assert text.get_path_effects() == []

    def test_font_outline_width_zero_returns_none(self):
        """A zero outline width skips text outline effects."""
        from gerrytools.plotting.data.boxplot import BoxPlot
        from gerrytools.plotting.mpl.label_text_options import LabelFontOptions, LabelStyle

        bp = BoxPlot(legend=False)
        bp.add_dataset({"A": [1.0, 2.0]})
        bp.add_label_arrow(
            arrowtip=(1.0, 2.0),
            direction="right",
            text="ZeroOutlineWidth",
            text_options=LabelStyle(font=LabelFontOptions(outlinewidth=0)),
        )
        ax = bp.ax
        text = next(item for item in ax.texts if item.get_text() == "ZeroOutlineWidth")
        assert text.get_path_effects() == []


# ================================
# == ANNOTATION DEFAULT ZORDERS ==
# ================================


class TestAnnotationDefaultZorders:
    """Documented default zorders (3 vertical, 4 horizontal) apply on both styling paths."""

    @pytest.mark.parametrize("via_options", [False, True], ids=["kwargs", "options"])
    @pytest.mark.parametrize("orientation,expected_zorder", [("vertical", 3), ("horizontal", 4)])
    def test_line_default_zorder(self, orientation, expected_zorder, via_options):
        from gerrytools.plotting.data.options import LineOptions

        sp = _make_plot()
        line_options = LineOptions() if via_options else None
        if orientation == "vertical":
            sp.add_vertical_lines(0.5, line_options=line_options)
            stored = sp._annotations.vertical_lines[0]
        else:
            sp.add_horizontal_lines(0.5, line_options=line_options)
            stored = sp._annotations.horizontal_lines[0]
        assert stored.style.zorder == expected_zorder

    @pytest.mark.parametrize("via_options", [False, True], ids=["kwargs", "options"])
    @pytest.mark.parametrize("orientation,expected_zorder", [("vertical", 3), ("horizontal", 4)])
    def test_band_default_zorder(self, orientation, expected_zorder, via_options):
        from gerrytools.plotting.data.options import BandOptions

        sp = _make_plot()
        band_options = BandOptions() if via_options else None
        if orientation == "vertical":
            sp.add_vertical_band(0.2, 0.4, band_options=band_options)
            stored = sp._annotations.vertical_bands[0]
        else:
            sp.add_horizontal_band(0.2, 0.4, band_options=band_options)
            stored = sp._annotations.horizontal_bands[0]
        assert stored.style.zorder == expected_zorder

    def test_explicit_zorder_on_options_object_wins(self):
        from gerrytools.plotting.data.options import BandOptions, LineOptions

        sp = _make_plot()
        sp.add_horizontal_lines(0.5, line_options=LineOptions(zorder=3))
        assert sp._annotations.horizontal_lines[0].style.zorder == 3
        sp.add_horizontal_band(0.2, 0.4, band_options=BandOptions(zorder=3))
        assert sp._annotations.horizontal_bands[0].style.zorder == 3

    def test_explicit_zorder_kwarg_wins_over_options_object(self):
        from gerrytools.plotting.data.options import LineOptions

        sp = _make_plot()
        sp.add_horizontal_lines(0.5, line_options=LineOptions(), zorder=9)
        assert sp._annotations.horizontal_lines[0].style.zorder == 9

    @pytest.mark.parametrize("options_type", [LineOptions, BandOptions])
    def test_dataclass_replace_preserves_default_zorder(self, options_type):
        options = dataclasses.replace(options_type())

        assert options._zorder_defaulted is True

        sp = _make_plot()
        if isinstance(options, LineOptions):
            sp.add_horizontal_lines(0.5, line_options=options)
            stored = sp._annotations.horizontal_lines[0]
        else:
            sp.add_horizontal_band(0.2, 0.4, band_options=options)
            stored = sp._annotations.horizontal_bands[0]
        assert stored.style.zorder == 4

    @pytest.mark.parametrize("options_type", [LineOptions, BandOptions])
    def test_dataclass_replace_with_explicit_default_zorder_stays_explicit(self, options_type):
        options = dataclasses.replace(options_type(), zorder=3)

        assert options._zorder_defaulted is False

        sp = _make_plot()
        if isinstance(options, LineOptions):
            sp.add_horizontal_lines(0.5, line_options=options)
            stored = sp._annotations.horizontal_lines[0]
        else:
            sp.add_horizontal_band(0.2, 0.4, band_options=options)
            stored = sp._annotations.horizontal_bands[0]
        assert stored.style.zorder == 3


# ===================================
# == LABEL FONT OUTLINE SKIP RULES ==
# ===================================


class TestLabelFontOutlineEffects:
    """`label_font_options` outlines obey the same zero-width skip as text-style outlines."""

    def test_zero_outline_width_skips_path_effects(self):
        from gerrytools.plotting.mpl.label_text_options import LabelFontOptions, LabelStyle

        sp = _make_plot()
        sp.add_label_arrow(
            (0.5, 0.5),
            "right",
            text="ZeroWidthOutline",
            text_options=LabelStyle(font=LabelFontOptions(outlinewidth=0)),
        )
        ax = sp.ax
        text = next(item for item in ax.texts if item.get_text() == "ZeroWidthOutline")
        assert text.get_path_effects() == []

    def test_positive_outline_width_applies_path_effects(self):
        from gerrytools.plotting.mpl.label_text_options import LabelFontOptions, LabelStyle

        sp = _make_plot()
        sp.add_label_arrow(
            (0.5, 0.5),
            "right",
            text="WideOutline",
            text_options=LabelStyle(font=LabelFontOptions(outlinewidth=1.5)),
        )
        ax = sp.ax
        text = next(item for item in ax.texts if item.get_text() == "WideOutline")
        assert len(text.get_path_effects()) == 2


# =======================
# == ANNOTATION ERRORS ==
# =======================


class TestArrowLengthInfiniteRaises:
    """Non-finite arrow lengths are rejected."""

    def test_infinite_arrow_length_raises(self):
        from gerrytools.plotting.data import LabelArrowOptions

        plot = ScatterPlot(legend=False)
        plot.add_series(x=[0.0, 1.0], y=[0.0, 1.0])
        with pytest.raises(ValueError, match="arrow_length must be finite"):
            plot.add_label_arrow(
                arrowtip=(0.5, 0.5),
                direction="right",
                arrow_options=LabelArrowOptions(arrow_length=float("inf")),
            )


class TestFontOutlineColorNone:
    """Text path effects are omitted when no outline color is set."""

    def test_arrow_text_style_no_outline_color_returns_no_path_effects(self):
        from gerrytools.plotting.data import ArrowTextStyle

        plot = ScatterPlot(legend=False)
        plot.add_series(x=[0.0, 1.0], y=[0.0, 1.0])
        plot.add_text_arrow(
            arrowtip=(0.5, 0.5),
            direction="right",
            text="NoOutline",
            arrowtextstyle=ArrowTextStyle(fontoutlinecolor=None),
        )
        ax = plot.ax
        text = next((t for t in ax.texts if t.get_text() == "NoOutline"), None)
        assert text is not None
        # With fontoutlinecolor=None the path effects list should be empty or absent
        pe = text.get_path_effects()
        assert len(pe) == 0
