import math
from typing import Literal

import matplotlib

matplotlib.use("Agg")

import matplotlib.colors as mcolors
import numpy as np
import pytest
from matplotlib.text import Annotation, Text

from gerrytools.colors import resolve_color_and_alpha
from gerrytools.plotting.data import (
    ArrowPlacement,
    ArrowTextStyle,
    LabelArrowOptions,
    LabelArrowStyle,
    ScatterPlot,
    TextArrowStyle,
)
from gerrytools.plotting.mpl.label_text_options import LabelBoxOptions, LabelFontOptions, LabelStyle


def _simple_scatter() -> ScatterPlot:
    plot = ScatterPlot(legend=False)
    plot.add_series(x=[0.0, 1.0], y=[0.0, 1.0])
    return plot


def _bbox_tip_display(*, arrow_text_artist: Text, direction: str) -> tuple[float, float]:
    bbox_patch = arrow_text_artist.get_bbox_patch()
    assert bbox_patch is not None
    vertices_display = bbox_patch.get_transform().transform(bbox_patch.get_path().vertices)
    points = [(float(vertex[0]), float(vertex[1])) for vertex in vertices_display]
    if direction == "right":
        return max(points, key=lambda point: point[0])
    if direction == "left":
        return min(points, key=lambda point: point[0])
    if direction == "up":
        return max(points, key=lambda point: point[1])
    return min(points, key=lambda point: point[1])


def _bbox_width_display(*, arrow_text_artist: Text) -> float:
    bbox_patch = arrow_text_artist.get_bbox_patch()
    assert bbox_patch is not None
    vertices_display = bbox_patch.get_transform().transform(bbox_patch.get_path().vertices)
    xs = [float(vertex[0]) for vertex in vertices_display]
    return max(xs) - min(xs)


def _label_arrow_annotation(ax):
    return next(
        (t for t in ax.texts if t.get_text() == "" and getattr(t, "arrow_patch", None) is not None),
        None,
    )


def _visible_text_by_content(ax, content: str) -> Text | None:
    matches = [t for t in ax.texts if t.get_text() == content]
    if len(matches) == 0:
        return None
    return max(matches, key=lambda text: float(mcolors.to_rgba(text.get_color())[3]))


def test_add_text_arrow_default_blank_text_draws_bbox():
    plot = _simple_scatter()
    plot.add_text_arrow(
        arrowtip=(0.8, 0.2),
        direction="right",
    )

    ax = plot.ax
    arrow_text = next((t for t in ax.texts if t.get_text() == "   "), None)
    assert arrow_text is not None
    assert arrow_text.get_bbox_patch() is not None


def test_add_text_arrow_tip_aligns_to_requested_tip():
    plot = _simple_scatter()
    plot.add_text_arrow(
        arrowtip=(0.8, 0.2),
        direction="right",
        text="POV Advantage",
    )

    ax = plot.ax
    plot.fig.canvas.draw()

    arrow_text = next((t for t in ax.texts if t.get_text() == "POV Advantage"), None)
    assert arrow_text is not None

    expected_tip_display = ax.transData.transform((0.8, 0.2))
    actual_tip_x, actual_tip_y = _bbox_tip_display(arrow_text_artist=arrow_text, direction="right")

    assert abs(actual_tip_x - float(expected_tip_display[0])) <= 1.5
    assert abs(actual_tip_y - float(expected_tip_display[1])) <= 1.5


def test_text_arrow_alignment_batches_canvas_draws(monkeypatch: pytest.MonkeyPatch):
    plot = _simple_scatter()
    directions: tuple[Literal["right", "left", "up", "down"], ...] = (
        "right",
        "left",
        "up",
        "down",
    )
    for index, direction in enumerate(directions):
        plot.add_text_arrow(
            arrowtip=(0.2 + (0.2 * index), 0.5),
            direction=direction,
            text=direction,
        )

    draw = plot.fig.canvas.draw
    draw_count = 0

    def counted_draw():
        nonlocal draw_count
        draw_count += 1
        return draw()

    monkeypatch.setattr(plot.fig.canvas, "draw", counted_draw)

    plot.ax

    assert draw_count == 2


def test_add_text_arrow_default_down_rotation_is_horizontal():
    plot = _simple_scatter()
    plot.add_text_arrow(
        arrowtip=(0.7, 0.3),
        direction="down",
        text="Readable",
    )

    ax = plot.ax
    arrow_text = _visible_text_by_content(ax, "Readable")
    assert arrow_text is not None
    assert abs(float(arrow_text.get_rotation()) - 0.0) <= 1e-9


@pytest.mark.parametrize("direction", ["up", "down"])
def test_add_text_arrow_vertical_direction_width_scales_with_text_length(
    direction: Literal["up", "down"],
):
    plot = _simple_scatter()
    short_text = "A"
    long_text = "This is intentionally much longer than A"
    plot.add_text_arrow(
        arrowtip=(0.3, 0.5),
        direction=direction,
        text=short_text,
    )
    plot.add_text_arrow(
        arrowtip=(0.7, 0.5),
        direction=direction,
        text=long_text,
    )

    ax = plot.ax
    plot.fig.canvas.draw()

    short_artist = _visible_text_by_content(ax, short_text)
    long_artist = _visible_text_by_content(ax, long_text)
    assert short_artist is not None
    assert long_artist is not None

    short_width = _bbox_width_display(arrow_text_artist=short_artist)
    long_width = _bbox_width_display(arrow_text_artist=long_artist)

    assert long_width > short_width * 1.5


def test_add_text_arrow_textrotation_overrides_default_rotation():
    plot = _simple_scatter()
    plot.add_text_arrow(
        arrowtip=(0.7, 0.3),
        direction="up",
        text="OverrideRotation",
        textrotation=12.5,
    )

    ax = plot.ax
    arrow_text = _visible_text_by_content(ax, "OverrideRotation")
    assert arrow_text is not None
    assert abs(float(arrow_text.get_rotation()) - 12.5) <= 1e-9


def test_add_annotation_arrow_textrotation_for_text_arrow():
    plot = _simple_scatter()
    plot.add_text_arrow(
        arrowtip=(0.7, 0.3),
        direction="up",
        text="WrapperRotation",
        textrotation=33.0,
    )

    ax = plot.ax
    arrow_text = _visible_text_by_content(ax, "WrapperRotation")
    assert arrow_text is not None
    assert abs(float(arrow_text.get_rotation()) - 33.0) <= 1e-9


def test_add_label_arrow_draws_arrow_and_label_box():
    plot = _simple_scatter()
    plot.add_label_arrow(
        arrowtip=(0.8, 0.2),
        direction="right",
        text="Toward POV",
        text_options=LabelStyle(
            font=LabelFontOptions(
                fontcolor="black",
                fontsize=8,
                outlinecolor="white",
                outlinewidth=0.5,
            ),
            box=LabelBoxOptions(
                enabled=True,
                boxstyle="round4",
                facecolor="white",
                edgecolor="black",
                edgewidth=0.6,
            ),
        ),
        arrow_options=LabelArrowOptions(placement=ArrowPlacement(tail_length=0.25)),
    )

    ax = plot.ax
    arrow_annotation = _label_arrow_annotation(ax)
    assert arrow_annotation is not None

    label_text = next((t for t in ax.texts if t.get_text() == "Toward POV"), None)
    assert label_text is not None
    assert label_text.get_bbox_patch() is not None


def test_label_arrow_outline_preserves_embedded_color_alpha():
    plot = _simple_scatter()
    plot.add_label_arrow(
        arrowtip=(0.8, 0.2),
        direction="right",
        text="Translucent outline",
        text_options=LabelStyle(
            font=LabelFontOptions(
                outlinecolor="#ff000040",
                outlinewidth=1.0,
            )
        ),
    )

    text = next(item for item in plot.ax.texts if item.get_text() == "Translucent outline")
    stroke = text.get_path_effects()[0]
    foreground = getattr(stroke, "_gc")["foreground"]
    assert foreground == pytest.approx((1.0, 0.0, 0.0, 0.25), abs=1 / 255)


def test_add_label_arrow_applies_label_padding_away_from_tail():
    plot = _simple_scatter()
    placement = ArrowPlacement(
        tail_length=0.2,
        label_padding=0.05,
    )
    plot.add_label_arrow(
        arrowtip=(0.8, 0.2),
        direction="right",
        text="Padded",
        arrow_options=LabelArrowOptions(placement=placement),
    )

    ax = plot.ax
    label_text = next((t for t in ax.texts if t.get_text() == "Padded"), None)
    assert label_text is not None

    tail_x = 0.8 - placement.tail_length
    label_x, _ = label_text.get_position()
    assert abs(float(label_x) - (tail_x - placement.label_padding)) <= 1e-9


def test_add_label_arrow_up_text_hangs_below_the_tail():
    plot = _simple_scatter()
    plot.add_label_arrow(arrowtip=(0.5, 0.8), direction="up", text="Below")
    ax = plot.ax
    label_text = next((t for t in ax.texts if t.get_text() == "Below"), None)
    assert label_text is not None
    assert label_text.get_verticalalignment() == "top"


def test_add_label_arrow_down_text_sits_above_the_tail():
    plot = _simple_scatter()
    plot.add_label_arrow(arrowtip=(0.5, 0.2), direction="down", text="Above")
    ax = plot.ax
    label_text = next((t for t in ax.texts if t.get_text() == "Above"), None)
    assert label_text is not None
    assert label_text.get_verticalalignment() == "bottom"


def test_add_label_arrow_arrow_length_overrides_tail_length():
    plot = _simple_scatter()
    plot.add_label_arrow(
        arrowtip=(0.8, 0.2),
        direction="right",
        text="LengthOverride",
        arrow_options=LabelArrowOptions(
            arrow_length=30.0,
            placement=ArrowPlacement(tail_length=0.1, label_padding=0.0),
        ),
    )

    ax = plot.ax
    arrow_annotation = _label_arrow_annotation(ax)
    assert arrow_annotation is not None

    tip_display = ax.transData.transform((0.8, 0.2))
    tail_display = ax.transData.transform(arrow_annotation.get_position())
    axes_width = float(ax.get_window_extent().width)
    distance_pixels = math.hypot(
        float(tip_display[0] - tail_display[0]),
        float(tip_display[1] - tail_display[1]),
    )
    assert abs(distance_pixels - (0.3 * axes_width)) <= 1.5


def test_add_label_arrow_arrow_length_100_is_full_axes_height_for_vertical_arrow():
    plot = _simple_scatter()
    plot.add_label_arrow(
        arrowtip=(0.6, 0.8),
        direction="up",
        arrow_options=LabelArrowOptions(
            arrow_length=100.0,
            placement=ArrowPlacement(
                label_padding=0.0,
            ),
        ),
    )

    ax = plot.ax
    arrow_annotation = _label_arrow_annotation(ax)
    assert arrow_annotation is not None

    tip_display = ax.transData.transform((0.6, 0.8))
    tail_display = ax.transData.transform(arrow_annotation.get_position())
    axes_height = float(ax.get_window_extent().height)
    distance_pixels = math.hypot(
        float(tip_display[0] - tail_display[0]),
        float(tip_display[1] - tail_display[1]),
    )
    assert abs(distance_pixels - axes_height) <= 2.0


def test_add_label_arrow_rejects_arrow_length_with_explicit_arrowtail():
    plot = _simple_scatter()
    with pytest.raises(
        ValueError, match="arrow_length cannot be set when placement.arrowtail is set"
    ):
        plot.add_label_arrow(
            arrowtip=(0.8, 0.2),
            direction="right",
            text="Invalid",
            arrow_options=LabelArrowOptions(
                arrow_length=0.3,
                placement=ArrowPlacement(arrowtail=(0.1, 0.2)),
            ),
        )


def test_add_label_arrow_rejects_arrow_length_outside_valid_range():
    plot = _simple_scatter()
    with pytest.raises(ValueError, match="arrow_length must be in \\[0, 100\\]"):
        plot.add_label_arrow(
            arrowtip=(0.8, 0.2),
            direction="right",
            text="Invalid",
            arrow_options=LabelArrowOptions(arrow_length=120.0),
        )


def test_text_arrow_supports_text_outline_from_textstyle():
    plot = _simple_scatter()
    plot.add_text_arrow(
        arrowtip=(0.7, 0.3),
        direction="left",
        text="Outline",
        arrowtextstyle=ArrowTextStyle(
            fontoutlinecolor="black",
            fontoutlinewidth=1.25,
        ),
    )

    ax = plot.ax
    arrow_text = next((t for t in ax.texts if t.get_text() == "Outline"), None)
    assert arrow_text is not None
    assert len(arrow_text.get_path_effects()) >= 2


def test_annotation_arrow_direct_color_overrides_style_colors():
    plot = _simple_scatter()
    plot.add_text_arrow(
        arrowtip=(0.75, 0.25),
        direction="left",
        text="Override",
        arrowstyle=TextArrowStyle(
            arrowfacecolor="alizarin",
            arrowedgecolor="black",
        ),
        arrowfacecolor="denim",
    )

    ax = plot.ax
    arrow_text = next((t for t in ax.texts if t.get_text() == "Override"), None)
    assert arrow_text is not None
    bbox_patch = arrow_text.get_bbox_patch()
    assert bbox_patch is not None
    face_hex = mcolors.to_hex(bbox_patch.get_facecolor(), keep_alpha=True)
    expected_hex = mcolors.to_hex(
        mcolors.to_rgba(resolve_color_and_alpha("denim")[0], alpha=1.0),
        keep_alpha=True,
    )
    assert face_hex.lower() == expected_hex.lower()


def test_clear_annotation_arrows_removes_all_arrows():
    plot = _simple_scatter()
    plot.add_text_arrow(arrowtip=(0.8, 0.2), direction="right", text="A")
    plot.add_label_arrow(
        arrowtip=(0.8, 0.2),
        direction="right",
        text="B",
    )

    plot.clear_annotation_arrows()
    ax = plot.ax
    assert all(text.get_text() not in {"A", "B"} for text in ax.texts)


class TestLabelArrowStyleEdgeCases:
    def test_infinite_arrowedgewidth_raises_valueerror(self):
        from gerrytools.plotting.data._gerryplot_dataclasses import LabelArrowStyle

        with pytest.raises(ValueError, match="arrowedgewidth must be finite"):
            LabelArrowStyle(arrowedgewidth=float("inf"))


class TestArrowDataEdgeCases:
    @pytest.mark.parametrize("method", ["add_text_arrow", "add_label_arrow"])
    @pytest.mark.parametrize(
        "direction",
        ["sideways", [], np.array(["right"])],
        ids=["unknown", "unhashable", "non-string-array"],
    )
    def test_invalid_direction_raises_at_call_time(self, method: str, direction: object):
        plot = _simple_scatter()

        with pytest.raises(ValueError, match="direction must be one of"):
            getattr(plot, method)(arrowtip=(0.5, 0.5), direction=direction)

        assert plot.ax is not None

    def test_infinite_arrow_length_percentage_raises_valueerror(self):
        from gerrytools.plotting.data._gerryplot_dataclasses import _LabelArrowData

        with pytest.raises(ValueError, match="arrow_length_percentage must be finite"):
            _LabelArrowData(
                arrowtip=(0.5, 0.5),
                direction="right",
                arrow_length_percentage=float("inf"),
            )


# ==================================
# == TEXT ARROW RENDERER BEHAVIOR ==
# ==================================
class TestTextArrowRendererBranches:
    def test_custom_boxstyle_is_applied_to_text_bbox(self):
        """A caller-supplied boxstyle replaces the default arrow-shaped bbox."""
        from matplotlib.patches import BoxStyle

        plot = _simple_scatter()
        plot.add_text_arrow(
            arrowtip=(0.5, 0.5),
            direction="right",
            text="BoxStyleTest",
            arrowstyle=TextArrowStyle(boxstyle="round,pad=0.3"),
        )
        ax = plot.ax
        arrow_text = next((t for t in ax.texts if t.get_text() == "BoxStyleTest"), None)
        assert arrow_text is not None
        bbox_patch = arrow_text.get_bbox_patch()
        assert bbox_patch is not None
        assert isinstance(bbox_patch.get_boxstyle(), BoxStyle.Round)

    def test_arrowedgecolor_none_renders_transparent_bbox_edge(self):
        """arrowedgecolor='none' renders the text-arrow bbox with a fully transparent edge."""
        plot = _simple_scatter()
        plot.add_text_arrow(
            arrowtip=(0.5, 0.5),
            direction="right",
            text="NoOutlineText",
            arrowstyle=TextArrowStyle(arrowedgecolor="none"),
        )
        ax = plot.ax
        arrow_text = next((t for t in ax.texts if t.get_text() == "NoOutlineText"), None)
        assert arrow_text is not None
        bbox_patch = arrow_text.get_bbox_patch()
        assert bbox_patch is not None
        assert mcolors.to_rgba(bbox_patch.get_edgecolor())[3] == 0.0

    def test_empty_text_is_padded_with_spaces(self):
        """text='' still renders an arrow body: the text is replaced with three spaces."""
        plot = _simple_scatter()
        plot.add_text_arrow(
            arrowtip=(0.5, 0.5),
            direction="right",
            text="",
        )
        ax = plot.ax
        arrow_text = next((t for t in ax.texts if t.get_text() == "   "), None)
        assert arrow_text is not None
        assert arrow_text.get_bbox_patch() is not None


# ================================
# == LABEL ARROW RENDERER PATHS ==
# ================================
class TestLabelArrowRendererBranches:
    def test_explicit_arrowtail_places_arrow_tail_directly(self):
        """An explicit arrow tail becomes the annotation's tail point verbatim."""
        plot = _simple_scatter()
        plot.add_label_arrow(
            arrowtip=(0.8, 0.5),
            direction="right",
            text="TailTest",
            arrow_options=LabelArrowOptions(placement=ArrowPlacement(arrowtail=(0.3, 0.5))),
        )
        ax = plot.ax
        label_text = next((t for t in ax.texts if t.get_text() == "TailTest"), None)
        assert label_text is not None
        annotation = _label_arrow_annotation(ax)
        assert isinstance(annotation, Annotation)
        assert tuple(annotation.xyann) == (0.3, 0.5)
        assert tuple(annotation.xy) == (0.8, 0.5)

    def test_arrowedgecolor_none_disables_label_arrow_outline(self):
        """arrowedgecolor='none' renders the arrow patch with a fully transparent edge."""
        plot = _simple_scatter()
        plot.add_label_arrow(
            arrowtip=(0.8, 0.5),
            direction="right",
            text="NoneOutlineLabel",
            arrow_options=LabelArrowOptions(style=LabelArrowStyle(arrowedgecolor="none")),
        )
        ax = plot.ax
        label_text = next((t for t in ax.texts if t.get_text() == "NoneOutlineLabel"), None)
        assert label_text is not None
        annotation = _label_arrow_annotation(ax)
        assert isinstance(annotation, Annotation)
        assert annotation.arrow_patch is not None
        assert mcolors.to_rgba(annotation.arrow_patch.get_edgecolor())[3] == 0.0

    def test_explicit_label_position_is_used_directly(self):
        """An explicit label position is used directly."""
        plot = _simple_scatter()
        plot.add_label_arrow(
            arrowtip=(0.8, 0.5),
            direction="right",
            text="LabelPosTest",
            label_position=(0.2, 0.7),
        )
        ax = plot.ax
        label_text = next((t for t in ax.texts if t.get_text() == "LabelPosTest"), None)
        assert label_text is not None
        # The text should be placed at label_position (0.2, 0.7) + offset (0,0)
        lx, ly = label_text.get_position()
        assert abs(float(lx) - 0.2) <= 1e-9
        assert abs(float(ly) - 0.7) <= 1e-9


def test_top_level_none_removes_text_arrow_fill_and_edge():
    plot = _simple_scatter()
    plot.add_text_arrow(
        arrowtip=(0.8, 0.5),
        direction="right",
        text="Transparent arrow",
        arrowfacecolor=None,
        arrowedgecolor=None,
    )

    arrow_text = _visible_text_by_content(plot.ax, "Transparent arrow")
    assert arrow_text is not None
    patch = arrow_text.get_bbox_patch()
    assert patch is not None
    assert mcolors.to_rgba(patch.get_facecolor())[3] == 0.0
    assert mcolors.to_rgba(patch.get_edgecolor())[3] == 0.0


def test_top_level_none_removes_axis_label_arrow_fill_and_text_color():
    plot = _simple_scatter()
    plot.add_axis_label_arrow(
        "x",
        "Transparent label",
        arrowfacecolor=None,
        fontcolor=None,
    )

    label_text = next(text for text in plot.ax.texts if text.get_text() == "Transparent label")
    assert mcolors.to_rgba(label_text.get_color())[3] == 0.0
    annotation = _label_arrow_annotation(plot.ax)
    assert isinstance(annotation, Annotation)
    assert annotation.arrow_patch is not None
    assert mcolors.to_rgba(annotation.arrow_patch.get_facecolor())[3] == 0.0
