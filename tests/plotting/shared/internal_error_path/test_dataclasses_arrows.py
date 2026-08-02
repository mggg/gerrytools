import pytest

from gerrytools.plotting.data._gerryplot_dataclasses import (
    ArrowPlacement,
    ArrowTextStyle,
    LabelArrowOptions,
    LabelArrowStyle,
    TextArrowOptions,
    TextArrowStyle,
    _LabelArrowData,
    _TextArrowData,
)
from tests.plotting._typing_utils import as_any


# ==============
# == LINEDATA ==
# ==============
class TestArrowTextStyle:
    def test_default_construction(self):
        ts = ArrowTextStyle()
        assert ts.fontsize == 10.0
        assert ts.fontcolor == "#000000"
        assert ts.fontoutlinewidth == 0.5

    def test_negative_fontsize_raises_valueerror(self):
        with pytest.raises(ValueError, match="nonnegative"):
            ArrowTextStyle(fontsize=-1.0)

    def test_nan_fontsize_raises_valueerror(self):
        with pytest.raises(ValueError, match="finite"):
            ArrowTextStyle(fontsize=float("nan"))

    def test_infinite_fontsize_raises_valueerror(self):
        with pytest.raises(ValueError, match="finite"):
            ArrowTextStyle(fontsize=float("inf"))

    def test_negative_fontoutlinewidth_raises_valueerror(self):
        with pytest.raises(ValueError, match="nonnegative"):
            ArrowTextStyle(fontoutlinewidth=-0.1)

    def test_infinite_fontoutlinewidth_raises_valueerror(self):
        with pytest.raises(ValueError, match="finite"):
            ArrowTextStyle(fontoutlinewidth=float("inf"))

    def test_nan_rotation_raises_valueerror(self):
        with pytest.raises(ValueError, match="finite"):
            ArrowTextStyle(rotation=float("nan"))

    def test_none_rotation_is_valid(self):
        ts = ArrowTextStyle(rotation=None)
        assert ts.rotation is None

    def test_rotation_coerced_to_float(self):
        ts = ArrowTextStyle(rotation=90)
        assert isinstance(ts.rotation, float)
        assert ts.rotation == 90.0

    def test_fontoutlinecolor_none_resets_fontoutlinewidth_to_zero(self):
        ts = ArrowTextStyle(fontoutlinecolor=None, fontoutlinewidth=2.0)
        assert ts.fontoutlinewidth == 0.0

    def test_fontoutlinecolor_none_string_resets_fontoutlinewidth_to_zero(self):
        ts = ArrowTextStyle(fontoutlinecolor="none", fontoutlinewidth=2.0)
        assert ts.fontoutlinewidth == 0.0

    def test_zero_fontsize_is_valid(self):
        ts = ArrowTextStyle(fontsize=0.0)
        assert ts.fontsize == 0.0


# ====================
# == TEXTARROWSTYLE ==
# ====================


class TestTextArrowStyle:
    def test_default_construction(self):
        style = TextArrowStyle()
        assert style.arrowedgewidth == 1.0
        assert style.boxpad == 0.3

    def test_negative_arrowedgewidth_raises_valueerror(self):
        with pytest.raises(ValueError, match="nonnegative"):
            TextArrowStyle(arrowedgewidth=-1.0)

    def test_infinite_arrowedgewidth_raises_valueerror(self):
        with pytest.raises(ValueError, match="finite"):
            TextArrowStyle(arrowedgewidth=float("inf"))

    def test_negative_boxpad_raises_valueerror(self):
        with pytest.raises(ValueError, match="nonnegative"):
            TextArrowStyle(boxpad=-0.1)

    def test_infinite_boxpad_raises_valueerror(self):
        with pytest.raises(ValueError, match="finite"):
            TextArrowStyle(boxpad=float("inf"))

    def test_outlinecolor_none_string_resets_outlinewidth_to_zero(self):
        style = TextArrowStyle(arrowedgecolor="none", arrowedgewidth=2.0)
        assert style.arrowedgewidth == 0.0

    def test_zero_boxpad_is_valid(self):
        style = TextArrowStyle(boxpad=0.0)
        assert style.boxpad == 0.0


class TestTextArrowOptions:
    def test_defaults(self):
        options = TextArrowOptions()
        assert isinstance(options.placement, ArrowPlacement)
        assert isinstance(options.style, TextArrowStyle)


# =====================
# == LABELARROWSTYLE ==
# =====================


class TestLabelArrowStyle:
    def test_default_construction(self):
        style = LabelArrowStyle()
        assert style.arrowstyle == "-|>"
        assert style.arrowhead_scale == 12.0

    def test_arrowhead_scale_coerces_to_float_on_the_real_field(self):
        # Regression: the coerced value used to be written to a nonexistent "mutation_scale"
        # attribute, so the float coercion never landed on arrowhead_scale.
        style = LabelArrowStyle(arrowhead_scale=12)
        assert style.arrowhead_scale == 12.0
        assert isinstance(style.arrowhead_scale, float)
        assert not hasattr(style, "mutation_scale")

    def test_negative_arrowhead_scale_raises_valueerror(self):
        with pytest.raises(ValueError, match="nonnegative"):
            LabelArrowStyle(arrowhead_scale=-1.0)

    def test_infinite_arrowhead_scale_raises_valueerror(self):
        with pytest.raises(ValueError, match="finite"):
            LabelArrowStyle(arrowhead_scale=float("inf"))

    def test_negative_shrink_a_raises_valueerror(self):
        with pytest.raises(ValueError, match="nonnegative"):
            LabelArrowStyle(shrink_a=-1.0)

    def test_negative_shrink_b_raises_valueerror(self):
        with pytest.raises(ValueError, match="nonnegative"):
            LabelArrowStyle(shrink_b=-1.0)

    def test_infinite_shrink_raises_valueerror(self):
        with pytest.raises(ValueError, match="finite"):
            LabelArrowStyle(shrink_a=float("inf"))

    def test_negative_arrowedgewidth_raises_valueerror(self):
        with pytest.raises(ValueError, match="nonnegative"):
            LabelArrowStyle(arrowedgewidth=-0.5)

    def test_outlinecolor_none_resets_outlinewidth_to_zero(self):
        style = LabelArrowStyle(arrowedgecolor="none", arrowedgewidth=3.0)
        assert style.arrowedgewidth == 0.0

    def test_zero_shrink_values_are_valid(self):
        style = LabelArrowStyle(shrink_a=0.0, shrink_b=0.0)
        assert style.shrink_a == 0.0
        assert style.shrink_b == 0.0


# ====================
# == ARROWPLACEMENT ==
# ====================


class TestArrowPlacement:
    def test_default_construction(self):
        ap = ArrowPlacement()
        assert ap.coordinate_system == "data"
        assert ap.text_offset == (0.0, 0.0)
        assert ap.tail_length == 0.08
        assert ap.zorder == 20
        assert ap.clip_on is False

    def test_negative_tail_length_raises_valueerror(self):
        with pytest.raises(ValueError, match="nonnegative"):
            ArrowPlacement(tail_length=-0.01)

    def test_infinite_tail_length_raises_valueerror(self):
        with pytest.raises(ValueError, match="finite"):
            ArrowPlacement(tail_length=float("inf"))

    def test_negative_label_padding_raises_valueerror(self):
        with pytest.raises(ValueError, match="nonnegative"):
            ArrowPlacement(label_padding=-0.001)

    def test_infinite_label_padding_raises_valueerror(self):
        with pytest.raises(ValueError, match="finite"):
            ArrowPlacement(label_padding=float("inf"))

    def test_non_finite_text_offset_raises_valueerror(self):
        with pytest.raises(ValueError, match="finite"):
            ArrowPlacement(text_offset=(float("nan"), 0.0))

    def test_non_finite_arrowtail_raises_valueerror(self):
        with pytest.raises(ValueError, match="finite"):
            ArrowPlacement(arrowtail=(0.0, float("inf")))

    def test_arrowtail_none_is_valid(self):
        ap = ArrowPlacement(arrowtail=None)
        assert ap.arrowtail is None

    def test_arrowtail_tuple_is_coerced_to_floats(self):
        ap = ArrowPlacement(arrowtail=(1, 2))
        assert ap.arrowtail == (1.0, 2.0)

    def test_text_offset_coerced_to_floats(self):
        ap = ArrowPlacement(text_offset=(1, 2))
        assert ap.text_offset == (1.0, 2.0)

    def test_zorder_coerced_to_int(self):
        ap = ArrowPlacement(zorder=as_any(10.8))
        assert isinstance(ap.zorder, int)


# =======================
# == LABELARROWOPTIONS ==
# =======================


class TestLabelArrowOptions:
    def test_defaults_match_axis_label_arrow_defaults(self):
        options = LabelArrowOptions()
        assert options.arrow_length is None
        assert options.placement.tail_length == 0.04
        assert isinstance(options.style, LabelArrowStyle)

    def test_arrow_length_is_coerced_to_float(self):
        options = LabelArrowOptions(arrow_length=25)
        assert options.arrow_length == 25.0

    @pytest.mark.parametrize("arrow_length", [-1, 101, float("nan"), float("inf")])
    def test_invalid_arrow_length_raises_valueerror(self, arrow_length):
        with pytest.raises(ValueError, match="arrow_length"):
            LabelArrowOptions(arrow_length=arrow_length)

    def test_arrow_length_rejects_explicit_tail(self):
        with pytest.raises(ValueError, match="cannot be set"):
            LabelArrowOptions(
                arrow_length=25,
                placement=ArrowPlacement(arrowtail=(0.1, 0.2)),
            )


# ===============
# == ARROWDATA ==
# ===============


class TestArrowData:
    """The text/label split makes the old cross-type misconfigurations unrepresentable."""

    def test_text_arrow_has_text_style(self):
        arrow = _TextArrowData(arrowtip=(0.5, 0.5), direction="right")
        assert isinstance(arrow.style, TextArrowStyle)

    def test_label_arrow_has_label_style(self):
        arrow = _LabelArrowData(arrowtip=(0.5, 0.5), direction="up")
        assert isinstance(arrow.style, LabelArrowStyle)

    def test_non_finite_arrowtip_raises_valueerror(self):
        with pytest.raises(ValueError, match="finite"):
            _TextArrowData(arrowtip=(float("nan"), 0.5), direction="right")
        with pytest.raises(ValueError, match="finite"):
            _LabelArrowData(arrowtip=(float("nan"), 0.5), direction="right")

    def test_arrowtip_coerced_to_float_tuple(self):
        arrow = _TextArrowData(arrowtip=(1, 2), direction="right")
        assert arrow.arrowtip == (1.0, 2.0)

    def test_label_arrow_rejects_arrow_length_with_explicit_tail(self):
        with pytest.raises(ValueError, match="cannot be set when"):
            _LabelArrowData(
                arrowtip=(0.5, 0.5),
                direction="up",
                arrow_length_percentage=50.0,
                placement=ArrowPlacement(arrowtail=(0.5, 0.3)),
            )

    def test_arrow_length_percentage_out_of_range_raises_valueerror(self):
        with pytest.raises(ValueError, match="must be in"):
            _LabelArrowData(
                arrowtip=(0.5, 0.5),
                direction="up",
                arrow_length_percentage=101.0,
            )

    def test_arrow_length_percentage_negative_raises_valueerror(self):
        with pytest.raises(ValueError, match="must be in"):
            _LabelArrowData(
                arrowtip=(0.5, 0.5),
                direction="up",
                arrow_length_percentage=-1.0,
            )

    def test_non_finite_label_position_raises_valueerror(self):
        with pytest.raises(ValueError, match="finite"):
            _LabelArrowData(
                arrowtip=(0.5, 0.5),
                direction="up",
                label_position=(float("inf"), 0.5),
            )

    def test_arrow_length_percentage_zero_is_valid(self):
        arrow = _LabelArrowData(
            arrowtip=(0.5, 0.5),
            direction="up",
            arrow_length_percentage=0.0,
        )
        assert arrow.arrow_length_percentage == 0.0

    def test_arrow_length_percentage_100_is_valid(self):
        arrow = _LabelArrowData(
            arrowtip=(0.5, 0.5),
            direction="up",
            arrow_length_percentage=100.0,
        )
        assert arrow.arrow_length_percentage == 100.0

    def test_all_four_directions_are_valid(self):
        for direction in ("right", "left", "up", "down"):
            arrow = _TextArrowData(arrowtip=(0.5, 0.5), direction=direction)
            assert arrow.direction == direction


# ====================
# == BOXPLOTSETDATA ==
# ====================
