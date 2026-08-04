"""Tests for matplotlib option dataclasses.

Covers: PointMarkerOptions, TickStyle, AxisLabelStyle, TitleStyle,
LabelFontOptions, LabelBoxOptions, LegendOptions.
"""

import pytest

from gerrytools.plotting.mpl.axis_title_style import AxisLabelStyle, TitleStyle
from gerrytools.plotting.mpl.label_text_options import LabelBoxOptions, LabelFontOptions
from gerrytools.plotting.mpl.legend_options import LegendOptions
from gerrytools.plotting.mpl.marker_options import PointMarkerOptions
from gerrytools.plotting.mpl.tick_style import TickStyle
from tests.plotting._typing_utils import as_any


# ========================
# == POINTMARKEROPTIONS ==
# ========================
class TestPointMarkerOptions:
    def test_default_construction(self):
        pmo = PointMarkerOptions()
        assert pmo.marker == "o"
        assert pmo.markersize == 6.0
        assert pmo.markeredgewidth == 0.6

    def test_negative_markersize_raises_valueerror(self):
        with pytest.raises(ValueError, match="nonnegative"):
            PointMarkerOptions(markersize=-1.0)

    def test_infinite_markersize_raises_valueerror(self):
        with pytest.raises(ValueError, match="finite"):
            PointMarkerOptions(markersize=float("inf"))

    def test_nan_markersize_raises_valueerror(self):
        with pytest.raises(ValueError, match="finite"):
            PointMarkerOptions(markersize=float("nan"))

    def test_negative_markeredgewidth_raises_valueerror(self):
        with pytest.raises(ValueError, match="nonnegative"):
            PointMarkerOptions(markeredgewidth=-0.1)

    def test_infinite_markeredgewidth_raises_valueerror(self):
        with pytest.raises(ValueError, match="finite"):
            PointMarkerOptions(markeredgewidth=float("inf"))

    def test_markeredgecolor_none_resets_edgewidth_to_zero(self):
        pmo = PointMarkerOptions(markeredgecolor="none", markeredgewidth=2.0)
        assert pmo.markeredgewidth == 0.0

    def test_markersize_int_coerced_to_float(self):
        pmo = PointMarkerOptions(markersize=10)
        assert isinstance(pmo.markersize, float)
        assert pmo.markersize == 10.0

    def test_markeredgewidth_int_coerced_to_float(self):
        pmo = PointMarkerOptions(markeredgewidth=1)
        assert isinstance(pmo.markeredgewidth, float)

    def test_zero_markersize_is_valid(self):
        pmo = PointMarkerOptions(markersize=0.0)
        assert pmo.markersize == 0.0

    def test_to_mpl_settings_dict_returns_expected_keys(self):
        pmo = PointMarkerOptions()
        d = pmo.to_mpl_settings_dict()
        expected_keys = {
            "markerfacecolor",
            "marker",
            "markersize",
            "markeredgecolor",
            "markeredgewidth",
            "zorder",
        }
        assert set(d.keys()) == expected_keys

    def test_to_mpl_scatter_settings_dict_returns_expected_keys(self):
        pmo = PointMarkerOptions()
        d = pmo.to_mpl_scatter_settings_dict()
        expected_keys = {"marker", "s", "edgecolor", "linewidths", "zorder"}
        assert set(d.keys()) == expected_keys

    def test_to_mpl_scatter_settings_dict_s_is_markersize_squared(self):
        pmo = PointMarkerOptions(markersize=5.0)
        d = pmo.to_mpl_scatter_settings_dict()
        assert d["s"] == pytest.approx(25.0)

    def test_to_mpl_settings_dict_facecolor_is_rgba_tuple(self):
        pmo = PointMarkerOptions(markerfacecolor="red")
        d = pmo.to_mpl_settings_dict()
        rgba = d["markerfacecolor"]
        assert isinstance(rgba, tuple)
        assert len(rgba) == 4

    def test_mutable_slots_dataclass_allows_mutation(self):
        pmo = PointMarkerOptions()
        pmo.markersize = 20.0
        assert pmo.markersize == 20.0


# ===============
# == TICKSTYLE ==
# ===============
class TestTickStyle:
    def test_default_construction(self):
        ts = TickStyle()
        assert ts.size == 10
        assert ts.rotation == 0
        assert ts.ticktype == "major"

    def test_non_numeric_size_raises_typeerror(self):
        with pytest.raises(TypeError, match="float or int"):
            TickStyle(size=as_any("large"))

    def test_non_finite_size_raises_valueerror(self):
        with pytest.raises(ValueError, match="finite"):
            TickStyle(size=float("inf"))

    def test_negative_size_raises_valueerror(self):
        with pytest.raises(ValueError, match="nonnegative"):
            TickStyle(size=-1.0)

    def test_nan_size_raises_valueerror(self):
        with pytest.raises(ValueError, match="finite"):
            TickStyle(size=float("nan"))

    def test_invalid_ticktype_raises_valueerror(self):
        with pytest.raises(ValueError, match="ticktype"):
            TickStyle(ticktype=as_any("invalid"))

    def test_valid_ticktypes_are_accepted(self):
        for ticktype in ("major", "minor", "both"):
            ts = TickStyle(ticktype=ticktype)
            assert ts.ticktype == ticktype

    def test_zero_size_is_valid(self):
        ts = TickStyle(size=0)
        assert ts.size == 0

    def test_int_size_is_valid(self):
        ts = TickStyle(size=12)
        assert ts.size == 12


# ====================
# == AXISLABELSTYLE ==
# ====================
class TestAxisLabelStyle:
    def test_default_construction(self):
        als = AxisLabelStyle()
        assert als.fontsize is None
        assert als.fontcolor == "#000000"

    def test_non_numeric_fontsize_raises_typeerror(self):
        with pytest.raises(TypeError, match="float or int"):
            AxisLabelStyle(fontsize=as_any("big"))

    def test_negative_fontsize_raises_valueerror(self):
        with pytest.raises(ValueError, match="nonnegative"):
            AxisLabelStyle(fontsize=-1.0)

    def test_infinite_fontsize_raises_valueerror(self):
        with pytest.raises(ValueError, match="finite"):
            AxisLabelStyle(fontsize=float("inf"))

    def test_non_numeric_labelpad_raises_typeerror(self):
        with pytest.raises(TypeError, match="float or int"):
            AxisLabelStyle(labelpad=as_any("wide"))

    def test_negative_labelpad_is_accepted(self):
        # Negative pads are legal in matplotlib (they pull the label inward).
        assert AxisLabelStyle(labelpad=-5.0).labelpad == -5.0

    def test_infinite_labelpad_raises_valueerror(self):
        with pytest.raises(ValueError, match="finite"):
            AxisLabelStyle(labelpad=float("inf"))

    def test_to_mpl_settings_dict_includes_color(self):
        als = AxisLabelStyle()
        d = als.to_mpl_settings_dict()
        assert "color" in d

    def test_to_mpl_settings_dict_omits_none_fields(self):
        als = AxisLabelStyle()
        d = als.to_mpl_settings_dict()
        assert "fontsize" not in d
        assert "fontweight" not in d

    def test_to_mpl_settings_dict_includes_set_fields(self):
        als = AxisLabelStyle(fontsize=14, fontweight="bold", fontstyle="italic")
        d = als.to_mpl_settings_dict()
        assert "fontsize" in d
        assert d["fontsize"] == 14
        assert "fontweight" in d
        assert d["fontweight"] == "bold"
        assert "fontstyle" in d
        assert d["fontstyle"] == "italic"

    def test_none_fontsize_is_valid(self):
        als = AxisLabelStyle(fontsize=None)
        assert als.fontsize is None


# ================
# == TITLESTYLE ==
# ================
class TestTitleStyle:
    def test_default_construction(self):
        ts = TitleStyle()
        assert ts.fontsize is None
        assert ts.loc is None

    def test_non_numeric_fontsize_raises_typeerror(self):
        with pytest.raises(TypeError, match="float or int"):
            TitleStyle(fontsize=as_any("big"))

    def test_negative_fontsize_raises_valueerror(self):
        with pytest.raises(ValueError, match="nonnegative"):
            TitleStyle(fontsize=-1.0)

    def test_infinite_fontsize_raises_valueerror(self):
        with pytest.raises(ValueError, match="finite"):
            TitleStyle(fontsize=float("inf"))

    def test_non_numeric_pad_raises_typeerror(self):
        with pytest.raises(TypeError, match="float or int"):
            TitleStyle(pad=as_any("wide"))

    def test_negative_pad_is_accepted(self):
        # Negative pads are legal in matplotlib (they pull the title inward).
        assert TitleStyle(pad=-1.0).pad == -1.0

    def test_infinite_pad_raises_valueerror(self):
        with pytest.raises(ValueError, match="finite"):
            TitleStyle(pad=float("inf"))

    def test_invalid_loc_raises_valueerror(self):
        with pytest.raises(ValueError, match="loc"):
            TitleStyle(loc=as_any("top"))

    def test_valid_loc_values_are_accepted(self):
        for loc in ("left", "center", "right"):
            ts = TitleStyle(loc=loc)
            assert ts.loc == loc

    def test_to_mpl_settings_dict_includes_color(self):
        ts = TitleStyle()
        d = ts.to_mpl_settings_dict()
        assert "color" in d

    def test_to_mpl_settings_dict_includes_loc_when_set(self):
        ts = TitleStyle(loc="left")
        d = ts.to_mpl_settings_dict()
        assert "loc" in d
        assert d["loc"] == "left"


# ======================
# == LABELFONTOPTIONS ==
# ======================
class TestLabelFontOptions:
    def test_default_construction(self):
        lfo = LabelFontOptions()
        assert lfo.fontsize == 6.0
        assert lfo.fontweight == "bold"

    def test_negative_fontsize_raises(self):
        with pytest.raises(ValueError, match="LabelFontOptions.fontsize must be nonnegative"):
            LabelFontOptions(fontsize=-1.0)

    def test_infinite_fontsize_raises(self):
        with pytest.raises(ValueError, match="LabelFontOptions.fontsize must be finite"):
            LabelFontOptions(fontsize=float("inf"))

    def test_negative_outlinewidth_raises(self):
        with pytest.raises(ValueError, match="LabelFontOptions.outlinewidth must be nonnegative"):
            LabelFontOptions(outlinewidth=-0.5)

    def test_to_mpl_text_kwargs_returns_expected_keys(self):
        lfo = LabelFontOptions()
        kw = lfo.to_mpl_text_kwargs()
        assert "color" in kw
        assert "fontsize" in kw
        assert "fontweight" in kw

    def test_fontfamily_none_omits_from_mpl_kwargs(self):
        lfo = LabelFontOptions(fontfamily=None)
        kw = lfo.to_mpl_text_kwargs()
        assert "fontfamily" not in kw

    def test_fontfamily_set_includes_in_mpl_kwargs(self):
        lfo = LabelFontOptions(fontfamily="monospace")
        kw = lfo.to_mpl_text_kwargs()
        assert kw["fontfamily"] == "monospace"

    def test_fontstretch_none_omits_from_mpl_kwargs(self):
        lfo = LabelFontOptions(fontstretch=None)
        kw = lfo.to_mpl_text_kwargs()
        assert "fontstretch" not in kw


# =====================
# == LABELBOXOPTIONS ==
# =====================
class TestLabelBoxOptions:
    def test_default_construction(self):
        lbo = LabelBoxOptions()
        assert lbo.enabled is True
        assert lbo.boxstyle == "round4"

    def test_disabled_returns_none_from_to_mpl_bbox(self):
        lbo = LabelBoxOptions(enabled=False)
        assert lbo.to_mpl_bbox() is None

    def test_enabled_returns_dict_from_to_mpl_bbox(self):
        lbo = LabelBoxOptions()
        bbox = lbo.to_mpl_bbox()
        assert isinstance(bbox, dict)
        assert "boxstyle" in bbox
        assert "fc" in bbox
        assert "ec" in bbox
        assert "lw" in bbox

    def test_pad_is_included_in_boxstyle_string(self):
        lbo = LabelBoxOptions(pad=0.5)
        bbox = lbo.to_mpl_bbox()
        assert bbox is not None
        assert "0.5" in bbox["boxstyle"]


# ===================
# == LEGENDOPTIONS ==
# ===================
class TestLegendOptions:
    def test_default_construction(self):
        lo = LegendOptions()
        assert lo.loc == "center left"
        assert lo.bbox_to_anchor == (1.01, 0.5)
        assert lo.ncols == 1

    def test_to_dict_excludes_none_values(self):
        lo = LegendOptions()
        d = lo.to_dict()
        assert "fontsize" not in d
        assert "framealpha" not in d

    def test_to_dict_includes_non_none_values(self):
        lo = LegendOptions(fontsize=12.0, title="Legend Title")
        d = lo.to_dict()
        assert d["fontsize"] == 12.0
        assert d["title"] == "Legend Title"

    def test_to_dict_includes_all_truthy_defaults(self):
        lo = LegendOptions()
        d = lo.to_dict()
        assert "loc" in d
        assert "ncols" in d
        assert "frameon" in d

    def test_mutable_dataclass_allows_mutation(self):
        lo = LegendOptions()
        lo.fontsize = 14.0
        assert lo.fontsize == 14.0
        lo.ncols = 3
        assert lo.ncols == 3


class TestAxisLabelStyleOptionalFields:
    def test_labelpad_validation_finite(self):
        style = AxisLabelStyle(labelpad=5.0)
        assert style.labelpad == pytest.approx(5.0)

    def test_to_mpl_settings_dict_includes_fontfamily(self):
        style = AxisLabelStyle(fontfamily="serif")
        d = style.to_mpl_settings_dict()
        assert "fontfamily" in d
        assert d["fontfamily"] == "serif"

    def test_to_mpl_settings_dict_includes_labelpad(self):
        style = AxisLabelStyle(labelpad=8.0)
        d = style.to_mpl_settings_dict()
        assert "labelpad" in d
        assert d["labelpad"] == pytest.approx(8.0)

    def test_to_mpl_settings_dict_omits_fontfamily_when_none(self):
        style = AxisLabelStyle()
        d = style.to_mpl_settings_dict()
        assert "fontfamily" not in d

    def test_to_mpl_settings_dict_omits_labelpad_when_none(self):
        style = AxisLabelStyle()
        d = style.to_mpl_settings_dict()
        assert "labelpad" not in d


class TestTitleStyleOptionalFields:
    def test_pad_validation_finite(self):
        style = TitleStyle(pad=6.0)
        assert style.pad == pytest.approx(6.0)

    def test_pad_not_numeric_raises_typeerror(self):
        with pytest.raises(TypeError, match="float or int"):
            TitleStyle(pad=as_any("large"))

    def test_to_mpl_settings_dict_includes_fontweight(self):
        style = TitleStyle(fontweight="bold")
        d = style.to_mpl_settings_dict()
        assert "fontweight" in d
        assert d["fontweight"] == "bold"

    def test_to_mpl_settings_dict_includes_fontstyle(self):
        style = TitleStyle(fontstyle="italic")
        d = style.to_mpl_settings_dict()
        assert "fontstyle" in d
        assert d["fontstyle"] == "italic"

    def test_to_mpl_settings_dict_includes_fontfamily(self):
        style = TitleStyle(fontfamily="serif")
        d = style.to_mpl_settings_dict()
        assert "fontfamily" in d
        assert d["fontfamily"] == "serif"

    def test_to_mpl_settings_dict_includes_loc(self):
        style = TitleStyle(loc="left")
        d = style.to_mpl_settings_dict()
        assert "loc" in d
        assert d["loc"] == "left"

    def test_to_mpl_settings_dict_includes_pad(self):
        style = TitleStyle(pad=4.0)
        d = style.to_mpl_settings_dict()
        assert "pad" in d
        assert d["pad"] == pytest.approx(4.0)

    def test_to_mpl_settings_dict_omits_fontweight_when_none(self):
        style = TitleStyle()
        d = style.to_mpl_settings_dict()
        assert "fontweight" not in d

    def test_to_mpl_settings_dict_omits_loc_when_none(self):
        style = TitleStyle()
        d = style.to_mpl_settings_dict()
        assert "loc" not in d

    def test_to_mpl_settings_dict_omits_pad_when_none(self):
        style = TitleStyle()
        d = style.to_mpl_settings_dict()
        assert "pad" not in d


class TestLabelFontOptionsStretch:
    def test_fontstretch_included_in_kwargs_when_set(self):
        opts = LabelFontOptions(fontstretch="condensed")
        kw = opts.to_mpl_text_kwargs()
        assert "fontstretch" in kw
        assert kw["fontstretch"] == "condensed"

    def test_fontstretch_omitted_when_none(self):
        opts = LabelFontOptions()
        kw = opts.to_mpl_text_kwargs()
        assert "fontstretch" not in kw
