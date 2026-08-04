import pytest

from gerrytools.plotting.data._gerryplot_dataclasses import (
    _BandData,
    _PointSetData,
)
from gerrytools.plotting.data.options import BandOptions, LineOptions
from gerrytools.plotting.mpl.marker_options import PointMarkerOptions
from tests.plotting._typing_utils import as_any


# =====================================
# == LINE/BAND OPTIONS AND RECORDS   ==
# =====================================
class TestLineOptionsValidation:
    """Line styling validation lives on LineOptions; _LineData is a slim record."""

    def test_default_construction_produces_valid_style(self):
        style = LineOptions()
        assert style.linewidth == 1.0
        assert style.zorder == 3

    def test_linewidth_coerced_from_int_to_float(self):
        style = LineOptions(linewidth=2)
        assert isinstance(style.linewidth, float)
        assert style.linewidth == 2.0

    def test_negative_linewidth_raises_valueerror(self):
        with pytest.raises(ValueError, match="nonnegative"):
            LineOptions(linewidth=-1.0)

    def test_infinite_linewidth_raises_valueerror(self):
        with pytest.raises(ValueError, match="finite"):
            LineOptions(linewidth=float("inf"))

    def test_nan_linewidth_raises_valueerror(self):
        with pytest.raises(ValueError, match="finite"):
            LineOptions(linewidth=float("nan"))

    def test_linecolor_none_with_positive_width_resets_width_to_zero(self):
        style = LineOptions(linecolor="none", linewidth=3.0)
        assert style.linewidth == 0.0

    def test_linecolor_none_string_is_case_insensitive(self):
        style = LineOptions(linecolor="None", linewidth=3.0)
        assert style.linewidth == 0.0

    def test_zorder_coerced_to_int(self):
        style = LineOptions(zorder=as_any(5.9))
        assert isinstance(style.zorder, int)
        assert style.zorder == 5


class TestBandDataAndOptions:
    def test_bounds_are_auto_sorted(self):
        band = _BandData(lower_bound=0.8, upper_bound=0.2, style=BandOptions())
        assert band.lower_bound == 0.2
        assert band.upper_bound == 0.8

    def test_infinite_bounds_raise_valueerror(self):
        with pytest.raises(ValueError, match="finite"):
            _BandData(lower_bound=float("inf"), upper_bound=0.5, style=BandOptions())

    def test_nan_bound_raises_valueerror(self):
        with pytest.raises(ValueError, match="finite"):
            _BandData(lower_bound=float("nan"), upper_bound=0.5, style=BandOptions())

    def test_negative_linewidth_raises_valueerror(self):
        with pytest.raises(ValueError, match="nonnegative"):
            BandOptions(linewidth=-1.0)

    def test_infinite_linewidth_raises_valueerror(self):
        with pytest.raises(ValueError, match="finite"):
            BandOptions(linewidth=float("inf"))

    def test_omitted_linecolor_defaults_to_bandcolor(self):
        style = BandOptions(bandcolor="red")
        assert style.linecolor == "#ff0000"

    def test_omitted_linecolor_with_no_bandcolor_falls_back_to_grey(self):
        style = BandOptions(bandcolor="none")
        assert style.linecolor == "#cccccc"

    def test_explicit_none_linecolor_removes_boundary(self):
        style = BandOptions(bandcolor="red", linecolor=None, linewidth=2.0)
        assert style.linecolor == "none"
        assert style.linewidth == 0.0

    def test_linecolor_none_string_resets_linewidth_to_zero(self):
        style = BandOptions(linecolor="none", linewidth=2.0)
        assert style.linewidth == 0.0

    def test_zorder_coerced_to_int(self):
        style = BandOptions(zorder=as_any(7.2))
        assert isinstance(style.zorder, int)

    def test_equal_bounds_after_sorting_are_valid(self):
        band = _BandData(lower_bound=0.5, upper_bound=0.5, style=BandOptions())
        assert band.lower_bound == band.upper_bound == 0.5

    def test_negative_bounds_are_valid(self):
        band = _BandData(lower_bound=-1.0, upper_bound=-0.5, style=BandOptions())
        assert band.lower_bound == -1.0
        assert band.upper_bound == -0.5


# ==================
# == POINTSETDATA ==
# ==================


class TestPointSetData:
    def test_frozen_cannot_be_mutated(self):
        ps = _PointSetData(
            name="A",
            values_dict={"A": 1.0},
            point_data=PointMarkerOptions(),
        )
        with pytest.raises(AttributeError):
            setattr(ps, "name", "B")


# ====================
# == ARROWTEXTSTYLE ==
# ====================
