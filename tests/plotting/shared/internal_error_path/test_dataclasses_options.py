"""Tests for the public plotting-options dataclasses.

Each dataclass collects the styling kwargs that one ``add_*`` method takes,
so users can compose a style once and reuse it. These tests focus on the
construction contract: defaults, validation, color resolution, and the
``edgecolor="none"`` width-zeroing rule that mirrors the existing data
dataclasses' behavior.
"""

from __future__ import annotations

import pytest

from gerrytools.plotting import (
    UNSET,
    BandOptions,
    BoxPlotOptions,
    HistogramOptions,
    LineOptions,
    PointMarkerOptions,
    SeaLevelLineOptions,
    SeatsVotesLineOptions,
    SeatsVotesMarkerOptions,
    ViolinPlotOptions,
)


class TestLineOptions:
    def test_defaults_construct_cleanly(self):
        options = LineOptions()
        assert isinstance(options.linecolor, str)
        assert options.linecolor.startswith("#")
        assert options.linewidth == 1.0
        assert options.zorder == 3

    def test_negative_linewidth_raises(self):
        with pytest.raises(ValueError, match="linewidth must be nonnegative"):
            LineOptions(linewidth=-1.0)

    def test_infinite_linewidth_raises(self):
        with pytest.raises(ValueError, match="linewidth must be finite"):
            LineOptions(linewidth=float("inf"))

    def test_named_color_resolves_to_hex(self):
        options = LineOptions(linecolor="red")
        assert options.linecolor == "#ff0000"

    def test_none_linecolor_with_positive_width_zeros_width(self):
        options = LineOptions(linecolor="none", linewidth=2.0)
        assert options.linewidth == 0.0


class TestBandOptions:
    def test_defaults_construct_cleanly(self):
        options = BandOptions()
        assert isinstance(options.bandcolor, str)
        assert options.bandcolor.startswith("#")
        assert options.linewidth == 1.0
        assert options.zorder == 3

    def test_negative_linewidth_raises(self):
        with pytest.raises(ValueError, match="linewidth must be nonnegative"):
            BandOptions(linewidth=-1.0)

    def test_named_bandcolor_resolves_to_hex(self):
        options = BandOptions(bandcolor="red")
        assert options.bandcolor == "#ff0000"

    def test_optional_linecolor_resolves_when_provided(self):
        options = BandOptions(bandcolor="red", linecolor="blue")
        assert options.linecolor == "#0000ff"

    def test_omitted_linecolor_inherits_the_band_fill(self):
        options = BandOptions(bandcolor="red")
        assert options.linecolor == "#ff0000"


class TestHistogramOptions:
    def test_defaults_construct_cleanly(self):
        options = HistogramOptions()
        assert isinstance(options.facecolor, str)
        assert options.facecolor.startswith("#")
        assert options.edgecolor == "#000000"
        assert options.edgewidth == 0.0
        assert options.histtype == "overlay"
        assert options.zorder == 2

    def test_named_colors_resolve_to_hex(self):
        options = HistogramOptions(facecolor="red", edgecolor="blue")
        assert options.facecolor == "#ff0000"
        assert options.edgecolor == "#0000ff"

    def test_negative_edgewidth_raises(self):
        with pytest.raises(ValueError, match="edgewidth must be nonnegative"):
            HistogramOptions(edgewidth=-1.0)

    def test_none_edgecolor_with_positive_width_zeros_width(self):
        options = HistogramOptions(edgecolor="none", edgewidth=2.0)
        assert options.edgewidth == 0.0


class TestBoxPlotOptions:
    def test_defaults_construct_cleanly(self):
        options = BoxPlotOptions()
        assert options.percentiles == (1, 99)
        assert options.showfliers is False
        assert isinstance(options.flier_options, PointMarkerOptions)

    def test_invalid_percentiles_raise(self):
        with pytest.raises(ValueError, match=r"percentiles must be within \[0, 100\]"):
            BoxPlotOptions(percentiles=(-1, 99))
        with pytest.raises(ValueError, match=r"percentiles must satisfy low < high"):
            BoxPlotOptions(percentiles=(99, 1))

    def test_named_facecolor_resolves(self):
        options = BoxPlotOptions(facecolor="red")
        assert options.facecolor == "#ff0000"

    def test_custom_flier_options_are_held(self):
        custom_fliers = PointMarkerOptions(markersize=12.0)
        options = BoxPlotOptions(flier_options=custom_fliers)
        assert options.flier_options.markersize == 12.0


class TestViolinPlotOptions:
    def test_defaults_construct_cleanly(self):
        options = ViolinPlotOptions()
        assert options.zorder == 1

    def test_negative_edgewidth_raises(self):
        with pytest.raises(ValueError, match="edgewidth must be nonnegative"):
            ViolinPlotOptions(edgewidth=-0.5)

    def test_named_colors_resolve(self):
        options = ViolinPlotOptions(facecolor="red", edgecolor="blue")
        assert options.facecolor == "#ff0000"
        assert options.edgecolor == "#0000ff"


class TestSeatsVotesLineOptions:
    def test_defaults_construct_cleanly(self):
        options = SeatsVotesLineOptions()
        assert options.linecolor is UNSET
        assert options.linealpha is None
        assert options.linewidth is None
        assert options.zorder == 1

    def test_alpha_out_of_range_raises(self):
        with pytest.raises(ValueError, match=r"linealpha must be in \[0, 1\]"):
            SeatsVotesLineOptions(linealpha=2.0)

    def test_negative_linewidth_raises(self):
        with pytest.raises(ValueError, match="linewidth must be nonnegative"):
            SeatsVotesLineOptions(linewidth=-1.0)


class TestSeatsVotesMarkerOptions:
    def test_defaults_construct_cleanly(self):
        options = SeatsVotesMarkerOptions()
        assert options.markerfacecolor is UNSET
        assert options.markeredgecolor is UNSET
        assert options.marker == "o"
        assert options.markeredgewidth == 0.0
        assert options.marker_zorder == 2

    def test_face_alpha_out_of_range_raises(self):
        with pytest.raises(ValueError, match=r"markerfacealpha must be in \[0, 1\]"):
            SeatsVotesMarkerOptions(markerfacealpha=1.5)

    def test_edge_alpha_out_of_range_raises(self):
        with pytest.raises(ValueError, match=r"markeredgealpha must be in \[0, 1\]"):
            SeatsVotesMarkerOptions(markeredgealpha=-0.1)

    def test_negative_markersize_raises(self):
        with pytest.raises(ValueError, match="markersize must be nonnegative"):
            SeatsVotesMarkerOptions(markersize=-1.0)

    def test_negative_markeredgewidth_raises(self):
        with pytest.raises(ValueError, match="markeredgewidth must be nonnegative"):
            SeatsVotesMarkerOptions(markeredgewidth=-0.5)


class TestSeaLevelLineOptions:
    def test_defaults_construct_cleanly(self):
        options = SeaLevelLineOptions()
        assert options.linecolor == "#000000"
        assert options.linewidth == 1.5
        assert options.zorder == 2

    def test_negative_linewidth_raises(self):
        with pytest.raises(ValueError, match="linewidth must be nonnegative"):
            SeaLevelLineOptions(linewidth=-2.0)

    def test_named_color_resolves(self):
        options = SeaLevelLineOptions(linecolor="red")
        assert options.linecolor == "#ff0000"
