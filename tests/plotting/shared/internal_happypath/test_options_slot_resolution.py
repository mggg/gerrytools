"""Tests for the ``*_options=`` slot-resolution rule.

Each ``add_*`` method accepts a styling options dataclass through
an ``options=`` (or ``line_options=``/``band_options=``/``marker_options=``) parameter.
The resolution rule is: explicit kwargs override the options' fields. These
tests assert that contract end-to-end through the public API.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402

from gerrytools.plotting import (  # noqa: E402
    BandOptions,
    BoxPlot,
    BoxPlotOptions,
    Histogram,
    HistogramOptions,
    LineOptions,
    PointMarkerOptions,
    ScatterPlot,
    SeaLevelLineOptions,
    SeaLevelPlot,
    SeatsVotesLineOptions,
    SeatsVotesMarkerOptions,
    SeatsVotesPlot,
    ViolinPlot,
    ViolinPlotOptions,
)
from gerrytools.plotting.data.boxplot import _BoxPlotSetData  # noqa: E402

# ----------------------------------------------------------------------
# add_vertical_lines / add_horizontal_lines accept line_options=
# ----------------------------------------------------------------------


class TestLineOptionsSlot:
    def test_options_alone_propagates_to_stored_state(self):
        plot = Histogram()
        plot.add_vertical_lines(
            [1.0, 2.0],
            line_options=LineOptions(linecolor="red", linewidth=2.5),
        )
        line = plot._annotations.vertical_lines[0]
        assert line.style.linecolor == "#ff0000"
        assert line.style.linewidth == 2.5

    def test_kwarg_overrides_options(self):
        plot = Histogram()
        plot.add_vertical_lines(
            [1.0],
            line_options=LineOptions(linecolor="red", linewidth=2.5),
            linewidth=5.0,  # explicit override
        )
        line = plot._annotations.vertical_lines[0]
        assert line.style.linecolor == "#ff0000"  # came from options
        assert line.style.linewidth == 5.0  # came from explicit kwarg

    def test_horizontal_lines_options_propagate(self):
        plot = Histogram()
        plot.add_horizontal_lines(
            [0.5],
            line_options=LineOptions(linecolor="blue", linestyle="--"),
        )
        line = plot._annotations.horizontal_lines[0]
        assert line.style.linecolor == "#0000ff"
        assert line.style.linestyle == "--"


# ----------------------------------------------------------------------
# add_vertical_band / add_horizontal_band accept band_options=
# ----------------------------------------------------------------------


class TestBandOptionsSlot:
    def test_options_alone_propagates(self):
        plot = Histogram()
        plot.add_vertical_band(
            0.0,
            1.0,
            band_options=BandOptions(bandcolor="green", bandalpha=0.3),
        )
        band = plot._annotations.vertical_bands[0]
        assert band.style.bandcolor == "#00ff00"
        assert band.style.bandalpha == 0.3

    def test_kwarg_overrides_options(self):
        plot = Histogram()
        plot.add_vertical_band(
            0.0,
            1.0,
            band_options=BandOptions(bandcolor="green", bandalpha=0.3),
            bandcolor="red",  # explicit override
        )
        band = plot._annotations.vertical_bands[0]
        assert band.style.bandcolor == "#ff0000"  # explicit kwarg won
        assert band.style.bandalpha == 0.3  # came from options


# ----------------------------------------------------------------------
# Histogram options=
# ----------------------------------------------------------------------


class TestHistogramOptionsSlot:
    def test_default_uses_black_edges(self):
        plot = Histogram()
        plot.add_dataset([1.0, 2.0, 3.0])
        hist = plot._hist_data_dict["overlay"][0]
        assert hist.style.edgecolor == "#000000"
        assert hist.style.edgewidth == 0.8

    def test_options_alone_propagates(self):
        plot = Histogram()
        plot.add_dataset(
            [1.0, 2.0, 3.0, 2.0, 1.0],
            options=HistogramOptions(facecolor="red", edgecolor="blue", edgewidth=1.5),
        )
        hist = plot._hist_data_dict["overlay"][0]
        assert hist.style.facecolor == "#ff0000"
        assert hist.style.edgecolor == "#0000ff"
        assert hist.style.edgewidth == 1.5

    def test_explicit_zero_edgewidth_in_options_stays_hidden(self):
        plot = Histogram()
        plot.add_dataset(
            [1.0, 2.0, 3.0],
            options=HistogramOptions(edgecolor="black", edgewidth=0.0),
        )

        assert plot._hist_data_dict["overlay"][0].style.edgewidth == 0.0

    def test_kwarg_overrides_options(self):
        plot = Histogram()
        plot.add_dataset(
            [1.0, 2.0, 3.0],
            options=HistogramOptions(facecolor="red"),
            facecolor="green",  # explicit
        )
        hist = plot._hist_data_dict["overlay"][0]
        assert hist.style.facecolor == "#00ff00"


# ----------------------------------------------------------------------
# BoxPlot options=
# ----------------------------------------------------------------------


class TestBoxPlotOptionsSlot:
    def test_options_alone_propagates(self):
        plot = BoxPlot()
        plot.add_dataset(
            {"A": [1.0, 2.0, 3.0]},
            options=BoxPlotOptions(facecolor="red", percentiles=(5, 95)),
        )
        bp = plot._boxplot_data_list[0]
        assert isinstance(bp, _BoxPlotSetData)
        assert bp.style.facecolor == "#ff0000"
        assert bp.style.percentiles == (5, 95)

    def test_kwarg_overrides_options(self):
        plot = BoxPlot()
        plot.add_dataset(
            {"A": [1.0, 2.0, 3.0]},
            options=BoxPlotOptions(facecolor="red"),
            facecolor="blue",
        )
        bp = plot._boxplot_data_list[0]
        assert bp.style.facecolor == "#0000ff"


# ----------------------------------------------------------------------
# ViolinPlot options=
# ----------------------------------------------------------------------


class TestViolinPlotOptionsSlot:
    def test_options_alone_propagates(self):
        plot = ViolinPlot()
        plot.add_dataset(
            {"A": [1.0, 2.0, 3.0, 4.0, 5.0]},
            options=ViolinPlotOptions(facecolor="red", edgewidth=1.2),
        )
        vp = plot._violinplot_data_list[0]
        assert vp.style.facecolor == "#ff0000"
        assert vp.style.edgewidth == 1.2


# ----------------------------------------------------------------------
# ScatterPlot marker_options=
# ----------------------------------------------------------------------


class TestScatterMarkerOptionsSlot:
    def test_options_alone_propagates(self):
        plot = ScatterPlot()
        plot.add_series(
            x=[0, 1],
            y=[0, 1],
            marker_options=PointMarkerOptions(
                markerfacecolor="red", markersize=12.0, markeredgecolor="none"
            ),
        )
        sd = plot._scatter_data_list[0]
        assert sd.marker_options.markerfacecolor == "#ff0000"
        assert sd.marker_options.markersize == 12.0

    def test_kwarg_overrides_marker_options(self):
        plot = ScatterPlot()
        plot.add_series(
            x=[0, 1],
            y=[0, 1],
            marker_options=PointMarkerOptions(
                markerfacecolor="red", markersize=12.0, markeredgecolor="none"
            ),
            markersize=20.0,
        )
        sd = plot._scatter_data_list[0]
        assert sd.marker_options.markerfacecolor == "#ff0000"  # from options
        assert sd.marker_options.markersize == 20.0  # from kwarg

    def test_explicit_zero_edgewidth_in_options_stays_hidden(self):
        plot = ScatterPlot()
        plot.add_series(
            x=[0, 1],
            y=[0, 1],
            marker_options=PointMarkerOptions(markeredgecolor="black", markeredgewidth=0.0),
        )

        assert plot._scatter_data_list[0].marker_options.markeredgewidth == 0.0


# ----------------------------------------------------------------------
# SeatsVotesPlot line_options + marker_options
# ----------------------------------------------------------------------


class TestSeatsVotesOptionsSlots:
    def test_line_options_propagate(self):
        plot = SeatsVotesPlot()
        plot.add_election(
            np.array([0.4, 0.5, 0.6]),
            np.array([1.0, 1.0, 1.0]),
            line_options=SeatsVotesLineOptions(linecolor="red", linewidth=3.5),
        )
        sv = plot._sv_data_list[0]
        assert sv.line_style.linecolor == "#ff0000"
        assert sv.line_style.linewidth == 3.5

    def test_marker_options_propagate(self):
        plot = SeatsVotesPlot()
        plot.add_election(
            np.array([0.4, 0.5, 0.6]),
            np.array([1.0, 1.0, 1.0]),
            marker_options=SeatsVotesMarkerOptions(markerfacecolor="green", markersize=15.0),
        )
        sv = plot._sv_data_list[0]
        assert sv.marker_style.markerfacecolor == "#00ff00"
        assert sv.marker_style.markersize == 15.0

    def test_kwarg_overrides_marker_options(self):
        plot = SeatsVotesPlot()
        plot.add_election(
            np.array([0.4, 0.5, 0.6]),
            np.array([1.0, 1.0, 1.0]),
            marker_options=SeatsVotesMarkerOptions(markerfacecolor="#00ff00"),
            markerfacecolor="#ff0000",
        )
        sv = plot._sv_data_list[0]
        assert sv.marker_style.markerfacecolor == "#ff0000"

    def test_explicit_zero_edgewidth_in_options_stays_hidden(self):
        plot = SeatsVotesPlot()
        plot.add_election(
            np.array([0.4, 0.5, 0.6]),
            marker_options=SeatsVotesMarkerOptions(
                markeredgecolor="black",
                markeredgewidth=0.0,
            ),
        )

        assert plot._sv_data_list[0].marker_style.markeredgewidth == 0.0


# ----------------------------------------------------------------------
# SeaLevelPlot line_options + marker_options
# ----------------------------------------------------------------------


class TestSeaLevelOptionsSlots:
    def test_line_options_propagate(self):
        plot = SeaLevelPlot()
        plot.add_dataset(
            {"A": 1.0, "B": 2.0},
            line_options=SeaLevelLineOptions(linecolor="red", linewidth=3.5),
        )
        sl = plot._sealevel_data_list[0]
        assert sl.style.linecolor == "#ff0000"
        assert sl.style.linewidth == 3.5

    def test_marker_options_propagate(self):
        plot = SeaLevelPlot()
        plot.add_dataset(
            {"A": 1.0, "B": 2.0},
            marker_options=PointMarkerOptions(
                markerfacecolor="blue", markersize=10.0, markeredgecolor="black"
            ),
        )
        sl = plot._sealevel_data_list[0]
        assert sl.markersettings.markerfacecolor == "#0000ff"
        assert sl.markersettings.markersize == 10.0
