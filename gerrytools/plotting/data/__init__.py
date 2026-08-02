"""Data-oriented Matplotlib plot classes."""

from gerrytools.plotting.data._gerryplot_dataclasses import (
    ArrowPlacement,
    ArrowTextStyle,
    LabelArrowOptions,
    LabelArrowStyle,
    TextArrowOptions,
    TextArrowStyle,
)
from gerrytools.plotting.data.barplot import BarPlot
from gerrytools.plotting.data.boxplot import BoxPlot, BoxPlotStats
from gerrytools.plotting.data.histogram import Histogram
from gerrytools.plotting.data.options import (
    BandOptions,
    BarPlotOptions,
    BoxPlotOptions,
    HistogramOptions,
    LineOptions,
    SeaLevelLineOptions,
    SeatsVotesLineOptions,
    SeatsVotesMarkerOptions,
    ViolinPlotOptions,
)
from gerrytools.plotting.data.paintball import PaintballPlot
from gerrytools.plotting.data.scatterplot import ScatterPlot
from gerrytools.plotting.data.sealevel import SeaLevelPlot
from gerrytools.plotting.data.seatsvotes import SeatsVotesPlot
from gerrytools.plotting.data.violin import ViolinPlot
from gerrytools.plotting.utils import UNSET, Unset

__all__ = [
    "ArrowPlacement",
    "ArrowTextStyle",
    "BandOptions",
    "BarPlot",
    "BarPlotOptions",
    "BoxPlot",
    "BoxPlotOptions",
    "BoxPlotStats",
    "Histogram",
    "HistogramOptions",
    "LabelArrowOptions",
    "LabelArrowStyle",
    "LineOptions",
    "PaintballPlot",
    "ScatterPlot",
    "SeaLevelPlot",
    "SeaLevelLineOptions",
    "SeatsVotesPlot",
    "SeatsVotesLineOptions",
    "SeatsVotesMarkerOptions",
    "TextArrowOptions",
    "TextArrowStyle",
    "UNSET",
    "Unset",
    "ViolinPlot",
    "ViolinPlotOptions",
]
