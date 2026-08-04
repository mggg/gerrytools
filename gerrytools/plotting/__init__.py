"""
Makes pretty pictures of districting plans, dual graphs, histograms, boxplots,
and violin plots 🎻.
"""

from gerrytools.plotting import data, geometry, mpl, other, plan
from gerrytools.plotting.data import (
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
from gerrytools.plotting.geometry._labels import LabelOptions
from gerrytools.plotting.geometry._layers._continuous import ColormapLayer
from gerrytools.plotting.geometry.dotdensity import DotDensityPlot
from gerrytools.plotting.geometry.geoplot import GeoPlot
from gerrytools.plotting.mpl.geoplot_options import ColorbarOptions
from gerrytools.plotting.mpl.label_text_options import (
    LABEL_STYLES,
    LabelBoxOptions,
    LabelFontOptions,
    LabelStyle,
)
from gerrytools.plotting.mpl.legend_options import LegendOptions
from gerrytools.plotting.mpl.marker_options import PointMarkerOptions
from gerrytools.plotting.other.subway import SubwaySignOptions, subway_signs
from gerrytools.plotting.plan import draw_graph, draw_graph_components
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
    "ColormapLayer",
    "ColorbarOptions",
    "data",
    "geometry",
    "GeoPlot",
    "DotDensityPlot",
    "Histogram",
    "HistogramOptions",
    "LabelArrowOptions",
    "LabelArrowStyle",
    "LABEL_STYLES",
    "LabelBoxOptions",
    "LabelFontOptions",
    "LabelOptions",
    "LabelStyle",
    "LegendOptions",
    "LineOptions",
    "mpl",
    "other",
    "PaintballPlot",
    "plan",
    "PointMarkerOptions",
    "ScatterPlot",
    "SeaLevelPlot",
    "SeaLevelLineOptions",
    "SeatsVotesPlot",
    "SeatsVotesLineOptions",
    "SeatsVotesMarkerOptions",
    "SubwaySignOptions",
    "TextArrowOptions",
    "TextArrowStyle",
    "UNSET",
    "Unset",
    "ViolinPlot",
    "ViolinPlotOptions",
    "draw_graph",
    "draw_graph_components",
    "subway_signs",
]
