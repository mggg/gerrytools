from .drawplan import drawplan
from .drawgraph import drawgraph
from .colors import redbluecmap, flare, purples, districtr
from .specification import PlotSpecification
from .plots import (
    plot_histogram,
    draw_arrow
)

"""
Makes pretty pictures of districting plans and dual graphs.
Makes histograms, violin plots, and boxplots of various scores.
"""

__all__ = [
    "drawplan",
    "drawgraph",
    "redbluecmap",
    "PlotSpecification",
    "flare",
    "purples",
    "districtr",
    "plot_histogram",
    "draw_arrow",
]
