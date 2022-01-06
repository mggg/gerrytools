
from .drawplan import drawplan
from .drawgraph import drawgraph
from .colors import redbluecmap, flare, purples, districtr
from .specification import PlotSpecification
from .histogram import histogram
from .violin import violin
from .bins import bins
from .annotation import arrow, ideal

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
    "histogram",
    "violin",
    "arrow",
    "ideal",
    "bins"
]
