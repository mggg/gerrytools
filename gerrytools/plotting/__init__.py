"""
Makes pretty pictures of districting plans, dual graphs, histograms, boxplots,
and violin plots 🎻.
"""

from .annotation import arrow, ideal
from .bins import bins
from .boxplot import boxplot
from .choropleth import choropleth
from .colors import districtr, flare, latex, purples, redbluecmap
from .districtnumbers import districtnumbers
from .drawgraph import drawgraph
from .drawplan import drawplan
from .gifs import gif_multidimensional
from .histogram import histogram
from .multidimensional import multidimensional
from .scatterplot import scatterplot
from .sealevel import sealevel
from .violin import violin

__all__ = [
    "drawplan",
    "drawgraph",
    "redbluecmap",
    "flare",
    "purples",
    "districtr",
    "histogram",
    "violin",
    "boxplot",
    "scatterplot",
    "sealevel",
    "multidimensional",
    "gif_multidimensional",
    "arrow",
    "ideal",
    "bins",
    "districtnumbers",
    "latex",
    "choropleth",
]
