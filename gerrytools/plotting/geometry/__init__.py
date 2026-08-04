"""Geometry-oriented plot classes for GeoDataFrame/GeoSeries rendering."""

from gerrytools.plotting.geometry._labels import LabelOptions
from gerrytools.plotting.geometry._layers._continuous import ColormapLayer
from gerrytools.plotting.geometry.dotdensity import DotDensityPlot
from gerrytools.plotting.geometry.geoplot import GeoPlot

# ``GeoPlotBase`` (in geoplotbase.py) is the abstract base shared by ``GeoPlot`` and
# ``DotDensityPlot``. It is importable for subclassing/typing but intentionally kept out
# of ``__all__``, mirroring how the data side keeps ``GerryPlotBase`` internal.

__all__ = [
    "ColormapLayer",
    "DotDensityPlot",
    "GeoPlot",
    "LabelOptions",
]
