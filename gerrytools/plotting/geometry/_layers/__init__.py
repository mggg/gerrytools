"""Internal layer adapters for the geometry plot classes.

Each layer is a frozen dataclass satisfying the `_Layer` protocol; the color
layers implement the `_GeoLayer` ABC, which owns the shared render pipeline.
"""

from gerrytools.plotting.geometry._layers._base import _as_geoseries, _GeoLayer, _Layer
from gerrytools.plotting.geometry._layers._categorical import _CategoricalColorLayer
from gerrytools.plotting.geometry._layers._continuous import ColormapLayer, _ContinuousColorLayer
from gerrytools.plotting.geometry._layers._marker import _MarkerLayer

__all__ = [
    "_as_geoseries",
    "_GeoLayer",
    "_Layer",
    "_CategoricalColorLayer",
    "_ContinuousColorLayer",
    "_MarkerLayer",
    "ColormapLayer",
]
