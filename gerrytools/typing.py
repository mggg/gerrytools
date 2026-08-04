import enum
from collections.abc import Hashable, Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, Final, Literal, TypeAlias


class Unset(enum.Enum):
    """Sentinel type distinguishing an omitted kwarg from an explicit ``None``.

    Kwargs where ``None`` is itself meaningful default to :data:`UNSET` rather than ``None``: an
    omitted kwarg keeps the stored/base value, while an explicit ``None`` is applied as a real
    value (for example "no fill/edge", clearing a field back to its matplotlib default, or
    restoring inheritance in the LaTeX hull options). Both names are public because they appear in
    ``add_*`` and ``set_*`` signatures: wrappers forwarding those kwargs use ``UNSET`` as their own
    default and ``Unset`` in their annotations. Defined in this dependency-light hub so both the
    plotting and LaTeX packages share the same sentinel object.
    """

    token = enum.auto()


UNSET: Final = Unset.token
"""The sentinel value: the sole member of :class:`Unset`."""


Numeric: TypeAlias = int | float
"""Numeric scalar type used in plotting/color APIs."""

NumericIterable: TypeAlias = Iterable[Numeric]
"""Iterable collection of numeric scalar values."""

RGBColor: TypeAlias = tuple[Numeric, Numeric, Numeric]
"""RGB color tuple in either 0-1 or 0-255 component scale."""

RGBAColor: TypeAlias = tuple[Numeric, Numeric, Numeric, Numeric]
"""RGBA color tuple in either 0-1 or 0-255 component scale."""

MplRGBAColor: TypeAlias = tuple[float, float, float, float]
"""Matplotlib-normalized RGBA tuple with components in ``[0.0, 1.0]``."""

HexColor: TypeAlias = str
"""Hex-encoded color string (for example ``#RRGGBB`` or ``#RRGGBBAA``)."""

Color: TypeAlias = str | RGBColor
"""Public color token type used by GerryTools plotting and LaTeX APIs."""

MplBaseColor: TypeAlias = str | RGBColor | RGBAColor
"""Matplotlib-compatible base color token without explicit external alpha override."""

MplCompatibleColor: TypeAlias = MplBaseColor | tuple[MplBaseColor, Numeric]
"""Color token accepted by Matplotlib conversion helpers, including ``(color, alpha)``."""

ResolvedColor: TypeAlias = tuple[str, float]
"""Resolved color tuple ``(hex6_or_none, alpha)`` returned by color normalizers."""

CategoryKey: TypeAlias = Hashable
"""Hashable category/group key used by categorical and geometry plot APIs."""

CategoryColorMap: TypeAlias = Mapping[CategoryKey, Color | None]
"""Mapping from category keys to colors or transparent fills for categorical layers."""

TickType: TypeAlias = Literal["major", "minor", "both"]
"""Matplotlib tick-set selector accepted by tick styling APIs."""

TikzLineStyle: TypeAlias = Literal[
    "solid",
    "dashed",
    "dotted",
    "dashdotted",
    "loosely dashed",
    "loosely dotted",
    "loosely dashdotted",
    "densely dashed",
    "densely dotted",
    "densely dashdotted",
]
"""Valid TikZ line-style tokens; the single source for runtime validation too."""

HistType: TypeAlias = Literal["overlay", "stack", "grouped", "outline"]
"""Multi-series histogram layout mode."""

MplKwargs: TypeAlias = dict[str, object]
"""Generic Matplotlib kwargs dictionary with JSON-like value constraints."""

# The aliases below need geopandas, matplotlib, numpy, pandas, or pyproj to evaluate. They are
# declared here for static checkers and built lazily at runtime by __getattr__ (PEP 562) so
# importing this module stays dependency-light for consumers that only need the aliases above.
if TYPE_CHECKING:
    from geopandas import GeoDataFrame, GeoSeries
    from matplotlib.artist import Artist
    from matplotlib.colors import Colormap
    from numpy.typing import NDArray
    from pandas import DataFrame, Series
    from pyproj import CRS

    GeoColorMap: TypeAlias = str | Color | Colormap | CategoryColorMap
    """Color mapping specification accepted by geometry plotting layers."""

    CRSLike: TypeAlias = CRS | str | int
    """Coordinate reference system token accepted by GeoPandas reprojection methods."""

    NumericArrayLike: TypeAlias = Numeric | NumericIterable | NDArray | Series | DataFrame
    """Flexible numeric data input accepted by plotting/data coercion utilities."""

    BinsType: TypeAlias = int | Sequence[float] | str | NDArray
    """Histogram bin specification accepted by binning APIs."""

    GeoSource: TypeAlias = GeoDataFrame | GeoSeries
    """Accepted geometry container type for geometry plotting APIs."""

    LegendHandle: TypeAlias = Artist
    """Legend handle artist type returned by plotting classes."""


def __getattr__(name: str) -> object:
    """Build the dependency-heavy aliases on first access and cache them in the module."""

    value: object
    if name == "GeoColorMap":
        from matplotlib.colors import Colormap

        value = str | Color | Colormap | CategoryColorMap
    elif name == "CRSLike":
        from pyproj import CRS

        value = CRS | str | int
    elif name == "NumericArrayLike":
        from numpy.typing import NDArray
        from pandas import DataFrame, Series

        value = Numeric | NumericIterable | NDArray | Series | DataFrame
    elif name == "BinsType":
        from numpy.typing import NDArray

        value = int | Sequence[float] | str | NDArray
    elif name == "GeoSource":
        from geopandas import GeoDataFrame, GeoSeries

        value = GeoDataFrame | GeoSeries
    elif name == "LegendHandle":
        from matplotlib.artist import Artist

        value = Artist
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    globals()[name] = value
    return value
