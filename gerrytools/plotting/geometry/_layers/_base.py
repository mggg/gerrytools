"""`_GeoLayer` ABC and the `_Layer` protocol every geometry layer satisfies."""

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol, cast, overload

import geopandas as gpd
import pandas as pd
from geopandas import GeoDataFrame
from matplotlib.artist import Artist
from matplotlib.axes import Axes

from gerrytools.colors import resolve_color_and_alpha
from gerrytools.typing import (
    Color,
    CRSLike,
    GeoColorMap,
    GeoSource,
    MplCompatibleColor,
)


class _Layer(Protocol):
    """The contract the plot classes rely on: a z-order and a render method.

    Satisfied structurally by ``_GeoLayer`` subclasses and ``_MarkerLayer``.
    """

    @property
    def zorder(self) -> int: ...

    def render(self, ax: Axes, *, target_crs: CRSLike | None = None) -> list[Artist]: ...


def _as_geoseries(source: GeoSource) -> gpd.GeoSeries:
    """Return geometry column as a ``GeoSeries`` for a GeoDataFrame/GeoSeries input.

    Args:
        source (GeoSource): GeoDataFrame or GeoSeries source object.

    Returns:
        gpd.GeoSeries: Geometry series extracted from ``source``.
    """
    return source.geometry if isinstance(source, gpd.GeoDataFrame) else source


def _mask_geoseries(geoseries: gpd.GeoSeries, mask: pd.Series) -> gpd.GeoSeries:
    """Apply a boolean row mask positionally to a GeoSeries.

    Geometry and color masks use the same row positions, including for duplicate or
    differently ordered index labels.
    """
    return cast("gpd.GeoSeries", geoseries.iloc[mask.to_numpy(dtype=bool)])


@overload
def _to_target_crs(geos: gpd.GeoSeries, target_crs: CRSLike | None) -> gpd.GeoSeries: ...
@overload
def _to_target_crs(geos: GeoDataFrame, target_crs: CRSLike | None) -> GeoDataFrame: ...
def _to_target_crs(
    geos: gpd.GeoSeries | GeoDataFrame, target_crs: CRSLike | None
) -> gpd.GeoSeries | GeoDataFrame:
    """Reproject geometries to ``target_crs`` when it is set and differs from the source.

    CRS-less geometries are accepted only when ``target_crs`` is also None. A CRS-bearing
    source likewise requires a target CRS so raw coordinates from unlike systems cannot mix.
    """
    source_crs = getattr(geos, "crs", None)
    if target_crs is None:
        if source_crs is not None:
            raise ValueError(
                "Cannot place CRS-bearing geometries on a plot with no target CRS; "
                "set a target CRS on the plot or remove the source CRS."
            )
        return geos

    if source_crs is None:
        raise ValueError(
            "Cannot place CRS-less geometries on a plot with a target CRS; "
            "set the source CRS before adding the layer."
        )

    if source_crs != target_crs:
        return geos.to_crs(target_crs)

    return geos


@dataclass(frozen=True, slots=True)
class _GeoLayer(ABC):
    """Abstract base class for a geographic layer to be rendered on a GeoPlotBase.

    Subclasses are pure color models: they provide ``color_series`` and the base
    class owns the render pipeline (column check, edge resolution, reprojection,
    geopandas plot call, artist capture).

    Attributes:
        geometry_source (GeoSource): The source of geometries for this layer.
        geometry_mask (pd.Series | None): Optional boolean mask to filter geometries.
            Default is None (no mask).
        column (str | None): Optional data column for color mapping. Default is None.
        colormap (GeoColorMap | None): Color mapping
            specification. Can be a single color, a named colormap, a Colormap object, or
            a mapping from data values to colors. Defaults to "Purples".
        missing_color (MplCompatibleColor | None): Color to use for missing data.
        facealpha (float | None): Alpha transparency for face colors. Default is None.
        edgecolor (Color | None): Color for geometry edges. None removes the edge.
            Default is "none".
        edgealpha (float | None): Alpha transparency for edge colors. Default is None.
        edgewidth (float): Width of geometry edges. Default is 0.5.
        zorder (int): Z-order for rendering. Default is 1.
    """

    # Try to keep the GeoSource as a reference so that users don't copy the polygons all the time.
    geometry_source: GeoSource
    geometry_mask: pd.Series | None = None
    column: str | None = None
    colormap: GeoColorMap | None = "Purples"
    missing_color: MplCompatibleColor | None = "lightgrey"
    facealpha: float | None = None
    edgecolor: Color | None = "none"
    edgealpha: float | None = None
    edgewidth: float = 0.5
    zorder: int = 1

    def _geometries_in_crs(self, target_crs: CRSLike | None) -> gpd.GeoSeries:
        """Return this layer's geometries (respecting mask) reprojected to target_crs.

        Args:
            target_crs (CRSLike | None): Target CRS to reproject to.

        Returns:
            gpd.GeoSeries: The geometries in the target CRS.
        """
        return _to_target_crs(self.geometries, target_crs)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "missing_color",
            resolve_color_and_alpha(
                self.missing_color, alpha=self.facealpha, field="missing_color"
            ),
        )

    @property
    @abstractmethod
    def color_series(self) -> pd.Series:
        """Get a series of colors indexed the same as the geometries."""
        raise NotImplementedError  # pragma: no cover - abstract stub

    @property
    def geometries(self) -> gpd.GeoSeries:
        """Get this layer's geometries, applying any geometry mask."""
        gs = _as_geoseries(self.geometry_source)
        if self.geometry_mask is not None:
            gs = _mask_geoseries(gs, self.geometry_mask)
        return gs

    def _data_series(self) -> pd.Series:
        """Get the data series (used in color mapping).

        geopandas leaves ``__getitem__`` untyped, so checkers infer an ndarray-bearing
        union; a column-label lookup returns a Series at runtime.
        """
        return cast("pd.Series", self.geometry_source[self.column])

    def render(self, ax: Axes, *, target_crs: CRSLike | None = None) -> list[Artist]:
        """Render this layer onto the given Axes.

        Args:
            ax (Axes): The Axes to render onto.
            target_crs (CRSLike | None, optional): The target CRS to reproject geometries to.
                Defaults to None.

        Returns:
            list[Artist]: Every matplotlib artist this layer created on ``ax``.
            On rebuild, only these gerrytools-managed artists are removed, so
            any artists the user added directly to ``ax`` are left untouched.
        """
        if (
            isinstance(self.geometry_source, GeoDataFrame)
            and self.column is not None
            and self.column not in self.geometry_source.columns
        ):
            raise KeyError(
                f"Column {self.column!r} not found in GeoDataFrame."
                f" Available columns: {list(self.geometry_source.columns)}"
            )

        edge_color_tup = resolve_color_and_alpha(
            self.edgecolor,
            alpha=self.edgealpha,
        )

        geoseries = self._geometries_in_crs(target_crs)
        return _capture_geopandas_artists(
            ax,
            plot_call=lambda: geoseries.plot(
                ax=ax,
                color=self.color_series,
                edgecolor=edge_color_tup,
                linewidth=self.edgewidth,
                zorder=self.zorder,
            ),
        )


def _capture_geopandas_artists(
    ax: Axes,
    *,
    plot_call: Callable[[], object],
) -> list[Artist]:
    """Snapshot ``ax`` collection/line/patch/text counts around a geopandas call.

    Geopandas' ``GeoSeries.plot(ax=ax, ...)`` returns the axes, not the
    artists it created. This helper diffs ``ax``'s artist lists before and
    after invoking ``plot_call()`` so the caller can hand the resulting
    artists to the registry.
    """
    before_collections = len(ax.collections)
    before_lines = len(ax.lines)
    before_patches = len(ax.patches)
    before_texts = len(ax.texts)

    plot_call()

    new_artists: list[Artist] = []
    new_artists.extend(list(ax.collections[before_collections:]))
    new_artists.extend(list(ax.lines[before_lines:]))
    new_artists.extend(list(ax.patches[before_patches:]))
    new_artists.extend(list(ax.texts[before_texts:]))
    return new_artists
