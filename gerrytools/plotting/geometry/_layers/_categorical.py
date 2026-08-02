"""`_CategoricalColorLayer` — categorical color mapping over a data column."""

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import cast
from warnings import warn

import matplotlib.pyplot as plt
import pandas as pd
from geopandas import GeoDataFrame, GeoSeries
from matplotlib.colors import Colormap, to_hex
from matplotlib.pyplot import get_cmap
from numpy import linspace

from gerrytools.colors import districtr, resolve_color_and_alpha
from gerrytools.plotting.geometry._layers._base import _GeoLayer
from gerrytools.typing import (
    CategoryColorMap,
    CategoryKey,
    Color,
    GeoColorMap,
    MplCompatibleColor,
    ResolvedColor,
)


def _resolves_as_color(value: str) -> bool:
    """Whether a string resolves as a gerrytools/matplotlib color."""
    try:
        resolve_color_and_alpha(value)
    except ValueError:
        return False
    return True


@dataclass(frozen=True, slots=True)
class _CategoricalColorLayer(_GeoLayer):
    """A geographic layer with categorical color mapping based on a data column.

    Attributes:
        geometry_source (GeoSource): The source of geometries for this layer.
        geometry_mask (pd.Series | None): Optional boolean mask to filter geometries.
            Default is None (no mask).
        column (str | None): Optional data column for color mapping. Default is None.
        colormap (GeoColorMap | None): Color mapping
            specification. Can be a single color, a named colormap, a Colormap object, or
            a mapping from data values to colors. When ``column`` is set, a string is
            resolved as a registered Matplotlib colormap name first and as a flat color
            otherwise; without a column, strings always resolve as flat colors. A string
            that is both a colormap name and a color (e.g. ``"pink"``) resolves as the
            colormap and emits a ``UserWarning``. Defaults to "districtr".
        missing_color (MplCompatibleColor | None): Color to use for missing data.
        facealpha (float | None): Alpha transparency for face colors. Default is None.
        edgecolor (Color | None): Color for geometry edges. None removes the edge.
            Default is "none".
        edgealpha (float | None): Alpha transparency for edge colors. Default is None.
        edgewidth (float): Width of geometry edges. Default is 0.5.
        zorder (int): Z-order for rendering. Default is 1.
    """

    colormap: GeoColorMap | None = "districtr"

    def __post_init__(self) -> None:
        super(_CategoricalColorLayer, self).__post_init__()

        if isinstance(self.geometry_source, GeoSeries) and self.colormap == "districtr":
            object.__setattr__(self, "colormap", "none")

        if self.colormap is not None and not isinstance(
            self.colormap, (str, tuple, Mapping, Colormap)
        ):
            raise TypeError(
                "'colormap' must be one of: None, str (named colormap or color), "
                "Colormap, or mapping; got "
                f"{type(self.colormap).__name__!r}",
            )

        needs_datacolumn = (
            self.colormap == "districtr"
            or isinstance(self.colormap, (Mapping, Colormap))
            or (isinstance(self.colormap, str) and self.colormap in plt.colormaps())
        )

        # A Colormap object or value->color dict needs a data column, which a bare GeoSeries
        # cannot supply. (Named-colormap strings are excluded: names like "gray" are also valid
        # single colors, which is exactly how GeoSeries-sourced outline/highlight layers use
        # them, so string colormaps on a GeoSeries resolve as plain colors at render time.)
        if isinstance(self.geometry_source, GeoSeries) and isinstance(
            self.colormap, (Mapping, Colormap)
        ):
            raise TypeError(
                "A Colormap or value->color mapping requires a data column; the layer's "
                "geo_source is a GeoSeries, which has none. Pass a GeoDataFrame and 'column'."
            )

        if (
            isinstance(self.geometry_source, GeoDataFrame)
            and needs_datacolumn
            and self.column is None
        ):
            raise TypeError("'column' must be set for color-mapped layers")

        # A string like "pink" is both a registered colormap name and a valid color; with a
        # data column set the colormap interpretation wins, so flag the ambiguity.
        if (
            isinstance(self.geometry_source, GeoDataFrame)
            and self.column is not None
            and isinstance(self.colormap, str)
            and self.colormap != "districtr"
            and self.colormap in plt.colormaps()
            and _resolves_as_color(self.colormap)
        ):
            warn(
                f"colormap={self.colormap!r} is both a registered Matplotlib colormap name and "
                "a valid color; since a data column is set, it is interpreted as the colormap. "
                "Pass a Colormap instance (e.g. matplotlib.pyplot.get_cmap"
                f"({self.colormap!r})) to map values, or an RGBA tuple / hex string for a "
                "flat color.",
                UserWarning,
                stacklevel=2,
            )

        if self.colormap == "districtr" and isinstance(self.geometry_source, GeoDataFrame):
            unique_values = self.geometry_source[self.column].unique()
            districtr_colors = districtr(len(unique_values))
            object.__setattr__(
                self,
                "colormap",
                self._map_unique_values_to_colors(unique_values, districtr_colors),
            )
        elif isinstance(self.geometry_source, GeoDataFrame) and (
            isinstance(self.colormap, Colormap)
            or (isinstance(self.colormap, str) and self.colormap in plt.colormaps())
        ):
            cmap = get_cmap(self.colormap) if isinstance(self.colormap, str) else self.colormap
            unique_values = self.geometry_source[self.column].unique()
            category_count = sum(bool(pd.notna(value)) for value in unique_values)
            n_colors = int(getattr(cmap, "N", 256))
            # Sample at most the LUT size; with more categories than LUT entries the shared
            # not-enough-colors guard in _map_unique_values_to_colors raises.
            colors = [
                to_hex(cmap(position), keep_alpha=True)
                for position in linspace(0.0, 1.0, min(category_count, n_colors))
            ]
            object.__setattr__(
                self,
                "colormap",
                self._map_unique_values_to_colors(unique_values, colors),
            )

    @staticmethod
    def _map_unique_values_to_colors(
        unique_values: Iterable[CategoryKey],
        color_list: Sequence[Color],
    ) -> CategoryColorMap:
        """Map unique values in the data to colors from the provided list. Filters out NaN values.

        Args:
            unique_values (Iterable[CategoryKey]): The unique values to map.
            color_list (Sequence[Color]): The colors to use for mapping.
        """
        n_colors = len(color_list)
        all_values = list(unique_values)
        non_na_values: list[CategoryKey] = []
        for value in all_values:
            # Keys are scalars, so pd.notna returns a bool; bool() pins the stub union.
            if bool(pd.notna(value)):
                non_na_values.append(value)

        if len(non_na_values) > n_colors:
            raise ValueError(
                "Not enough colors provided to map all unique values; "
                f"received {n_colors} colors for {len(non_na_values)} unique values",
            )

        # Sort numeric-looking labels numerically so strings and integral floats agree.
        try:
            key_number_pairs = [(key, float(str(key))) for key in non_na_values]
            if any(not math.isfinite(number) for _, number in key_number_pairs):
                raise ValueError
            sorted_keys = sorted(key_number_pairs, key=lambda pair: pair[1])
            keys_in_order = [k for (k, _) in sorted_keys]
        except (ValueError, TypeError):
            keys_in_order = sorted(non_na_values, key=lambda value: str(value))

        return {k: color_list[i] for i, k in enumerate(keys_in_order)}

    @property
    def color_series(self) -> pd.Series:
        """Get a series of colors indexed the same as the geometries.

        Any ``geometry_mask`` is applied positionally: label-based reindexing would raise on
        the duplicate index labels a ``pd.concat``-built GeoDataFrame routinely carries.

        Returns:
            pd.Series: A series of colors for each geometry.
        """
        ret_colors_series: pd.Series

        if self.colormap is None:
            ret_colors_series = pd.Series(
                ["none"] * len(self.geometry_source), index=self.geometry_source.index
            )

        elif isinstance(self.colormap, (str, tuple)):
            color = resolve_color_and_alpha(
                cast("MplCompatibleColor", self.colormap), alpha=self.facealpha
            )
            ret_colors_series = pd.Series(
                [color] * len(self.geometry_source), index=self.geometry_source.index
            )

        elif isinstance(self.colormap, Mapping):
            new_entries: list[ResolvedColor] = []
            for val in self._data_series():
                color = cast(
                    "MplCompatibleColor | None",
                    self.colormap.get(val, self.missing_color),
                )
                color_tup = resolve_color_and_alpha(color, alpha=self.facealpha)
                new_entries.append(color_tup)
            ret_colors_series = pd.Series(new_entries, index=self.geometry_source.index)
        else:  # pragma: no cover - __post_init__ normalizes or rejects every other input
            raise RuntimeError("Categorical colormap was not normalized.")
        if self.geometry_mask is None:
            return ret_colors_series
        return ret_colors_series.iloc[self.geometry_mask.to_numpy(dtype=bool)]
