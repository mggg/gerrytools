"""`_ContinuousColorLayer` — continuous color mapping over a data column."""

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import matplotlib.pyplot as plt
import pandas as pd
from geopandas import GeoDataFrame
from matplotlib.cm import ScalarMappable
from matplotlib.colors import BoundaryNorm, Colormap, ListedColormap, Normalize, to_hex
from matplotlib.pyplot import get_cmap
from numpy import linspace, searchsorted

from gerrytools.colors import resolve_color_and_alpha
from gerrytools.plotting.geometry._layers._base import _GeoLayer
from gerrytools.plotting.utils import _validated_finite
from gerrytools.typing import MplKwargs, ResolvedColor


@runtime_checkable
class ColormapLayer(Protocol):
    """Public protocol for layers that can produce a standalone colorbar.

    Any object implementing ``column`` and ``mappable()`` satisfies this
    protocol and can be passed to ``GeoPlot.save_colorbar()``.
    """

    @property
    def column(self) -> str | None:
        """Data column used to construct the color mapping."""
        ...

    def mappable(self) -> tuple[ScalarMappable, MplKwargs]:
        """Build the scalar mappable and keyword arguments for a colorbar.

        Returns:
            tuple[ScalarMappable, MplKwargs]: Mappable and colorbar keyword arguments.
        """
        ...


@dataclass(frozen=True, slots=True)
class _ContinuousColorLayer(_GeoLayer):
    """A geographic layer with continuous color mapping based on a data column.

    Attributes:
        geometry_source (GeoSource): The source of geometries for this layer.
        geometry_mask (pd.Series | None): Optional boolean mask to filter geometries.
            Default is None (no mask).
        column (str | None): Optional data column for color mapping. Default is None.
        colormap (str | Colormap): The colormap to use for continuous color mapping.
            Defaults to "Purples".
        missing_color (MplCompatibleColor | None): Color to use for missing data.
        facealpha (float | None): Alpha transparency for face colors. Default is None.
        edgecolor (Color | None): Color for geometry edges. None removes the edge.
            Default is "none".
        edgealpha (float | None): Alpha transparency for edge colors. Default is None.
        edgewidth (float): Width of geometry edges. Default is 0.5.
        zorder (int): Z-order for rendering. Default is 1.
        vmin (float | None): Lower bound value for color mapping.
        vmax (float | None): Upper bound value for color mapping.
        bins (int | list[float] | None): Optional binning specification for discrete intervals.
    """

    colormap: str | Colormap = "Purples"
    vmin: float | None = None
    vmax: float | None = None
    bins: int | list[float] | None = None

    def __post_init__(self) -> None:
        super(_ContinuousColorLayer, self).__post_init__()
        if not isinstance(self.geometry_source, GeoDataFrame):
            raise TypeError(
                "Tried to create a continuous color layer using geo_source of type "
                f"{type(self.geometry_source).__name__!r}; geo_source must be a GeoDataFrame",
            )

        if not isinstance(self.colormap, (str, Colormap)):
            raise TypeError(
                "'colormap' must be a str or Colormap for continuous color layers; got "
                f"{type(self.colormap).__name__!r}",
            )
        if isinstance(self.colormap, str) and self.colormap not in plt.colormaps():
            raise ValueError(
                f"Colormap name {self.colormap!r} not found in matplotlib colormaps. "
                f"Available colormaps are: {plt.colormaps()}",
            )
        if self.column is None:
            raise TypeError("'column' must be set for color-mapped layers")

        for field in ("vmin", "vmax"):
            value = getattr(self, field)
            if value is not None:
                object.__setattr__(self, field, _validated_finite(value, field=field))
        if self.vmin is not None and self.vmax is not None and self.vmin >= self.vmax:
            raise ValueError("'vmin' must be less than 'vmax'.")

        if isinstance(self.bins, bool) or (
            self.bins is not None and not isinstance(self.bins, (int, list))
        ):
            raise TypeError(
                f"'bins' must be an int, a list of breakpoints, or None; got "
                f"{type(self.bins).__name__!r}"
            )
        if isinstance(self.bins, int) and self.bins < 1:
            raise ValueError(f"'bins' must be at least 1; got {self.bins}.")
        if isinstance(self.bins, list):
            if len(self.bins) < 2:
                raise ValueError(
                    f"'bins' given as breakpoints needs at least 2 edges to form one "
                    f"interval; got {len(self.bins)}."
                )
            breaks = [_validated_finite(value, field="'bins' breakpoint") for value in self.bins]
            if self.vmin is not None:
                breaks[0] = self.vmin
            if self.vmax is not None:
                breaks[-1] = self.vmax
            if any(later <= earlier for earlier, later in zip(breaks, breaks[1:], strict=False)):
                raise ValueError("'bins' breakpoints must be strictly increasing.")
            object.__setattr__(self, "bins", breaks)

    def _effective_bounds(self, dataseries: pd.Series) -> tuple[float, float]:
        """Determine the effective data bounds for color mapping.

        Args:
            dataseries (pd.Series): The data series to analyze.
        """
        non_na = dataseries.dropna()
        if non_na.empty:
            lo = float(self.vmin if self.vmin is not None else 0.0)
            hi = float(self.vmax if self.vmax is not None else 1.0)
            return lo, hi

        lo = float(non_na.min() if self.vmin is None else self.vmin)
        hi = float(non_na.max() if self.vmax is None else self.vmax)
        if (self.vmin is not None or self.vmax is not None) and lo >= hi:
            raise ValueError("'vmin' must be less than 'vmax' after applying the data bounds.")
        return lo, hi

    def _bin_boundaries(self, lower: float, upper: float) -> pd.IntervalIndex:
        """Get the bin boundaries as an IntervalIndex.

        Args:
            lower (float): The lower bound of the data.
            upper (float): The upper bound of the data.
        """
        if self.bins is None:
            raise RuntimeError("Called _bin_boundaries but bins is None")

        if isinstance(self.bins, int):
            if lower == upper:
                # A constant column collapses the range; widen it so interval_range still
                # produces distinct bins (and the colorbar distinct boundaries).
                lower -= 0.5
                upper += 0.5
            return pd.interval_range(
                start=lower,
                end=upper,
                periods=self.bins,
                closed="left",
            )

        return pd.IntervalIndex.from_breaks(self.bins, closed="left")

    def _color_mapping_for_bins(
        self, boundaries: pd.IntervalIndex
    ) -> tuple[list[float], list[str]]:
        """Get the color mapping for the given bin boundaries.

        Args:
            boundaries (pd.IntervalIndex): The bin boundaries.

        Returns:
            tuple[list[float], list[str]]: The edges and corresponding hex colors for the bins.
        """
        cmap = get_cmap(self.colormap).resampled(len(boundaries))
        colors = []
        for i in range(len(boundaries)):
            rgba = cmap(i)
            if self.facealpha is not None:
                rgba = (rgba[0], rgba[1], rgba[2], float(self.facealpha))
            colors.append(to_hex(rgba, keep_alpha=True))
        edges = boundaries.left.tolist() + [boundaries.right[-1]]
        return edges, colors

    @staticmethod
    def _with_alpha(cmap: Colormap, alpha: float) -> Colormap:
        """Return a copy of the given colormap with the specified alpha applied.

        Args:
            cmap (Colormap): The original colormap.
            alpha (float): The alpha value to apply (0.0 to 1.0).

        Returns:
            Colormap: A new colormap with the specified alpha applied.
        """
        n = getattr(cmap, "N", 256)
        rgba = cmap(linspace(0, 1, n))
        rgba[:, 3] = float(alpha)
        return ListedColormap(rgba, name=f"{cmap.name}_a{alpha:g}")

    def mappable(self) -> tuple[ScalarMappable, MplKwargs]:
        """Get a ScalarMappable and the colorbar kwargs for this layer.

        Returns:
            tuple[ScalarMappable, MplKwargs]: The ScalarMappable and colorbar kwargs.
        """
        s = self._data_series()
        lower, upper = self._effective_bounds(s)

        if self.bins is not None:
            boundaries = self._bin_boundaries(lower, upper)
            edges, interval_hex_colors = self._color_mapping_for_bins(boundaries)

            listed = ListedColormap(interval_hex_colors)

            norm = BoundaryNorm(edges, ncolors=listed.N, clip=True)

            m = ScalarMappable(norm=norm, cmap=listed)
            m.set_array([])

            cbar_kwargs: MplKwargs = {"ticks": edges}
        else:
            norm = Normalize(vmin=lower, vmax=upper)

            cmap = get_cmap(self.colormap)
            if self.facealpha is not None:
                cmap = self._with_alpha(cmap, self.facealpha)
            m = ScalarMappable(norm=norm, cmap=cmap)
            m.set_array([])
            cbar_kwargs: MplKwargs = {}

        return m, cbar_kwargs

    @property
    def color_series(self) -> pd.Series:
        """Get a series of colors indexed the same as the geometries.

        Colors are built positionally so rows sharing a duplicate index label each keep the
        color of their own value; any ``geometry_mask`` is applied positionally for the same
        reason (label-based reindexing raises on duplicate labels).

        Returns:
            pd.Series: A series of colors for each geometry.
        """
        data_series = self._data_series()
        lower_bound, upper_bound = self._effective_bounds(data_series)
        missing_color = resolve_color_and_alpha(self.missing_color, alpha=self.facealpha)

        colors: list[ResolvedColor] = []

        if self.bins is not None:
            boundaries = self._bin_boundaries(lower_bound, upper_bound)
            edges, interval_hex_colors = self._color_mapping_for_bins(boundaries)

            for value in data_series:
                if pd.isna(value):
                    colors.append(missing_color)
                    continue

                # Bins are closed="left", so side="right" assigns interior breaks to the bin on
                # their right; clamping keeps below-range values in the first bin and the terminal
                # edge (and beyond) in the last.
                interval_i = int(searchsorted(edges, value, side="right")) - 1
                interval_i = min(max(interval_i, 0), len(boundaries) - 1)

                colors.append(
                    resolve_color_and_alpha(
                        interval_hex_colors[interval_i],
                        alpha=self.facealpha,
                    )
                )

        else:
            norm = Normalize(vmin=lower_bound, vmax=upper_bound)
            cmap = get_cmap(self.colormap)

            # One vectorized cmap(norm(...)) pass instead of a per-value loop.
            values = data_series.to_numpy(dtype=float)
            rgba_rows = cmap(norm(values))
            for value, rgba in zip(values, rgba_rows):
                if pd.isna(value):
                    colors.append(missing_color)
                else:
                    colors.append(
                        resolve_color_and_alpha(
                            to_hex(tuple(rgba), keep_alpha=True), alpha=self.facealpha
                        )
                    )

        colors_series = pd.Series(colors, index=self.geometry_source.index)
        if self.geometry_mask is None:
            return colors_series
        return colors_series.iloc[self.geometry_mask.to_numpy(dtype=bool)]
