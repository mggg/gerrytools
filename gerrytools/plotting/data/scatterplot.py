from dataclasses import dataclass, replace
from typing import Sequence

import numpy as np
from matplotlib.axes import Axes

from gerrytools.logging import get_logger
from gerrytools.plotting.data.gerryplot import GerryPlotBase
from gerrytools.plotting.data.options import DEFAULT_EDGE_WIDTH, _needs_default_edge_width
from gerrytools.plotting.mpl.marker_options import PointMarkerOptions, _marker_legend_handle
from gerrytools.plotting.utils import UNSET, Unset, _replace_with_color_overrides
from gerrytools.typing import Color, LegendHandle

logger = get_logger(__name__)


def _merged_marker_options(
    base: PointMarkerOptions,
    *,
    markerfacecolor: Color | None | Unset,
    markerfacealpha: float | None,
    marker: str | None,
    markersize: float | None,
    markeredgecolor: Color | None | Unset,
    markeredgealpha: float | None,
    markeredgewidth: float | None,
    zorder: int | None,
    edgewidth_given: bool,
) -> PointMarkerOptions:
    """Merge marker kwargs into a base style, preserving color/alpha pairing."""
    resolved = _replace_with_color_overrides(
        base,
        ("markerfacecolor", "markerfacealpha"),
        ("markeredgecolor", "markeredgealpha"),
        markerfacecolor=markerfacecolor,
        markerfacealpha=markerfacealpha,
        marker=marker,
        markersize=markersize,
        markeredgecolor=markeredgecolor,
        markeredgealpha=markeredgealpha,
        markeredgewidth=markeredgewidth,
        zorder=zorder,
    )
    if _needs_default_edge_width(
        edgewidth_given=edgewidth_given,
        resolved_edgewidth=resolved.markeredgewidth,
        resolved_edgecolor=resolved.markeredgecolor,
    ):
        resolved = replace(resolved, markeredgewidth=DEFAULT_EDGE_WIDTH)
    return resolved


@dataclass(slots=True, frozen=True)
class _ScatterData:
    """Data for a scatterplot."""

    x: np.ndarray
    y: np.ndarray
    name: str | None
    marker_options: PointMarkerOptions

    def __post_init__(self) -> None:
        if self.x.shape != self.y.shape:
            raise ValueError("x and y must have the same shape.")
        if self.x.ndim != 1:
            raise ValueError("x and y must be 1-dimensional arrays.")
        if self.x.size == 0:
            raise ValueError("x and y must not be empty.")


class ScatterPlot(GerryPlotBase):
    """A class for creating standard scatterplots."""

    def __init__(
        self,
        *,
        figure_size: tuple[float, float] | None = None,
        dpi: int | None = None,
        ax: Axes | None = None,
        legend: bool | None = None,
        xlabel: str | None = None,
        ylabel: str | None = None,
        title: str | None = None,
    ) -> None:
        """Initialize a ScatterPlot instance.

        Args:
            figure_size (tuple[float, float], optional): The size of the figure in inches.
                Defaults to (10, 6).
            dpi (int, optional): The dots per inch (DPI) of the figure. Defaults to 300.
            ax (matplotlib.axes.Axes | None, optional): Render onto an existing
                matplotlib ``Axes`` instead of creating a fresh figure. Defaults to None.
            legend (bool | None, optional): Whether to include a legend in the plot.
                ``None`` selects the class default (False). Defaults to None.
            xlabel (str | None, optional): The label for the x-axis. Defaults to None.
            ylabel (str | None, optional): The label for the y-axis. Defaults to None.
            title (str | None, optional): The title of the plot. Defaults to None.
        """
        super().__init__(
            figure_size=figure_size,
            dpi=dpi,
            ax=ax,
            legend=legend,
            xlabel=xlabel,
            ylabel=ylabel,
            title=title,
        )

        self._scatter_data_list: list[_ScatterData] = []

    def add_series(
        self,
        x: Sequence[float] | None = None,
        y: Sequence[float] | None = None,
        name: str | None = None,
        *,
        xy_pairs: list[tuple[float, float]] | None = None,
        marker_options: PointMarkerOptions | None = None,
        markerfacecolor: Color | None | Unset = UNSET,
        markerfacealpha: float | None = None,
        marker: str | None = None,
        markersize: float | None = None,
        markeredgecolor: Color | None | Unset = UNSET,
        markeredgealpha: float | None = None,
        markeredgewidth: float | None = None,
        zorder: int | None = None,
    ) -> None:
        """Add a set of points to the scatterplot.

        Args:
            x (Sequence[float] | None): The x-coordinates of the points. Defaults to None.
            y (Sequence[float] | None): The y-coordinates of the points. Defaults to None.
            xy_pairs (list[tuple[float, float]] | None): A list of (x, y) coordinate pairs.
                If provided, x and y should be None. Defaults to None.
            name (str | None, optional): Legend name for the point series. Defaults to None.
            marker_options (PointMarkerOptions | None, optional): Base marker styling. Explicit
                keyword arguments override matching fields. Defaults to None.
            markerfacecolor (Color | None, optional): The face color of the markers. Pass ``None``
                for no fill. Defaults to "#b0b0b0", a medium gray.
            markerfacealpha (float | None, optional): The alpha value for the marker face color.
                Defaults to None.
            marker (str, optional): The marker style. Defaults to "o".
            markersize (float, optional): The size of the markers. Defaults to 6.0.
            markeredgecolor (Color | None, optional): The edge color of the markers. Pass ``None``
                for no edge. With the default style, setting a visible color also selects a
                visible edge width.
            markeredgealpha (float | None, optional): The alpha value for the marker edge color.
                Defaults to the opacity from the base style or selected color.
            markeredgewidth (float | None, optional): The width of the marker edges. With the
                default style, a visible edge color uses 0.8 when the width is omitted; 0
                explicitly hides the edge. Defaults to None.
            zorder (int, optional): The z-order of the markers. Defaults to 1.

        Raises:
            ValueError: If both xy_pairs and x/y are provided, or if neither is provided.
        """
        if xy_pairs is not None:
            if x is not None or y is not None:
                raise ValueError("Specify either xy_pairs or x and y, not both.")
            if len(xy_pairs) == 0:
                raise ValueError("x and y must not be empty.")
            x, y = zip(*xy_pairs)

        if x is None or y is None:
            raise ValueError("Both x and y must be provided.")

        # The scatter-set default style: medium gray, no edge.
        base = (
            marker_options
            if marker_options is not None
            else PointMarkerOptions(
                markerfacecolor="#b0b0b0",
                markersize=6.0,
                markeredgecolor="none",
                markeredgewidth=0.0,
                zorder=1,
            )
        )
        resolved_marker_options = _merged_marker_options(
            base,
            markerfacecolor=markerfacecolor,
            markerfacealpha=markerfacealpha,
            marker=marker,
            markersize=markersize,
            markeredgecolor=markeredgecolor,
            markeredgealpha=markeredgealpha,
            markeredgewidth=markeredgewidth,
            zorder=zorder,
            edgewidth_given=markeredgewidth is not None or marker_options is not None,
        )

        pointset_data = _ScatterData(
            x=np.array(x),
            y=np.array(y),
            name=name,
            marker_options=resolved_marker_options,
        )
        self._scatter_data_list.append(pointset_data)
        self._claim_legend_if_named(name)

    def add_point(
        self,
        x: float,
        y: float,
        name: str,
        *,
        marker_options: PointMarkerOptions | None = None,
        markerfacecolor: Color | None | Unset = UNSET,
        markerfacealpha: float | None = None,
        marker: str | None = None,
        markersize: float | None = None,
        markeredgecolor: Color | None | Unset = UNSET,
        markeredgealpha: float | None = None,
        markeredgewidth: float | None = None,
        zorder: int | None = None,
    ) -> None:
        """Add a single point to the scatterplot.

        Args:
            x (float): The x-coordinate of the point.
            y (float): The y-coordinate of the point.
            name (str): Legend name for the point.
            marker_options (PointMarkerOptions | None, optional): Base marker styling. Explicit
                keyword arguments override matching fields. Defaults to None.
            markerfacecolor (Color | None, optional): The face color of the marker. Pass ``None``
                for no fill. Defaults to "denim".
            markerfacealpha (float | None, optional): The alpha value for the marker face color.
                Defaults to None.
            marker (str, optional): The marker style. Defaults to "o".
            markersize (float, optional): The size of the marker. Defaults to 6.0.
            markeredgecolor (Color | None, optional): The edge color of the marker. Pass ``None``
                for no edge. With the default style, setting a visible color also selects a
                visible edge width.
            markeredgealpha (float | None, optional): The alpha value for the marker edge color.
                Defaults to the opacity from the base style or selected color.
            markeredgewidth (float | None, optional): The width of the marker edge. With the
                default style, a visible edge color uses 0.8 when the width is omitted; 0
                explicitly hides the edge. Defaults to None.
            zorder (int, optional): The z-order of the marker. Defaults to 1.
        """
        # Default for a single labelled point: solid denim fill (distinct from
        # the medium-gray default used by add_series for crowds of points).
        base = (
            marker_options
            if marker_options is not None
            else PointMarkerOptions(
                markerfacecolor="denim",
                markersize=6.0,
                markeredgecolor="none",
                markeredgewidth=0.0,
                zorder=1,
            )
        )
        resolved_marker_options = _merged_marker_options(
            base,
            markerfacecolor=markerfacecolor,
            markerfacealpha=markerfacealpha,
            marker=marker,
            markersize=markersize,
            markeredgecolor=markeredgecolor,
            markeredgealpha=markeredgealpha,
            markeredgewidth=markeredgewidth,
            zorder=zorder,
            edgewidth_given=markeredgewidth is not None or marker_options is not None,
        )
        self.add_series(
            x=[x],
            y=[y],
            name=name,
            marker_options=resolved_marker_options,
        )

    def _draw_points(self) -> None:
        """Draw scatterplots on the plot axes.

        Returns:
            None
        """
        for sdata in self._scatter_data_list:
            point_lines = self._ax.plot(
                sdata.x,
                sdata.y,
                linestyle="none",
                clip_on=True,
                rasterized=True,
                **sdata.marker_options.to_mpl_settings_dict(),
            )
            self._artists.track(point_lines)

    def _build_plot(self) -> None:
        """Build the scatterplot by drawing point sets."""
        if not self._scatter_data_list:
            raise ValueError("No data added yet.")
        self._draw_points()

    def _dataset_legend_handles(self) -> list[LegendHandle]:
        """Generate legend handles for the named point series.

        Returns:
            list[LegendHandle]: A list of legend handles for the point series.
        """
        return [
            _marker_legend_handle(sdata.marker_options, sdata.name)
            for sdata in self._scatter_data_list
            if sdata.name is not None
        ]
