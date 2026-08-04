from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from itertools import chain
from warnings import warn

import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.markers import MarkerStyle
from matplotlib.patches import Patch
from numpy.typing import NDArray

from gerrytools.logging import get_logger
from gerrytools.plotting._axes_backed import deferred_axis_update
from gerrytools.plotting._axes_state import _recompute_data_limits
from gerrytools.plotting.data.gerryplot import GerryPlotBase
from gerrytools.plotting.data.options import (
    DEFAULT_EDGE_WIDTH,
    HistogramOptions,
    _needs_default_edge_width,
)
from gerrytools.plotting.mpl.marker_options import PointMarkerOptions, _marker_legend_handle
from gerrytools.plotting.utils import (
    UNSET,
    Unset,
    _coerce_to_1d_finite_float_array,
    _coerce_values_and_weights,
    _replace_with_color_overrides,
)
from gerrytools.typing import BinsType, Color, HistType, LegendHandle

logger = get_logger(__name__)


def _marker_clearance_pt(
    marker: str,
    markersize_pt: float,
    markeredgewidth_pt: float,
) -> float:
    """Compute the point-space clearance one marker glyph needs above a bar.

    Args:
        marker (str): Marker symbol passed to Matplotlib.
        markersize_pt (float): Marker size in points.
        markeredgewidth_pt (float): Marker edge width in points.

    Returns:
        float: Clearance in points.
    """
    marker_style = MarkerStyle(marker)
    path = marker_style.get_path().transformed(marker_style.get_transform())

    verts = np.asarray(path.vertices, dtype=float)
    ys = verts[:, 1]
    y0 = float(ys.min())
    y1 = float(ys.max())

    height_u = y1 - y0
    if (
        height_u == 0.0
    ):  # pragma: no cover - no standard matplotlib marker produces an exactly-zero y-extent; defensive guard
        return 0.5 * markersize_pt + 0.5 * markeredgewidth_pt  # pragma: no cover

    pt_per_u = markersize_pt / height_u
    bottom_pt = y0 * pt_per_u
    return (-bottom_pt) + 0.5 * markeredgewidth_pt


@dataclass(frozen=True)
class _HistPointList:
    """A dataclass representing a list of points to be plotted on a histogram figure.

    Attributes:
        name (str): The name of the point set.
        values (NDArray[np.float64]): The point values to be plotted.
        point_data (PointMarkerOptions): Settings for how the points should be marked on the plot.
        y_offset (float): Absolute y-offset added above the bar top for the first point
            landing in each bin.
        centered (bool): Whether points are centered on their histogram bins.
    """

    name: str
    values: NDArray[np.float64]
    point_data: PointMarkerOptions
    y_offset: float = 0.0
    centered: bool = False


@dataclass(frozen=True)
class _HistogramData:
    """One histogram layer/series.

    Attributes:
        name (str): Name of the histogram series.
        values (NDArray[np.float64]): 1D array of values to histogram.
        weights (NDArray[np.float64]): 1D array of weights for the values.
        style (HistogramOptions): Resolved styling for the series.
    """

    name: str
    values: NDArray[np.float64]
    weights: NDArray[np.float64]
    style: HistogramOptions

    def __post_init__(self) -> None:
        # add_dataset already rejects empty/non-finite input; only coerce shapes here.
        vals = np.asarray(self.values, dtype=float).ravel()
        object.__setattr__(self, "values", vals)

        weights_arr = np.asarray(self.weights, dtype=float).ravel()
        if weights_arr.size != vals.size:
            raise ValueError("weights must have the same length as values.")
        object.__setattr__(self, "weights", weights_arr)


class Histogram(GerryPlotBase):
    """Overlayed histogram comparison figure.

    Typical usage::

        h = Histogram()
        h.add_dataset(df["ensemble1"], name="Ensemble", facecolor="denim", facealpha=0.35)
        h.add_dataset(
            df["ensemble2"],
            name="Plan",
            histtype="outline",
            facecolor="black",
            facealpha=1.0,
        )
        h.add_vertical_lines(plan_value, linecolor="black", name="Plan value")
        h.show()

    """

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
        """Initialize the Histogram figure.

        Args:
            figure_size (tuple[float, float] | None, optional): The size of the
                figure in inches. Defaults to ``(10, 6)`` when ``ax`` is not provided.
            dpi (int | None, optional): The dots per inch (DPI) of the figure.
                Defaults to ``300`` when ``ax`` is not provided.
            ax (matplotlib.axes.Axes | None, optional): Render onto an existing
                matplotlib ``Axes`` instead of creating a fresh figure. Defaults to None.
            legend (bool, optional): Whether to include a legend in the plot.
                Defaults to False.
            xlabel (str | None, optional): The label for the x-axis. Defaults to None.
            ylabel (str | None, optional): The label for the y-axis. Defaults to None.
            title (str | None, optional): The title of the plot. Defaults to None.

        Toggle the grid or histogram-configuration warnings with :meth:`display_grid` and
        :meth:`display_warnings` after construction.
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
        self._show_warnings = True
        self._bins: BinsType | None = None
        self._bin_alignment = "edge"
        self._as_density_plot = False
        self._binwidth: float | None = None

        self._hist_data_dict: dict[str, list[_HistogramData]] = {
            "overlay": [],
            "grouped": [],
            "outline": [],
            "stack": [],
        }
        self._histpointlist_list: list[_HistPointList] = []

    @property
    def as_density_plot(self) -> bool:
        """Whether histogram values are normalized as densities.

        Each series is normalized on its own, except ``'stack'`` series, which are normalized
        jointly so the stacked total integrates to 1 (matching matplotlib's
        ``hist(stacked=True, density=True)``).
        """
        return self._as_density_plot

    @as_density_plot.setter
    @deferred_axis_update
    def as_density_plot(self, value: bool) -> None:
        self._as_density_plot = bool(value)

    @deferred_axis_update
    def center_bars(self) -> None:
        """Center histogram data on bin edges."""
        self._bin_alignment = "center"

    @deferred_axis_update
    def set_bins(self, bins: BinsType | None) -> None:
        """Set the bins for the histogram.

        Bins can be specified either as an array of bin edges, an integer number of bins,
        or a string (e.g., 'auto') to use numpy's histogram binning strategies. The binning
        strategies possible strategies are as follows:

        - ‘auto’: Minimum bin width between the ‘sturges’ and ‘fd’ estimators. Provides good
            all-around performance.
        - ‘fd’ (Freedman Diaconis Estimator): Robust (resilient to outliers) estimator that
            takes into account data variability and data size.
        - ‘doane’: An improved version of Sturges’ estimator that works better with non-normal
            datasets.
        - ‘scott’: Less robust estimator that takes into account data variability and data size.
        - ‘stone’: Estimator based on leave-one-out cross-validation estimate of the integrated
            squared error. Can be regarded as a generalization of Scott’s rule.
        - ‘rice’: Estimator does not take variability into account, only data size. Commonly
            overestimates number of bins required.
        - ‘sturges’: R’s default method, only accounts for data size. Only optimal for gaussian
            data and underestimates number of bins for large non-gaussian datasets.
        - ‘sqrt’: Square root (of data size) estimator, used by Excel and other programs for its
            speed and simplicity.

        Clears any bin width previously set via :meth:`set_bin_widths`,
        mirroring how :meth:`set_bin_widths` clears explicit bins.

        Args:
            bins (BinsType): Bin specification (array of bin edges, integer number of bins, or
                string specifying binning strategy). If None, defaults to 'auto'.

        Returns:
            None
        """
        self._bins = (
            bins
            if bins is None or isinstance(bins, (int, str))
            else np.asarray(bins, dtype=float).copy()
        )
        self._binwidth = None

    @deferred_axis_update
    def set_bin_widths(self, binwidth: float | None) -> None:
        """Set histogram bins by specifying a fixed bin width.

        The histogram bins will be computed automatically based on the minimum and maximum
        values across all histogram sets added to the figure.

        Args:
            binwidth (float | None): Desired width of each histogram bin. If None, the
                histogram bins will be computed automatically using numpy's default binning
                strategy ('auto').

        Raises:
            ValueError: If ``binwidth`` is not a positive finite number.
        """
        if binwidth is not None:
            binwidth = float(binwidth)
            if not math.isfinite(binwidth) or binwidth <= 0.0:
                raise ValueError(f"binwidth must be a positive finite number; got {binwidth!r}.")
        self._bins = None
        self._binwidth = binwidth

    @deferred_axis_update
    def clear_histograms(self) -> None:
        """Clear all histogram sets."""
        for key in self._hist_data_dict:
            self._hist_data_dict[key].clear()

    def display_warnings(self, enabled: bool) -> None:
        """Set whether warnings about potentially problematic settings are displayed.

        Args:
            enabled (bool): Whether to display warnings.
        """
        self._show_warnings = enabled

    def as_density(self, enabled: bool = True) -> None:
        """Set whether histograms are rendered as densities.

        ``'stack'`` series normalize jointly (the stacked total integrates to 1); every
        other histtype normalizes each series on its own.

        Args:
            enabled (bool, optional): Whether to render densities. Defaults to True.
        """
        self.as_density_plot = enabled

    def add_dataset(
        self,
        values: Iterable[float] | NDArray | pd.Series | pd.DataFrame,
        name: str | None = None,
        *,
        weights: Iterable[float] | NDArray | None = None,
        column: str | None = None,
        options: HistogramOptions | None = None,
        facecolor: Color | None | Unset = UNSET,
        facealpha: float | None = None,
        edgecolor: Color | None | Unset = UNSET,
        edgealpha: float | None = None,
        edgewidth: float | None = None,
        histtype: HistType | None = None,
        zorder: int | None = None,
    ) -> None:
        """Add a histogram to the figure.

        Note that multiple histograms can be added to the figure, and they will be
        rendered according to the specified ``histtype``.

        - 'overlay': Histograms are drawn on top of each other, with no stacking.
        - 'stack': Histograms are stacked on top of each other.
        - 'grouped': Histograms are drawn side-by-side, with each histogram bar
            divided into equal-width segments for each histogram.
        - 'outline': Histograms are drawn as outlines only, with no fill. Effectively
            a skyline plot.

        Note:
            Each type of histogram ('overlay', 'stack', 'grouped', 'outline') maintains its own
            separate list of histogram data. When adding a histogram, it is added to the list
            corresponding to the specified ``histtype``. Therefore, a histogram of type 'grouped'
            is added to a plot containing both 'overlay' histograms and 'grouped' histograms,
            only the 'grouped' histograms will be drawn in the 'grouped' style; the 'overlay'
            histograms will be drawn in the 'overlay' style and will not interact with the 'grouped'
            histograms. Similarly, 'stack' histograms are stacked only with other 'stack'
            histograms, and the stacking order is determined by the order in which they were added.

        Note:
            When using 'outline' histograms, it is recommended to set ``edgewidth`` to
            a positive value (e.g., 0.8) to ensure the outline is visible. Additionally,
            setting ``facecolor`` to 'none' and ``edgecolor`` to a visible color (e.g., 'black')
            is advisable for clarity.


        Args:
            values (Iterable[float] | NDArray | pd.Series | pd.DataFrame): Values used to
                build the histogram.
            weights (Iterable[float] | NDArray | None, optional): Optional weights for
                the histogram values. Defaults to None.
            column (str | None, optional): The column name to use if values is a DataFrame.
            name (str | None, optional): The name of the histogram series for the legend.
                Defaults to None.
            options (HistogramOptions | None, optional): Base histogram styling. Explicit
                keyword arguments override matching fields. Defaults to None.
            facecolor (Color | None, optional): The fill color of the histogram bars. Pass
                ``None`` for no fill. Defaults to "default_grey".
            facealpha (float | None, optional): The alpha transparency of the histogram bars.
                Defaults to None.
            edgecolor (Color | None, optional): The edge color of the histogram bars. Pass
                ``None`` for no edge. Defaults to "black".
            edgealpha (float | None, optional): The alpha transparency of the histogram bar edges.
                Defaults to None.
            edgewidth (float, optional): The width of the histogram bar edges. Defaults to None
                (unset): when a visible ``edgecolor`` is given but ``edgewidth`` is left unset, it
                falls back to 0.8 so ``edgecolor`` alone produces edged bars. Pass ``edgewidth=0``
                explicitly to keep the edge hidden.
            histtype (HistType, optional): The type of histogram to add. Must be one of
                'overlay', 'stack', 'grouped', 'outline'. Defaults to 'overlay'.
            zorder (int, optional): The z-order of the histogram. Defaults to 2.

        Raises:
            ValueError: If values, weights, histogram type, or styling options are invalid.
        """
        vals, wts = _coerce_values_and_weights(values, weights=weights, column=column)

        base = options if options is not None else HistogramOptions()
        style = base.merged(
            facecolor=facecolor,
            facealpha=facealpha,
            edgecolor=edgecolor,
            edgealpha=edgealpha,
            edgewidth=edgewidth,
            histtype=histtype,
            zorder=zorder,
        )

        if _needs_default_edge_width(
            edgewidth_given=not style._edgewidth_defaulted,
            resolved_edgewidth=style.edgewidth,
            resolved_edgecolor=style.edgecolor,
        ):
            style = style.merged(edgewidth=DEFAULT_EDGE_WIDTH)

        hist_list = self._hist_data_dict.get(style.histtype, None)
        if hist_list is None:
            raise ValueError(
                f"Invalid histtype {style.histtype!r}; must be one of "
                "'overlay', 'stack', 'grouped', 'outline'."
            )

        if style.histtype == "outline":
            # The fixes apply in one merge so they see each other: bumping the edgewidth while
            # the edge color is still "none" would otherwise be clamped straight back to zero.
            outline_fixes: dict[str, object] = {}
            if style.edgewidth <= 0.0:
                if self._show_warnings:
                    warn(
                        "Outline histogram specified with edgewidth <= 0; setting edgewidth "
                        f"to {DEFAULT_EDGE_WIDTH}.",
                        UserWarning,
                    )
                outline_fixes["edgewidth"] = DEFAULT_EDGE_WIDTH
            if style.facecolor != "none":
                if self._show_warnings:
                    warn(
                        "Outline histogram specified with facecolor != 'none'; setting "
                        "facecolor to 'none'.",
                        UserWarning,
                    )
                outline_fixes["facecolor"] = None
            if style.edgecolor == "none":
                if self._show_warnings:
                    warn(
                        "Outline histogram specified with edgecolor 'none'; setting "
                        "edgecolor to 'black'.",
                        UserWarning,
                    )
                outline_fixes["edgecolor"] = "black"
            if outline_fixes:
                style = style.merged(**outline_fixes)

        histogram_number = sum(len(datasets) for datasets in self._hist_data_dict.values()) + 1
        set_name = name or f"Histogram {histogram_number}"
        hist_list.append(_HistogramData(name=set_name, values=vals, weights=wts, style=style))
        self._claim_legend_if_named(name)

    def add_points_above(
        self,
        values: float | list[float] | pd.Series | pd.DataFrame | NDArray,
        name: str | None = None,
        *,
        column: str | None = None,
        marker_options: PointMarkerOptions | None = None,
        facecolor: Color | None | Unset = UNSET,
        facealpha: float | None = None,
        marker: str | None = None,
        markersize: float | None = None,
        markeredgecolor: Color | None | Unset = UNSET,
        markeredgealpha: float | None = None,
        markeredgewidth: float | None = None,
        zorder: int | None = None,
        y_offset: float = 0.0,
        centered_on_bin: bool = False,
    ) -> None:
        """Add a set of points drawn just above the histogram bars.

        Args:
            values (float | list[float] | pd.Series | pd.DataFrame | NDArray):
                The pointset values. Can be a single value, a list of values,
                a Series, an array, or a DataFrame.
            column (str | None, optional): The column name to use if values is a DataFrame.
            name (str | None, optional): The name of the point set. Defaults to None.
            marker_options (PointMarkerOptions | None, optional): Base marker styling. Explicit
                keyword arguments override matching fields. Defaults to None.
            facecolor (Color | None, optional): The face color of the points. Pass ``None`` for no
                fill. Defaults to "black".
            facealpha (float | None, optional): The alpha transparency of the points.
                Defaults to None.
            marker (str, optional): The marker style for the points. Defaults to "o".
            markersize (float, optional): The size of the point markers. Defaults to 7.0.
            markeredgecolor (Color | None, optional): The edge color of the point markers. Pass
                ``None`` for no edge. Defaults to "black".
            markeredgealpha (float | None, optional): The alpha transparency of the pointset
                marker edges. Defaults to None.
            markeredgewidth (float, optional): The width of the point marker edges.
                Defaults to 0.8.
            zorder (int, optional): The z-order of the points. Defaults to 3.
            y_offset (float, optional): An absolute y-offset added above the bar top for
                the first point landing in each bin. Defaults to 0.0.
            centered_on_bin (bool, optional): If True, center the points on the histogram
                bins. Defaults to False.

        Returns:
            None

        Raises:
            ValueError: If ``values`` has no finite entries, or ``y_offset`` is not finite.
        """
        vals = _coerce_to_1d_finite_float_array(values, column=column, field="values")
        if vals.size == 0:
            raise ValueError("values: must have at least one finite entry.")
        y_offset = float(y_offset)
        if not math.isfinite(y_offset):
            raise ValueError(f"y_offset must be finite; got {y_offset!r}.")

        # Use a points-above default marker style: black face, larger size, edged.
        base = (
            marker_options
            if marker_options is not None
            else PointMarkerOptions(
                markerfacecolor="black",
                markersize=7.0,
                markeredgecolor="black",
                markeredgewidth=0.8,
                zorder=3,
            )
        )
        resolved_marker_options = _replace_with_color_overrides(
            base,
            ("markerfacecolor", "markerfacealpha"),
            ("markeredgecolor", "markeredgealpha"),
            markerfacecolor=facecolor,
            markerfacealpha=facealpha,
            marker=marker,
            markersize=markersize,
            markeredgecolor=markeredgecolor,
            markeredgealpha=markeredgealpha,
            markeredgewidth=markeredgewidth,
            zorder=zorder,
        )

        marker_name = name or f"Point Marker {len(self._histpointlist_list) + 1}"
        self._histpointlist_list.append(
            _HistPointList(
                name=marker_name,
                values=vals,
                point_data=resolved_marker_options,
                y_offset=y_offset,
                centered=centered_on_bin,
            )
        )
        self._claim_legend_if_named(name)

    def _compute_bins(self) -> NDArray[np.float64]:
        """Compute histogram bins based on current settings and data."""
        bins = self._bins
        if bins is None and self._binwidth is None:
            bins = "auto"

        all_values = np.concatenate(
            [hdata.values for hdata in chain(*self._hist_data_dict.values())]
        )
        if bins is None and self._binwidth is not None:
            minscore, maxscore = np.min(all_values), np.max(all_values)
            binwidth = self._binwidth
            n_bins = max(1, int(np.ceil((maxscore - minscore) / binwidth)))
            bins = minscore + np.arange(n_bins + 1) * binwidth

        if (
            bins is None
        ):  # pragma: no cover - preceding logic always assigns bins; this is an unreachable guard
            raise RuntimeError("Failed to compute histogram bins.")  # pragma: no cover

        bins = np.histogram_bin_edges(all_values, bins=bins)
        return bins

    def _display_edges(self, bin_edges: NDArray[np.float64]) -> NDArray[np.float64]:
        """Bin edges shifted to where bars are drawn.

        Centering data on bin edges is a half-binwidth left shift of the drawn edges; binning
        itself always uses the raw edges. Everything (bars, outlines, grouped offsets, points)
        derives its x-positions from these display edges, so the alignment mode lives here only.
        """
        if self._bin_alignment == "center":
            return bin_edges - 0.5 * (bin_edges[1] - bin_edges[0])
        return bin_edges

    def _draw_histograms(self, bin_edges: NDArray[np.float64]) -> NDArray[np.float64]:
        """Draw the histograms on the plot.

        Args:
            bin_edges (NDArray[np.float64]): Shared bin edges computed once per build.

        Returns:
            NDArray[np.float64]: Per-bin maximum bar-top heights across every series (stacked tops
            for ``"stack"`` series), used by ``_draw_points`` to place markers above the bars.
        """
        max_heights = np.zeros(len(bin_edges) - 1)
        bin_widths = np.diff(bin_edges)

        if self._bin_alignment == "center" and not np.allclose(bin_widths, bin_widths[0]):
            raise ValueError(
                "Cannot center histogram data on bin edges when bin widths are not uniform."
            )
        display_edges = self._display_edges(bin_edges)

        for histtype, histlist in self._hist_data_dict.items():
            hist_bottoms = np.zeros(len(bin_edges) - 1)
            n_bins_per_bar = 1
            if histtype == "grouped":
                n_bins_per_bar = len(histlist)

            # Density-mode stacks normalize jointly so the stacked total integrates to 1
            # (matching matplotlib's hist(stacked=True, density=True)); other histtypes
            # normalize each series on its own.
            stack_density = histtype == "stack" and self.as_density_plot
            stacked_counts: list[NDArray[np.float64]] = []
            stack_total_weight = 0.0
            if stack_density:
                stacked_counts = [
                    np.histogram(hdata.values, bins=bin_edges, weights=hdata.weights)[0]
                    for hdata in histlist
                ]
                stack_total_weight = float(sum(counts.sum() for counts in stacked_counts))
                if histlist and stack_total_weight == 0.0:
                    raise ValueError(
                        "Stacked histogram weights sum to zero; cannot normalize as density."
                    )

            for i, hdata in enumerate(histlist):
                if stack_density:
                    hist_heights = stacked_counts[i] / (stack_total_weight * bin_widths)
                else:
                    if self.as_density_plot and float(hdata.weights.sum()) == 0.0:
                        raise ValueError(
                            f"Histogram series {hdata.name!r} weights sum to zero; "
                            "cannot normalize as density."
                        )
                    hist_heights = np.histogram(
                        hdata.values,
                        bins=bin_edges,
                        weights=hdata.weights,
                        density=self.as_density_plot,
                    )[0]
                if histtype != "stack":
                    max_heights = np.maximum(max_heights, hist_heights)

                # Special case for outline histograms that does the outline only (no internal
                # vertical lines)
                if histtype == "outline":
                    step_patch = self._ax.stairs(
                        hist_heights,
                        display_edges,
                        fill=False,
                        linewidth=hdata.style.edgewidth,
                        edgecolor=self._resolved_rgba(
                            hdata.style.edgecolor,
                            hdata.style.edgealpha,
                            field="edgecolor",
                        ),
                        zorder=hdata.style.zorder,
                        label=hdata.name,
                    )
                    self._artists.track(step_patch)
                    continue

                hist_edges = display_edges[:-1].copy()
                if n_bins_per_bar > 1:
                    hist_edges += (bin_widths / n_bins_per_bar) * i

                bar_container = self._ax.bar(
                    hist_edges,
                    hist_heights,
                    width=bin_widths / n_bins_per_bar,
                    bottom=hist_bottoms,
                    align="edge",
                    facecolor=self._resolved_rgba(
                        hdata.style.facecolor,
                        hdata.style.facealpha,
                        field="facecolor",
                    ),
                    edgecolor=self._resolved_rgba(
                        hdata.style.edgecolor,
                        hdata.style.edgealpha,
                        field="edgecolor",
                    ),
                    linewidth=hdata.style.edgewidth,
                    zorder=hdata.style.zorder,
                )
                # BarContainer is iterable over its Rectangle patches.
                self._artists.track(bar_container)

                if histtype == "stack":
                    hist_bottoms += hist_heights
                    max_heights = np.maximum(max_heights, hist_bottoms)

        return max_heights

    def _draw_points(
        self, bin_edges: NDArray[np.float64], max_heights: NDArray[np.float64]
    ) -> None:
        """Draw the points on the histogram plot.

        Positions each point just above the tallest bar in its bin, taking marker size and edge
        width into account.

        Args:
            bin_edges (NDArray[np.float64]): Shared bin edges computed once per build.
            max_heights (NDArray[np.float64]): Per-bin maximum bar-top heights from
                ``_draw_histograms``. Mutated in place as points stack above bars.
        """
        # The bars just drawn left the view stale, and the data limits may still include a
        # previous build's (since removed) artists. Recompute both before converting
        # point-space clearances through transData, so every build places points from the
        # same realized view.
        _recompute_data_limits(self._ax)
        self._ax.autoscale_view()

        display_edges = self._display_edges(bin_edges)
        bin_centers = (display_edges[:-1] + display_edges[1:]) / 2.0

        def clearance_data_units(clearance_pt: float, y_top: float) -> float:
            """Convert a point-space clearance to y-data units at ``y_top``."""
            px = clearance_pt * self._ax.figure.dpi / 72.0
            p = self._ax.transData.transform((0.0, y_top))
            y2 = self._ax.transData.inverted().transform(p + np.array([0.0, px]))[1]
            return y2 - y_top

        out_of_range_heights: dict[int, float] = {}
        for pointlist in self._histpointlist_list:
            # The marker glyph geometry is constant per point set.
            clearance_pt = _marker_clearance_pt(
                pointlist.point_data.marker,
                pointlist.point_data.markersize,
                pointlist.point_data.markeredgewidth,
            )
            x_positions = np.array(pointlist.values)
            y_positions = []
            visited_indexes = set()
            placement_edges = bin_edges if pointlist.centered else display_edges
            for i, val in enumerate(pointlist.values):
                bin_idx = np.searchsorted(placement_edges, val, side="right") - 1
                if val == placement_edges[-1]:
                    bin_idx -= 1
                in_range = 0 <= bin_idx < len(max_heights)
                stack_idx = bin_idx if in_range else (-1 if bin_idx < 0 else len(max_heights))
                height = (
                    float(max_heights[bin_idx])
                    if in_range
                    else out_of_range_heights.get(stack_idx, 0.0)
                )
                dy = clearance_data_units(clearance_pt, height)
                offset = dy
                if stack_idx not in visited_indexes:
                    visited_indexes.add(stack_idx)
                    offset += pointlist.y_offset

                y_positions.append(height + offset)
                height += offset + dy
                if in_range:
                    max_heights[bin_idx] = height
                    if pointlist.centered:
                        x_positions[i] = bin_centers[bin_idx]
                else:
                    out_of_range_heights[stack_idx] = height

            point_lines = self._ax.plot(
                x_positions,
                y_positions,
                linestyle="",
                marker=pointlist.point_data.marker,
                markersize=pointlist.point_data.markersize,
                markerfacecolor=self._resolved_rgba(
                    pointlist.point_data.markerfacecolor,
                    pointlist.point_data.markerfacealpha,
                    field="markerfacecolor",
                ),
                markeredgecolor=self._resolved_rgba(
                    pointlist.point_data.markeredgecolor,
                    pointlist.point_data.markeredgealpha,
                    field="markeredgecolor",
                ),
                markeredgewidth=pointlist.point_data.markeredgewidth,
                zorder=pointlist.point_data.zorder,
                label=pointlist.name,
            )
            self._artists.track(point_lines)

    def _build_plot(self) -> None:
        """Build the histogram plot."""
        if sum(len(lst) for lst in self._hist_data_dict.values()) == 0:
            raise ValueError("No histogram sets added yet.")
        bin_edges = self._compute_bins()
        max_heights = self._draw_histograms(bin_edges)
        self._draw_points(bin_edges, max_heights)

    def _dataset_legend_handles(self) -> list[LegendHandle]:
        """Get legend handles for the histogram sets."""
        handles: list[LegendHandle] = []
        for hdata in chain(*self._hist_data_dict.values()):
            handles.append(
                Patch(
                    facecolor=self._resolved_rgba(
                        hdata.style.facecolor,
                        hdata.style.facealpha,
                        field="facecolor",
                    ),
                    edgecolor=self._resolved_rgba(
                        hdata.style.edgecolor,
                        hdata.style.edgealpha,
                        field="edgecolor",
                    ),
                    linewidth=hdata.style.edgewidth,
                    label=hdata.name,
                )
            )
        return handles

    def _pointset_legend_handles(self) -> list[LegendHandle]:
        """Generate legend handles for point sets.

        Returns:
            list[LegendHandle]: A list of legend handles for the point sets.
        """
        return [
            _marker_legend_handle(histpoint.point_data, histpoint.name)
            for histpoint in self._histpointlist_list
        ]
