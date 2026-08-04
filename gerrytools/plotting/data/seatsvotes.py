from dataclasses import dataclass
from typing import Any, Sequence, cast

import numpy as np
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from numpy.typing import NDArray

from gerrytools._election_math import overall_election_point, seats_votes_curve_values
from gerrytools.plotting._axes_backed import deferred_axis_update
from gerrytools.plotting.data._unit_square_base import _UnitSquarePlotBase
from gerrytools.plotting.data.options import (
    DEFAULT_EDGE_WIDTH,
    SeatsVotesLineOptions,
    SeatsVotesMarkerOptions,
    _needs_default_edge_width,
)
from gerrytools.plotting.utils import (
    UNSET,
    Unset,
    _replace_non_none,
    _replace_with_color_overrides,
    _validated_nonneg_finite,
)
from gerrytools.typing import Color, LegendHandle


@dataclass(slots=True, frozen=True)
class _SeatsVotesData:
    """One seats-votes curve + optional overall marker.

    Styling is stored as the two validated options dataclasses; colors are already
    resolved (plot-level standard colors substituted for ``None``) at add time.

    Attributes:
        pov_party_vote_counts (NDArray): An array of vote counts for the party of interest in
            each district.
        total_vote_counts (NDArray): An array of total vote counts in each district. Must be the
            same shape as pov_party_vote_counts.
        name (str): The name of the election/series, used for labeling the seats-votes curve in
            the legend.
        line_style (SeatsVotesLineOptions): Styling for the seats-votes curve. ``None`` width
            inherits the plot-level default.
        marker_style (SeatsVotesMarkerOptions): Styling for the election-result marker. ``None``
            size inherits the plot-level default; an unset edge color falls back to the marker
            face at render time.
        marker_label (str): The label for the marker in the legend.
    """

    pov_party_vote_counts: NDArray
    total_vote_counts: NDArray
    name: str
    line_style: SeatsVotesLineOptions
    marker_style: SeatsVotesMarkerOptions
    marker_label: str

    def resolved_linewidth(self, default_linewidth: float) -> float:
        """Return per-series curve width, falling back to plot default.

        Args:
            default_linewidth (float): Plot-level default line width.

        Returns:
            float: Effective curve width for this series.
        """
        linewidth = self.line_style.linewidth
        return float(default_linewidth) if linewidth is None else float(linewidth)

    def resolved_markersize(self, default_markersize: float) -> float:
        """Return per-series marker size, falling back to plot default.

        Args:
            default_markersize (float): Plot-level default marker size.

        Returns:
            float: Effective marker size for this series.
        """
        markersize = self.marker_style.markersize
        return float(default_markersize) if markersize is None else float(markersize)

    def resolved_markeredgecolor(self) -> Color | None:
        """Return marker edge color, defaulting to marker face color."""
        edgecolor = self.marker_style.markeredgecolor
        facecolor = self.marker_style.resolved_markerfacecolor()
        return facecolor if isinstance(edgecolor, Unset) else edgecolor

    def resolved_markeredgealpha(self) -> float | None:
        """Return marker edge alpha, defaulting to marker face alpha."""
        edgealpha = self.marker_style.markeredgealpha
        return self.marker_style.markerfacealpha if edgealpha is None else edgealpha

    def seats_votes_curve_values(self) -> tuple[list[float], list[float]]:
        """
        Compute the standard "uniform swing" seats-votes step curve positions.

        vote_share_shift_positions are the x breakpoints for the step curve.
        seat_shares_shift_positions are the y values (0..1) stepped by district rank.
        """
        return seats_votes_curve_values(self.pov_party_vote_counts, self.total_vote_counts)


class SeatsVotesPlot(_UnitSquarePlotBase):
    """A class for creating seats-votes plots."""

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
        """Initialize a SeatsVotesPlot instance.

        Args:
            figure_size (tuple[float, float] | None, optional): The size of the
                figure in inches. Defaults to ``(10, 10)`` when ``ax`` is not provided.
            dpi (int | None, optional): The dots per inch (DPI) of the figure.
                Defaults to ``300`` when ``ax`` is not provided.
            ax (matplotlib.axes.Axes | None, optional): Render onto an existing
                matplotlib ``Axes`` instead of creating a fresh figure. Defaults to None.
            legend (bool, optional): Whether to include a legend in the plot.
                Defaults to False.
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

        self._sv_data_list: list[_SeatsVotesData] = []

        self._display_election_markers = True
        self.standard_marker_color: Color | None = "#daa520"
        self.standard_election_color: Color | None = "#006400"

        self._linewidth = 2.5
        self._markersize = 8.0

        self._fontsize = 16.0
        self._legend_options.fontsize = self._fontsize
        self.set_tick_style("x", size=self._fontsize)
        self.set_tick_style("y", size=self._fontsize)

    @property
    def linewidth(self) -> float:
        """Default line width for elections without an explicit override."""
        return self._linewidth

    @linewidth.setter
    @deferred_axis_update
    def linewidth(self, value: float) -> None:
        self._linewidth = _validated_nonneg_finite(value, field="linewidth")

    @property
    def markersize(self) -> float:
        """Default marker size for elections without an explicit override."""
        return self._markersize

    @markersize.setter
    @deferred_axis_update
    def markersize(self, value: float) -> None:
        self._markersize = _validated_nonneg_finite(value, field="markersize")

    def add_election(
        self,
        target_party_vote_shares: Sequence[int | float] | NDArray,
        total_votes: Sequence[int | float] | NDArray | None = None,
        name: str | None = None,
        *,
        line_options: SeatsVotesLineOptions | None = None,
        marker_options: SeatsVotesMarkerOptions | None = None,
        linecolor: Color | None | Unset = UNSET,
        linealpha: float | None = None,
        linestyle: str | None = None,
        linewidth: float | None = None,
        zorder: int | None = None,
        markerfacecolor: Color | None | Unset = UNSET,
        markerfacealpha: float | None = None,
        marker: str | None = None,
        markersize: float | None = None,
        markeredgecolor: Color | None | Unset = UNSET,
        markeredgealpha: float | None = None,
        markeredgewidth: float | None = None,
        marker_zorder: int | None = None,
        marker_label: str | None = None,
    ) -> None:
        """Add a seats-votes curve to the plot.

        Args:
            target_party_vote_shares (Sequence[int | float] | NDArray): A sequence or array of vote
                counts or vote shares for the party of interest in each district. When
                ``total_votes`` is omitted, values must be shares between 0 and 1.
            total_votes (Sequence[int | float] | NDArray, optional): A sequence or array of total
                vote counts or shares in each district. If None, ``target_party_vote_shares`` is
                assumed to contain vote shares between 0 and 1, and total vote share is assumed to
                be 1.0 for all districts. If provided, it must have the same shape as
                ``target_party_vote_shares``. Defaults to None.
            name (str | None, optional): The name of the election/series, used for labeling the
                seats-votes curve in the legend. Defaults to None.
            line_options (SeatsVotesLineOptions | None, optional): Base curve styling. Explicit
                line keyword arguments override matching fields. Defaults to None.
            marker_options (SeatsVotesMarkerOptions | None, optional): Base result-marker styling.
                Explicit marker keyword arguments override matching fields. Defaults to None.
            linecolor (Color | None, optional): The curve color. Pass ``None`` for no line. When
                omitted, uses ``self.standard_election_color``.
            linealpha (float | None, optional): The alpha transparency for the seats-votes curve.
                Defaults to None.
            linestyle (str, optional): The line style for the seats-votes curve. Defaults to "-".
            linewidth (float | None, optional): The line width for this seats-votes curve.
                Defaults to None, which uses ``self.linewidth``.
            zorder (int, optional): The z-order of the seats-votes curve. Defaults to 1.
            markerfacecolor (Color | None, optional): Overall marker fill. Pass ``None`` for no
                fill. When omitted, uses ``self.standard_marker_color``. Markers are controlled by
                ``display_election_markers(enabled)``.
            markerfacealpha (float | None, optional): The alpha transparency of the marker face.
                Defaults to None.
            marker (str, optional): The marker style for the election-result marker.
                Defaults to "o".
            markersize (float | None, optional): Marker size for this data set.
                Defaults to None, which uses ``self.markersize``.
            markeredgecolor (Color | None, optional): Marker edge color. Pass ``None`` for no edge.
                When omitted, uses ``markerfacecolor``.
            markeredgealpha (float | None, optional): Marker edge alpha. Defaults to None, which
                uses markerfacealpha.
            markeredgewidth (float, optional): Marker edge width. Defaults to None (unset): when a
                visible ``markeredgecolor`` is given but this is left unset, it falls back to 0.8 so
                the edge is drawn. Pass ``markeredgewidth=0`` explicitly to keep the edge hidden.
            marker_zorder (int, optional): The z-order of the marker point. Defaults to 2.
            marker_label (str | None, optional): The label for the election-result marker in the
                legend. If not
                provided, then all will be assigned the default marker label "Election Result"

        Raises:
            ValueError: If any vote value is non-finite, the two vote arrays differ in length,
                or shares fall outside [0, 1] when ``total_votes`` is omitted.
        """

        # Copy so later caller mutation cannot change the plot.
        vote_share_array = np.array(target_party_vote_shares, dtype=float)
        if not np.all(np.isfinite(vote_share_array)):
            raise ValueError("target_party_vote_shares must contain only finite values.")
        if total_votes is None:
            if any(v < 0 or v > 1 for v in target_party_vote_shares):
                raise ValueError(
                    "If total_votes is not provided, then target_party_vote_shares must be "
                    "vote shares (values between 0 and 1)."
                )
            total_votes = [1.0] * len(target_party_vote_shares)
        elif len(total_votes) != len(target_party_vote_shares):
            raise ValueError(
                f"total_votes has length {len(total_votes)} but target_party_vote_shares "
                f"has length {len(target_party_vote_shares)}; they must match per district."
            )
        elif np.any(vote_share_array < 0):
            raise ValueError("target_party_vote_shares cannot contain negative values.")
        total_votes_array = np.array(total_votes, dtype=float)
        if not np.all(np.isfinite(total_votes_array)):
            raise ValueError("total_votes must contain only finite values.")
        if np.any(total_votes_array <= 0):
            raise ValueError("total_votes must be positive for all districts.")
        if np.any(vote_share_array > total_votes_array):
            raise ValueError("target_party_vote_shares cannot exceed total_votes.")
        seats_votes_curve_values(vote_share_array, total_votes_array)

        # Explicit kwargs override the options objects; unset option colors inherit the
        # plot-level standard colors.
        line_base = line_options if line_options is not None else SeatsVotesLineOptions()
        line_style = _replace_with_color_overrides(
            line_base,
            ("linecolor", "linealpha"),
            linecolor=linecolor,
            linealpha=linealpha,
            linestyle=linestyle,
            linewidth=linewidth,
            zorder=zorder,
        )
        if isinstance(line_style.linecolor, Unset):
            line_style = _replace_with_color_overrides(
                line_style,
                ("linecolor", "linealpha"),
                linecolor=self.standard_election_color,
            )

        marker_base = marker_options if marker_options is not None else SeatsVotesMarkerOptions()
        marker_style = _replace_with_color_overrides(
            marker_base,
            ("markerfacecolor", "markerfacealpha"),
            ("markeredgecolor", "markeredgealpha"),
            markerfacecolor=markerfacecolor,
            markerfacealpha=markerfacealpha,
            marker=marker,
            markersize=markersize,
            markeredgecolor=markeredgecolor,
            markeredgealpha=markeredgealpha,
            markeredgewidth=markeredgewidth,
            marker_zorder=marker_zorder,
        )
        if isinstance(marker_style.markerfacecolor, Unset):
            marker_style = _replace_with_color_overrides(
                marker_style,
                ("markerfacecolor", "markerfacealpha"),
                markerfacecolor=self.standard_marker_color,
            )
        resolved_edgecolor: Color | None = (
            marker_style.resolved_markerfacecolor()
            if isinstance(marker_style.markeredgecolor, Unset)
            else marker_style.markeredgecolor
        )
        edgecolor_was_selected = not isinstance(markeredgecolor, Unset) or (
            marker_options is not None and not isinstance(marker_options.markeredgecolor, Unset)
        )
        if _needs_default_edge_width(
            edgewidth_given=not marker_style._markeredgewidth_defaulted,
            resolved_edgewidth=marker_style.markeredgewidth,
            resolved_edgecolor=resolved_edgecolor if edgecolor_was_selected else None,
        ):
            marker_style = _replace_non_none(marker_style, markeredgewidth=DEFAULT_EDGE_WIDTH)

        self._sv_data_list.append(
            _SeatsVotesData(
                pov_party_vote_counts=vote_share_array,
                total_vote_counts=total_votes_array,
                name=name if name is not None else "Election Seats-Votes Curve",
                line_style=line_style,
                marker_style=marker_style,
                marker_label=marker_label if marker_label is not None else "Election Result",
            )
        )
        self._claim_legend_if_named(name)

    @deferred_axis_update
    def display_election_markers(self, enabled: bool) -> None:
        """Set whether overall election-result markers are displayed.

        Args:
            enabled (bool): Whether to display the markers.
        """
        self._display_election_markers = enabled

    def _draw_seats_votes_curves(self) -> None:
        """Draw the seats-votes curves on the plot."""
        for sv_series in self._sv_data_list:
            vote_shares, seat_shares = sv_series.seats_votes_curve_values()

            # ax.step returns a list[Line2D] just like ax.plot.
            step_artists = self._ax.step(
                vote_shares,
                seat_shares,
                where="pre",
                color=self._resolved_rgba(
                    sv_series.line_style.resolved_linecolor(),
                    alpha=sv_series.line_style.linealpha,
                    field="linecolor",
                ),
                linestyle=sv_series.line_style.linestyle,
                linewidth=sv_series.resolved_linewidth(self.linewidth),
                zorder=sv_series.line_style.zorder,
            )
            self._artists.track(step_artists)

    def _draw_sv_markers(self) -> None:
        """Draw the overall election result markers on the plot."""
        for sv_series in self._sv_data_list:
            total_vote_share, total_seat_share = overall_election_point(
                sv_series.pov_party_vote_counts, sv_series.total_vote_counts
            )

            marker_style = sv_series.marker_style
            marker_artists = self._ax.plot(
                total_vote_share,
                total_seat_share,
                marker=marker_style.marker,
                linestyle="",
                markerfacecolor=self._resolved_rgba(
                    marker_style.resolved_markerfacecolor(),
                    alpha=marker_style.markerfacealpha,
                    field="markerfacecolor",
                ),
                markeredgecolor=self._resolved_rgba(
                    sv_series.resolved_markeredgecolor(),
                    alpha=sv_series.resolved_markeredgealpha(),
                    field="markeredgecolor",
                ),
                markeredgewidth=marker_style.markeredgewidth,
                markersize=sv_series.resolved_markersize(self.markersize),
                zorder=marker_style.marker_zorder,
            )
            self._artists.track(marker_artists)

    def _build_plot(self) -> None:
        """Build the plot by drawing all elements in the correct order."""
        self._draw_seats_votes_curves()
        self._draw_slope_lines()

        if self._display_election_markers:
            self._draw_sv_markers()

        self._draw_crosshairs()

    def _get_sv_curve_legend_handles(self) -> list[LegendHandle]:
        """Generate legend handles for seats-votes curves.

        Returns:
            list[LegendHandle]: A list of legend handles for the seats-votes curves.
        """
        # The frozen options dataclasses hash by value, so identical style+name pairs
        # collapse to a single legend entry.
        curve_entries = dict.fromkeys(
            (sdata.line_style, sdata.name) for sdata in self._sv_data_list
        )
        handles: list[LegendHandle] = [
            Line2D(
                [0],
                [0],
                linestyle=cast("Any", line_style.linestyle),
                marker="",
                label=name,
                color=self._resolved_rgba(
                    line_style.resolved_linecolor(),
                    alpha=line_style.linealpha,
                    field="linecolor",
                ),
                linewidth=(
                    self.linewidth if line_style.linewidth is None else float(line_style.linewidth)
                ),
            )
            for line_style, name in curve_entries
        ]
        return handles

    def _get_sv_marker_legend_handles(self) -> list[LegendHandle]:
        """Generate legend handles for election-result markers.

        Returns:
            list[LegendHandle]: A list of legend handles for election-result markers.
        """
        marker_entries = dict.fromkeys(
            (sdata.marker_style, sdata.marker_label) for sdata in self._sv_data_list
        )
        handles: list[LegendHandle] = []
        for marker_style, marker_label in marker_entries:
            edgecolor = (
                marker_style.resolved_markerfacecolor()
                if isinstance(marker_style.markeredgecolor, Unset)
                else marker_style.markeredgecolor
            )
            edgealpha = (
                marker_style.markerfacealpha
                if marker_style.markeredgealpha is None
                else marker_style.markeredgealpha
            )
            handles.append(
                Line2D(
                    [0],
                    [0],
                    linestyle="none",
                    label=marker_label,
                    marker=marker_style.marker,
                    markerfacecolor=self._resolved_rgba(
                        marker_style.resolved_markerfacecolor(),
                        alpha=marker_style.markerfacealpha,
                        field="markerfacecolor",
                    ),
                    markeredgecolor=self._resolved_rgba(
                        edgecolor,
                        alpha=edgealpha,
                        field="markeredgecolor",
                    ),
                    markeredgewidth=marker_style.markeredgewidth,
                    markersize=(
                        self.markersize
                        if marker_style.markersize is None
                        else float(marker_style.markersize)
                    ),
                )
            )
        return handles

    def _dataset_legend_handles(self) -> list[LegendHandle]:
        """Legend handles for the seats-votes curves and (when shown) their markers."""
        handles = self._get_sv_curve_legend_handles()
        if self._display_election_markers:
            handles.extend(self._get_sv_marker_legend_handles())
        return handles
