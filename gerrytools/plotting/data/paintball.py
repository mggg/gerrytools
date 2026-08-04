from __future__ import annotations

from typing import Iterable

from matplotlib.axes import Axes
from matplotlib.patches import Patch

from gerrytools._election_math import (
    horizontal_hull_vertices,
    normalize_paintball_data,
    paintball_coordinates,
)
from gerrytools.plotting._axes_backed import deferred_axis_update
from gerrytools.plotting.data._unit_square_base import _UnitSquarePlotBase
from gerrytools.plotting.data.options import _needs_default_edge_width, _PaintballHullStyle
from gerrytools.plotting.mpl.marker_options import PointMarkerOptions, _marker_legend_handle
from gerrytools.plotting.utils import UNSET, Unset, _replace_with_color_overrides
from gerrytools.typing import Color, LegendHandle


class PaintballPlot(_UnitSquarePlotBase):
    """A Matplotlib paintball plot with vote share on x and seat share on y."""

    _crosshair_default_width = 0.007
    _marker_default_edgewidth = 0.5
    _hull_default_edgewidth = 2.0

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
        """Initialize an empty paintball plot.

        Args:
            figure_size (tuple[float, float] | None, optional): Figure size in inches. Defaults to
                ``(10, 10)`` when ``ax`` is not provided.
            dpi (int | None, optional): Figure DPI. Defaults to ``300`` when ``ax`` is not provided.
            ax (matplotlib.axes.Axes | None, optional): Render onto an existing
                matplotlib ``Axes`` instead of creating a fresh figure. Defaults to None.
            legend (bool | None, optional): Whether to include the legend. ``None`` (the
                default) means "no opinion": the legend is omitted, but legend ownership is left
                unclaimed, so an external legend on a shared axes is left alone. An explicit
                ``True``/``False`` claims the legend unit. Defaults to None.
            xlabel (str | None, optional): X-axis label text. Defaults to None.
            ylabel (str | None, optional): Y-axis label text. Defaults to None.
            title (str | None, optional): Plot title text. Defaults to None.

        Add data and guide lines after construction::

            plot = PaintballPlot()
            plot.add_seats_votes_data(voteshares, seats)
            plot.add_efficiency_gap_line()
            plot.add_proportionality_line()
            plot.show()
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

        self._voteshare_data: list[float] = []
        self._seatshare_data: list[float] = []

        self.clear_options()

    @deferred_axis_update
    def clear_options(self) -> None:
        """Reset all display options to defaults."""
        self._marker_options = PointMarkerOptions(
            markerfacecolor="cadmiumgreen",
            markerfacealpha=0.8,
            marker="o",
            markersize=16.0,
            markeredgecolor="cadmiumgreen",
            markeredgealpha=1.0,
            markeredgewidth=0.5,
        )
        self._hull_style = _PaintballHullStyle()
        self.set_crosshair_options()
        self._aspect_ratio = 1.0
        self._draw_hull = False

        self.set_xlim(0.0, 1.0)
        self.set_ylim(0.0, 1.0)
        self.set_xticks(locations=[])
        self.set_yticks(locations=[])

    # ====================
    #   FEATURE ADDITION
    # ====================
    @deferred_axis_update
    def add_seats_votes_data(
        self,
        vote_share_data: Iterable[float],
        seats_data: Iterable[float],
        *,
        total_seats: int | None = None,
    ) -> None:
        """Add vote-share / seat-share data points to the paintball plot.

        Args:
            vote_share_data (Iterable[float]): Vote-share values to add. Every value must be
                in [0, 1].
            seats_data (Iterable[float]): Seat-share values or seat counts to add.
                If ``total_seats`` is None, values are interpreted as seat shares and must be
                in [0, 1]. If ``total_seats`` is provided, values are interpreted as seat counts
                and are normalized by dividing by ``total_seats``.
            total_seats (int | None, optional): Maximum seat count used to normalize
                ``seats_data`` to share values when provided. Defaults to None.
        """
        new_voteshare_data, new_seatshare_data = normalize_paintball_data(
            list(vote_share_data),
            list(seats_data),
            total_seats,
        )
        self._voteshare_data.extend(new_voteshare_data)
        self._seatshare_data.extend(new_seatshare_data)

    def add_lines_with_slope(
        self,
        slopes: Iterable[float],
        *,
        linecolor: Color | None = "black",
        linewidth: float = 1.0,
        linestyle: str = "-",
        linealpha: float | None = None,
        zorder: int = -1,
        name: str | None = None,
    ) -> None:
        """Add guide lines with specified slopes.

        Args:
            slopes (Iterable[float]): Slopes for lines constrained to pass through (0.5, 0.5).
            linecolor (Color | None, optional): Line color. Pass ``None`` for a
                transparent line. Defaults to "black".
            linealpha (float | None, optional): Line alpha override. Defaults to None.
            linewidth (float, optional): Line width. Defaults to 1.0.
            linestyle (str, optional): Matplotlib line style string. Defaults to "-".
            zorder (int, optional): Draw order for these lines. Defaults to -1.
            name (str | None, optional): If provided, all lines are stored as a named line for
                legend display. Defaults to None.
        """
        self._add_slope_lines(
            slopes,
            linecolor=linecolor,
            linealpha=linealpha,
            linestyle=linestyle,
            linewidth=linewidth,
            zorder=zorder,
            label=name,
            claim_name=name,
        )

    # ==================
    #   OPTION SETTERS
    # ==================
    @deferred_axis_update
    def display_hull(self, enabled: bool) -> None:
        """Set whether builds render the horizontal hull instead of the point cloud.

        Args:
            enabled (bool): Whether to display the hull.
        """
        self._draw_hull = bool(enabled)

    @deferred_axis_update
    def set_marker_options(
        self,
        size: float | None = None,
        color: Color | None | Unset = UNSET,
        alpha: float | None = None,
        edgecolor: Color | None | Unset = UNSET,
        edgewidth: float | None = None,
        edgealpha: float | None = None,
        *,
        marker: str | None = None,
    ) -> None:
        """Set marker display options.

        Args:
            size (float | None, optional): Marker size in points. Defaults to None.
            color (Color | None, optional): Marker face color. Pass ``None`` for no fill; omission
                keeps the current color.
            alpha (float | None, optional): Marker face alpha in [0, 1]. Defaults to None.
            edgecolor (Color | None, optional): Marker edge color. Pass ``None`` for no edge;
                omission keeps the current color.
            edgewidth (float | None, optional): Marker edge width. When a visible edge color is
                selected after the edge has been removed, omission restores the default width.
                Pass ``0`` explicitly to keep the edge hidden. Defaults to None.
            edgealpha (float | None, optional): Marker edge alpha in [0, 1]. Defaults to None.
            marker (str | None, optional): Matplotlib marker style string. Defaults to None.
        """
        marker_options = _replace_with_color_overrides(
            self._marker_options,
            ("markerfacecolor", "markerfacealpha"),
            ("markeredgecolor", "markeredgealpha"),
            markersize=size,
            markerfacecolor=color,
            markerfacealpha=alpha,
            markeredgecolor=edgecolor,
            markeredgewidth=edgewidth,
            markeredgealpha=edgealpha,
            marker=marker,
        )
        if _needs_default_edge_width(
            edgewidth_given=edgewidth is not None,
            resolved_edgewidth=marker_options.markeredgewidth,
            resolved_edgecolor=(
                marker_options.markeredgecolor if not isinstance(edgecolor, Unset) else None
            ),
        ):
            marker_options = _replace_with_color_overrides(
                marker_options,
                markeredgewidth=self._marker_default_edgewidth,
            )
        self._marker_options = marker_options

    @deferred_axis_update
    def set_hull_options(
        self,
        color: Color | None | Unset = UNSET,
        alpha: float | None = None,
        edgecolor: Color | None | Unset = UNSET,
        edgewidth: float | None = None,
        edgealpha: float | None = None,
    ) -> None:
        """Set horizontal-hull display options.

        Args:
            color (Color | None, optional): Hull fill color. Pass ``None`` for no fill; omission
                keeps the current setting.
            alpha (float | None, optional): Hull fill alpha in [0, 1]. Defaults to None.
            edgecolor (Color | None, optional): Hull edge color. Pass ``None`` for no edge;
                omission keeps the current setting.
            edgewidth (float | None, optional): Hull edge width. When a visible edge color is
                selected after the edge has been removed, omission restores the default width.
                Pass ``0`` explicitly to keep the edge hidden. Defaults to None.
            edgealpha (float | None, optional): Hull edge alpha in [0, 1]. Defaults to None.
        """
        hull_style = _replace_with_color_overrides(
            self._hull_style,
            ("facecolor", "facealpha"),
            ("edgecolor", "edgealpha"),
            facecolor=color,
            facealpha=alpha,
            edgecolor=edgecolor,
            edgewidth=edgewidth,
            edgealpha=edgealpha,
        )
        if _needs_default_edge_width(
            edgewidth_given=edgewidth is not None,
            resolved_edgewidth=hull_style.edgewidth,
            resolved_edgecolor=hull_style.edgecolor if not isinstance(edgecolor, Unset) else None,
        ):
            hull_style = _replace_with_color_overrides(
                hull_style,
                edgewidth=self._hull_default_edgewidth,
            )
        self._hull_style = hull_style

    # =================
    #   DRAW HELPERS
    # =================
    def _paintball_coordinates(self) -> tuple[list[float], list[float]]:
        """Return direct ``(vote share, seat share)`` coordinates."""
        return paintball_coordinates(self._voteshare_data, self._seatshare_data)

    def _horizontal_hull_vertices(self) -> list[tuple[float, float]]:
        """Compute the horizontal hull vertices for the paintball points."""
        x_coordinates, y_coordinates = self._paintball_coordinates()
        return horizontal_hull_vertices(zip(x_coordinates, y_coordinates))

    def _draw_points(self) -> None:
        """Draw paintball points."""
        x_coords, y_coords = self._paintball_coordinates()
        marker = self._marker_options
        point_artists = self._ax.plot(
            x_coords,
            y_coords,
            linestyle="none",
            marker=marker.marker,
            markersize=marker.markersize,
            markerfacecolor=self._resolved_rgba(
                marker.markerfacecolor, marker.markerfacealpha, field="markerfacecolor"
            ),
            markeredgecolor=self._resolved_rgba(
                marker.markeredgecolor, marker.markeredgealpha, field="markeredgecolor"
            ),
            markeredgewidth=marker.markeredgewidth,
            zorder=2,
        )
        self._artists.track(point_artists)

    def _resolved_hull_colors(
        self,
    ) -> tuple[tuple[float, float, float, float], tuple[float, float, float, float]]:
        """Resolve hull face and edge RGBA, inheriting unset colors from the marker style."""
        hull = self._hull_style
        marker = self._marker_options
        fillcolor = hull.facecolor if hull.facecolor is not None else marker.markerfacecolor
        fillalpha = hull.facealpha if hull.facealpha is not None else marker.markerfacealpha
        edgecolor = hull.edgecolor if hull.edgecolor is not None else marker.markeredgecolor
        edgealpha = hull.edgealpha if hull.edgealpha is not None else marker.markeredgealpha
        return (
            self._resolved_rgba(fillcolor, fillalpha, field="hullcolor"),
            self._resolved_rgba(edgecolor, edgealpha, field="hulledgecolor"),
        )

    def _draw_horizontal_hull(self) -> None:
        """Draw the horizontal hull polygon for paintball points."""
        hull_vertices = self._horizontal_hull_vertices()
        fill_rgba, edge_rgba = self._resolved_hull_colors()

        if len(hull_vertices) < 3:
            xs, ys = zip(*hull_vertices)
            hull_line_artists = self._ax.plot(
                xs,
                ys,
                color=edge_rgba,
                linewidth=self._hull_style.edgewidth,
                zorder=2,
            )
            self._artists.track(hull_line_artists)
            return

        x_coords = [x for x, _ in hull_vertices] + [hull_vertices[0][0]]
        y_coords = [y for _, y in hull_vertices] + [hull_vertices[0][1]]

        # ``ax.fill`` returns a list of Polygon patches.
        hull_polygons = self._ax.fill(
            x_coords,
            y_coords,
            facecolor=fill_rgba,
            edgecolor=edge_rgba,
            linewidth=self._hull_style.edgewidth,
            zorder=2,
        )
        self._artists.track(hull_polygons)

    def _build_plot(self) -> None:
        """Build the plot by drawing all elements in order."""
        if len(self._voteshare_data) == 0:
            raise ValueError("No paintball data added yet.")
        self._draw_crosshairs()
        self._draw_slope_lines()

        if self._draw_hull:
            self._draw_horizontal_hull()
        else:
            self._draw_points()

    def _dataset_legend_handles(self) -> list[LegendHandle]:
        """Legend handle for the paintball data: the hull patch or the point-cloud marker."""
        if self._draw_hull:
            fill_rgba, edge_rgba = self._resolved_hull_colors()
            return [
                Patch(
                    facecolor=fill_rgba,
                    edgecolor=edge_rgba,
                    linewidth=self._hull_style.edgewidth,
                    label="Horizontal Hull",
                )
            ]
        return [_marker_legend_handle(self._marker_options, "Plan Outcomes")]
