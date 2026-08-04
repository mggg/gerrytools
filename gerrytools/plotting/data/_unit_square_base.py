"""Shared base for plots drawn in the unit square (seats-votes and paintball plots).

Owns the pieces both plot classes previously duplicated: the slope guide-line record and
storage, the standard proportionality / efficiency-gap / custom line adders, center
crosshairs, aspect handling, and the slope-line legend handles.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Any, cast

from matplotlib.axes import Axes
from matplotlib.lines import Line2D

from gerrytools._geometry import line_segment_through_unit_square
from gerrytools.colors import validate_alpha
from gerrytools.plotting._axes_backed import deferred_axis_update
from gerrytools.plotting.data.gerryplot import GerryPlotBase
from gerrytools.plotting.data.options import _CrosshairStyle
from gerrytools.plotting.utils import _validated_nonneg_finite
from gerrytools.typing import Color, LegendHandle


@dataclass(frozen=True)
class _SlopeLine:
    """A guide line through (0.5, 0.5) with the given slope, clipped to the unit square.

    Attributes:
        slope (float): The slope of the line.
        linecolor (Color | None): The color of the line. None makes it transparent.
        linewidth (float): The width of the line.
        linestyle (str): The style of the line.
        linealpha (float | None): Optional alpha override for the line color.
        zorder (int | float): The z-order used to draw the line; coerced to int.
        label (str | None): The label for the line in the legend. Defaults to None.
    """

    slope: float
    linecolor: Color | None
    linewidth: float
    linestyle: str
    linealpha: float | None = None
    zorder: int | float = -1
    label: str | None = None

    def __post_init__(self) -> None:
        slope = float(self.slope)
        if math.isnan(slope):
            raise ValueError("slope must not be NaN.")
        object.__setattr__(self, "slope", slope)

        object.__setattr__(
            self, "linewidth", _validated_nonneg_finite(self.linewidth, field="linewidth")
        )
        if self.linealpha is not None:
            object.__setattr__(self, "linealpha", validate_alpha(self.linealpha, field="linealpha"))
        object.__setattr__(self, "zorder", int(self.zorder))


@dataclass(frozen=True)
class _SlopeLineGroup:
    """One named legend record containing one or more identically styled guide lines."""

    lines: tuple[_SlopeLine, ...]

    def __iter__(self) -> Iterator[_SlopeLine]:
        return iter(self.lines)


class _UnitSquarePlotBase(GerryPlotBase):
    """Shared machinery for unit-square plots (``SeatsVotesPlot``, ``PaintballPlot``)."""

    # Default crosshair band width in data units; subclasses override.
    _crosshair_default_width: float = 0.02

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
        """Initialize the shared unit-square plot state.

        Args:
            figure_size (tuple[float, float] | None, optional): The size of the
                figure in inches. Defaults to ``(10, 10)`` when ``ax`` is not provided.
            dpi (int | None, optional): The dots per inch (DPI) of the figure.
                Defaults to ``300`` when ``ax`` is not provided.
            ax (matplotlib.axes.Axes | None, optional): Render onto an existing
                matplotlib ``Axes`` instead of creating a fresh figure. Defaults to None.
            legend (bool | None, optional): Whether to include a legend in the plot.
                ``None`` selects the class default. Defaults to None.
            xlabel (str | None, optional): The label for the x-axis. Defaults to None.
            ylabel (str | None, optional): The label for the y-axis. Defaults to None.
            title (str | None, optional): The title of the plot. Defaults to None.
        """
        # Unit-square plots prefer a square 10x10 figure; only apply this default when
        # the user hasn't otherwise specified a size or supplied their own axes.
        if figure_size is None and ax is None:
            figure_size = (10, 10)
        super().__init__(
            figure_size=figure_size,
            dpi=dpi,
            ax=ax,
            legend=legend,
            xlabel=xlabel,
            ylabel=ylabel,
            title=title,
        )

        # Named groups live in a dict so re-adding under the same name replaces the group;
        # anonymous lines accumulate in a list and never appear in the legend.
        self._named_lines: dict[str, _SlopeLineGroup] = {}
        self._lines: list[_SlopeLine] = []
        self._display_line_legend = True

        self._crosshair_style: _CrosshairStyle | None = None
        self.set_crosshair_options()

        self._aspect_ratio = 1.0

        # Unit-square plots are always drawn in [0, 1] x [0, 1].
        # Deferred axis-limit setters keep the limits across rebuilds.
        self.set_xlim(0.0, 1.0)
        self.set_ylim(0.0, 1.0)

    # ========================
    # ==  Cosmetic helpers  ==
    # ========================
    @deferred_axis_update
    def set_crosshair_options(
        self,
        *,
        x_width: float | None = None,
        y_width: float | None = None,
        color: Color | None = "lightgrey",
        alpha: float = 1.0,
    ) -> None:
        """Add crosshairs centered at (0.5, 0.5) to the plot.

        Args:
            x_width (float | None, optional): The width of the vertical crosshair band in data
                units. Defaults to 0.02 for ``SeatsVotesPlot`` and 0.007 for ``PaintballPlot``.
            y_width (float | None, optional): The width of the horizontal crosshair band in data
                units. Defaults like ``x_width``.
            color (Color | None, optional): The color of the crosshair bands. Pass
                ``None`` for transparent bands. Defaults to "lightgrey".
            alpha (float, optional): The alpha transparency of the crosshair bands. Defaults to 1.0.
        """
        self._crosshair_style = _CrosshairStyle(
            color=color,
            alpha=alpha,
            x_width=x_width if x_width is not None else self._crosshair_default_width,
            y_width=y_width if y_width is not None else self._crosshair_default_width,
        )

    @deferred_axis_update
    def remove_crosshairs(self) -> None:
        """Remove crosshairs from the plot."""
        self._crosshair_style = None

    @deferred_axis_update
    def set_aspect(self, ratio: float) -> None:
        """Set the axes aspect ratio (drawn height of one y unit over one x unit).

        Args:
            ratio (float): Aspect ratio applied at build time. ``1.0`` (the default)
                renders the unit square as a square.
        """
        ratio = float(ratio)
        if not math.isfinite(ratio):
            raise ValueError("ratio must be finite.")
        if ratio <= 0:
            raise ValueError("ratio must be positive.")
        self._aspect_ratio = ratio
        self._axes_state.reclaim_without_value("aspect")

    def _apply_aspect_now(self) -> None:
        """Apply the configured aspect ratio so the unit square renders as intended."""
        self._ax.set_aspect(self._aspect_ratio, adjustable="box")

    @deferred_axis_update
    def display_additional_lines_in_legend(self, enabled: bool) -> None:
        """Set whether named guide lines appear in the legend.

        Args:
            enabled (bool): Whether named guide lines appear in the legend.
        """
        self._display_line_legend = enabled

    # =================
    # ==  Guide lines ==
    # =================
    @deferred_axis_update
    def _add_slope_lines(
        self,
        slopes: Iterable[float],
        *,
        linecolor: Color | None,
        linealpha: float | None,
        linestyle: str,
        linewidth: float,
        zorder: int,
        label: str | None,
        claim_name: str | None,
    ) -> None:
        """Store one guide-line group; a ``claim_name`` reclaims the legend unit.

        ``label`` may carry a default legend label (e.g. "Proportionality") even when the
        caller gave no name; only an explicit user name claims legend ownership.
        """
        lines = tuple(
            _SlopeLine(
                slope=slope,
                linecolor=linecolor,
                linealpha=linealpha,
                linestyle=linestyle,
                linewidth=linewidth,
                zorder=zorder,
                label=label,
            )
            for slope in slopes
        )
        if not lines:
            return
        if label is not None:
            self._named_lines[label] = _SlopeLineGroup(lines)
        else:
            self._lines.extend(lines)
        self._claim_legend_if_named(claim_name)

    def _add_slope_line(
        self,
        slope: float,
        *,
        linecolor: Color | None,
        linealpha: float | None,
        linestyle: str,
        linewidth: float,
        zorder: int,
        label: str | None,
        claim_name: str | None,
    ) -> None:
        """Store a single guide line through the shared group path."""
        self._add_slope_lines(
            (slope,),
            linecolor=linecolor,
            linealpha=linealpha,
            linestyle=linestyle,
            linewidth=linewidth,
            zorder=zorder,
            label=label,
            claim_name=claim_name,
        )

    def add_custom_line(
        self,
        slope: float,
        *,
        linecolor: Color | None = "black",
        linealpha: float | None = None,
        linestyle: str = "-",
        linewidth: float = 1.0,
        zorder: int = -1,
        name: str | None = None,
    ) -> None:
        """Add a custom guide line with the given slope to the plot.

        The line is clipped to the unit square and constrained to pass through
        the center point (0.5, 0.5).

        Args:
            slope (float): The slope of the line.
            linecolor (Color | None, optional): The color of the line. Pass ``None``
                for a transparent line. Defaults to "black".
            linealpha (float | None, optional): The alpha transparency of the line.
                Defaults to None.
            linestyle (str, optional): The style of the line. Defaults to "-".
            linewidth (float, optional): The width of the line. Defaults to 1.0.
            zorder (int, optional): The z-order of the line. Defaults to -1.
            name (str | None, optional): The label for the line in the legend. A named line
                replaces any previous line with the same name; an unnamed line is kept out of
                the legend. Defaults to None.
        """
        self._add_slope_line(
            slope,
            linecolor=linecolor,
            linealpha=linealpha,
            linestyle=linestyle,
            linewidth=linewidth,
            zorder=zorder,
            label=name,
            claim_name=name,
        )

    def add_proportionality_line(
        self,
        *,
        linecolor: Color | None = "grey",
        linealpha: float | None = None,
        linestyle: str = "--",
        linewidth: float = 2.0,
        zorder: int = -1,
        name: str | None = None,
    ) -> None:
        """Add a proportionality line (y = x through the center) to the plot.

        Args:
            linecolor (Color | None, optional): The color of the line. Pass ``None``
                for a transparent line. Defaults to "grey".
            linealpha (float | None, optional): The alpha transparency of the line.
                Defaults to None.
            linestyle (str, optional): The style of the line. Defaults to "--".
            linewidth (float, optional): The width of the line. Defaults to 2.0.
            zorder (int, optional): The z-order of the line. Defaults to -1.
            name (str | None, optional): The legend label for the line. Defaults to
                "Proportionality".
        """
        self._add_slope_line(
            1.0,
            linecolor=linecolor,
            linealpha=linealpha,
            linestyle=linestyle,
            linewidth=linewidth,
            zorder=zorder,
            label=name if name is not None else "Proportionality",
            claim_name=name,
        )

    def add_efficiency_gap_line(
        self,
        *,
        linecolor: Color | None = "grey",
        linealpha: float | None = None,
        linestyle: str = "-",
        linewidth: float = 2.0,
        zorder: int = -1,
        name: str | None = None,
    ) -> None:
        """Add an Efficiency Gap line (y = 2x - 0.5) to the plot.

        Args:
            linecolor (Color | None, optional): The color of the line. Pass ``None``
                for a transparent line. Defaults to "grey".
            linealpha (float | None, optional): The alpha transparency of the line.
                Defaults to None.
            linestyle (str, optional): The style of the line. Defaults to "-".
            linewidth (float, optional): The width of the line. Defaults to 2.0.
            zorder (int, optional): The z-order of the line. Defaults to -1.
            name (str | None, optional): The legend label for the line. Defaults to
                "Efficiency Gap".
        """
        self._add_slope_line(
            2.0,
            linecolor=linecolor,
            linealpha=linealpha,
            linestyle=linestyle,
            linewidth=linewidth,
            zorder=zorder,
            label=name if name is not None else "Efficiency Gap",
            claim_name=name,
        )

    @deferred_axis_update
    def clear_lines(self) -> None:
        """Remove all named and anonymous guide lines from the plot."""
        self._named_lines = {}
        self._lines = []

    # =================
    # ==  Draw helpers ==
    # =================
    def _draw_crosshairs(self) -> None:
        """Draw the crosshair bands, if enabled."""
        if self._crosshair_style is not None:
            self._artists.track(self._crosshair_style.draw(self._ax))

    def _draw_slope_lines(self) -> None:
        """Draw all named and anonymous guide lines."""
        named_lines = [line for group in self._named_lines.values() for line in group]
        for line in [*named_lines, *self._lines]:
            x_start, y_start, x_end, y_end = line_segment_through_unit_square(line.slope)
            line_artists = self._ax.plot(
                [x_start, x_end],
                [y_start, y_end],
                color=self._resolved_rgba(
                    line.linecolor,
                    alpha=line.linealpha,
                    field="linecolor",
                ),
                linestyle=line.linestyle,
                linewidth=line.linewidth,
                zorder=line.zorder,
            )
            self._artists.track(line_artists)

    def _slope_line_legend_handles(self) -> list[LegendHandle]:
        """Generate legend handles for the named guide lines.

        Returns:
            list[LegendHandle]: A list of legend handles for the named guide lines.
        """
        handles: list[LegendHandle] = [
            Line2D(
                [0],
                [0],
                linestyle=cast("Any", line.linestyle),
                marker="",
                label=line.label,
                color=self._resolved_rgba(
                    line.linecolor,
                    alpha=line.linealpha,
                    field="linecolor",
                ),
                linewidth=line.linewidth,
            )
            for group in self._named_lines.values()
            for line in group.lines[:1]
        ]
        return handles

    @property
    def _legend_handles(self) -> list[LegendHandle]:
        """Generated legend handles: datasets, point sets, named guide lines, then named
        annotation lines and bands."""
        handles = super()._legend_handles
        if self._display_line_legend:
            # Insert slope-line handles right after the dataset/point-set handles so plots
            # without named annotations keep the previous ordering exactly.
            n_data_handles = len(self._dataset_legend_handles()) + len(
                self._pointset_legend_handles()
            )
            handles[n_data_handles:n_data_handles] = self._slope_line_legend_handles()
        return handles
