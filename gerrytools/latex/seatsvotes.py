from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Literal, Sequence

import numpy as np

from gerrytools._election_math import overall_election_point, seats_votes_curve_values
from gerrytools._geometry import line_segment_through_unit_square
from gerrytools.latex._text import latex_escape
from gerrytools.latex._tikz_plot_base import (
    OptionValidator,
    _GuideLine,
    _TikzPlotBase,
    _to_tikz_linestyle,
    _ValidatedOptions,
    nonnegative_float_option,
    ordered_limits_option,
    passthrough_option,
    positive_float_option,
    unit_interval_option,
)
from gerrytools.typing import UNSET, Color, Unset


@dataclass(slots=True, frozen=True)
class _SeatsVotesData:
    """Container for one seats-votes series and its marker metadata.

    Attributes:
        pov_party_vote_counts (np.ndarray): Per-district party-of-interest vote totals.
        total_vote_counts (np.ndarray): Per-district total vote totals.
        name (str): Legend label for the seats-votes curve.
        linecolor (Color | None): Color for the step curve. None removes the line.
        markercolor (Color | None): Color for the election-result marker. None removes it.
        marker_label (str): Legend label for the marker.
    """

    pov_party_vote_counts: np.ndarray
    total_vote_counts: np.ndarray
    name: str
    linecolor: Color | None
    markercolor: Color | None
    marker_label: str

    def seats_votes_curve_values(self) -> tuple[list[float], list[float]]:
        """Compute standard uniform-swing seats-votes step-curve positions.

        Returns:
            tuple[list[float], list[float]]: Vote-share breakpoints and seat-share breakpoints.

        Raises:
            ValueError: If vote arrays do not align or contain nonpositive totals.
        """
        return seats_votes_curve_values(self.pov_party_vote_counts, self.total_vote_counts)


_SVPlotLine = _GuideLine


@dataclass(slots=True)
class SeatsVotesOptions(_ValidatedOptions):
    """Configuration for LaTeX seats-votes rendering.

    Attributes:
        crosshair_x_width (float): Half-width of the horizontal crosshair gap. Defaults to 0.02.
        crosshair_y_width (float): Half-width of the vertical crosshair gap. Defaults to 0.02.
        crosshair_color (Color | None): Crosshair color. None hides it. Defaults to
            ``"lightgrey"``.
        crosshair_alpha (float): Crosshair opacity in ``[0, 1]``. Defaults to 1.0.
        xlim (tuple[float, float]): Ordered vote-share limits. Defaults to ``(0, 1)``.
        ylim (tuple[float, float]): Ordered seat-share limits. Defaults to ``(0, 1)``.
        xscale (float): Plot width in TikZ units. Defaults to 10.0.
        yscale (float): Plot height in TikZ units. Defaults to 10.0.
        linewidth (float): Curve width in points. Defaults to 1.5.
        markersize (float): Election marker size in points. Defaults to 8.0.
        fontsize (float): Axis-label font size in points. Defaults to 16.0.
        legend_fontsize (float): Legend font size in points. Defaults to 16.0.
    """

    crosshair_x_width: float = 0.02
    crosshair_y_width: float = 0.02
    crosshair_color: Color | None = "lightgrey"
    crosshair_alpha: float = 1.0
    xlim: tuple[float, float] = (0.0, 1.0)
    ylim: tuple[float, float] = (0.0, 1.0)
    xscale: float = 10.0
    yscale: float = 10.0
    linewidth: float = 1.5
    markersize: float = 8.0
    fontsize: float = 16.0
    legend_fontsize: float = 16.0

    _VALIDATORS: ClassVar[dict[str, OptionValidator]] = {
        "crosshair_x_width": nonnegative_float_option("crosshair_x_width"),
        "crosshair_y_width": nonnegative_float_option("crosshair_y_width"),
        "crosshair_color": passthrough_option,
        "crosshair_alpha": unit_interval_option("crosshair_alpha"),
        "xlim": ordered_limits_option("xlim"),
        "ylim": ordered_limits_option("ylim"),
        "xscale": positive_float_option("xscale"),
        "yscale": positive_float_option("yscale"),
        "linewidth": nonnegative_float_option("linewidth"),
        "markersize": nonnegative_float_option("markersize"),
        "fontsize": nonnegative_float_option("fontsize"),
        "legend_fontsize": nonnegative_float_option("legend_fontsize"),
    }


class TikzSeatsVotesPlot(_TikzPlotBase):
    """Generate seats-votes plots as TikZ/LaTeX."""

    _options_cls = SeatsVotesOptions
    options: SeatsVotesOptions

    def __init__(
        self,
        *,
        legend: bool = False,
        xlabel: str | None = None,
        ylabel: str | None = None,
        title: str | None = None,
    ) -> None:
        """Initialize a LaTeX seats-votes plot.

        The plot renders at the options' ``xscale``/``yscale`` (10 by 10 TikZ units by default);
        use :meth:`set_scale` to change the drawn size.

        Args:
            legend (bool, optional): Whether to include a legend. Defaults to False.
            xlabel (str | None, optional): X-axis label text. Defaults to None.
            ylabel (str | None, optional): Y-axis label text. Defaults to None.
            title (str | None, optional): Plot title text. Defaults to None.
        """
        super().__init__()

        self.legend = legend
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.title = title

        self._sv_data_list: list[_SeatsVotesData] = []
        self._line_data_list: list[_SVPlotLine] = []

        self._show_crosshairs = True

        self._display_election_markers = True
        self.standard_marker_color: Color | None = "#daa520"
        self.standard_election_color: Color | None = "#006400"
        self._display_line_legend = True

    def clear_options(self) -> None:
        """Reset seats-votes options to defaults and restore the default crosshairs."""
        super().clear_options()
        self._show_crosshairs = True
        self._display_election_markers = True
        self._display_line_legend = True

    # ====================
    #   FEATURE ADDITION
    # ====================
    def add_election(
        self,
        target_party_vote_shares: Sequence[int | float],
        total_votes: Sequence[int | float] | None = None,
        *,
        name: str | None = None,
        linecolor: Color | None | Unset = UNSET,
        markercolor: Color | None | Unset = UNSET,
        marker_label: str | None = None,
    ) -> None:
        """Add a seats-votes curve to the plot.

        Args:
            target_party_vote_shares (Sequence[int | float]): Per-district vote totals or vote
                shares for the party of interest. If ``total_votes`` is None, these are interpreted
                as vote shares and must be in [0, 1].
            total_votes (Sequence[int | float] | None, optional): Per-district total vote
                totals. If None, all totals are treated as 1.0 and
                ``target_party_vote_shares`` is interpreted as vote shares.
                Defaults to None.
            name (str | None, optional): Legend label for the seats-votes curve.
                Defaults to None.
            linecolor (Color | None | Unset, optional): Curve color. None removes the line;
                omission uses ``self.standard_election_color``. Defaults to UNSET.
            markercolor (Color | None | Unset, optional): Election-result marker color. None
                removes the marker; omission uses ``self.standard_marker_color``. Defaults to
                UNSET.
            marker_label (str | None, optional): Legend label for election-result markers.
                Defaults to None.

        Raises:
            ValueError: If vote arrays are invalid or a share lies outside ``[0, 1]``.
        """
        if total_votes is None:
            if any(v < 0 or v > 1 for v in target_party_vote_shares):
                raise ValueError(
                    "If total_votes is not provided, then target_party_vote_shares must be "
                    "vote shares in [0, 1]."
                )
            total_votes = [1.0] * len(target_party_vote_shares)

        pov_counts = np.array(target_party_vote_shares, dtype=float)
        total_counts = np.array(total_votes, dtype=float)
        # Validate the series now so mistakes raise at add time rather than at render.
        seats_votes_curve_values(pov_counts, total_counts)

        self._sv_data_list.append(
            _SeatsVotesData(
                pov_party_vote_counts=pov_counts,
                total_vote_counts=total_counts,
                name=name if name is not None else "Election Seats-Votes Curve",
                linecolor=(
                    self.standard_election_color if isinstance(linecolor, Unset) else linecolor
                ),
                markercolor=(
                    self.standard_marker_color if isinstance(markercolor, Unset) else markercolor
                ),
                marker_label=marker_label if marker_label is not None else "Election Result",
            )
        )

    # ========================
    # ==  Cosmetic helpers  ==
    # ========================
    def update_crosshair_settings(
        self,
        *,
        x_width: float = 0.02,
        y_width: float = 0.02,
        color: Color | None = "lightgrey",
        alpha: float = 1.0,
    ) -> None:
        """Configure centered crosshair bands.

        Args:
            x_width (float, optional): Horizontal band width around ``x=0.5``.
                Defaults to ``0.02``.
            y_width (float, optional): Vertical band width around ``y=0.5``.
                Defaults to ``0.02``.
            color (Color | None, optional): Crosshair fill color. Pass None for transparent
                crosshairs. Defaults to ``"lightgrey"``.
            alpha (float, optional): Crosshair fill opacity in ``[0, 1]``.
                Defaults to ``1.0``.

        Returns:
            None
        """
        self.options.crosshair_x_width = x_width
        self.options.crosshair_y_width = y_width
        self.options.crosshair_color = color
        self.options.crosshair_alpha = alpha
        self._show_crosshairs = True

    def remove_crosshairs(self) -> None:
        """Remove crosshairs from the plot."""
        self._show_crosshairs = False

    def display_election_markers(self, enabled: bool) -> None:
        """Set whether overall election-result markers are displayed.

        Args:
            enabled (bool): Whether to display the markers.
        """
        self._display_election_markers = enabled

    def display_additional_lines_in_legend(self, enabled: bool) -> None:
        """Set whether additional guide lines appear in the legend.

        Args:
            enabled (bool): Whether to include guide lines in the legend.
        """
        self._display_line_legend = enabled

    def add_proportionality_line(
        self,
        *,
        linecolor: Color | None = "gray",
        linestyle: str = "dashed",
        linewidth: float = 1.0,
        name: str | None = None,
    ) -> None:
        """Add a proportionality line (y=x) to the plot.

        Args:
            linecolor (Color | None, optional): Line color. Pass None for a transparent
                line. Defaults to "gray".
            linestyle (str, optional): Line style (Matplotlib token or TikZ style).
                Defaults to "dashed".
            linewidth (float, optional): Line width. Defaults to 1.0.
            name (str | None, optional): Legend label. Defaults to "Proportionality".
        """
        self._line_data_list.append(
            _SVPlotLine(
                slope=1.0,
                linecolor=linecolor,
                linestyle=linestyle,
                linewidth=linewidth,
                label=name if name is not None else "Proportionality",
            )
        )

    def add_efficiency_gap_line(
        self,
        *,
        linecolor: Color | None = "gray",
        linestyle: str = "solid",
        linewidth: float = 1.0,
        name: str | None = None,
    ) -> None:
        """Add an efficiency-gap line (y=2x-0.5) to the plot.

        Args:
            linecolor (Color | None, optional): Line color. Pass None for a transparent
                line. Defaults to "gray".
            linestyle (str, optional): Line style (Matplotlib token or TikZ style).
                Defaults to "solid".
            linewidth (float, optional): Line width. Defaults to 1.0.
            name (str | None, optional): Legend label. Defaults to "Efficiency Gap".
        """
        self._line_data_list.append(
            _SVPlotLine(
                slope=2.0,
                linecolor=linecolor,
                linestyle=linestyle,
                linewidth=linewidth,
                label=name if name is not None else "Efficiency Gap",
            )
        )

    def add_custom_line(
        self,
        slope: float,
        *,
        linecolor: Color | None,
        linestyle: str,
        linewidth: float,
        name: str | None = None,
    ) -> None:
        """Add a custom slope-constrained line passing through (0.5, 0.5).

        Args:
            slope (float): Line slope.
            linecolor (Color | None): Line color. None makes the line transparent.
            linestyle (str): Line style (Matplotlib token or TikZ style).
            linewidth (float): Line width.
            name (str | None, optional): Legend label; unlabeled lines are drawn but do not
                appear in the legend. Defaults to None.
        """
        self._line_data_list.append(
            _SVPlotLine(
                slope=slope,
                linecolor=linecolor,
                linestyle=linestyle,
                linewidth=linewidth,
                label=name,
            )
        )

    def set_label_fontsize(self, fontsize: float) -> None:
        """Set the font size used for the title and axis labels.

        Args:
            fontsize (float): Font size in points.

        Returns:
            None
        """
        self.options.fontsize = fontsize

    def set_fontsize(self, fontsize: float) -> None:
        """Set a unified font size for the title, axis labels, and legend text.

        Args:
            fontsize (float): Font size in points.

        Returns:
            None
        """
        self.options.fontsize = fontsize
        self.options.legend_fontsize = fontsize

    def set_markersize(self, markersize: float) -> None:
        """Set election-result marker size.

        Args:
            markersize (float): Marker size in points.

        Returns:
            None
        """
        self.options.markersize = markersize

    def set_linewidth(self, linewidth: float) -> None:
        """Set seats-votes curve line width.

        Args:
            linewidth (float): Line width in points.

        Returns:
            None
        """
        self.options.linewidth = linewidth

    # =====================
    #   STRING GENERATORS
    # =====================
    @staticmethod
    def _fontsize_command(fontsize: float) -> str:
        """Build a LaTeX fontsize command.

        Args:
            fontsize (float): Font size in points.

        Returns:
            str: LaTeX command string that sets font size and baseline skip.
        """
        baseline_skip = fontsize + 2.0
        return rf"\fontsize{{{fontsize:0.2f}}}{{{baseline_skip:0.2f}}}\selectfont "

    @staticmethod
    def _step_path(vote_shares: list[float], seat_shares: list[float]) -> str:
        """Convert step-curve vectors into a TikZ path string.

        Args:
            vote_shares (list[float]): Vote-share breakpoints.
            seat_shares (list[float]): Seat-share breakpoints.

        Returns:
            str: TikZ path expression.

        Raises:
            ValueError: If vectors are empty or lengths differ.
        """
        if len(vote_shares) != len(seat_shares):
            raise ValueError("vote_shares and seat_shares must have same length.")
        if len(vote_shares) == 0:
            raise ValueError("vote_shares and seat_shares must not be empty.")

        path_points: list[tuple[float, float]] = [(vote_shares[0], seat_shares[0])]
        for i in range(1, len(vote_shares)):
            path_points.append((vote_shares[i - 1], seat_shares[i]))
            path_points.append((vote_shares[i], seat_shares[i]))

        return " -- ".join(f"({x:0.4f}, {y:0.4f})" for x, y in path_points)

    def _curve_legend_entries(self) -> list[tuple[Color | None, str]]:
        """Collect unique seats-votes curve legend entries.

        Returns:
            list[tuple[Color | None, str]]: Unique ``(linecolor, name)`` pairs in insertion
                order.
        """
        unique_pairs = dict.fromkeys((sdata.linecolor, sdata.name) for sdata in self._sv_data_list)
        return list(unique_pairs.keys())

    def _marker_legend_entries(self) -> list[tuple[Color | None, str]]:
        """Collect unique election-marker legend entries.

        Returns:
            list[tuple[Color | None, str]]: Unique ``(markercolor, marker_label)`` pairs in
                insertion order.
        """
        unique_pairs = dict.fromkeys(
            (sdata.markercolor, sdata.marker_label) for sdata in self._sv_data_list
        )
        return list(unique_pairs.keys())

    def _line_legend_entries(self) -> list[tuple[Color | None, str, str]]:
        """Collect legend entries for additional guide lines.

        Returns:
            list[tuple[Color | None, str, str]]: ``(linecolor, linestyle, label)`` entries for
                lines with non-None labels.
        """
        entries: list[tuple[Color | None, str, str]] = []
        for line in self._line_data_list:
            if line.label is None:
                continue
            entries.append((line.linecolor, line.linestyle, line.label))
        return entries

    def _add_labels(self, lines: list[str]) -> None:
        """Append title/xlabel/ylabel TikZ nodes to the output line list.

        Args:
            lines (list[str]): Mutable list of TikZ lines under construction.

        Returns:
            None
        """
        if self.title is None and self.xlabel is None and self.ylabel is None:
            return

        font_cmd = self._fontsize_command(self.options.fontsize)
        xmid = (self.options.xlim[0] + self.options.xlim[1]) / 2
        ymid = (self.options.ylim[0] + self.options.ylim[1]) / 2
        width = self.options.xlim[1] - self.options.xlim[0]
        height = self.options.ylim[1] - self.options.ylim[0]

        lines.append(
            f"\\begin{{scope}}[xscale={self.options.xscale}, yscale={self.options.yscale}]"
        )
        if self.title is not None:
            lines.append(
                rf"\node [anchor=south] at ({xmid:0.4f}, {self.options.ylim[1] + 0.03 * height:0.4f}) "
                rf"{{{font_cmd}{latex_escape(self.title)}}};"
            )
        if self.xlabel is not None:
            lines.append(
                rf"\node [anchor=north] at ({xmid:0.4f}, {self.options.ylim[0] - 0.03 * height:0.4f}) "
                rf"{{{font_cmd}{latex_escape(self.xlabel)}}};"
            )
        if self.ylabel is not None:
            lines.append(
                rf"\node [anchor=south, rotate=90] at ({self.options.xlim[0] - 0.03 * width:0.4f}, {ymid:0.4f}) "
                rf"{{{font_cmd}{latex_escape(self.ylabel)}}};"
            )
        lines.append(r"\end{scope}")

    def _add_legend(self, lines: list[str]) -> None:
        """Append legend glyph and text nodes to the output line list.

        Args:
            lines (list[str]): Mutable list of TikZ lines under construction.

        Returns:
            None
        """
        if not self.legend:
            return

        legend_rows: list[tuple[Literal["line", "marker"], Color | None, str, str]] = []
        legend_rows.extend(
            ("line", color, "solid", label) for color, label in self._curve_legend_entries()
        )
        if self._display_election_markers:
            legend_rows.extend(
                ("marker", color, "solid", label) for color, label in self._marker_legend_entries()
            )
        if self._display_line_legend:
            legend_rows.extend(
                ("line", color, linestyle, label)
                for color, linestyle, label in self._line_legend_entries()
            )

        if len(legend_rows) == 0:
            return

        legend_font_cmd = self._fontsize_command(self.options.legend_fontsize)

        # Scale legend geometry by the axis spans, like _add_labels, so non-unit limits keep
        # the legend proportioned and adjacent to the plot.
        width = self.options.xlim[1] - self.options.xlim[0]
        height = self.options.ylim[1] - self.options.ylim[0]
        x_start = self.options.xlim[1] + 0.03 * width
        y_step = 0.06 * height
        y_center = (self.options.ylim[0] + self.options.ylim[1]) / 2
        y_start = y_center + 0.5 * (len(legend_rows) - 1) * y_step
        line_length = 0.06 * width
        label_offset = 0.08 * width

        lines.append(
            f"\\begin{{scope}}[xscale={self.options.xscale}, yscale={self.options.yscale}]"
        )
        for idx, (row_type, color, linestyle, label) in enumerate(legend_rows):
            y_pos = y_start - idx * y_step
            color_token = self._to_latex_color(color)

            if row_type == "line":
                tikz_style = _to_tikz_linestyle(linestyle)
                lines.append(
                    self._draw_path_command(
                        path=(
                            rf"({x_start:0.4f}, {y_pos:0.4f}) -- "
                            rf"({x_start + line_length:0.4f}, {y_pos:0.4f})"
                        ),
                        color=color_token,
                        linewidth=1.2,
                        linestyle=tikz_style,
                    )
                )
            else:
                lines.append(
                    self._marker_node_command(
                        x=x_start + line_length / 2,
                        y=y_pos,
                        color=color_token,
                        size_pt=self.options.markersize,
                    )
                )

            lines.append(
                rf"\node [anchor=west] at ({x_start + label_offset:0.4f}, {y_pos:0.4f}) "
                rf"{{{legend_font_cmd}{latex_escape(label)}}};"
            )

        lines.append(r"\end{scope}")

    def _generate_latex(self) -> str:
        """Generate complete LaTeX/TikZ seats-votes plot content.

        Returns:
            str: Complete TikZ picture source for the configured plot.
        """
        lines = [r"\begin{tikzpicture}"]
        lines.append(
            f"\\begin{{scope}}[xscale={self.options.xscale}, yscale={self.options.yscale}]"
        )
        lines.append(
            rf"\clip [draw] ({self.options.xlim[0]:0.4f}, {self.options.ylim[0]:0.4f}) "
            rf"rectangle ({self.options.xlim[1]:0.4f}, {self.options.ylim[1]:0.4f});"
        )
        lines.append("")

        if self._show_crosshairs:
            crosshair_color = self._to_latex_color(self.options.crosshair_color)
            dx = self.options.crosshair_x_width / 2
            dy = self.options.crosshair_y_width / 2

            lines.append(
                self._fill_rectangle_command(
                    xmin=0.5 - dx,
                    ymin=self.options.ylim[0],
                    xmax=0.5 + dx,
                    ymax=self.options.ylim[1],
                    color=crosshair_color,
                    fill_opacity=self.options.crosshair_alpha,
                )
            )
            lines.append(
                self._fill_rectangle_command(
                    xmin=self.options.xlim[0],
                    ymin=0.5 - dy,
                    xmax=self.options.xlim[1],
                    ymax=0.5 + dy,
                    color=crosshair_color,
                    fill_opacity=self.options.crosshair_alpha,
                )
            )
            lines.append("")

        for line in self._line_data_list:
            x_start, y_start, x_end, y_end = line_segment_through_unit_square(
                line.slope, round_to=4
            )
            line_color = self._to_latex_color(line.linecolor)
            tikz_style = _to_tikz_linestyle(line.linestyle)
            lines.append(
                self._draw_path_command(
                    path=rf"({x_start:0.4f}, {y_start:0.4f}) -- ({x_end:0.4f}, {y_end:0.4f})",
                    color=line_color,
                    linewidth=line.linewidth,
                    linestyle=tikz_style,
                )
            )

        if len(self._line_data_list) > 0:
            lines.append("")

        for sv_series in self._sv_data_list:
            vote_shares, seat_shares = sv_series.seats_votes_curve_values()
            curve_color = self._to_latex_color(sv_series.linecolor)
            curve_path = self._step_path(vote_shares, seat_shares)
            lines.append(
                self._draw_path_command(
                    path=curve_path,
                    color=curve_color,
                    linewidth=self.options.linewidth,
                )
            )

        if len(self._sv_data_list) > 0:
            lines.append("")

        if self._display_election_markers:
            for sv_series in self._sv_data_list:
                marker_color = self._to_latex_color(sv_series.markercolor)
                total_vote_share, total_seat_share = overall_election_point(
                    sv_series.pov_party_vote_counts, sv_series.total_vote_counts
                )

                lines.append(
                    self._marker_node_command(
                        x=total_vote_share,
                        y=total_seat_share,
                        color=marker_color,
                        size_pt=self.options.markersize,
                    )
                )

        lines.append(r"\end{scope}")

        self._add_labels(lines)
        self._add_legend(lines)

        lines.append(r"\end{tikzpicture}")
        lines.append("")
        return "\n".join(lines)
