from dataclasses import dataclass
from typing import ClassVar, Iterable

from gerrytools._election_math import (
    horizontal_hull_vertices,
    normalize_paintball_data,
    paintball_coordinates,
)
from gerrytools._geometry import line_segment_through_unit_square
from gerrytools.latex._tikz_plot_base import (
    UNSET,
    OptionValidator,
    Unset,
    _GuideLine,
    _TikzPlotBase,
    _ValidatedOptions,
    nonnegative_float_option,
    optional_option,
    ordered_limits_option,
    passthrough_option,
    positive_float_option,
    unit_interval_option,
)
from gerrytools.latex.document import TexDocument
from gerrytools.typing import Color

_PaintballLine = _GuideLine


@dataclass(slots=True)
class PaintballOptions(_ValidatedOptions):
    """Class for storing paintball plot options.

    Attributes:
        markersize (float): The size of the markers in points.
        markercolor (Color | None): The color of the markers. None removes the fill.
        markeralpha (float): The opacity of the markers (0.0 to 1.0).
        markeredgecolor (Color | None): The color of the marker edges. None removes the edge.
        markeredgewidth (float): The width of the marker edges in points.
        markeredgealpha (float): The opacity of the marker edges (0.0 to 1.0).
        hullcolor (Color | None | Unset): The fill color of the convex hull. An omitted value
            inherits ``markercolor``; None removes the fill.
        hullalpha (float | None): The opacity of the hull fill (0.0 to 1.0); None (the default)
            inherits ``markeralpha``.
        hulledgecolor (Color | None | Unset): The color of the hull edge. An omitted value
            inherits ``markeredgecolor``; None removes the edge.
        hulledgewidth (float | None): The width of the hull edge in points; None inherits
            ``markeredgewidth``. Defaults to 2.0 (an explicit override, not the inherited
            marker edge width).
        hulledgealpha (float | None): The opacity of the hull edge (0.0 to 1.0); None (the
            default) inherits ``markeredgealpha``.
        crosshair_color (Color | None): The color of the crosshair lines. None removes them.
        crosshair_width (float): The width of the crosshair lines.
        xlim (tuple[float, float]): The x-axis limits.
        ylim (tuple[float, float]): The y-axis limits.
        xscale (float): The x-axis scale factor.
        yscale (float): The y-axis scale factor.
    """

    markersize: float = 8
    markercolor: Color | None = "cadmiumgreen"
    markeralpha: float = 0.8
    markeredgecolor: Color | None = "cadmiumgreen"
    markeredgewidth: float = 0.5
    markeredgealpha: float = 1.0
    hullcolor: Color | None | Unset = UNSET
    hullalpha: float | None = None
    hulledgecolor: Color | None | Unset = UNSET
    hulledgewidth: float | None = 2.0
    hulledgealpha: float | None = None
    crosshair_color: Color | None = "gray!50"
    crosshair_width: float = 5.0
    xlim: tuple[float, float] = (0.0, 1.0)
    ylim: tuple[float, float] = (0, 1)
    xscale: float = 10
    yscale: float = 10

    _VALIDATORS: ClassVar[dict[str, OptionValidator]] = {
        "markersize": positive_float_option("markersize", round_to=4),
        "markercolor": passthrough_option,
        "markeralpha": unit_interval_option("markeralpha", round_to=4),
        "markeredgecolor": passthrough_option,
        "markeredgewidth": nonnegative_float_option("markeredgewidth", round_to=4),
        "markeredgealpha": unit_interval_option("markeredgealpha", round_to=4),
        "hullcolor": passthrough_option,
        "hullalpha": optional_option(unit_interval_option("hullalpha", round_to=4)),
        "hulledgecolor": passthrough_option,
        "hulledgewidth": optional_option(nonnegative_float_option("hulledgewidth", round_to=4)),
        "hulledgealpha": optional_option(unit_interval_option("hulledgealpha", round_to=4)),
        "crosshair_color": passthrough_option,
        "crosshair_width": nonnegative_float_option("crosshair_width", round_to=4),
        "xlim": ordered_limits_option("xlim", round_to=4),
        "ylim": ordered_limits_option("ylim", round_to=4),
        "xscale": positive_float_option("xscale", round_to=4),
        "yscale": positive_float_option("yscale", round_to=4),
    }


class TikzPaintballPlot(_TikzPlotBase):
    """Class for generating paintball plots in TikZ/LaTeX.

    The paintball plot places vote share on x and seat share on y in the unit square. Vote shares
    are expected in [0, 1]. Seat data is either interpreted as shares in [0, 1] or normalized from
    seat counts using ``total_seats``. Guide lines are added with
    :meth:`add_efficiency_gap_line`, :meth:`add_proportionality_line`, or
    :meth:`add_lines_with_slope`.
    """

    _options_cls = PaintballOptions
    options: PaintballOptions

    def __init__(
        self,
        vote_share_data: Iterable[float],
        seats_data: Iterable[float],
        total_seats: int | None = None,
    ) -> None:
        """Initialize a LaTeX paintball plot.

        Args:
            vote_share_data (Iterable[float]): Vote-share values for each plan outcome.
                Every value must be in [0, 1].
            seats_data (Iterable[float]): Seat-share values or seat counts for each plan outcome.
                If ``total_seats`` is None, values are interpreted as seat shares and must be
                in [0, 1]. If ``total_seats`` is provided, values are interpreted as seat counts
                and normalized by dividing by ``total_seats``.
            total_seats (int | None, optional): Maximum seat count used to normalize
                ``seats_data`` when seat counts are provided. Defaults to None.
        """
        super().__init__()
        self._hull_document = TexDocument()
        self._hull_document.add_packages("tikz")
        self._voteshare_data, self._seatshare_data = normalize_paintball_data(
            list(vote_share_data),
            list(seats_data),
            total_seats,
        )

        # One (name | None, line) pair per added line; duplicate names are kept rather than
        # silently overwriting the earlier line.
        self._lines: list[tuple[str | None, _PaintballLine]] = []

    @property
    def hull_document(self) -> TexDocument:
        """Return the LaTeX document for the hull-based paintball plot.

        Returns:
            TexDocument: Document object containing hull-rendered TikZ code.
        """
        self._hull_document.body_string = self._generate_latex(hull=True)
        return self._hull_document

    def print(self, *, hull: bool = False) -> None:
        """Print the generated TikZ body to stdout.

        Args:
            hull (bool, optional): If True, print the hull-rendered plot; otherwise print
                the point-rendered plot. Defaults to False.

        Returns:
            None
        """
        if hull:
            print(self._generate_latex(hull=True))
        else:
            print(self._generate_latex())

    def preview(self, hull: bool = False) -> None:  # pragma: no cover
        """Preview the rendered LaTeX plot.

        Args:
            hull (bool, optional): If True, preview the hull-rendered plot; otherwise preview
                the point-rendered plot. Defaults to False.

        Returns:
            None
        """
        if hull:
            self.hull_document.preview()
        else:
            self.document.preview()

    # ====================
    #   FEATURE ADDITION
    # ====================

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
                and normalized by dividing by ``total_seats``.
            total_seats (int | None, optional): The maximum number of seats. If provided,
                seats_data will be scaled by this value to obtain seat shares. If None,
                seats_data is assumed to already be in seat share format (i.e., in [0, 1]).
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
        linecolor: Color | None = "black",
        linewidth: float = 1.0,
        linestyle: str = "solid",
        *,
        name: str | None = None,
    ) -> None:
        """Adds lines with specified slopes to the paintball plot.

        Args:
            slopes (Iterable[float]): The slopes of the lines to be added.
            linecolor (Color | None, optional): The color of the lines. Pass None for
                transparent lines. Defaults to "black".
            linewidth (float, optional): The width of the lines. Defaults to 1.0
            linestyle (str, optional): The style of the lines (Matplotlib token or TikZ style).
                Defaults to "solid".
            name (str | None, optional): An optional name for the line. If provided,
                the line can be referenced later by this name. Defaults to None.
        """
        for slope in slopes:
            line = _PaintballLine(
                slope=slope,
                linecolor=linecolor,
                linewidth=linewidth,
                linestyle=linestyle,
            )
            self._lines.append((name, line))

    def add_efficiency_gap_line(
        self,
        *,
        linecolor: Color | None = "gray",
        linewidth: float = 1.0,
        linestyle: str = "solid",
        name: str = "efficiency_gap",
    ) -> None:
        """Add the standard efficiency-gap guide line (slope 2 through (0.5, 0.5)).

        Args:
            linecolor (Color | None, optional): Line color. Pass None for a transparent
                line. Defaults to "gray".
            linewidth (float, optional): Line width. Defaults to 1.0.
            linestyle (str, optional): Line style (Matplotlib token or TikZ style). Defaults to
                "solid".
            name (str, optional): Name the line is stored under. Defaults to "efficiency_gap".
        """
        self.add_lines_with_slope(
            [2.0], linecolor=linecolor, linewidth=linewidth, linestyle=linestyle, name=name
        )

    def add_proportionality_line(
        self,
        *,
        linecolor: Color | None = "gray",
        linewidth: float = 1.0,
        linestyle: str = "dashed",
        name: str = "proportionality",
    ) -> None:
        """Add the standard proportionality guide line (slope 1 through (0.5, 0.5)).

        Args:
            linecolor (Color | None, optional): Line color. Pass None for a transparent
                line. Defaults to "gray".
            linewidth (float, optional): Line width. Defaults to 1.0.
            linestyle (str, optional): Line style (Matplotlib token or TikZ style). Defaults to
                "dashed".
            name (str, optional): Name the line is stored under. Defaults to "proportionality".
        """
        self.add_lines_with_slope(
            [1.0], linecolor=linecolor, linewidth=linewidth, linestyle=linestyle, name=name
        )

    def clear_lines(self) -> None:
        """Clears all added lines from the paintball plot."""
        self._lines = []

    # ==================
    #   OPTION SETTERS
    # ==================

    def set_crosshair_options(self, color: Color | None, width: float) -> None:
        """Sets the crosshair options for the paintball plot.

        Args:
            color (Color | None): The color of the crosshair. None removes it.
            width (float): The width of the crosshair lines.
        """
        self.options.crosshair_color = color
        self.options.crosshair_width = width

    def set_marker_options(
        self,
        size: float | None = None,
        color: Color | None | Unset = UNSET,
        alpha: float | None = None,
        edgecolor: Color | None | Unset = UNSET,
        edgewidth: float | None = None,
        edgealpha: float | None = None,
    ) -> None:
        """Sets the marker options for the paintball plot.

        Args:
            size (float | None, optional): The size of the markers in points. If None, the size is
                not changed from the previous setting. Defaults to None.
            color (Color | None | Unset, optional): The color of the markers. None removes
                the fill; omission retains the current color. Defaults to UNSET.
            alpha (float | None, optional): The opacity of the markers (0.0 to 1.0). If None, the
                opacity is not changed from the previous setting. Defaults to None.
            edgecolor (Color | None | Unset, optional): The edge color of the markers. None
                removes the edge; omission retains the current color. Defaults to UNSET.
            edgewidth (float | None, optional): The edge width of the markers. If None, the edge
                width is not changed from the previous setting. Defaults to None.
            edgealpha (float | None, optional): The edge opacity of the markers (0.0 to 1.0). If
                None, the edge opacity is not changed from the previous setting. Defaults to None.
        """
        if size is not None:
            self.options.markersize = size
        if not isinstance(color, Unset):
            self.options.markercolor = color
        if alpha is not None:
            self.options.markeralpha = alpha
        if not isinstance(edgecolor, Unset):
            self.options.markeredgecolor = edgecolor
        if edgewidth is not None:
            self.options.markeredgewidth = edgewidth
        if edgealpha is not None:
            self.options.markeredgealpha = edgealpha

    def set_hull_options(
        self,
        color: Color | None | Unset = UNSET,
        alpha: float | None | Unset = UNSET,
        edgecolor: Color | None | Unset = UNSET,
        edgewidth: float | None | Unset = UNSET,
        edgealpha: float | None | Unset = UNSET,
    ) -> None:
        """Sets the hull options for the paintball plot.

        Omitted keywords leave the current setting unchanged. For colors, passing ``None``
        removes the fill or edge. Alpha and width retain their existing behavior: passing
        ``None`` restores inheritance from the corresponding marker option.

        Args:
            color (Color | None | Unset): The fill color of the hull; None removes the fill.
                Defaults to UNSET (leave unchanged).
            alpha (float | None | Unset): The opacity of the hull fill (0.0 to 1.0); None inherits
                ``markeralpha``. Defaults to UNSET (leave unchanged).
            edgecolor (Color | None | Unset): The edge color of the hull; None removes the
                edge. Defaults to UNSET (leave unchanged).
            edgewidth (float | None | Unset): The edge width of the hull in points; None inherits
                ``markeredgewidth``. Defaults to UNSET (leave unchanged).
            edgealpha (float | None | Unset): The edge opacity of the hull (0.0 to 1.0); None
                inherits ``markeredgealpha``. Defaults to UNSET (leave unchanged).
        """
        if color is not UNSET:
            self.options.hullcolor = color
        if alpha is not UNSET:
            self.options.hullalpha = alpha
        if edgecolor is not UNSET:
            self.options.hulledgecolor = edgecolor
        if edgewidth is not UNSET:
            self.options.hulledgewidth = edgewidth
        if edgealpha is not UNSET:
            self.options.hulledgealpha = edgealpha

    # =====================
    #   STRING GENERATORS
    # =====================

    def _paintball_points_str(self) -> str:
        """Generate TikZ code for point markers.

        Returns:
            str: TikZ snippet that renders all paintball markers.
        """
        marker_fill = self._to_latex_color(self.options.markercolor)
        marker_edge = self._to_latex_color(self.options.markeredgecolor)
        x_coordinates, y_coordinates = paintball_coordinates(
            self._voteshare_data, self._seatshare_data
        )
        tex_string = "\\foreach \\votes/\\seats in {\n"
        for x_coord, y_coord in zip(x_coordinates, y_coordinates):
            tex_string += f"    {x_coord:0.4f}/{y_coord:0.4f},\n"
        tex_string = tex_string.rstrip(",\n") + "\n"  # Remove trailing comma
        tex_string += "} {\n"
        node = self._marker_node_command(
            x=r"\votes",
            y=r"\seats",
            color=marker_fill,
            size_pt=self.options.markersize,
            edge_color=marker_edge,
            fill_opacity=self.options.markeralpha,
            edge_width=self.options.markeredgewidth,
            edge_opacity=self.options.markeredgealpha,
            transform_shape=False,
        )
        tex_string += f"    {node}\n}}"
        return tex_string

    def _paintball_hull_str(self) -> str:
        """Generate TikZ code for the horizontal hull polygon.

        Returns:
            str: TikZ snippet that renders the hull polygon.
        """
        x_coordinates, y_coordinates = paintball_coordinates(
            self._voteshare_data, self._seatshare_data
        )
        hull_vertices = horizontal_hull_vertices(zip(x_coordinates, y_coordinates))

        # Use marker settings as defaults only when hull options are unset.
        fillcolor = (
            self.options.markercolor
            if isinstance(self.options.hullcolor, Unset)
            else self.options.hullcolor
        )
        fillalpha = (
            self.options.hullalpha
            if self.options.hullalpha is not None
            else self.options.markeralpha
        )
        linecolor = (
            self.options.markeredgecolor
            if isinstance(self.options.hulledgecolor, Unset)
            else self.options.hulledgecolor
        )
        linewidth = (
            self.options.hulledgewidth
            if self.options.hulledgewidth is not None
            else self.options.markeredgewidth
        )
        linealpha = (
            self.options.hulledgealpha
            if self.options.hulledgealpha is not None
            else self.options.markeredgealpha
        )
        path = "--\n".join(
            f"  ({x_coord:0.4f},{y_coord:0.4f})" for x_coord, y_coord in hull_vertices
        )
        # ``cycle`` closes the stroke like the matplotlib backend's ring; a bare path leaves
        # the closing edge filled but unstroked.
        return self._draw_path_command(
            path="\n" + path + " -- cycle",
            color=self._to_latex_color(linecolor),
            linewidth=linewidth,
            fill=self._to_latex_color(fillcolor),
            fill_opacity=fillalpha,
            draw_opacity=linealpha,
        )

    def _generate_latex(self, *, hull=False) -> str:
        """Generate complete TikZ content for the paintball plot.

        Args:
            hull (bool, optional): Whether to draw the horizontal hull of the paintball points
                rather than the individual points. Defaults to False.

        Returns:
            str: Complete TikZ picture source.
        """
        tex_string = "\\begin{tikzpicture}\n\\begin{scope}"
        tex_string += f"[xscale={self.options.xscale}, yscale={self.options.yscale}]\n\n"

        # Clip the drawing area
        tex_string += (
            f"\\clip [draw] ({self.options.xlim[0]}, {self.options.ylim[0]}) "
            f"rectangle ({self.options.xlim[1]}, {self.options.ylim[1]});\n\n"
        )

        # Draw crosshairs spanning the configured limits
        crosshair_color = self._to_latex_color(self.options.crosshair_color)
        xlim, ylim = self.options.xlim, self.options.ylim
        tex_string += self._draw_path_command(
            path=f"(0.5, {ylim[0]}) -- (0.5, {ylim[1]})",
            color=crosshair_color,
            linewidth=self.options.crosshair_width,
        )
        tex_string += "\n"
        tex_string += self._draw_path_command(
            path=f"({xlim[0]}, 0.5) -- ({xlim[1]}, 0.5)",
            color=crosshair_color,
            linewidth=self.options.crosshair_width,
        )
        tex_string += "\n\n"

        # Draw lines: named lines render before unnamed ones, preserving the emission order of
        # the earlier dict-plus-list storage.
        ordered_lines = [line for line_name, line in self._lines if line_name is not None]
        ordered_lines += [line for line_name, line in self._lines if line_name is None]
        for line in ordered_lines:
            starting_x, starting_y, ending_x, ending_y = line_segment_through_unit_square(
                line.slope, round_to=4
            )
            tex_string += self._draw_path_command(
                path=f"({starting_x}, {starting_y}) -- ({ending_x}, {ending_y})",
                color=self._to_latex_color(line.linecolor),
                linewidth=line.linewidth,
                linestyle=line.linestyle,
            )
            tex_string += "\n"

        # Add paintballs or hull
        tex_string += "\n"
        if hull:
            tex_string += self._paintball_hull_str()
        else:
            tex_string += self._paintball_points_str()

        tex_string += "\n\n"
        tex_string += "\\end{scope}\n\\end{tikzpicture}\n"

        return tex_string
