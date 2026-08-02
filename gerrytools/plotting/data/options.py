"""Public styling-options dataclasses for the data plots.

Each dataclass collects the styling kwargs that one ``add_*`` method takes,
so users can compose a style once and reuse it across calls. Kwargs on the
``add_*`` methods remain the primary path; ``*_options=`` is the secondary
compose-and-reuse path.

Resolution rule (documented per-method): start from ``options`` (or its
default if ``None``), then override each field with whatever explicit kwargs
the caller passed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from matplotlib.artist import Artist
from matplotlib.axes import Axes

from gerrytools.colors import resolve_color_and_alpha, resolve_rgba, validate_alpha
from gerrytools.logging import get_logger
from gerrytools.plotting.mpl.marker_options import PointMarkerOptions
from gerrytools.plotting.utils import (
    UNSET,
    Unset,
    _replace_with_color_overrides,
    _resolve_color_clamped_width,
    _validated_nonneg_finite,
)
from gerrytools.typing import Color, HistType

logger = get_logger(__name__)


# Edge width applied when a visible edge color is set but no width is given. An edge
# color with zero width draws nothing, so asking for a color is taken to mean "draw the
# edge": the width falls back to this default rather than forcing the caller to set both.
DEFAULT_EDGE_WIDTH = 0.8


class _DefaultZorder(int):
    """Internal marker that survives ``dataclasses.replace``."""


class _DefaultEdgeWidth(float):
    """Internal marker distinguishing an omitted zero width from an explicit zero."""


_DEFAULT_ANNOTATION_ZORDER = _DefaultZorder(3)
_DEFAULT_ZERO_EDGE_WIDTH = _DefaultEdgeWidth(0.0)


def _resolve_annotation_zorder(value: int | float | Unset) -> tuple[int, bool]:
    if isinstance(value, Unset):
        return _DEFAULT_ANNOTATION_ZORDER, True
    if isinstance(value, _DefaultZorder):
        return value, True
    return int(value), False


def _needs_default_edge_width(
    *,
    edgewidth_given: bool,
    resolved_edgewidth: float,
    resolved_edgecolor: Color | None,
) -> bool:
    """Whether an unset edge width should fall back to ``DEFAULT_EDGE_WIDTH``.

    A visible edge color with zero width draws nothing, so naming an edge color while
    leaving the width unset is taken to mean "draw the edge". An explicit width of 0
    still hides it.
    """
    return (
        not edgewidth_given
        and resolved_edgewidth == 0.0
        and resolved_edgecolor is not None
        and str(resolved_edgecolor).strip().lower() != "none"
    )


# ---------------------------------------------------------------------------
# Lines and bands (used by GerryPlotBase add_*_lines / add_*_band).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LineOptions:
    """Styling for vertical/horizontal annotation lines.

    Attributes:
        linecolor (Color | None): The color of the line. None removes it. Defaults to
            "#cccccc".
        linealpha (float | None): Optional alpha override. Defaults to None.
        linestyle (str): Matplotlib linestyle. Defaults to "-".
        linewidth (float): Line width in points. Defaults to 1.0.
        zorder (int | float): Z-order for layering; coerced to int. Defaults to 3, but the
            annotation add methods substitute their documented orientation default
            (3 for vertical, 4 for horizontal) when this is left unset.

    Raises:
        ValueError: If a color, alpha, width, or z-order value is invalid.
    """

    linecolor: Color | None = "#cccccc"
    linealpha: float | None = None
    linestyle: str = "-"
    linewidth: float = 1.0
    zorder: int | float | Unset = UNSET
    # True when the constructor received no explicit zorder; consumed (before any merge)
    # by the annotation add methods to substitute their orientation default.
    _zorder_defaulted: bool = field(init=False, compare=False, repr=False)

    def __post_init__(self) -> None:
        owner = type(self).__name__
        line_width_value = _validated_nonneg_finite(self.linewidth, field="linewidth")
        resolved_linecolor, resolved_linealpha, line_width_value = _resolve_color_clamped_width(
            self.linecolor,
            self.linealpha,
            line_width_value,
            color_field="linecolor",
            width_field="linewidth",
            owner=owner,
        )
        object.__setattr__(self, "linecolor", resolved_linecolor)
        object.__setattr__(self, "linealpha", resolved_linealpha)
        object.__setattr__(self, "linewidth", line_width_value)
        zorder, defaulted = _resolve_annotation_zorder(self.zorder)
        object.__setattr__(self, "zorder", zorder)
        object.__setattr__(self, "_zorder_defaulted", defaulted)


@dataclass(frozen=True)
class BandOptions:
    """Styling for vertical/horizontal annotation bands (filled regions).

    Attributes:
        bandcolor (Color | None): The fill color of the band. None removes it. Defaults to
            "#cccccc".
        bandalpha (float | None): Optional alpha override for the fill. Defaults to None.
        linecolor (Color | None | Unset): Optional bounding-line color. An omitted value
            falls back to ``bandcolor`` (or ``"#cccccc"`` when the band fill is "none").
            None removes the bounding line.
        linealpha (float | None): Optional alpha override for the bounding lines. Defaults to None.
        linestyle (str): Bounding-line linestyle. Defaults to "-".
        linewidth (float): Bounding-line width in points. Defaults to 1.0.
        zorder (int | float): Z-order for layering; coerced to int. Defaults to 3, but the
            annotation add methods substitute their documented orientation default
            (3 for vertical, 4 for horizontal) when this is left unset.

    Raises:
        ValueError: If a color, alpha, width, or z-order value is invalid.
    """

    bandcolor: Color | None = "#cccccc"
    bandalpha: float | None = None
    linecolor: Color | None | Unset = UNSET
    linealpha: float | None = None
    linestyle: str = "-"
    linewidth: float = 1.0
    zorder: int | float | Unset = UNSET
    # True when the constructor received no explicit zorder; consumed (before any merge)
    # by the annotation add methods to substitute their orientation default.
    _zorder_defaulted: bool = field(init=False, compare=False, repr=False)
    _linecolor_defaulted: bool = field(init=False, compare=False, repr=False)

    def __post_init__(self) -> None:
        linecolor_defaulted = isinstance(self.linecolor, Unset)
        line_width_value = _validated_nonneg_finite(self.linewidth, field="linewidth")
        object.__setattr__(self, "linewidth", line_width_value)

        resolved_bandcolor, resolved_bandalpha = resolve_color_and_alpha(
            self.bandcolor,
            self.bandalpha,
            allow_none=True,
            field="bandcolor",
            owner="BandOptions",
            logger=logger,
        )
        object.__setattr__(self, "bandcolor", resolved_bandcolor)
        object.__setattr__(self, "bandalpha", resolved_bandalpha)

        # Bounding lines default to the band fill; a transparent fill falls back to the
        # neutral default so the band still has a visible boundary color.
        line_color_input = self.linecolor
        if isinstance(line_color_input, Unset):
            line_color_input = resolved_bandcolor
            if isinstance(line_color_input, str) and line_color_input.lower() == "none":
                line_color_input = "#cccccc"
        resolved_linecolor, resolved_linealpha, line_width_value = _resolve_color_clamped_width(
            line_color_input,
            self.linealpha,
            line_width_value,
            color_field="linecolor",
            width_field="linewidth",
            owner="BandOptions",
        )
        object.__setattr__(self, "linecolor", resolved_linecolor)
        object.__setattr__(self, "linealpha", resolved_linealpha)
        object.__setattr__(self, "linewidth", line_width_value)
        zorder, defaulted = _resolve_annotation_zorder(self.zorder)
        object.__setattr__(self, "zorder", zorder)
        object.__setattr__(self, "_zorder_defaulted", defaulted)
        object.__setattr__(self, "_linecolor_defaulted", linecolor_defaulted)

    def resolved_edgecolor(
        self, *, owner: str = "BandOptions"
    ) -> str | tuple[float, float, float, float]:
        """Edge color to draw the band's bounding lines with.

        Encodes the one shared drawing rule: zero-width bounding lines resolve to
        ``"none"`` so matplotlib's default hairline edge never appears.

        Args:
            owner (str, optional): Owner name used in validation messages. Defaults to
                ``"BandOptions"``.

        Returns:
            str | tuple[float, float, float, float]: ``"none"`` or the resolved RGBA edge color.
        """
        if self.linewidth == 0.0:
            return "none"
        linecolor = self.linecolor
        assert not isinstance(linecolor, Unset)
        return resolve_rgba(linecolor, self.linealpha, field="linecolor", owner=owner)


# ---------------------------------------------------------------------------
# Histogram, BarPlot, BoxPlot, ViolinPlot.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _FaceEdgeStyle:
    """Shared face/edge styling block with one validation pass.

    The four distribution-family options classes inherit this: resolved colors, the
    edge-width checks, the invisible-edge clamp, and zorder coercion live here once.

    Attributes:
        facecolor (Color | None): Fill color. None removes the fill.
        facealpha (float | None): Optional alpha override for the fill.
        edgecolor (Color | None): Edge color. None removes the edge.
        edgealpha (float | None): Optional alpha override for the edge.
        edgewidth (float): Edge line width in points.
        zorder (int | float): Z-order for layering; coerced to int.
    """

    facecolor: Color | None = "default_grey"
    facealpha: float | None = None
    edgecolor: Color | None = "black"
    edgealpha: float | None = None
    edgewidth: float = 0.8
    zorder: int | float = 1

    def __post_init__(self) -> None:
        owner = type(self).__name__

        edge_width_value = _validated_nonneg_finite(self.edgewidth, field="edgewidth")
        object.__setattr__(self, "edgewidth", edge_width_value)

        resolved_facecolor, resolved_facealpha = resolve_color_and_alpha(
            self.facecolor,
            self.facealpha,
            allow_none=True,
            field="facecolor",
            owner=owner,
            logger=logger,
        )
        object.__setattr__(self, "facecolor", resolved_facecolor)
        object.__setattr__(self, "facealpha", resolved_facealpha)

        resolved_edgecolor, resolved_edgealpha, edge_width_value = _resolve_color_clamped_width(
            self.edgecolor,
            self.edgealpha,
            edge_width_value,
            color_field="edgecolor",
            width_field="edgewidth",
            owner=owner,
        )
        object.__setattr__(self, "edgecolor", resolved_edgecolor)
        object.__setattr__(self, "edgealpha", resolved_edgealpha)
        object.__setattr__(self, "edgewidth", edge_width_value)

        object.__setattr__(self, "zorder", int(self.zorder))

    def merged(
        self,
        *,
        facecolor: Color | None | Unset = UNSET,
        facealpha: float | None = None,
        edgecolor: Color | None | Unset = UNSET,
        edgealpha: float | None = None,
        **other: Any,
    ) -> Any:
        """Copy this style with the caller's explicit overrides applied.

        Encodes the color/alpha pairing rule once: an explicit alpha always wins; overriding
        only a color keeps the base alpha unless the base color was the fully transparent
        "none", whose 0.0 alpha would render the override invisibly. Colors use the ``UNSET``
        sentinel so an explicit ``None`` means "none" while an omitted kwarg inherits; every
        other field treats ``None`` as "inherit". The merged result re-runs validation.

        Use this for the face/edge options classes; ``_replace_non_none`` in
        :mod:`gerrytools.plotting.utils` is the plain None-inherits merge for every other
        options dataclass, where ``None`` is never a meaningful field value.

        Args:
            facecolor (Color | None | Unset, optional): Fill-color override. Defaults to unset.
            facealpha (float | None, optional): Fill-opacity override. Defaults to None.
            edgecolor (Color | None | Unset, optional): Edge-color override. Defaults to unset.
            edgealpha (float | None, optional): Edge-opacity override. Defaults to None.
            **other (Any): Non-None overrides for other fields.

        Returns:
            Any: A new instance of the same options class (``self`` when nothing was overridden).
        """
        return _replace_with_color_overrides(
            self,
            ("facecolor", "facealpha"),
            ("edgecolor", "edgealpha"),
            facecolor=facecolor,
            facealpha=facealpha,
            edgecolor=edgecolor,
            edgealpha=edgealpha,
            **other,
        )


@dataclass(frozen=True)
class HistogramOptions(_FaceEdgeStyle):
    """Styling for a single histogram series added via ``Histogram.add_dataset``.

    Defaults produce filled bars with visible black edges. For ``histtype="outline"``,
    the method enforces sensible-outline overrides (positive ``edgewidth``,
    ``facecolor="none"``, ``edgecolor="black"``).

    Attributes:
        facecolor (Color | None): Fill color for histogram bars. None removes the fill. Defaults to
            ``"default_grey"``.
        facealpha (float | None): Optional fill-opacity override. Defaults to None.
        edgecolor (Color | None): Edge color for histogram bars. None removes the edge. Defaults to
            ``"black"``.
        edgealpha (float | None): Optional edge-opacity override. Defaults to None.
        edgewidth (float): Edge line width in points. An omitted width becomes 0.8 when a visible
            edge color is selected; an explicit 0 hides the edge.
        histtype (HistType): One of "overlay", "stack", "grouped", "outline". Defaults to
            ``"overlay"``.
        zorder (int): Z-order for layering. Defaults to 2.

    Raises:
        ValueError: If a color, alpha, width, histogram type, or z-order value is invalid.
    """

    edgecolor: Color | None = "black"
    edgewidth: float = _DEFAULT_ZERO_EDGE_WIDTH
    histtype: HistType = "overlay"
    zorder: int = 2
    _edgewidth_defaulted: bool = field(init=False, compare=False, repr=False)

    def __post_init__(self) -> None:
        defaulted = isinstance(self.edgewidth, _DefaultEdgeWidth)
        super().__post_init__()
        if defaulted:
            object.__setattr__(self, "edgewidth", _DefaultEdgeWidth(self.edgewidth))
        object.__setattr__(self, "_edgewidth_defaulted", defaulted)


@dataclass(frozen=True)
class BarPlotOptions(_FaceEdgeStyle):
    """Styling for a single bar dataset added via ``BarPlot.add_dataset`` or
    ``BarPlot.add_counts_dataset``.

    Attributes:
        facecolor (Color | None): Fill color for bars. None removes the fill. Defaults to
            ``"default_grey"``.
        facealpha (float | None): Optional fill-opacity override. Defaults to None.
        edgecolor (Color | None): Edge color for bars. None removes the edge. Defaults to
            ``"black"``.
        edgealpha (float | None): Optional edge-opacity override. Defaults to None.
        edgewidth (float): Edge line width. Defaults to 0.8.
        zorder (int): Z-order for layering. Defaults to 1.

    Raises:
        ValueError: If a color, alpha, width, or z-order value is invalid.
    """


@dataclass(frozen=True)
class BoxPlotOptions(_FaceEdgeStyle):
    """Styling for a single boxplot dataset added via ``BoxPlot.add_dataset``.

    Attributes:
        facecolor (Color | None): Fill color for boxes. None removes the fill. Defaults to
            ``"default_grey"``.
        facealpha (float | None): Optional fill-opacity override. Defaults to None.
        edgecolor (Color | None): Edge color for boxes and whiskers. None removes the edge.
            Defaults to ``"black"``.
        edgealpha (float | None): Optional edge-opacity override. Defaults to None.
        edgewidth (float): Edge line width. Defaults to 0.8.
        percentiles (tuple[float, float]): Whisker percentile bounds; both values must be in
            ``[0, 100]`` and ``low < high``. Defaults to ``(1, 99)``.
        showfliers (bool): Whether to render outlier points. Defaults to False.
        flier_options (PointMarkerOptions): Marker styling for outliers. Defaults to
            ``PointMarkerOptions()``.
        zorder (int): Z-order for layering. Defaults to 1.

    Raises:
        ValueError: If the percentile bounds or styling values are invalid.
    """

    percentiles: tuple[float, float] = (1, 99)
    showfliers: bool = False
    flier_options: PointMarkerOptions = field(default_factory=PointMarkerOptions)

    def __post_init__(self) -> None:
        percentile_low, percentile_high = self.percentiles
        percentile_low = float(percentile_low)
        percentile_high = float(percentile_high)
        if not (0.0 <= percentile_low <= 100.0 and 0.0 <= percentile_high <= 100.0):
            raise ValueError("percentiles must be within [0, 100].")
        if not (percentile_low < percentile_high):
            raise ValueError("percentiles must satisfy low < high.")
        object.__setattr__(self, "percentiles", (percentile_low, percentile_high))

        super().__post_init__()


@dataclass(frozen=True)
class ViolinPlotOptions(_FaceEdgeStyle):
    """Styling for a single violin dataset added via ``ViolinPlot.add_dataset``.

    Attributes:
        facecolor (Color | None): Fill color for violins. None removes the fill. Defaults to
            ``"default_grey"``.
        facealpha (float | None): Optional fill-opacity override. Defaults to None.
        edgecolor (Color | None): Edge color for violin outlines. None removes the edge. Defaults
            to ``"black"``.
        edgealpha (float | None): Optional edge-opacity override. Defaults to None.
        edgewidth (float): Edge line width. Defaults to 0.8.
        zorder (int): Z-order for layering. Defaults to 1.

    Raises:
        ValueError: If a color, alpha, width, or z-order value is invalid.
    """


# ---------------------------------------------------------------------------
# SeatsVotesPlot — line and marker subsets.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SeatsVotesLineOptions:
    """Styling for the seats-votes curve line in ``SeatsVotesPlot.add_election``.

    An unset ``linecolor`` inherits the plot default; explicit ``None`` means no line.

    Attributes:
        linecolor (Color | None | Unset): Curve color. An omitted value inherits from the plot;
            ``None`` removes the line.
        linealpha (float | None): Optional alpha override. Defaults to None.
        linestyle (str): Matplotlib linestyle. Defaults to "-".
        linewidth (float | None): Optional width override; ``None`` inherits.
        zorder (int | float): Z-order for the curve; coerced to int. Defaults to 1.

    Raises:
        ValueError: If a color, alpha, width, or z-order value is invalid.
    """

    linecolor: Color | None | Unset = UNSET
    linealpha: float | None = None
    linestyle: str = "-"
    linewidth: float | None = None
    zorder: int | float = 1

    def __post_init__(self) -> None:
        if self.linealpha is not None:
            object.__setattr__(self, "linealpha", validate_alpha(self.linealpha, field="linealpha"))

        if self.linewidth is not None:
            object.__setattr__(
                self, "linewidth", _validated_nonneg_finite(self.linewidth, field="linewidth")
            )

        if not isinstance(self.linecolor, Unset):
            resolved_linecolor, resolved_linealpha = resolve_color_and_alpha(
                self.linecolor,
                self.linealpha,
                allow_none=True,
                field="linecolor",
                owner="SeatsVotesLineOptions",
                logger=logger,
            )
            object.__setattr__(self, "linecolor", resolved_linecolor)
            object.__setattr__(self, "linealpha", resolved_linealpha)

        object.__setattr__(self, "zorder", int(self.zorder))

    def resolved_linecolor(self) -> Color | None:
        """Return the concrete curve color after plot-level defaults are applied.

        Returns:
            Color | None: Resolved curve color, or None for no line.
        """
        assert not isinstance(self.linecolor, Unset)
        return self.linecolor


@dataclass(frozen=True)
class SeatsVotesMarkerOptions:
    """Styling for the election-result marker in ``SeatsVotesPlot.add_election``.

    Attributes:
        markerfacecolor (Color | None | Unset): Marker fill color. An omitted value inherits from
            the plot; ``None`` removes the fill.
        markerfacealpha (float | None): Optional fill-opacity override. Defaults to None.
        marker (str): Matplotlib marker style. Defaults to "o".
        markersize (float | None): Optional size override; ``None`` inherits.
        markeredgecolor (Color | None | Unset): Marker edge color. An omitted value inherits the
            marker face; ``None`` removes the edge.
        markeredgealpha (float | None): Optional edge-opacity override. Defaults to None.
        markeredgewidth (float): Marker edge width. An omitted width becomes 0.8 when a visible
            edge color is selected; an explicit 0 hides the edge.
        marker_zorder (int | float): Z-order for the marker; coerced to int. Defaults to 2.

    Raises:
        ValueError: If a color, alpha, size, width, or z-order value is invalid.
    """

    markerfacecolor: Color | None | Unset = UNSET
    markerfacealpha: float | None = None
    marker: str = "o"
    markersize: float | None = None
    markeredgecolor: Color | None | Unset = UNSET
    markeredgealpha: float | None = None
    markeredgewidth: float = _DEFAULT_ZERO_EDGE_WIDTH
    marker_zorder: int | float = 2
    _markeredgewidth_defaulted: bool = field(init=False, compare=False, repr=False)

    def __post_init__(self) -> None:
        edgewidth_defaulted = isinstance(self.markeredgewidth, _DefaultEdgeWidth)
        if self.markerfacealpha is not None:
            object.__setattr__(
                self,
                "markerfacealpha",
                validate_alpha(self.markerfacealpha, field="markerfacealpha"),
            )

        if self.markersize is not None:
            object.__setattr__(
                self, "markersize", _validated_nonneg_finite(self.markersize, field="markersize")
            )

        if self.markeredgealpha is not None:
            object.__setattr__(
                self,
                "markeredgealpha",
                validate_alpha(self.markeredgealpha, field="markeredgealpha"),
            )

        object.__setattr__(
            self,
            "markeredgewidth",
            _validated_nonneg_finite(self.markeredgewidth, field="markeredgewidth"),
        )

        if not isinstance(self.markerfacecolor, Unset):
            resolved_face, resolved_face_alpha = resolve_color_and_alpha(
                self.markerfacecolor,
                self.markerfacealpha,
                allow_none=True,
                field="markerfacecolor",
                owner="SeatsVotesMarkerOptions",
                logger=logger,
            )
            object.__setattr__(self, "markerfacecolor", resolved_face)
            object.__setattr__(self, "markerfacealpha", resolved_face_alpha)

        if not isinstance(self.markeredgecolor, Unset):
            resolved_edge, resolved_edge_alpha = resolve_color_and_alpha(
                self.markeredgecolor,
                self.markeredgealpha,
                allow_none=True,
                field="markeredgecolor",
                owner="SeatsVotesMarkerOptions",
                logger=logger,
            )
            object.__setattr__(self, "markeredgecolor", resolved_edge)
            object.__setattr__(self, "markeredgealpha", resolved_edge_alpha)

        object.__setattr__(self, "marker_zorder", int(self.marker_zorder))
        if edgewidth_defaulted:
            object.__setattr__(
                self,
                "markeredgewidth",
                _DefaultEdgeWidth(self.markeredgewidth),
            )
        object.__setattr__(self, "_markeredgewidth_defaulted", edgewidth_defaulted)

    def resolved_markerfacecolor(self) -> Color | None:
        """Return the concrete marker fill after plot-level defaults are applied.

        Returns:
            Color | None: Resolved marker fill, or None for no fill.
        """
        assert not isinstance(self.markerfacecolor, Unset)
        return self.markerfacecolor


# ---------------------------------------------------------------------------
# SeaLevelPlot — line subset only (markers reuse PointMarkerOptions).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SeaLevelLineOptions(LineOptions):
    """Styling for the connecting line in ``SeaLevelPlot.add_dataset``.

    A ``LineOptions`` overriding only the defaults: a black, slightly heavier line drawn
    above the sea-level markers.

    Attributes:
        linecolor (Color | None): Line color. None removes the line. Defaults to "black".
        linealpha (float | None): Optional alpha override. Defaults to None.
        linestyle (str): Matplotlib linestyle. Defaults to ``"-"``.
        linewidth (float): Line width in points. Defaults to 1.5.
        zorder (int | float): Z-order for the line; coerced to int. Defaults to 2.

    Raises:
        ValueError: If a color, alpha, width, or z-order value is invalid.
    """

    linecolor: Color | None = "black"
    linewidth: float = 1.5
    zorder: int | float = 2


@dataclass(frozen=True, slots=True)
class _CrosshairStyle:
    """Center crosshair guides at (0.5, 0.5), shared by SeatsVotesPlot and PaintballPlot.

    Widths are data-space band widths; the color resolves at draw time.

    Attributes:
        color (Color | None): Crosshair color. None removes the fill. Defaults to
            "lightgrey".
        alpha (float): Crosshair alpha in [0, 1]. Defaults to 1.0.
        x_width (float): Width of the vertical band in data units. Defaults to 0.02.
        y_width (float): Width of the horizontal band in data units. Defaults to 0.02.
        zorder (int): Draw order. Defaults to -2.
    """

    color: Color | None = "lightgrey"
    alpha: float = 1.0
    x_width: float = 0.02
    y_width: float = 0.02
    zorder: int = -2

    def __post_init__(self) -> None:
        object.__setattr__(self, "alpha", validate_alpha(self.alpha, field="alpha"))
        object.__setattr__(self, "x_width", _validated_nonneg_finite(self.x_width, field="x_width"))
        object.__setattr__(self, "y_width", _validated_nonneg_finite(self.y_width, field="y_width"))

    def draw(self, ax: Axes) -> list[Artist]:
        """Draw the two crosshair spans onto ``ax`` and return the created artists."""
        color = resolve_rgba(
            self.color, self.alpha, field="crosshair_color", owner="_CrosshairStyle"
        )
        vspan = ax.axvspan(
            xmin=0.5 - self.x_width / 2,
            xmax=0.5 + self.x_width / 2,
            color=color,
            zorder=self.zorder,
        )
        hspan = ax.axhspan(
            ymin=0.5 - self.y_width / 2,
            ymax=0.5 + self.y_width / 2,
            color=color,
            zorder=self.zorder,
        )
        return [vspan, hspan]


@dataclass(frozen=True, slots=True)
class _PaintballHullStyle:
    """Horizontal-hull styling for ``PaintballPlot``; None colors inherit the marker style.

    Attributes:
        facecolor (Color | None): Hull fill color; None inherits the marker face color.
        facealpha (float | None): Hull fill alpha; None inherits the marker face alpha.
        edgecolor (Color | None): Hull edge color; None inherits the marker edge color.
        edgealpha (float | None): Hull edge alpha; None inherits the marker edge alpha.
        edgewidth (float): Hull edge width in points. Defaults to 2.0.
    """

    facecolor: Color | None = None
    facealpha: float | None = None
    edgecolor: Color | None = None
    edgealpha: float | None = None
    edgewidth: float = 2.0

    def __post_init__(self) -> None:
        if self.facecolor is None:
            if self.facealpha is not None:
                object.__setattr__(self, "facealpha", validate_alpha(self.facealpha, field="alpha"))
        else:
            facecolor, facealpha = resolve_color_and_alpha(
                self.facecolor,
                self.facealpha,
                field="facecolor",
                owner="_PaintballHullStyle",
            )
            object.__setattr__(self, "facecolor", facecolor)
            object.__setattr__(self, "facealpha", facealpha)

        edgewidth = _validated_nonneg_finite(self.edgewidth, field="edgewidth")
        if self.edgecolor is None:
            if self.edgealpha is not None:
                object.__setattr__(
                    self, "edgealpha", validate_alpha(self.edgealpha, field="edgealpha")
                )
        else:
            edgecolor, edgealpha, edgewidth = _resolve_color_clamped_width(
                self.edgecolor,
                self.edgealpha,
                edgewidth,
                color_field="edgecolor",
                width_field="edgewidth",
                owner="_PaintballHullStyle",
            )
            object.__setattr__(self, "edgecolor", edgecolor)
            object.__setattr__(self, "edgealpha", edgealpha)
        object.__setattr__(self, "edgewidth", edgewidth)
