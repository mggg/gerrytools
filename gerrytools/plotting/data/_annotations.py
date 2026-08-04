"""`_Annotations` — collected vlines/hlines/bands/arrows owned by `GerryPlotBase`.

Internal module. End users still call `plot.add_vertical_lines(...)` etc.;
those methods on `GerryPlotBase` delegate to an `_Annotations` instance held
on the plot. The point of this module is to localize the state-management +
rendering loop for "decorations on top of a plot" and keep annotation
bookkeeping out of the base class.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from gerrytools.colors import resolve_rgba
from gerrytools.plotting._artist_registry import _ArtistRegistry
from gerrytools.plotting.data._arrow_renderer import _AnnotationArrowRenderer
from gerrytools.plotting.data._gerryplot_dataclasses import (
    _AnyArrowData,
    _BandData,
    _LineData,
)


@dataclass
class _Annotations:
    """Mutable holder for all "decoration" annotations on a plot.

    Five fields, one per decoration kind. Each ``add_*`` / ``clear_*`` method
    on `GerryPlotBase` is a thin facade that mutates one of these lists, and
    `apply()` walks them at render time.
    """

    vertical_lines: list[_LineData] = field(default_factory=list)
    vertical_bands: list[_BandData] = field(default_factory=list)
    horizontal_lines: list[_LineData] = field(default_factory=list)
    horizontal_bands: list[_BandData] = field(default_factory=list)
    annotation_arrows: list[_AnyArrowData] = field(default_factory=list)

    # -- mutation -----------------------------------------------------

    def clear_verticals(self) -> None:
        self.vertical_lines.clear()
        self.vertical_bands.clear()

    def clear_horizontals(self) -> None:
        self.horizontal_lines.clear()
        self.horizontal_bands.clear()

    def clear_annotation_arrows(self) -> None:
        self.annotation_arrows.clear()

    # -- rendering ----------------------------------------------------

    def apply(self, ax: Axes, *, fig: Figure, registry: _ArtistRegistry) -> None:
        """Render every annotation onto ``ax``.

        ``fig`` is required by the arrow renderer (canvas draw + axes-bbox
        measurements). ``registry`` collects every artist drawn so the rebuild
        flow can remove only gerrytools-managed artists on the next pass.
        """
        self._draw_lines_and_bands(
            registry,
            lines=self.vertical_lines,
            bands=self.vertical_bands,
            span=ax.axvspan,
            line=ax.axvline,
        )
        self._draw_lines_and_bands(
            registry,
            lines=self.horizontal_lines,
            bands=self.horizontal_bands,
            span=ax.axhspan,
            line=ax.axhline,
        )
        self._draw_arrows(ax, fig=fig, registry=registry)

    @staticmethod
    def _draw_lines_and_bands(
        registry: _ArtistRegistry,
        *,
        lines: list[_LineData],
        bands: list[_BandData],
        span: Callable[..., Rectangle],
        line: Callable[..., Line2D],
    ) -> None:
        """Draw one orientation's bands then lines via the given axvspan/axvline pair."""
        for band in bands:
            style = band.style
            polygon = span(
                band.lower_bound,
                band.upper_bound,
                facecolor=resolve_rgba(style.bandcolor, style.bandalpha, field="bandcolor"),
                edgecolor=style.resolved_edgecolor(),
                linestyle=style.linestyle,
                linewidth=style.linewidth,
                zorder=style.zorder,
            )
            registry.track(polygon)

        for line_data in lines:
            style = line_data.style
            for value in line_data.values:
                line2d = line(
                    value,
                    color=resolve_rgba(style.linecolor, style.linealpha, field="linecolor"),
                    linestyle=style.linestyle,
                    linewidth=style.linewidth,
                    zorder=style.zorder,
                )
                registry.track(line2d)

    def _draw_arrows(self, ax: Axes, *, fig: Figure, registry: _ArtistRegistry) -> None:
        if not self.annotation_arrows:
            return
        renderer = _AnnotationArrowRenderer(ax=ax, fig=fig, registry=registry)
        renderer.render_all(self.annotation_arrows)
