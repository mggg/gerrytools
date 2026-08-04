from abc import ABC, abstractmethod
from typing import Any, cast
from warnings import warn

from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from gerrytools.colors import resolve_rgba
from gerrytools.logging import get_logger
from gerrytools.plotting._artist_registry import _ArtistRegistry
from gerrytools.plotting._axes_backed import _AxesBackedPlot, deferred_axis_update
from gerrytools.plotting._axes_state import Unit, _frame_snapshot, _ManagedAxesState
from gerrytools.plotting._legend_mixin import _LegendMixin
from gerrytools.plotting.data._annotation_api import _AnnotationApiMixin
from gerrytools.plotting.data._annotations import _Annotations
from gerrytools.plotting.data._axis_api import _AxisApiMixin, _AxisState, _TitleText
from gerrytools.plotting.mpl.legend_options import LegendOptions
from gerrytools.typing import Color, LegendHandle

logger = get_logger(__name__)


class GerryPlotBase(_AxisApiMixin, _AnnotationApiMixin, _LegendMixin, _AxesBackedPlot, ABC):
    """Abstract base class for GerryPlot plotting classes."""

    # Class default for legend inclusion. Plots opt in explicitly with ``legend=True``.
    _legend_default: bool = False

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
        """Initialize a GerryPlotBase instance.

        Args:
            figure_size (tuple[float, float] | None, optional): The size of the figure
                in inches. Defaults to ``(10, 6)`` when ``ax`` is not provided. Ignored
                (with a warning) when ``ax`` is provided.
            dpi (int | None, optional): The dots per inch (DPI) of the figure. Defaults
                to ``300`` when ``ax`` is not provided. Ignored (with a warning) when
                ``ax`` is provided.
            ax (matplotlib.axes.Axes | None, optional): Render onto an existing
                matplotlib ``Axes`` instead of creating a fresh figure. Useful for
                callers familiar with matplotlib / seaborn idioms who want to compose
                this plot into a larger figure they control. When provided, the
                plot's lazy build draws onto this axes; content already on the
                axes is left in place, and on rebuilds only the artists this
                plot created are removed. Defaults to None.
            legend (bool | None, optional): Whether to include a legend in the plot.
                ``None`` selects the class default (False). Defaults to None.
            xlabel (str | None, optional): The label for the x-axis. Defaults to None.
            ylabel (str | None, optional): The label for the y-axis. Defaults to None.
            title (str | None, optional): The title of the plot. Defaults to None.
        """
        # --- Pass 1: resolve self._ax and self.fig ---
        # Remembered so ``bind_to_ax(None)`` can recreate a fresh figure with the same geometry as
        # construction.
        self._figure_size = figure_size if figure_size is not None else (10, 6)
        self._figure_dpi = dpi if dpi is not None else 300
        if ax is not None and (figure_size is not None or dpi is not None):
            warn(
                "figure_size and dpi are ignored when ax is provided; "
                "the plot will use the existing figure's size and dpi.",
                UserWarning,
                stacklevel=2,
            )
        self._attach_axes(ax)

        # --- Pass 2: initialize backing state to internal "no opinion" values ---
        self._title_text = _TitleText(unit="title")
        self._xaxis = _AxisState("x")
        self._yaxis = _AxisState("y")
        self._include_legend: bool = self._legend_default
        self._legend_options = LegendOptions()
        self._annotations = _Annotations()
        # Tri-state: None means no opinion, so an externally set ax.grid(...) survives builds.
        self._grid: bool | None = None
        self._frame_visibility: dict[str, bool] = {
            "top": True,
            "right": True,
            "bottom": True,
            "left": True,
        }

        # --- Pass 3: artist registry + managed-axes state ---
        # Created and initialized BEFORE re-applying constructor args so any
        # pre-existing state on a user-supplied axes is classified first.
        self._artists = _ArtistRegistry()
        self._axes_state = _ManagedAxesState()
        self._axes_state.initialize_from_ax(self._ax)

        # --- Pass 4: re-apply non-default constructor args via property setters ---
        # Setters reclaim their managed unit; omitted args stay "no opinion".
        if title is not None:
            self.title = title
        if xlabel is not None:
            self.xlabel = xlabel
        if ylabel is not None:
            self.ylabel = ylabel
        if legend is not None:
            self.legend = legend

    @property
    def legend(self) -> bool:
        """Whether to render a legend on rebuild."""
        return self._include_legend

    @legend.setter
    @deferred_axis_update
    def legend(self, value: bool) -> None:
        self._include_legend = bool(value)
        # Legend identity is recorded by ``_apply_legend`` at the next
        # rebuild. The store-and-claim path keeps last-applied history clean
        # of placeholder values.
        self._axes_state.reclaim_without_value("legend")

    @property
    def grid(self) -> bool | None:
        """Whether to draw a grid, or None to leave existing grid state alone."""
        return self._grid

    @grid.setter
    @deferred_axis_update
    def grid(self, value: bool | None) -> None:
        self._grid = None if value is None else bool(value)

    def display_grid(self, enabled: bool) -> None:
        """Set whether the plot displays a Matplotlib grid.

        Until this is called, builds leave any grid state already on the axes alone.

        Args:
            enabled (bool): Whether to display the grid.
        """
        self.grid = enabled

    def _resolved_rgba(
        self,
        color: Color | None,
        alpha: float | None = None,
        *,
        field: str = "color",
    ) -> tuple[float, float, float, float]:
        """Resolve a ``Color`` plus optional alpha override to an RGBA tuple.

        Args:
            color (Color | None): GerryTools color input. ``None`` resolves to the
                fully transparent ``"none"`` color.
            alpha (float | None, optional): Optional alpha override. Defaults to None.
            field (str, optional): Field name used in validation and warning messages.
                Defaults to ``"color"``.

        Returns:
            tuple[float, float, float, float]: Resolved RGBA values in ``[0, 1]``.
        """
        return resolve_rgba(color, alpha, field=field, owner=self.__class__.__name__)

    # ------------------------------------------------------------------
    # Aspect
    # ------------------------------------------------------------------

    def _apply_aspect_now(self) -> None:
        """Subclass hook: apply the desired aspect to ``self._ax``.

        Base implementation is a no-op (gerrytools defaults to matplotlib's
        ``"auto"``). Subclasses that need a fixed or computed aspect (e.g.
        ``SeatsVotesPlot``, ``PaintballPlot``) override this and call
        ``self._ax.set_aspect(...)``. The base ``_apply_aspect`` helper then
        records the resulting aspect as a gerrytools default so external
        ``ax.set_aspect(...)`` changes between rebuilds are detected and
        respected on the next pass.
        """

    def _apply_aspect(self, external: set[Unit]) -> None:
        """Reconcile the ``aspect`` managed unit.

        Subclass hook ``_apply_aspect_now`` controls what gerrytools applies.
        External changes (a user ``ax.set_aspect("auto")`` between rebuilds)
        win until a subclass override drives a new value through this path.
        """
        self._axes_state.reconcile(
            "aspect", external, self._apply_aspect_now, lambda: self._ax.get_aspect()
        )

    # ------------------------------------------------------------------
    # Frame
    # ------------------------------------------------------------------

    @deferred_axis_update
    def set_frame_visibility(
        self,
        top: bool = True,
        right: bool = True,
        bottom: bool = True,
        left: bool = True,
    ) -> None:
        """Set the visibility of each spine of the plot frame.

        The parameter order (top, right, bottom, left) matches the frame
        snapshot tuple used by the managed-axes state.

        Args:
            top (bool, optional): Whether to show the top spine. Defaults to True.
            right (bool, optional): Whether to show the right spine. Defaults to True.
            bottom (bool, optional): Whether to show the bottom spine. Defaults to True.
            left (bool, optional): Whether to show the left spine. Defaults to True.

        Returns:
            None
        """
        self._frame_visibility = {
            "top": top,
            "right": right,
            "bottom": bottom,
            "left": left,
        }
        self._apply_frame_visibility_now()
        self._axes_state.reclaim_and_mark("frame", _frame_snapshot(self._ax))

    def _apply_frame_visibility_now(self) -> None:
        """Write frame visibility to the axes without ownership bookkeeping.

        Shared by the public setter and the rebuild flow's external-aware
        applier below.
        """
        for spine, visible in self._frame_visibility.items():
            self._ax.spines[spine].set_visible(visible)

    def _apply_frame_visibility(self, external: set[Unit]) -> None:
        self._axes_state.reconcile(
            "frame",
            external,
            self._apply_frame_visibility_now,
            lambda: _frame_snapshot(self._ax),
        )

    # ------------------------------------------------------------------
    # Legend
    # ------------------------------------------------------------------

    def _get_named_line_legend_handles(self) -> list[LegendHandle]:
        """Get legend handles for all named lines.

        Returns:
            list[LegendHandle]: A list of legend handles.
        """
        handles: list[LegendHandle] = []
        for line in self._annotations.vertical_lines + self._annotations.horizontal_lines:
            if line.name is not None:
                style = line.style
                handle = Line2D(
                    [0],
                    [0],
                    color=self._resolved_rgba(
                        style.linecolor,
                        style.linealpha,
                        field="linecolor",
                    ),
                    linestyle=cast("Any", style.linestyle),
                    linewidth=style.linewidth,
                    label=line.name,
                )
                handles.append(handle)

        return handles

    def _get_named_band_legend_handles(self) -> list[LegendHandle]:
        """Get legend handles for all named bands.

        Returns:
            list[LegendHandle]: A list of legend handles.
        """
        handles: list[LegendHandle] = []
        for band in self._annotations.vertical_bands + self._annotations.horizontal_bands:
            if band.name is None:
                continue

            style = band.style
            handle = Patch(
                facecolor=self._resolved_rgba(
                    style.bandcolor,
                    style.bandalpha,
                    field="bandcolor",
                ),
                edgecolor=style.resolved_edgecolor(owner=self.__class__.__name__),
                linestyle=cast("Any", style.linestyle),
                linewidth=style.linewidth,
                label=band.name,
            )
            handles.append(handle)

        return handles

    def save_legend(
        self,
        filepath: str,
        *,
        outer_padding: float = 0.07,
        dpi: int | None = None,
        **legend_kwargs: object,
    ) -> None:
        """Save legend handles to a standalone image.

        Args:
            filepath (str): Output file path.
            outer_padding (float, optional): Fractional padding around the legend bounding box.
                Defaults to ``0.07``.
            dpi (int | None, optional): Output DPI. If None, uses figure DPI.
                Defaults to None.
            **legend_kwargs (object): Additional keyword arguments passed to
                ``matplotlib.axes.Axes.legend``.

        Returns:
            None
        """
        self._save_legend_handles(
            self._legend_handles,
            filepath,
            outer_padding=outer_padding,
            dpi=dpi,
            **legend_kwargs,
        )

    @property
    def _legend_enabled(self) -> bool:
        """Whether ``_apply_legend`` should place a legend; the mixin's enabled hook."""
        return self._include_legend

    # ------------------------------------------------------------------
    # The top-level rebuild flow
    # ------------------------------------------------------------------

    def _build_and_apply_settings(self) -> None:
        """Rebuild the plot: snapshot → remove gerrytools artists → redraw → apply units.

        Ordering highlights:
        - The pre-redraw snapshot is taken *before* removing artists so we can
          detect direct matplotlib changes since the last apply.
        - Only gerrytools-tracked artists are removed; external artists
          survive.
        - Autoscale-protected units are restored after artist drawing so
          matplotlib's autoscale cannot clobber externally set limits.
        - Ticks are applied after the legend so positions are consistent
          with any side effects from the legend draw.
        """
        before, external = self._axes_state.begin_rebuild(self._ax)

        self._artists.remove_all()
        if self._grid is not None:
            self._ax.grid(self._grid)
        self._build_plot()

        self._axes_state.restore_autoscale_protected(self._ax, before, external)
        self._apply_limits(self._xaxis, external)
        self._apply_limits(self._yaxis, external)
        self._apply_frame_visibility(external)
        self._apply_text(self._xaxis.label, external)
        self._apply_text(self._yaxis.label, external)
        self._apply_text(self._title_text, external)
        self._apply_scale(self._xaxis, external)
        self._apply_scale(self._yaxis, external)
        self._apply_aspect(external)
        self._apply_legend(external)

        # Apply ticks after the legend so positions reflect any legend-draw side effects on layout.
        self._apply_ticks(self._xaxis, external)
        self._apply_ticks(self._yaxis, external)
        self._apply_tick_style_unit(self._xaxis, external)
        self._apply_tick_style_unit(self._yaxis, external)

        # Annotations (arrows / vlines / hlines / bands) render last so their
        # canvas-draw-and-measure alignment passes (used for arrow tip placement) see the final
        # ticks, limits, and label layout.
        self._annotations.apply(self._ax, fig=self.fig, registry=self._artists)

        # Out-of-range annotations can autoscale the axes; re-reconcile limits, ticks and
        # tick styles so the recorded last-applied values match the final layout. Otherwise
        # the next rebuild would misread the autoscale shift as an external tick change.
        self._apply_ticks(self._xaxis, external)
        self._apply_ticks(self._yaxis, external)
        # Fixed tick locators may expand the view interval. Restore externally-owned limits,
        # then reapply explicit gerrytools limits so the most recent limit setter still wins.
        self._axes_state.restore_autoscale_protected(self._ax, before, external)
        self._apply_limits(self._xaxis, external)
        self._apply_limits(self._yaxis, external)
        self._apply_tick_style_unit(self._xaxis, external)
        self._apply_tick_style_unit(self._yaxis, external)

    @abstractmethod
    def _build_plot(self) -> None:
        """Build the plot by applying all settings and drawing elements."""
        pass  # pragma: no cover - abstract stub; every concrete subclass overrides this

    def _dataset_legend_handles(self) -> list[LegendHandle]:
        """Hook: legend handles for this plot's datasets; subclasses override."""
        return []

    def _pointset_legend_handles(self) -> list[LegendHandle]:
        """Hook: legend handles for this plot's point sets; subclasses override."""
        return []

    @property
    def _legend_handles(self) -> list[LegendHandle]:
        """Generated legend handles in the one shared ordering: datasets, then point sets,
        then named lines and bands."""
        handles: list[LegendHandle] = []
        handles.extend(self._dataset_legend_handles())
        handles.extend(self._pointset_legend_handles())
        handles.extend(self._get_named_line_legend_handles())
        handles.extend(self._get_named_band_legend_handles())
        return handles
