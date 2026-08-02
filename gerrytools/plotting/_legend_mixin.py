"""Shared legend behavior and ownership policy for axes-backed plot classes.

``_LegendMixin`` owns the legend handling shared by the data plots (``GerryPlotBase``) and
``DotDensityPlot``: storing ``LegendOptions``, the public ``set_legend_options`` API, the
named-add legend-ownership claim, standalone-legend saving, and the managed-unit
reconciliation (:meth:`_apply_legend`) that places, updates, or removes the in-axes legend
while respecting externally installed legends. Subclasses supply the handles via the
``_legend_handles`` hook and the enabled state via ``_legend_enabled``; each class defines
its own public ``save_legend`` (their signatures differ) over ``_save_legend_handles``.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any, Literal, Sequence, cast
from weakref import ReferenceType, WeakKeyDictionary, ref

from matplotlib.legend import Legend

from gerrytools.plotting._axes_backed import deferred_axis_update
from gerrytools.plotting._legend_utils import save_legend_handles
from gerrytools.plotting.mpl.legend_options import LegendAnchor, LegendOptions
from gerrytools.plotting.utils import UNSET, Unset, _replace_non_none
from gerrytools.typing import Color, LegendHandle

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from gerrytools.plotting._axes_state import Unit, _ManagedAxesState


class _LegendMixin:
    """Legend options storage, standalone saving, and managed in-axes placement.

    Pure capability mixin: it does not inherit ``_AxesBackedPlot``. The host class
    provides the axes-backed attributes declared below; every concrete plot base
    inherits ``_AxesBackedPlot`` exactly once and mixes this in beside it.
    """

    _legend_options: LegendOptions

    _placed_legends_by_axes: WeakKeyDictionary[Axes, ReferenceType[Legend]]

    if TYPE_CHECKING:
        fig: Figure
        _ax: Axes
        _axes_state: _ManagedAxesState

    @property
    def _last_placed_legend(self) -> Legend | None:
        """Legend last placed by this plot on its current axes, if any."""
        legends = getattr(self, "_placed_legends_by_axes", None)
        legend_ref = None if legends is None else legends.get(self._ax)
        return None if legend_ref is None else legend_ref()

    @_last_placed_legend.setter
    def _last_placed_legend(self, legend: Legend | None) -> None:
        legends = getattr(self, "_placed_legends_by_axes", None)
        if legends is None:
            if legend is None:
                return
            legends = WeakKeyDictionary()
            self._placed_legends_by_axes = legends
        if legend is None:
            legends.pop(self._ax, None)
        else:
            legends[self._ax] = ref(legend)

    @property
    def _legend_handles(self) -> list[LegendHandle]:
        """Hook: legend handles for this plot's named elements."""
        return []

    @property
    def _legend_enabled(self) -> bool:
        """Hook: whether this plot wants an in-axes legend on rebuild."""
        return True

    @deferred_axis_update
    def _claim_legend_if_named(self, name: str | None) -> None:
        """Reclaim the legend unit when an ``add_*`` is given a user-supplied name.

        Any ``add_*`` call with a non-None ``name`` reclaims the legend unit,
        because the resulting legend content is derived from named plot
        elements. ``add_*`` calls with ``name=None`` do not reclaim — they
        may still contribute to the legend handles (subclasses often fall
        back to an auto-generated label), but they should not displace an
        externally-placed legend.

        Named-add methods (``add_dataset``, ``add_pointset``, etc.) call this
        helper immediately after appending to their data list, and the base
        annotation helpers call it for named lines and bands.
        """
        if name is not None:
            self._axes_state.reclaim_without_value("legend")

    @deferred_axis_update
    def set_legend_options(
        self,
        options: LegendOptions | None = None,
        *,
        loc: str | int | None = None,
        bbox_to_anchor: LegendAnchor | None | Unset = UNSET,
        ncols: int | None = None,
        fontsize: float | str | None | Unset = UNSET,
        frameon: bool | None = None,
        fancybox: bool | None = None,
        shadow: bool | None = None,
        framealpha: float | None | Unset = UNSET,
        facecolor: Color | None | Unset = UNSET,
        edgecolor: Color | None | Unset = UNSET,
        title: str | None | Unset = UNSET,
        alignment: Literal["center", "left", "right"] | None = None,
        labelspacing: float | None = None,
        columnspacing: float | None = None,
    ) -> None:
        """Update the legend options used by ``Axes.legend`` during plot build.

        Kwargs merge over the currently stored options (or over ``options`` when given), so
        repeated calls accumulate: a second call only changes the fields it names and keeps
        everything set earlier. Pass a full ``LegendOptions`` as ``options`` to reset.
        Omitted kwargs inherit. For ``bbox_to_anchor``, ``fontsize``, ``framealpha``, and
        ``title``, an explicit ``None`` clears the field back to its matplotlib default.
        For ``facecolor`` and ``edgecolor``, an explicit ``None`` removes that color.

        Args:
            options (LegendOptions | None, optional): Pre-built options to start from,
                replacing the currently stored options before kwargs are applied.
                Defaults to None (merge over the current options).
            loc (str | int | None, optional): Matplotlib legend location. Defaults to None.
            bbox_to_anchor (LegendAnchor | None, optional): Legend anchor box, as ``(x, y)``
                or ``(x, y, width, height)``. ``None`` anchors at ``loc`` inside the axes.
                The stored default is ``(1.01, 0.5)``.
            ncols (int | None, optional): Number of legend columns. Defaults to None.
            fontsize (float | str | None, optional): Legend text size. Omit to inherit; pass None
                to use Matplotlib's default.
            frameon (bool | None, optional): Whether to draw the legend frame.
                Defaults to None.
            fancybox (bool | None, optional): Whether to use a rounded frame.
                Defaults to None.
            shadow (bool | None, optional): Whether to draw a shadow. Defaults to None.
            framealpha (float | None, optional): Frame alpha override. Omit to inherit; pass None
                to use Matplotlib's default.
            facecolor (Color | None, optional): Frame face color. Pass ``None`` for no
                fill.
            edgecolor (Color | None, optional): Frame edge color. Pass ``None`` for no
                edge.
            title (str | None, optional): Legend title. Omit to inherit; pass None to clear it.
            alignment (Literal["center", "left", "right"] | None, optional): Legend content
                alignment. Defaults to None.
            labelspacing (float | None, optional): Vertical spacing between entries.
                Defaults to None.
            columnspacing (float | None, optional): Horizontal spacing between columns.
                Defaults to None.

        Returns:
            None
        """
        base = options if options is not None else self._legend_options
        merged = _replace_non_none(
            base,
            loc=loc,
            ncols=ncols,
            frameon=frameon,
            fancybox=fancybox,
            shadow=shadow,
            alignment=alignment,
            labelspacing=labelspacing,
            columnspacing=columnspacing,
        )
        # None is a real value for these fields, so they use the UNSET sentinel instead of
        # the None-inherits convention above.
        sentinel_overrides = {
            name: value
            for name, value in (
                ("bbox_to_anchor", bbox_to_anchor),
                ("fontsize", fontsize),
                ("framealpha", framealpha),
                ("facecolor", facecolor),
                ("edgecolor", edgecolor),
                ("title", title),
            )
            if not isinstance(value, Unset)
        }
        if sentinel_overrides:
            merged = dataclasses.replace(merged, **sentinel_overrides)
        self._legend_options = merged
        # Legend identity is recorded by ``_apply_legend`` at the next rebuild
        # after the legend is re-placed with these new options.
        self._axes_state.reclaim_without_value("legend")

    def _save_legend_handles(
        self,
        handles: Sequence[LegendHandle],
        filepath: str,
        *,
        outer_padding: float = 0.07,
        dpi: int | None = None,
        **legend_kwargs: object,
    ) -> None:
        """Save the given handles as a standalone legend image using this plot's options."""
        save_legend_handles(
            handles=handles,
            legend_options=self._legend_options,
            filepath=filepath,
            outer_padding=outer_padding,
            dpi=dpi or self.fig.dpi,
            **legend_kwargs,
        )

    def _apply_legend(self, external: set[Unit]) -> None:
        """Place, update, or remove the legend per the managed-unit contract.

        Decision tree:
        - External legend present → skip entirely.
        - Legend disabled OR no handles: remove only a legend gerrytools itself
          placed; a legend gerrytools never placed is left alone.
        - Enabled and handles exist → place and record identity: explicit when a
          public setter reclaimed the unit, default for auto-placement.
        """
        if "legend" in external:
            return
        is_reclaimed = self._axes_state.is_reclaimed("legend")
        handles = self._legend_handles if self._legend_enabled else []
        if not handles:
            current = self._ax.get_legend()
            if current is not None:
                if current is not self._last_placed_legend:
                    # A legend gerrytools never placed (e.g. user-installed under a
                    # store-and-claim with zero handles) must survive.
                    return
                current.remove()
                self._last_placed_legend = None
            if is_reclaimed:
                self._axes_state.reclaim_and_mark("legend", None)
            return
        current = self._ax.get_legend()
        if current is not None and current is not self._last_placed_legend and not is_reclaimed:
            # External-change detection skips unclaimed units before the first build.
            return
        # A prior gerrytools legend on this axes is removed so the new one supersedes it
        # (matplotlib only renders one legend slot per axes).
        new_legend = self._place_legend(handles, remove_prior=is_reclaimed)
        self._last_placed_legend = new_legend
        if is_reclaimed:
            self._axes_state.reclaim_and_mark("legend", new_legend)
        else:
            # Auto-placed with no user legend request involved: record without claiming.
            self._axes_state.record_default("legend", new_legend)

    def _place_legend(
        self,
        handles: Sequence[LegendHandle],
        *,
        remove_prior: bool = False,
    ) -> Legend | None:
        """Place an in-axes legend using the stored options.

        Prior-legend removal is an ownership decision made by :meth:`_apply_legend`;
        this method never removes a legend unless ``remove_prior`` is passed. The
        legend is deliberately not tracked in the artist registry: removing a stale
        tracked legend after the user installed their own would null ``ax.legend_``
        (matplotlib's remove hook clears the slot unconditionally), destroying the
        external legend.

        Args:
            handles (Sequence[LegendHandle]): Handles to render.
            remove_prior (bool, optional): Whether to remove an existing legend on the axes
                first (matplotlib renders only one legend slot per axes). Defaults to False.

        Returns:
            Legend | None: The placed legend, if matplotlib created one.
        """
        if remove_prior:
            prior = self._ax.get_legend()
            if prior is not None:
                prior.remove()
        self._ax.legend(handles=handles, **cast("Any", self._legend_options.to_dict()))
        return self._ax.get_legend()
