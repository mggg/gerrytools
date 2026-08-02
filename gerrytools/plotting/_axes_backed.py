"""Shared figure-ownership lifecycle for axes-backed plot classes.

``_AxesBackedPlot`` owns the concerns every lazily rendered plot class shares: adopting a
caller-owned axes or creating an owned figure (with a close finalizer and Jupyter display
suppression), the non-destructive ``bind_to_ax`` rebind, and the build entry points
(``bind_to_ax``, ``ax``, ``show``, ``save``). Subclasses keep their own accumulated state and
implement ``_build_and_apply_settings`` to render it.
"""

from __future__ import annotations

import weakref
from collections.abc import Callable
from functools import wraps
from typing import TypeVar, cast

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from gerrytools._ipython import in_jupyter_kernel
from gerrytools.plotting._artist_registry import _ArtistRegistry
from gerrytools.plotting._axes_state import _ManagedAxesState
from gerrytools.plotting._figure_io import save_figure, show_figure

MethodT = TypeVar("MethodT", bound=Callable[..., object])


def deferred_axis_update(method: MethodT) -> MethodT:
    """Mark the plot's axes for a deferred update before a mutation."""

    @wraps(method)
    def wrapper(self: _AxesBackedPlot, *args: object, **kwargs: object) -> object:
        self._axis_needs_update = True
        return method(self, *args, **kwargs)

    return cast(MethodT, wrapper)


class _AxesBackedPlot:
    """Figure ownership, rebinding, and lazy-build entry points for plot classes.

    Subclasses call :meth:`_attach_axes` during construction, after storing the remembered
    figure geometry (``_figure_size``/``_figure_dpi``) and emitting any ignored-argument
    warnings for their own signature, and implement :meth:`_build_and_apply_settings`.
    """

    fig: Figure
    _ax: Axes
    _finalizer: weakref.finalize | None
    _figure_is_shared: bool
    _artists: _ArtistRegistry
    _artist_histories: dict[Axes, _ArtistRegistry]
    _axes_state: _ManagedAxesState
    _axis_needs_update: bool

    # Remembered construction geometry, used to recreate an owned figure on unbind. A ``None``
    # figure size lets matplotlib choose (the geometry plots size to their data instead).
    _figure_size: tuple[float, float] | None
    _figure_dpi: int

    # Fallback file naming used by ``show()`` outside notebooks and GUI backends.
    _non_gui_filename: str = "gerrytools_plot.png"
    _non_gui_prefix: str = "GerryTools Plotting"

    def _attach_axes(self, ax: Axes | None) -> None:
        """Adopt a caller-owned axes, or create and own a fresh figure.

        Owned figures register a finalizer so pyplot's figure manager cannot keep them alive
        forever, and are closed immediately in Jupyter so construction and rebinds never
        auto-display. A caller-supplied axes leaves figure lifetime with the caller: no
        finalizer is registered, and any previously registered one is detached. The exception
        is an axes on the plot's own owned figure, which keeps its finalizer and ownership.

        Rebinding away from an owned figure closes it: nothing else manages that figure,
        and pyplot's manager would otherwise hold it open forever. An external figure is
        left alone.
        """
        old_fig = getattr(self, "fig", None)
        old_owned = old_fig is not None and not getattr(self, "_figure_is_shared", True)
        if old_owned and (ax is None or ax.figure is not old_fig):
            old_finalizer = getattr(self, "_finalizer", None)
            if old_finalizer is not None:
                old_finalizer.detach()
                self._finalizer = None
            plt.close(old_fig)
        if ax is None:
            if self._figure_size is None:
                self.fig, self._ax = plt.subplots(dpi=self._figure_dpi)
            else:
                self.fig, self._ax = plt.subplots(figsize=self._figure_size, dpi=self._figure_dpi)
            self._figure_is_shared = False
            if in_jupyter_kernel():  # pragma: no cover - only reachable in a live Jupyter kernel
                plt.close(self.fig)  # pragma: no cover
            self._finalizer = weakref.finalize(self, plt.close, self.fig)
        else:
            new_fig = cast(Figure, ax.get_figure(root=True))
            # An axes on the plot's own owned figure keeps ownership; detaching the
            # finalizer here would leak the figure in pyplot's manager forever.
            if not (old_owned and new_fig is old_fig):
                self._figure_is_shared = True
                existing_finalizer = getattr(self, "_finalizer", None)
                if existing_finalizer is not None:
                    existing_finalizer.detach()
                self._finalizer = None
            self.fig = new_fig
            self._ax = ax
        self._axis_needs_update = True

    def bind_to_ax(self, ax: Axes | None) -> None:
        """Retarget this plot to render onto a different matplotlib ``Axes``.

        The plot's accumulated state (added data, layers, lines, labels, styles, etc.) is
        preserved and immediately applied to the new axes. Prior rendered output on an
        *external* old axes is left alone; this plot simply stops managing it. An old figure this
        plot created itself is closed, since nothing else keeps it alive.

        Pass ``ax=None`` to unbind: the plot creates a fresh figure for the next render, just as it
        did on construction.

        Args:
            ax (matplotlib.axes.Axes | None): The matplotlib axes to render onto, or ``None``
                to revert to a fresh-figure render.
        """
        old_ax = self._ax
        same_axes = ax is old_ax
        if not same_axes:
            self._axes_state.remember_axes(old_ax)
            histories = getattr(self, "_artist_histories", {})
            self._artist_histories = histories
            # Closed owned figures cannot be revisited. External axes and other axes on the
            # current owned figure can, so retain their managed artists until a later rebind.
            if self._figure_is_shared or (ax is not None and ax.figure is self.fig):
                histories[old_ax] = self._artists
        self._attach_axes(ax)

        if same_axes:
            self._artists.remove_all()
        else:
            self._artists = self._artist_histories.pop(self._ax, _ArtistRegistry())
            # Rebinding is non-destructive when leaving an axes. If that axes is revisited,
            # replace this plot's earlier rendering before drawing its current state.
            self._artists.remove_all()
            self._axes_state.restore_axes(self._ax)
            self._axes_state.initialize_from_ax(self._ax)
        if ax is not None:
            self._update_axis()

    @property
    def ax(self) -> Axes:
        """Build the plot and return the matplotlib ``Axes``.

        Access to this property triggers a **lazy render** after the plot changes: every
        accumulated setting (added data, layers, lines, labels, styles, etc.) is reapplied to
        the underlying axes. Repeated access without an intervening change returns the current
        axes directly.

        Why lazy? In a Jupyter notebook, instantiating a plot class without lazy rendering
        would auto-display an empty figure as the cell output. Deferring the render until
        ``.ax`` (or :meth:`show` / :meth:`save`) is accessed keeps the notebook clean.

        Calling ``.ax`` multiple times is safe. Use :meth:`bind_to_ax` to retarget the plot to a
        different ``Axes`` (e.g. one inside your own figure).

        Returns:
            Axes: The matplotlib ``Axes`` object with every setting applied.
        """
        self._update_axis()
        return self._ax

    def show(self, **kwargs: object) -> None:
        """Display inline in notebooks, or open a GUI window in scripts.

        Args:
            **kwargs (object): Additional keyword arguments passed to ``Figure.savefig``.
                Defaults: ``bbox_inches="tight"``, ``dpi=fig.dpi``.
        """
        self._update_axis()
        show_figure(
            self.fig,
            non_gui_filename=self._non_gui_filename,
            non_gui_prefix=self._non_gui_prefix,
            **kwargs,
        )

    def save(self, filepath: str, **kwargs: object) -> None:
        """Save the figure to a file.

        Args:
            filepath (str): Output file path.
            **kwargs (object): Additional keyword arguments passed to ``Figure.savefig``.
        """
        self._update_axis()
        save_figure(self.fig, filepath, **kwargs)

    def _update_axis(self) -> None:
        """Apply deferred plot changes once, leaving failed builds pending."""
        if not self._axis_needs_update:
            return
        self._build_and_apply_settings()
        self._axis_needs_update = False

    def _build_and_apply_settings(self) -> object:
        """Rebuild the plot from its accumulated state onto the current axes."""
        raise NotImplementedError
