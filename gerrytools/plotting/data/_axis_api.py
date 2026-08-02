"""Managed title and statistical-axis configuration.

Internal module. `_AxisState` holds one axis's managed configuration (tick locations/labels,
limits, tick style, axis label, managed-unit ids) plus thin accessors that dispatch to the
right matplotlib x/y method. `_ManagedText` is one managed text slot, with the subclasses
`_AxisLabelText` and `_TitleText` binding a slot to its matplotlib setter. `_TitleApiMixin`
is shared by statistical and geographic plots. `_AxisApiMixin` adds the statistical
axis-configuration API and rebuild-time helpers.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Generic, Literal, Sequence, TypeVar

import matplotlib
from matplotlib.axes import Axes
from matplotlib.text import Text

from gerrytools.colors import resolve_rgba
from gerrytools.plotting._axes_backed import deferred_axis_update
from gerrytools.plotting._axes_state import (
    Unit,
    _label_snapshot,
    _LabelSnapshot,
    _ManagedAxesState,
    _tick_snapshot,
    _tick_style_snapshot,
    _title_snapshot,
    _TitleSnapshot,
)
from gerrytools.plotting.mpl.axis_title_style import AxisLabelStyle, TitleStyle
from gerrytools.plotting.mpl.tick_style import TickStyle
from gerrytools.plotting.utils import UNSET, Unset
from gerrytools.typing import Color, TickType

StyleT = TypeVar("StyleT", AxisLabelStyle, TitleStyle)


def _resolve_tick_label_update(
    current_locations: list[float] | None,
    current_labels: list[str] | None,
    *,
    locations: list[float] | None,
    labels: list[str] | None,
) -> tuple[list[float] | None, list[str] | None]:
    """Validate a tick locations/labels update and return the new field pair.

    Shared by ``set_xticks`` and ``set_yticks``: the update rules (length agreement, partial
    updates against existing values) are axis-independent. Returns the new
    ``(locations, labels)`` values, where an entry not touched by the update keeps its
    current value.

    Raises:
        ValueError: If the lengths of provided locations and labels do not match each other
            or the existing values.
    """
    if locations is not None and labels is not None:
        if len(locations) != len(labels):
            raise ValueError(
                f"Locations length {len(locations)} does not match labels length {len(labels)}."
            )
        return list(locations), list(labels)

    if locations is not None:
        if current_locations == [] and current_labels == [] and locations:
            current_labels = None
        if (
            current_labels is not None
            and current_labels != []
            and locations != []
            and len(locations) != len(current_labels)
        ):
            raise ValueError(
                f"Locations length {len(locations)} does not match existing labels length "
                f"{len(current_labels)}."
            )
        if locations == []:
            return [], []
        return list(locations), current_labels

    if labels is not None:
        if labels == []:
            return current_locations, []
        if current_locations is not None and len(labels) != len(current_locations):
            raise ValueError(
                f"Labels length {len(labels)} does not match existing locations length "
                f"{len(current_locations)}."
            )
        return current_locations, list(labels)

    return current_locations, current_labels


class _ManagedText(Generic[StyleT]):
    """One managed text slot (an axis label or the title).

    Holds the user-set text (``UNSET`` meaning "no opinion", ``None`` meaning explicit clear),
    the optional style, and the managed-unit id used for ownership bookkeeping. Subclasses
    bind the slot to its matplotlib setter.
    """

    def __init__(self, *, unit: Unit) -> None:
        self.text: str | None | Unset = UNSET
        self.style: StyleT | None = None
        self.unit: Unit = unit

    @property
    def value(self) -> str | None:
        """User-visible text; ``UNSET`` reads as None."""
        return None if isinstance(self.text, Unset) else self.text

    def apply_now(self, ax: Axes) -> None:
        """Write the text (and style, when set) to the axes.

        ``UNSET`` means "no opinion": leave any pre-set external text alone. A real ``None``
        is the explicit-clear path (user code set the property to None after construction);
        that writes the empty string.
        """
        if isinstance(self.text, Unset):
            return
        self._write(ax, "" if self.text is None else self.text)

    def _write(self, ax: Axes, text: str) -> None:
        """Write text and style through the slot's matplotlib setter."""
        raise NotImplementedError

    def clear_style_now(self, ax: Axes, text: str) -> None:
        """Write text using Matplotlib's current default style."""
        self._write(ax, text)

    def snapshot(self, ax: Axes) -> _LabelSnapshot | _TitleSnapshot:
        """Read the slot's current on-axes state for ownership bookkeeping."""
        raise NotImplementedError


class _AxisLabelText(_ManagedText[AxisLabelStyle]):
    """Managed text slot for an x- or y-axis label."""

    def __init__(self, *, unit: Unit, axis: Literal["x", "y"]) -> None:
        super().__init__(unit=unit)
        self._axis: Literal["x", "y"] = axis

    def _write(self, ax: Axes, text: str) -> None:
        setter = ax.set_xlabel if self._axis == "x" else ax.set_ylabel
        if self.style is not None:
            setter(text, **self.style.to_mpl_settings_dict())
        else:
            setter(text)

    def clear_style_now(self, ax: Axes, text: str) -> None:
        setter = ax.set_xlabel if self._axis == "x" else ax.set_ylabel
        color = matplotlib.rcParams["axes.labelcolor"]
        if color == "inherit":
            color = matplotlib.rcParams["text.color"]
        setter(
            text,
            fontsize=matplotlib.rcParams["axes.labelsize"],
            fontweight=matplotlib.rcParams["axes.labelweight"],
            fontstyle=matplotlib.rcParams["font.style"],
            fontfamily=matplotlib.rcParams["font.family"],
            color=color,
            alpha=None,
            labelpad=matplotlib.rcParams["axes.labelpad"],
        )

    def snapshot(self, ax: Axes) -> _LabelSnapshot:
        return _label_snapshot(ax, self._axis)


class _TitleText(_ManagedText[TitleStyle]):
    """Managed text slot for the axes title."""

    def _write(self, ax: Axes, text: str) -> None:
        for location in ("left", "center", "right"):
            ax.set_title("", loc=location)
        if self.style is not None:
            ax.set_title(text, **self.style.to_mpl_settings_dict())
        else:
            ax.set_title(text)

    def clear_style_now(self, ax: Axes, text: str) -> None:
        for location in ("left", "center", "right"):
            ax.set_title("", loc=location)
        color = matplotlib.rcParams["axes.titlecolor"]
        if color == "auto":
            color = matplotlib.rcParams["text.color"]
        ax.set_title(
            text,
            loc=matplotlib.rcParams["axes.titlelocation"],
            fontsize=matplotlib.rcParams["axes.titlesize"],
            fontweight=matplotlib.rcParams["axes.titleweight"],
            fontstyle=matplotlib.rcParams["font.style"],
            fontfamily=matplotlib.rcParams["font.family"],
            color=color,
            alpha=None,
            pad=matplotlib.rcParams["axes.titlepad"],
        )

    def snapshot(self, ax: Axes) -> _TitleSnapshot:
        return _title_snapshot(ax)


class _AxisState:
    """One axis's managed configuration plus x/y-dispatching matplotlib accessors."""

    def __init__(self, axis: Literal["x", "y"]) -> None:
        self.axis: Literal["x", "y"] = axis
        self.tick_locations: list[float] | None = None
        self.tick_labels: list[str] | None = None
        self.limits: tuple[float, float] | None = None
        self.tick_style: TickStyle | None = None
        # Whether this plot's last tick apply hid the labels via tick_params. Hiding is a
        # sticky axes-level switch, so a later apply that intends visible labels must flip
        # it back; the flag keeps that re-enable from stomping hides this plot never made.
        self.labels_hidden: bool = False
        is_x = axis == "x"
        self.label = _AxisLabelText(unit="x_label" if is_x else "y_label", axis=axis)
        self.unit_ticks: Unit = "x_ticks" if is_x else "y_ticks"
        self.unit_tick_style: Unit = "x_tick_style" if is_x else "y_tick_style"
        self.unit_limits: Unit = "x_limits" if is_x else "y_limits"
        self.unit_scale: Unit = "x_scale" if is_x else "y_scale"

    # -- matplotlib accessors, dispatched on self.axis -----------------

    def set_ticks(self, ax: Axes, locations: list[float]) -> None:
        (ax.set_xticks if self.axis == "x" else ax.set_yticks)(locations)

    def set_ticklabels(self, ax: Axes, labels: list[str]) -> None:
        (ax.set_xticklabels if self.axis == "x" else ax.set_yticklabels)(labels)

    def get_ticks(self, ax: Axes) -> list[float]:
        ticks = ax.get_xticks() if self.axis == "x" else ax.get_yticks()
        return [float(tick) for tick in ticks]

    def hide_tick_labels(self, ax: Axes) -> None:
        if self.axis == "x":
            ax.tick_params(axis="x", labelbottom=False)
        else:
            ax.tick_params(axis="y", labelleft=False)
        self.labels_hidden = True

    def show_tick_labels(self, ax: Axes) -> None:
        if self.axis == "x":
            ax.tick_params(axis="x", labelbottom=True)
        else:
            ax.tick_params(axis="y", labelleft=True)
        self.labels_hidden = False

    def set_lim(self, ax: Axes, low: float, high: float) -> None:
        if self.axis == "x":
            ax.set_xlim(low, high)
        else:
            ax.set_ylim(low, high)

    def get_lim(self, ax: Axes) -> tuple[float, float]:
        low, high = ax.get_xlim() if self.axis == "x" else ax.get_ylim()
        return (float(low), float(high))

    def get_scale(self, ax: Axes) -> str:
        return ax.get_xscale() if self.axis == "x" else ax.get_yscale()


class _TitleApiMixin:
    """Managed plot-title API shared by data and geographic plots."""

    _ax: Axes
    _axes_state: _ManagedAxesState
    _title_text: _TitleText

    @property
    def title(self) -> str | None:
        """Plot title, or None if unset."""
        return self._title_text.value

    @title.setter
    def title(self, value: str | None) -> None:
        self._set_text(self._title_text, value)

    @deferred_axis_update
    def _set_text(self, slot: _ManagedText[StyleT], value: str | None) -> None:
        """Set a managed text slot's value, apply it, and reclaim its unit."""
        slot.text = value
        slot.apply_now(self._ax)
        self._axes_state.reclaim_and_mark(slot.unit, slot.snapshot(self._ax))

    def _apply_text(self, slot: _ManagedText[StyleT], external: set[Unit]) -> None:
        """Rebuild-time applier: skip when gerrytools has no opinion, reconcile otherwise."""
        if isinstance(slot.text, Unset):
            return
        self._axes_state.reconcile(
            slot.unit,
            external,
            lambda: slot.apply_now(self._ax),
            lambda: slot.snapshot(self._ax),
        )

    @deferred_axis_update
    def _set_text_style(self, slot: _ManagedText[StyleT], style: StyleT | None) -> None:
        """Install (or clear, with ``style=None``) a managed text slot's style.

        With text already set, the atomic text+style unit is reclaimed and applied
        immediately so future ax-level edits are detected as external. Style without text
        claims the unit so the pair applies on the next text assignment; the concrete value
        is recorded then.
        """
        slot.style = style
        if slot.value is not None:
            text = slot.value
            assert text is not None
            if style is None:
                slot.clear_style_now(self._ax, text)
            else:
                slot.apply_now(self._ax)
            self._axes_state.reclaim_and_mark(slot.unit, slot.snapshot(self._ax))
        elif style is not None:
            self._axes_state.reclaim_without_value(slot.unit)

    def set_title_style(
        self,
        *,
        fontsize: float | int | None = None,
        fontweight: str | None = None,
        fontstyle: Literal["normal", "italic", "oblique"] | None = None,
        fontfamily: str | None = None,
        fontcolor: Color | None = "black",
        fontalpha: float | None = None,
        loc: Literal["left", "center", "right"] | None = None,
        pad: float | None = None,
    ) -> None:
        """Set the styling for the axes title.

        Args:
            fontsize (float | int | None, optional): Font size for the title. Defaults to None.
            fontweight (str | None, optional): Font weight (e.g., "normal", "bold").
                Defaults to None.
            fontstyle (Literal["normal", "italic", "oblique"] | None, optional):
                Font style (e.g., "normal", "italic"). Defaults to None.
            fontfamily (str | None, optional): Font family (e.g., "sans-serif", "serif").
                Defaults to None.
            fontcolor (Color | None, optional): Color of the title. Pass ``None`` for
                transparent text. Defaults to "black".
            fontalpha (float | None, optional): Alpha transparency of the title color.
                If None, uses alpha from color if specified. Defaults to None.
            loc (Literal["left", "center", "right"] | None, optional): Title location.
                Defaults to None.
            pad (float | None, optional): Padding between the title and the axes in points.
                Defaults to None.

        Returns:
            None
        """
        self._set_text_style(
            self._title_text,
            TitleStyle(
                fontsize=fontsize,
                fontweight=fontweight,
                fontstyle=fontstyle,
                fontfamily=fontfamily,
                fontcolor=fontcolor,
                fontalpha=fontalpha,
                loc=loc,
                pad=pad,
            ),
        )

    def clear_title_style(self) -> None:
        """Clear the plot title styling, reverting to Matplotlib defaults."""
        self._set_text_style(self._title_text, None)


class _AxisApiMixin(_TitleApiMixin):
    """Axis/tick/label/limit/style configuration API for ``GerryPlotBase``.

    Operates on the two ``_AxisState`` objects and the ``_TitleText`` title slot created by
    the ``GerryPlotBase`` constructor; the attribute annotations below name that contract.
    """

    _xaxis: _AxisState
    _yaxis: _AxisState

    # ------------------------------------------------------------------
    # Managed text slots: properties, style setters, and apply helpers
    # ------------------------------------------------------------------

    @property
    def xlabel(self) -> str | None:
        """X-axis label text, or None if unset."""
        return self._xaxis.label.value

    @xlabel.setter
    def xlabel(self, value: str | None) -> None:
        self._set_text(self._xaxis.label, value)

    @property
    def ylabel(self) -> str | None:
        """Y-axis label text, or None if unset."""
        return self._yaxis.label.value

    @ylabel.setter
    def ylabel(self, value: str | None) -> None:
        self._set_text(self._yaxis.label, value)

    def set_axis_label_style(
        self,
        axis: Literal["x", "y"],
        *,
        fontsize: float | int | None = None,
        fontweight: str | None = None,
        fontstyle: Literal["normal", "italic", "oblique"] | None = None,
        fontfamily: str | None = None,
        fontcolor: Color | None = "black",
        fontalpha: float | None = None,
        labelpad: float | None = None,
    ) -> None:
        """Sets the styling for the label of one axis.

        Args:
            axis (Literal["x", "y"]): Which axis label to style.
            fontsize (float | int | None, optional): Font size for the axis label.
                Defaults to None.
            fontweight (str | None, optional): Font weight (e.g., "normal", "bold").
                Defaults to None.
            fontstyle (Literal["normal", "italic", "oblique"] | None, optional):
                Font style (e.g., "normal", "italic").
                Defaults to None.
            fontfamily (str | None, optional): Font family (e.g., "sans-serif", "serif").
                Defaults to None.
            fontcolor (Color | None, optional): Color of the axis label. Pass ``None``
                for transparent text. Defaults to "black".
            fontalpha (float | None, optional): Alpha transparency of the axis label color.
                If None, uses alpha from color if specified. Defaults to None.
            labelpad (float | None, optional): Padding between the axis label and the axis
                in points. Defaults to None.

        Returns:
            None
        """
        self._set_text_style(
            (self._xaxis if axis == "x" else self._yaxis).label,
            AxisLabelStyle(
                fontsize=fontsize,
                fontweight=fontweight,
                fontstyle=fontstyle,
                fontfamily=fontfamily,
                fontcolor=fontcolor,
                fontalpha=fontalpha,
                labelpad=labelpad,
            ),
        )

    def clear_xlabel_style(self) -> None:
        """Clear the x-axis label styling, reverting to matplotlib defaults."""
        self._set_text_style(self._xaxis.label, None)

    def clear_ylabel_style(self) -> None:
        """Clear the y-axis label styling, reverting to matplotlib defaults."""
        self._set_text_style(self._yaxis.label, None)

    # ------------------------------------------------------------------
    # Tick locations/labels
    # ------------------------------------------------------------------

    @deferred_axis_update
    def _update_tick_labels(
        self,
        axis: _AxisState,
        *,
        locations: list[float] | None,
        labels: list[str] | None,
    ) -> None:
        """Validate and store one axis's tick locations/labels update, claiming its unit."""
        if locations is None and labels is None:
            return
        axis.tick_locations, axis.tick_labels = _resolve_tick_label_update(
            axis.tick_locations,
            axis.tick_labels,
            locations=locations,
            labels=labels,
        )
        self._axes_state.reclaim_without_value(axis.unit_ticks)

    def set_xticks(
        self,
        locations: Sequence[float] | None = None,
        *,
        labels: Sequence[str] | None = None,
    ) -> None:
        """Set x-axis tick locations and/or labels.

        Passing only one of ``locations`` / ``labels`` updates that side against the
        existing values. An empty ``locations`` clears both tick locations and labels;
        an empty ``labels`` hides the labels while keeping the locations.

        Args:
            locations (Sequence[float] | None, optional): X-axis tick locations. Defaults to None.
            labels (Sequence[str] | None, optional): X-axis tick labels. Defaults to None.

        Raises:
            ValueError: If the lengths of provided locations and labels do not match each
                other or the existing values.

        Returns:
            None
        """
        self._update_tick_labels(
            self._xaxis,
            locations=None if locations is None else list(locations),
            labels=None if labels is None else list(labels),
        )

    def set_yticks(
        self,
        locations: Sequence[float] | None = None,
        *,
        labels: Sequence[str] | None = None,
    ) -> None:
        """Set y-axis tick locations and/or labels.

        Passing only one of ``locations`` / ``labels`` updates that side against the
        existing values. An empty ``locations`` clears both tick locations and labels;
        an empty ``labels`` hides the labels while keeping the locations.

        Args:
            locations (Sequence[float] | None, optional): Y-axis tick locations. Defaults to None.
            labels (Sequence[str] | None, optional): Y-axis tick labels. Defaults to None.

        Raises:
            ValueError: If the lengths of provided locations and labels do not match each
                other or the existing values.

        Returns:
            None
        """
        self._update_tick_labels(
            self._yaxis,
            locations=None if locations is None else list(locations),
            labels=None if labels is None else list(labels),
        )

    def _default_x_tick_locations(self) -> list[float] | None:
        """Get subclass-provided default x-tick locations.

        Returns:
            list[float] | None: Default x-tick locations, or None to keep Matplotlib defaults.
        """
        return None

    def _default_x_tick_labels(self, tick_locations: list[float]) -> list[str] | None:
        """Get subclass-provided default x-tick labels.

        Args:
            tick_locations (list[float]): Final x-tick locations selected for the axes.

        Returns:
            list[str] | None: Tick labels aligned to ``tick_locations``, or None to keep current
                labels.
        """
        return None

    def _apply_ticks(self, axis: _AxisState, external: set[Unit]) -> None:
        """Reconcile one axis's ticks unit: apply explicit or subclass-default ticks."""
        self._axes_state.reconcile(
            axis.unit_ticks,
            external,
            lambda: self._apply_ticks_now(axis),
            lambda: _tick_snapshot(self._ax, axis.axis),
        )

    def _apply_ticks_now(self, axis: _AxisState) -> None:
        """Write explicit or subclass-default tick locations/labels for one axis."""
        explicit_labels = axis.tick_labels
        # Determine locations: explicit user set, subclass default (x only), or leave matplotlib's
        # locator alone. Materializing get_ticks() into a set_ticks() call would convert
        # matplotlib's dynamic locator into a FixedLocator that persists across rebuilds.
        if axis.tick_locations is not None:
            tick_locations: list[float] | None = list(axis.tick_locations)
        elif axis.axis == "x":
            default_locations = self._default_x_tick_locations()
            tick_locations = list(default_locations) if default_locations is not None else None
        else:
            tick_locations = None
        if tick_locations is not None:
            axis.set_ticks(self._ax, tick_locations)
        if explicit_labels == [] and tick_locations != []:
            axis.hide_tick_labels(self._ax)
            return
        # Every remaining branch intends visible labels (explicit labels, subclass defaults,
        # or matplotlib's own), so undo a hide this plot applied on an earlier build.
        if axis.labels_hidden:
            axis.show_tick_labels(self._ax)
        if explicit_labels is None:
            if tick_locations is None:
                # Matplotlib will compute labels from the locator at draw time; nothing to apply.
                return
            if axis.axis == "x":
                default_labels = self._default_x_tick_labels(tick_locations)
                if default_labels is not None:
                    axis.set_ticklabels(self._ax, list(default_labels))
            return
        if tick_locations is None:
            tick_locations = axis.get_ticks(self._ax)
            axis.set_ticks(self._ax, tick_locations)
        if len(explicit_labels) != len(tick_locations):
            raise ValueError(
                f"Expected {len(tick_locations)} {axis.axis} tick labels, "
                f"got {len(explicit_labels)}."
            )
        axis.set_ticklabels(self._ax, list(explicit_labels))

    # ------------------------------------------------------------------
    # Limits and scale
    # ------------------------------------------------------------------

    @deferred_axis_update
    def _set_lim(self, axis: _AxisState, low: float, high: float) -> None:
        """Store and apply one axis's limits, reclaiming its unit.

        Apply-now: matplotlib resolves a limits pair against any axes immediately. Record the
        getter-read value so a later external set_xlim/set_ylim is detected as a difference.
        """
        axis.limits = (float(low), float(high))
        axis.set_lim(self._ax, *axis.limits)
        self._axes_state.reclaim_and_mark(axis.unit_limits, axis.get_lim(self._ax))

    def set_xlim(self, left: float, right: float) -> None:
        """Set x-axis limits.

        Matches the matplotlib convention ``Axes.set_xlim(left, right)``.

        Args:
            left (float): Left x-axis limit.
            right (float): Right x-axis limit.
        """
        self._set_lim(self._xaxis, left, right)

    def set_ylim(self, bottom: float, top: float) -> None:
        """Set y-axis limits.

        Matches the matplotlib convention ``Axes.set_ylim(bottom, top)``.

        Args:
            bottom (float): Bottom y-axis limit.
            top (float): Top y-axis limit.
        """
        self._set_lim(self._yaxis, bottom, top)

    def _apply_limits(self, axis: _AxisState, external: set[Unit]) -> None:
        """Reconcile one axis's limits unit.

        Without stored limits the apply step is a no-op: matplotlib's autoscale has set the
        data range, and the getter-read value is recorded as a gerrytools default.
        """

        def apply_stored_limits() -> None:
            if axis.limits is not None:
                axis.set_lim(self._ax, *axis.limits)

        self._axes_state.reconcile(
            axis.unit_limits, external, apply_stored_limits, lambda: axis.get_lim(self._ax)
        )

    def _apply_scale(self, axis: _AxisState, external: set[Unit]) -> None:
        """Reconcile one axis's scale managed unit.

        Gerrytools data plots do not currently set a matplotlib axis scale. The apply step is a
        no-op and records the current scale as the default used to detect later external changes.
        """
        self._axes_state.reconcile(
            axis.unit_scale, external, lambda: None, lambda: axis.get_scale(self._ax)
        )

    # ------------------------------------------------------------------
    # Tick style
    # ------------------------------------------------------------------

    @deferred_axis_update
    def set_tick_style(
        self,
        axis: Literal["x", "y"],
        *,
        size: float | int = 10,
        rotation: float | int = 0,
        fontcolor: Color | None = "black",
        fontalpha: float | None = None,
        tickcolor: Color | None = "black",
        tickalpha: float | None = None,
        fontweight: str = "normal",
        fontstyle: Literal["normal", "italic", "oblique"] = "normal",
        fontfamily: str = "sans-serif",
        ticktype: TickType = "major",
    ) -> None:
        """Set the tick style for one axis.

        Args:
            axis (Literal["x", "y"]): Which axis's ticks to style.
            size (float, optional): Font size of tick labels. Defaults to 10.
            rotation (float | int, optional): Rotation angle of tick labels in degrees.
                Defaults to 0.
            fontcolor (Color | None, optional): Color of tick labels. Pass ``None`` for
                transparent labels. Defaults to "black".
            fontalpha (float, optional): Alpha transparency of tick label color. If None,
                uses alpha from color if specified or will fall back to 1.0. Defaults to None.
            tickcolor (Color | None, optional): Color of tick marks. Pass ``None`` for
                transparent marks. Defaults to "black".
            tickalpha (float, optional): Alpha transparency of tick mark color. If None,
                uses alpha from color if specified or will fall back to 1.0. Defaults to None.
            fontweight (str, optional): Font weight of tick labels (e.g., 'normal 'bold').
                Defaults to "normal".
            fontstyle (Literal["normal", "italic", "oblique"], optional): Font style of tick
                labels (e.g., 'normal', 'italic'). Defaults to "normal".
            fontfamily (str, optional): Font family of tick labels (e.g., 'serif', 'sans-serif').
                Defaults to "sans-serif".
            ticktype (TickType, optional): Type of ticks to style ('major', 'minor', 'both').
                Defaults to 'major'.

        Returns:
            None
        """
        axis_state = self._xaxis if axis == "x" else self._yaxis
        # Store-and-claim: tick style depends on the rebuild flow having
        # already laid out ticks (so label-text artists exist).
        axis_state.tick_style = TickStyle(
            size=size,
            rotation=rotation,
            fontcolor=fontcolor,
            fontalpha=fontalpha,
            tickcolor=tickcolor,
            tickalpha=tickalpha,
            fontweight=fontweight,
            fontstyle=fontstyle,
            fontfamily=fontfamily,
            ticktype=ticktype,
        )
        self._axes_state.reclaim_without_value(axis_state.unit_tick_style)

    def _apply_tick_style_unit(self, axis: _AxisState, external: set[Unit]) -> None:
        """Reconcile one axis's tick-style unit; no stored style means no opinion."""
        style = axis.tick_style
        if style is None:
            return
        # Snapshot the tick set(s) the stored style targets, so external changes to minor
        # ticks are detected when ticktype is "minor" or "both".
        self._axes_state.reconcile(
            axis.unit_tick_style,
            external,
            lambda: self._apply_tick_style(axis.axis, style),
            lambda: _tick_style_snapshot(self._ax, axis.axis, style.ticktype),
        )

    @staticmethod
    def _apply_ticklabel_textprops(
        labels: Iterable[Text],
        *,
        fontweight: str | None = None,
        fontstyle: Literal["normal", "italic", "oblique"] | None = None,
        fontfamily: str | None = None,
    ) -> None:
        """Apply text properties to tick labels.

        Args:
            labels (Iterable[Text]): Iterable of Matplotlib tick-label ``Text`` objects.
            fontweight (str | None, optional): Font weight to apply. Defaults to None.
            fontstyle (Literal["normal", "italic", "oblique"] | None, optional):
                Font style to apply. Defaults to None.
            fontfamily (str | None, optional): Font family to apply. Defaults to None.

        Returns:
            None
        """
        # These are matplotlib.text.Text objects.
        for text in labels:
            if fontweight is not None:
                text.set_fontweight(fontweight)
            if fontstyle is not None:
                text.set_fontstyle(fontstyle)
            if fontfamily is not None:
                text.set_fontfamily(fontfamily)

    def _apply_tick_style(self, axis: Literal["x", "y"], style: TickStyle) -> None:
        """Apply tick style to the specified axis.

        Args:
            axis (Literal["x", "y"]): The axis to apply the style to.
            style (TickStyle): The tick style to apply.

        Returns:
            None
        """
        owner = self.__class__.__name__
        # Tick marks + tick label basics
        label_color_resolved = resolve_rgba(
            style.fontcolor, style.fontalpha, field="fontcolor", owner=owner
        )
        tick_color_resolved = resolve_rgba(
            style.tickcolor, style.tickalpha, field="tickcolor", owner=owner
        )
        self._ax.tick_params(
            axis=axis,
            which=style.ticktype,
            labelsize=style.size,
            rotation=style.rotation,
            labelcolor=label_color_resolved,
            color=tick_color_resolved,
        )

        # Tick label text styling (weight/style/family)
        tick_label_getter = self._ax.get_xticklabels if axis == "x" else self._ax.get_yticklabels
        if style.ticktype in ("major", "both"):
            self._apply_ticklabel_textprops(
                tick_label_getter(minor=False),
                fontweight=style.fontweight,
                fontstyle=style.fontstyle,
                fontfamily=style.fontfamily,
            )
        if style.ticktype in ("minor", "both"):
            self._apply_ticklabel_textprops(
                tick_label_getter(minor=True),
                fontweight=style.fontweight,
                fontstyle=style.fontstyle,
                fontfamily=style.fontfamily,
            )
