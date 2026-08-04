"""`_ManagedAxesState` — per-unit ownership tracking for axes-level state.

Gerrytools plots avoid ``ax.clear()`` on rebuild and instead rely on two
cooperating mechanisms:

1. The artist registry removes only the matplotlib artists gerrytools created.
2. This module tracks who most recently set each *axes-level* setting
   (xlim, scale, ticks, label, title, frame, legend, axis visibility) and
   uses that to decide, on every rebuild, whether to reapply gerrytools state
   or yield to direct matplotlib changes the user made.

Resolution is most-recent-wins per unit: whichever party (gerrytools or the
user) touched a given unit last owns it on the next rebuild.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, cast
from weakref import WeakKeyDictionary

import matplotlib.colors as mcolors
import numpy as np
from matplotlib.axes import Axes
from matplotlib.axis import Axis
from matplotlib.figure import Figure
from matplotlib.legend import Legend
from matplotlib.text import Text
from matplotlib.ticker import AutoLocator, MaxNLocator, NullLocator
from matplotlib.typing import ColorType

from gerrytools.typing import TickType

# The managed-unit vocabulary. Values are exactly the field names of
# ``_AxesSnapshot``, which is what lets per-unit snapshot access be a getattr.
Unit = Literal[
    "x_limits",
    "y_limits",
    "x_ticks",
    "y_ticks",
    "x_tick_style",
    "y_tick_style",
    "x_label",
    "y_label",
    "title",
    "x_scale",
    "y_scale",
    "frame",
    "legend",
    "axis_visibility",
    "aspect",
]


def _recompute_data_limits(ax: Axes) -> None:
    """Recompute limits, including collections that Matplotlib's ``relim`` skips."""
    ax.relim()
    for collection in ax.collections:
        bounds = collection.get_datalim(ax.transData)
        ax.update_datalim(bounds.get_points())


# Internal sentinel marking "gerrytools claimed the unit but has not yet
# recorded a concrete applied value." Never escapes to user code; never
# substituted for ``None`` because ``None`` is a real public value for some
# units (e.g. ``ax.get_legend()`` returns ``None`` when there is no legend).
_NO_LAST_APPLIED = object()

OwnershipState = Literal["external", "gerrytools_explicit", "gerrytools_default", "unclaimed"]

# Float tolerance for limit, tick-location, and RGBA comparisons, via
# ``math.isclose`` / ``np.allclose``.
_REL_TOL = 1e-9
_ABS_TOL = 1e-12


# ---------------------------------------------------------------------------
# Snapshot dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _LabelSnapshot:
    """Atomic text+style snapshot for an axis label.

    All fields read from matplotlib public getters on the underlying ``Text``
    artist, except ``labelpad`` which lives on ``ax.<x|y>axis.labelpad``.
    """

    text: str
    fontsize: float | str | None
    fontweight: str | int | None
    fontstyle: str | None
    fontfamily: tuple[str, ...] | None
    color: tuple[float, float, float, float]
    labelpad: float | None

    def approx_equals(self, other: object) -> bool:
        if not isinstance(other, _LabelSnapshot):
            return False
        return (
            self.text == other.text
            and self.fontsize == other.fontsize
            and self.fontweight == other.fontweight
            and self.fontstyle == other.fontstyle
            and self.fontfamily == other.fontfamily
            and self.labelpad == other.labelpad
            and _rgba_equal(self.color, other.color)
        )


@dataclass(frozen=True)
class _TitleArtistSnapshot:
    """Text and style for one of Matplotlib's three title artists.

    Title pad has no stable public getter in matplotlib 3.10.6, so it is not
    part of the snapshot; external direct changes to title pad are not
    detected — a known limitation of matplotlib's title API.
    """

    text: str
    fontsize: float | str | None
    fontweight: str | int | None
    fontstyle: str | None
    fontfamily: tuple[str, ...] | None
    color: tuple[float, float, float, float]

    def approx_equals(self, other: object) -> bool:
        if not isinstance(other, _TitleArtistSnapshot):
            return False
        return (
            self.text == other.text
            and self.fontsize == other.fontsize
            and self.fontweight == other.fontweight
            and self.fontstyle == other.fontstyle
            and self.fontfamily == other.fontfamily
            and _rgba_equal(self.color, other.color)
        )


@dataclass(frozen=True)
class _TitleSnapshot:
    """Atomic snapshot of Matplotlib's left, center, and right title artists."""

    left: _TitleArtistSnapshot
    center: _TitleArtistSnapshot
    right: _TitleArtistSnapshot

    def approx_equals(self, other: object) -> bool:
        return isinstance(other, _TitleSnapshot) and all(
            current.approx_equals(previous)
            for current, previous in zip(
                (self.left, self.center, self.right),
                (other.left, other.center, other.right),
                strict=True,
            )
        )


@dataclass(frozen=True)
class _TickSnapshot:
    """Atomic snapshot of tick locations, labels and label visibility."""

    locations: tuple[float, ...]
    labels: tuple[str, ...]
    labels_visible: bool

    def approx_equals(self, other: object) -> bool:
        if not isinstance(other, _TickSnapshot):
            return False
        return (
            _tick_locations_equal(self.locations, other.locations)
            and list(self.labels) == list(other.labels)
            and self.labels_visible == other.labels_visible
        )


@dataclass(frozen=True)
class _TickSetStyleSnapshot:
    """Styling of one tick set (major or minor): label and tick-mark style fields.

    Tick-mark colors come from ``Tick.tick1line.get_color()`` which encodes
    alpha in the RGBA tuple.

    Only the first tick's label and mark are sampled, so a direct external change to a
    single non-first tick is not detected — a known limitation, like title pad on
    ``_TitleSnapshot``.
    """

    label_size: float | str | None
    label_rotation: float
    label_color: tuple[float, float, float, float]
    label_weight: str | int | None
    label_style: str | None
    label_family: tuple[str, ...] | None
    tick_color: tuple[float, float, float, float]

    def approx_equals(self, other: object) -> bool:
        if not isinstance(other, _TickSetStyleSnapshot):
            return False
        return (
            self.label_size == other.label_size
            and math.isclose(self.label_rotation, other.label_rotation, abs_tol=1e-9)
            and self.label_weight == other.label_weight
            and self.label_style == other.label_style
            and self.label_family == other.label_family
            and _rgba_equal(self.label_color, other.label_color)
            and _rgba_equal(self.tick_color, other.tick_color)
        )


@dataclass(frozen=True)
class _TickStyleSnapshot:
    """Snapshot of tick styling for the tick set(s) a caller selected.

    ``None`` for a set means the snapshot carries no opinion about it; comparisons skip
    any set that either side omits. The full-axes snapshot captures both sets, while the
    tick-style reconcile captures only the set(s) named by the stored
    ``TickStyle.ticktype``, so external changes to the styled set are detected without
    pinning the other.
    """

    major: _TickSetStyleSnapshot | None
    minor: _TickSetStyleSnapshot | None

    def approx_equals(self, other: object) -> bool:
        if not isinstance(other, _TickStyleSnapshot):
            return False
        for mine, theirs in ((self.major, other.major), (self.minor, other.minor)):
            if mine is None or theirs is None:
                continue
            if not mine.approx_equals(theirs):
                return False
        return True


@dataclass(frozen=True)
class _AxesSnapshot:
    """All observable axes-level state read in one pass."""

    x_limits: tuple[float, float]
    y_limits: tuple[float, float]
    x_ticks: _TickSnapshot
    y_ticks: _TickSnapshot
    x_tick_style: _TickStyleSnapshot
    y_tick_style: _TickStyleSnapshot
    x_label: _LabelSnapshot
    y_label: _LabelSnapshot
    title: _TitleSnapshot
    x_scale: str
    y_scale: str
    frame: tuple[bool, bool, bool, bool]  # top, right, bottom, left
    legend: Legend | None
    axis_visibility: bool
    aspect: str | float  # matplotlib's "auto" default or a numeric ratio


# Every managed unit, derived from the snapshot fields the vocabulary mirrors.
_ALL_UNITS = cast(
    "tuple[Unit, ...]", tuple(field.name for field in dataclasses.fields(_AxesSnapshot))
)


# ---------------------------------------------------------------------------
# Snapshot helpers
# ---------------------------------------------------------------------------


def _rgba(color: object) -> tuple[float, float, float, float]:
    # matplotlib artist getters (e.g. ``get_color``) return loosely-typed color
    # values; ``to_rgba`` validates them at runtime.
    r, g, b, a = mcolors.to_rgba(cast(ColorType, color))
    return (float(r), float(g), float(b), float(a))


def _fontfamily_to_tuple(value: object) -> tuple[str, ...] | None:
    """Normalize a matplotlib fontfamily return value to a hashable tuple."""
    if value is None:
        return None
    if isinstance(value, str):
        return (value,)
    if isinstance(value, (list, tuple)):
        return tuple(str(v) for v in value)
    return (str(value),)


def _label_snapshot(ax: Axes, which: Literal["x", "y"]) -> _LabelSnapshot:
    text_artist = ax.xaxis.label if which == "x" else ax.yaxis.label
    return _LabelSnapshot(
        text=text_artist.get_text(),
        fontsize=text_artist.get_fontsize(),
        fontweight=text_artist.get_fontweight(),
        fontstyle=text_artist.get_fontstyle(),
        fontfamily=_fontfamily_to_tuple(text_artist.get_fontfamily()),
        color=_rgba(text_artist.get_color()),
        labelpad=float(ax.xaxis.labelpad if which == "x" else ax.yaxis.labelpad),
    )


def _title_snapshot(ax: Axes) -> _TitleSnapshot:
    def snapshot(text_artist: Text) -> _TitleArtistSnapshot:
        return _TitleArtistSnapshot(
            text=text_artist.get_text(),
            fontsize=text_artist.get_fontsize(),
            fontweight=text_artist.get_fontweight(),
            fontstyle=text_artist.get_fontstyle(),
            fontfamily=_fontfamily_to_tuple(text_artist.get_fontfamily()),
            color=_rgba(text_artist.get_color()),
        )

    return _TitleSnapshot(
        left=snapshot(cast(Text, getattr(ax, "_left_title"))),
        center=snapshot(ax.title),
        right=snapshot(cast(Text, getattr(ax, "_right_title"))),
    )


def _tick_snapshot(ax: Axes, which: Literal["x", "y"]) -> _TickSnapshot:
    if which == "x":
        locations = ax.get_xticks()
        labels = [t.get_text() for t in ax.get_xticklabels(minor=False)]
        visible_flags = [t.get_visible() for t in ax.get_xticklabels(minor=False)]
    else:
        locations = ax.get_yticks()
        labels = [t.get_text() for t in ax.get_yticklabels(minor=False)]
        visible_flags = [t.get_visible() for t in ax.get_yticklabels(minor=False)]
    # "Tick labels visible" is a single boolean for the unit: any-visible.
    labels_visible = bool(any(visible_flags)) if visible_flags else True
    return _TickSnapshot(
        locations=tuple(float(loc) for loc in locations),
        labels=tuple(labels),
        labels_visible=labels_visible,
    )


def _tick_set_style_snapshot(
    ax: Axes, which: Literal["x", "y"], *, minor: bool
) -> _TickSetStyleSnapshot:
    axis = ax.xaxis if which == "x" else ax.yaxis
    ticks = axis.get_minor_ticks() if minor else axis.get_major_ticks()
    label_texts = (
        ax.get_xticklabels(minor=minor) if which == "x" else ax.get_yticklabels(minor=minor)
    )
    if label_texts:
        first = label_texts[0]
        label_size = first.get_fontsize()
        label_rotation = float(first.get_rotation())
        label_color = _rgba(first.get_color())
        label_weight = first.get_fontweight()
        label_style = first.get_fontstyle()
        label_family = _fontfamily_to_tuple(first.get_fontfamily())
    else:
        label_size = None
        label_rotation = 0.0
        label_color = (0.0, 0.0, 0.0, 1.0)
        label_weight = None
        label_style = None
        label_family = None
    if ticks:
        tick_color = _rgba(ticks[0].tick1line.get_color())
    else:
        tick_color = (0.0, 0.0, 0.0, 1.0)
    return _TickSetStyleSnapshot(
        label_size=label_size,
        label_rotation=label_rotation,
        label_color=label_color,
        label_weight=label_weight,
        label_style=label_style,
        label_family=label_family,
        tick_color=tick_color,
    )


def _tick_style_snapshot(
    ax: Axes, which: Literal["x", "y"], ticktype: TickType = "major"
) -> _TickStyleSnapshot:
    """Snapshot the styling of the selected tick set(s) for one axis."""
    return _TickStyleSnapshot(
        major=(
            _tick_set_style_snapshot(ax, which, minor=False)
            if ticktype in ("major", "both")
            else None
        ),
        minor=(
            _tick_set_style_snapshot(ax, which, minor=True)
            if ticktype in ("minor", "both")
            else None
        ),
    )


def _frame_snapshot(ax: Axes) -> tuple[bool, bool, bool, bool]:
    return (
        ax.spines["top"].get_visible(),
        ax.spines["right"].get_visible(),
        ax.spines["bottom"].get_visible(),
        ax.spines["left"].get_visible(),
    )


def _is_default_locator(axis_obj: Axis) -> bool:
    """Heuristic: AutoLocator / MaxNLocator / NullLocator mean default-ish.

    FixedLocator and other concrete locators are taken as evidence that the
    user (or another library) set tick positions explicitly.
    """
    return isinstance(axis_obj.get_major_locator(), (AutoLocator, MaxNLocator, NullLocator))


def _fresh_axes_snapshot() -> _AxesSnapshot:
    """Snapshot the matplotlib default state of an untouched axes.

    Uses ``matplotlib.figure.Figure()`` (not ``plt.subplots()``) so the
    temporary figure never registers with pyplot's global figure manager:
    no Jupyter inline display side effects, and nothing to close afterwards.
    """
    fig = Figure()
    ax = fig.add_subplot(111)
    return _read_snapshot(ax)


def _read_snapshot(ax: Axes) -> _AxesSnapshot:
    x_left, x_right = ax.get_xlim()
    y_bottom, y_top = ax.get_ylim()
    return _AxesSnapshot(
        x_limits=(float(x_left), float(x_right)),
        y_limits=(float(y_bottom), float(y_top)),
        x_ticks=_tick_snapshot(ax, "x"),
        y_ticks=_tick_snapshot(ax, "y"),
        # Both tick sets: the baseline must be comparable against last-applied values
        # recorded for any stored ``TickStyle.ticktype`` (major, minor, or both).
        x_tick_style=_tick_style_snapshot(ax, "x", "both"),
        y_tick_style=_tick_style_snapshot(ax, "y", "both"),
        x_label=_label_snapshot(ax, "x"),
        y_label=_label_snapshot(ax, "y"),
        title=_title_snapshot(ax),
        x_scale=ax.get_xscale(),
        y_scale=ax.get_yscale(),
        frame=_frame_snapshot(ax),
        legend=ax.get_legend(),
        axis_visibility=bool(ax.axison),
        aspect=ax.get_aspect(),
    )


# ---------------------------------------------------------------------------
# Per-unit comparison helpers
# ---------------------------------------------------------------------------


def _rgba_equal(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> bool:
    return all(math.isclose(x, y, rel_tol=_REL_TOL, abs_tol=_ABS_TOL) for x, y in zip(a, b))


def _limits_equal(a: tuple[float, ...], b: tuple[float, ...]) -> bool:
    return len(a) == len(b) and all(
        math.isclose(x, y, rel_tol=_REL_TOL, abs_tol=_ABS_TOL) for x, y in zip(a, b)
    )


def _tick_locations_equal(a: tuple[float, ...], b: tuple[float, ...]) -> bool:
    if len(a) != len(b):
        return False
    if not a:
        return True
    return bool(np.allclose(np.asarray(a), np.asarray(b), rtol=_REL_TOL, atol=_ABS_TOL))


def _aspect_equal(a: str | float, b: str | float) -> bool:
    """Compare matplotlib aspect values.

    ``ax.get_aspect()`` returns either the string ``"auto"`` or a float
    (``"equal"`` is normalized to ``1.0`` by matplotlib internally). String
    and numeric values are never equal; numeric values use ``math.isclose``.
    """
    if isinstance(a, str) or isinstance(b, str):
        return a == b
    return math.isclose(float(a), float(b), rel_tol=_REL_TOL, abs_tol=_ABS_TOL)


_SNAPSHOT_VALUE_TYPES = (_LabelSnapshot, _TitleSnapshot, _TickSnapshot, _TickStyleSnapshot)


def _unit_equal(unit: Unit, a: object, b: object) -> bool:
    if unit == "legend":
        # Identity comparison: an external legend placed after last apply is a
        # different object than the one gerrytools tracked.
        return a is b
    if isinstance(a, _SNAPSHOT_VALUE_TYPES):
        return a.approx_equals(b)
    if unit in ("x_limits", "y_limits") and isinstance(a, tuple) and isinstance(b, tuple):
        # Limits values are always (low, high) float pairs by construction.
        return _limits_equal(cast("tuple[float, ...]", a), cast("tuple[float, ...]", b))
    if unit == "aspect" and isinstance(a, (str, int, float)) and isinstance(b, (str, int, float)):
        return _aspect_equal(a, b)
    # Frame, scales, axis_visibility: exact equality.
    return a == b


# ---------------------------------------------------------------------------
# Per-unit ownership record
# ---------------------------------------------------------------------------


@dataclass
class _UnitState:
    ownership: OwnershipState = "unclaimed"
    # last_applied is either a concrete snapshot field value, ``None`` for
    # units whose canonical "no value" is None (e.g. legend), or the
    # ``_NO_LAST_APPLIED`` sentinel for store-and-claim ownership without a
    # recorded value yet.
    last_applied: object = _NO_LAST_APPLIED


# ---------------------------------------------------------------------------
# Public state machine
# ---------------------------------------------------------------------------


class _ManagedAxesState:
    """Per-axes managed-unit ownership and last-applied history.

    One instance lives on each gerrytools plot. Plot configuration (e.g.
    the axis-state objects) stays on the plot; this class only knows about
    "who applied what to which axes most recently" at the matplotlib level.

    See the module docstring for the full contract.
    """

    def __init__(self) -> None:
        self._units: dict[Unit, _UnitState] = {unit: _UnitState() for unit in _ALL_UNITS}
        self._axes_histories: WeakKeyDictionary[Axes, dict[Unit, _UnitState]] = WeakKeyDictionary()

    # -- initialization & rebind ------------------------------------------------

    def initialize_from_ax(self, ax: Axes) -> None:
        """Classify each unit as externally set or default at bind time.

        Only touches units currently in the ``unclaimed`` state — units already
        reclaimed by explicit plot configuration (e.g. across a
        ``bind_to_ax`` call) are left alone so the explicit gerrytools state
        wins over pre-existing axes state on the new axes.

        Limits and ticks are classified by intent signals rather than value
        comparison: autoscale-off means someone set limits, and a non-default
        locator means someone set tick positions. Every other unit compares
        the current snapshot against a fresh-axes default snapshot.
        """
        default = _fresh_axes_snapshot()
        current = _read_snapshot(ax)

        if self._units["x_limits"].ownership == "unclaimed" and not ax.get_autoscalex_on():
            self._mark_external("x_limits", current.x_limits)
        if self._units["y_limits"].ownership == "unclaimed" and not ax.get_autoscaley_on():
            self._mark_external("y_limits", current.y_limits)

        if self._units["x_ticks"].ownership == "unclaimed" and not _is_default_locator(ax.xaxis):
            self._mark_external("x_ticks", current.x_ticks)
        if self._units["y_ticks"].ownership == "unclaimed" and not _is_default_locator(ax.yaxis):
            self._mark_external("y_ticks", current.y_ticks)

        special_cases: set[Unit] = {"x_limits", "y_limits", "x_ticks", "y_ticks"}
        for unit in _ALL_UNITS:
            if unit in special_cases or self._units[unit].ownership != "unclaimed":
                continue
            current_value = getattr(current, unit)
            if not _unit_equal(unit, current_value, getattr(default, unit)):
                self._mark_external(unit, current_value)

    def reset_history(self) -> None:
        """Clear per-axes last-applied history and external classifications.

        Reclaim flags survive because they describe plot configuration, not
        per-axes history. After ``reset_history``, ``initialize_from_ax``
        classifies only currently-unclaimed units against the new axes.
        """
        for state in self._units.values():
            if state.ownership in ("external", "gerrytools_default"):
                state.ownership = "unclaimed"
            state.last_applied = _NO_LAST_APPLIED

    def remember_axes(self, ax: Axes) -> None:
        """Save this plot's ownership history for an axes it is leaving."""
        self._axes_histories[ax] = {
            unit: dataclasses.replace(state) for unit, state in self._units.items()
        }

    def restore_axes(self, ax: Axes) -> None:
        """Restore prior history for ``ax``, while carrying current explicit claims."""
        explicit: set[Unit] = {
            unit for unit, state in self._units.items() if state.ownership == "gerrytools_explicit"
        }
        cached = self._axes_histories.get(ax)
        if cached is None:
            self.reset_history()
        else:
            self._units = {unit: dataclasses.replace(state) for unit, state in cached.items()}
        for unit in explicit:
            self._units[unit] = _UnitState("gerrytools_explicit", _NO_LAST_APPLIED)

    # -- snapshot & external detection ------------------------------------------

    def begin_rebuild(self, ax: Axes) -> tuple[_AxesSnapshot, set[Unit]]:
        """Open a rebuild pass: snapshot the axes and classify external ownership.

        Returns the pre-rebuild snapshot (needed later by
        :meth:`restore_autoscale_protected`) and the set of externally owned units.
        """
        before = _read_snapshot(ax)
        external = self.detect_external_changes(before)
        return before, external

    def detect_external_changes(self, snapshot: _AxesSnapshot) -> set[Unit]:
        """Return units that should be treated as externally owned this rebuild.

        Per-unit dispatch by current ownership state:

        - ``external``: kept in the returned set without value comparison.
          Once a unit yields, it stays external until a gerrytools API
          explicitly reclaims it.
        - ``gerrytools_explicit`` or ``gerrytools_default`` with a recorded
          last-applied value: compare snapshot to last-applied using the
          per-unit equality rules; if they differ, mark external and
          transition ownership so subsequent rebuilds keep it external.
        - ``gerrytools_explicit`` with no recorded value (store-and-claim
          ran but the apply hasn't happened yet): never added.
        - ``unclaimed`` (no last-applied history yet): never added; the
          next apply records the resulting ownership.
        """
        external: set[Unit] = set()
        for unit, state in self._units.items():
            if state.ownership == "external":
                external.add(unit)
                continue
            if state.ownership == "unclaimed":
                continue
            if state.last_applied is _NO_LAST_APPLIED:
                # gerrytools_explicit with no recorded value yet — store-and-claim.
                continue
            current = getattr(snapshot, unit)
            if not _unit_equal(unit, current, state.last_applied):
                external.add(unit)
                state.ownership = "external"
                state.last_applied = current
        return external

    # -- write paths ------------------------------------------------------------

    def reclaim_and_mark(self, unit: Unit, value: object) -> None:
        """Apply-now public setter path. Claim ownership and record value."""
        state = self._units[unit]
        state.ownership = "gerrytools_explicit"
        state.last_applied = value

    def reclaim_without_value(self, unit: Unit) -> None:
        """Store-and-claim public setter path. Claim without recording yet."""
        state = self._units[unit]
        state.ownership = "gerrytools_explicit"
        state.last_applied = _NO_LAST_APPLIED

    def record_default(self, unit: Unit, value: object) -> None:
        """Internal default applier path. Record value without claiming."""
        state = self._units[unit]
        # An already-explicit unit stays explicit; defaults never demote.
        if state.ownership != "gerrytools_explicit":
            state.ownership = "gerrytools_default"
        state.last_applied = value

    def release(self, unit: Unit, value: object) -> None:
        """Public clear path. Drop any explicit or external claim back to a default.

        Records ``value`` as last-applied so the next rebuild still detects external
        changes, while gerrytools stops asserting explicit ownership of the unit.
        """
        state = self._units[unit]
        state.ownership = "gerrytools_default"
        state.last_applied = value

    def reconcile(
        self,
        unit: Unit,
        external: set[Unit],
        apply_fn: Callable[[], None],
        read_current: Callable[[], object],
    ) -> None:
        """Reconcile one managed unit during a rebuild pass.

        The shared skeleton for the mechanical apply helpers: skip externally owned units
        entirely; otherwise run ``apply_fn`` (the unit's write-to-axes step, which may be a
        no-op) and record ``read_current()``. A unit a public setter has reclaimed keeps
        explicit ownership via :meth:`reclaim_and_mark`; anything else is recorded as a
        gerrytools default via :meth:`record_default`.
        """
        if unit in external:
            return
        apply_fn()
        value = read_current()
        if self.is_reclaimed(unit):
            self.reclaim_and_mark(unit, value)
        else:
            self.record_default(unit, value)

    # -- queries -----------------------------------------------------------------

    def is_reclaimed(self, unit: Unit) -> bool:
        """True iff the unit is currently ``gerrytools_explicit``.

        Default-owned and externally-owned units return False. Used by apply
        helpers that need to distinguish "gerrytools owns this and may remove
        external content" from "gerrytools has no claim."
        """
        return self._units[unit].ownership == "gerrytools_explicit"

    # -- restore -----------------------------------------------------------------

    def restore_autoscale_protected(
        self,
        ax: Axes,
        pre_redraw: _AxesSnapshot,
        external_units: set[Unit],
    ) -> None:
        """Restore externally-set xlim/ylim that artist drawing may have clobbered.

        The limit units are the only autoscale-protected ones; apply-or-skip
        units never need this because matplotlib's autoscale only affects
        limits during artist drawing.
        """
        if "x_limits" in external_units:
            ax.set_xlim(*pre_redraw.x_limits)
        if "y_limits" in external_units:
            ax.set_ylim(*pre_redraw.y_limits)

    # -- internal ---------------------------------------------------------------

    def _mark_external(self, unit: Unit, value: object) -> None:
        state = self._units[unit]
        state.ownership = "external"
        state.last_applied = value
