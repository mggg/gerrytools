"""Unit-to-unit drive-time table.

The scoring functions never build OSM graphs. They only ask this table for
``t(a, b)``. Build the table in pipeline code (or a future gerrytools helper)
and pass it in.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isnan
from typing import Hashable, Iterable, Mapping, Sequence

import numpy as np

UnitId = Hashable


def is_missing(t: float) -> bool:
    """True for a NaN travel time (missing/unreachable trip).

    Narrower than ``np.isfinite``: this only screens out NaN, so an ``inf``
    travel time (not expected from a Dijkstra-built table, but not impossible
    from a hand-built one) still passes through here. Code that collects
    individual trips (district.py, centroid.py) uses this; code that
    aggregates scores across districts uses the stricter ``is_finite`` in
    district.py, which also excludes ``inf``.
    """
    return isnan(t)


@dataclass(frozen=True, slots=True)
class TravelTimeTable:
    """Lookup of drive times between unit ids.

    Times are stored in a dense matrix aligned to ``unit_ids``. Missing / unreachable
    pairs should be ``NaN``. Diagonal should be 0. The table is treated as symmetric
    throughout this package (``t(a, b) == t(b, a)``); callers building a table from a
    directed source (e.g. one-way streets) are responsible for symmetrizing it, since
    nothing here checks or corrects for asymmetry.
    """

    unit_ids: tuple[UnitId, ...]
    times: np.ndarray  # shape (n, n), float; NaN = unreachable
    _index: Mapping[UnitId, int] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        arr = np.asarray(self.times, dtype=float)
        if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
            raise ValueError(f"times must be square; got shape {arr.shape}")
        if arr.shape[0] != len(self.unit_ids):
            raise ValueError(
                f"times side length {arr.shape[0]} != number of unit_ids {len(self.unit_ids)}"
            )
        if len(set(self.unit_ids)) != len(self.unit_ids):
            raise ValueError("unit_ids must be unique")
        # _index is derived here, never caller-supplied, so no table can exist
        # whose index disagrees with its unit_ids.
        object.__setattr__(self, "times", arr)
        object.__setattr__(self, "_index", {u: i for i, u in enumerate(self.unit_ids)})

    @classmethod
    def from_matrix(
        cls,
        unit_ids: Sequence[UnitId],
        times: np.ndarray,
    ) -> TravelTimeTable:
        """Build a table from unit id order and an ``(n, n)`` time matrix."""
        return cls(unit_ids=tuple(unit_ids), times=np.asarray(times, dtype=float))

    def travel_time(self, a: UnitId, b: UnitId) -> float:
        """Return drive time from ``a`` to ``b``, or NaN if missing/unreachable."""
        i = self._index.get(a)
        j = self._index.get(b)
        if i is None or j is None:
            return float("nan")
        return float(self.times[i, j])

    def has_unit(self, u: UnitId) -> bool:
        return u in self._index

    def require_units(self, units: Iterable[UnitId]) -> None:
        missing = [u for u in units if u not in self._index]
        if missing:
            raise KeyError(f"Units not in travel-time table: {missing!r}")
