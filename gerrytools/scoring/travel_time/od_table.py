"""Unit-to-unit drive-time table.

The scoring functions never build OSM graphs. They only ask this table for
``t(a, b)``. Build the table in pipeline code (or a future gerrytools helper)
and pass it in.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Iterable, Mapping, Sequence

import numpy as np

UnitId = Hashable


@dataclass(frozen=True, slots=True)
class TravelTimeTable:
    """Symmetric-style lookup of drive times between unit ids.

    Times are stored in a dense matrix aligned to ``unit_ids``. Missing / unreachable
    pairs should be ``NaN``. Diagonal should be 0.
    """

    unit_ids: tuple[UnitId, ...]
    times: np.ndarray  # shape (n, n), float; NaN = unreachable
    _index: Mapping[UnitId, int]

    @classmethod
    def from_matrix(
        cls,
        unit_ids: Sequence[UnitId],
        times: np.ndarray,
    ) -> TravelTimeTable:
        """Build a table from unit id order and an ``(n, n)`` time matrix."""
        ids = tuple(unit_ids)
        arr = np.asarray(times, dtype=float)
        if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
            raise ValueError(f"times must be square; got shape {arr.shape}")
        if arr.shape[0] != len(ids):
            raise ValueError(f"times side length {arr.shape[0]} != number of unit_ids {len(ids)}")
        if len(set(ids)) != len(ids):
            raise ValueError("unit_ids must be unique")
        index = {u: i for i, u in enumerate(ids)}
        return cls(unit_ids=ids, times=arr, _index=index)

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
