"""Unit-to-unit drive-time table.

The scoring functions never build OSM graphs. They only ask this table for
``t(a, b)``. Build the table in pipeline code (or a future gerrytools helper)
and pass it in.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Hashable, Iterable, Mapping, Sequence

import numpy as np

UnitId = Hashable


def is_missing(t: float) -> bool:
    """Return whether a travel time is missing or unreachable."""
    return not bool(np.isfinite(t))


@dataclass(frozen=True, slots=True)
class TravelTimeTable:
    """Lookup of symmetric drive times between unit ids.

    Times are stored in a dense matrix aligned to ``unit_ids``. Missing / unreachable
    pairs should be ``NaN`` and the diagonal must be 0. Scoring tables must be
    symmetric: use :meth:`from_directed_matrix` to average the two directions from a
    directed routing result.
    """

    unit_ids: tuple[UnitId, ...]
    times: np.ndarray  # shape (n, n), float; non-finite = unreachable
    _index: Mapping[UnitId, int] = field(init=False, repr=False, compare=False)
    _fully_reachable: bool = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        # Copy the caller's array so the cached connectivity result cannot be
        # invalidated later by mutating the original matrix.
        arr = np.array(self.times, dtype=float, copy=True)
        if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
            raise ValueError(f"times must be square; got shape {arr.shape}")
        if arr.shape[0] != len(self.unit_ids):
            raise ValueError(
                f"times side length {arr.shape[0]} != number of unit_ids {len(self.unit_ids)}"
            )
        if len(set(self.unit_ids)) != len(self.unit_ids):
            raise ValueError("unit_ids must be unique")
        diagonal = np.diag(arr)
        if not np.isfinite(diagonal).all() or not np.allclose(diagonal, 0.0):
            raise ValueError("times diagonal must contain finite zeros")
        if np.any(np.isfinite(arr) & (arr < 0)):
            raise ValueError("travel times must be nonnegative")
        if not np.allclose(arr, arr.T, rtol=1e-10, atol=1e-8, equal_nan=True):
            raise ValueError(
                "scoring travel times must be symmetric; use "
                "TravelTimeTable.from_directed_matrix() to average directions"
            )
        arr.setflags(write=False)
        # _index is derived here, never caller-supplied, so no table can exist
        # whose index disagrees with its unit_ids.
        object.__setattr__(self, "times", arr)
        object.__setattr__(self, "_index", {u: i for i, u in enumerate(self.unit_ids)})
        object.__setattr__(self, "_fully_reachable", bool(np.isfinite(arr).all()))

    @classmethod
    def from_matrix(
        cls,
        unit_ids: Sequence[UnitId],
        times: np.ndarray,
    ) -> TravelTimeTable:
        """Build a table from an already-symmetric ``(n, n)`` time matrix."""
        return cls(unit_ids=tuple(unit_ids), times=np.asarray(times, dtype=float))

    @classmethod
    def from_directed_matrix(
        cls,
        unit_ids: Sequence[UnitId],
        directed_times: np.ndarray,
    ) -> TravelTimeTable:
        """Average opposite directed trips into a symmetric scoring table.

        For distinct units ``i`` and ``j``, the scoring time is
        ``(directed[i, j] + directed[j, i]) / 2``. A missing direction therefore
        remains missing after averaging and will be rejected by
        :meth:`validate_connected`.
        """
        ids = tuple(unit_ids)
        directed = np.asarray(directed_times, dtype=float)
        if directed.ndim != 2 or directed.shape[0] != directed.shape[1]:
            raise ValueError(f"directed_times must be square; got shape {directed.shape}")
        if directed.shape[0] != len(ids):
            raise ValueError(
                f"directed_times side length {directed.shape[0]} != number of unit_ids {len(ids)}"
            )
        diagonal = np.diag(directed)
        if not np.isfinite(diagonal).all() or not np.allclose(diagonal, 0.0):
            raise ValueError("directed_times diagonal must contain finite zeros")
        if np.any(np.isfinite(directed) & (directed < 0)):
            raise ValueError("directed travel times must be nonnegative")
        symmetric = (directed + directed.T) / 2.0
        return cls(unit_ids=ids, times=symmetric)

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

    @property
    def is_fully_reachable(self) -> bool:
        """Whether every ordered unit pair has a finite travel time."""
        return self._fully_reachable

    def connectivity_components(self) -> tuple[tuple[UnitId, ...], ...]:
        """Return mutually reachable components for connectivity diagnostics.

        Two units are adjacent when their symmetric scoring time is finite. A
        valid table has exactly one component containing every unit.
        """
        mutual = np.isfinite(self.times) & np.isfinite(self.times.T)
        unseen = set(range(len(self.unit_ids)))
        components: list[tuple[UnitId, ...]] = []
        while unseen:
            first = min(unseen)
            unseen.remove(first)
            stack = [first]
            component: list[int] = []
            while stack:
                current = stack.pop()
                component.append(current)
                neighbors = sorted(
                    {int(i) for i in np.flatnonzero(mutual[current])}.intersection(unseen),
                    reverse=True,
                )
                unseen.difference_update(neighbors)
                stack.extend(neighbors)
            components.append(tuple(self.unit_ids[i] for i in sorted(component)))
        return tuple(sorted(components, key=len, reverse=True))

    def validate_connected(self) -> None:
        """Require finite travel times between every ordered pair of units.

        Travel-time scores use a single, statewide analysis universe. Missing
        pairs are therefore an input error rather than trips that may be
        silently removed from a district denominator.

        Raises:
            ValueError: If any ordered pair is NaN or infinite.
        """
        if self._fully_reachable:
            return

        components = self.connectivity_components()
        sizes = [len(component) for component in components]
        missing_mask = ~np.isfinite(self.times)
        missing_count = int(np.count_nonzero(missing_mask))
        example_indices: list[tuple[int, int]] = []
        for i, row in enumerate(missing_mask):
            remaining = 5 - len(example_indices)
            example_indices.extend((i, int(j)) for j in np.flatnonzero(row)[:remaining])
            if len(example_indices) == 5:
                break
        examples = [(self.unit_ids[i], self.unit_ids[j]) for i, j in example_indices]
        raise ValueError(
            "Travel-time scoring requires statewide connectivity: every included unit "
            "must reach every other included unit. "
            f"Found {missing_count} unreachable ordered pair(s) across "
            f"{len(components)} mutually reachable component(s) with sizes {sizes}; "
            f"example unreachable pairs: {examples}. Resolve the road network or "
            "apply one documented, plan-independent unit exclusion before scoring."
        )
