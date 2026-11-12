"""Validated unit-to-unit travel-time tables."""

from __future__ import annotations

from collections.abc import Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

UnitId = Hashable


@dataclass(frozen=True, slots=True, eq=False)
class TravelTimeTable:
    """Symmetric travel times aligned to geographic unit identifiers.

    Args:
        unit_ids (tuple[UnitId, ...]): Unit identifiers in matrix order.
        times (NDArray[np.float64]): Square symmetric matrix of nonnegative travel times.

    Raises:
        ValueError: If identifiers or travel times violate the table invariants.
    """

    unit_ids: tuple[UnitId, ...]
    times: NDArray[np.float64]
    _index: Mapping[UnitId, int] = field(init=False, repr=False, compare=False)
    _fully_reachable: bool = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
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
        object.__setattr__(self, "times", arr)
        object.__setattr__(self, "_index", {unit: i for i, unit in enumerate(self.unit_ids)})
        object.__setattr__(self, "_fully_reachable", bool(np.isfinite(arr).all()))

    def __eq__(self, other: object) -> bool:
        """Return whether two tables have identical identifiers and values."""
        if not isinstance(other, TravelTimeTable):
            return NotImplemented
        return self.unit_ids == other.unit_ids and bool(
            np.array_equal(self.times, other.times, equal_nan=True)
        )

    __hash__ = None  # type: ignore[assignment]

    @classmethod
    def from_matrix(
        cls,
        unit_ids: Sequence[UnitId],
        times: NDArray[np.floating],
    ) -> TravelTimeTable:
        """Build a table from an already symmetric matrix.

        Args:
            unit_ids (Sequence[UnitId]): Unit identifiers in matrix order.
            times (NDArray[np.floating]): Square symmetric matrix.

        Returns:
            TravelTimeTable: Validated immutable table.
        """
        return cls(unit_ids=tuple(unit_ids), times=np.asarray(times, dtype=float))

    @classmethod
    def from_directed_matrix(
        cls,
        unit_ids: Sequence[UnitId],
        directed_times: NDArray[np.floating],
    ) -> TravelTimeTable:
        """Average opposite directed routes into a symmetric table.

        Args:
            unit_ids (Sequence[UnitId]): Unit identifiers in matrix order.
            directed_times (NDArray[np.floating]): Square directed travel-time matrix.

        Returns:
            TravelTimeTable: Symmetric table whose off-diagonal values are the arithmetic mean
                of the two directions.
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
        return cls(unit_ids=ids, times=(directed + directed.T) / 2.0)

    def travel_time(self, source: UnitId, destination: UnitId) -> float:
        """Look up one symmetric travel time.

        Args:
            source (UnitId): Starting geographic unit.
            destination (UnitId): Ending geographic unit.

        Returns:
            float: Travel time, or ``NaN`` when either unit is absent.
        """
        source_index = self._index.get(source)
        destination_index = self._index.get(destination)
        if source_index is None or destination_index is None:
            return float("nan")
        return float(self.times[source_index, destination_index])

    def has_unit(self, unit: UnitId) -> bool:
        """Return whether ``unit`` occurs in the table."""
        return unit in self._index

    def require_units(self, units: Iterable[UnitId]) -> None:
        """Require every supplied unit to occur in the table.

        Args:
            units (Iterable[UnitId]): Unit identifiers to validate.

        Raises:
            KeyError: If any identifier is absent.
        """
        missing = [unit for unit in units if unit not in self._index]
        if missing:
            raise KeyError(f"Units not in travel-time table: {missing!r}")

    def matrix_for(self, units: Sequence[UnitId]) -> NDArray[np.float64]:
        """Return the travel-time submatrix for ``units`` in the supplied order.

        Args:
            units (Sequence[UnitId]): Unit identifiers to select.

        Returns:
            NDArray[np.float64]: A copied square matrix aligned to ``units``.

        Raises:
            KeyError: If any identifier is absent.
        """
        self.require_units(units)
        positions = np.fromiter((self._index[unit] for unit in units), dtype=np.intp)
        return np.array(self.times[np.ix_(positions, positions)], copy=True)

    @property
    def is_fully_reachable(self) -> bool:
        """Whether every ordered unit pair has a finite travel time."""
        return self._fully_reachable

    def connectivity_components(self) -> tuple[tuple[UnitId, ...], ...]:
        """Return mutually reachable components for connectivity diagnostics."""
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
        """Require a finite travel time between every ordered unit pair.

        Raises:
            ValueError: If any ordered pair is unreachable.
        """
        if self._fully_reachable:
            return
        components = self.connectivity_components()
        sizes = [len(component) for component in components]
        missing_mask = ~np.isfinite(self.times)
        missing_count = int(np.count_nonzero(missing_mask))
        locations = np.argwhere(missing_mask)[:5]
        examples = [(self.unit_ids[int(i)], self.unit_ids[int(j)]) for i, j in locations]
        raise ValueError(
            "Travel-time scoring requires statewide connectivity: every included unit "
            "must reach every other included unit. "
            f"Found {missing_count} unreachable ordered pair(s) across "
            f"{len(components)} mutually reachable component(s) with sizes {sizes}; "
            f"example unreachable pairs: {examples}. Resolve the road network or "
            "apply one documented, plan-independent unit exclusion before scoring."
        )
