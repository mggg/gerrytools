"""Choose a drive-time center (centroid unit) for CTT.

Centroid selection follows the *intent* of the weight mode, without a ``p_c``
self-penalty (which would favor lower-population centers):

- ``none``: minimize sum_u t(c, u)  — each unit equal
- ``p+p`` / ``p*p``: minimize sum_u t(c, u) * p_u  — near people
"""

from __future__ import annotations

from typing import Mapping, Sequence

from .od_table import TravelTimeTable, UnitId
from .weights import WeightMode


def choose_centroid(
    units: Sequence[UnitId],
    populations: Mapping[UnitId, float],
    od: TravelTimeTable,
    weight: WeightMode,
) -> UnitId | None:
    """Return the centroid unit id, or None if no valid candidate exists.

    A candidate is invalid if any trip from that candidate to a unit in
    ``units`` is NaN (unreachable). For people-aware modes, candidates with
    ``population <= 0`` are skipped.

    Ties break toward the smallest unit id (Python ``<`` on ids).
    """
    if not units:
        return None

    od.require_units(units)
    people_aware = weight in ("p+p", "p*p")

    best_id: UnitId | None = None
    best_cost = float("inf")

    for c in units:
        if people_aware and float(populations[c]) <= 0:
            continue

        cost = 0.0
        unreachable = False
        for u in units:
            t_cu = od.travel_time(c, u)
            if t_cu != t_cu:  # NaN
                unreachable = True
                break
            if people_aware:
                cost += t_cu * float(populations[u])
            else:
                cost += t_cu

        if unreachable:
            continue

        if cost < best_cost or (
            cost == best_cost and (best_id is None or c < best_id)  # type: ignore[operator]
        ):
            best_cost = cost
            best_id = c

    return best_id
