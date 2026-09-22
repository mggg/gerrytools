"""Choose a drive-time center (centroid unit) for CTT.

Centroid selection follows the *intent* of the weight mode, without a ``p_c``
self-penalty (which would favor lower-population centers):

- ``none``: minimize sum_u t(c, u)  — each unit equal
- ``p+p`` / ``p*p``: minimize sum_u t(c, u) * p_u  — near people
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from .od_table import TravelTimeTable, UnitId, is_missing
from .weights import PEOPLE_AWARE_MODES, WeightMode


def choose_centroid(
    units: Sequence[Any],
    populations: Mapping[Any, float],
    od: TravelTimeTable,
    weight: WeightMode,
) -> UnitId | None:
    """Return the centroid unit id, or None if no valid candidate exists.

    A candidate is invalid if any trip from that candidate to a unit in
    ``units`` is NaN (unreachable). For people-aware modes, candidates with
    ``population <= 0`` are skipped.

    Ties break toward whichever tied candidate appears first in ``units``.
    (``UnitId`` is only required to be ``Hashable``, not orderable, so we
    can't break ties with ``<`` on the ids themselves; input order is the
    only ordering guaranteed to exist.)
    """
    if not units:
        return None

    # Redundant with score_district's own od.require_units(units) call on the
    # only current caller, but choose_centroid is exported as public API
    # (see __init__.py) and may be called directly, so it validates its own
    # inputs rather than relying on a caller to have done so.
    od.require_units(units)
    people_aware = weight in PEOPLE_AWARE_MODES

    best_id: UnitId | None = None
    best_cost = float("inf")

    # Scanning in input order and only ever accepting a *strict* improvement
    # means the first candidate to reach a given cost is the one that sticks,
    # which is exactly the "tie breaks toward input order" rule above.
    for c in units:
        if people_aware and float(populations[c]) <= 0:
            continue

        cost = 0.0
        unreachable = False
        for u in units:
            t_cu = od.travel_time(c, u)
            if is_missing(t_cu):
                unreachable = True
                break
            if people_aware:
                cost += t_cu * float(populations[u])
            else:
                cost += t_cu

        if unreachable:
            continue

        if cost < best_cost:
            best_cost = cost
            best_id = c

    return best_id
