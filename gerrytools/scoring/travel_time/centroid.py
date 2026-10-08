"""Choose a drive-time center (centroid unit) for CTT.

``none`` minimizes ``sum_u t(c, u)``. ``p`` minimizes
``sum_u t(c, u) * p_u``. The candidate centroid's population never enters
selection: the centroid unit represents a location, not a population endpoint.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from .od_table import TravelTimeTable, UnitId, is_missing
from .weights import CENTROID_WEIGHT_MODES, CentroidWeightMode, validate_populations


def choose_centroid(
    units: Sequence[Any],
    populations: Mapping[Any, float],
    od: TravelTimeTable,
    weight: CentroidWeightMode,
) -> UnitId | None:
    """Return the centroid unit id, or None if no valid candidate exists.

    The OD table must be statewide connected. Every unit is an eligible
    candidate, including zero-population units, because candidate population
    is not part of the CTT definition.

    Ties break toward whichever tied candidate appears first in ``units``.
    (``UnitId`` is only required to be ``Hashable``, not orderable, so we
    can't break ties with ``<`` on the ids themselves; input order is the
    only ordering guaranteed to exist.)
    """
    if not units:
        return None

    if weight not in CENTROID_WEIGHT_MODES:
        raise ValueError(
            f"Unknown centroid weight mode {weight!r}; expected one of {CENTROID_WEIGHT_MODES}"
        )

    od.validate_connected()

    # Redundant with score_district's own od.require_units(units) call on the
    # only current caller, but choose_centroid is exported as public API
    # (see __init__.py) and may be called directly, so it validates its own
    # inputs rather than relying on a caller to have done so.
    od.require_units(units)
    population_weighted = weight == "p"
    if population_weighted:
        validate_populations(units, populations)

    best_id: UnitId | None = None
    best_cost = float("inf")

    # Scanning in input order and only ever accepting a *strict* improvement
    # means the first candidate to reach a given cost is the one that sticks,
    # which is exactly the "tie breaks toward input order" rule above.
    for c in units:
        cost = 0.0
        for u in units:
            t_cu = od.travel_time(c, u)
            if is_missing(t_cu):
                raise ValueError(
                    f"Missing travel time from centroid candidate {c!r} to unit {u!r}; "
                    "validate statewide connectivity before scoring"
                )
            if population_weighted:
                cost += t_cu * float(populations[u])
            else:
                cost += t_cu

        if cost < best_cost:
            best_cost = cost
            best_id = c

    return best_id
