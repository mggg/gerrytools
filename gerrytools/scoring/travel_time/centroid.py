"""Travel-time centroid selection."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .od_table import TravelTimeTable, UnitId
from .weights import CentroidWeightMode, validate_populations


def choose_centroid(
    units: Sequence[UnitId],
    populations: Mapping[Any, float] | None,
    od: TravelTimeTable,
    weight: CentroidWeightMode,
) -> UnitId | None:
    """Select the unit minimizing travel time to the district's units.

    Args:
        units (Sequence[UnitId]): District units in tie-breaking order.
        populations (Mapping[Any, float] | None): Unit populations. Required for destination-
            population weighting and ignored for unweighted selection.
        od (TravelTimeTable): Statewide-connected travel-time table.
        weight (CentroidWeightMode): Centroid weighting policy.

    Returns:
        UnitId | None: Selected unit, or ``None`` for an empty district.
    """
    ordered_units = list(units)
    if not ordered_units:
        return None
    if len(set(ordered_units)) != len(ordered_units):
        raise ValueError("District units must be unique")
    od.validate_connected()
    matrix = od.matrix_for(ordered_units)

    if weight is CentroidWeightMode.DESTINATION_POPULATION:
        if populations is None:
            raise ValueError("population_attr is required for centroid weight 'p'")
        validate_populations(ordered_units, populations)
        destination_populations = np.fromiter(
            (float(populations[unit]) for unit in ordered_units),
            dtype=float,
            count=len(ordered_units),
        )
        costs = matrix @ destination_populations
    else:
        costs = matrix.sum(axis=1)

    return ordered_units[int(np.argmin(costs))]
