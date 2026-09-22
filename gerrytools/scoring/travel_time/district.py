"""Layer A — one number (or NaN) per district.

District travel-time scores keep trip weights in the numerator and divide by
the number of trips in that sum:

- PTT: ``sum_{i<j} (t * w) / (# finite pairs)``
- CTT: ``sum_u (t * w) / (# finite centroid trips)``

Units therefore depend on the weight mode:

- ``none``: time
- ``p+p``: people-time (per trip)
- ``p*p``: people²-time (per trip)

Weighted maxes are ``max (t * w)`` in the same weight units (no trip division).
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Any, Mapping, Sequence

import numpy as np

from .centroid import choose_centroid
from .od_table import TravelTimeTable, UnitId, is_missing
from .weights import WeightMode, trip_weight


@dataclass(frozen=True, slots=True)
class DistrictTravelScores:
    """All Layer-A scores for one district under one weight mode."""

    ptt_mean: float
    ptt_max: float
    ctt_mean: float
    ctt_max: float
    centroid: UnitId | None


def _weighted_sum_per_trip_and_max(
    times: list[float],
    weights: list[float],
) -> tuple[float, float]:
    """Return (sum(t*w)/n_trips, max of t*w). Score is NaN if there are no trips."""
    n_trips = len(times)
    if n_trips < 1:
        return float("nan"), float("nan")

    tw = [t * w for t, w in zip(times, weights)]
    score = float(sum(tw) / n_trips)
    active = [v for v, w in zip(tw, weights) if w > 0]
    if not active:
        return score, float("nan")
    return score, float(max(active))


def pairwise_trips(
    units: Sequence[Any],
    populations: Mapping[Any, float],
    od: TravelTimeTable,
    weight: WeightMode,
) -> tuple[list[float], list[float]]:
    """Collect finite pairwise times and weights for PTT (i < j)."""
    times: list[float] = []
    weights: list[float] = []
    for i, j in combinations(units, 2):
        t_ij = od.travel_time(i, j)
        if is_missing(t_ij):
            continue
        w = trip_weight(populations[i], populations[j], weight)
        times.append(t_ij)
        weights.append(w)
    return times, weights


def centroid_trips(
    units: Sequence[Any],
    populations: Mapping[Any, float],
    od: TravelTimeTable,
    weight: WeightMode,
    centroid: UnitId,
) -> tuple[list[float], list[float]]:
    """Collect finite times and weights from centroid to each unit (incl. self).

    Reads ``od.travel_time(centroid, u)``, i.e. centroid-outbound. ``od`` is
    assumed symmetric (see :class:`~.od_table.TravelTimeTable`), so this is
    equivalent to the inbound direction and there's no separate case to handle.
    """
    times: list[float] = []
    weights: list[float] = []
    pop_c = float(populations[centroid])
    for u in units:
        t_cu = od.travel_time(centroid, u)
        if is_missing(t_cu):
            continue
        w = trip_weight(pop_c, float(populations[u]), weight)
        times.append(t_cu)
        weights.append(w)
    return times, weights


def score_district(
    units: Sequence[Any],
    populations: Mapping[Any, float],
    od: TravelTimeTable,
    weight: WeightMode,
) -> DistrictTravelScores:
    """Compute PTT/CTT per-trip scores and maxes for one district.

    Singletons (fewer than 2 units) yield NaN for all four scores.
    """
    units = list(units)
    if len(units) < 2:
        return DistrictTravelScores(
            ptt_mean=float("nan"),
            ptt_max=float("nan"),
            ctt_mean=float("nan"),
            ctt_max=float("nan"),
            centroid=None,
        )

    od.require_units(units)

    ptt_times, ptt_weights = pairwise_trips(units, populations, od, weight)
    ptt_mean, ptt_max = _weighted_sum_per_trip_and_max(ptt_times, ptt_weights)

    centroid = choose_centroid(units, populations, od, weight)
    if centroid is None:
        return DistrictTravelScores(
            ptt_mean=ptt_mean,
            ptt_max=ptt_max,
            ctt_mean=float("nan"),
            ctt_max=float("nan"),
            centroid=None,
        )

    ctt_times, ctt_weights = centroid_trips(units, populations, od, weight, centroid)
    ctt_mean, ctt_max = _weighted_sum_per_trip_and_max(ctt_times, ctt_weights)

    return DistrictTravelScores(
        ptt_mean=ptt_mean,
        ptt_max=ptt_max,
        ctt_mean=ctt_mean,
        ctt_max=ctt_max,
        centroid=centroid,
    )


def is_finite(x: float) -> bool:
    return bool(np.isfinite(x))
