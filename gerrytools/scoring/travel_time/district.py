"""Layer A — PTT and CTT numbers (or NaN) for one district.

District travel-time scores keep trip weights in the numerator and divide by
the number of trips in that sum:

- PTT: ``sum_{i<j} (t * w) / (# pairs)``
- CTT: ``sum_u (t * w_ctt) / (# centroid-to-unit trips)``

PTT supports ``none`` (time), ``p+p`` (people-time), and ``p*p``
(people²-time). CTT supports ``none`` (time) and destination-population
weight ``p`` (people-time). The centroid unit's population is never used.

Weighted maxes are ``max (t * w)`` in the same weight units (no trip division).
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Any, Mapping, Sequence

import numpy as np

from .centroid import choose_centroid
from .od_table import TravelTimeTable, UnitId, is_missing
from .weights import (
    CENTROID_WEIGHT_MODES,
    PAIRWISE_WEIGHT_MODES,
    CentroidWeightMode,
    PairwiseWeightMode,
    centroid_trip_weight,
    pairwise_trip_weight,
    validate_populations,
)


@dataclass(frozen=True, slots=True)
class DistrictTravelScores:
    """All Layer-A scores for one district under the selected family weights."""

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
    weight: PairwiseWeightMode,
) -> tuple[list[float], list[float]]:
    """Collect pairwise times and weights for PTT (i < j)."""
    times: list[float] = []
    weights: list[float] = []
    for i, j in combinations(units, 2):
        t_ij = od.travel_time(i, j)
        if is_missing(t_ij):
            raise ValueError(
                f"Missing travel time for pair ({i!r}, {j!r}); "
                "validate statewide connectivity before scoring"
            )
        w = pairwise_trip_weight(populations[i], populations[j], weight)
        times.append(t_ij)
        weights.append(w)
    return times, weights


def centroid_trips(
    units: Sequence[Any],
    populations: Mapping[Any, float],
    od: TravelTimeTable,
    weight: CentroidWeightMode,
    centroid: UnitId,
) -> tuple[list[float], list[float]]:
    """Collect times and weights from centroid to each unit (incl. self).

    Reads ``od.travel_time(centroid, u)``, i.e. centroid-outbound. ``od`` is
    assumed symmetric (see :class:`~.od_table.TravelTimeTable`), so this is
    equivalent to the inbound direction and there's no separate case to handle.
    """
    times: list[float] = []
    weights: list[float] = []
    for u in units:
        t_cu = od.travel_time(centroid, u)
        if is_missing(t_cu):
            raise ValueError(
                f"Missing travel time from centroid {centroid!r} to unit {u!r}; "
                "validate statewide connectivity before scoring"
            )
        w = centroid_trip_weight(float(populations[u]), weight)
        times.append(t_cu)
        weights.append(w)
    return times, weights


def score_district(
    units: Sequence[Any],
    populations: Mapping[Any, float],
    od: TravelTimeTable,
    *,
    ptt_weight: PairwiseWeightMode = "none",
    ctt_weight: CentroidWeightMode = "none",
) -> DistrictTravelScores:
    """Compute PTT/CTT per-trip scores and maxes for one district.

    The OD table must be statewide connected. Singletons (fewer than 2
    units) yield NaN for all four scores.
    """
    units = list(units)
    if ptt_weight not in PAIRWISE_WEIGHT_MODES:
        raise ValueError(
            f"Unknown pairwise weight mode {ptt_weight!r}; expected one of {PAIRWISE_WEIGHT_MODES}"
        )
    if ctt_weight not in CENTROID_WEIGHT_MODES:
        raise ValueError(
            f"Unknown centroid weight mode {ctt_weight!r}; expected one of {CENTROID_WEIGHT_MODES}"
        )
    od.validate_connected()
    od.require_units(units)
    if ptt_weight != "none" or ctt_weight == "p":
        validate_populations(units, populations)
    if len(units) < 2:
        return DistrictTravelScores(
            ptt_mean=float("nan"),
            ptt_max=float("nan"),
            ctt_mean=float("nan"),
            ctt_max=float("nan"),
            centroid=None,
        )
    ptt_times, ptt_weights = pairwise_trips(units, populations, od, ptt_weight)
    ptt_mean, ptt_max = _weighted_sum_per_trip_and_max(ptt_times, ptt_weights)

    centroid = choose_centroid(units, populations, od, ctt_weight)
    if centroid is None:
        return DistrictTravelScores(
            ptt_mean=ptt_mean,
            ptt_max=ptt_max,
            ctt_mean=float("nan"),
            ctt_max=float("nan"),
            centroid=None,
        )

    ctt_times, ctt_weights = centroid_trips(units, populations, od, ctt_weight, centroid)
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
