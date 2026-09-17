"""Layer A — one number (or NaN) per district.

District travel-time scores are weighted means or weighted maxes over trips.
Means use denominator sum(w), so units are *time* (same as ``t``).
Weighted maxes are in time × weight units.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Mapping, Sequence

import numpy as np

from .centroid import choose_centroid
from .od_table import TravelTimeTable, UnitId
from .weights import WeightMode, trip_weight


@dataclass(frozen=True, slots=True)
class DistrictTravelScores:
    """All Layer-A scores for one district under one weight mode."""

    ptt_mean: float
    ptt_max: float
    ctt_mean: float
    ctt_max: float
    centroid: UnitId | None


def _weighted_mean_and_max(times: list[float], weights: list[float]) -> tuple[float, float]:
    """Return (mean, max of t*w). Mean is NaN if no positive total weight."""
    if not times:
        return float("nan"), float("nan")

    tw = [t * w for t, w in zip(times, weights)]
    w_sum = float(sum(weights))
    mean = float(sum(tw) / w_sum) if w_sum > 0 else float("nan")
    # Weighted max: ignore trips with weight 0 (they are not in the mean either).
    active = [v for v, w in zip(tw, weights) if w > 0]
    if not active:
        # Unweighted mode always has w=1; if we are here, all weights were 0.
        return mean, float("nan")
    return mean, float(max(active))


def pairwise_trips(
    units: Sequence[UnitId],
    populations: Mapping[UnitId, float],
    od: TravelTimeTable,
    weight: WeightMode,
) -> tuple[list[float], list[float]]:
    """Collect finite pairwise times and weights for PTT (i < j)."""
    times: list[float] = []
    weights: list[float] = []
    for i, j in combinations(units, 2):
        t_ij = od.travel_time(i, j)
        if t_ij != t_ij:
            continue
        w = trip_weight(populations[i], populations[j], weight)
        times.append(t_ij)
        weights.append(w)
    return times, weights


def centroid_trips(
    units: Sequence[UnitId],
    populations: Mapping[UnitId, float],
    od: TravelTimeTable,
    weight: WeightMode,
    centroid: UnitId,
) -> tuple[list[float], list[float]]:
    """Collect finite times and weights from centroid to each unit (incl. self)."""
    times: list[float] = []
    weights: list[float] = []
    pop_c = float(populations[centroid])
    for u in units:
        t_cu = od.travel_time(centroid, u)
        if t_cu != t_cu:
            continue
        w = trip_weight(pop_c, float(populations[u]), weight)
        times.append(t_cu)
        weights.append(w)
    return times, weights


def score_district(
    units: Sequence[UnitId],
    populations: Mapping[UnitId, float],
    od: TravelTimeTable,
    weight: WeightMode,
) -> DistrictTravelScores:
    """Compute PTT/CTT mean and max for one district.

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
    ptt_mean, ptt_max = _weighted_mean_and_max(ptt_times, ptt_weights)

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
    ctt_mean, ctt_max = _weighted_mean_and_max(ctt_times, ctt_weights)

    return DistrictTravelScores(
        ptt_mean=ptt_mean,
        ptt_max=ptt_max,
        ctt_mean=ctt_mean,
        ctt_max=ctt_max,
        centroid=centroid,
    )


def is_finite(x: float) -> bool:
    return bool(np.isfinite(x))
