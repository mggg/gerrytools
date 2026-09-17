"""Layer B — plan-level summaries from district scores.

Plan mean averages district Layer-A scores.
Three max-style summaries are offered for experimentation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal, Mapping, Sequence

from .district import DistrictTravelScores, is_finite, score_district
from .od_table import TravelTimeTable, UnitId
from .weights import WeightMode

DistrictScoreKey = Literal["ptt_mean", "ptt_max", "ctt_mean", "ctt_max"]
PlanAgg = Literal["mean", "max_district", "global_max", "mean_of_maxes"]


@dataclass(frozen=True, slots=True)
class PlanTravelScores:
    """Plan summaries plus per-district detail."""

    by_district: dict[UnitId, DistrictTravelScores]
    # Plan means of district *means*
    ptt_mean: float
    ctt_mean: float
    # Max of district means (old-style "worst average district")
    ptt_max_district_mean: float
    ctt_max_district_mean: float
    # Global max of weighted trips (via district max scores)
    ptt_global_max: float
    ctt_global_max: float
    # Mean of per-district weighted maxes
    ptt_mean_of_maxes: float
    ctt_mean_of_maxes: float


def _mean_finite(values: Iterable[float]) -> float:
    vals = [v for v in values if is_finite(v)]
    return float(sum(vals) / len(vals)) if vals else float("nan")


def _max_finite(values: Iterable[float]) -> float:
    vals = [v for v in values if is_finite(v)]
    return float(max(vals)) if vals else float("nan")


def score_plan(
    parts: Mapping[UnitId, Sequence[UnitId]],
    populations: Mapping[UnitId, float],
    od: TravelTimeTable,
    weight: WeightMode,
) -> PlanTravelScores:
    """Score every district, then build plan-level summaries.

    Args:
        parts: Map district_id → sequence of unit ids in that district.
        populations: Population for every unit that appears in ``parts``.
        od: Statewide (or larger) travel-time table.
        weight: Trip weight mode.
    """
    by_district: dict[UnitId, DistrictTravelScores] = {}
    for district_id, units in parts.items():
        by_district[district_id] = score_district(units, populations, od, weight)

    rows = list(by_district.values())

    return PlanTravelScores(
        by_district=by_district,
        ptt_mean=_mean_finite(r.ptt_mean for r in rows),
        ctt_mean=_mean_finite(r.ctt_mean for r in rows),
        ptt_max_district_mean=_max_finite(r.ptt_mean for r in rows),
        ctt_max_district_mean=_max_finite(r.ctt_mean for r in rows),
        ptt_global_max=_max_finite(r.ptt_max for r in rows),
        ctt_global_max=_max_finite(r.ctt_max for r in rows),
        ptt_mean_of_maxes=_mean_finite(r.ptt_max for r in rows),
        ctt_mean_of_maxes=_mean_finite(r.ctt_max for r in rows),
    )


def plan_aggregate(scores: PlanTravelScores, family: Literal["ptt", "ctt"], agg: PlanAgg) -> float:
    """Pick one plan-level number from a scored plan (handy for experiments)."""
    if family == "ptt":
        if agg == "mean":
            return scores.ptt_mean
        if agg == "max_district":
            return scores.ptt_max_district_mean
        if agg == "global_max":
            return scores.ptt_global_max
        if agg == "mean_of_maxes":
            return scores.ptt_mean_of_maxes
    else:
        if agg == "mean":
            return scores.ctt_mean
        if agg == "max_district":
            return scores.ctt_max_district_mean
        if agg == "global_max":
            return scores.ctt_global_max
        if agg == "mean_of_maxes":
            return scores.ctt_mean_of_maxes
    raise ValueError(f"Unknown family/agg: {family!r}, {agg!r}")
