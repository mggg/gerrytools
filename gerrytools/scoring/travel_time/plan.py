"""Layer B — plan-level summaries from district scores.

Plan mean averages district Layer-A scores. Three max-style summaries are
offered for experimentation. A non-finite district value invalidates the
corresponding plan summary; districts are never silently dropped.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Literal, Mapping, Sequence

from .district import DistrictTravelScores, is_finite, score_district
from .od_table import TravelTimeTable, UnitId
from .weights import CentroidWeightMode, PairwiseWeightMode

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


def _mean_if_all_finite(values: Iterable[float]) -> float:
    """Return the mean only when every district contributes a finite score."""
    vals = list(values)
    if not vals or not all(is_finite(v) for v in vals):
        return float("nan")
    return float(sum(vals) / len(vals))


def _max_if_all_finite(values: Iterable[float]) -> float:
    """Return the maximum only when every district contributes a finite score."""
    vals = list(values)
    if not vals or not all(is_finite(v) for v in vals):
        return float("nan")
    return float(max(vals))


def score_plan(
    parts: Mapping[Any, Sequence[Any]],
    populations: Mapping[Any, float],
    od: TravelTimeTable,
    *,
    ptt_weight: PairwiseWeightMode = "none",
    ctt_weight: CentroidWeightMode = "none",
) -> PlanTravelScores:
    """Score every district, then build plan-level summaries.

    Args:
        parts: Map district_id → sequence of unit ids in that district.
        populations: Population for every unit that appears in ``parts``.
        od: Statewide (or larger) travel-time table.
        ptt_weight: Pairwise trip weight mode.
        ctt_weight: Centroid-to-unit trip weight mode.

    Notes:
        ``od`` must pass :meth:`TravelTimeTable.validate_connected`. A
        non-finite district score propagates to its plan-level summaries.
    """
    by_district: dict[UnitId, DistrictTravelScores] = {}
    for district_id, units in parts.items():
        by_district[district_id] = score_district(
            units,
            populations,
            od,
            ptt_weight=ptt_weight,
            ctt_weight=ctt_weight,
        )

    rows = list(by_district.values())

    return PlanTravelScores(
        by_district=by_district,
        ptt_mean=_mean_if_all_finite(r.ptt_mean for r in rows),
        ctt_mean=_mean_if_all_finite(r.ctt_mean for r in rows),
        ptt_max_district_mean=_max_if_all_finite(r.ptt_mean for r in rows),
        ctt_max_district_mean=_max_if_all_finite(r.ctt_mean for r in rows),
        ptt_global_max=_max_if_all_finite(r.ptt_max for r in rows),
        ctt_global_max=_max_if_all_finite(r.ctt_max for r in rows),
        ptt_mean_of_maxes=_mean_if_all_finite(r.ptt_max for r in rows),
        ctt_mean_of_maxes=_mean_if_all_finite(r.ctt_max for r in rows),
    )


def plan_aggregate(
    scores: PlanTravelScores,
    family: Literal["ptt", "ctt"],
    agg: PlanAgg | str,
) -> float:
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
