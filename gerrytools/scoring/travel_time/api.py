"""GerryTools-shaped public API for travel-time scores.

Mirrors the convenience-function + metric-object pattern used in
``gerrytools.scoring`` without importing gerrytools (keeps this package
lightweight and FIPS-friendly). When we port upstream, these become thin
wrappers around the same core.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Mapping, Sequence

from .district import DistrictTravelScores, score_district
from .od_table import TravelTimeTable, UnitId
from .plan import PlanAgg, PlanTravelScores, plan_aggregate, score_plan
from .weights import WeightMode

Family = Literal["ptt", "ctt"]
DistrictAgg = Literal["mean", "max"]


def _district_value(
    scores: DistrictTravelScores, family: Family, district_agg: DistrictAgg
) -> float:
    if family == "ptt" and district_agg == "mean":
        return scores.ptt_mean
    if family == "ptt" and district_agg == "max":
        return scores.ptt_max
    if family == "ctt" and district_agg == "mean":
        return scores.ctt_mean
    if family == "ctt" and district_agg == "max":
        return scores.ctt_max
    raise ValueError(f"Unknown family/district_agg: {family!r}, {district_agg!r}")


def pairwise_travel_time(
    parts: Mapping[UnitId, Sequence[UnitId]],
    populations: Mapping[UnitId, float],
    od: TravelTimeTable,
    *,
    weight: WeightMode = "none",
    district_agg: DistrictAgg = "mean",
    plan_agg: PlanAgg = "mean",
) -> float:
    """Convenience: one PTT plan-level number (gerrytools-style one-shot call)."""
    plan = score_plan(parts, populations, od, weight)
    if plan_agg == "mean" and district_agg == "mean":
        return plan.ptt_mean
    if plan_agg == "max_district" and district_agg == "mean":
        return plan.ptt_max_district_mean
    if plan_agg == "global_max":
        return plan.ptt_global_max
    if plan_agg == "mean_of_maxes":
        return plan.ptt_mean_of_maxes
    # Generic path for less common combos (e.g. mean of district means already covered).
    if plan_agg == "mean":
        vals = [_district_value(s, "ptt", district_agg) for s in plan.by_district.values()]
        finite = [v for v in vals if v == v]
        return float(sum(finite) / len(finite)) if finite else float("nan")
    if plan_agg == "max_district":
        vals = [_district_value(s, "ptt", district_agg) for s in plan.by_district.values()]
        finite = [v for v in vals if v == v]
        return float(max(finite)) if finite else float("nan")
    return plan_aggregate(plan, "ptt", plan_agg)


def centroid_travel_time(
    parts: Mapping[UnitId, Sequence[UnitId]],
    populations: Mapping[UnitId, float],
    od: TravelTimeTable,
    *,
    weight: WeightMode = "none",
    district_agg: DistrictAgg = "mean",
    plan_agg: PlanAgg = "mean",
) -> float:
    """Convenience: one CTT plan-level number."""
    plan = score_plan(parts, populations, od, weight)
    if plan_agg == "mean" and district_agg == "mean":
        return plan.ctt_mean
    if plan_agg == "max_district" and district_agg == "mean":
        return plan.ctt_max_district_mean
    if plan_agg == "global_max":
        return plan.ctt_global_max
    if plan_agg == "mean_of_maxes":
        return plan.ctt_mean_of_maxes
    if plan_agg == "mean":
        vals = [_district_value(s, "ctt", district_agg) for s in plan.by_district.values()]
        finite = [v for v in vals if v == v]
        return float(sum(finite) / len(finite)) if finite else float("nan")
    if plan_agg == "max_district":
        vals = [_district_value(s, "ctt", district_agg) for s in plan.by_district.values()]
        finite = [v for v in vals if v == v]
        return float(max(finite)) if finite else float("nan")
    return plan_aggregate(plan, "ctt", plan_agg)


@dataclass(frozen=True, slots=True)
class PairwiseTravelTime:
    """Reusable PTT metric description (port-ready for PlanEvaluator-style use)."""

    weight: WeightMode = "none"
    district_agg: DistrictAgg = "mean"
    plan_agg: PlanAgg = "mean"
    result_name: str = "pairwise_travel_time"

    def score(
        self,
        parts: Mapping[UnitId, Sequence[UnitId]],
        populations: Mapping[UnitId, float],
        od: TravelTimeTable,
    ) -> float:
        return pairwise_travel_time(
            parts,
            populations,
            od,
            weight=self.weight,
            district_agg=self.district_agg,
            plan_agg=self.plan_agg,
        )


@dataclass(frozen=True, slots=True)
class CentroidTravelTime:
    """Reusable CTT metric description."""

    weight: WeightMode = "none"
    district_agg: DistrictAgg = "mean"
    plan_agg: PlanAgg = "mean"
    result_name: str = "centroid_travel_time"

    def score(
        self,
        parts: Mapping[UnitId, Sequence[UnitId]],
        populations: Mapping[UnitId, float],
        od: TravelTimeTable,
    ) -> float:
        return centroid_travel_time(
            parts,
            populations,
            od,
            weight=self.weight,
            district_agg=self.district_agg,
            plan_agg=self.plan_agg,
        )


@dataclass(frozen=True, slots=True)
class TravelTimeScorer:
    """Score a plan once and expose all Layer-A/B fields (batch / debugging)."""

    weight: WeightMode = "none"

    def evaluate(
        self,
        parts: Mapping[UnitId, Sequence[UnitId]],
        populations: Mapping[UnitId, float],
        od: TravelTimeTable,
    ) -> PlanTravelScores:
        return score_plan(parts, populations, od, self.weight)

    def district(
        self,
        units: Sequence[UnitId],
        populations: Mapping[UnitId, float],
        od: TravelTimeTable,
    ) -> DistrictTravelScores:
        return score_district(units, populations, od, self.weight)
