"""GerryTools-shaped public API for travel-time scores.

Mirrors the convenience-function + metric-object pattern used in
``gerrytools.scoring`` without importing gerrytools (keeps this package
lightweight and FIPS-friendly). When we port upstream, these become thin
wrappers around the same core.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping, Sequence

from .district import DistrictTravelScores, score_district
from .od_table import TravelTimeTable
from .plan import PlanAgg, PlanTravelScores, plan_aggregate, score_plan
from .weights import WeightMode

Family = Literal["ptt", "ctt"]

# NOTE on PlanAgg: each value already fixes *both* how a district is
# summarized and how districts are combined into a plan number -- there
# is no independent "district_agg" choice underneath it:
#   "mean"           -> mean of district means
#   "max_district"   -> max of district means  ("worst average district")
#   "global_max"     -> max of district maxes  (single worst trip, plan-wide)
#   "mean_of_maxes"  -> mean of district maxes
# An earlier version of this API exposed district_agg and plan_agg as two
# separate knobs, which implied 8 combinations; only 4 are actually
# meaningful (PlanTravelScores only stores these 4), and the other 4 either
# silently ignored one of the two arguments or recomputed a value that
# duplicated one already on PlanTravelScores. PlanAgg is kept as the single
# source of truth for "how do I turn per-district scores into one number."


def pairwise_travel_time(
    parts: Mapping[Any, Sequence[Any]],
    populations: Mapping[Any, float],
    od: TravelTimeTable,
    *,
    weight: WeightMode = "none",
    plan_agg: PlanAgg = "mean",
) -> float:
    """Convenience: one PTT plan-level number (gerrytools-style one-shot call)."""
    plan = score_plan(parts, populations, od, weight)
    return plan_aggregate(plan, "ptt", plan_agg)


def centroid_travel_time(
    parts: Mapping[Any, Sequence[Any]],
    populations: Mapping[Any, float],
    od: TravelTimeTable,
    *,
    weight: WeightMode = "none",
    plan_agg: PlanAgg = "mean",
) -> float:
    """Convenience: one CTT plan-level number."""
    plan = score_plan(parts, populations, od, weight)
    return plan_aggregate(plan, "ctt", plan_agg)


@dataclass(frozen=True, slots=True)
class PairwiseTravelTime:
    """Reusable PTT metric description (port-ready for PlanEvaluator-style use)."""

    weight: WeightMode = "none"
    plan_agg: PlanAgg = "mean"
    # TODO(port): once wired into the compiled PlanEvaluator engine, result_name
    # becomes the column/field name the engine records this metric under. Unused
    # until then.
    result_name: str = "pairwise_travel_time"

    def score(
        self,
        parts: Mapping[Any, Sequence[Any]],
        populations: Mapping[Any, float],
        od: TravelTimeTable,
    ) -> float:
        return pairwise_travel_time(
            parts,
            populations,
            od,
            weight=self.weight,
            plan_agg=self.plan_agg,
        )


@dataclass(frozen=True, slots=True)
class CentroidTravelTime:
    """Reusable CTT metric description."""

    weight: WeightMode = "none"
    plan_agg: PlanAgg = "mean"
    # TODO(port): see PairwiseTravelTime.result_name.
    result_name: str = "centroid_travel_time"

    def score(
        self,
        parts: Mapping[Any, Sequence[Any]],
        populations: Mapping[Any, float],
        od: TravelTimeTable,
    ) -> float:
        return centroid_travel_time(
            parts,
            populations,
            od,
            weight=self.weight,
            plan_agg=self.plan_agg,
        )


@dataclass(frozen=True, slots=True)
class TravelTimeScorer:
    """Score a plan once and expose all Layer-A/B fields (batch / debugging)."""

    weight: WeightMode = "none"

    def evaluate(
        self,
        parts: Mapping[Any, Sequence[Any]],
        populations: Mapping[Any, float],
        od: TravelTimeTable,
    ) -> PlanTravelScores:
        return score_plan(parts, populations, od, self.weight)

    def district(
        self,
        units: Sequence[Any],
        populations: Mapping[Any, float],
        od: TravelTimeTable,
    ) -> DistrictTravelScores:
        return score_district(units, populations, od, self.weight)
