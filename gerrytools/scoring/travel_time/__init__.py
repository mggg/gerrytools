"""Travel-time plan scores (PTT / CTT) over a unit-to-unit drive-time table.

These metrics are **not** wired into the compiled ``PlanEvaluator`` engine. Callers
supply a finished :class:`TravelTimeTable` (built offline from a routable network).
Routes may leave a district; scores are best interpreted on fully routable states
and road-contiguous plans.

Weight modes
    ``none``
        Each unit pair counts equally.
    ``p+p``
        Weight ``pop_a + pop_b``.
    ``p*p``
        Weight ``pop_a * pop_b`` (person-pair interpretation).

Centroid selection follows weight *intent* without a ``p_c`` self-penalty:
``none`` minimizes ``sum_u t(c,u)``; ``p+p`` / ``p*p`` minimize ``sum_u t(c,u) p_u``.
"""

from .api import (
    CentroidTravelTime,
    PairwiseTravelTime,
    TravelTimeScorer,
    centroid_travel_time,
    pairwise_travel_time,
)
from .build_od import build_od_from_graph
from .centroid import choose_centroid
from .district import DistrictTravelScores, score_district
from .od_table import TravelTimeTable
from .plan import PlanTravelScores, plan_aggregate, score_plan
from .weights import WEIGHT_MODES, WeightMode, trip_weight

__all__ = [
    "WEIGHT_MODES",
    "WeightMode",
    "trip_weight",
    "TravelTimeTable",
    "build_od_from_graph",
    "choose_centroid",
    "DistrictTravelScores",
    "score_district",
    "PlanTravelScores",
    "score_plan",
    "plan_aggregate",
    "pairwise_travel_time",
    "centroid_travel_time",
    "PairwiseTravelTime",
    "CentroidTravelTime",
    "TravelTimeScorer",
]
