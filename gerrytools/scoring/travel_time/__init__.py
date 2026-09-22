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

Symmetric OD assumption
    Every travel time in this package is read as ``t(a, b)`` in one direction
    and treated as interchangeable with ``t(b, a)``; :class:`TravelTimeTable`
    is never checked for symmetry, so a caller building a table from a
    directed source (one-way streets, etc.) is responsible for symmetrizing
    it before scoring.
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
from .weights import PEOPLE_AWARE_MODES, WEIGHT_MODES, WeightMode, trip_weight

__all__ = [
    "PEOPLE_AWARE_MODES",
    "WEIGHT_MODES",
    "CentroidTravelTime",
    "DistrictTravelScores",
    "PairwiseTravelTime",
    "PlanTravelScores",
    "TravelTimeScorer",
    "TravelTimeTable",
    "WeightMode",
    "build_od_from_graph",
    "centroid_travel_time",
    "choose_centroid",
    "pairwise_travel_time",
    "plan_aggregate",
    "score_district",
    "score_plan",
    "trip_weight",
]
