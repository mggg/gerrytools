"""Travel-time plan scores (PTT / CTT) over a unit-to-unit drive-time table.

These metrics are **not** wired into the compiled ``PlanEvaluator`` engine. Callers
supply a finished :class:`TravelTimeTable` (built offline from a routable network).
Routes may leave a district. Every included unit must be mutually reachable through
the statewide road network; scoring rejects a disconnected table instead of silently
dropping unreachable trips.

Weight modes
    ``none``
        Each unit pair counts equally.
    ``p+p``
        Weight ``pop_a + pop_b``.
    ``p*p``
        Weight ``pop_a * pop_b`` (person-pair interpretation).

Centroid selection follows weight *intent* without a ``p_c`` self-penalty:
``none`` minimizes ``sum_u t(c,u)``; ``p+p`` / ``p*p`` minimize ``sum_u t(c,u) p_u``.

Symmetric OD definition
    Every scoring time is the arithmetic mean of the two directed route times.
    :class:`TravelTimeTable` enforces symmetry, and
    :meth:`TravelTimeTable.from_directed_matrix` performs the averaging.

Invalid districts
    A singleton district has no pairwise trip and receives NaN district scores.
    Non-finite district scores propagate to plan summaries; plan aggregation never
    silently removes a district.
"""

from .api import (
    CentroidTravelTime,
    PairwiseTravelTime,
    TravelTimeScorer,
    centroid_travel_time,
    pairwise_travel_time,
)
from .build_od import build_od_from_graph
from .builder import (
    DEFAULT_FERRY_SPEED_KPH,
    FerryBuildSummary,
    TravelTimeBuildReport,
    TravelTimeBuildResult,
    add_vehicle_ferries,
    build_osm_drive_graph,
    build_travel_time_table,
    build_travel_time_table_from_pbf,
    census_representative_points,
    compute_directed_travel_times,
    parse_osm_duration_seconds,
    snap_units_to_graph,
)
from .centroid import choose_centroid
from .district import DistrictTravelScores, score_district
from .od_table import TravelTimeTable
from .plan import PlanTravelScores, plan_aggregate, score_plan
from .table_io import (
    TravelTimeArtifact,
    load_travel_time_table,
    save_travel_time_table,
)
from .weights import PEOPLE_AWARE_MODES, WEIGHT_MODES, WeightMode, trip_weight

__all__ = [
    "PEOPLE_AWARE_MODES",
    "WEIGHT_MODES",
    "DEFAULT_FERRY_SPEED_KPH",
    "CentroidTravelTime",
    "DistrictTravelScores",
    "FerryBuildSummary",
    "PairwiseTravelTime",
    "PlanTravelScores",
    "TravelTimeScorer",
    "TravelTimeArtifact",
    "TravelTimeBuildReport",
    "TravelTimeBuildResult",
    "TravelTimeTable",
    "WeightMode",
    "add_vehicle_ferries",
    "build_osm_drive_graph",
    "build_od_from_graph",
    "build_travel_time_table",
    "build_travel_time_table_from_pbf",
    "census_representative_points",
    "centroid_travel_time",
    "choose_centroid",
    "compute_directed_travel_times",
    "load_travel_time_table",
    "pairwise_travel_time",
    "plan_aggregate",
    "parse_osm_duration_seconds",
    "score_district",
    "score_plan",
    "save_travel_time_table",
    "snap_units_to_graph",
    "trip_weight",
]
