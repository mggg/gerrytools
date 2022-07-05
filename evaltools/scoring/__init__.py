
"""
Basic functionality for evaluating districting plans.
"""

from .scores import (
    splits, pieces, competitive_contests, swing_districts, party_districts,
    opp_party_districts, party_wins_by_district, seats, aggregate_seats,
    responsive_proportionality, stable_proportionality, efficiency_gap,
    simplified_efficiency_gap, mean_median, partisan_bias, partisan_gini,
    summarize, summarize_many, demographic_shares, demographic_tallies,
    gingles_districts, max_deviation, eguia
)
from .population import deviations, unassigned_population
from .contiguity import unassigned_units, contiguous
from .demographics import demographic_updaters
from .reock import reock

__all__ = [
    "splits",
    "pieces",
    "competitive_contests",
    "swing_districts",
    "party_districts",
    "opp_party_districts",
    "party_wins_by_district",
    "seats",
    "aggregate_seats",
    "responsive_proportionality",
    "stable_proportionality",
    "efficiency_gap",
    "simplified_efficiency_gap",
    "mean_median",
    "partisan_bias",
    "partisan_gini",
    "summarize",
    "summarize_many",
    "deviations",
    "unassigned_population",
    "unassigned_units",
    "contiguous",
    "reock",
    "demographic_updaters",
    "demographic_tallies",
    "demographic_shares",
    "gingles_districts",
    "max_deviation",
    "eguia",
]
