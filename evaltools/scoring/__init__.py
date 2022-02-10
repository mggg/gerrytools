
"""
Basic functionality for evaluating districting plans.
"""

# from .splits import splits, pieces
from .scores import (
    splits,
    pieces,
    competitive_districts,
    swing_districts,
    party_districts,
    opp_party_districts,
    party_wins_by_district,
    seats,
    efficiency_gap,
    mean_median,
    partisan_bias,
    partisan_gini,
    summarize,
)
from .population import deviations, unassigned_population
from .contiguity import unassigned_units, contiguous
from .reock import reock

__all__ = [
    "splits",
    "pieces",
    "competitive_districts",
    "swing_districts",
    "party_districts",
    "opp_party_districts",
    "party_wins_by_district",
    "seats",
    "efficiency_gap",
    "mean_median",
    "partisan_bias",
    "partisan_gini",
    "summarize",
    "deviations",
    "unassigned_population",
    "unassigned_units",
    "contiguous",
    "reock"
]
