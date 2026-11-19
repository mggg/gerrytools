"""District travel-time scores from a precomputed statewide table.

``pairwise_travel_time`` averages travel time across unordered pairs of units in each district.
``centroid_travel_time`` selects the district unit minimizing travel time to all district units,
then averages trips from that unit. Population-weighted modes retain their weights in the score's
units and divide by the number of trips.
"""

from .api import centroid_travel_time, pairwise_travel_time, travel_time_scores
from .od_table import TravelTimeTable
from .weights import CentroidWeightMode, PairwiseWeightMode

__all__ = [
    "CentroidWeightMode",
    "PairwiseWeightMode",
    "TravelTimeTable",
    "centroid_travel_time",
    "pairwise_travel_time",
    "travel_time_scores",
]
