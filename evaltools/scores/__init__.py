
"""
Basic functionality for evaluating districting plans.
"""

from .splits import splits, pieces
from .population import deviations, unassigned_population
from .contiguity import unassigned_units, contiguous
from .reock import reock

__all__ = [
    "splits",
    "pieces",
    "deviations",
    "unassigned_population",
    "unassigned_units",
    "contiguous",
    "reock"
]
