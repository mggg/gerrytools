
"""
Basic functionality for evaluating districting plans.
"""

from .splits import splits, pieces
from .population import deviations, unassigned_population
from .contiguity import unassigned_units, contiguous
from .coi import block_level_coi_preservation

__all__ = [
    "splits",
    "pieces",
    "deviations",
    "unassigned_population",
    "unassigned_units",
    "contiguous",
    "block_level_coi_preservation"
]
