
"""
Facilities for processing districting plans in a standardized, functional-programmy
way.
"""

from .fetch import submissions, tabularized
from .remap import remap
from .URLs import ids, one, csvs
from .AssignmentCompressor import AssignmentCompressor
from gerrychain.graph import Graph
from gerrychain.partition import Partition

__all__ = [
    "submissions",
    "tabularized",
    "remap",
    "ids",
    "one",
    "csvs",
    "AssignmentCompressor"
]
