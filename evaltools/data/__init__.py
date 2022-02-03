
"""
Facilities for processing data and districting plans in a standardized fashion.
"""
from .acs import cvap, acs5
from .census import census, variables
from .fetch import submissions, tabularized, Submission
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
    "AssignmentCompressor",
    "Submission",
    "cvap",
    "acs5",
    "census",
    "variables"
]