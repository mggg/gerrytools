
"""
Facilities for processing data and districting plans in a standardized fashion.
"""
from .acs import cvap, acs5
from .census import census20, census10, variables
from .estimatecvap import estimatecvap2010, estimatecvap2020, fetchgeometries
from .fetch import submissions, tabularized, Submission
from .remap import remap
from .URLs import ids, one, csvs
from .AssignmentCompressor import AssignmentCompressor

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
    "census20",
    "variables",
    "estimatecvap2010",
    "estimatecvap2020",
    "fetchgeometries",
    "census10"
]
