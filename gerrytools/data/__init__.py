"""
Facilities for processing data and districting plans in a standardized fashion.
"""

from .acs import acs5, cvap
from .AssignmentCompressor import AssignmentCompressor
from .census import census10, census20, variables
from .estimatecvap import estimatecvap2010, estimatecvap2020, fetchgeometries
from .fetch import Submission, submissions, tabularized
from .geometries import geometries20
from .remap import remap
from .URLs import csvs, ids, one

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
    "census10",
    "geometries20",
]
