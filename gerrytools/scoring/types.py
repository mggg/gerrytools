from dataclasses import dataclass
from typing import Callable, Mapping, NamedTuple, Union

from geopandas import GeoDataFrame
from gerrychain import Partition

"""
    Typing Definitions:

    * A Score is a named tuple of a name and function that takes a `gerrychain.Partition` instance and
    returns a ScoreValue.  The function associated with the Score should be deterministic, that is
    always return the same value given the same partition.
    * A ScoreValue is either a numeric, a mapping from districts to numerics, or a mapping from
    elections to numerics.
"""

Numeric = Union[float, int]
DistrictID = Union[int, str]
ElectionID = str

PlanWideScoreValue = Numeric
DistrictWideScoreValue = Mapping[DistrictID, Numeric]
ElectionWideScoreValue = Mapping[ElectionID, Numeric]

ScoreValue = Union[PlanWideScoreValue, DistrictWideScoreValue, ElectionWideScoreValue]


@dataclass
class Score:
    name: str
    apply: Callable[[Union[Partition, GeoDataFrame]], ScoreValue]
    dissolved: bool = False
