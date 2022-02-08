import splits
from functools import partial
from gerrychain import Partition
from typing import Callable, Iterable, Mapping, Union, NamedTuple

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
ScoreValue = Union[Numeric, Mapping[DistrictID, Numeric], Mapping[ElectionID, Numeric]]

class Score(NamedTuple):
    name: str
    function: Callable[[Partition], ScoreValue]


"""
    Functionality for scoring plans/chains
"""

def summarize(part: Partition, scores: Iterable[Score] = []) -> dict:
    """
    """
    summary = {}
    for score in scores:
        summary[score.name] = score.function[part]
    return summary

def summarize_many(parts: Iterable[Partition], scores: Iterable[Score] = [], 
                   output_file: str = None, compress: bool = False):
    """
    """
    if output_file is None:
        return [summarize(part, scores=scores) for part in parts]
    else:
        for part in parts:
            pass


"""
    Commonly used score definitions
"""

def splits(unit: str, names:bool =False, alias: str = None):
    if alias is None:
        alias = unit
    return Score(f"{alias}_splits", partial(splits.splits, unit=unit, names=names))

def pieces(unit: str, names:bool =False, alias: str = None):
    if alias is None:
        alias = unit
    return Score(f"{alias}_pieces", partial(splits.pieces, unit=unit, names=names))