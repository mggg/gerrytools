import splits
import demographics
import partisan
from functools import partial
from gerrychain import Partition
from typing import Iterable, List
from score_types import *

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

def splits(unit: str, names:bool = False, alias: str = None) -> Score:
    if alias is None:
        alias = unit
    return Score(f"{alias}_splits", partial(splits.splits, unit=unit, names=names))

def pieces(unit: str, names:bool = False, alias: str = None) -> Score:
    if alias is None:
        alias = unit
    return Score(f"{alias}_pieces", partial(splits.pieces, unit=unit, names=names))


def demographic_tallies(population_cols: Iterable[str]) -> List[Score]:
    return [
            Score(col, partial(demographics._tally_pop(pop_col=col)))
            for col in population_cols
        ]

def demographic_shares(population_cols: Mapping[str, Iterable[str]]) -> List[Score]:
    scores = []

    for totalpop_col, subpop_cols in population_cols.items():
        scores.append([
            Score(f"{col}_share", partial(demographics._pop_shares(subpop_col=col, totpop_col=totalpop_col)))
            for col in subpop_cols
        ])
    return scores

def gingles_districts(population_cols: Mapping[str, Iterable[str]], threshold: float = 0.5) -> List[Score]:
    scores = []

    for totalpop_col, subpop_cols in population_cols.items():
        scores.append([
            Score(f"{col}_gingles_districts", partial(demographics._gingles_districts(subpop_col=col, totpop_col=totalpop_col, threshold=threshold)))
            for col in subpop_cols
        ])
    return scores