from .splits import _splits, _pieces
from .demographics import *
from .partisan import (
    _competitive_districts,
    _swing_districts,
    _party_districts,
    _opp_party_districts,
    _party_wins_by_district,
    _seats,
    _efficiency_gap,
    _mean_median,
    _partisan_bias,
    _partisan_gini,
)
from functools import partial
from gerrychain import Partition
from typing import Iterable, List
from .score_types import *

"""
    Functionality for scoring plans/chains
"""

def summarize(part: Partition, scores: Iterable[Score] = []) -> dict:
    """
    """
    summary = {}
    for score in scores:
        summary[score.name] = score.function(part)
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
    return Score(f"{alias}_splits", partial(_splits, unit=unit, names=names))

def pieces(unit: str, names:bool = False, alias: str = None) -> Score:
    if alias is None:
        alias = unit
    return Score(f"{alias}_pieces", partial(_pieces, unit=unit, names=names))

def competitive_districts(election_cols: Iterable[str], party: str, points_within: float) -> Score:
    return Score("competitive_districts", partial(_competitive_districts, election_cols=election_cols, party=party, points_within=points_within))

def swing_districts(election_cols: Iterable[str], party: str) -> Score:
    return Score("swing_districts", partial(_swing_districts, election_cols=election_cols, party=party))

def party_districts(election_cols: Iterable[str], party: str) -> Score:
    return Score("party_districts", partial(_party_districts, election_cols=election_cols, party=party))

def opp_party_districts(election_cols: Iterable[str], party: str) -> Score:
    return Score("opp_party_districts", partial(_opp_party_districts, election_cols=election_cols, party=party))

def party_wins_by_district(election_cols: Iterable[str], party: str) -> Score:
    return Score("party_wins_by_district", partial(_party_wins_by_district, election_cols=election_cols, party=party))

def seats(election_cols: Iterable[str], party: str) -> Score:
    return Score("seats", partial(_seats, election_cols=election_cols, party=party))

def efficiency_gap(election_cols: Iterable[str]) -> Score:
    return Score("efficiency_gap", partial(_efficiency_gap, election_cols=election_cols))

def mean_median(election_cols: Iterable[str]) -> Score:
    return Score("mean_median", partial(_mean_median, election_cols=election_cols))

def partisan_bias(election_cols: Iterable[str]) -> Score:
    return Score("partisan_bias", partial(_partisan_bias, election_cols=election_cols))

def partisan_gini(election_cols: Iterable[str]) -> Score:
    return Score("partisan_gini", partial(_partisan_gini, election_cols=election_cols))

def demographic_tallies(population_cols: Iterable[str]) -> List[Score]:
    return [
            Score(col, partial(_tally_pop(pop_col=col)))
            for col in population_cols
        ]

def demographic_shares(population_cols: Mapping[str, Iterable[str]]) -> List[Score]:
    scores = []

    for totalpop_col, subpop_cols in population_cols.items():
        scores.append([
            Score(f"{col}_share", partial(_pop_shares(subpop_col=col, totpop_col=totalpop_col)))
            for col in subpop_cols
        ])
    return scores

def gingles_districts(population_cols: Mapping[str, Iterable[str]], threshold: float = 0.5) -> List[Score]:
    scores = []

    for totalpop_col, subpop_cols in population_cols.items():
        scores.append([
            Score(f"{col}_gingles_districts", partial(_gingles_districts(subpop_col=col, totpop_col=totalpop_col, threshold=threshold)))
            for col in subpop_cols
        ])
    return scores