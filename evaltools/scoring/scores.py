from .splits import _splits, _pieces
from .demographics import (
    _pop_shares,
    _tally_pop,
    _gingles_districts,
)
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
    _eguia
)
from functools import partial
from gerrychain import Partition, Graph
from typing import Iterable, List, Mapping
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

def splits(unit: str, names : bool = False, alias: str = None) -> Score:
    """
    Score representing the number of units split by the districting plan.
    
    Bear in mind that this calculates the number of *unit splits*, not the number
    of *units split*: for example, if a district divides a county into three
    pieces, the former reports two splits (as a unit divided into three pieces is
    cut twice), while the latter would report one split (as there is one county 
    being split).

    Args:
        unit (str): Data column; each assigns a vertex to a unit.
            Generally, these units are counties, VTDs, precincts, etc.
        names (bool, optional): Whether we return the identifiers of the things
            being split.
    
    Returns:
        A score object with the name `{alias}_splits` and associated function that takes a
        partition and returns a PlanWideScoreValue for the number of splits.
    """
    if alias is None:
        alias = unit
    return Score(f"{alias}_splits", partial(_splits, unit=unit, names=names))

def pieces(unit: str, names:bool = False, alias: str = None) -> Score:
    """
    Score representing the number of "unit pieces" produced by the plan. For example,
    consider a state with 100 counties. Suppose that one county is split twice,
    and another once. Then, there are 3 + 2 = 5 "pieces," disregarding the
    counties kept whole.
    
    Bear in mind that this calculates the number of _unit splits_, not the number
    of _units split_: for example, if a district divides a county into three
    pieces, the former reports two splits (as a unit divided into three pieces is
    cut twice), while the latter would report one split (as there is one county 
    being split).

    Args:
        unit (str): Data column; each assigns a vertex to a unit.
            Generally, these units are counties, VTDs, precincts, etc.
        names (bool, optional): Whether we return the identifiers of the things
            being split.
    
    Returns:
        A score object with the name `{alias}_pieces` and associated function that takes a
        partition and returns a PlanWideScoreValue for the number of pieces.
    """
    if alias is None:
        alias = unit
    return Score(f"{alias}_pieces", partial(_pieces, unit=unit, names=names))

def competitive_districts(election_cols: Iterable[str], party: str, points_within: float = 0.03) -> Score:
    """
    Score representing the number of competitive districts in a plan.

    Args:
        election_cols (Iterable[str]): The names of the election updaters over which to compute
            results for.
        party (str): The "point of view" political party.
        points_within (float, optional): The margin from 0.5 that is considered competitive.
            Default is 0.03, corresponding to a competitive range of 47%-53%.
    
    Returns:
        A score object with name `competitive_districts` and associated function that takes a
        partition and returns a PlanWideScoreValue for the number of competitive districts.
    """
    return Score("competitive_districts", partial(_competitive_districts, election_cols=election_cols,
                                                  party=party, points_within=points_within))

def swing_districts(election_cols: Iterable[str], party: str) -> Score:
    """
    Score representing the number of swing districts in a plan.  A swing districts is one that is
    not solely won by a single party over a set of elections.

    Args:
        election_cols (Iterable[str]): The names of the election updaters over which to compute
            results for.
        party (str): The "point of view" political party.

    Returns:
        A score object with name `swing_districts` and associated function that takes a partition
        and returns a PlanWideScoreValue for the number of swing districts.
    """
    return Score("swing_districts", partial(_swing_districts, election_cols=election_cols, party=party))

def party_districts(election_cols: Iterable[str], party: str) -> Score:
    """
    Score representing the number of districts in a plan that are always won by the POV party over
    a set of elections.

    Args:
        election_cols (Iterable[str]): The names of the election updaters over which to compute
            results for.
        party (str): The "point of view" political party.

    Returns:
        A score object with name `party_districts` and associated function that takes a partition
        and returns a PlanWideScoreValue for the number of safe POV party districts.
    """
    return Score("party_districts", partial(_party_districts, election_cols=election_cols, party=party))

def opp_party_districts(election_cols: Iterable[str], party: str) -> Score:
    """
    Score representing the number of districts in a plan that are always won by the opposition party
    over a set of elections.
    Note that this assumes that all elections are two-party races.  In the case where elections have
    more than two parties running this score represents the number of districts that are never won
    by the POV party.

    Args:
        election_cols (Iterable[str]): The names of the election updaters over which to compute
            results for.
        party (str): The "point of view" political party.

    Returns:
        A score object with name `opp_party_districts` and associated function that takes a
        partition and returns a PlanWideScoreValue for the number of safe opposition party districts.
    """
    return Score("opp_party_districts", partial(_opp_party_districts, election_cols=election_cols, party=party))

def party_wins_by_district(election_cols: Iterable[str], party: str) -> Score:
    """
    Score representing how many elections the POV party won in each district in a given plan.

    Args:
        election_cols (Iterable[str]): The names of the election updaters over which to compute
            results for.
        party (str): The "point of view" political party.

    Returns:
        A score object with name `party_wins_by_district` and associated function that takes a
        partition and returns a DistrictWideScoreValue for the number of elections won by the POV
        party in each district.
    """
    return Score("party_wins_by_district", partial(_party_wins_by_district, election_cols=election_cols, party=party))

def seats(election_cols: Iterable[str], party: str) -> Score:
    """
    Score representing how many seats (districts) within a given plan the POV party won in each
    election 

    Args:
        election_cols (Iterable[str]): The names of the election updaters over which to compute
            results for.
        party (str): The "point of view" political party.

    Returns:
        A score object with name `{party}_seats` and associated function that takes a partition and
        returns an ElectionWideScoreValue for the number of seats won by the POV party for each
        election.
    """
    return Score(f"{party}_seats", partial(_seats, election_cols=election_cols, party=party))

def efficiency_gap(election_cols: Iterable[str]) -> Score:
    """
    Score representing the efficiency gap metric of a plan with respect to a set of elections.

    Args:
        election_cols (Iterable[str]): The names of the election updaters over which to compute
            results for.

    Returns:
        A score object with name `efficiency_gap`  and associated function that takes a partition
        and returns a PlanWideScoreValue for efficiency gap metric.
    """
    return Score("efficiency_gap", partial(_efficiency_gap, election_cols=election_cols))

def mean_median(election_cols: Iterable[str]) -> Score:
    """
    Score representing the mean median metric of a plan with respect to a set of elections.

    Args:
        election_cols (Iterable[str]): The names of the election updaters over which to compute
            results for.

    Returns:
        A score object with name `mean_median` and associated function that takes a partition and
        returns a PlanWideScoreValue for the mean median metric.
    """
    return Score("mean_median", partial(_mean_median, election_cols=election_cols))

def partisan_bias(election_cols: Iterable[str]) -> Score:
    """
    Score representing the partitisan bias metric of a plan with respect to a set of elections.

    Args:
        election_cols (Iterable[str]): The names of the election updaters over which to compute
            results for.
    Returns:
        A score object with name `partisan_bias` and associated function that takes a partition and
        returns a PlanWideScoreValue for partisan bias metric.
    """
    return Score("partisan_bias", partial(_partisan_bias, election_cols=election_cols))

def partisan_gini(election_cols: Iterable[str]) -> Score:
    """
    Score representing the partisan gini metric of a plan with respect to a set of elections.

    Args:
        election_cols (Iterable[str]): The names of the election updaters over which to compute
            results for.
        party (str): The "point of view" political party.

    Returns:
        A score object with name `partisan_gini` and associated function that takes a partition and
        returns a PlanWideScoreValue for the partisan gini metric.
    """
    return Score("partisan_gini", partial(_partisan_gini, election_cols=election_cols))

def eguia(election_cols: Iterable[str], party: str, graph: Graph, updaters: Mapping[str, Callable[[Partition], ScoreValue]],
          county_col: str, totpop_col: str = "population") -> Score:
    """
    Score representing

    Args:
        election_cols (Iterable[str]): The names of the election updaters over which to compute
            results for.
        party (str): The "point of view" political party.
        graph (gerrychain.Graph): The underlying dual graph of a partition.  Used to generated a
            plan of the counties.
        updaters (Mapping[str, Callable[[gerrychain.Partition], ScoreValue]]):  A set of updaters
            that contains a tally for the total population by district and the election updaters
            whose names are listed in election_cols.
        county_col (str): The column name in the dual graph that encodes the county assignment of
            each unit.
        totpop_col (str, optional): The name of the updater that computes total population by
            district.

    Returns:
        A score object with name `eguia` and associated function that takes a partition and returns
        a PlanWideScoreValue for the eguia metric. 
    """
    county_part = Partition(graph, county_col, updaters=updaters)
    return Score("eguia", partial(_eguia, election_cols=election_cols, party=party, 
                                  county_part=county_part, totpop_col=totpop_col))


def demographic_tallies(population_cols: Iterable[str]) -> List[Score]:
    return [
            Score(col, partial(_tally_pop, pop_col=col))
            for col in population_cols
        ]

def demographic_shares(population_cols: Mapping[str, Iterable[str]]) -> List[Score]:
    scores = []

    for totalpop_col, subpop_cols in population_cols.items():
        scores.extend([
            Score(f"{col}_share", partial(_pop_shares, subpop_col=col, totpop_col=totalpop_col))
            for col in subpop_cols
        ])
    return scores

def gingles_districts(population_cols: Mapping[str, Iterable[str]], threshold: float = 0.5) -> List[Score]:
    scores = []

    for totalpop_col, subpop_cols in population_cols.items():
        scores.extend([
            Score(f"{col}_gingles_districts", partial(_gingles_districts, subpop_col=col, 
                                                      totpop_col=totalpop_col, threshold=threshold))
            for col in subpop_cols
        ])
    return scores