from functools import cache
from gerrychain import Partition
import numpy as np
from typing import Iterable
from score_types import *

@cache
def _election_results(part: Partition, election_cols: Iterable[str], party: str):
    return np.array([
                        np.array([part[e].percent(party, d) for d in sorted(part.parts.keys()) if d != -1])
                        for e in election_cols
                    ])
@cache
def _election_stability(part: Partition, election_cols: Iterable[str], party: str):
    return (_election_results(election_cols, party, part) > 0.5).sum(axis=0)


def competitive_districts(part: Partition, election_cols: Iterable[str], party: str,
                          points_within: float = 0.03) -> PlanWideScoreValue:
    results = _election_results(election_cols, party, part)
    return int(np.logical_and(results > 0.5 - points_within, results < 0.5 + points_within).sum())

def swing_districts(part: Partition, election_cols: Iterable[str], party: str) -> PlanWideScoreValue:
    stability = _election_stability(election_cols, party, part)
    return int(np.logical_and(stability != 0, stability != len(election_cols)).sum())

def party_districts(part: Partition, election_cols: Iterable[str], party: str) -> PlanWideScoreValue:
    stability = _election_stability(election_cols, party, part)
    return int((stability == len(election_cols)).sum())

def opp_party_districts(part: Partition, election_cols: Iterable[str], party: str) -> PlanWideScoreValue:
    stability = _election_stability(election_cols, party, part)
    return int((stability == 0).sum())

def party_wins_by_district(part: Partition, election_cols: Iterable[str], party: str) -> DistrictWideScoreValue:
    stability = _election_stability(election_cols, party, part)
    districts = [d for d in sorted(part.parts.keys()) if d != -1]
    return {d: int(d_wins) for d, d_wins in zip(districts, stability)}

def seats(part: Partition, election_cols: Iterable[str], party: str) -> ElectionWideScoreValue:
    return {part[e].election.name: sum([part[e].won(party, d) for d in part.parts.keys() if d != -1]) for e in election_cols}

def efficiency_gap(part: Partition, election_cols: Iterable[str]) -> ElectionWideScoreValue:
    return {part[e].election.name: part[e].efficiency_gap() for e in election_cols}

def mean_median(part: Partition, election_cols: Iterable[str]) -> ElectionWideScoreValue:
    return {part[e].election.name: part[e].mean_median() for e in election_cols}

def partisan_bias(part: Partition, election_cols: Iterable[str]) -> ElectionWideScoreValue:
    return {part[e].election.name: part[e].partisan_bias() for e in election_cols}