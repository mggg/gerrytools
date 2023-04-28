from functools import cache
from typing import Iterable, Tuple

import numpy as np
from gerrychain import Partition

from .types import DistrictWideScoreValue, Numeric, PlanWideScoreValue, ScoreValue


@cache
def _election_results(part: Partition, election_cols: Tuple[str], party: str):
    return np.array(
        [
            np.array(
                [
                    part[e].percent(party, d)
                    for d in sorted(part.parts.keys())
                    if d != -1
                ]
            )
            for e in election_cols
        ]
    )


@cache
def _election_stability(part: Partition, election_cols: Tuple[str], party: str):
    return (_election_results(part, election_cols, party) > 0.5).sum(axis=0)


def _competitive_contests(
    part: Partition,
    election_cols: Iterable[str],
    party: str,
    points_within: float = 0.03,
) -> PlanWideScoreValue:
    results = _election_results(part, tuple(election_cols), party)
    return int(
        np.logical_and(
            results > 0.5 - points_within, results < 0.5 + points_within
        ).sum()
    )


def _swing_districts(
    part: Partition, election_cols: Iterable[str], party: str
) -> PlanWideScoreValue:
    stability = _election_stability(part, tuple(election_cols), party)
    return int(np.logical_and(stability != 0, stability != len(election_cols)).sum())


def _party_districts(
    part: Partition, election_cols: Iterable[str], party: str
) -> PlanWideScoreValue:
    stability = _election_stability(part, tuple(election_cols), party)
    return int((stability == len(election_cols)).sum())


def _opp_party_districts(
    part: Partition, election_cols: Iterable[str], party: str
) -> PlanWideScoreValue:
    stability = _election_stability(part, tuple(election_cols), party)
    return int((stability == 0).sum())


def _party_wins_by_district(
    part: Partition, election_cols: Iterable[str], party: str
) -> DistrictWideScoreValue:
    stability = _election_stability(part, tuple(election_cols), party)
    districts = [d for d in sorted(part.parts.keys()) if d != -1]
    return {d: int(d_wins) for d, d_wins in zip(districts, stability)}


def _aggregate_seats(
    part: Partition, election_cols: Iterable[str], party: str
) -> PlanWideScoreValue:
    stability = _election_stability(part, tuple(election_cols), party)
    return int(stability.sum())


def _seats(
    part: Partition, election_cols: Iterable[str], party: str, mean: bool = False
) -> ScoreValue:
    result = {
        e: sum([part[e].won(party, d) for d in part.parts.keys() if d != -1])
        for e in election_cols
    }
    return float(np.mean(list(result.values()))) if mean else result


def _responsive_proportionality(
    part: Partition, election_cols: Iterable[str], party: str
) -> PlanWideScoreValue:
    result = [
        (part[e].seats(party) / len(part)) - (part[e].percent(party))
        for e in election_cols
    ]
    return float(np.mean(result))


def _stable_proportionality(
    part: Partition, election_cols: Iterable[str], party: str
) -> PlanWideScoreValue:
    result = [
        abs((part[e].seats(party) / len(part)) - part[e].percent(party))
        for e in election_cols
    ]
    return float(np.mean(result))


def _efficiency_gap(
    part: Partition, election_cols: Iterable[str], mean: bool = False
) -> ScoreValue:
    result = {e: part[e].efficiency_gap() for e in election_cols}
    return float(np.mean(list(result.values()))) if mean else result


def _simplified_efficiency_gap(
    part: Partition, election_cols: Iterable[str], party: str, mean: bool = False
) -> ScoreValue:
    result = {}
    for e in election_cols:
        V = part[e].percent(party)
        S = part[e].seats(party) / len(part)
        result[e] = S + 0.5 - 2 * V
    return float(np.mean(list(result.values()))) if mean else result


def _mean_median(
    part: Partition, election_cols: Iterable[str], mean: bool = False
) -> ScoreValue:
    result = {e: part[e].mean_median() for e in election_cols}
    return float(np.mean(list(result.values()))) if mean else result


def _partisan_bias(
    part: Partition, election_cols: Iterable[str], mean: bool = False
) -> ScoreValue:
    result = {e: part[e].partisan_bias() for e in election_cols}
    return float(np.mean(list(result.values()))) if mean else result


def _partisan_gini(
    part: Partition, election_cols: Iterable[str], mean: bool = False
) -> ScoreValue:
    result = {e: part[e].partisan_gini() for e in election_cols}
    return float(np.mean(list(result.values()))) if mean else result


def _eguia_election(
    part: Partition, e: str, party: str, county_part: Partition, totpop_col: str
) -> Numeric:
    seat_share = part[e].seats(party) / len(part.parts)
    counties = county_part.parts
    county_results = np.array([county_part[e].won(party, c) for c in counties])
    county_pops = np.array([county_part[totpop_col][c] for c in counties])
    ideal = np.dot(county_results, county_pops) / county_pops.sum()
    return float(seat_share - ideal)


def _eguia(
    part: Partition,
    election_cols: Iterable[str],
    party: str,
    county_part: Partition,
    totpop_col: str,
    mean: bool = False,
) -> ScoreValue:
    result = {
        e: _eguia_election(part, e, party, county_part, totpop_col)
        for e in election_cols
    }
    return float(np.mean(list(result.values()))) if mean else result
