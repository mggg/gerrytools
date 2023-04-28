from typing import Iterable

from gerrychain import Partition
from gerrychain.updaters import Tally

from .types import DistrictWideScoreValue, Numeric, PlanWideScoreValue


def demographic_updaters(
    demographic_keys: Iterable[str], aliases: Iterable[str] = None
):
    updaters = {}
    aliases = aliases if aliases else demographic_keys
    for key, alias in zip(demographic_keys, aliases):
        updaters[alias] = Tally(key, alias=alias)
    return updaters


def _tally_pop(part: Partition, pop_col: str) -> DistrictWideScoreValue:
    return part[pop_col]


def _pop_shares(
    part: Partition, subpop_col: str, totpop_col: str
) -> DistrictWideScoreValue:
    total_pops = part[totpop_col]
    return {d: part[subpop_col][d] / pop for d, pop in total_pops.items()}


def _gingles_districts(
    part: Partition, subpop_col: str, totpop_col: str, threshold: float = 0.5
) -> PlanWideScoreValue:
    subpop_shares = _pop_shares(part, subpop_col, totpop_col)
    return sum([share >= threshold for share in subpop_shares.values()])


def _max_deviation(part: Partition, totpop_col: str, pct: bool = False) -> Numeric:
    totpop_counts = _tally_pop(part, totpop_col)
    ideal_population = sum(totpop_counts.values()) / len(part)
    max_deviation = max([abs(pop - ideal_population) for pop in totpop_counts.values()])
    return max_deviation / ideal_population if pct else max_deviation
