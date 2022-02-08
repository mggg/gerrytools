from gerrychain import Partition
from score_types import *

## demographic tallys
## demographic shares
## demographic num gingles districts.


def _tally_pop(pop_col: str, part: Partition) -> DistrictWideScoreValue:
    return part[pop_col]


def _pop_shares(subpop_col: str, totpop_col: str, part: Partition) -> DistrictWideScoreValue:
    total_pops = part[totpop_col]
    return {d: part[subpop_col][d] / pop for d, pop in total_pops.items()}

def _gingles_districts(subpop_col: str, totpop_col: str, part: Partition, threshold: float = 0.5) -> PlanWideScoreValue:
    subpop_shares = _pop_shares(subpop_col, totpop_col, part)
    return sum([share >= threshold for share in subpop_shares.values()])