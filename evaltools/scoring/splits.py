from gerrychain import Partition
from gerrychain.updaters import CountySplit, county_splits
from typing import List, Union
from .score_types import *

def _splits(P: Partition, unit: str, names: bool = False) -> Union[int, List[str]]:
    """
    Determines the number of units split by the districting plan.
    
    Bear in mind that this calculates the number of *unit splits*, not the number
    of *units split*: for example, if a district divides a county into three
    pieces, the former reports two splits (as a unit divided into three pieces is
    cut twice), while the latter would report one split (as there is one county 
    being split).

    Args:
        P (Partition): GerryChain Partition object.
        unit (str): Data column; each assigns a vertex to a unit.
            Generally, these units are counties, VTDs, precincts, etc.
        names (bool, optional): Whether we return the identifiers of the things
            being split.
    
    Returns:
        The number of splits or the list of things split.
    """
    geometrysplits = [
                        identifier 
                        for identifier, split in county_splits("", unit)(P).items()
                        if split.split != CountySplit.NOT_SPLIT
                    ]

    return geometrysplits if names else len(geometrysplits)


def _pieces(P: Partition, unit: str, names: bool = False) -> Union[int, List[str]]:
    """
    Determines the number of "unit pieces" produced by the plan. For example,
    consider a state with 100 counties. Suppose that one county is split twice,
    and another once. Then, there are 3 + 2 = 5 "pieces," disregarding the
    counties kept whole.
    
    Bear in mind that this calculates the number of _unit splits_, not the number
    of _units split_: for example, if a district divides a county into three
    pieces, the former reports two splits (as a unit divided into three pieces is
    cut twice), while the latter would report one split (as there is one county 
    being split).

    Args:
        P (Partition): GerryChain Partition object.
        unit (list): Data column; each assigns a vertex to a unit.
            Generally, these units are counties, VTDs, precincts, etc.
        names (bool, optional): Whether we return the identifiers of the things
            being pieced.

    Returns:
        The number of pieces or the list of things split.
    """

    if names:
        return splits(P, unit, names)

    geometrypieces = sum(
            len(split.contains)
            for split in county_splits("", unit)(P).values()
            if split.split != CountySplit.NOT_SPLIT
        )
    return geometrypieces

def _traversals(part: Partition, unit: str) -> PlanWideScoreValue:
    """
    TODO: Document.
    """
    unique_region_pairs = {district: set() for district in part.assignment.values()}
    for (n1, n2) in part.graph.edges:
        if (n1, n2) not in part.cut_edges:
            district = part.assignment[n1]
            region1 = part.graph.nodes[n1][self.unit]
            region2 = part.graph.nodes[n2][self.unit]
            if region1 != region2: 
                region_pair = tuple(sorted([region1, region2]))
                unique_region_pairs[district].add(region_pair)
    num_traversals = sum([len(pair_set) for district, pair_set in unique_region_pairs.items()])
    return num_traversals
