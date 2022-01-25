
from gerrychain.updaters import county_splits

def splits(P, units, names=False) -> dict:
    """
    Determines the number of units split by the districting plan.
    
    Bear in mind that this calculates the number of *unit splits*, not the number
    of *units split*: for example, if a district divides a county into three
    pieces, the former reports two splits (as a unit divided into three pieces is
    cut twice), while the latter would report one split (as there is one county 
    being split).

    Args:
        P (Partition): GerryChain Partition object.
        units (list): List of data columns; each assigns a vertex to a unit.
            Generally, these units are counties, VTDs, precincts, etc.
        names (bool, optional): Whether we return the identifiers of the things
            being split.
    
    Returns:
        A dictionary mapping column names to the number of splits or the list
        of things split.
    """
    if not names:
        geometrysplits = {
            unit: sum(
                1
                for split in county_splits("", unit)(P).values()
                if len(split.contains) > 1
            )
            for unit in units
        }
    else:
        geometrysplits = {
            unit: [
                identifier
                for identifier, split in county_splits("", unit)(P).items()
                if len(split.contains) > 1
            ]
            for unit in units
        }

    return geometrysplits


def pieces(P, units, names=False) -> dict:
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
        units (list): List of data columns; each assigns a vertex to a unit.
            Generally, these units are counties, VTDs, precincts, etc.
        names (bool, optional): Whether we return the identifiers of the things
            being pieced.

    Returns:
        A dictionary mapping column names to the number of pieces *or* the list of
        things cut into pieces.
    """
    if not names:
        geometrypieces = {
            unit: sum(
                len(split.contains)
                for split in county_splits("", unit)(P).values()
                if len(split.contains) > 1
            )
            for unit in units
        }
    else:
        geometrypieces = {
            unit: [
                identifier
                for identifier, split in county_splits("", unit)(P).items()
                if len(split.contains) > 1
            ]
            for unit in units
        }

    return geometrypieces
