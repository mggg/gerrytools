
from gerrychain.updaters import county_splits

def splits(P, units, names=False) -> dict:
    """
    Determines the number of units split by the districting plan.
    
    Bear in mind that this calculates the number of _units split_, not the number
    of _unit splits_: for example, if two districts divides a county into three
    pieces, the former reports one split (as the county is the unit split) and
    the latter reports 

    Args:
        P (Partition): GerryChain Partition object.
        units (list): List of data columns; each assigns a vertex to a unit.
            Generally, these units are counties, VTDs, precincts, etc.
        names (bool, optional): Should the dictionary include the identifiers of
            the split units instead of the number?
    
    Returns:
        A dictionary mapping column names to the number of splits; if the `names`
        parameter is truthy, we return a dictionary mapping column names to the
        list of units split.
    """
    # Check whether the `units` parameter is a list; if not, we force it to one.
    if type(units) != list:
        print("Passed `units` parameter is not a list; forcing to list and continuing.")
        units = [units]
    
    # If the `names` parameter is truthy, we want to get the unique identifiers
    # of the units being split; otherwise, we just report the number. However,
    # the number of units split can be retrieved even when the `names` parameter
    # is falsy by taking the length of the list of unique identifiers.
    if names:
        geometrysplits = {
            unit: [
                name for name, split in county_splits("", unit)(P).items()
                if len(split.contains) > 1
            ]
            for unit in units
        }
    else:
        geometrysplits = {
            unit: sum(
                1 for split in county_splits("", unit)(P).values()
                if len(split.contains) > 1
            )
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
        names (bool, optional): Should the dictionary include the identifiers of
            the split units instead of the number?

    Returns:
        A dictionary mapping column names to the number of pieces; if the `names`
        parameter is truthy, we return a dictionary mapping column names to the
        list of units split.
    """
    # Check whether the `units` parameter is a list; if not, we force it to one.
    if type(units) != list:
        print("Passed `units` parameter is not a list; forcing to list and continuing.")
        units = [units]

    # If the `names` parameter is truthy, we want to get the unique identifiers
    # of the units being split; otherwise, we just report the number. In this
    # case, however, the number of unit pieces *cannot* be retrieved from the
    # list of units split.
    if names:
        geometrysplits = {
            unit: [
                name for name, split in county_splits("", unit)(P).items()
                if len(split.contains) > 1
            ]
            for unit in units
        }
    else:
        geometrysplits = {
            unit: sum(
                len(split.contains) for split in county_splits("", unit)(P).values()
                if len(split.contains) > 1
            )
            for unit in units
        }
    
    return geometrysplits
