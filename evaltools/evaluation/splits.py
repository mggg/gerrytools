
from gerrychain.updaters import county_splits

def splits(P, units):
    """
    Determines the number of units split by the districting plan.
    
    Bear in mind that this calculates the number of *unit splits*, not the number
    of *units split*: for example, if a district divides a county into three
    pieces, the former reports two splits (as a unit divided into three pieces is
    cut twice), while the latter would report one split (as there is one county 
    being split).

    Args:
        P: `Partition` object.
        units: List of data columns; each assigns a vertex to a unit. Generally,
            these units are counties, VTDs, precincts, etc.
    
    Returns:
        A dictionary mapping column names to the number of splits.
    """
    geometrysplits = {
        unit: sum(
            (len(value.contains))-1
            # Takes advantage of the fact that P is a Partition and `county_splits`
            # returns a function. We don't need to name the partition, and the
            # column of units being split by the plan are in `units`, so we can
            # simply call the function returned by `county_splits`, passing it
            # our Partition `P`.
            for value in county_splits("", unit)(P).values()
        )
        for unit in units
    }

    return geometrysplits


def pieces(P, units):
    """
    Determines the number of "unit pieces" produced by the plan. For example,
    consider a state with 100 counties. Suppose that one county is split twice,
    and another in half. Then, there are 3 + 2 = 5 "pieces," disregarding the
    counties kept whole.
    
    Bear in mind that this calculates the number of *unit splits*, not the number
    of *units split*: for example, if a district divides a county into three
    pieces, the former reports two splits (as a unit divided into three pieces is
    cut twice), while the latter would report one split (as there is one county 
    being split).

    Args:
        P: `Partition` object.
        units: List of data columns; each assigns a vertex to a unit. Generally,
            these units are counties, VTDs, precincts, etc.

    Returns:
        A dictionary mapping column names to the number of splits.
    """
    geometrypieces = {
        unit: sum(
            len(value.contains) if len(value.contains) > 1 else 0
            # Takes advantage of the fact that P is a Partition and `county_splits`
            # returns a function. We don't need to name the partition, and the
            # column of units being split by the plan are in `units`, so we can
            # simply call the function returned by `county_splits`, passing it
            # our Partition `P`.
            for value in county_splits("", unit)(P).values()
        )
        for unit in units
    }

    return geometrypieces
