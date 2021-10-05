
from gerrychain.updaters import county_splits
from typing import List, Union
from ..geography import Graph, Partition


def splits(G:Graph, assignment:Union[str,dict], units:List[str]) -> dict:
    """
    Determines the number of units split by the districting plan.
    
    Bear in mind that this calculates the number of *unit splits*, not the number
    of *units split*: for example, if a district divides a county into three
    pieces, the former reports two splits (as a unit divided into three pieces is
    cut twice), while the latter would report one split (as there is one county 
    being split).

    :param G: `Graph` object.
    :param assignment: Column name or dictionary (keyed by vertex IDs) which
    assigns vertices to districts.
    :param units: List of data columns; each assigns a vertex to a unit. Generally,
    these units are counties, VTDs, precincts, etc.
    :returns: A dictionary mapping column names to the number of splits.
    """
    # Create the partition and get the function for calculating splits.
    partition = Partition(graph=G, assignment=assignment)
    geometrysplits = {}

    # For each of the unit columns, get the number of splits.
    for unit in units:
        splitter = county_splits("", unit)
        geometrysplits[unit] = sum(
            (len(value.contains))-1
            for value in splitter(partition).values()
        )

    return geometrysplits
