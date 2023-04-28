from typing import Union

import gerrychain
from gerrychain.constraints import contiguous as ctgs


def contiguous(P: gerrychain.Partition) -> bool:
    """
    Determines whether the districting plan defined by the partition is
    contiguous.

    Args:
        P (Partition): GerryChain Partition object.

    Returns:
        Whether the districting plan defined by the partition is contiguous.
    """
    return ctgs(P)


def unassigned_units(P: gerrychain.Partition, raw: bool = False) -> Union[float, int]:
    """
    Determines the proportion (or raw number) of units without a district
    assignment. An unassigned unit is a unit without a districting assignment an
    empty/corrupted assignment.

    Args:
        P (Partition): GerryChain Partition object.
        raw (bool, optional): If `True`, report the raw number of unassigned
            units. Defaults to `False`.

    Returns:
        `float` representing the proportion of units that are unassigned (or
        the whole number of unassigned units).
    """
    assignment = P.assignment

    # Retrive the length of the assignment; this corresponds to the number of
    # units which have an assignment key.
    total = len(P.graph.nodes())

    # Next, check for "bad" assignments for units: this includes empty strings
    # and NaNs, for now.
    units_assigned_well = len(
        {k: v for k, v in assignment.items() if v not in ["nan", "NaN", ""]}
    )

    return (
        1 - (units_assigned_well / total) if not raw else (total - units_assigned_well)
    )
