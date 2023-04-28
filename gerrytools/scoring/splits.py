from typing import List, Union

from gerrychain import Partition
from gerrychain.updaters import CountySplit

from ..geometry import dataframe


def _grouper(P: Partition, unit: str, popcol: str = None) -> list:
    """
    A nice little subroutine for doing the necessary pandas operations that we
    don't want to repeat.

    Args:
        P (Partition): GerryChain Partition object.
        unit (str): Data column; each assigns a vertex to a unit. Generally, these
            units are counties, VTDs, precincts, etc.
        popcol (str, optional): The population column on the `Partition`'s dual
            graph. If this is passed, then a unit is only considered "split" if
            the _populated_ base units end up in different districts.
        names (bool, optional): Whether we return the identifiers of the things
            being split.

    Returns:
        A list of `(identifier, DataFrame)` pairs which represent the set of
        child geometries of a single parent geometry; for example, the identifier
        could be a VTD identifier, and the `DataFrame` the data attached to blocks
        belonging to that VTD.
    """
    # Not to be confused with a probability distribution function. That's not what
    # this is.
    pdf = dataframe(
        P,
        index="id",
        assignment="DISTRICT",
        columns=[unit] + ([popcol] if popcol else []),
    )

    # Group by unit, then get the units split.
    groups = list(pdf.groupby(by=unit))

    return [
        (identifier, group[group[popcol] > 0].copy()) if popcol else (identifier, group)
        for identifier, group in groups
    ]


def _splits(
    P: Partition,
    unit: str,
    names: bool = False,
    popcol: str = None,
    how: str = "pandas",
    unit_info_updater_col: str = None,
) -> Union[int, List[str]]:
    """
    Determines the number of units split by the districting plan.

    Bear in mind that this calculates the number of *unit splits*, not the number
    of *units split*: for example, if a district divides a county into three
    pieces, the former reports two splits (as a unit divided into three pieces is
    cut twice), while the latter would report one split (as there is one county
    being split).

    Args:
        P (Partition): GerryChain Partition object.
        unit (str): Data column; each assigns a vertex to a unit. Generally, these
            units are counties, VTDs, precincts, etc.
        popcol (str, optional): The population column on the `Partition`'s dual
            graph. If this is passed, then a unit is only considered "split" if
            the _populated_ base units end up in different districts.
        how (str, optional): How do we perform these calculations on the back
            end? Acceptable values are `"pandas"` and `"gerrychain"`; defaults to
            `"pandas"`.
        names (bool, optional): Whether we return the identifiers of the things
            being split.
        unit_info_updater_col (str, optional): The name of the corresponsing
            county_splits updater in the partition.  If using "gerrychain" version
            and not specified defaults to the name of unit.

    Returns:
        The number of splits or the list of things split.
    """
    # Validate the `how` parameter.
    if how not in {"pandas", "gerrychain"}:
        print(f'"{how}" is not a valid parameter to `how`. Defaulting to pandas.')
        how = "pandas"

    # If we're calculating splits from a dataframe, create the dataframe and do
    # the required operations.
    if how == "pandas":
        # Get the groups for each unit, then do the appropriate calculations.
        groups = _grouper(P, unit, popcol)
        geometrysplits = [
            identifier
            for identifier, group in groups
            if len(group["DISTRICT"].unique()) > 1
        ]

    # Otherwise, do things the normal way!
    if how == "gerrychain":
        if unit_info_updater_col is None:
            unit_info_updater_col = unit
        geometrysplits = [
            identifier
            for identifier, split in P[unit_info_updater_col].items()
            if split.split != CountySplit.NOT_SPLIT
        ]

    return geometrysplits if names else len(geometrysplits)


def _pieces(
    P: Partition,
    unit: str,
    names: bool = False,
    popcol: str = None,
    how: str = "pandas",
    unit_info_updater_col: str = None,
) -> Union[int, List[str]]:
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
        unit (str): Data column; each assigns a vertex to a unit. Generally, these
            units are counties, VTDs, precincts, etc.
        popcol (str, optional): The population column on the `Partition`'s dual
            graph. If this is passed, then a unit is only considered "split" if
            the _populated_ base units end up in different districts.
        how (str, optional): How do we perform these calculations on the back
            end? Acceptable values are `"pandas"` and `"gerrychain"`; defaults to
            `"pandas"`.
        names (bool, optional): Whether we return the identifiers of the things
            being split.
        unit_info_updater_col (str, optional): The name of the corresponsind county_splits updater
            in the partition.  If using "gerrychain" version and not specified defaults to the name
            of unit.

    Returns:
        The number of pieces or the list of things split.
    """
    # Validate the `how` parameter.
    if how not in {"pandas", "gerrychain"}:
        print(f'"{how}" is not a valid parameter to `how`. Defaulting to pandas.')
        how = "pandas"

    # If they just want the list of names, return the splits.
    if names:
        return _splits(P, unit, names=names, popcol=popcol, how=how)

    # Otherwise, do some similar stuff to the splitting except that we're getting
    # the number of pieces instead of whether the unit's split.
    if how == "pandas":
        groups = _grouper(P, unit, popcol=popcol)
        geometrypieces = sum(
            [
                len(group["DISTRICT"].unique())
                for _, group in groups
                if len(group["DISTRICT"].unique()) > 1
            ]
        )

    if how == "gerrychain":
        if unit_info_updater_col is None:
            unit_info_updater_col = unit
        geometrypieces = sum(
            len(split.contains)
            for split in P[unit_info_updater_col].values()
            if split.split != CountySplit.NOT_SPLIT
        )

    return geometrypieces
