from gerrychain.constraints.validity import deviation_from_ideal as deviation
from gerrychain.updaters import Tally


def deviations(P, popcolumn) -> dict:
    """
    Determines the districting plan's population deviation percentages.

    Args:
        P (Partition): GerryChain Partition object.
        popcolumn (str): Column for tallying the desired population.

    Returns:
        A dictionary which maps district names to population deviation percentages.
    """
    # Create the partition and the corresponding updater.
    poptally = Tally(popcolumn, alias=popcolumn)
    P.updaters = {popcolumn: poptally}

    # Return a dictionary that maps district names to population deviation
    # percentages.
    return deviation(P, attribute=popcolumn)


def unassigned_population(P, popcolumn):
    """
    Determines the number of unassigned people in the districting plan.

    Args:
        P: `Partition` object.
        popcolumn: Column for tallying the desired population.

    Returns:
        Returns a
    """
    pass
