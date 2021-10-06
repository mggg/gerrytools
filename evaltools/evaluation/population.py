
from gerrychain.constraints.validity import deviation_from_ideal as deviation
from gerrychain.updaters import Tally
from ..auxiliary import Partition


def deviations(P:Partition, popcolumn:str) -> dict:
    """
    Determines the districting plan's population deviation percentages.

    :param P: `Partition` object.
    :param popcolumn: Column for tallying the desired population.
    """
    # Create the partition and the corresponding updater.
    poptally = Tally(popcolumn, alias=popcolumn)
    P.updaters = { popcolumn: poptally }

    # Return a dictionary that maps district names to population deviation
    # percentages.
    return deviation(P, attribute=popcolumn)
