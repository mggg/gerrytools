
from gerrychain.constraints.validity import deviation_from_ideal as deviation
from gerrychain.updaters import Tally
from typing import Union
from ..geography import Graph, Partition


def deviations(G: Graph, assignment:Union[str,dict], popcolumn:str) -> dict:
    """
    Determines the districting plan's population deviation percentages.

    :param G: `Graph` object.
    :param assignment: Column name or dictionary (keyed by vertex IDs) which
    assigns vertices to districts.
    :param popcolumn: Column for tallying the desired population.
    """
    # Create the partition and the corresponding updater.
    poptally = Tally(popcolumn, alias=popcolumn)
    partition = Partition(graph=G, assignment=assignment)
    partition.updaters = { popcolumn: poptally }

    # Return a dictionary that maps district names to population deviation
    # percentages.
    return deviation(partition, attribute=popcolumn)

