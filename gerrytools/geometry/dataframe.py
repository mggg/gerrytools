import pandas as pd
from gerrychain import Partition


def dataframe(
    P: Partition, index: str = "id", assignment: str = "DISTRICT", columns: list = None
) -> pd.DataFrame:
    """
    Converts a `Partition` into a `DataFrame`.

    Args:
        P (Partition): GerryChain `Partition` object to have its data framed.
        index (str, optional): Graph attribute to use as an index. The `networkx`
            default name is `"id"`.
        assignment (str, optional): Column name for assignment.
        columns (list, optional): List of columns to add to the dataframe, not
            including the index. If `None` (or another falsy value), gets all
            columns.

    Returns:
        `DataFrame` with attached graph data.
    """
    # Create dataframe.
    gdf = pd.DataFrame.from_records(
        {index: v, **d} for v, d in P.graph.nodes(data=True)
    )

    # Assign vertices.
    assignedvertices = P.assignment.to_dict()
    gdf[assignment] = gdf[index].map(assignedvertices)

    # Drop columns if necessary.
    if columns:
        gdf = gdf[[assignment, index] + columns]

    return gdf
