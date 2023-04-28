import geopandas as gpd
import gerrychain

from .optimize import calculate_dispersion, minimize_dispersion


def dispersion_updater_closure(
    units: gpd.GeoDataFrame, enacted_col: str, pop_col: str, verbose: bool = False
):
    """
    An updater to calculate best possible dispersion for a `gerrychain.Partition` object.

    Args:
        units: The units to optimize on. E.g. Census blocks.
        enacted_col: The column in the GeoDataFrame with the enacted districts.
        proposed_col: The column in the GeoDataFrame with the proposed districts.
        extra_constraints: Optional; A function that can add extra constraints
            to the model, such as parity (in the case of WI).
        verbose: If true, do not suppress solver output. Otherwise, stay quiet.

    Returns:
        An updater that calculates the minimal core dispersion of a Partition object.
    """

    def updater(partition: gerrychain.Partition):
        units["partition"] = partition.assignment.to_series()

        relabeling = minimize_dispersion(
            units, enacted_col, "partition", pop_col, verbose=verbose
        )
        units["partition"] = units["partition"].apply(lambda x: relabeling[x])

        return calculate_dispersion(units, enacted_col, "partition", pop_col)

    return updater
