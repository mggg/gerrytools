
from geopandas import GeoDataFrame


def dissolve(
    geometries, by="DISTRICTN", reset_index=True, keep=[], aggfunc="sum"
) -> GeoDataFrame:
    """
    Dissolves `geometries` on the column `by`. Intended to dissolve a set of
    source geometries (e.g. VTDs, blocks, block groups, etc.) to district
    geometries.

    Args:
        geometries (GeoDataFrame): Set of geometries to be dissolved.
        by (str): Name of the column used to group objects.
        reset_index (boolean, optional): If true, the index of the resulting
            GeoDataFrame will be set to an integer index, not `by`. Defaults to
            `True`.
        keep (list, optional): Additional columns to keep beyond the geometry
            and `by` columns. Defaults to an empty list, so no additional columns
            are kept.
        aggfunc (str, optional): Pandas groupby function type when aggregating;
            defaults to `"sum"`.

    Returns:
        A `GeoDataFrame` containing dissolved geometries and kept columns
        computed by the function designated by `aggfunc`.
    """
    # Pare down the geometries and dissolve.
    geometries = geometries[keep + [by, "geometry"]]
    geometries = geometries.dissolve(by=by, aggfunc=aggfunc)
    if reset_index:
        geometries = geometries.reset_index()

    return geometries
