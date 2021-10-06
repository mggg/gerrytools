
import geopandas as gpd

def dissolve(geometries, by="DISTRICTN", reset_index=True, keep=[],aggfunc="sum"):
    """
    Dissolves `geometries` on the column `by`. Intended to dissolve a set of
    source geometries (e.g. VTDs, blocks, block groups, etc.) to district
    geometries.

    Args:
        geometries: Set of geometries to be dissolved.
        by: Name of the column used to group objects.
        reset_index: Optional; if true, the index of the resulting GeoDataFrame
            will be set to an integer index, not `by`. 
        keep: Additional columns to keep beyond the geometry and `by` columns.
        aggfunc: Pandas groupby function type when aggregating; defaults to `"sum"`.

    Returns:
        A `GeoDataFrame` containing dissolved geometries and kept columns
        computed by the function designated by `aggfunc`.
    """
    # Pare down the geometries and dissolve.
    geometries = geometries[keep + [by, "geometry"]]
    geometries = geometries.dissolve(by=by, aggfunc=aggfunc)
    if reset_index: geometries = geometries.reset_index()

    return geometries
