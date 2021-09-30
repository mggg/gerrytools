
import geopandas as gpd

def dissolve(
        geometries:gpd.GeoDataFrame, by="DISTRICTN", reset_index=True, keep=[],
        aggfunc="sum"
    ) -> gpd.GeoDataFrame:
    """
    Dissolves `geometries` on the column `on`. Intended to dissolve a set of
    source geometries (e.g. VTDs, blocks, block groups, etc.) to district
    geometries.

    :param geometries: Set of geometries to be dissolved.
    :param by: Name of the column used to group objects.
    :param reset_index: Optional; if true, the index of the resulting GeoDataFrame
    will be set to an integer index, not `by`. 
    :param keep: Additional columns to keep beyond the geometry and `by` columns.
    :param aggfunc: Pandas groupby function type when aggregating; defaults to
    `"sum"`.
    """
    # Pare down the geometries and dissolve.
    geometries = geometries[keep + [by, "geometry"]]
    geometries = geometries.dissolve(by=by, aggfunc=aggfunc).reset_index()

    return geometries
