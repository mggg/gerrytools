
from gerrychain import Graph
import geopandas as gpd
import networkx as nx
from typing import Optional, Dict, List, Tuple


def dualgraph(
        geometries: gpd.GeoDataFrame, index: Optional[str]=None,
        geometrycolumn: Optional[str]="geometry", colmap: Optional[Dict]={},
        buffer: Optional[float]=0, edges_to_add: Optional[List[Tuple]]=[],
        edges_to_cut: Optional[List[Tuple]]=[]
    ) -> Graph:
    """
    Generates a graph dual to the provided geometric data.

    :param geometries: Geometric data represented as a GeoDataFrame.
    :param index: Unique identifiers; indexing column of `geometries`. If this
    value is not set, vertex labels are integer indices; otherwise, vertex
    labels are the values of this column.
    :param geometrycolumn: Optional; name of `geoemtries`' geometry column.
    Defaults to `"geometry"`.
    :param colmap: Optional; maps old column names to new column names.
    :param buffer: Optional; geometric buffer distance; defaults to `0`.
    :param edges_to_add: Optional; edges to add to the graph object. Assumed to
    be a list of pairs of objects, e.g. `[(u, v), ...]` where `u` and `v` are
    vertex labels consistent with `index`.
    :param edges_to_cut: Optional; edges to cut from the graph object. Assumed to
    be a list of pairs of objects, e.g. `[(u, v), ...]` where `u` and `v` are
    vertex labels consistent with `index`.
    """
    # Buffer geometries by default.
    geometries[geometrycolumn] = geometries.buffer(buffer)

    # Set indices and rename columns.
    if index: geometries = geometries.set_index(index)
    if colmap: geometries = geometries.rename(colmap, axis=1)

    # Generate the dual graph.
    dg = Graph.from_geodataframe(geometries)

    # Add and remove extraneous edges.
    for add in edges_to_add: dg.add_edge(*add)
    for cut in edges_to_cut: dg.remove_edge(*cut)

    # Return the graph!
    return dg
