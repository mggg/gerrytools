
from ..auxiliary import Graph


def dualgraph(
        geometries, index=None, geometrycolumn="geometry", colmap={}, buffer=0,
        edges_to_add=[], edges_to_cut=[]
    ):
    """
    Generates a graph dual to the provided geometric data.

    Args:
        geometries: Geometric data represented as a GeoDataFrame.
        index: Unique identifiers; indexing column of `geometries`. If this
            value is not set, vertex labels are integer indices; otherwise, vertex
            labels are the values of this column.
        geometrycolumn: Optional; name of `geoemtries`' geometry column.
            Defaults to `"geometry"`.
        colmap: Optional; maps old column names to new column names.
        buffer: Optional; geometric buffer distance; defaults to `0`.
        edges_to_add: Optional; edges to add to the graph object. Assumed to
            be a list of pairs of objects, e.g. `[(u, v), ...]` where `u` and `v` are
            vertex labels consistent with `index`.
        edges_to_cut: Optional; edges to cut from the graph object. Assumed to
            be a list of pairs of objects, e.g. `[(u, v), ...]` where `u` and `v` are
            vertex labels consistent with `index`.

    Returns:
        A gerrychain `Graph` object dual to the geometric data.
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
