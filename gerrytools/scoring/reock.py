
import numpy as np
import geopandas as gpd
from typing import Dict, Callable, Union, Any
from cv2 import minEnclosingCircle
from gerrychain import Graph, GeographicPartition
from gerrychain.updaters import boundary_nodes
from shapely.ops import unary_union
from math import pi


def reock(
    geodata: Union[gpd.GeoDataFrame, Graph]
) -> Callable[[GeographicPartition], Dict[Any, float]]:
    r"""
    Makes a Reock score function specialized to `geodata`.

    The Reock score of a district is its area defined by the area of its
    minimum enclosing circle. Computing the area of this circle is nontrivial,
    and the Reock score is distinct from other popular compactness scores
    (cut edges, Polsby-Popper) in its reliance on full district geometries
    (rather than district areas and perimeters, which can be computed
    efficiently from unit-level statistics).

    We precompute convex hulls of all geometries in `geodata` and
    build an index. The resulting score function uses this index for
    fast geometric computations; the function can only be used with
    dual graphs derived from `geodata`.

    Args:
        geodata (gpd.GeoDataFrame, Graph): Geographical data to precompute
            geometries from.
    Returns:
        A per-district Reock score updater specialized to `geodata`.
    """
    if isinstance(geodata, gpd.GeoDataFrame):
        geometries = dict(geodata.geometry.apply(lambda p: p.convex_hull))
    elif isinstance(geodata, Graph):
        geometries = {
            node: geom.convex_hull
            for node, geom in geodata.nodes('geometry')
        }
    else:
        raise ValueError(
            'Geodata must be a GeoDataFrame or a gerrychain.Graph.')

    def score_fn(partition: GeographicPartition) -> Dict[Any, float]:
        partition.updaters.update({"boundary_nodes": boundary_nodes})

        boundary = set.union(*(set(e) for e in partition["cut_edges"])).union(
            partition["boundary_nodes"]
        )
        part_scores = {}

        for part, nodes in partition.parts.items():
            geom = unary_union([
                geometries[node] for node in nodes if node in boundary
            ]).convex_hull
            coords = np.array(geom.exterior.coords.xy).T.astype(np.float32)
            _, radius = minEnclosingCircle(coords)
            score = float(partition['area'][part] / (pi * radius**2))
            assert 0 < score < 1
            part_scores[part] = score
        return part_scores

    return score_fn
