"""Auxiliary and specialized geographical scores for plan evaluation."""
import numpy as np
import geopandas as gpd
from typing import Dict, Callable, Union, Any
from cv2 import minEnclosingCircle
from gerrychain import Graph
from gerrychain import GeographicPartition
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

    </br>

    Below, we provide short proofs of correctness for the optimizations utilized
    in this updater. First, we show that the convex hull of a union of tiled
    geometries is the same as the convex hull of the union of the geometries' convex
    hulls. Next, we show that the convex hull of only the exterior tiled geometries
    of a given polygon is the same as the convex hull of all the geometries.
    Lastly, we show that the minimum bounding circle of the latter is the same
    as the minimum bounding circle of the former.

    <div class="proof">
        <p>
            <i>Proof (equality of convex hulls of unions).</i> Let \(X\) be a simple
            polygon in the plane; let \(S\) be a set of simple polygons \(s_1,
            \dots, s_n\) which tile \(X\), and \(S^*\) a set of polygons where
            \(s^*_i = \text{Hull}(s_i)\). Let \(V\) be the vertices which define
            \(\text{Hull}(\cup S)\), and \(V^*\) the vertices which define
            \(\text{Hull}(\cup S^*)\). We wish to show that \(V = V^*\).
        </p>
        <p>
            (\(\supseteq\)) Each vertex of \(s^*_i\) is a vertex of \(s_i\).
            Consequently, the vertices of \(\cup S^*\) are a subset of \(\cup S\)'s
            vertices, implying that \(\text{Hull}(\cup S^*)\)'s vertices are a subset
            of \(\text{Hull}(\cup S)\)'s. As such, \(V \supseteq V^*\).
        </p>

        <p>
            (\(\subseteq\)) Suppose, for the sake of contradiction, that \(V\)
            contains a vertex \(v\) that is <i>not</i> contained in \(V^*\),
            and that \(v\) is a vertex of the polygon \(s_i\). If \(v\) is not
            in \(V^*\), then it can't be on the hull of \(\cup S^*\); if \(v\)
            can't be on the hull of \(\cup S^*\), then it can't be on the hull
            of \(s^*_i\). If \(v\) isn't on the hull of \(s^*_i\), then it is
            a reflex vertex; if \(v\) is a reflex vertex, then it can't be on
            the convex hull of \(s_i\), and thus can't be on the convex hull of
            \(\cup S\), which is a contradiction. As such, \(V \subseteq V^*\).
        </p>
        <p>
            Because we have \(V \supseteq V^*\) and \(V \subseteq V^*\), we have
            \(V=V^*\), and the convex hull of \(\cup S\) is the same as \(\cup S^*\).
        </p>
    </div>

    <div class="proof">
        <p>
            <i>Proof (Equality of convex hull of exterior).</i> Let \(X\) and
            \(S^*\) be as before.
        </p>
    </div>
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
        boundary_nodes = set.union(*(set(e) for e in partition["cut_edges"])).union(
            partition["boundary_nodes"]
        )
        part_scores = {}

        for part, nodes in partition.parts.items():
            geom = unary_union([
                geometries[node] for node in nodes if node in boundary_nodes
            ]).convex_hull
            coords = np.array(geom.exterior.coords.xy).T.astype(np.float32)
            _, radius = minEnclosingCircle(coords)
            score = float(partition['area'][part] / (pi * radius**2))
            assert 0 < score < 1
            part_scores[part] = score
        return part_scores

    return score_fn
