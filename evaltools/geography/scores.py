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
    Lastly, we show that the minimum bounding circle of a given polygon is the
    same as the minimum bounding circle of the polygon's convex hull.

    <div class="proof">
        <p>
            <i>Proof (equality of convex hulls of unions).</i> Let \(X\) be a simple
            polygon in the plane; let \(S\) be a set of simple polygons \(s_1,
            \dots, s_n\) which tile \(X\) such that \(\cup S = X\), and \(S^*\)
            a set of polygons where \(s^*_i = \text{Hull}(s_i)\). Denote the
            union of polygons in \(S^*\) by \(X^*\), so  that \(\cup S^* = X^*\).
            Let \(V\) be the vertices which define \(\text{Hull}(X)\), and \(V^*\)
            the vertices which define  \(\text{Hull}(X^*)\). We wish to show that
            \(V = V^*\).
        </p>
        <p>
            (\(\supseteq\)) Each vertex of \(s^*_i\) is a vertex of \(s_i\).
            Consequently, the vertices of \(X^*\) are a subset of \(X\)'s
            vertices, implying that \(\text{Hull}(X^*)\)'s vertices are a subset
            of \(\text{Hull}(X)\)'s. As such, \(V \supseteq V^*\).
        </p>

        <p>
            (\(\subseteq\)) Suppose, for the sake of contradiction, that \(V\)
            contains a vertex \(v\) that is <i>not</i> contained in \(V^*\),
            and that \(v\) is a vertex of the polygon \(s_i\). If \(v\) is not
            in \(V^*\), then it can't be on the hull of \(X^*\); if \(v\)
            can't be on the hull of \(X^*\), then it can't be on the hull
            of \(s^*_i\). If \(v\) isn't on the hull of \(s^*_i\), then it is
            a reflex vertex; if \(v\) is a reflex vertex, then it can't be on
            the convex hull of \(X\), which is a contradiction. As such, \(V
            \subseteq V^*\).
        </p>
        <p>
            Because we have \(V \supseteq V^*\) and \(V \subseteq V^*\), we have
            \(V=V^*\), and the convex hull of \(X\) is the same as the convex
            hull of \(X^*\).
        </p>
    </div>

    <div class="proof">
        <p>
            <i>Proof (Equality of convex hull of exterior).</i> Let \(S^*\), \(X^*\),
            and \(V^*\) be as before. Let \(\partial X^*\) be \(X^*\)'s <i>boundary</i>,
            the set of points for which all \(\epsilon\)-neighborhoods intersect
            both the interior and exterior faces of \(X^*\). Let \(I^*\) be the
            subset of \(S^*\)'s polygons which do not contain a point on the
            boundary (i.e. the <i>interior</i> polygons), and let \(E^*\) be the
            subset of \(S^*\)'s polygons which contain a point on the boundary
            (i.e. the <i>exterior</i> polygons). Note that \(S^* = I^* \sqcup
            E^*\).
        </p>
        <p>
            Because each vertex in \(V^*\) is a boundary point, \(E^*\) contains
            all polygons with a vertex in \(V^*\). Because each vertex belongs
            to a polygon in \(E^*\), we know that \(V^*\) is also the set of
            vertices of \(\text{Hull}(\cup E^*)\). Now, because the hulls' vertices
            are the same, we have that $$\text{Hull}(X^*) = \text{Hull}(\cup E^*)$$
            which, because \(X^* = \cup S^*\), implies that $$\text{Hull}(\cup E^*)
            = \text{Hull}\big(\cup(E^* \sqcup I^*)\big) = \text{Hull}(\cup S^*).$$
        </p>
    </div>

    <div class="proof">
        <p>
            <i>Proof (equality of minimum bounding circles).</i> Given a polygon
            \(P\), its minimum bounding disk \(D\) – whose boundary is the minimum
            bounding circle \(C\) – necessarily contains \(P\)'s convex hull \(H\), the
            minimally convex region containing \(P\), and is defined by at most
            three vertices on \(H\). Thus, given two regions whose convex hulls are
            the same, the same set of vertices on their hulls define the minimum
            bounding disk.
        </p>
        <p>
            Let \(H_X\) be the convex hull of \(X\). Because the convex hulls of
            \(X\) and \(H_X\) are the same, the same set of vertices defines
            their minimum bounding disks; as such, the minimum bounding circles
            of \(X\) and \(H_X\) are the same.
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
