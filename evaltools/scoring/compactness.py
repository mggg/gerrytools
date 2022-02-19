"""Auxiliary and specialized geographical scores for plan evaluation."""
import numpy as np
from .score_types import *
from cv2 import minEnclosingCircle
from gerrychain import GeographicPartition, Partition
from gerrychain.metrics import polsby_popper
from gerrychain.updaters import boundary_nodes
from shapely.ops import unary_union
from math import pi

def _reock(part: GeographicPartition) -> ScoreValue:
    geometries = {
            node: geom.convex_hull
            for node, geom in part.graph.nodes('geometry')
        }
    part.updaters.update({"boundary_nodes": boundary_nodes})

    boundary = set.union(*(set(e) for e in part["cut_edges"])).union(
        part["boundary_nodes"]
    )
    part_scores = {}

    for part, nodes in part.parts.items():
        geom = unary_union([
            geometries[node] for node in nodes if node in boundary
        ]).convex_hull
        coords = np.array(geom.exterior.coords.xy).T.astype(np.float32)
        _, radius = minEnclosingCircle(coords)
        score = float(part['area'][part] / (pi * radius**2))
        assert 0 < score < 1
        part_scores[part] = score
    return part_scores

def _polsby_popper(part: GeographicPartition) -> ScoreValue:
    return polsby_popper(part)

def _cut_edges(part: Partition) -> ScoreValue:
    return len(part.cut_edges)