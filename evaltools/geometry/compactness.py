from gerrychain import (
    Graph,
    Partition,
)
from gerrychain.updaters import (
    perimeter,
    exterior_boundaries,
    interior_boundaries,
    boundary_nodes,
    cut_edges_by_part,
    Tally, # TODO: ask why this isn't already in GeographicPartition
)

from geopandas import GeoDataFrame
from cv2 import minEnclosingCircle
from shapely.ops import unary_union
from .dissolve import dissolve
from math import pi, sqrt
import numpy as np

def _reock(partition: Partition, gdf: GeoDataFrame, crs: str):
    """
    TODO: Add documentation.
    """
    gdf = gdf.to_crs(crs)
    gdf_graph = Graph.from_geodataframe(gdf, ignore_errors=True)
    geo_partition = Partition(graph=gdf_graph,
                              assignment=partition.assignment,
                              updaters={
                                  "boundary_nodes": boundary_nodes,
                                  "area": Tally("area", alias="area"),
                              }
                             )
    geometries = dict(gdf.geometry.apply(lambda p: p.convex_hull))

    boundary = set.union(*(set(e) for e in geo_partition["cut_edges"])).union(
        geo_partition["boundary_nodes"]
    )
    part_scores = {}
    for part, nodes in geo_partition.parts.items():
        geom = unary_union([
            geometries[node] for node in nodes if node in boundary
        ]).convex_hull
        coords = np.array(geom.exterior.coords.xy).T.astype(np.float32)
        _, radius = minEnclosingCircle(coords)
        score = float(geo_partition['area'][part] / (pi * radius**2))
        assert 0 < score < 1
        part_scores[part] = score
    return part_scores

def _polsby_popper(partition: Partition, dissolved_gdf: GeoDataFrame, crs: str):
    """
    TODO : Add documentation
    """
    dissolved_gdf = dissolved_gdf.to_crs(crs)
    gdf_graph = Graph.from_geodataframe(dissolved_gdf, ignore_errors=True)
    geo_partition = Partition(graph=gdf_graph,
                              assignment={n:n for n in gdf_graph.nodes},
                              updaters={
                                "area": Tally("area", alias="area"),
                                "perimeter": perimeter,
                                "exterior_boundaries": exterior_boundaries,
                                "interior_boundaries": interior_boundaries,
                                "boundary_nodes": boundary_nodes,
                                "cut_edges_by_part": cut_edges_by_part,
                              })
    part_scores = {}
    for part, _ in geo_partition.parts.items():
        part_scores[part] = (4*pi*geo_partition['area'][part])/(
        geo_partition['perimeter'][part] ** 2)
    return part_scores

def _schwartzberg(partition: Partition, dissolved_gdf: GeoDataFrame, crs: str):
    """
    TODO: Add documentation
    """
    polsby_scores = _polsby_popper(partition, dissolved_gdf, crs)
    part_scores = {k : 1/sqrt(polsby_scores[k]) for k in polsby_scores.keys()}
    return part_scores

def _convex_hull(partition: Partition, dissolved_gdf: GeoDataFrame, crs: str, index: str = "GEOID20"):
    """
    TODO: Add documentation.
    """
    dissolved_gdf = dissolved_gdf.to_crs(crs)
    state_geom = dissolved_gdf.dissolve().iloc[0].geometry

    # Boundary-clipped convex hulls
    dissolved_convex_hull_areas = dissolved_gdf.geometry.apply(
        lambda g: g.convex_hull.intersection(state_geom).area
    )
    dissolved_areas = dissolved_gdf.geometry.apply(lambda g: g.area)
    convex_hull_scores = dissolved_areas / dissolved_convex_hull_areas
    return convex_hull_scores

def _cut_edges(partition: Partition):
    return len(partition["cut_edges"])