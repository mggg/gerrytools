from math import pi, sqrt

import geopandas as gpd
import numpy as np
from cv2 import minEnclosingCircle
from geopandas import GeoDataFrame
from gerrychain import Graph, Partition
from gerrychain.updaters import (
    Tally,
    boundary_nodes,
    cut_edges_by_part,
    exterior_boundaries,
    interior_boundaries,
    perimeter,
)
from shapely.ops import unary_union


def _reock(dissolved_gdf: GeoDataFrame):
    """
    Arguments:
        dissolved_gdf (GeoDataFrame): GeoDataFrame corresponding to
            the plan's districts.

    Returns:
        Dictionary of reock scores by district.
    """
    gdf_graph = Graph.from_geodataframe(dissolved_gdf, ignore_errors=True)
    geo_partition = Partition(
        graph=gdf_graph,
        assignment={n: n for n in gdf_graph.nodes},
        updaters={
            "boundary_nodes": boundary_nodes,
            "area": Tally("area", alias="area"),
        },
    )
    geometries = dict(dissolved_gdf.geometry.apply(lambda p: p.convex_hull))

    boundary = set.union(*(set(e) for e in geo_partition["cut_edges"])).union(
        geo_partition["boundary_nodes"]
    )
    part_scores = {}
    for part, nodes in geo_partition.parts.items():
        geom = unary_union(
            [geometries[node] for node in nodes if node in boundary]
        ).convex_hull
        coords = np.array(geom.exterior.coords.xy).T.astype(np.float32)
        _, radius = minEnclosingCircle(coords)
        score = float(geo_partition["area"][part] / (pi * radius**2))
        assert 0 < score < 1
        part_scores[part] = score
    return part_scores


def _polsby_popper(dissolved_gdf: GeoDataFrame):
    """
    Arguments:
        dissolved_gdf (GeoDataFrame): GeoDataFrame corresponding to
            the plan's districts.

    Returns:
        Dictionary of polsby popper scores by district.
    """
    gdf_graph = Graph.from_geodataframe(dissolved_gdf, ignore_errors=True)
    geo_partition = Partition(
        graph=gdf_graph,
        assignment={n: n for n in gdf_graph.nodes},
        updaters={
            "area": Tally("area", alias="area"),
            "perimeter": perimeter,
            "exterior_boundaries": exterior_boundaries,
            "interior_boundaries": interior_boundaries,
            "boundary_nodes": boundary_nodes,
            "cut_edges_by_part": cut_edges_by_part,
        },
    )
    part_scores = {}
    for part, _ in geo_partition.parts.items():
        part_scores[part] = (4 * pi * geo_partition["area"][part]) / (
            geo_partition["perimeter"][part] ** 2
        )
    return part_scores


def _schwartzberg(dissolved_gdf: GeoDataFrame):
    """
    Arguments:
        dissolved_gdf (GeoDataFrame): GeoDataFrame corresponding to
            the plan's districts.
    Returns:
        Dictionary of schwartzberg scores by district.
    """
    polsby_scores = _polsby_popper(dissolved_gdf)
    part_scores = {k: 1 / sqrt(polsby_scores[k]) for k in polsby_scores.keys()}
    return part_scores


def _convex_hull(dissolved_gdf: GeoDataFrame):
    """
    Arguments:
        dissolved_gdf (GeoDataFrame): GeoDataFrame corresponding to
            the plan's districts.
    Returns:
        Dictionary of convex hull scores by district.
    """
    state_geom = dissolved_gdf.dissolve().iloc[0].geometry

    # Boundary-clipped convex hulls
    dissolved_convex_hull_areas = dissolved_gdf.geometry.apply(
        lambda g: g.convex_hull.intersection(state_geom).area
    )
    dissolved_areas = dissolved_gdf.geometry.apply(lambda g: g.area)
    convex_hull_scores = dissolved_areas / dissolved_convex_hull_areas
    convex_hull_scores = (
        convex_hull_scores.reset_index().set_index("assignment").to_dict()["geometry"]
    )
    return convex_hull_scores


def _cut_edges(partition: Partition):
    return len(partition["cut_edges"])


def _pop_polygon(dissolved_gdf: GeoDataFrame, block_gdf: GeoDataFrame, pop_col: str):
    """
    Arguments:
        dissolved_gdf (GeoDataFrame): GeoDataFrame corresponding to
            the plan's districts.
        block_gdf (GeoDataFrame): Block level GeoDataFrame corresponding
            to the area covered by the plan.
        pop_col (str): Field specifying total population data for the plan.

    Returns:
        Dictionary of population polygon scores by district.
    """
    gdf_graph = Graph.from_geodataframe(dissolved_gdf)

    geo_partition = Partition(
        graph=gdf_graph,
        assignment={n: n for n in gdf_graph.nodes},
        updaters={"population": Tally(pop_col, alias="population")},
    )

    district_hulls = dict(dissolved_gdf.geometry.apply(lambda p: p.convex_hull))

    pop_polygon_scores = {}
    for part in geo_partition.parts:
        hull_gdf = gpd.clip(block_gdf, district_hulls[part - 1])
        pop_polygon_scores[part] = geo_partition["population"][part] / sum(
            hull_gdf[pop_col]
        )

    return pop_polygon_scores
