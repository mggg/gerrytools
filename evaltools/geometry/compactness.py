from gerrychain import (
    Graph,
    Partition,
    GeographicPartition
)
from gerrychain.updaters import (
    boundary_nodes,
    Tally,
    perimeter,
    exterior_boundaries,
    interior_boundaries,
    cut_edges_by_part# TODO: ask why this isn't already in GeographicPartition
)

from geopandas import GeoDataFrame
from cv2 import minEnclosingCircle
from shapely.ops import unary_union
from .dissolve import dissolve
from math import pi
import numpy as np
import geopandas as gpd
import pandas as pd

def _reock(partition: Partition, gdf: GeoDataFrame, crs: str, assignment_col: str):
    """
    Args:
        partition (Partition): Partition for plan on any units.
        gdf (GeoDataFrame): GeoDataFrame for plan as dissolved districts.
        crs (str): The desired projection for calculating reock.
        assignment_col (str): The name of the assignment column in the dissolved
            geodataframe.
    Returns:
        Dictionary of reock scores per district in the plan.
    """
    gdf = gdf.to_crs(crs)
    assignment = gdf.to_dict()[assignment_col]

    gdf_graph = Graph.from_geodataframe(gdf)
    geo_partition = Partition(graph=gdf_graph,
                              assignment=assignment,
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

def _polsby_popper(partition: Partition, gdf: GeoDataFrame, crs: str, assignment_col:str):
    """
    Polsby popper is a contour-based compactness score that compares a district's
    area to its perimeter. The best polsby popper score a district can have is 1.
    Args:
        partition (Partition): Partition for plan on any units.
        gdf (GeoDataFrame): GeoDataFrame for plan as dissolved districts.
        crs (str): The desired projection for calculating polsby popper.
        assignment_col (str): The name of the assignment column in the dissolved
            geodataframe.
    Returns:
        Dictionary of polsby popper scores per district in the plan.
    """

    gdf = gdf.to_crs(crs)

    assignment = gdf.to_dict()[assignment_col]

    gdf_graph = Graph.from_geodataframe(gdf)
    geo_partition = Partition(graph=gdf_graph,
                              assignment=assignment,
                              updaters={
                                  "boundary_nodes": boundary_nodes,
                                  "area": Tally("area", alias="area"),
                                  "perimeter": perimeter,
                                  "exterior_boundaries": exterior_boundaries,
                                  "interior_boundaries": interior_boundaries,
                                  "cut_edges_by_part": cut_edges_by_part})
    part_scores = {}
    for part, nodes in geo_partition.parts.items():
        part_scores[part] = (4*pi*geo_partition['area'][part])/(
        geo_partition['perimeter'][part] ** 2)
    return part_scores

def _schwartzberg(partition: Partition, gdf: GeoDataFrame, crs: str, assignment_col: str):
    """
    Schwartzberg is a contour-based compactness score that compares a district's
    perimeter to its area. This score is inversely related to polsby popper, so the
    score ranges from 1-inf. The best Schwartzberg score a district can have is 1.
    Args:
        partition (Partition): Partition for the plan on any units.
        gdf (GeoDataFrame): GeoDataFrame for the plan as dissolved districts.
        crs (str): The desired projection for calculating schwartzberg.
        assignment_col (str): The name of the assignment column in the dissolved
            geodataframe.
    Returns:
        Dictionary of schwartzberg scores per district in the plan.
    """
    polsby_scores = _polsby_popper(partition, gdf, crs, assignment_col)
    part_scores = {k : 1/sqrt(polsby_scores[k]) for k in polsby_scores.keys()}
    return part_scores

def _convex_hull(partition: Partition, gdf: GeoDataFrame, crs: str, assignment_col: str):
    """
    Convex hull is defined as the ratio of a district's area to the district's convex hull.
    When the score is closer to 1, the district is more compact. Shapely's built in convex
    hull function is used to find the convex hull of each district.

    Args:
        partition (Partition): Partition for the plan on any units.
        gdf (GeoDataFrame): GeoDataFrame for the plan as dissolved districts.
        crs (str): The desired projection for calculating convex hull.
        assignment_col (str): The name of the assignment column in the dissolved
            geodataframe.
    Returns:
        Dictionary of convex hull scores per district in the plan.
    """
    gdf = gdf.to_crs(crs)
    assignment = {ix: row[assignment_col] for ix, row in gdf.iterrows()}

    state_geom = gdf.dissolve().iloc[0].geometry

    # Boundary-clipped convex hulls
    dissolved_convex_hull_areas = [hull for hull in gdf.geometry.apply(
        lambda g: g.convex_hull.intersection(state_geom).area
    )]

    dissolved_areas = [area for area in gdf.geometry.apply(lambda g: g.area)]

    convex_hull_scores = {assignment[ix]: dissolved_areas[ix] /
                            dissolved_convex_hull_areas[ix] for ix in range(len(dissolved_areas))}

    return convex_hull_scores

def _pop_polygon(partition: Partition, block_gdf: GeoDataFrame, gdf: GeoDataFrame, pop_col: str, crs: str, assignment_col: str):
    """
    Population polygon is defined as the ratio of the population in a district to the population in the convex hull of the district.
    When the convex hull of a district falls into the territory of another state, only the population within the state is considered.
    The best population polygon score a district can have is 1.

    Args:
        partition (Partition): Partition for plan on any units.
        block_gdf (GeoDataFrame): A block level geodataframe for the state.
            This is needed for accuracy when finding the population within a
            district's convex hull.
        gdf (GeoDataFrame): GeoDataFrame for the plan as dissolved districts.
            This should also include the aggregate population in each district.
        pop_col (str): The name of the population column in the dissolved and
            block geodataframe.
        crs (str): The desired projection for calculating convex hull.
        assignment_col (str): The name of the assignment column in the dissolved
            geodataframe.
    Returns:
        Dictionary of population polygon scores per district in the plan.
    """

    block_gdf = block_gdf.to_crs(crs)

    assignment = gdf.to_dict()[assignment_col]
    gdf_graph = Graph.from_geodataframe(gdf)


    geo_partition = Partition(graph = gdf_graph,
                              assignment = assignment,
                              updaters = {"population":Tally(pop_col, alias="population")})

    district_hulls = dict(gdf.geometry.apply(lambda p : p.convex_hull))
    block_areas = dict(block_gdf.set_index("GEOID20").geometry.apply(lambda g: g.area))
    pop_polygon_scores = {}

    for part, nodes in geo_partition.parts.items():
        hull_gdf = gpd.clip(block_gdf, district_hulls[part-1])
        district_pop = geo_partition["population"][part]

        contained = []
        for row in hull_gdf.itertuples():
             if abs(block_areas[row.GEOID20]- row.geometry.area) < 1E-6:
                 contained.append(getattr(row, "Index"))
        hull_gdf = hull_gdf.loc[contained]

        pop_polygon_scores[part] = geo_partition["population"][part]/sum(hull_gdf[pop_col])

    return pop_polygon_scores
