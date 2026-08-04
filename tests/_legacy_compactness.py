"""Unchanged historical GerryTools helpers retained only for differential tests."""

from collections.abc import Hashable
from typing import cast

import geopandas as gpd
from geopandas import GeoDataFrame
from gerrychain import Graph, Partition
from gerrychain.updaters import Tally
from shapely import Polygon


def convex_hull(dissolved_gdf: GeoDataFrame) -> dict[Hashable, int | float]:
    """Return the historical state-clipped convex-hull scores."""
    state_geom = dissolved_gdf.dissolve().iloc[0].geometry
    hull_areas = dissolved_gdf.geometry.apply(
        lambda geometry: geometry.convex_hull.intersection(state_geom).area
    )
    district_areas = dissolved_gdf.geometry.apply(lambda geometry: geometry.area)
    scores = district_areas / hull_areas
    return scores.reset_index().set_index("assignment").to_dict()["geometry"]


def population_polygon(
    dissolved_gdf: GeoDataFrame,
    block_gdf: GeoDataFrame,
    pop_col: str,
) -> dict[object, float]:
    """Return the historical full-weight population-polygon scores."""
    graph = Graph.from_geodataframe(dissolved_gdf)
    partition = Partition(
        graph=graph,
        assignment={node: node for node in graph.nodes},
        updaters={"population": Tally(pop_col, alias="population")},
    )
    district_hulls = dict(dissolved_gdf.geometry.apply(lambda geometry: geometry.convex_hull))

    scores = {}
    for part in partition.parts:
        hull = cast(Polygon, district_hulls[cast(int, part) - 1])
        clipped = gpd.clip(block_gdf, hull)
        scores[part] = partition["population"][part] / sum(clipped[pop_col])
    return scores
