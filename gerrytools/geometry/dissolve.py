from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Tuple

import geopandas as gpd
import us
from geopandas import GeoDataFrame
from pandas import DataFrame
from shapely.ops import unary_union

S3_CENSUS_2020_BASE = (
    "http://data.mggg.org.s3-website.us-east-2.amazonaws.com/census-2020"
)


def dissolve(
    geometries, by="DISTRICTN", reset_index=True, keep=[], aggfunc="sum"
) -> GeoDataFrame:
    """
    Dissolves `geometries` on the column `by`. Intended to dissolve a set of
    source geometries (e.g. VTDs, blocks, block groups, etc.) to district
    geometries.

    Args:
        geometries (GeoDataFrame): Set of geometries to be dissolved.
        by (str): Name of the column used to group objects.
        reset_index (boolean, optional): If true, the index of the resulting
            GeoDataFrame will be set to an integer index, not `by`. Defaults to
            `True`.
        keep (list, optional): Additional columns to keep beyond the geometry
            and `by` columns. Defaults to an empty list, so no additional columns
            are kept.
        aggfunc (str, optional): Pandas groupby function type when aggregating;
            defaults to `"sum"`.

    Returns:
        A `GeoDataFrame` containing dissolved geometries and kept columns
        computed by the function designated by `aggfunc`.
    """
    # Pare down the geometries and dissolve.
    geometries = geometries[keep + [by, "geometry"]]
    geometries = geometries.dissolve(by=by, aggfunc=aggfunc)
    if reset_index:
        geometries = geometries.reset_index()

    return geometries


@dataclass
class StateHierarchy:
    """A collection of `GeoDataFrame`s representing a state."""

    state: us.states.State
    blocks: Optional[gpd.GeoDataFrame]
    block_groups: Optional[gpd.GeoDataFrame]
    tracts: Optional[gpd.GeoDataFrame]
    counties: Optional[gpd.GeoDataFrame]

    @staticmethod
    def from_s3(
        cls, state: us.states.State, base: str = S3_CENSUS_2020_BASE
    ) -> "StateHierarchy":
        """Loads 2020 U.S. Census data from S3 (data.mggg)."""
        state_abbr = str(state.abbr).lower()
        blocks = gpd.read_file(f"{base}/{state_abbr}/{state_abbr}_block.zip").set_index(
            "GEOID20"
        )
        block_groups = (
            gpd.read_file(f"{base}/{state_abbr}/{state_abbr}_bg.zip")
            .to_crs(blocks.crs)
            .set_index("GEOID20")
        )
        tracts = (
            gpd.read_file(f"{base}/{state_abbr}/{state_abbr}_tract.zip")
            .to_crs(blocks.crs)
            .set_index("GEOID20")
        )
        counties = (
            gpd.read_file(f"{base}/{state_abbr}/{state_abbr}_county.zip")
            .to_crs(blocks.crs)
            .set_index("GEOID20")
        )

        return cls(
            state=state,
            blocks=blocks,
            block_groups=block_groups,
            tracts=tracts,
            counties=counties,
        )

    # TODO: load from GerryDB.


def hierarchical_block_dissolve(
    state: StateHierarchy,
    block_assignment: DataFrame,
    district_col: str,
    area_consistency_tol: float = 1e-4,
) -> Tuple[GeoDataFrame, Counter]:
    """Hierarchically dissolves Census blocks into polygons by assignment.
    Dissolving blocks into districts for an entire state is a notoriously
    expensive operation. However, real-life plans tend to preserve block group
    and county boundaries in many places. This suggests a dissolving algorithm
    that leverages the Census hierarchy (blocks -> block groups -> tracts ->
    counties) and merges the largest whole polygons possible---for instance,
    if a whole county is contained within a district, we use the precomputed
    county polygon rather than dissolving its constituent blocks.
    `GeoDataFrame`s of Census data must have the same CRS and Census vintage.
    They must be indexed by their vintage-specific `GEOID` (e.g. `GEOID10`
    or `GEOID20`).
    Args:
        state (State): `StateHierarchy` with Census units for a state.
        block_assignment: A DataFrame with block ids and a district column.
        district_col: Name of column with plan districts.
        area_consistency_tol: Relative tolerance for area consistency check.
    Raises:
        ValueError: If the CRSes of the `GeoDataFrame`s do not match
            or `block_gdf` is unspecified.
        DisolveError: If areas are not approximately consistent
            between block unions and hierarchical unions.
    Returns:
        A tuple containing a `GeoDataFrame` of dissolved district polygons
        (indexed by label) and counts of polygons used from each level
        in the hierarchy.
    """
    block_assignment = block_assignment.set_index("GEOID20").to_dict()[district_col]

    level_gdfs = {
        level: gdf
        for gdf, level in zip(
            (state.blocks, state.block_groups, state.tracts, state.counties),
            ("block", "bg", "tract", "county"),
        )
        if gdf is not None
    }

    # Build nesting indices of Census geometries.
    # Block GeoIDs are 15-16 characters long. The first 15 characters are
    # numerals representing the state (2) + county (3) + tract (6) +
    # block group (1) + block (3). There is an optional one-character block
    # suffix. (See https://www.census.gov/programs-surveys/geography/
    # guidance/geo-identifiers.html)

    level_prefixes = {"county": 5, "tract": 11, "bg": 12, "block": 16}
    level_geoids_to_blocks = _group_by_level(
        {str(b): b for b in state.blocks.index}, level_prefixes
    )
    level_geoids_to_assignments = _group_by_level(block_assignment, level_prefixes)

    # Find the minimal set of geometries representing each assignment.
    levels = tuple(
        level
        for level, _ in sorted(level_prefixes.items(), key=lambda kv: kv[1])
        if level in level_gdfs
    )  # sorted largest geometry (smallest prefix) -> smallest geometry
    level_geoms = {level: dict(gdf.geometry) for level, gdf in level_gdfs.items()}
    geoms_by_assignment = defaultdict(list)
    blocks_by_assignment = defaultdict(set)
    for block, assignment in block_assignment.items():
        blocks_by_assignment[assignment].add(block)
    remaining_blocks_by_assignment = deepcopy(blocks_by_assignment)
    level_counts = Counter()

    for level in levels:
        level_assignments = level_geoids_to_assignments[level]
        for level_geoid, assignments in level_assignments.items():
            if len(assignments) == 1:
                # This unit *may* be wholly contained in a single district.
                (assignment,) = tuple(assignments)
                remaining_blocks = remaining_blocks_by_assignment[assignment]
                level_blocks = level_geoids_to_blocks[level][level_geoid]
                if level_blocks.issubset(remaining_blocks):
                    # If we match, save the geometry and remove the
                    # corresponding blocks from the assignment's
                    # remaining blocks.
                    level_geom = level_geoms[level][level_geoid]
                    geoms_by_assignment[assignment].append(level_geom)
                    remaining_blocks_by_assignment[assignment] -= level_blocks
                    level_counts[level] += 1

    # Dissolve geometries by assignment.
    dissolved_gdf = (
        GeoDataFrame(
            [
                {"label": assignment, "geometry": unary_union(geometries)}
                for assignment, geometries in geoms_by_assignment.items()
            ]
        )
        .sort_values(by="label")
        .set_index("label")
    )
    dissolved_gdf.crs = state.blocks.crs

    # Basic consistency check: district areas should approximately match.
    block_geoms = level_geoms["block"]
    areas_from_blocks = {
        assignment: sum(block_geoms[b].area for b in block_geoids)
        for assignment, block_geoids in blocks_by_assignment.items()
    }
    dissolved_areas = dict(dissolved_gdf.geometry.apply(lambda g: g.area))
    for assignment, ref_area in areas_from_blocks.items():
        dissolved_area = dissolved_areas[assignment]
        relative_diff = abs(dissolved_area - ref_area) / ref_area
        if relative_diff >= area_consistency_tol:
            raise DissolveError(
                f"Area consistency check failed for district {assignment} "
                "after hierarchical dissolve. "
                "(area from blocks is {:.8f}, dissolved area is {:.8f}".format(
                    ref_area, dissolved_area
                )
            )

    return dissolved_gdf, level_counts


def _group_by_level(block_geoids: Mapping[str, Any], level_prefixes: Mapping[str, int]):
    """Groups values associated with block GeoIDs by level."""
    level_geoids_to_val = {level: defaultdict(set) for level in level_prefixes}
    for block_geoid, val in block_geoids.items():
        for level, prefix in level_prefixes.items():
            level_geoid = block_geoid[:prefix]
            level_geoids_to_val[level][level_geoid].add(val)
    return level_geoids_to_val


class DissolveError(Exception):
    """Raised when a custom dissolve operation fails."""
