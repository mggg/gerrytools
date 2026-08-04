"""Definition-level geometry oracles for scoring-engine compactness tests."""

from collections import defaultdict
from collections.abc import Hashable, Sequence
from math import fsum, isfinite, pi
from numbers import Integral, Real

from shapely import minimum_bounding_radius, union_all
from shapely.geometry.base import BaseGeometry

Number = int | float


def district_geometries(
    geometries: Sequence[BaseGeometry], assignment: Sequence[Hashable]
) -> dict[Hashable, BaseGeometry]:
    """Dissolve ordered unit geometries by district after validating the test model."""
    if not geometries:
        raise ValueError("geometries cannot be empty")
    if len(geometries) != len(assignment):
        raise ValueError("geometries and assignment must have equal lengths")

    grouped: defaultdict[Hashable, list[BaseGeometry]] = defaultdict(list)
    for geometry, district in zip(geometries, assignment, strict=True):
        if geometry is None or geometry.is_empty or not geometry.is_valid:
            raise ValueError("unit geometries must be nonempty and valid")
        if geometry.geom_type not in {"Polygon", "MultiPolygon"} or geometry.area <= 0:
            raise ValueError("unit geometries must be positive-area polygons")
        try:
            grouped[district].append(geometry)
        except TypeError as error:
            raise ValueError("district labels must be hashable") from error
    return {district: union_all(parts) for district, parts in grouped.items()}


def convex_hull_scores(
    geometries: Sequence[BaseGeometry], assignment: Sequence[Hashable]
) -> dict[Hashable, float]:
    """Return district area divided by ordinary convex-hull area."""
    districts = district_geometries(geometries, assignment)
    return {
        district: geometry.area / geometry.convex_hull.area
        for district, geometry in districts.items()
    }


def reock_scores(
    geometries: Sequence[BaseGeometry], assignment: Sequence[Hashable]
) -> dict[Hashable, float]:
    """Return district area divided by its minimum enclosing-circle area."""
    districts = district_geometries(geometries, assignment)
    return {
        district: geometry.area / (pi * minimum_bounding_radius(geometry) ** 2)
        for district, geometry in districts.items()
    }


def state_clipped_convex_hull_scores(
    geometries: Sequence[BaseGeometry],
    assignment: Sequence[Hashable],
    state: BaseGeometry,
) -> dict[Hashable, float]:
    """Return district area divided by state-clipped convex-hull area."""
    if state is None or state.is_empty or not state.is_valid:
        raise ValueError("state geometry must be nonempty and valid")
    if state.geom_type not in {"Polygon", "MultiPolygon"} or state.area <= 0:
        raise ValueError("state geometry must be a positive-area polygon")

    districts = district_geometries(geometries, assignment)
    if not state.covers(union_all(list(districts.values()))):
        raise ValueError("state geometry must cover every unit geometry")

    return {
        district: geometry.area / geometry.convex_hull.intersection(state).area
        for district, geometry in districts.items()
    }


def population_polygon_scores(
    geometries: Sequence[BaseGeometry],
    assignment: Sequence[Hashable],
    population_geometries: Sequence[BaseGeometry],
    weights: Sequence[Number],
    owners: Sequence[int],
) -> dict[Hashable, float]:
    """Evaluate the full-weight polygon-intersection definition directly.

    ``owners`` contains zero-based positions in ``geometries``. Any nonempty hull intersection
    contributes the population polygon's complete weight, including a boundary-only touch.
    Duplicate observations count separately.
    """
    districts = district_geometries(geometries, assignment)
    if not (len(population_geometries) == len(weights) == len(owners)):
        raise ValueError("population geometries, weights, and owners must have equal lengths")
    if not population_geometries:
        raise ValueError("population observations cannot be empty")

    observations: list[tuple[BaseGeometry, float, int]] = []
    for index, (population_geometry, weight, owner) in enumerate(
        zip(population_geometries, weights, owners, strict=True)
    ):
        if (
            population_geometry is None
            or population_geometry.is_empty
            or not population_geometry.is_valid
            or population_geometry.geom_type not in {"Polygon", "MultiPolygon"}
            or not isfinite(population_geometry.area)
            or population_geometry.area <= 0
        ):
            raise ValueError(f"population geometry {index} must be a positive-area polygon")
        if isinstance(weight, bool) or not isinstance(weight, Real) or not isfinite(float(weight)):
            raise ValueError(f"population weight {index} must be finite numeric")
        if weight < 0:
            raise ValueError(f"population weight {index} cannot be negative")
        if isinstance(owner, bool) or not isinstance(owner, Integral):
            raise ValueError(f"population owner {index} must be an integer node position")
        owner_index = int(owner)
        if owner_index < 0 or owner_index >= len(geometries):
            raise ValueError(f"population owner {index} is outside the graph-node range")

        if not geometries[owner_index].covers(population_geometry):
            raise ValueError(f"population geometry {index} is not covered by its owner geometry")
        observations.append((population_geometry, float(weight), owner_index))

    if not any(weight > 0 for _, weight, _ in observations):
        raise ValueError("population observations must have positive total weight")

    scores = {}
    for district, geometry in districts.items():
        numerator = fsum(
            weight for _, weight, owner in observations if assignment[owner] == district
        )
        if numerator <= 0:
            raise ValueError(f"district {district!r} has no positive population")
        hull = geometry.convex_hull
        denominator = fsum(
            weight
            for population_geometry, weight, _ in observations
            if hull.intersects(population_geometry)
        )
        if denominator < numerator:
            raise AssertionError("owner coverage invariant did not preserve district population")
        scores[district] = numerator / denominator
    return scores
