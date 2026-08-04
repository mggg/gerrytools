from collections.abc import Hashable, Sequence
from typing import cast
from unittest.mock import patch

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from shapely import affinity, union_all
from shapely.geometry import LineString, MultiPolygon, Point, Polygon, box
from shapely.geometry.base import BaseGeometry

from gerrytools.scoring import PlanEvaluator, StateClippedConvexHullRatio
from tests import _legacy_compactness as legacy_compactness
from tests._compactness_oracles import (
    convex_hull_scores,
    district_geometries,
    population_polygon_scores,
    state_clipped_convex_hull_scores,
)


def _legacy_convex_hull_scores(
    geometries: Sequence[BaseGeometry], assignment: Sequence[Hashable]
) -> dict[Hashable, int | float]:
    districts = district_geometries(geometries, assignment)
    frame = gpd.GeoDataFrame(
        {"geometry": list(districts.values())},
        index=pd.Index(list(districts), name="assignment"),
        crs="EPSG:3857",
    )
    return legacy_compactness.convex_hull(frame)


def _normalized_legacy_population_polygon_scores(
    dissolved: gpd.GeoDataFrame,
    population: gpd.GeoDataFrame,
    pop_col: str,
) -> dict[int, float]:
    """Run the unchanged legacy helper and normalize its labels in this harness."""
    partition_type = legacy_compactness.Partition

    def one_based_partition(*args, assignment, **kwargs):
        adjusted = {node: int(part) + 1 for node, part in assignment.items()}
        return partition_type(*args, assignment=adjusted, **kwargs)

    with patch.object(legacy_compactness, "Partition", one_based_partition):
        scores = legacy_compactness.population_polygon(dissolved, population, pop_col)
    return {cast(int, part) - 1: score for part, score in scores.items()}


def test_convex_hull_definitions_have_hand_computable_scores() -> None:
    units = [box(0, 0, 1, 1), box(1, 0, 2, 1), box(0, 1, 1, 2)]

    assert convex_hull_scores([units[0]], ["square"]) == {"square": pytest.approx(1)}
    assert convex_hull_scores(units, ["L", "L", "L"]) == {"L": pytest.approx(6 / 7)}

    standard = convex_hull_scores(units, ["inside", "diagonal", "diagonal"])
    clipped = state_clipped_convex_hull_scores(
        units,
        ["inside", "diagonal", "diagonal"],
        union_all(units),
    )
    assert standard == {"inside": pytest.approx(1), "diagonal": pytest.approx(2 / 3)}
    assert clipped == {"inside": pytest.approx(1), "diagonal": pytest.approx(4 / 5)}
    assert _legacy_convex_hull_scores(units, ["inside", "diagonal", "diagonal"]) == pytest.approx(
        clipped
    )


def test_clipped_hull_excludes_state_holes_and_gaps_between_islands() -> None:
    state_with_hole = Polygon(
        [(0, 0), (4, 0), (4, 4), (0, 4)],
        holes=[[(1, 1), (2, 1), (2, 2), (1, 2)]],
    )
    islands = MultiPolygon([box(0, 0, 1, 1), box(3, 0, 4, 1)])

    assert convex_hull_scores([state_with_hole], [0]) == {0: pytest.approx(15 / 16)}
    assert state_clipped_convex_hull_scores([state_with_hole], [0], state_with_hole) == {
        0: pytest.approx(1)
    }
    assert convex_hull_scores([islands], [0]) == {0: pytest.approx(1 / 2)}
    assert state_clipped_convex_hull_scores([islands], [0], islands) == {0: pytest.approx(1)}


def test_convex_hull_scores_are_invariant_under_decomposition_and_similarity() -> None:
    decomposed = [box(0, 0, 1, 1), box(1, 0, 2, 1), box(0, 1, 1, 2)]
    combined = [union_all(decomposed)]
    state = box(-1, -1, 3, 3)
    expected_standard = convex_hull_scores(decomposed, [7, 7, 7])
    expected_clipped = state_clipped_convex_hull_scores(decomposed, [7, 7, 7], state)

    assert convex_hull_scores(combined, [7]) == expected_standard
    assert state_clipped_convex_hull_scores(combined, [7], state) == expected_clipped

    def transform(geometry):
        return affinity.translate(
            affinity.rotate(
                affinity.scale(geometry, xfact=3.5, yfact=3.5, origin=(0, 0)),
                37,
                origin=(0, 0),
            ),
            xoff=100,
            yoff=-40,
        )

    transformed_units = [transform(geometry) for geometry in decomposed]
    assert convex_hull_scores(transformed_units, [7, 7, 7]) == pytest.approx(expected_standard)
    assert state_clipped_convex_hull_scores(
        transformed_units, [7, 7, 7], transform(state)
    ) == pytest.approx(expected_clipped)


def test_convex_hull_scores_are_invariant_under_node_and_district_relabeling() -> None:
    units = [box(0, 0, 1, 1), box(1, 0, 2, 1), box(0, 1, 1, 2), box(2, 0, 3, 1)]
    assignment = ["left", "left", "left", "right"]
    expected = convex_hull_scores(units, assignment)
    permutation = [2, 0, 3, 1]

    reordered = convex_hull_scores(
        [units[index] for index in permutation],
        [assignment[index] for index in permutation],
    )
    relabeled = convex_hull_scores(
        units,
        [17 if district == "left" else ("right", 2) for district in assignment],
    )

    assert reordered == expected
    assert relabeled == {17: expected["left"], ("right", 2): expected["right"]}


@settings(max_examples=300, deadline=None)
@given(st.lists(st.integers(0, 3), min_size=8, max_size=8))
def test_generated_native_clipped_scores_match_legacy_on_identical_inputs(
    assignment: list[int],
) -> None:
    units = [
        box(column, row, column + 1, row + 1)
        for row in range(3)
        for column in range(3)
        if (row, column) != (1, 1)
    ]
    state = union_all(units)
    frame = gpd.GeoDataFrame(geometry=units, crs="EPSG:3857")
    scorer = PlanEvaluator(nx.empty_graph(len(units)), geometry=frame).add_metric(
        StateClippedConvexHullRatio(state)
    )
    legacy = _legacy_convex_hull_scores(units, assignment)
    engine = scorer.evaluate(assignment)["state_clipped_convex_hull_ratio"]

    assert isinstance(engine, pd.Series)
    np.testing.assert_allclose(
        engine,
        [legacy[district] for district in engine.index],
        rtol=1e-9,
        atol=1e-12,
    )


@pytest.mark.parametrize(
    ("geometries", "assignment", "state", "message"),
    [
        ([], [], box(0, 0, 1, 1), "cannot be empty"),
        ([box(0, 0, 1, 1)], [], box(0, 0, 1, 1), "equal lengths"),
        ([Polygon()], [0], box(0, 0, 1, 1), "nonempty and valid"),
        ([LineString([(0, 0), (1, 1)])], [0], box(0, 0, 1, 1), "positive-area"),
        ([box(0, 0, 1, 1)], [0], Polygon(), "nonempty and valid"),
        ([box(0, 0, 1, 1)], [0], Point(0, 0), "positive-area"),
        ([box(0, 0, 2, 1)], [0], box(0, 0, 1, 1), "cover every unit"),
    ],
)
def test_convex_hull_oracle_rejects_inputs_outside_the_contract(
    geometries, assignment, state, message
) -> None:
    with pytest.raises(ValueError, match=message):
        state_clipped_convex_hull_scores(geometries, assignment, state)


def test_standard_convex_hull_oracle_shares_unit_validation() -> None:
    # Deliberately unhashable labels, smuggled past the signature to hit the runtime check.
    unhashable_labels = cast("Sequence[Hashable]", [["not", "hashable"]])
    with pytest.raises(ValueError, match="district labels must be hashable"):
        convex_hull_scores([box(0, 0, 1, 1)], unhashable_labels)


def test_population_polygon_has_hand_computable_disconnected_district_score() -> None:
    units = [box(0, 0, 1, 1), box(1, 0, 2, 1), box(2, 0, 3, 1)]
    population_geometries = [
        box(0.1, 0.1, 0.9, 0.9),
        box(1.1, 0.1, 1.9, 0.9),
        box(2.1, 0.1, 2.9, 0.9),
    ]
    scores = population_polygon_scores(
        units,
        ["A", "B", "A"],
        population_geometries,
        [10, 20, 30],
        [0, 1, 2],
    )

    assert scores == {"A": pytest.approx(2 / 3), "B": pytest.approx(1)}


def test_population_polygon_handles_holed_and_nested_island_units() -> None:
    surrounding_unit = Polygon(
        [(0, 0), (4, 0), (4, 4), (0, 4)],
        holes=[[(1, 1), (2, 1), (2, 2), (1, 2)]],
    )
    island_unit = box(1.2, 1.2, 1.8, 1.8)

    scores = population_polygon_scores(
        [surrounding_unit, island_unit],
        ["surrounding", "island"],
        [box(0.2, 0.2, 0.8, 0.8), box(1.3, 1.3, 1.7, 1.7)],
        [10, 5],
        [0, 1],
    )

    assert scores == {"surrounding": pytest.approx(2 / 3), "island": pytest.approx(1)}


def test_population_polygon_counts_boundary_touching_polygons_in_full() -> None:
    units = [box(0, 0, 1, 1), box(1, 0, 2, 1)]
    scores = population_polygon_scores(
        units,
        [0, 1],
        [box(0.5, 0.25, 1, 0.75), box(1, 0.25, 1.5, 0.75)],
        [5, 7],
        [0, 1],
    )

    assert scores == {0: pytest.approx(5 / 12), 1: pytest.approx(7 / 12)}


def test_population_polygon_counts_a_barely_intersecting_polygon_in_full() -> None:
    units = [
        box(0, 0, 0.2, 0.2),
        box(2, 0, 2.2, 0.2),
        box(0, 2, 0.2, 2.2),
        box(1, 1, 2, 2),
    ]
    assignment = ["A", "A", "A", "B"]
    population_geometries = [
        box(0.05, 0.05, 0.15, 0.15),
        box(2.05, 0.05, 2.15, 0.15),
        box(0.05, 2.05, 0.15, 2.15),
        units[3],
    ]
    weights = [10, 10, 10, 100]
    owners = [0, 1, 2, 3]

    scores = population_polygon_scores(
        units,
        assignment,
        population_geometries,
        weights,
        owners,
    )

    assert scores == {"A": pytest.approx(30 / 130), "B": pytest.approx(1)}


def test_population_polygon_depends_on_population_polygon_resolution() -> None:
    units = [box(0, 0, 1, 1), box(1, 0, 2, 1)]
    assignment = [0, 1]
    coarse = population_polygon_scores(
        units,
        assignment,
        [box(0.1, 0.1, 0.9, 0.9), box(1, 0.1, 2, 0.9)],
        [10, 100],
        [0, 1],
    )
    refined = population_polygon_scores(
        units,
        assignment,
        [
            box(0.1, 0.1, 0.9, 0.9),
            box(1, 0.1, 1.5, 0.9),
            box(1.5, 0.1, 2, 0.9),
        ],
        [10, 50, 50],
        [0, 1, 1],
    )

    assert coarse[0] == pytest.approx(10 / 110)
    assert refined[0] == pytest.approx(10 / 60)


def test_legacy_population_polygon_harness_normalizes_district_labels() -> None:
    dissolved = gpd.GeoDataFrame(
        {
            "population": [10, 20],
            "geometry": [box(0, 0, 1, 1), box(1, 0, 2, 1)],
        },
        index=pd.Index([0, 1], name="assignment"),
        crs="EPSG:3857",
    )
    population = gpd.GeoDataFrame(dissolved.reset_index(drop=True), crs=dissolved.crs)

    assert _normalized_legacy_population_polygon_scores(
        dissolved,
        population,
        "population",
    ) == {
        0: pytest.approx(1 / 3),
        1: pytest.approx(2 / 3),
    }


def test_population_polygon_is_invariant_under_order_and_colocated_weight_splitting() -> None:
    units = [box(0, 0, 1, 1), box(1, 0, 2, 1), box(2, 0, 3, 1)]
    assignment = ["A", "B", "A"]
    population_geometries = [
        box(0.1, 0.1, 0.9, 0.9),
        box(1.1, 0.1, 1.9, 0.9),
        box(2.1, 0.1, 2.9, 0.9),
    ]
    weights = [10, 20, 30]
    owners = [0, 1, 2]
    expected = population_polygon_scores(
        units,
        assignment,
        population_geometries,
        weights,
        owners,
    )

    assert (
        population_polygon_scores(
            units,
            assignment,
            [
                population_geometries[2],
                population_geometries[0],
                population_geometries[1],
            ],
            [weights[2], weights[0], weights[1]],
            [owners[2], owners[0], owners[1]],
        )
        == expected
    )
    assert (
        population_polygon_scores(
            units,
            assignment,
            [
                population_geometries[0],
                population_geometries[0],
                population_geometries[1],
                population_geometries[2],
            ],
            [4, 6, 20, 30],
            [0, 0, 1, 2],
        )
        == expected
    )


def test_population_polygon_is_invariant_under_similarity_and_node_reordering() -> None:
    units = [box(0, 0, 1, 1), box(1, 0, 2, 1), box(2, 0, 3, 1)]
    assignment = ["A", "B", "A"]
    population_geometries = [
        box(0.1, 0.1, 0.9, 0.9),
        box(1.1, 0.1, 1.9, 0.9),
        box(2.1, 0.1, 2.9, 0.9),
    ]
    weights = [10, 20, 30]
    owners = [0, 1, 2]
    expected = population_polygon_scores(
        units,
        assignment,
        population_geometries,
        weights,
        owners,
    )

    def transform(geometry):
        return affinity.translate(
            affinity.rotate(
                affinity.scale(geometry, xfact=2.25, yfact=2.25, origin=(0, 0)),
                -29,
                origin=(0, 0),
            ),
            xoff=-70,
            yoff=110,
        )

    assert population_polygon_scores(
        [transform(unit) for unit in units],
        assignment,
        [transform(geometry) for geometry in population_geometries],
        weights,
        owners,
    ) == pytest.approx(expected)

    permutation = [2, 0, 1]
    new_position = {old: new for new, old in enumerate(permutation)}
    assert (
        population_polygon_scores(
            [units[index] for index in permutation],
            [assignment[index] for index in permutation],
            population_geometries,
            weights,
            [new_position[owner] for owner in owners],
        )
        == expected
    )


def test_population_outside_a_hull_does_not_change_its_score() -> None:
    units = [box(0, 0, 1, 1), box(3, 0, 4, 1)]
    assignment = [0, 1]
    baseline = population_polygon_scores(
        units,
        assignment,
        [box(0.1, 0.1, 0.9, 0.9), box(3.1, 0.1, 3.9, 0.9)],
        [10, 20],
        [0, 1],
    )
    with_extra_observation = population_polygon_scores(
        units,
        assignment,
        [
            box(0.1, 0.1, 0.9, 0.9),
            box(3.1, 0.1, 3.5, 0.9),
            box(3.5, 0.1, 3.9, 0.9),
        ],
        [10, 12, 8],
        [0, 1, 1],
    )

    assert baseline[0] == with_extra_observation[0] == 1


@settings(max_examples=300, deadline=None)
@given(
    st.lists(st.integers(0, 3), min_size=9, max_size=9),
    st.lists(st.integers(1, 10**6), min_size=9, max_size=9),
)
def test_generated_population_polygon_scores_obey_the_owner_invariant(
    assignment: list[int], weights: list[int]
) -> None:
    units = [box(column, row, column + 1, row + 1) for row in range(3) for column in range(3)]
    scores = population_polygon_scores(
        units,
        assignment,
        [
            box(column + 0.1, row + 0.1, column + 0.9, row + 0.9)
            for row in range(3)
            for column in range(3)
        ],
        weights,
        list(range(9)),
    )

    assert all(0 < score <= 1 for score in scores.values())


@pytest.mark.parametrize(
    ("population_geometries", "weights", "owners", "message"),
    [
        ([], [], [], "cannot be empty"),
        ([box(0, 0, 1, 1)], [], [0], "equal lengths"),
        ([Point(0.5, 0.5)], [1], [0], "positive-area polygon"),
        ([box(0, 0, 1, 1)], [np.inf], [0], "finite numeric"),
        ([box(0, 0, 1, 1)], [-1], [0], "cannot be negative"),
        ([box(0, 0, 1, 1)], [1], [True], "integer node position"),
        ([box(0, 0, 1, 1)], [1], [1], "outside the graph-node range"),
        ([box(2, 2, 3, 3)], [1], [0], "not covered by its owner"),
        ([box(0, 0, 1, 1)], [0], [0], "positive total weight"),
    ],
)
def test_population_polygon_oracle_rejects_invalid_observations(
    population_geometries, weights, owners, message
) -> None:
    with pytest.raises(ValueError, match=message):
        population_polygon_scores(
            [box(0, 0, 1, 1)],
            [0],
            population_geometries,
            weights,
            owners,
        )


def test_population_polygon_rejects_a_zero_population_district() -> None:
    with pytest.raises(ValueError, match="district 1 has no positive population"):
        population_polygon_scores(
            [box(0, 0, 1, 1), box(1, 0, 2, 1)],
            [0, 1],
            [box(0.1, 0.1, 0.9, 0.9), box(1.1, 0.1, 1.9, 0.9)],
            [1, 0],
            [0, 1],
        )
