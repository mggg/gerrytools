from itertools import permutations

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_series_equal

from gerrytools.plan_comparison import (
    MinimumDispersion,
    minimum_population_dispersion,
    optimal_relabeling,
    population_dispersion,
    population_dispersion_by_district,
)


def test_optimal_relabeling_maximizes_total_overlap() -> None:
    overlap = pd.DataFrame(
        [[1.0, 9.0], [8.0, 2.0]],
        index=pd.Index(["red", "blue"]),
        columns=pd.Index([10, 20]),
    )

    assert optimal_relabeling(overlap) == {"red": 20, "blue": 10}


def test_optimal_relabeling_matches_exhaustive_search() -> None:
    rng = np.random.default_rng(20_260_722)
    for district_count in range(1, 7):
        source = pd.Index([f"source-{index}" for index in range(district_count)])
        target = pd.Index([f"target-{index}" for index in range(district_count)])
        for _ in range(20):
            weights = rng.integers(0, 1_000, size=(district_count, district_count))
            overlap = pd.DataFrame(weights, index=source, columns=target)

            relabeling = optimal_relabeling(overlap)
            observed = sum(overlap.at[left, right] for left, right in relabeling.items())
            expected = max(
                sum(weights[row, column] for row, column in enumerate(order))
                for order in permutations(range(district_count))
            )

            assert observed == expected


@pytest.mark.parametrize(
    "overlap",
    [
        pd.DataFrame(),
        pd.DataFrame([[1.0, 2.0]], index=pd.Index(["A"]), columns=pd.Index([1, 2])),
        pd.DataFrame([[1.0, -1.0], [2.0, 3.0]]),
        pd.DataFrame([[1.0, float("nan")], [2.0, 3.0]]),
        pd.DataFrame(
            [[1.0, 2.0], [3.0, 4.0]],
            index=pd.Index(["A", "A"], dtype=object),
        ),
        pd.DataFrame(
            [[1.0, 2.0], [3.0, 4.0]],
            columns=pd.Index(["A", None], dtype=object),
        ),
        pd.DataFrame([["one", "two"], ["three", "four"]]),
    ],
)
def test_optimal_relabeling_rejects_incomplete_or_invalid_matrices(overlap) -> None:
    with pytest.raises(ValueError):
        optimal_relabeling(overlap)


def test_optimal_relabeling_requires_a_dataframe() -> None:
    with pytest.raises(TypeError, match="must be a DataFrame"):
        optimal_relabeling([[1.0]])  # type: ignore


def test_population_dispersion_uses_direct_label_comparison() -> None:
    units = pd.DataFrame(
        {
            "reference": ["A", "A", "B", "B"],
            "comparison": ["A", "B", "A", "B"],
            "population": [5, 1, 2, 6],
        }
    )

    assert population_dispersion(
        units,
        reference="reference",
        comparison="comparison",
        population="population",
    ) == pytest.approx(3)
    assert_series_equal(
        population_dispersion_by_district(
            units,
            reference="reference",
            comparison="comparison",
            population="population",
        ),
        pd.Series(
            [1.0, 2.0],
            index=pd.Index(["A", "B"], name="reference"),
            name="population_dispersion",
        ),
    )


def test_population_dispersion_with_disjoint_label_sets_counts_all_population() -> None:
    # No comparison label ever matches a reference label, so nothing is retained: total
    # dispersion is the full population and each reference district's dispersion is its
    # whole column sum.
    units = pd.DataFrame(
        {
            "reference": ["A", "A", "B"],
            "comparison": [1, 2, 2],
            "population": [5, 1, 2],
        }
    )

    assert population_dispersion(
        units,
        reference="reference",
        comparison="comparison",
        population="population",
    ) == pytest.approx(8)
    assert_series_equal(
        population_dispersion_by_district(
            units,
            reference="reference",
            comparison="comparison",
            population="population",
        ),
        pd.Series(
            [6.0, 2.0],
            index=pd.Index(["A", "B"], name="reference"),
            name="population_dispersion",
        ),
    )


def test_population_dispersion_by_district_supports_tuple_labels() -> None:
    units = pd.DataFrame(
        {
            "reference": [("A", 1), ("A", 1), ("B", 2)],
            "comparison": [("A", 1), ("B", 2), ("A", 1)],
            "population": [5, 1, 2],
        }
    )

    result = population_dispersion_by_district(
        units,
        reference="reference",
        comparison="comparison",
        population="population",
    )

    assert result.index.tolist() == [("A", 1), ("B", 2)]
    assert result.index.name == "reference"
    assert result.name == "population_dispersion"
    np.testing.assert_allclose(result.to_numpy(), [1.0, 2.0])


def test_minimum_population_dispersion_supports_arbitrary_labels() -> None:
    units = pd.DataFrame(
        {
            "reference": ["A", "A", "B", "B"],
            "comparison": ["x", "y", "x", "y"],
            "population": [5, 1, 2, 6],
        }
    )

    result = minimum_population_dispersion(
        units,
        reference="reference",
        comparison="comparison",
        population="population",
    )

    assert result == MinimumDispersion(relabeling={"x": "A", "y": "B"}, population=3.0)


def test_minimum_population_dispersion_requires_equal_district_counts() -> None:
    units = pd.DataFrame(
        {
            "reference": ["A", "B", "C"],
            "comparison": ["x", "x", "y"],
            "population": [1, 1, 1],
        }
    )

    with pytest.raises(ValueError, match="square"):
        minimum_population_dispersion(
            units,
            reference="reference",
            comparison="comparison",
            population="population",
        )
