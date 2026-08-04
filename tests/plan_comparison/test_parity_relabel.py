from itertools import permutations
from typing import cast

import numpy as np
import pandas as pd
import pytest

from gerrytools.plan_comparison import minimum_population_dispersion_with_parity


def _units(weights: np.ndarray) -> pd.DataFrame:
    records = []
    for source in range(weights.shape[0]):
        for target in range(weights.shape[1]):
            records.append(
                {
                    "comparison": f"source-{source}",
                    "reference": f"target-{target}",
                    "population": weights[source, target],
                }
            )
    return pd.DataFrame.from_records(records)


def _objectives(
    weights: np.ndarray,
    assignment: tuple[int, ...],
    even_targets: set[int],
) -> tuple[float, float]:
    odd_targets = [target for target in range(weights.shape[1]) if target not in even_targets]
    even_sources = [source for source, target in enumerate(assignment) if target in even_targets]
    parity_shift = weights[np.ix_(even_sources, odd_targets)].sum()
    retained = sum(weights[source, target] for source, target in enumerate(assignment))
    return float(parity_shift), -float(retained)


def test_parity_dispersion_matches_exhaustive_lexicographic_search() -> None:
    rng = np.random.default_rng(20_260_722)
    for district_count in range(1, 8):
        groups = [set(), set(range(1, district_count, 2)), set(range(district_count))]
        for even_targets in groups:
            even_labels = [f"target-{target}" for target in even_targets]
            for _ in range(6):
                weights = rng.integers(0, 20, size=(district_count, district_count)).astype(float)
                units = _units(weights)

                result = minimum_population_dispersion_with_parity(
                    units,
                    reference="reference",
                    comparison="comparison",
                    population="population",
                    even_reference_districts=even_labels,
                )
                observed_assignment = tuple(
                    int(cast(str, result.relabeling[f"source-{source}"]).removeprefix("target-"))
                    for source in range(district_count)
                )
                expected_objective = min(
                    _objectives(weights, assignment, even_targets)
                    for assignment in permutations(range(district_count))
                )

                assert _objectives(weights, observed_assignment, even_targets) == expected_objective
                assert result.population == pytest.approx(weights.sum() + expected_objective[1])


def test_parity_dispersion_uses_explicit_arbitrary_reference_labels() -> None:
    units = pd.DataFrame(
        {
            "reference": ["north", "south", "north", "south"],
            "comparison": ["x", "x", "y", "y"],
            "population": [9, 1, 8, 7],
        }
    )

    result = minimum_population_dispersion_with_parity(
        units,
        reference="reference",
        comparison="comparison",
        population="population",
        even_reference_districts=["south"],
    )

    assert result.relabeling == {"x": "north", "y": "south"}
    assert result.population == pytest.approx(9)


@pytest.mark.parametrize(
    ("even_labels", "message"),
    [
        (["missing"], "unknown"),
        (["north", "north"], "unique"),
        ([["unhashable"]], "hashable"),
    ],
)
def test_parity_dispersion_rejects_invalid_even_labels(even_labels, message) -> None:
    units = pd.DataFrame(
        {
            "reference": ["north", "south"],
            "comparison": ["x", "y"],
            "population": [1, 1],
        }
    )

    with pytest.raises(ValueError, match=message):
        minimum_population_dispersion_with_parity(
            units,
            reference="reference",
            comparison="comparison",
            population="population",
            even_reference_districts=even_labels,
        )
