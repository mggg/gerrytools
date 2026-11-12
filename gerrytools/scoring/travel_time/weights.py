"""Weight policies shared by travel-time scoring functions."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from enum import StrEnum
from typing import Any


class PairwiseWeightMode(StrEnum):
    """Weight applied to each pairwise trip."""

    NONE = "none"
    POPULATION_SUM = "p+p"
    POPULATION_PRODUCT = "p*p"


class CentroidWeightMode(StrEnum):
    """Weight applied to each centroid-to-unit trip."""

    NONE = "none"
    DESTINATION_POPULATION = "p"


def normalize_pairwise_weight(value: PairwiseWeightMode | str) -> PairwiseWeightMode:
    """Return a validated pairwise weight mode."""
    try:
        return PairwiseWeightMode(value)
    except ValueError as exc:
        expected = tuple(mode.value for mode in PairwiseWeightMode)
        raise ValueError(
            f"Unknown pairwise weight mode {value!r}; expected one of {expected}"
        ) from exc


def normalize_centroid_weight(value: CentroidWeightMode | str) -> CentroidWeightMode:
    """Return a validated centroid weight mode."""
    try:
        return CentroidWeightMode(value)
    except ValueError as exc:
        expected = tuple(mode.value for mode in CentroidWeightMode)
        raise ValueError(
            f"Unknown centroid weight mode {value!r}; expected one of {expected}"
        ) from exc


def validate_populations(
    units: Sequence[Any],
    populations: Mapping[Any, float],
) -> None:
    """Require finite, nonnegative numeric populations for ``units``."""
    for unit in units:
        try:
            raw_population = populations[unit]
        except KeyError as exc:
            raise ValueError(f"Population is missing for unit {unit!r}") from exc
        if isinstance(raw_population, (bool, complex)):
            raise ValueError(
                f"Population for unit {unit!r} must be a finite, nonnegative number; "
                f"got {raw_population!r}"
            )
        try:
            population = float(raw_population)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Population for unit {unit!r} must be a finite, nonnegative number; "
                f"got {raw_population!r}"
            ) from exc
        if not math.isfinite(population) or population < 0:
            raise ValueError(
                f"Population for unit {unit!r} must be a finite, nonnegative number; "
                f"got {raw_population!r}"
            )
