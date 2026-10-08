"""Family-specific trip weights for travel-time scores."""

from __future__ import annotations

import math
from typing import Any, Literal, Mapping, Sequence

PairwiseWeightMode = Literal["none", "p+p", "p*p"]
CentroidWeightMode = Literal["none", "p"]

PAIRWISE_WEIGHT_MODES: tuple[PairwiseWeightMode, ...] = ("none", "p+p", "p*p")
CENTROID_WEIGHT_MODES: tuple[CentroidWeightMode, ...] = ("none", "p")


def validate_populations(
    units: Sequence[Any],
    populations: Mapping[Any, float],
) -> None:
    """Require finite, nonnegative numeric populations for ``units``.

    Population validation is intentionally performed before weighted scoring.
    Otherwise, a NaN can contaminate every candidate centroid cost and prevent
    any candidate from being selected without producing a useful error.
    """
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


def pairwise_trip_weight(
    pop_a: float,
    pop_b: float,
    mode: PairwiseWeightMode | str,
) -> float:
    """Return the PTT weight for a trip between two population endpoints."""
    if mode == "none":
        return 1.0
    if mode == "p+p":
        return float(pop_a) + float(pop_b)
    if mode == "p*p":
        return float(pop_a) * float(pop_b)
    raise ValueError(
        f"Unknown pairwise weight mode {mode!r}; expected one of {PAIRWISE_WEIGHT_MODES}"
    )


def centroid_trip_weight(
    destination_population: float,
    mode: CentroidWeightMode | str,
) -> float:
    """Return the CTT weight for a center-to-unit trip.

    The selected centroid unit is a location, not a population endpoint. Its
    population therefore never enters CTT selection or scoring.
    """
    if mode == "none":
        return 1.0
    if mode == "p":
        return float(destination_population)
    raise ValueError(
        f"Unknown centroid weight mode {mode!r}; expected one of {CENTROID_WEIGHT_MODES}"
    )
