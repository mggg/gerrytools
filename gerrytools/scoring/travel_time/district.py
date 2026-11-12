"""Vectorized travel-time calculations for one district."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from .centroid import choose_centroid
from .od_table import TravelTimeTable, UnitId
from .weights import (
    CentroidWeightMode,
    PairwiseWeightMode,
    validate_populations,
)


@dataclass(frozen=True, slots=True)
class PairwiseDistrictScores:
    """Pairwise mean and maximum for one district."""

    mean: float
    maximum: float


@dataclass(frozen=True, slots=True)
class CentroidDistrictScores:
    """Centroid mean, maximum, and selected unit for one district."""

    mean: float
    maximum: float
    centroid: UnitId | None


@dataclass(frozen=True, slots=True)
class DistrictTravelScores:
    """Combined pairwise and centroid details for one district."""

    ptt_mean: float
    ptt_max: float
    ctt_mean: float
    ctt_max: float
    centroid: UnitId | None


def _ordered_unique_units(units: Sequence[UnitId]) -> list[UnitId]:
    ordered = list(units)
    if len(set(ordered)) != len(ordered):
        duplicates = []
        seen: set[UnitId] = set()
        for unit in ordered:
            if unit in seen and unit not in duplicates:
                duplicates.append(unit)
            seen.add(unit)
        raise ValueError(f"District units must be unique; got duplicates: {duplicates!r}")
    return ordered


def score_pairwise_district(
    units: Sequence[UnitId],
    populations: Mapping[Any, float] | None,
    od: TravelTimeTable,
    weight: PairwiseWeightMode,
) -> PairwiseDistrictScores:
    """Calculate PTT details for one district.

    Args:
        units (Sequence[UnitId]): Unique district units.
        populations (Mapping[Any, float] | None): Unit populations. Required for population-
            weighted modes and ignored for unweighted scoring.
        od (TravelTimeTable): Statewide-connected travel-time table.
        weight (PairwiseWeightMode): Pairwise weighting policy.

    Returns:
        PairwiseDistrictScores: Mean and maximum weighted pairwise travel times.
    """
    ordered_units = _ordered_unique_units(units)
    od.validate_connected()
    matrix = od.matrix_for(ordered_units)
    if len(ordered_units) < 2:
        return PairwiseDistrictScores(float("nan"), float("nan"))

    pair_rows, pair_columns = np.triu_indices(len(ordered_units), k=1)
    pair_times = matrix[pair_rows, pair_columns]
    if weight is PairwiseWeightMode.NONE:
        weighted_times = pair_times
    else:
        if populations is None:
            raise ValueError(f"population_attr is required for pairwise weight {weight.value!r}")
        validate_populations(ordered_units, populations)
        population = np.fromiter(
            (float(populations[unit]) for unit in ordered_units),
            dtype=float,
            count=len(ordered_units),
        )
        if weight is PairwiseWeightMode.POPULATION_SUM:
            pair_weights = population[pair_rows] + population[pair_columns]
        else:
            pair_weights = population[pair_rows] * population[pair_columns]
        weighted_times = pair_times * pair_weights

    return PairwiseDistrictScores(
        mean=float(weighted_times.mean()),
        maximum=float(weighted_times.max()),
    )


def score_centroid_district(
    units: Sequence[UnitId],
    populations: Mapping[Any, float] | None,
    od: TravelTimeTable,
    weight: CentroidWeightMode,
) -> CentroidDistrictScores:
    """Calculate CTT details for one district.

    Args:
        units (Sequence[UnitId]): Unique district units.
        populations (Mapping[Any, float] | None): Unit populations. Required for destination-
            population weighting and ignored for unweighted scoring.
        od (TravelTimeTable): Statewide-connected travel-time table.
        weight (CentroidWeightMode): Centroid weighting policy.

    Returns:
        CentroidDistrictScores: Mean, maximum, and selected centroid unit.
    """
    ordered_units = _ordered_unique_units(units)
    od.validate_connected()
    matrix = od.matrix_for(ordered_units)
    if len(ordered_units) < 2:
        return CentroidDistrictScores(float("nan"), float("nan"), None)

    centroid = choose_centroid(ordered_units, populations, od, weight)
    if centroid is None:
        return CentroidDistrictScores(float("nan"), float("nan"), None)
    centroid_position = ordered_units.index(centroid)
    centroid_times = matrix[centroid_position]

    if weight is CentroidWeightMode.DESTINATION_POPULATION:
        if populations is None:
            raise ValueError("population_attr is required for centroid weight 'p'")
        validate_populations(ordered_units, populations)
        destination_populations = np.fromiter(
            (float(populations[unit]) for unit in ordered_units),
            dtype=float,
            count=len(ordered_units),
        )
        weighted_times = centroid_times * destination_populations
    else:
        weighted_times = centroid_times

    return CentroidDistrictScores(
        mean=float(weighted_times.mean()),
        maximum=float(weighted_times.max()),
        centroid=centroid,
    )


def score_district(
    units: Sequence[UnitId],
    populations: Mapping[Any, float] | None,
    od: TravelTimeTable,
    *,
    ptt_weight: PairwiseWeightMode = PairwiseWeightMode.NONE,
    ctt_weight: CentroidWeightMode = CentroidWeightMode.NONE,
) -> DistrictTravelScores:
    """Calculate both PTT and CTT details for one district.

    Args:
        units (Sequence[UnitId]): Unique district units.
        populations (Mapping[Any, float] | None): Unit populations when either mode requires them.
        od (TravelTimeTable): Statewide-connected travel-time table.
        ptt_weight (PairwiseWeightMode): Pairwise weighting policy.
        ctt_weight (CentroidWeightMode): Centroid weighting policy.

    Returns:
        DistrictTravelScores: Combined district details.
    """
    pairwise = score_pairwise_district(units, populations, od, ptt_weight)
    centroid = score_centroid_district(units, populations, od, ctt_weight)
    return DistrictTravelScores(
        ptt_mean=pairwise.mean,
        ptt_max=pairwise.maximum,
        ctt_mean=centroid.mean,
        ctt_max=centroid.maximum,
        centroid=centroid.centroid,
    )
