"""District relabeling and population-dispersion helpers."""

from collections.abc import Hashable, Iterable
from dataclasses import dataclass
from typing import cast

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from .overlap import population_overlap


@dataclass(frozen=True)
class MinimumDispersion:
    """The optimal district relabeling and its displaced population.

    Attributes:
        relabeling (dict[Hashable, Hashable]): Comparison-to-reference district label mapping.
        population (float): Population displaced under the relabeling.
    """

    relabeling: dict[Hashable, Hashable]
    population: float


def _relabeling_from_positions(
    overlap: pd.DataFrame,
    source_positions: np.ndarray,
    target_positions: np.ndarray,
) -> dict[Hashable, Hashable]:
    """Map assignment-solution index pairs back to the overlap's original labels."""
    return {
        cast(Hashable, overlap.index[source_position]): cast(
            Hashable, overlap.columns[target_position]
        )
        for source_position, target_position in zip(source_positions, target_positions, strict=True)
    }


def _dispersion(overlap: pd.DataFrame, relabeling: dict[Hashable, Hashable]) -> MinimumDispersion:
    """Package a relabeling with the population it fails to retain."""
    retained = sum(overlap.at[source, target] for source, target in relabeling.items())
    return MinimumDispersion(
        relabeling=relabeling,
        population=float(overlap.to_numpy().sum() - retained),
    )


def optimal_relabeling(overlap: pd.DataFrame) -> dict[Hashable, Hashable]:
    r"""Find the one-to-one relabeling that maximizes retained overlap.

    Let the rows of :math:`M \in \mathbb{R}_{\geq 0}^{n \times n}` represent the labels of a
    comparison plan and let the columns represent the labels of a reference plan. An entry

    .. math::

        M_{ij} = \sum_{u : c(u)=i,\ r(u)=j} w(u)

    is the population or area shared by comparison district :math:`i` and reference district
    :math:`j`. Let each of :math:`X_{ij}` be binary decision variables as follows:

    .. math::

        X_{ij} =
        \begin{cases}
        1 & \text{if comparison district } i \text{ receives reference label } j, \\
        0 & \text{otherwise.}
        \end{cases}

    Then optimal-labeling problem is

    .. math::

        \begin{aligned}
        \operatorname*{maximize}_{X}\quad
            & \sum_{i=1}^{n}\sum_{j=1}^{n} M_{ij}X_{ij} \\
        \text{subject to}\quad
            & \sum_{j=1}^{n} X_{ij}=1 && \text{for every comparison district } i, \\
            & \sum_{i=1}^{n} X_{ij}=1 && \text{for every reference district } j, \\
            & X_{ij}\in\{0,1\}.
        \end{aligned}

    The row constraints assign every comparison district exactly once. The column constraints use
    every reference label exactly once. Therefore every feasible :math:`X` is a permutation matrix
    and represents one complete relabeling.

    This is the linear sum-assignment problem, not a general mixed-integer program. Its constraint
    structure guarantees an integral optimum, and SciPy's
    :func:`scipy.optimize.linear_sum_assignment` solves it directly. With ``maximize=True``, SciPy
    returns two index arrays describing the selected cells :math:`(i, \pi(i))`. This function maps
    those numeric positions back to the DataFrame's original labels. The assignment step takes
    :math:`O(n^3)` time and :math:`O(n^2)` memory; constructing the overlap matrix is separate.

    For example, suppose

    .. math::

        M =
        \begin{bmatrix}
        8 & 1 & 0 \\
        2 & 7 & 1 \\
        0 & 2 & 9
        \end{bmatrix}.

    Selecting the diagonal retains :math:`8+7+9=24`. Any off-diagonal permutation retains less,
    so SciPy returns the row-column pairs :math:`(0,0)`, :math:`(1,1)`, and :math:`(2,2)`.

    The row labels are the source labels to replace and the column labels are the target labels.
    A square matrix is required because a district relabeling must be a complete bijection.

    Args:
        overlap (pd.DataFrame): Finite, nonnegative overlap weights with unique row and column
            labels.

    Returns:
        dict[Hashable, Hashable]: A dictionary mapping every source label to one target label.

    Raises:
        TypeError: If ``overlap`` is not a DataFrame.
        ValueError: If the matrix is empty or nonsquare, labels are missing or duplicated, or a
            weight is negative or nonfinite.
    """
    weights = _validated_overlap(overlap)
    source_indices, target_indices = linear_sum_assignment(weights, maximize=True)
    return _relabeling_from_positions(overlap, source_indices, target_indices)


def _validated_overlap(overlap: pd.DataFrame) -> np.ndarray:
    """Return numeric overlap weights after validating the assignment contract."""
    if not isinstance(overlap, pd.DataFrame):
        raise TypeError("overlap must be a DataFrame")
    if overlap.empty:
        raise ValueError("overlap cannot be empty")
    if overlap.shape[0] != overlap.shape[1]:
        raise ValueError("overlap must be square for a complete district relabeling")
    if overlap.index.has_duplicates or overlap.columns.has_duplicates:
        raise ValueError("overlap labels must be unique")
    if overlap.index.isna().any() or overlap.columns.isna().any():
        raise ValueError("overlap labels cannot be missing")

    try:
        weights = overlap.to_numpy(dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError("overlap must contain numeric values") from error
    if not np.isfinite(weights).all():
        raise ValueError("overlap must contain only finite values")
    if (weights < 0).any():
        raise ValueError("overlap cannot contain negative values")
    return weights


def _parity_relabeling(
    overlap: pd.DataFrame,
    even_reference_districts: Iterable[Hashable],
) -> dict[Hashable, Hashable]:
    """Solve the parity-first, overlap-second assignment problem."""
    weights = _validated_overlap(overlap)
    even_labels = tuple(even_reference_districts)
    try:
        even_set = set(even_labels)
    except TypeError as error:
        raise ValueError("even reference district labels must be hashable") from error
    target_set = set(cast(Iterable[Hashable], overlap.columns))
    if len(even_set) != len(even_labels):
        raise ValueError("even reference district labels must be unique")
    unknown = even_set - target_set
    if unknown:
        raise ValueError(f"unknown even reference district labels: {unknown!r}")

    even_targets = np.fromiter(
        (cast(Hashable, label) in even_set for label in overlap.columns),
        dtype=np.bool_,
        count=overlap.shape[1],
    )
    even_count = int(even_targets.sum())
    odd_population = weights[:, ~even_targets].sum(axis=1)

    if even_count == 0:
        must_even = np.zeros(overlap.shape[0], dtype=np.bool_)
        must_odd = np.ones(overlap.shape[0], dtype=np.bool_)
    elif even_count == overlap.shape[0]:
        must_even = np.ones(overlap.shape[0], dtype=np.bool_)
        must_odd = np.zeros(overlap.shape[0], dtype=np.bool_)
    else:
        # Exactly the rows below the kth order statistic must receive even labels. Rows tied at the
        # threshold stay flexible so the assignment solve can maximize retained population.
        threshold = np.partition(odd_population, even_count - 1)[even_count - 1]
        must_even = odd_population < threshold
        must_odd = odd_population > threshold

    # linear_sum_assignment minimizes. Negating the overlaps maximizes retained population, while
    # infinity forbids mappings that would violate a source district's required parity group.
    cost = -weights
    cost[np.ix_(must_even, ~even_targets)] = np.inf
    cost[np.ix_(must_odd, even_targets)] = np.inf
    source_indices, target_indices = linear_sum_assignment(cost)
    return _relabeling_from_positions(overlap, source_indices, target_indices)


def population_dispersion(
    units: pd.DataFrame,
    *,
    reference: str,
    comparison: str,
    population: str,
) -> float:
    """Return population whose reference and comparison district labels differ.

    Labels are compared directly. Use :func:`minimum_population_dispersion` when the comparison
    plan has arbitrary labels that should first be optimally matched to the reference plan.

    Args:
        units (pd.DataFrame): Unit-level table containing both assignments and population.
        reference (str): Reference-plan assignment column.
        comparison (str): Comparison-plan assignment column.
        population (str): Finite, nonnegative population column.

    Returns:
        float: The sum of population in units whose two labels differ.
    """
    overlap = population_overlap(units, source=comparison, target=reference, population=population)
    retained = sum(overlap.at[label, label] for label in overlap.index if label in overlap.columns)
    return float(overlap.to_numpy().sum() - retained)


def population_dispersion_by_district(
    units: pd.DataFrame,
    *,
    reference: str,
    comparison: str,
    population: str,
) -> pd.Series:
    """Return displaced population grouped by reference district.

    Args:
        units (pd.DataFrame): Unit-level table containing both assignments and population.
        reference (str): Reference-plan assignment column.
        comparison (str): Comparison-plan assignment column.
        population (str): Finite, nonnegative population column.

    Returns:
        pd.Series: A floating-point Series indexed by reference district. Each value is the
            population that the comparison plan assigns outside that same district label.
    """
    overlap = population_overlap(units, source=comparison, target=reference, population=population)
    displaced = overlap.sum(axis=0).astype(np.float64)
    for column_position, label in enumerate(overlap.columns):
        if label in overlap.index:
            displaced.iloc[column_position] -= overlap.at[label, label]
    displaced.name = "population_dispersion"
    return displaced


def minimum_population_dispersion(
    units: pd.DataFrame,
    *,
    reference: str,
    comparison: str,
    population: str,
) -> MinimumDispersion:
    r"""Optimally relabel a comparison plan and return its population dispersion.

    The relabeling maximizes population retained in the same reference district. Equivalently, it
    minimizes the population whose district label changes. Assignment labels may be any hashable
    values, and need not be consecutive or one-indexed.

    First, :func:`gerrytools.plan_comparison.population_overlap` constructs the overlap matrix
    :math:`M`, where :math:`M_{ij}` is the population simultaneously assigned to comparison
    district :math:`i` and reference district :math:`j`. Then :func:`optimal_relabeling` finds the
    permutation :math:`\pi` maximizing retained population

    .. math::

        R^* = \max_{\pi}\sum_i M_{i,\pi(i)}.

    If

    .. math::

        P = \sum_i\sum_j M_{ij}

    is the total population, the reported minimum dispersion is

    .. math::

        D^* = P - R^*.

    Thus a unit contributes to dispersion exactly when its comparison district cannot be relabeled
    to its reference district under the optimal one-to-one mapping. The overlap construction is
    linear in the number of units up to DataFrame grouping costs. The assignment solve takes
    :math:`O(n^3)` time for :math:`n` districts, which is normally negligible compared with
    unit-level data preparation.

    Args:
        units (pd.DataFrame): Unit-level table containing both assignments and population.
        reference (str): Reference-plan assignment column.
        comparison (str): Comparison-plan assignment column to relabel.
        population (str): Finite, nonnegative population column.

    Returns:
        MinimumDispersion: The optimal comparison-to-reference label mapping and the resulting
            displaced population.

    Raises:
        ValueError: If the plans have different numbers of realized districts or any input fails
            :func:`gerrytools.plan_comparison.population_overlap` validation.
    """
    overlap = population_overlap(units, source=comparison, target=reference, population=population)
    return _dispersion(overlap, optimal_relabeling(overlap))


def minimum_population_dispersion_with_parity(
    units: pd.DataFrame,
    *,
    reference: str,
    comparison: str,
    population: str,
    even_reference_districts: Iterable[Hashable],
) -> MinimumDispersion:
    r"""Return minimum dispersion after applying the Wisconsin parity convention.

    This is a lexicographic optimization. First, it chooses which comparison districts receive
    even reference labels to minimize population moving from odd reference districts into
    even-labeled comparison districts. Among every assignment attaining that minimum, it maximizes
    population retained in the same reference district. This reproduces the former two-stage
    parity calculation without requiring a mixed-integer solver.

    Reference labels may be arbitrary hashable values. The caller identifies which labels are
    considered even, avoiding an assumption that labels are consecutive integers or one-indexed.

    **Stage 1: minimize the parity shift.** Let :math:`E` be the supplied set of even reference
    labels, let :math:`O` be its complement, and let :math:`k=|E|`. For each comparison district
    :math:`i`, define

    .. math::

        q_i = \sum_{j\in O} M_{ij}.

    This is the population in comparison district :math:`i` that formerly belonged to an odd
    reference district. Let :math:`z_i=1` when comparison district :math:`i` will receive an even
    label. The first problem is

    .. math::

        \begin{aligned}
        \operatorname*{minimize}_{z}\quad
            & \sum_i q_i z_i \\
        \text{subject to}\quad
            & \sum_i z_i=k, \\
            & z_i\in\{0,1\}.
        \end{aligned}

    Each selection consumes one of the :math:`k` available slots, so no solver is needed for this
    stage: choose the :math:`k` smallest :math:`q_i` values. The code finds the :math:`k`-th order
    statistic :math:`\theta` with :func:`numpy.partition`. A source row with
    :math:`q_i<\theta` must receive an even label, one with :math:`q_i>\theta` must receive an odd
    label, and a row tied at :math:`\theta` remains flexible. Leaving threshold ties flexible is
    important because several source sets can have the same minimum parity cost.

    **Stage 2: maximize overlap without worsening Stage 1.** Introduce the assignment variables
    :math:`X_{ij}` from :func:`optimal_relabeling`. The second problem maximizes

    .. math::

        \sum_i\sum_j M_{ij}X_{ij}

    under the ordinary one-to-one row and column constraints, plus the parity restrictions

    .. math::

        \begin{aligned}
        X_{ij} &= 0 && \text{if } q_i<\theta \text{ and } j\in O, \\
        X_{ij} &= 0 && \text{if } q_i>\theta \text{ and } j\in E.
        \end{aligned}

    Threshold-tied rows may map into either group. Since exactly :math:`k` target columns are even,
    the assignment constraints automatically choose the required number of tied rows for the even
    group. This maximizes retained population over every assignment attaining the Stage 1 optimum,
    which makes the two objectives genuinely lexicographic rather than dependent on arbitrary tie
    breaking.

    SciPy's assignment routine minimizes a cost matrix, so the implementation uses
    :math:`C=-M`. It writes :math:`+\infty` into cells forbidden by the strict parity rules and then
    calls :func:`scipy.optimize.linear_sum_assignment`. There is always a feasible assignment:
    rows strictly below and above the threshold cannot consume more than their corresponding target
    groups, and tied rows fill the remaining columns. The returned dispersion is again

    .. math::

        D = \sum_i\sum_j M_{ij} - \sum_i M_{i,\pi(i)}.

    For example, suppose the odd-overlap totals are :math:`q=(9,8,2,1)` and two reference labels
    are even. Stage 1 requires the third and fourth comparison districts to receive even labels.
    Stage 2 then solves one optimal bijection within those parity restrictions. If instead
    :math:`q=(9,2,2,2)`, three districts are tied for the last two even-label slots; the overlap
    objective decides which two enter the even group and how all four districts are matched.

    Args:
        units (pd.DataFrame): Unit-level table containing both assignments and population.
        reference (str): Reference-plan assignment column.
        comparison (str): Comparison-plan assignment column to relabel.
        population (str): Finite, nonnegative population column.
        even_reference_districts (Iterable[Hashable]): Unique reference labels treated as even for
            the parity rule.

    Returns:
        MinimumDispersion: The parity-optimal comparison-to-reference mapping and its displaced
            population.

    Raises:
        ValueError: If the plans have different numbers of realized districts, an even label is
            duplicated or absent from the reference plan, or an overlap input is invalid.
    """
    overlap = population_overlap(units, source=comparison, target=reference, population=population)
    return _dispersion(overlap, _parity_relabeling(overlap, even_reference_districts))
