"""Metric descriptions registered with :class:`PlanEvaluator`."""

from __future__ import annotations

import math
import numbers
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Iterable, Literal, TypeAlias, cast

import numpy as np

from .._types import _Dtype, _ResultShape, is_valid_metric_name
from ._base import (
    _KeyedMetric,
    _merged_keys,
    _MetricBase,
    _OutputSpec,
    _ResourceSpec,
)
from ._geometry import (
    ConvexHullRatio,
    CutEdges,
    PolsbyPopper,
    PopulationPolygon,
    RegionParts,
    RegionPieces,
    RegionSplits,
    Reock,
    Schwartzberg,
    StateClippedConvexHullRatio,
    TallyByRegion,
)

if TYPE_CHECKING:
    from gerrytools._scoring_engine import ScoringEngine

    from ..evaluator import PlanEvaluator


@dataclass(frozen=True, slots=True, init=False)
class Tally(_KeyedMetric):
    """Sum one or more numeric graph columns by district.

    Multiple registrations merge so the scoring engine reads every requested graph column in one
    pass over each assignment.

    Args:
        *columns (str): One or more numeric graph columns to sum.
        result_name (str | None, optional): Result key. Defaults to the column name for one safe
            column, or ``"tally"`` otherwise.
    """

    _kind: ClassVar[str] = "tally"

    def _default_name(self) -> str:
        """Name a single-column tally after its column when path-safe."""
        if len(self.keys) == 1 and is_valid_metric_name(self.keys[0]):
            return self.keys[0]
        return self._kind

    def _merge(self, other: _MetricBase) -> _MetricBase | None:
        if not isinstance(other, Tally):
            return None
        return Tally(*_merged_keys(self.keys, other.keys))

    def _columns(self, evaluator: PlanEvaluator) -> list[list[float]]:
        return [evaluator._numeric_node_column(key) for key in self.keys]

    def _tally_keys(self) -> tuple[str, ...]:
        return self.keys

    def _validate(self, evaluator: PlanEvaluator) -> None:
        self._columns(evaluator)

    def _prepare(self, backend: ScoringEngine, evaluator: PlanEvaluator) -> _OutputSpec:
        backend.add_tally_projection(evaluator._tally_column_indices(self.keys))
        return _OutputSpec(_ResultShape.DISTRICT, self.keys, ("float",) * len(self.keys))


@dataclass(frozen=True, slots=True)
class Eguia(_MetricBase):
    r"""Score district seat share against a population-weighted regional benchmark.

    The benchmark is the share of the population living in fixed regions where the party vote
    strictly exceeds the opposition vote. If :math:`S` is the party's district seat share and
    :math:`J` is that regional share, the result is :math:`S-J`. District and region ties are wins
    for neither party.

    Regional vote and population totals are fixed by the input geography, so they are computed
    once during evaluator preparation. Only the two district vote tallies are updated as plans
    change, and those columns share the evaluator's engine tally bank with other metrics.

    Args:
        party_vote_attr (str): Column containing nonnegative party votes.
        opposition_vote_attr (str): Column containing nonnegative opposition votes.
        region_attr (str): Column containing complete, nonmissing fixed-region labels.
        population_attr (str): Column containing nonnegative population with positive total.
        result_name (str | None, optional): Result key. Defaults to ``"eguia"``.

    Raises:
        ValueError: If a column name is empty.

    References:
        - Eguia, "A Measure of Partisan Advantage in Redistricting," Election Law Journal 21
          (2022), 84-103. https://doi.org/10.1089/elj.2020.0691
        - Duchin et al., "Locating the Representational Baseline: Republicans in Massachusetts,"
          Election Law Journal 18 (2019), 388-401. https://arxiv.org/abs/1810.09051
    """

    _kind: ClassVar[str] = "eguia"
    party_vote_attr: str
    opposition_vote_attr: str
    region_attr: str
    population_attr: str

    def __post_init__(self) -> None:
        for name, value in (
            ("party_vote_attr", self.party_vote_attr),
            ("opposition_vote_attr", self.opposition_vote_attr),
            ("region_attr", self.region_attr),
            ("population_attr", self.population_attr),
        ):
            if not isinstance(value, str) or not value:
                raise ValueError(f"Eguia {name} must be a nonempty column name")

    def _tally_keys(self) -> tuple[str, ...]:
        return self.party_vote_attr, self.opposition_vote_attr

    def _benchmark(self, evaluator: PlanEvaluator) -> float:
        return evaluator._fixed_value(("eguia", self), lambda: self._compute_benchmark(evaluator))

    def _compute_benchmark(self, evaluator: PlanEvaluator) -> float:
        party = evaluator._nonnegative_node_column(self.party_vote_attr, "Eguia")
        opposition = evaluator._nonnegative_node_column(self.opposition_vote_attr, "Eguia")
        population = evaluator._nonnegative_node_column(self.population_attr, "Eguia")
        for key, values in (
            (self.party_vote_attr, party),
            (self.opposition_vote_attr, opposition),
            (self.population_attr, population),
        ):
            if not math.isfinite(sum(values)):
                raise ValueError(f"Eguia column {key!r} must have a finite total")
        regions, labels = evaluator._region_column(self.region_attr)
        if any(region is None for region in regions):
            raise ValueError("Eguia region labels cannot be missing")

        region_party = [0.0] * len(labels)
        region_opposition = [0.0] * len(labels)
        region_population = [0.0] * len(labels)
        for region, party_value, opposition_value, population_value in zip(
            regions,
            party,
            opposition,
            population,
            strict=True,
        ):
            assert region is not None
            region_party[region] += party_value
            region_opposition[region] += opposition_value
            region_population[region] += population_value

        total_population = sum(region_population)
        if total_population <= 0:
            raise ValueError("Eguia population must have positive total")
        winning_population = sum(
            population_value
            for party_value, opposition_value, population_value in zip(
                region_party,
                region_opposition,
                region_population,
                strict=True,
            )
            if party_value > opposition_value
        )
        return winning_population / total_population

    def _validate(self, evaluator: PlanEvaluator) -> None:
        self._benchmark(evaluator)

    def _resources(self, evaluator: PlanEvaluator) -> _ResourceSpec:
        columns = (self.party_vote_attr, self.opposition_vote_attr, self.population_attr)
        return _ResourceSpec(
            node_columns=frozenset(
                evaluator._ordinary_column_resource(column) for column in columns
            ),
            region_columns=frozenset((evaluator._ordinary_column_resource(self.region_attr),)),
            fixed_values=frozenset((("eguia", self),)),
        )

    def _prepare(self, backend: ScoringEngine, evaluator: PlanEvaluator) -> _OutputSpec:
        party, opposition = evaluator._tally_column_indices(self._tally_keys())
        backend.add_eguia(party, opposition, self._benchmark(evaluator))
        return _OutputSpec(_ResultShape.PLAN, ("score",), ("float",))

    def _options(self) -> dict[str, object]:
        return {
            "party_vote_attr": self.party_vote_attr,
            "opposition_vote_attr": self.opposition_vote_attr,
            "region_attr": self.region_attr,
            "population_attr": self.population_attr,
        }


def _column_name(value: object, metric: str, parameter: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{metric} {parameter} must be a nonempty column name")
    return value


def _column_sequence(value: object, metric: str, parameter: str) -> tuple[str, ...]:
    if isinstance(value, str):
        raise TypeError(f"{metric} {parameter} must be a sequence of column names")
    try:
        columns = tuple(
            _column_name(column, metric, parameter) for column in cast("Iterable[object]", value)
        )
    except TypeError:
        raise TypeError(f"{metric} {parameter} must be a sequence of column names") from None
    if not columns:
        raise ValueError(f"{metric} {parameter} cannot be empty")
    return columns


@dataclass(frozen=True, slots=True)
class _PairedTallyMetric(_MetricBase):
    party_vote_attr: str
    opposition_vote_attr: str
    _shape: ClassVar[_ResultShape]
    _dtype: ClassVar[_Dtype]

    def _resolved_turnout_model(self) -> Literal["equal", "observed"]:
        return getattr(self, "turnout_model", "equal")

    def __post_init__(self) -> None:
        metric = type(self).__name__
        _column_name(self.party_vote_attr, metric, "party_vote_attr")
        _column_name(self.opposition_vote_attr, metric, "opposition_vote_attr")
        if self._resolved_turnout_model() not in {"equal", "observed"}:
            raise ValueError(f"{metric} turnout_model must be 'equal' or 'observed'")

    def _tally_keys(self) -> tuple[str, ...]:
        return self.party_vote_attr, self.opposition_vote_attr

    def _validate(self, evaluator: PlanEvaluator) -> None:
        metric = type(self).__name__
        evaluator._nonnegative_node_column(self.party_vote_attr, metric)
        evaluator._nonnegative_node_column(self.opposition_vote_attr, metric)

    def _prepare(self, backend: ScoringEngine, evaluator: PlanEvaluator) -> _OutputSpec:
        party, opposition = evaluator._tally_column_indices(self._tally_keys())
        backend.add_paired_derived(
            self._kind,
            party,
            opposition,
            self._resolved_turnout_model(),
        )
        return _OutputSpec(self._shape, ("score",), (self._dtype,))

    def _options(self) -> dict[str, object]:
        return {
            "party_vote_attr": self.party_vote_attr,
            "opposition_vote_attr": self.opposition_vote_attr,
        }


@dataclass(frozen=True, slots=True)
class _TurnoutPairedTallyMetric(_PairedTallyMetric):
    """Paired-tally base for metrics that expose ``turnout_model`` as a public option."""

    turnout_model: Literal["equal", "observed"] = "equal"

    def _options(self) -> dict[str, object]:
        return {**_PairedTallyMetric._options(self), "turnout_model": self.turnout_model}


@dataclass(frozen=True, slots=True)
class DistrictVoteShares(_PairedTallyMetric):
    r"""Return the party's two-party vote share in each district.

    A zero-turnout district is reported as ``NaN``. See
    :func:`gerrytools.scoring.formulas.district_vote_shares` for the exact formula.

    Args:
        party_vote_attr (str): Column containing nonnegative party votes.
        opposition_vote_attr (str): Column containing nonnegative opposition votes.
        result_name (str | None, optional): Result key. Defaults to ``"district_vote_shares"``.
    """

    _kind: ClassVar[str] = "district_vote_shares"
    _shape = _ResultShape.DISTRICT
    _dtype: ClassVar[_Dtype] = "float"


@dataclass(frozen=True, slots=True)
class DistrictWins(_PairedTallyMetric):
    """Identify districts strictly won by the party; ties are not wins.

    Args:
        party_vote_attr (str): Column containing nonnegative party votes.
        opposition_vote_attr (str): Column containing nonnegative opposition votes.
        result_name (str | None, optional): Result key. Defaults to ``"district_wins"``.
    """

    _kind: ClassVar[str] = "district_wins"
    _shape = _ResultShape.DISTRICT
    _dtype: ClassVar[_Dtype] = "bool"


@dataclass(frozen=True, slots=True)
class Seats(_PairedTallyMetric):
    """Count districts strictly won by the party under the two-party vote.

    Args:
        party_vote_attr (str): Column containing nonnegative party votes.
        opposition_vote_attr (str): Column containing nonnegative opposition votes.
        result_name (str | None, optional): Result key. Defaults to ``"seats"``.
    """

    _kind: ClassVar[str] = "seats"
    _shape = _ResultShape.PLAN
    _dtype: ClassVar[_Dtype] = "int"


@dataclass(frozen=True, slots=True)
class OverallVoteShare(_PairedTallyMetric):
    """Return the party's turnout-weighted aggregate two-party vote share.

    Args:
        party_vote_attr (str): Column containing nonnegative party votes.
        opposition_vote_attr (str): Column containing nonnegative opposition votes.
        result_name (str | None, optional): Result key. Defaults to ``"overall_vote_share"``.
    """

    _kind: ClassVar[str] = "overall_vote_share"
    _shape = _ResultShape.PLAN
    _dtype: ClassVar[_Dtype] = "float"


@dataclass(frozen=True, slots=True)
class EfficiencyGap(_PairedTallyMetric):
    r"""Return the wasted-vote efficiency gap from the party's point of view.

    Positive values favor the party. The winning threshold and provisional tie convention exactly
    match :func:`gerrytools.scoring.formulas.efficiency_gap`.

    Args:
        party_vote_attr (str): Column containing nonnegative party votes.
        opposition_vote_attr (str): Column containing nonnegative opposition votes.
        result_name (str | None, optional): Result key. Defaults to ``"efficiency_gap"``.

    References:
        - Bernstein and Duchin, "A Formula Goes to Court: Partisan Gerrymandering and the
          Efficiency Gap." https://arxiv.org/abs/1705.10812
        - Stephanopoulos and McGhee, "Partisan Gerrymandering and the Efficiency Gap."
          https://lawreview.uchicago.edu/online-archive/partisan-gerrymandering-and-efficiency-gap
    """

    _kind: ClassVar[str] = "efficiency_gap"
    _shape = _ResultShape.PLAN
    _dtype: ClassVar[_Dtype] = "float"


@dataclass(frozen=True, slots=True)
class SimplifiedEfficiencyGap(_PairedTallyMetric):
    r"""Return :math:`S-2V+1/2`, the equal-turnout seat-vote efficiency-gap formula.

    See :class:`EfficiencyGap` when the original wasted-vote definition is required.

    Args:
        party_vote_attr (str): Column containing nonnegative party votes.
        opposition_vote_attr (str): Column containing nonnegative opposition votes.
        result_name (str | None, optional): Result key. Defaults to
            ``"simplified_efficiency_gap"``.
    """

    _kind: ClassVar[str] = "simplified_efficiency_gap"
    _shape = _ResultShape.PLAN
    _dtype: ClassVar[_Dtype] = "float"


@dataclass(frozen=True, slots=True)
class MeanMedian(_PairedTallyMetric):
    r"""Return median district vote share minus mean district vote share.

    The mean weights districts equally. A zero-turnout district makes the score ``NaN``.

    Args:
        party_vote_attr (str): Column containing nonnegative party votes.
        opposition_vote_attr (str): Column containing nonnegative opposition votes.
        result_name (str | None, optional): Result key. Defaults to ``"mean_median"``.

    References:
        - DeFord et al., "Implementing Partisan Symmetry: Problems and Paradoxes."
          https://doi.org/10.1017/pan.2021.49
        - Grofman, "Measures of Bias and Proportionality in Seats-Votes Relationships."
          https://www.jstor.org/stable/25791195
    """

    _kind: ClassVar[str] = "mean_median"
    _shape = _ResultShape.PLAN
    _dtype: ClassVar[_Dtype] = "float"


@dataclass(frozen=True, slots=True)
class Disproportionality(_PairedTallyMetric):
    r"""Return the party's signed seat-share minus aggregate vote-share gap.

    Positive values indicate seat overrepresentation; negative values indicate seat
    underrepresentation. District ties are not wins.

    Args:
        party_vote_attr (str): Column containing nonnegative party votes.
        opposition_vote_attr (str): Column containing nonnegative opposition votes.
        result_name (str | None, optional): Result key. Defaults to ``"disproportionality"``.
    """

    _kind: ClassVar[str] = "disproportionality"
    _shape = _ResultShape.PLAN
    _dtype: ClassVar[_Dtype] = "float"


@dataclass(frozen=True, slots=True)
class PartisanBias(_TurnoutPairedTallyMetric):
    r"""Return partisan bias at 50 percent under uniform partisan swing.

    Districts within numerical tolerance of the reference share contribute half a seat.
    ``turnout_model="equal"`` gives valid districts equal weight when locating the observed
    election; ``"observed"`` uses aggregate two-party turnout.

    Args:
        party_vote_attr (str): Column containing nonnegative party votes.
        opposition_vote_attr (str): Column containing nonnegative opposition votes.
        turnout_model (str, optional): Reference-vote model. Defaults to ``"equal"``; also accepts
            ``"observed"``.
        result_name (str | None, optional): Result key. Defaults to ``"partisan_bias"``.

    References:
        - DeFord et al., "Implementing Partisan Symmetry: Problems and Paradoxes."
          https://doi.org/10.1017/pan.2021.49
        - Katz, King, and Rosenblatt, "Theoretical Foundations and Empirical Evaluations of
          Partisan Fairness." https://doi.org/10.1017/S000305541900056X
    """

    _kind: ClassVar[str] = "partisan_bias"
    _shape = _ResultShape.PLAN
    _dtype: ClassVar[_Dtype] = "float"


@dataclass(frozen=True, slots=True)
class PartisanGini(_TurnoutPairedTallyMetric):
    r"""Return unsigned partisan Gini under uniform partisan swing.

    ``turnout_model`` has the same meaning as in :class:`PartisanBias`. A zero-turnout district
    makes the score ``NaN``.

    Args:
        party_vote_attr (str): Column containing nonnegative party votes.
        opposition_vote_attr (str): Column containing nonnegative opposition votes.
        turnout_model (str, optional): Reference-vote model. Defaults to ``"equal"``; also accepts
            ``"observed"``.
        result_name (str | None, optional): Result key. Defaults to ``"partisan_gini"``.

    References:
        - DeFord et al., "Implementing Partisan Symmetry: Problems and Paradoxes."
          https://doi.org/10.1017/pan.2021.49
        - Grofman, "Measures of Bias and Proportionality in Seats-Votes Relationships."
          https://www.jstor.org/stable/25791195
        - Katz, King, and Rosenblatt, "Theoretical Foundations and Empirical Evaluations of
          Partisan Fairness." https://doi.org/10.1017/S000305541900056X
    """

    _kind: ClassVar[str] = "partisan_gini"
    _shape = _ResultShape.PLAN
    _dtype: ClassVar[_Dtype] = "float"


@dataclass(frozen=True, slots=True)
class _PopulationMetric(_MetricBase):
    population_attr: str
    _shape: ClassVar[_ResultShape]

    def _resolved_relative_to_ideal(self) -> bool:
        return getattr(self, "relative_to_ideal", False)

    def __post_init__(self) -> None:
        _column_name(self.population_attr, type(self).__name__, "population_attr")
        if not isinstance(self._resolved_relative_to_ideal(), bool):
            raise TypeError(f"{type(self).__name__} relative_to_ideal must be a bool")

    def _tally_keys(self) -> tuple[str, ...]:
        return (self.population_attr,)

    def _validate(self, evaluator: PlanEvaluator) -> None:
        values = evaluator._nonnegative_node_column(self.population_attr, type(self).__name__)
        if sum(values) <= 0:
            raise ValueError(f"{type(self).__name__} population must have a positive total")

    def _prepare(self, backend: ScoringEngine, evaluator: PlanEvaluator) -> _OutputSpec:
        (population,) = evaluator._tally_column_indices(self._tally_keys())
        backend.add_population_derived(self._kind, population, self._resolved_relative_to_ideal())
        return _OutputSpec(self._shape, ("score",), ("float",))

    def _options(self) -> dict[str, object]:
        return {"population_attr": self.population_attr}


@dataclass(frozen=True, slots=True)
class _RelativePopulationMetric(_PopulationMetric):
    """Population base for metrics that expose ``relative_to_ideal`` as a public option."""

    relative_to_ideal: bool = False

    def _options(self) -> dict[str, object]:
        return {
            **_PopulationMetric._options(self),
            "relative_to_ideal": self.relative_to_ideal,
        }


@dataclass(frozen=True, slots=True)
class PopulationDeviations(_PopulationMetric):
    r"""Return each district's signed proportional deviation from ideal population.

    Args:
        population_attr (str): Column containing nonnegative population values.
        result_name (str | None, optional): Result key. Defaults to ``"population_deviations"``.

    References:
        - Evenwel v. Abbott, 578 U.S. 54 (2016).
          https://www.govinfo.gov/app/details/USREPORTS-578
        - Duchin and Walch, eds., *Political Geometry*.
          https://doi.org/10.1007/978-3-319-69161-9
    """

    _kind: ClassVar[str] = "population_deviations"
    _shape = _ResultShape.DISTRICT


@dataclass(frozen=True, slots=True)
class MaxAbsolutePopulationDeviation(_RelativePopulationMetric):
    """Return the largest one-district absolute departure from ideal population.

    Args:
        population_attr (str): Column containing nonnegative population values.
        relative_to_ideal (bool, optional): Whether to return a proportion of ideal population.
            Defaults to False, which returns a population count.
        result_name (str | None, optional): Result key. Defaults to
            ``"max_absolute_population_deviation"``.
    """

    _kind: ClassVar[str] = "max_absolute_population_deviation"
    _shape = _ResultShape.PLAN


@dataclass(frozen=True, slots=True)
class MaxPopulationDeviation(_RelativePopulationMetric):
    r"""Return the plan's top-to-bottom population range.

    With ``relative_to_ideal=True``, this is the conventional maximum population deviation
    described in Evenwel v. Abbott, 578 U.S. 54, 60 n.2 (2016).
    https://www.govinfo.gov/app/details/USREPORTS-578

    Args:
        population_attr (str): Column containing nonnegative population values.
        relative_to_ideal (bool, optional): Whether to return a proportion of ideal population.
            Defaults to False, which returns a population count.
        result_name (str | None, optional): Result key. Defaults to
            ``"max_population_deviation"``.
    """

    _kind: ClassVar[str] = "max_population_deviation"
    _shape = _ResultShape.PLAN


@dataclass(frozen=True, slots=True)
class _DemographicMetric(_MetricBase):
    subgroup_population_attr: str
    total_population_attr: str
    _shape: ClassVar[_ResultShape]
    _dtype: ClassVar[_Dtype]

    def _resolved_threshold(self) -> float:
        return getattr(self, "threshold", 0.5)

    def __post_init__(self) -> None:
        metric = type(self).__name__
        _column_name(self.subgroup_population_attr, metric, "subgroup_population_attr")
        _column_name(self.total_population_attr, metric, "total_population_attr")
        threshold = self._resolved_threshold()
        if isinstance(threshold, bool) or not isinstance(threshold, numbers.Real):
            raise ValueError(f"{metric} threshold must be finite and between zero and one")
        threshold_value = float(threshold)
        if not math.isfinite(threshold_value) or not 0 <= threshold_value <= 1:
            raise ValueError(f"{metric} threshold must be finite and between zero and one")

    def _tally_keys(self) -> tuple[str, ...]:
        return self.subgroup_population_attr, self.total_population_attr

    def _validate(self, evaluator: PlanEvaluator) -> None:
        metric = type(self).__name__
        subgroup = evaluator._nonnegative_node_column(self.subgroup_population_attr, metric)
        total = evaluator._nonnegative_node_column(self.total_population_attr, metric)
        if any(
            subgroup_value > total_value for subgroup_value, total_value in zip(subgroup, total)
        ):
            raise ValueError(f"{metric} subgroup cannot exceed total")

    def _prepare(self, backend: ScoringEngine, evaluator: PlanEvaluator) -> _OutputSpec:
        subgroup, total = evaluator._tally_column_indices(self._tally_keys())
        backend.add_demographic_derived(
            self._kind,
            subgroup,
            total,
            float(self._resolved_threshold()),
        )
        return _OutputSpec(self._shape, ("score",), (self._dtype,))

    def _options(self) -> dict[str, object]:
        return {
            "subgroup_population_attr": self.subgroup_population_attr,
            "total_population_attr": self.total_population_attr,
        }


@dataclass(frozen=True, slots=True)
class _ThresholdDemographicMetric(_DemographicMetric):
    """Demographic base for metrics that expose ``threshold`` as a public option."""

    threshold: float | np.floating = 0.5

    def _options(self) -> dict[str, object]:
        return {**_DemographicMetric._options(self), "threshold": float(self.threshold)}


@dataclass(frozen=True, slots=True)
class DemographicShares(_DemographicMetric):
    """Return a subgroup's share of the specified total in each district.

    Args:
        subgroup_population_attr (str): Column containing nonnegative subgroup population.
        total_population_attr (str): Column containing matching nonnegative total population.
        result_name (str | None, optional): Result key. Defaults to ``"demographic_shares"``.
    """

    _kind: ClassVar[str] = "demographic_shares"
    _shape = _ResultShape.DISTRICT
    _dtype: ClassVar[_Dtype] = "float"


@dataclass(frozen=True, slots=True)
class DistrictsAboveThreshold(_ThresholdDemographicMetric):
    """Count districts whose subgroup share is strictly greater than ``threshold``.

    Args:
        subgroup_population_attr (str): Column containing nonnegative subgroup population.
        total_population_attr (str): Column containing matching nonnegative total population.
        threshold (float, optional): Share threshold in ``[0, 1]``. Defaults to 0.5.
        result_name (str | None, optional): Result key. Defaults to
            ``"districts_above_threshold"``.
    """

    _kind: ClassVar[str] = "districts_above_threshold"
    _shape = _ResultShape.PLAN
    _dtype: ClassVar[_Dtype] = "int"


@dataclass(frozen=True, slots=True)
class _CrossElectionMetric(_MetricBase):
    party_vote_attrs: tuple[str, ...]
    opposition_vote_attrs: tuple[str, ...]
    _shape: ClassVar[_ResultShape]
    _dtype: ClassVar[_Dtype]

    def _resolved_vote_share_margin(self) -> float:
        return getattr(self, "vote_share_margin", 0.03)

    def __post_init__(self) -> None:
        metric = type(self).__name__
        party = _column_sequence(self.party_vote_attrs, metric, "party_vote_attrs")
        opposition = _column_sequence(
            self.opposition_vote_attrs,
            metric,
            "opposition_vote_attrs",
        )
        if len(party) != len(opposition):
            raise ValueError(
                f"{metric} party_vote_attrs and opposition_vote_attrs must have equal length"
            )
        object.__setattr__(self, "party_vote_attrs", party)
        object.__setattr__(self, "opposition_vote_attrs", opposition)
        margin = self._resolved_vote_share_margin()
        if isinstance(margin, bool) or not isinstance(margin, numbers.Real):
            raise ValueError(
                f"{metric} vote_share_margin must be finite and between zero and one half"
            )
        margin_value = float(margin)
        if not math.isfinite(margin_value) or not 0 <= margin_value <= 0.5:
            raise ValueError(
                f"{metric} vote_share_margin must be finite and between zero and one half"
            )

    def _tally_keys(self) -> tuple[str, ...]:
        return _merged_keys(self.party_vote_attrs, self.opposition_vote_attrs)

    def _validate(self, evaluator: PlanEvaluator) -> None:
        metric = type(self).__name__
        for column in self._tally_keys():
            evaluator._nonnegative_node_column(column, metric)

    def _prepare(self, backend: ScoringEngine, evaluator: PlanEvaluator) -> _OutputSpec:
        party = evaluator._tally_column_indices(self.party_vote_attrs)
        opposition = evaluator._tally_column_indices(self.opposition_vote_attrs)
        backend.add_cross_election_derived(
            self._kind,
            party,
            opposition,
            float(self._resolved_vote_share_margin()),
        )
        return _OutputSpec(self._shape, ("score",), (self._dtype,))

    def _options(self) -> dict[str, object]:
        return {
            "party_vote_attrs": self.party_vote_attrs,
            "opposition_vote_attrs": self.opposition_vote_attrs,
        }


@dataclass(frozen=True, slots=True)
class _PointsCrossElectionMetric(_CrossElectionMetric):
    """Cross-election base for metrics that expose ``vote_share_margin`` as an option."""

    vote_share_margin: float | np.floating = 0.03

    def _options(self) -> dict[str, object]:
        return {
            **_CrossElectionMetric._options(self),
            "vote_share_margin": float(self.vote_share_margin),
        }


@dataclass(frozen=True, slots=True)
class CompetitiveContests(_PointsCrossElectionMetric):
    """Count election-district contests in an open interval around 50 percent.

    Args:
        party_vote_attrs (tuple[str, ...]): Party-vote columns in election order.
        opposition_vote_attrs (tuple[str, ...]): Matching opposition-vote columns.
        vote_share_margin (float, optional): Half-width of the competitive interval. Defaults to
            0.03.
        result_name (str | None, optional): Result key. Defaults to ``"competitive_contests"``.
    """

    _kind: ClassVar[str] = "competitive_contests"
    _shape = _ResultShape.PLAN
    _dtype: ClassVar[_Dtype] = "int"


@dataclass(frozen=True, slots=True)
class PartyWinsByDistrict(_CrossElectionMetric):
    """Count strict party wins in each district across the supplied elections.

    Args:
        party_vote_attrs (tuple[str, ...]): Party-vote columns in election order.
        opposition_vote_attrs (tuple[str, ...]): Matching opposition-vote columns.
        result_name (str | None, optional): Result key. Defaults to ``"party_wins_by_district"``.
    """

    _kind: ClassVar[str] = "party_wins_by_district"
    _shape = _ResultShape.DISTRICT
    _dtype: ClassVar[_Dtype] = "int"


@dataclass(frozen=True, slots=True)
class SwingDistricts(_CrossElectionMetric):
    """Count districts that are not strict wins for one side in every election.

    Args:
        party_vote_attrs (tuple[str, ...]): Party-vote columns in election order.
        opposition_vote_attrs (tuple[str, ...]): Matching opposition-vote columns.
        result_name (str | None, optional): Result key. Defaults to ``"swing_districts"``.
    """

    _kind: ClassVar[str] = "swing_districts"
    _shape = _ResultShape.PLAN
    _dtype: ClassVar[_Dtype] = "int"


@dataclass(frozen=True, slots=True)
class PartyDistricts(_CrossElectionMetric):
    """Count districts strictly won by the party in every supplied election.

    Args:
        party_vote_attrs (tuple[str, ...]): Party-vote columns in election order.
        opposition_vote_attrs (tuple[str, ...]): Matching opposition-vote columns.
        result_name (str | None, optional): Result key. Defaults to ``"party_districts"``.
    """

    _kind: ClassVar[str] = "party_districts"
    _shape = _ResultShape.PLAN
    _dtype: ClassVar[_Dtype] = "int"


@dataclass(frozen=True, slots=True)
class OppositionPartyDistricts(_CrossElectionMetric):
    """Count districts strictly won by the opposition in every supplied election.

    Args:
        party_vote_attrs (tuple[str, ...]): Party-vote columns in election order.
        opposition_vote_attrs (tuple[str, ...]): Matching opposition-vote columns.
        result_name (str | None, optional): Result key. Defaults to
            ``"opposition_party_districts"``.
    """

    _kind: ClassVar[str] = "opposition_party_districts"
    _shape = _ResultShape.PLAN
    _dtype: ClassVar[_Dtype] = "int"


@dataclass(frozen=True, slots=True)
class AggregateSeats(_CrossElectionMetric):
    """Count strict party wins across every supplied election and district.

    Args:
        party_vote_attrs (tuple[str, ...]): Party-vote columns in election order.
        opposition_vote_attrs (tuple[str, ...]): Matching opposition-vote columns.
        result_name (str | None, optional): Result key. Defaults to ``"aggregate_seats"``.
    """

    _kind: ClassVar[str] = "aggregate_seats"
    _shape = _ResultShape.PLAN
    _dtype: ClassVar[_Dtype] = "int"


@dataclass(frozen=True, slots=True)
class MeanSignedSeatVoteGap(_CrossElectionMetric):
    """Average signed seat-share minus aggregate vote-share gaps over elections.

    Args:
        party_vote_attrs (tuple[str, ...]): Party-vote columns in election order.
        opposition_vote_attrs (tuple[str, ...]): Matching opposition-vote columns.
        result_name (str | None, optional): Result key. Defaults to
            ``"mean_signed_seat_vote_gap"``.
    """

    _kind: ClassVar[str] = "mean_signed_seat_vote_gap"
    _shape = _ResultShape.PLAN
    _dtype: ClassVar[_Dtype] = "float"


@dataclass(frozen=True, slots=True)
class MeanAbsoluteSeatVoteGap(_CrossElectionMetric):
    """Average absolute seat-share minus aggregate vote-share gaps over elections.

    Args:
        party_vote_attrs (tuple[str, ...]): Party-vote columns in election order.
        opposition_vote_attrs (tuple[str, ...]): Matching opposition-vote columns.
        result_name (str | None, optional): Result key. Defaults to
            ``"mean_absolute_seat_vote_gap"``.
    """

    _kind: ClassVar[str] = "mean_absolute_seat_vote_gap"
    _shape = _ResultShape.PLAN
    _dtype: ClassVar[_Dtype] = "float"


Metric: TypeAlias = (
    AggregateSeats
    | CompetitiveContests
    | ConvexHullRatio
    | CutEdges
    | DemographicShares
    | DistrictVoteShares
    | DistrictWins
    | DistrictsAboveThreshold
    | Disproportionality
    | EfficiencyGap
    | Eguia
    | MaxAbsolutePopulationDeviation
    | MaxPopulationDeviation
    | MeanAbsoluteSeatVoteGap
    | MeanMedian
    | MeanSignedSeatVoteGap
    | OppositionPartyDistricts
    | OverallVoteShare
    | PartisanBias
    | PartisanGini
    | PartyDistricts
    | PartyWinsByDistrict
    | PolsbyPopper
    | PopulationDeviations
    | PopulationPolygon
    | RegionPieces
    | RegionParts
    | RegionSplits
    | Reock
    | Schwartzberg
    | Seats
    | SimplifiedEfficiencyGap
    | StateClippedConvexHullRatio
    | SwingDistricts
    | Tally
    | TallyByRegion
)
"""Union of the metric descriptions :meth:`PlanEvaluator.add_metric` accepts.

The set is closed because registering a metric requires matching scoring-engine support, so
user-defined classes cannot be scored.
"""
