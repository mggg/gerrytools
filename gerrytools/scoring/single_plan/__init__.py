"""Single-plan functions for scoring-engine metric descriptions.

These functions prepare a fresh :class:`PlanEvaluator` for each call. Use an evaluator directly
when evaluating several metrics or plans so graph, geometry, and engine resources are prepared
once.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Literal, cast

from geopandas import GeoDataFrame
from pandas import DataFrame, Series

from ..metrics import (
    AggregateSeats,
    CompetitiveContests,
    DemographicShares,
    Disproportionality,
    DistrictsAboveThreshold,
    DistrictVoteShares,
    DistrictWins,
    EfficiencyGap,
    Eguia,
    MaxAbsolutePopulationDeviation,
    MaxPopulationDeviation,
    MeanAbsoluteSeatVoteGap,
    MeanMedian,
    MeanSignedSeatVoteGap,
    OppositionPartyDistricts,
    OverallVoteShare,
    PartisanBias,
    PartisanGini,
    PartyDistricts,
    PartyWinsByDistrict,
    PopulationDeviations,
    Seats,
    SimplifiedEfficiencyGap,
    SwingDistricts,
    Tally,
)
from ._base import (
    GeoAssignment,
    SinglePlanSource,
    _columns,
    _evaluate,
    _expect,
)
from ._geometry import (
    convex_hull_ratio as convex_hull_ratio,
)
from ._geometry import (
    cut_edges as cut_edges,
)
from ._geometry import (
    polsby_popper as polsby_popper,
)
from ._geometry import (
    population_polygon as population_polygon,
)
from ._geometry import (
    region_parts as region_parts,
)
from ._geometry import (
    region_pieces as region_pieces,
)
from ._geometry import (
    region_splits as region_splits,
)
from ._geometry import (
    reock as reock,
)
from ._geometry import (
    schwartzberg as schwartzberg,
)
from ._geometry import (
    state_clipped_convex_hull_ratio as state_clipped_convex_hull_ratio,
)
from ._geometry import (
    tally_by_region as tally_by_region,
)


def _election_columns(values: Sequence[str]) -> tuple[str, ...]:
    # Preserve a mistaken bare string so the metric description can reject it clearly.
    return cast("tuple[str, ...]", values) if isinstance(values, str) else tuple(values)


def tally(
    source: SinglePlanSource,
    assignment: GeoAssignment | None = None,
    *,
    columns: str | Iterable[str],
    result_name: str | None = None,
    geometry: GeoDataFrame | None = None,
    node_id_column: str | None = None,
) -> Series | DataFrame:
    """Sum one or more unit columns by district for a single plan.

    Args:
        source (SinglePlanSource): GerryChain Partition, NetworkX graph, or authoritative
            GeoDataFrame.
        assignment (GeoAssignment | None, optional): Graph assignment mapping or node-ordered
            sequence, or a GeoDataFrame assignment column, mapping, or row-ordered sequence.
            Defaults to None; a Partition supplies its own assignment.
        columns (str | Iterable[str]): One column name, or several to tally in a single pass.
        result_name (str | None): Optional result name. A single-column tally is named after its
            column when that name is usable as a path component, and ``"tally"`` otherwise.
        geometry (GeoDataFrame | None, optional): Authoritative GeoDataFrame for a graph or
            Partition source. Defaults to None.
        node_id_column (str | None, optional): Geometry column containing the graph's node
            labels. Defaults to None.

    Returns:
        Series | DataFrame: District-indexed sums. A single column produces a Series; multiple
            columns produce a DataFrame.
    """
    return _expect(
        (Series, DataFrame),
        _evaluate(
            source,
            assignment,
            Tally(*_columns(columns), result_name=result_name),
            geometry=geometry,
            node_id_column=node_id_column,
        ),
    )


def eguia(
    source: SinglePlanSource,
    assignment: GeoAssignment | None = None,
    *,
    party_vote_attr: str,
    opposition_vote_attr: str,
    region_attr: str,
    population_attr: str,
    geometry: GeoDataFrame | None = None,
    node_id_column: str | None = None,
) -> float:
    r"""Calculate Eguia's regional-benchmark partisan-advantage score for one plan.

    The score is the party's district seat share minus the share of the population living in
    fixed regions where the party strictly wins the two-party vote. District and region ties are
    wins for neither party. See :class:`Eguia` for the full definition and input contract.

    Args:
        source (SinglePlanSource): GerryChain Partition, NetworkX graph, or authoritative
            GeoDataFrame.
        assignment (GeoAssignment | None, optional): Graph assignment mapping or node-ordered
            sequence, or a GeoDataFrame assignment column, mapping, or row-ordered sequence.
            Defaults to None; a Partition supplies its own assignment.
        party_vote_attr (str): Column containing nonnegative party votes.
        opposition_vote_attr (str): Column containing nonnegative opposition votes.
        region_attr (str): Column containing complete fixed-region labels.
        population_attr (str): Column containing nonnegative population with positive total.
        geometry (GeoDataFrame | None, optional): Authoritative GeoDataFrame for a graph or
            Partition source. Defaults to None.
        node_id_column (str | None, optional): Geometry column containing the graph's node
            labels. Defaults to None.

    Returns:
        float: The party's district seat share minus its population-weighted regional benchmark.

    References:
        - Eguia, "A Measure of Partisan Advantage in Redistricting," Election Law Journal 21
          (2022), 84-103. https://doi.org/10.1089/elj.2020.0691
        - Duchin et al., "Locating the Representational Baseline: Republicans in Massachusetts,"
          Election Law Journal 18 (2019), 388-401. https://arxiv.org/abs/1810.09051
    """
    return _expect(
        float,
        _evaluate(
            source,
            assignment,
            Eguia(
                party_vote_attr=party_vote_attr,
                opposition_vote_attr=opposition_vote_attr,
                region_attr=region_attr,
                population_attr=population_attr,
            ),
            geometry=geometry,
            node_id_column=node_id_column,
        ),
    )


def district_vote_shares(
    source: SinglePlanSource,
    assignment: GeoAssignment | None = None,
    *,
    party_vote_attr: str,
    opposition_vote_attr: str,
    geometry: GeoDataFrame | None = None,
    node_id_column: str | None = None,
) -> Series:
    """Return the party's two-party vote share in each district of one plan.

    Args:
        source (SinglePlanSource): Partition, graph, or authoritative GeoDataFrame containing the
            vote columns.
        assignment (GeoAssignment | None): District assignment. Required for graph and
            GeoDataFrame sources and omitted for a Partition. A GeoDataFrame also accepts an
            assignment column name.
        party_vote_attr (str): Column containing nonnegative party votes.
        opposition_vote_attr (str): Column containing nonnegative opposition votes.
        geometry (GeoDataFrame | None, optional): Authoritative GeoDataFrame for a graph or
            Partition source. Defaults to None.
        node_id_column (str | None, optional): Geometry column containing the graph's node
            labels. Defaults to None.

    Returns:
        Series: District-indexed party vote shares, with ``NaN`` for zero-turnout districts.
    """
    return _expect(
        Series,
        _evaluate(
            source,
            assignment,
            DistrictVoteShares(party_vote_attr, opposition_vote_attr),
            geometry=geometry,
            node_id_column=node_id_column,
        ),
    )


def district_wins(
    source: SinglePlanSource,
    assignment: GeoAssignment | None = None,
    *,
    party_vote_attr: str,
    opposition_vote_attr: str,
    geometry: GeoDataFrame | None = None,
    node_id_column: str | None = None,
) -> Series:
    """Identify districts strictly won by the party in one plan.

    Args:
        source (SinglePlanSource): Partition, graph, or authoritative GeoDataFrame containing the
            vote columns.
        assignment (GeoAssignment | None): District assignment. Required for graph and
            GeoDataFrame sources and omitted for a Partition. A GeoDataFrame also accepts an
            assignment column name.
        party_vote_attr (str): Column containing nonnegative party votes.
        opposition_vote_attr (str): Column containing nonnegative opposition votes.
        geometry (GeoDataFrame | None, optional): Authoritative GeoDataFrame for a graph or
            Partition source. Defaults to None.
        node_id_column (str | None, optional): Geometry column containing the graph's node
            labels. Defaults to None.

    Returns:
        Series: District-indexed Boolean indicators. Ties and zero-turnout districts are False.
    """
    return _expect(
        Series,
        _evaluate(
            source,
            assignment,
            DistrictWins(party_vote_attr, opposition_vote_attr),
            geometry=geometry,
            node_id_column=node_id_column,
        ),
    )


def seats(
    source: SinglePlanSource,
    assignment: GeoAssignment | None = None,
    *,
    party_vote_attr: str,
    opposition_vote_attr: str,
    geometry: GeoDataFrame | None = None,
    node_id_column: str | None = None,
) -> int:
    """Count districts strictly won by the party in one plan.

    Args:
        source (SinglePlanSource): Partition, graph, or authoritative GeoDataFrame containing the
            vote columns.
        assignment (GeoAssignment | None): District assignment. Required for graph and
            GeoDataFrame sources and omitted for a Partition. A GeoDataFrame also accepts an
            assignment column name.
        party_vote_attr (str): Column containing nonnegative party votes.
        opposition_vote_attr (str): Column containing nonnegative opposition votes.
        geometry (GeoDataFrame | None, optional): Authoritative GeoDataFrame for a graph or
            Partition source. Defaults to None.
        node_id_column (str | None, optional): Geometry column containing the graph's node
            labels. Defaults to None.

    Returns:
        int: Number of districts in which the party strictly exceeds the opposition.
    """
    return _expect(
        int,
        _evaluate(
            source,
            assignment,
            Seats(party_vote_attr, opposition_vote_attr),
            geometry=geometry,
            node_id_column=node_id_column,
        ),
    )


def overall_vote_share(
    source: SinglePlanSource,
    assignment: GeoAssignment | None = None,
    *,
    party_vote_attr: str,
    opposition_vote_attr: str,
    geometry: GeoDataFrame | None = None,
    node_id_column: str | None = None,
) -> float:
    """Return the party's aggregate two-party vote share in one plan.

    Args:
        source (SinglePlanSource): Partition, graph, or authoritative GeoDataFrame containing the
            vote columns.
        assignment (GeoAssignment | None): District assignment. Required for graph and
            GeoDataFrame sources and omitted for a Partition. A GeoDataFrame also accepts an
            assignment column name.
        party_vote_attr (str): Column containing nonnegative party votes.
        opposition_vote_attr (str): Column containing nonnegative opposition votes.
        geometry (GeoDataFrame | None, optional): Authoritative GeoDataFrame for a graph or
            Partition source. Defaults to None.
        node_id_column (str | None, optional): Geometry column containing the graph's node
            labels. Defaults to None.

    Returns:
        float: The party's turnout-weighted aggregate two-party vote share.
    """
    return _expect(
        float,
        _evaluate(
            source,
            assignment,
            OverallVoteShare(party_vote_attr, opposition_vote_attr),
            geometry=geometry,
            node_id_column=node_id_column,
        ),
    )


def efficiency_gap(
    source: SinglePlanSource,
    assignment: GeoAssignment | None = None,
    *,
    party_vote_attr: str,
    opposition_vote_attr: str,
    geometry: GeoDataFrame | None = None,
    node_id_column: str | None = None,
) -> float:
    """Return the wasted-vote efficiency gap for one plan.

    Args:
        source (SinglePlanSource): Partition, graph, or authoritative GeoDataFrame containing the
            vote columns.
        assignment (GeoAssignment | None): District assignment. Required for graph and
            GeoDataFrame sources and omitted for a Partition. A GeoDataFrame also accepts an
            assignment column name.
        party_vote_attr (str): Column containing nonnegative party votes.
        opposition_vote_attr (str): Column containing nonnegative opposition votes.
        geometry (GeoDataFrame | None, optional): Authoritative GeoDataFrame for a graph or
            Partition source. Defaults to None.
        node_id_column (str | None, optional): Geometry column containing the graph's node
            labels. Defaults to None.

    Returns:
        float: The efficiency gap from the party's point of view; positive values favor the party.
    """
    return _expect(
        float,
        _evaluate(
            source,
            assignment,
            EfficiencyGap(party_vote_attr, opposition_vote_attr),
            geometry=geometry,
            node_id_column=node_id_column,
        ),
    )


def simplified_efficiency_gap(
    source: SinglePlanSource,
    assignment: GeoAssignment | None = None,
    *,
    party_vote_attr: str,
    opposition_vote_attr: str,
    geometry: GeoDataFrame | None = None,
    node_id_column: str | None = None,
) -> float:
    r"""Return the equal-turnout seat-vote efficiency-gap formula for one plan.

    Args:
        source (SinglePlanSource): Partition, graph, or authoritative GeoDataFrame containing the
            vote columns.
        assignment (GeoAssignment | None): District assignment. Required for graph and
            GeoDataFrame sources and omitted for a Partition. A GeoDataFrame also accepts an
            assignment column name.
        party_vote_attr (str): Column containing nonnegative party votes.
        opposition_vote_attr (str): Column containing nonnegative opposition votes.
        geometry (GeoDataFrame | None, optional): Authoritative GeoDataFrame for a graph or
            Partition source. Defaults to None.
        node_id_column (str | None, optional): Geometry column containing the graph's node
            labels. Defaults to None.

    Returns:
        float: The seat-vote value :math:`S - 2V + 1/2`.
    """
    return _expect(
        float,
        _evaluate(
            source,
            assignment,
            SimplifiedEfficiencyGap(party_vote_attr, opposition_vote_attr),
            geometry=geometry,
            node_id_column=node_id_column,
        ),
    )


def mean_median(
    source: SinglePlanSource,
    assignment: GeoAssignment | None = None,
    *,
    party_vote_attr: str,
    opposition_vote_attr: str,
    geometry: GeoDataFrame | None = None,
    node_id_column: str | None = None,
) -> float:
    """Return the mean-median partisan-symmetry score for one plan.

    Args:
        source (SinglePlanSource): Partition, graph, or authoritative GeoDataFrame containing the
            vote columns.
        assignment (GeoAssignment | None): District assignment. Required for graph and
            GeoDataFrame sources and omitted for a Partition. A GeoDataFrame also accepts an
            assignment column name.
        party_vote_attr (str): Column containing nonnegative party votes.
        opposition_vote_attr (str): Column containing nonnegative opposition votes.
        geometry (GeoDataFrame | None, optional): Authoritative GeoDataFrame for a graph or
            Partition source. Defaults to None.
        node_id_column (str | None, optional): Geometry column containing the graph's node
            labels. Defaults to None.

    Returns:
        float: Median district vote share minus mean district vote share.
    """
    return _expect(
        float,
        _evaluate(
            source,
            assignment,
            MeanMedian(party_vote_attr, opposition_vote_attr),
            geometry=geometry,
            node_id_column=node_id_column,
        ),
    )


def disproportionality(
    source: SinglePlanSource,
    assignment: GeoAssignment | None = None,
    *,
    party_vote_attr: str,
    opposition_vote_attr: str,
    geometry: GeoDataFrame | None = None,
    node_id_column: str | None = None,
) -> float:
    """Return the party's signed seat-share minus aggregate vote-share gap for one plan.

    Args:
        source (SinglePlanSource): Partition, graph, or authoritative GeoDataFrame containing the
            vote columns.
        assignment (GeoAssignment | None): District assignment. Required for graph and
            GeoDataFrame sources and omitted for a Partition. A GeoDataFrame also accepts an
            assignment column name.
        party_vote_attr (str): Column containing nonnegative party votes.
        opposition_vote_attr (str): Column containing nonnegative opposition votes.
        geometry (GeoDataFrame | None, optional): Authoritative GeoDataFrame for a graph or
            Partition source. Defaults to None.
        node_id_column (str | None, optional): Geometry column containing the graph's node
            labels. Defaults to None.

    Returns:
        float: Party seat share minus aggregate two-party vote share.
    """
    return _expect(
        float,
        _evaluate(
            source,
            assignment,
            Disproportionality(party_vote_attr, opposition_vote_attr),
            geometry=geometry,
            node_id_column=node_id_column,
        ),
    )


def partisan_bias(
    source: SinglePlanSource,
    assignment: GeoAssignment | None = None,
    *,
    party_vote_attr: str,
    opposition_vote_attr: str,
    turnout_model: Literal["equal", "observed"] = "equal",
    geometry: GeoDataFrame | None = None,
    node_id_column: str | None = None,
) -> float:
    """Return partisan bias at 50 percent under uniform partisan swing.

    Args:
        source (SinglePlanSource): Partition, graph, or authoritative GeoDataFrame containing the
            vote columns.
        assignment (GeoAssignment | None): District assignment. Required for graph and
            GeoDataFrame sources and omitted for a Partition. A GeoDataFrame also accepts an
            assignment column name.
        party_vote_attr (str): Column containing nonnegative party votes.
        opposition_vote_attr (str): Column containing nonnegative opposition votes.
        turnout_model (Literal["equal", "observed"], optional): Reference-vote model. ``"equal"``
            weights valid districts equally; ``"observed"`` uses aggregate two-party turnout.
            Defaults to ``"equal"``.
        geometry (GeoDataFrame | None, optional): Authoritative GeoDataFrame for a graph or
            Partition source. Defaults to None.
        node_id_column (str | None, optional): Geometry column containing the graph's node
            labels. Defaults to None.

    Returns:
        float: Party seat share at the modeled 50-percent vote point minus one half.
    """
    return _expect(
        float,
        _evaluate(
            source,
            assignment,
            PartisanBias(party_vote_attr, opposition_vote_attr, turnout_model),
            geometry=geometry,
            node_id_column=node_id_column,
        ),
    )


def partisan_gini(
    source: SinglePlanSource,
    assignment: GeoAssignment | None = None,
    *,
    party_vote_attr: str,
    opposition_vote_attr: str,
    turnout_model: Literal["equal", "observed"] = "equal",
    geometry: GeoDataFrame | None = None,
    node_id_column: str | None = None,
) -> float:
    """Return unsigned partisan Gini under uniform partisan swing.

    Args:
        source (SinglePlanSource): Partition, graph, or authoritative GeoDataFrame containing the
            vote columns.
        assignment (GeoAssignment | None): District assignment. Required for graph and
            GeoDataFrame sources and omitted for a Partition. A GeoDataFrame also accepts an
            assignment column name.
        party_vote_attr (str): Column containing nonnegative party votes.
        opposition_vote_attr (str): Column containing nonnegative opposition votes.
        turnout_model (Literal["equal", "observed"], optional): Reference-vote model. ``"equal"``
            weights valid districts equally; ``"observed"`` uses aggregate two-party turnout.
            Defaults to ``"equal"``.
        geometry (GeoDataFrame | None, optional): Authoritative GeoDataFrame for a graph or
            Partition source. Defaults to None.
        node_id_column (str | None, optional): Geometry column containing the graph's node
            labels. Defaults to None.

    Returns:
        float: Nonnegative area between the seats-votes curve and its partisan reflection.
    """
    return _expect(
        float,
        _evaluate(
            source,
            assignment,
            PartisanGini(party_vote_attr, opposition_vote_attr, turnout_model),
            geometry=geometry,
            node_id_column=node_id_column,
        ),
    )


def population_deviations(
    source: SinglePlanSource,
    assignment: GeoAssignment | None = None,
    *,
    population_attr: str,
    geometry: GeoDataFrame | None = None,
    node_id_column: str | None = None,
) -> Series:
    """Return each district's signed proportional deviation from ideal population.

    Args:
        source (SinglePlanSource): Partition, graph, or authoritative GeoDataFrame containing the
            population column.
        assignment (GeoAssignment | None): District assignment. Required for graph and
            GeoDataFrame sources and omitted for a Partition. A GeoDataFrame also accepts an
            assignment column name.
        population_attr (str): Column containing finite, nonnegative population with positive
            total.
        geometry (GeoDataFrame | None, optional): Authoritative GeoDataFrame for a graph or
            Partition source. Defaults to None.
        node_id_column (str | None, optional): Geometry column containing the graph's node
            labels. Defaults to None.

    Returns:
        Series: District-indexed signed deviations as proportions of ideal population.
    """
    return _expect(
        Series,
        _evaluate(
            source,
            assignment,
            PopulationDeviations(population_attr),
            geometry=geometry,
            node_id_column=node_id_column,
        ),
    )


def max_absolute_population_deviation(
    source: SinglePlanSource,
    assignment: GeoAssignment | None = None,
    *,
    population_attr: str,
    relative_to_ideal: bool = False,
    geometry: GeoDataFrame | None = None,
    node_id_column: str | None = None,
) -> float:
    """Return the largest one-district absolute departure from ideal population.

    Args:
        source (SinglePlanSource): Partition, graph, or authoritative GeoDataFrame containing the
            population column.
        assignment (GeoAssignment | None): District assignment. Required for graph and
            GeoDataFrame sources and omitted for a Partition. A GeoDataFrame also accepts an
            assignment column name.
        population_attr (str): Column containing finite, nonnegative population with positive
            total.
        relative_to_ideal (bool, optional): Return a proportion of ideal population instead of a
            population count. Defaults to False.
        geometry (GeoDataFrame | None, optional): Authoritative GeoDataFrame for a graph or
            Partition source. Defaults to None.
        node_id_column (str | None, optional): Geometry column containing the graph's node
            labels. Defaults to None.

    Returns:
        float: Largest absolute one-district deviation in the requested units.
    """
    return _expect(
        float,
        _evaluate(
            source,
            assignment,
            MaxAbsolutePopulationDeviation(population_attr, relative_to_ideal),
            geometry=geometry,
            node_id_column=node_id_column,
        ),
    )


def max_population_deviation(
    source: SinglePlanSource,
    assignment: GeoAssignment | None = None,
    *,
    population_attr: str,
    relative_to_ideal: bool = False,
    geometry: GeoDataFrame | None = None,
    node_id_column: str | None = None,
) -> float:
    """Return the top-to-bottom population deviation of one plan.

    Args:
        source (SinglePlanSource): Partition, graph, or authoritative GeoDataFrame containing the
            population column.
        assignment (GeoAssignment | None): District assignment. Required for graph and
            GeoDataFrame sources and omitted for a Partition. A GeoDataFrame also accepts an
            assignment column name.
        population_attr (str): Column containing finite, nonnegative population with positive
            total.
        relative_to_ideal (bool, optional): Return a proportion of ideal population instead of a
            population count. Defaults to False.
        geometry (GeoDataFrame | None, optional): Authoritative GeoDataFrame for a graph or
            Partition source. Defaults to None.
        node_id_column (str | None, optional): Geometry column containing the graph's node
            labels. Defaults to None.

    Returns:
        float: Maximum minus minimum district population in the requested units.
    """
    return _expect(
        float,
        _evaluate(
            source,
            assignment,
            MaxPopulationDeviation(population_attr, relative_to_ideal),
            geometry=geometry,
            node_id_column=node_id_column,
        ),
    )


def demographic_shares(
    source: SinglePlanSource,
    assignment: GeoAssignment | None = None,
    *,
    subgroup_population_attr: str,
    total_population_attr: str,
    geometry: GeoDataFrame | None = None,
    node_id_column: str | None = None,
) -> Series:
    """Return a subgroup's share of the specified total in each district.

    Args:
        source (SinglePlanSource): Partition, graph, or authoritative GeoDataFrame containing the
            demographic columns.
        assignment (GeoAssignment | None): District assignment. Required for graph and
            GeoDataFrame sources and omitted for a Partition. A GeoDataFrame also accepts an
            assignment column name.
        subgroup_population_attr (str): Column containing finite, nonnegative subgroup population.
        total_population_attr (str): Column containing finite, nonnegative totals that contain the
            subgroup.
        geometry (GeoDataFrame | None, optional): Authoritative GeoDataFrame for a graph or
            Partition source. Defaults to None.
        node_id_column (str | None, optional): Geometry column containing the graph's node
            labels. Defaults to None.

    Returns:
        Series: District-indexed subgroup shares, with ``NaN`` where a district total is zero.
    """
    return _expect(
        Series,
        _evaluate(
            source,
            assignment,
            DemographicShares(subgroup_population_attr, total_population_attr),
            geometry=geometry,
            node_id_column=node_id_column,
        ),
    )


def districts_above_threshold(
    source: SinglePlanSource,
    assignment: GeoAssignment | None = None,
    *,
    subgroup_population_attr: str,
    total_population_attr: str,
    threshold: float = 0.5,
    geometry: GeoDataFrame | None = None,
    node_id_column: str | None = None,
) -> int:
    """Count districts whose subgroup share is strictly above ``threshold``.

    Args:
        source (SinglePlanSource): Partition, graph, or authoritative GeoDataFrame containing the
            demographic columns.
        assignment (GeoAssignment | None): District assignment. Required for graph and
            GeoDataFrame sources and omitted for a Partition. A GeoDataFrame also accepts an
            assignment column name.
        subgroup_population_attr (str): Column containing finite, nonnegative subgroup population.
        total_population_attr (str): Column containing finite, nonnegative totals that contain the
            subgroup.
        threshold (float, optional): Share threshold in the inclusive range ``[0, 1]``. The
            comparison is strictly greater than. Defaults to 0.5.
        geometry (GeoDataFrame | None, optional): Authoritative GeoDataFrame for a graph or
            Partition source. Defaults to None.
        node_id_column (str | None, optional): Geometry column containing the graph's node
            labels. Defaults to None.

    Returns:
        int: Number of districts whose subgroup share exceeds ``threshold``.
    """
    return _expect(
        int,
        _evaluate(
            source,
            assignment,
            DistrictsAboveThreshold(
                subgroup_population_attr,
                total_population_attr,
                threshold,
            ),
            geometry=geometry,
            node_id_column=node_id_column,
        ),
    )


def competitive_contests(
    source: SinglePlanSource,
    assignment: GeoAssignment | None = None,
    *,
    party_vote_attrs: Sequence[str],
    opposition_vote_attrs: Sequence[str],
    vote_share_margin: float = 0.03,
    geometry: GeoDataFrame | None = None,
    node_id_column: str | None = None,
) -> int:
    """Count supplied election-district contests in an open interval around 50 percent.

    Args:
        source (SinglePlanSource): Partition, graph, or authoritative GeoDataFrame containing the
            election columns.
        assignment (GeoAssignment | None): District assignment. Required for graph and
            GeoDataFrame sources and omitted for a Partition. A GeoDataFrame also accepts an
            assignment column name.
        party_vote_attrs (Sequence[str]): Nonempty sequence of party-vote columns, one per election.
        opposition_vote_attrs (Sequence[str]): Matching sequence of opposition-vote columns.
        vote_share_margin (float, optional): Half-width of the open competitive interval around
            0.5. Must lie in the inclusive range ``[0, 0.5]``. Defaults to 0.03.
        geometry (GeoDataFrame | None, optional): Authoritative GeoDataFrame for a graph or
            Partition source. Defaults to None.
        node_id_column (str | None, optional): Geometry column containing the graph's node
            labels. Defaults to None.

    Returns:
        int: Number of election-district contests inside the competitive interval.
    """
    return _expect(
        int,
        _evaluate(
            source,
            assignment,
            CompetitiveContests(
                _election_columns(party_vote_attrs),
                _election_columns(opposition_vote_attrs),
                vote_share_margin,
            ),
            geometry=geometry,
            node_id_column=node_id_column,
        ),
    )


def party_wins_by_district(
    source: SinglePlanSource,
    assignment: GeoAssignment | None = None,
    *,
    party_vote_attrs: Sequence[str],
    opposition_vote_attrs: Sequence[str],
    geometry: GeoDataFrame | None = None,
    node_id_column: str | None = None,
) -> Series:
    """Count strict party wins in each district across supplied elections.

    Args:
        source (SinglePlanSource): Partition, graph, or authoritative GeoDataFrame containing the
            election columns.
        assignment (GeoAssignment | None): District assignment. Required for graph and
            GeoDataFrame sources and omitted for a Partition. A GeoDataFrame also accepts an
            assignment column name.
        party_vote_attrs (Sequence[str]): Nonempty sequence of party-vote columns, one per election.
        opposition_vote_attrs (Sequence[str]): Matching sequence of opposition-vote columns.
        geometry (GeoDataFrame | None, optional): Authoritative GeoDataFrame for a graph or
            Partition source. Defaults to None.
        node_id_column (str | None, optional): Geometry column containing the graph's node
            labels. Defaults to None.

    Returns:
        Series: District-indexed counts of strict party wins across the supplied elections.
    """
    return _expect(
        Series,
        _evaluate(
            source,
            assignment,
            PartyWinsByDistrict(
                _election_columns(party_vote_attrs),
                _election_columns(opposition_vote_attrs),
            ),
            geometry=geometry,
            node_id_column=node_id_column,
        ),
    )


def swing_districts(
    source: SinglePlanSource,
    assignment: GeoAssignment | None = None,
    *,
    party_vote_attrs: Sequence[str],
    opposition_vote_attrs: Sequence[str],
    geometry: GeoDataFrame | None = None,
    node_id_column: str | None = None,
) -> int:
    """Count districts that are not strict wins for one side in every election.

    Args:
        source (SinglePlanSource): Partition, graph, or authoritative GeoDataFrame containing the
            election columns.
        assignment (GeoAssignment | None): District assignment. Required for graph and
            GeoDataFrame sources and omitted for a Partition. A GeoDataFrame also accepts an
            assignment column name.
        party_vote_attrs (Sequence[str]): Nonempty sequence of party-vote columns, one per election.
        opposition_vote_attrs (Sequence[str]): Matching sequence of opposition-vote columns.
        geometry (GeoDataFrame | None, optional): Authoritative GeoDataFrame for a graph or
            Partition source. Defaults to None.
        node_id_column (str | None, optional): Geometry column containing the graph's node
            labels. Defaults to None.

    Returns:
        int: Number of districts not strictly won by the same side in every election.
    """
    return _expect(
        int,
        _evaluate(
            source,
            assignment,
            SwingDistricts(
                _election_columns(party_vote_attrs),
                _election_columns(opposition_vote_attrs),
            ),
            geometry=geometry,
            node_id_column=node_id_column,
        ),
    )


def party_districts(
    source: SinglePlanSource,
    assignment: GeoAssignment | None = None,
    *,
    party_vote_attrs: Sequence[str],
    opposition_vote_attrs: Sequence[str],
    geometry: GeoDataFrame | None = None,
    node_id_column: str | None = None,
) -> int:
    """Count districts strictly won by the party in every supplied election.

    Args:
        source (SinglePlanSource): Partition, graph, or authoritative GeoDataFrame containing the
            election columns.
        assignment (GeoAssignment | None): District assignment. Required for graph and
            GeoDataFrame sources and omitted for a Partition. A GeoDataFrame also accepts an
            assignment column name.
        party_vote_attrs (Sequence[str]): Nonempty sequence of party-vote columns, one per election.
        opposition_vote_attrs (Sequence[str]): Matching sequence of opposition-vote columns.
        geometry (GeoDataFrame | None, optional): Authoritative GeoDataFrame for a graph or
            Partition source. Defaults to None.
        node_id_column (str | None, optional): Geometry column containing the graph's node
            labels. Defaults to None.

    Returns:
        int: Number of districts strictly won by the party in every supplied election.
    """
    return _expect(
        int,
        _evaluate(
            source,
            assignment,
            PartyDistricts(
                _election_columns(party_vote_attrs),
                _election_columns(opposition_vote_attrs),
            ),
            geometry=geometry,
            node_id_column=node_id_column,
        ),
    )


def opposition_party_districts(
    source: SinglePlanSource,
    assignment: GeoAssignment | None = None,
    *,
    party_vote_attrs: Sequence[str],
    opposition_vote_attrs: Sequence[str],
    geometry: GeoDataFrame | None = None,
    node_id_column: str | None = None,
) -> int:
    """Count districts strictly won by the opposition in every supplied election.

    Args:
        source (SinglePlanSource): Partition, graph, or authoritative GeoDataFrame containing the
            election columns.
        assignment (GeoAssignment | None): District assignment. Required for graph and
            GeoDataFrame sources and omitted for a Partition. A GeoDataFrame also accepts an
            assignment column name.
        party_vote_attrs (Sequence[str]): Nonempty sequence of party-vote columns, one per election.
        opposition_vote_attrs (Sequence[str]): Matching sequence of opposition-vote columns.
        geometry (GeoDataFrame | None, optional): Authoritative GeoDataFrame for a graph or
            Partition source. Defaults to None.
        node_id_column (str | None, optional): Geometry column containing the graph's node
            labels. Defaults to None.

    Returns:
        int: Number of districts strictly won by the opposition in every supplied election.
    """
    return _expect(
        int,
        _evaluate(
            source,
            assignment,
            OppositionPartyDistricts(
                _election_columns(party_vote_attrs),
                _election_columns(opposition_vote_attrs),
            ),
            geometry=geometry,
            node_id_column=node_id_column,
        ),
    )


def aggregate_seats(
    source: SinglePlanSource,
    assignment: GeoAssignment | None = None,
    *,
    party_vote_attrs: Sequence[str],
    opposition_vote_attrs: Sequence[str],
    geometry: GeoDataFrame | None = None,
    node_id_column: str | None = None,
) -> int:
    """Count strict party wins across every supplied election and district.

    Args:
        source (SinglePlanSource): Partition, graph, or authoritative GeoDataFrame containing the
            election columns.
        assignment (GeoAssignment | None): District assignment. Required for graph and
            GeoDataFrame sources and omitted for a Partition. A GeoDataFrame also accepts an
            assignment column name.
        party_vote_attrs (Sequence[str]): Nonempty sequence of party-vote columns, one per election.
        opposition_vote_attrs (Sequence[str]): Matching sequence of opposition-vote columns.
        geometry (GeoDataFrame | None, optional): Authoritative GeoDataFrame for a graph or
            Partition source. Defaults to None.
        node_id_column (str | None, optional): Geometry column containing the graph's node
            labels. Defaults to None.

    Returns:
        int: Number of strict party wins across all supplied election-district contests.
    """
    return _expect(
        int,
        _evaluate(
            source,
            assignment,
            AggregateSeats(
                _election_columns(party_vote_attrs),
                _election_columns(opposition_vote_attrs),
            ),
            geometry=geometry,
            node_id_column=node_id_column,
        ),
    )


def mean_signed_seat_vote_gap(
    source: SinglePlanSource,
    assignment: GeoAssignment | None = None,
    *,
    party_vote_attrs: Sequence[str],
    opposition_vote_attrs: Sequence[str],
    geometry: GeoDataFrame | None = None,
    node_id_column: str | None = None,
) -> float:
    """Average signed seat-share minus vote-share gaps over supplied elections.

    Args:
        source (SinglePlanSource): Partition, graph, or authoritative GeoDataFrame containing the
            election columns.
        assignment (GeoAssignment | None): District assignment. Required for graph and
            GeoDataFrame sources and omitted for a Partition. A GeoDataFrame also accepts an
            assignment column name.
        party_vote_attrs (Sequence[str]): Nonempty sequence of party-vote columns, one per election.
        opposition_vote_attrs (Sequence[str]): Matching sequence of opposition-vote columns.
        geometry (GeoDataFrame | None, optional): Authoritative GeoDataFrame for a graph or
            Partition source. Defaults to None.
        node_id_column (str | None, optional): Geometry column containing the graph's node
            labels. Defaults to None.

    Returns:
        float: Mean party seat-share minus aggregate vote-share gap across elections.
    """
    return _expect(
        float,
        _evaluate(
            source,
            assignment,
            MeanSignedSeatVoteGap(
                _election_columns(party_vote_attrs),
                _election_columns(opposition_vote_attrs),
            ),
            geometry=geometry,
            node_id_column=node_id_column,
        ),
    )


def mean_absolute_seat_vote_gap(
    source: SinglePlanSource,
    assignment: GeoAssignment | None = None,
    *,
    party_vote_attrs: Sequence[str],
    opposition_vote_attrs: Sequence[str],
    geometry: GeoDataFrame | None = None,
    node_id_column: str | None = None,
) -> float:
    """Average absolute seat-share minus vote-share gaps over supplied elections.

    Args:
        source (SinglePlanSource): Partition, graph, or authoritative GeoDataFrame containing the
            election columns.
        assignment (GeoAssignment | None): District assignment. Required for graph and
            GeoDataFrame sources and omitted for a Partition. A GeoDataFrame also accepts an
            assignment column name.
        party_vote_attrs (Sequence[str]): Nonempty sequence of party-vote columns, one per election.
        opposition_vote_attrs (Sequence[str]): Matching sequence of opposition-vote columns.
        geometry (GeoDataFrame | None, optional): Authoritative GeoDataFrame for a graph or
            Partition source. Defaults to None.
        node_id_column (str | None, optional): Geometry column containing the graph's node
            labels. Defaults to None.

    Returns:
        float: Mean absolute seat-share minus aggregate vote-share gap across elections.
    """
    return _expect(
        float,
        _evaluate(
            source,
            assignment,
            MeanAbsoluteSeatVoteGap(
                _election_columns(party_vote_attrs),
                _election_columns(opposition_vote_attrs),
            ),
            geometry=geometry,
            node_id_column=node_id_column,
        ),
    )
