import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Literal, cast

import geopandas as gpd
import networkx as nx
import numpy as np
import pyarrow.parquet as pq
import pytest
from binary_ensemble import BenEncoder
from gerrychain import Partition
from pandas import Series
from shapely.geometry import box

import gerrytools.scoring as scoring
from gerrytools.scoring import formulas

Variant = Literal["standard", "mkv_chain", "twodelta"]
MetricCase = tuple[str, scoring.Metric, dict[str, object], Callable[..., object]]


def resources() -> tuple[nx.Graph, gpd.GeoDataFrame, list[str], Partition]:
    graph = nx.path_graph(6)
    frame = gpd.GeoDataFrame(
        {
            "party_1": [60, 10, 10, 20, 35, 15],
            "opposition_1": [40, 20, 30, 10, 15, 35],
            "party_2": [45, 25, 30, 10, 20, 35],
            "opposition_2": [35, 15, 10, 30, 30, 15],
            "population": [30, 20, 25, 25, 40, 10],
            "subgroup": [20, 15, 5, 20, 10, 5],
            "district": ["west", "west", "center", "center", "east", "east"],
        },
        geometry=[box(index, 0, index + 1, 1) for index in range(6)],
        crs="EPSG:3857",
    )
    assignment = list(frame["district"])
    partition = Partition(graph, dict(enumerate(assignment)))
    return graph, frame, assignment, partition


def district_tallies(
    frame: gpd.GeoDataFrame, assignment: Sequence[object], columns: Sequence[str]
) -> list[np.ndarray]:
    labels = tuple(dict.fromkeys(assignment))
    return [
        np.asarray(
            [
                frame.loc[
                    [district == label for district in assignment],
                    column,
                ].sum()
                for label in labels
            ],
            dtype=np.float64,
        )
        for column in columns
    ]


def metric_cases() -> list[MetricCase]:
    party: dict[str, object] = {
        "party_vote_attr": "party_1",
        "opposition_vote_attr": "opposition_1",
    }
    elections: dict[str, object] = {
        "party_vote_attrs": ("party_1", "party_2"),
        "opposition_vote_attrs": ("opposition_1", "opposition_2"),
    }
    return [
        (
            "district_vote_shares",
            scoring.DistrictVoteShares("party_1", "opposition_1"),
            party,
            formulas.district_vote_shares,
        ),
        (
            "district_wins",
            scoring.DistrictWins("party_1", "opposition_1"),
            party,
            formulas.district_wins,
        ),
        ("seats", scoring.Seats("party_1", "opposition_1"), party, formulas.seats),
        (
            "overall_vote_share",
            scoring.OverallVoteShare("party_1", "opposition_1"),
            party,
            formulas.overall_vote_share,
        ),
        (
            "disproportionality",
            scoring.Disproportionality("party_1", "opposition_1"),
            party,
            formulas.disproportionality,
        ),
        (
            "efficiency_gap",
            scoring.EfficiencyGap("party_1", "opposition_1"),
            party,
            formulas.efficiency_gap,
        ),
        (
            "simplified_efficiency_gap",
            scoring.SimplifiedEfficiencyGap("party_1", "opposition_1"),
            party,
            formulas.simplified_efficiency_gap,
        ),
        (
            "mean_median",
            scoring.MeanMedian("party_1", "opposition_1"),
            party,
            formulas.mean_median,
        ),
        (
            "partisan_bias",
            scoring.PartisanBias("party_1", "opposition_1", "observed"),
            {**party, "turnout_model": "observed"},
            formulas.partisan_bias,
        ),
        (
            "partisan_gini",
            scoring.PartisanGini("party_1", "opposition_1", "observed"),
            {**party, "turnout_model": "observed"},
            formulas.partisan_gini,
        ),
        (
            "population_deviations",
            scoring.PopulationDeviations("population"),
            {"population_attr": "population"},
            formulas.population_deviations,
        ),
        (
            "max_absolute_population_deviation",
            scoring.MaxAbsolutePopulationDeviation("population", True),
            {"population_attr": "population", "relative_to_ideal": True},
            formulas.max_absolute_population_deviation,
        ),
        (
            "max_population_deviation",
            scoring.MaxPopulationDeviation("population", True),
            {"population_attr": "population", "relative_to_ideal": True},
            formulas.max_population_deviation,
        ),
        (
            "demographic_shares",
            scoring.DemographicShares("subgroup", "population"),
            {
                "subgroup_population_attr": "subgroup",
                "total_population_attr": "population",
            },
            formulas.demographic_shares,
        ),
        (
            "districts_above_threshold",
            scoring.DistrictsAboveThreshold("subgroup", "population", 0.4),
            {
                "subgroup_population_attr": "subgroup",
                "total_population_attr": "population",
                "threshold": 0.4,
            },
            formulas.districts_above_threshold,
        ),
        (
            "competitive_contests",
            scoring.CompetitiveContests(
                ("party_1", "party_2"),
                ("opposition_1", "opposition_2"),
                0.1,
            ),
            {**elections, "vote_share_margin": 0.1},
            formulas.competitive_contests,
        ),
        (
            "party_wins_by_district",
            scoring.PartyWinsByDistrict(
                ("party_1", "party_2"),
                ("opposition_1", "opposition_2"),
            ),
            elections,
            formulas.party_wins_by_district,
        ),
        (
            "swing_districts",
            scoring.SwingDistricts(
                ("party_1", "party_2"),
                ("opposition_1", "opposition_2"),
            ),
            elections,
            formulas.swing_districts,
        ),
        (
            "party_districts",
            scoring.PartyDistricts(
                ("party_1", "party_2"),
                ("opposition_1", "opposition_2"),
            ),
            elections,
            formulas.party_districts,
        ),
        (
            "opposition_party_districts",
            scoring.OppositionPartyDistricts(
                ("party_1", "party_2"),
                ("opposition_1", "opposition_2"),
            ),
            elections,
            formulas.opposition_party_districts,
        ),
        (
            "aggregate_seats",
            scoring.AggregateSeats(
                ("party_1", "party_2"),
                ("opposition_1", "opposition_2"),
            ),
            elections,
            formulas.aggregate_seats,
        ),
        (
            "mean_signed_seat_vote_gap",
            scoring.MeanSignedSeatVoteGap(
                ("party_1", "party_2"),
                ("opposition_1", "opposition_2"),
            ),
            elections,
            formulas.mean_signed_seat_vote_gap,
        ),
        (
            "mean_absolute_seat_vote_gap",
            scoring.MeanAbsoluteSeatVoteGap(
                ("party_1", "party_2"),
                ("opposition_1", "opposition_2"),
            ),
            elections,
            formulas.mean_absolute_seat_vote_gap,
        ),
    ]


def formula_result(
    frame: gpd.GeoDataFrame,
    assignment: Sequence[object],
    name: str,
    options: dict[str, object],
    formula: Callable[..., object],
) -> object:
    if name in {
        "population_deviations",
        "max_absolute_population_deviation",
        "max_population_deviation",
    }:
        (population,) = district_tallies(
            frame,
            assignment,
            [str(options["population_attr"])],
        )
        kwargs = (
            {"relative_to_ideal": options["relative_to_ideal"]}
            if "relative_to_ideal" in options
            else {}
        )
        return formula(population, **kwargs)
    if name in {"demographic_shares", "districts_above_threshold"}:
        subgroup, total = district_tallies(
            frame,
            assignment,
            [
                str(options["subgroup_population_attr"]),
                str(options["total_population_attr"]),
            ],
        )
        kwargs = {"threshold": options["threshold"]} if "threshold" in options else {}
        return formula(subgroup, total, **kwargs)

    party_key = "party_vote_attr" if "party_vote_attr" in options else "party_vote_attrs"
    opposition_key = (
        "opposition_vote_attr" if "opposition_vote_attr" in options else "opposition_vote_attrs"
    )
    party_columns = options[party_key]
    opposition_columns = options[opposition_key]
    if isinstance(party_columns, str):
        party, opposition = district_tallies(
            frame, assignment, [party_columns, str(opposition_columns)]
        )
    else:
        party_rows = district_tallies(
            frame,
            assignment,
            list(cast(Sequence[str], party_columns)),
        )
        opposition_rows = district_tallies(
            frame,
            assignment,
            list(cast(Sequence[str], opposition_columns)),
        )
        party = np.stack(party_rows)
        opposition = np.stack(opposition_rows)
    kwargs = {key: options[key] for key in ("turnout_model", "vote_share_margin") if key in options}
    return formula(party, opposition, **kwargs)


def assert_same(actual: object, expected: object) -> None:
    values = actual.to_numpy() if isinstance(actual, Series) else actual
    if np.asarray(expected).dtype == np.bool_:
        np.testing.assert_array_equal(values, expected)
    else:
        np.testing.assert_allclose(
            np.asarray(values),
            np.asarray(expected),
            equal_nan=True,
            rtol=1e-13,
            atol=1e-13,
        )


@pytest.mark.parametrize(
    ("name", "metric", "options", "formula"),
    metric_cases(),
    ids=[case[0] for case in metric_cases()],
)
def test_single_plan_and_prepared_metrics_match_array_formulas(
    name: str,
    metric: scoring.Metric,
    options: dict[str, object],
    formula: Callable[..., object],
) -> None:
    graph, frame, assignment, partition = resources()
    expected = formula_result(frame, assignment, name, options, formula)
    prepared = (
        scoring.PlanEvaluator(graph, geometry=frame).add_metric(metric).evaluate(assignment)[name]
    )
    gdf_direct = getattr(scoring, name)(frame, "district", **options)
    partition_direct = getattr(scoring, name)(partition, geometry=frame, **options)

    assert_same(prepared, expected)
    assert_same(gdf_direct, expected)
    assert_same(partition_direct, expected)


def test_generated_native_metrics_match_formulas_under_relabeling_and_edge_cases() -> None:
    rng = np.random.default_rng(20_260_728)
    node_count = 60
    graph = nx.path_graph(node_count)
    population = rng.integers(0, 1_000, node_count)
    population[0] = 1
    frame = gpd.GeoDataFrame(
        {
            "party_1": rng.integers(0, 1_000, node_count),
            "opposition_1": rng.integers(0, 1_000, node_count),
            "party_2": rng.integers(0, 1_000, node_count),
            "opposition_2": rng.integers(0, 1_000, node_count),
            "population": population,
            "subgroup": rng.integers(0, population + 1),
        },
        geometry=[box(index, 0, index + 1, 1) for index in range(node_count)],
        crs="EPSG:3857",
    )
    cases = metric_cases()
    evaluator = scoring.PlanEvaluator(graph, geometry=frame)
    for _, metric, _, _ in cases:
        evaluator.add_metric(metric)

    for _ in range(50):
        dense = np.concatenate((np.arange(6), rng.integers(0, 6, node_count - 6)))
        rng.shuffle(dense)
        relabeling = rng.permutation([f"district-{index}" for index in range(6)])
        assignment = [str(relabeling[district]) for district in dense]
        result = evaluator.evaluate(assignment)
        for name, _, options, formula in cases:
            assert_same(
                result[name],
                formula_result(frame, assignment, name, options, formula),
            )


def test_zero_turnout_ties_and_both_turnout_models_match_formulas() -> None:
    graph = nx.path_graph(4)
    frame = gpd.GeoDataFrame(
        {
            "party": [0, 50, 90, 1],
            "opposition": [0, 50, 10, 9],
        },
        geometry=[box(index, 0, index + 1, 1) for index in range(4)],
        crs="EPSG:3857",
    )
    assignment = [0, 1, 2, 2]
    party, opposition = district_tallies(frame, assignment, ["party", "opposition"])

    for turnout_model in ("equal", "observed"):
        evaluator = (
            scoring.PlanEvaluator(graph, geometry=frame)
            .add_metric(scoring.PartisanBias("party", "opposition", turnout_model))
            .add_metric(scoring.PartisanGini("party", "opposition", turnout_model))
        )
        result = evaluator.evaluate(assignment)
        assert_same(
            result["partisan_bias"],
            formulas.partisan_bias(
                party,
                opposition,
                turnout_model=turnout_model,
            ),
        )
        assert_same(
            result["partisan_gini"],
            formulas.partisan_gini(
                party,
                opposition,
                turnout_model=turnout_model,
            ),
        )


@pytest.mark.parametrize(
    ("party", "opposition"),
    [
        ([0, 50, 90, 1], [0, 50, 10, 9]),
        ([0, 0, 0, 0], [0, 0, 0, 0]),
    ],
    ids=["zero_turnout_district", "all_zero_plan"],
)
def test_native_vote_share_metrics_match_formula_nan_propagation(
    party: list[int], opposition: list[int]
) -> None:
    graph = nx.path_graph(4)
    frame = gpd.GeoDataFrame(
        {"party": party, "opposition": opposition},
        geometry=[box(index, 0, index + 1, 1) for index in range(4)],
        crs="EPSG:3857",
    )
    assignment = [0, 1, 2, 2]
    party_tallies, opposition_tallies = district_tallies(frame, assignment, ["party", "opposition"])
    evaluator = (
        scoring.PlanEvaluator(graph, geometry=frame)
        .add_metric(scoring.DistrictVoteShares("party", "opposition"))
        .add_metric(scoring.MeanMedian("party", "opposition"))
        .add_metric(scoring.EfficiencyGap("party", "opposition"))
        .add_metric(scoring.SimplifiedEfficiencyGap("party", "opposition"))
        .add_metric(scoring.OverallVoteShare("party", "opposition"))
        .add_metric(scoring.Disproportionality("party", "opposition"))
    )

    result = evaluator.evaluate(assignment)

    shares = formulas.district_vote_shares(party_tallies, opposition_tallies)
    assert np.isnan(shares[0])  # district 0 always has zero turnout in these scenarios
    assert_same(result["district_vote_shares"], shares)
    assert np.isnan(result["mean_median"])  # any zero-turnout district poisons the mean and median
    assert_same(result["mean_median"], formulas.mean_median(party_tallies, opposition_tallies))
    assert_same(
        result["efficiency_gap"],
        formulas.efficiency_gap(party_tallies, opposition_tallies),
    )
    assert_same(
        result["simplified_efficiency_gap"],
        formulas.simplified_efficiency_gap(party_tallies, opposition_tallies),
    )
    assert_same(
        result["overall_vote_share"],
        formulas.overall_vote_share(party_tallies, opposition_tallies),
    )
    assert_same(
        result["disproportionality"],
        formulas.disproportionality(party_tallies, opposition_tallies),
    )
    if party_tallies.sum() + opposition_tallies.sum() == 0:
        # A plan with no two-party votes anywhere has no defined plan-level score at all.
        for name in (
            "efficiency_gap",
            "simplified_efficiency_gap",
            "overall_vote_share",
            "disproportionality",
        ):
            assert np.isnan(result[name])


def test_native_cross_election_mean_gaps_are_nan_for_a_zero_turnout_election() -> None:
    graph = nx.path_graph(4)
    frame = gpd.GeoDataFrame(
        {
            "party_1": [60, 0, 30, 10],
            "opposition_1": [40, 10, 20, 20],
            "party_2": [0, 0, 0, 0],
            "opposition_2": [0, 0, 0, 0],
        },
        geometry=[box(index, 0, index + 1, 1) for index in range(4)],
        crs="EPSG:3857",
    )
    evaluator = (
        scoring.PlanEvaluator(graph, geometry=frame)
        .add_metric(
            scoring.MeanSignedSeatVoteGap(
                ("party_1", "party_2"),
                ("opposition_1", "opposition_2"),
            )
        )
        .add_metric(
            scoring.MeanAbsoluteSeatVoteGap(
                ("party_1", "party_2"),
                ("opposition_1", "opposition_2"),
            )
        )
    )

    result = evaluator.evaluate([0, 0, 1, 1])

    assert np.isnan(result["mean_signed_seat_vote_gap"])
    assert np.isnan(result["mean_absolute_seat_vote_gap"])


def test_derived_metrics_return_natural_scalar_and_district_types() -> None:
    _, frame, _, _ = resources()

    assert isinstance(
        scoring.seats(
            frame,
            "district",
            party_vote_attr="party_1",
            opposition_vote_attr="opposition_1",
        ),
        int,
    )
    assert isinstance(
        scoring.overall_vote_share(
            frame,
            "district",
            party_vote_attr="party_1",
            opposition_vote_attr="opposition_1",
        ),
        float,
    )
    assert (
        scoring.district_wins(
            frame,
            "district",
            party_vote_attr="party_1",
            opposition_vote_attr="opposition_1",
        ).dtype
        == np.bool_
    )
    assert (
        scoring.party_wins_by_district(
            frame,
            "district",
            party_vote_attrs=("party_1", "party_2"),
            opposition_vote_attrs=("opposition_1", "opposition_2"),
        ).dtype
        == np.int64
    )


@pytest.mark.parametrize("variant", ["standard", "mkv_chain", "twodelta"])
def test_every_derived_metric_streams_like_evaluate_many(tmp_path: Path, variant: Variant) -> None:
    source = tmp_path / f"plans-{variant}.ben"
    output = tmp_path / "scores"
    plans = [[0, 0, 1, 1, 2, 2], [0, 1, 0, 1, 2, 2]]
    with BenEncoder(source, variant=variant) as stream:
        for assignment in plans:
            stream.write(assignment)

    graph, frame, _, _ = resources()
    evaluator = scoring.PlanEvaluator(graph, geometry=frame)
    for _, metric, _, _ in metric_cases():
        evaluator.add_metric(metric)
    expected = evaluator.evaluate_many(plans)

    evaluator.evaluate_stream(source, output, batch_size=1)

    manifest = json.loads((output / "manifest__scores.json").read_text())
    cases = metric_cases()
    assert [metric["instance"] for metric in manifest["metrics"]] == [case[0] for case in cases]
    for description, (name, _, options, _) in zip(manifest["metrics"], cases, strict=True):
        assert description["kind"] == name
        assert description["options"] == json.loads(json.dumps(options))
        assert description["subkeys"] == ["score"]
        assert description["shape"] in {"district", "plan"}
        actual = pq.read_table(output / f"{name}__scores.parquet").to_pydict()
        expected_values = expected.array(name)
        if expected_values.ndim == 2:
            np.testing.assert_allclose(
                actual["score"],
                expected_values[:, 0],
                equal_nan=True,
            )
        else:
            for district in range(expected_values.shape[-1]):
                np.testing.assert_allclose(
                    actual[f"score__district_{district}"],
                    expected_values[:, 0, district],
                    equal_nan=True,
                )


@pytest.mark.parametrize(
    "metric",
    [
        scoring.DistrictsAboveThreshold("subgroup", "population", np.float32(0.4)),
        scoring.CompetitiveContests(("party_1",), ("opposition_1",), np.float32(0.1)),
    ],
)
def test_numpy_scalar_metric_options_are_json_serializable(metric) -> None:
    assert isinstance(json.dumps(metric._options()), str)


@pytest.mark.parametrize(
    "constructor,message",
    [
        (lambda: scoring.Seats("", "opposition"), "nonempty column"),
        (
            lambda: scoring.PartisanBias(
                "party",
                "opposition",
                cast("Literal['equal', 'observed']", "weighted"),
            ),
            "turnout_model",
        ),
        (
            lambda: scoring.MaxPopulationDeviation(
                "population",
                relative_to_ideal=cast(bool, 1),
            ),
            "relative_to_ideal",
        ),
        (
            lambda: scoring.DistrictsAboveThreshold("subgroup", "total", 1.1),
            "threshold",
        ),
        (
            lambda: scoring.CompetitiveContests(("party",), ("opposition",), 0.6),
            "vote_share_margin",
        ),
        (
            lambda: scoring.PartyDistricts((), ()),
            "cannot be empty",
        ),
        (
            lambda: scoring.PartyDistricts(("party",), ("one", "two")),
            "equal length",
        ),
        (
            lambda: scoring.PartyDistricts(
                cast("tuple[str, ...]", "party"),
                cast("tuple[str, ...]", "opposition"),
            ),
            "sequence of column names",
        ),
    ],
)
def test_metric_descriptions_reject_invalid_options(
    constructor: Callable[[], object], message: str
) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        constructor()


def test_registration_rejects_invalid_derived_source_values() -> None:
    graph, frame, assignment, _ = resources()

    negative = frame.copy()
    negative.loc[0, "party_1"] = -1
    with pytest.raises(ValueError, match="cannot contain negative"):
        scoring.PlanEvaluator(graph, geometry=negative).add_metric(
            scoring.Seats("party_1", "opposition_1")
        ).evaluate(assignment)

    empty_population = frame.copy()
    empty_population["population"] = 0
    with pytest.raises(ValueError, match="positive total"):
        scoring.PlanEvaluator(graph, geometry=empty_population).add_metric(
            scoring.PopulationDeviations("population")
        ).evaluate(assignment)

    invalid_subgroup = frame.copy()
    invalid_subgroup.loc[0, "subgroup"] = invalid_subgroup.loc[0, "population"] + 1
    with pytest.raises(ValueError, match="subgroup cannot exceed total"):
        scoring.PlanEvaluator(graph, geometry=invalid_subgroup).add_metric(
            scoring.DemographicShares("subgroup", "population")
        ).evaluate(assignment)


def test_plan_names_are_distinct_from_array_formulas_and_old_alias_is_removed() -> None:
    for name, _, _, _ in metric_cases():
        assert getattr(scoring, name) is not getattr(formulas, name)
    assert not hasattr(scoring, "top_to_bottom_population_deviation")
