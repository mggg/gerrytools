from collections.abc import Hashable, Iterable
from typing import cast, get_args

import networkx as nx
import numpy as np
import pandas as pd
import pytest
from gerrychain import Partition

import gerrytools.scoring as scoring
from gerrytools import _scoring_engine
from gerrytools.scoring import (
    CutEdges,
    EvaluationSummary,
    ManyPlanEvalResult,
    Metric,
    PlanEvalResult,
    PlanEvaluator,
    RegionParts,
    RegionPieces,
    RegionSplits,
    Reock,
    Tally,
    TallyByRegion,
)
from gerrytools.scoring.metrics._base import _ResourceSpec


def test_public_api_exposes_the_evaluator_without_legacy_factories() -> None:
    # The compiled backend must publish its build flavor for debugging bug reports.
    assert isinstance(_scoring_engine.DEBUG_ASSERTIONS, bool)
    assert scoring.PlanEvaluator is PlanEvaluator
    assert not hasattr(scoring, "PlanScorer")
    assert hasattr(scoring, "PlanEvalResult")
    assert hasattr(scoring, "ManyPlanEvalResult")
    assert hasattr(scoring, "EnsembleEvalResult")
    assert not hasattr(scoring, "EvaluationRun")
    assert "EvaluationRun" not in scoring.__all__
    assert scoring.EvaluationSummary is EvaluationSummary
    assert not hasattr(scoring, "RunSummary")
    assert not hasattr(scoring, "PlanEvaluation")
    assert not hasattr(scoring, "EnsembleEvaluation")
    assert not hasattr(scoring, "EvaluationResult")
    assert not hasattr(scoring, "DistrictTable")
    assert not hasattr(scoring, "PlanTable")
    assert not hasattr(scoring, "ScoreResult")
    assert not hasattr(scoring, "RegionTally")
    for internal_name in (
        "Dtype",
        "EvaluationValue",
        "MetricResult",
        "ResultShape",
        "RunMetric",
    ):
        assert not hasattr(scoring, internal_name)
        assert internal_name not in scoring.__all__
    assert scoring.TallyByRegion is TallyByRegion
    assert hasattr(scoring, "Eguia")
    assert hasattr(scoring, "formulas")
    assert not hasattr(scoring, "derived")
    assert not hasattr(PlanEvaluator, "add")
    assert hasattr(PlanEvaluator, "add_geometry")
    assert hasattr(PlanEvaluator, "add_metric")
    assert hasattr(PlanEvaluator, "add_metrics")
    assert hasattr(PlanEvaluator, "to_updaters")
    assert hasattr(PlanEvaluator, "evaluate")
    assert hasattr(PlanEvaluator, "evaluate_many")
    assert hasattr(PlanEvaluator, "evaluate_stream")
    assert not hasattr(PlanEvaluator, "score")
    assert not hasattr(PlanEvaluator, "score_many")
    assert not hasattr(PlanEvaluator, "score_run")
    assert not hasattr(PlanEvaluator, "add_units")
    for name in (
        "convex_hull_ratio",
        "cut_edges",
        "eguia",
        "polsby_popper",
        "population_polygon",
        "region_pieces",
        "region_parts",
        "region_splits",
        "reock",
        "state_clipped_convex_hull_ratio",
        "tally",
        "tally_by_region",
    ):
        assert callable(getattr(scoring, name))
    for name in (
        "summarize",
        "summarize_many",
        "contiguous",
        "demographic_updaters",
        "pop_polygon",
        "responsive_proportionality",
        "stable_proportionality",
        "unassigned_units",
    ):
        assert not hasattr(scoring, name)


def test_many_plan_result_rejects_an_empty_metric_mapping() -> None:
    with pytest.raises(ValueError, match="at least one metric"):
        ManyPlanEvalResult({})


def graph() -> nx.Graph:
    graph = nx.Graph()
    graph.add_node("a", population=10, voting_age=8)
    graph.add_node("b", population=20, voting_age=15)
    graph.add_node("c", population=30, voting_age=20)
    graph.add_edge("a", "b")
    graph.add_edge("b", "c")
    return graph


def test_resource_spec_union_and_containment() -> None:
    prepared = _ResourceSpec(
        node_columns=frozenset((("graph", "population"),)),
        topology=True,
    )
    added = _ResourceSpec(
        node_columns=frozenset((("graph", "voting_age"),)),
        edge_columns=frozenset(("weight",)),
    )

    combined = prepared | added

    assert combined.contains(prepared)
    assert combined.contains(added)
    assert not prepared.contains(added)


def test_construction_prepares_no_metric_or_partition_resources() -> None:
    evaluator = PlanEvaluator(graph())

    assert evaluator._resources is None
    assert evaluator._engine is None
    assert evaluator._edge_labels is None


def test_node_id_column_requires_geometry() -> None:
    with pytest.raises(ValueError, match="node_id_column and target_crs require geometry"):
        PlanEvaluator(graph(), node_id_column="node")


def test_target_crs_requires_geometry() -> None:
    with pytest.raises(ValueError, match="node_id_column and target_crs require geometry"):
        PlanEvaluator(graph(), target_crs="EPSG:3857")


def test_first_evaluation_prepares_once_and_reuses_both_cache_layers() -> None:
    source = graph()
    nx.set_node_attributes(source, {"a": "A", "b": "A", "c": "B"}, "region")
    evaluator = (
        PlanEvaluator(source).add_metric(RegionPieces("region")).add_metric(RegionSplits("region"))
    )

    evaluator.evaluate([0, 0, 1])
    resources = evaluator._resources
    engine = evaluator._engine

    assert resources is not None
    assert resources.spec.region_columns == frozenset((("graph", "region"),))
    assert len(resources.region_columns) == 1
    assert resources.topology is None
    assert evaluator._edge_labels is None
    evaluator.evaluate([0, 1, 1])
    evaluator.evaluate_many([[0, 0, 1], [0, 1, 1]])
    assert evaluator._resources is resources
    assert evaluator._engine is engine


def test_to_updaters_evaluates_registered_metrics_once_per_partition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evaluator = PlanEvaluator(graph()).add_metrics(
        Tally("population", result_name="population_total"),
        Tally("voting_age", result_name="voting_age_total"),
    )
    original_evaluate = evaluator.evaluate
    calls = 0

    def counted(partition: Partition) -> PlanEvalResult:
        nonlocal calls
        calls += 1
        return original_evaluate(partition)

    monkeypatch.setattr(evaluator, "evaluate", counted)
    updaters = evaluator.to_updaters()
    partition = Partition(
        graph(),
        {"a": 0, "b": 0, "c": 1},
        updaters=updaters,
    )

    population = partition["population_total"]
    voting_age = partition["voting_age_total"]
    assert isinstance(population, pd.Series)
    assert isinstance(voting_age, pd.Series)
    assert population.to_dict() == {0: 30, 1: 30}
    assert voting_age.to_dict() == {0: 23, 1: 20}
    assert partition["population_total"] is population
    assert calls == 1

    node = partition.graph.internal_node_id_for_original_nx_node_id("b")
    child = partition.flip({node: 1})
    assert child["population_total"].to_dict() == {0: 10, 1: 50}
    assert child["voting_age_total"].to_dict() == {0: 8, 1: 35}
    assert calls == 2


def test_to_updaters_returns_nothing_without_metrics() -> None:
    assert PlanEvaluator(graph()).to_updaters() == {}


def test_metric_addition_reuses_or_extends_published_resources() -> None:
    source = graph()
    evaluator = PlanEvaluator(source).add_metric(Tally("population"))
    evaluator.evaluate([0, 0, 1])
    first = evaluator._resources
    first_engine = evaluator._engine
    assert first is not None
    population = first.node_columns[("graph", "population")]

    source.nodes["a"]["population"] = "not numeric"
    evaluator.add_metric(Tally("population", result_name="population_again")).evaluate([0, 0, 1])
    assert evaluator._resources is first
    assert evaluator._engine is not first_engine

    evaluator.add_metric(Tally("voting_age")).evaluate([0, 0, 1])
    extended = evaluator._resources
    assert extended is not None and extended is not first
    assert extended.node_columns[("graph", "population")] is population
    assert ("graph", "voting_age") in extended.node_columns


def test_failed_resource_extension_leaves_the_published_snapshot_unchanged() -> None:
    evaluator = PlanEvaluator(graph()).add_metric(Tally("population"))
    evaluator.evaluate([0, 0, 1])
    resources = evaluator._resources
    assert resources is not None

    evaluator.add_metrics(CutEdges(result_name="new_edges"), Tally("missing"))

    with pytest.raises(ValueError, match="missing"):
        evaluator.evaluate([0, 0, 1])
    assert evaluator._resources is resources
    assert resources.topology is None
    assert evaluator._engine is None
    assert evaluator.metrics == ("population", "new_edges", "missing")


def test_partition_verifier_is_plan_driven_and_independent_of_metric_resources() -> None:
    source = graph()
    nx.set_node_attributes(source, {"a": "A", "b": "A", "c": "B"}, "region")
    partition = Partition(source, {"a": 0, "b": 0, "c": 1})
    evaluator = PlanEvaluator(source).add_metric(RegionPieces("region"))

    evaluator.evaluate([0, 0, 1])
    resources = evaluator._resources
    engine = evaluator._engine
    assert evaluator._edge_labels is None

    evaluator.evaluate_many([[0, 0, 1], partition])

    assert evaluator._edge_labels is not None
    assert evaluator._resources is resources
    assert evaluator._engine is engine


def test_unweighted_topology_metric_prepares_no_edge_attribute_columns() -> None:
    evaluator = PlanEvaluator(graph()).add_metric(CutEdges())

    evaluator.evaluate([0, 0, 1])

    assert evaluator._resources is not None
    assert evaluator._resources.topology is not None
    assert evaluator._resources.edge_columns == {}


def test_tally_scores_arbitrary_district_labels_and_returns_labeled_table() -> None:
    scorer = PlanEvaluator(graph()).add_metric(Tally("population", "voting_age"))

    result = scorer.evaluate({"a": "west", "b": "west", "c": "east"})
    table = result["tally"]

    assert isinstance(result, PlanEvalResult)
    assert result.metrics == ("tally",)
    assert tuple(result) == ("tally",)
    assert isinstance(table, pd.DataFrame)
    assert table.index.names == ["district"]
    assert table.columns.names == ["metric"]
    assert table.index.tolist() == ["west", "east"]
    assert table.columns.tolist() == ["population", "voting_age"]
    np.testing.assert_allclose(table, [[30, 23], [30, 20]])
    raw = result.array("tally")
    np.testing.assert_allclose(raw, [[30, 30], [23, 20]])
    assert not raw.flags.writeable
    with pytest.raises(ValueError, match="WRITEABLE"):
        raw.setflags(write=True)
    with pytest.raises(ValueError, match="read-only"):
        raw[0, 0] = 0
    with pytest.raises(KeyError, match="'tally'"):
        result["missing"]


def test_getitem_results_are_writable_whatever_the_metrics_shape() -> None:
    evaluator = PlanEvaluator(graph()).add_metrics(
        Tally("population"),
        Tally("population", "voting_age", result_name="both"),
        CutEdges(),
    )
    plan = [1, 1, 2]

    single = evaluator.evaluate(plan)["population"]
    multi = evaluator.evaluate(plan)["both"]
    assert isinstance(single, pd.Series) and isinstance(multi, pd.DataFrame)
    single.iloc[0] = 99.0
    multi.iloc[0, 0] = 99.0
    assert single.iloc[0] == 99.0 and multi.iloc[0, 0] == 99.0

    many = evaluator.evaluate_many([plan, [1, 2, 2]])
    district_frame = many["population"]
    plan_series = many["cut_edges"]
    assert isinstance(district_frame, pd.DataFrame) and isinstance(plan_series, pd.Series)
    district_frame.iloc[0, 0] = 99.0
    plan_series.iloc[0] = 99
    assert district_frame.iloc[0, 0] == 99.0 and plan_series.iloc[0] == 99

    # array() keeps its documented immutability.
    assert not evaluator.evaluate(plan).array("population").flags.writeable


def test_logical_tallies_remain_separate_while_native_columns_merge() -> None:
    scorer = (
        PlanEvaluator(graph())
        .add_metric(Tally("population"))
        .add_metric(Tally("voting_age", "population"))
    )

    result = scorer.evaluate([1, 1, 2])
    population = result["population"]
    tally = result["tally"]

    assert result.metrics == ("population", "tally")
    assert isinstance(population, pd.Series)
    assert population.name == "population"
    np.testing.assert_allclose(population, [30, 30])
    assert isinstance(tally, pd.DataFrame)
    assert tally.columns.tolist() == ["voting_age", "population"]
    np.testing.assert_allclose(tally, [[23, 30], [20, 30]])
    np.testing.assert_allclose(result.array("population"), [[30, 30]])
    np.testing.assert_allclose(result.array("tally"), [[23, 20], [30, 30]])


def test_add_metrics_registers_metric_owned_and_default_names_in_order() -> None:
    evaluator = PlanEvaluator(graph()).add_metrics(
        Tally("population", result_name="district_population"),
        CutEdges(),
    )

    assert evaluator.metrics == ("district_population", "cut_edges")
    result = evaluator.evaluate([1, 1, 2])
    np.testing.assert_allclose(result["district_population"], [30, 30])
    assert result["cut_edges"] == 1


def test_add_metrics_name_errors_do_not_partially_register_the_batch() -> None:
    batches = (
        (CutEdges(result_name="new"), Tally("voting_age", result_name="population")),
        (CutEdges(result_name="repeated"), Tally("voting_age", result_name="repeated")),
        (CutEdges(result_name="new"), Tally("voting_age", result_name="../unsafe")),
    )
    for batch in batches:
        evaluator = PlanEvaluator(graph()).add_metric(Tally("population"))

        with pytest.raises(ValueError):
            evaluator.add_metrics(*batch)

        assert evaluator.metrics == ("population",)


def test_add_metrics_configuration_errors_do_not_partially_register_the_batch() -> None:
    evaluator = PlanEvaluator(graph()).add_metric(Tally("population"))

    with pytest.raises(RuntimeError, match="Reock requires geometry"):
        evaluator.add_metrics(CutEdges(result_name="new"), Reock())

    assert evaluator.metrics == ("population",)


def test_add_metrics_defers_column_validation_until_evaluation() -> None:
    evaluator = PlanEvaluator(graph()).add_metric(Tally("population"))

    evaluator.add_metrics(CutEdges(result_name="new"), Tally("missing_column"))

    assert evaluator.metrics == ("population", "new", "missing_column")
    with pytest.raises(ValueError, match="missing_column"):
        evaluator.evaluate([0, 0, 1])


def test_evaluate_many_preserves_plan_order_and_requires_stable_districts() -> None:
    scorer = PlanEvaluator(graph()).add_metric(Tally("population"))

    assignments = np.array([["x", "x", "y"], ["x", "y", "y"]])
    result = scorer.evaluate_many(assignments, sample_ids=(10, 20))
    table = result["population"]

    assert isinstance(result, ManyPlanEvalResult)
    assert result.summary == EvaluationSummary(samples=2, accepted=2)
    assert isinstance(table, pd.DataFrame)
    assert table.index.name == "sample"
    assert table.index.tolist() == [10, 20]
    assert table.columns.name == "district"
    assert table.columns.tolist() == ["x", "y"]
    np.testing.assert_allclose(table, [[30, 30], [10, 50]])
    np.testing.assert_allclose(result.array("population"), [[[30, 30]], [[10, 50]]])

    with pytest.raises(ValueError, match="same in every assignment"):
        scorer.evaluate_many([["x", "x", "y"], ["x", "y", "z"]])
    with pytest.raises(ValueError, match="cannot contain missing district labels"):
        scorer.evaluate_many([["x", "x", "y"], ["x", "y", np.nan]])
    with pytest.raises(ValueError, match="expected 2"):
        scorer.evaluate_many(assignments, sample_ids=[10])
    with pytest.raises(ValueError, match="unique"):
        scorer.evaluate_many(assignments, sample_ids=[10, 10])
    unhashable_ids = cast("Iterable[Hashable]", [[10], [20]])
    with pytest.raises(ValueError, match="hashable"):
        scorer.evaluate_many(assignments, sample_ids=unhashable_ids)
    with pytest.raises(TypeError, match="track_uniqueness"):
        scorer.evaluate_many(
            assignments,
            track_uniqueness=1,  # type: ignore
        )


def test_evaluation_summary_repr_omits_untracked_uniqueness() -> None:
    assert repr(EvaluationSummary(samples=10_000, accepted=10_000)) == (
        "EvaluationSummary(samples=10000, accepted=10000)"
    )
    assert repr(EvaluationSummary(10, 8, 7, 20)) == (
        "EvaluationSummary(samples=10, accepted=8, unique_plans=7, unique_districts=20)"
    )


def test_evaluate_many_optionally_counts_label_invariant_unique_plans_and_districts() -> None:
    evaluator = PlanEvaluator(graph()).add_metric(Tally("population"))

    result = evaluator.evaluate_many(
        [[0, 0, 1], [1, 1, 0], [0, 1, 1]],
        track_uniqueness=True,
    )

    assert result.summary == EvaluationSummary(
        samples=3,
        accepted=3,
        unique_plans=2,
        unique_districts=4,
    )


def test_evaluate_many_uses_range_index_and_exact_multi_value_axes_and_dtypes() -> None:
    source = graph()
    nx.set_node_attributes(source, {"a": "one", "b": "two", "c": "one"}, "region")
    nx.set_node_attributes(source, {node: "all" for node in source}, "zone")
    evaluator = (
        PlanEvaluator(source)
        .add_metric(Tally("population", "voting_age", result_name="demographics"))
        .add_metric(RegionSplits("region", "zone", result_name="splits"))
        .add_metric(CutEdges())
    )

    result = evaluator.evaluate_many([["west", "west", "east"], ["west", "east", "east"]])
    demographics = result["demographics"]
    splits = result["splits"]
    cut_edges = result["cut_edges"]

    assert isinstance(demographics, pd.DataFrame)
    assert isinstance(demographics.index, pd.RangeIndex)
    assert demographics.index.names == ["sample"]
    assert demographics.columns.names == ["metric", "district"]
    assert demographics.columns.tolist() == [
        ("population", "west"),
        ("population", "east"),
        ("voting_age", "west"),
        ("voting_age", "east"),
    ]
    assert demographics.dtypes.tolist() == [np.dtype("float64")] * 4
    np.testing.assert_allclose(
        demographics,
        [[30, 30, 23, 20], [10, 50, 8, 35]],
    )
    assert result.array("demographics").shape == (2, 2, 2)

    assert isinstance(splits, pd.DataFrame)
    assert splits.index.equals(pd.RangeIndex(2, name="sample"))
    assert splits.columns.equals(pd.Index(["region", "zone"], name="metric"))
    assert splits.dtypes.tolist() == [np.dtype("int64")] * 2
    np.testing.assert_array_equal(splits, [[1, 1], [1, 1]])
    assert result.array("splits").shape == (2, 2)
    assert result.array("splits").dtype == np.dtype("float64")

    assert isinstance(cut_edges, pd.Series)
    assert cut_edges.index.equals(pd.RangeIndex(2, name="sample"))
    assert cut_edges.name == "cut_edges"
    assert cut_edges.dtype == np.dtype("int64")
    np.testing.assert_array_equal(cut_edges, [1, 1])
    assert result.array("cut_edges").shape == (2, 1)


def test_tally_by_region_evaluate_many_has_sample_region_rows_and_metric_district_columns() -> None:
    source = graph()
    nx.set_node_attributes(source, {"a": "one", "b": "two", "c": "one"}, "region")
    evaluator = PlanEvaluator(source).add_metric(
        TallyByRegion(
            "region",
            {
                "population": "population",
                "voting_age": "voting_age",
            },
            include_count=True,
            result_name="regional_demographics",
        ),
    )

    result = evaluator.evaluate_many(
        [["west", "west", "east"], ["west", "east", "east"]],
        sample_ids=[10, 20],
    )
    table = result["regional_demographics"]

    assert isinstance(table, pd.DataFrame)
    assert table.index.equals(
        pd.MultiIndex.from_product(
            ([10, 20], ["one", "two"]),
            names=("sample", "region"),
        )
    )
    assert table.columns.equals(
        pd.MultiIndex.from_product(
            (["count", "population", "voting_age"], ["west", "east"]),
            names=("metric", "district"),
        )
    )
    assert table.dtypes.tolist() == [
        np.dtype("int64"),
        np.dtype("int64"),
        np.dtype("float64"),
        np.dtype("float64"),
        np.dtype("float64"),
        np.dtype("float64"),
    ]
    np.testing.assert_allclose(
        table,
        [
            [1, 1, 10, 30, 8, 20],
            [1, 0, 20, 0, 15, 0],
            [1, 1, 10, 30, 8, 20],
            [0, 1, 0, 20, 0, 15],
        ],
    )
    assert result.array("regional_demographics").shape == (2, 3, 2, 2)


def test_generated_tally_by_region_arrays_match_an_independent_grouping_oracle() -> None:
    rng = np.random.default_rng(0x7A11_8E61)
    node_count = 25
    source = nx.empty_graph(node_count)
    region_values = ["A", 2, None, "A", 2] * 5
    first = rng.normal(size=node_count)
    second = rng.integers(-20, 21, size=node_count)
    nx.set_node_attributes(source, dict(enumerate(region_values)), "region")
    nx.set_node_attributes(source, dict(enumerate(first)), "first")
    nx.set_node_attributes(source, dict(enumerate(second)), "second")
    evaluator = PlanEvaluator(source).add_metric(
        TallyByRegion(
            "region",
            {"first": "first", "second": "second"},
            include_count=True,
        )
    )
    assignments = []
    for _ in range(100):
        assignment = rng.integers(0, 3, size=node_count)
        assignment[:3] = [0, 1, 2]
        assignments.append(assignment.tolist())

    actual = evaluator.evaluate_many(assignments).array("tally_by_region")
    expected = np.zeros((len(assignments), 3, 2, 3))
    for sample, assignment in enumerate(assignments):
        for node, (region, district) in enumerate(zip(region_values, assignment, strict=True)):
            if region is None:
                continue
            region_index = 0 if region == "A" else 1
            expected[sample, 0, region_index, district] += 1
            expected[sample, 1, region_index, district] += first[node]
            expected[sample, 2, region_index, district] += second[node]

    np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-13)


def test_graph_resources_are_snapshotted_and_validated_at_preparation() -> None:
    source = graph()
    scorer = PlanEvaluator(source).add_metric(Tally("population"))
    source.nodes["a"]["population"] = 1_000

    np.testing.assert_allclose(
        scorer.evaluate([0, 0, 1]).array("population"),
        [[1_020, 30]],
    )
    source.nodes["a"]["population"] = 2_000
    np.testing.assert_allclose(scorer.evaluate([0, 0, 1]).array("population"), [[1_020, 30]])

    del source.nodes["b"]["voting_age"]
    with pytest.raises(ValueError, match="has no 'voting_age' attribute"):
        PlanEvaluator(source).add_metric(Tally("voting_age")).evaluate([0, 0, 1])


def test_plan_evaluator_rejects_invalid_registration_and_assignments() -> None:
    scorer = PlanEvaluator(graph())

    with pytest.raises(RuntimeError, match="at least one metric"):
        scorer.evaluate([0, 0, 1])
    with pytest.raises(ValueError, match="exactly match graph nodes"):
        scorer.add_metric(Tally("population")).evaluate({"a": 0, "b": 0})
    with pytest.raises(ValueError, match="at least one plan"):
        scorer.evaluate_many([])
    with pytest.raises(ValueError, match="Tally requires"):
        Tally()


def test_entry_point_error_precedence_prepares_before_plan_normalization() -> None:
    evaluator = PlanEvaluator(graph())

    with pytest.raises(RuntimeError, match="at least one metric"):
        evaluator.evaluate({"missing": 0})
    with pytest.raises(RuntimeError, match="at least one metric"):
        evaluator.evaluate_many([{"missing": 0}])
    with pytest.raises(ValueError, match="at least one plan"):
        evaluator.evaluate_many([])

    def failing_plans():
        raise LookupError("iterator failed")
        yield [0, 0, 1]

    with pytest.raises(LookupError, match="iterator failed"):
        evaluator.evaluate_many(failing_plans())


@pytest.mark.parametrize("missing", [None, np.nan])
def test_plan_evaluator_rejects_missing_district_labels(missing) -> None:
    scorer = PlanEvaluator(graph()).add_metric(Tally("population"))

    with pytest.raises(ValueError, match="assignment cannot contain missing district labels"):
        scorer.evaluate([0, missing, 1])


def test_add_metric_rejects_objects_outside_the_supported_metric_set() -> None:
    class UserMetric:
        _kind = "user_metric"

    with pytest.raises(TypeError, match="supported GerryTools metric"):
        PlanEvaluator(graph()).add_metric(cast(Metric, UserMetric()))

    # The exported Metric union still supports runtime isinstance checks.
    assert isinstance(Tally("population"), Metric)
    assert not isinstance(UserMetric(), Metric)


def test_metric_descriptors_are_slots_only() -> None:
    assert all(metric.__dictoffset__ == 0 for metric in get_args(Metric))


def test_single_column_tally_falls_back_when_the_column_is_not_a_usable_name() -> None:
    spaced = nx.Graph()
    spaced.add_node("a", **{"TOTAL POP": 10, "VAP": 8})
    spaced.add_node("b", **{"TOTAL POP": 20, "VAP": 15})
    spaced.add_edge("a", "b")

    assert PlanEvaluator(graph()).add_metric(Tally("population")).metrics == ("population",)

    unusable = PlanEvaluator(spaced).add_metric(Tally("TOTAL POP"))
    assert unusable.metrics == ("tally",)
    tallied = unusable.evaluate([0, 1])["tally"]
    assert isinstance(tallied, pd.Series)
    assert list(tallied) == [10.0, 20.0]

    named = PlanEvaluator(spaced).add_metric(Tally("TOTAL POP", result_name="totpop"))
    assert named.metrics == ("totpop",)


def test_metric_names_are_explicit_safe_and_never_suffixed() -> None:
    scorer = PlanEvaluator(graph()).add_metric(Tally("population"))

    with pytest.raises(ValueError, match="already registered"):
        scorer.add_metric(Tally("population"))
    with pytest.raises(ValueError, match="already registered"):
        scorer.add_metric(Tally("voting_age", result_name="population"))
    for name in (
        "",
        ".",
        "..",
        "manifest.json",
        "Manifest.json",
        "../population",
        "population/total",
        "pöpulation",
    ):
        with pytest.raises(ValueError, match="ASCII"):
            PlanEvaluator(graph()).add_metric(Tally("population", result_name=name))
    with pytest.raises(TypeError, match="must be a string"):
        PlanEvaluator(graph()).add_metric(
            Tally("population", result_name=1)  # type: ignore
        )

    evaluator = PlanEvaluator(graph()).add_metrics(
        Tally("population", result_name="population_once"),
        Tally("population", result_name="population_again"),
    )
    result = evaluator.evaluate([0, 0, 1])
    assert result.metrics == ("population_once", "population_again")
    assert len(evaluator._engine_prepared) == 1
    assert not any(name.endswith("_2") for name in result)
    assert Tally("population", result_name="population_once") == Tally(
        "population", result_name="population_again"
    )
    assert hash(Tally("population", result_name="population_once")) == hash(
        Tally("population", result_name="population_again")
    )


def test_duplicate_name_fails_before_validation_without_mutating_registrations() -> None:
    scorer = PlanEvaluator(graph()).add_metric(Tally("population"))

    with pytest.raises(ValueError, match="already registered"):
        scorer.add_metric(Tally("missing", result_name="population"))

    result = scorer.evaluate([0, 0, 1])
    assert result.metrics == ("population",)
    np.testing.assert_allclose(result["population"], [30, 30])


def test_edge_data_stays_paired_with_edges_when_graph_iteration_is_unsorted() -> None:
    source = nx.Graph()
    source.add_nodes_from(range(4), population=1)
    source.add_edge(0, 3, cut_weight=40)
    source.add_edge(0, 1, cut_weight=10)
    source.add_edge(0, 2, cut_weight=20)
    assert list(source.edges) == [(0, 3), (0, 1), (0, 2)]

    scorer = PlanEvaluator(source)
    source.edges[0, 3]["cut_weight"] = 4_000
    result = scorer.add_metric(CutEdges(weight_attr="cut_weight")).evaluate([0, 0, 0, 1])

    assert result["cut_edges"] == 4_000.0
    np.testing.assert_allclose(result.array("cut_edges"), [4_000])
    source.edges[0, 3]["cut_weight"] = 40_000
    assert scorer.evaluate([0, 0, 0, 1])["cut_edges"] == 4_000.0


def test_region_labels_preserve_first_seen_order_and_ignore_missing_values() -> None:
    source = nx.path_graph(5)
    labels = [("county", 1), None, 7, ("county", 1), pd.NA]
    nx.set_node_attributes(source, dict(enumerate(labels)), "region")
    scorer = (
        PlanEvaluator(source)
        .add_metric(RegionSplits("region"))
        .add_metric(RegionPieces("region"))
        .add_metric(RegionParts("region"))
        .add_metric(TallyByRegion("region", include_count=True))
    )

    result = scorer.evaluate([0, 0, 1, 0, 1])

    tally_by_region = result["tally_by_region"]
    assert isinstance(tally_by_region, pd.DataFrame)
    assert tally_by_region.index.equals(pd.Index([("county", 1), 7], name="region"))
    assert tally_by_region.columns.equals(
        pd.MultiIndex.from_product(
            (["count"], [0, 1]),
            names=("metric", "district"),
        )
    )
    np.testing.assert_array_equal(tally_by_region, [[2, 0], [0, 1]])
    np.testing.assert_allclose(
        result.array("tally_by_region"),
        [[[2, 0], [0, 1]]],
    )
    assert result["region_splits"] == 0
    assert result["region_pieces"] == 2
    assert result["region_parts"] == 3

    source.nodes[0]["region"] = ["unhashable"]
    with pytest.raises(ValueError, match="must be hashable"):
        PlanEvaluator(source).add_metric(TallyByRegion("region", include_count=True)).evaluate(
            [0, 0, 1, 0, 1]
        )


def test_python_boundary_accepts_500_districts() -> None:
    assert _scoring_engine.MAX_DISTRICTS == 2**16
    source = nx.empty_graph(501)
    nx.set_node_attributes(source, {node: 1 for node in source}, "population")
    scorer = PlanEvaluator(source).add_metric(Tally("population"))
    labels = list(range(0, 500_000, 1_000))

    table = scorer.evaluate([*labels, labels[0]])["population"]

    assert isinstance(table, pd.Series)
    assert table.index.tolist() == labels
    np.testing.assert_allclose(table, [2, *([1] * 499)])


@pytest.mark.parametrize(
    ("source", "message"),
    [
        (nx.DiGraph([(0, 1)]), "undirected"),
        (nx.MultiGraph([(0, 1), (0, 1)]), "simple"),
        (nx.Graph(), "at least one node"),
        (nx.Graph([(0, 0)]), "self-loop"),
    ],
)
def test_plan_evaluator_rejects_unsupported_graphs(source: nx.Graph, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        PlanEvaluator(source)
