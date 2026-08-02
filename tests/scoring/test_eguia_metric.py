import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Literal, cast

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest
from binary_ensemble import BenEncoder
from gerrychain import Partition
from shapely.geometry import box

import gerrytools.scoring as scoring
from gerrytools.scoring import Eguia, PlanEvaluator, Tally, formulas

Variant = Literal["standard", "mkv_chain", "twodelta"]


def resources() -> tuple[nx.Graph, gpd.GeoDataFrame, Partition]:
    graph = nx.path_graph(["a", "b", "c", "d"])
    frame = gpd.GeoDataFrame(
        {
            "party": [60, 10, 10, 20],
            "opposition": [40, 20, 30, 10],
            "population": [30, 20, 25, 25],
            "county": ["A", "A", "B", "B"],
            "district": ["north", "north", "south", "south"],
        },
        index=pd.Index(["a", "b", "c", "d"]),
        geometry=[box(index, 0, index + 1, 1) for index in range(4)],
        crs="EPSG:3857",
    )
    assignment = {str(node): str(district) for node, district in frame["district"].items()}
    return graph, frame, Partition(graph, assignment)


def metric(*, result_name: str | None = None) -> Eguia:
    return Eguia(
        party_vote_attr="party",
        opposition_vote_attr="opposition",
        region_attr="county",
        population_attr="population",
        result_name=result_name,
    )


def formula_score(frame: gpd.GeoDataFrame, assignment: Sequence[int | str]) -> float:
    districts = tuple(dict.fromkeys(assignment))
    district_party = [
        frame.loc[[label == district for label in assignment], "party"].sum()
        for district in districts
    ]
    district_opposition = [
        frame.loc[[label == district for label in assignment], "opposition"].sum()
        for district in districts
    ]
    grouped = frame.groupby("county", sort=False)
    return float(
        formulas.eguia(
            district_party,
            district_opposition,
            cast(pd.Series, grouped["party"].sum()),
            cast(pd.Series, grouped["opposition"].sum()),
            cast(pd.Series, grouped["population"].sum()),
        )
    )


def test_single_plan_and_prepared_eguia_match_the_array_formula() -> None:
    graph, frame, partition = resources()
    assignment = list(frame["district"])
    expected = formula_score(frame, assignment)

    prepared = (
        PlanEvaluator(graph, geometry=frame).add_metric(metric()).evaluate(partition)["eguia"]
    )

    assert prepared == pytest.approx(expected)
    assert scoring.eguia(
        partition,
        party_vote_attr="party",
        opposition_vote_attr="opposition",
        region_attr="county",
        population_attr="population",
        geometry=frame,
    ) == pytest.approx(expected)
    assert scoring.eguia(
        frame,
        "district",
        party_vote_attr="party",
        opposition_vote_attr="opposition",
        region_attr="county",
        population_attr="population",
    ) == pytest.approx(expected)


def test_eguia_and_public_tallies_share_one_hidden_native_bank() -> None:
    graph, frame, _ = resources()
    evaluator = (
        PlanEvaluator(graph, geometry=frame)
        .add_metric(metric())
        .add_metric(Tally("party", "population", result_name="reported_tallies"))
    )

    result = evaluator.evaluate(["north", "north", "south", "south"])

    assert result.metrics == ("eguia", "reported_tallies")
    assert evaluator._tally_indices == {"party": 0, "opposition": 1, "population": 2}
    assert len(evaluator._engine_prepared) == 2
    assert result["eguia"] == pytest.approx(0)
    np.testing.assert_allclose(result["reported_tallies"], [[70, 50], [30, 50]])


def test_eguia_benchmark_is_prepared_once_and_reused(monkeypatch: pytest.MonkeyPatch) -> None:
    graph, frame, _ = resources()
    calls = 0
    original = Eguia._compute_benchmark

    def counted(self: Eguia, evaluator: PlanEvaluator) -> float:
        nonlocal calls
        calls += 1
        return original(self, evaluator)

    monkeypatch.setattr(Eguia, "_compute_benchmark", counted)
    evaluator = PlanEvaluator(graph, geometry=frame).add_metric(metric())
    assert calls == 0

    evaluator.evaluate(["north", "north", "south", "south"])
    evaluator.evaluate(["east", "west", "east", "west"])
    assert calls == 1

    evaluator.add_metric(Tally("population", result_name="reported_population")).evaluate(
        ["north", "north", "south", "south"]
    )
    assert calls == 1


def test_generated_eguia_scores_match_the_array_formula() -> None:
    rng = np.random.default_rng(20_260_728)
    node_count = 40
    graph = nx.path_graph(node_count)
    frame = gpd.GeoDataFrame(
        {
            "party": rng.integers(0, 1_000, node_count),
            "opposition": rng.integers(0, 1_000, node_count),
            "population": rng.integers(1, 1_000, node_count),
            "county": rng.integers(0, 8, node_count),
        },
        geometry=[box(index, 0, index + 1, 1) for index in range(node_count)],
        crs="EPSG:3857",
    )
    evaluator = PlanEvaluator(graph, geometry=frame).add_metric(metric())

    for _ in range(100):
        dense = rng.integers(0, 5, node_count).tolist()
        if len(set(dense)) != 5:
            continue
        labels = rng.permutation([f"district-{index}" for index in range(5)])
        assignment = [str(labels[district]) for district in dense]
        actual = evaluator.evaluate(assignment)["eguia"]
        assert actual == pytest.approx(formula_score(frame, assignment))


@pytest.mark.parametrize(
    "column,value,message",
    [
        ("party", -1, "cannot contain negative"),
        ("opposition", -1, "cannot contain negative"),
        ("population", -1, "cannot contain negative"),
        ("population", 0, "positive total"),
        ("county", None, "labels cannot be missing"),
    ],
)
def test_eguia_rejects_invalid_fixed_inputs(column: str, value: object, message: str) -> None:
    graph, frame, _ = resources()
    if column == "population" and value == 0:
        frame[column] = 0
    else:
        frame.loc["a", column] = value

    with pytest.raises(ValueError, match=message):
        PlanEvaluator(graph, geometry=frame).add_metric(metric()).evaluate(
            ["north", "north", "south", "south"]
        )


@pytest.mark.parametrize(
    "column",
    ["party_vote_attr", "opposition_vote_attr", "region_attr", "population_attr"],
)
def test_eguia_rejects_invalid_column_names(column: str) -> None:
    options = {
        "party_vote_attr": "party",
        "opposition_vote_attr": "opposition",
        "region_attr": "county",
        "population_attr": "population",
    }
    options[column] = ""

    with pytest.raises(ValueError, match=f"Eguia {column} must be a nonempty column name"):
        Eguia(**options)


def test_eguia_rejects_missing_columns() -> None:
    graph, frame, _ = resources()
    with pytest.raises(ValueError, match="has no 'missing' attribute"):
        PlanEvaluator(graph, geometry=frame).add_metric(
            Eguia("missing", "opposition", "county", "population")
        ).evaluate(["north", "north", "south", "south"])


@pytest.mark.parametrize("column", ["party", "opposition", "population"])
def test_eguia_rejects_finite_values_whose_fixed_total_overflows(column: str) -> None:
    graph, frame, _ = resources()
    frame[column] = 1e308

    with pytest.raises(ValueError, match="finite total"):
        PlanEvaluator(graph, geometry=frame).add_metric(metric()).evaluate(
            ["north", "north", "south", "south"]
        )


@pytest.mark.parametrize("variant", ["standard", "mkv_chain", "twodelta"])
def test_streamed_eguia_matches_in_memory_and_hides_tally_dependencies(
    tmp_path: Path, variant: Variant
) -> None:
    source = tmp_path / f"plans-{variant}.ben"
    output = tmp_path / "scores"
    plans = [["north", "north", "south", "south"], ["north", "south", "north", "south"]]
    encoded = [[0, 0, 1, 1], [0, 1, 0, 1]]
    with BenEncoder(source, variant=variant) as stream:
        for assignment in encoded:
            stream.write(assignment)

    graph, frame, _ = resources()
    evaluator = PlanEvaluator(graph, geometry=frame).add_metric(metric(result_name="eguia_2020"))
    expected = evaluator.evaluate_many(plans).array("eguia_2020")[:, 0]

    evaluator.evaluate_stream(source, output, batch_size=2)

    manifest = json.loads((output / "manifest__scores.json").read_text())
    assert [description["instance"] for description in manifest["metrics"]] == ["eguia_2020"]
    table_path = output / "eguia_2020__scores.parquet"
    assert manifest["metrics"][0] == {
        "kind": "eguia",
        "instance": "eguia_2020",
        "options": {
            "party_vote_attr": "party",
            "opposition_vote_attr": "opposition",
            "region_attr": "county",
            "population_attr": "population",
        },
        "shape": "plan",
        "subkeys": ["score"],
        "axes": {"metric": ["score"]},
        "dtypes": ["float"],
        "tables": [
            {
                "path": "eguia_2020__scores.parquet",
                "subkeys": ["score"],
                "size": table_path.stat().st_size,
                "sha256": hashlib.sha256(table_path.read_bytes()).hexdigest(),
            }
        ],
    }
    table = pq.read_table(table_path).to_pydict()
    np.testing.assert_allclose(table["score"], expected)


def test_formula_namespace_replaces_derived_and_top_level_eguia_is_plan_oriented() -> None:
    assert not hasattr(scoring, "derived")
    assert scoring.eguia is not formulas.eguia
    assert formulas.eguia([1], [0], [1], [0], [1]) == pytest.approx(0)
