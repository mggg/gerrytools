import hashlib
import json
import weakref
from pathlib import Path
from typing import Literal, cast

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest
from binary_ensemble import BendlEncoder, BenEncoder, compress_stream
from shapely.geometry import box

from gerrytools.scoring import (
    ConvexHullRatio,
    CutEdges,
    EnsembleEvalResult,
    EvaluationMemoryError,
    EvaluationSummary,
    PlanEvaluator,
    PolsbyPopper,
    PopulationPolygon,
    RegionParts,
    RegionPieces,
    RegionSplits,
    Reock,
    StateClippedConvexHullRatio,
    Tally,
    TallyByRegion,
)
from gerrytools.scoring import evaluator as evaluator_module
from gerrytools.scoring import result as result_module

Variant = Literal["standard", "mkv_chain", "twodelta"]
Container = Literal["ben", "bendl", "bendl_xben"]


def grid_resources() -> tuple[nx.Graph, gpd.GeoDataFrame]:
    graph = nx.Graph()
    graph.add_node(0, population=10, area=1, boundary_perim=2, COUNTY="A", MUNI="L")
    graph.add_node(1, population=20, area=1, boundary_perim=2, COUNTY="A", MUNI="R")
    graph.add_node(2, population=30, area=1, boundary_perim=2, COUNTY="B", MUNI="L")
    graph.add_node(3, population=40, area=1, boundary_perim=2, COUNTY="B", MUNI="R")
    graph.add_edge(0, 1, shared_perim=1)
    graph.add_edge(0, 2, shared_perim=1)
    graph.add_edge(1, 3, shared_perim=1)
    graph.add_edge(2, 3, shared_perim=1)
    geometry = gpd.GeoDataFrame(
        {
            "population": [10, 20, 30, 40],
            "COUNTY": ["A", "A", "B", "B"],
            "MUNI": ["L", "R", "L", "R"],
        },
        geometry=[
            box(0, 1, 1, 2),
            box(1, 1, 2, 2),
            box(0, 0, 1, 1),
            box(1, 0, 2, 1),
        ],
        crs="EPSG:3857",
    )
    return graph, geometry


def scorer() -> PlanEvaluator:
    graph, geometry = grid_resources()
    return (
        PlanEvaluator(graph, geometry=geometry)
        .add_metric(Tally("population"))
        .add_metric(PopulationPolygon("population"))
        .add_metric(Reock())
        .add_metric(ConvexHullRatio())
        .add_metric(StateClippedConvexHullRatio(box(0, 0, 2, 2)))
        .add_metric(PolsbyPopper(area_attr="area", result_name="polsby_graph"))
        .add_metric(PolsbyPopper(result_name="polsby_geometry"))
        .add_metric(CutEdges())
        .add_metric(RegionSplits("COUNTY", "MUNI"))
        .add_metric(RegionPieces("COUNTY", "MUNI"))
        .add_metric(RegionParts("COUNTY", "MUNI"))
        .add_metric(
            TallyByRegion(
                "COUNTY",
                {"population": "population"},
                include_count=True,
            )
        )
    )


def assert_evaluation_value_equal(
    actual: object,
    expected: object,
    index_name: str,
) -> None:
    assert isinstance(actual, (pd.Series, pd.DataFrame))
    assert isinstance(expected, (pd.Series, pd.DataFrame))
    expected = expected.copy()
    names = list(expected.index.names)
    names[0] = index_name
    expected.index = expected.index.set_names(names)
    if isinstance(expected, pd.Series):
        assert isinstance(actual, pd.Series)
        pd.testing.assert_series_equal(actual, expected)
    else:
        assert isinstance(actual, pd.DataFrame)
        pd.testing.assert_frame_equal(actual, expected)


def write_source(path: Path, container: Container, variant: Variant) -> None:
    plans = ([0, 0, 1, 1], [0, 1, 0, 1])
    if container == "ben":
        with BenEncoder(path, variant=variant) as stream:
            for plan in plans:
                stream.write(plan)
        return

    encoder = BendlEncoder(path)
    with encoder.ben_stream(variant=variant) as stream:
        for plan in plans:
            stream.write(plan)
    if container == "bendl_xben":
        compress_stream(path)


def run_manifest(output: Path) -> Path:
    return output / f"manifest__{output.name}.json"


def metric_table(output: Path, instance: str, index: int = 0) -> Path:
    manifest = json.loads(run_manifest(output).read_text())
    metric = next(metric for metric in manifest["metrics"] if metric["instance"] == instance)
    return output / metric["tables"][index]["path"]


def refresh_table_integrity(output: Path, instance: str, index: int = 0) -> None:
    """Update one table's test manifest metadata after deliberate structural corruption."""
    manifest_path = run_manifest(output)
    manifest = json.loads(manifest_path.read_text())
    metric = next(metric for metric in manifest["metrics"] if metric["instance"] == instance)
    table_description = metric["tables"][index]
    table = output / table_description["path"]
    table_description["size"] = table.stat().st_size
    table_description["sha256"] = hashlib.sha256(table.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest))


@pytest.mark.parametrize("container", ["ben", "bendl", "bendl_xben"])
@pytest.mark.parametrize("variant", ["standard", "mkv_chain", "twodelta"])
def test_evaluate_stream_supports_every_container_and_variant(
    tmp_path: Path, container: Container, variant: Variant
) -> None:
    source = tmp_path / f"plans-{variant}.{container}"
    output = tmp_path / "scores"
    write_source(source, container, variant)

    plan_evaluator = scorer()
    expected = plan_evaluator.evaluate_many([[0, 0, 1, 1], [0, 1, 0, 1]]).array(
        "state_clipped_convex_hull_ratio"
    )
    run = plan_evaluator.evaluate_stream(source, output, batch_size=1, track_uniqueness=True)

    assert isinstance(run, EnsembleEvalResult)
    assert run.summary == EvaluationSummary(
        samples=2, accepted=2, unique_plans=2, unique_districts=4
    )
    actual = pq.read_table(output / "state_clipped_convex_hull_ratio__scores.parquet").to_pydict()
    np.testing.assert_allclose(actual["score__district_0"], expected[:, 0, 0])
    np.testing.assert_allclose(actual["score__district_1"], expected[:, 0, 1])


def test_evaluate_stream_round_trips_every_metric_and_manifest_field(tmp_path: Path) -> None:
    source = tmp_path / "plans.ben"
    output = tmp_path / "scores"
    plans = [[0, 0, 1, 1], [0, 1, 0, 1]]
    write_source(source, "ben", "standard")
    plan_evaluator = scorer()
    expected = plan_evaluator.evaluate_many(plans)

    run = plan_evaluator.evaluate_stream(source, output, batch_size=1, track_uniqueness=True)

    assert run.summary == EvaluationSummary(
        samples=2, accepted=2, unique_plans=2, unique_districts=4
    )
    manifest = json.loads(run_manifest(output).read_text())
    metric_contracts = [
        ("tally", "population", {}, "district", ["population"]),
        (
            "population_polygon",
            "population_polygon",
            {
                "model": "full_weight_intersection",
                "population_col": "population",
                "surface": "scorer_geometry",
            },
            "district",
            ["score"],
        ),
        ("reock", "reock", {"source": "geometry"}, "district", ["score"]),
        (
            "convex_hull_ratio",
            "convex_hull_ratio",
            {"source": "geometry"},
            "district",
            ["score"],
        ),
        (
            "state_clipped_convex_hull_ratio",
            "state_clipped_convex_hull_ratio",
            {"source": "geometry", "state_geometry": "explicit"},
            "district",
            ["score"],
        ),
        (
            "polsby_popper",
            "polsby_graph",
            {
                "area_attr": "area",
                "boundary_perimeter_attr": "boundary_perim",
                "shared_perimeter_attr": "shared_perim",
                "source": "graph",
            },
            "district",
            ["score"],
        ),
        (
            "polsby_popper",
            "polsby_geometry",
            {"source": "geometry"},
            "district",
            ["score"],
        ),
        ("cut_edges", "cut_edges", {"weight_attr": None}, "plan", ["count"]),
        ("region_splits", "region_splits", {}, "plan", ["COUNTY", "MUNI"]),
        (
            "region_pieces",
            "region_pieces",
            {},
            "plan",
            ["COUNTY", "MUNI"],
        ),
        ("region_parts", "region_parts", {}, "plan", ["COUNTY", "MUNI"]),
        (
            "tally_by_region",
            "tally_by_region",
            {
                "region_attr": "COUNTY",
                "columns": {"population": "population"},
                "include_count": True,
            },
            "region",
            [
                "count__region_0",
                "count__region_1",
                "population__region_0",
                "population__region_1",
            ],
        ),
    ]
    expected_metrics = []
    for kind, instance, options, shape, subkeys in metric_contracts:
        if kind == "tally":
            physical_tables = [(f"{instance}/{subkeys[0]}_tallies__scores.parquet", subkeys)]
        elif kind == "tally_by_region":
            physical_tables = [
                (
                    "tally_by_region/count_tallies_by_region__scores.parquet",
                    subkeys[:2],
                ),
                (
                    "tally_by_region/population_tallies_by_region__scores.parquet",
                    subkeys[2:],
                ),
            ]
        else:
            physical_tables = [(f"{instance}__scores.parquet", subkeys)]
        description: dict[str, object] = {
            "kind": kind,
            "instance": instance,
            "options": options,
            "shape": shape,
            "subkeys": subkeys,
            "axes": {"metric": subkeys},
            "dtypes": (
                ["int", "float"]
                if instance == "tally_by_region"
                else ["int"] * len(subkeys)
                if instance in {"cut_edges", "region_splits", "region_pieces", "region_parts"}
                else ["float"] * len(subkeys)
            ),
        }
        description["tables"] = [
            {
                "path": table_name,
                "subkeys": table_subkeys,
                "size": (output / table_name).stat().st_size,
                "sha256": hashlib.sha256((output / table_name).read_bytes()).hexdigest(),
            }
            for table_name, table_subkeys in physical_tables
        ]
        if shape == "region":
            description["axes"] = {
                "metric": ["count", "population"],
                "region": {
                    "name": "COUNTY",
                    "labels": [
                        {"kind": "str", "value": "A"},
                        {"kind": "str", "value": "B"},
                    ],
                },
            }
        expected_metrics.append(description)
    assert manifest == {
        "format_version": 2,
        "source": {"path": str(source)},
        "summary": {
            "samples": 2,
            "accepted": 2,
            "unique_plans": 2,
            "unique_districts": 4,
        },
        "district_ids": [0, 1],
        "prefix_columns": [
            {"name": "sample_offset", "dtype": "uint64"},
            {"name": "repetitions", "dtype": "uint16"},
            {"name": "accepted_index", "dtype": "uint64"},
        ],
        "metrics": expected_metrics,
    }
    assert run.metrics == tuple(instance for _, instance, _, _, _ in metric_contracts)
    pd.testing.assert_frame_equal(
        run.frames,
        pd.DataFrame(
            {"sample_offset": [0, 1], "repetitions": [1, 1]},
            index=pd.RangeIndex(2, name="accepted"),
        ),
    )

    prefixes = {
        "sample_offset": [0, 1],
        "repetitions": [1, 1],
        "accepted_index": [0, 1],
    }
    for _, instance, _, shape, subkeys in metric_contracts:
        actual = run.raw(instance).to_dict(orient="list")
        assert {key: actual.pop(key) for key in prefixes} == prefixes
        values = expected.array(instance)
        if shape in {"district", "region"}:
            for column_index, column in enumerate(subkeys):
                for district_index, district in enumerate(manifest["district_ids"]):
                    np.testing.assert_allclose(
                        actual.pop(f"{column}__district_{district}"),
                        values.reshape(len(plans), -1, len(manifest["district_ids"]))[
                            :, column_index, district_index
                        ],
                    )
        else:
            for column_index, column in enumerate(subkeys):
                np.testing.assert_allclose(
                    actual.pop(str(column)),
                    values[:, column_index],
                )
        assert actual == {}
        assert_evaluation_value_equal(run.read(instance), expected[instance], "accepted")
        assert_evaluation_value_equal(
            run.read(instance, expand_repetitions=True),
            expected[instance],
            "sample",
        )

    reopened = EnsembleEvalResult.open(output)
    assert reopened.summary == run.summary
    assert reopened.metrics == run.metrics
    pd.testing.assert_frame_equal(reopened.frames, run.frames)


def test_inferred_graph_polsby_manifest_records_resolved_columns(tmp_path: Path) -> None:
    source = tmp_path / "plans.ben"
    output = tmp_path / "scores"
    write_source(source, "ben", "standard")
    graph, _ = grid_resources()
    evaluator = PlanEvaluator(graph).add_metric(PolsbyPopper())

    evaluator.evaluate_stream(source, output)

    manifest = json.loads(run_manifest(output).read_text())
    assert manifest["metrics"][0]["options"] == {
        "source": "graph",
        "area_attr": "area",
        "boundary_perimeter_attr": "boundary_perim",
        "shared_perimeter_attr": "shared_perim",
    }


def test_evaluate_stream_preserves_separate_logical_metrics_after_native_tally_merging(
    tmp_path: Path,
) -> None:
    source = tmp_path / "plans.ben"
    output = tmp_path / "scores"
    write_source(source, "ben", "standard")
    graph, _ = grid_resources()
    evaluator = PlanEvaluator(graph).add_metric(Tally("population")).add_metric(Tally("area"))
    expected = evaluator.evaluate_many([[0, 0, 1, 1], [0, 1, 0, 1]])

    evaluator.evaluate_stream(source, output, batch_size=1)

    for name in ("population", "area"):
        actual = pq.read_table(output / name / f"{name}_tallies__scores.parquet").to_pydict()
        values = expected.array(name)
        np.testing.assert_allclose(actual[f"{name}__district_0"], values[:, 0, 0])
        np.testing.assert_allclose(actual[f"{name}__district_1"], values[:, 0, 1])


def test_evaluate_stream_tags_mixed_integer_and_string_region_labels_without_column_collisions(
    tmp_path: Path,
) -> None:
    source = tmp_path / "plans.ben"
    output = tmp_path / "scores"
    write_source(source, "ben", "standard")
    graph, _ = grid_resources()
    nx.set_node_attributes(graph, {0: 1, 1: "1", 2: 1, 3: "1"}, "MIXED")
    evaluator = PlanEvaluator(graph).add_metric(
        TallyByRegion("MIXED", include_count=True, result_name="mixed_regions"),
    )

    run = evaluator.evaluate_stream(source, output, batch_size=1)

    manifest = json.loads(run_manifest(output).read_text())
    metric = manifest["metrics"][0]
    assert metric["shape"] == "region"
    assert metric["subkeys"] == ["count__region_0", "count__region_1"]
    assert metric["axes"] == {
        "metric": ["count"],
        "region": {
            "name": "MIXED",
            "labels": [
                {"kind": "int", "value": 1},
                {"kind": "str", "value": "1"},
            ],
        },
    }
    columns = pq.read_schema(
        output / "mixed_regions" / "count_tallies_by_region__scores.parquet"
    ).names
    assert columns == [
        "sample_offset",
        "repetitions",
        "accepted_index",
        "count__region_0__district_0",
        "count__region_0__district_1",
        "count__region_1__district_0",
        "count__region_1__district_1",
    ]
    result = run.read("mixed_regions")
    assert isinstance(result, pd.DataFrame)
    assert result.index.names == ["accepted", "MIXED"]
    assert isinstance(result.index, pd.MultiIndex)
    assert list(result.index.levels[1]) == [1, "1"]


def test_evaluation_run_expands_repetitions_and_caps_the_last_frame(tmp_path: Path) -> None:
    source = tmp_path / "plans.ben"
    output = tmp_path / "scores"
    first = [0, 0, 1, 1]
    second = [0, 1, 0, 1]
    with BenEncoder(source, variant="mkv_chain") as stream:
        for plan in [first, first, first, second, second]:
            stream.write(plan)
    graph, _ = grid_resources()
    evaluator = PlanEvaluator(graph).add_metric(Tally("population"))

    run = evaluator.evaluate_stream(
        source,
        output,
        max_samples=4,
        batch_size=1,
        track_uniqueness=True,
    )

    assert run.summary == EvaluationSummary(
        samples=4, accepted=2, unique_plans=2, unique_districts=4
    )
    pd.testing.assert_frame_equal(
        run.frames,
        pd.DataFrame(
            {"sample_offset": [0, 3], "repetitions": [3, 1]},
            index=pd.RangeIndex(2, name="accepted"),
        ),
    )
    expected_frames = evaluator.evaluate_many([first, second])["population"]
    expected_samples = evaluator.evaluate_many([first, first, first, second])["population"]
    assert isinstance(expected_frames, pd.DataFrame)
    assert isinstance(expected_samples, pd.DataFrame)
    assert_evaluation_value_equal(run.read("population"), expected_frames, "accepted")
    assert_evaluation_value_equal(
        run.read("population", expand_repetitions=True),
        expected_samples,
        "sample",
    )


def test_evaluation_run_iterators_reproduce_eager_results(tmp_path: Path) -> None:
    source = tmp_path / "plans.ben"
    output = tmp_path / "scores"
    write_source(source, "ben", "standard")
    run = scorer().evaluate_stream(source, output)

    for name in ("cut_edges", "population", "tally_by_region"):
        eager = run.read(name)
        batches = list(run.iter_batches(name, batch_size=1))
        assert batches
        if isinstance(eager, pd.Series):
            assert all(isinstance(batch, pd.Series) for batch in batches)
            actual = pd.concat(cast("list[pd.Series]", batches))
            pd.testing.assert_series_equal(actual, eager)
        else:
            assert isinstance(eager, pd.DataFrame)
            assert all(isinstance(batch, pd.DataFrame) for batch in batches)
            actual = pd.concat(cast("list[pd.DataFrame]", batches))
            pd.testing.assert_frame_equal(actual, eager)

    raw = run.raw("cut_edges")
    pd.testing.assert_frame_equal(
        pd.concat(run.iter_raw_batches("cut_edges", batch_size=1)),
        raw,
    )
    pd.testing.assert_frame_equal(
        pd.concat(run.iter_frame_batches(batch_size=1)),
        run.frames,
    )


def test_evaluation_run_expanded_iterator_splits_one_repetition(tmp_path: Path) -> None:
    source = tmp_path / "plans.ben"
    output = tmp_path / "scores"
    first = [0, 0, 1, 1]
    second = [0, 1, 0, 1]
    with BenEncoder(source, variant="mkv_chain") as stream:
        for plan in [first, first, first, second, second]:
            stream.write(plan)
    graph, _ = grid_resources()
    run = (
        PlanEvaluator(graph)
        .add_metric(Tally("population"))
        .evaluate_stream(
            source,
            output_dir=output,
        )
    )

    batches = list(run.iter_batches("population", batch_size=2, expand_repetitions=True))

    assert all(isinstance(batch, pd.DataFrame) for batch in batches)
    frames = cast("list[pd.DataFrame]", batches)
    assert [len(batch) for batch in frames] == [2, 2, 1]
    expected = run.read("population", expand_repetitions=True)
    assert isinstance(expected, pd.DataFrame)
    pd.testing.assert_frame_equal(pd.concat(frames), expected)


def test_evaluation_run_memory_guard_warns_and_raises_before_parquet(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "plans.ben"
    output = tmp_path / "scores"
    write_source(source, "ben", "standard")
    run = scorer().evaluate_stream(source, output)
    estimate = result_module._ERROR_BYTES + 1
    monkeypatch.setattr(result_module, "_estimate_memory", lambda *args, **kwargs: estimate)

    with pytest.warns(UserWarning) as captured:
        with pytest.raises(EvaluationMemoryError) as raised:
            run.read("cut_edges")

    assert raised.value.estimated_bytes == estimate
    assert raised.value.limit_bytes == result_module._ERROR_BYTES
    assert isinstance(raised.value, MemoryError)
    assert Path(captured[0].filename) == Path(__file__)


def test_evaluation_run_allow_large_warns_and_proceeds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "plans.ben"
    output = tmp_path / "scores"
    write_source(source, "ben", "standard")
    run = scorer().evaluate_stream(source, output)
    monkeypatch.setattr(
        result_module,
        "_estimate_memory",
        lambda *args, **kwargs: result_module._ERROR_BYTES + 1,
    )

    with pytest.warns(UserWarning):
        result = run.read("cut_edges", allow_large=True)

    assert isinstance(result, pd.Series)


@pytest.mark.parametrize(
    ("corruption", "message"),
    [
        ("magic", "footer magic"),
        ("length", "footer length"),
    ],
)
def test_evaluation_run_rejects_invalid_footer_before_parquet_construction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    corruption: str,
    message: str,
) -> None:
    source = tmp_path / "plans.ben"
    output = tmp_path / "scores"
    write_source(source, "ben", "standard")
    scorer().evaluate_stream(source, output)
    table = output / "cut_edges__scores.parquet"
    contents = bytearray(table.read_bytes())
    if corruption == "magic":
        contents[-4:] = b"NOPE"
    else:
        contents[-8:-4] = len(contents).to_bytes(4, "little")
    table.write_bytes(contents)
    refresh_table_integrity(output, "cut_edges")
    run = EnsembleEvalResult.open(output)

    def fail_parquet(*args, **kwargs):
        raise AssertionError("ParquetFile must not be constructed")

    monkeypatch.setattr(result_module.pq, "ParquetFile", fail_parquet)

    with pytest.raises(ValueError, match=message):
        run.raw("cut_edges")


def test_evaluation_run_rejects_footer_allowance_before_parquet_construction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "plans.ben"
    output = tmp_path / "scores"
    write_source(source, "ben", "standard")
    run = scorer().evaluate_stream(source, output)
    monkeypatch.setattr(result_module, "_WARNING_BYTES", 1)
    monkeypatch.setattr(result_module, "_ERROR_BYTES", 1)

    def fail_parquet(*args, **kwargs):
        raise AssertionError("ParquetFile must not be constructed")

    monkeypatch.setattr(result_module.pq, "ParquetFile", fail_parquet)

    with pytest.warns(UserWarning):
        with pytest.raises(EvaluationMemoryError):
            run.raw("cut_edges")


def test_evaluation_run_uses_actual_row_group_floor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "plans.ben"
    output = tmp_path / "scores"
    write_source(source, "ben", "standard")
    run = scorer().evaluate_stream(source, output)
    real_estimate = result_module._estimate_memory
    physical_rows: list[int | None] = []

    def tracking_estimate(*args, **kwargs):
        physical_rows.append(kwargs.get("physical_rows"))
        return real_estimate(*args, **kwargs)

    monkeypatch.setattr(result_module, "_estimate_memory", tracking_estimate)

    list(run.iter_raw_batches("cut_edges", batch_size=1))

    assert 2 in physical_rows


def test_evaluation_run_rejects_unsafe_row_group_before_requesting_a_batch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "plans.ben"
    output = tmp_path / "scores"
    write_source(source, "ben", "standard")
    run = scorer().evaluate_stream(source, output)
    parquet_file = result_module.pq.ParquetFile

    class NoBatchParquetFile:
        def __init__(self, *args, **kwargs) -> None:
            self.inner = parquet_file(*args, **kwargs)

        def __getattr__(self, name: str):
            return getattr(self.inner, name)

        def iter_batches(self, *args, **kwargs):
            raise AssertionError("record batches must not be requested")

    def row_group_estimate(*args, **kwargs):
        return result_module._ERROR_BYTES + 1 if kwargs.get("physical_rows") == 2 else 0

    monkeypatch.setattr(result_module.pq, "ParquetFile", NoBatchParquetFile)
    monkeypatch.setattr(result_module, "_estimate_memory", row_group_estimate)

    with pytest.warns(UserWarning):
        with pytest.raises(EvaluationMemoryError, match="allow_large=True"):
            next(run.iter_raw_batches("cut_edges", batch_size=1))


def test_evaluation_run_iterator_estimates_include_previous_outputs(tmp_path: Path) -> None:
    source = tmp_path / "plans.ben"
    output = tmp_path / "scores"
    write_source(source, "ben", "standard")
    run = scorer().evaluate_stream(source, output)
    metric = run._metric_metadata["cut_edges"]

    assert (
        result_module._estimate_memory(
            metric,
            run.summary,
            run._districts,
            "read",
            batch_size=1,
            physical_rows=1,
        )
        == 157
    )
    assert (
        result_module._estimate_memory(
            metric,
            run.summary,
            run._districts,
            "raw",
            batch_size=1,
            physical_rows=1,
        )
        == 269
    )
    assert (
        result_module._estimate_memory(
            metric,
            run.summary,
            run._districts,
            "frames",
            batch_size=1,
            physical_rows=1,
        )
        == 177
    )


def test_evaluation_run_estimates_semantic_shapes_and_expansion(tmp_path: Path) -> None:
    source = tmp_path / "plans.ben"
    output = tmp_path / "scores"
    write_source(source, "ben", "standard")
    run = scorer().evaluate_stream(source, output)

    assert {
        name: result_module._estimate_memory(
            run._metric_metadata[name],
            run.summary,
            run._districts,
            "read",
        )
        for name in ("cut_edges", "population", "tally_by_region")
    } == {
        "cut_edges": 273,
        "population": 393,
        "tally_by_region": 1_203,
    }
    assert (
        result_module._estimate_memory(
            run._metric_metadata["population"],
            EvaluationSummary(samples=5, accepted=2),
            run._districts,
            "read",
            expand_repetitions=True,
        )
        == 513
    )


def test_evaluation_run_disables_threaded_parquet_reads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "plans.ben"
    output = tmp_path / "scores"
    write_source(source, "ben", "standard")
    run = scorer().evaluate_stream(source, output)
    parquet_file = result_module.pq.ParquetFile
    calls: list[tuple[str, bool | None]] = []

    class TrackingArrowObject:
        def __init__(self, inner) -> None:
            self.inner = inner

        def to_pandas(self, *args, **kwargs):
            calls.append(("to_pandas", kwargs.get("use_threads")))
            return self.inner.to_pandas(*args, **kwargs)

    class TrackingParquetFile:
        def __init__(self, *args, **kwargs) -> None:
            self.inner = parquet_file(*args, **kwargs)

        def __getattr__(self, name: str):
            return getattr(self.inner, name)

        def read(self, *args, **kwargs):
            calls.append(("read", kwargs.get("use_threads")))
            return TrackingArrowObject(self.inner.read(*args, **kwargs))

        def iter_batches(self, *args, **kwargs):
            calls.append(("iter_batches", kwargs.get("use_threads")))
            return (
                TrackingArrowObject(batch) for batch in self.inner.iter_batches(*args, **kwargs)
            )

    monkeypatch.setattr(result_module.pq, "ParquetFile", TrackingParquetFile)

    run.read("cut_edges")
    list(run.iter_batches("cut_edges", batch_size=1))
    list(run.iter_raw_batches("cut_edges", batch_size=1))
    list(run.iter_frame_batches(batch_size=1))

    assert calls == [
        ("read", False),
        ("to_pandas", False),
        ("iter_batches", False),
        ("to_pandas", False),
        ("to_pandas", False),
        ("iter_batches", False),
        ("to_pandas", False),
        ("to_pandas", False),
        ("iter_batches", False),
        ("to_pandas", False),
        ("to_pandas", False),
    ]


def test_expanded_iterator_releases_inputs_before_requesting_the_next_batch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "plans.ben"
    output = tmp_path / "scores"
    first = [0, 0, 1, 1]
    second = [0, 1, 0, 1]
    with BenEncoder(source, variant="mkv_chain") as stream:
        for plan in [first, first, first, second, second]:
            stream.write(plan)
    graph, _ = grid_resources()
    run = PlanEvaluator(graph).add_metric(Tally("population")).evaluate_stream(source, output)

    parquet_file = result_module.pq.ParquetFile
    batch_refs: list[weakref.ReferenceType[object]] = []
    table_refs: list[weakref.ReferenceType[object]] = []
    physical_refs: list[weakref.ReferenceType[np.ndarray]] = []
    repetition_refs: list[weakref.ReferenceType[np.ndarray]] = []
    semantic_input_refs: list[weakref.ReferenceType[np.ndarray]] = []

    class TrackingBatch:
        def __init__(self, inner) -> None:
            self.inner = inner

        def to_pandas(self, *args, **kwargs):
            table = self.inner.to_pandas(*args, **kwargs)
            table_refs.append(weakref.ref(table))
            return table

    class TrackingBatches:
        def __init__(self, inner) -> None:
            self.inner = iter(inner)
            self.requested = 0

        def __iter__(self):
            return self

        def __next__(self):
            if self.requested:
                assert physical_refs[-1]() is None
                assert repetition_refs[-1]() is None
            self.requested += 1
            batch = TrackingBatch(next(self.inner))
            batch_refs.append(weakref.ref(batch))
            return batch

    class TrackingParquetFile:
        def __init__(self, *args, **kwargs) -> None:
            self.inner = parquet_file(*args, **kwargs)

        def __getattr__(self, name: str):
            return getattr(self.inner, name)

        def iter_batches(self, *args, **kwargs):
            kwargs["batch_size"] = 1
            return TrackingBatches(self.inner.iter_batches(*args, **kwargs))

    physical_values = result_module._physical_values
    semantic_value = result_module._semantic_value
    series_to_numpy = pd.Series.to_numpy

    def tracking_physical_values(table):
        values = physical_values(table)
        physical_refs.append(weakref.ref(values))
        return values

    def tracking_semantic_value(*args, **kwargs):
        semantic_input_refs.append(weakref.ref(args[3]))
        return semantic_value(*args, **kwargs)

    def tracking_to_numpy(series, *args, **kwargs):
        values = series_to_numpy(series, *args, **kwargs)
        if series.name == "repetitions" and kwargs.get("copy"):
            repetition_refs.append(weakref.ref(values))
        return values

    monkeypatch.setattr(result_module.pq, "ParquetFile", TrackingParquetFile)
    monkeypatch.setattr(result_module, "_physical_values", tracking_physical_values)
    monkeypatch.setattr(result_module, "_semantic_value", tracking_semantic_value)
    monkeypatch.setattr(pd.Series, "to_numpy", tracking_to_numpy)

    batches = run.iter_batches("population", batch_size=2, expand_repetitions=True)
    first_output = next(batches)
    assert isinstance(first_output, pd.DataFrame)
    assert len(first_output) == 2
    assert batch_refs[0]() is None
    assert table_refs[0]() is None
    assert physical_refs[0]() is not None
    assert repetition_refs[0]() is not None
    assert semantic_input_refs[0]() is None

    second_output = next(batches)
    assert isinstance(second_output, pd.DataFrame)
    assert len(second_output) == 1
    third_output = next(batches)
    assert isinstance(third_output, pd.DataFrame)
    assert len(third_output) == 2
    assert physical_refs[0]() is None
    assert repetition_refs[0]() is None


@pytest.mark.parametrize("operation", ["frames", "iterator"])
def test_evaluation_run_attributes_other_memory_warnings_to_the_caller(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
) -> None:
    source = tmp_path / "plans.ben"
    output = tmp_path / "scores"
    write_source(source, "ben", "standard")
    run = scorer().evaluate_stream(source, output)
    monkeypatch.setattr(
        result_module,
        "_estimate_memory",
        lambda *args, **kwargs: result_module._WARNING_BYTES + 1,
    )

    with pytest.warns(UserWarning) as captured:
        if operation == "frames":
            _ = run.frames
        else:
            list(run.iter_raw_batches("cut_edges", batch_size=1))

    assert len(captured) == 1
    assert Path(captured[0].filename) == Path(__file__)


def test_evaluation_run_validates_iterator_options_before_file_io(tmp_path: Path) -> None:
    source = tmp_path / "plans.ben"
    output = tmp_path / "scores"
    write_source(source, "ben", "standard")
    run = scorer().evaluate_stream(source, output)
    (output / "cut_edges__scores.parquet").unlink()

    with pytest.raises(ValueError, match="batch_size"):
        next(run.iter_batches("cut_edges", batch_size=0))
    with pytest.raises(TypeError, match="allow_large"):
        next(
            run.iter_raw_batches("cut_edges", allow_large=1)  # type: ignore
        )


def test_evaluation_run_rehashes_after_a_successful_read(tmp_path: Path) -> None:
    source = tmp_path / "plans.ben"
    output = tmp_path / "scores"
    write_source(source, "ben", "standard")
    run = scorer().evaluate_stream(source, output)
    run.read("cut_edges")
    table = output / "cut_edges__scores.parquet"
    contents = bytearray(table.read_bytes())
    contents[-1] ^= 1
    table.write_bytes(contents)

    with pytest.raises(ValueError, match="integrity"):
        run.read("cut_edges")


def test_evaluation_run_reports_unknown_metrics_and_exposes_raw_table(tmp_path: Path) -> None:
    source = tmp_path / "plans.ben"
    output = tmp_path / "scores"
    write_source(source, "ben", "standard")

    run = scorer().evaluate_stream(source, output)

    with pytest.raises(KeyError, match="available"):
        run.read("missing")
    raw = run.raw("cut_edges")
    assert list(raw.columns) == [
        "sample_offset",
        "repetitions",
        "accepted_index",
        "count",
    ]


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("format_version", 3, "format version 1 or 2"),
        ("prefix_columns", [], "prefix columns"),
        ("summary", None, "requires a summary"),
        ("summary", {"samples": True, "accepted": 0}, "nonnegative integer"),
        ("summary", {"samples": 0, "accepted": 1}, "cannot exceed samples"),
        ("district_ids", "all", "requires district_ids"),
        ("district_ids", [0, 0], "must be unique"),
        ("metrics", [], "at least one metric"),
    ],
)
def test_evaluation_run_rejects_invalid_manifest_contracts(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    source = tmp_path / "plans.ben"
    output = tmp_path / "scores"
    write_source(source, "ben", "standard")
    scorer().evaluate_stream(source, output)
    manifest_path = run_manifest(output)
    manifest = json.loads(manifest_path.read_text())
    manifest[field] = value
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match=message):
        EnsembleEvalResult.open(output)


def test_evaluation_run_rejects_non_object_manifest(tmp_path: Path) -> None:
    source = tmp_path / "plans.ben"
    output = tmp_path / "scores"
    write_source(source, "ben", "standard")
    scorer().evaluate_stream(source, output)
    run_manifest(output).write_text("[]")

    with pytest.raises(ValueError, match="must be an object"):
        EnsembleEvalResult.open(output)


def test_evaluation_run_rejects_duplicate_metric_names(tmp_path: Path) -> None:
    source = tmp_path / "plans.ben"
    output = tmp_path / "scores"
    write_source(source, "ben", "standard")
    scorer().evaluate_stream(source, output)
    manifest_path = run_manifest(output)
    manifest = json.loads(manifest_path.read_text())
    manifest["metrics"].append(manifest["metrics"][0])
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="metric names must be unique"):
        EnsembleEvalResult.open(output)


def test_evaluation_run_rejects_invalid_metric_metadata_and_missing_tables(
    tmp_path: Path,
) -> None:
    source = tmp_path / "plans.ben"
    output = tmp_path / "scores"
    write_source(source, "ben", "standard")
    scorer().evaluate_stream(source, output)
    manifest_path = run_manifest(output)
    manifest = json.loads(manifest_path.read_text())

    manifest["metrics"][0]["tables"][0]["sha256"] = "bad"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="SHA-256"):
        EnsembleEvalResult.open(output)

    refresh_table_integrity(output, "population")
    manifest = json.loads(manifest_path.read_text())
    manifest["metrics"][0]["dtypes"] = []
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="logical dtypes"):
        EnsembleEvalResult.open(output)

    manifest["metrics"][0]["dtypes"] = ["float"]
    manifest["metrics"][0]["tables"][0]["path"] = "../scores.parquet"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="table path"):
        EnsembleEvalResult.open(output)

    manifest["metrics"][0]["tables"][0]["path"] = "population/population_tallies__scores.parquet"
    manifest_path.write_text(json.dumps(manifest))
    (output / "population" / "population_tallies__scores.parquet").unlink()
    with pytest.raises(FileNotFoundError, match="does not exist"):
        EnsembleEvalResult.open(output)


def test_evaluation_run_rejects_metric_axes_that_disagree_with_subkeys(tmp_path: Path) -> None:
    source = tmp_path / "plans.ben"
    output = tmp_path / "scores"
    write_source(source, "ben", "standard")
    scorer().evaluate_stream(source, output)
    manifest_path = run_manifest(output)
    manifest = json.loads(manifest_path.read_text())
    metric = next(metric for metric in manifest["metrics"] if metric["instance"] == "region_splits")
    metric["axes"]["metric"] = ["wrong_county", "wrong_muni"]
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="axis values disagree with its subkeys"):
        EnsembleEvalResult.open(output)


@pytest.mark.parametrize(
    ("corruption", "message"),
    [
        ("axes", "requires axis metadata"),
        ("region", "requires a region axis"),
        ("region_name", "requires a region-axis name"),
        ("labels", "requires region labels"),
        ("duplicate_labels", "duplicate region labels"),
        ("empty_metric_axis", "requires metric-axis values"),
        ("subkeys", "subkeys disagree with its axes"),
        ("label_shape", "invalid region label"),
        ("label_value", "invalid region label"),
    ],
)
def test_evaluation_run_rejects_invalid_region_metric_metadata(
    tmp_path: Path,
    corruption: str,
    message: str,
) -> None:
    source = tmp_path / "plans.ben"
    output = tmp_path / "scores"
    write_source(source, "ben", "standard")
    scorer().evaluate_stream(source, output)
    manifest_path = run_manifest(output)
    manifest = json.loads(manifest_path.read_text())
    metric = next(
        metric for metric in manifest["metrics"] if metric["instance"] == "tally_by_region"
    )

    if corruption == "axes":
        metric["axes"] = None
    elif corruption == "region":
        metric["axes"]["region"] = None
    elif corruption == "region_name":
        metric["axes"]["region"]["name"] = ""
    elif corruption == "labels":
        metric["axes"]["region"]["labels"] = "all"
    elif corruption == "duplicate_labels":
        labels = metric["axes"]["region"]["labels"]
        metric["axes"]["region"]["labels"] = [labels[0], labels[0]]
    elif corruption == "empty_metric_axis":
        metric["axes"]["metric"] = []
        metric["subkeys"] = []
        metric["dtypes"] = []
        metric["tables"] = [metric["tables"][0]]
        metric["tables"][0]["subkeys"] = []
    elif corruption == "subkeys":
        metric["subkeys"] = ["wrong"]
    elif corruption == "label_shape":
        metric["axes"]["region"]["labels"][0] = {"value": "A"}
    else:
        metric["axes"]["region"]["labels"][0] = {"kind": "float", "value": 1.0}
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match=message):
        EnsembleEvalResult.open(output)


@pytest.mark.parametrize(
    ("corruption", "message"),
    [
        ("axes", "requires axis metadata"),
        ("region", "cannot define region axes"),
    ],
)
def test_evaluation_run_rejects_invalid_nonregion_axis_metadata(
    tmp_path: Path,
    corruption: str,
    message: str,
) -> None:
    source = tmp_path / "plans.ben"
    output = tmp_path / "scores"
    write_source(source, "ben", "standard")
    scorer().evaluate_stream(source, output)
    manifest_path = run_manifest(output)
    manifest = json.loads(manifest_path.read_text())
    metric = next(metric for metric in manifest["metrics"] if metric["instance"] == "population")
    if corruption == "axes":
        metric["axes"] = None
    else:
        metric["axes"]["region"] = {"name": "region", "labels": []}
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match=message):
        EnsembleEvalResult.open(output)


@pytest.mark.parametrize(
    ("corruption", "message"),
    [
        ("columns", "Parquet columns disagree"),
        ("row_count", "row count"),
        ("sample_offset", "sample offsets"),
        ("repetitions", "must be positive"),
        ("accepted_index", "not contiguous"),
    ],
)
def test_evaluation_run_rejects_corrupt_physical_prefixes(
    tmp_path: Path, corruption: str, message: str
) -> None:
    source = tmp_path / "plans.ben"
    output = tmp_path / "scores"
    write_source(source, "ben", "standard")
    run = scorer().evaluate_stream(source, output)
    table_path = output / "cut_edges__scores.parquet"
    table = pd.read_parquet(table_path)
    if corruption == "columns":
        table = table.rename(columns={"accepted_index": "wrong"})
    elif corruption == "row_count":
        table = table.iloc[:-1]
    elif corruption == "sample_offset":
        table.loc[1, "sample_offset"] = 9
    elif corruption == "repetitions":
        table.loc[1, "repetitions"] = 0
    else:
        table.loc[1, "accepted_index"] = 0
    table.to_parquet(table_path)
    refresh_table_integrity(output, "cut_edges")
    run = EnsembleEvalResult.open(output)

    with pytest.raises(ValueError, match=message):
        run.read("cut_edges")


def test_evaluation_run_rejects_changed_metric_values(tmp_path: Path) -> None:
    source = tmp_path / "plans.ben"
    output = tmp_path / "scores"
    write_source(source, "ben", "standard")
    run = scorer().evaluate_stream(source, output)
    table_path = output / "cut_edges__scores.parquet"
    table = pd.read_parquet(table_path)
    table.loc[0, "count"] += 1
    table.to_parquet(table_path)

    with pytest.raises(ValueError, match="integrity"):
        run.read("cut_edges")


def test_evaluation_run_rejects_same_size_table_corruption(tmp_path: Path) -> None:
    source = tmp_path / "plans.ben"
    output = tmp_path / "scores"
    write_source(source, "ben", "standard")
    run = scorer().evaluate_stream(source, output)
    table_path = output / "cut_edges__scores.parquet"
    contents = bytearray(table_path.read_bytes())
    contents[-1] ^= 1
    table_path.write_bytes(contents)

    with pytest.raises(ValueError, match="integrity"):
        run.read("cut_edges")


def test_evaluate_stream_scores_an_alternative_population_surface_like_evaluate_many(
    tmp_path: Path,
) -> None:
    source = tmp_path / "plans.ben"
    output = tmp_path / "scores"
    plans = [[0, 0, 1, 1], [0, 1, 0, 1]]
    write_source(source, "ben", "standard")
    graph, geometry = grid_resources()
    # Finer population polygons inside the four unit squares exercise the inferred-owner path.
    population = gpd.GeoDataFrame(
        {"population": [4, 6, 20, 30, 25, 15]},
        geometry=[
            box(0.1, 1.1, 0.4, 1.9),
            box(0.6, 1.1, 0.9, 1.9),
            box(1.1, 1.1, 1.9, 1.9),
            box(0.1, 0.1, 0.9, 0.9),
            box(1.1, 0.1, 1.4, 0.9),
            box(1.6, 0.1, 1.9, 0.9),
        ],
        crs="EPSG:3857",
    )
    evaluator = PlanEvaluator(graph, geometry=geometry).add_metric(
        PopulationPolygon("population", population_units=population)
    )
    expected = evaluator.evaluate_many(plans).array("population_polygon")

    run = evaluator.evaluate_stream(source, output, batch_size=1)

    assert run.summary == EvaluationSummary(samples=2, accepted=2)
    manifest = json.loads(run_manifest(output).read_text())
    assert manifest["metrics"][0]["options"] == {
        "model": "full_weight_intersection",
        "population_col": "population",
        "surface": "population_units",
        "observations": 6,
    }
    actual = pq.read_table(output / "population_polygon__scores.parquet").to_pydict()
    for district in range(2):
        np.testing.assert_allclose(
            actual[f"score__district_{district}"],
            expected[:, 0, district],
        )


def test_evaluate_stream_accepts_a_matching_embedded_bendl_graph(tmp_path: Path) -> None:
    source = tmp_path / "plans.bendl"
    output = tmp_path / "scores"
    graph, _ = grid_resources()
    encoder = BendlEncoder(source)
    encoder.add_graph(graph)
    with encoder.ben_stream(variant="standard") as stream:
        stream.write([0, 0, 1, 1])

    run = PlanEvaluator(graph).add_metric(Tally("population")).evaluate_stream(source, output)

    assert run.summary == EvaluationSummary(samples=1, accepted=1)


def test_evaluate_stream_verifies_every_bendl_asset_before_scoring(tmp_path: Path) -> None:
    source = tmp_path / "plans.bendl"
    output = tmp_path / "scores"
    graph, _ = grid_resources()
    encoder = BendlEncoder(source)
    encoder.add_graph(graph)
    encoder.add_asset("sentinel.txt", b"integrity-sentinel", "text")
    with encoder.ben_stream(variant="standard") as stream:
        stream.write([0, 0, 1, 1])
    data = bytearray(source.read_bytes())
    offset = data.index(b"integrity-sentinel")
    data[offset] ^= 1
    source.write_bytes(data)

    with pytest.raises(ValueError, match="checksum"):
        PlanEvaluator(graph).add_metric(Tally("population")).evaluate_stream(source, output)

    assert not output.exists()


def test_evaluate_stream_rejects_mismatched_bendl_graph_order_before_output(tmp_path: Path) -> None:
    source = tmp_path / "plans.bendl"
    output = tmp_path / "scores"
    graph, _ = grid_resources()
    reordered = nx.Graph()
    for node in [1, 0, 2, 3]:
        reordered.add_node(node, **graph.nodes[node])
    reordered.add_edges_from(graph.edges(data=True))
    encoder = BendlEncoder(source)
    encoder.add_graph(reordered)
    with encoder.ben_stream(variant="standard") as stream:
        stream.write([0, 0, 1, 1])

    with pytest.raises(ValueError, match="node order"):
        PlanEvaluator(graph).add_metric(Tally("population")).evaluate_stream(source, output)

    assert not output.exists()


def test_evaluate_stream_rejects_an_unrecognized_bendl_version_before_output(
    tmp_path: Path,
) -> None:
    source = tmp_path / "plans.bendl"
    output = tmp_path / "scores"
    graph, _ = grid_resources()
    encoder = BendlEncoder(source)
    encoder.add_graph(graph)
    with encoder.ben_stream(variant="standard") as stream:
        stream.write([0, 0, 1, 1])
    data = bytearray(source.read_bytes())
    assert data[:8] == b"BENDL\0\0\x01"
    data[7] = 2  # synthetic future BENDL version
    source.write_bytes(bytes(data))

    with pytest.raises(ValueError, match="unrecognized BENDL version"):
        PlanEvaluator(graph).add_metric(Tally("population")).evaluate_stream(source, output)

    assert not output.exists()


@pytest.mark.parametrize("label", [("county", 1), 2**63])
def test_evaluate_stream_rejects_region_labels_without_a_lossless_manifest_type(
    tmp_path: Path,
    label,
) -> None:
    source = tmp_path / "plans.ben"
    output = tmp_path / "scores"
    write_source(source, "ben", "standard")
    graph, _ = grid_resources()
    nx.set_node_attributes(graph, {node: label for node in graph}, "REGION")
    evaluator = PlanEvaluator(graph).add_metric(TallyByRegion("REGION", include_count=True))

    result = evaluator.evaluate([0, 0, 1, 1])["tally_by_region"]
    assert isinstance(result, pd.DataFrame)
    with pytest.raises(ValueError, match="cannot be represented"):
        evaluator.evaluate_stream(source, output)

    assert not output.exists()


@pytest.mark.parametrize("label", [True, np.bool_(True)])
def test_evaluate_rejects_boolean_region_labels(label) -> None:
    graph, _ = grid_resources()
    nx.set_node_attributes(graph, {node: label for node in graph}, "REGION")

    with pytest.raises(ValueError, match="cannot be boolean"):
        PlanEvaluator(graph).add_metric(TallyByRegion("REGION", include_count=True)).evaluate(
            [0, 0, 1, 1]
        )


def test_evaluate_stream_rejects_boolean_regions_even_when_they_equal_integer_labels() -> None:
    graph, _ = grid_resources()
    nx.set_node_attributes(graph, {0: 1, 1: True, 2: 1, 3: True}, "REGION")

    with pytest.raises(ValueError, match="cannot be boolean"):
        PlanEvaluator(graph).add_metric(TallyByRegion("REGION", include_count=True)).evaluate(
            [0, 0, 1, 1]
        )


def test_evaluate_stream_supports_an_empty_region_axis(tmp_path: Path) -> None:
    source = tmp_path / "plans.ben"
    output = tmp_path / "scores"
    write_source(source, "ben", "standard")
    graph, _ = grid_resources()
    nx.set_node_attributes(graph, {node: None for node in graph}, "REGION")
    evaluator = PlanEvaluator(graph).add_metric(TallyByRegion("REGION", include_count=True))

    in_memory = evaluator.evaluate_many([[0, 0, 1, 1], [0, 1, 0, 1]])
    assert in_memory.array("tally_by_region").shape == (2, 1, 0, 2)
    table = in_memory["tally_by_region"]
    assert isinstance(table, pd.DataFrame)
    assert table.empty
    assert table.index.names == ["sample", "REGION"]

    evaluator.evaluate_stream(source, output)

    manifest = json.loads(run_manifest(output).read_text())
    assert manifest["metrics"][0]["subkeys"] == []
    assert manifest["metrics"][0]["axes"]["region"]["labels"] == []
    assert pq.read_schema(
        output / "tally_by_region" / "count_tallies_by_region__scores.parquet"
    ).names == [
        "sample_offset",
        "repetitions",
        "accepted_index",
    ]


def test_evaluate_stream_validates_public_options_before_opening_files(tmp_path: Path) -> None:
    source = tmp_path / "missing.ben"
    output = tmp_path / "scores"
    plan_evaluator = scorer()

    with pytest.raises(ValueError, match="max_samples"):
        plan_evaluator.evaluate_stream(source, output, max_samples=-1)
    with pytest.raises(ValueError, match="batch_size"):
        plan_evaluator.evaluate_stream(source, output, batch_size=0)
    with pytest.raises(TypeError, match="track_uniqueness"):
        plan_evaluator.evaluate_stream(
            source,
            output,
            track_uniqueness=1,  # type: ignore
        )
    with pytest.raises(TypeError, match="update"):
        plan_evaluator.evaluate_stream(
            source,
            output,
            update=1,  # type: ignore
        )
    with pytest.raises(FileNotFoundError):
        plan_evaluator.evaluate_stream(source, output)
    graph, _ = grid_resources()
    with pytest.raises(RuntimeError, match="at least one metric"):
        PlanEvaluator(graph).evaluate_stream(source, output)
    assert not output.exists()


def test_evaluate_stream_accepts_numpy_integer_options(tmp_path: Path) -> None:
    source = tmp_path / "plans.ben"
    output = tmp_path / "scores"
    write_source(source, "ben", "standard")

    run = scorer().evaluate_stream(
        source,
        output,
        max_samples=np.int64(0),
        batch_size=np.int64(1),
    )

    assert run.summary == EvaluationSummary(samples=0, accepted=0)


def test_evaluate_stream_rejects_an_unrecognized_existing_directory(tmp_path: Path) -> None:
    source = tmp_path / "plans.ben"
    output = tmp_path / "scores"
    write_source(source, "ben", "standard")
    output.mkdir()
    marker = output / "keep.txt"
    marker.write_text("keep")

    with pytest.raises(FileNotFoundError):
        scorer().evaluate_stream(source, output)

    assert marker.read_text() == "keep"


def test_evaluate_stream_adds_new_scores_to_an_existing_run(tmp_path: Path) -> None:
    source = tmp_path / "plans.ben"
    output = tmp_path / "scores"
    write_source(source, "ben", "standard")
    graph, _ = grid_resources()
    first = PlanEvaluator(graph).add_metric(
        Tally("population", "area", result_name="district_totals")
    )
    first.evaluate_stream(source, output)
    population_path = output / "district_totals" / "population_tallies__scores.parquet"
    population_bytes = population_path.read_bytes()

    run = PlanEvaluator(graph).add_metric(CutEdges()).evaluate_stream(source, output)

    assert run.metrics == ("district_totals", "cut_edges")
    assert population_path.read_bytes() == population_bytes
    assert (output / "district_totals" / "area_tallies__scores.parquet").is_file()
    assert (output / "cut_edges__scores.parquet").is_file()


def test_evaluate_stream_requires_update_to_replace_an_existing_score(tmp_path: Path) -> None:
    source = tmp_path / "plans.ben"
    output = tmp_path / "scores"
    write_source(source, "ben", "standard")
    graph, _ = grid_resources()
    evaluator = PlanEvaluator(graph).add_metric(Tally("population"))
    evaluator.evaluate_stream(source, output)
    manifest = run_manifest(output).read_bytes()
    table = (output / "population" / "population_tallies__scores.parquet").read_bytes()

    with pytest.raises(FileExistsError, match="update=True"):
        evaluator.evaluate_stream(source, output)

    assert run_manifest(output).read_bytes() == manifest
    assert (output / "population" / "population_tallies__scores.parquet").read_bytes() == table


def test_evaluate_stream_update_replaces_only_matching_scores(tmp_path: Path) -> None:
    source = tmp_path / "plans.ben"
    output = tmp_path / "scores"
    write_source(source, "ben", "standard")
    graph, _ = grid_resources()
    PlanEvaluator(graph).add_metric(
        Tally("population", "area", result_name="district_totals")
    ).add_metric(CutEdges()).evaluate_stream(source, output)
    cut_edges = (output / "cut_edges__scores.parquet").read_bytes()

    run = (
        PlanEvaluator(graph)
        .add_metric(Tally("population", result_name="district_totals"))
        .evaluate_stream(source, output, update=True)
    )

    assert run.metrics == ("district_totals", "cut_edges")
    assert (output / "district_totals" / "population_tallies__scores.parquet").is_file()
    assert not (output / "district_totals" / "area_tallies__scores.parquet").exists()
    assert (output / "cut_edges__scores.parquet").read_bytes() == cut_edges
    assert run._metric_metadata["district_totals"].subkeys == ("population",)


def test_evaluate_stream_rolls_back_tables_when_manifest_publication_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "plans.ben"
    output = tmp_path / "scores"
    write_source(source, "ben", "standard")
    graph, _ = grid_resources()
    PlanEvaluator(graph).add_metric(
        Tally("population", "area", result_name="district_totals")
    ).evaluate_stream(source, output)
    expected = EnsembleEvalResult.open(output).read("district_totals")
    original = {
        path.relative_to(output): path.read_bytes() for path in output.rglob("*") if path.is_file()
    }

    def fail_manifest(*args, **kwargs):
        raise OSError("injected manifest failure")

    monkeypatch.setattr(evaluator_module, "_write_run_manifest", fail_manifest)
    with pytest.raises(OSError, match="injected manifest failure"):
        PlanEvaluator(graph).add_metric(
            Tally("population", result_name="district_totals")
        ).evaluate_stream(source, output, update=True)

    assert {
        path.relative_to(output): path.read_bytes() for path in output.rglob("*") if path.is_file()
    } == original
    pd.testing.assert_frame_equal(EnsembleEvalResult.open(output).read("district_totals"), expected)


def test_evaluate_stream_does_not_overwrite_an_untracked_score_path(tmp_path: Path) -> None:
    source = tmp_path / "plans.ben"
    output = tmp_path / "scores"
    write_source(source, "ben", "standard")
    graph, _ = grid_resources()
    PlanEvaluator(graph).add_metric(Tally("population")).evaluate_stream(source, output)
    marker = output / "cut_edges__scores.parquet"
    marker.write_bytes(b"untracked")
    manifest = run_manifest(output).read_bytes()

    with pytest.raises(FileExistsError, match="untracked"):
        PlanEvaluator(graph).add_metric(CutEdges()).evaluate_stream(source, output)

    assert marker.read_bytes() == b"untracked"
    assert run_manifest(output).read_bytes() == manifest


def test_evaluate_stream_rejects_new_scores_with_different_run_dimensions(
    tmp_path: Path,
) -> None:
    source = tmp_path / "plans.ben"
    shorter_source = tmp_path / "shorter.ben"
    output = tmp_path / "scores"
    write_source(source, "ben", "standard")
    with BenEncoder(shorter_source, variant="standard") as stream:
        stream.write([0, 0, 1, 1])
    graph, _ = grid_resources()
    PlanEvaluator(graph).add_metric(Tally("population")).evaluate_stream(source, output)
    manifest = run_manifest(output).read_bytes()

    with pytest.raises(ValueError, match="run dimensions"):
        PlanEvaluator(graph).add_metric(CutEdges()).evaluate_stream(shorter_source, output)

    assert run_manifest(output).read_bytes() == manifest
    assert not (output / "cut_edges__scores.parquet").exists()


def test_evaluate_stream_adds_scores_to_a_version_one_run(tmp_path: Path) -> None:
    source = tmp_path / "plans.ben"
    output = tmp_path / "scores"
    write_source(source, "ben", "standard")
    graph, _ = grid_resources()
    PlanEvaluator(graph).add_metric(CutEdges()).evaluate_stream(source, output)

    current_manifest = run_manifest(output)
    manifest = json.loads(current_manifest.read_text())
    table = manifest["metrics"][0]["tables"][0]
    legacy_table = output / "cut_edges" / "scores.parquet"
    legacy_table.parent.mkdir()
    (output / table["path"]).rename(legacy_table)
    metric = manifest["metrics"][0]
    metric["table"] = "cut_edges/scores.parquet"
    metric["table_size"] = table["size"]
    metric["table_sha256"] = table["sha256"]
    del metric["tables"]
    manifest["format_version"] = 1
    (output / "manifest.json").write_text(json.dumps(manifest))
    current_manifest.unlink()

    run = PlanEvaluator(graph).add_metric(Tally("population")).evaluate_stream(source, output)

    assert run.metrics == ("cut_edges", "population")
    assert run_manifest(output).is_file()
    assert not (output / "manifest.json").exists()
    assert legacy_table.is_file()
    assert cast(pd.Series, run.read("cut_edges")).tolist() == [2, 2]


def test_evaluation_run_can_open_a_renamed_run_directory(tmp_path: Path) -> None:
    source = tmp_path / "plans.ben"
    output = tmp_path / "original_scores"
    renamed = tmp_path / "renamed_scores"
    write_source(source, "ben", "standard")
    graph, _ = grid_resources()
    PlanEvaluator(graph).add_metric(CutEdges()).evaluate_stream(source, output)
    output.rename(renamed)

    run = EnsembleEvalResult.open(renamed)

    assert run.metrics == ("cut_edges",)
    assert cast(pd.Series, run.read("cut_edges")).tolist() == [2, 2]


def test_evaluate_stream_zero_limit_writes_a_valid_empty_run(tmp_path: Path) -> None:
    source = tmp_path / "plans.ben"
    output = tmp_path / "scores"
    write_source(source, "ben", "standard")

    run = scorer().evaluate_stream(source, output, max_samples=0)

    assert run.summary == EvaluationSummary(samples=0, accepted=0)
    manifest = json.loads(run_manifest(output).read_text())
    assert manifest["summary"] == {
        "samples": 0,
        "accepted": 0,
    }
    assert manifest["district_ids"] == []
    for metric in manifest["metrics"]:
        for table in metric["tables"]:
            assert pq.read_table(output / table["path"]).num_rows == 0
        value = run.read(metric["instance"], expand_repetitions=True)
        assert isinstance(value, (pd.Series, pd.DataFrame))
        assert value.empty
        assert list(run.iter_batches(metric["instance"])) == []
        assert list(run.iter_raw_batches(metric["instance"])) == []
    assert run.frames.empty
    assert list(run.iter_frame_batches()) == []


def test_evaluate_stream_removes_flushed_output_after_a_late_input_error(tmp_path: Path) -> None:
    source = tmp_path / "corrupt.ben"
    output = tmp_path / "scores"
    with BenEncoder(source, variant="standard") as stream:
        for index in range(1_025):
            stream.write([0, index % 2, 1, 1])
    source.write_bytes(source.read_bytes()[:-1])
    graph, _ = grid_resources()

    with pytest.raises((OSError, ValueError), match="buffer|frame"):
        PlanEvaluator(graph).add_metric(Tally("population")).evaluate_stream(
            source, output, batch_size=64
        )

    assert not output.exists()
    assert not list(tmp_path.glob(f".{output.name}.tmp-*"))
