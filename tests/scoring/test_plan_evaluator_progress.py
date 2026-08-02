from pathlib import Path
from typing import Any

import networkx as nx
import pandas as pd
import pytest
from binary_ensemble import BendlEncoder

import gerrytools.scoring.evaluator as evaluator_module
from gerrytools.scoring import PlanEvaluator, Tally


class ProgressRecorder:
    def __init__(self, **options: Any) -> None:
        self.options = options
        self.total = options.get("total")
        self.updates: list[int] = []
        self.closed = False

    def __enter__(self) -> "ProgressRecorder":
        return self

    def __exit__(self, *_: object) -> None:
        self.closed = True

    def update(self, amount: int) -> None:
        self.updates.append(amount)

    def reset(self, *, total: int) -> None:
        self.total = total


def evaluator() -> PlanEvaluator:
    graph = nx.path_graph(3)
    nx.set_node_attributes(graph, {0: 10, 1: 20, 2: 30}, "population")
    return PlanEvaluator(graph).add_metric(Tally("population"))


def record_progress(monkeypatch: pytest.MonkeyPatch) -> list[ProgressRecorder]:
    bars: list[ProgressRecorder] = []

    def create(**options: Any) -> ProgressRecorder:
        bar = ProgressRecorder(**options)
        bars.append(bar)
        return bar

    monkeypatch.setattr(evaluator_module, "tqdm", create)
    return bars


def test_evaluate_many_reports_completed_native_batches(monkeypatch: pytest.MonkeyPatch) -> None:
    bars = record_progress(monkeypatch)
    plans = [[0, 0, 1] for _ in range(300)]

    result = evaluator().evaluate_many(plans, progress=True)
    population = result["population"]

    assert isinstance(population, pd.DataFrame)
    assert len(population) == 300
    assert len(bars) == 1
    assert bars[0].options == {
        "total": 300,
        "desc": "Evaluating plans",
        "unit": "plan",
    }
    assert bars[0].updates == [256, 44]
    assert bars[0].closed


@pytest.mark.parametrize(("max_samples", "expected"), [(None, 3), (2, 2), (10, 3)])
def test_evaluate_stream_reports_bendl_progress_without_a_second_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    max_samples: int | None,
    expected: int,
) -> None:
    source = tmp_path / "plans.bendl"
    output = tmp_path / "scores"
    encoder = BendlEncoder(source)
    with encoder.ben_stream(variant="mkv_chain") as stream:
        stream.write([0, 0, 1])
        stream.write([0, 0, 1])
        stream.write([0, 1, 1])
    bars = record_progress(monkeypatch)

    run = evaluator().evaluate_stream(
        source,
        output,
        max_samples=max_samples,
        progress=True,
    )

    assert run.summary.samples == expected
    assert len(bars) == 1
    assert bars[0].options == {
        "total": max_samples,
        "desc": "Evaluating ensemble",
        "unit": "sample",
    }
    assert bars[0].total == expected
    assert sum(bars[0].updates) == expected
    assert bars[0].closed


@pytest.mark.parametrize("error", [KeyboardInterrupt("ctrl-c"), RuntimeError("boom")])
def test_evaluate_stream_preserves_progress_callback_exception(
    error: BaseException,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "plans.bendl"
    encoder = BendlEncoder(source)
    with encoder.ben_stream(variant="mkv_chain") as stream:
        stream.write([0, 0, 1])

    class RaisingProgress(ProgressRecorder):
        def update(self, amount: int) -> None:
            super().update(amount)
            raise error

    bar = RaisingProgress()
    monkeypatch.setattr(evaluator_module, "tqdm", lambda **_: bar)

    with pytest.raises(type(error), match=str(error)):
        evaluator().evaluate_stream(source, tmp_path / "scores", progress=True)

    assert bar.closed


def test_progress_must_be_boolean(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="progress must be a boolean"):
        evaluator().evaluate_many(
            [[0, 0, 1]],
            progress=1,  # type: ignore
        )
    with pytest.raises(TypeError, match="progress must be a boolean"):
        evaluator().evaluate_stream(
            tmp_path / "missing.ben",
            tmp_path / "scores",
            progress=1,  # type: ignore
        )
