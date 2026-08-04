"""Docker-free tests for RunContainer's output processing."""

import os
from typing import Any, cast

import docker.errors
import networkx as nx
import pytest
from gerrychain import Graph

from gerrytools.mgrp import RunContainer
from gerrytools.mgrp.run_container import (
    RunnerConfig,
    _preserve_outputs_on_failure,
    _validate_output_file_name,
)
from tests.mgrp.conftest import make_fake_run_container


def make_container_and_graph():
    # The factory skips __init__ so no Docker client is needed; _process_output
    # takes the graph as an argument.
    graph = nx.Graph()
    graph.add_nodes_from([0, 1, 2, 3])
    graph.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 0)])
    for node in graph.nodes:
        graph.nodes[node]["TOTPOP"] = 1
    return make_fake_run_container(), Graph.from_networkx(graph)


def test_process_output_yields_independent_dicts():
    # Regression: results used to share one mutated updater_values dict, so
    # collecting them into a list showed only the final step's values.
    container, graph = make_container_and_graph()
    updaters = {"num_cut_edges": lambda part: len(part["cut_edges"])}

    results = []
    for line in [
        {"assignment": [1, 1, 2, 2], "sample": 1},  # cuts edges (1,2) and (3,0)
        {"assignment": [1, 2, 1, 2], "sample": 2},  # cuts all four edges
    ]:
        results.extend(container._process_output(graph, line, updaters))

    (first, first_err), (second, second_err) = results
    assert first["updaters"] is not second["updaters"]
    assert first == {"sample": 1, "updaters": {"num_cut_edges": 2}}
    assert second == {"sample": 2, "updaters": {"num_cut_edges": 4}}
    assert first_err is None and second_err is None


def test_process_output_registers_user_updater_dependencies():
    container, graph = make_container_and_graph()
    updaters = {
        "district_count": lambda partition: len(partition.parts),
        "double_district_count": lambda partition: 2 * partition["district_count"],
    }

    [(result, error)] = container._process_output(
        graph,
        {"assignment": [1, 1, 2, 2], "sample": 1},
        updaters,
    )

    assert result == {
        "sample": 1,
        "updaters": {"district_count": 2, "double_district_count": 4},
    }
    assert error is None


def test_rejects_non_runner_configuration():
    # The type check runs before any Docker connection is attempted.
    bad_config = cast(RunnerConfig, "not a config")
    with pytest.raises(TypeError, match="RecomRunnerConfig"):
        RunContainer(configuration=bad_config)


def test_output_file_name_rejects_windows_drive_relative_path():
    with pytest.raises(ValueError, match="bare file name"):
        _validate_output_file_name("C:unrelated.jsonl")


def test_mcmc_run_with_updaters_rejects_incompatible_run_info():
    container = make_fake_run_container()

    with pytest.raises(TypeError, match="does not carry updaters"):
        list(container.mcmc_run_with_updaters(object()))


def test_methods_raise_outside_with_block():
    container = make_fake_run_container()
    with pytest.raises(RuntimeError, match="with"):
        container._running_container()


def test_client_access_raises_outside_active_context():
    container = make_fake_run_container()
    with pytest.raises(RuntimeError, match="outside an active context"):
        container._running_client()


def test_process_output_passes_error_through():
    container, graph = make_container_and_graph()
    [(result, error)] = container._process_output(
        graph, {"assignment": [1, 1, 2, 2], "sample": 7}, {}, error="boom"
    )
    assert result == {"sample": 7, "updaters": {}}
    assert error == "boom"


def test_process_output_wraps_schema_mismatched_lines_in_runtime_error():
    # A metadata-shaped line (no assignment/sample) used to surface a bare KeyError.
    container, graph = make_container_and_graph()
    metadata_line = {"metadata": {"engine": "rustrecom", "version": "0.1.4"}}

    with pytest.raises(RuntimeError, match="'assignment'.*rustrecom"):
        list(container._process_output(graph, metadata_line, {}))


@pytest.mark.parametrize(
    ("line", "message"),
    [
        (["not", "an", "object"], "JSON object"),
        ({"assignment": "oops", "sample": 1}, "assignment.*list"),
        ({"assignment": [1, 1, 2], "sample": 1}, "assignment.*4 entries"),
        ({"assignment": [1, None, 2, 2], "sample": 1}, "assignment.*integers"),
        ({"assignment": [1, 1, 2, 2], "sample": True}, "sample.*integer"),
    ],
)
def test_process_output_rejects_malformed_canonical_values(line, message):
    container, graph = make_container_and_graph()

    with pytest.raises(RuntimeError, match=message):
        list(container._process_output(graph, line, {}))


# ==================================
# == STREAM DECODING / FINAL JSON ==
# ==================================


class _FakeExecAPI:
    """Minimal docker-py low-level API: one exec with a canned stream and exit code."""

    def __init__(self, chunks, exit_code=0):
        # Kept lazy so generator chunks can simulate an engine writing files mid-run.
        self.stream = _FakeStream(chunks)
        self.exit_code = exit_code

    def exec_create(self, container_id, cmd, **kwargs):
        return {"Id": "exec-id"}

    def exec_start(self, exec_id, **kwargs):
        return self.stream

    def exec_inspect(self, exec_id):
        return {"ExitCode": self.exit_code}


class _FakeStream:
    def __init__(self, chunks):
        self._chunks = iter(chunks)
        self.closed = False

    def __iter__(self):
        return self

    def __next__(self):
        return next(self._chunks)

    def close(self):
        self.closed = True


class _FakeContainer:
    id = "fake-container"

    def __init__(self):
        self.removed = False
        self.remove_error: Exception | None = None
        self.events: list[tuple[str, object]] = []

    def exec_run(self, cmd):
        self.events.append(("exec_run", cmd))
        return 0, b""

    def remove(self, *, force=False):
        assert force
        self.events.append(("remove", force))
        if self.remove_error is not None:
            raise self.remove_error
        self.removed = True


class _FakeConfig(RunnerConfig):
    """Bypasses RunnerConfig.__init__: fixed engine name, log path, and optional output path."""

    engine = "fake-engine"
    container_output_dir = "/output"

    def __init__(self, log_path, output_path=None):
        self._log_path = log_path
        self._output_path = output_path

    def _stem(self, run_info):
        return "fake"

    def _output_name(self, run_info):
        return None if self._output_path is None else self._output_path.name

    def _base_config(self, run_info):
        return {}

    def run_command(self, run_info):
        return ["cmd"]

    def log_file(self, run_info):
        return str(self._log_path)

    def output_file(self, run_info):
        return None if self._output_path is None else str(self._output_path)


def make_streaming_container(chunks, exit_code=0, config=None):
    """A RunContainer wired to a fake Docker client; chunks must be one-sided
    (stdout, None) / (None, stderr) frames, matching docker-py's demux contract.
    """
    from types import SimpleNamespace

    return make_fake_run_container(
        config=(
            config
            if config is not None
            else SimpleNamespace(
                engine="fake-engine",
                container_output_dir="/output",
                run_command=lambda run_info: ["cmd"],
            )
        ),
        client=SimpleNamespace(api=_FakeExecAPI(chunks, exit_code)),
        container=_FakeContainer(),
    )


def test_iter_json_lines_reassembles_multibyte_characters_split_across_frames():
    # Regression: chunks were decoded independently, so a UTF-8 character split across two
    # Docker frames raised UnicodeDecodeError.
    container = make_streaming_container([(b'{"name": "caf\xc3', None), (b'\xa9"}\n', None)])
    results = list(container._iter_json_lines(["cmd"]))
    assert results == [({"name": "café"}, None)]


def test_iter_json_lines_parses_final_json_line_without_trailing_newline():
    # Regression: a non-blank residual stdout buffer was silently dropped after the loop.
    container = make_streaming_container([(b'{"sample": 1}\n{"sample": 2}', None)])
    results = list(container._iter_json_lines(["cmd"]))
    assert results == [({"sample": 1}, None), ({"sample": 2}, None)]


def test_iter_json_lines_preserves_invalid_stderr_bytes():
    container = make_streaming_container([(None, b"err caf\xc3")])
    assert list(container._iter_json_lines(["cmd"])) == [
        (None, "err caf"),
        (None, "\\xc3"),
    ]


def test_iter_json_lines_raises_on_invalid_final_json():
    container = make_streaming_container([(b'{"sample": 1}\n{"bad', None)])
    iterator = container._iter_json_lines(["cmd"])
    assert next(iterator) == ({"sample": 1}, None)
    with pytest.raises(RuntimeError, match="JSON"):
        next(iterator)


def test_run_decodes_multibyte_characters_split_across_frames(tmp_path, capsys):
    log_path = tmp_path / "run.log"
    container = make_streaming_container(
        [(b"out caf\xc3", None), (None, b"err caf\xc3"), (b"\xa9\n", None), (None, b"\xa9")],
        config=_FakeConfig(log_path),
    )

    assert container.run(cast(Any, None)) is None
    assert capsys.readouterr().out == "out café\n"
    assert log_path.read_text() == "err café"


def test_run_preserves_invalid_stderr_bytes(tmp_path):
    log_path = tmp_path / "run.log"
    container = make_streaming_container([(None, b"invalid \xff")], config=_FakeConfig(log_path))

    assert container.run(cast(Any, None)) is None
    assert log_path.read_text() == "invalid \\xff"


def test_closing_stream_terminates_the_engine_container():
    container = make_streaming_container([(b'{"sample": 1}\n', None), (b'{"sample": 2}\n', None)])
    api = cast(_FakeExecAPI, cast(Any, container.client).api)
    docker_container = cast(_FakeContainer, container.container)
    iterator = container.run_iter(cast(Any, None))
    assert next(iterator) == ({"sample": 1}, None)

    iterator.close()

    assert api.stream.closed
    assert docker_container.removed
    assert container.container is None
    if hasattr(os, "getuid"):
        assert [event[0] for event in docker_container.events] == ["exec_run", "remove"]


def test_stream_parse_error_survives_container_removal_failure():
    container = make_streaming_container([(b"not-json\n", None)])
    docker_container = cast(_FakeContainer, container.container)
    docker_container.remove_error = OSError("daemon unavailable")

    with pytest.raises(RuntimeError, match="parse container output") as excinfo:
        list(container.run_iter(cast(Any, None)))

    assert any("Could not stop the abandoned" in note for note in excinfo.value.__notes__)
    assert container.container is docker_container


# ==================================
# == EXEC EXIT STATUS / OUTPUTS   ==
# ==================================


def test_run_raises_on_nonzero_exit(tmp_path):
    # Regression: the exec's exit status was never inspected, so failing engine commands
    # looked like successful runs.
    log_path = tmp_path / "run.log"
    container = make_streaming_container(
        [(None, b"boom\n")], exit_code=2, config=_FakeConfig(log_path)
    )

    with pytest.raises(RuntimeError, match="fake-engine.*exit code 2"):
        container.run(cast(Any, None))
    # stderr still lands in the log before the failure is raised.
    assert log_path.read_text() == "boom\n"


def test_run_preserves_previous_output_on_nonzero_exit(tmp_path):
    output_path = tmp_path / "out.jsonl"
    output_path.write_text("previous\n")
    container = make_streaming_container(
        [(None, b"boom\n")],
        exit_code=2,
        config=_FakeConfig(tmp_path / "run.log", output_path),
    )

    with pytest.raises(RuntimeError, match="fake-engine.*exit code 2"):
        container.run(cast(Any, None))
    assert output_path.read_text() == "previous\n"
    assert not list(tmp_path.glob(".gerrytools-backup-*"))


def test_run_aborts_container_and_restores_output_on_stream_failure(tmp_path):
    output_path = tmp_path / "out.jsonl"
    output_path.write_text("previous\n")

    def engine_chunks():
        output_path.write_text("partial\n")
        yield (b"started\n", None)
        raise RuntimeError("stream failed")

    container = make_streaming_container(
        engine_chunks(), config=_FakeConfig(tmp_path / "run.log", output_path)
    )
    api = cast(_FakeExecAPI, cast(Any, container.client).api)
    docker_container = cast(_FakeContainer, container.container)

    with pytest.raises(RuntimeError, match="stream failed"):
        container.run(cast(Any, None))

    assert api.stream.closed
    assert docker_container.removed
    assert container.container is None
    assert output_path.read_text() == "previous\n"
    assert (tmp_path / "out.jsonl.partial").read_text() == "partial\n"


def test_run_checks_nonzero_exit_before_decoder_flush(tmp_path):
    container = make_streaming_container(
        [(None, b"truncated \xc3")],
        exit_code=2,
        config=_FakeConfig(tmp_path / "run.log"),
    )

    with pytest.raises(RuntimeError, match="fake-engine.*exit code 2"):
        container.run(cast(Any, None))


def test_run_iter_raises_on_nonzero_exit_at_exhaustion():
    container = make_streaming_container([(b'{"sample": 1}\n', None)], exit_code=3)
    iterator = container.run_iter(cast(Any, None))

    assert next(iterator) == ({"sample": 1}, None)
    with pytest.raises(RuntimeError, match="fake-engine.*exit code 3"):
        next(iterator)


def test_run_rejects_stale_output_from_a_previous_run(tmp_path):
    # A leftover file must not satisfy the success check, but a failed rerun must preserve it.
    output_path = tmp_path / "out.jsonl"
    output_path.write_text("stale\n")
    container = make_streaming_container(
        [(b"done\n", None)], config=_FakeConfig(tmp_path / "run.log", output_path)
    )

    with pytest.raises(RuntimeError, match="out.jsonl"):
        container.run(cast(Any, None))
    assert output_path.read_text() == "stale\n"


def test_run_returns_output_file_written_during_the_run(tmp_path):
    output_path = tmp_path / "out.jsonl"
    output_path.write_text("stale\n")

    def engine_chunks():
        # The stale leftover is cleared before the engine starts; this write is the run's.
        assert not output_path.exists()
        output_path.write_text("{}\n")
        yield (b"done\n", None)

    container = make_streaming_container(
        engine_chunks(), config=_FakeConfig(tmp_path / "run.log", output_path)
    )

    assert container.run(cast(Any, None)) == str(output_path)
    assert output_path.read_text() == "{}\n"


def test_run_raises_when_output_file_missing_after_zero_exit(tmp_path):
    output_path = tmp_path / "out.jsonl"
    container = make_streaming_container(
        [(b"done\n", None)], config=_FakeConfig(tmp_path / "run.log", output_path)
    )

    with pytest.raises(RuntimeError, match="out.jsonl"):
        container.run(cast(Any, None))


def test_run_rejects_empty_output_and_restores_previous_file(tmp_path):
    output_path = tmp_path / "out.jsonl"
    output_path.write_text("previous\n")

    def engine_chunks():
        output_path.touch()
        yield (b"done\n", None)

    container = make_streaming_container(
        engine_chunks(), config=_FakeConfig(tmp_path / "run.log", output_path)
    )

    with pytest.raises(RuntimeError, match="nonempty.*out.jsonl"):
        container.run(cast(Any, None))
    assert output_path.read_text() == "previous\n"
    assert (tmp_path / "out.jsonl.partial").read_text() == ""
    assert not list(tmp_path.glob(".gerrytools-backup-*"))


def test_run_raises_when_promised_sidecar_missing_after_zero_exit(tmp_path):
    output_path = tmp_path / "out.jsonl"
    scores_path = tmp_path / "out_scores.csv"
    output_path.write_text("old output\n")
    scores_path.write_text("old scores\n")

    class _SidecarConfig(_FakeConfig):
        def expected_files(self, run_info):
            return super().expected_files(run_info) + [str(scores_path)]

    def engine_chunks():
        output_path.write_text("{}\n")  # primary output only; the sidecar never appears
        yield (b"done\n", None)

    container = make_streaming_container(
        engine_chunks(), config=_SidecarConfig(tmp_path / "run.log", output_path)
    )

    with pytest.raises(RuntimeError, match="out_scores.csv"):
        container.run(cast(Any, None))
    assert output_path.read_text() == "old output\n"
    assert scores_path.read_text() == "old scores\n"
    assert (tmp_path / "out.jsonl.partial").read_text() == "{}\n"


def test_failed_first_run_preserves_partial_output(tmp_path):
    output_path = tmp_path / "out.jsonl"

    def engine_chunks():
        output_path.write_text("partial\n")
        yield (None, b"boom\n")

    container = make_streaming_container(
        engine_chunks(),
        exit_code=2,
        config=_FakeConfig(tmp_path / "run.log", output_path),
    )

    with pytest.raises(RuntimeError, match="exit code 2") as excinfo:
        container.run(cast(Any, None))

    partial = tmp_path / "out.jsonl.partial"
    assert not output_path.exists()
    assert partial.read_text() == "partial\n"
    assert str(partial) in "\n".join(excinfo.value.__notes__)


def test_failed_rerun_restores_previous_log(tmp_path):
    # The log is under the same preservation policy as outputs: a rerun must not
    # leave a failed attempt's log in place of the previous successful run's.
    log_path = tmp_path / "run.log"
    log_path.write_text("previous stderr\n")
    container = make_streaming_container(
        [(None, b"boom\n")], exit_code=2, config=_FakeConfig(log_path)
    )

    with pytest.raises(RuntimeError, match="exit code 2"):
        container.run(cast(Any, None))
    assert log_path.read_text() == "previous stderr\n"
    assert not list(tmp_path.glob(".gerrytools-backup-*"))


def test_failed_first_run_keeps_its_log(tmp_path):
    # With no previous log to restore, the failed run's stderr capture is the
    # primary diagnostic and must survive.
    log_path = tmp_path / "run.log"
    container = make_streaming_container(
        [(None, b"boom\n")], exit_code=2, config=_FakeConfig(log_path)
    )

    with pytest.raises(RuntimeError, match="exit code 2"):
        container.run(cast(Any, None))
    assert log_path.read_text() == "boom\n"


def test_successful_rerun_replaces_previous_log(tmp_path):
    log_path = tmp_path / "run.log"
    log_path.write_text("previous stderr\n")
    container = make_streaming_container([(None, b"fresh stderr\n")], config=_FakeConfig(log_path))

    assert container.run(cast(Any, None)) is None
    assert log_path.read_text() == "fresh stderr\n"
    assert not list(tmp_path.glob(".gerrytools-backup-*"))


# ==================================
# == OUTPUT PRESERVATION EDGES    ==
# ==================================


def test_preserve_outputs_rejects_directory_output_path(tmp_path):
    (tmp_path / "out.jsonl").mkdir()

    with pytest.raises(IsADirectoryError, match="not a file"):
        with _preserve_outputs_on_failure([str(tmp_path / "out.jsonl")]):
            pytest.fail("the run body must not start")


def test_preserve_outputs_reports_unrestorable_backups(tmp_path):
    # When restoring the previous outputs itself fails, both the run's error and the
    # recovery errors surface together instead of one masking the other.
    output_path = tmp_path / "out.jsonl"
    output_path.write_text("previous\n")

    with pytest.raises(BaseExceptionGroup, match="could not be fully restored") as excinfo:
        with _preserve_outputs_on_failure([str(output_path)]):
            backup_dir = next(tmp_path.glob(".gerrytools-backup-*"))
            (backup_dir / "out.jsonl").unlink()  # sabotage the backup mid-run
            raise RuntimeError("run failed")

    grouped = excinfo.value.exceptions
    assert any(str(error) == "run failed" for error in grouped)
    assert any(isinstance(error, FileNotFoundError) for error in grouped)


# ==================================
# == UPDATER STREAM WIRING        ==
# ==================================


def make_updater_container(tmp_path, monkeypatch, graph, chunks):
    """A RunContainer on a real RecomRunnerConfig with Graph.from_json faked to `graph`."""
    from gerrytools.mgrp import RecomRunnerConfig

    monkeypatch.setattr(Graph, "from_json", staticmethod(lambda _path: Graph.from_networkx(graph)))
    runner = RecomRunnerConfig(
        "./graphs/testing.json", output_folder=str(tmp_path), log_folder=str(tmp_path)
    )
    return make_streaming_container(chunks, config=runner)


def test_mcmc_run_with_updaters_streams_assignments_and_stderr(tmp_path, monkeypatch):
    # End-to-end wiring: JSON assignment lines flow through _iter_json_lines into
    # _process_output, and stderr chunks pass through as (None, text).
    from gerrytools.mgrp import RecomRunInfo

    square_graph = nx.Graph()
    square_graph.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 0)])
    container = make_updater_container(
        tmp_path,
        monkeypatch,
        square_graph,
        [
            (None, b"engine warming up\n"),
            (b'{"assignment": [1, 1, 2, 2], "sample": 1}\n', None),
        ],
    )
    run_info = RecomRunInfo(
        pop_col="TOTPOP",
        assignment_col="CD",
        variant="A",
        updaters={"num_cut_edges": lambda partition: len(partition["cut_edges"])},
    )

    results = list(container.mcmc_run_with_updaters(run_info))

    assert results == [
        (None, "engine warming up\n"),
        ({"sample": 1, "updaters": {"num_cut_edges": 2}}, None),
    ]


def test_closing_updater_stream_terminates_the_engine_container(tmp_path, monkeypatch):
    from gerrytools.mgrp import RecomRunInfo

    graph = nx.cycle_graph(4)
    container = make_updater_container(
        tmp_path,
        monkeypatch,
        graph,
        [
            (b'{"assignment": [1, 1, 2, 2], "sample": 1}\n', None),
            (b'{"assignment": [1, 2, 1, 2], "sample": 2}\n', None),
        ],
    )
    docker_container = cast(_FakeContainer, container.container)
    iterator = container.mcmc_run_with_updaters(
        RecomRunInfo(
            pop_col="TOTPOP",
            assignment_col="CD",
            variant="A",
            updaters={"district_count": lambda partition: len(partition.parts)},
        )
    )
    first, _ = next(iterator)
    assert first is not None
    assert first["sample"] == 1

    iterator.close()

    assert docker_container.removed
    assert container.container is None


def test_updater_error_survives_container_removal_failure(tmp_path, monkeypatch):
    from gerrytools.mgrp import RecomRunInfo

    def failing_updater(_partition):
        raise ValueError("updater failed")

    container = make_updater_container(
        tmp_path,
        monkeypatch,
        nx.cycle_graph(4),
        [(b'{"assignment": [1, 1, 2, 2], "sample": 1}\n', None)],
    )
    docker_container = cast(_FakeContainer, container.container)
    docker_container.remove_error = OSError("daemon unavailable")
    iterator = container.mcmc_run_with_updaters(
        RecomRunInfo(
            pop_col="TOTPOP",
            assignment_col="CD",
            variant="A",
            updaters={"failure": failing_updater},
        )
    )

    with pytest.raises(ValueError, match="updater failed") as excinfo:
        list(iterator)

    assert any("Could not stop the abandoned" in note for note in excinfo.value.__notes__)
    assert container.container is docker_container


@pytest.mark.parametrize(
    "node_labels",
    [["a", "b", "c"], [1, 2, 3]],
    ids=["string-labels", "shifted-integers"],
)
def test_mcmc_run_with_updaters_rejects_non_positional_node_labels(
    tmp_path, monkeypatch, node_labels
):
    # The engines emit positional assignment lists; string labels used to raise a bare
    # KeyError and permuted integer labels would be silently matched to the wrong nodes.
    from gerrytools.mgrp import RecomRunInfo

    labeled_graph = nx.Graph()
    labeled_graph.add_edges_from(zip(node_labels, node_labels[1:]))
    container = make_updater_container(tmp_path, monkeypatch, labeled_graph, [])
    run_info = RecomRunInfo(pop_col="TOTPOP", assignment_col="CD", variant="A")

    with pytest.raises(RuntimeError, match="node labels to be exactly 0..2"):
        list(container.mcmc_run_with_updaters(run_info))


def test_mcmc_run_with_updaters_rejects_a_permuted_complete_label_set(tmp_path, monkeypatch):
    from gerrytools.mgrp import RecomRunInfo

    permuted = nx.Graph()
    permuted.add_edges_from([(3, 1), (1, 0), (0, 2)])
    assert list(permuted.nodes) == [3, 1, 0, 2]
    assert set(permuted.nodes) == set(range(4))  # A set comparison would pass.

    container = make_updater_container(tmp_path, monkeypatch, permuted, [])
    run_info = RecomRunInfo(pop_col="TOTPOP", assignment_col="CD", variant="A")

    with pytest.raises(RuntimeError, match="ascending order"):
        list(container.mcmc_run_with_updaters(run_info))


# ==================================
# == WARNINGS HYGIENE             ==
# ==================================


def test_importing_mgrp_leaves_unrelated_resource_warnings_observable():
    # Regression: gerrytools.mgrp installed a process-wide
    # filterwarnings("ignore", message="unclosed", category=ResourceWarning) at import time,
    # silencing matching warnings from user code.
    import importlib
    import warnings

    import gerrytools.mgrp

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        importlib.reload(gerrytools.mgrp)
        warnings.warn("unclosed file <fake>", ResourceWarning, stacklevel=1)

    assert any("unclosed" in str(warning.message) for warning in caught)


# =========================
# == CONTAINER LIFECYCLE ==
# =========================


def test_docker_connection_is_deferred_until_context_entry(tmp_path, monkeypatch):
    from gerrytools.mgrp import RecomRunnerConfig

    runner = RecomRunnerConfig(
        "./graphs/testing.json", output_folder=str(tmp_path), log_folder=str(tmp_path)
    )
    attempts = []

    def fail_connection():
        attempts.append(True)
        raise docker.errors.DockerException("not running")

    monkeypatch.setattr("gerrytools.mgrp.run_container.docker.from_env", fail_connection)

    container = RunContainer(runner)
    assert container.client is None
    assert attempts == []

    with pytest.raises(RuntimeError, match="Could not connect to Docker"):
        container.__enter__()
    assert attempts == [True]


def test_failed_context_entry_closes_client(tmp_path):
    from types import SimpleNamespace

    from gerrytools.mgrp import RecomRunnerConfig

    runner = RecomRunnerConfig(
        "./graphs/testing.json", output_folder=str(tmp_path), log_folder=str(tmp_path)
    )
    close_calls = []

    def fail_create(**_kwargs):
        raise RuntimeError

    client = SimpleNamespace(
        close=lambda: close_calls.append(True),
        images=SimpleNamespace(pull=lambda _image: None),
        containers=SimpleNamespace(create=fail_create),
    )
    container = make_fake_run_container(config=runner, client=client)

    with pytest.raises(RuntimeError, match="Could not start"):
        container.__enter__()

    assert close_calls == [True]


def test_failed_start_removes_created_container(tmp_path):
    # Regression: containers.run() does no cleanup when create succeeds but start fails,
    # so every retry accumulated another container stuck in the Created state.
    from types import SimpleNamespace

    from gerrytools.mgrp import RecomRunnerConfig

    runner = RecomRunnerConfig(
        "./graphs/testing.json", output_folder=str(tmp_path), log_folder=str(tmp_path)
    )
    remove_calls = []
    close_calls = []
    created = SimpleNamespace(
        start=lambda: (_ for _ in ()).throw(RuntimeError("start failed")),
        remove=lambda **kwargs: remove_calls.append(kwargs),
    )
    client = SimpleNamespace(
        close=lambda: close_calls.append(True),
        images=SimpleNamespace(pull=lambda _image: None),
        containers=SimpleNamespace(create=lambda **_kwargs: created),
    )
    container = make_fake_run_container(config=runner, client=client)

    with pytest.raises(RuntimeError, match="Could not start the Docker container") as error:
        container.__enter__()

    assert remove_calls == [{"force": True}]
    assert close_calls == [True]
    assert container.container is None
    assert str(error.value.__cause__) == "start failed"


def raise_keyboard_interrupt(*_args, **_kwargs):
    raise KeyboardInterrupt


def test_interrupt_during_pull_closes_client(tmp_path):
    # Ctrl-C mid-pull must not fall into the local-image fallback, and must not leak the
    # freshly opened client (__exit__ never runs when __enter__ raises).
    from types import SimpleNamespace

    from gerrytools.mgrp import RecomRunnerConfig

    runner = RecomRunnerConfig(
        "./graphs/testing.json", output_folder=str(tmp_path), log_folder=str(tmp_path)
    )
    close_calls = []
    client = SimpleNamespace(
        close=lambda: close_calls.append(True),
        images=SimpleNamespace(pull=raise_keyboard_interrupt),
        containers=SimpleNamespace(create=lambda **_kwargs: pytest.fail("must not create")),
    )
    container = make_fake_run_container(config=runner, client=client)

    with pytest.raises(KeyboardInterrupt):
        container.__enter__()

    assert close_calls == [True]
    assert container.client is None


def test_interrupt_between_create_and_start_removes_container(tmp_path):
    # Regression: the create/start recovery caught only Exception, so a KeyboardInterrupt
    # after create() leaked the created container and the open client.
    from types import SimpleNamespace

    from gerrytools.mgrp import RecomRunnerConfig

    runner = RecomRunnerConfig(
        "./graphs/testing.json", output_folder=str(tmp_path), log_folder=str(tmp_path)
    )
    remove_calls = []
    close_calls = []
    created = SimpleNamespace(
        start=raise_keyboard_interrupt,
        remove=lambda **kwargs: remove_calls.append(kwargs),
    )
    client = SimpleNamespace(
        close=lambda: close_calls.append(True),
        images=SimpleNamespace(pull=lambda _image: None),
        containers=SimpleNamespace(create=lambda **_kwargs: created),
    )
    container = make_fake_run_container(config=runner, client=client)

    # The interrupt propagates as itself, not wrapped in the RuntimeError.
    with pytest.raises(KeyboardInterrupt):
        container.__enter__()

    assert remove_calls == [{"force": True}]
    assert close_calls == [True]
    assert container.container is None


def test_interrupt_during_create_closes_client(tmp_path):
    from types import SimpleNamespace

    from gerrytools.mgrp import RecomRunnerConfig

    runner = RecomRunnerConfig(
        "./graphs/testing.json", output_folder=str(tmp_path), log_folder=str(tmp_path)
    )
    close_calls = []
    client = SimpleNamespace(
        close=lambda: close_calls.append(True),
        images=SimpleNamespace(pull=lambda _image: None),
        containers=SimpleNamespace(create=raise_keyboard_interrupt),
    )
    container = make_fake_run_container(config=runner, client=client)

    with pytest.raises(KeyboardInterrupt):
        container.__enter__()

    assert close_calls == [True]
    assert container.container is None


def test_input_mount_is_read_only(tmp_path):
    from gerrytools.mgrp import RecomRunnerConfig

    input_path = tmp_path / "input" / "graph.json"
    runner = RecomRunnerConfig(str(input_path), output_folder=str(tmp_path / "output"))

    volumes = runner.configure_volumes()["volumes"]

    assert volumes[str(input_path.parent.resolve())]["mode"] == "ro"
    assert volumes[str((tmp_path / "output" / "graph").resolve())]["mode"] == "rw"


def test_container_configuration_has_no_fixed_name(tmp_path):
    from types import SimpleNamespace

    from gerrytools.mgrp import RecomRunnerConfig

    runner = RecomRunnerConfig(
        "./graphs/testing.json", output_folder=str(tmp_path), log_folder=str(tmp_path)
    )
    create_kwargs = {}

    class _FakeContainers:
        def create(self, **kwargs):
            create_kwargs.update(kwargs)
            return SimpleNamespace(name="docker-generated", start=lambda: None)

    class _FakeImages:
        def pull(self, _image):
            return None

    container = make_fake_run_container(
        config=runner, client=SimpleNamespace(containers=_FakeContainers(), images=_FakeImages())
    )

    container.__enter__()

    # Docker generates the name so concurrent runs and post-crash reruns never collide;
    # removal is explicit in __exit__, not auto_remove.
    assert "name" not in create_kwargs
    assert "auto_remove" not in create_kwargs
    assert create_kwargs["network_mode"] == "none"


def test_pull_failure_falls_back_to_the_local_image(tmp_path, caplog):
    from types import SimpleNamespace

    from gerrytools.mgrp import RecomRunnerConfig

    runner = RecomRunnerConfig(
        "./graphs/testing.json", output_folder=str(tmp_path), log_folder=str(tmp_path)
    )
    create_calls = []
    local_container = SimpleNamespace(name="local-image", start=lambda: None)
    client = SimpleNamespace(
        images=SimpleNamespace(pull=lambda _image: (_ for _ in ()).throw(RuntimeError("offline"))),
        containers=SimpleNamespace(
            create=lambda **kwargs: create_calls.append(kwargs) or local_container
        ),
    )
    container = make_fake_run_container(config=runner, client=client)

    with caplog.at_level("WARNING"):
        assert container.__enter__() is container

    assert container.container is local_container
    assert len(create_calls) == 1
    assert "Attempting to run using a local copy" in caplog.text


def test_chown_failure_is_only_a_warning(tmp_path, caplog):
    from types import SimpleNamespace

    container = make_fake_run_container(
        config=SimpleNamespace(container_output_dir=str(tmp_path)),
        container=SimpleNamespace(
            exec_run=lambda _cmd: (_ for _ in ()).throw(RuntimeError("permission denied"))
        ),
    )

    with caplog.at_level("WARNING"):
        container._chown_outputs()

    if hasattr(__import__("os"), "getuid"):
        assert "Could not restore ownership" in caplog.text


def test_chown_nonzero_exit_is_only_a_warning(tmp_path, caplog):
    from types import SimpleNamespace

    container = make_fake_run_container(
        config=SimpleNamespace(container_output_dir=str(tmp_path)),
        container=SimpleNamespace(exec_run=lambda _cmd: (1, b"read-only file system")),
    )

    with caplog.at_level("WARNING"):
        container._chown_outputs()

    if hasattr(__import__("os"), "getuid"):
        assert "exit code 1" in caplog.text
        assert "read-only file system" in caplog.text


def make_exiting_container(tmp_path, remove):
    """A RunContainer ready for __exit__, recording chown execs and client closes."""
    from types import SimpleNamespace

    from gerrytools.mgrp import RecomRunnerConfig

    runner = RecomRunnerConfig(
        "./graphs/testing.json", output_folder=str(tmp_path), log_folder=str(tmp_path)
    )
    exec_calls = []
    close_calls = []
    container = make_fake_run_container(
        config=runner,
        client=SimpleNamespace(close=lambda: close_calls.append(True)),
        container=SimpleNamespace(
            remove=remove,
            exec_run=lambda cmd: exec_calls.append(cmd) or (0, b""),
        ),
    )
    return container, exec_calls, close_calls


def test_context_exit_requests_removal_exactly_once(tmp_path):
    remove_calls = []
    container, exec_calls, close_calls = make_exiting_container(
        tmp_path, remove=lambda **kwargs: remove_calls.append(kwargs)
    )

    assert container.__exit__(None, None, None) is False

    assert remove_calls == [{"force": True}]
    assert close_calls == [True]
    assert container.container is None
    if hasattr(__import__("os"), "getuid"):
        # Ownership repair runs before removal so chown can still reach the files.
        assert exec_calls and exec_calls[0][0] == "chown"


def test_context_exit_with_pending_exception_still_cleans_up(tmp_path):
    remove_calls = []
    container, exec_calls, close_calls = make_exiting_container(
        tmp_path, remove=lambda **kwargs: remove_calls.append(kwargs)
    )
    body_error = RuntimeError("body failed")

    # Returning False lets the with-body's exception propagate normally.
    assert container.__exit__(RuntimeError, body_error, None) is False

    assert remove_calls == [{"force": True}]
    assert close_calls == [True]
    assert container.container is None
    if hasattr(__import__("os"), "getuid"):
        assert exec_calls and exec_calls[0][0] == "chown"


@pytest.mark.skipif(not hasattr(os, "getuid"), reason="chown runs only on POSIX")
def test_context_exit_interrupt_during_chown_still_removes_container(tmp_path):
    remove_calls = []
    container, _exec_calls, close_calls = make_exiting_container(
        tmp_path, remove=lambda **kwargs: remove_calls.append(kwargs)
    )
    cast(Any, container.container).exec_run = lambda _cmd: (_ for _ in ()).throw(
        KeyboardInterrupt()
    )

    with pytest.raises(KeyboardInterrupt):
        container.__exit__(None, None, None)

    assert remove_calls == [{"force": True}]
    assert close_calls == [True]
    assert container.container is None


def test_context_exit_remove_failure_is_only_a_warning(tmp_path, caplog):
    # Regression: only docker.errors.APIError was caught, so a dead daemon raising anything
    # else masked the with-body's exception and left a stale container reference behind.
    def failing_remove(**_kwargs):
        raise RuntimeError("daemon gone")

    container, _exec_calls, close_calls = make_exiting_container(tmp_path, remove=failing_remove)

    with caplog.at_level("WARNING"):
        assert container.__exit__(RuntimeError, RuntimeError("body failed"), None) is False

    assert "Error removing Docker container" in caplog.text
    assert close_calls == [True]
    assert container.container is None


def test_context_exit_client_close_failure_is_only_a_warning(tmp_path, caplog):
    container, _exec_calls, _close_calls = make_exiting_container(
        tmp_path, remove=lambda **_kwargs: None
    )

    def failing_close():
        raise RuntimeError("close failed")

    client = container.client
    assert client is not None
    client.close = failing_close

    with caplog.at_level("WARNING"):
        assert container.__exit__(RuntimeError, RuntimeError("body failed"), None) is False

    assert "Error closing Docker client" in caplog.text
    assert container.client is None
