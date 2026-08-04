"""Tests for effective run configuration emission and Docker transport."""

import json
from threading import Lock
from types import SimpleNamespace

import networkx as nx
import pytest
from gerrychain import Graph

from gerrytools.mgrp import (
    Constraints,
    ForestRunInfo,
    ForestRunnerConfig,
    RecomRunInfo,
    RecomRunnerConfig,
    SMCRunInfo,
    SMCRunnerConfig,
)
from tests.mgrp.conftest import make_fake_run_container


class LockingUpdater:
    def __init__(self):
        self.lock = Lock()

    def __call__(self, _partition):
        return 0


def test_recom_effective_config_is_complete():
    runner = RecomRunnerConfig("./graphs/testing.json")
    run_info = RecomRunInfo(
        pop_col="TOTPOP",
        assignment_col="CD",
        variant="A",
        constraint={
            "constraint": "district_share_floor",
            "numerator_col": "BVAP",
            "denominator_cols": ["VAP"],
            "threshold": 0.4,
        },
    )

    config = runner.run_config(run_info)

    # rustrecom's native chain config: flat, CLI-mirroring field names and
    # variant spellings, with a version/command envelope. Derived file names
    # carry the config hash so distinct configs never collide.
    config_hash = runner.config_hash(run_info)
    assert config == {
        "version": 1,
        "command": "chain",
        "graph_json": "/home/recom/shapefiles/testing.json",
        "n_steps": 10,
        "tol": 0.05,
        "pop_col": "TOTPOP",
        "assignment_col": "CD",
        "rng_seed": 42,
        "variant": "cut-edges-mst",
        "target_pop": None,
        "balance_ub": 0,
        "n_threads": 1,
        "batch_size": 1,
        "writer": "canonical",
        "sum_cols": [],
        "region_weights": {},
        "edge_weight_keys": [],
        "cut_edges_count": False,
        "output_file": f"/home/recom/output/testing/RecomA_CD_42_10_{config_hash}.jsonl",
        "bendl_graph_order": "none",
        "show_progress": False,
        "constraint": {
            "constraint": "district_share_floor",
            "numerator_col": "BVAP",
            "denominator_cols": ["VAP"],
            "threshold": 0.4,
        },
    }
    assert json.loads(runner.run_command(run_info)[4]) == config


def test_recom_config_excludes_updaters_before_copying():
    runner = RecomRunnerConfig("./graphs/testing.json")
    updater = LockingUpdater()
    run_info = RecomRunInfo(
        pop_col="TOTPOP",
        assignment_col="CD",
        variant="A",
        updaters={"locked": updater},
    )

    config = json.loads(runner.run_command(run_info)[4])

    assert "updaters" not in config
    assert run_info.updaters["locked"] is updater


def test_forest_effective_config_records_internal_defaults():
    runner = ForestRunnerConfig("./graphs/testing.json")
    run_info = ForestRunInfo(
        levels=["county", "precinct"],
        pop_col="TOTPOP",
    )

    config = runner.run_config(run_info)

    config_hash = runner.config_hash(run_info)
    assert config == {
        "version": 1,
        "engine": "forest",
        "io": {
            "graph": "/home/forest/shapefiles/testing.json",
            "output": (
                f"/home/forest/output/testing/Forest_42_atlas_gamma0.0_10_{config_hash}.jsonl"
            ),
            "writer": "jsonl",
        },
        "run": {
            "levels": ["county", "precinct"],
            "pop_col": "TOTPOP",
            "num_dists": 2,
            "pop_dev": 0.1,
            "gamma": 0.0,
            "n_steps": 10,
            "rng_seed": 42,
            "edge_weights": "connections",
            "output_freq": 1,
        },
        "constraints": [],
    }
    assert json.loads(runner.run_command(run_info)[4]) == config


def test_smc_effective_config_splits_map_and_run_sections():
    runner = SMCRunnerConfig("./shapefiles/testing")
    run_info = SMCRunInfo(pop_col="TOTPOP", n_dists=4, n_sims=20)

    config = runner.run_config(run_info)

    config_hash = runner.config_hash(run_info)
    assert config == {
        "version": 1,
        "engine": "smc",
        "io": {
            "graph": "/home/smc/shapefiles/testing",
            "output": f"/home/smc/output/testing/SMC_42_20_{config_hash}.jsonl",
            "writer": "jsonl",
        },
        "map": {
            "pop_col": "TOTPOP",
            "n_dists": 4,
            "pop_tol": 0.01,
            "pop_bounds": [],
        },
        "run": {
            "n_sims": 20,
            "rng_seed": 42,
            "compactness": 1.0,
            "resample": False,
            "adapt_k_thresh": 0.985,
            "seq_alpha": 0.5,
            "pop_temper": 0.0,
            "final_infl": 1.0,
            "verbose": False,
            "silent": False,
            "tally_columns": [],
        },
        "constraints": [],
    }
    assert json.loads(runner.run_command(run_info)[4]) == config


def test_post_construction_constraint_assignment_reaches_run_config():
    # Regression: the resolved constraint specs were computed once in __post_init__, so a
    # constraint assigned after construction was silently dropped from the emitted config.
    recom_runner = RecomRunnerConfig("./graphs/testing.json")
    recom_info = RecomRunInfo(pop_col="TOTPOP", assignment_col="CD", variant="A")
    recom_info.constraint = {
        "constraint": "district_share_floor",
        "numerator_col": "BVAP",
        "denominator_cols": ["VAP"],
        "threshold": 0.4,
    }
    assert recom_runner.run_config(recom_info).get("constraint") == recom_info.constraint

    forest_runner = ForestRunnerConfig("./graphs/testing.json")
    forest_info = ForestRunInfo(levels=["county", "precinct"], pop_col="TOTPOP")
    forest_info.constraints = Constraints().max_coarse_node_splits(max_splits=8)
    assert forest_runner.run_config(forest_info)["constraints"] == [
        {"constraint": "max_coarse_node_splits", "max_splits": 8}
    ]

    smc_runner = SMCRunnerConfig("./shapefiles/testing")
    smc_info = SMCRunInfo(pop_col="TOTPOP", n_dists=4, n_sims=20)
    smc_info.constraints = Constraints().splits(strength=100.0, admin_col="COUNTY")
    assert smc_runner.run_config(smc_info)["constraints"] == [
        {"constraint": "splits", "strength": 100.0, "admin_col": "COUNTY"}
    ]


def test_post_construction_invalid_fields_are_revalidated_before_config_emission():
    recom_runner = RecomRunnerConfig("./graphs/testing.json")
    recom_info = RecomRunInfo(pop_col="TOTPOP", assignment_col="CD", variant="A")
    recom_info.n_steps = 0
    with pytest.raises(ValueError, match="n_steps"):
        recom_runner.run_config(recom_info)

    forest_runner = ForestRunnerConfig("./graphs/testing.json")
    forest_info = ForestRunInfo(levels=["county"], pop_col="TOTPOP")
    forest_info.levels.clear()
    with pytest.raises(ValueError, match="levels"):
        forest_runner.run_config(forest_info)

    smc_runner = SMCRunnerConfig("./shapefiles/testing")
    smc_info = SMCRunInfo(pop_col="TOTPOP", n_dists=4, n_sims=20)
    smc_info.pop_bounds[:] = [100, 50, 10]
    with pytest.raises(ValueError, match="ordered"):
        smc_runner.run_config(smc_info)


def test_config_hash_separates_configs_that_share_a_stem():
    # Regression: file stems were pure functions of a subset of config fields, so two runs
    # differing only elsewhere (e.g. pop_tol) overwrote each other's outputs and logs.
    runner = RecomRunnerConfig("./graphs/testing.json")
    base = RecomRunInfo(pop_col="TOTPOP", assignment_col="CD", variant="A")
    same = RecomRunInfo(pop_col="TOTPOP", assignment_col="CD", variant="A")
    different = RecomRunInfo(pop_col="TOTPOP", assignment_col="CD", variant="A", pop_tol=0.02)

    assert base.stem() == different.stem()
    assert runner.output_file(base) == runner.output_file(same)  # identical configs reuse paths
    assert runner.output_file(base) != runner.output_file(different)


def test_config_hash_separates_inputs_that_share_a_filename():
    run_info = RecomRunInfo(pop_col="TOTPOP", assignment_col="CD", variant="A")
    first = RecomRunnerConfig("/tmp/dir_a/foo.json")
    second = RecomRunnerConfig("/tmp/dir_b/foo.json")

    assert first.config_hash(run_info) != second.config_hash(run_info)
    assert first.output_file(run_info) != second.output_file(run_info)
    assert first.config_hash(run_info) == RecomRunnerConfig("/tmp/dir_a/foo.json").config_hash(
        run_info
    )


@pytest.mark.parametrize(
    ("runner", "run_info", "mutate"),
    [
        (
            RecomRunnerConfig("./graphs/testing.json"),
            RecomRunInfo(pop_col="TOTPOP", assignment_col="CD", variant="A", sum_cols=["BVAP"]),
            lambda config: config["sum_cols"].append("INJECTED"),
        ),
        (
            ForestRunnerConfig("./graphs/testing.json"),
            ForestRunInfo(levels=["county"], pop_col="TOTPOP"),
            lambda config: config["run"]["levels"].append("INJECTED"),
        ),
        (
            SMCRunnerConfig("./shapefiles/testing"),
            SMCRunInfo(pop_col="TOTPOP", n_dists=4, n_sims=20, pop_bounds=[1, 2, 3]),
            lambda config: config["map"]["pop_bounds"].append(99),
        ),
    ],
    ids=["recom", "forest", "smc"],
)
def test_run_config_does_not_alias_the_run_infos_mutable_fields(runner, run_info, mutate):
    before_hash = runner.config_hash(run_info)
    before_output = runner.output_file(run_info)

    mutate(runner.run_config(run_info))

    assert runner.config_hash(run_info) == before_hash
    assert runner.output_file(run_info) == before_output
    assert "INJECTED" not in json.dumps(runner.run_config(run_info))


@pytest.mark.parametrize(
    ("runner", "run_info", "suffix"),
    [
        (
            ForestRunnerConfig("./graphs/testing.json"),
            ForestRunInfo(levels=["county"], pop_col="TOTPOP", writer="ben"),
            ".jsonl.ben",
        ),
        (
            SMCRunnerConfig("./shapefiles/testing"),
            SMCRunInfo(pop_col="TOTPOP", n_dists=4, n_sims=20, writer="csv"),
            ".csv",
        ),
        (
            RecomRunnerConfig("./graphs/testing.json"),
            RecomRunInfo(
                pop_col="TOTPOP",
                assignment_col="CD",
                variant="A",
                writer="bendl",
            ),
            ".bendl",
        ),
    ],
)
def test_runner_writer_suffixes_are_consistent(runner, run_info, suffix):
    output = runner.output_file(run_info)
    assert output is not None and output.endswith(suffix)


@pytest.mark.parametrize(
    ("runner", "run_info"),
    [
        (
            ForestRunnerConfig("./graphs/testing.json"),
            ForestRunInfo(
                levels=["county"],
                pop_col="TOTPOP",
                writer="ben",
                output_file_name="chosen.output",
            ),
        ),
        (
            SMCRunnerConfig("./shapefiles/testing"),
            SMCRunInfo(
                pop_col="TOTPOP",
                n_dists=4,
                n_sims=20,
                writer="csv",
                output_file_name="chosen.output",
            ),
        ),
    ],
)
def test_runner_output_file_overrides_are_verbatim(runner, run_info):
    output = runner.output_file(run_info)
    assert output is not None and output.endswith("/chosen.output")


@pytest.mark.parametrize(
    "bad_name",
    ["/tmp/unrelated.jsonl", "../../escaped.jsonl", "nested/name.jsonl", "", ".", ".."],
)
def test_runner_output_file_overrides_must_be_bare_file_names(bad_name):
    # A separator or traversal component would escape the mounted output directory
    # and let the host and container paths identify different files.
    forest_runner = ForestRunnerConfig("./graphs/testing.json")
    forest_info = ForestRunInfo(levels=["county"], pop_col="TOTPOP", output_file_name=bad_name)
    with pytest.raises(ValueError, match="bare file name"):
        forest_runner.output_file(forest_info)

    smc_runner = SMCRunnerConfig("./shapefiles/testing")
    smc_info = SMCRunInfo(pop_col="TOTPOP", n_dists=4, n_sims=20, output_file_name=bad_name)
    with pytest.raises(ValueError, match="bare file name"):
        smc_runner.output_file(smc_info)


def test_log_file_layout_and_directory_creation(tmp_path):
    # The log lives at log_folder/<input_stem>/<file_stem>.log, and asking for the
    # path creates the per-input directory so run() can open it directly.
    from pathlib import Path

    runner = RecomRunnerConfig(
        "./graphs/testing.json",
        output_folder=str(tmp_path / "output"),
        log_folder=str(tmp_path / "logs"),
    )
    run_info = RecomRunInfo(pop_col="TOTPOP", assignment_col="CD", variant="A")

    log_file = runner.log_file(run_info)

    log_dir = tmp_path / "logs" / "testing"
    assert Path(log_file) == log_dir / f"{runner.file_stem(run_info)}.log"
    assert log_dir.is_dir()


@pytest.mark.parametrize(
    ("runner", "foreign_run_info", "expected_type"),
    [
        (
            ForestRunnerConfig("./graphs/testing.json"),
            SMCRunInfo(pop_col="TOTPOP", n_dists=4, n_sims=20),
            "ForestRunInfo",
        ),
        (
            SMCRunnerConfig("./shapefiles/testing"),
            ForestRunInfo(levels=["county"], pop_col="TOTPOP"),
            "SMCRunInfo",
        ),
        (
            RecomRunnerConfig("./graphs/testing.json"),
            SMCRunInfo(pop_col="TOTPOP", n_dists=4, n_sims=20),
            "RecomRunInfo",
        ),
    ],
)
def test_runner_shared_paths_reject_foreign_run_infos(runner, foreign_run_info, expected_type):
    for operation in (runner.run_command, runner.output_file, runner.log_file):
        with pytest.raises(TypeError, match=expected_type):
            operation(foreign_run_info)


class FakeDockerAPI:
    def __init__(self):
        self.exec_calls = []

    def exec_create(self, container_id, **kwargs):
        self.exec_calls.append((container_id, kwargs))
        return {"Id": len(self.exec_calls)}

    def exec_start(self, exec_id, **kwargs):
        return []

    def exec_inspect(self, exec_id):
        return {"ExitCode": 0}


def test_all_execution_paths_pass_the_config_in_argv(tmp_path, monkeypatch):
    runner = RecomRunnerConfig(
        "./graphs/testing.json", output_folder=str(tmp_path), log_folder=str(tmp_path)
    )
    api = FakeDockerAPI()
    container = make_fake_run_container(
        config=runner,
        client=SimpleNamespace(api=api),
        container=SimpleNamespace(id="container-id"),
    )
    monkeypatch.setattr(
        Graph, "from_json", staticmethod(lambda _path: Graph.from_networkx(nx.Graph()))
    )

    # force_print keeps the fake run's stdout path: no output file is expected on disk.
    run_info = RecomRunInfo(
        pop_col="TOTPOP", assignment_col="CD", variant="A", writer="jsonl", force_print=True
    )
    container.run(run_info)
    list(container.run_iter(run_info))
    list(container.mcmc_run_with_updaters(run_info))

    # The config rides in the command's "$1" argv slot; nothing travels in
    # the exec environment on any path.
    commands = [call[1]["cmd"] for call in api.exec_calls]
    assert len(commands) == 3
    assert all("environment" not in call[1] for call in api.exec_calls)
    assert json.loads(commands[0][4]) == runner.run_config(run_info)
    assert json.loads(commands[1][4]) == runner.run_config(run_info)

    updater_config = json.loads(commands[2][4])
    assert updater_config["writer"] == "canonical"
    assert updater_config["output_file"] is None


def test_forest_updater_path_uses_stdout_jsonl_config(tmp_path, monkeypatch):
    runner = ForestRunnerConfig(
        "./graphs/testing.json", output_folder=str(tmp_path), log_folder=str(tmp_path)
    )
    api = FakeDockerAPI()
    container = make_fake_run_container(
        config=runner,
        client=SimpleNamespace(api=api),
        container=SimpleNamespace(id="container-id"),
    )
    monkeypatch.setattr(
        Graph, "from_json", staticmethod(lambda _path: Graph.from_networkx(nx.Graph()))
    )

    run_info = ForestRunInfo(
        levels=["county", "precinct"],
        pop_col="TOTPOP",
        writer="ben",
    )
    list(container.mcmc_run_with_updaters(run_info))

    updater_config = json.loads(api.exec_calls[0][1]["cmd"][4])
    assert updater_config["io"]["writer"] == "jsonl"
    assert updater_config["io"]["output"] is None
