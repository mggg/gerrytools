"""Command-builder tests for the Forest (MSMS) and SMC runners. No Docker required."""

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

from gerrytools.mgrp import (
    ForestRunInfo,
    ForestRunnerConfig,
    SMCRunInfo,
    SMCRunnerConfig,
)

FOREST_DIRECT_COMMAND = (
    'export JULIA_PROJECT="/home/forest"; '
    '/usr/bin/time -v julia /home/forest/cli/multi_cli.jl --config "$1"'
)
SMC_DIRECT_COMMAND = '/usr/bin/time -v Rscript /home/smc/cli/smc_cli.R --config "$1"'


def parser_pipeline(engine_command, parser_name):
    """The engine command wrapped in the parser pipeline template.

    Pins the failure-propagation construct: plain sh reports only the last pipeline
    command's status, so the template must capture the engine's status through a temp
    file and fail with it even when the parser exits cleanly.
    """
    return (
        "engine_status_file=$(mktemp) || exit 1; "
        f'{{ {engine_command}; echo $? > "$engine_status_file"; }}'
        f' | {parser_name} --config "$1"; '
        "parser_status=$?; "
        'read engine_status < "$engine_status_file"; '
        'rm -f "$engine_status_file"; '
        '[ "${engine_status:-1}" -eq 0 ] || exit "${engine_status:-1}"; '
        'exit "$parser_status"'
    )


FOREST_PARSER_COMMAND = parser_pipeline(FOREST_DIRECT_COMMAND, "msms_parser")
SMC_PARSER_COMMAND = parser_pipeline(SMC_DIRECT_COMMAND, "smc_parser")


def command_config(argv):
    """The JSON config riding in the command's "$1" argv slot."""
    assert argv[:2] == ["sh", "-c"]
    assert len(argv) == 5
    return json.loads(argv[4])


def base_forest_info(**overrides):
    return ForestRunInfo(
        levels=["county", "precinct"],
        pop_col="TOTPOP",
        **overrides,
    )


def test_forest_uses_exact_static_templates():
    runner = ForestRunnerConfig("./graphs/testing.json")

    assert runner.run_command(base_forest_info())[:4] == [
        "sh",
        "-c",
        FOREST_PARSER_COMMAND,
        "forest",
    ]
    assert runner.run_command(base_forest_info(writer="ben"))[2] == FOREST_PARSER_COMMAND
    assert runner.run_command(base_forest_info(writer="raw"))[2] == FOREST_DIRECT_COMMAND


def test_forest_values_change_only_the_environment():
    runner = ForestRunnerConfig("./graphs/testing.json")
    first = base_forest_info()
    second = ForestRunInfo(
        levels=["county $(touch nope)", "precinct with spaces"],
        pop_col="POP; false",
        num_dists=7,
        rng_seed=999,
        output_file_name="sentinel output.jsonl",
    )

    # The shell template (argv[:4]) is static; only the "$1" data slot varies.
    assert runner.run_command(first)[:4] == runner.run_command(second)[:4]
    command = runner.run_command(second)[2]
    assert "touch nope" not in command
    assert "sentinel output" not in command

    first_config = command_config(runner.run_command(first))
    second_config = command_config(runner.run_command(second))
    assert first_config != second_config
    assert second_config["run"]["levels"] == ["county $(touch nope)", "precinct with spaces"]
    assert second_config["io"]["output"].endswith("/sentinel output.jsonl")


def test_forest_force_print_changes_config_not_template():
    runner = ForestRunnerConfig("./graphs/testing.json")
    file_run = base_forest_info()
    stdout_run = base_forest_info(force_print=True)

    assert runner.run_command(file_run)[:4] == runner.run_command(stdout_run)[:4]
    config = command_config(runner.run_command(stdout_run))
    assert config["io"]["output"] is None


def test_forest_binary_writer_cannot_force_print():
    with pytest.raises(ValueError, match="binary output cannot be decoded"):
        base_forest_info(writer="ben", force_print=True)


def base_smc_info(**overrides):
    return SMCRunInfo(pop_col="TOTPOP", n_dists=4, n_sims=20, **overrides)


def test_smc_uses_exact_static_templates():
    runner = SMCRunnerConfig("./shapefiles/testing")

    assert runner.run_command(base_smc_info())[:4] == [
        "sh",
        "-c",
        SMC_PARSER_COMMAND,
        "smc",
    ]
    assert runner.run_command(base_smc_info(writer="ben"))[2] == SMC_PARSER_COMMAND
    assert runner.run_command(base_smc_info(writer="csv"))[2] == SMC_DIRECT_COMMAND


def test_smc_values_change_only_the_config_slot():
    runner = SMCRunnerConfig("./shapefiles/testing")
    first = base_smc_info()
    second = SMCRunInfo(
        pop_col="POP; false",
        n_dists=8,
        pop_bounds=[9000, 10000, 11000],
        n_sims=99,
        rng_seed=123,
        tally_columns=["VAP $(touch nope)"],
        output_file_name="sentinel output",
    )

    assert runner.run_command(first)[:4] == runner.run_command(second)[:4]
    command = runner.run_command(second)[2]
    assert "touch nope" not in command
    assert "sentinel output" not in command

    config = command_config(runner.run_command(second))
    assert config["map"]["pop_bounds"] == [9000, 10000, 11000]
    assert config["run"]["tally_columns"] == ["VAP $(touch nope)"]
    assert config["io"]["output"].endswith("/sentinel output")


def test_smc_pop_bounds_wrong_length_rejected():
    with pytest.raises(ValueError, match="pop_bounds"):
        base_smc_info(pop_bounds=[9000, 11000])


def test_smc_unordered_pop_bounds_rejected():
    with pytest.raises(ValueError, match="ordered"):
        base_smc_info(pop_bounds=[11000, 10000, 9000])


@pytest.mark.parametrize("pop_bounds", [[False, True, True], [0, 10.5, 20]])
def test_smc_pop_bounds_require_nonnegative_integers(pop_bounds):
    with pytest.raises(ValueError, match="pop_bounds"):
        base_smc_info(pop_bounds=pop_bounds)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("n_dists", 0),
        ("n_sims", -1),
        ("pop_tol", True),
        ("pop_tol", -0.01),
        ("pop_tol", float("nan")),
        ("compactness", float("nan")),
        ("adapt_k_thresh", 2),
        ("seq_alpha", -1),
        ("pop_temper", float("inf")),
        ("final_infl", float("nan")),
        ("rng_seed", True),
        ("rng_seed", -(2**31)),
        ("rng_seed", 2**31),
    ],
)
def test_smc_numeric_domains_rejected_at_construction(field, value):
    settings: dict[str, Any] = dict(pop_col="TOTPOP", n_dists=4, n_sims=20)
    settings[field] = value
    with pytest.raises(ValueError, match=field):
        SMCRunInfo(**settings)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("num_dists", 0),
        ("n_steps", -1),
        ("pop_dev", True),
        ("pop_dev", -0.5),
        ("pop_dev", float("inf")),
        ("gamma", float("nan")),
        ("rng_seed", True),
        ("rng_seed", -1),
        ("rng_seed", 2**63),
    ],
)
def test_forest_numeric_domains_rejected_at_construction(field, value):
    with pytest.raises(ValueError, match=field):
        base_forest_info(**{field: value})


def test_engine_specific_seed_bounds_are_accepted():
    assert base_forest_info(rng_seed=2**63 - 1).rng_seed == 2**63 - 1
    assert base_smc_info(rng_seed=-(2**31 - 1)).rng_seed == -(2**31 - 1)


@pytest.mark.parametrize(
    ("constructor", "field"),
    [
        (lambda: ForestRunInfo(levels=["county"], pop_col=""), "pop_col"),
        (lambda: base_forest_info(force_print=1), "force_print"),
        (lambda: SMCRunInfo(pop_col="", n_dists=4, n_sims=20), "pop_col"),
        (lambda: base_smc_info(resample=1), "resample"),
        (lambda: base_smc_info(tally_columns=["VAP", ""]), "tally_columns"),
    ],
)
def test_engine_config_field_types_rejected(constructor, field):
    with pytest.raises(ValueError, match=field):
        constructor()


def test_smc_unknown_writer_rejected():
    with pytest.raises(ValueError, match="writer"):
        base_smc_info(writer="parquet")


def test_forest_unknown_writer_rejected():
    with pytest.raises(ValueError, match="writer"):
        base_forest_info(writer="parquet")


def test_output_file_paths():
    forest_config = ForestRunnerConfig("./graphs/testing.json", output_folder="./output")
    forest_file = forest_config.output_file(base_forest_info())
    forest_hash = forest_config.config_hash(base_forest_info())
    assert forest_file is not None
    assert forest_file.endswith(f"/output/testing/Forest_42_atlas_gamma0.0_10_{forest_hash}.jsonl")
    assert forest_config.output_file(base_forest_info(force_print=True)) is None

    smc_config = SMCRunnerConfig("./shapefiles/testing", output_folder="./output")
    smc_file = smc_config.output_file(base_smc_info())
    smc_hash = smc_config.config_hash(base_smc_info())
    assert smc_file is not None
    assert smc_file.endswith(f"/output/testing/SMC_42_20_{smc_hash}.jsonl")


def test_config_hash_separates_configs_that_share_a_stem():
    # Regression: stems used only headline settings, so runs differing elsewhere
    # (e.g. num_dists or compactness) overwrote each other's outputs and logs.
    forest_config = ForestRunnerConfig("./graphs/testing.json", output_folder="./output")
    assert forest_config.output_file(base_forest_info()) == forest_config.output_file(
        base_forest_info()
    )
    assert forest_config.output_file(base_forest_info()) != forest_config.output_file(
        base_forest_info(num_dists=7)
    )

    smc_config = SMCRunnerConfig("./shapefiles/testing", output_folder="./output")
    assert smc_config.output_file(base_smc_info()) == smc_config.output_file(base_smc_info())
    assert smc_config.output_file(base_smc_info()) != smc_config.output_file(
        base_smc_info(compactness=0.5)
    )


@pytest.mark.parametrize(
    ("engine_exit", "parser_exit", "expected_exit"),
    [(3, 0, 3), (0, 5, 5), (0, 0, 0)],
)
def test_parser_pipeline_exit_status_under_local_sh(
    tmp_path, engine_exit, parser_exit, expected_exit
):
    # Runs the exact template _shell_command builds under a real sh. The engine-fail case
    # is the regression: without engine-status capture, sh reports the parser's clean exit
    # and a silently truncated ensemble would look like success.
    if shutil.which("sh") is None:
        pytest.skip("requires a POSIX sh on PATH")
    engine_script = tmp_path / "fake_engine"
    engine_script.write_text(f"#!/bin/sh\necho '{{\"sample\": 1}}'\nexit {engine_exit}\n")
    parser_script = tmp_path / "fake_parser"
    parser_script.write_text(
        f'#!/bin/sh\ncat > "{tmp_path}/parser_stdin"\n'
        f'printf %s "$2" > "{tmp_path}/parser_config"\nexit {parser_exit}\n'
    )
    for script in (engine_script, parser_script):
        script.chmod(0o755)

    class ProbeRunnerConfig(SMCRunnerConfig):
        parser_name = str(parser_script)

    runner = ProbeRunnerConfig("./shapefiles/testing")
    argv = runner._shell_command(
        f'{engine_script} --config "$1"', {"probe": True}, with_parser=True
    )
    result = subprocess.run(argv, capture_output=True, text=True, check=False)

    assert result.returncode == expected_exit
    # The parser stage still receives the engine's stdout and the "$1" config verbatim.
    assert (tmp_path / "parser_stdin").read_text() == '{"sample": 1}\n'
    assert (tmp_path / "parser_config").read_text() == '{"probe": true}'


def test_smc_expected_files_include_metadata_and_tallies_sidecars():
    runner = SMCRunnerConfig("./shapefiles/testing", output_folder="./output")

    plain = base_smc_info()
    plain_output = runner.output_file(plain)
    assert plain_output is not None
    assert runner.expected_files(plain) == [
        plain_output,
        str(Path(plain_output).with_name(f"{Path(plain_output).stem}_metadata.jsonl")),
    ]

    # An engine exiting 0 without the tallies CSV must fail the expected-file check, and a
    # failed rerun must restore the previous tallies alongside the primary output.
    tallied = base_smc_info(tally_columns=["VAP"])
    tallied_output = runner.output_file(tallied)
    assert tallied_output is not None
    tallied_stem = Path(tallied_output).stem
    assert runner.expected_files(tallied) == [
        tallied_output,
        str(Path(tallied_output).with_name(f"{tallied_stem}_metadata.jsonl")),
        str(Path(tallied_output).with_name(f"{tallied_stem}_tallies.csv")),
    ]


def test_smc_csv_writer_expects_assignments_but_no_tallies_sidecar():
    # The csv writer keeps tallies in the plans CSV itself and writes an assignments CSV.
    runner = SMCRunnerConfig("./shapefiles/testing", output_folder="./output")
    run_info = base_smc_info(writer="csv", tally_columns=["VAP"])
    output = runner.output_file(run_info)
    assert output is not None
    output_stem = Path(output).stem
    assert runner.expected_files(run_info) == [
        output,
        str(Path(output).with_name(f"{output_stem}_metadata.jsonl")),
        str(Path(output).with_name(f"{output_stem}_assignments.csv")),
    ]


def test_forest_expected_files_include_provenance_for_every_file_output():
    runner = ForestRunnerConfig("./graphs/testing.json", output_folder="./output")
    parsed = base_forest_info()
    output = runner.output_file(parsed)
    assert output is not None
    assert runner.expected_files(parsed) == [
        output,
        str(Path(output).with_name(f"{Path(output).stem}_metadata.jsonl")),
    ]
    raw = base_forest_info(writer="raw")
    raw_output = runner.output_file(raw)
    assert raw_output is not None
    assert runner.expected_files(raw) == [
        raw_output,
        str(Path(raw_output).with_name(f"{Path(raw_output).stem}_metadata.jsonl")),
    ]
    assert runner.expected_files(base_forest_info(force_print=True)) == []


def test_runner_configs_reject_foreign_run_infos():
    # Only recom had an exact-type dispatch; forest and SMC failed incidentally (or not at
    # all) when handed the wrong run-info class. The casts exercise the runtime guards.
    from typing import cast

    forest_runner = ForestRunnerConfig("./graphs/testing.json")
    smc_runner = SMCRunnerConfig("./shapefiles/testing")
    smc_info = base_smc_info()
    forest_info = base_forest_info()

    with pytest.raises(TypeError, match="ForestRunInfo"):
        forest_runner.run_command(cast(ForestRunInfo, smc_info))
    with pytest.raises(TypeError, match="ForestRunInfo"):
        forest_runner.run_config(cast(ForestRunInfo, smc_info))
    with pytest.raises(TypeError, match="SMCRunInfo"):
        smc_runner.run_command(cast(SMCRunInfo, forest_info))
    with pytest.raises(TypeError, match="SMCRunInfo"):
        smc_runner.run_config(cast(SMCRunInfo, forest_info))
