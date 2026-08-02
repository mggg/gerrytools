"""Command-builder tests for the rustrecom (Rust ReCom) runner.

These assert the exact shell command handed to docker exec, pinned against the
CLI surface of the 0.1.4 branch of mggg/rustrecom. Under config transport every
run-specific value rides in the command's "$1" config argv slot, so the template
is constant and the tests assert that property directly. No Docker required.
"""

import json
from typing import Any, cast

import pytest

from gerrytools.mgrp.runners.recom import (
    ChainVariant,
    RecomRunInfo,
    RecomRunnerConfig,
    Writer,
)

RECOM_COMMAND = (
    '. /root/.cargo/env; /usr/bin/time -v rustrecom chain --config "$1" --overwrite-output'
)


def build(run_info):
    """The static shell template. The JSON config rides as the "$1" argv slot."""
    config = RecomRunnerConfig("./graphs/testing.json")
    argv = config.run_command(run_info)
    assert argv[:2] == ["sh", "-c"]
    assert argv[3] == "rustrecom"  # $0 for the -c script
    assert len(argv) == 5
    return argv[2]


def command_config(run_info):
    runner = RecomRunnerConfig("./graphs/testing.json")
    return json.loads(runner.run_command(run_info)[4])


@pytest.mark.parametrize("writer", ["pcompress", "ben", "bendl"])
def test_force_print_with_binary_writers_rejected(writer):
    # Binary writers stream bytes that die mid-decode on the console path; bendl used to
    # be the only one rejected.
    with pytest.raises(ValueError, match=f"force_print is not supported with the {writer} writer"):
        RecomRunInfo(
            pop_col="TOTPOP",
            assignment_col="CD",
            variant="A",
            writer=cast(Writer, writer),
            force_print=True,
        )


def test_basic_variant_a_command():
    command = build(RecomRunInfo(pop_col="TOTPOP", assignment_col="CD", variant="A"))
    assert command == RECOM_COMMAND


def test_shadowed_subcommand_attribute_cannot_alter_command():
    # Regression: the sh -c token used to read run_info.subcommand, which instance data
    # could shadow; the token must come from the type-keyed RUN_INFO_SUBCOMMANDS mapping.
    run_info = RecomRunInfo(pop_col="TOTPOP", assignment_col="CD", variant="A")
    setattr(run_info, "subcommand", "chain; rm -rf /")

    assert build(run_info) == RECOM_COMMAND
    assert command_config(run_info)["command"] == "chain"


def test_run_info_subclasses_have_no_subcommand():
    # Runner dispatch is exact-type, so unknown run-info subclasses fail before command assembly.
    class CustomRunInfo(RecomRunInfo):
        pass

    run_info = CustomRunInfo(pop_col="TOTPOP", assignment_col="CD", variant="A")
    with pytest.raises(TypeError, match="RecomRunnerConfig requires.*RecomRunInfo"):
        RecomRunnerConfig("./graphs/testing.json").run_command(run_info)


def test_values_change_only_the_config_slot():
    first = RecomRunInfo(pop_col="TOTPOP", assignment_col="CD", variant="A")
    second = RecomRunInfo(
        pop_col="POP; false",
        assignment_col="CD $(touch nope)",
        variant="BW",
        region_weights={"COUNTY": 2.0},
        n_steps=99,
        rng_seed=7,
        writer="ben",
        sum_cols=["BVAP", "VAP"],
        cut_edges_count=True,
        show_progress=True,
    )

    assert build(first) == build(second) == RECOM_COMMAND
    command = build(second)
    assert "touch nope" not in command
    assert "COUNTY" not in command

    first_config = command_config(first)
    second_config = command_config(second)
    assert first_config != second_config
    assert second_config["assignment_col"] == "CD $(touch nope)"
    assert second_config["region_weights"] == {"COUNTY": 2.0}
    assert second_config["writer"] == "ben"


def test_bw_variant_is_reachable():
    # Regression: the old guard list contained "VW" instead of "BW".
    run_info = RecomRunInfo(
        pop_col="TOTPOP",
        assignment_col="CD",
        variant="BW",
        region_weights={"COUNTY": 2.0},
    )
    config = command_config(run_info)
    assert config["variant"] == "district-pairs-region-aware"
    assert config["region_weights"] == {"COUNTY": 2.0}


def test_new_014_fields_travel_in_the_config():
    run_info = RecomRunInfo(
        pop_col="TOTPOP",
        assignment_col="CD",
        variant="A",
        target_pop=760_000,
        edge_weight_keys=["water_len", "road_len"],
        constraint={
            "constraint": "district_share_floor",
            "numerator_col": "BVAP",
            "denominator_cols": ["VAP"],
            "threshold": 0.4,
        },
        cut_edges_count=True,
        show_progress=True,
    )
    command = build(run_info)
    config = command_config(run_info)

    assert config["target_pop"] == 760_000
    assert config["edge_weight_keys"] == ["water_len", "road_len"]
    assert config["cut_edges_count"] is True
    assert config["show_progress"] is True
    assert config["constraint"] == {
        "constraint": "district_share_floor",
        "numerator_col": "BVAP",
        "denominator_cols": ["VAP"],
        "threshold": 0.4,
    }
    assert "district_share_floor" not in command
    assert "water_len" not in command


def test_bendl_writer_config_names_output_file():
    run_info = RecomRunInfo(
        pop_col="TOTPOP",
        assignment_col="CD",
        variant="A",
        writer="bendl",
        bendl_graph_order="mlc",
    )
    assert build(run_info) == RECOM_COMMAND
    config = command_config(run_info)
    config_hash = RecomRunnerConfig("./graphs/testing.json").config_hash(run_info)
    assert config["writer"] == "bendl"
    assert config["output_file"] == (
        f"/home/recom/output/testing/RecomA_CD_42_10_{config_hash}.bendl"
    )
    assert config["bendl_graph_order"] == "mlc"


def test_force_print_config_has_null_output():
    run_info = RecomRunInfo(
        pop_col="TOTPOP",
        assignment_col="CD",
        variant="A",
        force_print=True,
    )
    assert build(run_info) == RECOM_COMMAND
    assert command_config(run_info)["output_file"] is None


@pytest.mark.parametrize("bad_order", ["mcl", "MLC", "key:", "key", "rcm "])
def test_unknown_bendl_graph_order_rejected_at_construction(bad_order):
    # A typo used to surface only after the container started.
    with pytest.raises(ValueError, match="bendl_graph_order"):
        RecomRunInfo(
            pop_col="TOTPOP",
            assignment_col="CD",
            variant="A",
            writer="bendl",
            bendl_graph_order=bad_order,
        )


def test_key_bendl_graph_order_accepted():
    run_info = RecomRunInfo(
        pop_col="TOTPOP",
        assignment_col="CD",
        variant="A",
        writer="bendl",
        bendl_graph_order="key:COUNTY",
    )
    assert command_config(run_info)["bendl_graph_order"] == "key:COUNTY"


def test_bendl_graph_order_requires_bendl_writer():
    with pytest.raises(ValueError, match="bendl"):
        build(
            RecomRunInfo(
                pop_col="TOTPOP",
                assignment_col="CD",
                variant="A",
                bendl_graph_order="rcm",
            )
        )


def test_canonical_stdout_command_clears_bendl_graph_order():
    runner = RecomRunnerConfig("./graphs/testing.json")
    run_info = RecomRunInfo(
        pop_col="TOTPOP",
        assignment_col="CD",
        variant="A",
        writer="bendl",
        bendl_graph_order="mlc",
    )

    config = json.loads(runner.canonical_stdout_command(run_info)[4])

    assert config["writer"] == "canonical"
    assert config["output_file"] is None
    assert config["bendl_graph_order"] == "none"


def test_edge_weight_keys_rejected_for_ust_variants():
    with pytest.raises(ValueError, match="edge_weight_keys"):
        build(
            RecomRunInfo(
                pop_col="TOTPOP",
                assignment_col="CD",
                variant="C",
                edge_weight_keys=["water_len"],
            )
        )


def test_invalid_variant_rejected():
    # Intentionally invalid at the type level too; the cast exercises the
    # runtime guard that protects untyped callers.
    with pytest.raises(ValueError):
        build(RecomRunInfo(pop_col="TOTPOP", assignment_col="CD", variant=cast(ChainVariant, "VW")))


def test_rustrecom_variant_names_accepted():
    # The long rustrecom names normalize to their letter codes, so output naming is
    # unchanged no matter which spelling was used.
    run_info = RecomRunInfo(
        pop_col="TOTPOP",
        assignment_col="CD",
        variant="district-pairs-region-aware",
        region_weights={"COUNTY": 2.0},
    )
    assert run_info.variant == "BW"
    config = command_config(run_info)
    config_hash = RecomRunnerConfig("./graphs/testing.json").config_hash(run_info)
    assert config["variant"] == "district-pairs-region-aware"
    assert config["output_file"].endswith(f"/RecomBW_CD_42_10_{config_hash}.jsonl")


def test_region_aware_variants_require_region_weights():
    with pytest.raises(ValueError, match="region_weights"):
        RecomRunInfo(pop_col="TOTPOP", assignment_col="CD", variant="AW")


@pytest.mark.parametrize("bad_weight", [float("nan"), float("inf"), "2.0", None, True], ids=repr)
def test_region_weight_values_must_be_finite_numbers(bad_weight):
    # NaN in particular serializes to a bare NaN token that the container-side parser
    # rejects only after Docker has started, defeating fail-before-Docker.
    with pytest.raises(ValueError, match="region_weights"):
        RecomRunInfo(
            pop_col="TOTPOP",
            assignment_col="CD",
            variant="AW",
            region_weights={"COUNTY": bad_weight},
        )


def test_nan_region_weight_assigned_after_construction_never_reaches_the_container():
    # Backstop: json.dumps(..., allow_nan=False) rejects a NaN smuggled in post-construction.
    run_info = RecomRunInfo(
        pop_col="TOTPOP",
        assignment_col="CD",
        variant="AW",
        region_weights={"COUNTY": 2.0},
    )
    run_info.region_weights["COUNTY"] = float("nan")
    runner = RecomRunnerConfig("./graphs/testing.json")
    with pytest.raises(ValueError):
        runner.run_command(run_info)


def test_post_construction_variant_and_writer_assignments_are_revalidated():
    # Mirrors the output_file_name pattern: a post-construction assignment used to bypass
    # __post_init__, surfacing later as a bare KeyError at command build (variant) or as a
    # nonsense writer flowing into the emitted config.
    run_info = RecomRunInfo(pop_col="TOTPOP", assignment_col="CD", variant="A")
    run_info.variant = "district-pairs-mst"  # long spelling normalizes at use
    assert command_config(run_info)["variant"] == "district-pairs-mst"

    run_info.variant = cast(ChainVariant, "Z")
    with pytest.raises(ValueError, match="Unknown variant 'Z'"):
        command_config(run_info)

    run_info.variant = "A"
    run_info.writer = cast(Writer, "parquet")
    with pytest.raises(ValueError, match="Unknown writer 'parquet'"):
        command_config(run_info)


def test_reversible_requires_balance_ub():
    with pytest.raises(ValueError, match="balance_ub"):
        RecomRunInfo(pop_col="TOTPOP", assignment_col="CD", variant="R")


@pytest.mark.parametrize("balance_ub", [True, 1.5])
def test_balance_ub_rejects_non_integers(balance_ub: object):
    # Regression: balance_ub was typed float and silently truncated via int(...) at
    # serialization time; the engine contract is an unsigned 32-bit integer.
    with pytest.raises(TypeError, match="integer"):
        RecomRunInfo(
            pop_col="TOTPOP",
            assignment_col="CD",
            variant="A",
            balance_ub=cast(int, balance_ub),
        )


@pytest.mark.parametrize("balance_ub", [-1, 2**32])
def test_balance_ub_rejects_values_outside_u32(balance_ub: int):
    with pytest.raises(ValueError, match="balance_ub"):
        RecomRunInfo(
            pop_col="TOTPOP",
            assignment_col="CD",
            variant="A",
            balance_ub=balance_ub,
        )


@pytest.mark.parametrize("balance_ub", [0, 2**32 - 1])
def test_balance_ub_accepts_u32_bounds_for_non_reversible_variants(balance_ub: int):
    run_info = RecomRunInfo(
        pop_col="TOTPOP",
        assignment_col="CD",
        variant="A",
        balance_ub=balance_ub,
    )
    assert command_config(run_info)["balance_ub"] == balance_ub


def test_unknown_writer_rejected():
    # Intentionally invalid at the type level too; the cast exercises the
    # runtime guard that protects untyped callers.
    with pytest.raises(ValueError, match="writer"):
        RecomRunInfo(pop_col="TOTPOP", assignment_col="CD", variant="A", writer=cast(Writer, "csv"))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("n_steps", 0),
        ("n_threads", -1),
        ("batch_size", 0),
        ("pop_tol", -0.05),
        ("pop_tol", float("nan")),
        ("target_pop", 0),
    ],
)
def test_recom_numeric_domains_rejected_at_construction(field, value):
    with pytest.raises(ValueError, match=field):
        RecomRunInfo(pop_col="TOTPOP", assignment_col="CD", variant="A", **{field: value})


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("pop_col", ""),
        ("assignment_col", ""),
        ("rng_seed", True),
        ("rng_seed", -1),
        ("rng_seed", 2**64),
        ("sum_cols", ["TOTPOP", ""]),
        ("cut_edges_count", 1),
        ("show_progress", 1),
    ],
)
def test_recom_config_field_types_rejected(field, value):
    settings: dict[str, Any] = {
        "pop_col": "TOTPOP",
        "assignment_col": "CD",
        "variant": "A",
    }
    settings[field] = value
    with pytest.raises(ValueError, match=field):
        RecomRunInfo(**settings)


def test_recom_accepts_u64_seed_bound():
    run_info = RecomRunInfo(pop_col="TOTPOP", assignment_col="CD", variant="A", rng_seed=2**64 - 1)
    assert command_config(run_info)["rng_seed"] == 2**64 - 1


@pytest.mark.parametrize("assignment_col", ["CD/2020", r"..\CD"])
def test_assignment_column_must_be_safe_for_derived_file_names(assignment_col):
    with pytest.raises(ValueError, match="assignment_col must be a bare file name"):
        RecomRunInfo(pop_col="TOTPOP", assignment_col=assignment_col, variant="A")


def test_output_file_matches_container_output():
    config = RecomRunnerConfig("./graphs/testing.json", output_folder="./output")
    run_info = RecomRunInfo(pop_col="TOTPOP", assignment_col="CD", variant="A")
    output_file = config.output_file(run_info)
    assert output_file is not None
    config_hash = config.config_hash(run_info)
    assert output_file.endswith(f"/output/testing/RecomA_CD_42_10_{config_hash}.jsonl")

    bendl_info = RecomRunInfo(pop_col="TOTPOP", assignment_col="CD", variant="A", writer="bendl")
    bendl_file = config.output_file(bendl_info)
    assert bendl_file is not None and bendl_file.endswith(".bendl")

    printed = RecomRunInfo(pop_col="TOTPOP", assignment_col="CD", variant="A", force_print=True)
    assert config.output_file(printed) is None
