"""Command-builder tests for the rustrecom optimizer runners. No Docker required."""

import json
from pathlib import Path
from typing import Any, Literal, cast

import pytest

from gerrytools.mgrp import Objective, RecomRunnerConfig, ShortBurstsRunSpec, TiltedRunSpec
from gerrytools.mgrp.objectives import Aggregation, validate_objective_spec
from gerrytools.mgrp.runners.recom import WRITERS, Writer

OBJECTIVE = {
    "objective": "by_district_abs_deviation",
    "target_values": [0.25, 0.25, 0.25, 0.25],
    "pov_counts_col": "TOTPOP",
    "total_counts_col": "TOTPOP",
}


def build(run_spec):
    """The static shell template. The JSON config rides as the "$1" argv slot."""
    config = RecomRunnerConfig("./graphs/testing.json")
    argv = config.run_command(run_spec)
    assert argv[:2] == ["sh", "-c"]
    assert argv[3] == "rustrecom"
    assert len(argv) == 5
    return argv[2]


def command_config(run_spec):
    runner = RecomRunnerConfig("./graphs/testing.json")
    return json.loads(runner.run_command(run_spec)[4])


def short_bursts_spec(**overrides: Any) -> ShortBurstsRunSpec:
    settings: dict[str, Any] = dict(
        pop_col="TOTPOP",
        assignment_col="CD",
        objective=dict(OBJECTIVE),
        burst_length=5,
        n_steps=50,
        maximize=False,
    )
    settings.update(overrides)
    return ShortBurstsRunSpec(**settings)


def tilted_spec(**overrides: Any) -> TiltedRunSpec:
    settings: dict[str, Any] = dict(
        pop_col="TOTPOP",
        assignment_col="CD",
        objective=dict(OBJECTIVE),
        n_steps=50,
        maximize=False,
    )
    settings.update(overrides)
    return TiltedRunSpec(**settings)


def test_short_bursts_command_and_config():
    command = build(short_bursts_spec())
    assert command == (
        ". /root/.cargo/env; /usr/bin/time -v rustrecom "
        'short-bursts --config "$1" --overwrite-output'
    )

    config = command_config(short_bursts_spec())
    config_hash = RecomRunnerConfig("./graphs/testing.json").config_hash(short_bursts_spec())
    assert config == {
        "version": 1,
        "command": "short-bursts",
        "graph_json": "/home/recom/shapefiles/testing.json",
        "n_steps": 50,
        "tol": 0.05,
        "pop_col": "TOTPOP",
        "assignment_col": "CD",
        "rng_seed": 42,
        "objective": OBJECTIVE,
        "maximize": False,
        "n_threads": 1,
        "variant": "district-pairs-mst",
        "writer": "canonical",
        "sum_cols": [],
        "partial_sum_cols": [],
        "region_weights": {},
        "edge_weight_keys": [],
        "output_file": f"/home/recom/output/testing/SBB_CD_42_50_5_{config_hash}.jsonl",
        "scores_output_file": (
            f"/home/recom/output/testing/SBB_CD_42_50_5_{config_hash}_scores.csv"
        ),
        "show_progress": False,
        "write_improved_scores_only": False,
        "burst_length": 5,
    }


def test_tilted_command_and_config():
    command = build(tilted_spec(accept_rule="fixed", accept_worse_prob=0.05))
    assert command == (
        '. /root/.cargo/env; /usr/bin/time -v rustrecom tilted --config "$1" --overwrite-output'
    )

    config = command_config(tilted_spec(accept_rule="fixed", accept_worse_prob=0.05))
    config_hash = RecomRunnerConfig("./graphs/testing.json").config_hash(
        tilted_spec(accept_rule="fixed", accept_worse_prob=0.05)
    )
    assert config["command"] == "tilted"
    assert config["accept_rule"] == "fixed"
    assert config["accept_worse_prob"] == 0.05
    assert "acceptance_beta" not in config
    assert config["output_file"] == (
        f"/home/recom/output/testing/TiltedB_fixed_CD_42_50_{config_hash}.jsonl"
    )
    assert config["scores_output_file"] == (
        f"/home/recom/output/testing/TiltedB_fixed_CD_42_50_{config_hash}_scores.csv"
    )

    # The default rule leaves the engine-side beta default implicit.
    config = command_config(tilted_spec())
    assert config["accept_rule"] == "linear"
    assert "accept_worse_prob" not in config
    assert "acceptance_beta" not in config


def test_values_travel_only_in_the_config_slot():
    hostile = short_bursts_spec(assignment_col='CD $(touch nope) "; rm -rf')
    command = build(hostile)
    assert "touch nope" not in command
    assert command_config(hostile)["assignment_col"] == 'CD $(touch nope) "; rm -rf'


def test_output_and_scores_paths():
    runner = RecomRunnerConfig("./graphs/testing.json", output_folder="./output")
    output = runner.output_file(short_bursts_spec())
    scores = runner.scores_file(short_bursts_spec())
    stem = f"SBB_CD_42_50_5_{runner.config_hash(short_bursts_spec())}"
    assert output is not None and output.endswith(f"/output/testing/{stem}.jsonl")
    assert scores is not None and scores.endswith(f"/output/testing/{stem}_scores.csv")
    # An explicit output_file_name is used verbatim, with no hash folded in.
    custom = runner.output_file(short_bursts_spec(output_file_name="custom.jsonl"))
    assert custom is not None and custom.endswith("/output/testing/custom.jsonl")
    custom_scores = runner.scores_file(short_bursts_spec(output_file_name="custom.jsonl"))
    assert custom_scores is not None and custom_scores.endswith("/output/testing/custom_scores.csv")

    from gerrytools.mgrp import RecomRunSpec

    chain = RecomRunSpec(pop_col="TOTPOP", assignment_col="CD", variant="A")
    assert runner.scores_file(chain) is None
    # File output includes a provenance sidecar; optimizers also promise scores.
    metadata = str(Path(output).with_name(f"{Path(output).stem}_metadata.jsonl"))
    assert runner.expected_files(short_bursts_spec()) == [output, metadata, scores]
    chain_output = runner.output_file(chain)
    assert chain_output is not None
    chain_metadata = str(Path(chain_output).with_name(f"{Path(chain_output).stem}_metadata.jsonl"))
    assert runner.expected_files(chain) == [chain_output, chain_metadata]


@pytest.mark.parametrize("writer", WRITERS)
def test_recom_expected_files_match_writer_provenance(writer: Writer):
    from gerrytools.mgrp import RecomRunSpec

    runner = RecomRunnerConfig("./graphs/testing.json", output_folder="./output")
    run_spec = RecomRunSpec(pop_col="TOTPOP", assignment_col="CD", variant="A", writer=writer)
    output = runner.output_file(run_spec)
    assert output is not None
    expected = [output]
    if writer not in {"jsonl", "jsonl-full", "bendl"}:
        expected.append(str(Path(output).with_name(f"{Path(output).stem}_metadata.jsonl")))

    assert runner.expected_files(run_spec) == expected


@pytest.mark.parametrize("run_spec_factory", [short_bursts_spec, tilted_spec])
@pytest.mark.parametrize("writer", WRITERS)
def test_optimizer_expected_files_match_writer_provenance(run_spec_factory, writer: Writer):
    runner = RecomRunnerConfig("./graphs/testing.json", output_folder="./output")
    run_spec = run_spec_factory(writer=writer)
    output = runner.output_file(run_spec)
    scores = runner.scores_file(run_spec)
    assert output is not None and scores is not None
    expected = [output]
    if writer != "bendl":
        expected.append(str(Path(output).with_name(f"{Path(output).stem}_metadata.jsonl")))
    expected.append(scores)

    assert runner.expected_files(run_spec) == expected


@pytest.mark.parametrize("bad_name", ["/tmp/unrelated.jsonl", "../../escaped.jsonl", ""])
def test_optimizer_output_file_override_must_be_a_bare_file_name(bad_name):
    with pytest.raises(ValueError, match="bare file name"):
        short_bursts_spec(output_file_name=bad_name)

    # Regression: a post-construction override skipped validation, and RunnerSession.run
    # unlinks the resolved host path, so a traversal name deleted files outside the
    # output folder. The override now revalidates wherever the name is used.
    run_spec = short_bursts_spec()
    run_spec.output_file_name = bad_name
    with pytest.raises(ValueError, match="bare file name"):
        run_spec.output_name()
    with pytest.raises(ValueError, match="bare file name"):
        RecomRunnerConfig("./graphs/testing.json").output_file(run_spec)


@pytest.mark.parametrize("assignment_col", ["CD/2020", r"..\CD"])
def test_optimizer_assignment_column_must_be_safe_for_derived_file_names(assignment_col):
    with pytest.raises(ValueError, match="assignment_col must be a bare file name"):
        short_bursts_spec(assignment_col=assignment_col)


def test_post_construction_optimizer_variant_and_writer_assignments_are_revalidated():
    # Same pattern as the objective and output_file_name fields: assignments after
    # construction are rechecked at config build with the constructor's messages.
    run_spec = short_bursts_spec()
    run_spec.variant = "cut-edges-mst"  # long spelling normalizes at use
    assert command_config(run_spec)["variant"] == "cut-edges-mst"

    run_spec.variant = cast(Any, "R")
    with pytest.raises(ValueError, match="Unknown optimizer variant 'R'"):
        command_config(run_spec)

    run_spec.variant = "B"
    run_spec.writer = cast(Any, "parquet")
    with pytest.raises(ValueError, match="Unknown writer 'parquet'"):
        command_config(run_spec)


@pytest.mark.parametrize("bad_weight", [float("nan"), float("inf"), "2.0", True], ids=repr)
def test_optimizer_region_weight_values_must_be_finite_numbers(bad_weight):
    with pytest.raises(ValueError, match="region_weights"):
        short_bursts_spec(region_weights={"COUNTY": bad_weight})


def test_post_construction_objective_assignment_is_validated_and_reaches_config():
    # Mirrors the constraints live-property pattern: the raw field is stored, and
    # resolved_objective validates on access, so the config build catches a bad
    # post-assignment and a good one reaches the emitted run_config.
    run_spec = short_bursts_spec()
    run_spec.objective = Objective.gingles_partial(threshold=0.5, min_pop="BVAP", total_pop="VAP")
    assert command_config(run_spec)["objective"] == {
        "objective": "gingles_partial",
        "threshold": 0.5,
        "min_pop": "BVAP",
        "total_pop": "VAP",
    }

    run_spec.objective = cast(Any, {"not_an_objective": 1})
    with pytest.raises(ValueError, match="objective"):
        command_config(run_spec)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("n_steps", 0),
        ("n_threads", -1),
        ("pop_tol", True),
        ("pop_tol", -0.05),
        ("rng_seed", True),
        ("rng_seed", -1),
        ("rng_seed", 2**64),
    ],
)
def test_optimizer_numeric_domains_rejected_at_construction(field, value):
    with pytest.raises(ValueError, match=field):
        short_bursts_spec(**{field: value})


def test_tilted_accept_worse_prob_must_be_a_probability():
    with pytest.raises(ValueError, match="accept_worse_prob"):
        tilted_spec(accept_rule="fixed", accept_worse_prob=2)
    with pytest.raises(ValueError, match="acceptance_beta"):
        tilted_spec(acceptance_beta=-1.0)


def test_tilted_acceptance_beta_reaches_optimizer_config():
    config = command_config(tilted_spec(acceptance_beta=2.5))

    assert config["acceptance_beta"] == 2.5


def test_optimizer_validation():
    with pytest.raises(ValueError, match="Unknown optimizer variant"):
        short_bursts_spec(variant="R")
    with pytest.raises(ValueError, match="Unknown optimizer variant"):
        short_bursts_spec(variant="AW")
    with pytest.raises(ValueError, match="burst_length"):
        short_bursts_spec(burst_length=0)
    with pytest.raises(ValueError, match="burst_length"):
        short_bursts_spec(burst_length=True)
    with pytest.raises(ValueError, match="objective"):
        short_bursts_spec(objective={"not_an_objective": 1})
    with pytest.raises(ValueError, match="missing"):
        short_bursts_spec(
            objective=cast(
                Any,
                {
                    "objective": "gingles_partial",
                    "threshold": 0.5,
                    "min_pop": "BVAP",
                },
            )
        )
    with pytest.raises(TypeError, match="threshold"):
        short_bursts_spec(
            objective=cast(
                Any,
                {
                    "objective": "gingles_partial",
                    "threshold": "0.5",
                    "min_pop": "BVAP",
                    "total_pop": "VAP",
                },
            )
        )
    with pytest.raises(ValueError, match="edge_weight_keys"):
        short_bursts_spec(variant="C", edge_weight_keys=["water_len"])

    with pytest.raises(ValueError, match="accept_worse_prob"):
        tilted_spec(accept_rule="fixed")
    with pytest.raises(ValueError, match="acceptance_beta"):
        tilted_spec(accept_rule="fixed", accept_worse_prob=0.5, acceptance_beta=2.0)
    with pytest.raises(ValueError, match="accept_worse_prob"):
        tilted_spec(accept_rule="linear", accept_worse_prob=0.5)
    with pytest.raises(ValueError, match="Unknown accept_rule"):
        tilted_spec(accept_rule="metropolis")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("pop_col", ""),
        ("maximize", 1),
        ("sum_cols", ["TOTPOP", ""]),
        ("partial_sum_cols", "VAP"),
        ("show_progress", 1),
        ("write_improved_scores_only", 1),
    ],
)
def test_optimizer_config_field_types_rejected(field, value):
    with pytest.raises(ValueError, match=field):
        short_bursts_spec(**{field: value})


def test_rustrecom_variant_names_normalize():
    run_spec = short_bursts_spec(variant="cut-edges-mst")
    assert run_spec.variant == "A"
    assert command_config(run_spec)["variant"] == "cut-edges-mst"


def test_objective_builders_match_rustrecom_schemas():
    assert Objective.by_district_abs_deviation(
        target_values=[0.1, 0.4], pov_counts_col="BVAP", total_counts_col="VAP"
    ) == {
        "objective": "by_district_abs_deviation",
        "target_values": [0.1, 0.4],
        "pov_counts_col": "BVAP",
        "total_counts_col": "VAP",
    }
    assert Objective.abs_deviation(
        target=0.5, n_target_districts=3, pov_counts_col="BVAP", total_count=1000
    ) == {
        "objective": "abs_deviation",
        "target": 0.5,
        "n_target_districts": 3,
        "pov_counts_col": "BVAP",
        "total_count": 1000,
    }
    assert Objective.gingles_partial(threshold=0.5, min_pop="BVAP", total_pop="VAP") == {
        "objective": "gingles_partial",
        "threshold": 0.5,
        "min_pop": "BVAP",
        "total_pop": "VAP",
    }
    assert Objective.banded_gingles_partial(
        lower_threshold=0.55, upper_threshold=0.65, min_pop="BVAP", total_pop="VAP"
    ) == {
        "objective": "banded_gingles_partial",
        "lower_threshold": 0.55,
        "upper_threshold": 0.65,
        "min_pop": "BVAP",
        "total_pop": "VAP",
    }
    assert Objective.election_wins(
        elections=[{"votes_a": "D18", "votes_b": "R18"}], target="b", aggregation="min"
    ) == {
        "objective": "election_wins",
        "elections": [{"votes_a": "D18", "votes_b": "R18"}],
        "target": "b",
        "aggregation": "min",
    }
    assert Objective.polsby_popper(
        area_col="area", shared_perim_col="shared", boundary_perim_col="hull"
    ) == {
        "objective": "polsby_popper",
        "area_col": "area",
        "shared_perim_col": "shared",
        "aggregation": "mean",
        "boundary_perim_col": "hull",
    }

    # Builder output feeds straight into a run spec.
    run_spec = short_bursts_spec(
        objective=Objective.gingles_partial(threshold=0.5, min_pop="BVAP", total_pop="VAP")
    )
    assert command_config(run_spec)["objective"]["objective"] == "gingles_partial"


@pytest.mark.parametrize(
    "spec",
    [
        Objective.by_district_abs_deviation(
            target_values=[0.4],
            pov_counts_col="BVAP",
            total_count=1000,
        ),
        Objective.abs_deviation(
            target=0.5,
            n_target_districts=2,
            pov_counts_col="BVAP",
            total_counts_col="VAP",
        ),
        Objective.polsby_popper(
            area_col="area",
            shared_perim_col="shared",
            perim_col="perimeter",
        ),
    ],
)
def test_objective_alternative_field_shapes_round_trip(spec):
    assert validate_objective_spec(spec) == spec


@pytest.mark.parametrize(
    "spec",
    [
        {
            "objective": "banded_gingles_partial",
            "lower_threshold": 0.55,
            "upper_threshold": 0.65,
            "min_pop": "BVAP",
            "total_pop": "VAP",
        },
        {
            "objective": "election_wins",
            "elections": [{"votes_a": "D18", "votes_b": "R18"}],
            "target": "b",
            "aggregation": "min",
        },
        {
            "objective": "polsby_popper",
            "area_col": "area",
            "shared_perim_col": "shared",
            "boundary_perim_col": "boundary",
            "aggregation": "mean",
        },
    ],
)
def test_validate_objective_spec_round_trips_every_remaining_kind(spec):
    assert validate_objective_spec(spec) == spec


def test_objective_builder_validation():
    with pytest.raises(ValueError, match="exactly one of"):
        Objective.by_district_abs_deviation(target_values=[0.5], pov_counts_col="BVAP")
    with pytest.raises(ValueError, match="exactly one of"):
        Objective.abs_deviation(
            target=0.5,
            n_target_districts=2,
            pov_counts_col="BVAP",
            total_counts_col="VAP",
            total_count=100,
        )
    with pytest.raises(ValueError, match="between 0 and 1"):
        Objective.gingles_partial(threshold=1.5, min_pop="BVAP", total_pop="VAP")
    with pytest.raises(ValueError, match="at least lower_threshold"):
        Objective.banded_gingles_partial(
            lower_threshold=0.65, upper_threshold=0.55, min_pop="BVAP", total_pop="VAP"
        )
    with pytest.raises(ValueError, match="votes_a"):
        Objective.election_wins(elections=cast(Any, [{"votes_a": "D18"}]))
    # The casts are intentionally invalid at the type level; they exercise the
    # runtime guards that protect untyped callers.
    with pytest.raises(ValueError, match="'a' or 'b'"):
        Objective.election_wins(
            elections=[{"votes_a": "D", "votes_b": "R"}],
            target=cast(Literal["a", "b"], "c"),
        )
    with pytest.raises(ValueError, match="aggregation"):
        Objective.election_wins(
            elections=[{"votes_a": "D", "votes_b": "R"}],
            aggregation=cast(Aggregation, "max"),
        )
    with pytest.raises(ValueError, match="perim_col"):
        Objective.polsby_popper(area_col="area", shared_perim_col="shared")
    with pytest.raises(TypeError, match="n_target_districts"):
        Objective.abs_deviation(
            target=0.5,
            n_target_districts=cast(Any, 1.5),
            pov_counts_col="BVAP",
            total_count=100,
        )
    with pytest.raises(TypeError, match="votes_a"):
        Objective.election_wins(elections=cast(Any, [{"votes_a": 1, "votes_b": "R18"}]))
    with pytest.raises(ValueError, match="min_pop"):
        Objective.gingles_partial(threshold=0.5, min_pop="", total_pop="VAP")
    with pytest.raises(TypeError, match="target_values"):
        Objective.by_district_abs_deviation(
            target_values=cast(Any, [True]), pov_counts_col="BVAP", total_counts_col="VAP"
        )
    with pytest.raises(ValueError, match=r"target_values.*\[0, 1\]"):
        Objective.by_district_abs_deviation(
            target_values=[1.1], pov_counts_col="BVAP", total_counts_col="VAP"
        )
    with pytest.raises(ValueError, match="lower_threshold"):
        Objective.banded_gingles_partial(
            lower_threshold=-0.1, upper_threshold=0.6, min_pop="BVAP", total_pop="VAP"
        )


def test_objective_spec_rejects_unexpected_fields():
    spec = dict(Objective.gingles_partial(threshold=0.5, min_pop="BVAP", total_pop="VAP"))
    spec["surprise"] = 1

    with pytest.raises(ValueError, match="unexpected.*surprise"):
        validate_objective_spec(spec)


def test_canonical_stdout_rejects_optimizer_run_spec():
    runner = RecomRunnerConfig("./graphs/testing.json")

    with pytest.raises(TypeError, match="RecomRunSpec"):
        runner.canonical_stdout_command(tilted_spec())


@pytest.mark.parametrize(
    ("spec", "message"),
    [
        (
            {
                "objective": "by_district_abs_deviation",
                "target_values": [],
                "pov_counts_col": "BVAP",
                "total_counts_col": "VAP",
            },
            "target_values must be nonempty",
        ),
        (
            {
                "objective": "by_district_abs_deviation",
                "target_values": [0.5],
                "pov_counts_col": "BVAP",
                "total_count": float("nan"),
            },
            "total_count must be a finite number",
        ),
        (
            {
                "objective": "by_district_abs_deviation",
                "target_values": [0.5],
                "pov_counts_col": "BVAP",
                "total_count": 0,
            },
            "total_count must be positive",
        ),
        (
            {
                "objective": "by_district_abs_deviation",
                "target_values": [0.5],
                "pov_counts_col": "BVAP",
                "total_count": -1,
            },
            "total_count must be positive",
        ),
        (
            {
                "objective": "abs_deviation",
                "target": 0.5,
                "n_target_districts": 0,
                "pov_counts_col": "BVAP",
                "total_counts_col": "VAP",
            },
            "objective field 'n_target_districts' must be positive",
        ),
    ],
)
def test_objective_specs_reject_empty_or_nonpositive_values(spec, message):
    with pytest.raises(ValueError, match=message):
        validate_objective_spec(spec)
