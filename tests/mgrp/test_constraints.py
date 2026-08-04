"""Tests for the shared Constraints builder and its per-runner translation."""

import json
from typing import cast

import pytest

from gerrytools.mgrp import (
    Constraints,
    ForestRunInfo,
    ForestRunnerConfig,
    RecomRunInfo,
    RecomRunnerConfig,
    SMCRunInfo,
    SMCRunnerConfig,
)
from gerrytools.mgrp.constraints import ConstraintSpec, validate_constraint_spec


def test_builder_chains_and_accumulates():
    constraints = (
        Constraints()
        .group_hinge(strength=1500.0, group_pop_col="BVAP", total_pop_col="VAP")
        .splits(strength=100.0, admin_col="COUNTY")
    )
    specs = constraints.specs()
    assert [spec["constraint"] for spec in specs] == ["group_hinge", "splits"]
    assert specs[0]["targets"] == [0.55]
    # specs() returns copies; mutating them does not touch the builder.
    specs[0]["strength"] = 0
    assert constraints.specs()[0]["strength"] == 1500.0


def test_group_power_and_status_quo_builders_validate_and_accumulate():
    constraints = (
        Constraints()
        .group_power(
            strength=10.0,
            group_pop_col="BVAP",
            total_pop_col="VAP",
            target_group=0.4,
            target_other=0.6,
            pow=2.0,
        )
        .status_quo(strength=5.0, plan_col="CD")
    )

    assert constraints.specs() == [
        {
            "constraint": "group_power",
            "strength": 10.0,
            "group_pop_col": "BVAP",
            "total_pop_col": "VAP",
            "target_group": 0.4,
            "target_other": 0.6,
            "pow": 2.0,
        },
        {
            "constraint": "status_quo",
            "strength": 5.0,
            "plan_col": "CD",
        },
    ]


@pytest.mark.parametrize(
    ("spec", "exception", "message"),
    [
        (
            {
                "constraint": "group_power",
                "strength": 1.0,
                "group_pop_col": "BVAP",
                "total_pop_col": "VAP",
                "target_group": 0.5,
                "target_other": 0.5,
                "pow": 0,
            },
            ValueError,
            "pow.*positive",
        ),
        (
            {
                "constraint": "group_power",
                "strength": float("nan"),
                "group_pop_col": "BVAP",
                "total_pop_col": "VAP",
                "target_group": 0.5,
                "target_other": 0.5,
                "pow": 1.0,
            },
            ValueError,
            "strength.*finite",
        ),
        (
            {
                "constraint": "max_discontinuous_traversal_segments",
                "max_line_segments": 2,
            },
            ValueError,
            "must be 1",
        ),
        (
            {"constraint": "max_coarse_node_splits", "max_splits": -1},
            ValueError,
            "non-negative",
        ),
        (
            {
                "constraint": "allowed_excess_dists_in_coarse_nodes",
                "allowable_excess": -1,
            },
            ValueError,
            "non-negative",
        ),
        (
            {
                "constraint": "district_share_floor",
                "numerator_col": "BVAP",
                "denominator_cols": ["VAP"],
                "threshold": 1.1,
            },
            ValueError,
            r"threshold.*\[0, 1\]",
        ),
        (
            {
                "constraint": "group_hinge",
                "strength": 1.0,
                "group_pop_col": "BVAP",
                "total_pop_col": "VAP",
                "targets": [-0.1],
            },
            ValueError,
            r"targets.*\[0, 1\]",
        ),
        (
            {
                "constraint": "group_power",
                "strength": 1.0,
                "group_pop_col": "BVAP",
                "total_pop_col": "VAP",
                "target_group": 1.1,
                "target_other": 0.5,
                "pow": 1.0,
            },
            ValueError,
            r"target_group.*\[0, 1\]",
        ),
    ],
)
def test_raw_constraint_specs_reject_invalid_domains(spec, exception, message):
    with pytest.raises(exception, match=message):
        validate_constraint_spec(spec)


def test_nested_constraint_lists_do_not_alias_caller_or_builder_data():
    # Regression: shallow copies shared nested lists across the builder, its specs()
    # output, and emitted run configs, so mutating any one leaked into the others and
    # changed the config_hash after construction.
    constraints = Constraints().district_share_floor(
        numerator_col="BVAP", denominator_cols=["VAP"], threshold=0.4
    )
    emitted_specs = constraints.specs()
    cast(list, emitted_specs[0]["denominator_cols"]).append("CVAP")
    assert constraints.specs()[0]["denominator_cols"] == ["VAP"]

    runner = RecomRunnerConfig("./graphs/testing.json")
    run_info = RecomRunInfo(
        pop_col="TOTPOP", assignment_col="CD", variant="A", constraint=constraints
    )
    hash_before = runner.config_hash(run_info)
    emitted_config = cast(dict, runner.run_config(run_info))
    emitted_config["constraint"]["denominator_cols"].append("HVAP")

    assert runner.config_hash(run_info) == hash_before
    assert cast(dict, runner.run_config(run_info))["constraint"]["denominator_cols"] == ["VAP"]

    # Raw-dict path: an already-emitted config document is decoupled from the caller's list.
    denominator_cols = ["VAP"]
    raw_info = RecomRunInfo(
        pop_col="TOTPOP",
        assignment_col="CD",
        variant="A",
        constraint={
            "constraint": "district_share_floor",
            "numerator_col": "BVAP",
            "denominator_cols": denominator_cols,
            "threshold": 0.4,
        },
    )
    raw_config = cast(dict, runner.run_config(raw_info))
    denominator_cols.append("CVAP")
    assert raw_config["constraint"]["denominator_cols"] == ["VAP"]


def test_recom_constraints_travel_only_in_config():
    run_info = RecomRunInfo(
        pop_col="TOTPOP",
        assignment_col="CD",
        variant="A",
        constraint=Constraints().district_share_floor(
            numerator_col="BVAP", denominator_cols=["VAP"], threshold=0.4
        ),
    )
    runner = RecomRunnerConfig("./graphs/testing.json")
    command = runner.run_command(run_info)[2]
    config = json.loads(runner.run_command(run_info)[4])

    assert config["constraint"] == {
        "constraint": "district_share_floor",
        "numerator_col": "BVAP",
        "denominator_cols": ["VAP"],
        "threshold": 0.4,
    }
    assert "district_share_floor" not in command


def test_recom_rejects_multiple_constraints():
    two = Constraints()
    two.district_share_floor("BVAP", ["VAP"], 0.4)
    two.district_share_floor("HVAP", ["VAP"], 0.3)
    with pytest.raises(ValueError, match="single constraint"):
        RecomRunInfo(pop_col="TOTPOP", assignment_col="CD", variant="A", constraint=two)


def test_recom_rejects_smc_constraint_naming_engines():
    smc_only = Constraints().group_hinge(strength=1.0, group_pop_col="BVAP")
    with pytest.raises(ValueError, match="available on: smc"):
        RecomRunInfo(pop_col="TOTPOP", assignment_col="CD", variant="A", constraint=smc_only)


def test_forest_constraints_travel_only_in_config():
    run_info = ForestRunInfo(
        levels=["county", "precinct"],
        pop_col="TOTPOP",
        constraints=Constraints().max_coarse_node_splits(2).allowed_excess_dists_in_coarse_nodes(1),
    )
    runner = ForestRunnerConfig("./graphs/testing.json")
    command = runner.run_command(run_info)[2]
    config = json.loads(runner.run_command(run_info)[4])

    assert config["constraints"] == [
        {"constraint": "max_coarse_node_splits", "max_splits": 2},
        {
            "constraint": "allowed_excess_dists_in_coarse_nodes",
            "allowable_excess": 1,
        },
    ]
    assert "max_coarse_node_splits" not in command

    with pytest.raises(ValueError, match="forest runner does not support"):
        ForestRunInfo(
            levels=["county", "precinct"],
            pop_col="TOTPOP",
            constraints=Constraints().splits(strength=1.0, admin_col="COUNTY"),
        )


def test_smc_constraints_travel_only_in_config():
    run_info = SMCRunInfo(
        pop_col="TOTPOP",
        n_dists=4,
        n_sims=20,
        constraints=Constraints().group_hinge(
            strength=1500.0,
            group_pop_col="BVAP",
            total_pop_col="VAP",
            targets=[0.45, 0.55],
        ),
    )
    runner = SMCRunnerConfig("./shapefiles/testing")
    command = runner.run_command(run_info)[2]
    config = json.loads(runner.run_command(run_info)[4])

    assert config["constraints"] == [
        {
            "constraint": "group_hinge",
            "strength": 1500.0,
            "group_pop_col": "BVAP",
            "total_pop_col": "VAP",
            "targets": [0.45, 0.55],
        }
    ]
    assert "group_hinge" not in command

    with pytest.raises(ValueError, match="smc runner does not support"):
        SMCRunInfo(pop_col="TOTPOP", n_dists=4, n_sims=20, constraints=Constraints().pack_nodes())


def test_unknown_constraint_dict_rejected():
    with pytest.raises(ValueError, match="Unknown constraint"):
        ForestRunInfo(
            levels=["county", "precinct"],
            pop_col="TOTPOP",
            constraints=[cast(ConstraintSpec, {"constraint": "make_it_pretty"})],
        )


def test_raw_constraint_shape_and_types_rejected():
    base = {
        "constraint": "district_share_floor",
        "numerator_col": "BVAP",
        "denominator_cols": ["VAP"],
    }
    with pytest.raises(ValueError, match="missing"):
        RecomRunInfo(
            pop_col="TOTPOP",
            assignment_col="CD",
            variant="A",
            constraint=cast(ConstraintSpec, base),
        )
    with pytest.raises(TypeError, match="threshold"):
        RecomRunInfo(
            pop_col="TOTPOP",
            assignment_col="CD",
            variant="A",
            constraint=cast(ConstraintSpec, {**base, "threshold": "0.4"}),
        )


def test_constraint_column_names_must_be_nonempty():
    with pytest.raises(ValueError, match="numerator_col"):
        Constraints().district_share_floor("", ["VAP"], 0.4)
    with pytest.raises(ValueError, match="denominator_cols"):
        Constraints().district_share_floor("BVAP", [""], 0.4)


def test_raw_dict_still_accepted_for_recom():
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
    command = RecomRunnerConfig("./graphs/testing.json").run_command(run_info)
    config = json.loads(command[4])
    assert config["constraint"]["constraint"] == "district_share_floor"


def test_traversal_segments_other_than_one_rejected_at_construction():
    # MSMS asserts at runtime for any value but 1; the builder fails fast instead.
    with pytest.raises(ValueError, match="max_line_segments=1"):
        Constraints().max_discontinuous_traversal_segments(max_line_segments=2)
    assert Constraints().max_discontinuous_traversal_segments().specs() == [
        {"constraint": "max_discontinuous_traversal_segments", "max_line_segments": 1}
    ]


def test_incumbency_spec_no_longer_recognized():
    # The builder was removed because the Docker translation cannot produce a working
    # configuration; a raw spec dict now fails name validation too.
    with pytest.raises(ValueError, match="Unknown constraint"):
        SMCRunInfo(
            pop_col="TOTPOP",
            n_dists=4,
            n_sims=20,
            constraints=[
                cast(
                    ConstraintSpec,
                    {"constraint": "incumbency", "strength": 1.0, "incumbents_col": "INC"},
                )
            ],
        )
