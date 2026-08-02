"""Live tests for the pinned mgrp Docker image."""

import csv
import json
import os
from pathlib import Path

import geopandas as gpd
import pytest

from gerrytools.mgrp import (
    Constraints,
    ForestRunInfo,
    ForestRunnerConfig,
    Objective,
    RecomRunInfo,
    RecomRunnerConfig,
    RunContainer,
    ShortBurstsRunInfo,
    SMCRunInfo,
    SMCRunnerConfig,
    TiltedRunInfo,
)

pytestmark = pytest.mark.mgrp_live
FIXTURES = Path(__file__).parents[2] / "fixtures"


def metadata_path(output_path):
    output_path = Path(output_path)
    return output_path.with_name(f"{output_path.stem}_metadata.jsonl")


def assert_provenance(runner, run_args, output_path):
    # Every runner ships its config as the command's "$1" argv slot.
    raw_config = runner.run_command(*run_args)[4]
    sidecar = metadata_path(output_path)
    assert sidecar.read_text().rstrip("\n") == raw_config
    assert json.loads(sidecar.read_text()) == runner.run_config(*run_args)
    if hasattr(os, "getuid"):
        assert Path(output_path).stat().st_uid == os.getuid()
        assert sidecar.stat().st_uid == os.getuid()


def read_jsonl(output_path):
    records = [
        json.loads(line) for line in Path(output_path).read_text().splitlines() if line.strip()
    ]
    assert records
    return records


def assert_assignment_jsonl(output_path, *, node_count, district_count):
    records = read_jsonl(output_path)
    for record in records:
        assert isinstance(record, dict)
        assert isinstance(record.get("sample"), int) and not isinstance(record["sample"], bool)
        assignment = record.get("assignment")
        assert isinstance(assignment, list)
        assert len(assignment) == node_count
        assert all(isinstance(label, int) and not isinstance(label, bool) for label in assignment)
        assert len(set(assignment)) == district_count


def assert_inline_recom_provenance(runner, run_info, output_path):
    records = read_jsonl(output_path)
    raw_config = runner.run_command(run_info)[4]
    assert records[0] == {"meta": {"config": raw_config}}
    assert "init" in records[1]
    assert not metadata_path(output_path).exists()


def read_nonempty_csv(path):
    with Path(path).open(newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        rows = list(reader)
    assert reader.fieldnames
    assert rows
    return reader.fieldnames, rows


def graph_node_count(path):
    return len(json.loads(Path(path).read_text())["nodes"])


@pytest.mark.parametrize(
    "writer",
    [
        "tsv",
        "jsonl",
        "pcompress",
        "jsonl-full",
        "assignments",
        "canonicalized-assignments",
        "canonical",
        "ben",
        "bendl",
    ],
)
def test_recom_config_transport_and_provenance(mgrp_image, tmp_path, writer):
    graph = FIXTURES / "mgrp_7x7.json"
    runner = RecomRunnerConfig(
        str(graph),
        output_folder=str(tmp_path / "output"),
        log_folder=str(tmp_path / "logs"),
    )
    run_info = RecomRunInfo(
        pop_col="TOTPOP",
        assignment_col="district",
        variant="A",
        n_steps=2,
        pop_tol=0.2,
        writer=writer,
        constraint=Constraints().district_share_floor(
            numerator_col="TOTPOP",
            denominator_cols=["TOTPOP"],
            threshold=0.5,
        ),
    )

    with RunContainer(runner, docker_image_name=mgrp_image) as container:
        output_path = container.run(run_info)

    assert output_path is not None
    if writer == "canonical":
        assert_assignment_jsonl(
            output_path,
            node_count=graph_node_count(graph),
            district_count=4,
        )
        assert_provenance(runner, (run_info,), output_path)
    else:
        if writer in ("jsonl", "jsonl-full"):
            assert_inline_recom_provenance(runner, run_info, output_path)
        else:
            for expected in runner.expected_files(run_info):
                assert Path(expected).is_file()
                assert Path(expected).stat().st_size > 0


def test_recom_streaming_apis(mgrp_image, tmp_path):
    graph = FIXTURES / "mgrp_7x7.json"
    runner = RecomRunnerConfig(
        str(graph),
        output_folder=str(tmp_path / "output"),
        log_folder=str(tmp_path / "logs"),
    )
    run_info = RecomRunInfo(
        pop_col="TOTPOP",
        assignment_col="district",
        variant="A",
        n_steps=2,
        pop_tol=0.2,
        writer="canonical",
        force_print=True,
        updaters={"district_count": lambda partition: len(partition.parts)},
    )

    with RunContainer(runner, docker_image_name=mgrp_image) as container:
        raw_records = [record for record, _ in container.run_iter(run_info) if record is not None]
        updated_records = [
            record for record, _ in container.mcmc_run_with_updaters(run_info) if record is not None
        ]

    assert raw_records
    for record in raw_records:
        assert len(record["assignment"]) == graph_node_count(graph)
        assert len(set(record["assignment"])) == 4
    assert updated_records
    assert all(record["updaters"]["district_count"] == 4 for record in updated_records)


def test_optimizers_run_and_write_scores(mgrp_image, tmp_path):
    graph = FIXTURES / "mgrp_7x7.json"
    runner = RecomRunnerConfig(
        str(graph),
        output_folder=str(tmp_path / "output"),
        log_folder=str(tmp_path / "logs"),
    )
    objective = Objective.by_district_abs_deviation(
        target_values=[0.25, 0.25, 0.25, 0.25],
        pov_counts_col="TOTPOP",
        total_counts_col="TOTPOP",
    )
    short_bursts = ShortBurstsRunInfo(
        pop_col="TOTPOP",
        assignment_col="district",
        objective=objective,
        burst_length=5,
        n_steps=20,
        pop_tol=0.2,
        maximize=False,
    )
    tilted = TiltedRunInfo(
        pop_col="TOTPOP",
        assignment_col="district",
        objective=objective,
        n_steps=20,
        pop_tol=0.2,
        accept_rule="fixed",
        accept_worse_prob=0.05,
        maximize=False,
    )

    # Ownership is handed back to the invoking user when the container exits,
    # so run both optimizers first and assert afterwards.
    outputs = []
    with RunContainer(runner, docker_image_name=mgrp_image) as container:
        for run_info in (short_bursts, tilted):
            outputs.append((run_info, container.run(run_info)))

    for run_info, output_path in outputs:
        assert output_path is not None
        assert_assignment_jsonl(
            output_path,
            node_count=graph_node_count(graph),
            district_count=4,
        )
        scores_file = runner.scores_file(run_info)
        assert scores_file is not None
        read_nonempty_csv(scores_file)
        assert_provenance(runner, (run_info,), output_path)


def test_forest_config_transport_and_provenance(mgrp_image, tmp_path):
    graph = FIXTURES / "mgrp_7x7.json"
    runner = ForestRunnerConfig(
        str(graph),
        output_folder=str(tmp_path / "output"),
        log_folder=str(tmp_path / "logs"),
    )
    run_info = ForestRunInfo(
        levels=["county", "precinct"],
        pop_col="TOTPOP",
        num_dists=4,
        n_steps=2,
        pop_dev=0.2,
        constraints=Constraints().max_coarse_node_splits(5),
    )

    with RunContainer(runner, docker_image_name=mgrp_image) as container:
        output_path = container.run(run_info)

    assert output_path is not None
    assert_assignment_jsonl(
        output_path,
        node_count=graph_node_count(graph),
        district_count=run_info.num_dists,
    )
    assert_provenance(runner, (run_info,), output_path)


def test_smc_config_transport_and_provenance(mgrp_image, tmp_path):
    geopackage = FIXTURES / "testing_12x12.gpkg"
    runner = SMCRunnerConfig(
        str(geopackage),
        output_folder=str(tmp_path / "output"),
        log_folder=str(tmp_path / "logs"),
    )
    run_info = SMCRunInfo(
        pop_col="tot_pop",
        n_dists=4,
        pop_tol=0.2,
        n_sims=20,
        tally_columns=["tot_pop"],
        constraints=Constraints().group_hinge(
            strength=1.0,
            group_pop_col="maj_pop",
            total_pop_col="tot_pop",
        ),
    )

    with RunContainer(runner, docker_image_name=mgrp_image) as container:
        output_path = container.run(run_info)

    assert output_path is not None
    assert_assignment_jsonl(
        output_path,
        node_count=len(gpd.read_file(geopackage)),
        district_count=run_info.n_dists,
    )
    assert_provenance(runner, (run_info,), output_path)

    # Tally columns survive the jsonl pipeline via the CSV sidecar.
    tally_path = Path(output_path).with_name(f"{Path(output_path).stem}_tallies.csv")
    tally_columns, _ = read_nonempty_csv(tally_path)
    assert "tot_pop" in tally_columns
