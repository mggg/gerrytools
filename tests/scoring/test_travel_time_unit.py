"""Unit tests: pure formula pieces with a hand-built OD matrix (no routing)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

from gerrytools.scoring.travel_time import (
    TravelTimeTable,
    centroid_trip_weight,
    choose_centroid,
    load_travel_time_table,
    pairwise_trip_weight,
    plan_aggregate,
    save_travel_time_table,
    score_district,
    score_plan,
)
from gerrytools.scoring.travel_time.od_table import TravelTimeTable as OD


def _line_od() -> TravelTimeTable:
    """Three units on a line: 0--10--1--10--2 (end-to-end 20)."""
    times = np.array(
        [
            [0.0, 10.0, 20.0],
            [10.0, 0.0, 10.0],
            [20.0, 10.0, 0.0],
        ]
    )
    return TravelTimeTable.from_matrix([0, 1, 2], times)


def test_family_specific_trip_weight_modes():
    assert pairwise_trip_weight(3, 4, "none") == 1.0
    assert pairwise_trip_weight(3, 4, "p+p") == 7.0
    assert pairwise_trip_weight(3, 4, "p*p") == 12.0
    assert centroid_trip_weight(4, "none") == 1.0
    assert centroid_trip_weight(4, "p") == 4.0
    with pytest.raises(ValueError, match="pairwise weight"):
        pairwise_trip_weight(1, 1, "p")
    with pytest.raises(ValueError, match="centroid weight"):
        centroid_trip_weight(1, "p*p")


def test_od_table_rejects_bad_shapes():
    with pytest.raises(ValueError):
        OD.from_matrix([0, 1], np.zeros((2, 3)))
    with pytest.raises(ValueError):
        OD.from_matrix([0, 0], np.zeros((2, 2)))
    with pytest.raises(ValueError, match="side length"):
        OD.from_matrix([0, 1, 2], np.zeros((2, 2)))
    with pytest.raises(KeyError):
        _line_od().require_units([0, 99])


def test_od_table_lookups_for_units_outside_the_table():
    od = _line_od()
    assert od.has_unit(0)
    assert not od.has_unit(99)
    assert od.travel_time(0, 99) != od.travel_time(0, 99)  # NaN


def test_od_table_index_cannot_be_supplied_by_the_caller():
    # _index is derived in __post_init__, not an init field, so a caller can't
    # construct a table whose index disagrees with its unit_ids.
    import dataclasses

    init_names = {f.name for f in dataclasses.fields(OD) if f.init}
    assert "_index" not in init_names
    assert {f.name for f in dataclasses.fields(OD)} == {
        "unit_ids",
        "times",
        "_index",
        "_fully_reachable",
    }


def test_od_table_validates_statewide_connectivity():
    connected = _line_od()
    assert connected.is_fully_reachable
    connected.validate_connected()

    times = np.array(
        [
            [0.0, 10.0, np.nan],
            [10.0, 0.0, np.nan],
            [np.nan, np.nan, 0.0],
        ]
    )
    disconnected = TravelTimeTable.from_matrix(["a", "b", "island"], times)
    assert not disconnected.is_fully_reachable
    assert [len(component) for component in disconnected.connectivity_components()] == [2, 1]
    with pytest.raises(ValueError, match="requires statewide connectivity"):
        disconnected.validate_connected()


def test_od_table_copies_and_freezes_times_to_preserve_connectivity_invariant():
    source = np.array([[0.0, 5.0], [5.0, 0.0]])
    od = TravelTimeTable.from_matrix([0, 1], source)
    source[0, 1] = np.nan
    assert od.travel_time(0, 1) == pytest.approx(5.0)
    with pytest.raises(ValueError, match="read-only"):
        od.times[0, 1] = np.nan


def test_od_table_averages_directed_times_for_scoring():
    directed = np.array([[0.0, 5.0], [15.0, 0.0]])
    od = TravelTimeTable.from_directed_matrix(["a", "b"], directed)
    assert od.travel_time("a", "b") == pytest.approx(10.0)
    assert od.travel_time("b", "a") == pytest.approx(10.0)


@pytest.mark.parametrize(
    ("unit_ids", "times", "message"),
    [
        (["a", "b"], np.zeros((2, 3)), "must be square"),
        (["a", "b", "c"], np.zeros((2, 2)), "side length"),
        (["a", "b"], np.array([[1.0, 5.0], [5.0, 0.0]]), "diagonal"),
        (["a", "b"], np.array([[0.0, -5.0], [5.0, 0.0]]), "nonnegative"),
    ],
)
def test_directed_matrix_rejects_invalid_inputs(unit_ids, times, message):
    with pytest.raises(ValueError, match=message):
        TravelTimeTable.from_directed_matrix(unit_ids, times)


def test_od_table_rejects_an_asymmetric_scoring_matrix():
    with pytest.raises(ValueError, match="must be symmetric"):
        TravelTimeTable.from_matrix(["a", "b"], np.array([[0.0, 5.0], [15.0, 0.0]]))


@pytest.mark.parametrize(
    "times",
    [
        np.array([[1.0, 5.0], [5.0, 0.0]]),
        np.array([[0.0, -5.0], [-5.0, 0.0]]),
    ],
)
def test_od_table_rejects_invalid_scoring_values(times):
    with pytest.raises(ValueError):
        TravelTimeTable.from_matrix(["a", "b"], times)


def test_travel_time_artifact_round_trip(tmp_path: Path):
    path = tmp_path / "travel_times.npz"
    save_travel_time_table(
        path,
        _line_od(),
        time_unit="seconds",
        metadata={"state": "KS", "geography": "vtd2020"},
    )

    artifact = load_travel_time_table(path)
    assert artifact.table.unit_ids == (0, 1, 2)
    assert artifact.table.times == pytest.approx(_line_od().times)
    assert artifact.time_unit == "seconds"
    assert artifact.metadata == {"state": "KS", "geography": "vtd2020"}


def test_artifact_save_rejects_disconnected_table(tmp_path: Path):
    disconnected = TravelTimeTable.from_matrix(
        ["mainland", "island"],
        np.array([[0.0, np.nan], [np.nan, 0.0]]),
    )
    with pytest.raises(ValueError, match="requires statewide connectivity"):
        save_travel_time_table(tmp_path / "invalid.npz", disconnected)


def test_artifact_save_rejects_unsupported_values(tmp_path: Path):
    with pytest.raises(ValueError, match="time_unit"):
        save_travel_time_table(
            tmp_path / "bad-unit.npz",
            _line_od(),
            time_unit=cast(Any, "hours"),
        )
    with pytest.raises(TypeError, match="unit IDs"):
        save_travel_time_table(
            tmp_path / "bad-id.npz",
            TravelTimeTable.from_matrix([1.5], np.zeros((1, 1))),
        )
    with pytest.raises(TypeError, match="JSON-serializable"):
        save_travel_time_table(
            tmp_path / "bad-metadata.npz",
            _line_od(),
            metadata={"not_json": object()},
        )


def _write_artifact_parts(
    path: Path,
    *,
    unit_ids: object = (0, 1),
    metadata: object = None,
    include_times: bool = True,
) -> None:
    if metadata is None:
        metadata = {
            "schema_version": 1,
            "time_unit": "seconds",
            "symmetrization": "mean_of_directions",
            "matrix_dtype": "float32",
            "metadata": {},
        }
    values: dict[str, Any] = {
        "unit_ids_json": np.array(json.dumps(unit_ids)),
        "metadata_json": np.array(json.dumps(metadata)),
    }
    if include_times:
        values["times"] = np.array([[0.0, 5.0], [5.0, 0.0]], dtype=np.float32)
    np.savez_compressed(path, **values)


def test_artifact_load_rejects_missing_fields(tmp_path: Path):
    path = tmp_path / "missing.npz"
    _write_artifact_parts(path, include_times=False)
    with pytest.raises(ValueError, match="missing fields"):
        load_travel_time_table(path)


@pytest.mark.parametrize(
    ("unit_ids", "metadata", "message"),
    [
        ([True, 1], None, "unit_ids_json"),
        ([0, 1], [], "metadata_json"),
        ([0, 1], {"schema_version": 999}, "schema version"),
        (
            [0, 1],
            {"schema_version": 1, "symmetrization": "minimum"},
            "mean-of-directions",
        ),
        (
            [0, 1],
            {
                "schema_version": 1,
                "symmetrization": "mean_of_directions",
                "time_unit": "hours",
            },
            "travel-time unit",
        ),
        (
            [0, 1],
            {
                "schema_version": 1,
                "symmetrization": "mean_of_directions",
                "time_unit": "seconds",
                "metadata": [],
            },
            "metadata field",
        ),
    ],
)
def test_artifact_load_rejects_invalid_schema(tmp_path: Path, unit_ids, metadata, message):
    path = tmp_path / "invalid-schema.npz"
    _write_artifact_parts(path, unit_ids=unit_ids, metadata=metadata)
    with pytest.raises(ValueError, match=message):
        load_travel_time_table(path)


def test_ptt_unweighted_divides_by_number_of_pairs():
    # pairs 10+20+10 = 40 over 3 pairs
    scores = score_district([0, 1, 2], {0: 1.0, 1: 1.0, 2: 1.0}, _line_od())
    assert scores.ptt_mean == pytest.approx(40.0 / 3.0)
    assert scores.ptt_max == pytest.approx(20.0)


def test_ptt_two_units_one_pair():
    scores = score_district([0, 1], {0: 1.0, 1: 1.0}, _line_od())
    assert scores.ptt_mean == pytest.approx(10.0)


def test_ptt_p_times_p_keeps_weights_in_numerator():
    pops = {0: 100.0, 1: 100.0, 2: 1.0}
    scores = score_district([0, 1, 2], pops, _line_od(), ptt_weight="p*p")
    num = 10 * 10000 + 20 * 100 + 10 * 100
    assert scores.ptt_mean == pytest.approx(num / 3.0)


def test_ptt_p_plus_p_keeps_weights_in_numerator():
    pops = {0: 100.0, 1: 100.0, 2: 1.0}
    scores = score_district([0, 1, 2], pops, _line_od(), ptt_weight="p+p")
    num = 10 * 200 + 20 * 101 + 10 * 101
    assert scores.ptt_mean == pytest.approx(num / 3.0)


def test_centroid_unweighted_picks_middle():
    assert choose_centroid([0, 1, 2], {0: 1.0, 1: 1.0, 2: 1.0}, _line_od(), "none") == 1


def test_centroid_people_aware_prefers_populated_side():
    pops = {0: 100.0, 1: 1.0, 2: 0.0}
    assert choose_centroid([0, 1, 2], pops, _line_od(), "p") == 0


@pytest.mark.parametrize("bad_population", [np.nan, np.inf, -np.inf, -1.0, "unknown", True])
def test_population_weighted_centroid_rejects_invalid_populations(bad_population):
    populations = {0: 1.0, 1: bad_population, 2: 1.0}
    with pytest.raises(ValueError, match=r"Population for unit 1.*finite, nonnegative"):
        choose_centroid([0, 1, 2], populations, _line_od(), "p")


def test_population_weighted_centroid_rejects_missing_population():
    with pytest.raises(ValueError, match="Population is missing for unit 1"):
        choose_centroid([0, 1, 2], {0: 1.0, 2: 1.0}, _line_od(), "p")


def test_weighted_ptt_rejects_invalid_population_before_scoring():
    with pytest.raises(ValueError, match=r"Population for unit 1.*finite, nonnegative"):
        score_district(
            [0, 1, 2],
            {0: 1.0, 1: np.nan, 2: 1.0},
            _line_od(),
            ptt_weight="p+p",
        )


def test_centroid_population_weight_never_uses_candidate_population():
    pops = {0: 100.0, 1: 0.0, 2: 100.0}
    times = np.array([[0.0, 5.0, 20.0], [5.0, 0.0, 5.0], [20.0, 5.0, 0.0]])
    od = TravelTimeTable.from_matrix([0, 1, 2], times)
    assert choose_centroid([0, 1, 2], pops, od, "p") == 1


def test_centroid_returns_none_for_empty_units():
    assert choose_centroid([], {}, _line_od(), "none") is None


def test_centroid_rejects_a_disconnected_statewide_table():
    times = np.array(
        [
            [0.0, 10.0, np.nan],
            [10.0, 0.0, 10.0],
            [np.nan, 10.0, 0.0],
        ]
    )
    od = TravelTimeTable.from_matrix([0, 1, 2], times)
    with pytest.raises(ValueError, match="requires statewide connectivity"):
        choose_centroid([0, 1, 2], {0: 1.0, 1: 1.0, 2: 1.0}, od, "none")


def test_centroid_breaks_ties_by_input_order_for_unorderable_ids():
    class Unorderable:
        """A UnitId with no ``<``, to prove ties don't rely on comparing ids."""

        def __init__(self, label: str, h: int) -> None:
            self.label = label
            self._h = h

        def __hash__(self) -> int:
            return self._h

        def __eq__(self, other: object) -> bool:
            return isinstance(other, Unorderable) and self._h == other._h

    first, second = Unorderable("first", 1), Unorderable("second", 2)
    times = np.array([[0.0, 5.0], [5.0, 0.0]])
    od = TravelTimeTable.from_matrix([first, second], times)
    pops = {first: 10.0, second: 10.0}

    # Both candidates cost 5 (0 + 5), a tie; input order should pick `first`.
    assert choose_centroid([first, second], pops, od, "none") is first
    assert choose_centroid([second, first], pops, od, "none") is second


def test_ctt_divides_by_number_of_centroid_trips():
    scores = score_district([0, 1, 2], {0: 1.0, 1: 1.0, 2: 1.0}, _line_od())
    assert scores.centroid == 1
    # from middle: 10 + 0 + 10 = 20 over 3 trips
    assert scores.ctt_mean == pytest.approx(20.0 / 3.0)
    assert scores.ctt_max == pytest.approx(10.0)


def test_district_rejects_unreachable_pairs_instead_of_silently_dropping_them():
    times = np.array(
        [
            [0.0, 10.0, np.nan],
            [10.0, 0.0, 15.0],
            [np.nan, 15.0, 0.0],
        ]
    )
    od = TravelTimeTable.from_matrix([0, 1, 2], times)
    with pytest.raises(ValueError, match="requires statewide connectivity"):
        score_district([0, 1, 2], {0: 1.0, 1: 1.0, 2: 1.0}, od)


def test_district_rejects_a_fully_disconnected_statewide_table():
    times = np.array(
        [
            [0.0, np.nan, np.nan],
            [np.nan, 0.0, np.nan],
            [np.nan, np.nan, 0.0],
        ]
    )
    od = TravelTimeTable.from_matrix([0, 1, 2], times)
    with pytest.raises(ValueError, match="requires statewide connectivity"):
        score_district([0, 1, 2], {0: 1.0, 1: 1.0, 2: 1.0}, od)


def test_centroid_trips_rejects_a_missing_trip_from_a_caller_supplied_centroid():
    from gerrytools.scoring.travel_time.district import centroid_trips

    times = np.array([[0.0, 10.0, np.nan], [10.0, 0.0, 5.0], [np.nan, 5.0, 0.0]])
    od = TravelTimeTable.from_matrix([0, 1, 2], times)
    with pytest.raises(ValueError, match="Missing travel time"):
        centroid_trips([0, 1, 2], {0: 1.0, 1: 1.0, 2: 1.0}, od, "none", 0)


def test_all_zero_population_keeps_centroid_but_has_no_weighted_max():
    scores = score_district(
        [0, 1, 2],
        {0: 0.0, 1: 0.0, 2: 0.0},
        _line_od(),
        ptt_weight="p+p",
        ctt_weight="p",
    )
    assert scores.ptt_mean == pytest.approx(0.0)
    assert scores.ptt_max != scores.ptt_max  # NaN
    assert scores.centroid == 0
    assert scores.ctt_mean == pytest.approx(0.0)
    assert scores.ctt_max != scores.ctt_max  # NaN


def test_singleton_is_nan():
    scores = score_district([1], {1: 5.0}, _line_od())
    assert scores.centroid is None
    assert all(x != x for x in (scores.ptt_mean, scores.ptt_max, scores.ctt_mean, scores.ctt_max))


def test_singleton_still_requires_od_membership():
    with pytest.raises(KeyError, match="99"):
        score_district([99], {99: 5.0}, _line_od())


def test_weighted_singleton_still_validates_population():
    with pytest.raises(ValueError, match=r"Population for unit 1.*finite, nonnegative"):
        score_district([1], {1: np.nan}, _line_od(), ctt_weight="p")


def test_invalid_district_scores_propagate_to_plan_summaries():
    plan = score_plan(
        {"ordinary": [0, 1], "singleton": [2]},
        {0: 1.0, 1: 1.0, 2: 1.0},
        _line_od(),
    )
    assert all(
        value != value
        for value in (
            plan.ptt_mean,
            plan.ctt_mean,
            plan.ptt_max_district_mean,
            plan.ctt_max_district_mean,
            plan.ptt_global_max,
            plan.ctt_global_max,
            plan.ptt_mean_of_maxes,
            plan.ctt_mean_of_maxes,
        )
    )


def test_plan_layer_b_aggregates():
    pops = {0: 1.0, 1: 1.0, 2: 1.0}
    plan = score_plan({"A": [0, 1], "B": [0, 1, 2]}, pops, _line_od())
    # A: 10/1 = 10; B: 40/3
    assert plan.ptt_mean == pytest.approx((10.0 + 40.0 / 3.0) / 2.0)
    assert plan.ptt_max_district_mean == pytest.approx(40.0 / 3.0)
    assert plan.ptt_global_max == pytest.approx(20.0)
    assert plan.ptt_mean_of_maxes == pytest.approx((10.0 + 20.0) / 2.0)
    assert plan_aggregate(plan, "ptt", "mean") == plan.ptt_mean


@pytest.mark.parametrize(
    ("family", "agg", "expected_attr"),
    [
        ("ptt", "mean", "ptt_mean"),
        ("ptt", "max_district", "ptt_max_district_mean"),
        ("ptt", "global_max", "ptt_global_max"),
        ("ptt", "mean_of_maxes", "ptt_mean_of_maxes"),
        ("ctt", "mean", "ctt_mean"),
        ("ctt", "max_district", "ctt_max_district_mean"),
        ("ctt", "global_max", "ctt_global_max"),
        ("ctt", "mean_of_maxes", "ctt_mean_of_maxes"),
    ],
)
def test_plan_aggregate_covers_every_family_and_agg(family, agg, expected_attr):
    pops = {0: 1.0, 1: 1.0, 2: 1.0}
    plan = score_plan({"A": [0, 1], "B": [0, 1, 2]}, pops, _line_od())
    assert plan_aggregate(plan, family, agg) == getattr(plan, expected_attr)


def test_plan_aggregate_rejects_unknown_agg():
    pops = {0: 1.0, 1: 1.0}
    plan = score_plan({"A": [0, 1]}, pops, _line_od())
    with pytest.raises(ValueError, match="Unknown family/agg"):
        plan_aggregate(plan, "ptt", "bogus")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Unknown family/agg"):
        plan_aggregate(plan, "ctt", "bogus")  # type: ignore[arg-type]
