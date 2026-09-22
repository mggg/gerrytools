"""Unit tests: pure formula pieces with a hand-built OD matrix (no routing)."""

from __future__ import annotations

import numpy as np
import pytest

from gerrytools.scoring.travel_time import (
    TravelTimeTable,
    choose_centroid,
    plan_aggregate,
    score_district,
    score_plan,
    trip_weight,
)
from gerrytools.scoring.travel_time.od_table import TravelTimeTable as OD
from gerrytools.scoring.travel_time.weights import trip_weight as tw


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


def test_trip_weight_modes():
    assert trip_weight(3, 4, "none") == 1.0
    assert trip_weight(3, 4, "p+p") == 7.0
    assert trip_weight(3, 4, "p*p") == 12.0
    with pytest.raises(ValueError):
        tw(1, 1, "nope")  # type: ignore[arg-type]


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
    assert {f.name for f in dataclasses.fields(OD)} == {"unit_ids", "times", "_index"}


def test_ptt_unweighted_divides_by_number_of_pairs():
    # pairs 10+20+10 = 40 over 3 pairs
    scores = score_district([0, 1, 2], {0: 1.0, 1: 1.0, 2: 1.0}, _line_od(), "none")
    assert scores.ptt_mean == pytest.approx(40.0 / 3.0)
    assert scores.ptt_max == pytest.approx(20.0)


def test_ptt_two_units_one_pair():
    scores = score_district([0, 1], {0: 1.0, 1: 1.0}, _line_od(), "none")
    assert scores.ptt_mean == pytest.approx(10.0)


def test_ptt_p_times_p_keeps_weights_in_numerator():
    pops = {0: 100.0, 1: 100.0, 2: 1.0}
    scores = score_district([0, 1, 2], pops, _line_od(), "p*p")
    num = 10 * 10000 + 20 * 100 + 10 * 100
    assert scores.ptt_mean == pytest.approx(num / 3.0)


def test_ptt_p_plus_p_keeps_weights_in_numerator():
    pops = {0: 100.0, 1: 100.0, 2: 1.0}
    scores = score_district([0, 1, 2], pops, _line_od(), "p+p")
    num = 10 * 200 + 20 * 101 + 10 * 101
    assert scores.ptt_mean == pytest.approx(num / 3.0)


def test_centroid_unweighted_picks_middle():
    assert choose_centroid([0, 1, 2], {0: 1.0, 1: 1.0, 2: 1.0}, _line_od(), "none") == 1


def test_centroid_people_aware_prefers_populated_side():
    pops = {0: 100.0, 1: 1.0, 2: 0.0}
    assert choose_centroid([0, 1, 2], pops, _line_od(), "p*p") == 0


def test_centroid_returns_none_for_empty_units():
    assert choose_centroid([], {}, _line_od(), "none") is None


def test_centroid_skips_a_candidate_that_cannot_reach_every_unit():
    times = np.array(
        [
            [0.0, 10.0, np.nan],
            [10.0, 0.0, 10.0],
            [np.nan, 10.0, 0.0],
        ]
    )
    od = TravelTimeTable.from_matrix([0, 1, 2], times)
    # 0 and 2 can't reach each other, so only 1 is a valid centroid candidate.
    assert choose_centroid([0, 1, 2], {0: 1.0, 1: 1.0, 2: 1.0}, od, "none") == 1


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
    scores = score_district([0, 1, 2], {0: 1.0, 1: 1.0, 2: 1.0}, _line_od(), "none")
    assert scores.centroid == 1
    # from middle: 10 + 0 + 10 = 20 over 3 trips
    assert scores.ctt_mean == pytest.approx(20.0 / 3.0)
    assert scores.ctt_max == pytest.approx(10.0)


def test_district_skips_unreachable_pairs_but_still_scores_the_rest():
    times = np.array(
        [
            [0.0, 10.0, np.nan],
            [10.0, 0.0, 15.0],
            [np.nan, 15.0, 0.0],
        ]
    )
    od = TravelTimeTable.from_matrix([0, 1, 2], times)
    scores = score_district([0, 1, 2], {0: 1.0, 1: 1.0, 2: 1.0}, od, "none")
    # Only the 0-1 and 1-2 pairs are reachable: (10 + 15) / 2.
    assert scores.ptt_mean == pytest.approx(12.5)


def test_district_all_pairs_unreachable_has_no_ptt_and_no_centroid():
    times = np.array(
        [
            [0.0, np.nan, np.nan],
            [np.nan, 0.0, np.nan],
            [np.nan, np.nan, 0.0],
        ]
    )
    od = TravelTimeTable.from_matrix([0, 1, 2], times)
    scores = score_district([0, 1, 2], {0: 1.0, 1: 1.0, 2: 1.0}, od, "none")
    assert scores.ptt_mean != scores.ptt_mean  # NaN: zero reachable pairs
    assert scores.centroid is None


def test_centroid_trips_skips_a_trip_missing_from_a_caller_supplied_centroid():
    # score_district always picks a centroid that reaches every unit (see
    # choose_centroid), so this path only matters if centroid_trips is called
    # directly with an arbitrary centroid, which it's public enough to allow.
    from gerrytools.scoring.travel_time.district import centroid_trips

    times = np.array([[0.0, 10.0, np.nan], [10.0, 0.0, 5.0], [np.nan, 5.0, 0.0]])
    od = TravelTimeTable.from_matrix([0, 1, 2], times)
    times_out, weights_out = centroid_trips([0, 1, 2], {0: 1.0, 1: 1.0, 2: 1.0}, od, "none", 0)
    assert times_out == [0.0, 10.0]  # 0->2 is NaN and dropped
    # p+p weight is 0 for every trip when every population is 0: the mean is a
    # legitimate 0, but there's no *active* (weight > 0) trip to take a max
    # over, and a people-aware centroid has no candidate with population > 0.
    scores = score_district([0, 1, 2], {0: 0.0, 1: 0.0, 2: 0.0}, _line_od(), "p+p")
    assert scores.ptt_mean == pytest.approx(0.0)
    assert scores.ptt_max != scores.ptt_max  # NaN
    assert scores.centroid is None
    assert scores.ctt_mean != scores.ctt_mean  # NaN


def test_singleton_is_nan():
    scores = score_district([1], {1: 5.0}, _line_od(), "none")
    assert scores.centroid is None
    assert all(x != x for x in (scores.ptt_mean, scores.ptt_max, scores.ctt_mean, scores.ctt_max))


def test_plan_layer_b_aggregates():
    pops = {0: 1.0, 1: 1.0, 2: 1.0}
    plan = score_plan({"A": [0, 1], "B": [0, 1, 2]}, pops, _line_od(), "none")
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
    plan = score_plan({"A": [0, 1], "B": [0, 1, 2]}, pops, _line_od(), "none")
    assert plan_aggregate(plan, family, agg) == getattr(plan, expected_attr)


def test_plan_aggregate_rejects_unknown_agg():
    pops = {0: 1.0, 1: 1.0}
    plan = score_plan({"A": [0, 1]}, pops, _line_od(), "none")
    with pytest.raises(ValueError, match="Unknown family/agg"):
        plan_aggregate(plan, "ptt", "bogus")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Unknown family/agg"):
        plan_aggregate(plan, "ctt", "bogus")  # type: ignore[arg-type]
