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
    with pytest.raises(KeyError):
        _line_od().require_units([0, 99])


def test_ptt_mean_unweighted_line():
    scores = score_district([0, 1, 2], {0: 1.0, 1: 1.0, 2: 1.0}, _line_od(), "none")
    assert scores.ptt_mean == pytest.approx(40.0 / 3.0)
    assert scores.ptt_max == pytest.approx(20.0)


def test_ptt_mean_p_times_p_emphasizes_large_pair():
    pops = {0: 100.0, 1: 100.0, 2: 1.0}
    scores = score_district([0, 1, 2], pops, _line_od(), "p*p")
    num = 10 * 10000 + 20 * 100 + 10 * 100
    den = 10000 + 100 + 100
    assert scores.ptt_mean == pytest.approx(num / den)


def test_ptt_mean_p_plus_p():
    pops = {0: 100.0, 1: 100.0, 2: 1.0}
    scores = score_district([0, 1, 2], pops, _line_od(), "p+p")
    # pairs (0,1): 10*200; (0,2): 20*101; (1,2): 10*101
    num = 10 * 200 + 20 * 101 + 10 * 101
    den = 200 + 101 + 101
    assert scores.ptt_mean == pytest.approx(num / den)


def test_centroid_unweighted_picks_middle():
    assert choose_centroid([0, 1, 2], {0: 1.0, 1: 1.0, 2: 1.0}, _line_od(), "none") == 1


def test_centroid_people_aware_prefers_populated_side():
    pops = {0: 100.0, 1: 1.0, 2: 0.0}
    assert choose_centroid([0, 1, 2], pops, _line_od(), "p*p") == 0


def test_ctt_mean_and_max_from_middle():
    scores = score_district([0, 1, 2], {0: 1.0, 1: 1.0, 2: 1.0}, _line_od(), "none")
    assert scores.centroid == 1
    assert scores.ctt_mean == pytest.approx(20.0 / 3.0)
    assert scores.ctt_max == pytest.approx(10.0)


def test_singleton_is_nan():
    scores = score_district([1], {1: 5.0}, _line_od(), "none")
    assert scores.centroid is None
    assert all(x != x for x in (scores.ptt_mean, scores.ptt_max, scores.ctt_mean, scores.ctt_max))


def test_plan_layer_b_aggregates():
    pops = {0: 1.0, 1: 1.0, 2: 1.0}
    plan = score_plan({"A": [0, 1], "B": [0, 1, 2]}, pops, _line_od(), "none")
    assert plan.ptt_mean == pytest.approx((10.0 + 40.0 / 3.0) / 2.0)
    assert plan.ptt_max_district_mean == pytest.approx(40.0 / 3.0)
    assert plan.ptt_global_max == pytest.approx(20.0)
    assert plan.ptt_mean_of_maxes == pytest.approx((10.0 + 20.0) / 2.0)
    assert plan_aggregate(plan, "ptt", "mean") == plan.ptt_mean
