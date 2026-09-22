"""End-to-end tests: tiny road graph → OD table → district/plan → public API.

No OSM download. This is the full scoring pipeline on a synthetic routable network.
"""

from __future__ import annotations

import networkx as nx
import pytest

from gerrytools.scoring.travel_time import (
    CentroidTravelTime,
    PairwiseTravelTime,
    TravelTimeScorer,
    build_od_from_graph,
    centroid_travel_time,
    pairwise_travel_time,
)


def _corridor_graph() -> tuple[nx.DiGraph, dict[int, int]]:
    """Four units snapped to a corridor: 10↔20↔30↔40, 10 min per hop."""
    G = nx.DiGraph()
    G.add_edge(10, 20, travel_time=10.0)
    G.add_edge(20, 30, travel_time=10.0)
    G.add_edge(30, 40, travel_time=10.0)
    G.add_edge(20, 10, travel_time=10.0)
    G.add_edge(30, 20, travel_time=10.0)
    G.add_edge(40, 30, travel_time=10.0)
    unit_to_node = {0: 10, 1: 20, 2: 30, 3: 40}
    return G, unit_to_node


def test_e2e_build_od_shortest_paths():
    G, snaps = _corridor_graph()
    od = build_od_from_graph(G, snaps)
    assert od.travel_time(0, 0) == pytest.approx(0.0)
    assert od.travel_time(0, 1) == pytest.approx(10.0)
    assert od.travel_time(0, 3) == pytest.approx(30.0)
    assert od.travel_time(3, 0) == pytest.approx(30.0)


def test_build_od_rejects_snap_node_not_in_graph():
    G, snaps = _corridor_graph()
    snaps[4] = 999  # 999 was never added to G
    with pytest.raises(KeyError, match="Snap nodes not in graph"):
        build_od_from_graph(G, snaps)


def test_build_od_rejects_edge_missing_the_weight_attribute():
    G, snaps = _corridor_graph()
    G.add_edge(40, 10)  # no travel_time attribute
    with pytest.raises(ValueError, match="missing the 'travel_time' attribute"):
        build_od_from_graph(G, snaps)


def test_build_od_marks_genuinely_disconnected_units_as_nan():
    G, snaps = _corridor_graph()
    G.add_edge(50, 60, travel_time=1.0)  # a second, disconnected component
    snaps[4] = 50
    od = build_od_from_graph(G, snaps)
    assert od.travel_time(0, 4) != od.travel_time(0, 4)  # NaN: no path exists


def test_e2e_paths_may_leave_district():
    """District {0, 2} routes 0→2 through node 20 even though unit 1 is elsewhere."""
    G, snaps = _corridor_graph()
    od = build_od_from_graph(G, snaps)
    pops = {0: 10.0, 1: 10.0, 2: 10.0, 3: 10.0}
    scorer = TravelTimeScorer(weight="none")
    d = scorer.district([0, 2], pops, od)
    assert d.ptt_mean == pytest.approx(20.0)  # one pair
    assert d.ptt_max == pytest.approx(20.0)


def test_e2e_plan_through_public_api_knobs():
    G, snaps = _corridor_graph()
    od = build_od_from_graph(G, snaps)
    pops = {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0}
    parts = {1: [0, 1], 2: [2, 3]}

    # Each district: one pair of time 10 → 10
    assert pairwise_travel_time(parts, pops, od, weight="none", plan_agg="mean") == pytest.approx(
        10.0
    )
    assert pairwise_travel_time(
        parts, pops, od, weight="none", plan_agg="max_district"
    ) == pytest.approx(10.0)
    assert pairwise_travel_time(
        parts, pops, od, weight="none", plan_agg="global_max"
    ) == pytest.approx(10.0)

    ptt = PairwiseTravelTime(weight="none", plan_agg="mean")
    ctt = CentroidTravelTime(weight="none", plan_agg="mean")
    assert ptt.score(parts, pops, od) == pytest.approx(10.0)
    # CTT: times 10 and 0 → 10/2
    assert ctt.score(parts, pops, od) == pytest.approx(5.0)


def test_e2e_uneven_plan_global_max_and_mean_of_maxes():
    G, snaps = _corridor_graph()
    od = build_od_from_graph(G, snaps)
    pops = {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0}
    parts = {"west": [0, 1], "east": [0, 1, 2, 3]}

    plan = TravelTimeScorer("none").evaluate(parts, pops, od)
    assert plan.by_district["west"].ptt_mean == pytest.approx(10.0)
    assert plan.by_district["east"].ptt_mean == pytest.approx((10 + 20 + 30 + 10 + 20 + 10) / 6.0)
    assert plan.ptt_global_max == pytest.approx(30.0)
    assert plan.ptt_mean_of_maxes == pytest.approx((10.0 + 30.0) / 2.0)
    assert plan.ptt_max_district_mean == pytest.approx(plan.by_district["east"].ptt_mean)

    assert centroid_travel_time(parts, pops, od, plan_agg="global_max") == pytest.approx(
        plan.ctt_global_max
    )


def test_e2e_weighted_p_times_p_keeps_population_in_units():
    G, snaps = _corridor_graph()
    od = build_od_from_graph(G, snaps)
    pops = {0: 100.0, 1: 100.0, 2: 1.0, 3: 1.0}
    parts = {1: [0, 1, 2, 3]}

    unweighted = pairwise_travel_time(parts, pops, od, weight="none")
    people = pairwise_travel_time(parts, pops, od, weight="p*p")
    # Same /#pairs; p*p leaves people² in the numerator so the score is larger.
    assert people > unweighted
