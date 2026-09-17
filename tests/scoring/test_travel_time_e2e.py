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
    """Four units snapped to a one-way corridor: 10→20→30→40.

    Edge travel times (minutes): 10 between consecutive nodes.
    Unit 0→node10, 1→20, 2→30, 3→40.
    """
    G = nx.DiGraph()
    G.add_edge(10, 20, travel_time=10.0)
    G.add_edge(20, 30, travel_time=10.0)
    G.add_edge(30, 40, travel_time=10.0)
    # Return arcs so reverse travel is also defined (like a two-way road).
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


def test_e2e_paths_may_leave_district():
    """District {0, 2} routes 0→2 through node 20 even though unit 1 is elsewhere."""
    G, snaps = _corridor_graph()
    od = build_od_from_graph(G, snaps)
    pops = {0: 10.0, 1: 10.0, 2: 10.0, 3: 10.0}
    scorer = TravelTimeScorer(weight="none")
    d = scorer.district([0, 2], pops, od)
    assert d.ptt_mean == pytest.approx(20.0)
    assert d.ptt_max == pytest.approx(20.0)


def test_e2e_plan_through_public_api_knobs():
    G, snaps = _corridor_graph()
    od = build_od_from_graph(G, snaps)
    pops = {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0}
    # Two equal-size districts on the corridor.
    parts = {1: [0, 1], 2: [2, 3]}

    # District means are both 10; plan mean of PTT means is 10.
    assert pairwise_travel_time(parts, pops, od, weight="none", plan_agg="mean") == pytest.approx(
        10.0
    )
    assert pairwise_travel_time(
        parts, pops, od, weight="none", plan_agg="max_district"
    ) == pytest.approx(10.0)
    assert pairwise_travel_time(
        parts, pops, od, weight="none", plan_agg="global_max"
    ) == pytest.approx(10.0)

    # Metric objects (gerrytools-style reusable descriptors).
    ptt = PairwiseTravelTime(weight="none", plan_agg="mean")
    ctt = CentroidTravelTime(weight="none", plan_agg="mean")
    assert ptt.score(parts, pops, od) == pytest.approx(10.0)
    assert ctt.score(parts, pops, od) == pytest.approx(5.0)  # from center: times 10,0 → mean 5


def test_e2e_uneven_plan_global_max_and_mean_of_maxes():
    G, snaps = _corridor_graph()
    od = build_od_from_graph(G, snaps)
    pops = {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0}
    # Compact west vs long east span.
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


def test_e2e_weighted_p_times_p_changes_plan_mean():
    G, snaps = _corridor_graph()
    od = build_od_from_graph(G, snaps)
    pops = {0: 100.0, 1: 100.0, 2: 1.0, 3: 1.0}
    parts = {1: [0, 1, 2, 3]}  # one district = whole corridor

    unweighted = pairwise_travel_time(parts, pops, od, weight="none")
    people = pairwise_travel_time(parts, pops, od, weight="p*p")
    # People weight emphasizes the short 0–1 hop (100*100) over long sparse hops.
    assert people < unweighted
