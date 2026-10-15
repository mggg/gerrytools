"""Projection selection and secondary geometry alignment for scoring."""

import geopandas as gpd
import networkx as nx
import numpy as np
import pytest
from geopandas.testing import assert_geodataframe_equal
from pyproj import CRS
from shapely.geometry import box

from gerrytools._geodataframe import aligned_geodataframe
from gerrytools.scoring import (
    ConvexHullRatio,
    PlanEvaluator,
    PolsbyPopper,
    PopulationPolygon,
    Reock,
    Schwartzberg,
    StateClippedConvexHullRatio,
    state_clipped_convex_hull_ratio,
)


def units() -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        {"population": [10, 20]},
        geometry=[box(-105.02, 40, -105.01, 40.01), box(-105.01, 40, -105, 40.01)],
        crs="EPSG:4326",
    )


@pytest.mark.parametrize("southern_hemisphere", [False, True])
@pytest.mark.parametrize("metric", [Reock(), PolsbyPopper(), Schwartzberg(), ConvexHullRatio()])
def test_automatic_projection_matches_explicit_projection(
    metric: Reock | PolsbyPopper | Schwartzberg | ConvexHullRatio,
    southern_hemisphere: bool,
) -> None:
    frame = units()
    if southern_hemisphere:
        frame.geometry = frame.geometry.translate(yoff=-80)
    original = frame.copy()
    projected = frame.to_crs(frame.estimate_utm_crs())
    assert projected.crs is not None
    automatic = PlanEvaluator(nx.path_graph(2), geometry=frame).add_metric(metric)
    explicit = PlanEvaluator(nx.path_graph(2), geometry=projected).add_metric(metric)
    for assignment in ([0, 0], [0, 1]):
        np.testing.assert_allclose(
            automatic.evaluate(assignment)[metric._default_name()],
            explicit.evaluate(assignment)[metric._default_name()],
        )
    assert_geodataframe_equal(frame, original)


def test_projection_is_chosen_once_and_explicit_override_skips_estimation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = units()
    chosen = frame.estimate_utm_crs()
    calls = []

    def estimate(self: gpd.GeoDataFrame) -> CRS:
        calls.append(len(self))
        return chosen

    monkeypatch.setattr(gpd.GeoDataFrame, "estimate_utm_crs", estimate)
    scorer = PlanEvaluator(nx.path_graph(2), geometry=frame).add_metric(Reock())
    scorer.evaluate([0, 0])
    scorer.evaluate([0, 1])
    assert calls == [2]
    calls.clear()
    for source, target in [(frame, "EPSG:5070"), (frame.to_crs(chosen), None)]:
        result = aligned_geodataframe([0, 1], source, crs=target)
        assert result.crs == (target or chosen)
    assert not calls


def test_projection_failure_requires_explicit_crs() -> None:
    polar = gpd.GeoDataFrame(geometry=[box(0, 88, 1, 89)], crs="EPSG:4326")
    with pytest.raises(ValueError, match="explicit projected CRS"):
        aligned_geodataframe([0], polar)
    with pytest.raises(ValueError, match="projected CRS"):
        aligned_geodataframe([0], polar, crs="EPSG:4326")


@pytest.mark.parametrize("population_crs", ["EPSG:4326", "EPSG:5070"])
def test_population_polygons_follow_scoring_projection(population_crs: str) -> None:
    frame = units()
    population = gpd.GeoDataFrame(
        {"population": [10, 20]},
        geometry=[box(-105.019, 40.001, -105.011, 40.009), box(-105.009, 40.001, -105.001, 40.009)],
        crs=frame.crs,
    ).to_crs(population_crs)
    original = population.copy()
    projected = frame.to_crs(frame.estimate_utm_crs())
    assert projected.crs is not None
    automatic = PlanEvaluator(nx.path_graph(2), geometry=frame).add_metric(
        PopulationPolygon("population", population_units=population)
    )
    explicit = PlanEvaluator(nx.path_graph(2), geometry=projected).add_metric(
        PopulationPolygon("population", population_units=population.to_crs(projected.crs))
    )
    np.testing.assert_allclose(
        automatic.evaluate([0, 1])["population_polygon"],
        explicit.evaluate([0, 1])["population_polygon"],
    )
    assert_geodataframe_equal(population, original)


def test_state_boundary_crs_matches_explicitly_projected_boundary() -> None:
    frame = units()
    state = box(-105.03, 39.99, -104.99, 40.02)
    projected = frame.to_crs(frame.estimate_utm_crs())
    assert projected.crs is not None
    projected_state = gpd.GeoSeries([state], crs=frame.crs).to_crs(projected.crs).iloc[0]
    expected = state_clipped_convex_hull_ratio(projected, [0, 0], state_geometry=projected_state)
    actual = state_clipped_convex_hull_ratio(
        frame, [0, 0], state_geometry=state, state_crs=frame.crs
    )
    np.testing.assert_allclose(actual, expected)
    scorer = PlanEvaluator(nx.path_graph(2), geometry=frame).add_metric(
        StateClippedConvexHullRatio(state, state_crs=frame.crs)
    )
    np.testing.assert_allclose(scorer.evaluate([0, 0])["state_clipped_convex_hull_ratio"], expected)
