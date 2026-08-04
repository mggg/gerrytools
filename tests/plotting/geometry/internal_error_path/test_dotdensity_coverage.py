import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest
from geopandas import GeoDataFrame, GeoSeries
from shapely import contains_xy
from shapely.geometry import LineString, Polygon, box

from gerrytools.plotting.geometry._dot_sampling import _make_random_points, _random_xy_in_poly
from gerrytools.plotting.geometry.dotdensity import DotDensityPlot


# ===========================
# == RANDOM POINTS IN POLY ==
# ===========================
class TestRandomXYInPoly:
    def test_zero_area_polygon_raises(self):
        """A degenerate polygon (zero area) raises ValueError."""
        from shapely.geometry import LineString

        rng = np.random.default_rng(42)
        # A line has zero area
        line = LineString([(0, 0), (1, 1)])
        with pytest.raises(ValueError, match="zero area"):
            _random_xy_in_poly(line, 5, rng=rng)

    def test_axis_aligned_degenerate_geometry_raises_value_error(self):
        """A flat axis-aligned line has a zero-area bounding box; no ZeroDivisionError."""
        rng = np.random.default_rng(42)
        flat_line = LineString([(0, 0), (1, 0)])
        with pytest.raises(ValueError, match="zero area"):
            _random_xy_in_poly(flat_line, 5, rng=rng)

    def test_generates_exact_count_inside_polygon(self):
        rng = np.random.default_rng(42)
        poly = box(0, 0, 10, 10)
        xs, ys = _random_xy_in_poly(poly, 100, rng=rng)
        assert len(xs) == len(ys) == 100
        assert contains_xy(poly, xs, ys).all()

    def test_low_inclusion_probability_still_generates_exact_count(self):
        rng = np.random.default_rng(42)
        poly = Polygon([(0, 0), (1, 1), (1, 1 + 5e-5), (0, 5e-5)])
        xs, ys = _random_xy_in_poly(poly, 1, rng=rng)
        assert len(xs) == len(ys) == 1
        assert contains_xy(poly, xs, ys).all()

    def test_impractically_thin_polygon_raises_instead_of_sampling_forever(self):
        rng = np.random.default_rng(42)
        poly = Polygon([(0, 0), (1, 1), (1, 1 + 1e-9), (0, 1e-9)])

        with pytest.raises(ValueError, match="rejection sampling"):
            _random_xy_in_poly(poly, 1, rng=rng)


@pytest.mark.parametrize("geometry", [None, Polygon(), LineString([(0, 0), (1, 1)])])
def test_make_random_points_rejects_invalid_geometry_before_parallel_work(geometry):
    gdf = GeoDataFrame({"value": [10]}, geometry=[geometry])

    with pytest.raises(ValueError, match="zero area"):
        _make_random_points(gdf, 10, "value", np.random.default_rng(42), n_jobs=1)


def test_zero_population_degenerate_rows_are_tolerated():
    """Degenerate rows whose value rounds to zero dots pass; real census extracts carry
    zero-population rows with empty or missing geometry."""
    # Missing geometry (None) is valid at runtime, but the geopandas stubs only admit
    # Geometry elements; an object-dtype array carries the mixed column past both checkers.
    geometry_with_missing = np.array([box(0, 0, 1, 1), Polygon(), None], dtype=object)
    gdf = GeoDataFrame(
        {"value": [100.0, 0.0, 0.0]},
        geometry=GeoSeries(geometry_with_missing),
    )
    xs, ys, polyids = _make_random_points(gdf, 10, "value", np.random.default_rng(42), n_jobs=1)
    assert len(xs) == len(ys) == 10
    assert set(polyids.tolist()) == {0}


def test_make_random_points_rejects_empty_geodataframe():
    gdf = GeoDataFrame({"value": []}, geometry=[])

    with pytest.raises(ValueError, match="empty"):
        _make_random_points(gdf, 10, "value", np.random.default_rng(42), n_jobs=1)


# ======================
# == DOT COUNT ROUNDING ==
# ======================


class TestDotCountRounding:
    def test_fractional_counts_conserve_expected_dots(self):
        """100 polygons at 0.49 dots apiece must yield roughly 49 dots, never zero."""
        polygons = [box(i, 0, i + 1, 1) for i in range(100)]
        gdf = GeoDataFrame({"value": [49.0] * 100}, geometry=polygons)
        xs, ys, _ = _make_random_points(gdf, 100, "value", np.random.default_rng(42), n_jobs=1)
        assert len(xs) == len(ys)
        # E[total] = 49; Binomial(100, 0.49) has sigma = 5, so 24..74 is a 5-sigma band.
        assert 0 < len(xs)
        assert 24 <= len(xs) <= 74

    def test_exact_multiples_give_exact_counts(self):
        from collections import Counter

        polygons = [box(i, 0, i + 1, 1) for i in range(5)]
        gdf = GeoDataFrame(
            {"value": [300.0, 100.0, 200.0, 0.0, 100.0]},
            geometry=polygons,
        )
        xs, _, polyids = _make_random_points(gdf, 100, "value", np.random.default_rng(0), n_jobs=1)
        assert len(xs) == 7
        assert Counter(polyids.tolist()) == {0: 3, 1: 1, 2: 2, 4: 1}


# ===============
# == ZERO DOTS ==
# ===============


class TestDotDensityZeroDots:
    def test_zero_population_rows_skipped(self, testing_gdf, tmp_path):
        """Rows with population = 0 produce no dots (n_dots <= 0 path)."""
        gdf = testing_gdf.copy()
        gdf["zero_pop"] = 0
        gdf.loc[gdf.index[0], "zero_pop"] = 100  # one row with dots
        plot = DotDensityPlot(
            gdf,
            outline_column="district",
            dpi=50,
            silent=True,
            people_per_dot=50,
            show_labels=False,
        )
        plot.add_density_layer(column="zero_pop", color="red")
        plot.save(str(tmp_path / "zero_pop.png"))
        assert (tmp_path / "zero_pop.png").exists()

    def test_all_zero_population_empty_result(self, testing_gdf, tmp_path):
        """Zero population produces no points for every geometry."""
        gdf = testing_gdf.copy()
        gdf["small_pop"] = 0
        plot = DotDensityPlot(
            gdf,
            outline_column="district",
            dpi=50,
            silent=True,
            people_per_dot=10000,
            show_labels=False,
        )
        plot.add_density_layer(column="small_pop", color="blue")
        plot.save(str(tmp_path / "all_zero_dots.png"))
        assert (tmp_path / "all_zero_dots.png").exists()

        cached = np.load(plot._density_layers["small_pop"].cache_path)
        assert len(cached["x"]) == 0 and len(cached["y"]) == 0


# ======================
# == EMPTY DOT LAYERS ==
# ======================


class TestDotDensityEmptyDict:
    def test_draw_all_dots_with_no_layers(self, testing_gdf, tmp_path):
        """Building a DotDensityPlot with no dot layers still works."""
        plot = DotDensityPlot(
            testing_gdf,
            outline_column="district",
            dpi=50,
            silent=True,
            people_per_dot=500,
            show_labels=False,
        )
        # Don't add any dot density layers; _draw_all_dots should return early
        plot.save(str(tmp_path / "no_dots.png"))
        assert (tmp_path / "no_dots.png").exists()


# ====================
# == LEGEND OPTIONS ==
# ====================


class TestDotDensitySetLegendOptions:
    def test_set_legend_options(self, testing_gdf, tmp_path):
        """set_legend_options updates legend and renders correctly."""
        plot = DotDensityPlot(
            testing_gdf,
            outline_column="district",
            dpi=50,
            silent=True,
            people_per_dot=500,
            show_legend=True,
            show_labels=False,
        )
        plot.set_legend_options(
            loc="upper right",
            ncols=2,
            fontsize=8,
            title="Legend",
            frameon=True,
        )
        plot.add_density_layer(column="tot_pop", color="red")
        plot.save(str(tmp_path / "legend_opts.png"))
        assert (tmp_path / "legend_opts.png").exists()
