import logging
import warnings
from pathlib import Path
from typing import cast

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest
from geopandas import GeoDataFrame
from matplotlib.collections import PathCollection
from matplotlib.colors import to_rgba
from shapely import contains_xy
from shapely.geometry import box

from gerrytools.plotting.geometry.dotdensity import DotDensityPlot


# ========================
# == CRS / REPROJECTION ==
# ========================
def test_dots_and_outlines_share_the_target_crs():
    gdf = GeoDataFrame(
        {"district": [1], "population": [20]},
        geometry=[box(-90.0, 40.0, -89.99, 40.01)],
        crs="EPSG:4326",
    )
    plot = DotDensityPlot(
        gdf,
        outline_column="district",
        target_crs="EPSG:3857",
        people_per_dot=10,
        show_labels=False,
        silent=True,
        rng_seed=42,
    )
    plot.add_density_layer("population", "red", n_jobs=1)

    ax = plot.ax
    minx, miny, maxx, maxy = gdf.to_crs("EPSG:3857").total_bounds
    offsets = np.asarray(ax.collections[-1].get_offsets(), dtype=float)

    assert np.all((offsets[:, 0] >= minx) & (offsets[:, 0] <= maxx))
    assert np.all((offsets[:, 1] >= miny) & (offsets[:, 1] <= maxy))


def test_target_crs_reassignment_resamples_dots():
    """Reassigning target_crs invalidates cached dots so they land in the new CRS."""
    gdf = GeoDataFrame(
        {"district": [1], "population": [30]},
        geometry=[box(-90.0, 40.0, -89.9, 40.1)],
        crs="EPSG:4326",
    )
    plot = DotDensityPlot(
        gdf,
        outline_column="district",
        target_crs="EPSG:4326",
        people_per_dot=10,
        show_labels=False,
        silent=True,
        rng_seed=3,
    )
    plot.add_density_layer("population", "red", n_jobs=1)
    plot.focus_axes()
    plot.ax  # build in the original CRS

    with pytest.warns(UserWarning, match="previous CRS"):
        plot.target_crs = "EPSG:3857"
    offsets = np.asarray(plot.ax.collections[-1].get_offsets(), dtype=float)
    minx, miny, maxx, maxy = gdf.to_crs("EPSG:3857").total_bounds
    assert len(offsets) == 3
    assert np.all((offsets[:, 0] >= minx) & (offsets[:, 0] <= maxx))
    assert np.all((offsets[:, 1] >= miny) & (offsets[:, 1] <= maxy))


def test_mutating_density_values_invalidates_cached_dots():
    gdf = GeoDataFrame(
        {"district": [1], "population": [100]},
        geometry=[box(0, 0, 1, 1)],
        crs="EPSG:3857",
    )
    plot = DotDensityPlot(
        gdf,
        outline_column="district",
        people_per_dot=10,
        show_labels=False,
        silent=True,
        rng_seed=3,
    )
    plot.add_density_layer("population", "red", n_jobs=1)
    before = len(np.asarray(plot.ax.collections[-1].get_offsets()))

    plot.gdf["population"] *= 2
    plot.show_axis = True
    after = len(np.asarray(plot.ax.collections[-1].get_offsets()))

    assert after == 2 * before


class TestDotDensityRngContract:
    """Layer sampling and interleaving draw from independent derived generators."""

    @staticmethod
    def _dot_plot(testing_gdf, seed: int | None = 7):
        return DotDensityPlot(
            testing_gdf,
            outline_column="district",
            dpi=50,
            silent=True,
            people_per_dot=500,
            rng_seed=seed,
            show_labels=False,
        )

    @staticmethod
    def _cached_xy(plot, column):
        with np.load(plot._density_layers[column].cache_path) as cached:
            return cached["x"].copy(), cached["y"].copy()

    def test_render_between_layer_adds_does_not_shift_dots(self, testing_gdf):
        interleaved = self._dot_plot(testing_gdf)
        interleaved.add_density_layer("maj_pop", "red", n_jobs=1)
        interleaved.ax  # render between the two adds
        interleaved.add_density_layer("min_pop", "blue", n_jobs=1)

        back_to_back = self._dot_plot(testing_gdf)
        back_to_back.add_density_layer("maj_pop", "red", n_jobs=1)
        back_to_back.add_density_layer("min_pop", "blue", n_jobs=1)

        for column in ("maj_pop", "min_pop"):
            for got, expected in zip(
                self._cached_xy(interleaved, column), self._cached_xy(back_to_back, column)
            ):
                np.testing.assert_array_equal(got, expected)

    def test_chunk_count_does_not_change_dots(self, testing_gdf):
        few_chunks = self._dot_plot(testing_gdf)
        few_chunks.add_density_layer("tot_pop", "red", n_jobs=1, n_chunks=2)
        many_chunks = self._dot_plot(testing_gdf)
        many_chunks.add_density_layer("tot_pop", "red", n_jobs=1, n_chunks=10)

        for got, expected in zip(
            self._cached_xy(few_chunks, "tot_pop"), self._cached_xy(many_chunks, "tot_pop")
        ):
            np.testing.assert_array_equal(got, expected)

    def test_rebuild_does_not_change_dot_draw_data(self, testing_gdf):
        plot = self._dot_plot(testing_gdf)
        plot.add_density_layer("maj_pop", "red", n_jobs=1)
        plot.add_density_layer("min_pop", "blue", n_jobs=1)
        first_offsets = np.asarray(plot.ax.collections[-1].get_offsets(), dtype=float).copy()
        plot.show_axis = True
        second_offsets = np.asarray(plot.ax.collections[-1].get_offsets(), dtype=float)
        np.testing.assert_array_equal(first_offsets, second_offsets)

    def test_interleaving_keeps_each_dot_attached_to_its_layer_color(self, testing_gdf):
        plot = self._dot_plot(testing_gdf)
        plot.add_density_layer("maj_pop", "red", n_jobs=1)
        plot.add_density_layer("min_pop", "blue", n_jobs=1)

        expected = {}
        for column, color in (("maj_pop", "red"), ("min_pop", "blue")):
            x, y = self._cached_xy(plot, column)
            expected.update(
                {(float(px), float(py)): to_rgba(color) for px, py in zip(x, y, strict=True)}
            )

        collection = cast(PathCollection, plot.ax.collections[-1])
        offsets = np.asarray(collection.get_offsets(), dtype=float)
        colors = np.asarray(collection.get_facecolor(), dtype=float)
        assert len(offsets) == len(colors) == len(expected)
        for point, color in zip(offsets, colors, strict=True):
            np.testing.assert_allclose(color, expected[(float(point[0]), float(point[1]))])

    def test_setting_rng_seed_invalidates_and_changes_dots(self, testing_gdf):
        plot = self._dot_plot(testing_gdf, seed=1)
        plot.add_density_layer("tot_pop", "red", n_jobs=1)
        before = np.asarray(plot.ax.collections[-1].get_offsets(), dtype=float).copy()
        plot.rng_seed = 2
        after = np.asarray(plot.ax.collections[-1].get_offsets(), dtype=float)
        assert before.shape != after.shape or not np.array_equal(before, after)

    def test_seedless_property_round_trip_keeps_dots(self, testing_gdf):
        plot = self._dot_plot(testing_gdf, seed=None)
        plot.add_density_layer("tot_pop", "red", n_jobs=1)
        before = self._cached_xy(plot, "tot_pop")
        seed_root = plot._seed_root

        plot.rng_seed = plot.rng_seed

        assert plot._seed_root == seed_root
        after = self._cached_xy(plot, "tot_pop")
        for actual, expected in zip(after, before, strict=True):
            np.testing.assert_array_equal(actual, expected)


class TestDotDensityRngSeedSetter:
    def test_set_rng_seed(self, testing_gdf):
        """Setting rng_seed property should update internal RNG."""
        plot = DotDensityPlot(
            testing_gdf, outline_column="district", dpi=50, silent=True, people_per_dot=500
        )
        plot.rng_seed = 99
        assert plot.rng_seed == 99

    def test_same_seed_reproduces_identical_dots(self, testing_gdf):
        def generated_dots(seed):
            plot = DotDensityPlot(
                testing_gdf,
                outline_column="district",
                dpi=50,
                silent=True,
                people_per_dot=500,
                rng_seed=seed,
                show_labels=False,
            )
            plot.add_density_layer(column="tot_pop", color="red", n_jobs=1)
            with np.load(plot._density_layers["tot_pop"].cache_path) as cached:
                return cached["x"].copy(), cached["y"].copy()

        first_x, first_y = generated_dots(7)
        second_x, second_y = generated_dots(7)
        assert len(first_x) > 0
        np.testing.assert_array_equal(first_x, second_x)
        np.testing.assert_array_equal(first_y, second_y)


class TestDotDensityMarkerOptions:
    def test_set_marker_options_and_build(self, testing_gdf, tmp_path):
        """set_marker_options followed by a build should not raise."""
        plot = DotDensityPlot(
            testing_gdf,
            outline_column="district",
            dpi=50,
            silent=True,
            people_per_dot=500,
            show_labels=False,
        )
        plot.set_marker_options(marker="^", markersize=2.0)
        plot.add_density_layer(column="tot_pop", color="blue")
        plot.save(str(tmp_path / "dot_marker.png"))
        assert (tmp_path / "dot_marker.png").exists()

    def test_explicit_none_marker_edge_removes_edge(self, testing_gdf):
        plot = DotDensityPlot(
            testing_gdf,
            outline_column="district",
            silent=True,
            people_per_dot=500,
            show_labels=False,
        )
        plot.set_marker_options(markeredgecolor="red")
        plot.set_marker_options(markeredgecolor=None)

        assert plot._marker_options.markeredgecolor == "none"
        assert plot._marker_options.markeredgealpha == 0.0
        assert plot._marker_options.markeredgewidth == 0.0

    def test_visible_marker_edge_restores_default_width_after_none(self, testing_gdf):
        plot = DotDensityPlot(
            testing_gdf,
            outline_column="district",
            silent=True,
            people_per_dot=500,
            show_labels=False,
        )
        plot.set_marker_options(markeredgecolor=None)
        plot.set_marker_options(markeredgecolor="red")

        assert plot._marker_options.markeredgecolor == "#ff0000"
        assert plot._marker_options.markeredgewidth == 0.8

    def test_explicit_zero_marker_edgewidth_stays_hidden(self, testing_gdf):
        plot = DotDensityPlot(
            testing_gdf,
            outline_column="district",
            silent=True,
            people_per_dot=500,
            show_labels=False,
        )
        plot.set_marker_options(markeredgecolor="red", markeredgewidth=0.0)

        assert plot._marker_options.markeredgewidth == 0.0


class TestDotDensityValidationErrors:
    def test_missing_column_raises(self, testing_gdf):
        """Referencing a non-existent column should raise ValueError."""
        plot = DotDensityPlot(
            testing_gdf, outline_column="district", dpi=50, silent=True, people_per_dot=500
        )
        with pytest.raises(ValueError, match="not found in GeoDataFrame"):
            plot.add_density_layer(column="nonexistent_column", color="red")

    def test_negative_values_raises(self, testing_gdf):
        """A column with negative values should raise ValueError."""
        gdf = testing_gdf.copy()
        gdf["neg_col"] = -1
        plot = DotDensityPlot(
            gdf, outline_column="district", dpi=50, silent=True, people_per_dot=500
        )
        with pytest.raises(ValueError, match="negative values"):
            plot.add_density_layer(column="neg_col", color="red")

    def test_nan_values_raises(self, testing_gdf):
        """A column with NaN values should raise ValueError."""
        gdf = testing_gdf.copy()
        gdf["nan_col"] = np.nan
        plot = DotDensityPlot(
            gdf, outline_column="district", dpi=50, silent=True, people_per_dot=500
        )
        with pytest.raises(ValueError, match="NaN values"):
            plot.add_density_layer(column="nan_col", color="red")

    def test_infinite_values_raises(self, testing_gdf):
        """An infinite population would ask for infinitely many dots."""
        gdf = testing_gdf.copy()
        gdf["inf_col"] = np.inf
        plot = DotDensityPlot(
            gdf, outline_column="district", dpi=50, silent=True, people_per_dot=500
        )
        with pytest.raises(ValueError, match="infinite values"):
            plot.add_density_layer(column="inf_col", color="red")

    def test_nonnumeric_column_raises(self, testing_gdf):
        """A string column cannot be converted to dot counts."""
        gdf = testing_gdf.copy()
        gdf["str_col"] = "a lot"
        plot = DotDensityPlot(
            gdf, outline_column="district", dpi=50, silent=True, people_per_dot=500
        )
        with pytest.raises(ValueError, match="numeric"):
            plot.add_density_layer(column="str_col", color="red")

    @pytest.mark.parametrize("people_per_dot", [0, -1, float("inf"), float("nan"), "10"])
    def test_invalid_people_per_dot_rejected_at_construction(self, testing_gdf, people_per_dot):
        # Regression: people_per_dot=0 used to overflow inside dot generation and
        # people_per_dot=-1 silently produced a plot with no dots.
        with pytest.raises(ValueError, match="people_per_dot"):
            DotDensityPlot(
                testing_gdf,
                outline_column="district",
                dpi=50,
                silent=True,
                people_per_dot=people_per_dot,
            )

    @pytest.mark.parametrize("bad_value", [0, -5, float("nan"), "10"])
    def test_people_per_dot_setter_validates(self, testing_gdf, bad_value):
        plot = DotDensityPlot(
            testing_gdf, outline_column="district", dpi=50, silent=True, people_per_dot=500
        )
        with pytest.raises(ValueError, match="people_per_dot"):
            plot.people_per_dot = bad_value
        assert plot.people_per_dot == 500

    @pytest.mark.parametrize(
        ("field", "value"),
        [("n_chunks", 0), ("n_chunks", -1), ("n_jobs", 0), ("n_jobs", -2)],
    )
    def test_invalid_parallelization_arguments_rejected(self, testing_gdf, field, value):
        # Regression: n_chunks=0 used to raise ZeroDivisionError and n_jobs=0 silently
        # meant "all available cores".
        plot = DotDensityPlot(
            testing_gdf, outline_column="district", dpi=50, silent=True, people_per_dot=500
        )
        with pytest.raises(ValueError, match=field):
            plot.add_density_layer(column="tot_pop", color="red", **{field: value})

    def test_numpy_chunk_count_is_accepted(self, testing_gdf):
        plot = DotDensityPlot(
            testing_gdf, outline_column="district", dpi=50, silent=True, people_per_dot=500
        )

        plot.add_density_layer("tot_pop", "red", n_jobs=1, n_chunks=np.int64(2))

        assert "tot_pop" in plot._density_layers

    def test_boolean_chunk_count_is_rejected(self, testing_gdf):
        plot = DotDensityPlot(
            testing_gdf, outline_column="district", dpi=50, silent=True, people_per_dot=500
        )

        with pytest.raises(ValueError, match="n_chunks"):
            plot.add_density_layer("tot_pop", "red", n_jobs=1, n_chunks=True)

    @pytest.mark.parametrize(
        ("color", "expected"),
        [("#ff000080", "#ff000080"), ("none", "none")],
    )
    def test_density_layer_preserves_alpha_and_none(self, testing_gdf, color, expected):
        plot = DotDensityPlot(
            testing_gdf, outline_column="district", dpi=50, silent=True, people_per_dot=500
        )

        plot.add_density_layer("tot_pop", color, n_jobs=1)

        assert plot._density_layers["tot_pop"].color == expected

    def test_same_column_same_color_warns(self, testing_gdf):
        """Adding the same column+color twice should emit a UserWarning."""
        plot = DotDensityPlot(
            testing_gdf,
            outline_column="district",
            dpi=50,
            silent=True,
            people_per_dot=500,
        )
        plot.add_density_layer(column="tot_pop", color="red")
        with pytest.warns(UserWarning, match="already exist"):
            plot.add_density_layer(column="tot_pop", color="red")

    def test_same_column_different_color_recolors_without_warning(self, testing_gdf):
        """Re-adding a column with a new color recolors the cached dots.

        Regression: the different-color branch used to warn "Overwriting" and return before
        updating the color dict, so nothing was overwritten.
        """
        plot = DotDensityPlot(
            testing_gdf,
            outline_column="district",
            dpi=50,
            silent=True,
            people_per_dot=500,
        )
        plot.add_density_layer(column="tot_pop", color="red")
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            plot.add_density_layer(column="tot_pop", color="blue")
        assert plot._density_layers["tot_pop"].color == "#0000ff"


class TestDotDensityLegend:
    def test_build_with_show_legend(self, testing_gdf, tmp_path):
        """show_legend=True should render legend without error."""
        plot = DotDensityPlot(
            testing_gdf,
            outline_column="district",
            dpi=50,
            silent=True,
            people_per_dot=500,
            show_legend=True,
            show_labels=False,
        )
        plot.add_density_layer(column="maj_pop", color="#1b7837")
        plot.save(str(tmp_path / "dot_legend.png"))
        assert (tmp_path / "dot_legend.png").exists()


class TestDotDensityExternalLegend:
    def test_external_legend_survives_rebuild(self, testing_gdf):
        # Regression: the legend redraw used to delete the axes legend unconditionally,
        # destroying a legend the user installed between rebuilds.
        from matplotlib.lines import Line2D

        plot = DotDensityPlot(
            testing_gdf,
            outline_column="district",
            dpi=50,
            silent=True,
            people_per_dot=500,
            show_legend=True,
            show_labels=False,
        )
        plot.add_density_layer(column="maj_pop", color="#1b7837")
        ax = plot.ax
        assert ax.get_legend() is not None
        external_legend = ax.legend(handles=[Line2D([0], [0], label="external")])
        plot.ax  # rebuild
        assert plot._ax.get_legend() is external_legend

    def test_external_legend_survives_rebuild_when_legend_disabled(self, testing_gdf):
        from matplotlib.lines import Line2D

        plot = DotDensityPlot(
            testing_gdf,
            outline_column="district",
            dpi=50,
            silent=True,
            people_per_dot=500,
            show_labels=False,
        )
        plot.add_density_layer(column="maj_pop", color="#1b7837")
        ax = plot.ax
        external_legend = ax.legend(handles=[Line2D([0], [0], label="external")])
        plot.ax  # rebuild
        assert plot._ax.get_legend() is external_legend

    def test_show_legend_setter_reclaims_over_external_legend(self, testing_gdf):
        from matplotlib.lines import Line2D

        plot = DotDensityPlot(
            testing_gdf,
            outline_column="district",
            dpi=50,
            silent=True,
            people_per_dot=500,
            show_legend=True,
            show_labels=False,
        )
        plot.add_density_layer(column="maj_pop", color="#1b7837")
        ax = plot.ax
        external_legend = ax.legend(handles=[Line2D([0], [0], label="external")])
        # Most-recent-wins: re-asserting show_legend reclaims the legend unit.
        plot.show_legend = True
        plot.ax
        new_legend = plot._ax.get_legend()
        assert new_legend is not None
        assert new_legend is not external_legend


class TestDotDensitySilentFalse:
    def test_build_silent_false(self, testing_gdf, tmp_path, capsys):
        """Building DotDensityPlot with silent=False should print progress."""
        plot = DotDensityPlot(
            testing_gdf,
            outline_column="district",
            dpi=50,
            silent=False,
            people_per_dot=500,
            show_labels=False,
        )
        plot.add_density_layer(column="tot_pop", color="red")
        plot.save(str(tmp_path / "dot_silent_false.png"))
        captured = capsys.readouterr()
        # Either "Generating dots" or "Rendering ... dots" should appear
        assert "dots" in captured.out.lower() or "rendering" in captured.out.lower()


class TestDotDensitySaveLegend:
    def test_save_legend_creates_file(self, testing_gdf, tmp_path):
        """save_legend should write a file when data has been added."""
        plot = DotDensityPlot(
            testing_gdf,
            outline_column="district",
            dpi=50,
            silent=True,
            people_per_dot=500,
            show_labels=False,
        )
        plot.add_density_layer(column="maj_pop", color="green")
        legend_path = str(tmp_path / "legend.png")
        plot.save_legend(legend_path)
        assert Path(legend_path).exists()

    def test_save_legend_with_display_names(self, testing_gdf, tmp_path):
        """save_legend with display_names should rename labels."""
        plot = DotDensityPlot(
            testing_gdf,
            outline_column="district",
            dpi=50,
            silent=True,
            people_per_dot=500,
            show_labels=False,
        )
        plot.add_density_layer(column="maj_pop", color="purple")
        legend_path = str(tmp_path / "legend_named.png")
        plot.save_legend(legend_path, display_names={"maj_pop": "Majority Population"})
        assert Path(legend_path).exists()


class TestDotDensitySaveLegendEmpty:
    def test_save_legend_empty_logs_warning(self, testing_gdf, tmp_path, caplog):
        """save_legend on a plot with no data warns through the gerrytools logger."""
        plot = DotDensityPlot(
            testing_gdf,
            outline_column="district",
            dpi=50,
            silent=True,
            people_per_dot=500,
            show_labels=False,
        )
        legend_path = str(tmp_path / "empty_legend.png")
        with caplog.at_level(logging.WARNING, logger="gerrytools"):
            plot.save_legend(legend_path)
        assert any("no legend to save" in message.lower() for message in caplog.messages)
        assert not Path(legend_path).exists()


def test_string_index_renamed_geometry_counts_and_containment():
    polygons = [box(0, 0, 1, 1), box(1, 0, 2, 1)]
    gdf = GeoDataFrame(
        {"district": [1, 2], "population": [20, 10]},
        geometry=polygons,
    )
    gdf.index = ["left", "right"]
    gdf.rename_geometry("shape", inplace=True)
    plot = DotDensityPlot(
        gdf,
        outline_column="district",
        people_per_dot=10,
        show_labels=False,
        silent=True,
        rng_seed=42,
    )
    plot.add_density_layer("population", "blue", n_jobs=1, n_chunks=2)

    cache_path = plot._density_layers["population"].cache_path
    with np.load(cache_path) as points:
        xs = points["x"]
        ys = points["y"]
        polyids = points["polyids"]

    assert len(xs) == 3
    assert polyids.tolist().count(0) == 2
    assert polyids.tolist().count(1) == 1
    for polyid, polygon in enumerate(polygons):
        mask = polyids == polyid
        assert contains_xy(polygon, xs[mask], ys[mask]).all()


def test_string_index_with_an_empty_chunk_uses_pickle_free_cache_ids():
    gdf = GeoDataFrame(
        {"district": [1, 1, 1], "population": [0, 0, 10]},
        geometry=[box(i, 0, i + 1, 1) for i in range(3)],
    )
    gdf.index = ["zero-a", "zero-b", "populated"]
    plot = DotDensityPlot(
        gdf,
        outline_column="district",
        people_per_dot=10,
        show_labels=False,
        silent=True,
        rng_seed=42,
    )
    plot.add_density_layer("population", "blue", n_jobs=1, n_chunks=2)

    with np.load(plot._density_layers["population"].cache_path) as points:
        assert points["polyids"].tolist() == [2]


def test_density_column_path_separator_does_not_enter_cache_path():
    column = "population/total"
    gdf = GeoDataFrame(
        {"district": [1], column: [10]},
        geometry=[box(0, 0, 1, 1)],
    )
    plot = DotDensityPlot(
        gdf,
        outline_column="district",
        people_per_dot=10,
        show_labels=False,
        silent=True,
        rng_seed=42,
    )

    plot.add_density_layer(column, "blue", n_jobs=1)

    cache_path = plot._density_layers[column].cache_path
    assert cache_path.parent == Path(plot._temp_dir_name)
    assert cache_path.is_file()


class TestDotDensityLegendMarkerSize:
    def test_legend_glyphs_floor_at_readable_size(self, testing_gdf):
        # Regression: sub-point map dots produced invisible legend glyphs.
        plot = DotDensityPlot(
            testing_gdf,
            outline_column="district",
            dpi=50,
            silent=True,
            people_per_dot=500,
            show_legend=True,
            show_labels=False,
        )
        plot.set_marker_options(markersize=0.7, markeredgecolor="none")
        plot.add_density_layer(column="maj_pop", color="#1b7837")
        from matplotlib.lines import Line2D

        legend = plot.ax.get_legend()
        assert legend is not None
        legend_marker_sizes = [
            handle.get_markersize()
            for handle in legend.legend_handles
            if isinstance(handle, Line2D)
        ]
        assert legend_marker_sizes
        assert all(size >= 6.0 for size in legend_marker_sizes)

    def test_saved_legend_glyphs_use_the_same_floor(self, testing_gdf, monkeypatch, tmp_path):
        plot = DotDensityPlot(
            testing_gdf,
            outline_column="district",
            silent=True,
            people_per_dot=500,
            show_labels=False,
        )
        plot.set_marker_options(markersize=0.7)
        plot.add_density_layer("maj_pop", "#1b7837")
        captured = []
        monkeypatch.setattr(
            plot,
            "_save_legend_handles",
            lambda handles, *_args, **_kwargs: captured.extend(handles),
        )

        plot.save_legend(str(tmp_path / "legend.png"))

        assert [handle.get_markersize() for handle in captured] == [6.0]


class TestDotDensityTempDirLifecycle:
    def test_plot_and_temp_dir_are_released_on_deletion(self, testing_gdf):
        # Regression: atexit.register(self._close) held a bound-method reference that pinned
        # every DotDensityPlot (and its dot cache directory) for the life of the process.
        import gc
        import weakref

        plot = DotDensityPlot(
            testing_gdf, outline_column="district", dpi=50, silent=True, people_per_dot=500
        )
        plot_ref = weakref.ref(plot)
        temp_dir = Path(plot._temp_dir_name)
        assert temp_dir.exists()

        del plot
        gc.collect()

        assert plot_ref() is None
        assert not temp_dir.exists()
