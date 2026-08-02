from collections.abc import Hashable
from pathlib import Path
from typing import cast

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from gerrytools.plotting.geometry.geoplot import GeoPlot


# ========================
# == CRS / REPROJECTION ==
# ========================
class TestCategoricalColorLayerColorSeries:
    """Cover __map_unique_values_to_colors and color_series branches."""

    def test_districtr_colormap_default(self, testing_gdf, tmp_path):
        """Default 'districtr' colormap should build without error."""
        plot = GeoPlot(testing_gdf, dpi=50, silent=True)
        plot.add_districting_plan_layer(plan_column="district")
        plot.save(str(tmp_path / "districtr.png"))
        assert (tmp_path / "districtr.png").exists()

    def test_none_colormap(self, testing_gdf, tmp_path):
        """colormap=None should produce 'none' fill for all geometries."""
        plot = GeoPlot(testing_gdf, dpi=50, silent=True)
        plot.add_districting_plan_layer(plan_column="district", colormap=None)
        plot.save(str(tmp_path / "none_cmap.png"))
        assert (tmp_path / "none_cmap.png").exists()

    def test_dict_colormap(self, testing_gdf, tmp_path):
        """A dict colormap should map district values to specific colors."""
        districts: list[int] = sorted(map(int, testing_gdf["district"].unique()))
        colors: dict[Hashable, str] = {d: "#FF0000" for d in districts}
        plot = GeoPlot(testing_gdf, dpi=50, silent=True)
        plot.add_districting_plan_layer(plan_column="district", colormap=colors)
        plot.save(str(tmp_path / "dict_cmap.png"))
        assert (tmp_path / "dict_cmap.png").exists()

    def test_too_many_unique_values_raises(self, testing_gdf):
        """More unique values than provided colors should raise ValueError."""
        from gerrytools.plotting.geometry._layers import _CategoricalColorLayer

        unique_vals = pd.Index(list(range(10)))
        # Only 2 colors provided for 10 unique values
        with pytest.raises(ValueError, match="Not enough colors"):
            _CategoricalColorLayer._map_unique_values_to_colors(unique_vals, ["#FF0000", "#00FF00"])


# ===========================
# == GeoPlotBase XY LIMITS ==
# ===========================


class TestCategoricalColormapColorAmbiguity:
    """Strings that name both a colormap and a color warn when the colormap wins."""

    def test_ambiguous_string_with_column_warns(self, testing_gdf):
        plot = GeoPlot(testing_gdf, dpi=50, silent=True)
        with pytest.warns(UserWarning, match="both a registered Matplotlib colormap"):
            plot.add_districting_plan_layer("district", colormap="pink")

    def test_colormap_only_string_does_not_warn(self, testing_gdf):
        import warnings

        plot = GeoPlot(testing_gdf, dpi=50, silent=True)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            plot.add_districting_plan_layer("district", colormap="viridis")

    def test_color_only_string_without_column_does_not_warn(self, testing_gdf):
        import warnings

        plot = GeoPlot(testing_gdf, dpi=50, silent=True)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            plot.add_highlight_layer(facecolor="orange")

    def test_districtr_default_does_not_warn(self, testing_gdf):
        import warnings

        plot = GeoPlot(testing_gdf, dpi=50, silent=True)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            plot.add_districting_plan_layer("district", colormap="districtr")


def test_external_aspect_survives_geoplot_rebuild(testing_gdf):
    plot = GeoPlot(testing_gdf, silent=True)
    ax = plot.ax
    ax.set_aspect("auto")

    plot.title = "rebuild"

    assert plot.ax.get_aspect() == "auto"


class TestGeoPlotPositionalArgs:
    def test_choropleth_datacolumn_is_first_positional(self, testing_gdf):
        plot = GeoPlot(testing_gdf, dpi=50, silent=True)
        layer = plot.add_choropleth_layer("tot_pop")
        assert layer.column == "tot_pop"

    def test_choropleth_geo_source_is_keyword_only(self, testing_gdf, tmp_path):
        plot = GeoPlot(testing_gdf, dpi=50, silent=True)
        plot.add_choropleth_layer("tot_pop", geo_source=testing_gdf)
        plot.save(str(tmp_path / "choro_pos.png"))
        assert (tmp_path / "choro_pos.png").exists()

    def test_districting_plancolumn_is_first_positional(self, testing_gdf, tmp_path):
        plot = GeoPlot(testing_gdf, dpi=50, silent=True)
        plot.add_districting_plan_layer("district")
        plot.save(str(tmp_path / "plan_pos.png"))
        assert (tmp_path / "plan_pos.png").exists()

    def test_highlight_label_column_positional_and_geo_source_keyword(self, testing_gdf, tmp_path):
        plot = GeoPlot(testing_gdf, dpi=50, silent=True)
        plot.add_highlight_layer("district", geo_source=testing_gdf.iloc[:1])
        plot.save(str(tmp_path / "hl_pos.png"))
        assert (tmp_path / "hl_pos.png").exists()


class TestGeoPlotContinuousLayer:
    def test_choropleth_with_facealpha(self, testing_gdf, tmp_path):
        """facealpha triggers _with_alpha path in _mappable."""
        plot = GeoPlot(testing_gdf, dpi=50, silent=True)
        plot.add_choropleth_layer(column="tot_pop", facealpha=0.5)
        plot.save(str(tmp_path / "choropleth_alpha.png"))
        assert (tmp_path / "choropleth_alpha.png").exists()

    def test_choropleth_with_integer_bins(self, testing_gdf, tmp_path):
        """Integer bins triggers _bin_boundaries integer path."""
        plot = GeoPlot(testing_gdf, dpi=50, silent=True)
        plot.add_choropleth_layer(column="tot_pop", bins=5)
        plot.save(str(tmp_path / "choropleth_bins_int.png"))
        assert (tmp_path / "choropleth_bins_int.png").exists()

    def test_choropleth_with_list_bins(self, testing_gdf, tmp_path):
        """List bins triggers IntervalIndex.from_breaks path."""
        plot = GeoPlot(testing_gdf, dpi=50, silent=True)
        plot.add_choropleth_layer(column="tot_pop", bins=[0, 1000, 3000, 5000, 7000, 10000])
        plot.save(str(tmp_path / "choropleth_bins_list.png"))
        assert (tmp_path / "choropleth_bins_list.png").exists()

    def test_choropleth_with_all_nan_column(self, testing_gdf, tmp_path):
        """All-NaN column hits _effective_bounds empty path."""
        gdf = testing_gdf.copy()
        gdf["nan_col"] = np.nan
        plot = GeoPlot(gdf, dpi=50, silent=True)
        plot.add_choropleth_layer(column="nan_col")
        plot.save(str(tmp_path / "choropleth_nan.png"))
        assert (tmp_path / "choropleth_nan.png").exists()

    def test_choropleth_with_vmin_vmax(self, testing_gdf, tmp_path):
        """Explicit vmin/vmax overrides _effective_bounds."""
        plot = GeoPlot(testing_gdf, dpi=50, silent=True)
        plot.add_choropleth_layer(column="tot_pop", vmin=1000, vmax=9000)
        plot.save(str(tmp_path / "choropleth_vminvmax.png"))
        assert (tmp_path / "choropleth_vminvmax.png").exists()

    def test_choropleth_geoseries_raises_typeerror(self, testing_gdf):
        """Passing a GeoSeries as geo_source to choropleth should raise TypeError."""
        from gerrytools.plotting.geometry._layers import _ContinuousColorLayer

        with pytest.raises(TypeError, match="geo_source must be a GeoDataFrame"):
            _ContinuousColorLayer(
                geometry_source=testing_gdf.geometry,
                column="tot_pop",
            )

    def test_choropleth_invalid_colormap_raises(self, testing_gdf):
        """Passing an unknown colormap name should raise ValueError."""
        plot = GeoPlot(testing_gdf, dpi=50, silent=True)
        with pytest.raises(ValueError, match="not found in matplotlib colormaps"):
            plot.add_choropleth_layer(column="tot_pop", colormap="this_colormap_does_not_exist")


# =====================================
# == GeoPlot COLORBAR OPTIONS ==
# =====================================


class TestGeoPlotColorbar:
    def test_choropleth_with_colorbar(self, testing_gdf, tmp_path):
        """show_colorbar=True should create colorbar without error."""
        plot = GeoPlot(testing_gdf, dpi=50, silent=True)
        plot.add_choropleth_layer(column="tot_pop", show_colorbar=True)
        plot.save(str(tmp_path / "choropleth_colorbar.png"))
        assert (tmp_path / "choropleth_colorbar.png").exists()

    def test_set_colorbar_layout(self, testing_gdf, tmp_path):
        """set_colorbar_layout modifies options and builds correctly."""
        plot = GeoPlot(testing_gdf, dpi=50, silent=True)
        plot.add_choropleth_layer(column="tot_pop", show_colorbar=True)
        plot.set_colorbar_layout(outer_pad=0.02, inner_pad=0.1)
        plot.save(str(tmp_path / "choropleth_colorbar_layout.png"))
        assert (tmp_path / "choropleth_colorbar_layout.png").exists()


# ==================================
# == GeoPlot WITH MESSAGES ==
# ==================================


class TestGeoPlotSilentFalse:
    def test_build_silent_false_prints(self, testing_gdf, tmp_path, capsys):
        """Building with silent=False triggers print statements."""
        plot = GeoPlot(testing_gdf, dpi=50, silent=False)
        plot.add_choropleth_layer(column="tot_pop")
        plot.save(str(tmp_path / "silent_false.png"))
        captured = capsys.readouterr()
        assert "Rendering" in captured.out


# ================================
# == GeoPlot DISTRICTING ==
# ================================


class TestGeoPlotDistrictingPlan:
    def test_add_districting_plan_layer_dissolve_labels(self, testing_gdf, tmp_path):
        """add_districting_plan_layer with dissolve=True and show_labels=True."""
        plot = GeoPlot(testing_gdf, dpi=50, silent=True)
        plot.add_districting_plan_layer(
            plan_column="district",
            dissolve=True,
            show_labels=True,
        )
        plot.save(str(tmp_path / "plan_dissolve_labels.png"))
        assert (tmp_path / "plan_dissolve_labels.png").exists()


# ====================
# == DotDensityPlot ==
# ====================


class TestStringPlanLabels:
    """coerce_labels except branch with non-numeric string district labels."""

    def test_string_district_labels_hit_coerce_except(self, testing_gdf, tmp_path):
        """Plan column with non-numeric strings triggers the except branch in coerce_labels."""

        from gerrytools.plotting.geometry import GeoPlot

        str_gdf = testing_gdf.copy()
        str_gdf["str_district"] = str_gdf["district"].map(
            lambda d: chr(65 + int(d))  # 0->"A", 1->"B", etc.
        )

        plot = GeoPlot(str_gdf, dpi=50)
        plot.add_districting_plan_layer(plan_column="str_district", show_labels=True)
        out = str(tmp_path / "str_labels.png")
        plot.save(out)
        assert Path(out).exists()


# ===============================
# == DOTDENSITY ZERO-DOT PATHS ==
# ===============================


class TestAddColorbarPostHoc:
    def test_add_colorbar_after_layer(self, testing_gdf, tmp_path):
        plot = GeoPlot(testing_gdf, dpi=50, silent=True)
        layer = plot.add_choropleth_layer("tot_pop")
        plot.add_colorbar(layer, label="Population")
        assert len(plot._colorbar_requests) == 1
        assert plot._colorbar_requests[0].label == "Population"
        plot.save(str(tmp_path / "posthoc_colorbar.png"))
        assert (tmp_path / "posthoc_colorbar.png").exists()


class TestFocusAxesPerSidePad:
    def test_four_tuple_pads_each_side(self, testing_gdf):
        plot = GeoPlot(testing_gdf, dpi=50, silent=True)
        minx, miny, maxx, maxy = testing_gdf.total_bounds
        width, height = maxx - minx, maxy - miny
        plot.focus_axes(pad=(0.3, 0.3, 0.01, 0.01))
        assert plot._xlim == pytest.approx((minx - 0.01 * width, maxx + 0.3 * width))
        assert plot._ylim == pytest.approx((miny - 0.01 * height, maxy + 0.3 * height))

    def test_bad_tuple_length_raises(self, testing_gdf):
        plot = GeoPlot(testing_gdf, dpi=50, silent=True)
        bad_pad: object = (0.1, 0.2, 0.3)
        with pytest.raises(ValueError, match="2 or 4"):
            plot.focus_axes(pad=cast("float", bad_pad))


class TestSaveColorbarOptions:
    def test_options_label_and_tight_crop(self, testing_gdf, tmp_path):
        from gerrytools.plotting import ColorbarOptions

        plot = GeoPlot(testing_gdf, dpi=50, silent=True)
        layer = plot.add_choropleth_layer("tot_pop")
        out = tmp_path / "bar.png"
        plot.save_colorbar(
            layer,
            str(out),
            label="",
            options=ColorbarOptions(
                force_ticks=[0, 50, 100],
                force_ticklabels=["0%", "50%", "100%"],
                shrink=0.8,
                aspect=15,
            ),
        )
        assert out.exists()

    def test_label_defaults_to_column(self, testing_gdf, tmp_path):
        plot = GeoPlot(testing_gdf, dpi=50, silent=True)
        layer = plot.add_choropleth_layer("tot_pop")
        out = tmp_path / "bar2.png"
        plot.save_colorbar(layer, str(out))
        assert out.exists()


class TestInFigureHorizontalColorbar:
    def test_horizontal_colorbar_sits_below_map(self, testing_gdf):
        from gerrytools.plotting import ColorbarOptions

        plot = GeoPlot(testing_gdf, dpi=50, silent=True)
        plot.add_choropleth_layer(
            "tot_pop",
            show_colorbar=True,
            colorbar_options=ColorbarOptions(orientation="horizontal"),
        )
        plot.ax.figure.canvas.draw()
        bar_pos = plot._colorbar_axes[0].get_position()
        main_pos = plot._ax.get_position()
        assert bar_pos.width > bar_pos.height
        assert bar_pos.y1 <= main_pos.y0

    def test_two_horizontal_colorbars_stack_downward(self, testing_gdf):
        from gerrytools.plotting import ColorbarOptions

        plot = GeoPlot(testing_gdf, dpi=50, silent=True)
        options = ColorbarOptions(orientation="horizontal")
        plot.add_choropleth_layer("tot_pop", show_colorbar=True, colorbar_options=options)
        plot.add_choropleth_layer("min_pop", show_colorbar=True, colorbar_options=options)
        plot.ax.figure.canvas.draw()
        first_pos, second_pos = (cax.get_position() for cax in plot._colorbar_axes)
        assert second_pos.y1 < first_pos.y0


class TestColorbarRebind:
    """Regression: the first rebuild after ``bind_to_ax`` used to call ``cax.remove()`` on
    the OLD figure's colorbar axes, destroying content the rebind contract leaves alone."""

    def _plot_with_colorbar_on(self, testing_gdf, ax):
        plot = GeoPlot(testing_gdf, silent=True)
        plot.add_choropleth_layer("tot_pop", show_colorbar=True)
        plot.bind_to_ax(ax)
        return plot

    def test_rebind_preserves_old_figure_colorbar(self, testing_gdf):
        fig_a, ax_a = plt.subplots()
        plot = self._plot_with_colorbar_on(testing_gdf, ax_a)
        assert len(fig_a.axes) == 2

        _, ax_b = plt.subplots()
        plot.bind_to_ax(ax_b)
        plot.ax  # rebuild on the new figure
        assert len(fig_a.axes) == 2

    def test_rebuild_after_rebind_creates_colorbar_on_new_figure(self, testing_gdf):
        fig_a, ax_a = plt.subplots()
        plot = self._plot_with_colorbar_on(testing_gdf, ax_a)

        fig_b, ax_b = plt.subplots()
        plot.bind_to_ax(ax_b)
        plot.ax
        assert len(fig_b.axes) == 2
        assert all(cax.figure is fig_b for cax in plot._colorbar_axes)

        # A further rebuild replaces only the new figure's colorbar axes.
        plot.focus_axes()
        plot.ax
        assert len(fig_b.axes) == 2
        assert len(fig_a.axes) == 2

    def test_rebind_to_same_axes_replaces_colorbar(self, testing_gdf):
        fig, ax = plt.subplots()
        plot = self._plot_with_colorbar_on(testing_gdf, ax)

        plot.bind_to_ax(ax)
        plot.ax

        assert len(fig.axes) == 2
        assert len(plot._colorbar_axes) == 1

    def test_rebind_back_to_previous_axes_replaces_colorbar(self, testing_gdf):
        fig_a, ax_a = plt.subplots()
        plot = self._plot_with_colorbar_on(testing_gdf, ax_a)
        fig_b, ax_b = plt.subplots()

        for target, figure in ((ax_b, fig_b), (ax_a, fig_a), (ax_b, fig_b), (ax_a, fig_a)):
            plot.bind_to_ax(target)
            assert len(figure.axes) == 2
            assert len(plot._colorbar_axes) == 1

    def test_rebind_rebuilds_caller_removed_inactive_colorbar(self, testing_gdf):
        fig_a, ax_a = plt.subplots()
        plot = self._plot_with_colorbar_on(testing_gdf, ax_a)
        old_colorbar = fig_a.axes[1]
        _, ax_b = plt.subplots()
        plot.bind_to_ax(ax_b)

        old_colorbar.remove()
        plot.bind_to_ax(ax_a)

        assert len(fig_a.axes) == 2
        assert len(plot._colorbar_axes) == 1


class TestDuplicateIndexWithGeometryMask:
    """Regression: with a ``geometry_mask``, ``color_series`` reindexed by label and raised
    'cannot reindex on an axis with duplicate labels' on concat-style GeoDataFrames."""

    def test_masked_outline_layer_with_duplicate_index_builds(self, testing_gdf, tmp_path):
        import geopandas as gpd

        doubled = gpd.GeoDataFrame(pd.concat([testing_gdf, testing_gdf]))
        assert doubled.index.has_duplicates
        mask = doubled["district"] == 0
        plot = GeoPlot(doubled, dpi=50, silent=True)
        plot.add_outline_layer(geometry_mask=mask)
        plot.save(str(tmp_path / "dup_masked.png"))
        assert (tmp_path / "dup_masked.png").exists()

    def test_masked_categorical_mapping_colors_by_row(self):
        import geopandas as gpd
        from shapely.geometry import box

        from gerrytools.plotting.geometry._layers import _CategoricalColorLayer

        gdf = gpd.GeoDataFrame(
            {
                "district": [1, 2, 1],
                "geometry": [box(i, 0, i + 1, 1) for i in range(3)],
            },
            index=pd.Index([0, 0, 1]),
        )
        mask = pd.Series([True, True, False], index=gdf.index)
        layer = _CategoricalColorLayer(
            geometry_source=gdf,
            column="district",
            colormap={1: "#ff0000", 2: "#00ff00"},
            geometry_mask=mask,
        )
        colors = layer.color_series
        assert len(colors) == 2
        assert colors.iloc[0][0] == "#ff0000"
        assert colors.iloc[1][0] == "#00ff00"


class TestInFigureColorbarShrink:
    def test_shrink_reduces_colorbar_height(self, testing_gdf):
        from gerrytools.plotting import ColorbarOptions

        plot = GeoPlot(testing_gdf, dpi=50, silent=True)
        plot.add_choropleth_layer(
            "tot_pop",
            show_colorbar=True,
            colorbar_options=ColorbarOptions(shrink=0.5),
        )
        plot.ax.figure.canvas.draw()
        map_height = plot._ax.get_position().height
        bar_height = plot._colorbar_axes[0].get_position().height
        assert bar_height == pytest.approx(map_height * 0.5, rel=0.05)
