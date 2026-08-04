import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

from gerrytools.colors import resolve_color_and_alpha
from gerrytools.plotting.geometry._labels import LabelOptions
from gerrytools.plotting.geometry._layers import _ContinuousColorLayer
from gerrytools.plotting.geometry.geoplot import GeoPlot
from gerrytools.plotting.mpl.geoplot_options import ColorbarOptions
from tests.plotting._typing_utils import as_any


# ===========================
# == CONTINUOUS LAYER INIT ==
# ===========================
class TestContinuousColorLayerPostInit:
    def test_invalid_colormap_type_raises(self, testing_gdf):
        """Non-str, non-Colormap colormap raises TypeError."""
        with pytest.raises(TypeError, match="colormap.*must be a str or Colormap"):
            _ContinuousColorLayer(
                geometry_source=testing_gdf,
                column="tot_pop",
                colormap=as_any({"A": "red"}),
            )

    def test_missing_datacolumn_raises(self, testing_gdf):
        """Missing column raises TypeError."""
        with pytest.raises(TypeError, match="column.*must be set"):
            _ContinuousColorLayer(
                geometry_source=testing_gdf,
                column=None,
                colormap="viridis",
            )


# ====================
# == BIN BOUNDARIES ==
# ====================


class TestBinBoundariesError:
    def test_bin_boundaries_none_bins_raises(self, testing_gdf):
        """Calling _bin_boundaries with bins=None raises RuntimeError."""
        layer = _ContinuousColorLayer(
            geometry_source=testing_gdf,
            column="tot_pop",
            bins=None,
        )
        with pytest.raises(RuntimeError, match="_bin_boundaries"):
            layer._bin_boundaries(0.0, 1.0)


# =======================
# == BIN COLOR MAPPING ==
# =======================


class TestContinuousLayerColormapObject:
    def test_colormap_object_works(self, testing_gdf, tmp_path):
        """A Colormap object can be used instead of a string."""
        import matplotlib.pyplot as plt

        cmap = plt.get_cmap("viridis")
        plot = GeoPlot(testing_gdf, dpi=50, silent=True)
        plot.add_choropleth_layer(column="tot_pop", colormap=cmap)
        plot.save(str(tmp_path / "colormap_obj.png"))
        assert (tmp_path / "colormap_obj.png").exists()


class TestColorMappingForBinsAlpha:
    def test_bins_with_facealpha(self, testing_gdf, tmp_path):
        """bins + facealpha triggers _color_mapping_for_bins alpha path."""
        plot = GeoPlot(testing_gdf, dpi=50, silent=True)
        plot.add_choropleth_layer(
            column="tot_pop",
            bins=4,
            facealpha=0.6,
            show_colorbar=True,
        )
        plot.save(str(tmp_path / "bins_facealpha.png"))
        assert (tmp_path / "bins_facealpha.png").exists()

    def test_facealpha_no_bins_with_colorbar(self, testing_gdf, tmp_path):
        """facealpha without bins and with colorbar triggers _with_alpha in _mappable."""
        plot = GeoPlot(testing_gdf, dpi=50, silent=True)
        plot.add_choropleth_layer(
            column="tot_pop",
            facealpha=0.5,
            bins=None,
            show_colorbar=True,
        )
        plot.save(str(tmp_path / "facealpha_colorbar.png"))
        assert (tmp_path / "facealpha_colorbar.png").exists()


# =========================
# == BINNED COLOR SERIES ==
# =========================


class TestContinuousColorSeriesBinPaths:
    def test_nan_value_in_bins_uses_missing_color(self, testing_gdf, tmp_path):
        """NaN values with bins get missing_color."""
        gdf = testing_gdf.copy()
        gdf["pop_with_nan"] = gdf["tot_pop"].astype(float)
        gdf.loc[gdf.index[0], "pop_with_nan"] = np.nan
        plot = GeoPlot(gdf, dpi=50, silent=True)
        plot.add_choropleth_layer(column="pop_with_nan", bins=4)
        plot.save(str(tmp_path / "bins_nan.png"))
        assert (tmp_path / "bins_nan.png").exists()

    def test_value_below_lower_bound_gets_first_bin(self, testing_gdf):
        """Value below the bin lower bound gets index 0."""
        gdf = testing_gdf.copy()
        min_val = gdf["tot_pop"].min()
        # Set vmin higher so values are below the first bin
        layer = _ContinuousColorLayer(
            geometry_source=gdf,
            column="tot_pop",
            bins=[min_val + 10000, min_val + 20000, min_val + 30000],
        )
        cs = layer.color_series
        boundaries = layer._bin_boundaries(*layer._effective_bounds(layer._data_series()))
        _, colors = layer._color_mapping_for_bins(boundaries)
        assert set(cs) == {resolve_color_and_alpha(colors[0])}

    def test_value_above_upper_bound_gets_last_bin(self, testing_gdf):
        """Value above the bin upper bound gets last bin index."""
        gdf = testing_gdf.copy()
        layer = _ContinuousColorLayer(
            geometry_source=gdf,
            column="tot_pop",
            bins=[0.0, 1.0, 2.0],  # All values above the last bin
        )
        cs = layer.color_series
        boundaries = layer._bin_boundaries(*layer._effective_bounds(layer._data_series()))
        _, colors = layer._color_mapping_for_bins(boundaries)
        assert set(cs) == {resolve_color_and_alpha(colors[-1])}


# ===================
# == RENDER ERRORS ==
# ===================


class TestContinuousLayerRender:
    def test_render_unknown_kwargs_raises(self, testing_gdf):
        """Unknown kwargs to render raises TypeError."""
        import matplotlib.pyplot as plt

        layer = _ContinuousColorLayer(
            geometry_source=testing_gdf,
            column="tot_pop",
        )
        fig, ax = plt.subplots()
        with pytest.raises(TypeError, match="unexpected keyword argument"):
            layer.render(ax, **as_any({"bad_kwarg": "oops"}))
        plt.close(fig)

    def test_render_missing_column_keyerror(self, testing_gdf):
        """Rendering with a column absent from geometry_source raises KeyError.

        _ContinuousColorLayer.__post_init__ only checks column is not None,
        not that the column actually exists in the GDF, so the failure surfaces
        as a KeyError at render time.
        """
        import matplotlib.pyplot as plt

        fake_layer = _ContinuousColorLayer(
            geometry_source=testing_gdf,
            column="__nonexistent_col__",
        )
        fig, ax = plt.subplots()
        with pytest.raises(KeyError):
            fake_layer.render(ax)
        plt.close(fig)


# ============================
# == DISTRICTING PLAN LAYER ==
# ============================


class TestAddDistrictingPlanLayerExtras:
    def test_add_plan_layer_with_geosource(self, testing_gdf, tmp_path):
        """add_districting_plan_layer with explicit geo_source uses that."""
        plot = GeoPlot(testing_gdf, dpi=50, silent=True)
        subset = testing_gdf[testing_gdf["district"].isin([0, 1])].copy()
        plot.add_districting_plan_layer(
            geo_source=subset,
            plan_column="district",
        )
        plot.save(str(tmp_path / "plan_geosource.png"))
        assert (tmp_path / "plan_geosource.png").exists()

    def test_add_plan_layer_show_labels_with_exclude(self, testing_gdf, tmp_path):
        """show_labels=True with exclude_labels excludes them."""
        plot = GeoPlot(testing_gdf, dpi=50, silent=True)
        plot.add_districting_plan_layer(
            plan_column="district",
            show_labels=True,
            label_options=LabelOptions(exclude=[0, 1]),
        )
        plot.save(str(tmp_path / "plan_exclude_labels.png"))
        assert (tmp_path / "plan_exclude_labels.png").exists()


# =====================
# == COLORBAR LAYOUT ==
# =====================


class TestClearColorbarsAndResetLayout:
    def test_clear_colorbars_called_on_rebuild(self, testing_gdf, tmp_path):
        """Building twice exercises cax.remove() in _clear_colorbars_and_reset_layout."""
        plot = GeoPlot(testing_gdf, dpi=50, silent=True)
        plot.add_choropleth_layer(column="tot_pop", show_colorbar=True)
        # First build creates the colorbar axes
        plot.save(str(tmp_path / "build1.png"))
        # Second build triggers _clear_colorbars_and_reset_layout with non-empty _colorbar_axes
        plot.save(str(tmp_path / "build2.png"))
        assert (tmp_path / "build2.png").exists()


class TestSetColorbarLayoutAllParams:
    def test_set_colorbar_layout_all_params(self, testing_gdf, tmp_path):
        """set_colorbar_layout with all params updates _colorbar_layout_options."""
        plot = GeoPlot(testing_gdf, dpi=50, silent=True)
        plot.add_choropleth_layer(column="tot_pop", show_colorbar=True)
        plot.set_colorbar_layout(
            outer_pad=0.01,
            inner_pad=0.01,
            width=0.02,
            right_margin=0.01,
        )
        plot.save(str(tmp_path / "colorbar_all_params.png"))
        assert (tmp_path / "colorbar_all_params.png").exists()


# ======================
# == COLORBAR OPTIONS ==
# ======================


class TestColorbarOptions:
    def test_force_ticklabels_require_force_ticks(self):
        with pytest.raises(ValueError, match="requires force_ticks"):
            ColorbarOptions(force_ticklabels=["low", "high"])

    def test_force_ticks_and_labels_must_have_matching_lengths(self):
        with pytest.raises(ValueError, match="same length"):
            ColorbarOptions(force_ticks=[0, 1], force_ticklabels=["low"])

    def test_colorbar_with_custom_label(self, testing_gdf, tmp_path):
        """colorbar_label override is used instead of column."""
        plot = GeoPlot(testing_gdf, dpi=50, silent=True)
        plot.add_choropleth_layer(
            column="tot_pop",
            show_colorbar=True,
            colorbar_label="Population",
        )
        plot.save(str(tmp_path / "colorbar_custom_label.png"))
        assert (tmp_path / "colorbar_custom_label.png").exists()

    def test_colorbar_with_label_fontsize_options(self, testing_gdf, tmp_path):
        """ColorbarOptions with label_fontsize triggers extended set_label call."""
        cb_options = ColorbarOptions(label_fontsize=10, label_rotation=90, label_pad=5.0)
        plot = GeoPlot(testing_gdf, dpi=50, silent=True)
        plot.add_choropleth_layer(
            column="tot_pop",
            show_colorbar=True,
            colorbar_options=cb_options,
        )
        plot.save(str(tmp_path / "colorbar_label_opts.png"))
        assert (tmp_path / "colorbar_label_opts.png").exists()

    def test_colorbar_force_ticks(self, testing_gdf, tmp_path):
        """force_ticks is applied to colorbar."""
        cb_options = ColorbarOptions(force_ticks=[1000, 3000, 5000])
        plot = GeoPlot(testing_gdf, dpi=50, silent=True)
        plot.add_choropleth_layer(
            column="tot_pop",
            show_colorbar=True,
            colorbar_options=cb_options,
        )
        plot.save(str(tmp_path / "colorbar_force_ticks.png"))
        assert (tmp_path / "colorbar_force_ticks.png").exists()

    def test_colorbar_force_ticklabels(self, testing_gdf, tmp_path):
        """force_ticklabels is applied to colorbar."""
        cb_options = ColorbarOptions(
            force_ticks=[1000, 3000, 5000],
            force_ticklabels=["low", "mid", "high"],
        )
        plot = GeoPlot(testing_gdf, dpi=50, silent=True)
        plot.add_choropleth_layer(
            column="tot_pop",
            show_colorbar=True,
            colorbar_options=cb_options,
        )
        plot.save(str(tmp_path / "colorbar_force_ticklabels.png"))
        assert (tmp_path / "colorbar_force_ticklabels.png").exists()

    def test_colorbar_max_n_ticks(self, testing_gdf, tmp_path):
        """max_n_ticks reduces the number of displayed ticks."""
        cb_options = ColorbarOptions(max_n_ticks=3)
        plot = GeoPlot(testing_gdf, dpi=50, silent=True)
        plot.add_choropleth_layer(
            column="tot_pop",
            show_colorbar=True,
            colorbar_options=cb_options,
        )
        plot.save(str(tmp_path / "colorbar_max_ticks.png"))
        assert (tmp_path / "colorbar_max_ticks.png").exists()

    def test_max_n_ticks_never_exceeded(self, testing_gdf):
        """The kept tick count must never exceed max_n_ticks (floor-step regression)."""
        cb_options = ColorbarOptions(max_n_ticks=4)
        plot = GeoPlot(testing_gdf, dpi=50, silent=True)
        # bins=8 yields 9 edge ticks; the old floor step kept 5 of them.
        plot.add_choropleth_layer(
            column="tot_pop",
            bins=8,
            show_colorbar=True,
            colorbar_options=cb_options,
        )
        plot.ax.figure.canvas.draw()
        assert len(plot._colorbar_axes[0].get_yticks()) <= 4

    @pytest.mark.parametrize("max_n_ticks", [0, -1, 1.5, True])
    def test_max_n_ticks_must_be_a_positive_integer(self, max_n_ticks):
        with pytest.raises(ValueError, match="max_n_ticks must be a positive integer"):
            ColorbarOptions(max_n_ticks=as_any(max_n_ticks))

    def test_partial_label_options_preserve_vertical_rotation(self, testing_gdf):
        """Setting only label_fontsize must not reset the vertical label's rotation to 0."""
        cb_options = ColorbarOptions(label_fontsize=10)
        plot = GeoPlot(testing_gdf, dpi=50, silent=True)
        plot.add_choropleth_layer(
            column="tot_pop",
            show_colorbar=True,
            colorbar_options=cb_options,
        )
        plot.ax.figure.canvas.draw()
        colorbar_label = plot._colorbar_axes[0].yaxis.label
        assert colorbar_label.get_fontsize() == 10
        assert colorbar_label.get_rotation() == 90.0

    def test_colorbar_bins_with_ticks(self, testing_gdf, tmp_path):
        """Bins path with colorbar uses ticks from layer_defaults."""
        plot = GeoPlot(testing_gdf, dpi=50, silent=True)
        plot.add_choropleth_layer(
            column="tot_pop",
            bins=4,
            show_colorbar=True,
        )
        plot.save(str(tmp_path / "colorbar_bins_ticks.png"))
        assert (tmp_path / "colorbar_bins_ticks.png").exists()
