import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest
from geopandas import GeoSeries
from matplotlib.colors import ListedColormap, to_hex
from shapely.geometry import Point

from gerrytools.colors import resolve_color_and_alpha
from gerrytools.plotting.geometry.geoplot import GeoPlot
from gerrytools.plotting.geometry.geoplotbase import (
    _CategoricalColorLayer,
    _MarkerLayer,
)
from gerrytools.plotting.mpl.label_text_options import LabelFontOptions
from tests.plotting._typing_utils import as_any


# ======================
# == CRS REPROJECTION ==
# ======================
class TestGeometriesInCRSReproject:
    """CRS reprojection when source CRS != target CRS."""

    def test_reproject_different_crs(self, testing_gdf):
        """The default outline keeps its source CRS and is reprojected with other layers."""
        gdf_crs = testing_gdf.copy().set_crs("EPSG:4326")
        plot = GeoPlot(gdf_crs, dpi=50, silent=True, target_crs="EPSG:3857")
        ax = plot.ax

        outline_bounds = ax.collections[0].get_datalim(ax.transData).bounds
        minx, miny, maxx, maxy = gdf_crs.to_crs("EPSG:3857").total_bounds

        np.testing.assert_allclose(
            outline_bounds,
            (minx, miny, maxx - minx, maxy - miny),
        )

    def test_crsless_overlay_rejected_when_plot_crs_is_known(self, testing_gdf):
        base = testing_gdf.copy().set_crs("EPSG:3857")
        overlay = GeoSeries(testing_gdf.geometry.values)
        plot = GeoPlot(base, dpi=50, silent=True, default_outline=False)
        plot.add_outline_layer(geo_source=overlay)

        with pytest.raises(ValueError, match="CRS-less geometries"):
            plot.ax


# ============================
# == CATEGORICAL LAYER INIT ==
# ============================


class TestCategoricalColorLayerGeoSeries:
    """GeoSeries geo_source with districtr colormap is silently set to 'none'."""

    def test_geoseries_districtr_maps_to_none(self, testing_gdf):
        gs = testing_gdf.geometry
        layer = _CategoricalColorLayer(
            geometry_source=gs,
            colormap="districtr",
            missing_color="none",
            facealpha=0.0,
            edgecolor="black",
        )
        # After __post_init__, colormap should have been changed to "none"
        assert layer.colormap == "none"


# =============================
# == DATACOLUMN REQUIREMENTS ==
# =============================


class TestCategoricalColorLayerNeedsDatacolumn:
    def test_mpl_colormap_without_datacolumn_raises(self, testing_gdf):
        with pytest.raises(TypeError, match="column.*must be set"):
            _CategoricalColorLayer(
                geometry_source=testing_gdf,
                colormap="viridis",  # valid mpl colormap
                column=None,
                missing_color="lightgrey",
                facealpha=0.0,
                edgecolor="none",
            )

    def test_dict_colormap_without_datacolumn_raises(self, testing_gdf):
        with pytest.raises(TypeError, match="column.*must be set"):
            _CategoricalColorLayer(
                geometry_source=testing_gdf,
                colormap={"A": "red"},
                column=None,
                missing_color="lightgrey",
                facealpha=0.0,
                edgecolor="none",
            )


# ============================
# == UNIQUE VALUE COLOR MAP ==
# ============================


class TestMapUniqueValuesStringSort:
    def test_string_keys_sorted_by_string(self, testing_gdf):
        """Non-integer category keys fall through to string-sort path."""
        gdf = testing_gdf.copy()
        gdf["str_col"] = ["alpha", "beta", "gamma", "delta"] * (len(testing_gdf) // 4) + [
            "alpha"
        ] * (len(testing_gdf) % 4)
        layer = _CategoricalColorLayer(
            geometry_source=gdf,
            column="str_col",
            colormap="districtr",
            missing_color="lightgrey",
            facealpha=None,
            edgecolor="none",
        )
        # Should have a dict colormap from string sort
        assert isinstance(layer.colormap, dict)

    def test_integral_float_keys_sort_numerically(self):
        colors = ["one", "two", "three", "ten", "twenty"]

        result = _CategoricalColorLayer._map_unique_values_to_colors(
            [20.0, 3.0, 10.0, 2.0, 1.0], colors
        )

        assert list(result) == [1.0, 2.0, 3.0, 10.0, 20.0]


# ==================
# == COLOR SERIES ==
# ==================


class TestCategoricalColorSeriesBranches:
    def test_colormap_object(self, testing_gdf):
        """Colormap instance colormap branch is covered."""
        import matplotlib.pyplot as plt

        cmap = plt.get_cmap("viridis")
        layer = _CategoricalColorLayer(
            geometry_source=testing_gdf,
            column="district",
            colormap=cmap,
            missing_color="lightgrey",
            facealpha=None,
            edgecolor="none",
        )
        cs = layer.color_series
        assert len(cs) == len(testing_gdf)

    def test_named_mpl_colormap_string(self, testing_gdf):
        """String colormap that is in plt.colormaps() uses Colormap path."""
        layer = _CategoricalColorLayer(
            geometry_source=testing_gdf,
            column="district",
            colormap="viridis",
            missing_color="lightgrey",
            facealpha=None,
            edgecolor="none",
        )
        cs = layer.color_series
        assert len(cs) == len(testing_gdf)

    @pytest.mark.parametrize(
        "colormap",
        ["viridis", ListedColormap(["red", "orange", "yellow", "green", "blue"])],
    )
    def test_colormap_samples_span_the_full_range(self, testing_gdf, colormap):
        gdf = testing_gdf.iloc[:5].copy()
        gdf["category"] = range(5)
        layer = _CategoricalColorLayer(
            geometry_source=gdf,
            column="category",
            colormap=colormap,
            missing_color="lightgrey",
            edgecolor="none",
        )

        colors = layer.color_series
        cmap = matplotlib.colormaps.get_cmap(colormap)

        assert colors.nunique() == 5
        assert colors.iloc[0][0] == to_hex(cmap(0.0))
        assert colors.iloc[-1][0] == to_hex(cmap(1.0))

    def test_colormap_rejects_more_categories_than_lut_colors(self, testing_gdf):
        gdf = testing_gdf.iloc[:3].copy()
        gdf["category"] = range(3)

        with pytest.raises(ValueError, match="Not enough colors"):
            _CategoricalColorLayer(
                geometry_source=gdf,
                column="category",
                colormap=ListedColormap(["red", "blue"]),
                missing_color="lightgrey",
                edgecolor="none",
            )

    def test_color_series_invalid_colormap_raises(self, testing_gdf):
        """Invalid colormap types fail while the input union is normalized."""
        gdf = testing_gdf.copy()
        with pytest.raises(TypeError, match="colormap.*must be one of"):
            _CategoricalColorLayer(
                geometry_source=gdf,
                column=None,
                colormap=as_any(12345),
                missing_color="lightgrey",
                facealpha=None,
                edgecolor="none",
            )

    def test_color_series_with_nan_in_column(self, testing_gdf):
        """NaN in column uses missing_color."""
        gdf = testing_gdf.copy()
        # Set some values to NaN for a Colormap path
        gdf["district_with_nan"] = gdf["district"].astype(float)
        gdf.loc[gdf.index[0], "district_with_nan"] = np.nan
        import matplotlib.pyplot as plt

        cmap = plt.get_cmap("viridis")
        layer = _CategoricalColorLayer(
            geometry_source=gdf,
            column="district_with_nan",
            colormap=cmap,
            missing_color="lightgrey",
            facealpha=None,
            edgecolor="none",
        )
        cs = layer.color_series
        assert len(cs) == len(gdf)
        assert cs.iloc[0] == resolve_color_and_alpha(layer.missing_color)
        assert cs.iloc[1] != cs.iloc[0]


# ===================
# == RENDER KWARGS ==
# ===================


class TestCategoricalRenderUnknownKwargs:
    def test_render_unknown_kwargs_raises(self, testing_gdf):
        import matplotlib.pyplot as plt

        layer = _CategoricalColorLayer(
            geometry_source=testing_gdf,
            colormap="none",
            missing_color="none",
            facealpha=0.0,
            edgecolor="black",
        )
        fig, ax = plt.subplots()
        with pytest.raises(TypeError, match="unexpected keyword argument"):
            layer.render(ax, **as_any({"bad_kwarg": "oops"}))
        plt.close(fig)


# =========================
# == RENDER MISSING DATA ==
# =========================


class TestCategoricalRenderMissingColumn:
    def test_render_missing_datacolumn_raises(self, testing_gdf):
        import matplotlib.pyplot as plt

        layer = _CategoricalColorLayer(
            geometry_source=testing_gdf,
            column="nonexistent_col",
            colormap={"A": "red"},
            missing_color="lightgrey",
            facealpha=None,
            edgecolor="none",
        )
        fig, ax = plt.subplots()
        with pytest.raises(KeyError):
            layer.render(ax)
        plt.close(fig)


# =======================
# == MARKER LAYER INIT ==
# =======================


class TestMarkerLayerPostInit:
    def test_none_point_geometries_raises(self):
        with pytest.raises(TypeError, match="point_geometries"):
            _MarkerLayer(point_geometries=as_any(None))

    def test_labels_wrong_length_raises(self, testing_gdf):
        pts = GeoSeries([Point(0, 0), Point(1, 1)])
        with pytest.raises(ValueError, match="same length"):
            _MarkerLayer(
                point_geometries=pts,
                labels=["only_one_label"],
            )

    def test_omitted_marker_options_uses_default(self, testing_gdf):
        pts = GeoSeries([Point(0, 0)])
        layer = _MarkerLayer(point_geometries=pts)
        assert layer.marker_options is not None


# =========================
# == MARKER LAYER RENDER ==
# =========================


class TestMarkerLayerRender:
    def test_render_unknown_kwargs_raises(self):
        import matplotlib.pyplot as plt

        pts = GeoSeries([Point(0, 0)])
        layer = _MarkerLayer(point_geometries=pts)
        fig, ax = plt.subplots()
        with pytest.raises(TypeError, match="unexpected keyword argument"):
            layer.render(ax, **as_any({"bad_kwarg": "oops"}))
        plt.close(fig)

    def test_render_with_crs_reprojection(self):
        """Points with CRS != target_crs triggers to_crs."""
        import matplotlib.pyplot as plt

        pts = GeoSeries([Point(-90, 40), Point(-91, 41)], crs="EPSG:4326")
        layer = _MarkerLayer(point_geometries=pts, show_labels=False)
        fig, ax = plt.subplots()
        [marker_line] = layer.render(ax, target_crs="EPSG:3857")
        expected = pts.to_crs("EPSG:3857")

        np.testing.assert_allclose(marker_line.get_xdata(), expected.x)
        np.testing.assert_allclose(marker_line.get_ydata(), expected.y)
        plt.close(fig)

    def test_render_show_labels_false(self):
        """show_labels=False skips the label-drawing branch."""
        import matplotlib.pyplot as plt

        pts = GeoSeries([Point(0, 0), Point(1, 1)])
        layer = _MarkerLayer(
            point_geometries=pts,
            show_labels=False,
            labels=None,
        )
        fig, ax = plt.subplots()
        artists = layer.render(ax)
        [marker_line] = artists

        np.testing.assert_allclose(marker_line.get_xdata(), [0, 1])
        np.testing.assert_allclose(marker_line.get_ydata(), [0, 1])
        assert len(ax.texts) == 0
        plt.close(fig)

    def test_render_show_labels_true_no_labels(self):
        """show_labels=True but labels=None still goes to the no-label branch."""
        import matplotlib.pyplot as plt

        pts = GeoSeries([Point(0, 0)])
        layer = _MarkerLayer(
            point_geometries=pts,
            show_labels=True,
            labels=None,
        )
        fig, ax = plt.subplots()
        artists = layer.render(ax)
        [marker_line] = artists

        np.testing.assert_allclose(marker_line.get_xdata(), [0])
        np.testing.assert_allclose(marker_line.get_ydata(), [0])
        assert len(ax.texts) == 0
        plt.close(fig)

    def test_label_outline_preserves_embedded_alpha(self):
        import matplotlib.pyplot as plt

        layer = _MarkerLayer(
            point_geometries=GeoSeries([Point(0, 0)]),
            labels=["A"],
            label_font_options=LabelFontOptions(outlinecolor="#ff000040"),
        )
        fig, ax = plt.subplots()

        layer.render(ax)

        stroke = ax.texts[0].get_path_effects()[0]
        assert getattr(stroke, "_gc")["foreground"] == pytest.approx(
            (1.0, 0.0, 0.0, 0.25), abs=1 / 255
        )
        plt.close(fig)
