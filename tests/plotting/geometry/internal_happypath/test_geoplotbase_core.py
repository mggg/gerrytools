from pathlib import Path
from typing import Any, cast

import matplotlib
import pytest

matplotlib.use("Agg")


from gerrytools.plotting.geometry._labels import LabelOptions
from gerrytools.plotting.geometry._layers._base import _to_target_crs
from gerrytools.plotting.geometry.geoplot import GeoPlot


# ========================
# == CRS / REPROJECTION ==
# ========================
class TestGeoLayerCRSReprojection:
    """Cover _geometries_in_crs paths when CRS is missing or already matches."""

    def test_no_crs_returns_as_is(self, testing_gdf):
        """GDF with no CRS should not attempt reprojection."""
        assert testing_gdf.crs is None
        geometries = testing_gdf.geometry
        assert _to_target_crs(geometries, None) is geometries

    def test_matching_crs_returns_as_is(self, testing_gdf):
        """When source CRS == target CRS, layer is returned unchanged."""
        geometries = testing_gdf.geometry.set_crs("EPSG:4326")
        assert _to_target_crs(geometries, "EPSG:4326") is geometries

    def test_crs_bearing_layer_requires_a_plot_crs(self, testing_gdf):
        geometries = testing_gdf.geometry.set_crs("EPSG:4326")
        with pytest.raises(ValueError, match="CRS-bearing geometries"):
            _to_target_crs(geometries, None)


# ================================
# == GEOMETRY_MASK in _GeoLayer ==
# ================================


class TestGeoLayerGeometryMask:
    """Cover the geometry_mask filtering code path in _GeoLayer."""

    def test_geometry_mask_filters_rows(self, testing_gdf, tmp_path):
        """A boolean geometry_mask should restrict which geometries are rendered."""
        mask = testing_gdf["district"] == 0
        plot = GeoPlot(testing_gdf, dpi=50, silent=True)
        plot.add_outline_layer(geometry_mask=mask)
        plot.save(str(tmp_path / "masked.png"))
        assert (tmp_path / "masked.png").exists()

    def test_geometry_mask_applies_before_dissolve(self, testing_gdf):
        """The mask is aligned to input rows, so dissolving keeps only the masked district."""
        import warnings

        mask = testing_gdf["district"] == 3
        plot = GeoPlot(testing_gdf, dpi=50, silent=True, default_outline=False)
        plot.add_outline_layer(geometry_mask=mask, dissolve_column="district", show_labels=True)

        layer = plot._outline_layers[-1]
        assert list(layer.geometry_source["district"]) == [3]
        assert layer.geometry_mask is None
        assert list(plot._label_requests[-1].gdf["district"]) == [3]
        assert plot._label_requests[-1].dissolved

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            plot.ax  # builds without a pandas reindex warning


# ===============================================
# == _CategoricalColorLayer COLOR SERIES PATHS ==
# ===============================================


class TestGeoPlotXYLimits:
    """Cover set_xlim, set_ylim, set_xlim, set_ylim, and clear_limits."""

    def test_set_xlim_and_ylim(self, testing_gdf, tmp_path):
        plot = GeoPlot(testing_gdf, dpi=50, silent=True)
        plot.set_xlim(0, 12)
        plot.set_ylim(0, 12)
        assert plot._xlim == (0.0, 12.0)
        assert plot._ylim == (0.0, 12.0)
        plot.save(str(tmp_path / "limits.png"))

    def test_clear_limits(self, testing_gdf):
        plot = GeoPlot(testing_gdf, dpi=50, silent=True)
        plot.set_xlim(0, 12)
        plot.set_ylim(0, 12)
        plot.clear_limits()
        assert plot._xlim is None
        assert plot._ylim is None

    def test_clear_limits_restores_autoscaling_on_rebuild(self, testing_gdf):
        """set_xlim then clear_limits: the next build autoscales again."""
        reference = GeoPlot(testing_gdf, dpi=50, silent=True)
        auto_xlim = reference.ax.get_xlim()
        auto_ylim = reference.ax.get_ylim()

        plot = GeoPlot(testing_gdf, dpi=50, silent=True)
        plot.set_xlim(-100.0, 100.0)
        plot.set_ylim(-100.0, 100.0)
        assert plot.ax.get_xlim() == (-100.0, 100.0)

        plot.clear_limits()
        ax = plot.ax  # rebuild
        assert ax.get_autoscalex_on()
        assert ax.get_autoscaley_on()
        assert ax.get_xlim() == auto_xlim
        assert ax.get_ylim() == auto_ylim


# =========================
# == CHECK AXIS FOCUSING ==
# =========================


class TestGeoPlotFocusAxes:
    """Cover focus_axes method paths."""

    def test_focus_axes_default(self, testing_gdf, tmp_path):
        """focus_axes() with no args uses base gdf bounds."""
        plot = GeoPlot(testing_gdf, dpi=50, silent=True)
        plot.focus_axes()
        minx, miny, maxx, maxy = testing_gdf.total_bounds
        assert plot.ax.get_xlim() == pytest.approx(
            (minx - 0.02 * (maxx - minx), maxx + 0.02 * (maxx - minx))
        )
        assert plot.ax.get_ylim() == pytest.approx(
            (miny - 0.02 * (maxy - miny), maxy + 0.02 * (maxy - miny))
        )

    def test_focus_axes_with_geosource_and_pad(self, testing_gdf, tmp_path):
        """focus_axes(geo_source=subset, pad=0.1) should limit to subset bounds."""
        subset = testing_gdf[testing_gdf["district"] == 0]
        plot = GeoPlot(testing_gdf, dpi=50, silent=True)
        plot.focus_axes(geo_source=subset, pad=0.1)
        minx, miny, maxx, maxy = subset.total_bounds
        expected_xlim = (minx - 0.1 * (maxx - minx), maxx + 0.1 * (maxx - minx))
        expected_ylim = (miny - 0.1 * (maxy - miny), maxy + 0.1 * (maxy - miny))

        assert plot.ax.get_xlim() == pytest.approx(expected_xlim)
        assert plot.ax.get_ylim() == pytest.approx(expected_ylim)

    def test_changing_target_crs_discards_limits_from_the_old_projection(self):
        import geopandas as gpd
        from shapely.geometry import box

        gdf = gpd.GeoDataFrame(
            {"v": [1.0, 2.0, 3.0]},
            geometry=[box(index, 0, index + 1, 1) for index in range(3)],
            crs="EPSG:3857",
        )
        plot = GeoPlot(gdf, dpi=50, silent=True)
        plot.add_choropleth_layer("v")
        plot.focus_axes()
        assert plot.ax.get_xlim() == pytest.approx((-0.06, 3.06))

        with pytest.warns(UserWarning, match="previous CRS"):
            plot.target_crs = "EPSG:4326"

        assert plot._xlim is None and plot._ylim is None
        low, high = plot.ax.get_xlim()
        reprojected = gdf.to_crs("EPSG:4326").total_bounds
        assert low <= reprojected[0] and high >= reprojected[2]
        assert high < 1e-3  # framed on the degree-scale data, not the old metre extent

    def test_focus_axes_pad_data_mode(self, testing_gdf):
        """focus_axes with pad_mode='data' should work."""
        plot = GeoPlot(testing_gdf, dpi=50, silent=True)
        plot.focus_axes(pad=0.5, pad_mode="data")
        minx, miny, maxx, maxy = testing_gdf.total_bounds
        assert plot.ax.get_xlim() == pytest.approx((minx - 0.5, maxx + 0.5))
        assert plot.ax.get_ylim() == pytest.approx((miny - 0.5, maxy + 0.5))


# ======================================
# == VALIDATE _apply_limits EXECUTION ==
# ======================================


class TestGeoPlotApplyLimits:
    """Cover _apply_limits when xlim and ylim are set."""

    def test_apply_limits_on_build(self, testing_gdf, tmp_path):
        plot = GeoPlot(testing_gdf, dpi=50, silent=True)
        plot.set_xlim(0, 12)
        plot.set_ylim(0, 12)
        plot.save(str(tmp_path / "applied.png"))
        assert (tmp_path / "applied.png").exists()


# =========================
# == SAVE AND SHOW TESTS ==
# =========================


class TestGeoPlotSaveMethod:
    def test_save_creates_file(self, testing_gdf, tmp_path):
        plot = GeoPlot(testing_gdf, dpi=50, silent=True)
        out = str(tmp_path / "out.png")
        plot.save(out)
        assert Path(out).exists()


class TestGeoPlotShowMethod:
    def test_show_does_not_raise(self, testing_gdf, tmp_path, monkeypatch):
        """show() should not raise in non-GUI (Agg) backend."""
        # show() lives on the shared _AxesBackedPlot base, so patch its module's show_figure.
        import gerrytools.plotting._axes_backed as axes_backed_module

        saved = []

        def fake_show(fig, *, non_gui_filename, non_gui_prefix):
            out = tmp_path / non_gui_filename
            fig.savefig(str(out))
            saved.append(str(out))

        monkeypatch.setattr(axes_backed_module, "show_figure", fake_show)
        plot = GeoPlot(testing_gdf, dpi=50, silent=True)
        plot.show()
        assert saved
        assert Path(saved[0]).exists()


# ================================
# == RETRIEVING LABEL POSITIONS ==
# ================================


class TestGeoPlotGetLabelPositions:
    def test_get_label_positions_returns_dict(self, testing_gdf, tmp_path):
        """get_label_positions() should return a (crs_str, dict) tuple."""
        plot = GeoPlot(testing_gdf, dpi=50, silent=True)
        plot.add_outline_layer(dissolve_column="district", show_labels=True)
        crs_str, positions = plot.get_label_positions()
        assert isinstance(crs_str, str)
        assert isinstance(positions, dict)

    def test_as_lat_long_without_crs_raises_curated_error(self, testing_gdf):
        """A CRS-less plot cannot reproject to lat/long; the error names the problem
        instead of surfacing geopandas' raw 'Cannot transform naive geometries'."""
        assert testing_gdf.crs is None
        plot = GeoPlot(testing_gdf, dpi=50, silent=True)
        plot.add_outline_layer(dissolve_column="district", show_labels=True)
        with pytest.raises(ValueError, match="no CRS"):
            plot.get_label_positions(as_lat_long=True)

    def test_labeled_layer_added_after_build_invalidates_cache(self, testing_gdf):
        """A layer added post-build must not leave get_label_positions stale."""
        plot = GeoPlot(testing_gdf, dpi=50, silent=True)
        plot.ax  # first build caches (empty) label positions
        plot.add_outline_layer(dissolve_column="district", show_labels=True)
        _, positions = plot.get_label_positions()
        assert positions, "expected fresh label positions without a manual rebuild"


# ============
# == LABELS ==
# ============


class TestGeoPlotOutlineLayerLabels:
    def test_outline_layer_with_labels(self, testing_gdf, tmp_path):
        plot = GeoPlot(testing_gdf, dpi=50, silent=True)
        plot.add_outline_layer(dissolve_column="district", show_labels=True)
        plot.save(str(tmp_path / "labeled_outline.png"))
        assert (tmp_path / "labeled_outline.png").exists()

    def test_highlight_layer_with_labels(self, testing_gdf, tmp_path):
        subset = testing_gdf[testing_gdf["district"] == 0].copy()
        plot = GeoPlot(testing_gdf, dpi=50, silent=True)
        plot.add_highlight_layer(
            geo_source=subset,
            show_labels=True,
            label_column="district",
        )
        plot.save(str(tmp_path / "labeled_highlight.png"))
        assert (tmp_path / "labeled_highlight.png").exists()


class TestGeoPlotWithLabels:
    def test_build_with_label_request(self, testing_gdf, tmp_path):
        """Exercising the deferred-label draw path."""
        plot = GeoPlot(testing_gdf, dpi=50, silent=True)
        plot.add_outline_layer(dissolve_column="county", show_labels=True)
        plot.save(str(tmp_path / "county_labels.png"))
        assert (tmp_path / "county_labels.png").exists()


# ===============================
# == GeoPlot choropleth ==
# ===============================


class TestLabelStyles:
    def test_badge_style_equalizes_circle_pads(self, testing_gdf):
        import geopandas as gpd

        from gerrytools.plotting.geometry.geoplot import GeoPlot

        plot = GeoPlot(testing_gdf, silent=True)
        points = testing_gdf.geometry.representative_point().iloc[:2]
        plot.add_label_layer(
            points_geoseries=gpd.GeoSeries(points, crs=testing_gdf.crs),
            labels=["7", "14"],
            label_options=LabelOptions(label_style="badge"),
        )
        # The style rides on one marker layer and resolves the box per label at render.
        layer = plot._marker_layers[-1]
        assert layer.label_style is not None
        plot.ax.figure.canvas.draw()
        pads = {}
        for text_artist in plot.ax.texts:
            if text_artist.get_text() in {"7", "14"}:
                bbox_patch = text_artist.get_bbox_patch()
                assert bbox_patch is not None
                pads[text_artist.get_text()] = getattr(bbox_patch.get_boxstyle(), "pad")
        assert set(pads) == {"7", "14"}
        assert pads["7"] > pads["14"]

    def test_unknown_label_style_raises(self, testing_gdf):
        from gerrytools.plotting.geometry.geoplot import GeoPlot

        plot = GeoPlot(testing_gdf, silent=True)
        with pytest.raises(ValueError, match="Unknown label style"):
            plot.add_label_layer(
                latlon_list=[(33.0, -84.0)],
                labels=["1"],
                label_options=LabelOptions(label_style="nope"),
            )

    def test_label_style_conflicts_with_explicit_options(self, testing_gdf):
        from gerrytools.plotting.geometry.geoplot import GeoPlot
        from gerrytools.plotting.mpl.label_text_options import LabelFontOptions

        plot = GeoPlot(testing_gdf, silent=True)
        with pytest.raises(ValueError, match="not both"):
            plot.add_label_layer(
                latlon_list=[(33.0, -84.0)],
                labels=["1"],
                label_options=LabelOptions(label_style="badge", font_options=LabelFontOptions()),
            )


class TestAllLabelStyles:
    def test_every_registered_style_builds(self, testing_gdf):
        import geopandas as gpd

        from gerrytools.plotting import LABEL_STYLES
        from gerrytools.plotting.geometry.geoplot import GeoPlot

        points = gpd.GeoSeries(
            testing_gdf.geometry.representative_point().iloc[:2], crs=testing_gdf.crs
        )
        for label_style_name in LABEL_STYLES:
            plot = GeoPlot(testing_gdf, silent=True)
            plot.add_label_layer(
                points_geoseries=points,
                labels=["1", "12"],
                label_options=LabelOptions(label_style=label_style_name),
            )
            assert plot.ax is not None


class TestLabelStyleOnLayers:
    def test_plan_layer_accepts_label_style(self, testing_gdf):
        from gerrytools.plotting.geometry.geoplot import GeoPlot

        plot = GeoPlot(testing_gdf, silent=True)
        plot.add_districting_plan_layer(
            "district",
            dissolve=True,
            show_labels=True,
            label_options=LabelOptions(label_style="halo"),
        )
        assert plot._label_requests[-1].options.label_style is not None
        assert plot.ax is not None

    def test_outline_layer_accepts_label_style(self, testing_gdf):
        from gerrytools.plotting.geometry.geoplot import GeoPlot

        plot = GeoPlot(testing_gdf, silent=True)
        plot.add_outline_layer(
            geo_source=testing_gdf,
            dissolve_column="district",
            show_labels=True,
            label_options=LabelOptions(label_style="ink"),
        )
        assert plot._label_requests[-1].options.label_style is not None
        assert plot.ax is not None

    def test_marker_layer_accepts_label_style(self, testing_gdf):
        from gerrytools.plotting.geometry.geoplot import GeoPlot

        plot = GeoPlot(testing_gdf.copy().set_crs("EPSG:4326"), silent=True, target_crs="EPSG:4326")
        plot.add_marker_layer(
            latlon_list=[(33.0, -84.0)],
            input_crs="EPSG:4326",
            labels=["Atlanta"],
            label_options=LabelOptions(label_style="tag"),
        )
        assert plot._marker_layers[-1].label_style is not None
        assert plot.ax is not None

    def test_label_style_conflicts_on_every_method(self, testing_gdf):
        from gerrytools.plotting.mpl.label_text_options import LabelFontOptions

        font = LabelFontOptions()
        with pytest.raises(ValueError, match="not both"):
            LabelOptions(label_style="halo", font_options=font)

    def test_with_font_tweak_applies(self, testing_gdf):
        from gerrytools.plotting import LABEL_STYLES
        from gerrytools.plotting.geometry.geoplot import GeoPlot

        plot = GeoPlot(testing_gdf.copy().set_crs("EPSG:4326"), silent=True, target_crs="EPSG:4326")
        plot.add_marker_layer(
            latlon_list=[(33.0, -84.0)],
            input_crs="EPSG:4326",
            labels=["x"],
            label_options=LabelOptions(label_style=LABEL_STYLES["badge"].with_font(fontsize=12)),
        )
        applied_style = plot._marker_layers[-1].label_style
        assert applied_style is not None
        assert applied_style.font.fontsize == 12


class TestLabelAdjustmentsAndFontsize:
    def test_label_layer_adjustment_moves_text(self, testing_gdf):
        import geopandas as gpd

        plot = GeoPlot(testing_gdf, silent=True)
        point = testing_gdf.geometry.representative_point().iloc[[0]]
        plot.add_label_layer(
            points_geoseries=gpd.GeoSeries(point, crs=testing_gdf.crs),
            labels=["7"],
            label_options=LabelOptions(label_style="badge", adjustments={"7": (2.0, -1.5)}),
        )
        plot.ax.figure.canvas.draw()
        text = next(t for t in plot.ax.texts if t.get_text() == "7")
        x, y = text.get_position()
        assert x == pytest.approx(point.iloc[0].x + 2.0)
        assert y == pytest.approx(point.iloc[0].y - 1.5)

    def test_label_layer_fontsize_overrides(self, testing_gdf):
        import geopandas as gpd

        plot = GeoPlot(testing_gdf, silent=True)
        points = testing_gdf.geometry.representative_point().iloc[:2]
        plot.add_label_layer(
            points_geoseries=gpd.GeoSeries(points, crs=testing_gdf.crs),
            labels=["7", "14"],
            label_options=LabelOptions(label_style="badge", fontsize={7: 4}),
        )
        plot.ax.figure.canvas.draw()
        sizes = {t.get_text(): t.get_fontsize() for t in plot.ax.texts}
        assert sizes["7"] == 4
        assert sizes["14"] == 8  # the badge style default

    def test_plan_layer_label_adjustments_and_fontsize(self, testing_gdf):
        plot = GeoPlot(testing_gdf, silent=True)
        plot.add_districting_plan_layer(
            "district",
            dissolve=True,
            show_labels=True,
            label_options=LabelOptions(
                label_style="badge", adjustments={0: (3.0, 0.0)}, fontsize={1: 5}
            ),
        )
        plot.ax.figure.canvas.draw()
        sizes = {t.get_text(): t.get_fontsize() for t in plot.ax.texts}
        assert sizes["1"] == 5
        assert sizes["0"] == 8

    def test_scalar_fontsize_applies_to_all(self, testing_gdf):
        plot = GeoPlot(testing_gdf, silent=True)
        plot.add_outline_layer(
            dissolve_column="district",
            show_labels=True,
            label_options=LabelOptions(label_style="badge", fontsize=5),
        )
        plot.ax.figure.canvas.draw()
        sizes = {t.get_fontsize() for t in plot.ax.texts}
        assert sizes == {5}


class TestLabelExcludeOnMarkerPath:
    """Regression: ``LabelOptions.exclude`` used to be ignored by the marker path
    (``add_marker_layer`` / ``add_label_layer``); only dissolved layers honored it."""

    def test_label_layer_exclude_drops_labels_and_points(self, testing_gdf):
        import geopandas as gpd

        plot = GeoPlot(testing_gdf, silent=True)
        points = testing_gdf.geometry.representative_point().iloc[:3]
        plot.add_label_layer(
            points_geoseries=gpd.GeoSeries(points, crs=testing_gdf.crs),
            labels=["01", "2", "3"],
            # Int keys must match string/zero-padded labels (dtype-insensitive).
            label_options=LabelOptions(exclude=[1, "3"]),
        )
        layer = plot._marker_layers[-1]
        assert list(layer.labels or []) == ["2"]
        assert len(layer.point_geometries) == 1
        assert layer.point_geometries.iloc[0].equals(points.iloc[1])

    def test_integral_float_labels_match_integer_exclusions(self, testing_gdf):
        import geopandas as gpd

        plot = GeoPlot(testing_gdf, silent=True)
        points = testing_gdf.geometry.representative_point().iloc[:2]
        plot.add_label_layer(
            points_geoseries=gpd.GeoSeries(points, crs=testing_gdf.crs),
            labels=cast(Any, [1.0, 2.0]),
            label_options=LabelOptions(exclude=[2]),
        )

        assert list(plot._marker_layers[-1].labels or []) == [1.0]

    def test_marker_layer_exclude_drops_labels_and_points(self, testing_gdf):
        import geopandas as gpd

        plot = GeoPlot(testing_gdf, silent=True)
        points = testing_gdf.geometry.representative_point().iloc[:2]
        plot.add_marker_layer(
            points_geoseries=gpd.GeoSeries(points, crs=testing_gdf.crs),
            labels=["7", "8"],
            show_labels=True,
            label_options=LabelOptions(exclude=["07"]),
        )
        layer = plot._marker_layers[-1]
        assert list(layer.labels or []) == ["8"]
        assert len(layer.point_geometries) == 1

    def test_marker_layer_without_exclude_keeps_all(self, testing_gdf):
        import geopandas as gpd

        plot = GeoPlot(testing_gdf, silent=True)
        points = testing_gdf.geometry.representative_point().iloc[:2]
        plot.add_marker_layer(
            points_geoseries=gpd.GeoSeries(points, crs=testing_gdf.crs),
            labels=["7", "8"],
            show_labels=True,
        )
        layer = plot._marker_layers[-1]
        assert list(layer.labels or []) == ["7", "8"]
        assert len(layer.point_geometries) == 2

    def test_mismatched_labels_with_exclude_raises_clear_length_error(self, testing_gdf):
        """Length validation fires before the exclude mask, so a mismatched ``labels``
        raises the clear message instead of a pandas IndexError."""
        import geopandas as gpd

        plot = GeoPlot(testing_gdf, silent=True)
        points = testing_gdf.geometry.representative_point().iloc[:3]
        with pytest.raises(ValueError, match="same length"):
            plot.add_marker_layer(
                points_geoseries=gpd.GeoSeries(points, crs=testing_gdf.crs),
                labels=["1", "2"],
                show_labels=True,
                label_options=LabelOptions(exclude=["1"]),
            )


class TestLabelExcludeOnDissolvedPath:
    """Dissolved-path ``LabelOptions.exclude`` verified behaviorally, not just by a
    png existing: excluded labels are absent from the computed positions."""

    def test_dissolved_outline_exclude_drops_label_positions(self, testing_gdf):
        district_count = testing_gdf["district"].nunique()
        plot = GeoPlot(testing_gdf, dpi=50, silent=True)
        plot.add_outline_layer(
            dissolve_column="district",
            show_labels=True,
            label_options=LabelOptions(exclude=[0]),
        )
        _, positions = plot.get_label_positions()
        assert "0" not in positions
        assert len(positions) == district_count - 1


class TestTargetCrsSetter:
    def test_target_crs_setter_reprojects_layers_on_next_build(self, testing_gdf):
        """Setting ``target_crs`` post-construction defers a rebuild that reprojects
        the layers (previously only DotDensityPlot's override was covered)."""
        import numpy as np

        gdf = testing_gdf.copy().set_crs("EPSG:4326")
        plot = GeoPlot(gdf, dpi=50, silent=True)

        def rendered_x_max():
            vertex_arrays = [
                path.vertices
                for collection in plot.ax.collections
                for path in collection.get_paths()
            ]
            return float(np.concatenate(vertex_arrays)[:, 0].max())

        assert rendered_x_max() < 100.0  # degrees
        plot.target_crs = "EPSG:3857"
        assert plot._axis_needs_update
        assert plot.target_crs == "EPSG:3857"
        assert rendered_x_max() > 1e5  # meters: layers were reprojected on rebuild


class TestLabelStyleShorthand:
    """The top-level ``label_style=`` kwarg mirrors ``label_options.label_style``."""

    def _points(self, testing_gdf):
        import geopandas as gpd

        points = testing_gdf.geometry.representative_point().iloc[:2]
        return gpd.GeoSeries(points, crs=testing_gdf.crs)

    def test_label_style_kwarg_matches_label_options_form(self, testing_gdf):
        from gerrytools.plotting.mpl.label_text_options import resolve_label_style

        shorthand = GeoPlot(testing_gdf, silent=True)
        shorthand.add_label_layer(
            points_geoseries=self._points(testing_gdf),
            labels=["1", "2"],
            label_style="badge",
        )
        bundled = GeoPlot(testing_gdf, silent=True)
        bundled.add_label_layer(
            points_geoseries=self._points(testing_gdf),
            labels=["1", "2"],
            label_options=LabelOptions(label_style="badge"),
        )
        assert shorthand._marker_layers[-1].label_style == resolve_label_style("badge")
        assert shorthand._marker_layers[-1].label_style == bundled._marker_layers[-1].label_style

    def test_label_style_kwarg_merges_into_options_without_style(self, testing_gdf):
        plot = GeoPlot(testing_gdf, silent=True)
        plot.add_label_layer(
            points_geoseries=self._points(testing_gdf),
            labels=["01", "2"],
            label_style="halo",
            label_options=LabelOptions(exclude=[1]),
        )
        layer = plot._marker_layers[-1]
        assert list(layer.labels or []) == ["2"]
        assert layer.label_style is not None

    def test_label_style_kwarg_conflicts_with_bundled_label_style(self, testing_gdf):
        plot = GeoPlot(testing_gdf, silent=True)
        with pytest.raises(ValueError, match="not both"):
            plot.add_label_layer(
                points_geoseries=self._points(testing_gdf),
                labels=["1", "2"],
                label_style="badge",
                label_options=LabelOptions(label_style="halo"),
            )

    def test_districting_plan_layer_accepts_label_style_kwarg(self, testing_gdf):
        from gerrytools.plotting.mpl.label_text_options import resolve_label_style

        plot = GeoPlot(testing_gdf, silent=True)
        plot.add_districting_plan_layer(
            "district", dissolve=True, show_labels=True, label_style="badge"
        )
        request = plot._label_requests[-1]
        assert request.options.resolved_label_style == resolve_label_style("badge")
        assert request.dissolved
